"""
LLM-based extraction of structured injury impact from pre-game reports.

Uses Gemini's JSON mode to guarantee valid structured output.
Set GOOGLE_API_KEY in your environment before running.
"""

import json
import logging
import os
import time

import google.generativeai as genai

from src.utils.config_loader import load_config

logger = logging.getLogger(__name__)

_PROMPT_TEMPLATE = """You are an NBA analyst estimating the scoring impact of pre-game injuries.
{team_avg_line}
The importance score (0–1, team-relative) reflects each player's share of scoring
contribution within their team. Use it alongside PPG and USG% to scale each player's
impact continuously — higher importance means a larger negative impact when out.
Questionable players may still play — discount their impact by ~50%.

Team: {team_name}  |  Game date: {game_date}

Injury report:
{injury_text}

Player stats (importance 0–1 team-relative, PPG, USG%):
{importance_text}

If a player appears in the injury report but is not listed in the player stats,
estimate their impact using your best judgment of their role.

Return a JSON object with exactly these fields:
- impact_score: total estimated point impact summed across all injured players (float, negative or 0)
- n_out: count of players with status "Out" (int)
- n_questionable: count of players with status "Questionable" or "Day-To-Day" (int)"""

_MAX_RETRIES = 4
_RETRY_BASE_DELAY = 5  # seconds; doubles each attempt

_cfg = load_config()
genai.configure(api_key=os.environ.get("GOOGLE_API_KEY", ""))
_model = genai.GenerativeModel(
    model_name=_cfg.injury_features.llm_model,
    generation_config=genai.GenerationConfig(response_mime_type="application/json"),
)


def _validate(raw: dict) -> dict:
    required = {"impact_score", "n_out", "n_questionable"}
    missing = required - raw.keys()
    if missing:
        raise ValueError(f"LLM response missing fields: {missing}  raw={raw}")
    return {
        "impact_score": float(raw["impact_score"]),
        "n_out": int(raw["n_out"]),
        "n_questionable": int(raw["n_questionable"]),
    }


def extract_impact(
    team_name: str,
    game_date: str,
    injury_list: list[dict],
    importance_map: dict[str, float],
    player_stats: dict[str, dict],
    team_avg: float | None,
) -> dict:
    """
    Call Gemini to extract structured impact from a team's pre-game injury report.

    Args:
        injury_list: [{"player_name": str, "status": str, "reason": str}]
        importance_map: {player_name: importance_score (0-1)}
        player_stats: {player_name: {"ppg": float, "usg": float}}
        team_avg: team's rolling average score before this game, or None if unavailable

    Returns:
        {"impact_score": float, "n_out": int, "n_questionable": int}
    """
    if not injury_list:
        return {"impact_score": 0.0, "n_out": 0, "n_questionable": 0}

    injury_text = "\n".join(
        f"- {p['player_name']} ({p['status']}): {p.get('reason', '')}" for p in injury_list
    )

    lines = []
    for name, score in importance_map.items():
        stats = player_stats.get(name, {})
        ppg = stats.get("ppg")
        usg = stats.get("usg")
        parts = [f"importance={score:.2f}"]
        if ppg is not None:
            parts.append(f"{ppg:.1f} PPG")
        if usg is not None:
            parts.append(f"{usg * 100:.1f}% USG")
        lines.append(f"- {name}: {', '.join(parts)}")
    importance_text = "\n".join(lines) or "No player data — use best judgment."

    window = _cfg.features.rolling_window
    team_avg_line = (
        f"This team averages {team_avg:.1f} PPG over their last {window} games."
        if team_avg is not None
        else "Team scoring average unavailable."
    )

    prompt = _PROMPT_TEMPLATE.format(
        team_avg_line=team_avg_line,
        team_name=team_name,
        game_date=game_date,
        injury_text=injury_text,
        importance_text=importance_text,
    )

    for attempt in range(_MAX_RETRIES):
        try:
            response = _model.generate_content(prompt)
            raw = json.loads(response.text)
            result = _validate(raw)
            logger.debug(f"Impact for {team_name} on {game_date}: {result}")
            delay = 60.0 / _cfg.injury_features.api_calls_per_minute
            time.sleep(delay)
            return result
        except Exception as e:
            wait = _RETRY_BASE_DELAY * (2 ** attempt)
            if attempt < _MAX_RETRIES - 1:
                logger.warning(f"Gemini error for {team_name} on {game_date} (attempt {attempt + 1}/{_MAX_RETRIES}), retrying in {wait}s: {e}")
                time.sleep(wait)
            else:
                logger.error(f"Gemini failed after {_MAX_RETRIES} attempts for {team_name} on {game_date}: {e}")
                raise
