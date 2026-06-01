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
An average NBA team scores ~115 points. A star player out is typically -4 to -8 pts;
a rotation player -1 to -3; a bench player 0 to -1.
Questionable players may still play — discount their impact by ~50%.

Team: {team_name}  |  Game date: {game_date}

Injury report:
{injury_text}

Player importance scores (0–1, team-relative; >0.5 = star contributor):
{importance_text}

If a player appears in the injury report but is not listed in the importance scores,
estimate their impact using the heuristics above based on your best judgment of their role.

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
) -> dict:
    """
    Call Gemini to extract structured impact from a team's pre-game injury report.

    Args:
        injury_list: [{"player_name": str, "status": str, "reason": str}]
        importance_map: {player_name: importance_score}

    Returns:
        {"impact_score": float, "n_out": int, "n_questionable": int}
    """
    if not injury_list:
        return {"impact_score": 0.0, "n_out": 0, "n_questionable": 0}

    injury_text = "\n".join(
        f"- {p['player_name']} ({p['status']}): {p.get('reason', '')}" for p in injury_list
    )
    importance_text = (
        "\n".join(f"- {name}: {score:.2f}" for name, score in importance_map.items())
        or "No importance data — use best judgment."
    )
    prompt = _PROMPT_TEMPLATE.format(
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
