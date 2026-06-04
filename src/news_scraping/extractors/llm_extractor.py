"""
LLM-based extraction of structured injury impact from pre-game reports.

Uses Gemini's JSON mode to guarantee valid structured output.
Set GOOGLE_API_KEY in your environment before running.
"""

import json
import logging
import os
import threading
import time

from google import genai
from google.genai import types

from src.utils.config_loader import load_config

logger = logging.getLogger(__name__)

_PROMPT_TEMPLATE = """You are an NBA analyst estimating the per-game scoring impact of pre-game injuries.
{team_avg_line}
The importance score (0–1, team-relative) reflects each player's share of scoring
contribution within their team. Use it alongside PPG and USG% to scale each player's
impact continuously — higher importance means a larger negative impact when out.
Questionable players may still play — discount their impact by ~50%.

impact_score is the NET single-game point loss — the injured player's typical contribution
minus what replacement-level play provides. When a player is Out, teammates absorb their
minutes and scoring load, so the net impact is substantially less than the missing player's
PPG. Use PPG and importance to rank each player's significance, not to sum their full output.

Team: {team_name}  |  Game date: {game_date}

Injury report:
{injury_text}

Player stats (importance 0–1 team-relative, PPG, USG%):
{importance_text}

If a player appears in the injury report but has no stats listed, treat their impact as 0.

Return a JSON object with exactly these fields:
- impact_score: total estimated point impact summed across all injured players (float, negative or 0)
- n_out: count of players with status "Out" (int)
- n_questionable: count of players with status "Questionable" or "Day-To-Day" (int)"""

_MAX_RETRIES = 4
_RETRY_BASE_DELAY = 5  # seconds; doubles each attempt

_cfg = load_config()
_client = genai.Client(api_key=os.environ.get("GOOGLE_API_KEY", ""))


class _RateLimiter:
    """Global rate limiter shared across all threads."""
    def __init__(self, calls_per_minute: int):
        self._interval = 60.0 / calls_per_minute
        self._lock = threading.Lock()
        self._last = 0.0

    def wait(self):
        with self._lock:
            now = time.monotonic()
            gap = self._interval - (now - self._last)
            if gap > 0:
                time.sleep(gap)
            self._last = time.monotonic()


_rate_limiter = _RateLimiter(_cfg.injury_features.api_calls_per_minute)


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


def _clip_impact(
    result: dict,
    injury_list: list[dict],
    player_stats: dict[str, dict],
    team_avg: float | None,
    team_name: str,
    game_date: str,
) -> dict:
    out_ppg = sum(
        (player_stats.get(p["player_name"], {}).get("ppg") or 0)
        for p in injury_list if p.get("status") == "Out"
    )
    q_ppg = sum(
        (player_stats.get(p["player_name"], {}).get("ppg") or 0)
        for p in injury_list if p.get("status") in ("Questionable", "Day-To-Day")
    ) * 0.5

    bounds = []
    if out_ppg + q_ppg > 0:
        bounds.append(out_ppg + q_ppg)
    if team_avg is not None:
        bounds.append(team_avg)

    if not bounds:
        return result

    max_abs = min(bounds)
    if result["impact_score"] < -max_abs:
        logger.warning(
            f"Clipping impact_score {result['impact_score']:.2f} → {-max_abs:.2f} "
            f"for {team_name} on {game_date} (bound: out_ppg={out_ppg:.1f}, q_ppg={q_ppg:.1f}, team_avg={team_avg})"
        )
        result = dict(result)
        result["impact_score"] = round(-max_abs, 2)
    return result


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
    importance_text = "\n".join(lines) or "No player data available — treat all impacts as 0."

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
            response = _client.models.generate_content(
                model=_cfg.injury_features.llm_model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    thinking_config=types.ThinkingConfig(thinking_budget=0),
                ),
            )
            raw = json.loads(response.text)
            result = _validate(raw)
            result = _clip_impact(result, injury_list, player_stats, team_avg, team_name, game_date)
            logger.debug(f"Impact for {team_name} on {game_date}: {result}")
            _rate_limiter.wait()
            return result
        except Exception as e:
            wait = _RETRY_BASE_DELAY * (2 ** attempt)
            if attempt < _MAX_RETRIES - 1:
                logger.warning(f"Gemini error for {team_name} on {game_date} (attempt {attempt + 1}/{_MAX_RETRIES}), retrying in {wait}s: {e}")
                time.sleep(wait)
            else:
                logger.error(f"Gemini failed after {_MAX_RETRIES} attempts for {team_name} on {game_date}: {e}")
                raise
