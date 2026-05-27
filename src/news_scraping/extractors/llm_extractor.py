"""
LLM-based extraction of structured injury impact from pre-game reports.

Uses Gemini's JSON mode to guarantee valid structured output.
Free tier: 1,500 requests/day at aistudio.google.com — enough for backfill and nightly use.
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

Return a JSON object with exactly these fields:
- impact_score (float, negative or 0)
- n_out (int)
- n_questionable (int)
- star_out (bool)"""


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
        {"impact_score": float, "n_out": int, "n_questionable": int, "star_out": bool}
    """
    if not injury_list:
        return {"impact_score": 0.0, "n_out": 0, "n_questionable": 0, "star_out": False}

    cfg = load_config()
    genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

    model = genai.GenerativeModel(
        model_name=cfg.injury_features.llm_model,
        generation_config=genai.GenerationConfig(response_mime_type="application/json"),
    )

    injury_text = "\n".join(
        f"- {p['player_name']} ({p['status']}): {p.get('reason', '')}" for p in injury_list
    )
    importance_text = (
        "\n".join(f"- {name}: {score:.2f}" for name, score in importance_map.items())
        or "No importance data — use best judgment."
    )

    response = model.generate_content(
        _PROMPT_TEMPLATE.format(
            team_name=team_name,
            game_date=game_date,
            injury_text=injury_text,
            importance_text=importance_text,
        )
    )

    result = json.loads(response.text)
    logger.debug(f"Impact for {team_name} on {game_date}: {result}")

    # Respect free-tier RPM limit — sleep after every call
    delay = 60.0 / cfg.injury_features.api_calls_per_minute
    time.sleep(delay)

    return result
