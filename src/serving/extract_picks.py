"""
Vision-based extraction: a bookmaker screenshot -> structured picks (teams
translated to canonical NBA nicknames, spread, odds), feeding
src/serving/recommend.py's recommend_game. Uses this project's existing LLM
pattern (google-genai / Gemini, same client library and GOOGLE_API_KEY env
var as src/availability/llm_estimator.py's GeminiClient) -- not a new
provider or dependency.

Screenshots vary by bookmaker (language, market structure -- e.g. a 2-way
spread vs. a 3-way spread-with-push market). This intentionally extracts a
loose, mostly-optional schema per game rather than assuming one fixed
layout, and pushes the "which fields does this specific bookmaker show"
question onto the model doing the reading rather than hand-coding a parser
per bookmaker.
"""

import json
import logging
import os

from src.serving.team_lookup import build_nickname_to_team_id

logger = logging.getLogger(__name__)

_MIME_TYPES = {"jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png", "webp": "image/webp"}

_SYSTEM_INSTRUCTIONS_TEMPLATE = """You read screenshots of basketball betting odds -- often in Hebrew, \
sometimes other languages -- and extract structured picks for NBA games only. Ignore any non-NBA \
games in the image (other leagues, exhibitions), and ignore any UI chrome that isn't itself an odds \
value -- promo/boost badges, bet-type labels (e.g. "SD"), or icon buttons (e.g. "recommendation", \
"source") that sit next to the odds table but aren't part of it.

For each NBA game visible, extract a JSON object with these fields. Every field except \
home_team_nickname/away_team_nickname is optional -- omit a field entirely if that market isn't \
shown, rather than guessing a value:

- home_team_nickname, away_team_nickname (required): the team's canonical NBA nickname, translated \
  from whatever language/script the image uses. Choose ONLY from this exact list, verbatim: \
  {nicknames}. If a team can't be confidently matched to one of these, drop that entire game rather \
  than guessing.
- home_spread, away_spread: the point spread shown next to each team, with its own printed sign \
  (negative = favored by that many points). Most bookmakers show a plain 2-way spread (often a \
  half-point line specifically to rule out a tie) -- don't assume a 3-way market exists.
- home_spread_odds, away_spread_odds: decimal odds for each side of the spread.
- push_odds: decimal odds for an exact-margin "push"/tie outcome, ONLY if the market shows one as a \
  separate bettable option distinct from either team (some bookmakers show a 3-way market: \
  team / exact-push / team, usually with a whole-number line instead of a half-point one).
- home_moneyline_odds, away_moneyline_odds: decimal odds to win outright, if a separate moneyline \
  market is shown.
- total_line: the over/under total points line, if shown.
- over_odds, under_odds: decimal odds for over/under, if shown.

Return ONLY a JSON array, one object per NBA game. If no NBA game is visible, return an empty array."""


def _gemini_client():
    from google import genai

    return genai.Client(api_key=os.environ.get("GOOGLE_API_KEY", ""))


def extract_picks_from_screenshot(
    image_path: str, model: str = "gemini-2.5-flash", client=None
) -> list[dict]:
    """Reads image_path, returns a list of pick dicts (schema above, plus
    home_team_id/away_team_id resolved from the nickname). Every returned
    pick's team_ids are guaranteed to be real NBA team_ids -- anything the
    model couldn't confidently map to a canonical nickname is dropped, not
    guessed. Raises ValueError if the model's response isn't valid JSON.

    `client`: injectable for tests (anything exposing
    `.models.generate_content(...) -> object with .text`, the same surface
    `google.genai.Client` exposes) -- defaults to a real Gemini client, same
    pattern as src/availability/llm_estimator.py's GeminiClient being
    swappable for a fake."""
    from google.genai import types

    nick2id = build_nickname_to_team_id()
    nicknames = sorted(nick2id)

    ext = image_path.rsplit(".", 1)[-1].lower()
    mime_type = _MIME_TYPES.get(ext, "image/png")
    with open(image_path, "rb") as f:
        image_bytes = f.read()

    client = client or _gemini_client()
    response = client.models.generate_content(
        model=model,
        contents=[
            types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
            "Extract the NBA betting picks from this screenshot.",
        ],
        config=types.GenerateContentConfig(
            system_instruction=_SYSTEM_INSTRUCTIONS_TEMPLATE.format(nicknames=", ".join(nicknames)),
            response_mime_type="application/json",
            temperature=0.0,
        ),
    )

    try:
        raw_picks = json.loads(response.text)
    except (json.JSONDecodeError, TypeError) as e:
        raise ValueError(f"Model did not return valid JSON: {response.text!r}") from e

    picks = []
    for raw in raw_picks:
        home_nick = raw.get("home_team_nickname")
        away_nick = raw.get("away_team_nickname")
        if home_nick not in nick2id or away_nick not in nick2id:
            logger.warning(f"Dropping pick with unrecognized team(s): {raw}")
            continue
        pick = {
            "home_team_id": nick2id[home_nick],
            "away_team_id": nick2id[away_nick],
            **{k: v for k, v in raw.items() if k not in ("home_team_nickname", "away_team_nickname")},
        }
        picks.append(pick)
    return picks
