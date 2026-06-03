"""
Live NBA injury data from the public ESPN API.
Used for nightly runs. No auth required.

For historical backfill use nba_injury_pdf.py instead.
"""

import logging

import requests

logger = logging.getLogger(__name__)

_URL = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/injuries"
_HEADERS = {"User-Agent": "Mozilla/5.0"}

_TRACKED_STATUSES = {"Out", "Doubtful", "Questionable", "Day-To-Day"}


def fetch_current_injuries() -> list[dict]:
    """
    Fetch today's injury list from ESPN.
    Returns [{"team_abbreviation": str, "player_name": str, "status": str, "reason": str}].
    Filters to only statuses that meaningfully affect game-day availability.
    """
    resp = requests.get(_URL, headers=_HEADERS, timeout=10)
    resp.raise_for_status()
    data = resp.json()

    results = []
    for team_entry in data.get("injuries", []):
        abbr = team_entry.get("team", {}).get("abbreviation", "")
        for injury in team_entry.get("injuries", []):
            status = injury.get("status", "")
            if status not in _TRACKED_STATUSES:
                continue
            results.append({
                "team_abbreviation": abbr,
                "player_name": injury.get("athlete", {}).get("displayName", ""),
                "status": status,
                "reason": injury.get("longComment") or injury.get("shortComment", ""),
            })

    logger.info(f"ESPN: {len(results)} injury entries fetched")
    return results
