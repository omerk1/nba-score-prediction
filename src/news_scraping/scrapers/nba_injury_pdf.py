"""
NBA official pre-game injury report PDFs (available from the 2021-22 season).

The NBA publishes a PDF injury report before each game day at:
  https://official.nba.com/nba-injury-report-{MM}-{DD}-{YYYY}-{time}.pdf

Multiple versions are released throughout the day as rosters update (01pm, 05pm,
06pm, 09pm are the most common). We try all known times and use the latest found,
so the report reflects the most up-to-date status before tip-off.

PDF column layout (0-indexed):
  0: Game Date  1: Game Time  2: Matchup  3: Team  4: Player Name
  5: Current Status  6: Reason
"""

import logging
import time
from datetime import date
from io import BytesIO
from typing import Optional

import pdfplumber
import requests
from nba_api.stats.static import teams as nba_teams

logger = logging.getLogger(__name__)

_BASE_URL = "https://official.nba.com"
_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    )
}

_REPORT_TIMES = ["01pm", "05pm", "06pm", "09pm"]
_TRACKED_STATUSES = {"Out", "Doubtful", "Questionable"}

_TEAM_MAP: dict[str, str] = {
    t["full_name"]: t["abbreviation"] for t in nba_teams.get_teams()
}


def _pdf_url(game_date: date, time_str: str) -> str:
    return f"{_BASE_URL}/nba-injury-report-{game_date.strftime('%m-%d-%Y')}-{time_str}.pdf"


def _fetch_pdf_bytes(url: str) -> Optional[bytes]:
    try:
        resp = requests.get(url, headers=_HEADERS, timeout=15)
        if resp.status_code == 200 and "pdf" in resp.headers.get("content-type", ""):
            return resp.content
    except requests.RequestException:
        pass
    return None


def _parse_pdf(content: bytes) -> list[dict]:
    rows = []
    with pdfplumber.open(BytesIO(content)) as pdf:
        for page in pdf.pages:
            table = page.extract_table()
            if not table:
                continue
            for row in table:
                if not row or len(row) < 6:
                    continue
                team_name = (row[3] or "").strip()
                player = (row[4] or "").strip()
                status = (row[5] or "").strip()
                reason = (row[6] or "").strip() if len(row) > 6 else ""

                if not player or not team_name or status not in _TRACKED_STATUSES:
                    continue

                abbr = _TEAM_MAP.get(team_name)
                if not abbr:
                    logger.debug(f"Unknown team name in PDF: '{team_name}'")
                    continue

                rows.append({
                    "team_abbreviation": abbr,
                    "player_name": player,
                    "status": status,
                    "reason": reason,
                    "days_out": 0,  # PDFs don't include days out; absence decay defaults to 1.0
                })
    return rows


def fetch_injuries_for_date(game_date: date) -> list[dict]:
    """
    Download the latest NBA official injury report PDF for game_date.
    Tries all known report times and uses the last successful one.
    Returns [] if no report exists (pre-2021, off-season, or no games that day).
    """
    best_content: Optional[bytes] = None
    for time_str in _REPORT_TIMES:
        content = _fetch_pdf_bytes(_pdf_url(game_date, time_str))
        if content:
            best_content = content
            logger.debug(f"Found report for {game_date} at {time_str}")
        time.sleep(0.2)

    if best_content is None:
        return []

    rows = _parse_pdf(best_content)
    logger.info(f"NBA PDF {game_date}: {len(rows)} injury entries")
    return rows
