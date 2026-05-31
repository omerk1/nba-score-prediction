"""
NBA official pre-game injury report PDFs (available from the 2021-22 season).

Reports are published to the NBA's CDN throughout game days at:
  https://ak-static.cms.nba.com/referee/injury/Injury-Report_{YYYY-MM-DD}_{HH}{AM|PM}.pdf

Multiple versions are released throughout the day as teams submit updates
(typically 10AM through 9PM). We try all common hours and use the latest
available report, which reflects the most up-to-date pre-game status.

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

_CDN_BASE = "https://ak-static.cms.nba.com/referee/injury"
_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    )
}

# Report hours ordered latest → earliest.
# We stop at the first hit so we always get the most up-to-date pre-game report
# without making unnecessary requests for earlier (less complete) versions.
_REPORT_HOURS = [
    "11PM", "10PM", "09PM", "08PM", "07PM", "06PM", "05PM",
    "04PM", "03PM", "02PM", "01PM", "12PM", "11AM", "10AM",
]

_TRACKED_STATUSES = {"Out", "Doubtful", "Questionable"}

_TEAM_MAP: dict[str, str] = {
    t["full_name"]: t["abbreviation"] for t in nba_teams.get_teams()
}
# 2023-24+ PDFs concatenate team names without spaces (e.g. "LosAngelesLakers")
_TEAM_MAP_CONCAT: dict[str, str] = {
    t["full_name"].replace(" ", ""): t["abbreviation"] for t in nba_teams.get_teams()
}


def _pdf_url(game_date: date, hour_str: str) -> str:
    return f"{_CDN_BASE}/Injury-Report_{game_date.isoformat()}_{hour_str}.pdf"


def _fetch_pdf_bytes(url: str) -> Optional[bytes]:
    try:
        resp = requests.get(url, headers=_HEADERS, timeout=5)
        if resp.status_code == 200 and "pdf" in resp.headers.get("content-type", ""):
            return resp.content
    except requests.RequestException:
        pass
    return None


def _normalize_name(pdf_name: str) -> str:
    """Convert PDF 'Last, First' or 'Last,First' format to nba_api 'First Last' format."""
    sep = ", " if ", " in pdf_name else ","
    parts = pdf_name.split(sep, 1)
    return f"{parts[1]} {parts[0]}" if len(parts) == 2 else pdf_name


_TEXT_STRATEGY = {"vertical_strategy": "text", "horizontal_strategy": "text"}


def _extract_table(page) -> list[list]:
    """Try default extraction first; fall back to text-alignment strategy for borderless PDFs."""
    table = page.extract_table()
    if table and len(table) > 1:
        return table
    return page.extract_table(_TEXT_STRATEGY) or []


def _parse_row(row: list) -> dict | None:
    """Parse a single PDF table row into an injury dict, or None if not relevant."""
    if not row or len(row) < 6:
        return None
    team_name = (row[3] or "").strip()
    player_raw = (row[4] or "").strip()
    status = (row[5] or "").strip()
    reason = (row[6] or "").strip() if len(row) > 6 else ""

    if not player_raw or not team_name or status not in _TRACKED_STATUSES:
        return None

    abbr = _TEAM_MAP.get(team_name) or _TEAM_MAP_CONCAT.get(team_name)
    if not abbr:
        logger.debug(f"Unknown team name in PDF: '{team_name}'")
        return None

    return {
        "team_abbreviation": abbr,
        "player_name": _normalize_name(player_raw),
        "status": status,
        "reason": reason,
        "days_out": 0,
    }


def _parse_pdf(content: bytes) -> list[dict]:
    rows = []
    with pdfplumber.open(BytesIO(content)) as pdf:
        for page in pdf.pages:
            for row in _extract_table(page):
                entry = _parse_row(row)
                if entry:
                    rows.append(entry)
    return rows


def fetch_injuries_for_date(game_date: date) -> list[dict]:
    """
    Download the latest NBA official injury report PDF for game_date.
    Scans from latest hour to earliest and returns on the first hit,
    so we always get the most up-to-date pre-game report with minimal requests.
    Returns [] if no report found.
    """
    for hour_str in _REPORT_HOURS:
        content = _fetch_pdf_bytes(_pdf_url(game_date, hour_str))
        if content:
            rows = _parse_pdf(content)
            logger.info(f"NBA PDF {game_date}: {len(rows)} injury entries")
            return rows
        time.sleep(0.05)

    return []
