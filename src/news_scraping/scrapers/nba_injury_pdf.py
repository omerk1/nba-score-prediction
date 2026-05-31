"""
NBA official pre-game injury report PDFs (available from the 2021-22 season).

Reports are published to the NBA's CDN throughout game days at:
  Before ~Dec 23 2025: Injury-Report_{YYYY-MM-DD}_{HH}{AM|PM}.pdf
  After  ~Dec 22 2025: Injury-Report_{YYYY-MM-DD}_{HH}_{MM}{AM|PM}.pdf  (minutes added)

Multiple versions are released throughout the day as teams submit updates.
We scan from latest to earliest and stop at the first hit.

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

# Around Dec 23 2025 the NBA added minutes to the filename.
_NEW_FORMAT_CUTOVER = date(2025, 12, 22)

# Old format: HH{AM|PM} — ordered latest → earliest
_REPORT_HOURS_OLD = [
    "11PM", "10PM", "09PM", "08PM", "07PM", "06PM", "05PM",
    "04PM", "03PM", "02PM", "01PM", "12PM", "11AM", "10AM",
]

# New format: HH_MM{AM|PM} — try :45/:30/:15/:00 per hour, latest → earliest
_REPORT_HOURS_NEW = [
    f"{h:02d}_{m:02d}{period}"
    for h, period in [
        (11, "PM"), (10, "PM"), (9, "PM"), (8, "PM"), (7, "PM"), (6, "PM"),
        (5, "PM"), (4, "PM"), (3, "PM"), (2, "PM"), (1, "PM"), (12, "PM"),
        (11, "AM"), (10, "AM"),
    ]
    for m in (45, 30, 15, 0)
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


def _col_indices(header: list) -> dict[str, int]:
    """Map normalised column names → indices. Strips spaces so 'Player Name' == 'PlayerName'."""
    return {
        (cell or "").strip().replace(" ", ""): i
        for i, cell in enumerate(header)
        if cell and cell.strip()
    }


def _parse_pdf(content: bytes) -> list[dict]:
    # 2023-24+ PDFs: header only on first page (6 cols), continuation pages have 4 cols
    # (Team, PlayerName, CurrentStatus, Reason) with no header row.
    _CONTINUATION_COL = {"Team": 0, "PlayerName": 1, "CurrentStatus": 2, "Reason": 3}

    rows = []
    current_abbr = None  # persists across pages so team context carries over page breaks

    with pdfplumber.open(BytesIO(content)) as pdf:
        for page in pdf.pages:
            table = _extract_table(page)
            if not table:
                continue

            # Locate header row — normalise to handle 'Player Name' vs 'PlayerName'
            col = None
            data_start = 0
            for i, row in enumerate(table):
                normalised = [(c or "").strip().replace(" ", "") for c in row] if row else []
                if "Team" in normalised and "PlayerName" in normalised:
                    col = _col_indices(row)
                    data_start = i + 1
                    break

            if col is None:
                # Continuation page: no header. Infer format from column count.
                first_row = next((r for r in table if r and any(r)), None)
                if first_row and len(first_row) == 4:
                    col = _CONTINUATION_COL
                    data_start = 0
                else:
                    continue

            team_i = col.get("Team")
            player_i = col.get("PlayerName")
            status_i = col.get("CurrentStatus")
            reason_i = col.get("Reason")
            if any(x is None for x in [team_i, player_i, status_i]):
                continue

            for row in table[data_start:]:
                if not row or not any(row):
                    continue
                team_name = (row[team_i] or "").strip()
                player_raw = (row[player_i] or "").strip()
                status = (row[status_i] or "").strip()
                reason = (row[reason_i] or "").strip() if reason_i is not None and len(row) > reason_i else ""

                if not player_raw:
                    continue

                # Carry team context forward — subsequent players on same team have empty team column
                if team_name:
                    abbr = _TEAM_MAP.get(team_name) or _TEAM_MAP_CONCAT.get(team_name)
                    if abbr:
                        current_abbr = abbr
                    else:
                        logger.debug(f"Unknown team name in PDF: '{team_name}'")
                        current_abbr = None

                if not current_abbr or status not in _TRACKED_STATUSES:
                    continue

                rows.append({
                    "team_abbreviation": current_abbr,
                    "player_name": _normalize_name(player_raw),
                    "status": status,
                    "reason": reason,
                    "days_out": 0,
                })
    return rows


def fetch_injuries_for_date(game_date: date) -> list[dict]:
    """
    Download the latest NBA official injury report PDF for game_date.
    Scans from latest hour to earliest and returns on the first hit,
    so we always get the most up-to-date pre-game report with minimal requests.
    Returns [] if no report found.
    """
    hour_list = _REPORT_HOURS_NEW if game_date >= _NEW_FORMAT_CUTOVER else _REPORT_HOURS_OLD
    for hour_str in hour_list:
        content = _fetch_pdf_bytes(_pdf_url(game_date, hour_str))
        if content:
            rows = _parse_pdf(content)
            logger.info(f"NBA PDF {game_date}: {len(rows)} injury entries")
            return rows
        time.sleep(0.05)

    return []
