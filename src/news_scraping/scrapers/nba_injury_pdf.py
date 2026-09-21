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
from datetime import date, datetime
from io import BytesIO
from pathlib import Path
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
    "11PM",
    "10PM",
    "09PM",
    "08PM",
    "07PM",
    "06PM",
    "05PM",
    "04PM",
    "03PM",
    "02PM",
    "01PM",
    "12PM",
    "11AM",
    "10AM",
]

# New format: HH_MM{AM|PM} — try :45/:30/:15/:00 per hour, latest → earliest
_REPORT_HOURS_NEW = [
    f"{h:02d}_{m:02d}{period}"
    for h, period in [
        (11, "PM"),
        (10, "PM"),
        (9, "PM"),
        (8, "PM"),
        (7, "PM"),
        (6, "PM"),
        (5, "PM"),
        (4, "PM"),
        (3, "PM"),
        (2, "PM"),
        (1, "PM"),
        (12, "PM"),
        (11, "AM"),
        (10, "AM"),
    ]
    for m in (45, 30, 15, 0)
]

_TRACKED_STATUSES = {"Out", "Doubtful", "Questionable"}
# Every status the reports use, tracked or not. Used only to locate the status
# COLUMN on header-less continuation pages, so it must include the ones we
# discard (an "Available" row still marks where the column is).
_ALL_STATUSES = _TRACKED_STATUSES | {"Available", "Probable", "Not With Team", "Not Yet Submitted"}

_TEAM_MAP: dict[str, str] = {t["full_name"]: t["abbreviation"] for t in nba_teams.get_teams()}
# 2023-24+ PDFs concatenate team names without spaces (e.g. "LosAngelesLakers")
_TEAM_MAP_CONCAT: dict[str, str] = {
    t["full_name"].replace(" ", ""): t["abbreviation"] for t in nba_teams.get_teams()
}

# Names the PDFs use that do not match nba_api's `full_name`. The reports write
# "LA Clippers", never "Los Angeles Clippers", so before this map every Clippers
# listing in every report failed to resolve and was dropped without a trace --
# the team simply looked healthy. Extraction telemetry (`unknown_team_names` in
# _parse_pdf's stats) is what surfaced it; add new variants here as it reports
# them.
_TEAM_MAP_ALIASES: dict[str, str] = {
    "LA Clippers": "LAC",
    "Los Angeles Clippers": "LAC",
}
_TEAM_MAP_CONCAT.update({k.replace(" ", ""): v for k, v in _TEAM_MAP_ALIASES.items()})


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
        (cell or "").strip().replace(" ", ""): i for i, cell in enumerate(header) if cell and cell.strip()
    }


def _infer_continuation_columns(table: list[list]) -> dict[str, int] | None:
    """Locate columns on a header-less continuation page from cell contents.

    The status column is the one whose cells most often hold a known status
    word; player name sits immediately left of it, reason immediately right,
    and team two to the left when that column exists at all. Returns None when
    no status-like column is found, which is the genuine "this page holds no
    listings" case rather than a layout we failed to read.
    """
    counts: dict[int, int] = {}
    for row in table:
        for i, cell in enumerate(row or []):
            if (cell or "").strip() in _ALL_STATUSES:
                counts[i] = counts.get(i, 0) + 1
    if not counts:
        return None
    status_i = max(counts, key=lambda k: counts[k])
    if status_i < 1:  # nothing to the left means no player-name column
        return None
    col = {"CurrentStatus": status_i, "PlayerName": status_i - 1, "Reason": status_i + 1}
    if status_i >= 2:
        col["Team"] = status_i - 2
    return col


def _parse_game_date(raw: str) -> str | None:
    """PDF 'Game Date' cell -> ISO date. Seen as MM/DD/YYYY and MM/DD/YY; ISO is
    accepted defensively. Returns None when the cell is empty or unrecognised,
    which makes the caller carry the previous row's date forward."""
    s = (raw or "").strip()
    if not s:
        return None
    for fmt in ("%m/%d/%Y", "%m/%d/%y", "%Y-%m-%d"):
        try:
            return datetime.strptime(s, fmt).date().isoformat()
        except ValueError:
            continue
    return None


def _parse_pdf(content: bytes) -> tuple[list[dict], dict]:
    """Returns (rows, stats). `stats` is extraction telemetry -- pages seen,
    pages skipped and why, rows kept, and cells that failed to resolve -- so a
    silent regression in table detection is visible instead of showing up as a
    team with nobody injured. See docs/features/injury_pdf_extraction_scope.md.

    Each row carries `game_date`, read from the PDF's own 'Game Date' column and
    carried forward across rows and pages (continuation pages omit it, exactly
    as they omit the repeated team name). A report published on date D covers
    both D's late games and D+1's, so this column -- not the report's own date --
    says which game a listing belongs to.
    """
    # 2023-24+ PDFs: header only on first page (6 cols), continuation pages have 4 cols
    # (Team, PlayerName, CurrentStatus, Reason) with no header row.
    _CONTINUATION_COL = {"Team": 0, "PlayerName": 1, "CurrentStatus": 2, "Reason": 3}

    rows = []
    current_abbr = None  # persists across pages so team context carries over page breaks
    current_date = None  # same, for the Game Date column (absent on continuation pages)
    current_matchup = None  # same, for Matchup -- the only game identifier from 2023-24 on
    stats = {
        "pages": 0,
        "pages_no_table": 0,
        "pages_no_header": 0,
        "pages_missing_columns": 0,
        "rows_seen": 0,
        "rows_kept": 0,
        "rows_no_team": 0,
        "rows_untracked_status": 0,
        "rows_no_date": 0,
        "rows_no_date_or_matchup": 0,
        "unknown_team_names": set(),
        "has_matchup_column": False,
        "has_date_column": False,
    }

    with pdfplumber.open(BytesIO(content)) as pdf:
        for page in pdf.pages:
            stats["pages"] += 1
            table = _extract_table(page)
            if not table:
                stats["pages_no_table"] += 1
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
                # Continuation page: no header row. Column COUNT is unreliable --
                # the text-alignment fallback emits only the columns that happen to
                # have content on that page, so a page where every row inherits the
                # team name collapses to 3 columns and a page with a matchup keeps 5.
                # Locate the status column by its contents instead, and derive the
                # rest from its position.
                col = _infer_continuation_columns(table)
                data_start = 0
                if col is None:
                    stats["pages_no_header"] += 1
                    continue

            team_i = col.get("Team")
            player_i = col.get("PlayerName")
            status_i = col.get("CurrentStatus")
            reason_i = col.get("Reason")
            date_i = col.get("GameDate")
            # 2023-24 onward the ruled-table extraction yields only the header row,
            # so _extract_table falls back to text alignment -- and that fallback
            # drops the Game Date and Game Time columns entirely, keeping Matchup
            # ("MIN@BOS") as the only game identifier. Matchup plus the report date
            # resolves to a date against the schedule, since a report covers only
            # that evening and the next day (see resolve_dates_from_matchup in
            # scripts/rebuild_injury_dates.py).
            matchup_i = col.get("Matchup")
            if date_i is not None:
                stats["has_date_column"] = True
            if matchup_i is not None:
                stats["has_matchup_column"] = True
            # `team_i` may legitimately be absent on a continuation page whose rows
            # all inherit the team from the previous page, so only player and
            # status are required.
            if any(x is None for x in [player_i, status_i]):
                stats["pages_missing_columns"] += 1
                continue

            for row in table[data_start:]:
                if not row or not any(row):
                    continue
                stats["rows_seen"] += 1
                if date_i is not None and len(row) > date_i:
                    parsed = _parse_game_date(row[date_i])
                    if parsed:
                        current_date = parsed
                if matchup_i is not None and len(row) > matchup_i:
                    m = (row[matchup_i] or "").strip().replace(" ", "")
                    if "@" in m:
                        current_matchup = m
                team_name = (row[team_i] or "").strip() if team_i is not None and len(row) > team_i else ""
                player_raw = (row[player_i] or "").strip()
                status = (row[status_i] or "").strip()
                reason = (row[reason_i] or "").strip() if reason_i is not None and len(row) > reason_i else ""

                if not player_raw:
                    continue

                # Carry team context forward — subsequent players on same team have empty team column
                if team_name:
                    abbr = (
                        _TEAM_MAP.get(team_name)
                        or _TEAM_MAP_ALIASES.get(team_name)
                        or _TEAM_MAP_CONCAT.get(team_name)
                    )
                    if abbr:
                        current_abbr = abbr
                    else:
                        logger.debug(f"Unknown team name in PDF: '{team_name}'")
                        stats["unknown_team_names"].add(team_name)
                        current_abbr = None

                if not current_abbr:
                    stats["rows_no_team"] += 1
                    continue
                if status not in _TRACKED_STATUSES:
                    stats["rows_untracked_status"] += 1
                    continue
                if current_date is None:
                    stats["rows_no_date"] += 1
                    if current_matchup is None:
                        stats["rows_no_date_or_matchup"] += 1

                stats["rows_kept"] += 1
                rows.append(
                    {
                        "team_abbreviation": current_abbr,
                        "player_name": _normalize_name(player_raw),
                        "status": status,
                        "reason": reason,
                        "game_date": current_date,
                        "matchup": current_matchup,
                        "days_out": 0,
                    }
                )
    stats["unknown_team_names"] = sorted(stats["unknown_team_names"])
    return rows, stats


def fetch_injuries_for_date(
    report_date: date, cache_dir: str | None = None
) -> tuple[list[dict], str | None, dict]:
    """
    Download the latest NBA official injury report PDF published on `report_date`.
    Scans from latest hour to earliest and returns on the first hit,
    so we always get the most up-to-date report with minimal requests.

    The argument is the REPORT's date, which is not the date of the games it
    covers: an 11PM report lists both that evening's late games and the next
    day's. Each returned row carries its own `game_date`, read from the PDF.

    `cache_dir` stores every fetched PDF on disk so later re-parses never
    re-download (see docs/features/injury_pdf_extraction_scope.md, risks).

    Returns (entries, report_time, stats); ([], None, {}) if no report found.
    """
    hour_list = _REPORT_HOURS_NEW if report_date >= _NEW_FORMAT_CUTOVER else _REPORT_HOURS_OLD
    for hour_str in hour_list:
        content = _cached_pdf_bytes(report_date, hour_str, cache_dir)
        if content:
            rows, stats = _parse_pdf(content)
            dates = {r["game_date"] for r in rows if r["game_date"]}
            logger.info(
                f"NBA PDF {report_date} ({hour_str}): {len(rows)} entries "
                f"covering game dates {sorted(dates)}"
            )
            if stats["rows_no_date"]:
                logger.warning(f"NBA PDF {report_date}: {stats['rows_no_date']} rows without a game date")
            return rows, hour_str, stats
        time.sleep(0.05)

    return [], None, {}


def _cached_pdf_bytes(report_date: date, hour_str: str, cache_dir: str | None) -> Optional[bytes]:
    """Fetch a report PDF, reading from and writing to `cache_dir` when given."""
    if cache_dir is None:
        return _fetch_pdf_bytes(_pdf_url(report_date, hour_str))
    path = Path(cache_dir) / f"Injury-Report_{report_date.isoformat()}_{hour_str}.pdf"
    if path.exists():
        return path.read_bytes()
    content = _fetch_pdf_bytes(_pdf_url(report_date, hour_str))
    if content:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    return content
