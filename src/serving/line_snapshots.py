"""
Append-only log of Winner's lines next to Polymarket's fair value, one row
per game per capture. The daily job writes an "afternoon" row per game and
the evening job (scripts/evening_snapshot.py) an "evening" row after the
NBA's afternoon injury report, so the two can be compared later: does
Winner's line fall behind Polymarket's when news breaks?
(docs/features/winner_vs_polymarket/PLAN.md, step 3.)
"""

import csv
import datetime
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

SNAPSHOT_PATH = Path("outputs/line_snapshots.csv")
PICK_COLUMNS = (
    "home_spread",
    "home_spread_odds",
    "away_spread_odds",
    "total_line",
    "over_odds",
    "under_odds",
)
PM_COLUMNS = (
    "pm_fair_margin",
    "pm_fair_total",
    "pm_spread_lines",
    "pm_total_lines",
    "home_spread_pm_probability",
    "home_spread_pm_ev",
    "away_spread_pm_probability",
    "away_spread_pm_ev",
    "over_pm_probability",
    "over_pm_ev",
    "under_pm_probability",
    "under_pm_ev",
)
COLUMNS = ("label", "captured_at", "game_date", "home_team_id", "away_team_id") + PICK_COLUMNS + PM_COLUMNS


def append_snapshot(rows: list[dict], label: str, path: Path = None, now: datetime.datetime = None) -> int:
    """`rows`: dicts carrying game_date, team ids, Winner's lines/odds and the
    polymarket_pricing fields (missing ones are left blank). Returns rows written."""
    if not rows:
        return 0
    path = path or SNAPSHOT_PATH
    captured_at = (now or datetime.datetime.now(datetime.timezone.utc)).isoformat(timespec="seconds")
    write_header = not path.exists() or path.stat().st_size == 0
    if not write_header:
        with open(path, newline="") as f:
            if tuple(next(csv.reader(f), ())) != COLUMNS:
                raise ValueError(f"{path} header doesn't match the current columns; not appending")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS, lineterminator="\n", extrasaction="ignore")
        if write_header:
            writer.writeheader()
        for r in rows:
            writer.writerow({**{c: r.get(c) for c in COLUMNS}, "label": label, "captured_at": captured_at})
    logger.info(f"appended {len(rows)} {label} line snapshot(s) to {path}")
    return len(rows)
