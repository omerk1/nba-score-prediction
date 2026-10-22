"""
Who is listed out on a slate day, straight from the raw injury records the
nightly ESPN step stores (player_injuries). Display only: the model's own
injury features are the aggregated counts in injury_features.
"""

import sqlite3
from pathlib import Path

from src.utils.config_loader import load_config

SHOWN_STATUSES = ("Out", "Doubtful", "Questionable", "Day-To-Day")


def injuries_for_date(day, db_path: str | None = None) -> dict[int, list[dict]] | None:
    """{team_id: [{player_name, status, reason}]} for the day's ESPN snapshot,
    Out first. None when that day has no snapshot at all (scrape failed) —
    distinct from {} / a missing team, which means nobody is listed."""
    db_path = db_path or load_config().injury_features.db_path
    if not Path(db_path).exists():
        return None
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        logged = conn.execute(
            "SELECT 1 FROM scrape_log WHERE game_date = ? AND source = 'espn'", (str(day),)
        ).fetchone()
        if not logged:
            return None
        rows = conn.execute(
            "SELECT team_id, player_name, status, reason FROM player_injuries "
            "WHERE game_date = ? AND source = 'espn'",
            (str(day),),
        ).fetchall()
    finally:
        conn.close()
    out: dict[int, list[dict]] = {}
    for team_id, name, status, reason in rows:
        if status in SHOWN_STATUSES:
            out.setdefault(team_id, []).append({"player_name": name, "status": status, "reason": reason or ""})
    for players in out.values():
        players.sort(key=lambda p: SHOWN_STATUSES.index(p["status"]))
    return out
