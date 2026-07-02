"""
DB access helpers for the A7 style-matchup module.

Safety: `nba_api.sqlite` and `injury_features.sqlite` are symlinked into this
worktree from the human's live working copy of the repo (they are gitignored,
data-only files). They are opened strictly read-only (SQLite URI mode=ro) so
this module can never write to, corrupt, or race with the human's local copy.
All new caching lives in a separate additive file: outputs/a7_matchups_cache.sqlite.
"""

import sqlite3
from pathlib import Path

from src.matchups.config import CACHE_DB, INJURY_DB, NBA_API_DB


def ro_connect(path: str) -> sqlite3.Connection:
    """Open an existing sqlite file strictly read-only."""
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def nba_api_conn() -> sqlite3.Connection:
    return ro_connect(NBA_API_DB)


def injury_conn() -> sqlite3.Connection:
    return ro_connect(INJURY_DB)


def cache_conn() -> sqlite3.Connection:
    """Read-write connection to our own additive cache DB (outputs/a7_matchups_cache.sqlite)."""
    Path(CACHE_DB).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(CACHE_DB)
    conn.row_factory = sqlite3.Row
    return conn


def table_exists(conn: sqlite3.Connection, table: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)
    ).fetchone()
    return row is not None


CACHE_SCHEMAS = {
    "box_score_stats": """
        CREATE TABLE IF NOT EXISTS box_score_stats (
            game_id       TEXT NOT NULL,
            team_id       INTEGER NOT NULL,
            is_home       INTEGER NOT NULL,
            game_date     TEXT,
            fgm REAL, fga REAL, fg3m REAL, fg3a REAL, ftm REAL, fta REAL,
            oreb REAL, dreb REAL, ast REAL, stl REAL, blk REAL, tov REAL, pts REAL,
            PRIMARY KEY (game_id, team_id)
        )
    """,
    "player_name_resolution": """
        CREATE TABLE IF NOT EXISTS player_name_resolution (
            player_name TEXT PRIMARY KEY,
            player_id INTEGER,
            resolution_method TEXT,
            confidence TEXT
        )
    """,
    "player_archetypes": """
        CREATE TABLE IF NOT EXISTS player_archetypes (
            player_id INTEGER NOT NULL,
            season TEXT NOT NULL,
            archetype TEXT,
            ast_pct REAL, ppg_pct REAL, blk_pct REAL, reb_pct REAL, stl_pct REAL,
            PRIMARY KEY (player_id, season)
        )
    """,
    "injury_calibration": """
        CREATE TABLE IF NOT EXISTS injury_calibration (
            archetype TEXT NOT NULL,
            metric TEXT NOT NULL,
            delta REAL,
            n_games_without INTEGER,
            n_games_baseline INTEGER,
            PRIMARY KEY (archetype, metric)
        )
    """,
    "matchup_fingerprints": """
        CREATE TABLE IF NOT EXISTS matchup_fingerprints (
            game_id TEXT NOT NULL,
            team_id INTEGER NOT NULL,
            game_date TEXT,
            layer INTEGER NOT NULL,
            pace_score REAL, three_pt_reliance REAL, paint_activity REAL,
            defensive_rating REAL, assist_rate REAL,
            n_games_in_window INTEGER,
            PRIMARY KEY (game_id, team_id, layer)
        )
    """,
}


def init_cache_db() -> None:
    conn = cache_conn()
    for name, ddl in CACHE_SCHEMAS.items():
        if not table_exists(conn, name):
            conn.execute(ddl)
    conn.commit()
    conn.close()
