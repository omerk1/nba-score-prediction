"""sqlite storage for the availability module (own file, additive, never touches nba_api.sqlite)."""

import sqlite3
from pathlib import Path

DEFAULT_DB_PATH = "data/raw/availability.sqlite"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS player_game_log (
    player_id    INTEGER NOT NULL,
    player_name  TEXT    NOT NULL,
    team_id      INTEGER NOT NULL,
    game_id      TEXT    NOT NULL,
    game_date    TEXT    NOT NULL,
    season       TEXT    NOT NULL,
    season_type  TEXT    NOT NULL,
    minutes      REAL    NOT NULL,
    pts          REAL,
    PRIMARY KEY (player_id, game_id)
);
CREATE INDEX IF NOT EXISTS idx_pgl_team_date ON player_game_log(team_id, game_date);
CREATE INDEX IF NOT EXISTS idx_pgl_player_date ON player_game_log(player_id, game_date);

CREATE TABLE IF NOT EXISTS backfill_log (
    season       TEXT NOT NULL,
    season_type  TEXT NOT NULL,
    n_rows       INTEGER NOT NULL,
    fetched_at   TEXT NOT NULL,
    PRIMARY KEY (season, season_type)
);
"""


def get_conn(db_path: str = DEFAULT_DB_PATH) -> sqlite3.Connection:
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.executescript(_SCHEMA)
    return conn
