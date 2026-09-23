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

CREATE TABLE IF NOT EXISTS llm_cache (
    prompt_hash   TEXT PRIMARY KEY,
    model         TEXT NOT NULL,
    variant       TEXT NOT NULL,
    prompt        TEXT NOT NULL,
    response_json TEXT NOT NULL,
    created_at    TEXT NOT NULL
);

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
    # check_same_thread=False: the LLM estimator writes cache rows from worker
    # threads, serialized by its own lock.
    #
    # isolation_level=None (autocommit): Python's sqlite3 module defaults to
    # isolation_level="", which implicitly opens a transaction on the first
    # INSERT and does not release it until an explicit commit(). The callers
    # here batch commit() every ~200 calls to avoid the overhead of committing
    # per row, which means a single process can hold the WAL write lock open
    # for MINUTES while it accumulates a batch -- far longer than any
    # reasonable busy_timeout, so a second process's write is refused outright
    # rather than actually waiting. This was the real cause of the "database
    # is locked" crashes seen running LLM eval scripts concurrently
    # (2026-09-23); WAL + busy_timeout alone did not fix it, because the lock
    # genuinely was held that long, not merely contended for an instant.
    # Autocommit makes each INSERT release the lock immediately, so the
    # batched conn.commit() calls elsewhere become harmless no-ops.
    conn = sqlite3.connect(db_path, check_same_thread=False, isolation_level=None)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=30000")
    conn.executescript(_SCHEMA)
    return conn
