import sqlite3
from contextlib import contextmanager
from pathlib import Path

_SCHEMA = """
CREATE TABLE IF NOT EXISTS player_importance (
    player_id        INTEGER,
    player_name      TEXT,
    team_id          INTEGER,
    as_of_date       TEXT,
    minutes_per_game REAL,
    pts_per_game     REAL,
    usage_rate       REAL,
    updated_at       TEXT NOT NULL,
    PRIMARY KEY (player_id, team_id, as_of_date)
);

CREATE TABLE IF NOT EXISTS player_injuries (
    game_date   TEXT NOT NULL,
    team_id     INTEGER NOT NULL,
    player_name TEXT NOT NULL,
    status      TEXT NOT NULL,
    reason      TEXT,
    source      TEXT NOT NULL DEFAULT 'pdf',
    PRIMARY KEY (game_date, team_id, player_name)
);

CREATE TABLE IF NOT EXISTS scrape_log (
    game_date   TEXT NOT NULL,
    source      TEXT NOT NULL,
    report_time TEXT,
    n_entries   INTEGER NOT NULL DEFAULT 0,
    scraped_at  TEXT NOT NULL,
    PRIMARY KEY (game_date, source)
);

CREATE TABLE IF NOT EXISTS injury_features (
    game_date      TEXT NOT NULL,
    team_id        INTEGER NOT NULL,
    scorer         TEXT NOT NULL DEFAULT 'formula',
    impact_score   REAL NOT NULL,
    n_out          INTEGER NOT NULL,
    n_questionable INTEGER NOT NULL,
    updated_at     TEXT NOT NULL,
    PRIMARY KEY (game_date, team_id, scorer)
);
"""


def init_db(db_path: str | Path) -> None:
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        conn.executescript(_SCHEMA)
        conn.execute("PRAGMA journal_mode=WAL")


@contextmanager
def get_conn(db_path: str | Path):
    conn = sqlite3.connect(db_path, timeout=30)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()
