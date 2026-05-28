import sqlite3
from contextlib import contextmanager
from pathlib import Path

_SCHEMA = """
CREATE TABLE IF NOT EXISTS player_importance (
    player_id        INTEGER,
    player_name      TEXT,
    team_id          INTEGER,
    as_of_date       TEXT,
    importance_score REAL,
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
    days_out    INTEGER DEFAULT 0,
    PRIMARY KEY (game_date, team_id, player_name)
);

CREATE TABLE IF NOT EXISTS injury_features (
    game_date      TEXT NOT NULL,
    team_id        INTEGER NOT NULL,
    scorer         TEXT NOT NULL DEFAULT 'formula',
    impact_score   REAL NOT NULL,
    n_out          INTEGER NOT NULL,
    n_questionable INTEGER NOT NULL,
    star_out       INTEGER NOT NULL,
    updated_at     TEXT NOT NULL,
    PRIMARY KEY (game_date, team_id, scorer)
);
"""


def init_db(db_path: str | Path) -> None:
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        conn.executescript(_SCHEMA)


@contextmanager
def get_conn(db_path: str | Path):
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()
