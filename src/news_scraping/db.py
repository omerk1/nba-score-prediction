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
    updated_at       TEXT DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (player_id, team_id, as_of_date)
);

CREATE TABLE IF NOT EXISTS injury_features (
    game_date      TEXT,
    team_id        INTEGER,
    impact_score   REAL,
    n_out          INTEGER,
    n_questionable INTEGER,
    star_out       INTEGER,
    raw_report     TEXT,
    updated_at     TEXT DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (game_date, team_id)
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