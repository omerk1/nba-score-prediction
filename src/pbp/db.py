"""SQLite storage for raw play-by-play events and the derived possession table.

Lives in its own additive file (config: pbp.db_path, default
data/raw/pbp.sqlite) -- never writes to nba_api.sqlite, same isolation
convention as src/matchups/db.py and the on/off-splits cache.
"""

import sqlite3
from pathlib import Path

SCHEMAS = {
    # One row per PlayByPlayV3 action, columns kept 1:1 with the endpoint
    # (snake_cased) so the parser can be re-run without re-fetching.
    "pbp_events": """
        CREATE TABLE IF NOT EXISTS pbp_events (
            game_id        TEXT NOT NULL,
            action_id      INTEGER NOT NULL,
            action_number  INTEGER,
            period         INTEGER,
            clock          TEXT,
            team_id        INTEGER,
            person_id      INTEGER,
            player_name    TEXT,
            player_name_i  TEXT,
            action_type    TEXT,
            sub_type       TEXT,
            description    TEXT,
            shot_distance  INTEGER,
            shot_result    TEXT,
            shot_value     INTEGER,
            x_legacy       INTEGER,
            y_legacy       INTEGER,
            score_home     TEXT,
            score_away     TEXT,
            location       TEXT,
            PRIMARY KEY (game_id, action_id)
        )
    """,
    # One row per fetch attempt outcome per game -- the resumability key.
    "pbp_fetch_log": """
        CREATE TABLE IF NOT EXISTS pbp_fetch_log (
            game_id     TEXT PRIMARY KEY,
            season      TEXT,
            game_date   TEXT,
            status      TEXT NOT NULL,
            n_events    INTEGER,
            elapsed_s   REAL,
            error       TEXT,
            fetched_at  TEXT
        )
    """,
    # One row per possession. Lineups are comma-joined sorted player ids;
    # *_complete flags are 0 when the parser could not account for exactly
    # five players (see possessions.py for why that happens).
    "possessions": """
        CREATE TABLE IF NOT EXISTS possessions (
            game_id             TEXT NOT NULL,
            poss_idx            INTEGER NOT NULL,
            period              INTEGER,
            off_team_id         INTEGER,
            def_team_id         INTEGER,
            is_home_off         INTEGER,
            start_secs          REAL,
            end_secs            REAL,
            duration            REAL,
            margin_start        INTEGER,
            margin_end          INTEGER,
            points              INTEGER,
            outcome             TEXT,
            n_fga               INTEGER,
            n_fg3a              INTEGER,
            n_fta               INTEGER,
            n_oreb              INTEGER,
            n_tov               INTEGER,
            n_fgm               INTEGER,
            n_fg3m              INTEGER,
            n_ftm               INTEGER,
            last_shot_value     INTEGER,
            last_shot_distance  INTEGER,
            last_shot_x         INTEGER,
            last_shot_y         INTEGER,
            last_shot_subtype   TEXT,
            lineup_off          TEXT,
            lineup_def          TEXT,
            lineup_off_complete INTEGER,
            lineup_def_complete INTEGER,
            PRIMARY KEY (game_id, poss_idx)
        )
    """,
    # Per-game parse diagnostics: does the possession table reproduce the
    # box-score final, and how much of the game has full lineups.
    "possession_game_summary": """
        CREATE TABLE IF NOT EXISTS possession_game_summary (
            game_id             TEXT PRIMARY KEY,
            home_team_id        INTEGER,
            away_team_id        INTEGER,
            n_poss_home         INTEGER,
            n_poss_away         INTEGER,
            pts_home_poss       INTEGER,
            pts_away_poss       INTEGER,
            tech_ft_pts_home    INTEGER,
            tech_ft_pts_away    INTEGER,
            pts_home_box        REAL,
            pts_away_box        REAL,
            points_reconciled   INTEGER,
            lineup_complete_rate REAL,
            n_events            INTEGER,
            parsed_at           TEXT
        )
    """,
}


def connect(db_path: str | Path) -> sqlite3.Connection:
    """Read-write connection; creates the file and all tables if missing."""
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    for ddl in SCHEMAS.values():
        conn.execute(ddl)
    _add_missing_columns(conn, "possessions", SCHEMAS["possessions"])
    conn.commit()
    return conn


def _add_missing_columns(conn: sqlite3.Connection, table: str, ddl: str) -> None:
    """Additive migration: columns present in the DDL but not in an existing
    table are appended (NULL for old rows; a --force re-parse fills them)."""
    existing = {r[1] for r in conn.execute(f"PRAGMA table_info({table})")}
    body = ddl.split("(", 1)[1].rsplit(")", 1)[0]
    for line in body.splitlines():
        parts = line.strip().rstrip(",").split()
        if len(parts) >= 2 and parts[0] not in existing and parts[0].upper() != "PRIMARY":
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {parts[0]} {parts[1]}")
