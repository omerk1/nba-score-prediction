"""Fetch raw play-by-play events from nba_api's PlayByPlayV3 and store them.

V3 (not V2) because it carries shot coordinates, shot distance and shot value
as typed columns, and its schema is identical from 2016-17 through today
(checked on games 0021600001 and 0022400500). The one V3 gap: a
Substitution row identifies only the OUTGOING player by id; the incoming
player appears by last name in the description. possessions.py resolves that.
"""

import datetime
import logging
import sqlite3
import time

import pandas as pd
from nba_api.stats.endpoints import PlayByPlayV3

logger = logging.getLogger(__name__)

SLEEP_SECONDS = 0.7  # stats.nba.com rate-limit convention shared with fetch_data.py
MAX_RETRIES = 3

# nba_api column -> pbp_events column
EVENT_COLUMNS = {
    "gameId": "game_id",
    "actionId": "action_id",
    "actionNumber": "action_number",
    "period": "period",
    "clock": "clock",
    "teamId": "team_id",
    "personId": "person_id",
    "playerName": "player_name",
    "playerNameI": "player_name_i",  # 'L. James' -- the form substitution text uses for shared surnames
    "actionType": "action_type",
    "subType": "sub_type",
    "description": "description",
    "shotDistance": "shot_distance",
    "shotResult": "shot_result",
    "shotValue": "shot_value",
    "xLegacy": "x_legacy",
    "yLegacy": "y_legacy",
    "scoreHome": "score_home",
    "scoreAway": "score_away",
    "location": "location",
}


def fetch_game_events(game_id: str, max_retries: int = MAX_RETRIES) -> pd.DataFrame:
    """Fetch one game's events. Returns an empty frame after all retries fail
    (caller logs the failure so the game can be retried on a later run)."""
    for attempt in range(max_retries):
        try:
            time.sleep(SLEEP_SECONDS)
            df = PlayByPlayV3(game_id=game_id).get_data_frames()[0]
            if df.empty:
                return pd.DataFrame()
            out = df[list(EVENT_COLUMNS)].rename(columns=EVENT_COLUMNS)
            out["game_id"] = out["game_id"].astype(str)
            return out
        except Exception as e:  # noqa: BLE001 -- endpoint raises a mix of HTTP/JSON errors
            wait = 2**attempt
            if attempt < max_retries - 1:
                logger.debug(f"PBP fetch {game_id} attempt {attempt + 1} failed: {e}; retry in {wait}s")
                time.sleep(wait)
            else:
                logger.error(f"PBP fetch {game_id} failed after {max_retries} attempts: {e}")
                raise
    return pd.DataFrame()


def store_game_events(conn: sqlite3.Connection, events: pd.DataFrame) -> int:
    """INSERT OR REPLACE a game's events. Returns rows written."""
    if events.empty:
        return 0
    cols = list(EVENT_COLUMNS.values())
    rows = [tuple(None if pd.isna(v) else v for v in r) for r in events[cols].itertuples(index=False)]
    placeholders = ",".join("?" * len(cols))
    conn.executemany(f"INSERT OR REPLACE INTO pbp_events ({','.join(cols)}) VALUES ({placeholders})", rows)
    return len(rows)


def log_fetch(
    conn: sqlite3.Connection,
    game_id: str,
    season: str,
    game_date: str,
    status: str,
    n_events: int,
    elapsed_s: float,
    error: str | None = None,
) -> None:
    conn.execute(
        "INSERT OR REPLACE INTO pbp_fetch_log "
        "(game_id, season, game_date, status, n_events, elapsed_s, error, fetched_at) "
        "VALUES (?,?,?,?,?,?,?,?)",
        (
            game_id,
            season,
            game_date,
            status,
            n_events,
            round(elapsed_s, 3),
            error,
            datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        ),
    )


def fetched_game_ids(conn: sqlite3.Connection, status: str = "ok") -> set[str]:
    return {r[0] for r in conn.execute("SELECT game_id FROM pbp_fetch_log WHERE status = ?", (status,))}


def load_game_events(conn: sqlite3.Connection, game_id: str) -> pd.DataFrame:
    df = pd.read_sql_query(
        "SELECT * FROM pbp_events WHERE game_id = ? ORDER BY action_id", conn, params=(game_id,)
    )
    for c in (
        "score_home",
        "score_away",
        "description",
        "sub_type",
        "action_type",
        "player_name",
        "player_name_i",
        "shot_result",
        "location",
    ):
        df[c] = df[c].fillna("")
    return df
