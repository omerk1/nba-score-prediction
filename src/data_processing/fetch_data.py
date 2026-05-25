"""
Fetch NBA game data from nba_api and store in SQLite.
- First run: backfills all seasons from data_start_date (in config) to today.
- Subsequent runs: only re-fetches the current season (new games INSERT OR IGNORE past ones).
Re-running is always safe — existing rows are skipped via INSERT OR IGNORE on game_id PK.
"""

import datetime
import logging
import sqlite3
import time
from pathlib import Path
from typing import Optional

import pandas as pd
from nba_api.stats.endpoints import LeagueGameLog, ScoreboardV2

from src.utils.config_loader import load_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DB_PATH = Path("data/raw/nba_api.sqlite")
SLEEP_SECONDS = 0.7  # required to avoid stats.nba.com rate limiting
SEASON_TYPES = ["Regular Season", "Playoffs"]

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS game (
    game_id       TEXT PRIMARY KEY,
    game_date     TEXT,
    season_id     TEXT,
    season_type   TEXT,
    team_id_home  INTEGER,
    team_id_away  INTEGER,
    pts_home      REAL,
    pts_away      REAL,
    wl_home       TEXT,
    fg_pct_home   REAL,
    ft_pct_home   REAL,
    fg3_pct_home  REAL,
    ast_home      INTEGER,
    reb_home      INTEGER,
    fg_pct_away   REAL,
    ft_pct_away   REAL,
    fg3_pct_away  REAL,
    ast_away      INTEGER,
    reb_away      INTEGER
)
"""


def _date_to_season(date: str) -> str:
    """'2016-10-01' → '2016-17'.  Dates before October belong to the previous season."""
    d = datetime.date.fromisoformat(date[:10])
    year = d.year if d.month >= 10 else d.year - 1
    return f"{year}-{str(year + 1)[2:]}"


def _season_list(start: str) -> list[str]:
    start_year = int(start[:4])
    today = datetime.date.today()
    current_year = today.year if today.month >= 10 else today.year - 1
    return [f"{y}-{str(y + 1)[2:]}" for y in range(start_year, current_year + 1)]


def _get_last_game_date(conn: sqlite3.Connection) -> Optional[str]:
    row = conn.execute("SELECT MAX(game_date) FROM game").fetchone()
    return row[0] if row and row[0] else None


def _fetch_season(season: str, season_type: str) -> pd.DataFrame:
    """Pull one season/type from the API and return a wide-format DataFrame."""
    df = LeagueGameLog(
        season=season,
        season_type_all_star=season_type,
        league_id="00",
    ).get_data_frames()[0]

    if df.empty:
        return pd.DataFrame()

    # MATCHUP is "X vs. Y" for the home team's row, "X @ Y" for away
    home = df[df["MATCHUP"].str.contains("vs\\.")].copy()
    away = df[df["MATCHUP"].str.contains("@")].copy()
    merged = home.merge(away, on="GAME_ID", suffixes=("_home", "_away"))

    return pd.DataFrame({
        "game_id":      merged["GAME_ID"],
        "game_date":    merged["GAME_DATE_home"],
        "season_id":    merged["SEASON_ID_home"],
        "season_type":  season_type,
        "team_id_home": merged["TEAM_ID_home"],
        "team_id_away": merged["TEAM_ID_away"],
        "pts_home":     merged["PTS_home"],
        "pts_away":     merged["PTS_away"],
        "wl_home":      merged["WL_home"],
        "fg_pct_home":  merged["FG_PCT_home"],
        "ft_pct_home":  merged["FT_PCT_home"],
        "fg3_pct_home": merged["FG3_PCT_home"],
        "ast_home":     merged["AST_home"],
        "reb_home":     merged["REB_home"],
        "fg_pct_away":  merged["FG_PCT_away"],
        "ft_pct_away":  merged["FT_PCT_away"],
        "fg3_pct_away": merged["FG3_PCT_away"],
        "ast_away":     merged["AST_away"],
        "reb_away":     merged["REB_away"],
    })


def _insert_games(conn: sqlite3.Connection, df: pd.DataFrame) -> int:
    cols = list(df.columns)
    sql = f"INSERT OR IGNORE INTO game ({', '.join(cols)}) VALUES ({', '.join('?' * len(cols))})"
    rows = [tuple(r) for r in df.itertuples(index=False)]
    cursor = conn.executemany(sql, rows)
    conn.commit()
    return cursor.rowcount


def fetch_upcoming_games(target_date: Optional[str] = None) -> pd.DataFrame:
    """
    Return today's (or target_date's) scheduled matchups from the NBA scoreboard.
    Games not yet played will have NaN scores — suitable for inference.
    """
    if target_date is None:
        target_date = datetime.date.today().isoformat()

    scoreboard = ScoreboardV2(game_date=target_date).game_header.get_data_frame()
    if scoreboard.empty:
        return pd.DataFrame()

    return pd.DataFrame({
        "game_id":      scoreboard["GAME_ID"],
        "game_date":    target_date,
        "season_id":    scoreboard.get("SEASON", pd.NA),
        "season_type":  scoreboard.get("LIVE_PERIOD_TIME_BCAST", pd.NA),  # placeholder
        "team_id_home": scoreboard["HOME_TEAM_ID"],
        "team_id_away": scoreboard["VISITOR_TEAM_ID"],
    })


def main():
    cfg = load_config()
    start_season = _date_to_season(cfg.datasets_loading.data_start_date)

    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute(CREATE_TABLE_SQL)
    conn.commit()

    last_date = _get_last_game_date(conn)
    if last_date:
        # Incremental: only re-fetch from the season that contains the last stored game.
        # New games only appear in the current season; INSERT OR IGNORE skips duplicates.
        incremental_start = _date_to_season(last_date)
        seasons = [s for s in _season_list(start_season) if s >= incremental_start]
        logger.info(f"Incremental fetch from {incremental_start} (last game: {last_date})")
    else:
        seasons = _season_list(start_season)
        logger.info(f"Full backfill from {start_season}")

    total = 0
    for season in seasons:
        for season_type in SEASON_TYPES:
            try:
                logger.info(f"Fetching {season} {season_type}...")
                df = _fetch_season(season, season_type)
                if df.empty:
                    logger.info("  No data, skipping.")
                else:
                    inserted = _insert_games(conn, df)
                    logger.info(f"  {inserted} new rows ({len(df)} fetched).")
                    total += inserted
            except Exception as e:
                logger.error(f"  Error: {e}")
            time.sleep(SLEEP_SECONDS)

    conn.close()
    logger.info(f"Done. Total new games inserted: {total:,}")


if __name__ == "__main__":
    main()