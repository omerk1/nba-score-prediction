"""
Backfill per-game player minutes into data/raw/availability.sqlite.

Labels for the availability task ("did a Questionable/Doubtful player actually
play, and for how long") need per-game minutes, which nothing in the repo stores
(player_stats_cache holds rolling averages only). nba_api's LeagueGameLog with
player_or_team_abbreviation="P" returns every player-game row for a whole
season in ONE call, so a full backfill is ~2 calls per season (regular season +
playoffs), not one per player or per game.

Only players who actually played appear in the log; a listed player with no row
on a date their team played is the "did not play" label (see labels.py).

Usage:
  venv/bin/python3 scripts/backfill_player_game_logs.py [--start 2021-22] [--force]
"""

import argparse
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from nba_api.stats.endpoints import LeagueGameLog

from src.availability.db import DEFAULT_DB_PATH, get_conn
from src.data_processing.fetch_data import _season_list

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

SEASON_TYPES = ("Regular Season", "Playoffs")
SLEEP_SECONDS = 1.5
MAX_RETRIES = 3


def _fetch(season: str, season_type: str) -> pd.DataFrame:
    df = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            df = LeagueGameLog(
                season=season,
                season_type_all_star=season_type,
                player_or_team_abbreviation="P",
                league_id="00",
                timeout=60,
            ).get_data_frames()[0]
            break
        except Exception as e:  # network / rate limit
            if attempt == MAX_RETRIES:
                raise
            wait = 5 * attempt
            logger.warning(f"{season} {season_type}: attempt {attempt} failed ({e}); retry in {wait}s")
            time.sleep(wait)
    if df is None or df.empty:
        return pd.DataFrame()
    return pd.DataFrame(
        {
            "player_id": df["PLAYER_ID"].astype(int),
            "player_name": df["PLAYER_NAME"].astype(str),
            "team_id": df["TEAM_ID"].astype(int),
            "game_id": df["GAME_ID"].astype(str),
            "game_date": pd.to_datetime(df["GAME_DATE"]).dt.strftime("%Y-%m-%d"),
            "season": season,
            "season_type": season_type,
            "minutes": pd.to_numeric(df["MIN"], errors="coerce").fillna(0.0).astype(float),
            "pts": pd.to_numeric(df["PTS"], errors="coerce"),
        }
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2021-22", help="first season (injury reports start 2021-22)")
    ap.add_argument("--db", default=DEFAULT_DB_PATH)
    ap.add_argument("--force", action="store_true", help="re-fetch seasons already logged")
    args = ap.parse_args()

    conn = get_conn(args.db)
    done = {tuple(r) for r in conn.execute("SELECT season, season_type FROM backfill_log")}
    seasons = _season_list(args.start.split("-")[0])

    for season in seasons:
        for season_type in SEASON_TYPES:
            if (season, season_type) in done and not args.force:
                logger.info(f"skip {season} {season_type} (already backfilled)")
                continue
            df = _fetch(season, season_type)
            if not df.empty:
                conn.executemany(
                    "INSERT OR REPLACE INTO player_game_log "
                    "(player_id, player_name, team_id, game_id, game_date, season, season_type, minutes, pts) "
                    "VALUES (?,?,?,?,?,?,?,?,?)",
                    df.itertuples(index=False, name=None),
                )
            conn.execute(
                "INSERT OR REPLACE INTO backfill_log VALUES (?,?,?,?)",
                (season, season_type, len(df), datetime.now(timezone.utc).isoformat()),
            )
            conn.commit()
            logger.info(f"{season} {season_type}: stored {len(df)} player-game rows")
            time.sleep(SLEEP_SECONDS)

    n = conn.execute("SELECT COUNT(*), MIN(game_date), MAX(game_date) FROM player_game_log").fetchone()
    logger.info(f"player_game_log: {n[0]} rows, {n[1]} -> {n[2]}")


if __name__ == "__main__":
    main()
