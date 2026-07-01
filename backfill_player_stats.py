"""
Backfill player_stats_cache with historical player statistics.

This script populates the player_stats_cache table with game-level player statistics
from the NBA API. It processes games chronologically and can be resumed if interrupted.

Usage:
    python backfill_player_stats.py                    # Full backfill
    python backfill_player_stats.py --start 2023-10-01 # Partial from date
    python backfill_player_stats.py --update            # Incremental (new games only)

Options:
    --start DATE    Start date (YYYY-MM-DD) — defaults to config.data_start_date
    --end DATE      End date (YYYY-MM-DD) — defaults to today
    --update        Incremental mode: only backfill games after the last cached date
    --batch-size N  Games per batch (default: 100)
    --dry-run       Log progress without writing to database

Exit codes:
    0: Success
    1: Error (check logs)
"""

import argparse
import datetime
import logging
import sqlite3
import sys
import time
from pathlib import Path
from typing import Optional

import pandas as pd
from nba_api.stats.endpoints import BoxScoreTraditionalV3
import numpy as np

from src.data_processing.data_loader import NBADataLoader
from src.migrations.migration_create_player_stats_cache import migrate_player_stats_cache
from src.utils.config_loader import load_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
)
logger = logging.getLogger(__name__)

# API throttling to avoid rate limits
SLEEP_SECONDS = 0.6

# Statistics to cache and their column names in box score data
# V3 uses camelCase column names
STAT_COLUMNS = {
    'PPG': 'points',                # Points Per Game
    'AST': 'assists',               # Assists
    'REB': 'reboundsTotal',         # Rebounds (total)
    'BLK': 'blocks',                # Blocks
    'STL': 'steals',                # Steals
    'FG%': 'fieldGoalsPercentage',  # Field Goal Percentage
}


def _get_last_cached_date(db_path: str) -> Optional[str]:
    """Get the most recent game_date in player_stats_cache."""
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT MAX(game_date) FROM player_stats_cache")
        result = cursor.fetchone()
        conn.close()
        return result[0] if result and result[0] else None
    except sqlite3.OperationalError:
        # Table doesn't exist yet
        return None


def _fetch_box_score(game_id: str, max_retries: int = 3) -> pd.DataFrame:
    """
    Fetch player box score data for a game from the NBA API with exponential backoff.

    Args:
        game_id: NBA game ID
        max_retries: Maximum retry attempts on failure

    Returns:
        DataFrame with columns including personId, points, assists, reboundsTotal, blocks, steals, fieldGoalsPercentage
        Returns empty DataFrame if all attempts fail.
    """
    for attempt in range(max_retries):
        try:
            time.sleep(SLEEP_SECONDS)
            box_score = BoxScoreTraditionalV3(game_id=game_id)
            df = box_score.get_data_frames()[0]

            if df.empty:
                logger.warning(f"Empty box score for game {game_id}")
                return pd.DataFrame()

            # Filter to player rows (exclude team totals and DNPs)
            # Keep only rows with valid personId
            df = df[df['personId'].notna()]

            return df
        except Exception as e:
            wait_time = 2 ** attempt  # Exponential backoff: 1s, 2s, 4s
            if attempt < max_retries - 1:
                logger.debug(
                    f"Failed to fetch game {game_id} (attempt {attempt + 1}/{max_retries}): {e}. "
                    f"Retrying in {wait_time}s..."
                )
                time.sleep(wait_time)
            else:
                logger.error(f"Failed to fetch box score for game {game_id} after {max_retries} attempts: {e}")

    return pd.DataFrame()


def _insert_player_stats(
    db_path: str,
    game_date: str,
    game_id: str,
    box_score_df: pd.DataFrame,
    dry_run: bool = False,
) -> int:
    """
    Insert player stats from a box score into the cache table.

    Args:
        db_path: Path to the SQLite database
        game_date: Game date (YYYY-MM-DD)
        game_id: NBA game ID
        box_score_df: Box score data
        dry_run: If True, log but don't write

    Returns:
        Number of stats rows inserted
    """
    if box_score_df.empty:
        return 0

    rows_to_insert = []
    for _, row in box_score_df.iterrows():
        player_id = int(row['personId'])

        # Insert one row per stat for this player
        for stat_name, col_name in STAT_COLUMNS.items():
            try:
                stat_value = float(row[col_name])
                # Skip NaN/None values to avoid NOT NULL constraint violations
                if pd.isna(stat_value):
                    logger.debug(f"Skipping {stat_name} for player {player_id}: NaN value")
                    continue
                rows_to_insert.append((player_id, game_date, stat_name, stat_value))
            except (ValueError, KeyError, TypeError) as e:
                logger.debug(f"Skipping {stat_name} for player {player_id}: {e}")

    if not rows_to_insert:
        return 0

    if dry_run:
        logger.info(f"[DRY RUN] Would insert {len(rows_to_insert)} stats from game {game_id}")
        return len(rows_to_insert)

    conn = sqlite3.connect(db_path)
    try:
        cursor = conn.cursor()
        sql = """
        INSERT OR REPLACE INTO player_stats_cache (player_id, game_date, stat_name, stat_value)
        VALUES (?, ?, ?, ?)
        """
        cursor.executemany(sql, rows_to_insert)
        conn.commit()
        return cursor.rowcount
    finally:
        conn.close()


def backfill_player_stats(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    batch_size: int = 100,
    dry_run: bool = False,
) -> None:
    """
    Backfill player_stats_cache with historical data.

    Args:
        start_date: Start date (YYYY-MM-DD). Defaults to config.data_start_date.
        end_date: End date (YYYY-MM-DD). Defaults to today.
        batch_size: Number of games per batch (for progress logging).
        dry_run: If True, log progress without writing to database.
    """
    config = load_config()
    db_path = config.data_paths.raw_db

    # Default dates from config
    if start_date is None:
        start_date = config.datasets_loading.data_start_date
    if end_date is None:
        end_date = datetime.date.today().isoformat()

    logger.info(f"Backfilling player stats from {start_date} to {end_date}")
    if dry_run:
        logger.info("[DRY RUN MODE] — no data will be written")

    # Create or migrate the cache table
    if not dry_run:
        try:
            migrate_player_stats_cache(db_path)
            logger.info(f"Ensured player_stats_cache table exists in {db_path}")
        except Exception as e:
            logger.error(f"Failed to create cache table: {e}")
            return

    # Load games in the date range
    loader = NBADataLoader(db_path=db_path)
    try:
        games_df = loader.load_games(
            start_date=start_date,
            end_date=end_date,
            allowed_season_types=['Regular Season'],
        )
    finally:
        loader.close()

    if games_df.empty:
        logger.info(f"No games found in {start_date} to {end_date}")
        return

    logger.info(f"Backfilling {len(games_df):,} games")

    # Process games in batches
    total_stats_inserted = 0
    games_processed = 0

    for idx, (_, game_row) in enumerate(games_df.iterrows(), 1):
        game_id = game_row['GAME_ID']
        game_date = game_row['GAME_DATE'].strftime('%Y-%m-%d')

        # Fetch box score from API
        box_score = _fetch_box_score(game_id)

        if not box_score.empty:
            # Insert stats into cache
            stats_inserted = _insert_player_stats(db_path, game_date, game_id, box_score, dry_run)
            total_stats_inserted += stats_inserted
            games_processed += 1

        # Log progress every batch_size games
        if idx % batch_size == 0:
            logger.info(
                f"Progress: {idx:,} / {len(games_df):,} games | {total_stats_inserted:,} stats cached"
            )

    logger.info(
        f"Backfill complete: {games_processed:,} games processed, {total_stats_inserted:,} stats cached"
    )


def backfill_incremental(batch_size: int = 100, dry_run: bool = False) -> None:
    """
    Incremental backfill: populate cache for games after the last cached date.

    This is useful for regularly updating the cache with new games without
    re-processing historical data.

    Args:
        batch_size: Number of games per batch (for progress logging).
        dry_run: If True, log progress without writing to database.
    """
    config = load_config()
    db_path = config.data_paths.raw_db

    # Find the last cached date
    last_date = _get_last_cached_date(db_path)
    if last_date:
        logger.info(f"Last cached date: {last_date}")
        start_date = last_date
    else:
        logger.info("Cache is empty, performing full backfill")
        start_date = config.datasets_loading.data_start_date

    end_date = datetime.date.today().isoformat()
    backfill_player_stats(
        start_date=start_date,
        end_date=end_date,
        batch_size=batch_size,
        dry_run=dry_run,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Backfill player_stats_cache with historical NBA player statistics"
    )
    parser.add_argument(
        "--start",
        type=str,
        default=None,
        help="Start date (YYYY-MM-DD). Defaults to config.data_start_date.",
    )
    parser.add_argument(
        "--end",
        type=str,
        default=None,
        help="End date (YYYY-MM-DD). Defaults to today.",
    )
    parser.add_argument(
        "--update",
        action="store_true",
        help="Incremental mode: backfill only games after the last cached date.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Number of games per batch for progress logging (default: 100).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Log progress without writing to database.",
    )

    args = parser.parse_args()

    try:
        if args.update:
            backfill_incremental(batch_size=args.batch_size, dry_run=args.dry_run)
        else:
            backfill_player_stats(
                start_date=args.start,
                end_date=args.end,
                batch_size=args.batch_size,
                dry_run=args.dry_run,
            )
    except Exception as e:
        logger.error(f"Backfill failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
