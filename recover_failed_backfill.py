"""
Recover failed games from a backfill run.

Extracts failed game IDs from backfill.log and retries them with exponential backoff.

Usage:
    python recover_failed_backfill.py                    # Use backfill.log
    python recover_failed_backfill.py --logfile other.log # Use different log
    python recover_failed_backfill.py --max-retries 5     # Retry each game up to 5 times
"""

import argparse
import logging
import re
import sys
import time
from pathlib import Path
from typing import Set

import pandas as pd
from nba_api.stats.endpoints import BoxScoreTraditionalV3

from src.data_processing.data_loader import NBADataLoader
from src.utils.config_loader import load_config
from backfill_player_stats import _insert_player_stats, STAT_COLUMNS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
)
logger = logging.getLogger(__name__)

SLEEP_SECONDS = 0.6


def extract_failed_games(logfile: str) -> Set[str]:
    """
    Extract failed game IDs from backfill log.

    Args:
        logfile: Path to backfill.log

    Returns:
        Set of game IDs that failed during backfill
    """
    failed_games = set()
    pattern = re.compile(r'Failed to fetch box score for game (\d+):')

    try:
        with open(logfile, 'r') as f:
            for line in f:
                match = pattern.search(line)
                if match:
                    game_id = match.group(1)
                    failed_games.add(game_id)
    except FileNotFoundError:
        logger.error(f"Log file not found: {logfile}")
        return set()

    return failed_games


def fetch_box_score_with_retry(game_id: str, max_retries: int = 3) -> pd.DataFrame:
    """
    Fetch player box score data with exponential backoff retry.

    Args:
        game_id: NBA game ID
        max_retries: Maximum retry attempts

    Returns:
        DataFrame with player box score data, or empty DataFrame if all retries fail
    """
    for attempt in range(max_retries):
        try:
            time.sleep(SLEEP_SECONDS)
            box_score = BoxScoreTraditionalV3(game_id=game_id)
            df = box_score.get_data_frames()[0]

            if df.empty:
                logger.warning(f"Empty box score for game {game_id}")
                return pd.DataFrame()

            df = df[df['personId'].notna()]
            logger.info(f"✓ Fetched game {game_id} (attempt {attempt + 1}/{max_retries})")
            return df

        except Exception as e:
            wait_time = 2 ** attempt  # Exponential backoff: 1s, 2s, 4s
            if attempt < max_retries - 1:
                logger.warning(
                    f"Failed to fetch game {game_id} (attempt {attempt + 1}/{max_retries}): {e}. "
                    f"Retrying in {wait_time}s..."
                )
                time.sleep(wait_time)
            else:
                logger.error(f"Failed to fetch game {game_id} after {max_retries} attempts: {e}")

    return pd.DataFrame()


def recover_failed_backfill(
    logfile: str = "backfill.log",
    max_retries: int = 3,
    dry_run: bool = False,
) -> None:
    """
    Retry all failed games from a backfill run.

    Args:
        logfile: Path to backfill log
        max_retries: Max retries per game
        dry_run: If True, log but don't write
    """
    # Extract failed games
    failed_games = extract_failed_games(logfile)
    if not failed_games:
        logger.info("No failed games found in log")
        return

    logger.info(f"Found {len(failed_games)} failed games: {sorted(failed_games)}")

    config = load_config()
    db_path = config.data_paths.raw_db

    # Load game data to get dates
    loader = NBADataLoader(db_path=db_path)
    try:
        all_games = loader.load_games(
            start_date="2016-10-01",
            end_date="2026-07-01",
            allowed_season_types=['Regular Season'],
        )
    finally:
        loader.close()

    # Build game_id -> date mapping
    game_dates = dict(zip(all_games['GAME_ID'].astype(str), all_games['GAME_DATE'].dt.strftime('%Y-%m-%d')))

    # Retry each failed game
    recovered = 0
    total_stats = 0

    for game_id in sorted(failed_games):
        if game_id not in game_dates:
            logger.warning(f"Game {game_id} not found in database")
            continue

        game_date = game_dates[game_id]
        logger.info(f"Recovering game {game_id} (date: {game_date})")

        box_score = fetch_box_score_with_retry(game_id, max_retries=max_retries)

        if not box_score.empty:
            stats_inserted = _insert_player_stats(db_path, game_date, game_id, box_score, dry_run)
            if stats_inserted > 0:
                recovered += 1
                total_stats += stats_inserted

    logger.info(f"Recovery complete: {recovered} games recovered, {total_stats} stats cached")


def main():
    parser = argparse.ArgumentParser(description="Recover failed games from backfill run")
    parser.add_argument(
        "--logfile",
        type=str,
        default="backfill.log",
        help="Path to backfill log (default: backfill.log)",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Max retries per game (default: 3)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Log progress without writing to database",
    )

    args = parser.parse_args()

    try:
        recover_failed_backfill(
            logfile=args.logfile,
            max_retries=args.max_retries,
            dry_run=args.dry_run,
        )
    except Exception as e:
        logger.error(f"Recovery failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
