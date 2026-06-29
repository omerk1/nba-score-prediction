"""
Player Statistics Cache Query Utility.

This module provides fast lookups for player statistics using the player_stats_cache table.
It computes rolling averages on-demand from cached game-level stats.

Usage:
    from src.utils.player_stats_cache import get_player_projections, get_player_rolling_avg

    # Get all stat projections for a player on a given date
    projections = get_player_projections(player_id=201939, game_date='2024-04-14')
    # Returns: {'PPG': 24.5, 'AST': 5.2, 'REB': 7.1, ...}

    # Get a single stat's rolling average
    ppg = get_player_rolling_avg(player_id=201939, stat_name='PPG', game_date='2024-04-14', window=10)
"""

import logging
import sqlite3
from pathlib import Path
from typing import Optional

from src.utils.config_loader import load_config

logger = logging.getLogger(__name__)


def get_player_rolling_avg(
    db_path: str,
    player_id: int,
    stat_name: str,
    game_date: str,
    window: int = 10,
) -> float:
    """
    Compute rolling average for a player's stat up to a given date.

    Retrieves the player's last `window` games before `game_date` and computes
    the average of `stat_name` across those games.

    Args:
        db_path: Path to the SQLite database
        player_id: NBA player ID
        stat_name: One of 'PPG', 'AST', 'REB', 'BLK', 'STL', 'FG%'
        game_date: Query date (YYYY-MM-DD format, exclusive)
        window: Rolling window size (e.g., 5, 10, 20 games)

    Returns:
        float: Rolling average value. Returns 0.0 if fewer than 1 game found.
    """
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Query the player's games before game_date, ordered descending by date
        # Limit to window size
        query = """
        SELECT stat_value
        FROM player_stats_cache
        WHERE player_id = ?
          AND stat_name = ?
          AND game_date < ?
        ORDER BY game_date DESC
        LIMIT ?
        """
        cursor.execute(query, (player_id, stat_name, game_date, window))
        rows = cursor.fetchall()
        conn.close()

        # Extract values and compute average
        if not rows:
            logger.debug(
                f"No history for player {player_id} stat {stat_name} before {game_date}"
            )
            return 0.0

        values = [row[0] for row in rows]
        avg = sum(values) / len(values)
        return float(avg)

    except Exception as e:
        logger.error(
            f"Error querying rolling average for player {player_id} stat {stat_name}: {e}"
        )
        return 0.0


def get_player_projections(
    db_path: str,
    player_id: int,
    game_date: str,
    windows: Optional[list[int]] = None,
) -> dict[str, float]:
    """
    Get all stat projections for a player on a given date.

    Computes rolling averages for all 6 tracked statistics (PPG, AST, REB, BLK, STL, FG%)
    using the configured rolling windows from config.features.rolling_windows.

    If multiple windows are configured, returns the average of those windows for each stat.

    Args:
        db_path: Path to the SQLite database
        player_id: NBA player ID
        game_date: Query date (YYYY-MM-DD format, exclusive)
        windows: List of rolling window sizes. If None, loads from config.

    Returns:
        dict mapping stat names to projected values:
            {
                'PPG': float,
                'AST': float,
                'REB': float,
                'BLK': float,
                'STL': float,
                'FG%': float
            }
        Returns 0.0 for any stat with no historical data.

    Example:
        >>> projections = get_player_projections(
        ...     db_path="data/raw/nba_api.sqlite",
        ...     player_id=201939,  # LeBron James
        ...     game_date="2024-04-14"
        ... )
        >>> projections['PPG']
        24.5
    """
    if windows is None:
        config = load_config()
        windows = config.features.rolling_windows

    stat_names = ['PPG', 'AST', 'REB', 'BLK', 'STL', 'FG%']
    projections = {}

    for stat in stat_names:
        # Compute rolling averages for each window and average them
        window_values = []
        for window in windows:
            avg = get_player_rolling_avg(db_path, player_id, stat, game_date, window)
            window_values.append(avg)

        # Return the average across all windows (or 0.0 if all windows returned 0.0)
        if window_values:
            projections[stat] = sum(window_values) / len(window_values)
        else:
            projections[stat] = 0.0

    return projections


def ensure_cache_exists(db_path: str) -> None:
    """
    Ensure player_stats_cache table exists in the database.

    If the table doesn't exist, creates it. This is idempotent.

    Args:
        db_path: Path to the SQLite database

    Raises:
        Exception: If unable to create the table
    """
    from src.migrations.migration_create_player_stats_cache import migrate_player_stats_cache

    try:
        migrate_player_stats_cache(db_path)
    except Exception as e:
        logger.error(f"Failed to ensure cache table exists: {e}")
        raise
