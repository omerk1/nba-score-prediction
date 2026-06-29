"""
Migration: Create player_stats_cache table.

This migration creates a table to cache player-level game statistics,
enabling fast lookups for rolling average computations without requiring
API calls.

Usage:
    from src.migrations.001_create_player_stats_cache import migrate_player_stats_cache
    migrate_player_stats_cache(db_path)
"""

import logging
import sqlite3
from pathlib import Path

logger = logging.getLogger(__name__)


def migrate_player_stats_cache(db_path: str | Path) -> None:
    """
    Create player_stats_cache table if it doesn't exist.

    This is idempotent — running it multiple times is safe.

    Args:
        db_path: Path to the SQLite database

    Raises:
        FileNotFoundError: If database file doesn't exist
    """
    db_path = Path(db_path)
    if not db_path.exists():
        raise FileNotFoundError(f"Database not found at {db_path}")

    conn = sqlite3.connect(db_path)
    try:
        cursor = conn.cursor()

        # Check if table already exists
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='player_stats_cache'"
        )
        if cursor.fetchone():
            logger.info("player_stats_cache table already exists, skipping creation")
            return

        # Create the table
        sql = """
        CREATE TABLE player_stats_cache (
            player_id INTEGER NOT NULL,
            game_date TEXT NOT NULL,
            stat_name TEXT NOT NULL,
            stat_value REAL NOT NULL,
            PRIMARY KEY (player_id, game_date, stat_name)
        )
        """
        cursor.execute(sql)

        # Create indexes for efficient querying
        # Index for player_id, game_date (most common query pattern)
        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_player_stats_player_date
            ON player_stats_cache(player_id, game_date DESC)
            """
        )

        # Index for game_date (useful for backfill operations)
        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_player_stats_date
            ON player_stats_cache(game_date DESC)
            """
        )

        conn.commit()
        logger.info("Created player_stats_cache table and indexes")

    finally:
        conn.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m src.migrations.001_create_player_stats_cache <db_path>")
        sys.exit(1)

    db_path = sys.argv[1]
    migrate_player_stats_cache(db_path)
    print(f"Migration complete: {db_path}")
