"""
Migration: Create player_on_off_splits table.

Lives in its own additive database (default: data/raw/player_on_off_splits.sqlite,
see configs/config.yaml's on_off_splits.db_path), NOT inside the shared
data/raw/nba_api.sqlite -- this data is fully reconstructable from nba_api's
TeamPlayerOnOffSummary endpoint (see docs/on_off_splits_decisions.md), so keeping it
in its own file avoids any risk of writing into the shared/symlinked core DB
(same rationale src/matchups/db.py's docstring gives for style_fingerprint_cache.sqlite
being separate from nba_api.sqlite).

Row granularity: one row per (player_id, team_id, split_type, opponent_team_id,
as_of_date checkpoint) -- NOT per game_id. TeamPlayerOnOffSummary returns a
season-cumulative-to-date number for whatever DateTo cutoff is passed, so a row here
represents "this player's on/off split, as of this checkpoint date," not a single
game. See the decisions doc's granularity/backfill-scope sections for why a
checkpoint cadence (not per-game) was chosen.

Usage:
    from src.migrations.migration_create_player_on_off_splits import migrate_player_on_off_splits
    migrate_player_on_off_splits(db_path)
"""

import logging
import sqlite3
from pathlib import Path

logger = logging.getLogger(__name__)


def migrate_player_on_off_splits(db_path: str | Path) -> None:
    """
    Create player_on_off_splits table (and its indexes) if it doesn't exist.

    Idempotent -- running it multiple times is safe. Unlike
    migration_create_player_stats_cache.py, this does NOT require the target
    database to already exist -- this is a brand-new additive cache file, so it's
    created here if missing (parent directory too).

    Args:
        db_path: Path to the SQLite database (created if missing).
    """
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(db_path)
    try:
        cursor = conn.cursor()

        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='player_on_off_splits'"
        )
        if cursor.fetchone():
            logger.info("player_on_off_splits table already exists, skipping creation")
        else:
            cursor.execute(
                """
                CREATE TABLE player_on_off_splits (
                    player_id         INTEGER NOT NULL,
                    player_name       TEXT,
                    team_id           INTEGER NOT NULL,
                    split_type        TEXT NOT NULL,   -- 'overall' | 'home' | 'away' | 'vs_opponent'
                    opponent_team_id  INTEGER,          -- NULL unless split_type='vs_opponent'
                    as_of_date        TEXT NOT NULL,    -- the DateTo cutoff used for the fetch
                    season            TEXT NOT NULL,
                    gp_on REAL, gp_off REAL,
                    min_on REAL, min_off REAL,
                    plus_minus_on REAL, plus_minus_off REAL,
                    net_rating_on REAL, net_rating_off REAL,
                    on_off_plus_minus REAL,   -- plus_minus_on - plus_minus_off
                    on_off_net_rating REAL,   -- net_rating_on  - net_rating_off
                    fetched_at        TEXT NOT NULL,
                    PRIMARY KEY (player_id, team_id, split_type, opponent_team_id, as_of_date)
                )
                """
            )
            # Index for feature_builder.py's merge_asof-style lookups: for a given
            # (team_id, split_type[, opponent_team_id]), find the most recent
            # as_of_date <= target game_date.
            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_on_off_team_split_date
                ON player_on_off_splits(team_id, split_type, as_of_date DESC)
                """
            )
            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_on_off_player_date
                ON player_on_off_splits(player_id, as_of_date DESC)
                """
            )
            conn.commit()
            logger.info(f"Created player_on_off_splits table and indexes at {db_path}")
    finally:
        conn.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m src.migrations.migration_create_player_on_off_splits <db_path>")
        sys.exit(1)

    migrate_player_on_off_splits(sys.argv[1])
    print(f"Migration complete: {sys.argv[1]}")
