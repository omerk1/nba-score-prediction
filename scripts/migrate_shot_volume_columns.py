"""
One-time migration: backfill shot-volume columns (makes/attempts) into an existing
`game` table that predates them.

Background (A8 follow-up): feature_builder.py's rolling FG_PCT/FT_PCT/FG3_PCT features
used to average per-game *percentage* columns directly, which is statistically wrong
(a low-attempt outlier game swings the average as much as a normal-volume game). The
fix requires volume-weighted rolling percentages: sum(makes) / sum(attempts) over the
window. That needs fgm/fga/fg3m/fg3a/ftm/fta columns, which fetch_data.py's `game`
table didn't store even though the nba_api LeagueGameLog response includes them.

This script:
  1. ALTERs the existing `game` table to add the new columns (skips any that already
     exist — safe to re-run).
  2. Re-fetches every configured season/season_type via LeagueGameLog (the same
     machinery fetch_data.py uses for a full backfill — ~10 seasons x 2 season_types,
     not a per-game call) and UPDATEs the new columns into the already-present rows,
     matched by game_id. This does NOT touch any other column and does NOT insert rows.

Usage:
    python scripts/migrate_shot_volume_columns.py                 # run for real
    python scripts/migrate_shot_volume_columns.py --dry-run        # log only, no writes
    python scripts/migrate_shot_volume_columns.py --skip-backup    # DB already backed up
"""

import argparse
import logging
import shutil
import sqlite3
import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.data_processing.fetch_data import (
    DB_PATH,
    SEASON_TYPES,
    SHOT_VOLUME_COLUMNS,
    SLEEP_SECONDS,
    _date_to_season,
    _fetch_season,
    _season_list,
)
from src.utils.config_loader import load_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _existing_columns(conn: sqlite3.Connection) -> set[str]:
    return {row[1] for row in conn.execute("PRAGMA table_info(game)").fetchall()}


def _migrate_schema(conn: sqlite3.Connection, dry_run: bool) -> list[str]:
    """ADD COLUMN for any shot-volume column not already present. Returns columns added."""
    present = _existing_columns(conn)
    added = []
    for col, sqltype in SHOT_VOLUME_COLUMNS:
        if col in present:
            logger.info(f"  Column already present, skipping: {col}")
            continue
        logger.info(f"  ALTER TABLE game ADD COLUMN {col} {sqltype}")
        if not dry_run:
            conn.execute(f"ALTER TABLE game ADD COLUMN {col} {sqltype}")
        added.append(col)
    if not dry_run:
        conn.commit()
    return added


def _update_shot_volume(conn: sqlite3.Connection, df, dry_run: bool) -> int:
    """UPDATE existing rows' shot-volume columns by game_id. Does not insert new rows."""
    cols = [
        "fgm_home", "fga_home", "fg3m_home", "fg3a_home", "ftm_home", "fta_home",
        "fgm_away", "fga_away", "fg3m_away", "fg3a_away", "ftm_away", "fta_away",
    ]
    set_clause = ", ".join(f"{c} = ?" for c in cols)
    sql = f"UPDATE game SET {set_clause} WHERE game_id = ?"

    game_ids = df["game_id"].tolist()
    if dry_run:
        # Count how many of these game_ids actually exist in the DB, so the dry-run
        # estimate matches what a real run's cursor.rowcount would report — the fetched
        # season may include games not present in this (possibly partial) DB.
        placeholders = ",".join("?" * len(game_ids))
        return conn.execute(
            f"SELECT COUNT(*) FROM game WHERE game_id IN ({placeholders})", game_ids
        ).fetchone()[0]

    rows = [tuple(r[c] for c in cols) + (r["game_id"],) for _, r in df.iterrows()]
    cursor = conn.executemany(sql, rows)
    conn.commit()
    return cursor.rowcount


def migrate(db_path: Path = DB_PATH, dry_run: bool = False, skip_backup: bool = False) -> None:
    if not db_path.exists():
        raise FileNotFoundError(f"Database not found at {db_path}")

    if not skip_backup and not dry_run:
        backup_path = db_path.with_name(db_path.name + ".bak-a8")
        logger.info(f"Backing up {db_path} -> {backup_path}")
        shutil.copy2(db_path, backup_path)

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    before_count, before_min, before_max = conn.execute(
        "SELECT COUNT(*), MIN(game_date), MAX(game_date) FROM game"
    ).fetchone()
    logger.info(f"Before migration: {before_count:,} rows ({before_min} - {before_max})")

    logger.info("Step 1: schema migration (ALTER TABLE)")
    added = _migrate_schema(conn, dry_run)
    logger.info(f"  Columns added: {added or '(none — already migrated)'}")

    cfg = load_config()
    start_season = _date_to_season(cfg.datasets_loading.data_start_date)
    seasons = _season_list(start_season)
    logger.info(f"Step 2: re-fetching {len(seasons)} seasons x {len(SEASON_TYPES)} season types "
                f"to backfill shot-volume data for existing rows")

    total_updated = 0
    for season in seasons:
        for season_type in SEASON_TYPES:
            logger.info(f"Fetching {season} {season_type}...")
            try:
                df = _fetch_season(season, season_type)
            except Exception as e:
                logger.error(f"  Error fetching {season} {season_type}: {e}")
                time.sleep(SLEEP_SECONDS)
                continue

            if df.empty:
                logger.info("  No data, skipping.")
            else:
                updated = _update_shot_volume(conn, df, dry_run)
                logger.info(f"  {updated} rows updated ({len(df)} fetched).")
                total_updated += updated
            time.sleep(SLEEP_SECONDS)

    after_count, after_min, after_max = conn.execute(
        "SELECT COUNT(*), MIN(game_date), MAX(game_date) FROM game"
    ).fetchone()
    non_null = conn.execute(
        "SELECT COUNT(*) FROM game WHERE fgm_home IS NOT NULL"
    ).fetchone()[0] if not dry_run else None

    conn.close()

    logger.info(f"After migration: {after_count:,} rows ({after_min} - {after_max})")
    if non_null is not None:
        logger.info(f"Rows with fgm_home populated: {non_null:,} / {after_count:,}")
    logger.info(f"Done. Total row-updates applied: {total_updated:,}")


def main():
    parser = argparse.ArgumentParser(
        description="One-time migration: backfill shot-volume (makes/attempts) columns into the game table"
    )
    parser.add_argument("--db-path", type=str, default=None, help="Override DB path (default: fetch_data.DB_PATH)")
    parser.add_argument("--dry-run", action="store_true", help="Log planned changes without writing")
    parser.add_argument("--skip-backup", action="store_true", help="Skip the pre-migration .bak-a8 copy")
    args = parser.parse_args()

    db_path = Path(args.db_path) if args.db_path else DB_PATH

    try:
        migrate(db_path=db_path, dry_run=args.dry_run, skip_backup=args.skip_backup)
    except Exception as e:
        logger.error(f"Migration failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
