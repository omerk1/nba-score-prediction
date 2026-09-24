"""
Recompute the injury_features table (n_out, n_questionable, team_deficit) from
the game-date-corrected listings in player_injuries_dated, instead of the
report-date-keyed player_injuries table the live pipeline still uses.

This is the extrinsic test for docs/PIPELINE_AUDIT.md's 2026-09-17 finding:
`_add_injury_features` joins on equal(game_date), so a game on date G is
currently given the counts from the report published AFTER G tipped off,
mostly describing G+1's game rather than G's. Whether fixing that moves the
composite score is what this table, plus feature_builder.py's
injury_features.use_corrected_dates flag, is for.

Reuses the exact same scoring logic the live pipeline uses --
src.news_scraping.extractors.formula_scorer.compute_team_deficit and
src.news_scraping.pipeline._get_importance_map -- so the only thing that
differs from the live table is which dates the listings are attached to, not
how impact is computed from them. Writes ONLY to data/raw/injury_dates.sqlite
(this worktree's own file); data/raw/injury_features.sqlite (shared with the
main checkout and other worktrees via symlink) is opened read-only for
player_importance and never written.

Per (game_date, team_id, player_name), multiple report_dates can cover the
same game_date (the game's own report, and/or the day-before report that
usually names it -- see rebuild_injury_dates.py's docstring). The row with the
latest report_date is used, since that is the status closest to and still
before tip-off.

Usage: venv/bin/python3 scripts/build_injury_features_dated.py
"""

import logging
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

load_dotenv()  # src.news_scraping.pipeline imports llm_extractor, which reads
# GOOGLE_API_KEY at module load time even though this script never calls it

import pandas as pd

from src.availability.db import get_conn as get_dated_conn
from src.news_scraping.extractors.formula_scorer import compute_team_deficit
from src.news_scraping.pipeline import _get_importance_map, _season_start
from src.utils.config_loader import load_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DATED_DB = "data/raw/injury_dates.sqlite"
SCORER_NAME = "formula_dated"

_SCHEMA_ADDON = """
CREATE TABLE IF NOT EXISTS injury_features_dated (
    game_date      TEXT NOT NULL,
    team_id        INTEGER NOT NULL,
    scorer         TEXT NOT NULL,
    n_out          INTEGER NOT NULL,
    n_questionable INTEGER NOT NULL,
    team_deficit   REAL NOT NULL,
    updated_at     TEXT NOT NULL,
    PRIMARY KEY (game_date, team_id, scorer)
);
"""


def load_latest_per_listing(dated_db: str) -> pd.DataFrame:
    """One row per (game_date, team_id, player_name): the latest report_date's
    status, i.e. the one closest to but still before tip-off."""
    with sqlite3.connect(dated_db) as conn:
        df = pd.read_sql_query(
            """
            SELECT game_date, report_date, team_id, player_name, status, reason
            FROM player_injuries_dated
            """,
            conn,
        )
    df = df.sort_values("report_date")
    return df.drop_duplicates(["game_date", "team_id", "player_name"], keep="last")


def main() -> None:
    cfg = load_config()
    listings = load_latest_per_listing(DATED_DB)
    logger.info(f"{len(listings)} de-duplicated (game_date, team, player) listings")

    conn = get_dated_conn(DATED_DB)
    conn.executescript(_SCHEMA_ADDON)
    conn.execute("DELETE FROM injury_features_dated WHERE scorer = ?", (SCORER_NAME,))

    importance_db = cfg.injury_features.db_path  # shared file, read-only
    season_start_cache: dict[str, str] = {}
    rows_out = []
    n = 0
    for (game_date, team_id), g in listings.groupby(["game_date", "team_id"]):
        if game_date not in season_start_cache:
            season_start_cache[game_date] = _season_start(cfg.data_paths.raw_db, game_date)
        season_start = season_start_cache[game_date]
        importance_map = _get_importance_map(importance_db, int(team_id), game_date, season_start)

        players = [
            {"player_name": r.player_name, "status": r.status, "reason": r.reason}
            for r in g.itertuples(index=False)
        ]
        n_out = sum(1 for p in players if p["status"] == "Out")
        n_q = sum(1 for p in players if p["status"] in ("Questionable", "Day-To-Day"))
        team_deficit = compute_team_deficit(
            players, importance_map, cfg.injury_features.severity_weights, cfg.injury_features.doubtful_weight
        )
        rows_out.append(
            (
                game_date,
                int(team_id),
                SCORER_NAME,
                n_out,
                n_q,
                team_deficit,
                datetime.now(timezone.utc).isoformat(),
            )
        )
        n += 1
        if n % 2000 == 0:
            logger.info(f"{n}/{listings.groupby(['game_date', 'team_id']).ngroups} team-games scored")

    conn.executemany("INSERT OR REPLACE INTO injury_features_dated VALUES (?,?,?,?,?,?,?)", rows_out)
    logger.info(f"wrote {len(rows_out)} rows to {DATED_DB}:injury_features_dated (scorer={SCORER_NAME})")

    check = conn.execute(
        "SELECT COUNT(*), MIN(game_date), MAX(game_date), AVG(team_deficit) FROM injury_features_dated WHERE scorer = ?",
        (SCORER_NAME,),
    ).fetchone()
    logger.info(f"sanity check: {check}")


if __name__ == "__main__":
    main()
