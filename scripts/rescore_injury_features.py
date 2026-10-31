"""
Re-score injury_features from the stored player_injuries listings (no PDF
re-download), writing to a SEPARATE sqlite so the live table the daily job reads
is never touched.

Purpose: ablate injury_features.importance_prior_stats_fallback. Run once with
--fallback off (must reproduce the live 'formula' rows exactly -- the
correctness check) and once with --fallback on (the treatment). Point
training at the output with injury_features.features_db_path.

Usage:
    venv/bin/python3 scripts/rescore_injury_features.py --fallback off --out <path>
    venv/bin/python3 scripts/rescore_injury_features.py --fallback on  --out <path>
"""

import argparse
import logging
import shutil
import sqlite3
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

load_dotenv()  # pipeline imports the LLM extractor, which reads GOOGLE_API_KEY at import

import os

os.environ.setdefault("GOOGLE_API_KEY", "unused-placeholder")

from src.news_scraping import pipeline
from src.news_scraping.extractors.formula_scorer import score_team
from src.utils.config_loader import load_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

SCORER = "formula"
SCHEMA = """
CREATE TABLE IF NOT EXISTS injury_features (
    game_date TEXT NOT NULL, team_id INTEGER NOT NULL, scorer TEXT NOT NULL,
    n_out INTEGER NOT NULL, n_questionable INTEGER NOT NULL,
    team_deficit REAL NOT NULL, updated_at TEXT NOT NULL,
    PRIMARY KEY (game_date, team_id, scorer)
);
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fallback", choices=["on", "off"], required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    cfg = load_config()
    cfg.injury_features.importance_prior_stats_fallback = args.fallback == "on"
    pipeline.load_config = lambda: cfg

    # Work on a copy with a player_name index: the fallback looks players up by
    # name and the live file has no such index.
    tmpdir = tempfile.mkdtemp()
    src_db = str(Path(tmpdir) / "importance.sqlite")
    shutil.copy(cfg.injury_features.db_path, src_db)
    with sqlite3.connect(src_db) as conn:
        conn.execute("CREATE INDEX IF NOT EXISTS ix_pi_name ON player_importance(player_name, as_of_date)")
        conn.execute("CREATE INDEX IF NOT EXISTS ix_pi_team ON player_importance(team_id, as_of_date)")
        listings = conn.execute(
            "SELECT game_date, team_id, player_name, status, reason FROM player_injuries "
            "ORDER BY game_date, team_id"
        ).fetchall()

    groups: dict[tuple[str, int], list[dict]] = {}
    for gd, tid, name, status, reason in listings:
        groups.setdefault((gd, tid), []).append({"player_name": name, "status": status, "reason": reason})

    season_start_cache: dict[str, str] = {}
    now = datetime.now(timezone.utc).isoformat()
    out_rows = []
    for i, ((gd, tid), players) in enumerate(groups.items(), 1):
        if gd not in season_start_cache:
            season_start_cache[gd] = pipeline._season_start(cfg.data_paths.raw_db, gd)
        imap = pipeline._get_importance_map(
            src_db, tid, gd, season_start_cache[gd], [p["player_name"] for p in players]
        )
        r = score_team(players, imap, cfg.injury_features.severity_weights, cfg.injury_features.doubtful_weight)
        out_rows.append((gd, tid, SCORER, r["n_out"], r["n_questionable"], r["team_deficit"], now))
        if i % 3000 == 0:
            logger.info(f"{i}/{len(groups)} team-dates scored")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(args.out) as conn:
        conn.executescript(SCHEMA)
        conn.execute("DELETE FROM injury_features WHERE scorer = ?", (SCORER,))
        conn.executemany("INSERT INTO injury_features VALUES (?,?,?,?,?,?,?)", out_rows)
    shutil.rmtree(tmpdir, ignore_errors=True)
    logger.info(f"wrote {len(out_rows)} rows to {args.out} (fallback={args.fallback})")


if __name__ == "__main__":
    main()
