"""
Script to populate and maintain the injury features database.

Steps must run in order — player importance must exist before injury extraction.
All steps are resumable: they use INSERT OR REPLACE, so re-running skips nothing
but safely overwrites any existing rows.

Usage:
    # Run all steps (full backfill from config date range)
    python build_injury_features.py

    # Run a specific step
    python build_injury_features.py --run build_player_importance
    python build_injury_features.py --run backfill_historical_injuries
    python build_injury_features.py --run nightly_update

    # Test on a small date range before committing to the full backfill
    python build_injury_features.py --run backfill_historical_injuries --start 2023-01-01 --end 2023-01-14

    # Resume a partial backfill (just move --start forward to where you left off)
    python build_injury_features.py --run backfill_historical_injuries --start 2021-03-01

Scoring (controlled by injury_features.scorer in config.yaml):
    formula — deterministic weighted sum, no API calls, full backfill completes in minutes
    llm     — Gemini call per team per date; free tier: ~14 days for full backfill ($1-2 on paid tier)
"""

import argparse
import logging
from datetime import date

from dotenv import load_dotenv

from src.utils.config_loader import load_config

load_dotenv()  # reads GOOGLE_API_KEY (and others) from .env before any module uses them

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _season_year(d: date) -> int:
    """Starting year of the NBA season containing this date (season starts in October)."""
    return d.year if d.month >= 10 else d.year - 1


def _season_str(year: int) -> str:
    return f"{year}-{str(year + 1)[-2:]}"


def _resolve_seasons_and_dates() -> tuple[list[str], date, date]:
    cfg = load_config()
    dl = cfg.datasets_loading
    start = date.fromisoformat(dl.train_start_date)
    end = date.fromisoformat(dl.test_end_date)
    seasons = [
        _season_str(y)
        for y in range(_season_year(start), _season_year(end) + 1)
    ]
    return seasons, start, end


def build_player_importance():
    """
    Build weekly importance snapshots for every season in the config date range.
    Hits nba_api twice per snapshot (base + advanced stats) with a 1s sleep each.
    Runtime: ~10 minutes for 8 seasons. Safe to re-run.
    """
    from src.news_scraping.player_importance import backfill_season

    seasons, _, _ = _resolve_seasons_and_dates()
    logger.info(f"Building player importance for seasons: {seasons}")
    for season in seasons:
        logger.info(f"  Season {season}")
        backfill_season(season)
    logger.info("build_player_importance complete")


def backfill_historical_injuries(start: date | None = None, end: date | None = None):
    """
    Scrape injury reports for each game date and compute impact scores.

    Scoring method is set by injury_features.scorer in config.yaml.
    Resumable: already-processed dates are overwritten safely (INSERT OR REPLACE).
    Progress is visible in the logs (one line per team per date).

    Segmented by season internally: transactions are fetched once per season
    (~8 HTTP requests), then replayed in memory for each date — not re-fetched per date.

    Full backfill with formula scorer: minutes (no API calls beyond scraping).
    Full backfill with llm scorer: ~14 days free tier, ~30 min on paid tier.
    """
    from src.news_scraping.pipeline import run_historical

    _, config_start, config_end = _resolve_seasons_and_dates()
    start = start or config_start
    end = end or config_end
    logger.info(f"Backfilling injury features {start} → {end}")
    run_historical(start, end)
    logger.info("backfill_historical_injuries complete")


def nightly_update():
    """
    Fetch today's ESPN injury report and extract impact scores.
    Schedule this daily (e.g. cron at 11:00 AM ET on game days).
    """
    from src.news_scraping.pipeline import run_nightly

    logger.info("Running nightly injury update")
    run_nightly()
    logger.info("nightly_update complete")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build and maintain the injury features database.")
    parser.add_argument(
        "--run",
        choices=["build_player_importance", "backfill_historical_injuries", "nightly_update"],
        help="Run a specific step only (default: run all in order)",
    )
    parser.add_argument(
        "--start",
        type=date.fromisoformat,
        metavar="YYYY-MM-DD",
        help="Override start date for backfill_historical_injuries",
    )
    parser.add_argument(
        "--end",
        type=date.fromisoformat,
        metavar="YYYY-MM-DD",
        help="Override end date for backfill_historical_injuries",
    )
    args = parser.parse_args()

    if args.run == "backfill_historical_injuries" or args.start or args.end:
        backfill_historical_injuries(start=args.start, end=args.end)
    elif args.run == "build_player_importance":
        build_player_importance()
    elif args.run == "nightly_update":
        nightly_update()
    else:
        build_player_importance()
        backfill_historical_injuries()
        nightly_update()
