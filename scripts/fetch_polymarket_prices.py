"""
CLI entry point for the Polymarket price-history pipeline.

Standalone data-science pipeline - independent of the model's feature
engineering (feature_builder.py / train_model.py / predict_game.py) and of
the earlier A5 pre-game-odds collector (src/data_processing/polymarket_collector.py).
Collects general per-game win-probability time series; comeback analysis
(analyze_polymarket_comebacks.py) is one use of this data, not the only one.

Usage:
    python scripts/fetch_polymarket_prices.py --slugs nba-orl-chi-2026-04-10 nba-mem-uta-2026-04-10 ...

    # Or discover + fetch an entire date range (e.g. a full season) via the
    # Gamma /events listing (discovery.py), instead of listing slugs by hand:
    python scripts/fetch_polymarket_prices.py --start-date 2025-09-01 --end-date 2026-08-01

Scale note: initial checkpoints (5-game, then 50-game) validated timing/
storage/coverage before any full-season run was authorized -- see
docs/polymarket_trades_api_step1_findings.md and the games.csv history for
that process. Full-season/date-range runs are expected usage now.
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from polymarket_prices.discovery import discover_nba_game_slugs  # noqa: E402
from polymarket_prices.pipeline import run_pipeline  # noqa: E402

RAW_DIR = "data/polymarket_prices/raw"
GAMES_CSV = "data/polymarket_prices/games.csv"
SERIES_DIR = "data/polymarket_prices/series"


def main():
    parser = argparse.ArgumentParser(description="Fetch Polymarket in-game price history for a list of game slugs")
    parser.add_argument("--slugs", nargs="+", help="Gamma event slugs, e.g. nba-lal-bos-2026-01-15")
    parser.add_argument("--start-date", help="Discover + fetch all resolved NBA games with startDate >= this (YYYY-MM-DD)")
    parser.add_argument("--end-date", help="Discover + fetch all resolved NBA games with startDate <= this (YYYY-MM-DD)")
    parser.add_argument("--force-refresh", action="store_true", help="Ignore caches and re-fetch everything")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if args.slugs and (args.start_date or args.end_date):
        parser.error("--slugs and --start-date/--end-date are mutually exclusive")
    if args.slugs:
        slugs = args.slugs
    elif args.start_date or args.end_date:
        slugs = discover_nba_game_slugs(
            start_date_min=f"{args.start_date}T00:00:00Z" if args.start_date else None,
            start_date_max=f"{args.end_date}T00:00:00Z" if args.end_date else None,
            ascending=True,
        )
        print(f"Discovered {len(slugs)} game slugs in range")
    else:
        parser.error("must pass either --slugs or --start-date/--end-date")

    df = run_pipeline(
        slugs=slugs,
        raw_dir=RAW_DIR,
        games_csv_path=GAMES_CSV,
        series_dir=SERIES_DIR,
        force_refresh=args.force_refresh,
    )
    print(f"\nProcessed {len(df)} games -> {GAMES_CSV}")
    print(df[["slug", "winner_team", "data_quality", "in_game_min_price_winner_robust"]].to_string(index=False))


if __name__ == "__main__":
    main()
