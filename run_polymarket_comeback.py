"""
CLI entry point for the Polymarket comeback-analysis pipeline.

Standalone data-science pipeline - independent of the model's feature
engineering (feature_builder.py / train_model.py / predict_game.py) and of
the earlier A5 pre-game-odds collector (src/data_processing/polymarket_collector.py).

Usage:
    python run_polymarket_comeback.py --slugs nba-orl-chi-2026-04-10 nba-mem-uta-2026-04-10 ...

IMPORTANT (owner instruction): this is deliberately run on a small, explicit
list of game slugs (~5 games from one recent week) as a checkpoint before any
larger-scale run. Do not pass a large slug list without the owner's go-ahead.
"""

import argparse
import logging
import sys

sys.path.insert(0, "src")

from polymarket_comeback.pipeline import run_pipeline  # noqa: E402

RAW_DIR = "data/polymarket_comeback/raw"
GAMES_CSV = "data/polymarket_comeback/games.csv"
SERIES_DIR = "data/polymarket_comeback/series"


def main():
    parser = argparse.ArgumentParser(description="Run the Polymarket comeback-analysis pipeline on a list of game slugs")
    parser.add_argument("--slugs", nargs="+", required=True, help="Gamma event slugs, e.g. nba-lal-bos-2026-01-15")
    parser.add_argument("--force-refresh", action="store_true", help="Ignore caches and re-fetch everything")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    df = run_pipeline(
        slugs=args.slugs,
        raw_dir=RAW_DIR,
        games_csv_path=GAMES_CSV,
        series_dir=SERIES_DIR,
        force_refresh=args.force_refresh,
    )
    print(f"\nProcessed {len(df)} games -> {GAMES_CSV}")
    print(df[["slug", "winner_team", "data_quality", "in_game_min_price_winner_robust"]].to_string(index=False))


if __name__ == "__main__":
    main()
