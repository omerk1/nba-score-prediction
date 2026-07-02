"""
Retry failed Polymarket markets from a previous backfill run.

Reads polymarket_failed_markets.csv and retries each failed market
with exponential backoff.

Usage:
    python src/data_processing/recover_polymarket_failed.py \
        --failed-csv outputs/polymarket_failed_markets.csv \
        --db-path data/raw/nba_api.sqlite
"""

import argparse
import logging
import pandas as pd
import json
import time
from pathlib import Path

from src.data_processing.polymarket_collector import (
    find_nba_markets,
    fetch_price_history,
    infer_market_type,
    extract_teams_from_question,
    lookup_game_id,
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


def retry_failed_markets(
    failed_csv: str,
    db_path: str,
    output_csv: str = "outputs/polymarket_odds_recovered.csv",
    max_retries: int = 3,
    rate_limit_delay: float = 0.3
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Retry failed markets with exponential backoff.

    Args:
        failed_csv: Path to polymarket_failed_markets.csv
        db_path: Path to nba_api.sqlite
        output_csv: Where to save recovered odds
        max_retries: Max retry attempts per market
        rate_limit_delay: Seconds between API calls

    Returns:
        Tuple of (recovered_odds_df, still_failed_df)
    """
    logger.info(f"Reading failed markets from {failed_csv}")

    # Load failed markets
    failed_df = pd.read_csv(failed_csv)
    logger.info(f"Attempting to recover {len(failed_df)} failed markets")

    recovered = []
    still_failed = []

    for idx, row in failed_df.iterrows():
        game_date = row["game_date"]
        slug = row["slug"]
        question = row["question"]
        original_error = row["error"]

        logger.info(f"Retrying {idx+1}/{len(failed_df)}: {slug}")

        # Re-fetch the market from API (in case it's now available)
        retry_count = 0
        success = False

        while retry_count < max_retries and not success:
            try:
                # Search for the event again
                # Extract year from slug or game_date
                year = game_date.split("-")[0]
                events = find_nba_markets(game_date, game_date)

                # Find matching market
                for event in events:
                    if event.get("slug") == slug:
                        for market in event.get("markets", []):
                            if market.get("question", "").startswith(question):
                                # Found it! Try to fetch price history
                                clobTokenIds_str = market.get("clobTokenIds", "[]")
                                token_ids = json.loads(clobTokenIds_str)

                                if token_ids:
                                    prices = fetch_price_history(token_ids[0])
                                    if prices:
                                        # Success!
                                        market_type = infer_market_type(market.get("question", ""))
                                        home_team, away_team = extract_teams_from_question(
                                            market.get("question", "")
                                        )
                                        game_id = lookup_game_id(db_path, game_date, home_team, away_team)

                                        recovered_row = {
                                            "game_id": game_id,
                                            "game_date": game_date,
                                            "home_team": home_team,
                                            "away_team": away_team,
                                            "market_type": market_type,
                                            "opening_price": prices.get("opening_price"),
                                            "closing_price": prices.get("closing_price"),
                                            "volume": market.get("volume"),
                                        }
                                        recovered.append(recovered_row)
                                        success = True
                                        logger.info(f"  ✓ Recovered on attempt {retry_count + 1}")
                                        break

                if not success:
                    retry_count += 1
                    if retry_count < max_retries:
                        wait = 2 ** retry_count  # 2s, 4s, 8s
                        logger.debug(f"  Retrying in {wait}s...")
                        time.sleep(wait)
                else:
                    break

            except Exception as e:
                retry_count += 1
                if retry_count < max_retries:
                    wait = 2 ** retry_count
                    logger.debug(f"  Error: {e}, retrying in {wait}s...")
                    time.sleep(wait)
                else:
                    logger.warning(f"  Failed after {max_retries} attempts: {e}")

        if not success:
            still_failed.append({
                "game_date": game_date,
                "slug": slug,
                "question": question,
                "original_error": original_error,
                "retry_error": "Unable to recover"
            })

        time.sleep(rate_limit_delay)

    # Save results
    df_recovered = pd.DataFrame(recovered)
    df_still_failed = pd.DataFrame(still_failed)

    if len(df_recovered) > 0:
        df_recovered.to_csv(output_csv, index=False)
        logger.info(f"Saved {len(df_recovered)} recovered odds to {output_csv}")

    if len(df_still_failed) > 0:
        still_failed_csv = output_csv.replace(".csv", "_still_failed.csv")
        df_still_failed.to_csv(still_failed_csv, index=False)
        logger.warning(f"{len(df_still_failed)} markets still failed (see {still_failed_csv})")

    logger.info(f"Recovery complete: {len(df_recovered)} recovered, {len(df_still_failed)} still failed")

    return df_recovered, df_still_failed


def main():
    parser = argparse.ArgumentParser(description="Recover failed Polymarket markets")
    parser.add_argument("--failed-csv", required=True, help="Path to polymarket_failed_markets.csv")
    parser.add_argument("--db-path", required=True, help="Path to nba_api.sqlite")
    parser.add_argument("--output", default="outputs/polymarket_odds_recovered.csv", help="Output CSV")
    parser.add_argument("--max-retries", type=int, default=3, help="Max retry attempts")

    args = parser.parse_args()

    retry_failed_markets(
        failed_csv=args.failed_csv,
        db_path=args.db_path,
        output_csv=args.output,
        max_retries=args.max_retries
    )


if __name__ == "__main__":
    main()
