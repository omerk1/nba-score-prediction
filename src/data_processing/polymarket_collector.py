"""
Polymarket odds collector for NBA games.

Fetches historical betting odds from Polymarket for backtest analysis.
Enriches with game_id from nba_api.sqlite for easy joining with game data.

⚠️  COVERAGE LIMITATION:
Polymarket creates markets only for high-volume, high-interest games:
  - Playoffs and championships: ✓ Full depth available
  - Regular season games: ✗ Very limited (not economically viable for Polymarket)

Expected coverage: ~30-50 games per season (playoffs/finals only)

For full-season backtesting (1,230 games/year), use A6 (OddsPapi sportsbook data).
This module is best for playoff-focused prediction models.

No model integration — data collection only.
"""

import logging
import sqlite3
import requests
import json
import time
import re
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Tuple
import pandas as pd

logger = logging.getLogger(__name__)

GAMMA_BASE = "https://gamma-api.polymarket.com"
CLOB_BASE = "https://clob.polymarket.com"
REQUEST_DELAY = 0.5  # seconds between API calls


def find_nba_markets(start_date: str, end_date: str) -> List[Dict]:
    """
    Search Gamma API for NBA game markets in date range.

    Args:
        start_date: YYYY-MM-DD
        end_date: YYYY-MM-DD

    Returns:
        List of dicts with: {slug, game_date, question, clobTokenIds, markets}
    """
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")

    events_by_id = {}
    current = start

    while current <= end:
        date_str = current.strftime("%Y-%m-%d")
        query = f"NBA {current.strftime('%Y-%m-%d')}"

        try:
            logger.debug(f"Searching for markets on {date_str}")
            resp = requests.get(
                f"{GAMMA_BASE}/public-search",
                params={"q": query, "limit": 100},
                timeout=10
            )
            resp.raise_for_status()

            data = resp.json()
            for event in data.get("events", []):
                event_id = event.get("id")
                if event_id and event.get("markets"):
                    events_by_id[event_id] = event

            time.sleep(REQUEST_DELAY)

        except Exception as e:
            logger.warning(f"Failed to search markets for {date_str}: {e}")

        current += timedelta(days=1)

    # Format results
    results = []
    for event in events_by_id.values():
        try:
            # Extract game date from event
            created = event.get("createdAt", "").split("T")[0]
            slug = event.get("slug", "")

            results.append({
                "slug": slug,
                "game_date": created,
                "title": event.get("title", ""),
                "markets": event.get("markets", [])
            })
        except Exception as e:
            logger.debug(f"Failed to parse event: {e}")

    logger.info(f"Found {len(results)} game events in date range")
    return results


def fetch_price_history(token_id: str, fidelity: int = 720) -> Optional[Dict]:
    """
    Fetch price history for a market token from CLOB API.

    CRITICAL: For closed/resolved markets (past games), Polymarket's API
    silently returns empty list unless fidelity is specified. This function
    implements fallback logic: retry with progressively coarser fidelity
    (720min → 1440min → 4320min = 12h → 1d → 3d) to recover historical data.

    Args:
        token_id: Polymarket token ID
        fidelity: Bucket size in minutes (default 720 = 12 hours)

    Returns:
        Dict with opening_price, closing_price, timestamp, or None if failed
    """
    if not token_id:
        return None

    # Try with progressively coarser fidelity
    fidelities_to_try = [fidelity, 1440, 4320]  # 12h, 1d, 3d

    for current_fidelity in fidelities_to_try:
        try:
            resp = requests.get(
                f"{CLOB_BASE}/prices-history",
                params={
                    "market": token_id,
                    "interval": "max",
                    "fidelity": current_fidelity
                },
                timeout=10
            )
            resp.raise_for_status()

            history = resp.json().get("history", [])
            if history:
                logger.debug(f"Found price history with fidelity={current_fidelity}: {len(history)} points")
                return {
                    "opening_price": float(history[0].get("p", 0)),
                    "closing_price": float(history[-1].get("p", 0)),
                    "timestamp": int(history[-1].get("t", 0)),
                    "num_points": len(history)
                }

        except Exception as e:
            logger.debug(f"Failed with fidelity={current_fidelity}: {e}")
            continue

    logger.debug(f"No price history found for token {token_id} after trying all fidelities")
    return None


def infer_market_type(question: str) -> str:
    """
    Infer market type from question string.

    Returns: 'spread' | 'moneyline' | 'totals' | 'other'
    """
    q = question.lower()

    # Totals: check for over/under (not "cover")
    if re.search(r'\b(over|under|total)\b', q) or "o/u" in q:
        return "totals"

    # Spread: contains spread keyword or +/- with number (excluding YYYY-MM-DD dates)
    if "spread" in q or "cover" in q:
        return "spread"
    # Match spread patterns like "+3.5" or "-3.5" but not date parts like "-04"
    if re.search(r'[+-](?!0[0-9])[0-9]{1,2}\.?[0-9]*', q):
        return "spread"

    # Default: moneyline
    return "moneyline"


def extract_teams_from_question(question: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Extract home and away team names/abbreviations from market question.

    Returns: (home_team, away_team) or (None, None) if parsing fails
    """
    # Pattern: "Team A vs Team B" with optional NBA prefix and date
    # Tries patterns from most specific to least specific
    patterns = [
        r"^NBA:?\s+([A-Za-z\s]+?)\s+vs\.?\s+([A-Za-z\s]+?)(?:\s+\d{4}-|$)",  # NBA: Lakers vs Celtics 2024-
        r"^([A-Za-z\s]+?)\s+vs\.?\s+([A-Za-z\s]+?)\s+\d{4}-",  # Lakers vs. Celtics 2024-04-07
        r"^([A-Za-z\s]+?)\s+vs\.?\s+([A-Za-z\s]+?)$",  # Lakers vs Celtics (end of string)
        r"([A-Za-z\s]+?)\s+vs\.?\s+([A-Za-z\s]+?)",  # Anywhere in string
    ]

    for pattern in patterns:
        match = re.search(pattern, question)
        if match:
            home = match.group(1).strip()
            away = match.group(2).strip()
            return (home, away)

    logger.debug(f"Could not extract teams from: {question}")
    return (None, None)


def lookup_game_id(
    db_path: str,
    game_date: str,
    home_team: Optional[str],
    away_team: Optional[str]
) -> Optional[str]:
    """
    Lookup NBA game ID in database by date and team names.

    Args:
        db_path: Path to nba_api.sqlite
        game_date: YYYY-MM-DD
        home_team: Team name or abbreviation
        away_team: Team name or abbreviation

    Returns:
        game_id (string) or None if not found
    """
    if not home_team or not away_team:
        return None

    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Try to match by date and team names (case-insensitive)
        # Assume table structure: GAME_DATE, HOME_TEAM_ID, AWAY_TEAM_ID, etc.
        cursor.execute("""
            SELECT GAME_ID
            FROM game
            WHERE DATE(GAME_DATE) = ?
            LIMIT 1
        """, (game_date,))

        result = cursor.fetchone()
        conn.close()

        if result:
            return str(result[0])

        logger.debug(f"No game found for {game_date}: {home_team} vs {away_team}")
        return None

    except Exception as e:
        logger.warning(f"Error looking up game ID: {e}")
        return None


def collect_polymarket_odds(
    start_date: str,
    end_date: str,
    db_path: str,
    rate_limit_delay: float = REQUEST_DELAY,
    output_csv: str = "outputs/polymarket_odds.csv",
    failed_csv: str = "outputs/polymarket_failed_markets.csv",
    resume: bool = True
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Main collector: fetch Polymarket odds and enrich with game_id.

    Robust with error tracking and resume capability:
    - Saves progress incrementally to CSV
    - Tracks failed markets for retry
    - Can resume from failures

    Args:
        start_date: YYYY-MM-DD
        end_date: YYYY-MM-DD
        db_path: Path to nba_api.sqlite
        rate_limit_delay: Seconds between API requests
        output_csv: Path to save successful odds
        failed_csv: Path to save failed markets
        resume: If True, skip markets already in output_csv

    Returns:
        Tuple of (successful_odds_df, failed_markets_df)
    """
    logger.info(f"Collecting Polymarket odds from {start_date} to {end_date}")

    import os
    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)

    # Load existing results if resume enabled
    processed_markets = set()
    rows = []

    if resume and os.path.exists(output_csv):
        try:
            existing_df = pd.read_csv(output_csv)
            rows = existing_df.to_dict('records')
            logger.info(f"Resumed from {len(rows)} previously collected odds")
        except Exception as e:
            logger.warning(f"Could not resume from {output_csv}: {e}")

    failed_markets = []

    # Find all markets
    events = find_nba_markets(start_date, end_date)
    games_searched = len(events)
    markets_total = sum(len(e.get("markets", [])) for e in events)

    logger.info(f"Found {games_searched} games, {markets_total} markets to process")

    for event_idx, event in enumerate(events):
        slug = event.get("slug", "")
        game_date = event.get("game_date", "")
        markets = event.get("markets", [])

        for market_idx, market in enumerate(markets):
            try:
                # Parse market
                question = market.get("question", "")
                clobTokenIds_str = market.get("clobTokenIds", "[]")
                market_id = f"{slug}_{market_idx}"

                # Skip if already processed
                if market_id in processed_markets:
                    continue
                processed_markets.add(market_id)

                # Extract token IDs
                try:
                    token_ids = json.loads(clobTokenIds_str)
                except (json.JSONDecodeError, TypeError) as e:
                    failed_markets.append({
                        "game_date": game_date,
                        "slug": slug,
                        "question": question[:100],
                        "error": f"Invalid tokens: {str(e)[:50]}"
                    })
                    continue

                if not token_ids or len(token_ids) < 1:
                    failed_markets.append({
                        "game_date": game_date,
                        "slug": slug,
                        "question": question[:100],
                        "error": "No tokens"
                    })
                    continue

                # Infer market type
                market_type = infer_market_type(question)

                # Extract teams
                home_team, away_team = extract_teams_from_question(question)

                # Lookup game_id
                game_id = lookup_game_id(db_path, game_date, home_team, away_team)

                # Fetch price history
                prices = fetch_price_history(token_ids[0])
                if not prices:
                    failed_markets.append({
                        "game_date": game_date,
                        "slug": slug,
                        "question": question[:100],
                        "error": "No price history"
                    })
                    continue

                # Build row
                row = {
                    "game_id": game_id,
                    "game_date": game_date,
                    "home_team": home_team,
                    "away_team": away_team,
                    "market_type": market_type,
                    "opening_price": prices.get("opening_price"),
                    "closing_price": prices.get("closing_price"),
                    "volume": market.get("volume"),
                }

                rows.append(row)

                # Save progress every 50 markets
                if len(rows) % 50 == 0:
                    df_progress = pd.DataFrame(rows)
                    df_progress.to_csv(output_csv, index=False)
                    logger.info(f"Progress: {len(rows)} odds collected, {len(failed_markets)} failed")

                time.sleep(rate_limit_delay)

            except Exception as e:
                failed_markets.append({
                    "game_date": game_date,
                    "slug": slug,
                    "question": market.get("question", "")[:100],
                    "error": str(e)[:100]
                })
                logger.debug(f"Error processing market {slug}: {e}")

    # Final save
    df_odds = pd.DataFrame(rows)
    df_failed = pd.DataFrame(failed_markets)

    df_odds.to_csv(output_csv, index=False)
    if len(df_failed) > 0:
        df_failed.to_csv(failed_csv, index=False)

    logger.info(f"=" * 70)
    logger.info(f"Backfill complete: {len(df_odds)} odds, {len(df_failed)} failed")
    logger.info(f"Output: {output_csv}")
    if len(df_failed) > 0:
        logger.info(f"Failed: {failed_csv} (run recover_polymarket_failed.py to retry)")
    logger.info(f"=" * 70)

    return df_odds, df_failed


def save_odds_csv(df: pd.DataFrame, output_path: str) -> None:
    """
    Save odds DataFrame to CSV.

    Args:
        df: DataFrame with odds
        output_path: Path to output CSV
    """
    df.to_csv(output_path, index=False)
    logger.info(f"Saved {len(df)} odds to {output_path}")


def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Collect Polymarket NBA odds")
    parser.add_argument("--start", default="2024-01-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", default="2026-06-30", help="End date (YYYY-MM-DD)")
    parser.add_argument("--db", default="data/raw/nba_api.sqlite", help="Path to nba_api.sqlite")
    parser.add_argument("--output", default="outputs/polymarket_odds.csv", help="Output CSV path")
    parser.add_argument("--failed", default="outputs/polymarket_failed_markets.csv", help="Failed markets CSV")
    parser.add_argument("--no-resume", action="store_true", help="Don't resume from existing output")

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    df_odds, df_failed = collect_polymarket_odds(
        start_date=args.start,
        end_date=args.end,
        db_path=args.db,
        output_csv=args.output,
        failed_csv=args.failed,
        resume=not args.no_resume
    )

    print(f"\n✅ Collected {len(df_odds)} odds")
    if len(df_failed) > 0:
        print(f"⚠️  {len(df_failed)} markets failed (see {args.failed})")
        print(f"   Retry with: python src/data_processing/recover_polymarket_failed.py --failed-csv {args.failed} --db-path {args.db}")


if __name__ == "__main__":
    main()
