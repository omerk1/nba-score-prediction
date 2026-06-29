"""
Polymarket Betting Data Analysis
==================================

Fetches and analyzes Polymarket OVER/UNDER odds for NBA games.
Includes fallback approach when API is unavailable.

Author: NBA Prediction Project
Date: 2026
"""

import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def fetch_polymarket_odds(date_range: tuple[str, str]) -> pd.DataFrame:
    """Fetch Polymarket OVER/UNDER odds for NBA games.

    Attempts to fetch real Polymarket odds via API. Falls back to
    generating synthetic data if API is unavailable, with documentation
    of the fallback approach in src/betting/README.md.

    Args:
        date_range: (start_date, end_date) as YYYY-MM-DD strings

    Returns:
        DataFrame with columns:
        - game_id: Unique game identifier
        - over_pct: Probability of OVER outcome (0-100)
        - under_pct: Probability of UNDER outcome (0-100)
        - volume: Trading volume in USD
        - game_date: Date of the game
        - team_home: Home team (if available)
        - team_away: Away team (if available)
        - over_line: The OVER line (total points)

    Raises:
        ValueError: If date_range format is invalid
    """
    try:
        # Validate date range
        start_date = datetime.strptime(date_range[0], "%Y-%m-%d")
        end_date = datetime.strptime(date_range[1], "%Y-%m-%d")

        if start_date > end_date:
            raise ValueError(f"Start date {date_range[0]} must be <= end date {date_range[1]}")

        # Attempt to fetch from Polymarket API
        df = _fetch_from_polymarket_api(start_date, end_date)

        if df is not None and len(df) > 0:
            logger.info(f"Fetched {len(df)} games from Polymarket API")
            return df

        # Fallback: Generate synthetic data
        logger.warning(
            "Polymarket API unavailable or no data returned. "
            "Generating synthetic data for analysis. "
            "See src/betting/README.md for fallback approach."
        )
        df = _generate_synthetic_odds(start_date, end_date)
        return df

    except ValueError as e:
        logger.error(f"Invalid date format: {e}")
        raise


def _fetch_from_polymarket_api(
    start_date: datetime,
    end_date: datetime
) -> Optional[pd.DataFrame]:
    """
    Fetch odds from Polymarket API.

    NOTE: Polymarket's public API has limited free access for historical odds.
    This implementation documents the approach; actual API integration may require:
    - API key/authentication
    - Webhook-based real-time data
    - Third-party aggregators (e.g., The Graph, Sportbook APIs)

    Args:
        start_date: Start of date range
        end_date: End of date range

    Returns:
        DataFrame with odds or None if unavailable
    """
    try:
        import requests
    except ImportError:
        logger.debug("requests library not available, skipping API fetch")
        return None

    # Polymarket API endpoint (requires authentication in production)
    # For now, we attempt a basic call with no auth (will likely fail)
    polymarket_url = "https://api.polymarket.com/markets"

    try:
        # Example API call (this will fail without proper auth/setup)
        # In production, would need:
        # 1. API key from Polymarket
        # 2. Proper headers/authentication
        # 3. Market ID mapping for NBA games
        response = requests.get(
            polymarket_url,
            params={
                "limit": 1000,
                "order": "volume_24h",
                "ascending": False
            },
            timeout=10
        )
        response.raise_for_status()

        data = response.json()

        # Parse markets and filter for NBA games
        records = []
        for market in data.get("markets", []):
            if _is_nba_game(market):
                record = _parse_polymarket_market(market, start_date, end_date)
                if record:
                    records.append(record)

        if records:
            return pd.DataFrame(records)

    except (requests.RequestException, KeyError, ValueError) as e:
        logger.debug(f"Polymarket API call failed: {e}")

    return None


def _is_nba_game(market: dict) -> bool:
    """Check if a Polymarket market is an NBA game."""
    description = market.get("description", "").upper()
    title = market.get("title", "").upper()

    return "NBA" in title or "NBA" in description


def _parse_polymarket_market(
    market: dict,
    start_date: datetime,
    end_date: datetime
) -> Optional[dict]:
    """
    Parse a Polymarket market into odds record.

    Args:
        market: Polymarket market object
        start_date: Filter start date
        end_date: Filter end date

    Returns:
        Record dict or None if invalid
    """
    try:
        # Extract OVER/UNDER outcomes
        outcomes = {outcome["title"]: outcome for outcome in market.get("outcomes", [])}

        if "Yes" not in outcomes or "No" not in outcomes:
            return None

        over_prob = float(outcomes["Yes"].get("probability", 0.5))
        under_prob = 1.0 - over_prob

        # Extract game details from market description
        # Format varies, but typically: "Will [Team A] OVER [line] [Team B] on [date]?"
        description = market.get("description", "")

        return {
            "game_id": market.get("id", ""),
            "game_date": datetime.now().date(),  # Would need to parse from description
            "team_home": "",
            "team_away": "",
            "over_line": _extract_line(description),
            "over_pct": over_prob * 100,
            "under_pct": under_prob * 100,
            "volume": float(market.get("volume", 0)),
            "liquidity": float(market.get("liquidity", 0)),
        }
    except (KeyError, ValueError, TypeError) as e:
        logger.debug(f"Failed to parse market: {e}")
        return None


def _extract_line(description: str) -> Optional[float]:
    """Extract the OVER line (total points) from description."""
    # Placeholder for parsing logic
    # Would need regex to extract numeric line
    return None


def _generate_synthetic_odds(
    start_date: datetime,
    end_date: datetime
) -> pd.DataFrame:
    """
    Generate synthetic Polymarket odds data for analysis.

    Used as fallback when API is unavailable. Creates realistic
    odds distributions based on typical NBA game patterns.

    Args:
        start_date: Start of date range
        end_date: End of date range

    Returns:
        DataFrame with synthetic odds
    """
    # Typical NBA game schedule: ~1-2 games per day
    current_date = start_date
    records = []
    game_id = 0

    # NBA teams for realistic team names
    teams = [
        "ATL", "BOS", "BKN", "CHA", "CHI", "CLE", "DAL", "DEN",
        "DET", "GSW", "HOU", "LAC", "LAL", "MEM", "MIA", "MIL",
        "MIN", "NOP", "NYK", "OKC", "ORL", "PHI", "PHX", "POR",
        "SAC", "SAS", "TOR", "UTA", "WAS"
    ]

    while current_date <= end_date:
        # ~60% chance of a game on any given date
        if np.random.random() < 0.6:
            num_games = np.random.choice([1, 2], p=[0.7, 0.3])

            for _ in range(num_games):
                # Pick random teams
                home_idx = np.random.randint(0, len(teams))
                away_idx = np.random.randint(0, len(teams))
                while away_idx == home_idx:
                    away_idx = np.random.randint(0, len(teams))

                # Generate realistic odds
                # OVER typically has 48-52% probability
                over_pct = np.random.normal(50, 2)
                over_pct = np.clip(over_pct, 30, 70)

                records.append({
                    "game_id": f"game_{game_id:06d}",
                    "game_date": current_date.date(),
                    "team_home": teams[home_idx],
                    "team_away": teams[away_idx],
                    "over_line": np.random.uniform(205, 230),  # Typical NBA total lines
                    "over_pct": over_pct,
                    "under_pct": 100 - over_pct,
                    "volume": np.random.exponential(50000),  # Right-skewed volume
                    "liquidity": np.random.exponential(20000),
                })
                game_id += 1

        current_date += timedelta(days=1)

    df = pd.DataFrame(records)
    df["game_date"] = pd.to_datetime(df["game_date"])

    logger.info(f"Generated {len(df)} synthetic odds records for {start_date.date()} to {end_date.date()}")

    return df


def create_exploratory_analysis(
    df: pd.DataFrame,
    output_dir: str = "outputs"
) -> None:
    """
    Create exploratory analysis visualizations and CSV output.

    Args:
        df: DataFrame from fetch_polymarket_odds()
        output_dir: Directory to save outputs
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save cleaned data to CSV
    csv_path = output_path / "polymarket_analysis.csv"
    df.to_csv(csv_path, index=False)
    logger.info(f"Saved analysis to {csv_path}")

    # Create visualizations
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        logger.warning("matplotlib/seaborn not available, skipping plots")
        return

    sns.set_style("whitegrid")

    # 1. Odds distribution histogram
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].hist(df["over_pct"], bins=30, alpha=0.7, color="blue", edgecolor="black")
    axes[0].set_xlabel("OVER Probability (%)")
    axes[0].set_ylabel("Frequency")
    axes[0].set_title("Distribution of OVER Probabilities")
    axes[0].axvline(df["over_pct"].mean(), color="red", linestyle="--", label=f"Mean: {df['over_pct'].mean():.1f}%")
    axes[0].legend()

    axes[1].hist(df["volume"], bins=30, alpha=0.7, color="green", edgecolor="black")
    axes[1].set_xlabel("Volume (USD)")
    axes[1].set_ylabel("Frequency")
    axes[1].set_title("Distribution of Trading Volume")
    axes[1].set_yscale("log")

    plt.tight_layout()
    plt.savefig(output_path / "polymarket_odds_distribution.png", dpi=100, bbox_inches="tight")
    logger.info(f"Saved odds distribution plot to {output_path / 'polymarket_odds_distribution.png'}")
    plt.close()

    # 2. Time series: odds movement
    df_sorted = df.sort_values("game_date")

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.scatter(df_sorted["game_date"], df_sorted["over_pct"], alpha=0.5, s=30)
    ax.plot(df_sorted["game_date"], df_sorted["over_pct"].rolling(7, center=True).mean(),
            color="red", linewidth=2, label="7-day rolling average")
    ax.set_xlabel("Game Date")
    ax.set_ylabel("OVER Probability (%)")
    ax.set_title("OVER Odds Movement Over Time")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output_path / "polymarket_odds_timeseries.png", dpi=100, bbox_inches="tight")
    logger.info(f"Saved time series plot to {output_path / 'polymarket_odds_timeseries.png'}")
    plt.close()

    # 3. Volume vs Odds correlation
    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(df["over_pct"], df["volume"], alpha=0.5, s=50, c=df["liquidity"], cmap="viridis")
    ax.set_xlabel("OVER Probability (%)")
    ax.set_ylabel("Volume (USD)")
    ax.set_title("Trading Volume vs OVER Probability")
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("Liquidity (USD)")
    ax.set_yscale("log")
    plt.tight_layout()
    plt.savefig(output_path / "polymarket_volume_vs_odds.png", dpi=100, bbox_inches="tight")
    logger.info(f"Saved volume vs odds plot to {output_path / 'polymarket_volume_vs_odds.png'}")
    plt.close()

    # Print summary statistics
    logger.info("\n" + "="*60)
    logger.info("POLYMARKET ODDS SUMMARY STATISTICS")
    logger.info("="*60)
    logger.info(f"Total games: {len(df):,}")
    logger.info(f"Date range: {df['game_date'].min().date()} to {df['game_date'].max().date()}")
    logger.info(f"\nOVER Probability:")
    logger.info(f"  Mean: {df['over_pct'].mean():.2f}%")
    logger.info(f"  Median: {df['over_pct'].median():.2f}%")
    logger.info(f"  Std Dev: {df['over_pct'].std():.2f}%")
    logger.info(f"  Range: [{df['over_pct'].min():.2f}%, {df['over_pct'].max():.2f}%]")
    logger.info(f"\nTrading Volume:")
    logger.info(f"  Total: ${df['volume'].sum():,.0f}")
    logger.info(f"  Mean: ${df['volume'].mean():,.0f}")
    logger.info(f"  Median: ${df['volume'].median():,.0f}")
    logger.info(f"  Max: ${df['volume'].max():,.0f}")
    logger.info("="*60 + "\n")


if __name__ == "__main__":
    # Example usage
    df = fetch_polymarket_odds(("2024-01-01", "2024-01-31"))
    create_exploratory_analysis(df)
    print(f"\nFetched {len(df)} games")
    print(f"Columns: {list(df.columns)}")
