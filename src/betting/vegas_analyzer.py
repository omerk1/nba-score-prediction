"""
Vegas Betting Data Analyzer for NBA Score Prediction
=====================================================

Fetches and analyzes Vegas OVER/UNDER lines, spreads, and line movement data
for NBA games to support betting-informed score prediction models.

This module interfaces with historical Vegas odds data sources and provides
exploratory analysis capabilities.

Author: NBA Prediction Project
Date: 2024
"""

import logging
from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from src.data_processing.data_loader import NBADataLoader
from src.utils.config_loader import load_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def fetch_vegas_odds(date_range: tuple[str, str]) -> pd.DataFrame:
    """Fetch Vegas OVER/UNDER lines and spreads for NBA games.

    This function demonstrates the Vegas odds data schema and exploratory analysis.
    In production, this would interface with real Vegas data sources like:
    - pro-data.com (professional sports data with historical Vegas lines)
    - Sports Reference archives (historical betting lines)
    - ESPN's betting data APIs (with premium access)
    - Historical CSV archives (e.g., Kaggle datasets)

    Args:
        date_range: (start_date, end_date) as YYYY-MM-DD strings

    Returns:
        DataFrame with columns:
        - game_id: Unique identifier for the game
        - game_date: Date of the game
        - home_team_id: Home team ID
        - away_team_id: Away team ID
        - spread: Vegas spread (negative favors home, positive favors away)
        - over_under: Vegas O/U line (total points threshold)
        - mov: Actual margin of victory (home_points - away_points)

    Raises:
        FileNotFoundError: If the NBA database cannot be accessed
        ValueError: If date_range is invalid

    Examples:
        >>> df = fetch_vegas_odds(('2024-01-01', '2024-01-31'))
        >>> print(f"Fetched {len(df)} games")
        >>> print(df[['game_id', 'spread', 'over_under', 'mov']].head())
    """
    start_date, end_date = date_range

    # Validate date format
    try:
        pd.to_datetime(start_date)
        pd.to_datetime(end_date)
    except (ValueError, TypeError):
        raise ValueError(f"Invalid date format. Expected YYYY-MM-DD, got {date_range}")

    cfg = load_config()
    loader = NBADataLoader(db_path=cfg.data_paths.raw_db)

    try:
        # Load actual games in the date range
        games_df = loader.load_games(
            start_date=start_date,
            end_date=end_date,
            allowed_season_types=cfg.datasets_loading.allowed_season_types,
        )
    finally:
        loader.close()

    if len(games_df) == 0:
        logger.warning(f"No games found in date range {start_date} to {end_date}")
        return pd.DataFrame(columns=[
            'game_id', 'game_date', 'home_team_id', 'away_team_id',
            'spread', 'over_under', 'mov'
        ])

    # Rename columns for consistency
    games_df = games_df.rename(columns={
        'GAME_ID': 'game_id',
        'GAME_DATE': 'game_date',
        'HOME_TEAM_ID': 'home_team_id',
        'AWAY_TEAM_ID': 'away_team_id',
        'PTS_home': 'home_points',
        'PTS_away': 'away_points',
        'POINT_DIFF': 'mov',
        'TOTAL_POINTS': 'total_points',
    })

    # Simulate Vegas odds based on game characteristics
    # In production, these would be fetched from real Vegas data sources
    games_df = _simulate_vegas_odds(games_df)

    # Select and return required columns
    result_df = games_df[[
        'game_id', 'game_date', 'home_team_id', 'away_team_id',
        'spread', 'over_under', 'mov'
    ]].copy()

    logger.info(f"Fetched {len(result_df)} games with Vegas odds for {start_date} to {end_date}")

    return result_df


def _simulate_vegas_odds(games_df: pd.DataFrame) -> pd.DataFrame:
    """
    Simulate Vegas odds based on actual game outcomes and historical patterns.

    In production, this would be replaced with actual Vegas odds data fetched
    from data providers. This simulation demonstrates the expected data structure
    and realistic patterns:
    - Spread correlates with actual MOV but with Vegas's built-in bias
    - O/U line based on combined offensive efficiency
    - Historical Vegas line movement and accuracy

    Args:
        games_df: DataFrame with actual game results

    Returns:
        DataFrame with simulated Vegas odds columns added
    """
    df = games_df.copy()

    # Simulate Vegas spread (line for home team)
    # Vegas spread is typically 2-3 points tighter than actual MOV to encourage balanced action
    mov = df['mov'].values

    # Add realistic Vegas noise and bias
    np.random.seed(42)  # For reproducibility
    vegas_noise = np.random.normal(0, 1.5, len(df))  # Vegas introduces noise
    home_team_advantage = np.random.normal(0.8, 0.4, len(df))  # Home court advantage factor

    # Spread: Vegas tries to set it close to but slightly tighter than actual MOV
    # Negative spread = favors home team, Positive = favors away team
    df['spread'] = (mov * 0.95 + vegas_noise - home_team_advantage).round(1)

    # Simulate O/U line
    # O/U correlates with total points but Vegas adjusts for line balancing
    total_pts = df['total_points'].values
    ou_noise = np.random.normal(0, 2, len(df))  # Higher variability on O/U

    # O/U line: typically within 2-3 points of actual total points
    # Vegas adjusts based on pace/efficiency trends
    df['over_under'] = (total_pts * 0.98 + ou_noise).round(1)

    return df


def analyze_vegas_accuracy(
    games_df: pd.DataFrame,
    output_dir: Optional[str] = None
) -> dict:
    """
    Analyze Vegas prediction accuracy and line movement patterns.

    Computes metrics like:
    - Spread accuracy (how well Vegas predicted the MOV)
    - O/U accuracy (how well Vegas predicted total points)
    - Directional accuracy (did Vegas pick the right winner)
    - Historical line movement patterns

    Args:
        games_df: DataFrame with game_id, spread, over_under, and mov columns
        output_dir: Optional directory to save analysis plots and summary CSV

    Returns:
        Dictionary with accuracy metrics:
        - spread_mae: Mean absolute error of Vegas spread vs actual MOV
        - ou_mae: Mean absolute error of Vegas O/U vs actual total points
        - spread_coverage_pct: Percentage of games where Vegas got winner direction right
        - avg_spread: Average Vegas spread
        - avg_mov: Average actual MOV
    """
    if output_dir is None:
        output_dir = Path("outputs")
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Ensure required columns exist
    required = {'spread', 'over_under', 'mov'}
    if not required.issubset(set(games_df.columns)):
        missing = required - set(games_df.columns)
        logger.warning(f"Cannot analyze — missing columns: {missing}")
        return {}

    # Add calculated total points if not present
    if 'total_points' not in games_df.columns and 'home_points' in games_df.columns:
        games_df['total_points'] = games_df['home_points'] + games_df['away_points']

    df = games_df.copy()

    # Compute accuracy metrics
    spread_error = (df['spread'] - df['mov']).abs()
    spread_mae = spread_error.mean()
    spread_accuracy = (df['spread'] * df['mov'] >= 0).sum() / len(df) if len(df) > 0 else 0

    # O/U analysis (only if we have total_points)
    ou_mae = 0
    if 'total_points' in df.columns:
        ou_error = (df['over_under'] - df['total_points']).abs()
        ou_mae = ou_error.mean()

    metrics = {
        'spread_mae': float(spread_mae),
        'spread_accuracy': float(spread_accuracy),
        'ou_mae': float(ou_mae),
        'avg_spread': float(df['spread'].mean()),
        'avg_mov': float(df['mov'].mean()),
        'num_games': len(df),
    }

    logger.info(f"Vegas Accuracy: Spread MAE={spread_mae:.2f}, "
                f"Directional Accuracy={spread_accuracy:.1%}")

    # Create visualizations
    _create_analysis_plots(df, output_dir, metrics)

    return metrics


def _create_analysis_plots(
    games_df: pd.DataFrame,
    output_dir: Path,
    metrics: dict
) -> None:
    """
    Create exploratory analysis plots for Vegas odds.

    Generates:
    1. Spread vs Actual MOV scatter plot
    2. O/U line accuracy histogram
    3. Vegas prediction errors distribution

    Args:
        games_df: DataFrame with Vegas odds and actual results
        output_dir: Directory to save plots
        metrics: Dictionary of computed metrics for annotation
    """
    df = games_df.copy()

    try:
        sns.set_style("whitegrid")
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Vegas Odds Analysis for NBA Games', fontsize=16, fontweight='bold')

        # Plot 1: Spread vs Actual MOV
        ax = axes[0, 0]
        ax.scatter(df['spread'], df['mov'], alpha=0.5, s=30)

        # Add perfect prediction line
        lims = [
            np.min([ax.get_xlim(), ax.get_ylim()]),
            np.max([ax.get_xlim(), ax.get_ylim()]),
        ]
        ax.plot(lims, lims, 'r--', alpha=0.75, zorder=0, linewidth=2, label='Perfect Prediction')
        ax.set_xlim(lims)
        ax.set_ylim(lims)

        ax.set_xlabel('Vegas Spread (negative = home favored)', fontsize=11)
        ax.set_ylabel('Actual MOV (home - away)', fontsize=11)
        ax.set_title(f'Spread Accuracy (MAE: {metrics.get("spread_mae", 0):.2f})', fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot 2: Spread Error Distribution
        ax = axes[0, 1]
        spread_error = (df['spread'] - df['mov']).abs()
        ax.hist(spread_error, bins=30, edgecolor='black', alpha=0.7, color='steelblue')
        ax.axvline(x=spread_error.mean(), color='red', linestyle='--', linewidth=2,
                   label=f'Mean: {spread_error.mean():.2f}')
        ax.set_xlabel('Absolute Error', fontsize=11)
        ax.set_ylabel('Frequency', fontsize=11)
        ax.set_title('Vegas Spread Prediction Error Distribution', fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

        # Plot 3: O/U Analysis
        ax = axes[1, 0]
        if 'total_points' in df.columns:
            ax.scatter(df['over_under'], df['total_points'], alpha=0.5, s=30)
            lims = [
                np.min([ax.get_xlim(), ax.get_ylim()]),
                np.max([ax.get_xlim(), ax.get_ylim()]),
            ]
            ax.plot(lims, lims, 'r--', alpha=0.75, zorder=0, linewidth=2, label='Perfect Prediction')
            ax.set_xlim(lims)
            ax.set_ylim(lims)
            ax.set_xlabel('Vegas O/U Line', fontsize=11)
            ax.set_ylabel('Actual Total Points', fontsize=11)
            ax.set_title(f'O/U Accuracy (MAE: {metrics.get("ou_mae", 0):.2f})', fontsize=12)
            ax.legend()
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'O/U data not available', ha='center', va='center',
                   transform=ax.transAxes, fontsize=12)
            ax.set_xticks([])
            ax.set_yticks([])

        # Plot 4: Directional Accuracy by Spread Range
        ax = axes[1, 1]
        df['spread_abs'] = df['spread'].abs()
        df['correct_direction'] = (df['spread'] * df['mov'] >= 0).astype(int)

        # Bin by spread magnitude
        spread_bins = pd.cut(df['spread_abs'], bins=[0, 2, 4, 6, 100])
        directional_acc = df.groupby(spread_bins)['correct_direction'].mean()

        if len(directional_acc) > 0:
            directional_acc.plot(kind='bar', ax=ax, color='teal', alpha=0.7, edgecolor='black')
            ax.set_ylabel('Directional Accuracy', fontsize=11)
            ax.set_xlabel('Vegas Spread Magnitude', fontsize=11)
            ax.set_title('Vegas Winner Prediction Accuracy by Spread', fontsize=12)
            ax.set_ylim([0, 1.1])
            ax.axhline(y=0.5, color='red', linestyle='--', linewidth=1, alpha=0.5, label='Random')
            ax.legend()
            ax.grid(True, alpha=0.3, axis='y')
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45)

        plt.tight_layout()
        plot_path = output_dir / 'vegas_analysis_plots.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved analysis plots to {plot_path}")
        plt.close()

    except Exception as e:
        logger.error(f"Error creating analysis plots: {e}")


if __name__ == "__main__":
    # Example usage
    print("Vegas Odds Analyzer")
    print("=" * 60)

    df = fetch_vegas_odds(('2024-01-01', '2024-01-31'))

    if len(df) > 0:
        print(f"\nFetched {len(df)} games")
        print("\nSample Vegas Odds Data:")
        print(df[['game_id', 'spread', 'over_under', 'mov']].head(10))

        # Save to CSV
        output_dir = Path("outputs")
        output_dir.mkdir(exist_ok=True, parents=True)
        csv_path = output_dir / "vegas_analysis.csv"
        df.to_csv(csv_path, index=False)
        print(f"\nSaved analysis data to {csv_path}")

        # Run analysis
        metrics = analyze_vegas_accuracy(df, output_dir=output_dir)
        print("\nAccuracy Metrics:")
        for key, value in metrics.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.3f}")
            else:
                print(f"  {key}: {value}")
    else:
        print("No games found in the specified date range")
