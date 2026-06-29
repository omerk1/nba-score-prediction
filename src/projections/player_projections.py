"""
Player Box Score Projections
=============================

Module for projecting individual player scoring contributions in NBA games
using rolling average statistics from historical game logs.

Author: NBA Prediction Project
Date: 2024
"""

import logging
import time
from typing import Optional

import pandas as pd
from nba_api.stats.endpoints import BoxScoreTraditionalV2, PlayerGameLog

from src.data_processing.data_loader import NBADataLoader
from src.utils.config_loader import load_config

# Setup logging (do not call basicConfig at module level — let host application configure)
logger = logging.getLogger(__name__)

# API throttling to avoid rate limits
SLEEP_SECONDS = 0.6


def _get_game_id(
    date: str,
    home_team_id: int,
    away_team_id: int,
    db_path: str,
) -> Optional[str]:
    """
    Find the game_id for a specific matchup and date.

    Args:
        date: game date (YYYY-MM-DD)
        home_team_id: home team ID
        away_team_id: away team ID
        db_path: path to the NBA database

    Returns:
        game_id if found, None otherwise
    """
    loader = NBADataLoader(db_path=db_path)
    try:
        games_df = loader.load_games(
            start_date=date,
            end_date=date,
        )

        # Find matching game
        matching = games_df[
            (games_df['team_id_home'] == home_team_id) &
            (games_df['team_id_away'] == away_team_id)
        ]

        if matching.empty:
            logger.warning(
                f"No game found for {date}: {home_team_id} vs {away_team_id}"
            )
            return None

        # Handle doubleheader case: if multiple games, log warning but return first
        if len(matching) > 1:
            logger.warning(
                f"Multiple games found on {date}: {home_team_id} vs {away_team_id}. "
                f"Returning first game_id. Consider filtering by game time in future."
            )

        return matching.iloc[0]['game_id']
    finally:
        loader.close()


def _get_player_box_scores(game_id: str) -> pd.DataFrame:
    """
    Fetch player box scores for a specific game.

    Args:
        game_id: NBA game ID

    Returns:
        DataFrame with player box score data (includes PLAYER_ID and PTS columns)
    """
    time.sleep(SLEEP_SECONDS)
    box_score = BoxScoreTraditionalV2(game_id=game_id)

    # Get player stats (second dataframe is players, first is team totals)
    data_sets = box_score.get_data_frames()
    if not data_sets or len(data_sets) < 2:
        logger.warning(f"Unexpected box score format for game {game_id}: expected 2+ data sets, got {len(data_sets)}")
        return pd.DataFrame()

    player_stats = data_sets[1]  # Player stats, not team totals
    if player_stats is None or player_stats.empty:
        logger.warning(f"Empty player stats in box score for game {game_id}")
        return pd.DataFrame()

    return player_stats


def _get_player_game_log(
    player_id: int,
    date_to: str,
    n_games: int = 20,
) -> pd.DataFrame:
    """
    Fetch recent game log for a player up to a certain date.

    Args:
        player_id: NBA player ID
        date_to: cutoff date (YYYY-MM-DD)
        n_games: number of recent games to fetch

    Returns:
        DataFrame with player game log data
    """
    time.sleep(SLEEP_SECONDS)
    try:
        game_log = PlayerGameLog(
            player_id=player_id,
            season="",  # All seasons
            date_from_nullable="",
            date_to_nullable=date_to,
            league_id_nullable="00",
        )
        df = game_log.get_data_frames()[0]

        # Keep most recent n games, sorted ascending by date
        if 'GAME_DATE' in df.columns:
            df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])
            df = df.nlargest(n_games, 'GAME_DATE').sort_values('GAME_DATE', ascending=True)

        return df
    except Exception as e:
        logger.warning(f"Error fetching game log for player {player_id}: {e}")
        return pd.DataFrame()


def _calculate_rolling_avg_ppg(game_log: pd.DataFrame, window: int = 5) -> float:
    """
    Calculate rolling average PPG from a player's game log.

    Args:
        game_log: DataFrame with player game log (must have 'PTS' column)
        window: rolling window size in games

    Returns:
        Average PPG over the window (or average of all available games if fewer)
    """
    if 'PTS' not in game_log.columns:
        return 0.0

    pts = game_log['PTS'].values
    if len(pts) == 0:
        return 0.0

    # Use minimum of window size and available games
    actual_window = min(window, len(pts))
    return float(pts[-actual_window:].mean())


def project_game_contributions(
    date: str,
    home_team_id: int,
    away_team_id: int,
) -> dict[int, float]:
    """
    Project scoring contributions for all available players in a game.

    Uses rolling average statistics from player game logs to project
    Points Per Game (PPG) for each player.

    Performance Note:
        Makes 1 API call per player (~30+ calls per game) with rate-limit throttling.
        Typical latency: 18+ seconds per game. Future optimization should consider
        batching API calls or caching player stats to reduce latency and API load.

    Args:
        date: game date (YYYY-MM-DD)
        home_team_id: home team ID (10-digit NBA team ID)
        away_team_id: away team ID (10-digit NBA team ID)

    Returns:
        Dictionary mapping player_id (int) to projected PPG (float).
        Returns 0.0 if no historical data available (not actual game PPG).
        Returns empty dict if game not found or data unavailable.

    Example:
        >>> projections = project_game_contributions('2024-04-14', 1610612744, 1610612762)
        >>> len(projections)
        48  # Both rosters
        >>> all(isinstance(ppg, float) for ppg in projections.values())
        True
    """
    config = load_config()
    db_path = config.data_paths.raw_db
    rolling_windows = config.features.rolling_windows  # [5, 10, 20] typically

    # Step 1: Find the game
    game_id = _get_game_id(date, home_team_id, away_team_id, db_path)
    if game_id is None:
        logger.warning(f"Could not find game ID for {date}: {home_team_id} vs {away_team_id}")
        return {}

    logger.info(f"Found game_id {game_id} for {date}: {home_team_id} vs {away_team_id}")

    # Step 2: Get box score to identify players
    box_scores = _get_player_box_scores(game_id)
    if box_scores.empty:
        logger.warning(f"Could not fetch box scores for game {game_id}")
        return {}

    logger.info(f"Found {len(box_scores)} players in box score")

    # Step 3: Project PPG for each player
    projections: dict[int, float] = {}

    for _, row in box_scores.iterrows():
        # Skip rows with NaN or invalid player IDs
        if pd.isna(row['PLAYER_ID']):
            continue

        player_id = int(row['PLAYER_ID'])

        # Skip rows without valid positive player IDs (team totals, etc.)
        if player_id <= 0:
            continue

        # Fetch player's recent game log (up to game date, excluding today)
        # Only fetch the number of games needed for the largest rolling window
        max_window = max(rolling_windows) if rolling_windows else 5
        game_log = _get_player_game_log(player_id, date_to=date, n_games=max_window)

        if not game_log.empty:
            # Calculate rolling average PPG using primary window (typically 5 games)
            primary_window = rolling_windows[0] if rolling_windows else 5
            avg_ppg = _calculate_rolling_avg_ppg(game_log, window=primary_window)
        else:
            # No historical data: return 0.0 projection (not actual PPG from box score).
            # Returning actual PPG would confuse projection with observed performance.
            avg_ppg = 0.0

        projections[player_id] = float(avg_ppg)

    logger.info(f"Projected PPG for {len(projections)} players")
    return projections


if __name__ == "__main__":
    # Test the module
    import sys

    test_date = "2024-04-14"
    test_home = 1610612744  # Warriors
    test_away = 1610612762  # Jazz

    logger.info(f"Testing projections for {test_date}: {test_home} vs {test_away}")
    result = project_game_contributions(test_date, test_home, test_away)

    if result:
        logger.info(f"Projected {len(result)} players")
        logger.info(f"Min PPG: {min(result.values()):.1f}, Max PPG: {max(result.values()):.1f}")
    else:
        logger.error("No projections returned")
        sys.exit(1)
