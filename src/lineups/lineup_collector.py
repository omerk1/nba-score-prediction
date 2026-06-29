"""
Lineup Data Collection Module
==============================

Collects and caches team roster data from nba_api.

Data Source: nba_api (stats.nba.com)
- Uses CommonTeamRoster endpoint to fetch active players for a team
- Caches results per (season_id, team_id) to minimize API calls
- Returns player IDs for roster composition analysis

Author: NBA Prediction Project
Date: 2024
"""

import logging
import time
from typing import Optional

from nba_api.stats.endpoints import CommonTeamRoster
from src.utils.season_utils import season_id_to_string

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Simple in-memory cache for roster data
# Key: (season_id, team_id) -> list[int] of player IDs
_ROSTER_CACHE = {}

# Rate limiting: nba_api requests require delay
SLEEP_SECONDS = 0.7


def _get_roster_cached(season_id: int, team_id: int) -> list[int]:
    """
    Get roster player IDs from cache or fetch from API.

    Args:
        season_id: NBA season ID in composite format (e.g., 22023 for 2023-24).
                   Format: '2' + start_year (so 22023 = 2023-24 season)
        team_id: team ID

    Returns:
        list[int] of player IDs on the roster
    """
    cache_key = (season_id, team_id)

    if cache_key in _ROSTER_CACHE:
        logger.info(f"Roster cache hit for team {team_id} season {season_id}")
        return _ROSTER_CACHE[cache_key]

    try:
        # Fetch roster from nba_api
        time.sleep(SLEEP_SECONDS)  # Rate limiting
        season_str = season_id_to_string(season_id)
        roster_data = CommonTeamRoster(
            team_id=team_id,
            season=season_str,
        ).get_data_frames()[0]

        if roster_data.empty:
            logger.warning(
                f"Empty roster returned for team {team_id} season {season_id}"
            )
            player_ids = []
        else:
            # Extract player IDs from the response
            # Filter to only active players (STATUS should be 'Active')
            active_players = roster_data[
                roster_data.get("STATUS", "") == "Active"
            ]
            player_ids = active_players["PLAYER_ID"].astype(int).tolist()

            logger.info(
                f"Fetched {len(player_ids)} active players for team {team_id} "
                f"season {season_id}"
            )

        # Cache the result
        _ROSTER_CACHE[cache_key] = player_ids
        return player_ids

    except Exception as e:
        logger.error(
            f"Failed to fetch roster for team {team_id} season {season_id}: {e}"
        )
        return []


def get_available_lineup(
    season_id: int, team_id: int, game_date: str
) -> list[int]:
    """
    Get available players (roster) for a team on a given game date.

    Fetches the active roster from nba_api CommonTeamRoster endpoint.
    Note: This returns all active roster players, not actual game-day availability
    (injuries/suspensions are not filtered). For injury-adjusted availability,
    cross-reference with injury data.

    Args:
        season_id: NBA season ID in composite format (e.g., 22023 for 2023-24 season).
                   Format: '2' + start_year (20000 + year). Example: 22023 = 2023-24.
                   Will be converted to "2023-24" format for nba_api.
        team_id: team ID (e.g., 1610612738 for Celtics)
        game_date: game date in YYYY-MM-DD format (included for API compatibility,
                   roster is season-wide)

    Returns:
        list[int]: player IDs of active roster members

    Raises:
        ValueError: if parameters are invalid (invalid season_id format, team_id, or game_date)
    """
    # Validate season_id format will happen in season_id_to_string
    try:
        season_id_to_string(season_id)  # Validates format and range
    except ValueError:
        raise

    if not isinstance(team_id, int) or team_id < 1000:
        raise ValueError(f"Invalid team_id: {team_id}")

    if not isinstance(game_date, str) or len(game_date) != 10:
        raise ValueError(f"Invalid game_date format (expected YYYY-MM-DD): {game_date}")

    # Fetch the full season roster
    # The game_date parameter is included in the signature for API compatibility
    # but roster is not date-specific (CommonTeamRoster returns season roster)
    roster = _get_roster_cached(season_id, team_id)

    return roster


def clear_cache() -> None:
    """Clear the in-memory roster cache."""
    global _ROSTER_CACHE
    _ROSTER_CACHE.clear()
    logger.info("Roster cache cleared")


class LineupDataLoader:
    """
    Wrapper class for consistent lineup data access.

    Provides context-manager support and additional query methods.
    Follows the same pattern as NBADataLoader for consistency.
    """

    def __init__(self):
        """Initialize the LineupDataLoader."""
        self._cache_enabled = True

    def get_lineup(
        self, season_id: int, team_id: int, game_date: str
    ) -> list[int]:
        """
        Get available lineup for a team on a given date.

        Args:
            season_id: NBA season ID
            team_id: team ID
            game_date: game date (YYYY-MM-DD)

        Returns:
            list[int]: player IDs
        """
        return get_available_lineup(season_id, team_id, game_date)

    def clear_cache(self) -> None:
        """Clear the cached roster data."""
        clear_cache()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        if not self._cache_enabled:
            self.clear_cache()
