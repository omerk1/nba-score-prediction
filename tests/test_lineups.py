"""
Unit tests for the lineup data collection module.

Tests cover:
- Function signatures and return types
- Parameter validation
- Caching behavior
- Basic API integration
"""

import pytest
from unittest.mock import patch, MagicMock
import pandas as pd

from src.lineups.lineup_collector import (
    get_available_lineup,
    clear_cache,
    LineupDataLoader,
)


class TestGetAvailableLineup:
    """Test suite for get_available_lineup() function."""

    @patch("src.lineups.lineup_collector.CommonTeamRoster")
    def test_returns_list_of_ints(self, mock_roster):
        """Test that the function returns a list of integers."""
        # Mock the CommonTeamRoster API response
        mock_instance = MagicMock()
        mock_dataframe = pd.DataFrame({
            "PLAYER_ID": [1, 2, 3, 4, 5],
            "STATUS": ["Active", "Active", "Active", "Active", "Active"],
        })
        mock_instance.get_data_frames.return_value = [mock_dataframe]
        mock_roster.return_value = mock_instance

        clear_cache()  # Clear cache before test
        result = get_available_lineup(20231, 1610612738, "2024-01-15")

        assert isinstance(result, list), "Result should be a list"
        assert all(isinstance(x, int) for x in result), "All elements should be integers"

    @patch("src.lineups.lineup_collector.CommonTeamRoster")
    def test_roster_size_realistic(self, mock_roster):
        """Test that returned roster has realistic size (12-15 players)."""
        mock_instance = MagicMock()
        # Simulate a typical NBA roster with 14 active players
        player_ids = list(range(201950, 201964))  # 14 players
        mock_dataframe = pd.DataFrame({
            "PLAYER_ID": player_ids,
            "STATUS": ["Active"] * len(player_ids),
        })
        mock_instance.get_data_frames.return_value = [mock_dataframe]
        mock_roster.return_value = mock_instance

        clear_cache()
        result = get_available_lineup(20231, 1610612738, "2024-01-15")

        assert 12 <= len(result) <= 20, f"Roster size {len(result)} outside typical range"

    @patch("src.lineups.lineup_collector.CommonTeamRoster")
    def test_filters_inactive_players(self, mock_roster):
        """Test that inactive players are filtered out."""
        mock_instance = MagicMock()
        mock_dataframe = pd.DataFrame({
            "PLAYER_ID": [1, 2, 3, 4, 5],
            "STATUS": ["Active", "Inactive", "Active", "Inactive", "Active"],
        })
        mock_instance.get_data_frames.return_value = [mock_dataframe]
        mock_roster.return_value = mock_instance

        clear_cache()
        result = get_available_lineup(20231, 1610612738, "2024-01-15")

        # Should only include active players (indices 0, 2, 4)
        assert len(result) == 3, "Should filter to 3 active players"
        assert set(result) == {1, 3, 5}, "Should only return active player IDs"

    def test_invalid_season_id(self):
        """Test that invalid season_id raises ValueError."""
        with pytest.raises(ValueError, match="Invalid season_id"):
            get_available_lineup(1000, 1610612738, "2024-01-15")

    def test_invalid_team_id(self):
        """Test that invalid team_id raises ValueError."""
        with pytest.raises(ValueError, match="Invalid team_id"):
            get_available_lineup(20231, 100, "2024-01-15")

    def test_invalid_game_date_format(self):
        """Test that invalid game_date format raises ValueError."""
        with pytest.raises(ValueError, match="Invalid game_date format"):
            get_available_lineup(20231, 1610612738, "2024-01-15T00:00:00")

    def test_invalid_game_date_length(self):
        """Test that game_date with wrong length raises ValueError."""
        with pytest.raises(ValueError, match="Invalid game_date format"):
            get_available_lineup(20231, 1610612738, "2024-01")

    @patch("src.lineups.lineup_collector.CommonTeamRoster")
    def test_caching_behavior(self, mock_roster):
        """Test that results are cached and API is only called once."""
        mock_instance = MagicMock()
        mock_dataframe = pd.DataFrame({
            "PLAYER_ID": [1, 2, 3],
            "STATUS": ["Active", "Active", "Active"],
        })
        mock_instance.get_data_frames.return_value = [mock_dataframe]
        mock_roster.return_value = mock_instance

        clear_cache()

        # First call should hit the API
        result1 = get_available_lineup(20231, 1610612738, "2024-01-15")
        assert mock_roster.call_count == 1

        # Second call with same parameters should use cache
        result2 = get_available_lineup(20231, 1610612738, "2024-01-15")
        assert mock_roster.call_count == 1  # No additional call

        # Results should be identical
        assert result1 == result2

    @patch("src.lineups.lineup_collector.CommonTeamRoster")
    def test_different_teams_separate_cache(self, mock_roster):
        """Test that different teams have separate cache entries."""
        mock_instance = MagicMock()
        mock_dataframe = pd.DataFrame({
            "PLAYER_ID": [1, 2, 3],
            "STATUS": ["Active", "Active", "Active"],
        })
        mock_instance.get_data_frames.return_value = [mock_dataframe]
        mock_roster.return_value = mock_instance

        clear_cache()

        # Call for team 1
        get_available_lineup(20231, 1610612738, "2024-01-15")
        assert mock_roster.call_count == 1

        # Call for different team should trigger another API call
        get_available_lineup(20231, 1610612742, "2024-01-15")
        assert mock_roster.call_count == 2

    @patch("src.lineups.lineup_collector.CommonTeamRoster")
    def test_empty_roster_handling(self, mock_roster):
        """Test handling of empty roster response."""
        mock_instance = MagicMock()
        mock_dataframe = pd.DataFrame()  # Empty dataframe
        mock_instance.get_data_frames.return_value = [mock_dataframe]
        mock_roster.return_value = mock_instance

        clear_cache()
        result = get_available_lineup(20231, 1610612738, "2024-01-15")

        assert isinstance(result, list), "Should return list even if empty"
        assert len(result) == 0, "Empty roster should return empty list"

    @patch("src.lineups.lineup_collector.CommonTeamRoster")
    def test_api_error_handling(self, mock_roster):
        """Test handling of API errors."""
        mock_roster.side_effect = Exception("API connection error")

        clear_cache()
        result = get_available_lineup(20231, 1610612738, "2024-01-15")

        # Should return empty list on error
        assert isinstance(result, list)
        assert len(result) == 0


class TestClearCache:
    """Test suite for cache management functions."""

    @patch("src.lineups.lineup_collector.CommonTeamRoster")
    def test_clear_cache_removes_entries(self, mock_roster):
        """Test that clear_cache() actually clears cached entries."""
        mock_instance = MagicMock()
        mock_dataframe = pd.DataFrame({
            "PLAYER_ID": [1, 2, 3],
            "STATUS": ["Active", "Active", "Active"],
        })
        mock_instance.get_data_frames.return_value = [mock_dataframe]
        mock_roster.return_value = mock_instance

        clear_cache()

        # Populate cache
        get_available_lineup(20231, 1610612738, "2024-01-15")
        assert mock_roster.call_count == 1

        # Clear cache
        clear_cache()

        # Next call should hit API again
        get_available_lineup(20231, 1610612738, "2024-01-15")
        assert mock_roster.call_count == 2


class TestLineupDataLoader:
    """Test suite for LineupDataLoader class."""

    @patch("src.lineups.lineup_collector.CommonTeamRoster")
    def test_context_manager(self, mock_roster):
        """Test that LineupDataLoader works as context manager."""
        mock_instance = MagicMock()
        mock_dataframe = pd.DataFrame({
            "PLAYER_ID": [1, 2, 3],
            "STATUS": ["Active", "Active", "Active"],
        })
        mock_instance.get_data_frames.return_value = [mock_dataframe]
        mock_roster.return_value = mock_instance

        clear_cache()

        with LineupDataLoader() as loader:
            result = loader.get_lineup(20231, 1610612738, "2024-01-15")
            assert isinstance(result, list)
            assert all(isinstance(x, int) for x in result)

    @patch("src.lineups.lineup_collector.CommonTeamRoster")
    def test_get_lineup_method(self, mock_roster):
        """Test LineupDataLoader.get_lineup() method."""
        mock_instance = MagicMock()
        mock_dataframe = pd.DataFrame({
            "PLAYER_ID": [201950, 201951, 201952],
            "STATUS": ["Active", "Active", "Active"],
        })
        mock_instance.get_data_frames.return_value = [mock_dataframe]
        mock_roster.return_value = mock_instance

        clear_cache()

        loader = LineupDataLoader()
        result = loader.get_lineup(20231, 1610612738, "2024-01-15")

        assert len(result) == 3
        assert result == [201950, 201951, 201952]

    def test_loader_clear_cache_method(self):
        """Test that LineupDataLoader.clear_cache() works."""
        loader = LineupDataLoader()
        # Should not raise any exception
        loader.clear_cache()


class TestIntegration:
    """Integration tests with realistic scenarios."""

    @patch("src.lineups.lineup_collector.CommonTeamRoster")
    def test_multiple_teams_season(self, mock_roster):
        """Test querying multiple teams in the same season."""
        # Create different responses for different teams
        def mock_roster_side_effect(*args, **kwargs):
            team_id = kwargs.get("team_id", args[0] if args else None)
            mock_instance = MagicMock()
            if team_id == 1610612738:  # Celtics
                player_ids = list(range(1, 15))  # 14 players
            else:  # Lakers
                player_ids = list(range(100, 114))  # 14 players
            mock_dataframe = pd.DataFrame({
                "PLAYER_ID": player_ids,
                "STATUS": ["Active"] * len(player_ids),
            })
            mock_instance.get_data_frames.return_value = [mock_dataframe]
            return mock_instance

        mock_roster.side_effect = mock_roster_side_effect

        clear_cache()

        # Query Celtics
        celtics_lineup = get_available_lineup(20231, 1610612738, "2024-01-15")
        # Query Lakers
        lakers_lineup = get_available_lineup(20231, 1610612742, "2024-01-15")

        assert len(celtics_lineup) == 14
        assert len(lakers_lineup) == 14
        assert set(celtics_lineup) != set(lakers_lineup)
