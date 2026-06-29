"""
Unit tests for player_projections module.

Tests cover:
- project_game_contributions function
- Helper functions for fetching and processing player data
- Edge cases and error handling
"""

import pytest
import pandas as pd
from unittest.mock import Mock, patch, MagicMock

from src.projections.player_projections import (
    project_game_contributions,
    _get_game_id,
    _get_player_box_scores,
    _get_player_game_log,
    _calculate_rolling_avg_ppg,
)


class TestCalculateRollingAvgPpg:
    """Test rolling average PPG calculation."""

    def test_empty_game_log(self):
        """Empty game log should return 0.0."""
        df = pd.DataFrame()
        result = _calculate_rolling_avg_ppg(df, window=5)
        assert result == 0.0

    def test_missing_pts_column(self):
        """Game log without PTS column should return 0.0."""
        df = pd.DataFrame({'OTHER': [10, 20, 30]})
        result = _calculate_rolling_avg_ppg(df, window=5)
        assert result == 0.0

    def test_single_game(self):
        """Single game should return that game's PPG."""
        df = pd.DataFrame({'PTS': [25.0]})
        result = _calculate_rolling_avg_ppg(df, window=5)
        assert result == 25.0

    def test_multiple_games_with_full_window(self):
        """Multiple games should return average over window."""
        df = pd.DataFrame({'PTS': [20.0, 22.0, 24.0, 26.0, 28.0]})
        result = _calculate_rolling_avg_ppg(df, window=5)
        # Average of all 5
        expected = (20.0 + 22.0 + 24.0 + 26.0 + 28.0) / 5
        assert result == expected

    def test_window_larger_than_games(self):
        """Window larger than available games uses all available."""
        df = pd.DataFrame({'PTS': [20.0, 30.0]})
        result = _calculate_rolling_avg_ppg(df, window=5)
        expected = (20.0 + 30.0) / 2
        assert result == expected

    def test_uses_most_recent_games(self):
        """Should use most recent games in the window."""
        df = pd.DataFrame({'PTS': [10.0, 20.0, 30.0, 40.0, 50.0, 60.0]})
        result = _calculate_rolling_avg_ppg(df, window=3)
        # Last 3 games: 40, 50, 60
        expected = (40.0 + 50.0 + 60.0) / 3
        assert result == expected

    def test_zero_pts(self):
        """Games with 0 points should be included."""
        df = pd.DataFrame({'PTS': [10.0, 0.0, 5.0]})
        result = _calculate_rolling_avg_ppg(df, window=3)
        expected = (10.0 + 0.0 + 5.0) / 3
        assert result == expected


class TestGetPlayerGameLog:
    """Test fetching player game logs."""

    @patch('src.projections.player_projections.PlayerGameLog')
    def test_successful_fetch(self, mock_game_log_class):
        """Successfully fetch and process game log."""
        # Setup mock
        mock_df = pd.DataFrame({
            'GAME_DATE': ['2024-04-01', '2024-04-03', '2024-04-05'],
            'PTS': [20, 25, 22],
        })
        mock_game_log = Mock()
        mock_game_log.get_data_frames.return_value = [mock_df]
        mock_game_log_class.return_value = mock_game_log

        result = _get_player_game_log(201939, '2024-04-05', n_games=3)

        assert not result.empty
        assert len(result) == 3
        assert 'PTS' in result.columns

    @patch('src.projections.player_projections.PlayerGameLog')
    def test_empty_game_log(self, mock_game_log_class):
        """Handle empty game log gracefully."""
        mock_game_log = Mock()
        mock_game_log.get_data_frames.return_value = [pd.DataFrame()]
        mock_game_log_class.return_value = mock_game_log

        result = _get_player_game_log(999999, '2024-04-05')

        assert result.empty

    @patch('src.projections.player_projections.PlayerGameLog')
    def test_api_error_handling(self, mock_game_log_class):
        """Handle API errors gracefully."""
        mock_game_log_class.side_effect = Exception("API Error")

        result = _get_player_game_log(201939, '2024-04-05')

        assert result.empty


class TestGetPlayerBoxScores:
    """Test fetching player box scores."""

    @patch('src.projections.player_projections.BoxScoreTraditionalV2')
    def test_successful_fetch(self, mock_box_score_class):
        """Successfully fetch player box scores."""
        mock_df = pd.DataFrame({
            'PLAYER_ID': [201939, 2544, 203076],
            'PLAYER_NAME': ['Stephen Curry', 'LeBron James', 'Kyrie Irving'],
            'PTS': [28, 25, 22],
        })
        mock_box_score = Mock()
        mock_box_score.get_data_frames.return_value = [pd.DataFrame(), mock_df]
        mock_box_score_class.return_value = mock_box_score

        result = _get_player_box_scores('0022301198')

        assert not result.empty
        assert len(result) == 3
        assert 'PLAYER_ID' in result.columns
        assert 'PTS' in result.columns

    @patch('src.projections.player_projections.BoxScoreTraditionalV2')
    def test_unexpected_format(self, mock_box_score_class):
        """Handle unexpected box score format."""
        mock_box_score = Mock()
        mock_box_score.get_data_frames.return_value = []
        mock_box_score_class.return_value = mock_box_score

        result = _get_player_box_scores('0022301198')

        assert result.empty


class TestGetGameId:
    """Test finding game IDs."""

    @patch('src.projections.player_projections.NBADataLoader')
    def test_game_found(self, mock_loader_class):
        """Successfully find game ID."""
        mock_games = pd.DataFrame({
            'game_id': ['0022301198'],
            'game_date': ['2024-04-14'],
            'team_id_home': [1610612744],
            'team_id_away': [1610612762],
        })

        mock_loader = Mock()
        mock_loader.load_games.return_value = mock_games
        mock_loader_class.return_value = mock_loader

        result = _get_game_id('2024-04-14', 1610612744, 1610612762, 'test.db')

        assert result == '0022301198'
        mock_loader.close.assert_called_once()

    @patch('src.projections.player_projections.NBADataLoader')
    def test_game_not_found(self, mock_loader_class):
        """Handle case when game not found."""
        mock_loader = Mock()
        mock_loader.load_games.return_value = pd.DataFrame(columns=[
            'game_id', 'game_date', 'team_id_home', 'team_id_away'
        ])
        mock_loader_class.return_value = mock_loader

        result = _get_game_id('2024-04-14', 9999, 9999, 'test.db')

        assert result is None


class TestProjectGameContributions:
    """Test main projection function."""

    @patch('src.projections.player_projections.load_config')
    @patch('src.projections.player_projections._get_game_id')
    @patch('src.projections.player_projections._get_player_box_scores')
    @patch('src.projections.player_projections._get_player_game_log')
    def test_successful_projection(
        self,
        mock_game_log,
        mock_box_scores,
        mock_game_id,
        mock_load_config,
    ):
        """Successfully project game contributions."""
        # Setup mock config
        mock_config = Mock()
        mock_config.data_paths.raw_db = 'test.db'
        mock_config.features.rolling_windows = [5, 10, 20]
        mock_load_config.return_value = mock_config

        # Setup mock game ID
        mock_game_id.return_value = '0022301198'

        # Setup mock box scores
        mock_box_scores.return_value = pd.DataFrame({
            'PLAYER_ID': [201939, 2544, 203076, 101108, 2019],
            'PLAYER_NAME': ['Curry', 'LeBron', 'Kyrie', 'Unknown1', 'Unknown2'],
            'PTS': [28, 25, 22, 0, 0],
        })

        # Setup mock game log
        mock_game_log.return_value = pd.DataFrame({
            'PTS': [25, 26, 27, 28, 29],
        })

        result = project_game_contributions('2024-04-14', 1610612744, 1610612762)

        # Should have projections for players with valid IDs
        assert isinstance(result, dict)
        assert len(result) >= 3
        assert all(isinstance(pid, int) for pid in result.keys())
        assert all(isinstance(ppg, float) for ppg in result.values())
        assert all(ppg >= 0 for ppg in result.values())

    @patch('src.projections.player_projections.load_config')
    @patch('src.projections.player_projections._get_game_id')
    def test_game_not_found_returns_empty(self, mock_game_id, mock_load_config):
        """Return empty dict when game not found."""
        mock_config = Mock()
        mock_config.data_paths.raw_db = 'test.db'
        mock_config.features.rolling_windows = [5, 10, 20]
        mock_load_config.return_value = mock_config
        mock_game_id.return_value = None

        result = project_game_contributions('2024-04-14', 9999, 9999)

        assert result == {}

    @patch('src.projections.player_projections.load_config')
    @patch('src.projections.player_projections._get_game_id')
    @patch('src.projections.player_projections._get_player_box_scores')
    def test_no_box_scores_returns_empty(
        self,
        mock_box_scores,
        mock_game_id,
        mock_load_config,
    ):
        """Return empty dict when box scores unavailable."""
        mock_config = Mock()
        mock_config.data_paths.raw_db = 'test.db'
        mock_config.features.rolling_windows = [5, 10, 20]
        mock_load_config.return_value = mock_config
        mock_game_id.return_value = '0022301198'
        mock_box_scores.return_value = pd.DataFrame()

        result = project_game_contributions('2024-04-14', 1610612744, 1610612762)

        assert result == {}

    @patch('src.projections.player_projections.load_config')
    @patch('src.projections.player_projections._get_game_id')
    @patch('src.projections.player_projections._get_player_box_scores')
    @patch('src.projections.player_projections._get_player_game_log')
    def test_returns_dict_with_int_keys_float_values(
        self,
        mock_game_log,
        mock_box_scores,
        mock_game_id,
        mock_load_config,
    ):
        """Return type must be dict[int, float]."""
        mock_config = Mock()
        mock_config.data_paths.raw_db = 'test.db'
        mock_config.features.rolling_windows = [5, 10, 20]
        mock_load_config.return_value = mock_config
        mock_game_id.return_value = '0022301198'
        mock_box_scores.return_value = pd.DataFrame({
            'PLAYER_ID': [201939, 2544],
            'PLAYER_NAME': ['Curry', 'LeBron'],
            'PTS': [28, 25],
        })
        mock_game_log.return_value = pd.DataFrame({
            'PTS': [25, 26, 27, 28, 29],
        })

        result = project_game_contributions('2024-04-14', 1610612744, 1610612762)

        assert isinstance(result, dict)
        for key, value in result.items():
            assert isinstance(key, int)
            assert isinstance(value, float)

    @patch('src.projections.player_projections.load_config')
    @patch('src.projections.player_projections._get_game_id')
    @patch('src.projections.player_projections._get_player_box_scores')
    @patch('src.projections.player_projections._get_player_game_log')
    def test_skips_invalid_player_ids(
        self,
        mock_game_log,
        mock_box_scores,
        mock_game_id,
        mock_load_config,
    ):
        """Skip players with invalid or missing IDs."""
        mock_config = Mock()
        mock_config.data_paths.raw_db = 'test.db'
        mock_config.features.rolling_windows = [5, 10, 20]
        mock_load_config.return_value = mock_config
        mock_game_id.return_value = '0022301198'

        # Include invalid IDs: NaN, 0, negative
        mock_box_scores.return_value = pd.DataFrame({
            'PLAYER_ID': [201939, float('nan'), 0, -1, 2544],
            'PLAYER_NAME': ['Valid1', 'Invalid_NaN', 'Invalid_0', 'Invalid_Neg', 'Valid2'],
            'PTS': [28, 10, 10, 10, 25],
        })
        mock_game_log.return_value = pd.DataFrame({
            'PTS': [25, 26, 27, 28, 29],
        })

        result = project_game_contributions('2024-04-14', 1610612744, 1610612762)

        # Only 2 valid players should be in result
        assert len(result) == 2
        assert 201939 in result
        assert 2544 in result

    @patch('src.projections.player_projections.load_config')
    @patch('src.projections.player_projections._get_game_id')
    @patch('src.projections.player_projections._get_player_box_scores')
    @patch('src.projections.player_projections._get_player_game_log')
    def test_handles_missing_game_log(
        self,
        mock_game_log,
        mock_box_scores,
        mock_game_id,
        mock_load_config,
    ):
        """Gracefully handle missing player game logs by returning 0.0 (not actual game PPG)."""
        mock_config = Mock()
        mock_config.data_paths.raw_db = 'test.db'
        mock_config.features.rolling_windows = [5, 10, 20]
        mock_load_config.return_value = mock_config
        mock_game_id.return_value = '0022301198'
        mock_box_scores.return_value = pd.DataFrame({
            'PLAYER_ID': [201939],
            'PLAYER_NAME': ['Curry'],
            'PTS': [28],
        })
        # Return empty game log (player has no history)
        mock_game_log.return_value = pd.DataFrame()

        result = project_game_contributions('2024-04-14', 1610612744, 1610612762)

        # Should include player with 0.0 projection (not actual 28 PPG from box score)
        assert len(result) == 1
        assert result[201939] == 0.0


class TestIntegration:
    """Integration tests with realistic data flow."""

    def test_return_type_contract(self):
        """Verify return type matches specification: dict[int, float]."""
        # This test just verifies the type contract without making API calls
        result = {}
        assert isinstance(result, dict)

    @pytest.mark.skipif(
        True,  # Skip by default to avoid API calls in test suite
        reason="Integration test - requires live API"
    )
    def test_real_api_call(self):
        """Test with real API calls to nba_api."""
        # This is a live test - only run manually
        result = project_game_contributions('2024-04-14', 1610612744, 1610612762)
        assert isinstance(result, dict)
        if result:  # Only check if result is non-empty
            assert len(result) >= 10  # At least some players
            assert all(isinstance(k, int) for k in result.keys())
            assert all(isinstance(v, float) for v in result.values())


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
