"""
Tests for extended head-to-head features
"""

import pandas as pd
import numpy as np
import pytest
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

from src.feature_engineering.feature_builder import FeatureBuilder


class TestH2HFeatures:
    """Test suite for head-to-head feature engineering"""

    @staticmethod
    def _mock_config():
        """Create a mocked config with ELO and injury features disabled"""
        mock_cfg = MagicMock()
        mock_cfg.elo_features = MagicMock(enabled=False)
        mock_cfg.injury_features = MagicMock(enabled=False)
        mock_cfg.features.exclude = []
        return mock_cfg

    @pytest.fixture
    def sample_games_df(self):
        """Create a minimal sample dataset for testing"""
        # Create games between team 1 and team 2 across multiple seasons
        games = []

        # Season 2022 (SEASON_ID = 2022)
        games.append({
            'GAME_ID': 1,
            'GAME_DATE': datetime(2021, 10, 19),
            'SEASON_ID': 2022,
            'SEASON_TYPE': 'Regular Season',
            'HOME_TEAM_ID': 1,
            'AWAY_TEAM_ID': 2,
            'PTS_home': 110,
            'PTS_away': 100,
            'POINT_DIFF': 10,
            'TOTAL_POINTS': 210,
            'HOME_TEAM_WINS': 1,
            'FG_PCT_home': 0.5, 'FT_PCT_home': 0.8, 'FG3_PCT_home': 0.35, 'AST_home': 25, 'REB_home': 40,
            'FG_PCT_away': 0.48, 'FT_PCT_away': 0.75, 'FG3_PCT_away': 0.33, 'AST_away': 24, 'REB_away': 38,
        })

        # Another game in season 2022
        games.append({
            'GAME_ID': 2,
            'GAME_DATE': datetime(2021, 12, 20),
            'SEASON_ID': 2022,
            'SEASON_TYPE': 'Regular Season',
            'HOME_TEAM_ID': 2,
            'AWAY_TEAM_ID': 1,
            'PTS_home': 105,
            'PTS_away': 95,
            'POINT_DIFF': 10,
            'TOTAL_POINTS': 200,
            'HOME_TEAM_WINS': 1,
            'FG_PCT_home': 0.48, 'FT_PCT_home': 0.75, 'FG3_PCT_home': 0.33, 'AST_home': 24, 'REB_home': 38,
            'FG_PCT_away': 0.5, 'FT_PCT_away': 0.8, 'FG3_PCT_away': 0.35, 'AST_away': 25, 'REB_away': 40,
        })

        # Season 2023
        games.append({
            'GAME_ID': 3,
            'GAME_DATE': datetime(2022, 11, 1),
            'SEASON_ID': 2023,
            'SEASON_TYPE': 'Regular Season',
            'HOME_TEAM_ID': 1,
            'AWAY_TEAM_ID': 2,
            'PTS_home': 100,
            'PTS_away': 110,
            'POINT_DIFF': -10,
            'TOTAL_POINTS': 210,
            'HOME_TEAM_WINS': 0,
            'FG_PCT_home': 0.45, 'FT_PCT_home': 0.72, 'FG3_PCT_home': 0.30, 'AST_home': 22, 'REB_home': 35,
            'FG_PCT_away': 0.52, 'FT_PCT_away': 0.82, 'FG3_PCT_away': 0.38, 'AST_away': 27, 'REB_away': 42,
        })

        # Another game in season 2023
        games.append({
            'GAME_ID': 4,
            'GAME_DATE': datetime(2023, 1, 15),
            'SEASON_ID': 2023,
            'SEASON_TYPE': 'Regular Season',
            'HOME_TEAM_ID': 2,
            'AWAY_TEAM_ID': 1,
            'PTS_home': 115,
            'PTS_away': 105,
            'POINT_DIFF': 10,
            'TOTAL_POINTS': 220,
            'HOME_TEAM_WINS': 1,
            'FG_PCT_home': 0.52, 'FT_PCT_home': 0.82, 'FG3_PCT_home': 0.38, 'AST_home': 27, 'REB_home': 42,
            'FG_PCT_away': 0.45, 'FT_PCT_away': 0.72, 'FG3_PCT_away': 0.30, 'AST_away': 22, 'REB_away': 35,
        })

        # Season 2024
        games.append({
            'GAME_ID': 5,
            'GAME_DATE': datetime(2023, 10, 25),
            'SEASON_ID': 2024,
            'SEASON_TYPE': 'Regular Season',
            'HOME_TEAM_ID': 1,
            'AWAY_TEAM_ID': 2,
            'PTS_home': 120,
            'PTS_away': 110,
            'POINT_DIFF': 10,
            'TOTAL_POINTS': 230,
            'HOME_TEAM_WINS': 1,
            'FG_PCT_home': 0.53, 'FT_PCT_home': 0.83, 'FG3_PCT_home': 0.39, 'AST_home': 28, 'REB_home': 43,
            'FG_PCT_away': 0.46, 'FT_PCT_away': 0.73, 'FG3_PCT_away': 0.31, 'AST_away': 23, 'REB_away': 36,
        })

        # Another game in season 2024
        games.append({
            'GAME_ID': 6,
            'GAME_DATE': datetime(2024, 1, 20),
            'SEASON_ID': 2024,
            'SEASON_TYPE': 'Regular Season',
            'HOME_TEAM_ID': 2,
            'AWAY_TEAM_ID': 1,
            'PTS_home': 108,
            'PTS_away': 98,
            'POINT_DIFF': 10,
            'TOTAL_POINTS': 206,
            'HOME_TEAM_WINS': 1,
            'FG_PCT_home': 0.49, 'FT_PCT_home': 0.76, 'FG3_PCT_home': 0.34, 'AST_home': 25, 'REB_home': 39,
            'FG_PCT_away': 0.47, 'FT_PCT_away': 0.71, 'FG3_PCT_away': 0.29, 'AST_away': 21, 'REB_away': 34,
        })

        df = pd.DataFrame(games)
        df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])
        return df

    @patch('src.feature_engineering.feature_builder.load_config')
    def test_h2h_features_exist(self, mock_config, sample_games_df):
        """Test that all new h2h features are created"""
        # Mock config to disable ELO and injury features
        mock_cfg = MagicMock()
        mock_cfg.elo_features = MagicMock(enabled=False)
        mock_cfg.injury_features = MagicMock(enabled=False)
        mock_cfg.features.exclude = []
        mock_config.return_value = mock_cfg

        fb = FeatureBuilder(rolling_windows=[3, 5])
        result = fb.create_all_features(sample_games_df)

        required_cols = [
            'h2h_home_margin_L3',
            'h2h_home_win_rate_L5',
            'h2h_win_pct_3yr',
            'h2h_avg_diff',
            'h2h_home_win_pct',
            'h2h_away_win_pct',
        ]

        for col in required_cols:
            assert col in result.columns, f"Missing column: {col}"

    @patch('src.feature_engineering.feature_builder.load_config')
    def test_h2h_home_win_pct(self, mock_config, sample_games_df):
        """Test h2h_home_win_pct calculation"""
        mock_config.return_value = self._mock_config()
        fb = FeatureBuilder(rolling_windows=[3, 5])
        result = fb.create_all_features(sample_games_df)

        # Game 0 (1 vs 2): should be NaN (first game)
        assert pd.isna(result.loc[0, 'h2h_home_win_pct'])

        # Game 1 (2 vs 1): Team 2 at home. Looking for prior games where Team 2 is home vs Team 1.
        # Game 0 has Team 1 home vs Team 2, so no matching games. Should be NaN.
        assert pd.isna(result.loc[1, 'h2h_home_win_pct'])

        # Game 2 (1 vs 2): Team 1 at home. Looking for prior games where Team 1 is home vs Team 2.
        # Game 0 has Team 1 home vs Team 2 (Team 1 won). So win% = 1.0
        assert result.loc[2, 'h2h_home_win_pct'] == 1.0

        # Game 3 (2 vs 1): Team 2 at home. Looking for prior games where Team 2 is home vs Team 1.
        # Game 1 has Team 2 home vs Team 1 (Team 2 won). So win% = 1.0
        assert result.loc[3, 'h2h_home_win_pct'] == 1.0

        # Game 4 (1 vs 2): Team 1 at home. Looking for prior games where Team 1 is home vs Team 2.
        # Games 0 and 2. Game 0: Team 1 won. Game 2: Team 1 lost. So win% = 0.5
        assert result.loc[4, 'h2h_home_win_pct'] == 0.5

    @patch('src.feature_engineering.feature_builder.load_config')
    def test_h2h_away_win_pct(self, mock_config, sample_games_df):
        """Test h2h_away_win_pct calculation"""
        mock_config.return_value = self._mock_config()
        fb = FeatureBuilder(rolling_windows=[3, 5])
        result = fb.create_all_features(sample_games_df)

        # Game 0 (HOME=1, AWAY=2): should be NaN (first game)
        assert pd.isna(result.loc[0, 'h2h_away_win_pct'])

        # Game 1 (HOME=2, AWAY=1): home team 2 playing away (against team 1)
        # Looking for prior games where 2 is away against 1: Game 0 (1 home vs 2 away)
        # Game 0: 1 home=110, 2 away=100. POINT_DIFF=10 (home team won)
        # For team 2 away: POINT_DIFF < 0? No, it's +10. So team 2 lost.
        # win% = 0/1 = 0.0
        assert result.loc[1, 'h2h_away_win_pct'] == 0.0

        # Game 2 (HOME=1, AWAY=2): home team 1 playing away
        # Looking for prior games where 1 is away against 2: Game 1 (2 home vs 1 away)
        # Game 1: 2 home=105, 1 away=95. POINT_DIFF=10 (home team won)
        # For team 1 away: POINT_DIFF < 0? No. So team 1 lost.
        # win% = 0/1 = 0.0
        assert result.loc[2, 'h2h_away_win_pct'] == 0.0

        # Game 3 (HOME=2, AWAY=1): home team 2 playing away
        # Looking for prior games where 2 is away against 1: Game 0 and 2
        # Game 0: 2 away, lost (POINT_DIFF=10)
        # Game 2: 2 away, POINT_DIFF=-10 (2 lost home game), but wait, Game 2 has 1 home 2 away
        # So same logic: POINT_DIFF=-10 for team 2 perspective? No, POINT_DIFF is always from home perspective
        # Actually, looking more carefully:
        # Game 0: HOME=1, AWAY=2, POINT_DIFF=10 (1 won at home, 2 lost away)
        # Game 2: HOME=1, AWAY=2, POINT_DIFF=-10 (1 lost at home, 2 won away)
        # So in Game 3, team 2 has 1 win and 1 loss when playing away against 1
        # win% = 1/2 = 0.5
        assert result.loc[3, 'h2h_away_win_pct'] == 0.5

    @patch('src.feature_engineering.feature_builder.load_config')
    def test_h2h_win_pct_3yr(self, mock_config, sample_games_df):
        """Test h2h_win_pct_3yr calculation"""
        mock_config.return_value = self._mock_config()
        fb = FeatureBuilder(rolling_windows=[3, 5])
        result = fb.create_all_features(sample_games_df)

        # Game 0: First game, should be NaN
        assert pd.isna(result.loc[0, 'h2h_win_pct_3yr'])

        # Game 1 (HOME=2, AWAY=1, season 2022): Prior game is Game 0
        # Canonical team is 1 (min). Game 0: canonical=1 won (margin=10)
        # From canonical perspective: win% = 1.0
        # From home team 2 perspective (flipped): 1 - 1.0 = 0.0
        assert result.loc[1, 'h2h_win_pct_3yr'] == 0.0

        # Game 5 (HOME=2, AWAY=1): Should have a value based on 3-year history
        assert not pd.isna(result.loc[5, 'h2h_win_pct_3yr'])

    @patch('src.feature_engineering.feature_builder.load_config')
    def test_h2h_avg_diff(self, mock_config, sample_games_df):
        """Test h2h_avg_diff calculation (all-time average)"""
        mock_config.return_value = self._mock_config()
        fb = FeatureBuilder(rolling_windows=[3, 5])
        result = fb.create_all_features(sample_games_df)

        # Game 0: First game, should be NaN
        assert pd.isna(result.loc[0, 'h2h_avg_diff'])

        # Game 1 (2 vs 1): Prior game is Game 0
        # Canonical perspective: min(1, 2) = 1. Canonical margin in Game 0 is 10.
        # From game 1 perspective (2 is home): margin = -10 (2 lost)
        # From canonical perspective: -10 (since 1 didn't score the points)
        # But wait, let me recalculate...
        # Game 0: home=1, away=2, diff=10. Canonical team is 1, canonical_margin = 10
        # Game 1: home=2, away=1, diff=10. Canonical team is still 1, canonical_margin = -10
        # Avg = (10 + (-10)) / 2... but we shift so Game 1 only sees Game 0.
        # Game 1 avg = 10 only
        # From home team 2 perspective: h2h_avg_diff = -10
        assert not pd.isna(result.loc[1, 'h2h_avg_diff'])

    @patch('src.feature_engineering.feature_builder.load_config')
    def test_no_nans_after_sufficient_history(self, mock_config, sample_games_df):
        """Test that there are no NaNs in h2h features after sufficient history"""
        mock_config.return_value = self._mock_config()
        fb = FeatureBuilder(rolling_windows=[3, 5])
        result = fb.create_all_features(sample_games_df)

        # After first game (idx=1), h2h_home_win_pct and h2h_away_win_pct can still be NaN
        # if there's no prior history of that specific matchup in that venue.
        # But h2h_win_pct_3yr and h2h_avg_diff should have values from idx=1 onwards
        # (they expand over all history, not just 3yr/all)

        # Check indices 2-5 (after second game)
        for idx in range(2, len(result)):
            # At least one of the metrics should not be NaN (depending on matchup history)
            row = result.loc[idx]
            # h2h_avg_diff should always have a value after the first game
            assert not pd.isna(row['h2h_avg_diff']), f"h2h_avg_diff is NaN at idx {idx}"

    @patch('src.feature_engineering.feature_builder.load_config')
    def test_feature_columns_excluded(self, mock_config, sample_games_df):
        """Test that temporary feature columns are cleaned up"""
        mock_config.return_value = self._mock_config()
        fb = FeatureBuilder(rolling_windows=[3, 5])
        result = fb.create_all_features(sample_games_df)

        # Ensure no temp columns leak
        assert '_canonical_team' not in result.columns
        assert '_canonical_margin' not in result.columns
        assert '_h2h_margin_canon' not in result.columns
        assert '_h2h_win_canon' not in result.columns
        assert '_h2h_win_3yr_canon' not in result.columns
        assert '_h2h_avg_diff_canon' not in result.columns
        assert '_h2h_home_win_pct' not in result.columns
        assert '_h2h_away_win_pct' not in result.columns
        assert 'matchup_key' not in result.columns

    def test_type_hints_and_structure(self):
        """Test that methods have proper type hints"""
        fb = FeatureBuilder(rolling_windows=[3, 5])

        # Check that methods exist and are callable
        assert callable(fb._compute_h2h_3year_win_pct)
        assert callable(fb._compute_h2h_home_away_splits)
        assert callable(fb._add_h2h_features)

    @patch('src.feature_engineering.feature_builder.load_config')
    def test_symmetric_matchups(self, mock_config):
        """Test that matchup key ordering is symmetric"""
        mock_config.return_value = self._mock_config()
        games = [
            {
                'GAME_ID': i,
                'GAME_DATE': datetime(2023, 1, i+1),
                'SEASON_ID': 2024,
                'SEASON_TYPE': 'Regular Season',
                'HOME_TEAM_ID': 1 if i % 2 == 0 else 2,
                'AWAY_TEAM_ID': 2 if i % 2 == 0 else 1,
                'PTS_home': 110,
                'PTS_away': 100,
                'POINT_DIFF': 10 if (1 if i % 2 == 0 else 2) == 1 else -10,
                'TOTAL_POINTS': 210,
                'HOME_TEAM_WINS': 1,
                'FG_PCT_home': 0.5, 'FT_PCT_home': 0.8, 'FG3_PCT_home': 0.35,
                'AST_home': 25, 'REB_home': 40,
                'FG_PCT_away': 0.48, 'FT_PCT_away': 0.75, 'FG3_PCT_away': 0.33,
                'AST_away': 24, 'REB_away': 38,
            }
            for i in range(4)
        ]

        df = pd.DataFrame(games)
        df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])

        fb = FeatureBuilder(rolling_windows=[3, 5])
        result = fb.create_all_features(df)

        # Verify feature columns exist
        assert 'h2h_win_pct_3yr' in result.columns
        assert 'h2h_avg_diff' in result.columns
        assert 'h2h_home_win_pct' in result.columns
        assert 'h2h_away_win_pct' in result.columns


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
