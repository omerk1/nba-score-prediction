"""
Regression test: rolling FG_PCT/FT_PCT/fg3_pct features must be
volume-weighted (sum of makes / sum of attempts over the window), not a naive mean of
per-game percentages. A naive mean lets a low-attempt outlier game (e.g. 1-for-2) swing
the rolling average exactly as much as a normal-volume game (e.g. 15-for-40), which is
statistically wrong.
"""

import pandas as pd
import pytest
from datetime import datetime
from unittest.mock import MagicMock, patch

from src.feature_engineering.feature_builder import FeatureBuilder


def _mock_config():
    mock_cfg = MagicMock()
    mock_cfg.elo_features = MagicMock(enabled=False)
    mock_cfg.injury_features = MagicMock(enabled=False)
    mock_cfg.style_matchup = MagicMock(enabled=False, raw_features_enabled=False)
    mock_cfg.features.exclude = []
    return mock_cfg


def _noise_game(game_id, game_date):
    """
    An unrelated game between two other teams (999 home vs 300 away).

    Purely there to give HOME_TEAM_ID/AWAY_TEAM_ID more than one distinct group: with
    only a single group, pandas' groupby(...).apply(...) in _add_rest_features's
    games_in_4_nights (unrelated to this fix) returns a wide DataFrame instead of a
    flat Series and raises. That's a pre-existing quirk unrelated to this fix — this dummy
    row just keeps these fixtures out of its way without touching that code.
    """
    return _make_game(game_id, game_date, 999, 300, fgm_home=10, fga_home=20)


def _make_game(game_id, game_date, home_team_id, away_team_id, fgm_home, fga_home):
    """Minimal game row. Only home-team FG makes/attempts vary meaningfully across
    fixtures below — everything else (away team, FT, 3PT) is held constant/trivial."""
    fg_pct_home = fgm_home / fga_home
    return {
        'GAME_ID': game_id,
        'GAME_DATE': game_date,
        'SEASON_ID': 2024,
        'SEASON_TYPE': 'Regular Season',
        'HOME_TEAM_ID': home_team_id,
        'AWAY_TEAM_ID': away_team_id,
        'PTS_home': 100,
        'PTS_away': 95,
        'POINT_DIFF': 5,
        'TOTAL_POINTS': 195,
        'HOME_TEAM_WINS': 1,
        'FG_PCT_home': fg_pct_home, 'FT_PCT_home': 0.8, 'FG3_PCT_home': 0.35,
        'AST_home': 20, 'REB_home': 40,
        'FG_PCT_away': 0.45, 'FT_PCT_away': 0.75, 'FG3_PCT_away': 0.33,
        'AST_away': 20, 'REB_away': 38,
        'FGM_home': fgm_home, 'FGA_home': fga_home,
        'FTM_home': 16, 'FTA_home': 20,
        'FG3M_home': 7, 'FG3A_home': 20,
        'FGM_away': 36, 'FGA_away': 80,
        'FTM_away': 15, 'FTA_away': 20,
        'FG3M_away': 6.6, 'FG3A_away': 20,
    }


class TestVolumeWeightedRollingPct:

    @patch('src.feature_engineering.feature_builder.load_config')
    def test_weighted_differs_from_naive_mean_on_low_volume_outlier(self, mock_config):
        """
        Team 100's first 3 home games (all vs distinct opponents, so groupby(HOME_TEAM_ID)
        rolls only over team 100's own games):
          Game 1: 10-for-20  (50.0%, normal volume)
          Game 2: 1-for-1    (100.0%, extreme low-attempt outlier)
          Game 3: 10-for-20  (50.0%, normal volume)

        Game 4 (also team 100 at home) measures the rolling L3 window over games 1-3:
          naive mean-of-pct  = (0.50 + 1.00 + 0.50) / 3 = 0.6667
          volume-weighted    = (10 + 1 + 10) / (20 + 1 + 20) = 21/41 = 0.5122

        These differ by >0.15 (15 percentage points) — proof the outlier's influence is
        actually reduced by weighting, not just numerically different by noise.
        """
        mock_config.return_value = _mock_config()

        games = [
            _make_game(1, datetime(2023, 10, 1), 100, 200, fgm_home=10, fga_home=20),
            _make_game(2, datetime(2023, 10, 3), 100, 200, fgm_home=1, fga_home=1),
            _make_game(3, datetime(2023, 10, 5), 100, 200, fgm_home=10, fga_home=20),
            _make_game(4, datetime(2023, 10, 7), 100, 200, fgm_home=12, fga_home=22),
            _noise_game(99, datetime(2023, 10, 2)),
        ]
        df = pd.DataFrame(games)
        df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])

        fb = FeatureBuilder(rolling_windows=[3])
        result = fb.create_all_features(df)

        naive_mean = (0.50 + 1.00 + 0.50) / 3
        weighted = (10 + 1 + 10) / (20 + 1 + 20)

        actual = result.loc[result['GAME_ID'] == 4, 'home_team_fg_pct_L3'].iloc[0]

        # The fix must match the volume-weighted value, not the naive per-game mean.
        assert actual == pytest.approx(weighted, abs=1e-9)
        assert actual != pytest.approx(naive_mean, abs=1e-6)
        # And the two candidate formulas must themselves differ meaningfully — otherwise
        # this fixture wouldn't actually be exercising the bug.
        assert abs(naive_mean - weighted) > 0.15

    @patch('src.feature_engineering.feature_builder.load_config')
    def test_fg3_pct_is_also_volume_weighted(self, mock_config):
        """Same check for home_team_fg3_pct_L{window} (_add_style_features), which uses
        FG3M/FG3A rather than FGM/FGA."""
        mock_config.return_value = _mock_config()

        def game(game_id, date, fg3m, fg3a):
            g = _make_game(game_id, date, 100, 200, fgm_home=10, fga_home=20)
            g['FG3M_home'] = fg3m
            g['FG3A_home'] = fg3a
            g['FG3_PCT_home'] = fg3m / fg3a
            return g

        games = [
            game(1, datetime(2023, 10, 1), fg3m=5, fg3a=10),   # 50%, normal volume
            game(2, datetime(2023, 10, 3), fg3m=1, fg3a=1),    # 100%, low-attempt outlier
            game(3, datetime(2023, 10, 5), fg3m=5, fg3a=10),   # 50%, normal volume
            game(4, datetime(2023, 10, 7), fg3m=6, fg3a=12),
            _noise_game(99, datetime(2023, 10, 2)),
        ]
        df = pd.DataFrame(games)
        df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])

        fb = FeatureBuilder(rolling_windows=[3])
        result = fb.create_all_features(df)

        naive_mean = (0.50 + 1.00 + 0.50) / 3
        weighted = (5 + 1 + 5) / (10 + 1 + 10)

        actual = result.loc[result['GAME_ID'] == 4, 'home_team_fg3_pct_L3'].iloc[0]

        assert actual == pytest.approx(weighted, abs=1e-9)
        assert actual != pytest.approx(naive_mean, abs=1e-6)

    @patch('src.feature_engineering.feature_builder.load_config')
    def test_missing_makes_attempts_columns_skips_feature_gracefully(self, mock_config):
        """If FGM/FGA (or FG3M/FG3A) columns aren't present, the rolling pct feature
        should simply be omitted rather than raising — matches the existing guarded
        pattern used for other optional columns in feature_builder.py."""
        mock_config.return_value = _mock_config()

        games = []
        for i, (date, home, away) in enumerate([
            (datetime(2023, 10, 1), 100, 200),
            (datetime(2023, 10, 3), 100, 200),
            (datetime(2023, 10, 2), 999, 300),  # unrelated game — see _noise_game's docstring
        ]):
            games.append({
                'GAME_ID': i + 1,
                'GAME_DATE': date,
                'SEASON_ID': 2024,
                'SEASON_TYPE': 'Regular Season',
                'HOME_TEAM_ID': home,
                'AWAY_TEAM_ID': away,
                'PTS_home': 100,
                'PTS_away': 95,
                'POINT_DIFF': 5,
                'TOTAL_POINTS': 195,
                'HOME_TEAM_WINS': 1,
            })
        df = pd.DataFrame(games)
        df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])

        fb = FeatureBuilder(rolling_windows=[3])
        result = fb.create_all_features(df)  # should not raise

        assert 'home_team_fg_pct_L3' not in result.columns
        assert 'home_team_fg3_pct_L3' not in result.columns


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
