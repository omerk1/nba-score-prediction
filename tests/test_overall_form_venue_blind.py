"""
Regression test: the venue-scoped rolling win_pct/diff_avg features
(home_team_win_pct_L{w}, computed only over a team's *home* games) do not
capture a team's overall recent form across both venues. This test checks
the new venue-blind counterparts (*_win_pct_overall_L{w}, *_diff_avg_overall_L{w})
correctly roll over a team's last N games regardless of home/away role.
"""

from datetime import datetime
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.feature_engineering.feature_builder import FeatureBuilder


def _mock_config():
    mock_cfg = MagicMock()
    mock_cfg.elo_features = MagicMock(enabled=False)
    mock_cfg.injury_features = MagicMock(enabled=False)
    mock_cfg.style_matchup = MagicMock(enabled=False, raw_features_enabled=False)
    mock_cfg.features.exclude = []
    return mock_cfg


def _make_game(game_id, game_date, home_team_id, away_team_id, home_pts, away_pts):
    return {
        "GAME_ID": game_id,
        "GAME_DATE": game_date,
        "SEASON_ID": 2024,
        "SEASON_TYPE": "Regular Season",
        "HOME_TEAM_ID": home_team_id,
        "AWAY_TEAM_ID": away_team_id,
        "PTS_home": home_pts,
        "PTS_away": away_pts,
        "POINT_DIFF": home_pts - away_pts,
        "TOTAL_POINTS": home_pts + away_pts,
        "HOME_TEAM_WINS": int(home_pts > away_pts),
    }


class TestVenueBlindOverallForm:

    @patch("src.feature_engineering.feature_builder.load_config")
    def test_overall_form_mixes_home_and_away_games(self, mock_config):
        """
        Team 100's first 3 games, alternating home/away (all vs distinct
        opponents so venue-scoped groupby only ever sees one of these games):
          Game 1 (home): won by 10
          Game 2 (away): lost by 4 (point diff from team 100's perspective: -4)
          Game 3 (home): won by 6

        Game 4 (team 100 at home) measures the L3 overall window over games 1-3:
          win_pct_overall = 2/3 (won games 1 and 3, lost game 2)
          diff_avg_overall = (10 - 4 + 6) / 3 = 4.0

        The venue-scoped home_team_win_pct_L3/diff_avg_L3 at game 4 would only
        see games 1 and 3 (team 100's prior *home* games) — this test targets
        the venue-blind counterpart instead.
        """
        mock_config.return_value = _mock_config()

        games = [
            _make_game(1, datetime(2023, 10, 1), 100, 200, home_pts=110, away_pts=100),  # team100 home, +10
            _make_game(2, datetime(2023, 10, 3), 300, 100, home_pts=104, away_pts=100),  # team100 away, -4
            _make_game(3, datetime(2023, 10, 5), 100, 400, home_pts=106, away_pts=100),  # team100 home, +6
            _make_game(
                4, datetime(2023, 10, 7), 100, 500, home_pts=100, away_pts=90
            ),  # team100 home, measure here
        ]
        df = pd.DataFrame(games)
        df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])

        fb = FeatureBuilder(rolling_windows=[3])
        result = fb.create_all_features(df)

        row = result.loc[result["GAME_ID"] == 4]
        assert row["home_team_win_pct_overall_L3"].iloc[0] == pytest.approx(2 / 3)
        assert row["home_team_diff_avg_overall_L3"].iloc[0] == pytest.approx(4.0)

        # Venue-scoped feature (unchanged, still present) only sees team100's prior
        # *home* games (1 and 3), confirming the two features genuinely differ.
        assert row["home_team_win_pct_L3"].iloc[0] == pytest.approx(1.0)
        assert row["home_team_diff_avg_L3"].iloc[0] == pytest.approx(8.0)

    @patch("src.feature_engineering.feature_builder.load_config")
    def test_no_leakage_first_game_is_nan(self, mock_config):
        """A team's first-ever game has no prior history — overall form must be NaN,
        matching the shift(1) leakage-safety pattern used everywhere else."""
        mock_config.return_value = _mock_config()

        games = [
            _make_game(1, datetime(2023, 10, 1), 100, 200, home_pts=110, away_pts=100),
            # Unrelated game between other teams -- gives HOME_TEAM_ID/AWAY_TEAM_ID more
            # than one distinct group, needed to avoid a pre-existing, unrelated quirk in
            # _add_rest_features's games_in_4_nights groupby(...).apply(...) (see
            # tests/test_rolling_shooting_pct.py's _noise_game docstring).
            _make_game(2, datetime(2023, 10, 2), 999, 300, home_pts=100, away_pts=95),
        ]
        df = pd.DataFrame(games)
        df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])

        fb = FeatureBuilder(rolling_windows=[3])
        result = fb.create_all_features(df)

        assert pd.isna(result["home_team_win_pct_overall_L3"].iloc[0])
        assert pd.isna(result["away_team_win_pct_overall_L3"].iloc[0])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
