"""
Regression test: rest_days/back_to_back/games_in_4_nights (_add_rest_features) must be
computed venue-blind -- a team's true rest is time since its LAST game regardless of
home/away role. The prior implementation grouped directly on HOME_TEAM_ID/AWAY_TEAM_ID,
which silently skipped any interleaved game the team played in the other role, so a team
coming off a genuine back-to-back could be reported as fully rested if the second game
happened to be in the other venue role from a prior lookback point. Verified against real
data: 45.3% of home-role rows had a wrong rest_days value, and back_to_back missed 6.8% of
true back-to-backs (always false negatives, never false positives).
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


def _make_game(game_id, game_date, home_team_id, away_team_id, home_pts=100, away_pts=90):
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


class TestRestFeaturesVenueBlind:

    @patch("src.feature_engineering.feature_builder.load_config")
    def test_back_to_back_detected_across_venue_roles(self, mock_config):
        """
        Team 100 plays away on day 1, then home on day 2 (a genuine back-to-back
        that spans both venue roles). The venue-scoped groupby (grouping only on
        HOME_TEAM_ID for the home_team_* columns) would never see team 100's day-1
        away game, and would instead look back to some earlier home game -- reporting
        artificially high rest instead of a back-to-back.
        """
        mock_config.return_value = _mock_config()

        games = [
            _make_game(1, datetime(2023, 10, 1), 200, 100),  # team100 away, day 1
            _make_game(2, datetime(2023, 10, 2), 100, 300),  # team100 home, day 2 -- true b2b
        ]
        df = pd.DataFrame(games)
        df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])

        fb = FeatureBuilder(rolling_windows=[3])
        result = fb.create_all_features(df)

        row = result.loc[result["GAME_ID"] == 2]
        assert row["home_team_rest_days"].iloc[0] == pytest.approx(1.0)
        assert row["home_team_back_to_back"].iloc[0] == 1

    @patch("src.feature_engineering.feature_builder.load_config")
    def test_first_game_defaults_to_three_days_rest(self, mock_config):
        """A team's first-ever game has no prior history -- rest_days defaults to 3
        (the pre-existing fillna convention), not NaN and not flagged as a back-to-back."""
        mock_config.return_value = _mock_config()

        games = [_make_game(1, datetime(2023, 10, 1), 100, 200)]
        df = pd.DataFrame(games)
        df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])

        fb = FeatureBuilder(rolling_windows=[3])
        result = fb.create_all_features(df)

        assert result["home_team_rest_days"].iloc[0] == pytest.approx(3.0)
        assert result["away_team_rest_days"].iloc[0] == pytest.approx(3.0)
        assert result["home_team_back_to_back"].iloc[0] == 0
        assert result["away_team_back_to_back"].iloc[0] == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
