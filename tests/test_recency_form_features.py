"""Recency-weighted form features: point-in-time safety, season-to-date
reset at the season boundary, and EWM weighting recent games more."""

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.feature_engineering.feature_builder import FeatureBuilder


def _cfg(enabled=True, halflife=2.0, season_to_date=True):
    cfg = MagicMock()
    cfg.recency_form = MagicMock(
        enabled=enabled, halflife_games=halflife, season_to_date_enabled=season_to_date
    )
    return cfg


def _games():
    # Team 1 alternates home/away vs team 2; season 1 it wins by 10 every game,
    # season 2 it loses by 10 every game.
    rows = []
    for i in range(8):
        season = 1 if i < 4 else 2
        margin_t1 = 10 if season == 1 else -10
        t1_home = i % 2 == 0
        home, away = (1, 2) if t1_home else (2, 1)
        diff = margin_t1 if t1_home else -margin_t1
        rows.append(
            {
                "GAME_ID": f"g{i}",
                "GAME_DATE": pd.Timestamp("2024-01-01") + pd.Timedelta(days=i),
                "SEASON_ID": season,
                "HOME_TEAM_ID": home,
                "AWAY_TEAM_ID": away,
                "PTS_home": 100 + max(diff, 0),
                "PTS_away": 100 + max(-diff, 0),
                "POINT_DIFF": diff,
            }
        )
    return pd.DataFrame(rows)


def _team1(out, col):
    """Team 1's value of `{prefix}_{col}` per game, whichever side it played."""
    return np.where(out["HOME_TEAM_ID"] == 1, out[f"home_team_{col}"], out[f"away_team_{col}"])


@patch("src.feature_engineering.feature_builder.load_config")
def test_disabled_adds_nothing(mock_cfg):
    mock_cfg.return_value = _cfg(enabled=False)
    df = _games()
    out = FeatureBuilder(rolling_windows=[3])._add_recency_form_features(df.copy())
    assert list(out.columns) == list(df.columns)


@patch("src.feature_engineering.feature_builder.load_config")
def test_point_in_time_and_season_reset(mock_cfg):
    mock_cfg.return_value = _cfg()
    out = FeatureBuilder(rolling_windows=[3])._add_recency_form_features(_games())

    games = _team1(out, "season_games")
    margin = _team1(out, "season_margin")
    assert list(games) == [0, 1, 2, 3, 0, 1, 2, 3]
    # First game of each season has no season-to-date history.
    assert np.isnan(margin[0]) and np.isnan(margin[4])
    assert margin[1:4].tolist() == [10, 10, 10]
    assert margin[5:8].tolist() == [-10, -10, -10]

    # EWM is shift(1): first game NaN, then only prior outcomes.
    ewm = _team1(out, "ewm_margin")
    assert np.isnan(ewm[0])
    assert ewm[4] == pytest.approx(10.0)  # all prior games were +10
    # Decays toward the new season's -10 rather than resetting.
    assert 10 > ewm[5] > ewm[6] > ewm[7] > -10


@patch("src.feature_engineering.feature_builder.load_config")
def test_own_outcome_never_used(mock_cfg):
    mock_cfg.return_value = _cfg()
    df = _games()
    base = FeatureBuilder(rolling_windows=[3])._add_recency_form_features(df.copy())
    df.loc[5, ["PTS_home", "PTS_away", "POINT_DIFF"]] = [150, 50, 100]
    changed = FeatureBuilder(rolling_windows=[3])._add_recency_form_features(df.copy())
    cols = [c for c in base.columns if "ewm" in c or "season_" in c]
    pd.testing.assert_frame_equal(base.loc[:5, cols], changed.loc[:5, cols])
    assert not base.loc[6, cols].equals(changed.loc[6, cols])
