"""
Tests for `compute_elo_residual_form` (src/feature_engineering/elo.py): a
team's mean (actual margin - Elo-expected margin) over its last w games.

Ratings are hand-picked (bypassing compute_elo_ratings), so expected margins
and residuals are exact known numbers.
"""

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from src.feature_engineering.elo import compute_elo_residual_form

A, B, C = 100, 200, 300
HA, PER_POINT = 100.0, 25.0


def _games(rows, start=datetime(2024, 1, 1)):
    """rows: (home, away, home_elo, away_elo, point_diff), one game every 2 days."""
    games, ratings = [], []
    for i, (h, a, he, ae, diff) in enumerate(rows):
        gid = f"g{i}"
        date = start + timedelta(days=2 * i)
        games.append({"GAME_ID": gid, "GAME_DATE": date, "HOME_TEAM_ID": h, "AWAY_TEAM_ID": a, "POINT_DIFF": diff})
        ratings.append({"GAME_ID": gid, "home_team_elo": he, "away_team_elo": ae})
    return pd.DataFrame(games), pd.DataFrame(ratings)


def _run(rows, windows=(2,)):
    g, r = _games(rows)
    return compute_elo_residual_form(g, r, HA, PER_POINT, list(windows)).set_index("GAME_ID")


def test_residual_is_actual_minus_elo_expected_and_excludes_the_game_itself():
    # Equal ratings + home advantage 100 -> Elo expects home by 100/25 = 4.
    rows = [
        (A, B, 1500, 1500, 10),  # A residual +6
        (A, C, 1500, 1500, 0),  # A residual -4
        (A, B, 1500, 1500, 30),  # A's L2 here = mean(+6, -4) = +1, not using this game's +26
    ]
    out = _run(rows)
    assert np.isnan(out.loc["g0", "home_team_elo_resid_L2"])
    assert np.isnan(out.loc["g1", "home_team_elo_resid_L2"])  # only 1 prior game
    assert out.loc["g2", "home_team_elo_resid_L2"] == pytest.approx(1.0)


def test_rating_gap_sets_the_expectation():
    # Home 1550 vs away 1500: expected home margin (1550 + 100 - 1500) / 25 = 6.
    rows = [(A, B, 1550, 1500, 6), (A, C, 1550, 1500, 16), (A, B, 1550, 1500, 0)]
    assert _run(rows).loc["g2", "home_team_elo_resid_L2"] == pytest.approx((0 + 10) / 2)


def test_venue_blind_with_away_sign_flipped():
    # A plays home then away; as away team its residual is -(home residual).
    rows = [
        (A, B, 1500, 1500, 10),  # A home: +6
        (C, A, 1500, 1500, 10),  # A away: home C +6, so A -6
        (B, A, 1500, 1500, 0),  # A away again: L2 = mean(+6, -6) = 0
    ]
    out = _run(rows)
    assert out.loc["g2", "away_team_elo_resid_L2"] == pytest.approx(0.0)
    # B: g0 away (-6), g2 home -> L2 needs 2 priors, B has 1 -> NaN
    assert np.isnan(out.loc["g2", "home_team_elo_resid_L2"])


def test_diff_column_is_home_minus_away():
    rows = [
        (A, B, 1500, 1500, 10),  # A +6, B -6
        (A, B, 1500, 1500, 20),  # A +16, B -16
        (A, B, 1500, 1500, 0),
    ]
    out = _run(rows)
    assert out.loc["g2", "home_team_elo_resid_L2"] == pytest.approx(11.0)
    assert out.loc["g2", "away_team_elo_resid_L2"] == pytest.approx(-11.0)
    assert out.loc["g2", "elo_resid_diff_L2"] == pytest.approx(22.0)


def test_future_scores_never_change_past_features():
    base = [(A, B, 1500, 1500, 10), (A, C, 1500, 1500, 0), (A, B, 1500, 1500, 5), (A, C, 1500, 1500, 3)]
    changed = base[:3] + [(A, C, 1500, 1500, 99)]
    a, b = _run(base), _run(changed)
    pd.testing.assert_series_equal(a["home_team_elo_resid_L2"], b["home_team_elo_resid_L2"])
