import numpy as np
import pandas as pd
import pytest

from src.models.residual_corrector import ResidualCorrector, build_inputs, rest_inputs, team_error_input


def _games(n=8):
    # Teams 1 and 2 alternate home/away, one game per day, one season.
    rows = []
    for i in range(n):
        home, away = (1, 2) if i % 2 == 0 else (2, 1)
        rows.append(
            (
                pd.Timestamp("2024-11-01") + pd.Timedelta(days=i),
                22024,
                home,
                away,
                10.0 if home == 1 else -10.0,
            )
        )
    return pd.DataFrame(
        rows, columns=["GAME_DATE", "SEASON_ID", "HOME_TEAM_ID", "AWAY_TEAM_ID", "POINT_DIFF"]
    )


def test_rest_inputs_are_home_minus_away_and_capped():
    df = pd.DataFrame(
        {
            "home_team_back_to_back": [1, 0],
            "away_team_back_to_back": [0, 1],
            "home_team_rest_days": [1, 9],
            "away_team_rest_days": [3, 1],
            "home_team_games_in_4_nights": [1, 0],
            "away_team_games_in_4_nights": [0, 0],
        }
    )
    X = rest_inputs(df)
    assert X.b2b_diff.tolist() == [1, -1]
    assert X.rest_diff.tolist() == [-2, 5]
    assert X.dense_diff.tolist() == [1, 0]


def test_team_error_uses_only_prior_games():
    g = _games()
    pred = np.zeros(len(g))  # model always predicts 0; team 1 wins every game by 10
    x = team_error_input(g, pred, window=10, min_games=3)
    # Fewer than 3 prior games for each team -> 0.
    assert (x[:3] == 0).all()
    # From game 4 on: team 1's errors are +10, team 2's -10, from each side.
    np.testing.assert_allclose(x[3:], np.where(g.HOME_TEAM_ID[3:] == 1, 20.0, -20.0))
    # Changing a later result never changes an earlier input.
    g2 = g.copy()
    g2.loc[7, "POINT_DIFF"] = 99.0
    np.testing.assert_allclose(team_error_input(g2, pred)[:8], x[:8])


def test_team_error_resets_each_season():
    g = _games()
    g.loc[4:, "SEASON_ID"] = 22025
    x = team_error_input(g, np.zeros(len(g)), min_games=3)
    assert (x[4:7] == 0).all()


def test_fit_recovers_coefficients_without_intercept():
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(5000, 2)), columns=["a", "b"])
    y = 1.5 * X.a - 0.5 * X.b + rng.normal(scale=0.1, size=5000)
    c = ResidualCorrector.fit(X, y, alpha=1.0)
    assert c.coef == pytest.approx((1.5, -0.5), abs=0.01)
    np.testing.assert_allclose(c.adjustment(X[["b", "a"]]), c.adjustment(X))


def test_build_inputs_columns():
    g = _games().assign(
        home_team_back_to_back=0,
        away_team_back_to_back=0,
        home_team_rest_days=1,
        away_team_rest_days=1,
        home_team_games_in_4_nights=0,
        away_team_games_in_4_nights=0,
    )
    assert list(build_inputs(g, np.zeros(len(g)), use_team_error=False).columns) == [
        "b2b_diff",
        "rest_diff",
        "dense_diff",
    ]
    assert "team_error_diff" in build_inputs(g, np.zeros(len(g)), use_team_error=True).columns
