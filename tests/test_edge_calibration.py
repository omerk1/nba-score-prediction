import datetime
import json

import numpy as np
import pandas as pd
import pytest

from src.serving import edge_calibration as E
from src.serving import recommend as R

DAY = datetime.date(2026, 10, 20)


def test_below_min_games_prices_at_the_line():
    w = E.fit_gap_weight(np.ones(10), np.ones(10), min_games=50, z=1.0)
    assert w.weight == 0.0 and np.isnan(w.slope) and w.n == 10


def test_fully_informative_gap_gets_full_weight():
    gap = np.linspace(-8, 8, 200)
    w = E.fit_gap_weight(gap, gap, min_games=50, z=1.0)
    assert w.slope == pytest.approx(1.0) and w.weight == pytest.approx(1.0)


def test_uninformative_gap_gets_zero_weight():
    rng = np.random.default_rng(0)
    gap = rng.normal(0, 4, 500)
    realised = rng.normal(0, 13, 500)  # the line already knew everything
    w = E.fit_gap_weight(gap, realised, min_games=50, z=1.0)
    assert abs(w.slope) < 0.3 and w.weight == 0.0


def test_partly_informative_gap_is_priced_below_its_estimate():
    rng = np.random.default_rng(1)
    gap = rng.normal(0, 4, 2000)
    realised = 0.6 * gap + rng.normal(0, 3, 2000)
    w = E.fit_gap_weight(gap, realised, min_games=50, z=1.0)
    assert w.slope == pytest.approx(0.6, abs=0.05)
    assert w.weight == pytest.approx(w.slope - w.se) and 0 < w.weight < w.slope


def test_shrink_toward_line():
    assert E.shrink_toward_line(10.0, 4.0, 0.0) == 4.0
    assert E.shrink_toward_line(10.0, 4.0, 1.0) == 10.0
    assert E.shrink_toward_line(10.0, 4.0, 0.5) == 7.0


def _log(model="m1", home=1):
    rec = {
        "predicted_diff": 3.0,
        "predicted_total": 230.0,
        "home_spread": -1.5,  # home favoured by 1.5: the line is a +1.5 home margin
        "home_cover_probability": 0.6,
        "away_cover_probability": 0.4,
        "home_spread_odds": 1.8,
        "away_spread_odds": 1.8,
        "home_spread_edge": 0.04,
        "away_spread_edge": -0.16,
        "total_line": 222.5,
        "over_probability": 0.6,
        "under_probability": 0.4,
        "over_odds": 1.8,
        "under_odds": 1.8,
        "over_edge": 0.04,
        "under_edge": -0.16,
    }
    return pd.DataFrame(
        [
            {
                "game_date": DAY.isoformat(),
                "home_team_id": home,
                "away_team_id": 2,
                "model_trained_at": model,
                "rec_json": json.dumps(rec),
            }
        ]
    )


def _results(home=1):
    return pd.DataFrame(
        [{"game_date": DAY, "home_team_id": home, "away_team_id": 2, "pts_home": 115, "pts_away": 110}]
    )


def test_line_games_measures_against_the_line():
    games = E.line_games(_log(), _results()).set_index("market")
    # Spread: predicted +3 vs line +1.5, actual +5.
    assert games.loc["spread", "gap"] == pytest.approx(1.5)
    assert games.loc["spread", "realised"] == pytest.approx(3.5)
    # Total: predicted 230 vs line 222.5, actual 225.
    assert games.loc["total", "gap"] == pytest.approx(7.5)
    assert games.loc["total", "realised"] == pytest.approx(2.5)


def test_only_the_current_models_games_count():
    log = pd.concat([_log("old", home=1), _log("new", home=3)])
    results = pd.concat([_results(home=1), _results(home=3)])
    weights = E.fit_gap_weights(log, results, "new", min_games=1, z=0.0)
    assert weights["spread"].n == 1 and weights["total"].n == 1
    assert E.fit_gap_weights(log, results, None, min_games=1, z=0.0)["spread"].n == 2


class _Predictor:
    feature_names = ["x"]

    def predict(self, X):
        return np.array([[120.0, 108.0]])  # home by 12, total 228


def _resources(weights):
    rng = np.random.default_rng(2)
    res = rng.normal(0, 13, 2000)
    return R.PredictionResources(
        predictor=_Predictor(),
        diff_residuals=res,
        diff_bandwidth=2.0,
        total_residuals=res,
        total_bandwidth=2.0,
        gap_weights=weights,
    )


@pytest.fixture
def _no_features(monkeypatch):
    monkeypatch.setattr(R, "build_live_game_features", lambda *a, **k: pd.DataFrame({"x": [0.0]}))


def test_weight_zero_prices_at_the_line_and_kills_the_edge(_no_features):
    kwargs = dict(home_spread=-4.5, home_spread_odds=1.8, away_spread_odds=1.8, total_line=220.5, over_odds=1.8, under_odds=1.8)
    raw = R.recommend_game(1, 2, _resources({}), **kwargs)
    cal = R.recommend_game(1, 2, _resources({"spread": 0.0, "total": 0.0}), **kwargs)
    assert raw["home_cover_probability"] > 0.65 and raw["home_spread_edge"] > 0.05
    assert cal["home_cover_probability"] == pytest.approx(0.5, abs=0.03)
    assert cal["home_spread_edge"] < 0 and cal["over_edge"] < 0
    assert cal["spread_gap_weight"] == 0.0 and raw["spread_gap_weight"] == 1.0
    # The point prediction itself is untouched; only its pricing changes.
    assert cal["predicted_diff"] == raw["predicted_diff"] == 12.0
