import datetime
import json
import types
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

import src.serving.recommend as R
import src.serving.residual_shadow as S
from src.models.residual_corrector import ResidualCorrector, latest_team_errors, team_error_input
from src.serving.slate_html import render_slate_html

TEAMS = {1: "Home", 2: "Away"}


def _games(n=9):
    rows = []
    for i in range(n):
        home, away = (1, 2) if i % 2 == 0 else (2, 1)
        rows.append((pd.Timestamp("2026-11-01") + pd.Timedelta(days=i), 22026, home, away, float(i)))
    return pd.DataFrame(
        rows, columns=["GAME_DATE", "SEASON_ID", "HOME_TEAM_ID", "AWAY_TEAM_ID", "POINT_DIFF"]
    )


def test_latest_team_errors_matches_what_the_next_game_would_see():
    g = _games()
    pred = np.linspace(-3, 3, len(g))
    latest = latest_team_errors(g.iloc[:-1], pred[:-1], window=4, min_games=3)
    x = team_error_input(g, pred, window=4, min_games=3)
    last = g.iloc[-1]
    assert latest[last.HOME_TEAM_ID] - latest[last.AWAY_TEAM_ID] == pytest.approx(x[-1])
    # Below min_games -> 0, as in the fit.
    assert latest_team_errors(g.iloc[:2], pred[:2], min_games=3) == {1: 0.0, 2: 0.0}


def test_corrector_round_trips_through_dict():
    c = ResidualCorrector(("a", "b"), (1.5, -0.25))
    assert ResidualCorrector.from_dict(json.loads(json.dumps(c.to_dict()))) == c


def test_season_start():
    assert S.season_start(datetime.date(2026, 11, 23)) == datetime.date(2026, 8, 1)
    assert S.season_start(datetime.date(2027, 3, 1)) == datetime.date(2026, 8, 1)


def _config(tmp_path, enabled=True, spec=None):
    path = tmp_path / "rc.json"
    if spec is not None:
        path.write_text(json.dumps(spec))
    return types.SimpleNamespace(
        residual_correction=types.SimpleNamespace(enabled=enabled, coefficients_path=str(path))
    )


def _spec(**over):
    spec = {
        "columns": ["b2b_diff", "rest_diff", "dense_diff"],
        "coef": [-2.0, 0.0, -3.0],
        "error_window": 10,
        "error_min_games": 3,
        "model_trained_at": "2026-09-25 07:08 UTC",
    }
    return {**spec, **over}


def _val():
    g = _games(40)
    z = np.zeros(len(g))
    return g.assign(
        home_team_back_to_back=z,
        away_team_back_to_back=z,
        home_team_rest_days=z + 2,
        away_team_rest_days=z + 2,
        home_team_games_in_4_nights=z,
        away_team_games_in_4_nights=z,
    )


@pytest.fixture
def _trained(monkeypatch):
    monkeypatch.setattr("src.serving.prediction_log._model_trained_at", lambda: "2026-09-25 07:08 UTC")


def test_load_correction_off_or_unconfigured_is_none(tmp_path, _trained):
    val = _val()
    assert (
        S.load_correction(None, _config(tmp_path, enabled=False, spec=_spec()), val, np.zeros(len(val)))
        is None
    )
    # A MagicMock config (as many tests build) must not switch it on.
    assert S.load_correction(None, MagicMock(), val, np.zeros(len(val))) is None


def test_load_correction_skips_a_stale_fit(tmp_path, _trained):
    val = _val()
    cfg = _config(tmp_path, spec=_spec(model_trained_at="2026-01-01 00:00 UTC"))
    assert S.load_correction(None, cfg, val, np.zeros(len(val))) is None
    cfg = _config(tmp_path, spec=_spec(error_window=5))
    assert S.load_correction(None, cfg, val, np.zeros(len(val))) is None
    # Missing file: logged, None.
    assert S.load_correction(None, _config(tmp_path), val, np.zeros(len(val))) is None


def test_load_correction_rest_only_needs_no_history(tmp_path, _trained):
    val = _val()
    corr = S.load_correction(None, _config(tmp_path, spec=_spec()), val, np.zeros(len(val)))
    assert corr is not None and corr.team_errors == {}
    np.testing.assert_allclose(corr.diff_residuals, val.POINT_DIFF)  # no rest gaps -> no adjustment


class _Predictor:
    feature_names = ["x"]

    def predict(self, X):
        return np.array([[112.0, 110.0]])  # home by 2


def _row(home_b2b=1):
    return pd.DataFrame(
        {
            "x": [0.0],
            "HOME_TEAM_ID": [1],
            "AWAY_TEAM_ID": [2],
            "home_team_back_to_back": [home_b2b],
            "away_team_back_to_back": [0],
            "home_team_rest_days": [1],
            "away_team_rest_days": [2],
            "home_team_games_in_4_nights": [0],
            "away_team_games_in_4_nights": [0],
        }
    )


def _resources(corrector, team_errors=None):
    res = np.random.default_rng(0).normal(0, 12, 3000)
    corr = S.CorrectionResources(corrector, team_errors or {}, res, 2.0, "2026-11-23")
    return R.PredictionResources(_Predictor(), res, 2.0, res, 2.0, correction=corr)


MARKETS = dict(
    home_spread=-1.5,
    home_spread_odds=1.9,
    away_spread_odds=1.9,
    home_moneyline_odds=1.8,
    away_moneyline_odds=2.0,
    total_line=220.5,
    over_odds=1.9,
    under_odds=1.9,
)


def test_corrected_entry_prices_like_the_live_model(monkeypatch):
    monkeypatch.setattr(R, "build_live_game_features", lambda *a, **k: _row())
    # Zero coefficients: the corrected view must equal the live one exactly.
    zero = R.recommend_game(1, 2, _resources(ResidualCorrector(("b2b_diff",), (0.0,))), **MARKETS)
    entry = zero["residual_correction"]
    for k in ("home_win_probability", "home_cover_probability", "home_spread_edge", "away_moneyline_edge"):
        assert entry[k] == pytest.approx(zero[k])

    corr = ResidualCorrector(("b2b_diff", "team_error_diff"), (-2.0, 0.5))
    rec = R.recommend_game(1, 2, _resources(corr, {1: 1.0, 2: -1.0}), **MARKETS)
    entry = rec["residual_correction"]
    assert entry["parts"] == {"b2b_diff": -2.0, "team_error_diff": 1.0}
    assert entry["adjustment"] == pytest.approx(-1.0)
    assert entry["predicted_diff"] == pytest.approx(1.0)
    assert entry["home_cover_probability"] < rec["home_cover_probability"]
    # Totals and the live fields are untouched; the entry is JSON-safe for the log.
    assert "over_edge" not in entry and rec["predicted_diff"] == 2.0
    json.dumps(rec)


def test_a_failing_correction_never_costs_the_pick(monkeypatch):
    monkeypatch.setattr(R, "build_live_game_features", lambda *a, **k: _row())
    monkeypatch.setattr(S, "corrected_entry", lambda *a, **k: 1 / 0)
    rec = R.recommend_game(1, 2, _resources(ResidualCorrector(("b2b_diff",), (-2.0,))), **MARKETS)
    assert "residual_correction" not in rec and "home_spread_edge" in rec


def _rec(home_edge, corrected_home_edge, parts=None):
    rec = {
        "home_team_id": 1,
        "away_team_id": 2,
        "predicted_home_score": 112.0,
        "predicted_away_score": 110.0,
        "predicted_diff": 2.0,
        "predicted_total": 222.0,
        "home_win_probability": 0.56,
        "home_spread": -1.5,
        "home_cover_probability": 0.5 + home_edge,
        "away_cover_probability": 0.5 - home_edge,
        "home_spread_market_probability": 0.5,
        "away_spread_market_probability": 0.5,
        "home_spread_odds": 2.0,
        "away_spread_odds": 2.0,
        "home_spread_edge": home_edge,
        "away_spread_edge": -home_edge,
    }
    if corrected_home_edge is not None:
        rec["residual_correction"] = {
            "parts": parts or {"b2b_diff": -2.0},
            "inputs": {"b2b_diff": 1.0},
            "home_cover_probability": 0.5 + corrected_home_edge,
            "away_cover_probability": 0.5 - corrected_home_edge,
            "home_spread_edge": corrected_home_edge,
            "away_spread_edge": -corrected_home_edge,
        }
    return rec


def test_slate_shows_two_full_sets_of_bets():
    page = render_slate_html([_rec(0.08, -0.06)], TEAMS, "Mon", 0.0)
    live, corrected = page.split("Corrected model (test) · Strong")
    assert "Live model · Strong" in live and "Home -1.5" in live
    assert "Away +1.5" in corrected and "Home -1.5" not in corrected.split("All games")[0]
    # Off or unavailable today: the page is exactly as before.
    plain = render_slate_html([_rec(0.08, None)], TEAMS, "Mon", 0.0)
    assert "Corrected" not in plain and "Live model" not in plain
    assert "Not available for 1 game(s)" in render_slate_html([_rec(0.08, 0.07), _rec(0.08, None)], TEAMS, "Mon", 0.0)


def test_caption_lists_both_models(monkeypatch):
    from src.serving import notify_telegram as T

    monkeypatch.setattr(T, "build_team_id_to_name", lambda: TEAMS)
    caption = T.format_recommended_summary([_rec(0.08, -0.06)], "Mon", 0.0)
    live, corrected = caption.split("Corrected model (test)")
    assert "Live model" in live and "Home -1.5" in live and "Away +1.5" in corrected
    assert "Corrected" not in T.format_recommended_summary([_rec(0.08, None)], "Mon", 0.0)
