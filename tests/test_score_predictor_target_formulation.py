"""
Tests for ScorePredictor's target_formulation option (EXPERIMENTS.md section
3.3: home_away status quo vs. diff_total reformulation). See
ScorePredictor._to_training_targets/_from_training_targets for the mechanism
-- the model is fit on whatever space target_formulation calls for, but
predict()/evaluate()'s public contract always stays [home, away], so
cv_harness.py, predict_game.py, and naive_baseline_metrics need zero changes
regardless of mode.

Pure-transform tests (no model training, fast) cover the math directly.
TestScorePredictorIntegration trains tiny real CatBoost models on synthetic
data (few rows, 5 iterations) to prove the two modes actually wire up
end-to-end, including the critical regression guarantee: home_away mode
(default) must be BYTE-IDENTICAL to today's pre-existing behavior.
"""

import os
import tempfile

import numpy as np
import pandas as pd
import pytest

from src.models.score_predictor import ScorePredictor


def _make_predictor(**kwargs):
    return ScorePredictor(model_type="catboost", random_state=42, verbose=False, **kwargs)


class TestTargetTransforms:

    def test_home_away_passthrough(self):
        predictor = _make_predictor(target_formulation="home_away")
        y = pd.DataFrame({"home": [110.0, 95.0], "away": [102.0, 108.0]})
        fit_targets = predictor._to_training_targets(y)
        assert np.array_equal(fit_targets, y.to_numpy(dtype=float))

        raw = np.array([[111.0, 100.0], [90.0, 109.0]])
        assert np.array_equal(predictor._from_training_targets(raw), raw)

    def test_diff_total_round_trip_reconstructs_home_away_exactly(self):
        predictor = _make_predictor(target_formulation="diff_total", target_lambda_weight=0.5)
        y = pd.DataFrame({"home": [110.0, 95.0, 120.0], "away": [102.0, 108.0, 118.0]})

        fit_targets = predictor._to_training_targets(y)
        # fit_targets[:, 0] = diff, fit_targets[:, 1] = total * sqrt(0.5)
        expected_diff = y["home"] - y["away"]
        expected_total_scaled = (y["home"] + y["away"]) * np.sqrt(0.5)
        assert np.allclose(fit_targets[:, 0], expected_diff)
        assert np.allclose(fit_targets[:, 1], expected_total_scaled)

        reconstructed = predictor._from_training_targets(fit_targets)
        assert np.allclose(reconstructed[:, 0], y["home"])
        assert np.allclose(reconstructed[:, 1], y["away"])

    def test_diff_total_scaling_uses_configured_lambda_not_hardcoded(self):
        """A non-default lambda_weight must actually change the scaling --
        catches a hardcoded 0.5 regression."""
        y = pd.DataFrame({"home": [110.0], "away": [100.0]})
        p_default = _make_predictor(target_formulation="diff_total", target_lambda_weight=0.5)
        p_other = _make_predictor(target_formulation="diff_total", target_lambda_weight=0.2)

        total_scaled_default = p_default._to_training_targets(y)[0, 1]
        total_scaled_other = p_other._to_training_targets(y)[0, 1]
        assert not np.isclose(total_scaled_default, total_scaled_other)
        assert np.isclose(total_scaled_default, 210.0 * np.sqrt(0.5))
        assert np.isclose(total_scaled_other, 210.0 * np.sqrt(0.2))

        # And each still round-trips correctly with its own lambda.
        for p in (p_default, p_other):
            fit_targets = p._to_training_targets(y)
            reconstructed = p._from_training_targets(fit_targets)
            assert np.allclose(reconstructed[0], [110.0, 100.0])

    def test_invalid_target_formulation_rejected(self):
        with pytest.raises(ValueError, match="target_formulation"):
            _make_predictor(target_formulation="not_a_real_mode")

    def test_diff_total_requires_positive_lambda_weight(self):
        with pytest.raises(ValueError, match="target_lambda_weight"):
            _make_predictor(target_formulation="diff_total", target_lambda_weight=0.0)

    def test_home_away_is_the_default(self):
        predictor = ScorePredictor(model_type="catboost")
        assert predictor.target_formulation == "home_away"


def _synthetic_training_data(n=40, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(
        {
            "feat_a": rng.normal(size=n),
            "feat_b": rng.normal(size=n),
        }
    )
    home = 105 + 8 * X["feat_a"] + rng.normal(scale=2, size=n)
    away = 100 - 5 * X["feat_b"] + rng.normal(scale=2, size=n)
    y = pd.DataFrame({"home": home, "away": away})
    return X, y


class TestScorePredictorIntegration:
    """Small real CatBoost fits (few rows, few iterations) -- fast, but
    exercises the actual model.fit/predict path, not just the pure-python
    transform helpers above."""

    def test_home_away_mode_matches_default_byte_identical(self):
        """Explicit target_formulation='home_away' must be indistinguishable
        from omitting it entirely -- the whole point of this feature shipping
        disabled-by-default is that nothing changes for existing callers."""
        X, y = _synthetic_training_data()
        params = dict(model_type="catboost", random_state=42, iterations=5, verbose=False)

        p_explicit = ScorePredictor(target_formulation="home_away", **params)
        p_default = ScorePredictor(**params)

        p_explicit.train(X, y)
        p_default.train(X, y)

        pred_explicit = p_explicit.predict(X)
        pred_default = p_default.predict(X)
        assert np.array_equal(pred_explicit, pred_default)

    def test_diff_total_mode_predicts_correct_shape_and_finite_values(self):
        X, y = _synthetic_training_data()
        predictor = ScorePredictor(
            model_type="catboost",
            random_state=42,
            iterations=5,
            verbose=False,
            target_formulation="diff_total",
        )
        predictor.train(X, y)
        preds = predictor.predict(X)

        assert preds.shape == (len(X), 2)
        assert np.all(np.isfinite(preds))
        # Reconstructed home/away should be in the right ballpark (loose bound
        # -- this is a smoke test, not an accuracy assertion).
        assert preds[:, 0].mean() == pytest.approx(y["home"].mean(), abs=20)
        assert preds[:, 1].mean() == pytest.approx(y["away"].mean(), abs=20)

    def test_diff_total_mode_actually_differs_from_home_away_mode(self):
        """Guards against an accidental no-op: the two modes must fit
        genuinely different underlying models, not silently converge to the
        same predictions."""
        X, y = _synthetic_training_data()
        params = dict(model_type="catboost", random_state=42, iterations=5, verbose=False)

        p_home_away = ScorePredictor(target_formulation="home_away", **params)
        p_diff_total = ScorePredictor(target_formulation="diff_total", **params)
        p_home_away.train(X, y)
        p_diff_total.train(X, y)

        assert not np.array_equal(p_home_away.predict(X), p_diff_total.predict(X))

    def test_diff_total_evaluate_returns_same_metric_keys_as_home_away(self):
        """evaluate() itself needs zero mode-awareness -- it always sees
        predict()'s [home, away] output regardless of target_formulation."""
        X, y = _synthetic_training_data()
        predictor = ScorePredictor(
            model_type="catboost",
            random_state=42,
            iterations=5,
            verbose=False,
            target_formulation="diff_total",
        )
        predictor.train(X, y)
        metrics = predictor.evaluate(X, y)
        assert set(metrics.keys()) == {
            "diff_mae",
            "diff_rmse",
            "diff_within_3",
            "diff_within_5",
            "diff_within_10",
            "home_mae",
            "away_mae",
            "home_rmse",
            "away_rmse",
            "total_mae",
            "total_rmse",
            "win_accuracy",
            "brier_score",
            "diff_correlation",
        }
        assert all(np.isfinite(v) for v in metrics.values())

    def test_diff_total_determinism_two_independent_fits_are_byte_identical(self):
        """Same guarantee already verified for home_away mode's fixed
        random_state -- diff_total mode must be equally deterministic, not
        just accurate. Two fresh ScorePredictor instances, same data/params,
        never sharing any state."""
        X, y = _synthetic_training_data()
        params = dict(
            model_type="catboost",
            random_state=42,
            iterations=10,
            verbose=False,
            target_formulation="diff_total",
        )
        p1 = ScorePredictor(**params)
        p2 = ScorePredictor(**params)
        p1.train(X, y)
        p2.train(X, y)
        assert np.array_equal(p1.predict(X), p2.predict(X))

    def test_diff_total_survives_save_load_round_trip(self):
        """predict_game.py loads models via ScorePredictor.load() for live
        prediction -- target_formulation/target_lambda_weight must round-trip
        through save()/load() (they live in self.model_params, persisted as
        part of the pickled model_data dict) so a loaded diff_total model
        still correctly inverse-transforms its raw predictions."""
        X, y = _synthetic_training_data()
        predictor = ScorePredictor(
            model_type="catboost",
            random_state=42,
            iterations=5,
            verbose=False,
            target_formulation="diff_total",
            target_lambda_weight=0.5,
        )
        predictor.train(X, y)
        pred_before = predictor.predict(X)

        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "model.pkl")
            predictor.save(path)
            loaded = ScorePredictor.load(path)

        assert loaded.target_formulation == "diff_total"
        assert loaded.target_lambda_weight == 0.5
        assert np.array_equal(loaded.predict(X), pred_before)
