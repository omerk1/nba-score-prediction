import numpy as np

from src.evaluation.conformal import (
    compute_interval_metrics,
    compute_prediction_intervals,
    conformal_quantile,
    evaluate_interval_coverage,
)


def test_conformal_quantile_matches_manual_calculation():
    # n=4, alpha=0.2 -> q_level = ceil(5*0.8)/4 = ceil(4.0)/4 = 1.0 -> max residual.
    residuals = np.array([1.0, 2.0, 3.0, 10.0])
    assert conformal_quantile(residuals, alpha=0.2) == 10.0

    # n=9, alpha=0.1 -> q_level = ceil(10*0.9)/9 = ceil(9.0)/9 = 1.0 -> max residual.
    residuals = np.arange(1, 10, dtype=float)  # 1..9
    assert conformal_quantile(residuals, alpha=0.1) == 9.0

    # n=9, alpha=0.5 -> q_level = ceil(10*0.5)/9 = ceil(5.0)/9 = 5/9.
    # "higher" interpolation on 1..9 at the 5/9 quantile -> 6.0.
    assert conformal_quantile(residuals, alpha=0.5) == 6.0


def test_compute_prediction_intervals_constant_width_and_centering():
    val_true = np.array([0.0, 0.0, 0.0, 0.0])
    val_pred = np.array([1.0, -1.0, 2.0, -2.0])  # abs residuals: 1,1,2,2
    test_pred = np.array([10.0, -5.0, 0.0])

    lower, upper, q_hat = compute_prediction_intervals(val_true, val_pred, test_pred, alpha=0.5)

    assert np.allclose(upper - lower, 2 * q_hat)
    assert np.allclose((lower + upper) / 2, test_pred)


def test_evaluate_interval_coverage():
    test_true = np.array([1.0, 5.0, 10.0, -3.0])
    lower = np.array([0.0, 0.0, 0.0, 0.0])
    upper = np.array([2.0, 2.0, 20.0, 2.0])
    # covered: 1 in [0,2] yes; 5 in [0,2] no; 10 in [0,20] yes; -3 in [0,2] no.
    result = evaluate_interval_coverage(test_true, lower, upper)

    assert result["coverage"] == 0.5
    assert result["mean_width"] == np.mean(upper - lower)


def test_compute_interval_metrics_shape_and_keys():
    n = 20
    rng = np.random.default_rng(0)
    diff_val_true = rng.normal(size=n)
    diff_val_pred = diff_val_true + rng.normal(scale=0.1, size=n)
    diff_test_true = rng.normal(size=n)
    diff_test_pred = diff_test_true + rng.normal(scale=0.1, size=n)
    total_val_true = rng.normal(loc=220, size=n)
    total_val_pred = total_val_true + rng.normal(scale=0.1, size=n)
    total_test_true = rng.normal(loc=220, size=n)
    total_test_pred = total_test_true + rng.normal(scale=0.1, size=n)

    metrics = compute_interval_metrics(
        diff_val_true,
        diff_val_pred,
        diff_test_true,
        diff_test_pred,
        total_val_true,
        total_val_pred,
        total_test_true,
        total_test_pred,
        alpha=0.1,
    )

    assert set(metrics.keys()) == {
        "alpha",
        "diff_q_hat",
        "diff_coverage",
        "diff_mean_width",
        "total_q_hat",
        "total_coverage",
        "total_mean_width",
    }
    assert metrics["alpha"] == 0.1
    assert 0.0 <= metrics["diff_coverage"] <= 1.0
    assert 0.0 <= metrics["total_coverage"] <= 1.0
    assert metrics["diff_mean_width"] > 0
    assert metrics["total_mean_width"] > 0
