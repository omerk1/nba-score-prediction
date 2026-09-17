"""Post-hoc split-conformal prediction intervals around ScorePredictor's
existing diff/total point predictions.

Purely additive/diagnostic: these functions never touch training or the
point predictions themselves. `q_hat` is calibrated from validation-fold
absolute residuals and applied as a constant-width band around the
(chronologically later) test-fold point predictions -- judged by coverage
and width, never by diff_mae/total_mae. See CLAUDE.md's ablation-gated
feature workflow: this is deliberately NOT that workflow, since it can't
move val_score_mean (predictions are unchanged).

Caveat, not solved here: split conformal's coverage guarantee assumes
exchangeability between the calibration (val) and test sets. Under this
project's expanding-window CV, val and test are different, chronologically
ordered seasons, so that assumption is formally violated -- the same
generalization gap already implicit in every val->test claim in this
codebase. Empirical test-fold coverage vs. the nominal rate is the check on
how much this matters in practice.
"""

import numpy as np


def conformal_quantile(abs_residuals: np.ndarray, alpha: float) -> float:
    """Split-conformal quantile of absolute residuals, with the standard
    finite-sample correction: the (n+1)(1-alpha)/n -th empirical quantile
    (clipped to 1.0), taken with the "higher" interpolation so the returned
    q_hat is achievable from the sample and coverage is never rounded down."""
    n = len(abs_residuals)
    q_level = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
    return float(np.quantile(abs_residuals, q_level, method="higher"))


def compute_prediction_intervals(
    val_true: np.ndarray, val_pred: np.ndarray, test_pred: np.ndarray, alpha: float
) -> tuple[np.ndarray, np.ndarray, float]:
    """Calibrates q_hat from validation absolute residuals, applies it as a
    constant-width band around the test point predictions. Returns
    (lower, upper, q_hat)."""
    q_hat = conformal_quantile(np.abs(np.asarray(val_true) - np.asarray(val_pred)), alpha)
    test_pred = np.asarray(test_pred)
    return test_pred - q_hat, test_pred + q_hat, q_hat


def evaluate_interval_coverage(test_true: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> dict:
    """Empirical coverage (fraction of true values inside [lower, upper]) and
    mean interval width on the test set."""
    test_true = np.asarray(test_true)
    covered = (test_true >= lower) & (test_true <= upper)
    return {
        "coverage": float(np.mean(covered)),
        "mean_width": float(np.mean(upper - lower)),
    }


def compute_interval_metrics(
    diff_val_true: np.ndarray,
    diff_val_pred: np.ndarray,
    diff_test_true: np.ndarray,
    diff_test_pred: np.ndarray,
    total_val_true: np.ndarray,
    total_val_pred: np.ndarray,
    total_test_true: np.ndarray,
    total_test_pred: np.ndarray,
    alpha: float,
) -> dict:
    """Calibrates + evaluates conformal intervals for both targets (point
    differential, total score). Flat dict, ready to log."""
    diff_lower, diff_upper, diff_q_hat = compute_prediction_intervals(
        diff_val_true, diff_val_pred, diff_test_pred, alpha
    )
    diff_coverage = evaluate_interval_coverage(diff_test_true, diff_lower, diff_upper)

    total_lower, total_upper, total_q_hat = compute_prediction_intervals(
        total_val_true, total_val_pred, total_test_pred, alpha
    )
    total_coverage = evaluate_interval_coverage(total_test_true, total_lower, total_upper)

    return {
        "alpha": alpha,
        "diff_q_hat": diff_q_hat,
        "diff_coverage": diff_coverage["coverage"],
        "diff_mean_width": diff_coverage["mean_width"],
        "total_q_hat": total_q_hat,
        "total_coverage": total_coverage["coverage"],
        "total_mean_width": total_coverage["mean_width"],
    }
