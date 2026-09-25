"""
Prototype: conformalized quantile regression (CQR) as a second, independent
attempt at per-game (heteroscedastic) prediction intervals, after
scripts/prototype_conditional_intervals.py's secondary-residual-model
approach found ~zero correlation between its sigma_hat and actual test
error (i.e. no recoverable signal via that method).

CQR (Romano/Patterson/Candes 2019): train CatBoost's native MultiQuantile
loss to predict conditional quantiles (5th/50th/95th) directly through the
tree-splitting objective -- a different mechanism than regressing |residual|
as a separate target -- then apply a standard split-conformal correction on
top (conformalize) so the marginal coverage guarantee holds even if the raw
quantiles are mis-specified. Same point-in-time features as the champion
model; a fresh single-target model per fold per target (diff, total),
independent of the champion's own MultiRMSE point predictions -- diagnostic
only, doesn't touch training or point predictions used for val_score_mean.

Run: venv/bin/python3 scripts/prototype_cqr_intervals.py
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

import numpy as np  # noqa: E402
from catboost import CatBoostRegressor  # noqa: E402

from src.evaluation.conformal import conformal_quantile  # noqa: E402
from src.evaluation.cv_harness import run_split  # noqa: E402
from src.utils.config_loader import load_config  # noqa: E402

ALPHA = 0.1  # nominal 90% interval, matches config.prediction_intervals.alpha
QUANTILE_ALPHAS = (0.05, 0.5, 0.95)


def fit_quantile_model(X, y, config, seed):
    alphas_str = ",".join(str(a) for a in QUANTILE_ALPHAS)
    model = CatBoostRegressor(
        iterations=config.model.iterations,
        depth=config.model.depth,
        learning_rate=config.model.learning_rate,
        subsample=config.model.subsample,
        colsample_bylevel=config.model.colsample_bylevel,
        l2_leaf_reg=config.model.l2_leaf_reg,
        min_data_in_leaf=config.model.min_data_in_leaf,
        bootstrap_type="Bernoulli",
        loss_function=f"MultiQuantile:alpha={alphas_str}",
        random_seed=seed,
        verbose=False,
    )
    model.fit(X, y)
    return model


def evaluate_target(name, X_train, y_train, X_val, y_val, X_test, y_test, config, seed):
    model = fit_quantile_model(X_train, y_train, config, seed)

    q_val = model.predict(X_val)  # columns: [q05, q50, q95]
    q_test = model.predict(X_test)
    q_lo_val, q_med_val, q_hi_val = q_val[:, 0], q_val[:, 1], q_val[:, 2]
    q_lo_test, q_med_test, q_hi_test = q_test[:, 0], q_test[:, 1], q_test[:, 2]

    # CQR nonconformity score: how far outside [q_lo, q_hi] the true value
    # falls (negative if inside). conformal_quantile just takes an empirical
    # quantile of whatever array it's given, so it's reused here unchanged
    # even though its param is named for absolute residuals elsewhere.
    E_val = np.maximum(q_lo_val - y_val, y_val - q_hi_val)
    q_hat = conformal_quantile(E_val, ALPHA)

    lower, upper = q_lo_test - q_hat, q_hi_test + q_hat
    covered = (y_test >= lower) & (y_test <= upper)

    raw_width = q_hi_test - q_lo_test
    abs_err_from_median = np.abs(y_test - q_med_test)
    corr = float(np.corrcoef(raw_width, abs_err_from_median)[0, 1])

    # constant-width conformal baseline for comparison, calibrated around
    # this same quantile model's own median prediction on val/test.
    abs_resid_val = np.abs(y_val - q_med_val)
    q_hat_const = conformal_quantile(abs_resid_val, ALPHA)
    lower_const = q_med_test - q_hat_const
    upper_const = q_med_test + q_hat_const
    covered_const = (y_test >= lower_const) & (y_test <= upper_const)

    return {
        "q_hat": q_hat,
        "coverage": float(covered.mean()),
        "width_mean": float(np.mean(upper - lower)),
        "width_std": float(np.std(upper - lower)),
        "raw_width_vs_abs_err_corr": corr,
        "coverage_const": float(covered_const.mean()),
        "width_const_mean": float(np.mean(upper_const - lower_const)),
    }


def evaluate_fold(config, fold, lambda_weight: float = 0.5) -> dict:
    result = run_split(
        config,
        fold.train_end_date,
        fold.validation_start_date,
        fold.validation_end_date,
        fold.test_start_date,
        fold.test_end_date,
        lambda_weight=lambda_weight,
        keep_artifacts=True,
    )
    feature_cols = result.feature_cols
    home_col, away_col = config.features.targets[0], config.features.targets[1]
    train_features, val_features, test_features = (
        result.train_features,
        result.val_features,
        result.test_features,
    )

    X_train, X_val, X_test = (
        train_features[feature_cols],
        val_features[feature_cols],
        test_features[feature_cols],
    )

    diff_train = (train_features[home_col] - train_features[away_col]).to_numpy()
    diff_val = (val_features[home_col] - val_features[away_col]).to_numpy()
    diff_test = (test_features[home_col] - test_features[away_col]).to_numpy()
    total_train = (train_features[home_col] + train_features[away_col]).to_numpy()
    total_val = (val_features[home_col] + val_features[away_col]).to_numpy()
    total_test = (test_features[home_col] + test_features[away_col]).to_numpy()

    targets = {
        "diff": (diff_train, diff_val, diff_test),
        "total": (total_train, total_val, total_test),
    }

    return {
        name: evaluate_target(
            name, X_train, y_train, X_val, y_val, X_test, y_test, config, config.model.random_state
        )
        for name, (y_train, y_val, y_test) in targets.items()
    }


def main():
    config = load_config()
    folds = config.cv.folds
    per_fold = {}
    for fold in folds:
        print(f"=== {fold.name} ===")
        res = evaluate_fold(config, fold)
        per_fold[fold.name] = res
        for target_name, r in res.items():
            print(
                f"  [{target_name}] coverage={r['coverage']:.4f} (const={r['coverage_const']:.4f}) | "
                f"width_mean={r['width_mean']:.2f} (std={r['width_std']:.2f}, const={r['width_const_mean']:.2f}) | "
                f"raw_width~|err| corr={r['raw_width_vs_abs_err_corr']:.3f}"
            )

    print("\n=== Mean across folds ===")
    for target_name in ("diff", "total"):
        cov = np.mean([per_fold[f.name][target_name]["coverage"] for f in folds])
        cov_const = np.mean([per_fold[f.name][target_name]["coverage_const"] for f in folds])
        width = np.mean([per_fold[f.name][target_name]["width_mean"] for f in folds])
        width_const = np.mean([per_fold[f.name][target_name]["width_const_mean"] for f in folds])
        corr = np.mean([per_fold[f.name][target_name]["raw_width_vs_abs_err_corr"] for f in folds])
        print(
            f"[{target_name}] coverage={cov:.4f} (const={cov_const:.4f}) | "
            f"width={width:.2f} (const={width_const:.2f}) | mean raw_width~|err| corr={corr:.3f}"
        )


if __name__ == "__main__":
    main()
