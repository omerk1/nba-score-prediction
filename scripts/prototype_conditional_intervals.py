"""
Prototype: locally-weighted (heteroscedastic) conformal intervals, as a
diagnostic comparison against the existing constant-width post-hoc conformal
intervals (src/evaluation/conformal.py, config.prediction_intervals).

Motivation: CatBoost's native uncertainty estimation (virtual ensembles) does
NOT support MultiRMSE -- confirmed empirically (CatBoostError: "unsupported
loss function for uncertainty MultiRMSE") -- so it can't be used directly on
the champion model (model.target_formulation=diff_total, loss_function=
MultiRMSE). This script instead fits a small secondary CatBoost model per
fold that predicts each game's |residual| (an uncertainty proxy, sigma_hat)
from the same point-in-time features, then conformalizes the NORMALIZED
residual (|actual - pred| / sigma_hat) instead of the raw residual. This
keeps the champion model completely untouched (purely additive/diagnostic,
same as the existing conformal module) while letting interval width vary
per game instead of being fixed.

To avoid the secondary model calibrating on the same residuals it was
fit on, each fold's validation season is split chronologically in half:
first half fits sigma_hat, second half calibrates the conformal quantile
(mirrors the existing module's val->test calibrate/apply split, one level
deeper).

Diagnostic only -- not wired into train_model.py, does not touch
val_score_mean/test_score_mean. Run: venv/bin/python3 scripts/prototype_conditional_intervals.py
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from catboost import CatBoostRegressor  # noqa: E402

from src.evaluation.conformal import conformal_quantile  # noqa: E402
from src.evaluation.cv_harness import run_split  # noqa: E402
from src.utils.config_loader import load_config  # noqa: E402

ALPHA = 0.1  # matches config.prediction_intervals.alpha


def fit_sigma_model(X: pd.DataFrame, abs_resid: np.ndarray, seed: int) -> CatBoostRegressor:
    """Small, shallow single-target model predicting |residual| from features
    -- deliberately low-capacity (a few hundred fit-half rows per fold) to
    avoid overfitting the uncertainty signal itself."""
    model = CatBoostRegressor(
        iterations=200,
        depth=3,
        learning_rate=0.1,
        loss_function="RMSE",
        random_seed=seed,
        verbose=False,
    )
    model.fit(X, abs_resid)
    return model


def evaluate_target(
    name: str,
    X_val: pd.DataFrame,
    val_true: np.ndarray,
    val_pred: np.ndarray,
    X_test: pd.DataFrame,
    test_true: np.ndarray,
    test_pred: np.ndarray,
    idx_fit: np.ndarray,
    idx_cal: np.ndarray,
    seed: int,
) -> dict:
    resid = val_true - val_pred
    abs_resid = np.abs(resid)

    sigma_model = fit_sigma_model(X_val.iloc[idx_fit], abs_resid[idx_fit], seed)
    floor = max(float(np.quantile(abs_resid[idx_fit], 0.05)), 0.5)

    sigma_cal = np.maximum(sigma_model.predict(X_val.iloc[idx_cal]), floor)
    norm_score_cal = abs_resid[idx_cal] / sigma_cal
    q_hat_lw = conformal_quantile(norm_score_cal, ALPHA)
    q_hat_const = conformal_quantile(abs_resid[idx_cal], ALPHA)

    sigma_test = np.maximum(sigma_model.predict(X_test), floor)
    lower_lw, upper_lw = test_pred - q_hat_lw * sigma_test, test_pred + q_hat_lw * sigma_test
    lower_const, upper_const = test_pred - q_hat_const, test_pred + q_hat_const

    covered_lw = (test_true >= lower_lw) & (test_true <= upper_lw)
    covered_const = (test_true >= lower_const) & (test_true <= upper_const)
    abs_resid_test = np.abs(test_true - test_pred)
    corr = float(np.corrcoef(sigma_test, abs_resid_test)[0, 1])

    terciles = pd.qcut(sigma_test, 3, labels=["low_unc", "med_unc", "high_unc"], duplicates="drop")
    buckets = {}
    for b in pd.unique(terciles):
        m = np.asarray(terciles == b)
        buckets[b] = {
            "n": int(m.sum()),
            "coverage_lw": float(covered_lw[m].mean()),
            "width_lw_mean": float(np.mean(upper_lw[m] - lower_lw[m])),
            "coverage_const": float(covered_const[m].mean()),
        }

    return {
        "q_hat_lw": q_hat_lw,
        "q_hat_const": q_hat_const,
        "coverage_lw": float(covered_lw.mean()),
        "width_lw_mean": float(np.mean(upper_lw - lower_lw)),
        "width_lw_std": float(np.std(upper_lw - lower_lw)),
        "coverage_const": float(covered_const.mean()),
        "width_const_mean": float(np.mean(upper_const - lower_const)),
        "sigma_vs_abs_resid_corr": corr,
        "buckets": buckets,
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
    predictor = result.predictor
    val_features, test_features = result.val_features, result.test_features
    feature_cols = result.feature_cols
    home_col, away_col = config.features.targets[0], config.features.targets[1]

    X_val, X_test = val_features[feature_cols], test_features[feature_cols]
    val_pred, test_pred = predictor.predict(X_val), predictor.predict(X_test)

    diff_val_true = (val_features[home_col] - val_features[away_col]).to_numpy()
    total_val_true = (val_features[home_col] + val_features[away_col]).to_numpy()
    diff_test_true = (test_features[home_col] - test_features[away_col]).to_numpy()
    total_test_true = (test_features[home_col] + test_features[away_col]).to_numpy()

    order = np.argsort(val_features["GAME_DATE"].to_numpy())
    half = len(order) // 2
    idx_fit, idx_cal = order[:half], order[half:]

    targets = {
        "diff": (
            diff_val_true,
            val_pred[:, 0] - val_pred[:, 1],
            diff_test_true,
            test_pred[:, 0] - test_pred[:, 1],
        ),
        "total": (
            total_val_true,
            val_pred[:, 0] + val_pred[:, 1],
            total_test_true,
            test_pred[:, 0] + test_pred[:, 1],
        ),
    }

    return {
        name: evaluate_target(
            name, X_val, v_true, v_pred, X_test, t_true, t_pred, idx_fit, idx_cal, config.model.random_state
        )
        for name, (v_true, v_pred, t_true, t_pred) in targets.items()
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
                f"  [{target_name}] coverage lw={r['coverage_lw']:.4f} const={r['coverage_const']:.4f} | "
                f"width lw_mean={r['width_lw_mean']:.2f} (std={r['width_lw_std']:.2f}) const_mean={r['width_const_mean']:.2f} | "
                f"sigma~|resid| corr={r['sigma_vs_abs_resid_corr']:.3f}"
            )
            for bucket, stats in r["buckets"].items():
                print(
                    f"    {bucket:9s} n={stats['n']:3d}  coverage_lw={stats['coverage_lw']:.3f}  "
                    f"width_lw={stats['width_lw_mean']:.2f}  coverage_const={stats['coverage_const']:.3f}"
                )

    print("\n=== Mean across folds ===")
    for target_name in ("diff", "total"):
        cov_lw = np.mean([per_fold[f.name][target_name]["coverage_lw"] for f in folds])
        cov_const = np.mean([per_fold[f.name][target_name]["coverage_const"] for f in folds])
        width_lw = np.mean([per_fold[f.name][target_name]["width_lw_mean"] for f in folds])
        width_const = np.mean([per_fold[f.name][target_name]["width_const_mean"] for f in folds])
        corr = np.mean([per_fold[f.name][target_name]["sigma_vs_abs_resid_corr"] for f in folds])
        print(
            f"[{target_name}] coverage lw={cov_lw:.4f} const={cov_const:.4f} | "
            f"width lw={width_lw:.2f} const={width_const:.2f} | mean sigma~|resid| corr={corr:.3f}"
        )


if __name__ == "__main__":
    main()
