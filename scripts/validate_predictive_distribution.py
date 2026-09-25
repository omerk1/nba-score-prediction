"""
Calibration check for src/evaluation/predictive_distribution.py's
Gaussian-KDE predictive distribution, full 5-fold CV, plus one worked
example (a fold-5 test game's full margin heatmap + derived win/cover
probabilities) to show the actual output shape.

Calibration check: for each test-fold game, evaluate the CDF (fit on that
fold's OWN validation residuals, never touching training or the point
predictions) at the actual outcome, offset by that game's own point
prediction -- this is the probability integral transform (PIT). If the
predictive distribution is well-calibrated, these PIT values are ~Uniform(0,1)
across the test set: mean ~0.5, and the fraction landing inside the central
50/80/90% band should track those nominal rates (the same coverage check
`prediction_interval_conformal_v1` used at one alpha, here at three, computed
directly from the CDF instead of separately-calibrated interval widths).

Run: venv/bin/python3 scripts/validate_predictive_distribution.py
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

import numpy as np  # noqa: E402

from src.evaluation.cv_harness import run_split  # noqa: E402
from src.evaluation.predictive_distribution import (  # noqa: E402
    cover_probability,
    margin_pmf,
    residual_cdf,
    silverman_bandwidth,
    win_probability,
)
from src.utils.config_loader import load_config  # noqa: E402

NOMINAL_LEVELS = (0.5, 0.8, 0.9)


def pit_coverage(pit: np.ndarray) -> dict:
    stats = {"mean_pit": float(np.mean(pit))}
    for level in NOMINAL_LEVELS:
        lo, hi = (1 - level) / 2, 1 - (1 - level) / 2
        stats[f"coverage_{int(level * 100)}"] = float(np.mean((pit >= lo) & (pit <= hi)))
    return stats


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

    fold_stats = {}
    for name, val_true, v_pred, test_true, t_pred in (
        (
            "diff",
            (val_features[home_col] - val_features[away_col]).to_numpy(),
            val_pred[:, 0] - val_pred[:, 1],
            (test_features[home_col] - test_features[away_col]).to_numpy(),
            test_pred[:, 0] - test_pred[:, 1],
        ),
        (
            "total",
            (val_features[home_col] + val_features[away_col]).to_numpy(),
            val_pred[:, 0] + val_pred[:, 1],
            (test_features[home_col] + test_features[away_col]).to_numpy(),
            test_pred[:, 0] + test_pred[:, 1],
        ),
    ):
        val_resid = val_true - v_pred
        bandwidth = silverman_bandwidth(val_resid)
        test_resid = test_true - t_pred  # actual - point_pred, exactly what the CDF's argument expects
        pit = residual_cdf(test_resid, val_resid, bandwidth)
        fold_stats[name] = {"bandwidth": bandwidth, **pit_coverage(pit)}

    return fold_stats, (predictor, test_features, feature_cols, val_features)


def main():
    config = load_config()
    folds = config.cv.folds
    per_fold = {}
    last_fold_artifacts = None
    for fold in folds:
        stats, artifacts = evaluate_fold(config, fold)
        per_fold[fold.name] = stats
        last_fold_artifacts = (fold, artifacts)
        for target_name, s in stats.items():
            print(
                f"{fold.name} [{target_name}] bandwidth={s['bandwidth']:.2f} | mean_pit={s['mean_pit']:.3f} | "
                f"coverage50={s['coverage_50']:.3f} coverage80={s['coverage_80']:.3f} coverage90={s['coverage_90']:.3f}"
            )

    print("\n=== Mean across folds ===")
    for target_name in ("diff", "total"):
        for key in ("mean_pit", "coverage_50", "coverage_80", "coverage_90"):
            vals = [per_fold[f.name][target_name][key] for f in folds]
            print(f"[{target_name}] {key}: {np.mean(vals):.4f}  (per-fold: {[round(v, 3) for v in vals]})")

    # Worked example on the last (most recent) fold's test set.
    fold, (predictor, test_features, feature_cols, val_features) = last_fold_artifacts
    home_col, away_col = config.features.targets[0], config.features.targets[1]
    val_pred = predictor.predict(val_features[feature_cols])
    diff_val_resid = (val_features[home_col] - val_features[away_col]).to_numpy() - (
        val_pred[:, 0] - val_pred[:, 1]
    )
    bandwidth = silverman_bandwidth(diff_val_resid)

    test_pred = predictor.predict(test_features[feature_cols])
    diff_pred_example = float(test_pred[0, 0] - test_pred[0, 1])
    home_team = test_features.iloc[0].get("HOME_TEAM_ID", "home")
    away_team = test_features.iloc[0].get("AWAY_TEAM_ID", "away")

    print(f"\n=== Worked example: {fold.name} test game 0 ({home_team} vs {away_team}) ===")
    print(f"Point prediction (home - away): {diff_pred_example:+.1f}")
    print(f"Win probability (home): {win_probability(diff_pred_example, diff_val_resid, bandwidth):.1%}")
    for line in (5.5, 9.5):
        print(
            f"P(home covers -{line}): {cover_probability(diff_pred_example, line, diff_val_resid, bandwidth):.1%} | "
            f"P(away covers +{line}): {1 - cover_probability(diff_pred_example, -line, diff_val_resid, bandwidth):.1%}"
        )

    margins, probs = margin_pmf(diff_pred_example, diff_val_resid, bandwidth, lo=-30, hi=30)
    print("\nMargin heatmap (home margin : probability), top 15 by probability:")
    top = np.argsort(probs)[::-1][:15]
    for i in sorted(top):
        print(f"  {margins[i]:+3d}: {probs[i]:.4f}")


if __name__ == "__main__":
    main()
