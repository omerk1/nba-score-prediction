"""
NBA Score Prediction - Main Training Pipeline
==============================================

End-to-end pipeline for training the NBA score prediction model.

Usage:
    python train_model.py --run-name baseline_stats_only
    python train_model.py --run-name injury_features_v1 --notes "first run with injury impact"
    python train_model.py --run-name champion_cv_check --protocol cv --no-log

Every run appends one row to outputs/experiments_v2.csv for ablation
comparison, unless --no-log is given. outputs/experiments.csv (the old
16-column schema) is a frozen historical snapshot, no longer written to --
see scripts/migrate_experiments_schema.py. --protocol selects the evaluation harness:
`single_split` (default, today's fixed configs/config.yaml split) or `cv`
(expanding-window CV over configs/config.yaml's cv.folds -- see
src/evaluation/cv_harness.py and CLAUDE.md's "Project Rules (ML
experimentation)" section).
"""

import argparse
import csv
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

# Add src to path
sys.path.append(str(Path(__file__).parent))

from src.evaluation.cv_harness import run_expanding_window_cv, run_split
from src.utils.config_loader import load_config

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def _save_experiment(
    run_name: str,
    notes: str,
    config,
    val_metrics: dict,
    test_metrics: dict,
    n_features: int,
    protocol: str = "single_split",
    session_id: str = "",
    val_score_mean: float = None,
    val_score_per_fold: str = "",
    test_score_mean: float = None,
    experiments_csv: str = "outputs/experiments_v2.csv",
) -> None:
    """Append one row to `experiments_csv` (default outputs/experiments_v2.csv,
    the shared production log -- the new CV-protocol schema; the old
    outputs/experiments.csv is a frozen historical snapshot, never written
    to going forward). Creates the file with headers if absent.

    `experiments_csv` override (added for A7's KNN-score integration test): lets
    exploratory/throwaway runs log to a separate file instead of polluting the
    shared history — see --experiments-csv below.

    `protocol`/`session_id`/`val_score_mean`/`val_score_per_fold`/`test_score_mean`
    are the leaderboard columns from CLAUDE.md's "Project Rules (ML
    experimentation)" section. For `protocol="single_split"`, val_score_mean/
    test_score_mean are that one split's own composite score (see
    src/evaluation/cv_harness.compute_composite_score) and val_score_per_fold
    is empty (no folds). For `protocol="cv"`, they're the mean across folds
    and the per-fold list respectively."""
    out = Path(experiments_csv)
    out.parent.mkdir(parents=True, exist_ok=True)

    row = {
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "run_name": run_name,
        # Spread market
        "val_diff_mae": round(val_metrics["diff_mae"], 3),
        "test_diff_mae": round(test_metrics["diff_mae"], 3),
        "val_diff_within_5": round(val_metrics["diff_within_5"], 4),
        "test_diff_within_5": round(test_metrics["diff_within_5"], 4),
        # Over/under market
        "val_total_mae": round(val_metrics["total_mae"], 3),
        "test_total_mae": round(test_metrics["total_mae"], 3),
        # Moneyline market
        "val_win_acc": round(val_metrics["win_accuracy"], 4),
        "test_win_acc": round(test_metrics["win_accuracy"], 4),
        "val_brier": round(val_metrics["brier_score"], 4),
        "test_brier": round(test_metrics["brier_score"], 4),
        # Run metadata
        "n_features": n_features,
        "injury_enabled": bool(config.injury_features and config.injury_features.enabled),
        "rolling_windows": ",".join(str(w) for w in config.features.rolling_windows),
        # Leaderboard / CV protocol columns (CLAUDE.md's Project Rules)
        "val_score_mean": round(val_score_mean, 4) if val_score_mean is not None else "",
        "val_score_per_fold": val_score_per_fold,
        "test_score_mean": round(test_score_mean, 4) if test_score_mean is not None else "",
        "protocol": protocol,
        "session_id": session_id,
        "notes": notes,
    }

    write_header = not out.exists()
    with open(out, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if write_header:
            writer.writeheader()
        writer.writerow(row)

    logger.info(f"Experiment saved → {out}  (run: {run_name})")


def _run_single_split(config, args) -> None:
    """Today's default protocol: one fixed split from configs/config.yaml's
    datasets_loading dates. Preserves every side effect the pipeline has
    always had (feature CSVs, saved model, feature-importance reports,
    example predictions, training metadata) -- only the core train+evaluate
    step now goes through the shared src/evaluation/cv_harness.run_split."""
    try:
        result = run_split(
            config,
            train_end_date=config.datasets_loading.train_end_date,
            validation_start_date=config.datasets_loading.validation_start_date,
            validation_end_date=config.datasets_loading.validation_end_date,
            test_start_date=config.datasets_loading.test_start_date,
            test_end_date=config.datasets_loading.test_end_date,
            keep_artifacts=True,
        )
    except FileNotFoundError as e:
        logger.error(str(e))
        return

    logger.info(
        f"Train: {result.n_train:,} | Val: {result.n_val:,} | Test: {result.n_test:,} | "
        f"Features: {result.n_features}"
    )

    naive_window = config.features.naive_rolling_baseline
    if naive_window not in config.features.rolling_windows:
        logger.warning(
            f"naive_rolling_baseline {naive_window} is not in rolling_windows "
            f"{config.features.rolling_windows} — skipping naive baseline"
        )
    else:
        logger.info(
            f"Naive baseline (rolling-{naive_window}) — "
            f"val diff_mae: {result.naive_val_metrics['diff_mae']:.2f} | "
            f"test diff_mae: {result.naive_test_metrics['diff_mae']:.2f} | "
            f"val win_acc: {result.naive_val_metrics['win_accuracy']:.1%} | "
            f"test win_acc: {result.naive_test_metrics['win_accuracy']:.1%}"
        )
        if not args.no_log:
            baseline_run_name = f"naive_rolling_{naive_window}"
            experiments_path = Path(args.experiments_csv)
            already_logged = experiments_path.exists() and baseline_run_name in experiments_path.read_text()
            if not already_logged:
                _save_experiment(
                    baseline_run_name,
                    f"auto-generated baseline (rolling {naive_window}-game avg)",
                    config,
                    result.naive_val_metrics,
                    result.naive_test_metrics,
                    2,
                    experiments_csv=args.experiments_csv,
                )

    # keep_artifacts=True above means result already carries the trained
    # predictor and feature DataFrames -- reused directly below, no re-run.
    train_features = result.train_features
    val_features = result.val_features
    test_features = result.test_features
    predictor = result.predictor

    features_dir = Path("data/features")
    features_dir.mkdir(exist_ok=True, parents=True)
    train_features.to_csv(features_dir / "train_features.csv", index=False)
    val_features.to_csv(features_dir / "val_features.csv", index=False)
    test_features.to_csv(features_dir / "test_features.csv", index=False)

    target_cols = config.features.targets
    feature_cols = result.feature_cols
    X_test, y_test = test_features[feature_cols], test_features[target_cols]
    n_train, n_val, n_test = result.n_train, result.n_val, result.n_test

    importance_df = predictor.get_feature_importance(top_n=20)
    print("\nTop 20 features:\n" + importance_df.to_string(index=False))

    reports_dir = Path("outputs/reports")
    reports_dir.mkdir(exist_ok=True, parents=True)
    importance_df.to_csv(reports_dir / f"feature_importance_{args.run_name}.csv", index=False)

    # outputs/reports/ is gitignored (and the model itself is gitignored too), so
    # per-feature importance artifacts don't survive once the worktree is cleaned
    # up -- only the top-20 above survived, print-only. Save the FULL table (every
    # feature, not just any one experimental module's) to a non-gitignored path
    # (see the matching !outputs/full_feature_importance_*.csv exception in
    # .gitignore) so the coordinator can see where H2H/Elo/rolling-stats/etc. rank
    # alongside any new experimental features across runs.
    full_importance_df = predictor.get_feature_importance(top_n=len(feature_cols))
    full_importance_path = Path(f"outputs/full_feature_importance_{args.run_name}.csv")
    full_importance_df.to_csv(full_importance_path, index=False)
    logger.info(
        f"Full feature importance ({len(full_importance_df)} features) saved -> {full_importance_path}"
    )

    predictions = predictor.predict(X_test.head(10))
    examples_df = pd.DataFrame(
        {
            "Date": test_features.head(10)["GAME_DATE"].values,
            "Home": test_features.head(10)["HOME_TEAM_ID"].values,
            "Away": test_features.head(10)["AWAY_TEAM_ID"].values,
            "Act_H": y_test.head(10)["PTS_home"].values,
            "Act_A": y_test.head(10)["PTS_away"].values,
            "Pred_H": predictions[:, 0].round(1),
            "Pred_A": predictions[:, 1].round(1),
            "Act_Diff": (y_test.head(10)["PTS_home"] - y_test.head(10)["PTS_away"]).values,
            "Pred_Diff": (predictions[:, 0] - predictions[:, 1]).round(1),
        }
    )
    print("\nExample predictions (first 10 test games):\n" + examples_df.to_string(index=False))

    models_dir = Path("data/models")
    models_dir.mkdir(exist_ok=True, parents=True)
    model_path = models_dir / "score_predictor.pkl"
    predictor.save(model_path)

    metadata = {
        "train_date": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "train_games": n_train,
        "val_games": n_val,
        "test_games": n_test,
        "features": len(feature_cols),
        "train_metrics": result.train_metrics,
        "val_metrics": result.val_metrics,
        "test_metrics": result.test_metrics,
        "model_type": "catboost",
        "rolling_windows": config.features.rolling_windows,
        "random_state": config.model.random_state,
    }
    with open(models_dir / "training_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2, default=str)

    if not args.no_log:
        _save_experiment(
            args.run_name,
            args.notes,
            config,
            result.val_metrics,
            result.test_metrics,
            result.n_features,
            protocol="single_split",
            val_score_mean=result.val_score,
            test_score_mean=result.test_score,
            experiments_csv=args.experiments_csv,
        )
    logger.info(
        f"Test — diff_mae: {result.test_metrics['diff_mae']:.2f} | "
        f"win_acc: {result.test_metrics['win_accuracy']:.1%} | "
        f"score: {result.test_score:.4f}"
    )
    logger.info(f"Model saved to {model_path}")


def _run_cv(config, args) -> None:
    """--protocol cv: expanding-window CV over configs/config.yaml's cv.folds.
    Prints per-fold results + means; does NOT save data/features CSVs, a
    model, or feature-importance reports (those represent one deployed
    snapshot, not meaningful for a 5-fold aggregate). Logs one aggregate row
    to experiments.csv unless --no-log."""
    cv_result = run_expanding_window_cv(config)

    print("\nExpanding-window CV results:")
    for name, r in zip(cv_result.fold_names, cv_result.fold_results):
        print(
            f"  {name}: val_score={r.val_score:.4f} (diff_mae={r.val_metrics['diff_mae']:.2f}, "
            f"total_mae={r.val_metrics['total_mae']:.2f}) | "
            f"test_score={r.test_score:.4f} (diff_mae={r.test_metrics['diff_mae']:.2f}, "
            f"total_mae={r.test_metrics['total_mae']:.2f})"
        )
    print(
        f"\nMean val_score:  {cv_result.val_score_mean:.4f}"
        f"\nMean test_score: {cv_result.test_score_mean:.4f}"
    )

    if args.no_log:
        logger.info("--no-log set: not writing to experiments.csv")
        return

    last = cv_result.fold_results[-1]
    _save_experiment(
        args.run_name,
        args.notes,
        config,
        val_metrics={
            "diff_mae": sum(r.val_metrics["diff_mae"] for r in cv_result.fold_results)
            / len(cv_result.fold_results),
            "diff_within_5": sum(r.val_metrics["diff_within_5"] for r in cv_result.fold_results)
            / len(cv_result.fold_results),
            "total_mae": sum(r.val_metrics["total_mae"] for r in cv_result.fold_results)
            / len(cv_result.fold_results),
            "win_accuracy": sum(r.val_metrics["win_accuracy"] for r in cv_result.fold_results)
            / len(cv_result.fold_results),
            "brier_score": sum(r.val_metrics["brier_score"] for r in cv_result.fold_results)
            / len(cv_result.fold_results),
        },
        test_metrics={
            "diff_mae": sum(r.test_metrics["diff_mae"] for r in cv_result.fold_results)
            / len(cv_result.fold_results),
            "diff_within_5": sum(r.test_metrics["diff_within_5"] for r in cv_result.fold_results)
            / len(cv_result.fold_results),
            "total_mae": sum(r.test_metrics["total_mae"] for r in cv_result.fold_results)
            / len(cv_result.fold_results),
            "win_accuracy": sum(r.test_metrics["win_accuracy"] for r in cv_result.fold_results)
            / len(cv_result.fold_results),
            "brier_score": sum(r.test_metrics["brier_score"] for r in cv_result.fold_results)
            / len(cv_result.fold_results),
        },
        n_features=last.n_features,
        protocol="cv",
        val_score_mean=cv_result.val_score_mean,
        val_score_per_fold=",".join(f"{r.val_score:.4f}" for r in cv_result.fold_results),
        test_score_mean=cv_result.test_score_mean,
        experiments_csv=args.experiments_csv,
    )


def main():
    """Main training pipeline"""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-name", required=True, help="Short name for this ablation run, e.g. 'baseline' or 'injury_v1'"
    )
    parser.add_argument("--notes", default="", help="Optional free-text notes saved to experiments.csv")
    parser.add_argument(
        "--protocol",
        choices=["single_split", "cv"],
        default="single_split",
        help="single_split (default): today's fixed configs/config.yaml split. "
        "cv: expanding-window CV over configs/config.yaml's cv.folds.",
    )
    parser.add_argument(
        "--no-log", action="store_true", help="Print results but don't append a row to experiments.csv."
    )
    parser.add_argument(
        "--experiments-csv",
        default="outputs/experiments_v2.csv",
        help="Path to append this run's row to (default: outputs/experiments_v2.csv, the shared "
        "production log -- the new CV-protocol schema; outputs/experiments.csv, the old "
        "16-column schema, is a frozen historical snapshot). Override for exploratory/throwaway "
        "runs that shouldn't land in the shared history, e.g. --experiments-csv "
        "outputs/a7_integration_test_results.csv",
    )
    args = parser.parse_args()

    logger.info(f"Training pipeline started at {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    config = load_config()

    if args.protocol == "cv":
        _run_cv(config, args)
    else:
        _run_single_split(config, args)


if __name__ == "__main__":
    main()
