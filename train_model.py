"""
NBA Score Prediction - Main Training Pipeline
==============================================

End-to-end pipeline for training the NBA score prediction model.

Usage:
    python train_model.py --run-name baseline_stats_only
    python train_model.py --run-name injury_features_v1 --notes "first run with injury impact"

Every run appends one row to outputs/experiments.csv for ablation comparison.
"""

import argparse
import csv
import json
import sys
import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

# Add src to path
sys.path.append(str(Path(__file__).parent))

from src.data_processing.data_loader import NBADataLoader, load_training_data
from src.feature_engineering.feature_builder import FeatureBuilder
from src.models.score_predictor import ScorePredictor
from src.utils.config_loader import load_config, get_config_value

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def _naive_baseline_metrics(features_df: pd.DataFrame, y_true: pd.DataFrame, window: int) -> dict:
    """Compute metrics for a naive predictor that uses each team's rolling avg score."""
    home_pred = features_df[f'home_team_pts_avg_L{window}'].values
    away_pred = features_df[f'away_team_pts_avg_L{window}'].values
    home_true = y_true.iloc[:, 0].values
    away_true = y_true.iloc[:, 1].values

    diff_true = home_true - away_true
    diff_pred = home_pred - away_pred
    total_true = home_true + away_true
    total_pred = home_pred + away_pred

    abs_diff_err = np.abs(diff_true - diff_pred)
    residual_std = np.std(diff_true - diff_pred) or 1.0
    return {
        'diff_mae':      float(np.mean(abs_diff_err)),
        'diff_within_5': float(np.mean(abs_diff_err <= 5)),
        'total_mae':     float(np.mean(np.abs(total_true - total_pred))),
        'win_accuracy':  float(np.mean((diff_true > 0) == (diff_pred > 0))),
        'brier_score':   float(np.mean(
            (norm.cdf(diff_pred / residual_std) - (diff_true > 0).astype(float)) ** 2
        )),
    }


def _save_experiment(run_name: str, notes: str, config, val_metrics: dict, test_metrics: dict, n_features: int) -> None:
    """Append one row to outputs/experiments.csv. Creates the file with headers if absent."""
    out = Path("outputs/experiments.csv")
    out.parent.mkdir(parents=True, exist_ok=True)

    row = {
        "timestamp":          datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "run_name":           run_name,
        # Spread market
        "val_diff_mae":       round(val_metrics["diff_mae"], 3),
        "test_diff_mae":      round(test_metrics["diff_mae"], 3),
        "val_diff_within_5":  round(val_metrics["diff_within_5"], 4),
        "test_diff_within_5": round(test_metrics["diff_within_5"], 4),
        # Over/under market
        "val_total_mae":      round(val_metrics["total_mae"], 3),
        "test_total_mae":     round(test_metrics["total_mae"], 3),
        # Moneyline market
        "val_win_acc":        round(val_metrics["win_accuracy"], 4),
        "test_win_acc":       round(test_metrics["win_accuracy"], 4),
        "val_brier":          round(val_metrics["brier_score"], 4),
        "test_brier":         round(test_metrics["brier_score"], 4),
        # Run metadata
        "n_features":         n_features,
        "injury_enabled":     bool(config.injury_features and config.injury_features.enabled),
        "rolling_windows":    ",".join(str(w) for w in config.features.rolling_windows),
        "notes":              notes,
    }

    write_header = not out.exists()
    with open(out, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if write_header:
            writer.writeheader()
        writer.writerow(row)

    logger.info(f"Experiment saved → {out}  (run: {run_name})")


def main():
    """Main training pipeline"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", required=True, help="Short name for this ablation run, e.g. 'baseline' or 'injury_v1'")
    parser.add_argument("--notes", default="", help="Optional free-text notes saved to experiments.csv")
    args = parser.parse_args()

    logger.info(f"Training pipeline started at {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    config = load_config()
    
    try:
        train_df, val_df, test_df = load_training_data(
            db_path=config.data_paths.raw_db,
            train_start_date=config.datasets_loading.train_start_date,
            train_end_date=config.datasets_loading.train_end_date,
            val_start_date=config.datasets_loading.validation_start_date,
            val_end_date=config.datasets_loading.validation_end_date,
            test_start_date=config.datasets_loading.test_start_date,
            test_end_date=config.datasets_loading.test_end_date,
            allowed_season_types=config.datasets_loading.allowed_season_types,
            data_start_date=config.datasets_loading.data_start_date,
        )
    except FileNotFoundError as e:
        logger.error(str(e))
        return

    feature_builder = FeatureBuilder(
        rolling_windows=config.features.rolling_windows,
        h2h_margin_window=config.features.h2h_margin_window,
        h2h_win_rate_window=config.features.h2h_win_rate_window,
    )
    train_features = feature_builder.create_all_features(train_df)
    train_features = train_features[
        train_features['GAME_DATE'] >= pd.Timestamp(config.datasets_loading.train_start_date)
    ].reset_index(drop=True)
    val_features = feature_builder.create_all_features(val_df)
    test_features = feature_builder.create_all_features(test_df)

    features_dir = Path("data/features")
    features_dir.mkdir(exist_ok=True, parents=True)
    train_features.to_csv(features_dir / "train_features.csv", index=False)
    val_features.to_csv(features_dir / "val_features.csv", index=False)
    test_features.to_csv(features_dir / "test_features.csv", index=False)

    target_cols = config.features.targets
    feature_cols = feature_builder.get_feature_names(train_features)

    X_train = train_features[feature_cols]
    y_train = train_features[target_cols]
    X_val = val_features[feature_cols]
    y_val = val_features[target_cols]
    X_test = test_features[feature_cols]
    y_test = test_features[target_cols]

    logger.info(f"Train: {len(X_train):,} | Val: {len(X_val):,} | Test: {len(X_test):,} | Features: {len(feature_cols)}")

    predictor = ScorePredictor(
        model_type='catboost',
        iterations=200,
        depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bylevel=0.8,
        random_state=config.model.random_state,
        verbose=False
    )

    train_metrics, val_metrics = predictor.train(X_train, y_train, X_val, y_val)
    test_metrics = predictor.evaluate(X_test, y_test, dataset_name="Test")

    naive_window = config.features.naive_rolling_baseline
    if naive_window not in config.features.rolling_windows:
        logger.warning(f"naive_rolling_baseline {naive_window} is not in rolling_windows {config.features.rolling_windows} — skipping naive baseline")
    else:
        baseline_val  = _naive_baseline_metrics(val_features,  y_val,  naive_window)
        baseline_test = _naive_baseline_metrics(test_features, y_test, naive_window)
        logger.info(
            f"Naive baseline (rolling-{naive_window}) — "
            f"val diff_mae: {baseline_val['diff_mae']:.2f} | "
            f"test diff_mae: {baseline_test['diff_mae']:.2f} | "
            f"val win_acc: {baseline_val['win_accuracy']:.1%} | "
            f"test win_acc: {baseline_test['win_accuracy']:.1%}"
        )
    if naive_window in config.features.rolling_windows:
        baseline_run_name = f"naive_rolling_{naive_window}"
        experiments_path = Path("outputs/experiments.csv")
        already_logged = (
            experiments_path.exists()
            and baseline_run_name in experiments_path.read_text()
        )
        if not already_logged:
            _save_experiment(
                baseline_run_name, f"auto-generated baseline (rolling {naive_window}-game avg)",
                config, baseline_val, baseline_test, 2
            )

    importance_df = predictor.get_feature_importance(top_n=20)
    print("\nTop 20 features:\n" + importance_df.to_string(index=False))

    reports_dir = Path("outputs/reports")
    reports_dir.mkdir(exist_ok=True, parents=True)
    importance_df.to_csv(reports_dir / f"feature_importance_{args.run_name}.csv", index=False)

    predictions = predictor.predict(X_test.head(10))
    examples_df = pd.DataFrame({
        'Date': test_features.head(10)['GAME_DATE'].values,
        'Home': test_features.head(10)['HOME_TEAM_ID'].values,
        'Away': test_features.head(10)['AWAY_TEAM_ID'].values,
        'Act_H': y_test.head(10)['PTS_home'].values,
        'Act_A': y_test.head(10)['PTS_away'].values,
        'Pred_H': predictions[:, 0].round(1),
        'Pred_A': predictions[:, 1].round(1),
        'Act_Diff': (y_test.head(10)['PTS_home'] - y_test.head(10)['PTS_away']).values,
        'Pred_Diff': (predictions[:, 0] - predictions[:, 1]).round(1),
    })
    print("\nExample predictions (first 10 test games):\n" + examples_df.to_string(index=False))

    models_dir = Path("data/models")
    models_dir.mkdir(exist_ok=True, parents=True)
    model_path = models_dir / "score_predictor.pkl"
    predictor.save(model_path)

    metadata = {
        'train_date': datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC'),
        'train_games': len(X_train),
        'val_games': len(X_val),
        'test_games': len(X_test),
        'features': len(feature_cols),
        'train_metrics': train_metrics,
        'val_metrics': val_metrics,
        'test_metrics': test_metrics,
        'model_type': 'catboost',
        'rolling_windows': config.features.rolling_windows,
        'random_state': config.model.random_state,
    }
    with open(models_dir / "training_metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2, default=str)

    _save_experiment(args.run_name, args.notes, config, val_metrics, test_metrics, len(feature_cols))
    logger.info(f"Test — diff_mae: {test_metrics['diff_mae']:.2f} | win_acc: {test_metrics['win_accuracy']:.1%}")
    logger.info(f"Model saved to {model_path}")


if __name__ == "__main__":
    main()