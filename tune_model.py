"""
NBA Score Prediction - Hyperparameter Tuning
=============================================

Bayesian HP search (Optuna/TPE) over CatBoost parameters.
Optimizes val diff_MAE with early stopping on every trial.
Logs the best configuration to outputs/experiments.csv.

Usage:
    python tune_model.py --run-name hp_tuned --n-trials 40
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import optuna
import pandas as pd

sys.path.append(str(Path(__file__).parent))

from src.data_processing.data_loader import load_training_data
from src.feature_engineering.feature_builder import FeatureBuilder
from src.models.score_predictor import ScorePredictor
from src.utils.config_loader import load_config
from train_model import _naive_baseline_metrics, _save_experiment

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
optuna.logging.set_verbosity(optuna.logging.WARNING)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-name', required=True)
    parser.add_argument('--n-trials', type=int, default=40)
    parser.add_argument('--notes', default='')
    args = parser.parse_args()

    config = load_config()

    # --- Load data once ---
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
            context_season_types=config.datasets_loading.context_season_types,
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
    val_features = val_features[
        (val_features['GAME_DATE'] >= pd.Timestamp(config.datasets_loading.validation_start_date)) &
        (val_features['SEASON_TYPE'].isin(config.datasets_loading.allowed_season_types))
    ].reset_index(drop=True)

    test_features = feature_builder.create_all_features(test_df)
    test_features = test_features[
        (test_features['GAME_DATE'] >= pd.Timestamp(config.datasets_loading.test_start_date)) &
        (test_features['SEASON_TYPE'].isin(config.datasets_loading.allowed_season_types))
    ].reset_index(drop=True)

    feature_cols = feature_builder.get_feature_names(train_features)
    target_cols = config.features.targets

    X_train = train_features[feature_cols]
    y_train = train_features[target_cols]
    X_val   = val_features[feature_cols]
    y_val   = val_features[target_cols]
    X_test  = test_features[feature_cols]
    y_test  = test_features[target_cols]

    logger.info(f"Train: {len(X_train):,} | Val: {len(X_val):,} | Test: {len(X_test):,} | Features: {len(feature_cols)}")

    # --- Optuna objective ---
    def objective(trial):
        params = {
            'depth':             trial.suggest_int('depth', 4, 10),
            'learning_rate':     trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'l2_leaf_reg':       trial.suggest_float('l2_leaf_reg', 1.0, 30.0, log=True),
            'min_data_in_leaf':  trial.suggest_int('min_data_in_leaf', 1, 30),
            'subsample':         trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.5, 1.0),
            'iterations':        config.model.iterations,
            'early_stopping_rounds': config.model.early_stopping_rounds,
        }
        predictor = ScorePredictor(
            model_type='catboost',
            random_state=config.model.random_state,
            verbose=False,
            **params,
        )
        _, val_metrics = predictor.train(X_train, y_train, X_val, y_val)
        return val_metrics['diff_mae']

    # --- Run study ---
    logger.info(f"Starting Optuna study: {args.n_trials} trials")
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=args.n_trials, show_progress_bar=True)

    best = study.best_params
    logger.info(f"Best val diff_MAE: {study.best_value:.4f}")
    logger.info(f"Best params: {best}")

    # --- Retrain on best params and evaluate on test ---
    best_predictor = ScorePredictor(
        model_type='catboost',
        random_state=config.model.random_state,
        verbose=False,
        iterations=config.model.iterations,
        early_stopping_rounds=config.model.early_stopping_rounds,
        **best,
    )
    _, val_metrics = best_predictor.train(X_train, y_train, X_val, y_val)
    test_metrics = best_predictor.evaluate(X_test, y_test, dataset_name='Test')

    importance_df = best_predictor.get_feature_importance(top_n=20)
    print('\nTop 20 features:\n' + importance_df.to_string(index=False))

    reports_dir = Path('outputs/reports')
    reports_dir.mkdir(exist_ok=True, parents=True)
    importance_df.to_csv(reports_dir / f'feature_importance_{args.run_name}.csv', index=False)

    model_path = Path(config.data_paths.models) / 'score_predictor.pkl'
    best_predictor.save(str(model_path))

    params_str = ', '.join(f'{k}={v:.4g}' if isinstance(v, float) else f'{k}={v}' for k, v in best.items())
    notes = args.notes or f'Optuna {args.n_trials} trials; best: {params_str}'
    _save_experiment(args.run_name, notes, config, val_metrics, test_metrics, len(feature_cols))

    logger.info(f"Test — diff_mae: {test_metrics['diff_mae']:.2f} | win_acc: {test_metrics['win_accuracy']:.1%}")
    logger.info(f"Model saved to {model_path}")


if __name__ == '__main__':
    main()
