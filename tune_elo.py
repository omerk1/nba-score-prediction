"""
NBA Score Prediction - Elo Parameter Tuning (Expanded)
=======================================================

Bayesian search (Optuna/TPE) over Elo rating formula parameters
(k_factor, home_advantage, season_regression, mov_multiplier).

Explores expanded search space:
- k_factor: 16–48 (vs. previous 5–40)
- home_advantage: current value ±3% (vs. previous 50–150)
- season_regression: 0.1–0.5 (vs. previous 0.0–0.75)
- mov_multiplier: True and False (categorical)

CatBoost hyperparameters are held fixed at elo_v1's best trial — this
isolates the effect of Elo's own formula parameters from model
hyperparameters (one change at a time).

Results are logged to outputs/elo_tuning_results.csv with columns:
k_factor, home_advantage, season_regression, mov_multiplier,
mae (diff_mae), mae_within_5 (diff_within_5), win_accuracy.

Usage:
    python tune_elo.py --run-name elo_param_search
    python tune_elo.py --run-name elo_param_search --n-trials 50
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, Any

import optuna
import pandas as pd

sys.path.append(str(Path(__file__).parent))

from src.data_processing.data_loader import NBADataLoader, load_training_data
from src.feature_engineering.elo import compute_elo_ratings
from src.feature_engineering.feature_builder import FeatureBuilder
from src.models.score_predictor import ScorePredictor
from src.utils.config_loader import load_config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
optuna.logging.set_verbosity(optuna.logging.WARNING)

# CatBoost hyperparameters fixed at elo_v1's best trial (#22, val_diff_mae=11.050)
FIXED_MODEL_PARAMS = {
    'depth': 3,
    'learning_rate': 0.100958003768401,
    'l2_leaf_reg': 6.327651488514239,
    'min_data_in_leaf': 43,
    'subsample': 0.9233447788848452,
    'colsample_bylevel': 0.8167267884483036,
}

ELO_COLS = ['home_team_elo', 'away_team_elo', 'elo_diff']

# Expanded search space for Elo parameters
# Current home_advantage is 117.87 from config; ±3% gives [114.33, 121.40]
SEARCH_SPACE = {
    'k_factor': (16.0, 48.0),
    'home_advantage': (114.33, 121.40),  # 117.87 ± 3%
    'season_regression': (0.1, 0.5),
    'mov_multiplier': [True, False],
}

# Storage for trial results to log to CSV
trial_results: list[Dict[str, Any]] = []


def apply_elo(features_df: pd.DataFrame, elo_df: pd.DataFrame, home_advantage: float) -> pd.DataFrame:
    """
    Apply Elo ratings to features dataframe.

    Args:
        features_df: Features dataframe with GAME_ID column
        elo_df: Elo ratings dataframe with GAME_ID, home_team_elo, away_team_elo
        home_advantage: Home advantage parameter for Elo differential calculation

    Returns:
        Updated features dataframe with Elo columns
    """
    merged = features_df[['GAME_ID']].merge(elo_df, on='GAME_ID', how='left')
    out = features_df.copy()
    out['home_team_elo'] = merged['home_team_elo'].values
    out['away_team_elo'] = merged['away_team_elo'].values
    out['elo_diff'] = merged['home_team_elo'].values + home_advantage - merged['away_team_elo'].values
    return out


def save_tuning_results(output_path: Path) -> None:
    """
    Save all trial results to CSV with columns:
    k_factor, home_advantage, season_regression, mov_multiplier,
    mae, mae_within_5, win_accuracy.

    Args:
        output_path: Path to save the CSV file
    """
    if not trial_results:
        logger.warning("No trial results to save")
        return

    results_df = pd.DataFrame(trial_results)
    results_df = results_df[[
        'k_factor', 'home_advantage', 'season_regression', 'mov_multiplier',
        'mae', 'mae_within_5', 'win_accuracy'
    ]]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_path, index=False)
    logger.info(f"Tuning results saved to {output_path} ({len(results_df)} trials)")


def main() -> None:
    """Execute Elo parameter tuning with expanded search space."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-name', required=True, help='Name for this tuning run')
    parser.add_argument('--n-trials', type=int, default=None, help='Override config elo_features.tuning.n_trials')
    args = parser.parse_args()

    config = load_config()
    if not config.elo_features or config.elo_features.tuning is None:
        logger.error("No 'tuning' section found under elo_features in config.yaml.")
        return
    et = config.elo_features.tuning

    train_df, val_df, _ = load_training_data(
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

    feature_builder = FeatureBuilder(
        rolling_windows=config.features.rolling_windows,
        h2h_margin_window=config.features.h2h_margin_window,
        h2h_win_rate_window=config.features.h2h_win_rate_window,
    )

    # Build base features once (Elo columns computed with default config params,
    # overwritten per-trial below)
    train_features = feature_builder.create_all_features(train_df)
    train_features = train_features[
        train_features['GAME_DATE'] >= pd.Timestamp(config.datasets_loading.train_start_date)
    ].reset_index(drop=True)

    val_features = feature_builder.create_all_features(val_df)
    val_features = val_features[
        (val_features['GAME_DATE'] >= pd.Timestamp(config.datasets_loading.validation_start_date)) &
        (val_features['SEASON_TYPE'].isin(config.datasets_loading.allowed_season_types))
    ].reset_index(drop=True)

    feature_cols = feature_builder.get_feature_names(train_features)
    target_cols = config.features.targets

    y_train = train_features[target_cols]
    y_val = val_features[target_cols]

    # Full chronological game history, loaded once for Elo recomputation
    loader = NBADataLoader(db_path=config.data_paths.raw_db)
    all_games = loader.load_games(
        start_date=config.datasets_loading.data_start_date,
        end_date=config.datasets_loading.test_end_date,
        allowed_season_types=config.datasets_loading.context_season_types or config.datasets_loading.allowed_season_types,
    )
    loader.close()

    def objective(trial: optuna.Trial) -> float:
        """
        Objective function for Optuna tuning.

        Explores Elo parameters using SEARCH_SPACE, trains a model,
        and returns validation diff_mae. Also records all metrics
        to trial_results for CSV logging.

        Args:
            trial: Optuna trial object

        Returns:
            Validation diff_mae (mean absolute error on point differential)
        """
        k_factor = trial.suggest_float('k_factor', *SEARCH_SPACE['k_factor'])
        home_advantage = trial.suggest_float('home_advantage', *SEARCH_SPACE['home_advantage'])
        season_regression = trial.suggest_float('season_regression', *SEARCH_SPACE['season_regression'])
        mov_multiplier = trial.suggest_categorical('mov_multiplier', SEARCH_SPACE['mov_multiplier'])

        elo_df = compute_elo_ratings(
            all_games,
            initial_rating=config.elo_features.initial_rating,
            k_factor=k_factor,
            home_advantage=home_advantage,
            mov_multiplier=mov_multiplier,
            season_regression=season_regression,
        )

        X_train = apply_elo(train_features, elo_df, home_advantage)[feature_cols]
        X_val = apply_elo(val_features, elo_df, home_advantage)[feature_cols]

        predictor = ScorePredictor(
            model_type='catboost',
            random_state=config.model.random_state,
            verbose=False,
            iterations=config.model.iterations,
            early_stopping_rounds=config.model.early_stopping_rounds,
            **FIXED_MODEL_PARAMS,
        )
        _, val_metrics = predictor.train(X_train, y_train, X_val, y_val)

        # Record trial result for CSV logging
        trial_result = {
            'k_factor': k_factor,
            'home_advantage': home_advantage,
            'season_regression': season_regression,
            'mov_multiplier': mov_multiplier,
            'mae': val_metrics['diff_mae'],
            'mae_within_5': val_metrics['diff_within_5'],
            'win_accuracy': val_metrics['win_accuracy'],
        }
        trial_results.append(trial_result)

        return val_metrics['diff_mae']

    storage_path = Path('outputs/optuna_study.db')
    storage_path.parent.mkdir(parents=True, exist_ok=True)
    study = optuna.create_study(
        study_name=args.run_name,
        direction='minimize',
        storage=f'sqlite:///{storage_path}',
        load_if_exists=True,
    )

    n_trials = args.n_trials if args.n_trials is not None else et.n_trials
    logger.info(f"Starting Elo param search: {n_trials} trials")
    logger.info(f"Search space:")
    logger.info(f"  k_factor: {SEARCH_SPACE['k_factor']}")
    logger.info(f"  home_advantage: {SEARCH_SPACE['home_advantage']}")
    logger.info(f"  season_regression: {SEARCH_SPACE['season_regression']}")
    logger.info(f"  mov_multiplier: {SEARCH_SPACE['mov_multiplier']}")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    # Save results to CSV
    csv_path = Path('outputs/elo_tuning_results.csv')
    save_tuning_results(csv_path)

    # Save Optuna trials to CSV as well (for historical reference)
    trials_df = study.trials_dataframe()
    trials_df.to_csv(Path('outputs') / f'optuna_trials_{args.run_name}.csv', index=False)

    logger.info(f"Best val diff_MAE: {study.best_value:.4f}")
    logger.info(f"Best Elo params: {study.best_params}")
    logger.info("Default Elo params (k=20, home_adv=100, mov=True, season_reg=0.33) gave val_diff_mae=11.050 (elo_v1)")


if __name__ == '__main__':
    main()
