"""
NBA Score Prediction - Elo Parameter Tuning
============================================

Bayesian search (Optuna/TPE) over Elo rating formula parameters
(k_factor, home_advantage, season_regression, mov_multiplier).

CatBoost hyperparameters are held fixed at elo_v1's best trial — this
isolates the effect of Elo's own formula parameters from model
hyperparameters (one change at a time).

Usage:
    python tune_elo.py --run-name elo_param_search
"""

import argparse
import logging
import sys
from pathlib import Path

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-name', required=True)
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

    def apply_elo(features_df: pd.DataFrame, elo_df: pd.DataFrame, home_advantage: float) -> pd.DataFrame:
        merged = features_df[['GAME_ID']].merge(elo_df, on='GAME_ID', how='left')
        out = features_df.copy()
        out['home_team_elo'] = merged['home_team_elo'].values
        out['away_team_elo'] = merged['away_team_elo'].values
        out['elo_diff'] = merged['home_team_elo'].values + home_advantage - merged['away_team_elo'].values
        return out

    def objective(trial):
        k_factor = trial.suggest_float('k_factor', *et.k_factor)
        home_advantage = trial.suggest_float('home_advantage', *et.home_advantage)
        season_regression = trial.suggest_float('season_regression', *et.season_regression)
        mov_multiplier = trial.suggest_categorical('mov_multiplier', [True, False])

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
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    trials_df = study.trials_dataframe()
    trials_df.to_csv(Path('outputs') / f'optuna_trials_{args.run_name}.csv', index=False)

    logger.info(f"Best val diff_MAE: {study.best_value:.4f}")
    logger.info(f"Best Elo params: {study.best_params}")
    logger.info("Default Elo params (k=20, home_adv=100, mov=True, season_reg=0.33) gave val_diff_mae=11.050 (elo_v1)")


if __name__ == '__main__':
    main()
