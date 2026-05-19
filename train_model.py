"""
NBA Score Prediction - Main Training Pipeline
==============================================

End-to-end pipeline for training the NBA score prediction model.

Usage:
    python train_model.py
"""

import json
import sys
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

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


def main():
    """Main training pipeline"""

    logger.info(f"Training pipeline started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
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
        )
    except FileNotFoundError as e:
        logger.error(str(e))
        return

    feature_builder = FeatureBuilder(
        rolling_window=config.features.rolling_window,
        h2h_margin_window=config.features.h2h_margin_window,
        h2h_win_rate_window=config.features.h2h_win_rate_window,
    )
    train_features = feature_builder.create_all_features(train_df)
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

    importance_df = predictor.get_feature_importance(top_n=20)
    print("\nTop 20 features:\n" + importance_df.to_string(index=False))

    reports_dir = Path("outputs/reports")
    reports_dir.mkdir(exist_ok=True, parents=True)
    importance_df.to_csv(reports_dir / "feature_importance.csv", index=False)

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
        'train_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'train_games': len(X_train),
        'val_games': len(X_val),
        'test_games': len(X_test),
        'features': len(feature_cols),
        'train_metrics': train_metrics,
        'val_metrics': val_metrics,
        'test_metrics': test_metrics,
        'model_type': 'catboost',
        'rolling_window': config.features.rolling_window,
        'random_state': config.model.random_state,
    }
    with open(models_dir / "training_metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2, default=str)

    logger.info(f"Test — diff_mae: {test_metrics['diff_mae']:.2f} | win_acc: {test_metrics['win_accuracy']:.1%}")
    logger.info(f"Model saved to {model_path}")


if __name__ == "__main__":
    main()