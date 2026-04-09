"""
NBA Score Prediction - Main Training Pipeline
==============================================

End-to-end pipeline for training the NBA score prediction model.

Usage:
    python train_model.py
"""

import sys
import logging
from datetime import datetime
from pathlib import Path
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

    print("\n" + "="*80)
    print("NBA SCORE PREDICTION - TRAINING PIPELINE")
    print("="*80)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Load configuration
    config = load_config()
    
    try:
        train_df, test_df = load_training_data(
            db_path=config.data_paths.raw_db,
            train_start_date=config.datasets_loading.train_start_date,
            train_end_date=config.datasets_loading.train_end_date,
            test_start_date=config.datasets_loading.test_start_date,
            test_end_date=config.datasets_loading.test_end_date
        )
    except FileNotFoundError as e:
        logger.error(f"\n❌ {str(e)}")
        logger.error("Please download the dataset and place it in data/raw/")
        return

    # =========================================================================
    # STEP 2: FEATURE ENGINEERING
    # =========================================================================
    print("\n" + "="*80)
    print("STEP 2: FEATURE ENGINEERING")
    print("="*80)

    feature_builder = FeatureBuilder(rolling_window=config.features.rolling_window)

    print("\nBuilding training features...")
    train_features = feature_builder.create_all_features(train_df)

    print("\nBuilding test features...")
    test_features = feature_builder.create_all_features(test_df)

    # Save features
    features_dir = Path("data/features")
    features_dir.mkdir(exist_ok=True, parents=True)

    train_features.to_csv(features_dir / "train_features.csv", index=False)
    test_features.to_csv(features_dir / "test_features.csv", index=False)
    logger.info(f"\n✓ Features saved to {features_dir}/")

    # =========================================================================
    # STEP 3: PREPARE TRAINING DATA
    # =========================================================================
    print("\n" + "="*80)
    print("STEP 3: PREPARING TRAINING DATA")
    print("="*80)

    # Define targets and features
    target_cols = config.features.targets
    exclude_cols = config.features.exclude

    feature_cols = [col for col in train_features.columns if col not in exclude_cols]

    X_train = train_features[feature_cols]
    y_train = train_features[target_cols]
    X_test = test_features[feature_cols]
    y_test = test_features[target_cols]

    logger.info(f"\nDataset Summary:")
    logger.info(f"  Training games:   {len(X_train):,}")
    logger.info(f"  Test games:       {len(X_test):,}")
    logger.info(f"  Features:         {len(feature_cols)}")
    logger.info(f"  Targets:          {', '.join(target_cols)}")

    # Show sample feature names
    logger.info(f"\nSample features:")
    for i, col in enumerate(feature_cols[:10], 1):
        logger.info(f"  {i:2d}. {col}")
    if len(feature_cols) > 10:
        logger.info(f"  ... and {len(feature_cols) - 10} more")

    # =========================================================================
    # STEP 4: TRAIN MODEL
    # =========================================================================
    print("\n" + "="*80)
    print("STEP 4: TRAINING MODEL")
    print("="*80)

    # Train CatBoost model
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

    train_metrics, test_metrics = predictor.train(
        X_train, y_train,
        X_test, y_test
    )

    # =========================================================================
    # STEP 5: ANALYZE RESULTS
    # =========================================================================
    print("\n" + "="*80)
    print("STEP 5: ANALYZING RESULTS")
    print("="*80)

    # Feature importance
    print("\n📊 TOP 20 MOST IMPORTANT FEATURES:")
    print("-" * 80)
    importance_df = predictor.get_feature_importance(top_n=20)
    print(importance_df.to_string(index=False))

    # Save feature importance
    reports_dir = Path("outputs/reports")
    reports_dir.mkdir(exist_ok=True, parents=True)
    importance_df.to_csv(reports_dir / "feature_importance.csv", index=False)

    # Example predictions
    print("\n" + "="*80)
    print("EXAMPLE PREDICTIONS (First 10 test games)")
    print("="*80)

    predictions = predictor.predict(X_test.head(10))
    examples_df = pd.DataFrame({
        'Date': test_features.head(10)['GAME_DATE'].values,
        'Home_Team': test_features.head(10)['HOME_TEAM_ID'].values,
        'Away_Team': test_features.head(10)['VISITOR_TEAM_ID'].values,
        'Actual_Home': y_test.head(10)['PTS_home'].values,
        'Actual_Away': y_test.head(10)['PTS_away'].values,
        'Pred_Home': predictions[:, 0].round(1),
        'Pred_Away': predictions[:, 1].round(1),
        'Actual_Diff': (y_test.head(10)['PTS_home'] - y_test.head(10)['PTS_away']).values,
        'Pred_Diff': (predictions[:, 0] - predictions[:, 1]).round(1),
    })

    print(examples_df.to_string(index=False))

    # =========================================================================
    # STEP 6: SAVE MODEL
    # =========================================================================
    print("\n" + "="*80)
    print("STEP 6: SAVING MODEL")
    print("="*80)

    models_dir = Path("data/models")
    models_dir.mkdir(exist_ok=True, parents=True)

    model_path = models_dir / "score_predictor.pkl"
    predictor.save(model_path)

    # Save training metadata
    metadata = {
        'train_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'train_games': len(X_train),
        'test_games': len(X_test),
        'features': len(feature_cols),
        'train_metrics': train_metrics,
        'test_metrics': test_metrics,
        'model_type': 'catboost',
        'rolling_window': config.features.rolling_window,
        'random_state': config.model.random_state,
        'config': vars(config)  # Convert to dict for JSON
    }

    import json
    with open(models_dir / "training_metadata.json", 'w') as f:
        # Convert numpy types to python types for JSON serialization
        metadata_json = {
            k: (float(v) if isinstance(v, (np.floating, np.integer)) else v)
            for k, v in metadata.items()
        }
        json.dump(metadata_json, f, indent=2, default=str)

    logger.info(f"✓ Model saved to {model_path}")
    logger.info(f"✓ Metadata saved to {models_dir}/training_metadata.json")

    # =========================================================================
    # FINAL SUMMARY
    # =========================================================================
    print("\n" + "="*80)
    print("TRAINING COMPLETE - FINAL SUMMARY")
    print("="*80)

    print(f"\n🎯 TEST SET PERFORMANCE:")
    print(f"  Point Differential MAE:  {test_metrics['diff_mae']:.2f} points")
    print(f"  Differential within ±5:  {test_metrics['diff_within_5']:.1%}")
    print(f"  Win/Loss Accuracy:       {test_metrics['win_accuracy']:.1%}")

    print(f"\n📊 INDIVIDUAL SCORES:")
    print(f"  Home Score MAE:          {test_metrics['home_mae']:.2f} points")
    print(f"  Away Score MAE:          {test_metrics['away_mae']:.2f} points")

    print(f"\n💾 SAVED FILES:")
    print(f"  Model:                   {model_path}")
    print(f"  Features:                {features_dir}/")
    print(f"  Reports:                 {reports_dir}/")

    print(f"\n✅ SUCCESS! Model ready for predictions.")
    print(f"   Use: python predict_game.py --home TEAM1 --away TEAM2")

    print("\n" + "="*80)
    print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80 + "\n")


if __name__ == "__main__":
    import numpy as np  # Needed for metadata serialization
    main()