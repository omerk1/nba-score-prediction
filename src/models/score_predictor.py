"""
NBA Score Prediction Model
===========================

Multi-output regression model to predict both home and away scores.
Focus on point differential accuracy while maintaining realistic scores.

Author: NBA Prediction Project
Date: 2024
"""

import logging
from typing import Optional
import joblib

import pandas as pd
import numpy as np

import xgboost as xgb
import lightgbm as lgb

from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ScorePredictor:
    """
    Predicts NBA game scores using multi-output regression.

    Predicts both home_score and away_score simultaneously,
    with custom evaluation focusing on point differential accuracy.
    """

    def __init__(
            self,
            model_type: str = 'xgboost',
            random_state: int = 42,
            **model_params
    ):
        """
        Initialize the score predictor.

        Args:
            model_type: 'xgboost' or 'lightgbm'
            random_state: Random seed for reproducibility
            **model_params: Additional parameters for the model
        """
        self.model_type = model_type
        self.random_state = random_state
        self.model_params = model_params
        self.model = None
        self.feature_names = None

        logger.info(f"ScorePredictor initialized with {model_type}")

    def _create_model(self):
        """Create the base model"""
        if self.model_type == 'xgboost':
            base_model = xgb.XGBRegressor(
                random_state=self.random_state,
                n_estimators=self.model_params.get('n_estimators', 200),
                max_depth=self.model_params.get('max_depth', 6),
                learning_rate=self.model_params.get('learning_rate', 0.1),
                subsample=self.model_params.get('subsample', 0.8),
                colsample_bytree=self.model_params.get('colsample_bytree', 0.8),
                objective='reg:squarederror'
            )
        elif self.model_type == 'lightgbm':
            base_model = lgb.LGBMRegressor(
                random_state=self.random_state,
                n_estimators=self.model_params.get('n_estimators', 200),
                max_depth=self.model_params.get('max_depth', 6),
                learning_rate=self.model_params.get('learning_rate', 0.1),
                subsample=self.model_params.get('subsample', 0.8),
                colsample_bytree=self.model_params.get('colsample_bytree', 0.8),
            )
        else:
            raise ValueError(f"Unknown model_type: {self.model_type}")

        # Wrap in MultiOutputRegressor for simultaneous prediction
        return MultiOutputRegressor(base_model)

    def train(
            self,
            X_train: pd.DataFrame,
            y_train: pd.DataFrame,
            X_val: Optional[pd.DataFrame] = None,
            y_val: Optional[pd.DataFrame] = None
    ):
        """
        Train the model.

        Args:
            X_train: Training features
            y_train: Training targets [home_score, away_score]
            X_val: Validation features (optional)
            y_val: Validation targets (optional)
        """
        logger.info("\n" + "=" * 70)
        logger.info("TRAINING MODEL")
        logger.info("=" * 70)

        # Store feature names
        self.feature_names = X_train.columns.tolist()

        # Create and train model
        self.model = self._create_model()

        logger.info(f"Training {self.model_type} on {len(X_train):,} games...")
        logger.info(f"Features: {len(self.feature_names)}")
        logger.info(f"Targets: home_score, away_score")

        self.model.fit(X_train, y_train)

        logger.info("✓ Training complete!")

        # Evaluate on training set
        train_metrics = self.evaluate(X_train, y_train, dataset_name="Training")

        # Evaluate on validation set if provided
        if X_val is not None and y_val is not None:
            val_metrics = self.evaluate(X_val, y_val, dataset_name="Validation")
            return train_metrics, val_metrics

        return train_metrics

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict scores for games.

        Args:
            X: Features

        Returns:
            Array of shape (n_games, 2) with [home_score, away_score]
        """
        if self.model is None:
            raise ValueError("Model not trained yet. Call train() first.")

        return self.model.predict(X)

    def evaluate(
            self,
            X: pd.DataFrame,
            y_true: pd.DataFrame,
            dataset_name: str = "Dataset"
    ) -> dict[str, float]:
        """
        Evaluate model with focus on point differential accuracy.

        Args:
            X: Features
            y_true: True scores [home_score, away_score]
            dataset_name: Name for logging

        Returns:
            Dictionary of evaluation metrics
        """
        # Predict
        y_pred = self.predict(X)

        # Extract individual scores
        home_true = y_true.iloc[:, 0].values
        away_true = y_true.iloc[:, 1].values
        home_pred = y_pred[:, 0]
        away_pred = y_pred[:, 1]

        # Calculate differentials
        diff_true = home_true - away_true
        diff_pred = home_pred - away_pred

        # Calculate totals
        total_true = home_true + away_true
        total_pred = home_pred + away_pred

        # Metrics
        metrics = {
            # Point Differential (MOST IMPORTANT)
            'diff_mae': mean_absolute_error(diff_true, diff_pred),
            'diff_rmse': np.sqrt(mean_squared_error(diff_true, diff_pred)),
            'diff_within_3': np.mean(np.abs(diff_true - diff_pred) <= 3),
            'diff_within_5': np.mean(np.abs(diff_true - diff_pred) <= 5),
            'diff_within_10': np.mean(np.abs(diff_true - diff_pred) <= 10),

            # Individual Scores
            'home_mae': mean_absolute_error(home_true, home_pred),
            'away_mae': mean_absolute_error(away_true, away_pred),
            'home_rmse': np.sqrt(mean_squared_error(home_true, home_pred)),
            'away_rmse': np.sqrt(mean_squared_error(away_true, away_pred)),

            # Total Points
            'total_mae': mean_absolute_error(total_true, total_pred),
            'total_rmse': np.sqrt(mean_squared_error(total_true, total_pred)),

            # Win/Loss Prediction
            'win_accuracy': np.mean((diff_true > 0) == (diff_pred > 0)),

            # Correlation
            'diff_correlation': np.corrcoef(diff_true, diff_pred)[0, 1],
        }

        # Log results
        logger.info(f"\n{'=' * 70}")
        logger.info(f"{dataset_name.upper()} EVALUATION")
        logger.info('=' * 70)
        logger.info(f"\n🎯 POINT DIFFERENTIAL (Primary Metric):")
        logger.info(f"  MAE:             {metrics['diff_mae']:.2f} points")
        logger.info(f"  RMSE:            {metrics['diff_rmse']:.2f} points")
        logger.info(f"  Within ±3:       {metrics['diff_within_3']:.1%}")
        logger.info(f"  Within ±5:       {metrics['diff_within_5']:.1%}")
        logger.info(f"  Within ±10:      {metrics['diff_within_10']:.1%}")
        logger.info(f"  Correlation:     {metrics['diff_correlation']:.3f}")

        logger.info(f"\n📊 INDIVIDUAL SCORES:")
        logger.info(f"  Home MAE:        {metrics['home_mae']:.2f} points")
        logger.info(f"  Away MAE:        {metrics['away_mae']:.2f} points")
        logger.info(f"  Home RMSE:       {metrics['home_rmse']:.2f} points")
        logger.info(f"  Away RMSE:       {metrics['away_rmse']:.2f} points")

        logger.info(f"\n📈 TOTAL POINTS:")
        logger.info(f"  MAE:             {metrics['total_mae']:.2f} points")
        logger.info(f"  RMSE:            {metrics['total_rmse']:.2f} points")

        logger.info(f"\n🏆 WIN/LOSS PREDICTION:")
        logger.info(f"  Accuracy:        {metrics['win_accuracy']:.1%}")

        return metrics

    def get_feature_importance(self, top_n: int = 20) -> pd.DataFrame:
        """
        Get feature importance.

        Args:
            top_n: Number of top features to return

        Returns:
            DataFrame with feature importance
        """
        if self.model is None:
            raise ValueError("Model not trained yet.")

        # Average importance across both output models
        importances = []
        for estimator in self.model.estimators_:
            if hasattr(estimator, 'feature_importances_'):
                importances.append(estimator.feature_importances_)

        avg_importance = np.mean(importances, axis=0)

        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': avg_importance
        }).sort_values('importance', ascending=False).head(top_n)

        return importance_df

    def save(self, filepath: str):
        """Save model to disk"""
        if self.model is None:
            raise ValueError("No model to save. Train first.")

        model_data = {
            'model': self.model,
            'feature_names': self.feature_names,
            'model_type': self.model_type,
            'random_state': self.random_state,
            'model_params': self.model_params
        }

        joblib.dump(model_data, filepath)
        logger.info(f"✓ Model saved to {filepath}")

    @classmethod
    def load(cls, filepath: str):
        """Load model from disk"""
        model_data = joblib.load(filepath)

        predictor = cls(
            model_type=model_data['model_type'],
            random_state=model_data['random_state'],
            **model_data['model_params']
        )

        predictor.model = model_data['model']
        predictor.feature_names = model_data['feature_names']

        logger.info(f"✓ Model loaded from {filepath}")
        return predictor


# Example usage
if __name__ == "__main__":
    from pathlib import Path

    # Load processed features
    features_dir = Path("data/features")
    train_df = pd.read_csv(features_dir / "train_features.csv")
    test_df = pd.read_csv(features_dir / "test_features.csv")

    # Prepare features and targets
    target_cols = ['PTS_home', 'PTS_away']
    exclude_cols = [
        'GAME_ID', 'GAME_DATE', 'SEASON', 'HOME_TEAM_ID', 'VISITOR_TEAM_ID',
        'PTS_home', 'PTS_away', 'POINT_DIFF', 'TOTAL_POINTS', 'HOME_TEAM_WINS',
        'matchup_key'
    ]
    feature_cols = [col for col in train_df.columns if col not in exclude_cols]

    X_train = train_df[feature_cols]
    y_train = train_df[target_cols]
    X_test = test_df[feature_cols]
    y_test = test_df[target_cols]

    logger.info(f"\nDataset sizes:")
    logger.info(f"  Train: {len(X_train):,} games")
    logger.info(f"  Test:  {len(X_test):,} games")
    logger.info(f"  Features: {len(feature_cols)}")

    # Train model
    predictor = ScorePredictor(
        model_type='xgboost',
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1
    )

    train_metrics, test_metrics = predictor.train(X_train, y_train, X_test, y_test)

    # Show feature importance
    print("\n" + "=" * 70)
    print("TOP 20 MOST IMPORTANT FEATURES")
    print("=" * 70)
    importance_df = predictor.get_feature_importance(top_n=20)
    print(importance_df.to_string(index=False))

    # Save model
    models_dir = Path("data/models")
    models_dir.mkdir(exist_ok=True, parents=True)
    predictor.save(models_dir / "score_predictor.pkl")

    print("\n" + "=" * 70)
    print("✓ MODEL TRAINING COMPLETE")
    print("=" * 70)