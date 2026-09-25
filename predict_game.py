"""
NBA Score Prediction - Game Predictor
======================================

Predict the score of an upcoming NBA game using the trained model.

Usage:
    python predict_game.py --home 1610612747 --away 1610612744
    python predict_game.py --home 1610612747 --away 1610612744 --date 2026-03-15
"""

import argparse
import logging
import sys
import traceback
from pathlib import Path

import pandas as pd

# Add src to path
sys.path.append(str(Path(__file__).parent))

from src.models.score_predictor import ScorePredictor
from src.serving.live_features import build_live_game_features
from src.utils.config_loader import load_config

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def load_model(model_path: str = "data/models/score_predictor.pkl"):
    """Load the trained model"""
    if not Path(model_path).exists():
        raise FileNotFoundError(
            f"Model not found at {model_path}\n" f"Please train the model first: python train_model.py"
        )
    return ScorePredictor.load(model_path)


def predict_game(home_team_id: int, away_team_id: int, game_date: str = None):
    """
    Predict the score of a game.

    Args:
        home_team_id: Home team ID
        away_team_id: Away team ID
        game_date: Date of the game (default: today)
    """
    config = load_config()
    model_path = Path(config.data_paths.models) / "score_predictor.pkl"

    predictor = ScorePredictor.load(str(model_path))
    logger.info(f"Model loaded from {model_path}")

    game_features = build_live_game_features(home_team_id, away_team_id, game_date, config)
    prediction_date = pd.Timestamp(game_date).normalize() if game_date else pd.Timestamp.now().normalize()

    if game_features.empty:
        logger.error(
            "Could not compute features for this matchup. Not enough recent game history for one or both teams."
        )
        return

    feature_cols = predictor.feature_names
    prediction_features = game_features[feature_cols].iloc[[-1]]
    prediction = predictor.predict(prediction_features)[0]

    home_score = round(prediction[0])
    away_score = round(prediction[1])
    point_diff = home_score - away_score

    winner = "Home" if point_diff > 0 else "Away"
    logger.info(
        f"\nPrediction ({prediction_date.date()}): Team {home_team_id} {home_score} - {away_score} Team {away_team_id}"
    )
    logger.info(f"Winner: {winner} by {abs(point_diff)} | Total: {home_score + away_score}")


def main():
    """Main execution"""
    parser = argparse.ArgumentParser(
        description="Predict NBA game scores",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python predict_game.py --home 1610612747 --away 1610612744
  python predict_game.py --home 1610612747 --away 1610612744 --date 2024-03-15
        """,
    )

    parser.add_argument("--home", type=int, required=True, help="Home team ID")

    parser.add_argument("--away", type=int, required=True, help="Away team ID")

    parser.add_argument(
        "--date", type=str, default=None, help="Game date (YYYY-MM-DD format). Default: today"
    )

    args = parser.parse_args()

    try:
        predict_game(home_team_id=args.home, away_team_id=args.away, game_date=args.date)
    except Exception as e:
        logger.error(f"\nError: {str(e)}")
        logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()
