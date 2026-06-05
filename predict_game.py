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

from src.data_processing.data_loader import NBADataLoader
from src.feature_engineering.feature_builder import FeatureBuilder
from src.models.score_predictor import ScorePredictor
from src.utils.config_loader import load_config

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def load_model(model_path: str = "data/models/score_predictor.pkl"):
    """Load the trained model"""
    if not Path(model_path).exists():
        raise FileNotFoundError(
            f"Model not found at {model_path}\n"
            f"Please train the model first: python train_model.py"
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

    loader = NBADataLoader(db_path=config.data_paths.raw_db)

    try:
        # Load last rolling_window*3 games per team — enough to fill all rolling features
        # regardless of season start or schedule gaps
        n_games = config.features.rolling_window * 3

        home_recent = loader.load_recent_team_games(home_team_id, n_games)
        away_recent = loader.load_recent_team_games(away_team_id, n_games)

        recent_games = (
            pd.concat([home_recent, away_recent])
            .drop_duplicates('GAME_ID')
            .sort_values('GAME_DATE')
            .reset_index(drop=True)
        )

        if game_date:
            recent_games = recent_games[recent_games['GAME_DATE'] <= game_date]

        logger.info(f"Loaded {len(recent_games)} recent games for both teams")

        feature_builder = FeatureBuilder(
            rolling_window=config.features.rolling_window,
            h2h_margin_window=config.features.h2h_margin_window,
            h2h_win_rate_window=config.features.h2h_win_rate_window,
        )
        features_df = feature_builder.create_all_features(recent_games)

        home_games = features_df[features_df['HOME_TEAM_ID'] == home_team_id]
        away_games = features_df[features_df['AWAY_TEAM_ID'] == away_team_id]

        if home_games.empty or away_games.empty:
            logger.error("Not enough games found for both teams in their roles. Check team IDs.")
            return

        home_row = home_games.iloc[-1]
        away_row = away_games.iloc[-1]

        # Build prediction row: home_* cols from home team's row, away_* cols from away team's row
        feature_cols = predictor.feature_names
        row = {}
        for col in feature_cols:
            if col.startswith('away_'):
                row[col] = away_row[col] if col in away_row.index else 0
            else:
                row[col] = home_row[col] if col in home_row.index else 0

        prediction_features = pd.DataFrame([row])
        prediction = predictor.predict(prediction_features)[0]

        home_score = round(prediction[0])
        away_score = round(prediction[1])
        point_diff = home_score - away_score

        winner = "Home" if point_diff > 0 else "Away"
        date_str = f" ({game_date})" if game_date else ""
        logger.info(f"\nPrediction{date_str}: Team {home_team_id} {home_score} - {away_score} Team {away_team_id}")
        logger.info(f"Winner: {winner} by {abs(point_diff)} | Total: {home_score + away_score}")

    finally:
        loader.close()


def main():
    """Main execution"""
    parser = argparse.ArgumentParser(
        description="Predict NBA game scores",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python predict_game.py --home 1610612747 --away 1610612744
  python predict_game.py --home 1610612747 --away 1610612744 --date 2024-03-15
        """
    )

    parser.add_argument(
        '--home',
        type=int,
        required=True,
        help='Home team ID'
    )

    parser.add_argument(
        '--away',
        type=int,
        required=True,
        help='Away team ID'
    )

    parser.add_argument(
        '--date',
        type=str,
        default=None,
        help='Game date (YYYY-MM-DD format). Default: today'
    )

    args = parser.parse_args()

    try:
        predict_game(
            home_team_id=args.home,
            away_team_id=args.away,
            game_date=args.date
        )
    except Exception as e:
        logger.error(f"\nError: {str(e)}")
        logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()