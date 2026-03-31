"""
NBA Score Prediction - Game Predictor
======================================

Predict the score of an upcoming NBA game using the trained model.

Usage:
    python predict_game.py --home "LAL" --away "GSW"
    python predict_game.py --game-date "2024-03-15"

Author: NBA Prediction Project
Date: 2024
"""

import sys
import logging
from pathlib import Path
from datetime import datetime, timedelta
import argparse
import pandas as pd


# Add src to path
sys.path.append(str(Path(__file__).parent))

from src.data_processing.data_loader import NBADataLoader
from src.feature_engineering.feature_builder import FeatureBuilder
from src.models.score_predictor import ScorePredictor

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
    logger.info("\n" + "=" * 80)
    logger.info("NBA SCORE PREDICTION")
    logger.info("=" * 80)

    # Load model
    logger.info("\nLoading model...")
    predictor = load_model()
    logger.info("✓ Model loaded")

    # Load recent game data to compute features
    logger.info("\nLoading recent game data...")
    loader = NBADataLoader()

    try:
        # Get last 30 days of games for computing rolling features
        end_date = game_date if game_date else datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.strptime(end_date, '%Y-%m-%d') - timedelta(days=30)).strftime('%Y-%m-%d')

        recent_games = loader.load_games(start_date=start_date, end_date=end_date)
        logger.info(f"✓ Loaded {len(recent_games)} recent games")

        # Build features for all recent games
        feature_builder = FeatureBuilder(rolling_window=10)
        features_df = feature_builder.create_all_features(recent_games)

        # Find the most recent game for each team to get current features
        home_recent = features_df[
            (features_df['HOME_TEAM_ID'] == home_team_id) |
            (features_df['VISITOR_TEAM_ID'] == home_team_id)
            ].iloc[-1] if len(features_df) > 0 else None

        away_recent = features_df[
            (features_df['HOME_TEAM_ID'] == away_team_id) |
            (features_df['VISITOR_TEAM_ID'] == away_team_id)
            ].iloc[-1] if len(features_df) > 0 else None

        if home_recent is None or away_recent is None:
            logger.error("❌ Could not find recent games for both teams")
            logger.error("   Teams might not have played recently or IDs are incorrect")
            return

        # Construct features for the matchup
        # This is simplified - in practice, you'd compute proper matchup features
        logger.info("\nConstructing features for prediction...")

        # Get feature columns
        feature_cols = predictor.feature_names

        # Create a synthetic game row with features from both teams
        # This is a simplified approach - you'd want more sophisticated feature construction
        prediction_features = pd.DataFrame([{
            col: home_recent.get(col.replace('home_', ''), away_recent.get(col.replace('away_', ''), 0))
            for col in feature_cols
        }])

        # Make prediction
        logger.info("\nMaking prediction...")
        prediction = predictor.predict(prediction_features)[0]

        home_score = round(prediction[0])
        away_score = round(prediction[1])
        point_diff = home_score - away_score

        # Display results
        logger.info("\n" + "=" * 80)
        logger.info("PREDICTION RESULTS")
        logger.info("=" * 80)

        logger.info(f"\n🏀 Matchup: Team {home_team_id} (Home) vs Team {away_team_id} (Away)")
        if game_date:
            logger.info(f"📅 Date: {game_date}")

        logger.info(f"\n🎯 PREDICTED SCORES:")
        logger.info(f"  Home Team: {home_score} points")
        logger.info(f"  Away Team: {away_score} points")
        logger.info(f"  Total:     {home_score + away_score} points")

        logger.info(f"\n📊 PREDICTED OUTCOME:")
        winner = "Home" if point_diff > 0 else "Away"
        logger.info(f"  Winner:           Team {home_team_id if point_diff > 0 else away_team_id} ({winner})")
        logger.info(f"  Point Differential: {abs(point_diff)} points")

        logger.info("\n" + "=" * 80)
        logger.info("Note: This is a statistical prediction based on recent performance.")
        logger.info("Actual results may vary due to injuries, lineup changes, and other factors.")
        logger.info("=" * 80 + "\n")

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
        logger.error(f"\n❌ Error: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()