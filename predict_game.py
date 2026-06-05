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
        n_games = max(config.features.rolling_windows) * 3

        home_recent = loader.load_recent_team_games(home_team_id, n_games)
        away_recent = loader.load_recent_team_games(away_team_id, n_games)

        recent_games = (
            pd.concat([home_recent, away_recent])
            .drop_duplicates('GAME_ID')
            .sort_values('GAME_DATE')
            .reset_index(drop=True)
        )

        prediction_date = pd.Timestamp(game_date).normalize() if game_date else pd.Timestamp.now().normalize()

        if game_date:
            recent_games = recent_games[recent_games['GAME_DATE'] <= prediction_date]

        logger.info(f"Loaded {len(recent_games)} recent games for both teams")

        # Inject a synthetic upcoming-game row so the feature builder computes all
        # rolling, H2H, and matchup features correctly for this exact matchup and date.
        # Outcome columns are zeroed — safe because all rolling features use shift(1),
        # so the synthetic row's unknown outcome never affects its own feature values.
        current_season_id = recent_games.iloc[-1]['SEASON_ID']
        synthetic_row = {col: 0 for col in recent_games.columns}
        synthetic_row.update({
            'GAME_ID': 'upcoming',
            'GAME_DATE': prediction_date,
            'HOME_TEAM_ID': home_team_id,
            'AWAY_TEAM_ID': away_team_id,
            'SEASON_ID': current_season_id,
            'SEASON_TYPE': 'Regular Season',
        })
        all_games = pd.concat(
            [recent_games, pd.DataFrame([synthetic_row])],
            ignore_index=True,
        ).sort_values('GAME_DATE').reset_index(drop=True)

        feature_builder = FeatureBuilder(
            rolling_windows=config.features.rolling_windows,
            h2h_margin_window=config.features.h2h_margin_window,
            h2h_win_rate_window=config.features.h2h_win_rate_window,
        )
        features_df = feature_builder.create_all_features(all_games)

        game_features = features_df[
            (features_df['HOME_TEAM_ID'] == home_team_id) &
            (features_df['AWAY_TEAM_ID'] == away_team_id) &
            (pd.to_datetime(features_df['GAME_DATE']).dt.normalize() == prediction_date)
        ]

        if game_features.empty:
            logger.error("Could not compute features for this matchup. Not enough recent game history for one or both teams.")
            return

        feature_cols = predictor.feature_names
        prediction_features = game_features[feature_cols].iloc[[-1]]
        prediction = predictor.predict(prediction_features)[0]

        home_score = round(prediction[0])
        away_score = round(prediction[1])
        point_diff = home_score - away_score

        winner = "Home" if point_diff > 0 else "Away"
        logger.info(f"\nPrediction ({prediction_date.date()}): Team {home_team_id} {home_score} - {away_score} Team {away_team_id}")
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