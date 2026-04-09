"""
Data Loader for NBA Score Prediction
=====================================

Handles loading and initial processing of the Kaggle NBA SQLite database.
Focuses on game-level data needed for score prediction.

Author: NBA Prediction Project
Date: 2024
"""

import logging
from pathlib import Path
from typing import Optional

import sqlite3
import pandas as pd
from src.utils.config_loader import load_config, get_config_value

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NBADataLoader:
    """
    Loads and preprocesses NBA game data from SQLite database.

    Main responsibilities:
    - Load game results with scores
    - Load team statistics
    - Join and validate data
    - Handle missing values
    """

    def __init__(self, db_path: str) -> None:
        """
        Initialize the data loader.

        Args:
            db_path: Path to the SQLite database file
        """
        self.db_path = Path(db_path)
        self.conn = None
        self._validate_database()

    def _validate_database(self):
        """Check if database exists and is accessible"""
        if not self.db_path.exists():
            raise FileNotFoundError(
                f"Database not found at {self.db_path}\n"
                f"Please download from: https://www.kaggle.com/datasets/wyattowalsh/basketball"
            )
        logger.info(f"✓ Database found: {self.db_path}")

    def connect(self):
        """Establish database connection"""
        if self.conn is None:
            self.conn = sqlite3.connect(self.db_path)
            logger.info("✓ Database connection established")
        return self.conn

    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()
            self.conn = None
            logger.info("✓ Database connection closed")

    def load_games(
            self,
            start_date: Optional[str] = None,
            end_date: Optional[str] = None,
            season: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Load game-level data with scores.

        Args:
            start_date: Filter games from this date (YYYY-MM-DD)
            end_date: Filter games until this date (YYYY-MM-DD)
            season: Filter by specific season (e.g., 2023 for 2023-24 season)

        Returns:
            DataFrame with game results
        """
        self.connect()

        # Base query
        query = """
        SELECT 
            GAME_ID,
            GAME_DATE_EST as GAME_DATE,
            SEASON,
            HOME_TEAM_ID,
            VISITOR_TEAM_ID,
            PTS_home,
            PTS_away,
            HOME_TEAM_WINS,
            FG_PCT_home,
            FT_PCT_home,
            FG3_PCT_home,
            AST_home,
            REB_home,
            FG_PCT_away,
            FT_PCT_away,
            FG3_PCT_away,
            AST_away,
            REB_away
        FROM game
        WHERE 1=1
        """

        # Add filters
        conditions = []
        if start_date:
            conditions.append(f"AND GAME_DATE_EST >= '{start_date}'")
        if end_date:
            conditions.append(f"AND GAME_DATE_EST <= '{end_date}'")
        if season:
            conditions.append(f"AND SEASON = {season}")

        query += " ".join(conditions)
        query += " ORDER BY GAME_DATE_EST"

        logger.info("Loading game data...")
        df = pd.read_sql(query, self.conn)

        # Data type conversions
        df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])
        df['SEASON'] = df['SEASON'].astype(int)

        # Calculate point differential (home team perspective)
        df['POINT_DIFF'] = df['PTS_home'] - df['PTS_away']
        df['TOTAL_POINTS'] = df['PTS_home'] + df['PTS_away']

        logger.info(f"✓ Loaded {len(df):,} games")
        logger.info(f"  Date range: {df['GAME_DATE'].min()} to {df['GAME_DATE'].max()}")
        logger.info(f"  Seasons: {df['SEASON'].min()} to {df['SEASON'].max()}")

        return df

    def get_data_summary(self) -> dict:
        """
        Get summary statistics about the database.

        Returns:
            Dictionary with summary information
        """
        self.connect()

        summary = {}

        # Count games
        games = pd.read_sql("SELECT COUNT(*) as count FROM game", self.conn)
        summary['total_games'] = games['count'].iloc[0]

        # Date range
        dates = pd.read_sql(
            "SELECT MIN(GAME_DATE_EST) as min_date, MAX(GAME_DATE_EST) as max_date FROM game",
            self.conn
        )
        summary['date_range'] = (dates['min_date'].iloc[0], dates['max_date'].iloc[0])

        # Seasons
        seasons = pd.read_sql(
            "SELECT MIN(SEASON) as min_season, MAX(SEASON) as max_season FROM game",
            self.conn
        )
        summary['season_range'] = (seasons['min_season'].iloc[0], seasons['max_season'].iloc[0])

        # Teams
        teams = pd.read_sql("SELECT COUNT(DISTINCT TEAM_ID) as count FROM team", self.conn)
        summary['total_teams'] = teams['count'].iloc[0]

        return summary

    def __enter__(self):
        """Context manager entry"""
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.close()


def load_training_data(
        db_path: str,
        train_start_date: str,
        train_end_date: str,
        test_start_date: str,
        test_end_date: Optional[str] = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Convenience function to load train and test data with time-based split.

    Args:
        db_path: Path to the SQLite database file
        train_start_date: Start of training period
        train_end_date: End of training period
        test_start_date: Start of test period
        test_end_date: End of test period (None = all available)

    Returns:
        Tuple of (train_df, test_df)
    """
    loader = NBADataLoader(db_path=db_path)

    try:
        # Load training data
        logger.info("\n" + "=" * 70)
        logger.info("LOADING TRAINING DATA")
        logger.info("=" * 70)
        train_df = loader.load_games(start_date=train_start_date, end_date=train_end_date)

        # Load test data
        logger.info("\n" + "=" * 70)
        logger.info("LOADING TEST DATA")
        logger.info("=" * 70)
        test_df = loader.load_games(start_date=test_start_date, end_date=test_end_date)

        return train_df, test_df

    finally:
        loader.close()


# Example usage
if __name__ == "__main__":
    config = load_config()
    
    train_df, test_df = load_training_data(
        db_path=config.data_paths.raw_db,
        train_start_date=config.datasets_loading.train_start_date,
        train_end_date=config.datasets_loading.train_end_date,
        test_start_date=config.datasets_loading.test_start_date
    )
    print(f"\nTrain set: {len(train_df):,} games")
    print(f"Test set:  {len(test_df):,} games")

    print("\nSample training data:")
    print(train_df[['GAME_DATE', 'HOME_TEAM_ID', 'VISITOR_TEAM_ID',
                    'PTS_home', 'PTS_away', 'POINT_DIFF']].head())