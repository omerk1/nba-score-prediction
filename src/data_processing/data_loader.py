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
    """Loads and preprocesses NBA game data from SQLite database."""

    # Shared SELECT clause — aliases map actual DB column names to downstream names
    _GAME_SELECT = """
        SELECT
            game_id          AS GAME_ID,
            game_date        AS GAME_DATE,
            season_id        AS SEASON_ID,
            team_id_home     AS HOME_TEAM_ID,
            team_id_away     AS AWAY_TEAM_ID,
            pts_home         AS PTS_home,
            pts_away         AS PTS_away,
            CASE WHEN wl_home = 'W' THEN 1 ELSE 0 END AS HOME_TEAM_WINS,
            fg_pct_home      AS FG_PCT_home,
            ft_pct_home      AS FT_PCT_home,
            fg3_pct_home     AS FG3_PCT_home,
            ast_home         AS AST_home,
            reb_home         AS REB_home,
            fg_pct_away      AS FG_PCT_away,
            ft_pct_away      AS FT_PCT_away,
            fg3_pct_away     AS FG3_PCT_away,
            ast_away         AS AST_away,
            reb_away         AS REB_away
        FROM game
    """

    def __init__(self, db_path: str) -> None:
        self.db_path = Path(db_path)
        self.conn = None
        if not self.db_path.exists():
            raise FileNotFoundError(
                f"Database not found at {self.db_path}\n"
                f"Please download from: https://www.kaggle.com/datasets/wyattowalsh/basketball"
            )

    def connect(self):
        if self.conn is None:
            self.conn = sqlite3.connect(self.db_path)
        return self.conn

    def close(self):
        if self.conn:
            self.conn.close()
            self.conn = None

    def _post_process(self, df: pd.DataFrame) -> pd.DataFrame:
        df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])
        df['SEASON_ID'] = df['SEASON_ID'].astype(int)
        df['POINT_DIFF'] = df['PTS_home'] - df['PTS_away']
        df['TOTAL_POINTS'] = df['PTS_home'] + df['PTS_away']
        return df

    def load_games(
            self,
            start_date: Optional[str] = None,
            end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """Load games filtered by date range or season."""
        self.connect()

        conditions = ["WHERE 1=1"]
        params: list = []
        if start_date:
            conditions.append("AND game_date >= ?")
            params.append(start_date)
        if end_date:
            conditions.append("AND game_date <= ?")
            params.append(end_date)

        query = self._GAME_SELECT + " ".join(conditions) + " ORDER BY game_date"
        df = pd.read_sql(query, self.conn, params=params or None)
        df = self._post_process(df)
        logger.info(f"Loaded {len(df):,} games ({df['GAME_DATE'].min().date()} – {df['GAME_DATE'].max().date()})")
        return df

    def load_recent_team_games(self, team_id: int, n_games: int) -> pd.DataFrame:
        """Load the last n_games for a team (regardless of home/away role)."""
        self.connect()
        query = (
            self._GAME_SELECT
            + "WHERE team_id_home = ? OR team_id_away = ? "
            + "ORDER BY game_date DESC LIMIT ?"
        )
        df = pd.read_sql(query, self.conn, params=[team_id, team_id, n_games])
        df = self._post_process(df)
        return df.sort_values('GAME_DATE').reset_index(drop=True)

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
            "SELECT MIN(GAME_DATE) as min_date, MAX(GAME_DATE) as max_date FROM game",
            self.conn
        )
        summary['date_range'] = (dates['min_date'].iloc[0], dates['max_date'].iloc[0])

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
        val_start_date: str,
        val_end_date: str,
        test_start_date: str,
        test_end_date: Optional[str] = None
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load train, validation, and test splits using time-based boundaries."""
    loader = NBADataLoader(db_path=db_path)
    try:
        train_df = loader.load_games(start_date=train_start_date, end_date=train_end_date)
        val_df = loader.load_games(start_date=val_start_date, end_date=val_end_date)
        test_df = loader.load_games(start_date=test_start_date, end_date=test_end_date)
        return train_df, val_df, test_df
    finally:
        loader.close()


if __name__ == "__main__":
    config = load_config()

    train_df, val_df, test_df = load_training_data(
        db_path=config.data_paths.raw_db,
        train_start_date=config.datasets_loading.train_start_date,
        train_end_date=config.datasets_loading.train_end_date,
        val_start_date=config.datasets_loading.validation_start_date,
        val_end_date=config.datasets_loading.validation_end_date,
        test_start_date=config.datasets_loading.test_start_date,
        test_end_date=config.datasets_loading.test_end_date,
    )
    print(f"Train: {len(train_df):,} | Val: {len(val_df):,} | Test: {len(test_df):,} games")
    print(train_df[['GAME_DATE', 'HOME_TEAM_ID', 'AWAY_TEAM_ID', 'PTS_home', 'PTS_away', 'POINT_DIFF']].head())