"""
Feature Engineering for NBA Score Prediction
=============================================

Creates statistical and matchup-aware features for predicting game scores.
Focus on capturing team strength and style mismatches.

Author: NBA Prediction Project
Date: 2024
"""

import logging
from pathlib import Path
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeatureBuilder:
    """
    Builds features for NBA game prediction.

    Features include:
    - Rolling averages (recent performance)
    - Style metrics (pace, shooting, defense)
    - Matchup features (style advantages/mismatches)
    - Situational features (rest, home advantage)
    - Head-to-head history
    """

    def __init__(self, rolling_window: int = 10):
        """
        Initialize feature builder.

        Args:
            rolling_window: Number of recent games for rolling stats
        """
        self.rolling_window = rolling_window
        logger.info(f"FeatureBuilder initialized with rolling_window={rolling_window}")

    def create_all_features(self, games_df: pd.DataFrame) -> pd.DataFrame:
        """
        Create complete feature set for all games.

        Args:
            games_df: DataFrame with game results

        Returns:
            DataFrame with all features
        """
        logger.info("\n" + "=" * 70)
        logger.info("BUILDING FEATURES")
        logger.info("=" * 70)

        # Make a copy to avoid modifying original
        df = games_df.copy()

        # Ensure data is sorted by date
        df = df.sort_values('GAME_DATE').reset_index(drop=True)

        # 1. Basic features
        logger.info("Creating basic features...")
        df = self._add_basic_features(df)

        # 2. Rolling statistics
        logger.info("Creating rolling statistics...")
        df = self._add_rolling_features(df)

        # 3. Rest and schedule features
        logger.info("Creating rest/schedule features...")
        df = self._add_rest_features(df)

        # 4. Style features
        logger.info("Creating style features...")
        df = self._add_style_features(df)

        # 5. Matchup features
        logger.info("Creating matchup features...")
        df = self._add_matchup_features(df)

        # 6. Head-to-head features
        logger.info("Creating head-to-head features...")
        df = self._add_h2h_features(df)

        # Drop rows with insufficient data (early games without rolling stats)
        initial_rows = len(df)
        df = df.dropna(subset=self._get_feature_columns(df))
        dropped_rows = initial_rows - len(df)

        logger.info(f"\n✓ Feature engineering complete!")
        logger.info(f"  Total features: {len(self._get_feature_columns(df))}")
        logger.info(f"  Dropped {dropped_rows} rows due to insufficient data")
        logger.info(f"  Final dataset: {len(df):,} games")

        return df

    def _add_basic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add basic game-level features"""
        # Home indicator
        df['is_home'] = 1

        # Season progress (0 to 1)
        df['season_progress'] = df.groupby('SEASON').cumcount() / df.groupby('SEASON')['SEASON'].transform('count')

        return df

    def _add_rolling_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add rolling average features for recent performance.

        Captures team momentum and current form.
        """
        window = self.rolling_window

        for team_col, pts_col, prefix in [
            ('HOME_TEAM_ID', 'PTS_home', 'home'),
            ('VISITOR_TEAM_ID', 'PTS_away', 'away')
        ]:
            # Group by team
            grouped = df.groupby(team_col)

            # Rolling average points scored
            df[f'{prefix}_pts_avg_L{window}'] = grouped[pts_col].transform(
                lambda x: x.shift(1).rolling(window, min_periods=1).mean()
            )

            # Rolling win percentage
            is_win = df['POINT_DIFF'] > 0 if prefix == 'home' else df['POINT_DIFF'] < 0
            df[f'{prefix}_win_pct_L{window}'] = grouped[team_col].transform(
                lambda x: is_win.shift(1).rolling(window, min_periods=1).mean()
            )

            # Rolling average point differential
            point_diff_col = 'POINT_DIFF' if prefix == 'home' else df['POINT_DIFF'] * -1
            df[f'{prefix}_diff_avg_L{window}'] = grouped[team_col].transform(
                lambda x: point_diff_col.shift(1).rolling(window, min_periods=1).mean()
            )

            # Rolling shooting percentages
            for stat in ['FG_PCT', 'FG3_PCT', 'FT_PCT']:
                stat_col = f'{stat}_{prefix.split("_")[0]}'
                if stat_col in df.columns:
                    df[f'{prefix}_{stat.lower()}_L{window}'] = grouped[team_col].transform(
                        lambda x: df[stat_col].shift(1).rolling(window, min_periods=1).mean()
                    )

        return df

    def _add_rest_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add rest and schedule-related features.

        Captures fatigue and travel effects.
        """
        for team_col, prefix in [
            ('HOME_TEAM_ID', 'home'),
            ('VISITOR_TEAM_ID', 'away')
        ]:
            # Days of rest since last game
            df[f'{prefix}_rest_days'] = df.groupby(team_col)['GAME_DATE'].diff().dt.days.fillna(3)

            # Back-to-back games
            df[f'{prefix}_back_to_back'] = (df[f'{prefix}_rest_days'] == 1).astype(int)

            # 3 games in 4 nights (fatigue indicator)
            df[f'{prefix}_games_in_4_nights'] = df.groupby(team_col).apply(
                lambda x: x['GAME_DATE'].diff().dt.days.rolling(3, min_periods=1).sum() <= 4
            ).reset_index(level=0, drop=True).astype(int)

        return df

    def _add_style_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add team style metrics.

        Captures how teams play (pace, shooting style, etc.)
        These are crucial for matchup analysis.
        """
        window = self.rolling_window

        for team_col, prefix in [
            ('HOME_TEAM_ID', 'home'),
            ('VISITOR_TEAM_ID', 'away')
        ]:
            grouped = df.groupby(team_col)

            # Offensive pace (approximated by total points)
            df[f'{prefix}_pace_L{window}'] = grouped['TOTAL_POINTS'].transform(
                lambda x: x.shift(1).rolling(window, min_periods=1).mean()
            )

            # Three-point shooting rate (approximated)
            if f'FG3_PCT_{prefix.split("_")[0]}' in df.columns:
                df[f'{prefix}_3pt_rate_L{window}'] = grouped[f'FG3_PCT_{prefix.split("_")[0]}'].transform(
                    lambda x: x.shift(1).rolling(window, min_periods=1).mean()
                )

            # Offensive efficiency (points per game)
            pts_col = 'PTS_home' if prefix == 'home' else 'PTS_away'
            df[f'{prefix}_off_eff_L{window}'] = grouped[pts_col].transform(
                lambda x: x.shift(1).rolling(window, min_periods=1).mean()
            )

            # Defensive efficiency (opponent points)
            opp_pts_col = 'PTS_away' if prefix == 'home' else 'PTS_home'
            df[f'{prefix}_def_eff_L{window}'] = grouped[opp_pts_col].transform(
                lambda x: x.shift(1).rolling(window, min_periods=1).mean()
            )

        return df

    def _add_matchup_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add matchup-specific features.

        CRITICAL: This captures style advantages and mismatches.
        Examples:
        - Fast team vs slow team (pace mismatch)
        - Good shooting team vs bad perimeter defense
        - Offensive powerhouse vs defensive team
        """
        window = self.rolling_window

        # Pace mismatch (advantage for faster team)
        if f'home_pace_L{window}' in df.columns and f'away_pace_L{window}' in df.columns:
            df['pace_differential'] = df[f'home_pace_L{window}'] - df[f'away_pace_L{window}']
            df['pace_mismatch_advantage'] = df['pace_differential'] * df[f'home_pace_L{window}']

        # Offensive vs Defensive matchup
        if f'home_off_eff_L{window}' in df.columns and f'away_def_eff_L{window}' in df.columns:
            # Home offense vs away defense
            df['home_off_vs_away_def'] = df[f'home_off_eff_L{window}'] - df[f'away_def_eff_L{window}']
            # Away offense vs home defense
            df['away_off_vs_home_def'] = df[f'away_off_eff_L{window}'] - df[f'home_def_eff_L{window}']

        # Style mismatch: 3-point shooting
        if f'home_3pt_rate_L{window}' in df.columns:
            # Home 3PT shooting advantage
            df['home_3pt_advantage'] = df[f'home_3pt_rate_L{window}'] - df[f'away_3pt_rate_L{window}']

        # Recent form differential
        if f'home_win_pct_L{window}' in df.columns:
            df['form_differential'] = df[f'home_win_pct_L{window}'] - df[f'away_win_pct_L{window}']

        # Strength differential (point differential trends)
        if f'home_diff_avg_L{window}' in df.columns:
            df['strength_differential'] = df[f'home_diff_avg_L{window}'] - df[f'away_diff_avg_L{window}']

        return df

    def _add_h2h_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add head-to-head historical features.

        Captures direct matchup history between teams.
        """
        # Create matchup key (sorted to ensure consistency)
        df['matchup_key'] = df.apply(
            lambda row: '_'.join(sorted([str(row['HOME_TEAM_ID']), str(row['VISITOR_TEAM_ID'])])),
            axis=1
        )

        # Last 3 meetings: average point differential
        df['h2h_home_margin_L3'] = df.groupby('matchup_key')['POINT_DIFF'].transform(
            lambda x: x.shift(1).rolling(3, min_periods=1).mean()
        )

        # Win rate in last 5 meetings (home team perspective)
        df['h2h_home_win_rate_L5'] = df.groupby('matchup_key').apply(
            lambda x: (x['POINT_DIFF'] > 0).shift(1).rolling(5, min_periods=1).mean()
        ).reset_index(level=0, drop=True)

        return df

    def _get_feature_columns(self, df: pd.DataFrame) -> list[str]:
        """Get list of feature columns (exclude targets and identifiers)"""
        exclude = [
            'GAME_ID', 'GAME_DATE', 'SEASON', 'HOME_TEAM_ID', 'VISITOR_TEAM_ID',
            'PTS_home', 'PTS_away', 'POINT_DIFF', 'TOTAL_POINTS', 'HOME_TEAM_WINS',
            'matchup_key'
        ]

        feature_cols = [col for col in df.columns if col not in exclude]
        return feature_cols

    def get_feature_names(self, df: pd.DataFrame) -> list[str]:
        """Public method to get feature column names"""
        return self._get_feature_columns(df)


# Example usage
if __name__ == "__main__":
    from src.data_processing.data_loader import load_training_data

    # Load data
    train_df, test_df = load_training_data()

    # Build features
    feature_builder = FeatureBuilder(rolling_window=10)

    print("\n" + "=" * 70)
    print("BUILDING TRAINING FEATURES")
    print("=" * 70)
    train_features = feature_builder.create_all_features(train_df)

    print("\n" + "=" * 70)
    print("BUILDING TEST FEATURES")
    print("=" * 70)
    test_features = feature_builder.create_all_features(test_df)

    # Show feature summary
    feature_cols = feature_builder.get_feature_names(train_features)
    print("\n" + "=" * 70)
    print("FEATURE SUMMARY")
    print("=" * 70)
    print(f"\nTotal features: {len(feature_cols)}")
    print("\nFeature categories:")

    categories = {
        'Rolling Stats': [f for f in feature_cols if '_L10' in f or '_L5' in f],
        'Rest/Schedule': [f for f in feature_cols if 'rest' in f or 'back_to_back' in f],
        'Style Metrics': [f for f in feature_cols if 'pace' in f or '3pt' in f or 'eff' in f],
        'Matchup Features': [f for f in feature_cols if 'differential' in f or 'advantage' in f or 'vs' in f],
        'Head-to-Head': [f for f in feature_cols if 'h2h' in f],
        'Other': [f for f in feature_cols if not any(x in f for x in
                                                     ['_L', 'rest', 'back_to_back', 'pace', '3pt', 'eff',
                                                      'differential', 'advantage', 'vs', 'h2h'])]
    }

    for category, features in categories.items():
        if features:
            print(f"\n{category} ({len(features)}):")
            for f in features[:5]:  # Show first 5
                print(f"  - {f}")
            if len(features) > 5:
                print(f"  ... and {len(features) - 5} more")

    # Save processed data
    output_dir = Path("data/features")
    output_dir.mkdir(exist_ok=True, parents=True)

    train_features.to_csv(output_dir / "train_features.csv", index=False)
    test_features.to_csv(output_dir / "test_features.csv", index=False)

    print(f"\n✓ Features saved to {output_dir}/")