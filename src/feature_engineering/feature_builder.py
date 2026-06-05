"""
Feature Engineering for NBA Score Prediction
=============================================

Creates statistical and matchup-aware features for predicting game scores.
Focus on capturing team strength and style mismatches.
"""

import logging
import sqlite3
from pathlib import Path

import pandas as pd

from src.utils.config_loader import load_config

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

    def __init__(self, rolling_windows: list[int], h2h_margin_window: int = 3, h2h_win_rate_window: int = 5):
        self.rolling_windows = sorted(rolling_windows)
        self.h2h_margin_window = h2h_margin_window
        self.h2h_win_rate_window = h2h_win_rate_window

    def create_all_features(self, games_df: pd.DataFrame) -> pd.DataFrame:
        """
        Create complete feature set for all games.

        Args:
            games_df: DataFrame with game results

        Returns:
            DataFrame with all features
        """
        df = games_df.copy()
        df = df.sort_values('GAME_DATE').reset_index(drop=True)

        df = self._add_basic_features(df)
        df = self._add_rolling_features(df)
        df = self._add_rest_features(df)
        df = self._add_style_features(df)
        df = self._add_home_advantage_features(df)
        df = self._add_matchup_features(df)
        df = self._add_h2h_features(df)
        df = self._add_injury_features(df)

        initial_rows = len(df)
        df = df.dropna(subset=self._get_feature_columns(df))
        dropped = initial_rows - len(df)

        logger.info(f"Features built: {len(self._get_feature_columns(df))} cols, {len(df):,} games ({dropped} dropped)")

        return df

    def _add_basic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add basic game-level features"""
        # Season progress (0 to 1)
        df['season_progress'] = df.groupby('SEASON_ID').cumcount() / df.groupby('SEASON_ID')['SEASON_ID'].transform('count')

        return df

    def _add_rolling_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add rolling average features for recent performance."""
        for team_col, pts_col, prefix in [
            ('HOME_TEAM_ID', 'PTS_home', 'home_team'),
            ('AWAY_TEAM_ID', 'PTS_away', 'away_team')
        ]:
            _win = f'_win_{prefix}'
            _diff = f'_diff_{prefix}'
            df[_win] = (df['POINT_DIFF'] > 0) if prefix == 'home_team' else (df['POINT_DIFF'] < 0)
            df[_diff] = df['POINT_DIFF'] if prefix == 'home_team' else -df['POINT_DIFF']

            for window in self.rolling_windows:
                df[f'{prefix}_pts_avg_L{window}'] = df.groupby(team_col)[pts_col].transform(
                    lambda x, w=window: x.shift(1).rolling(w, min_periods=1).mean()
                )
                df[f'{prefix}_win_pct_L{window}'] = df.groupby(team_col)[_win].transform(
                    lambda x, w=window: x.shift(1).rolling(w, min_periods=1).mean()
                )
                df[f'{prefix}_diff_avg_L{window}'] = df.groupby(team_col)[_diff].transform(
                    lambda x, w=window: x.shift(1).rolling(w, min_periods=1).mean()
                )
                for stat in ['FG_PCT', 'FG3_PCT', 'FT_PCT']:
                    stat_col = f'{stat}_{prefix.split("_")[0]}'
                    if stat_col in df.columns:
                        df[f'{prefix}_{stat.lower()}_L{window}'] = df.groupby(team_col)[stat_col].transform(
                            lambda x, w=window: x.shift(1).rolling(w, min_periods=1).mean()
                        )

            df.drop(columns=[_win, _diff], inplace=True)

        return df

    def _add_rest_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add rest and schedule-related features.

        Captures fatigue and travel effects.
        """
        for team_col, prefix in [
            ('HOME_TEAM_ID', 'home_team'),
            ('AWAY_TEAM_ID', 'away_team')
        ]:
            # Days of rest since last game
            df[f'{prefix}_rest_days'] = df.groupby(team_col)['GAME_DATE'].diff().dt.days.fillna(3)

            # Back-to-back games
            df[f'{prefix}_back_to_back'] = (df[f'{prefix}_rest_days'] == 1).astype(int)

            # 3 games in 4 nights (fatigue indicator)
            df[f'{prefix}_games_in_4_nights'] = df.groupby(team_col, group_keys=False).apply(
                lambda x: (x['GAME_DATE'].diff().dt.days.rolling(3, min_periods=1).sum() <= 4).astype(int)
            )

        return df

    def _add_style_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add team style metrics.

        Captures how teams play (pace, shooting style, etc.)
        These are crucial for matchup analysis.
        """
        for team_col, prefix in [
            ('HOME_TEAM_ID', 'home_team'),
            ('AWAY_TEAM_ID', 'away_team')
        ]:
            pts_col = 'PTS_home' if prefix == 'home_team' else 'PTS_away'
            opp_pts_col = 'PTS_away' if prefix == 'home_team' else 'PTS_home'
            fg3_col = f'FG3_PCT_{prefix.split("_")[0]}'

            for window in self.rolling_windows:
                grouped = df.groupby(team_col)

                if fg3_col in df.columns:
                    df[f'{prefix}_3pt_rate_L{window}'] = grouped[fg3_col].transform(
                        lambda x, w=window: x.shift(1).rolling(w, min_periods=1).mean()
                    )

                df[f'{prefix}_off_eff_L{window}'] = grouped[pts_col].transform(
                    lambda x, w=window: x.shift(1).rolling(w, min_periods=1).mean()
                )
                df[f'{prefix}_def_eff_L{window}'] = grouped[opp_pts_col].transform(
                    lambda x, w=window: x.shift(1).rolling(w, min_periods=1).mean()
                )

        return df

    def _add_home_advantage_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add per-team venue delta: rolling avg pts at home minus rolling avg pts on the road.

        The role-split rolling features already capture venue-specific form (home_pts_avg tracks
        a team when they're home, away_pts_avg when they're away). This feature makes the
        home/road gap explicit, capturing teams that are particularly strong or weak at home.
        """
        for window in self.rolling_windows:
            df = self._compute_venue_delta(df, window)
        return df

    def _compute_venue_delta(self, df: pd.DataFrame, window: int) -> pd.DataFrame:
        home_lookup = (
            df[['GAME_DATE', 'HOME_TEAM_ID']]
            .assign(home_roll=df.groupby('HOME_TEAM_ID')['PTS_home'].transform(
                lambda x, w=window: x.shift(1).rolling(w, min_periods=1).mean()
            ))
            .rename(columns={'HOME_TEAM_ID': 'team_id'})
            .sort_values('GAME_DATE')
        )
        away_lookup = (
            df[['GAME_DATE', 'AWAY_TEAM_ID']]
            .assign(away_roll=df.groupby('AWAY_TEAM_ID')['PTS_away'].transform(
                lambda x, w=window: x.shift(1).rolling(w, min_periods=1).mean()
            ))
            .rename(columns={'AWAY_TEAM_ID': 'team_id'})
            .sort_values('GAME_DATE')
        )

        home_query = df[['GAME_DATE', 'HOME_TEAM_ID']].rename(columns={'HOME_TEAM_ID': 'team_id'})
        away_query = df[['GAME_DATE', 'AWAY_TEAM_ID']].rename(columns={'AWAY_TEAM_ID': 'team_id'})

        home_team_away_roll = pd.merge_asof(
            home_query, away_lookup, on='GAME_DATE', by='team_id', direction='backward'
        )['away_roll']
        away_team_home_roll = pd.merge_asof(
            away_query, home_lookup, on='GAME_DATE', by='team_id', direction='backward'
        )['home_roll']

        home_roll = df.groupby('HOME_TEAM_ID')['PTS_home'].transform(
            lambda x, w=window: x.shift(1).rolling(w, min_periods=1).mean()
        )
        away_roll = df.groupby('AWAY_TEAM_ID')['PTS_away'].transform(
            lambda x, w=window: x.shift(1).rolling(w, min_periods=1).mean()
        )

        df[f'home_team_venue_delta_L{window}'] = home_roll.values - home_team_away_roll.values
        df[f'away_team_venue_delta_L{window}'] = away_team_home_roll.values - away_roll.values

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
        for window in self.rolling_windows:
            if f'home_team_off_eff_L{window}' in df.columns:
                df[f'home_off_vs_away_def_L{window}'] = df[f'home_team_off_eff_L{window}'] - df[f'away_team_def_eff_L{window}']
                df[f'away_off_vs_home_def_L{window}'] = df[f'away_team_off_eff_L{window}'] - df[f'home_team_def_eff_L{window}']

            if f'home_team_3pt_rate_L{window}' in df.columns:
                df[f'home_3pt_advantage_L{window}'] = df[f'home_team_3pt_rate_L{window}'] - df[f'away_team_3pt_rate_L{window}']

            if f'home_team_win_pct_L{window}' in df.columns:
                df[f'form_differential_L{window}'] = df[f'home_team_win_pct_L{window}'] - df[f'away_team_win_pct_L{window}']

            if f'home_team_diff_avg_L{window}' in df.columns:
                df[f'strength_differential_L{window}'] = df[f'home_team_diff_avg_L{window}'] - df[f'away_team_diff_avg_L{window}']

        return df

    def _add_h2h_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add head-to-head historical features from the current home team's perspective."""
        mw = self.h2h_margin_window
        ww = self.h2h_win_rate_window

        df['matchup_key'] = df.apply(
            lambda row: '_'.join(sorted([str(row['HOME_TEAM_ID']), str(row['AWAY_TEAM_ID'])])),
            axis=1
        )

        # Canonical margin: always from the lower-ID team's perspective so sign is consistent
        # across all historical rows in the same matchup group.
        df['_canonical_team'] = df[['HOME_TEAM_ID', 'AWAY_TEAM_ID']].min(axis=1)
        df['_canonical_margin'] = df.apply(
            lambda r: r['POINT_DIFF'] if r['HOME_TEAM_ID'] == r['_canonical_team'] else -r['POINT_DIFF'],
            axis=1
        )

        df['_h2h_margin_canon'] = df.groupby('matchup_key', group_keys=False).apply(
            lambda x: x['_canonical_margin'].shift(1).rolling(mw, min_periods=1).mean()
        )
        df['_h2h_win_canon'] = df.groupby('matchup_key', group_keys=False).apply(
            lambda x: (x['_canonical_margin'] > 0).shift(1).rolling(ww, min_periods=1).mean()
        )

        # Flip sign/rate back to current home team's perspective
        is_canon_home = df['HOME_TEAM_ID'] == df['_canonical_team']
        df[f'h2h_home_margin_L{mw}'] = df['_h2h_margin_canon'].where(is_canon_home, -df['_h2h_margin_canon'])
        df[f'h2h_home_win_rate_L{ww}'] = df['_h2h_win_canon'].where(is_canon_home, 1 - df['_h2h_win_canon'])

        df.drop(columns=['_canonical_team', '_canonical_margin', '_h2h_margin_canon', '_h2h_win_canon'], inplace=True)

        return df

    def _add_injury_features(self, df: pd.DataFrame) -> pd.DataFrame:
        cfg = load_config()
        if not cfg.injury_features or not cfg.injury_features.enabled:
            return df

        db_path = cfg.injury_features.db_path
        if not Path(db_path).exists():
            logger.warning(f"Injury DB not found at {db_path} — skipping injury features")
            return df

        scorer = cfg.injury_features.scorer
        with sqlite3.connect(db_path) as conn:
            injury_df = pd.read_sql_query(
                "SELECT game_date, team_id, impact_score, n_out, n_questionable "
                "FROM injury_features WHERE scorer = ?",
                conn,
                params=(scorer,),
            )

        injury_df["game_date"] = pd.to_datetime(injury_df["game_date"]).dt.normalize()
        game_dates = pd.to_datetime(df["GAME_DATE"]).dt.normalize()

        for team_col, prefix in [("HOME_TEAM_ID", "home_team"), ("AWAY_TEAM_ID", "away_team")]:
            lookup = pd.DataFrame({
                "game_date": game_dates.values,
                "team_id": df[team_col].values,
            })
            merged = lookup.merge(injury_df, on=["game_date", "team_id"], how="left")
            df[f"{prefix}_injury_impact"] = merged["impact_score"].fillna(0).values
            df[f"{prefix}_n_out"] = merged["n_out"].fillna(0).astype(int).values
            df[f"{prefix}_n_questionable"] = merged["n_questionable"].fillna(0).astype(int).values

        dates_with_coverage = set(injury_df["game_date"])
        df["has_injury_data"] = game_dates.isin(dates_with_coverage).astype(int)

        return df

    def _get_feature_columns(self, df: pd.DataFrame) -> list[str]:
        exclude = load_config().features.exclude
        return [col for col in df.columns if col not in exclude]

    def get_feature_names(self, df: pd.DataFrame) -> list[str]:
        """Public method to get feature column names"""
        return self._get_feature_columns(df)


# Example usage
if __name__ == "__main__":
    from src.data_processing.data_loader import load_training_data

    cfg = load_config()
    train_df, val_df, test_df = load_training_data(
        db_path=cfg.data_paths.raw_db,
        train_start_date=cfg.datasets_loading.train_start_date,
        train_end_date=cfg.datasets_loading.train_end_date,
        val_start_date=cfg.datasets_loading.validation_start_date,
        val_end_date=cfg.datasets_loading.validation_end_date,
        test_start_date=cfg.datasets_loading.test_start_date,
        test_end_date=cfg.datasets_loading.test_end_date,
        data_start_date=cfg.datasets_loading.data_start_date,
    )

    fb = FeatureBuilder(rolling_windows=cfg.features.rolling_windows)

    train_features = fb.create_all_features(train_df)
    test_features = fb.create_all_features(test_df)

    cols = fb.get_feature_names(train_features)
    print(f"Features: {len(cols)}")
    for c in cols:
        print(f"  {c}")

    output_dir = Path("data/features")
    output_dir.mkdir(exist_ok=True, parents=True)
    train_features.to_csv(output_dir / "train_features.csv", index=False)
    test_features.to_csv(output_dir / "test_features.csv", index=False)
    print(f"Saved to {output_dir}/")