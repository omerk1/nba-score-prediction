"""
Feature Engineering for NBA Score Prediction
=============================================

Creates statistical and matchup-aware features for predicting game scores.
Focus on capturing team strength and style mismatches.
"""

import logging
import math
import sqlite3
from pathlib import Path

import pandas as pd

from src.utils.config_loader import load_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# (lat, lon, utc_offset_hours) for each NBA team's home arena.
# UTC offsets are standard (winter) time — the relative difference between
# teams stays constant regardless of DST since all US zones shift together
# (except Phoenix, which never observes DST and is fixed at UTC-7).
_TEAM_LOCATIONS: dict[int, tuple[float, float, int]] = {
    1610612737: (33.749,  -84.388,  -5),  # ATL
    1610612738: (42.360,  -71.059,  -5),  # BOS
    1610612739: (41.499,  -81.694,  -5),  # CLE
    1610612740: (29.951,  -90.072,  -6),  # NOP
    1610612741: (41.878,  -87.630,  -6),  # CHI
    1610612742: (32.777,  -96.797,  -6),  # DAL
    1610612743: (39.739, -104.990,  -7),  # DEN
    1610612744: (37.768, -122.388,  -8),  # GSW
    1610612745: (29.760,  -95.370,  -6),  # HOU
    1610612746: (34.043, -118.267,  -8),  # LAC
    1610612747: (34.043, -118.267,  -8),  # LAL
    1610612748: (25.762,  -80.192,  -5),  # MIA
    1610612749: (43.039,  -87.907,  -6),  # MIL
    1610612750: (44.978,  -93.265,  -6),  # MIN
    1610612751: (40.683,  -73.972,  -5),  # BKN
    1610612752: (40.751,  -73.993,  -5),  # NYK
    1610612753: (28.538,  -81.379,  -5),  # ORL
    1610612754: (39.768,  -86.158,  -5),  # IND
    1610612755: (39.953,  -75.165,  -5),  # PHI
    1610612756: (33.448, -112.074,  -7),  # PHX (no DST, fixed UTC-7)
    1610612757: (45.523, -122.677,  -8),  # POR
    1610612758: (38.582, -121.494,  -8),  # SAC
    1610612759: (29.424,  -98.494,  -6),  # SAS
    1610612760: (35.468,  -97.516,  -6),  # OKC
    1610612761: (43.653,  -79.383,  -5),  # TOR
    1610612762: (40.761, -111.891,  -7),  # UTA
    1610612763: (35.150,  -90.049,  -6),  # MEM
    1610612764: (38.898,  -77.037,  -5),  # WAS
    1610612765: (42.331,  -83.046,  -5),  # DET
    1610612766: (35.227,  -80.843,  -5),  # CHA
}


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
        df = games_df.copy()
        df = df.sort_values('GAME_DATE').reset_index(drop=True)

        df = self._add_basic_features(df)
        df = self._add_rolling_features(df)
        df = self._add_rest_features(df)
        df = self._add_style_features(df)
        df = self._add_opponent_quality_features(df)
        df = self._add_home_advantage_features(df)
        df = self._add_matchup_features(df)
        df = self._add_h2h_features(df)
        df = self._add_travel_features(df)
        df = self._add_elo_features(df)
        df = self._add_injury_features(df)

        feature_cols = self._get_feature_columns(df)
        nan_games = df[feature_cols].isna().any(axis=1).sum()

        logger.info(f"Features built: {len(feature_cols)} cols, {len(df):,} games ({nan_games} with NaN — kept, CatBoost handles natively)")

        return df

    def _add_basic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        new_cols = {
            'season_progress': df.groupby('SEASON_ID').cumcount() / df.groupby('SEASON_ID')['SEASON_ID'].transform('count')
        }
        return pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)

    def _add_rolling_features(self, df: pd.DataFrame) -> pd.DataFrame:
        new_cols = {}
        for team_col, pts_col, prefix in [
            ('HOME_TEAM_ID', 'PTS_home', 'home_team'),
            ('AWAY_TEAM_ID', 'PTS_away', 'away_team'),
        ]:
            win_series = (df['POINT_DIFF'] > 0) if prefix == 'home_team' else (df['POINT_DIFF'] < 0)
            diff_series = df['POINT_DIFF'] if prefix == 'home_team' else -df['POINT_DIFF']

            # Temp columns needed for groupby.transform
            df[f'_win_{prefix}'] = win_series
            df[f'_diff_{prefix}'] = diff_series

            for window in self.rolling_windows:
                new_cols[f'{prefix}_pts_avg_L{window}'] = df.groupby(team_col)[pts_col].transform(
                    lambda x, w=window: x.shift(1).rolling(w, min_periods=1).mean()
                )
                new_cols[f'{prefix}_win_pct_L{window}'] = df.groupby(team_col)[f'_win_{prefix}'].transform(
                    lambda x, w=window: x.shift(1).rolling(w, min_periods=1).mean()
                )
                new_cols[f'{prefix}_diff_avg_L{window}'] = df.groupby(team_col)[f'_diff_{prefix}'].transform(
                    lambda x, w=window: x.shift(1).rolling(w, min_periods=1).mean()
                )
                for stat in ['FG_PCT', 'FG3_PCT', 'FT_PCT']:
                    stat_col = f'{stat}_{prefix.split("_")[0]}'
                    if stat_col in df.columns:
                        new_cols[f'{prefix}_{stat.lower()}_L{window}'] = df.groupby(team_col)[stat_col].transform(
                            lambda x, w=window: x.shift(1).rolling(w, min_periods=1).mean()
                        )

            df.drop(columns=[f'_win_{prefix}', f'_diff_{prefix}'], inplace=True)

        return pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)

    def _add_rest_features(self, df: pd.DataFrame) -> pd.DataFrame:
        new_cols = {}
        for team_col, prefix in [
            ('HOME_TEAM_ID', 'home_team'),
            ('AWAY_TEAM_ID', 'away_team'),
        ]:
            rest_days = df.groupby(team_col)['GAME_DATE'].diff().dt.days.fillna(3)
            new_cols[f'{prefix}_rest_days'] = rest_days
            new_cols[f'{prefix}_back_to_back'] = (rest_days == 1).astype(int)
            new_cols[f'{prefix}_games_in_4_nights'] = df.groupby(team_col, group_keys=False).apply(
                lambda x: (x['GAME_DATE'].diff().dt.days.rolling(3, min_periods=1).sum() <= 4).astype(int)
            )
        return pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)

    def _add_style_features(self, df: pd.DataFrame) -> pd.DataFrame:
        new_cols = {}
        for team_col, prefix in [
            ('HOME_TEAM_ID', 'home_team'),
            ('AWAY_TEAM_ID', 'away_team'),
        ]:
            pts_col = 'PTS_home' if prefix == 'home_team' else 'PTS_away'
            opp_pts_col = 'PTS_away' if prefix == 'home_team' else 'PTS_home'
            fg3_col = f'FG3_PCT_{prefix.split("_")[0]}'

            for window in self.rolling_windows:
                grouped = df.groupby(team_col)
                if fg3_col in df.columns:
                    new_cols[f'{prefix}_3pt_rate_L{window}'] = grouped[fg3_col].transform(
                        lambda x, w=window: x.shift(1).rolling(w, min_periods=1).mean()
                    )
                new_cols[f'{prefix}_off_eff_L{window}'] = grouped[pts_col].transform(
                    lambda x, w=window: x.shift(1).rolling(w, min_periods=1).mean()
                )
                new_cols[f'{prefix}_def_eff_L{window}'] = grouped[opp_pts_col].transform(
                    lambda x, w=window: x.shift(1).rolling(w, min_periods=1).mean()
                )
        return pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)

    def _add_opponent_quality_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        For each team, add the rolling average quality of opponents faced.

        Uses already-computed off_eff/def_eff features (which are shift(1)-based,
        so no leakage): for each of a team's last N games, record the opponent's
        efficiency and average those values.
        """
        new_cols = {}
        for window in self.rolling_windows:
            if f'home_team_def_eff_L{window}' not in df.columns:
                continue

            for opp_stat, home_col, away_col in [
                ('opp_def_quality', f'away_team_def_eff_L{window}', f'home_team_def_eff_L{window}'),
                ('opp_off_quality', f'away_team_off_eff_L{window}', f'home_team_off_eff_L{window}'),
            ]:
                home_rows = pd.DataFrame({
                    'GAME_DATE': df['GAME_DATE'].values,
                    'team_id':   df['HOME_TEAM_ID'].values,
                    'opp_q':     df[home_col].values,
                })
                away_rows = pd.DataFrame({
                    'GAME_DATE': df['GAME_DATE'].values,
                    'team_id':   df['AWAY_TEAM_ID'].values,
                    'opp_q':     df[away_col].values,
                })
                long_df = (
                    pd.concat([home_rows, away_rows])
                    .sort_values('GAME_DATE')
                    .reset_index(drop=True)
                )
                long_df['rolling'] = long_df.groupby('team_id')['opp_q'].transform(
                    lambda x, w=window: x.shift(1).rolling(w, min_periods=1).mean()
                )

                for team_col, prefix in [('HOME_TEAM_ID', 'home_team'), ('AWAY_TEAM_ID', 'away_team')]:
                    query = df[['GAME_DATE', team_col]].rename(columns={team_col: 'team_id'})
                    merged = query.merge(
                        long_df[['GAME_DATE', 'team_id', 'rolling']],
                        on=['GAME_DATE', 'team_id'],
                        how='left',
                    )
                    new_cols[f'{prefix}_{opp_stat}_L{window}'] = merged['rolling'].values

        return pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)

    def _add_home_advantage_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add per-team venue delta: rolling avg pts at home minus rolling avg pts on the road.

        The role-split rolling features already capture venue-specific form (home_pts_avg tracks
        a team when they're home, away_pts_avg when they're away). This feature makes the
        home/road gap explicit, capturing teams that are particularly strong or weak at home.
        """
        new_cols = {}
        for window in self.rolling_windows:
            venue_cols = self._compute_venue_delta(df, window)
            new_cols.update(venue_cols)
        return pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)

    def _compute_venue_delta(self, df: pd.DataFrame, window: int) -> dict:
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

        return {
            f'home_team_venue_delta_L{window}': home_roll.values - home_team_away_roll.values,
            f'away_team_venue_delta_L{window}': away_team_home_roll.values - away_roll.values,
        }

    def _add_matchup_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add matchup-specific features.

        CRITICAL: This captures style advantages and mismatches.
        Examples:
        - Good shooting team vs bad perimeter defense
        - Offensive powerhouse vs defensive team
        """
        new_cols = {}
        for window in self.rolling_windows:
            if f'home_team_off_eff_L{window}' in df.columns:
                new_cols[f'home_off_vs_away_def_L{window}'] = df[f'home_team_off_eff_L{window}'] - df[f'away_team_def_eff_L{window}']
                new_cols[f'away_off_vs_home_def_L{window}'] = df[f'away_team_off_eff_L{window}'] - df[f'home_team_def_eff_L{window}']

            if f'home_team_3pt_rate_L{window}' in df.columns:
                new_cols[f'home_3pt_advantage_L{window}'] = df[f'home_team_3pt_rate_L{window}'] - df[f'away_team_3pt_rate_L{window}']

            if f'home_team_win_pct_L{window}' in df.columns:
                new_cols[f'form_differential_L{window}'] = df[f'home_team_win_pct_L{window}'] - df[f'away_team_win_pct_L{window}']

            if f'home_team_diff_avg_L{window}' in df.columns:
                new_cols[f'strength_differential_L{window}'] = df[f'home_team_diff_avg_L{window}'] - df[f'away_team_diff_avg_L{window}']

        return pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)

    def _add_h2h_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add head-to-head historical features from the current home team's perspective."""
        mw = self.h2h_margin_window
        ww = self.h2h_win_rate_window

        # Temp columns needed for groupby operations
        df['matchup_key'] = df.apply(
            lambda row: '_'.join(sorted([str(row['HOME_TEAM_ID']), str(row['AWAY_TEAM_ID'])])),
            axis=1
        )
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

        is_canon_home = df['HOME_TEAM_ID'] == df['_canonical_team']
        new_cols = {
            f'h2h_home_margin_L{mw}':   df['_h2h_margin_canon'].where(is_canon_home, -df['_h2h_margin_canon']),
            f'h2h_home_win_rate_L{ww}': df['_h2h_win_canon'].where(is_canon_home, 1 - df['_h2h_win_canon']),
        }

        df.drop(columns=['_canonical_team', '_canonical_margin', '_h2h_margin_canon', '_h2h_win_canon'], inplace=True)

        return pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)

    @staticmethod
    def _haversine_miles(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        R = 3958.8  # Earth radius in miles
        dlat = math.radians(lat2 - lat1)
        dlon = math.radians(lon2 - lon1)
        a = math.sin(dlat / 2) ** 2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2
        return R * 2 * math.asin(math.sqrt(a))

    def _add_travel_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add travel distance and timezone shift for each team since their last game.

        Game location = home team's city. For each team-game, we look up their
        previous game's city, compute Haversine distance, and the timezone delta.
        Positive tz_shift = traveling east (harder on the body clock).
        """
        loc = _TEAM_LOCATIONS

        # Long format: one row per (game, team) recording where the game is played
        home_rows = pd.DataFrame({
            'GAME_DATE': df['GAME_DATE'].values,
            'team_id':   df['HOME_TEAM_ID'].values,
            'city_team': df['HOME_TEAM_ID'].values,  # game is in home team's city
        })
        away_rows = pd.DataFrame({
            'GAME_DATE': df['GAME_DATE'].values,
            'team_id':   df['AWAY_TEAM_ID'].values,
            'city_team': df['HOME_TEAM_ID'].values,  # away team travels to home team's city
        })
        long_df = (
            pd.concat([home_rows, away_rows])
            .sort_values('GAME_DATE')
            .reset_index(drop=True)
        )

        long_df['prev_city_team'] = long_df.groupby('team_id')['city_team'].shift(1)
        # Default previous city to team's own home (no travel) when no prior game exists
        long_df['prev_city_team'] = long_df['prev_city_team'].fillna(long_df['team_id'])

        def travel_miles(row):
            curr = loc.get(int(row['city_team']))
            prev = loc.get(int(row['prev_city_team']))
            if curr is None or prev is None:
                return 0.0
            return self._haversine_miles(prev[0], prev[1], curr[0], curr[1])

        def tz_shift(row):
            curr = loc.get(int(row['city_team']))
            prev = loc.get(int(row['prev_city_team']))
            if curr is None or prev is None:
                return 0
            return curr[2] - prev[2]  # positive = traveled east

        long_df['travel_miles'] = long_df.apply(travel_miles, axis=1)
        long_df['tz_shift']     = long_df.apply(tz_shift,     axis=1)

        # Rolling travel miles over last 7 and 14 days (day-windows capture road-trip
        # fatigue better than game-count windows since schedule density varies).
        long_df['GAME_DATE'] = pd.to_datetime(long_df['GAME_DATE'])
        long_df = long_df.sort_values(['team_id', 'GAME_DATE'])
        for days in [7, 14]:
            col = f'travel_miles_{days}d'
            long_df[col] = (
                long_df.groupby('team_id', group_keys=False)
                .apply(lambda g, d=days: (
                    g.set_index('GAME_DATE')['travel_miles']
                    .rolling(f'{d}D', closed='both')
                    .sum()
                    .set_axis(g.index)
                ))
            )

        rolling_cols = ['travel_miles_7d', 'travel_miles_14d']
        new_cols = {}
        for team_col, prefix in [('HOME_TEAM_ID', 'home_team'), ('AWAY_TEAM_ID', 'away_team')]:
            query = df[['GAME_DATE', team_col]].rename(columns={team_col: 'team_id'})
            query['GAME_DATE'] = pd.to_datetime(query['GAME_DATE'])
            merged = query.merge(
                long_df[['GAME_DATE', 'team_id', 'travel_miles', 'tz_shift'] + rolling_cols],
                on=['GAME_DATE', 'team_id'],
                how='left',
            )
            new_cols[f'{prefix}_travel_miles']      = merged['travel_miles'].fillna(0).values
            new_cols[f'{prefix}_tz_shift']           = merged['tz_shift'].fillna(0).values
            new_cols[f'{prefix}_travel_miles_7d']   = merged['travel_miles_7d'].fillna(0).values
            new_cols[f'{prefix}_travel_miles_14d']  = merged['travel_miles_14d'].fillna(0).values

        return pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)

    def _add_elo_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add pre-game Elo ratings (home_team_elo, away_team_elo, elo_diff).

        Elo is inherently sequential — a team's rating depends on every prior
        result, not just a recent window. So ratings are computed once over
        the full chronological game history (not just the rows in `df`),
        then merged onto `df` by GAME_ID. This ensures val/test games carry
        forward ratings accumulated during train, rather than restarting at
        initial_rating each split.
        """
        cfg = load_config()
        if not cfg.elo_features or not cfg.elo_features.enabled:
            return df

        from src.data_processing.data_loader import NBADataLoader
        from src.feature_engineering.elo import compute_elo_ratings

        loader = NBADataLoader(db_path=cfg.data_paths.raw_db)
        try:
            all_games = loader.load_games(
                start_date=cfg.datasets_loading.data_start_date,
                end_date=cfg.datasets_loading.test_end_date,
                allowed_season_types=cfg.datasets_loading.context_season_types or cfg.datasets_loading.allowed_season_types,
            )
        finally:
            loader.close()

        elo_cfg = cfg.elo_features
        elo_df = compute_elo_ratings(
            all_games,
            initial_rating=elo_cfg.initial_rating,
            k_factor=elo_cfg.k_factor,
            home_advantage=elo_cfg.home_advantage,
            mov_multiplier=elo_cfg.mov_multiplier,
            season_regression=elo_cfg.season_regression,
        )

        merged = df[['GAME_ID']].merge(elo_df, on='GAME_ID', how='left')

        new_cols = {
            'home_team_elo': merged['home_team_elo'].values,
            'away_team_elo': merged['away_team_elo'].values,
            'elo_diff': merged['home_team_elo'].values + elo_cfg.home_advantage - merged['away_team_elo'].values,
        }
        return pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)

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

        new_cols = {}
        for team_col, prefix in [("HOME_TEAM_ID", "home_team"), ("AWAY_TEAM_ID", "away_team")]:
            lookup = pd.DataFrame({
                "game_date": game_dates.values,
                "team_id": df[team_col].values,
            })
            merged = lookup.merge(injury_df, on=["game_date", "team_id"], how="left")
            new_cols[f"{prefix}_injury_impact"] = merged["impact_score"].fillna(0).values
            new_cols[f"{prefix}_n_out"] = merged["n_out"].fillna(0).astype(int).values
            new_cols[f"{prefix}_n_questionable"] = merged["n_questionable"].fillna(0).astype(int).values

        dates_with_coverage = set(injury_df["game_date"])
        new_cols["has_injury_data"] = game_dates.isin(dates_with_coverage).astype(int)

        return pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)

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
