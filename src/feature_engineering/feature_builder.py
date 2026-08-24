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
from typing import Optional

import numpy as np
import pandas as pd

from src.matchups.config import CACHE_DB, NBA_API_DB
from src.utils.config_loader import InjuryMissingValueStrategy, load_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# (lat, lon, utc_offset_hours) for each NBA team's home arena.
# UTC offsets are standard (winter) time — the relative difference between
# teams stays constant regardless of DST since all US zones shift together
# (except Phoenix, which never observes DST and is fixed at UTC-7).
_TEAM_LOCATIONS: dict[int, tuple[float, float, int]] = {
    1610612737: (33.749, -84.388, -5),  # ATL
    1610612738: (42.360, -71.059, -5),  # BOS
    1610612739: (41.499, -81.694, -5),  # CLE
    1610612740: (29.951, -90.072, -6),  # NOP
    1610612741: (41.878, -87.630, -6),  # CHI
    1610612742: (32.777, -96.797, -6),  # DAL
    1610612743: (39.739, -104.990, -7),  # DEN
    1610612744: (37.768, -122.388, -8),  # GSW
    1610612745: (29.760, -95.370, -6),  # HOU
    1610612746: (34.043, -118.267, -8),  # LAC
    1610612747: (34.043, -118.267, -8),  # LAL
    1610612748: (25.762, -80.192, -5),  # MIA
    1610612749: (43.039, -87.907, -6),  # MIL
    1610612750: (44.978, -93.265, -6),  # MIN
    1610612751: (40.683, -73.972, -5),  # BKN
    1610612752: (40.751, -73.993, -5),  # NYK
    1610612753: (28.538, -81.379, -5),  # ORL
    1610612754: (39.768, -86.158, -5),  # IND
    1610612755: (39.953, -75.165, -5),  # PHI
    1610612756: (33.448, -112.074, -7),  # PHX (no DST, fixed UTC-7)
    1610612757: (45.523, -122.677, -8),  # POR
    1610612758: (38.582, -121.494, -8),  # SAC
    1610612759: (29.424, -98.494, -6),  # SAS
    1610612760: (35.468, -97.516, -6),  # OKC
    1610612761: (43.653, -79.383, -5),  # TOR
    1610612762: (40.761, -111.891, -7),  # UTA
    1610612763: (35.150, -90.049, -6),  # MEM
    1610612764: (38.898, -77.037, -5),  # WAS
    1610612765: (42.331, -83.046, -5),  # DET
    1610612766: (35.227, -80.843, -5),  # CHA
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
        # Fit-once-on-train, reuse-on-val/test cache for _fit_elo_margin_scale
        # (see _add_season_motivation_features) -- a real "fit statistic" (a
        # least-squares regression coefficient), unlike Elo ratings/standings
        # snapshots, which are safe to recompute fresh with more history on
        # every call. Set on this instance's FIRST create_all_features() call
        # (assumed train, matching every current caller's own call order) and
        # reused unchanged on later calls, so it's never refit using val/test
        # outcomes. A fresh FeatureBuilder() (one per run_split() call) means
        # this never leaks across folds either.
        self._fitted_elo_margin_scale = None

    def create_all_features(
        self, games_df: pd.DataFrame, context_end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """
        context_end_date: upper bound for any feature that needs more history
        than `games_df` alone to compute correctly (Elo ratings, season_motivation
        standings/schedule) -- see `_add_elo_features`/`_add_season_motivation_features`.
        Defaults to `games_df`'s own max GAME_DATE (never reaches beyond the data
        this specific call was given) when not provided; a CV/multi-split caller
        should pass this split's own true end date explicitly (see
        src/evaluation/cv_harness.run_split) so a train-time call can't reach into
        val/test-period games.
        """
        df = games_df.copy()
        df = df.sort_values("GAME_DATE").reset_index(drop=True)
        context_end_date = context_end_date or df["GAME_DATE"].max().strftime("%Y-%m-%d")

        df = self._add_basic_features(df)
        df = self._add_rolling_features(df)
        df = self._add_rest_features(df)
        df = self._add_style_features(df)
        df = self._add_opponent_quality_features(df)
        df = self._add_home_advantage_features(df)
        df = self._add_matchup_features(df)
        df = self._add_h2h_features(df)
        df = self._add_travel_features(df)
        df = self._add_elo_features(df, context_end_date)
        df = self._add_injury_features(df)
        df = self._add_style_matchup_features(df)
        df = self._add_style_fingerprint_features(df)
        df = self._add_on_off_splits_features(df)
        df = self._add_season_motivation_features(df, context_end_date)

        feature_cols = self._get_feature_columns(df)
        nan_games = df[feature_cols].isna().any(axis=1).sum()

        logger.info(
            f"Features built: {len(feature_cols)} cols, {len(df):,} games ({nan_games} with NaN — kept, CatBoost handles natively)"
        )

        return df

    def _add_basic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        new_cols = {
            "season_progress": df.groupby("SEASON_ID").cumcount()
            / df.groupby("SEASON_ID")["SEASON_ID"].transform("count")
        }
        return pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)

    def _add_rolling_features(self, df: pd.DataFrame) -> pd.DataFrame:
        new_cols = {}
        temp_cols = []
        for team_col, pts_col, prefix in [
            ("HOME_TEAM_ID", "PTS_home", "home_team"),
            ("AWAY_TEAM_ID", "PTS_away", "away_team"),
        ]:
            win_series = (df["POINT_DIFF"] > 0) if prefix == "home_team" else (df["POINT_DIFF"] < 0)
            diff_series = df["POINT_DIFF"] if prefix == "home_team" else -df["POINT_DIFF"]

            # Temp columns needed for groupby.transform
            df[f"_win_{prefix}"] = win_series
            df[f"_diff_{prefix}"] = diff_series
            temp_cols += [f"_win_{prefix}", f"_diff_{prefix}"]

            for window in self.rolling_windows:
                # pts_avg omitted — off_eff in _add_style_features is identical
                new_cols[f"{prefix}_win_pct_L{window}"] = df.groupby(team_col)[f"_win_{prefix}"].transform(
                    lambda x, w=window: x.shift(1).rolling(w, min_periods=1).mean()
                )
                new_cols[f"{prefix}_diff_avg_L{window}"] = df.groupby(team_col)[f"_diff_{prefix}"].transform(
                    lambda x, w=window: x.shift(1).rolling(w, min_periods=1).mean()
                )
                # FG3_PCT omitted — fg3_pct in _add_style_features is identical.
                # Volume-weighted (sum of makes / sum of attempts over the window), NOT a mean
                # of per-game percentages — the latter would let a low-attempt outlier game
                # (e.g. 1-for-2) swing the rolling average as much as a normal-volume game.
                for stat, made_stat, att_stat in [("FG_PCT", "FGM", "FGA"), ("FT_PCT", "FTM", "FTA")]:
                    team_suffix = prefix.split("_")[0]
                    made_col = f"{made_stat}_{team_suffix}"
                    att_col = f"{att_stat}_{team_suffix}"
                    if made_col in df.columns and att_col in df.columns:
                        made_roll = df.groupby(team_col)[made_col].transform(
                            lambda x, w=window: x.shift(1).rolling(w, min_periods=1).sum()
                        )
                        att_roll = df.groupby(team_col)[att_col].transform(
                            lambda x, w=window: x.shift(1).rolling(w, min_periods=1).sum()
                        )
                        new_cols[f"{prefix}_{stat.lower()}_L{window}"] = made_roll / att_roll.replace(
                            0, np.nan
                        )

        # Venue-blind overall form: win pct / point-diff avg over a team's last N
        # games regardless of home/away role. The loop above is venue-scoped only
        # (grouped by HOME_TEAM_ID / AWAY_TEAM_ID separately), per the pipeline-audit
        # finding that no overall-form feature exists. Added alongside the venue-scoped
        # features, not replacing them, via the team-perspective long-frame pattern
        # already used in _add_opponent_quality_features.
        home_rows = pd.DataFrame(
            {
                "GAME_DATE": df["GAME_DATE"].values,
                "team_id": df["HOME_TEAM_ID"].values,
                "win": df["_win_home_team"].values,
                "diff": df["_diff_home_team"].values,
            }
        )
        away_rows = pd.DataFrame(
            {
                "GAME_DATE": df["GAME_DATE"].values,
                "team_id": df["AWAY_TEAM_ID"].values,
                "win": df["_win_away_team"].values,
                "diff": df["_diff_away_team"].values,
            }
        )
        long_df = pd.concat([home_rows, away_rows]).sort_values("GAME_DATE").reset_index(drop=True)

        overall_cols = []
        for window in self.rolling_windows:
            win_col = f"win_pct_overall_L{window}"
            diff_col = f"diff_avg_overall_L{window}"
            long_df[win_col] = long_df.groupby("team_id")["win"].transform(
                lambda x, w=window: x.shift(1).rolling(w, min_periods=1).mean()
            )
            long_df[diff_col] = long_df.groupby("team_id")["diff"].transform(
                lambda x, w=window: x.shift(1).rolling(w, min_periods=1).mean()
            )
            overall_cols += [win_col, diff_col]

        for team_col, prefix in [("HOME_TEAM_ID", "home_team"), ("AWAY_TEAM_ID", "away_team")]:
            query = df[["GAME_DATE", team_col]].rename(columns={team_col: "team_id"})
            merged = query.merge(
                long_df[["GAME_DATE", "team_id"] + overall_cols], on=["GAME_DATE", "team_id"], how="left"
            )
            for col in overall_cols:
                new_cols[f"{prefix}_{col}"] = merged[col].values

        df.drop(columns=temp_cols, inplace=True)

        return pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)

    def _add_rest_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Rest/schedule-density features, computed venue-blind: a team's true rest is
        time since its LAST game regardless of home/away role, not time since its
        last game in that specific role. Grouping directly on HOME_TEAM_ID/
        AWAY_TEAM_ID (the prior implementation) silently skips over any interleaved
        game the team played in the other role, understating true fatigue -- same
        bug class _add_rolling_features's venue-blind overall-form fix addressed.
        Uses the same team-perspective long-frame pattern as that fix and as
        _add_opponent_quality_features.

        `back_to_back` (rest_days == 1) is, by construction, exactly "this is the
        second game of a back-to-back" for that team -- a separate second_of_b2b
        column would be a pure duplicate once rest_days is computed correctly.
        """
        home_rows = pd.DataFrame({"GAME_DATE": df["GAME_DATE"].values, "team_id": df["HOME_TEAM_ID"].values})
        away_rows = pd.DataFrame({"GAME_DATE": df["GAME_DATE"].values, "team_id": df["AWAY_TEAM_ID"].values})
        long_df = (
            pd.concat([home_rows, away_rows]).sort_values(["team_id", "GAME_DATE"]).reset_index(drop=True)
        )

        long_df["rest_days"] = long_df.groupby("team_id")["GAME_DATE"].diff().dt.days.fillna(3)
        long_df["back_to_back"] = (long_df["rest_days"] == 1).astype(int)
        long_df["games_in_4_nights"] = long_df.groupby("team_id", group_keys=False).apply(
            lambda x: (x["GAME_DATE"].diff().dt.days.rolling(3, min_periods=1).sum() <= 4).astype(int)
        )

        new_cols = {}
        for team_col, prefix in [("HOME_TEAM_ID", "home_team"), ("AWAY_TEAM_ID", "away_team")]:
            query = df[["GAME_DATE", team_col]].rename(columns={team_col: "team_id"})
            merged = query.merge(
                long_df[["GAME_DATE", "team_id", "rest_days", "back_to_back", "games_in_4_nights"]],
                on=["GAME_DATE", "team_id"],
                how="left",
            )
            new_cols[f"{prefix}_rest_days"] = merged["rest_days"].values
            new_cols[f"{prefix}_back_to_back"] = merged["back_to_back"].values
            new_cols[f"{prefix}_games_in_4_nights"] = merged["games_in_4_nights"].values
        return pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)

    def _add_style_features(self, df: pd.DataFrame) -> pd.DataFrame:
        new_cols = {}
        for team_col, prefix in [
            ("HOME_TEAM_ID", "home_team"),
            ("AWAY_TEAM_ID", "away_team"),
        ]:
            pts_col = "PTS_home" if prefix == "home_team" else "PTS_away"
            opp_pts_col = "PTS_away" if prefix == "home_team" else "PTS_home"
            team_suffix = prefix.split("_")[0]
            fg3m_col = f"FG3M_{team_suffix}"
            fg3a_col = f"FG3A_{team_suffix}"

            for window in self.rolling_windows:
                grouped = df.groupby(team_col)
                # Volume-weighted (sum of makes / sum of attempts), not a mean of per-game
                # percentages — see the matching comment in _add_rolling_features.
                if fg3m_col in df.columns and fg3a_col in df.columns:
                    fg3m_roll = grouped[fg3m_col].transform(
                        lambda x, w=window: x.shift(1).rolling(w, min_periods=1).sum()
                    )
                    fg3a_roll = grouped[fg3a_col].transform(
                        lambda x, w=window: x.shift(1).rolling(w, min_periods=1).sum()
                    )
                    new_cols[f"{prefix}_fg3_pct_L{window}"] = fg3m_roll / fg3a_roll.replace(0, np.nan)
                new_cols[f"{prefix}_off_eff_L{window}"] = grouped[pts_col].transform(
                    lambda x, w=window: x.shift(1).rolling(w, min_periods=1).mean()
                )
                new_cols[f"{prefix}_def_eff_L{window}"] = grouped[opp_pts_col].transform(
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
            if f"home_team_def_eff_L{window}" not in df.columns:
                continue

            for opp_stat, home_col, away_col in [
                ("opp_def_quality", f"away_team_def_eff_L{window}", f"home_team_def_eff_L{window}"),
                ("opp_off_quality", f"away_team_off_eff_L{window}", f"home_team_off_eff_L{window}"),
            ]:
                home_rows = pd.DataFrame(
                    {
                        "GAME_DATE": df["GAME_DATE"].values,
                        "team_id": df["HOME_TEAM_ID"].values,
                        "opp_q": df[home_col].values,
                    }
                )
                away_rows = pd.DataFrame(
                    {
                        "GAME_DATE": df["GAME_DATE"].values,
                        "team_id": df["AWAY_TEAM_ID"].values,
                        "opp_q": df[away_col].values,
                    }
                )
                long_df = pd.concat([home_rows, away_rows]).sort_values("GAME_DATE").reset_index(drop=True)
                long_df["rolling"] = long_df.groupby("team_id")["opp_q"].transform(
                    lambda x, w=window: x.shift(1).rolling(w, min_periods=1).mean()
                )

                for team_col, prefix in [("HOME_TEAM_ID", "home_team"), ("AWAY_TEAM_ID", "away_team")]:
                    query = df[["GAME_DATE", team_col]].rename(columns={team_col: "team_id"})
                    merged = query.merge(
                        long_df[["GAME_DATE", "team_id", "rolling"]],
                        on=["GAME_DATE", "team_id"],
                        how="left",
                    )
                    new_cols[f"{prefix}_{opp_stat}_L{window}"] = merged["rolling"].values

        return pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)

    def _add_home_advantage_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add per-team venue delta: rolling avg pts at home minus rolling avg pts on the road.

        The role-split rolling features already capture venue-specific form (home_off_eff tracks
        a team when they're home, away_off_eff when they're away). This feature makes the
        home/road gap explicit, capturing teams that are particularly strong or weak at home.
        """
        new_cols = {}
        for window in self.rolling_windows:
            venue_cols = self._compute_venue_delta(df, window)
            new_cols.update(venue_cols)
        return pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)

    def _compute_venue_delta(self, df: pd.DataFrame, window: int) -> dict:
        home_lookup = (
            df[["GAME_DATE", "HOME_TEAM_ID"]]
            .assign(
                home_roll=df.groupby("HOME_TEAM_ID")["PTS_home"].transform(
                    lambda x, w=window: x.shift(1).rolling(w, min_periods=1).mean()
                )
            )
            .rename(columns={"HOME_TEAM_ID": "team_id"})
            .sort_values("GAME_DATE")
        )
        away_lookup = (
            df[["GAME_DATE", "AWAY_TEAM_ID"]]
            .assign(
                away_roll=df.groupby("AWAY_TEAM_ID")["PTS_away"].transform(
                    lambda x, w=window: x.shift(1).rolling(w, min_periods=1).mean()
                )
            )
            .rename(columns={"AWAY_TEAM_ID": "team_id"})
            .sort_values("GAME_DATE")
        )

        home_query = df[["GAME_DATE", "HOME_TEAM_ID"]].rename(columns={"HOME_TEAM_ID": "team_id"})
        away_query = df[["GAME_DATE", "AWAY_TEAM_ID"]].rename(columns={"AWAY_TEAM_ID": "team_id"})

        home_team_away_roll = pd.merge_asof(
            home_query, away_lookup, on="GAME_DATE", by="team_id", direction="backward"
        )["away_roll"]
        away_team_home_roll = pd.merge_asof(
            away_query, home_lookup, on="GAME_DATE", by="team_id", direction="backward"
        )["home_roll"]

        home_roll = df.groupby("HOME_TEAM_ID")["PTS_home"].transform(
            lambda x, w=window: x.shift(1).rolling(w, min_periods=1).mean()
        )
        away_roll = df.groupby("AWAY_TEAM_ID")["PTS_away"].transform(
            lambda x, w=window: x.shift(1).rolling(w, min_periods=1).mean()
        )

        return {
            f"home_team_venue_delta_L{window}": home_roll.values - home_team_away_roll.values,
            f"away_team_venue_delta_L{window}": away_team_home_roll.values - away_roll.values,
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
            if f"home_team_off_eff_L{window}" in df.columns:
                new_cols[f"home_off_vs_away_def_L{window}"] = (
                    df[f"home_team_off_eff_L{window}"] - df[f"away_team_def_eff_L{window}"]
                )
                new_cols[f"away_off_vs_home_def_L{window}"] = (
                    df[f"away_team_off_eff_L{window}"] - df[f"home_team_def_eff_L{window}"]
                )

            if f"home_team_fg3_pct_L{window}" in df.columns:
                new_cols[f"home_3pt_advantage_L{window}"] = (
                    df[f"home_team_fg3_pct_L{window}"] - df[f"away_team_fg3_pct_L{window}"]
                )

            if f"home_team_win_pct_L{window}" in df.columns:
                new_cols[f"form_differential_L{window}"] = (
                    df[f"home_team_win_pct_L{window}"] - df[f"away_team_win_pct_L{window}"]
                )

            if f"home_team_diff_avg_L{window}" in df.columns:
                new_cols[f"strength_differential_L{window}"] = (
                    df[f"home_team_diff_avg_L{window}"] - df[f"away_team_diff_avg_L{window}"]
                )

        return pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)

    def _add_h2h_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add head-to-head historical features from the current home team's perspective.

        Features computed:
        - h2h_home_margin_L{mw}: rolling avg margin over last mw games
        - h2h_home_win_rate_L{ww}: rolling win% over last ww games
        - h2h_win_pct_3yr: win% over last 3 seasons
        - h2h_avg_diff: average point differential across all history
        - h2h_home_win_pct: win% when playing at home
        - h2h_away_win_pct: win% when playing away
        """
        mw = self.h2h_margin_window
        ww = self.h2h_win_rate_window

        # Temp columns needed for groupby operations
        df["matchup_key"] = df.apply(
            lambda row: "_".join(sorted([str(row["HOME_TEAM_ID"]), str(row["AWAY_TEAM_ID"])])), axis=1
        )
        df["_canonical_team"] = df[["HOME_TEAM_ID", "AWAY_TEAM_ID"]].min(axis=1)
        df["_canonical_margin"] = df.apply(
            lambda r: r["POINT_DIFF"] if r["HOME_TEAM_ID"] == r["_canonical_team"] else -r["POINT_DIFF"],
            axis=1,
        )
        # For h2h features, process each matchup_key separately to avoid index issues
        h2h_results = {}
        for key in ["_h2h_margin_canon", "_h2h_win_canon", "_h2h_win_3yr_canon", "_h2h_avg_diff_canon"]:
            h2h_results[key] = pd.Series(index=df.index, dtype=float)

        for matchup_key in df["matchup_key"].unique():
            mask = df["matchup_key"] == matchup_key
            group_indices = df[mask].index
            group_df = df.loc[group_indices].copy()

            # _h2h_margin_canon
            h2h_results["_h2h_margin_canon"].loc[group_indices] = (
                group_df["_canonical_margin"].shift(1).rolling(mw, min_periods=1).mean().values
            )

            # _h2h_win_canon
            h2h_results["_h2h_win_canon"].loc[group_indices] = (
                (group_df["_canonical_margin"] > 0).shift(1).rolling(ww, min_periods=1).mean().values
            )

            # _h2h_win_3yr_canon
            h2h_results["_h2h_win_3yr_canon"].loc[group_indices] = self._compute_h2h_3year_win_pct(
                group_df
            ).values

            # _h2h_avg_diff_canon
            h2h_results["_h2h_avg_diff_canon"].loc[group_indices] = (
                group_df["_canonical_margin"].shift(1).expanding(min_periods=1).mean().values
            )

        for key, series in h2h_results.items():
            df[key] = series

        # 3. Home/away split win percentages (from home team perspective)
        df["_h2h_home_win_pct"] = self._compute_h2h_home_away_splits(df, "home")
        df["_h2h_away_win_pct"] = self._compute_h2h_home_away_splits(df, "away")

        is_canon_home = df["HOME_TEAM_ID"] == df["_canonical_team"]
        new_cols = {
            f"h2h_home_margin_L{mw}": df["_h2h_margin_canon"].where(is_canon_home, -df["_h2h_margin_canon"]),
            f"h2h_home_win_rate_L{ww}": df["_h2h_win_canon"].where(is_canon_home, 1 - df["_h2h_win_canon"]),
            "h2h_win_pct_3yr": df["_h2h_win_3yr_canon"].where(is_canon_home, 1 - df["_h2h_win_3yr_canon"]),
            "h2h_avg_diff": df["_h2h_avg_diff_canon"].where(is_canon_home, -df["_h2h_avg_diff_canon"]),
            "h2h_home_win_pct": df["_h2h_home_win_pct"],
            "h2h_away_win_pct": df["_h2h_away_win_pct"],
        }

        df.drop(
            columns=[
                "matchup_key",
                "_canonical_team",
                "_canonical_margin",
                "_h2h_margin_canon",
                "_h2h_win_canon",
                "_h2h_win_3yr_canon",
                "_h2h_avg_diff_canon",
                "_h2h_home_win_pct",
                "_h2h_away_win_pct",
            ],
            inplace=True,
        )

        return pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)

    def _compute_h2h_3year_win_pct(self, matchup_games: pd.DataFrame) -> pd.Series:
        """
        Compute head-to-head win% over last 3 seasons.

        Args:
            matchup_games: DataFrame with all games between a specific pair of teams, sorted by date

        Returns:
            Series with win% from canonical team's perspective, shifted (no leakage)
        """
        if len(matchup_games) == 0:
            return pd.Series(dtype=float)

        # Preserve original index but sort by date
        orig_index = matchup_games.index
        matchup_sorted = matchup_games.sort_values("GAME_DATE")
        sorted_orig_index = matchup_sorted.index  # original labels, in date-sorted order
        matchup_sorted = matchup_sorted.reset_index(drop=True)
        current_season = matchup_sorted["SEASON_ID"].values
        canonical_win = (matchup_sorted["_canonical_margin"] > 0).astype(float).values

        result = []
        for i in range(len(matchup_sorted)):
            if i == 0:
                result.append(float("nan"))
            else:
                # Look back at all games within last 3 seasons
                curr_season = current_season[i]
                recent_3yr_mask = current_season[:i] >= curr_season - 3
                if recent_3yr_mask.sum() == 0:
                    result.append(float("nan"))
                else:
                    win_pct = canonical_win[:i][recent_3yr_mask].mean()
                    result.append(win_pct)

        # Return Series with original index
        result_series = pd.Series(result, index=sorted_orig_index)
        return result_series.reindex(orig_index)

    def _compute_h2h_home_away_splits(self, df: pd.DataFrame, venue_type: str) -> pd.Series:
        """
        Compute home team's win% based on venue (home or away).

        Args:
            df: Games DataFrame with HOME_TEAM_ID, AWAY_TEAM_ID, GAME_DATE, POINT_DIFF
            venue_type: 'home' or 'away'

        Returns:
            Series with win% (indexed by df.index)
        """
        result = pd.Series(float("nan"), index=df.index)

        for idx in df.index:
            if idx == 0:
                continue

            home_team_id = df.loc[idx, "HOME_TEAM_ID"]
            away_team_id = df.loc[idx, "AWAY_TEAM_ID"]

            # Get all prior games
            prior = df.loc[: idx - 1]

            if venue_type == "home":
                # Home team playing at home against this opponent
                matching = prior[
                    (prior["HOME_TEAM_ID"] == home_team_id) & (prior["AWAY_TEAM_ID"] == away_team_id)
                ]
                if len(matching) > 0:
                    wins = (matching["POINT_DIFF"] > 0).sum()
                    result.loc[idx] = wins / len(matching)
            else:  # away
                # Home team playing away against this opponent
                matching = prior[
                    (prior["HOME_TEAM_ID"] == away_team_id) & (prior["AWAY_TEAM_ID"] == home_team_id)
                ]
                if len(matching) > 0:
                    # Win when away means POINT_DIFF < 0
                    wins = (matching["POINT_DIFF"] < 0).sum()
                    result.loc[idx] = wins / len(matching)

        return result

    @staticmethod
    def _haversine_miles(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        R = 3958.8  # Earth radius in miles
        dlat = math.radians(lat2 - lat1)
        dlon = math.radians(lon2 - lon1)
        a = (
            math.sin(dlat / 2) ** 2
            + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2
        )
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
        home_rows = pd.DataFrame(
            {
                "GAME_DATE": df["GAME_DATE"].values,
                "team_id": df["HOME_TEAM_ID"].values,
                "city_team": df["HOME_TEAM_ID"].values,  # game is in home team's city
            }
        )
        away_rows = pd.DataFrame(
            {
                "GAME_DATE": df["GAME_DATE"].values,
                "team_id": df["AWAY_TEAM_ID"].values,
                "city_team": df["HOME_TEAM_ID"].values,  # away team travels to home team's city
            }
        )
        long_df = pd.concat([home_rows, away_rows]).sort_values("GAME_DATE").reset_index(drop=True)

        long_df["prev_city_team"] = long_df.groupby("team_id")["city_team"].shift(1)
        # Default previous city to team's own home (no travel) when no prior game exists
        long_df["prev_city_team"] = long_df["prev_city_team"].fillna(long_df["team_id"])

        def travel_miles(row):
            curr = loc.get(int(row["city_team"]))
            prev = loc.get(int(row["prev_city_team"]))
            if curr is None or prev is None:
                return 0.0
            return self._haversine_miles(prev[0], prev[1], curr[0], curr[1])

        def tz_shift(row):
            curr = loc.get(int(row["city_team"]))
            prev = loc.get(int(row["prev_city_team"]))
            if curr is None or prev is None:
                return 0
            return curr[2] - prev[2]  # positive = traveled east

        long_df["travel_miles"] = long_df.apply(travel_miles, axis=1)
        long_df["tz_shift"] = long_df.apply(tz_shift, axis=1)

        # Rolling travel miles over last 7 and 14 days (day-windows capture road-trip
        # fatigue better than game-count windows since schedule density varies).
        long_df["GAME_DATE"] = pd.to_datetime(long_df["GAME_DATE"])
        long_df = long_df.sort_values(["team_id", "GAME_DATE"])
        for days in [7, 14]:
            col = f"travel_miles_{days}d"
            long_df[col] = long_df.groupby("team_id", group_keys=False).apply(
                lambda g, d=days: (
                    g.set_index("GAME_DATE")["travel_miles"]
                    .rolling(f"{d}D", closed="both")
                    .sum()
                    .set_axis(g.index)
                )
            )

        rolling_cols = ["travel_miles_7d", "travel_miles_14d"]
        new_cols = {}
        for team_col, prefix in [("HOME_TEAM_ID", "home_team"), ("AWAY_TEAM_ID", "away_team")]:
            query = df[["GAME_DATE", team_col]].rename(columns={team_col: "team_id"})
            query["GAME_DATE"] = pd.to_datetime(query["GAME_DATE"])
            merged = query.merge(
                long_df[["GAME_DATE", "team_id", "travel_miles", "tz_shift"] + rolling_cols],
                on=["GAME_DATE", "team_id"],
                how="left",
            )
            new_cols[f"{prefix}_travel_miles"] = merged["travel_miles"].fillna(0).values
            new_cols[f"{prefix}_tz_shift"] = merged["tz_shift"].fillna(0).values
            new_cols[f"{prefix}_travel_miles_7d"] = merged["travel_miles_7d"].fillna(0).values
            new_cols[f"{prefix}_travel_miles_14d"] = merged["travel_miles_14d"].fillna(0).values

        return pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)

    def _add_elo_features(self, df: pd.DataFrame, context_end_date: Optional[str] = None) -> pd.DataFrame:
        """
        Add pre-game Elo ratings (home_team_elo, away_team_elo, elo_diff).

        Elo is inherently sequential — a team's rating depends on every prior
        result, not just a recent window. So ratings are computed once over
        the full chronological game history up to `context_end_date` (not just
        the rows in `df`, but never beyond this call's own applicable end date
        either — see `create_all_features`'s docstring), then merged onto `df`
        by GAME_ID. This ensures val/test games carry forward ratings
        accumulated during train, rather than restarting at initial_rating
        each split, WITHOUT a train-time call being able to reach into
        val/test-period games (the bug this replaced: this used to always load
        through the global `datasets_loading.test_end_date`, regardless of
        which split was being processed).
        """
        cfg = load_config()
        if not cfg.elo_features or not cfg.elo_features.enabled:
            return df

        from src.data_processing.data_loader import NBADataLoader
        from src.feature_engineering.elo import compute_elo_momentum, compute_elo_ratings

        loader = NBADataLoader(db_path=cfg.data_paths.raw_db)
        try:
            all_games = loader.load_games(
                start_date=cfg.datasets_loading.data_start_date,
                end_date=context_end_date,
                allowed_season_types=cfg.datasets_loading.context_season_types
                or cfg.datasets_loading.allowed_season_types,
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
        momentum_df = compute_elo_momentum(all_games, elo_df, windows=self.rolling_windows)

        merged = df[["GAME_ID"]].merge(elo_df, on="GAME_ID", how="left")
        momentum_merged = df[["GAME_ID"]].merge(momentum_df, on="GAME_ID", how="left")

        new_cols = {
            "home_team_elo": merged["home_team_elo"].values,
            "away_team_elo": merged["away_team_elo"].values,
            "elo_diff": merged["home_team_elo"].values
            + elo_cfg.home_advantage
            - merged["away_team_elo"].values,
        }
        for col in momentum_df.columns:
            if col == "GAME_ID":
                continue
            new_cols[col] = momentum_merged[col].values
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
                "SELECT game_date, team_id, n_out, n_questionable, team_deficit "
                "FROM injury_features WHERE scorer = ?",
                conn,
                params=(scorer,),
            )

        injury_df["game_date"] = pd.to_datetime(injury_df["game_date"]).dt.normalize()
        game_dates = pd.to_datetime(df["GAME_DATE"]).dt.normalize()

        # E4 (EXPERIMENTS.md, session rs_20260808_1): zero_fill (status quo)
        # treats "no matching injury_df row" as "confirmed nobody out" (0).
        # native_nan instead leaves it as a true missing value, letting
        # CatBoost's own missing-value handling operate -- distinguishing
        # "no injury data available" from "checked, nobody out" the same way
        # `has_injury_data` already lets downstream code do, but at the
        # feature-value level instead of via a separate indicator column.
        native_nan = cfg.injury_features.missing_value_strategy == InjuryMissingValueStrategy.native_nan

        new_cols = {}
        home_merged, away_merged = None, None
        for team_col, prefix in [("HOME_TEAM_ID", "home_team"), ("AWAY_TEAM_ID", "away_team")]:
            lookup = pd.DataFrame(
                {
                    "game_date": game_dates.values,
                    "team_id": df[team_col].values,
                }
            )
            merged = lookup.merge(injury_df, on=["game_date", "team_id"], how="left")
            if native_nan:
                new_cols[f"{prefix}_n_out"] = merged["n_out"].values
                new_cols[f"{prefix}_n_questionable"] = merged["n_questionable"].values
                new_cols[f"{prefix}_team_deficit"] = merged["team_deficit"].values
            else:
                new_cols[f"{prefix}_n_out"] = merged["n_out"].fillna(0).astype(int).values
                new_cols[f"{prefix}_n_questionable"] = merged["n_questionable"].fillna(0).astype(int).values
                new_cols[f"{prefix}_team_deficit"] = merged["team_deficit"].fillna(0).values
            if prefix == "home_team":
                home_merged = merged
            else:
                away_merged = merged

        if native_nan:
            new_cols["team_deficit_diff"] = (
                home_merged["team_deficit"].values - away_merged["team_deficit"].values
            )
        else:
            new_cols["team_deficit_diff"] = (
                home_merged["team_deficit"].fillna(0).values - away_merged["team_deficit"].fillna(0).values
            )

        dates_with_coverage = set(injury_df["game_date"])
        new_cols["has_injury_data"] = game_dates.isin(dates_with_coverage).astype(int)

        return pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)

    def _add_style_matchup_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        A7 style-matchup score (src/matchups/) — precomputed offline by
        src/matchups/precompute_scores.py into outputs/style_fingerprint_cache.sqlite's
        style_matchup_scores table (game_id -> style_matchup_score/confidence),
        NOT computed live here: this method only left-joins the cached result
        onto df by GAME_ID, keeping this diff small and reusing the
        already-validated KNN similarity-search pipeline as-is (see
        docs/a7_style_matchup_design.md / docs/a7_phase_log.md's KNN-Score
        Integration Test section for the feature-integration test this wiring
        supports).

        KNOWN GAP (deliberately deferred, not an oversight): this exact-GAME_ID
        join has the same live-prediction bug `_add_style_fingerprint_features`
        below used to have — predict_game.py's synthetic 'upcoming' GAME_ID can
        never match a cached game_id, so this always returns NaN in live
        prediction. Left as-is here because this feature stays `enabled: false`
        (not adopted, see configs/config.yaml). If it's ever revisited/adopted,
        it needs the same asof-on-(team_id, game_date) fix applied below.
        """
        cfg = load_config()
        if not cfg.style_matchup or not cfg.style_matchup.enabled:
            return df

        cache_db = Path(CACHE_DB)
        if not cache_db.exists():
            logger.warning(f"Style matchup cache not found at {cache_db} — skipping style matchup features")
            return df

        with sqlite3.connect(f"file:{cache_db}?mode=ro", uri=True) as conn:
            scores_df = pd.read_sql_query(
                "SELECT game_id, style_matchup_score, confidence FROM style_matchup_scores",
                conn,
            )

        merged = df[["GAME_ID"]].merge(scores_df, left_on="GAME_ID", right_on="game_id", how="left")

        new_cols = {
            "style_matchup_score": merged["style_matchup_score"].values,
            "style_matchup_confidence": merged["confidence"].values,
        }
        return pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)

    # Metrics with a calibrated (Layer 2, injury-adjusted) value already validated
    # across A7's early stages -- see docs/a7_phase_log.md.
    _RAW_STYLE_CALIBRATED_METRICS = [
        "pace_score",
        "three_pt_reliance",
        "paint_activity",
        "defensive_rating",
        "assist_rate",
    ]
    # Added by the raw-fingerprint feature redesign -- offensive-quality counterpart
    # to defensive_rating. Layer 1 (uncalibrated) only, a deliberate scope cut -- see
    # fingerprint.py's docstring. Read from layer=1 explicitly rather than layer=2
    # (where it would be numerically identical, since no injury delta touches it --
    # see injury_layer.py).
    _RAW_STYLE_UNCALIBRATED_METRIC = "offensive_rating"
    # Track C pace/possession swap-in test (docs/NEW_DATA_FEASIBILITY.md): official
    # PACE/POSS from nba_api's TeamGameLogs (Advanced). Layer 1 only, same treatment
    # as offensive_rating -- gated separately via style_matchup.official_pace_enabled
    # (default false) so this candidate is independently ablatable.
    _RAW_STYLE_OFFICIAL_PACE_METRICS = ["official_pace", "official_poss"]

    def _add_style_fingerprint_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        A7's raw-fingerprint feature redesign — raw per-team style-fingerprint
        components plus explicit home-vs-away differentials, mirroring
        `_add_matchup_features`'s existing pattern (home_off_vs_away_def_L{window},
        etc.) instead of `_add_style_matchup_features`'s KNN-similarity-search
        lookup above.

        Motivation (see docs/a7_phase_log.md's Raw-Fingerprint Feature Redesign /
        KNN-Score Integration Test sections): the KNN-score integration test found
        `style_matchup_score` — a pre-aggregated KNN-average "mini-prediction" —
        added essentially zero value to the trained model (29th of 109 features,
        confidence had zero importance) despite a decent standalone correlation.
        CatBoost never saw the ingredients that produced that one opaque number.
        This method exposes the raw ingredients directly instead, so CatBoost can
        learn which specific stylistic clashes matter on its own, the same way it
        already does for `_add_matchup_features`'s rolling-efficiency differentials.

        No KNN/similarity search is involved at all — this reads directly from
        the already-existing `matchup_fingerprints` cache table (built offline by
        src/matchups/fingerprint.py + src/matchups/injury_layer.py): layer=2
        (injury-adjusted, calibrated) for the five original hand-picked metrics,
        layer=1 (uncalibrated — no injury delta calibrated for it this round, a
        deliberate scope cut) for the new `offensive_rating` metric.

        Join strategy — asof on (team_id, game_date), NOT exact game_id match:
        an exact `game_id` join (the original implementation) works for training,
        where every row is a real, already-played, already-cached game. But it is
        silently broken for live prediction: predict_game.py builds a synthetic
        row for the matchup being predicted with GAME_ID='upcoming' (a placeholder
        string that can never match any cached game_id), so an exact-match join
        would leave every style-fingerprint column NaN for the one row we're
        actually trying to predict. `_add_injury_features` above already solves
        the equivalent problem for injuries by joining on (team_id, game_date), a
        natural key that works for any date, not just already-played games. Here
        we do the same via `pd.merge_asof`: for each team_id, take that team's
        most recently *computed* fingerprint at or before the target game_date
        (direction="backward", allow_exact_matches=True). For historical/training
        rows this returns exactly the same value as the old exact-game_id join —
        each cached fingerprint's own game_date is trivially "the most recent
        fingerprint at or before itself" — so this changes *how* the lookup
        matches, not what value it produces for any already-cached row (verified
        directly against the full cache: zero mismatches, zero cases where one
        method had NaN and the other didn't). For a genuinely-uncached team_id/
        date (either no cache built at all, or too little history yet — same
        `min_games_played` gate `precompute_scores.py` already applies), asof
        naturally returns NaN, matching today's existing NaN convention.

        `pd.merge_asof` requires both sides sorted ascending by the "on" column
        (`game_date`) — the cache is explicitly sorted here since it comes back
        from SQL in no guaranteed order; `df` is already sorted ascending by
        GAME_DATE at the top of `create_all_features`, same assumption already
        relied on by `_compute_venue_delta`'s merge_asof calls above.

        Gated independently from `_add_style_matchup_features` via the separate
        `style_matchup.raw_features_enabled` flag (default true — adopted as the
        committed production config: away_style_pace_score/home_style_pace_score
        were the #1/#2 most important features in the trained model, with a
        consistent total_mae improvement on both val and test splits, at the cost
        of flat-to-slightly-worse win_acc/brier). `_add_style_matchup_features`'s
        KNN-lookup flag stays independently toggleable (default false, not
        adopted — no real signal found).

        Adds, for each of the 6 metrics: two raw columns (home_style_{metric},
        away_style_{metric}) and one differential column (style_{metric}_diff,
        home - away) — 18 new columns total. If `style_matchup.official_pace_enabled`
        is also true, adds official_pace/official_poss (nba_api's own PACE/POSS,
        src/matchups/pace_possession.py) the same way — 6 more columns, alongside
        pace_score rather than replacing it (Track C pace/possession swap-in test,
        docs/NEW_DATA_FEASIBILITY.md).

        Unlike `_add_style_matchup_features` above (soft warn+skip on a missing
        cache — that feature stays optional/disabled by default), this flag is
        now the committed default, so a missing cache must not silently produce a
        model missing its top features with no error: raises RuntimeError instead.

        NOTE: `_add_style_matchup_features` (the KNN-lookup method above) is NOT
        fixed by this change and still joins by exact GAME_ID — it remains
        `enabled: false` (not adopted) and is deliberately out of scope for this
        pass. If that feature is ever revisited/adopted, it has the exact same
        live-prediction NaN bug this method used to have, and would need the same
        asof-on-(team_id, game_date) treatment.
        """
        cfg = load_config()
        if not cfg.style_matchup or not cfg.style_matchup.raw_features_enabled:
            return df

        cache_db = Path(CACHE_DB)
        if not cache_db.exists():
            raise RuntimeError(
                f"Style fingerprint cache not found at {cache_db} — "
                "run `python src/matchups/precompute_scores.py` first to build it "
                "(style_matchup.raw_features_enabled is true, so this feature is "
                "required, not optional)."
            )

        self._warn_if_style_fingerprint_cache_stale(cache_db)

        calibrated = self._RAW_STYLE_CALIBRATED_METRICS
        uncalibrated = [self._RAW_STYLE_UNCALIBRATED_METRIC]
        if cfg.style_matchup.official_pace_enabled:
            uncalibrated = uncalibrated + self._RAW_STYLE_OFFICIAL_PACE_METRICS
        all_metrics = calibrated + uncalibrated

        with sqlite3.connect(f"file:{cache_db}?mode=ro", uri=True) as conn:
            layer2 = pd.read_sql_query(
                "SELECT game_id, team_id, game_date, "
                + ", ".join(calibrated)
                + " FROM matchup_fingerprints WHERE layer = 2",
                conn,
            )
            layer1_uncalibrated = pd.read_sql_query(
                "SELECT game_id, team_id, "
                + ", ".join(uncalibrated)
                + " FROM matchup_fingerprints WHERE layer = 1",
                conn,
            )

        fingerprints = layer2.merge(layer1_uncalibrated, on=["game_id", "team_id"], how="left")
        # Normalize to date granularity (drop any time-of-day component) so this
        # matches on the same terms as _add_injury_features's (team_id, game_date)
        # join, and sort ascending — merge_asof's "on" column must be sorted on
        # both sides (globally, not just within each `by` group).
        fingerprints["game_date"] = pd.to_datetime(fingerprints["game_date"]).dt.normalize()
        fingerprints = fingerprints.sort_values("game_date").reset_index(drop=True)

        query_dates = pd.to_datetime(df["GAME_DATE"]).dt.normalize()

        new_cols = {}
        side_values = {}
        for team_col, side in [("HOME_TEAM_ID", "home"), ("AWAY_TEAM_ID", "away")]:
            lookup = pd.DataFrame(
                {
                    "game_date": query_dates.values,
                    "team_id": df[team_col].values,
                }
            )
            merged = pd.merge_asof(
                lookup,
                fingerprints[["game_date", "team_id"] + all_metrics],
                on="game_date",
                by="team_id",
                direction="backward",
                allow_exact_matches=True,
            )
            side_values[side] = {metric: merged[metric].values for metric in all_metrics}
            for metric in all_metrics:
                new_cols[f"{side}_style_{metric}"] = side_values[side][metric]

        for metric in all_metrics:
            new_cols[f"style_{metric}_diff"] = side_values["home"][metric] - side_values["away"][metric]

        return pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)

    @staticmethod
    def _warn_if_style_fingerprint_cache_stale(cache_db: Path) -> None:
        """`matchup_fingerprints` (CACHE_DB) is a periodic offline precompute
        (`src/matchups/precompute_scores.py`), not computed fresh per call — see
        that script's "OPERATIONAL REQUIREMENT" docstring. Nothing previously
        checked its age against the raw DB it's derived from (docs/PIPELINE_AUDIT.md's
        fingerprint-cache-freshness finding), so a cache that stopped being
        refreshed would silently keep serving each team's last-cached fingerprint
        with no signal that it had fallen behind. This only warns (asof lookups in
        `_add_style_fingerprint_features` above already degrade gracefully to the
        most recent cached value, so staleness isn't fatal) — it makes drift
        visible instead of leaving it undetectable.
        """
        with sqlite3.connect(f"file:{cache_db}?mode=ro", uri=True) as conn:
            cache_max = conn.execute("SELECT MAX(game_date) FROM matchup_fingerprints").fetchone()[0]
        with sqlite3.connect(f"file:{NBA_API_DB}?mode=ro", uri=True) as conn:
            raw_max = conn.execute(
                "SELECT MAX(game_date) FROM game WHERE season_type = 'Regular Season'"
            ).fetchone()[0]

        if cache_max is None or raw_max is None:
            return
        if pd.Timestamp(cache_max) < pd.Timestamp(raw_max):
            logger.warning(
                f"style_fingerprint cache is stale: cache max game_date={cache_max}, "
                f"raw DB regular-season max game_date={raw_max}. Live lookups will "
                "keep using each team's last-cached fingerprint until the cache is "
                "rebuilt — run `python src/matchups/precompute_scores.py` to refresh."
            )

    def _add_on_off_splits_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Player on/off-court splits, integrated as an injury-aware team-level feature
        rather than raw per-player columns (per docs/on_off_splits_decisions.md's
        open design question and the coordinator's phase-2 direction): for each
        team-game, sum the on/off plus-minus impact of that team's currently-missing
        players (per `injury_features.sqlite`'s `player_injuries` table, already
        joined by `_add_injury_features` above via `(team_id, game_date)`) — an
        "expected point-differential impact from currently-missing players" signal,
        directly extending the existing injury-feature join pattern instead of
        introducing an arbitrary raw per-player aggregate or a single
        headline-player proxy (both considered and rejected — see the decisions
        doc's Open Risks §6.1).

        `Out` players count at full weight; `Doubtful` players count at
        `injury_features.doubtful_weight` (0.8) — the same fractional weight
        `formula_scorer.compute_team_deficit` already applies to Doubtful players
        for the unrelated team-deficit feature, reused here for consistency rather
        than inventing a second weight. `Questionable`/`Day-To-Day` players are
        still excluded entirely, mirroring `compute_team_deficit` again (which
        counts them separately but never folds them into its weighted sum) —
        players in that status usually do play, so treating them as partially
        "missing" would mostly add noise rather than signal.

        Data source: `player_on_off_splits` (built by
        scripts/backfill_on_off_splits.py from nba_api's TeamPlayerOnOffSummary,
        see docs/on_off_splits_decisions.md), one row per
        (player_id, team_id, split_type, opponent_team_id, as_of_date checkpoint).
        Player identity for the "who's out" side comes from `player_injuries`
        (game_date, team_id, player_name, status) — resolved to player_id via the
        existing `player_name_resolution` table in the A7 style-fingerprint cache
        (`CACHE_DB`, confidence in ('high','medium')), reusing the exact same
        name-resolution table `src/matchups/injury_layer.py` already relies on for
        its own Out-player lookup, rather than rebuilding a second resolution table.

        Split preference per (out player, team-game): the venue-specific split
        (`home` for the home team's rows, `away` for the away team's rows) beats
        `overall` (always-available fallback). Implemented via two separate
        `merge_asof` lookups combined with `combine_first`.

        `vs_opponent` (this exact opponent) was deliberately dropped from this
        preference chain (was: vs_opponent > venue > overall) after reviewing real
        backfilled data: a single season only has 2-4 meetings between two specific
        teams, and vs_opponent checkpoints are taken per-game rather than on the
        same weekly cadence as venue/overall, so the value can swing enormously
        from meeting to meeting purely from small-sample arithmetic (e.g. one
        blowout as the only meeting so far dominates the number until enough games
        accumulate to dilute it — there's no guarantee that ever happens within a
        season). This is a real, structural volatility problem, not a coverage gap
        that more backfilling fixes. A proper fix would need multi-season pooling
        (mirroring `_add_h2h_features`'s 3-year lookback) — not implemented here;
        the already-backfilled `player_on_off_splits` rows with `split_type=
        'vs_opponent'` are left in place (harmless, reusable if that fix is ever
        built) but are no longer read by this method.

        Leakage guard: `date_to_nullable` was empirically confirmed INCLUSIVE of
        games played on that exact calendar date (Boston's cumulative GP jumped
        from 39 to 40 exactly on the date of an actual BOS game, not the day after
        — see the decisions doc / backfill script docstring). So the lookup key
        used here is `game_date - 1 day`, not `game_date` itself — this guarantees
        no same-day leakage regardless of how a cached checkpoint's `as_of_date`
        happens to line up with the target game's date. `merge_asof(direction=
        "backward", allow_exact_matches=True)` then finds the most recent
        checkpoint at or before that shifted date, working identically for
        historical training rows and for a live/future prediction date (same
        (team_id, game_date) natural-key join strategy `_add_injury_features` and
        `_add_style_fingerprint_features` already use — never an exact `game_id`
        match).

        A player's on_off_plus_minus is only trusted if BOTH their on-court AND
        off-court minutes independently meet `on_off_splits.min_on_off_minutes`
        (small-sample noise gate, applied when the cache is loaded, before any
        lookup). Gating on the min() of the two sides, not their sum, matters: a
        rotation player can rack up hundreds of "on" minutes while having only a
        handful of "off" minutes (rarely rested) — the combined total looks
        large, but the tiny "off" side alone produces a wildly noisy plus-minus
        (observed directly: a rookie with 620 on-minutes but only 4 off-minutes
        produced an on/off swing over +150, which is not a real effect, just a
        4-minute sample). Requiring both sides independently >= the threshold
        catches this; requiring only the sum does not.

        Adds, for each of home_team/away_team: `_missing_player_on_off_impact` (sum
        of resolved Out/Doubtful players' weighted on/off plus-minus — 0.0, not
        NaN, when no missing players are resolved, since "no one out" is a
        legitimate zero-impact case, not a missing-data case), `_n_missing_total`
        (count of Out/Doubtful players found, before resolution), and
        `_n_missing_resolved_on_off` (how many of those were successfully resolved
        to an on/off value — a confidence indicator, since a large gap between the
        two counts means the sum understates the true impact). Plus one
        differential column, `missing_player_on_off_impact_diff` (home − away),
        mirroring `_add_matchup_features`'s differential pattern.

        Soft-disabled (warn + skip, matching `_add_style_matchup_features`'s
        not-yet-adopted convention, not `_add_style_fingerprint_features`'s
        hard-raise) if any of the three required caches (on/off splits, injury
        features, name resolution) is missing — this feature has not yet gone
        through the ablation-pipeline adoption process those two features did.
        """
        cfg = load_config()
        if not cfg.on_off_splits or not cfg.on_off_splits.enabled:
            return df

        on_off_db = Path(cfg.on_off_splits.db_path)
        injury_db = Path(cfg.injury_features.db_path) if cfg.injury_features else None
        name_res_db = Path(CACHE_DB)

        if not on_off_db.exists() or injury_db is None or not injury_db.exists() or not name_res_db.exists():
            logger.warning(
                "On/off splits feature requires all of: "
                f"{on_off_db} (on/off cache), {injury_db} (injury cache), "
                f"{name_res_db} (name resolution) — one or more missing, skipping "
                "on/off splits features."
            )
            return df

        min_minutes = cfg.on_off_splits.min_on_off_minutes

        with sqlite3.connect(f"file:{on_off_db}?mode=ro", uri=True) as conn:
            # Only overall/home/away are read -- vs_opponent rows exist in this table
            # (see method docstring for why they're no longer used) but are excluded
            # here rather than fetched and ignored downstream.
            splits = pd.read_sql_query(
                "SELECT player_id, team_id, split_type, as_of_date, "
                "min_on, min_off, on_off_plus_minus FROM player_on_off_splits "
                "WHERE split_type != 'vs_opponent'",
                conn,
            )
        if splits.empty:
            logger.warning("player_on_off_splits cache is empty — skipping on/off splits features")
            return df

        splits["as_of_date"] = pd.to_datetime(splits["as_of_date"])
        # Gate on min(on, off), not the sum — see method docstring: a player can have
        # hundreds of "on" minutes and almost none "off" (or vice versa), which makes
        # the combined total look ample while the thin side alone is pure noise.
        min_side_minutes = np.minimum(splits["min_on"].fillna(0), splits["min_off"].fillna(0))
        splits = splits[min_side_minutes >= min_minutes].copy()

        doubtful_weight = cfg.injury_features.doubtful_weight if cfg.injury_features else 1.0

        with sqlite3.connect(f"file:{injury_db}?mode=ro", uri=True) as conn:
            out_players = pd.read_sql_query(
                "SELECT game_date, team_id, player_name, status FROM player_injuries "
                "WHERE status IN ('Out', 'Doubtful')",
                conn,
            )
        with sqlite3.connect(f"file:{name_res_db}?mode=ro", uri=True) as conn:
            name_res = pd.read_sql_query(
                "SELECT player_name, player_id FROM player_name_resolution "
                "WHERE confidence IN ('high', 'medium')",
                conn,
            )
        out_players["game_date"] = pd.to_datetime(out_players["game_date"]).dt.normalize()
        out_players["weight"] = np.where(out_players["status"] == "Out", 1.0, doubtful_weight)
        out_players_resolved = out_players.merge(name_res, on="player_name", how="inner")

        def _asof_lookup(
            rows: pd.DataFrame, pool: pd.DataFrame, split_type: str, by_cols: list[str]
        ) -> pd.Series:
            sub_pool = pool[pool["split_type"] == split_type].sort_values("as_of_date")
            if sub_pool.empty:
                return pd.Series([float("nan")] * len(rows), index=rows.index)
            # merge_asof's `by` columns require matching dtypes on both sides -- cast
            # explicitly rather than relying on pandas' inference (this caught a real
            # bug when `by_cols` included opponent_team_id for the now-removed
            # vs_opponent lookup: that column was NULL for every other split_type in
            # the same SQL read, so pandas inferred float64 for the whole column even
            # though it's non-null within the vs_opponent slice specifically).
            left = rows[["lookup_date"] + by_cols].copy()
            right = sub_pool[["as_of_date"] + by_cols + ["on_off_plus_minus"]].copy()
            for col in by_cols:
                left[col] = left[col].astype("int64")
                right[col] = right[col].astype("int64")
            merged = pd.merge_asof(
                left,
                right,
                left_on="lookup_date",
                right_on="as_of_date",
                by=by_cols,
                direction="backward",
                allow_exact_matches=True,
            )
            return merged["on_off_plus_minus"]

        new_cols = {}
        for team_col, venue_split, prefix in [
            ("HOME_TEAM_ID", "home", "home_team"),
            ("AWAY_TEAM_ID", "away", "away_team"),
        ]:
            rows = pd.DataFrame(
                {
                    "row_id": df.index,
                    "game_date": pd.to_datetime(df["GAME_DATE"]).dt.normalize().values,
                    "team_id": df[team_col].values,
                }
            )
            rows = rows.merge(
                out_players_resolved[["game_date", "team_id", "player_name", "player_id", "weight"]],
                on=["game_date", "team_id"],
                how="inner",
            )

            if rows.empty:
                new_cols[f"{prefix}_missing_player_on_off_impact"] = 0.0
                new_cols[f"{prefix}_n_missing_total"] = 0
                new_cols[f"{prefix}_n_missing_resolved_on_off"] = 0
                continue

            # Leakage guard: DateTo is confirmed INCLUSIVE of same-day games, so the
            # lookup key must be the day BEFORE the target game, not the game date
            # itself (see method docstring).
            rows["lookup_date"] = rows["game_date"] - pd.Timedelta(days=1)
            rows = rows.sort_values("lookup_date").reset_index(drop=True)

            venue = _asof_lookup(rows, splits, venue_split, ["player_id", "team_id"])
            overall = _asof_lookup(rows, splits, "overall", ["player_id", "team_id"])

            rows["impact"] = venue.combine_first(overall)
            rows["resolved"] = rows["impact"].notna().astype(int)
            rows["weighted_impact"] = rows["impact"] * rows["weight"]

            agg = rows.groupby("row_id").agg(
                impact_sum=("weighted_impact", "sum"),
                resolved_n=("resolved", "sum"),
                out_n=("player_id", "count"),
            )
            merged = pd.DataFrame(index=df.index).join(agg)
            new_cols[f"{prefix}_missing_player_on_off_impact"] = merged["impact_sum"].fillna(0.0).values
            new_cols[f"{prefix}_n_missing_total"] = merged["out_n"].fillna(0).astype(int).values
            new_cols[f"{prefix}_n_missing_resolved_on_off"] = (
                merged["resolved_n"].fillna(0).astype(int).values
            )

        new_cols["missing_player_on_off_impact_diff"] = (
            new_cols["home_team_missing_player_on_off_impact"]
            - new_cols["away_team_missing_player_on_off_impact"]
        )

        return pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)

    def _add_season_motivation_features(
        self, df: pd.DataFrame, context_end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Season motivation / seeding-incentive features. See
        `season_motivation.py`'s module docstring for what each signal
        computes and docs/SEASON_MOTIVATION_LOG.md for CV results/adoption.

        No new backfill needed -- standings/schedule derive in-memory from
        `game` (only `game_date`/team-id columns read from "future" rows,
        never outcomes). Delegates to `src/feature_engineering/season_motivation.py`,
        same separation `_add_elo_features` uses.

        `context_end_date` bounds the internal `all_games` load to this call's
        own applicable end date (never the global `datasets_loading.test_end_date`
        regardless of which split is being processed -- see `create_all_features`'s
        docstring). Standings/roster/preferred_opponent_delta are point-in-time
        snapshots, safe to recompute fresh from however much history is available
        on every call. `elo_margin_scale` is NOT -- it's a least-squares fit, so
        it's fit once (on this instance's first call, assumed train) and cached
        on `self._fitted_elo_margin_scale` for reuse, never refit using val/test
        outcomes.

        Adds, for each of home_team/away_team: `_motivation_score`,
        `_games_to_clinch_ceiling`/`_games_to_clinch_floor`,
        `_recent_minutes_trend_score` (all gated by
        `motivation_score_enabled` -- not adopted, see FINAL SUMMARY);
        `_performance_vs_expectation_score`/`_opponent_adjusted_form_score`
        (each independently gated, requires `elo_features.enabled=true` --
        not adopted, failed a window-robustness check);
        `_opponent_adjusted_off_score`/`_opponent_adjusted_def_score` (gated
        by `opponent_adjusted_efficiency_enabled` -- does NOT need
        `elo_features.enabled`, computed straight from points scored/allowed;
        docs/NEXT_PHASE_SESSIONS.md backlog item 5's retrospective
        opponent-adjustment idea, extending `opponent_adjusted_form`'s own
        template to off_eff/def_eff); and `_preferred_opponent_delta` (gated
        by `preferred_opponent_delta_enabled` -- adopted, the only signal to
        pass that check).

        Soft-disabled (warn + skip) if the injury features cache is missing.
        """
        cfg = load_config()
        if not cfg.season_motivation or not cfg.season_motivation.enabled:
            return df

        injury_db = Path(cfg.injury_features.db_path) if cfg.injury_features else None
        if injury_db is None or not injury_db.exists():
            logger.warning(
                f"Injury features DB not found at {injury_db} -- season motivation's "
                "roster-behavior component depends on it, skipping season motivation features."
            )
            return df

        from src.data_processing.data_loader import NBADataLoader
        from src.feature_engineering.season_motivation import (
            _fit_elo_margin_scale,
            compute_opponent_adjusted_form_scores,
            compute_performance_vs_expectation_scores,
            compute_preferred_opponent_delta_scores,
            compute_recent_minutes_trend_scores,
            compute_roster_behavior_scores,
            compute_standings_metrics,
            compute_team_performance_history,
        )

        sm_cfg = cfg.season_motivation
        loader = NBADataLoader(db_path=cfg.data_paths.raw_db)
        try:
            all_games = loader.load_games(
                start_date=cfg.datasets_loading.data_start_date,
                end_date=context_end_date,
                allowed_season_types=cfg.datasets_loading.allowed_season_types,
            )
        finally:
            loader.close()

        game_dates = pd.to_datetime(df["GAME_DATE"]).dt.normalize()

        standings = None
        roster_behavior = None
        recent_minutes_trend = None
        if sm_cfg.motivation_score_enabled:
            standings = compute_standings_metrics(
                all_games,
                sm_cfg.playoff_line_seed,
                sm_cfg.direct_playoff_seed,
                sm_cfg.direct_playoff_weight,
            )

            season_start_by_season = (
                all_games.assign(GAME_DATE=pd.to_datetime(all_games["GAME_DATE"]).dt.normalize())
                .groupby("SEASON_ID")["GAME_DATE"]
                .min()
                .to_dict()
            )
            team_dates = pd.concat(
                [
                    pd.DataFrame(
                        {
                            "team_id": df["HOME_TEAM_ID"].values,
                            "game_date": game_dates.values,
                            "season_id": df["SEASON_ID"].values,
                        }
                    ),
                    pd.DataFrame(
                        {
                            "team_id": df["AWAY_TEAM_ID"].values,
                            "game_date": game_dates.values,
                            "season_id": df["SEASON_ID"].values,
                        }
                    ),
                ],
                ignore_index=True,
            )

            roster_behavior = compute_roster_behavior_scores(
                team_dates,
                str(injury_db),
                cfg.injury_features.importance_weights,
                sm_cfg.min_importance_games,
                season_start_by_season,
            )
            recent_minutes_trend = compute_recent_minutes_trend_scores(
                team_dates,
                str(injury_db),
                cfg.injury_features.importance_weights,
                sm_cfg.min_importance_games,
                season_start_by_season,
                sm_cfg.recent_trend_lookback_weeks,
            )

        performance_vs_expectation = None
        opponent_adjusted_form = None
        if sm_cfg.performance_vs_expectation_enabled or sm_cfg.opponent_adjusted_form_enabled:
            if not cfg.elo_features or not cfg.elo_features.enabled:
                logger.warning(
                    "performance_vs_expectation/opponent_adjusted_form need elo_features.enabled=true "
                    "(they reuse its ratings) -- skipping both, other season motivation columns unaffected."
                )
            else:
                from src.feature_engineering.elo import compute_elo_ratings

                elo_cfg = cfg.elo_features
                elo_ratings = compute_elo_ratings(
                    all_games,
                    initial_rating=elo_cfg.initial_rating,
                    k_factor=elo_cfg.k_factor,
                    home_advantage=elo_cfg.home_advantage,
                    mov_multiplier=elo_cfg.mov_multiplier,
                    season_regression=elo_cfg.season_regression,
                )
                # Fit ONCE on this instance's first call (assumed train -- see
                # __init__), reused unchanged on later val/test calls. This is
                # a least-squares fit, not a point-in-time snapshot -- refitting
                # it fresh on every call (using whatever `all_games` that call's
                # own context_end_date allows) would mean val/test outcomes
                # influence a "constant" the model also uses to score training
                # rows. See CLAUDE.md's leakage-safety rule.
                if self._fitted_elo_margin_scale is None:
                    self._fitted_elo_margin_scale = _fit_elo_margin_scale(
                        all_games, elo_ratings, elo_cfg.home_advantage
                    )
                elo_margin_scale = self._fitted_elo_margin_scale
                team_performance = compute_team_performance_history(
                    all_games,
                    elo_ratings,
                    elo_cfg.home_advantage,
                    elo_margin_scale,
                )
                if sm_cfg.performance_vs_expectation_enabled:
                    performance_vs_expectation = compute_performance_vs_expectation_scores(
                        team_performance,
                        sm_cfg.performance_vs_expectation_window,
                    )
                if sm_cfg.opponent_adjusted_form_enabled:
                    opponent_adjusted_form = compute_opponent_adjusted_form_scores(
                        team_performance,
                        sm_cfg.opponent_adjusted_form_window,
                    )

        opponent_adjusted_efficiency = None
        if sm_cfg.opponent_adjusted_efficiency_enabled:
            from src.feature_engineering.season_motivation import (
                compute_opponent_adjusted_efficiency_scores,
                compute_team_offense_defense_history,
            )

            off_def_history = compute_team_offense_defense_history(
                all_games, sm_cfg.opponent_adjusted_efficiency_window
            )
            opponent_adjusted_efficiency = compute_opponent_adjusted_efficiency_scores(
                off_def_history,
                sm_cfg.opponent_adjusted_efficiency_window,
            )

        preferred_opponent_delta = None
        if sm_cfg.preferred_opponent_delta_enabled:
            preferred_opponent_delta = compute_preferred_opponent_delta_scores(
                all_games,
                sm_cfg.preferred_opponent_delta_window_games,
            )

        new_cols = {}
        for team_col, prefix in [("HOME_TEAM_ID", "home_team"), ("AWAY_TEAM_ID", "away_team")]:
            lookup = pd.DataFrame(
                {
                    "season_id": df["SEASON_ID"].values,
                    "team_id": df[team_col].values,
                    "snapshot_date": game_dates.values,
                }
            )
            rb_lookup = pd.DataFrame({"team_id": df[team_col].values, "game_date": game_dates.values})

            if standings is not None:
                standings_merged = lookup.merge(
                    standings, on=["season_id", "team_id", "snapshot_date"], how="left"
                )
                rb_merged = rb_lookup.merge(roster_behavior, on=["team_id", "game_date"], how="left")
                roster_score = rb_merged["roster_behavior_score"].fillna(0.0).values

                trend_merged = rb_lookup.merge(recent_minutes_trend, on=["team_id", "game_date"], how="left")
                trend_score = trend_merged["recent_minutes_trend_score"].fillna(0.0).values

                pressure = standings_merged["pressure_raw"].fillna(0.0).values
                motivation = np.clip(pressure * (1 - sm_cfg.roster_behavior_weight * roster_score), 0.0, 1.0)

                new_cols[f"{prefix}_motivation_score"] = motivation
                new_cols[f"{prefix}_games_to_clinch_ceiling"] = (
                    standings_merged["games_to_clinch_ceiling"].fillna(0.0).values
                )
                new_cols[f"{prefix}_games_to_clinch_floor"] = (
                    standings_merged["games_to_clinch_floor"].fillna(0.0).values
                )
                new_cols[f"{prefix}_recent_minutes_trend_score"] = trend_score

            if performance_vs_expectation is not None:
                pve_merged = rb_lookup.merge(
                    performance_vs_expectation, on=["team_id", "game_date"], how="left"
                )
                new_cols[f"{prefix}_performance_vs_expectation_score"] = (
                    pve_merged["performance_vs_expectation_score"].fillna(0.0).values
                )
            if opponent_adjusted_form is not None:
                oaf_merged = rb_lookup.merge(opponent_adjusted_form, on=["team_id", "game_date"], how="left")
                new_cols[f"{prefix}_opponent_adjusted_form_score"] = (
                    oaf_merged["opponent_adjusted_form_score"].fillna(0.0).values
                )

            if opponent_adjusted_efficiency is not None:
                oae_merged = rb_lookup.merge(
                    opponent_adjusted_efficiency, on=["team_id", "game_date"], how="left"
                )
                new_cols[f"{prefix}_opponent_adjusted_off_score"] = (
                    oae_merged["opponent_adjusted_off_score"].fillna(0.0).values
                )
                new_cols[f"{prefix}_opponent_adjusted_def_score"] = (
                    oae_merged["opponent_adjusted_def_score"].fillna(0.0).values
                )

            if preferred_opponent_delta is not None:
                pod_merged = lookup.merge(
                    preferred_opponent_delta, on=["season_id", "team_id", "snapshot_date"], how="left"
                )
                new_cols[f"{prefix}_preferred_opponent_delta"] = (
                    pod_merged["preferred_opponent_delta"].fillna(0.0).values
                )

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
