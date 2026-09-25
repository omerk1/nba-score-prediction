"""
Builds the point-in-time feature row for a single upcoming (not-yet-played)
game, given team IDs and a date. Extracted from predict_game.py's own
inline logic so the CLI and src/serving/recommend.py share exactly one
implementation of the synthetic-row construction rather than two copies
that could drift apart.
"""

import pandas as pd

from src.data_processing.data_loader import NBADataLoader
from src.feature_engineering.elo import compute_elo_momentum, compute_elo_ratings
from src.feature_engineering.feature_builder import FeatureBuilder


def _fill_synthetic_row_elo(result: pd.DataFrame, all_games: pd.DataFrame, config) -> pd.DataFrame:
    """`_add_elo_features` (feature_builder.py) computes Elo by reloading
    real games from the DB and merging back onto GAME_ID -- it can never see
    our synthetic "upcoming" row (not in the DB, GAME_ID="upcoming" matches
    nothing), so its Elo/momentum columns come out NaN for exactly that row
    no matter how much real history is loaded into `create_all_features`.
    Recompute directly from `all_games` (which does include the synthetic
    row, safely -- same "last row chronologically, never affects an earlier
    rating" reasoning as every other feature here) and splice the result
    into `result` for GAME_ID=="upcoming" only; every other row/column in
    `result` is untouched."""
    elo_cfg = config.elo_features
    elo_df = compute_elo_ratings(
        all_games,
        initial_rating=elo_cfg.initial_rating,
        k_factor=elo_cfg.k_factor,
        home_advantage=elo_cfg.home_advantage,
        mov_multiplier=elo_cfg.mov_multiplier,
        season_regression=elo_cfg.season_regression,
    )
    momentum_df = compute_elo_momentum(all_games, elo_df, windows=config.features.rolling_windows)

    synthetic_elo = elo_df[elo_df["GAME_ID"] == "upcoming"].iloc[0]
    synthetic_momentum = momentum_df[momentum_df["GAME_ID"] == "upcoming"].iloc[0]

    result = result.copy()
    result["home_team_elo"] = synthetic_elo["home_team_elo"]
    result["away_team_elo"] = synthetic_elo["away_team_elo"]
    result["elo_diff"] = (
        synthetic_elo["home_team_elo"] + elo_cfg.home_advantage - synthetic_elo["away_team_elo"]
    )
    for col in momentum_df.columns:
        if col == "GAME_ID":
            continue
        result[col] = synthetic_momentum[col]
    return result


def build_live_game_features(
    home_team_id: int,
    away_team_id: int,
    game_date: str | None,
    config,
) -> pd.DataFrame:
    """Returns the single-row (possibly empty) feature DataFrame for the
    home_team_id vs away_team_id matchup on game_date (default: today).

    Loads the FULL game history from config.datasets_loading.data_start_date
    through game_date (every team, not just these two) -- the same context
    window load_training_data gives every CV/training split, not a per-team
    "last N games" slice. This matters beyond the simple rolling-window
    features: Elo (compute_elo_ratings) and season_motivation are explicitly
    "computed once over the full chronological game history" and are wrong
    (Elo comes out NaN, confirmed empirically) if seeded with a truncated
    per-team history instead.

    Injects a synthetic "upcoming" row into that history so FeatureBuilder
    computes rolling/H2H/matchup/Elo features for this exact matchup and
    date -- safe because every rolling feature uses shift(1), so the
    synthetic row's zeroed outcome never affects its own feature values.

    Caller selects the trained model's own feature_names columns from the
    result and handles an empty return (no history at all before this
    date)."""
    prediction_date = pd.Timestamp(game_date).normalize() if game_date else pd.Timestamp.now().normalize()
    end_date = str(prediction_date.date())

    loader = NBADataLoader(db_path=config.data_paths.raw_db)
    try:
        context_types = (
            config.datasets_loading.context_season_types or config.datasets_loading.allowed_season_types
        )
        history = loader.load_games(
            start_date=config.datasets_loading.data_start_date,
            end_date=end_date,
            allowed_season_types=context_types,
        )

        if history.empty:
            return history

        # Drop any real game already on prediction_date involving either of
        # these two teams -- otherwise it collides with the synthetic row on
        # the same (GAME_DATE, team_id) merge key _add_rolling_features joins
        # on, duplicating that key and corrupting the merge (surfaces as a
        # length-mismatch error deep in feature_builder). Matters for
        # backtesting a date that already has a real game in the DB, not
        # just for genuinely future dates. Other teams' real games on the
        # same date are legitimate context, left untouched.
        our_teams = {home_team_id, away_team_id}
        same_day_collision = (history["GAME_DATE"] == prediction_date) & (
            history["HOME_TEAM_ID"].isin(our_teams) | history["AWAY_TEAM_ID"].isin(our_teams)
        )
        history = history[~same_day_collision]
        if history.empty:
            return history

        current_season_id = history.iloc[-1]["SEASON_ID"]
        synthetic_row = {col: 0 for col in history.columns}
        synthetic_row.update(
            {
                "GAME_ID": "upcoming",
                "GAME_DATE": prediction_date,
                "HOME_TEAM_ID": home_team_id,
                "AWAY_TEAM_ID": away_team_id,
                "SEASON_ID": current_season_id,
                "SEASON_TYPE": "Regular Season",
            }
        )
        all_games = (
            pd.concat([history, pd.DataFrame([synthetic_row])], ignore_index=True)
            .sort_values("GAME_DATE")
            .reset_index(drop=True)
        )

        feature_builder = FeatureBuilder(
            rolling_windows=config.features.rolling_windows,
            h2h_margin_window=config.features.h2h_margin_window,
            h2h_win_rate_window=config.features.h2h_win_rate_window,
        )
        features_df = feature_builder.create_all_features(all_games, context_end_date=end_date)

        result = features_df[
            (features_df["HOME_TEAM_ID"] == home_team_id)
            & (features_df["AWAY_TEAM_ID"] == away_team_id)
            & (pd.to_datetime(features_df["GAME_DATE"]).dt.normalize() == prediction_date)
        ]
        if config.elo_features and config.elo_features.enabled and not result.empty:
            result = _fill_synthetic_row_elo(result, all_games, config)
        return result
    finally:
        loader.close()
