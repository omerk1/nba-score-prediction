"""
Elo Rating Computation
=======================

Point-in-time Elo ratings for NBA teams, following the 538 NBA Elo formula
(margin-of-victory adjustment + season regression to the mean).

Ratings are computed once over the full chronological game history so that
val/test games carry forward accumulated team strength rather than
restarting at the initial rating each split.
"""

import numpy as np
import pandas as pd


def compute_elo_ratings(
    games_df: pd.DataFrame,
    initial_rating: float,
    k_factor: float,
    home_advantage: float,
    mov_multiplier: bool,
    season_regression: float,
) -> pd.DataFrame:
    """
    Compute pre-game Elo ratings for the home and away team of every game.

    Args:
        games_df: must contain GAME_ID, GAME_DATE, SEASON_ID, HOME_TEAM_ID,
            AWAY_TEAM_ID, POINT_DIFF, sorted or not (will be sorted here).
        initial_rating: starting rating for a team's first-ever game.
        k_factor: base rating-change magnitude per game.
        home_advantage: rating-point bonus for the home team when computing
            the expected-score (win probability).
        mov_multiplier: if True, scale rating changes by margin of victory
            (538 formula), dampened when the winner was already favored.
        season_regression: fraction a team's rating is reverted toward
            initial_rating at the start of each new season (0 = no
            regression, 1 = full reset).

    Returns:
        DataFrame with one row per game: GAME_ID, home_team_elo, away_team_elo
        — both are the teams' ratings BEFORE this game (no leakage).
    """
    games = games_df.sort_values("GAME_DATE").reset_index(drop=True)

    ratings: dict[int, float] = {}
    last_season: dict[int, int] = {}

    home_elo = np.empty(len(games))
    away_elo = np.empty(len(games))

    for i, row in enumerate(games.itertuples(index=False)):
        home_id = row.HOME_TEAM_ID
        away_id = row.AWAY_TEAM_ID
        season = row.SEASON_ID

        for team_id in (home_id, away_id):
            if team_id not in ratings:
                ratings[team_id] = initial_rating
                last_season[team_id] = season
            elif last_season[team_id] != season:
                ratings[team_id] = initial_rating + (1 - season_regression) * (
                    ratings[team_id] - initial_rating
                )
                last_season[team_id] = season

        r_home = ratings[home_id]
        r_away = ratings[away_id]
        home_elo[i] = r_home
        away_elo[i] = r_away

        expected_home = 1.0 / (1.0 + 10 ** (-(r_home + home_advantage - r_away) / 400.0))
        actual_home = 1.0 if row.POINT_DIFF > 0 else 0.0

        if mov_multiplier:
            margin = abs(row.POINT_DIFF)
            elo_diff_winner = (
                (r_home + home_advantage - r_away)
                if actual_home == 1.0
                else (r_away - (r_home + home_advantage))
            )
            mult = np.log(margin + 1) * (2.2 / (elo_diff_winner * 0.001 + 2.2))
        else:
            mult = 1.0

        delta = k_factor * mult * (actual_home - expected_home)
        ratings[home_id] = r_home + delta
        ratings[away_id] = r_away - delta

    return pd.DataFrame(
        {
            "GAME_ID": games["GAME_ID"].values,
            "home_team_elo": home_elo,
            "away_team_elo": away_elo,
        }
    )


def compute_elo_momentum(
    games_df: pd.DataFrame,
    elo_ratings: pd.DataFrame,
    windows: list[int],
) -> pd.DataFrame:
    """
    Per-team Elo rate-of-change ("momentum"): net rating change over a team's
    last `w` games, for each `w` in `windows`.

    Point-in-time safety note (this is NOT the same shift(1)-before-rolling
    pattern used for raw per-game box-score stats elsewhere in this codebase):
    `elo_ratings`'s home_team_elo/away_team_elo (compute_elo_ratings' output)
    are already each team's rating BEFORE that specific game -- by
    construction, not by convention. So for a team's chronological rating
    sequence pre_game_elo[1..n], pre_game_elo[k] - pre_game_elo[k-w] (the
    momentum at game k) only ever differences two values that are each
    already "known strictly before game k" -- no additional shift is needed
    or correct here. A team with fewer than `w` prior games has no shift(w)
    value to difference against, so momentum is NaN until it does (unlike a
    rolling mean's min_periods=1 graceful degradation, momentum's fixed
    w-game lookback can't shrink its window and still measure the same
    thing).

    Args:
        games_df: must contain GAME_ID, GAME_DATE, HOME_TEAM_ID, AWAY_TEAM_ID
            -- same rows `elo_ratings` was computed from.
        elo_ratings: `compute_elo_ratings`'s output (GAME_ID, home_team_elo,
            away_team_elo).
        windows: team-game window lengths, e.g. [5, 10, 20].

    Returns:
        One row per GAME_ID: home_team_elo_momentum_L{w}, away_team_elo_momentum_L{w},
        elo_momentum_diff_L{w} (home - away), for each w in windows.
    """
    merged = games_df[["GAME_ID", "GAME_DATE", "HOME_TEAM_ID", "AWAY_TEAM_ID"]].merge(
        elo_ratings[["GAME_ID", "home_team_elo", "away_team_elo"]], on="GAME_ID"
    )

    home_rows = pd.DataFrame(
        {
            "GAME_DATE": merged["GAME_DATE"].values,
            "team_id": merged["HOME_TEAM_ID"].values,
            "pre_game_elo": merged["home_team_elo"].values,
        }
    )
    away_rows = pd.DataFrame(
        {
            "GAME_DATE": merged["GAME_DATE"].values,
            "team_id": merged["AWAY_TEAM_ID"].values,
            "pre_game_elo": merged["away_team_elo"].values,
        }
    )
    long_df = pd.concat([home_rows, away_rows]).sort_values(["team_id", "GAME_DATE"]).reset_index(drop=True)

    momentum_cols = []
    for w in windows:
        col = f"elo_momentum_L{w}"
        long_df[col] = long_df.groupby("team_id")["pre_game_elo"].transform(lambda x, w=w: x - x.shift(w))
        momentum_cols.append(col)

    out = merged[["GAME_ID"]].copy()
    for team_col, prefix in [("HOME_TEAM_ID", "home_team"), ("AWAY_TEAM_ID", "away_team")]:
        query = merged[["GAME_DATE", team_col]].rename(columns={team_col: "team_id"})
        m = query.merge(
            long_df[["GAME_DATE", "team_id"] + momentum_cols], on=["GAME_DATE", "team_id"], how="left"
        )
        for col in momentum_cols:
            out[f"{prefix}_{col}"] = m[col].values

    for w in windows:
        out[f"elo_momentum_diff_L{w}"] = (
            out[f"home_team_elo_momentum_L{w}"] - out[f"away_team_elo_momentum_L{w}"]
        )

    return out
