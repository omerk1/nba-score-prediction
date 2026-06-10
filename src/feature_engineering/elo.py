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
    games = games_df.sort_values('GAME_DATE').reset_index(drop=True)

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
                ratings[team_id] = initial_rating + (1 - season_regression) * (ratings[team_id] - initial_rating)
                last_season[team_id] = season

        r_home = ratings[home_id]
        r_away = ratings[away_id]
        home_elo[i] = r_home
        away_elo[i] = r_away

        expected_home = 1.0 / (1.0 + 10 ** (-(r_home + home_advantage - r_away) / 400.0))
        actual_home = 1.0 if row.POINT_DIFF > 0 else 0.0

        if mov_multiplier:
            margin = abs(row.POINT_DIFF)
            elo_diff_winner = (r_home + home_advantage - r_away) if actual_home == 1.0 else (r_away - (r_home + home_advantage))
            mult = np.log(margin + 1) * (2.2 / (elo_diff_winner * 0.001 + 2.2))
        else:
            mult = 1.0

        delta = k_factor * mult * (actual_home - expected_home)
        ratings[home_id] = r_home + delta
        ratings[away_id] = r_away - delta

    return pd.DataFrame({
        'GAME_ID': games['GAME_ID'].values,
        'home_team_elo': home_elo,
        'away_team_elo': away_elo,
    })
