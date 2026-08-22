"""
Regression test for `compute_elo_momentum` (src/feature_engineering/elo.py):
Elo's rate-of-change feature, added per docs/FEATURE_REPRESENTATION_AUDIT.md
priority #3 (elo previously had zero volatility/rate-of-change representation,
only the point-in-time running rating).

Tests operate directly on a synthetic `elo_ratings` DataFrame (hand-picked
pre_game_elo values, bypassing compute_elo_ratings entirely) so the expected
momentum values are exact, known numbers rather than something that itself
needs computing via the Elo formula.
"""

from datetime import datetime, timedelta

import pandas as pd
import pytest

from src.feature_engineering.elo import compute_elo_momentum

TEAM_A = 100
TEAM_B = 200
TEAM_C = 300


def _games_and_ratings(team_ratings_sequence, opponent_id, start_date=datetime(2024, 1, 1)):
    """team_ratings_sequence: list of pre_game_elo values for TEAM_A across its
    consecutive games, all played at home vs a distinct opponent each time (so a
    real compute_elo_ratings call would never need to disambiguate -- irrelevant
    here since compute_elo_momentum is given `elo_ratings` directly)."""
    games = []
    ratings_rows = []
    date = start_date
    for i, elo in enumerate(team_ratings_sequence):
        game_id = f"g{i}"
        games.append(
            {
                "GAME_ID": game_id,
                "GAME_DATE": date,
                "HOME_TEAM_ID": TEAM_A,
                "AWAY_TEAM_ID": opponent_id,
            }
        )
        ratings_rows.append({"GAME_ID": game_id, "home_team_elo": elo, "away_team_elo": 1500.0})
        date += timedelta(days=2)
    return pd.DataFrame(games), pd.DataFrame(ratings_rows)


class TestEloMomentum:
    def test_momentum_value_is_exact_difference_from_w_games_back(self):
        """TEAM_A's pre-game ratings across 6 games: 1500, 1510, 1495, 1520, 1540, 1530.
        At game index 5 (0-indexed, the 6th game, rating 1530), momentum_L3 should be
        1530 - <rating at game index 2> = 1530 - 1495 = 35."""
        ratings_seq = [1500.0, 1510.0, 1495.0, 1520.0, 1540.0, 1530.0]
        games_df, elo_ratings = _games_and_ratings(ratings_seq, TEAM_B)

        result = compute_elo_momentum(games_df, elo_ratings, windows=[3])

        row = result.loc[result["GAME_ID"] == "g5"]
        assert row["home_team_elo_momentum_L3"].iloc[0] == pytest.approx(1530.0 - 1495.0)
        # away side (TEAM_B, flat 1500 rating every game) should have zero momentum.
        assert row["away_team_elo_momentum_L3"].iloc[0] == pytest.approx(0.0)
        assert row["elo_momentum_diff_L3"].iloc[0] == pytest.approx(35.0)

    def test_no_leakage_fewer_than_w_prior_games_is_nan(self):
        """A team with only 2 prior games can't have a real L3 momentum value --
        must be NaN, not silently computed from a shorter window (that would be a
        different statistic pretending to be the same one)."""
        ratings_seq = [1500.0, 1510.0, 1495.0]
        games_df, elo_ratings = _games_and_ratings(ratings_seq, TEAM_B)

        result = compute_elo_momentum(games_df, elo_ratings, windows=[3])

        # g0, g1, g2 (indices 0,1,2) each have fewer than 3 PRIOR games (0, 1, 2
        # respectively) -- none can compute a true L3 momentum.
        for gid in ["g0", "g1", "g2"]:
            row = result.loc[result["GAME_ID"] == gid]
            assert pd.isna(row["home_team_elo_momentum_L3"].iloc[0]), f"{gid} should be NaN"

    def test_exactly_w_prior_games_gives_a_real_value(self):
        """The (w+1)th game (0-indexed position w) has exactly w prior games, so
        momentum_L{w} should be a real, non-NaN value there -- the boundary case
        right after the all-NaN prefix."""
        ratings_seq = [1500.0, 1510.0, 1495.0, 1520.0]  # 4 games, w=3
        games_df, elo_ratings = _games_and_ratings(ratings_seq, TEAM_B)

        result = compute_elo_momentum(games_df, elo_ratings, windows=[3])

        row = result.loc[result["GAME_ID"] == "g3"]
        assert row["home_team_elo_momentum_L3"].iloc[0] == pytest.approx(1520.0 - 1500.0)

    def test_only_uses_games_strictly_before_the_row_being_featurized(self):
        """Directly checks the leakage-safety claim in compute_elo_momentum's
        docstring: momentum at game k must be computable from pre_game_elo values
        at positions < k only. Verified here by confirming that changing what
        WOULD have been the rating at a later, not-yet-reached game (i.e. a value
        the function's own row-by-row logic could only see by looking ahead)
        never appears in an earlier row's momentum -- perturb the last rating in
        the sequence and confirm every earlier row's momentum is unchanged."""
        base_seq = [1500.0, 1510.0, 1495.0, 1520.0, 1540.0]
        games_df, elo_ratings = _games_and_ratings(base_seq, TEAM_B)
        result_base = compute_elo_momentum(games_df, elo_ratings, windows=[2])

        perturbed_seq = base_seq[:-1] + [1540.0 + 999.0]  # only the LAST rating changes
        games_df2, elo_ratings2 = _games_and_ratings(perturbed_seq, TEAM_B)
        result_perturbed = compute_elo_momentum(games_df2, elo_ratings2, windows=[2])

        for gid in ["g0", "g1", "g2", "g3"]:  # every row before the perturbed one
            v_base = result_base.loc[result_base["GAME_ID"] == gid, "home_team_elo_momentum_L2"].iloc[0]
            v_pert = result_perturbed.loc[
                result_perturbed["GAME_ID"] == gid, "home_team_elo_momentum_L2"
            ].iloc[0]
            if pd.isna(v_base):
                assert pd.isna(v_pert)
            else:
                assert v_base == pytest.approx(v_pert), f"{gid} changed when a LATER game's rating changed"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
