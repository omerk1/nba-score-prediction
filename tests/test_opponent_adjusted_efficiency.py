"""
Regression test for `compute_team_offense_defense_history`/
`compute_opponent_adjusted_efficiency_scores` (src/feature_engineering/
season_motivation.py): the retrospective opponent-adjustment feature from
docs/NEXT_PHASE_SESSIONS.md backlog item 5, a direct extension of
`compute_opponent_adjusted_form_scores`' own template (per-game signed
residual vs. opponent's pre-game quality, then a shift(1) rolling mean of
that residual) from win/loss to points scored/allowed.

Tests operate on small synthetic `games_df` frames (hand-picked PTS_home/
PTS_away sequences) so expected values are exact, hand-verifiable numbers.
"""

from datetime import datetime, timedelta

import pandas as pd
import pytest

from src.feature_engineering.season_motivation import (
    compute_opponent_adjusted_efficiency_scores,
    compute_team_offense_defense_history,
)

TEAM_A = "A"
TEAM_B = "B"
TEAM_C = "C"


def _games(rows, start_date=datetime(2020, 1, 1)):
    """rows: list of (game_id, home_id, away_id, pts_home, pts_away). Dates are
    assigned in list order, 2 days apart, so sort order is unambiguous."""
    date = start_date
    out = []
    for game_id, home_id, away_id, pts_home, pts_away in rows:
        out.append(
            {
                "GAME_ID": game_id,
                "GAME_DATE": date,
                "HOME_TEAM_ID": home_id,
                "AWAY_TEAM_ID": away_id,
                "PTS_home": pts_home,
                "PTS_away": pts_away,
            }
        )
        date += timedelta(days=2)
    return pd.DataFrame(out)


class TestOpponentAdjustedEfficiencyHistory:
    def test_first_game_has_no_prior_quality_and_is_nan(self):
        """A team's (or its opponent's) very first game has no prior games to
        average -- off_eff_before/def_eff_before, and therefore the
        signed residual, must be NaN, not a fabricated neutral default (which
        would bias early-season rows toward whatever that default happened to
        be, the same concern win_pct_before's 0.5 default doesn't have since
        win% has a natural neutral point but a points total does not)."""
        games_df = _games(
            [
                ("g1", TEAM_A, TEAM_B, 100, 95),
                ("g2", TEAM_A, TEAM_C, 105, 90),
            ]
        )
        hist = compute_team_offense_defense_history(games_df, window=10)

        g1_a = hist[(hist["game_id"] == "g1") & (hist["team_id"] == TEAM_A)].iloc[0]
        assert pd.isna(g1_a["off_eff_before"])
        assert pd.isna(g1_a["def_eff_before"])
        assert pd.isna(g1_a["signed_offensive_adjusted_score"])
        assert pd.isna(g1_a["signed_defensive_adjusted_score"])

        # g2's opponent is C, also on its first-ever game here -- opponent
        # quality is undefined too, so A's g2 residual must also be NaN even
        # though A itself has one prior game.
        g2_a = hist[(hist["game_id"] == "g2") & (hist["team_id"] == TEAM_A)].iloc[0]
        assert g2_a["off_eff_before"] == pytest.approx(100.0)
        assert pd.isna(g2_a["opponent_def_quality"])
        assert pd.isna(g2_a["signed_offensive_adjusted_score"])

    def test_residual_exact_value(self):
        """Hand-verified scenario: A plays B, C, B, C. At A's 3rd game (g3, vs
        B, A scores 110/allows 100), window=10 (>= A's 2 prior games, so the
        windowed and a cumulative average coincide here): A's off_eff_before =
        mean(100,105) = 102.5, def_eff_before = mean(95,90) = 92.5. B's own
        history going into g3 (B's 2nd game): off_eff_before=95 (from g1),
        def_eff_before=100 (from g1) -- these become A's
        opponent_off_quality/opponent_def_quality. signed_offensive_adjusted_score
        = 110 - 100 = 10 (A beat what B typically allows).
        signed_defensive_adjusted_score = 95 - 100 = -5 (A allowed 5 more than
        what B typically scores -- below-average defense against B's own
        tendency)."""
        games_df = _games(
            [
                ("g1", TEAM_A, TEAM_B, 100, 95),
                ("g2", TEAM_C, TEAM_A, 90, 105),
                ("g3", TEAM_A, TEAM_B, 110, 100),
                ("g4", TEAM_C, TEAM_A, 95, 108),
            ]
        )
        hist = compute_team_offense_defense_history(games_df, window=10)

        g3_a = hist[(hist["game_id"] == "g3") & (hist["team_id"] == TEAM_A)].iloc[0]
        assert g3_a["off_eff_before"] == pytest.approx(102.5)
        assert g3_a["def_eff_before"] == pytest.approx(92.5)
        assert g3_a["opponent_off_quality"] == pytest.approx(95.0)
        assert g3_a["opponent_def_quality"] == pytest.approx(100.0)
        assert g3_a["signed_offensive_adjusted_score"] == pytest.approx(10.0)
        assert g3_a["signed_defensive_adjusted_score"] == pytest.approx(-5.0)

    def test_window_actually_windows_not_cumulative(self):
        """Regression test for the fold3-diagnosis fix itself: with a SHORT
        window, a team's 3rd-game quality estimate must reflect only its last
        `window` games, not its full history -- proves this is genuinely a
        rolling window now, not a cumulative average with a new column name.
        A plays 3 games (vs B, C, B), each a different, deliberately
        increasing PTS_home so a cumulative mean and a window=1 mean diverge.
        At A's 4th game (g4, vs C), window=1 -> off_eff_before must equal
        exactly g3's own score (130), not mean(100,110,130)=113.33."""
        games_df = _games(
            [
                ("g1", TEAM_A, TEAM_B, 100, 50),
                ("g2", TEAM_A, TEAM_C, 110, 50),
                ("g3", TEAM_A, TEAM_B, 130, 50),
                ("g4", TEAM_A, TEAM_C, 140, 50),
            ]
        )
        hist_w1 = compute_team_offense_defense_history(games_df, window=1)
        g4_a_w1 = hist_w1[(hist_w1["game_id"] == "g4") & (hist_w1["team_id"] == TEAM_A)].iloc[0]
        assert g4_a_w1["off_eff_before"] == pytest.approx(130.0)

        hist_cum_equivalent = compute_team_offense_defense_history(games_df, window=10)
        g4_a_wide = hist_cum_equivalent[
            (hist_cum_equivalent["game_id"] == "g4") & (hist_cum_equivalent["team_id"] == TEAM_A)
        ].iloc[0]
        assert g4_a_wide["off_eff_before"] == pytest.approx((100 + 110 + 130) / 3)
        assert g4_a_w1["off_eff_before"] != pytest.approx(g4_a_wide["off_eff_before"])


class TestOpponentAdjustedEfficiencyScores:
    def test_rolling_mean_uses_only_strictly_prior_residuals(self):
        """A's g4 (4th overall game) should have a rolling off/def score equal
        to exactly g3's own residual (10.0 / -5.0) -- g1/g2 have NaN residuals
        (no opponent quality yet) and must be excluded from the rolling mean,
        not averaged in as if they were zero."""
        games_df = _games(
            [
                ("g1", TEAM_A, TEAM_B, 100, 95),
                ("g2", TEAM_C, TEAM_A, 90, 105),
                ("g3", TEAM_A, TEAM_B, 110, 100),
                ("g4", TEAM_C, TEAM_A, 95, 108),
            ]
        )
        hist = compute_team_offense_defense_history(games_df, window=10)
        scores = compute_opponent_adjusted_efficiency_scores(hist, window=10)

        g4_a = scores[
            (scores["team_id"] == TEAM_A) & (scores["game_date"] == pd.Timestamp("2020-01-07"))
        ].iloc[0]
        assert g4_a["opponent_adjusted_off_score"] == pytest.approx(10.0)
        assert g4_a["opponent_adjusted_def_score"] == pytest.approx(-5.0)

    def test_no_leakage_perturbing_a_later_game_does_not_change_earlier_rows(self):
        """Directly checks the leakage-safety claim: the rolling score at game
        k must be computable from residuals at positions < k only. Perturb
        only the LAST game's score and confirm every earlier row's
        opponent-adjusted score is unchanged."""
        base_rows = [
            ("g1", TEAM_A, TEAM_B, 100, 95),
            ("g2", TEAM_C, TEAM_A, 90, 105),
            ("g3", TEAM_A, TEAM_B, 110, 100),
            ("g4", TEAM_C, TEAM_A, 95, 108),
            ("g5", TEAM_A, TEAM_B, 102, 99),
        ]
        games_base = _games(base_rows)
        hist_base = compute_team_offense_defense_history(games_base, window=10)
        scores_base = compute_opponent_adjusted_efficiency_scores(hist_base, window=10)

        perturbed_rows = base_rows[:-1] + [("g5", TEAM_A, TEAM_B, 999, 1)]
        games_pert = _games(perturbed_rows)
        hist_pert = compute_team_offense_defense_history(games_pert, window=10)
        scores_pert = compute_opponent_adjusted_efficiency_scores(hist_pert, window=10)

        for gid, team_id in [("g1", TEAM_A), ("g2", TEAM_C), ("g3", TEAM_A), ("g4", TEAM_C)]:
            date = games_base.loc[games_base["GAME_ID"] == gid, "GAME_DATE"].iloc[0]
            row_base = scores_base[
                (scores_base["team_id"] == team_id) & (scores_base["game_date"] == date)
            ].iloc[0]
            row_pert = scores_pert[
                (scores_pert["team_id"] == team_id) & (scores_pert["game_date"] == date)
            ].iloc[0]
            assert row_base["opponent_adjusted_off_score"] == pytest.approx(
                row_pert["opponent_adjusted_off_score"]
            ), f"{gid}/{team_id} off score changed when a LATER game's result changed"
            assert row_base["opponent_adjusted_def_score"] == pytest.approx(
                row_pert["opponent_adjusted_def_score"]
            ), f"{gid}/{team_id} def score changed when a LATER game's result changed"

    def test_first_game_ever_defaults_to_zero_not_nan(self):
        """A team's first-ever game has no prior residuals at all to average --
        matches compute_opponent_adjusted_form_scores' own convention of
        filling with a neutral 0.0 rather than leaving the model-facing
        feature NaN."""
        games_df = _games([("g1", TEAM_A, TEAM_B, 100, 95)])
        hist = compute_team_offense_defense_history(games_df, window=10)
        scores = compute_opponent_adjusted_efficiency_scores(hist, window=10)

        row = scores[scores["team_id"] == TEAM_A].iloc[0]
        assert row["opponent_adjusted_off_score"] == 0.0
        assert row["opponent_adjusted_def_score"] == 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
