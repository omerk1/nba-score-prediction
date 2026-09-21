"""Aggregate/screen tests: context flags, block assignment, ratio construction
and the persistence statistics, on a hand-built possession table."""

import numpy as np
import pandas as pd
import pytest

from src.pbp.aggregates import (
    assign_blocks,
    block_aggregates,
    persistence,
    sequence_stats_per_game,
    team_view,
)

H, A = 1, 2


def _poss(**kw):
    row = dict(game_id="G1", poss_idx=0, period=1, off_team_id=H, def_team_id=A, is_home_off=1,
               start_secs=0.0, end_secs=14.0, duration=14.0, margin_start=0, margin_end=0, points=0,
               outcome="miss", n_fga=1, n_fg3a=0, n_fta=0, n_oreb=0, n_tov=0, n_fgm=0, n_fg3m=0, n_ftm=0,
               last_shot_value=2, last_shot_distance=3, last_shot_x=0, last_shot_y=0, last_shot_subtype="",
               lineup_off="1,2,3,4,5", lineup_def="6,7,8,9,10",
               lineup_off_complete=1, lineup_def_complete=1, game_date="2024-10-22")
    row.update(kw)
    return row


def test_team_view_doubles_rows_and_flips_margin():
    p = pd.DataFrame([_poss(margin_start=7, points=2)])
    tv = team_view(p)
    assert len(tv) == 2
    off = tv[tv.side == "off"].iloc[0]
    de = tv[tv.side == "def"].iloc[0]
    assert (off.team, off.opp, off.team_margin) == (H, A, 7)
    assert (de.team, de.opp, de.team_margin) == (A, H, -7)
    # points stay with the possession, not the team: the defense's row carries
    # the points it conceded, which is what makes def_rtg work.
    assert off.points == de.points == 2


@pytest.mark.parametrize("period,start_secs,margin,expect_garbage,expect_clutch", [
    (1, 0.0, 0, False, False),          # tip-off
    (4, 2200.0, 30, True, False),       # 11:20 left in regulation, +30 -> garbage
    (4, 2200.0, 20, False, False),      # same time, +20 -> under the 25-point tier
    (4, 2830.0, 12, True, False),       # 50s left, +12 -> garbage by the last tier
    (4, 2700.0, 3, False, True),        # 3:00 left, within 5 -> clutch
    (5, 2900.0, 2, False, True),        # overtime, within 5 -> clutch
    (5, 2900.0, 2, False, True),
])
def test_context_flags(period, start_secs, margin, expect_garbage, expect_clutch):
    tv = team_view(pd.DataFrame([_poss(period=period, start_secs=start_secs, margin_start=margin)]))
    assert bool(tv.garbage.iloc[0]) is expect_garbage
    assert bool(tv.clutch.iloc[0]) is expect_clutch


def test_ratings_are_pooled_ratios_not_mean_of_ratios():
    # Game 1: home scores 10 on 10 possessions. Game 2: 0 on 90 possessions.
    # Pooled off_rtg = 100*10/100 = 10, not the mean of (100, 0) = 50.
    rows = [_poss(game_id="G1", poss_idx=i, points=1) for i in range(10)]
    rows += [_poss(game_id="G2", poss_idx=i, points=0, game_date="2024-10-24") for i in range(90)]
    # Mirror possessions so each team has both sides; otherwise net rating is
    # undefined by construction rather than by the behaviour under test.
    rows += [_poss(game_id="G1", poss_idx=500 + i, off_team_id=A, def_team_id=H, is_home_off=0, points=1)
             for i in range(10)]
    rows += [_poss(game_id="G2", poss_idx=500 + i, off_team_id=A, def_team_id=H, is_home_off=0, points=0,
                   game_date="2024-10-24") for i in range(90)]
    tv = assign_blocks(team_view(pd.DataFrame(rows)), block_games=2)
    agg = block_aggregates(tv).set_index("team")
    assert agg.loc[H, "off_rtg"] == pytest.approx(10.0)
    assert agg.loc[A, "def_rtg"] == pytest.approx(10.0)
    assert agg.loc[H, "net_rtg"] == pytest.approx(-agg.loc[A, "net_rtg"])
    assert agg.loc[H, "n_games"] == 2


def test_luck_adjustment_removes_hot_three_point_shooting():
    # Home takes 100 threes and makes them all; league rate comes from this same
    # frame, so the adjustment should pull its rating back toward the unlucky version.
    rows = [_poss(poss_idx=i, n_fga=1, n_fg3a=1, n_fgm=1, n_fg3m=1, points=3) for i in range(50)]
    rows += [_poss(game_id="G2", poss_idx=i, off_team_id=A, def_team_id=H, is_home_off=0,
                   n_fga=1, n_fg3a=1, n_fgm=0, n_fg3m=0, points=0, game_date="2024-10-24") for i in range(50)]
    tv = assign_blocks(team_view(pd.DataFrame(rows)), block_games=2)
    agg = block_aggregates(tv).set_index("team")
    assert agg.loc[H, "net_rtg"] > agg.loc[H, "net_rtg_luckadj"]
    assert agg.loc[H, "own_3p_luck_per100"] > 0


def test_garbage_filter_changes_rating():
    rows = [_poss(poss_idx=i, period=4, start_secs=2800.0, margin_start=30, points=3) for i in range(20)]
    rows += [_poss(poss_idx=100 + i, period=1, start_secs=float(i), margin_start=0, points=0) for i in range(20)]
    tv = assign_blocks(team_view(pd.DataFrame(rows)), block_games=1)
    agg = block_aggregates(tv).set_index("team")
    assert agg.loc[H, "off_rtg"] == pytest.approx(150.0)
    assert agg.loc[H, "off_rtg_nogarbage"] == pytest.approx(0.0)


def test_assign_blocks_orders_by_date_and_drops_partials():
    rows = []
    for i, d in enumerate(["2024-10-22", "2024-10-24", "2024-10-26"]):
        rows.append(_poss(game_id=f"G{i}", game_date=d))
    tv = assign_blocks(team_view(pd.DataFrame(rows)), block_games=2)
    assert set(tv.block) == {0}          # the third game is a partial block, dropped
    assert tv.game_id.nunique() == 2
    tv_partial = assign_blocks(team_view(pd.DataFrame(rows)), block_games=2, drop_partial_min=1)
    assert set(tv_partial.block) == {0, 1}


def test_sequence_stats_runs_and_lead_share():
    # Home scores 8 straight, away answers with 5, home leads throughout.
    rows = []
    margin = 0
    for i in range(4):
        rows.append(_poss(poss_idx=2 * i, points=2, margin_start=margin, duration=10.0))
        margin += 2
        rows.append(_poss(poss_idx=2 * i + 1, off_team_id=A, def_team_id=H, is_home_off=0,
                          points=0, margin_start=-margin, duration=10.0))
    for i in range(3):
        rows.append(_poss(poss_idx=100 + i, off_team_id=A, def_team_id=H, is_home_off=0,
                          points=2 if i < 2 else 1, margin_start=-margin, duration=10.0))
        margin -= 2 if i < 2 else 1
    seq = sequence_stats_per_game(pd.DataFrame(rows)).set_index("team")
    assert seq.loc[H, "largest_run_for"] == 8
    assert seq.loc[H, "largest_run_against"] == 5
    assert seq.loc[A, "largest_run_for"] == 5
    assert seq.loc[H, "lead_changes"] == 0
    assert seq.loc[H, "time_leading_share"] > seq.loc[A, "time_leading_share"]


def test_persistence_recovers_known_correlations():
    rng = np.random.default_rng(7)
    rows = []
    for team in range(20):
        level = rng.normal(0, 5)
        for block in range(6):
            rows.append({"team": team, "block": block,
                         "sticky": level + rng.normal(0, 1),      # mostly team level -> high r_self
                         "noise": rng.normal(0, 5),               # pure noise -> r_self ~ 0
                         "margin_pg": level + rng.normal(0, 1)})
    res = persistence(pd.DataFrame(rows), ["sticky", "noise"]).set_index("stat")
    assert res.loc["sticky", "r_self"] > 0.8
    assert abs(res.loc["noise", "r_self"]) < 0.3
    # A stat driven by the same team level predicts next-block margin; noise does not.
    assert res.loc["sticky", "r_next_margin"] > 0.7
    assert abs(res.loc["noise", "r_next_margin"]) < 0.3
    # Demeaning within team removes the level, so the sticky stat's within-team r collapses.
    assert res.loc["sticky", "r_within"] < 0.3
    assert res.loc["sticky", "n_pairs"] == 20 * 5
