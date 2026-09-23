"""Shot-quality tests: opponent attribution, the league location baseline,
offense/defense aggregation and the location-adjusted rating."""

import numpy as np
import pandas as pd
import pytest

from src.pbp.shots import (
    MIN_SHOTS,
    fit_xfg,
    load_shots,
    location_adjusted_net_rating,
    shot_quality_blocks,
)

H, A = 1, 2


class _FakeConn:
    """Stands in for a sqlite connection: load_shots issues one parameterised
    read_sql_query, so returning the frame directly keeps the test off disk."""

    def __init__(self, df):
        self.df = df

    def cursor(self):  # pragma: no cover - pandas probes for this
        raise NotImplementedError


def _events(rows):
    cols = ["game_id", "action_id", "period", "team_id", "action_type", "shot_distance",
            "shot_value", "x_legacy", "y_legacy", "sub_type"]
    return pd.DataFrame(rows, columns=cols)


def _shot(game_id, action_id, team, made, distance, value=2):
    return [game_id, action_id, 1, team, "Made Shot" if made else "Missed Shot",
            distance, value, 0, 0, ""]


@pytest.fixture
def shots(monkeypatch):
    rows = [_shot("G1", i, H, i % 2 == 0, 2) for i in range(10)]          # home: 5/10 at the rim
    rows += [_shot("G1", 100 + i, A, i < 2, 25, 3) for i in range(10)]    # away: 2/10 from three
    ev = _events(rows)
    monkeypatch.setattr(pd, "read_sql_query", lambda *a, **k: ev.copy())
    return load_shots(_FakeConn(ev))


def test_load_shots_assigns_opponent_and_bins(shots):
    assert len(shots) == 20
    assert set(shots[shots.team_id == H].opp_id) == {A}
    assert set(shots[shots.team_id == A].opp_id) == {H}
    assert shots[shots.team_id == H].made.sum() == 5
    assert shots[shots.team_id == H].points.sum() == 10
    assert shots[shots.team_id == A].points.sum() == 6
    # Rim and three-point attempts land in different distance bins.
    assert shots[shots.team_id == H].dist_bin.nunique() == 1
    assert shots[shots.team_id == H].dist_bin.iloc[0] != shots[shots.team_id == A].dist_bin.iloc[0]


def test_fit_xfg_is_league_rate_per_location(shots):
    xfg = fit_xfg(shots)
    two = shots[shots.shot_value == 2]
    three = shots[shots.shot_value == 3]
    assert xfg.loc[(2, two.dist_bin.iloc[0])] == pytest.approx(0.5)
    assert xfg.loc[(3, three.dist_bin.iloc[0])] == pytest.approx(0.2)


def test_shot_quality_splits_offense_and_defense(shots):
    block_map = pd.DataFrame({"team": [H, A], "game_id": ["G1", "G1"], "block": [0, 0]})
    sq = shot_quality_blocks(shots, block_map)
    # Only one game, so each team is under MIN_SHOTS and every ratio is masked.
    assert sq.xpps.isna().all()

    big = pd.concat([shots.assign(game_id=f"G{i}", action_id=shots.action_id + 1000 * i)
                     for i in range(MIN_SHOTS // 5)], ignore_index=True)
    bm = pd.DataFrame([{"team": t, "game_id": g, "block": 0}
                       for t in (H, A) for g in big.game_id.unique()])
    sq = shot_quality_blocks(big, bm)
    # Home shoots twos at the league rate for twos: actual equals expected.
    assert sq.loc[(H, 0), "pps"] == pytest.approx(1.0)
    assert sq.loc[(H, 0), "xpps"] == pytest.approx(1.0)
    assert sq.loc[(H, 0), "pps_vs_x"] == pytest.approx(0.0, abs=1e-9)
    # Home's defensive row describes the shots AWAY took against it.
    assert sq.loc[(H, 0), "opp_pps"] == pytest.approx(sq.loc[(A, 0), "pps"])
    assert sq.loc[(H, 0), "opp_xpps"] == pytest.approx(sq.loc[(A, 0), "xpps"])


def test_location_adjusted_rating_weights_own_luck():
    blocks = pd.DataFrame({
        "net_rtg_nogarbage": [0.0],
        "poss_off": [100.0], "poss_def": [100.0],
        "fg_pts": [110.0], "fg_xpts": [100.0],      # team made 10 points more than locations imply
        "opp_fg_pts": [100.0], "opp_fg_xpts": [100.0],
    })
    assert location_adjusted_net_rating(blocks, own_weight=1.0).iloc[0] == pytest.approx(-10.0)
    assert location_adjusted_net_rating(blocks, own_weight=0.5).iloc[0] == pytest.approx(-5.0)
    assert location_adjusted_net_rating(blocks, own_weight=0.0).iloc[0] == pytest.approx(0.0)

    # Opponents overshooting their locations is credited back to the team in full.
    lucky_opp = blocks.assign(fg_pts=100.0, opp_fg_pts=110.0)
    assert location_adjusted_net_rating(lucky_opp).iloc[0] == pytest.approx(10.0)


def test_points_per_shot_is_twice_efg():
    # PPS = (2*FGM + FG3M)/FGA = 2*(FGM + 0.5*FG3M)/FGA = 2*eFG, so the two
    # pipelines (raw events vs possession-table counts) must agree exactly.
    fgm, fg3m, fga = 40, 12, 90
    pps = (2 * fgm + fg3m) / fga
    efg = (fgm + 0.5 * fg3m) / fga
    assert pps == pytest.approx(2 * efg)
