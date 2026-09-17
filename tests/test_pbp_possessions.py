"""Possession parser tests: a real game fixture (PlayByPlayV3 events for
0022400500, MIN 108 - LAC 106, 2025-01-06) plus small synthetic streams for
the segmentation edge cases the fixture does not exercise."""

import json
from pathlib import Path

import pandas as pd
import pytest

from src.pbp.collector import EVENT_COLUMNS
from src.pbp.possessions import build_possessions, parse_clock, secs_elapsed

FIXTURE = Path(__file__).parent / "fixtures" / "pbp_0022400500.json"
HOME, AWAY = 1610612750, 1610612746


@pytest.fixture(scope="module")
def fixture_events() -> pd.DataFrame:
    return pd.DataFrame(json.load(open(FIXTURE))).rename(columns=EVENT_COLUMNS)


def test_clock_parsing():
    assert parse_clock("PT11M42.00S") == 702.0
    assert parse_clock("PT00M03.50S") == 3.5
    assert secs_elapsed(1, "PT12M00.00S") == 0.0
    assert secs_elapsed(2, "PT06M00.00S") == 720 + 360
    assert secs_elapsed(5, "PT05M00.00S") == 2880.0  # start of first OT


def test_fixture_reconciles_with_box_score(fixture_events):
    poss, s = build_possessions(fixture_events)
    assert (s["home_team_id"], s["away_team_id"]) == (HOME, AWAY)
    assert s["pts_home_poss"] + s["tech_ft_pts_home"] == 108
    assert s["pts_away_poss"] + s["tech_ft_pts_away"] == 106
    assert poss.n_fga.sum() == 92 + 89          # box-score FGA
    assert poss.n_fta.sum() == 42 - 2           # box-score FTA minus the two technical FTs
    assert 85 <= s["n_poss_home"] <= 110 and 85 <= s["n_poss_away"] <= 110


def test_fixture_timing_and_lineups(fixture_events):
    poss, s = build_possessions(fixture_events)
    # Durations tile each period exactly.
    assert poss.groupby("period").duration.sum().round(2).eq(720.0).all()
    assert (poss.duration >= 0).all()
    assert (poss.start_secs.shift(-1).dropna() == poss.end_secs.iloc[:-1]).all()
    # Consecutive possessions alternate teams within a period.
    same = poss.off_team_id.eq(poss.off_team_id.shift(1)) & poss.period.eq(poss.period.shift(1))
    assert not same.any()
    # Five-man lineups resolved for every possession of this game.
    assert s["lineup_complete_rate"] == 1.0
    assert poss.lineup_off.str.count(",").eq(4).all()
    assert poss.lineup_def.str.count(",").eq(4).all()
    assert set(poss.lineup_off.iloc[0].split(",")).isdisjoint(poss.lineup_def.iloc[0].split(","))


def test_fixture_team_turnover_row_segments(fixture_events):
    # V3 logs team turnovers (shot clock) with team_id=0 and the team id in
    # person_id; the parser must still end the possession on it.
    poss, _ = build_possessions(fixture_events)
    assert poss.n_tov.sum() == 28
    assert (poss.outcome == "turnover").sum() == 28


def _ev(action_id, period, clock, team, atype, sub="", desc="", pid=0, sh="", sa="", sv=0, loc=""):
    return dict(game_id="G", action_id=action_id, action_number=action_id, period=period, clock=clock,
                team_id=team, person_id=pid, player_name="", action_type=atype, sub_type=sub,
                description=desc, shot_distance=0, shot_result="", shot_value=sv, x_legacy=0, y_legacy=0,
                score_home=sh, score_away=sa, location=loc)


def _synthetic_game():
    H, A = 1, 2
    rows = [
        _ev(1, 1, "PT12M00.00S", 0, "period", "start"),
        _ev(2, 1, "PT12M00.00S", H, "Jump Ball", loc="h"),
        # And-one: made 2 + made FT stays one possession, 3 points.
        _ev(3, 1, "PT11M40.00S", H, "Made Shot", "Layup", sh="2", sa="0", sv=2, loc="h"),
        _ev(4, 1, "PT11M40.00S", A, "Foul", "Shooting", loc="v"),
        _ev(5, 1, "PT11M40.00S", H, "Free Throw", "Free Throw 1 of 1", sh="3", sa="0", loc="h"),
        # Away: miss, dead-ball team rebound after missed FT1 must not count as oreb
        _ev(6, 1, "PT11M20.00S", A, "Missed Shot", "Jump Shot", sv=2, loc="v"),
        _ev(7, 1, "PT11M18.00S", A, "Rebound", loc="v"),                      # live oreb -> counts
        _ev(8, 1, "PT11M10.00S", H, "Foul", "Shooting", loc="h"),
        _ev(9, 1, "PT11M10.00S", A, "Free Throw", "Free Throw 1 of 2", desc="MISS X Free Throw 1 of 2", loc="v"),
        _ev(10, 1, "PT11M10.00S", A, "Rebound", loc="v"),                     # dead-ball, not oreb
        _ev(11, 1, "PT11M10.00S", A, "Free Throw", "Free Throw 2 of 2", sh="3", sa="1", loc="v"),
        # Home miss, away defensive rebound ends it.
        _ev(12, 1, "PT11M00.00S", H, "Missed Shot", "Jump Shot", sv=2, loc="h"),
        _ev(13, 1, "PT10M58.00S", A, "Rebound", loc="v"),
        # Technical FT for home during home's own possession: not a possession event.
        _ev(14, 1, "PT10M50.00S", A, "Missed Shot", "Jump Shot", sv=3, loc="v"),
        _ev(15, 1, "PT10M48.00S", H, "Rebound", loc="h"),
        _ev(16, 1, "PT10M40.00S", A, "Foul", "Technical", loc="v"),
        _ev(17, 1, "PT10M40.00S", H, "Free Throw", "Free Throw Technical", sh="4", sa="1", loc="h"),
        _ev(18, 1, "PT10M30.00S", H, "Turnover", "Bad Pass", loc="h"),
        _ev(19, 1, "PT10M20.00S", A, "Made Shot", "Jump Shot", sh="4", sa="4", sv=3, loc="v"),
        _ev(20, 1, "PT00M00.00S", 0, "period", "end"),
    ]
    return pd.DataFrame(rows), H, A


def test_synthetic_segmentation():
    ev, H, A = _synthetic_game()
    poss, s = build_possessions(ev, H, A)
    assert list(poss.off_team_id) == [H, A, H, A, H, A]
    assert list(poss.points) == [3, 1, 0, 0, 0, 3]
    assert list(poss.outcome) == ["ft", "ft", "miss", "miss", "turnover", "made_fg"]
    assert list(poss.n_oreb) == [0, 1, 0, 0, 0, 0]
    assert list(poss.n_fta) == [1, 2, 0, 0, 0, 0]
    assert s["tech_ft_pts_home"] == 1 and s["tech_ft_pts_away"] == 0
    assert s["pts_home_poss"] + s["tech_ft_pts_home"] == 4
    assert s["pts_away_poss"] == 4
    # Margins are from the offense's perspective at possession start.
    assert list(poss.margin_start) == [0, -3, 2, -2, 2, -3]
    # A defensive rebound ends the prior possession at the rebound's clock.
    assert list(poss.end_secs[:3]) == [20.0, 50.0, 62.0]
    assert poss.duration.sum() == 720.0
