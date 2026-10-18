import datetime
import json

import pandas as pd
import pytest

from src.serving import grade_predictions as G

DAY = datetime.date(2026, 10, 20)


def _rec(**extra):
    return {
        "predicted_diff": 3.0,
        "predicted_total": 230.0,
        "home_spread": -1.5,
        "home_cover_probability": 0.6,
        "away_cover_probability": 0.4,
        "home_spread_odds": 1.85,
        "away_spread_odds": 1.75,
        "home_spread_edge": 0.06,
        "away_spread_edge": -0.17,
        "total_line": 222.5,
        "over_probability": 0.55,
        "under_probability": 0.45,
        "over_odds": 1.8,
        "under_odds": 1.7,
        "over_edge": -0.005,
        "under_edge": -0.14,
        **extra,
    }


def _log(rec, home=1, away=2):
    return pd.DataFrame(
        [{"game_date": DAY.isoformat(), "home_team_id": home, "away_team_id": away, "rec_json": json.dumps(rec)}]
    )


def _results(home_pts, away_pts, day=DAY):
    return pd.DataFrame(
        [{"game_date": day, "home_team_id": 1, "away_team_id": 2, "pts_home": home_pts, "pts_away": away_pts}]
    )


def _by_side(bets):
    return {(r.market, r.side): r for r in bets.itertuples()}


def test_spread_and_total_outcomes_and_profit():
    # Home wins 115-110: margin +5 covers -1.5; total 225 is over 222.5.
    sides = _by_side(G.grade(_log(_rec()), _results(115, 110)))
    assert sides[("spread", "home")].outcome == "win"
    assert sides[("spread", "home")].profit == pytest.approx(0.85)
    assert sides[("spread", "away")].outcome == "loss" and sides[("spread", "away")].profit == -1.0
    assert sides[("total", "over")].outcome == "win"
    assert sides[("total", "under")].outcome == "loss"


def test_push_on_a_whole_number_line():
    bets = G.grade(_log(_rec(home_spread=-5.0)), _results(115, 110))
    assert _by_side(bets)[("spread", "home")].outcome == "push"
    assert _by_side(bets)[("spread", "home")].profit == 0.0


def test_unplayed_game_is_skipped_and_result_window_is_respected():
    assert G.grade(_log(_rec()), _results(115, 110).iloc[0:0]).empty
    late = DAY + datetime.timedelta(days=G.RESULT_WINDOW_DAYS + 1)
    assert G.grade(_log(_rec()), _results(115, 110, day=late)).empty


def test_old_rows_without_odds_use_the_implied_probability():
    rec = _rec()
    del rec["home_spread_odds"]
    rec["home_spread_market_probability"] = 0.5
    assert _by_side(G.grade(_log(rec), _results(115, 110)))[("spread", "home")].odds == pytest.approx(2.0)


def test_summarize_buckets_hit_rate_and_roi():
    bets = pd.DataFrame(
        {
            "edge": [0.07, 0.08, -0.02],
            "model_prob": [0.6, 0.6, 0.4],
            "outcome": ["win", "loss", "win"],
            "profit": [0.85, -1.0, 0.75],
        }
    )
    table = G.summarize(bets)
    top = table.loc[table["bets"] == 2].iloc[0]
    assert top["hit_rate"] == 0.5 and top["roi"] == pytest.approx(-0.075)
