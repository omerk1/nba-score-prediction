import datetime
import json

import pandas as pd

from src.serving import morning_report as M

DAY = datetime.date(2026, 10, 20)


def _log(rec):
    return pd.DataFrame(
        [{"game_date": DAY.isoformat(), "home_team_id": 1610612737, "away_team_id": 1610612738, "rec_json": json.dumps(rec)}]
    )


REC = {
    "predicted_diff": 3.0,
    "predicted_total": 230.0,
    "total_line": 222.5,
    "over_probability": 0.4,
    "under_probability": 0.6,
    "over_odds": 1.8,
    "under_odds": 1.8,
    "over_edge": -0.1,
    "under_edge": 0.08,
}


def _results(home, away):
    return pd.DataFrame(
        [{"game_date": DAY, "home_team_id": 1610612737, "away_team_id": 1610612738, "pts_home": home, "pts_away": away}]
    )


def test_report_counts_recommended_win_and_tiers():
    caption, page = M.build_report(_log(REC), _results(100, 95))  # total 195 < 222.5: under wins
    assert "1-0 (+80.0% · +0.8u)" in caption
    assert "+3.0 → +5" in page and "195" in page and "230" in page and "Marginal" not in page


def test_marginal_edge_is_not_tracked():
    rec = {**REC, "under_edge": 0.04}
    caption, page = M.build_report(_log(rec), _results(100, 95))
    assert "strong bets: —" in caption and "No strong bets" in page


def test_report_with_nothing_graded():
    assert M.build_report(_log(REC), _results(100, 95).iloc[0:0]) is None
