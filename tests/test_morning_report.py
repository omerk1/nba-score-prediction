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
    assert "model picks: —" in caption and "No model picks" in page


def test_report_with_nothing_graded():
    assert M.build_report(_log(REC), _results(100, 95).iloc[0:0]) is None


def test_report_shows_whether_the_model_beats_the_line():
    caption, page = M.build_report(_log(REC), _results(100, 95))
    assert "Is the model beating the line?" in page
    assert "needs 50 games" in page and "off (0%)" in page
    assert "Recommended on the slate" not in page  # same as the model picks when calibration is off
    assert "Disagreement with the line that came true: spread —, total —" in caption


def test_model_picks_are_tracked_on_raw_edge_even_when_the_slate_priced_them_away():
    # Calibrated edge below the bar (nothing on the slate), raw edge above it.
    rec = {**REC, "under_edge": -0.05, "under_edge_raw": 0.08}
    caption, page = M.build_report(_log(rec), _results(100, 95), calibration_on=True)
    assert "model picks 1-0" in caption
    assert "Recommended on the slate (calibrated edge ≥ 5%)</td><td class=\"n\">— · no bets" in page
