import datetime
import json

import pandas as pd
import pytest
from scipy.stats import norm

from src.serving import evening_snapshot
from src.serving import morning_report as M
from src.serving import polymarket_pricing as P
from src.serving.grade_predictions import grade
from src.serving.line_snapshots import COLUMNS, append_snapshot
from src.serving.recommended_bets import polymarket_bets
from src.serving.slate_html import render_slate_html

TEAMS = {
    1: {"abbreviation": "BKN", "nickname": "Nets"},
    2: {"abbreviation": "CHA", "nickname": "Hornets"},
}


def test_spread_parse_matches_team_names_exactly():
    # 'Nets' is a substring of 'Hornets': the away team's line must not be read as the home team's.
    line, idx = P.parse_spread("Spread: Hornets (-3.5)", ["Hornets", "Nets"], "Nets")
    assert (line, idx) == (3.5, 1)
    line, idx = P.parse_spread("Spread: Nets (-2.5)", ["Nets", "Hornets"], "Nets")
    assert (line, idx) == (-2.5, 0)


def test_total_parse():
    assert P.parse_total("Hornets vs. Nets: O/U 221.5", ["Over", "Under"]) == (221.5, 0)
    assert P.parse_total("Hornets vs. Nets: O/U 221.5", ["Under", "Over"]) == (221.5, 1)


def test_fair_value_at_even_price_is_the_line_and_drops_extreme_prices():
    lines = [
        {"market": "spread", "line": -4.5, "p": 0.5, "volume": 10_000},  # home -4.5 at 50% -> home by 4.5
        {"market": "spread", "line": -12.5, "p": 0.05, "volume": 99_999},  # flat tail: ignored
        {"market": "total", "line": 220.5, "p": 0.5, "volume": 5_000},
    ]
    assert P.fair_value(lines, "spread") == (pytest.approx(4.5), 1)
    assert P.fair_value(lines, "total") == (pytest.approx(220.5), 1)
    assert P.fair_value(lines, "spread")[0] == pytest.approx(4.5)
    assert P.fair_value([], "total") == (None, 0)


def test_side_probabilities_are_complementary_and_point_the_right_way():
    # Fair home margin +4.5; Winner gives home only -3.5 -> home side better than even.
    home = P.side_probability("spread", "home", 4.5, -3.5)
    assert home == pytest.approx(norm.cdf(1 / 13.5)) and home > 0.5
    assert home + P.side_probability("spread", "away", 4.5, -3.5) == pytest.approx(1.0)
    assert P.side_probability("total", "over", 225.0, 220.5) > 0.5


def _fake_get(event):
    def get(url, params=None):
        if url == P.GAMMA_EVENTS:
            return [event] if params["slug"] == "nba-cha-bkn-2026-11-20" else []
        return {"history": [{"t": 50, "p": 0.40}, {"t": 90, "p": 0.5}, {"t": 200, "p": 0.99}]}

    return get


def _market(kind, question, outcomes, volume=10_000):
    return {
        "sportsMarketType": kind,
        "question": question,
        "outcomes": json.dumps(outcomes),
        "clobTokenIds": json.dumps([f"tok-{i}" for i in range(len(outcomes))]),
        "volume": str(volume),
    }


def test_price_pick_end_to_end_uses_the_latest_price_before_now():
    event = {
        "markets": [
            _market("spreads", "Spread: Nets (-2.5)", ["Nets", "Hornets"]),
            _market("totals", "Hornets vs. Nets: O/U 220.5", ["Over", "Under"]),
            _market("spreads", "Spread: Nets (-9.5)", ["Nets", "Hornets"], volume=10),  # too thin
            _market("moneyline", "Hornets vs. Nets", ["Hornets", "Nets"]),
        ]
    }
    pick = {
        "home_team_id": 1,
        "away_team_id": 2,
        "home_spread": -2.5,
        "home_spread_odds": 1.80,
        "away_spread_odds": 1.80,
        "total_line": 222.5,
        "over_odds": 1.75,
        "under_odds": 1.85,
    }
    out = P.price_pick(pick, "2026-11-20", ts=100, get=_fake_get(event), teams=TEAMS)
    # Every priced line sits at 0.5 at t<=100 (the 0.99 point is after "now").
    assert out["pm_fair_margin"] == pytest.approx(2.5) and out["pm_spread_lines"] == 1
    assert out["pm_fair_total"] == pytest.approx(220.5)
    assert out["home_spread_pm_probability"] == pytest.approx(0.5)
    assert out["home_spread_pm_ev"] == pytest.approx(0.5 * 1.80 - 1)
    assert out["under_pm_probability"] > 0.5 and out["under_pm_ev"] > out["over_pm_ev"]


def test_price_pick_unlisted_game_is_empty():
    pick = {"home_team_id": 2, "away_team_id": 1, "home_spread": -1.5}
    assert P.price_pick(pick, "2026-11-20", ts=100, get=_fake_get({}), teams=TEAMS) == {}


REC = {
    "home_team_id": 1,
    "away_team_id": 2,
    "home_spread": -2.5,
    "home_spread_odds": 1.8,
    "away_spread_odds": 1.8,
    "total_line": 222.5,
    "over_odds": 1.75,
    "under_odds": 1.85,
    "home_spread_pm_probability": 0.53,
    "home_spread_pm_ev": 0.53 * 1.8 - 1,  # -4.6%
    "away_spread_pm_probability": 0.47,
    "away_spread_pm_ev": 0.47 * 1.8 - 1,  # -15.4%
}


def test_polymarket_bets_keep_only_sides_above_the_bar_best_first():
    bets = polymarket_bets(REC, -0.07, {1: "Nets", 2: "Hornets"})
    assert [b["pick"] for b in bets] == ["Nets -2.5"]
    assert polymarket_bets(REC, -0.20, {})[0]["pm_ev"] > polymarket_bets(REC, -0.20, {})[1]["pm_ev"]
    assert polymarket_bets({"home_team_id": 1, "away_team_id": 2}, -1.0, {}) == []


def test_slate_leads_with_polymarket_bets_and_labels_model_picks_as_info():
    rec = {
        **REC,
        "predicted_home_score": 110,
        "predicted_away_score": 108,
        "predicted_diff": 2.0,
        "predicted_total": 218.0,
        "home_win_probability": 0.56,
        "pm_fair_margin": 3.1,
    }
    page = render_slate_html([rec], {1: "Nets", 2: "Hornets"}, "Thu Nov 20", 0.0, pm_min_ev=-0.07)
    assert page.index("Polymarket-priced") < page.index("for info, not a bet")
    assert "Nets -2.5" in page and "expected −4.6%" not in page and "expected -4.6%" in page
    assert "Polymarket: Nets by 3.1" in page
    off = render_slate_html([rec], {1: "Nets", 2: "Hornets"}, "Thu Nov 20", 0.0)
    assert "Polymarket-priced" not in off and "for info" not in off


def test_graded_sides_carry_polymarket_ev_and_the_report_tracks_them():
    day = datetime.date(2026, 11, 20)
    log = pd.DataFrame(
        [
            {
                "game_date": day.isoformat(),
                "home_team_id": 1,
                "away_team_id": 2,
                "rec_json": json.dumps(
                    {
                        **REC,
                        "predicted_diff": 1.0,
                        "predicted_total": 220.0,
                        "home_cover_probability": 0.5,
                        "away_cover_probability": 0.5,
                        "home_spread_edge": -0.1,
                        "away_spread_edge": -0.1,
                    }
                ),
            }
        ]
    )
    results = pd.DataFrame(
        [{"game_date": day, "home_team_id": 1, "away_team_id": 2, "pts_home": 110, "pts_away": 100}]
    )
    bets = grade(log, results)
    home = bets[bets["side"] == "home"].iloc[0]
    assert home["pm_ev"] == pytest.approx(REC["home_spread_pm_ev"]) and home["outcome"] == "win"

    caption, page = M.build_report(log, results, pm_min_ev=-0.07)
    assert "Polymarket-priced bets: last night 1-0 (+80.0% · +0.8u)" in caption
    assert "expected -4.6%" in caption
    assert "Last night's bets (Polymarket-priced" in page
    _, off = M.build_report(log, results)
    assert "Polymarket" not in off


def test_snapshot_log_appends_under_a_fixed_header(tmp_path):
    path = tmp_path / "snap.csv"
    now = datetime.datetime(2026, 11, 20, 20, 30, tzinfo=datetime.timezone.utc)
    row = {"game_date": "2026-11-20", "home_team_id": 1, "away_team_id": 2, **REC, "extra": "ignored"}
    assert append_snapshot([row], "afternoon", path, now) == 1
    assert append_snapshot([row], "evening", path, now) == 1
    df = pd.read_csv(path)
    assert tuple(df.columns) == COLUMNS and list(df["label"]) == ["afternoon", "evening"]
    assert df["pm_fair_margin"].isna().all() and df["home_spread_pm_ev"].iloc[0] == pytest.approx(
        REC["home_spread_pm_ev"]
    )
    path.write_text("other,header\n")
    with pytest.raises(ValueError):
        append_snapshot([row], "evening", path, now)


@pytest.mark.parametrize(
    "hhmm,expected", [((22, 30), True), ((23, 59), True), ((1, 0), True), ((1, 30), False), ((10, 0), False)]
)
def test_evening_window(hhmm, expected):
    assert evening_snapshot.in_window(datetime.datetime(2026, 11, 20, *hhmm)) is expected


def test_late_scheduled_evening_run_is_skipped_before_capturing(monkeypatch):
    def boom(**kwargs):
        raise AssertionError("must not capture outside the window")

    monkeypatch.setattr("src.serving.capture_winner.capture_nba_lines", boom)
    assert evening_snapshot.run(now=datetime.datetime(2026, 11, 21, 9, 0), scheduled=True) == 0
