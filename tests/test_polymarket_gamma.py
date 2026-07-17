"""
Core-logic tests for src/polymarket_prices/gamma.py: market classification
(picking the highest-volume line per market type) and question-string parsing.
"""

import json

import pandas as pd
import pytest

from src.polymarket_prices.gamma import (
    classify_event_markets,
    parse_game_start_time,
    parse_spread_line,
    parse_totals_line,
)


def _market(market_type, volume, outcomes=("Yes", "No"), prices=(1.0, 0.0),
            question="", game_start_time=None, start_date=None):
    return {
        "conditionId": f"cond-{market_type}-{volume}",
        "slug": f"slug-{market_type}-{volume}",
        "question": question,
        "sportsMarketType": market_type,
        "outcomes": json.dumps(list(outcomes)),
        "outcomePrices": json.dumps([str(p) for p in prices]),
        "clobTokenIds": json.dumps(["tok-a", "tok-b"]),
        "volume": volume,
        "gameStartTime": game_start_time,
        "startDate": start_date,
        "endDate": None,
    }


_DEFAULT_TEAMS = [
    {"ordering": "home", "name": "Celtics"},
    {"ordering": "away", "name": "Lakers"},
]


def _event(markets, slug="nba-lal-bos-2026-01-15", teams=_DEFAULT_TEAMS):
    return {
        "slug": slug,
        "id": "evt-1",
        "teams": teams,
        "markets": markets,
    }


class TestClassifyEventMarkets:
    def test_picks_highest_volume_moneyline(self):
        event = _event([_market("moneyline", 1000), _market("moneyline", 5000)])
        gm = classify_event_markets(event)
        assert gm.moneyline.volume == 5000
        assert any("multiple_moneyline_markets" in f for f in gm.flags)

    def test_picks_highest_volume_spread_line_among_many(self):
        event = _event([
            _market("spreads", 100, question="Spread: Lakers (-3.5)"),
            _market("spreads", 9000, question="Spread: Lakers (-5.5)"),
            _market("spreads", 50, question="Spread: Lakers (-7.5)"),
        ])
        gm = classify_event_markets(event)
        assert gm.spread.volume == 9000
        assert parse_spread_line(gm.spread.question) == {"team": "Lakers", "line": -5.5}

    def test_missing_spread_and_totals_are_flagged(self):
        event = _event([_market("moneyline", 1000)])
        gm = classify_event_markets(event)
        assert gm.spread is None and gm.totals is None
        assert "missing_spread_market" in gm.flags
        assert "missing_totals_market" in gm.flags

    def test_missing_moneyline_is_flagged(self):
        event = _event([_market("spreads", 100, question="Spread: Lakers (-3.5)")])
        gm = classify_event_markets(event)
        assert gm.moneyline is None
        assert "MISSING_MONEYLINE_MARKET" in gm.flags

    def test_non_full_game_markets_are_dropped_not_misclassified(self):
        event = _event([
            _market("moneyline", 1000),
            _market("first_half_moneyline", 500),
            _market("player_points", 200),
        ])
        gm = classify_event_markets(event)
        assert len(gm.dropped) == 2
        assert gm.moneyline.volume == 1000

    def test_home_away_from_teams_ordering(self):
        event = _event([_market("moneyline", 1000)], teams=[
            {"ordering": "home", "name": "Warriors"},
            {"ordering": "away", "name": "Nuggets"},
        ])
        gm = classify_event_markets(event)
        assert gm.home_team == "Warriors"
        assert gm.away_team == "Nuggets"
        assert "teams_from_slug_fallback" not in gm.flags

    def test_falls_back_to_slug_when_teams_missing(self):
        event = _event([_market("moneyline", 1000)], slug="nba-den-gsw-2026-02-01", teams=[])
        gm = classify_event_markets(event)
        assert gm.away_team == "den"
        assert gm.home_team == "gsw"
        assert "teams_from_slug_fallback" in gm.flags

    def test_winner_index_picks_resolved_outcome(self):
        event = _event([_market("moneyline", 1000, prices=(0.0, 1.0))])
        gm = classify_event_markets(event)
        assert gm.moneyline.winner_index == 1


class TestParsing:
    def test_parse_spread_line_basic(self):
        assert parse_spread_line("Spread: Spurs (-5.5)") == {"team": "Spurs", "line": -5.5}

    def test_parse_spread_line_positive(self):
        assert parse_spread_line("Spread: Knicks (+3.5)") == {"team": "Knicks", "line": 3.5}

    def test_parse_spread_line_no_match(self):
        assert parse_spread_line("something unrelated") is None

    def test_parse_totals_line_basic(self):
        assert parse_totals_line("Knicks vs. Spurs: O/U 217.5") == 217.5

    def test_parse_totals_line_no_match(self):
        assert parse_totals_line("no total here") is None

    def test_parse_game_start_time_prefers_gameStartTime(self):
        from src.polymarket_prices.gamma import MarketRecord
        record = MarketRecord(
            condition_id="c", slug="s", question="q", sports_market_type="moneyline",
            outcomes=[], outcome_prices=[], clob_token_ids=[], volume=0,
            game_start_time="2026-06-14 00:30:00+00", start_date="2026-06-13 00:00:00+00", end_date=None,
        )
        ts = parse_game_start_time(record)
        assert ts.tz.zone == "UTC" if hasattr(ts.tz, "zone") else True
        assert ts.hour == 0 and ts.minute == 30

    def test_parse_game_start_time_falls_back_to_start_date(self):
        from src.polymarket_prices.gamma import MarketRecord
        record = MarketRecord(
            condition_id="c", slug="s", question="q", sports_market_type="moneyline",
            outcomes=[], outcome_prices=[], clob_token_ids=[], volume=0,
            game_start_time=None, start_date="2026-06-13 00:00:00+00", end_date=None,
        )
        ts = parse_game_start_time(record)
        assert ts is not None and ts.day == 13

    def test_parse_game_start_time_returns_none_when_both_missing(self):
        from src.polymarket_prices.gamma import MarketRecord
        record = MarketRecord(
            condition_id="c", slug="s", question="q", sports_market_type="moneyline",
            outcomes=[], outcome_prices=[], clob_token_ids=[], volume=0,
            game_start_time=None, start_date=None, end_date=None,
        )
        assert parse_game_start_time(record) is None
