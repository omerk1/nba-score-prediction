"""
Tests for polymarket_collector.py
"""

import pytest
import pandas as pd
from unittest.mock import patch, MagicMock
from src.data_processing.polymarket_collector import (
    infer_market_type,
    extract_teams_from_question,
    fetch_price_history,
    collect_polymarket_odds,
)


class TestInferMarketType:
    """Test market type inference from question strings."""

    def test_spread_detection(self):
        assert infer_market_type("Clippers -3.5") == "spread"
        assert infer_market_type("Lakers +2.5") == "spread"
        assert infer_market_type("Will the Clippers cover -3.5?") == "spread"

    def test_totals_detection(self):
        assert infer_market_type("Over 215.5") == "totals"
        assert infer_market_type("Under 215.5") == "totals"
        assert infer_market_type("Total points Over 220") == "totals"

    def test_moneyline_detection(self):
        assert infer_market_type("Clippers vs Mavericks") == "moneyline"
        assert infer_market_type("Will the Lakers win?") == "moneyline"
        assert infer_market_type("Clippers to win 2024-04-07") == "moneyline"


class TestExtractTeams:
    """Test team extraction from question strings."""

    def test_vs_pattern(self):
        home, away = extract_teams_from_question("Clippers vs Mavericks")
        assert home == "Clippers"
        assert away == "Mavericks"

    def test_vs_period_pattern(self):
        home, away = extract_teams_from_question("Lakers vs. Celtics 2024-04-07")
        assert home == "Lakers"
        assert away == "Celtics"

    def test_full_names(self):
        home, away = extract_teams_from_question("Los Angeles Clippers vs Dallas Mavericks")
        assert home == "Los Angeles Clippers"
        assert away == "Dallas Mavericks"

    def test_invalid_format(self):
        home, away = extract_teams_from_question("Invalid question format")
        assert home is None
        assert away is None


class TestFetchPriceHistory:
    """Test price history fetching from CLOB API."""

    @patch("src.data_processing.polymarket_collector.requests.get")
    def test_valid_history(self, mock_get):
        """Test successful price history fetch."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "history": [
                {"p": 0.52, "t": 1680000000},
                {"p": 0.51, "t": 1680001000},
                {"p": 0.49, "t": 1680002000},
            ]
        }
        mock_get.return_value = mock_response

        result = fetch_price_history("token123")

        assert result is not None
        assert result["opening_price"] == 0.52
        assert result["closing_price"] == 0.49
        assert result["num_points"] == 3

    @patch("src.data_processing.polymarket_collector.requests.get")
    def test_empty_history(self, mock_get):
        """Test handling of empty price history."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"history": []}
        mock_get.return_value = mock_response

        result = fetch_price_history("token123")
        assert result is None

    def test_none_token(self):
        """Test handling of None token."""
        result = fetch_price_history(None)
        assert result is None


class TestCollectPolymarketOdds:
    """Integration tests for odds collection."""

    @patch("src.data_processing.polymarket_collector.find_nba_markets")
    @patch("src.data_processing.polymarket_collector.fetch_price_history")
    @patch("src.data_processing.polymarket_collector.lookup_game_id")
    def test_collect_sample(self, mock_lookup, mock_prices, mock_markets):
        """Test collecting odds from sample data."""
        # Mock market data
        mock_markets.return_value = [
            {
                "slug": "nba-lac-dal-2023-04-07",
                "game_date": "2023-04-07",
                "title": "NBA: Clippers vs Mavericks",
                "markets": [
                    {
                        "question": "Clippers vs Mavericks 2023-04-07",
                        "clobTokenIds": '["token1", "token2"]',
                        "volume": "125000",
                    }
                ]
            }
        ]

        # Mock price history
        mock_prices.return_value = {
            "opening_price": 0.52,
            "closing_price": 0.49,
            "timestamp": 1680000000,
            "num_points": 10
        }

        # Mock game ID lookup
        mock_lookup.return_value = "0022300812"

        # Collect (returns tuple now)
        df_odds, df_failed = collect_polymarket_odds(
            "2023-04-07",
            "2023-04-07",
            "data/raw/nba_api.sqlite",
            rate_limit_delay=0,  # No delay for tests
            output_csv="/tmp/test_odds.csv",
            failed_csv="/tmp/test_failed.csv",
            resume=False
        )

        # Verify
        assert len(df_odds) > 0
        assert "game_id" in df_odds.columns
        assert "game_date" in df_odds.columns
        assert "market_type" in df_odds.columns
        assert "opening_price" in df_odds.columns
        assert "closing_price" in df_odds.columns


class TestEndToEnd:
    """End-to-end tests with real (or mocked) API calls."""

    @pytest.mark.skip(reason="Requires real API - use for manual testing")
    def test_real_backfill_sample(self):
        """Test real backfill on small date range (manual test only)."""
        df = collect_polymarket_odds(
            "2023-04-01",
            "2023-04-05",
            "data/raw/nba_api.sqlite"
        )

        # Verify we got some data
        assert len(df) > 0
        assert df["game_date"].notna().all()
        assert df["market_type"].isin(["spread", "moneyline", "totals"]).all()
