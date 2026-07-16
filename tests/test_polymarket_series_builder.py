"""
Core-logic tests for src/polymarket_prices/series_builder.py: winner-token price
normalization, OHLCV bar resampling, and the raw/robust in-game minimum calculation.
"""

import pandas as pd
import pytest

from src.polymarket_prices.series_builder import (
    build_game_series,
    normalize_trades,
    resample_to_bars,
)

WINNER_TOKEN = "winner_token_id"
LOSER_TOKEN = "loser_token_id"


def _trade(ts, price, size, asset=WINNER_TOKEN, side="BUY", outcome="Yes"):
    return {"timestamp": ts, "price": price, "size": size, "asset": asset, "side": side, "outcome": outcome}


class TestNormalizeTrades:
    def test_winner_token_price_kept_as_is(self):
        df = normalize_trades([_trade(100, 0.7, 10, asset=WINNER_TOKEN)], WINNER_TOKEN)
        assert df.loc[0, "price_winner_ref"] == pytest.approx(0.7)
        assert df.loc[0, "is_winner_token"]

    def test_loser_token_price_is_flipped(self):
        df = normalize_trades([_trade(100, 0.7, 10, asset=LOSER_TOKEN)], WINNER_TOKEN)
        assert df.loc[0, "price_winner_ref"] == pytest.approx(0.3)
        assert not df.loc[0, "is_winner_token"]

    def test_sorted_ascending_by_timestamp(self):
        df = normalize_trades([_trade(300, 0.5, 1), _trade(100, 0.5, 1), _trade(200, 0.5, 1)], WINNER_TOKEN)
        assert df["timestamp"].tolist() == [100, 200, 300]

    def test_empty_input(self):
        df = normalize_trades([], WINNER_TOKEN)
        assert df.empty
        assert list(df.columns) == [
            "timestamp", "price_raw", "price_winner_ref", "size", "side", "outcome", "is_winner_token",
        ]


class TestResampleToBars:
    def test_ohlcv_aggregation(self):
        df = normalize_trades(
            [_trade(0, 0.40, 10), _trade(10, 0.50, 20), _trade(20, 0.30, 5), _trade(50, 0.45, 10)],
            WINNER_TOKEN,
        )
        df["is_in_game"] = True
        bars = resample_to_bars(df)

        # First 3 trades (ts 0,10,20) fall in the same 1-minute bar; the 4th (ts 50) also
        # falls in that same bar since 50s < 60s -- so this is one bar for all 4 trades.
        assert len(bars) == 1
        row = bars.iloc[0]
        assert row["open"] == pytest.approx(0.40)
        assert row["high"] == pytest.approx(0.50)
        assert row["low"] == pytest.approx(0.30)
        assert row["close"] == pytest.approx(0.45)
        assert row["trade_count"] == 4
        assert row["volume"] == pytest.approx(45)
        expected_vwap = (0.40 * 10 + 0.50 * 20 + 0.30 * 5 + 0.45 * 10) / 45
        assert row["vwap"] == pytest.approx(expected_vwap, abs=1e-4)

    def test_separate_minutes_produce_separate_bars(self):
        df = normalize_trades([_trade(0, 0.5, 1), _trade(90, 0.6, 1)], WINNER_TOKEN)
        df["is_in_game"] = True
        bars = resample_to_bars(df)
        assert len(bars) == 2

    def test_empty_input(self):
        assert resample_to_bars(pd.DataFrame()).empty


class TestBuildGameSeries:
    GAME_START = 1000

    def _meta(self, capped=False, earliest_ts=None):
        return {"capped": capped, "buy_capped": capped, "sell_capped": False, "earliest_trade_ts": earliest_ts}

    def test_pregame_price_is_last_trade_before_tipoff(self):
        trades = [_trade(900, 0.55, 1), _trade(950, 0.60, 1), _trade(1100, 0.50, 1)]
        series = build_game_series(trades, WINNER_TOKEN, self.GAME_START, self._meta())
        assert series.pregame_price_winner == pytest.approx(0.60)
        assert series.pregame_trade_timestamp == 950
        assert series.pregame_coverage

    def test_no_trades_before_tipoff_flags_missing_pregame_coverage(self):
        trades = [_trade(1100, 0.50, 1), _trade(1200, 0.40, 1)]
        series = build_game_series(trades, WINNER_TOKEN, self.GAME_START, self._meta())
        assert series.pregame_price_winner is None
        assert not series.pregame_coverage

    def test_robust_min_damps_single_outlier_trade(self):
        # A lone 1-share outlier at 0.01 surrounded by trades around 0.40 shouldn't
        # drag the ROBUST min down to 0.01 the way the RAW min does.
        trades = [_trade(GT, 0.40, 10) for GT in range(1010, 1010 + 6 * 10, 10)]
        trades.insert(3, _trade(1035, 0.01, 1))
        series = build_game_series(trades, WINNER_TOKEN, self.GAME_START, self._meta())
        assert series.in_game_min_price_raw == pytest.approx(0.01)
        assert series.in_game_min_price_robust > 0.30  # robust min ignores the outlier

    def test_no_trades_at_all(self):
        series = build_game_series([], WINNER_TOKEN, self.GAME_START, self._meta())
        assert series.data_quality == "no_trades"
        assert series.in_game_min_price_raw is None

    def test_only_pregame_trades_is_sparse_not_ok(self):
        trades = [_trade(900, 0.5, 1), _trade(950, 0.5, 1)]
        series = build_game_series(trades, WINNER_TOKEN, self.GAME_START, self._meta())
        assert series.data_quality == "sparse_trades"
        assert series.in_game_trade_count == 0

    def test_capped_without_pregame_coverage_flags_trades_capped(self):
        trades = [_trade(1100 + i, 0.5, 1) for i in range(25)]
        series = build_game_series(trades, WINNER_TOKEN, self.GAME_START, self._meta(capped=True))
        assert series.data_quality == "trades_capped"

    def test_capped_but_pregame_coverage_recovered_is_still_ok(self):
        trades = [_trade(900, 0.5, 1)] + [_trade(1100 + i, 0.5, 1) for i in range(25)]
        series = build_game_series(trades, WINNER_TOKEN, self.GAME_START, self._meta(capped=True))
        assert series.data_quality == "ok"
        assert series.pregame_coverage

    def test_loser_sanity_check_is_complementary_to_raw_min(self):
        trades = [
            _trade(1010, 0.40, 10, asset=WINNER_TOKEN),
            _trade(1020, 0.60, 10, asset=LOSER_TOKEN),  # loser-token trade at 0.60 -> winner-ref 0.40
        ] + [_trade(1030 + i, 0.5, 1) for i in range(20)]
        series = build_game_series(trades, WINNER_TOKEN, self.GAME_START, self._meta())
        # The only direct loser-token trade was at raw price 0.60.
        assert series.in_game_max_price_loser_sanity == pytest.approx(0.60)
