"""
Build the unified per-game price series (Step 3) from raw moneyline trades,
then resample it into 1-minute OHLCV bars (Step 4).

A trade on the "other" token at price q is equivalent to a trade on the
reference token at price (1 - q); this module normalizes all trades onto a
single reference token (the eventual winner's token) so there's one clean
time series of "implied win probability of the team that ended up winning"
per game.

Robust-min methodology (design doc: "minimum of a rolling median over 5-10
consecutive trades, ... Store both raw and robust values"):
we use a rolling median with window=7 (midpoint of the suggested 5-10 range)
over the time-ordered in-game reference-price series, and take the min of
that rolling median. This damps single-share outlier fills (e.g. a lone
1-share trade at 0.01 surrounded by trades at 0.40) without needing a
volume threshold, which would require picking an arbitrary dollar cutoff.

Bar-level robust min (Step 4): once trades are resampled into 1-minute bars
(open/high/low/close/vwap/volume/trade_count), we additionally take the min
of the per-bar VWAP (volume-weighted average price) across in-game bars.
This is a second, coarser-grained robust estimate that's immune to any
single-trade outlier by construction (a bar's VWAP reflects every trade in
that minute, weighted by size) - reported alongside, not instead of, the
tick-level raw/rolling-median minimums.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)

ROLLING_WINDOW = 7
BAR_FREQ = "1min"

BARS_COLUMNS = [
    "bar_start", "open", "high", "low", "close", "vwap", "volume", "trade_count", "is_in_game",
]


@dataclass
class GameSeries:
    """Unified winner-referenced trade series for one game, plus summary stats."""

    df: pd.DataFrame  # columns: timestamp, price_raw, price_winner_ref, size, side, outcome, is_winner_token, is_in_game
    bars: pd.DataFrame  # 1-minute OHLCV bars, see BARS_COLUMNS - this is the primary stored series (Step 4)
    pregame_price_winner: Optional[float]
    pregame_trade_timestamp: Optional[int]
    in_game_min_price_raw: Optional[float]
    in_game_min_price_raw_ts: Optional[int]
    in_game_min_price_robust: Optional[float]
    in_game_min_price_robust_ts: Optional[int]
    in_game_min_price_bar_vwap: Optional[float]
    in_game_min_price_bar_vwap_ts: Optional[int]
    in_game_max_price_loser_sanity: Optional[float]  # independent check: should be ~ 1 - min_raw
    in_game_trade_count: int
    in_game_size_sum: float
    data_quality: str
    trade_source: str  # data_api / subgraph / mixed
    was_capped: bool  # True if the Data API's per-side offset cap was hit on at least one side
    earliest_trade_ts: Optional[int]
    pregame_coverage: bool  # True if at least one trade was observed strictly before tip-off
    notes: List[str] = field(default_factory=list)


def normalize_trades(
    trades: List[Dict[str, Any]], winner_token_id: str
) -> pd.DataFrame:
    """
    Convert raw trade records into a unified DataFrame referenced to
    `winner_token_id`: price is the winner-token-equivalent price for every
    trade, regardless of which of the two tokens was actually traded.
    """
    if not trades:
        return pd.DataFrame(
            columns=["timestamp", "price_raw", "price_winner_ref", "size", "side", "outcome", "is_winner_token"]
        )

    df = pd.DataFrame(trades)
    df["timestamp"] = df["timestamp"].astype("int64")
    df["price"] = df["price"].astype("float32")
    df["size"] = df["size"].astype("float32")

    # Store only a boolean flag for which side of the pair was traded, not
    # the full 78-digit token id string (recoverable from games.csv +
    # gamma.py if ever needed) - this is the single biggest win for
    # per-game parquet size at full-season scale.
    df["is_winner_token"] = df["asset"] == winner_token_id
    df["price_winner_ref"] = df["price"].where(df["is_winner_token"], 1.0 - df["price"]).astype("float32")
    df["side"] = df["side"].astype("category")
    df["outcome"] = df["outcome"].astype("category")

    df = df.sort_values("timestamp").reset_index(drop=True)
    return df.rename(columns={"price": "price_raw"})[
        ["timestamp", "price_raw", "price_winner_ref", "size", "side", "outcome", "is_winner_token"]
    ]


def resample_to_bars(df: pd.DataFrame, freq: str = BAR_FREQ) -> pd.DataFrame:
    """
    Resample a normalized, is_in_game-flagged trade DataFrame into 1-minute
    OHLCV bars over price_winner_ref. Empty input -> empty (but correctly
    typed/columned) output.
    """
    if df.empty or "is_in_game" not in df.columns:
        return pd.DataFrame(columns=BARS_COLUMNS)

    d = df.copy()
    d["bar_start"] = pd.to_datetime(d["timestamp"], unit="s", utc=True).dt.floor(freq)

    grouped = d.groupby("bar_start", sort=True)
    bars = grouped.agg(
        open=("price_winner_ref", "first"),
        high=("price_winner_ref", "max"),
        low=("price_winner_ref", "min"),
        close=("price_winner_ref", "last"),
        volume=("size", "sum"),
        trade_count=("price_winner_ref", "size"),
        is_in_game=("is_in_game", "any"),
    )

    notional = (d["price_winner_ref"].astype("float64") * d["size"].astype("float64"))
    vwap = notional.groupby(d["bar_start"]).sum() / d.groupby("bar_start")["size"].sum()
    bars["vwap"] = vwap

    bars = bars.reset_index()
    # astype("int64") on a tz-aware datetime column returns seconds or
    # nanoseconds depending on the column's underlying resolution (pandas
    # >= 2.0 can produce either datetime64[s]/[ns] here) - normalize to
    # datetime64[ns, UTC] first so the //10**9 -> unix-seconds conversion is
    # unambiguous across pandas versions.
    bars["bar_start"] = bars["bar_start"].astype("datetime64[ns, UTC]").astype("int64") // 10**9
    for c in ("open", "high", "low", "close", "vwap"):
        bars[c] = bars[c].astype("float32")
    bars["volume"] = bars["volume"].astype("float32")
    bars["trade_count"] = bars["trade_count"].astype("int32")
    bars["is_in_game"] = bars["is_in_game"].astype("bool")

    return bars[BARS_COLUMNS]


def _empty_series(
    df: pd.DataFrame,
    bars: pd.DataFrame,
    pregame_price: Optional[float],
    pregame_ts: Optional[int],
    data_quality: str,
    trade_source: str,
    was_capped: bool,
    earliest_trade_ts: Optional[int],
    pregame_coverage: bool,
    notes: List[str],
) -> GameSeries:
    return GameSeries(
        df=df,
        bars=bars,
        pregame_price_winner=pregame_price,
        pregame_trade_timestamp=pregame_ts,
        in_game_min_price_raw=None,
        in_game_min_price_raw_ts=None,
        in_game_min_price_robust=None,
        in_game_min_price_robust_ts=None,
        in_game_min_price_bar_vwap=None,
        in_game_min_price_bar_vwap_ts=None,
        in_game_max_price_loser_sanity=None,
        in_game_trade_count=0,
        in_game_size_sum=0.0,
        data_quality=data_quality,
        trade_source=trade_source,
        was_capped=was_capped,
        earliest_trade_ts=earliest_trade_ts,
        pregame_coverage=pregame_coverage,
        notes=notes,
    )


def build_game_series(
    trades: List[Dict[str, Any]],
    winner_token_id: str,
    game_start_ts: Optional[int],
    trade_meta: Dict[str, Any],
) -> GameSeries:
    """
    Build the unified series + 1-minute bars + summary stats for one game's
    moneyline market.

    `trade_meta` is the dict returned alongside trades by
    data_api.fetch_all_trades: {capped, buy_capped, sell_capped,
    earliest_trade_ts, buy_count, sell_count}.
    """
    notes: List[str] = []
    df = normalize_trades(trades, winner_token_id)
    was_capped = bool(trade_meta.get("capped", False))
    trade_source = "data_api"
    meta_earliest_ts = trade_meta.get("earliest_trade_ts")

    if df.empty:
        return _empty_series(
            df=df,
            bars=pd.DataFrame(columns=BARS_COLUMNS),
            pregame_price=None,
            pregame_ts=None,
            data_quality="no_trades",
            trade_source=trade_source,
            was_capped=was_capped,
            earliest_trade_ts=meta_earliest_ts,
            pregame_coverage=False,
            notes=["no trades returned by Data API"],
        )

    earliest_trade_ts = meta_earliest_ts if meta_earliest_ts is not None else int(df["timestamp"].min())

    if game_start_ts is None:
        notes.append("gameStartTime missing; in-game/pre-game split unavailable")
        df["is_in_game"] = True  # can't distinguish; treat all as in-game, flag it
        pregame_price = None
        pregame_ts = None
        pregame_coverage = False
    else:
        df["is_in_game"] = df["timestamp"] >= game_start_ts
        pregame_df = df[df["timestamp"] < game_start_ts]
        if not pregame_df.empty:
            last_pregame = pregame_df.iloc[-1]  # sorted ascending -> last row is latest pre-game trade
            pregame_price = float(last_pregame["price_winner_ref"])
            pregame_ts = int(last_pregame["timestamp"])
            pregame_coverage = True
        else:
            pregame_price = None
            pregame_ts = None
            pregame_coverage = False
            notes.append("no pre-game trades found (market may have opened at/after tip-off)")

    bars = resample_to_bars(df)

    in_game = df[df["is_in_game"]].reset_index(drop=True)

    if in_game.empty:
        notes.append("no in-game trades found")
        data_quality = "sparse_trades"
        return _empty_series(
            df=df,
            bars=bars,
            pregame_price=pregame_price,
            pregame_ts=pregame_ts,
            data_quality=data_quality,
            trade_source=trade_source,
            was_capped=was_capped,
            earliest_trade_ts=earliest_trade_ts,
            pregame_coverage=pregame_coverage,
            notes=notes,
        )

    # Raw min
    raw_min_idx = in_game["price_winner_ref"].idxmin()
    raw_min_price = float(in_game.loc[raw_min_idx, "price_winner_ref"])
    raw_min_ts = int(in_game.loc[raw_min_idx, "timestamp"])

    # Robust min: rolling median (window=7, min_periods=1 for short series)
    window = min(ROLLING_WINDOW, len(in_game))
    rolling_median = in_game["price_winner_ref"].rolling(window=window, min_periods=1, center=True).median()
    robust_min_idx = rolling_median.idxmin()
    robust_min_price = float(rolling_median.loc[robust_min_idx])
    robust_min_ts = int(in_game.loc[robust_min_idx, "timestamp"])

    # Bar-level robust min: min of per-minute VWAP across in-game bars.
    in_game_bars = bars[bars["is_in_game"]]
    if not in_game_bars.empty:
        bar_min_idx = in_game_bars["vwap"].idxmin()
        bar_vwap_min_price = float(in_game_bars.loc[bar_min_idx, "vwap"])
        bar_vwap_min_ts = int(in_game_bars.loc[bar_min_idx, "bar_start"])
    else:
        bar_vwap_min_price = None
        bar_vwap_min_ts = None

    # Independent sanity check: max price observed directly on the LOSER token
    # (not converted) should be ~ 1 - raw_min_price if normalization is correct.
    loser_direct = in_game[~in_game["is_winner_token"]]
    if not loser_direct.empty:
        loser_max_sanity = float(loser_direct["price_raw"].max())
    else:
        loser_max_sanity = None
        notes.append("no direct in-game trades on loser token to cross-check normalization")

    trade_count = len(in_game)
    size_sum = float(in_game["size"].sum())

    data_quality = "ok"
    if was_capped and not pregame_coverage:
        data_quality = "trades_capped"
        notes.append(
            f"Data API per-side offset cap reached on at least one side and no "
            f"pre-game trade recovered by the other side; only {len(df)} trades "
            f"available (older trades missing)"
        )
    elif was_capped:
        notes.append(
            "Data API per-side offset cap reached on at least one side, but "
            "side-split merge still recovered pre-game coverage from the other side"
        )
    if trade_count < 20:
        notes.append(f"sparse in-game trades (n={trade_count})")
        if data_quality == "ok":
            data_quality = "sparse_trades"

    return GameSeries(
        df=df,
        bars=bars,
        pregame_price_winner=pregame_price,
        pregame_trade_timestamp=pregame_ts,
        in_game_min_price_raw=raw_min_price,
        in_game_min_price_raw_ts=raw_min_ts,
        in_game_min_price_robust=robust_min_price,
        in_game_min_price_robust_ts=robust_min_ts,
        in_game_min_price_bar_vwap=bar_vwap_min_price,
        in_game_min_price_bar_vwap_ts=bar_vwap_min_ts,
        in_game_max_price_loser_sanity=loser_max_sanity,
        in_game_trade_count=trade_count,
        in_game_size_sum=size_sum,
        data_quality=data_quality,
        trade_source=trade_source,
        was_capped=was_capped,
        earliest_trade_ts=earliest_trade_ts,
        pregame_coverage=pregame_coverage,
        notes=notes,
    )
