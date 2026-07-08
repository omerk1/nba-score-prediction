"""
Build the unified per-game price series (Step 3) from raw moneyline trades.

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
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)

ROLLING_WINDOW = 7


@dataclass
class GameSeries:
    """Unified winner-referenced trade series for one game, plus summary stats."""

    df: pd.DataFrame  # columns: timestamp, price_raw, price_winner_ref, size, side, outcome, is_winner_token, is_in_game
    pregame_price_winner: Optional[float]
    pregame_trade_timestamp: Optional[int]
    in_game_min_price_raw: Optional[float]
    in_game_min_price_raw_ts: Optional[int]
    in_game_min_price_robust: Optional[float]
    in_game_min_price_robust_ts: Optional[int]
    in_game_max_price_loser_sanity: Optional[float]  # independent check: should be ~ 1 - min_raw
    in_game_trade_count: int
    in_game_size_sum: float
    data_quality: str
    notes: List[str]


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


def build_game_series(
    trades: List[Dict[str, Any]],
    winner_token_id: str,
    game_start_ts: Optional[int],
    trades_capped: bool,
) -> GameSeries:
    """Build the unified series + summary stats for one game's moneyline market."""
    notes: List[str] = []
    df = normalize_trades(trades, winner_token_id)

    if df.empty:
        return GameSeries(
            df=df,
            pregame_price_winner=None,
            pregame_trade_timestamp=None,
            in_game_min_price_raw=None,
            in_game_min_price_raw_ts=None,
            in_game_min_price_robust=None,
            in_game_min_price_robust_ts=None,
            in_game_max_price_loser_sanity=None,
            in_game_trade_count=0,
            in_game_size_sum=0.0,
            data_quality="no_trades",
            notes=["no trades returned by Data API"],
        )

    if game_start_ts is None:
        notes.append("gameStartTime missing; in-game/pre-game split unavailable")
        df["is_in_game"] = True  # can't distinguish; treat all as in-game, flag it
        pregame_price = None
        pregame_ts = None
    else:
        df["is_in_game"] = df["timestamp"] >= game_start_ts
        pregame_df = df[df["timestamp"] < game_start_ts]
        if not pregame_df.empty:
            last_pregame = pregame_df.iloc[-1]  # sorted ascending -> last row is latest pre-game trade
            pregame_price = float(last_pregame["price_winner_ref"])
            pregame_ts = int(last_pregame["timestamp"])
        else:
            pregame_price = None
            pregame_ts = None
            notes.append("no pre-game trades found (market may have opened at/after tip-off)")

    in_game = df[df["is_in_game"]].reset_index(drop=True)

    if in_game.empty:
        notes.append("no in-game trades found")
        data_quality = "sparse_trades"
        return GameSeries(
            df=df,
            pregame_price_winner=pregame_price,
            pregame_trade_timestamp=pregame_ts,
            in_game_min_price_raw=None,
            in_game_min_price_raw_ts=None,
            in_game_min_price_robust=None,
            in_game_min_price_robust_ts=None,
            in_game_max_price_loser_sanity=None,
            in_game_trade_count=0,
            in_game_size_sum=0.0,
            data_quality=data_quality,
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
    if trades_capped:
        data_quality = "trades_capped"
        notes.append(f"Data API offset cap reached; only {len(df)} most-recent trades available (older trades missing)")
    if trade_count < 20:
        notes.append(f"sparse in-game trades (n={trade_count})")
        if data_quality == "ok":
            data_quality = "sparse_trades"

    return GameSeries(
        df=df,
        pregame_price_winner=pregame_price,
        pregame_trade_timestamp=pregame_ts,
        in_game_min_price_raw=raw_min_price,
        in_game_min_price_raw_ts=raw_min_ts,
        in_game_min_price_robust=robust_min_price,
        in_game_min_price_robust_ts=robust_min_ts,
        in_game_max_price_loser_sanity=loser_max_sanity,
        in_game_trade_count=trade_count,
        in_game_size_sum=size_sum,
        data_quality=data_quality,
        notes=notes,
    )
