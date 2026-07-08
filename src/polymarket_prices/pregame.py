"""
Step 3b: pre-game "closing line" snapshot for spread/totals markets (and a
CLOB cross-check helper for the moneyline pre-game price).

Pre-game price is defined as the last traded price strictly BEFORE
gameStartTime. Trades are returned newest-first by the Data API (verified),
so we page forward until we find a trade with timestamp < game_start_ts and
take that one - we don't need the full series for these markets.
"""

import logging
from typing import Any, Dict, List, Optional

from .data_api import fetch_trades_until_before
from .http_utils import get_json

logger = logging.getLogger(__name__)

CLOB_BASE = "https://clob.polymarket.com"


def find_last_pregame_trade(
    trades: List[Dict[str, Any]], game_start_ts: int
) -> Optional[Dict[str, Any]]:
    """Given trades (any order), return the one with the largest timestamp < game_start_ts."""
    candidates = [t for t in trades if t["timestamp"] < game_start_ts]
    if not candidates:
        return None
    return max(candidates, key=lambda t: t["timestamp"])


def fetch_pregame_snapshot(
    condition_id: str, game_start_ts: int, raw_dir: str, force_refresh: bool = False
) -> Dict[str, Any]:
    """
    Fetch (bounded) trades for a market and return the last pre-game trade's
    price/outcome/timestamp, or a flagged empty result if none found.
    """
    trades, capped = fetch_trades_until_before(
        condition_id, game_start_ts, raw_dir, force_refresh=force_refresh
    )
    last = find_last_pregame_trade(trades, game_start_ts)
    if last is None:
        return {
            "price": None,
            "outcome": None,
            "outcome_index": None,
            "timestamp": None,
            "capped": capped,
            "flag": "no_pregame_trades_found",
        }
    return {
        "price": float(last["price"]),
        "outcome": last.get("outcome"),
        "outcome_index": last.get("outcomeIndex"),
        "timestamp": int(last["timestamp"]),
        "capped": capped,
        "flag": None,
    }


def clob_cross_check(token_id: str, start_ts: int, end_ts: int) -> Optional[Dict[str, Any]]:
    """
    Optional cross-check (design doc Step 3b): query CLOB /prices-history
    with a startTs/endTs window ending at tip-off, to see whether pre-game
    candles survive resolution even when in-game ones don't. Best-effort;
    returns None on any failure or empty history.
    """
    try:
        data = get_json(
            f"{CLOB_BASE}/prices-history",
            params={"market": token_id, "startTs": start_ts, "endTs": end_ts, "fidelity": 10},
        )
    except Exception as e:
        logger.debug(f"CLOB cross-check failed for {token_id}: {e}")
        return None

    history = data.get("history", [])
    if not history:
        return None
    last = history[-1]
    return {"clob_pregame_price": float(last["p"]), "clob_pregame_ts": int(last["t"]), "num_points": len(history)}
