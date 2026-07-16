"""
Step 3b: pre-game "closing line" snapshot for spread/totals markets.

Pre-game price is defined as the last traded price strictly BEFORE
gameStartTime. Trades are returned newest-first by the Data API (verified),
so we page forward until we find a trade with timestamp < game_start_ts and
take that one - we don't need the full series for these markets.
"""

import logging
from typing import Any, Dict, List, Optional

from .data_api import fetch_trades_until_before

logger = logging.getLogger(__name__)


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
