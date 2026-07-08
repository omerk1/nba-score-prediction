"""
Orchestration: given a list of NBA game slugs, run discovery -> trade
download -> series construction -> per-game summary -> pre-game snapshot,
and write:

- data/polymarket_prices/raw/                cached raw API responses (via http_utils/data_api)
- data/polymarket_prices/games.csv           one row per game (Step 3 + Step 3b columns)
- data/polymarket_prices/series/{slug}.parquet   full per-game unified price series

Resumable: if a game's row is already present in games.csv AND its series
parquet exists, it is skipped (unless force_refresh=True). Raw API responses
are independently cached (see gamma.py / data_api.py), so even a partial
re-run avoids re-downloading anything already fetched.
"""

import logging
import os
from typing import Any, Dict, List, Optional

import pandas as pd

from .data_api import fetch_all_trades
from .gamma import (
    GameMarkets,
    classify_event_markets,
    fetch_event_by_slug,
    parse_game_start_time,
    parse_spread_line,
    parse_totals_line,
)
from .pregame import fetch_pregame_snapshot
from .series_builder import build_game_series

logger = logging.getLogger(__name__)


def _find_outcome_index(outcomes: List[str], name: str) -> Optional[int]:
    name_l = name.lower()
    for i, o in enumerate(outcomes):
        if o.lower() == name_l or name_l in o.lower() or o.lower() in name_l:
            return i
    return None


def process_game(
    slug: str, raw_dir: str, force_refresh: bool = False
) -> Optional[Dict[str, Any]]:
    """
    Run the full pipeline for one game. Returns a dict with two keys:
    'summary' (the games.csv row) and 'series' (the unified series DataFrame),
    or None if the game couldn't be processed at all (e.g. no event found).
    """
    logger.info(f"Processing {slug} ...")
    flags: List[str] = []

    event = fetch_event_by_slug(slug, raw_dir, force_refresh=force_refresh)
    if event is None:
        logger.error(f"{slug}: no event found, skipping")
        return None

    gm: GameMarkets = classify_event_markets(event)
    flags.extend(gm.flags)

    if gm.moneyline is None:
        logger.error(f"{slug}: no moneyline market found, skipping")
        return {
            "summary": {
                "slug": slug,
                "data_quality": "no_moneyline_market",
                "flags": ";".join(flags),
            },
            "series": pd.DataFrame(),
        }

    ml = gm.moneyline
    winner_idx = ml.winner_index
    if winner_idx is None or len(ml.clob_token_ids) < 2:
        logger.error(f"{slug}: moneyline market unresolved or malformed tokens, skipping")
        return {
            "summary": {
                "slug": slug,
                "data_quality": "unresolved_or_malformed_moneyline",
                "flags": ";".join(flags),
            },
            "series": pd.DataFrame(),
        }

    winner_team = ml.outcomes[winner_idx]
    loser_idx = 1 - winner_idx
    loser_team = ml.outcomes[loser_idx]
    winner_token = ml.clob_token_ids[winner_idx]

    game_start = parse_game_start_time(ml)
    if game_start is None:
        flags.append("gameStartTime_and_startDate_missing")
    game_start_ts = int(game_start.timestamp()) if game_start is not None else None

    trades, capped = fetch_all_trades(ml.condition_id, raw_dir, force_refresh=force_refresh)

    series = build_game_series(
        trades=trades,
        winner_token_id=winner_token,
        game_start_ts=game_start_ts,
        trades_capped=capped,
    )
    flags.extend(series.notes)

    row: Dict[str, Any] = {
        "slug": slug,
        "event_id": gm.event_id,
        "game_date": gm.game_date,
        "game_start_time_utc": game_start.isoformat() if game_start is not None else None,
        "away_team": gm.away_team,
        "home_team": gm.home_team,
        "winner_team": winner_team,
        "loser_team": loser_team,
        "moneyline_condition_id": ml.condition_id,
        "moneyline_volume": ml.volume,
        # Step 3 summary
        "pregame_price_winner": series.pregame_price_winner,
        "pregame_trade_timestamp": series.pregame_trade_timestamp,
        "in_game_min_price_winner_raw": series.in_game_min_price_raw,
        "in_game_min_price_winner_raw_ts": series.in_game_min_price_raw_ts,
        "in_game_min_price_winner_robust": series.in_game_min_price_robust,
        "in_game_min_price_winner_robust_ts": series.in_game_min_price_robust_ts,
        "in_game_max_price_loser_sanity": series.in_game_max_price_loser_sanity,
        "in_game_trade_count": series.in_game_trade_count,
        "in_game_size_sum": series.in_game_size_sum,
        "data_quality": series.data_quality,
    }

    # --- Step 3b: pre-game snapshot for spread + totals ---
    if gm.spread is not None:
        spread_parsed = parse_spread_line(gm.spread.question)
        row["spread_condition_id"] = gm.spread.condition_id
        row["spread_volume"] = gm.spread.volume
        row["spread_team"] = spread_parsed["team"] if spread_parsed else None
        row["spread_line"] = spread_parsed["line"] if spread_parsed else None
        favorite_idx = (
            _find_outcome_index(gm.spread.outcomes, spread_parsed["team"]) if spread_parsed else None
        )
        row["spread_favorite_covered"] = (
            bool(gm.spread.outcome_prices[favorite_idx] >= 0.5)
            if favorite_idx is not None and gm.spread.outcome_prices
            else None
        )
        if game_start_ts is not None:
            snap = fetch_pregame_snapshot(
                gm.spread.condition_id, game_start_ts, raw_dir, force_refresh=force_refresh
            )
            if snap["price"] is None or favorite_idx is None:
                row["spread_pregame_price_favorite"] = None
                if favorite_idx is None:
                    flags.append("spread_favorite_team_unparsed")
            elif snap.get("outcome_index") == favorite_idx:
                row["spread_pregame_price_favorite"] = snap["price"]
            else:
                row["spread_pregame_price_favorite"] = 1 - snap["price"]
            if snap.get("flag"):
                flags.append(f"spread_{snap['flag']}")
        else:
            row["spread_pregame_price_favorite"] = None
    else:
        for c in (
            "spread_condition_id", "spread_volume", "spread_team", "spread_line",
            "spread_favorite_covered", "spread_pregame_price_favorite",
        ):
            row[c] = None

    if gm.totals is not None:
        total_line = parse_totals_line(gm.totals.question)
        row["totals_condition_id"] = gm.totals.condition_id
        row["totals_volume"] = gm.totals.volume
        row["totals_line"] = total_line
        over_idx = _find_outcome_index(gm.totals.outcomes, "Over")
        row["totals_over_hit"] = (
            bool(gm.totals.outcome_prices[over_idx] >= 0.5)
            if over_idx is not None and gm.totals.outcome_prices
            else None
        )
        if game_start_ts is not None:
            snap = fetch_pregame_snapshot(
                gm.totals.condition_id, game_start_ts, raw_dir, force_refresh=force_refresh
            )
            if snap["price"] is None or over_idx is None:
                row["totals_pregame_price_over"] = None
                if over_idx is None:
                    flags.append("totals_over_outcome_unparsed")
            elif snap.get("outcome_index") == over_idx:
                row["totals_pregame_price_over"] = snap["price"]
            else:
                row["totals_pregame_price_over"] = 1 - snap["price"]
            if snap.get("flag"):
                flags.append(f"totals_{snap['flag']}")
        else:
            row["totals_pregame_price_over"] = None
    else:
        for c in ("totals_condition_id", "totals_volume", "totals_line", "totals_over_hit", "totals_pregame_price_over"):
            row[c] = None

    row["flags"] = ";".join(flags)

    return {"summary": row, "series": series.df}


def run_pipeline(
    slugs: List[str],
    raw_dir: str,
    games_csv_path: str,
    series_dir: str,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """Run process_game for each slug, writing games.csv and series parquet files. Resumable."""
    os.makedirs(series_dir, exist_ok=True)
    os.makedirs(os.path.dirname(games_csv_path) or ".", exist_ok=True)

    existing_rows: Dict[str, Dict[str, Any]] = {}
    if not force_refresh and os.path.exists(games_csv_path):
        existing_df = pd.read_csv(games_csv_path)
        existing_rows = {r["slug"]: r.to_dict() for _, r in existing_df.iterrows()}

    rows = []
    for slug in slugs:
        series_path = os.path.join(series_dir, f"{slug}.parquet")
        if not force_refresh and slug in existing_rows and os.path.exists(series_path):
            logger.info(f"{slug}: already processed, skipping (resumable)")
            rows.append(existing_rows[slug])
            continue

        result = process_game(slug, raw_dir, force_refresh=force_refresh)
        if result is None:
            continue

        rows.append(result["summary"])
        if not result["series"].empty:
            # zstd gives a better compression ratio than the pyarrow default
            # (snappy) for this kind of repetitive numeric/categorical data;
            # matters once this is ~1,230 files instead of 5.
            result["series"].to_parquet(series_path, index=False, compression="zstd")
        logger.info(f"{slug}: done (data_quality={result['summary'].get('data_quality')})")

    # games.csv accumulates across separate pipeline invocations that each
    # target a different batch of slugs (e.g. a 5-game checkpoint followed
    # later by a disjoint 50-game batch) -- carry forward any previously
    # processed games NOT in this call's slug list, so they aren't silently
    # dropped from the CSV on overwrite (their series parquet files are
    # untouched either way; this only concerns the summary CSV).
    processed_slugs = set(slugs)
    for slug, row in existing_rows.items():
        if slug not in processed_slugs:
            rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(games_csv_path, index=False)
    logger.info(f"Wrote {len(df)} game rows to {games_csv_path}")
    return df
