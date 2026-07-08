"""
Gap 1 infrastructure: raw box-score inputs (FGM/FGA/FG3M/FG3A/FTM/FTA/OREB/DREB/
AST/STL/BLK/TOV/PTS) for A7 fingerprints.

`data/raw/nba_api.sqlite`'s `game` table doesn't store these columns, but the same
nba_api LeagueGameLog endpoint src/data_processing/fetch_data.py already calls DOES
return them per team per game — fetch_data.py just doesn't select them. Rather than
re-implement season listing / season-type handling, this module reuses
`_season_list`, `_date_to_season`, and `SEASON_TYPES` from fetch_data.py (read-only
import, no modification).

Required parity check: the set of game_ids fetched here must match the `game` table
1:1 for the same season range and season types. A mismatch is reported, not silently
ignored, since A7's rolling windows would otherwise reference a different game set
than A2/H2H.
"""

import logging
import time

import pandas as pd
from nba_api.stats.endpoints import LeagueGameLog

from src.data_processing.fetch_data import SEASON_TYPES, _date_to_season, _season_list
from src.matchups.db import cache_conn, init_cache_db, nba_api_conn, table_exists
from src.utils.config_loader import load_config

logger = logging.getLogger(__name__)

SLEEP_SECONDS = 0.7
BOX_SCORE_COLS = [
    "fgm", "fga", "fg3m", "fg3a", "ftm", "fta",
    "oreb", "dreb", "ast", "stl", "blk", "tov", "pts",
]


def _existing_game_ids() -> set[str]:
    """game_ids currently in the `game` table (read-only)."""
    with nba_api_conn() as conn:
        rows = conn.execute("SELECT game_id FROM game").fetchall()
    return {r["game_id"] for r in rows}


def _cached_game_ids() -> set[str]:
    conn = cache_conn()
    if not table_exists(conn, "box_score_stats"):
        conn.close()
        return set()
    rows = conn.execute("SELECT DISTINCT game_id FROM box_score_stats").fetchall()
    conn.close()
    return {r["game_id"] for r in rows}


def _fetch_season_boxscores(season: str, season_type: str) -> pd.DataFrame:
    df = LeagueGameLog(
        season=season, season_type_all_star=season_type, league_id="00"
    ).get_data_frames()[0]
    if df.empty:
        return pd.DataFrame()

    is_home = df["MATCHUP"].str.contains(r"vs\.")
    out = pd.DataFrame({
        "game_id": df["GAME_ID"],
        "team_id": df["TEAM_ID"],
        "is_home": is_home.astype(int),
        "game_date": df["GAME_DATE"],
        "fgm": df["FGM"], "fga": df["FGA"],
        "fg3m": df["FG3M"], "fg3a": df["FG3A"],
        "ftm": df["FTM"], "fta": df["FTA"],
        "oreb": df["OREB"], "dreb": df["DREB"],
        "ast": df["AST"], "stl": df["STL"], "blk": df["BLK"], "tov": df["TOV"],
        "pts": df["PTS"],
    })
    return out


def build_box_score_cache(force_refresh: bool = False) -> dict:
    """
    Fetch box scores for the exact game_id set already present in the `game` table
    (same season range / season types fetch_data.py uses), cache them, and run the
    required parity check.

    Returns a dict summary: {n_game_table, n_cached, n_matched, n_missing_in_cache,
    n_extra_in_cache, parity_ok, missing_game_ids (sample)}.
    """
    init_cache_db()
    cfg = load_config()
    game_ids_needed = _existing_game_ids()
    already_cached = set() if force_refresh else _cached_game_ids()
    missing = game_ids_needed - already_cached

    logger.info(
        f"Box score cache: {len(already_cached)} already cached, "
        f"{len(missing)} to fetch (of {len(game_ids_needed)} total games in `game` table)"
    )

    if missing:
        start_season = _date_to_season(cfg.datasets_loading.data_start_date)
        seasons = _season_list(start_season)
        conn = cache_conn()
        total_inserted = 0
        for season in seasons:
            for season_type in SEASON_TYPES:
                try:
                    df = _fetch_season_boxscores(season, season_type)
                except Exception as e:
                    logger.error(f"Error fetching {season} {season_type}: {e}")
                    time.sleep(SLEEP_SECONDS)
                    continue
                if df.empty:
                    time.sleep(SLEEP_SECONDS)
                    continue
                df = df[df["game_id"].isin(missing)]
                if not df.empty:
                    rows = [tuple(r) for r in df.itertuples(index=False)]
                    cols = list(df.columns)
                    sql = (
                        f"INSERT OR IGNORE INTO box_score_stats ({', '.join(cols)}) "
                        f"VALUES ({', '.join('?' * len(cols))})"
                    )
                    cursor = conn.executemany(sql, rows)
                    conn.commit()
                    total_inserted += cursor.rowcount
                time.sleep(SLEEP_SECONDS)
        conn.close()
        logger.info(f"Inserted {total_inserted} new box_score_stats rows")

    # --- Required parity check ---
    cached_now = _cached_game_ids()
    n_matched = len(game_ids_needed & cached_now)
    n_missing = len(game_ids_needed - cached_now)
    n_extra = len(cached_now - game_ids_needed)
    parity_ok = n_missing == 0

    summary = {
        "n_game_table": len(game_ids_needed),
        "n_cached": len(cached_now),
        "n_matched": n_matched,
        "n_missing_in_cache": n_missing,
        "n_extra_in_cache": n_extra,
        "parity_ok": parity_ok,
        "missing_sample": sorted(game_ids_needed - cached_now)[:10],
    }
    if not parity_ok:
        logger.warning(f"PARITY MISMATCH: {summary}")
    else:
        logger.info(f"Parity OK: {n_matched}/{len(game_ids_needed)} game_ids match 1:1")
    return summary


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    result = build_box_score_cache()
    print(result)
