"""
Official per-team-game PACE/POSS from nba_api's TeamGameLogs endpoint
(MeasureType=Advanced) -- a season-level bulk pull (one API call per
season/season_type, same call granularity box_scores.py already uses for
Base-measure box scores), not a per-game pull.

Added for the Track C pace/possession swap-in test (docs/NEW_DATA_FEASIBILITY.md):
fingerprint.py's `pace_score` is a box-score-derived proxy
(PTS + OPP_PTS + TOV - 0.44*FTA); this module fetches the NBA's own computed
PACE/POSS as a candidate NEW pair of columns, cached alongside pace_score, not
replacing it.

Required parity check, same convention as box_scores.py: the set of game_ids
fetched here must match the `game` table 1:1 for the same season range/types.
"""

import logging
import time

import pandas as pd
from nba_api.stats.endpoints import TeamGameLogs

from src.data_processing.fetch_data import SEASON_TYPES, _date_to_season, _season_list
from src.matchups.db import cache_conn, init_cache_db, nba_api_conn, table_exists
from src.utils.config_loader import load_config

logger = logging.getLogger(__name__)

SLEEP_SECONDS = 0.7


def _existing_game_ids() -> set[str]:
    """game_ids currently in the `game` table (read-only)."""
    with nba_api_conn() as conn:
        rows = conn.execute("SELECT game_id FROM game").fetchall()
    return {r["game_id"] for r in rows}


def _cached_game_id_counts() -> dict[str, int]:
    """Row count per game_id currently in the team_advanced_stats cache. Each
    game needs exactly 2 rows (home + away team), same convention as
    box_scores.py's _cached_game_id_counts."""
    conn = cache_conn()
    if not table_exists(conn, "team_advanced_stats"):
        conn.close()
        return {}
    rows = conn.execute("SELECT game_id, COUNT(*) AS n FROM team_advanced_stats GROUP BY game_id").fetchall()
    conn.close()
    return {r["game_id"]: r["n"] for r in rows}


def _fetch_season_advanced(season: str, season_type: str) -> pd.DataFrame:
    df = TeamGameLogs(
        season_nullable=season,
        season_type_nullable=season_type,
        measure_type_player_game_logs_nullable="Advanced",
    ).get_data_frames()[0]
    if df.empty:
        return pd.DataFrame()

    return pd.DataFrame(
        {
            "game_id": df["GAME_ID"],
            "team_id": df["TEAM_ID"],
            "pace": df["PACE"],
            "poss": df["POSS"],
        }
    )


def build_advanced_stats_cache(force_refresh: bool = False) -> dict:
    """
    Fetch official PACE/POSS for the exact game_id set already present in the
    `game` table (same season range/season types box_scores.py uses), cache
    them, and run the required parity check. Mirrors
    box_scores.build_box_score_cache's structure exactly.

    Returns a dict summary: {n_game_table, n_cached, n_matched,
    n_missing_in_cache, n_incomplete_in_cache, parity_ok, missing_sample,
    incomplete_sample}.
    """
    init_cache_db()
    cfg = load_config()
    game_ids_needed = _existing_game_ids()
    cached_counts = {} if force_refresh else _cached_game_id_counts()
    complete = {gid for gid, n in cached_counts.items() if n >= 2}
    missing = game_ids_needed - complete

    logger.info(
        f"Advanced stats cache: {len(complete)} complete, "
        f"{len(missing)} to (re-)fetch (of {len(game_ids_needed)} total games in `game` table)"
    )

    if missing:
        start_season = _date_to_season(cfg.datasets_loading.data_start_date)
        seasons = _season_list(start_season)
        conn = cache_conn()
        total_inserted = 0
        for season in seasons:
            for season_type in SEASON_TYPES:
                try:
                    df = _fetch_season_advanced(season, season_type)
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
                        f"INSERT OR IGNORE INTO team_advanced_stats ({', '.join(cols)}) "
                        f"VALUES ({', '.join('?' * len(cols))})"
                    )
                    cursor = conn.executemany(sql, rows)
                    conn.commit()
                    total_inserted += cursor.rowcount
                time.sleep(SLEEP_SECONDS)
        conn.close()
        logger.info(f"Inserted {total_inserted} new team_advanced_stats rows")

    cached_counts_now = _cached_game_id_counts()
    cached_now = set(cached_counts_now)
    complete_now = {gid for gid, n in cached_counts_now.items() if n >= 2}
    incomplete_now = {gid for gid in game_ids_needed if cached_counts_now.get(gid, 0) == 1}
    n_matched = len(game_ids_needed & complete_now)
    n_missing = len(game_ids_needed - cached_now)
    n_incomplete = len(incomplete_now)
    parity_ok = n_missing == 0 and n_incomplete == 0

    summary = {
        "n_game_table": len(game_ids_needed),
        "n_cached": len(cached_now),
        "n_matched": n_matched,
        "n_missing_in_cache": n_missing,
        "n_incomplete_in_cache": n_incomplete,
        "parity_ok": parity_ok,
        "missing_sample": sorted(game_ids_needed - cached_now)[:10],
        "incomplete_sample": sorted(incomplete_now)[:10],
    }
    if not parity_ok:
        logger.warning(f"PARITY MISMATCH: {summary}")
    else:
        logger.info(
            f"Parity OK: {n_matched}/{len(game_ids_needed)} game_ids have exactly 2 rows (home + away)"
        )
    return summary


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    result = build_advanced_stats_cache()
    print(result)
