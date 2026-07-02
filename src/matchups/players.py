"""
Gap 2 infrastructure: resolve player_injuries.player_name (free text) to player_id
(as used by player_stats_cache), and classify player-seasons into archetypes.

Join strategy: nba_api.stats.static.players.get_players() is the same live static
list src/news_scraping/pipeline.py:_resolve_team_id already uses (for teams, via
nba_api.stats.static.teams) — here we use its player equivalent.

Normalization order (per project instructions): normalize both sides (strip
whitespace/periods, unify case, strip Jr./Sr./II/III/IV suffixes, strip accents)
and match on that first. Only if the normalized name has zero candidates do we
retry with an unnormalized exact match (handles rare cases where normalization
over-stripped something on the static-list side).

Disambiguation: when a normalized name maps to >1 candidate player_id, prefer
whichever candidate has player_stats_cache activity within ~60 days of the
player's earliest player_injuries.game_date.
"""

import logging
import re
import unicodedata
from collections import defaultdict
from datetime import date, timedelta

import pandas as pd
from nba_api.stats.static import players as nba_players

from src.matchups.db import cache_conn, init_cache_db, injury_conn, nba_api_conn, table_exists

logger = logging.getLogger(__name__)

_SUFFIX_RE = re.compile(r"\b(jr|sr|ii|iii|iv|v)\.?$", re.IGNORECASE)
DISAMBIGUATION_WINDOW_DAYS = 60


def normalize_name(name: str) -> str:
    if not name:
        return ""
    n = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode("ascii")
    n = n.lower().replace(".", "").strip()
    n = re.sub(r"\s+", " ", n)
    n = _SUFFIX_RE.sub("", n).strip()
    return n


def _static_player_index() -> tuple[dict[str, list[int]], dict[str, list[int]]]:
    """Return (normalized_name -> [player_id,...], exact_name -> [player_id,...])."""
    all_players = nba_players.get_players()
    norm_idx: dict[str, list[int]] = defaultdict(list)
    exact_idx: dict[str, list[int]] = defaultdict(list)
    for p in all_players:
        norm_idx[normalize_name(p["full_name"])].append(p["id"])
        exact_idx[p["full_name"]].append(p["id"])
    return dict(norm_idx), dict(exact_idx)


def _player_active_near(conn, player_id: int, center_date: str, window_days: int) -> bool:
    d = date.fromisoformat(center_date[:10])
    lo = (d - timedelta(days=window_days)).isoformat()
    hi = (d + timedelta(days=window_days)).isoformat()
    row = conn.execute(
        "SELECT 1 FROM player_stats_cache WHERE player_id = ? AND game_date >= ? AND game_date <= ? LIMIT 1",
        (player_id, lo, hi),
    ).fetchone()
    return row is not None


def build_name_resolution_cache(force_refresh: bool = False) -> dict:
    """Resolve every distinct player_injuries.player_name to a player_id. Caches results.

    Returns coverage summary: {n_distinct_names, n_resolved, coverage_rate, n_ambiguous, n_unmatched}.
    """
    init_cache_db()
    cache = cache_conn()
    if not force_refresh and table_exists(cache, "player_name_resolution"):
        existing = {r["player_name"] for r in cache.execute("SELECT player_name FROM player_name_resolution")}
    else:
        existing = set()

    with injury_conn() as conn:
        rows = conn.execute(
            "SELECT player_name, MIN(game_date) as first_date FROM player_injuries GROUP BY player_name"
        ).fetchall()
    names = [(r["player_name"], r["first_date"]) for r in rows if r["player_name"] not in existing]

    norm_idx, exact_idx = _static_player_index()
    nba_conn = nba_api_conn()

    n_resolved, n_ambiguous, n_unmatched = 0, 0, 0
    inserts = []
    for player_name, first_date in names:
        norm = normalize_name(player_name)
        candidates = norm_idx.get(norm, [])
        method = "normalized"
        if not candidates:
            candidates = exact_idx.get(player_name, [])
            method = "exact_fallback"

        if len(candidates) == 0:
            n_unmatched += 1
            inserts.append((player_name, None, "unmatched", "none"))
        elif len(candidates) == 1:
            n_resolved += 1
            inserts.append((player_name, candidates[0], method, "high"))
        else:
            active = [
                pid for pid in candidates
                if _player_active_near(nba_conn, pid, first_date, DISAMBIGUATION_WINDOW_DAYS)
            ]
            if len(active) == 1:
                n_resolved += 1
                inserts.append((player_name, active[0], f"{method}+disambiguated", "medium"))
            else:
                n_ambiguous += 1
                # Still record a best-effort pick (most recently active candidate) but flag low confidence.
                pick = active[0] if active else candidates[0]
                inserts.append((player_name, pick, f"{method}+ambiguous", "low"))

    nba_conn.close()

    if inserts:
        cache.executemany(
            "INSERT OR REPLACE INTO player_name_resolution "
            "(player_name, player_id, resolution_method, confidence) VALUES (?, ?, ?, ?)",
            inserts,
        )
        cache.commit()
    cache.close()

    total_names_row = None
    with injury_conn() as conn:
        total_names_row = conn.execute(
            "SELECT COUNT(DISTINCT player_name) FROM player_injuries"
        ).fetchone()
    total_distinct = total_names_row[0]

    cache2 = cache_conn()
    high_conf = cache2.execute(
        "SELECT COUNT(*) FROM player_name_resolution WHERE confidence IN ('high','medium')"
    ).fetchone()[0]
    cache2.close()

    coverage_rate = high_conf / total_distinct if total_distinct else 0.0
    summary = {
        "n_distinct_names": total_distinct,
        "n_resolved_this_run": n_resolved,
        "n_ambiguous_this_run": n_ambiguous,
        "n_unmatched_this_run": n_unmatched,
        "coverage_rate": round(coverage_rate, 4),
    }
    logger.info(f"Name resolution coverage: {summary}")
    return summary


# --- Archetype classification (percentile-based, era-adaptive) ---

def _date_to_season_str(d: str) -> str:
    dd = date.fromisoformat(d[:10])
    year = dd.year if dd.month >= 10 else dd.year - 1
    return f"{year}-{str(year + 1)[2:]}"


def _load_player_season_stats() -> pd.DataFrame:
    """Long-format player_stats_cache -> per-player-season averages of PPG/AST/REB/BLK/STL."""
    with nba_api_conn() as conn:
        df = pd.read_sql_query(
            "SELECT player_id, game_date, stat_name, stat_value FROM player_stats_cache", conn
        )
    df["season"] = df["game_date"].map(_date_to_season_str)
    wide = df.pivot_table(
        index=["player_id", "season"], columns="stat_name", values="stat_value", aggfunc="mean"
    ).reset_index()
    wide["n_games"] = df.groupby(["player_id", "season"]).size().reindex(
        pd.MultiIndex.from_frame(wide[["player_id", "season"]])
    ).values
    return wide


def classify_archetypes(min_games: int = 20, percentiles: dict | None = None) -> pd.DataFrame:
    """Percentile-based archetype classification, computed separately per season (era-adaptive).

    percentiles: dict as in style_matchup.archetype_percentiles config (facilitator/scorer/
    rim_protector/perimeter_specialist -> {metric_pct: threshold}).
    Returns a DataFrame with player_id, season, archetype (may be None), and percentile ranks.
    """
    if percentiles is None:
        from src.matchups.config import load_style_matchup_config
        percentiles = load_style_matchup_config()["archetype_percentiles"]

    stats = _load_player_season_stats()
    stats = stats[stats["n_games"] >= min_games].copy()

    for col in ["PPG", "AST", "BLK", "REB", "STL"]:
        if col not in stats.columns:
            stats[col] = 0.0

    stats["ppg_pct"] = stats.groupby("season")["PPG"].rank(pct=True)
    stats["ast_pct"] = stats.groupby("season")["AST"].rank(pct=True)
    stats["blk_pct"] = stats.groupby("season")["BLK"].rank(pct=True)
    stats["reb_pct"] = stats.groupby("season")["REB"].rank(pct=True)
    stats["stl_pct"] = stats.groupby("season")["STL"].rank(pct=True)

    def classify(row) -> str | None:
        fac = percentiles["facilitator"]
        sco = percentiles["scorer"]
        rim = percentiles["rim_protector"]
        per = percentiles["perimeter_specialist"]
        combo = percentiles.get("combo")
        # combo (dual-threat playmaker-scorer) checked first: it is the one archetype
        # that can overlap with a loosened facilitator/scorer band at the margins,
        # and it's the more specific claim (both high) so it takes priority.
        if combo and row["ppg_pct"] >= combo["ppg_pct"] and row["ast_pct"] >= combo["ast_pct"]:
            return "combo"
        if row["ast_pct"] >= fac["ast_pct"] and row["ppg_pct"] <= fac["ppg_pct"]:
            return "facilitator"
        if row["ppg_pct"] >= sco["ppg_pct"] and row["ast_pct"] <= sco["ast_pct"]:
            return "scorer"
        if row["blk_pct"] >= rim["blk_pct"] and row["reb_pct"] >= rim["reb_pct"]:
            return "rim_protector"
        if row["blk_pct"] <= per["blk_pct"] and row["stl_pct"] >= per["stl_pct"]:
            return "perimeter_specialist"
        return None

    stats["archetype"] = stats.apply(classify, axis=1)
    return stats[["player_id", "season", "archetype", "ast_pct", "ppg_pct", "blk_pct", "reb_pct", "stl_pct"]]


def build_archetype_cache(min_games: int = 20) -> dict:
    init_cache_db()
    df = classify_archetypes(min_games=min_games)
    conn = cache_conn()
    rows = [
        (r.player_id, r.season, r.archetype, r.ast_pct, r.ppg_pct, r.blk_pct, r.reb_pct, r.stl_pct)
        for r in df.itertuples(index=False)
    ]
    conn.executemany(
        "INSERT OR REPLACE INTO player_archetypes "
        "(player_id, season, archetype, ast_pct, ppg_pct, blk_pct, reb_pct, stl_pct) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        rows,
    )
    conn.commit()
    conn.close()
    counts = df["archetype"].value_counts(dropna=False).to_dict()
    logger.info(f"Archetype classification: {counts}")
    return {"n_player_seasons": len(df), "archetype_counts": {str(k): v for k, v in counts.items()}}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    print(build_name_resolution_cache())
    print(build_archetype_cache())
