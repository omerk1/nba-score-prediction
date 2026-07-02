"""
Phase 0: empirically calibrate Layer 2 injury-impact deltas (design doc "v2").

For each archetype, compare a team's Layer-1 style fingerprint in games where a
player of that archetype was reported Out (via the pre-game NBA official injury
PDF backfill — see run_historical() in src/news_scraping/pipeline.py, which is
pre-game reporting, so this is valid for both backtesting and live prediction;
this was already resolved and is not re-investigated here) vs. that same
team-season's baseline (games where the archetype was NOT missing).

Scope is restricted to game_dates present in scrape_log (source='pdf') so that
"no rows in player_injuries for this team/date" reliably means "confirmed no one
out" rather than "not scraped that day".
"""

import logging

import pandas as pd

from src.matchups.config import CONFIG_PATH, load_style_matchup_config
from src.matchups.db import cache_conn, init_cache_db, injury_conn
from src.matchups.fingerprint import FINGERPRINT_METRICS, build_fingerprint_cache
from src.matchups.players import _date_to_season_str, build_archetype_cache, build_name_resolution_cache

logger = logging.getLogger(__name__)

# Only these metrics matter per archetype (design doc injury_impact block) —
# calibration still computes all 5 for transparency but only these feed Layer 2.
TARGET_METRICS = {
    "facilitator": ["assist_rate", "pace_score"],
    "scorer": ["three_pt_reliance", "paint_activity"],
    # combo is new (added in Phase 0 archetype exploration, not in the design doc) —
    # no prior assumption about which metric it should move, so calibrate against
    # all 5 fingerprint metrics and let the data decide.
    "combo": list(FINGERPRINT_METRICS),
    "rim_protector": ["defensive_rating", "paint_activity"],
    "perimeter_specialist": ["defensive_rating"],
}


def _scraped_dates() -> set[str]:
    with injury_conn() as conn:
        rows = conn.execute("SELECT DISTINCT game_date FROM scrape_log WHERE source = 'pdf'").fetchall()
    return {r["game_date"][:10] for r in rows}


def _out_players_by_team_date() -> pd.DataFrame:
    """player_injuries Out rows, resolved to player_id + archetype."""
    with injury_conn() as conn:
        inj = pd.read_sql_query(
            "SELECT game_date, team_id, player_name FROM player_injuries WHERE status = 'Out'", conn
        )
    cache = cache_conn()
    res = pd.read_sql_query(
        "SELECT player_name, player_id, confidence FROM player_name_resolution "
        "WHERE confidence IN ('high','medium')",
        cache,
    )
    arch = pd.read_sql_query("SELECT player_id, season, archetype FROM player_archetypes", cache)
    cache.close()

    inj = inj.merge(res, on="player_name", how="inner")
    inj["season"] = inj["game_date"].map(_date_to_season_str)
    inj = inj.merge(arch, on=["player_id", "season"], how="left")
    inj = inj[inj["archetype"].notna()]
    return inj[["game_date", "team_id", "archetype"]].drop_duplicates()


def run_calibration(min_games: int = 5) -> dict:
    init_cache_db()
    name_summary = build_name_resolution_cache()
    build_archetype_cache()
    build_fingerprint_cache(layer=1, min_games=min_games)

    scraped = _scraped_dates()
    out_map = _out_players_by_team_date()
    out_map = out_map[out_map["game_date"].str[:10].isin(scraped)]

    conn = cache_conn()
    fp = pd.read_sql_query(
        "SELECT game_id, team_id, game_date, " + ", ".join(FINGERPRINT_METRICS) +
        " FROM matchup_fingerprints WHERE layer = 1",
        conn,
    )
    conn.close()
    fp = fp[fp["game_date"].str[:10].isin(scraped)].copy()
    fp["season"] = fp["game_date"].map(_date_to_season_str)
    fp["team_season"] = fp["team_id"].astype(str) + "_" + fp["season"]

    missing_set = set(zip(out_map["game_date"].str[:10], out_map["team_id"]))
    archetype_by_key: dict[tuple, set[str]] = {}
    for gd, tid, arch in out_map.itertuples(index=False):
        archetype_by_key.setdefault((gd[:10], tid), set()).add(arch)

    results = []
    coverage_note = name_summary["coverage_rate"]
    low_conf_flag = coverage_note < 0.8

    for archetype, metrics in TARGET_METRICS.items():
        missing_keys = {k for k, archs in archetype_by_key.items() if archetype in archs}
        fp["_missing_this_archetype"] = fp.apply(
            lambda r: (r["game_date"][:10], r["team_id"]) in missing_keys, axis=1
        )

        missing_rows = fp[fp["_missing_this_archetype"]]
        # Baseline: same team-seasons, games where this archetype was NOT missing.
        relevant_team_seasons = missing_rows["team_season"].unique()
        baseline_rows = fp[
            (~fp["_missing_this_archetype"]) & (fp["team_season"].isin(relevant_team_seasons))
        ]

        for metric in metrics:
            n_without = len(missing_rows)
            n_baseline = len(baseline_rows)
            if n_without == 0 or n_baseline == 0:
                delta = 0.0
            else:
                delta = float(missing_rows[metric].mean() - baseline_rows[metric].mean())
            results.append({
                "archetype": archetype, "metric": metric, "delta": round(delta, 4),
                "n_games_without": n_without, "n_games_baseline": n_baseline,
            })

    conn = cache_conn()
    conn.executemany(
        "INSERT OR REPLACE INTO injury_calibration (archetype, metric, delta, n_games_without, n_games_baseline) "
        "VALUES (:archetype, :metric, :delta, :n_games_without, :n_games_baseline)",
        results,
    )
    conn.commit()
    conn.close()

    summary = {
        "name_resolution_coverage_rate": coverage_note,
        "low_confidence": low_conf_flag,
        "deltas": results,
    }
    logger.info(f"Phase 0 calibration results: {summary}")
    return summary


def write_deltas_to_config(results: list[dict]) -> None:
    """Append a `style_matchup` block to configs/config.yaml with the empirical deltas.

    IMPORTANT: this appends a plain-text block rather than round-tripping the file
    through yaml.safe_load/safe_dump. A load+dump round-trip strips every inline
    `#` comment in the existing file (PyYAML doesn't preserve comments) and
    reformats quoting/indentation — that would silently destroy human-authored
    documentation throughout config.yaml. Since style_matchup does not exist yet
    in the file, a straight text append is safe and idempotent-checked below.
    """
    with open(CONFIG_PATH) as f:
        existing = f.read()

    if "style_matchup:" in existing:
        logger.info("style_matchup block already present in config.yaml — skipping append")
        return

    impact: dict = {}
    for row in results:
        impact.setdefault(row["archetype"], {})[row["metric"]] = row["delta"]

    lines = [
        "",
        "style_matchup:",
        "  # A7 exploratory module (src/matchups/). Not read by config_loader.py's Config",
        "  # schema (not integrated into feature_builder yet) — parsed directly by",
        "  # src/matchups/config.py. See docs/a7_style_matchup_design.md and",
        "  # docs/A7_PHASE_LOG.md for how these values were chosen.",
        "  fingerprint_window: 20",
        "  decay_halflife: 5",
        "  encoding: hand_picked",
        "  similarity_method: cosine   # cosine | knn -- see Phase 3 findings in phase log",
        "  similarity_threshold: 0.70",
        "  knn_k: 30",
        "  min_confidence_sample: 10",
        "  full_confidence_sample: 50",
        "  low_confidence_fallback: h2h",
        "  archetype_method: percentile",
        "  injury_impact_calibrated: true  # values below are Phase-0 empirical deltas, not v1 guesses",
        "  injury_impact:",
    ]
    for archetype, metrics in impact.items():
        lines.append(f"    {archetype}:")
        if not metrics:
            lines.append("      {}")
        for metric, delta in metrics.items():
            lines.append(f"      {metric}: {delta}")

    with open(CONFIG_PATH, "a") as f:
        f.write("\n".join(lines) + "\n")
    logger.info("Appended calibrated injury_impact deltas to configs/config.yaml")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    result = run_calibration()
    write_deltas_to_config(result["deltas"])
    print(result)
