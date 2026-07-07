"""
Item #1 (wrap-up round): re-run the player-archetype KMeans clustering exploration
(originally done ad hoc in Phase 0, not persisted as a module -- see phase log) now
that minutes_per_game/usage_rate (from player_importance, injury_features.sqlite) are
joined into players._load_player_season_stats().

Phase 0's finding: clustering on [PPG, AST, REB, BLK, STL, FG%] alone mostly recovered
a playing-time/usage tier split (a monotonic "more of everything" axis), not stylistic
groups, because that stat set has no minutes/usage data to separate "how much" from
"how". This module checks whether adding minutes_per_game/usage_rate fixes that, and
-- per the explicit instruction to not stop at one negative result if a fix is easy to
reason about -- also tries a PER-MINUTE-normalized variant (rate stats instead of raw
counting stats + a separate minutes feature) if the raw-plus-usage variant doesn't
clearly separate by style either.

Two feature-set variants:
    raw_plus_usage       : [PPG, AST, REB, BLK, STL, minutes_per_game, usage_rate]
                            (Phase 0's 5 raw box counting stats + the 2 new columns,
                            concatenated -- the direct/obvious way to add "how much")
    per_minute_plus_usage: [PPG/min, AST/min, REB/min, BLK/min, STL/min, usage_rate]
                            (counting stats converted to PER-MINUTE rates, removing
                            playing time as a raw scaling factor entirely, since a
                            bench player and a starter with the same per-minute
                            production profile should cluster together; usage_rate is
                            already a rate stat and kept as-is)

For each variant and each k in {4,5,6,8}, this reports the FULL per-cluster raw-unit
centroid table (not just a correlation number) so a human can visually judge whether
clusters read as stylistic groups or playing-time tiers, plus a simple diagnostic:
the correlation between a cluster's rank-by-mean-minutes and its rank-by-mean-PPG
(near +/-1 => still a monotonic "more of everything" ordering; closer to 0 => clusters
are cutting across playing-time levels, i.e. actually stylistic).
"""

import logging

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from src.matchups.players import _load_player_season_stats

logger = logging.getLogger(__name__)

RAW_COLS = ["PPG", "AST", "REB", "BLK", "STL"]
K_GRID = [4, 5, 6, 8]
MIN_GAMES = 20
MIN_MINUTES_FOR_RATE = 5.0  # guard against divide-by-near-zero for per-minute variant


def _prep_base(min_games: int = MIN_GAMES) -> pd.DataFrame:
    stats = _load_player_season_stats()
    stats = stats[stats["n_games"] >= min_games].copy()
    for c in RAW_COLS:
        if c not in stats.columns:
            stats[c] = 0.0
    return stats


def build_variant(variant: str, min_games: int = MIN_GAMES) -> pd.DataFrame:
    """Returns stats df restricted to rows with non-null minutes/usage, with the
    variant's feature columns added (prefixed `feat_`)."""
    stats = _prep_base(min_games=min_games)
    stats = stats[stats["minutes_per_game"].notna() & stats["usage_rate"].notna()].copy()

    if variant == "raw_plus_usage":
        feat_cols = RAW_COLS + ["minutes_per_game", "usage_rate"]
        for c in feat_cols:
            stats[f"feat_{c}"] = stats[c]
    elif variant == "per_minute_plus_usage":
        stats = stats[stats["minutes_per_game"] >= MIN_MINUTES_FOR_RATE].copy()
        for c in RAW_COLS:
            stats[f"feat_{c}_per_min"] = stats[c] / stats["minutes_per_game"]
        stats["feat_usage_rate"] = stats["usage_rate"]
        feat_cols = [f"{c}_per_min" for c in RAW_COLS] + ["usage_rate"]
    else:
        raise ValueError(f"Unknown variant: {variant}")

    stats.attrs["feat_cols"] = feat_cols
    return stats


def run_clustering(variant: str, k: int, min_games: int = MIN_GAMES) -> dict:
    stats = build_variant(variant, min_games=min_games)
    feat_cols = stats.attrs["feat_cols"]
    feat_matrix_cols = [f"feat_{c}" for c in feat_cols]

    X = stats[feat_matrix_cols].to_numpy(dtype=np.float64)
    Xs = StandardScaler().fit_transform(X)

    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    stats = stats.copy()
    stats["cluster"] = km.fit_predict(Xs)

    # Per-cluster RAW-unit centroid table (not standardized) -- includes minutes_per_game
    # and PPG explicitly regardless of variant, for the monotonicity diagnostic below.
    report_cols = list(dict.fromkeys(RAW_COLS + ["minutes_per_game", "usage_rate"]))
    centroids = stats.groupby("cluster")[report_cols].mean()
    centroids["n_player_seasons"] = stats.groupby("cluster").size()
    centroids = centroids.sort_values("minutes_per_game")

    # Monotonicity diagnostic: rank-correlation between cluster-mean-minutes and
    # cluster-mean-PPG. Close to 1.0 => clusters are still ~= playing-time tiers on
    # the PPG axis specifically. Reported for EVERY report_col (not just PPG) below,
    # since the real question is whether minutes explains BLK/REB/AST/STL shape just
    # as tightly as it explains PPG -- if so, "style" hasn't actually separated from
    # "how much"; if minutes correlates much more weakly with, say, BLK or AST than
    # with PPG/usage_rate, that is evidence style IS emerging as a second axis.
    rank_minutes = centroids["minutes_per_game"].rank()
    monotonicity_by_col = {
        c: float(rank_minutes.corr(centroids[c].rank(), method="spearman")) if k > 1 else np.nan
        for c in report_cols if c != "minutes_per_game"
    }

    return {
        "variant": variant,
        "k": k,
        "n_player_seasons": len(stats),
        "feat_cols": feat_cols,
        "centroids": centroids,
        "monotonicity_minutes_vs_ppg": monotonicity_by_col["PPG"],
        "monotonicity_by_col": monotonicity_by_col,
    }


def run_all() -> list[dict]:
    results = []
    for variant in ["raw_plus_usage", "per_minute_plus_usage"]:
        for k in K_GRID:
            r = run_clustering(variant, k)
            results.append(r)
            logger.info(
                f"[{variant}] k={k}: n={r['n_player_seasons']} "
                f"monotonicity(minutes vs ppg)={r['monotonicity_minutes_vs_ppg']:.3f}"
            )
    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    pd.set_option("display.width", 160)
    pd.set_option("display.max_columns", 20)
    for r in run_all():
        mono_str = ", ".join(f"{c}={v:.2f}" for c, v in r["monotonicity_by_col"].items())
        print(f"\n=== variant={r['variant']} k={r['k']} n={r['n_player_seasons']} ===")
        print(f"monotonicity(minutes vs X): {mono_str}")
        print(r["centroids"].round(3).to_string())
