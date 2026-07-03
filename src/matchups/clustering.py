"""
Item #2b: team-style clustering as an alternative SEARCH method (design doc
Encoding Phase 3), tried unconditionally this run alongside cosine/KNN lookup and
PCA encoding -- not gated behind "only if cosine/KNN return thin samples" (they
don't; fallback rate is ~0%), because the instructions ask for a genuine method
comparison, not just an escape hatch.

Method: fit KMeans (k=8, per the design doc's example) on the injury-adjusted
(layer=2, per coordinator clarification -- must be comparable to the 0.281
baseline) hand-picked 5-dim per-team-game style vectors, restricted to TRAIN-split
team-games only (avoids leakage from validation-range rows influencing the cluster
centroids). Every team-game (train + validation) is then assigned a cluster id via
`.predict()` on the already-fitted centroids. A game's score is the leakage-safe
historical average actual_home_margin for its (home_cluster, away_cluster) pair --
NOT a per-vector cosine/KNN search, a discrete bucket lookup instead, exactly as
Encoding Phase 3 describes ("archetype A vs archetype B -> look up historical
outcomes for that pair directly").

Leakage discipline for the bucket lookup itself: this is NOT simply an expanding
mean shifted by one row, because sorting only by date and using .shift(1) within a
groupby does not exclude OTHER games on the same date within the same cluster-pair
bucket (multiple games can share both a date and a cluster-pair). Instead, for
every row, only bucket-mates with game_date STRICTLY BEFORE the target's date are
averaged, via the same np.searchsorted-per-group pattern used everywhere else in
this module (similarity.py's date-based exclusion is the reference implementation
this mirrors, just applied per-cluster-pair-group instead of over the whole
history array).
"""

import logging

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from src.matchups.fingerprint import FINGERPRINT_METRICS
from src.matchups.split import get_split_dates
from src.matchups.tuning import apply_injury_deltas, build_fingerprints_inmemory, load_constants

logger = logging.getLogger(__name__)

N_CLUSTERS = 8


def _zscore(fp: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = fp.copy()
    for c in cols:
        mu, sd = out[c].mean(), out[c].std()
        out[c] = (out[c] - mu) / (sd if sd > 1e-9 else 1.0)
    return out


def _leakage_safe_bucket_mean(df: pd.DataFrame, group_col: str, date_col: str, value_col: str) -> np.ndarray:
    """For every row, the mean of `value_col` over all OTHER rows in the same
    `group_col` bucket with `date_col` strictly before this row's date. Mirrors
    similarity.py's np.searchsorted exclusion (all same-date rows excluded, not
    just earlier row positions), applied per-bucket instead of over full history."""
    out = np.full(len(df), np.nan)
    for _, g in df.groupby(group_col):
        g = g.sort_values(date_col)
        idxs = g.index.to_numpy()
        dates = g[date_col].to_numpy()
        vals = g[value_col].to_numpy()
        for i in range(len(g)):
            end_pos = int(np.searchsorted(dates, dates[i], side="left"))
            out[idxs[i]] = vals[:end_pos].mean() if end_pos > 0 else np.nan
    return out


def build_cluster_index(window: int = 20, halflife: float = 5.0, layer: int = 2, n_clusters: int = N_CLUSTERS) -> dict:
    consts = load_constants()
    fp1 = build_fingerprints_inmemory(consts["raw"], window=window, halflife=halflife)
    fp = apply_injury_deltas(fp1, consts["delta_lookup"]) if layer == 2 else fp1
    fp = _zscore(fp, FINGERPRINT_METRICS)

    splits = get_split_dates()
    train_mask = (fp["game_date"] >= splits["train_start"]) & (fp["game_date"] <= splits["train_end"])

    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    km.fit(fp.loc[train_mask, FINGERPRINT_METRICS].to_numpy())
    fp["cluster"] = km.predict(fp[FINGERPRINT_METRICS].to_numpy())

    games = consts["games"]
    home = fp[["game_id", "team_id", "cluster"]].rename(columns={"team_id": "th", "cluster": "home_cluster"})
    away = fp[["game_id", "team_id", "cluster"]].rename(columns={"team_id": "ta", "cluster": "away_cluster"})
    merged = games.merge(home, left_on=["game_id", "team_id_home"], right_on=["game_id", "th"], how="inner").drop(columns=["th"])
    merged = merged.merge(away, left_on=["game_id", "team_id_away"], right_on=["game_id", "ta"], how="inner").drop(columns=["ta"])
    merged["cluster_pair"] = merged["home_cluster"].astype(str) + "_" + merged["away_cluster"].astype(str)
    merged = merged.sort_values("game_date").reset_index(drop=True)

    merged["style_score"] = _leakage_safe_bucket_mean(merged, "cluster_pair", "game_date", "actual_home_margin")
    # Fallback: cluster pairs with no prior history at all (e.g. first-ever occurrence)
    # fall back to 0 (league-average-ish; no A2 h2h merge here since this bucket-lookup
    # method is evaluated as its own standalone alternative-search comparison).
    n_no_bucket_history = int(merged["style_score"].isna().sum())
    merged["style_score"] = merged["style_score"].fillna(0.0)

    cluster_sizes = fp.loc[train_mask, "cluster"].value_counts().to_dict()
    return {
        "df": merged[["game_id", "game_date", "actual_home_margin", "style_score", "cluster_pair"]],
        "n_clusters": n_clusters,
        "cluster_sizes_train": {int(k): int(v) for k, v in cluster_sizes.items()},
        "n_no_bucket_history": n_no_bucket_history,
    }


def evaluate_clustering(window: int = 20, halflife: float = 5.0, layer: int = 2, n_clusters: int = N_CLUSTERS) -> dict:
    built = build_cluster_index(window=window, halflife=halflife, layer=layer, n_clusters=n_clusters)
    df = built["df"]
    splits = get_split_dates()

    out = {}
    for split_name, (s, e) in {
        "train": (splits["train_start"], splits["train_end"]),
        "validation": (splits["validation_start"], splits["validation_end"]),
    }.items():
        sub = df[(df["game_date"] >= s) & (df["game_date"] <= e)]
        corr = float(sub["style_score"].corr(sub["actual_home_margin"])) if len(sub) else 0.0
        out[split_name] = {"corr": corr, "n_games": len(sub)}
    out["n_clusters"] = built["n_clusters"]
    out["cluster_sizes_train"] = built["cluster_sizes_train"]
    out["n_no_bucket_history"] = built["n_no_bucket_history"]
    return out


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    import json
    print(json.dumps(evaluate_clustering(), indent=2, default=str))
