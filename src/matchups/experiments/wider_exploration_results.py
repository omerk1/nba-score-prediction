"""
Orchestrates the Hyperparameter Search & Alternative Methods stage's four
comparisons (Optuna hyperparameter search, cluster-bucket search, PCA encoding,
supervised model) and writes their results into outputs/a7_style_matchup_results.csv,
extending its schema (add new columns for new information rather than overload
existing ones) with:
    method_family        - lookup-cosine | lookup-knn | lookup-cluster |
                            encoding-pca | supervised-model
    split                - train | validation
    fingerprint_window   - the window used for this row (was implicitly 20 before)
    decay_halflife       - the half-life used for this row (was implicitly 5 before)

Existing rows from the previous run (Phases 0-4) are preserved as-is; this module
only ADDS the new columns (blank for old rows) and appends new rows -- it never
deletes or rewrites the meaning of prior data, per the safety instructions.
"""

import logging
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from src.matchups.experiments.cluster_bucket_similarity_search import build_cluster_index
from src.matchups.config import PROJECT_ROOT
from src.matchups.experiments.encoding_pca import build_pca_matchup_index
from src.matchups.split import get_split_dates
from src.matchups.experiments.supervised_model_baseline import run_supervised_model
from src.matchups.tuning import evaluate_config, load_constants, run_search_inmemory

logger = logging.getLogger(__name__)

RESULTS_CSV = PROJECT_ROOT / "outputs" / "a7_style_matchup_results.csv"

FIELDNAMES = [
    "timestamp", "run_name", "encoding_phase", "similarity_method",
    "similarity_threshold_or_k", "layers_enabled", "n_games_evaluated",
    "fallback_rate", "mean_confidence", "corr_style_vs_margin",
    "mae_high_conf", "mae_low_conf", "corr_a2_alone", "corr_a7_alone",
    "corr_a2_plus_a7", "notes",
    "method_family", "split", "fingerprint_window", "decay_halflife",
]

# Best config found by the Optuna joint search this run (src/matchups/tuning.py,
# run_optuna_search(n_trials=40, layer=2, seed=42) — see docs/A7_PHASE_LOG.md for
# the full trace). Hardcoded here (rather than re-running the ~4-minute search) so
# this module can be re-run cheaply to regenerate CSV rows / rerun evaluation only.
BEST_HP_CONFIG = {
    "fingerprint_window": 37,
    "decay_halflife": 13.199390932957819,
    "similarity_method": "knn",
    "knn_k": 81,
    "min_confidence_sample": 21,
    "full_confidence_sample": 82,
    "similarity_threshold": 0.7,  # unused (method=knn) but kept for evaluate_config's signature
}
DEFAULT_HP_CONFIG = {
    "fingerprint_window": 20,
    "decay_halflife": 5.0,
    "similarity_method": "cosine",
    "similarity_threshold": 0.70,
    "knn_k": 30,
    "min_confidence_sample": 10,
    "full_confidence_sample": 50,
}


def _combined_a2_a7_corr(h2h: np.ndarray, style: np.ndarray, margin: np.ndarray) -> float:
    X = np.column_stack([np.ones_like(h2h), h2h, style])
    coefs, *_ = np.linalg.lstsq(X, margin, rcond=None)
    fitted = X @ coefs
    return float(np.corrcoef(fitted, margin)[0, 1])


def _full_row(
    run_name: str, method_family: str, split_name: str, df: pd.DataFrame, h2h: pd.DataFrame,
    encoding_phase, similarity_method: str, threshold_or_k, layers_enabled: str,
    fingerprint_window, decay_halflife, notes: str,
) -> dict:
    df = df.drop(columns=["h2h_score"], errors="ignore")
    merged = df.merge(h2h[["game_id", "h2h_score"]], on="game_id", how="left")
    merged["h2h_score"] = merged["h2h_score"].fillna(0.0)
    corr_a7 = float(merged["style_score"].corr(merged["actual_home_margin"])) if merged["style_score"].std() > 0 else 0.0
    corr_a2 = float(merged["h2h_score"].corr(merged["actual_home_margin"]))
    corr_combined = _combined_a2_a7_corr(
        merged["h2h_score"].to_numpy(), merged["style_score"].to_numpy(), merged["actual_home_margin"].to_numpy()
    )
    fallback_rate = round(float(df["fallback_used"].mean()), 4) if "fallback_used" in df.columns else ""
    mean_confidence = round(float(df["confidence"].mean()), 4) if "confidence" in df.columns else ""
    mae_high, mae_low = "", ""
    if "confidence" in df.columns:
        high = df[df["confidence"] >= 0.5]
        low = df[df["confidence"] < 0.5]
        mae_high = round(float((high["style_score"] - high["actual_home_margin"]).abs().mean()), 4) if len(high) else ""
        mae_low = round(float((low["style_score"] - low["actual_home_margin"]).abs().mean()), 4) if len(low) else ""
    return {
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "run_name": run_name,
        "encoding_phase": encoding_phase,
        "similarity_method": similarity_method,
        "similarity_threshold_or_k": threshold_or_k,
        "layers_enabled": layers_enabled,
        "n_games_evaluated": len(df),
        "fallback_rate": fallback_rate,
        "mean_confidence": mean_confidence,
        "corr_style_vs_margin": round(corr_a7, 4),
        "mae_high_conf": mae_high,
        "mae_low_conf": mae_low,
        "corr_a2_alone": round(corr_a2, 4),
        "corr_a7_alone": round(corr_a7, 4),
        "corr_a2_plus_a7": round(corr_combined, 4),
        "notes": notes,
        "method_family": method_family,
        "split": split_name,
        "fingerprint_window": fingerprint_window,
        "decay_halflife": decay_halflife,
    }


def _append_rows(rows: list[dict]) -> None:
    if RESULTS_CSV.exists():
        existing = pd.read_csv(RESULTS_CSV, dtype=str)
    else:
        existing = pd.DataFrame(columns=FIELDNAMES)
    for col in FIELDNAMES:
        if col not in existing.columns:
            existing[col] = ""
    new_df = pd.DataFrame(rows)
    for col in FIELDNAMES:
        if col not in new_df.columns:
            new_df[col] = ""
    combined = pd.concat([existing[FIELDNAMES], new_df[FIELDNAMES]], ignore_index=True)
    combined.to_csv(RESULTS_CSV, index=False)
    logger.info(f"Wrote {len(rows)} new rows to {RESULTS_CSV} (total now {len(combined)})")


def build_all_rows() -> list[dict]:
    consts = load_constants()
    h2h = consts["h2h"]
    splits = get_split_dates()
    split_ranges = {
        "train": (splits["train_start"], splits["train_end"]),
        "validation": (splits["validation_start"], splits["validation_end"]),
    }
    rows = []

    # --- Hyperparameter search comparison: default vs. Optuna-found best, both splits ---
    for label, cfg, family in [
        ("hpsearch_default", DEFAULT_HP_CONFIG, f"lookup-{DEFAULT_HP_CONFIG['similarity_method']}"),
        ("hpsearch_best", BEST_HP_CONFIG, f"lookup-{BEST_HP_CONFIG['similarity_method']}"),
    ]:
        for split_name, (s, e) in split_ranges.items():
            result = evaluate_config(
                consts, window=cfg["fingerprint_window"], halflife=cfg["decay_halflife"],
                method=cfg["similarity_method"], threshold=cfg["similarity_threshold"], k=cfg["knn_k"],
                min_confidence_sample=cfg["min_confidence_sample"], full_confidence_sample=cfg["full_confidence_sample"],
                eval_start=s, eval_end=e, layer=2,
            )
            notes = (
                "Item #1: Optuna joint hyperparameter search (40 trials, TPE, seed=42) over "
                "fingerprint_window/decay_halflife/similarity_method+threshold-or-k/"
                "min_confidence_sample/full_confidence_sample, selected on TRAIN split only."
                if label == "hpsearch_best" else
                "Item #1: previous run's hand-picked default config, re-evaluated on the "
                "train/validation split (guardrail #4) for apples-to-apples comparison with hpsearch_best."
            )
            rows.append(_full_row(
                f"{label}_{split_name}", family, split_name, result["df"], h2h,
                encoding_phase=1, similarity_method=cfg["similarity_method"],
                threshold_or_k=cfg["knn_k"] if cfg["similarity_method"] == "knn" else cfg["similarity_threshold"],
                layers_enabled="L1+L2+L3", fingerprint_window=cfg["fingerprint_window"],
                decay_halflife=round(cfg["decay_halflife"], 4), notes=notes,
            ))

    # --- PCA encoding, default similarity params (window=20, halflife=5), both splits ---
    pca_built = build_pca_matchup_index(window=20, halflife=5.0, layer=2)
    pca_idx = pca_built["idx"]
    for split_name, (s, e) in split_ranges.items():
        df = run_search_inmemory(
            pca_idx, h2h, method="cosine", threshold=0.70, k=30,
            min_confidence_sample=10, full_confidence_sample=50, eval_start=s, eval_end=e,
        )
        notes = (
            f"Item #2a: PCA encoding (11 raw box-score metrics -> 5 components via "
            f"StandardScaler+PCA fit on TRAIN split only; explained_variance_ratio="
            f"{[round(v, 3) for v in pca_built['explained_variance_ratio']]}), same cosine@0.70 "
            f"search params as the hand-picked default, injury-adjusted (layer=2)."
        )
        rows.append(_full_row(
            f"pca_encoding_{split_name}", "encoding-pca", split_name, df, h2h,
            encoding_phase=2, similarity_method="cosine", threshold_or_k=0.70,
            layers_enabled="L1+L2+L3", fingerprint_window=20, decay_halflife=5.0, notes=notes,
        ))

    # --- Clustering bucket-lookup (k=8), default window/halflife, both splits ---
    cluster_built = build_cluster_index(window=20, halflife=5.0, layer=2, n_clusters=8)
    cluster_df = cluster_built["df"]
    for split_name, (s, e) in split_ranges.items():
        sub = cluster_df[(cluster_df["game_date"] >= s) & (cluster_df["game_date"] <= e)]
        notes = (
            f"Item #2b: KMeans (k=8) archetype-pair bucket lookup (leakage-safe expanding mean "
            f"per (home_cluster,away_cluster) pair, centroids fit on TRAIN split only), "
            f"injury-adjusted (layer=2). Alternative SEARCH method, not per-vector cosine/KNN. "
            f"{cluster_built['n_no_bucket_history']} games had no prior history for their bucket "
            f"(fell back to 0.0)."
        )
        rows.append(_full_row(
            f"clustering_k8_{split_name}", "lookup-cluster", split_name, sub, h2h,
            encoding_phase=1, similarity_method="cluster_bucket", threshold_or_k=8,
            layers_enabled="L1+L2+L3(bucket)", fingerprint_window=20, decay_halflife=5.0, notes=notes,
        ))

    # --- Supervised catboost model on the 10-dim matchup vector, both splits ---
    sup = run_supervised_model(window=20, halflife=5.0, layer=2)
    for split_name, df in [("train", sup["train_pred_df"]), ("validation", sup["val_pred_df"])]:
        notes = (
            f"Item #3: CatBoostRegressor (depth=4, lr=0.05, iterations<=300, early-stopped at "
            f"iteration {sup['best_iteration']} on an internal chronological dev slice carved from "
            f"the tail of TRAIN -- never touches true validation during fitting), direct regression "
            f"on the 10-dim injury-adjusted (layer=2) matchup vector, NOT a lookup-and-average method."
        )
        rows.append(_full_row(
            f"supervised_catboost_{split_name}", "supervised-model", split_name, df, h2h,
            encoding_phase=1, similarity_method="none (direct regression)", threshold_or_k="",
            layers_enabled="L1+L2 (no L3 search)", fingerprint_window=20, decay_halflife=5.0, notes=notes,
        ))

    _append_rows(rows)
    return rows


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    result_rows = build_all_rows()
    for r in result_rows:
        print(r["run_name"], "|", r["split"], "| corr_a7=", r["corr_a7_alone"], "| corr_a2=", r["corr_a2_alone"])
