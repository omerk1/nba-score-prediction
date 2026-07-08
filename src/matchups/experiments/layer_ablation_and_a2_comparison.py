"""
Phase 4: Validation + ablation CSV — the most important A7 deliverable.

Writes one row per ablation configuration to outputs/a7_style_matchup_results.csv
(DictWriter, header written if the file doesn't exist yet — same pattern as
train_model.py:_save_experiment / tune_elo.py's outputs/elo_tuning_results.csv).
Kept as a standalone file from outputs/experiments.csv since that one is for
trained-model runs and this validates the raw signal before it enters a model.

Layer ablation (Layer 1 only / Layer 1+2 / Layer 1+2+3), per the design doc:
  - "Layer 1 only" / "Layer 1+2" are evaluated WITHOUT a similarity search — a
    zero-parameter naive score (sum of the 5 z-scored home-away metric diffs)
    on the layer=1 (no injury adjustment) or layer=2 (injury-adjusted)
    fingerprints. This isolates what each layer's fingerprint construction adds
    on its own, before Layer 3's search is introduced.
  - "Layer 1+2+3" is the full pipeline: injury-adjusted (layer=2) fingerprints
    fed through Layer 3's similarity search (both cosine and knn are reported,
    per the pre-settled "try both, keep the better one" instruction).

A2/A7 comparison: corr_a2_alone (h2h_score vs actual margin), corr_a7_alone
(style_score vs actual margin), corr_a2_plus_a7 (OLS combination of both vs
actual margin, correlation of fitted vs actual — tests whether A7 adds signal on
top of A2 rather than just overlapping with it).
"""

import csv
import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src.matchups.baseline_a2 import build_a2_h2h_scores
from src.matchups.config import PROJECT_ROOT, load_style_matchup_config
from src.matchups.matchup_index import FINGERPRINT_METRICS, build_matchup_index
from src.matchups.similarity import run_similarity_search

logger = logging.getLogger(__name__)

RESULTS_CSV = PROJECT_ROOT / "outputs" / "a7_style_matchup_results.csv"

FIELDNAMES = [
    "timestamp", "run_name", "encoding_phase", "similarity_method",
    "similarity_threshold_or_k", "layers_enabled", "n_games_evaluated",
    "fallback_rate", "mean_confidence", "corr_style_vs_margin",
    "mae_high_conf", "mae_low_conf", "corr_a2_alone", "corr_a7_alone",
    "corr_a2_plus_a7", "notes",
]


def _naive_diff_score(layer: int, eval_start_date: str) -> pd.DataFrame:
    """Zero-parameter baseline: sum of z-scored home-away metric diffs, no search."""
    idx = build_matchup_index(layer=layer)
    idx = idx[idx["game_date"] >= eval_start_date].copy()
    diff_cols = []
    for m in FINGERPRINT_METRICS:
        col = f"_diff_{m}"
        idx[col] = idx[f"home_{m}"] - idx[f"away_{m}"]
        diff_cols.append(col)
    idx["style_score"] = idx[diff_cols].sum(axis=1)
    return idx[["game_id", "game_date", "actual_home_margin", "style_score"]]


def _combined_a2_a7_corr(h2h: np.ndarray, style: np.ndarray, margin: np.ndarray) -> float:
    """OLS fit margin ~ h2h + style, correlation of fitted vs actual (tests whether
    A7 adds signal on top of A2, not just overlap)."""
    X = np.column_stack([np.ones_like(h2h), h2h, style])
    coefs, *_ = np.linalg.lstsq(X, margin, rcond=None)
    fitted = X @ coefs
    return float(np.corrcoef(fitted, margin)[0, 1])


def _mae_by_confidence(df: pd.DataFrame, split: float = 0.5) -> tuple[float, float]:
    high = df[df["confidence"] >= split]
    low = df[df["confidence"] < split]
    mae_high = float((high["style_score"] - high["actual_home_margin"]).abs().mean()) if len(high) else float("nan")
    mae_low = float((low["style_score"] - low["actual_home_margin"]).abs().mean()) if len(low) else float("nan")
    return mae_high, mae_low


def _row_from_naive(run_name: str, layer: int, df: pd.DataFrame, h2h: pd.DataFrame, notes: str) -> dict:
    merged = df.merge(h2h[["game_id", "h2h_score"]], on="game_id", how="left")
    merged["h2h_score"] = merged["h2h_score"].fillna(0.0)
    corr_a7 = merged["style_score"].corr(merged["actual_home_margin"])
    corr_a2 = merged["h2h_score"].corr(merged["actual_home_margin"])
    corr_combined = _combined_a2_a7_corr(
        merged["h2h_score"].to_numpy(), merged["style_score"].to_numpy(), merged["actual_home_margin"].to_numpy()
    )
    return {
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "run_name": run_name,
        "encoding_phase": 1,
        "similarity_method": "none (naive diff sum, no search)",
        "similarity_threshold_or_k": "",
        "layers_enabled": "L1" if layer == 1 else "L1+L2",
        "n_games_evaluated": len(merged),
        "fallback_rate": 0.0,
        "mean_confidence": "",
        "corr_style_vs_margin": round(corr_a7, 4),
        "mae_high_conf": "",
        "mae_low_conf": "",
        "corr_a2_alone": round(corr_a2, 4),
        "corr_a7_alone": round(corr_a7, 4),
        "corr_a2_plus_a7": round(corr_combined, 4),
        "notes": notes,
    }


def _row_from_similarity(run_name: str, method: str, param_label: str, df: pd.DataFrame, notes: str) -> dict:
    corr_a7 = df["style_score"].corr(df["actual_home_margin"])
    corr_a2 = df["h2h_score"].corr(df["actual_home_margin"])
    corr_combined = _combined_a2_a7_corr(
        df["h2h_score"].to_numpy(), df["style_score"].to_numpy(), df["actual_home_margin"].to_numpy()
    )
    mae_high, mae_low = _mae_by_confidence(df)
    return {
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "run_name": run_name,
        "encoding_phase": 1,
        "similarity_method": method,
        "similarity_threshold_or_k": param_label,
        "layers_enabled": "L1+L2+L3",
        "n_games_evaluated": len(df),
        "fallback_rate": round(df["fallback_used"].mean(), 4),
        "mean_confidence": round(df["confidence"].mean(), 4),
        "corr_style_vs_margin": round(corr_a7, 4),
        "mae_high_conf": round(mae_high, 4) if not np.isnan(mae_high) else "",
        "mae_low_conf": round(mae_low, 4) if not np.isnan(mae_low) else "",
        "corr_a2_alone": round(corr_a2, 4),
        "corr_a7_alone": round(corr_a7, 4),
        "corr_a2_plus_a7": round(corr_combined, 4),
        "notes": notes,
    }


def _append_rows(rows: list[dict]) -> None:
    write_header = not RESULTS_CSV.exists()
    RESULTS_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_CSV, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if write_header:
            writer.writeheader()
        for row in rows:
            writer.writerow(row)


def run_phase4_ablation(eval_start_date: str = "2023-10-01") -> list[dict]:
    cfg = load_style_matchup_config()
    h2h = build_a2_h2h_scores()

    rows = []

    l1_df = _naive_diff_score(layer=1, eval_start_date=eval_start_date)
    rows.append(_row_from_naive(
        "layer_ablation_L1_only", 1, l1_df, h2h,
        "Naive equal-weight sum of z-scored home-away diffs, layer=1 (no injury adj), no similarity search.",
    ))

    l2_df = _naive_diff_score(layer=2, eval_start_date=eval_start_date)
    rows.append(_row_from_naive(
        "layer_ablation_L1_L2", 2, l2_df, h2h,
        "Naive equal-weight sum of z-scored home-away diffs, layer=2 (injury-adjusted), no similarity search.",
    ))

    cosine_df = run_similarity_search(
        layer=2, method="cosine", threshold=cfg["similarity_threshold"],
        min_confidence_sample=cfg["min_confidence_sample"],
        full_confidence_sample=cfg["full_confidence_sample"],
        eval_start_date=eval_start_date,
    )
    rows.append(_row_from_similarity(
        "layer_ablation_L1_L2_L3_cosine", "cosine", cfg["similarity_threshold"], cosine_df,
        "Full pipeline: injury-adjusted fingerprints + cosine similarity search, H2H fallback below min_confidence_sample.",
    ))

    knn_df = run_similarity_search(
        layer=2, method="knn", k=cfg["knn_k"],
        min_confidence_sample=cfg["min_confidence_sample"],
        full_confidence_sample=cfg["full_confidence_sample"],
        eval_start_date=eval_start_date,
    )
    rows.append(_row_from_similarity(
        "layer_ablation_L1_L2_L3_knn", "knn", cfg["knn_k"], knn_df,
        "Full pipeline: injury-adjusted fingerprints + KNN similarity search, H2H fallback below min_confidence_sample.",
    ))

    # Magic-number exploration rows: cosine threshold sweep + knn k sweep (Phase 3 findings,
    # recorded here too since they're ablation configurations in their own right).
    for t in [0.5, 0.6, 0.8, 0.9]:
        df = run_similarity_search(
            layer=2, method="cosine", threshold=t,
            min_confidence_sample=cfg["min_confidence_sample"],
            full_confidence_sample=cfg["full_confidence_sample"],
            eval_start_date=eval_start_date,
        )
        rows.append(_row_from_similarity(
            f"threshold_sweep_cosine_{t}", "cosine", t, df,
            "Magic-number exploration: cosine threshold sweep (see Phase 3 phase log).",
        ))

    for k in [10, 20, 50, 100]:
        df = run_similarity_search(
            layer=2, method="knn", k=k,
            min_confidence_sample=cfg["min_confidence_sample"],
            full_confidence_sample=cfg["full_confidence_sample"],
            eval_start_date=eval_start_date,
        )
        rows.append(_row_from_similarity(
            f"k_sweep_knn_{k}", "knn", k, df,
            "Magic-number exploration: KNN k sweep (see Phase 3 phase log).",
        ))

    _append_rows(rows)
    logger.info(f"Wrote {len(rows)} rows to {RESULTS_CSV}")
    return rows


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    result_rows = run_phase4_ablation()
    for r in result_rows:
        print(r["run_name"], "| corr_a7=", r["corr_a7_alone"], "| corr_a2=", r["corr_a2_alone"],
              "| corr_combined=", r["corr_a2_plus_a7"], "| fallback=", r["fallback_rate"])
