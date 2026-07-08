"""
Part of the Critique & Bug-Fix Pass: per-fold (point-in-time) z-score normalization
fix.

Writes a side-by-side OLD (leaky global z-score) vs NEW (corrected, point-in-time
per-fold z-score) comparison into outputs/a7_style_matchup_results.csv, extending the
schema with one new column (`zscore_point_in_time`, True/False/blank) rather than
overloading any existing column -- old rows get a blank value for this column via the
generic column-union append logic below (mirrors walkforward_4fold_results.py's
approach).

See matchup_index.py / tuning.py / walkforward.py docstrings for the actual fix; this
module only re-runs the walk-forward harness in both modes and records the comparison.
"""

import logging
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from src.matchups.config import PROJECT_ROOT
from src.matchups.walkforward import FOLDS, REFERENCE_METHODS, run_walkforward, summarize

logger = logging.getLogger(__name__)

RESULTS_CSV = PROJECT_ROOT / "outputs" / "a7_style_matchup_results.csv"


def _append_rows_generic(rows: list[dict]) -> None:
    """Column-union append: any column in `rows` not already in the CSV is added
    (blank-filled for existing rows); any existing column not in `rows` is added to
    the new rows (blank-filled). Extends the schema without disturbing prior runs'
    rows/columns."""
    new_df = pd.DataFrame(rows)
    if RESULTS_CSV.exists():
        existing = pd.read_csv(RESULTS_CSV, dtype=str)
    else:
        existing = pd.DataFrame(columns=list(new_df.columns))
    for col in new_df.columns:
        if col not in existing.columns:
            existing[col] = ""
    for col in existing.columns:
        if col not in new_df.columns:
            new_df[col] = ""
    combined = pd.concat([existing, new_df[existing.columns]], ignore_index=True)
    combined.to_csv(RESULTS_CSV, index=False)
    logger.info(f"Wrote {len(rows)} new rows to {RESULTS_CSV} (total now {len(combined)})")


def _fold_row(method_name: str, cfg: dict, fold: dict, corr: float, fallback_rate: float,
              mean_confidence: float, n_games: int, zscore_pit: bool) -> dict:
    threshold_or_k = cfg["k"] if cfg["method"] in ("knn", "knn_floor") else cfg["threshold"]
    tag = "NEW_point_in_time" if zscore_pit else "OLD_leaky_global"
    return {
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "run_name": f"item7_zscorefix_{tag}_{method_name}_fold{fold['fold']}",
        "encoding_phase": 1,
        "similarity_method": cfg["method"],
        "similarity_threshold_or_k": threshold_or_k,
        "layers_enabled": "L1+L2+L3",
        "n_games_evaluated": n_games,
        "fallback_rate": round(fallback_rate, 4),
        "mean_confidence": round(mean_confidence, 4),
        "corr_style_vs_margin": round(corr, 4),
        "mae_high_conf": "",
        "mae_low_conf": "",
        "corr_a2_alone": "",
        "corr_a7_alone": round(corr, 4),
        "corr_a2_plus_a7": "",
        "notes": (
            f"Item #7 (wrap-up round): fold {fold['fold']} ({fold['season']}), method={method_name}, "
            f"zscore_point_in_time={zscore_pit} ('NEW'=corrected per-fold z-score fit on data strictly "
            f"before this fold's validation_start; 'OLD'=leaky global z-score fit across the full "
            f"fingerprint history, reproducing the previous walk-forward-CV run's numbers exactly)."
        ),
        "method_family": f"lookup-{cfg['method']}",
        "split": "validation",
        "fingerprint_window": cfg["window"],
        "decay_halflife": round(cfg["halflife"], 4),
        "fold": fold["fold"],
        "recency_years": "unbounded",
        "floor_threshold": cfg["floor"] if cfg["floor"] is not None else "",
        "zscore_point_in_time": zscore_pit,
    }


def _summary_row(method_name: str, cfg: dict, fold_corrs: list[float], zscore_pit: bool) -> dict:
    mean_c, std_c = float(np.mean(fold_corrs)), float(np.std(fold_corrs, ddof=1))
    threshold_or_k = cfg["k"] if cfg["method"] in ("knn", "knn_floor") else cfg["threshold"]
    tag = "NEW_point_in_time" if zscore_pit else "OLD_leaky_global"
    return {
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "run_name": f"item7_zscorefix_{tag}_{method_name}_SUMMARY",
        "encoding_phase": 1,
        "similarity_method": cfg["method"],
        "similarity_threshold_or_k": threshold_or_k,
        "layers_enabled": "L1+L2+L3",
        "n_games_evaluated": "",
        "fallback_rate": "",
        "mean_confidence": "",
        "corr_style_vs_margin": round(mean_c, 4),
        "mae_high_conf": "",
        "mae_low_conf": "",
        "corr_a2_alone": "",
        "corr_a7_alone": round(mean_c, 4),
        "corr_a2_plus_a7": "",
        "notes": (
            f"Item #7 SUMMARY across 4 folds: method={method_name}, zscore_point_in_time={zscore_pit}, "
            f"mean_corr={mean_c:.4f}, std_corr={std_c:.4f}, per_fold_corrs={[round(c, 4) for c in fold_corrs]}."
        ),
        "method_family": f"lookup-{cfg['method']}",
        "split": "validation",
        "fingerprint_window": cfg["window"],
        "decay_halflife": round(cfg["halflife"], 4),
        "fold": "summary",
        "recency_years": "unbounded",
        "floor_threshold": cfg["floor"] if cfg["floor"] is not None else "",
        "zscore_point_in_time": zscore_pit,
    }


def build_and_write_rows() -> dict:
    rows = []
    comparison = {}
    for zscore_pit in (False, True):  # OLD first, then NEW, for a readable diff in the CSV
        df = run_walkforward(zscore_point_in_time=zscore_pit)
        for method_name, cfg in REFERENCE_METHODS.items():
            sub = df[df["method"] == method_name].sort_values("fold")
            fold_corrs = []
            for _, r in sub.iterrows():
                fold = next(f for f in FOLDS if f["fold"] == r["fold"])
                rows.append(_fold_row(
                    method_name, cfg, fold, r["corr"], r["fallback_rate"], r["mean_confidence"],
                    r["n_games"], zscore_pit,
                ))
                fold_corrs.append(r["corr"])
            rows.append(_summary_row(method_name, cfg, fold_corrs, zscore_pit))
            comparison.setdefault(method_name, {})["new" if zscore_pit else "old"] = {
                "fold_corrs": fold_corrs, "mean": float(np.mean(fold_corrs)), "std": float(np.std(fold_corrs, ddof=1)),
            }

    _append_rows_generic(rows)
    logger.info(f"Z-score normalization fix comparison: {comparison}")
    return comparison


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    import json
    result = build_and_write_rows()
    print(json.dumps(result, indent=2, default=str))
