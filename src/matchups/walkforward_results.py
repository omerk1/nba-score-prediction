"""
Writes this run's (walk-forward CV) results into outputs/a7_style_matchup_results.csv,
extending its schema (per instructions: add columns for new information rather than
overload existing ones) with:
    fold             - 1-4 for a per-fold row, "summary" for a mean/std-across-folds
                        row, or "" for a hybrid-grid-search row (not fold-specific)
    recency_years    - the recency-cutoff value used ("unbounded" or a float), item #3
    floor_threshold  - the knn_floor hybrid method's floor (item #2 only, else blank)

Existing rows from both previous runs are preserved as-is; this module only ADDS the
new columns (blank for old rows) and appends new rows.
"""

import logging
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from src.matchups.config import PROJECT_ROOT
from src.matchups.hybrid_similarity import run_grid_search
from src.matchups.recency_sweep import RECENCY_GRID, run_recency_sweep
from src.matchups.tuning import build_idx_for_config, load_constants, run_search_inmemory
from src.matchups.walkforward import FOLDS, REFERENCE_METHODS

logger = logging.getLogger(__name__)

RESULTS_CSV = PROJECT_ROOT / "outputs" / "a7_style_matchup_results.csv"

FIELDNAMES = [
    "timestamp", "run_name", "encoding_phase", "similarity_method",
    "similarity_threshold_or_k", "layers_enabled", "n_games_evaluated",
    "fallback_rate", "mean_confidence", "corr_style_vs_margin",
    "mae_high_conf", "mae_low_conf", "corr_a2_alone", "corr_a7_alone",
    "corr_a2_plus_a7", "notes",
    "method_family", "split", "fingerprint_window", "decay_halflife",
    "fold", "recency_years", "floor_threshold",
]


def _combined_a2_a7_corr(h2h: np.ndarray, style: np.ndarray, margin: np.ndarray) -> float:
    X = np.column_stack([np.ones_like(h2h), h2h, style])
    coefs, *_ = np.linalg.lstsq(X, margin, rcond=None)
    fitted = X @ coefs
    return float(np.corrcoef(fitted, margin)[0, 1])


def _row_from_out_df(
    run_name: str, method_family: str, out: pd.DataFrame,
    encoding_phase, similarity_method: str, threshold_or_k, layers_enabled: str,
    fingerprint_window, decay_halflife, notes: str,
    fold, recency_years, floor_threshold: str = "",
) -> dict:
    corr_a7 = float(out["style_score"].corr(out["actual_home_margin"])) if out["style_score"].std() > 0 else 0.0
    corr_a2 = float(out["h2h_score"].corr(out["actual_home_margin"]))
    corr_combined = _combined_a2_a7_corr(
        out["h2h_score"].to_numpy(), out["style_score"].to_numpy(), out["actual_home_margin"].to_numpy()
    )
    high = out[out["confidence"] >= 0.5]
    low = out[out["confidence"] < 0.5]
    mae_high = round(float((high["style_score"] - high["actual_home_margin"]).abs().mean()), 4) if len(high) else ""
    mae_low = round(float((low["style_score"] - low["actual_home_margin"]).abs().mean()), 4) if len(low) else ""
    return {
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "run_name": run_name,
        "encoding_phase": encoding_phase,
        "similarity_method": similarity_method,
        "similarity_threshold_or_k": threshold_or_k,
        "layers_enabled": layers_enabled,
        "n_games_evaluated": len(out),
        "fallback_rate": round(float(out["fallback_used"].mean()), 4),
        "mean_confidence": round(float(out["confidence"].mean()), 4),
        "corr_style_vs_margin": round(corr_a7, 4),
        "mae_high_conf": mae_high,
        "mae_low_conf": mae_low,
        "corr_a2_alone": round(corr_a2, 4),
        "corr_a7_alone": round(corr_a7, 4),
        "corr_a2_plus_a7": round(corr_combined, 4),
        "notes": notes,
        "method_family": method_family,
        "split": "validation",
        "fingerprint_window": fingerprint_window,
        "decay_halflife": round(decay_halflife, 4) if isinstance(decay_halflife, float) else decay_halflife,
        "fold": fold,
        "recency_years": recency_years,
        "floor_threshold": floor_threshold,
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


def build_item1_rows(consts: dict) -> list[dict]:
    """Per-fold rows (item #1) for all 3 reference methods, plus a mean/std summary
    row per method."""
    rows = []
    for method_name, cfg in REFERENCE_METHODS.items():
        idx = build_idx_for_config(consts, window=cfg["window"], halflife=cfg["halflife"], layer=2)
        fold_corrs = []
        for fold in FOLDS:
            out = run_search_inmemory(
                idx, consts["h2h"], method=cfg["method"], threshold=cfg["threshold"], k=cfg["k"],
                floor=cfg["floor"], min_confidence_sample=cfg["min_confidence_sample"],
                full_confidence_sample=cfg["full_confidence_sample"],
                eval_start=fold["validation_start"], eval_end=fold["validation_end"],
            )
            notes = (
                f"Item #1 walk-forward CV: fold {fold['fold']} ({fold['season']} season), "
                f"train_end={fold['train_end']} (inert for this fixed-hyperparameter lookup "
                f"method -- similarity search corpus is unbounded prior history per game, not "
                f"limited to the fold's train window), layer=2."
            )
            row = _row_from_out_df(
                run_name=f"walkforward_{method_name}_fold{fold['fold']}", method_family=f"lookup-{cfg['method']}",
                out=out, encoding_phase=1, similarity_method=cfg["method"],
                threshold_or_k=cfg["k"] if cfg["method"] in ("knn", "knn_floor") else cfg["threshold"],
                layers_enabled="L1+L2+L3", fingerprint_window=cfg["window"], decay_halflife=cfg["halflife"],
                notes=notes, fold=fold["fold"], recency_years="unbounded",
                floor_threshold=cfg["floor"] if cfg["floor"] is not None else "",
            )
            rows.append(row)
            fold_corrs.append(row["corr_a7_alone"])

        mean_c, std_c = float(np.mean(fold_corrs)), float(np.std(fold_corrs, ddof=1))
        rows.append({
            "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
            "run_name": f"walkforward_{method_name}_SUMMARY",
            "encoding_phase": 1,
            "similarity_method": cfg["method"],
            "similarity_threshold_or_k": cfg["k"] if cfg["method"] in ("knn", "knn_floor") else cfg["threshold"],
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
                f"Item #1 SUMMARY across 4 folds ({', '.join(str(f['fold']) for f in FOLDS)}): "
                f"mean_corr={mean_c:.4f}, std_corr={std_c:.4f}, "
                f"per_fold_corrs={[round(c, 4) for c in fold_corrs]}."
            ),
            "method_family": f"lookup-{cfg['method']}",
            "split": "validation",
            "fingerprint_window": cfg["window"],
            "decay_halflife": round(cfg["halflife"], 4),
            "fold": "summary",
            "recency_years": "unbounded",
            "floor_threshold": cfg["floor"] if cfg["floor"] is not None else "",
        })
    return rows


def build_item2_rows(grid_result: dict) -> list[dict]:
    """Full (k, floor) grid rows (item #2) from hybrid_similarity.run_grid_search's
    TRAIN-split selection grid, plus the best config's validation-split confirmation row."""
    rows = []
    grid_df = grid_result["grid_df"]
    for _, r in grid_df.iterrows():
        rows.append({
            "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
            "run_name": f"hybrid_grid_k{int(r['k'])}_floor{r['floor']}",
            "encoding_phase": 1,
            "similarity_method": "knn_floor",
            "similarity_threshold_or_k": int(r["k"]),
            "layers_enabled": "L1+L2+L3",
            "n_games_evaluated": "",
            "fallback_rate": round(float(r["fallback_rate"]), 4),
            "mean_confidence": "",
            "corr_style_vs_margin": round(float(r["train_corr"]), 4),
            "mae_high_conf": "", "mae_low_conf": "",
            "corr_a2_alone": "", "corr_a7_alone": round(float(r["train_corr"]), 4), "corr_a2_plus_a7": "",
            "notes": (
                f"Item #2 grid search cell (TRAIN split, guardrail #4): k={int(r['k'])}, "
                f"floor={r['floor']}, mean_n_similar={r['mean_n_similar']:.1f}, "
                f"window={grid_result['window']}, halflife={round(grid_result['halflife'], 4)}. "
                f"floor=-1.0 rows are plain-KNN anchors (no floor applied)."
            ),
            "method_family": "lookup-knn_floor",
            "split": "train",
            "fingerprint_window": grid_result["window"],
            "decay_halflife": round(grid_result["halflife"], 4),
            "fold": "",
            "recency_years": "unbounded",
            "floor_threshold": r["floor"],
        })
    rows.append({
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "run_name": "hybrid_grid_BEST_validation",
        "encoding_phase": 1,
        "similarity_method": "knn_floor",
        "similarity_threshold_or_k": grid_result["best_k"],
        "layers_enabled": "L1+L2+L3",
        "n_games_evaluated": "",
        "fallback_rate": "",
        "mean_confidence": "",
        "corr_style_vs_margin": round(grid_result["best_validation_corr"], 4),
        "mae_high_conf": "", "mae_low_conf": "",
        "corr_a2_alone": "", "corr_a7_alone": round(grid_result["best_validation_corr"], 4), "corr_a2_plus_a7": "",
        "notes": (
            f"Item #2: best (k={grid_result['best_k']}, floor={grid_result['best_floor']}) selected on "
            f"TRAIN split (train_corr={grid_result['best_train_corr']:.4f}), re-evaluated on the single "
            f"static VALIDATION split for the guardrail-consistent report. Plain-KNN-same-k train_corr="
            f"{grid_result['plain_knn_same_k_train_corr']:.4f} (i.e. the floor made no difference at the "
            f"winning k on the train split either)."
        ),
        "method_family": "lookup-knn_floor",
        "split": "validation",
        "fingerprint_window": grid_result["window"],
        "decay_halflife": round(grid_result["halflife"], 4),
        "fold": "",
        "recency_years": "unbounded",
        "floor_threshold": grid_result["best_floor"],
    })
    return rows


def build_item3_rows(recency_df: pd.DataFrame) -> list[dict]:
    """Summary (mean/std across folds) rows per (method, recency_years) from the
    recency-cutoff sweep (item #3)."""
    rows = []
    summary = recency_df.groupby(["method", "recency_years"], dropna=False)["corr"].agg(
        ["mean", "std", "min", "max"]
    ).reset_index()
    for _, r in summary.iterrows():
        rec = r["recency_years"]
        cfg = REFERENCE_METHODS[r["method"]]
        rows.append({
            "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
            "run_name": f"recency_sweep_{r['method']}_{rec}",
            "encoding_phase": 1,
            "similarity_method": cfg["method"],
            "similarity_threshold_or_k": cfg["k"] if cfg["method"] in ("knn", "knn_floor") else cfg["threshold"],
            "layers_enabled": "L1+L2+L3",
            "n_games_evaluated": "",
            "fallback_rate": "",
            "mean_confidence": "",
            "corr_style_vs_margin": round(float(r["mean"]), 4),
            "mae_high_conf": "", "mae_low_conf": "",
            "corr_a2_alone": "", "corr_a7_alone": round(float(r["mean"]), 4), "corr_a2_plus_a7": "",
            "notes": (
                f"Item #3 recency-cutoff sweep SUMMARY across 4 folds: method={r['method']}, "
                f"recency_years={rec}, mean_corr={r['mean']:.4f}, std_corr={r['std']:.4f}, "
                f"min={r['min']:.4f}, max={r['max']:.4f}."
            ),
            "method_family": f"lookup-{cfg['method']}",
            "split": "validation",
            "fingerprint_window": cfg["window"],
            "decay_halflife": round(cfg["halflife"], 4),
            "fold": "summary",
            "recency_years": rec,
            "floor_threshold": cfg["floor"] if cfg["floor"] is not None else "",
        })
    return rows


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    consts = load_constants()

    logger.info("Building item #1 rows (walk-forward folds, 3 reference methods)...")
    item1_rows = build_item1_rows(consts)

    logger.info("Building item #2 rows (hybrid grid search)...")
    grid_result = run_grid_search()
    item2_rows = build_item2_rows(grid_result)

    logger.info("Building item #3 rows (recency-cutoff sweep)...")
    recency_df = run_recency_sweep()
    item3_rows = build_item3_rows(recency_df)

    all_rows = item1_rows + item2_rows + item3_rows
    _append_rows(all_rows)
    logger.info(f"Done. Wrote {len(item1_rows)} item#1 + {len(item2_rows)} item#2 + {len(item3_rows)} item#3 rows.")
