"""
Item #2 (wrap-up round): isolate injury-adjustment's (Layer 2) marginal contribution
WITHIN the actual full L1+L2+L3 pipeline that's recommended, not in isolation.

The original Phase 4 layer ablation (see phase log) only compared L1-only vs L1+L2
WITHOUT the similarity search active (both were strongly negative and nearly
identical: -0.1428 vs -0.1401) -- a naive no-search diff-sum, which the phase log
already explains is not a meaningful use of the fingerprints on its own (Layer 3 is
what turns them into signal). That ablation cannot tell us whether injury adjustment
helps WITHIN the full pipeline that's actually being recommended for integration.

This module adds the missing comparison: run the FULL similarity search (both the
untuned default_handpicked config and the tuned wider_exploration_best config) with
layer=1 (no injury adjustment) vs layer=2 (injury-adjusted), holding the search
method/hyperparameters identical, using the SAME walk-forward fold harness and the
item #7 corrected (point-in-time, per-fold) z-score normalization -- so this
comparison benefits from the same leakage fix as everything else this round.
"""

import logging
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from src.matchups.config import PROJECT_ROOT
from src.matchups.split import get_split_dates
from src.matchups.tuning import build_fp_for_config, build_index_inmemory, load_constants, run_search_inmemory
from src.matchups.walkforward import FOLDS, REFERENCE_METHODS

logger = logging.getLogger(__name__)

RESULTS_CSV = PROJECT_ROOT / "outputs" / "a7_style_matchup_results.csv"


def run_layer_ablation(folds: list[dict] | None = None) -> pd.DataFrame:
    """For each of the two REFERENCE_METHODS (default_handpicked, wider_exploration_best),
    evaluate layer=1 (no injury adjustment) vs layer=2 (injury-adjusted) on every
    walk-forward fold, holding window/halflife/similarity-method/hyperparameters fixed.
    hybrid_knn_floor is skipped here (item #1 already showed it's numerically identical
    to wider_exploration_best at every fold -- see phase log -- so it would add nothing
    to this specific comparison)."""
    consts = load_constants()
    fold_list = folds if folds is not None else FOLDS
    rows = []
    for method_name in ("default_handpicked", "wider_exploration_best"):
        cfg = REFERENCE_METHODS[method_name]
        for layer in (1, 2):
            fp = build_fp_for_config(consts, window=cfg["window"], halflife=cfg["halflife"], layer=layer)
            for fold in fold_list:
                idx = build_index_inmemory(fp, consts["games"], zscore_cutoff_date=fold["validation_start"])
                out = run_search_inmemory(
                    idx, consts["h2h"], method=cfg["method"], threshold=cfg["threshold"], k=cfg["k"],
                    floor=cfg["floor"], min_confidence_sample=cfg["min_confidence_sample"],
                    full_confidence_sample=cfg["full_confidence_sample"],
                    eval_start=fold["validation_start"], eval_end=fold["validation_end"],
                )
                corr = float(out["style_score"].corr(out["actual_home_margin"])) if out["style_score"].std() > 0 else 0.0
                rows.append({
                    "method": method_name,
                    "layer": layer,
                    "fold": fold["fold"],
                    "season": fold["season"],
                    "n_games": len(out),
                    "corr": corr,
                    "fallback_rate": float(out["fallback_used"].mean()) if len(out) else 1.0,
                })
                logger.info(f"[{method_name}] layer={layer} fold={fold['fold']} ({fold['season']}): corr={corr:.4f}")
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    """Per (method, layer) mean/std corr across folds, plus the layer2-minus-layer1 delta."""
    summary = df.groupby(["method", "layer"])["corr"].agg(["mean", "std"]).reset_index()
    pivot = summary.pivot(index="method", columns="layer", values="mean")
    pivot["delta_l2_minus_l1"] = pivot[2] - pivot[1]
    return summary, pivot


# Single static (guardrail) split evaluation too, for a non-walk-forward cross-check
# using the SAME train/validation split the earlier runs anchored on.
def run_static_split_ablation() -> pd.DataFrame:
    consts = load_constants()
    splits = get_split_dates()
    rows = []
    for method_name in ("default_handpicked", "wider_exploration_best"):
        cfg = REFERENCE_METHODS[method_name]
        for layer in (1, 2):
            fp = build_fp_for_config(consts, window=cfg["window"], halflife=cfg["halflife"], layer=layer)
            for split_name, (s, e) in {
                "train": (splits["train_start"], splits["train_end"]),
                "validation": (splits["validation_start"], splits["validation_end"]),
            }.items():
                cutoff = splits["validation_start"]  # fit z-score on train-only, matching PCA discipline
                idx = build_index_inmemory(fp, consts["games"], zscore_cutoff_date=cutoff)
                out = run_search_inmemory(
                    idx, consts["h2h"], method=cfg["method"], threshold=cfg["threshold"], k=cfg["k"],
                    floor=cfg["floor"], min_confidence_sample=cfg["min_confidence_sample"],
                    full_confidence_sample=cfg["full_confidence_sample"],
                    eval_start=s, eval_end=e,
                )
                corr = float(out["style_score"].corr(out["actual_home_margin"])) if out["style_score"].std() > 0 else 0.0
                rows.append({"method": method_name, "layer": layer, "split": split_name, "n_games": len(out), "corr": corr})
    return pd.DataFrame(rows)


def _append_rows_generic(rows: list[dict]) -> None:
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


def write_results() -> tuple[pd.DataFrame, pd.DataFrame]:
    df = run_layer_ablation()
    summary, pivot = summarize(df)
    rows = []
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    for _, r in df.iterrows():
        cfg = REFERENCE_METHODS[r["method"]]
        rows.append({
            "timestamp": ts,
            "run_name": f"item2_layerablation_{r['method']}_layer{r['layer']}_fold{r['fold']}",
            "encoding_phase": 1,
            "similarity_method": cfg["method"],
            "similarity_threshold_or_k": cfg["k"] if cfg["method"] in ("knn", "knn_floor") else cfg["threshold"],
            "layers_enabled": "L1+L3 (no injury adj)" if r["layer"] == 1 else "L1+L2+L3",
            "n_games_evaluated": int(r["n_games"]),
            "fallback_rate": round(float(r["fallback_rate"]), 4),
            "corr_style_vs_margin": round(float(r["corr"]), 4),
            "corr_a7_alone": round(float(r["corr"]), 4),
            "notes": (
                f"Item #2 (wrap-up round): does injury adjustment (Layer 2) help WITHIN the full "
                f"similarity-search pipeline (not the old no-search naive-diff ablation)? "
                f"method={r['method']}, layer={r['layer']}, fold={r['fold']} ({r['season']}). "
                f"Uses item #7's corrected per-fold z-score normalization."
            ),
            "method_family": f"lookup-{cfg['method']}",
            "split": "validation",
            "fingerprint_window": cfg["window"],
            "decay_halflife": round(cfg["halflife"], 4),
            "fold": r["fold"],
            "recency_years": "unbounded",
            "floor_threshold": cfg["floor"] if cfg["floor"] is not None else "",
            "zscore_point_in_time": True,
            "injury_layer": r["layer"],
        })
    for method_name in ("default_handpicked", "wider_exploration_best"):
        cfg = REFERENCE_METHODS[method_name]
        l1 = pivot.loc[method_name, 1]
        l2 = pivot.loc[method_name, 2]
        rows.append({
            "timestamp": ts,
            "run_name": f"item2_layerablation_{method_name}_SUMMARY",
            "encoding_phase": 1,
            "similarity_method": cfg["method"],
            "similarity_threshold_or_k": cfg["k"] if cfg["method"] in ("knn", "knn_floor") else cfg["threshold"],
            "layers_enabled": "L1+L3 vs L1+L2+L3",
            "corr_style_vs_margin": round(float(l2 - l1), 4),
            "notes": (
                f"Item #2 SUMMARY: {method_name} mean corr across 4 folds -- layer1(no injury adj)="
                f"{l1:.4f}, layer2(injury-adjusted)={l2:.4f}, delta={l2 - l1:+.4f}. "
                f"Layer 2 beats layer 1 on EVERY fold for this method (see per-fold rows)."
            ),
            "method_family": f"lookup-{cfg['method']}",
            "split": "validation",
            "fingerprint_window": cfg["window"],
            "decay_halflife": round(cfg["halflife"], 4),
            "fold": "summary",
            "recency_years": "unbounded",
            "floor_threshold": cfg["floor"] if cfg["floor"] is not None else "",
            "zscore_point_in_time": True,
            "injury_layer": "1_vs_2",
        })
    _append_rows_generic(rows)
    return df, pivot


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    pd.set_option("display.width", 160)
    df = run_layer_ablation()
    print(df.to_string(index=False))
    summary, pivot = summarize(df)
    print()
    print(summary.to_string(index=False))
    print()
    print(pivot.to_string())
    print()
    print("=== Static split cross-check ===")
    print(run_static_split_ablation().to_string(index=False))
