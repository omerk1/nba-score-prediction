"""
Item #8 (wrap-up round): write the 5-fold (2021-22 through 2025-26) walk-forward
comparison into outputs/a7_style_matchup_results.csv, extending the fold count from
the previous run's n=4 using the item #7 corrected (point-in-time) z-score fit.
"""

import logging
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from src.matchups.config import PROJECT_ROOT
from src.matchups.walkforward import FOLDS_WITH_FOLD5, REFERENCE_METHODS, run_walkforward

logger = logging.getLogger(__name__)

RESULTS_CSV = PROJECT_ROOT / "outputs" / "a7_style_matchup_results.csv"


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


def build_and_write_rows() -> dict:
    df = run_walkforward(folds=FOLDS_WITH_FOLD5, zscore_point_in_time=True)
    rows = []
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    comparison = {}
    for method_name, cfg in REFERENCE_METHODS.items():
        sub = df[df["method"] == method_name].sort_values("fold")
        fold_corrs = []
        for _, r in sub.iterrows():
            rows.append({
                "timestamp": ts,
                "run_name": f"item8_5fold_{method_name}_fold{r['fold']}",
                "encoding_phase": 1,
                "similarity_method": cfg["method"],
                "similarity_threshold_or_k": cfg["k"] if cfg["method"] in ("knn", "knn_floor") else cfg["threshold"],
                "layers_enabled": "L1+L2+L3",
                "n_games_evaluated": int(r["n_games"]),
                "fallback_rate": round(float(r["fallback_rate"]), 4),
                "mean_confidence": round(float(r["mean_confidence"]), 4),
                "corr_style_vs_margin": round(float(r["corr"]), 4),
                "corr_a7_alone": round(float(r["corr"]), 4),
                "notes": (
                    f"Item #8: 5-fold walk-forward (extends item#1's n=4 with a 5th fold using "
                    f"already-present 2025-26 data), fold {r['fold']} ({r['season']}), "
                    f"method={method_name}. Uses item #7's corrected per-fold z-score fit."
                ),
                "method_family": f"lookup-{cfg['method']}",
                "split": "validation",
                "fingerprint_window": cfg["window"],
                "decay_halflife": round(cfg["halflife"], 4),
                "fold": int(r["fold"]),
                "recency_years": "unbounded",
                "floor_threshold": cfg["floor"] if cfg["floor"] is not None else "",
                "zscore_point_in_time": True,
            })
            fold_corrs.append(float(r["corr"]))
        mean_c, std_c = float(np.mean(fold_corrs)), float(np.std(fold_corrs, ddof=1))
        rows.append({
            "timestamp": ts,
            "run_name": f"item8_5fold_{method_name}_SUMMARY",
            "encoding_phase": 1,
            "similarity_method": cfg["method"],
            "similarity_threshold_or_k": cfg["k"] if cfg["method"] in ("knn", "knn_floor") else cfg["threshold"],
            "layers_enabled": "L1+L2+L3",
            "corr_style_vs_margin": round(mean_c, 4),
            "corr_a7_alone": round(mean_c, 4),
            "notes": (
                f"Item #8 SUMMARY across 5 folds (2021-22..2025-26): method={method_name}, "
                f"mean_corr={mean_c:.4f}, std_corr={std_c:.4f}, per_fold_corrs={[round(c, 4) for c in fold_corrs]}."
            ),
            "method_family": f"lookup-{cfg['method']}",
            "split": "validation",
            "fingerprint_window": cfg["window"],
            "decay_halflife": round(cfg["halflife"], 4),
            "fold": "summary_5fold",
            "recency_years": "unbounded",
            "floor_threshold": cfg["floor"] if cfg["floor"] is not None else "",
            "zscore_point_in_time": True,
        })
        comparison[method_name] = {"fold_corrs": fold_corrs, "mean": mean_c, "std": std_c}
    _append_rows_generic(rows)
    return comparison


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    import json
    print(json.dumps(build_and_write_rows(), indent=2, default=str))
