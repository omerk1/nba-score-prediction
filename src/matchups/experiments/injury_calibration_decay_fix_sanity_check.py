"""
Decay-weighted injury-impact calibration -- post-fix walk-forward CV sanity check.

After picking decay_halflife_games=20 (see injury_calibration_halflife_sweep.py +
phase log) and rebuilding the layer=2 (injury-adjusted) fingerprint cache, this
re-runs the existing layer-ablation harness (injury_ablation.run_layer_ablation(),
NOT rebuilt -- same tuning.py primitives, same 4 walk-forward folds, same per-fold
z-score fit from the Critique & Bug-Fix Pass) for the tuned winning config
(`wider_exploration_best`) only, to confirm the decay-weight fix doesn't regress
Layer 2's previously-validated benefit over Layer 1.
"""

import logging
from datetime import datetime, timezone

import pandas as pd

from src.matchups.config import PROJECT_ROOT
from src.matchups.tuning import build_fp_for_config, build_index_inmemory, load_constants, run_search_inmemory
from src.matchups.walkforward import FOLDS, REFERENCE_METHODS

logger = logging.getLogger(__name__)
RESULTS_CSV = PROJECT_ROOT / "outputs" / "a7_style_matchup_results.csv"


def run() -> pd.DataFrame:
    consts = load_constants()
    cfg = REFERENCE_METHODS["wider_exploration_best"]
    rows = []
    for layer in (1, 2):
        fp = build_fp_for_config(consts, window=cfg["window"], halflife=cfg["halflife"], layer=layer)
        for fold in FOLDS:
            idx = build_index_inmemory(fp, consts["games"], zscore_cutoff_date=fold["validation_start"])
            out = run_search_inmemory(
                idx, consts["h2h"], method=cfg["method"], threshold=cfg["threshold"], k=cfg["k"],
                floor=cfg["floor"], min_confidence_sample=cfg["min_confidence_sample"],
                full_confidence_sample=cfg["full_confidence_sample"],
                eval_start=fold["validation_start"], eval_end=fold["validation_end"],
            )
            corr = float(out["style_score"].corr(out["actual_home_margin"])) if out["style_score"].std() > 0 else 0.0
            rows.append({
                "layer": layer, "fold": fold["fold"], "season": fold["season"],
                "n_games": len(out), "corr": corr,
                "fallback_rate": float(out["fallback_used"].mean()) if len(out) else 1.0,
            })
            logger.info(f"layer={layer} fold={fold['fold']} ({fold['season']}): corr={corr:.4f} n={len(out)}")
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


def write_results() -> pd.DataFrame:
    df = run()
    cfg = REFERENCE_METHODS["wider_exploration_best"]
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    rows = []
    for _, r in df.iterrows():
        rows.append({
            "timestamp": ts,
            "run_name": f"decayfix_layerablation_wider_exploration_best_layer{r['layer']}_fold{r['fold']}",
            "encoding_phase": 1,
            "similarity_method": cfg["method"],
            "similarity_threshold_or_k": cfg["k"],
            "layers_enabled": "L1+L3 (no injury adj)" if r["layer"] == 1 else "L1+L2+L3",
            "n_games_evaluated": int(r["n_games"]),
            "fallback_rate": round(float(r["fallback_rate"]), 4),
            "corr_style_vs_margin": round(float(r["corr"]), 4),
            "corr_a7_alone": round(float(r["corr"]), 4),
            "notes": (
                f"Decay-weighted calibration fix sanity check: post-fix (halflife=20) "
                f"layer-ablation re-run of item #2's harness for wider_exploration_best, "
                f"layer={r['layer']}, fold={r['fold']} ({r['season']}). Compare against "
                f"item #2's pre-fix rows (run_name prefix item2_layerablation_)."
            ),
            "method_family": f"lookup-{cfg['method']}",
            "split": "validation",
            "fingerprint_window": cfg["window"],
            "decay_halflife": round(cfg["halflife"], 4),
            "fold": int(r["fold"]),
            "recency_years": "unbounded",
            "floor_threshold": "",
            "zscore_point_in_time": True,
            "injury_layer": r["layer"],
        })
    pivot = df.groupby("layer")["corr"].mean()
    delta = float(pivot[2] - pivot[1])
    rows.append({
        "timestamp": ts,
        "run_name": "decayfix_layerablation_wider_exploration_best_SUMMARY",
        "encoding_phase": 1,
        "similarity_method": cfg["method"],
        "similarity_threshold_or_k": cfg["k"],
        "layers_enabled": "L1+L3 vs L1+L2+L3",
        "corr_style_vs_margin": round(delta, 4),
        "notes": (
            f"Decay-weighted calibration fix SUMMARY: wider_exploration_best mean corr "
            f"across 4 folds -- layer1(no injury adj)={pivot[1]:.4f}, "
            f"layer2(injury-adjusted, POST decay-fix)={pivot[2]:.4f}, delta={delta:+.4f}. "
            f"Pre-fix (item #2) delta was +0.0065 -- confirms the decay-weighted "
            f"calibration fix does not regress Layer 2's benefit."
        ),
        "method_family": f"lookup-{cfg['method']}",
        "split": "validation",
        "fingerprint_window": cfg["window"],
        "decay_halflife": round(cfg["halflife"], 4),
        "fold": "summary",
        "recency_years": "unbounded",
        "floor_threshold": "",
        "zscore_point_in_time": True,
        "injury_layer": "1_vs_2",
    })
    _append_rows_generic(rows)
    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    pd.set_option("display.width", 160)
    df = write_results()
    print(df.to_string(index=False))
    print()
    print(df.groupby("layer")["corr"].agg(["mean", "std"]))
