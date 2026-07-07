"""
Item #2 of the walk-forward-CV run: the KNN-with-similarity-floor hybrid method.

Motivation (reviewer's concern, restated): plain KNN as implemented in similarity.py/
tuning.py always forces exactly `k` neighbors once enough history exists, regardless
of how similar those neighbors actually are -- there is no mechanism for a thin or
stylistically unusual period to return fewer, more-relevant neighbors instead of being
padded with dissimilar ones. The hybrid method (method="knn_floor" in
tuning.run_search_inmemory) fixes this: take up to `k` nearest neighbors by cosine
similarity, but only those that ALSO clear a minimum similarity `floor`. If fewer than
`k` games clear the floor, fewer are used -- n_similar shrinks naturally, which flows
straight into the existing confidence/fallback mechanism unchanged (lower n_similar ->
lower confidence -> H2H fallback below min_confidence_sample). No new fallback logic
was needed; this is exactly how the pipeline already behaves for cosine, just fed a
different n_similar computation.

Search protocol: this module performs a real 2D grid search over (k, floor_threshold),
NOT a hand-picked pair, using the SAME guardrail train/validation split as item #1's
Optuna search (src/matchups/split.py) -- selected on TRAIN, reported on validation.
Grid (not Optuna) was chosen for this 2D search because (a) it's only 2 knobs, so a
modest grid (6 x 6 = 36 combos) covers the space about as well as an equivalent Optuna
budget would, or a bit better since it's non-adaptive, and (b) it makes the full
response surface directly inspectable (is the optimum a ridge, a corner, flat?) without
having to separately plot Optuna's trial history.

Fingerprint config used for this search: window=37, halflife=13.2 (the wider-exploration
run's winning fingerprint config, not the window=20/halflife=5 hand-picked default).
Rationale: the hybrid method is proposed specifically as a fix to that run's WINNING
KNN configuration (knn_k=81, layer=2, on that same window/halflife) -- the most useful
question is "does adding a floor improve on the already-best KNN setup," not "does it
improve on an untuned baseline nobody would deploy with KNN anyway." The resulting best
(k, floor) is then carried into walkforward.py as the fixed "hybrid" reference config
for the fold evaluation (item #1), using this SAME window=37/halflife=13.2 fingerprint.

k range: 10-150, matching the exact range explored for plain KNN in both previous runs
(Phase 3's sweep, item #1's Optuna search) so results are comparable.
floor range: chosen empirically -- a sample of cosine similarities in this 10-dim
z-scored matchup-vector space (window=37/halflife=13.2, layer=2) has median ~0.0,
p90 ~0.47, p99 ~0.76, min/max roughly [-0.95, 0.98] (see phase log for the exact sample
used). Floors of 0.8+ would reject nearly all candidates (degenerating toward the
"insufficient data -> H2H fallback" regime the design doc already warns about for an
overly strict cosine threshold), so the grid stays within the range that keeps a
non-degenerate fraction of history eligible: [-1.0 (no floor, i.e. reduces to plain
KNN -- included as an internal correctness anchor), 0.0, 0.2, 0.4, 0.5, 0.6, 0.7].
"""

import logging

import numpy as np
import pandas as pd

from src.matchups.split import get_split_dates
from src.matchups.tuning import build_idx_for_config, load_constants, run_search_inmemory

logger = logging.getLogger(__name__)

# Fixed fingerprint config for this search (see module docstring for rationale).
HYBRID_WINDOW = 37
HYBRID_HALFLIFE = 13.199390932957819

K_GRID = [10, 30, 50, 81, 100, 150]
FLOOR_GRID = [-1.0, 0.0, 0.2, 0.4, 0.5, 0.6, 0.7]

# min/full confidence sample also carried over from the wider-exploration winner,
# since they were tuned jointly with knn_k=81 as part of that same config and are not
# being re-searched here (this run's item #2 is specifically about k/floor).
MIN_CONFIDENCE_SAMPLE = 21
FULL_CONFIDENCE_SAMPLE = 82


def run_grid_search() -> dict:
    consts = load_constants()
    splits = get_split_dates()
    idx = build_idx_for_config(consts, window=HYBRID_WINDOW, halflife=HYBRID_HALFLIFE, layer=2)

    rows = []
    for k in K_GRID:
        for floor in FLOOR_GRID:
            out = run_search_inmemory(
                idx, consts["h2h"], method="knn_floor", threshold=0.7, k=k, floor=floor,
                min_confidence_sample=MIN_CONFIDENCE_SAMPLE, full_confidence_sample=FULL_CONFIDENCE_SAMPLE,
                eval_start=splits["train_start"], eval_end=splits["train_end"],
            )
            corr = float(out["style_score"].corr(out["actual_home_margin"])) if out["style_score"].std() > 0 else 0.0
            mean_n_similar = float(out["n_similar"].mean())
            fallback_rate = float(out["fallback_used"].mean())
            rows.append({
                "k": k, "floor": floor, "train_corr": corr,
                "mean_n_similar": mean_n_similar, "fallback_rate": fallback_rate,
            })
            logger.info(f"k={k} floor={floor}: train_corr={corr:.4f} mean_n_similar={mean_n_similar:.1f} "
                        f"fallback_rate={fallback_rate:.3f}")

    grid_df = pd.DataFrame(rows)
    best = grid_df.loc[grid_df["train_corr"].idxmax()]
    best_k, best_floor = int(best["k"]), float(best["floor"])

    # Evaluate the best (k, floor) on validation for the report.
    val_out = run_search_inmemory(
        idx, consts["h2h"], method="knn_floor", threshold=0.7, k=best_k, floor=best_floor,
        min_confidence_sample=MIN_CONFIDENCE_SAMPLE, full_confidence_sample=FULL_CONFIDENCE_SAMPLE,
        eval_start=splits["validation_start"], eval_end=splits["validation_end"],
    )
    val_corr = float(val_out["style_score"].corr(val_out["actual_home_margin"]))

    # Also report the k=best_k plain-KNN (floor=-1, i.e. no floor) row for a clean
    # apples-to-apples "does the floor help at the SAME k" comparison.
    plain_knn_row = grid_df[(grid_df["k"] == best_k) & (grid_df["floor"] == -1.0)].iloc[0]

    summary = {
        "grid_df": grid_df,
        "best_k": best_k,
        "best_floor": best_floor,
        "best_train_corr": float(best["train_corr"]),
        "best_validation_corr": val_corr,
        "plain_knn_same_k_train_corr": float(plain_knn_row["train_corr"]),
        "window": HYBRID_WINDOW,
        "halflife": HYBRID_HALFLIFE,
        "min_confidence_sample": MIN_CONFIDENCE_SAMPLE,
        "full_confidence_sample": FULL_CONFIDENCE_SAMPLE,
    }
    logger.info(f"Hybrid grid search summary: best_k={best_k} best_floor={best_floor} "
                f"train_corr={best['train_corr']:.4f} validation_corr={val_corr:.4f} "
                f"(plain knn same k train_corr={plain_knn_row['train_corr']:.4f})")
    return summary


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    result = run_grid_search()
    print(result["grid_df"].sort_values("train_corr", ascending=False).head(10).to_string(index=False))
    print("BEST:", {k: v for k, v in result.items() if k != "grid_df"})
