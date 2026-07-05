"""
Item #1 of this run: walk-forward (expanding-window) cross-validation harness.

Why this exists: the wider-exploration run's headline result (a hyperparameter-
searched KNN config improving validation corr from 0.285 -> 0.323) was measured on a
SINGLE static train (2018-10-16 to 2024-04-14) / validation (2024-10-22 to 2025-04-13)
split. One split cannot distinguish "this config is robustly better" from "this config
fits the specific calendar quirks of one validation window." This module builds
multiple chronological folds instead and evaluates the SAME three reference methods on
every fold: (a) the untuned hand-picked default, (b) the wider-exploration run's
winning config, (c) the new KNN-with-similarity-floor hybrid (see hybrid_similarity.py
for how its (k, floor) was itself selected).

Fold scheme: validate on each NBA REGULAR SEASON from 2021-22 through 2024-25 (four
folds), training on everything before that season's start -- ending with the existing
`validation_end_date` (2025-04-13) so fold 4 is identical to the guardrail split
already used elsewhere in this project. Season start/regular-season-end dates were
derived directly from the actual game-date data (not guessed): for each season, the
first game date in the Aug-Jul window is the season start; the regular-season end is
detected as the date immediately before the largest date gap in March-May (the
few-day gap between the regular season finale / play-in and the playoffs). This
detection method was validated against the two seasons already pinned in
configs/config.yaml (2023-24 regular season end = 2024-04-14, 2024-25 = 2025-04-13)
and reproduced BOTH exactly, so it is trusted for the two earlier seasons that aren't
already in config.yaml.

Per-fold "training window" note: none of the three reference methods here FIT anything
on the training window (no PCA, no cluster centroids, no supervised model) -- they are
lookup-and-average methods with FIXED hyperparameters (chosen once, either as the
design-doc default or by an earlier search on the single static split). Per the task
instructions, the similarity search's OWN candidate pool is explicitly NOT bounded by
the fold's training window -- it remains the full prior history up to each evaluated
game's date, exactly as in every other module in this project. So the "train window"
for these three methods is inert for anything except documentation/consistency with
the fold-scheme description; it is still recorded per fold below for completeness and
in case a future run adds a method that DOES need to fit something per fold (PCA/
clustering/supervised walk-forward would need it).

**Item #7 fix (wrap-up round):** the paragraph above described a real leakage bug, not
just a "mild simplification" -- confirmed as a genuine bug in discussion with the human
coordinator. The z-score normalization used to build each config's matchup vectors was
fit on the FULL fingerprint history (including rows dated AFTER a given fold's
validation window), so a game evaluated in, say, fold 1 (2021-22) was normalized using
mean/std statistics that include 2024-2026 data that did not exist yet at prediction
time. This module now fits z-score stats PER FOLD, using only fingerprint rows strictly
before that fold's `validation_start` (the fold's own expanding training window -- the
natural, already-existing boundary from the fold scheme above, not an invented one) --
mirroring exactly how `encoding_pca.py` already correctly fits its StandardScaler/PCA on
TRAIN-split-only data. `run_walkforward(..., zscore_point_in_time=True)` (the new
default) uses this corrected per-fold fit; `zscore_point_in_time=False` reproduces the
OLD (leaky) global-stats behavior so the two can be compared side by side -- see the
phase log for the actual corrected-vs-leaky numbers. The similarity search's OWN
date-based candidate-pool exclusion was already fully fold-respecting before this fix
and is unchanged.
"""

import logging

import numpy as np
import pandas as pd

from src.matchups.hybrid_similarity import HYBRID_HALFLIFE, HYBRID_WINDOW
from src.matchups.split import get_split_dates
from src.matchups.tuning import build_fp_for_config, build_index_inmemory, load_constants, run_search_inmemory

logger = logging.getLogger(__name__)

# Season boundaries derived from the actual game-date data (see module docstring for
# the detection method + validation against config.yaml's two known dates).
FOLDS = [
    {"fold": 1, "season": "2021-22", "train_end": "2021-10-01",
     "validation_start": "2021-10-19", "validation_end": "2022-04-10"},
    {"fold": 2, "season": "2022-23", "train_end": "2022-10-01",
     "validation_start": "2022-10-18", "validation_end": "2023-04-09"},
    {"fold": 3, "season": "2023-24", "train_end": "2023-10-01",
     "validation_start": "2023-10-24", "validation_end": "2024-04-14"},
    {"fold": 4, "season": "2024-25", "train_end": "2024-10-01",
     "validation_start": "2024-10-22", "validation_end": "2025-04-13"},
]

# Item #8 (wrap-up round): a 5th fold using the 2025-26 season, already present in
# nba_api.sqlite through 2026-05-24 (confirmed: 1225 regular-season games, 0 with a
# null final score) but beyond the existing validation_end_date (2025-04-13) guardrail
# boundary. Season start/regular-season-end derived with the SAME method as the four
# FOLDS above (first game date in the Aug-Jul window = season start; day before the
# largest March-May date gap = regular season end) -- start=2025-10-21 confirmed
# directly from the game table; end=2026-04-12 confirmed as the date immediately before
# the single largest gap in the March-May 2026 window (a 6-day gap before 2026-04-18,
# where per-day game counts drop from 15/day to 2-4/day -- the regular-season-finale to
# playoffs transition, same signature as every other season). Kept SEPARATE from FOLDS
# (not appended in place) so existing code that specifically wants the original
# 4-fold walk-forward-CV-run scheme is unaffected; FOLDS_WITH_FOLD5 is what item #8's
# analysis passes to run_walkforward(folds=...).
FOLD_5 = {"fold": 5, "season": "2025-26", "train_end": "2025-10-01",
          "validation_start": "2025-10-21", "validation_end": "2026-04-12"}
FOLDS_WITH_FOLD5 = FOLDS + [FOLD_5]

# Fold 4's validation range must equal the existing guardrail split exactly (sanity
# anchor -- checked at import time so a future edit to config.yaml's dates doesn't
# silently desync this module from the rest of the project).
_splits = get_split_dates()
assert FOLDS[-1]["validation_start"] == _splits["validation_start"], (
    f"Fold 4 validation_start {FOLDS[-1]['validation_start']} != config.yaml "
    f"validation_start_date {_splits['validation_start']}"
)
assert FOLDS[-1]["validation_end"] == _splits["validation_end"], (
    f"Fold 4 validation_end {FOLDS[-1]['validation_end']} != config.yaml "
    f"validation_end_date {_splits['validation_end']}"
)

# --- The three reference methods (item #1) ---------------------------------------
REFERENCE_METHODS = {
    "default_handpicked": {
        "window": 20, "halflife": 5.0, "method": "cosine", "threshold": 0.70, "k": 30,
        "floor": None, "min_confidence_sample": 10, "full_confidence_sample": 50,
    },
    "wider_exploration_best": {
        "window": 37, "halflife": 13.199390932957819, "method": "knn", "threshold": 0.7, "k": 81,
        "floor": None, "min_confidence_sample": 21, "full_confidence_sample": 82,
    },
    "hybrid_knn_floor": {
        # (k=81, floor=0.4) chosen by hybrid_similarity.py's grid search -- see that
        # module's docstring/phase-log entry for the full grid and the finding that
        # floor <=0.4 is a near-exact tie with plain KNN at k=81 on this fingerprint
        # config (floor doesn't meaningfully bind until ~0.6-0.7, where it starts
        # hurting). floor=0.4 was picked over the technically-tied floor=0.0/0.2 so
        # the "hybrid" reference config actually exercises a non-degenerate floor.
        "window": HYBRID_WINDOW, "halflife": HYBRID_HALFLIFE, "method": "knn_floor", "threshold": 0.7,
        "k": 81, "floor": 0.4, "min_confidence_sample": 21, "full_confidence_sample": 82,
    },
}


def run_walkforward(
    recency_years: float | None = None, zscore_point_in_time: bool = True,
    folds: list[dict] | None = None,
) -> pd.DataFrame:
    """Evaluates all three REFERENCE_METHODS on all four FOLDS. Returns a long
    DataFrame: one row per (method, fold) with corr/n_games/fallback_rate/mean_confidence.

    `recency_years` (item #3): if set, bounds every method's similarity-search corpus
    to this many years of prior history instead of the full unbounded history. Passed
    straight through to tuning.run_search_inmemory -- see that function's docstring.

    `zscore_point_in_time` (item #7, wrap-up round): if True (the corrected default),
    each fold's matchup-vector z-score stats are fit ONLY on fingerprint rows strictly
    before that fold's validation_start (no look-ahead). If False, reproduces the OLD
    (leaky) behavior -- one global fit across the full fingerprint history, shared by
    every fold -- purely so the two can be compared side by side (see phase log).

    `folds` (item #8): override the module-level FOLDS list, e.g. to add additional
    folds built from newly-available 2025-26 data. Defaults to the original 4 FOLDS.
    """
    consts = load_constants()
    fold_list = folds if folds is not None else FOLDS
    rows = []
    for method_name, cfg in REFERENCE_METHODS.items():
        fp = build_fp_for_config(consts, window=cfg["window"], halflife=cfg["halflife"], layer=2)
        # Global (leaky) index built once, reused for every fold when zscore_point_in_time=False.
        idx_global = None if zscore_point_in_time else build_index_inmemory(fp, consts["games"])
        for fold in fold_list:
            if zscore_point_in_time:
                idx = build_index_inmemory(fp, consts["games"], zscore_cutoff_date=fold["validation_start"])
            else:
                idx = idx_global
            out = run_search_inmemory(
                idx, consts["h2h"], method=cfg["method"], threshold=cfg["threshold"], k=cfg["k"],
                floor=cfg["floor"], min_confidence_sample=cfg["min_confidence_sample"],
                full_confidence_sample=cfg["full_confidence_sample"],
                eval_start=fold["validation_start"], eval_end=fold["validation_end"],
                recency_years=recency_years,
            )
            corr = float(out["style_score"].corr(out["actual_home_margin"])) if out["style_score"].std() > 0 else 0.0
            rows.append({
                "method": method_name,
                "fold": fold["fold"],
                "season": fold["season"],
                "validation_start": fold["validation_start"],
                "validation_end": fold["validation_end"],
                "recency_years": recency_years if recency_years is not None else "unbounded",
                "zscore_point_in_time": zscore_point_in_time,
                "n_games": len(out),
                "corr": corr,
                "fallback_rate": float(out["fallback_used"].mean()) if len(out) else 1.0,
                "mean_confidence": float(out["confidence"].mean()) if len(out) else 0.0,
            })
            logger.info(
                f"[{method_name}] fold={fold['fold']} ({fold['season']}) recency={recency_years} "
                f"zscore_pit={zscore_point_in_time}: "
                f"corr={corr:.4f} n={len(out)} fallback_rate={rows[-1]['fallback_rate']:.3f}"
            )
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    """Per-method mean/std corr across folds (the key robustness table)."""
    return df.groupby("method")["corr"].agg(["mean", "std", "min", "max"]).reset_index()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    result = run_walkforward()
    print(result.to_string(index=False))
    print()
    print(summarize(result).to_string(index=False))
