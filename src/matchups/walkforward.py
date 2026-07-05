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

One caveat carried over unchanged from build_index_inmemory / build_matchup_index
(NOT something this module changes): the z-score normalization used to build each
config's matchup vectors is fit on the FULL fingerprint history (all rows, not just
each fold's training window). This is the same choice the previous run's tuning.py
and the original matchup_index.py both made ("normalize before concatenating" using
global mean/std) -- it is a mild, pre-existing, project-wide simplification, not a
new leakage introduced here. It is flagged again in the phase log for visibility, but
not "fixed" this run, since the task's specific leakage-discipline requirement is about
the similarity search's date-based exclusion (which is fully fold-respecting: the
search corpus for any evaluated game is still strictly its own prior history), not
about where the z-score statistics come from.
"""

import logging

import numpy as np
import pandas as pd

from src.matchups.hybrid_similarity import HYBRID_HALFLIFE, HYBRID_WINDOW
from src.matchups.split import get_split_dates
from src.matchups.tuning import build_idx_for_config, load_constants, run_search_inmemory

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


def run_walkforward(recency_years: float | None = None) -> pd.DataFrame:
    """Evaluates all three REFERENCE_METHODS on all four FOLDS. Returns a long
    DataFrame: one row per (method, fold) with corr/n_games/fallback_rate/mean_confidence.

    `recency_years` (item #3): if set, bounds every method's similarity-search corpus
    to this many years of prior history instead of the full unbounded history. Passed
    straight through to tuning.run_search_inmemory -- see that function's docstring.
    """
    consts = load_constants()
    rows = []
    for method_name, cfg in REFERENCE_METHODS.items():
        idx = build_idx_for_config(consts, window=cfg["window"], halflife=cfg["halflife"], layer=2)
        for fold in FOLDS:
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
                "n_games": len(out),
                "corr": corr,
                "fallback_rate": float(out["fallback_used"].mean()) if len(out) else 1.0,
                "mean_confidence": float(out["confidence"].mean()) if len(out) else 0.0,
            })
            logger.info(
                f"[{method_name}] fold={fold['fold']} ({fold['season']}) recency={recency_years}: "
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
