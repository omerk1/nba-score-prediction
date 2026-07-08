"""
Hyperparameter search recheck (post z-score-fix confirmation).

Context: the Hyperparameter Search & Alternative Methods stage's
`tuning.run_optuna_search()` (40 Optuna trials, TPE, seed=42) found
`fingerprint_window=37, decay_halflife=13.2, similarity_method=knn, knn_k=81,
min_confidence_sample=21, full_confidence_sample=82` -- but that search ran BEFORE
the Critique & Bug-Fix Pass's z-score fix (the z-score mean/std used to build
matchup vectors were computed globally across the full fingerprint history, a
genuine look-ahead leak). That fix was applied to the walk-forward harness (fitting
each fold's z-score stats on data strictly before that fold's validation_start) and
confirmed the chosen config still wins fold-by-fold -- but explicitly flagged, as a
still-open item, that the SEARCH itself (the single static train/validation split
used by `run_optuna_search`) was never re-run under the corrected normalization: "If
the hyperparameter search were re-run with corrected normalization, the specific
best (window, halflife, k, ...) values might shift slightly." This module closes
that gap.

What this does: re-runs `tuning.run_optuna_search` with the IDENTICAL search space (40
trials, TPE, seed=42, same fingerprint_window/decay_halflife/similarity_method/
similarity_threshold/knn_k/min_confidence_sample/full_confidence_sample bounds) but
with `zscore_point_in_time=True` -- z-score stats fit only on fingerprint rows strictly
before `train_end_date` (2024-04-14, the static split's own train/validation boundary),
applied consistently to both the train (selection) and validation (report) evaluation
windows. Same mechanism as the Critique & Bug-Fix Pass's `zscore_cutoff_date` /
walkforward.py's `zscore_point_in_time`, just applied to this one static split
instead of per-fold (see tuning.py's `run_optuna_search` docstring for the wiring).

Same seed (42) as the original search is deliberately reused (not varied) -- the goal
here is a like-for-like "same search, corrected normalization" comparison, not a wider
seed/trial-budget sweep (out of this run's narrow scope).

Also evaluates, on the identical corrected-normalization splits:
  - the OLD wider-exploration winning config (window=37/halflife=13.2/knn k=81/
    min_confidence=21/full_confidence=82) -- the fair "old config, corrected
    normalization" comparison point (its previously-reported 0.218/0.323 numbers used
    the OLD leaky global z-score fit).
  - the untuned hand-picked default (window=20/halflife=5/cosine@0.70), for the same
    reason the original search reported it.

Writes rows into the existing outputs/a7_style_matchup_results.csv (same schema
wider_exploration_results.py already established -- method_family/split/
fingerprint_window/decay_halflife columns already exist, so no new columns are
introduced here).
"""

import logging

import pandas as pd

from src.matchups.config import PROJECT_ROOT
from src.matchups.split import get_split_dates
from src.matchups.tuning import evaluate_config, load_constants, run_optuna_search
from src.matchups.experiments.wider_exploration_results import DEFAULT_HP_CONFIG, _full_row

logger = logging.getLogger(__name__)

RESULTS_CSV = PROJECT_ROOT / "outputs" / "a7_style_matchup_results.csv"


def _append_rows_generic(rows: list[dict]) -> None:
    """Column-union append (same pattern as zscore_normalization_fix_results.py /
    walkforward_5fold_results.py): any column in `rows` not already in the CSV is
    added (blank-filled for existing rows); any existing column not in `rows` is
    added to the new rows (blank-filled). Deliberately NOT
    wider_exploration_results.py's `_append_rows`, which restricts to a fixed,
    now-stale FIELDNAMES list that predates columns added by later runs (`fold`,
    `recency_years`, `floor_threshold`, `zscore_point_in_time`, `injury_layer`) --
    using it would silently drop those columns from every pre-existing row."""
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

# The Hyperparameter Search & Alternative Methods stage's winning config
# (docs/a7_phase_log.md) -- re-evaluated here under CORRECTED (point-in-time)
# z-score normalization, which it was never evaluated under before (the Critique &
# Bug-Fix Pass only corrected the walk-forward harness, not this static split).
OLD_BEST_HP_CONFIG = {
    "fingerprint_window": 37,
    "decay_halflife": 13.199390932957819,
    "similarity_method": "knn",
    "knn_k": 81,
    "min_confidence_sample": 21,
    "full_confidence_sample": 82,
    "similarity_threshold": 0.7,  # unused (method=knn), kept for evaluate_config's signature
}


def run_recheck(n_trials: int = 40, seed: int = 42) -> dict:
    search_summary = run_optuna_search(n_trials=n_trials, layer=2, seed=seed, zscore_point_in_time=True)
    bp = search_summary["best_params"]
    new_best_cfg = {
        "fingerprint_window": bp["fingerprint_window"],
        "decay_halflife": bp["decay_halflife"],
        "similarity_method": bp["similarity_method"],
        "knn_k": bp["knn_k"],
        "min_confidence_sample": bp["min_confidence_sample"],
        "full_confidence_sample": bp["full_confidence_sample"],
        "similarity_threshold": bp["similarity_threshold"],
    }

    consts = load_constants()
    splits = get_split_dates()
    zscore_cutoff = splits["train_end"]
    split_ranges = {
        "train": (splits["train_start"], splits["train_end"]),
        "validation": (splits["validation_start"], splits["validation_end"]),
    }

    rows = []
    for label, cfg in [
        ("hpsearch_recheck_new_best", new_best_cfg),
        ("hpsearch_recheck_old_best_corrected_norm", OLD_BEST_HP_CONFIG),
        ("hpsearch_recheck_default_corrected_norm", DEFAULT_HP_CONFIG),
    ]:
        for split_name, (s, e) in split_ranges.items():
            result = evaluate_config(
                consts, window=cfg["fingerprint_window"], halflife=cfg["decay_halflife"],
                method=cfg["similarity_method"], threshold=cfg["similarity_threshold"], k=cfg["knn_k"],
                min_confidence_sample=cfg["min_confidence_sample"],
                full_confidence_sample=cfg["full_confidence_sample"],
                eval_start=s, eval_end=e, layer=2, zscore_cutoff_date=zscore_cutoff,
            )
            notes = (
                f"Hyperparameter search recheck (post z-score-fix confirmation): {label}. "
                f"z-score fit point-in-time, cutoff={zscore_cutoff} (train_end_date), applied to "
                f"both train and validation windows -- same static guardrail split the original "
                f"wider-exploration search used, but now with item #7's zscore_cutoff_date wired "
                f"in (previously only the walk-forward harness used it)."
            )
            row = _full_row(
                f"{label}_{split_name}", f"lookup-{cfg['similarity_method']}", split_name, result["df"],
                consts["h2h"], encoding_phase=1, similarity_method=cfg["similarity_method"],
                threshold_or_k=cfg["knn_k"] if cfg["similarity_method"] == "knn" else cfg["similarity_threshold"],
                layers_enabled="L1+L2+L3", fingerprint_window=cfg["fingerprint_window"],
                decay_halflife=round(cfg["decay_halflife"], 4), notes=notes,
            )
            row["zscore_point_in_time"] = True
            rows.append(row)

    _append_rows_generic(rows)

    summary = {
        "search_summary": search_summary,
        "new_best_cfg": new_best_cfg,
        "old_best_train_corr_corrected_norm": search_summary["old_best_train_corr"],
        "old_best_validation_corr_corrected_norm": search_summary["old_best_validation_corr"],
    }
    logger.info(f"Hyperparameter search recheck summary: {summary}")
    return summary


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    import json
    print(json.dumps(run_recheck(), indent=2, default=str))
