"""
Item #3: recency cutoff as an EXPLORED axis, not a hardcoded decision.

A human reviewer asked whether the similarity search should be bounded to a rolling
recency window (e.g. "only search the last N years") to avoid drawing neighbors from
stylistically stale eras. This module treats "recency cutoff in years" (including
"no cutoff" = current unbounded behavior) as another swept parameter, reusing
walkforward.py's fold harness so the recency question is answered with the SAME
robustness lens as item #1 (per-fold, not a single split).

Swept for BOTH `default_handpicked` and `wider_exploration_best` (the two methods
with materially different fingerprint smoothness -- window=20/halflife=5 vs
window=37/halflife=13.2 -- so the recency finding isn't specific to one fingerprint
config). `hybrid_knn_floor` is not separately swept here: item #1's fold results
showed it produces IDENTICAL output to `wider_exploration_best` at every fold (the
floor never binds at k=81 on this fingerprint), so a recency sweep on it would be
redundant with wider_exploration_best's sweep below.

recency_years grid: [1, 2, 3, 5, unbounded]. Chosen to bracket the reviewer's
example ("e.g. only the last N years") on both sides -- 1-2 years is aggressive
(would likely start starving folds of candidates, especially fold 1 which already
has the least warm-start history), 5 years is a mild bound that should barely
differ from unbounded for the later folds (which have 8+ years of history to spare)
but might still matter for fold 1. unbounded is the current/previous-runs' behavior
(no cutoff), included as the reference point every other row is compared against.
"""

import logging

import pandas as pd

from src.matchups.walkforward import run_walkforward

logger = logging.getLogger(__name__)

RECENCY_GRID = [1.0, 2.0, 3.0, 5.0, None]  # None = unbounded


def run_recency_sweep() -> pd.DataFrame:
    all_rows = []
    for recency_years in RECENCY_GRID:
        df = run_walkforward(recency_years=recency_years)
        all_rows.append(df)
    combined = pd.concat(all_rows, ignore_index=True)
    return combined


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    """Mean/std corr across folds, per (method, recency_years) -- the key table."""
    return df.groupby(["method", "recency_years"], dropna=False)["corr"].agg(
        ["mean", "std", "min", "max"]
    ).reset_index()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    result = run_recency_sweep()
    summary = summarize(result)
    print(summary.sort_values(["method", "recency_years"], na_position="last").to_string(index=False))
