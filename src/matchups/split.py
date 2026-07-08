"""
Guardrail added by the Hyperparameter Search & Alternative Methods stage: a real
chronological train/validation split for A7 hyperparameter search and method
comparison.

Reuses configs/config.yaml's EXISTING datasets_loading date ranges
(train_start_date/train_end_date/validation_start_date/validation_end_date) —
the same split the production model (train_model.py / tune_model.py) already
uses — rather than inventing a new A7-specific date scheme. This keeps the A7
exploration consistent with how the rest of the project defines "train" vs
"validation," and means the numbers reported here are comparable to how the
production pipeline would eventually see this data if A7 were integrated.

Note: the previous A7 run's Phase 3/4 evaluation window (2023-10-01 onward) is
NOT reused for the split-based comparisons in this run, since it straddles both
the train (through 2024-04-14) and validation (2024-10-22+) ranges below and
was never intended as a train/test boundary — it was a "recent enough to be
representative, cheap enough to compute" window, not a leakage-safe split. It
remains valid for what it was used for (the original ablation), but hyperparameter
search in this run must not select against it.

The `test_start_date`/`test_end_date` range in config.yaml is deliberately NOT
used here (instructions only call for train/validation) and is left untouched,
consistent with the project's held-out test set discipline.
"""

from src.utils.config_loader import load_config


def get_split_dates() -> dict:
    """Returns train_start/train_end/validation_start/validation_end as strings
    (YYYY-MM-DD), read directly from configs/config.yaml's datasets_loading block."""
    cfg = load_config()
    dl = cfg.datasets_loading
    return {
        "train_start": dl.train_start_date,
        "train_end": dl.train_end_date,
        "validation_start": dl.validation_start_date,
        "validation_end": dl.validation_end_date,
    }


if __name__ == "__main__":
    print(get_split_dates())
