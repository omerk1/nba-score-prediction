"""E4: Injury missing-data handling (session rs_20260808_1).

Compares injury_features.missing_value_strategy: zero_fill (status quo) vs.
native_nan, full 5-fold each (fold1-relevant by construction, per D1's
finding that fold1's train+val AND fold2's train window have zero injury
coverage -- not a cheap-3-fold screen).

Reframed from EXPERIMENTS.md's original 3-variant description after finding
(during D2's related work) that "variant 1: native NaN, status quo" wasn't
accurate -- the code already zero-fills, and "variant 2: availability
indicator" (has_injury_data) already ships. The real open question is
zero_fill vs. native_nan; the third "folds-start-at-2021" variant doesn't need
a separate training run -- it's a restricted (folds 3-5 only, the folds where
D1 confirmed the *training* window has real coverage) slice of the SAME two
full-5-fold runs below, reported alongside the full-5-fold means rather than
as a third run.

Usage: venv/bin/python3 scripts/run_injury_missingdata_comparison.py --session-id <id>

Temporarily toggles configs/config.yaml's injury_features.missing_value_strategy,
restores the committed default (zero_fill) afterward regardless of outcome.
SIGTERM/SIGINT-safe restore, same pattern as the other orchestration scripts.
"""

import argparse
import csv
import os
import re
import signal
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
CONFIG_PATH = REPO / "configs" / "config.yaml"
sys.path.append(str(REPO))
os.chdir(REPO)


def _handle_term(signum, frame):
    raise KeyboardInterrupt(f"received signal {signum}")


signal.signal(signal.SIGTERM, _handle_term)
signal.signal(signal.SIGINT, _handle_term)

from src.evaluation.cv_harness import run_split, validate_fold_definitions  # noqa: E402
from src.utils.config_loader import load_config  # noqa: E402

COMMITTED_DEFAULT = "zero_fill"
COMPARABLE_FOLD_NAMES = {"fold3", "fold4", "fold5"}  # D1: fold1 train+val and fold2 train have zero coverage


def set_missing_value_strategy(value: str):
    text = CONFIG_PATH.read_text()
    new_text = re.sub(
        r"(missing_value_strategy: )(zero_fill|native_nan)",
        lambda m: m.group(1) + value,
        text,
        count=1,
    )
    assert new_text != text or f"missing_value_strategy: {value}" in text
    CONFIG_PATH.write_text(new_text)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session-id", required=True)
    args = parser.parse_args()
    session_csv = REPO / "results" / "sessions" / f"{args.session_id}.csv"

    if load_config().injury_features.missing_value_strategy.value != COMMITTED_DEFAULT:
        print("WARNING: missing_value_strategy not at committed default at startup -- resetting.", flush=True)
        set_missing_value_strategy(COMMITTED_DEFAULT)

    try:
        cfg = load_config()
        folds = cfg.cv.folds
        validate_fold_definitions(folds)

        results = {}  # (label, fold_name) -> SplitResult
        for label, value in [("zero_fill", "zero_fill"), ("native_nan", "native_nan")]:
            set_missing_value_strategy(value)
            cfg = load_config()
            assert cfg.injury_features.missing_value_strategy.value == value
            print(f"\n### missing_value_strategy={value} ###", flush=True)
            for f in folds:
                result = run_split(
                    cfg,
                    f.train_end_date,
                    f.validation_start_date,
                    f.validation_end_date,
                    f.test_start_date,
                    f.test_end_date,
                )
                print(
                    f"  {f.name}: val_score={result.val_score:.4f} "
                    f"(diff_mae={result.val_metrics['diff_mae']:.2f}, total_mae={result.val_metrics['total_mae']:.2f}) | "
                    f"test_score={result.test_score:.4f}",
                    flush=True,
                )
                results[(label, f.name)] = result

        print("\n=== zero_fill vs native_nan, per fold ===")
        for f in folds:
            zf, nn = results[("zero_fill", f.name)], results[("native_nan", f.name)]
            comparable = (
                " (comparable)"
                if f.name in COMPARABLE_FOLD_NAMES
                else " (fold1/fold2: zero train-time injury coverage per D1)"
            )
            print(
                f"  {f.name}{comparable}: val_score zero_fill={zf.val_score:.4f} native_nan={nn.val_score:.4f} "
                f"delta(nan-zf)={nn.val_score - zf.val_score:+.4f}"
            )

        for label in ("zero_fill", "native_nan"):
            all_vs = [results[(label, f.name)].val_score for f in folds]
            all_ts = [results[(label, f.name)].test_score for f in folds]
            comp_vs = [results[(label, f.name)].val_score for f in folds if f.name in COMPARABLE_FOLD_NAMES]
            comp_ts = [results[(label, f.name)].test_score for f in folds if f.name in COMPARABLE_FOLD_NAMES]
            print(
                f"\n{label}: full-5-fold mean val={sum(all_vs)/len(all_vs):.4f} test={sum(all_ts)/len(all_ts):.4f} | "
                f"folds3-5-only mean val={sum(comp_vs)/len(comp_vs):.4f} test={sum(comp_ts)/len(comp_ts):.4f}"
            )

            r_last = results[(label, folds[-1].name)]

            def agg(field, metric):
                return sum(getattr(results[(label, f.name)], field)[metric] for f in folds) / len(folds)

            row = {
                "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
                "run_name": f"injury_missingdata_{label}_full5fold",
                "val_diff_mae": round(agg("val_metrics", "diff_mae"), 3),
                "test_diff_mae": round(agg("test_metrics", "diff_mae"), 3),
                "val_diff_within_5": round(agg("val_metrics", "diff_within_5"), 4),
                "test_diff_within_5": round(agg("test_metrics", "diff_within_5"), 4),
                "val_total_mae": round(agg("val_metrics", "total_mae"), 3),
                "test_total_mae": round(agg("test_metrics", "total_mae"), 3),
                "val_win_acc": round(agg("val_metrics", "win_accuracy"), 4),
                "test_win_acc": round(agg("test_metrics", "win_accuracy"), 4),
                "val_brier": round(agg("val_metrics", "brier_score"), 4),
                "test_brier": round(agg("test_metrics", "brier_score"), 4),
                "n_features": r_last.n_features,
                "injury_enabled": True,
                "rolling_windows": ",".join(str(w) for w in cfg.features.rolling_windows),
                "val_score_mean": round(sum(all_vs) / len(all_vs), 4),
                "val_score_per_fold": ",".join(f"{results[(label, f.name)].val_score:.4f}" for f in folds),
                "test_score_mean": round(sum(all_ts) / len(all_ts), 4),
                "protocol": "cv",
                "session_id": args.session_id,
                "notes": (
                    f"E4 injury missing-data handling: missing_value_strategy={label}, full 5-fold "
                    f"(fold1-relevant by construction). folds3-5-only mean (the folds D1 confirmed have "
                    f"real train-time injury coverage): val={sum(comp_vs)/len(comp_vs):.4f} "
                    f"test={sum(comp_ts)/len(comp_ts):.4f} -- this is E4's 'folds-start-at-2021' variant, "
                    "a restricted slice of this same run, not a separate training pass. Compare zero_fill "
                    "vs native_nan rows for the ablation delta, on both the full-5-fold and folds3-5-only view."
                ),
            }
            session_csv.parent.mkdir(parents=True, exist_ok=True)
            write_header = not session_csv.exists()
            with open(session_csv, "a", newline="") as fh:
                writer = csv.DictWriter(fh, fieldnames=row.keys())
                if write_header:
                    writer.writeheader()
                writer.writerow(row)
        print(f"\nLogged both rows to {session_csv}")

    finally:
        set_missing_value_strategy(COMMITTED_DEFAULT)
        cfg_check = load_config()
        assert (
            cfg_check.injury_features.missing_value_strategy.value == COMMITTED_DEFAULT
        ), "FAILED TO RESTORE"
        print(f"config restored: missing_value_strategy={COMMITTED_DEFAULT}")


if __name__ == "__main__":
    main()
