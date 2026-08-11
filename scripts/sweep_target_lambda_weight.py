"""target_lambda_weight sweep (EXPERIMENTS.md section 3.3 follow-up): now that
model.target_formulation=diff_total is champion, is its target_lambda_weight
(0.5, carried over unchanged from compute_composite_score's own diff/total
weighting) actually the best value for the TRAINING loss? diff_total scales
`total` by sqrt(target_lambda_weight) before fitting MultiRMSE, so this
directly controls how much the fitted model prioritizes total-point accuracy
vs. point-differential accuracy -- unlike home_away mode, where both
dimensions were always fit unweighted.

Two stages:
  1. Cheap screen -- folds 3-5 only, wider grid, to bound cost per CLAUDE.md's
     "cheap screening runs may use the last 3 folds only."
  2. Full 5-fold CV for the 0.5 champion (reference) plus any candidate that
     looks promising on the cheap screen -- promotion requires the full-CV
     guardrail, not just a 3-fold read.

Usage: venv/bin/python3 scripts/sweep_target_lambda_weight.py

Temporarily toggles configs/config.yaml's model.target_lambda_weight, restores
the committed default (0.5) afterward regardless of outcome. SIGTERM/SIGINT-
safe restore, same pattern as run_target_formulation_experiment.py.
"""

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

COMMITTED_DEFAULT = 0.5
SCREEN_GRID = [0.1, 0.25, 0.5, 0.75, 1.0, 2.0]


def set_lambda(value: float):
    text = CONFIG_PATH.read_text()
    new_text = re.sub(
        r"(target_lambda_weight: )([0-9.]+)",
        lambda m: m.group(1) + repr(value),
        text,
        count=1,
    )
    assert new_text != text or f"target_lambda_weight: {value}" in text
    CONFIG_PATH.write_text(new_text)


def _run_fold(cfg, f):
    return run_split(
        cfg,
        f.train_end_date,
        f.validation_start_date,
        f.validation_end_date,
        f.test_start_date,
        f.test_end_date,
    )


def _log_row(target_csv, run_name, folds, results_by_fold, cfg, notes):
    vs = [results_by_fold[f.name].val_score for f in folds]
    ts = [results_by_fold[f.name].test_score for f in folds]
    r_last = results_by_fold[folds[-1].name]

    def agg(field, metric):
        return sum(getattr(results_by_fold[f.name], field)[metric] for f in folds) / len(folds)

    row = {
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "run_name": run_name,
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
        "val_score_mean": round(sum(vs) / len(vs), 4),
        "val_score_per_fold": ",".join(f"{results_by_fold[f.name].val_score:.4f}" for f in folds),
        "test_score_mean": round(sum(ts) / len(ts), 4),
        "protocol": "cv",
        "session_id": "",
        "notes": notes,
    }
    target_csv.parent.mkdir(parents=True, exist_ok=True)
    write_header = not target_csv.exists()
    with open(target_csv, "a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=row.keys())
        if write_header:
            writer.writeheader()
        writer.writerow(row)
    return row


def main():
    target_csv = REPO / "outputs" / "experiments_v2.csv"

    if load_config().model.target_lambda_weight != COMMITTED_DEFAULT:
        print("WARNING: target_lambda_weight not at committed default at startup -- resetting.", flush=True)
        set_lambda(COMMITTED_DEFAULT)

    try:
        cfg = load_config()
        all_folds = cfg.cv.folds
        validate_fold_definitions(all_folds)
        screen_folds = all_folds[-3:]

        print(
            f"\n=== Stage 1: cheap screen (folds {[f.name for f in screen_folds]}), grid={SCREEN_GRID} ===",
            flush=True,
        )
        screen_results = {}
        for lam in SCREEN_GRID:
            set_lambda(lam)
            cfg = load_config()
            assert cfg.model.target_lambda_weight == lam
            print(f"\n-- target_lambda_weight={lam} --", flush=True)
            by_fold = {}
            for f in screen_folds:
                result = _run_fold(cfg, f)
                by_fold[f.name] = result
                print(
                    f"  {f.name}: val_score={result.val_score:.4f} "
                    f"(diff_mae={result.val_metrics['diff_mae']:.2f}, total_mae={result.val_metrics['total_mae']:.2f})",
                    flush=True,
                )
            screen_results[lam] = by_fold

        print("\n=== Screen summary (mean val_score over folds 3-5) ===")
        means = {}
        for lam in SCREEN_GRID:
            vs = [screen_results[lam][f.name].val_score for f in screen_folds]
            means[lam] = sum(vs) / len(vs)
            print(f"  target_lambda_weight={lam}: mean val_score={means[lam]:.4f}")

        best_lam = min(means, key=means.get)
        print(
            f"\nBest on cheap screen: target_lambda_weight={best_lam} (mean val_score={means[best_lam]:.4f})"
        )

        candidates = sorted({COMMITTED_DEFAULT, best_lam})
        print(f"\n=== Stage 2: full 5-fold CV for candidates {candidates} ===", flush=True)
        full_results = {}
        for lam in candidates:
            set_lambda(lam)
            cfg = load_config()
            print(f"\n-- target_lambda_weight={lam} (full CV) --", flush=True)
            by_fold = {}
            for f in all_folds:
                result = _run_fold(cfg, f)
                by_fold[f.name] = result
                print(
                    f"  {f.name}: val_score={result.val_score:.4f} test_score={result.test_score:.4f}",
                    flush=True,
                )
            full_results[lam] = by_fold

            notes = (
                f"target_lambda_weight sweep (EXPERIMENTS.md section 3.3 follow-up): "
                f"target_lambda_weight={lam}, diff_total mode, full 5-fold, champion config otherwise unchanged. "
                f"Cheap 3-fold screen over {SCREEN_GRID} motivated this candidate "
                f"({'committed default, reference point' if lam == COMMITTED_DEFAULT else 'best on screen'})."
            )
            row = _log_row(target_csv, f"target_lambda_weight_{lam}", all_folds, by_fold, cfg, notes)
            print(f"Logged: val_score_mean={row['val_score_mean']}")

        print("\n=== Full-CV comparison ===")
        for f in all_folds:
            parts = [f"{lam}={full_results[lam][f.name].val_score:.4f}" for lam in candidates]
            print(f"  {f.name}: " + "  ".join(parts))
        for lam in candidates:
            vs = [full_results[lam][f.name].val_score for f in all_folds]
            print(f"target_lambda_weight={lam}: mean val_score={sum(vs)/len(vs):.4f}")

    finally:
        set_lambda(COMMITTED_DEFAULT)
        cfg_check = load_config()
        assert cfg_check.model.target_lambda_weight == COMMITTED_DEFAULT, "FAILED TO RESTORE"
        print(f"\nconfig restored: target_lambda_weight={COMMITTED_DEFAULT}")


if __name__ == "__main__":
    main()
