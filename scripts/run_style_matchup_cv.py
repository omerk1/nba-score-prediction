"""Evaluate style_matchup.enabled=true (the KNN-similarity score) under the
real 5-fold expanding-window CV, rebuilding the precompute cache once per
fold with that fold's own train_end_date as the z-score cutoff -- the cache
(src/matchups/precompute_scores.py) has no fold-awareness of its own, unlike
the rest of the feature pipeline's context_end_date threading, so evaluating
it under CV without this per-fold rebuild would leak later folds' data into
earlier folds' z-score normalization.

Usage: venv/bin/python3 scripts/run_style_matchup_cv.py --session-id <id>
           [--run-name style_matchup_knn_cv] [--notes "..."]

Reuses cv_harness.run_split unmodified per fold -- does not touch the CV
harness itself.

Temporarily flips configs/config.yaml's style_matchup.enabled to true for the
duration of the run, restores it to false (the committed default) afterward
regardless of success/failure. Registers a SIGTERM/SIGINT handler so a
graceful kill still restores the config -- a bare signal does NOT run
`finally` blocks by default, and a prior run of this pattern was SIGKILLed
mid-flight and left the config stuck at enabled=true.
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
from src.matchups.precompute_scores import precompute_and_cache  # noqa: E402
from src.utils.config_loader import load_config  # noqa: E402


def set_style_matchup_enabled(value: bool):
    text = CONFIG_PATH.read_text()
    new_text = re.sub(
        r"(style_matchup:\n(?:.*\n)*?  enabled: )(true|false)",
        lambda m: m.group(1) + ("true" if value else "false"),
        text,
        count=1,
    )
    assert new_text != text or f"enabled: {str(value).lower()}" in text, "enabled flag not found/changed"
    CONFIG_PATH.write_text(new_text)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session-id", required=True, help="Research session_id, e.g. 20260810_1400_slug")
    parser.add_argument("--run-name", default="style_matchup_knn_cv")
    parser.add_argument("--notes", default="style_matchup.enabled=true (KNN-similarity score) under full CV.")
    args = parser.parse_args()
    session_csv = REPO / "results" / "sessions" / f"{args.session_id}.csv"

    # Self-heal: a prior killed run may have left this stuck at true.
    if load_config().style_matchup.enabled is not False:
        print("WARNING: style_matchup.enabled was not false at startup -- resetting first.", flush=True)
        set_style_matchup_enabled(False)
    set_style_matchup_enabled(True)
    try:
        cfg = load_config()
        assert cfg.style_matchup.enabled is True, "config did not pick up enabled=true"
        folds = cfg.cv.folds
        validate_fold_definitions(folds)

        fold_results = []
        for f in folds:
            print(
                f"=== {f.name}: rebuilding style_matchup cache (zscore_cutoff={f.train_end_date}) ===",
                flush=True,
            )
            summary = precompute_and_cache(zscore_cutoff_date=f.train_end_date)
            print(f"    cache summary: {summary}", flush=True)

            result = run_split(
                cfg,
                f.train_end_date,
                f.validation_start_date,
                f.validation_end_date,
                f.test_start_date,
                f.test_end_date,
            )
            print(
                f"    {f.name}: val_score={result.val_score:.4f} "
                f"(diff_mae={result.val_metrics['diff_mae']:.2f}, "
                f"total_mae={result.val_metrics['total_mae']:.2f}) | "
                f"test_score={result.test_score:.4f} "
                f"(diff_mae={result.test_metrics['diff_mae']:.2f}, "
                f"total_mae={result.test_metrics['total_mae']:.2f})",
                flush=True,
            )
            fold_results.append((f.name, result))

        val_scores = [r.val_score for _, r in fold_results]
        test_scores = [r.test_score for _, r in fold_results]
        val_mean = sum(val_scores) / len(val_scores)
        test_mean = sum(test_scores) / len(test_scores)

        print(f"\nMean val_score:  {val_mean:.4f}")
        print(f"Mean test_score: {test_mean:.4f}")
        print("(compare against the current champion row in outputs/experiments_v2.csv)")

        def agg(metric):
            return sum(r.val_metrics[metric] for _, r in fold_results) / len(fold_results)

        def agg_test(metric):
            return sum(r.test_metrics[metric] for _, r in fold_results) / len(fold_results)

        last = fold_results[-1][1]
        row = {
            "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
            "run_name": args.run_name,
            "val_diff_mae": round(agg("diff_mae"), 3),
            "test_diff_mae": round(agg_test("diff_mae"), 3),
            "val_diff_within_5": round(agg("diff_within_5"), 4),
            "test_diff_within_5": round(agg_test("diff_within_5"), 4),
            "val_total_mae": round(agg("total_mae"), 3),
            "test_total_mae": round(agg_test("total_mae"), 3),
            "val_win_acc": round(agg("win_accuracy"), 4),
            "test_win_acc": round(agg_test("win_accuracy"), 4),
            "val_brier": round(agg("brier_score"), 4),
            "test_brier": round(agg_test("brier_score"), 4),
            "n_features": last.n_features,
            "injury_enabled": True,
            "rolling_windows": ",".join(str(w) for w in cfg.features.rolling_windows),
            "val_score_mean": round(val_mean, 4),
            "val_score_per_fold": ",".join(f"{r.val_score:.4f}" for _, r in fold_results),
            "test_score_mean": round(test_mean, 4),
            "protocol": "cv",
            "session_id": args.session_id,
            "notes": args.notes,
        }

        session_csv.parent.mkdir(parents=True, exist_ok=True)
        write_header = not session_csv.exists()
        with open(session_csv, "a", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=row.keys())
            if write_header:
                writer.writeheader()
            writer.writerow(row)
        print(f"\nLogged to {session_csv}")

    finally:
        set_style_matchup_enabled(False)
        cfg_check = load_config()
        assert cfg_check.style_matchup.enabled is False, "FAILED TO RESTORE enabled=false"
        print("config restored: style_matchup.enabled=false")


if __name__ == "__main__":
    main()
