"""Re-run champion_cv_baseline under full 5-fold CV with the injury_layer.py
multi-archetype accumulation fix applied (see EXPERIMENTS.md's D2 section and
PR #44) -- same committed config as the original champion_cv_baseline
(style_matchup.raw_features_enabled=true, preferred_opponent_delta_enabled=true,
style_matchup.enabled=false), just re-scored against the corrected
matchup_fingerprints cache. Doesn't touch cv_harness.py or the fold
definitions; reuses run_split unmodified, same pattern as every other
orchestration script in this directory.

Usage: venv/bin/python3 scripts/run_champion_rebaseline.py [--run-name ...] [--notes ...]
"""

import argparse
import csv
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.append(str(REPO))
os.chdir(REPO)

from src.evaluation.cv_harness import run_split, validate_fold_definitions  # noqa: E402
from src.utils.config_loader import load_config  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", default="champion_cv_baseline_post_injury_fix")
    parser.add_argument(
        "--notes",
        default=(
            "Re-run of champion_cv_baseline with injury_layer.py's multi-archetype delta "
            "accumulation bug fixed (PR #44) -- same committed config, cache rebuilt fresh "
            "with the fix. Compare against champion_cv_baseline (val_score_mean=1.3850, "
            "val_score_per_fold=1.4407,1.3876,1.3804,1.3579,1.3585)."
        ),
    )
    args = parser.parse_args()

    cfg = load_config()
    assert cfg.style_matchup.raw_features_enabled is True
    assert cfg.season_motivation.preferred_opponent_delta_enabled is True
    assert cfg.style_matchup.enabled is False

    folds = cfg.cv.folds
    validate_fold_definitions(folds)

    fold_results = []
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
            f"{f.name}: val_score={result.val_score:.4f} "
            f"(diff_mae={result.val_metrics['diff_mae']:.2f}, total_mae={result.val_metrics['total_mae']:.2f}) | "
            f"test_score={result.test_score:.4f} "
            f"(diff_mae={result.test_metrics['diff_mae']:.2f}, total_mae={result.test_metrics['total_mae']:.2f})",
            flush=True,
        )
        fold_results.append((f.name, result))

    val_scores = [r.val_score for _, r in fold_results]
    test_scores = [r.test_score for _, r in fold_results]
    val_mean = sum(val_scores) / len(val_scores)
    test_mean = sum(test_scores) / len(test_scores)
    print(f"\nMean val_score:  {val_mean:.4f}")
    print(f"Mean test_score: {test_mean:.4f}")

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
        "session_id": "",
        "notes": args.notes,
    }

    target = REPO / "outputs" / "experiments_v2.csv"
    write_header = not target.exists()
    with open(target, "a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=row.keys())
        if write_header:
            writer.writeheader()
        writer.writerow(row)
    print(f"\nLogged to {target}")


if __name__ == "__main__":
    main()
