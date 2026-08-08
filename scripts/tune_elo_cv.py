"""E1: Expand Elo -- grid-search elo_features config under the real 5-fold
expanding-window CV (session rs_20260808_1).

`scripts/tune_elo.py` cannot be reused as-is: it's single-split only (never
imports cv_harness) and holds CatBoost hyperparameters fixed at values that
differ from cv_harness.run_split's hardcoded ones (depth 3 vs. 6, subsample
0.92 vs. 0.8, etc.) -- tuning Elo against that model config wouldn't transfer
to the model we actually run. This script instead reuses run_split unmodified
per fold, so every candidate is scored with the exact model config the
champion uses.

Hypothesis (EXPERIMENTS.md E1): current k_factor=11.02/season_regression=0.522
were tuned via tune_elo.py under the old single_split protocol, predating the
CV harness and never re-validated per-fold. Elo's fold1 weakness (ratio 0.74,
per the family-importance inventory) plus its efficiency (high permutation
cost relative to modest CatBoost share) makes season_regression -- which
governs how fast ratings regress to the mean between seasons, directly
relevant to a fold1 cold-start problem -- the primary axis to test.

Protocol: full 5-fold per candidate (fold1 is the object of study, no
cheap-3-fold screening). Small, hypothesis-driven grid to bound wall-clock
cost; extend only if a direction looks promising.

Elo is computed fresh per split from context_end_date-bounded raw games (no
separate cache to rebuild per fold, unlike style_matchup's KNN score).

Usage: venv/bin/python3 scripts/tune_elo_cv.py --session-id <id> [--grid season_regression|both]

Temporarily edits configs/config.yaml's elo_features.season_regression (and
optionally k_factor), restores the committed defaults afterward regardless of
outcome. SIGTERM/SIGINT-safe restore, same pattern as the other orchestration
scripts in this directory.
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

COMMITTED_DEFAULTS = {"k_factor": 11.02, "home_advantage": 117.87, "season_regression": 0.522}


def set_elo_field(field: str, value: float):
    text = CONFIG_PATH.read_text()
    pattern = rf"(^  {field}: )[0-9.]+"
    new_text, n_subs = re.subn(pattern, lambda m: m.group(1) + repr(value), text, count=1, flags=re.MULTILINE)
    assert n_subs == 1, f"{field} not found at top-level elo_features indentation"
    CONFIG_PATH.write_text(new_text)


def restore_defaults():
    for field, value in COMMITTED_DEFAULTS.items():
        set_elo_field(field, value)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session-id", required=True)
    parser.add_argument(
        "--grid",
        choices=["season_regression", "both"],
        default="season_regression",
        help="season_regression: 1-D grid on season_regression alone (default, primary hypothesis). "
        "both: also cross with a small k_factor grid.",
    )
    args = parser.parse_args()
    session_csv = REPO / "results" / "sessions" / f"{args.session_id}.csv"

    cfg_check = load_config()
    for field, value in COMMITTED_DEFAULTS.items():
        actual = getattr(cfg_check.elo_features, field)
        if abs(actual - value) > 1e-9:
            print(
                f"WARNING: elo_features.{field}={actual} != committed default {value} -- resetting.",
                flush=True,
            )
            restore_defaults()
            break

    season_regression_grid = [0.30, 0.40, 0.522, 0.65, 0.80]
    k_factor_grid = [11.02] if args.grid == "season_regression" else [7.0, 11.02, 16.0]

    candidates = [
        {"season_regression": sr, "k_factor": kf} for sr in season_regression_grid for kf in k_factor_grid
    ]
    print(
        f"E1 grid: {len(candidates)} candidates x 5 folds = {len(candidates) * 5} full-CV fold trainings",
        flush=True,
    )

    try:
        results_summary = []
        for cand in candidates:
            set_elo_field("season_regression", cand["season_regression"])
            set_elo_field("k_factor", cand["k_factor"])
            cfg = load_config()
            assert abs(cfg.elo_features.season_regression - cand["season_regression"]) < 1e-9
            assert abs(cfg.elo_features.k_factor - cand["k_factor"]) < 1e-9

            label = f"elo_sr{cand['season_regression']}_k{cand['k_factor']}"
            print(f"\n### {label} ###", flush=True)

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
                    f"    {f.name}: val_score={result.val_score:.4f} "
                    f"(diff_mae={result.val_metrics['diff_mae']:.2f}, total_mae={result.val_metrics['total_mae']:.2f}) | "
                    f"test_score={result.test_score:.4f}",
                    flush=True,
                )
                fold_results.append((f.name, result))

            val_scores = [r.val_score for _, r in fold_results]
            test_scores = [r.test_score for _, r in fold_results]
            val_mean = sum(val_scores) / len(val_scores)
            test_mean = sum(test_scores) / len(test_scores)
            print(f"  {label}: mean val_score={val_mean:.4f} mean test_score={test_mean:.4f}", flush=True)

            def agg(metric):
                return sum(r.val_metrics[metric] for _, r in fold_results) / len(fold_results)

            def agg_test(metric):
                return sum(r.test_metrics[metric] for _, r in fold_results) / len(fold_results)

            last = fold_results[-1][1]
            row = {
                "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
                "run_name": label,
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
                "notes": (
                    f"E1 Expand Elo grid candidate: season_regression={cand['season_regression']}, "
                    f"k_factor={cand['k_factor']} (home_advantage held at committed default "
                    f"{COMMITTED_DEFAULTS['home_advantage']}). Compare val_score_mean/per-fold against "
                    "champion_cv_baseline (season_regression=0.522, k_factor=11.02, val_score_mean=1.3850, "
                    "val_score_per_fold=1.4407,1.3876,1.3804,1.3579,1.3585)."
                ),
            }
            session_csv.parent.mkdir(parents=True, exist_ok=True)
            write_header = not session_csv.exists()
            with open(session_csv, "a", newline="") as fh:
                writer = csv.DictWriter(fh, fieldnames=row.keys())
                if write_header:
                    writer.writeheader()
                writer.writerow(row)
            results_summary.append((label, val_mean, test_mean, row["val_score_per_fold"]))

        print("\n=== E1 GRID SUMMARY (sorted by val_score_mean, lower is better) ===")
        for label, vm, tm, per_fold in sorted(results_summary, key=lambda x: x[1]):
            print(f"  {label}: val_mean={vm:.4f} test_mean={tm:.4f} per_fold={per_fold}")
        print(f"\nLogged {len(results_summary)} rows to {session_csv}")

    finally:
        restore_defaults()
        cfg_check = load_config()
        for field, value in COMMITTED_DEFAULTS.items():
            actual = getattr(cfg_check.elo_features, field)
            assert abs(actual - value) < 1e-9, f"FAILED TO RESTORE elo_features.{field}"
        print("config restored: elo_features at committed defaults")


if __name__ == "__main__":
    main()
