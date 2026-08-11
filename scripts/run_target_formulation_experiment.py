"""Target reformulation experiment (EXPERIMENTS.md section 3.3): does fitting
MultiRMSE on [diff, total] instead of [home, away] improve the composite
score, by aligning the training loss with what the metric actually rewards?

Runs the champion config under full 5-fold CV, both target_formulation
values. home_away is expected to exactly reproduce champion_cv_baseline_post_
injury_fix (a correctness check on the refactor itself, not just the
experiment) -- ScorePredictor.predict()'s public contract is unchanged in
that mode, so this run should be byte-identical to the existing champion row.

Usage: venv/bin/python3 scripts/run_target_formulation_experiment.py

Temporarily toggles configs/config.yaml's model.target_formulation, restores
the committed default (home_away) afterward regardless of outcome.
SIGTERM/SIGINT-safe restore, same pattern as the other orchestration scripts.
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

COMMITTED_DEFAULT = "home_away"


def set_target_formulation(value: str):
    text = CONFIG_PATH.read_text()
    new_text = re.sub(
        r"(target_formulation: )(home_away|diff_total)",
        lambda m: m.group(1) + value,
        text,
        count=1,
    )
    assert new_text != text or f"target_formulation: {value}" in text
    CONFIG_PATH.write_text(new_text)


def main():
    # Manual/interactive run, not an autonomous session -- logs directly to
    # experiments_v2.csv per CLAUDE.md ("Manual one-off experiments run
    # interactively may log directly to experiments_v2.csv"), session_id blank,
    # same convention as champion_cv_baseline/preferred_opponent_delta_treatment.
    target_csv = REPO / "outputs" / "experiments_v2.csv"

    if load_config().model.target_formulation.value != COMMITTED_DEFAULT:
        print("WARNING: target_formulation not at committed default at startup -- resetting.", flush=True)
        set_target_formulation(COMMITTED_DEFAULT)

    try:
        cfg = load_config()
        folds = cfg.cv.folds
        validate_fold_definitions(folds)

        results = {}  # (label, fold_name) -> SplitResult
        for label, value in [("home_away", "home_away"), ("diff_total", "diff_total")]:
            set_target_formulation(value)
            cfg = load_config()
            assert cfg.model.target_formulation.value == value
            print(f"\n### target_formulation={value} ###", flush=True)
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

        print("\n=== home_away vs diff_total, per fold ===")
        for f in folds:
            ha, dt = results[("home_away", f.name)], results[("diff_total", f.name)]
            print(
                f"  {f.name}: val_score  home_away={ha.val_score:.4f}  diff_total={dt.val_score:.4f}  "
                f"delta(dt-ha)={dt.val_score - ha.val_score:+.4f}   "
                f"test_score  home_away={ha.test_score:.4f}  diff_total={dt.test_score:.4f}  "
                f"delta(dt-ha)={dt.test_score - ha.test_score:+.4f}"
            )

        for label in ("home_away", "diff_total"):
            vs = [results[(label, f.name)].val_score for f in folds]
            ts = [results[(label, f.name)].test_score for f in folds]
            print(f"\n{label}: mean val_score={sum(vs)/len(vs):.4f}  mean test_score={sum(ts)/len(ts):.4f}")

            r_last = results[(label, folds[-1].name)]

            def agg(field, metric):
                return sum(getattr(results[(label, f.name)], field)[metric] for f in folds) / len(folds)

            row = {
                "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
                "run_name": f"target_formulation_{label}",
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
                "val_score_per_fold": ",".join(f"{results[(label, f.name)].val_score:.4f}" for f in folds),
                "test_score_mean": round(sum(ts) / len(ts), 4),
                "protocol": "cv",
                "session_id": "",
                "notes": (
                    f"Target reformulation experiment (EXPERIMENTS.md section 3.3): "
                    f"target_formulation={label}, full 5-fold, champion config otherwise unchanged. "
                    "Compare home_away (expected byte-identical to champion_cv_baseline_post_injury_fix) "
                    "vs diff_total rows for the effect of aligning the training loss with the composite metric."
                ),
            }
            target_csv.parent.mkdir(parents=True, exist_ok=True)
            write_header = not target_csv.exists()
            with open(target_csv, "a", newline="") as fh:
                writer = csv.DictWriter(fh, fieldnames=row.keys())
                if write_header:
                    writer.writeheader()
                writer.writerow(row)
        print(f"\nLogged both rows to {target_csv}")

    finally:
        set_target_formulation(COMMITTED_DEFAULT)
        cfg_check = load_config()
        assert cfg_check.model.target_formulation.value == COMMITTED_DEFAULT, "FAILED TO RESTORE"
        print(f"config restored: target_formulation={COMMITTED_DEFAULT}")


if __name__ == "__main__":
    main()
