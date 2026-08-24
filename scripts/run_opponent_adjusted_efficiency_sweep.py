"""Retrospective opponent-adjustment (docs/NEXT_PHASE_SESSIONS.md backlog
item 5): extends `compute_opponent_adjusted_form_scores`' own template
(per-game signed residual vs. opponent's pre-game quality, then a shift(1)
rolling mean) from win/loss to off_eff/def_eff, via the new
`opponent_adjusted_off_score`/`opponent_adjusted_def_score` columns
(season_motivation.py, gated by `season_motivation.opponent_adjusted_efficiency_enabled`).

Given the documented instability in the closest precedent
(`opponent_adjusted_form_score`: passed CV at window=10 only, sign-inverted
at windows 5/15 -- docs/features/season_motivation_log.md section 10), this
session tests ALL of L5/L10/L20 (this project's standard rolling windows),
full 5-fold CV each, not a single window in isolation.

Per this session's own instruction, reports pre-CV correlation of the new
columns against (a) existing off_eff_L{w}/def_eff_L{w} and (b)
`opponent_quality_features`' opp_def_quality_L{w}/opp_off_quality_L{w}, at
every window, BEFORE running CV -- same discipline as the elo_momentum/
official_pace sessions. Also checks correlation between the two new columns
themselves (off_score vs. def_score), per the official_pace lesson that
collinearity among a candidate's own new columns is a distinct failure mode
from collinearity against existing features.

Usage: venv/bin/python3 scripts/run_opponent_adjusted_efficiency_sweep.py --session-id <id>

Temporarily edits configs/config.yaml's season_motivation.opponent_adjusted_efficiency_enabled/
_window, restores the committed defaults (false / 10) afterward regardless
of outcome. SIGTERM/SIGINT-safe restore, same pattern as the other
orchestration scripts in this directory.
"""

import argparse
import csv
import os
import re
import signal
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

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

COMMITTED_DEFAULTS = {
    "opponent_adjusted_efficiency_enabled": False,
    "opponent_adjusted_efficiency_window": 10,
}
WINDOWS = [5, 10, 20]


def set_enabled(value: bool):
    text = CONFIG_PATH.read_text()
    new_text, n_subs = re.subn(
        r"(^  opponent_adjusted_efficiency_enabled: )(true|false)",
        lambda m: m.group(1) + ("true" if value else "false"),
        text,
        count=1,
        flags=re.MULTILINE,
    )
    assert (
        n_subs == 1
    ), "opponent_adjusted_efficiency_enabled not found at top-level season_motivation indentation"
    CONFIG_PATH.write_text(new_text)


def set_window(value: int):
    text = CONFIG_PATH.read_text()
    new_text, n_subs = re.subn(
        r"(^  opponent_adjusted_efficiency_window: )\d+",
        lambda m: m.group(1) + str(value),
        text,
        count=1,
        flags=re.MULTILINE,
    )
    assert (
        n_subs == 1
    ), "opponent_adjusted_efficiency_window not found at top-level season_motivation indentation"
    CONFIG_PATH.write_text(new_text)


def restore_defaults():
    set_enabled(COMMITTED_DEFAULTS["opponent_adjusted_efficiency_enabled"])
    set_window(COMMITTED_DEFAULTS["opponent_adjusted_efficiency_window"])


def report_correlations(train_features, window):
    pairs = []
    for side in ["home_team", "away_team"]:
        oae_off = f"{side}_opponent_adjusted_off_score"
        oae_def = f"{side}_opponent_adjusted_def_score"
        off_eff = f"{side}_off_eff_L{window}"
        def_eff = f"{side}_def_eff_L{window}"
        opp_off_q = f"{side}_opp_off_quality_L{window}"
        opp_def_q = f"{side}_opp_def_quality_L{window}"

        for new_col, existing_col, label in [
            (oae_off, off_eff, f"{side} opponent_adjusted_off_score vs off_eff_L{window}"),
            (oae_def, def_eff, f"{side} opponent_adjusted_def_score vs def_eff_L{window}"),
            (oae_off, opp_def_q, f"{side} opponent_adjusted_off_score vs opp_def_quality_L{window}"),
            (oae_def, opp_off_q, f"{side} opponent_adjusted_def_score vs opp_off_quality_L{window}"),
        ]:
            if new_col not in train_features.columns or existing_col not in train_features.columns:
                print(f"    SKIP {label}: column missing")
                continue
            sub = train_features[[new_col, existing_col]].apply(pd.to_numeric, errors="coerce").dropna()
            r = np.corrcoef(sub[new_col], sub[existing_col])[0, 1]
            pairs.append((label, r))

        # Between the two new columns themselves -- the official_pace lesson.
        if oae_off in train_features.columns and oae_def in train_features.columns:
            sub = train_features[[oae_off, oae_def]].apply(pd.to_numeric, errors="coerce").dropna()
            r = np.corrcoef(sub[oae_off], sub[oae_def])[0, 1]
            pairs.append(
                (f"{side} opponent_adjusted_off_score vs opponent_adjusted_def_score (NEW vs NEW)", r)
            )

    print(f"\n  --- Pre-CV correlation check, window={window} ---")
    for label, r in pairs:
        print(f"    {label}: r={r:+.3f}")
    return pairs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session-id", required=True)
    args = parser.parse_args()
    session_csv = REPO / "results" / "sessions" / f"{args.session_id}.csv"

    cfg_check = load_config()
    if (
        cfg_check.season_motivation.opponent_adjusted_efficiency_enabled
        != COMMITTED_DEFAULTS["opponent_adjusted_efficiency_enabled"]
        or cfg_check.season_motivation.opponent_adjusted_efficiency_window
        != COMMITTED_DEFAULTS["opponent_adjusted_efficiency_window"]
    ):
        print(
            "WARNING: opponent_adjusted_efficiency config not at committed defaults at startup -- resetting."
        )
        restore_defaults()

    try:
        results_summary = []
        for window in WINDOWS:
            set_enabled(True)
            set_window(window)
            cfg = load_config()
            assert cfg.season_motivation.opponent_adjusted_efficiency_enabled is True
            assert cfg.season_motivation.opponent_adjusted_efficiency_window == window

            label = f"opponent_adjusted_efficiency_L{window}"
            print(f"\n### {label} ###", flush=True)

            folds = cfg.cv.folds
            validate_fold_definitions(folds)

            # Fold5 train_features (largest window) for the pre-CV correlation check,
            # same fold used by family_correlation_vif.py's own precedent.
            fold5 = folds[-1]
            probe = run_split(
                cfg,
                fold5.train_end_date,
                fold5.validation_start_date,
                fold5.validation_end_date,
                fold5.test_start_date,
                fold5.test_end_date,
                keep_artifacts=True,
            )
            report_correlations(probe.train_features, window)

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
                    f"Retrospective opponent-adjustment (docs/NEXT_PHASE_SESSIONS.md backlog item 5): "
                    f"opponent_adjusted_off_score/opponent_adjusted_def_score at window={window}. "
                    "Compare val_score_mean/per-fold against b2_elo_momentum baseline "
                    "(val_score_mean=1.3803, per-fold=1.4336,1.3888,1.3708,1.3479,1.3605)."
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

        print("\n=== OPPONENT-ADJUSTED EFFICIENCY WINDOW SWEEP SUMMARY (lower val_score = better) ===")
        for label, vm, tm, per_fold in sorted(results_summary, key=lambda x: x[1]):
            print(f"  {label}: val_mean={vm:.4f} test_mean={tm:.4f} per_fold={per_fold}")
        print(f"\nLogged {len(results_summary)} rows to {session_csv}")

    finally:
        restore_defaults()
        cfg_check = load_config()
        assert (
            cfg_check.season_motivation.opponent_adjusted_efficiency_enabled
            is COMMITTED_DEFAULTS["opponent_adjusted_efficiency_enabled"]
        ), "FAILED TO RESTORE opponent_adjusted_efficiency_enabled"
        assert (
            cfg_check.season_motivation.opponent_adjusted_efficiency_window
            == COMMITTED_DEFAULTS["opponent_adjusted_efficiency_window"]
        ), "FAILED TO RESTORE opponent_adjusted_efficiency_window"
        print("config restored: opponent_adjusted_efficiency at committed defaults")


if __name__ == "__main__":
    main()
