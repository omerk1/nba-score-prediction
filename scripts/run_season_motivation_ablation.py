"""E2: Prune the dead tail -- cheap-screen ablation (CLAUDE.md: "cheap
screening runs may use the last 3 folds only") for
season_motivation.preferred_opponent_delta_enabled (session rs_20260808_1).

Hypothesis (EXPERIMENTS.md E2): this is the only adopted season_motivation
sub-signal, and it has the single negative permutation delta of all 13
families in the family-importance inventory -- does it perform flat-to-
marginally-better OFF under the real, leak-fixed CV harness?

Runs BOTH preferred_opponent_delta_enabled=true (current committed default)
and =false fresh, on the last 3 CV folds only, for a clean directly-comparable
pair. Only this one flag -- rest_features/home_advantage_features stay a
separate future check per EXPERIMENTS.md's own watch-list note, not bundled
in here.

Usage: venv/bin/python3 scripts/run_season_motivation_ablation.py --session-id <id>

Temporarily toggles configs/config.yaml, restores the committed default
(true) afterward regardless of outcome. SIGTERM/SIGINT-safe restore, same
pattern as scripts/run_fingerprint_ablation.py.
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

from src.evaluation.cv_harness import run_split  # noqa: E402
from src.utils.config_loader import load_config  # noqa: E402

COMMITTED_DEFAULT = True  # preferred_opponent_delta_enabled: true is the adopted committed default


def set_preferred_opponent_delta_enabled(value: bool):
    text = CONFIG_PATH.read_text()
    new_text = re.sub(
        r"(preferred_opponent_delta_enabled: )(true|false)",
        lambda m: m.group(1) + ("true" if value else "false"),
        text,
        count=1,
    )
    assert new_text != text or f"preferred_opponent_delta_enabled: {str(value).lower()}" in text
    CONFIG_PATH.write_text(new_text)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session-id", required=True, help="Research session_id, e.g. 20260810_1400_slug")
    args = parser.parse_args()
    session_csv = REPO / "results" / "sessions" / f"{args.session_id}.csv"

    if load_config().season_motivation.preferred_opponent_delta_enabled is not COMMITTED_DEFAULT:
        print(
            "WARNING: preferred_opponent_delta_enabled not at committed default at startup -- resetting.",
            flush=True,
        )
        set_preferred_opponent_delta_enabled(COMMITTED_DEFAULT)

    try:
        cfg = load_config()
        last_3_folds = cfg.cv.folds[-3:]
        print(f"Cheap screen folds: {[f.name for f in last_3_folds]}", flush=True)

        results = {}  # (config_label, fold_name) -> SplitResult
        for label, enabled in [("on", True), ("off", False)]:
            set_preferred_opponent_delta_enabled(enabled)
            cfg = load_config()
            assert cfg.season_motivation.preferred_opponent_delta_enabled is enabled
            print(f"\n### preferred_opponent_delta_enabled={enabled} ###", flush=True)
            for f in last_3_folds:
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
                    f"(diff_mae={result.val_metrics['diff_mae']:.2f}, "
                    f"total_mae={result.val_metrics['total_mae']:.2f}) | "
                    f"test_score={result.test_score:.4f} "
                    f"(diff_mae={result.test_metrics['diff_mae']:.2f}, "
                    f"total_mae={result.test_metrics['total_mae']:.2f})",
                    flush=True,
                )
                results[(label, f.name)] = result

        print("\n=== ON vs OFF, per fold ===")
        for f in last_3_folds:
            on_r, off_r = results[("on", f.name)], results[("off", f.name)]
            print(
                f"  {f.name}: val_score  on={on_r.val_score:.4f}  off={off_r.val_score:.4f}  "
                f"delta(on-off)={on_r.val_score - off_r.val_score:+.4f}   "
                f"test_score  on={on_r.test_score:.4f}  off={off_r.test_score:.4f}  "
                f"delta(on-off)={on_r.test_score - off_r.test_score:+.4f}"
            )

        for label in ("on", "off"):
            vs = [results[(label, f.name)].val_score for f in last_3_folds]
            ts = [results[(label, f.name)].test_score for f in last_3_folds]
            print(f"\n{label}: mean val_score={sum(vs)/len(vs):.4f}  mean test_score={sum(ts)/len(ts):.4f}")

            r_last = results[(label, last_3_folds[-1].name)]

            def agg(field, metric):
                return sum(getattr(results[(label, f.name)], field)[metric] for f in last_3_folds) / len(
                    last_3_folds
                )

            row = {
                "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
                "run_name": f"season_motivation_ablation_{label}_cheap3fold",
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
                "val_score_per_fold": ",".join(
                    f"{results[(label, f.name)].val_score:.4f}" for f in last_3_folds
                ),
                "test_score_mean": round(sum(ts) / len(ts), 4),
                "protocol": "cv",
                "session_id": args.session_id,
                "notes": (
                    f"Cheap-screen ablation (folds 3-5 only, per CLAUDE.md) for "
                    f"season_motivation.preferred_opponent_delta_enabled={str(label == 'on').lower()} -- "
                    "the only adopted season_motivation sub-signal, and the single family with a negative "
                    "permutation delta in the family-importance inventory. Compare on vs off rows for the "
                    "ablation delta."
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
        set_preferred_opponent_delta_enabled(COMMITTED_DEFAULT)
        cfg_check = load_config()
        assert (
            cfg_check.season_motivation.preferred_opponent_delta_enabled is COMMITTED_DEFAULT
        ), "FAILED TO RESTORE"
        print(f"config restored: preferred_opponent_delta_enabled={COMMITTED_DEFAULT}")


if __name__ == "__main__":
    main()
