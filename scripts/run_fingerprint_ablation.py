"""Cheap-screen ablation (CLAUDE.md: "cheap screening runs may use the last 3
folds only") for style_matchup.raw_features_enabled -- does the adopted
raw-fingerprint block (18 dims) still earn its place under the real,
leak-fixed CV harness (src/evaluation/cv_harness.py), not just a single split?

Runs BOTH raw_features_enabled=true (current committed default) and =false
fresh, on the last 3 CV folds only, for a clean directly-comparable pair
rather than reusing remembered numbers from earlier runs.

Usage: venv/bin/python3 scripts/run_fingerprint_ablation.py --session-id <id>

Temporarily toggles configs/config.yaml, restores the committed default
(true) afterward regardless of outcome. Registers a SIGTERM/SIGINT handler so
a graceful kill still restores the config -- a bare signal does NOT run
`finally` blocks by default, and a prior run of this pattern was SIGKILLed
mid-flight and left the config stuck at a non-default value.

This script is specific to style_matchup.raw_features_enabled; it isn't a
generic "ablate any config flag" tool.
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

COMMITTED_DEFAULT = True  # raw_features_enabled: true is the adopted committed default


def set_raw_features_enabled(value: bool):
    text = CONFIG_PATH.read_text()
    new_text = re.sub(
        r"(raw_features_enabled: )(true|false)",
        lambda m: m.group(1) + ("true" if value else "false"),
        text,
        count=1,
    )
    assert new_text != text or f"raw_features_enabled: {str(value).lower()}" in text
    CONFIG_PATH.write_text(new_text)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session-id", required=True, help="Research session_id, e.g. 20260810_1400_slug")
    args = parser.parse_args()
    session_csv = REPO / "results" / "sessions" / f"{args.session_id}.csv"

    if load_config().style_matchup.raw_features_enabled is not COMMITTED_DEFAULT:
        print("WARNING: raw_features_enabled not at committed default at startup -- resetting.", flush=True)
        set_raw_features_enabled(COMMITTED_DEFAULT)

    try:
        cfg = load_config()
        last_3_folds = cfg.cv.folds[-3:]
        print(f"Cheap screen folds: {[f.name for f in last_3_folds]}", flush=True)

        results = {}  # (config_label, fold_name) -> SplitResult
        for label, enabled in [("on", True), ("off", False)]:
            set_raw_features_enabled(enabled)
            cfg = load_config()
            assert cfg.style_matchup.raw_features_enabled is enabled
            print(f"\n### raw_features_enabled={enabled} ###", flush=True)
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
                "run_name": f"fingerprint_ablation_{label}_cheap3fold",
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
                    f"style_matchup.raw_features_enabled={str(label == 'on').lower()} -- does the adopted "
                    "raw-fingerprint block (18 dims) earn its place under the real CV harness. Compare "
                    "on vs off rows for the ablation delta."
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
        set_raw_features_enabled(COMMITTED_DEFAULT)
        cfg_check = load_config()
        assert cfg_check.style_matchup.raw_features_enabled is COMMITTED_DEFAULT, "FAILED TO RESTORE"
        print(f"config restored: raw_features_enabled={COMMITTED_DEFAULT}")


if __name__ == "__main__":
    main()
