"""
Baseline vs. treatment ablation for the play-by-play possession feature
(`pbp.enabled`), the one aggregate of 45 that beat raw points margin in the
persistence screens — see docs/features/pbp_possessions_log.md.

Runs both arms over the same folds through the shared `run_split` path, so the
only difference between them is the config flag. Temporarily toggles
configs/config.yaml and restores the committed default in a `finally` block,
mirroring scripts/run_season_motivation_ablation.py.

Usage:
    venv/bin/python3 scripts/run_pbp_ablation.py --session-id 20260918_1835_pbp-luckadj-ablation
    venv/bin/python3 scripts/run_pbp_ablation.py --session-id <id> --folds all
"""

import argparse
import csv
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
CONFIG_PATH = REPO / "configs" / "config.yaml"
sys.path.append(str(REPO))
os.chdir(REPO)

from src.evaluation.cv_harness import run_split  # noqa: E402
from src.utils.config_loader import load_config  # noqa: E402

COMMITTED_DEFAULT = False  # pbp.enabled ships disabled until an ablation says otherwise


def set_pbp_enabled(value: bool) -> None:
    """Flip `enabled` inside the `pbp:` block only, never another section's."""
    text = CONFIG_PATH.read_text()
    block = re.search(r"(?ms)^pbp:\n(?:[ \t].*\n|\n)*", text)
    if block is None:
        raise RuntimeError("no pbp: section in configs/config.yaml")
    new_block = re.sub(r"(\n  enabled: )(true|false)", rf"\g<1>{str(value).lower()}", block.group(0), count=1)
    CONFIG_PATH.write_text(text[: block.start()] + new_block + text[block.end():])
    assert load_config().pbp.enabled is value, "config toggle did not take effect"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--session-id", required=True, help="Research session_id, e.g. 20260918_1835_slug")
    ap.add_argument("--folds", choices=["last3", "all"], default="last3",
                    help="last3 (default): cheap screen per CLAUDE.md. all: full CV, required before promotion.")
    args = ap.parse_args()
    session_csv = REPO / "results" / "sessions" / f"{args.session_id}.csv"

    if load_config().pbp.enabled is not COMMITTED_DEFAULT:
        print("WARNING: pbp.enabled not at committed default at startup -- resetting.", flush=True)
        set_pbp_enabled(COMMITTED_DEFAULT)

    try:
        cfg = load_config()
        folds = cfg.cv.folds if args.folds == "all" else cfg.cv.folds[-3:]
        print(f"Folds: {[f.name for f in folds]}", flush=True)

        results = {}
        for label, enabled in [("off", False), ("on", True)]:
            set_pbp_enabled(enabled)
            cfg = load_config()
            assert cfg.pbp.enabled is enabled
            print(f"\n### pbp.enabled={enabled} ###", flush=True)
            for f in folds:
                r = run_split(cfg, f.train_end_date, f.validation_start_date, f.validation_end_date,
                              f.test_start_date, f.test_end_date)
                print(f"  {f.name}: val_score={r.val_score:.4f} "
                      f"(diff_mae={r.val_metrics['diff_mae']:.2f}, total_mae={r.val_metrics['total_mae']:.2f}) | "
                      f"test_score={r.test_score:.4f} n_features={r.n_features}", flush=True)
                results[(label, f.name)] = r

        print("\n=== ON vs OFF, per fold (validation decides; test is logged only) ===")
        for f in folds:
            on_r, off_r = results[("on", f.name)], results[("off", f.name)]
            print(f"  {f.name}: val on={on_r.val_score:.4f} off={off_r.val_score:.4f} "
                  f"delta={on_r.val_score - off_r.val_score:+.4f}  |  "
                  f"test on={on_r.test_score:.4f} off={off_r.test_score:.4f} "
                  f"delta={on_r.test_score - off_r.test_score:+.4f}")
        on_mean = sum(results[("on", f.name)].val_score for f in folds) / len(folds)
        off_mean = sum(results[("off", f.name)].val_score for f in folds) / len(folds)
        wins = sum(results[("on", f.name)].val_score < results[("off", f.name)].val_score for f in folds)
        print(f"\nmean val_score: on={on_mean:.4f} off={off_mean:.4f} delta={on_mean - off_mean:+.4f} "
              f"| treatment better on {wins}/{len(folds)} folds")

        for label in ("off", "on"):
            vs = [results[(label, f.name)].val_score for f in folds]
            ts = [results[(label, f.name)].test_score for f in folds]

            def agg(field, metric, _l=label):
                return sum(getattr(results[(_l, f.name)], field)[metric] for f in folds) / len(folds)

            row = {
                "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
                "run_name": f"pbp_net_rtg_luckadj_{label}_{args.folds}",
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
                "n_features": results[(label, folds[-1].name)].n_features,
                "injury_enabled": True,
                "rolling_windows": ",".join(str(w) for w in cfg.features.rolling_windows),
                "val_score_mean": round(sum(vs) / len(vs), 4),
                "val_score_per_fold": ",".join(f"{results[(label, f.name)].val_score:.4f}" for f in folds),
                "test_score_mean": round(sum(ts) / len(ts), 4),
                "protocol": "cv",
                "session_id": args.session_id,
                "notes": (
                    f"pbp.enabled={str(label == 'on').lower()} over "
                    f"{'all 5 folds' if args.folds == 'all' else 'folds 3-5 (cheap screen)'}. "
                    "Treatment adds a garbage-time-filtered net rating with three-point and "
                    "free-throw variance partly removed, rolled over each team's previous 10 games. "
                    "Compare on vs off rows for the delta."
                ),
            }
            session_csv.parent.mkdir(parents=True, exist_ok=True)
            write_header = not session_csv.exists()
            with open(session_csv, "a", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=row.keys())
                if write_header:
                    w.writeheader()
                w.writerow(row)
        print(f"\nLogged both rows to {session_csv}")

    finally:
        set_pbp_enabled(COMMITTED_DEFAULT)
        assert load_config().pbp.enabled is COMMITTED_DEFAULT, "FAILED TO RESTORE"
        print(f"config restored: pbp.enabled={COMMITTED_DEFAULT}")


if __name__ == "__main__":
    main()
