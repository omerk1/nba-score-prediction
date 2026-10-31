"""Ablation: injury_features.importance_prior_stats_fallback off vs on.

Both arms read injury features rescored by scripts/rescore_injury_features.py
(--fallback off / on), so the only difference is the fallback. Nothing in
configs/config.yaml or the live injury table is modified.

Usage:
    venv/bin/python3 scripts/run_injury_fallback_ablation.py --session-id <id> \
        --off-db <path> --on-db <path> [--folds last3|all]
"""

import argparse
import csv
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.append(str(REPO))
os.chdir(REPO)

import src.feature_engineering.feature_builder as fb  # noqa: E402
from src.evaluation.cv_harness import run_split, validate_fold_definitions  # noqa: E402
from src.utils.config_loader import load_config  # noqa: E402

_orig_load_config = fb.load_config


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--session-id", required=True)
    ap.add_argument("--off-db", required=True)
    ap.add_argument("--on-db", required=True)
    ap.add_argument("--folds", choices=["last3", "all"], default="last3")
    args = ap.parse_args()

    cfg = load_config()
    folds = cfg.cv.folds
    validate_fold_definitions(folds)
    if args.folds == "last3":
        folds = folds[-3:]

    results = {}
    for label, db in [("off", args.off_db), ("on", args.on_db)]:
        def patched(*a, _db=db, **k):
            c = _orig_load_config(*a, **k)
            c.injury_features.features_db_path = _db
            return c

        fb.load_config = patched
        print(f"\n### fallback={label} ({db}) ###", flush=True)
        for f in folds:
            t0 = time.time()
            r = run_split(cfg, f.train_end_date, f.validation_start_date, f.validation_end_date,
                          f.test_start_date, f.test_end_date)
            results[(label, f.name)] = r
            print(f"  {f.name}: val_score={r.val_score:.4f} diff_mae={r.val_metrics['diff_mae']:.3f} "
                  f"total_mae={r.val_metrics['total_mae']:.3f} test_score={r.test_score:.4f} "
                  f"({time.time() - t0:.0f}s)", flush=True)
    fb.load_config = _orig_load_config

    print("\n=== per fold: val_score off -> on (negative delta = treatment better) ===")
    deltas = []
    for f in folds:
        a, b = results[("off", f.name)].val_score, results[("on", f.name)].val_score
        deltas.append(b - a)
        print(f"  {f.name}: {a:.4f} -> {b:.4f}  delta={b - a:+.4f}")
    print(f"  mean delta={sum(deltas) / len(deltas):+.4f}, treatment better on "
          f"{sum(d < 0 for d in deltas)}/{len(deltas)} folds")

    session_csv = REPO / "results" / "sessions" / f"{args.session_id}.csv"
    session_csv.parent.mkdir(parents=True, exist_ok=True)
    for label in ("off", "on"):
        vs = [results[(label, f.name)].val_score for f in folds]
        ts = [results[(label, f.name)].test_score for f in folds]
        row = {
            "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
            "run_name": f"injury_prior_stats_fallback_{label}_{args.folds}",
            "val_score_mean": round(sum(vs) / len(vs), 4),
            "val_score_per_fold": ",".join(f"{v:.4f}" for v in vs),
            "test_score_mean": round(sum(ts) / len(ts), 4),
            "protocol": "cv",
            "session_id": args.session_id,
            "notes": f"importance_prior_stats_fallback={label}; folds={args.folds}; both arms rescored from player_injuries",
        }
        write_header = not session_csv.exists()
        with open(session_csv, "a", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=row.keys())
            if write_header:
                w.writeheader()
            w.writerow(row)
    print(f"\nLogged to {session_csv}")


if __name__ == "__main__":
    main()
