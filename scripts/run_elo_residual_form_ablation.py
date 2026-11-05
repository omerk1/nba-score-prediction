"""Ablation: elo_residual_form off vs on (per-team mean of actual minus
Elo-expected margin over the last N games).

Toggles the flag through a patched load_config in the feature builder only,
so configs/config.yaml is not modified. Runs the CV harness's run_split per
fold (no model or feature artifacts written) and logs both arms to
results/sessions/<session_id>.csv.

Usage:
    venv/bin/python3 scripts/run_elo_residual_form_ablation.py --session-id <id> \
        [--folds last2|last3|all] [--windows 10]
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
from src.utils.config_loader import EloResidualFormConfig, load_config  # noqa: E402

_orig_load_config = fb.load_config
FOLD_SETS = {"last2": -2, "last3": -3, "all": None}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--session-id", required=True)
    ap.add_argument("--folds", choices=list(FOLD_SETS), default="last2")
    ap.add_argument("--windows", type=int, nargs="+", default=[10])
    ap.add_argument("--diff-only", action="store_true", help="Treatment adds only the home-minus-away column")
    args = ap.parse_args()

    cfg = load_config()
    folds = cfg.cv.folds
    validate_fold_definitions(folds)
    if FOLD_SETS[args.folds] is not None:
        folds = folds[FOLD_SETS[args.folds] :]

    results = {}
    for label, enabled in (("off", False), ("on", True)):

        def patched(*a, _enabled=enabled, **k):
            c = _orig_load_config(*a, **k)
            return c.model_copy(
                update={
                    "elo_residual_form": EloResidualFormConfig(
                        enabled=_enabled, windows=args.windows, diff_only=args.diff_only
                    )
                }
            )

        fb.load_config = patched
        print(f"\n### elo_residual_form={label} windows={args.windows} ###", flush=True)
        for f in folds:
            t0 = time.time()
            r = run_split(
                cfg, f.train_end_date, f.validation_start_date, f.validation_end_date, f.test_start_date, f.test_end_date
            )
            results[(label, f.name)] = r
            print(
                f"  {f.name}: val_score={r.val_score:.4f} diff_mae={r.val_metrics['diff_mae']:.3f} "
                f"total_mae={r.val_metrics['total_mae']:.3f} test_score={r.test_score:.4f} "
                f"features={r.n_features} ({time.time() - t0:.0f}s)",
                flush=True,
            )
    fb.load_config = _orig_load_config

    print("\n=== per fold: val_score off -> on (negative delta = treatment better) ===")
    deltas = []
    for f in folds:
        a, b = results[("off", f.name)].val_score, results[("on", f.name)].val_score
        deltas.append(b - a)
        print(f"  {f.name}: {a:.4f} -> {b:.4f}  delta={b - a:+.4f}")
    print(
        f"  mean delta={sum(deltas) / len(deltas):+.4f}, treatment better on "
        f"{sum(d < 0 for d in deltas)}/{len(deltas)} folds"
    )

    session_csv = REPO / "results" / "sessions" / f"{args.session_id}.csv"
    session_csv.parent.mkdir(parents=True, exist_ok=True)
    win = "_".join(str(w) for w in args.windows)
    for label in ("off", "on"):
        vs = [results[(label, f.name)].val_score for f in folds]
        ts = [results[(label, f.name)].test_score for f in folds]
        row = {
            "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
            "run_name": f"elo_resid_form_{label}"
            + (f"_L{win}" + ("_diffonly" if args.diff_only else "") if label == "on" else "")
            + f"_{args.folds}",
            "val_score_mean": round(sum(vs) / len(vs), 4),
            "val_score_per_fold": ",".join(f"{v:.4f}" for v in vs),
            "test_score_mean": round(sum(ts) / len(ts), 4),
            "protocol": "cv",
            "session_id": args.session_id,
            "notes": f"elo_residual_form.enabled={label == 'on'}; windows={args.windows}; "
            f"diff_only={args.diff_only}; "
            f"folds={','.join(f.name for f in folds)}",
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
