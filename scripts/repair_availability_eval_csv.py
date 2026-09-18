"""
One-off repair of outputs/availability_eval.csv.

Early runs appended rows with different column sets (the `sample` and
`n_failed` fields were added after the first run), which left the append-only
log ragged and unreadable by pandas. This rewrites every row into the single
column order the eval script now writes, filling the fields older rows never
carried. Values are never altered, only reordered and padded.

Usage: venv/bin/python3 scripts/repair_availability_eval_csv.py [--dry-run]
"""

import argparse
import csv
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

CANONICAL = [
    "tag",
    "run_at",
    "sample",
    "estimator",
    "slice",
    "n",
    "play_rate",
    "brier",
    "log_loss",
    "auc",
    "ece",
    "n_failed",
]

# The two historical row shapes, in the order they were written.
SHAPES = {
    10: ["tag", "run_at", "estimator", "slice", "n", "play_rate", "brier", "log_loss", "auc", "ece"],
    12: [
        "tag",
        "run_at",
        "sample",
        "estimator",
        "slice",
        "n_failed",
        "n",
        "play_rate",
        "brier",
        "log_loss",
        "auc",
        "ece",
    ],
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", default="outputs/availability_eval.csv")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    path = Path(args.path)
    rows = list(csv.reader(path.open()))
    header, data = rows[0], rows[1:]
    if header == CANONICAL:
        print("already canonical, nothing to do")
        return

    out, counts = [], {}
    for r in data:
        shape = SHAPES.get(len(r))
        if shape is None:
            raise SystemExit(f"unrecognised row width {len(r)}: {r[:4]}")
        counts[len(r)] = counts.get(len(r), 0) + 1
        d = dict(zip(shape, r))
        out.append([d.get(c, "") for c in CANONICAL])

    print(f"{len(out)} rows; widths seen: {counts}")
    if args.dry_run:
        return
    shutil.copy(path, path.with_suffix(".csv.bak"))
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(CANONICAL)
        w.writerows(out)
    print(f"rewrote {path} ({len(out)} rows); original kept at {path.with_suffix('.csv.bak')}")


if __name__ == "__main__":
    main()
