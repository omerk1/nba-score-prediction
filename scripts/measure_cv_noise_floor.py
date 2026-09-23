"""
Measure the CV harness's own run-to-run noise floor.

Every ablation in this project is judged on differences in `val_score` of
order 0.001-0.01, but the size of a difference that means nothing has never
been measured. CatBoost is stochastic (`random_seed`, `subsample`,
`colsample_bylevel`), so re-running an IDENTICAL config under different seeds
gives the spread attributable to nothing at all.

Config is untouched except `model.random_state`, varied in memory via
pydantic `model_copy` — `configs/config.yaml` on disk is never written, unlike
a feature ablation which must toggle a flag there.

Read the output as: any ablation delta inside this spread is unresolvable by
this harness, per fold and in the mean.

Usage:
    venv/bin/python3 scripts/measure_cv_noise_floor.py --seeds 42,7,1234,2026
"""

import argparse
import csv
import os
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.append(str(REPO))
os.chdir(REPO)

from src.evaluation.cv_harness import run_split  # noqa: E402
from src.utils.config_loader import load_config  # noqa: E402

OUT = REPO / "outputs" / "cv_noise_floor.csv"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", default="42,7,1234", help="comma-separated random_state values")
    ap.add_argument("--tag", default="champion_config", help="label for the config being measured")
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]

    base = load_config()
    assert base.pbp is None or not base.pbp.enabled, "measure the committed baseline, not a treatment"
    folds = base.cv.folds
    scores: dict[tuple[int, str], float] = {}

    for seed in seeds:
        cfg = base.model_copy(update={"model": base.model.model_copy(update={"random_state": seed})})
        assert cfg.model.random_state == seed
        print(f"\n### random_state={seed} ###", flush=True)
        for f in folds:
            r = run_split(cfg, f.train_end_date, f.validation_start_date, f.validation_end_date,
                          f.test_start_date, f.test_end_date)
            scores[(seed, f.name)] = r.val_score
            print(f"  {f.name}: val_score={r.val_score:.4f}", flush=True)

    print("\n=== spread across seeds, identical config ===")
    rows = []
    for f in folds:
        vals = [scores[(s, f.name)] for s in seeds]
        spread = max(vals) - min(vals)
        sd = statistics.stdev(vals) if len(vals) > 1 else 0.0
        print(f"  {f.name}: " + "  ".join(f"{v:.4f}" for v in vals) +
              f"   range={spread:.4f}  sd={sd:.4f}")
        rows.append({"fold": f.name, "range": round(spread, 5), "sd": round(sd, 5),
                     "values": ",".join(f"{v:.4f}" for v in vals)})

    means = [statistics.mean(scores[(s, f.name)] for f in folds) for s in seeds]
    mean_range = max(means) - min(means)
    mean_sd = statistics.stdev(means) if len(means) > 1 else 0.0
    print("\n  mean val_score per seed: " + "  ".join(f"{m:.4f}" for m in means))
    print(f"  MEAN-LEVEL range={mean_range:.4f}  sd={mean_sd:.4f}")
    worst_fold = max(r["range"] for r in rows)
    print(f"\nPer-fold noise floor (largest range): {worst_fold:.4f}")
    print(f"Mean-level noise floor (range):        {mean_range:.4f}")
    rows.append({"fold": "MEAN", "range": round(mean_range, 5), "sd": round(mean_sd, 5),
                 "values": ",".join(f"{m:.4f}" for m in means)})

    OUT.parent.mkdir(parents=True, exist_ok=True)
    write_header = not OUT.exists()
    with open(OUT, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["timestamp", "tag", "seeds", "fold", "range", "sd", "values"])
        if write_header:
            w.writeheader()
        for r in rows:
            w.writerow({"timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
                        "tag": args.tag, "seeds": ",".join(map(str, seeds)), **r})
    print(f"\nWritten to {OUT}")


if __name__ == "__main__":
    main()
