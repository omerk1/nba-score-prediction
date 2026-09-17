"""
Paired comparison of two availability estimators on the same rows.

The intrinsic eval reports each estimator's Brier score separately. With ~1k
rows in the post-cutoff slice, a difference of a few thousandths is not
self-evidently real, so this script does the paired test the decision actually
needs: per-row squared-error differences, their mean, and a bootstrap
confidence interval over the same rows.

Reads the per-row prediction dumps written by run_availability_eval.py
(outputs/availability_llm_details_<tag>_<estimator>.csv) for the LLM
estimators, and recomputes tabular baseline predictions on demand.

Usage:
  venv/bin/python3 scripts/compare_availability_estimators.py \
      --tag phase1_llm --a llm --b catboost --slice post_cutoff
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from src.availability.baselines import all_baselines


def _tabular_predictions(ctx: pd.DataFrame, name: str) -> pd.Series:
    est = next((e for e in all_baselines() if e.name == name), None)
    if est is None:
        raise SystemExit(f"unknown tabular estimator {name}")
    seasons = sorted(ctx["season"].unique())
    pred = pd.Series(np.nan, index=ctx.index)
    for s in seasons[1:]:
        train, test = ctx[ctx["season"] < s], ctx[ctx["season"] == s]
        est.fit(train)
        pred[test.index] = est.predict(test)
    return pred


def _llm_predictions(tag: str, name: str, ctx: pd.DataFrame) -> pd.Series:
    path = Path(f"outputs/availability_llm_details_{tag}_{name}.csv")
    if not path.exists():
        raise SystemExit(f"missing {path} — run the eval with --estimators including {name}")
    d = pd.read_csv(path)
    # The dump carries the identifying columns; join back to the context index.
    key = ["report_date", "player_name", "status"]
    merged = ctx.reset_index().merge(d[key + ["p_play"]], on=key, how="left").set_index("index")
    return merged["p_play"].reindex(ctx.index)


def get_predictions(name: str, tag: str, ctx: pd.DataFrame) -> pd.Series:
    return _llm_predictions(tag, name, ctx) if name.startswith("llm") else _tabular_predictions(ctx, name)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--context-cache", required=True, help="parquet written by run_availability_eval.py")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--a", required=True, help="estimator A (e.g. llm)")
    ap.add_argument("--b", required=True, help="estimator B (e.g. catboost)")
    ap.add_argument("--slice", default="post_cutoff", choices=["pooled", "post_cutoff", "pre_cutoff"])
    ap.add_argument("--cutoff", default="2025-06-01")
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    ctx = pd.read_parquet(args.context_cache)
    pa = get_predictions(args.a, args.tag, ctx)
    pb = get_predictions(args.b, args.tag, ctx)

    seasons = sorted(ctx["season"].unique())
    mask = ctx["season"].isin(seasons[1:])
    if args.slice == "post_cutoff":
        mask &= ctx["report_date"] >= args.cutoff
    elif args.slice == "pre_cutoff":
        mask &= ctx["report_date"] < args.cutoff
    mask &= pa.notna() & pb.notna()

    y = ctx.loc[mask, "played"].to_numpy(dtype=float)
    a, b = pa[mask].to_numpy(dtype=float), pb[mask].to_numpy(dtype=float)
    da, db = (a - y) ** 2, (b - y) ** 2
    diff = da - db  # negative => A better

    rng = np.random.default_rng(args.seed)
    idx = rng.integers(0, len(diff), size=(args.n_boot, len(diff)))
    boot = diff[idx].mean(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])

    print(f"slice={args.slice}  n={len(diff)}  play_rate={y.mean():.3f}")
    print(f"brier({args.a}) = {da.mean():.4f}")
    print(f"brier({args.b}) = {db.mean():.4f}")
    print(f"difference (A - B) = {diff.mean():+.4f}   95% CI [{lo:+.4f}, {hi:+.4f}]")
    print(f"P(A better than B) over bootstrap = {(boot < 0).mean():.3f}")
    print(f"correlation between the two estimators' probabilities = {np.corrcoef(a, b)[0, 1]:.3f}")
    verdict = "A is better" if hi < 0 else ("B is better" if lo > 0 else "no significant difference")
    print(f"verdict: {verdict}")


if __name__ == "__main__":
    main()
