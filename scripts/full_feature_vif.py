"""A4: VIF trim on the post-Track-B feature set.

Diagnostic-first: computes VIF across the FULL live feature set (all
`_get_feature_columns` columns, currently 148 post-elo_momentum), not just
the 3-family scope E3 covered (`scripts/family_correlation_vif.py`). Reuses
the same correlation-matrix-inverse VIF shortcut (no statsmodels dependency)
and the same fold5-train-features methodology as E3/market_benchmark.py.

Read-only: writes outputs/full_feature_vif_<tag>.csv and prints a per-family
summary. Does not modify config or feature code.

Usage: venv/bin/python3 scripts/full_feature_vif.py --tag <label>
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.append(str(REPO))
os.chdir(REPO)

from scripts.run_family_importance import build_col_to_family  # noqa: E402
from src.evaluation.cv_harness import run_split  # noqa: E402
from src.utils.config_loader import load_config  # noqa: E402

VIF_FLAG_THRESHOLD = 10.0


def compute_vif(df: pd.DataFrame) -> pd.Series:
    z = (df - df.mean()) / df.std(ddof=0)
    z = z.loc[:, z.std(ddof=0) > 1e-12]
    corr = np.corrcoef(z.to_numpy(), rowvar=False)
    inv = np.linalg.pinv(corr)
    return pd.Series(np.diag(inv), index=z.columns)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", required=True)
    args = parser.parse_args()

    cfg = load_config()
    print("Building structural feature->family mapping...", flush=True)
    col_to_family = build_col_to_family(cfg)

    fold5 = cfg.cv.folds[-1]
    print(f"Training fold5 ({fold5.name}, keep_artifacts=True) for the real feature matrix...", flush=True)
    result = run_split(
        cfg,
        fold5.train_end_date,
        fold5.validation_start_date,
        fold5.validation_end_date,
        fold5.test_start_date,
        fold5.test_end_date,
        keep_artifacts=True,
    )
    train_features = result.train_features

    from src.feature_engineering.feature_builder import FeatureBuilder

    fb = FeatureBuilder(rolling_windows=cfg.features.rolling_windows)
    all_cols = fb._get_feature_columns(train_features)
    print(f"Full live feature set: {len(all_cols)} columns", flush=True)

    X = train_features[all_cols].apply(pd.to_numeric, errors="coerce").dropna(how="any")
    print(f"Rows usable for VIF (no NaN across all {len(all_cols)} columns): {len(X):,}", flush=True)

    vif = compute_vif(X).sort_values(ascending=False)
    vif_df = vif.rename("vif").to_frame()
    vif_df["family"] = vif_df.index.map(col_to_family)
    out_path = REPO / "outputs" / f"full_feature_vif_{args.tag}.csv"
    vif_df.to_csv(out_path)
    print(f"\nSaved per-column VIF to {out_path}")

    dropped = [c for c in all_cols if c not in vif_df.index]
    if dropped:
        print(f"\n{len(dropped)} columns dropped as constant/degenerate before VIF: {dropped}")

    flagged = vif_df[vif_df["vif"] > VIF_FLAG_THRESHOLD].sort_values("vif", ascending=False)
    print(f"\n=== {len(flagged)}/{len(vif_df)} columns exceed VIF={VIF_FLAG_THRESHOLD} ===")
    print(flagged.to_string(float_format=lambda v: f"{v:.2f}"))

    print("\n=== Mean VIF by family ===")
    print(
        vif_df.groupby("family")["vif"]
        .mean()
        .sort_values(ascending=False)
        .to_string(float_format=lambda v: f"{v:.2f}")
    )

    # Window-block breakout: for families whose columns are parameterized by
    # L{w}, group flagged columns by window to see whether L10 (the "middle"
    # window) is disproportionately redundant given L5+L20 anchor it, per
    # E3's within-family finding.
    print("\n=== Flagged columns by rolling window (L5/L10/L20) ===")
    for w in cfg.features.rolling_windows:
        suffix = f"L{w}"
        w_flagged = flagged[flagged.index.str.contains(f"_{suffix}(_|$)|{suffix}$", regex=True)]
        print(f"  {suffix}: {len(w_flagged)} flagged columns")

    print("\nDiagnostic only -- no trim executed by this script.")


if __name__ == "__main__":
    main()
