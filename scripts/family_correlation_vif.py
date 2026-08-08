"""E3: Refine top families -- collinearity check (session rs_20260808_1).

Diagnostic only -- computes correlation + VIF (variance inflation factor)
among the engineered columns of `style_fingerprint_features`, `style_features`,
and `opponent_quality_features` (the 3 families EXPERIMENTS.md section 2/3.2
flags as candidates for real overlap: two broad "team style/quality" proxies
plus the "used but not predictive" family). Does NOT execute any
consolidation -- per the execution-scope guardrail, any actual family
drop/merge decision is logged as "proposed, pending review", not acted on
here, regardless of what this script finds.

Reuses scripts/run_family_importance.py's `build_col_to_family` (same
structural feature->family mapping, not reimplemented) to identify which
columns belong to each of the 3 families. Feature values come from fold5's
training features (largest, most representative window) via
cv_harness.run_split(keep_artifacts=True) -- retrains a model as a byproduct
of getting the exact real feature matrix (same columns, same point-in-time
construction the champion actually uses), but the model itself isn't used for
anything here.

VIF computed via the standard correlation-matrix-inverse shortcut (no
statsmodels dependency, which isn't installed in this project's venv):
for z-scored columns, VIF_i = diag(inv(correlation_matrix))_i. Equivalent to
regressing each column on all the others and taking 1/(1-R^2), just without
needing a regression library.

Usage: venv/bin/python3 scripts/family_correlation_vif.py --tag <label>

Read-only: touches no config. Writes outputs/family_vif_<tag>.csv (per-column
VIF) for the record; console output is the correlation summary.
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

TARGET_FAMILIES = [
    "_add_style_fingerprint_features",
    "_add_style_features",
    "_add_opponent_quality_features",
]

VIF_FLAG_THRESHOLD = 10.0  # conventional rule-of-thumb cutoff for "problematic" collinearity


def compute_vif(df: pd.DataFrame) -> pd.Series:
    """VIF per column via the correlation-matrix-inverse shortcut."""
    z = (df - df.mean()) / df.std(ddof=0)
    z = z.loc[:, z.std(ddof=0) > 1e-12]  # drop constant columns (would make corr singular)
    corr = np.corrcoef(z.to_numpy(), rowvar=False)
    inv = np.linalg.pinv(corr)  # pinv: robust to near-singular correlation matrices
    return pd.Series(np.diag(inv), index=z.columns)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", required=True, help="Run tag for the output filename, e.g. a session_id")
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
    print(f"  train_features: {len(train_features):,} rows", flush=True)

    family_cols = {}
    for fam in TARGET_FAMILIES:
        cols = [c for c, f in col_to_family.items() if f == fam and c in train_features.columns]
        family_cols[fam] = cols
        print(f"  {fam}: {len(cols)} columns", flush=True)

    all_cols = [c for cols in family_cols.values() for c in cols]
    X = train_features[all_cols].apply(pd.to_numeric, errors="coerce").dropna(how="any")
    print(f"\nRows usable for VIF (no NaN across all {len(all_cols)} columns): {len(X):,}", flush=True)

    vif = compute_vif(X).sort_values(ascending=False)
    vif_df = vif.rename("vif").to_frame()
    vif_df["family"] = vif_df.index.map(col_to_family)
    out_path = REPO / "outputs" / f"family_vif_{args.tag}.csv"
    vif_df.to_csv(out_path)
    print(f"\nSaved per-column VIF to {out_path}")

    print(f"\n=== VIF by column (flag threshold={VIF_FLAG_THRESHOLD}) ===")
    print(vif_df.to_string(float_format=lambda v: f"{v:.2f}"))

    flagged = vif_df[vif_df["vif"] > VIF_FLAG_THRESHOLD]
    print(f"\n{len(flagged)}/{len(vif_df)} columns exceed VIF={VIF_FLAG_THRESHOLD}")

    print("\n=== Mean VIF by family ===")
    print(
        vif_df.groupby("family")["vif"]
        .mean()
        .sort_values(ascending=False)
        .to_string(float_format=lambda v: f"{v:.2f}")
    )

    print("\n=== Cross-family correlation (mean |r| between column pairs from DIFFERENT families) ===")
    full_corr = X.corr()
    for i, fam_a in enumerate(TARGET_FAMILIES):
        for fam_b in TARGET_FAMILIES[i + 1 :]:
            cols_a, cols_b = family_cols[fam_a], family_cols[fam_b]
            if not cols_a or not cols_b:
                continue
            sub = full_corr.loc[cols_a, cols_b].to_numpy()
            print(f"  {fam_a} vs {fam_b}: mean|r|={np.abs(sub).mean():.3f} max|r|={np.abs(sub).max():.3f}")

    print(
        "\nDiagnostic only -- no consolidation executed. Any drop/merge action based on these numbers "
        "is logged in EXPERIMENTS.md as 'proposed, pending review', per the execution-scope guardrail."
    )


if __name__ == "__main__":
    main()
