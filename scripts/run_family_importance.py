"""Family-level CatBoost importance + permutation importance under the current
committed config, across all 5 real CV folds (not just one split) -- gives both
an aggregate view (mean across folds) and a fold1-vs-later-folds breakout (does
a family need more history to stabilize, or is it robust on the smallest
training window?).

Usage: venv/bin/python3 scripts/run_family_importance.py --tag <label>

Does NOT touch configs/config.yaml -- read-only wrt config, no restore needed.
Reuses cv_harness.run_split unmodified per fold, same pattern as
scripts/run_style_matchup_cv.py.

Writes outputs/family_importance_inventory_<tag>.csv -- tagged per run (same
convention as outputs/full_feature_importance_<run_name>.csv) so successive
runs coexist in git history instead of silently clobbering each other.
"""

import argparse
import inspect
import os
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.append(str(REPO))
os.chdir(REPO)

from src.evaluation.cv_harness import compute_composite_score, run_split  # noqa: E402
from src.feature_engineering.feature_builder import FeatureBuilder  # noqa: E402
from src.utils.config_loader import load_config  # noqa: E402

FAMILY_STEPS = [
    "_add_basic_features",
    "_add_rolling_features",
    "_add_rest_features",
    "_add_style_features",
    "_add_opponent_quality_features",
    "_add_home_advantage_features",
    "_add_matchup_features",
    "_add_h2h_features",
    "_add_travel_features",
    "_add_elo_features",
    "_add_injury_features",
    "_add_style_matchup_features",
    "_add_style_fingerprint_features",
    "_add_on_off_splits_features",
    "_add_season_motivation_features",
]


def build_col_to_family(cfg) -> dict:
    """Structural mapping (feature name -> family), independent of which fold's
    data is used -- built once from a tiny real sample by calling each
    FeatureBuilder._add_*_features step directly and recording which columns
    it adds."""
    conn = sqlite3.connect(cfg.data_paths.raw_db)
    df = pd.read_sql(
        "SELECT game_id AS GAME_ID, game_date AS GAME_DATE, season_id AS SEASON_ID, "
        "season_type AS SEASON_TYPE, team_id_home AS HOME_TEAM_ID, team_id_away AS AWAY_TEAM_ID, "
        "pts_home AS PTS_home, pts_away AS PTS_away, "
        "CASE WHEN wl_home='W' THEN 1 ELSE 0 END AS HOME_TEAM_WINS, "
        "fg_pct_home AS FG_PCT_home, ft_pct_home AS FT_PCT_home, fg3_pct_home AS FG3_PCT_home, "
        "ast_home AS AST_home, reb_home AS REB_home, "
        "fg_pct_away AS FG_PCT_away, ft_pct_away AS FT_PCT_away, fg3_pct_away AS FG3_PCT_away, "
        "ast_away AS AST_away, reb_away AS REB_away, "
        "fgm_home AS FGM_home, fga_home AS FGA_home, fg3m_home AS FG3M_home, fg3a_home AS FG3A_home, "
        "ftm_home AS FTM_home, fta_home AS FTA_home, "
        "fgm_away AS FGM_away, fga_away AS FGA_away, fg3m_away AS FG3M_away, fg3a_away AS FG3A_away, "
        "ftm_away AS FTM_away, fta_away AS FTA_away "
        "FROM game WHERE game_date <= '2020-08-14' ORDER BY game_date",
        conn,
    )
    conn.close()
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    df["SEASON_ID"] = df["SEASON_ID"].astype(int)
    df["POINT_DIFF"] = df["PTS_home"] - df["PTS_away"]
    df["TOTAL_POINTS"] = df["PTS_home"] + df["PTS_away"]

    fb = FeatureBuilder(
        rolling_windows=cfg.features.rolling_windows,
        h2h_margin_window=cfg.features.h2h_margin_window,
        h2h_win_rate_window=cfg.features.h2h_win_rate_window,
    )
    df = df.sort_values("GAME_DATE").reset_index(drop=True)
    prev_cols = set(df.columns)
    col_to_family = {}
    for step in FAMILY_STEPS:
        fn = getattr(fb, step)
        sig = inspect.signature(fn)
        df = fn(df, "2020-08-14") if "context_end_date" in sig.parameters else fn(df)
        new_cols = [c for c in df.columns if c not in prev_cols]
        for c in new_cols:
            col_to_family[c] = step
        prev_cols = set(df.columns)
    return col_to_family


def permute_family_and_score(predictor, X_val, y_val, family_cols, naive_val_metrics, lambda_weight=0.5):
    """Shuffles a family's columns (as a block, preserving within-family row
    correlations) and returns the resulting composite score -- how much worse
    does the model do once this family's real signal is destroyed?"""
    rng = np.random.default_rng(42)
    X_perm = X_val.copy()
    perm_idx = rng.permutation(len(X_perm))
    present = [c for c in family_cols if c in X_perm.columns]
    if not present:
        return None
    X_perm[present] = X_perm[present].to_numpy()[perm_idx]
    y_pred = predictor.predict(X_perm)
    diff_true = y_val.iloc[:, 0].values - y_val.iloc[:, 1].values
    diff_pred = y_pred[:, 0] - y_pred[:, 1]
    total_true = y_val.iloc[:, 0].values + y_val.iloc[:, 1].values
    total_pred = y_pred[:, 0] + y_pred[:, 1]
    diff_mae = float(np.mean(np.abs(diff_true - diff_pred)))
    total_mae = float(np.mean(np.abs(total_true - total_pred)))
    return compute_composite_score(
        diff_mae, total_mae, naive_val_metrics["diff_mae"], naive_val_metrics["total_mae"], lambda_weight
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", required=True, help="Run tag for the output filename, e.g. a session_id")
    args = parser.parse_args()

    cfg = load_config()
    print("Building structural feature->family mapping...", flush=True)
    col_to_family = build_col_to_family(cfg)
    print(
        f"  mapped {len(col_to_family)} feature columns to {len(set(col_to_family.values()))} families",
        flush=True,
    )

    folds = cfg.cv.folds
    per_fold_rows = []  # (fold_name, family, catboost_importance_share, perm_delta)

    for f in folds:
        print(f"\n=== {f.name}: training (keep_artifacts=True) ===", flush=True)
        result = run_split(
            cfg,
            f.train_end_date,
            f.validation_start_date,
            f.validation_end_date,
            f.test_start_date,
            f.test_end_date,
            keep_artifacts=True,
        )
        print(
            f"    val_score={result.val_score:.4f} (diff_mae={result.val_metrics['diff_mae']:.2f}, "
            f"total_mae={result.val_metrics['total_mae']:.2f})",
            flush=True,
        )

        predictor = result.predictor
        feature_cols = result.feature_cols
        val_features = result.val_features
        X_val = val_features[feature_cols]
        y_val = val_features[cfg.features.targets]

        importance_df = predictor.get_feature_importance(top_n=len(feature_cols))
        importance_df["family"] = importance_df["feature"].map(col_to_family)
        unmapped = importance_df[importance_df["family"].isna()]
        if len(unmapped):
            print(
                f"    WARNING: {len(unmapped)} features unmapped to a family: {list(unmapped['feature'])}",
                flush=True,
            )
        fam_importance = importance_df.groupby("family")["importance"].sum().sort_values(ascending=False)
        total_importance = fam_importance.sum()

        print(f"    CatBoost importance by family ({f.name}):", flush=True)
        for fam, imp in fam_importance.items():
            share = imp / total_importance
            print(f"      {fam:35s} {share:6.1%}", flush=True)

            perm_score = permute_family_and_score(
                predictor,
                X_val,
                y_val,
                [c for c, fa in col_to_family.items() if fa == fam],
                result.naive_val_metrics,
            )
            perm_delta = (perm_score - result.val_score) if perm_score is not None else None
            per_fold_rows.append((f.name, fam, share, perm_delta))

    df = pd.DataFrame(per_fold_rows, columns=["fold", "family", "importance_share", "perm_delta"])
    out_path = REPO / "outputs" / f"family_importance_inventory_{args.tag}.csv"
    df.to_csv(out_path, index=False)
    print(f"\nSaved raw per-fold results to {out_path}")

    print("\n=== AGGREGATE (mean across all 5 folds) ===")
    agg = (
        df.groupby("family")[["importance_share", "perm_delta"]]
        .mean()
        .sort_values("importance_share", ascending=False)
    )
    print(agg.to_string(float_format=lambda x: f"{x:.4f}"))

    print("\n=== FOLD1 vs MEAN(FOLDS 2-5) ===")
    fold1 = df[df["fold"] == "fold1"].set_index("family")["importance_share"]
    later = df[df["fold"] != "fold1"].groupby("family")["importance_share"].mean()
    cmp_df = pd.DataFrame({"fold1_share": fold1, "folds2_5_mean_share": later})
    cmp_df["ratio_fold1_to_later"] = cmp_df["fold1_share"] / cmp_df["folds2_5_mean_share"]
    print(cmp_df.sort_values("ratio_fold1_to_later").to_string(float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    main()
