"""
Item #2a: PCA encoding (design doc Encoding Phase 2), tried unconditionally this
run (not gated behind "only if hand-picked is weak" -- it wasn't weak, per Phase 4,
but this run's instructions ask for the comparison regardless).

Scope note: the design doc's Phase 2 raw-metric list includes several shot-chart/
play-type-derived metrics (second_chance_rate, fast_break_rate, avg_shot_distance,
pull_up_rate, catch_shoot_rate) that require nba_api shot-chart endpoints -- those
are explicitly OUT OF SCOPE for this run (deferred to docs/backlog.md). This module
therefore uses only the box-score-derivable subset of the design doc's Phase 2 list:
pace/three_pt_rate/paint_rate/ast_rate/reb_rate/to_rate/ft_rate/def_reb_rate/
opp_3pt_allowed/opp_paint_allowed, PLUS the exact 5 hand-picked metrics (needed
verbatim so the existing calibrated injury deltas -- which target those 5 metric
names specifically -- can still be applied before PCA; see apply injury note below).
11 raw metrics total (design doc suggested 15-20; this is what's available without
shot-chart data).

Injury adjustment (per coordinator clarification: PCA's comparison number must be
on injury-adjusted / layer=2 inputs, same as the 0.281 baseline): the 5 calibrated
injury deltas are applied to the matching 5 columns of the 11-raw-metric vector
BEFORE PCA is fit/transformed (reusing tuning.py's apply_injury_deltas /
delta_lookup mechanism -- injury deltas have no defined meaning in PCA-component
space, only in the original metric space, so adjustment must happen pre-PCA). The
other 6 raw metrics are not covered by any calibrated delta and are left
unadjusted -- documented as a known limitation, not silently glossed over.

Leakage discipline for PCA fitting: PCA (StandardScaler + PCA pipeline) is fit
ONLY on team-games whose game_date falls within the TRAIN split
(configs/config.yaml's train_start_date/train_end_date, via split.py) -- never on
validation-range rows, matching the design doc's explicit "fit PCA on training set
only" instruction. `.transform()` is then applied to all rows (train + validation)
using the already-fitted object. Similarity search on top of the resulting 5-dim-
per-team PCA vectors reuses tuning.py's run_search_inmemory verbatim (its leakage
guard -- np.searchsorted excluding same-date games -- is encoding-agnostic).
"""

import logging

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.matchups.baseline_a2 import build_a2_h2h_scores
from src.matchups.fingerprint import FINGERPRINT_METRICS, _decayed_weighted_mean
from src.matchups.matchup_index import _load_games
from src.matchups.split import get_split_dates
from src.matchups.tuning import _flatten_injury_deltas, run_search_inmemory
from src.matchups.injury_layer import _out_players_with_reason, _team_game_archetype_severity
from src.matchups.config import load_style_matchup_config
from src.matchups.db import cache_conn

logger = logging.getLogger(__name__)

EXTRA_RAW_METRICS = [
    "reb_rate", "to_rate", "ft_rate", "def_reb_rate",
    "opp_3pt_rate_allowed", "opp_paint_rate_allowed",
]
RAW_METRICS = FINGERPRINT_METRICS + EXTRA_RAW_METRICS  # 11 total
N_PCA_COMPONENTS = 5  # matches hand-picked dimensionality for an apples-to-apples 10-dim matchup vector


def _load_raw_richer_metrics() -> pd.DataFrame:
    """One row per (game_id, team_id) with 11 raw per-game metrics (5 hand-picked +
    6 additional box-score-derivable ones from the design doc's Phase 2 list)."""
    conn = cache_conn()
    box = pd.read_sql_query("SELECT * FROM box_score_stats", conn)
    conn.close()

    opp_cols = ["pts", "fga", "fg3a", "fta", "oreb", "dreb"]
    opp = box[["game_id", "team_id"] + opp_cols].rename(
        columns={"team_id": "opp_team_id", **{c: f"opp_{c}" for c in opp_cols}}
    )
    merged = box.merge(opp, on="game_id")
    merged = merged[merged["team_id"] != merged["opp_team_id"]].copy()
    merged = merged.drop_duplicates(subset=["game_id", "team_id"])

    possessions = merged["fga"] - merged["oreb"] + merged["tov"] + 0.44 * merged["fta"]
    possessions = possessions.replace(0, np.nan)

    # --- exact hand-picked 5 (verbatim from fingerprint.py, needed for injury-delta compatibility) ---
    merged["pace_score"] = merged["pts"] + merged["opp_pts"] + merged["tov"] - 0.44 * merged["fta"]
    merged["three_pt_reliance"] = (merged["fg3a"] / merged["fga"].replace(0, np.nan)).fillna(0.0)
    merged["paint_activity"] = merged["fta"]
    merged["defensive_rating"] = (merged["opp_pts"] / possessions * 100).fillna(0.0)
    merged["assist_rate"] = (merged["ast"] / merged["fgm"].replace(0, np.nan)).fillna(0.0)

    # --- 6 additional raw metrics (design doc Phase 2 list, box-score-derivable subset) ---
    merged["reb_rate"] = (
        (merged["oreb"] + merged["dreb"]) /
        (merged["oreb"] + merged["dreb"] + merged["opp_oreb"] + merged["opp_dreb"]).replace(0, np.nan)
    ).fillna(0.5)
    merged["to_rate"] = (merged["tov"] / possessions).fillna(0.0)
    merged["ft_rate"] = (merged["fta"] / merged["fga"].replace(0, np.nan)).fillna(0.0)
    merged["def_reb_rate"] = (
        merged["dreb"] / (merged["dreb"] + merged["opp_oreb"]).replace(0, np.nan)
    ).fillna(0.5)
    merged["opp_3pt_rate_allowed"] = (merged["opp_fg3a"] / merged["opp_fga"].replace(0, np.nan)).fillna(0.0)
    merged["opp_paint_rate_allowed"] = (merged["opp_fta"] / merged["opp_fga"].replace(0, np.nan)).fillna(0.0)

    return merged[["game_id", "team_id", "game_date"] + RAW_METRICS].sort_values(["team_id", "game_date"])


def _rolling_decay(raw: pd.DataFrame, metrics: list[str], window: int, halflife: float) -> pd.DataFrame:
    """Generic version of fingerprint.compute_rolling_fingerprints, parameterized by
    an arbitrary metric list (fingerprint.py hardcodes FINGERPRINT_METRICS)."""
    out_frames = []
    for team_id, g in raw.groupby("team_id", sort=False):
        g = g.sort_values("game_date").reset_index(drop=True)
        result = {"game_id": g["game_id"], "team_id": team_id, "game_date": g["game_date"]}
        n_games = g[metrics[0]].shift(1).rolling(window, min_periods=1).count()
        for metric in metrics:
            shifted = g[metric].shift(1)
            result[metric] = shifted.rolling(window, min_periods=1).apply(
                lambda x: _decayed_weighted_mean(x, halflife), raw=True
            )
        result["n_games_in_window"] = n_games
        out_frames.append(pd.DataFrame(result))
    fp = pd.concat(out_frames, ignore_index=True)
    return fp[fp["n_games_in_window"] >= 5].copy()


def build_pca_matchup_index(window: int = 20, halflife: float = 5.0, layer: int = 2) -> dict:
    """Full PCA-encoding pipeline: raw richer metrics -> rolling decay -> injury
    adjustment (layer=2, on the 5 hand-picked columns only) -> PCA fit on TRAIN split
    -> transform all -> z-score -> 10-dim (5 home + 5 away) matchup index, same shape
    as the hand-picked matchup_index.build_matchup_index output."""
    raw = _load_raw_richer_metrics()
    fp = _rolling_decay(raw, RAW_METRICS, window=window, halflife=halflife)

    if layer == 2:
        out_df = _out_players_with_reason()
        severity_map = _team_game_archetype_severity(out_df)
        cfg = load_style_matchup_config()
        delta_lookup = _flatten_injury_deltas(severity_map, cfg["injury_impact"])
        fp["game_date_d"] = fp["game_date"].str[:10]
        fp = fp.merge(delta_lookup, on=["game_date_d", "team_id"], how="left")
        for m in FINGERPRINT_METRICS:  # only the 5 hand-picked columns have a calibrated delta
            fp[m] = fp[m] + fp[f"delta_{m}"].fillna(0.0)
        fp = fp.drop(columns=["game_date_d"] + [f"delta_{m}" for m in FINGERPRINT_METRICS])

    splits = get_split_dates()
    train_mask = (fp["game_date"] >= splits["train_start"]) & (fp["game_date"] <= splits["train_end"])

    pipe = Pipeline([("scaler", StandardScaler()), ("pca", PCA(n_components=N_PCA_COMPONENTS, random_state=42))])
    pipe.fit(fp.loc[train_mask, RAW_METRICS].to_numpy())
    pca_vals = pipe.transform(fp[RAW_METRICS].to_numpy())
    explained_var = pipe.named_steps["pca"].explained_variance_ratio_

    pc_cols = [f"pc{i}" for i in range(N_PCA_COMPONENTS)]
    for i, c in enumerate(pc_cols):
        fp[c] = pca_vals[:, i]

    # z-score each PCA dim across full history before concatenating (same convention as hand-picked)
    for c in pc_cols:
        mu, sd = fp[c].mean(), fp[c].std()
        fp[c] = (fp[c] - mu) / (sd if sd > 1e-9 else 1.0)

    games = _load_games()
    home = fp.rename(columns={c: f"home_{c}" for c in pc_cols})
    away = fp.rename(columns={c: f"away_{c}" for c in pc_cols})
    merged = games.merge(
        home[["game_id", "team_id"] + [f"home_{c}" for c in pc_cols]],
        left_on=["game_id", "team_id_home"], right_on=["game_id", "team_id"], how="inner",
    ).drop(columns=["team_id"])
    merged = merged.merge(
        away[["game_id", "team_id"] + [f"away_{c}" for c in pc_cols]],
        left_on=["game_id", "team_id_away"], right_on=["game_id", "team_id"], how="inner",
    ).drop(columns=["team_id"])
    vector_cols = [f"home_{c}" for c in pc_cols] + [f"away_{c}" for c in pc_cols]
    idx = merged[
        ["game_id", "game_date", "team_id_home", "team_id_away", "actual_home_margin"] + vector_cols
    ].sort_values("game_date").reset_index(drop=True)

    return {"idx": idx, "explained_variance_ratio": explained_var.tolist()}


def evaluate_pca_encoding(
    window: int = 20, halflife: float = 5.0, layer: int = 2,
    method: str = "cosine", threshold: float = 0.70, k: int = 30,
    min_confidence_sample: int = 10, full_confidence_sample: int = 50,
) -> dict:
    built = build_pca_matchup_index(window=window, halflife=halflife, layer=layer)
    idx = built["idx"]
    h2h = build_a2_h2h_scores()[["game_id", "h2h_score"]]
    splits = get_split_dates()

    out = {}
    for split_name, (s, e) in {
        "train": (splits["train_start"], splits["train_end"]),
        "validation": (splits["validation_start"], splits["validation_end"]),
    }.items():
        df = run_search_inmemory(
            idx, h2h, method=method, threshold=threshold, k=k,
            min_confidence_sample=min_confidence_sample, full_confidence_sample=full_confidence_sample,
            eval_start=s, eval_end=e,
        )
        corr = float(df["style_score"].corr(df["actual_home_margin"])) if len(df) else 0.0
        out[split_name] = {
            "corr": corr, "n_games": len(df),
            "fallback_rate": float(df["fallback_used"].mean()) if len(df) else 1.0,
        }
    out["explained_variance_ratio"] = built["explained_variance_ratio"]
    return out


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    import json
    print(json.dumps(evaluate_pca_encoding(), indent=2, default=str))
