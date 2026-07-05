"""
Item #1 (real hyperparameter search) + the in-memory pipeline both item #1 and
items #2/#3 build on.

IMPORTANT (per coordinator clarification): the established baseline this whole
run compares against (corr_a7_alone = 0.281, Phase 3/4 in the phase log) uses
LAYER=2 (injury-adjusted) matchup vectors -- `similarity.run_similarity_search`
defaults to layer=2 for that reason. Every "does this beat the baseline" number
produced in this module is therefore computed on injury-adjusted vectors. Layer
1 (no injury adjustment) is only used if explicitly requested (e.g. to isolate
injury-adjustment's own marginal contribution under a new hyperparameter set),
and any such row is labeled `layer=1` explicitly so it is never confused with a
fair layer=2-vs-layer=2 comparison.

Why an in-memory pipeline instead of reusing fingerprint.build_fingerprint_cache
/ injury_layer.build_injury_adjusted_fingerprints / matchup_index.build_matchup_index
directly: those functions persist every call into `outputs/a7_matchups_cache.sqlite`'s
`matchup_fingerprints` table (DELETE + INSERT ~25k/50k rows). Hyperparameter search
calls this pipeline dozens of times with different fingerprint_window/decay_halflife
values -- writing the shared cache table on every trial would (a) be slow churn,
and (b) leave the cache in a state inconsistent with the phase-log-documented
defaults (fingerprint_window=20, decay_halflife=5) other code/humans expect to find
there. So this module recomputes the window/halflife-dependent parts of the
pipeline purely in memory and never touches matchup_fingerprints. Everything that
does NOT depend on the searched hyperparameters (box score loading, injury-Out-
player resolution + severity classification, the A2 H2H baseline, the raw game
list) is computed ONCE up front and reused across all trials -- this is the same
data, same leakage-safe join logic as fingerprint.py/injury_layer.py/similarity.py,
just restructured so the expensive constant parts aren't recomputed 40+ times.

Leakage discipline: identical pattern to similarity.py -- for a target game at
row i (rows sorted ascending by game_date), the search corpus is every row with
game_date STRICTLY BEFORE the target's date, via `np.searchsorted(dates, target_date,
side="left")` on the date-sorted vector array. This excludes every other game on
the same date (not just earlier row positions) exactly like the reference
implementation. Nothing in this module weakens that guard.
"""

import logging

import numpy as np
import pandas as pd

from src.matchups.baseline_a2 import build_a2_h2h_scores
from src.matchups.config import load_style_matchup_config
from src.matchups.fingerprint import FINGERPRINT_METRICS, _decayed_weighted_mean, _load_raw_team_game_metrics
from src.matchups.injury_layer import _out_players_with_reason, _team_game_archetype_severity
from src.matchups.matchup_index import _load_games
from src.matchups.split import get_split_dates

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constant precomputation (independent of the searched hyperparameters)
# ---------------------------------------------------------------------------

def load_constants() -> dict:
    """Everything the search loop needs that does NOT vary per-trial."""
    raw = _load_raw_team_game_metrics()
    games = _load_games()
    h2h = build_a2_h2h_scores()[["game_id", "h2h_score"]]

    out_df = _out_players_with_reason()
    severity_map = _team_game_archetype_severity(out_df)
    cfg = load_style_matchup_config()
    delta_lookup = _flatten_injury_deltas(severity_map, cfg["injury_impact"])

    return {
        "raw": raw,
        "games": games,
        "h2h": h2h,
        "delta_lookup": delta_lookup,
    }


def _flatten_injury_deltas(severity_map: dict, injury_impact: dict) -> pd.DataFrame:
    """Precompute the TOTAL per-metric additive delta for every (game_date, team_id)
    that had at least one archetype-classified Out player -- this is independent of
    fingerprint_window/decay_halflife (only the base fingerprint values it gets added
    to change), so it is computed once and merged onto any fp1 in apply_injury_deltas().

    Reproduces injury_layer.py's exact stacking rule: multiple archetypes missing
    simultaneously stack additively; within one archetype, severity_map already holds
    the MAX severity multiplier (injury_layer._team_game_archetype_severity), so no
    double-counting of same-archetype multiple absences.
    """
    rows = []
    for (gd, tid), archs in severity_map.items():
        deltas = {m: 0.0 for m in FINGERPRINT_METRICS}
        for archetype, sev in archs.items():
            for metric, delta in injury_impact.get(archetype, {}).items():
                if metric in FINGERPRINT_METRICS:
                    deltas[metric] += delta * sev
        row = {"game_date_d": gd, "team_id": tid}
        row.update({f"delta_{m}": deltas[m] for m in FINGERPRINT_METRICS})
        rows.append(row)
    if not rows:
        cols = ["game_date_d", "team_id"] + [f"delta_{m}" for m in FINGERPRINT_METRICS]
        return pd.DataFrame(columns=cols)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Per-trial (hyperparameter-dependent) pipeline
# ---------------------------------------------------------------------------

def build_fingerprints_inmemory(raw: pd.DataFrame, window: int, halflife: float, min_games: int = 5) -> pd.DataFrame:
    """Same math as fingerprint.compute_rolling_fingerprints, parameterized, operating
    on an already-loaded `raw` frame (avoids re-querying box_score_stats per trial)."""
    out_frames = []
    for team_id, g in raw.groupby("team_id", sort=False):
        g = g.sort_values("game_date").reset_index(drop=True)
        result = {"game_id": g["game_id"], "team_id": team_id, "game_date": g["game_date"]}
        n_games = g[FINGERPRINT_METRICS[0]].shift(1).rolling(window, min_periods=1).count()
        for metric in FINGERPRINT_METRICS:
            shifted = g[metric].shift(1)
            result[metric] = shifted.rolling(window, min_periods=1).apply(
                lambda x: _decayed_weighted_mean(x, halflife), raw=True
            )
        result["n_games_in_window"] = n_games
        out_frames.append(pd.DataFrame(result))
    fp = pd.concat(out_frames, ignore_index=True)
    return fp[fp["n_games_in_window"] >= min_games].copy()


def apply_injury_deltas(fp1: pd.DataFrame, delta_lookup: pd.DataFrame) -> pd.DataFrame:
    """Layer 2: additive injury deltas, vectorized merge (see _flatten_injury_deltas)."""
    fp2 = fp1.copy()
    fp2["game_date_d"] = fp2["game_date"].str[:10]
    fp2 = fp2.merge(delta_lookup, on=["game_date_d", "team_id"], how="left")
    for m in FINGERPRINT_METRICS:
        fp2[m] = fp2[m] + fp2[f"delta_{m}"].fillna(0.0)
    return fp2.drop(columns=["game_date_d"] + [f"delta_{m}" for m in FINGERPRINT_METRICS])


def build_index_inmemory(fp: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    """Same z-score + concat logic as matchup_index.build_matchup_index, operating on
    an in-memory fingerprint frame instead of reading matchup_fingerprints by layer."""
    stats = {}
    for m in FINGERPRINT_METRICS:
        mu, sd = fp[m].mean(), fp[m].std()
        stats[m] = (mu, sd if sd > 1e-9 else 1.0)

    fp_norm = fp.copy()
    for m in FINGERPRINT_METRICS:
        mu, sd = stats[m]
        fp_norm[m] = (fp_norm[m] - mu) / sd

    home = fp_norm.rename(columns={m: f"home_{m}" for m in FINGERPRINT_METRICS})
    away = fp_norm.rename(columns={m: f"away_{m}" for m in FINGERPRINT_METRICS})

    merged = games.merge(
        home[["game_id", "team_id"] + [f"home_{m}" for m in FINGERPRINT_METRICS]],
        left_on=["game_id", "team_id_home"], right_on=["game_id", "team_id"], how="inner",
    ).drop(columns=["team_id"])
    merged = merged.merge(
        away[["game_id", "team_id"] + [f"away_{m}" for m in FINGERPRINT_METRICS]],
        left_on=["game_id", "team_id_away"], right_on=["game_id", "team_id"], how="inner",
    ).drop(columns=["team_id"])

    vector_cols = [f"home_{m}" for m in FINGERPRINT_METRICS] + [f"away_{m}" for m in FINGERPRINT_METRICS]
    return merged[
        ["game_id", "game_date", "team_id_home", "team_id_away", "actual_home_margin"] + vector_cols
    ].sort_values("game_date").reset_index(drop=True)


def run_search_inmemory(
    idx: pd.DataFrame, h2h: pd.DataFrame, method: str, threshold: float, k: int,
    min_confidence_sample: int, full_confidence_sample: int,
    eval_start: str, eval_end: str,
    floor: float | None = None, recency_years: float | None = None,
) -> pd.DataFrame:
    """Identical leakage-guard/search logic to similarity.run_similarity_search,
    operating on an in-memory idx built by build_index_inmemory (no DB read).

    Extended this run (walk-forward CV) with two new, backward-compatible axes:
      method="knn_floor" -- the hybrid method (item #2): up to `k` nearest neighbors
        by cosine similarity, but ONLY those that also clear `floor` (a minimum
        cosine similarity). If fewer than `k` games clear the floor, fewer are used
        (never padded with dissimilar games to force the count) -- this is what lets
        n_similar shrink naturally in thin/mismatched periods, which flows straight
        into the existing confidence/fallback mechanism unchanged.
      recency_years -- (item #3) if set, the search corpus for a target game is
        further restricted to prior games within `recency_years` of the target's
        date (in addition to the existing strict "before this date" exclusion).
        None (default) preserves the original unbounded-history behavior.

    Leakage discipline unchanged: dates are converted to datetime64 (from the
    date-sorted idx) and both the upper bound (strictly before target date, via
    np.searchsorted side="left") and the new lower bound (recency cutoff, same
    technique) are computed via np.searchsorted on this sorted array -- same
    same-date-exclusion guarantee as the original string-based version (datetime64
    ordering matches ISO-date-string ordering exactly for same-format dates).
    """
    vector_cols = [c for c in idx.columns if c.startswith("home_") or c.startswith("away_")]
    V = idx[vector_cols].to_numpy(dtype=np.float64)
    norms = np.linalg.norm(V, axis=1)
    norms[norms == 0] = 1e-9
    dates = pd.to_datetime(idx["game_date"]).to_numpy()
    margins = idx["actual_home_margin"].to_numpy()

    idx = idx.merge(h2h, on="game_id", how="left")
    idx["h2h_score"] = idx["h2h_score"].fillna(0.0)

    eval_mask = (idx["game_date"] >= eval_start) & (idx["game_date"] <= eval_end)
    eval_positions = np.where(eval_mask.to_numpy())[0]

    results = []
    for i in eval_positions:
        target_date = dates[i]
        end_pos = int(np.searchsorted(dates, target_date, side="left"))
        start_pos = 0
        if recency_years is not None:
            cutoff = target_date - np.timedelta64(int(round(recency_years * 365.25)), "D")
            start_pos = int(np.searchsorted(dates, cutoff, side="left"))
        if end_pos <= start_pos:
            n_similar = 0
            style_score = None
        else:
            hist_V = V[start_pos:end_pos]
            hist_norms = norms[start_pos:end_pos]
            hist_margins = margins[start_pos:end_pos]
            sims = (hist_V @ V[i]) / (hist_norms * norms[i])
            if method == "cosine":
                keep = np.where(sims >= threshold)[0]
            elif method == "knn":
                n_take = min(k, len(sims))
                keep = np.argpartition(-sims, n_take - 1)[:n_take] if n_take > 0 else np.array([], dtype=int)
            elif method == "knn_floor":
                floor_idx = np.where(sims >= floor)[0]
                if len(floor_idx) <= k:
                    keep = floor_idx
                else:
                    sub_sims = sims[floor_idx]
                    n_take = min(k, len(floor_idx))
                    top_local = np.argpartition(-sub_sims, n_take - 1)[:n_take]
                    keep = floor_idx[top_local]
            else:
                raise ValueError(f"Unknown similarity_method: {method}")
            n_similar = len(keep)
            style_score = float(hist_margins[keep].mean()) if n_similar else None

        confidence = min(n_similar / full_confidence_sample, 1.0) if n_similar else 0.0
        fallback_used = n_similar < min_confidence_sample
        final_score = idx.at[i, "h2h_score"] if fallback_used else style_score
        if final_score is None:
            final_score = idx.at[i, "h2h_score"]

        results.append({
            "game_id": idx.at[i, "game_id"],
            "game_date": idx.at[i, "game_date"],
            "actual_home_margin": idx.at[i, "actual_home_margin"],
            "h2h_score": idx.at[i, "h2h_score"],
            "n_similar": n_similar,
            "confidence": confidence,
            "fallback_used": fallback_used,
            "style_score": final_score,
        })
    return pd.DataFrame(results)


def evaluate_config(
    consts: dict, window: int, halflife: float, method: str, threshold: float, k: int,
    min_confidence_sample: int, full_confidence_sample: int,
    eval_start: str, eval_end: str, layer: int = 2,
    floor: float | None = None, recency_years: float | None = None,
) -> dict:
    """Full in-memory pipeline for one hyperparameter configuration. layer=2 (injury-
    adjusted) is the default and the one comparable to the established 0.281 baseline;
    layer=1 is available for explicit ablation only (see module docstring).

    `floor` (method="knn_floor") and `recency_years` (any method) are the two new
    axes added this run -- see run_search_inmemory's docstring."""
    fp1 = build_fingerprints_inmemory(consts["raw"], window=window, halflife=halflife)
    fp = apply_injury_deltas(fp1, consts["delta_lookup"]) if layer == 2 else fp1
    idx = build_index_inmemory(fp, consts["games"])
    out = run_search_inmemory(
        idx, consts["h2h"], method=method, threshold=threshold, k=k,
        min_confidence_sample=min_confidence_sample, full_confidence_sample=full_confidence_sample,
        eval_start=eval_start, eval_end=eval_end, floor=floor, recency_years=recency_years,
    )
    if len(out) == 0 or out["style_score"].std() == 0:
        corr = 0.0
    else:
        corr = float(out["style_score"].corr(out["actual_home_margin"]))
    return {
        "corr": corr,
        "n_games": len(out),
        "fallback_rate": float(out["fallback_used"].mean()) if len(out) else 1.0,
        "mean_confidence": float(out["confidence"].mean()) if len(out) else 0.0,
        "df": out,
    }


def build_idx_for_config(consts: dict, window: int, halflife: float, layer: int = 2) -> pd.DataFrame:
    """Build the (window, halflife, layer)-dependent matchup index ONCE, so callers that
    need to evaluate the SAME fingerprint config across many eval windows (e.g. the
    walk-forward folds, item #1 of this run) don't redo the fingerprint/injury/z-score
    work per fold -- only run_search_inmemory (cheap relative to fingerprint building)
    needs to be re-run per fold/eval-window."""
    fp1 = build_fingerprints_inmemory(consts["raw"], window=window, halflife=halflife)
    fp = apply_injury_deltas(fp1, consts["delta_lookup"]) if layer == 2 else fp1
    return build_index_inmemory(fp, consts["games"])


# ---------------------------------------------------------------------------
# Optuna search (item #1)
# ---------------------------------------------------------------------------

def run_optuna_search(n_trials: int = 40, layer: int = 2, seed: int = 42) -> dict:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    consts = load_constants()
    splits = get_split_dates()
    train_start, train_end = splits["train_start"], splits["train_end"]

    def objective(trial: "optuna.Trial") -> float:
        window = trial.suggest_int("fingerprint_window", 10, 40)
        halflife = trial.suggest_float("decay_halflife", 2.0, 15.0)
        method = trial.suggest_categorical("similarity_method", ["cosine", "knn"])
        threshold = trial.suggest_float("similarity_threshold", 0.4, 0.9) if method == "cosine" else 0.7
        k = trial.suggest_int("knn_k", 10, 150) if method == "knn" else 30
        min_conf = trial.suggest_int("min_confidence_sample", 5, 30)
        full_conf = trial.suggest_int("full_confidence_sample", max(min_conf, 30), 150)

        result = evaluate_config(
            consts, window=window, halflife=halflife, method=method,
            threshold=threshold, k=k, min_confidence_sample=min_conf, full_confidence_sample=full_conf,
            eval_start=train_start, eval_end=train_end, layer=layer,
        )
        return result["corr"]

    study = optuna.create_study(
        study_name="a7_wider_exploration_hpsearch",
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=seed),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best_params = dict(study.best_params)
    best_params.setdefault("similarity_threshold", 0.7)
    best_params.setdefault("knn_k", 30)

    # Evaluate the best config on BOTH splits (train = selection, validation = report).
    train_eval = evaluate_config(
        consts, window=best_params["fingerprint_window"], halflife=best_params["decay_halflife"],
        method=best_params["similarity_method"], threshold=best_params["similarity_threshold"],
        k=best_params["knn_k"], min_confidence_sample=best_params["min_confidence_sample"],
        full_confidence_sample=best_params["full_confidence_sample"],
        eval_start=train_start, eval_end=train_end, layer=layer,
    )
    val_eval = evaluate_config(
        consts, window=best_params["fingerprint_window"], halflife=best_params["decay_halflife"],
        method=best_params["similarity_method"], threshold=best_params["similarity_threshold"],
        k=best_params["knn_k"], min_confidence_sample=best_params["min_confidence_sample"],
        full_confidence_sample=best_params["full_confidence_sample"],
        eval_start=splits["validation_start"], eval_end=splits["validation_end"], layer=layer,
    )

    # Also evaluate the PREVIOUS run's hand-picked default (window=20, halflife=5,
    # cosine@0.70) on these exact same splits, for an apples-to-apples comparison
    # (the original 0.281 number was measured on a different, overlapping window).
    default_train = evaluate_config(
        consts, window=20, halflife=5.0, method="cosine", threshold=0.70, k=30,
        min_confidence_sample=10, full_confidence_sample=50,
        eval_start=train_start, eval_end=train_end, layer=layer,
    )
    default_val = evaluate_config(
        consts, window=20, halflife=5.0, method="cosine", threshold=0.70, k=30,
        min_confidence_sample=10, full_confidence_sample=50,
        eval_start=splits["validation_start"], eval_end=splits["validation_end"], layer=layer,
    )

    summary = {
        "best_params": best_params,
        "best_trial_train_corr": study.best_value,
        "train_corr_reeval": train_eval["corr"],
        "validation_corr": val_eval["corr"],
        "train_n_games": train_eval["n_games"],
        "validation_n_games": val_eval["n_games"],
        "default_train_corr": default_train["corr"],
        "default_validation_corr": default_val["corr"],
        "n_trials": n_trials,
        "layer": layer,
    }
    logger.info(f"Optuna search summary: {summary}")
    return summary


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    import json
    print(json.dumps(run_optuna_search(n_trials=40), indent=2, default=str))
