"""Market benchmark -- compare the champion model's genuinely held-out
predictions against Polymarket's pre-game market odds on the same games.

Purpose: not a model change. This answers whether the model's performance
ceiling reflects a weak model, or a fine model whose edge (if any) hasn't
been located yet -- edge lives in DISAGREEMENT with the market, not
agreement. Polymarket data is read here and only here: it is never turned
into a model feature, never passed into feature_builder.py or
predictor.predict, and never feeds back into training or the composite
score. This script does not import or modify cv_harness.py,
score_predictor.py, feature_builder.py, or configs/config.yaml -- it only
calls their existing public functions (run_split, validate_fold_definitions,
load_config) exactly as scripts/family_correlation_vif.py already does.

Held-out predictions are regenerated fresh every run via
run_split(..., keep_artifacts=True) -- this script never accepts an
externally-supplied predictions file, so genuine held-out-ness is
guaranteed by construction rather than trusted by convention. Polymarket
coverage today is 2025-26 season only (games.csv), which is why the
default --fold is the CV harness's latest fold (fold5) -- future folds
gain coverage as more seasons of odds data accumulate. Picking a fold with
no date overlap against games.csv fails loudly (ValueError) before any
join is attempted, rather than silently producing an empty/meaningless
comparison.

Usage: venv/bin/python3 scripts/market_benchmark.py --tag <label>

Outputs:
  outputs/market_benchmark_games_<tag>.csv   -- per-game join, overwritten per run
  outputs/market_benchmark_summary.csv       -- one row appended per run (cross-run
                                                 history, never overwritten/truncated)
"""

import argparse
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from nba_api.stats.static import teams as nba_teams
from scipy.stats import norm

REPO = Path(__file__).resolve().parent.parent
sys.path.append(str(REPO))
os.chdir(REPO)

from src.evaluation.cv_harness import run_split, validate_fold_definitions  # noqa: E402
from src.utils.config_loader import load_config  # noqa: E402

# Rows in games.csv that don't correspond to a real, mappable NBA team pair --
# identified by direct inspection, not a heuristic: one preseason game logged
# with lowercase abbreviations instead of nicknames (also outside every CV
# fold's regular-season test window anyway), and 3 All-Star Weekend exhibition
# rows (not real team matchups).
KNOWN_UNMAPPABLE_TEAMS = {"Team USA Stars", "Team USA Stripes", "Team World", "bkn", "pho"}

MARKET_CLOSENESS_BUCKETS = [(0, 3, "pick_em"), (3, 7, "modest_favorite"), (7, np.inf, "big_favorite")]


def build_nickname_to_team_id() -> dict:
    return {t["nickname"]: t["id"] for t in nba_teams.get_teams()}


def load_polymarket_games(games_csv: Path) -> pd.DataFrame:
    games = pd.read_csv(games_csv)
    nick2id = build_nickname_to_team_id()

    unmappable = games[~games["home_team"].isin(nick2id) | ~games["away_team"].isin(nick2id)]
    if len(unmappable):
        print(
            f"  Dropping {len(unmappable)} games.csv rows with unmappable team names "
            f"(not real NBA franchise nicknames): {sorted(set(unmappable['home_team']) | set(unmappable['away_team']))}"
        )
        unexpected = (set(unmappable["home_team"]) | set(unmappable["away_team"])) - KNOWN_UNMAPPABLE_TEAMS
        if unexpected:
            print(
                f"  WARNING: unmapped team names not in the known set -- new/unexpected data-quality "
                f"issue in games.csv, investigate: {sorted(unexpected)}"
            )
    games = games[games["home_team"].isin(nick2id) & games["away_team"].isin(nick2id)].copy()

    games["home_team_id"] = games["home_team"].map(nick2id)
    games["away_team_id"] = games["away_team"].map(nick2id)

    # games["game_date"] (and the slug it's parsed from) is the UTC calendar
    # date of tipoff; nba_api's GAME_DATE is the US/Eastern "game night" date
    # (the standard NBA-database convention). Any game tipping off after
    # ~8pm ET crosses midnight UTC, landing on the next UTC day -- verified
    # against real mismatched games (e.g. game_start_time_utc
    # "2025-10-27T01:00:00+00:00" = 9pm ET Oct 26 -> nba_api GAME_DATE
    # 2025-10-26, but games["game_date"]="2025-10-27"). Deriving the join
    # date from game_start_time_utc converted to US/Eastern instead of the
    # raw game_date column fixes the join at its root cause. Rows missing
    # game_start_time_utc (already flagged upstream as
    # "gameStartTime_and_startDate_missing") fall back to the raw date,
    # which may still be off by a day for late tipoffs -- rare, and no
    # better date source exists for those rows.
    start_utc = pd.to_datetime(games["game_start_time_utc"], utc=True, errors="coerce")
    game_date_et = start_utc.dt.tz_convert("US/Eastern").dt.normalize().dt.tz_localize(None)
    games["game_date_norm"] = game_date_et.fillna(pd.to_datetime(games["game_date"]).dt.normalize())

    # The ET-derived date correctly identifies a handful of true duplicates
    # in games.csv itself: Polymarket occasionally has TWO separate market
    # listings (different slug/event_id) for the same real game (identical
    # game_start_time_utc, same two teams) -- verified directly, e.g.
    # nba-ind-mem-2025-10-25 ($791K moneyline volume, data_quality="ok") vs
    # nba-ind-mem-2025-10-26 ($2.6K volume, "sparse_trades") for the exact
    # same tipoff. This was invisible under the old raw-date join (the two
    # rows landed on different, wrong dates and never collided) -- fixing
    # the date convention surfaces it. Keep the higher-volume (more liquid,
    # more price-informative) listing per collision, drop the other.
    dup_key = ["home_team_id", "away_team_id", "game_date_norm"]
    n_before = len(games)
    games = games.sort_values("moneyline_volume", ascending=False).drop_duplicates(dup_key, keep="first")
    n_dropped_dupes = n_before - len(games)
    if n_dropped_dupes:
        print(
            f"  Deduplicated {n_dropped_dupes} thin/duplicate Polymarket market listing(s) that share a "
            f"real game with a higher-volume listing (kept the higher moneyline_volume row each time)."
        )
    games = games.sort_values("game_date_norm").reset_index(drop=True)

    # market_p_home: pregame_price_winner is the price for whichever team
    # ENDED UP winning, not fixed to home/away. Relabeling via the known
    # winner_team is a safe re-expression of a still-pre-game price -- not a
    # leak, since the price itself was recorded before the game.
    games["market_p_home"] = np.where(
        games["winner_team"] == games["home_team"],
        games["pregame_price_winner"],
        1 - games["pregame_price_winner"],
    )

    # market_diff_home: spread_line is signed relative to whichever team is
    # the FAVORITE (parsed from the market question, e.g. "Spurs (-5.5)"),
    # not always home. Betting convention: a negative line means the
    # favorite is expected to win by |line|, i.e. favorite's expected
    # margin ~= -spread_line.
    is_home_fav = games["spread_team"] == games["home_team"]
    is_away_fav = games["spread_team"] == games["away_team"]
    unmatched_favorite = games["spread_line"].notna() & ~(is_home_fav | is_away_fav)
    if unmatched_favorite.any():
        raise ValueError(
            f"{unmatched_favorite.sum()} games.csv rows have a spread_line but spread_team doesn't "
            f"match either home_team or away_team -- favorite-sign logic can't be trusted, aborting "
            f"rather than silently producing wrong-signed market_diff_home."
        )
    games["market_diff_home"] = np.select(
        [is_home_fav, is_away_fav], [-games["spread_line"], games["spread_line"]], default=np.nan
    )
    games["market_total"] = games["totals_line"]

    games["has_spread"] = games["spread_line"].notna()
    games["has_totals"] = games["totals_line"].notna()
    games["has_moneyline"] = games["pregame_price_winner"].notna()

    return games


def generate_holdout_predictions(cfg, fold) -> tuple:
    result = run_split(
        cfg,
        fold.train_end_date,
        fold.validation_start_date,
        fold.validation_end_date,
        fold.test_start_date,
        fold.test_end_date,
        keep_artifacts=True,
    )

    test_features = result.test_features
    cols = [
        "GAME_ID",
        "GAME_DATE",
        "HOME_TEAM_ID",
        "AWAY_TEAM_ID",
        "PTS_home",
        "PTS_away",
        "home_team_back_to_back",
        "away_team_back_to_back",
        "home_team_n_out",
        "away_team_n_out",
    ]
    pred_df = test_features[[c for c in cols if c in test_features.columns]].copy()

    preds = result.predictor.predict(test_features[result.feature_cols])
    pred_df["model_home_pred"] = preds[:, 0]
    pred_df["model_away_pred"] = preds[:, 1]
    pred_df["model_diff_pred"] = pred_df["model_home_pred"] - pred_df["model_away_pred"]
    pred_df["model_total_pred"] = pred_df["model_home_pred"] + pred_df["model_away_pred"]
    pred_df["actual_diff"] = pred_df["PTS_home"] - pred_df["PTS_away"]
    pred_df["actual_total"] = pred_df["PTS_home"] + pred_df["PTS_away"]

    # residual_std computed fresh from this run's own held-out sample --
    # never imported from elsewhere, matching naive_baseline_metrics' own
    # per-split convention in cv_harness.py.
    residual_std = np.std(pred_df["actual_diff"] - pred_df["model_diff_pred"]) or 1.0
    pred_df["model_p_home"] = norm.cdf(pred_df["model_diff_pred"] / residual_std)

    pred_df["game_date_norm"] = pd.to_datetime(pred_df["GAME_DATE"]).dt.normalize()

    test_start = pd.Timestamp(fold.test_start_date)
    test_end = pd.Timestamp(fold.test_end_date)
    if not pred_df["game_date_norm"].between(test_start, test_end).all():
        bad = pred_df.loc[~pred_df["game_date_norm"].between(test_start, test_end), "game_date_norm"]
        raise AssertionError(
            f"{len(bad)} predicted games fall outside fold {fold.name!r}'s own test window "
            f"[{fold.test_start_date}, {fold.test_end_date}] -- held-out guarantee violated, aborting. "
            f"Offending dates: {sorted(bad.unique())[:5]}"
        )

    return pred_df, result


def join_predictions_with_market(pred_df: pd.DataFrame, market_df: pd.DataFrame, fold) -> tuple:
    key = ["HOME_TEAM_ID", "AWAY_TEAM_ID", "game_date_norm"]
    market_key = ["home_team_id", "away_team_id", "game_date_norm"]

    dup_pred = pred_df[pred_df.duplicated(key, keep=False)]
    if len(dup_pred):
        raise AssertionError(
            f"{len(dup_pred)} duplicate (home,away,date) keys in held-out predictions -- unexpected."
        )
    dup_mkt = market_df[market_df.duplicated(market_key, keep=False)]
    if len(dup_mkt):
        raise AssertionError(f"{len(dup_mkt)} duplicate (home,away,date) keys in games.csv -- unexpected.")

    test_start = pd.Timestamp(fold.test_start_date)
    test_end = pd.Timestamp(fold.test_end_date)
    market_in_window = market_df[market_df["game_date_norm"].between(test_start, test_end)]
    n_market_in_window = len(market_in_window)

    if n_market_in_window == 0:
        raise ValueError(
            f"Fold {fold.name!r}'s test window ({fold.test_start_date}..{fold.test_end_date}) has "
            f"ZERO date overlap with games.csv's own date range "
            f"({market_df['game_date_norm'].min().date()}..{market_df['game_date_norm'].max().date()}). "
            f"Polymarket coverage only exists for the 2025-26 season today; pick a later --fold or wait "
            f"for more seasons of odds data. Refusing to produce an empty/meaningless comparison."
        )

    joined = pred_df.merge(
        market_df,
        left_on=key,
        right_on=market_key,
        how="inner",
        suffixes=("", "_mkt"),
    )

    unmatched_pred = pred_df[~pred_df.set_index(key).index.isin(joined.set_index(key).index)]
    unmatched_market = market_in_window[
        ~market_in_window.set_index(market_key).index.isin(joined.set_index(key).index)
    ]

    # Diagnostic-only: would a +/-1-day tolerance have caught more matches?
    # Never used to actually relax the join -- keeps the primary result
    # strict/deterministic; only surfaces whether a systematic date-boundary
    # mismatch (e.g. a timezone-crossing tipoff) is worth investigating.
    near_miss = 0
    if len(unmatched_pred) and len(unmatched_market):
        for _, row in unmatched_pred.iterrows():
            same_teams = unmatched_market[
                (unmatched_market["home_team_id"] == row["HOME_TEAM_ID"])
                & (unmatched_market["away_team_id"] == row["AWAY_TEAM_ID"])
            ]
            if len(same_teams) and (
                same_teams["game_date_norm"] - row["game_date_norm"]
            ).abs().min() <= pd.Timedelta(days=1):
                near_miss += 1

    diagnostics = {
        "n_test": len(pred_df),
        "n_market_in_window": n_market_in_window,
        "n_joined": len(joined),
        "n_unmatched_pred": len(unmatched_pred),
        "n_unmatched_market": len(unmatched_market),
        "n_near_miss_1day": near_miss,
        "sample_unmatched_pred_game_ids": unmatched_pred["GAME_ID"].head(5).tolist(),
        "sample_unmatched_market_slugs": (
            unmatched_market["slug"].head(5).tolist() if "slug" in unmatched_market else []
        ),
    }
    return joined, diagnostics


def sanity_check_sign_conventions(joined: pd.DataFrame) -> dict:
    spread_subset = joined[joined["has_spread"]]
    corr_spread = spread_subset[["market_diff_home", "actual_diff"]].corr().iloc[0, 1]
    if not (corr_spread > 0.3):
        raise AssertionError(
            f"market_diff_home vs actual_diff correlation is {corr_spread:.3f} -- expected solidly "
            f"positive; check the favorite/home sign logic (possible flip bug)."
        )

    ml_subset = joined[joined["has_moneyline"]].copy()
    ml_subset["centered"] = ml_subset["market_p_home"] - 0.5
    corr_ml = ml_subset[["centered", "actual_diff"]].corr().iloc[0, 1]
    if not (corr_ml > 0.2):
        raise AssertionError(
            f"market_p_home vs actual_diff correlation is {corr_ml:.3f} -- check winner_team "
            f"relabeling logic."
        )

    market_pick_acc = ((ml_subset["market_p_home"] > 0.5) == (ml_subset["actual_diff"] > 0)).mean()
    if not (market_pick_acc > 0.5):
        raise AssertionError(
            f"Market win-pick accuracy is {market_pick_acc:.1%} -- should beat a coin flip; check "
            f"relabeling logic."
        )

    return {
        "market_diff_home_actual_diff_corr": corr_spread,
        "market_p_home_actual_diff_corr": corr_ml,
        "market_pick_acc": market_pick_acc,
    }


def compute_accuracy_comparison(joined: pd.DataFrame) -> dict:
    diff_sub = joined[joined["has_spread"]]
    total_sub = joined[joined["has_totals"]]
    ml_sub = joined[joined["has_moneyline"]]

    return {
        "n_has_spread": len(diff_sub),
        "n_has_totals": len(total_sub),
        "n_has_moneyline": len(ml_sub),
        "model_diff_mae": (diff_sub["actual_diff"] - diff_sub["model_diff_pred"]).abs().mean(),
        "market_diff_mae": (diff_sub["actual_diff"] - diff_sub["market_diff_home"]).abs().mean(),
        "model_total_mae": (total_sub["actual_total"] - total_sub["model_total_pred"]).abs().mean(),
        "market_total_mae": (total_sub["actual_total"] - total_sub["market_total"]).abs().mean(),
        "model_win_acc": ((diff_sub["model_diff_pred"] > 0) == (diff_sub["actual_diff"] > 0)).mean(),
        "market_win_acc": ((diff_sub["market_diff_home"] > 0) == (diff_sub["actual_diff"] > 0)).mean(),
        "model_brier": ((ml_sub["model_p_home"] - (ml_sub["actual_diff"] > 0).astype(float)) ** 2).mean(),
        "market_brier": ((ml_sub["market_p_home"] - (ml_sub["actual_diff"] > 0).astype(float)) ** 2).mean(),
    }


def compute_disagreement_analysis(joined: pd.DataFrame, quantile: float) -> tuple:
    diff_sub = joined[joined["has_spread"]].copy()
    total_sub = joined[joined["has_totals"]].copy()

    diff_sub["disagreement_diff"] = diff_sub["model_diff_pred"] - diff_sub["market_diff_home"]
    diff_sub["abs_disagreement_diff"] = diff_sub["disagreement_diff"].abs()
    diff_sub["model_diff_err"] = (diff_sub["actual_diff"] - diff_sub["model_diff_pred"]).abs()
    diff_sub["market_diff_err"] = (diff_sub["actual_diff"] - diff_sub["market_diff_home"]).abs()
    diff_sub["diff_winner"] = np.select(
        [
            diff_sub["model_diff_err"] < diff_sub["market_diff_err"],
            diff_sub["model_diff_err"] > diff_sub["market_diff_err"],
        ],
        ["model", "market"],
        default="tie",
    )
    diff_sub["model_would_cover"] = np.sign(
        diff_sub["model_diff_pred"] - diff_sub["market_diff_home"]
    ) == np.sign(diff_sub["actual_diff"] - diff_sub["market_diff_home"])

    total_sub["disagreement_total"] = total_sub["model_total_pred"] - total_sub["market_total"]
    total_sub["abs_disagreement_total"] = total_sub["disagreement_total"].abs()
    total_sub["model_total_err"] = (total_sub["actual_total"] - total_sub["model_total_pred"]).abs()
    total_sub["market_total_err"] = (total_sub["actual_total"] - total_sub["market_total"]).abs()
    total_sub["total_winner"] = np.select(
        [
            total_sub["model_total_err"] < total_sub["market_total_err"],
            total_sub["model_total_err"] > total_sub["market_total_err"],
        ],
        ["model", "market"],
        default="tie",
    )

    threshold = diff_sub["abs_disagreement_diff"].quantile(quantile)
    high = diff_sub[diff_sub["abs_disagreement_diff"] >= threshold]
    low = diff_sub[diff_sub["abs_disagreement_diff"] < threshold]

    summary = {
        "diff_winner_rate_model": (diff_sub["diff_winner"] == "model").mean(),
        "diff_winner_rate_market": (diff_sub["diff_winner"] == "market").mean(),
        "diff_winner_rate_tie": (diff_sub["diff_winner"] == "tie").mean(),
        "total_winner_rate_model": (total_sub["total_winner"] == "model").mean(),
        "total_winner_rate_market": (total_sub["total_winner"] == "market").mean(),
        "total_winner_rate_tie": (total_sub["total_winner"] == "tie").mean(),
        "high_disagreement_threshold": threshold,
        "n_high_disagreement": len(high),
        "model_win_rate_high_disagreement": (high["diff_winner"] == "model").mean() if len(high) else np.nan,
        "market_win_rate_high_disagreement": (
            (high["diff_winner"] == "market").mean() if len(high) else np.nan
        ),
        "n_low_disagreement": len(low),
        "model_win_rate_low_disagreement": (low["diff_winner"] == "model").mean() if len(low) else np.nan,
        "market_win_rate_low_disagreement": (low["diff_winner"] == "market").mean() if len(low) else np.nan,
        "model_would_cover_rate": diff_sub["model_would_cover"].mean(),
    }
    return summary, diff_sub, total_sub


def compute_concentration_buckets(diff_sub: pd.DataFrame, min_bucket_n: int) -> pd.DataFrame:
    rows = []

    # (a) disagreement-magnitude quartiles -- boundaries computed fresh from
    # this run's own sample (a fixed RULE, not a fixed number).
    quartile_labels = pd.qcut(
        diff_sub["abs_disagreement_diff"], 4, labels=["Q1", "Q2", "Q3", "Q4"], duplicates="drop"
    )
    for label in quartile_labels.cat.categories if hasattr(quartile_labels, "cat") else []:
        sub = diff_sub[quartile_labels == label]
        rows.append(_bucket_row("disagreement_quartile", str(label), sub, min_bucket_n))

    # (b) market-closeness buckets -- fixed absolute cutoffs, decided before
    # seeing any results.
    for lo, hi, label in MARKET_CLOSENESS_BUCKETS:
        sub = diff_sub[(diff_sub["market_diff_home"].abs() >= lo) & (diff_sub["market_diff_home"].abs() < hi)]
        rows.append(_bucket_row("market_closeness", label, sub, min_bucket_n))

    # (c) optional condition buckets already available in test_features.
    if "home_team_back_to_back" in diff_sub.columns and "away_team_back_to_back" in diff_sub.columns:
        b2b_mask = (diff_sub["home_team_back_to_back"] == 1) | (diff_sub["away_team_back_to_back"] == 1)
        rows.append(_bucket_row("schedule_condition", "either_team_b2b", diff_sub[b2b_mask], min_bucket_n))
        rows.append(_bucket_row("schedule_condition", "neither_team_b2b", diff_sub[~b2b_mask], min_bucket_n))

    if "home_team_n_out" in diff_sub.columns and "away_team_n_out" in diff_sub.columns:
        n_out = diff_sub["home_team_n_out"].fillna(0) + diff_sub["away_team_n_out"].fillna(0)
        for lo, hi, label in [(0, 1, "n_out_0"), (1, 2, "n_out_1"), (2, np.inf, "n_out_2plus")]:
            sub = diff_sub[(n_out >= lo) & (n_out < hi)]
            rows.append(_bucket_row("injury_condition", label, sub, min_bucket_n))

    return pd.DataFrame(rows)


def _bucket_row(dimension: str, label: str, sub: pd.DataFrame, min_bucket_n: int) -> dict:
    n = len(sub)
    return {
        "bucket_dimension": dimension,
        "bucket_label": label,
        "n": n,
        "model_win_rate": (sub["diff_winner"] == "model").mean() if n else np.nan,
        "market_win_rate": (sub["diff_winner"] == "market").mean() if n else np.nan,
        "tie_rate": (sub["diff_winner"] == "tie").mean() if n else np.nan,
        "low_n_flag": n < min_bucket_n,
    }


def print_banner(cfg, fold, diagnostics: dict, sanity: dict) -> None:
    print("=" * 78)
    print(f"MARKET BENCHMARK -- fold={fold.name}")
    print(f"  train:      {cfg.datasets_loading.train_start_date} .. {fold.train_end_date}")
    print(f"  validation: {fold.validation_start_date} .. {fold.validation_end_date}")
    print(f"  test:       {fold.test_start_date} .. {fold.test_end_date}")
    print(
        "  Test window is chronologically disjoint from train/validation (enforced by "
        "validate_fold_definitions + this script's own GAME_DATE assertion) -- predictions below "
        "are genuinely held-out, not in-sample."
    )
    print("-" * 78)
    print(f"  n_test (full held-out set):        {diagnostics['n_test']:,}")
    print(f"  n_market_in_window (games.csv):    {diagnostics['n_market_in_window']:,}")
    print(f"  n_joined:                          {diagnostics['n_joined']:,}")
    print(f"  n_unmatched (our side):            {diagnostics['n_unmatched_pred']:,}")
    print(f"  n_unmatched (market side):         {diagnostics['n_unmatched_market']:,}")
    print(f"  n_near_miss (+/-1 day, diagnostic only, not used in join): {diagnostics['n_near_miss_1day']:,}")
    if diagnostics["n_unmatched_pred"]:
        print(f"    sample unmatched GAME_IDs (ours): {diagnostics['sample_unmatched_pred_game_ids']}")
    if diagnostics["n_unmatched_market"]:
        print(f"    sample unmatched slugs (market):  {diagnostics['sample_unmatched_market_slugs']}")
    print("-" * 78)
    print("  Sign-convention sanity checks (all must pass or the script aborts):")
    print(
        f"    corr(market_diff_home, actual_diff)   = {sanity['market_diff_home_actual_diff_corr']:+.3f}  (must be > 0.3)"
    )
    print(
        f"    corr(market_p_home-0.5, actual_diff)  = {sanity['market_p_home_actual_diff_corr']:+.3f}  (must be > 0.2)"
    )
    print(f"    market straight-up pick accuracy      = {sanity['market_pick_acc']:.1%}  (must be > 50%)")
    print("=" * 78)


def write_per_game_csv(joined: pd.DataFrame, tag: str) -> Path:
    out_cols = [
        "GAME_ID",
        "GAME_DATE",
        "HOME_TEAM_ID",
        "AWAY_TEAM_ID",
        "home_team",
        "away_team",
        "PTS_home",
        "PTS_away",
        "actual_diff",
        "actual_total",
        "model_home_pred",
        "model_away_pred",
        "model_diff_pred",
        "model_total_pred",
        "model_p_home",
        "market_p_home",
        "market_diff_home",
        "market_total",
        "spread_team",
        "spread_line",
        "totals_line",
        "data_quality",
        "has_spread",
        "has_totals",
        "has_moneyline",
        "home_team_back_to_back",
        "away_team_back_to_back",
        "home_team_n_out",
        "away_team_n_out",
    ]
    out_path = REPO / "outputs" / f"market_benchmark_games_{tag}.csv"
    joined[[c for c in out_cols if c in joined.columns]].to_csv(out_path, index=False)
    return out_path


def append_summary_row(row: dict, summary_csv: Path) -> None:
    row_df = pd.DataFrame([row])
    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    write_header = not summary_csv.exists()
    row_df.to_csv(summary_csv, mode="a", header=write_header, index=False)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--tag", required=True, help="Run tag, used in the per-game output filename")
    parser.add_argument("--fold", default=None, help="CV fold name (default: latest fold, cfg.cv.folds[-1])")
    parser.add_argument("--games-csv", default="data/polymarket_prices/games.csv")
    parser.add_argument("--summary-csv", default="outputs/market_benchmark_summary.csv")
    parser.add_argument("--high-disagreement-quantile", type=float, default=0.5)
    parser.add_argument("--min-bucket-n", type=int, default=30)
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    validate_fold_definitions(cfg.cv.folds)

    if args.fold:
        fold = next((f for f in cfg.cv.folds if f.name == args.fold), None)
        if fold is None:
            raise ValueError(f"No fold named {args.fold!r} in configs/config.yaml's cv.folds")
    else:
        fold = cfg.cv.folds[-1]

    print(
        f"Generating held-out predictions for fold {fold.name!r} (keep_artifacts=True, fresh train+predict)..."
    )
    pred_df, split_result = generate_holdout_predictions(cfg, fold)

    print(f"Loading Polymarket odds from {args.games_csv}...")
    market_df = load_polymarket_games(REPO / args.games_csv)

    joined, diagnostics = join_predictions_with_market(pred_df, market_df, fold)
    sanity = sanity_check_sign_conventions(joined)
    print_banner(cfg, fold, diagnostics, sanity)

    print("\n=== Q1: accuracy comparison (model vs. market) ===")
    q1 = compute_accuracy_comparison(joined)
    for k, v in q1.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
    print(
        "  NOTE: brier scores compare against Polymarket's raw last-traded price -- no de-vig exists "
        "for this market (each of moneyline/spread/totals is an independent binary market, not a "
        "two-sided sportsbook line with an explicit vig to strip). This is NOT a comparison against a "
        "de-vigged 'true' probability."
    )
    print(
        f"\n  For reference, model's full held-out test-set metrics ({diagnostics['n_test']} games, not just the "
        f"{diagnostics['n_joined']} joined with market data): {split_result.test_metrics}"
    )

    print("\n=== Q2: disagreement analysis ===")
    q2, diff_sub, total_sub = compute_disagreement_analysis(joined, args.high_disagreement_quantile)
    for k, v in q2.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
    print(
        "  'Winner' = whoever's prediction had smaller |actual - prediction| for that game (magnitude/"
        "closeness, not a binary cover/no-cover threshold). model_would_cover is a secondary, ATS-style "
        "framing (would following the model's pick relative to spread_line have covered) -- informational "
        "only, not the primary definition of 'right'."
    )

    print("\n=== Q3: concentration of disagreement-accuracy ===")
    q3 = compute_concentration_buckets(diff_sub, args.min_bucket_n)
    print(q3.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    low_n = q3[q3["low_n_flag"]]
    if len(low_n):
        print(
            f"\n  {len(low_n)} bucket(s) below --min-bucket-n={args.min_bucket_n}, excluded from headline claims:"
        )
        print(f"    {low_n['bucket_label'].tolist()}")

    games_csv_path = write_per_game_csv(joined, args.tag)
    print(f"\nWrote per-game join to {games_csv_path}")

    summary_row = {
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "tag": args.tag,
        "fold_name": fold.name,
        "test_start_date": fold.test_start_date,
        "test_end_date": fold.test_end_date,
        **diagnostics,
        **{f"q1_{k}": v for k, v in q1.items()},
        **{f"q2_{k}": v for k, v in q2.items()},
        **sanity,
    }
    summary_row.pop("sample_unmatched_pred_game_ids", None)
    summary_row.pop("sample_unmatched_market_slugs", None)
    summary_csv_path = REPO / args.summary_csv
    append_summary_row(summary_row, summary_csv_path)
    print(f"Appended summary row to {summary_csv_path}")


if __name__ == "__main__":
    main()
