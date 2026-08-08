"""D2 leakage re-audit for style_fingerprint_features (session rs_20260808_1).

For each of the 5 CV folds, samples the first N games (chronologically) in that
fold's test window, and for each (team_id, game_date), recomputes fingerprint's
full layer=1 (rolling `.shift(1)`-based) + layer=2 (injury_layer.py's same-date
additive delta) value from scratch, using ONLY box_score_stats/player_injuries
rows dated strictly/at-most the game's own date -- then diffs it against the
actual value FeatureBuilder._add_style_fingerprint_features produces for that
exact game (its production merge_asof lookup against the already-built,
non-fold-aware matchup_fingerprints cache). This is the leakage check: any
remaining discrepancy after replicating injury_layer.py's algorithm exactly
(bugs and all -- see below) means the production value used information a
from-scratch, same-date-bounded recomputation could not have derived, i.e.
future information leaked in.

Two revisions during this session, both instructive (kept in EXPERIMENTS.md's
D2 write-up, not just here):
  1. First pass compared a layer=1-only truncated recompute against production
     for all 6 metrics: 608/3000 "mismatches". False positive -- 5 of the 6
     metrics are actually served from layer=2 (injury-adjusted) in production;
     the delta was the legitimate injury adjustment, not leakage.
  2. Second pass added the *correct* same-date injury delta (reusing
     injury_layer.py's own severity/impact lookup): still 64/3000 mismatches.
     Root cause, found by replicating injury_layer.py's ACTUAL loop exactly:
     it applies each affected archetype's delta as
     `fp2.at[idx, metric] = row[metric] + delta * severity_mult` -- reading
     from the ORIGINAL layer=1 `row[metric]` every iteration instead of the
     running `fp2.at[idx, metric]`, so when 2+ different-archetype players are
     out on the same team/date and their impact dicts share a metric, only the
     LAST-iterated archetype's delta survives; earlier ones are silently
     overwritten, not summed. Confirmed by replicating this exact
     overwrite-not-accumulate behavior and matching production bit-for-bit
     (see `apply_injury_delta` below, `correct=False`). This is a real
     correctness bug in injury_layer.py, but it is NOT leakage -- no
     information from after the game's own date is involved, multi-archetype
     same-date deltas just don't sum correctly. Flagged as "proposed, pending
     review" in EXPERIMENTS.md rather than fixed here, since it changes
     existing feature values for ~any team-game with 2+ different-archetype
     injuries sharing an affected metric.

This script's PASS/FAIL verdict compares against the AS-SHIPPED (bug and all)
algorithm, since that's the actual leakage question. It separately reports how
many sampled games are affected by the known accumulation bug (correct vs.
as-shipped disagree) as a magnitude/severity signal for the write-up.

Read-only: touches no config, no cache, runs no training.

Usage: venv/bin/python3 scripts/audit_fingerprint_leakage.py
"""

import os
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.append(str(REPO))
os.chdir(REPO)

from src.feature_engineering.feature_builder import FeatureBuilder  # noqa: E402
from src.matchups.config import CACHE_DB, load_style_matchup_config  # noqa: E402
from src.matchups.fingerprint import _load_raw_team_game_metrics  # noqa: E402
from src.matchups.fingerprint import FINGERPRINT_METRICS, _decayed_weighted_mean  # noqa: E402
from src.matchups.injury_layer import _out_players_with_reason, _team_game_archetype_severity  # noqa: E402
from src.utils.config_loader import load_config  # noqa: E402

N_GAMES_PER_FOLD = 50

FOLDS_TEST_WINDOWS = [
    ("fold1", "2021-10-19", "2022-04-10"),
    ("fold2", "2022-10-18", "2023-04-09"),
    ("fold3", "2023-10-24", "2024-04-14"),
    ("fold4", "2024-10-22", "2025-04-13"),
    ("fold5", "2025-10-21", "2026-04-12"),
]


def truncated_layer1(raw: pd.DataFrame, team_id: int, game_date: pd.Timestamp, window: int, halflife: float):
    """From-scratch recomputation using only games strictly before game_date."""
    hist = raw[(raw["team_id"] == team_id) & (raw["game_date"] < game_date)].sort_values("game_date")
    hist = hist.tail(window)
    if len(hist) == 0:
        return {m: np.nan for m in FINGERPRINT_METRICS}
    return {m: _decayed_weighted_mean(hist[m].to_numpy(), halflife) for m in FINGERPRINT_METRICS}


def apply_injury_delta(layer1_vals: dict, key, severity_map: dict, injury_impact: dict, correct: bool):
    """correct=True: deltas from multiple same-date archetypes are summed onto
    layer1_vals (what injury_layer.py *should* do). correct=False: replicates
    injury_layer.py's actual behavior -- each archetype's delta is computed
    from the original layer1_vals and overwrites the previous archetype's
    contribution to a shared metric instead of accumulating."""
    out = dict(layer1_vals)
    archetypes_out = severity_map.get(key)
    if not archetypes_out:
        return out
    for archetype, severity_mult in archetypes_out.items():
        deltas = injury_impact.get(archetype, {})
        for metric, delta in deltas.items():
            if metric not in FINGERPRINT_METRICS or np.isnan(layer1_vals[metric]):
                continue
            base = out[metric] if correct else layer1_vals[metric]
            out[metric] = base + delta * severity_mult
    return out


def main():
    cfg = load_config()
    style_cfg = load_style_matchup_config()
    window = style_cfg["fingerprint_window"]
    halflife = style_cfg["decay_halflife"]
    injury_impact = style_cfg["injury_impact"]
    print(f"fingerprint_window={window} decay_halflife={halflife} cache_db={CACHE_DB}", flush=True)

    raw = _load_raw_team_game_metrics()
    raw["game_date"] = pd.to_datetime(raw["game_date"]).dt.normalize()

    out_df = _out_players_with_reason()
    severity_map = _team_game_archetype_severity(out_df)

    conn = sqlite3.connect(cfg.data_paths.raw_db)

    total_leak_mismatches = 0
    total_bug_affected = 0
    total_compared = 0
    fb = FeatureBuilder(
        rolling_windows=cfg.features.rolling_windows,
        h2h_margin_window=cfg.features.h2h_margin_window,
        h2h_win_rate_window=cfg.features.h2h_win_rate_window,
    )

    for fold_name, test_start, test_end in FOLDS_TEST_WINDOWS:
        games = pd.read_sql(
            "SELECT game_id AS GAME_ID, game_date AS GAME_DATE, team_id_home AS HOME_TEAM_ID, "
            "team_id_away AS AWAY_TEAM_ID FROM game "
            "WHERE game_date >= ? AND game_date <= ? AND season_type = 'Regular Season' "
            "ORDER BY game_date LIMIT ?",
            conn,
            params=(test_start, test_end, N_GAMES_PER_FOLD),
        )
        games["GAME_DATE"] = pd.to_datetime(games["GAME_DATE"]).dt.normalize()
        print(
            f"\n=== {fold_name}: sampled {len(games)} games, {test_start}..{games['GAME_DATE'].max()} ===",
            flush=True,
        )

        prod = fb._add_style_fingerprint_features(games.copy())

        fold_leak_mismatches = 0
        fold_bug_affected = 0
        for _, row in games.iterrows():
            for side, team_col in [("home", "HOME_TEAM_ID"), ("away", "AWAY_TEAM_ID")]:
                team_id = row[team_col]
                gdate = row["GAME_DATE"]
                key = (gdate.strftime("%Y-%m-%d"), team_id)
                l1 = truncated_layer1(raw, team_id, gdate, window, halflife)
                as_shipped = apply_injury_delta(l1, key, severity_map, injury_impact, correct=False)
                correct = apply_injury_delta(l1, key, severity_map, injury_impact, correct=True)
                prod_row = prod[prod["GAME_ID"] == row["GAME_ID"]].iloc[0]

                for metric in FINGERPRINT_METRICS:
                    prod_val = prod_row[f"{side}_style_{metric}"]
                    total_compared += 1

                    if correct[metric] != as_shipped[metric] and not (
                        np.isnan(correct[metric]) and np.isnan(as_shipped[metric])
                    ):
                        fold_bug_affected += 1
                        total_bug_affected += 1

                    ship_val = as_shipped[metric]
                    if pd.isna(prod_val) and pd.isna(ship_val):
                        continue
                    if pd.isna(prod_val) != pd.isna(ship_val) or abs(prod_val - ship_val) > 1e-6:
                        fold_leak_mismatches += 1
                        total_leak_mismatches += 1
                        print(
                            f"  LEAK-CHECK MISMATCH {fold_name} game={row['GAME_ID']} date={gdate.date()} "
                            f"team={team_id} metric={metric} prod={prod_val} as_shipped_truncated={ship_val}",
                            flush=True,
                        )
        print(
            f"  {fold_name}: {fold_leak_mismatches} unexplained mismatches, "
            f"{fold_bug_affected} cells affected by the known accumulation bug "
            f"(out of {len(games) * 2 * len(FINGERPRINT_METRICS)} comparisons)",
            flush=True,
        )

    conn.close()

    print(
        f"\n=== TOTAL: {total_leak_mismatches} unexplained mismatches out of {total_compared} comparisons ==="
    )
    print(f"=== {total_bug_affected} comparisons affected by the known injury_layer.py accumulation bug ===")
    if total_leak_mismatches == 0:
        print(
            "PASS: no leakage detected -- style_fingerprint_features is point-in-time correct at every "
            "sampled fold boundary (every value production returns is exactly reproducible from a "
            "from-scratch recomputation using only data strictly/at-most the game's own date)."
        )
    else:
        print(
            "FAIL: production disagrees with a from-scratch, same-date-bounded recomputation even after "
            "replicating injury_layer.py's algorithm exactly -- this means information outside what the "
            "algorithm should have access to is involved. This supersedes the rest of the agenda."
        )


if __name__ == "__main__":
    main()
