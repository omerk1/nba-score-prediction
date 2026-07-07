"""
Decay-weighted injury-impact calibration -- halflife exploration.

Fixes the wrap-up round's item #3 finding: calibration.py's Phase 0 empirical
calibration treated every qualifying Out team-game identically regardless of how far
into a continuous absence streak it fell, which let Collin Sexton's ~5-month
ACL-recovery absence (127 of 680 qualifying event-rows for perimeter_specialist alone)
dominate the sample and flip perimeter_specialist's defensive_rating delta to the wrong
sign (-0.0889; excluding Sexton alone flips it to +0.9566). See docs/A7_PHASE_LOG.md
item #3 for the full diagnosis.

This module tries a grid of `decay_halflife_games` values (see HALFLIFE_GRID) with
calibration.py's new decay-weighted delta computation and reports, per halflife:
  - every archetype's calibrated delta (not just perimeter_specialist)
  - Collin Sexton's and Otto Porter Jr.'s effective weighted contribution to the
    perimeter_specialist sample (does it visibly shrink as halflife shrinks?)

Reuses calibration.py's `prepare_calibration_inputs()` / `compute_deltas()` /
`_archetype_event_weights()` directly -- streak_position is computed ONCE (it doesn't
depend on halflife, only the weight conversion does), so the grid search only repeats
the cheap weighting/aggregation step, not the DB reads / fingerprint rebuild.
"""

import logging

import pandas as pd

from src.matchups.calibration import TARGET_METRICS, _archetype_event_weights, compute_deltas, prepare_calibration_inputs
from src.matchups.db import cache_conn

logger = logging.getLogger(__name__)

# 5/10/20/40 games span "aggressive" to "mild" decay over a season-scale absence;
# 10000 is a practically-no-decay reference point (weight at streak position 200 is
# still 0.5**(200/10000) = 0.986) for comparison against the un-fixed behavior.
HALFLIFE_GRID = [5.0, 10.0, 20.0, 40.0, 10000.0]

DIAGNOSTIC_PLAYERS = ["Collin Sexton", "Otto Porter Jr."]


def _diagnostic_player_ids() -> dict:
    conn = cache_conn()
    res = pd.read_sql_query(
        "SELECT player_name, player_id FROM player_name_resolution WHERE player_name IN (%s)"
        % ",".join("?" * len(DIAGNOSTIC_PLAYERS)),
        conn,
        params=DIAGNOSTIC_PLAYERS,
    )
    conn.close()
    return dict(zip(res["player_name"], res["player_id"]))


def _player_contribution(out_events: pd.DataFrame, archetype: str, halflife: float, player_ids: dict) -> dict:
    """For `archetype`'s sample at this halflife: each diagnostic player's own summed
    weight (using THEIR OWN streak position, i.e. not collapsed across simultaneous
    same-archetype absentees) as a fraction of the sample's total collapsed weight (the
    actual denominator compute_deltas()/`_archetype_event_weights` uses for the weighted
    mean). This can exceed the "true" contribution when multiple same-archetype players
    were simultaneously Out (rare) since the collapsed total takes the max weight per
    team-game, not a sum -- reported as a diagnostic ratio, not a partition."""
    sub = out_events[out_events["archetype"] == archetype]
    weight_map = _archetype_event_weights(out_events, archetype, halflife)
    total_weight = sum(weight_map.values())

    contributions = {}
    for name, pid in player_ids.items():
        if pid is None:
            contributions[name] = {"n_events": 0, "own_weight_sum": 0.0, "pct_of_sample_weight": 0.0}
            continue
        prows = sub[sub["player_id"] == pid]
        own_weight_sum = float(sum(0.5 ** (pos / halflife) for pos in prows["streak_position"]))
        contributions[name] = {
            "n_events": int(len(prows)),
            "own_weight_sum": round(own_weight_sum, 2),
            "pct_of_sample_weight": round(own_weight_sum / total_weight, 4) if total_weight > 0 else 0.0,
        }
    return contributions


def run_exploration() -> dict:
    inputs = prepare_calibration_inputs(min_games=5)
    fp, archetype_by_key, out_events = inputs["fp"], inputs["archetype_by_key"], inputs["out_events"]
    player_ids = _diagnostic_player_ids()
    logger.info(f"Diagnostic player IDs resolved: {player_ids}")

    table = []
    contributions_by_halflife = {}
    for halflife in HALFLIFE_GRID:
        deltas = compute_deltas(fp, archetype_by_key, out_events, halflife)
        for row in deltas:
            table.append({"halflife": halflife, **row})
        contributions_by_halflife[halflife] = _player_contribution(
            out_events, "perimeter_specialist", halflife, player_ids
        )

    df = pd.DataFrame(table)
    return {
        "player_ids": player_ids,
        "deltas_table": df,
        "perimeter_specialist_contributions": contributions_by_halflife,
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    pd.set_option("display.width", 160)
    result = run_exploration()
    print("=== Per-halflife deltas (all archetypes) ===")
    print(result["deltas_table"].to_string(index=False))
    print()
    print("=== perimeter_specialist defensive_rating delta by halflife ===")
    ps = result["deltas_table"]
    ps = ps[(ps["archetype"] == "perimeter_specialist") & (ps["metric"] == "defensive_rating")]
    print(ps.to_string(index=False))
    print()
    print("=== Collin Sexton / Otto Porter Jr. contribution to perimeter_specialist sample ===")
    for halflife, contrib in result["perimeter_specialist_contributions"].items():
        print(f"halflife={halflife}: {contrib}")
