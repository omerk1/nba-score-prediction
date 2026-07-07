"""
Phase 0: empirically calibrate Layer 2 injury-impact deltas (design doc "v2").

For each archetype, compare a team's Layer-1 style fingerprint in games where a
player of that archetype was reported Out (via the pre-game NBA official injury
PDF backfill — see run_historical() in src/news_scraping/pipeline.py, which is
pre-game reporting, so this is valid for both backtesting and live prediction;
this was already resolved and is not re-investigated here) vs. that same
team-season's baseline (games where the archetype was NOT missing).

Scope is restricted to game_dates present in scrape_log (source='pdf') so that
"no rows in player_injuries for this team/date" reliably means "confirmed no one
out" rather than "not scraped that day".

**Decay-weighted "missing" sample (fix for the wrap-up round's item #3 finding):**
a continuous multi-month absence (e.g. Collin Sexton's ~5-month ACL-recovery
stretch) is not a clean repeated natural experiment — deep into a long absence, a
team's roster/rotation/trades increasingly reflect "a team that has adapted," not
"the marginal effect of missing this player," so treating every qualifying Out
team-game identically let one long-absence player dominate and flip
perimeter_specialist's sign (see docs/A7_PHASE_LOG.md item #3 and the "Decay-weighted
injury-impact calibration" section). Instead of a hard cutoff on absence length (an
equally arbitrary, unexplored threshold), each Out-event is down-weighted by an
exponential decay on its position within its continuous absence streak (1 = first
game of the absence, 2 = second consecutive game, ...), using the SAME decay
mechanism (`_decay_weight`, `weight = 0.5 ** (position / halflife)`) already used by
fingerprint.py's rolling-window recency weighting. Baseline games are NOT part of any
absence streak, so no decay applies there — the baseline mean stays a plain mean.
"""

import logging

import numpy as np
import pandas as pd

from src.matchups.config import CONFIG_PATH, load_style_matchup_config
from src.matchups.db import cache_conn, init_cache_db, injury_conn
from src.matchups.fingerprint import FINGERPRINT_METRICS, _decay_weight, build_fingerprint_cache
from src.matchups.players import _date_to_season_str, build_archetype_cache, build_name_resolution_cache

logger = logging.getLogger(__name__)

# Only these metrics matter per archetype (design doc injury_impact block) —
# calibration still computes all 5 for transparency but only these feed Layer 2.
TARGET_METRICS = {
    "facilitator": ["assist_rate", "pace_score"],
    "scorer": ["three_pt_reliance", "paint_activity"],
    # combo is new (added in Phase 0 archetype exploration, not in the design doc) —
    # no prior assumption about which metric it should move, so calibrate against
    # all 5 fingerprint metrics and let the data decide.
    "combo": list(FINGERPRINT_METRICS),
    "rim_protector": ["defensive_rating", "paint_activity"],
    "perimeter_specialist": ["defensive_rating"],
}

# Decay halflife (in consecutive qualifying team-games) for down-weighting Out-events
# by their position in a continuous absence streak. Chosen from a grid (5, 10, 20, 40,
# 10000-as-no-decay-reference) explored in the "Decay-weighted injury-impact
# calibration" phase-log section: 20 is the smallest grid value at which
# perimeter_specialist's defensive_rating delta is UNAMBIGUOUSLY positive (+0.2948, not
# a razor-thin crossing like halflife=40's +0.1231) while every other archetype still
# retains >=76% of its effective (weighted) sample size -- halflife=10 gets a stronger
# perimeter_specialist flip (+0.5122) but costs noticeably more effective sample for
# archetypes that do NOT have this problem (scorer retention drops to 67.5%,
# facilitator's pace_score delta more than doubles in magnitude) for marginal
# additional benefit given 20 already resolves the sign flip solidly. See phase log for
# the full per-halflife table and per-archetype retention numbers.
DEFAULT_DECAY_HALFLIFE_GAMES = 20.0


def _scraped_dates() -> set[str]:
    with injury_conn() as conn:
        rows = conn.execute("SELECT DISTINCT game_date FROM scrape_log WHERE source = 'pdf'").fetchall()
    return {r["game_date"][:10] for r in rows}


def _out_events_by_player() -> pd.DataFrame:
    """player_injuries Out rows resolved to player_id + archetype, at PLAYER granularity
    (not collapsed to team-game-archetype) — needed to reconstruct each player's own
    continuous absence streak. game_date is normalized to date-only (YYYY-MM-DD).
    See _out_players_by_team_date() for the collapsed, archetype-presence-only view
    used to flag which fingerprint rows qualify as "missing"."""
    with injury_conn() as conn:
        inj = pd.read_sql_query(
            "SELECT game_date, team_id, player_name FROM player_injuries WHERE status = 'Out'", conn
        )
    cache = cache_conn()
    res = pd.read_sql_query(
        "SELECT player_name, player_id, confidence FROM player_name_resolution "
        "WHERE confidence IN ('high','medium')",
        cache,
    )
    arch = pd.read_sql_query("SELECT player_id, season, archetype FROM player_archetypes", cache)
    cache.close()

    inj = inj.merge(res, on="player_name", how="inner")
    inj["season"] = inj["game_date"].map(_date_to_season_str)
    inj = inj.merge(arch, on=["player_id", "season"], how="left")
    inj = inj[inj["archetype"].notna()]
    inj["game_date"] = inj["game_date"].str[:10]
    return inj[["game_date", "team_id", "player_id", "archetype"]].drop_duplicates()


def _out_players_by_team_date() -> pd.DataFrame:
    """(game_date, team_id, archetype) presence flags, collapsed from _out_events_by_player()
    (drops player_id — multiple same-archetype players Out on one team-game collapse to a
    single row, matching the pre-existing "was this archetype missing" binary semantics)."""
    events = _out_events_by_player()
    return events[["game_date", "team_id", "archetype"]].drop_duplicates()


def _streak_positions(out_events: pd.DataFrame, team_game_dates: dict) -> pd.DataFrame:
    """Adds a `streak_position` column to out_events: 1 = first game of this continuous
    absence, 2 = second consecutive game missed, etc. Reconstructed per (player_id,
    team_id) against that team's actual qualifying game schedule (`team_game_dates`:
    team_id -> sorted list of date-only strings, already restricted to scraped dates by
    the caller — same "no report = confirmed not out" logic used elsewhere in this
    module). A qualifying team-game where the player was NOT reported Out ends the
    streak; so does a gap where the date isn't in team_game_dates at all.
    """
    out_events = out_events.copy()
    positions = pd.Series(1, index=out_events.index, dtype=int)
    for (pid, tid), grp in out_events.groupby(["player_id", "team_id"]):
        dates = team_game_dates.get(tid, [])
        out_dates = set(grp["game_date"])
        pos_by_date: dict = {}
        streak = 0
        for d in dates:
            if d in out_dates:
                streak += 1
                pos_by_date[d] = streak
            else:
                streak = 0
        for idx, d in grp["game_date"].items():
            # Fallback (should not normally trigger — team_game_dates is built from the
            # same scraped/fp universe out_events is filtered to): default to streak
            # position 1 (max weight / least aggressive down-weighting) if a date is
            # somehow missing from that team's reconstructed schedule.
            positions.at[idx] = pos_by_date.get(d, 1)
    out_events["streak_position"] = positions
    return out_events


def _archetype_event_weights(out_events_with_streak: pd.DataFrame, archetype: str, halflife: float) -> dict:
    """(game_date, team_id) -> decay weight for this archetype's Out-event.

    When multiple players of the SAME archetype are Out simultaneously for one
    team-game, use the LEAST-contaminated one (max weight / min streak_position) rather
    than summing — mirrors the existing "max, not summed" convention injury_layer.py
    already uses for severity multipliers in the identical multi-player-out situation
    (see injury_layer.py's module docstring): the calibrated delta represents "this
    archetype was missing" as a binary condition, not a per-player marginal effect, so
    summing/averaging weights across simultaneous absentees would misrepresent it.
    """
    sub = out_events_with_streak[out_events_with_streak["archetype"] == archetype]
    best_position: dict = {}
    for gd, tid, pos in zip(sub["game_date"], sub["team_id"], sub["streak_position"]):
        key = (gd, tid)
        if key not in best_position or pos < best_position[key]:
            best_position[key] = pos
    return {key: float(_decay_weight(pos, halflife)) for key, pos in best_position.items()}


def prepare_calibration_inputs(min_games: int = 5) -> dict:
    """Shared prep for run_calibration() and the halflife-exploration script: builds the
    Layer-1 fingerprint table, the archetype-presence map, and the streak-annotated Out
    events. Split out so the exploration script can try several `decay_halflife_games`
    values WITHOUT re-running build_fingerprint_cache()/re-reading the DB each time —
    streak_position itself doesn't depend on halflife, only the weight conversion does.
    """
    init_cache_db()
    name_summary = build_name_resolution_cache()
    build_archetype_cache()
    build_fingerprint_cache(layer=1, min_games=min_games)

    scraped = _scraped_dates()
    out_map = _out_players_by_team_date()
    out_map = out_map[out_map["game_date"].isin(scraped)]

    conn = cache_conn()
    fp = pd.read_sql_query(
        "SELECT game_id, team_id, game_date, " + ", ".join(FINGERPRINT_METRICS) +
        " FROM matchup_fingerprints WHERE layer = 1",
        conn,
    )
    conn.close()
    fp["game_date"] = fp["game_date"].str[:10]
    fp = fp[fp["game_date"].isin(scraped)].copy()
    fp["season"] = fp["game_date"].map(_date_to_season_str)
    fp["team_season"] = fp["team_id"].astype(str) + "_" + fp["season"]

    archetype_by_key: dict[tuple, set[str]] = {}
    for gd, tid, arch in out_map.itertuples(index=False):
        archetype_by_key.setdefault((gd, tid), set()).add(arch)

    # Streak reconstruction (player-level events -> per-event streak position), shared
    # across every archetype/metric/halflife.
    out_events = _out_events_by_player()
    out_events = out_events[out_events["game_date"].isin(scraped)]
    team_game_dates = {tid: sorted(g["game_date"].unique().tolist()) for tid, g in fp.groupby("team_id")}
    out_events = _streak_positions(out_events, team_game_dates)

    return {
        "fp": fp,
        "archetype_by_key": archetype_by_key,
        "out_events": out_events,
        "coverage_rate": name_summary["coverage_rate"],
    }


def compute_deltas(fp: pd.DataFrame, archetype_by_key: dict, out_events: pd.DataFrame, decay_halflife_games: float) -> list[dict]:
    """Decay-weighted delta computation for every (archetype, metric) in TARGET_METRICS,
    given the shared prep from prepare_calibration_inputs()."""
    results = []
    for archetype, metrics in TARGET_METRICS.items():
        missing_keys = {k for k, archs in archetype_by_key.items() if archetype in archs}
        missing_mask = pd.Series(
            [(gd, tid) in missing_keys for gd, tid in zip(fp["game_date"], fp["team_id"])], index=fp.index
        )

        missing_rows = fp[missing_mask]
        # Baseline: same team-seasons, games where this archetype was NOT missing.
        # Baseline games are never part of an absence streak, so no decay weighting
        # applies to them -- plain mean, as before.
        relevant_team_seasons = missing_rows["team_season"].unique()
        baseline_rows = fp[(~missing_mask) & (fp["team_season"].isin(relevant_team_seasons))]

        weight_map = _archetype_event_weights(out_events, archetype, decay_halflife_games)
        row_weights = np.array([
            weight_map.get((gd, tid), 1.0) for gd, tid in zip(missing_rows["game_date"], missing_rows["team_id"])
        ])

        for metric in metrics:
            n_without = len(missing_rows)
            n_baseline = len(baseline_rows)
            if n_without == 0 or n_baseline == 0:
                delta = 0.0
            else:
                if row_weights.sum() > 0:
                    weighted_mean = float(np.average(missing_rows[metric], weights=row_weights))
                else:
                    weighted_mean = float(missing_rows[metric].mean())
                delta = weighted_mean - float(baseline_rows[metric].mean())
            results.append({
                "archetype": archetype, "metric": metric, "delta": round(delta, 4),
                "n_games_without": n_without, "n_games_baseline": n_baseline,
            })
    return results


def run_calibration(min_games: int = 5, decay_halflife_games: float | None = None, persist: bool = True) -> dict:
    if decay_halflife_games is None:
        decay_halflife_games = load_style_matchup_config().get(
            "injury_calibration_decay_halflife_games", DEFAULT_DECAY_HALFLIFE_GAMES
        )
    inputs = prepare_calibration_inputs(min_games)
    coverage_note = inputs["coverage_rate"]
    low_conf_flag = coverage_note < 0.8
    results = compute_deltas(inputs["fp"], inputs["archetype_by_key"], inputs["out_events"], decay_halflife_games)

    if not persist:
        summary = {
            "name_resolution_coverage_rate": coverage_note,
            "low_confidence": low_conf_flag,
            "decay_halflife_games": decay_halflife_games,
            "deltas": results,
        }
        return summary

    conn = cache_conn()
    conn.executemany(
        "INSERT OR REPLACE INTO injury_calibration (archetype, metric, delta, n_games_without, n_games_baseline) "
        "VALUES (:archetype, :metric, :delta, :n_games_without, :n_games_baseline)",
        results,
    )
    conn.commit()
    conn.close()

    summary = {
        "name_resolution_coverage_rate": coverage_note,
        "low_confidence": low_conf_flag,
        "decay_halflife_games": decay_halflife_games,
        "deltas": results,
    }
    logger.info(f"Phase 0 calibration results: {summary}")
    return summary


def write_deltas_to_config(results: list[dict]) -> None:
    """Append a `style_matchup` block to configs/config.yaml with the empirical deltas.

    IMPORTANT: this appends a plain-text block rather than round-tripping the file
    through yaml.safe_load/safe_dump. A load+dump round-trip strips every inline
    `#` comment in the existing file (PyYAML doesn't preserve comments) and
    reformats quoting/indentation — that would silently destroy human-authored
    documentation throughout config.yaml. Since style_matchup does not exist yet
    in the file, a straight text append is safe and idempotent-checked below.
    """
    with open(CONFIG_PATH) as f:
        existing = f.read()

    if "style_matchup:" in existing:
        logger.info("style_matchup block already present in config.yaml — skipping append")
        return

    impact: dict = {}
    for row in results:
        impact.setdefault(row["archetype"], {})[row["metric"]] = row["delta"]

    lines = [
        "",
        "style_matchup:",
        "  # A7 exploratory module (src/matchups/). Not read by config_loader.py's Config",
        "  # schema (not integrated into feature_builder yet) — parsed directly by",
        "  # src/matchups/config.py. See docs/a7_style_matchup_design.md and",
        "  # docs/A7_PHASE_LOG.md for how these values were chosen.",
        "  fingerprint_window: 20",
        "  decay_halflife: 5",
        "  encoding: hand_picked",
        "  similarity_method: cosine   # cosine | knn -- see Phase 3 findings in phase log",
        "  similarity_threshold: 0.70",
        "  knn_k: 30",
        "  min_confidence_sample: 10",
        "  full_confidence_sample: 50",
        "  low_confidence_fallback: h2h",
        "  archetype_method: percentile",
        "  injury_impact_calibrated: true  # values below are Phase-0 empirical deltas, not v1 guesses",
        "  injury_impact:",
    ]
    for archetype, metrics in impact.items():
        lines.append(f"    {archetype}:")
        if not metrics:
            lines.append("      {}")
        for metric, delta in metrics.items():
            lines.append(f"      {metric}: {delta}")

    with open(CONFIG_PATH, "a") as f:
        f.write("\n".join(lines) + "\n")
    logger.info("Appended calibrated injury_impact deltas to configs/config.yaml")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    result = run_calibration()
    write_deltas_to_config(result["deltas"])
    print(result)
