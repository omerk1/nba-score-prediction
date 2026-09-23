"""Shot-quality aggregates from the raw event stream.

Distinct from the shot-*mix* metrics in `aggregates.py` (rim/mid/three rate),
which describe where a team shoots from. This module asks how good those shots
are relative to a league baseline, and whether makes ran above or below what
the locations imply:

  xpps        expected points per field-goal attempt, from a league make rate
              per (shot value, distance bin) applied to the team's own shots
  pps         actual points per field-goal attempt
  pps_vs_x    pps - xpps, i.e. shot-making over expectation

The defensive mirror is the interesting contrast. `opp_xpps` (the quality of
shot a defense concedes) is a defensive skill claim; `opp_pps_vs_x` (whether
opponents then miss more than the locations imply) is mostly luck and is
expected to fail the persistence screen. Both are measured so the screen can
separate them.

Reads `pbp_events` directly rather than the possession table, which keeps only
each possession's last shot.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Finer near the rim, where make rate falls off fastest with distance.
DISTANCE_BINS = [-0.1, 3, 7, 11, 15, 19, 22, 23.75, 26, 29, 35, 100]
SHOT_TYPES = ("Made Shot", "Missed Shot")
MIN_SHOTS = 50  # below this a block's shot-quality ratio is NaN rather than noise


def load_shots(conn, game_ids: list[str] | None = None) -> pd.DataFrame:
    """One row per field-goal attempt, with the shooting team and its opponent."""
    q = ("SELECT game_id, action_id, period, team_id, action_type, shot_distance, "
         "shot_value, x_legacy, y_legacy, sub_type FROM pbp_events WHERE action_type IN (?, ?)")
    shots = pd.read_sql_query(q, conn, params=SHOT_TYPES)
    shots["made"] = (shots.action_type == "Made Shot").astype(int)
    shots["points"] = shots.made * shots.shot_value
    if game_ids is not None:
        shots = shots[shots.game_id.isin(game_ids)]
    # Opponent per (game, team): the other team id appearing in that game's shots.
    teams = shots.groupby("game_id").team_id.unique()
    opp_map = {}
    for gid, ts in teams.items():
        if len(ts) == 2:
            opp_map[(gid, ts[0])] = ts[1]
            opp_map[(gid, ts[1])] = ts[0]
    shots["opp_id"] = [opp_map.get((g, t), np.nan) for g, t in zip(shots.game_id, shots.team_id)]
    shots["dist_bin"] = pd.cut(shots.shot_distance, DISTANCE_BINS, labels=False)
    return shots.dropna(subset=["opp_id"])


def fit_xfg(shots: pd.DataFrame) -> pd.Series:
    """League make rate per (shot_value, distance bin) — the expectation every
    team's shots are scored against. Low-dimensional on purpose: the signal
    being tested is a team's deviation from league norms at a location, so the
    baseline must not absorb team identity."""
    return shots.groupby(["shot_value", "dist_bin"]).made.mean().rename("lg_make")


def shot_quality_blocks(shots: pd.DataFrame, block_map: pd.DataFrame,
                        xfg: pd.Series | None = None) -> pd.DataFrame:
    """Per (team, block) offensive and defensive shot-quality aggregates.

    `block_map` assigns each (team, game_id) to a block; it comes from the same
    `assign_blocks` call the possession aggregates use, so the two tables share
    block boundaries exactly.
    """
    if xfg is None:
        xfg = fit_xfg(shots)
    s = shots.merge(xfg.reset_index(), on=["shot_value", "dist_bin"], how="left")
    s["x_points"] = s.lg_make * s.shot_value

    # Offensive rows key on the shooting team, defensive rows on its opponent.
    off = s.rename(columns={"team_id": "team"})
    de = s.rename(columns={"opp_id": "team"})
    out = {}
    for name, frame in (("", off), ("opp_", de)):
        f = frame.merge(block_map, on=["team", "game_id"])
        g = f.groupby(["team", "block"])
        n = g.size()
        xpps = g.x_points.mean().where(n >= MIN_SHOTS)
        pps = g.points.mean().where(n >= MIN_SHOTS)
        out[f"{name}xpps"] = xpps
        out[f"{name}pps"] = pps
        out[f"{name}pps_vs_x"] = pps - xpps
        # Totals for the location-based rating adjustment below.
        out[f"{name}fg_pts"] = g.points.sum()
        out[f"{name}fg_xpts"] = g.x_points.sum()
    return pd.DataFrame(out)


def location_adjusted_net_rating(blocks: pd.DataFrame, own_weight: float = 1.0) -> pd.Series:
    """Net rating with every field goal valued at what its location implies,
    for both the team and its opponents.

    A generalisation of the 3P/FT luck adjustment in `aggregates.py`: instead
    of correcting only three-pointers against one league rate, it corrects
    every attempt against the league make rate at that distance and shot
    value. `own_weight` scales how much of the team's own over-performance is
    removed; the 3P/FT sweep found full removal overshoots, since making more
    than a location implies is partly shooting skill rather than variance.

    Needs `net_rtg_nogarbage`, `poss_off`, `poss_def` and the shot totals from
    `shot_quality_blocks` on the same block index.
    """
    own_luck = blocks.fg_pts - blocks.fg_xpts
    opp_luck = blocks.opp_fg_pts - blocks.opp_fg_xpts
    return (blocks.net_rtg_nogarbage
            - 100 * own_weight * own_luck / blocks.poss_off
            + 100 * opp_luck / blocks.poss_def)
