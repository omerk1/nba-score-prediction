"""Context-conditioned team aggregates over the possession table.

Everything here is a filtered or re-weighted sum over possessions, grouped by
(team, block) where a block is any set of a team's games (a rolling window, a
season chunk, ...). Ratios are formed at the block level from summed
numerators and denominators, never as means of per-game ratios, so sparse
contexts (clutch, trailing-by-8) are usable.

The one input is the possession table (`src/pbp/possessions.py`) with a
`game_date` column joined on; `team_view` turns it into a team-perspective
long table (each possession once for its offense and once for its defense),
which every aggregate below consumes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

REGULATION_SECS = 2880.0

# (seconds remaining in regulation, absolute margin) pairs: a possession that
# starts inside any of these is garbage time. Tiered so a 25-point lead with
# 12 minutes left and a 10-point lead with 1 minute left are both excluded.
GARBAGE_TIERS = ((720, 25), (360, 20), (180, 15), (60, 10))
CLUTCH_SECS, CLUTCH_MARGIN = 300, 5
DEFICIT_MARGIN = 8
RIM_FT, MID_FT = 4, 22
MIN_CONTEXT_POSS = 30  # below this a context ratio is NaN rather than noise


def team_view(poss: pd.DataFrame) -> pd.DataFrame:
    """Long table: one row per (possession, side) with margin from the team's
    perspective and the context flags every aggregate keys on."""
    p = poss
    remaining_reg = np.where(p.period <= 4, REGULATION_SECS - p.start_secs, np.nan)
    period_end = np.where(p.period <= 4, p.period * 720.0, REGULATION_SECS + (p.period - 4) * 300.0)
    remaining_period = period_end - p.start_secs
    abs_margin = p.margin_start.abs().to_numpy()
    garbage = np.zeros(len(p), dtype=bool)
    for secs, margin in GARBAGE_TIERS:
        garbage |= (remaining_reg <= secs) & (abs_margin >= margin)
    clutch = ((p.period >= 4) & (remaining_period <= CLUTCH_SECS) & (abs_margin <= CLUTCH_MARGIN)).to_numpy()

    base = pd.DataFrame({
        "game_id": p.game_id, "game_date": p.game_date, "poss_idx": p.poss_idx, "period": p.period,
        "start_secs": p.start_secs, "duration": p.duration, "points": p.points,
        "n_fga": p.n_fga, "n_fg3a": p.n_fg3a, "n_fta": p.n_fta, "n_tov": p.n_tov, "n_oreb": p.n_oreb,
        "n_fgm": p.n_fgm, "n_fg3m": p.n_fg3m, "n_ftm": p.n_ftm,
        "last_shot_distance": p.last_shot_distance, "last_shot_value": p.last_shot_value,
        "lineup_off": p.lineup_off, "lineup_complete": (p.lineup_off_complete & p.lineup_def_complete).astype(bool),
        "garbage": garbage, "clutch": clutch,
    })
    # `own_lineup` is the team's own five on either side of the ball, which is
    # what lineup-conditioned aggregates key on; `lineup_off` stays the
    # offense's five regardless of perspective.
    off = base.assign(team=p.off_team_id.to_numpy(), opp=p.def_team_id.to_numpy(), side="off",
                      team_margin=p.margin_start.to_numpy(), own_lineup=p.lineup_off.to_numpy())
    de = base.assign(team=p.def_team_id.to_numpy(), opp=p.off_team_id.to_numpy(), side="def",
                     team_margin=-p.margin_start.to_numpy(), own_lineup=p.lineup_def.to_numpy())
    return pd.concat([off, de], ignore_index=True)


def assign_blocks(tv: pd.DataFrame, block_games: int, drop_partial_min: int | None = None) -> pd.DataFrame:
    """Add a per-team `block` id: consecutive chunks of `block_games` games in
    date order. Blocks with fewer than `drop_partial_min` games are dropped
    (default: only full blocks kept)."""
    order = (tv[["team", "game_id", "game_date"]].drop_duplicates()
             .sort_values(["team", "game_date", "game_id"]))
    order["game_no"] = order.groupby("team").cumcount()
    order["block"] = order.game_no // block_games
    tv = tv.merge(order[["team", "game_id", "block"]], on=["team", "game_id"])
    min_games = block_games if drop_partial_min is None else drop_partial_min
    sizes = tv.groupby(["team", "block"]).game_id.nunique()
    keep = sizes[sizes >= min_games].index
    return tv.set_index(["team", "block"]).loc[keep].reset_index()


def _rate(num: pd.Series, den: pd.Series, per: float = 100.0, min_den: float = 1) -> pd.Series:
    return (per * num / den).where(den >= min_den)


def _net(g: pd.DataFrame, mask: pd.Series | None = None, min_poss: int = 1) -> pd.DataFrame:
    """Off/def rating and net rating per block over the masked possessions."""
    d = g if mask is None else g[mask]
    agg = d.groupby(["team", "block", "side"]).agg(pts=("points", "sum"), poss=("points", "size")).unstack("side")
    agg.columns = [f"{a}_{b}" for a, b in agg.columns]
    for c in ("pts_off", "poss_off", "pts_def", "poss_def"):
        if c not in agg:
            agg[c] = 0
    agg = agg.fillna(0)
    off = _rate(agg.pts_off, agg.poss_off, min_den=min_poss)
    de = _rate(agg.pts_def, agg.poss_def, min_den=min_poss)
    return pd.DataFrame({"off_rtg": off, "def_rtg": de, "net_rtg": off - de, "poss_off": agg.poss_off, "poss_def": agg.poss_def})


def sequence_stats_per_game(poss: pd.DataFrame) -> pd.DataFrame:
    """Order-aware per-team-game stats from the possession sequence: largest
    scoring run for/against, lead changes, share of game time spent leading."""
    rows = []
    for gid, g in poss.sort_values(["game_id", "poss_idx"]).groupby("game_id", sort=False):
        home = g.is_home_off.iloc[0] and g.off_team_id.iloc[0] or g.def_team_id.iloc[0]
        teams = {int(g.off_team_id.iloc[0]), int(g.def_team_id.iloc[0])}
        away = (teams - {int(home)}).pop()
        # home-perspective margin path
        margin = np.where(g.is_home_off == 1, g.margin_start, -g.margin_start)
        scored_by = np.where(g.points > 0, np.where(g.is_home_off == 1, 1, -1), 0)
        pts = g.points.to_numpy()
        run, run_team, best = {1: 0, -1: 0}, 0, {1: 0, -1: 0}
        for s, ptv in zip(scored_by, pts):
            if s == 0:
                continue
            if s == run_team:
                run[s] += ptv
            else:
                run_team, run[s] = s, ptv
            best[s] = max(best[s], run[s])
        sign = np.sign(margin)
        nz = sign[sign != 0]
        lead_changes = int((np.diff(nz) != 0).sum()) if len(nz) > 1 else 0
        dur = g.duration.to_numpy()
        total = dur.sum() or 1.0
        home_lead_share = float(dur[margin > 0].sum() / total)
        rows.append({"game_id": gid, "team": int(home), "largest_run_for": best[1], "largest_run_against": best[-1],
                     "lead_changes": lead_changes, "time_leading_share": home_lead_share})
        rows.append({"game_id": gid, "team": away, "largest_run_for": best[-1], "largest_run_against": best[1],
                     "lead_changes": lead_changes, "time_leading_share": float(dur[margin < 0].sum() / total)})
    return pd.DataFrame(rows)


def block_aggregates(tv: pd.DataFrame, seq: pd.DataFrame | None = None,
                     lg_fg3_pct: float | None = None, lg_ft_pct: float | None = None) -> pd.DataFrame:
    """All candidate aggregates per (team, block). `lg_*_pct` default to the
    rates observed in `tv` itself (fine for a screen; a live feature must use
    point-in-time league rates)."""
    off = tv[tv.side == "off"]
    # A zero-attempt denominator would make the league rate NaN and poison every
    # luck-adjusted rating; with no attempts the correction term is zero anyway.
    if lg_fg3_pct is None:
        lg_fg3_pct = off.n_fg3m.sum() / off.n_fg3a.sum() if off.n_fg3a.sum() else 0.0
    if lg_ft_pct is None:
        lg_ft_pct = off.n_ftm.sum() / off.n_fta.sum() if off.n_fta.sum() else 0.0

    out = _net(tv)
    games = tv.groupby(["team", "block"]).game_id.nunique().rename("n_games")
    out = out.join(games)
    pts = tv.groupby(["team", "block", "side"]).points.sum().unstack("side").fillna(0)
    out["margin_pg"] = (pts["off"] - pts["def"]) / out.n_games
    game_secs = tv[tv.side == "off"].groupby(["team", "block"]).duration.sum() + \
        tv[tv.side == "def"].groupby(["team", "block"]).duration.sum()
    out["pace48"] = (out.poss_off + out.poss_def) / 2 / (game_secs / REGULATION_SECS)

    # Garbage-time-filtered
    ng = _net(tv, ~tv.garbage)
    out["net_rtg_nogarbage"] = ng.net_rtg
    out["off_rtg_nogarbage"], out["def_rtg_nogarbage"] = ng.off_rtg, ng.def_rtg

    # Luck-adjusted: replace own and opponent 3P/FT makes with league-rate expectations.
    shots = tv.groupby(["team", "block", "side"])[["n_fg3a", "n_fg3m", "n_fta", "n_ftm"]].sum().unstack("side").fillna(0)
    own_luck = 3 * (shots[("n_fg3m", "off")] - lg_fg3_pct * shots[("n_fg3a", "off")]) + \
        (shots[("n_ftm", "off")] - lg_ft_pct * shots[("n_fta", "off")])
    opp_luck = 3 * (shots[("n_fg3m", "def")] - lg_fg3_pct * shots[("n_fg3a", "def")]) + \
        (shots[("n_ftm", "def")] - lg_ft_pct * shots[("n_fta", "def")])
    out["net_rtg_luckadj"] = out.net_rtg - 100 * own_luck / out.poss_off + 100 * opp_luck / out.poss_def
    out["net_rtg_luckadj_nogarbage"] = ng.net_rtg - 100 * own_luck / out.poss_off + 100 * opp_luck / out.poss_def
    out["own_3p_luck_per100"] = 100 * 3 * (shots[("n_fg3m", "off")] - lg_fg3_pct * shots[("n_fg3a", "off")]) / out.poss_off
    out["opp_fg3_pct"] = shots[("n_fg3m", "def")] / shots[("n_fg3a", "def")]

    # Clutch / non-clutch, deficit / lead contexts
    out["net_rtg_clutch"] = _net(tv, tv.clutch, MIN_CONTEXT_POSS).net_rtg
    out["net_rtg_nonclutch"] = _net(tv, ~tv.clutch & ~tv.garbage).net_rtg
    out["net_rtg_trailing8"] = _net(tv, (tv.team_margin <= -DEFICIT_MARGIN) & ~tv.garbage, MIN_CONTEXT_POSS).net_rtg
    out["net_rtg_leading8"] = _net(tv, (tv.team_margin >= DEFICIT_MARGIN) & ~tv.garbage, MIN_CONTEXT_POSS).net_rtg
    out["net_rtg_close"] = _net(tv, tv.team_margin.abs() < DEFICIT_MARGIN, MIN_CONTEXT_POSS).net_rtg
    out["net_rtg_1h"] = _net(tv, tv.period <= 2).net_rtg
    out["net_rtg_2h"] = _net(tv, (tv.period >= 3) & ~tv.garbage).net_rtg

    # Four-factor style pieces (offense only) and shot mix by last-shot location
    o = off.groupby(["team", "block"])
    fga = o.n_fga.sum()
    out["tov_rate"] = 100 * o.n_tov.sum() / out.poss_off
    out["oreb_per100"] = 100 * o.n_oreb.sum() / out.poss_off
    out["fta_rate"] = o.n_fta.sum() / fga
    shot = off[off.n_fga > 0]
    sg = shot.groupby(["team", "block"])
    out["three_rate"] = o.n_fg3a.sum() / fga
    out["rim_rate"] = sg.apply(lambda d: (d.last_shot_distance <= RIM_FT).mean())
    out["mid_rate"] = sg.apply(lambda d: ((d.last_shot_distance > RIM_FT) & (d.last_shot_value == 2) & (d.last_shot_distance < MID_FT)).mean())
    out["efg"] = (o.n_fgm.sum() + 0.5 * o.n_fg3m.sum()) / fga

    # Lineup continuity (complete lineups only)
    lc = off[off.lineup_complete]
    lg = lc.groupby(["team", "block"])
    out["top_lineup_share"] = lg.lineup_off.agg(lambda s: s.value_counts(normalize=True).iloc[0] if len(s) else np.nan)
    out["top3_lineup_share"] = lg.lineup_off.agg(lambda s: s.value_counts(normalize=True).iloc[:3].sum() if len(s) else np.nan)
    out["lineups_per100"] = 100 * lg.lineup_off.nunique() / lg.size()
    out["lineup_coverage"] = lg.size() / out.poss_off

    # Lineup-conditioned efficiency: how the team plays with its real rotation
    # on the floor, as distinct from how stable that rotation is. Restricted to
    # possessions where both fives are known.
    known = tv[tv.lineup_complete & (tv.own_lineup != "")]
    usage = known.groupby(["team", "block", "own_lineup"]).size().rename("poss")
    rank = usage.groupby(["team", "block"]).rank(method="first", ascending=False).rename("rank")
    ranked = known.merge(rank.reset_index(), on=["team", "block", "own_lineup"])
    top1 = _net(ranked, ranked["rank"] == 1, MIN_CONTEXT_POSS)
    top5 = _net(ranked, ranked["rank"] <= 5, MIN_CONTEXT_POSS)
    rest = _net(ranked, ranked["rank"] > 5, MIN_CONTEXT_POSS)
    out["net_rtg_top1_lineup"] = top1.net_rtg
    out["net_rtg_top5_lineups"] = top5.net_rtg
    out["net_rtg_bench_lineups"] = rest.net_rtg
    out["net_rtg_top5_minus_bench"] = top5.net_rtg - rest.net_rtg

    if seq is not None:
        blk = tv[["team", "game_id", "block"]].drop_duplicates()
        s = seq.merge(blk, on=["team", "game_id"]).groupby(["team", "block"])[
            ["largest_run_for", "largest_run_against", "lead_changes", "time_leading_share"]].mean()
        out = out.join(s)
    return out.reset_index()


def persistence(blocks: pd.DataFrame, stats: list[str], reference: str = "net_rtg",
                target: str = "margin_pg") -> pd.DataFrame:
    """Per stat: `r_self` = pooled Pearson r between a team's value in block b
    and block b+1 (persistence); `r_within` = the same after demeaning within
    team; `r_next_margin` = r between the stat in block b and `target` in
    block b+1 (what a pre-game feature actually needs); `r_vs_ref` = same-block
    r with `reference` (redundancy)."""
    b = blocks.sort_values(["team", "block"]).copy()
    nxt_target = b[["team", "block", target]].assign(block=b.block - 1).rename(columns={target: "_next_target"})
    rows = []
    for s in stats:
        cur = b[["team", "block", s]].dropna()
        nxt = cur.assign(block=cur.block - 1).rename(columns={s: "next"})
        pair = cur.merge(nxt, on=["team", "block"])
        row = {"stat": s, "r_self": np.nan, "r_within": np.nan, "r_next_margin": np.nan,
               "n_pairs": len(pair), "r_vs_ref": np.nan}
        if len(pair) >= 10:
            row["r_self"] = np.corrcoef(pair[s], pair["next"])[0, 1]
            dm = pair.assign(a=pair[s] - pair.groupby("team")[s].transform("mean"),
                             c=pair["next"] - pair.groupby("team")["next"].transform("mean"))
            row["r_within"] = np.corrcoef(dm.a, dm.c)[0, 1]
        pt = cur.merge(nxt_target, on=["team", "block"]).dropna()
        if len(pt) >= 10:
            row["r_next_margin"] = np.corrcoef(pt[s], pt["_next_target"])[0, 1]
        if reference in b.columns and s != reference:
            ref = b[["team", "block", reference, s]].dropna()
            if len(ref) > 10:
                row["r_vs_ref"] = np.corrcoef(ref[reference], ref[s])[0, 1]
        rows.append(row)
    return pd.DataFrame(rows)
