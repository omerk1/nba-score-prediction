"""Model-free pre-check: does roster continuity (share of last season's top-8
minutes still playing for the team ~3 weeks in) predict how far a team's
margin moves from last season's level, beyond plain regression to the mean?"""

import sys

sys.path.insert(0, ".")
import sqlite3

import numpy as np
import pandas as pd

from src.data_processing.data_loader import NBADataLoader

TOP = 8
SNAP_IDX = 2  # 3rd in-season snapshot (~3 weeks in)

pi = pd.read_sql(
    "select player_id, team_id, as_of_date, minutes_per_game m, pts_per_game p from player_importance",
    sqlite3.connect("data/raw/injury_features.sqlite"),
)
pi["d"] = pd.to_datetime(pi.as_of_date)
pi["season"] = np.where(pi.d.dt.month >= 8, pi.d.dt.year, pi.d.dt.year - 1)

g = NBADataLoader(db_path="data/raw/nba_api.sqlite").load_games(
    start_date="2018-08-01",
    end_date="2026-07-01",
    allowed_season_types=["Regular Season"],
)
L = pd.concat(
    [
        pd.DataFrame({"d": g.GAME_DATE, "team_id": g.HOME_TEAM_ID, "m": g.POINT_DIFF}),
        pd.DataFrame({"d": g.GAME_DATE, "team_id": g.AWAY_TEAM_ID, "m": -g.POINT_DIFF}),
    ]
)
L["season"] = np.where(L.d.dt.month >= 8, L.d.dt.year, L.d.dt.year - 1)

rows = []
for s in range(2019, 2026):
    prev, cur = pi[pi.season == s - 1], pi[pi.season == s]
    if prev.empty or cur.empty:
        continue
    prev_final = prev[prev.d == prev.d.max()]
    snaps = sorted(cur.d.unique())
    if len(snaps) <= SNAP_IDX:
        continue
    snap_d = snaps[SNAP_IDX]
    snap = cur[cur.d == snap_d]
    prev_team_of = prev_final.set_index("player_id").team_id
    prev_mpg = prev_final.set_index("player_id").m
    prev_ppg = prev_final.set_index("player_id").p
    for t in snap.team_id.unique():
        top = prev_final[prev_final.team_id == t].nlargest(TOP, "m")
        now = snap[(snap.team_id == t) & (snap.m > 0)]
        here = top.player_id.isin(now.player_id)
        ret_min = top.m[here].sum() / top.m.sum()
        ret_pts = top.p[here].sum() / top.p.sum()
        # incoming: current top-8 players who were on another team last season
        now_top = now.nlargest(TOP, "m")
        inc = now_top[
            now_top.player_id.map(prev_team_of).fillna(-1).ne(t)
            & now_top.player_id.isin(prev_team_of.index)
        ]
        inc_min = inc.player_id.map(prev_mpg).sum() / max(top.m.sum(), 1e-9)
        inc_pts = inc.player_id.map(prev_ppg).sum() / max(top.p.sum(), 1e-9)
        tl = L[(L.team_id == t)]
        prev_margin = tl[tl.season == s - 1].m.mean()
        after = tl[(tl.season == s) & (tl.d > snap_d)].sort_values("d")
        before = tl[(tl.season == s) & (tl.d < snap_d)]
        rows.append(
            dict(
                season=s,
                team_id=t,
                ret_min=ret_min,
                ret_pts=ret_pts,
                inc_min=inc_min,
                inc_pts=inc_pts,
                prev_margin=prev_margin,
                early=before.m.mean(),
                n_early=len(before),
                next30=after.m.head(30).mean(),
                rest=after.m.mean(),
            )
        )

df = pd.DataFrame(rows).dropna()
print(f"{len(df)} team-seasons, seasons {sorted(df.season.unique())}")
print(df[["ret_min", "ret_pts", "inc_min", "inc_pts"]].describe().round(2).to_string())


def ols(y, cols):
    X = np.c_[np.ones(len(df)), df[cols].values]
    b, *_ = np.linalg.lstsq(X, df[y].values, rcond=None)
    r = df[y].values - X @ b
    sigma2 = r @ r / (len(df) - X.shape[1])
    se = np.sqrt(np.diag(sigma2 * np.linalg.inv(X.T @ X)))
    r2 = 1 - r @ r / ((df[y] - df[y].mean()) ** 2).sum()
    return b, se, r2


for y in ["next30", "rest"]:
    print(
        f"\n== outcome: margin over {'next 30 games' if y == 'next30' else 'rest of season'} after snapshot"
    )
    for cols in (
        ["prev_margin"],
        ["prev_margin", "ret_min"],
        ["prev_margin", "ret_pts"],
        ["prev_margin", "ret_pts", "inc_pts"],
        ["prev_margin", "early"],
        ["prev_margin", "early", "ret_pts"],
        ["prev_margin", "early", "ret_pts", "inc_pts"],
    ):
        b, se, r2 = ols(y, cols)
        terms = "  ".join(
            f"{c}={b[i+1]:+.2f} (t={b[i+1]/se[i+1]:+.1f})" for i, c in enumerate(cols)
        )
        print(f"  R2={r2:.3f}  {terms}")
    # does continuity moderate how much last season carries over?
    df["prev_x_ret"] = df.prev_margin * df.ret_pts
    b, se, r2 = ols(y, ["prev_margin", "ret_pts", "prev_x_ret"])
    print(
        f"  R2={r2:.3f}  interaction prev_margin*ret_pts={b[3]:+.2f} (t={b[3]/se[3]:+.1f})"
    )

print("\nper-season corr(ret_pts, next30 - prev_margin):")
print(
    df.groupby("season")
    .apply(lambda x: x.ret_pts.corr(x.next30 - x.prev_margin))
    .round(2)
    .to_string()
)
