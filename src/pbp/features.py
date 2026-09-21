"""Pre-game rolling feature from the possession table.

Builds the one candidate that survived the persistence screens
(`docs/features/pbp_possessions_log.md`): garbage-time-filtered net rating
with shooting variance partly removed, i.e. three-point and free-throw makes
pulled toward league rates. Own over-performance is removed at half weight,
since a team making more than league rate is partly skill; opponent
over-performance is removed in full, since a team has little control over
whether opponents hit threes against it.

Leakage discipline, the two places it matters:

1. **Rolling window** — every team-game's value is a pooled ratio over that
   team's PREVIOUS `window` games only, via `shift(1)` before the rolling sum.
   The game's own possessions never enter its own feature.
2. **League rates** — the three-point and free-throw baselines are expanding
   means over games strictly BEFORE the game date, not season or full-sample
   constants. A fixed rate fitted on all data would leak future league-wide
   shooting into early-season rows.

Ratios are formed from summed numerators and denominators over the window,
never as a mean of per-game ratios, matching how the screen measured them.
"""

from __future__ import annotations

import sqlite3

import numpy as np
import pandas as pd

REGULATION_SECS = 2880
# Mirrors aggregates.GARBAGE_TIERS; expressed as SQL so the per-possession
# filter runs in the database rather than pulling every row into pandas.
GARBAGE_SQL = f"""
    period <= 4 AND (
        ({REGULATION_SECS} - start_secs <= 720 AND ABS(margin_start) >= 25) OR
        ({REGULATION_SECS} - start_secs <= 360 AND ABS(margin_start) >= 20) OR
        ({REGULATION_SECS} - start_secs <= 180 AND ABS(margin_start) >= 15) OR
        ({REGULATION_SECS} - start_secs <=  60 AND ABS(margin_start) >= 10)
    )
"""
OWN_LUCK_WEIGHT = 0.5
MIN_WINDOW_GAMES = 5  # fewer prior games than this -> NaN, which CatBoost handles natively


def load_team_game_possessions(pbp_db: str, raw_db: str) -> pd.DataFrame:
    """One row per (team, game): points and possessions for and against, plus
    the shooting volumes the luck adjustment needs, garbage time excluded."""
    with sqlite3.connect(f"file:{pbp_db}?mode=ro", uri=True) as conn:
        side = pd.read_sql_query(
            "SELECT game_id, off_team_id, def_team_id, "
            "SUM(points) AS pts, COUNT(*) AS poss, "
            "SUM(n_fg3a) AS fg3a, SUM(n_fg3m) AS fg3m, "
            "SUM(n_fta) AS fta, SUM(n_ftm) AS ftm "
            f"FROM possessions WHERE NOT ({GARBAGE_SQL}) "
            "GROUP BY game_id, off_team_id, def_team_id",
            conn,
        )
    if side.empty:
        return side

    off = side.rename(columns={"off_team_id": "team", "def_team_id": "opp"})
    # The defensive view of a game is the opponent's offensive row, so joining
    # the table to itself on the flipped team pair gives points allowed.
    de = side.rename(columns={
        "def_team_id": "team", "off_team_id": "opp", "pts": "pts_against", "poss": "poss_def",
        "fg3a": "opp_fg3a", "fg3m": "opp_fg3m", "fta": "opp_fta", "ftm": "opp_ftm",
    })
    tg = off.merge(de, on=["game_id", "team", "opp"], how="inner")
    tg = tg.rename(columns={"pts": "pts_for", "poss": "poss_off"})

    with sqlite3.connect(f"file:{raw_db}?mode=ro", uri=True) as conn:
        games = pd.read_sql_query(
            "SELECT game_id, game_date FROM game WHERE season_type = 'Regular Season'", conn
        )
    games["game_date"] = pd.to_datetime(games["game_date"])
    return tg.merge(games, on="game_id").sort_values(["team", "game_date", "game_id"])


def _expanding_league_rates(tg: pd.DataFrame) -> pd.DataFrame:
    """League three-point and free-throw rates from games strictly before each
    date. Shifting the daily cumulative totals is what keeps a game's own
    shooting, and every later game's, out of its own baseline."""
    daily = tg.groupby("game_date")[["fg3a", "fg3m", "fta", "ftm"]].sum().sort_index()
    prior = daily.cumsum().shift(1)
    rates = pd.DataFrame({
        "lg_fg3_pct": prior.fg3m / prior.fg3a,
        "lg_ft_pct": prior.ftm / prior.fta,
    }, index=daily.index)
    # Before any prior game exists the rate is undefined; fall back to the
    # first date that has one so early rows get a sane baseline rather than NaN.
    # A rate that is still undefined means no attempts of that kind have been
    # taken at all, where the correction term is zero by definition — but it
    # must be filled explicitly, since 0 attempts times a NaN rate is NaN and
    # would silently blank the whole feature rather than leaving it uncorrected.
    return rates.bfill().fillna(0.0).reset_index()


def build_pbp_rolling_features(pbp_db: str, raw_db: str, window: int) -> pd.DataFrame:
    """Per (team, game_date) pre-game value of the luck-adjusted, garbage-time
    filtered net rating over that team's previous `window` games."""
    tg = load_team_game_possessions(pbp_db, raw_db)
    if tg.empty:
        return pd.DataFrame(columns=["team", "game_date", "pbp_net_rtg_luckadj"])

    tg = tg.merge(_expanding_league_rates(tg), on="game_date", how="left")

    sum_cols = ["pts_for", "poss_off", "pts_against", "poss_def",
                "fg3a", "fg3m", "fta", "ftm", "opp_fg3a", "opp_fg3m", "opp_fta", "opp_ftm"]
    g = tg.groupby("team", sort=False)
    # shift(1) BEFORE rolling: the window ends at the previous game, so the
    # current game contributes nothing to its own feature.
    rolled = g[sum_cols].apply(lambda d: d.shift(1).rolling(window, min_periods=MIN_WINDOW_GAMES).sum())
    rolled.index = tg.index
    r = pd.concat([tg[["team", "game_id", "game_date", "lg_fg3_pct", "lg_ft_pct"]], rolled], axis=1)

    off_rtg = 100 * r.pts_for / r.poss_off
    def_rtg = 100 * r.pts_against / r.poss_def
    own_luck = 3 * (r.fg3m - r.lg_fg3_pct * r.fg3a) + (r.ftm - r.lg_ft_pct * r.fta)
    opp_luck = 3 * (r.opp_fg3m - r.lg_fg3_pct * r.opp_fg3a) + (r.opp_ftm - r.lg_ft_pct * r.opp_fta)
    r["pbp_net_rtg_luckadj"] = (
        (off_rtg - def_rtg)
        - 100 * OWN_LUCK_WEIGHT * own_luck / r.poss_off
        + 100 * opp_luck / r.poss_def
    )
    r["pbp_net_rtg_luckadj"] = r["pbp_net_rtg_luckadj"].replace([np.inf, -np.inf], np.nan)
    return r[["team", "game_id", "game_date", "pbp_net_rtg_luckadj"]]
