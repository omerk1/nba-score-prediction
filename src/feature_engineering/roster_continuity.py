"""
Roster continuity: how much of last season's rotation is actually playing for
a team now. Covers departures (traded/released/retired players) and
season-long injuries at once -- both leave the team-level history (Elo,
rolling form) describing a roster that no longer takes the floor, and
neither shows up in the injury deficit once the player has no current-season
stats.

Source: `player_importance` weekly snapshots (per-season cumulative per-game
minutes/points, keyed by the player's current team).

For team T in season S and each of T's season-S snapshots:
    prior   = T's top-N players by minutes in T's last season-(S-1) snapshot
    present = players with minutes > 0 in T's snapshot
    return_share = sum(stat of prior players in present) / sum(stat of prior)
for stat in {points per game, minutes per game}.

Point-in-time: a game uses the team's latest same-season snapshot dated
strictly before the game date. The prior rotation comes entirely from the
previous season. Games before the season's first snapshot get NaN.
"""

import numpy as np
import pandas as pd

STATS = {"pts": "pts_per_game", "min": "minutes_per_game"}


def snapshot_season(as_of: pd.Series) -> pd.Series:
    """Season start year of a snapshot date: August onward belongs to the
    season starting that year, January-July to the one that started the year
    before. Holds for every snapshot in the table, including the 2020 bubble
    (no snapshots between 2020-06-30 and 2020-12-24)."""
    return np.where(as_of.dt.month >= 8, as_of.dt.year, as_of.dt.year - 1)


def game_season(season_id: pd.Series) -> pd.Series:
    """Season start year from SEASON_ID (e.g. 22025 regular season, 42025
    playoffs -> 2025)."""
    return season_id.astype(int) % 10000


def build_continuity_table(snapshots: pd.DataFrame, top_n: int) -> pd.DataFrame:
    """One row per (team_id, season, as_of_date) snapshot with
    `return_pts` / `return_min`. `snapshots` needs player_id, team_id,
    as_of_date (datetime), minutes_per_game, pts_per_game."""
    snap = snapshots.copy()
    snap["season"] = snapshot_season(snap["as_of_date"])

    last_date = snap.groupby(["team_id", "season"])["as_of_date"].transform("max")
    final = snap[snap["as_of_date"] == last_date]
    prior = (
        final.sort_values("minutes_per_game", ascending=False)
        .groupby(["team_id", "season"])
        .head(top_n)
        .assign(season=lambda x: x["season"] + 1)  # prior rotation for the NEXT season
    )[["team_id", "season", "player_id", *STATS.values()]]

    totals = prior.groupby(["team_id", "season"])[list(STATS.values())].sum()

    present = snap[snap["minutes_per_game"] > 0][["team_id", "season", "as_of_date", "player_id"]]
    matched = present.merge(prior, on=["team_id", "season", "player_id"])
    returned = matched.groupby(["team_id", "season", "as_of_date"])[list(STATS.values())].sum()

    dates = snap[["team_id", "season", "as_of_date"]].drop_duplicates()
    out = dates.merge(totals.reset_index(), on=["team_id", "season"])  # drops teams with no prior season
    out = out.merge(
        returned.reset_index(), on=["team_id", "season", "as_of_date"], how="left", suffixes=("_tot", "_ret")
    )
    for short, col in STATS.items():
        ret = out[f"{col}_ret"].fillna(0.0)
        tot = out[f"{col}_tot"]
        out[f"return_{short}"] = np.where(tot > 0, ret / tot.where(tot > 0, 1.0), np.nan)
    return out[["team_id", "season", "as_of_date", "return_pts", "return_min"]]


def lookup_for_games(table: pd.DataFrame, games: pd.DataFrame, team_col: str) -> pd.DataFrame:
    """For each row of `games` (GAME_DATE, SEASON_ID, `team_col`), the team's
    latest same-season snapshot strictly before GAME_DATE. Returned in the
    original row order of `games`."""
    query = pd.DataFrame(
        {
            "_row": np.arange(len(games)),
            "GAME_DATE": pd.to_datetime(games["GAME_DATE"]).values,
            "team_id": games[team_col].values,
            "season": game_season(games["SEASON_ID"]).values,
        }
    ).sort_values("GAME_DATE")
    tab = table.rename(columns={"as_of_date": "GAME_DATE"}).sort_values("GAME_DATE")
    tab["team_id"] = tab["team_id"].astype(query["team_id"].dtype)
    tab["season"] = tab["season"].astype(query["season"].dtype)
    merged = pd.merge_asof(
        query,
        tab,
        on="GAME_DATE",
        by=["team_id", "season"],
        allow_exact_matches=False,
        direction="backward",
    )
    return merged.sort_values("_row")[["return_pts", "return_min"]].reset_index(drop=True)
