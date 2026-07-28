"""
Season motivation & seeding-incentive computation.

See docs/SEASON_MOTIVATION_DECISIONS.md for the full data audit and the
justification behind every formula below. Two independent pieces:

- `compute_standings_metrics`: point-in-time conference standings, derived
  entirely from the already-complete `game` table (no new backfill -- every
  season this touches is a completed historical season, so its full schedule
  is already sitting in that table). Produces `standings_pressure` plus
  `games_to_clinch_ceiling`/`games_to_clinch_floor` -- these are exposed as
  separate raw feature columns (feature_builder.py's
  `_add_season_motivation_features`), not pre-combined into a single score.
- `compute_roster_behavior_scores`: per (team, game night), how much of a
  team's full-strength quality is sitting out for a non-injury reason (rest,
  personal reasons, coach's decision, ...), reusing the existing
  `player_importance` table and `_get_importance_map`-style weighted formula
  from `src/news_scraping/pipeline.py` rather than inventing a second one.

No tiebreakers (head-to-head, division, conference record) are modeled --
deliberately a continuous proxy, not exact combinatorial seeding logic, per
docs/SEASON_MOTIVATION_DECISIONS.md.
"""

import sqlite3

import numpy as np
import pandas as pd

# Conference assignment is a fixed fact, not a tunable parameter -- no team has
# changed conference across the 2018-2026 window this project covers. Same
# in-module-dict convention as feature_builder.py's _TEAM_LOCATIONS.
_TEAM_CONFERENCE: dict[int, str] = {
    1610612737: "East",  # ATL
    1610612738: "East",  # BOS
    1610612739: "East",  # CLE
    1610612740: "West",  # NOP
    1610612741: "East",  # CHI
    1610612742: "West",  # DAL
    1610612743: "West",  # DEN
    1610612744: "West",  # GSW
    1610612745: "West",  # HOU
    1610612746: "West",  # LAC
    1610612747: "West",  # LAL
    1610612748: "East",  # MIA
    1610612749: "East",  # MIL
    1610612750: "West",  # MIN
    1610612751: "East",  # BKN
    1610612752: "East",  # NYK
    1610612753: "East",  # ORL
    1610612754: "East",  # IND
    1610612755: "East",  # PHI
    1610612756: "West",  # PHX
    1610612757: "West",  # POR
    1610612758: "West",  # SAC
    1610612759: "West",  # SAS
    1610612760: "West",  # OKC
    1610612761: "East",  # TOR
    1610612762: "West",  # UTA
    1610612763: "West",  # MEM
    1610612764: "East",  # WAS
    1610612765: "East",  # DET
    1610612766: "East",  # CHA
}

# Real NBA injury-report reasons that mean "healthy, but sat by team choice" --
# not an actual injury. See docs/SEASON_MOTIVATION_DECISIONS.md sec 1 for the
# full survey of raw `reason` values this was chosen from. Deliberately excludes
# health/compliance reasons (Health and Safety Protocols, Concussion Protocol,
# Suspension, Trade Pending, Ineligible to Play) -- none of those are a team
# resting a healthy, available player for competitive reasons.
NON_INJURY_REASONS: set[str] = {
    "rest",
    "rest-rest",
    "rest - rest",
    "rest - load management",
    "personal reasons",
    "personalreasons",
    "not with team",
    "notwithteam",
    "coach's decision",
    "coach'sdecision",
}


def _build_team_game_log(games_df: pd.DataFrame) -> pd.DataFrame:
    """Long-format per-team-per-game log with pre-game standings state.

    One row per (team, game). `wins_before`/`losses_before`/`games_remaining`
    describe that team's standing entering that game (i.e. counting only
    games strictly before this one) -- `games_remaining` counts tonight's
    game as still-remaining, matching the pre-game feature convention used
    everywhere else in feature_builder.py (rolling stats via shift(1), Elo's
    pre-game rating, on/off-splits' `game_date - 1 day` lookup key).
    """
    home = pd.DataFrame({
        "season_id": games_df["SEASON_ID"].values,
        "game_date": games_df["GAME_DATE"].values,
        "team_id": games_df["HOME_TEAM_ID"].values,
        "win": games_df["HOME_TEAM_WINS"].astype(bool).values,
    })
    away = pd.DataFrame({
        "season_id": games_df["SEASON_ID"].values,
        "game_date": games_df["GAME_DATE"].values,
        "team_id": games_df["AWAY_TEAM_ID"].values,
        "win": ~games_df["HOME_TEAM_WINS"].astype(bool).values,
    })
    team_games = pd.concat([home, away], ignore_index=True)
    team_games["game_date"] = pd.to_datetime(team_games["game_date"]).dt.normalize()
    team_games = team_games.sort_values(["season_id", "team_id", "game_date"]).reset_index(drop=True)

    grp = team_games.groupby(["season_id", "team_id"])
    team_games["games_played_before"] = grp.cumcount()
    team_games["wins_before"] = grp["win"].cumsum() - team_games["win"].astype(int)
    team_games["losses_before"] = team_games["games_played_before"] - team_games["wins_before"]
    team_games["total_games"] = grp["team_id"].transform("count")
    team_games["games_remaining"] = team_games["total_games"] - team_games["games_played_before"]

    # Post-game state -- used to answer "team T's standing as of date D" for
    # teams NOT playing on D themselves (asof-matched against these rows).
    team_games["wins_after"] = team_games["wins_before"] + team_games["win"].astype(int)
    team_games["losses_after"] = team_games["losses_before"] + (~team_games["win"]).astype(int)
    team_games["games_remaining_after"] = team_games["games_remaining"] - 1
    return team_games


def _standings_panel(team_games: pd.DataFrame) -> pd.DataFrame:
    """(season_id, team_id, snapshot_date) -> wins/losses/games_remaining as of
    strictly before snapshot_date, for every date at least one game was played
    that season (standings only change on game dates, so this covers every
    date a lookup could ever need without interpolating).
    """
    panels = []
    for season_id, season_group in team_games.groupby("season_id"):
        snapshot_dates = pd.Series(sorted(season_group["game_date"].unique()))
        lookup_dates = snapshot_dates - pd.Timedelta(days=1)

        for team_id, team_group in season_group.groupby("team_id"):
            team_group = team_group.sort_values("game_date")
            total_games = team_group["total_games"].iloc[0]

            left = pd.DataFrame({"snapshot_date": snapshot_dates, "lookup_date": lookup_dates})
            right = team_group[["game_date", "wins_after", "losses_after", "games_remaining_after"]].rename(
                columns={"game_date": "as_of"}
            )
            merged = pd.merge_asof(
                left, right, left_on="lookup_date", right_on="as_of",
                direction="backward", allow_exact_matches=True,
            )
            merged["wins"] = merged["wins_after"].fillna(0).astype(int)
            merged["losses"] = merged["losses_after"].fillna(0).astype(int)
            merged["games_remaining"] = merged["games_remaining_after"].fillna(total_games).astype(int)
            merged["season_id"] = season_id
            merged["team_id"] = team_id
            panels.append(merged[["season_id", "team_id", "snapshot_date", "wins", "losses", "games_remaining"]])

    return pd.concat(panels, ignore_index=True)


def compute_standings_metrics(games_df: pd.DataFrame, playoff_line_seed: int) -> pd.DataFrame:
    """Returns (season_id, team_id, snapshot_date) -> pressure_raw,
    games_to_clinch_ceiling, games_to_clinch_floor. See
    docs/SEASON_MOTIVATION_DECISIONS.md sections 2a/3 for the formulas.
    """
    team_games = _build_team_game_log(games_df)
    panel = _standings_panel(team_games)
    panel["conference"] = panel["team_id"].map(_TEAM_CONFERENCE)

    panel["win_pct"] = np.where(
        (panel["wins"] + panel["losses"]) > 0,
        panel["wins"] / (panel["wins"] + panel["losses"]),
        0.5,
    )

    grp_cols = ["season_id", "snapshot_date", "conference"]
    panel = panel.sort_values(grp_cols + ["win_pct"], ascending=[True, True, True, False])
    panel["conf_rank"] = panel.groupby(grp_cols).cumcount() + 1

    panel = panel.sort_values(grp_cols + ["conf_rank"]).reset_index(drop=True)
    g = panel.groupby(grp_cols)
    panel["above_wins"] = g["wins"].shift(1)
    panel["below_wins"] = g["wins"].shift(-1)
    panel["below_games_remaining"] = g["games_remaining"].shift(-1)

    line = panel[panel["conf_rank"] == playoff_line_seed][grp_cols + ["wins", "losses"]].rename(
        columns={"wins": "line_wins", "losses": "line_losses"}
    )
    panel = panel.merge(line, on=grp_cols, how="left")

    gb_from_line = ((panel["line_wins"] - panel["wins"]) + (panel["losses"] - panel["line_losses"])) / 2
    panel["pressure_raw"] = (1 - gb_from_line.abs() / (panel["games_remaining"] + 1)).clip(0.0, 1.0)

    max_final_wins = panel["wins"] + panel["games_remaining"]
    min_final_wins = panel["wins"]
    panel["games_to_clinch_ceiling"] = (max_final_wins - panel["above_wins"]).clip(lower=0).fillna(0.0)
    panel["games_to_clinch_floor"] = (
        panel["below_wins"] + panel["below_games_remaining"] - min_final_wins
    ).clip(lower=0).fillna(0.0)

    return panel[[
        "season_id", "team_id", "snapshot_date",
        "pressure_raw", "games_to_clinch_ceiling", "games_to_clinch_floor",
    ]]


def _player_importance_score(rows: pd.DataFrame, weights) -> pd.Series:
    """Same weighted-share formula as src/news_scraping/pipeline.py's
    _get_importance_map (minutes_share/usage_rate/pts_share), reused as-is
    rather than inventing a second player-quality formula. `rows` must have
    one row per player: minutes_per_game, pts_per_game, usage_rate.
    """
    total_minutes = rows["minutes_per_game"].sum() or 1.0
    total_pts = rows["pts_per_game"].sum() or 1.0
    max_usg = rows["usage_rate"].max() or 1.0
    return (
        (rows["minutes_per_game"] / total_minutes) * weights.minutes_share
        + (rows["usage_rate"] / max_usg) * weights.usage_rate
        + (rows["pts_per_game"] / total_pts) * weights.pts_share
    ).clip(0.0, 1.0)


def compute_roster_behavior_scores(
    team_dates: pd.DataFrame,  # columns: team_id, game_date, season_id
    injury_db_path: str,
    importance_weights,
    min_importance_games: int,
    season_start_by_season: dict,
) -> pd.DataFrame:
    """Returns (team_id, game_date) -> roster_behavior_score in [0, 1]: the
    fraction of a team's full-strength quality sitting out for a non-injury
    reason that night. See docs/SEASON_MOTIVATION_DECISIONS.md section 2b.

    0.0 (not NaN) whenever there's no player_importance history yet, or no
    sat-out-healthy players -- both are legitimate "nothing to report" cases,
    not missing-data cases (same convention on/off-splits uses for its own
    zero-impact rows).
    """
    with sqlite3.connect(f"file:{injury_db_path}?mode=ro", uri=True) as conn:
        importance = pd.read_sql_query(
            "SELECT player_id, player_name, team_id, as_of_date, "
            "minutes_per_game, pts_per_game, usage_rate FROM player_importance",
            conn,
        )
        injuries = pd.read_sql_query(
            "SELECT game_date, team_id, player_name, reason FROM player_injuries WHERE status = 'Out'",
            conn,
        )

    importance["as_of_date"] = pd.to_datetime(importance["as_of_date"])
    injuries["game_date"] = pd.to_datetime(injuries["game_date"]).dt.normalize()
    injuries["reason_norm"] = injuries["reason"].fillna("").str.strip().str.lower()
    sat_healthy = injuries[injuries["reason_norm"].isin(NON_INJURY_REASONS)]

    unique_pairs = team_dates.drop_duplicates(subset=["team_id", "game_date"])

    results = []
    for team_id, team_pairs in unique_pairs.groupby("team_id"):
        team_importance = importance[importance["team_id"] == team_id]
        team_sat_healthy = sat_healthy[sat_healthy["team_id"] == team_id]

        for row in team_pairs.itertuples():
            season_start = season_start_by_season.get(row.season_id)
            if season_start is None:
                # Playoff-tagged season_ids (e.g. 42023) show up here as val/test
                # warm-up context (see datasets_loading.context_season_types) --
                # they never reach the scored dataset, and standings/motivation
                # aren't meaningful once a team is already in the playoffs, so a
                # safe 0.0 default (not a crash) is correct here.
                results.append((team_id, row.game_date, 0.0))
                continue
            pool = team_importance[
                (team_importance["as_of_date"] < row.game_date)
                & (team_importance["as_of_date"] >= season_start)
            ]
            if pool.empty:
                results.append((team_id, row.game_date, 0.0))
                continue

            counts = pool.groupby("player_id").size()
            eligible = counts[counts >= min_importance_games].index
            eligible_pool = pool[pool["player_id"].isin(eligible)]
            if eligible_pool.empty:
                results.append((team_id, row.game_date, 0.0))
                continue

            latest = eligible_pool.sort_values("as_of_date").groupby("player_id").tail(1)
            latest = latest.assign(importance=_player_importance_score(latest, importance_weights))
            full_strength_quality = latest["importance"].sum()
            if full_strength_quality <= 0:
                results.append((team_id, row.game_date, 0.0))
                continue

            sat_tonight = team_sat_healthy[team_sat_healthy["game_date"] == row.game_date]
            sat_quality = latest[latest["player_name"].isin(sat_tonight["player_name"])]["importance"].sum()

            score = min(max(sat_quality / full_strength_quality, 0.0), 1.0)
            results.append((team_id, row.game_date, score))

    return pd.DataFrame(results, columns=["team_id", "game_date", "roster_behavior_score"])
