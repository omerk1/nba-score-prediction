"""
Season motivation & seeding-incentive computation.

See docs/SEASON_MOTIVATION_DECISIONS.md for formulas and
docs/SEASON_MOTIVATION_LOG.md for CV results / adoption decisions.

Standings/roster inputs (not adopted -- log FINAL SUMMARY):
- `compute_standings_metrics`: point-in-time conference standings from the
  already-complete `game` table. Feeds `motivation_score`'s pressure term
  plus `games_to_clinch_ceiling`/`games_to_clinch_floor`.
- `compute_roster_behavior_scores`: per (team, game night), how much
  full-strength quality is sitting out for a non-injury reason, reusing
  `_get_importance_map`'s weighted formula from `src/news_scraping/pipeline.py`.
- `compute_recent_minutes_trend_scores`: per (team, game night), genuine
  multi-week minutes reduction -- catches "soft" tanking the single-night
  snapshot above cannot.

Behavior-based signals (log section 10, not adopted -- passed CV at one
window value, inverted at neighboring ones):
- `compute_performance_vs_expectation_scores`: rolling (actual margin -
  Elo-implied expected margin).
- `compute_opponent_adjusted_form_scores`: rolling opponent-strength-weighted
  record.

Seeding-target signal (log section 11, adopted):
- `compute_preferred_opponent_delta_scores`: how much a team's Round 1
  opponent would change in strength if its own seed shifted by one spot --
  passed CV *and* held up across all three tested windows, unlike the two
  behavior-based signals above.

No tiebreakers modeled (head-to-head, division, conference record) --
deliberate continuous proxy, per docs/SEASON_MOTIVATION_DECISIONS.md.
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


def _ranked_standings_panel(games_df: pd.DataFrame) -> pd.DataFrame:
    """(season_id, team_id, snapshot_date) -> wins, losses, games_remaining,
    conference, win_pct, conf_rank. Shared standings computation underneath
    both `compute_standings_metrics` and `compute_preferred_opponent_delta_scores`
    -- `conf_rank` is each team's rank within its own conference on that date
    (1 = best record), broken by `win_pct` only, same as both callers rely on.
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
    return panel.sort_values(grp_cols + ["conf_rank"]).reset_index(drop=True)


def _pressure_from_seed(panel: pd.DataFrame, grp_cols: list[str], seed: int) -> pd.Series:
    """clip(1 - |GB_from_seed| / (games_remaining + 1), 0, 1) against whichever
    team currently holds `seed` in each (season_id, snapshot_date, conference)
    group. Standard games-back arithmetic: positive when trailing the seed,
    negative when leading it, zero at the seed itself."""
    line = panel[panel["conf_rank"] == seed][grp_cols + ["wins", "losses"]].rename(
        columns={"wins": "line_wins", "losses": "line_losses"}
    )
    merged = panel.merge(line, on=grp_cols, how="left")
    gb_from_line = ((merged["line_wins"] - merged["wins"]) + (merged["losses"] - merged["line_losses"])) / 2
    return (1 - gb_from_line.abs() / (panel["games_remaining"] + 1)).clip(0.0, 1.0)


def compute_standings_metrics(games_df: pd.DataFrame, playoff_line_seed: int, direct_playoff_seed: int = None,
                               direct_playoff_weight: float = 0.5) -> pd.DataFrame:
    """Returns (season_id, team_id, snapshot_date) -> pressure_raw,
    games_to_clinch_ceiling, games_to_clinch_floor. See
    docs/SEASON_MOTIVATION_DECISIONS.md sections 2a/3 for the formulas.

    `direct_playoff_seed` (optional): weighted average (`direct_playoff_weight`)
    of pressure vs. `playoff_line_seed` (postseason cutoff) and vs. this seed
    (direct-berth cutoff, e.g. 6th) -- so a team clear of missing the
    postseason but fighting for a direct berth picks up pressure a
    single-threshold formula misses. `null` = single-threshold (unchanged
    behavior). A `max()`-based combination was tried first and found to bias
    pressure upward while compressing variance (log section 6.1); weighted
    average fixed that but is still not adopted.
    """
    panel = _ranked_standings_panel(games_df)
    grp_cols = ["season_id", "snapshot_date", "conference"]
    g = panel.groupby(grp_cols)
    panel["above_wins"] = g["wins"].shift(1)
    panel["below_wins"] = g["wins"].shift(-1)
    panel["below_games_remaining"] = g["games_remaining"].shift(-1)

    if direct_playoff_seed is None:
        panel["pressure_raw"] = _pressure_from_seed(panel, grp_cols, playoff_line_seed)
    else:
        pressure_postseason = _pressure_from_seed(panel, grp_cols, playoff_line_seed)
        pressure_direct = _pressure_from_seed(panel, grp_cols, direct_playoff_seed)
        panel["pressure_raw"] = (
            direct_playoff_weight * pressure_direct + (1 - direct_playoff_weight) * pressure_postseason
        )

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
                # Playoff-tagged season_ids (e.g. 42023) are val/test warm-up
                # context only (never scored) -- safe 0.0 default, not a crash.
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


def compute_recent_minutes_trend_scores(
    team_dates: pd.DataFrame,  # columns: team_id, game_date, season_id
    injury_db_path: str,
    importance_weights,
    min_importance_games: int,
    season_start_by_season: dict,
    lookback_weeks: int = 4,
) -> pd.DataFrame:
    """Returns (team_id, game_date) -> recent_minutes_trend_score in [0, 1]:
    how much of a team's full-strength quality has seen a meaningful minutes
    REDUCTION over the last `lookback_weeks` weeks, vs. each rostered
    player's own cumulative average from before that window.

    Complementary to `roster_behavior_score` -- that one is a single-night
    snapshot and misses "soft" tanking (minutes quietly cut over several
    games without an official injury-report tag). See log section 6.2.

    `player_importance` stores CUMULATIVE per-game averages with no
    games-played column, so an exact "this week's minutes" isn't directly
    computable -- instead compares current cumulative average against the
    snapshot from `lookback_weeks` earlier; a real drop still signals
    genuinely reduced recent minutes without isolating one exact week.

    0.0 (not NaN) when there's no history or no prior-enough snapshot to
    compare -- legitimate "nothing to report," same convention
    `compute_roster_behavior_scores` uses.
    """
    with sqlite3.connect(f"file:{injury_db_path}?mode=ro", uri=True) as conn:
        importance = pd.read_sql_query(
            "SELECT player_id, player_name, team_id, as_of_date, "
            "minutes_per_game, pts_per_game, usage_rate FROM player_importance",
            conn,
        )
    importance["as_of_date"] = pd.to_datetime(importance["as_of_date"])

    unique_pairs = team_dates.drop_duplicates(subset=["team_id", "game_date"])
    lookback = pd.Timedelta(weeks=lookback_weeks)

    results = []
    for team_id, team_pairs in unique_pairs.groupby("team_id"):
        team_importance = importance[importance["team_id"] == team_id]

        for row in team_pairs.itertuples():
            season_start = season_start_by_season.get(row.season_id)
            if season_start is None:
                # Playoff-tagged season_ids show up as val/test warm-up context
                # (see compute_roster_behavior_scores' identical guard) -- never
                # reach the scored dataset, safe 0.0 default.
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

            current = eligible_pool.sort_values("as_of_date").groupby("player_id").tail(1)
            current = current.assign(importance=_player_importance_score(current, importance_weights))
            full_strength_quality = current["importance"].sum()
            if full_strength_quality <= 0:
                results.append((team_id, row.game_date, 0.0))
                continue

            prior_cutoff = row.game_date - lookback
            prior_pool = eligible_pool[eligible_pool["as_of_date"] <= prior_cutoff]
            if prior_pool.empty:
                results.append((team_id, row.game_date, 0.0))
                continue
            prior = prior_pool.sort_values("as_of_date").groupby("player_id").tail(1)

            merged = current[["player_id", "importance", "minutes_per_game"]].merge(
                prior[["player_id", "minutes_per_game"]].rename(columns={"minutes_per_game": "prior_minutes"}),
                on="player_id", how="inner",  # only players with a prior-enough snapshot to compare against
            )
            if merged.empty:
                results.append((team_id, row.game_date, 0.0))
                continue

            drop = np.where(
                merged["prior_minutes"] > 0,
                ((merged["prior_minutes"] - merged["minutes_per_game"]) / merged["prior_minutes"]).clip(0.0, 1.0),
                0.0,
            )
            weighted_drop = (merged["importance"] * drop).sum()

            score = min(max(weighted_drop / full_strength_quality, 0.0), 1.0)
            results.append((team_id, row.game_date, score))

    return pd.DataFrame(results, columns=["team_id", "game_date", "recent_minutes_trend_score"])


def _fit_elo_margin_scale(games_df: pd.DataFrame, elo_ratings: pd.DataFrame, home_advantage: float) -> float:
    """Least-squares slope (through the origin) of actual point margin on
    Elo rating gap, fit once from the full historical game set. Elo's own
    formula converts a rating gap into a win probability, not a point
    margin -- no universal "N Elo points per point of margin" constant
    exists, and this repo's Elo params are independently tuned (`tune_elo.py`),
    so a borrowed external constant wouldn't match this repo's own scale.
    """
    merged = games_df.merge(elo_ratings, on="GAME_ID", how="inner")
    elo_diff = merged["home_team_elo"] + home_advantage - merged["away_team_elo"]
    denom = (elo_diff ** 2).sum()
    if denom <= 0:
        return 0.0
    return float((elo_diff * merged["POINT_DIFF"]).sum() / denom)


def compute_team_performance_history(
    games_df: pd.DataFrame, elo_ratings: pd.DataFrame, home_advantage: float, elo_margin_scale: float,
) -> pd.DataFrame:
    """Long-format per-team-per-game log of actual vs. Elo-expected margin
    and opponent-adjusted outcome, from each team's own perspective -- shared
    input for `compute_performance_vs_expectation_scores` and
    `compute_opponent_adjusted_form_scores`.

    Per (team, game): `actual_margin` (signed), `expected_margin`
    (`elo_diff * elo_margin_scale`), `performance_residual` (actual -
    expected), `win`, `opponent_win_pct` (opponent's own cumulative win% pre-game,
    leakage-safe), `signed_opponent_adjusted_score` (`opponent_win_pct` for a
    win, `-(1 - opponent_win_pct)` for a loss).
    """
    merged = games_df.merge(elo_ratings, on="GAME_ID", how="left")
    home = pd.DataFrame({
        "game_id": merged["GAME_ID"].values,
        "game_date": merged["GAME_DATE"].values,
        "team_id": merged["HOME_TEAM_ID"].values,
        "opponent_id": merged["AWAY_TEAM_ID"].values,
        "actual_margin": merged["POINT_DIFF"].values,
        "elo_diff": (merged["home_team_elo"] + home_advantage - merged["away_team_elo"]).values,
        "win": (merged["POINT_DIFF"] > 0).values,
    })
    away = pd.DataFrame({
        "game_id": merged["GAME_ID"].values,
        "game_date": merged["GAME_DATE"].values,
        "team_id": merged["AWAY_TEAM_ID"].values,
        "opponent_id": merged["HOME_TEAM_ID"].values,
        "actual_margin": (-merged["POINT_DIFF"]).values,
        "elo_diff": (merged["away_team_elo"] - (merged["home_team_elo"] + home_advantage)).values,
        "win": (merged["POINT_DIFF"] < 0).values,
    })
    team_games = pd.concat([home, away], ignore_index=True)
    team_games["game_date"] = pd.to_datetime(team_games["game_date"]).dt.normalize()
    team_games["expected_margin"] = team_games["elo_diff"] * elo_margin_scale
    team_games["performance_residual"] = team_games["actual_margin"] - team_games["expected_margin"]

    team_games = team_games.sort_values(["team_id", "game_date"]).reset_index(drop=True)
    grp = team_games.groupby("team_id")
    games_played_before = grp.cumcount()
    wins_before = grp["win"].cumsum() - team_games["win"].astype(int)
    team_games["win_pct_before"] = np.where(games_played_before > 0, wins_before / games_played_before, 0.5)

    opponent_pct = team_games[["game_id", "team_id", "win_pct_before"]].rename(
        columns={"team_id": "opponent_id", "win_pct_before": "opponent_win_pct"}
    )
    team_games = team_games.merge(opponent_pct, on=["game_id", "opponent_id"], how="left")

    team_games["signed_opponent_adjusted_score"] = np.where(
        team_games["win"], team_games["opponent_win_pct"], -(1 - team_games["opponent_win_pct"]),
    )
    return team_games


def compute_performance_vs_expectation_scores(team_games: pd.DataFrame, window: int) -> pd.DataFrame:
    """Returns (team_id, game_date) -> performance_vs_expectation_score:
    rolling mean of (actual_margin - Elo-expected_margin) over the previous
    `window` games (`shift(1)`, excludes tonight's own result -- same
    pre-game convention `_add_rolling_features` uses), normalized by the
    residual's global std so the score is roughly scale-free.
    """
    team_games = team_games.sort_values(["team_id", "game_date"])
    rolling_mean = team_games.groupby("team_id")["performance_residual"].transform(
        lambda x: x.shift(1).rolling(window, min_periods=1).mean()
    )
    residual_std = team_games["performance_residual"].std() or 1.0
    score = (rolling_mean / residual_std).fillna(0.0)
    return pd.DataFrame({
        "team_id": team_games["team_id"].values,
        "game_date": team_games["game_date"].values,
        "performance_vs_expectation_score": score.values,
    })


def compute_opponent_adjusted_form_scores(team_games: pd.DataFrame, window: int) -> pd.DataFrame:
    """Returns (team_id, game_date) -> opponent_adjusted_form_score: the
    rolling mean of `signed_opponent_adjusted_score` over each team's
    previous `window` games (`shift(1)`, same pre-game convention as above).
    High = recently winning the games a motivated team should win (and
    beating good teams); low/negative = recently losing games a motivated
    team shouldn't (especially to weak opponents)."""
    team_games = team_games.sort_values(["team_id", "game_date"])
    rolling_mean = team_games.groupby("team_id")["signed_opponent_adjusted_score"].transform(
        lambda x: x.shift(1).rolling(window, min_periods=1).mean()
    )
    return pd.DataFrame({
        "team_id": team_games["team_id"].values,
        "game_date": team_games["game_date"].values,
        "opponent_adjusted_form_score": rolling_mean.fillna(0.0).values,
    })


def compute_preferred_opponent_delta_scores(games_df: pd.DataFrame, games_remaining_window: int) -> pd.DataFrame:
    """Returns (season_id, team_id, snapshot_date) -> preferred_opponent_delta.

    The standard NBA bracket pairs conference seed s against seed (9 - s) in
    Round 1. Measures how much that Round 1 opponent's `win_pct` would
    change if the team's OWN seed shifted by one spot (up or down,
    whichever is the larger swing) -- a team can be better off NOT chasing
    the best seed it can reach, since a one-seed swing can swap in a
    materially easier or harder opponent than "higher seed = better"
    suggests. Distinct from `motivation_score`, which only cares about
    distance from a cutoff, not which specific opponent is on the other side.

    Sign: positive = the available move faces a STRONGER opponent (current
    draw already favorable, no incentive to jockey). Negative = a WEAKER
    opponent (real incentive to shift one seed).

    0.0 unless holding a direct playoff seed (conf_rank 1-8) within
    `games_remaining_window` of season's end -- a late-season phenomenon,
    same reasoning `compute_recent_minutes_trend_scores` uses for its
    lookback bound.

    Known limitations (continuous proxy, not exact combinatorial seeding
    logic, per docs/SEASON_MOTIVATION_DECISIONS.md): only a single seed
    step considered (no 2+-seed jumps); the adjacent seed's occupant is
    read off the current snapshot, not resimulated (no full conference
    picture).
    """
    panel = _ranked_standings_panel(games_df)
    join_cols = ["season_id", "snapshot_date", "conference"]

    seed_win_pct = panel[join_cols + ["conf_rank", "win_pct"]].rename(
        columns={"conf_rank": "seed", "win_pct": "seed_win_pct"}
    )

    def _opponent_win_pct_at_seed(hypothetical_seed: pd.Series) -> pd.Series:
        # hypothetical_seed itself must be a real direct-playoff seed (1-8)
        # for "9 - hypothetical_seed" to mean anything -- e.g. hypothetical
        # seed 0 (a seed-1 team's invalid "one better") maps to 9 - 0 = 9,
        # and seed 9 IS a real team (just outside the playoff bracket), so
        # without this guard the lookup would silently return a real but
        # meaningless value instead of correctly invalidating that direction.
        lookup = panel[join_cols].copy()
        lookup["seed"] = 9 - hypothetical_seed
        merged = lookup.merge(seed_win_pct, on=join_cols + ["seed"], how="left")
        return merged["seed_win_pct"].where(hypothetical_seed.between(1, 8).values)

    current_opponent_win_pct = _opponent_win_pct_at_seed(panel["conf_rank"])
    opponent_win_pct_up = _opponent_win_pct_at_seed(panel["conf_rank"] - 1)
    opponent_win_pct_down = _opponent_win_pct_at_seed(panel["conf_rank"] + 1)

    # NaN (no team at that hypothetical seed, e.g. seed 0 or seed 9 from the
    # edge of the 1-8 range) means that direction's swing isn't available --
    # 0.0 correctly keeps it from ever being picked as the larger-magnitude one.
    delta_up = (opponent_win_pct_up - current_opponent_win_pct).fillna(0.0)
    delta_down = (opponent_win_pct_down - current_opponent_win_pct).fillna(0.0)
    use_up = delta_up.abs() >= delta_down.abs()
    delta = np.where(use_up, delta_up, delta_down)

    in_playoff_seed = panel["conf_rank"].between(1, 8)
    late_season = panel["games_remaining"] <= games_remaining_window
    delta = np.where(in_playoff_seed & late_season, delta, 0.0)

    return pd.DataFrame({
        "season_id": panel["season_id"].values,
        "team_id": panel["team_id"].values,
        "snapshot_date": panel["snapshot_date"].values,
        "preferred_opponent_delta": delta,
    })
