"""
Point-in-time context for one uncertain injury listing.

For a listing published in the report dated D about the game on G = D + 1, every
retrieved fact must be knowable at report time. The bound used everywhere here
is `game_date < D` (strictly before the report date): games played on D itself
may still be in progress when the 11PM report is published, so they are
excluded rather than argued about. As-of tables (`player_importance`) use the
latest snapshot dated before D.

The context is produced as one flat numeric row per listing (input to the
tabular baselines) and can be rendered into a prompt later; both views come
from the same frame so the LLM and the baseline see exactly the same facts.
"""

import logging
import re
import sqlite3

import numpy as np
import pandas as pd

from src.news_scraping.extractors.formula_scorer import classify_severity
from src.utils.config_loader import load_config

logger = logging.getLogger(__name__)

RAW_DB_DEFAULT = "data/raw/nba_api.sqlite"

_ILLNESS_RE = re.compile(
    r"flu|covid|protocol|virus|gastro|;\s*illness|n/?a;?\s*illness", re.I
)  # not the "Injury/Illness" prefix
_MAINTENANCE_RE = re.compile(r"management|maintenance|reconditioning|rest|conditioning", re.I)
_PERSONAL_RE = re.compile(r"personal|birth|family|suspension|not with team", re.I)
_BODY_RE = re.compile(
    r"(ankle|knee|hamstring|back|hip|calf|foot|groin|quad|shoulder|wrist|hand|finger|thumb|"
    r"elbow|achilles|thigh|toe|neck|rib|adductor|concussion|head|eye|nose|jaw|heel|shin|leg)",
    re.I,
)

CONTEXT_COLUMNS = [
    "is_doubtful",
    "severity",
    "is_illness",
    "is_maintenance",
    "is_personal",
    "reason_empty",
    "own_n_prior_uncertain",
    "own_prior_play_rate",
    "own_n_same_reason",
    "own_same_reason_play_rate",
    "listed_streak_days",
    "last_status_out",
    "days_since_last_played",
    "min_last10_mean",
    "min_last_game",
    "team_games_missed_last10",
    "season_games_played_share",
    "imp_minutes_per_game",
    "imp_pts_per_game",
    "imp_usage_rate",
    "is_home",
    "rest_days",
    "back_to_back",
    "team_win_pct_std",
    "opp_win_pct_std",
    "days_into_season",
]


def normalize_reason(reason: str) -> str:
    """Collapse PDF spacing/punctuation variants: 'Injury/Illness-RightAnkle;Sprain' ==
    'Injury/Illness - Right Ankle; Sprain'."""
    return re.sub(r"[^a-z]", "", (reason or "").lower())


def _reason_flags(df: pd.DataFrame, severity_weights) -> pd.DataFrame:
    r = df["reason"].fillna("")
    return pd.DataFrame(
        {
            "is_doubtful": (df["status"] == "Doubtful").astype(int),
            "severity": r.map(lambda x: classify_severity(x, severity_weights)),
            "is_illness": r.str.contains(_ILLNESS_RE).astype(int),
            "is_maintenance": r.str.contains(_MAINTENANCE_RE).astype(int),
            "is_personal": r.str.contains(_PERSONAL_RE).astype(int),
            "reason_empty": (r.str.strip() == "").astype(int),
            "body_part": r.str.extract(_BODY_RE, expand=False).fillna("").str.lower(),
        },
        index=df.index,
    )


def load_games(raw_db: str = RAW_DB_DEFAULT) -> pd.DataFrame:
    with sqlite3.connect(raw_db) as conn:
        g = pd.read_sql_query(
            "SELECT game_id, substr(game_date,1,10) AS game_date, team_id_home, team_id_away, pts_home, pts_away "
            "FROM game WHERE game_date >= '2021-06-01'",
            conn,
        )
    return g


def load_importance(injury_db: str) -> pd.DataFrame:
    with sqlite3.connect(injury_db) as conn:
        return pd.read_sql_query(
            "SELECT player_id, as_of_date, minutes_per_game, pts_per_game, usage_rate FROM player_importance",
            conn,
        )


def _team_game_frame(games: pd.DataFrame) -> pd.DataFrame:
    """One row per (team, game): venue, opponent, result, season-to-date win pct BEFORE the game."""
    home = games.rename(columns={"team_id_home": "team_id", "team_id_away": "opp_id"}).assign(
        is_home=1, won=(games["pts_home"] > games["pts_away"]).astype(int)
    )
    away = games.rename(columns={"team_id_away": "team_id", "team_id_home": "opp_id"}).assign(
        is_home=0, won=(games["pts_away"] > games["pts_home"]).astype(int)
    )
    tg = pd.concat([home, away])[["game_id", "game_date", "team_id", "opp_id", "is_home", "won"]]
    tg["season"] = tg["game_date"].map(lambda d: int(d[:4]) if int(d[5:7]) >= 10 else int(d[:4]) - 1)
    tg = tg.sort_values(["team_id", "game_date"]).reset_index(drop=True)
    grp = tg.groupby(["team_id", "season"])
    tg["wins_before"] = grp["won"].cumsum() - tg["won"]
    tg["games_before"] = grp.cumcount()
    tg["win_pct_std"] = np.where(
        tg["games_before"] > 0, tg["wins_before"] / tg["games_before"].clip(lower=1), 0.5
    )
    prev_date = grp["game_date"].shift(1)
    tg["rest_days"] = (pd.to_datetime(tg["game_date"]) - pd.to_datetime(prev_date)).dt.days
    tg["rest_days"] = tg["rest_days"].fillna(7).clip(upper=7)
    tg["back_to_back"] = (tg["rest_days"] == 1).astype(int)
    season_start = grp["game_date"].transform("min")
    tg["days_into_season"] = (pd.to_datetime(tg["game_date"]) - pd.to_datetime(season_start)).dt.days
    return tg


def _player_history_features(queries: pd.DataFrame, history: pd.DataFrame) -> pd.DataFrame:
    """Own prior listings (any status) strictly before the report date."""
    hist = history.sort_values("report_date")
    hist = hist.assign(reason_key=hist["reason"].map(normalize_reason))
    by_player = {pid: g for pid, g in hist.groupby("player_id")}
    rows = []
    for q in queries.itertuples(index=False):
        h = by_player.get(q.player_id)
        if h is not None:
            h = h[h["report_date"] < q.report_date]
        if h is None or h.empty:
            rows.append((0, np.nan, 0, np.nan, 0, 0))
            continue
        unc = h[h["status"].isin(("Questionable", "Doubtful"))]
        same = unc[unc["reason_key"] == normalize_reason(q.reason)]
        # consecutive prior report dates on which the player was listed (any status)
        streak, d = 0, pd.Timestamp(q.report_date)
        listed = set(pd.to_datetime(h["report_date"]))
        while (d - pd.Timedelta(days=streak + 1)) in listed:
            streak += 1
        last_out = int(h.iloc[-1]["status"] == "Out")
        rows.append(
            (
                len(unc),
                unc["played"].mean() if len(unc) else np.nan,
                len(same),
                same["played"].mean() if len(same) else np.nan,
                streak,
                last_out,
            )
        )
    cols = [
        "own_n_prior_uncertain",
        "own_prior_play_rate",
        "own_n_same_reason",
        "own_same_reason_play_rate",
        "listed_streak_days",
        "last_status_out",
    ]
    return pd.DataFrame(rows, columns=cols, index=queries.index)


def _minutes_features(
    queries: pd.DataFrame, game_log: pd.DataFrame, team_games: pd.DataFrame
) -> pd.DataFrame:
    """Recent minutes and games missed, from games strictly before the report date."""
    gl = game_log.sort_values("game_date")
    by_player = {pid: g for pid, g in gl.groupby("player_id")}
    tg_by_team = {tid: g for tid, g in team_games.sort_values("game_date").groupby("team_id")}
    rows = []
    for q in queries.itertuples(index=False):
        p = by_player.get(q.player_id)
        p = p[(p["game_date"] < q.report_date) & (p["season"] == q.season)] if p is not None else None
        t = tg_by_team.get(q.team_id)
        t = (
            t[(t["game_date"] < q.report_date) & (t["game_date"] >= q.season[:4] + "-06-01")]
            if t is not None
            else None
        )
        n_team = len(t) if t is not None else 0
        if p is None or p.empty:
            missed10 = min(n_team, 10)
            rows.append((np.nan, np.nan, np.nan, missed10, 0.0))
            continue
        last10 = p.tail(10)
        last_played = pd.Timestamp(p.iloc[-1]["game_date"])
        days_since = (pd.Timestamp(q.report_date) - last_played).days
        played_dates = set(p["game_date"])
        recent_team_dates = list(t["game_date"].tail(10)) if t is not None else []
        missed10 = sum(1 for d in recent_team_dates if d not in played_dates)
        share = min(len(p) / n_team, 1.0) if n_team else 0.0  # >1 only for mid-season trades
        rows.append((last10["minutes"].mean(), p.iloc[-1]["minutes"], days_since, missed10, share))
    cols = [
        "min_last10_mean",
        "min_last_game",
        "days_since_last_played",
        "team_games_missed_last10",
        "season_games_played_share",
    ]
    return pd.DataFrame(rows, columns=cols, index=queries.index)


def _importance_features(queries: pd.DataFrame, importance: pd.DataFrame) -> pd.DataFrame:
    imp = importance.sort_values("as_of_date")
    imp["as_of_ts"] = pd.to_datetime(imp["as_of_date"])
    q = queries[["player_id", "report_date"]].copy()
    q["report_ts"] = pd.to_datetime(q["report_date"]) - pd.Timedelta(days=1)  # snapshot strictly before D
    q["_order"] = np.arange(len(q))
    merged = pd.merge_asof(
        q.sort_values("report_ts"),
        imp.sort_values("as_of_ts"),
        left_on="report_ts",
        right_on="as_of_ts",
        by="player_id",
        direction="backward",
    ).sort_values("_order")
    out = merged[["minutes_per_game", "pts_per_game", "usage_rate"]]
    out.columns = ["imp_minutes_per_game", "imp_pts_per_game", "imp_usage_rate"]
    out.index = queries.index
    return out


def _game_context(queries: pd.DataFrame, team_games: pd.DataFrame) -> pd.DataFrame:
    tg = team_games[
        [
            "game_id",
            "team_id",
            "opp_id",
            "is_home",
            "rest_days",
            "back_to_back",
            "win_pct_std",
            "days_into_season",
        ]
    ]
    opp = team_games[["game_id", "team_id", "win_pct_std"]].rename(
        columns={"team_id": "opp_id", "win_pct_std": "opp_win_pct_std"}
    )
    ctx = queries[["game_id", "team_id"]].merge(tg, on=["game_id", "team_id"], how="left")
    ctx = ctx.merge(opp, on=["game_id", "opp_id"], how="left")
    ctx.index = queries.index
    return ctx[
        ["is_home", "rest_days", "back_to_back", "win_pct_std", "opp_win_pct_std", "days_into_season"]
    ].rename(columns={"win_pct_std": "team_win_pct_std"})


def build_context(
    queries: pd.DataFrame,
    history: pd.DataFrame,
    game_log: pd.DataFrame,
    importance: pd.DataFrame,
    games: pd.DataFrame,
) -> pd.DataFrame:
    """Return `queries` with CONTEXT_COLUMNS (+ `body_part`) appended. All facts
    are bounded to strictly before each row's report_date."""
    cfg = load_config()
    team_games = _team_game_frame(games)
    parts = [
        queries,
        _reason_flags(queries, cfg.injury_features.severity_weights),
        _player_history_features(queries, history),
        _minutes_features(queries, game_log, team_games),
        _importance_features(queries, importance),
        _game_context(queries, team_games),
    ]
    out = pd.concat(parts, axis=1)
    missing = [c for c in CONTEXT_COLUMNS if c not in out.columns]
    if missing:
        raise RuntimeError(f"context build missing columns: {missing}")
    return out
