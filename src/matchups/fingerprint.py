"""
Layer 1: Team Style Fingerprint (Encoding Phase 1 — hand-picked metrics).

Rolling pre-game fingerprint per team per game, built strictly from games
BEFORE the game date (no leakage), using exponential decay (half-life =
style_matchup.decay_halflife games) over the trailing style_matchup.fingerprint_window
games.

Metrics (per design doc):
    pace_score        = PTS + OPP_PTS + TOV - FTA*0.44          (per-game proxy, then decay-avg)
    three_pt_reliance = FG3A / FGA
    paint_activity    = FTA
    defensive_rating  = OPP_PTS / possessions * 100
    assist_rate       = AST / FGM

possessions is estimated with the standard box-score formula:
    possessions = FGA - OREB + TOV + 0.44*FTA

Added by the raw-fingerprint feature redesign -- offensive_rating (PTS / possessions
* 100, same possessions estimate used for defensive_rating, just the team's own
scoring instead of the opponent's). Motivation: 4 of the original 5 metrics are
volume/style metrics (pace_score, three_pt_reliance, paint_activity, assist_rate)
that describe *what* a team tends to do, not *how well* -- e.g. three_pt_reliance
(3PA/FGA) can't tell a 40%-volume team shooting 40% from three apart from one
shooting 28%. defensive_rating was already the one exception (captures quality, not
just emphasis); offensive_rating adds the missing offensive-quality counterpart.

Deliberate scope cut: this is a Layer 1 (raw, uncalibrated) metric only. No
injury-adjustment delta is calibrated for it -- doing so properly would require a
full Phase-0-style calibration pass (per-archetype grid search across halflifes,
sign-flip checks, etc., see docs/a7_phase_log.md's Initial Build / Critique &
Bug-Fix Pass / Injury-Calibration Decay Fix sections) which was out of scope for
this redesign. Candidate for a future pass if the raw-features approach shows
promise. Layer 2 (injury_layer.py)
therefore leaves offensive_rating untouched (no injury_impact config entry
references it), so its layer=2 value is identical to its layer=1 value --
FeatureBuilder's new _add_style_fingerprint_features reads it from layer=1
directly rather than layer=2, to make that explicit rather than relying on a
pass-through.
"""

import logging

import numpy as np
import pandas as pd

from src.matchups.box_scores import build_box_score_cache
from src.matchups.config import load_style_matchup_config
from src.matchups.db import cache_conn, init_cache_db, nba_api_conn
from src.matchups.pace_possession import build_advanced_stats_cache

logger = logging.getLogger(__name__)

FINGERPRINT_METRICS = [
    "pace_score",
    "three_pt_reliance",
    "paint_activity",
    "defensive_rating",
    "assist_rate",
    "offensive_rating",
]

# Track C pace/possession swap-in test (docs/NEW_DATA_FEASIBILITY.md): official
# PACE/POSS from nba_api's TeamGameLogs (Advanced), src/matchups/pace_possession.py.
# Deliberately NOT added to FINGERPRINT_METRICS -- same "Layer 1 (uncalibrated)
# only" scope cut as offensive_rating (see this module's own docstring): no
# archetype's injury_impact config entry references these, and unlike
# offensive_rating this list is never passed through injury_layer.py at all
# (kept fully decoupled from that module rather than relying on a no-op
# pass-through), so feature_builder.py reads these from layer=1 directly.
OFFICIAL_PACE_POSS_METRICS = ["official_pace", "official_poss"]


def _load_raw_team_game_metrics() -> pd.DataFrame:
    """One row per (game_id, team_id) with the 6 raw per-game style metrics +
    official_pace/official_poss + game_date."""
    conn = cache_conn()
    box = pd.read_sql_query("SELECT * FROM box_score_stats", conn)
    advanced = pd.read_sql_query("SELECT game_id, team_id, pace, poss FROM team_advanced_stats", conn)
    conn.close()

    # Opponent points, joined on game_id (the other team's row).
    opp = box[["game_id", "team_id", "pts"]].rename(columns={"team_id": "opp_team_id", "pts": "opp_pts"})
    merged = box.merge(opp, on="game_id")
    merged = merged[merged["team_id"] != merged["opp_team_id"]].copy()
    # Each game_id should now have exactly 2 rows (home perspective + away perspective).
    merged = merged.drop_duplicates(subset=["game_id", "team_id"])

    possessions = merged["fga"] - merged["oreb"] + merged["tov"] + 0.44 * merged["fta"]
    possessions = possessions.replace(0, np.nan)

    merged["pace_score"] = merged["pts"] + merged["opp_pts"] + merged["tov"] - 0.44 * merged["fta"]
    merged["three_pt_reliance"] = (merged["fg3a"] / merged["fga"].replace(0, np.nan)).fillna(0.0)
    merged["paint_activity"] = merged["fta"]
    merged["defensive_rating"] = (merged["opp_pts"] / possessions * 100).fillna(0.0)
    merged["assist_rate"] = (merged["ast"] / merged["fgm"].replace(0, np.nan)).fillna(0.0)
    # Offensive-quality counterpart to defensive_rating, same possessions estimate.
    merged["offensive_rating"] = (merged["pts"] / possessions * 100).fillna(0.0)

    # Official PACE/POSS (Track C candidate) -- left join, team_advanced_stats has
    # full parity with box_score_stats's game_id set as of the last cache build, but
    # left join (not inner) so a future gap degrades to NaN rather than silently
    # dropping team-games from every other metric too.
    merged = merged.merge(advanced, on=["game_id", "team_id"], how="left")
    merged = merged.rename(columns={"pace": "official_pace", "poss": "official_poss"})

    all_metrics = FINGERPRINT_METRICS + OFFICIAL_PACE_POSS_METRICS
    return merged[["game_id", "team_id", "game_date"] + all_metrics].sort_values(["team_id", "game_date"])


def _decay_weight(position, halflife: float):
    """weight = 0.5 ** (position / halflife). Pulled out as its own function so the
    identical decay math can be reused elsewhere in the project rather than reimplemented
    -- e.g. calibration.py's decay-weighted injury-impact calibration reuses this directly
    for down-weighting Out-events by their position in a continuous absence streak (see
    docs/A7_PHASE_LOG.md "Decay-weighted injury-impact calibration"). Accepts a scalar or
    an array; `position` here is "age" (0 = most recent) for the rolling-window use below,
    or "streak index" (1 = first game of an absence) for calibration.py's use."""
    return 0.5 ** (np.asarray(position, dtype=float) / halflife)


def _decayed_weighted_mean(values: np.ndarray, halflife: float) -> float:
    n = len(values)
    if n == 0:
        return np.nan
    age = np.arange(n - 1, -1, -1)  # most recent -> age 0
    weights = _decay_weight(age, halflife)
    mask = ~np.isnan(values)
    if not mask.any():
        return np.nan
    return float(np.average(values[mask], weights=weights[mask]))


def compute_rolling_fingerprints(window: int = 20, halflife: float = 5.0) -> pd.DataFrame:
    """Pre-game rolling fingerprint per (game_id, team_id). Strictly excludes the game itself.

    CRITICAL (data leakage): the current game's own row is excluded via `.shift(1)` before
    the rolling window is applied — never `df[df['date'] < game_date]` is skipped in favor
    of the equivalent shift, which is correct here because rows are already sorted per-team
    by game_date with one row per game (shift(1) == "all games strictly before this one").
    """
    raw = _load_raw_team_game_metrics()
    all_metrics = FINGERPRINT_METRICS + OFFICIAL_PACE_POSS_METRICS
    out_frames = []
    for team_id, g in raw.groupby("team_id", sort=False):
        g = g.sort_values("game_date").reset_index(drop=True)
        result = {"game_id": g["game_id"], "team_id": team_id, "game_date": g["game_date"]}
        n_games = g[FINGERPRINT_METRICS[0]].shift(1).rolling(window, min_periods=1).count()
        for metric in all_metrics:
            shifted = g[metric].shift(1)
            result[metric] = shifted.rolling(window, min_periods=1).apply(
                lambda x: _decayed_weighted_mean(x, halflife), raw=True
            )
        result["n_games_in_window"] = n_games
        out_frames.append(pd.DataFrame(result))

    fp = pd.concat(out_frames, ignore_index=True)
    return fp


def build_fingerprint_cache(layer: int = 1, min_games: int = 5) -> dict:
    """Compute Layer 1 fingerprints and cache them (matchup_fingerprints, layer=1).

    Rows with fewer than `min_games` prior games are dropped (reuses
    features.min_games_played from the main config rather than introducing a new
    magic number).
    """
    init_cache_db()
    build_box_score_cache()  # ensure box scores are present/parity-checked
    build_advanced_stats_cache()  # ensure official PACE/POSS are present/parity-checked
    cfg = load_style_matchup_config()
    fp = compute_rolling_fingerprints(window=cfg["fingerprint_window"], halflife=cfg["decay_halflife"])
    fp = fp[fp["n_games_in_window"] >= min_games].copy()
    fp["layer"] = layer

    conn = cache_conn()
    conn.execute("DELETE FROM matchup_fingerprints WHERE layer = ?", (layer,))
    rows = [
        (
            r.game_id,
            int(r.team_id),
            r.game_date,
            layer,
            r.pace_score,
            r.three_pt_reliance,
            r.paint_activity,
            r.defensive_rating,
            r.assist_rate,
            r.offensive_rating,
            r.official_pace,
            r.official_poss,
            int(r.n_games_in_window),
        )
        for r in fp.itertuples(index=False)
    ]
    conn.executemany(
        """INSERT OR REPLACE INTO matchup_fingerprints
           (game_id, team_id, game_date, layer, pace_score, three_pt_reliance,
            paint_activity, defensive_rating, assist_rate, offensive_rating,
            official_pace, official_poss, n_games_in_window)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        rows,
    )
    conn.commit()
    conn.close()

    summary = {
        "layer": layer,
        "n_team_games_total": len(fp) + 0,
        "n_rows_cached": len(rows),
        "sample": fp.head(3).to_dict("records"),
    }
    logger.info(f"Fingerprint cache built: layer={layer}, rows={len(rows)}")
    return summary


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    print(build_fingerprint_cache(layer=1))
