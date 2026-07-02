"""
Layer 2: Injury-Adjusted Style.

Runs BEFORE Layer 3 (similarity search) per the design doc's corrected layer
ordering — this module reads Layer 1 fingerprints (layer=1 in
matchup_fingerprints) and writes injury-adjusted fingerprints (layer=2), which
matchup_index.py / similarity.py then use for Layer 3.

Adjustment: for every (game_id, team_id) with an Out player on that game_date,
resolved to an archetype (Phase 0's name_resolution + archetype tables), apply
that archetype's calibrated delta (configs/config.yaml: style_matchup.injury_impact,
Phase 0 output) scaled by a severity multiplier. Severity is classified from the
player_injuries.reason text using the EXISTING
src/news_scraping/extractors/formula_scorer.classify_severity() (reused, not
duplicated) against the EXISTING injury_features.severity_weights config block
(reused via src/matchups/config.py:severity_weights(), not a second
injury_severity_multipliers block).

Multiple-players-out handling (not fully specified by the design doc): if two+
Out players share the same archetype on the same team-game, the archetype's delta
is applied ONCE, scaled by the MAX severity multiplier among them (not summed) —
summing would imply the calibrated delta was a per-player marginal effect, but it
was estimated as "this archetype was missing" (binary), so summing multiple
instances would double-count. Different archetypes missing simultaneously DO
stack additively (each archetype's config block targets its own metrics
independently, matching the design doc's per-archetype injury_impact structure).
"""

import logging

import pandas as pd

from src.matchups.config import load_style_matchup_config, severity_weights
from src.matchups.db import cache_conn, init_cache_db, injury_conn
from src.matchups.fingerprint import FINGERPRINT_METRICS
from src.matchups.players import _date_to_season_str
from src.news_scraping.extractors.formula_scorer import classify_severity

logger = logging.getLogger(__name__)


class _SW:
    """Adapter so classify_severity's `severity_weights.severe/.moderate/.minor`
    attribute access works against the plain dict returned by
    src/matchups/config.py:severity_weights() (that dict is reused as-is from
    injury_features.severity_weights rather than re-validated as a pydantic model)."""

    def __init__(self, d: dict):
        self.severe = d["severe"]
        self.moderate = d["moderate"]
        self.minor = d["minor"]


def _out_players_with_reason() -> pd.DataFrame:
    with injury_conn() as conn:
        inj = pd.read_sql_query(
            "SELECT game_date, team_id, player_name, reason FROM player_injuries WHERE status = 'Out'",
            conn,
        )
    cache = cache_conn()
    res = pd.read_sql_query(
        "SELECT player_name, player_id FROM player_name_resolution WHERE confidence IN ('high','medium')",
        cache,
    )
    arch = pd.read_sql_query("SELECT player_id, season, archetype FROM player_archetypes", cache)
    cache.close()

    inj = inj.merge(res, on="player_name", how="inner")
    inj["season"] = inj["game_date"].map(_date_to_season_str)
    inj = inj.merge(arch, on=["player_id", "season"], how="left")
    inj = inj[inj["archetype"].notna()]

    sw = _SW(severity_weights())
    inj["severity_mult"] = inj["reason"].apply(lambda r: classify_severity(r, sw))
    return inj[["game_date", "team_id", "archetype", "severity_mult"]]


def _team_game_archetype_severity(out_df: pd.DataFrame) -> dict[tuple, dict[str, float]]:
    """(game_date[:10], team_id) -> {archetype: max_severity_mult}"""
    result: dict[tuple, dict[str, float]] = {}
    for (gd, tid), grp in out_df.groupby([out_df["game_date"].str[:10], "team_id"]):
        by_arch: dict[str, float] = {}
        for arch, s in zip(grp["archetype"], grp["severity_mult"]):
            by_arch[arch] = max(by_arch.get(arch, 0.0), s)
        result[(gd, tid)] = by_arch
    return result


def build_injury_adjusted_fingerprints() -> dict:
    """Reads layer=1 fingerprints, applies injury deltas, writes layer=2."""
    init_cache_db()
    cfg = load_style_matchup_config()
    injury_impact = cfg["injury_impact"]

    conn = cache_conn()
    fp1 = pd.read_sql_query(
        "SELECT * FROM matchup_fingerprints WHERE layer = 1", conn
    )
    conn.close()

    out_df = _out_players_with_reason()
    severity_map = _team_game_archetype_severity(out_df)

    n_adjusted = 0
    fp2 = fp1.copy()
    for idx, row in fp2.iterrows():
        key = (row["game_date"][:10], row["team_id"])
        archetypes_out = severity_map.get(key)
        if not archetypes_out:
            continue
        n_adjusted += 1
        for archetype, severity_mult in archetypes_out.items():
            deltas = injury_impact.get(archetype, {})
            for metric, delta in deltas.items():
                if metric in FINGERPRINT_METRICS:
                    fp2.at[idx, metric] = row[metric] + delta * severity_mult

    fp2["layer"] = 2
    conn = cache_conn()
    conn.execute("DELETE FROM matchup_fingerprints WHERE layer = 2")
    rows = [
        (
            r.game_id, int(r.team_id), r.game_date, 2,
            r.pace_score, r.three_pt_reliance, r.paint_activity,
            r.defensive_rating, r.assist_rate, int(r.n_games_in_window),
        )
        for r in fp2.itertuples(index=False)
    ]
    conn.executemany(
        """INSERT OR REPLACE INTO matchup_fingerprints
           (game_id, team_id, game_date, layer, pace_score, three_pt_reliance,
            paint_activity, defensive_rating, assist_rate, n_games_in_window)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        rows,
    )
    conn.commit()
    conn.close()

    summary = {
        "n_team_games_total": len(fp2),
        "n_team_games_adjusted": n_adjusted,
        "pct_adjusted": round(n_adjusted / len(fp2), 4) if len(fp2) else 0.0,
    }
    logger.info(f"Layer 2 injury adjustment: {summary}")
    return summary


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    print(build_injury_adjusted_fingerprints())
