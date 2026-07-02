"""
Standalone A2 (Extended H2H) baseline for A7 comparison purposes only.

This is a deliberately small re-implementation of the *core* idea behind
`FeatureBuilder._add_h2h_features` in `src/feature_engineering/feature_builder.py`
(read-only reference — that file is never imported or modified, per project rules):
for a given home/away pair and game date, the expanding average point
differential across all their prior meetings (from the home team's perspective,
canonicalized so a "home" vs "away" mislabeling across meetings doesn't flip
sign), shifted so the game itself is never included (no leakage).

Used for two things in the A7 exploration:
  1. `low_confidence_fallback: h2h` in Layer 3 (similarity.py) — what to return
     when a style-based match has too few similar games.
  2. Phase 4's side-by-side A2 vs A7 vs A2+A7 comparison CSV.

Not a replacement for the real A2 feature (h2h_avg_diff / h2h_home_margin_L{w}) —
those live in feature_builder.py and are unaffected by anything here.
"""

import pandas as pd

from src.matchups.db import nba_api_conn


def _load_games_for_h2h() -> pd.DataFrame:
    with nba_api_conn() as conn:
        games = pd.read_sql_query(
            "SELECT game_id, game_date, team_id_home, team_id_away, pts_home, pts_away "
            "FROM game WHERE pts_home IS NOT NULL AND pts_away IS NOT NULL "
            "ORDER BY game_date",
            conn,
        )
    games["home_margin"] = games["pts_home"] - games["pts_away"]
    games["matchup_key"] = games.apply(
        lambda r: "_".join(sorted([str(r["team_id_home"]), str(r["team_id_away"])])), axis=1
    )
    games["canonical_team"] = games[["team_id_home", "team_id_away"]].min(axis=1)
    games["canonical_margin"] = games.apply(
        lambda r: r["home_margin"] if r["team_id_home"] == r["canonical_team"] else -r["home_margin"],
        axis=1,
    )
    return games


def build_a2_h2h_scores() -> pd.DataFrame:
    """Returns game_id -> h2h_score (home-perspective expanding-mean margin, shifted)."""
    games = _load_games_for_h2h()
    games["h2h_score"] = 0.0
    for key, grp in games.groupby("matchup_key"):
        idx = grp.index
        expanding_canon = grp["canonical_margin"].shift(1).expanding(min_periods=1).mean()
        is_canon_home = grp["team_id_home"] == grp["canonical_team"]
        h2h_home_perspective = expanding_canon.where(is_canon_home, -expanding_canon)
        games.loc[idx, "h2h_score"] = h2h_home_perspective.fillna(0.0).values
    return games[["game_id", "game_date", "team_id_home", "team_id_away", "home_margin", "h2h_score"]]


if __name__ == "__main__":
    df = build_a2_h2h_scores()
    print(df.head(10))
    print(f"Total rows: {len(df)}")
    print(f"corr(h2h_score, home_margin) = {df['h2h_score'].corr(df['home_margin']):.4f}")
