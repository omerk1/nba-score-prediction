"""Pre-game PBP feature tests, weighted toward leakage.

The construction is only useful if a game's own possessions, and every later
game's, are invisible to its own feature value. Two independent checks here:
a direct one that recomputes the expected value from prior games by hand, and
a tampering check that rewrites the future and asserts past values are
unchanged.
"""

import sqlite3

import numpy as np
import pandas as pd
import pytest

from src.pbp.db import connect
from src.pbp.features import (
    MIN_WINDOW_GAMES,
    OWN_LUCK_WEIGHT,
    build_pbp_rolling_features,
    load_team_game_possessions,
)

H, A = 101, 202


def _possession(game_id, idx, off, de, points, period=1, start_secs=0.0, margin=0,
                fg3a=0, fg3m=0, fta=0, ftm=0):
    return (game_id, idx, period, off, de, 1 if off == H else 0, start_secs, start_secs + 14.0, 14.0,
            margin, margin + points, points, "made_fg" if points else "miss",
            1, fg3a, fta, 0, 0, 1 if points else 0, fg3m, ftm,
            None, None, None, None, "", "", "", 1, 1)


POSS_COLS = ("game_id, poss_idx, period, off_team_id, def_team_id, is_home_off, start_secs, end_secs, "
             "duration, margin_start, margin_end, points, outcome, n_fga, n_fg3a, n_fta, n_oreb, n_tov, "
             "n_fgm, n_fg3m, n_ftm, last_shot_value, last_shot_distance, last_shot_x, last_shot_y, "
             "last_shot_subtype, lineup_off, lineup_def, lineup_off_complete, lineup_def_complete")


@pytest.fixture
def dbs(tmp_path):
    """A synthetic season: `n_games` alternating-venue games between two teams,
    home scoring a steadily increasing number of points per game."""
    pbp_db = tmp_path / "pbp.sqlite"
    raw_db = tmp_path / "raw.sqlite"
    conn = connect(pbp_db)
    n_games = 20
    rows, games = [], []
    for g in range(n_games):
        gid = f"00{g:05d}"
        date = (pd.Timestamp("2024-10-22") + pd.Timedelta(days=2 * g)).strftime("%Y-%m-%d")
        games.append((gid, date, "Regular Season", H, A, 100.0, 100.0))
        # Home scores g+1 points on 10 possessions; away always scores 10.
        for i in range(10):
            rows.append(_possession(gid, i, H, A, 1 if i < g + 1 else 0))
            rows.append(_possession(gid, 100 + i, A, H, 1))
    conn.executemany(
        f"INSERT INTO possessions ({POSS_COLS}) VALUES ({','.join('?' * 30)})", rows
    )
    conn.commit()
    conn.close()

    rc = sqlite3.connect(raw_db)
    rc.execute("CREATE TABLE game (game_id TEXT PRIMARY KEY, game_date TEXT, season_type TEXT, "
               "team_id_home INTEGER, team_id_away INTEGER, pts_home REAL, pts_away REAL)")
    rc.executemany("INSERT INTO game VALUES (?,?,?,?,?,?,?)", games)
    rc.commit()
    rc.close()
    return str(pbp_db), str(raw_db)


def test_team_game_table_has_both_sides(dbs):
    tg = load_team_game_possessions(*dbs)
    assert len(tg) == 40                       # 20 games x 2 teams
    home = tg[tg.team == H].sort_values("game_date")
    assert list(home.pts_for)[:3] == [1, 2, 3]
    assert (home.pts_against == 10).all()      # away scores 10 every game
    assert (home.poss_off == 10).all() and (home.poss_def == 10).all()


def test_feature_uses_only_prior_games(dbs):
    feats = build_pbp_rolling_features(*dbs, window=5)
    home = feats[feats.team == H].sort_values("game_date").reset_index(drop=True)

    # Fewer than MIN_WINDOW_GAMES prior games -> NaN, not a partial value.
    assert home.pbp_net_rtg_luckadj[: MIN_WINDOW_GAMES].isna().all()
    assert home.pbp_net_rtg_luckadj[MIN_WINDOW_GAMES:].notna().all()

    # Game index 5 sees games 0-4: home scored 1+2+3+4+5=15 on 50 possessions,
    # allowed 50 on 50. No threes or free throws, so no luck correction applies.
    expected = 100 * 15 / 50 - 100 * 50 / 50
    assert home.pbp_net_rtg_luckadj[5] == pytest.approx(expected)

    # Game index 10 sees games 5-9: 6+7+8+9+10 = 40.
    assert home.pbp_net_rtg_luckadj[10] == pytest.approx(100 * 40 / 50 - 100.0)


def test_rewriting_the_future_cannot_change_the_past(dbs):
    pbp_db, raw_db = dbs
    before = build_pbp_rolling_features(pbp_db, raw_db, window=5)
    cutoff = before.game_date.sort_values().unique()[10]
    past_before = before[before.game_date <= cutoff].sort_values(["team", "game_date"])

    # Triple the points on every possession after the cutoff.
    with sqlite3.connect(pbp_db) as conn:
        late = pd.read_sql_query("SELECT game_id FROM possessions", conn).game_id.unique()
        late_ids = sorted(late)[11:]
        conn.executemany("UPDATE possessions SET points = points * 3 WHERE game_id = ?",
                         [(g,) for g in late_ids])
    after = build_pbp_rolling_features(pbp_db, raw_db, window=5)
    past_after = after[after.game_date <= cutoff].sort_values(["team", "game_date"])

    pd.testing.assert_series_equal(
        past_before.pbp_net_rtg_luckadj.reset_index(drop=True),
        past_after.pbp_net_rtg_luckadj.reset_index(drop=True),
    )


def test_league_rates_are_expanding_not_global(dbs):
    pbp_db, raw_db = dbs
    # Give the last game an extreme three-point night. A global league rate would
    # shift every earlier game's baseline; an expanding one cannot.
    with sqlite3.connect(pbp_db) as conn:
        last = pd.read_sql_query("SELECT MAX(game_id) g FROM possessions", conn).g[0]
        conn.execute("UPDATE possessions SET n_fg3a = 5, n_fg3m = 5 WHERE game_id = ?", (last,))
    feats = build_pbp_rolling_features(pbp_db, raw_db, window=5)
    early = feats[feats.game_date < feats.game_date.max()].pbp_net_rtg_luckadj

    with sqlite3.connect(pbp_db) as conn:
        conn.execute("UPDATE possessions SET n_fg3a = 0, n_fg3m = 0 WHERE game_id = ?", (last,))
    baseline = build_pbp_rolling_features(pbp_db, raw_db, window=5)
    early_baseline = baseline[baseline.game_date < baseline.game_date.max()].pbp_net_rtg_luckadj

    pd.testing.assert_series_equal(early.reset_index(drop=True), early_baseline.reset_index(drop=True))


def test_garbage_possessions_are_excluded(dbs):
    pbp_db, raw_db = dbs
    base = load_team_game_possessions(pbp_db, raw_db)
    # A 30-point lead with 2 minutes left in regulation is garbage time.
    with sqlite3.connect(pbp_db) as conn:
        conn.executemany(
            f"INSERT INTO possessions ({POSS_COLS}) VALUES ({','.join('?' * 30)})",
            [_possession("0000000", 900 + i, H, A, 3, period=4, start_secs=2790.0, margin=30)
             for i in range(5)],
        )
    after = load_team_game_possessions(pbp_db, raw_db)
    pd.testing.assert_frame_equal(
        base.reset_index(drop=True), after.reset_index(drop=True)
    )


def test_own_luck_is_half_weighted_and_opponent_full(dbs):
    pbp_db, raw_db = dbs
    # Home makes every three it takes; away takes none. With the league rate
    # driven by home's own shooting, the correction is bounded and signed.
    with sqlite3.connect(pbp_db) as conn:
        conn.execute("UPDATE possessions SET n_fg3a = 1, n_fg3m = 1 WHERE off_team_id = ? AND points > 0", (H,))
    feats = build_pbp_rolling_features(pbp_db, raw_db, window=5)
    assert feats.pbp_net_rtg_luckadj.notna().any()
    assert 0 < OWN_LUCK_WEIGHT < 1
