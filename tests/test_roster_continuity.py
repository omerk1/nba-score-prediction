"""Roster continuity: returning-share arithmetic, traded/absent players,
point-in-time snapshot lookup, and season boundaries."""

import numpy as np
import pandas as pd
import pytest

from src.feature_engineering.roster_continuity import build_continuity_table, lookup_for_games

T1, T2 = 1, 2


def _snap(rows):
    df = pd.DataFrame(
        rows, columns=["player_id", "team_id", "as_of_date", "minutes_per_game", "pts_per_game"]
    )
    df["as_of_date"] = pd.to_datetime(df["as_of_date"])
    return df


def _snapshots():
    return _snap(
        [
            # 2024-25 season, an early snapshot then the final one (only the final counts)
            (10, T1, "2024-11-01", 5, 1),
            (10, T1, "2025-05-01", 30, 20),  # star, leaves for T2
            (11, T1, "2025-05-01", 20, 10),  # stays
            (12, T1, "2025-05-01", 10, 10),  # out all season (never appears)
            (13, T1, "2025-05-01", 1, 0),  # outside top-3, ignored
            (20, T2, "2025-05-01", 30, 15),
            # 2025-26 season
            (11, T1, "2025-10-27", 22, 12),
            (10, T2, "2025-10-27", 31, 22),
            (20, T2, "2025-10-27", 30, 15),
            (11, T1, "2025-11-03", 22, 12),
            (12, T1, "2025-11-03", 12, 8),  # returns from injury
            (10, T2, "2025-11-03", 31, 22),
        ]
    )


def test_return_shares():
    t = build_continuity_table(_snapshots(), top_n=3).set_index(["team_id", "as_of_date"])
    # T1 prior top-3 = players 10/11/12: pts 20/10/10 (40), min 30/20/10 (60)
    r = t.loc[(T1, pd.Timestamp("2025-10-27"))]
    assert r.return_pts == pytest.approx(10 / 40)
    assert r.return_min == pytest.approx(20 / 60)
    r = t.loc[(T1, pd.Timestamp("2025-11-03"))]
    assert r.return_pts == pytest.approx(20 / 40)
    assert r.return_min == pytest.approx(30 / 60)
    # The traded-in star does not count as "returning" for his new team.
    assert t.loc[(T2, pd.Timestamp("2025-10-27"))].return_pts == pytest.approx(1.0)
    # Player 20 missing from T2's second snapshot -> zero returning.
    assert t.loc[(T2, pd.Timestamp("2025-11-03"))].return_pts == pytest.approx(0.0)
    # No 2023-24 data -> no prior rotation for 2024-25 rows.
    assert not (t.reset_index()["season"] == 2024).any()


def test_lookup_is_strictly_before_and_same_season():
    table = build_continuity_table(_snapshots(), top_n=3)
    games = pd.DataFrame(
        {
            "GAME_DATE": pd.to_datetime(
                ["2025-10-22", "2025-10-27", "2025-10-28", "2025-11-03", "2025-11-04"]
            ),
            "SEASON_ID": [22025] * 5,
            "HOME_TEAM_ID": [T1] * 5,
        }
    )
    out = lookup_for_games(table, games, "HOME_TEAM_ID")
    # Before the first snapshot: NaN (not last season's numbers).
    assert np.isnan(out.return_pts[0])
    # Snapshot dated the same day as the game is not used.
    assert np.isnan(out.return_pts[1])
    assert out.return_pts[2] == pytest.approx(0.25)
    assert out.return_pts[3] == pytest.approx(0.25)
    assert out.return_pts[4] == pytest.approx(0.5)


def test_lookup_preserves_row_order_and_playoff_season_id():
    table = build_continuity_table(_snapshots(), top_n=3)
    games = pd.DataFrame(
        {
            "GAME_DATE": pd.to_datetime(["2025-11-04", "2025-10-28"]),
            "SEASON_ID": [42025, 22025],
            "HOME_TEAM_ID": [T1, T1],
        }
    )
    out = lookup_for_games(table, games, "HOME_TEAM_ID")
    assert out.return_pts.tolist() == pytest.approx([0.5, 0.25])


def test_hook_early_season_cutoff_and_diff_only(tmp_path):
    import sqlite3
    from unittest.mock import MagicMock, patch

    from src.feature_engineering.feature_builder import FeatureBuilder

    db = tmp_path / "imp.sqlite"
    with sqlite3.connect(db) as conn:
        s = _snapshots()
        s["as_of_date"] = s["as_of_date"].dt.strftime("%Y-%m-%d")
        s.to_sql("player_importance", conn, index=False)
    games = pd.DataFrame(
        {
            "GAME_DATE": pd.to_datetime(["2025-11-04", "2025-11-05", "2025-11-06"]),
            "SEASON_ID": [22025] * 3,
            "HOME_TEAM_ID": [T1, T2, T1],
            "AWAY_TEAM_ID": [T2, T1, T2],
        }
    )
    cfg = MagicMock()
    cfg.roster_continuity = MagicMock(enabled=True, top_n=3, db_path=str(db), max_season_games=2, diff_only=True)
    with patch("src.feature_engineering.feature_builder.load_config", return_value=cfg):
        out = FeatureBuilder(rolling_windows=[3])._add_roster_continuity_features(games.copy())
    assert [c for c in out.columns if "roster" in c] == ["roster_return_pts_diff"]
    # T1 0.5, T2 0.0 (as of 2025-11-03); each team's 3rd game is blanked.
    assert out.roster_return_pts_diff[0] == pytest.approx(0.5)
    assert out.roster_return_pts_diff[1] == pytest.approx(-0.5)
    assert np.isnan(out.roster_return_pts_diff[2])
