"""
Regression tests for `_add_on_off_splits_features`.

This method reads three separate caches (player_on_off_splits, player_injuries,
player_name_resolution) and combines three `pd.merge_asof` lookups (vs_opponent /
venue / overall) with `combine_first` -- exactly the kind of logic that already
produced one real dtype bug (see docs/on_off_splits_log.md's "Second bug found and
fixed") during manual smoke-testing. These tests lock in the properties that bug
fix and the rest of the method's design rely on:

  1. A cached on/off value for a resolved `Out` player flows through to the team-
     level impact column correctly (not NaN, not the wrong value).
  2. Split preference: vs_opponent (for the correct opponent) beats venue-specific
     (home/away) beats overall.
  3. The min(on, off) noise gate excludes a player whose combined minutes look
     ample but where one side alone is a tiny, noisy sample (the exact bug found
     and fixed in feature_builder.py).
  4. The leakage guard: a checkpoint dated exactly on the target game's own date
     must NOT be used (DateTo was confirmed inclusive of that date at fetch time),
     only a checkpoint strictly before `game_date - 1 day` is eligible.
  5. No `Out` players resolved -> 0.0 (not NaN) impact, since "nobody out" is a
     legitimate zero-impact case, not missing data.
  6. `on_off_splits.enabled=False` short-circuits before touching any cache file
     at all.
  7. A direct regression test for the dtype-mismatch bug: opponent_team_id is
     non-null for vs_opponent rows and NULL for other split types within the SAME
     table read, which makes pandas infer float64 for the whole column -- must not
     raise and must still return the correct value.
"""

import sqlite3
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.feature_engineering.feature_builder import FeatureBuilder

MIN_ON_OFF_MINUTES = 50.0
HOME_TEAM = 100
AWAY_TEAM = 200
OTHER_TEAM = 300


def _mock_config(on_off_db_path, injury_db_path, enabled: bool = True, min_on_off_minutes: float = MIN_ON_OFF_MINUTES):
    mock_cfg = MagicMock()
    mock_cfg.on_off_splits = MagicMock(
        enabled=enabled, db_path=str(on_off_db_path), min_on_off_minutes=min_on_off_minutes,
    )
    mock_cfg.injury_features = MagicMock(db_path=str(injury_db_path))
    return mock_cfg


def _write_on_off_db(path, rows):
    """
    rows: list of dicts with keys player_id, team_id, split_type, opponent_team_id
    (None unless split_type='vs_opponent'), as_of_date, min_on, min_off,
    on_off_plus_minus. Matches the real schema in
    src/migrations/migration_create_player_on_off_splits.py.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.execute(
        """
        CREATE TABLE player_on_off_splits (
            player_id INTEGER, player_name TEXT, team_id INTEGER, split_type TEXT,
            opponent_team_id INTEGER, as_of_date TEXT, season TEXT,
            gp_on REAL, gp_off REAL, min_on REAL, min_off REAL,
            plus_minus_on REAL, plus_minus_off REAL, net_rating_on REAL, net_rating_off REAL,
            on_off_plus_minus REAL, on_off_net_rating REAL, fetched_at TEXT
        )
        """
    )
    for r in rows:
        conn.execute(
            """INSERT INTO player_on_off_splits
               (player_id, player_name, team_id, split_type, opponent_team_id, as_of_date,
                season, min_on, min_off, on_off_plus_minus, fetched_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                r["player_id"], r.get("player_name", "Test Player"), r["team_id"], r["split_type"],
                r.get("opponent_team_id"), r["as_of_date"], r.get("season", "2023-24"),
                r["min_on"], r["min_off"], r["on_off_plus_minus"], "2024-01-01T00:00:00",
            ),
        )
    conn.commit()
    conn.close()


def _write_injury_db(path, rows):
    """rows: list of (game_date, team_id, player_name, status) tuples."""
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.execute(
        """
        CREATE TABLE player_injuries (
            game_date TEXT, team_id INTEGER, player_name TEXT, status TEXT, reason TEXT, source TEXT
        )
        """
    )
    for game_date, team_id, player_name, status in rows:
        conn.execute(
            "INSERT INTO player_injuries (game_date, team_id, player_name, status, reason, source) "
            "VALUES (?, ?, ?, ?, '', 'pdf')",
            (game_date, team_id, player_name, status),
        )
    conn.commit()
    conn.close()


def _write_name_res_db(path, rows):
    """rows: list of (player_name, player_id, confidence) tuples."""
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.execute(
        """
        CREATE TABLE player_name_resolution (
            player_name TEXT PRIMARY KEY, player_id INTEGER, resolution_method TEXT, confidence TEXT
        )
        """
    )
    for player_name, player_id, confidence in rows:
        conn.execute(
            "INSERT INTO player_name_resolution (player_name, player_id, resolution_method, confidence) "
            "VALUES (?, ?, 'normalized', ?)",
            (player_name, player_id, confidence),
        )
    conn.commit()
    conn.close()


def _query_df(game_date, home_team_id=HOME_TEAM, away_team_id=AWAY_TEAM):
    return pd.DataFrame([{
        "GAME_ID": "upcoming",
        "GAME_DATE": pd.Timestamp(game_date),
        "HOME_TEAM_ID": home_team_id,
        "AWAY_TEAM_ID": away_team_id,
    }])


def _setup(tmp_path, monkeypatch, mock_config, on_off_rows, injury_rows, name_res_rows, enabled=True, min_on_off_minutes=MIN_ON_OFF_MINUTES):
    on_off_db = tmp_path / "data" / "player_on_off_splits.sqlite"
    injury_db = tmp_path / "data" / "injury_features.sqlite"
    name_res_db = tmp_path / "outputs" / "style_fingerprint_cache.sqlite"

    _write_on_off_db(on_off_db, on_off_rows)
    _write_injury_db(injury_db, injury_rows)
    _write_name_res_db(name_res_db, name_res_rows)

    monkeypatch.setattr("src.feature_engineering.feature_builder.CACHE_DB", str(name_res_db))
    mock_config.return_value = _mock_config(on_off_db, injury_db, enabled=enabled, min_on_off_minutes=min_on_off_minutes)
    return on_off_db, injury_db, name_res_db


class TestOnOffSplitsFeature:

    @patch("src.feature_engineering.feature_builder.load_config")
    def test_resolved_out_player_impact_flows_through(self, mock_config, tmp_path, monkeypatch):
        """A cached `overall` on/off value for a resolved Out player must appear as
        the team-level impact, not NaN and not some other value."""
        _setup(
            tmp_path, monkeypatch, mock_config,
            on_off_rows=[{
                "player_id": 501, "team_id": HOME_TEAM, "split_type": "overall",
                "opponent_team_id": None, "as_of_date": "2024-01-05",
                "min_on": 200, "min_off": 100, "on_off_plus_minus": 12.5,
            }],
            injury_rows=[("2024-01-10", HOME_TEAM, "Player One", "Out")],
            name_res_rows=[("Player One", 501, "high")],
        )

        df = _query_df("2024-01-10")
        fb = FeatureBuilder(rolling_windows=[3])
        result = fb._add_on_off_splits_features(df)

        assert result.loc[0, "home_team_missing_player_on_off_impact"] == pytest.approx(12.5)
        assert result.loc[0, "home_team_n_out_total"] == 1
        assert result.loc[0, "home_team_n_out_resolved_on_off"] == 1
        assert result.loc[0, "away_team_missing_player_on_off_impact"] == 0.0
        assert result.loc[0, "missing_player_on_off_impact_diff"] == pytest.approx(12.5)

    @patch("src.feature_engineering.feature_builder.load_config")
    def test_split_preference_vs_opponent_beats_venue_beats_overall(self, mock_config, tmp_path, monkeypatch):
        """Same player has cached overall/home/vs_opponent values -- the most
        specific available split (vs_opponent for the CORRECT opponent) must win,
        not home (venue) or overall."""
        _setup(
            tmp_path, monkeypatch, mock_config,
            on_off_rows=[
                {"player_id": 501, "team_id": HOME_TEAM, "split_type": "overall",
                 "opponent_team_id": None, "as_of_date": "2024-01-05", "min_on": 200, "min_off": 100, "on_off_plus_minus": 5.0},
                {"player_id": 501, "team_id": HOME_TEAM, "split_type": "home",
                 "opponent_team_id": None, "as_of_date": "2024-01-05", "min_on": 150, "min_off": 80, "on_off_plus_minus": 10.0},
                # Wrong opponent -- must NOT be picked even though it's vs_opponent.
                {"player_id": 501, "team_id": HOME_TEAM, "split_type": "vs_opponent",
                 "opponent_team_id": OTHER_TEAM, "as_of_date": "2024-01-05", "min_on": 100, "min_off": 60, "on_off_plus_minus": 999.0},
                # Correct opponent (AWAY_TEAM) -- this is the one that should win.
                {"player_id": 501, "team_id": HOME_TEAM, "split_type": "vs_opponent",
                 "opponent_team_id": AWAY_TEAM, "as_of_date": "2024-01-05", "min_on": 100, "min_off": 60, "on_off_plus_minus": 20.0},
            ],
            injury_rows=[("2024-01-10", HOME_TEAM, "Player One", "Out")],
            name_res_rows=[("Player One", 501, "high")],
        )

        df = _query_df("2024-01-10", home_team_id=HOME_TEAM, away_team_id=AWAY_TEAM)
        fb = FeatureBuilder(rolling_windows=[3])
        result = fb._add_on_off_splits_features(df)

        assert result.loc[0, "home_team_missing_player_on_off_impact"] == pytest.approx(20.0)

    @patch("src.feature_engineering.feature_builder.load_config")
    def test_min_on_off_minutes_gate_excludes_thin_sample(self, mock_config, tmp_path, monkeypatch):
        """Regression test for the noise-gate bug: combined on+off minutes look
        ample (604) but the off-court side alone (4 minutes) is far below the
        threshold -- this player's on_off_plus_minus must be excluded entirely
        (contributes 0, not counted as resolved), even though gp/GP look fine."""
        _setup(
            tmp_path, monkeypatch, mock_config,
            on_off_rows=[{
                "player_id": 501, "team_id": HOME_TEAM, "split_type": "overall",
                "opponent_team_id": None, "as_of_date": "2024-01-05",
                "min_on": 600, "min_off": 4, "on_off_plus_minus": 150.0,  # noisy, must be gated out
            }],
            injury_rows=[("2024-01-10", HOME_TEAM, "Player One", "Out")],
            name_res_rows=[("Player One", 501, "high")],
        )

        df = _query_df("2024-01-10")
        fb = FeatureBuilder(rolling_windows=[3])
        result = fb._add_on_off_splits_features(df)

        assert result.loc[0, "home_team_missing_player_on_off_impact"] == 0.0
        assert result.loc[0, "home_team_n_out_total"] == 1  # still counted as Out
        assert result.loc[0, "home_team_n_out_resolved_on_off"] == 0  # but not resolved

    @patch("src.feature_engineering.feature_builder.load_config")
    def test_same_day_checkpoint_not_used_leakage_guard(self, mock_config, tmp_path, monkeypatch):
        """DateTo was confirmed INCLUSIVE of games on that exact date at fetch
        time, so a checkpoint dated exactly on the target game's own date must
        NOT be used as its as-of value (that would leak same-day information) --
        only a checkpoint strictly before `game_date - 1 day` is eligible."""
        _setup(
            tmp_path, monkeypatch, mock_config,
            on_off_rows=[
                # Same-day checkpoint -- must be REJECTED by the day-shift guard.
                {"player_id": 501, "team_id": HOME_TEAM, "split_type": "overall",
                 "opponent_team_id": None, "as_of_date": "2024-01-10", "min_on": 200, "min_off": 100, "on_off_plus_minus": 99.0},
                # Genuinely prior checkpoint -- this is the one that should be used.
                {"player_id": 501, "team_id": HOME_TEAM, "split_type": "overall",
                 "opponent_team_id": None, "as_of_date": "2024-01-08", "min_on": 200, "min_off": 100, "on_off_plus_minus": 7.0},
            ],
            injury_rows=[("2024-01-10", HOME_TEAM, "Player One", "Out")],
            name_res_rows=[("Player One", 501, "high")],
        )

        df = _query_df("2024-01-10")
        fb = FeatureBuilder(rolling_windows=[3])
        result = fb._add_on_off_splits_features(df)

        assert result.loc[0, "home_team_missing_player_on_off_impact"] == pytest.approx(7.0)

    @patch("src.feature_engineering.feature_builder.load_config")
    def test_no_out_players_returns_zero_not_nan(self, mock_config, tmp_path, monkeypatch):
        """A team-game with no `Out` players at all must yield 0.0 (a real,
        legitimate zero-impact value), never NaN."""
        _setup(
            tmp_path, monkeypatch, mock_config,
            on_off_rows=[{
                "player_id": 501, "team_id": HOME_TEAM, "split_type": "overall",
                "opponent_team_id": None, "as_of_date": "2024-01-05", "min_on": 200, "min_off": 100, "on_off_plus_minus": 12.5,
            }],
            injury_rows=[],  # nobody out
            name_res_rows=[("Player One", 501, "high")],
        )

        df = _query_df("2024-01-10")
        fb = FeatureBuilder(rolling_windows=[3])
        result = fb._add_on_off_splits_features(df)

        assert result.loc[0, "home_team_missing_player_on_off_impact"] == 0.0
        assert not pd.isna(result.loc[0, "home_team_missing_player_on_off_impact"])
        assert result.loc[0, "home_team_n_out_total"] == 0
        assert result.loc[0, "home_team_n_out_resolved_on_off"] == 0
        assert result.loc[0, "missing_player_on_off_impact_diff"] == 0.0

    @patch("src.feature_engineering.feature_builder.load_config")
    def test_disabled_flag_skips_entirely(self, mock_config, tmp_path, monkeypatch):
        """on_off_splits.enabled=False must short-circuit before touching any
        cache file -- none of the three DBs are created here, so if the method
        tried to open any of them it would raise."""
        mock_cfg = MagicMock()
        mock_cfg.on_off_splits = MagicMock(enabled=False)
        mock_config.return_value = mock_cfg
        monkeypatch.chdir(tmp_path)

        df = _query_df("2024-01-10")
        fb = FeatureBuilder(rolling_windows=[3])
        result = fb._add_on_off_splits_features(df)

        assert "home_team_missing_player_on_off_impact" not in result.columns

    @patch("src.feature_engineering.feature_builder.load_config")
    def test_dtype_mismatch_regression_mixed_opponent_team_id(self, mock_config, tmp_path, monkeypatch):
        """Regression test for the dtype bug: opponent_team_id is non-null only
        for vs_opponent rows and NULL for every other split_type in the SAME
        table read -- pandas infers float64 for the whole column in that case,
        which crashed the original merge_asof call with a dtype mismatch against
        the int64 query-side column. Must not raise, and must still resolve to
        the correct (vs_opponent) value."""
        _setup(
            tmp_path, monkeypatch, mock_config,
            on_off_rows=[
                # NULL opponent_team_id (overall/home/away rows) mixed with a
                # non-null one (vs_opponent) in the same table/read -- this is
                # exactly the condition that produced float64 inference.
                {"player_id": 501, "team_id": HOME_TEAM, "split_type": "overall",
                 "opponent_team_id": None, "as_of_date": "2024-01-05", "min_on": 200, "min_off": 100, "on_off_plus_minus": 5.0},
                {"player_id": 501, "team_id": HOME_TEAM, "split_type": "home",
                 "opponent_team_id": None, "as_of_date": "2024-01-05", "min_on": 150, "min_off": 80, "on_off_plus_minus": 10.0},
                {"player_id": 501, "team_id": HOME_TEAM, "split_type": "vs_opponent",
                 "opponent_team_id": AWAY_TEAM, "as_of_date": "2024-01-05", "min_on": 100, "min_off": 60, "on_off_plus_minus": 20.0},
            ],
            injury_rows=[("2024-01-10", HOME_TEAM, "Player One", "Out")],
            name_res_rows=[("Player One", 501, "high")],
        )

        df = _query_df("2024-01-10", home_team_id=HOME_TEAM, away_team_id=AWAY_TEAM)
        fb = FeatureBuilder(rolling_windows=[3])

        # Must not raise pandas.errors.MergeError (the original bug).
        result = fb._add_on_off_splits_features(df)

        assert result.loc[0, "home_team_missing_player_on_off_impact"] == pytest.approx(20.0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
