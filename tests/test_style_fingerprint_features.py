"""
Regression tests for `_add_style_fingerprint_features`'s cache lookup.

Bug being guarded against: the original implementation joined the precomputed
`matchup_fingerprints` cache onto the input dataframe by an *exact* `game_id`
match. That's fine for training (every row is a real, already-cached game_id),
but it is silently broken for live prediction: predict_game.py builds a
synthetic row for the matchup being predicted with GAME_ID='upcoming' (a
placeholder string that can never match a cached game_id), so the old join
produced NaN for every style-fingerprint column on exactly the row we're
trying to predict.

The fix replaces the exact-game_id join with `pd.merge_asof` on
(team_id, game_date): for each team, find its most recent precomputed
fingerprint at or before the target date. These tests lock in the three
properties that fix relies on:
  1. A brand-new/never-cached GAME_ID for a real team_id + a date after that
     team's last cached entry still resolves to that team's most recent
     cached fingerprint (the predict_game.py case).
  2. An exact game_id/date match still returns that game's own cached value
     (parity with the old behavior for historical/training rows).
  3. A team with no qualifying prior history returns NaN, same as today.
"""

import sqlite3
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.feature_engineering.feature_builder import FeatureBuilder

# Must match FeatureBuilder._RAW_STYLE_CALIBRATED_METRICS / _RAW_STYLE_UNCALIBRATED_METRIC
CALIBRATED_METRICS = ["pace_score", "three_pt_reliance", "paint_activity", "defensive_rating", "assist_rate"]
UNCALIBRATED_METRIC = "offensive_rating"
ALL_METRICS = CALIBRATED_METRICS + [UNCALIBRATED_METRIC]


def _mock_config():
    mock_cfg = MagicMock()
    mock_cfg.style_matchup = MagicMock(enabled=False, raw_features_enabled=True)
    return mock_cfg


def _metric_values(seed: float) -> dict:
    """Distinct, deterministic values per metric so a wrong-row bug can't accidentally match."""
    return {metric: seed + i * 0.01 for i, metric in enumerate(ALL_METRICS)}


def _write_cache_db(cache_path, fingerprint_rows):
    """
    fingerprint_rows: list of (game_id, team_id, game_date_str, seed) tuples.
    Writes both layer=1 and layer=2 rows (same values in both layers) for each entry,
    matching the real `matchup_fingerprints` schema built by src/matchups/fingerprint.py.
    """
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(cache_path))
    conn.execute(
        """
        CREATE TABLE matchup_fingerprints (
            game_id TEXT, team_id INTEGER, game_date TEXT, layer INTEGER,
            pace_score REAL, three_pt_reliance REAL, paint_activity REAL,
            defensive_rating REAL, assist_rate REAL, offensive_rating REAL,
            n_games_in_window INTEGER
        )
        """
    )
    for game_id, team_id, game_date, seed in fingerprint_rows:
        values = _metric_values(seed)
        for layer in (1, 2):
            conn.execute(
                """INSERT INTO matchup_fingerprints
                   (game_id, team_id, game_date, layer, pace_score, three_pt_reliance,
                    paint_activity, defensive_rating, assist_rate, offensive_rating,
                    n_games_in_window)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    game_id, team_id, game_date, layer,
                    values["pace_score"], values["three_pt_reliance"], values["paint_activity"],
                    values["defensive_rating"], values["assist_rate"], values["offensive_rating"],
                    10,
                ),
            )
    conn.commit()
    conn.close()


def _query_df(game_id, game_date, home_team_id, away_team_id):
    return pd.DataFrame([{
        "GAME_ID": game_id,
        "GAME_DATE": pd.Timestamp(game_date),
        "HOME_TEAM_ID": home_team_id,
        "AWAY_TEAM_ID": away_team_id,
    }])


class TestStyleFingerprintAsofLookup:

    @patch("src.feature_engineering.feature_builder.load_config")
    def test_new_game_id_after_last_cached_entry_gets_most_recent_fingerprint(self, mock_config, tmp_path, monkeypatch):
        """
        Mimics predict_game.py's exact pattern: a never-before-seen GAME_ID
        ('upcoming') for a real team_id, dated after that team's last cached
        fingerprint. Must resolve to that team's most recent cached value, not NaN.
        """
        mock_config.return_value = _mock_config()
        monkeypatch.chdir(tmp_path)
        _write_cache_db(
            tmp_path / "outputs" / "a7_matchups_cache.sqlite",
            [
                ("g1", 100, "2024-01-01", 1.0),
                ("g2", 100, "2024-01-10", 2.0),  # most recent for team 100
            ],
        )

        df = _query_df("upcoming", "2024-01-15", home_team_id=100, away_team_id=999)

        fb = FeatureBuilder(rolling_windows=[3])
        result = fb._add_style_fingerprint_features(df)

        expected = _metric_values(2.0)
        for metric in ALL_METRICS:
            assert result.loc[0, f"home_style_{metric}"] == pytest.approx(expected[metric])
            assert not pd.isna(result.loc[0, f"home_style_{metric}"])

        # Away team (999) has no cache at all -> NaN, and diff carries that NaN through.
        for metric in ALL_METRICS:
            assert pd.isna(result.loc[0, f"away_style_{metric}"])
            assert pd.isna(result.loc[0, f"style_{metric}_diff"])

    @patch("src.feature_engineering.feature_builder.load_config")
    def test_exact_game_id_and_date_match_still_returns_that_games_value(self, mock_config, tmp_path, monkeypatch):
        """Parity check: querying with the SAME game_id/date as a cached row must still
        return that row's own value (the historical/training behavior the old exact-match
        join already handled correctly)."""
        mock_config.return_value = _mock_config()
        monkeypatch.chdir(tmp_path)
        _write_cache_db(
            tmp_path / "outputs" / "a7_matchups_cache.sqlite",
            [
                ("g1", 100, "2024-01-01", 1.0),
                ("g2", 100, "2024-01-10", 2.0),
            ],
        )

        # Query at exactly g1's own date -- asof (backward, allow_exact_matches=True)
        # must land on g1's value, not skip past it or fall through to NaN.
        df = _query_df("g1", "2024-01-01", home_team_id=100, away_team_id=999)

        fb = FeatureBuilder(rolling_windows=[3])
        result = fb._add_style_fingerprint_features(df)

        expected = _metric_values(1.0)
        for metric in ALL_METRICS:
            assert result.loc[0, f"home_style_{metric}"] == pytest.approx(expected[metric])

    @patch("src.feature_engineering.feature_builder.load_config")
    def test_no_prior_history_returns_nan(self, mock_config, tmp_path, monkeypatch):
        """A team with zero qualifying cached fingerprints (or none at/before the query
        date) must yield NaN -- same convention as today's insufficient-history case,
        not a new failure mode."""
        mock_config.return_value = _mock_config()
        monkeypatch.chdir(tmp_path)
        _write_cache_db(
            tmp_path / "outputs" / "a7_matchups_cache.sqlite",
            [
                # Team 100 only has a fingerprint AFTER the query date below.
                ("g1", 100, "2024-02-01", 1.0),
            ],
        )

        df = _query_df("upcoming", "2024-01-15", home_team_id=100, away_team_id=999)

        fb = FeatureBuilder(rolling_windows=[3])
        result = fb._add_style_fingerprint_features(df)

        for metric in ALL_METRICS:
            assert pd.isna(result.loc[0, f"home_style_{metric}"])
            assert pd.isna(result.loc[0, f"away_style_{metric}"])
            assert pd.isna(result.loc[0, f"style_{metric}_diff"])

    @patch("src.feature_engineering.feature_builder.load_config")
    def test_disabled_flag_skips_entirely(self, mock_config, tmp_path, monkeypatch):
        """raw_features_enabled=False must short-circuit before touching the cache at all."""
        mock_cfg = MagicMock()
        mock_cfg.style_matchup = MagicMock(enabled=False, raw_features_enabled=False)
        mock_config.return_value = mock_cfg
        monkeypatch.chdir(tmp_path)
        # No cache DB written at all -- if the method tried to read it, this would raise.

        df = _query_df("upcoming", "2024-01-15", home_team_id=100, away_team_id=999)
        fb = FeatureBuilder(rolling_windows=[3])
        result = fb._add_style_fingerprint_features(df)

        assert "home_style_pace_score" not in result.columns


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
