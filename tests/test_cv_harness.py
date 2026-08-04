"""
Sanity tests for src/evaluation/cv_harness.py -- the expanding-window CV
harness (CLAUDE.md's "Project Rules (ML experimentation)" section). These
are fast (no DB access, no model training): fold-definition validation is
pure date-string logic, and the naive-baseline test uses small synthetic
DataFrames, not the full pipeline.

Covers:
  1. Fold boundaries never overlap and val/test seasons follow train
     chronologically, for the real configs/config.yaml folds -- plus
     negative cases proving validate_fold_definitions actually rejects a
     walk-BACKWARD ordering and a chronologically-broken single fold, not
     just passes trivially.
  2. Loading configs/config.yaml twice yields identical fold definitions
     (deterministic parsing).
  3. naive_baseline_metrics recomputes real per-input values, not a
     constant -- two differently-scored synthetic fold windows produce
     different results.
"""

import sqlite3

import pandas as pd
import pytest

from src.evaluation.cv_harness import naive_baseline_metrics, validate_fold_definitions
from src.utils.config_loader import CVFoldConfig, load_config


def _fold(name, train_end, val_start, val_end, test_start, test_end):
    return CVFoldConfig(
        name=name,
        train_end_date=train_end,
        validation_start_date=val_start,
        validation_end_date=val_end,
        test_start_date=test_start,
        test_end_date=test_end,
    )


class TestValidateFoldDefinitions:

    def test_real_config_folds_pass(self):
        """configs/config.yaml's actual cv.folds must validate cleanly --
        the harness's own committed folds are not exempt from their own check."""
        cfg = load_config()
        validate_fold_definitions(cfg.cv.folds)

    def test_empty_folds_rejected(self):
        with pytest.raises(ValueError, match="No CV folds"):
            validate_fold_definitions([])

    def test_single_fold_non_chronological_dates_rejected(self):
        """A fold whose own train_end/val/test dates aren't in order (e.g.
        validation_end before validation_start) must be rejected."""
        bad = [_fold("fold1", "2020-08-14", "2021-05-16", "2020-12-22", "2021-10-19", "2022-04-10")]
        with pytest.raises(ValueError, match="not chronological"):
            validate_fold_definitions(bad)

    def test_walk_backward_ordering_rejected(self):
        """The exact mistake this rule exists to catch: folds ordered
        newest -> oldest instead of oldest -> newest. validation_start_date
        must strictly increase fold-to-fold."""
        cfg = load_config()
        reversed_folds = list(reversed(cfg.cv.folds))
        with pytest.raises(ValueError, match="must walk forward in time"):
            validate_fold_definitions(reversed_folds)

    def test_validation_predating_prior_folds_training_window_rejected(self):
        """The 3rd assertion (no fold's validation/test predates an earlier
        fold's own train_end) is mathematically implied by the first two (see
        validate_fold_definitions' docstring) -- it cannot be independently
        triggered through the public function once those hold. Verified
        directly here instead: for the real, valid config folds, every later
        fold's validation_start truly is after every earlier fold's
        train_end (the property (3) exists to guarantee)."""
        cfg = load_config()
        folds = cfg.cv.folds
        for i, cur in enumerate(folds):
            for prev in folds[:i]:
                assert cur.validation_start_date > prev.train_end_date
                assert cur.test_start_date > prev.train_end_date


class TestConfigReloadDeterminism:

    def test_reloading_config_yields_identical_folds(self):
        """Two independent load_config() calls (no caching between them --
        each re-reads and re-parses configs/config.yaml from disk) must
        produce byte-identical fold definitions."""
        cfg1 = load_config()
        cfg2 = load_config()
        assert cfg1.cv.folds == cfg2.cv.folds
        assert [f.name for f in cfg1.cv.folds] == [f.name for f in cfg2.cv.folds]


class TestNaiveBaselineRecomputedPerFold:

    def test_naive_baselines_differ_across_differently_scored_folds(self):
        """naive_baseline_metrics must derive its output FROM the input data
        every call, not return a fixed/cached value -- the property that
        guarantees run_expanding_window_cv's per-fold naive baselines are
        genuinely fold-specific, never a constant reused across folds."""
        # Fold A: high-scoring games, naive rolling prediction close to true.
        features_a = pd.DataFrame(
            {
                "home_team_off_eff_L10": [118.0, 122.0, 115.0, 120.0],
                "away_team_off_eff_L10": [110.0, 108.0, 112.0, 109.0],
            }
        )
        y_a = pd.DataFrame({"PTS_home": [120, 119, 116, 123], "PTS_away": [108, 110, 111, 107]})

        # Fold B: low-scoring games, deliberately different naive prediction
        # error pattern (larger gaps between predicted and true).
        features_b = pd.DataFrame(
            {
                "home_team_off_eff_L10": [95.0, 90.0, 98.0, 92.0],
                "away_team_off_eff_L10": [93.0, 96.0, 91.0, 94.0],
            }
        )
        y_b = pd.DataFrame({"PTS_home": [130, 85, 140, 80], "PTS_away": [80, 130, 82, 138]})

        result_a = naive_baseline_metrics(features_a, y_a, window=10)
        result_b = naive_baseline_metrics(features_b, y_b, window=10)

        assert result_a["diff_mae"] != result_b["diff_mae"]
        assert result_a["total_mae"] != result_b["total_mae"]

    def test_naive_baseline_drops_cold_start_nan_rows_not_propagate(self):
        """A team's first game or two of a new season has no same-season
        rolling history yet (off_eff_L{window} is NaN) -- those rows must be
        EXCLUDED from the mean, not silently NaN-poison every metric via
        numpy's default mean() behavior."""
        features = pd.DataFrame(
            {
                "home_team_off_eff_L10": [118.0, float("nan"), 115.0],
                "away_team_off_eff_L10": [110.0, 108.0, 112.0],
            }
        )
        y = pd.DataFrame({"PTS_home": [120, 119, 116], "PTS_away": [108, 110, 111]})

        result = naive_baseline_metrics(features, y, window=10)

        assert result["diff_mae"] == result["diff_mae"]  # not NaN
        # Computed from exactly the 2 valid rows (index 0, 2):
        # diff_true = [12, 5], diff_pred = [8, 3] -> abs err = [4, 2] -> mean 3.0
        assert result["diff_mae"] == pytest.approx(3.0)


def _write_minimal_game_db(path, rows):
    """rows: list of (game_id, game_date, team_home, team_away). Minimal
    schema matching NBADataLoader._GAME_SELECT's required columns."""
    conn = sqlite3.connect(str(path))
    conn.execute("""CREATE TABLE game (
            game_id TEXT PRIMARY KEY, game_date TEXT, season_id TEXT, season_type TEXT,
            team_id_home INTEGER, team_id_away INTEGER, pts_home REAL, pts_away REAL, wl_home TEXT,
            fg_pct_home REAL, ft_pct_home REAL, fg3_pct_home REAL, ast_home INTEGER, reb_home INTEGER,
            fg_pct_away REAL, ft_pct_away REAL, fg3_pct_away REAL, ast_away INTEGER, reb_away INTEGER,
            fgm_home INTEGER, fga_home INTEGER, fg3m_home INTEGER, fg3a_home INTEGER,
            ftm_home INTEGER, fta_home INTEGER,
            fgm_away INTEGER, fga_away INTEGER, fg3m_away INTEGER, fg3a_away INTEGER,
            ftm_away INTEGER, fta_away INTEGER
        )""")
    for game_id, game_date, home, away in rows:
        conn.execute(
            "INSERT INTO game (game_id, game_date, season_id, season_type, team_id_home, team_id_away, "
            "pts_home, pts_away, wl_home) VALUES (?, ?, '22024', 'Regular Season', ?, ?, 100, 90, 'W')",
            (game_id, game_date, home, away),
        )
    conn.commit()
    conn.close()


class TestLoadTrainingDataWarmStartContext:
    """Regression test for the warm-start fix: val_df/test_df must include
    raw games from BEFORE their own split's start (all the way back to
    data_start_date, same as train_df already did) so rolling/aggregate
    features computed on them have real historical context to draw on,
    instead of restarting cold at each split boundary. This is not leakage --
    every included row is still strictly before that row's own game; it's the
    opposite bug (discarding legitimate point-in-time history)."""

    def test_val_and_test_df_reach_back_to_data_start_date(self, tmp_path):
        from src.data_processing.data_loader import load_training_data

        db_path = tmp_path / "nba_api.sqlite"
        _write_minimal_game_db(
            db_path,
            [
                ("g1", "2018-11-01", 1610612737, 1610612738),  # well before train_end
                ("g2", "2020-03-01", 1610612737, 1610612738),  # train period
                ("g3", "2020-11-01", 1610612737, 1610612738),  # val period
                ("g4", "2021-11-01", 1610612737, 1610612738),  # test period
            ],
        )
        train_df, val_df, test_df = load_training_data(
            db_path=str(db_path),
            train_start_date="2018-10-16",
            train_end_date="2020-08-14",
            val_start_date="2020-12-22",
            val_end_date="2021-05-16",
            test_start_date="2021-10-19",
            test_end_date="2022-04-10",
            allowed_season_types=["Regular Season"],
            data_start_date="2016-10-01",
            context_season_types=["Regular Season"],
        )
        # Before the fix, val_df/test_df started at train_end_date/val_end_date
        # respectively, so g1 (2018) and g2 (2020-03, training-period) would
        # NEVER appear in val_df or test_df -- only the post-fix behavior
        # includes them as legitimate warm-start context.
        assert (
            "g1" in val_df["GAME_ID"].values
        ), "val_df must reach back to data_start_date, not train_end_date"
        assert "g2" in val_df["GAME_ID"].values, "val_df must include training-period games as context"
        assert "g1" in test_df["GAME_ID"].values, "test_df must reach back to data_start_date"
        assert "g3" in test_df["GAME_ID"].values, "test_df must include val-period games as context"
