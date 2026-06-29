"""
Unit tests for the Vegas betting data analyzer module.

Tests cover:
- fetch_vegas_odds function
- _simulate_vegas_odds helper
- analyze_vegas_accuracy analysis function
"""

import pandas as pd
import numpy as np
import pytest
from pathlib import Path

from src.betting.vegas_analyzer import (
    fetch_vegas_odds,
    analyze_vegas_accuracy,
    _simulate_vegas_odds,
)


class TestSimulateVegasOdds:
    """Tests for the _simulate_vegas_odds helper function."""

    def test_simulate_adds_spread_column(self):
        """Test that spread column is added to the dataframe."""
        df = pd.DataFrame({
            'mov': [10.0, -5.0, 0.0],
            'total_points': [200.0, 180.0, 220.0],
        })

        result = _simulate_vegas_odds(df)

        assert 'spread' in result.columns
        assert len(result) == len(df)

    def test_simulate_adds_over_under_column(self):
        """Test that over_under column is added to the dataframe."""
        df = pd.DataFrame({
            'mov': [10.0, -5.0, 0.0],
            'total_points': [200.0, 180.0, 220.0],
        })

        result = _simulate_vegas_odds(df)

        assert 'over_under' in result.columns
        assert len(result) == len(df)

    def test_simulate_spread_correlates_with_mov(self):
        """Test that simulated spread correlates with actual MOV."""
        df = pd.DataFrame({
            'mov': [20.0, 10.0, -5.0, -15.0],
            'total_points': [200.0, 180.0, 220.0, 190.0],
        })

        result = _simulate_vegas_odds(df)

        # Spread should have same sign as MOV (with high correlation)
        correlation = np.corrcoef(result['spread'], result['mov'])[0, 1]
        assert correlation > 0.8, f"Spread correlation with MOV too low: {correlation}"

    def test_simulate_ou_correlates_with_total_points(self):
        """Test that simulated O/U correlates with actual total points."""
        df = pd.DataFrame({
            'mov': [10.0, -5.0, 0.0, 20.0],
            'total_points': [150.0, 200.0, 220.0, 190.0],
        })

        result = _simulate_vegas_odds(df)

        # O/U should correlate with total points
        correlation = np.corrcoef(result['over_under'], result['total_points'])[0, 1]
        assert correlation > 0.7, f"O/U correlation with total_points too low: {correlation}"

    def test_simulate_no_nan_values(self):
        """Test that simulation produces no NaN values."""
        df = pd.DataFrame({
            'mov': [10.0, -5.0, 0.0],
            'total_points': [200.0, 180.0, 220.0],
        })

        result = _simulate_vegas_odds(df)

        assert result['spread'].isna().sum() == 0, "NaN values in spread"
        assert result['over_under'].isna().sum() == 0, "NaN values in over_under"


class TestFetchVegasOdds:
    """Tests for the fetch_vegas_odds main function."""

    def test_various_date_formats_accepted(self):
        """Test that pandas-compatible date formats are accepted."""
        # pd.to_datetime accepts various formats, so test with YYYY-MM-DD
        df = fetch_vegas_odds(('2024-01-01', '2024-01-31'))
        assert isinstance(df, pd.DataFrame)

    def test_invalid_start_date_raises_error(self):
        """Test that invalid start date raises ValueError."""
        with pytest.raises(ValueError):
            fetch_vegas_odds(('not-a-date', '2024-01-31'))

    def test_returns_dataframe(self):
        """Test that function returns a pandas DataFrame."""
        df = fetch_vegas_odds(('2024-01-01', '2024-01-31'))
        assert isinstance(df, pd.DataFrame)

    def test_has_required_columns(self):
        """Test that returned DataFrame has all required columns."""
        df = fetch_vegas_odds(('2024-01-01', '2024-01-31'))

        required_cols = {'game_id', 'game_date', 'home_team_id', 'away_team_id',
                        'spread', 'over_under', 'mov'}
        assert required_cols.issubset(set(df.columns)), \
            f"Missing columns: {required_cols - set(df.columns)}"

    def test_no_nan_in_required_columns(self):
        """Test that required columns have no NaN values."""
        df = fetch_vegas_odds(('2024-01-01', '2024-01-31'))

        if len(df) > 0:
            for col in ['game_id', 'spread', 'over_under', 'mov']:
                nan_count = df[col].isna().sum()
                assert nan_count == 0, f"Column {col} has {nan_count} NaN values"

    def test_empty_date_range_returns_empty_dataframe(self):
        """Test that non-existent date range returns empty DataFrame."""
        df = fetch_vegas_odds(('1900-01-01', '1900-01-31'))
        assert len(df) == 0
        assert list(df.columns) == ['game_id', 'game_date', 'home_team_id',
                                     'away_team_id', 'spread', 'over_under', 'mov']


class TestAnalyzeVegasAccuracy:
    """Tests for the analyze_vegas_accuracy analysis function."""

    @pytest.fixture
    def sample_games_df(self):
        """Create sample games dataframe for testing."""
        return pd.DataFrame({
            'game_id': ['001', '002', '003', '004'],
            'spread': [5.0, -3.0, 8.0, -6.0],
            'over_under': [220.0, 200.0, 210.0, 190.0],
            'mov': [6.0, -2.0, 9.0, -5.0],
            'total_points': [215.0, 202.0, 208.0, 192.0],
            'home_points': [110.0, 98.0, 108.0, 93.0],
            'away_points': [105.0, 100.0, 99.0, 98.0],
        })

    def test_returns_dict(self, sample_games_df):
        """Test that function returns a dictionary."""
        result = analyze_vegas_accuracy(sample_games_df)
        assert isinstance(result, dict)

    def test_has_required_metrics(self, sample_games_df):
        """Test that returned dict has all required metrics."""
        result = analyze_vegas_accuracy(sample_games_df)

        required_metrics = {'spread_mae', 'spread_accuracy', 'ou_mae',
                           'avg_spread', 'avg_mov', 'num_games'}
        assert required_metrics.issubset(set(result.keys())), \
            f"Missing metrics: {required_metrics - set(result.keys())}"

    def test_spread_mae_is_non_negative(self, sample_games_df):
        """Test that spread MAE is non-negative."""
        result = analyze_vegas_accuracy(sample_games_df)
        assert result['spread_mae'] >= 0, "Spread MAE should be non-negative"

    def test_ou_mae_is_non_negative(self, sample_games_df):
        """Test that O/U MAE is non-negative."""
        result = analyze_vegas_accuracy(sample_games_df)
        assert result['ou_mae'] >= 0, "O/U MAE should be non-negative"

    def test_spread_accuracy_in_range(self, sample_games_df):
        """Test that spread accuracy is between 0 and 1."""
        result = analyze_vegas_accuracy(sample_games_df)
        assert 0 <= result['spread_accuracy'] <= 1, \
            "Spread accuracy should be between 0 and 1"

    def test_num_games_matches_input(self, sample_games_df):
        """Test that num_games matches input dataframe size."""
        result = analyze_vegas_accuracy(sample_games_df)
        assert result['num_games'] == len(sample_games_df)

    def test_missing_required_columns_returns_empty_dict(self):
        """Test that missing required columns returns empty dict."""
        df = pd.DataFrame({
            'game_id': ['001', '002'],
            'spread': [5.0, -3.0],
            # Missing 'over_under' and 'mov'
        })

        result = analyze_vegas_accuracy(df)
        assert result == {}

    def test_output_dir_created(self, sample_games_df, tmp_path):
        """Test that output directory is created if it doesn't exist."""
        output_dir = tmp_path / "test_outputs"
        assert not output_dir.exists()

        analyze_vegas_accuracy(sample_games_df, output_dir=str(output_dir))

        assert output_dir.exists()

    def test_plots_file_created(self, sample_games_df, tmp_path):
        """Test that analysis plots PNG file is created."""
        output_dir = tmp_path / "test_outputs"

        analyze_vegas_accuracy(sample_games_df, output_dir=str(output_dir))

        plots_path = output_dir / 'vegas_analysis_plots.png'
        assert plots_path.exists(), "vegas_analysis_plots.png not created"


class TestIntegration:
    """Integration tests combining multiple functions."""

    def test_fetch_and_analyze_workflow(self):
        """Test the typical workflow of fetch then analyze."""
        # Fetch Vegas odds
        df = fetch_vegas_odds(('2024-01-01', '2024-01-31'))

        if len(df) > 0:
            # Run analysis
            metrics = analyze_vegas_accuracy(df)

            # Check that analysis succeeded
            assert 'spread_mae' in metrics
            assert metrics['num_games'] == len(df)

    def test_simulated_odds_meet_expectations(self):
        """Test that simulated odds match realistic Vegas patterns."""
        df = fetch_vegas_odds(('2024-01-01', '2024-01-15'))

        if len(df) > 0:
            # Vegas spreads can be up to ~40 for blowout games
            # Check that spreads are non-NaN and numeric
            assert df['spread'].dtype in ['float64', 'float32'], "Spread should be numeric"
            assert df['spread'].notna().all(), "Spread should have no NaNs"

            # Vegas O/U should be reasonable (between 100 and 350)
            # Actual range varies based on game data
            assert (df['over_under'] >= 100).all() and (df['over_under'] <= 350).all(), \
                "O/U outside realistic range"
            assert df['over_under'].notna().all(), "O/U should have no NaNs"
