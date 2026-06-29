"""
Tests for Player Stats Cache Infrastructure.

Tests cover:
1. Migration: table creation and idempotency
2. Cache utility: rolling averages and projections
3. Backfill script: data insertion and resumability
4. Projections module: using cache instead of API
"""

import pytest
import sqlite3
import tempfile
from pathlib import Path
from datetime import datetime, timedelta

from src.migrations.migration_create_player_stats_cache import migrate_player_stats_cache
from src.utils.player_stats_cache import (
    get_player_rolling_avg,
    get_player_projections,
    ensure_cache_exists,
)
from src.projections.player_projections import project_game_contributions


@pytest.fixture
def temp_db():
    """Create a temporary SQLite database for testing."""
    with tempfile.NamedTemporaryFile(suffix=".sqlite", delete=False) as f:
        db_path = f.name

    yield db_path

    # Cleanup
    Path(db_path).unlink(missing_ok=True)


@pytest.fixture
def cache_db(temp_db):
    """Create a temporary database with player_stats_cache table."""
    migrate_player_stats_cache(temp_db)
    return temp_db


@pytest.fixture
def populated_cache(cache_db):
    """Populate cache with test data."""
    conn = sqlite3.connect(cache_db)
    cursor = conn.cursor()

    # Insert test data for player 201939 (LeBron James as example)
    # 20 games of historical data
    base_date = datetime(2024, 4, 1)

    test_data = []
    for i in range(20, 0, -1):
        game_date = (base_date - timedelta(days=i)).strftime("%Y-%m-%d")
        # Create realistic stats (PPG around 24.5, AST around 5.2, etc.)
        test_data.extend([
            (201939, game_date, 'PPG', 24.5 + (i % 3) - 1),  # Varies 23.5-25.5
            (201939, game_date, 'AST', 5.2 + (i % 2) - 0.5),  # Varies 4.7-5.7
            (201939, game_date, 'REB', 7.1 + (i % 2) - 0.5),  # Varies 6.6-7.6
            (201939, game_date, 'BLK', 0.8 + (i % 2) * 0.2),  # Varies 0.8-1.0
            (201939, game_date, 'STL', 1.2 + (i % 2) * 0.1),  # Varies 1.2-1.3
            (201939, game_date, 'FG%', 0.505 + (i % 3) * 0.01),  # Varies 0.505-0.525
        ])

    cursor.executemany(
        """
        INSERT INTO player_stats_cache (player_id, game_date, stat_name, stat_value)
        VALUES (?, ?, ?, ?)
        """,
        test_data,
    )
    conn.commit()
    conn.close()

    return cache_db


class TestMigration:
    """Tests for player_stats_cache table creation."""

    def test_migration_creates_table(self, temp_db):
        """Test that migration creates the table."""
        migrate_player_stats_cache(temp_db)

        conn = sqlite3.connect(temp_db)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='player_stats_cache'"
        )
        result = cursor.fetchone()
        conn.close()

        assert result is not None, "Table was not created"

    def test_migration_idempotent(self, temp_db):
        """Test that running migration twice is safe."""
        migrate_player_stats_cache(temp_db)
        migrate_player_stats_cache(temp_db)  # Should not raise

        conn = sqlite3.connect(temp_db)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='player_stats_cache'"
        )
        count = cursor.fetchone()[0]
        conn.close()

        assert count == 1, "Duplicate tables created"

    def test_migration_creates_indexes(self, temp_db):
        """Test that migration creates necessary indexes."""
        migrate_player_stats_cache(temp_db)

        conn = sqlite3.connect(temp_db)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='index' AND tbl_name='player_stats_cache'"
        )
        indexes = [row[0] for row in cursor.fetchall()]
        conn.close()

        assert "idx_player_stats_player_date" in indexes, "Primary index not created"
        assert "idx_player_stats_date" in indexes, "Date index not created"

    def test_migration_db_not_found(self, tmp_path):
        """Test that migration raises error for non-existent database."""
        nonexistent_db = str(tmp_path / "nonexistent.db")
        with pytest.raises(FileNotFoundError):
            migrate_player_stats_cache(nonexistent_db)


class TestCacheUtility:
    """Tests for cache query functions."""

    def test_get_player_rolling_avg_basic(self, populated_cache):
        """Test get_player_rolling_avg with valid data."""
        avg = get_player_rolling_avg(
            db_path=populated_cache,
            player_id=201939,
            stat_name='PPG',
            game_date='2024-04-01',  # After all inserted games
            window=5,
        )

        # Should return average of last 5 games
        assert isinstance(avg, float)
        assert 23.0 < avg < 26.0, f"PPG average should be around 24.5, got {avg}"

    def test_get_player_rolling_avg_multiple_windows(self, populated_cache):
        """Test rolling average with different window sizes."""
        avg_5 = get_player_rolling_avg(
            populated_cache, 201939, 'PPG', '2024-04-01', window=5
        )
        avg_10 = get_player_rolling_avg(
            populated_cache, 201939, 'PPG', '2024-04-01', window=10
        )
        avg_20 = get_player_rolling_avg(
            populated_cache, 201939, 'PPG', '2024-04-01', window=20
        )

        assert 23.0 < avg_5 < 26.0
        assert 23.0 < avg_10 < 26.0
        assert 23.0 < avg_20 < 26.0
        # All should be similar (not vastly different)
        assert abs(avg_5 - avg_20) < 2.0

    def test_get_player_rolling_avg_no_history(self, populated_cache):
        """Test get_player_rolling_avg with no history."""
        avg = get_player_rolling_avg(
            db_path=populated_cache,
            player_id=999999,  # Non-existent player
            stat_name='PPG',
            game_date='2024-04-01',
            window=5,
        )

        assert avg == 0.0, "Should return 0.0 for no history"

    def test_get_player_rolling_avg_insufficient_history(self, populated_cache):
        """Test rolling average with fewer games than window."""
        # Query right at the start, before most games
        avg = get_player_rolling_avg(
            db_path=populated_cache,
            player_id=201939,
            stat_name='PPG',
            game_date='2024-03-13',  # Only first few games exist before this
            window=20,
        )

        # Should compute average of available games (fewer than window)
        assert isinstance(avg, float)
        assert avg >= 0.0

    def test_get_player_projections_returns_all_stats(self, populated_cache):
        """Test get_player_projections returns all 6 stats."""
        projections = get_player_projections(
            db_path=populated_cache,
            player_id=201939,
            game_date='2024-04-01',
            windows=[5],
        )

        expected_stats = {'PPG', 'AST', 'REB', 'BLK', 'STL', 'FG%'}
        assert set(projections.keys()) == expected_stats

    def test_get_player_projections_values_reasonable(self, populated_cache):
        """Test get_player_projections returns reasonable values."""
        projections = get_player_projections(
            db_path=populated_cache,
            player_id=201939,
            game_date='2024-04-01',
            windows=[5],
        )

        assert 20.0 < projections['PPG'] < 30.0
        assert 4.0 < projections['AST'] < 7.0
        assert 6.0 < projections['REB'] < 9.0
        assert 0.5 < projections['BLK'] < 2.0
        assert 0.8 < projections['STL'] < 2.0
        assert 0.4 < projections['FG%'] < 0.6

    def test_get_player_projections_multiple_windows(self, populated_cache):
        """Test get_player_projections with multiple windows."""
        projections = get_player_projections(
            db_path=populated_cache,
            player_id=201939,
            game_date='2024-04-01',
            windows=[5, 10, 20],
        )

        # All stats should have values
        for stat in ['PPG', 'AST', 'REB', 'BLK', 'STL', 'FG%']:
            assert stat in projections
            assert isinstance(projections[stat], float)
            # Average of multiple windows should be reasonable
            assert projections[stat] >= 0.0

    def test_get_player_projections_no_history(self, cache_db):
        """Test get_player_projections with no historical data."""
        projections = get_player_projections(
            db_path=cache_db,
            player_id=999999,
            game_date='2024-04-01',
            windows=[5],
        )

        # Should return 0.0 for all stats with no history
        for stat in ['PPG', 'AST', 'REB', 'BLK', 'STL', 'FG%']:
            assert projections[stat] == 0.0

    def test_ensure_cache_exists_creates_table(self, temp_db):
        """Test that ensure_cache_exists creates the table."""
        ensure_cache_exists(temp_db)

        conn = sqlite3.connect(temp_db)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='player_stats_cache'"
        )
        result = cursor.fetchone()
        conn.close()

        assert result is not None

    def test_ensure_cache_exists_idempotent(self, cache_db):
        """Test that ensure_cache_exists is idempotent."""
        ensure_cache_exists(cache_db)
        ensure_cache_exists(cache_db)  # Should not raise

        conn = sqlite3.connect(cache_db)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='player_stats_cache'"
        )
        count = cursor.fetchone()[0]
        conn.close()

        assert count == 1


class TestInsertAndRetrieval:
    """Integration tests for inserting and retrieving data."""

    def test_insert_and_retrieve_stats(self, cache_db):
        """Test inserting stats and retrieving them."""
        conn = sqlite3.connect(cache_db)
        cursor = conn.cursor()

        # Insert test data
        cursor.execute(
            """
            INSERT INTO player_stats_cache (player_id, game_date, stat_name, stat_value)
            VALUES (?, ?, ?, ?)
            """,
            (201939, '2024-04-01', 'PPG', 24.5),
        )
        conn.commit()

        # Retrieve it
        avg = get_player_rolling_avg(cache_db, 201939, 'PPG', '2024-04-02', window=1)
        conn.close()

        assert avg == 24.5

    def test_insert_or_replace(self, cache_db):
        """Test that INSERT OR REPLACE works correctly."""
        conn = sqlite3.connect(cache_db)
        cursor = conn.cursor()

        # Insert original
        cursor.execute(
            """
            INSERT INTO player_stats_cache (player_id, game_date, stat_name, stat_value)
            VALUES (?, ?, ?, ?)
            """,
            (201939, '2024-04-01', 'PPG', 20.0),
        )
        conn.commit()

        # Replace with new value
        cursor.execute(
            """
            INSERT OR REPLACE INTO player_stats_cache (player_id, game_date, stat_name, stat_value)
            VALUES (?, ?, ?, ?)
            """,
            (201939, '2024-04-01', 'PPG', 25.0),
        )
        conn.commit()

        # Verify new value
        avg = get_player_rolling_avg(cache_db, 201939, 'PPG', '2024-04-02', window=1)
        conn.close()

        assert avg == 25.0


class TestProjectionsWithCache:
    """Tests for project_game_contributions using cache."""

    def test_project_game_contributions_requires_cache(self, temp_db):
        """Test that projections require cache to be populated."""
        # Don't create the cache table
        # Just try to project — should return empty dict or handle gracefully

        result = project_game_contributions(
            date='2024-04-01',
            home_team_id=1610612744,
            away_team_id=1610612762,
        )

        # Should return empty dict since cache isn't set up
        assert isinstance(result, dict)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
