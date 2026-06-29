"""Database migrations for NBA Score Prediction."""

from pathlib import Path


def get_migrations_dir() -> Path:
    """Get the migrations directory path."""
    return Path(__file__).parent
