"""
Player box score projections module for NBA Score Prediction.

This module provides player-level scoring projections using historical
rolling averages from player game logs.
"""

from .player_projections import project_game_contributions

__all__ = ["project_game_contributions"]
