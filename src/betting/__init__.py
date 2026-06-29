"""
Betting Module for NBA Score Prediction
========================================

Provides utilities for integrating Vegas odds and betting data
into NBA score prediction models.
"""

from src.betting.vegas_analyzer import fetch_vegas_odds, analyze_vegas_accuracy

__all__ = [
    'fetch_vegas_odds',
    'analyze_vegas_accuracy',
]
