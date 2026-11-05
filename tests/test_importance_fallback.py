"""Injured players with no current-season player_importance row (out since before
the season, or traded in and not yet played) must score from their latest earlier
snapshot when the fallback flag is on, and stay at 0 importance when it is off."""

import os
import sqlite3

import pytest

# pipeline imports the LLM extractor, which builds a genai client at import time.
# No call is made here; a placeholder key just lets the import succeed.
os.environ.setdefault("GOOGLE_API_KEY", "test-placeholder")

from src.news_scraping import pipeline
from src.utils.config_loader import load_config

LAC, TOR = 1610612746, 1610612761


@pytest.fixture
def db(tmp_path):
    path = str(tmp_path / "inj.sqlite")
    conn = sqlite3.connect(path)
    conn.execute(
        "CREATE TABLE player_importance (player_id INTEGER, player_name TEXT, team_id INTEGER, "
        "as_of_date TEXT, minutes_per_game REAL, pts_per_game REAL, usage_rate REAL, updated_at TEXT)"
    )
    rows = [
        (1, "Healthy A", LAC, "2026-10-22", 30.0, 20.0, 0.25, "x"),
        (2, "Healthy B", LAC, "2026-10-22", 30.0, 10.0, 0.20, "x"),
        # Ingram: only a prior-season row, under another team
        (3, "Star Out", TOR, "2026-05-27", 33.8, 21.5, 0.28, "x"),
    ]
    conn.executemany("INSERT INTO player_importance VALUES (?,?,?,?,?,?,?,?)", rows)
    conn.commit()
    conn.close()
    return path


def _set_flag(monkeypatch, value):
    cfg = load_config()
    cfg.injury_features.importance_prior_stats_fallback = value
    monkeypatch.setattr(pipeline, "load_config", lambda: cfg)


def test_flag_off_unlisted_player_absent(db, monkeypatch):
    _set_flag(monkeypatch, False)
    m = pipeline._get_importance_map(db, LAC, "2026-10-30", "2026-10-21", ["Star Out"])
    assert "Star Out" not in m


def test_flag_on_scores_from_prior_snapshot(db, monkeypatch):
    _set_flag(monkeypatch, True)
    m = pipeline._get_importance_map(db, LAC, "2026-10-30", "2026-10-21", ["Star Out"])
    assert m["Star Out"] > 0.2
    assert set(m) >= {"Healthy A", "Healthy B", "Star Out"}


def test_flag_on_does_not_override_current_season_row(db, monkeypatch):
    _set_flag(monkeypatch, True)
    base = pipeline._get_importance_map(db, LAC, "2026-10-30", "2026-10-21")
    m = pipeline._get_importance_map(db, LAC, "2026-10-30", "2026-10-21", ["Healthy A"])
    assert m["Healthy A"] == base["Healthy A"]


def test_prior_snapshot_must_predate_game(db, monkeypatch):
    _set_flag(monkeypatch, True)
    m = pipeline._get_importance_map(db, LAC, "2026-05-01", "2025-10-21", ["Star Out"])
    assert "Star Out" not in m
