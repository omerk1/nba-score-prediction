import datetime
import json
import sqlite3

import numpy as np

from src.serving import prediction_log
from src.serving.prediction_log import save_predictions

DAY = datetime.date(2026, 10, 14)


def _rec(home, away, diff=3.5, **extra):
    return {
        "home_team_id": home,
        "away_team_id": away,
        "predicted_home_score": 112.0,
        "predicted_away_score": 108.5,
        "predicted_diff": diff,
        "predicted_total": 220.5,
        "home_win_probability": np.float64(0.61),
        "margin_heatmap": {3: 0.04, -2: 0.03},
        **extra,
    }


def _rows(db):
    conn = sqlite3.connect(db)
    conn.row_factory = sqlite3.Row
    try:
        return [dict(r) for r in conn.execute("SELECT * FROM predictions ORDER BY home_team_id")]
    finally:
        conn.close()


def test_writes_headline_columns_and_full_json(tmp_path, monkeypatch):
    meta = tmp_path / "meta.json"
    meta.write_text(json.dumps({"train_date": "2026-09-25 07:08 UTC"}))
    monkeypatch.setattr(prediction_log, "TRAINING_METADATA_PATH", meta)
    db = tmp_path / "sub" / "predictions.sqlite"

    n = save_predictions([_rec(1, 2, total_line=221.5), _rec(3, 4)], game_date=DAY, db_path=db)

    assert n == 2
    first, second = _rows(db)
    assert first["game_date"] == "2026-10-14"
    assert first["predicted_diff"] == 3.5
    assert first["home_win_probability"] == 0.61
    assert first["total_line"] == 221.5
    assert second["total_line"] is None  # market field absent for that pick
    assert first["model_trained_at"] == "2026-09-25 07:08 UTC"
    assert json.loads(first["rec_json"])["margin_heatmap"] == {"3": 0.04, "-2": 0.03}


def test_same_day_rerun_replaces_not_duplicates(tmp_path):
    db = tmp_path / "predictions.sqlite"
    save_predictions([_rec(1, 2, diff=3.5)], game_date=DAY, db_path=db)
    save_predictions([_rec(1, 2, diff=-1.0)], game_date=DAY, db_path=db)
    save_predictions([_rec(1, 2, diff=2.0)], game_date=DAY + datetime.timedelta(days=1), db_path=db)

    rows = _rows(db)
    assert len(rows) == 2
    assert {r["game_date"]: r["predicted_diff"] for r in rows} == {
        "2026-10-14": -1.0,
        "2026-10-15": 2.0,
    }


def test_missing_metadata_and_empty_input(tmp_path, monkeypatch):
    monkeypatch.setattr(prediction_log, "TRAINING_METADATA_PATH", tmp_path / "missing.json")
    db = tmp_path / "predictions.sqlite"
    assert save_predictions([], game_date=DAY, db_path=db) == 0
    assert not db.exists()
    save_predictions([_rec(1, 2)], game_date=DAY, db_path=db)
    assert _rows(db)[0]["model_trained_at"] is None
