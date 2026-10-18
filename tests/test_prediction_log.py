import datetime
import json

import numpy as np
import pandas as pd
import pytest

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


@pytest.fixture(autouse=True)
def metadata(tmp_path, monkeypatch):
    meta = tmp_path / "meta.json"
    meta.write_text(json.dumps({"train_date": "2026-09-25 07:08 UTC"}))
    monkeypatch.setattr(prediction_log, "TRAINING_METADATA_PATH", meta)
    return meta


def test_writes_headline_columns_and_full_json(tmp_path):
    log = tmp_path / "sub" / "predictions_log.csv"

    n = save_predictions(
        [_rec(1, 2, total_line=221.5, over_odds=1.8, under_odds=1.7), _rec(3, 4)],
        game_date=DAY,
        log_path=log,
    )

    assert n == 2
    df = pd.read_csv(log)
    assert list(df.columns) == list(prediction_log.COLUMNS)
    first, second = df.to_dict("records")
    assert first["game_date"] == "2026-10-14"
    assert first["predicted_diff"] == 3.5
    assert first["home_win_probability"] == 0.61
    assert first["total_line"] == 221.5
    assert (first["over_odds"], first["under_odds"]) == (1.8, 1.7)
    assert pd.isna(second["total_line"])  # market field absent for that pick
    assert first["model_trained_at"] == "2026-09-25 07:08 UTC"
    assert json.loads(first["rec_json"])["margin_heatmap"] == {"3": 0.04, "-2": 0.03}


def test_reruns_append_never_replace(tmp_path):
    log = tmp_path / "predictions_log.csv"
    save_predictions([_rec(1, 2, diff=3.5)], game_date=DAY, log_path=log)
    save_predictions([_rec(1, 2, diff=-1.0)], game_date=DAY, log_path=log)

    df = pd.read_csv(log)
    assert df["predicted_diff"].tolist() == [3.5, -1.0]
    assert (df["game_date"] == "2026-10-14").all()


def test_mismatched_header_refuses_to_append(tmp_path):
    log = tmp_path / "predictions_log.csv"
    log.write_text("game_date,something_else\n2026-10-13,1\n")
    with pytest.raises(ValueError, match="header"):
        save_predictions([_rec(1, 2)], game_date=DAY, log_path=log)
    assert log.read_text() == "game_date,something_else\n2026-10-13,1\n"


def test_missing_metadata_and_empty_input(tmp_path, monkeypatch):
    monkeypatch.setattr(prediction_log, "TRAINING_METADATA_PATH", tmp_path / "missing.json")
    log = tmp_path / "predictions_log.csv"
    assert save_predictions([], game_date=DAY, log_path=log) == 0
    assert not log.exists()
    save_predictions([_rec(1, 2)], game_date=DAY, log_path=log)
    assert pd.isna(pd.read_csv(log)["model_trained_at"][0])
