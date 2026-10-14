"""
Persists every recommendation the daily job delivers, so live picks can
later be graded against final scores (the Telegram message and the
30-day-pruned job logs were otherwise the only record).

Unlike the project's sqlite caches, this record can't be rebuilt after the
fact (that day's model, data and market lines are gone), so it's a small
git-tracked CSV (whitelisted in .gitignore), same as outputs/experiments_v2.csv.

Append-only: one row per delivered recommendation, never rewritten. A
same-day re-run (daily job --force) appends new rows for the same games;
created_at tells them apart. Headline numbers get their own columns; the
full recommend_game() dict (market lines, edges, margin heatmap) is kept as
JSON in rec_json.
"""

import csv
import datetime
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

LOG_PATH = Path("outputs/predictions_log.csv")
TRAINING_METADATA_PATH = Path("data/models/training_metadata.json")

# Headline fields copied straight out of the recommendation dict; any that
# a given pick didn't produce (e.g. no total line extracted) stay empty.
REC_COLUMNS = (
    "predicted_home_score",
    "predicted_away_score",
    "predicted_diff",
    "predicted_total",
    "home_win_probability",
    "home_spread",
    "home_cover_probability",
    "total_line",
    "over_probability",
)

COLUMNS = (
    "game_date",
    "home_team_id",
    "away_team_id",
    *REC_COLUMNS,
    "model_trained_at",  # training_metadata.json's train_date: which model made this
    "created_at",
    "rec_json",
)


def _model_trained_at() -> str | None:
    try:
        return json.loads(TRAINING_METADATA_PATH.read_text()).get("train_date")
    except (OSError, ValueError):
        return None


def save_predictions(
    recs: list[dict],
    game_date: datetime.date | None = None,
    log_path: Path | None = None,
) -> int:
    """Appends one row per recommendation; returns the number written."""
    if not recs:
        return 0
    log_path = log_path or LOG_PATH
    game_date = (game_date or datetime.date.today()).isoformat()
    model_trained_at = _model_trained_at()
    created_at = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")

    rows = [
        {
            "game_date": game_date,
            "home_team_id": rec["home_team_id"],
            "away_team_id": rec["away_team_id"],
            **{c: rec.get(c) for c in REC_COLUMNS},
            "model_trained_at": model_trained_at,
            "created_at": created_at,
            "rec_json": json.dumps(rec),
        }
        for rec in recs
    ]

    write_header = not log_path.exists() or log_path.stat().st_size == 0
    if not write_header:
        # Appending under a different header would silently misalign every
        # new row; fail loudly instead (the job turns this into a notice).
        with open(log_path, newline="") as f:
            existing = tuple(next(csv.reader(f), ()))
        if existing != COLUMNS:
            raise ValueError(f"{log_path} header doesn't match the current columns; not appending")

    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)
    logger.info(f"appended {len(rows)} prediction(s) for {game_date} to {log_path}")
    return len(rows)
