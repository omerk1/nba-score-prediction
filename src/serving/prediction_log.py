"""
Persists every recommendation the daily job delivers, so live picks can
later be graded against final scores (the Telegram message and the
30-day-pruned job logs were otherwise the only record).

Lives in its own additive database (data/raw/predictions.sqlite), not the
shared game DBs -- same rationale as the other additive sqlite files: the
serving path never writes into source data.

One row per (game_date, home_team_id, away_team_id). A deliberate same-day
re-run (daily job --force) replaces that day's row, so the table holds the
prediction that was actually delivered last. Headline numbers get their own
columns for easy querying; the full recommend_game() dict (market lines,
edges, margin heatmap) is kept as JSON in rec_json.
"""

import datetime
import json
import logging
import sqlite3
from pathlib import Path

logger = logging.getLogger(__name__)

DB_PATH = Path("data/raw/predictions.sqlite")
TRAINING_METADATA_PATH = Path("data/models/training_metadata.json")

# Headline columns copied straight out of the recommendation dict; any
# that a given pick didn't produce (e.g. no total line extracted) stay NULL.
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

_SCHEMA = f"""
CREATE TABLE IF NOT EXISTS predictions (
    game_date         TEXT NOT NULL,
    home_team_id      INTEGER NOT NULL,
    away_team_id      INTEGER NOT NULL,
    {", ".join(f"{c} REAL" for c in REC_COLUMNS)},
    model_trained_at  TEXT,   -- training_metadata.json's train_date: which model made this
    created_at        TEXT NOT NULL,
    rec_json          TEXT NOT NULL,
    PRIMARY KEY (game_date, home_team_id, away_team_id)
)
"""


def _model_trained_at() -> str | None:
    try:
        return json.loads(TRAINING_METADATA_PATH.read_text()).get("train_date")
    except (OSError, ValueError):
        return None


def save_predictions(
    recs: list[dict],
    game_date: datetime.date | None = None,
    db_path: Path | None = None,
) -> int:
    """Upserts one row per recommendation; returns the number written."""
    if not recs:
        return 0
    db_path = db_path or DB_PATH
    game_date = (game_date or datetime.date.today()).isoformat()
    model_trained_at = _model_trained_at()
    created_at = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")

    rows = [
        (
            game_date,
            rec["home_team_id"],
            rec["away_team_id"],
            *(rec.get(c) for c in REC_COLUMNS),
            model_trained_at,
            created_at,
            json.dumps(rec),
        )
        for rec in recs
    ]

    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    try:
        conn.execute(_SCHEMA)
        placeholders = ", ".join("?" * len(rows[0]))
        conn.executemany(f"INSERT OR REPLACE INTO predictions VALUES ({placeholders})", rows)
        conn.commit()
    finally:
        conn.close()
    logger.info(f"saved {len(rows)} prediction(s) for {game_date} to {db_path}")
    return len(rows)
