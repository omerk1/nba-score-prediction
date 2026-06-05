# NBA Score Prediction

Predicting NBA game scores with emphasis on point differential accuracy using statistical features and injury impact analysis.

## Goal

Predict exact scores of NBA games, prioritizing accurate point differential over absolute score proximity.

**Example:** Actual score — Home 108, Away 101 (diff: +7)
- Better: Home 98, Away 91 (diff: +7)
- Worse: Home 106, Away 104 (diff: +2)

## Architecture

**Approach:** Statistical Feature Engineering + Gradient Boosting

1. **Statistical features** — team performance, rolling averages, rest days, head-to-head matchups
2. **Injury features** — per-game impact scores derived from official NBA injury reports (formula-based or LLM-based)
3. **Model** — CatBoost/LightGBM predicting point differential, converted to final scores

## Quick Start

### Prerequisites

```
Python >= 3.9
```

### Installation

```bash
git clone <your-repo-url>
cd nba-score-prediction
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Fetch data

```bash
python src/data_processing/fetch_data.py
```

Data is pulled from the [nba_api](https://github.com/swar/nba_api) and stored in `data/raw/nba_api.sqlite`. Subsequent runs are incremental — existing rows are skipped via `INSERT OR IGNORE`.

### Train

```bash
python train_model.py --run-name baseline_stats_only
python train_model.py --run-name injury_v1 --notes "with injury features enabled"
```

Every run appends a row to `outputs/experiments.csv` for ablation comparison.

### Predict

```bash
python predict_game.py --home 1610612747 --away 1610612744
python predict_game.py --home 1610612747 --away 1610612744 --date 2026-03-15
```

`--home` and `--away` take numeric NBA team IDs. `--date` defaults to today.

## Project Structure

```
nba-score-prediction/
├── src/
│   ├── data_processing/          # Data fetching and loading (nba_api -> SQLite)
│   ├── feature_engineering/      # Statistical and injury feature construction
│   ├── models/                   # Model training and evaluation
│   ├── news_scraping/            # Injury scrapers, DB, and impact extractors
│   └── utils/                    # Config loading and shared utilities
├── configs/
│   └── config.yaml               # All runtime parameters
├── data/
│   ├── raw/                      # SQLite databases (not in git)
│   ├── processed/                # Cleaned data
│   ├── features/                 # Engineered features
│   └── models/                   # Trained model artifacts
├── notebooks/                    # Exploration notebooks
├── outputs/                      # Predictions and experiments.csv
├── train_model.py                # Training entry point
├── predict_game.py               # Inference entry point
└── build_injury_features.py      # Injury feature backfill pipeline
```

## Injury Features Pipeline

Player injury data is scraped from ESPN and official NBA PDFs, scored for game impact, and joined into the model at training time. Toggled via `injury_features.enabled` in `configs/config.yaml`.

Two scorer modes are available:

| Mode | Speed | API Required | Notes |
|------|-------|-------------|-------|
| `formula` | Fast | No | Deterministic: importance × status weight |
| `llm` | Slower | Yes (Gemini) | Richer context-aware impact scores |

Set `scorer: formula` or `scorer: llm` in `configs/config.yaml`.

### Setup (LLM mode only)

```bash
cp .env.example .env
# Add GOOGLE_API_KEY to .env
```

### Running the backfill

Steps must run in order. All steps are resumable — use `INSERT OR REPLACE`, so stopping and restarting is safe.

```bash
# Step 1: Build player importance scores from nba_api (~10 min, no LLM)
python build_injury_features.py --run build_player_importance

# Step 2: Backfill historical injury features
python build_injury_features.py --run backfill_historical_injuries

# Step 3 (daily): Fetch today's injuries before game time
python build_injury_features.py --run nightly_update
```

Test on a small range before committing to the full backfill:

```bash
python build_injury_features.py --run backfill_historical_injuries --start 2023-01-01 --end 2023-01-14
```

To resume after an interruption, advance `--start`:

```bash
python build_injury_features.py --run backfill_historical_injuries --start 2021-03-01
```

NBA official injury PDFs are available from the 2021-22 season onward (`pdf_era_start` in config). Earlier seasons fall back to ESPN scraping.

### Enable in training

Once the DB is populated, set `injury_features.enabled: true` in `configs/config.yaml` and retrain.

## Configuration

All parameters are in `configs/config.yaml`:

- **Data date ranges** — train/validation/test splits by season
- **Rolling window sizes** — for team averages and head-to-head features
- **Model hyperparameters**
- **Injury feature settings** — scorer mode, DB path, API rate limits, importance weights

## License

MIT
