# NBA Score Prediction Project

Predicting NBA game scores with focus on point differential accuracy using statistical features and news analysis.

## 🎯 Project Goal

Predict exact scores of NBA games, prioritizing accurate point differential over absolute score proximity.

**Example:** If the actual score is Home 108 - Away 101 (diff: +7):
- ✅ Better: Home 98 - Away 91 (diff: +7)
- ❌ Worse: Home 106 - Away 104 (diff: +2)

## 🏗️ Architecture

**Approach:** News-Driven Feature Engineering + Gradient Boosting

1. **Statistical Features**: Team performance, rolling averages, rest days, matchups
2. **News Features**: Injury reports, momentum, team chemistry (LLM-extracted)
3. **Model**: CatBoost/LightGBM for point differential → Convert to scores

## 📊 Data Sources

- **Historical Stats**: [Kaggle NBA Database](https://www.kaggle.com/datasets/wyattowalsh/basketball)
- **News**: NBA.com, ESPN, Reddit r/nba, Basketball-Reference

## 🚀 Quick Start

### Prerequisites
```bash
python >= 3.9
pip or conda
```

### Installation
```bash
# Clone the repository
git clone <your-repo-url>
cd nba-score-prediction

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Download Data
1. Download the [Kaggle NBA dataset](https://www.kaggle.com/datasets/wyattowalsh/basketball)
2. Place `basketball.sqlite` in `data/raw/`


## 📁 Project Structure

```
nba-score-prediction/
├── data/
│   ├── raw/              # Original data (not in git)
│   ├── processed/        # Cleaned data
│   ├── features/         # Engineered features
│   └── models/           # Trained models
├── src/
│   ├── data_processing/  # Data loading and cleaning
│   ├── feature_engineering/  # Statistical & news features
│   ├── models/           # Model training and evaluation
│   ├── news_scraping/    # News collection scripts
│   └── utils/            # Helper functions
├── notebooks/            # Jupyter notebooks for exploration
├── tests/                # Unit tests
├── configs/              # Configuration files
├── outputs/              # Predictions and reports
└── docs/                 # Documentation
```

## 🩹 Injury Features Pipeline

Player injury data is extracted via LLM and stored in a local SQLite DB, then joined into the model features at training time. The feature is toggled via `configs/config.yaml` (`injury_features.enabled`).

### Setup

```bash
# 1. Get a free Gemini API key at aistudio.google.com (takes 30 seconds)
cp .env.example .env
# Edit .env and add: GOOGLE_API_KEY=your-key-here
```

### Running the backfill

Steps must run **in order**. All steps are **resumable** — they use `INSERT OR REPLACE`, so you can stop and restart at any point without duplicating work.

```bash
# Step 1: Build player importance scores from nba_api (~10 min, no LLM)
python build_injury_features.py --run build_player_importance

# Step 2: Backfill historical injury features (LLM-based, takes time — see note below)
python build_injury_features.py --run backfill_historical_injuries

# Step 3 (daily cron): Fetch today's injuries before game time
python build_injury_features.py --run nightly_update
```

**Test first on a small range** before committing to the full backfill:
```bash
python build_injury_features.py --run backfill_historical_injuries --start 2023-01-01 --end 2023-01-14
```

**Resuming** after hitting the daily API limit — just move `--start` forward:
```bash
python build_injury_features.py --run backfill_historical_injuries --start 2021-03-01
```

### Rate limits (Gemini free tier)

| Limit | Value | Impact |
|-------|-------|--------|
| Requests/minute | 15 RPM | ~4.3s sleep after each call (auto-managed) |
| Requests/day | 1,500 | Full backfill (8 seasons) takes ~14 days |

Internally the scraper fetches all transactions **once per season**, not once per date, so the bottleneck is the LLM cap — not scraping speed.

### Enabling the feature

Once the DB is populated, set `injury_features.enabled: true` in `configs/config.yaml` and retrain.

---

## 🔄 Development Workflow

1. **Local**: Data exploration, feature engineering, experimentation
2. **Colab** (when needed): LLM feature extraction, GPU-intensive tasks
3. **Local**: Model training, inference, iteration

## 📈 Roadmap

- [x] Project setup
- [ ] Phase 1: Data exploration & statistical features
- [ ] Phase 2: News scraping & LLM feature extraction
- [ ] Phase 3: Baseline model training
- [ ] Phase 4: Model optimization & evaluation
- [ ] Phase 5: Production pipeline

## 📝 License

MIT License - Feel free to use and modify

## 🤝 Contributing

This is a personal project, but suggestions and improvements are welcome!

---

**Note**: This project is designed to run locally with optional GPU usage via Google Colab for LLM inference.
