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
3. **Model**: XGBoost/LightGBM for point differential → Convert to scores

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
