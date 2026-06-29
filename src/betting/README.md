# Vegas Betting Data Module

## Overview

The `betting` module provides utilities for fetching, analyzing, and integrating Vegas odds data (spreads, O/U lines, line movement) into NBA score prediction models.

## Data Sources

### Current Implementation

The current `vegas_analyzer.py` uses **simulated Vegas odds** based on actual game outcomes to demonstrate the data schema and analysis capabilities. The simulation:

- Generates spreads that correlate with actual point differentials
- Creates O/U lines based on total points and game characteristics
- Uses realistic noise and bias patterns observed in Vegas markets
- Enables exploratory analysis and model validation without external API dependencies

### Production Data Sources

To integrate real Vegas odds data, consider these authoritative sources:

#### 1. **pro-data.com** (Recommended)
- **Coverage**: Complete historical Vegas lines for NBA games (1990s–present)
- **Data Points**: Opening lines, closing lines, line movement history
- **Format**: CSV export or API access
- **Cost**: Premium subscription ($100–500/month)
- **Pros**: Most comprehensive, professionally maintained, supports research
- **Cons**: Requires paid subscription
- **Integration**: Would replace `_simulate_vegas_odds()` with API call to pro-data

#### 2. **Sports Reference Archives**
- **Coverage**: Historical betting lines for NBA games
- **Format**: Web scraping from archived pages
- **Cost**: Free
- **Pros**: Public data, no API keys needed
- **Cons**: Requires web scraping, incomplete coverage for older seasons
- **Integration**: BeautifulSoup scraper to fetch lines from Sports Reference

#### 3. **ESPN Betting Data APIs** (Premium)
- **Coverage**: Current and recent season Vegas lines
- **Format**: REST API
- **Cost**: Requires ESPN+ or corporate partnership
- **Pros**: Real-time data, official source
- **Cons**: Limited historical access, requires API key
- **Integration**: Direct API calls with rate limiting

#### 4. **Kaggle Datasets**
- **Coverage**: Varies; some datasets have historical Vegas lines
- **Format**: CSV files
- **Cost**: Free
- **Pros**: One-time download, no ongoing cost
- **Cons**: Coverage gaps, update frequency varies
- **Examples**:
  - "NBA Games Historical Data with Odds" (if available)
  - "Sports Betting Data" (aggregate dataset)
- **Integration**: Direct CSV load with `pd.read_csv()`

#### 5. **Historical CSV Backups**
- **Coverage**: Custom datasets archived from previous seasons
- **Format**: CSV with game_id, spread, over_under columns
- **Cost**: Free (internal)
- **Pros**: No external dependencies, consistent format
- **Cons**: Manual maintenance required
- **Integration**: Load from `data/historical_vegas/` directory

## Module API

### `fetch_vegas_odds(date_range: tuple[str, str]) -> pd.DataFrame`

Fetches Vegas odds for games in the specified date range.

**Parameters:**
- `date_range`: Tuple of (start_date, end_date) as YYYY-MM-DD strings

**Returns:**
```python
DataFrame with columns:
  - game_id: Unique game identifier
  - game_date: Date of game
  - home_team_id: Home team's NBA team ID
  - away_team_id: Away team's NBA team ID
  - spread: Vegas spread (negative = home favored, positive = away favored)
  - over_under: Vegas O/U line (total points threshold)
  - mov: Actual margin of victory (home_points - away_points)
```

**Example:**
```python
from src.betting.vegas_analyzer import fetch_vegas_odds

df = fetch_vegas_odds(('2024-01-01', '2024-01-31'))
print(f"Fetched {len(df)} games")
print(df[['game_id', 'spread', 'over_under']].head())
```

### `analyze_vegas_accuracy(games_df, output_dir=None) -> dict`

Analyzes Vegas prediction accuracy and generates exploratory plots.

**Parameters:**
- `games_df`: DataFrame with Vegas odds and actual results
- `output_dir`: Directory to save plots (default: "outputs")

**Returns:**
```python
Dictionary with metrics:
  - spread_mae: Mean absolute error (Vegas spread vs actual MOV)
  - spread_accuracy: % of games where Vegas picked winner direction correctly
  - ou_mae: Mean absolute error (Vegas O/U vs actual total points)
  - avg_spread: Average Vegas spread
  - avg_mov: Average actual MOV
  - num_games: Number of games analyzed
```

**Output Files:**
- `vegas_analysis_plots.png`: 4-panel visualization of accuracy metrics

**Example:**
```python
metrics = analyze_vegas_accuracy(df, output_dir="outputs")
print(f"Spread MAE: {metrics['spread_mae']:.2f}")
print(f"Directional Accuracy: {metrics['spread_accuracy']:.1%}")
```

## Data Schema

### Input: Game Results (from `data_loader.py`)

| Column        | Type    | Description                      |
|---------------|---------|----------------------------------|
| GAME_ID       | str     | Unique game identifier           |
| GAME_DATE     | datetime| Date of game                     |
| HOME_TEAM_ID  | int     | NBA team ID for home team        |
| AWAY_TEAM_ID  | int     | NBA team ID for away team        |
| PTS_home      | float   | Home team points scored          |
| PTS_away      | float   | Away team points scored          |
| TOTAL_POINTS  | float   | Sum of both teams' points        |
| POINT_DIFF    | float   | Home team points - Away team pts |

### Output: Vegas Odds

| Column        | Type    | Description                                      |
|---------------|---------|--------------------------------------------------|
| game_id       | str     | Unique game identifier                           |
| game_date     | datetime| Date of game                                     |
| home_team_id  | int     | Home team ID                                     |
| away_team_id  | int     | Away team ID                                     |
| spread        | float   | Vegas spread (negative favors home)              |
| over_under    | float   | Vegas O/U line (total points)                    |
| mov           | float   | Actual margin of victory                         |

## Usage in Feature Engineering

**Currently:** Vegas data is NOT integrated into `feature_builder.py`.

**Future Integration:**
Vegas odds can be added as features for score prediction models:

1. **Spread Feature**: `vegas_spread` — direct use or as a calibration target
2. **O/U Mismatch**: `ou_vs_actual` = actual total points - Vegas O/U line
3. **Line Movement**: `spread_movement` = closing line - opening line
4. **Accuracy Metrics**: Features capturing Vegas's historical accuracy on similar matchups

Example integration in `feature_builder.py`:
```python
def _add_vegas_features(self, df: pd.DataFrame) -> pd.DataFrame:
    """Add Vegas odds-based features for score prediction."""
    vegas_df = fetch_vegas_odds((df['GAME_DATE'].min(), df['GAME_DATE'].max()))
    merged = df.merge(vegas_df[['game_id', 'spread', 'over_under']], 
                      left_on='GAME_ID', right_on='game_id', how='left')
    # Feature engineering...
    return merged
```

## Files in This Module

- `vegas_analyzer.py`: Main module with fetching and analysis functions
- `__init__.py`: Package initialization
- `README.md`: This documentation

## Next Steps

1. **Secure Data Source**: Choose and configure one of the production sources above
2. **Replace Simulation**: Update `_simulate_vegas_odds()` to fetch real data
3. **Validate Data Quality**: Ensure date coverage and data completeness
4. **Feature Integration**: Add Vegas features to `feature_builder.py`
5. **Model Evaluation**: Benchmark improvement from Vegas features

## References

- Vegas Spread Basics: https://www.covers.com/betting-learning/spread
- Line Movement Analysis: https://www.statsbomb.com/
- Sports Betting Data Standards: https://www.kaggle.com/
