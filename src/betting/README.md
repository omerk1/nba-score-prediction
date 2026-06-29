# Betting Data Analysis

This module analyzes Polymarket OVER/UNDER odds for NBA games to support score prediction.

## Data Source: Polymarket

**Polymarket** is a decentralized prediction market platform where users trade shares reflecting probabilities of future events. For NBA games, markets typically offer OVER/UNDER spreads on total game points.

### Polymarket Odds Format
- **OVER**: Probability that total points exceed the line (e.g., > 215.5)
- **UNDER**: Probability that total points fall short (automatically 1 - OVER)
- **Volume**: USD amount traded in the market
- **Liquidity**: Available depth for trades

## API Access & Limitations

### Direct API
Polymarket provides a public REST API at `https://api.polymarket.com/markets`, but:
- Limited free access to historical odds
- No built-in NBA game-to-market mapping
- Real-time data requires webhook subscriptions
- Most endpoint access requires authentication

### Alternative Data Sources (Fallback Approaches)

#### 1. **The Graph (Recommended)**
Query Polymarket's subgraph on The Graph:
```bash
# GraphQL endpoint: https://api.thegraph.com/subgraphs/name/polymarket/polymarket
# Query recent orders, market prices, order books
```
Example: Fetch all NBA markets and their historical price movements.

#### 2. **Manual Download + CSV Import**
Steps:
1. Visit https://polymarket.com/markets
2. Filter for NBA games
3. Export historical odds (if available) to CSV
4. Load via `pd.read_csv()` in the analyzer

#### 3. **Third-Party Aggregators**
- **Sports betting aggregators** (e.g., Odds Shark, Covers) sometimes include Polymarket data
- **Sports APIs** like SportsRadar or ESPN might track prediction market odds

#### 4. **Synthetic/Backtest Data (Current)**
When real data is unavailable, the analyzer generates synthetic data:
- Realistic OVER/UNDER distributions (48-52% probability, typical of efficient markets)
- Matching NBA schedule patterns (~1-2 games/day, ~60% game likelihood)
- Right-skewed volume distribution (Exponential)
- Allows for consistent testing and reproducible analysis

## Module Usage

### Basic Usage

```python
from src.betting.polymarket_analyzer import fetch_polymarket_odds, create_exploratory_analysis

# Fetch odds for a date range
df = fetch_polymarket_odds(("2024-01-01", "2024-01-31"))

# Create analysis and visualizations
create_exploratory_analysis(df, output_dir="outputs")
```

### Output Files

When `create_exploratory_analysis()` is called, the module generates:

1. **polymarket_analysis.csv**
   - Columns: game_id, game_date, team_home, team_away, over_line, over_pct, under_pct, volume, liquidity
   - All records with complete odds data

2. **polymarket_odds_distribution.png**
   - Histogram of OVER probabilities
   - Histogram of trading volume (log scale)

3. **polymarket_odds_timeseries.png**
   - Scatter plot of odds over time
   - 7-day rolling average overlay

4. **polymarket_volume_vs_odds.png**
   - Correlation between odds and volume
   - Color-coded by liquidity

## DataFrame Schema

```python
game_id : str
    Unique market identifier (e.g., "game_000123")

game_date : datetime64
    Date of the NBA game

team_home : str
    Home team abbreviation (e.g., "LAL")

team_away : str
    Away team abbreviation (e.g., "BOS")

over_line : float
    The spread (total points threshold)

over_pct : float
    OVER probability as percentage (0-100)

under_pct : float
    UNDER probability as percentage (0-100)

volume : float
    Trading volume in USD

liquidity : float
    Available liquidity in USD
```

## Integration Notes

- **No feature pipeline integration yet** — this module is exploratory only
- Exported CSV can be merged with game data via `game_date` and team names
- Odds movement (rolling averages) could signal informational edges
- Volume/liquidity may indicate confidence in predictions

## Example Analysis

```python
import pandas as pd
from src.betting.polymarket_analyzer import fetch_polymarket_odds

# Fetch February 2024 odds
df = fetch_polymarket_odds(("2024-02-01", "2024-02-29"))

# Summary stats
print(f"Average OVER: {df['over_pct'].mean():.2f}%")
print(f"Volatility: {df['over_pct'].std():.2f}%")
print(f"Total volume: ${df['volume'].sum():,.0f}")

# Identify games with extreme odds
extreme_games = df[(df['over_pct'] < 40) | (df['over_pct'] > 60)]
print(f"\n{len(extreme_games)} games with OVER < 40% or > 60%")
```

## Future Enhancements

- [ ] Real Polymarket API integration with authentication
- [ ] Merge odds with actual game outcomes for calibration
- [ ] Time series features: odds movement, implied volatility
- [ ] Feature engineering for model: odds-to-outcome correlation
- [ ] Historical odds tracking (currently only point-in-time)

## Development Notes

- Module uses `requests` for API calls (gracefully fails if unavailable)
- Matplotlib/seaborn required for plotting (optional)
- Logging configured at INFO level; set to DEBUG for detailed traces
- All date inputs/outputs use YYYY-MM-DD string format and datetime64 internally
