# Lineup Data Collection Module

## Overview

The `lineups` module provides team roster and player availability tracking for the NBA Score Prediction system.

## Data Source

**Primary Source**: `nba_api` (stats.nba.com)

- **Endpoint**: `CommonTeamRoster`
- **Coverage**: Active roster players for each NBA team per season
- **Availability**: Season-level data (not real-time game-day specific)
- **Updates**: Query as needed; results are cached in-memory per (season_id, team_id) pair

## API Rate Limiting

Requests to nba_api enforce a rate limit of ~0.7 seconds between calls to avoid throttling by stats.nba.com. The module includes automatic delays.

## Main Functions

### `get_available_lineup(season_id, team_id, game_date) -> list[int]`

Fetches the active roster (list of player IDs) for a team in a given season.

**Parameters:**
- `season_id` (int): NBA season ID, e.g., 20231 for 2023-24 season
- `team_id` (int): Team ID, e.g., 1610612738 (Celtics)
- `game_date` (str): Game date in YYYY-MM-DD format (included for API signature compatibility; roster is season-wide)

**Returns:**
- `list[int]`: Player IDs of active roster members (typically 12–15 players)

**Example:**
```python
from src.lineups.lineup_collector import get_available_lineup

# Get Celtics roster for 2023-24 season
player_ids = get_available_lineup(20231, 1610612738, '2024-01-15')
print(f"Available {len(player_ids)} players")
# Output: Available 14 players
```

### `LineupDataLoader` Class

Context-manager-based wrapper for lineup queries. Follows the same pattern as `NBADataLoader`.

**Example:**
```python
from src.lineups.lineup_collector import LineupDataLoader

with LineupDataLoader() as loader:
    lineup = loader.get_lineup(20231, 1610612738, '2024-01-15')
    print(f"Players: {lineup}")
```

## Limitations

1. **Season-level Data**: CommonTeamRoster returns season rosters. It does not provide real-time game-day injury/suspension status. For injury-adjusted rosters, cross-reference with injury data from the news_scraping module.

2. **No Active Games Filter**: The endpoint does not distinguish between games the team has played in a season. Player IDs returned are for all season roster members, not specifically those available for a particular game.

3. **API Dependency**: Requires a live connection to stats.nba.com. Network issues will propagate as exceptions.

## Caching

Results are cached in-memory by (season_id, team_id) to minimize redundant API calls. To clear the cache:

```python
from src.lineups.lineup_collector import clear_cache
clear_cache()
```

## Integration Notes

- **No Feature Pipeline Integration Yet**: This module is standalone; it does not currently integrate into `feature_builder.py`.
- **Future Enhancements**: Consider integrating with injury data to provide injury-adjusted rosters.
- **Player ID Format**: All player IDs are integers matching the nba_api / stats.nba.com format.

## Testing

Run unit tests:
```bash
python -m pytest tests/test_lineups.py -v
```

Run end-to-end test:
```bash
python -c "from src.lineups.lineup_collector import get_available_lineup; lineup = get_available_lineup(20231, 1610612738, '2024-01-15'); print(f'Available {len(lineup)} players'); assert isinstance(lineup, list)"
```

## Author

NBA Prediction Project
