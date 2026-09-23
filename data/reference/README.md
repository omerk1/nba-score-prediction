# Reference data

Small, static datasets committed to the repo because they are cheap to store
and expensive or impossible to re-derive. Unlike `data/raw/` (gitignored, bulk,
rebuildable from an API), everything here is intended to outlive its source.

## `nba_officials_1996_2023.csv.gz`

Referee assignments: which officials worked which game.

| | |
|---|---|
| Rows | 70,971 (one per official per game) |
| Games | 23,575 |
| Officials | 235 |
| Span | 1996-11-08 to 2023-06-12 |
| Size | 0.33 MB gzipped |

Columns: `game_id`, `game_date`, `season_type`, `official_id`, `first_name`,
`last_name`, `jersey_num`. `game_id` is a zero-padded string — read it as text
(`pd.read_csv(..., dtype={"game_id": str})`) or the leading zeros are lost.

**Provenance**: extracted from the `officials` table of
`data/raw/basketball.sqlite`, a Kaggle NBA dump
(<https://www.kaggle.com/datasets/wyattowalsh/basketball>) that no code path
read and that was deleted to reclaim 2.2 GB. `game_date`/`season_type` were
joined in from that dump's own `game` table so this file stands alone;
`nba_api.sqlite` only reaches back to 2016 and could not supply dates for the
older two thirds. 56 All-Star games appear twice in the source `game` table
under inconsistent `season_type` spellings (`All-Star` vs `All Star`) — these
were normalised and de-duplicated before the join, and the output row count is
asserted equal to the source `officials` count.

**Status**: not used by any feature. The referee-tendency backlog item was
tested and **rejected** — officials differ persistently in fouls called
(split-half r ≈ 0.49) and free throws (≈ 0.52), but not in total points
(0.07) or point margin (0.15), and coverage stops in June 2023 with nothing
for the 2024-25 or 2025-26 seasons. Full evidence in
`docs/NEXT_PHASE_SESSIONS.md` under the referee/officiating-crew item.

Kept because it is 0.33 MB, cannot be regenerated once the source is gone, and
makes the rejection re-checkable.
