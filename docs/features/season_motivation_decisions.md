# Season Motivation & Seeding Incentive — Data Decisions

Data audit + formulas for `motivation_score`, `games_to_clinch_ceiling`,
`games_to_clinch_floor`, written before implementation. See
`docs/features/season_motivation_log.md` for what was actually built and validated.

## Goal

A continuous "how hard is this team trying to win tonight?" signal plus two
seed-lock countdowns, leakage-safe for historical rows and reusable live.

## 1. No new backfill needed

Every ingredient already exists in already-backfilled tables covering the
full 2018-19 through 2025-26 window:

- **Standings**: `nba_api.sqlite.game` has zero NULL scores — point-in-time
  standings as of any date are just an aggregation of rows strictly before
  it. No separate standings endpoint needed.
- **Remaining schedule**: every touched season is already historically
  complete, so the full schedule (who plays whom, when) already sits in
  `game` — only `game_date`/team-id columns are read from "future" rows,
  never outcomes, so this is schedule metadata (public before the season
  starts), not a leakage risk. **Known production gap:** live inference on
  an in-progress season needs a fresh `ScheduleLeagueV2` fetch — not built
  here, flagged as a follow-up.
- **Conference assignment**: not in any DB table, but a fixed fact (no team
  changed conference 2018-2026) — hardcoded as `_TEAM_CONFERENCE`, same
  convention as `_TEAM_LOCATIONS`.
- **Roster quality**: `injury_features.sqlite.player_importance` (weekly,
  2018-10-22 through 2026-05-27, 281 snapshots / 144,479 rows) has
  `minutes_per_game`/`pts_per_game`/`usage_rate` per player. Reused
  `_get_importance_map`'s existing weighted formula
  (`minutes_share=0.4, usage_rate=0.4, pts_share=0.2`) rather than inventing
  a second one.
- **Who sat and why**: `player_injuries.reason` has a clean `Rest` value
  (328 rows) plus variants, `Personal Reasons`, `Not With Team`, `Coach's
  Decision` — all non-injury sit-outs. **Known coverage limit:**
  `player_injuries` has zero rows before **2021-10-19**
  (`injury_features.pdf_era_start`, the official PDF-report era start) —
  the roster-behavior component is a structural constant zero before that
  date, same limit on/off-splits already hit.

**A3/A4 examined, NOT usable** despite "Complete" backlog status: A3's
`_get_team_roster` is an unfinished placeholder (`return []`), and
`project_game_contributions` doesn't filter by team at all. A4's
`get_available_lineup` returns `CommonTeamRoster`'s season-end roster, not
per-game participants (same finding on/off-splits already made).

**Net: no new backfill script or DB table.** Everything computed in-memory
from three already-complete tables, same "derive it, don't refetch it"
approach `_add_rolling_features`/`_add_elo_features` use.

## 2. `motivation_score` formula

### 2a. Standings pressure

For team T as of date D (Regular Season only):
- `wins(T,D)`/`losses(T,D)`: aggregated from `game` rows with `game_date < D`.
- `games_remaining(T,D) = total_games_in_season(T) - games_played(T,D)`.
- Conference ranked by win% (0.5 default before any games played).
- `GB_from_line(T,D) = ((wins(line) - wins(T)) + (losses(T) - losses(line))) / 2`
  against whichever team holds the **10th seed** (the actual play-in cutoff,
  not top-6-only — a team at 9th/10th is still fighting).
- `pressure_raw(T,D) = clip(1 - |GB_from_line(T,D)| / (games_remaining(T,D)+1), 0, 1)`
  — the **absolute** gap gives the required two-sided decay (far below the
  line = hopeless, far above = clinched, both → 0; peaks at 1 exactly at the
  line). `+1` avoids divide-by-zero on the season's last game.
- (This GB-based formula and §3's ceiling/floor use deliberately different
  arithmetic — GB-vs-line answers "how close is the race now," raw
  win-count projection answers "when does this stop being mathematically
  possible.")

### 2b. Roster-behavior signal

- `full_strength_quality(T,D)` = sum of importance scores for T's rostered
  players, latest `player_importance` snapshot before D.
- `sat_healthy_quality(T,D)` = same sum, restricted to players `Out` on D
  for a non-injury reason (documented set: `rest`, `rest-rest`, `rest - rest`,
  `rest - load management`, `personal reasons`, `personalreasons`, `not with
  team`, `notwithteam`, `coach's decision`, `coach'sdecision` — deliberately
  excludes health/safety protocols, suspensions, trade-pending).
- `roster_behavior_score(T,D) = clip(sat_healthy_quality / full_strength_quality, 0, 1)`.

### 2c. Combination

```
motivation_score(T,D) = pressure_raw(T,D) * (1 - roster_behavior_weight * roster_behavior_score(T,D))
```

Multiplicative, not additive: sitting stars should pull the score *down*
from wherever standings pressure put it, not vote independently — also
keeps the result in [0,1] for free when `roster_behavior_weight <= 1`.
`roster_behavior_weight` explored empirically (see log doc §3).

**Documented limitation:** only captures *behavioral* tanking (visibly
sitting healthy players) — a team playing regulars normally but not trying
in-game is invisible; no data source distinguishes this without richer
play-by-play effort proxies.

## 3. `games_to_clinch_ceiling` / `games_to_clinch_floor`

A **continuous proxy**, not exact combinatorial/tiebreaker logic (no
head-to-head, no multi-team scenarios — one adjacent rival at a time). Same
simplification Phase 2's `preferred_opponent_delta` documents as a known
limitation, referenced there rather than re-justified.

- `max_final_wins(T,D) = wins(T,D) + games_remaining(T,D)` — only decreases
  (by exactly 1) on a loss, unchanged by a win, so it's usable as a
  monotonic countdown without per-game simulation.
- `min_final_wins(T,D) = wins(T,D)`.
- `above`/`below` = teams ranked one spot better/worse (skip if T is 1st or
  last in the play-in picture).

```
games_to_clinch_ceiling(T,D) = max(0, max_final_wins(T,D) - wins(above,D))
games_to_clinch_floor(T,D)   = max(0, wins(below,D) + games_remaining(below,D) - min_final_wins(T,D))
```

Ceiling hits 0 once T's best case can no longer exceed `above`'s *current*
total (conservative, doesn't require guessing `above`'s future results).
Floor hits 0 once `below`'s best case can no longer reach T's *worst* case.
Both 0 → nothing left to play for (feeds §2's pressure → ~0, and Phase 2's
gating). 1st-seed ceiling / last-play-in-spot floor = 0 by convention.

## 4. Config

```yaml
season_motivation:
  enabled: false
  playoff_line_seed: 10          # play-in cutoff, current NBA format
  roster_behavior_weight: 1.0    # explored in validation, see log doc
  min_importance_games: 5        # player needs >=N snapshots before counting
                                  # toward full_strength_quality
```

No `db_path` — reads existing tables directly (`data_paths.raw_db`,
`injury_features.db_path`) rather than owning a cache.

## 5. Known limitations

1. Tiebreakers not modeled (head-to-head, division, conference record) —
   deliberate continuous-proxy simplification.
2. Roster-behavior signal is zero before 2021-10-19 (`player_injuries`
   coverage start).
3. Pure strategic tanking is invisible (see §2c).
4. Live/production schedule gap — needs a fresh `ScheduleLeagueV2` fetch,
   not built here.
5. Mid-season trades: `player_importance.team_id` is snapshotted weekly, so
   a traded player's contribution shifts within at most a week — good
   enough, not treated as a special case.
