# Season Motivation & Seeding Incentive — Data Decisions (Phase 1)

> **Status:** Investigation complete, before any implementation. This document
> covers what data already exists, what (if anything) is missing, and the exact
> formulas chosen for `motivation_score`, `games_to_clinch_ceiling`, and
> `games_to_clinch_floor`. `feature_builder.py` is untouched as of this writing.

## Goal

A continuous "how hard is this team trying to win tonight?" signal plus two
continuous seed-lock countdowns, computed leakage-safe for historical training
rows and reusable for live (not-yet-played) predictions.

---

## 1. What I found: no new backfill is needed

The brief asked me to check whether standings/schedule backfill was needed before
building `scripts/backfill_season_motivation.py`. It isn't — every ingredient
already exists in already-backfilled tables, fully covering the model's entire
training/val/test window (2018-19 through 2025-26):

- **Standings (wins/losses per team, point-in-time)**: `nba_api.sqlite`'s `game`
  table has zero NULL scores (`SELECT COUNT(*) FROM game WHERE wl_home IS NULL` →
  0 of 12,793 rows) — every Regular-Season game the model ever sees is a completed
  historical game, so point-in-time standings as of any date are just an
  aggregation of `game` rows strictly before that date. No separate standings
  endpoint/backfill needed.
- **Remaining schedule (games left, per team, per season)**: since every season in
  scope is a *completed historical* season by the time it's used for
  training/val/test, the full season's game list (who plays whom, on what date) is
  already sitting in the same `game` table — I only ever read `game_date` and the
  two team-id columns from "future" rows relative to a given training row, never
  their outcome columns. This is not a leakage risk: an NBA season's schedule is
  published before the season starts and is public knowledge at every point during
  it — it is schedule metadata, not a game result. **Known production gap** (not
  blocking training/ablation): live inference on an in-progress season would need a
  fresh schedule fetch (`nba_api`'s `ScheduleLeagueV2`, same idea as
  `fetch_data.py`'s `fetch_upcoming_games`) for the *current* season's remaining
  games, since those rows don't exist in `game` yet. Flagged as a follow-up, not
  built here.
- **Conference assignment**: not in any DB table (nba_api's `static.teams` doesn't
  include it), but it's a fixed, non-tunable fact — no team has changed conference
  in the 2018-2026 window. Hardcoded as a `_TEAM_CONFERENCE` dict in
  `feature_builder.py`, same convention as the existing `_TEAM_LOCATIONS` dict
  (lat/lon/utc-offset) a few lines above it.
- **Roster full-strength quality + per-player "importance"**: `injury_features.sqlite`'s
  `player_importance` table (built by `src/news_scraping/player_importance.py`,
  already weekly-backfilled from **2018-10-22 through 2026-05-27**, 281 snapshots,
  144,479 rows) stores `minutes_per_game`, `pts_per_game`, `usage_rate` per
  `(player_id, team_id, as_of_date)`. `src/news_scraping/pipeline.py`'s
  `_get_importance_map` already combines these into a single 0-1 "importance"
  score per player via a configured weighted formula
  (`injury_features.importance_weights`: `minutes_share=0.4`, `usage_rate=0.4`,
  `pts_share=0.2`) — reused here as-is rather than inventing a second quality
  formula, for the same reason on/off-splits reused `compute_team_deficit`'s
  Doubtful weighting instead of a new one.
- **Who sat out and why (for the behavioral-tanking signal)**: `player_injuries`
  (same DB) already has a `reason` free-text column alongside `status`. A clean
  `Rest` reason value exists (328 rows) plus close variants (`rest-rest`,
  `rest - rest`, `rest - load management`), and separately `Personal Reasons` /
  `PersonalReasons`, `Not With Team` / `NotWithTeam`, and `Coach's Decision` /
  `Coach'sDecision` — none of these are real injuries, all are a team choosing to
  sit a player. **Known coverage limit** (already discovered and documented during
  the on/off-splits work, re-confirmed here): `player_injuries` has zero rows
  before **2021-10-19** (`injury_features.pdf_era_start`) — the NBA's official
  injury-report PDFs only go back that far. Before that date the roster-behavior
  component of `motivation_score` is a structural constant zero, same limitation
  on/off-splits hit and documented.

**A3 and A4 were examined and are NOT usable as-is** for the roster-behavior
signal, despite backlog A3/A4 both being marked "Complete":
- `src/projections/player_projections.py`'s `_get_team_roster` is an explicit
  unfinished placeholder — `return []`, its own comment says "the caller must
  provide player IDs" — and `project_game_contributions` doesn't actually filter by
  team at all (`SELECT DISTINCT player_id FROM player_stats_cache ... LIMIT 1000`
  across *all* players in the cache, ignoring `home_team_id`/`away_team_id`
  entirely). Not usable for a team-scoped quality comparison without real
  team-filtering logic that doesn't currently exist.
- `src/lineups/lineup_collector.py`'s `get_available_lineup` returns
  `CommonTeamRoster`'s **season-end roster**, not a specific game night's actual
  participants — its own docstring says so ("roster is season-wide"). Same finding
  the on/off-splits decisions doc already made about this module.

Given `player_importance` already has real per-team, per-week, backfilled
minutes/usage/points with an existing importance formula, and `player_injuries`
already has real per-team, per-game-date sit-out records with reasons, neither A3
nor A4 is needed here — the roster-behavior signal is built directly from the two
`injury_features.sqlite` tables instead.

**Net result: no `scripts/backfill_season_motivation.py` and no new DB table.**
Everything is computed in-memory inside `feature_builder.py` from three
already-complete tables (`nba_api.sqlite.game`, `injury_features.sqlite.player_importance`,
`injury_features.sqlite.player_injuries`), the same "derive it, don't refetch it"
approach `_add_rolling_features`/`_add_elo_features` already use for their own
point-in-time stats. `outputs/season_motivation_backfill_log.csv` therefore has
nothing to log — noted as N/A in the log doc rather than fabricating a backfill run
that didn't need to happen.

---

## 2. `motivation_score` — formula

Two ingredients, both continuous, combined as one team-game-level score in [0, 1]:

### 2a. Standings pressure

For team T as of date D (Regular Season games only, matching
`datasets_loading.allowed_season_types`):
- `wins(T, D)`, `losses(T, D)`: aggregated from `game` rows with
  `game_date < D` in T's season.
- `games_remaining(T, D)`: `total_games_in_season(T) - games_played(T, D)`, from
  the full (already-known) season schedule.
- Conference standings rank computed from `wins/losses` for all 15 teams in T's
  conference (`_TEAM_CONFERENCE`), using standard games-back (`GB`) arithmetic
  against the **10th seed** (the actual play-in cutoff under the current format,
  not a top-6-only cutoff — a team at 9th or 10th is still fighting for its season).
- `pressure_raw(T, D) = clip(1 - GB_from_10_seed / (games_remaining(T, D) + 1), 0, 1)`
  — the `+1` avoids a divide-by-zero on the last game of the season and makes a
  1-game-back-with-1-to-play situation read as near-maximum pressure rather than
  exactly 1.0/0.0 on a knife edge. Teams already comfortably in (GB negative /
  clinched) or hopelessly out (GB larger than games left) both correctly clip to
  the low end.
- This reuses the exact same ceiling/floor win-count arithmetic as §3 below, so the
  "team is already locked in either direction" case falls out for free instead of
  needing a second special case.

### 2b. Roster-behavior signal

- `full_strength_quality(T, D)` = sum of `_get_importance_map`-style importance
  scores (reusing `injury_features.importance_weights`) for T's rostered players,
  using the latest `player_importance` snapshot before D.
- `sat_healthy_quality(T, D)` = sum of the same importance scores, restricted to
  players in `player_injuries` with `team_id=T`, `game_date=D`, `status='Out'`, and
  `reason` (lowercased/trimmed) in a documented non-injury set: `rest`,
  `rest-rest`, `rest - rest`, `rest - load management`, `personal reasons`,
  `personalreasons`, `not with team`, `notwithteam`, `coach's decision`,
  `coach'sdecision`. Deliberately excludes `health and safety protocols`,
  `concussion protocol`, `league suspension`/`team suspension`,
  `trade pending`/`ineligible to play` — none of those are a team choosing to rest
  a healthy player for competitive reasons.
- `roster_behavior_score(T, D) = clip(sat_healthy_quality / full_strength_quality, 0, 1)`.

### 2c. Combination

```
motivation_score(T, D) = pressure_raw(T, D) * (1 - roster_behavior_weight * roster_behavior_score(T, D))
```
Multiplicative rather than a weighted sum: "the standings say you should be
fighting hard, but you're sitting your stars" should pull the score *down* from
wherever standings pressure put it, not get averaged in as an independent
half-vote. It also keeps the result in [0, 1] for free as long as
`roster_behavior_weight <= 1`, without extra clipping. `roster_behavior_weight` is
a config parameter, explored empirically in the Phase 1 validation round (§6 of the
log doc) rather than guessed once and left fixed.

**Documented limitation (per the brief, not attempted here):** this only captures
*behavioral* tanking (visibly sitting healthy, important players). A team that
plays its regulars normally but simply isn't trying as hard in-game is invisible
to this signal — there is no data source available that would distinguish
"trying at 100%" from "playing but not trying" without much richer play-by-play
effort proxies, which is out of scope for this round.

---

## 3. `games_to_clinch_ceiling` / `games_to_clinch_floor`

Standard sports-analytics "magic number" arithmetic, deliberately simplified to a
**continuous proxy** rather than an exact combinatorial/tiebreaker-aware
elimination calculation (the brief explicitly prefers this): no head-to-head
tiebreakers, no multi-team simultaneous scenarios — one adjacent rival at a time.
This exact simplification is also what Phase 2's `preferred_opponent_delta` is
told to document as a known limitation ("does not model multi-seed jumps or
conference picture complexity"), so it's introduced once here and referenced
there rather than re-justified twice.

For team T, ranked in position `rank` in its conference as of date D:
- `max_final_wins(T, D) = wins(T, D) + games_remaining(T, D)` — T's best possible
  final win total. This value only ever decreases (by exactly 1) when T **loses**
  a game; it is unchanged by a win (a win converts one unit of "remaining upside"
  into one already-banked win — no net change to the ceiling). This is what makes
  it usable as a monotonic countdown rather than needing per-game schedule
  simulation.
- `min_final_wins(T, D) = wins(T, D)` — T's worst possible final total (loses out).

Let `above` = the team currently ranked one spot better than T (skip if T is
already 1st seed), `below` = the team currently ranked one spot worse (skip if T
is already last in the play-in picture).

```
games_to_clinch_ceiling(T, D) = max(0, max_final_wins(T, D) - wins(above, D))
games_to_clinch_floor(T, D)   = max(0, wins(below, D) + games_remaining(below, D) - min_final_wins(T, D))
```

- Ceiling hits 0 the moment T's best-case final total can no longer exceed
  `above`'s *current* total (a conservative, always-eventually-true bound — doesn't
  depend on guessing `above`'s future results). Reads naturally as "T is
  mathematically no closer to improving its seed."
- Floor hits 0 the moment `below`'s best-case final total can no longer reach
  T's *worst*-case final total — T's current standing is safe no matter what.
- When both are 0, per the brief: T has nothing left to play for from a seeding
  standpoint (feeds directly into §2's pressure term already going to ~0 in this
  case, and into Phase 2's gating).
- 1st-seed ceiling and last-play-in-spot floor are defined as 0 by convention
  (nothing above to improve past / nothing below to protect against within the
  play-in picture).

---

## 4. Config

New `season_motivation` section, following `OnOffSplitsConfig`/`StyleMatchupConfig`'s
exact pattern (`enabled: bool` gate, defaults `false` until validated):

```yaml
season_motivation:
  enabled: false
  playoff_line_seed: 10          # play-in cutoff, current NBA format
  roster_behavior_weight: 1.0    # explored in Phase 1 validation, see log doc
  min_importance_games: 5        # player must have >=N player_importance snapshots
                                  # before counting toward full_strength_quality —
                                  # guards against a 1-game-old callup skewing the sum
```

No `db_path` field — unlike `on_off_splits`/`style_matchup`, this feature reads
existing tables directly (`data_paths.raw_db`, `injury_features.db_path`) rather
than owning its own cache.

---

## 5. Open risks / known limitations (carried into the log doc's Phase 1 section)

1. **Tiebreakers not modeled** — real NBA seeding uses head-to-head record,
   division standing, and conference record as tiebreakers before falling back to
   full-conference record. This design uses raw win total only. Documented, not
   fixed — matches the brief's explicit preference for a continuous proxy over
   exact correctness.
2. **Roster-behavior signal is zero before 2021-10-19** — `player_injuries`
   coverage start, same hard limit already hit and documented during the
   on/off-splits work.
3. **Pure strategic tanking is invisible** — see §2c.
4. **Live/production schedule gap** — remaining-schedule computation depends on
   the season already being complete in `game`; a live in-season prediction needs
   a fresh `ScheduleLeagueV2` fetch not built here (documented as a follow-up).
5. **Mid-season trades**: `player_importance`'s `team_id` is snapshotted weekly,
   so a player's `full_strength_quality` contribution shifts to his new team
   within at most one week of a trade — good enough for this feature's purposes,
   not treated as a special case.
