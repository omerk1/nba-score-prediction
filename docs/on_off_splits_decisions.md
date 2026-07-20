# Player On/Off Splits — Data Decisions (Phase 1)

> **Status:** Investigation complete. This document covers data-source, schema,
> join-key, and backfill-scope decisions only. **No implementation has been done** —
> `feature_builder.py` is untouched, no backfill script exists yet, nothing has been
> run against the live `nba_api.sqlite`/`player_stats_cache`. Phase 2 (implementation)
> starts only after this document is reviewed and explicitly approved.

## Goal

Give the model per-player on/off impact: how a team's point differential changes
with a given player on vs. off the court, split by overall/home/away/vs-opponent,
usable both for historical training rows and for live (not-yet-played) predictions
without a separate leakage or join-key fix later (see "Critical prior lesson" below,
which this design bakes in from the start).

---

## 1. What I found: direct on/off endpoint vs. play-by-play reconstruction

### The existing "lineups" and box-score modules do NOT already have this data

- `src/lineups/lineup_collector.py` — read in full. Despite the directory name, it
  only wraps nba_api's `CommonTeamRoster` endpoint, cached by `(season_id, team_id)`.
  It returns a season-wide roster (list of player IDs), not on-court/off-court state,
  not a game-by-game lineup, no substitution timing. `get_available_lineup()`'s own
  docstring says as much ("roster is season-wide", "not actual game-day
  availability"). **Confirmed: this module cannot produce on/off splits.**
- `src/matchups/box_scores.py` / the `box_score_stats` table it builds — per
  `(game_id, team_id)` aggregate shooting/rebounding/assist totals from
  `LeagueGameLog`. No player-level rows, no on-court-time information at all.
  **Confirmed: cannot produce on/off splits either.**
- `src/utils/player_stats_cache.py` / `player_stats_cache` table — per
  `(player_id, game_date, stat_name)` box-score stats (PPG/AST/REB/BLK/STL/FG%) for
  rolling averages. No plus-minus, no on/off-court awareness.
  **Confirmed: not usable for on/off either**, though its schema/backfill pattern
  (see §3/§4) is a good precedent to reuse.
- The original backlog entry for this feature ("B1" in `docs/backlog.md`, not
  referenced further here per naming convention) assumed the data source would be
  "actual lineups from box scores" depending on the lineup-collection module. That
  premise is **wrong** — neither box scores nor the roster collector track
  on-court/off-court state. This is worth flagging explicitly since it changes the
  dependency picture from what was originally planned.

### nba_api's `TeamPlayerOnOffSummary` / `TeamPlayerOnOffDetails` — actually tested, not just read

Both endpoints share an identical parameter surface (`teamplayeronoffsummary.py`,
`teamplayeronoffdetails.py` in `venv/lib/python3.12/site-packages/nba_api/stats/endpoints/`)
— `Details` adds full shooting/rebounding/turnover box-score columns per on/off row,
`Summary` gives just `GP, MIN, PLUS_MINUS, OFF_RATING, DEF_RATING, NET_RATING`.
**Recommend `Summary`** — it's lighter, and `PLUS_MINUS`/`NET_RATING` are exactly
`on_off_plus_minus`'s ingredients; nothing in `Details`' extra box-score columns is
needed for the three features asked for (on/off plus-minus, home/away splits,
vs-opponent splits).

I made real API calls against `TeamPlayerOnOffSummary` (not just read the source) to
answer the single most important open question from the brief: **does this endpoint
support genuine historical point-in-time / rolling-window queries, or only
"full season, as of right now"?**

**Response shape** (team_id=Boston Celtics, season=2023-24, `get_data_frames()`
returns `[overall, players_off_court, players_on_court]`): one row per roster player
per court-status, e.g.:

```
VS_PLAYER_NAME   COURT_STATUS  GP   MIN     PLUS_MINUS  OFF_RATING  DEF_RATING  NET_RATING
Brown, Jaylen    Off           82   1623.0  15.3        123.1       106.4       16.7
Brown, Jaylen    On            70   2343.0  8.5         120.2       111.6       8.7
```
`on_off_plus_minus` for a player = `(On row's PLUS_MINUS) - (Off row's PLUS_MINUS)`.
Confirmed this `PLUS_MINUS`/`NET_RATING` here is **already rate-normalized** (small
values like 8-17), unlike the team-level "Overall" dataframe's `PLUS_MINUS`, which is
a season total (e.g. 930.0) — so no extra per-minute normalization is needed on our
side.

**Test results (all run live against stats.nba.com from this worktree's venv):**

| # | Call | Result | Conclusion |
|---|------|--------|------------|
| 1 | Full season, no date filter | GP=82 | baseline |
| 2 | `date_to_nullable='2024-01-15'`, no `last_n_games` | GP=40 | **`DateTo` genuinely restricts to a historical cutoff** (season is ~40 games deep by mid-January) |
| 3 | `last_n_games=10`, no date filter | GP=10 | works alone, as expected |
| 4 | `last_n_games=10` **and** `date_to_nullable='2024-01-15'` | **empty dataframe** | `LastNGames` does not compose with `DateTo` at all |
| 5 | `last_n_games=10` and `date_to_nullable='2024-03-01'` (retry, later cutoff) | **empty dataframe** | not a fluke — confirmed twice |
| 6 | `last_n_games=10` and `date_from_nullable='2023-11-01'` (no `date_to`) | GP=10, identical to test 3 | `LastNGames` **ignores** `date_from` too |
| 7 | `date_from_nullable='2023-12-15'`, `date_to_nullable='2024-01-15'` (range, no `last_n_games`) | GP=17 | day-range windowing works correctly (17 games in a month matches NBA schedule density) |
| 8 | `opponent_team_id=<MIA>`, full season | GP=3 | vs-opponent filter works |
| 9 | `opponent_team_id=<MIA>` **and** `date_to_nullable='2024-01-15'` | GP=1 (of eventual 3) | vs-opponent filter **composes correctly** with historical cutoff |
| 10 | `location_nullable='Home'`, full season | GP=41 (exactly half of 82) | home/away filter works |
| 11 | `location_nullable='Home'` **and** `date_to_nullable='2024-01-15'` | GP=19 (of eventual 41) | home/away filter **composes correctly** with historical cutoff |
| 12 | `date_to_nullable='2023-10-01'` (before season start) | empty dataframe, no error | clean degenerate case |

**This is the crux finding:** `LastNGames` is a real-time-only convenience parameter
("last N games of the season as of today") that **cannot** be used for a leakage-safe
historical rolling window — it silently returns nothing when combined with any date
filter. But `DateFrom`/`DateTo` **do** give genuine historical point-in-time (or
day-range/"rolling window") snapshots, and — critically — they compose correctly with
both `OpponentTeamID` (vs-opponent splits) and `Location` (home/away splits), which
are exactly the two other splits this feature needs. Single API call latency is
~1.7s (measured), consistent with this repo's existing throttle convention
(`SLEEP_SECONDS = 0.6-0.7` in `box_scores.py`/`backfill_player_stats.py`).

### Recommendation: direct endpoint, not play-by-play reconstruction

Given the above, reconstructing on/off state from `playbyplayv2`/`playbyplayv3`
substitution events would mean building and validating a whole new
substitution-tracking + lineup-reconstruction pipeline (parse substitution events,
maintain 5-man lineups per team per moment, correlate with scoring plays) to
reproduce data the API already returns directly, pre-aggregated, with a working
leakage-safe historical filter. There is no scenario uncovered in this investigation
where play-by-play reconstruction is necessary — `DateTo`/`DateFrom` fully cover the
point-in-time requirement. **Recommend: `TeamPlayerOnOffSummary`, called with
`date_to_nullable` (and optionally `date_from_nullable` for a bounded window) set to
enforce the as-of-date cutoff, for all three split types.** Play-by-play
reconstruction is not recommended and should only be revisited if a future need
requires sub-day granularity (e.g., in-game live on/off state), which is out of scope
here.

---

## 2. Critical prior lesson — applied here

Per the coordinator's brief: `_add_style_matchup_features` in `feature_builder.py`
originally joined a precomputed cache to `df` by exact `GAME_ID`, which works for
training (every row is an already-played, already-cached game) but silently produces
NaN for every live prediction (`predict_game.py`'s synthetic `GAME_ID='upcoming'` row
can never match a cached game_id). `_add_injury_features` avoids this by joining on
`(team_id, game_date)`; `_add_style_fingerprint_features` was fixed to do the same via
`pd.merge_asof(..., by="team_id", direction="backward", allow_exact_matches=True)`.

**On/off splits must follow the same pattern from the start.** The cache this
feature needs is naturally keyed by `(player_id, team_id, split_type,
opponent_team_id [nullable], as_of_date)` — never by `game_id`. At feature-build
time, lookups must use `pd.merge_asof` on `game_date`, `by=team_id` (and
`by=[team_id, opponent_team_id]` for the vs-opponent split — `merge_asof` supports a
list for `by`), `direction="backward"`, `allow_exact_matches=True`. This works
identically whether `game_date` is a real historical date (training) or today's/a
future date (`predict_game.py`'s live prediction row), because the key is a natural
date/team key, not a historical-game identifier. No separate live-prediction fix
should be needed later, unlike A7's original implementation.

---

## 3. New table vs. columns on existing tables

**Recommendation: a new, dedicated table** (name TBD in phase 2, e.g.
`player_on_off_splits` — avoiding backlog-label naming per the repo's naming
convention). Reasons:

- None of the existing tables share this table's natural key. `game`/`box_score_stats`
  are keyed by `(game_id, team_id)`; `player_stats_cache` is keyed by
  `(player_id, game_date, stat_name)` — neither has a `split_type`/`opponent_team_id`
  dimension, and neither is meant to hold "value as of an arbitrary checkpoint date,"
  which on/off splits fundamentally are (the API returns a season-cumulative-to-date
  number, not a single game's number).
- Proposed schema:
  ```sql
  CREATE TABLE player_on_off_splits (
      player_id         INTEGER NOT NULL,
      team_id           INTEGER NOT NULL,
      split_type        TEXT NOT NULL,   -- 'overall' | 'home' | 'away' | 'vs_opponent'
      opponent_team_id  INTEGER,          -- NULL unless split_type='vs_opponent'
      as_of_date        TEXT NOT NULL,    -- the DateTo cutoff used for the fetch (exclusive of leakage)
      season             TEXT NOT NULL,
      gp_on REAL, gp_off REAL,
      min_on REAL, min_off REAL,
      plus_minus_on REAL, plus_minus_off REAL,
      net_rating_on REAL, net_rating_off REAL,
      on_off_plus_minus REAL,   -- plus_minus_on - plus_minus_off, precomputed for convenience
      on_off_net_rating REAL,   -- net_rating_on  - net_rating_off
      PRIMARY KEY (player_id, team_id, split_type, opponent_team_id, as_of_date)
  )
  ```
- **Where it should live:** not decided definitively here (phase-2 call), but flagging
  the tradeoff. `player_stats_cache` lives directly inside `data/raw/nba_api.sqlite`
  (via its own migration script) — that's the more common convention. A7's
  `style_fingerprint_cache.sqlite` is a *separate* file specifically because A7's
  `db.py` opens `nba_api.sqlite` strictly read-only via a symlink and needed its own
  write location. **This worktree currently has no `nba_api.sqlite` at all** (only
  `.gitkeep` in `data/raw/` — it's gitignored and wasn't symlinked in for this task),
  so whoever implements phase 2 will need to either symlink it in read-write (if this
  feature will insert directly into the shared DB, following `player_stats_cache`'s
  precedent) or use a dedicated additive file (following A7's precedent) — worth
  deciding explicitly before writing the backfill script. Given this cache is fully
  reconstructable from the API (nothing here is irreplaceable raw data) and given the
  benefit of not touching the shared core DB while this feature is being iterated on,
  **I lean toward a separate additive file** (e.g. `data/raw/player_on_off_splits.sqlite`),
  but this is a soft recommendation, not a hard blocker either way.
- **Do not add on/off columns to `player_stats_cache` or `game`/`box_score_stats`** —
  wrong key shape for both (as above), and it would conflate "per-game box score fact"
  with "as-of-date derived aggregate," which are different things with different
  staleness/leakage semantics.

---

## 4. Backfill scope

I queried the actual (read-only) `data/raw/nba_api.sqlite` in the main checkout to
size this concretely (no writes made):

- `game` table: **12,793 games** since `data_start_date` (2016-10-01), through
  **8,787 games** in the actual `train_start_date`(2018-10-16)..
  `validation_end_date`(2025-04-13) window that's used as real training/val samples.
- **30 distinct teams.**
- `player_stats_cache` (existing precedent, for scale comparison): 1,850,508 rows,
  1,602 distinct game dates — built via one `BoxScoreTraditionalV3` call **per game**
  (not per team), at `SLEEP_SECONDS = 0.6`.

Unlike `player_stats_cache`'s backfill, the on/off endpoint is called **once per
team** (not per game) and returns a season-cumulative-to-date number for whatever
`DateTo` you pass — so the naive "exact as-of-every-game" approach costs far more
calls, not fewer:

- **"Overall" split, one snapshot per team per game they play:** ~8,787 games × 2
  teams ≈ **17,574 calls**, just for the overall split. At ~1.7s/call (measured) +
  throttle (~0.7s, matching repo convention) ≈ 2.4s/call → **~11.7 hours**.
  home/away roughly doubles this; vs-opponent (computed only for the specific
  pairing actually scheduled, not all 29 possible opponents — see below) adds
  roughly the same order again. Full per-game precision across all three split
  types is realistically **a full day-plus one-time job** — not impossible, but a
  lot heavier than any existing backfill in this repo.
- **Recommended alternative: a coarser checkpoint cadence, not per-game.** Because
  the endpoint returns a *cumulative* number, its value barely moves between two
  adjacent games (one additional game out of 40+ played). A weekly (or
  every-~5-games) checkpoint per team, merged onto actual game dates via
  `merge_asof` (§2), gives up at most a few days of staleness on a signal that's
  already a slow-moving season aggregate — the same tradeoff this repo already
  accepts elsewhere (rolling averages, style fingerprints). At a weekly cadence:
  ~30 teams × ~9 seasons × ~26 checkpoints/season ≈ **~7,020 calls** for "overall"
  alone, ~14,000 with home/away — **under 2 hours** at the same throttle. This is
  the approach I'd recommend for phase 2, with the exact cadence tuned then.
- **vs-opponent is the expensive/risky one and should be scoped down explicitly.**
  A single-season pairing has only 2-4 games (`GP=3` for BOS-vs-MIA in the full
  2023-24 season, confirmed above) — a tiny, noisy sample if computed per season.
  Computing it for *every* team × every one of 29 possible opponents at the same
  weekly cadence would be ~30 × 29 × 26 × 9 ≈ 200,000+ calls — infeasible. Recommend
  computing vs-opponent splits **lazily, only for the specific opponent pairing that
  actually occurs** in each historical game (bounds cost to roughly the game count,
  not the combinatorial team×opponent space), and treating it as a **second-priority
  feature** behind overall/home-away splits given both its cost and its inherently
  low per-season sample size (see Open Risks).
- All of the above should follow the repo's established throttle convention
  (`SLEEP_SECONDS` in the 0.6-0.7s range) and the resumable/incremental pattern
  `scripts/backfill_player_stats.py` already established (checkpoint by
  last-cached date, `--update` incremental mode, retry with backoff).

---

## 5. Granularity

- **Cache row granularity: one row per `(player_id, team_id, split_type,
  opponent_team_id, as_of_date checkpoint)`** — not per `game_id`. The underlying
  API value is a season-cumulative-to-date figure that only changes when a new
  qualifying game is added, so storing it once per checkpoint (not once per game) is
  both cheaper and more honest about what the data actually represents.
- **Leakage safety is enforced at fetch time, not just at join time**: because
  `DateTo` was empirically confirmed to restrict the underlying query to games at or
  before that date (test #2/#9/#11 above), simply setting `date_to_nullable` to the
  checkpoint date (always in the past relative to when the row is used) makes the
  fetched value leakage-safe by construction. The `merge_asof(direction="backward")`
  join at feature-build time is then a second, independent safeguard — it can only
  select a checkpoint whose `as_of_date` is at or before the target game's date,
  never a future one.
- **Recommended checkpoint cadence:** weekly (or similarly coarse, e.g. every
  ~5 team-games) rather than per-game, per the cost analysis in §4. Exact cadence is
  a phase-2 tuning decision, but the *direction* (checkpoint-cache-and-merge_asof,
  mirroring `_add_style_fingerprint_features`'s already-validated pattern) should be
  fixed now rather than reconsidered later.
- **Not per-season-only**: a single per-season-end snapshot (updated once a season)
  would be too coarse for in-season prediction — it would use last season's fully
  resolved on/off numbers for the first ~half of the following season and never
  reflect the current season's roster/role changes until the season is nearly over.
  A checkpoint cadence within-season (as above) is needed for a genuinely
  leakage-safe rolling feature that still reflects reasonably current form.

---

## 6. Open risks / unknowns not resolved in this phase

1. **Team-level aggregation of player-level on/off splits is unresolved.** The raw
   output is per-player. A game-level model feature needs a fixed-width team-level
   number(s), not 15 raw per-player columns. Candidate approaches not yet decided:
   (a) headline/highest-usage player's on/off differential as a proxy, (b) a
   minutes/usage-weighted average across the current likely-active roster (reusing
   `player_stats_cache`'s existing minutes/usage signal), or (c) — the option I'd
   flag as most promising given it directly extends the already-working
   `_add_injury_features` join — combine this with the existing
   `injury_features.sqlite` table (`n_out`/`n_questionable` per `(team_id,
   game_date)`) to produce something like "expected point swing from currently
   missing players' on/off impact," which ties directly into the backlog's
   already-planned "Player Availability Impact" follow-on. This is a real design
   decision for phase 2, not resolved here.
2. **vs-opponent sample-size noise.** Even with the lazy, per-actual-pairing
   approach above, a single-season vs-opponent split has only 2-4 games of support
   — likely too noisy to use directly. May need multi-season pooling (mirroring
   `_add_h2h_features`'s 3-year lookback), which the API's `Season` parameter
   doesn't natively support in one call — would require fetching per season and
   combining GP-weighted on our side. Not tested in this phase.
2b. Where exactly the new cache DB should physically live (shared `nba_api.sqlite`
   vs. a dedicated additive file) is flagged in §3 as a soft, not hard, decision —
   depends on whether phase 2 gets a read-write symlink into the shared DB or not.
3. **Checkpoint cadence is a guess, not tuned.** Weekly was chosen as a reasonable
   starting point balancing staleness vs. API cost; it hasn't been validated
   against how quickly on/off numbers actually drift week-to-week for a real
   roster. Worth a quick sensitivity check early in phase 2 before committing to a
   full backfill at that cadence.
4. **`TeamPlayerOnOffDetails` was not independently tested** for the same
   `DateTo`/`LastNGames`/`OpponentTeamID`/`Location` composability — it shares
   identical parameter-handling code with `TeamPlayerOnOffSummary` (same base
   `Endpoint` class, same `__init__`/`get_request`/parameters dict, only
   `expected_data` differs), so behavior should be identical, but this was inferred
   from source rather than independently verified by a live call. Low risk given
   `Summary` is the recommended endpoint anyway and `Details` isn't needed.
5. **Multi-year "career-to-date" queries** (spanning season boundaries) were not
   tested — the `Season` parameter appears to require a single season string; a
   query spanning multiple seasons (if ever needed) would require multiple calls
   combined client-side. Not needed for the current recommendation but noted in
   case phase 2 wants a longer-than-one-season baseline.

---

## Recommendation summary

- **Data source:** `nba_api`'s `TeamPlayerOnOffSummary` endpoint, called with
  `date_to_nullable` (and optionally `date_from_nullable`) set to enforce a
  leakage-safe historical cutoff — confirmed to work via real API calls. Do **not**
  reconstruct from play-by-play; nothing in this investigation shows it's necessary.
  Do **not** rely on `LastNGames` for anything historical — confirmed broken when
  combined with date filters.
- **Storage:** a new dedicated table (schema in §3), keyed by
  `(player_id, team_id, split_type, opponent_team_id, as_of_date)`, populated at a
  checkpoint cadence (weekly-ish, not per-game) rather than a per-game row.
- **Join strategy:** `pd.merge_asof` on `(team_id [, opponent_team_id], game_date)`,
  `direction="backward"`, exactly mirroring the already-fixed
  `_add_style_fingerprint_features` pattern — never an exact `game_id` join.
- **Backfill scope:** feasible within a few hours for overall + home/away splits at
  a weekly checkpoint cadence (~14,000 calls); vs-opponent splits should be
  scoped down to lazy per-actual-pairing fetches and treated as second priority
  given cost and sample-size noise.
- **Biggest open question for phase 2:** how to collapse per-player on/off splits
  into team-level model features — recommend exploring the injury-data tie-in
  (§6.1) as the most promising angle, but this needs a real design pass, not a
  guess baked into this document.
