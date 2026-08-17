# Project Backlog

## Phase 1 Status (Complete)

### Group A — Independent Baseline Features (All Complete, Ready to Merge)

- **A1** ✅ ELO Hyperparameter Tuning v2 (committed to main)
- **A2** ✅ Extended H2H Features (PR #16, ready to merge)
- **A3** ✅ Player Box Score Projections (PR #18, awaiting expansion confirmation to 6 stats)
- **A4** ✅ Lineup Data Collection (PR #14, ready to merge)
- **A5** ✅ Polymarket Signals (branch `feature/polymarket-comeback-analysis` — supersedes the original collector, which had an unfixed game_id-join bug and weaker fuzzy-search/regex-based discovery; covers moneyline in-game comeback analysis + pre-game moneyline/spread/O-U ratios at ~94-99% real coverage on the 2025-26 season; see "Standalone: Polymarket In-Game Price History Pipeline" below)
- **A6** 🔄 OddsPapi Sportsbook Signals (planned — full season coverage, 250+ bookmakers)
- **A7** ✅ Style Matchup Score (branch `feature/a7-style-matchup`, 9 rounds; see `docs/features/a7_phase_log.md`) — real ablation-pipeline comparison decided the two variants differently: KNN-similarity score (`style_matchup.enabled`) confirmed no signal (~29th of 109 features, confidence ~zero importance), **not adopted**, stays `false`; raw-fingerprint redesign (`style_matchup.raw_features_enabled`) showed genuine signal (`away_style_pace_score`/`home_style_pace_score` ranked #1/#2 importance, consistent `total_mae` improvement on both val and test, flat-to-slightly-worse `win_acc`/`brier`), **adopted** as the new committed default (`true`)
- **A8** ✅ Feature-Builder Fixes (found during A7's EDA): fixed `h2h_win_pct_3yr`'s index-reindex bug (~99.7% NaN → NaN only for genuine first-meetings) + renamed `home_team_3pt_rate_L{window}` (FG3_PCT) → `home_team_fg3_pct_L{window}` to disambiguate from `home_style_three_pt_reliance` (3PA/FGA) + made rolling FG_PCT/FT_PCT/fg3_pct volume-weighted (sum(makes)/sum(attempts) over the window instead of a naive mean of per-game percentages, which let low-attempt outlier games swing the average as much as normal-volume ones) — required storing FGM/FGA/FG3M/FG3A/FTM/FTA in the `game` table (`scripts/migrate_shot_volume_columns.py` backfills these into pre-existing DBs)

### Backfill Infrastructure ✅
- **Backfill Resilience** (PR #21, merged)
  - Exponential backoff retry in backfill_player_stats.py
  - recover_failed_backfill.py script for manual recovery
  - Database fully recovered: 1,850,508 stats (all 23 previously failed games recovered)
---

## Standalone: Polymarket In-Game Price History Pipeline

**Status:** Built (branch `feature/polymarket-comeback-analysis`), not integrated with the prediction model — general-purpose per-game win-probability time series (moneyline in-game + moneyline/spread/O-U pre-game snapshots) for comeback analysis and betting-ratio use, independent of feature_builder.py.
**Known limitation (deferred, not blocking):** the BUY/SELL side-split fetch (`data_api.py`) fixes the Data API's 3000-trade offset cap for the vast majority of games, but at extreme Finals-level volume (~6,000+ trades) both sides can still individually hit their own cap (observed directly on `nba-sas-okc-2026-05-26`, 6,427 trades). The Goldsky subgraph fallback from the original design doc was deliberately not built since it wasn't needed for the accepted scope; revisit only if comeback analysis on the highest-volume playoff games turns out to need it.

---

## Phase 2: Dependent Features (Group B)

### B1: Player On/Off Splits Analysis
**Status:** ✅ Implemented and fully iterated (branch `feature/on-off-splits`), **not adopted** — `on_off_splits.enabled: false`. Full backfill, a 5-fold expanding-window CV, and a Doubtful-partial-weighting refinement all confirm the same small/mixed result; this cleared the "conclusive result" bar in the sense that the answer is now conclusively "no clear win," not "not decided yet." See `docs/features/on_off_splits_decisions.md` (phase 1 data-source investigation — the original "A4 lineup data" premise below turned out to be wrong, see Note) and `docs/features/on_off_splits_log.md` (phase 2 implementation + validation, sections 7-8 for the full backfill/CV/weighting rounds) for the full story.
**What was actually built:** direct nba_api `TeamPlayerOnOffSummary` backfill at a checkpoint cadence (not per-game, not play-by-play reconstruction — `LastNGames` was confirmed unusable for leakage-safe historical queries) into a new `player_on_off_splits` cache; `_add_on_off_splits_features` in `feature_builder.py` sums currently-missing (`Out` full weight, `Doubtful` at `injury_features.doubtful_weight`) players' on/off plus-minus per team-game (folds B2's injury-tie-in framing directly into this one feature rather than a separate follow-on step). `vs_opponent` was fetched but ultimately dropped from the feature (small per-season samples, per-game not weekly checkpoints — real, structural volatility, not a coverage gap).
**Real result:** full checkpoint backfill across all 8 seasons the model's training/val/test window touches (2018-19 through 2025-26). A real MAE comparison (`train_model.py`, baseline vs. treatment) and a 5-fold expanding-window CV both came back small and mixed — some metrics consistently favor the treatment (diff_mae, brier, val-side across CV folds), others don't (total_mae, win_acc on val) — not the kind of clean, consistent win that got `style_matchup.raw_features_enabled` adopted. New feature columns consistently rank in the bottom third of ~132 features by importance.
**Decision:** parked, not adopted. Code, backfill data, and tests are merged and safe (feature is a no-op while `enabled: false`) in case the underlying idea is revisited later (e.g. with proper multi-season pooling for `vs_opponent`, which was never built).
**Note (corrects the original plan below):** the original data-source assumption ("actual lineups from box scores" via A4) was wrong — A4's lineup collector only fetches season rosters (`CommonTeamRoster`), never in-game on-court/off-court state, and box scores have no on-court-time data either. nba_api's `TeamPlayerOnOffSummary` endpoint was the real, working data source, confirmed via live API calls.

### B2: Player Availability Impact on Model
**Status:** Folded into B1's actual implementation above — the injury-tie-in ("which currently-missing players, weighted by their on/off impact") IS `_add_on_off_splits_features`, not a separate later step. Not a distinct remaining backlog item.

### B3: Betting Data Integration (Real)
**Status:** Planned (blocked on external data source)
**Goal:** Replace A5/A6 with proper pre-game odds + live odds + historical backfill
**Requirements:** 
  - Pre-game spread/over-under from Vegas or Sports Reference
  - Live in-game odds updates (if available)
  - Historical backfill (5+ years)
**Data sources:** TBD (Sports Reference API? ESPN? Manual CSV download?)
**Output:** DataFrame with game_id, spread, over_under, implied_probability, timestamp
**Note:** Lower priority until data source identified

### B4: Season Motivation Signal (revised from an earlier "tanking/playoff status" idea)
**Status:** ✅ Implemented and fully iterated (branch `feature/season-motivation`,
PR #36) — **partially adopted.** `season_motivation.enabled: true`, but only for
`preferred_opponent_delta` (`preferred_opponent_delta_enabled: true`, window=20);
every other signal tried stays disabled via its own flag
(`motivation_score_enabled: false`, `performance_vs_expectation_enabled: false`,
`opponent_adjusted_form_enabled: false`). See `docs/features/season_motivation_decisions.md`
(data audit + formulas) and `docs/features/season_motivation_log.md` (full validation story,
11 sections + FINAL SUMMARY) for everything tried.
**Real result:** the original standings-pressure/roster-behavior design
(`motivation_score`) looked promising on a single split but did not survive a 5-fold
expanding-window CV (53% of metric-instances favorable, later variants worse), nor
did two later behavior-based signals (passed CV initially, failed a window-robustness
sweep). The one signal that passed CV *and* held up under that same robustness check —
`preferred_opponent_delta` (how much a team's Round 1 opponent would change if its own
seed shifted by one spot) — is what's enabled. No new backfill or DB table was needed
at all for any of this — standings, schedule, and roster-quality data all already
existed in already-complete tables.
**Original idea (dropped, found in earlier session history):** hard seed-range cutoffs
(bottom-4 = tanking, top-4 = secure, 8-14 = playoff race, 8-10 = playin) producing binary
`home_team_tanking`/`home_team_playoff_race` (0/1) columns.
**Why revised:** both the seed cutoffs and the 0/1 output are exactly the kind of
unexplored magic numbers A7's work found repeatedly costly — hardcoded thresholds
guessed upfront rather than derived from data, and a binary flag throws away information
a continuous signal would preserve.
**Revised goal:** a continuous "how much is this team playing to win" signal instead of a
hardcoded seed-bucket flag. Two data-driven ingredients to explore, not assume:
  - Standings-based: distance from the playoff line computed dynamically from actual
    current standings, not a fixed seed range.
  - **Roster-quality-based (new idea):** compare the quality of the players actually in a
    given game's rotation (using A3 player projections/stats) against that team's
    season-long "full-strength" rotation quality. A team resting good, healthy players is
    a more direct tanking signal than a seed threshold, and reuses A3/B1/B2 infrastructure
    already planned rather than needing new data.
**Data sources:** A3 (player quality), standings data, B1/B2 (roster availability).
**Output:** a continuous score (not a 0/1 flag) — if any bucketing is used at all, treat
the cutoffs as a tuned/explored parameter, not a guess.

### B5: Playoff Seed Already Clinched (split off from the same original idea)
**Status:** Folded into B4's actual implementation above —
`games_to_clinch_ceiling`/`games_to_clinch_floor` (continuous countdowns, not a
binary flag) were built as part of the same `feature/season-motivation` branch and
Phase 1 round, sharing the same win-count/games-remaining machinery the
standings-pressure component needs anyway. Not a distinct remaining backlog item.
**Goal:** whether a team's playoff seed (or lottery position band) is already
mathematically locked at the time of a given game, computed cleanly from standings +
remaining schedule — distinct from B4's motivation signal, since a team can have nothing
left to play for without it being "tanking" (e.g. seed locked in with weeks to go).
**Data sources:** standings + remaining schedule (games left, other teams' records).
**Output:** a continuous proxy (e.g. games until mathematically clinched) preferred over
a binary flag, same reasoning as B4.

---

## Phase 3+ Backlog

### Future: Real-Time Injury Pipeline
**Status:** Blocked (requires live nba_api injury reports)
**Goal:** Enable Layer 4 (role mismatch flags) for production use
**Work:** Separate real-time injury scraper + lineup predictor

### Future: Betting Data Integration
**Status:** Blocked (data source TBD)
**Goal:** Complete B3 once data source identified

### Future: Model Retraining with A2 + A7
**Status:** Done — raw-fingerprint style features adopted as the production default
(`style_matchup.raw_features_enabled: true`), retrained and logged as `raw_fingerprint_adopted`
in `outputs/experiments.csv` (reproduces `style_matchup_raw_fingerprint`'s probe numbers exactly:
125 features, val/test total_mae 14.752/15.452 vs baseline `elo_v2`'s 14.958/15.458). KNN-score
variant (`style_matchup.enabled`) confirmed not worth adopting, stays `false`.

### Future: Richer Style-Fingerprint Inputs (Shot Charts)
**Status:** Deferred — deliberately skipped for the first A7 exploration to keep scope contained
**Goal:** Add shot-distance/shot-zone tendencies to Layer 1 fingerprints, beyond the current 5
hand-picked box-score-derived metrics (pace, 3pt reliance, paint activity, def rating, assist rate)
**Data source:** `nba_api` shot-chart/shot-zone endpoints — per-game granular calls, real
backfill time and rate-limit cost, bigger scope than the current box-score-only approach
**Revisit when:** the current 5-metric encoding's ceiling looks limiting (e.g. Phase 2/PCA
exploration plateaus below what richer inputs might unlock)

---

## Notes

- **PR #21 merged** — backfill resilience (backfill_player_stats.py retry logic, recover_failed_backfill.py)
- **Phase 1 Group A PRs** (#14, #16, #18) ready to merge once reviewed
- **A5 rebuilt** (`feature/polymarket-comeback-analysis`, supersedes the original PR #22 collector) — full 2025-26 season already collected (moneyline + spread/O-U), use A6 (OddsPapi) for additional sportsbook coverage if needed
- **A6 planned** — OddsPapi sportsbook integration blocked on API key setup
- **Player stats cache** fully backfilled (1,850,508 stats) — ready for A3 expansion and B1/B2 work
