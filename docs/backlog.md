# Project Backlog

## Phase 1 Status (Complete)

### Group A — Independent Baseline Features (All Complete, Ready to Merge)

- **A1** ✅ ELO Hyperparameter Tuning v2 (committed to main)
- **A2** ✅ Extended H2H Features (PR #16, ready to merge)
- **A3** ✅ Player Box Score Projections (PR #18, awaiting expansion confirmation to 6 stats)
- **A4** ✅ Lineup Data Collection (PR #14, ready to merge)
- **A5** ✅ Polymarket Signals (PR #22, merged — real Polymarket API data, robust backfill, playoffs/championships focus)
- **A6** 🔄 OddsPapi Sportsbook Signals (planned — full season coverage, 250+ bookmakers)

### Backfill Infrastructure ✅
- **Backfill Resilience** (PR #21, merged)
  - Exponential backoff retry in backfill_player_stats.py
  - recover_failed_backfill.py script for manual recovery
  - Database fully recovered: 1,850,508 stats (all 23 previously failed games recovered)
- **A5 Robust Backfill** (PR #22, merged)
  - Error tracking and recovery script (recover_polymarket_failed.py)
  - Incremental progress saves (resume-safe)
  - 529 playoff/championship odds collected (93.3% success rate)

---

## Phase 2: Dependent Features (Group B)

### B1: Player On/Off Splits Analysis
**Status:** Planned (depends on A3 player projections, A4 lineup data)
**Goal:** Compute +/- impact of each player on team performance (home/away, vs specific opponents)
**Data sources:** A3 (player stats), A4 (actual lineups from box scores)
**Output:** DataFrame with player_id, on_off_plus_minus, vs_opponent splits
**Note:** Use historical lineups from box scores; real-time injury data (future work)

### B2: Player Availability Impact on Model
**Status:** Planned (depends on A3, B1)
**Goal:** Integrate B1 on/off metrics into feature_builder.py as injury-aware features
**Output:** New columns: player_availability_impact, team_roster_strength_delta
**Integration:** _add_player_features() method in feature_builder.py

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
**Status:** Planned — revised before building, original spec was too rigid
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
**Status:** Planned — split out because it's a genuinely different signal from B4
**Goal:** whether a team's playoff seed (or lottery position band) is already
mathematically locked at the time of a given game, computed cleanly from standings +
remaining schedule — distinct from B4's motivation signal, since a team can have nothing
left to play for without it being "tanking" (e.g. seed locked in with weeks to go).
**Data sources:** standings + remaining schedule (games left, other teams' records).
**Output:** a continuous proxy (e.g. games until mathematically clinched) preferred over
a binary flag, same reasoning as B4.

---

## Phase 3: Feature Engineering Refinements

### A7: Style Matchup Score

**Current Status:** Implemented and validated on branch `feature/a7-style-matchup` — Phases
0-5 complete, plus five follow-up rounds: hyperparameter search + PCA/clustering/
supervised-model comparison, a walk-forward CV robustness check (tuned config wins on
every fold), a wrap-up round (fixed a z-score normalization leak, added minutes/usage data
to archetype classification, isolated injury-adjustment's real marginal contribution), a
decay-weighted calibration fix (resolved the perimeter_specialist sign flip), and a
confirmatory hyperparameter-search rerun (confirmed the standing config still wins under
the normalization fix — no change). Style signal robustly beats the A2 H2H baseline across
multiple independent validation folds. `style_matchup` is now a formally typed section of
`configs/config.yaml` (`src/utils/config_loader.py`'s `StyleMatchupConfig`).
Not yet integrated into `feature_builder.py`.

All architecture decisions that used to be open here have been decided, implemented, and —
where later questioned — re-validated with fresh scrutiny; see `docs/a7_phase_log.md` for
what was actually built, tried, and found, rather than duplicating that detail here. The `combo`
archetype was also redefined using real usage-rate data (previously a workaround); the
recalibrated `injury_impact` values are already in `configs/config.yaml`.

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
**Status:** Post-A7 validation
**Goal:** Integrate style_matchup features into model, measure improvement vs A2 alone

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

- **PR #21, #22 merged** — backfill resilience + A5 robust Polymarket collector ready
- **Phase 1 Group A PRs** (#14, #16, #18) ready to merge once reviewed
- **A5 complete** (real Polymarket data) — playoffs/championships focus, use A6 (OddsPapi) for full season
- **A6 planned** — OddsPapi sportsbook integration blocked on API key setup
- **Player stats cache** fully backfilled (1,850,508 stats) — ready for A3 expansion and B1/B2 work
