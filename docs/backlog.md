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

---

## Phase 3: Feature Engineering Refinements

### A7: Style Matchup Score (Architectural Review Needed)

**Current Status:** Design doc complete, but flagged for architecture discussion

**User Concerns & Proposed Solutions:**

#### 1. Player Archetypes — Replace Magic Numbers
**Current:** Hard-coded thresholds
```python
"facilitator": {"ast": (3.5, inf), "ppg": (0, 14)}
"scorer": {"ppg": (15, inf), "ast": (0, 4)}
```

**Options (to decide):**
- **Option A:** Clustering (K-means on PPG, AST, REB, BLK, STL, FG%) — let data define 4-5 roles
- **Option B:** Percentile-based (top 25% AST + bottom 25% PPG = facilitator) — scales with era
- **Option C:** Team-relative archetypes (each team has own "scorer" definition)

**Recommended:** Option B (percentile) — simpler than clustering, more robust than magic numbers

---

#### 2. Injury Impact Shifts — Data-Driven Not Guessed
**Current:** Hard-coded shifts per archetype
```python
"facilitator": {"assist_rate": -0.2, "pace_score": -0.1}
```

**Options (to decide):**
- **Option A:** Empirical (query historical: for each player, measure actual team style shift when out vs in)
- **Option B:** Regression (fit: team_style ~ archetype_availability + backup_availability)
- **Option C:** Config + calibration (base values in config, tune on validation set)

**Recommended:** Option A + store in config — empirical per-player-per-archetype shifts, calibrate from data

---

#### 3. Similarity Metric — Fix Conceptual Gap
**Current Issue:**
```python
similarity = cosine_similarity(normalized_home.values(), normalized_away.values())
```
Compares HOME style to AWAY style directly, but Layer 3 goal is to find similar *historical matchups* (home_style_then vs away_style_then ≈ home_style_now vs away_style_now).

**Fix:** Concatenate into single vector
```python
current_matchup = [home_style_metrics] + [away_style_metrics]
historical_matchup = [historical_home_metrics] + [historical_away_metrics]
similarity = cosine_similarity(current_matchup, historical_matchup)
```

---

#### 4. Magic Numbers → Config
**Move to config.yaml:**
```yaml
style_matchup:
  fingerprint_window: 20           # games
  decay_halflife: 5                # games
  similarity_threshold: 0.80       # cosine
  min_sample_size: 10              # games
  archetype_method: "percentile"   # or "clustering", "hardcoded"
  archetype_percentiles:
    facilitator: {"ast_pct": 0.75, "ppg_pct": 0.25}
    scorer: {"ppg_pct": 0.75, "ast_pct": 0.25}
    # ... etc
  injury_calibration: "empirical"  # or "static", "regression"
```

**Add:** Calibration script to tune thresholds on validation set

---

#### 5. Known Infra Gaps (found during implementation planning)

**Gap 1 — Phase 1 fingerprint inputs don't exist where expected.**
`data/raw/nba_api.sqlite`'s `game` table only stores fg_pct/ft_pct/fg3_pct/ast/reb —
no FGA, FGM, FTA, FG3A, TOV, which the pace/paint/3pt-reliance/assist-rate formulas
need. `src/data_processing/fetch_data.py` already calls `nba_api.stats.endpoints.
LeagueGameLog` for this table; that endpoint returns the missing columns too, they're
just not selected/stored today.
**Fix:** source raw box-score inputs from a fresh `LeagueGameLog` call (same
convention as `fetch_data.py`), cached in a new additive table (not an `ALTER TABLE`
on `game` — mirror the `player_stats_cache` pattern instead). Do **not** use
`data/raw/basketball.sqlite` — a static one-time Kaggle dump (last touched
2023-07-06), not part of the live pipeline, and not kept in sync.

**Gap 2 — Phase 0 calibration needs player_id, injury records only have player_name.**
`player_injuries` (in `data/raw/injury_features.sqlite`) stores `player_name` as free
text with no `player_id`; `player_stats_cache` (needed for archetype classification)
is keyed by `player_id`. No existing join key between them.
**Fix:** resolve names via `nba_api.stats.static.players.get_players()` (same
convention `src/news_scraping/pipeline.py:_resolve_team_id` already uses for teams
via `nba_api.stats.static.teams`), disambiguating duplicate names using whichever
candidate has `player_stats_cache` activity near the injury date. Track the overall
resolution coverage rate — if low, Phase 0's calibration deltas are unreliable and
that must be flagged, not silently ignored. Note: the doc's own Layer ablation
(L1 only vs. L1+2 vs. L1+2+3) acts as a natural check here — if a bad join corrupts
Layer 2, it'll show up as L1+2 correlating *worse* than L1 alone, not as a silently
wrong "style signal works" conclusion.

---

### A7 Decision Matrix

| Item | Option A | Option B | Option C | Recommendation |
|------|----------|----------|----------|---|
| Archetypes | Clustering (complex) | Percentile (balanced) | Team-relative (overkill) | **B** |
| Injury shifts | Empirical (data-driven) | Regression (model-based) | Config + tune (simpler) | **A + store in config** |
| Similarity | Concat vectors | Separate metrics | Current (wrong) | **Concat vectors** |
| Config | All hardcoded | All in config | Hybrid | **All in config** |

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

---

## Decision Checklist for A7

- [ ] Decide: Archetype method (percentile recommended)
- [ ] Decide: Injury shift approach (empirical + config recommended)
- [ ] Decide: Similarity metric fix (concat vectors)
- [ ] Decide: Go with percentile implementation, empirical injury, config-everything approach?

Once decided → implement in feature/a7-style-matchup

---

## Notes

- **PR #21, #22 merged** — backfill resilience + A5 robust Polymarket collector ready
- **Phase 1 Group A PRs** (#14, #16, #18) ready to merge once reviewed
- **A5 complete** (real Polymarket data) — playoffs/championships focus, use A6 (OddsPapi) for full season
- **A6 planned** — OddsPapi sportsbook integration blocked on API key setup
- **Player stats cache** fully backfilled (1,850,508 stats) — ready for A3 expansion and B1/B2 work
