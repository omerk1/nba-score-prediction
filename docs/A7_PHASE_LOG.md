# A7 Style Matchup — Phase Log

All work lives in `src/matchups/` (new module, not imported by `feature_builder.py`
or any training path). Cache tables are additive-only, in a new file
`outputs/a7_matchups_cache.sqlite` — `data/raw/nba_api.sqlite` and
`data/raw/injury_features.sqlite` are opened strictly read-only throughout (SQLite
URI `mode=ro`), since they are symlinked in from the human's live working copy of
the repo.

---

### Phase 0 — Injury impact calibration (+ Gap 1 / Gap 2 foundational infra)
**Status:** complete

**What was built:**
- `src/matchups/box_scores.py` (Gap 1): fetches box scores (FGM/FGA/FG3M/FG3A/FTM/FTA/
  OREB/DREB/AST/STL/BLK/TOV/PTS) via a fresh `nba_api.stats.endpoints.LeagueGameLog`
  call, reusing `SEASON_TYPES`/`_season_list`/`_date_to_season` from
  `src/data_processing/fetch_data.py` (read-only import). Caches into
  `outputs/a7_matchups_cache.sqlite:box_score_stats`.
- `src/matchups/players.py` (Gap 2): resolves `player_injuries.player_name` (free
  text) to `player_id` via `nba_api.stats.static.players.get_players()`, with
  normalization (accent-strip, casefold, strip periods, strip Jr./Sr./II/III/IV/V
  suffixes) tried first and an unnormalized exact-match fallback second.
  Disambiguates multi-candidate names using `player_stats_cache` activity within
  60 days of the player's earliest `player_injuries.game_date`. Also computes
  percentile-based archetype classification per player-season.
- `src/matchups/calibration.py` (Phase 0 proper): empirical injury-impact deltas —
  for each archetype, compares Layer-1 fingerprint metrics in team-games where that
  archetype was reported `Out` (pre-game NBA official PDF report) vs. the same
  team-season's baseline games. Writes results to `injury_calibration` cache table
  and appends a `style_matchup` block to `configs/config.yaml` (plain-text append,
  not a YAML round-trip — see Fallbacks below).
- `src/matchups/fingerprint.py` built as a dependency (Layer 1 raw fingerprints
  needed by calibration) — logged in detail under Phase 1 below since it's that
  phase's actual deliverable.

**Required parity check (Gap 1):** fetched box scores for the exact `game_id` set
already in the `game` table (2016-10-01 start, Regular Season + Playoffs). Result:
**12,793 / 12,793 game_ids matched 1:1, 0 missing, 0 extra.** A7's rolling windows
reference the same game set A2/H2H uses.

**Name resolution coverage (Gap 2):** 1,028 distinct `player_injuries.player_name`
values. **935 resolved this run** (normalization handled cases like "AJ Green" vs
"A.J. Green"), 92 unmatched (mostly true aliases the normalization rules don't
cover, e.g. "Alex Sarr" vs. static-list "Alexandre Sarr" — not a
whitespace/case/suffix/accent issue, out of scope per instructions), 1 ambiguous.
**Overall coverage rate: 90.95%** (high+medium confidence / distinct names) —
**above the 80% minimum**, so calibration deltas below are NOT marked low-confidence.

**Archetype taxonomy — widened beyond the design doc's fixed 4 categories, per
explicit course-correction mid-phase.** Two things were tried, not just threshold
tuning:

1. *Per-archetype independent threshold grid* (not one shared percentile knob):
   - facilitator (`ast_pct>=hi, ppg_pct<=1-hi`): hi=0.60→114, 0.65→40, 0.70→14,
     0.75→1 (design default). **Kept 0.65** (40 player-seasons; 0.75 has no
     statistical power).
   - scorer (`ppg_pct>=hi, ast_pct<=1-hi`): hi=0.60→73, 0.65→23, 0.70→8, 0.75→2
     (design default). **Kept 0.65** (23 player-seasons).
   - rim_protector (`blk_pct>=t1, reb_pct>=t2`, t1/t2 varied independently 0.60-0.80):
     908-1667 player-seasons across the whole grid — already ample at the design
     default (0.75/0.75, n=908). **Kept 0.75/0.75**, no need to loosen.
   - perimeter_specialist (`blk_pct<=t1, stl_pct>=t2`, varied independently
     0.15-0.35 / 0.60-0.80): design default (0.25/0.75) gave only 31. **Loosened to
     0.30/0.70** (80 player-seasons) — still a selective "low rim protection, high
     steal rate" profile.
2. *A genuinely different taxonomy, not just different thresholds*: KMeans
   (k=4,5,6,8) on standardized [PPG, AST, REB, BLK, STL, FG%] per player-season
   (`docs/backlog.md`'s A7 Option A). **Finding: clusters separate almost entirely
   by playing-time/usage tier (bench garbage-time / low-usage bench / rotation /
   starter-star), not by style** — at every k, cluster centroids move on all 6
   stats together (e.g. k=8's cluster 2: PPG=20.4 AST=6.2 REB=5.1 BLK=0.43 STL=1.26
   vs cluster 1: PPG=1.0 AST=0.3 REB=0.6 BLK=0.05 STL=0.10 — a monotonic "more of
   everything" axis, not a stylistic split). This happens because
   `player_stats_cache` only has PPG/AST/REB/BLK/STL/FG% — no minutes, usage rate,
   or shot-location data to separate "how much" from "how". **Decision: kept the
   percentile approach** (already era/season-adjusted via within-season ranking)
   as primary; clustering isn't a clear win given the available stats and would
   need per-minute/per-possession inputs (not present in this table) to actually
   separate style from playing time.

**Taxonomy gaps addressed:**
- Added **`combo`** archetype (`ppg_pct>=0.85 AND ast_pct>=0.85`, n=496): the
  design doc's facilitator/scorer are mutually exclusive by construction (high AST
  requires low PPG and vice versa), which drops genuine dual-threat
  playmaker-scorers entirely. Threshold set high (0.85, not e.g. 0.70→n=1135)
  because PPG and AST both scale with playing time — a lower bar mostly just
  re-selects "played a lot of minutes," not a distinct style.
- Considered and **rejected** a third "versatile_defender" mid-band bucket between
  rim_protector and perimeter_specialist (to address "nothing in between" on the
  defensive spectrum). Tested BLK/STL percentile bands (0.40-0.75): captured a
  diffuse 11-15% "everyone in the middle" group with no distinct separation —
  not a real archetype given only 6 available stats (no minutes/usage/matchup
  data to define a genuine third defensive profile). Not added.
- Final taxonomy: **facilitator, scorer, combo, rim_protector,
  perimeter_specialist** (5 archetypes; `nan`/unclassified is the majority class,
  4,076 of 5,426 player-seasons — expected, most players don't have an extreme
  statistical profile).

**Empirical injury-impact deltas (replacing v1 estimates):**

| archetype | metric | delta | n_without | n_baseline | design v1 guess | direction match? |
|---|---|---|---|---|---|---|
| facilitator | assist_rate | -0.0066 | 273 | 1281 | -0.15 | yes (weak magnitude) |
| facilitator | pace_score | -1.2722 | 273 | 1281 | -0.1 | yes |
| scorer | three_pt_reliance | +0.0237 | 123 | 615 | +0.1 | yes |
| scorer | paint_activity | +0.344 | 123 | 615 | +0.1 | yes |
| rim_protector | defensive_rating | +0.5376 | 3849 | 7775 | +2.5 | yes (weaker) |
| rim_protector | paint_activity | -0.2716 | 3849 | 7775 | -0.15 | yes |
| perimeter_specialist | defensive_rating | **-0.3131** | 407 | 1886 | +1.5 | **NO — opposite sign** |
| combo (new) | all 5 metrics | see config.yaml | 3340 | 8202 | n/a (no v1 guess) | n/a |

**Key findings:**
- Directionally, 3 of 4 design-doc archetypes' calibrated deltas match the v1
  guessed sign (facilitator, scorer, rim_protector), just smaller in magnitude —
  the v1 estimates were reasonable ballpark guesses, just too large.
- **perimeter_specialist is the outlier: defensive_rating improves (goes down)
  when the team's perimeter specialist is out**, the opposite of the v1 guess.
  Plausible explanations: (a) small sample (407 games) and noise, (b) confound —
  perimeter specialists in this classification often play alongside a strong
  rim_protector, and the archetype is defined by *low* BLK, so this could be
  capturing lineup/opponent-quality effects rather than a causal individual
  effect. **Flagging for human review — do not treat this sign as ground truth
  without further investigation.**
- The Layer-3 built-in check mentioned in the instructions (L1 vs L1+2 vs L1+2+3
  correlation) is deferred to Phase 4's ablation, where it will surface if this
  perimeter_specialist sign flip actually hurts the injury-adjusted signal.

**Fallbacks used:**
- `write_deltas_to_config` appends a plain-text `style_matchup:` block to
  `configs/config.yaml` instead of `yaml.safe_load` + `yaml.safe_dump` round-trip.
  The round-trip was tried first and found to **strip every inline `#` comment in
  the existing file** (PyYAML doesn't preserve comments) and reformat
  quoting/indentation — caught via `git diff` before committing, reverted with
  `git checkout --`, and replaced with an idempotent text append guarded by a
  `"style_matchup:" in existing` check.
- `config_loader.py`'s `Config` pydantic schema is not extended to know about
  `style_matchup` (would require editing a file outside `src/matchups/`).
  `src/matchups/config.py` reads the block directly via `yaml.safe_load` as a raw
  dict, with hardcoded design-doc defaults for any missing keys.
- Cache DB placed at `outputs/a7_matchups_cache.sqlite` rather than under `data/`
  — the instructions permit new writes only in `src/matchups/`, `outputs/`,
  `docs/`, and `config.yaml`, so `data/processed/` was not used even though it
  would have been the more conventional location for a cache DB.
- `min_games` floor of 20 games/season for archetype eligibility, and 5 games
  (reusing `features.min_games_played` from the main config) for fingerprint
  validity — not in the design doc, added to avoid noisy small-sample
  classifications/fingerprints.

**Metrics:**
- 12,793/12,793 game_id parity (Gap 1).
- 90.95% name resolution coverage (935/1028 resolved to high/medium confidence).
- 5,426 player-seasons classified; 1,350 (25%) fall into one of the 5 archetypes.
- Calibration sample sizes range from 123 (scorer) to 3,849 (rim_protector) games.

**Magic numbers explored:** see per-archetype grid above. `min_games=20` for
archetype eligibility and `min_games=5` for fingerprint validity were not gridded
(reused existing config value / picked as a sane floor) — flagged as
not-deeply-explored if a human wants to revisit.

**Next phase dependencies:** Phase 1 (fingerprints) already had to be built as a
dependency of this phase — see its own entry below for that phase's specific
findings (rolling window / decay behavior, cache size). Phase 2 (Layer 2 injury
adjustment) will read `injury_impact` from `configs/config.yaml`'s new
`style_matchup` block (empirically calibrated) and `severity_weights` from the
existing `injury_features.severity_weights` (reused, not duplicated).

---

### Phase 1 — Layer 1: fingerprints + config + historical index
**Status:** complete

**What was built:**
- `src/matchups/fingerprint.py`: rolling pre-game style fingerprint per
  (game_id, team_id) — `pace_score`, `three_pt_reliance`, `paint_activity`,
  `defensive_rating`, `assist_rate` — computed from `box_score_stats` (Gap 1
  cache). Strictly pre-game: each team's own game-day row is `.shift(1)`-ed off
  before the rolling window is applied (rows are pre-sorted per-team by
  `game_date`, one row per game, so `shift(1)` == "all games strictly before this
  one" — equivalent to an explicit `date < game_date` filter but avoids a
  quadratic self-join). Window = 20 games, exponential decay half-life = 5 games
  (both from `style_matchup.fingerprint_window` / `decay_halflife`, per design
  doc defaults — not re-tuned this phase, see Magic Numbers below).
- `src/matchups/matchup_index.py`: builds the design doc's target artifact — one
  row per game with `matchup_vector` (10 values: 5 home + 5 away, z-scored) and
  `actual_home_margin`. Encoding = hand-picked (Encoding Phase 1), per pre-settled
  decision.

**Key findings:**
- 25,436 of 25,586 team-games (99.4%) produced a valid fingerprint (the rest —
  early-season team-games with `n_games_in_window < 5`, reusing
  `features.min_games_played` — were dropped, not zero-filled).
- 12,714 of 12,793 games (99.4%) have valid vectors for BOTH teams and a final
  score, forming the historical index.
- **Zero NaNs** in the 12,714 × 10 matchup vector matrix (design doc sanity check).
- Quick sanity read (raw home-away diff per dimension vs. `actual_home_margin`,
  Layer 1 only, no similarity search yet): `defensive_rating` diff correlates
  **-0.2313** with margin (higher home defensive_rating relative to away → home
  team wins by less — correct sign, meaningful magnitude even before any
  similarity-search aggregation). `three_pt_reliance` +0.088, `paint_activity`
  +0.037, `pace_score` -0.043, `assist_rate` +0.024 — small but present. This is
  encouraging for Phase 3/4 (a real Layer-3 similarity search that borrows sample
  size across similar matchups should sharpen this further) — the formal
  correlation gate (< 0.2 → escalate to PCA) is evaluated properly in Phase 4 on
  the actual style_matchup_score, not on raw per-dimension diffs.

**Fallbacks used / design-doc-underspecified choices:**
- Normalization method for "normalize before concatenating" (design doc doesn't
  specify): used **z-score** (mean/std across the full fingerprint history for
  the layer), not min-max. Reasoning: min-max is more sensitive to outlier games
  early in a rolling window (small `n_games_in_window` → noisier extremes); with
  z-score, cosine similarity in Phase 3 is driven by relative deviation from
  league-average style rather than raw scale.
- `possessions` estimate for `defensive_rating` uses the standard box-score
  formula `FGA - OREB + TOV + 0.44*FTA` (own team's box line) — the design doc
  doesn't specify a possession formula.
- The half-life-based rolling window is implemented as `.shift(1).rolling(20,
  min_periods=1).apply(decayed_weighted_mean)` rather than a literal `df[df['date']
  < game_date]` filter + decay — mathematically equivalent here since rows are
  pre-sorted one-per-team-game (flagged explicitly since the design doc calls out
  this exact leakage risk with `.shift()`).

**Metrics:** 25,436 team-game fingerprints (layer=1) cached; 12,714 games in the
historical index; 0 NaN; sanity-check correlations reported above.

**Magic numbers explored:** `fingerprint_window=20` and `decay_halflife=5` were
**not** gridded this phase — kept at design doc defaults since Phase 0's archetype
exploration and calibration already consumed the phase's exploration budget, and
these two parameters are more naturally tuned against the Phase 4 correlation
metric (where the cost of re-running is one fingerprint rebuild, not a
re-architecture). Flagged as a candidate for revisiting if Phase 4's correlation
is marginal. `min_games=5` for fingerprint eligibility reuses
`features.min_games_played` (not a new magic number).

**Next phase dependencies:** Phase 2 (Layer 2 injury adjustment) needs to produce
layer=2 fingerprints (same schema, `matchup_fingerprints` table, `layer=2`) by
applying calibrated deltas on top of these Layer 1 fingerprints, then Phase 3
rebuilds the matchup index at `layer=2` for the similarity search.

---
