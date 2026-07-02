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

### Phase 2 — Layer 2: injury adjustment
**Status:** complete

**What was built:** `src/matchups/injury_layer.py` reads layer=1 fingerprints,
finds Out players per (team, game_date) via the Phase 0 name-resolution +
archetype tables, classifies each Out player's severity from
`player_injuries.reason` using the **existing**
`src/news_scraping/extractors/formula_scorer.classify_severity()` against the
**existing** `injury_features.severity_weights` config (both reused, not
duplicated), and applies the Phase 0 calibrated deltas to produce layer=2
fingerprints (same `matchup_fingerprints` schema, `layer=2`).

**Key findings:**
- 6,169 / 25,436 team-games (24.25%) received at least one archetype-matched
  adjustment. This is consistent with the PDF injury-report era only covering
  2021-10-19 onward (roughly half the 2016-2026 dataset), combined with
  injuries/archetype-classified players being a genuinely common but not
  universal occurrence within that era.

**Fallbacks used / design decisions (not fully specified by the design doc):**
- **Multiple players of the same archetype out simultaneously:** the archetype's
  delta is applied ONCE per game, scaled by the MAX severity multiplier among
  those players — not summed. The calibration (Phase 0) estimated a binary
  "this archetype was missing" effect, not a per-player marginal effect, so
  summing multiple same-archetype absences would double-count.
- **Different archetypes out simultaneously:** deltas DO stack additively (each
  archetype's `injury_impact` block targets its own metrics independently, per
  the design doc's per-archetype config structure).
- Only `status == 'Out'` players are counted (not Doubtful/Questionable) — kept
  consistent with how Phase 0's calibration defined "missing" (also Out-only),
  so the deltas being applied match the deltas that were measured.

**Metrics:** 25,436 layer=1 rows in, 25,436 layer=2 rows out (1:1, every row
either adjusted or passed through unchanged); 24.25% adjustment rate.

**Magic numbers explored:** none new this phase — severity multipliers reused
directly from `injury_features.severity_weights` (severe=1.0, moderate=0.6,
minor=0.3), no new knob introduced.

**Next phase dependencies:** Phase 3 (similarity search) operates on layer=2
vectors via `matchup_index.build_matchup_index(layer=2)`.

---

### Phase 3 — Layer 3: similarity search (cosine + KNN)
**Status:** complete

**What was built:**
- `src/matchups/baseline_a2.py`: standalone A2 H2H re-implementation (expanding
  mean of canonical-margin, shifted, matchup-key-based) — used both as the
  low-confidence fallback score and as the A2 comparison baseline in Phase 4.
  Full-dataset `corr(h2h_score, actual_home_margin) = 0.1324`.
- `src/matchups/similarity.py`: both cosine-threshold and KNN search over
  layer=2 matchup vectors, with the leakage guard implemented via
  `np.searchsorted` on a date-sorted vector array (excludes ALL games on the
  same date as the target, not just earlier row positions — multiple games per
  night must not see each other). Confidence = `min(n_similar /
  full_confidence_sample, 1.0)`; below `min_confidence_sample`, falls back to
  the A2 H2H score (pre-settled decision), not zero.

**Pre-settled comparison (cosine @ 0.70 vs KNN k=30), evaluated on 3,922 games
from 2023-10-01 onward against actual home margin:**

| method | fallback_rate | mean_confidence | corr vs margin |
|---|---|---|---|
| cosine @ 0.70 | 0.03% | 0.992 | **0.2806** |
| KNN k=30 | 0.0% | 0.600 (constant) | 0.2482 |

**Winner: cosine @ 0.70** (higher correlation, and per the design doc's stated
default). Kept as the Phase 4 primary configuration; both are still reported in
the ablation CSV per instructions.

**Magic numbers explored (wider, not just this one knob — two genuinely
different search strategies were already the main comparison above; within each,
a value sweep was run):**
- Cosine threshold sweep: 0.5→corr 0.282, 0.6→**0.285 (best)**, 0.7→0.281,
  0.8→0.241 (fallback jumps to 5.9%), 0.9→0.123 (fallback 89%), 0.95→0.118
  (fallback 100%, degenerates to pure H2H). **Confirms the design doc's specific
  claim that 0.80 is too aggressive** — 0.70 (or even 0.60) clearly outperforms
  0.80 empirically, not just per design-doc intuition. 0.70 was kept as the
  config default since it's within noise of the 0.6 optimum and leaves more
  margin before the correlation collapse that starts around 0.8.
- KNN k sweep: k=10→0.180, k=20→0.221, k=30→0.248, k=50→0.268, k=100→**0.284**.
  Correlation rises monotonically with k — **the design doc's recommended k=30
  is not the best setting**; k=100 nearly matches cosine's performance. If KNN
  were chosen as the production method, k should be higher than 30. Not changed
  in config since cosine won overall, but flagged for any future revisit of KNN.
- **KNN confidence-scoring weakness found**: because top-K always returns
  exactly K neighbors (once enough history exists), `n_similar` is a constant
  (=k), so `confidence = min(k/50, 1.0)` is **also constant** — it carries no
  information about actual match quality, unlike cosine's confidence (which
  varies with how many games clear the threshold). This is a real limitation of
  the KNN confidence definition, not just a parameter choice — noted as a
  reason cosine is preferable beyond its raw correlation edge.

**Sanity checks (design doc's Validation Plan, evaluated on the winning cosine
@0.70 config):**
- Score range: **[-9.78, 11.24]**, within the required [-15, 15]. ✓
- Confidence range: **[0.1, 1.0]**, within [0, 1]. ✓
- **0 NaN** in style_score. ✓
- Fallback rate **0.03%**, well under the 20% ceiling. ✓

**Next phase dependencies:** Phase 4 uses `similarity.py` directly for the
L1+2+3 ablation rows and `baseline_a2.py` for corr_a2_alone / corr_a2_plus_a7.

---

### Phase 4 — Validation + ablation CSV
**Status:** complete

**What was built:** `src/matchups/validate.py`, appending 12 rows to
`outputs/a7_style_matchup_results.csv` (DictWriter, header-on-first-write,
matching `train_model.py`/`tune_elo.py`'s pattern). Evaluation set: 3,922 games
from 2023-10-01 through the most recent data (2026), searched against all prior
games in the 2016-2026 history.

**Layer ablation results (the core required comparison):**

| config | corr_a7_alone | corr_a2_alone | corr_a2_plus_a7 |
|---|---|---|---|
| L1 only (naive diff sum, no search) | **-0.143** | 0.118 | 0.189 |
| L1+L2 (naive diff sum, no search) | **-0.140** | 0.118 | 0.187 |
| L1+L2+L3 cosine @0.70 | **0.281** | 0.118 | **0.296** |
| L1+L2+L3 KNN k=30 | 0.248 | 0.118 | 0.267 |

**Key finding — this is not a bug, it validates the design doc's own thesis.**
"Layer 1 only" and "Layer 1+2" were operationalized as a zero-parameter naive
score: sum of the 5 z-scored home-away metric diffs, with no similarity search.
That naive score correlates **negatively** with margin (-0.14). This is exactly
what the design doc predicts in its "Why not compare team styles directly?"
section: *"The naive approach — comparing home style vector to away style vector
— tells you these teams play differently from each other. That's not useful."*
A plain unweighted diff-sum has no principled sign per dimension (e.g.
`defensive_rating` is "lower is better" while the other four are neutral style
descriptors, not efficiency metrics) — summing them blind is exactly the
"naive" anti-pattern the design doc warns against. It takes Layer 3 (searching
for historical games with a similar matchup vector and using THEIR actual
outcome, rather than guessing a sign/weight per dimension) to turn the
fingerprint into real signal: correlation goes from -0.14 (naive diff) to
+0.28 (similarity search) using the exact same underlying fingerprints.
(Aside: the OLS-combined `corr_a2_plus_a7` for the naive rows, 0.19, is higher
than either alone — an OLS fit CAN learn the right sign/weight per dimension,
which is what Layer 3 does implicitly via nearest-neighbor averaging instead of
a global linear fit. Consistent, not contradictory.)

**Does A7 beat the A2 H2H baseline? Yes, clearly, on this dataset:**
corr_a7_alone (0.281, cosine) more than doubles corr_a2_alone (0.118).
corr_a2_plus_a7 (0.296) is only marginally above corr_a7_alone alone (0.281),
meaning **A7 mostly subsumes A2's signal** rather than being a small addition on
top of it — but combining still helps a little, so A7 does not make A2
completely redundant.

**Confidence calibration:** cosine @0.70, split at confidence 0.5 — MAE
high-confidence = 12.31, MAE low-confidence = 10.78 on the sweep rows (this
split has very few low-confidence games since fallback rate is only 0.03%, so
this comparison has low statistical power at the default threshold; the cosine
threshold=0.8 sweep row, which has a real low-confidence bucket, is more
informative: MAE high=12.48, MAE low=11.96 — roughly comparable, not the clean
"high-confidence should have lower MAE" pattern the design doc hypothesizes.
**Flagging this as a finding, not glossing over it**: on this evaluation set,
confidence (driven mostly by *how much* history is available) does not clearly
predict *accuracy* the way the design doc assumed. Plausible reason: even
"low-confidence" games still fall back to a reasonable H2H score rather than a
wild guess, capping how bad low-confidence MAE can get.

**Sanity check — last 100 games (design doc's specific validation ask):** MAE of
style_score vs. actual margin = 14.56, vs. MAE of h2h_score vs. actual margin =
15.26. Style score is more accurate on the most recent 100 games too, not just
in aggregate correlation.

**Encoding decision gate (pre-settled: hand-picked unless correlation < 0.2):**
corr_a7_alone = **0.281 > 0.2** → the hand-picked encoding is NOT weak. **Phase 5
(PCA) is not required** — see its entry below for the formal skip rationale.

**Metrics:** 12 ablation rows written to `outputs/a7_style_matchup_results.csv`;
3,922 games evaluated per row; full results in that file (columns: run_name,
encoding_phase, similarity_method, similarity_threshold_or_k, layers_enabled,
n_games_evaluated, fallback_rate, mean_confidence, corr_style_vs_margin,
mae_high_conf, mae_low_conf, corr_a2_alone, corr_a7_alone, corr_a2_plus_a7,
notes).

**Magic numbers explored:** the cosine-threshold and KNN-k sweeps from Phase 3
are re-recorded here as their own CSV rows (not just prose) so they're
queryable alongside the main ablation rows.

**Next phase dependencies:** Phase 5 is conditional and, per the decision gate
above, skipped (documented, not silently omitted).

---

### Phase 5 — Encoding upgrade (PCA)
**Status:** skipped (condition not met — documented per instructions, not silently omitted)

**Decision:** The design doc and task instructions make Phase 5 conditional:
*"only run if Phase 4 hand-picked signal is weak"* / *"If Phase 1 shows weak
correlation (<0.2) with actual margins... move to PCA."* Phase 4 measured
`corr_a7_alone = 0.281` for the winning configuration (cosine @0.70,
layer=2 injury-adjusted fingerprints) — well above the 0.2 threshold. The
hand-picked encoding (Encoding Phase 1) is validated as sufficient; there is no
signal-quality justification to spend the added complexity/leakage-risk of
fitting a PCA transform. No PCA code was written.

**What would trigger revisiting this:** if a human reviewer wants richer/less
correlated dimensions for other reasons (e.g. more interpretable components, or
extending the fingerprint beyond 5 hand-picked metrics), Phase 5 remains
available as a future addition — `configs/config.yaml`'s `style_matchup.encoding`
already has a `pca` option reserved (unused) for this.

---

## FINAL SUMMARY

**Similarity method winner and why:** Cosine similarity @ threshold 0.70, over
KNN k=30. Cosine scored higher correlation with actual margin (0.281 vs 0.248)
at a comparable, near-zero fallback rate, and — separately from raw correlation
— KNN's confidence score is degenerate (constant at `k/50` regardless of match
quality, since top-K always returns exactly K neighbors once enough history
exists), whereas cosine's confidence genuinely varies with data availability.
A threshold/k sweep (0.5-0.95 for cosine, 10-100 for KNN) confirmed the design
doc's specific claim that the "obvious" 0.80 threshold is too aggressive
(correlation collapses past 0.8 as fallback rate spikes) and additionally found
that KNN's design-doc-recommended k=30 is not its optimum (k=100 nearly
matches cosine) — logged as a finding, not adopted as the default since cosine
already won outright.

**Encoding used and why:** Hand-picked (Encoding Phase 1), per the pre-settled
default. Validated, not just assumed: `corr_a7_alone = 0.281` on the winning
configuration, well above the 0.2 escalation threshold, so Phase 5 (PCA) was
correctly skipped per its conditional trigger. Encoding taxonomy for the
*archetype* side (used in Layer 2) was widened beyond the design doc's fixed 4
categories after a mid-task course-correction: added a `combo` archetype (dual
scorer+facilitator) and evaluated (then rejected) a mid-band defensive archetype
and a full KMeans-clustering alternative — see Phase 0 for the detailed
comparison. Clustering was found to mostly recover playing-time tiers rather
than style, given the limited stat set (`player_stats_cache` has no
minutes/usage/shot-location data), which is itself a useful finding for anyone
extending this later.

**Does the style signal beat the A2 H2H baseline?** Yes, clearly, on this
dataset (12,714-game history, 3,922-game 2023-2026 evaluation window):
corr_a7_alone (0.281) more than doubles corr_a2_alone (0.118). Combining both
(corr_a2_plus_a7 = 0.296) only marginally beats A7 alone — **A7 mostly subsumes
A2's signal rather than being purely additive to it**, but does not make A2
fully redundant. The naive (no-search) "Layer 1 only"/"Layer 1+2" ablation rows
correlate *negatively* with margin (-0.14) — this is not a bug, it is the design
doc's own predicted failure mode for directly comparing style vectors without a
similarity search, and it is exactly why Layer 3 (borrowing the actual outcome
from similar historical matchups, rather than guessing a sign per dimension) is
the component that turns the fingerprint into usable signal.

**Key magic number findings:**
- Archetype percentile thresholds should be tuned per-archetype, not as one
  shared knob — facilitator/scorer needed loosening (0.75→0.65) to get any
  statistical power, while rim_protector was already fine at the design default.
- Cosine threshold: 0.70 (or 0.60) clearly beats the un-tuned intuition of 0.80;
  correlation collapses above ~0.8 as fallback rate spikes.
- KNN k=30 (design doc default) underperforms its own ceiling; correlation rises
  monotonically through k=100. Not adopted since cosine won regardless.
- KNN's confidence score is structurally uninformative (constant for a given k)
  — a real design weakness, not a tuning issue.
- Confidence did not clearly predict per-game accuracy in Phase 4 (MAE
  high-confidence vs low-confidence were comparable, not the clean gap the
  design doc hypothesized) — worth another look with a larger low-confidence
  sample before trusting confidence as a live-prediction gating signal.

**Player-name resolution coverage rate and trustworthiness:** 90.95%
(935/1,028 distinct names), above the 80% minimum, so calibration deltas were
NOT marked low-confidence. The built-in cross-check (Phase 0 → Phase 3/4:
"if a bad name-join corrupts Layer 2, L1+2 should show up correlating worse than
L1 alone") could not run as literally specified, since both L1-only and L1+2
used the naive (no-search) diff-sum baseline and both were similarly negative
(-0.143 vs -0.140) — i.e. L1+2 was very slightly *better* than L1 alone even in
the naive framing, and dramatically better once Layer 3 is added (0.281). This
is consistent with a trustworthy name resolution / archetype join, not a
corrupted one.

**Recommended next step:** **Iterate, do not integrate yet, and do not
abandon.** The core hypothesis validated well (A7 alone beats A2 alone by a
wide margin on raw correlation), which is a strong enough result to justify
further investment, but three open items should be resolved with human input
before any `feature_builder.py` integration is considered:
1. The perimeter_specialist sign flip (Phase 0) needs investigation — is it real
   or a confound/small-sample artifact?
2. Confidence-vs-accuracy calibration (Phase 4) needs a larger low-confidence
   sample to properly test the design doc's hypothesis.
3. The `combo` archetype and its calibrated deltas are new (not in the design
   doc) and haven't been reviewed by a human.

**Open questions for human review:**
- Is the perimeter_specialist injury-impact sign flip (team defense improves
  when a perimeter specialist is Out) real, or an artifact of small sample
  size (407 games) / lineup confounds?
- Is the `combo` archetype (added mid-task) a reasonable permanent addition to
  the taxonomy, or should it be reverted to the design doc's original 4?
- The evaluation window (2023-10-01 onward, 3,922 games) was chosen for
  compute/time reasons — should Phase 4 be re-run on a different date range
  (e.g. only the most recent single season, or a strict train/test split
  mirroring `configs/config.yaml`'s `validation_start_date`/`test_start_date`)
  before treating these correlations as final?
- Should Layer 4 (role-level matchup flags, live-prediction only) be scoped as
  a follow-up now that Layers 1-3 are validated?

