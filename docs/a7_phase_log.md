# A7 Style Matchup — Phase Log

Condensed record of what was built, tried, and found across nine work rounds (all
nine now complete). Full historical detail (per-fold tables, centroid dumps, halflife
grids) lived here in earlier drafts — trimmed for conciseness; the numbers that
mattered are kept below. Core method lives in `src/matchups/`; Round 7 wired the
KNN-lookup score into `src/feature_engineering/feature_builder.py` (gated off by
default); Round 8 added a second, independently-gated raw-components-plus-
differentials feature set to the same file (also gated off by default — see Round 8).

---

## Status

**Style signal robustly beats the A2 H2H baseline in isolation** (correlation with
actual margin). Confirmed via a single static split and a 5-fold walk-forward CV (not
just one split): the tuned config wins on every fold, with lower variance than the
untuned default. Two real bugs were found and fixed along the way (a z-score
normalization leak, an injury-calibration sign-flip artifact) — not just parameter
tuning.

**Round 7 (feature-integration test) found this does not translate into a measurable
real-model accuracy improvement** once wired into `feature_builder.py`/`train_model.py`
alongside the existing feature set (rolling efficiency, Elo, H2H, injury deficit) —
see Round 7 below. `style_matchup.enabled` stays `false` by default; not adopted.

**Round 8 (raw components + explicit differentials, a redesign not a retune)**
tried the structural fix Round 7's finding implied — expose the fingerprint's raw
ingredients (plus a new `offensive_rating` quality metric) instead of one
pre-aggregated KNN-average number. Result: real feature importance this time
(top-2 overall features), but the signal sharpens `total_mae`, not `win_acc`/spread
accuracy — mixed, not adopted by default either. See Round 8 below.
`style_matchup.raw_features_enabled` also stays `false` by default.

**Round 9a (expanding-window model CV, checking Round 8's robustness)** ran the
same real-model comparison across `walkforward.py`'s 5 chronological folds instead
of one static split. `total_mae` (and `diff_mae`/`brier`) improve consistently
across all 5 folds — Round 8's headline finding holds, more robustly than one
split could show. But `win_acc` — the metric that drove Round 8's "do not adopt"
call — reverses sign on 2 of 5 folds and is roughly net-neutral in aggregate, so
Round 8's "win_acc gets worse" characterization does not hold up as a stable
property. `pace_score`'s #1/#2 importance (ahead of `elo_diff`) is confirmed
consistent across every fold. Still not adopted by default (validation run, not
an adoption decision) — see Round 9a below.

**Current recommended config** (`configs/config.yaml`'s `style_matchup` block):
`fingerprint_window=37, decay_halflife=13.2, encoding=hand_picked, similarity_method=knn,
knn_k=81, min_confidence_sample=21, full_confidence_sample=82, layer=2 (injury-adjusted),
injury_calibration_decay_halflife_games=20`.

**Round 6 closed the last open item.** The hyperparameter search that found this config
ran before the z-score fix; Round 4 confirmed the fix didn't change which config wins
(walk-forward CV) but never reran the search itself under corrected normalization.
Round 6 reran it (identical search space/trials/seed) — the corrected-normalization
search's own best config does not beat the standing recommendation above (re-evaluated
under the same corrected normalization: 0.220 train / 0.321 validation vs. the new
search's 0.208 train / 0.319 validation). No config change.

---

## Round 1 — Foundation (Phases 0–5)

Built from scratch: box-score cache (`box_scores.py`, via `nba_api.LeagueGameLog`,
parity-checked 1:1 against the `game` table), player-name→id resolution (`players.py`,
90.95% coverage), percentile-based archetype classification, empirical injury-impact
calibration (`calibration.py`), rolling style fingerprints (`fingerprint.py`), injury
adjustment (Layer 2), and cosine/KNN similarity search (`similarity.py`).

- Leakage verified at every boundary: fingerprint rolling window (`.shift(1)` before
  `.rolling()`), similarity search (`np.searchsorted(..., side="left")`, excludes
  same-date games too), H2H baseline (same shift-then-expand pattern as
  `FeatureBuilder._add_h2h_features`).
- **Result: style score correlates 0.281 with actual home margin vs. A2's 0.118** —
  more than double. Cosine@0.70 beat untuned KNN k=30 (0.281 vs 0.248, later reversed
  in Round 2 once KNN was properly tuned). Hand-picked encoding cleared the design
  doc's 0.2 bar, so PCA was skipped per the doc's own conditional rule (tested anyway
  in Round 2).
- The naive "Layer 1 only / Layer 1+2" (no similarity search) ablation correlates
  *negatively* (-0.14) — not a bug, confirms the design doc's own warning against
  comparing raw style vectors directly. Layer 3 (the search) is what turns the
  fingerprint into signal.
- Archetype taxonomy widened beyond the doc's original 4: added `combo`
  (facilitator+scorer overlap), tried KMeans clustering (k=4-8) as an alternative to
  percentiles — clusters separated by playing-time tier, not style (only
  PPG/AST/REB/BLK/STL/FG% available, no minutes/usage data at the time). Kept
  percentiles.
- `perimeter_specialist`'s calibrated injury delta had the wrong sign vs. the design
  doc's v1 guess — flagged for investigation, resolved in Round 5.

---

## Round 2 — Wider exploration

Real hyperparameter search (`tuning.py`, Optuna, 40 trials) instead of hand-picked
grids; PCA and clustering tried unconditionally (not just as escape hatches); a
CatBoost supervised-model paradigm tried as a genuine alternative to lookup-and-average.

- **Best config found: window=37, halflife=13.2, similarity_method=knn, knn_k=81** —
  validation corr 0.323 vs. 0.285 untuned default. This *reversed* Round 1's
  cosine-beats-KNN finding — KNN wasn't worse, it was under-tuned at k=30.
- PCA (0.262) and clustering (0.192, weakest method) both lost to hand-picked cosine.
- Supervised CatBoost tied the untuned default (0.284) but lost to the tuned lookup
  config, and was the only method showing train>validation overfitting.
- Correlation was systematically lower on train than validation for every method —
  flagged as a "corpus-depth" hypothesis (thin early history), not yet verified.

---

## Round 3 — Walk-forward CV

Built a proper multi-fold CV (`walkforward.py`, 4 folds validating on 2021-22 through
2024-25) to check whether Round 2's winning config was robust or a one-split fluke.
Also tested a KNN-with-similarity-floor hybrid and a recency-cutoff sweep.

- **Confirmed robust: the tuned config beats the untuned default on every single
  fold** (mean 0.2547 vs 0.1962), with *lower* variance (0.052 vs 0.073) — not just a
  better average.
- Corpus-depth hypothesis confirmed: correlation rises fold 1→4 for every method.
- KNN-with-floor hybrid ties plain KNN exactly at k=81 — the floor never binds at
  that k (only hurts if set very aggressively, ≥0.6-0.7).
- Recency cutoff: 2-5 years is statistically indistinguishable from unbounded; only a
  1-year cutoff clearly hurts. **Recommendation: no recency cutoff needed.**

---

## Round 4 — Wrap-up (5 items, critique mode)

### Z-score normalization leak (fixed)
Found the z-score mean/std used to build matchup vectors was computed globally across
*all* cached history (2016–2026), not point-in-time — a genuine leak (later folds'
normalization constants included data from years after earlier folds' games). Fixed by
threading a per-fold point-in-time cutoff through `matchup_index.py`/`tuning.py`/
`walkforward.py`, mirroring how PCA already fit on train-split-only data.
**Headline conclusion survived unchanged — the fix, if anything, slightly improved
correlation** (largest gain in fold 1: +0.055 for the default config), since
era-relative normalization turned out to be a more faithful similarity metric, not
just a leakage-hygiene fix.

### Minutes/usage data added to archetype classification
Joined `player_importance` (in `injury_features.sqlite`, minutes/usage already
populated, previously unused) into archetype classification. Tried two variants:
concatenating raw minutes/usage onto the existing 6 stats (didn't fix the
playing-time-tier problem), and per-minute-normalized rates instead of raw counts
(partial fix — BLK's correlation with minutes dropped from ~0.8 to 0.24, a genuine
rim-protection axis separated from playing time). AST/STL/PPG/usage_rate stayed tied
to minutes even after normalization — concluded to be a real "better players get more
minutes" selection effect, not a fixable artifact. **Kept percentile method** (doesn't
have this failure mode). Redefined `combo` using real usage_rate/assist-rate
percentiles instead of an artificially high PPG/AST threshold (verified: preserves
~79% of the old population). Triggered a full recalibration, now in `config.yaml`.

### Perimeter_specialist sign-flip — root cause found
`perimeter_specialist`'s calibrated `defensive_rating` delta had the wrong sign
(-0.089, should be positive). Traced to one player: Collin Sexton's ~5-month
continuous ACL-recovery absence alone accounts for 127 of 680 qualifying event-rows.
Excluding just his rows flips the delta to +0.957. Verdict: small-sample/
misclassification artifact (an offense-first guard misclassified by a stat-based
archetype definition), not a real basketball effect. Fix implemented in Round 5.

### Injury adjustment's real contribution — isolated
Earlier rounds only compared Layer 1 vs. Layer 2 *without* the similarity search
active (both near -0.14, uninformative). Built the missing comparison: full pipeline,
layer=1 vs layer=2, search method held constant. **Layer 2 beats Layer 1 on every
single fold for both configs** (small, consistent, +0.005 to +0.007 mean) — resolves
a question left open since Round 1.

### Walk-forward extended to 5 folds
Added a 5th fold using already-present 2025-26 data. Tuned config's advantage held
(highest correlation of any fold); variance margin over the default narrowed
slightly at n=5, as expected with one more data point.

---

## Round 5 — Decay-weighted injury calibration (fixes Round 4's flagged issue)

Calibration previously treated every qualifying `Out` team-game identically regardless
of how long the absence had run. Fixed by weighting each event by
`0.5 ** (streak_position / halflife)` — reusing the exact decay math already used for
fingerprint rolling windows (`fingerprint.py`'s `_decay_weight`), not new math.

**Halflife grid (all archetypes checked, not just the one that motivated this):**

| halflife | perimeter_specialist def_rating | notes |
|---|---|---|
| no decay | -0.088 | original (wrong sign) |
| 40 | +0.123 | sign flips, but razor-thin |
| **20 (chosen)** | **+0.295** | unambiguous, ≥76% sample retained for every other archetype |
| 10 | +0.512 | stronger, but costs `scorer` down to 67.5% retention |
| 5 | +0.686 | most aggressive; converges toward the +0.957 leave-one-out value |

`combo`/`facilitator` deltas shift materially with halflife (up to +157% at halflife=5);
`rim_protector` (no diagnosed problem) stays essentially flat across the whole grid —
confirms this is a general, correctly-targeted mechanism, not a one-archetype patch.

Config updated (`injury_calibration_decay_halflife_games: 20`, new `injury_impact`
deltas), Layer 2 cache rebuilt. Walk-forward CV sanity check (layer2 vs layer1, tuned
config): mean advantage unchanged post-fix (+0.0067 vs. pre-fix +0.0065) — no
regression.

---

## Round 6 — Hyperparameter search recheck

Reran the Round 2 Optuna search (`tuning.run_optuna_search`, identical search space,
40 trials, TPE, seed=42) under Round 4's z-score fix — the one residual gap Round 4
explicitly flagged and left open (the fix was verified against the walk-forward harness,
never against the search that originally produced this config). New
`zscore_point_in_time` flag on `run_optuna_search` reuses the existing
`zscore_cutoff_date` mechanism, fit once at `train_end_date` (2024-04-14) and applied to
both the train (selection) and validation (report) windows — same principle as the
per-fold fix, just for one static cutoff instead of five.

**All four numbers below use identical corrected normalization — a genuine
apples-to-apples table:**

| config | train corr | validation corr |
|---|---|---|
| untuned default (window=20/halflife=5/cosine@0.70) | 0.176 | 0.288 |
| new search's best (window=40/halflife=7.56/knn k=103/min_conf=26/full_conf=84) | 0.208 | 0.319 |
| **standing config** (window=37/halflife=13.2/knn k=81/min_conf=21/full_conf=82) | **0.220** | **0.321** |

- The standing config's own numbers barely moved from its originally-reported
  (leaky-normalization) 0.218/0.323 — confirms Round 4's general finding that this fix
  doesn't differentially affect already-tuned configs.
- **The fresh 40-trial search did not find anything better — its own best config scores
  *below* the standing config on both splits**, under the identical metric it was
  selecting on. The new config is a minor variation in the same neighborhood (large
  window, KNN, large k), not a different regime; the shortfall is small (~0.01-0.02, well
  within the ~0.05-0.07 fold-to-fold noise seen elsewhere in this project) and most
  plausibly reflects ordinary TPE search variance (observed objective values shift under
  corrected normalization, so even a fixed seed's trial trajectory diverges from the
  original run).
- **Conclusion: the original config selection holds up.** This is the expected,
  hoped-for outcome — no config change, no further validation spend (per instructed
  scope, a "reconfirms" result doesn't trigger a walk-forward CV re-check).

---

## Round 7 — Feature integration test

The acid test: does A7 improve the actual trained model's real prediction accuracy,
not just correlation-with-margin in isolation? Reversed two standing rules
(`feature_builder.py` and `train_model.py` execution were previously off-limits) —
both now in scope, per explicit instruction.

**Built:**
- `src/matchups/precompute_scores.py` — runs the validated pipeline (unchanged
  hyperparameters) once over the full `game` table (~12.7k games) and caches
  `style_matchup_score`/`confidence`/`fallback_used`/`n_similar` keyed by `game_id`
  into `outputs/a7_matchups_cache.sqlite`'s new `style_matchup_scores` table.
  Runs in ~50s (box scores, fingerprints, injury adjustment, KNN search all
  included) — not the bottleneck. Also builds `player_name_resolution`/
  `player_archetypes` if missing (a fresh cache DB, e.g. a new worktree, silently
  no-ops Layer 2 otherwise — found and fixed: 0% team-games adjusted before the
  fix, 24.75% after, matching the design doc's expectation).
- Found and fixed en route: `configs/config.yaml`'s `style_matchup` block still had
  the untuned Phase-0 defaults (window=20/halflife=5/cosine/k=30) — despite the
  design doc and Round 2/6 both documenting window=37/halflife=13.2/knn/k=81 as the
  validated winner, that config was never actually written into the yaml (only
  hardcoded separately inside `tuning.py`/`walkforward.py`). Corrected as part of
  this round (not a re-tune — using the already-identified winner).
- `similarity.run_similarity_search` gained an optional `zscore_cutoff_date` param
  (default `None`, backward compatible) so the precompute pass fits matchup-vector
  z-score stats only on pre-`train_end_date` (2024-04-14) data — avoids a look-ahead
  leak into val/test scores, reusing the exact mechanism validated in Round 6.
- `FeatureBuilder._add_style_matchup_features` — left-joins the cache onto the
  training dataframe by `GAME_ID`, gated by new `style_matchup.enabled` config flag
  (mirrors the `elo_features`/`injury_features` gating pattern exactly). Left
  `enabled: false` as the committed default.
- `train_model.py` gained a `--experiments-csv` override (default unchanged) so
  this round's two runs could log to `outputs/a7_integration_test_results.csv`
  instead of the shared `outputs/experiments.csv` (coordinator correction
  mid-run — the shared log should only get entries the coordinator has reviewed).

**Experiment:** identical settings otherwise, only `style_matchup.enabled` toggled.

| metric | baseline (107 feat) | +style_matchup (109 feat) | delta |
|---|---|---|---|
| val diff_mae | 11.162 | 11.133 | −0.029 |
| test diff_mae | 11.572 | 11.558 | −0.014 |
| val diff_within_5 | 0.2890 | 0.2906 | +0.0016 |
| test diff_within_5 | 0.2939 | 0.2939 | 0.0000 |
| val total_mae | 15.070 | 15.066 | −0.004 |
| test total_mae | 15.654 | 15.631 | −0.023 |
| val win_acc | 0.6531 | 0.6571 | +0.0040 |
| test win_acc | 0.6735 | 0.6792 | +0.0057 |
| val brier | 0.2129 | 0.2142 | +0.0013 (worse) |
| test brier | 0.2108 | 0.2095 | −0.0013 (better) |

Full rows: `outputs/a7_integration_test_results.csv` (run names
`a7_integration_baseline` / `a7_integration_with_style_matchup`) — NOT
`outputs/experiments.csv`, per coordinator correction (see above).

`style_matchup_score` ranked 29th of 109 features by CatBoost importance (~1% of
total importance, well below every rolling/Elo/venue feature); `style_matchup_confidence`
had **zero** importance — the model never split on it.

**Does it help? No measurable improvement.** Every delta above is within
third-decimal noise (smaller than the run-to-run variation already visible between
e.g. `elo_v1`→`elo_v2` in the real `experiments.csv`), and several metrics move in
opposite directions (val brier worse, test brier better) — not a consistent signal
in either direction. This is smaller than, not merely "less than," the 0.28–0.32
standalone correlation would suggest: the isolated correlation measures how well
the style score alone tracks margin; it says nothing about how much *new*
information it adds once CatBoost already has rolling off/def efficiency (multiple
windows), Elo, H2H, venue, and injury-deficit features. The low feature-importance
rank plus near-zero accuracy delta together support one conclusion: **CatBoost's
existing feature set already captures most of the signal A7 would add** — the
design doc's own note that "combining A7+A2 barely beats A7 alone" (A7 subsumes A2)
generalizes one step further here: the full existing feature set subsumes A7 too,
in practice, even though A7 beats A2 alone in isolation.

**Recommendation: does not help (as integrated) — do not adopt at this time.**
Not a failure of the exploration (six rounds of rigorous validation stand), but a
real, informative negative result for *this* integration path. Two paths forward if
revisited: (a) richer style inputs (shot-chart data, already backlogged) might carry
information the existing feature set genuinely lacks, where a correlation-with-margin
edge would actually convert to accuracy; (b) `style_matchup_confidence`'s zero
importance suggests it's dead weight even if `style_matchup_score` were kept — drop
it rather than carry an unused column. Config default correctly left `false`;
`predict_game.py`/live-prediction wiring was out of scope and untouched, consistent
with this result (no reason to build a live-refresh pipeline for a feature that
doesn't move accuracy).

---

## Round 8 — Raw fingerprint components + explicit differentials (redesign, not a retune)

Round 7's `style_matchup_score` is a pre-aggregated KNN-average "mini-prediction" —
CatBoost never sees the ingredients, just one opaque number (29th of 109 features,
confidence 0% importance). This round tests a structurally different alternative:
expose the fingerprint's raw per-team components plus explicit home-vs-away
differentials directly, mirroring `_add_matchup_features`'s existing pattern
(`home_off_vs_away_def_L{window}` etc.) instead of another black-box lookup. No KNN
involved at all. Additive, independently gated — old method/pipeline untouched.

**Also fixed a real gap, not just a re-encoding:** 4 of the 5 original metrics
(`pace_score`, `three_pt_reliance`, `paint_activity`, `assist_rate`) are volume/style
metrics (*what* a team does), not quality (*how well*) — `defensive_rating` was the
one exception. Added `offensive_rating` (`PTS/possessions*100`, same possessions
estimate as `defensive_rating`) as a 6th `FINGERPRINT_METRICS` entry in
`fingerprint.py`. **Layer 1 only** — no injury-adjustment delta calibrated for it
(deliberate scope cut; a full Phase-0-style calibration is out of scope for this
redesign round, candidate for later if this approach shows promise). Rebuilt
layer=1/layer=2 fingerprint caches (25,436 team-games; 24.75% injury-adjusted,
matching Round 7's coverage exactly); layer=2 passes `offensive_rating` through
unmodified (verified 100% match vs layer=1). `db.py`'s `matchup_fingerprints` schema
gained the column plus an `ALTER TABLE` migration for pre-existing cache DBs.

**Built:** `FeatureBuilder._add_style_fingerprint_features` — reads
`matchup_fingerprints` directly (no similarity search), layer=2 for the 5 calibrated
metrics + layer=1 for `offensive_rating`, joined by `(game_id, team_id)` for both
teams. Adds 18 columns: `home_style_{metric}` / `away_style_{metric}` (12 raw) +
`style_{metric}_diff` (6, home−away). Gated by new `style_matchup.raw_features_enabled`
(default `false`), independent of `style_matchup.enabled` — both can be toggled
separately. `config_loader.py`'s `StyleMatchupConfig` gained the field (default
`False` so any caller not yet passing it still works); `tests/test_h2h.py`'s mock
config updated (same MagicMock-truthiness fix pattern as Round 7).

**Blocker:** fresh worktree had no `data/raw/*.sqlite` symlinks and an empty
`outputs/a7_matchups_cache.sqlite` (git-ignored data files, not carried over) — and
the `work/a7-fingerprint-features` branch was already checked out in another,
unmodified worktree, blocking checkout in this one. Fallback: symlinked
`nba_api.sqlite`/`injury_features.sqlite`/`basketball.sqlite` from the human's
working copy (read-only, matches `db.py`'s existing symlink convention), removed
the stale unused worktree registration (clean, no lost work, confirmed via
`git status`/`git diff` before removing), and rebuilt every cache table from
scratch. Rebuilt coverage numbers (name resolution 90.95%, archetypes, 24.75%
injury-adjusted) matched Round 7's documented values exactly, confirming the
rebuild reproduced the same environment.

**Experiment:** reran all three configs end-to-end (not reused from Round 7's CSV —
having rebuilt the environment from empty, wanted freshly-generated, directly
comparable full feature-importance tables for all three, which Round 7's artifacts
never saved). `train_model.py` now also dumps the FULL per-feature importance table
(not just top-20) to `outputs/a7_feature_importance_<run_name>.csv` — Round 7 only
kept top-20 (print + gitignored `outputs/reports/`), losing the full ranking once the
worktree was cleaned up.

| metric | baseline (107 feat) | old KNN score (109 feat) | new raw+diff (125 feat) |
|---|---|---|---|
| val diff_mae | 11.162 | 11.133 | **11.117** |
| test diff_mae | 11.572 | **11.558** | 11.603 |
| val diff_within_5 | 0.2890 | 0.2906 | **0.2939** |
| test diff_within_5 | **0.2939** | **0.2939** | 0.2857 |
| val total_mae | 15.070 | 15.066 | **14.905** |
| test total_mae | 15.654 | 15.631 | **15.410** |
| val win_acc | 0.6531 | **0.6571** | 0.6514 |
| test win_acc | 0.6735 | **0.6792** | 0.6637 |
| val brier | **0.2129** | 0.2142 | **0.2129** |
| test brier | **0.2108** | 0.2095 | 0.2113 |

Baseline and old-KNN rows match Round 7's `a7_integration_test_results.csv` exactly
(e.g. baseline val/test diff_mae 11.162/11.572 both rounds) — confirms unchanged
environment/data despite the from-scratch rebuild. `style_matchup_score` still
ranks 29th (~1% importance), confidence still 0% — Round 7's finding reconfirmed,
unaffected by this round's additive changes.

**Does the redesign get non-trivial importance? Yes, clearly** — unlike the old
score's ~1%/0%. In the new run, `home_style_pace_score`/`away_style_pace_score` rank
**#1 and #2 overall** (11.6 and 9.7, ahead of `elo_diff` at #3), ~17% of total
importance between them. `style_offensive_rating_diff` ranks #14 (1.67) and
`away_style_offensive_rating` #13 (1.70) — both clear top-15, **supporting the
direction/magnitude hypothesis specifically**: the new quality metric is informative
where several old volume-only metrics aren't (`home_style_three_pt_reliance` and
`away_style_defensive_rating` get exactly zero importance; `style_assist_rate_diff`
too). Of the 18 new columns, 3 get zero importance, most others land rank 30-104.

**Does it translate into accuracy? Mixed, not a clean win.** `pace_score`'s dominance
is a plausible artifact of what it measures — `PTS + OPP_PTS + TOV - FTA*0.44` is
essentially a rolling proxy for *combined scoring level*, which the model
(`MultiRMSE` over `PTS_home`/`PTS_away` jointly) can exploit directly for absolute
score magnitude — consistent with `total_mae` improving clearly on both splits
(val 14.905 vs 15.070 baseline; test 15.410 vs 15.654) — the biggest total_mae gain
of any A7 variant tried. But that is a different thing from sharpening the
home-minus-away *margin*: `diff_mae`/`win_acc` do not follow — val `diff_mae` is
marginally best of the three (11.117), test `diff_mae` is marginally worst (11.603);
`win_acc` is worse on both splits than baseline (65.14% vs 65.31% val, 66.37% vs
67.35% test) and worse than the old KNN approach on both. Brier is a wash (val ties
baseline, test slightly worse than both). Net: high feature importance this time —
the redesign's core hypothesis (expose ingredients, let CatBoost learn interactions)
is confirmed — but the *specific* new information the model latches onto
(pace/scoring-level) sharpens total-points prediction, not the moneyline/spread
markets the project cares about most.

**Recommendation: do not adopt by default at this time**, but for a different reason
than Round 7 — not "no signal" (there clearly is, and it's structurally the fix Round
7's finding called for) but "the signal that emerged sharpens the wrong market."
Two concrete follow-ups if revisited: (a) since `total_mae` improved substantially,
consider whether a total-points-focused evaluation (or an over/under product
surface) would find this variant worth adopting even though win_acc doesn't improve —
out of scope to judge here, flagging for the coordinator; (b) if pursuing the
moneyline/spread angle further, the zero-importance columns (`home_style_
three_pt_reliance`, `away_style_defensive_rating`, `style_assist_rate_diff`) are
candidates to drop, and the asymmetry between `home_style_offensive_rating` (rank
102, ~0) and `away_style_offensive_rating` (rank 13) is worth a closer look before
concluding the metric itself is the win — it may be specifically the *away* team's
offensive quality relative to home defensive/rolling features already in the model
that's informative, not offensive_rating symmetrically. Config defaults correctly
left `false` for both `style_matchup.enabled` and `raw_features_enabled`;
`predict_game.py` untouched, per instructions.

Full rows: `outputs/a7_fingerprint_features_results.csv` (run names `a7r8_baseline` /
`a7r8_old_knn_style_matchup` / `a7r8_raw_fingerprint_features`) — NOT
`outputs/experiments.csv`. Full per-feature importance (all features, every run):
`outputs/a7_feature_importance_<run_name>.csv`.

---

## Round 9a — Expanding-window CV for the raw fingerprint features

Round 8's mixed result (`total_mae` improves, `win_acc` worsens) was measured on a
SINGLE static train/validation/test split — the same limitation Round 3/6 already
addressed for A7's standalone similarity-search pipeline via `walkforward.py`'s
5-fold walk-forward CV. This round applies the same rigor to the full trained
model: reuses `walkforward.py`'s exact `FOLDS_WITH_FOLD5` boundaries unmodified
(train-through-cutoff / validate-on-next-season), but instead of evaluating A7's
standalone lookup per fold, builds real features via `FeatureBuilder.
create_all_features` and trains the actual CatBoost model per fold — reusing
`train_model.py`'s data-loading/feature-building/training/metric flow and
hyperparameters as-is (`src/matchups/experiments/round9_modelcv.py`; kept in this
directory despite training the full model, since its entire purpose is checking
an A7 finding's robustness and it reuses `walkforward.py`'s fold scheme directly).
No re-tuning of anything. `style_matchup.enabled`/`raw_features_enabled` toggled
via an in-memory `model_copy(update=...)` monkeypatch of `feature_builder.
load_config` — reversible per-process, the committed `config.yaml` is never
touched. No third split per fold (`walkforward.py`'s own scheme is train/validate
only), so — per instructions — only each fold's validation season is reported,
not a val+test pair like Round 8's single-split table.

**Per-fold results (validation season only; baseline=107 feat, raw_features=125 feat):**

| fold | season | config | diff_mae | diff_within_5 | total_mae | win_acc | brier |
|---|---|---|---|---|---|---|---|
| 1 | 2021-22 | baseline | 11.315 | 0.2984 | 15.183 | 0.6472 | 0.2258 |
| 1 | 2021-22 | raw_features | 11.316 | 0.2886 | **15.035** | 0.6366 | **0.2251** |
| 2 | 2022-23 | baseline | 10.249 | 0.2951 | 15.200 | 0.6325 | 0.2278 |
| 2 | 2022-23 | raw_features | **10.188** | **0.3065** | **15.012** | 0.6317 | **0.2270** |
| 3 | 2023-24 | baseline | 11.039 | **0.3041** | 15.363 | 0.6480 | 0.2129 |
| 3 | 2023-24 | raw_features | **11.037** | 0.3000 | **15.107** | **0.6593** | **0.2128** |
| 4 | 2024-25 | baseline | 11.198 | 0.2857 | 14.962 | 0.6490 | 0.2139 |
| 4 | 2024-25 | raw_features | **11.143** | **0.2890** | **14.795** | **0.6604** | **0.2126** |
| 5 | 2025-26 | baseline | 11.516 | **0.2931** | 15.762 | **0.6759** | 0.2088 |
| 5 | 2025-26 | raw_features | **11.426** | 0.2963 | **15.379** | 0.6743 | **0.2070** |

**Aggregate (mean ± std across the 5 folds):**

| config | diff_mae | diff_within_5 | total_mae | win_acc | brier |
|---|---|---|---|---|---|
| baseline | 11.063 ± 0.487 | 0.2953 ± 0.0068 | 15.294 ± 0.298 | 0.6505 ± 0.0157 | 0.2178 ± 0.0084 |
| raw_features | 11.022 ± 0.490 | 0.2961 ± 0.0076 | **15.066 ± 0.210** | 0.6525 ± 0.0178 | 0.2169 ± 0.0087 |

Full rows: `outputs/a7_round9_modelcv_results.csv` — NOT `outputs/experiments.csv`.
Full per-feature importance, every fold × config:
`outputs/a7_round9_feature_importance_fold{1-5}_{baseline,raw_features}.csv`.

**Does Round 8's pattern hold consistently across folds? Partially — total_mae
robustly holds, win_acc does not.**

- `total_mae`: raw_features improves on **all 5 of 5 folds**, unanimous, no
  exceptions — the single cleanest, most robust result in this whole round. It
  also comes with *lower* variance (±0.210 vs baseline's ±0.298), not just a
  better mean. This is the strongest possible confirmation Round 8's headline
  `total_mae` finding wasn't a one-split artifact.
- `diff_mae`: raw_features ties-or-improves on all 5 folds (fold 1 is a
  statistical wash, +0.001; folds 2-5 all improve, up to −0.090 on fold 5) —
  more consistently positive than Round 8's single-split "mixed" framing
  (val better/test worse) suggested.
- `brier`: raw_features improves-or-ties on all 5 folds too — another metric
  that looked like a wash on the single split but is quietly, consistently
  favorable across the walk-forward view.
- `diff_within_5`: genuinely mixed, 3 of 5 folds better (2, 4, 5), 2 of 5 worse
  (1, 3) — matches Round 8's "mixed" characterization for this specific metric.
- **`win_acc`: does NOT hold consistently — it reverses sign on 2 of 5 folds.**
  Worse on folds 1, 2, 5 (2021-22, 2022-23, 2025-26: −0.0106, −0.0008, −0.0016)
  but *better* on folds 3, 4 (2023-24, 2024-25: +0.0113, +0.0114). Folds 3-4's
  gains are larger in magnitude than folds 1/2/5's losses, so the 5-fold mean
  win_acc is actually marginally **higher** for raw_features than baseline
  (0.6525 vs 0.6505) — the opposite direction from Round 8's single-split
  conclusion that win_acc gets worse. Round 8's win_acc-worse finding held on
  the one split tested (which is closest to fold 4/5 here) but is not a stable
  property of this feature set across folds — it is fold-dependent, close to a
  coin flip in sign, and net-neutral-to-slightly-positive on average.

**Caveat on reproducing Round 8's exact split:** fold 4 here (train-through-
2024-10-01, validate 2024-10-22 to 2025-04-13) is nominally the same window as
Round 8's static-split validation set, but the numbers don't match Round 8's
reported val row (this round: baseline win_acc 0.6490/raw 0.6604, raw *better*;
Round 8: baseline 0.6531/raw 0.6514, raw *worse* — for what should be the same
games). Root cause: this worktree's `matchup_fingerprints`/injury-calibration
caches were rebuilt from scratch this round (see Blocker below) against the
*current* `nba_api.sqlite`/`injury_features.sqlite` — more games and injury
reports have been added since Round 8 ran, and the empirical, decay-weighted
injury calibration (Round 5) is recomputed from the *full* available history
every time it's built, so historical Layer-2 fingerprint values can legitimately
shift when the underlying data grows, even for old games. Not a bug in this
round's harness (row counts/coverage match Round 8's documented values, and this
is the same recalibrate-from-full-history behavior Round 5 already documented as
intentional) — but it means "same nominal split" isn't perfectly reproducible
across time for this pipeline, itself a data point on how much noise to expect
from this feature set's accuracy effect on any one split.

**Is pace_score's dominant importance consistent across folds? Yes, completely.**
`home_style_pace_score`/`away_style_pace_score` rank **#1 and #2 overall in every
single one of the 5 folds** (order between the two swaps fold to fold, magnitude
ranges ~7-13 depending on fold, but never dislodged from the top 2), always ahead
of `elo_diff` at #3. This is not a one-split artifact — it is the single most
robust finding of this round. (A parallel feature-EDA task, run against this same
codebase but tracked on a separate `feature/feature-eda` branch since it covers
the whole model's features, not just A7, independently explains *why*:
`pace_score` correlates ~0 with `elo_diff`/spread/moneyline labels but 0.37-0.38
with `TOTAL_POINTS` — genuinely
new information, but specifically for the total/over-under market, which lines up
exactly with this round's `total_mae`-robust/`win_acc`-fold-dependent split.)

**Blocker:** same as Round 8 — fresh worktree, no `data/raw/*.sqlite` symlinks,
and the copied-in `outputs/a7_matchups_cache.sqlite` (from the human's working
copy) predated the Round 8 `offensive_rating` column/migration (present in code,
but the cached data still only had 5 metrics). Fallback: symlinked the 3 raw DBs
same as Round 8's convention, copied (not symlinked, so as to not write into the
human's live copy) the cache DB, ran `init_cache_db()` (triggers the `ALTER TABLE`
migration) followed by `build_fingerprint_cache(layer=1)` +
`build_injury_adjusted_fingerprints()` to populate real `offensive_rating` values
and refresh Layer 2 — reused the already-built `player_name_resolution`/
`player_archetypes`/`injury_calibration` tables rather than rebuilding those too
(independent of fingerprint_window/decay_halflife, per `precompute_scores.py`'s
own skip-if-populated logic). Confirmed 25,436 team-games both layers, 0 NULL
`offensive_rating` post-rebuild, 24.25% injury-adjusted (vs Round 8's 24.75% —
small drift, consistent with the caveat above: more current data than Round 8 had).

**Recommendation: still do not flip either flag by default (this is a validation
run, not an adoption decision) — but the evidence is more favorable to the raw
fingerprint approach than Round 8's single split suggested.** `total_mae`,
`diff_mae`, and `brier` all now look like consistent, low-risk wins across 5
independent folds (not just one), and `win_acc` — the metric that drove Round 8's
"do not adopt" call — turns out not to be a stable cost at all: it's fold-
dependent and roughly net-neutral in aggregate. If this feature set is revisited,
it's now on stronger footing than Round 8 alone implied; a natural next step
(flagged, not executed here, per this round's no-retuning scope) would be a
total-points-focused product evaluation given how unanimous and variance-
reducing the `total_mae` gain is — echoing the open item Round 8 already flagged
in "Known open items" below, now with 5-fold confirmation instead of one split.

---

## Known open items (not blockers, intentionally deferred)

- **PCA `n_components` sweep, further supervised-model tuning** — both already lose to
  hand-picked/lookup by a clear, CV-confirmed margin; low expected value.
- **Richer/shot-chart style inputs** (`nba_api` shot-chart endpoints) — real scope
  expansion, backlogged (`docs/backlog.md`); Round 7 found the current hand-picked
  metric set doesn't add real signal on top of the existing model, so this is now
  the more load-bearing open item, not just a nice-to-have.
- **`config_loader.py` formalization** — done (`StyleMatchupConfig` added, 78 tests pass;
  `enabled` field added Round 7, `raw_features_enabled` added Round 8).
- **`feature_builder.py` integration** — done (Round 7, KNN-lookup score: no measurable
  accuracy improvement). Round 8 tried a structurally different raw+differential
  redesign: gets real feature importance (unlike Round 7's ~1%/0%) but the signal it
  captures sharpens total-points accuracy, not win_acc/spread accuracy — mixed result,
  not adopted by default either. Round 9a confirmed the `total_mae`/importance
  finding across a 5-fold expanding-window model CV (consistent, unanimous), but
  found Round 8's `win_acc`-worse finding was itself fold-dependent (reverses on
  2/5 folds, net-neutral in aggregate) — not the stable cost Round 8's one split
  implied. Both flags still stay `false` (validation run, not an adoption call).
- **Offensive_rating injury-calibration** — deliberately deferred in Round 8 (Layer 1
  only); a candidate full Phase-0-style calibration pass if the raw-features approach
  is revisited.
- **Total-points-focused evaluation** — Round 8 flagged that its redesign's `total_mae`
  gain didn't show up in Round 7's original motivation (win_acc/spread), raising
  whether a total/over-under-focused product surface would value this differently.
  Round 9a's 5-fold CV strengthens the case for this being worth doing (unanimous,
  variance-reducing `total_mae` gain across every fold) — still not evaluated here,
  coordinator call.
