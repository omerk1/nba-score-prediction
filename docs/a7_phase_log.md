# A7 Style Matchup — Phase Log

Condensed record of what was built, tried, and found across six work rounds (all six
now complete). Full historical detail (per-fold tables, centroid dumps, halflife grids)
lived here in earlier drafts — trimmed for conciseness; the numbers that mattered are
kept below. All code lives in `src/matchups/` (not imported by `feature_builder.py` —
exploratory, not yet integrated).

---

## Status

**Style signal robustly beats the A2 H2H baseline.** Confirmed via a single static
split and a 5-fold walk-forward CV (not just one split): the tuned config wins on
every fold, with lower variance than the untuned default. Two real bugs were found
and fixed along the way (a z-score normalization leak, an injury-calibration
sign-flip artifact) — not just parameter tuning. Not yet integrated into
`feature_builder.py`; that's a separate future decision.

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

## Known open items (not blockers, intentionally deferred)

- **PCA `n_components` sweep, further supervised-model tuning** — both already lose to
  hand-picked/lookup by a clear, CV-confirmed margin; low expected value.
- **Richer/shot-chart style inputs** (`nba_api` shot-chart endpoints) — real scope
  expansion, backlogged (`docs/backlog.md`).
- **`config_loader.py` formalization** — done (`StyleMatchupConfig` added, 78 tests pass).
- **Actual `feature_builder.py` integration** — a distinct future decision, not part of
  this exploration.
