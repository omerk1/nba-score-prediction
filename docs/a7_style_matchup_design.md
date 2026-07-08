# A7: Style Matchup Score — Architecture & Design (as-built)

> **Status:** Validated across nine work stages (see `docs/a7_phase_log.md` for full
> evidence trail). The KNN-Score Integration Test integrated it into
> `feature_builder.py` behind a `style_matchup.enabled` config flag (default `false`)
> and ran a real train/val/test comparison — result: no measurable accuracy
> improvement over the existing feature set, so it is not enabled by default. A
> follow-up redesign (raw fingerprint components instead of the KNN score) found real
> feature importance but a mixed accuracy effect — see the phase log's KNN-Score
> Integration Test and Raw-Fingerprint Feature Redesign sections for the full
> comparison and recommendation.

---

## Problem Statement

**Current state:** A2 (Extended H2H) uses historical head-to-head records but ignores *how* teams play. Two teams with identical 3–0 H2H records could have completely different matchup dynamics if one team has shifted to perimeter-heavy while the other plays interior-focused.

**Solution:** A style-based matchup score capturing how roster compositions and playing styles interact, including injury-driven playstyle shifts (missing a facilitator forces more iso, missing a rim-protector weakens interior D, etc.).

**Result:** Validated — style score correlates 0.281–0.323 with actual home margin (depending on config) vs. A2 alone's 0.118, robustly confirmed via walk-forward CV, not just one split. See phase log for the full evidence trail.

---

## How Matchup Vectors Work

### Why not compare team styles directly?

The naive approach — comparing home style vector to away style vector — tells you "these teams play differently from each other." That's not useful. You want to know *who wins when this stylistic dynamic exists.* (Confirmed empirically: a naive per-dimension diff-sum correlates *negatively* with margin, -0.14 — see phase log's Initial Build section.)

### The right framing: a matchup is a single entity

Concatenate both teams' style fingerprints into one vector (home first, always):

```
matchup_vector = [home_pace, home_3pt, home_paint, home_def, home_ast,
                  away_pace, away_3pt, away_paint, away_def, away_ast]
```

Search history for games whose matchup vector looks similar to tonight's; the average point differential of those similar games becomes the prediction signal.

**Why order matters:** `[fast_home vs slow_away]` ≠ `[slow_home vs fast_away]`. Concatenation preserves this directionality; direct comparison loses it.

### Building the vector: rolling pre-game window

Built from each team's **last N games before the game date** — never the game itself (leakage). Implemented as `.shift(1)` before `.rolling()` (`src/matchups/fingerprint.py`), verified leakage-safe.

This vector is computed for every historical game upfront, forming a searchable index (`src/matchups/matchup_index.py`): `game_id | date | matchup_vector (10 values) | actual_home_margin`.

**Normalization:** z-score, fit point-in-time (only on data available up to the evaluation cutoff — a global-history fit was found to leak and was fixed; see phase log's Critique & Bug-Fix Pass section).

---

## Encoding: Hand-Picked Metrics (final choice)

```python
{
    "pace_score":        avg((PTS + OPP_PTS + TO - FTA×0.44) / games),
    "three_pt_reliance": avg(3PA / FGA),
    "paint_activity":    avg(FTA / game),
    "defensive_rating":  avg(OPP_PTS / possessions × 100),
    "assist_rate":       avg(AST / FGM),
}
```

**Decay:** exponential, half-life = `decay_halflife` games (tuned to 13.2, see below).

**Alternatives tried and rejected** (phase log's Hyperparameter Search & Alternative Methods section): PCA (5 components on 11 raw
metrics) and KMeans clustering (k=8 archetype-pair bucket lookup) both scored lower
than hand-picked on every split tested. Clustering was the weakest method overall —
discretizing into cluster-pair buckets throws away the fine-grained ranking cosine
similarity preserves. Neither is used; hand-picked wins outright with the metric set
available (no shot-chart data — see Future Work).

**Also tried and rejected:** a CatBoost supervised model regressing directly on the
matchup vector (no similarity search). Tied the untuned lookup default but lost to
the tuned lookup config, and was the only method to show overfitting (train >
validation). Lookup-and-average remains the better paradigm for this data.

---

## Architecture Layers

### Layer 1: Team Style Fingerprint
Rolling fingerprint per team per game, hand-picked encoding (above). `fingerprint_window=37`, `decay_halflife=13.2` (tuned; design defaults were 20/5).

### Layer 2: Injury-Adjusted Style
Applied before Layer 3, per the original design intent.

**Player Archetypes** (percentile-based, era-adaptive; final taxonomy — widened from the original 4):

| Archetype | Criteria |
|-----------|----------|
| Facilitator | AST ≥65th pct, PPG ≤35th pct |
| Scorer | PPG ≥65th pct, AST ≤35th pct |
| Combo | usage_rate ≥80th pct AND assist-rate ≥80th pct (dual scorer/facilitator; needed since Facilitator/Scorer are mutually exclusive by construction) |
| Rim Protector | BLK ≥75th pct, REB ≥75th pct |
| Perimeter Specialist | BLK ≤30th pct, STL ≥70th pct |

KMeans clustering was tried as an alternative (twice — once on raw box-score stats, once with minutes/usage data added). Both times it recovered playing-time tiers, not style, except for a genuine partial win on the interior-defense (BLK) axis once per-minute-normalized. Playmaking/scoring stats stay tied to minutes even after normalization — a real "better players get more court time" selection effect, not a fixable artifact. Percentile method kept as primary.

**Injury Impact — empirically calibrated, decay-weighted (final):**

For each archetype, compare team style metrics in games where that archetype was
`Out` vs. the team-season baseline — but each `Out` event is weighted by
`0.5 ** (streak_position / halflife)` where `streak_position` is how many consecutive
games into that continuous absence this occurrence is (`halflife=20` games). This
resolves a real bug found in the naive (unweighted) version: one player's ~5-month
continuous absence dominated the `perimeter_specialist` sample and flipped its sign
(see phase log's Injury-Calibration Decay Fix section). Severity multiplier (severe/moderate/minor) reuses the
existing `injury_features.severity_weights` config, not duplicated.

**Resolved:** `injury_features.sqlite`'s historical backfill sources from NBA official
pre-game PDF injury reports — valid for backtesting, not just live prediction.

### Layer 3: Historical Matchup Similarity

**Final method: KNN, k=81** (not cosine — reversed after proper tuning; cosine@0.70
was the Round-1 default but under-tuned KNN looked worse than it actually is).
`min_confidence_sample=21`, `full_confidence_sample=82`. Below the confidence floor,
falls back to A2 (H2H), never zero.

A KNN-with-similarity-floor hybrid (take up to k neighbors, but only those clearing a
minimum similarity) was tested to address the concern that plain KNN forces exactly k
neighbors even when not all are truly similar — it ties plain KNN exactly at k=81 (the
floor never binds at that k; only hurts if set aggressively, ≥0.6-0.7). Not adopted —
no benefit at the k values that work.

A recency cutoff (bounding the search to the last N years, to guard against
stylistically stale eras) was also tested: 2-5 years is statistically indistinguishable
from unbounded; only ≤1 year clearly hurts. **Not adopted** — full history (warm-started
from 2016) is used unbounded.

**Leakage discipline:** `np.searchsorted(dates, target_date, side="left")` on a
date-sorted array — excludes every game on the same date as the target, not just
earlier row positions (multiple games per night must not see each other).

### Layer 4: Role-Level Matchup Flags
Not built — requires real-time injury data (~30min pre-tip), live-prediction only. Out of scope for the exploration phase; still a candidate for future live-prediction work.

---

## Configuration (as implemented)

`configs/config.yaml`'s `style_matchup` block (formally typed via
`src/utils/config_loader.py`'s `StyleMatchupConfig`):

```yaml
style_matchup:
  fingerprint_window: 37
  decay_halflife: 13.2
  encoding: hand_picked
  similarity_method: knn
  knn_k: 81
  min_confidence_sample: 21
  full_confidence_sample: 82
  low_confidence_fallback: h2h
  archetype_method: percentile
  injury_calibration_decay_halflife_games: 20
  injury_impact:
    facilitator: {assist_rate: ..., pace_score: ...}
    scorer: {three_pt_reliance: ..., paint_activity: ...}
    combo: {...}
    rim_protector: {defensive_rating: ..., paint_activity: ...}
    perimeter_specialist: {defensive_rating: ...}
```

(Exact calibrated values in `configs/config.yaml` directly — they're re-derived from
data, not hand-set, so this doc doesn't duplicate them.)

---

## Validation Results

- **Signal beats A2 baseline**: 0.281–0.323 (config-dependent) vs. A2 alone's 0.118.
  Combining both barely beats A7 alone — A7 mostly subsumes A2's signal.
- **Robust across a 5-fold walk-forward CV** (not just one split): tuned config wins
  every fold, with lower variance than the untuned default.
- **Sanity checks pass**: score range within [-15,15], confidence within [0,1], 0 NaN,
  fallback rate well under 20%.
- Confidence did not clearly predict per-game accuracy (high vs. low confidence MAE
  were comparable) — noted as a real finding, not the clean gap originally hypothesized.

Full numbers, per-round detail, and every rejected alternative: `docs/a7_phase_log.md`.

---

## Resolved Decision Gate

- Encoding: hand-picked (validated, not just assumed).
- Similarity: KNN k=81 (not cosine@0.70 — reversed after tuning).
- Calibration: empirical, decay-weighted by absence duration.
- Injury data timing: confirmed pre-game.
- Low-confidence fallback: H2H (confirmed).
- Layer order: Layer 2 before Layer 3 (confirmed).

**Confirmed, no open items:** the hyperparameter search that chose the current config
predated a later-fixed normalization bug; a rerun under the fix (phase log's Post-Fix
Hyperparameter Recheck section) found nothing better, confirming the config selection
holds up. `feature_builder.py` integration itself was completed and tested by the
KNN-Score Integration Test — result: no measurable real-model accuracy improvement,
not adopted by default. A follow-up redesign (Raw-Fingerprint Feature Redesign, plus
an Expanding-Window Model CV robustness check) found real feature importance but a
signal that sharpens total-points accuracy rather than win/spread accuracy — also not
adopted by default (see phase log for full detail).

---

## Future Work (backlogged, not blockers)

- Richer style inputs (shot-chart/shot-zone data) — real scope expansion, see `docs/backlog.md`.
- PCA `n_components` sweep, further supervised-model tuning — both already lose by a clear margin; low expected value.
- Real-time injury pipeline for Layer 4 (live prediction only).
