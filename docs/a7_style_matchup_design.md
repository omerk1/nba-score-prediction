i m# A7: Style Matchup Score — Architecture & Design (v2)

> **Status:** EXPERIMENTAL — not yet integrated into feature_builder  
> **Supersedes:** A7_STYLE_MATCHUP_DESIGN.md (v1)  
> **Changes in v2:** Encoding alternatives added (PCA, clustering, embeddings); injury impact calibration approach revised; Layer ordering fix; data leakage warning; fallback behavior clarified.

---

## Problem Statement

**Current state:** A2 (Extended H2H) uses historical head-to-head records but ignores *how* teams play. Two teams with identical 3–0 H2H records could have completely different matchup dynamics if one team has shifted to perimeter-heavy while the other plays interior-focused.

**Solution:** Build a style-based matchup score that captures how roster compositions and playing styles interact. Injuries matter not just as player loss, but as *playstyle shifts* (missing a facilitator forces more iso, missing a rim-protector weakens interior D, etc.).

**Core hypothesis:** Certain stylistic matchups systematically favor one side, regardless of which specific teams are playing. By borrowing sample size from historically similar matchups across the league, we can generate signal even for team pairs with thin H2H history.

**Scope:** Exploratory module (no feature_builder integration yet). Validate that style-based signals improve on A2 alone before integrating.

---

## How Matchup Vectors Work

### Why not compare team styles directly?

The naive approach — comparing home style vector to away style vector — tells you "these teams play differently from each other." That's not useful. You want to know *who wins when this stylistic dynamic exists.*

### The right framing: a matchup is a single entity

Concatenate both teams' style fingerprints into one vector (home first, always):

```
matchup_vector = [home_pace, home_3pt, home_paint, home_def, home_ast,
                  away_pace, away_3pt, away_paint, away_def, away_ast]
```

Then search history for games whose matchup vector looks similar to tonight's. The average point differential of those similar games becomes your prediction signal.

**Why order matters:** `[fast_home vs slow_away]` ≠ `[slow_home vs fast_away]`. Home court + pace mismatch plays differently depending on which side has which style. Concatenation preserves this directionality. Direct comparison loses it.

### Building the vector: rolling pre-game window

The vector is built from each team's **last N games before the game date** — never from the game itself (data leakage).

```python
def build_matchup_vector(home_team, away_team, game_date):
    # CRITICAL: strict before=game_date filter to prevent leakage
    home_games = get_last_n_games(home_team, before=game_date, n=20)
    away_games = get_last_n_games(away_team, before=game_date, n=20)

    home_style = compute_fingerprint(home_games)
    away_style = compute_fingerprint(away_games)

    home_norm = normalize(home_style)   # normalize before concatenating
    away_norm = normalize(away_style)

    return [*home_norm, *away_norm]     # 10 values total
```

This vector is computed for **every historical game** upfront, creating a searchable index:

```
game_id | date       | matchup_vector (10 values)       | actual_home_margin
--------|------------|----------------------------------|-------------------
g_001   | 2020-01-05 | [0.8, 0.4, 0.3, 0.7, 0.6, ...] | +7
g_002   | 2020-01-06 | [0.3, 0.7, 0.8, 0.2, 0.4, ...] | -4
...
```

At prediction time: build tonight's vector, find similar rows in the index, average their margins.

---

## Encoding Approaches (Ablation)

This is the most important design decision. Three approaches are documented below in order of complexity. Start with Phase 1 and only upgrade if signal is validated.

---

### Encoding Phase 1: Hand-Picked Metrics (Current Design)

Compute a rolling style vector for each team from the last 20 games:

```python
{
    "pace_score":        avg((PTS + OPP_PTS + TO - FTA×0.44) / games),
    "three_pt_reliance": avg(3PA / FGA),
    "paint_activity":    avg(FTA / game),
    "defensive_rating":  avg(OPP_PTS / possessions × 100),
    "assist_rate":       avg(AST / FGM),
}
```

**Decay:** Exponential decay with half-life = 5 games. Recent games weighted ~2× heavier than older ones.

**Pros:** Simple, interpretable, directly tied to existing box score data.

**Known weaknesses:**
- Metrics are correlated (pace and assist_rate co-move; 3pt_reliance and paint_activity are inversely correlated). Correlated dimensions cause cosine similarity to double-count some aspects and underweight others.
- Only 5 dimensions → coarse style representation.
- Values are intuition-derived, not data-validated.

**When to move on:** If Phase 1 shows weak correlation (<0.2) with actual margins, or if similar-game lookup is consistently returning <10 matches.

---

### Encoding Phase 2: PCA on Richer Metrics (Recommended Upgrade)

Instead of hand-picking 5 metrics, compute 15–20 raw box score metrics per team, then use PCA to derive uncorrelated components.

```python
raw_metrics = [
    pace, three_pt_rate, paint_rate, ast_rate, reb_rate,
    to_rate, ft_rate, def_reb_rate, opp_3pt_allowed,
    opp_paint_allowed, second_chance_rate, fast_break_rate,
    avg_shot_distance, pull_up_rate, catch_shoot_rate
]

# fit PCA once on all team-seasons in training data
pca = PCA(n_components=5)
pca.fit(all_team_season_metrics)

# transform any team's rolling window into style vector
style_vector = pca.transform(team_rolling_metrics)  # 5 uncorrelated dimensions
```

Components naturally emerge as interpretable axes, e.g.:
- Component 1 ≈ pace / tempo axis
- Component 2 ≈ perimeter vs interior axis
- Component 3 ≈ defensive scheme axis

**Pros:** Uncorrelated dimensions → cosine similarity works cleanly. Data-derived rather than intuition-derived. Richer representation.

**Cons:** Less interpretable per-dimension. Requires fitting PCA on training data only (no leakage into test).

**Important:** Fit PCA on training set only. Transform test/live data using the fitted object.

---

### Encoding Phase 3: Clustering (Simpler Search, Guaranteed Sample Sizes)

Pre-cluster all team-seasons into style archetypes using k-means. A matchup becomes "archetype A vs archetype B" — look up historical outcomes for that pair directly.

```python
# fit once on all team-season rolling vectors
kmeans = KMeans(n_clusters=8)
kmeans.fit(all_team_rolling_vectors)

# at prediction time
home_archetype = kmeans.predict(home_style_vector)  # e.g. cluster 2
away_archetype = kmeans.predict(away_style_vector)  # e.g. cluster 5

# lookup historical average margin for this archetype pair
score = historical_margins[(home_archetype, away_archetype)].mean()
```

Example clusters (labels assigned post-hoc by inspection):
```
Cluster 0: fast/3pt-heavy    (Warriors-style)
Cluster 1: slow/paint-dominant (old Spurs-style)
Cluster 2: defensive grind   (early Bucks-style)
Cluster 3: transition-heavy  (early Suns-style)
...
```

**Pros:** Guaranteed sample sizes per matchup type (no "insufficient data" problem). Most interpretable. Easiest to debug and explain.

**Cons:** Loses nuance between teams within the same cluster. Sensitive to k choice. Discretizes a continuous space.

**When to use:** If Phase 2's cosine similarity search still returns thin samples, clustering ensures every matchup has a populated bucket.

---

### Encoding Phase 4 (Future): Learned Team Embeddings

Train a small neural network where each team-season gets a learned embedding. Similar to word2vec — teams with similar playing patterns end up close in embedding space automatically. Requires more data and infrastructure. Out of scope for current exploration phase.

---

## Architecture Layers

### Layer 1: Team Style Fingerprint

Compute rolling fingerprint per team per game using whichever encoding phase is active. See Encoding Approaches above.

### Layer 2: Injury-Adjusted Style

**Correct order:** Layer 2 must adjust the fingerprint *before* Layer 3 uses it for similarity search. The similarity search should operate on injury-adjusted vectors, not base vectors.

**Player Archetypes** (from player stats cache, percentile-based):

| Archetype | Criteria | Why Percentile? |
|-----------|----------|-----------------|
| Facilitator | AST >75th pct, PPG <25th pct | Era-adaptive — 15 PPG meant different things in 1990 vs 2020 |
| Scorer | PPG >75th pct, AST <25th pct | |
| Rim Protector | BLK >75th pct, REB >75th pct | |
| Perimeter Specialist | BLK <25th pct, STL >75th pct | |

**Injury Impact — v1 (config-based, estimated):**

When archetype X is out, shift the team's style fingerprint:

```yaml
injury_impact:
  facilitator:          {assist_rate: -0.15, pace_score: -0.1}
  scorer:               {three_pt_reliance: +0.1, paint_activity: +0.1}
  rim_protector:        {defensive_rating: +2.5, paint_activity: -0.15}
  perimeter_specialist: {defensive_rating: +1.5}
```

Apply severity multiplier: severe ×1.0, moderate ×0.6, minor ×0.3.

**⚠️ Known issue with v1:** These delta values are manually estimated, not data-derived. They should be treated as placeholders until calibration is run.

**Injury Impact — v2 (empirically calibrated, recommended):**

Before implementing Layer 2, run a calibration script to derive deltas from historical data:

```python
# for each archetype, find all games where that archetype was missing
# compare team's style metrics in those games vs their season baseline
# the empirical delta IS the config value

for archetype in ['facilitator', 'scorer', 'rim_protector', 'perimeter_specialist']:
    games_without = get_games_missing_archetype(archetype)
    games_baseline = get_season_baseline(same_teams, same_seasons)
    
    delta = games_without[metrics].mean() - games_baseline[metrics].mean()
    print(f"{archetype}: {delta.to_dict()}")
    # write output back to config
```

This script should run once and populate `injury_impact` in config before Layer 2 is built. Turns guesses into data-grounded constants.

**Open question:** Does `injury_features.sqlite` store injury status as it was known *pre-game*, or does it reflect post-game reporting? If post-game, Layer 2 is only usable going forward (live predictions), not historically (backtesting).

### Layer 3: Historical Matchup Similarity

**Goal:** Find past games with similar matchup dynamics and use their outcomes as a signal.

**Implementation (cosine similarity approach):**

1. Build matchup vector for every historical game (injury-adjusted, normalized)
2. At prediction time, build tonight's matchup vector
3. Compute cosine similarity against all historical vectors
4. Keep games above threshold
5. Average their actual home margins → style_matchup_score

**Similarity threshold:** Start at **0.70**, not 0.80. The original 0.80 is aggressive for a 10-dimensional space and will cause frequent "no similar games found" failures. Tighten only if results are noisy.

**KNN alternative (more robust):** Instead of a hard threshold, take the **K most similar games** regardless of score (K=30 recommended). Guarantees a sample always exists. Degrades gracefully for unusual matchups rather than returning null.

```python
# threshold approach (original)
similar = [(game, sim) for game, sim in history if sim > 0.70]

# KNN approach (more robust)
similar = sorted(history, key=lambda x: x.similarity, reverse=True)[:30]
```

**Confidence scoring:**
```
confidence = min(similar_games_count / 50, 1.0)
```
- 50+ similar games → confidence = 1.0
- <10 similar games → confidence < 0.2 (treat as weak signal)

**Fallback behavior (previously undefined):** When confidence < min_confidence_sample threshold, do NOT return score=0. Instead fall back to A2 (H2H score). This preserves signal continuity and prevents the feature from silently contributing nothing on low-data matchups.

**Warm-start:** Use pre-training data (2016–2020) to seed the historical index. Gives 4 seasons of matchup history before training begins.

**⚠️ Data leakage check:** Ensure the rolling window for any game strictly excludes that game's date. Vectorized pandas operations are easy to get wrong here — use explicit `df[df['date'] < game_date]` filters, not just `.shift()`.

### Layer 4: Role-Level Matchup Flags (Exploration Only)

Flag specific archetype mismatches at the roster level:
- Opponent missing rim protector → home paint advantage flag
- Opponent missing perimeter defender → home 3pt advantage flag

**Not integrated into training pipeline yet** — requires real-time injury data (injuries reported ~30min before tip-off). Useful for live prediction only, not historical backtesting.

---

## Data Dependencies

| Data | Source | Status | Notes |
|------|--------|--------|-------|
| Box scores (rolling) | Data warehouse | ✅ Ready | For Layer 1 fingerprints |
| Player stats cache | PR #19 (1.85M rows) | ✅ Ready | For archetype classification |
| Injury severity | injury_features.sqlite | ✅ Ready | Confirm pre-game vs post-game reporting |
| Historical lineups | A4 (box scores) | ✅ Ready | For Layer 4 (validation only) |

---

## Configuration

```yaml
style_matchup:
  # fingerprint
  fingerprint_window: 20            # games in rolling window
  decay_halflife: 5                 # exponential decay half-life in games
  encoding: "hand_picked"           # hand_picked | pca | clustering

  # pca settings (encoding: pca only)
  pca_n_components: 5
  pca_raw_metrics:                  # which raw metrics to feed into PCA
    - pace, three_pt_rate, paint_rate, ast_rate, reb_rate
    - to_rate, ft_rate, def_reb_rate, opp_3pt_allowed, opp_paint_allowed

  # clustering settings (encoding: clustering only)
  clustering_k: 8

  # similarity search
  similarity_method: "cosine"       # cosine | knn
  similarity_threshold: 0.70        # used when method=cosine
  knn_k: 30                         # used when method=knn
  min_confidence_sample: 10
  full_confidence_sample: 50
  low_confidence_fallback: "h2h"    # what to return when confidence < min: h2h | zero

  # archetypes
  archetype_method: "percentile"
  archetype_percentiles:
    facilitator:          {ast_pct: 0.75, ppg_pct: 0.25}
    scorer:               {ppg_pct: 0.75, ast_pct: 0.25}
    rim_protector:        {blk_pct: 0.75, reb_pct: 0.75}
    perimeter_specialist: {blk_pct: 0.25, stl_pct: 0.75}

  # injury impact (populate from calibration script before using)
  injury_impact:
    facilitator:          {assist_rate: -0.15, pace_score: -0.1}      # estimated
    scorer:               {three_pt_reliance: +0.1, paint_activity: +0.1}
    rim_protector:        {defensive_rating: +2.5, paint_activity: -0.15}
    perimeter_specialist: {defensive_rating: +1.5}

  injury_severity_multipliers:
    severe:   1.0
    moderate: 0.6
    minor:    0.3
```

---

## Output Format

```python
def style_matchup_score(home_team_id, away_team_id, game_date,
                        home_injuries, away_injuries) -> dict:
    return {
        "score": 2.3,                   # ±pts for home; falls back to H2H if low confidence
        "confidence": 0.75,             # 0–1 based on similar_games_count
        "similar_games_count": 18,
        "similarity_method": "cosine",  # which method was used
        "encoding_phase": 1,            # which encoding was active
        "fallback_used": False,         # True if H2H fallback was triggered
        "bucket_name": "fast_vs_slow",  # descriptive label (clustering only)
        "home_style": {...},            # full adjusted fingerprint
        "away_style": {...},            # full adjusted fingerprint
        "role_flags": {...},            # Layer 4 (optional, None if not computed)
    }
```

---

## Validation Plan

**Sanity checks:**
- Score in range [-15, 15]
- Confidence ∈ [0, 1]
- No NaN in fingerprints
- Fallback rate < 20% (if higher, threshold is too strict)

**Signal validation:**
- Correlation of style_score vs actual home margin on test set (target: >0.3)
- Confidence calibration: high-confidence predictions should have lower MAE than low-confidence ones

**Comparison:**
- Side-by-side CSV: A2 (H2H) vs A7 (style) vs A2+A7 combined on last 100 games
- Does A7 add signal *on top of* A2, or does it mostly overlap?

**Encoding ablation:**
- Phase 1 (hand-picked) vs Phase 2 (PCA) — does richer encoding improve correlation?
- Cosine threshold vs KNN — which returns better sample sizes with similar accuracy?

**Layer ablation:**
- Layer 1 only
- Layer 1 + 2 (+ injury adjustment)
- Layer 1 + 2 + 3 (+ similarity search)

---

## Implementation Roadmap

| Phase | Work | Depends On | Notes |
|-------|------|-----------|-------|
| 0 | Run injury impact calibration script | Historical data | Derive Layer 2 deltas empirically before building Layer 2 |
| 1 | Layer 1: fingerprints + config + historical index | None | Use hand-picked encoding (Phase 1) |
| 2 | Layer 2: injury adjustment | Phase 0 calibration + archetype classifier | Must run before Layer 3 |
| 3 | Layer 3: similarity search (cosine + KNN) | Layers 1+2 | Try both; compare fallback rates |
| 4 | Validation + exploration CSV | Layers 1–3 | A2 vs A7 side-by-side |
| 5 | Encoding upgrade (PCA) | Layer 4 validation | Only if Phase 1 encoding shows weak signal |
| 6 | Integration into feature_builder | A7 validated | Post-exploration only |

---

## Decision Gate

Before implementation, confirm:

- [ ] Encoding phase to start with: hand-picked (Phase 1) ← recommended
- [ ] Similarity method: cosine with threshold=0.70, or KNN with k=30?
- [ ] Run calibration script (Phase 0) before building Layer 2?
- [ ] Confirm injury_features.sqlite is pre-game reporting (not post-game)
- [ ] Low-confidence fallback: H2H score (recommended) or zero?
- [ ] Layer order confirmed: Layer 2 (injury adjust) runs before Layer 3 (similarity search)

**Proceed?** _(User to confirm)_
