# PIPELINE_AUDIT.md

Read-only reconnaissance — systematic audit of the raw-data-to-final-feature-vector path, hunting for latent logical bugs (not just surface issues), prioritizing high-weight families (`style_fingerprint`, `elo`, `style`, `matchup`, `rolling`) and sources with prior known bugs. No fixes applied, nothing proposed — findings only, for triage.

**Headline: zero leakage bugs found**, across genuinely adversarial checking of loading, the top-6 families' processing, pipeline wiring, and final-vector representation. That's a meaningfully strong, reassuring result given how hard this was pushed to find one — it means the champion's current numbers are very unlikely to be inflated by a hidden leakage bug. Everything below is either a latent/unguarded risk (not currently live) or a genuine intent question, not an active corruption of results.

---

## Stage 1 — Loading: clean, two latent (not live) gaps

- **`data_loader.py`**: no joins, no row-loss risk, correct type/date handling. One fragility note: the train/val/test date-boundary enforcement is split across two files (loader applies the upper bound, `cv_harness.py` applies the lower) — correct today, but a future "simplification" of either file in isolation could silently reintroduce a leak.
- **Box-score parity check** (`box_scores.py`): only verifies every game has *at least one* team's box score row, not exactly two (home+away). A partially-failed fetch could pass this check while being silently incomplete. **Verified empirically clean today (0/12,793 games affected)** — this is a gap in the safeguard itself, not a live bug.
- **Style-fingerprint cache freshness** (`feature_builder.py:896`): no check that the cache is current relative to the raw DB. **Verified not stale right now** — cache max game_date (`2026-05-24`, includes playoffs) is actually *ahead* of the raw DB's regular-season max (`2026-04-12`). But nothing would catch it if this ever drifted, and it's the #1-importance family.

## Stage 2 — Processing: clean on correctness, one real intent-clarity finding

- **Off-by-one/lookahead**: consistent `shift(1)`-before-`rolling()` everywhere across rolling/style/opponent-quality/venue-delta — checked line by line, no inconsistency. `_add_opponent_quality_features`'s apparent double-shift is two different, both-necessary lags (inner: opponent's own point-in-time stat; outer: excludes the team's current game from its own window), not a bug.
- **H2H sign/canonical-team logic**: fully traced and verified correct — margins negate and win-rates complement correctly depending on which side is home, at every conversion point. This is exactly the kind of asymmetric-pairwise-feature bug class that tends to hide sign flips, and it's clean. `_compute_h2h_home_away_splits`'s correctness is fragile-by-invariant (depends on `create_all_features`'s index sort+reset three calls upstream) — correct today, worth a defensive assertion if that code is ever touched.
- **`opponent_quality_features`**: verified genuinely uses the *opponent's* stat, not self-referential — real schedule-strength signal, not circular.
- **`_compute_venue_delta`**: sign convention consistent (home-avg minus away-avg on both sides), `by="team_id"` grouping correct. One low-severity fragility note: relies on inherited sort order from `create_all_features` rather than re-sorting locally.
- **The one real finding**: `home_team_win_pct_L20` etc. are **venue-scoped**, not overall form — computed only over that team's last 20 *home* games (same for away). This is intentional and self-documented in the code, and every downstream family (matchup, venue-delta) inherits it consistently — but it means **there is currently no venue-blind "team's actual overall recent form" feature anywhere in the model.** Open question: is venue-split form what was intended, or is a venue-blind form feature a real gap?
- **NaN handling**: clean everywhere checked — genuine NaN for true first-game-in-window cases, never silently filled with a false zero.

## Stage 3 — Wiring: clean

Family call order (`create_all_features`) matches the documented order exactly (cross-checked against `run_family_importance.py`'s `FAMILY_STEPS`). Every temp column (H2H's canonical-team scratch columns, rolling's win/diff scratch columns) is verifiably dropped before the model sees anything — no orphans, swept the whole file. Exclude-list checked exhaustively against all 30 raw SQL columns — full coverage, no raw box-score stat leaks through as a "feature." Column set is config-gated (identical across train/val/test), not data-content-gated, so no split-misalignment risk found.

## Stage 4 — Representation: clean, confirms the one known gap and nothing more

127 real features, all numeric (116 float64, 11 int64), zero stray string columns — confirmed empirically against a real built feature frame, not traced/guessed. All 11 int64 columns are genuinely ordinal/binary (rest-day counts, tz-shift, injury counts, back-to-back flags), not disguised categoricals. Checked every plausible additional candidate (`SEASON_TYPE`, injury reason codes, day-of-week, the `scorer` flag) — all absent from the feature vector or non-issues. **Team ID remains the only real `cat_features` gap** — nothing new found. Exclude-list is opt-out architecture (a stray future column silently becomes a feature unless someone remembers to exclude it) — currently harmless, worth knowing. Continuous features (rolling stats, ratings, differentials) explicitly not flagged for hand-built target encoding — CatBoost's own splits already model these natively; manual target encoding would only add leakage surface for no benefit.

---

## Ranked for triage

| # | Finding | Type | Leakage/Degradation | Status |
|---|---|---|---|---|
| 1 | Rolling features are venue-scoped only, no overall-form feature exists | Intent question | Neither — a scope decision to confirm | Needs explicit call |
| 2 | Fingerprint cache has no freshness check vs. raw DB | Wiring gap | Degradation (if it ever fires) | Unguarded, not live today |
| 3 | Box-score parity check only verifies ≥1 row, not =2 | Correctness gap in a safeguard | Degradation (if it ever fires) | Unguarded, not live today |
| 4 | Team ID never reaches the model, not even as `cat_features` | Representation gap | Neither — untested upside | Cheap, isolated test candidate |
| 5 | Exclude-list is opt-out, not opt-in | Structural risk | Neither currently | Harmless today, worth knowing |
| 6 | Train/val date-boundary logic split across two files (loader + `cv_harness.py`) | Fragility | Neither currently | Harmless today, worth knowing |

**Resolution status (as of 2026-08-24; table above left as originally written):** #1 — implemented and adopted, venue-blind overall form (`docs/NEXT_PHASE_SESSIONS.md` Track A item 2, `docs/EXPERIMENTS.md`'s `a2_venue_blind_overall_form`). #2/#3 — both fixed as safeguards (Track A item 1, `a1_fingerprint_freshness_box_score_parity`). #4/#5/#6 — still open, no fix attempted (#4 is also flagged, independently, in `docs/EXPLORATION.md` Area 3).

---

## Addendum — representation-granularity investigation (rest, travel, recent form, workload)

Separate from the categorical/target-encoding question above: for families that compress rich underlying schedule/workload structure into a scalar or a few summary values, checked whether the model is seeing enough of that structure.

**Rest & schedule** — real structure discarded. Current: `rest_days`, `back_to_back`, `games_in_4_nights` (3 cols/side). Missing: (1) first-vs-second night of a back-to-back — currently indistinguishable, though the second night (already played yesterday, often after travel) is the higher-fatigue game — propose `{prefix}_second_of_b2b`, 2 cols; (2) density beyond the single 3-in-4-nights window — propose `{prefix}_games_last_7d`/`_14d`, 4 cols; (3) days since last genuine break (2+ rest days), distinct from days since last game, 2 cols. **Total: 8 columns.**

**Travel** — real structure discarded too. Current: `travel_miles`, `tz_shift`, `travel_miles_7d/14d` (4 cols/side). Missing: (1) road-trip length in stops / homestand length, not just cumulative miles, 2–4 cols; (2) days since last timezone change (cumulative jet-lag, distinct from the single most-recent shift), 2 cols; (3) altitude (already flagged elsewhere, folded in here), 1–2 cols. **Total: ~6 columns.**

**Recent form** — partially already tested. Rolling std/variance and L5−L20 trend of scoring margin were already measured this session and found **not supported** (correlations under 0.03, R² gains under 0.15% of baseline) — not re-proposed. One genuinely untested angle: **streak length** (`{prefix}_current_streak`, signed win/loss run length) — distinct from win_pct over a fixed window, since two teams can share the same rate with opposite momentum. 2 cols, moderate-low confidence.

**Workload** — no raw structure exists to compress; out of scope. Checked directly: `box_score_stats` (team-level) has no MIN/OT column at all. `player_stats_cache` (player-level) stores only `AST/BLK/FG%/PPG/REB/STL` — no MIN either. There is no minutes-played or overtime data stored anywhere in this project. This is a genuine data-collection gap (same category as the play-by-play/lineup gaps already flagged elsewhere), not a representation-compression problem — flagging honestly rather than forcing a proposal onto data that doesn't exist.

**Overfitting discipline**: ~16 columns combined if built in one shot — too much for one experiment. Stage it: `second_of_b2b` alone first (cheapest, most theoretically load-bearing) → density windows if that shows anything → streak length as its own separately-tested hypothesis → travel trip-length/tz-recency as its own batch. Every batch needs the full fold2-5 CV guardrail *and* a market-benchmark re-run — the actual question this serves is whether disagreement-accuracy against the market improves, not just whether `val_score_mean` ticks up. Every proposed column is one independently-interpretable scalar (a count, a signed integer, a binary flag) that CatBoost's axis-aligned splits use directly — none of this is a learned encoder or a blended/dense representation.

---

**Update (2026-08-24):** `second_of_b2b` was tested (Track A item 3) and confirmed a pure duplicate of the already-venue-blind-fixed `back_to_back` — not added. The home/away rest differential proposal (distinct from `second_of_b2b`) was later built and tested as `b2b_diff`/`rest_diff` (`docs/EXPERIMENTS.md`'s `b2b_rest_diff`) — favorable mean val_score but failed the per-fold guardrail, rejected. Density windows, days-since-last-break, travel trip-length/tz-recency, and streak length remain untested. Everything else above (loading/processing/wiring/representation findings, the ranked-triage table) is unchanged.
