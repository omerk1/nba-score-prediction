# Season Motivation & Seeding Incentive — Implementation Log

> Companion to `docs/SEASON_MOTIVATION_DECISIONS.md` (data audit + formula
> decisions). Branch `feature/season-motivation`, committed incrementally.

## 1. Data Audit

- No new backfill needed — standings/schedule derive in-memory from `nba_api.sqlite`'s
  `game` table (every touched season, 2018-19 through 2025-26, is already complete),
  same "derive, don't refetch" approach as `_add_elo_features`/`_add_rolling_features`.
- A3/A4 (player projections/lineup collector) confirmed unusable for roster-behavior
  despite "Complete" backlog status (A3's `_get_team_roster` is an unfinished
  placeholder; A4 returns season-end rosters, not per-game participants). Reused
  `injury_features.sqlite`'s `player_importance` (full-strength quality) and
  `player_injuries` (who sat, why) instead.
- Conference assignment hardcoded as `_TEAM_CONFERENCE` (fixed fact 2018-2026,
  same convention as `_TEAM_LOCATIONS`).
- Bug found+fixed: Playoff-tagged `season_id`s (`4YYYY`) appear as val/test
  warm-up context and have no entry in the Regular-Season-only
  `season_start_by_season` map — `compute_roster_behavior_scores` crashed
  comparing a date against `None`. Fixed: missing lookup → default 0.0
  (these rows never reach the scored dataset anyway).

## 2. Implementation

`src/feature_engineering/season_motivation.py` (mirrors `elo.py`'s separation
of sequential computation from `feature_builder.py`'s thin per-feature methods):
- `compute_standings_metrics`: per-team game log → `(season_id, team_id,
  snapshot_date)` standings panel via `merge_asof`. Ranks by win% within
  conference; computes `pressure_raw` (games-back from the 10-seed line,
  moderated by games remaining) and clinch countdowns.
- `compute_roster_behavior_scores`: per `(team_id, game_date)`, ratio of
  sat-out-healthy player importance (reused `_get_importance_map`-style
  formula from `src/news_scraping/pipeline.py`) to full-strength quality.

`_add_season_motivation_features` combines: `motivation_score =
clip(pressure_raw * (1 - roster_behavior_weight * roster_behavior_score), 0, 1)`.
Soft-disabled if the injury features cache is missing. New
`SeasonMotivationConfig` (`enabled`, `playoff_line_seed=10`,
`roster_behavior_weight`, `min_importance_games=5`) — no `db_path`, reads
existing tables directly.

## 3. Validation Results

Sanity checks (9,509 Regular Season games): 0 NaN rows, 0 out-of-[0,1]
`motivation_score`, 0 negative clinch values. Distribution matches the
intended two-sided decay: mid-season mean ≈0.95, late-season spreads across
[0,1] with a large mass at 0.0.

**MAE comparison** (baseline 125 features vs. treatment 131 features,
`roster_behavior_weight=1.0`):

| metric | baseline | treatment | direction |
|---|---|---|---|
| val diff_mae | 11.130 | **11.081** | better |
| test diff_mae | 11.592 | **11.543** | better |
| val win_acc | 0.6588 | **0.6661** | better |
| test win_acc | 0.6596 | **0.6735** | better |
| val brier | 0.2129 | **0.2109** | better |
| test brier | 0.2109 | **0.2095** | better |
| val total_mae | 14.752 | 14.803 | slightly worse |
| test total_mae | 15.452 | 15.531 | slightly worse |

5/8 metrics improve simultaneously. Feature importance: the 6 new columns
rank 30th/37th/38th/70th/75th/111th of 131 — 3 of 6 in the top third. Clinch
countdowns rank higher than `motivation_score` itself (a hint later exploited
in §5).

**Weight exploration** (`roster_behavior_weight`, full ablation pipeline):

| weight | val diff_mae | test diff_mae | val win_acc | test win_acc | val brier | test brier |
|---|---|---|---|---|---|---|
| 0.0 | 11.115 | 11.580 | 0.6571 | 0.6710 | 0.2124 | 0.2111 |
| 0.5 | 11.115 | 11.579 | 0.6571 | 0.6710 | 0.2124 | 0.2110 |
| **1.0** | **11.081** | **11.543** | **0.6661** | **0.6735** | **0.2109** | **0.2095** |
| 1.5 | 11.126 | 11.594 | 0.6612 | 0.6743 | 0.2126 | 0.2111 |
| 2.0 | 11.101 | 11.566 | 0.6588 | 0.6678 | 0.2118 | 0.2098 |
| 3.0 | 11.115 | 11.589 | 0.6539 | 0.6743 | 0.2123 | 0.2110 |

`1.0` wins 5/6 metrics cleanly; only 2.2% of team-game rows ever have a
nonzero `roster_behavior_score` (median 0.065, max 0.234), which explains
why 1.5-3.0 form a noisy, non-monotonic band rather than a clean curve — too
few affected rows for more resolution than "off/half-off vs. on vs.
overshooting." Kept `1.0` as final default.

## 4. Expanding-Window CV

5-fold walk-forward (1 season val + 1 test, expanding backward; fold1 =
committed default split). `roster_behavior_weight=1.0` throughout.

| fold | train through | val | test | metrics favoring treatment |
|---|---|---|---|---|
| 1 (default) | 2023-24 | 2024-25 | 2025-26 | **6/6** |
| 2 | 2022-23 | 2023-24 | 2024-25 | 1/6 |
| 3 | 2021-22 | 2022-23 | 2023-24 | 2/6 |
| 4 | 2020-21 | 2021-22 | 2022-23 | 5/6 |
| 5 | 2019-20 | 2020-21 | 2021-22 | 2/6 |

**16/30 (53%) — essentially a coin flip.** Fold3's test_diff_mae regresses
from 10.992 to 11.213. The single-split result (fold1) does not generalize.
Unlike on/off-splits (folds 4-5 byte-identical to baseline, since that
feature was a structural no-op before `player_injuries` coverage began),
`motivation_score`'s standings/clinch components are injury-independent, so
even early folds show real (non-identical) differences — doesn't rescue the
overall result, though.

## 5. Raw-Component Decomposition (Round 2)

Motivation: clinch columns (raw) ranked top-third of importance while
`motivation_score` (combined) ranked near bottom — same pattern as A7's
style-matchup redesign, where decomposing a KNN-similarity score into raw
components turned near-zero-importance features into the top 2. Tried the
same move here: retired `motivation_score`, exposed `standings_pressure` +
`roster_behavior_score` as raw columns plus 4 home-minus-away diff columns
(6 → 12 columns). Added `tests/test_season_motivation_features.py` (13
tests, missing since Phase 1).

| fold | baseline diff_mae (val/test) | v2 diff_mae (val/test) | baseline win_acc (val/test) | v2 win_acc (val/test) | metrics favoring v2 |
|---|---|---|---|---|---|
| 1 | 11.130 / 11.592 | 11.167 / 11.609 | 0.6588 / 0.6596 | 0.6571 / 0.6678 | 1/6 |
| 2 | 11.053 / 11.118 | 11.131 / 11.150 | 0.6455 / 0.6563 | 0.6415 / 0.6563 | 0/6 |
| 3 | 10.265 / 10.992 | 10.244 / 11.163 | 0.6252 / 0.6577 | 0.6244 / 0.6577 | 2/6 |
| 4 | 11.299 / 10.355 | 11.294 / 10.385 | 0.6366 / 0.6130 | 0.6407 / 0.6106 | 2/6 |
| 5 | 11.268 / 11.311 | 11.291 / 11.390 | 0.6241 / 0.6382 | 0.6306 / 0.6195 | 1/6 |

**Worse, not better: 6/30 (20%), down from 16/30 (53%).** Fold1's clean 6/6
collapsed to 1/6. Why the A7 precedent didn't transfer: A7's combined score
discarded genuinely independent information (5 style dimensions → 1
similarity number) that decomposition restored; here `motivation_score`
combined exactly two already-simple signals with one multiplicative formula
— nothing hidden to unlock, and the 4 added diffs are redundant with the 8
raw columns a tree can already split on. **Reverted.** Recorded as a
genuine negative result: decomposition isn't automatically a win just
because it worked once elsewhere.

## 6. Open Improvement Ideas (documented, then tried)

### 6.1 Dual-threshold standings pressure (6-seed AND 10-seed)

Gap: `standings_pressure` only measures distance from the 10-seed
(play-in cutoff), never the 6-seed (direct-berth cutoff) — a team safely
clear of 10th but fighting for 6th reads as low-pressure.

**Max-based fix**: `pressure_raw = max(pressure_vs_10, pressure_vs_6)`.
Verified on real data: mean pressure rose 0.617→0.735 (real movement).

**CV: 10/30 (33%)** — worse than single-threshold's 53%. Diagnosed
directly (not just re-tried blindly): `max` systematically shifts pressure
**upward** (mean 0.776→0.851) while **compressing variance** (std
0.310→0.260) — less spread for a tree to split on, regardless of the
underlying idea's merit.

**Sharpened fix**: weighted average (`direct_playoff_weight`, default 0.5)
instead of `max` — preserves the mean (0.778) with much less variance loss
(std 0.282).

**CV: 12/30 (40%)** — improved from 33%, confirming the diagnosis, but
still below single-threshold's 53%. **Not adopted; `direct_playoff_seed`
stays `null`.** Code/tests kept (harmless when null) — e.g. `direct_playoff_weight`
itself was only tried at 0.5.

### 6.2 Recent-minutes trend (deeper tanking detection)

Gap: `roster_behavior_score` is a single-night snapshot — misses "soft"
tanking (a coach quietly cutting minutes over several games without an
official injury-report tag). Implemented and CV-tested in §9: compares each
player's current cumulative minutes average against their own average from
`recent_trend_lookback_weeks` (4) earlier — a genuine drop over that window
signals reduced recent minutes without needing a `games_played` count
`player_importance` doesn't store.

## 7. Reverted Round 2, Restored Test Coverage

Round 2 (§5) reverted — combined `motivation_score` restored as current
state. Added `tests/test_season_motivation_features.py` (13 tests: standings/clinch
formulas, roster-behavior scoring, config-gating, the combination formula).

## 8. Magnitude-Weighted Cross-Variant Comparison

Every CV verdict so far used win-count (treats a tiny loss same as a large
one). Redone with mean per-metric deltas across the 5 folds:

| variant (win-count) | val diff_mae | **test diff_mae** | val win_acc | test win_acc | val brier | test brier |
|---|---|---|---|---|---|---|
| v1 single-threshold (53%) | −0.0018 | **−0.0666** | +0.0023 | −0.0005 | +0.0001 | −0.0017 |
| v2 raw-decomposition (20%) | −0.0224 | **−0.0658** | +0.0008 | −0.0026 | −0.0012 | −0.0016 |
| dual-threshold MAX (33%) | −0.0266 | **−0.0628** | −0.0039 | −0.0006 | −0.0005 | −0.0018 |
| dual-threshold AVG (40%) | −0.0054 | **−0.0766** | 0.0000 | −0.0049 | 0.0000 | −0.0020 |

**`test_diff_mae` is negative by almost the same amount (−0.063 to −0.077)
in every variant, regardless of win-count or formula.** v1's "best" 53% was
driven by noisier val-side/win_acc metrics, not by test_diff_mae — the one
metric that stays consistently negative across every design is the one
measuring held-out point-differential accuracy. Reads as a structural cost
of adding these features (more model capacity, same real signal, slightly
worse test generalization), not something a different formula is likely to
fix.

## 9. Recent-Minutes-Trend Signal

`recent_minutes_trend_score` (§6.2) CV-tested on top of v1's base design.
Fires nonzero in ~99% of rows (mean 0.021) — vs. `roster_behavior_score`'s
~2-4%, raising concern it's picking up routine variance, not deliberate role
reduction.

- vs. pure baseline: 8/30 (27%).
- vs. v1 (isolated marginal effect): 11/30 (37%).
- Magnitude: `test_diff_mae` mean **+0.0224** — looks like the first
  improvement on the metric that's been negative everywhere else — but
  per-fold it's −0.021, −0.014, **+0.081**, −0.043, **+0.109**: 3/5 folds
  worse, the positive mean driven entirely by 2 large swings, not a
  consistent effect.

**Not adopted.** Win-count worse than v1 alone, and the one promising
average doesn't survive a fold-by-fold check. The 99%-nonzero rate suggests
this proxy mostly captures normal roster fluctuation, not deliberate
tanking. Code/tests kept.

## 10. Phase 1 Iteration — Behavioral Signals

§8's ceiling was structural: every *input*-based variant (standings
position, roster/rest decisions) degraded `test_diff_mae` by about the same
amount regardless of formula. This round tests recent game *behavior*
directly instead.

### Signals tried

- **`performance_vs_expectation_score`**: rolling mean (window=10) of
  `actual margin − Elo-expected margin`, normalized by the residual's global
  std. Elo-diff-to-margin scale fit via least-squares from this repo's own
  history (`_fit_elo_margin_scale`) — not an external heuristic, since this
  repo's Elo params are independently tuned.
- **`opponent_adjusted_form_score`**: rolling mean (window=10) of a signed,
  opponent-strength-weighted outcome (`opponent_win_pct` for a win,
  `-(1-opponent_win_pct)` for a loss).

Both independent raw columns (own `..._enabled` flag, `style_matchup`'s
`enabled`/`raw_features_enabled` convention), reusing `elo_features`'
ratings.

### CV results (isolated on top of current base design: `motivation_score`
+ clinch + `recent_minutes_trend_score`)

| variant | win-count | test_diff_mae mean delta | test_diff_mae per-fold (fold1→5) |
|---|---|---|---|
| Signal 1 alone | 17/30 (57%) | **+0.0510** | −0.039, +0.045, +0.083, **+0.124**, +0.042 |
| Signal 2 alone | 17/30 (57%) | **+0.0444** | +0.011, −0.012, +0.081, +0.092, +0.050 |
| Both combined | 19/30 (63%) | −0.0076 | +0.041, −0.048, −0.007, +0.023, −0.047 |

(vs. pure baseline, win-counts are lower — 43%/40%/57% — and noisier, since
that comparison also carries the base design's own mixed effect.)

### Verdict

- **Signal 1: passes** — 4/5 folds improve, the one exception small.
  First signal in the whole investigation to show *consistent* (not just
  average-favorable) `test_diff_mae` improvement.
- **Signal 2: passes**, same standard (4/5 folds, one small exception).
- **Combined: does not clearly pass** — win-count is highest yet (63%), but
  the specific bar both passed individually breaks down when stacked (2/5
  folds, mean turns negative).

### Structural findings

Two signals passing individually but failing to combine cleanly on
`test_diff_mae` (while combining fine on win-count) reads as **redundancy,
not complementarity** — both are built from the same substrate (recent game
outcomes/margins), just filtered differently (Elo expectation vs. opponent
strength). Which one (if either) to adopt was left open pending the window
sweep below.

### Window sensitivity sweep

Both passed at `window=10`. Repeated the same isolated CV at `window=5` and
`window=15` (4 sweep points × 5 folds = 20 runs) before treating that as real.

| variant | win-count | test_diff_mae mean delta | test_diff_mae per-fold (fold1→5) | folds improved |
|---|---|---|---|---|
| Signal 1, window=5 | 17/30 (57%) | −0.0384 | +0.040, −0.097, −0.005, −0.102, −0.028 | 1/5 |
| Signal 1, window=10 | 17/30 (57%) | **+0.0510** | −0.039, +0.045, +0.083, +0.124, +0.042 | 4/5 |
| Signal 1, window=15 | 19/30 (63%) | −0.0586 | +0.010, −0.057, −0.074, −0.130, −0.042 | 1/5 |
| Signal 2, window=5 | 13/30 (43%) | −0.0386 | +0.016, −0.072, −0.024, −0.089, −0.024 | 1/5 |
| Signal 2, window=10 | 17/30 (57%) | **+0.0444** | +0.011, −0.012, +0.081, +0.092, +0.050 | 4/5 |
| Signal 2, window=15 | 15/30 (50%) | −0.0498 | +0.042, −0.067, −0.064, −0.111, −0.049 | 1/5 |

Full test suite: 150/150 passing. `config.yaml` byte-identical to HEAD
after the sweep.

**Overturns the window=10 verdict.** At both neighboring windows, for both
signals, `test_diff_mae` flips to *consistently negative* (4/5 folds worse)
by a magnitude comparable to or larger than window=10's improvement. The
same fold (fold1) stays positive at window=5/15 for every point, while the
folds that drove window=10's passes (2-5) all reverse. `pve_w15`'s 63%
win-count is the highest of any variant this whole investigation, yet its
fold-consistency is the worst (1/5) — the win-count-vs-magnitude divergence
§8 warned about.

A signal that only clears the bar at one of three window values, and
inverts (not just weakens) at the others, is much more consistent with
**window=10 being a favorable draw against these 5 CV splits** than a real
effect. Nothing rules out a real, window-specific effect, but that would
need a mechanism, not just this data — the sweep can't distinguish "real
but window-specific" from "overfit to one hyperparameter draw."

**Revised verdict: neither passes a window-robustness check.** Both stay
implemented, tested, disabled by default — available for future
re-evaluation (more folds, a different window/fold-count combination) but
not adopted on current evidence.

## 11. Phase 2 — Preferred Opponent Targeting

§1-10 all measure *how much* a team is fighting for position. Phase 2 asks
a different question: teams near a seed boundary sometimes care not just
whether they make the playoffs, but *which specific opponent* they'd draw
in Round 1 — a one-seed swing can swap in a materially easier or harder
Round 1 opponent than "higher seed = better" suggests.

### Signal tried

`preferred_opponent_delta`: signed win_pct delta, at whichever of
(own_seed−1, own_seed+1) is the larger-magnitude swing, between that
adjacent seed's Round 1 opponent and the current seed's (bracket: seed *s*
plays seed *9−s*). Positive = the available move faces a *stronger*
opponent (current draw already favorable). Negative = a *weaker* opponent
(real incentive to shift one seed). Zero unless holding a direct playoff
seed (1-8) within `preferred_opponent_delta_window_games` of season's end.
Built on `compute_standings_metrics`'s panel (refactored into a shared
`_ranked_standings_panel` helper) — a standings phenomenon, not the
Elo-based machinery §10 uses.

**Known limitations:** single seed-step only (no 2+-seed jumps); adjacent
seed's occupant read off the current snapshot, not resimulated (no full
conference picture); no tiebreakers modeled, same as every signal here.

### CV results (window sweep 15/20/25, isolated on the current base design
— neither §10 signal enabled)

| variant | win-count | test_diff_mae mean delta | test_diff_mae per-fold (fold1→5) | folds improved |
|---|---|---|---|---|
| window=15 | 17/30 (57%) | **−0.0610** | +0.023, −0.089, −0.096, −0.106, −0.037 | 4/5 |
| window=20 | 16/30 (53%) | **−0.0560** | +0.001, −0.048, −0.070, −0.084, −0.079 | 4/5 |
| window=25 | 21/30 (70%) | **−0.0448** | +0.018, −0.053, −0.043, −0.111, −0.035 | 4/5 |

Full test suite: 156/156 passing. `config.yaml` byte-identical to HEAD after
the sweep.

### Verdict

**Passes, and — unlike §10 — robustly across the sweep.** All three windows
show `test_diff_mae` improving in the same 4/5 folds, same small exception
(fold1) each time, comparable magnitude regardless of window — the opposite
of §10's window=10-only, sign-inverting pattern. (`val_diff_mae` is noisier
— means of −0.032/−0.018/−0.035 for 15/20/25 — but the fold-consistency bar
here has always been defined on `test_diff_mae`, precisely because val is
smaller/noisier.) Window=25 has the best win-count (70%), window=15 the
largest raw improvement — a genuine three-way tie on the metric that
matters, no tiebreak needed.

Doesn't resolve §8's structural finding on its own, but is the first signal
in the whole investigation to clear the fold-consistency bar robustly — via
a specific, well-defined mechanism (which team occupies the adjacent seed),
not a general "more motivation information" attempt.

### Adopted

`preferred_opponent_delta_enabled=true`, window=20 (middle of the three
passing values — no CV-based tiebreaker favored 15 or 25). `season_motivation.enabled=true`,
with the new `motivation_score_enabled=false` keeping every non-adopted
§1-10 column out of the shipped feature set.

Real (non-CV) training run on the default split confirmed the direction
outside the CV harness — `preferred_opponent_delta_treatment` in
`outputs/experiments.csv`: test diff_mae 11.54 (vs. 11.592 baseline), test
win_acc 67.8% (vs. 65.96%), val win_acc 66.3% (vs. 65.88%).

## FINAL SUMMARY

11 sections of formula variants; one thing cleared the bar robustly.

**Enabled:** `season_motivation.enabled=true`, `preferred_opponent_delta_enabled=true`,
window=20. Everything else — including the new `motivation_score_enabled`,
added to keep the non-adopted Phase 1 columns out now that `enabled` is
true — stays `false`.

**Parked (implemented, tested, disabled, available for reconsideration):**
- `motivation_score` + clinch (§1-4): 53% win-count, 2/5 folds unfavorable. Not demonstrated.
- Raw-component decomposition (§5): made it worse (20%). Reverted.
- Dual-threshold pressure, max (§6.1, 33%) and weighted-average (§8, 40%): neither cleared the bar. Weighted-average stays wired in via optional `direct_playoff_seed` (currently unset).
- `recent_minutes_trend_score` (§9): 27-37%. Failed.
- `performance_vs_expectation_score` / `opponent_adjusted_form_score` (§10): passed at window=10 only, inverted at 5/15 — very likely a favorable draw against these 5 splits, not a robust effect.

**Enabled:** `preferred_opponent_delta` (§11) — the only signal to pass fold-consistency *and* hold up across a window sweep. Confirmed with a real training run (see §11).

**Honest picture:** every standings/roster-input variant (§1-9) showed the
same `test_diff_mae` degradation (−0.06 to −0.08) regardless of formula.
§10's signals briefly looked like they broke this, but didn't survive the
window sweep — read as inconclusive-trending-negative, not a second win.
`preferred_opponent_delta` is the one signal that both passed initially and
held up under scrutiny.

**Was §8's ceiling resolved? Partially, only by one signal.** §8's finding
was that every *general* "how much is this team motivated by standings"
attempt hits a ceiling regardless of formula. `preferred_opponent_delta`
doesn't contradict that for the general case (`motivation_score` stays
unenabled, still short of the bar) — it succeeds by asking a narrower
question (which specific opponent) that isn't subject to the same ceiling.

**Open questions for human review:**
1. ~~Window value?~~ Resolved: 20 (all three of 15/20/25 pass equally; 20 is the brief's suggested default, no CV tiebreaker favored 15 or 25).
2. Is 5 folds enough validation before shipping anything, given §10's signals also looked convincing before their own window sweep caught the problem?
3. Is the standings/roster-input approach (`motivation_score`, `recent_minutes_trend_score`, §10's two signals) worth revisiting at all, or a dead end for this repo's data — nine sections of formula variants didn't move it; a future attempt would need new data (play-by-play, betting-market effort proxies) or a different mechanism, not another formula.
4. Runtime: `_add_season_motivation_features` takes ~2.5 min on the full training set, driven by `compute_roster_behavior_scores`'s per-`(team,date)` Python loop (unrelated to `preferred_opponent_delta`) — fine for the offline pipeline, worth revisiting if it becomes a bottleneck.

**Known limitations across every signal:** no tiebreakers modeled
(head-to-head, division, conference record); roster-behavior signals are
structurally zero before 2021-10-19 (`player_injuries` coverage start);
pure strategic tanking (playing normally but not trying) has no data source
and is invisible everywhere here; live in-season inference needs a fresh
schedule fetch (`ScheduleLeagueV2`) not built here, since every season
touched during training/validation was already historically complete.
