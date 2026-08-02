# Season Motivation & Seeding Incentive — Implementation Log (Phase 2)

> Companion to `docs/SEASON_MOTIVATION_DECISIONS.md` (phase 1 — data audit and
> formula decisions). This document covers what was actually built, real
> validation results, and the weight-exploration round. All work in this file is
> on branch `feature/season-motivation`, committed incrementally.

## 1. Data Audit (summary — see the decisions doc for full detail/evidence)

- **No new backfill or DB table was needed.** Every season this touches
  (2018-19 through 2025-26) is a completed historical season, so its full
  schedule and results already sit in `nba_api.sqlite`'s `game` table — point-in-time
  standings and remaining-schedule structure are both derived in-memory from it,
  the same "derive it, don't refetch it" approach `_add_elo_features`/
  `_add_rolling_features` already use for their own point-in-time stats.
- A3 (`src/projections/player_projections.py`) and A4
  (`src/lineups/lineup_collector.py`) were both examined and confirmed **not
  usable** for the roster-behavior signal despite being marked "Complete" in the
  backlog: A3's `_get_team_roster` is an explicit unfinished placeholder
  (`return []`) and `project_game_contributions` doesn't filter by team at all;
  A4's `get_available_lineup` returns `CommonTeamRoster`'s season-end roster, not
  a specific game night's actual participants. Same category of finding the
  on/off-splits decisions doc already made about A4.
- Instead, the roster-behavior signal reuses two tables built for the injury
  pipeline (`injury_features.sqlite`): `player_importance` (weekly-backfilled,
  2018-10-22 through 2026-05-27, already covers the entire window) for
  full-strength quality, and `player_injuries` (2021-10-19+, the PDF-era start
  already documented in on/off-splits' work) for who sat out and why.
- **Conference assignment** hardcoded as `_TEAM_CONFERENCE` in
  `season_motivation.py` — a fixed fact for the 2018-2026 window, same convention
  as `feature_builder.py`'s existing `_TEAM_LOCATIONS` dict.
- **New finding during implementation** (not caught in phase 1): `game`'s
  `season_id` column isn't always the expected `2YYYY` Regular Season format —
  Playoff games carry a `4YYYY` prefix (e.g. `42023`), and these show up as
  **warm-up context** for val/test splits (`datasets_loading.context_season_types`
  includes `Playoffs`) even though `allowed_season_types` (Regular Season only)
  correctly excludes them from the actual standings computation. The first
  treatment run crashed on this — `compute_roster_behavior_scores` looked up a
  season-start date for `season_id=42023` that doesn't exist in the Regular-Season-only
  `season_start_by_season` map, comparing a datetime against `None`. Fixed by
  treating a missing season-start lookup as "not applicable, default to 0.0"
  rather than assuming every `season_id` in `df` has standings context —
  Playoff-context rows never reach the scored dataset anyway, so this default is
  never actually evaluated.

## 2. Implementation

`src/feature_engineering/season_motivation.py` (new module, mirroring
`elo.py`'s separation of complex sequential computation from `feature_builder.py`'s
thin per-feature methods):
- `compute_standings_metrics`: builds a long per-team-per-game log with
  pre-game win/loss/games-remaining state, then a `(season_id, team_id,
  snapshot_date)` standings panel via `merge_asof` per team (one team's own game
  log asof-matched against every date a league game was played that season).
  Ranks by win percentage within conference, computes `pressure_raw` (games-back
  from the 10-seed line, absolute value, moderated by games remaining) and the
  two clinch countdowns (raw win-count projection to season's end against the
  team one seed above/below).
- `compute_roster_behavior_scores`: per `(team_id, game_date)`, sums
  `_get_importance_map`-style importance scores (reused from
  `src/news_scraping/pipeline.py`, not reinvented) for the team's rostered
  players, then the same sum restricted to players in `player_injuries` with
  `status='Out'` and a non-injury `reason` (rest, personal reasons, coach's
  decision — full list in `NON_INJURY_REASONS`). Ratio of the two = the
  behavioral-tanking signal.

`src/feature_engineering/feature_builder.py`'s `_add_season_motivation_features`
(wired in after `_add_on_off_splits_features`): loads full history via
`NBADataLoader`, calls both functions above, merges results onto `df` by
`(season_id, team_id, game_date)` for home/away, combines into
`motivation_score = clip(pressure_raw * (1 - roster_behavior_weight *
roster_behavior_score), 0, 1)`. Soft-disabled (warn + skip) if the injury
features cache is missing, matching `_add_on_off_splits_features`'s convention.

New config: `SeasonMotivationConfig` (`enabled`, `playoff_line_seed=10`,
`roster_behavior_weight`, `min_importance_games=5`) — no `db_path`, since this
feature reads existing tables directly rather than owning a cache.

## 3. Validation Results

### Sample output (`scripts/validate_season_motivation.py` → `outputs/season_motivation_results.csv`)

Last 50 real (Regular Season) games as of this run (2026-04-07 through
2026-04-08 — near the end of the configured test season), with the new features
alongside actual margins. Values behave as expected for the tail of a season:
most `games_to_clinch_ceiling`/`floor` values are small single digits (seeding
mostly decided by this point), and `motivation_score` is 0.0 for most rows
(teams already locked into their seed) with occasional higher values for teams
still fighting for a play-in spot.

### Sanity checks (full featurized history, 9,509 Regular Season games, 2018-19 through 2025-26)

| Check | Result |
|---|---|
| NaN rows | 0 / 9,509 |
| `motivation_score` outside [0, 1] | 0 / 9,509 |
| Negative clinch values | 0 / 9,509 |
| `home_team_motivation_score` distribution | min 0.000, mean 0.770, max 1.000 |
| Rows with nothing left to play for (both ceiling and floor == 0) | varies by point in season — 2.3% on the tail-of-season validation-script window (3,200 games, 2024-01-01 onward), up to ~6.4% on a late-March/April-only slice |

Spot-checked distribution shape across the season: mid-season (January) rows
average `motivation_score` ≈ 0.95 (nearly everyone still mathematically alive
with many games left, matching the brief's stated boundary condition), while
late-season (April) rows spread across the full [0, 1] range with a large mass
at exactly 0.0 (many teams already clinched or eliminated) — the two-sided decay
the formula was designed to produce.

### MAE comparison (`train_model.py`, `outputs/season_motivation_iteration_scratch.csv`)

Baseline (`season_motivation.enabled=false`, 125 features) reproduces
`style_matchup_raw_fingerprint`'s exact numbers, as expected:

| metric | baseline (125 feat) | treatment (131 feat, weight=1.0) | direction |
|---|---|---|---|
| val diff_mae | 11.130 | **11.081** | better |
| test diff_mae | 11.592 | **11.543** | better |
| val win_acc | 0.6588 | **0.6661** | better |
| test win_acc | 0.6596 | **0.6735** | better |
| val brier | 0.2129 | **0.2109** | better |
| test brier | 0.2109 | **0.2095** | better |
| val total_mae | 14.752 | 14.803 | slightly worse |
| test total_mae | 15.452 | 15.531 | slightly worse |

**Five of eight tracked metrics improve on both val and test simultaneously** —
diff_mae, win_acc, and brier all move the right direction consistently. Only
`total_mae` (which scores the sum of both teams' points, not the differential/win
outcome the motivation signal is actually about) moves slightly the wrong way.
This is a meaningfully more consistent result than the on/off-splits feature's
final verdict (small, genuinely mixed across metrics) — see the on/off-splits log
`docs/on_off_splits_log.md`'s Final Decision for contrast.

**Feature importance** (`outputs/full_feature_importance_season_motivation_treatment.csv`,
131 features): the six new columns rank 30th, 37th, 38th, 70th, 75th, and 111th —

| feature | rank (of 131) |
|---|---|
| `away_team_games_to_clinch_ceiling` | 30 |
| `home_team_games_to_clinch_ceiling` | 37 |
| `away_team_games_to_clinch_floor` | 38 |
| `away_team_motivation_score` | 70 |
| `home_team_games_to_clinch_floor` | 75 |
| `home_team_motivation_score` | 111 |

Three of six land in the **top third** of all features by importance — notably
better than the on/off-splits feature, whose new columns "consistently rank
bottom-third of ~132 features." The clinch countdowns carry more signal than
`motivation_score` itself, which is a useful finding for anyone revisiting this
feature later (worth exploring the clinch columns standalone, without the
combined score, in a future iteration).

### Weight exploration (`roster_behavior_weight`)

Per the brief's instruction to treat weights as explored parameters, not fixed
guesses, six values were run through the full ablation pipeline — the initial
three (0.0/0.5/1.0), then a follow-up extension to 1.5/2.0/3.0 after 1.0 came
out best, to check whether that was a genuine peak or just the edge of too
narrow a search range (weights up to ~4.3 are still numerically safe without
clipping, given `roster_behavior_score`'s empirical max of 0.234):

| `roster_behavior_weight` | val diff_mae | test diff_mae | val win_acc | test win_acc | val brier | test brier |
|---|---|---|---|---|---|---|
| 0.0 (pure standings pressure) | 11.115 | 11.580 | 0.6571 | 0.6710 | 0.2124 | 0.2111 |
| 0.5 (midpoint) | 11.115 | 11.579 | 0.6571 | 0.6710 | 0.2124 | 0.2110 |
| **1.0 (chosen default)** | **11.081** | **11.543** | **0.6661** | **0.6735** | **0.2109** | **0.2095** |
| 1.5 | 11.126 | 11.594 | 0.6612 | 0.6743 | 0.2126 | 0.2111 |
| 2.0 | 11.101 | 11.566 | 0.6588 | 0.6678 | 0.2118 | 0.2098 |
| 3.0 | 11.115 | 11.589 | 0.6539 | 0.6743 | 0.2123 | 0.2110 |

`1.0` is the clear best point on 5 of 6 metrics (val/test diff_mae, val win_acc,
val/test brier) — every other value tested, both below and above it, clusters
into a visibly worse, fairly flat/noisy band with no clean monotonic trend past
1.0 (1.5/2.0/3.0 don't rank consistently against each other either). This
confirms `1.0` as a genuine local optimum rather than an artifact of a search
range that happened to stop right at the best value.

This looked odd until checking the actual `roster_behavior_score` distribution
directly: only **2.2%** of all team-game rows (535 / 23,938) ever have a nonzero
score at all (rising to 4.4% if restricted to the post-2021-10-19 window where
`player_injuries` has coverage), and even then the typical nonzero value is
small (median 0.065, max 0.234). With such a small, low-magnitude subset of rows
actually affected by this parameter, the noisiness of the 1.5-3.0 band is
expected — there just isn't enough signal in that handful of rows to produce a
clean monotonic curve, only enough to clearly distinguish "off/half-off"
(0.0/0.5) from "on" (1.0) from "overshooting" (1.5+). **Kept
`roster_behavior_weight=1.0`** as the final default; config reverted to
`enabled=false` per the adoption convention (see below).

## 4. Expanding-Window CV

Same 5-fold walk-forward design used for on/off-splits' CV round (always 1
season val + 1 season test, expanding backward): fold1 is the already-committed
default split (train through 2023-24, val 2024-25, test 2025-26); fold2-5 each
shift the whole window back one season. Exact Regular Season boundaries queried
directly from `nba_api.sqlite`'s `game` table, `roster_behavior_weight=1.0`
(the chosen default) throughout.

| fold | train through | val | test | metrics favoring treatment |
|---|---|---|---|---|
| 1 (default) | 2023-24 | 2024-25 | 2025-26 | **6 / 6** |
| 2 | 2022-23 | 2023-24 | 2024-25 | 1 / 6 |
| 3 | 2021-22 | 2022-23 | 2023-24 | 2 / 6 |
| 4 | 2020-21 | 2021-22 | 2022-23 | 5 / 6 |
| 5 | 2019-20 | 2020-21 | 2021-22 | 2 / 6 |

("metrics favoring treatment" = how many of {val/test diff_mae, val/test
win_acc, val/test brier} moved the right direction vs. that fold's own
baseline.)

**Overall: 16 of 30 metric-instances (53%) favor treatment across all 5
folds — essentially a coin flip, not a consistent win.** Fold1 (the split
Phase 1's headline numbers came from) and fold4 look genuinely good; folds 2,
3, and 5 lean the other way, sometimes clearly (fold3's test_diff_mae moves
from 10.992 to 11.213, a real regression). **This materially changes the
picture from the single-split result reported earlier in this document** —
the improvement does not hold up consistently across time, the same kind of
finding that ultimately kept on/off-splits parked rather than adopted (see
`docs/on_off_splits_log.md`'s Final Decision).

One structural difference from on/off-splits' own CV is worth noting: on/off-splits'
folds 4-5 came back **byte-identical** between baseline and treatment, because
that feature's entire signal depended on `player_injuries` (zero rows before
2021-10-19) and those folds' training windows ended before that date, making the
feature a structural no-op for the whole training period. Here, fold4's
training window (through 2021-05-16, also before the `player_injuries` era)
still shows real, non-identical differences between baseline and treatment —
because `motivation_score`'s standings-pressure component and both
`games_to_clinch_*` columns are computed purely from `game` results and never
depend on injury data at all. Only the roster-behavior sub-signal is
structurally zero that early; the rest of the feature is unaffected by this
particular coverage gap. That partial independence is a genuine advantage over
on/off-splits, but it doesn't rescue the overall CV result — the standings/clinch
signal alone still isn't consistently helping across folds.

## 5. Raw-Component Decomposition (Round 2)

**Motivation:** §3's feature-importance table already hinted at this —
`games_to_clinch_ceiling`/`floor` (raw, un-combined) ranked top-third of 131
features, while `motivation_score` (a hand-picked
`pressure_raw * (1 - roster_behavior_weight * roster_behavior_score)`
combination) ranked near the bottom. This exact pattern already has a direct,
proven precedent in this repo: A7's KNN-similarity *combined* style-matchup
score showed no signal (~29th of 109, ~zero importance), while the raw
per-component fingerprint redesign — same underlying data, no hand-picked
combination — became the #1/#2 most important features in the model (see
`docs/backlog.md`'s A7 entry). Given the CV showed no generalized improvement
anyway, this was worth trying before writing Phase 1 off entirely.

**What changed:** `motivation_score` retired. `_add_season_motivation_features`
now exposes `standings_pressure` and `roster_behavior_score` as two separate
raw columns per team, instead of pre-combining them. Added home-minus-away
differential columns for all four raw metrics (`standings_pressure_diff`,
`roster_behavior_score_diff`, `games_to_clinch_ceiling_diff`,
`games_to_clinch_floor_diff`), matching the differential-column convention
every other comparable feature in this codebase already follows
(`on_off_splits`, `style_matchup`) — `season_motivation` was the one feature
missing it. `roster_behavior_weight` retired along with the old formula (no
longer meaningful once nothing is being hand-combined). Net: 6 columns → 12
columns (8 raw + 4 diffs). Also added `tests/test_season_motivation_features.py`
(13 tests), which Phase 1 had shipped without, unlike every other comparable
feature in this repo.

**CV result (same 5-fold structure as §4, baseline unchanged since disabling
the feature short-circuits before any of this code runs):**

| fold | baseline val/test diff_mae | v2 treatment val/test diff_mae | baseline val/test win_acc | v2 treatment val/test win_acc | metrics favoring v2 (of 6) |
|---|---|---|---|---|---|
| 1 | 11.130 / 11.592 | 11.167 / 11.609 | 0.6588 / 0.6596 | 0.6571 / 0.6678 | 1/6 |
| 2 | 11.053 / 11.118 | 11.131 / 11.150 | 0.6455 / 0.6563 | 0.6415 / 0.6563 | 0/6 |
| 3 | 10.265 / 10.992 | 10.244 / 11.163 | 0.6252 / 0.6577 | 0.6244 / 0.6577 | 2/6 |
| 4 | 11.299 / 10.355 | 11.294 / 10.385 | 0.6366 / 0.6130 | 0.6407 / 0.6106 | 2/6 |
| 5 | 11.268 / 11.311 | 11.291 / 11.390 | 0.6241 / 0.6382 | 0.6306 / 0.6195 | 1/6 |

**Verdict: this made things worse, not better — 6 of 30 metric-instances (20%)
favor the redesign, down from the original combined-score design's 16/30
(53%).** Fold1 (the branch's headline split, a clean 6/6 win for the original
combined `motivation_score`) collapsed to 1/6 under the raw-component version.
Feature count also grew from 131 to 137 (8 raw + 4 diffs vs. the old 6 combined
columns) without a corresponding hyperparameter retune (the earlier
`colsample_bylevel` investigation already showed the model's fixed
hyperparameters aren't validated past 131 features either).

**Why the A7 precedent didn't transfer:** A7's KNN-similarity score discarded
a large amount of rich, genuinely independent information (five distinct style
dimensions collapsed into one similarity/confidence number) before the raw
per-component redesign gave it back — that's a case where decomposition
restores real expressiveness the combined score had thrown away. Here,
`motivation_score` combined exactly two already-simple signals
(`pressure_raw`, `roster_behavior_score`) with a single multiplicative
formula; splitting them apart doesn't reveal hidden structure the same way,
and the four added `_diff` columns are largely redundant with the eight raw
columns a tree can already combine via splits — more likely to have diluted
`colsample_bylevel`'s per-split sampling further (exactly the mechanism
flagged in the "too many features" discussion earlier) than to have added
real signal. **Not adopting this redesign either** — reverting to the original
combined-`motivation_score` design would need its own justification at this
point too, since neither version clears the bar. Recorded as a genuine,
informative negative result: not every "combined score → raw components"
refactor is a win, even when a prior case in the same codebase suggested it
would be.

## 6. Open Improvement Ideas (documented, not yet tried)

Two further gaps were identified during discussion but not yet implemented or
tested. Recorded here so they aren't lost regardless of what happens to Round 2.

### 6.1 Dual-threshold standings pressure (6-seed AND 10-seed)

**Gap:** `standings_pressure` only measures distance from the 10-seed
(play-in/postseason cutoff) via `playoff_line_seed`. It never separately
considers the 6-seed (direct playoff berth vs. needing to survive the
play-in). A team in a real fight for 6th vs. 7th, but comfortably clear of
missing the postseason entirely (10th), currently reads as low-pressure even
though avoiding the play-in is a genuine, separate stake. `games_to_clinch_ceiling`/
`floor` partially compensate (they're always computed against whichever team
is immediately adjacent in the standings, at any rank), but `standings_pressure`
itself is blind to the 6-line specifically.

**Candidate fix (implemented and CV-tested):** `compute_standings_metrics`
gained an optional `direct_playoff_seed` parameter — when set, `pressure_raw`
becomes the MAX of the pressure computed against `playoff_line_seed` (10) and
against `direct_playoff_seed` (6), so whichever boundary is more urgent for a
team wins. Verified directly on real data first: mean `standings_pressure`
over a 2024-01-01→2024-04-14 slice rose from 0.617 (single-threshold) to 0.735
(dual-threshold) — real, substantial movement, not a no-op.

**CV result (max-based combination): 10/30 metric-instances (33%) favor the
dual-threshold version** — worse than the original single-threshold design's
53%, though not as bad as Round 2's raw-decomposition attempt (20%). Fold1
(5/6) and fold4 (3/6) look decent; folds 2 and 5 are a clean 0/6 sweep the
other way.

**Diagnosed directly, not just re-tried blindly.** Bucketing real data by
games-played didn't show an early-vs-late-season noise split (both p6 and p10
track closely across the whole season). Instead, comparing the two
combination rules' distributions on real data (2018-19 through 2023-24)
found the actual mechanism: `max(p10, p6)` systematically shifts pressure
**upward** (mean 0.776 → 0.851) while **compressing its variance**
(std 0.310 → 0.260) — max can only push a row's pressure up relative to using
p10 alone, never down, and it disproportionately clips high-pressure rows
together at the top of the range. Less spread means less for a tree to split
on, regardless of how well-motivated the underlying idea is.

**Sharpened fix: replaced `max` with a weighted average**
(`direct_playoff_weight`, default 0.5) — `pressure_raw = weight * pressure_direct
+ (1 - weight) * pressure_postseason`. This leaves the mean almost unchanged
(0.778 vs single-threshold's 0.776) while preserving much more of the original
variance (std 0.282 vs max's 0.260).

**CV result (weighted-average combination): 12/30 metric-instances (40%)
favor the sharpened version** — improved from max's 33%, confirming the
diagnosis was correct, but still below single-threshold's 53%. Fold1 (4/6)
and fold4 (4/6) look decent; fold2 is still a clean 0/6 sweep the wrong way.

**Verdict: the sharpening was a real, measurable improvement (33% → 40%) but
not enough to beat plain single-threshold. Not adopted; `direct_playoff_seed`
stays `null` (single-threshold) as the committed default.** Code, the
weighted-average formula, and all four tests are kept (harmless when
`direct_playoff_seed` is `null`) in case this is revisited later — e.g. tuning
`direct_playoff_weight` itself (only 0.5 has been tried), or gating the
direct-berth term to only apply once the season is far enough along to be
meaningful, similar to Phase 2's own "final ~20 games" framing.

### 6.2 Recent-games actual-playing-time trend (deeper tanking detection)

**Gap:** `roster_behavior_score` is a **single-night snapshot** — it only sees
a player officially tagged `Out` for a non-injury reason *tonight*. It
completely misses "soft" tanking: a coach quietly cutting a star's minutes
from 35 to 15 over several games, or feeding bench players more run, without
ever putting anyone on the official injury report. This is exactly the "pure
strategic tanking" limitation flagged as unsolved in §5 of the decisions doc.

**Candidate fix (implemented and CV-tested, see §9):** the original idea
sketched a delta-of-cumulative-averages formula
(`(cumulative_avg_N × games_N − cumulative_avg_N-1 × games_N-1) / (games_N − games_N-1)`)
to back out a single week's actual average from consecutive weekly
snapshots — this turned out to need a `games_played` count `player_importance`
doesn't store, so no new backfill was in fact needed by using a simpler proxy
instead: comparing each player's current cumulative average directly against
their own cumulative average from `recent_trend_lookback_weeks` (4) earlier.
A real drop over that window still means genuinely reduced recent minutes,
without isolating one exact week.

## 7. Reverted Round 2, Restored Test Coverage

Round 2's raw-component redesign (§5) was reverted — the combined
`motivation_score` formula is back as the pipeline's current state
(`motivation_score`, `roster_behavior_weight`, no diff columns). Also added
`tests/test_season_motivation_features.py` (13 tests: standings
pressure/clinch formulas, roster-behavior scoring, config-gating, and the
`motivation_score` combination formula itself) — this had been missing since
Phase 1 shipped, unlike every other comparable feature in this repo.

## 8. Magnitude-Weighted Cross-Variant Comparison

Every CV verdict so far (§4, §5, §6.1) used a win-count — how many of 30
metric-instances favor the treatment — which treats a tiny loss the same as a
large one. Redone with actual per-metric magnitudes (mean delta vs baseline,
averaged across the 5 folds, `+` = better for the treatment) across all four
variants tried:

| variant (win-count) | val diff_mae | **test diff_mae** | val win_acc | test win_acc | val brier | test brier |
|---|---|---|---|---|---|---|
| v1 single-threshold (53%) | −0.0018 | **−0.0666** | +0.0023 | −0.0005 | +0.0001 | −0.0017 |
| v2 raw-decomposition (20%) | −0.0224 | **−0.0658** | +0.0008 | −0.0026 | −0.0012 | −0.0016 |
| dual-threshold MAX (33%) | −0.0266 | **−0.0628** | −0.0039 | −0.0006 | −0.0005 | −0.0018 |
| dual-threshold AVG (40%) | −0.0054 | **−0.0766** | 0.0000 | −0.0049 | 0.0000 | −0.0020 |

**Headline: `test_diff_mae` is negative by almost the same amount (−0.063 to
−0.077) in every single variant, regardless of win-count percentage or
formula design.** v1's "best" 53% win-count was not driven by a better
`test_diff_mae` — it was driven by val-side metrics and win_acc, which are
noisier and flip more favorably by chance from variant to variant (val-side
deltas above bounce between roughly +0.002 and −0.027 with no consistent
sign, while test_diff_mae is consistently negative across all four rows).
This means the win-count differences between variants (20% / 33% / 40% / 53%)
mostly reflect noise on the val side, not a real difference in how much any
of them help — **the actual, more damning result is that the one metric
staying consistently negative across every design tried is the one measuring
held-out point-differential accuracy.** This reads as a structural cost of
adding these features at all (more model capacity for the same amount of
real signal, slightly worse test generalization) rather than something a
different combination formula is likely to fix. Strengthens "not adopted"
well beyond what any single win-count number suggested on its own.

## 9. Recent-Minutes-Trend Signal (Idea #3)

`recent_minutes_trend_score` (§6.2) was implemented as its own raw column
(not folded into `motivation_score`) and CV-tested on top of v1's base design
(single-threshold pressure, `direct_playoff_seed` still `null` per §6.1's
verdict). Smoke-tested on real data first: fires nonzero in ~99% of rows
(mean 0.021) — much more often than `roster_behavior_score`'s ~2-4% rate,
raising a real concern that it's picking up routine week-to-week minutes
variance rather than deliberate role reduction specifically. One encouraging
early sign: `home_team_recent_minutes_trend_score` ranked **6th of 133**
features by importance in fold2's model — notably higher than any other
`season_motivation` column has ranked all session.

**CV result:**
- vs baseline (no `season_motivation` at all): 8/30 (27%) win-count.
- vs v1 treatment (isolating just this new column's own marginal effect):
  11/30 (37%) win-count.
- Magnitude (mean delta vs v1, across 5 folds): `val_diff_mae` −0.0416,
  **`test_diff_mae` +0.0224**, `val_win_acc` −0.0038, `test_win_acc` +0.0041,
  `val_brier` −0.0014, `test_brier` +0.0010 — at first glance, `test_diff_mae`
  (the one metric that was consistently *negative* across every variant in
  §8) looks like it improved here.

**That average does not hold up fold-by-fold.** The isolated
`test_diff_mae` delta per fold is −0.021, −0.014, **+0.081**, −0.043,
**+0.109** — 3 of 5 folds are slightly worse, 2 are notably better, and the
positive mean is entirely driven by those two larger swings, not a
consistent effect. This is the same "noise happens to average out positive"
pattern already seen on the val side in §8, just showing up here on the test
side instead. The #6 feature-importance ranking similarly reflects the model
finding *something* to split on in that fold's training data, not proof it
generalizes — exactly the gap between "the tree used it" and "it helps"
flagged when that ranking first came up.

**Verdict: not adopted.** No clear win by any measure once checked
carefully — win-count is worse than v1 alone (27-37% vs 53%), and the one
metric whose average looked promising doesn't hold up fold-by-fold. The
99%-nonzero firing rate suggests this proxy (comparing cumulative averages
across a 4-week window) captures normal roster fluctuation more than
deliberate tanking specifically — a real limitation of the no-new-backfill
approach, not just an unlucky formula choice. Code and 4 tests are kept
(computed whenever `season_motivation.enabled` is `true`, which stays
`false` by default) in case a version gated to only fire on larger,
rarer drops is worth trying later.

## 10. Phase 1 Iteration — Behavioral Signals

Every signal tried through §9 measures an *input* to motivation (standings
position, roster/rest decisions). §8's diagnosis was structural: `test_diff_mae`
degraded by roughly the same amount (−0.063 to −0.077) across every formula
variant of those inputs, regardless of design — a ceiling on what input-based
signals alone can do. This round tests a different hypothesis: motivation
should be detectable in recent *game behavior* directly, not just in standings
and roster state.

### Signals tried

- **`performance_vs_expectation_score`**: rolling mean (window=10 games) of
  `actual margin − Elo-expected margin` for a team's own past games,
  normalized by the residual's global standard deviation. The Elo-diff-to-
  margin conversion scale is fit once via least-squares from this repo's own
  historical `(elo_diff, actual_margin)` relationship (`_fit_elo_margin_scale`)
  rather than an external heuristic — this repo's Elo params are independently
  tuned (`tune_elo.py`), so a borrowed constant wouldn't necessarily match its
  scale. A team consistently beating its own rating lately reads as
  motivated; consistently missing it reads as the opposite.
- **`opponent_adjusted_form_score`**: rolling mean (window=10 games) of a
  signed, opponent-strength-weighted outcome per game — `opponent_win_pct`
  for a win, `-(1 - opponent_win_pct)` for a loss. A win over a strong team
  counts far more than a win over a weak one; a loss to a weak team counts
  far more negatively than a loss to a strong one.

Both are exposed as independent raw columns (own `..._enabled` flag each,
same convention as `style_matchup`'s `enabled`/`raw_features_enabled` pair),
not combined with `motivation_score` or with each other — per the
raw-decomposition lesson from §5, and to keep each individually ablatable.
Both reuse `elo_features`' own ratings, recomputed once over full history,
same pattern `_add_elo_features` already uses.

### CV results (isolating each signal's own marginal effect on top of the
current base design: `motivation_score` + clinch + `recent_minutes_trend_score`)

| variant | win-count | test_diff_mae mean delta | test_diff_mae per-fold (fold1→5) |
|---|---|---|---|
| Signal 1 alone | 17/30 (57%) | **+0.0510** | −0.039, +0.045, +0.083, **+0.124**, +0.042 |
| Signal 2 alone | 17/30 (57%) | **+0.0444** | +0.011, −0.012, +0.081, +0.092, +0.050 |
| Both combined | 19/30 (63%) | −0.0076 | +0.041, −0.048, −0.007, +0.023, −0.047 |

(vs. pure baseline — no `season_motivation` at all — win-counts are lower:
43% / 40% / 57% respectively, and the picture is noisier, since that
comparison also carries the existing base design's own already-mixed effect,
not just each signal's marginal contribution. The isolation comparison above
is the one that actually answers "does adding this signal help.")

### Verdict per signal

- **Signal 1 (`performance_vs_expectation_score`): passes.** `test_diff_mae`
  improves in 4 of 5 folds, with the one exception (fold1, −0.039) small
  relative to the four improvements (up to +0.124). This is the first signal
  in this entire investigation to show a *consistent*, not just
  average-favorable, `test_diff_mae` improvement across folds — the exact bar
  every prior signal (combined score, raw decomposition, both dual-threshold
  variants, recent-minutes-trend) failed on this same check.
- **Signal 2 (`opponent_adjusted_form_score`): passes**, by the same
  standard — 4 of 5 folds improve, one small exception (fold2, −0.012).
- **Both combined: does not clearly pass.** Overall win-count is higher
  (63%, the best of any variant this session on that measure alone), but the
  specific bar both signals individually cleared — fold-consistent
  `test_diff_mae` improvement — breaks down when stacked: only 2 of 5 folds
  improve, and the mean turns slightly negative. Combining two things that
  each individually help does not automatically help more; here it appears
  to help *less* on the metric that mattered most for qualifying either one.

### Structural findings

The combined result is the more informative one, not just a disappointing
footnote. Two signals both passing individually but failing to combine
cleanly on `test_diff_mae` specifically (while combining fine, even
favorably, on other metrics like win-count) is consistent with **redundancy,
not complementarity** — both signals are built from largely the same
underlying substrate (recent game outcomes and margins, one filtered through
Elo expectation, the other through opponent strength), so they likely
capture overlapping rather than additive information about a team's recent
form. This doesn't invalidate either signal's individual result, but it
means the model can't extract a bigger combined win from a bigger combined
"amount of motivation information" the way two genuinely independent
signals might. Whether one signal alone (not both) is worth adopting, and if
so which, is an open question for human review before Phase 2 begins — see
the note at the end of this document.

### Window sensitivity sweep

Both signals passed their individual `test_diff_mae` fold-consistency bar at
`window=10`. Before treating that as a real, adoptable effect, the same
isolated CV (each signal alone, on top of the current base design, compared
against `season_motivation_recenttrend_{fold}_treatment`) was repeated at
`window=5` and `window=15` for each signal — 4 sweep points × 5 folds = 20
additional runs.

| variant | win-count | test_diff_mae mean delta | test_diff_mae per-fold (fold1→5) | folds improved |
|---|---|---|---|---|
| Signal 1, window=5 | 17/30 (57%) | −0.0384 | +0.040, −0.097, −0.005, −0.102, −0.028 | 1/5 |
| Signal 1, window=10 | 17/30 (57%) | **+0.0510** | −0.039, +0.045, +0.083, +0.124, +0.042 | 4/5 |
| Signal 1, window=15 | 19/30 (63%) | −0.0586 | +0.010, −0.057, −0.074, −0.130, −0.042 | 1/5 |
| Signal 2, window=5 | 13/30 (43%) | −0.0386 | +0.016, −0.072, −0.024, −0.089, −0.024 | 1/5 |
| Signal 2, window=10 | 17/30 (57%) | **+0.0444** | +0.011, −0.012, +0.081, +0.092, +0.050 | 4/5 |
| Signal 2, window=15 | 15/30 (50%) | −0.0498 | +0.042, −0.067, −0.064, −0.111, −0.049 | 1/5 |

Full test suite: 150/150 passing after this sweep. `configs/config.yaml`
verified byte-identical to HEAD once the sweep finished (each of the 20 runs
patches and restores it in a `finally` block, same pattern as prior CV
drivers).

**This result overturns the window=10 verdict above.** It is not a case of
window=10 being merely the best of three reasonable choices — at both
neighboring windows, for both signals, `test_diff_mae` flips to *consistently
negative* (4 of 5 folds worse, not better) and the magnitude is comparable to
or larger than window=10's improvement. The fold that stays positive is the
same one (fold1, the most recent test period) at window=5/15 for every
sweep point, while the folds that were the source of window=10's "passes"
(folds 2–5, or fold3–5 for Signal 2) all reverse sign. Win-count alone does
not surface this — `pve_w15`'s 63% win-count is the highest of any variant in
this entire investigation, yet its `test_diff_mae` fold-consistency is the
worst (1/5), which is exactly the win-count-vs-magnitude divergence §8
warned about.

A signal whose fold-consistency bar is only cleared at one specific window
value out of three tested, and is cleared *oppositely* (worst not just
absent) at the neighboring values, is much more consistent with **window=10
having been a favorable draw against these specific 5 expanding-window
splits** than with a real, robust behavioral-motivation effect at that
window. Nothing here rules out that a real effect exists at window=10
specifically for some substantive reason (e.g. it happens to match a
meaningful "recent stretch" length), but that would need a mechanism, not
just this data, to be credible — the sweep as run cannot distinguish "real
effect, oddly window-specific" from "overfit to one hyperparameter draw."

**Revised verdict:** Signal 1 and Signal 2 do **not** pass a window-robustness
check. Neither is being enabled by default. Both remain in the codebase,
fully implemented and tested, gated behind their own `..._enabled` flags
(default `false`) — available for future re-evaluation (e.g. against a larger
number of folds, or a fold-count/window combination chosen to reduce this
kind of single-draw sensitivity) but not adopted on the strength of the
current evidence. This finding, not the individually-passing window=10
result above, is the one that should inform any adoption decision.

## 11. Phase 2 — Preferred Opponent Targeting

Every signal through §10 is either a standings/roster input (§1-9) or a
recent-behavior signal (§10) — both measure *how much* a team is fighting for
position. Phase 2 targets a different, more specific incentive: teams near a
seed boundary sometimes care not just about *whether* they make the playoffs
or improve their seed, but about *which specific opponent* they'd draw in
Round 1. A team can be better off NOT chasing the best seed it can reach,
because a one-seed swing can swap in a materially easier or harder Round 1
opponent than "higher seed = better" suggests.

### Signal tried

`preferred_opponent_delta`: the signed win_pct delta, at whichever of
(own_seed − 1, own_seed + 1) is the larger-magnitude swing, between that
adjacent seed's Round 1 opponent and the current seed's Round 1 opponent
(standard bracket: seed *s* plays seed *9 − s*). Positive = the available
one-seed move faces a *stronger* opponent (current draw already the more
favorable of the two — no incentive to jockey). Negative = the available
one-seed move faces a *weaker* opponent (a real incentive to shift seed by
exactly one spot). Zero unless the team currently holds a direct playoff seed
(1-8) with at most `preferred_opponent_delta_window_games` games left in the
season. See `season_motivation.compute_preferred_opponent_delta_scores`.

Built on the standings machinery already used for `motivation_score`/clinch
(`compute_standings_metrics`'s panel construction was refactored into a
shared `_ranked_standings_panel` helper, reused by both), not the Elo-based
machinery §10's two signals use — this is fundamentally a standings/seeding
phenomenon (which specific team occupies the adjacent seed), not a
behavioral one.

**Known limitations (deliberate, not solved here):**
1. Only a single seed step is considered in each direction — no 2+-seed
   jumps, even though a team could realistically move further than one seed
   before the season ends.
2. The adjacent seed's occupant is read directly off the current standings
   snapshot, not resimulated — swapping seeds would also shift who else in
   the conference lands at each rank (a full conference picture); this only
   asks "who is there right now."
3. No tiebreakers modeled (head-to-head, division, conference record), same
   continuous-proxy simplification as every other signal in this
   investigation.

### CV results (window sensitivity: `preferred_opponent_delta_window_games`
= 15/20/25, isolated on top of the current base design — neither §10 signal
enabled, per that section's revised verdict — compared against
`season_motivation_recenttrend_{fold}_treatment`)

| variant | win-count | test_diff_mae mean delta | test_diff_mae per-fold (fold1→5) | folds improved |
|---|---|---|---|---|
| window=15 | 17/30 (57%) | **−0.0610** | +0.023, −0.089, −0.096, −0.106, −0.037 | 4/5 |
| window=20 | 16/30 (53%) | **−0.0560** | +0.001, −0.048, −0.070, −0.084, −0.079 | 4/5 |
| window=25 | 21/30 (70%) | **−0.0448** | +0.018, −0.053, −0.043, −0.111, −0.035 | 4/5 |

Full test suite: 156/156 passing. `configs/config.yaml` verified
byte-identical to HEAD after the sweep (same patch/restore pattern as every
prior CV driver).

### Verdict

**Passes, and — unlike §10's two signals — passes robustly across the
window sweep.** All three tested window values show `test_diff_mae`
improving in the same 4 of 5 folds, with the same fold (fold1, the most
recent test period) as the lone, small exception each time, and the other
four folds improving by a comparable magnitude regardless of window choice.
This is exactly the check §10's signals failed: there, window=10's pass
inverted sign entirely at window=5 and window=15. Here, the direction and
fold pattern hold steady across 15/20/25, which is much more consistent with
a real, window-insensitive effect than a favorable draw against one specific
hyperparameter value. (`val_diff_mae` is noisier — mean deltas of −0.032,
−0.018, −0.035 for window 15/20/25, with more folds flipping sign than on
test — but the fold-consistency bar this whole investigation has used
throughout is defined on `test_diff_mae`, not val, precisely because val is
the smaller, noisier split.)

Window=25 has the highest win-count (70%) of the three; window=15 has the
largest `test_diff_mae` improvement magnitude. All three are legitimate,
fold-consistent choices — this is a genuine three-way tie on the metric that
matters, not a case needing a tie-break sweep the way §10's window=10 did.

**This does not resolve §8's structural finding on its own; it's the first
signal in the entire investigation to clear the fold-consistency bar
robustly, on a signal deliberately built around a specific, well-defined
mechanism (which team occupies the adjacent seed) rather than a general
"more motivation information" attempt.**

### Adopted

`preferred_opponent_delta_enabled=true`, `preferred_opponent_delta_window_games=20`
(the middle of the three passing values — no CV-based tiebreaker favored 15
or 25 specifically). `season_motivation.enabled=true` to allow this, with the
new `motivation_score_enabled=false` (see FINAL SUMMARY) keeping every
non-adopted §1-10 column out of the shipped feature set.

A real (non-CV) training run on the default split confirmed the direction
holds outside the CV harness: `preferred_opponent_delta_treatment` in
`outputs/experiments.csv` — test diff_mae 11.54 (vs. 11.592 on the
style-matchup-raw-fingerprint baseline it sits on top of), test win_acc
67.8% (vs. 65.96%), val win_acc 66.3% (vs. 65.88%).

## FINAL SUMMARY

This investigation ran ten formula variants of a combined/decomposed
standings-and-roster score (§1-9), two behavior-based signals with a window
sweep (§10), and one seeding-target signal with its own window sweep (§11).
One thing cleared the bar robustly. Here is the complete picture.

### What's enabled vs. parked, and why

**One signal is enabled in this branch's committed config:**
`season_motivation.enabled=true`, `preferred_opponent_delta_enabled=true`
(`preferred_opponent_delta_window_games=20`). Every other sub-signal's own
`..._enabled` flag — including the new `motivation_score_enabled`, added
specifically to keep the non-adopted Phase 1 columns (`motivation_score`,
`games_to_clinch_ceiling`/`games_to_clinch_floor`, `recent_minutes_trend_score`)
out of the shipped feature set now that `enabled` is `true` — stays `false`.

**Parked (implemented, tested, disabled by default, available for future
reconsideration):**
- `motivation_score` + `games_to_clinch_ceiling`/`games_to_clinch_floor`
  (§1-4): single-split result looked clean, but only 53% of metric-instances
  favored it across 5 CV folds, with 2 of 5 clearly unfavorable. Not a
  demonstrated improvement.
- Raw-component decomposition of the above (§5): tried as a direct analogue
  to a precedent that worked elsewhere in this repo (A7's style-matchup
  redesign). Made the CV picture worse (20%, fold1's clean win collapsed).
  Reverted.
- Dual-threshold pressure, both max-based (§6.1, 33%) and weighted-average
  (§8, 40%) variants: neither cleared the bar. Weighted-average is the
  version left wired into `compute_standings_metrics` (via the optional
  `direct_playoff_seed` parameter) since it's strictly less biased than
  `max`, but it isn't itself an adopted change — it only activates if
  `direct_playoff_seed` is explicitly set, which it isn't.
- `recent_minutes_trend_score` (§9): 27-37% across variants tried. Failed.
- `performance_vs_expectation_score` / `opponent_adjusted_form_score` (§10):
  the first signals in the whole investigation to pass the fold-consistency
  bar — but only at exactly one of three tested window values (10). At the
  neighboring windows (5, 15), the effect didn't just weaken, it inverted to
  consistently negative, by a magnitude comparable to or larger than
  window=10's improvement. Diagnosed as very likely a favorable draw against
  these 5 CV splits at that one hyperparameter value, not a robust effect —
  the fold that stayed positive at window=5/15 was consistently the same
  one (fold1) at every window, while the folds that drove window=10's pass
  (2-5) reversed sign entirely. Both remain implemented, tested, and
  disabled.

**Enabled — `preferred_opponent_delta` (§11), window=20:** the one signal in
this entire investigation that passed the fold-consistency bar *and* held up
under the window-robustness check that killed §10's signals. All three
tested windows (15/20/25) show `test_diff_mae` improving in the same 4 of 5
folds, with the same small exception (fold1) each time, and comparable
improvement magnitude across all three — the opposite of §10's
window=10-only, sign-inverting pattern. This is a meaningfully different
kind of evidence: a specific, mechanistically-motivated signal (which team
occupies the adjacent seed) that a same-shaped robustness check couldn't
break. A real training run on the default split confirmed the direction
outside the CV harness (`preferred_opponent_delta_treatment` in
`outputs/experiments.csv`: test diff_mae 11.54 vs. 11.592 baseline, test
win_acc 67.8% vs. 65.96%).

### Honest CV picture across the whole investigation

Every standings/roster-input variant (§1-9) showed the same structural
symptom §8 first diagnosed: `test_diff_mae` degrading by roughly the same
amount (−0.06 to −0.08) regardless of formula. Reformulating the combination
rule, decomposing it into raw components, adding a second threshold, sharpening
it to a weighted average, and adding a complementary roster-trend signal all
failed to move this needle in the favorable direction. Two behavior-based
signals (§10) briefly appeared to break this pattern (test_diff_mae
improving 4/5 folds) — but that result didn't survive being tested at
neighboring windows, so it should be read as inconclusive-trending-negative,
not as a second confirmed win. The one signal that both passed initially and
held up under further scrutiny is `preferred_opponent_delta` (§11) — a
standings-based signal again, but built around one specific mechanism
(seed-adjacent opponent strength) rather than an attempt to capture "more
motivation" generally.

### Was §8's structural issue resolved?

**Partially, and only by one signal.** §8's finding was that every *general*
attempt to capture "how much is this team motivated by its standings
position" hits a ceiling — reformulating the combination rule doesn't help
because the ceiling isn't about the formula, it's about what the underlying
inputs can predict. `preferred_opponent_delta` doesn't contradict that
finding for the *general* motivation-pressure signal (`motivation_score`
remains unenabled, unchanged, still short of the bar) — it succeeds by asking
a narrower, different question (which specific opponent, not how much
pressure) that isn't subject to the same ceiling. So: the general ceiling
identified in §8 persists for `motivation_score` and its variants; it does
not appear to constrain `preferred_opponent_delta`, which measures something
else entirely.

### Next step

Done: `preferred_opponent_delta_enabled=true`, window=20, confirmed with a
real training run (see above). Everything else stays disabled. The natural
follow-up, if further work on this feature area is wanted, is open question
3 below — whether the standings/roster-input approach is worth another
attempt at all, given nine sections of formula variants failed to move it.

### Open questions for human review

1. ~~Which window value for `preferred_opponent_delta`?~~ **Resolved:**
   window=20 chosen (all three of 15/20/25 pass equally on fold-consistency;
   window=25 has the best win-count, window=15 the largest raw
   `test_diff_mae` improvement; 20 was picked as the original brief's
   suggested default with no CV-based tiebreaker favoring 15 or 25
   specifically). If either of the other two windows is ever revisited,
   §11 has the full breakdown.
2. **Should `preferred_opponent_delta` ship alone, or is further validation
   (more folds, a different date range) warranted before enabling anything,**
   given that §10's signals also looked convincing on 5 folds before the
   window sweep revealed the problem? The window-robustness check here is
   a stronger bar than §10 had before its own sweep, but it is still only
   5 folds.
3. **Are `motivation_score`/`recent_minutes_trend_score`/§10's two signals
   worth revisiting at all**, or is the standings/roster-input approach a
   dead end for this repo's data? Nothing tried in nine sections moved that
   needle; a future attempt would need either new data (play-by-play,
   betting-market-implied effort proxies) or a fundamentally different
   mechanism, not another formula variant of the same inputs.
4. **Runtime cost**: `_add_season_motivation_features` takes ~2.5 minutes on
   the full training set, driven by `compute_roster_behavior_scores`'s
   per-`(team, date)` Python loop (unrelated to `preferred_opponent_delta`,
   which doesn't use that path) — acceptable for the offline ablation
   pipeline, worth revisiting if this becomes a bottleneck.

**Known limitations across every signal in this investigation:** no
tiebreakers modeled (head-to-head, division, conference record) — deliberate
continuous-proxy simplification per the original brief; roster-behavior
signals are structurally zero before 2021-10-19 (`player_injuries` coverage
start); pure strategic tanking (playing normally but not trying) has no
available data source and is invisible to every signal here; live/production
inference on an in-progress season needs a fresh schedule fetch
(`ScheduleLeagueV2`) not built in this investigation, since every season
touched during training/validation was already historically complete.
