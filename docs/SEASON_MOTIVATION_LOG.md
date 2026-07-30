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

**Candidate fix:** compute `GB_from_line` against whichever of {6th, 10th} is
nearer to the team's current rank, not just the 10th unconditionally. Needs
its own exploration round (which of the two lines "wins" when a team is
roughly equidistant from both is itself a design choice worth testing a couple
of variants on, same as `roster_behavior_weight`'s grid search).

### 6.2 Recent-games actual-playing-time trend (deeper tanking detection)

**Gap:** `roster_behavior_score` is a **single-night snapshot** — it only sees
a player officially tagged `Out` for a non-injury reason *tonight*. It
completely misses "soft" tanking: a coach quietly cutting a star's minutes
from 35 to 15 over several games, or feeding bench players more run, without
ever putting anyone on the official injury report. This is exactly the "pure
strategic tanking" limitation flagged as unsolved in §5 of the decisions doc.

**Candidate fix:** `player_importance` already stores **weekly cumulative**
per-player minutes — no new backfill needed. A recent-week's *actual* (non-
cumulative) average can be backed out from the delta between two consecutive
weekly snapshots: `(cumulative_avg_N × games_N − cumulative_avg_N-1 × games_N-1)
/ (games_N − games_N-1)`. Comparing that recent-week actual-minutes
distribution against full-strength quality (rather than relying solely on
tonight's official Out list) would catch gradual, undeclared minutes
reductions that the current point-in-time signal cannot. This is a genuinely
different, complementary signal to `roster_behavior_score`, not a tuning
tweak — worth its own dedicated implementation and ablation round.

## 7. Reverted Round 2, Restored Test Coverage

Round 2's raw-component redesign (§5) was reverted — the combined
`motivation_score` formula is back as the pipeline's current state
(`motivation_score`, `roster_behavior_weight`, no diff columns). Also added
`tests/test_season_motivation_features.py` (13 tests: standings
pressure/clinch formulas, roster-behavior scoring, config-gating, and the
`motivation_score` combination formula itself) — this had been missing since
Phase 1 shipped, unlike every other comparable feature in this repo.

## FINAL SUMMARY (Phase 1)

**Bottom line after the expanding-window CV: the single-split result does not
generalize, and a raw-component redesign made the CV picture worse, not
better.** The headline fold (fold1, the branch's default train/val/test split)
shows a clean win on every tracked metric under the original combined-score
design, and three of six new columns rank in the top third of feature
importance — genuinely more promising on paper than on/off-splits ever looked
on a single split. But across all 5 CV folds, only 53% of metric-instances
favor the original design, with 2 of 5 folds clearly unfavorable. §5's
follow-up attempt — decomposing `motivation_score` into raw
`standings_pressure`/`roster_behavior_score` columns, motivated by a directly
analogous precedent (A7's style-matchup redesign) that worked well elsewhere
in this repo — dropped the result further, to 20% (6/30), and fold1's clean
6/6 win collapsed to 1/6. Neither version is a demonstrated, reproducible
improvement. `season_motivation.enabled` stays `false`, and so does its
internal design choice (combined score vs. raw components) — both were tried,
neither cleared the bar.

**What was added:** `src/feature_engineering/season_motivation.py` (new module)
plus `feature_builder.py`'s `_add_season_motivation_features`, adding 6 columns
(`{home,away}_team_motivation_score`, `{home,away}_team_games_to_clinch_ceiling`,
`{home,away}_team_games_to_clinch_floor`). Gated by
`config.season_motivation.enabled` (currently `false`, pending the same
conclusive-ablation bar every other not-yet-adopted feature in this repo is held
to), soft-disabled if the injury features cache is missing.

**No new backfill or DB table was needed** — every ingredient (standings,
schedule, roster quality, sit-out reasons) already existed in already-complete
tables (`nba_api.sqlite.game`, `injury_features.sqlite.player_importance`/
`player_injuries`). This is a meaningfully cheaper story than on/off-splits,
which needed a multi-thousand-call live API backfill.

**Single-split result looked like a genuine improvement — the 5-fold CV shows
it doesn't hold up.** On the default split, five of eight tracked metrics
(diff_mae, win_acc, brier — both val and test) moved the right direction
simultaneously, and three of six new columns ranked in the top third of feature
importance. But §4's expanding-window CV puts this in context: only 53% of
metric-instances favor the treatment across all 5 folds, with 2 of 5 clearly
unfavorable. The `roster_behavior_weight` parameter itself was explored
properly (0.0/0.5/1.0/1.5/2.0/3.0), with `1.0` a genuine, confirmed local
optimum among those tested — the parameter tuning is sound, the underlying
feature's overall value is not confirmed.

**Known limitations (documented, not solved here):**
1. No tiebreakers modeled (head-to-head, division, conference record) — a
   deliberate continuous-proxy simplification per the brief.
2. Roster-behavior signal is structurally zero before 2021-10-19
   (`player_injuries` coverage start).
3. Pure strategic tanking (playing normally but not trying) has no available
   data source and is invisible to this signal.
4. Live/production inference on an in-progress season needs a fresh schedule
   fetch (`ScheduleLeagueV2`) not built here — every season this touches during
   training/validation is already historically complete.
5. Runtime cost: `_add_season_motivation_features` takes ~2.5 minutes on the
   full training set (9,509 games), driven by `compute_roster_behavior_scores`'s
   per-`(team, date)` Python loop. Acceptable for the offline ablation pipeline,
   worth revisiting if this becomes a bottleneck in more frequent iteration.

**Open question before Phase 2 / merge:** given the CV's mixed result, should
Phase 2 (`preferred_opponent_delta`) still be built on top of a Phase 1 that
hasn't cleared the bar on its own? Recommendation: yes, still worth building —
Phase 2 is a genuinely different signal (seeding-incentive targeting near the
end of the season) that could independently carry its own weight even if
Phase 1's standings-pressure/roster-behavior combination doesn't, and the brief
called for completing both phases before an adoption decision. But the final
adoption call should now explicitly weigh Phase 1's CV result, not just treat
it as a foregone "already proven" prerequisite — `season_motivation.enabled`
stays `false` regardless of Phase 2's outcome unless a future ablation round
finds something the current CV didn't.
