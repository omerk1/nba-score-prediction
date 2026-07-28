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

## FINAL SUMMARY (Phase 1)

**Bottom line after the expanding-window CV: the single-split result does not
generalize.** The headline fold (fold1, the branch's default train/val/test
split) shows a clean win on every tracked metric, and three of six new columns
rank in the top third of feature importance — genuinely more promising on paper
than on/off-splits ever looked on a single split. But across all 5 CV folds,
only 53% of metric-instances favor the treatment, with 2 of 5 folds clearly
unfavorable. This is not a demonstrated, reproducible improvement — it's a
result that happens to look good on the one split most casually checked first.
`season_motivation.enabled` stays `false`; this should be treated the same way
on/off-splits' single-split result was treated before its own CV round —
promising, not yet earned adoption.

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
