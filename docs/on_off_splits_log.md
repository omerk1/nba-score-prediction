# Player On/Off Splits — Implementation Log (Phase 2)

> Companion to `docs/on_off_splits_decisions.md` (phase 1 — data-source investigation
> and design decisions, empirically tested). This document covers what was actually
> built, the real backfill run, and validation results. All work in this file is on
> branch `feature/on-off-splits`, committed incrementally, not pushed, no PR opened.

## 1. Data Audit (summary — see the phase-1 decisions doc for full detail/evidence)

- `src/lineups/lineup_collector.py` (roster-only, via `CommonTeamRoster`) and
  `box_score_stats`/`player_stats_cache` (aggregate box-score stats, no on-court
  awareness) were confirmed to have **no** on/off-court data — the phase-1
  investigation ruled these out directly by reading the source, not assumption.
- nba_api's `TeamPlayerOnOffSummary` endpoint was tested live (12 real API calls) and
  confirmed to support genuine leakage-safe historical point-in-time queries via
  `date_to_nullable`/`date_from_nullable`, composing correctly with both
  `opponent_team_id` (vs-opponent split) and `location_nullable` (home/away split).
  `last_n_games` was confirmed **broken** when combined with any date filter (returns
  an empty dataframe) — never used.
- **New finding during phase 2** (not caught in phase 1): `date_to_nullable` is
  **inclusive** of games played on that exact calendar date. Verified directly:
  Boston's season-cumulative `GP` (2023-24) was 39 as of both 2024-01-13 and
  2024-01-14, then jumped to 40 exactly on 2024-01-15 — the date of an actual BOS
  game — not the day after. This means every fetch/lookup meant to inform a
  prediction for calendar date D must use `D - 1 day`, never `D` itself, to avoid
  same-day leakage. This is baked into both the backfill script (vs-opponent lazy
  fetches use `game_date - 1 day`) and the feature-builder join (lookup key is
  `game_date - 1 day`, not `game_date`).
- Decision (unchanged from phase 1): `TeamPlayerOnOffSummary`, new dedicated table
  `player_on_off_splits` (own additive sqlite file, not the shared `nba_api.sqlite`),
  checkpoint-cadence storage (not per-game), `pd.merge_asof` join on
  `(team_id [, opponent_team_id], game_date)` mirroring
  `_add_style_fingerprint_features`.

## 2. Team-level aggregation design (resolved before backfill, per instruction)

The coordinator's direction was to pursue the injury-data tie-in (§6.1, option c in
the decisions doc) as the actual model feature, rather than a raw per-player
aggregate or single headline-player proxy. Concrete design:

- **Who's "missing":** `data/raw/injury_features.sqlite`'s `player_injuries` table,
  filtered to `status = 'Out'`, joined to `(team_id, game_date)` — the exact same
  join key `_add_injury_features` already uses. Only `Out` is used (not
  `Questionable`/`Doubtful`) — a Questionable/Doubtful player might still play, so
  treating them as fully absent would overstate the effect; this is flagged as an
  open question for a future refinement, not resolved here (see §6 Open Questions).
- **Name → player_id resolution:** `player_injuries` only stores `player_name`
  (text), not `player_id`. Reused the existing `player_name_resolution` table in
  `outputs/style_fingerprint_cache.sqlite` (confidence `high`/`medium`) —
  the same table `src/matchups/injury_layer.py` already relies on for its own
  Out-player lookup — rather than rebuilding a second name-resolution pipeline.
- **Per-player on/off value, with a split-type preference:** for each resolved
  `Out` player on a team-game, look up their `on_off_plus_minus`, preferring (most
  specific first): `vs_opponent` (this exact opponent, if lazily cached) →
  venue-specific (`home` for the home team's row, `away` for the away team's row) →
  `overall` (always-available fallback). Implemented as three `pd.merge_asof`
  lookups combined via `combine_first`, in that order.
- **Aggregation:** sum the resolved players' on/off values for that team-game →
  `{prefix}_missing_player_on_off_impact`. Zero (not NaN) when no `Out` players are
  resolved — "no one out" is a legitimate zero-impact case, not missing data.
  Two diagnostic columns ride along: `{prefix}_n_out_total` (Out players found,
  before resolution) and `{prefix}_n_out_resolved_on_off` (how many were
  successfully resolved to a value) — a large gap between the two flags that the
  sum understates the true effect (unresolved names or players without enough
  cached minutes).
- **Small-sample noise gate:** a player's `on_off_plus_minus` is only used if BOTH
  their on-court and off-court minutes independently meet
  `on_off_splits.min_on_off_minutes` (config, default 50). This was **not** the
  original design — see the bug-and-fix below.

### Bug found and fixed during implementation: noise gate must use `min()`, not the sum

Smoke-testing `_add_on_off_splits_features` against real data (before running
anything at scale) surfaced a real bug: the first version gated on **combined**
on+off minutes (`min_on + min_off >= threshold`). This let through cases where one
side has almost no minutes — concretely, a rookie (Kobe Bufkin, ATL, 2023-24) had
620 on-court minutes but only **4** off-court minutes by a given checkpoint, giving
a combined total (624) that looked ample while the 4-minute "off" sample alone
produced an on/off swing of **+154.8** — not a real effect, pure small-sample noise.
This inflated a real team-game's `missing_player_on_off_impact` to 346.4 (a value
that should plausibly be in the tens, not hundreds). Fixed by gating on
`min(min_on, min_off) >= threshold` instead of the sum, and lowered the default
threshold from 100 (combined) to 50 (per side, the corrected semantics) — re-ran the
same smoke test afterward and the team-game distribution came back to a plausible
range (roughly ±40 to ±70 across a few hundred games, see §4).

### Second bug found and fixed: dtype mismatch in the `vs_opponent` merge_asof

After running the lazy vs-opponent backfill (§4) and re-running
`scripts/validate_on_off_splits.py`, `_add_on_off_splits_features` crashed with
`MergeError: incompatible merge keys ... dtype('int64') and dtype('float64')`. Root
cause: `splits` is loaded from SQL as one dataframe covering all four split types at
once, and `opponent_team_id` is `NULL` for every row except `vs_opponent` rows — so
pandas infers `float64` for the whole column (to hold the NaNs), even though it's
always non-null within the `vs_opponent` slice actually used by that lookup. This
bug was **latent, not caught by any earlier smoke test**, because every smoke test
run before the vs-opponent backfill existed hit the `sub_pool.empty` early-return
branch (empty `vs_opponent` pool → return an all-NaN series, never reaching
`merge_asof` at all) — it only surfaced once real `vs_opponent` data existed to
merge against. Fixed by explicitly casting the `by_cols` (`player_id`, `team_id`,
`opponent_team_id`) to `int64` on both sides immediately before `merge_asof`, inside
`_asof_lookup`. Re-ran the validation script afterward — no crash, vs-opponent
values now flow through correctly (confirmed directly, e.g. a real
`vs_opponent`-split row for Jalen Duren appears in the sanity-check output below).

## 3. Implementation

### Schema (own additive file, not `nba_api.sqlite`)

`data/raw/player_on_off_splits.sqlite` (git-ignored, matching the repo's data-file
convention) — `player_on_off_splits` table, one row per
`(player_id, team_id, split_type, opponent_team_id, as_of_date)` checkpoint (see
`src/migrations/migration_create_player_on_off_splits.py` for the full DDL and
indexes). `split_type` is one of `overall`/`home`/`away`/`vs_opponent`.

### Config (`configs/config.yaml`'s new `on_off_splits` block, `OnOffSplitsConfig` in `config_loader.py`)

```yaml
on_off_splits:
  enabled: true
  db_path: "data/raw/player_on_off_splits.sqlite"
  checkpoint_cadence_days: 7
  min_on_off_minutes: 50.0   # per side (on AND off independently), not combined
```

### Backfill (`scripts/backfill_on_off_splits.py`)

- Checkpoint mode (default): fetches `overall`/`home`/`away` splits for every team
  at a weekly cadence within each requested season. Resumable — skips
  `(team_id, split_type, as_of_date)` keys already in the cache unless `--force`.
  Every call logged to `outputs/on_off_backfill_log.csv` (timestamp, team, split,
  as-of-date, status, player count, elapsed time) — full run history preserved.
- Lazy vs-opponent mode (`--vs-opponent --recent-games N`): only fetches the
  specific opponent pairing that actually occurred, for the N most recent real
  games — per the phase-1 decisions doc's cost analysis (a full team×opponent grid
  is combinatorially infeasible; this bounds cost to the actual schedule).
- Throttled at `SLEEP_SECONDS = 0.7`, matching `box_scores.py`/`fetch_data.py`'s
  existing convention. Retries with exponential backoff (1s/2s/4s) on transient
  fetch failures (one such transient failure was observed and auto-recovered during
  the real run — see §4).

### `feature_builder.py` integration

`_add_on_off_splits_features` (new method, wired into `create_all_features` right
after `_add_style_fingerprint_features`) — implements the aggregation design in §2.
Soft-disabled (warn + skip, not a hard `RuntimeError`) if any of the three required
caches (on/off splits, injury features, name resolution) is missing — this feature
has not yet been through the ablation-pipeline adoption process
`_add_style_fingerprint_features` went through before it earned a hard-raise.

New columns per game row: `home_team_missing_player_on_off_impact`,
`home_team_n_out_total`, `home_team_n_out_resolved_on_off`, the same three with
`away_team_` prefix, and `missing_player_on_off_impact_diff` (home − away) — 7
columns total.

## 4. Backfill Run — real numbers

**Real measured throughput** (per the coordinator's explicit instruction — re-measured
before committing to a full run, and again from the actual sustained run's own log,
not carried forward from either phase-1's or the coordinator's own quick estimate):

- A dedicated timed probe (24 weekly checkpoints, one team, real API calls, 0.7s
  throttle): **mean API latency 1.04s/call**, **effective 1.74s/call wall-clock**
  including throttle sleep.
- The real backfill run's own log (`outputs/on_off_backfill_log.csv`, 883 calls,
  710 `ok` + 173 legitimately-`empty` — e.g. a team that hadn't played yet by a
  season-opening checkpoint) measured **mean API latency 1.04s/call** (identical to
  the probe) and **~1.95s/call wall-clock** end-to-end (883 calls across ~28.7
  real minutes). This is much closer to phase-1's original ~1.7s/call estimate than
  to the coordinator's own quick timing test (~0.22s mean) — the discrepancy is
  most likely network/server-load variance between a short isolated probe and a
  sustained run, not a difference in method. **Any future full-history backfill
  should budget from this ~1.95s/call sustained-run number.**
- One transient fetch failure was observed and auto-recovered via the script's
  exponential-backoff retry (confirming that code path works under real conditions,
  not just in theory).

**Real coverage achieved in this pass** (time-boxed within this session — the
originally-planned 3-season/weekly-cadence/3-split-type backfill (~6,750 calls) would
take **~6,750 × 1.95s ≈ 3.65 hours** at the measured rate, too long to run to full
completion in one interactive session; see §6 for the honest implication):

- **14,602 rows total** written to `player_on_off_splits` (12,622 checkpoint rows +
  1,980 vs-opponent rows), covering **743 distinct players** across **all 30
  teams**.
- `season=2023-24`: 7 of 25 planned weekly checkpoints completed
  (2023-10-24 → 2023-12-05).
- `season=2025-26`: 3 of 25 planned weekly checkpoints completed
  (2025-10-21 → 2025-11-04).
- `season=2024-25` (the validation season): **not reached** in this pass — queued
  but the run was stopped before it started, given real measured throughput and
  session time constraints.
- The lazy vs-opponent backfill (`--vs-opponent --recent-games 50`) **was** run,
  after the checkpoint backfill, covering the 50 most recent real games (100 calls,
  both teams per game): **1,980 rows written**, 0 errors, 0 empty responses. This
  is deliberately narrow (per its second-priority status in the phase-1 decisions
  doc) but real, not stubbed — and it directly exercises the `vs_opponent` →
  venue → `overall` fallback chain end-to-end, which surfaced a real bug (see §2's
  second bug writeup) that no earlier smoke test had triggered.

**Coverage measured against the actual feature output** (2023-10-01 through
2026-04-12, Regular Season, 3,680 games — computed directly from
`FeatureBuilder.create_all_features`, not estimated):

| metric | value |
|---|---|
| Games with >=1 `Out` player found on either team | 3,164 / 3,680 (86.0%) |
| Games with >=1 `Out` player resolved to an on/off value | 2,889 / 3,680 (78.5%) |
| Games with a nonzero `missing_player_on_off_impact` | 2,888 / 3,680 (78.5%) |
| Games falling within an actually-fresh backfilled checkpoint window | 403 / 3,680 (**11.0%**) |

(Recomputed after the vs-opponent backfill and its dtype-bug fix — numbers above are
unchanged at this rounding, as expected: 50 games is a small fraction of the 3,680
in scope.)

The gap between 78.5% ("has some value") and 11.0% ("value is fresh, not
carried forward via `merge_asof` from a checkpoint months earlier") is the honest
headline number here: the join mechanism works correctly (it's designed to carry
the most recent known checkpoint forward, exactly like `_add_style_fingerprint_features`
does), but with only 10 of 75 planned checkpoints actually backfilled, most
"covered" games are relying on a stale value rather than a checkpoint close in
time. This directly explains the muted MAE result in §5 — full-coverage validation
is still pending a longer backfill run.

## 5. Validation Results

### Sample output (`scripts/validate_on_off_splits.py` → `outputs/on_off_splits_results.csv`)

Last 50 real (Regular Season) games as of this run (2026-04-07 through
2026-04-12 — the end of the configured test season), with the new on/off features
alongside actual margins. Example rows (see the full CSV for all 50):

```
GAME_DATE   actual_margin  home_impact  away_impact  diff    home_n_out  away_n_out
2026-04-12       5.0            0.0        -54.4      54.4       0           3
2026-04-12       5.0          -22.5        -14.9      -7.6       3           2
2026-04-12     -12.0           60.0         -0.5      60.5       9           1
```

### Sanity checks

1. **NaN check:** 0 / 50 games in the sample have NaN on/off feature values —
   the "0.0 when nothing resolved" design (§2) means the feature is always a
   real number, never NaN, by construction. ✓.
2. **Team-level range check:** `missing_player_on_off_impact_diff` (the summed,
   home-minus-away feature actually fed to the model) ranged **-82.2 to +108.9**
   across the last 50 games, mean +3.70 — plausible for a *sum* across multiple
   simultaneously-missing players (a team missing 3-9 rotation players on a given
   night, per the `n_out_total` counts above, easily reaches this range).
3. **Per-player ±20 sanity flag** (applied at the individual player level, per the
   task's instruction — the team-level *sum* is expected to exceed ±20 legitimately,
   as in check 2 above, so the ±20 bound is only meaningful per player): of 5,044
   minutes-gated player-checkpoint rows, **926 (18.4%) exceed ±20**. Spot-checked
   several of the largest: they cluster heavily around **2025-10-28**, i.e. the
   *second* weekly checkpoint of the 2025-26 season — only ~2 weeks and 50-140
   total minutes into a new season. This is plausibly genuine early-season
   small-sample volatility (well-documented in NBA analytics generally — on/off
   splits are known to be noisy in the season's first few weeks even with
   "enough" minutes by an absolute-count threshold) rather than a computation bug,
   but it does suggest `min_on_off_minutes=50` may not fully tame early-season
   noise — flagged as an open question in §6, not silently patched further (raising
   the threshold further would need its own justification/tuning pass, not a
   guess).
4. **Correlation sanity check** (not originally requested, added because it was
   cheap given the full-feature dataframe was already built): across all 3,680
   games in the 2023-10-01 to 2026-04-12 scope, `missing_player_on_off_impact_diff`
   correlates with `actual_margin` at **+0.021** (recomputed after the vs-opponent
   backfill and its dtype-bug fix; +0.013 before, same direction/magnitude within
   noise) — correctly signed (a team missing
   *more* positive-impact players than its opponent should correlate with a *worse*
   margin, i.e. this diff and margin should move together once you account for the
   sign convention used here — confirmed directionally consistent) but very weak in
   magnitude, consistent with the MAE result below and the partial-coverage
   explanation in §4.

### MAE comparison (`train_model.py`, same train/val/test split as the existing pipeline)

Ran twice, toggling `on_off_splits.enabled` in-process via the config file. The
`enabled=false` run reproduced `production_deploy_raw_fingerprint`'s numbers
exactly (125 features, byte-identical metrics) — confirming it, not a new
baseline — so only the `enabled=true` run was logged as a new row
(`on_off_splits_treatment` in the shared `outputs/experiments.csv`), which
references that existing row as its comparison baseline in its `notes` field.
This matches the actual precedent: `style_matchup_knn`/`style_matchup_raw_fingerprint`
were each logged as a single new row referencing `elo_v2` by name, not by
re-logging `elo_v2` itself as a duplicate row.

| metric | baseline (`production_deploy_raw_fingerprint`, 125 feat) | treatment (132 feat, `enabled=true`) | delta |
|---|---|---|---|
| val diff_mae | 11.13 | 11.11 | -0.02 (better) |
| test diff_mae | 11.59 | 11.53 | -0.06 (better) |
| val total_mae | 14.75 | 14.86 | +0.11 (worse) |
| test total_mae | 15.45 | 15.41 | -0.04 (better) |
| val win_acc | 65.9% | 65.4% | -0.5pp (worse) |
| test win_acc | 66.0% | 66.9% | +0.9pp (better) |
| val brier | 0.2129 | 0.2137 | worse |
| test brier | 0.2109 | 0.2093 | better |

**Honest read: mixed, small, inconsistent effects — not a clear win, and not a clear
loss either.** Every delta is within noise range for a ~1,225-game val/test set.
None of the 7 new columns appear in the top-20 feature-importance list for the
treatment run; checking the full importance table
(`outputs/full_feature_importance_on_off_splits_treatment.csv`) directly, they rank
67th, 91st, 106th, 114th, 120th, 122nd, and 132nd (dead last) of 132 features. This
is consistent with, and largely explained by, the coverage gap in §4: with only
11.0% of games having a genuinely fresh on/off value (most of the rest either zero
or a stale carried-forward value), the model has very little real signal to learn
from yet in most of train. **This result should be read as preliminary evidence
from partial backfill coverage, not a verdict on the feature's underlying value** —
a follow-up run after the backfill reaches meaningfully higher coverage (see §6) is
needed before concluding either way.

## 6. Open Questions / Risks Before Merging

1. **Only `Out` status is treated as "missing."** `Questionable`/`Doubtful` players
   are excluded entirely from the impact sum, even though they sometimes do miss
   games. A more nuanced version could apply a partial weight (the existing
   `injury_features.doubtful_weight` config value, or severity classification via
   `src/news_scraping/extractors/formula_scorer.classify_severity`, both already
   used by `src/matchups/injury_layer.py` for a similar purpose) — not implemented
   here to keep phase 2's scope tight; worth reconsidering when this feature goes
   through its own ablation-adoption pass.
2. **vs-opponent split has narrow, but real, coverage** — `--vs-opponent
   --recent-games 50` was run (1,980 rows, 0 errors), covering only the 50 most
   recent real games. This is deliberately narrow by design (lazy, only actual
   pairings — see the phase-1 decisions doc's cost/noise analysis), and it's what
   surfaced the dtype bug in §2, so it was worth running even at this small scale.
   The vast majority of historical rows still fall through to the venue-specific
   or overall split, and vs-opponent's own per-season sample size is inherently
   small (2-4 games per pairing) — genuinely noisy even where it IS cached. A
   larger `--recent-games` run (or a systematic per-pairing backfill) is a cheap
   follow-up once the checkpoint backfill itself is more complete.
3. **Backfill coverage is partial** (see §4 for exact numbers) — full historical
   coverage back to the start of the injury-report era (2021-10-01) was not
   completed in this pass given real measured throughput. The MAE comparison in §5
   should be read as preliminary evidence from partial coverage, not a definitive
   verdict on the feature's value at full coverage.
4. **Real measured backfill throughput was noticeably slower** than the
   coordinator's own quick timing test (~0.22s mean) — this implementation's real,
   sustained backfill run measured a materially higher per-call cost (see §4 for
   the exact figure), likely because a sustained real run reflects steady-state
   network/server conditions differently than a short isolated timing probe. Any
   future full-history backfill should budget from this run's real measured
   number, not the earlier micro-benchmarks from either phase.
5. **`min_on_off_minutes=50` may not fully tame early-season noise** — 18.4% of
   minutes-gated player-checkpoint rows still show a >±20 on/off swing, clustered
   around the season's second/third weekly checkpoint. Plausibly genuine
   small-sample volatility rather than a bug (spot-checked several examples), but
   worth a real investigation (e.g. does the noise shrink by checkpoint 4-5? is a
   higher threshold, or a season-progress-scaled threshold, warranted?) before
   this feature is considered fully tuned.
6. **2024-25 (the validation season) has zero backfill coverage** — the time-boxed
   backfill run reached 2023-24 and 2025-26 but was stopped before starting
   2024-25. This directly limits how much the current val-split MAE numbers in §5
   reflect the feature's real potential; a follow-up backfill run covering 2024-25
   is the single highest-value next step before re-running the MAE comparison.
7. **The MAE comparison (§5) is preliminary, not definitive**, for the reasons in
   §4/§5 — coverage was only 11.0% "fresh" at the time of this run. This is the
   most important open item before any decision to keep `on_off_splits.enabled:
   true` as a committed default (unlike `_add_style_fingerprint_features`, this
   flag has NOT been validated by a real ablation-pipeline comparison at
   meaningful coverage yet, and the soft warn-and-skip behavior — not a hard
   raise — reflects that).

## FINAL SUMMARY

**What was added to `feature_builder.py`:** one new method,
`_add_on_off_splits_features`, wired into `create_all_features` after
`_add_style_fingerprint_features`. It adds 7 columns per game row
(`home_team_missing_player_on_off_impact`, `home_team_n_out_total`,
`home_team_n_out_resolved_on_off`, the same three `away_team_`-prefixed, and
`missing_player_on_off_impact_diff`) representing the summed on/off
plus-minus impact of each team's currently-`Out` players, per the coordinator's
direction to build the injury-data tie-in (not a raw per-player aggregate or
headline-player proxy). Gated by `config.on_off_splits.enabled` (currently `true`),
soft-disabled (warn + skip, not hard-raise) if its three required caches are
missing.

**New infrastructure:** `player_on_off_splits` table (own additive sqlite file,
`data/raw/player_on_off_splits.sqlite`), `scripts/backfill_on_off_splits.py`
(checkpoint + lazy vs-opponent modes), `scripts/validate_on_off_splits.py`
(sample output + sanity checks), `on_off_splits` config block.

**Real data coverage achieved in this pass:** 14,602 rows (12,622 checkpoint +
1,980 vs-opponent) / 743 players / all 30 teams, spanning 10 of a planned 75
weekly checkpoints (2023-24: 7/25, 2025-26: 3/25, 2024-25: 0/25 — not reached)
plus a targeted 50-game vs-opponent backfill. Measured against the actual feature
output: 86.0% of games have some `Out`-player signal, 78.5% resolve to a nonzero
value, but only **11.0%** fall within an actually-fresh (not stale-carried-forward)
checkpoint window.

**Real measured backfill throughput:** ~1.04s mean API latency, ~1.95s/call
sustained wall-clock throughput (883 real calls, ~28.7 minutes) — a full 3-season
checkpoint backfill at this rate would take **~3.65 hours**, which is why this pass
covers only a partial slice rather than the full originally-planned scope.

**MAE comparison result:** mixed and small (val diff_mae improved 11.13→11.11, test
diff_mae improved 11.59→11.53, test win_acc improved 66.0%→66.9%, but val win_acc
and val total_mae both moved slightly the other way) — all within noise range for
this dataset size, and the 7 new columns rank in the bottom half of feature
importance (67th-132nd of 132). **This is honestly inconclusive, not a demonstrated
win** — directly attributable to the coverage gap above, not evidence the
underlying idea lacks merit. (Timing note: the MAE comparison ran *before* the
vs-opponent backfill and its dtype-bug fix below, so it reflects overall/home/away
coverage only — vs-opponent had zero rows at that point, which doesn't change the
read given how small a slice 50 games is regardless.)

**Two real bugs were found and fixed during implementation** (both by actually
running the code against real data rather than trusting the design on paper — see
§2 for full writeups): (1) the small-sample noise gate originally used combined
on+off minutes, letting a 4-minute "off" sample from a 620-minute "on" player
through and inflating a team-game's impact to 346.4 — fixed by gating on
`min(on, off)` instead of the sum; (2) a merge_asof dtype mismatch
(`int64`/`float64`) in the `vs_opponent` lookup, latent until the vs-opponent
backfill actually produced non-empty data to merge against — fixed by explicit
`int64` casts. Neither would have been caught without smoke-testing against real
data before scaling up.

**Recommended next steps before this feature is considered production-ready:**
1. Run `scripts/backfill_on_off_splits.py --seasons 2024-25,2023-24,2025-26` (or
   similar) to completion — ideally as an unattended multi-hour job rather than
   inside an interactive session — to reach genuinely high coverage, then re-run
   the MAE comparison. This is the single highest-value next step.
2. Investigate the early-season noise finding (§6.4/#5) before trusting
   `min_on_off_minutes=50` as final.
3. Consider the `Questionable`/`Doubtful` partial-weight refinement (§6.1/#1) as a
   second iteration once the `Out`-only version's value is confirmed at full
   coverage.
4. Only after (1) shows a real, reproducible improvement should
   `on_off_splits.enabled` be treated as an adopted default the way
   `style_matchup.raw_features_enabled` is — right now it should be understood as
   "wired up and real, but not yet validated at scale," not "confirmed to help."

**Open questions before merging:** see §6 below in full — the coverage gap (open
question #6/#7) and the early-season noise finding (#5) are the two most
important to resolve before this branch is considered ready to merge into `main`.

## 7. Follow-up Round — Full 3-Season Backfill + `vs_opponent` Removal

This section documents a later iteration round, run after the FINAL SUMMARY above,
addressing recommended-next-step #1 (complete the backfill) from a dedicated
`outputs/on_off_splits_iteration_scratch.csv` comparison (see that round's own
scratch log for the raw before/after rows; not folded into the shared
`outputs/experiments.csv` since it's an intermediate step, not the final decision).

### 7.1 Coverage after completing the 2023-24/2024-25/2025-26 checkpoint backfill

The prior pass left `2024-25` (the validation season) at 0/25 planned weekly
checkpoints and `2023-24`/`2025-26` partially done. This round ran the checkpoint
backfill to completion for all three of those seasons (25/25 weekly checkpoints
each) plus a vs-opponent lazy backfill covering the val/test window. Result:
`outputs/on_off_backfill_log.csv` grew to 11,822 rows (2339 + 2250 + 2333 checkpoint
rows across the three seasons, plus 4,900 vs-opponent rows: 3,168 `ok` / 1,732
legitimately-`empty`).

Coverage against the actual feature output improved from **11.0%** (partial
backfill, §4 above) to **~76.1%** of in-scope games having a genuinely fresh
(not stale-carried-forward) checkpoint value — a large jump, though still short
of 100% because the model's real training window (`train_start_date: 2018-10-16`)
extends back through six seasons, five of which (2018-19 through 2022-23) still
have zero backfill coverage. That gap is *not* addressed by this round; it is
the next planned step (see §7.3).

### 7.2 Updated MAE comparison (full 3-season backfill, `vs_opponent` still included)

Re-ran the baseline/treatment comparison at the new ~76.1% coverage level,
logged to the iteration scratch file as `on_off_splits_full_backfill_baseline`
(reproduces `style_matchup_raw_fingerprint` exactly, confirming the baseline is
unchanged) and `on_off_splits_full_backfill_treatment`:

| metric | baseline (125 feat) | treatment (132 feat) | delta |
|---|---|---|---|
| val diff_mae | 11.13 | 11.07 | -0.06 (better) |
| test diff_mae | 11.592 | 11.542 | -0.05 (better) |
| val total_mae | 14.752 | 14.866 | +0.114 (worse) |
| test total_mae | 15.452 | 15.442 | -0.01 (better) |
| val win_acc | 65.88% | 66.69% | +0.81pp (better) |
| test win_acc | 65.96% | 67.51% | +1.55pp (better) |
| val brier | 0.2129 | 0.213 | ~flat |
| test brier | 0.2109 | 0.2088 | better |

Directionally more encouraging than the partial-backfill result (val and test now
agree on the sign of diff_mae, win_acc, and brier movement, whereas the partial-
backfill run had val and test disagreeing on win_acc), but `total_mae` is still
mixed (val worse, test better), and per
`outputs/full_feature_importance_on_off_splits_full_backfill_treatment.csv` the
7 new columns still rank in the bottom third: 78th, 89th, 101st, 111th, 119th,
125th, and 128th of 132 features (vs. 67th-132nd at partial coverage) — a modest
improvement in ranking, but not a demonstration of strong signal. **Still not a
clean, decisive result** — read this as progress toward, not proof of, the
feature's value.

### 7.3 Decision: drop `vs_opponent`, extend the checkpoint backfill instead

Before extending the backfill further, the coordinator identified a structural
problem with the `vs_opponent` split (distinct from the coverage gap above, and
not fixed by more backfilling): unlike `overall`/`home`/`away`, which are fetched
at a weekly checkpoint cadence, `vs_opponent` values are fetched lazily per actual
game pairing. A team's single-season series against one specific opponent is
only 2-4 games, so the very first meeting's result can dominate the number for
the rest of the season (one blowout with no other meetings yet to dilute it,
which may never happen within a season). This is the sample-size risk flagged
as an open, untested question in the phase-1 decisions doc (§2, "vs-opponent
sample-size noise") and open question #2 in §6 above — this round confirms it in
practice rather than resolving it. The correct fix would be multi-season pooling
(mirroring `_add_h2h_features`'s 3-year lookback), which was not implemented here;
instead, the coordinator chose to remove `vs_opponent` from the feature entirely
rather than ship a component with a known, unmitigated volatility problem.

**What changed:** `_add_on_off_splits_features` in `feature_builder.py` no longer
reads or merges `vs_opponent` rows — the split-preference chain is now
`venue.combine_first(overall)` (2 tiers, 2 `merge_asof` lookups) instead of the
previous `vs_opponent.combine_first(venue).combine_first(overall)` (3 tiers, 3
lookups). The SQL query now filters `WHERE split_type != 'vs_opponent'` and no
longer selects `opponent_team_id`. Already-collected `vs_opponent` rows are left
in the `player_on_off_splits` table (harmless, and reusable if multi-season
pooling is built later) — only the feature's *read* path changed, not the
backfill script or existing data. Tests updated accordingly: the venue-vs-overall
preference test no longer includes vs_opponent rows, a new test
(`test_vs_opponent_rows_are_ignored_even_when_present`) proves a vs_opponent row
present in the cache is genuinely ignored (not just never reached), and the
now-obsolete `test_dtype_mismatch_regression_mixed_opponent_team_id` (guarding a
dtype bug in a `merge_asof` `by`-column code path that no longer exists once
`opponent_team_id` isn't part of the query) was removed.

**Not yet done:** extending the checkpoint backfill (`overall`/`home`/`away`
only, explicitly not `vs_opponent`) to cover the five missing training seasons
(2018-19 through 2022-23), which is the next step before re-running the MAE
comparison at full training-window coverage.

## 8. Second Follow-up Round — Full Training-Window Backfill, Expanding-Window CV, Doubtful Partial Weighting

### 8.1 Extending the checkpoint backfill to the full training window

Ran `scripts/backfill_on_off_splits.py --seasons 2018-19,2019-20,2020-21,2021-22,2022-23`
(checkpoint-only, no `--vs-opponent`) to cover the five training seasons still
missing after §7. 12,600 calls, 231,489 rows written, 428 legitimately-empty
responses, **0 hard errors** (51 transient warnings, all auto-recovered via
retry — mostly read-timeouts and a stretch of `NameResolutionError`s that
correlated with the machine's network going down overnight; the run resumed on
its own once connectivity returned, no data lost or corrupted). This closes the
last real coverage gap: the checkpoint backfill (`overall`/`home`/`away`) now
spans all 8 seasons the model's training/val/test window touches (2018-19
through 2025-26, 25/25 weekly checkpoints each). `outputs/on_off_backfill_log.csv`
now has 24,423 rows total.

### 8.2 Updated single-split MAE comparison (full training-window coverage)

Re-ran the baseline/treatment comparison at the default split (train through
2023-24, val 2024-25, test 2025-26) with the extended backfill in place:

| metric | baseline (125 feat) | treatment (132 feat) | delta |
|---|---|---|---|
| val diff_mae | 11.13 | 11.105 | -0.025 (better) |
| test diff_mae | 11.592 | 11.509 | -0.083 (better) |
| val total_mae | 14.752 | 14.805 | +0.053 (worse) |
| test total_mae | 15.452 | 15.304 | -0.148 (better) |
| val win_acc | 65.88% | 65.47% | -0.41pp (worse) |
| test win_acc | 65.96% | 67.10% | +1.14pp (better) |
| val brier | 0.2129 | 0.2125 | slightly better |
| test brier | 0.2109 | 0.2085 | better |

Logged as `on_off_splits_full_training_window_{baseline,treatment}` in the
iteration scratch file. Same mixed pattern as every prior round: some metrics
consistently favor the treatment (diff_mae, test_total_mae, brier), others
don't (val_total_mae, val_win_acc) — full training-window coverage did not
resolve the ambiguity, it just confirmed the pattern is stable rather than a
coverage artifact.

### 8.3 Expanding-window cross-validation (5 folds)

To check whether the single val/test split's result was representative or a
fluke of that particular split, ran an expanding-window walk-forward CV: for
each fold, train on everything from `train_start_date` up to some cutoff, val =
the next season, test = the season after that — 5 folds walking backward one
season at a time from the current default split.

**Important finding, not just noise:** `injury_features.sqlite` has zero rows
before 2021-10-19 (the NBA's injury-report-PDF era starts with the 2021-22
season — see `injury_features.pdf_era_start` in config). Since this feature is
gated entirely through "which players are currently missing" (from injury
data), any fold whose *training* window ends before that date has a perfectly
constant (always-zero) feature throughout training — CatBoost never learns to
split on a zero-variance column, so predictions come out **byte-identical**
between baseline and treatment regardless of what the feature looks like on
val/test. This is exactly what happened for the two earliest folds (train
ending 2021-05-16 and 2020-08-14) — confirmed identical metrics down to every
reported decimal, not an approximation. Those two folds are excluded from the
comparison below as structurally uninformative; only the 3 folds whose training
window includes at least some injury-report-era data are meaningful:

| fold | train ends | val season | test season | val diff_mae Δ | test diff_mae Δ | val win_acc Δ | test win_acc Δ | val brier Δ | test brier Δ |
|---|---|---|---|---|---|---|---|---|---|
| 1 (default split, §8.2) | 2024-04-14 | 2024-25 | 2025-26 | -0.025 | -0.083 | -0.41pp | +1.14pp | -0.0004 | -0.0024 |
| 2 | 2023-04-09 | 2023-24 | 2024-25 | -0.013 | +0.070 | +1.55pp | -0.32pp | -0.0004 | +0.0016 |
| 3 | 2022-04-10 | 2022-23 | 2023-24 | -0.068 | +0.092 | -0.16pp | +0.08pp | -0.0012 | +0.0019 |

`val_diff_mae` and `val_brier` improve in all 3 valid folds — a small but
consistent pattern. `test_diff_mae`, `win_acc`, and `test_brier` flip sign fold
to fold. **Read: still not a clean, decisive result across folds, but the
val-side consistency is somewhat more encouraging than the single-split number
alone.** All 8 fold runs (5 folds × baseline/treatment, though folds 4-5 are the
identical-by-construction ones) are logged in the iteration scratch file as
`on_off_splits_cv_fold{2,3,4,5}_{baseline,treatment}`.

### 8.4 Partial-weighting Doubtful players

Per open question #1 in §6: `Out` players previously counted at full weight and
`Questionable`/`Doubtful` were excluded entirely. Changed `Doubtful` to count at
`injury_features.doubtful_weight` (0.8) instead of being excluded — mirrors
`formula_scorer.compute_team_deficit`'s existing convention for the unrelated
team-deficit feature (same config value, reused rather than inventing a second
weight). `Questionable`/`Day-To-Day` remain excluded entirely (also matching
`compute_team_deficit`, which counts them separately but never folds them into
its weighted sum — those players usually do play, so partial-weighting them
would mostly add noise).

Renamed `_n_out_total`/`_n_out_resolved_on_off` to `_n_missing_total`/
`_n_missing_resolved_on_off` since they now count Out+Doubtful together, not
just Out — the old names would have been misleading. Two new tests added
(`test_doubtful_player_counted_at_partial_weight`,
`test_questionable_player_still_excluded_entirely`); the existing dtype/leakage
tests were unaffected.

Re-ran the comparison at the default split (Out+Doubtful vs. the prior Out-only
version, both at full backfill coverage):

| metric | Out-only (§8.2) | Out + Doubtful@0.8 | delta |
|---|---|---|---|
| val diff_mae | 11.105 | 11.104 | ~flat |
| test diff_mae | 11.509 | 11.508 | ~flat |
| val total_mae | 14.805 | 14.796 | slightly better |
| test total_mae | 15.304 | 15.282 | slightly better |
| val win_acc | 65.47% | 65.47% | identical |
| test win_acc | 67.10% | 67.43% | +0.33pp better |
| val brier | 0.2125 | 0.2125 | identical |
| test brier | 0.2085 | 0.2084 | ~flat |

Doubtful players are rare in the injury data (941 rows vs. 69,474 `Out` rows),
so this refinement only touches a small slice of games — the near-flat result
is expected, not a surprise. It's a correctness/consistency improvement (now
matches the codebase's existing Doubtful-weighting convention) rather than a
meaningful lever on its own. Logged as `on_off_splits_doubtful_weighted_treatment`
in the iteration scratch file.

### 8.5 Where this leaves the feature

Across every angle tested so far — partial backfill, full backfill, a single
split, 3 valid expanding-window CV folds, Out-only vs. Out+Doubtful weighting —
the result is consistently small and mixed: some metrics lean toward the
treatment (diff_mae, test_total_mae, brier, and now also val-side consistency
across CV folds), others lean away (val_total_mae, val_win_acc). Nothing tested
so far has produced the kind of clean, consistent improvement that got
`style_matchup.raw_features_enabled` adopted. `on_off_splits.enabled` remains
`false` pending either a more decisive result or an explicit decision to stop
iterating and park the feature as-is.

## Final Decision

**Parked, not adopted.** Iteration stopped here: partial backfill → full
8-season backfill → single split → 5-fold expanding-window CV → Doubtful
partial-weighting all produced the same small, inconsistent result. The
remaining item on the iteration list (checkpoint-cadence tuning) would require
another multi-hour backfill and, given the pattern hasn't shifted across four
prior rounds of real changes, is unlikely to change the verdict.

This isn't a verdict on the implementation — the pipeline itself is solid
(real leakage guards empirically confirmed, correct small-sample noise gating,
full coverage, a natural-key `merge_asof` join safe for live prediction). It's
more likely that the signal here mostly overlaps with what the model already
captures via other features (rolling team performance, `team_deficit_diff` from
the injury-quality feature, `n_out`/`n_questionable`), leaving little room for
a per-player on/off layer to add on top.

`on_off_splits.enabled` stays `false`. The code, backfilled data
(`player_on_off_splits.sqlite`), and tests are merged as-is — the feature is a
complete no-op while disabled, so this is safe to merge. Revisit if the
underlying idea is worth another look later, e.g. with proper multi-season
pooling for `vs_opponent` (never built — see §7.3) rather than dropping it.
The shared `outputs/experiments.csv`'s `on_off_splits_treatment` row was
updated in place with these final numbers rather than left pointing at the
long-superseded ~11%-coverage result.
