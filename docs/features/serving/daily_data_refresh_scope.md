# Daily data refresh — scope

Goal: the serving path (`src/serving/live_features.py` → `recommend.py`)
builds features point-in-time from the local data stores at call time, so a
daily recommendation job is only as fresh as those stores. This scopes a
daily job that keeps them current through yesterday's games. Scoped from
code reading only — nothing here was run against the live API.

## What already exists (most of the work is done)

- **`src/data_processing/fetch_data.py` is already incremental and
  idempotent.** On any run after the first it re-fetches only from the
  season containing `MAX(game_date)` in the DB (`_get_last_game_date` →
  `_date_to_season`) — in-season that's 2 `LeagueGameLog` calls (Regular
  Season + Playoffs for the current season), not a full refetch. Writes via
  `INSERT OR IGNORE` on the `game_id` PK, so re-running on the same day is
  safe: no duplicates, existing rows skipped. There is no upsert — a row
  once inserted is never corrected (acceptable: `LeagueGameLog` only lists
  completed games; post-hoc NBA stat corrections won't propagate).
- **`src/matchups/precompute_scores.py`** rebuilds the style-fingerprint
  cache; its own docstring already says to run it "right after each
  fetch_data.py refresh". Its box-score input cache is incremental
  (`build_box_score_cache` fetches only game_ids missing from
  `box_score_stats`, `INSERT OR IGNORE`), but the fingerprint recompute
  itself covers the full ~12.7k-game history each run — minutes, not
  seconds; fine daily, not per-prediction.
- **`scripts/build_injury_features.py --run nightly_update`** already
  exists for the injury side: scrapes ESPN's current injury page (not
  stats.nba.com) and writes today's `(game_date, team_id)` rows — the exact
  key `_add_injury_features` joins on.

## What must be refreshed daily (and what not)

Traced through `feature_builder.create_all_features` against the `enabled`
flags in `configs/config.yaml`:

1. **`game` table** (`data/raw/nba_api.sqlite`, via `fetch_data.py`) —
   feeds everything computed in-memory: rolling/rest/style/opponent-quality/
   venue/matchup/H2H/travel features, Elo + momentum (`elo_features.enabled:
   true`), and season motivation's `preferred_opponent_delta`
   (`season_motivation.enabled: true`; standings derive in-memory from
   `game`, no separate store). Stale games here silently shift every rolling
   window and Elo rating — no error, just a worse prediction. This is the
   one store with no degradation signal.
2. **Style fingerprint cache** (`outputs/style_fingerprint_cache.sqlite`,
   via `precompute_scores.py`) — `style_matchup.raw_features_enabled: true`,
   and the pace-score columns are the #1/#2 features by importance.
   Degrades gracefully (asof lookup falls back to each team's last cached
   fingerprint, `_warn_if_style_fingerprint_cache_stale` logs it), so a
   missed day is tolerable; a missed month is not.
3. **Injury features** (`data/raw/injury_features.sqlite`, nightly update
   above) — `injury_features.enabled: true`, exact-date join: no row for
   today → `zero_fill` + `has_injury_data=0`, i.e. a live game silently
   loses its injury signal. Weekly (not daily): a current-season
   `player_importance` snapshot (`backfill_season` skips existing dates;
   scoring asof-joins the latest snapshot, so staleness degrades gently).

**Explicitly not needed** (behind disabled flags): on/off splits, pbp,
availability agent, KNN style-matchup score, official pace, prediction
intervals, player game logs (availability labels), Polymarket prices.

## No retraining required — confirmed

`recommend_game` → `build_live_game_features` loads the full game history
from the DB and builds the feature row at call time; the model artifact is
only read, never dependent on data recency. `load_resources` recomputes
fold5 residuals from `data/features/val_features.csv` + the loaded model —
both fixed artifacts from the same past `train_model.py --protocol
single_split` run, untouched by data refresh. New games land after every
split boundary (test ends 2026-04-12), so a refresh doesn't even change
training inputs. Separate concern: a periodic retrain (roughly once per
season, rolling `datasets_loading`/`cv.folds` dates forward one season)
would keep the residual sample and hyperparameters current — that touches
fold definitions, which per project rules need explicit sign-off first.
Whenever it happens, `model.pkl` + `val_features.csv` must ship as a pair.

## Proposed entry point

`scripts/daily_update.py` — a thin orchestrator, not new fetch logic (each
store's incremental/idempotent path already exists; a flag on
`fetch_data.py` can't cover the other two stores):

1. `fetch_data.main()` (must run first — the other two derive from `game`).
2. `precompute_scores.precompute_and_cache()`.
3. Injury nightly update; weekly, the current-season importance snapshot.
4. Freshness check: fail (non-zero exit) if `MAX(game_date)` < yesterday
   while yesterday had scheduled games — the only non-self-reporting
   staleness case (#1 above).

## Failure handling

- Each step is independently idempotent — rerunning the whole script after
  a partial failure is the recovery mechanism; no state to clean up.
- `fetch_data.main()` currently swallows per-season errors (logged,
  continues, exits 0). The orchestrator should add the retry pattern the
  backfill scripts already use (`scripts/backfill_player_game_logs.py`: 3
  attempts, linear backoff, `timeout=60`) and surface failure via exit code.
- NBA API down / partial day: DB stays at its last good state; serving
  still works, predictions just use slightly staler form. The future daily
  recommendation job should run after this job and check the freshness
  gate (step 4) before sending anything, rather than assume success.

## nba_api constraints (where this can run)

- Established rate-limit convention: 0.6–1.5s sleep between calls
  (`SLEEP_SECONDS = 0.7` in `fetch_data.py`/`box_scores.py`/pbp collector).
  Daily volume is tiny: ~2 `LeagueGameLog` calls + ~2 box-score calls +
  1 ESPN request (+2 `leaguedashplayerstats` weekly) — well under any limit.
- `docs/NEW_DATA_FEASIBILITY.md` flags stats.nba.com rate-limiting/blocking
  risk, and stats.nba.com is known to block datacenter/cloud IPs outright —
  so this runs on a residential machine (user's own box, cron/launchd), not
  GitHub Actions or a cloud VM. The ESPN injury scrape has no such issue.
- Timing: games final late ET → run early morning ET; the injury nightly
  update's own docstring suggests ~11:00 ET on game days (reports cover
  that day's games). Two invocations, or one late-morning run, both work —
  decide with the recommendation job's own send time.

## Open questions

- `fetch_data.py` hardcodes `DB_PATH = Path("data/raw/nba_api.sqlite")`
  instead of reading `data_paths.raw_db` (same value today) — align while
  touching it, or leave?
- One run (late morning ET, after injury reports) vs. two (box scores at
  ~06:00 ET, injuries at ~11:00 ET)?
- `precompute_scores` runtime on this machine is unmeasured ("minutes" is
  an estimate from its full-history docstring) — measure once before
  committing to a schedule.
- Historical injury rows come from NBA PDFs, nightly rows from ESPN — a
  known source mismatch inherited from the existing pipeline, not new here;
  worth a one-time spot check that ESPN-day rows look comparable.
