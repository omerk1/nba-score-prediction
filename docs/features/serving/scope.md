# Serving — scope

The screenshot-to-recommendation product idea: a screenshot of today's games
and spreads in, model bets with confidence intervals and a margin heatmap
out. Two pieces:

1. **Extraction** — screenshot -> structured picks (teams translated to
   canonical NBA names, spread, odds).
2. **Serving** — structured picks -> model prediction ->
   `src/evaluation/predictive_distribution.py` -> a recommendation (win
   probability, cover probability against the given line, edge vs. the
   market-implied probability, margin heatmap).

## Already built (`research/predictive-distribution-heatmap`, merged #75)

- `src/evaluation/predictive_distribution.py`: validated homoscedastic
  predictive distribution (`win_probability`, `cover_probability`,
  `margin_pmf`). Takes `(point_pred, residual_sample, bandwidth)` — it has no
  notion of how to get a live point prediction or which residual sample to
  use; both are this feature's job to supply.
- `heteroscedastic_interval_screen` (`docs/EXPERIMENTS.md`): no per-game
  confidence variation is recoverable from the current feature set, so every
  game's heatmap has the same shape, just recentered on that game's own
  prediction — an accepted simplification carried into this scope, not
  something to re-litigate here.

## Gaps found while scoping (real, need decisions before building)

1. **No fresh champion model artifact.** `predict_game.py` already does
   single-game point prediction (team IDs + optional date -> synthetic
   feature row -> `predictor.predict`) — a reusable entry point, don't
   rebuild it. But `data/models/score_predictor.pkl` is stale: its saved
   `model_params` show the pre-`hp_tuning_cv` hyperparameters (`depth=6,
   learning_rate=0.1`, no `l2_leaf_reg`/`min_data_in_leaf` at all) — it
   predates the adopted champion (`depth=2, learning_rate=0.0907,
   l2_leaf_reg=2.84, min_data_in_leaf=37`, `docs/EXPERIMENTS.md`'s
   `hp_tuning_cv` entry). Needs a fresh `train_model.py --protocol
   single_split` run against the current config before serving anything
   real.
2. **No residual sample for a live game.** `predictive_distribution.py`
   needs a validation-fold residual sample to build the heatmap; live games
   aren't part of any CV fold. Open question below.
3. **Team-name translation.** Bookmaker screenshots use Hebrew team
   names/abbreviations. `nba_api.stats.static.teams` already gives a
   canonical English nickname -> `team_id` mapping, and
   `scripts/market_benchmark.py`'s `build_nickname_to_team_id()` already
   uses it for exactly this kind of join — reuse that, don't reinvent it. No
   Hebrew mapping exists yet. Plan: have the vision-extraction step
   translate straight to the canonical English nickname (not hand-build a
   static Hebrew lookup table), then validate the result against
   `build_nickname_to_team_id()`'s known keys and flag anything that doesn't
   match rather than silently guessing.
4. **Odds format.** The screenshot shows decimal odds (e.g. `1.80`) and a
   spread as a signed point value per team. Need: decimal-odds -> implied
   probability (`1/odds`) for the edge comparison, and the spread's
   favorite/underdog sign convention confirmed against the bookmaker's own
   layout (not assumed) before it's turned into the "diff must exceed this"
   value `predictive_distribution.cover_probability` expects.

## Built (this branch)

- `data/models/score_predictor.pkl` retrained against the current champion
  config (`serving_champion_model_refresh`, `outputs/experiments_v2.csv`).
- `src/serving/live_features.py`: `build_live_game_features` (extracted from
  `predict_game.py`'s own inline logic, now shared by both). Fixes two real
  pre-existing bugs found while building this, neither introduced here:
  - `NBADataLoader.load_recent_team_games` had no date cutoff at the SQL
    level (`ORDER BY game_date DESC LIMIT n_games` over the *whole* table),
    so predicting any date other than "today" silently returned zero rows
    once the DB held more than `n_games` of a team's games after it. Fixed
    with an `end_date` param — but superseded below anyway.
  - **Elo came out NaN on every live prediction.** `_add_elo_features`
    (`feature_builder.py`) computes Elo by reloading real games from the DB
    and merging back onto `GAME_ID` — it can never see the synthetic
    "upcoming" row `build_live_game_features` injects (not in the DB, no
    matching `GAME_ID`), regardless of how much history is loaded. `elo_diff`
    is a top-3 feature by importance in the current champion, so this wasn't
    a minor gap — every live prediction was silently missing its
    single most important signal. Fixed by also loading the *full* game
    history (`data_start_date` through the prediction date, matching
    `load_training_data`'s own context window, not just a per-team recent
    slice) and recomputing Elo/momentum directly from that in-memory history
    (which does include the synthetic row) via `compute_elo_ratings`/
    `compute_elo_momentum`, splicing the result into just the synthetic
    row. Verified: predicting a known historical matchup now nearly
    reproduces the CV/test-fold pipeline's own prediction for that exact
    game (110.1/107.6 live vs. 110.1/107.3 via `run_split`) and 0/148
    feature columns are NaN, vs. NaN Elo and a visibly different (110→113
    home score) prediction before the fix.
  - Also fixes a same-date collision: predicting a date that already has a
    real game for either team duplicates a merge key inside
    `_add_rolling_features` and crashes — now excluded explicitly, which
    matters for backtesting a known past matchup (used above to verify the
    Elo fix), not just genuinely future dates.
- `src/serving/recommend.py`: `load_resources` (model + fold5 val residuals
  + bandwidths) and `recommend_game` (matchup + optional spread/moneyline/
  total lines and decimal odds -> point prediction, win probability, cover
  probability, edge vs. given odds, margin heatmap). Market-sign-convention
  and decimal-odds handling from gaps #3/#4 below implemented here.
- `scripts/recommend_game.py`: CLI demo, already-structured picks in
  (mirrors the screenshot's shape: team IDs, spread, decimal odds on both
  sides, total line) — run end-to-end, output sanity-checked (e.g. a home
  team predicted to win by only +2.6 correctly shows a large negative edge
  against a -9 home spread priced near even odds).
- `src/serving/team_lookup.py`: `build_nickname_to_team_id` (canonical NBA
  nickname -> team_id), moved out of `scripts/market_benchmark.py` so the
  extraction module below reuses the exact same mapping instead of a second
  copy. `market_benchmark.py` now imports it; behavior unchanged.
- `src/serving/extract_picks.py`: `extract_picks_from_screenshot`, vision
  extraction via Gemini (`google-genai`, `GOOGLE_API_KEY` — this project's
  existing LLM pattern from `src/availability/llm_estimator.py`, not a new
  provider). Deliberately extracts a loose, mostly-optional per-game schema
  (gap #3/#4's Hebrew-nickname-translation and decimal-odds handling; a
  `push_odds` field for the 3-way spread-with-push market structure the
  original screenshot example actually showed, since a bookmaker's exact
  market shape shouldn't be hand-coded per layout) rather than one fixed
  layout, closed-vocabulary-constrained to `build_nickname_to_team_id()`'s
  own real NBA nicknames so a team that can't be confidently matched is
  dropped, not guessed. `scripts/recommend_from_screenshot.py`: ties this to
  `recommend_game` end-to-end (screenshot in, one recommendation per
  recognized NBA game out).
- **Runtime-verified against a real screenshot — 2026-09-25, credits
  restored.** `screenshot_spread_example.png` (full context: team names,
  NBA badge, time): both games extracted with 100% correct team IDs, spread
  signs/values, and odds (Heat -1.5/Celtics +1.5 and Mavericks
  -11.5/Warriors +11.5, both @ 1.80 — verified against nba_api's real team
  IDs, not just visually plausible output). `screenshot_total_example.png`
  (a bare odds-row crop with NO team name or NBA badge anywhere in it)
  correctly returned an empty array rather than guessing which game the
  total belonged to — exactly the "can't confidently match, drop rather
  than guess" behavior the prompt asks for, not a bug. Real takeaway: a
  screenshot needs to include the team-name/context row for a market to be
  extracted at all — a totals-only crop with no team context is an
  unreasonable input, not a gap in this code.
- **Home/away bug found and fixed.** Re-running extraction on the same
  spread screenshot repeatedly (still `temperature=0`) showed one game's
  home/away assignment flip-flopping across calls (the other stayed
  stable) — the screenshot itself has no home/away marker at all, just two
  teams' odds boxes side by side, so the model was guessing. Adding an
  explicit layout rule to the prompt ("right side is home team," per this
  bookmaker's Hebrew/RTL convention) *reduced* but didn't eliminate the
  instability (still flipped 2/4 repeated runs). Given this model's
  `home_advantage` is a large, tuned Elo term, a backwards home/away isn't
  noise, it's a silently wrong prediction — not something to leave to a
  vision model's guess. Fixed properly with `src/serving/schedule_lookup.py`
  (`resolve_home_away`): cross-references the two extracted teams + the
  game date against the NBA's own published schedule (`nba_api`'s
  `ScoreboardV3`), and `extract_picks_from_screenshot` now corrects (swaps
  every paired `home_*`/`away_*` field, not just the team IDs) whenever the
  schedule disagrees with the model's guess; falls back to the model's
  guess only if no scheduled game is found (wrong/missing date, a game the
  live scoreboard doesn't cover). Verified against known ground truth (a
  game already in the local DB with a known true home team) — correct.
  12 new tests in `tests/test_extract_picks.py` cover the override/swap
  logic with the schedule call mocked out (no live network call in the
  suite); the ground-truth check itself was run manually, once, not added
  as a test (a real network call every run for no coverage beyond what the
  mocked tests already give — see `CLAUDE.md`'s Cost discipline section).
- **Two real example screenshots** (real NBA games, Hebrew) saved as
  `tests/fixtures/screenshot_spread_example.png` and
  `screenshot_total_example.png` — both for the test above and for live
  re-verification later. They corrected two assumptions from the earlier
  placeholder mockup:
  - Real spread markets on this bookmaker are a plain 2-way market with a
    half-point line specifically to rule out a push (e.g. Heat -1.5 /
    Celtics +1.5), not the 3-way team/push/team structure the placeholder
    example showed. `push_odds` is kept in the schema (harmless, optional)
    for a bookmaker that does show one, but 2-way is the common case, not
    the exception — the extraction prompt now says so explicitly.
  - Real screenshots carry UI chrome next to the odds (a promo/boost badge,
    a "SD" bet-type label, "recommendation"/"source" icon buttons) that
    isn't itself odds data. The extraction prompt now explicitly says to
    ignore it — a genuine risk once real images are involved, invisible
    when testing against clean structured JSON alone.

## Not yet built

- No API/web-app wrapper yet — CLI only, per this branch's non-goals below.

## Explicit non-goals for this branch

- No live odds fetching/API integration beyond what's already in the
  screenshot.
- No user accounts, persistence, or bet placement.
- No web/API framework yet — CLI/script first; API vs. web app is a later
  decision, out of scope here.

## Decisions

- **Residual sample for live heatmaps: fold5's validation residuals.**
  Simplest, reuses the already-computed/already-validated 2024-25
  out-of-sample set from `predictive_distribution_heatmap`, no new
  training/holdout step. Fold5's validation season (2024-25) is still older
  than live games — a real, untested drift assumption — but that risk isn't
  resolved by holding out a fresh slice either, so it's accepted rather than
  engineered around for now.
- **Stale model retrain: done in this branch.** It's a blocker for testing
  anything in this feature end-to-end anyway (serving would otherwise run
  against pre-`hp_tuning_cv` hyperparameters), so bundling it keeps the
  branch buildable and testable on its own rather than depending on a
  separate untracked fix landing first.
