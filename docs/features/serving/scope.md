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

## Not yet built

- Extraction module — vision-based screenshot -> structured picks (team
  names translated to canonical NBA nicknames per gap #3, feeding
  `recommend_game`'s team-ID/spread/odds arguments). Not yet designed in
  detail (prompt shape, output schema, error handling for unrecognized
  teams/layouts).
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
