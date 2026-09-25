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

## Proposed shape (not yet built)

- `src/serving/` (new):
  - extraction module — vision-based screenshot -> structured picks. Not
    yet designed in detail (prompt shape, output schema, error handling for
    unrecognized teams/layouts).
  - recommendation module — structured pick -> prediction +
    `predictive_distribution` -> a recommendation dict (model probability,
    market-implied probability, edge, heatmap).
- A CLI or small script to run end-to-end on a screenshot path first,
  before any API/web-app wrapper — matches how every other piece of this
  project got built (script first, product surface later).

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
