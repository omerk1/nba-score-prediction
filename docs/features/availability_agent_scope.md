# Availability Agent — Scope

Status: scoped 2026-09-17, not started. Ships disabled by default and goes through
the ablation-gated workflow in CLAUDE.md before any flag flips. Companion:
`docs/LLM_COMPONENT_OPTIONS.md` (why this option).

## 1. Hypothesis

Injury reports list players as Out, Questionable, or Doubtful. The pipeline treats
Out as fully absent, Doubtful as 80% absent (`injury_features.doubtful_weight`),
and Questionable as present (counted only in `n_questionable`). A per-case estimate
of P(plays) and expected minutes share, built from the player's own history and
game context, should be a better absence weight than those fixed constants, and the
resulting `team_deficit` should improve the composite score on the folds that have
injury coverage.

Secondary goal: gain experience building an LLM pipeline with retrieval, honest
baselines, caching, and a memorization check.

## 2. Task definition

- **Unit**: one row of `player_injuries` with status in {Questionable, Doubtful}.
  Pool: ~7,300 rows, 2021-10 to 2026-04; ~1,380 since 2025-06 (the post-cutoff
  slice for `gemini-2.5-flash`, training cutoff January 2025).
- **Outputs**: `p_play` in [0, 1], `expected_minutes_share` in [0, 1] (fraction of
  the player's recent per-game minutes), and a short rationale kept for debugging
  only, never used as a feature.
- **Labels** (evaluation only): `played` = minutes > 0 in that game's box score;
  `minutes_share` = minutes / trailing-10-game mean minutes. Out rows are excluded
  from the task but included in the history retrieved for later rows.

## 3. Prerequisite: label backfill

Per-game player minutes are not stored anywhere in the repo (`player_stats_cache`
holds rolling averages only). Add `scripts/backfill_player_game_logs.py`:

- Source: nba_api `PlayerGameLog` per (player, season), which is a few hundred
  calls per season, not one per game. Reuse the rate-limit and incremental-save
  patterns from `scripts/backfill_player_stats.py`.
- Store in a new file `data/raw/availability.sqlite`, table
  `player_game_log(player_id, game_id, game_date, team_id, minutes, started)`.
- Name resolution: `player_injuries` keys on `player_name`; reuse the resolver
  `src/matchups/injury_layer.py` already uses. Log unresolved names, do not guess.

## 4. Retrieval (the RAG part)

Every query builds a context pack for one (game_date D, team, player). Every
lookup is bounded by `game_date < D`; the D-1 rule from the on/off work applies to
any nba_api-derived as-of table.

1. **Own injury history**: previous listings for this player (status, reason, and
   whether they played, from `player_game_log`), with same-reason listings marked.
2. **Minutes trend**: last 10 games' minutes, days since last game played,
   games missed in the last 30 days.
3. **Importance**: `player_importance` as of D-1.
4. **Game context**: rest days and back-to-back for the team, standing and
   season-motivation state, Elo difference vs. the opponent, all from the
   existing feature tables.
5. **Similar-reason base rate** (phase 3): nearest injury-reason strings across
   all players by embedding similarity, with their play rate. Start with
   normalized exact matching on the reason text; add embeddings only if phase 1
   shows the reason text matters.

The same context pack, as numbers, is the input to the tabular baseline. That is
what makes the comparison honest: the LLM and the baseline see the same facts.

## 5. Baselines (must be beaten)

1. **Status prior**: global play rate per status.
2. **Status × severity bucket**: play rate per (status, severe/moderate/minor
   from `classify_severity`).
3. **Tabular model**: logistic regression and a small CatBoost on the numeric
   context pack, trained on seasons before the evaluation season (expanding,
   same spirit as the CV folds, but on the availability task only).

## 6. LLM estimator

- Provider: the existing Gemini client and config fields (`llm_model`,
  `api_calls_per_minute`, `parallel_workers`); structured JSON output as in
  `src/news_scraping/extractors/llm_extractor.py`. Swapping provider is one file.
- **Anonymized prompt**: no player, team, or opponent names, no dates. Positions,
  numeric history, and the injury reason text only. Reason text can still hint at
  identity in rare cases; accepted as noise.
- **Cache**: `llm_cache(prompt_hash, model, response_json, created_at)` in
  `availability.sqlite`, so reruns and re-evaluations cost nothing.
- Cost: ~7,300 calls of ~1.5k tokens on a Flash-class model is negligible.

## 7. Evaluation

**Intrinsic** (`scripts/run_availability_eval.py`, writes
`outputs/availability_eval.csv`, one row per method × slice):

- Brier, log-loss, AUC for `p_play`; reliability curve in deciles.
- MAE of `expected_minutes_share` among players who played.
- Slices: status, season, pre-cutoff vs. post-cutoff.
- **Memorization check**: named vs. anonymized prompt on pre-cutoff seasons. If
  named beats anonymized by more than the season-to-season noise, memorization is
  confirmed and only anonymized, post-cutoff numbers are reported as the result.

**Extrinsic** (the real gate): new config section `availability_agent` with
`enabled: false`, `source: tabular | llm`. When enabled, `compute_team_deficit`
uses `1 - p_play` (times severity) as the absence weight for Questionable and
Doubtful players instead of 0 and `doubtful_weight`. Run baseline vs. treatment
under `--protocol cv`, log to `outputs/experiments_v2.csv`, write up in
`docs/EXPERIMENTS.md`. Expect the effect to sit in folds 3-5 only; folds 1-2 have
`has_injury_data = 0` for most rows.

Also run the extrinsic ablation with `source: tabular`. If the tabular estimate
helps and the LLM does not add to it, the feature can still be adopted without the
LLM, and the LLM part is logged as a failed experiment.

## 8. Module layout

```
src/availability/
  labels.py        # join player_injuries rows to player_game_log, build labels
  retrieval.py     # context pack per (date, team, player); point-in-time bounded
  baselines.py     # status prior, status×severity, tabular models
  llm_estimator.py # anonymized prompt, structured output, cache
  pipeline.py      # batch run: retrieve → estimate → store p_play per row
scripts/backfill_player_game_logs.py
scripts/run_availability_eval.py
tests/test_availability_retrieval.py   # every retrieved fact predates the query date
tests/test_availability_prompt.py      # anonymized prompt contains no names/dates
```

## 9. Phases and gates

| Phase | Work | Gate to continue |
|---|---|---|
| 0 | Label backfill, retrieval, baselines, eval script. No LLM. | Labels join for ≥95% of uncertain rows; baselines reproduce sensible priors. |
| 1 | LLM estimator, anonymized, cached; intrinsic eval + memorization check. | LLM beats the tabular baseline on Brier in the post-cutoff slice. Otherwise log as failed, keep tabular. |
| 2 | Config section, `compute_team_deficit` integration, 5-fold CV ablation for tabular and LLM sources. | Composite improves on folds 3-5 without regressing 1-2. |
| 3 (optional) | Embedding retrieval over reason text; agentic variant where the LLM calls retrieval tools itself instead of receiving a fixed pack. Compare fixed-pack vs. agentic on the same eval. | Only if phase 1 passed. |

## 10. Hard rules

- No label ever appears in a prompt or a retrieval result.
- All retrieval is bounded by `game_date < D`; as-of tables use D-1.
- CV folds, harness, and metric are untouched; the extrinsic test uses them as is.
- Failed twice at the phase 1 gate: log as failed, move on.
