# Availability Agent — Scope

Status: scoped 2026-09-17. Phase 0 (labels, retrieval, baselines) complete the
same day. Phase 1 complete 2026-09-18: **the LLM estimator was rejected** — it
lost to every baseline including the status prior, and added nothing on top of
the tabular model. The tabular estimator survives and carries into phase 2. Both
result sections are at the end of this file. Ships disabled by default and goes
through the ablation-gated workflow in CLAUDE.md before any flag flips.
Companion: `docs/LLM_COMPONENT_OPTIONS.md` (why this option).

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

---

## Phase 0 results (2026-09-17)

Branch `feature/availability-agent`. Built: `scripts/backfill_player_game_logs.py`
(140k player-game rows, 2021-22 to 2025-26, 10 API calls), `src/availability/`
(`labels`, `names`, `retrieval`, `baselines`, `db`), `scripts/run_availability_eval.py`
(writes `outputs/availability_eval.csv`), `tests/test_availability.py`, and the
`availability_agent` config section (disabled).

**Finding that changed the design**: `player_injuries.game_date` is the PDF
report date, and 99% of uncertain listings describe the team's game on the next
day (details and the effect on the live injury feature: `docs/PIPELINE_AUDIT.md`,
2026-09-17 addendum). Labels therefore use report date D -> game on D+1. Every
retrieval fact is bounded to `game_date < D`.

**Gate 1 (labels)**: 7,304 uncertain listings; 829 G League rows and 50 rows
with no team game on D+1 dropped; 99 unresolved names (1.5%). 6,326 labeled rows,
resolved share 98.5% (gate: 95%). Play rate: Questionable 53%, Doubtful 6%. The
current fixed weights (Questionable = plays, Doubtful = 80% absent) are wrong on
half of all Questionable rows.

**Gate 2 (baselines)**, expanding by season, 4 evaluated seasons, n = 4,766:

| estimator | Brier pooled | Brier post-cutoff | AUC | ECE |
|---|---:|---:|---:|---:|
| status prior | 0.2249 | 0.2250 | 0.607 | 0.021 |
| status x severity prior | 0.2251 | 0.2250 | 0.604 | 0.024 |
| logistic (context) | 0.2225 | 0.2223 | 0.665 | 0.038 |
| catboost (context) | 0.2204 | 0.2150 | 0.680 | 0.045 |

The tabular models beat the priors on every slice; the gain is modest because
Questionable is close to a coin flip. Strongest single facts by correlation with
the label: the player's own play rate under the same reason text (0.18), minutes
in their last game (0.14), recent minutes and importance (0.12), own overall play
rate on prior uncertain listings (0.13). The bar for the LLM (phase 1) is the
catboost row on the post-cutoff slice: Brier 0.2150.

**Coverage caveat**: back-to-back second nights are almost absent from the
uncertain pool (0.3% vs. 16% of team-games), because the 11PM report on D rarely
carries next-day entries for teams that played on D. Any downstream feature
inherits this gap; it is a data-collection limit, not a modeling choice.

---

## Phase 1 results (2026-09-18) — LLM estimator rejected, tabular estimator kept

Built: `src/availability/prompt.py` (anonymized and named prompt variants),
`src/availability/llm_estimator.py` (Gemini, JSON output, temperature 0,
responses cached in `availability.sqlite`), `src/availability/llm_derived.py`
(isotonic calibration and a stacked model), `scripts/attach_llm_predictions.py`,
`scripts/compare_availability_estimators.py`, `tests/test_availability_llm.py`.
9,532 calls, zero failures, all cached; the whole eval re-runs offline.

**Result: the LLM loses to every baseline, including the status prior.** Brier
on the evaluated rows (n = 4,766 pooled, 980 post-cutoff):

| estimator | pooled | post-cutoff | AUC | ECE |
|---|---:|---:|---:|---:|
| catboost (facts only) | 0.2204 | 0.2150 | 0.680 | 0.045 |
| logistic (facts only) | 0.2225 | 0.2223 | 0.665 | 0.038 |
| status prior | 0.2249 | 0.2250 | 0.607 | 0.021 |
| llm, calibrated | 0.2280 | 0.2310 | 0.633 | 0.034 |
| llm, raw | 0.2407 | 0.2472 | 0.630 | 0.097 |

Paired bootstrap over the same rows, 10,000 resamples, post-cutoff slice: the
raw LLM is worse than the status prior by 0.0222 (95% CI 0.0133 to 0.0316) and
worse than catboost by 0.0322 (0.0217 to 0.0428). Calibrated it is still worse
than the status prior by 0.0060 (0.0016 to 0.0105). Every gap is significant.

**Why it loses**: not ranking, calibration. The raw LLM's AUC (0.630) beats the
status prior's (0.607), so it does order rows by risk. But it states low
probabilities far more strongly than reality supports: rows where it said 0.05
played 14% of the time, and rows where it said 0.21 played 47%. Isotonic
calibration cut the calibration error from 0.097 to 0.034 and recovered most of
the Brier gap, but ranking alone was never enough to pass the prior.

**Does it add anything the facts miss? No.** Adding the LLM's probability as a
feature to catboost changed nothing measurable: 0.2216 vs. 0.2204 pooled,
difference 0.0012 (CI -0.0001 to 0.0027), not significant, and the same at
post-cutoff. Its probabilities correlate 0.63 with catboost's, so it is mostly
re-deriving the same signal less precisely.

**Memorization check: negative, and worth stating plainly.** The named prompt
(player, team, and date restored) scored *worse* than the anonymized one on the
pre-cutoff seasons, 0.2408 vs 0.2390. Identity gives the model no recall
advantage on these games, so the anonymized numbers are trustworthy and the
anonymization costs nothing.

**Decision**: the LLM path is rejected per the phase 1 gate (two attempts: raw,
then calibrated and stacked). `availability_agent.source` stays `tabular`.
Phase 2's CV ablation proceeds with the catboost estimator, which does beat the
status prior significantly (post-cutoff 0.0100 better, CI 0.0027 to 0.0171).
The LLM code stays in the tree, disabled, because it is what makes the negative
result reproducible.

**Honest read on the wider question**: on a task with ~4,000 labeled training
rows and a fact set that is already numeric, a gradient-boosted model on those
facts beats a frontier LLM reading the same facts as prose. The LLM's one
potential edge here was the free-text injury reason, and the same-reason play
rate the retrieval layer computes from history already captures it.

### Third attempt: few-shot examples and a reasoning budget (2026-09-18)

The first two attempts used a bare prompt with reasoning disabled, so "the
prompt was too constrained" was a fair objection. Tested directly: 16 worked
examples drawn only from training seasons with their real outcomes, plus a
2048-token reasoning budget, same anonymized facts, same 980 post-cutoff rows.

| variant | brier | AUC | ECE |
|---|---:|---:|---:|
| catboost (facts only) | 0.2150 | 0.698 | 0.049 |
| status prior | 0.2250 | 0.602 | 0.032 |
| llm, bare prompt, no reasoning | 0.2472 | 0.630 | 0.097 |
| llm, 16 examples + reasoning | **0.2942** | **0.584** | 0.219 |

**It got worse, and ranking degraded too.** AUC fell from 0.630 to 0.584, below
the status prior, so this is not a calibration problem that post-processing
could repair. The reliability curve shows why: rows it scored 0.04 played 23% of
the time and rows it scored 0.88 played 53%, wildly overconfident at both ends.
The worked examples carry binary outcomes (PLAYED / DID NOT PLAY), and the model
imitated that binary form instead of estimating a rate. On a task whose base
rate is near a coin flip, confident imitation is the worst possible failure.

**Conclusion after three attempts**: the ceiling is the information in the facts,
not the prompt. The tabular model extracts more from the same facts than a
frontier LLM does, and giving the LLM more freedom moved it further away.
