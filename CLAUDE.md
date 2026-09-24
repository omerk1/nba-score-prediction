# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

NBA game score prediction via CatBoost gradient boosting, prioritizing point-differential accuracy over absolute score proximity. No installed package (no `setup.py`/`pyproject.toml`) — everything runs from the repo root.

## Environment

- Use `venv/bin/python3` explicitly for every command (training, scripts, tests) — never a bare `python`/`python3`/`pytest`. There's no activation assumed, and a bare interpreter in a subprocess won't reliably resolve to this venv.
- Actual dev venv is Python 3.12.7 (README says `>=3.9` — that's a floor, not the tested version).
- Only `catboost` is installed for modeling — no `lightgbm`/`xgboost`, despite the README mentioning "CatBoost/LightGBM".
- `.env` with `GOOGLE_API_KEY` is only needed for the LLM injury-scorer mode (`injury_features.scorer: llm`); the default formula-based scorer needs no API key.

## Testing

`venv/bin/python3 -m pytest tests/ -q` from repo root. No `conftest.py` — tests import via absolute paths (`from src.feature_engineering...`), which only resolves with repo root on `sys.path`/as cwd.

## Config-driven architecture

`configs/config.yaml` is the single source of truth, validated by pydantic schemas in `src/utils/config_loader.py`. Every experimental module gets its own config section with an `enabled: bool = False` field (see `style_matchup`, `on_off_splits`, `season_motivation`).

## Ablation-gated feature workflow (required)

Any new experimental feature must ship disabled by default and go through a real ablation — `train_model.py` baseline vs. treatment, ideally a 5-fold expanding-window CV (`--protocol cv`) — before its flag is flipped to `true`. Every `train_model.py` run appends a row to `outputs/experiments_v2.csv`, the shared cross-feature ablation log; don't lose or overwrite prior rows. (`outputs/experiments.csv`, the old 16-column schema, is a frozen historical snapshot — never written to going forward.) `docs/BACKLOG.md` tracks feature status; `docs/*_log.md` files hold the real validation write-ups behind each adoption/rejection decision.

## Leakage safety

Features must be point-in-time (pre-game state only) — the recurring pattern is `shift(1)`/`merge_asof` before use, checked repeatedly throughout `src/feature_engineering/`. Verify this explicitly for any new feature touching historical game data.

## Documentation style

Keep `.md` docs and logs concise — state findings and numbers tersely, don't narrate every step taken. Grouping related work under a clear, self-explanatory name (e.g. "modeling improvements", "feature enrichments") is fine and encouraged. Never use bare internal codes (e.g. "A7", "B4", "Round 3") anywhere outside `docs/BACKLOG.md` itself — not in other docs, not in code comments/docstrings, not in commit messages. Elsewhere, describe the technical reason directly.

## Git conventions

- Branches: `type/description` (e.g. `feature/season-motivation`, `chore/prune-ablation-csvs`).
- Commits/PR titles: `type: description (#N)` or `type(scope): description (#N)` (types seen: feat, fix, chore, docs, refactor, analysis).
- No `Co-Authored-By` trailer in commit messages.

## Project Rules (ML experimentation)

**Status: the expanding-window CV harness is implemented** (`src/evaluation/cv_harness.py`, folds in `configs/config.yaml`'s `cv.folds`, 5 folds oldest → newest, mechanically validated by `validate_fold_definitions`). `train_model.py --protocol cv` runs it; `--protocol single_split` (default) still runs today's one fixed split from `datasets_loading`'s dates — both go through the same `run_split` code path. `docs/EXPERIMENTS.md` and `results/sessions/` (for the session-leaderboard rules below) still don't exist yet — create on first use.

### Running experiments
- One experiment = one command: `venv/bin/python3 train_model.py --run-name <experiment_id> --notes "..." [--protocol single_split|cv]`.
- Every run is logged, one row per run, per the leaderboard rules below. No run without a row.
- Numbers → CSV only. Interpretation → `docs/EXPERIMENTS.md` decision log (doesn't exist yet — create on first use), referenced by experiment_id.

### Leaderboards & research sessions
- `outputs/experiments_v2.csv` is the master registry (the CV-protocol schema — `val_score_mean`, `val_score_per_fold`, `test_score_mean`, `protocol`, `session_id`, plus the original per-metric columns). `outputs/experiments.csv` (the old 16-column schema) is a frozen historical snapshot, seeded into `experiments_v2.csv` once (`scripts/migrate_experiments_schema.py`) with those 5 new columns empty — pre-CV-harness rows' own naive-baseline values were never recorded, so their composite score can't be retroactively computed. Autonomous/research sessions never append to `experiments_v2.csv` directly during the run.
- Each research session gets a session_id (`YYYYMMDD_HHMM_<slug>`, e.g. `20260804_1430_champion-cv-baseline` — timestamp for free uniqueness/sorting, a short freeform slug so `results/sessions/` and the `docs/EXPERIMENTS.md` log stay scannable without opening files; keep the slug to 2-4 words since experiment IDs are prefixed with it) and logs every run to `results/sessions/<session_id>.csv` (`results/sessions/` doesn't exist yet — create on first use), same schema as `experiments_v2.csv` + `session_id`.
- At session end, append to `experiments_v2.csv`: (a) the session's best row by mean validation score under full CV, and (b) any other row that beats the current champion. Session CSVs are archived, never deleted.
- Manual one-off experiments run interactively may log directly to `experiments_v2.csv` (already how `train_model.py` works today).

### Hard constraints (never violate)
- Evaluation = expanding-window CV over seasons (`--protocol cv`). Per fold: train = from `datasets_loading.train_start_date` through that fold's `train_end_date`, validation = the next season, test = the season after. Folds roll forward one season at a time (oldest → newest); defined in `configs/config.yaml`'s `cv.folds`, never inline.
- Model selection and tuning use aggregated VALIDATION scores only (mean across folds, also log per-fold). Test-fold scores are logged but never used to choose between experiments — consult them only for a final champion evaluation.
- Leaderboard rows record: `val_score_mean`, `val_score_per_fold`, `test_score_mean`, `protocol`, `session_id` (`outputs/experiments_v2.csv`'s schema).
- No feature may use post-tipoff information; rolling features at fold boundaries must be computed only from data available at that point in time.
- Never modify the CV fold definitions, eval harness, or metric computation without asking.

### Metric
- Primary score (minimize): `(diff_mae / naive_diff_mae) + 0.5 * (total_mae / naive_total_mae)` — both terms normalized against that SAME split/fold's own naive rolling-baseline values (`src/evaluation/cv_harness.naive_baseline_metrics`, recomputed fresh per split/fold, never a fixed constant), so the two MAEs (different typical magnitude, same units) combine without an arbitrary scale fix. `diff_mae` (point-differential MAE) dominates; `total_mae` counts at half weight (`compute_composite_score`). Judged on validation only.

### Process
- Branch `experiments`, one commit per experiment (message = experiment_id).
- After each run: append to `docs/EXPERIMENTS.md` decision log (hypothesis → result → conclusion → next). At session end: append a session summary (session_id, what was explored, what was promoted, what was dropped and why).
- Cheap screening runs may use the last 3 folds only; full CV required before an experiment is promoted or declared a new best.
- Failed twice → log as failed, move on.
- Preprocessing changes go through the central pipeline only, no per-feature ad hoc handling.

## Phase history

**Rolling-window representation-enrichment phase — closed 2026-08-24**
(full detail: `docs/NEXT_PHASE_SESSIONS.md`, marked CLOSED at the top of
that doc). Went through the existing feature families (rolling box-score
aggregates, Elo, opponent quality, a retrospective opponent-adjustment
idea) looking for richer representations than the current mean/point-in-
time values capture. Result: 2 features adopted out of 9 tested
candidates — a venue-blind overall-form feature and an Elo momentum
feature — taking the live feature set from 127 to 148 columns. Cumulative
full-CV `val_score_mean` improved Δ−0.0018, and all 4 `market_benchmark`
metrics improved; the model-vs-Polymarket gap narrowed roughly 8-12% per
metric but did not close. The method itself (testing richer
representations of already-existing families) is judged low-expected-
value going forward on this feature set: a 22% hit rate, and several of
the rejected candidates — including the one that initially looked like
the phase's strongest result — needed multiple dedicated diagnostic
sessions before they could be correctly rejected rather than failing
cleanly on a first pass.

One thing from this phase is logged but not scheduled, for whoever picks
model-quality work back up: a set of scoped-but-untested creative feature
ideas (explicit trend/slope over rolling windows, distributional-shape
features, asymmetric style-clash features, lineup-stability/continuity,
referee-tendency data — `docs/NEXT_PHASE_SESSIONS.md`'s backlog section)
that are candidates to consider, not a queue to work through by default.

The phase's other loose end, play-by-play data, is no longer open: it was
built out in full 2026-09-17/19 (10,739 games, ~2.1M possessions,
validated parser — points reconcile with the box score on every game) and
**rejected**. 45 candidate aggregates were pre-screened on block-to-block
persistence; the one promoted to full CV proved a single-fold artifact,
and the later noise-floor measurement downgraded that verdict to
*unresolvable* — reseeding the champion moved the score more than the
feature did. `pbp.enabled` stays `false`, the possession table kept as
reusable infrastructure. The transferable finding is that **within-team
persistence is ≈0 across all 45 candidates**: what persists is
between-team spread, which Elo and rolling margin already encode
(`docs/EXPERIMENTS.md`'s `pbp_net_rtg_luckadj` entry,
`docs/features/pbp_possessions_log.md`).

**LLM/agentic-component investigation — closed 2026-09-23**
(full detail: `docs/LLM_COMPONENT_OPTIONS.md`, marked CLOSED at the top of
that doc). Explored whether an LLM component could add value anywhere in
this pipeline — an availability estimator for uncertain injury listings
(three prompt variants, isotonic calibration, stacking, isolated
chain-of-thought), semantic retrieval over injury-reason text (two variants,
a full tuning/sealed split), and a direct post-hoc adjustment of the trained
model's own predictions. All six attempts rejected, each on its own clean
evidence, with one consistent mechanism: the ceiling was the information in
the engineered features, not how it was read, and results got monotonically
worse as more prompting sophistication was added. Two real, unrelated
correctness bugs were found and fixed along the way (an injury-report
team-name mismatch silently dropping every Clippers listing since 2021;
most injury listings attached to the wrong game via the PDF report's own
publish date) — kept regardless of the LLM outcome, though a downstream
ablation wiring the corrected dates into the live injury feature did not
clear its own screen either (`docs/features/injury_pdf_extraction_scope.md`
phase D).

Not scheduled for revisit on this codebase: re-running variants of "read
engineered features, output a number" is a closed question here, not an
open one — this project's data is exactly the case (dense, well-labeled,
tabular, years of history) where a tuned gradient booster already wins and
an LLM has nothing to add. All of it (PR #70, merged 2026-09-24) is on `main`
now, not a separate branch: the box-score availability labels and tabular
estimator (non-LLM, does beat its baselines) plus the injury-PDF extraction
fixes, both still gated behind disabled flags pending their own adoption
decisions.
