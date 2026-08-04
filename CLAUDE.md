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

Any new experimental feature must ship disabled by default and go through a real ablation — `train_model.py` baseline vs. treatment, ideally a 5-fold expanding-window CV (`--protocol cv`) — before its flag is flipped to `true`. Every `train_model.py` run appends a row to `outputs/experiments_v2.csv`, the shared cross-feature ablation log; don't lose or overwrite prior rows. (`outputs/experiments.csv`, the old 16-column schema, is a frozen historical snapshot — never written to going forward.) `docs/backlog.md` tracks feature status; `docs/*_log.md` files hold the real validation write-ups behind each adoption/rejection decision.

## Leakage safety

Features must be point-in-time (pre-game state only) — the recurring pattern is `shift(1)`/`merge_asof` before use, checked repeatedly throughout `src/feature_engineering/`. Verify this explicitly for any new feature touching historical game data.

## Documentation style

Keep `.md` docs and logs concise — state findings and numbers tersely, don't narrate every step taken. Grouping related work under a clear, self-explanatory name (e.g. "modeling improvements", "feature enrichments") is fine and encouraged. Never use bare internal codes (e.g. "A7", "B4", "Round 3") anywhere outside `docs/backlog.md` itself — not in other docs, not in code comments/docstrings, not in commit messages. Elsewhere, describe the technical reason directly.

## Git conventions

- Branches: `type/description` (e.g. `feature/season-motivation`, `chore/prune-ablation-csvs`).
- Commits/PR titles: `type: description (#N)` or `type(scope): description (#N)` (types seen: feat, fix, chore, docs, refactor, analysis).
- No `Co-Authored-By` trailer in commit messages.

## Project Rules (ML experimentation)

**Status: the expanding-window CV harness is implemented** (`src/evaluation/cv_harness.py`, folds in `configs/config.yaml`'s `cv.folds`, 5 folds oldest → newest, mechanically validated by `validate_fold_definitions`). `train_model.py --protocol cv` runs it; `--protocol single_split` (default) still runs today's one fixed split from `datasets_loading`'s dates — both go through the same `run_split` code path. `EXPERIMENTS.md` and `results/sessions/` (for the session-leaderboard rules below) still don't exist yet — create on first use.

### Running experiments
- One experiment = one command: `venv/bin/python3 train_model.py --run-name <experiment_id> --notes "..." [--protocol single_split|cv]`.
- Every run is logged, one row per run, per the leaderboard rules below. No run without a row.
- Numbers → CSV only. Interpretation → `EXPERIMENTS.md` decision log (doesn't exist yet — create on first use), referenced by experiment_id.

### Leaderboards & research sessions
- `outputs/experiments_v2.csv` is the master registry (the CV-protocol schema — `val_score_mean`, `val_score_per_fold`, `test_score_mean`, `protocol`, `session_id`, plus the original per-metric columns). `outputs/experiments.csv` (the old 16-column schema) is a frozen historical snapshot, seeded into `experiments_v2.csv` once (`scripts/migrate_experiments_schema.py`) with those 5 new columns empty — pre-CV-harness rows' own naive-baseline values were never recorded, so their composite score can't be retroactively computed. Autonomous/research sessions never append to `experiments_v2.csv` directly during the run.
- Each research session gets a session_id (`rs_YYYYMMDD_n`) and logs every run to `results/sessions/<session_id>.csv` (`results/sessions/` doesn't exist yet — create on first use), same schema as `experiments_v2.csv` + `session_id`. Experiment IDs are prefixed with the session_id.
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
- After each run: append to `EXPERIMENTS.md` decision log (hypothesis → result → conclusion → next). At session end: append a session summary (session_id, what was explored, what was promoted, what was dropped and why).
- Cheap screening runs may use the last 3 folds only; full CV required before an experiment is promoted or declared a new best.
- Failed twice → log as failed, move on.
- Preprocessing changes go through the central pipeline only, no per-feature ad hoc handling.
