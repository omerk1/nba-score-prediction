# EXPERIMENTS.md

Decision log and research agenda for the ablation-gated feature workflow (CLAUDE.md's "Project Rules (ML experimentation)" section). Numbers → `outputs/experiments_v2.csv` / `results/sessions/<session_id>.csv`. Interpretation → here, referenced by run_name / session_id.

## 1. Ground truth

**Composite score** (minimize): `diff_mae/naive_diff_mae + 0.5 * total_mae/naive_total_mae`, both terms normalized against that same split's own freshly-recomputed naive rolling-baseline. The naive baseline's own score is **always exactly 1.5** for any fold — a mathematical identity of the formula (evaluate it on the naive predictor against itself: both ratios are 1, so `1 + 0.5*1 = 1.5`), not something that needs a separate run to establish. Every score below is judged against that fixed 1.5 floor.

**CV protocol**: 5 expanding-window folds, oldest → newest, `configs/config.yaml`'s `cv.folds`, mechanically validated by `validate_fold_definitions` (fold ordering, no overlap, no fold's val/test predating an earlier fold's own training window). Fold5 = today's `--protocol single_split` boundaries exactly.

**Champion baseline** (`champion_cv_baseline`, `outputs/experiments_v2.csv`): `style_matchup.raw_features_enabled=true`, `preferred_opponent_delta_enabled=true`, `style_matchup.enabled=false`.

| fold | val_score | test_score |
|---|---:|---:|
| 1 | 1.4407 | 1.3868 |
| 2 | 1.3876 | 1.3936 |
| 3 | 1.3804 | 1.3720 |
| 4 | 1.3579 | 1.3787 |
| 5 | 1.3585 | 1.3308 |
| **mean** | **1.3850** | **1.3724** |

**Fold-1 gap, called out explicitly**: fold1's val_score (1.4407) is ~4–6% worse than every other fold (1.36–1.39 range) — the smallest training window (2018-10-16 → 2020-08-14) is the weakest fold by a clear margin, not noise. Section 2's fold-1 breakout and section 3's diagnostics/experiments are largely about understanding and addressing this gap.

## 2. Family inventory findings

5-fold mean, from the family-importance inventory (`scripts/run_family_importance.py`, session `20260805_1611_family-inventory`). Ranked by permutation importance (destroy the family, how much does composite score worsen) — CatBoost's own split-importance is shown alongside since the two disagree in informative ways.

| Rank | Family | Perm Δ (mean) | CatBoost share | Fold1 : later ratio |
|---|---|---:|---:|---:|
| 1 | `style_fingerprint_features` | 0.0342 | 27.4% | 1.13 |
| 2 | `elo_features` | 0.0303 | 5.8% | 0.74 |
| 3 | `style_features` | 0.0174 | 20.1% | 1.06 |
| 4 | `matchup_features` | 0.0115 | 8.9% | 0.94 |
| 5 | `rolling_features` | 0.0077 | 12.0% | 1.21 |
| 6 | `opponent_quality_features` | 0.0022 | 9.8% | 0.87 |
| 7 | `injury_features` | 0.0017 | 1.4% | **0.00** |
| 8 | `travel_features` | 0.0014 | 3.7% | 1.19 |
| 9 | `h2h_features` | 0.0011 | 3.1% | 0.92 |
| 10 | `home_advantage_features` | 0.0007 | 4.4% | 0.89 |
| 11 | `rest_features` | 0.0007 | 1.4% | 1.06 |
| 12 | `basic_features` | 0.0004 | 1.75% | 0.21 |
| 13 | `season_motivation_features` | **−0.0003** | 0.27% | 0.56 |

**Signal is concentrated in the top 6**: they account for ~95% of total permutation delta (0.103 of 0.109); the bottom 7 combined are ~5%, and one of them is negative.

**Elo is the efficiency standout**: 6th by CatBoost share but 2nd by permutation cost. CatBoost under-splits on it — likely correlated with rolling/style as another "team strength" proxy, so a split-selection algorithm doesn't need it when substitutes are available — but the model can't fully recover once it's actually removed. The mirror case is `opponent_quality_features` (4th by share, 6th by permutation): real CatBoost usage, almost no permutation cost — "used but not predictive," likely redundant with rolling/style rather than adding anything of its own.

**`style_fingerprint_features` is the earning-but-verify dominator**: 27.4% share, also the single highest permutation cost — not just a CatBoost artifact, genuinely earning its place. Flagged for a leakage re-audit anyway (section 3.1) purely because of magnitude: its point-in-time construction was verified once, in isolation, before this session's CV-fold-boundary work existed — that verification hasn't been re-run against fold boundaries specifically.

**Fold-1 zeros/near-zeros, not all the same story**: `injury_features` is a literal, hard zero on fold1 — I believe this is a real data-coverage gap (injury data likely doesn't exist that far back), not a modeling weakness; section 3.1 makes confirming that (and deciding the fix) a gate before treating it as a feature-quality problem. `basic_features` (`season_progress`, ratio 0.21) and `season_motivation_features` (ratio 0.56) are both substantially reduced on fold1 too, but neither is a hard zero — plausible genuine data-hunger (within-season position needs more of the season observed; motivation signals need enough season-progression variance to matter).

**`injury_features`' 0.0017 mean permutation delta (rank 7) is understated, not weak** — it's a 5-fold average that blends a forced 0.0 on fold1 (no data at all, not a real predictive-value reading) with whatever the family actually contributes on folds 2–5. The true value in folds where the data exists is higher than 0.0017 suggests. Do not read this number as a prune signal, and don't group it with the genuinely weak tail (`travel`/`h2h`/`home_advantage`/`rest`) on the strength of the aggregate alone — section 3.2's E4 gets the real per-fold picture instead of relying on this diluted mean.

**`season_motivation_features` is the clear prune candidate**: the only family with a negative permutation delta — removing it very slightly helps on average, despite having passed its own single-metric ablation test at adoption time (`preferred_opponent_delta_treatment`, single_split). Not a contradiction of that earlier result, just a different, more holistic lens now that the full feature set + CV are both in play.

## 3. Research agenda

Every item below: hypothesis, config change, protocol, expected effect, effort, risk. Screening protocol note — "cheap 3-fold" (folds 3–5, per CLAUDE.md) only applies when fold1 isn't the object of study; fold1-specific questions need fold1 in the run, so cheap-3-fold does not apply to those.

### 3.1 Diagnostics (must-run gates — before anything downstream treats their targets as a modeling problem)

**D1. Injury data-coverage check.**
- Hypothesis: `injury_features.sqlite` doesn't cover fold1's train/val window at all (already know it starts 2021-10-19; fold1's val ends 2021-05-16) — the fold1 zero is a coverage gap, not a signal-quality problem.
- Action: query `injury_features.sqlite` per season for row counts / completeness, not just the min date (confirm there isn't a partial-coverage gap somewhere else in the covered range too).
- Decision this gates: confirms coverage before E4 (section 3.2) empirically compares the actual handling strategies rather than picking one by inspection. Not a training run itself — output is a coverage confirmation feeding E4.
- Effort: low (one query + a written decision). Risk: none. Screening: n/a.

**D2. `style_fingerprint_features` leakage re-audit.**
- Hypothesis: the family's dominance (27.4% share, highest permutation cost) is genuine, not a fold-boundary leak that the original (pre-CV-harness) leakage verification wouldn't have caught.
- Action: reuse this session's empirical audit technique (compare truncated-history vs. full-history fingerprint computation for a sample of games right at each fold's `test_start_date`) — confirm the first test-period game for a team only reflects fingerprint history strictly before that fold's own boundary.
- Expected effect: pass (no leak) is the expected/likely outcome, given the underlying `.shift(1)` construction; if it fails, this supersedes every other item in this section.
- Effort: low-medium (a script, no retraining). Risk: none (read-only). Screening: n/a.

### 3.2 High-value experiments (pending D1/D2 passing)

**E1. Expand Elo.**
- Hypothesis: current `k_factor=11.02`/`season_regression=0.522` were tuned via `scripts/tune_elo.py` under the old single_split protocol, predating the CV harness and never re-validated per-fold. Given Elo's fold1 weakness (ratio 0.74) and its efficiency (high permutation cost relative to modest CatBoost share), retuning — or specifically testing whether a different `season_regression` narrows the fold1 gap — is worth it.
- Protocol: **full 5-fold, not cheap-3-fold** (fold1 is the whole point).
- Expected effect: fold1 val_score improves without materially moving folds 2–5 (guardrail: must hold on 2–5, not just fix fold1).
- Effort: medium (small grid/Optuna sweep via existing `tune_elo.py` infra, re-run under CV instead of single_split). Risk: low (isolated to `elo_features` config).

**E2. Prune the dead tail.**
- Hypothesis: `season_motivation.preferred_opponent_delta_enabled: false` performs flat-to-marginally-better than `true` under full CV (matches its negative permutation delta).
- Protocol: cheap 3-fold screen (folds 3–5) first; full 5-fold only if the cheap screen looks promising enough to consider promoting.
- Expected effect: val_score_mean roughly flat or very slightly improved.
- Effort: very low (single flag flip, `scripts/run_fingerprint_ablation.py`'s pattern reused for this flag instead). Risk: very low.
- Watch list, not bundled into E2's own test (avoid conflating simultaneous changes): `rest_features`, `home_advantage_features` — both near-zero permutation cost though not negative. Worth their own single-flag checks later, not this run.

**E3. Refine top families — collinearity check.**
- Hypothesis: `style_fingerprint_features` and `style_features` (both broad "team style/quality" proxies) and/or `opponent_quality_features` (the "used but not predictive" family from section 2) have real overlap — consolidating could simplify the model without losing signal, or confirm `opponent_quality_features` is safe to prune alongside `season_motivation_features`.
- Action: correlation/VIF analysis between the three families' engineered columns — diagnostic only, no retraining required first.
- Effort: medium (a script). Risk: none for the diagnostic; any resulting drop-a-family experiment inherits E2's protocol.

**E4. Injury missing-data handling** (depends on D1; explicitly not part of E2's prune group — `injury_features`' aggregate score is understated, not weak, see section 2).
- Hypothesis: how missing injury data is represented matters more than whether the feature itself is weak. Three variants, compared directly:
  1. Native CatBoost NaN handling (status quo — missing rows pass through untouched).
  2. Explicit availability-indicator feature (`home/away_team_injury_data_available`) alongside the existing columns.
  3. Folds-start-at-2021 — restrict the comparison to the date range where coverage actually exists (drops/truncates fold1, per D1's coverage findings).
- Protocol: each variant screened only on the folds where it's actually comparable to the others (variants 1–2 run across all 5 folds; variant 3 is inherently restricted to the covered range, so its comparison is scoped to whichever folds overlap). Not a cheap-3-fold screen — this is fold1-relevant by construction.
- Expected effect: may lift fold1 specifically; watch per-fold, not just the mean (guardrail applies — no promotion on a fold1-only improvement).
- Effort: medium (indicator feature is a small `feature_builder.py` addition; the fold-restriction variant reuses existing fold-filtering, no harness changes). Risk: low.

### 3.3 Model-agnostic axes (broader, lower immediate priority — not driven by the family inventory directly)

- **Target reformulation**: retarget the model to predict `[diff, total]` directly instead of `[home, away]` (currently `MultiRMSE` on `PTS_home`/`PTS_away`), to align the training loss with what the composite metric actually rewards (diff-dominant, total at half weight).
- **Hyperparameter tuning**: CatBoost's `depth`/`learning_rate`/`iterations`/`subsample`/`colsample_bylevel` are hardcoded in `cv_harness.run_split`, never tuned via config or CV.
- **Calibration**: brier-score / win-probability calibration check (isotonic or Platt scaling against the normal-CDF approximation currently used).
- **Model-class, narrowed to CatBoost-internal**: compare `MultiRMSE` vs. two single-target `RMSE` models vs. `Quantile` loss (for the calibration angle) — other GBDT libraries or NN architectures deprioritized given dataset size (~2.3k–7k training games/fold), see prior discussion.

## 4. Decision log

**Guardrail (applies to every entry below and every future one)**: per-fold deltas must be shown, not just the mean. No experiment is promoted on fold1's strength alone — an improvement must hold on folds 2–5.

---

**`champion_cv_baseline`** (`outputs/experiments_v2.csv`, no session_id — manual one-off)
- Hypothesis: establish the reference point for all CV-protocol comparisons going forward, using the already-adopted single_split feature set re-evaluated under the audited/leak-fixed CV harness.
- Result: val_score_mean 1.3850, test 1.3724. Per-fold val: 1.4407 / 1.3876 / 1.3804 / 1.3579 / 1.3585. Fold1 is the clear outlier (see section 1).
- Conclusion: this is the standing champion under CV. Nothing compared against it yet has beaten it.
- Next: section 3's agenda.

**`style_matchup_knn_fixed_cv`** (session `20260805_1611_family-inventory`)
- Hypothesis: `matchup_index.py`'s `FINGERPRINT_METRICS` bug (missing `offensive_rating`, 5 metrics instead of 6 — fixed in #40) materially changes the KNN-similarity feature's behavior; re-test under full CV (never tested under CV before, only single_split in July, not adopted).
- Result: mean val 1.3830, test 1.3690. Per-fold val: 1.4341 / 1.3846 / 1.3788 / 1.3521 / 1.3653. Per-fold test: 1.3862 / 1.3873 / 1.3718 / 1.3620 / 1.3379.
- Conclusion: within noise of champion (1.3850/1.3724) on every fold — the fix is real (99.5% of a 200-game sample had a changed similarity score) but doesn't change the adoption verdict. Not adopted; `style_matchup.enabled` stays `false`.
- Next: none planned — this line is closed pending a genuinely new signal (e.g. richer style inputs, per the design doc's Future Work), not a rerun of the same feature.

**`fingerprint_ablation_on_cheap3fold` / `fingerprint_ablation_off_cheap3fold`** (session `20260805_1611_family-inventory`)
- Hypothesis: does the adopted raw-fingerprint block (18 dims) still earn its place under the real, leak-fixed CV harness (previously only validated via the pre-this-session `walkforward.py` CV)?
- Result (folds 3–5 only, cheap screen): on mean val 1.3656/test 1.3605, off mean val 1.3725/test 1.3620. Per-fold (on / off / delta): fold3 1.3804/1.3912/−0.0108, fold4 1.3579/1.3565/+0.0014, fold5 1.3585/1.3699/−0.0114 (val); fold3 1.3720/1.3756/−0.0036, fold4 1.3787/1.3689/+0.0098, fold5 1.3308/1.3417/−0.0108 (test).
- Conclusion: 2 of 3 folds favor keeping it on, 1 (fold4) favors off — not unanimous, but directionally consistent with section 2's finding that this family has the highest permutation cost of all 13. Re-confirms the original adoption under a more rigorous, independently-built harness.
- Next: none planned — already adopted, this was re-verification, not a new proposal.

---

**Session summary — `20260805_1611_family-inventory`**
- Explored: the `FINGERPRINT_METRICS` bug fix + CV re-test of `style_matchup.enabled`; the raw-fingerprint on/off cheap-3-fold ablation; the 13-family CatBoost + permutation importance inventory across all 5 folds.
- Promoted: nothing new to `experiments_v2.csv` beyond `champion_cv_baseline` itself (already logged separately, not part of this session's own runs).
- Dropped: nothing newly disabled — `style_matchup.enabled` confirmed staying `false`, `raw_features_enabled` confirmed staying `true`.
- Produced: the family-importance inventory that drives section 3's entire agenda — this session's main output was diagnostic/evidence-gathering, not a feature change.
