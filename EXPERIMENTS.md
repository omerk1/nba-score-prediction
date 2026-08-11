# EXPERIMENTS.md

Decision log and research agenda for the ablation-gated feature workflow (CLAUDE.md's "Project Rules (ML experimentation)" section). Numbers → `outputs/experiments_v2.csv` / `results/sessions/<session_id>.csv`. Interpretation → here, referenced by run_name / session_id.

## 1. Ground truth

**Composite score** (minimize): `diff_mae/naive_diff_mae + 0.5 * total_mae/naive_total_mae`, both terms normalized against that same split's own freshly-recomputed naive rolling-baseline. The naive baseline's own score is **always exactly 1.5** for any fold — a mathematical identity of the formula (evaluate it on the naive predictor against itself: both ratios are 1, so `1 + 0.5*1 = 1.5`), not something that needs a separate run to establish. Every score below is judged against that fixed 1.5 floor.

**CV protocol**: 5 expanding-window folds, oldest → newest, `configs/config.yaml`'s `cv.folds`, mechanically validated by `validate_fold_definitions` (fold ordering, no overlap, no fold's val/test predating an earlier fold's own training window). Fold5 = today's `--protocol single_split` boundaries exactly.

**Reproducibility, verified at full precision**: two independent full 5-fold CV runs of the (then-)champion config, diffed at full float64 precision (raw metric dicts and actual val/test predictions, not the 4dp values in logged CSV rows) — byte-identical on every fold, every metric, every prediction. `std = 0.0` exactly. `target_formulation=diff_total`'s own determinism was separately re-verified (two independent fold5 runs, full precision, byte-identical) before adoption — see the `target_formulation_diff_total` decision-log entry.

**Champion baseline** (`target_lambda_weight_0.75`, `outputs/experiments_v2.csv`): `style_matchup.raw_features_enabled=true`, `preferred_opponent_delta_enabled=true`, `style_matchup.enabled=false`, `model.target_formulation=diff_total`, `model.target_lambda_weight=0.75`.

| fold | val_score | test_score |
|---|---:|---:|
| 1 | 1.4368 | 1.3948 |
| 2 | 1.3898 | 1.3852 |
| 3 | 1.3719 | 1.3713 |
| 4 | 1.3527 | 1.3713 |
| 5 | 1.3592 | 1.3299 |
| **mean** | **1.3821** | **1.3705** |

**Fold-1 gap, called out explicitly**: fold1's val_score (1.4368) is ~4–6% worse than every other fold (1.35–1.39 range) — the smallest training window (2018-10-16 → 2020-08-14) is the weakest fold by a clear margin, not noise. Section 2's fold-1 breakout and section 3's diagnostics/experiments (run under the prior `home_away` champion) are largely about understanding and addressing this gap; it persists under the current champion too, just slightly narrowed.

**Superseded reference points** (all kept in `experiments_v2.csv` for provenance, never overwritten, per CLAUDE.md):
- `champion_cv_baseline` (val_score_mean 1.3850) → `champion_cv_baseline_post_injury_fix` (1.3851): PR #44 fixed a bug in `injury_layer.py`'s multi-archetype injury-delta accumulation. Effect was negligible (fold1 exactly unchanged, folds 2–5 shifted −0.0019 to +0.0028, no consistent direction) — see PR #44/#45.
- `champion_cv_baseline_post_injury_fix` (1.3851) → `champion_cv_baseline_diff_total` (1.3838): the target-reformulation experiment (section 3.3) — fitting `MultiRMSE` on `[POINT_DIFF, TOTAL_POINTS]` instead of `[PTS_home, PTS_away]` beat the old formulation on val_score in **all 5 folds individually**, the first candidate in this project's CV-protocol history to cleanly clear the per-fold guardrail. `configs/config.yaml`'s `model.target_formulation` is now `diff_total`.
- `champion_cv_baseline_diff_total` (1.3838) → `target_lambda_weight_0.75` (1.3821): the `target_lambda_weight` sweep (section 3.3 follow-up) — 0.5 had only ever been inherited from `compute_composite_score`'s own diff/total weighting, never independently tuned for the training loss; a cheap 3-fold screen found 0.75 best, and full 5-fold CV confirmed it beats 0.5 on val_score in 4 of 5 folds (only fold2 regresses). `configs/config.yaml`'s `model.target_lambda_weight` is now `0.75`. No dedicated `champion_cv_baseline_*` row this time — the experiment's own row already has byte-identical values, so a second row would be pure duplication; `EXPERIMENTS.md` (here) is the pointer, not the CSV row name.

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
- **Confirmed (session `rs_20260808_1`)**: direct per-fold query against `injury_features.sqlite` (`scorer='formula'`) — `fold1`: 0 dates with data in both train (≤2020-08-14) and val (2020-12-22→2021-05-16) windows, its test window (2021-10-19→2022-04-10) has full coverage (165/165 dates). `fold2`: train (≤2021-05-16) is *also* 0 coverage — the training window itself predates the data, not just val. `fold3` onward: full coverage in every window (train/val/test all ≥160 dates). No coverage exists anywhere before 2021-10-19, and coverage is dense (155–165 distinct dates/season, effectively every game day) once it starts — no partial-coverage gaps in the covered range. Hypothesis confirmed exactly: this is a hard data floor, not a signal-quality problem. Decision: E4 variant (c) (folds-start-at-2021) should be understood precisely as "exclude fold1 and fold2's *training* window," not just "exclude fold1" — fold2's train set has zero injury rows too, even though fold2's val/test don't.

**D2. `style_fingerprint_features` leakage re-audit.**
- Hypothesis: the family's dominance (27.4% share, highest permutation cost) is genuine, not a fold-boundary leak that the original (pre-CV-harness) leakage verification wouldn't have caught.
- Action: reuse this session's empirical audit technique (compare truncated-history vs. full-history fingerprint computation for a sample of games right at each fold's `test_start_date`) — confirm the first test-period game for a team only reflects fingerprint history strictly before that fold's own boundary.
- Expected effect: pass (no leak) is the expected/likely outcome, given the underlying `.shift(1)` construction; if it fails, this supersedes every other item in this section.
- Effort: low-medium (a script, no retraining). Risk: none (read-only). Screening: n/a.
- **Result (session `rs_20260808_1`, `scripts/audit_fingerprint_leakage.py`): PASS, no leakage.** For each of the 5 folds, sampled the first 50 test-period games (500 team-game instances, 3000 metric comparisons total) and recomputed each one from scratch using only `box_score_stats`/`player_injuries` rows dated strictly/at-most that game's own date — every value production returns is exactly reproducible (0 unexplained mismatches out of 3000, `abs diff < 1e-6` everywhere) from a from-scratch, same-date-bounded recomputation. `.shift(1)` + backward-only `merge_asof` is confirmed correct at real fold boundaries, not just in isolation.
- **Bug found along the way (not leakage, flagged separately): `injury_layer.py`'s multi-archetype delta accumulation.** First audit pass (layer=1-only truncated recompute vs. production, all 6 metrics) showed 608/3000 "mismatches" — a false start: 5 of 6 metrics are actually served from layer=2 (injury-adjusted) in production, and the gap was the legitimate injury delta, not leakage. Adding the correct same-date injury delta closed all but 64/3000. Root cause, confirmed by exactly replicating `injury_layer.py`'s own loop: `build_injury_adjusted_fingerprints` computes each archetype's delta as `fp2.at[idx, metric] = row[metric] + delta * severity_mult` — reading from the *original* layer=1 `row[metric]` every iteration instead of the running `fp2.at[idx, metric]`. When 2+ different-archetype players are out for the same team on the same date and their `injury_impact` dicts share a metric (common — `combo`'s dict touches 5 of 6 metrics), only the last-iterated archetype's delta survives; earlier ones are silently overwritten instead of summed. Replicating this exact overwrite-not-accumulate behavior reproduces production bit-for-bit (confirms the mechanism, not a guess). Affects 64/3000 = 2.1% of sampled comparisons — real, but not leakage (no information from after the game's own date is involved), and not something D2 was chartered to fix.
- **Logged as "proposed, pending review"** per the execution-scope guardrail: this is a correctness fix to existing feature *values* (not a new feature, not a feature-set structure change), which is ambiguous under "flip config flags / tune parameters" — treated as out-of-scope-for-execution per the guardrail's own tie-breaker ("when unsure, treat as proposed"). Suggested fix: change the accumulation line to add onto the running `fp2.at[idx, metric]` instead of re-deriving from `row[metric]` each iteration. Until fixed, `style_fingerprint_features`' injury-adjusted values are a systematic (bug-consistent, not random) under-count of the true multi-injury adjustment on the ~2% of team-games where multiple different-archetype players are simultaneously out — worth noting as a caveat on section 2's `style_fingerprint_features` dominator finding (27.4% share), though far too small a fraction of games to explain that magnitude on its own.
- **Next-session candidate (not yet scheduled):** fix the accumulation line, then re-run `champion_cv_baseline` fresh under full 5-fold CV and log it as its own new dated `experiments_v2.csv` row (not a retroactive edit to the existing champion row — the baseline itself moves, however slightly, so it needs its own validation run per the ablation-gated workflow). Won't change any of this session's E1/E2/E4 conclusions (the bug is orthogonal to every axis those experiments varied), but gives a more accurate absolute champion score and a truer read on `style_fingerprint_features`' real importance. Cheap (one CV run, no new script needed — reuses the existing champion-run pattern).

### 3.2 High-value experiments (pending D1/D2 passing)

**E1. Expand Elo.**
- Hypothesis: current `k_factor=11.02`/`season_regression=0.522` were tuned via `scripts/tune_elo.py` under the old single_split protocol, predating the CV harness and never re-validated per-fold. Given Elo's fold1 weakness (ratio 0.74) and its efficiency (high permutation cost relative to modest CatBoost share), retuning — or specifically testing whether a different `season_regression` narrows the fold1 gap — is worth it.
- Protocol: **full 5-fold, not cheap-3-fold** (fold1 is the whole point).
- Expected effect: fold1 val_score improves without materially moving folds 2–5 (guardrail: must hold on 2–5, not just fix fold1).
- Effort: medium (small grid/Optuna sweep via existing `tune_elo.py` infra, re-run under CV instead of single_split). Risk: low (isolated to `elo_features` config).
- **Result (session `rs_20260808_1`, `scripts/tune_elo_cv.py`): no promotion.** `tune_elo.py` couldn't be reused as-is (single-split only, and its fixed CatBoost params — depth 3, subsample 0.92 — differ from `run_split`'s actual depth 6/subsample 0.8, so tuning against it wouldn't transfer); wrote a new script that grid-searches `season_regression` ∈ {0.30, 0.40, 0.522, 0.65, 0.80} through `run_split` unmodified, full 5-fold per candidate (`k_factor` held at 11.02). `season_regression=0.522` (current default) reproduced `champion_cv_baseline` exactly (val_mean=1.3850, per-fold identical to 4dp) — a nice byproduct confirmation of the determinism finding from section 1. Best candidate: `season_regression=0.65`, val_mean=1.3841 (Δ−0.0009 vs. champion). Per-fold delta (0.65 − champion): fold1 −0.0059, fold2 +0.0025, fold3 +0.0018, fold4 −0.0059, fold5 +0.0030 — improves fold1 (the target) and fold4, but regresses folds 2/3/5. Guardrail requires holding on folds 2–5, not just fold1; 3 of those 4 folds got worse, so this does not qualify despite the marginally better mean.
- Conclusion: **not promoted.** No tested `season_regression` value improves on the champion in a way that survives the fold-2–5 guardrail; the whole 1-D grid is essentially flat (val_mean range 1.3841–1.3852, well under a 0.001 spread — deterministic, not noise, but not a meaningful lever either). Decided **not** to extend to the `k_factor` cross-grid (`--grid both`, 15 candidates): with the single axis most directly tied to the fold1-cold-start hypothesis already showing no clean win, and mixed (not uniformly fold1-favoring) per-fold behavior, a 3x-larger grid looked unlikely to change the verdict — reallocating remaining session time to E2/E4, which have sharper, more specific hypotheses (E4 especially, given D1 also found fold2's *training* window has zero injury coverage, another concrete, already-diagnosed lever on the same cold-start problem this experiment was chasing).
- Next: none planned on this axis for now. `k_factor`/`home_advantage` retuning remains a candidate for a future, separately-scoped session if the E4/E2 results suggest revisiting Elo specifically.

**E2. Prune the dead tail.**
- Hypothesis: `season_motivation.preferred_opponent_delta_enabled: false` performs flat-to-marginally-better than `true` under full CV (matches its negative permutation delta).
- Protocol: cheap 3-fold screen (folds 3–5) first; full 5-fold only if the cheap screen looks promising enough to consider promoting.
- Expected effect: val_score_mean roughly flat or very slightly improved.
- Effort: very low (single flag flip, `scripts/run_fingerprint_ablation.py`'s pattern reused for this flag instead). Risk: very low.
- Watch list, not bundled into E2's own test (avoid conflating simultaneous changes): `rest_features`, `home_advantage_features` — both near-zero permutation cost though not negative. Worth their own single-flag checks later, not this run.
- **Result (session `rs_20260808_1`, `scripts/run_season_motivation_ablation.py`): cheap-screen only, not escalated.** Folds 3–5: on mean val=1.3656/test=1.3605, off mean val=1.3650/test=1.3519. Per-fold delta (on−off): fold3 −0.0019, fold4 **+0.0087**, fold5 −0.0051 — 2 of 3 folds favor off (matches the hypothesis direction), fold4 favors keeping it on fairly clearly. Overall val_mean improvement from turning it off is +0.0006 — matches the expected "roughly flat" outcome exactly, not a real lever either way.
- Conclusion: **not escalated to full 5-fold this session.** Effect is tiny and directionally mixed (not the "looks promising enough to consider promoting" bar the protocol set for escalation) — reallocated the remaining time to E4, which has a sharper, D1-gated hypothesis. `preferred_opponent_delta_enabled` stays `true` (unchanged).
- Next: candidate for a future cheap addition (very low effort) if a session has idle budget, but not a priority — the tiny magnitude here doesn't justify a full-5-fold run on its own.

**E3. Refine top families — collinearity check.**
- Hypothesis: `style_fingerprint_features` and `style_features` (both broad "team style/quality" proxies) and/or `opponent_quality_features` (the "used but not predictive" family from section 2) have real overlap — consolidating could simplify the model without losing signal, or confirm `opponent_quality_features` is safe to prune alongside `season_motivation_features`.
- Action: correlation/VIF analysis between the three families' engineered columns — diagnostic only, no retraining required first.
- Effort: medium (a script). Risk: none for the diagnostic; any resulting drop-a-family experiment inherits E2's protocol.
- **Result (session `rs_20260808_1`, `scripts/family_correlation_vif.py`, fold5 train features, 7,059 rows, 48 columns across the 3 families):** cross-family correlation is modest, not the "real overlap" the hypothesis expected — mean |r|: `style_fingerprint_features` vs. `style_features` 0.163, `style_fingerprint_features` vs. `opponent_quality_features` 0.162, `style_features` vs. `opponent_quality_features` 0.209 (all well under a redundancy-level threshold). Highest single cross-family pair: `away_style_pace_score` vs. `away_team_def_eff_L20`, r=0.753 — a real but isolated overlap between two individual columns, not evidence of broad family-level redundancy. Mean VIF by family: `style_fingerprint_features` **1.31** (essentially no collinearity issue anywhere), `opponent_quality_features` 5.72, `style_features` 6.25. Only 2/48 columns exceed the conventional VIF=10 flag (`home/away_team_off_eff_L10`, 11.9/11.3) — and those are *within-family* (both `style_features`), not cross-family.
- Conclusion (**proposed, pending review** — no consolidation executed): the original hypothesis doesn't hold at the family level — `style_fingerprint_features` (the 27.4%-share dominator) is essentially collinearity-free and stays distinct from the other two, arguing *against* touching it. What the VIF numbers actually show is **within-family** redundancy in `style_features` and `opponent_quality_features` specifically — their own multiple rolling-window variants (L5/L10/L20 of the same underlying efficiency/quality stat) correlate heavily with each other. A more targeted, better-supported proposal than "merge families": test trimming redundant rolling windows *within* `style_features`/`opponent_quality_features` (e.g., L5+L20 only, dropping L10) rather than any cross-family consolidation. This is a refinement of the original hypothesis, not a confirmation of it — logged for review, nothing executed.
- Next: a scoped rolling-window-trim experiment for `style_features`/`opponent_quality_features` is a better-motivated future item than the original family-merge idea; not run this session.

**E4. Injury missing-data handling** (depends on D1; explicitly not part of E2's prune group — `injury_features`' aggregate score is understated, not weak, see section 2).
- Hypothesis: how missing injury data is represented matters more than whether the feature itself is weak. Three variants, compared directly:
  1. Native CatBoost NaN handling (status quo — missing rows pass through untouched).
  2. Explicit availability-indicator feature (`home/away_team_injury_data_available`) alongside the existing columns.
  3. Folds-start-at-2021 — restrict the comparison to the date range where coverage actually exists (drops/truncates fold1, per D1's coverage findings).
- Protocol: each variant screened only on the folds where it's actually comparable to the others (variants 1–2 run across all 5 folds; variant 3 is inherently restricted to the covered range, so its comparison is scoped to whichever folds overlap). Not a cheap-3-fold screen — this is fold1-relevant by construction.
- Expected effect: may lift fold1 specifically; watch per-fold, not just the mean (guardrail applies — no promotion on a fold1-only improvement).
- Effort: medium (indicator feature is a small `feature_builder.py` addition; the fold-restriction variant reuses existing fold-filtering, no harness changes). Risk: low.
- **Reframed before running** (see EXPERIMENTS.md's flagged findings from planning): "variant 1: native NaN, status quo" was inaccurate — the code actually zero-fills; and "variant 2: availability indicator" already ships (`has_injury_data`). Real comparison run: `injury_features.missing_value_strategy: zero_fill` (true status quo) vs. `native_nan` (new — a config-gated code change removing the `fillna(0)` imputation), full 5-fold each. Variant 3 (folds-start-at-2021) computed as a folds3-5-only slice of these same two runs, not a third training pass (D1 confirmed folds 1–2 have zero *training*-time coverage, so folds 3–5 are where the strategy can actually matter).
- **Result (session `rs_20260808_1`, `scripts/run_injury_missingdata_comparison.py`):** full-5-fold mean val: zero_fill=1.3850, native_nan=1.3849 — nearly identical. Per-fold delta (native_nan − zero_fill): fold1 −0.0051, fold2 −0.0001, fold3 **+0.0079**, fold4 −0.0060, fold5 **+0.0027**. **folds3-5-only mean val: zero_fill=1.3656, native_nan=1.3672 — native_nan is *worse* specifically on the folds where the feature has real training-time signal.** The apparent full-5-fold parity is fold1's improvement (a fold with zero training-time injury data either way) offsetting a real regression on folds 3–5. (Test scores logged only, not decision-driving per CLAUDE.md's hard constraint — they actually point the opposite direction on folds 3–5, underscoring why val governs.)
- Conclusion: **not promoted; original hypothesis not supported.** This is exactly the guardrail's fold1-alone trap in a new form — the only fold(s) where `native_nan` looks good are the ones D1 already proved carry no real injury signal during training either way, and on the folds that do carry signal, it's mildly worse. `missing_value_strategy` stays `zero_fill`. Closes out E4 as a diagnostic question: representation strategy isn't the fold1-gap lever; the coverage floor itself (D1) is the real constraint, and no missing-value encoding trick recovers information that was never in the training data.
- Next: none — the config flag stays in place (disabled-from-adoption, ships at `zero_fill`) for any future revisit, but this line is closed.

### 3.3 Model-agnostic axes (broader, lower immediate priority — not driven by the family inventory directly)

- **Target reformulation — executed and ADOPTED.** See the `target_formulation_diff_total` decision-log entry below (section 4): implemented as `model.target_formulation` config flag (`ScorePredictor._to_training_targets`/`_from_training_targets`), full 5-fold CV comparison run. `diff_total` beat `home_away` on val_score in all 5 folds, mean 1.3838 vs. 1.3851 — the first candidate in this project's CV-protocol history to cleanly clear the per-fold guardrail. Re-verified after an explicit re-audit (CatBoost's `MultiRMSE` loss formula confirmed against official docs, determinism re-checked, save/load round-trip tested) before adoption. `configs/config.yaml`'s `model.target_formulation` is now `diff_total` — this is the new standing champion.
- **Hyperparameter tuning**: CatBoost's `depth`/`learning_rate`/`iterations`/`subsample`/`colsample_bylevel` are hardcoded in `cv_harness.run_split`, never tuned via config or CV.
- **Calibration**: brier-score / win-probability calibration check (isotonic or Platt scaling against the normal-CDF approximation currently used).
- **Model-class, narrowed to CatBoost-internal**: compare `MultiRMSE` vs. two single-target `RMSE` models vs. `Quantile` loss (for the calibration angle) — other GBDT libraries or NN architectures deprioritized given dataset size (~2.3k–7k training games/fold), see prior discussion.

## 4. Decision log

**Guardrail (applies to every entry below and every future one)**: per-fold deltas must be shown, not just the mean. No experiment is promoted on fold1's strength alone — an improvement must hold on folds 2–5.

**No significance floor on top of that**: the champion config was verified byte-identical (see section 1) across two independent full 5-fold CV runs — `std = 0.0` exactly, not just small. Only the champion config was directly tested, but the mechanism is config-independent (`ScorePredictor`'s fixed `random_state` governs CatBoost's own bagging RNG regardless of which features are enabled, and nothing else in the pipeline introduces unseeded randomness), so this should generalize to any config in the agenda. Since there is no measured run-to-run variance at all, an experiment's reported delta is never competing with sampling noise; the per-fold-deltas-required guardrail above is sufficient on its own. Adding an arbitrary significance multiplier here would be inventing a floor the evidence doesn't call for.

**Autonomous-session execution scope**: an unattended run may fully execute anything that only flips a config flag or tunes a parameter within the existing model/loss/metric — this covers both diagnostics (D1, D2) and the config-level experiments (E1 Expand Elo, E2 Prune the dead tail, E4 Injury missing-data handling). It may run E3's correlation/VIF analysis (diagnostic, no retraining). It must NOT execute, on its own authority, any change to model architecture, the training loss, or the composite metric itself — specifically: E3's actual family-consolidation action (dropping or merging `style_fingerprint_features`/`style_features`/`opponent_quality_features` columns, once the VIF analysis motivates it) and the 3.3 target reformulation (`[home,away]` → `[diff,total]` changes what `MultiRMSE` is fit against, i.e. changes the loss). Anything in this second category gets logged in section 4 as **"proposed, pending review"** — hypothesis, the analysis that motivates it (e.g. the VIF numbers, or the loss-alignment argument), and expected effect — but not run, pending my review.

---

**`champion_cv_baseline`** (`outputs/experiments_v2.csv`, no session_id — manual one-off)
- Hypothesis: establish the reference point for all CV-protocol comparisons going forward, using the already-adopted single_split feature set re-evaluated under the audited/leak-fixed CV harness.
- Result: val_score_mean 1.3850, test 1.3724. Per-fold val: 1.4407 / 1.3876 / 1.3804 / 1.3579 / 1.3585. Fold1 is the clear outlier (see section 1).
- Conclusion: this is the standing champion under CV. Nothing compared against it yet has beaten it.
- Next: section 3's agenda.

**`style_matchup_knn_fixed_cv`** (session `20260805_1611_family-inventory`)
- Hypothesis: `matchup_index.py`'s `FINGERPRINT_METRICS` bug (missing `offensive_rating`, 5 metrics instead of 6 — fixed in #40) materially changes the KNN-similarity feature's behavior; re-test under full CV (never tested under CV before, only single_split in July, not adopted).
- Result: mean val 1.3830, test 1.3690. Per-fold val: 1.4341 / 1.3846 / 1.3788 / 1.3521 / 1.3653. Per-fold test: 1.3862 / 1.3873 / 1.3718 / 1.3620 / 1.3379.
- Conclusion: a small measured effect vs. champion (1.3850/1.3724 → 1.3830/1.3690), not clearly distinguishable from zero — deterministic delta from a genuine 2-feature config difference (129 vs. 127 features, `style_matchup_score`/`confidence`), not stochastic noise. The underlying fix is real (99.5% of a 200-game sample had a changed similarity score) but doesn't change the adoption verdict. Not adopted; `style_matchup.enabled` stays `false`.
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

---

**`D1` — injury data-coverage check** (session `rs_20260808_1`)
- Hypothesis: `injury_features.sqlite` doesn't cover fold1's train/val window at all — the fold1 zero is a coverage gap, not a signal-quality problem.
- Result: confirmed exactly, plus one refinement — fold2's *training* window (≤2021-05-16) also has zero coverage, not just fold1. Fold3 onward: full, dense coverage (155–165 distinct dates/season) in every window. No coverage gaps anywhere else in the covered range.
- Conclusion: hard data floor, not a signal-quality problem. Gates E4's fold-restriction variant.
- Next: E4.

**`D2` — `style_fingerprint_features` leakage re-audit** (session `rs_20260808_1`, `scripts/audit_fingerprint_leakage.py`)
- Hypothesis: the family's dominance (27.4% share, highest permutation cost) is genuine, not a fold-boundary leak.
- Result: PASS, no leakage — 0/3000 unexplained mismatches across all 5 folds' fold-boundary samples (see section 2 for full detail). Along the way, found and confirmed (by exact replication) a real but unrelated correctness bug in `injury_layer.py`'s multi-archetype delta accumulation (overwrite instead of sum when 2+ archetypes share an affected metric on the same team/date) — affects 64/3000 = 2.1% of sampled comparisons.
- Conclusion: champion is not invalidated — no leakage found. The accumulation bug is logged as **proposed, pending review** (a correctness fix to existing feature values, not executed per the execution-scope guardrail), not blocking the rest of this session's agenda.
- Next: E1, E2, E4, E3 proceed as planned.

**`E1` — Expand Elo** (session `rs_20260808_1`, `scripts/tune_elo_cv.py`)
- Hypothesis: `season_regression=0.522` (tuned pre-CV-harness, single_split) may be narrowable to reduce fold1's cold-start gap.
- Result: full 5-fold grid over `season_regression` ∈ {0.30, 0.40, 0.522, 0.65, 0.80}, `k_factor` held at 11.02. Best: `season_regression=0.65`, val_mean=1.3841 vs. champion's 1.3850. Per-fold delta (best − champion): fold1 −0.0059, fold2 +0.0025, fold3 +0.0018, fold4 −0.0059, fold5 +0.0030.
- Conclusion: **not promoted** — improvement doesn't hold on folds 2–5 (3 of 4 non-fold1/4 folds regressed), guardrail violation despite a marginally better mean. Whole grid essentially flat (1.3841–1.3852 range). `season_regression=0.522` exactly reproduced `champion_cv_baseline`'s numbers to 4dp, a useful incidental determinism re-confirmation.
- Next: `k_factor` cross-grid deliberately not run (see section 3.2's full writeup for rationale) — time reallocated to E2/E4.

**`E2` — Prune the dead tail** (session `rs_20260808_1`, `scripts/run_season_motivation_ablation.py`)
- Hypothesis: `preferred_opponent_delta_enabled=false` performs flat-to-marginally-better than `true` (matches its negative permutation delta).
- Result: cheap 3-fold screen (folds 3–5). On mean val=1.3656/test=1.3605, off mean val=1.3650/test=1.3519. Per-fold delta (on−off): fold3 −0.0019, fold4 +0.0087, fold5 −0.0051.
- Conclusion: **not escalated to full 5-fold** — matches the "roughly flat" expected effect exactly (+0.0006 mean), but direction is mixed across folds (2/3 favor off, fold4 favors on). Doesn't clear the "promising enough to escalate" bar the protocol set. `preferred_opponent_delta_enabled` stays `true`, unchanged.
- Next: none planned this session — low-priority future candidate given the tiny magnitude either way.

**`E4` — Injury missing-data handling** (session `rs_20260808_1`, `scripts/run_injury_missingdata_comparison.py`)
- Hypothesis: representation of missing injury data (zero_fill vs. native_nan) matters more than the feature's aggregate weakness; may lift fold1 specifically.
- Result: full 5-fold, both variants. Full-5-fold mean val nearly identical (zero_fill=1.3850, native_nan=1.3849), but folds3-5-only mean val: zero_fill=1.3656 vs. native_nan=1.3672 — native_nan is worse specifically on the folds with real training-time injury coverage. Per-fold delta (nan−zf): fold1 −0.0051, fold2 −0.0001, fold3 +0.0079, fold4 −0.0060, fold5 +0.0027.
- Conclusion: **not promoted; hypothesis not supported.** native_nan's only apparent gains are on fold1/fold2, which D1 already proved have zero training-time injury data regardless of encoding — the guardrail's fold1-alone trap, recognized and not acted on. `missing_value_strategy` stays `zero_fill`. Closes the diagnostic: D1's coverage floor is the real constraint, not encoding.
- Next: none — config flag stays in place, disabled-from-adoption, for any future revisit.

**`E3` — Refine top families, collinearity check** (session `rs_20260808_1`, `scripts/family_correlation_vif.py`, diagnostic only — no consolidation executed)
- Hypothesis: `style_fingerprint_features`/`style_features`/`opponent_quality_features` have real cross-family overlap worth consolidating.
- Result: cross-family mean |r| modest throughout (0.16–0.21). Mean VIF: `style_fingerprint_features` 1.31 (no collinearity issue), `opponent_quality_features` 5.72, `style_features` 6.25. Only 2/48 columns exceed VIF=10, both within `style_features` (its own L10/L20 rolling-window variants of `off_eff`).
- Conclusion: **proposed, pending review — hypothesis not confirmed as stated.** The dominant family is collinearity-free and distinct; the real redundancy is *within* `style_features`/`opponent_quality_features` (their own multiple rolling windows), not *across* the three families. A rolling-window trim is the better-motivated next step than a family merge — logged, not executed.
- Next: none this session — a scoped follow-up experiment, not a priority.

---

**Session summary — `rs_20260808_1`**
- Explored: full section 3 agenda in order — D1 (injury coverage), D2 (fingerprint leakage gate), E1 (Elo grid), E2 (season_motivation prune screen), E4 (injury missing-data handling), E3 (collinearity/VIF diagnostic).
- Gates: D1 confirmed the coverage floor exactly (plus a refinement — fold2's train window is also affected, not just fold1). D2 passed clean, no leakage — champion not invalidated.
- Found along the way (not part of the planned agenda): a real correctness bug in `injury_layer.py`'s multi-archetype injury-delta accumulation (overwrite instead of sum, ~2% of team-games affected). Not leakage, doesn't invalidate anything this session concluded (orthogonal to every experiment's own varied axis). Logged as **proposed, pending review** — not fixed.
- Promoted to `outputs/experiments_v2.csv`: two rows, for leaderboard completeness only (CLAUDE.md's "any row that beats the current champion" rule) — `elo_sr0.65_k11.02` (val_score_mean=1.3841) and `injury_missingdata_native_nan_full5fold` (val_score_mean=1.3849). **Neither is adopted or the new champion** — both explicitly fail the per-fold guardrail (fold1-driven, not holding across folds 2–5), exactly the pattern the guardrail exists to catch. `champion_cv_baseline` remains the standing champion; `configs/config.yaml`'s adopted defaults are unchanged from `main`.
- Dropped/not promoted: E1 (Elo retune), E2 (season_motivation prune, not even escalated to full CV), E4 (injury missing-data handling) — all negative or inconclusive results, all informative.
- Proposed, pending review (not executed, per the execution-scope guardrail): the `injury_layer.py` accumulation-bug fix (D2), and E3's rolling-window-trim follow-up (a more precise, better-motivated proposal than the original family-merge hypothesis it replaces).
- Net effect on the champion: none — no promotion this session. The main value was diagnostic: a clean leakage bill of health for the dominant feature family (with one real bug found and flagged, not fixed), three ruled-out directions (Elo season_regression, season_motivation pruning, injury missing-data encoding) with clear per-fold evidence for each, and a corrected, more actionable version of the collinearity hypothesis for a future session.
- Config state: `configs/config.yaml` on this branch differs from `main` only by the new (disabled-from-adoption) `injury_features.missing_value_strategy: zero_fill` field added for E4 — no adopted defaults changed.

---

**`target_formulation_home_away` / `target_formulation_diff_total`** (manual one-off, `scripts/run_target_formulation_experiment.py`)
- Hypothesis (section 3.3): fitting `MultiRMSE` on `[POINT_DIFF, TOTAL_POINTS]` instead of `[PTS_home, PTS_away]` should improve the composite score by aligning the training loss with what it actually rewards (diff-dominant, total at half weight).
- Implementation: `model.target_formulation` config flag (`home_away` default / `diff_total`), entirely internal to `ScorePredictor` — `_to_training_targets`/`_from_training_targets` convert in/out of fit-space, but `predict()`/`evaluate()`'s public contract always stays `[home, away]`. `diff_total` mode scales `total` by `sqrt(target_lambda_weight)` before fitting (and unscales after) — the standard trick for asymmetric per-output weighting in a loss that otherwise weights every dimension's squared error equally, since CatBoost's `MultiRMSE` has no native per-dimension weight parameter. `cv_harness.py` untouched beyond threading 2 new params into the existing `ScorePredictor(...)` call — no changes to fold logic, `naive_baseline_metrics`, or `compute_composite_score`.
- Correctness check: `target_formulation_home_away`, full 5-fold CV, reproduced `champion_cv_baseline_post_injury_fix` **exactly** — val_score_mean 1.3851, per-fold `1.4407,1.3878,1.3796,1.3560,1.3613`, identical to 4dp on every logged metric. Confirms the refactor is genuinely a no-op in the default mode, not just in unit tests.
- Result: `target_formulation_diff_total`, full 5-fold CV. val_score_mean **1.3838** vs. home_away's 1.3851 (Δ−0.0013). Per-fold val delta (diff_total − home_away): fold1 −0.0036, fold2 −0.0008, fold3 −0.0016, fold4 −0.0006, fold5 −0.0001 — **every single fold improves**, not just fold1. Test scores also mostly favor diff_total (mean 1.3694 vs. 1.3727), informational only per the hard constraint.
- Conclusion: **this is the first candidate in this project's CV-protocol history to cleanly clear the per-fold guardrail** — unanimous improvement across all 5 folds, full CV, not a fold1-driven artifact. Magnitude is modest (0.0013 on val_score_mean) but real and consistent.
- Re-audit before adoption (requested explicitly, given this changes the training loss): (1) re-derived the `_to_training_targets`/`_from_training_targets` round-trip algebra by hand; (2) confirmed CatBoost's `MultiRMSE` loss formula against official docs — `sqrt(Σᵢ Σ_d (pred−true)² · wᵢ) / Σᵢ wᵢ`, only a per-sample weight exists, no per-dimension weight (also confirmed by CatBoost GitHub issue #2906, an open feature request for per-dimension weighting *because it doesn't exist yet*) — this is exactly what the `sqrt(lambda)` scaling trick relies on; (3) empirically swept `target_lambda_weight` (0.01/0.5/50.0) on synthetic data and confirmed `total_mae`/`diff_mae` trade off monotonically in the expected direction; (4) re-verified determinism specifically for `diff_total` mode on real fold5 data (two independent runs, full float64 precision, byte-identical — the earlier determinism proof only covered `home_away` mode); (5) found and closed a real gap — `predict_game.py` loads models via `ScorePredictor.load()` for live prediction, and the save/load round-trip for `target_formulation`/`target_lambda_weight` had never been tested; verified it round-trips correctly, added as a permanent regression test; (6) diffed `target_formulation_home_away`'s full-CV row against `champion_cv_baseline_post_injury_fix` on **every** logged column (not just val_score_mean) — exact match throughout.
- **ADOPTED.** `configs/config.yaml`'s `model.target_formulation` flipped to `diff_total` — new standing champion, logged as `champion_cv_baseline_diff_total` (a fresh `run_name`, matching the `champion_cv_baseline`/`champion_cv_baseline_post_injury_fix` naming convention — not left under the experiment's own comparison-arm name). Values are byte-identical to `target_formulation_diff_total`'s row, not re-executed: `diff_total` mode's determinism is proven, so this row *is* what any future run of the now-adopted config would produce.
- Next: `target_lambda_weight` itself was never swept (only the champion's existing `lambda_weight=0.5` composite-score weighting was carried through) — see the `target_lambda_weight_0.75` entry below.

---

**`target_lambda_weight_0.75`** (manual one-off, `scripts/sweep_target_lambda_weight.py`)
- Hypothesis (section 3.3 follow-up): `target_lambda_weight=0.5` was never independently tuned for the `diff_total` training loss — it was only ever inherited from `compute_composite_score`'s own diff/total weighting (a metric-level choice, not a loss-fitting one). A different value might fit `MultiRMSE` in a way that better serves the composite score.
- Stage 1, cheap screen (folds 3–5), grid `[0.1, 0.25, 0.5, 0.75, 1.0, 2.0]`:

  | λ | mean val_score (folds 3-5) |
  |---|---:|
  | 0.1 | 1.3673 |
  | 0.25 | 1.3624 |
  | 0.5 (champion) | 1.3649 |
  | **0.75** | **1.3613** |
  | 1.0 | 1.3656 |
  | 2.0 | 1.3616 |

  0.75 best on screen; not a monotonic trend (0.1 and 1.0 both worse than 0.75 and 2.0), so this reads as a shallow, fairly flat optimum around 0.5–2.0 rather than a sharp one — still enough signal to promote 0.75 to a full-CV check per the escalation protocol.
- Stage 2, full 5-fold CV, `0.5` (reference, not re-logged — identical to `champion_cv_baseline_diff_total`'s already-proven-deterministic numbers) vs. `0.75`:

  | fold | 0.5 | 0.75 |
  |---|---:|---:|
  | 1 | 1.4371 | **1.4368** |
  | 2 | **1.3870** | 1.3898 |
  | 3 | 1.3779 | **1.3719** |
  | 4 | 1.3554 | **1.3527** |
  | 5 | 1.3612 | **1.3592** |
  | **mean** | 1.3838 | **1.3821** |

  0.75 wins val_score on 4 of 5 folds (only fold2 regresses, +0.0028) — not fold1-driven (fold1 also improves, by a small margin). Test-score mean is flat-to-slightly-worse (1.3705 vs. 1.3694), expected noise, not a selection criterion.
- Conclusion: a real, mostly-consistent improvement — smaller magnitude than the `diff_total` reformulation itself (which was 5/5 folds), but clears the "not just one fold" bar with a clean majority.
- **ADOPTED.** `configs/config.yaml`'s `model.target_lambda_weight` flipped to `0.75` — new standing champion is the `target_lambda_weight_0.75` row above. Unlike the two prior supersessions, no separate `champion_cv_baseline_*` row was added: that convention was only ever about giving the CSV a self-describing name, but it's pure duplication when the experiment's own row already carries the winning values — this doc's "Champion baseline" line (section 1) is the actual pointer.
- Next: none — closed.
