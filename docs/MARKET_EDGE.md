# MARKET_EDGE.md

Strategic record: does an exploitable edge over the market exist, and where. Distinct from `docs/EXPLORATION.md` (pre-experiment hypotheses) and `docs/EXPERIMENTS.md` (CV-protocol training experiments) — this file tracks answers to "are we actually better than the market, and why/why not," using the market-benchmark tool (`scripts/market_benchmark.py`) and targeted investigations. Each entry: dated, question/method/finding/implication. Negative findings are recorded as plainly as positive ones — a ruled-out edge-mechanism is exactly as valuable to have on record as a confirmed one.

---

## 2026-08-17 — Market benchmark: does the model beat Polymarket's pre-game odds?

**Question**: on held-out predictions, is our accuracy comparable to, better than, or worse than the market's, and where (if anywhere) does disagreement with the market favor us?

**Method**: `scripts/market_benchmark.py`, fold5 (test window 2025-10-21 → 2026-04-12, genuinely held-out — trained through 2024-04-14, validated 2024-10-22→2025-04-13), joined against Polymarket's pre-game odds (`data/polymarket_prices/games.csv`) by team+date. 1,225 held-out games, 1,225 joined after fixing a date-convention bug in the join (see below). "Winner" per game = whichever side (model or market) had the smaller absolute error vs. the actual result — a magnitude/closeness comparison, not a binary threshold.

**Finding — negative, stated plainly**: the market beat the model on every accuracy metric measured:

| Metric | Model | Market |
|---|---:|---:|
| Diff MAE | 11.49 | **10.81** |
| Total MAE | 15.40 | **14.68** |
| Win accuracy | 67.4% | **69.4%** |
| Brier score | 0.209 | **0.194** (vs. raw price, no de-vig exists for this market) |

Disagreement analysis: on games where the model and market disagree, the market is closer more often (56% vs. 44% on diff calls) — and this gets *worse* for the model on the highest-disagreement half of games specifically (41% model vs. 59% market), not better. That's the opposite of "the edge is hiding in our disagreements." Disagreement-magnitude quartiles show a clean monotonic decline in model win-rate (48% → 37%) from smallest to largest disagreement — the model is least trustworthy exactly where it deviates most from the market.

**One narrow exception**: in `pick'em` games (`|market spread| < 3`, n=244), the model is roughly even with the market (50.4% vs. 49.6%) — its only close-to-competitive bucket. Schedule (back-to-back) and injury-count conditions showed no meaningful differential either way.

**Implication**: on this one season of evidence, the honest read is *not* "the model is fine, the edge just hasn't been found yet" — it trails the market broadly, and its disagreements skew toward being wrong rather than right, with one narrow exception (close games) worth a second look. This is a real, reusable finding, not a final verdict — re-run after any future model change to see if the gap closes. Tool is reusable (`--tag`, appends to `outputs/market_benchmark_summary.csv`) specifically so this can be tracked over time rather than re-derived from scratch each time.

**Data note (mechanical, not a finding)**: the initial join only matched 1,135/1,225 games (92.7%) due to a real date-convention mismatch — Polymarket's `game_date` is UTC calendar date, nba_api's `GAME_DATE` is US/Eastern "game night" date, so any game tipping off after ~8pm ET landed on different calendar dates in the two sources. Fixed by deriving the join date from `game_start_time_utc` converted to US/Eastern instead of the raw `game_date` column — closed the gap to 1,225/1,225 (100%). That fix also surfaced 5 genuine duplicate Polymarket market listings (same real game, two `slug`s) previously invisible under the old, buggy join; resolved by keeping the higher-volume listing per collision.

---

## 2026-08-17 — Injury/rest information-timing: do we know anything before the market does?

**Question**: before hunting for a betting edge, does the one plausible edge-*mechanism* even exist — could our injury or rest data be fresher, more complete, or earlier than what the market prices in?

**Method**: traced the full data-provenance chain for both signal types directly in code (not docstrings) — source, publication timing, and a scan of the actual cached data (not just the schema) for correctness.

**Finding — negative, stated plainly, on both counts**:

- **Injury timing**: both our data paths (`run_historical`'s NBA official injury-report PDF, `run_nightly`'s ESPN injuries page) are **100% public sources** — the exact same documents any sportsbook or bettor watches, at the same publication cadence. Verified the pre-game cutoff itself is timezone-safe end to end: trade timestamps are raw Unix-epoch integers (no interpretation ambiguity), and `game_start_ts` is built from a tz-aware UTC timestamp — confirmed empirically that 0 of 49,731 cached `gameStartTime` values (and 0 of 49,515 `startDate` values) across the *entire* cached history parse as naive/timezone-ambiguous. **No informational-timing edge exists — same source, same timing, zero exclusivity.**
- **Injury data correctness (a second, independent finding surfaced along the way)**: two real bugs found in the injury pipeline, unrelated to timing. (1) A PDF table-extraction failure: the Lakers' 3 clearly-listed "Out" players on 2025-10-21 (LeBron James, Maxi Kleber, Adou Thiero) were silently dropped from structured extraction — `n_out` fell back to 0, indistinguishable from "confirmed healthy." (2) Cross-date contamination: every old-format ("pre-Dec-22-2025") late-night report checked (4/4 sampled) mixes in next-day preview entries that get filed under the wrong date, since the parser never reads the PDF's own per-row "GameDate" column — explains why `injury_features` has more (date, team) row pairs (3,064) than actual scheduled team-games (2,450). Neither bug is leakage (injury reports encode pre-game roster decisions, not outcomes), but both are real correctness/completeness problems.
- **Rest timing**: `rest_days`/`back_to_back`/`games_in_4_nights` are derived entirely from the public NBA schedule (known weeks/months ahead) — no informational edge possible by construction. Checked for a *behavioral* (processing-speed) edge instead: the market-benchmark's own back-to-back bucket showed no differential (model win-rate ~43-44% regardless of B2B status on either side) — empirical evidence against a market-lag-on-rest hypothesis, not just a theoretical argument.

**Implication**: this line of investigation is closed as a source of edge — not because the data is clean and simply unremarkable, but because even a perfectly clean version of this data would still just match what the market already has. Separately, the two injury-pipeline bugs found need fixing before the data is even reliable for *training*, independent of any edge question — logged here since they were discovered during this investigation, but they belong to `docs/PIPELINE_AUDIT.md`'s remit for triage, not this file's.

---

## 2026-08-18 — Calibration: are our stated win-probabilities honest?

**Question**: the EV/betting question depends on whether `model_p_home` (the model's implied win-probability, `norm.cdf(diff_pred/residual_std)`) is trustworthy — never measured before now. If it's systematically off, no +EV analysis built on it means anything, regardless of what Q1-Q3 show about point-estimate accuracy.

**Method**: extended `scripts/market_benchmark.py` (Q4, read-only, no model/metric changes — reuses the exact same held-out predictions and market join Q1-Q3 already build). Same set as the first entry above: fold5, test window 2025-10-21 → 2026-04-12, genuinely held-out (trained through 2024-04-14, validated 2024-10-22→2025-04-13, disjointness mechanically checked). Restricted to the `has_moneyline` subset (n=1,224). Reliability curve = 10 quantile deciles of stated probability vs. actual home-win rate per decile. ECE = Σ(bucket_n/N)·|predicted_mean − actual_rate|, computed identically for the model (`model_p_home`) and the market (`market_p_home`, raw Polymarket price — no de-vig exists for this market, same caveat as Q1's Brier score).

**Finding — real miscalibration, and it has a specific, non-obvious shape**:

- **Headline**: `model_ece = 0.054`, `market_ece = 0.027` — the market is roughly **2× better calibrated** than the model. Consistent with Q1's Brier-score finding (market already won there); this shows *why* in more direct terms.
- **The miscalibration is concentrated, not uniform — specifically at both probability extremes, in the same direction: underconfidence.** The middle of the curve (deciles 3-8, stated probability roughly 0.40-0.76) is close to on-target, gaps mostly under ±0.07 with no consistent sign. But the two ends both show the *actual* outcome more lopsided than the *stated* probability: bottom decile stated 28.0% home-win, actual was only 18.7% (home lost even more often than predicted); top decile stated 80.0%, actual was 86.2% (home won even more often than predicted). Sign-adjusted for direction, that's **underconfidence at both tails, not overconfidence** — when the model does identify a clear favorite, reality is more extreme than it says. This is the opposite of the failure mode usually assumed by default for ML probability outputs.
- **The coarse 3-bucket "confidence extremity" breakdown (toss-up/lean/confident) *hides* this pattern** — its `confident` bucket lumps both tails together, and since the miscalibration direction (raw gap sign) flips between the low and high tail, averaging them nets out to a small number (0.015, the *smallest* of the three buckets) even though the two tails are individually the most miscalibrated part of the curve. Recorded here explicitly as a methodology note: the decile-level reliability curve is the finding; the coarse concentration table is not sensitive enough to see it and should not be read as contradicting it.
- **Other concentration dimensions (market closeness, back-to-back, injury count) showed no clean pattern** — gaps scattered 0.003-0.037 with no consistent direction or magnitude trend, i.e., the extremes-underconfidence pattern is about the model's own stated probability, not about any external game-type condition tested so far.

**Implication**: the model's point-estimate competitiveness in close games (the one positive signal from the first entry above) does **not** extend to trustworthy probabilities — calibration is worse than the market's by a real margin, and it fails in a specific, actionable-to-diagnose way (extremes, both directions, underconfidence) rather than randomly. Per the question this was meant to gate: **no +EV/betting analysis should be built on `model_p_home` as it stands** — the probabilities need calibration work (e.g. isotonic/Platt recalibration, or revisiting the `norm.cdf(diff_pred/residual_std)` construction itself, which uses one global residual std for every game regardless of how extreme the prediction is) before that question can even be asked meaningfully. Re-run this after any calibration fix or model change to confirm it actually closes the gap — nothing proposed or executed here per the read-only scope of this pass.

---

## 2026-08-18 — Recalibration: does a standard post-hoc fix close the gap found above?

**Question**: the calibration finding above was monotonic and shaped like a textbook fixable pattern (systematic underconfidence at both tails). Does actually fitting a standard recalibrator — isotonic regression, Platt scaling — recover it out-of-sample, and does that change the market verdict?

**Method**: extended `scripts/market_benchmark.py` further (Q5). **Leakage discipline, stated exactly**: the calibrators are fit *only* on the champion model's own predictions on fold5's **validation** set (2024-10-22 → 2025-04-13, n=1,225) — the same single trained model instance used everywhere else in this tool, never re-fit. They are then applied to, and evaluated on, the test-set `has_moneyline` subset (2025-10-21 → 2026-04-12, n=1,224) that Q4 already used — chronologically strictly later, zero overlap with the fit data. This is the standard train/calibrate/evaluate split (fit=validation, evaluate=test), not a new mechanism; the date ordering is mechanically asserted in code (`generate_validation_calibration_fit_data` raises if `validation_end_date >= test_start_date` or if any fit-set game falls after `validation_end_date`), not just documented. Fit and apply/evaluate sets share zero games, confirmed by construction (disjoint date ranges, disjoint dataframes).

**Finding — helps, but doesn't close the gap; and the two methods disagree with each other**:

| Metric | Raw model | Isotonic | Platt | Market |
|---|---:|---:|---:|---:|
| ECE | 0.0538 | **0.0410** | 0.0528 | 0.0268 |
| Win accuracy | 0.6738 | 0.6724 | 0.6724 | 0.6943 |
| Brier | 0.2093 | 0.2101 | **0.2088** | 0.1943 |

- **Isotonic regression meaningfully reduces ECE** (0.054 → 0.041, a ~24% reduction) — the flexible, non-parametric mapping does recover some of the tail-underconfidence pattern. **Platt scaling barely moves it** (0.054 → 0.053, ~2%) — the parametric logistic map is too rigid to capture the specific double-tail shape found above (which isn't a single global over/under-confidence bias, so a 1-parameter rescaling has little to fix).
- **Neither gets remotely close to the market's 0.027.** Isotonic's post-recalibration ECE is still ~53% worse than the market's; the gap narrows but does not close.
- **Win-accuracy is essentially unaffected, confirmed empirically rather than assumed** — thresholding the recalibrated probability at 0.5 gives 0.6724 for both methods, versus the raw model's 0.6738 (computed from `diff_pred`'s sign in Q1). The tiny difference (2 games out of 1,224) comes from isotonic's plateau occasionally landing exactly on 0.5; this is not evidence that recalibration changes which side the model actually picks — it doesn't, materially.
- **Brier is a mixed, small-magnitude result, not a second calibration win**: isotonic's Brier is actually slightly *worse* than raw (0.2101 vs 0.2093) despite its better ECE — a known possible side effect of isotonic's flexibility trading away some sharpness/resolution for reliability. Platt's Brier is slightly *better* (0.2088) despite barely improving ECE. Both remain clearly behind the market's 0.1943 either way.

**Implication**: recalibration is a real, measurable, out-of-sample improvement (isotonic specifically) — it is not nothing — but it does not change the market verdict. The model's probabilities go from clearly-worse-than-market to still-clearly-worse-than-market on every metric checked, just by a smaller margin on ECE and not at all on win-accuracy. **Answering the question this session was framed around: recalibration does not fix the underlying finding that the market is better calibrated, and does not on its own create a probability edge to build a betting strategy on.** For the still-pending venue-thinness question: given win-accuracy and Brier are essentially unchanged by recalibration, and venue-thinness (the pick'em-game finding) was itself a point-estimate/closeness result (Q2's magnitude-based "winner," not a probability-based metric), **it should run on the same basis it already ran on (point predictions) — recalibration has nothing to add there, since it only touches the probability column, not `model_diff_pred`.** Nothing proposed or executed beyond this analysis.

---

## 2026-08-18 — Market microstructure: is Polymarket mispriced where it's thin?

**Question**: the last remaining plausible pre-game edge-mechanism (after ruling out informational timing and finding our probabilities are worse-calibrated than the market's) — does Polymarket itself get mispriced in low-liquidity games, independent of anything our model does, giving us an edge purely from market thinness rather than model skill?

**Method**: extended `scripts/market_benchmark.py` (Q6). Point predictions only (`model_diff_pred` vs. actual, Q2's existing `diff_winner` magnitude-closeness definition) — explicitly not probabilities, per the Q5 finding that recalibration never touches `model_diff_pred`, so this test is unaffected by anything in Q4/Q5. Bucketed the fold5 `has_spread` sample (n=1,217) into liquidity quartiles by `spread_volume` (games.csv's own spread-market trading volume — the same *kind* of volume data the earlier date-join dedup fix used, `moneyline_volume`, but `spread_volume` specifically since that's the market a diff/spread comparison is actually about). Vig bar set explicitly, before looking at any bucket: a bucket's model win-rate must exceed 50%+4.5% (54.5%), not just 50%, to plausibly represent real tradeable edge over realistic transaction costs.

**Finding — clean negative, no pocket exists at any liquidity level**:

| Bucket (spread_volume) | n | Model win-rate | Market win-rate |
|---|---:|---:|---:|
| Q1 thinnest ($0–64K) | 305 | 44.3% | 55.7% |
| Q2 ($64K–187K) | 304 | 44.7% | 55.3% |
| Q3 ($187K–458K) | 304 | 42.4% | 57.6% |
| Q4 thickest ($458K–2.2M) | 304 | 43.1% | 56.9% |

- **The model doesn't even beat the market outright (>50%) in any liquidity bucket** — every quartile sits in a narrow 42-45% band, well below even the raw 50% bar, let alone the 54.5% vig bar. There's no need to reach the vig question at all here — nothing clears the lower bar first.
- **The liquidity-thinness hypothesis itself finds no support**: `corr(log(spread_volume), model_advantage) = +0.031` (n=1,217) — statistically indistinguishable from zero, and if anything points the wrong direction (the hypothesis predicted a *negative* correlation, model doing relatively better as volume shrinks). The model's shortfall vs. the market is close to uniform across the full liquidity range, not concentrated in thin markets.
- **Held-out confirmation attempted, found genuinely infeasible, not just skipped**: tried running the same tool against fold4 (2024-25 season) to check whether any of this — or the earlier venue-thinness pick'em signal — replicates on a different season. It hit the tool's own existing sign-convention safety check (`market_diff_home` vs. `actual_diff` correlation came back `NaN`, aborting before producing any result). Quantified directly: **0 of 1,193 games in fold4's test window (2024-10-22→2025-04-13) have any spread data at all** — Polymarket had zero spread-market coverage for that entire regular season (confirmed earlier in this file's first entry's investigation). The only period that season with real spread coverage is the 2025 playoffs (33 games, 85-100% coverage) — but playoff games are structurally excluded from every fold's held-out val/test set by this project's own `datasets_loading.allowed_season_types: ["Regular Season"]` (`configs/config.yaml`), a convention this analysis did not override. **There is currently no way to get a genuine held-out second-season spread confirmation — not because it wasn't tried, but because the data doesn't exist for the only season where it could be checked.**

**Implication**: nothing clears the vig bar on held-out data — and more specifically, nothing clears *any* bar (not even beating the market outright) on the one season this could be tested at all, and a second-season confirmation isn't currently obtainable regardless. Market microstructure/liquidity is closed as a source of edge, same disposition as the injury/rest-timing entry above: not because the check was skipped, but because it was run and came back empty. Combined with the calibration and recalibration entries, all three plausible pre-game edge-mechanisms tested this cycle (informational timing, probability calibration, market thinness) have come back negative. Nothing proposed or executed beyond this analysis.

---

## Pending

- Venue-thinness result (pick'em-game competitiveness, sample-size-qualified) — flagged in the market-benchmark entry above as a narrow positive signal; not yet independently investigated for robustness.
