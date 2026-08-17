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

## Pending

- Calibration result (distributional/quantile objective vs. market-implied probabilities) — not yet run.
- Venue-thinness result (pick'em-game competitiveness, sample-size-qualified) — flagged in the market-benchmark entry above as a narrow positive signal; not yet independently investigated for robustness.
