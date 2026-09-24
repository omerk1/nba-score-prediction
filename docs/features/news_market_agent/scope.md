# Live News / Market-Reaction Agent — Scope

Status: scoped 2026-09-24, nothing built. Origin: the one surviving idea in
`docs/LLM_COMPONENT_OPTIONS.md` (closed investigation, six LLM attempts
rejected). Ships disabled (`news_market_agent.enabled: false`); never feeds the
score model; no autonomous trading.

This is the first thing in the repo where retrieval and an LLM judgment are
plausibly the right tools rather than substitutes for a better one, so the
scope is deliberately heavier on evaluation than on the agent itself.

## 1. Hypothesis

Public pre-game news (intraday official injury-report editions, beat-reporter
posts, lineup announcements) reaches the Polymarket moneyline price with a
measurable lag. An agent that reads the raw item, judges it credible and
material, and compares it to the current price can flag a game *before* the
price moves.

It decomposes into four claims, each separately falsifiable and each owned by
one phase:

| # | Claim | Falsified by |
|---|---|---|
| 1 | The price moves *after* publication, not at it | Phase 0 event study |
| 2 | We can detect an item faster than the market reprices | Phase 1 detection-latency eval |
| 3 | Precedent retrieval says something structured features don't | Phase 2 retrieval diagnostic |
| 4 | An LLM judgment beats a deterministic rule on the same events | Phase 3 ablation ladder |

Any one failing closes the idea at that phase. Claim 1 is the cheapest and runs
first.

## 2. What the record already says (read before building)

- `docs/MARKET_EDGE.md` (2026-08-17): no *informational* edge — these sources
  are the same public documents the market reads. This scope does not dispute
  that; it tests *reaction speed* on the same public information, which that
  entry never measured for news. Its one speed check (rest days) came back
  empty — a mild prior against.
- The repo has never observed intraday updates: `nba_injury_pdf.py` keeps only
  the last report of each day (`scrape_log.report_time` is `11PM`/`11_45PM` on
  all 814 dates). The signal this agent needs is exactly the data currently
  discarded.
- Polymarket trade series exist for all of 2025-26 (`src/polymarket_prices/`,
  `data/polymarket_prices/series`), pre-game coverage reaching 2.6–8.3 days
  before tip-off. Price reaction to a timestamped event is measurable from data
  already on disk.
- Option 4 in `LLM_COMPONENT_OPTIONS.md` (news as a *training feature*) was
  rejected for needing a years-long point-in-time archive. A live agent needs
  no backfill; that rejection does not apply.
- **The retrieval lesson** (`docs/features/reason_retrieval_log.md`): semantic
  retrieval over injury-reason text failed twice, and the diagnosis is the
  design constraint here. Cross-player retrieval returned the league base rate
  because the predictive content was about the *player*, not the words
  (retrieved value correlated +0.009 with the outcome against +0.127 for the
  sparse feature it was densifying). Within-player retrieval correlated 0.755
  with a feature already present and predicted worse than it. Section 5's
  diagnostic exists to catch both failure shapes before anything is swept.

## 3. Phase 0 — does the lag exist (no LLM, no retrieval)

The cheapest test and the gate for everything else. Entirely historical.

1. **Backfill intraday editions.** Establish first whether the 1:30 PM /
   5:30 PM ET and pre-tip editions are still retrievable for past 2025-26
   dates. If they are not, phase 0 cannot run as designed — see the fallback
   below. Store every status *change* between consecutive editions:
   `injury_report_events(report_ts, game_date, team, player, status_before,
   status_after, reason)`.
2. **Event study.** For each change on a player above an importance threshold
   (`src/news_scraping/player_importance.py`), compute the signed moneyline
   move over [t, t+15m], [t, t+30m], [t, t+2h], [t, tip-off], signed toward
   the affected team's side.
3. **Controls.** A placebo of random pre-game timestamps on the same games,
   matched on time-to-tip; and the pre-publication window [t−30m, t] to detect
   the price moving *before* the edition (i.e. the news already out via another
   channel).
4. **Report.** Mean signed move per window with paired-bootstrap CIs, sign hit
   rate, share of the total pre-tip move occurring after publication, and the
   **repricing half-life** — minutes until half the eventual post-publication
   move has happened.

**Gate**: post-publication move distinguishable from placebo, 95% bootstrap CI
excluding zero. If the price moves mostly *before* publication, stop — the
edition is not the market's source and no reader beats an already-priced
signal.

**The half-life is an output, not a footnote.** It sets phase 1's detection
budget and phase 3's latency budget. A half-life of 90 seconds means a
15-minute poll and a 20-second LLM call are both already too slow, and the
honest response is to stop rather than to build them.

**Fallback if editions are unretrievable**: run the same event study on
`player_injuries` day-over-day changes at a coarser resolution to get a
directional read on whether injury news moves this market at all. A coarse
negative still closes the idea; a coarse positive does *not* pass the gate,
since it cannot separate pre- from post-publication.

## 4. Phase 1 — live collector (no LLM)

Runs from the 2026-27 preseason (October 2026); nothing live is possible
before then.

| Source | Cadence | Gives |
|---|---|---|
| Official injury-report editions | Each publication | Authoritative status changes, timestamped |
| ESPN injuries page (`espn_injuries.py`) | Poll | Faster-moving, less structured |
| 1–2 beat-reporter feeds | Poll | Genuinely unstructured; the only novel input |

- **Timestamps.** Every item carries its **fetch timestamp**, never a parsed
  publication date. The injury-report parser filed every listing under the
  containing report's date instead of the game date it referred to
  (`docs/features/injury_pdf_extraction_scope.md`); the same class of bug here
  would silently invent the entire result.
- **Price at first sight.** Snapshot the Polymarket moneyline at each poll
  (`src/polymarket_prices/`), so every item has a price attached at detection.
- **Dedup** across sources on (player, status_after, game) within a window,
  keeping the earliest fetch — the cross-source lead time is itself a finding.
- Store: `data/raw/news_market.sqlite`. Point-in-time by construction.

**Detection-latency eval** (claim 2): for every official status change, the gap
between publication and our fetch. Report the distribution, not the mean.

**Gate**: ≥ 2 weeks of clean collection, < 5% failed polls, and median
detection latency inside the repricing half-life from phase 0. Missing that
last one closes the idea regardless of how good the agent would be.

## 5. Phase 2 — the retrieval layer (RAG, still no judgment)

The corpus is the accumulated item/event store, where every past item has a
**known price outcome** attached from the series data. The retrieval question
is precedent: *given this new item, what did the market do after similar past
items?*

- **Index.** Reuse the embedder, on-disk cache and index in
  `src/availability/reason_retrieval.py`, including its point-in-time rule — a
  neighbour contributes only if its timestamp precedes the query's
  (`tests/test_reason_retrieval.py` exists to enforce exactly this).
- **Query key.** Item text embedding, plus structured filters: player
  importance band, time-to-tip band, market-liquidity band, status transition
  type.
- **Retrieved value.** Neighbours' signed post-publication move at each
  horizon, similarity-weighted, with coverage recorded.
- **Corpus cold start.** At live launch the corpus is only what phase 0
  backfilled. If phase 0 ran in fallback mode, there is no usable corpus and
  this phase waits a season — an acceptable outcome, not a reason to
  improvise.

### The diagnostic that runs before any sweep

Verbatim from what the reason-retrieval work established *after* spending 54
settings to learn it. All three run on the tuning split, in this order, and any
one failing stops the phase:

1. **Does it predict?** Correlate the retrieved neighbour-move with the
   target's actual move, computed on the rows retrieval is meant to help (thin
   or no structured history), against the correlation of the structured-only
   baseline on the same rows. Near-zero where the baseline is materially
   positive is the cross-player failure repeating.
2. **Is it new?** Correlate the retrieved value with the structured baseline
   prediction. High correlation plus worse standalone prediction is the
   within-player failure repeating — a noisier copy of something already
   present.
3. **Is it a plumbing artifact?** Report coverage and neighbour-count
   distribution alongside, so a null is attributable to signal rather than an
   empty index.

**Gate**: retrieval beats a structured-only baseline (player importance ×
time-to-tip × liquidity) at predicting the signed move, on held-out tuning-split
events. Otherwise the agent gets the structured pack only, and the RAG layer is
logged as a tested negative.

## 6. Phase 3 — the judgment

Per new item: context → judgment → flag.

- **Output** (structured): `material` (bool), `direction` (home/away),
  `already_priced` (bool), `confidence` in [0, 1], one-line rationale retained
  for review only, never as an input to anything.
- **Harness**: reuse `GeminiClient`, `prompt_key` caching and the sqlite
  locking already fixed in `src/availability/llm_estimator.py` — both were real
  bugs found there, and re-implementing them would re-introduce them.
- **Tools** (top rung only): fetch current price, fetch the source page.

### The ablation ladder

The closed investigation's defining finding was that results got
*monotonically worse* as prompting sophistication was added. The response is to
make every increment pay for itself, on one shared event set, reported
together:

| Rung | What it sees | Tests |
|---|---|---|
| Rule | Status change + importance threshold | The baseline that must be beaten |
| Text-only | Raw item text | Does reading the text add anything at all |
| + context | Structured pack: listing history, price path, same-game items | Does structure help the read |
| + precedent | Retrieved neighbours from phase 2 | Does RAG earn its place |
| + tools | Live price/source fetch, agentic loop | Does tool use earn its place |

**Adopt the lowest rung that wins.** A higher rung must beat the rung below it,
not merely the rule, to be kept. If text-only loses to the rule, the remaining
rungs are not run — that is the same result as the six closed attempts and
deserves the same quick exit.

**Gate**: the adopted rung beats the rule flagger on sign hit rate and mean
signed move, 95% bootstrap CI excluding zero. Failed twice → log as failed,
move on (CLAUDE.md). Flags stay human-reviewed throughout; no auto-action is in
scope at any rung.

## 7. Evaluation

Each component is judged on its own before it is chained to the next, because
the closed investigation's hardest lesson was that a stacked pipeline hides
which part is failing.

**Splits, fixed before any knob is turned.** Tuning = the phase 0 backfilled
2025-26 events. Sealed = the live 2026-27 season, which is a genuine
prospective out-of-sample test and the strongest evidence this design can
produce. Nothing tuned on the sealed season, ever; if tuning yields no
candidate, the sealed season stays untouched, exactly as the reason-retrieval
work correctly left it.

**Intrinsic (per component).**

| Component | Metric | Bar |
|---|---|---|
| Sources | Recall of known status changes; detection latency distribution | Median latency < repricing half-life |
| Extraction | Precision/recall of (player, team, status, direction) on ~200 hand-labeled items | ≥ 95%, matching the availability work's resolution bar |
| Retrieval | Predictive correlation vs. structured baseline; coverage; precision@k on a labeled relevance sample | Section 5's three-part diagnostic |
| Judgment | Agreement with human labels on `material` / `direction` | Beats the rule on the same labeled set |
| Confidence | Brier and ECE, isotonic recalibration if needed | Calibrated enough that a threshold means something |

Confidence calibration is a gate, not decoration: the whole point of
`confidence` is to set a flag threshold, and `docs/MARKET_EDGE.md` already
found this model family's raw probabilities worse-calibrated than the market's,
with isotonic the fix that actually helped.

**Extrinsic (the real question).** Signed post-flag move at 15m / 30m / 2h /
tip-off, against placebo and against each lower rung. Secondary: hit rate
against the vig bar `docs/MARKET_EDGE.md` set (54.5%) — reported, never a gate,
because one season cannot settle it.

**Human labels.** ~200 items, labeled before any model output is visible, stored
in the db with the labeler and timestamp. Used for extraction and judgment
evals. No LLM-as-judge anywhere: this repo has an evidence base for LLM
judgment being the weak link, so grading itself with one would be circular.

**Cost and latency budget.** Per-item LLM cost and wall-clock from detection to
flag, reported every run, with a configured cap. A rung that wins on accuracy
but exceeds the latency budget has not won — the entire hypothesis is a claim
about speed.

**Sample size, stated up front.** A season yields a few hundred material status
changes. Everything from one season is directional; "inconclusive" is an
allowed and expected outcome, and is not a reason to keep re-cutting the data.

**Memorization.** Irrelevant live (post-cutoff by construction). Any LLM run on
historical 2025-26 items follows the anonymization and named-vs-anonymized
controls in `docs/LLM_COMPONENT_OPTIONS.md`.

## 8. Layout and config

```
src/news_market/
    collector.py       polling, dedup, fetch-timestamped writes
    sources/           injury_editions.py, espn.py, beat_feeds.py
    context.py         structured pack assembly
    retrieval.py       precedent index (wraps availability/reason_retrieval)
    judge.py           prompt, structured output, rung selection
    event_study.py     signed-move computation, placebo, bootstrap CIs
    evaluation.py      intrinsic evals, calibration, ablation table
scripts/run_news_market_backfill.py      phase 0
scripts/run_news_market_collector.py     phase 1
scripts/run_news_market_eval.py          phases 2-3, ablation ladder
data/raw/news_market.sqlite
```

```yaml
news_market_agent:
  enabled: false
  db_path: "data/raw/news_market.sqlite"
  poll_seconds: 900                 # revised down if phase 0's half-life demands it
  importance_threshold: 0.5
  llm_model: "gemini-2.5-flash"
  rung: "rule"                      # rule | text_only | context | precedent | tools
  retrieval:
    enabled: false                  # only after phase 2's diagnostic passes
    neighbours: 10
    similarity_floor: 0.80
  budget:
    max_seconds_to_flag: 60         # set from phase 0's repricing half-life
    max_cost_per_item_usd: 0.01
```

## 9. Hard rules

- Never a feature in `feature_builder.py`; never touches CV folds, the
  harness, or the metric.
- Every stored item carries its fetch timestamp; nothing is backdated.
- Retrieval is point-in-time bound; a neighbour must predate its query.
- The sealed season is never tuned on.
- No trade is ever placed by code in this repo.
- Phase 0 negative → the idea is closed on this codebase, same as the six
  before it.
