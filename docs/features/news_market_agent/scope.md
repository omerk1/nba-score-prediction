# Live News / Market-Reaction Agent — Scope

Status: scoped 2026-09-24, nothing built. Origin: the one surviving idea in
`docs/LLM_COMPONENT_OPTIONS.md` (closed investigation, six LLM attempts
rejected). Ships disabled (`news_market_agent.enabled: false`); never feeds the
score model; no autonomous trading.

## 1. Hypothesis

Public pre-game news (intraday official injury-report updates, beat-reporter
posts, lineup announcements) reaches the Polymarket moneyline price with a
measurable lag. An agent that reads the raw item, judges it credible and
material, and compares it to the current price can flag a game *before* the
price moves.

Falsifiable form: for flagged games, the signed price move after the flag (in
the flag's direction) is larger than for unflagged games and for a placebo
sample of random timestamps.

## 2. What the record already says (read before building)

- `docs/MARKET_EDGE.md` (2026-08-17): no *informational* edge — the injury
  sources used here are the same public documents the market reads. This scope
  does not dispute that. It tests *reaction speed* on the same public
  information, which that entry did not measure for news (its one speed check,
  rest days, came back empty — a mild prior against).
- The repo has never observed intraday updates: `nba_injury_pdf.py` keeps only
  the last report of each day (`scrape_log.report_time` is `11PM`/`11_45PM` on
  all 814 dates). The pre-game signal this agent needs is exactly the data
  currently discarded.
- Polymarket trade series exist for the full 2025-26 season
  (`src/polymarket_prices/`, `data/polymarket_prices/series`), with pre-game
  coverage reaching 2.6–8.3 days before tip-off. Price reaction to a
  timestamped event is therefore measurable from data already on disk.
- Option 4 in `LLM_COMPONENT_OPTIONS.md` (news as a *training feature*) was
  rejected for needing a years-long point-in-time news archive. A live agent
  needs no backfill; that rejection does not apply here.

## 3. Phase 0 — is there a lag at all (no LLM)

The cheapest test, and the gate for everything else.

1. Check whether intraday official injury reports (the 1:30 PM / 5:30 PM ET
   and pre-tip editions) are still retrievable for past 2025-26 dates. If yes,
   backfill them to a `injury_report_events(report_ts, game_date, team,
   player, status_before, status_after, reason)` table — every status *change*
   between consecutive editions, with the edition's publish time.
2. Event study on the 2025-26 Polymarket moneyline series: for each status
   change on a player above an importance threshold
   (`src/news_scraping/player_importance.py`), signed price move over
   [t, t+30 min], [t, t+2 h], [t, tip-off], with direction = the affected
   team's side. Compare to a placebo of random pre-game timestamps on the same
   games.
3. Report: mean signed move, sign hit rate, and share of the eventual
   pre-tip move that occurs *after* publication.

**Gate**: a post-publication move that is distinguishable from placebo (95%
bootstrap CI excluding zero). If the market reprices before or within the
edition's own resolution, stop — no reader, however good, beats an
already-priced signal. Log as failed in `docs/EXPERIMENTS.md`.

## 4. Phase 1 — live collector (no LLM)

Runs from the 2026-27 preseason (October 2026); nothing live is possible
before then.

- Poll, per game day: every official injury-report edition, the ESPN
  injuries page (`espn_injuries.py`, already scraped nightly), and one or two
  beat-reporter feeds (RSS or a public timeline endpoint — X/Twitter API cost
  is the main friction and is decided here, not assumed).
- Store every raw item with its **fetch timestamp**, never a parsed
  publication date (the injury-report parser filed every listing under the
  containing report's date instead of the game date it referred to —
  `docs/features/injury_pdf_extraction_scope.md`).
- Snapshot the Polymarket moneyline price at each poll (Gamma/data API
  clients in `src/polymarket_prices/`), so every item has a price at first
  sight.
- New store: `data/raw/news_market.sqlite`. Point-in-time by construction.

**Gate**: ≥ 2 weeks of clean collection with < 5% failed polls.

## 5. Phase 2 — the agent

Per new item: context pack → LLM judgment → flag.

- **Context pack**: the player's own listing history (`player_injuries`),
  importance, current price and the last few hours of price path, any other
  items on the same game today.
- **Judgment** (structured output): `material` (bool), `direction`
  (home/away), `already_priced` (bool), `confidence` in [0, 1], one-line
  rationale kept for review only. Tools available to the call: fetch current
  price, fetch the source page.
- **Harness**: reuse `GeminiClient`, `prompt_key` caching and the sqlite
  locking already fixed in `src/availability/llm_estimator.py`.
- **Baseline that must be beaten**: a deterministic flagger — status change
  on a player above the importance threshold → flag in the affected team's
  direction. This is the availability-agent lesson applied in advance: if the
  LLM only re-reads facts the rule already encodes, it will not win.

**Gate**: on the same event set, the LLM flagger beats the rule flagger on
sign hit rate and mean signed move, 95% bootstrap CI excluding zero. Two
failed attempts → log as failed, move on. Flags are human-reviewed until the
gate passes; no auto-action is in scope at all.

## 6. Evaluation details

- Primary: signed post-flag move at 30 min / 2 h / tip-off, vs. placebo and
  vs. the rule baseline.
- Secondary: hit rate against the vig bar `docs/MARKET_EDGE.md` set
  (54.5%), reported but not a gate — the sample will be too small for a
  season to settle it.
- Sample size is the honest limit: a season yields a few hundred material
  status changes. Everything from one season is directional.
- Memorization: irrelevant live (post-cutoff by construction). Any LLM run on
  historical 2025-26 items must follow the anonymization and named-vs-
  anonymized controls in `docs/LLM_COMPONENT_OPTIONS.md`.

## 7. Layout and config

```
src/news_market/          collector.py, sources/, context.py, judge.py, event_study.py
scripts/run_news_market_collector.py
scripts/run_news_market_event_study.py
data/raw/news_market.sqlite
```

```yaml
news_market_agent:
  enabled: false
  db_path: "data/raw/news_market.sqlite"
  poll_minutes: 15
  importance_threshold: 0.5
  llm_model: "gemini-2.5-flash"
```

## 8. Hard rules

- Never a feature in `feature_builder.py`; never touches CV folds, harness,
  or the metric.
- Every stored item carries its fetch timestamp; nothing is backdated.
- No trade is ever placed by code in this repo.
- Phase 0 negative → the idea is closed on this codebase, same as the six
  before it.
