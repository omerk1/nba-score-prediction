# New Data Feasibility (Track C)

Read-only feasibility assessment, not modeling. Written for a human decision on
whether/which of these are worth a scoped session — nothing here is a recommendation
to proceed autonomously, and nothing has been integrated. Evaluated against the
post-Track-B state (148 features; `elo_diff`/`elo_momentum`, the 18-column
`style_fingerprint_features`, `rest_features`) plus what's already built and tested
in `src/lineups/`, `src/matchups/`, and `src/news_scraping/`.

---

## 1. Player availability / confirmed lineups (pre-tip)

**Already substantially built, tested, and in production or explicitly rejected —
this is the most-covered of the three candidates, not a gap.**

- **Injury status (who's out), pre-tip, already live.** Two real data paths:
  NBA official injury-report PDFs (`run_historical`, back to 2021-22) and ESPN's
  injuries page (`run_nightly`, same-day). Both feed `injury_features` and, via
  `src/matchups/injury_layer.py`, apply calibrated per-archetype style-fingerprint
  deltas (`style_matchup.injury_impact`, Phase-0 empirical values) — this is a live,
  adopted signal (`injury_features.enabled: true`), not a gap. `MARKET_EDGE.md`'s
  2026-08-17 entry already checked whether this data path offers any *edge*: it's
  100% public, same source/timing the market prices in — no informational edge, but
  that's a market-edge finding, not an accuracy finding; it doesn't argue against
  the feature's value to the model itself, only against betting against the market
  on the strength of it.
- **On/off-court player-impact weighting, tested and rejected.** `on_off_splits`
  (`docs/features/on_off_splits_log.md`, PR #34) is the direct, already-executed
  version of "does knowing who's missing, weighted by their actual on/off impact,
  help" — built on nba_api's `TeamPlayerOnOffSummary`, full 8-season backfill, a
  5-fold expanding-window CV, and a Doubtful-partial-weighting refinement. Result:
  small and mixed (some metrics favor it, others don't; new columns rank bottom-third
  of ~132 features by importance). **Not adopted**, `on_off_splits.enabled: false`,
  parked rather than closed.
- **What's genuinely still missing: confirmed starting lineups** (who actually
  starts tonight, distinct from injury-report status). `src/lineups/lineup_collector.py`
  only wraps `CommonTeamRoster` — a season-wide roster list, explicitly documented as
  "not actual game-day availability" in its own docstring, confirmed by the
  `on_off_splits` phase-1 audit to carry no on-court/lineup information at all.
  - **Source**: no clean nba_api endpoint publishes pre-tip *confirmed* starters
    early enough to be useful — official starting lineups are typically not locked
    and public until roughly 30 minutes before tip (via team PR feeds / RotoWire /
    the NBA's own in-app lineup card), not something crawlable hours ahead.
  - **Timing problem, concrete, not hypothetical**: this project's own injury pull
    (`run_nightly`) already runs same-day, not minutes-before-tip — the existing
    pipeline has no mechanism for a near-tip-time re-run, and building one is a
    different kind of system (a live cron close to game time) than every other data
    path in this repo, which is batch/nightly.
  - **Plausible incremental lift over what's already tested**: low. The
    accuracy-relevant content of "who's starting" is almost entirely subsumed by
    "who's out" (injury status) plus "how much does a missing/reduced-role player
    matter" (on/off splits) — both already built. The marginal information in
    starter-vs-bench beyond Out/Doubtful status (e.g. a healthy player benched for
    matchup reasons) is a thin, noisy signal riding on top of a feature family
    (`on_off_splits`) that already tested weak. No reason to expect it clears a bar
    the more direct version of the same idea didn't.

**Verdict: not worth pursuing as new integration work.** The valuable parts of this
idea are already built (injury status, live) or built-and-rejected (on/off impact
weighting). The one piece that's genuinely missing (confirmed starters) has a real
sourcing/timing obstacle and a low expected incremental yield over what's already
been tested. If revisited, it should be framed as "is `on_off_splits` worth
re-testing with a starter-confirmation gate," not as a new data source.

---

## 2. Pace / possession data

**Already the single highest-importance signal in the model, via a box-score-derived
proxy — this is a refinement candidate on an existing adopted feature, not new
orthogonal data.**

- **What exists today**: `style_fingerprint_features`' `pace_score` (per team,
  decay-weighted mean, `_add_style_fingerprint_features`) is computed from a standard
  box-score possession estimate: `PTS + OPP_PTS + TOV - 0.44*FTA`. Per
  `docs/features/feature_eda_insights.md`, `pace_score` is the #1/#2-ranked feature
  by importance and correlates ~0 with `elo_diff` — genuinely additive information,
  not redundant with Elo. This is already the adopted default
  (`style_matchup.raw_features_enabled: true`).
- **What a real "pace/possession data" integration would add**: nba_api's advanced
  box-score endpoints (`BoxScoreAdvancedV2` / `LeagueDashTeamStats` with
  `measure_type='Advanced'`) expose the NBA's own computed `PACE`/`POSS` per game —
  a more precise number than the box-score formula estimate above (which is itself
  a standard, well-known approximation, not a crude guess, but still an estimate).
- **Source/cost**: no new data source — same `nba_api` package already a hard
  dependency (`requirements.txt`), same per-game or per-team-per-date call pattern
  already used elsewhere (e.g. `TeamPlayerOnOffSummary` in `on_off_splits`, rate
  ceiling already tuned there). No new library, no new cost, no new leakage-timing
  question — historical pace is exactly as safely lagged (`shift(1)`/rolling) as
  every other rolling family already handles.
- **Integration effort**: small — a per-team-game backfill call plus a swap-in
  replacement of the existing `pace_score` formula (or an additional column
  alongside it), following the exact precedent this project already used for
  `style_matchup`'s raw-fingerprint redesign.
- **Plausible lift beyond what's already captured**: **low-to-modest, and this is
  the important caveat.** The box-score pace estimate and the NBA's own official
  `PACE` are near-mechanical restatements of the same underlying box-score numbers
  (both derive from FGA/OREB/TOV/FTA) — expect very high correlation between them,
  in the same near-duplicate range (r≈0.98-1.0) that `docs/NEXT_PHASE_SESSIONS.md`'s
  B1 session (`b1_style_and_rolling_decay_weighted`) already found causes a
  collinear-near-duplicate failure mode: CatBoost splits importance across the two
  correlated columns without adding real generalization signal, or (per B1's
  `rolling_features` result) can even regress slightly from the added complexity.
  A genuinely different, non-redundant angle — true per-possession *efficiency*
  (points per 100 possessions, using the official possession count rather than the
  points-based proxy) is a smaller, more defensible ask than "add possession data"
  broadly, but should be pre-screened for correlation against existing `off_eff`/
  `def_eff`/`pace_score` columns before any CV run, per B1's own stated lesson.

**Verdict: not a Track-C item at all — it's a cheap, low-risk B-series-style
refinement of an existing adopted feature, if pursued.** Recommend reframing as
"swap the pace_score box-score estimate for nba_api's official PACE/POSS, correlation
check first" rather than a new-data integration, and running it with the same
adoption bar as any other feature-representation experiment (majority-of-folds CV
improvement, correlation check against `pace_score`/`off_eff`/`def_eff` before the
CV run, not after).

---

## 3. Play-by-play shot-quality data

**Genuinely new and orthogonal — this is the one real candidate in this batch — but
materially higher cost and lower confidence than the other two, and already scoped
(and deliberately deferred) once before.**

- **Prior scoping**: `docs/BACKLOG.md`'s "Future: Richer Style-Fingerprint Inputs
  (Shot Charts)" entry already flagged this exact idea during the original A7 style-
  fingerprint work — deliberately deferred to keep that session's scope contained,
  revisit condition stated as "the current 5-metric encoding's ceiling looks
  limiting." That condition hasn't obviously been met: Track B's rollup found the
  two mean-only families tested for enrichment (`style_features`, `rolling_features`)
  came back null/regression, and `style_fingerprint_features` itself was explicitly
  *excluded* from the B-series list as already well-represented (6 metrics,
  already decay-weighted) — there's no current evidence the existing encoding is
  the bottleneck, which is the condition that was supposed to trigger revisiting
  this.
- **Source**: nba_api's `shotchartdetail` (shot location, distance, zone, make/miss,
  shot type) or `PlayByPlayV2` (full event stream, including shot clock context in
  some seasons). Both are per-game granular calls — unlike every other data source
  currently in this project (season roster, per-team-per-date checkpoints, per-game
  box-score aggregates), this is the first candidate requiring one API call *per
  game* for anything beyond a single-game spot check.
- **Cost/integration effort — real and non-trivial**:
  - **Backfill scale**: the training/val/test window covers roughly 8 seasons of
    regular-season games (`data/raw/nba_api.sqlite`'s `game` table currently has
    ~12,793 rows, per `docs/PIPELINE_AUDIT.md`'s box-score parity check). At the
    rate-limiting cadence already established elsewhere in this codebase
    (`SLEEP_SECONDS = 0.7` in `lineup_collector.py`), a per-game pull alone is
    ~2.5 hours of pure sleep time before accounting for actual request latency,
    pagination, or the retry/backoff logic every other backfill in this project
    needed (`backfill_player_stats.py`'s exponential-backoff retry,
    `recover_failed_backfill.py`) — this is a multi-hour, multi-session backfill
    effort, not an afternoon script.
  - **Ongoing cost**: a live/production pipeline would need a per-game pull for
    every upcoming prediction too — but shot-location data for a *future* game
    doesn't exist pre-tip by definition, so any live use is necessarily a rolling
    *historical* shot-tendency feature (same point-in-time pattern as everything
    else, `shift(1)`/rolling), not a same-game input. That's fine, but worth being
    explicit about: this can only ever inform "how this team has shot recently,"
    never "how this specific matchup's shots will go," same ceiling every other
    rolling family already has.
  - **Rate-limit/blocking risk**: stats.nba.com is known to rate-limit and
    occasionally block aggressive per-game scraping; this project's existing
    `nba_api` usage is comparatively light (per-team-per-date checkpoints, not
    per-game-event streams) — a shot-chart backfill is a step up in aggressiveness
    that would need its own resilience work, not just reuse of the existing
    checkpoint pattern.
- **What "shot quality" would actually require, and a real caveat on scope**: raw
  shot *location*/zone tendencies (frequency of rim/mid-range/3PT attempts) are
  fetchable directly from `shotchartdetail`. But this project's existing metrics
  already encode a meaningful chunk of that: `three_pt_reliance` (3PA/FGA) and
  `paint_activity` are already shot-mix/zone-derived box-score metrics in
  `style_fingerprint_features` today. A *quality* metric in the stricter sense
  (expected FG% given shot location/type — an xFG-style model) needs more than a
  raw pull: defender-proximity/shot-clock-context data that would make a real
  quality adjustment (vs. a raw volume/zone-mix restatement) is tracking-camera
  data (Second Spectrum), not something nba_api's public endpoints expose reliably.
  Building a genuine xFG baseline from shot location/type/distance alone is a small
  modeling subproject in itself (fit an expected-FG% model, then measure
  actual-minus-expected as the "quality" signal) — a materially bigger ask than "pull
  and average a new box-score-like column," and worth naming as its own risk before
  committing to it.
- **Redundancy check against post-Track-B features**: partial overlap with
  `three_pt_reliance`/`paint_activity` (shot-mix), no overlap with `elo`
  (structurally pace/possession-blind per `docs/EXPLORATION.md`'s correlation work)
  or `rest_features`. The genuinely new axis would be *efficiency conditional on shot
  type/location* (an xFG-style over/under-performance signal), not shot volume/mix,
  which is the part already covered.

**Verdict: the one real candidate for new orthogonal data, but high-cost and
lower-confidence than it might first appear** — a raw shot-zone-mix pull risks
being a near-duplicate of `three_pt_reliance`/`paint_activity` (same failure mode
flagged for pace/possession above); a genuine shot-*quality* signal needs an xFG
model built from scratch, not just a data pull, and the backlog's own stated
revisit condition ("current encoding's ceiling looks limiting") isn't clearly met
by anything found in Track B. If pursued, scope it as two separable questions —
(a) cheap: does raw shot-zone mix add anything beyond `three_pt_reliance`/
`paint_activity` (fast to falsify, reuses existing infra pattern), before (b)
expensive: build and validate an xFG quality signal — rather than one combined
shot-chart integration effort.

---

## Summary table

| Candidate | New data? | Source/cost | Integration effort | Plausible lift beyond current features |
|---|---|---|---|---|
| Player availability / confirmed lineups | No — mostly already built | n/a (existing pipelines) | Low if pursued (starter-confirmation gate only) | Low — injury status live, on/off impact already tested and rejected |
| Pace / possession data | No — refines an existing top feature | None (same `nba_api` dep, existing rate pattern) | Low (swap-in, correlation check first) | Low-to-modest, high near-duplicate risk vs. existing `pace_score` |
| Play-by-play shot-quality | Yes — genuinely new | `nba_api` shot-chart/PBP endpoints, real backfill time + rate-limit risk | High (multi-session backfill; xFG modeling for true "quality") | Moderate-speculative; partial overlap with `three_pt_reliance`/`paint_activity` for the cheap version, genuinely orthogonal only for the expensive xFG version |

**Framed as a decision, not a recommendation to act**: of the three, only shot-quality
data is genuinely new information the model hasn't seen in any form. It's also the
most expensive and the least de-risked (no prior CV/ablation evidence either way,
unlike the other two candidates, which already have real tested results to point to).
The other two are better understood as "revisit an already-tested/adopted feature"
than "new data" — pace/possession as a cheap swap-in refinement worth a small session
if a spare cycle opens up, player availability as effectively closed pending some new
angle nobody has proposed yet. Nothing here has been started or scheduled.
