# Next Phase — Session Guide

Live/in-game is dropped — staying pre-game. Sequence: finish queued fixes
(A), inventory + enrich the representation of top feature families (B),
trim collinearity on the settled feature set (A4, moved here), new
orthogonal data stays optional/last (C).

Read this alongside `CLAUDE.md` and `docs/MARKET_EDGE.md` before starting
any session. One investigation = one session. `/clear` between them.

---

## Track A — Queued model-quality fixes

Run in order, one Claude Code session each, **stop after each**, full CV +
market_benchmark after every item. **A4 (VIF trim) moved to run after
Track B** — no point trimming collinearity in the L10 block before
deciding whether that block's representation is changing.

### A1 — Fingerprint-cache freshness + box-score parity
```
Read CLAUDE.md and docs/EXPERIMENTS.md. Current step: Track A, item 1 of 3
(fingerprint-cache freshness + box-score parity fixes).

Task: audit the style_fingerprint cache for staleness — confirm it's being
rebuilt/invalidated on new box-score data, not silently serving stale values
mid-season. Cross-check fingerprint-derived features against a fresh box-score
pull for a sample of recent games; flag any parity mismatches.

Rules:
- CLAUDE.md rules are inviolable.
- Fix only what you find broken. Do not touch unrelated features.
- After any file write, verify with ls -la and wc -l on the changed files —
  do not assume a write succeeded because the tool call returned success.
- Run the full CV harness and market_benchmark.py after the fix.
- Report: what was stale/mismatched, the fix, CV delta, market_benchmark delta.
- STOP after reporting. Do not start item 2.
```

### A2 — Venue-blind overall-form feature
```
Read CLAUDE.md, docs/EXPERIMENTS.md, docs/PIPELINE_AUDIT.md (re: the
venue-scoped-form finding). Current step: Track A, item 2 of 3.

Task: the pipeline audit found "form" is venue-scoped only (home form from
home games, away form from away games). Add a venue-blind overall-form
feature alongside the existing venue-scoped ones — add, don't replace, so
we can compare importance.

Rules: same as A1 — CLAUDE.md inviolable, verify writes, full CV +
market_benchmark, report family-importance placement of the new feature
vs. the existing top ~6 families. STOP after reporting.
```

### A3 — second_of_b2b
```
Read CLAUDE.md and docs/EXPERIMENTS.md. Current step: Track A, item 3 of 3.

Task: add a second-game-of-back-to-back indicator. Check whether existing
rest-day features already partially capture this — if so, note the overlap
rather than assuming novelty.

Rules: same as prior sessions. STOP after reporting. After this item, do
NOT run VIF trim yet — that's now Track A4, scheduled after Track B.
```

---

## Track B — Rolling-window representation audit + enrichment (the core of this phase)

**The question:** feature importance tells you the *current* representation
of a family is valuable. It doesn't tell you the family's full information
is being captured. A family can look "efficient" either because the
underlying signal really is simple, or because it's richer than a mean
captures and CatBoost is only extracting what a single scalar can offer.
You can't tell these apart from importance rankings — you have to test a
richer representation and see if it adds anything.

This is family-by-family, not a blanket "add std everywhere":
- **Rolling box-score aggregates** (point differential, shooting splits
  over L5/L10/L20, etc.) are the classic mean-only case — plausibly losing
  variance, trend, shape.
- **Elo** is a running state variable, not a window aggregate — "add std"
  doesn't obviously apply. The parallel question is whether elo
  *volatility* or *rate of change* is captured, vs. only point-in-time
  rating.
- **Rest** is close to irreducibly scalar (days since last game); the
  schedule-density idea from the original A2 scope is the vector version
  of this question, already partly covered.
- **style_fingerprint** — check what its current output vector actually
  is before assuming it needs enriching; the name suggests it may already
  be multi-dimensional.

### B0 — Inventory (read-only, do this first)
```
Read CLAUDE.md, docs/EXPERIMENTS.md, docs/EXPLORATION.md. Current step:
Track B, item 0 — representation inventory. READ-ONLY. Do not modify any
feature code or retrain anything in this session.

For each of the top ~6 families in the family-importance inventory
(elo, style_fingerprint, rest/schedule, and the others — pull the actual
list from docs/), document precisely:
- What is the current output vector? List every dimension/column this
  family actually emits today.
- For any rolling-window aggregate: is it mean-only, or does it already
  include std/trend/other moments? State this per window length if it
  varies (L5 vs L10 vs L20).
- For elo: is there any feature capturing volatility or rate-of-change,
  or only point-in-time rating?
- For style_fingerprint: what is its actual dimensionality — confirm
  whether it's already multi-dimensional before assuming it's a candidate
  for enrichment.
- For rest/schedule: confirm what's covered by the existing scalar
  features vs. what a schedule-density vector would add (cross-reference
  the original A2 finding if relevant).

Deliverable: docs/FEATURE_REPRESENTATION_AUDIT.md — one section per family,
factual inventory only. Then a short prioritized list: which families are
plausibly under-represented (scalar-only where richer representation is
possible) and worth a B-series enrichment session, ranked by how much
importance that family already carries (higher current importance +
scalar-only = higher priority, since that's where a representation fix is
most likely to move CV/market_benchmark). STOP after the doc is written.
Do not implement anything.
```

### B1+ — Per-family enrichment (one session per family, in priority order from B0)
```
Read CLAUDE.md, docs/EXPERIMENTS.md, docs/FEATURE_REPRESENTATION_AUDIT.md.
Current step: Track B, enrichment for [family name] (priority #[N] from
the B0 inventory).

Task: for [family name]'s [specific window/feature] currently represented
as [current representation, e.g. "mean only, L10"], add a richer
representation as NEW features alongside the existing ones — don't replace,
so we can compare. Candidates to test (adapt to what B0 found is missing):
std, trend/slope over the window, min/max, recency-weighted decay. Test
each as a separate addition if feasible, or as a small bundle if that's
more practical — report results for each addition individually either way,
including any that show no effect.

Rules:
- Only this family. Do not fold in other families even if related.
- CLAUDE.md inviolable. Verify all writes with ls -la / wc -l.
- Full CV harness + market_benchmark.py before/after, for each addition.
- Report CV delta, market_benchmark delta, and new-feature placement in
  family-importance ranking. A null result (no movement) is a complete,
  valid outcome — report it plainly, don't keep adding variants to find
  one that "works." One or a small bundle of test additions per session,
  not an open-ended search.
- STOP after reporting. Do not start the next family's session.
```

Repeat B1+ for each family B0 flagged, in priority order, same
stop-after-each discipline as Track A. Note on scope: this is deliberately
*not* an open-ended trial-and-error search — each session tests a small,
specific, hypothesis-driven set of representation candidates per family,
not "try many things and see what correlates." That keeps each result
interpretable as a real test rather than one lucky hit in a wide search
scored against the same CV folds (a genuine risk with broad automated
feature search — it's easy to find "improvements" that are just multiple-
comparisons noise against a finite set of folds).

### B-final — Rollup
```
Read CLAUDE.md, docs/EXPERIMENTS.md, docs/FEATURE_REPRESENTATION_AUDIT.md.
Current step: Track B rollup, after all per-family enrichment sessions.

Produce a cumulative comparison: pre-Track-B baseline vs. fully enriched
model, CV delta and market_benchmark delta. Save to
docs/FEATURE_REPRESENTATION_AUDIT.md. STOP — do not proceed to A4 or
Track C without a separate human decision.
```

---

## A4 — VIF L10 trim (moved here, run after Track B settles)

```
Read CLAUDE.md, docs/EXPERIMENTS.md, docs/FEATURE_REPRESENTATION_AUDIT.md.
Current step: A4 — VIF trim, run on the post-Track-B feature set (not the
original one, since Track B may have added new L10-window features).

Task: run VIF analysis on the full L10 feature block as it now stands,
identify high-collinearity redundancy, trim.

Rules: same as prior sessions — CLAUDE.md inviolable, verify writes, full
CV + market_benchmark. Report cumulative delta from the very original
baseline (pre-Track-A) through Track A + Track B + this trim, so there's
one clear before/after for the whole phase. Save to docs/EXPERIMENTS.md.
STOP after reporting.
```

---

## Track C — New orthogonal data (optional, deprioritized — only after A4)

```
Read docs/MARKET_EDGE.md, docs/PIPELINE_AUDIT.md, and
docs/FEATURE_REPRESENTATION_AUDIT.md. Read-only feasibility task, not
modeling.

Assess obtainability and plausible lift for: player availability/confirmed
lineups (pre-tip), pace/possession data, play-by-play shot-quality data.
For each: source, cost, integration effort, and whether it plausibly adds
signal beyond what elo/style_fingerprint/rest capture post-Track-B, or is
redundant with it.

Deliverable: docs/NEW_DATA_FEASIBILITY.md, framed as proposals for a human
decision, not autonomous next steps. Do not start integrating any source.
```

---

## Track D — Ongoing hygiene (applies to every session above)

- One investigation per session. `/clear` before starting a new track or
  queue item. `/compact` only within a single long investigation.
- Every fresh session opens with: "Read CLAUDE.md + docs/[relevant].md,
  current step is [X]."
- Every file-write claim gets verified with `ls -la` / `wc -l` before being
  reported as done.
- A negative or null result is a complete, valid deliverable — don't keep
  a session going "just in case" after a clean finding, and don't let a
  per-family enrichment session widen into an open-ended search for a
  candidate that finally moves the number.

---

## Why this ordering

A finishes what's committed. B is now the center of the phase: an
inventory that replaces guessing with facts, then targeted, one-family,
hypothesis-driven enrichment — not a broad search — so any CV/benchmark
movement is trustworthy rather than a multiple-comparisons artifact. A4
moves after B so collinearity gets trimmed once, on the final
representation, not twice. C stays last and optional — it's a bet on new
data mattering more than better-represented data you already have, worth
revisiting only if B genuinely runs dry.