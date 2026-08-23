# Next Phase — Session Guide

Live/in-game is dropped — staying pre-game. Sequence: finish queued fixes
(A), inventory + enrich the representation of top feature families (B),
trim collinearity on the settled feature set (A4, moved here), new
orthogonal data stays optional/last (C).

Read this alongside `CLAUDE.md` and `docs/MARKET_EDGE.md` before starting
any session. One investigation = one session. `/clear` between them.

---

## Status log (every session updates this before stopping)

This doc is not a fixed plan — it's the continuity mechanism between
sessions, so it has to reflect what actually happened, not just what was
intended. Every session prompt below ends with "append a status log entry"
— that means literally editing this section before stopping, not just
reporting results in the chat transcript that gets `/clear`'d away.

Each entry, in order completed:

```
### [Track/item, e.g. "A1"] — [date]
Result: [CV delta, market_benchmark delta, or "audit only, no numeric change"]
Findings: [anything unexpected, one or two lines]
Adjusts later steps: [does this change priority/scope of any item below?
"None" is a valid answer]
```

**Before starting any session, read this log first** — not just the
static prompt for that item — in case an earlier session already changed
its scope or priority. If a prompt below conflicts with something the log
says happened, the log wins; update the prompt to match before running it.

### B0 — 2026-08-20
Result: audit only, no numeric change. Deliverable: `docs/FEATURE_REPRESENTATION_AUDIT.md`.
Findings: `style_fingerprint_features` (rank 1, 29.0%) is already 6-dimensional and
already decay-weighted (`halflife=13.2`) — not an enrichment candidate despite being the
top-importance family. `style_features`/`rolling_features` (ranks 2/3, 18.9%/16.1%) are
both genuinely mean-only (or volume-weighted-ratio-only) at every window (L5/L10/L20),
no std/trend anywhere. `elo_features` (rank 4, 8.3%) has zero volatility/rate-of-change
representation — only a point-in-time running rating. `opponent_quality_features` (rank
6, 7.7%) is mean-only but structurally capped by schedule balance (per
`docs/EXPERIMENTS.md` §2), not a representation gap. `matchup_features` (rank 5, 7.9%)
is purely derived from families 2/3, no independent representation of its own.
`rest_features` isn't top-6 by importance (11th, 1.4%) despite being named in Track B's
framing, but the rest/back-to-back differential (`b2b_diff`/`rest_diff`) from
`docs/EXPLORATION.md`'s Area 2 is a specific, already-measured, unimplemented feature —
Track A item 3 fixed the underlying venue-blind bug and confirmed `second_of_b2b` would
be a pure duplicate, but did not add the separate `b2b_diff`/`rest_diff` differential.
Adjusts later steps: priority order for B1+ is (1) `style_features` +
`rolling_features` decay-weighting swap, bundled as one session — highest importance
among genuinely enrichable families, cheapest test (zero new columns, reuse
`fingerprint.py`'s proven `_decayed_weighted_mean`); (2) `elo_features`
volatility/rate-of-change — real importance, untested axis, but needs new construction
(no drop-in swap available); (3) `opponent_quality_features` and `matchup_features` —
deprioritized, first for a measured structural ceiling, second for having no independent
representation to enrich; (4) `style_fingerprint_features` — excluded from the B-series
list entirely, no gap to close. Flagging one cross-cutting note for whoever sequences
B1+: `rest_features`' differential idea ranks #1 by evidence quality in
`docs/EXPERIMENTS.md`'s own decisive shortlist despite ranking 11th by importance —
worth considering for an early slot on evidence-quality grounds even though the
family-importance-first ordering above puts it last.

### B0 correction pass — 2026-08-20
Result: audit only, no numeric change. Edited `docs/FEATURE_REPRESENTATION_AUDIT.md`'s
priority-list justifications only — no code, training, or family-importance re-run.
Findings: (1) `elo_features`' priority-3 justification cited a permutation-importance
number (0.0303, rank 2) from the pre-Track-A family-importance run, while every other
family in the list is ordered by the newer post-A2 CatBoost-share numbers — a
methodology mismatch. Removed the permutation citation entirely; re-verified priority 3
holds on two grounds that don't need it: elo's post-A2 CatBoost share (8.3%) is already
the highest among the non-swap, non-excluded families under the same metric used
throughout the doc, and a volatility/rate-of-change feature has no existing function to
reuse (unlike items 1-2's drop-in decay-weight swap), so it's correctly costed as more
expensive/lower-confidence than the swap regardless of which importance metric is used.
(2) Checked whether `opponent_quality_features` had the same stale-permutation-number
issue — it doesn't, because no post-A2 permutation number was ever logged for it (only
`rolling_features` got a fresh permutation delta in the A2 log entry), so there's no
newer figure to compare against and nothing to find. Its deprioritization was also never
built on a permutation number in the first place — it rests on the schedule-balance
construction argument alone, independent of any importance metric. No finding
manufactured for this family; stated plainly in the doc instead.
Adjusts later steps: **none — priority order for B1+ is unchanged**
(style_features+rolling_features swap, then elo, then opponent_quality/matchup
deprioritized, style_fingerprint excluded, rest_features flagged separately on
evidence-quality grounds; see the B0 entry above for the full order). Only the stated
justification for elo's rank changed, not the rank itself or any other family's rank.

### B1 (style_features + rolling_features decay-weighting swap) — 2026-08-21
Result: rejected, both steps. Step 1 (`style_features`'s `off_eff_L{w}`/`def_eff_L{w}` →
decay-weighted, added alongside as new cols): full CV val_score_mean 1.3814 vs. baseline
(`a3_rest_venue_blind_fix`) 1.3811, Δ+0.0003, 2/5 folds improve, 3/5 regress — null,
`market_benchmark` mixed (diff_mae worse, total_mae better, win_acc worse, brier flat).
Step 2 (also `rolling_features`'s `win_pct_L{w}`/`diff_avg_L{w}`/`win_pct_overall_L{w}`/
`diff_avg_overall_L{w}` → decay-weighted): full CV val_score_mean 1.3832 cumulative
(Δ+0.0021 vs. baseline) / Δ+0.0018 incremental from step 1 — `rolling_features`' own
contribution regresses 3/5 folds incrementally, 4/5 folds cumulatively vs. baseline.
`market_benchmark` unanimous regression on all 4 metrics vs. both baseline and step 1.
Full write-up: `docs/EXPERIMENTS.md`'s `b1_style_features_decay_weighted` /
`b1_style_and_rolling_decay_weighted` entry.
Findings: diagnostic single_split importance run showed 7/12 `style_features` decay
columns land in the top-20 CatBoost-importance features (all 12 in top 33), but none of
the 24 `rolling_features` decay columns crack the top 40 — and every flat-mean/
decay-weighted column pair checked correlates at 0.985-0.998 (near-duplicate, since
`halflife=13.2` games is long relative to L5/L10/L20 windows). This explains both
results: high split-importance without real generalization gain for `style_features`
(collinear near-duplicate, CatBoost splits importance across it and the original without
adding signal), and a real regression for `rolling_features` (same collinearity, but
these columns rank low to begin with, so the added complexity is pure cost with no
offsetting benefit). Code reverted (`git checkout` on `feature_builder.py`) — no
decay-weighted columns left in the feature set; the two CV rows + two market_benchmark
rows stay in `outputs/experiments_v2.csv`/`outputs/market_benchmark_summary.csv` as
permanent negative-result evidence.
Adjusts later steps: **elo_features (priority #3) is unaffected in scope** — its own
audit finding (zero volatility/rate-of-change representation, no existing function to
reuse) is independent of this result and was never contingent on B1 succeeding. This
result does add one general caution for however elo's volatility feature ends up built:
watch for the same near-duplicate-collinearity failure mode if any candidate
construction (e.g. a rolling std/slope of Elo ratings) ends up highly correlated with
the existing point-in-time `elo_diff` — check correlation against the base feature
before or alongside the CV run, not only after a null/negative result, given this
session's evidence that CatBoost's importance ranking alone doesn't distinguish "adds
signal" from "correlated near-duplicate." No change to the deprioritization of
`opponent_quality_features`/`matchup_features` or the exclusion of
`style_fingerprint_features` — those calls didn't depend on this result either. A4 (VIF
trim, still deferred until Track B settles) remains appropriately scoped for exactly
this class of near-duplicate-collinearity issue, now with one concrete empirical
instance (0.985-0.998 correlation) as a worked example of why it matters, though A4
itself has nothing to trim here since the code was reverted rather than adopted.

### B2 (elo_features momentum + volatility enrichment) — 2026-08-21
Result: **split — momentum adopted, volatility rejected.** Step 1
(`elo_momentum_L{5,10,20}`, 9 new cols): full CV val_score_mean 1.3803 vs. baseline
(`a3_rest_venue_blind_fix`) 1.3811, Δ−0.0008, 3/5 folds improve (no catastrophic
regressions) — a real, modest improvement. `market_benchmark` leans positive (win_acc
+0.0057, total_mae/brier better, only diff_mae slightly worse). Correlation with
`elo_diff` (computed before CV, per this session's explicit instruction): 0.20-0.36 —
low-to-moderate, genuinely distinct information, unlike B1's 0.985-0.998 near-duplicates.
**Adopted as an always-on structural feature** (139→148 total columns).
Step 2 (also `elo_volatility_L{5,10,20}`, rolling std of rating deltas, 9 more cols):
val_score_mean 1.3816 cumulative (Δ+0.0005 vs. baseline, roughly flat) but Δ+0.0013
incremental from step 1 — volatility's own contribution is a regression, driven
substantially by one large single-fold miss (fold3 +0.0099). `market_benchmark`
unanimous regression on all 4 metrics vs. both baseline and step 1. Correlation with
`elo_diff` (~0) and momentum (0.06-0.10) ruled out B1-style collinearity up front, but
the CV/benchmark result was negative anyway — diagnosed as a different failure mode: a
real, non-collinear signal (own CatBoost importance rank 23-33/157) that's simply too
noisy to generalize (std over only 2-20 delta values is a high-variance estimator at
these window lengths). **Rejected** — `compute_elo_volatility` and its wiring/test
removed after the result came in; momentum's wiring/test kept.
Full write-up: `docs/EXPERIMENTS.md`'s `b2_elo_momentum` / `b2_elo_momentum_and_volatility`
entry.
Findings: the pre-CV correlation check (adopted from the B1 postmortem, applied here for
the first time) correctly predicted momentum would NOT suffer B1's collinearity failure
mode, and it didn't — but it also correctly showed volatility wasn't collinear either,
and volatility still failed, for an unrelated reason (small-sample noise, not
redundancy). So the correlation check is necessary-but-not-sufficient for predicting
generalization: it rules out one specific failure mode, not all of them. Worth carrying
forward as standard practice for any future new-construction feature (elo or otherwise),
but not as a green light on its own.
Adjusts later steps: this was the last priority family from the B0 inventory (style+
rolling bundled #1-2, elo #3; opponent_quality/matchup deprioritized; style_fingerprint
excluded) — **a Track B rollup (per the B-final template) is now appropriate**, not
started in this session per its own scope (elo only). The rollup should compare
pre-Track-B baseline (`a3_rest_venue_blind_fix`, 1.3811) against the current state
(elo_momentum only survives from the B-series, 1.3803) — a small net win for Track B as
a whole so far. A4 (VIF trim) stays correctly deferred until after that rollup, per the
existing plan; nothing about elo momentum specifically demands an early A4 (only 9 new,
non-collinear columns, no known redundancy to trim).

### B-final (Track B rollup) — 2026-08-22
Result: rollup only, numbers pulled from B1/B2's already-logged rows, nothing re-run.
Pre-Track-B baseline (`a3_rest_venue_blind_fix`, 139 features) → current live state
(`b2_elo_momentum`, 148 features, the only surviving B-series addition): full CV
val_score_mean 1.3811 → 1.3803, Δ**−0.0008** (3/5 folds improve, largest regression only
+0.0019). `market_benchmark` (fold5): diff_mae +0.018 worse, total_mae −0.009 better,
win_acc +0.0057 better, brier −0.0002 flat/better — 3/4 metrics improve. Full numbers
and the per-family yield table saved to `docs/FEATURE_REPRESENTATION_AUDIT.md`'s new
"Track B rollup" section.
Findings: of the three B0-priority families, only `elo_features` produced an adopted
feature (`elo_momentum`, 1 of its own 2 tested candidates); `style_features` and
`rolling_features` were both tested and rejected (a confirmed null and a confirmed small
regression respectively) — real, documented information, but zero net feature-set
change. Stated plainly per this session's instruction: the cumulative movement (Δ−0.0008)
is modest, on the same order as this project's established noise floor, smaller than
earlier clear wins in this project's history (`target_lambda_weight_0.75`,
`target_formulation_diff_total`) — a real, adoption-bar-clearing result, not a
breakthrough. Two of three targeted families yielded nothing.
Adjusts later steps: **A4 (VIF trim) is next**, per the existing plan — Track B is
settled (all B0-priority families have had their session; `style_fingerprint_features`
was excluded from the start, `opponent_quality_features`/`matchup_features` stay
deprioritized on their original structural grounds, unaffected by this rollup). A4
should run on the current 148-feature set (139 baseline + elo_momentum's 9 non-collinear
columns) — no known redundancy to trim from elo_momentum specifically (correlation with
`elo_diff` was checked and found low, 0.20-0.36), so A4's scope is unchanged from what
was already planned, not expanded by this track. Track C stays last/optional per the
original ordering. **Per this session's explicit instruction, A4/Track C are not started
here — that's a separate human decision.**

### A4 (VIF trim, L10/L20 feature block) — 2026-08-22
Result: rejected, not adopted. Full-feature-set VIF (`scripts/full_feature_vif.py`, new
script) on the live 148-column feature set flagged 22 columns above VIF=10; the largest,
cleanest cluster was `_add_rolling_features`' venue-blind "overall" `win_pct`/`diff_avg`
at `L10`/`L20` (8 cols, VIF 19.5-36.5 — the top 8 of the whole ranking), vs. the same
block's own `L5` columns (VIF 9.1-9.6, the only window not flagged). Tested dropping the
`L10`/`L20` overall columns (148→140 features): full CV val_score_mean 1.3801 vs.
baseline (`b2_elo_momentum`) 1.3803 — Δ−0.0002, but 3 of 5 folds regress (only 2 improve),
failing the fold-majority guardrail despite the near-flat mean. `market_benchmark`
leaned slightly negative too (3/4 metrics worse, 1 flat). Code reverted to the
post-Track-B state; both rows kept in the CSVs as negative-result evidence. Full
write-up: `docs/EXPERIMENTS.md`'s `a4_vif_trim_overall_form` entry, including the
phase-wide cumulative delta (pre-Track-A `target_lambda_weight_0.75`, 127 features →
final `b2_elo_momentum` state, 148 features: CV Δ−0.0018, 4/5 folds improve;
`market_benchmark` unanimous improvement, all 4 metrics).
Findings: an incidental process error this session — `git checkout --` (intended to
revert only the trim edit) discarded the *entire* uncommitted `feature_builder.py` diff,
including the not-yet-committed `b2_elo_momentum` wiring from the prior session. Caught
immediately via `git status`/`git diff HEAD`, and the wiring was reconstructed by hand
(re-adding the `compute_elo_momentum` call + column merge in `_add_elo_features`) and
verified by re-running full CV — reproduced `b2_elo_momentum`'s exact numbers (mean
1.3803, per-fold 1.4336/1.3888/1.3708/1.3479/1.3605, byte-identical to 4dp) before
proceeding, so no work was actually lost, but this is a sharp reminder that `git checkout
-- <file>` on a file with *prior* uncommitted changes reverts all of them, not just the
change just made — a narrower tool (manual edit reversal, or committing intermediate
states before testing a risky one) is safer when a file already carries unstaged work
worth keeping in future sessions of this kind.
Adjusts later steps: none — A4 is closed, rejected. The other 14 VIF-flagged-but-untested
columns (style_fingerprint `offensive_rating`/`defensive_rating`, raw `elo_features`
levels, `opponent_quality_features` `L20`, venue-scoped `win_pct_L10`/`fg_pct_L10`) are
left flagged in `outputs/full_feature_vif_a4_diag.csv` for a future session's own
judgment call — none is queued. Track C (new orthogonal data) is next per the original
ordering, pending a separate human decision; the live feature set stays exactly at
`b2_elo_momentum`'s 148 columns.

### Track C (new orthogonal data feasibility) — 2026-08-22
Result: audit only, no numeric change. Deliverable: `docs/NEW_DATA_FEASIBILITY.md`.
Findings: (1) Player availability/confirmed lineups — mostly already built or
already tested. Injury status (who's out) is a live, adopted pipeline (NBA PDF +
ESPN nightly → `injury_layer.py`'s archetype-based style-fingerprint adjustment);
`MARKET_EDGE.md` already found this offers no *market* edge (100% public, same
timing), which doesn't argue against its accuracy value. On/off-court impact
weighting for missing players (`on_off_splits`) was fully built, backfilled, CV'd,
and found small/mixed — parked, not adopted. The one real gap (confirmed starting
lineups, distinct from injury status) has a genuine sourcing/timing problem
(typically public ~30min pre-tip, this project's pipeline is same-day/nightly, not
near-tip) and low expected incremental lift over the already-tested on/off-splits
result. (2) Pace/possession data — not new data at all: `pace_score`
(`style_fingerprint_features`) is already the #1/#2-importance feature, computed
from a standard box-score possession estimate. nba_api's official `PACE`/`POSS`
would be a more precise version of the same signal, cheap to fetch (existing dep,
existing rate-limit pattern), but flagged as high near-duplicate risk against the
existing proxy (same collinear-near-duplicate failure mode B1 already found for
decay-weighted columns) — reframed as a cheap B-series-style refinement, not a
Track-C item. (3) Play-by-play shot-quality — the one genuinely new/orthogonal
candidate, but high-cost (per-game granular calls, ~12,793-game backfill scale,
multi-hour/multi-session effort, real stats.nba.com rate-limit/blocking risk) and
lower-confidence than it first appears: raw shot-zone-mix risks being a
near-duplicate of existing `three_pt_reliance`/`paint_activity`; a genuine
shot-*quality* (xFG-style) signal needs a small modeling subproject of its own, not
just a data pull, since nba_api's public endpoints don't expose defender-proximity/
tracking data. Already scoped once before (`docs/BACKLOG.md`'s "Richer
Style-Fingerprint Inputs (Shot Charts)" entry, deferred) — its stated revisit
condition ("current 5-metric encoding's ceiling looks limiting") isn't clearly met
by anything Track B found (style_fingerprint was explicitly excluded from B-series
as already well-represented).
Adjusts later steps: none — this is a feasibility read for a human decision, not a
scheduling change. If pursued, recommends splitting shot-quality into a cheap
zone-mix redundancy check first, before any xFG modeling work; recommends
reframing pace/possession as a small swap-in refinement (correlation check against
`pace_score` before any CV run) rather than a full Track-C session; recommends
treating player-availability/lineups as effectively closed pending a genuinely new
angle. Full source/cost/effort/redundancy breakdown and a summary table in
`docs/NEW_DATA_FEASIBILITY.md`. Nothing implemented or scheduled.

### Track C follow-up (pace/possession swap-in test) — 2026-08-23
Result: **rejected, not promoted.** Full 5-fold CV: val_score_mean 1.3832 vs.
baseline (`b2_elo_momentum`) 1.3803 — Δ+0.0029, 4/5 folds regress (only fold5
improves). `market_benchmark` 3/4 metrics worse (diff_mae, total_mae, brier;
win_acc slightly better). Full write-up:
`docs/EXPERIMENTS.md`'s `official_pace_poss_new_columns` entry.
Findings: implemented `src/matchups/pace_possession.py` (new nba_api
`TeamGameLogs`/Advanced collection module, season-level bulk calls — ~20 calls,
~1 minute, not per-game) + a new `team_advanced_stats` cache table, wired
official PACE/POSS into `fingerprint.py`'s rolling decay-weighted computation
(Layer 1/uncalibrated only, same scope cut as `offensive_rating`) and
`feature_builder.py` (gated by new `style_matchup.official_pace_enabled`, added
alongside `pace_score`, not replacing it — 148→154 features when enabled).
Backfill: 12,793/12,793 games, 100% parity. Pre-CV correlation check (required
by this session's own instruction): `official_pace` vs. `pace_score` — 0.72-0.73
(moderate, NOT a B1-style near-duplicate of the existing feature, so the
standard collinearity-against-existing-features check would not have flagged
this in advance). But `official_pace` vs. `official_poss` (the two NEW columns
against **each other**) — 0.976, a near-duplicate pair. Diagnosis: single_split
importance shows both new columns landing very high (`away_style_official_pace`
rank 4/154, `away_style_official_poss` rank 7/154) — the same "high in-sample
importance, no CV generalization" signature B1 found, but via a different
mechanism than B1's: not redundancy with an existing feature, but redundancy
**between the two new columns themselves** (CatBoost splits importance across
two near-identical new axes instead of one, without adding two independent
units of real signal). Code kept (not reverted, unlike `b1`/`a4`) since the
collection infra is a genuine, cheap, working data source — `official_pace_enabled`
stays `false`, same treatment as the `injury_features.missing_value_strategy`
rejected variant (kept as a disabled option, not deleted), in case a future
session wants to test pace or poss alone rather than both together. One
pre-existing test fixture needed a fix (a bare `MagicMock()` attribute is
truthy by default, so `tests/test_style_fingerprint_features.py`'s mock config
needed `official_pace_enabled=False` set explicitly) — unrelated to the
ablation result itself, a mechanical fallout of adding a new boolean config
field. Full suite: 192/192 passing with the flag at its shipped default (false).
Adjusts later steps: closes the pace/possession half of
`docs/NEW_DATA_FEASIBILITY.md`'s three candidates as tested-and-rejected (that
doc's prior verdict — "cheap swap-in worth testing" — now has a real, negative,
CV-backed answer instead of a prediction). Relevant to the remaining shot-quality
candidate specifically: this session shows that a correlation check against
*existing* features (the standard B1-derived discipline) is not sufficient when
a candidate adds multiple new columns at once — check collinearity **among the
new columns themselves** too, before CV, not just against what's already in the
model. Shot-quality would likely add several zone/quality columns in one batch,
the same shape as this session's pace/poss pair — worth carrying this specific
lesson into that session if it's ever run. No change to Track C's overall
standing (still optional/deprioritized, per the original ordering) — this was a
single-candidate follow-up, not a new track.

(Log entries go here as sessions complete.)

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

Before stopping: append a status log entry to docs/NEXT_PHASE_SESSIONS.md
per the Status log section — "Adjusts later steps" should list the
priority order for B1+ sessions, since that order isn't decided until this
inventory exists.
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
- Before stopping: append a status log entry. "Adjusts later steps" should
  say explicitly whether this result changes the priority or scope of the
  remaining B-series families — e.g. a strong hit might make a related
  family worth promoting; a clean null might deprioritize a family you
  expected to matter for similar reasons. "No change to remaining order"
  is a valid entry if that's genuinely the case.
```

Before running each subsequent B1+ session, re-read the status log — the
priority order is whatever B0 set *as adjusted by* any later entries, not
a fixed list decided once. Repeat B1+ for each family, same
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

## Backlog — creative feature engineering (untested)

Candidate ideas only — not scheduled work, no priority order, no session
scoped for any of these yet. Each was checked against existing scripts/docs
before being logged here (same discipline as B0's inventory: don't assume
novelty without checking).

1. **Explicit trend/slope over rolling windows** (linear-regression slope of
   efficiency over L10). Distinct from the decay-weighted mean tested and
   rejected in B1. **Partial overlap, not the same construction**:
   `docs/EXPLORATION.md` (Area 1) already tested a *proxy* for trend —
   `trend = diff_avg_L5 − diff_avg_L20` — and found it structurally, not
   just empirically, redundant (exact linear combination of two columns
   already in the model, residual correlation exactly 0.0000 after
   regressing on the L5/L10/L20 baseline). That result is about *this
   specific two-point-difference proxy*, not a true within-window
   linear-regression slope (which weights every game in the window, not
   just the two endpoint means) — the idea as stated here has not actually
   been tried. Worth noting the redundancy argument doesn't obviously
   transfer: a real regression slope is a different statistic from a
   difference of two existing columns, so it isn't automatically covered by
   the same "exactly zero residual correlation" argument.
2. **Distributional shape features**: skewness of the scoring-margin
   distribution; explicit clutch/close-game (≤5 pt) performance splits
   separate from blowout performance. **Not covered.** `docs/MARKET_EDGE.md`
   has a "close games" finding, but that's `market_benchmark.py` doing
   post-hoc diagnostic analysis of already-made predictions against market
   odds — not a training feature, and not skewness in any form. No existing
   script computes skewness or a clutch/blowout split as a model input.
3. **Asymmetric style-clash features**: team A's specific offensive
   strength vs. team B's specific defensive weakness, rather than the
   symmetric differential `matchup_features` already computes. **Partial
   overlap, coarser than what's proposed.** `_add_matchup_features` already
   has some directional asymmetry (`home_off_vs_away_def_L{w}`,
   `away_off_vs_home_def_L{w}`), but only at the level of single aggregate
   `off_eff`/`def_eff` scalars — it doesn't decompose which *specific*
   style dimension (pace, 3pt reliance, paint activity, assist rate, etc.,
   the 6 metrics `style_fingerprint_features` already tracks) is the
   strength/weakness being exploited. `style_matchup`'s KNN score
   (`run_style_matchup_cv.py`) is a single symmetric similarity score, not
   a directional clash either. A genuine dimension-level asymmetric
   strength-vs-weakness pairing hasn't been built or tested.
4. **Rest × elo / schedule-density × style interaction terms** — worth
   pursuing only where the interaction requires domain framing CatBoost's
   own tree splits are unlikely to find on raw columns alone. **Argued
   against, not empirically tested.** `docs/EXPLORATION.md` (Area 3,
   "Interaction features") already reasoned through this class of idea:
   CatBoost's depth-6 trees can already combine two features within one
   tree, so hand-engineering pairwise interactions like `elo×rest` or
   `injury×style` was judged speculative and deprioritized — weaker case
   than the one interaction feature that *was* confirmed to help (the rest
   differential, Area 2/A3). That's a reasoned skip, not a CV result, so it
   doesn't rule this out definitively — but any future attempt should
   engage with that argument directly (why would this specific interaction
   need domain framing a tree split can't discover on its own?) rather than
   restart from scratch.
5. **Retrospective opponent-adjustment**: a team's own rolling stats
   adjusted for the quality of opponents already faced in that window
   (distinct from `opponent_quality_features`, which is about the upcoming
   opponent). **Already tried for one construction, rejected — drop the
   "untested" framing for that variant.** `season_motivation`'s
   `opponent_adjusted_form_score` (`docs/features/season_motivation_log.md`
   §10, `configs/config.yaml`'s `opponent_adjusted_form_enabled: false`) is
   exactly this idea applied to win/loss outcomes: a rolling mean (window=10)
   of a signed, opponent-strength-weighted result. It passed an initial CV
   screen at window=10 but inverted at windows 5/15 in a robustness sweep —
   judged a favorable draw, not a robust effect, and disabled
   (`docs/BACKLOG.md`'s B-series entry). The narrower open question: nobody
   has tried the same opponent-adjustment logic on *other* rolling stats
   (e.g. opponent-quality-adjusted `off_eff`/`def_eff`, not just win/loss
   outcome) — a real gap, but one that inherits genuine skepticism from
   this result rather than starting fresh.
6. **Lineup stability/continuity as its own signal**, separate from average
   roster quality — check first whether existing player/on-off-split work
   already covers this angle. **Checked, not covered.**
   `docs/features/on_off_splits_decisions.md`/`on_off_splits_log.md`
   (`on_off_splits.enabled: false`, not adopted) is about per-player on/off
   point-differential impact for currently-missing players — "how much does
   this player's absence cost," not "how stable/continuous is the team's
   rotation." No existing feature or script measures lineup
   continuity/turnover as its own signal.
7. **Referee/officiating-crew tendencies** (foul rate, pace inflation) —
   flagged as data-feasibility-first, likely exotic/low-priority. **Not
   covered anywhere in the codebase.** No existing data source, script, or
   doc addresses officiating at all (the only text match for "referee" in
   the whole codebase is an unrelated CDN URL string in the injury-PDF
   scraper). Genuinely untested, and per the idea's own framing, feasibility
   (does usable referee-assignment/tendency data even exist pre-tip) should
   be checked before any construction work, same as Track C's other
   candidates.

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
- Every session — A1 through C, not just Track B — ends with a status log
  entry per the Status log section above, before stopping. This is what
  makes the doc trustworthy as the continuity mechanism between `/clear`
  boundaries: if a session doesn't write back what it found, the next
  fresh session has no way to know.

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