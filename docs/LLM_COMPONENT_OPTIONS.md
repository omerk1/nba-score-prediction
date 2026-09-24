# LLM / Agentic Component — Options

**Investigation CLOSED 2026-09-23.** Full closing summary at the bottom of this
file. Bottom line: six independent hypotheses tested over the following week,
all rejected, for one consistent structural reason (see closing section). Not
a good direction for building an LLM workflow on this codebase; two real,
unrelated correctness bugs were found and fixed along the way.

Scoping note (2026-09-17). Goal: add one LLM-based component to the project mainly
for engineering experience (pipeline, evaluation, retrieval), with a real but
modest chance of helping the model. Candidates below are ordered by fit.
The chosen one is scoped in `docs/features/availability_agent_scope.md`.

## Cross-cutting risk: LLM memorization

Every option that asks an LLM about a historical game risks the LLM recalling the
outcome from its training data. All CV validation folds fall inside typical
training cutoffs. This is a second, separate leakage channel on top of the
point-in-time rule for features: the prompt can be perfectly pre-game and the
answer can still be contaminated. Required controls for any option:

- Anonymize prompts (no player names, team names, or dates; numeric history and
  injury text only).
- Report a post-cutoff evaluation slice (games after the chosen model's training
  cutoff) separately from the full history.
- Memorization check: run the same task named vs. anonymized on pre-cutoff
  seasons. If names help materially, only anonymized results count.

## 1. Availability agent (chosen)

Per uncertain injury listing (Questionable/Doubtful), estimate P(plays) and expected
minutes share from retrieved pre-game context (player's injury history, minutes
trend, importance, rest, standing). Ground truth is the box score. Replaces the
fixed `doubtful_weight` with a per-case estimate inside `injury_features`.

- Pros: own labels independent of the score model, honest tabular baselines on
  the same retrieved facts, retrieval over existing sqlite data, clean downstream
  ablation through the existing CV workflow.
- Cons: label backfill needed (per-game player minutes are not stored); injury
  data only covers 2021-22 onward, so the effect is confined to folds 3-5.

## 2. Agentic residual analyst (dev tool)

After a CV run, an agent pulls the largest-residual games, gathers injuries, odds,
and feature values via tool calls, and drafts hypotheses into `docs/EXPERIMENTS.md`.

- Pros: real tool-use loop, no memorization concern (it reads our own data), no
  leakage surface since it never touches predictions.
- Cons: evaluation is soft (did its hypotheses lead to adopted features?); it is
  tooling, not a model component.

## 3. LLM judge on model-vs-market disagreements

When the model and Polymarket disagree by more than a threshold, an agent gathers
context and outputs "trust model / trust market". Label: which side was closer.

- Pros: clean binary label, plugs into `scripts/market_benchmark.py`.
- Cons: `docs/MARKET_EDGE.md` shows the model is wrong on most large
  disagreements (41% vs. 59%), so the ceiling is low; heaviest memorization risk
  of all options because the question is literally the game result.

## 4. News retrieval for pre-game context

Retrieve beat-reporter and injury news per game, extract structured signals (star
returning, load management, coaching change, trade), feed as features.

- Pros: the most textbook retrieval-augmented setup.
- Cons: needs a timestamped point-in-time news archive for backfill (articles
  written after tip-off are leakage); expensive to collect and hard to audit.

## 5. LLM injury-report extractor (small, adjacent to option 1)

`docs/MARKET_EDGE.md` (2026-08-17) records two parser bugs in the PDF path: listed
"Out" players silently dropped, and next-day entries filed under the wrong date.
An LLM extractor with a strict schema, evaluated against the current parser on a
labeled sample, fixes a known bug and gives an extraction-accuracy eval as a
warm-up for option 1.

---

## Outcome (2026-09-18)

Option 1 was built and its LLM component **rejected** after three attempts
(bare prompt, isotonic-calibrated, and few-shot with a reasoning budget). Full
numbers and reasoning: `docs/features/availability_agent_scope.md`. The
non-LLM half of the work survives: box-score availability labels, a
point-in-time retrieval layer, and a tabular estimator that does beat the
status prior, which carries into the feature ablation.

Transferable lesson for the remaining options: on a task with a few thousand
labeled rows where the inputs are already numeric, the LLM has no room to win.
Options 2 and 4 are the ones where an LLM reads something no tabular model can
(free text, or its own tool results), so they remain the better candidates if
this is picked up again.

---

## Bounded score-adjustment test (2026-09-20)

Direct test of "give the engineered features to a language model and let it
reason about the score", on the real target rather than by analogy. Post-hoc and
read-only (`scripts/run_score_adjustment_test.py`): it consumes `run_split`'s
fold5 held-out predictions and never touches training, features, folds or the
metric. Per game it sends the model's predicted differential plus 15 pre-game
facts and asks for an adjustment in [-4, +4] points. Team names and dates are
included, since fold5's test window is entirely after the model's training
cutoff, which removes the recall path while giving it real basketball knowledge.
1,225 games, zero failed calls.

| quantity | value |
|---|---:|
| differential MAE, model alone | 11.4378 |
| differential MAE, model plus adjustment | 11.4442 |
| difference (95% CI, paired bootstrap) | +0.0064 (-0.036, +0.049) |
| correlation of adjustment with the model's residual | 0.003 |
| win accuracy, before and after | 0.680 / 0.680 |

**Verdict: no significant effect. The adjustments are noise.** This is a cleaner
negative than the availability result, and a different one: there the model was
decisively worse, here it contributes exactly nothing. Correlation with the
residual is 0.003, and correlation of adjustment size with error size is 0.04.

**It followed instructions well, which makes the null more meaningful.** Told
the prior on any adjustment was zero, it left 75% of games untouched and
averaged 0.37 points of adjustment overall. This is not a model flailing.

**Its theory was consistent, confident, and worthless.** Nearly every non-zero
adjustment cited players ruled out, with mean stated confidence 0.72, on the
argument that the model underweights injury counts. Worth exactly zero.

**One caveat that matters, and it points at the other piece of work.** The
injury counts it leaned on are attached to the wrong day (`docs/PIPELINE_AUDIT.md`,
2026-09-17). The one signal it chose is the one currently corrupted, so this
test cannot fully close the hypothesis until
`docs/features/injury_pdf_extraction_scope.md`'s phase A lands. Re-running this
afterwards is cheap: the harness exists and responses are cached.

---

## Closing summary (2026-09-23)

### The full record

| # | Attempt | Mechanism tested | Result |
|---|---|---|---|
| 1 | LLM availability estimator, bare prompt | Same numeric facts as a tabular model | Brier 0.2472 vs. tabular's 0.2150, vs. 0.2250 for the naive status-rate prior. Loses to a model with no facts at all. |
| 2 | + isotonic calibration / stacked with the tabular model | Same, post-processed | Narrows the gap, doesn't close it; adds nothing when combined with the tabular model (Δ not significant). |
| 3 | + 16 few-shot examples, reasoning enabled | Same facts, more technique | Brier 0.2942 — *worse* than the bare prompt. AUC fell below the status prior. |
| 4 | Reasoning alone, isolated from the examples | Closes the confound in #3 | Brier 0.2597 — still worse than bare, confirming reasoning itself is the problem, not just the examples. |
| 5 | Semantic retrieval over injury-reason text (two variants) | Real RAG: embeddings, similarity search, 54-setting sweep, tuning/sealed split | Every setting worse than no retrieval; the diagnosis showed the signal in "similar reason" is about the *player*, not the injury wording, so averaging across players erases exactly what mattered. |
| 6 | LLM adjustment on top of the trained model's own prediction | Same engineered features, asked to reason about the final score | No effect: Δ+0.0064 MAE, 95% CI crossing zero, adjustment-to-residual correlation 0.003. |
| 7 | Corrected injury dates wired into the live feature | Not an LLM test — the data-correctness fix's actual model impact | Flat to very slightly worse on a 3-fold screen; did not clear the guardrail. |

Every LLM attempt (1–6) failed in the same direction for the same reason:
**the ceiling was the information in the facts, not the reading of them.**
Attempts 3 and 4 make this concrete — brier got *monotonically worse* as more
prompting sophistication was added (bare < reasoning alone < reasoning with
examples), the opposite of what more effort should produce if the underlying
signal were extractable by a better read of the same numbers.

### What was genuinely real, independent of the negative results

- Two correctness bugs fixed in the live injury-report scraper: every
  Clippers listing silently dropped since 2021 (a team-name spelling
  mismatch), and most listings attached to the wrong day's game (the parser
  never read the PDF's own game-date column). Neither depends on any LLM
  question being right — `docs/features/injury_pdf_extraction_scope.md`.
- A survivorship bias fixed in the availability labels themselves (player
  name resolution scoped to one team's roster silently dropped anyone who
  missed a full season, exactly the population being estimated).
- Two real infrastructure bugs in the LLM harness itself, caught and fixed
  because attempt 4 was pushed to genuine isolation rather than accepted at
  face value: a cache-key collision that let one run silently replay another
  run's answers, and a sqlite locking bug that crashed under concurrent
  access. Both are now regression-tested.

### The lesson, stated once, plainly

This codebase is a well-posed tabular regression problem with years of dense,
point-in-time-safe training data. A tuned gradient booster already extracts
what's extractable from that data. An LLM reading the same numeric facts, or
retrieving similar historical text, or adjusting the booster's own output, has
nothing to add in that setting — not because the effort was shallow, but
because the task structurally favors the tool already in use. The one
exception found (`option 5` above, the PDF extractor) is real: unstructured
input with no tabular competitor. It stayed unbuilt because the corrected-date
ablation it would have justified didn't clear its own screen — see
`docs/features/injury_pdf_extraction_scope.md`'s phase D.

**Recommendation**: do not add another LLM component to this codebase's
prediction pipeline without a genuinely new mechanism (unstructured input this
project doesn't already parse, or a decision no tabular model can encode) --
re-running variations of "read engineered features, output a number" is now a
closed question, not an open one. For the hands-on RAG/pipeline/agentic
experience that motivated this whole investigation, pick a task where the
input is actually unstructured or the workflow requires live tool use, not a
structured-data problem that already has the right tool.

---

## Future direction: a live pre-game news and market-reaction agent

One idea survives the closing recommendation above because it has a mechanism
none of the six rejected attempts had: live, unstructured input with no
historical training corpus, and a flag/alert action rather than a number fed
to the score model. Scoped in `docs/features/news_market_agent/scope.md`.

**What it does.** Watches for breaking pre-game information in the hours
before tip-off — injury updates, lineup announcements, beat-reporter posts —
and judges whether Polymarket's current price already reflects it. When it
judges a real, unpriced gap, it flags the game with its reasoning.

**Why the mechanism actually fits, unlike everything just closed out.**
Option 4 above (news as a training feature) was rejected because building a
supervised feature needs a timestamped, point-in-time news archive spanning
years — expensive, and the retroactive labeling is itself hard to get right.
An agent that acts live, today, needs none of that backfill. It only has to
be good in the moment it runs. Same data source, a use case the earlier
rejection doesn't touch.

The input is genuinely unstructured (a tweet, a vague coach's quote, a
lineup graphic) and the judgment — is this credible, is it already priced
in — is not something the existing structured injury-PDF pipeline can
express as a feature at all.

**Shape.** A monitoring loop (poll a small set of sources for each of
today's games) → retrieval of that game's recent context (this repo's own
injury history, past price moves, this team's recent news) as grounding →
an LLM judgment call with tools (fetch the current Polymarket price, fetch
the source article) → a flag with stated reasoning and confidence, not an
autonomous trade.

**Evaluation, the part that keeps it honest.** Track flagged games forward:
did the price move in the flagged direction afterward, and by how much
relative to games it didn't flag. That is a real, checkable question,
independent of whether the underlying prediction model ever uses the
output. Start with human-reviewed flags before any auto-action.

**Honest caveats.** Source access (X/Twitter API, beat-reporter feeds) is
the main friction, not the agent logic. Small sample size per season limits
how confidently the evaluation can speak. This is a monitoring/alerting
tool, not a new model feature — its value is a faster or more careful read
of public information, not exclusive information, matching what
`docs/MARKET_EDGE.md` already found about this specific market.
