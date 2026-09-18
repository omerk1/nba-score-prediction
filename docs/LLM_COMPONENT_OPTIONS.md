# LLM / Agentic Component — Options

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
