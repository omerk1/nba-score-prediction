# Semantic Retrieval over Injury Reason Text

Status: built and tested 2026-09-22. **Rejected on the tuning split; the sealed
season was never touched.** Both the cross-player and within-player variants
failed, each for a different and diagnosable reason, recorded below because the
reasons are more useful than the result.

## Hypothesis

The strongest single fact in the availability context is how often a player
played when previously listed with the *same* reason, and it is absent on 51%
of rows because exact string matching is sparse. There are ~1,400 distinct
normalized reason strings, many of them near-duplicates ("Right Ankle; Sprain"
versus "Left Ankle; Sprain"). Embedding the vocabulary and retrieving similar
reasons should densify the feature and improve the model that already wins.

## Protocol, fixed before any knob was turned

- Tuning seasons 2022-23 through 2024-25. Every parameter chosen there.
- 2025-26 sealed, and also after the embedding model's cutoff.
- Within either split, a season is predicted by a model fit only on earlier ones.
- The test is not whether the retrieved value predicts on its own. It is the
  Brier delta of CatBoost with the new columns against the identical CatBoost
  without them, on the same rows.

Embeddings behave as intended: left versus right ankle sprain scores 0.969,
ankle versus knee 0.811, injury versus personal reasons 0.59.

## Result 1: cross-player retrieval, rejected

54 settings swept over neighbour count, similarity floor, weight exponent and
shrinkage. **Every one was worse than no retrieval.** Best delta +0.00105,
paired-bootstrap CI [-0.00104, +0.00316], so strictly "no significant effect"
rather than harmful. Coverage was healthy at 78-82%, so this is not a plumbing
failure.

**Why.** On the rows the feature was built to rescue, those with no exact
same-reason history, the retrieved rate correlates **+0.009** with the outcome,
against +0.127 for the sparse feature it was meant to densify. The predictive
content of "same reason" is about the *player* -- their role, their pain
tolerance, how cautious their team is -- not about the injury words. Averaging
across other players discards precisely that and returns the league base rate,
which is why the feature's standard deviation is only 0.038.

A by-product, `reason_nbr_out_share`, is an empirical severity measure built
from how often a similar reason appears as Out. It correlates 0.475 with the
hand-written keyword buckets in `classify_severity` and ranks 9th of 30 in
importance, so it reproduces the existing heuristic rather than improving on it.

## Result 2: within-player retrieval, rejected

The obvious follow-up from that diagnosis: match semantically but stay inside
one player's history. It does densify, coverage rising from 48.7% to 66.4% at a
0.80 floor, and its overall correlation with the outcome (+0.122) is close to
the exact-match feature's (+0.127). It still does not help: best delta
-0.00058 at n=5,037, indistinguishable from noise.

**Why.** It correlates **0.755** with `own_prior_play_rate`, the player's
overall rate across all prior uncertain listings, which is already a feature.
And it predicts the outcome *worse* than that existing feature does, 0.122
against 0.141. Most of a player's listings concern the same recurring problem,
so restricting to semantically similar ones barely changes the set and mostly
adds noise.

## What this closes

The gap is not in the reason vocabulary. Rows missing the exact-match feature
have a median of 7 prior uncertain listings, so history is not scarce either.
The two candidate widenings are a rate over the wrong population (other
players) and a noisier copy of a rate already present (the same player). There
is no third route through the reason text that the tabular features do not
already cover.

## What was kept

`src/availability/reason_retrieval.py` (embedder with an on-disk cache, index,
both feature builders), `scripts/tune_reason_retrieval.py` (sweep and sealed
test), and `tests/test_reason_retrieval.py`, whose point is the point-in-time
bound: a neighbour contributes only if its report predates the query's. The
sealed 2025-26 season remains unused and available, which is the correct
outcome of a tuning split that produced no candidate.
