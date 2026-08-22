# Feature Representation Audit (Track B, item 0)

Read-only inventory. No feature code modified, no training run in this session. All
claims below are from direct code reads of `src/feature_engineering/feature_builder.py`,
`src/feature_engineering/elo.py`, `src/matchups/fingerprint.py`, and
`configs/config.yaml`, cross-referenced against `docs/EXPERIMENTS.md` §2/§3 and
`docs/EXPLORATION.md`.

**Family list used**: the most recent family-importance ranking is
`docs/EXPERIMENTS.md`'s `a2_venue_blind_overall_form` entry (`run_family_importance.py`,
mean across 5 folds, run *after* the rolling-features venue-blind fix but *before* the
rest-features venue-blind fix, both of which are always-on structural changes not config
flags — no importance rerun exists post-A3 yet):

| Rank | Family | CatBoost share |
|---|---|---:|
| 1 | `style_fingerprint_features` | 29.0% |
| 2 | `style_features` | 18.9% |
| 3 | `rolling_features` | 16.1% |
| 4 | `elo_features` | 8.3% |
| 5 | `matchup_features` | 7.9% |
| 6 | `opponent_quality_features` | 7.7% |

These 6 are documented below in full. `rest_features` is also included as a 7th
section — it is **not** in the top 6 by current importance (older permutation-ranked
table has it 11th of 13, 1.4% CatBoost share), but Track B's own framing in
`docs/NEXT_PHASE_SESSIONS.md` names rest/schedule explicitly as one of the axes this
inventory must answer, so it's covered for completeness. This mismatch (named in scope,
not actually top-ranked) is called out again in the priority list at the end — it argues
for *lower*, not higher, priority.

`self.rolling_windows` (`configs/config.yaml`'s `features.rolling_windows`) = `[5, 10, 20]`
throughout — every family below that varies "per window" uses exactly these three.

---

## 1. `style_fingerprint_features` (29.0% share, rank 1)

**Output vector**: 18 columns. For each of 6 metrics (`pace_score`, `three_pt_reliance`,
`paint_activity`, `defensive_rating`, `assist_rate`, `offensive_rating`): `home_style_{metric}`,
`away_style_{metric}`, `style_{metric}_diff` (home − away). Source: `_add_style_fingerprint_features`
(`feature_builder.py:884-1028`), reading from the offline `matchup_fingerprints` cache
(`src/matchups/fingerprint.py`), joined via `merge_asof` on `(team_id, game_date)`.
5 of the 6 metrics (`pace_score`, `three_pt_reliance`, `paint_activity`,
`defensive_rating`, `assist_rate`) are injury-adjusted (layer=2); `offensive_rating` is
layer=1 (uncalibrated), a deliberate scope cut.

**Is it already multi-dimensional?** Yes, confirmed — this is not a scalar family and was
never a candidate for "add dimensions," it already has 6 independent metrics × 3 views
(home/away/diff) = 18 columns. The word "fingerprint" in the name is accurate: it's the
most structurally rich family in the inventory, not a single number.

**Aggregation method — not mean-only, already the most sophisticated in the codebase.**
`compute_rolling_fingerprints` (`fingerprint.py:113+`) uses `_decayed_weighted_mean`
(`fingerprint.py:101-110`): `weight = 0.5 ** (age / halflife)` applied to each of the
last `fingerprint_window=37` games (config), `halflife=13.199390932957819` games
(`configs/config.yaml:216-217`), `shift(1)` before the rolling window (point-in-time
safe). This is a recency-weighted mean, not a flat mean — no std/trend/other moment is
computed anywhere in `fingerprint.py`; only the single decay-weighted mean per metric
per team is produced.

**Verdict**: dimensionally rich (6 metrics), aggregation-wise already using the most
sophisticated single-statistic method in the codebase (decay-weighting vs. every other
family's flat mean). The only still-open question from `docs/EXPLORATION.md` (Area 1) is
whether `rolling_features`/`style_features` should borrow *this family's* decay-weighting
function — not that this family itself needs enrichment.

---

## 2. `style_features` (18.9% share, rank 2)

**Output vector**: 18 columns. Source: `_add_style_features` (`feature_builder.py:268-298`).
Per side (`home_team`, `away_team`) × per window (`L5`, `L10`, `L20`): `{prefix}_fg3_pct_L{w}`,
`{prefix}_off_eff_L{w}`, `{prefix}_def_eff_L{w}` — 3 sub-features × 3 windows × 2 sides = 18.

- `off_eff_L{w}` / `def_eff_L{w}`: flat rolling **mean** of `PTS_home`/`PTS_away`
  (own points / opponent points), `shift(1).rolling(w, min_periods=1).mean()`. No std,
  trend, min/max at any window length — identical construction at L5, L10, L20.
- `fg3_pct_L{w}`: **not** a mean of per-game percentages — volume-weighted
  (`sum(FG3M)/sum(FG3A)` over the window, comment at `feature_builder.py:282-283`
  explains why: avoids a low-attempt outlier game swinging the average). Still a single
  scalar per window, no dispersion statistic.

**Rolling-window aggregate**: mean-only (or volume-weighted-ratio-only for `fg3_pct`), no
variance/trend/other moments, uniformly across L5/L10/L20 — same construction pattern at
every window length, nothing window-length-dependent beyond the window size itself.

---

## 3. `rolling_features` (16.1% share, rank 3)

**Output vector**: 36 columns. Source: `_add_rolling_features` (`feature_builder.py:138-226`).
Two sub-blocks, both mean-only:

- **Venue-scoped** (24 cols): per side × window: `{prefix}_win_pct_L{w}`,
  `{prefix}_diff_avg_L{w}`, `{prefix}_fg_pct_L{w}`, `{prefix}_ft_pct_L{w}` — 4 × 3 × 2 = 24.
  `win_pct`/`diff_avg` are flat rolling means (`shift(1).rolling(w, min_periods=1).mean()`);
  `fg_pct`/`ft_pct` are volume-weighted ratios (`sum(made)/sum(att)`, same pattern as
  `style_features`' `fg3_pct`), not per-game-mean.
- **Venue-blind overall form** (12 cols, added in Track A item 2/`a2_venue_blind_overall_form`):
  per side × window: `{prefix}_win_pct_overall_L{w}`, `{prefix}_diff_avg_overall_L{w}` —
  2 × 3 × 2 = 12. Same flat-mean construction, computed via a team-perspective long frame
  (mixes home+away games) instead of the venue-scoped groupby, then merged back onto both
  the home-row and away-row perspective.

**Rolling-window aggregate**: mean-only (or volume-weighted-ratio for the shooting
splits) at every window length, both the venue-scoped and venue-blind sub-blocks. No
std/trend anywhere in this family. `pts_avg` and `FG3_PCT` are deliberately *not*
duplicated here (comment at `feature_builder.py:154,161`: identical values already exist
as `off_eff`/`fg3_pct` in `style_features`).

---

## 4. `elo_features` (8.3% share, rank 4)

**Output vector**: 3 columns only — `home_team_elo`, `away_team_elo`, `elo_diff`
(`= home_team_elo + home_advantage - away_team_elo`). Source: `_add_elo_features`
(`feature_builder.py:702-755`) + `compute_elo_ratings` (`elo.py:17-89`).

**Volatility / rate-of-change**: confirmed absent — `elo.py`'s only stored state per
team is a single running scalar `ratings[team_id]`, mutated in place each game
(`elo.py:82-83`). No rating history, no delta-from-N-games-ago, no rolling std of rating
changes, no explicit rate-of-change column is computed or exposed anywhere in `elo.py`
or `_add_elo_features`. Margin-of-victory *does* feed into the per-game rating update
(`mov_multiplier`, `elo.py:74-77`, standard 538 log-margin formula) — but that only
affects how much the persisted point-estimate rating moves that game; it does not itself
become a feature. Only the point-in-time rating (and the derived `elo_diff`) reaches the
model. This confirms `docs/NEXT_PHASE_SESSIONS.md`'s framing exactly: Elo is a running
state variable with no volatility/rate-of-change representation at all, only
point-in-time level.

---

## 5. `matchup_features` (7.9% share, rank 5)

**Output vector**: up to 15 columns (5 sub-features × 3 windows, some conditionally
skipped if the underlying source column is absent — not the case in the current
codebase). Source: `_add_matchup_features` (`feature_builder.py:405-439`). Per window:
`home_off_vs_away_def_L{w}`, `away_off_vs_home_def_L{w}`, `home_3pt_advantage_L{w}`,
`form_differential_L{w}`, `strength_differential_L{w}`.

All 5 are direct arithmetic differentials of already-existing `style_features`/
`rolling_features` columns (e.g. `strength_differential_L{w} = home_team_diff_avg_L{w} -
away_team_diff_avg_L{w}`) — no independent aggregation logic of its own, purely
derived. Since the underlying source columns are themselves mean-only (families 2/3
above), these differentials are differences-of-means, not a distinct representation
question — enriching this family only makes sense downstream of enriching `style_features`/
`rolling_features` first.

---

## 6. `opponent_quality_features` (7.7% share, rank 6)

**Output vector**: 12 columns. Source: `_add_opponent_quality_features`
(`feature_builder.py:300-345`). Per side × window: `{prefix}_opp_def_quality_L{w}`,
`{prefix}_opp_off_quality_L{w}` — 2 × 3 × 2 = 12.

Construction: re-averages the **same** `off_eff_L{w}`/`def_eff_L{w}` columns already
live in `style_features`, re-keyed over "opponents faced" via a team-perspective long
frame, `shift(1).rolling(w, min_periods=1).mean()` — flat mean, same as `style_features`.
`docs/EXPERIMENTS.md`'s §2 finding: this family is structurally capped by schedule
balance (opponent-quality variance is inherently smaller than own-quality variance in a
balanced 82-game schedule), not a construction defect — this is a second-order
aggregation of a primitive the model already sees directly.

---

## 7. `rest_features` (not top-6 by importance — 11th of 13, 1.4% share on the
last-measured ranking; included because Track B's framing names rest/schedule explicitly)

**Output vector**: 6 columns. Source: `_add_rest_features` (`feature_builder.py:228-266`,
rewritten venue-blind by Track A item 3/`a3_rest_venue_blind_fix`). Per side:
`{prefix}_rest_days`, `{prefix}_back_to_back`, `{prefix}_games_in_4_nights`.

- `rest_days`: days since the team's previous game (venue-blind, `GAME_DATE.diff()` on a
  team-perspective long frame), `.fillna(3)` for a team's first-ever game.
- `back_to_back`: `rest_days == 1`. Per A3's own investigation, this is now — by
  construction, once `rest_days` is computed venue-blind — exactly "second game of a
  back-to-back"; a separate `second_of_b2b` column would be a pure duplicate (A3 verified
  this and did not add one).
- `games_in_4_nights`: rolling 3-game date-gap sum ≤ 4, a single binary density flag
  (`feature_builder.py:251-253`). This is the only "window" concept in the family, and
  it's a fixed 3-game/4-night rule, not parameterized by `self.rolling_windows` at all —
  no L5/L10/L20 variants exist for rest.

**What's covered vs. what a schedule-density vector would add**: `docs/EXPLORATION.md`'s
Area 2 (measured, not just argued) already tested the natural next step here — a
home/away rest **differential** (`b2b_diff`, `rest_diff` clipped ±5 days) — and found it
real (`corr(b2b_diff, POINT_DIFF) = -0.079`, `corr(rest_diff, POINT_DIFF) = +0.050`,
2-3× the magnitude of the std/trend null result), but that differential has **not yet
been implemented** — `_add_rest_features`'s only differential-shaped feature is the
single fixed `games_in_4_nights` flag; no home-minus-away `rest_diff`/`b2b_diff` exists
in the codebase today (confirmed by reading the full method body above — only the 3
per-side columns listed exist, `matchup_features` does not include a rest differential
either). A genuine schedule-*density* vector (a second window, e.g. `games_in_6_nights`,
beyond the one existing `games_in_4_nights` flag) is a distinct, still-untested idea
flagged in `docs/EXPLORATION.md` ("Multi-game fatigue density windows... a plausible
future add but has zero measurement behind it") — separate from the differential, which
does have measurement behind it.

---

## Prioritized list — families worth a B-series enrichment session

Ranked by (current importance) × (scalar-only where a richer representation is
plausible, per `docs/EXPLORATION.md`'s Area 1 measurement) — i.e. where a representation
fix is most likely to move CV/market_benchmark:

1. **`style_features` (18.9% share, rank 2) — decay-weighting swap.** Mean-only at every
   window (L5/L10/L20), highest-importance family that is genuinely still scalar/flat-mean.
   `docs/EXPLORATION.md`'s std/trend test found *adding* dispersion/trend columns doesn't
   help (measured, R² deltas <0.15%), but explicitly flagged the *unexplored* alternative:
   swapping the flat mean for `fingerprint.py`'s already-proven `_decayed_weighted_mean`
   (zero new columns, pure replace, motivated by precedent from family 1 above, where
   decay-weighting demonstrably earns its place). Highest-priority candidate: most
   importance among genuinely-untested-this-way families, cheapest possible test (no new
   columns, reuse of existing tuned infrastructure).

2. **`rolling_features` (16.1% share, rank 3) — same decay-weighting swap.** Same
   argument as `style_features` — mean-only (and volume-weighted-ratio-only) throughout,
   no dispersion/trend, and the exact same unexplored decay-weighting idea applies
   (`docs/EXPLORATION.md` names `rolling_features`/`style_features` together as the pair
   this applies to). Second priority mainly because it's third by importance, not because
   the representation question differs at all from `style_features`' case — these two
   are naturally one enrichment session, not two, given they're the identical fix.

3. **`elo_features` (8.3% share, rank 4) — volatility/rate-of-change.** Confirmed
   scalar (point-in-time rating only, no history/volatility/momentum feature exists, §4
   above). `docs/EXPERIMENTS.md`'s E1 already tested and rejected re-tuning
   `season_regression` (flat 1.3841-1.3852 range, fold-2-5 guardrail failure) — but
   that's a different lever (how fast the *existing* scalar mean-reverts) from *adding a
   volatility/rate-of-change feature*, which has never been tested. Third priority on two
   grounds, neither depending on any permutation-importance number: (a) by the same
   post-A2 CatBoost-share metric used to rank every other family in this list — the one
   importance metric this doc uses consistently throughout — elo is 4th at 8.3%, the
   highest share among the families not already covered by the items 1-2 swap, so it's
   next in line on importance alone; (b) unlike the style/rolling decay-weighting swap
   (reusing `fingerprint.py`'s already-tuned `_decayed_weighted_mean`, zero new columns),
   a volatility/rate-of-change feature for elo has no existing function to reuse — window
   length, which statistic (std of rating deltas? a days-weighted slope?), and
   leakage-safety all need designing from scratch, a materially higher-cost,
   lower-confidence test than items 1-2. Real, consistently-measured importance plus a
   confirmed, code-verified representation gap, weighed against genuinely higher build
   cost than the swap, is sufficient on its own to place elo at priority 3 — above the two
   deprioritized families below (8.3% exceeds both their shares) and below the cheap
   sure-thing swap. This argument holds without invoking permutation cost at all; no
   permutation number is cited here.

4. **`opponent_quality_features` (7.7% share, rank 6) — deprioritize.**
   Mean-only, but `docs/EXPERIMENTS.md` §2 already gives a structural reason enrichment
   won't help here specifically: the achievable signal is capped by schedule balance
   (opponent-quality variance is inherently small in a balanced 82-game schedule), not by
   representation. Enriching a schedule-balance-capped signal is lower-expected-value than
   items 1-3 even though its importance is comparable to `matchup_features`.
   Checked the same staleness question raised for elo: the post-A2
   (`a2_venue_blind_overall_form`) family-importance run reports fresh CatBoost-share
   numbers for all 6 top families, but only reports an updated permutation-importance
   delta for `rolling_features` itself (+0.0137 — the family A2's fix targeted); no
   post-A2 permutation number was ever logged for `opponent_quality_features` to compare
   against its pre-Track-A value (0.0022, rank 6, `docs/EXPERIMENTS.md` §2's original
   table), so there's nothing to check for movement — the data doesn't exist either way.
   This doesn't matter for the ranking above regardless: unlike elo's now-removed
   citation, this family's deprioritization was never built on a permutation number in
   the first place — it rests entirely on the schedule-balance construction argument,
   which doesn't depend on any importance metric, old or new.

5. **`matchup_features` (7.9% share, rank 5) — deprioritize, dependent on 1/2.**
   Purely derived (differentials of `style_features`/`rolling_features` columns) — has no
   independent representation to enrich. If items 1/2 (the decay-weighting swap) are
   adopted, `matchup_features`' differentials would need re-deriving from the new
   columns as a mechanical follow-on, not a separate research question. Not a standalone
   B-series session target.

6. **`style_fingerprint_features` (29.0% share, rank 1) — no enrichment session
   warranted.** Despite being the single highest-importance family, it's already
   6-dimensional and already decay-weighted (the most sophisticated representation in the
   codebase) — there's no "scalar-only where richer representation is possible" gap here
   to close. Explicitly excluded from the B-series priority list on the evidence, not by
   omission.

7. **`rest_features` (1.4% share, not top-6) — lowest priority, but has the one
   concretely un-implemented, already-measured fix in this whole inventory.** Its
   importance is too low to justify a broad enrichment session on priority grounds alone.
   But unlike every family above, `docs/EXPLORATION.md`'s rest/back-to-back differential
   (`b2b_diff`, `rest_diff`) is not a speculative representation question — it's a
   **specific, already-measured, sign-sensible, unimplemented feature**
   (`docs/EXPERIMENTS.md`'s own decisive shortlist ranks it #1 by evidence quality across
   the whole memo). If a B-series session is being picked by "cheapest, most
   evidence-backed single test" rather than "highest current importance," this is the
   strongest candidate in the entire inventory — flagging the tension explicitly since the
   two ranking criteria (importance vs. evidence quality) disagree here.

**Recommended B1 order given both criteria**: `style_features` + `rolling_features`
decay-weighting swap (items 1-2, bundled as one session per Track B's framing pairing
them) first — highest importance among genuinely enrichable families, cheapest possible
test. The rest differential (item 7) is a strong second pick on evidence-quality grounds
even though it's not top-6 by importance — worth flagging to whoever sequences B1+ as a
candidate for an early slot despite its low rank, precisely because `docs/EXPERIMENTS.md`'s
own decisive shortlist already ranked it #1 by evidence quality across the entire
exploration memo, and it wasn't in scope for Track A item 3 (which fixed the underlying
venue-blind bug but didn't add the differential itself, per A3's own log: "no new column
added — fixed the existing computation instead," referring to `second_of_b2b`
specifically, not the separate `b2b_diff`/`rest_diff` differential idea).

---

## Track B rollup — pre-Track-B baseline vs. fully-settled state

All three B0-priority families have now had their enrichment session (`docs/EXPERIMENTS.md`'s
`b1_style_features_decay_weighted` / `b1_style_and_rolling_decay_weighted` /
`b2_elo_momentum` / `b2_elo_momentum_and_volatility` entries). This section is a rollup
only — numbers pulled from those already-logged rows, nothing re-run or re-derived.

**Before (`a3_rest_venue_blind_fix`, pre-Track-B, 2026-08-19):** 139 features. Full CV
val_score_mean **1.3811** (per-fold: 1.4345, 1.3883, 1.3720, 1.3522, 1.3586).
`market_benchmark` (fold5): diff_mae 11.412, total_mae 15.366, win_acc 0.6738,
brier 0.20797.

**After (`b2_elo_momentum`, current live state, 2026-08-21):** 148 features (+9, all
`elo_momentum_L{5,10,20}` — the only surviving B-series addition). Full CV
val_score_mean **1.3803** (per-fold: 1.4336, 1.3888, 1.3708, 1.3479, 1.3605).
`market_benchmark` (fold5): diff_mae 11.430, total_mae 15.357, win_acc 0.6795,
brier 0.20779.

**Cumulative delta (after − before):** val_score_mean **−0.0008** (improvement).
Per-fold: fold1 −0.0009, fold2 +0.0005, fold3 −0.0012, fold4 −0.0043, fold5 +0.0019 — 3
of 5 folds improve, the other 2 regress only slightly (largest +0.0019, inside this
project's established noise floor). `market_benchmark`: diff_mae +0.018 (worse),
total_mae −0.009 (better), win_acc +0.0057 (better), brier −0.0002 (flat/better) — 3 of
4 metrics move the right way.

**Net yield of the three B0-priority families, stated plainly:**

| Family | Candidate(s) tested | Outcome |
|---|---|---|
| `style_features` | decay-weighted `off_eff`/`def_eff` | Rejected — null (mixed CV, mixed benchmark) |
| `rolling_features` | decay-weighted `win_pct`/`diff_avg` (venue-scoped + venue-blind) | Rejected — real, small, consistent regression |
| `elo_features` | rate-of-change (`elo_momentum`) | **Adopted** — real, small, consistent improvement |
| `elo_features` | volatility (`elo_volatility`, rolling std of rating deltas) | Rejected — real, small, consistent regression |

One of three targeted families (elo) yielded an adopted feature; the other two
(style_features, rolling_features) were tested and closed off with a confirmed null and
a confirmed regression respectively — real information (a specific, now-documented
reason not to revisit either representation question without new evidence), but zero
net addition to the feature set. Of elo's own two candidates, one of two was adopted.

This is a **small, real, but genuinely modest result** — not a breakthrough. The
cumulative CV movement (Δ−0.0008) is on the same order of magnitude as this project's
established noise floor (cf. A1's own +0.0003 jitter, attributed to incremental raw-data
refresh, not a real change) and smaller than the clearer wins earlier in this project's
history (`target_lambda_weight_0.75`'s Δ−0.0017 on a 4/5-fold margin, the `diff_total`
reformulation's unanimous 5/5-fold Δ−0.0013). It clears this project's adoption bar
(majority-of-folds improvement, no catastrophic single-fold regression, plausible
mechanism — low correlation with the existing `elo_diff`) and is worth keeping, but two
of the three families this track targeted produced no feature-set change at all, and the
one that did was a genuinely marginal win, not a decisive one. Stated for the record so
a future session doesn't read "Track B shipped a feature" as "Track B was a clear
success" — both are true, but the second overstates the first.
