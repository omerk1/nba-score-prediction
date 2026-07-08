# Feature EDA Insights

Concise findings from `notebooks/02_feature_eda.ipynb` — a full-feature-set EDA
(127 cols: 107 default-enabled + `style_matchup.enabled`'s 2 KNN-lookup cols +
`raw_features_enabled`'s 18 raw-fingerprint cols, both flags locally overridden to
`true` for this analysis only, `configs/config.yaml` unchanged). General feature
analysis, not A7-specific — see `docs/a7_phase_log.md`'s Feature EDA section for
the pointer.

---

## 1. The pace_score / elo_diff question (the raw-fingerprint feature redesign's motivating finding)

**`pace_score` is not a redundant proxy for Elo — it's capturing something Elo
structurally cannot.**

- `home_style_pace_score` / `away_style_pace_score` correlate ~0 with `elo_diff`
  (-0.06 / +0.05) and ~0 with `POINT_DIFF` (-0.04 / +0.01) and `HOME_TEAM_WINS`
  (-0.04 / -0.00).
- Both correlate **0.37-0.38 with `TOTAL_POINTS`** — `pace_score`
  (`PTS + OPP_PTS + TOV - FTA*0.44`) is fundamentally a combined-scoring-level
  proxy; `elo_diff` (built from margin-of-victory, never point totals) correlates
  -0.003 with `TOTAL_POINTS` — it has no mechanism to see this at all.
- This directly explains the raw-fingerprint feature redesign's finding:
  `pace_score` ranks #1/#2 in CatBoost importance and drives the `total_mae` gain,
  but doesn't move `win_acc`/spread — it's adding genuinely new information, just
  information that's only useful for the over/under market.
- **Contrast:** `style_offensive_rating_diff` (also top-15 in that redesign, rank #14)
  correlates **0.75 with `elo_diff`** and 0.31 with `POINT_DIFF` — this one *is*
  substantially redundant with what Elo already encodes. Two features that both
  got real importance arrived there by opposite mechanisms.

## 2. Bug found: `h2h_win_pct_3yr` is ~99.7% NaN (not a coverage gap)

- Root cause: `FeatureBuilder._compute_h2h_3year_win_pct` builds
  `matchup_sorted = matchup_games.sort_values('GAME_DATE').reset_index(drop=True)`
  (fresh `0..n-1` index) then returns
  `pd.Series(result, index=matchup_sorted.index).reindex(orig_index)` —
  `orig_index` is the *original*, non-contiguous full-dataframe index, not
  `0..n-1`. The reindex looks up labels that mostly don't exist in a `0..n-1`
  index, so it comes back NaN almost everywhere. Only 36 of 12,717 rows survive
  (coincidental small-index overlaps) — confirmed by direct reproduction in the
  notebook (Section 3).
- **Consequence:** `h2h_win_pct_3yr` shows up in the feature-label correlation
  top-15 for `POINT_DIFF` (+0.271) and `HOME_TEAM_WINS` (+0.250) — both numbers
  are noise from ~36 rows, not signal. Ignore them.
- Out of scope to fix this round (`feature_builder.py` off-limits) — flagging for
  whoever picks up `_add_h2h_features` next. Not currently gated by any config
  flag, so this bug is live in the *default* 107-feature set today, independent
  of A7.

## 3. `style_matchup_confidence` is effectively constant

- 99.3% of games land on the exact same value (~0.9878, the
  `full_confidence_sample=82` ceiling). With 12.7k+ games of corpus depth, almost
  every KNN search hits full confidence.
- This is a structural explanation for the KNN-score integration test's finding
  that `style_matchup_confidence` got **zero** CatBoost importance — not that the
  model ignored a useful signal, the column itself carries almost no
  distinguishing information. Reinforces that test's own recommendation to drop
  it if `style_matchup_score` were ever adopted.

## 4. Redundant / highly-correlated features (candidates to consider dropping)

- `home_team_win_pct_L20` / `home_team_diff_avg_L20`: r = 0.915 (both summarize
  the same last-20-games outcomes — win-rate vs. margin-average).
- `form_differential_L20` / `strength_differential_L20`: r = 0.902 (same
  construction pattern, `_add_matchup_features`, one window).
- L5-vs-L20 same-stat rolling windows sit around r = 0.6-0.75, L10-vs-L20 around
  0.85-0.89 — real redundancy but not extreme; each window still resolves
  genuine distinguishing information (recent-5 form diverges from full-20 more
  than noise alone would predict). **No single window looks safely droppable on
  correlation grounds alone.**
- `home_style_three_pt_reliance` and `away_style_defensive_rating` had exactly
  zero CatBoost importance in the raw-fingerprint feature redesign and show no
  standout linear correlation with any label here either — consistent candidates
  to drop if the raw fingerprint feature set is ever adopted (already flagged in
  that redesign).
- Travel/rest features (`travel_miles`, `tz_shift`, `rest_days`,
  `back_to_back`) correlate ~0 (|r| < 0.04) with all three labels. Doesn't mean
  they're useless (could matter via non-linear interactions CatBoost can find),
  but they don't carry meaningful *linear* signal on their own.

## 5. Preprocessing / transforms

- Several features are heavily skewed (`n_questionable` skew ~30-46,
  `games_in_4_nights` ~20-22, `rest_days` ~6.6, `travel_miles` zero-inflated at
  ~55% exactly 0) — but **CatBoost is tree-based and invariant to monotonic
  transforms**, so log-transforming or otherwise reshaping these would not be
  expected to help this model. Flagging for completeness, not recommending
  action — this matters more for a linear/distance-based model than for
  CatBoost.
- The one outlier worth a second look elsewhere (`home_team_n_out = 21`, game
  `0022100433`, 2021-12-17) is a real value — the December 2021 COVID
  postponement wave — not a data error. No cleanup needed.

## 6. Era-based coverage gaps — confirmed, no surprises

- `has_injury_data` / `team_deficit` nonzero-rate: 0% every year before 2021,
  32% in 2021 (matches `pdf_era_start=2021-10-01` mid-season), 93-94% from 2022
  on. Exactly as documented elsewhere in this project.
- No comparable era gap for A7 style-fingerprint features — Layer 1/2 only need
  box-score history, available back to 2016 for every season in this dataset.

## 7. Era drift — visible directly in the features

- League scoring level (`home_team_off_eff_L20`, rolling points-scored proxy):
  ~105 (2016) → ~116 (2025-26), ~10% rise.
- True 3PT attempt rate (`home_style_three_pt_reliance`, 3PA/FGA): 0.31 → ~0.42,
  +35% relative — the well-documented pace-and-space shift, visible directly.
- **Naming trap:** `home_team_3pt_rate_L{window}` is actually rolling `FG3_PCT`
  (make-rate, not attempt rate) and stays flat ~0.35-0.37 across all eras — a
  different quantity from `home_style_three_pt_reliance` despite the similar
  name. Worth disambiguating in any future discussion of "the 3-point era" using
  this feature set.
- Home-court advantage (`HOME_TEAM_WINS` rate by year) shows no clear secular
  trend, staying in a 0.53-0.58 band — a stable structural effect, not
  era-drifting.

## 8. Anything else

- Full feature-family breakdown (127 cols): rolling per-team 24, style-rolling
  18, opponent-quality 12, matchup differentials 15, travel 8, injury 8, H2H 6,
  rest/schedule 6, venue-delta 6, A7 raw-fingerprint 18, elo 3, A7 KNN-lookup 2,
  basic 1 — see notebook Section 1 for the full mapping to `feature_builder.py`
  methods.
- No feature (of the 127) is entirely NaN or zero-variance; only 5 are
  effectively binary (`*_back_to_back`, `*_games_in_4_nights`,
  `has_injury_data`) — all expected.
