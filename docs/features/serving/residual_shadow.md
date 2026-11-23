# Corrected model next to the live picks (shadow mode)

**What**: the second-stage margin correction (`src/models/residual_corrector.py`, evidence in `docs/EXPERIMENTS.md` `residual_correction`) runs next to the live model. The slate shows its own full set of bets under the live ones; the delivered picks, edge calibration and the morning report record are unchanged.

**Flag**: `residual_correction.enabled`. Coefficients come from `scripts/fit_residual_corrector.py` (all 5 folds' validation predictions) and are saved to `data/models/residual_corrector.json`, tagged with the model's `train_date`. After a retrain the tag no longer matches, so the correction is skipped (logged) until the fit is rerun.

**Per run** (`src/serving/residual_shadow.py`):

- **Team errors**: this season's played regular-season games are re-predicted with the live model, then averaged per team over the last 10 games (0 below 3). The model was trained through 2023-24, so every such game is out of sample, as in the fit.
- **Corrected entry**: each rec gets a `residual_correction` entry in `rec_json` with the adjustment, its parts and inputs, the corrected margin, and spread/moneyline sides. The sides are priced by the same `price_margin_markets` as the live ones, against the corrected model's own 2024-25 validation residuals. Totals aren't corrected. The CSV columns are unchanged.
- **Failures**: any failure leaves the rec without the entry. The slate always goes out.

**Shown**: two complete sets of bets, in the same format.

- **HTML slate**: "Live model" (strong, then marginal), then "Corrected model (test)" (strong, then marginal). Totals are identical in both.
- **Telegram caption**: the top 3 of each.

**Review**: `scripts/compare_residual_correction.py` reports margin MAE for both, then strong picks split into both / live only / corrected only (hit rate, flat-stake return).

- **Rule**, fixed before live data: at 300 graded games, switch the picks to the corrected model only if its MAE isn't worse and the picks it changes didn't do worse than the ones they replaced.
- **What it shows**: it's a sanity check. A margin MAE gain this size (~0.04) needs more than a season of live games to confirm statistically.
