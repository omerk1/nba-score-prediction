"""
Full per-game predictive distribution around ScorePredictor's existing
diff/total point predictions, built with a Gaussian-kernel density estimate
over validation-fold residuals -- the same homoscedastic (same shape for
every game) assumption `conformal.py`'s constant-width intervals already use
and validated in `prediction_interval_conformal_v1`
(docs/EXPERIMENTS.md), just carried to a full distribution instead of one
alpha-level interval. `heteroscedastic_interval_screen`
(docs/EXPERIMENTS.md) tried two independent methods to make the width vary
per game and found ~zero recoverable signal in the current feature set, so
that isn't attempted here -- every game gets the same residual shape,
shifted to its own point prediction.

Purely additive/diagnostic, same as conformal.py: never touches training or
the point predictions themselves, judged on calibration (PIT/coverage), not
val_score_mean.
"""

import numpy as np
from scipy.stats import norm


def silverman_bandwidth(residuals: np.ndarray) -> float:
    """Rule-of-thumb Gaussian KDE bandwidth (Silverman 1986), robust to
    outliers via min(std, IQR/1.34) -- standard choice, no fitting needed."""
    residuals = np.asarray(residuals, dtype=float)
    n = len(residuals)
    std = np.std(residuals, ddof=1)
    q75, q25 = np.percentile(residuals, [75, 25])
    iqr = q75 - q25
    sigma = min(std, iqr / 1.34) if iqr > 0 else std
    return 0.9 * sigma * n ** (-1 / 5)


def residual_cdf(x: np.ndarray, residuals: np.ndarray, bandwidth: float) -> np.ndarray:
    """Vectorized Gaussian-kernel-smoothed empirical CDF of `residuals`,
    evaluated at each point in `x`: mean over calibration residuals r of
    Phi((x - r) / bandwidth). This is the CDF of the same kernel density
    `margin_pmf` below discretizes -- consistent by construction, and O(len(x)
    * len(residuals)) which is trivial at this project's fold sizes
    (hundreds to low thousands of validation games)."""
    x = np.atleast_1d(np.asarray(x, dtype=float))
    residuals = np.asarray(residuals, dtype=float)
    z = (x[:, None] - residuals[None, :]) / bandwidth
    return norm.cdf(z).mean(axis=1)


def margin_pmf(
    point_pred: float, residuals: np.ndarray, bandwidth: float, lo: int, hi: int
) -> tuple[np.ndarray, np.ndarray]:
    """Discretized probability of the outcome landing on each integer margin
    in [lo, hi], for a game whose point prediction is `point_pred`. Kernel
    density evaluated at integer offsets from the point prediction (bin
    width 1, bandwidth is normally several points wide, so the
    point-density approximation of the bin mass is accurate here) and
    renormalized over the requested range. Returns (margins, probabilities).
    """
    margins = np.arange(lo, hi + 1)
    residuals = np.asarray(residuals, dtype=float)
    z = (margins[:, None] - point_pred - residuals[None, :]) / bandwidth
    density = norm.pdf(z).mean(axis=1)
    probs = density / density.sum()
    return margins, probs


def win_probability(point_pred: float, residuals: np.ndarray, bandwidth: float) -> float:
    """P(diff > 0) for a `diff` point prediction -- moneyline win probability."""
    return float(1.0 - residual_cdf(np.array([-point_pred]), residuals, bandwidth)[0])


def cover_probability(point_pred: float, line: float, residuals: np.ndarray, bandwidth: float) -> float:
    """P(diff > line) for a `diff` point prediction against a market spread
    `line` -- e.g. line=-9 for a home team favored by 9 ("home -9"): a bet on
    home covering wins iff actual diff > 9, so pass line=9 (favorite's own
    line's magnitude), not the signed -9 the market quotes it as. Caller's
    responsibility to convert the quoted spread to "diff must exceed this"
    before calling."""
    return float(1.0 - residual_cdf(np.array([line - point_pred]), residuals, bandwidth)[0])
