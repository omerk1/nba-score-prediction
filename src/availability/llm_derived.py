"""
Estimators that build on the raw LLM output.

The raw LLM probability ranks rows better than the status prior but is badly
calibrated (it states low probabilities far more confidently than reality
supports). Two standard remedies, both fit only on seasons before the
evaluated one, so they stay point-in-time like every other estimator here:

- CalibratedLLM: isotonic regression mapping raw p_play -> observed play rate.
- CatBoostPlusLLM: the tabular model with the raw p_play added as one more
  feature. This answers the question the plain comparison cannot: does the LLM
  carry any signal the retrieved facts do not already contain?

Both need an `llm_p_play` column on the context frame (see
scripts/attach_llm_predictions.py), which is served entirely from the response
cache.
"""

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

from src.availability.baselines import CatBoostBaseline
from src.availability.retrieval import CONTEXT_COLUMNS

LLM_COLUMN = "llm_p_play"


class CalibratedLLM:
    name = "llm_calibrated"

    def __init__(self, column: str = LLM_COLUMN):
        self.column = column
        self.model = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)

    def fit(self, df: pd.DataFrame):
        ok = df[self.column].notna()
        self.model.fit(
            df.loc[ok, self.column].to_numpy(dtype=float), df.loc[ok, "played"].to_numpy(dtype=float)
        )
        self.fallback = float(df["played"].mean())
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        raw = df[self.column].to_numpy(dtype=float)
        out = np.full(len(raw), self.fallback)
        ok = ~np.isnan(raw)
        out[ok] = self.model.predict(raw[ok])
        return out


class CatBoostPlusLLM(CatBoostBaseline):
    name = "catboost_plus_llm"

    def __init__(self, column: str = LLM_COLUMN, seed: int = 42):
        super().__init__(columns=CONTEXT_COLUMNS + [column], seed=seed)
