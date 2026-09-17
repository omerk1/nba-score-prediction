"""
Non-LLM estimators for P(plays) on uncertain injury listings.

All estimators share one interface: fit(train_df) -> self, predict(df) -> array
of probabilities. `train_df`/`df` are context frames from retrieval.build_context
(CONTEXT_COLUMNS present, `played` present on train). Every estimator is
trained only on seasons before the one it is evaluated on (see eval script), so
the priors are themselves point-in-time.

1. StatusPrior          — play rate per status.
2. StatusSeverityPrior  — play rate per (status, severity bucket), with the
                          status prior as a back-off for unseen cells.
3. LogisticBaseline     — logistic regression on the numeric context.
4. CatBoostBaseline     — small CatBoost classifier on the same context.

The LLM estimator (later phase) must beat 3 and 4 on the same rows before it
is wired into the feature pipeline.
"""

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.availability.retrieval import CONTEXT_COLUMNS


class StatusPrior:
    name = "status_prior"

    def fit(self, df: pd.DataFrame):
        self.rates = df.groupby("status")["played"].mean().to_dict()
        self.global_rate = df["played"].mean()
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return df["status"].map(self.rates).fillna(self.global_rate).to_numpy(dtype=float)


class StatusSeverityPrior:
    name = "status_severity_prior"
    MIN_CELL = 20

    def fit(self, df: pd.DataFrame):
        self.back_off = StatusPrior().fit(df)
        g = df.groupby(["status", "severity"])["played"].agg(["mean", "count"])
        self.rates = g[g["count"] >= self.MIN_CELL]["mean"].to_dict()
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        fallback = self.back_off.predict(df)
        keys = list(zip(df["status"], df["severity"]))
        return np.array([self.rates.get(k, f) for k, f in zip(keys, fallback)], dtype=float)


class LogisticBaseline:
    name = "logistic"

    def __init__(self, columns=CONTEXT_COLUMNS):
        self.columns = list(columns)
        self.model = make_pipeline(
            SimpleImputer(strategy="median", add_indicator=True),
            StandardScaler(),
            LogisticRegression(C=0.5, max_iter=2000),
        )

    def fit(self, df: pd.DataFrame):
        self.model.fit(df[self.columns].astype(float), df["played"])
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(df[self.columns].astype(float))[:, 1]


class CatBoostBaseline:
    name = "catboost"

    def __init__(self, columns=CONTEXT_COLUMNS, seed: int = 42):
        from catboost import CatBoostClassifier

        self.columns = list(columns)
        self.model = CatBoostClassifier(
            iterations=400,
            depth=4,
            learning_rate=0.05,
            l2_leaf_reg=5,
            loss_function="Logloss",
            random_seed=seed,
            verbose=False,
            allow_writing_files=False,
        )

    def fit(self, df: pd.DataFrame):
        self.model.fit(df[self.columns].astype(float), df["played"])
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(df[self.columns].astype(float))[:, 1]


def all_baselines() -> list:
    return [StatusPrior(), StatusSeverityPrior(), LogisticBaseline(), CatBoostBaseline()]
