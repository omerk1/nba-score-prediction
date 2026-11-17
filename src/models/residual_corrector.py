"""
Second-stage margin correction fit on the main model's out-of-sample errors.

The main model under-uses information it already has (a back-to-back team is
over-rated by ~1.3 pts; a team's prediction errors persist within a season),
and adding the obvious features didn't fix it. This layer instead fits a
small ridge regression of the residual (actual - predicted home margin) on a
fixed, pre-registered input list, using predictions for seasons the main model
never trained on (earlier CV folds' validation seasons), and adds the fitted
adjustment to new predictions. Totals are not touched.

Inputs (home minus away, so the correction is antisymmetric in the teams):
- rest: back-to-back, rest days (capped at +/-5), games in 4 nights;
- team_error (optional): each team's mean prediction error over its last
  `error_window` games this season, from that team's side, 0 until it has
  `min_games` such games. Only games already played enter, scored by the same
  model that makes the prediction, so it is point-in-time.

No intercept: the layer only moves predictions in response to its inputs.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

REST_INPUTS = ["b2b_diff", "rest_diff", "dense_diff"]
TEAM_ERROR_INPUT = "team_error_diff"


def rest_inputs(df: pd.DataFrame) -> pd.DataFrame:
    """Home-minus-away rest inputs from the feature builder's own rest columns."""
    return pd.DataFrame(
        {
            "b2b_diff": df["home_team_back_to_back"].values - df["away_team_back_to_back"].values,
            "rest_diff": np.clip(df["home_team_rest_days"].values - df["away_team_rest_days"].values, -5, 5),
            "dense_diff": df["home_team_games_in_4_nights"].values - df["away_team_games_in_4_nights"].values,
        },
        index=df.index,
    ).astype(float)


def team_error_input(df: pd.DataFrame, pred_diff, window: int = 10, min_games: int = 3) -> np.ndarray:
    """home team's recent mean error minus away team's, each from its own side.

    `df` needs GAME_DATE, SEASON_ID, HOME_TEAM_ID, AWAY_TEAM_ID and the actual
    margin POINT_DIFF (used only for games strictly before the row's game).
    """
    err_home = df["POINT_DIFF"].values - np.asarray(pred_diff, dtype=float)
    long_df = pd.concat(
        [
            pd.DataFrame(
                {"row": np.arange(len(df)), "side": 1, "team": df["HOME_TEAM_ID"].values, "err": err_home}
            ),
            pd.DataFrame(
                {"row": np.arange(len(df)), "side": -1, "team": df["AWAY_TEAM_ID"].values, "err": -err_home}
            ),
        ]
    )
    long_df["date"] = np.tile(pd.to_datetime(df["GAME_DATE"]).values, 2)
    long_df["season"] = np.tile(df["SEASON_ID"].values, 2)
    long_df = long_df.sort_values(["date", "row"], kind="stable")
    grp = long_df.groupby(["season", "team"])["err"]
    prior_mean = grp.transform(lambda x: x.shift(1).rolling(window, min_periods=min_games).mean())
    long_df["recent"] = prior_mean.fillna(0.0)
    home = long_df[long_df.side == 1].set_index("row")["recent"].reindex(range(len(df))).values
    away = long_df[long_df.side == -1].set_index("row")["recent"].reindex(range(len(df))).values
    return home - away


def build_inputs(
    df: pd.DataFrame, pred_diff, use_team_error: bool, window: int = 10, min_games: int = 3
) -> pd.DataFrame:
    X = rest_inputs(df)
    if use_team_error:
        X[TEAM_ERROR_INPUT] = team_error_input(df, pred_diff, window, min_games)
    return X


@dataclass(frozen=True)
class ResidualCorrector:
    columns: tuple
    coef: tuple

    @classmethod
    def fit(cls, X: pd.DataFrame, residual, alpha: float = 10.0) -> "ResidualCorrector":
        A = X.to_numpy(dtype=float)
        y = np.asarray(residual, dtype=float)
        coef = np.linalg.solve(A.T @ A + alpha * np.eye(A.shape[1]), A.T @ y)
        return cls(tuple(X.columns), tuple(float(c) for c in coef))

    def adjustment(self, X: pd.DataFrame) -> np.ndarray:
        return X[list(self.columns)].to_numpy(dtype=float) @ np.asarray(self.coef)
