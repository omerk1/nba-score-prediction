"""Regression test for a real bug found while scoping CV-integrated
hyperparameter tuning: ScorePredictor._create_model (src/models/
score_predictor.py) accepted l2_leaf_reg/min_data_in_leaf into
**model_params but never read them back out before constructing
CatBoostRegressor -- scripts/tune_model.py's Optuna search could sweep
those two dimensions with zero effect on the actual trained model. Fixed
alongside making CatBoost hyperparameters config-driven
(src/utils/config_loader.py's ModelConfig, src/evaluation/cv_harness.py).
"""

from src.models.score_predictor import ScorePredictor


def test_l2_leaf_reg_and_min_data_in_leaf_reach_catboost():
    predictor = ScorePredictor(
        model_type="catboost",
        random_state=42,
        verbose=False,
        l2_leaf_reg=17.5,
        min_data_in_leaf=23,
    )
    model = predictor._create_model()
    params = model.get_params()

    assert params["l2_leaf_reg"] == 17.5
    assert params["min_data_in_leaf"] == 23


def test_l2_leaf_reg_and_min_data_in_leaf_default_to_catboost_native_defaults():
    predictor = ScorePredictor(model_type="catboost", random_state=42, verbose=False)
    model = predictor._create_model()
    params = model.get_params()

    assert params["l2_leaf_reg"] == 3.0
    assert params["min_data_in_leaf"] == 1


def test_all_tunable_hyperparameters_reach_catboost():
    predictor = ScorePredictor(
        model_type="catboost",
        random_state=42,
        verbose=False,
        depth=4,
        learning_rate=0.03,
        subsample=0.6,
        colsample_bylevel=0.7,
        l2_leaf_reg=9.0,
        min_data_in_leaf=5,
    )
    params = predictor._create_model().get_params()

    assert params["depth"] == 4
    assert params["learning_rate"] == 0.03
    assert params["subsample"] == 0.6
    assert params["colsample_bylevel"] == 0.7
    assert params["l2_leaf_reg"] == 9.0
    assert params["min_data_in_leaf"] == 5
