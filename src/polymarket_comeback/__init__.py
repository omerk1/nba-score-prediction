"""
Polymarket in-game comeback analysis pipeline.

Standalone data-science pipeline (NOT integrated with feature_builder.py /
train_model.py / predict_game.py). Downloads in-game price histories of
resolved Polymarket NBA moneyline markets and builds a dataset for comeback
analysis: for each game, the minimum in-game price ever traded for the
eventual winner's token.

See the module docstrings in gamma.py / data_api.py for empirically-verified
API schema notes and known limitations (offset cap, CLOB coarseness, etc.),
and run_polymarket_comeback.py / analysis.py at the repo root for the CLI
entry points.
"""
