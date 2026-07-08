"""
Polymarket in-game price history pipeline.

Standalone data-science pipeline (NOT integrated with feature_builder.py /
train_model.py / predict_game.py). Downloads in-game price histories
(win-probability time series) of resolved Polymarket NBA moneyline markets
and builds a general-purpose per-game dataset -- comeback analysis (the
minimum in-game price ever traded for the eventual winner's token) is one
use of this data, not the only one; the raw price series is stored in full
so other questions can be asked of it later.

See the module docstrings in gamma.py / data_api.py for empirically-verified
API schema notes and known limitations (offset cap, CLOB coarseness, etc.),
and scripts/fetch_polymarket_prices.py for the collection CLI entry point
(scripts/analyze_polymarket_comebacks.py is the comeback-specific analysis
script built on top of this data, not part of this package).
"""
