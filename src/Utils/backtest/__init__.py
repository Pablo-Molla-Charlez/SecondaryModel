"""Financial backtest utilities (package).

Split from the previous ``_impl.py`` / ``_comparison_impl.py`` monoliths into:

- :mod:`engine` — raw prices, Sharpe/drawdown, combined backtest, feature backtest
- :mod:`plots` — ``_plot_path_equity``
- :mod:`comparison` — per-gran vs unified comparison, paradigm comparison
"""
from Utils.backtest.engine import *  # noqa: F401,F403
from Utils.backtest.engine import (
    _annualization_factor,
    _build_spread_equity,
    _calc_drawdown,
    _calc_sharpe,
    _equity_horizon_returns,
    _load_raw_close_prices,
    run_feature_backtest,
    run_combined_backtest,
)
from Utils.backtest.plots import _plot_path_equity
from Utils.backtest.comparison import (
    GRAN_ORDER,
    run_comparison,
    run_paradigm_comparison,
)