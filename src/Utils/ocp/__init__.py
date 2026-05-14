"""Online Conformal Prediction / SAOCP (package).

Split from the previous ``_saocp_impl.py`` monolith into:

- :mod:`saocp` — SAOCP core (calibration windows, feeds, online runners)
- :mod:`plots` — Mondrian diagnostic plots

``analysis.py`` / ``theory.py`` remain CLI entrypoints that delegate to the
large ``_analysis_impl.py`` / ``_theory_impl.py`` modules — those were NOT
split in this pass (each is 60-85 KB of tightly-coupled analysis code whose
bucketing was ambiguous).
"""
# ┏━━━━━━━━━━ Stronly Adaptive Online Conformal Prediction (SAOCP) ━━━━━━━━━━┓
from Utils.ocp.saocp import (calib_window_for_gran,
                             _make_saocp,
                             _saocp_feed,
                             _warm_saocp,
                             _ocp_conformity_score,
                             _run_saocp_online,
                             _cost_grid_search,
                             _realized_volatility,
                             _run_cost_deferral_online,
                             _ocp_threshold_to_op)

# ┏━━━━━━━━━━ Conformal Plotting ━━━━━━━━━━┓
from Utils.ocp.plots import plot_mondrian_diagnostics, plot_ocp_threshold_evolution