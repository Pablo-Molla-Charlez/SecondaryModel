"""Thin caller for results-summary plots.

All implementations live in plots.py. This script just invokes them.
"""
from pathlib import Path
import sys

# Ensure project root is importable for `from src.Utils.backtest.engine import ...`
_HERE = Path(__file__).resolve().parent
_SRC  = _HERE.parents[1]   # M2_DS/Secondary-Model/src
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from Utils.feature_selection.plots import (
    plot_cpcv_filter_confusion,
    compute_tab_vs_ctts_comparison,
    plot_results_matrices,
    build_combined_metrics_dict,
)

bt_root   = Path("/home/pablo/M2_DS/Secondary-Model/src/Output")
edge_root = bt_root / "Analysis" / "Edge_NoCal"
results_root = bt_root / "Analysis" / "Results"
results_root.mkdir(parents=True, exist_ok=True)

# 1) CPCV filter confusion bar chart
plot_cpcv_filter_confusion(
    edge_root=edge_root,
    bt_root=bt_root,
    save_path=edge_root / "cpcv_edge_heatmap_test.png",
)

# 2) Tab vs CTTS comparison JSON
compute_tab_vs_ctts_comparison(
    bt_root=bt_root,
    edge_root=edge_root,
    save_path=edge_root / "tab_vs_ctts_comparison.json",
)

# 3) Results matrices summary
build_combined_metrics_dict(
    bt_root=bt_root,
    edge_root=edge_root,
    save_path=results_root / "metrics_combined_dict.pkl",
)

plot_results_matrices(
    bt_root=bt_root,
    edge_root=edge_root,
    save_path=results_root / "results_matrices_summary.png",
)
