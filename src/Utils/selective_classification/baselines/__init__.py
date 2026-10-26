"""Reliability-gate baselines for the M2 comparison (Kronos as M1).

Each module scores M1 signals and executes those above a validation-selected
threshold, so all baselines share the M2 operating-point convention:

- :mod:`quantiles_selective`        — M1+PI, native predictive-interval width gate (no training).
- :mod:`spci.scpi_selective`        — M1+SPCI, conformal interval gate (wraps the vendored
                            upstream SPCI source in ``spci/``).
- :mod:`random_selective` — random coverage-matched control with an empirical p-value.

:mod:`transfer_baselines_table` aggregates their CSVs into the paper's LaTeX table.
"""

from Utils.selective_classification.baselines._common import (  # noqa: F401
    GRANULARITIES,
    SIDES,
    SignalFrame,
    discover_run_dir,
    list_assets,
    load_signal_frame,
    m1_baseline_precision,
    pick_threshold,
    selective_metrics,
    split_indices,
    subset,
)

__all__ = [
    "GRANULARITIES", "SIDES", "SignalFrame", "discover_run_dir", "list_assets",
    "load_signal_frame", "m1_baseline_precision", "pick_threshold",
    "selective_metrics", "split_indices", "subset",
]
