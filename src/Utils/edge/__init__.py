"""Edge analysis package.

Split from ``_impl.py`` into:

- :mod:`edge` — CPCV splits, baselines, single/seed/cpcv trial runners,
  convergence scoring, CLI ``main()``
- :mod:`plots` — per-trial edge curves, summary boxplots, cross-gran plots,
  split matrix visualization, path-metric boxplots

Kept as a 2-way split rather than the suggested 5-way because the seed-trial
and CPCV runners share substantial helper machinery (``_run_single_trial``,
``_build_edge_model``, baseline helpers) that would require heavy re-plumbing
to separate cleanly.

Backward-compat: ``_compute_embargo_splits`` and ``_gran_to_timedelta``
remain importable from ``Utils.edge`` for ``kronos_tree.py`` callers.
"""
from Utils.edge.edge import *  # noqa: F401,F403
from Utils.edge.edge import (
    _compute_embargo_splits,
    _gran_to_timedelta,
    _compute_m1_baselines,
    run_seeds_analysis,
    run_cpcv_analysis,
    compute_edge_convergence_score,
    CAL_SPLIT_RATIO,
    CPCV_OOB_CAL_RATIO,
    EDGE_SEED,
)
from Utils.edge import plots  # noqa: F401 — ensure plot fns registered for internal calls
