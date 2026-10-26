"""Edge analysis package.

- :mod:`edge` — datetime blocks, baselines, the CPCV and expanding-window CV
  runners, CLI ``main()``
- :mod:`plots` — per-trial edge curves, summary boxplots, cross-gran plots,
  split matrix visualization, path-metric boxplots

Both runners write a per-sample CSV (``cpcv_samples.csv`` / ``cvexpand_samples.csv``)
that the regime analysis in ``Utils.analysis.analysis_m2`` consumes.

Backward-compat: ``_compute_embargo_splits`` and ``_gran_to_timedelta``
remain importable from ``Utils.edge`` for ``m2_pipeline.py`` callers.
"""
# ┏━━━━━━━━━━ Edge ━━━━━━━━━━┓
from Utils.edge.edge import (_compute_embargo_splits,
                             _gran_to_timedelta,
                             _compute_m1_baselines,
                             run_cpcv_analysis,
                             run_cvexpand_analysis,
                             CAL_SPLIT_RATIO,
                             CPCV_OOB_CAL_RATIO,
                             EDGE_SEED)

# ┏━━━━━━━━━━ Plotting from Edge ━━━━━━━━━━┓
from Utils.edge import plots
