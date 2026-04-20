"""Time-series cross-validation utilities."""

# ┏━━━━━━━━━━ Imports from base class ━━━━━━━━━━┓
from Utils.ts_cross_validation._ts_cross_validation import BaseTimeSeriesCV

# ┏━━━━━━━━━━ Imports from combinatorial purged CV ━━━━━━━━━━┓
from Utils.ts_cross_validation.combinatorial_purged_cv import (CombinatorialPurgedCV,
                                                               _gran_to_timedelta,
                                                               _build_datetime_blocks,
                                                               _assign_blocks,
                                                               _generate_cpcv_splits,
                                                               _reconstruct_paths)

# ┏━━━━━━━━━━ Imports from purged embargo CV ━━━━━━━━━━┓
from Utils.ts_cross_validation.purged_embargo_cv import PurgedEmbargoTimeSeriesCV

# ┏━━━━━━━━━━ Imports from sklearn ts CV ━━━━━━━━━━┓
from Utils.ts_cross_validation.sklearn_ts_cv import SklearnTimeSeriesCV

# ┏━━━━━━━━━━ Imports from embargo splits ━━━━━━━━━━┓
from Utils.ts_cross_validation.embargo_splits import (compute_embargo_splits,
                                                      compute_seeds_embargo_splits,
                                                      CAL_SPLIT_RATIO,
                                                      SEEDS_TRAIN_FRAC,
                                                      SEEDS_CAL_FRAC,
                                                      SEEDS_OPT_FRAC)

# ┏━━━━━━━━━━ List of all exported classes and functions ━━━━━━━━━━┓
__all__ = ["BaseTimeSeriesCV",
           "CombinatorialPurgedCV",
           "PurgedEmbargoTimeSeriesCV",
           "SklearnTimeSeriesCV",
           "compute_embargo_splits",
           "compute_seeds_embargo_splits",
           "CAL_SPLIT_RATIO",
           "SEEDS_TRAIN_FRAC",
           "SEEDS_CAL_FRAC",
           "SEEDS_OPT_FRAC",
           "_gran_to_timedelta",
           "_build_datetime_blocks",
           "_assign_blocks",
           "_generate_cpcv_splits",
           "_reconstruct_paths"]