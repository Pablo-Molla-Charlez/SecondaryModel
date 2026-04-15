"""4-way Train / Val-Cal / Val-Opt / Test split with purge at every boundary.

Moved verbatim from ``Utils/edge.py::_compute_embargo_splits`` so it can be
reused without depending on the (large) ``edge`` module. Return dict shape is
preserved — ``{idx_train, idx_cal, idx_opt, idx_test, cal_end, purge_td}``.
"""
import numpy as np
import pandas as pd

from Utils.ts_cross_validation.combinatorial_purged_cv import _gran_to_timedelta


# ┏━━━━━━━━━━ Calibration split ratio (Val-Cal / Val-Opt) ━━━━━━━━━━┓
# First CAL_SPLIT_RATIO of Val = Val-Cal (Calibration Set)
# Last (1 - CAL_SPLIT_RATIO) of Val = Val-Opt (Threshold Optimization Set)
CAL_SPLIT_RATIO = 0.40


def compute_embargo_splits(dates_valid, train_end, val_end, horizon, granularity):
    """Compute indices for Train / Val-Cal / Val-Opt / Test with embargo gaps.

    Purge = horizon x bar_width at each boundary to prevent label leakage.
    Cal/Opt split is determined by ``CAL_SPLIT_RATIO`` applied to the Val window.

    Returns dict with keys: idx_train, idx_cal, idx_opt, idx_test, cal_end, purge_td.
    """
    bar_td = _gran_to_timedelta(granularity)
    purge_td = bar_td * horizon

    t_train_end = pd.Timestamp(train_end)
    t_val_end = pd.Timestamp(val_end)

    val_duration = t_val_end - t_train_end
    t_cal_end = t_train_end + CAL_SPLIT_RATIO * val_duration

    idx_train, idx_cal, idx_opt, idx_test = [], [], [], []
    for i, d in enumerate(dates_valid):
        if d <= t_train_end:
            idx_train.append(i)
        elif d <= t_train_end + purge_td:
            continue
        elif d <= t_cal_end:
            idx_cal.append(i)
        elif d <= t_cal_end + purge_td:
            continue
        elif d <= t_val_end:
            idx_opt.append(i)
        elif d <= t_val_end + purge_td:
            continue
        else:
            idx_test.append(i)

    return {
        "idx_train": np.array(idx_train),
        "idx_cal": np.array(idx_cal),
        "idx_opt": np.array(idx_opt),
        "idx_test": np.array(idx_test),
        "cal_end": t_cal_end,
        "purge_td": purge_td,
    }
