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
    # ┏━━━━━━━━━━ Determine bar_td and purge_td ━━━━━━━━━━┓
    bar_td = _gran_to_timedelta(granularity)
    purge_td = bar_td * horizon

    # ┏━━━━━━━━━━ Convert train_end and val_end to Timestamps ━━━━━━━━━━┓
    t_train_end = pd.Timestamp(train_end)
    t_val_end = pd.Timestamp(val_end)

    # ┏━━━━━━━━━━ Compute calibration end time ━━━━━━━━━━┓
    val_duration = t_val_end - t_train_end
    t_cal_end = t_train_end + CAL_SPLIT_RATIO * val_duration

    # ┏━━━━━━━━━━ Initialize index lists ━━━━━━━━━━┓
    idx_train, idx_cal, idx_opt, idx_test = [], [], [], []
    
    # ┏━━━━━━━━━━ Iterate over dates to assign indices ━━━━━━━━━━┓
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

    return {"idx_train": np.array(idx_train),
            "idx_cal": np.array(idx_cal),
            "idx_opt": np.array(idx_opt),
            "idx_test": np.array(idx_test),
            "cal_end": t_cal_end,
            "purge_td": purge_td}


# ┏━━━━━━━━━━ Seeds-experiment split ratios (carved inside train+val ONLY) ━━━━━━━━━━┓
# The seeds-convergence experiment must never touch the real test set. We take
# the union of the configured Train + Val window, purge the tail-most slice of
# the real test window, and carve four sub-splits entirely inside it:
#
#   [  Train  | purge | Val-Cal | purge | Val-Opt | purge | Val-Eval  ]  (Test is untouched)
#
# Ratios apply to the Train+Val timeline (t0 .. val_end). "Val-Eval" is the
# hold-out used for the per-seed noise measurement; it stands in for the real
# test set so the real test set stays unseen until final evaluation.
SEEDS_TRAIN_FRAC    = 0.65   # first 65% of train+val window
SEEDS_CAL_FRAC      = 0.12   # next 12%
SEEDS_OPT_FRAC      = 0.10   # next 10%
# Val-Eval = remaining 13%


def compute_seeds_embargo_splits(dates_valid, train_end, val_end, horizon, granularity):
    """4-way split carved inside train+val only — real test is never used.

    Unlike ``compute_embargo_splits`` (used for final evaluation, which exposes
    the held-out test window), this split keeps the final test window fully
    invisible. Per-seed variance is measured on ``idx_val_eval``, a purged
    hold-out carved from the tail of the train+val timeline.

    Returns dict with keys:
        idx_train, idx_cal, idx_opt, idx_val_eval, purge_td,
        boundaries (dict of pd.Timestamp boundary markers for logging/plots).
    """
    # ┏━━━━━━━━━━ Determine bar_td and purge_td ━━━━━━━━━━┓
    bar_td = _gran_to_timedelta(granularity)
    purge_td = bar_td * horizon

    # ┏━━━━━━━━━━ Convert train_end and val_end to Timestamps ━━━━━━━━━━┓
    t0 = min(dates_valid) if len(dates_valid) > 0 else pd.Timestamp(train_end)
    t_val_end = pd.Timestamp(val_end)
    window = t_val_end - t0
    if window.total_seconds() <= 0:
        raise ValueError(f"compute_seeds_embargo_splits: non-positive window "
                         f"({t0} → {t_val_end})")

    # ┏━━━━━━━━━━ Compute split end times ━━━━━━━━━━┓
    t_train_end   = t0 + SEEDS_TRAIN_FRAC                                    * window
    t_cal_end     = t0 + (SEEDS_TRAIN_FRAC + SEEDS_CAL_FRAC)                 * window
    t_opt_end     = t0 + (SEEDS_TRAIN_FRAC + SEEDS_CAL_FRAC + SEEDS_OPT_FRAC) * window
    # Val-Eval runs from t_opt_end + purge → t_val_end

    # ┏━━━━━━━━━━ Initialize index lists ━━━━━━━━━━┓
    idx_train, idx_cal, idx_opt, idx_val_eval = [], [], [], []
    
    # ┏━━━━━━━━━━ Iterate over dates to assign indices ━━━━━━━━━━┓
    for i, d in enumerate(dates_valid):
        if d > t_val_end:
            continue                                    # real TEST — untouched
        elif d <= t_train_end:
            idx_train.append(i)
        elif d <= t_train_end + purge_td:
            continue
        elif d <= t_cal_end:
            idx_cal.append(i)
        elif d <= t_cal_end + purge_td:
            continue
        elif d <= t_opt_end:
            idx_opt.append(i)
        elif d <= t_opt_end + purge_td:
            continue
        else:
            idx_val_eval.append(i)

    return {"idx_train":    np.array(idx_train),
            "idx_cal":      np.array(idx_cal),
            "idx_opt":      np.array(idx_opt),
            "idx_val_eval": np.array(idx_val_eval),
            "purge_td":     purge_td,
            "boundaries": {"t0":          t0,
                           "train_end":   t_train_end,
                           "cal_end":     t_cal_end,
                           "opt_end":     t_opt_end,
                           "val_end":     t_val_end},
        }
