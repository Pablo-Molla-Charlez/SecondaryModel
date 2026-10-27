"""A genuinely held-out validation-backtest split (tb_pilot_plan.md Phase 3, corrected).

The original Phase 3 wiring backtested on Val-Cal + Val-Opt -- but
`_find_best_utility_threshold` (Utils/selective_classification/thresholds.py) picks
the approval threshold by directly maximizing net return/utility over exactly that
window. Backtesting the chosen threshold on the same window it was tuned on measures
how well the optimizer fit that data, not held-out performance -- confirmed empirically
(Sharpe ~5-6 there vs ~0.9-1.4 on this repo's real, disjoint Test split using the same
threshold-selection method).

This module carves a fifth slice, Val-Backtest, out of the tail of the training
window -- used for nothing except a k-selection backtest after the model is already
fit and the threshold already chosen. Val-Cal, Val-Opt and Test are left untouched
(byte-identical to Utils.edge._compute_embargo_splits); Train shrinks by the same
wall-clock length as the existing Val window (train_end..val_end), so Val-Backtest has
comparable statistical power to Val-Cal+Val-Opt combined.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from Utils.ts_cross_validation import CAL_SPLIT_RATIO, _gran_to_timedelta


def compute_held_out_split(dates_valid, train_end, val_end, horizon, granularity) -> dict:
    """Train(fit) / Val-Backtest(new) / Val-Cal / Val-Opt / Test, purged by
    horizon x bar_width at every boundary (same convention as
    Utils.edge._compute_embargo_splits, which this extends rather than replaces).

    Returns a dict with idx_train, idx_val_bt, idx_cal, idx_opt, idx_test (positions
    into `dates_valid`), plus train_fit_end, cal_end, purge_td.
    """
    bar_td = _gran_to_timedelta(granularity)
    purge_td = bar_td * horizon

    t_train_end = pd.Timestamp(train_end)
    t_val_end = pd.Timestamp(val_end)
    val_duration = t_val_end - t_train_end
    t_cal_end = t_train_end + CAL_SPLIT_RATIO * val_duration

    t_train_fit_end = t_train_end - val_duration
    earliest = min(dates_valid)
    if t_train_fit_end <= pd.Timestamp(earliest):
        raise ValueError(
            f"not enough history to carve a held-out Val-Backtest slice: train_fit_end "
            f"({t_train_fit_end.date()}) would fall before the earliest available date "
            f"({pd.Timestamp(earliest).date()}). Shorten the Val window or widen the data range."
        )

    idx_train: list[int] = []
    idx_val_bt: list[int] = []
    idx_cal: list[int] = []
    idx_opt: list[int] = []
    idx_test: list[int] = []
    for i, d in enumerate(dates_valid):
        if d <= t_train_fit_end:
            idx_train.append(i)
        elif d <= t_train_fit_end + purge_td:
            continue  # new purge zone
        elif d <= t_train_end:
            idx_val_bt.append(i)
        elif d <= t_train_end + purge_td:
            continue  # existing purge zone (Utils.edge convention)
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
        "idx_val_bt": np.array(idx_val_bt),
        "idx_cal": np.array(idx_cal),
        "idx_opt": np.array(idx_opt),
        "idx_test": np.array(idx_test),
        "train_fit_end": t_train_fit_end,
        "cal_end": t_cal_end,
        "purge_td": purge_td,
    }
