"""A genuinely held-out validation-backtest split (tb_pilot_plan.md Phase 3, corrected).

`compute_held_out_split` must subdivide Train into {Train(fit), Val-Backtest} while
leaving Val-Cal/Val-Opt/Test byte-identical to Utils.edge's existing 4-way split -- the
whole point is to stop reusing Val-Cal+Val-Opt (the utility-threshold optimizer's own
data) as if it were held-out performance.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tb_pilot.held_out_split import compute_held_out_split
from Utils.edge import _compute_embargo_splits

GRANULARITY = "1d"
HORIZON = 7


def _daily_dates(start: str, end: str) -> list[pd.Timestamp]:
    return list(pd.date_range(start, end, freq="D"))


@pytest.fixture
def wide_dates():
    # 2 years of daily dates: plenty of history before train_end to carve a
    # val-sized held-out slice without tripping the "not enough history" guard.
    return _daily_dates("2023-01-01", "2025-12-31")


TRAIN_END = "2025-05-30"
VAL_END = "2025-10-01"


def test_cal_opt_test_are_byte_identical_to_the_original_4_way_split(wide_dates):
    """The whole fix is additive: Val-Cal/Val-Opt/Test must not move at all."""
    original = _compute_embargo_splits(wide_dates, TRAIN_END, VAL_END, HORIZON, GRANULARITY)
    held_out = compute_held_out_split(wide_dates, TRAIN_END, VAL_END, HORIZON, GRANULARITY)

    np.testing.assert_array_equal(held_out["idx_cal"], original["idx_cal"])
    np.testing.assert_array_equal(held_out["idx_opt"], original["idx_opt"])
    np.testing.assert_array_equal(held_out["idx_test"], original["idx_test"])
    assert held_out["cal_end"] == original["cal_end"]
    assert held_out["purge_td"] == original["purge_td"]


def test_train_and_val_bt_partition_the_original_train_slice_modulo_purge(wide_dates):
    """idx_train(new) + purge-gap + idx_val_bt must reconstruct the original idx_train."""
    original = _compute_embargo_splits(wide_dates, TRAIN_END, VAL_END, HORIZON, GRANULARITY)
    held_out = compute_held_out_split(wide_dates, TRAIN_END, VAL_END, HORIZON, GRANULARITY)

    reconstructed = set(held_out["idx_train"]) | set(held_out["idx_val_bt"])
    original_train = set(original["idx_train"])
    # Only the new purge gap (introduced at train_fit_end) is missing from the union.
    missing = original_train - reconstructed
    purge_days = held_out["purge_td"]
    for i in missing:
        d = wide_dates[i]
        assert held_out["train_fit_end"] < d <= held_out["train_fit_end"] + purge_days
    assert reconstructed <= original_train


def test_val_bt_is_the_same_length_as_the_original_val_window(wide_dates):
    held_out = compute_held_out_split(wide_dates, TRAIN_END, VAL_END, HORIZON, GRANULARITY)
    val_duration = pd.Timestamp(VAL_END) - pd.Timestamp(TRAIN_END)
    assert pd.Timestamp(TRAIN_END) - held_out["train_fit_end"] == val_duration


def test_no_two_slices_overlap(wide_dates):
    held_out = compute_held_out_split(wide_dates, TRAIN_END, VAL_END, HORIZON, GRANULARITY)
    slices = [held_out[k] for k in ("idx_train", "idx_val_bt", "idx_cal", "idx_opt", "idx_test")]
    seen: set[int] = set()
    for s in slices:
        s_set = set(s.tolist())
        assert not (s_set & seen), "overlapping indices between slices"
        seen |= s_set


def test_purge_gap_around_train_fit_end_is_excluded_from_every_slice(wide_dates):
    held_out = compute_held_out_split(wide_dates, TRAIN_END, VAL_END, HORIZON, GRANULARITY)
    purge_start = held_out["train_fit_end"]
    purge_end = purge_start + held_out["purge_td"]
    in_any_slice = set(held_out["idx_train"].tolist()) | set(held_out["idx_val_bt"].tolist())
    for i, d in enumerate(wide_dates):
        if purge_start < d <= purge_end:
            assert i not in in_any_slice


def test_raises_when_there_is_not_enough_history_for_the_held_out_slice():
    # train_end only a few days after the earliest date: no room for a val-sized
    # held-out slice before it.
    short_dates = _daily_dates("2025-05-01", "2025-12-31")
    with pytest.raises(ValueError, match="not enough history"):
        compute_held_out_split(short_dates, TRAIN_END, VAL_END, HORIZON, GRANULARITY)


def test_val_bt_has_comparable_sample_size_to_cal_plus_opt(wide_dates):
    held_out = compute_held_out_split(wide_dates, TRAIN_END, VAL_END, HORIZON, GRANULARITY)
    n_val_bt = len(held_out["idx_val_bt"])
    n_cal_opt = len(held_out["idx_cal"]) + len(held_out["idx_opt"])
    # Same wall-clock length by construction, but Val-Backtest has one purge zone
    # trimmed from it (at train_fit_end) while Cal+Opt has two (at train_end and at
    # cal_end) -- so Val-Backtest ends up one purge window larger, not equal.
    purge_days = held_out["purge_td"].days
    assert n_val_bt - n_cal_opt == purge_days
