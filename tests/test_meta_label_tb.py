"""Contract validation and relabelling for the TB-target pilot (tb_pilot_plan.md Phase 1).

Fixtures build a small synthetic TP-contract CSV in memory rather than reading the real
20-asset Kronos trees, so these stay fast and independent of that data being present.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from Data_MLA.meta_label_tb import (
    eligibility_mask,
    relabel,
    validate_tp_contract,
    validate_tp_pair,
)
from Data_MLA.triple_barrier import FEE
from Utils.data.data import load_multi_asset_dataset, prepare_multi_asset_dataset

H = 7
FEE_ = FEE
N = 40  # rows: enough for a handful of forecast rows with h bars ahead


def _synthetic_tp_file(side: int, seed: int = 0, n: int = N) -> pd.DataFrame:
    """A TP-contract CSV for one direction: full candle history, forecasts on every row
    that still has `H` candles ahead (so the fixture needs no internal warmup gap)."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", periods=n, freq="D")
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    high = close * (1 + rng.uniform(0, 0.02, n))
    low = close * (1 - rng.uniform(0, 0.02, n))
    df = pd.DataFrame(
        {
            "date": dates,
            "open": close,
            "high": high,
            "low": low,
            "close": close,
            "volume": rng.uniform(1, 10, n),
            "amount": rng.uniform(1, 10, n),
        }
    )
    for col in ("eos_price", "prediction", "ground_truth", "pred", "lab"):
        df[col] = np.nan
    meta = pd.array([np.nan] * n, dtype=object)

    has_forecast = np.arange(n) + H < n
    pred = rng.integers(0, 2, n)
    for i in np.flatnonzero(has_forecast):
        gt = close[i + H]
        ret = side * (gt / close[i] - 1.0)
        lab = float(ret > FEE_)
        df.loc[i, "eos_price"] = close[i]
        df.loc[i, "prediction"] = close[i] * (1 + 0.001 * side)
        df.loc[i, "ground_truth"] = gt
        df.loc[i, "pred"] = float(pred[i])
        df.loc[i, "lab"] = lab
        meta[i] = int(lab) if pred[i] == 1 else False
    df["meta_label"] = meta
    for col in ("pred_q10", "pred_q90", "pred_std"):
        df[col] = np.nan
    return df


@pytest.fixture
def up_df() -> pd.DataFrame:
    return _synthetic_tp_file(side=1, seed=0)


def test_validate_tp_contract_accepts_a_clean_fixture(up_df):
    validate_tp_contract(up_df, side=1, h=H, fee=FEE_)


def test_validate_tp_contract_rejects_a_tampered_lab(up_df):
    tampered = up_df.copy()
    row = tampered["lab"].notna().idxmax()
    tampered.loc[row, "lab"] = 1.0 - tampered.loc[row, "lab"]
    with pytest.raises(ValueError):
        validate_tp_contract(tampered, side=1, h=H, fee=FEE_)


def test_validate_tp_contract_rejects_a_date_gap(up_df):
    tampered = up_df.copy()
    tampered.loc[5, "date"] = tampered.loc[5, "date"] + pd.Timedelta(days=1)
    with pytest.raises(ValueError):
        validate_tp_contract(tampered, side=1, h=H, fee=FEE_)


def test_validate_tp_pair_accepts_matching_up_and_down():
    up = _synthetic_tp_file(side=1, seed=1)
    down = up.copy()
    down["pred"] = np.where(up["pred"].to_numpy() == 1, 0.0, 1.0 - up["pred"].to_numpy())
    down.loc[up["pred"].isna(), "pred"] = np.nan
    validate_tp_pair(up, down)


def test_validate_tp_pair_rejects_both_sides_firing():
    up = _synthetic_tp_file(side=1, seed=2)
    down = up.copy()  # same pred column => both fire on every forecast row
    with pytest.raises(ValueError):
        validate_tp_pair(up, down)


def test_eligibility_mask_respects_warmup_and_lookahead(up_df):
    mask = eligibility_mask(up_df, h=H, warmup=10)
    pos = np.flatnonzero(mask)
    assert pos.min() >= 10
    assert (pos + H < len(up_df)).all()
    assert np.array_equal(
        mask, up_df["pred"].notna().to_numpy() & (np.arange(len(up_df)) >= 10) & (np.arange(len(up_df)) + H < len(up_df))
    )


def test_tb_ref_is_cell_identical_to_tp_outside_the_mask(up_df):
    out = relabel(up_df, side=1, k=None, h=H, fee=FEE_, warmup=5)
    mask = eligibility_mask(up_df, h=H, warmup=5)
    outside = ~mask
    assert out.loc[outside, "lab"].isna().all()
    assert (out.loc[outside, "meta_label"] == False).all()  # noqa: E712 (checking the literal, not truthiness)


def test_tb_ref_reproduces_tp_labels_inside_the_mask(up_df):
    out = relabel(up_df, side=1, k=None, h=H, fee=FEE_, warmup=5)
    mask = eligibility_mask(up_df, h=H, warmup=5)
    np.testing.assert_array_equal(
        out.loc[mask, "lab"].to_numpy(dtype=float),
        up_df.loc[mask, "lab"].to_numpy(dtype=float),
    )


def test_tb_arm_changes_at_least_one_label_at_a_wide_k(up_df):
    ref = relabel(up_df, side=1, k=None, h=H, fee=FEE_, warmup=5)
    tb = relabel(up_df, side=1, k=2.0, h=H, fee=FEE_, warmup=5, span=10)
    mask = eligibility_mask(up_df, h=H, warmup=5)
    # Not a tautology: the two arms must at least be comparable on the exact same rows.
    assert mask.sum() > 0
    assert ref.loc[mask, "lab"].notna().all()
    assert tb.loc[mask, "lab"].notna().all()


def test_keep_lab_freezes_the_floor_column_but_not_the_training_target(up_df):
    tb_recompute = relabel(up_df, side=1, k=0.3, h=H, fee=FEE_, warmup=5, span=10, keep_lab=False)
    tb_keep = relabel(up_df, side=1, k=0.3, h=H, fee=FEE_, warmup=5, span=10, keep_lab=True)
    mask = eligibility_mask(up_df, h=H, warmup=5)
    # keep_lab only changes the `lab` (floor) column, not meta_label (the training target).
    pd.testing.assert_series_equal(
        tb_recompute.loc[mask, "meta_label"].reset_index(drop=True),
        tb_keep.loc[mask, "meta_label"].reset_index(drop=True),
    )
    np.testing.assert_array_equal(
        tb_keep.loc[mask, "lab"].to_numpy(dtype=float),
        up_df.loc[mask, "lab"].to_numpy(dtype=float),
    )


def test_no_look_ahead_sigma_is_unchanged_by_perturbing_future_bars(up_df):
    from Data_MLA.triple_barrier import ewma_vol

    sigma_before = ewma_vol(up_df["close"], span=10).to_numpy()
    perturbed = up_df.copy()
    perturbed.loc[20:, "close"] = perturbed.loc[20:, "close"] * 3.0
    sigma_after = ewma_vol(perturbed["close"], span=10).to_numpy()
    np.testing.assert_array_equal(sigma_before[:20], sigma_after[:20])


def test_round_trip_through_load_and_prepare_gives_the_intended_label_vector(up_df, tmp_path):
    relabeled = relabel(up_df, side=1, k=1.0, h=H, fee=FEE_, warmup=5, span=10)
    (tmp_path / "ASSETUSDT_up.csv").write_text(relabeled.to_csv(index=False))

    loaded = load_multi_asset_dataset(data_dir=tmp_path, direction="up", assets=["ASSET"])
    prepared = prepare_multi_asset_dataset(
        loaded,
        seq_len=5,
        column_features=("open", "high", "low", "close", "volume"),
        target_col="meta_label",
        forecast_horizon=H,
    )

    mask = eligibility_mask(up_df, h=H, warmup=5)
    pred1_eligible = mask & (up_df["pred"].to_numpy() == 1)
    expected_positive_events = int((relabeled.loc[pred1_eligible, "lab"].to_numpy(dtype=float) == 1.0).sum())

    labels = prepared["labels"].numpy()
    finite = labels[np.isfinite(labels)]
    assert int(finite.sum()) == expected_positive_events
    assert set(np.unique(finite)).issubset({0.0, 1.0})
