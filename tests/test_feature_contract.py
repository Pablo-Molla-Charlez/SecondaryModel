"""The M2 engineered-feature vector contract that this repo's trained bundles expect.

compute_window_features returns one value per ENG_FEATURE_NAMES entry, in that order,
and the ATR/MACD features are the raw indicator values in price units (not divided by
the last close, unlike the newer Secondary-Model feature version 2).
"""

import numpy as np
import pytest

from Utils.data.data import ENG_FEATURE_NAMES, compute_window_features

T = 64
RAW = {"atr_14": 1234.5, "macd_histogram": -87.25, "atr_w30": 2468.0, "r_vol_30_ann": 0.61, "atr_norm": 0.0123}


@pytest.fixture(scope="module")
def window() -> tuple[np.ndarray, np.ndarray, list[str]]:
    rng = np.random.default_rng(0)
    close = 100_000 * np.exp(np.cumsum(rng.normal(0, 0.01, T)))
    high, low = close * 1.01, close * 0.99
    ohlcv = np.column_stack([close, high, low, close, rng.uniform(1, 10, T)])
    cols = list(RAW)
    extrinsic = np.tile(np.array([RAW[c] for c in cols]), (T, 1))
    return ohlcv, extrinsic, cols


def test_width_matches_feature_names(window):
    out = compute_window_features(*window)
    assert out.shape == (len(ENG_FEATURE_NAMES),)
    assert len(set(ENG_FEATURE_NAMES)) == len(ENG_FEATURE_NAMES)
    assert np.isfinite(out).all()


def test_r_vol_30_ann_is_in_the_vector(window):
    assert "r_vol_30_ann_last" in ENG_FEATURE_NAMES
    out = compute_window_features(*window)
    assert out[ENG_FEATURE_NAMES.index("r_vol_30_ann_last")] == pytest.approx(RAW["r_vol_30_ann"], rel=1e-6)


@pytest.mark.parametrize(
    ("feature", "column"),
    [("atr_last", "atr_14"), ("macd_last", "macd_histogram"), ("atr_w30_last", "atr_w30"), ("atr_norm_last", "atr_norm")],
)
def test_indicator_features_are_raw_values(window, feature, column):
    ohlcv, *_ = window
    out = compute_window_features(*window)
    value = out[ENG_FEATURE_NAMES.index(feature)]
    assert value == pytest.approx(RAW[column], rel=1e-6)
    assert value != pytest.approx(RAW[column] / ohlcv[-1, 3], rel=1e-3)
