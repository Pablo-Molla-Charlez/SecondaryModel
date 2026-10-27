"""Hand-computed touch scenarios for the vendored triple_barrier() (tb_pilot_plan.md Phase 1).

Fixtures use mode="hl" (resting orders on bar highs/lows) throughout, h=3 unless noted,
width=0.05 (a +-5% barrier), pt=sl=1.0. Each candles DataFrame only needs close/high/low.
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from Data_MLA.triple_barrier import LOWER, UPPER, VERTICAL, triple_barrier

WIDTH = np.array([0.05])


def _candles(close: list[float], high: list[float], low: list[float]) -> pd.DataFrame:
    return pd.DataFrame({"close": close, "high": high, "low": low})


def test_profit_touch_fills_at_the_upper_barrier():
    candles = _candles(
        close=[100.0, 101.0, 105.0, 106.0],
        high=[100.0, 102.0, 106.0, 107.0],
        low=[100.0, 99.0, 100.0, 101.0],
    )
    touches = triple_barrier(candles, t0=np.array([0]), width=WIDTH, h=3, side=np.array([1.0]), mode="hl")
    assert touches.outcome[0] == UPPER
    assert touches.bars[0] == 2
    assert touches.ret[0] == pytest.approx(0.05)
    assert not touches.ambiguous[0]


def test_stop_touch_fills_at_the_lower_barrier():
    candles = _candles(
        close=[100.0, 99.0, 95.0, 94.0],
        high=[100.0, 101.0, 102.0, 103.0],
        low=[100.0, 98.0, 94.0, 93.0],
    )
    touches = triple_barrier(candles, t0=np.array([0]), width=WIDTH, h=3, side=np.array([1.0]), mode="hl")
    assert touches.outcome[0] == LOWER
    assert touches.bars[0] == 2
    assert touches.ret[0] == pytest.approx(-0.05)
    assert not touches.ambiguous[0]


def test_expiry_exits_at_the_vertical_barrier_on_the_actual_close():
    candles = _candles(
        close=[100.0, 100.5, 101.0, 102.0],
        high=[100.0, 101.0, 102.0, 103.0],
        low=[100.0, 99.0, 98.0, 97.0],
    )
    touches = triple_barrier(candles, t0=np.array([0]), width=WIDTH, h=3, side=np.array([1.0]), mode="hl")
    assert touches.outcome[0] == VERTICAL
    assert touches.bars[0] == 3
    assert touches.ret[0] == pytest.approx(0.02)


def test_same_bar_double_touch_counts_as_a_stop():
    candles = _candles(close=[100.0, 100.0], high=[100.0, 108.0], low=[100.0, 92.0])
    touches = triple_barrier(candles, t0=np.array([0]), width=WIDTH, h=1, side=np.array([1.0]), mode="hl")
    assert touches.outcome[0] == LOWER
    assert touches.ambiguous[0]
    assert touches.bars[0] == 1
    assert touches.ret[0] == pytest.approx(-0.05)


def test_side_symmetry_flips_the_outcome_on_the_same_candles():
    # Same candles as the stop-touch case: a drop that stops out a long is a profit for a short.
    candles = _candles(
        close=[100.0, 99.0, 95.0, 94.0],
        high=[100.0, 101.0, 102.0, 103.0],
        low=[100.0, 98.0, 94.0, 93.0],
    )
    touches = triple_barrier(candles, t0=np.array([0]), width=WIDTH, h=3, side=np.array([-1.0]), mode="hl")
    assert touches.outcome[0] == UPPER
    assert touches.bars[0] == 2
    assert touches.ret[0] == pytest.approx(0.05)


def test_mode_must_be_close_or_hl():
    candles = _candles(close=[100.0, 101.0], high=[100.0, 101.0], low=[100.0, 101.0])
    with pytest.raises(ValueError):
        triple_barrier(candles, t0=np.array([0]), width=WIDTH, h=1, mode="bogus")


def test_every_event_needs_h_bars_after_t0():
    candles = _candles(close=[100.0, 101.0], high=[100.0, 101.0], low=[100.0, 101.0])
    with pytest.raises(ValueError):
        triple_barrier(candles, t0=np.array([0]), width=WIDTH, h=5, mode="hl")


@pytest.mark.skipif(
    not os.environ.get("NII_TB_REF"), reason="set NII_TB_REF=1 to check parity against report/triple_barrier/analysis/tb.py"
)
def test_parity_against_the_report_source():
    ref_path = Path(__file__).resolve().parents[3] / "report/triple_barrier/analysis/tb.py"
    spec = importlib.util.spec_from_file_location("tb_ref", ref_path)
    assert spec is not None and spec.loader is not None
    tb_ref = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tb_ref)

    rng = np.random.default_rng(0)
    n = 60
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    high = close * (1 + rng.uniform(0, 0.02, n))
    low = close * (1 - rng.uniform(0, 0.02, n))
    candles = pd.DataFrame({"close": close, "high": high, "low": low})
    h = 7
    t0 = np.arange(n - h)
    width = np.full(len(t0), 0.03)
    side = rng.choice([-1.0, 1.0], size=len(t0))

    ours = triple_barrier(candles, t0, width, h, side=side, mode="hl")
    theirs = tb_ref.triple_barrier(candles, t0, width, h, side=side, mode="hl")

    np.testing.assert_array_equal(ours.outcome, theirs.outcome)
    np.testing.assert_array_equal(ours.bars, theirs.bars)
    np.testing.assert_allclose(ours.ret, theirs.ret)
    np.testing.assert_array_equal(ours.ambiguous, theirs.ambiguous)
