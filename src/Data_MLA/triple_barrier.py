"""Triple-barrier labelling on daily OHLC bars, following Lopez de Prado (2018), chapter 3.

Vendored verbatim (EWMA_SPAN, FEE, UPPER/VERTICAL/LOWER, ewma_vol, Touches, forward_matrix,
first_true, triple_barrier) from report/triple_barrier/analysis/tb.py in the nii-thesis repo,
sha256 ac49ac5376b3b1b6375366581175909fa53d580d5a10ba8c2d043abb38726764 as of 2026-10-26 (that
file is untracked in nii-thesis's git history, so a commit hash cannot pin it). SecondaryModel's
own clone has no nii-thesis checkout, so this file does not import across repos (tb_pilot_plan.md
Phase 1, item 1) — re-sync by hand and update this hash if the source changes.

Book conventions reproduced here (Advances in Financial Machine Learning, AFML):
- sigma_t: exponentially weighted std of daily returns, span 100 (Snippet 3.1, getDailyVol).
- barriers on side-adjusted returns: upper = +pt * width, lower = -sl * width (Snippet 3.2).
- vertical barrier: h bars after t0 (Snippet 3.4).
- outcome: the first barrier touched; the label is the sign of the side-adjusted return at
  that touch (Snippet 3.5).

Two ways to look for a touch:
- "close": only bar closes, exactly like the book's code (applyPtSlOnT1 reads close prices);
- "hl": bar highs and lows, i.e. a resting take-profit / stop-loss order. If both barriers lie
  inside one bar, the order of the touches is unknown; we count the stop-loss (conservative).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

EWMA_SPAN = 100  # AFML Snippet 3.1 default
FEE = 0.002  # round-trip fee = the pipeline's rho (forecaster/forecast_features.py RHO)

UPPER, VERTICAL, LOWER = 1, 0, -1


def ewma_vol(close: pd.Series, span: int = EWMA_SPAN) -> pd.Series:
    """Daily volatility at each close (AFML Snippet 3.1 on daily bars): known at t, no look-ahead."""
    return close.pct_change().ewm(span=span).std()


@dataclass(frozen=True)
class Touches:
    """Result of a triple-barrier search, one entry per event."""

    outcome: np.ndarray  # UPPER / VERTICAL / LOWER
    bars: np.ndarray  # bars from t0 to the exit (1..h)
    ret: np.ndarray  # side-adjusted return at the exit
    ambiguous: np.ndarray  # both barriers inside the exit bar ("hl" only)


def forward_matrix(x: np.ndarray, base: np.ndarray, h: int) -> np.ndarray:
    """M[i, j-1] = x[i + j] / base[i] - 1 for j = 1..h (NaN past the end)."""
    n = len(x)
    out = np.full((n, h), np.nan)
    for j in range(1, h + 1):
        out[: n - j, j - 1] = x[j:] / base[: n - j] - 1
    return out


def first_true(mask: np.ndarray) -> np.ndarray:
    """Index (1-based) of the first True per row, or a large number if none."""
    hit = mask.any(axis=1)
    return np.where(hit, mask.argmax(axis=1) + 1, np.iinfo(np.int32).max)


def triple_barrier(
    candles: pd.DataFrame,
    t0: np.ndarray,
    width: np.ndarray,
    h: int,
    side: np.ndarray | None = None,
    pt: float = 1.0,
    sl: float = 1.0,
    mode: str = "close",
) -> Touches:
    """First touch of the three barriers for events starting at integer positions `t0`.

    width: unit barrier width per event, as a return (e.g. sigma_t * sqrt(h), or 0.10).
    side:  +1 long / -1 short per event (meta-labelling); None means long (side-less labels).
    pt, sl: multiples of `width` for the upper and lower barrier; 0 disables a barrier.
    """
    if mode not in ("close", "hl"):
        raise ValueError(f"mode must be 'close' or 'hl', got {mode!r}")
    t0 = np.asarray(t0, dtype=int)
    width = np.asarray(width, dtype=float)
    side = np.ones(len(t0)) if side is None else np.asarray(side, dtype=float)
    if not (len(width) == len(side) == len(t0)):
        raise ValueError("t0, width and side must have the same length")
    if np.any(t0 + h >= len(candles)):
        raise ValueError("every event needs h bars after t0")

    close = candles["close"].to_numpy()
    r_close = forward_matrix(close, close, h)[t0] * side[:, None]
    if mode == "close":
        best = worst = r_close
    else:
        r_high = forward_matrix(candles["high"].to_numpy(), close, h)[t0]
        r_low = forward_matrix(candles["low"].to_numpy(), close, h)[t0]
        long = side[:, None] > 0
        best = np.where(long, r_high, -r_low)
        worst = np.where(long, r_low, -r_high)

    up_level = np.where(pt > 0, pt * width, np.inf)[:, None]
    dn_level = np.where(sl > 0, -sl * width, -np.inf)[:, None]
    j_up = first_true(best >= up_level)
    j_dn = first_true(worst <= dn_level)
    j_exit = np.minimum(np.minimum(j_up, j_dn), h)

    outcome = np.full(len(t0), VERTICAL)
    outcome[(j_up <= j_exit) & (j_up < j_dn)] = UPPER
    outcome[j_dn <= np.minimum(j_up, h)] = LOWER  # same bar: stop-loss first
    ambiguous = (j_up == j_dn) & (j_up <= h)

    ret = r_close[np.arange(len(t0)), j_exit - 1]
    if mode == "hl":  # resting orders fill at the barrier
        ret = np.where(outcome == UPPER, up_level[:, 0], ret)
        ret = np.where(outcome == LOWER, dn_level[:, 0], ret)
    return Touches(outcome=outcome, bars=j_exit, ret=ret, ambiguous=ambiguous)
