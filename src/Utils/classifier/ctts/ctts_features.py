"""Extract raw close-price windows for CTTS from the cached OHLCV tensor.

The cache stores ``sub["ohlcv"]`` with shape ``(N, 5, S)`` where channel 3
is the close price.  This module slices the last ``seq_len`` bars from each
window and applies per-window MinMax scaling — exactly matching the original
CTTS ``data_preprocessing.py`` (lines 310-325).

Usage::

    from Utils.classifier.ctts.ctts_features import extract_close_windows
    X = extract_close_windows(sub, seq_len=90)   # (N, seq_len) float32
"""

import numpy as np
import torch


def extract_close_windows(sub: dict, seq_len: int = 90) -> np.ndarray:
    """Return MinMax-scaled close-price windows from a per-granularity cache slice.

    Parameters
    ----------
    sub : dict
        One granularity slice from ``multi.sub[gran]``.  Must contain an
        ``"ohlcv"`` key with shape ``(N, 5, S)`` — channel order is
        ``[open, high, low, close, volume]``.
    seq_len : int
        Number of trailing bars to keep per window.  If the stored sequence
        length ``S`` is shorter than ``seq_len``, the window is **left-padded
        with zeros** before normalisation.

    Returns
    -------
    np.ndarray, shape (N, seq_len), dtype float32
        Per-window MinMax-scaled close prices ready for ``CTTSClassifier``.
    """
    # ┏━━━━━━━━━━ Load OHLCV tensor ━━━━━━━━━━┓
    ohlcv = sub["ohlcv"]
    if isinstance(ohlcv, torch.Tensor):
        ohlcv = ohlcv.numpy()

    # ┏━━━━━━━━━━ Extract close channel (index 3) ━━━━━━━━━━┓
    close = ohlcv[:, 3, :]  # (N, S)
    N, S = close.shape

    # ┏━━━━━━━━━━ Slice / pad to seq_len ━━━━━━━━━━┓
    if S >= seq_len:
        windows = close[:, -seq_len:].copy()  # (N, seq_len)
    else:
        # Left-pad with the first available close value (avoids a zero→non-zero
        # discontinuity that would confuse MinMax into a near-zero range).
        pad_width = seq_len - S
        first_vals = close[:, :1]                                     # (N, 1)
        pad = np.repeat(first_vals, pad_width, axis=1)                # (N, pad_width)
        windows = np.concatenate([pad, close], axis=1).copy()         # (N, seq_len)

    # ┏━━━━━━━━━━ Per-window MinMax scaling (matches CTTS data_preprocessing.py) ━━━━━━━━━━┓
    w_min = windows.min(axis=1, keepdims=True)   # (N, 1)
    w_max = windows.max(axis=1, keepdims=True)   # (N, 1)
    diff = w_max - w_min
    diff[diff == 0.0] = 1.0                       # constant windows → 0 after scaling
    windows = (windows - w_min) / diff

    return windows.astype(np.float32)
