"""Combinatorial Purged Cross-Validation.

Two modes are supported:

- ``mode="datetime"`` (default) — mirrors the production semantics used by
  ``Utils/edge.py`` (datetime-based block boundaries, time-based purge window
  ``purge_td = horizon * bar_width``). This is the semantics the paper's edge
  convergence protocol relies on.
- ``mode="index"`` — legacy sklearn-style behavior that builds blocks with
  ``np.array_split(indices, n_splits)`` and uses a count-based embargo (fraction
  of rows). Preserved unchanged for existing call sites (e.g. the
  Interpretability scripts).
"""
import pandas as pd
import numpy as np
import math
from math import comb

from __future__ import annotations

from itertools import combinations
from typing import Iterator, Optional, Tuple, Union

import numpy as np
import pandas as pd

# ┏━━━━━━━━━━ Granularity → Timedelta ━━━━━━━━━━┓
def _gran_to_timedelta(granularity: str) -> pd.Timedelta:
    """Convert a granularity string (e.g. ``"4h"``, ``"30m"``, ``"1d"``) to a
    :class:`pandas.Timedelta`. Copied verbatim from ``Utils.edge``.
    """
    if granularity.endswith("m"):
        return pd.Timedelta(minutes=int(granularity[:-1]))
    elif granularity.endswith("h"):
        return pd.Timedelta(hours=int(granularity[:-1]))
    elif granularity.endswith("d"):
        return pd.Timedelta(days=int(granularity[:-1]))
    return pd.Timedelta(days=1)


# ┏━━━━━━━━━━ Datetime block helpers (ported verbatim from edge.py) ━━━━━━━━━━┓
def _build_datetime_blocks(dates, n_blocks: int):
    """Divide dates into ``n_blocks`` by datetime boundaries (not row count)."""
    ts_sorted = sorted(set(dates))
    t_min, t_max = ts_sorted[0], ts_sorted[-1]
    block_duration = (t_max - t_min) / n_blocks
    return [(t_min + b * block_duration, t_min + (b + 1) * block_duration)
            for b in range(n_blocks)]


def _assign_blocks(dates_arr, boundaries) -> np.ndarray:
    """Assign each index to a block number based on its datetime."""
    n = len(dates_arr)
    n_blocks = len(boundaries)
    block_ids = np.full(n, -1, dtype=int)
    for i in range(n):
        t = dates_arr[i]
        for b in range(n_blocks):
            b_start, b_end = boundaries[b]
            if b == n_blocks - 1:
                if b_start <= t <= b_end:
                    block_ids[i] = b
                    break
            else:
                if b_start <= t < b_end:
                    block_ids[i] = b
                    break
    return block_ids


def _generate_cpcv_splits(n_blocks, k_test, block_ids, dates_arr, purge_td, boundaries):
    """Generate all C(n_blocks, k_test) train/test splits with purge."""
    splits = []
    for test_blocks in combinations(range(n_blocks), k_test):
        test_set = set(test_blocks)
        train_blocks = [b for b in range(n_blocks) if b not in test_set]

        idx_test = np.where(np.isin(block_ids, list(test_blocks)))[0]
        idx_train_raw = np.where(np.isin(block_ids, train_blocks))[0]
        if len(idx_test) == 0 or len(idx_train_raw) == 0:
            continue

        test_boundaries = [boundaries[tb] for tb in test_blocks]
        purged_mask = np.zeros(len(dates_arr), dtype=bool)
        for tb_start, tb_end in test_boundaries:
            for i in idx_train_raw:
                t = dates_arr[i]
                if (tb_start - purge_td) <= t < tb_start:
                    purged_mask[i] = True
                elif tb_end < t <= (tb_end + purge_td):
                    purged_mask[i] = True

        idx_purged = np.where(purged_mask)[0]
        idx_train = np.array([i for i in idx_train_raw if not purged_mask[i]])

        splits.append({
            "test_blocks": test_blocks,
            "idx_train": idx_train,
            "idx_test": idx_test,
            "idx_purged": idx_purged,
        })
    return splits


def _reconstruct_paths(splits, n_blocks, k_test):
    """Reconstruct C(N-1, k-1) chronological paths from CPCV splits."""
    n_paths = math.comb(n_blocks - 1, k_test - 1)
    block_to_splits = {b: [] for b in range(n_blocks)}
    for si, sp in enumerate(splits):
        for b in sp["test_blocks"]:
            block_to_splits[b].append(si)

    paths = []
    for p in range(n_paths):
        path_entries = []
        for b in range(n_blocks):
            split_idx = block_to_splits[b][p]
            path_entries.append({"block": b, "split_idx": split_idx})
        paths.append(path_entries)
    return paths


# class CombinatorialPurgedCV(BaseTimeSeriesCV):
#     """Combinatorial Purged Cross-Validation.
#
#     Parameters
#     ----------
#     n_splits : int
#         Number of blocks (N).
#     n_test_splits : int
#         Number of blocks used for testing (k).
#     mode : {"datetime", "index"}, default "datetime"
#         - ``"datetime"`` — block boundaries are determined by datetime (equal
#           calendar-time intervals) and the purge window is time-based
#           (``purge_td`` or ``horizon * _gran_to_timedelta(granularity)``).
#           Mirrors the production semantics of ``Utils/edge.py``.
#         - ``"index"`` — legacy behavior: blocks from ``np.array_split`` and
#           count-based embargo. Requires ``t1`` and accepts ``embargo_pct``.
#     dates : pd.Series, optional
#         Datetime series aligned with ``X``. Required for ``mode="datetime"``.
#     purge_td : pd.Timedelta, optional
#         Explicit time-based purge window for ``mode="datetime"``. If not
#         provided, you must provide both ``horizon`` and ``granularity``.
#     horizon : int, optional
#         Forecast horizon (number of bars). Used with ``granularity`` to compute
#         ``purge_td`` when it is not provided explicitly.
#     granularity : str, optional
#         Bar granularity string (e.g. ``"4h"``, ``"30m"``, ``"1d"``).
#     t1 : pd.Series, optional
#         Label end times — required for ``mode="index"``.
#     embargo_pct : float, default 0.0
#         Count-based embargo as a fraction of rows (``mode="index"`` only).
#     random_state : int, optional
#     """
#
#     def __init__(
#         self,
#         n_splits: int,
#         n_test_splits: int,
#         mode: str = "datetime",
#         # mode="datetime" params:
#         dates: Optional[pd.Series] = None,
#         purge_td: Optional[pd.Timedelta] = None,
#         horizon: Optional[int] = None,
#         granularity: Optional[str] = None,
#         # mode="index" params (legacy):
#         t1: Optional[pd.Series] = None,
#         embargo_pct: float = 0.0,
#         random_state: Optional[int] = None,
#     ):
#         super().__init__(n_splits=n_splits, random_state=random_state)
#
#         if not isinstance(n_test_splits, int) or n_test_splits < 1:
#             raise ValueError("n_test_splits must be >= 1")
#         if n_test_splits >= n_splits:
#             raise ValueError("n_test_splits must be < n_splits")
#
#         if not 0.0 <= embargo_pct < 1.0:
#             raise ValueError("embargo_pct must be in [0, 1)")
#
#         if mode not in ("datetime", "index"):
#             raise ValueError(f"mode must be 'datetime' or 'index', got {mode!r}")
#
#         self.mode = mode
#         self.n_test_splits = n_test_splits
#         self.embargo_pct = embargo_pct
#         # store for potential introspection
#         self.n_blocks = n_splits
#         self.k_test = n_test_splits
#
#         if mode == "datetime":
#             if dates is None:
#                 raise ValueError("mode='datetime' requires dates=")
#             # Accept Series, Index, or list-like of timestamps
#             if isinstance(dates, pd.Series):
#                 dates_list = [pd.Timestamp(d) for d in dates.values]
#             else:
#                 dates_list = [pd.Timestamp(d) for d in dates]
#
#             if purge_td is None:
#                 if horizon is None or granularity is None:
#                     raise ValueError(
#                         "mode='datetime' requires either purge_td, or both "
#                         "horizon and granularity"
#                     )
#                 purge_td = _gran_to_timedelta(granularity) * int(horizon)
#
#             if t1 is not None or embargo_pct:
#                 # Not fatal — but warn via a noisy error to avoid silent misuse.
#                 raise ValueError(
#                     "mode='datetime' does not accept t1/embargo_pct — these "
#                     "are mode='index' parameters"
#                 )
#
#             self._dates_list = dates_list
#             self.purge_td = purge_td
#             self.horizon = horizon
#             self.granularity = granularity
#
#             # Precompute block structure (same as edge.py)
#             self._boundaries = _build_datetime_blocks(dates_list, n_splits)
#             self._block_ids = _assign_blocks(dates_list, self._boundaries)
#             self._splits_cache: Optional[list] = None
#
#             # Unused in this mode
#             self.t1 = None
#             self.embargo_pct = 0.0
#
#         else:  # mode == "index"
#             if t1 is None:
#                 raise ValueError("mode='index' requires t1=")
#             if not isinstance(t1, pd.Series):
#                 raise TypeError("t1 must be a pandas Series")
#             if not 0.0 <= embargo_pct < 1.0:
#                 raise ValueError("embargo_pct must be in [0, 1)")
#
#             self.t1 = t1
#             self.embargo_pct = embargo_pct
#
#             # Unused in this mode
#             self._dates_list = None
#             self.purge_td = None
#             self.horizon = None
#             self.granularity = None
#             self._boundaries = None
#             self._block_ids = None
#             self._splits_cache = None
#
#     # ┏━━━━━━━━━━ Public split ━━━━━━━━━━┓
#     def split(
#         self,
#         X: Union[np.ndarray, pd.DataFrame, None] = None,
#         y: Optional[np.ndarray] = None,
#         groups=None,
#     ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
#         if self.mode == "datetime":
#             yield from self._split_datetime(X)
#         else:
#             yield from self._split_index(X)
#
#     # ┏━━━━━━━━━━ Datetime split ━━━━━━━━━━┓
#     def _split_datetime(self, X) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
#         splits = self._get_datetime_splits()
#         for sp in splits:
#             if len(sp["idx_train"]) == 0 or len(sp["idx_test"]) == 0:
#                 continue
#             yield sp["idx_train"], sp["idx_test"]
#
#     def _get_datetime_splits(self):
#         if self._splits_cache is None:
#             self._splits_cache = _generate_cpcv_splits(
#                 self.n_splits,
#                 self.n_test_splits,
#                 self._block_ids,
#                 self._dates_list,
#                 self.purge_td,
#                 self._boundaries,
#             )
#         return self._splits_cache
#
#     # ┏━━━━━━━━━━ Legacy index split (teammate's original logic) ━━━━━━━━━━┓
#     def _split_index(self, X) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
#         if X is None:
#             raise ValueError("mode='index' requires X= passed to split()")
#
#         n_samples = len(X)
#         indices = np.arange(n_samples)
#
#         if isinstance(X, pd.DataFrame):
#             time_index = X.index
#         else:
#             time_index = pd.RangeIndex(start=0, stop=n_samples)
#
#         if not self.t1.index.equals(time_index):
#             raise ValueError("t1 index must align with X index")
#
#         groups = np.array_split(indices, self.n_splits)
#         embargo_size = int(n_samples * self.embargo_pct)
#
#         for test_group_ids in combinations(range(self.n_splits), self.n_test_splits):
#             test_group_ids = sorted(test_group_ids)
#             test_idx = np.sort(np.concatenate([groups[i] for i in test_group_ids]))
#
#             train_mask = np.ones(n_samples, dtype=bool)
#             train_mask[test_idx] = False
#
#             for gid in test_group_ids:
#                 block_start_time = time_index[groups[gid][0]]
#                 block_end_time = time_index[groups[gid][-1]]
#                 overlap = (time_index >= block_start_time) & (time_index <= block_end_time)
#                 train_mask[overlap] = False
#
#             if embargo_size > 0:
#                 embargo_mask = np.zeros(n_samples, dtype=bool)
#                 for gid in test_group_ids:
#                     embargo_start = groups[gid][-1] + 1
#                     embargo_end = min(n_samples, embargo_start + embargo_size)
#                     embargo_mask[embargo_start:embargo_end] = True
#                 train_mask[embargo_mask] = False
#
#             train_idx = indices[train_mask]
#
#             if len(train_idx) == 0 or len(test_idx) == 0:
#                 continue
#
#             yield train_idx, test_idx
#
#     @property
#     def name(self):
#         return "CombinatorialPurgedEmbargoCV"
#
#     def get_n_splits(self, X=None, y=None, groups=None) -> int:
#         return comb(self.n_splits, self.n_test_splits)
#
#     # ┏━━━━━━━━━━ Reconstruct chronological paths (mode=datetime only) ━━━━━━━━━━┓
#     def reconstruct_paths(self):
#         """Return C(N-1, k-1) chronological paths (mirrors ``edge.py``).
#
#         Only available for ``mode="datetime"``.
#         """
#         if self.mode != "datetime":
#             raise ValueError("reconstruct_paths() is only available for mode='datetime'")
#         splits = self._get_datetime_splits()
#         return _reconstruct_paths(splits, self.n_splits, self.n_test_splits)
#
#     # ┏━━━━━━━━━━ Convenience accessors ━━━━━━━━━━┓
#     @property
#     def boundaries(self):
#         return self._boundaries
#
#     @property
#     def block_ids(self):
#         return self._block_ids
#
#     def get_datetime_splits(self):
#         """Return the full splits list with test_blocks/idx_train/idx_test/idx_purged
#         (mirrors ``edge.py::_generate_cpcv_splits`` output exactly).
#         """
#         if self.mode != "datetime":
#             raise ValueError("get_datetime_splits() is only available for mode='datetime'")
#         return self._get_datetime_splits()
