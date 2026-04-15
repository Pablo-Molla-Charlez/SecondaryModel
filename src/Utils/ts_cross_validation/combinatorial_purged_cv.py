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
from __future__ import annotations
import pandas as pd
import numpy as np
import math
from math import comb



from itertools import combinations, permutations
from collections import defaultdict
from typing import Iterator, Optional, Tuple, Union

import numpy as np
import pandas as pd

from Utils.ts_cross_validation._ts_cross_validation import BaseTimeSeriesCV


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


class CombinatorialPurgedCV(BaseTimeSeriesCV):
    """
    Combinatorial Purged Cross-Validation (CPCV)

    Parameters
    ----------
    n_splits : int
        Number of groups to divide the data into (N)
    n_test_splits : int
        Number of groups used for testing (k)
    embargo_pct : float
    random_state : int or None
    """
    
    def __init__(
        self,
        n_splits: int,
        n_test_splits: int,
        embargo_pct: float = 0.0,
        random_state: Optional[int] = None
    ):
        super().__init__(n_splits=n_splits, random_state=random_state)
        
        if not isinstance(n_test_splits, int) or n_test_splits < 1:
            raise ValueError("n_test_splits must be >= 1")
        
        if n_test_splits >= n_splits:
            raise ValueError("n_test_splits must be < n_splits")
        
        if not 0.0 <= embargo_pct < 1.0:
            raise ValueError("embargo_pct must be in [0, 1)")
        
        self.n_test_splits = n_test_splits
        self.embargo_pct = embargo_pct
    
    def split(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Optional[np.ndarray] = None,
        groups=None
    ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        
        n_samples = len(X)
        indices = np.arange(n_samples)
        
        if isinstance(X, pd.DataFrame):
            time_index = X.index
        else:
            time_index = pd.RangeIndex(start=0, stop=n_samples)
        
        groups = np.array_split(indices, self.n_splits)
        embargo_size = int(n_samples * self.embargo_pct)
        
        for test_group_ids in combinations(range(self.n_splits), self.n_test_splits):
            
            test_group_ids = sorted(test_group_ids)
            test_idx = np.sort(np.concatenate([groups[i] for i in test_group_ids]))
            test_times = time_index[test_idx]
            
            train_mask = np.ones(n_samples, dtype=bool)
            train_mask[test_idx] = False
            
            # Purge: per test block, remove training samples whose t1 overlaps that block
            for gid in test_group_ids:
                block_start_time = time_index[groups[gid][0]]
                block_end_time = time_index[groups[gid][-1]]
                overlap = (time_index >= block_start_time) & (time_index <= block_end_time)
                train_mask[overlap] = False
            
            # Embargo: apply after each test block's trailing edge
            if embargo_size > 0:
                embargo_mask = np.zeros(n_samples, dtype=bool)
                for gid in test_group_ids:
                    embargo_start = groups[gid][-1] + 1
                    embargo_end = min(n_samples, embargo_start + embargo_size)
                    embargo_mask[embargo_start:embargo_end] = True
                train_mask[embargo_mask] = False
            
            train_idx = indices[train_mask]
            
            if len(train_idx) == 0 or len(test_idx) == 0:
                continue
            
            yield train_idx, test_idx
    
    @property
    def name(self):
        return "CombinatorialPurgedEmbargoCV"
    
    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        return comb(self.n_splits, self.n_test_splits)

    # ┏━━━━━━━━━━ Reconstruct chronological paths (mode=datetime only) ━━━━━━━━━━┓
    def reconstruct_paths(self):
        """Return C(N-1, k-1) chronological paths (mirrors ``edge.py``).

        Only available for ``mode="datetime"``.
        """
        if self.mode != "datetime":
            raise ValueError("reconstruct_paths() is only available for mode='datetime'")
        splits = self._get_datetime_splits()
        return _reconstruct_paths(splits, self.n_splits, self.n_test_splits)

    # ┏━━━━━━━━━━ Convenience accessors ━━━━━━━━━━┓
    @property
    def boundaries(self):
        return self._boundaries

    @property
    def block_ids(self):
        return self._block_ids

    def get_datetime_splits(self):
        """Return the full splits list with test_blocks/idx_train/idx_test/idx_purged
        (mirrors ``edge.py::_generate_cpcv_splits`` output exactly).
        """
        if self.mode != "datetime":
            raise ValueError("get_datetime_splits() is only available for mode='datetime'")
        return self._get_datetime_splits()
    
    def get_evaluation_path_ids(self):
        splits = list(combinations(range(self.n_splits), self.n_test_splits))
        """Reconstruct C(N-1, k-1) chronological paths from CPCV splits."""
        # ┏━━━━━━━━━━ Initialize variables ━━━━━━━━━━┓
        n_paths = math.comb(self.n_splits - 1, self.n_test_splits - 1)
        block_to_splits = {b: [] for b in range(self.n_splits)}
        
        # ┏━━━━━━━━━━ Map each block to the splits that test it ━━━━━━━━━━┓
        for si, sp in enumerate(splits):
            for b in sp:
                block_to_splits[b].append(si)
        
        # ┏━━━━━━━━━━ Create paths by picking the p-th split for every block ━━━━━━━━━━┓
        paths = []
        for p in range(n_paths):
            path_entries = []
            for b in range(self.n_splits):
                split_idx = block_to_splits[b][p]
                path_entries.append({"block": b, "split_idx": split_idx})
            paths.append(path_entries)
        
        return paths
