from Utils.ts_cross_validation._ts_cross_validation import BaseTimeSeriesCV
import pandas as pd
import numpy as np
from itertools import combinations
from typing import Iterator, Tuple, Optional, Union


class CombinatorialPurgedCV(BaseTimeSeriesCV):
    """
    Combinatorial Purged Cross-Validation (CPCV)

    Parameters
    ----------
    n_splits : int
        Number of groups to divide the data into (N)
    n_test_splits : int
        Number of groups used for testing (k)
    t1 : pd.Series
        Label end times (index aligned with X)
    embargo_pct : float
    random_state : int or None
    """

    def __init__(
        self,
        n_splits: int,
        n_test_splits: int,
        t1: pd.Series,
        embargo_pct: float = 0.0,
        random_state: Optional[int] = None
    ):
        super().__init__(n_splits=n_splits, random_state=random_state)

        if not isinstance(n_test_splits, int) or n_test_splits < 1:
            raise ValueError("n_test_splits must be >= 1")

        if n_test_splits >= n_splits:
            raise ValueError("n_test_splits must be < n_splits")

        if not isinstance(t1, pd.Series):
            raise TypeError("t1 must be a pandas Series")

        if not 0.0 <= embargo_pct < 1.0:
            raise ValueError("embargo_pct must be in [0, 1)")

        self.n_test_splits = n_test_splits
        self.t1 = t1
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
        
        if not self.t1.index.equals(time_index):
            raise ValueError("t1 index must align with X index")
        
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
                overlap = (self.t1 >= block_start_time) & (time_index <= block_end_time)
                train_mask[overlap.values] = False
            
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
    
    # def split(
    #     self,
    #     X: Union[np.ndarray, pd.DataFrame],
    #     y: Optional[np.ndarray] = None,
    #     groups=None
    # ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    #     """
    #     Generate train/test splits as indices.
    #
    #     Yields
    #     ------
    #     train_idx : np.ndarray
    #     test_idx : np.ndarray
    #     """
    #     for test_blocks in combinations(range(self.n_blocks), self.k_test):
    #         test_set = set(test_blocks)
    #         train_blocks = [b for b in range(self.n_blocks) if b not in test_set]
    #
    #         idx_test = np.where(np.isin(self.block_ids, list(test_blocks)))[0]
    #         idx_train_raw = np.where(np.isin(self.block_ids, train_blocks))[0]
    #         if len(idx_test) == 0 or len(idx_train_raw) == 0:
    #             continue
    #
    #         test_boundaries = [self.boundaries[tb] for tb in test_blocks]
    #         purged_mask = np.zeros(len(self.dates_arr), dtype=bool)
    #         for tb_start, tb_end in test_boundaries:
    #             for i in idx_train_raw:
    #                 t = self.dates_arr[i]
    #                 if (tb_start - self.purge_td) <= t < tb_start:
    #                     purged_mask[i] = True
    #                 elif tb_end < t <= (tb_end + self.purge_td):
    #                     purged_mask[i] = True
    #
    #         idx_train = np.array([i for i in idx_train_raw if not purged_mask[i]])
    #
    #         # NOTE prevent empty splits!
    #         if len(idx_train) == 0 or len(idx_test) == 0:
    #             continue
    #
    #         yield idx_train, idx_test