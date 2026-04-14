from Utils.ts_cross_validation._ts_cross_validation import BaseTimeSeriesCV
import pandas as pd
import numpy as np
from math import comb
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
