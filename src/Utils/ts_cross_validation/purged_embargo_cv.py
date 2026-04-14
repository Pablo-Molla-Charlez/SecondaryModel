from Utils.ts_cross_validation._ts_cross_validation import BaseTimeSeriesCV
import pandas as pd
import numpy as np
from typing import Iterator, Tuple, Optional, Union


class PurgedEmbargoTimeSeriesCV(BaseTimeSeriesCV):
    """
    Purged + Embargo Time Series Cross-Validation (Lopez de Prado)

    Parameters
    ----------
    n_splits : int
    t1 : pd.Series
        Series of label end times (index aligned with X)
    embargo_pct : float
        Fraction of dataset to embargo after each test split
    random_state : int or None
    """

    def __init__(
        self,
        n_splits: int,
        t1: pd.Series,
        embargo_pct: float = 0.0,
        random_state: Optional[int] = None
    ):
        super().__init__(n_splits=n_splits, random_state=random_state)

        if not isinstance(t1, pd.Series):
            raise TypeError("t1 must be a pandas Series")

        if not 0.0 <= embargo_pct < 1.0:
            raise ValueError("embargo_pct must be in [0, 1)")

        self.t1 = t1
        self.embargo_pct = embargo_pct
    
    def split(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Optional[np.ndarray] = None,
        groups=None
    ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        
        # --- Handle input types ---
        if isinstance(X, pd.DataFrame):
            index = X.index
            n_samples = len(X)
        elif isinstance(X, np.ndarray):
            n_samples = X.shape[0]
            # create artificial index to align with t1
            if not hasattr(self.t1, "index"):
                raise ValueError("t1 must have an index when X is numpy array")
            index = self.t1.index
            if len(index) != n_samples:
                raise ValueError("X and t1 must have the same length")
        else:
            raise TypeError("X must be a pandas DataFrame or numpy array")
        
        # --- Check alignment ---
        if not index.equals(self.t1.index):
            raise ValueError("X and t1 must have the same index")
        
        indices = np.arange(n_samples)
        test_ranges = np.array_split(indices, self.n_splits)
        embargo_size = int(n_samples * self.embargo_pct)
        
        for test_idx in test_ranges:
            
            # --- Skip empty test splits (can happen with small datasets) ---
            if len(test_idx) == 0:
                continue
            
            test_start = test_idx[0]
            test_end = test_idx[-1]
            
            test_times = index[test_idx]
            
            train_mask = np.ones(n_samples, dtype=bool)
            
            # --- Remove test samples ---
            train_mask[test_idx] = False
            
            # --- PURGING ---
            test_start_time = test_times[0]
            test_end_time = test_times[-1]
            
            overlap = (self.t1 >= test_start_time) & (index <= test_end_time)
            train_mask[np.asarray(overlap)] = False
            
            # --- EMBARGO ---
            if embargo_size > 0:
                embargo_start = test_end + 1
                embargo_end = min(n_samples, embargo_start + embargo_size)
                train_mask[embargo_start:embargo_end] = False
            
            train_idx = indices[train_mask]
            
            # --- Ensure non-empty splits ---
            if len(train_idx) == 0 or len(test_idx) == 0:
                continue  # skip invalid split
            
            yield train_idx, test_idx

    @property
    def name(self):
        return "PurgedEmbargoCV"