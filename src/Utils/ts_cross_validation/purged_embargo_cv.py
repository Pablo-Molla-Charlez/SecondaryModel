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
        embargo_pct: float = 0.0,
        random_state: Optional[int] = None
    ):
        super().__init__(n_splits=n_splits, random_state=random_state)

        if not 0.0 <= embargo_pct < 1.0:
            raise ValueError("embargo_pct must be in [0, 1)")

        self.embargo_pct = embargo_pct
    
    def split(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Optional[np.ndarray] = None,
        groups=None
    ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        
        n_samples = len(X)
        
        # --- Handle input types ---
        if isinstance(X, pd.DataFrame):
            time_index = X.index
        else:
            time_index = pd.RangeIndex(start=0, stop=n_samples)
        
        
        indices = np.arange(n_samples)
        test_ranges = np.array_split(indices, self.n_splits)
        embargo_size = int(n_samples * self.embargo_pct)
        
        for test_idx in test_ranges:
            
            # --- Skip empty test splits (can happen with small datasets) ---
            if len(test_idx) == 0:
                continue
            
            test_start = test_idx[0]
            test_end = test_idx[-1]
            
            test_times = time_index[test_idx]
            
            train_mask = np.ones(n_samples, dtype=bool)
            
            # --- Remove test samples ---
            train_mask[test_idx] = False
            
            # --- PURGING ---
            test_start_time = test_times[0]
            test_end_time = test_times[-1]
            
            overlap = (time_index >= test_start_time) & (time_index <= test_end_time)
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