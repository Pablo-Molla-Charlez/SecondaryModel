from Utils.ts_cross_validation._ts_cross_validation import BaseTimeSeriesCV
import pandas as pd
import numpy as np
from typing import Iterator, Tuple, Optional


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
        X: pd.DataFrame,
        y: np.ndarray
    ) -> Iterator[Tuple[pd.DataFrame, np.ndarray, pd.DataFrame, np.ndarray]]:

        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame")

        if not isinstance(y, np.ndarray):
            raise TypeError("y must be a numpy array")

        if len(X) != len(y):
            raise ValueError("X and y must have same length")

        if not X.index.equals(self.t1.index):
            raise ValueError("X and t1 must have the same index")

        n_samples = len(X)
        indices = np.arange(n_samples)

        # Split indices into contiguous folds
        test_ranges = np.array_split(indices, self.n_splits)

        embargo_size = int(n_samples * self.embargo_pct)

        for test_idx in test_ranges:
            test_start = test_idx[0]
            test_end = test_idx[-1]

            test_times = X.index[test_idx]

            # --- PURGING ---
            train_mask = np.ones(n_samples, dtype=bool)

            # Remove test indices
            train_mask[test_idx] = False

            # Remove overlapping labels
            test_start_time = test_times[0]
            test_end_time = test_times[-1]

            overlap = (self.t1 >= test_start_time) & (X.index <= test_end_time)
            train_mask[overlap.values] = False

            # --- EMBARGO ---
            if embargo_size > 0:
                embargo_start = test_end + 1
                embargo_end = min(n_samples, embargo_start + embargo_size)

                train_mask[embargo_start:embargo_end] = False

            train_idx = indices[train_mask]

            yield (
                X.iloc[train_idx],
                y[train_idx],
                X.iloc[test_idx],
                y[test_idx],
            )