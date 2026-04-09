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
            # fallback: assume integer time
            time_index = pd.RangeIndex(start=0, stop=n_samples)

        if not self.t1.index.equals(time_index):
            raise ValueError("t1 index must align with X index")

        groups = np.array_split(indices, self.n_splits)

        embargo_size = int(n_samples * self.embargo_pct)

        for test_group_ids in combinations(range(self.n_splits), self.n_test_splits):

            test_idx = np.concatenate([groups[i] for i in test_group_ids])
            test_idx = np.sort(test_idx)

            test_times = time_index[test_idx]

            train_mask = np.ones(n_samples, dtype=bool)

            # --- remove test samples ---
            train_mask[test_idx] = False

            test_start_time = test_times[0]
            test_end_time = test_times[-1]

            overlap = (self.t1 >= test_start_time) & (time_index <= test_end_time)
            train_mask[overlap.values] = False

            if embargo_size > 0:
                embargo_mask = np.zeros(n_samples, dtype=bool)

                for idx in test_idx:
                    embargo_start = idx + 1
                    embargo_end = min(n_samples, embargo_start + embargo_size)
                    embargo_mask[embargo_start:embargo_end] = True

                train_mask[embargo_mask] = False

            train_idx = indices[train_mask]

            if len(train_idx) == 0 or len(test_idx) == 0:
                continue

            yield train_idx, test_idx