from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from typing import Iterator, Tuple, Optional, Union


class BaseTimeSeriesCV(ABC):
    """
    Abstract base class for temporal cross-validation.

    Compatible with sklearn API + provides convenience helpers.
    """

    def __init__(
        self,
        n_splits: int,
        random_state: Optional[int] = None
    ):
        if not isinstance(n_splits, int) or n_splits < 1:
            raise ValueError("n_splits must be a positive integer")

        self.n_splits = n_splits
        self.random_state = random_state
        self._rng = np.random.default_rng(random_state)

    @abstractmethod
    def split(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Optional[np.ndarray] = None,
        groups=None
    ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate train/test splits as indices.

        Yields
        ------
        train_idx : np.ndarray
        test_idx : np.ndarray
        """
        pass

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        return self.n_splits

    def split_data(
        self,
        X: pd.DataFrame,
        y: np.ndarray
    ) -> Iterator[Tuple[pd.DataFrame, np.ndarray, pd.DataFrame, np.ndarray]]:
        """
        Generate train/test splits returning actual data.

        This wraps the sklearn-compatible split().
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame")

        if not isinstance(y, np.ndarray):
            raise TypeError("y must be a numpy array")

        if len(X) != len(y):
            raise ValueError("X and y must have same length")

        for train_idx, test_idx in self.split(X, y):
            yield (
                X.iloc[train_idx],
                y[train_idx],
                X.iloc[test_idx],
                y[test_idx],
            )

    def _validate_input(
        self,
        X: pd.DataFrame,
        y: Optional[np.ndarray] = None
    ):
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame")

        if y is not None:
            if not isinstance(y, np.ndarray):
                raise TypeError("y must be a numpy array")

            if len(X) != len(y):
                raise ValueError("X and y must have same length")