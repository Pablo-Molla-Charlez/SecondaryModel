from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from typing import Iterator, Tuple, Optional


class BaseTimeSeriesCV(ABC):
    """
    Abstract base class for temporal cross-validation.

    Enforces:
        - n_splits
        - reproducibility via random_state
    """

    def __init__(
        self,
        n_splits: int,
        random_state: Optional[int] = None
    ):
        """
        Parameters
        ----------
        n_splits : int
            Number of splits/folds.
        random_state : int or None
            Seed for reproducibility.
        """
        if not isinstance(n_splits, int) or n_splits < 1:
            raise ValueError("n_splits must be a positive integer")

        self.n_splits = n_splits
        self.random_state = random_state
        self._rng = np.random.default_rng(random_state)

    @abstractmethod
    def split(
        self,
        X: pd.DataFrame,
        y: np.ndarray
    ) -> Iterator[Tuple[pd.DataFrame, np.ndarray, pd.DataFrame, np.ndarray]]:
        """
        Generate train/test splits.

        Returns
        -------
        Iterator over:
            (X_train, y_train, X_test, y_test)
        """
        pass