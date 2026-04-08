from Utils.ts_cross_validation._ts_cross_validation import BaseTimeSeriesCV
from sklearn.model_selection import TimeSeriesSplit
import numpy as np
import pandas as pd


class SklearnTimeSeriesCV(BaseTimeSeriesCV):

    def __init__(
        self,
        n_splits: int = 5,
        max_train_size: int = None,
        test_size: int = None,
        random_state: int = None
    ):
        super().__init__(n_splits=n_splits, random_state=random_state)

        self.max_train_size = max_train_size
        self.test_size = test_size

        self._tscv = TimeSeriesSplit(
            n_splits=self.n_splits,
            max_train_size=max_train_size,
            test_size=test_size
        )

    def split(self, X: pd.DataFrame, y: np.ndarray):
        """
        Yield temporal splits as actual data (not indices).
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame")

        if not isinstance(y, np.ndarray):
            raise TypeError("y must be a numpy array")

        if len(X) != len(y):
            raise ValueError("X and y must have same length")

        for train_idx, test_idx in self._tscv.split(X):
            X_train = X.iloc[train_idx]
            X_test = X.iloc[test_idx]

            y_train = y[train_idx]
            y_test = y[test_idx]

            yield X_train, y_train, X_test, y_test