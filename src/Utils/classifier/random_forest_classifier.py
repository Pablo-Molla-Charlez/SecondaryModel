from typing import Union

import numpy as np
import pandas as pd

from Utils.classifier._classifier import (BaseClassifier)
from sklearn.ensemble import RandomForestClassifier

try:
    import skops.io as sio
except ImportError:
    sio = None


class RFClassifier(BaseClassifier):

    def __init__(self, random_state=None, **kwargs) -> None:
        super().__init__(random_state)

        if kwargs:
            kwargs.setdefault("random_state", self.random_state)
            self._clf = RandomForestClassifier(**kwargs)
        else:
            self._clf = RandomForestClassifier(n_estimators=500,
                                               max_depth=6,
                                               min_samples_leaf=20,
                                               random_state=self.random_state,
                                               n_jobs=-1,
                                               class_weight="balanced")

        self._fitted_clf = None

    def predict(self, X_test: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        if self._fitted_clf is None:
            raise AttributeError("The model has not been fitted yet.")
        return self._fitted_clf.predict(X_test)

    def predict_proba(self, X_test: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        if self._fitted_clf is None:
            raise AttributeError("The model has not been fitted yet.")
        return self._fitted_clf.predict_proba(X_test)

    def fit(self, X_train: Union[np.ndarray, pd.DataFrame], y_train: np.ndarray) -> object:
        self._fitted_clf = self._clf.fit(X_train, y_train)

        return self._fitted_clf

    def get_params(self, deep: bool = True) -> dict:
        params = super().get_params(deep)
        if deep:
            params.update(self._clf.get_params(deep))
        return params

    def save_model(self, model_path: str) -> None:
        if self._fitted_clf is None:
            raise AttributeError("The model has not been fitted yet.")
        if sio is None:
            raise ImportError("skops is required for RFClassifier.save_model()")
        sio.dump(self._fitted_clf, f"{model_path}.skops")

    def load_model(self, model_path: str) -> None:
        if sio is None:
            raise ImportError("skops is required for RFClassifier.load_model()")
        self._fitted_clf = sio.load(f"{model_path}.skops", trusted=True)
