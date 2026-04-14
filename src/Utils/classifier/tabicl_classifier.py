from typing import Union

import numpy as np
import pandas as pd

from Utils.classifier._classifier import (BaseClassifier)
from tabicl import TabICLClassifier



class TabICL(BaseClassifier):

    def __init__(self, random_state=None, **kwargs) -> None:
        super().__init__(random_state)

        self._clf = TabICLClassifier(**kwargs)

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
        if self._fitted_clf is not None:
            self._fitted_clf.save(
                f"{model_path}.pkl",
                save_model_weights=False,  # if False, reload from checkpoint on load
                save_training_data=True,  # if True, include training data; if False, discard it (requires KV cache)
                save_kv_cache=True,  # if True and KV cache exists, save it
            )
        else:
            raise AttributeError("The model has not been fitted yet.")

    def load_model(self, model_path: str) -> None:
        self._fitted_clf = TabICLClassifier.load(f"{model_path}.pkl")
