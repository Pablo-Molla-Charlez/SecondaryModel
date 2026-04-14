from typing import Union

import numpy as np
import pandas as pd

from Utils.classifier._classifier import (BaseClassifier)

from tabpfn import TabPFNClassifier
from tabpfn.model_loading import (
    load_fitted_tabpfn_model,
    save_fitted_tabpfn_model,
)

class TabPFN(BaseClassifier):

    def __init__(self, random_state=None, **kwargs) -> None:
        super().__init__(random_state)

        self.clf = TabPFNClassifier(**kwargs)

        self._fitted_clf = None

    def fit(self, X_train: Union[np.ndarray, pd.DataFrame], y_train: np.ndarray) -> object:
        self._fitted_clf = self.clf.fit(X_train, y_train)

        return self._fitted_clf

    def predict(self, X_test: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        if self._fitted_clf is None:
            raise AttributeError("The model has not been fitted yet.")
        return self._fitted_clf.predict(X_test)

    def predict_proba(self, X_test: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        if self._fitted_clf is None:
            raise AttributeError("The model has not been fitted yet.")
        return self._fitted_clf.predict_proba(X_test)

    def get_params(self, deep: bool = True) -> dict:
        params = super().get_params(deep)
        if deep:
            params.update(self._clf.get_params(deep))
        return params

    def save_model(self, model_path: str) -> None:
        if self._fitted_clf is not None:
            self._fitted_clf.save_model(model_path)
            save_fitted_tabpfn_model(self._fitted_clf,f"{model_path}.tabpfn_fit")
        else:
            raise AttributeError("The model has not been fitted yet.")

    def load_model(self, model_path: str, device="cpu") -> None:
        self._fitted_clf = load_fitted_tabpfn_model(f"{model_path}.tabpfn_fit", device=device)

