from typing import Union

import numpy as np
import pandas as pd

from Utils.classifier._classifier import (BaseClassifier)


class AutogluonClassifier(BaseClassifier):

    def __init__(self, random_state=None, **kwargs) -> None:
        super().__init__(random_state)

        raise NotImplementedError

    def fit(self, X_train: Union[np.ndarray, pd.DataFrame], y_train: np.ndarray) -> object:
        pass

    def predict(self, X_test: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        pass

    def predict_proba(self, X_test: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        pass

    def get_params(self, deep: bool = True) -> dict:
        pass

    def save_model(self, model_path: str) -> None:
        pass

    def load_model(self, model_path: str) -> None:
        pass