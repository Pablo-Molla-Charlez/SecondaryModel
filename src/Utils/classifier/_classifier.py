from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from typing import Iterator, Tuple, Optional, Union


class BaseClassifier(ABC):
    """
    Abstract base class for M2 classifiers.

    Compatible with sklearn API.
    """

    def __init__(
        self,
        random_state: Optional[int] = None
    ):
        self.random_state = random_state if random_state is not None else np.random.randint(0, 100)
        self._rng = np.random.default_rng(random_state)

    @abstractmethod
    def fit(
        self,
        X_train: Union[np.ndarray, pd.DataFrame],
        y_train: np.ndarray
    ) -> object:
        """
        Fits the classifier; In case of foundation models this should handle the finetuning.

        Returns the fitted classifier


        """
        pass

    @abstractmethod
    def predict(
            self,
            X_test: Union[np.ndarray, pd.DataFrame]
    ) -> np.ndarray:
        """
        Predicts the class labels for X_test
        Args:
            X_test: Feature matrix

        Returns: class labels
        """
        pass

    @abstractmethod
    def predict_proba(
            self,
            X_test: Union[np.ndarray, pd.DataFrame]
    ) -> np.ndarray:
        """
        Predicts the class probabilities for X_test
        Args:
            X_test: Feature matrix

        Returns: class probabilities
        """
        pass

    @abstractmethod
    def get_params(
            self,
            deep: bool = True
    ) -> dict:
        """
        Required for sklearn.base.clone compatibility.
        """
        return {"random_state": self.random_state}

    @abstractmethod
    def save_model(
            self,
            model_path: str,
    ) -> None:
        """
        Saves the classifier model
        Returns:

        """
        pass

    @abstractmethod
    def load_model(
            self,
            model_path: str
    ) -> None:
        """
        Loads the classifier model
        Args:
            model_path: path to the saved model

        Returns:

        """
        pass