from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from typing import Optional


class FeatureSelection(ABC):

    def __init__(self, clf, scoring, cross_validation_strategy, **kwargs) -> None:
        self.clf = clf
        self.scoring = scoring
        self.cross_validation_strategy = cross_validation_strategy
        self.kwargs = kwargs

    @abstractmethod
    def select_features(self,
                        X: pd.DataFrame,
                        y: np.ndarray,
                        n_features: int,
                        X_test: Optional[pd.DataFrame] = None,
                        y_test: Optional[np.ndarray] = None,
                        **kwargs) -> pd.DataFrame:
        """

        Args:
            X: Analysis feature matrix
            y: Analysis target vector
            n_features: how many features to select
            X_test: Test feature matrix (Optional to directly make evaluation)
            y_test: Test target vector (Optional to directly make evaluation)
            **kwargs: any arguments that may are required

        Returns:
            a pandas dataframe with features and evaluation metrics
            {"feature_set": [feature names], "evaluation": [accuracy splits]}
        """
        pass