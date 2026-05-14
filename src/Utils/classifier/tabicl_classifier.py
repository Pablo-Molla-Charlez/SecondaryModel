"""TabICL classifier — BaseClassifier-compliant wrapper.

Ported verbatim from the legacy ``Utils.models._TabICLWrapper`` (absorbed in
Step 1 of the Utils/ modularization).

TabICL is a tabular In-Context Learning model from INRIA/Soda that uses a
Transformer pre-trained on synthetic datasets for zero-shot classification.
Like TabPFN, it internally normalises features — do NOT pass StandardScaler-
transformed data.

Reference: https://github.com/soda-inria/tabicl
"""
from typing import Union

import numpy as np
import pandas as pd
import warnings

from Utils.classifier._classifier import BaseClassifier

_TABICL_MAX_ROWS = 100_000


class TabICL(BaseClassifier):
    """Sklearn-compatible wrapper around TabICLClassifier."""

    def __init__(self,
                 n_estimators:        int   = 8,
                 softmax_temperature: float = 0.9,
                 random_state:        int   = 42,
                 device:              str   = "cuda") -> None:
        super().__init__(random_state)
        self.n_estimators        = n_estimators
        self.softmax_temperature = softmax_temperature
        self.random_state        = random_state
        self.device              = device
        self._clf                = None
        self.classes_            = None

    # ┏━━━━━━━━━━ Fit ━━━━━━━━━━┓
    def fit(self, X_train: Union[np.ndarray, pd.DataFrame], y_train: np.ndarray, sample_weight=None):
        from tabicl import TabICLClassifier

        X = np.asarray(X_train, dtype=np.float32)
        y = np.asarray(y_train)

        # ┏━━━━━━━━━━ Cap row count to prevent OOM ━━━━━━━━━━┓
        if len(X) > _TABICL_MAX_ROWS:
            warnings.warn(f"TabICL: training set has {len(X):,} rows (recommended ≤ {_TABICL_MAX_ROWS:,}). "
                          f"Randomly sub-sampling to {_TABICL_MAX_ROWS:,} rows.")
            rng = np.random.default_rng(self.random_state)
            idx = rng.choice(len(X), _TABICL_MAX_ROWS, replace=False)
            
            # Subsample X, y, and sample_weight if provided
            X = X[idx]
            y = y[idx]
            if sample_weight is not None:
                sample_weight = np.asarray(sample_weight)[idx]

        self.n_features_in_ = X.shape[1]

        self.classes_ = np.unique(y)

        self._clf = TabICLClassifier(n_estimators        = self.n_estimators,
                                     softmax_temperature = self.softmax_temperature,
                                     random_state        = self.random_state,
                                     device              = self.device)
        self._clf.fit(X, y)
        return self

    # ┏━━━━━━━━━━ Predict ━━━━━━━━━━┓
    def predict(self, X_test: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        if self._clf is None:
            raise AttributeError("The model has not been fitted yet.")
        return self._clf.predict(np.asarray(X_test, dtype=np.float32))

    # ┏━━━━━━━━━━ Predict Probabilities ━━━━━━━━━━┓
    def predict_proba(self, X_test: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        if self._clf is None:
            raise AttributeError("The model has not been fitted yet.")
        return self._clf.predict_proba(np.asarray(X_test, dtype=np.float32))

    # ┏━━━━━━━━━━ Feature Importance (uniform fallback) ━━━━━━━━━━┓
    @property
    def feature_importances_(self):
        n_feat = getattr(self, "n_features_in_", 1)
        return np.ones(n_feat) / n_feat

    # ┏━━━━━━━━━━ Get Parameters ━━━━━━━━━━┓
    def get_params(self, deep: bool = True) -> dict:
        return {"n_estimators":        self.n_estimators,
                "softmax_temperature": self.softmax_temperature,
                "random_state":        self.random_state,
                "device":              self.device}

    # ┏━━━━━━━━━━ Save Model ━━━━━━━━━━┓
    def save_model(self, model_path: str) -> None:
        if self._clf is None:
            raise AttributeError("The model has not been fitted yet.")
        self._clf.save(f"{model_path}.pkl",
                       save_model_weights = False,
                       save_training_data = True,
                       save_kv_cache      = True)

    # ┏━━━━━━━━━━ Load Model ━━━━━━━━━━┓
    def load_model(self, model_path: str) -> None:
        from tabicl import TabICLClassifier
        self._clf = TabICLClassifier.load(f"{model_path}.pkl")
