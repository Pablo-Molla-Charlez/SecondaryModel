"""TabPFN (zero-shot) classifier — BaseClassifier-compliant wrapper.

Ported verbatim from the legacy ``Utils.models._TabPFNWrapper`` (absorbed in
Step 1 of the Utils/ modularization).  Preserves every knob HPO exercises:

  * ``_TABPFN_MAX_ROWS`` soft row limit with warning + random sub-sampling
  * ``inference_config`` support (PREPROCESS_TRANSFORMS, OUTLIER_REMOVAL_STD, ...)
  * ``average_before_softmax``, ``softmax_temperature``, ``balance_probabilities``
  * ``fit_mode``, ``n_estimators``, ``device``, ``random_state``

Exposes sklearn-compat attributes ``classes_``, ``n_features_in_``,
``feature_importances_``, and a ``get_params`` dict that mirrors every
HPO-tunable knob.
"""
from typing import Union

import numpy as np
import pandas as pd

from Utils.classifier._classifier import BaseClassifier

# ┏━━━━━━━━━━ Constants (ported verbatim from Utils/models.py) ━━━━━━━━━━┓
# Max training rows TabPFN handles comfortably (soft limit — we warn, not crash)
_TABPFN_MAX_ROWS = 50_000


class TabPFN(BaseClassifier):
    """Sklearn-compatible wrapper around TabPFNClassifier (zero-shot).

    Uses the pre-trained TabPFN prior for in-context learning — no gradient
    updates.  Quality comes from the ensemble size (n_estimators) and from
    softmax_temperature which controls calibration.

    TabPFN internally normalises features, so do NOT pass StandardScaler-
    transformed data — use raw engineered features.

    Limitations: recommended < 50 000 rows, < 2 000 features.  Rows exceeding
    _TABPFN_MAX_ROWS are randomly sub-sampled with a warning.
    """

    def __init__(self,
                 device:                 str   = "cuda",
                 n_estimators:           int   = 16,
                 softmax_temperature:    float = 0.9,
                 balance_probabilities:  bool  = False,
                 average_before_softmax: bool  = False,
                 fit_mode:               str   = "fit_preprocessors",
                 inference_config:       dict | None = None,
                 random_state:           int   = 42) -> None:
        super().__init__(random_state)
        self.device                 = device
        self.n_estimators           = n_estimators           # higher = better, costs linear RAM
        self.softmax_temperature    = softmax_temperature    # <1 sharpens, >1 smooths predictions
        self.balance_probabilities  = balance_probabilities  # True corrects for class imbalance in prior
        self.average_before_softmax = average_before_softmax # ensemble averaging mode (logits vs probs)
        self.fit_mode               = fit_mode               # "fit_preprocessors" is the default; "fit_with_cache" not yet supported in v2.6
        self.inference_config       = inference_config       # dict of preprocessing knobs (PREPROCESS_TRANSFORMS, OUTLIER_REMOVAL_STD, ...)
        self.random_state           = random_state
        self._clf                   = None
        self.classes_               = None

    # ┏━━━━━━━━━━ Fit ━━━━━━━━━━┓
    def fit(self, X_train: Union[np.ndarray, pd.DataFrame], y_train: np.ndarray, sample_weight=None):
        import warnings
        from tabpfn import TabPFNClassifier

        X = np.asarray(X_train, dtype=np.float32)
        y = np.asarray(y_train)
        self.n_features_in_ = X.shape[1]

        # ┏━━━━━━━━━━ Soft row limit ━━━━━━━━━━┓
        if len(X) > _TABPFN_MAX_ROWS:
            warnings.warn(f"TabPFN: training set has {len(X):,} rows (recommended ≤ {_TABPFN_MAX_ROWS:,}). "
                          f"Randomly sub-sampling to {_TABPFN_MAX_ROWS:,} rows.",
                          UserWarning, stacklevel=2)
            rng = np.random.default_rng(self.random_state)
            idx = rng.choice(len(X), _TABPFN_MAX_ROWS, replace=False)
            X, y = X[idx], y[idx]

        self.classes_ = np.unique(y)

        # ┏━━━━━━━━━━ Initialize TabPFNClassifier (local weights, downloaded once) ━━━━━━━━━━┓
        clf_kwargs = dict(n_estimators           = self.n_estimators,
                          softmax_temperature    = self.softmax_temperature,
                          balance_probabilities  = self.balance_probabilities,
                          average_before_softmax = self.average_before_softmax,
                          fit_mode               = self.fit_mode,
                          device                 = self.device,
                          random_state           = self.random_state)
        if self.inference_config is not None:
            clf_kwargs["inference_config"] = self.inference_config
        self._clf = TabPFNClassifier(**clf_kwargs)
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

    # ┏━━━━━━━━━━ Get Params ━━━━━━━━━━┓
    def get_params(self, deep: bool = True) -> dict:
        return {"device":                 self.device,
                "n_estimators":           self.n_estimators,
                "softmax_temperature":    self.softmax_temperature,
                "balance_probabilities":  self.balance_probabilities,
                "average_before_softmax": self.average_before_softmax,
                "fit_mode":               self.fit_mode,
                "inference_config":       self.inference_config,
                "random_state":           self.random_state}

    # ┏━━━━━━━━━━ Save Model ━━━━━━━━━━┓
    def save_model(self, model_path: str) -> None:
        if self._clf is None:
            raise AttributeError("The model has not been fitted yet.")
        try:
            from tabpfn.model_loading import save_fitted_tabpfn_model
        except Exception as e:
            raise ImportError("save_fitted_tabpfn_model unavailable") from e
        save_fitted_tabpfn_model(self._clf, f"{model_path}.tabpfn_fit")

    # ┏━━━━━━━━━━ Load Model ━━━━━━━━━━┓
    def load_model(self, model_path: str, device: str = "cpu") -> None:
        try:
            from tabpfn.model_loading import load_fitted_tabpfn_model
        except Exception as e:
            raise ImportError("load_fitted_tabpfn_model unavailable") from e
        self._clf = load_fitted_tabpfn_model(f"{model_path}.tabpfn_fit", device=device)
