"""TabPFN fine-tuned classifier — BaseClassifier-compliant wrapper.

Ported verbatim from the legacy ``Utils.models._TabPFNFineTunedWrapper``
(absorbed in Step 1 of the Utils/ modularization). Runs gradient-based
fine-tuning on the training set so the model adapts its pre-trained weights
to the specific financial feature distribution. Defaults tuned for
meta-labeling on ~2-3k rows, 23 features, binary classification.
"""
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd

from Utils.classifier._classifier import BaseClassifier

# ┏━━━━━━━━━━ Constants (ported verbatim from Utils/models.py) ━━━━━━━━━━┓
# Max training rows TabPFN handles comfortably (soft limit — we warn, not crash)
_TABPFN_MAX_ROWS = 50_000


class TabPFNFineTuned(BaseClassifier):
    """Sklearn-compatible wrapper around FinetunedTabPFNClassifier."""

    # ┏━━━━━━━━━━ Constructor ━━━━━━━━━━┓
    def __init__(self,
                 device:                  str   = "cuda",
                 epochs:                  int   = 40,
                 learning_rate:           float = 1e-5,
                 weight_decay:            float = 0.01,
                 grad_clip_value:         float = 1.0,
                 validation_split_ratio:  float = 0.10,
                 early_stopping:          bool  = True,
                 early_stopping_patience: int   = 10,
                 min_delta:               float = 1e-4,
                 use_lr_scheduler:        bool  = True,
                 n_estimators:            int   = 8,
                 random_state:            int   = 42) -> None:
        super().__init__(random_state)
        self.device                  = device
        self.epochs                  = epochs                  # 40: enough for 2k rows; early stopping guards overfit
        self.learning_rate           = learning_rate           # 1e-5: conservative — don't destroy the pre-trained prior
        self.weight_decay            = weight_decay            # L2 regularisation
        self.grad_clip_value         = grad_clip_value         # stabilise gradients on noisy financial data
        self.validation_split_ratio  = validation_split_ratio  # 10% held out for early stopping
        self.early_stopping          = early_stopping
        self.early_stopping_patience = early_stopping_patience # 10: generous — financial loss surfaces are noisy
        self.min_delta               = min_delta
        self.use_lr_scheduler        = use_lr_scheduler        # cosine-with-warmup built in
        self.n_estimators            = n_estimators            # unified: finetune/validation/inference must match (v2.5 requirement)
        self.random_state            = random_state
        self._clf                    = None
        self.classes_                = None
        self._output_dir             = None

    # ┏━━━━━━━━━━ Fit ━━━━━━━━━━┓
    def fit(self, X_train: Union[np.ndarray, pd.DataFrame], y_train: np.ndarray, sample_weight=None):
        import warnings
        import tempfile
        from tabpfn.finetuning import FinetunedTabPFNClassifier

        X = np.asarray(X_train, dtype=np.float32)
        y = np.asarray(y_train)
        self.n_features_in_ = X.shape[1]

        # ┏━━━━━━━━━━ Soft row limit ━━━━━━━━━━┓
        if len(X) > _TABPFN_MAX_ROWS:
            warnings.warn(f"TabPFN-FT: training set has {len(X):,} rows (recommended ≤ {_TABPFN_MAX_ROWS:,}). "
                          f"Randomly sub-sampling to {_TABPFN_MAX_ROWS:,} rows.",
                          UserWarning, stacklevel=2)
            rng = np.random.default_rng(self.random_state)
            idx = rng.choice(len(X), _TABPFN_MAX_ROWS, replace=False)
            X, y = X[idx], y[idx]

        self.classes_ = np.unique(y)

        # ┏━━━━━━━━━━ Temp dir for checkpointing (avoids "no output_dir" warning) ━━━━━━━━━━┓
        self._output_dir = Path(tempfile.mkdtemp(prefix="tabpfn_ft_"))

        # ┏━━━━━━━━━━ Initialize FinetunedTabPFNClassifier ━━━━━━━━━━┓
        # n_estimators_finetune == n_estimators_validation == n_estimators_final_inference
        # is required by use_fixed_preprocessing_seed (v2.5 constraint)
        self._clf = FinetunedTabPFNClassifier(device                       = self.device,
                                              epochs                       = self.epochs,
                                              learning_rate                = self.learning_rate,
                                              weight_decay                 = self.weight_decay,
                                              grad_clip_value              = self.grad_clip_value,
                                              validation_split_ratio       = self.validation_split_ratio,
                                              early_stopping               = self.early_stopping,
                                              early_stopping_patience      = self.early_stopping_patience,
                                              min_delta                    = self.min_delta,
                                              use_lr_scheduler             = self.use_lr_scheduler,
                                              n_estimators_finetune        = self.n_estimators,
                                              n_estimators_validation      = self.n_estimators,
                                              n_estimators_final_inference = self.n_estimators,
                                              random_state                 = self.random_state)
        self._clf.fit(X, y, output_dir=self._output_dir)
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
        return {"device":                  self.device,
                "epochs":                  self.epochs,
                "learning_rate":           self.learning_rate,
                "weight_decay":            self.weight_decay,
                "grad_clip_value":         self.grad_clip_value,
                "validation_split_ratio":  self.validation_split_ratio,
                "early_stopping":          self.early_stopping,
                "early_stopping_patience": self.early_stopping_patience,
                "min_delta":               self.min_delta,
                "use_lr_scheduler":        self.use_lr_scheduler,
                "n_estimators":            self.n_estimators,
                "random_state":            self.random_state}

    # ┏━━━━━━━━━━ Save Model ━━━━━━━━━━┓
    def save_model(self, model_path: str) -> None:
        if self._clf is None:
            raise AttributeError("The model has not been fitted yet.")
        if hasattr(self._clf, "save_model"):
            self._clf.save_model(model_path)
        else:
            raise NotImplementedError("save_model not supported for TabPFNFineTuned")

    # ┏━━━━━━━━━━ Load Model ━━━━━━━━━━┓
    def load_model(self, model_path: str) -> None:
        raise NotImplementedError("load_model not supported for TabPFNFineTuned")
