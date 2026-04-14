"""AutoGluon classifier — BaseClassifier-compliant wrapper.

Ported verbatim from the legacy ``Utils.models._AutoGluonWrapper`` (absorbed
in Step 1 of the Utils/ modularization). Sklearn-compatible wrapper around
AutoGluon's TabularPredictor with helpers for leaderboard / model-info
export and persistent save.
"""
import json
import tempfile
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd

from Utils.classifier._classifier import BaseClassifier
from Utils.utils import _safe_json


class AutoGluon(BaseClassifier):
    """Sklearn-compatible wrapper around AutoGluon TabularPredictor."""

    def __init__(self,
                 feature_names=None,
                 time_limit: int = 300,
                 presets: str = "best_quality",
                 class_weight_ratio: float = 1.0,
                 random_state: int | None = None) -> None:
        super().__init__(random_state)
        self.feature_names = feature_names
        self.time_limit = time_limit
        self.presets = presets
        self.class_weight_ratio = class_weight_ratio
        self._predictor = None
        self._label_col = "__target__"
        self._weight_col = "__sample_weight__"

    def _to_df(self, X):
        cols = self.feature_names if self.feature_names else [f"f{i}" for i in range(X.shape[1])]
        return pd.DataFrame(X, columns=cols)

    def fit(self, X_train: Union[np.ndarray, pd.DataFrame], y_train: np.ndarray, sample_weight=None):
        from autogluon.tabular import TabularPredictor

        X = np.asarray(X_train) if not isinstance(X_train, pd.DataFrame) else X_train.values
        self.n_features_in_ = X.shape[1]
        df = self._to_df(X)
        df[self._label_col] = y_train

        ag_ctor_kwargs = {}
        if sample_weight is not None:
            df[self._weight_col] = sample_weight
            ag_ctor_kwargs["sample_weight"] = self._weight_col

        save_dir = tempfile.mkdtemp(prefix="ag_model_")
        self._predictor = TabularPredictor(label       = self._label_col,
                                           path        = save_dir,
                                           eval_metric = "f1",
                                           verbosity   = 1,
                                           **ag_ctor_kwargs).fit(train_data = df,
                                                                 time_limit = self.time_limit,
                                                                 presets    = self.presets)
        self._train_df = df  # kept for feature_importance
        self.classes_ = np.unique(np.asarray(y_train))
        return self

    def predict(self, X_test: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        if self._predictor is None:
            raise AttributeError("The model has not been fitted yet.")
        X = np.asarray(X_test) if not isinstance(X_test, pd.DataFrame) else X_test.values
        df = self._to_df(X)
        return self._predictor.predict(df).values.astype(int)

    def predict_proba(self, X_test: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        if self._predictor is None:
            raise AttributeError("The model has not been fitted yet.")
        X = np.asarray(X_test) if not isinstance(X_test, pd.DataFrame) else X_test.values
        df = self._to_df(X)
        proba = self._predictor.predict_proba(df)
        return proba.values

    @property
    def feature_importances_(self):
        if self._predictor is None:
            raise AttributeError("The model has not been fitted yet.")
        imp = self._predictor.feature_importance(data=self._train_df, silent=True)
        return imp["importance"].values

    def leaderboard(self):
        """Print and return the AutoGluon model leaderboard."""
        return self._predictor.leaderboard(silent=False)

    def model_info(self, save_dir):
        """Save detailed model info (leaderboard + per-model hyperparams) to files."""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        # ┏━━━━━━━━━━ 1. Leaderboard CSV ━━━━━━━━━━┓
        lb = self._predictor.leaderboard(silent=True)
        lb_path = save_dir / "ag_leaderboard.csv"
        lb.to_csv(lb_path, index=False)
        print(f"  AutoGluon leaderboard saved to {lb_path}")

        # ┏━━━━━━━━━━ 2. Per-model detailed info JSON ━━━━━━━━━━┓
        full_info = self._predictor.info()
        model_info_dict = full_info.get("model_info", {})

        export = {"presets":            self.presets,
                  "time_limit":         self.time_limit,
                  "eval_metric":        "f1",
                  "num_models_trained": len(model_info_dict),
                  "models":             {}}

        # ┏━━━━━━━━━━ 3. Per-model detailed info JSON ━━━━━━━━━━┓
        for name, info in model_info_dict.items():
            export["models"][name] = {"model_type":      str(info.get("model_type", "N/A")),
                                      "hyperparameters": {k: _safe_json(v) for k, v in info.get("hyperparameters", {}).items()},
                                      "num_features":    info.get("num_features", None),
                                      "stack_level":     info.get("stacker_info", {}).get("stacker_level", 0) if isinstance(info.get("stacker_info"), dict) else 0,
                                      "fit_time":        round(info.get("fit_time", 0), 2),
                                      "pred_time_val":   round(info.get("pred_time_val", 0), 4),
                                      "val_score":       round(info.get("val_score", 0), 4),
                                      "children":        info.get("children_info", {}).get("children", []) if isinstance(info.get("children_info"), dict) else []}

        # ┏━━━━━━━━━━ 4. Save detailed info JSON ━━━━━━━━━━┓
        info_path = save_dir / "ag_model_info.json"
        with open(info_path, "w") as f:
            json.dump(export, f, indent=2, default=str)
        print(f"  AutoGluon model info saved to {info_path}")
        return export

    def save_to(self, path):
        """Persist the AutoGluon predictor to a permanent directory."""
        import shutil
        dest = Path(path) / "ag_model"
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(self._predictor.path, str(dest))
        print(f"  AutoGluon model saved to {dest}")
        return dest

    def get_params(self, deep: bool = True) -> dict:
        return {
            "feature_names":      self.feature_names,
            "time_limit":         self.time_limit,
            "presets":            self.presets,
            "class_weight_ratio": self.class_weight_ratio,
            "random_state":       self.random_state,
        }

    def save_model(self, model_path: str) -> None:
        if self._predictor is None:
            raise AttributeError("The model has not been fitted yet.")
        self.save_to(model_path)

    def load_model(self, model_path: str) -> None:
        from autogluon.tabular import TabularPredictor
        self._predictor = TabularPredictor.load(str(Path(model_path) / "ag_model"))


# Backward-compat alias — tests / older code may still import `AutogluonClassifier`.
AutogluonClassifier = AutoGluon
