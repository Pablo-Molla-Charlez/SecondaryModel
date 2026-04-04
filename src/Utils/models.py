# Utils/models.py
import pandas as pd
import tempfile
import json
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from Utils.utils import _safe_json

# ┏━━━━━━━━━━ Model choices ━━━━━━━━━━┓
MODEL_CHOICES = ("rf", "xgboost", "autogluon")

# ┏━━━━━━━━━━ AutoGluon parameters ━━━━━━━━━━┓
_AG_TIME_LIMIT = 3600          # overridden by --ag-time-limit
_AG_PRESETS = "best_quality"  # overridden by --ag-presets


# ┏━━━━━━━━━━ AutoGluon wrapper ━━━━━━━━━━┓
class _AutoGluonWrapper:
    """Sklearn-compatible wrapper around AutoGluon TabularPredictor."""

    def __init__(self, feature_names=None, time_limit=300, presets="best_quality", class_weight_ratio=1.0):
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

    def fit(self, X, y, sample_weight=None):
        from autogluon.tabular import TabularPredictor
        import tempfile

        df = self._to_df(X)
        df[self._label_col] = y

        ag_ctor_kwargs = {}
        if sample_weight is not None:
            df[self._weight_col] = sample_weight
            ag_ctor_kwargs["sample_weight"] = self._weight_col

        save_dir = tempfile.mkdtemp(prefix="ag_model_")
        self._predictor = TabularPredictor(label = self._label_col,
                                           path = save_dir,
                                           eval_metric = "f1",
                                           verbosity = 1,
                                           **ag_ctor_kwargs).fit(train_data = df,
                                                                 time_limit = self.time_limit,
                                                                 presets = self.presets)
        self._train_df = df  # kept for feature_importance
        return self

    def predict(self, X):
        df = self._to_df(X)
        return self._predictor.predict(df).values.astype(int)

    def predict_proba(self, X):
        df = self._to_df(X)
        proba = self._predictor.predict_proba(df)
        return proba.values

    @property
    def feature_importances_(self):
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

        export = {"presets": self.presets,
                  "time_limit": self.time_limit,
                  "eval_metric": "f1",
                  "num_models_trained": len(model_info_dict),
                  "models": {}}
        
        # ┏━━━━━━━━━━ 3. Per-model detailed info JSON ━━━━━━━━━━┓
        for name, info in model_info_dict.items():
            export["models"][name] = {"model_type": str(info.get("model_type", "N/A")),
                                      "hyperparameters": {k: _safe_json(v) for k, v in info.get("hyperparameters", {}).items()},
                                      "num_features": info.get("num_features", None),
                                      "stack_level": info.get("stacker_info", {}).get("stacker_level", 0) if isinstance(info.get("stacker_info"), dict) else 0,
                                      "fit_time": round(info.get("fit_time", 0), 2),
                                      "pred_time_val": round(info.get("pred_time_val", 0), 4),
                                      "val_score": round(info.get("val_score", 0), 4),
                                      "children": info.get("children_info", {}).get("children", []) if isinstance(info.get("children_info"), dict) else []}

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


# ┏━━━━━━━━━━ Build Tree Model [Random Forest, XGBoost, AutoGluon] ━━━━━━━━━━┓
def _build_tree_model(model_name: str, n_samples: int, class_weight_ratio: float = 1.0,
                      feature_names=None, time_limit=None, presets="best_quality"):
    """Return a scikit-learn-compatible classifier.

    Args:
        model_name: 'rf', 'xgboost', or 'autogluon'
        n_samples: number of training samples (used to calibrate XGB)
        class_weight_ratio: n_neg / n_pos for scale_pos_weight in XGB
        feature_names: list of feature names (required for autogluon)
        time_limit: training time limit in seconds (autogluon only)
        presets: autogluon model preset
    """
    # ┏━━━━━━━━━━ Random Forest ━━━━━━━━━━┓
    if model_name == "rf":
        return RandomForestClassifier(n_estimators     = 500, 
                                      max_depth        = 6, 
                                      min_samples_leaf = 20, 
                                      random_state     = 42, 
                                      n_jobs           = -1, 
                                      class_weight     = "balanced")
    
    # ┏━━━━━━━━━━ XGBoost ━━━━━━━━━━┓
    elif model_name == "xgboost":
        return XGBClassifier(n_estimators       = 300,
                             max_depth        = 4,
                             learning_rate    = 0.05,
                             min_child_weight = 20,
                             subsample        = 0.8,
                             colsample_bytree = 0.8,
                             gamma            = 1.0,
                             reg_alpha        = 0.1,
                             reg_lambda       = 1.0,
                             scale_pos_weight = class_weight_ratio,
                             random_state     = 42,
                             n_jobs           = -1,
                             eval_metric      = "logloss",
                             verbosity        = 0)

    # ┏━━━━━━━━━━ AutoGluon ━━━━━━━━━━┓
    elif model_name == "autogluon":
        return _AutoGluonWrapper(feature_names      = feature_names,
                                 time_limit         = time_limit if time_limit is not None else _AG_TIME_LIMIT,
                                 presets            = presets,
                                 class_weight_ratio = class_weight_ratio)
    else:
        raise ValueError(f"Unknown model: {model_name}. Choose from {MODEL_CHOICES}")
