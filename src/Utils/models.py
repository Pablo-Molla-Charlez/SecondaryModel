# Utils/models.py
import pandas as pd
import numpy as np
import tempfile
import json
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from Utils.utils import _safe_json

# ┏━━━━━━━━━━ Model choices ━━━━━━━━━━┓
MODEL_CHOICES = ("rf", "xgboost", "autogluon", "tabpfn", "tabpfn_ft")

# ┏━━━━━━━━━━ TabPFN Parameters ━━━━━━━━━━┓
# Models that must NOT receive StandardScaler-transformed data
MODELS_NO_SCALING = {"tabpfn", "tabpfn_ft"}

# Max training rows TabPFN handles comfortably (soft limit — we warn, not crash)
_TABPFN_MAX_ROWS = 50_000

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


# ┏━━━━━━━━━━ TabPFN zero-shot wrapper ━━━━━━━━━━┓
class _TabPFNWrapper:
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
                 device:                str   = "auto",
                 n_estimators:          int   = 16,
                 softmax_temperature:   float = 0.9,
                 balance_probabilities: bool  = False,
                 fit_mode:              str   = "fit_preprocessors",
                 random_state:          int   = 42):
        self.device                = device
        self.n_estimators          = n_estimators          # higher = better, costs linear RAM
        self.softmax_temperature   = softmax_temperature   # <1 sharpens, >1 smooths predictions
        self.balance_probabilities = balance_probabilities  # True corrects for class imbalance in prior
        self.fit_mode              = fit_mode               # "fit_preprocessors" is the default; "fit_with_cache" not yet supported in v2.6
        self.random_state          = random_state
        self._clf                  = None
        self.classes_              = None

    # ┏━━━━━━━━━━ fit ━━━━━━━━━━┓
    def fit(self, X, y, sample_weight=None):
        import warnings
        from tabpfn import TabPFNClassifier

        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y)
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
        self._clf = TabPFNClassifier(n_estimators          = self.n_estimators,
                                     softmax_temperature   = self.softmax_temperature,
                                     balance_probabilities = self.balance_probabilities,
                                     fit_mode              = self.fit_mode,
                                     device                = self.device,
                                     random_state          = self.random_state)
        self._clf.fit(X, y)
        return self

    # ┏━━━━━━━━━━ Predict ━━━━━━━━━━┓
    def predict(self, X):
        return self._clf.predict(np.asarray(X, dtype=np.float32))

    # ┏━━━━━━━━━━ Predict Probabilities ━━━━━━━━━━┓
    def predict_proba(self, X):
        return self._clf.predict_proba(np.asarray(X, dtype=np.float32))

    # ┏━━━━━━━━━━ Feature Importance (uniform fallback) ━━━━━━━━━━┓
    @property
    def feature_importances_(self):
        n_feat = getattr(self, "n_features_in_", 1)
        return np.ones(n_feat) / n_feat


# ┏━━━━━━━━━━ TabPFN fine-tuned wrapper ━━━━━━━━━━┓
class _TabPFNFineTunedWrapper:
    """Sklearn-compatible wrapper around FinetunedTabPFNClassifier.

    Runs gradient-based fine-tuning on the training set so the model
    adapts its pre-trained weights to the specific financial feature
    distribution.  Defaults tuned for meta-labeling on ~2-3k rows,
    23 features, binary classification.
    """

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
                 random_state:            int   = 42):
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

    # ┏━━━━━━━━━━ fit ━━━━━━━━━━┓
    def fit(self, X, y, sample_weight=None):
        import warnings
        import tempfile
        from tabpfn.finetuning import FinetunedTabPFNClassifier

        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y)
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
    def predict(self, X):
        return self._clf.predict(np.asarray(X, dtype=np.float32))

    # ┏━━━━━━━━━━ Predict Probabilities ━━━━━━━━━━┓
    def predict_proba(self, X):
        return self._clf.predict_proba(np.asarray(X, dtype=np.float32))

    # ┏━━━━━━━━━━ Feature Importance (uniform fallback) ━━━━━━━━━━┓
    @property
    def feature_importances_(self):
        n_feat = getattr(self, "n_features_in_", 1)
        return np.ones(n_feat) / n_feat


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

    # ┏━━━━━━━━━━ TabPFN (zero-shot) ━━━━━━━━━━┓
    elif model_name == "tabpfn":
        return _TabPFNWrapper()

    # ┏━━━━━━━━━━ TabPFN (fine-tuned) ━━━━━━━━━━┓
    elif model_name == "tabpfn_ft":
        return _TabPFNFineTunedWrapper()

    else:
        raise ValueError(f"Unknown model: {model_name}. Choose from {MODEL_CHOICES}")
