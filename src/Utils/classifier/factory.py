"""Classifier factory — absorbed from legacy ``Utils.models``.

Provides ``_build_tree_model`` and the ``MODEL_CHOICES`` / ``MODELS_NO_SCALING``
registries. Constructors route to the BaseClassifier implementations now
living inside ``Utils.classifier``.
"""
from __future__ import annotations

from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# ┏━━━━━━━━━━ Model choices (verbatim from Utils/models.py) ━━━━━━━━━━┓
MODEL_CHOICES = ("rf", "xgboost", "autogluon", "tabpfn", "tabpfn_ft", "tabicl")

# ┏━━━━━━━━━━ Models that must NOT receive StandardScaler-transformed data ━━━━━━━━━━┓
MODELS_NO_SCALING = {"tabpfn", "tabpfn_ft", "tabicl"}

# ┏━━━━━━━━━━ TabPFN Parameters ━━━━━━━━━━┓
# Max training rows TabPFN handles comfortably (soft limit — we warn, not crash)
_TABPFN_MAX_ROWS = 50_000

# ┏━━━━━━━━━━ AutoGluon parameters ━━━━━━━━━━┓
_AG_TIME_LIMIT = 3600          # overridden by --ag-time-limit
_AG_PRESETS = "best_quality"  # overridden by --ag-presets


# ┏━━━━━━━━━━ Build Tree Model [Random Forest, XGBoost, AutoGluon, TabPFN, TabICL] ━━━━━━━━━━┓
def _build_tree_model(model_name: str,
                      n_samples: int,
                      class_weight_ratio: float = 1.0,
                      feature_names=None,
                      time_limit=None,
                      presets: str = "best_quality"):
    """Return a scikit-learn-compatible classifier.

    Args:
        model_name: 'rf', 'xgboost', 'autogluon', 'tabpfn', 'tabpfn_ft', or 'tabicl'
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
        return XGBClassifier(n_estimators     = 300,
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
        from Utils.classifier.autogluon_classifier import AutoGluon
        return AutoGluon(feature_names      = feature_names,
                         time_limit         = time_limit if time_limit is not None else _AG_TIME_LIMIT,
                         presets            = presets,
                         class_weight_ratio = class_weight_ratio)

    # ┏━━━━━━━━━━ TabPFN (zero-shot) ━━━━━━━━━━┓
    elif model_name == "tabpfn":
        from Utils.classifier.tabpfn_classifier import TabPFN
        return TabPFN()

    # ┏━━━━━━━━━━ TabPFN (fine-tuned) ━━━━━━━━━━┓
    elif model_name == "tabpfn_ft":
        from Utils.classifier.tabpfn_finetuned_classifier import TabPFNFineTuned
        return TabPFNFineTuned()

    # ┏━━━━━━━━━━ TabICL (in-context learning) ━━━━━━━━━━┓
    elif model_name == "tabicl":
        from Utils.classifier.tabicl_classifier import TabICL
        return TabICL()

    else:
        raise ValueError(f"Unknown model: {model_name}. Choose from {MODEL_CHOICES}")
