"""Classifier factory — absorbed from legacy ``Utils.models``.

Provides ``_build_tree_model`` and the ``MODEL_CHOICES`` / ``MODELS_NO_SCALING``
registries. Every branch returns a ``BaseClassifier`` subclass so all models
share the same ``fit`` / ``predict`` / ``save_model`` / ``load_model`` interface.
"""

# ┏━━━━━━━━━━ Model registry ━━━━━━━━━━┓
MODEL_CHOICES = ("randforest", "xgboost", "autogluon", "tabpfn", "tabpfn_ft", "tabicl")

# ┏━━━━━━━━━━ Models that must NOT receive StandardScaler-transformed data ━━━━━━━━━━┓
MODELS_NO_SCALING = {"tabpfn", "tabpfn_ft", "tabicl"}

# ┏━━━━━━━━━━ TabPFN soft row limit ━━━━━━━━━━┓
_TABPFN_MAX_ROWS = 50_000

# ┏━━━━━━━━━━ AutoGluon defaults ━━━━━━━━━━┓
_AG_TIME_LIMIT = 3600          # overridden by --ag-time-limit
_AG_PRESETS = "best_quality"  # overridden by --ag-presets


def _build_tree_model(model_name: str,  # TODO probabaly needs to be renamed bc we also use it for tabpfn etc.
                      n_samples: int,  # TODO never used?!
                      class_weight_ratio: float = 1.0,
                      feature_names=None,
                      time_limit=None,
                      presets: str = "best_quality"):
    """Return a ``BaseClassifier``-compatible instance for the requested model.

    Parameters
    ----------
    model_name : str
        One of ``MODEL_CHOICES``.
    n_samples : int
        Number of training samples (used to calibrate XGBoost ``scale_pos_weight``).
    class_weight_ratio : float
        n_neg / n_pos; passed to XGBoost and AutoGluon for class weighting.
    feature_names : list[str] | None
        Column names required by AutoGluon.
    time_limit : int | None
        Training time cap in seconds (AutoGluon only); falls back to ``_AG_TIME_LIMIT``.
    presets : str
        AutoGluon quality preset.
    """
    # ┏━━━━━━━━━━ Random Forest ━━━━━━━━━━┓
    if model_name == "randforest":
        from Utils.classifier.random_forest_classifier import RFClassifier
        return RFClassifier(n_estimators     = 500,
                            max_depth        = 6,
                            min_samples_leaf = 20,
                            random_state     = 42,
                            n_jobs           = -1,
                            class_weight     = "balanced")

    # ┏━━━━━━━━━━ XGBoost ━━━━━━━━━━┓
    elif model_name == "xgboost":
        from xgboost import XGBClassifier
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
        raise ValueError(f"Unknown model: {model_name!r}. Choose from {MODEL_CHOICES}")