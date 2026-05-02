"""Classifier factory — absorbed from legacy ``Utils.models``.

Provides ``_build_tree_model`` and the ``MODEL_CHOICES`` / ``MODELS_NO_SCALING``
registries. Every branch returns a ``BaseClassifier`` subclass so all models
share the same ``fit`` / ``predict`` / ``save_model`` / ``load_model`` interface.
Also exposes ``_save_final_model`` for persisting the single production model
used for Test predictions.
"""

import pickle
from pathlib import Path

# ┏━━━━━━━━━━ Model registry ━━━━━━━━━━┓
MODEL_CHOICES = ("rf", "xgboost", "autogluon", "tabpfn", "tabpfn_ft", "tabicl", "tabm")

# ┏━━━━━━━━━━ Models that must NOT receive StandardScaler-transformed data ━━━━━━━━━━┓
# TabM benefits from standardised inputs (it's an MLP), so it stays out of this set.
MODELS_NO_SCALING = {"tabpfn", "tabpfn_ft", "tabicl"}

# ┏━━━━━━━━━━ TabPFN soft row limit ━━━━━━━━━━┓
_TABPFN_MAX_ROWS = 50_000

# ┏━━━━━━━━━━ AutoGluon defaults ━━━━━━━━━━┓
_AG_TIME_LIMIT = 3600          # overridden by --ag-time-limit
_AG_PRESETS = "best_quality"   # overridden by --ag-presets


def _build_tree_model(model_name:         str,
                      n_samples:          int,
                      class_weight_ratio: float = 1.0,
                      feature_names=      None,
                      time_limit=         None,
                      presets:            str = "best_quality",
                      params:             dict | None = None):
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
    params : dict | None
        Best hyperparameters from HPO (overrides defaults for rf/tabpfn/tabicl).
    """
    # ┏━━━━━━━━━━ Random Forest ━━━━━━━━━━┓
    if model_name == "rf":
        from Utils.classifier.random_forest_classifier import RFClassifier
        defaults = {"n_estimators":     500,
                    "max_depth":        6,
                    "min_samples_leaf": 20,
                    "min_samples_split": 2,
                    "max_features":     "sqrt",
                    "class_weight":     "balanced"}
        if params:
            defaults.update({k: v for k, v in params.items() if k in defaults})
        return RFClassifier(**defaults,
                            random_state = 42,
                            n_jobs       = -1)

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
        if params:
            return TabPFN(n_estimators           = params.get("n_estimators", 4),
                          softmax_temperature    = params.get("softmax_temperature", 0.9),
                          balance_probabilities  = params.get("balance_probabilities", False),
                          average_before_softmax = params.get("average_before_softmax", False),
                          inference_config       = params.get("inference_config"),
                          random_state           = 42)
        return TabPFN()

    # ┏━━━━━━━━━━ TabPFN (fine-tuned) ━━━━━━━━━━┓
    elif model_name == "tabpfn_ft":
        from Utils.classifier.tabpfn_finetuned_classifier import TabPFNFineTuned
        return TabPFNFineTuned()

    # ┏━━━━━━━━━━ TabICL (in-context learning) ━━━━━━━━━━┓
    elif model_name == "tabicl":
        from Utils.classifier.tabicl_classifier import TabICL
        if params:
            return TabICL(n_estimators        = params.get("n_estimators", 8),
                          softmax_temperature = params.get("softmax_temperature", 0.9),
                          random_state        = 42)
        return TabICL()

    # ┏━━━━━━━━━━ TabM (deep tabular MLP-Mixer ensemble) ━━━━━━━━━━┓
    elif model_name == "tabm":
        from Utils.classifier.tabm_classifier import TabMClassifier
        if params:
            return TabMClassifier(k            = params.get("k", 32),
                                  n_blocks     = params.get("n_blocks", 2),
                                  d_block      = params.get("d_block", 256),
                                  lr           = params.get("lr", 2e-3),
                                  weight_decay = params.get("weight_decay", 3e-4),
                                  dropout      = params.get("dropout", 0.1),
                                  arch_type    = params.get("arch_type", "tabm"),
                                  n_bins       = params.get("n_bins"),
                                  d_embedding  = params.get("d_embedding"),
                                  random_state = 42)
        return TabMClassifier(random_state=42)

    else:
        raise ValueError(f"Unknown model: {model_name!r}. Choose from {MODEL_CHOICES}")


# ------------------------------------------------------------------------------
# Final production-model persistence
# ------------------------------------------------------------------------------
def _save_final_model(artifacts: dict,
                      save_dir: Path,
                      model_name: str,
                      features_used: list,
                      best_params: dict | None,
                      meta: dict) -> None:
    """Persist the final train+cal+opt-fitted model used for Test predictions.

    Excludes CPCV fold models, seed-experiment models, and per-trial HPO models.
    Only the single production model returned by ``temporal_eval(all features)``
    is saved, together with the pre-processing scaler, isotonic calibrator,
    feature list, chosen threshold, and the HPO params that built it.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    model = artifacts["model"]
    bundle = {"scaler":        artifacts["scaler"],
              "calibrator":    artifacts["calibrator"],
              "col_indices":   artifacts["col_indices"],
              "val_op":        artifacts["val_op"],
              "features_used": features_used,
              "model_name":    model_name,
              "best_params":   best_params,
              "meta":          meta}

    # ┏━━━━━━━━━━ Model payload (format depends on classifier type) ━━━━━━━━━━┓
    if model_name == "autogluon":
        model.save_to(save_dir)                          # AutoGluon writes its own directory layout
        model.save_best_hyperparameters(save_dir)        # Save best model's params for CPCV reuse
    else:
        try:
            model.save_model(str(save_dir / "model"))
        except Exception as e:                   # fallback when classifier lacks a native serializer
            print(f"    [save] native save_model failed ({e}); falling back to pickle")
            with open(save_dir / "model.pkl", "wb") as f:
                pickle.dump(model, f)

    with open(save_dir / "bundle.pkl", "wb") as f:
        pickle.dump(bundle, f)
    print(f"    Final model saved to {save_dir}")