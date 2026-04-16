"""Optuna suggest functions per model + param->model builder."""
import optuna

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Search spaces per model
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# ┏━━━━━━━━━━ Random Forest ━━━━━━━━━━┓
def _suggest_rf(trial: optuna.Trial) -> dict:
    """Suggest Random Forest hyperparameters."""

    return {"n_estimators":     trial.suggest_int("n_estimators", 100, 1000, step=100),
            "max_depth":        trial.suggest_int("max_depth", 3, 12),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 5, 50),
            "min_samples_split":trial.suggest_int("min_samples_split", 5, 50),
            "max_features":     trial.suggest_categorical("max_features", ["sqrt", "log2", 0.5, 0.7, 0.9]),
            "class_weight":     trial.suggest_categorical("class_weight", ["balanced", "balanced_subsample"])}

# ┏━━━━━━━━━━ TabPFN ━━━━━━━━━━┓
def _suggest_tabpfn(trial: optuna.Trial) -> dict:
    """Suggest TabPFN hyperparameters.

    Mirrors the official tabpfn-extensions `TunedTabPFN` search space
    (inference + preprocessing knobs — no gradient updates).
    Curated subset of preprocess transforms to keep the search tractable.
    """
    # ┏━━━━━━━━━━ Preprocess transform choices ━━━━━━━━━━┓
    # Each entry is a list of PreprocessorConfig dicts (TabPFN spec).
    preprocess_choices = {
        "quantile_uni_coarse":      [{"name": "quantile_uni_coarse",
                                      "global_transformer_name": None,
                                      "categorical_name": "numeric",
                                      "append_original": False}],
        "quantile_norm_coarse":     [{"name": "quantile_norm_coarse",
                                      "global_transformer_name": None,
                                      "categorical_name": "numeric",
                                      "append_original": False}],
        "kdi_alpha_0.3":            [{"name": "kdi_alpha_0.3",
                                      "global_transformer_name": None,
                                      "categorical_name": "numeric",
                                      "append_original": False}],
        "kdi_alpha_3.0":            [{"name": "kdi_alpha_3.0",
                                      "global_transformer_name": None,
                                      "categorical_name": "numeric",
                                      "append_original": False}],
        "none":                     [{"name": "none",
                                      "global_transformer_name": None,
                                      "categorical_name": "numeric",
                                      "append_original": False}],
        "safepower+quantile_uni":   [{"name": "safepower",
                                      "global_transformer_name": None,
                                      "categorical_name": "numeric",
                                      "append_original": False},
                                     {"name": "quantile_uni",
                                      "global_transformer_name": None,
                                      "categorical_name": "numeric",
                                      "append_original": False}],
        "squashing_scaler_default": [{"name": "squashing_scaler_default",
                                      "global_transformer_name": None,
                                      "categorical_name": "numeric",
                                      "append_original": False}],
    }

    # ┏━━━━━━━━━━ Preprocess transform ━━━━━━━━━━┓
    transform_key = trial.suggest_categorical("preprocess_transform",
                                              list(preprocess_choices.keys()))

    # ┏━━━━━━━━━━ inference_config block ━━━━━━━━━━┓
    inference_config = {"PREPROCESS_TRANSFORMS":             preprocess_choices[transform_key],
                        "FINGERPRINT_FEATURE":               trial.suggest_categorical("fingerprint_feature", [True, False]),
                        "OUTLIER_REMOVAL_STD":               trial.suggest_categorical("outlier_removal_std", [None, 7.0, 12.0]),
                        "MIN_UNIQUE_FOR_NUMERICAL_FEATURES": trial.suggest_categorical("min_unique_for_numerical", [1, 5, 10, 30]),
                        "POLYNOMIAL_FEATURES":               "no"}

    return {"n_estimators":           trial.suggest_int("n_estimators", 4, 32, step=4),
            "softmax_temperature":    trial.suggest_categorical("softmax_temperature", [0.75, 0.8, 0.9, 0.95, 1.0, 1.05]),
            "balance_probabilities":  trial.suggest_categorical("balance_probabilities", [True, False]),
            "average_before_softmax": trial.suggest_categorical("average_before_softmax", [True, False]),
            "inference_config":       inference_config}

# ┏━━━━━━━━━━ TabICL ━━━━━━━━━━┓
def _suggest_tabicl(trial: optuna.Trial) -> dict:
    """Suggest TabICL hyperparameters."""
    return {"n_estimators":        trial.suggest_int("n_estimators", 4, 32, step=4),
            "softmax_temperature": trial.suggest_float("softmax_temperature", 0.3, 1.5, step=0.1)}


_SUGGEST_FN = {"rf":     _suggest_rf,
               "tabpfn": _suggest_tabpfn,
               "tabicl": _suggest_tabicl}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Model builders from suggested params
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _build_model_from_params(model_name: str, params: dict, seed: int = 42):
    """Build an sklearn-compatible model from suggested hyperparameters."""
    # ┏━━━━━━━━━━ Random Forest ━━━━━━━━━━┓
    if model_name == "rf":
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(n_estimators      = params["n_estimators"],
                                      max_depth         = params["max_depth"],
                                      min_samples_leaf  = params["min_samples_leaf"],
                                      min_samples_split = params["min_samples_split"],
                                      max_features      = params["max_features"],
                                      class_weight      = params["class_weight"],
                                      random_state      = seed,
                                      n_jobs            = -1)

    # ┏━━━━━━━━━━ TabPFN ━━━━━━━━━━┓
    elif model_name == "tabpfn":
        from Utils.classifier import TabPFN
        return TabPFN(n_estimators           = params["n_estimators"],
                      softmax_temperature    = params["softmax_temperature"],
                      balance_probabilities  = params["balance_probabilities"],
                      average_before_softmax = params.get("average_before_softmax", False),
                      inference_config       = params.get("inference_config"),
                      random_state           = seed)

    # ┏━━━━━━━━━━ TabICL ━━━━━━━━━━┓
    elif model_name == "tabicl":
        from Utils.classifier import TabICL
        return TabICL(n_estimators        = params["n_estimators"],
                      softmax_temperature = params["softmax_temperature"],
                      random_state        = seed)

    else:
        raise ValueError(f"HPO not supported for model: {model_name}")
