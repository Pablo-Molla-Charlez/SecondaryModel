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


# ┏━━━━━━━━━━ TabM ━━━━━━━━━━┓
def _suggest_tabm(trial: optuna.Trial) -> dict:
    """Suggest TabM hyperparameters.

    Mirrors the search space in the official TabM repository
    (https://github.com/yandex-research/tabm). When the optional
    PiecewiseLinearEmbeddings module is enabled (``use_plr = True``), n_blocks
    is restricted to [1, 4] and the n_bins / d_embedding axes are sampled.
    Otherwise the plain TabM space (n_blocks ∈ [1, 5]) is used.
    """
    # ┏━━━━━━━━━━ Piecewise Linear Embeddings (PLE) ━━━━━━━━━━┓
    use_plr = trial.suggest_categorical("use_plr", [False, True])

    # ┏━━━━━━━━━━ k is fixed at 32 in the paper (not tuned) ━━━━━━━━━━┓
    params = {"k":            32,
              "d_block":      trial.suggest_int("d_block", 64, 1024, step=16),
              "lr":           trial.suggest_float("lr", 1e-4, 5e-3, log=True),
              "use_plr":      use_plr,
              "arch_type":    "tabm"}
              
    # ┏━━━━━━━━━━ weight_decay is sampled from {0} ∪ LogUniform[1e-4, 1e-1] ━━━━━━━━━━┓
    # Store the resolved float directly so best_params.json always has "weight_decay".
    if trial.suggest_categorical("weight_decay_zero", [True, False]):
        params["weight_decay"] = 0.0
    else:
        params["weight_decay"] = trial.suggest_float("weight_decay_val", 1e-4, 1e-1, log=True)

    # ┏━━━━━━━━━━ n_blocks, n_bins, d_embedding ━━━━━━━━━━┓
    if use_plr:
        params["n_blocks"]    = trial.suggest_int("n_blocks_plr", 1, 4)
        params["n_bins"]      = trial.suggest_int("n_bins", 2, 128)
        params["d_embedding"] = trial.suggest_int("d_embedding", 8, 32, step=4)
    else:
        params["n_blocks"]    = trial.suggest_int("n_blocks", 1, 5)
        params["n_bins"]      = None
        params["d_embedding"] = None
    
    return params


# ┏━━━━━━━━━━ CTTS ━━━━━━━━━━┓
def _suggest_ctts(trial: optuna.Trial) -> dict:
    """Suggest CTTS hyperparameters.

    The architecture knobs (CNN channels, Transformer depth) are kept at
    reasonable ranges.  The training regime (lr, weight_decay, batch_size)
    has the most impact and is given the widest search ranges.
    """
    return {"cnn_embed_dim": [trial.suggest_categorical("cnn_c1", [32, 64]),
                              trial.suggest_categorical("cnn_c2", [64, 128, 256])],
            "cnn_kernel":    [trial.suggest_categorical("cnn_k1", [5, 7, 9]),
                              trial.suggest_categorical("cnn_k2", [3, 5])],
            "cnn_stride":    [2, 1],
            "trans_heads":   trial.suggest_categorical("trans_heads", [2, 4, 8]),
            "trans_layers":  trial.suggest_int("trans_layers", 1, 4),
            "trans_ff":      trial.suggest_categorical("trans_ff", [128, 256, 512]),
            "trans_dropout": trial.suggest_float("trans_dropout", 0.0, 0.3, step=0.05),
            "mlp_hidden":    trial.suggest_categorical("mlp_hidden", [64, 128, 256]),
            "mlp_dropout":   trial.suggest_float("mlp_dropout", 0.0, 0.4, step=0.05),
            "mlp_pooling":   trial.suggest_categorical("mlp_pooling", ["attention", "meanmax"]),
            "lr":            trial.suggest_float("lr", 1e-5, 5e-3, log=True),
            "weight_decay":  trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True),
            "batch_size":    trial.suggest_categorical("batch_size", [64, 128, 256])}


_SUGGEST_FN = {"rf":     _suggest_rf,
               "tabpfn": _suggest_tabpfn,
               "tabicl": _suggest_tabicl,
               "tabm":   _suggest_tabm,
               "ctts":   _suggest_ctts}


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

    # ┏━━━━━━━━━━ TabM ━━━━━━━━━━┓
    elif model_name == "tabm":
        from Utils.classifier import TabMClassifier
        # weight_decay is always stored as a resolved float in best_params.json;
        # fall back to 0 for backward-compat with old JSONs that only have weight_decay_zero.
        wd = params.get("weight_decay", 0.0 if params.get("weight_decay_zero", True) else 3e-4)
        return TabMClassifier(k             = params.get("k", 32),
                              n_blocks      = params["n_blocks"],
                              d_block       = params["d_block"],
                              lr            = params["lr"],
                              weight_decay  = wd,
                              dropout       = params.get("dropout", 0.1),
                              arch_type     = params.get("arch_type", "tabm"),
                              n_bins        = params.get("n_bins"),
                              d_embedding   = params.get("d_embedding"),
                              random_state  = seed)

    # ┏━━━━━━━━━━ CTTS ━━━━━━━━━━┓
    elif model_name == "ctts":
        from Utils.classifier.ctts_classifier import CTTSClassifier
        return CTTSClassifier(
            cnn_embed_dim = params.get("cnn_embed_dim", [64, 128]),
            cnn_kernel    = params.get("cnn_kernel", [7, 5]),
            cnn_stride    = params.get("cnn_stride", [2, 1]),
            trans_heads   = params.get("trans_heads", 4),
            trans_layers  = params.get("trans_layers", 2),
            trans_ff      = params.get("trans_ff", 256),
            trans_dropout = params.get("trans_dropout", 0.1),
            mlp_hidden    = params.get("mlp_hidden", 128),
            mlp_dropout   = params.get("mlp_dropout", 0.1),
            mlp_pooling   = params.get("mlp_pooling", "attention"),
            lr            = params.get("lr", 1e-4),
            weight_decay  = params.get("weight_decay", 1e-4),
            batch_size    = params.get("batch_size", 128),
            random_state  = seed)

    else:
        raise ValueError(f"HPO not supported for model: {model_name}")

