"""Hyperparameter Optimization for M2 tabular reliability estimators.

Uses Optuna Bayesian optimization to tune Random Forest, TabPFN, and TabICL
hyperparameters for each granularity and direction.  The objective is the
utility score on the Val-Opt split (after isotonic calibration on Val-Cal),
which mirrors the exact same calibration-first pipeline used in kronos_tree.py.

Output structure:
    Output/<M1_bucket>/HPO/<model>/<direction>/<granularity>/
        best_params.json        — optimal hyperparameters
        study_history.csv       — all trial results
        optuna_study.db         — Optuna SQLite storage (resume-friendly)

Usage:
    python Utils/HPO.py --config config_kronos.yaml
    python Utils/HPO.py --config config_kronos.yaml --models rf tabpfn
    python Utils/HPO.py --config config_kronos.yaml --models tabicl --n-trials 50
    python Utils/HPO.py --config config_kronos.yaml --directions up --grans 4h 6h 8h
    python Utils/HPO.py --config config_kronos.yaml --cache /path/to/cache_dir_or_file.pt
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import optuna
import torch

# ┏━━━━━━━━━━ Paths ━━━━━━━━━━┓
_SRC = Path(__file__).resolve().parent.parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

# ┏━━━━━━━━━━ Data Preprocessing ━━━━━━━━━━┓
from Utils.data_preprocessing import (resolve_feature_names,
                                      split_by_global_time)

# ┏━━━━━━━━━━ Models ━━━━━━━━━━┓
from Utils.models import MODELS_NO_SCALING

# ┏━━━━━━━━━━ Selective Classification ━━━━━━━━━━┓
from Utils.selective_classification import (calibrate_probabilities,
                                            _find_best_utility_threshold)

# ┏━━━━━━━━━━ Utils ━━━━━━━━━━┓
from Utils.utils import (_load_config,
                         _resolve_caches,
                         _load_multi_cache,
                         _infer_direction,
                         m1_output_bucket)

# ┏━━━━━━━━━━ Edge ━━━━━━━━━━┓
from Utils.edge import _compute_embargo_splits

# ┏━━━━━━━━━━ Scaling ━━━━━━━━━━┓
from sklearn.preprocessing import StandardScaler

# ┏━━━━━━━━━━ Constants ━━━━━━━━━━┓
ALL_GRANS  = ["1d", "12h", "8h", "6h", "4h", "2h", "1h", "30m"]
DIRECTIONS = ["up", "down"]
HPO_MODELS = ["rf", "tabpfn", "tabicl"]


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
        from Utils.models import _TabPFNWrapper
        return _TabPFNWrapper(n_estimators           = params["n_estimators"],
                              softmax_temperature    = params["softmax_temperature"],
                              balance_probabilities  = params["balance_probabilities"],
                              average_before_softmax = params.get("average_before_softmax", False),
                              inference_config       = params.get("inference_config"),
                              random_state           = seed)

    # ┏━━━━━━━━━━ TabICL ━━━━━━━━━━┓
    elif model_name == "tabicl":
        from Utils.models import _TabICLWrapper
        return _TabICLWrapper(n_estimators        = params["n_estimators"],
                              softmax_temperature = params["softmax_temperature"],
                              random_state        = seed)

    else:
        raise ValueError(f"HPO not supported for model: {model_name}")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Data loading (per granularity)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _load_dataset_for_gran(multi_cache, granularity: str) -> dict:
    """Extract a single-granularity dataset from a multi-granularity cache."""
    if hasattr(multi_cache, "sub"):
        # ┏━━━━━━━━━━ MultiGranDataset ━━━━━━━━━━┓
        if granularity not in multi_cache.sub:
            raise ValueError(f"Granularity '{granularity}' not found in cache. Available: {list(multi_cache.sub.keys())}")
        return multi_cache.sub[granularity]
    else:
        # ┏━━━━━━━━━━ Single-granularity dataset ━━━━━━━━━━┓
        return multi_cache


def _prepare_splits(dataset: dict, cfg: dict, granularity: str, direction: str):
    """Prepare 4-way embargo splits and extract feature/label/return arrays.

    Returns:
        X_train, y_train, X_cal, y_cal, X_opt, y_opt, opt_returns, feature_names
    """
    # ┏━━━━━━━━━━ Config ━━━━━━━━━━┓
    split_cfg = cfg.get("data", {}).get("split", {})
    train_end = split_cfg["train_end"]
    val_end   = split_cfg["val_end"]
    fh        = int(cfg.get("data", {}).get("load", {}).get("forecast_horizon", 7))
    fee       = cfg.get("evaluation", {}).get("fee_per_trade", 0.002)

    # ┏━━━━━━━━━━ Raw arrays to numpy ━━━━━━━━━━┓
    eng = dataset["eng_features"]
    if isinstance(eng, torch.Tensor):
        eng = eng.numpy()
    labels = dataset["labels"]
    if isinstance(labels, torch.Tensor):
        labels = labels.numpy()
    returns_all = dataset["returns"]
    if isinstance(returns_all, torch.Tensor):
        returns_all = returns_all.numpy()
    dates = dataset["dates"]

    # ┏━━━━━━━━━━ Valid mask (non-NaN labels) ━━━━━━━━━━┓
    valid_mask = ~np.isnan(labels)
    dates_valid = [dates[i] for i in range(len(dates)) if valid_mask[i]]
    valid_indices = np.where(valid_mask)[0]

    # ┏━━━━━━━━━━ 4-way embargo split ━━━━━━━━━━┓
    embargo = _compute_embargo_splits(dates_valid, train_end, val_end, fh, granularity)
    idx_train = valid_indices[embargo["idx_train"]]
    idx_cal   = valid_indices[embargo["idx_cal"]]
    idx_opt   = valid_indices[embargo["idx_opt"]]

    # ┏━━━━━━━━━━ Feature names ━━━━━━━━━━┓
    feature_names = resolve_feature_names(eng.shape[1])

    # ┏━━━━━━━━━━ Extract arrays ━━━━━━━━━━┓
    X_train = eng[idx_train]
    y_train = labels[idx_train].astype(int)
    X_cal   = eng[idx_cal]
    y_cal   = labels[idx_cal].astype(int)
    X_opt   = eng[idx_opt]
    y_opt   = labels[idx_opt].astype(int)
    opt_returns = returns_all[idx_opt].copy()

    # ┏━━━━━━━━━━ Direction-aware returns ━━━━━━━━━━┓
    if direction.lower() == "down":
        opt_returns = -opt_returns

    # ┏━━━━━━━━━━ Drop NaN returns from Val-Opt ━━━━━━━━━━┓
    valid_opt = ~np.isnan(opt_returns)
    if not valid_opt.all():
        X_opt       = X_opt[valid_opt]
        y_opt       = y_opt[valid_opt]
        opt_returns = opt_returns[valid_opt]

    return X_train, y_train, X_cal, y_cal, X_opt, y_opt, opt_returns, feature_names, fee


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Objective function
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _create_objective(model_name: str,
                      X_train: np.ndarray,
                      y_train: np.ndarray,
                      X_cal: np.ndarray,
                      y_cal: np.ndarray,
                      X_opt: np.ndarray,
                      y_opt: np.ndarray,
                      opt_returns: np.ndarray,
                      fee: float,
                      seed: int = 42):
    """Create an Optuna objective closure.

    The objective mirrors the kronos_tree.py calibration-first pipeline:
      1. Fit model on Train
      2. Calibrate on Val-Cal (isotonic regression)
      3. Optimize threshold on Val-Opt (utility function)
      4. Return utility score as the objective
    """
    suggest_fn = _SUGGEST_FN[model_name]
    needs_scaling = model_name not in MODELS_NO_SCALING

    # ┏━━━━━━━━━━ Pre-fit scaler once ━━━━━━━━━━┓
    scaler = StandardScaler()
    if needs_scaling:
        X_train_s = scaler.fit_transform(X_train)
        X_cal_s   = scaler.transform(X_cal)
        X_opt_s   = scaler.transform(X_opt)
    else:
        X_train_s = X_train
        X_cal_s   = X_cal
        X_opt_s   = X_opt

    def objective(trial: optuna.Trial) -> float:
        # ┏━━━━━━━━━━ Suggest hyperparameters ━━━━━━━━━━┓
        params = suggest_fn(trial)

        # ┏━━━━━━━━━━ Build and fit model ━━━━━━━━━━┓
        try:
            model = _build_model_from_params(model_name, params, seed=seed)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(X_train_s, y_train)
        except Exception as e:
            print(f"  [Trial {trial.number}] Model fit failed: {e}")
            return float("-inf")

        # ┏━━━━━━━━━━ Calibrate on Val-Cal ━━━━━━━━━━┓
        try:
            raw_cal_probs = model.predict_proba(X_cal_s)[:, 1]
            calib = calibrate_probabilities(raw_cal_probs, y_cal)
            calibrator = calib["calibrator"]
        except Exception as e:
            print(f"  [Trial {trial.number}] Calibration failed: {e}")
            return float("-inf")

        # ┏━━━━━━━━━━ Threshold optimization on Val-Opt ━━━━━━━━━━┓
        try:
            raw_opt_probs = model.predict_proba(X_opt_s)[:, 1]
            cal_opt_probs = calibrator.predict(raw_opt_probs)
            op = _find_best_utility_threshold(cal_opt_probs, opt_returns,
                                              fee=fee, labels=y_opt)
        except Exception as e:
            print(f"  [Trial {trial.number}] Threshold optimization failed: {e}")
            return float("-inf")

        # ┏━━━━━━━━━━ Extract metrics ━━━━━━━━━━┓
        utility   = op.get("utility", float("-inf"))
        threshold = op.get("threshold", 0.5)
        coverage  = op.get("coverage", 1.0)
        sel_mean  = op.get("sel_mean_return", 0.0)
        sel_sharpe = op.get("sel_sharpe", 0.0)

        # ┏━━━━━━━━━━ Compute selective precision on Val-Opt ━━━━━━━━━━┓
        sel_mask = cal_opt_probs >= threshold
        n_sel = int(sel_mask.sum())
        sel_prec = float(y_opt[sel_mask].mean()) if n_sel > 0 else 0.0

        # ┏━━━━━━━━━━ Store trial attributes ━━━━━━━━━━┓
        trial.set_user_attr("threshold", float(threshold))
        trial.set_user_attr("coverage", float(coverage))
        trial.set_user_attr("sel_precision", float(sel_prec))
        trial.set_user_attr("sel_mean_return", float(sel_mean))
        trial.set_user_attr("sel_sharpe", float(sel_sharpe))
        trial.set_user_attr("n_selected", int(n_sel))
        trial.set_user_attr("n_opt_total", int(len(y_opt)))

        return utility

    return objective


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Run HPO for one (model, direction, granularity) configuration
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def run_hpo_single(model_name: str,
                   direction: str,
                   granularity: str,
                   cfg: dict,
                   multi_cache,
                   output_root: Path,
                   n_trials: int = 100,
                   seed: int = 42) -> dict | None:
    """Run HPO for a single (model, direction, granularity) configuration."""
    # ┏━━━━━━━━━━ Output directory ━━━━━━━━━━┓
    m1_bucket = m1_output_bucket(cfg)
    out_dir = output_root / m1_bucket / "HPO" / model_name / direction.upper() / granularity
    out_dir.mkdir(parents=True, exist_ok=True)

    # ┏━━━━━━━━━━ Skip if already completed ━━━━━━━━━━┓
    best_path = out_dir / "best_params.json"
    if best_path.exists():
        print(f"  [SKIP] {model_name.upper()} {direction.upper()} {granularity} — best_params.json exists")
        with open(best_path) as f:
            return json.load(f)

    print(f"\n{'='*60}")
    print(f"  HPO: {model_name.upper()} | {direction.upper()} | {granularity}")
    print(f"  Output: {out_dir}")
    print(f"{'='*60}")

    # ┏━━━━━━━━━━ Load single-granularity dataset ━━━━━━━━━━┓
    try:
        dataset = _load_dataset_for_gran(multi_cache, granularity)
    except ValueError as e:
        print(f"  [SKIP] {e}")
        return None

    # ┏━━━━━━━━━━ Prepare splits ━━━━━━━━━━┓
    try:
        (X_train, y_train, 
         X_cal, y_cal,
         X_opt, y_opt, 
         opt_returns, 
         feature_names, 
         fee) = _prepare_splits(dataset, cfg, granularity, direction)
    except Exception as e:
        print(f"  [SKIP] Split preparation failed: {e}")
        return None

    print(f"  Train: {len(y_train):,}  Cal: {len(y_cal):,}  Opt: {len(y_opt):,}")
    print(f"  Fee: {fee}  Direction: {direction}")

    if len(y_train) < 100 or len(y_cal) < 20 or len(y_opt) < 50:
        print(f"  [SKIP] Insufficient data for HPO")
        return None

    # ┏━━━━━━━━━━ Create Optuna study ━━━━━━━━━━┓
    db_path = out_dir / "optuna_study.db"
    study_name = f"HPO_{model_name}_{direction}_{granularity}"
    storage = f"sqlite:///{db_path}"

    study = optuna.create_study(study_name     = study_name,
                                storage        = storage,
                                direction      = "maximize",
                                load_if_exists = True,
                                sampler        = optuna.samplers.TPESampler(seed=seed),
                                pruner         = optuna.pruners.NopPruner())

    # ┏━━━━━━━━━━ Create objective ━━━━━━━━━━┓
    objective = _create_objective(model_name  = model_name,
                                  X_train     = X_train,
                                  y_train     = y_train,
                                  X_cal       = X_cal,
                                  y_cal       = y_cal,
                                  X_opt       = X_opt,
                                  y_opt       = y_opt,
                                  opt_returns = opt_returns,
                                  fee         = fee,
                                  seed        = seed)

    # ┏━━━━━━━━━━ Remaining trials ━━━━━━━━━━┓
    completed = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    remaining = max(0, n_trials - completed)
    if remaining == 0:
        print(f"  Already completed {completed} trials — loading best.")
    else:
        print(f"  Running {remaining} trials ({completed} already completed)...")
        t0 = time.time()
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study.optimize(objective, n_trials=remaining, show_progress_bar=True)
        elapsed = time.time() - t0
        print(f"  Completed in {elapsed:.0f}s")

    # ┏━━━━━━━━━━ Extract best ━━━━━━━━━━┓
    best = study.best_trial
    result = {"model":       model_name,
              "direction":   direction,
              "granularity": granularity,
              "best_trial":  best.number,
              "best_utility": best.value,
              "best_params": best.params,
              "best_metrics": {k: v for k, v in best.user_attrs.items()},
              "n_trials":    len(study.trials)}

    # ┏━━━━━━━━━━ Save best params ━━━━━━━━━━┓
    with open(best_path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"  Best utility: {best.value:.4f}")
    print(f"  Best params: {best.params}")
    print(f"  Saved: {best_path}")

    # ┏━━━━━━━━━━ Save study history ━━━━━━━━━━┓
    history_path = out_dir / "study_history.csv"
    df = study.trials_dataframe()
    df.to_csv(history_path, index=False)

    return result


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Orchestrator: run HPO for all combinations
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def run_hpo(config: str,
            models: list[str],
            directions: list[str],
            grans: list[str],
            n_trials: int = 100,
            seed: int = 42,
            cache: str | None = None):
    """Run HPO for all (model x direction x granularity) combinations."""
    # ┏━━━━━━━━━━ Load config ━━━━━━━━━━┓
    cfg = _load_config(config)
    output_root = Path(cfg["paths"]["output_root"])

    # ┏━━━━━━━━━━ Print header ━━━━━━━━━━┓
    print(f"\n{'#'*70}")
    print(f"# M2 Hyperparameter Optimization")
    print(f"# Models:     {models}")
    print(f"# Directions: {directions}")
    print(f"# Grans:      {grans}")
    print(f"# Trials:     {n_trials}")
    print(f"# Config:     {config}")
    if cache:
        print(f"# Cache:      {cache}")
    print(f"{'#'*70}")

    # ┏━━━━━━━━━━ Load caches per direction ━━━━━━━━━━┓
    caches = {}
    if cache:
        cache_path = Path(cache)
        if cache_path.is_dir():
            # ┏━━━━━━━━━━ Directory mode ━━━━━━━━━━┓
            for direction in directions:
                candidates = sorted(cache_path.glob(f"*_{direction}_*.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
                if candidates:
                    print(f"\n  Loading multi-cache for {direction.upper()}: {candidates[0].name}")
                    caches[direction] = _load_multi_cache(candidates[0])
                else:
                    print(f"  [WARN] No {direction} cache found in {cache_path}")
        elif cache_path.is_file():
            # ┏━━━━━━━━━━ Single file mode: infer direction from filename ━━━━━━━━━━┓
            direction = _infer_direction(cache_path)
            print(f"\n  Loading multi-cache for {direction.upper()}: {cache_path.name}")
            caches[direction] = _load_multi_cache(cache_path)
        else:
            raise FileNotFoundError(f"Cache path not found: {cache_path}")
    else:
        for direction in directions:
            cache_map = _resolve_caches(cfg, explicit=None)
            if direction in cache_map:
                print(f"\n  Loading multi-cache for {direction.upper()}: {cache_map[direction].name}")
                caches[direction] = _load_multi_cache(cache_map[direction])
            else:
                print(f"  [WARN] No cache found for direction={direction}")

    # ┏━━━━━━━━━━ Run HPO for each combination ━━━━━━━━━━┓
    all_results = []
    total = len(models) * len(directions) * len(grans)
    i = 0

    # ┏━━━━━━━━━━ Run HPO for each combination ━━━━━━━━━━┓
    for model_name in models:
        for direction in directions:
            if direction not in caches:
                print(f"  [SKIP] No cache for direction={direction}")
                continue
            
            # ┏━━━━━━━━━━ Get cache for this direction ━━━━━━━━━━┓
            multi_cache = caches[direction]
            available_grans = list(multi_cache.sub.keys()) if hasattr(multi_cache, "sub") else []

            # ┏━━━━━━━━━━ Run HPO for each granularity ━━━━━━━━━━┓
            for gran in grans:
                i += 1
                print(f"\n  [{i}/{total}] {model_name.upper()} {direction.upper()} {gran}")

                if gran not in available_grans:
                    print(f"    [SKIP] Granularity {gran} not in cache (available: {available_grans})")
                    continue

                # ┏━━━━━━━━━━ Run HPO for single combination ━━━━━━━━━━┓
                result = run_hpo_single(model_name  = model_name,
                                        direction   = direction,
                                        granularity = gran,
                                        cfg         = cfg,
                                        multi_cache = multi_cache,
                                        output_root = output_root,
                                        n_trials    = n_trials,
                                        seed        = seed)

                if result:
                    all_results.append(result)

    # ┏━━━━━━━━━━ Summary ━━━━━━━━━━┓
    print(f"\n{'='*70}")
    print(f"HPO SUMMARY")
    print(f"{'='*70}")
    print(f"{'Model':<10} {'Dir':<6} {'Gran':<6} {'Utility':>10} {'Prec':>8} {'Cov':>8} {'Sharpe':>8} {'Thr':>8}")
    print(f"{'-'*70}")
    for r in all_results:
        m = r["best_metrics"]
        print(f"{r['model']:<10} {r['direction']:<6} {r['granularity']:<6} "
              f"{r['best_utility']:>10.4f} "
              f"{m.get('sel_precision', 0):>8.3f} "
              f"{m.get('coverage', 0):>8.3f} "
              f"{m.get('sel_sharpe', 0):>8.2f} "
              f"{m.get('threshold', 0.5):>8.3f}")
    print(f"{'='*70}")
    print(f"Completed {len(all_results)}/{total} configurations.")

    # ┏━━━━━━━━━━ Save global summary ━━━━━━━━━━┓
    m1_bucket = m1_output_bucket(cfg)
    summary_path = output_root / m1_bucket / "HPO" / "hpo_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"Summary saved: {summary_path}")

    return all_results


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CLI
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def main():
    # ┏━━━━━━━━━━ Parse arguments ━━━━━━━━━━┓
    parser = argparse.ArgumentParser(description="M2 Hyperparameter Optimization (RF, TabPFN, TabICL)")

    parser.add_argument("--config",     type=str,  required=True, help="Path to config YAML (e.g. config_kronos.yaml)")
    parser.add_argument("--models",     nargs="+", default=HPO_MODELS, choices=HPO_MODELS, help=f"Models to optimize (default: {HPO_MODELS})")
    parser.add_argument("--directions", nargs="+", default=DIRECTIONS,choices=DIRECTIONS, help="Directions to optimize (default: up down)")
    parser.add_argument("--grans",      nargs="+", default=ALL_GRANS, help=f"Granularities to optimize (default: {ALL_GRANS})")
    parser.add_argument("--n-trials",   type=int,  default=100, help="Number of Optuna trials per configuration (default: 100)")
    parser.add_argument("--seed",       type=int,  default=42, help="Random seed (default: 42)")
    parser.add_argument("--cache",      type=str,  default=None, help="Path to cache .pt file or directory containing up/down caches (skips auto-detection / rebuilding)")

    args = parser.parse_args()

    # ┏━━━━━━━━━━ Run HPO ━━━━━━━━━━┓
    run_hpo(config     = args.config,
            models     = args.models,
            directions = args.directions,
            grans      = args.grans,
            n_trials   = args.n_trials,
            seed       = args.seed,
            cache      = args.cache)


if __name__ == "__main__":
    main()