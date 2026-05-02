"""Data split preparation + Optuna objective closure."""
import warnings
import numpy as np
import optuna
import torch

# ┏━━━━━━━━━━ Data Preprocessing ━━━━━━━━━━┓
from Utils.data import (resolve_feature_names,
                        split_by_global_time)

# ┏━━━━━━━━━━ Models ━━━━━━━━━━┓
from Utils.classifier import MODELS_NO_SCALING

# ┏━━━━━━━━━━ Selective Classification ━━━━━━━━━━┓
from Utils.selective_classification import (calibrate_probabilities,
                                            _find_best_utility_threshold)

# ┏━━━━━━━━━━ Edge ━━━━━━━━━━┓
from Utils.edge import _compute_embargo_splits

# ┏━━━━━━━━━━ Scaling ━━━━━━━━━━┓
from sklearn.preprocessing import StandardScaler

from Utils.hpo.search_spaces import _SUGGEST_FN, _build_model_from_params


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
        X_train, y_train, X_cal, y_cal, X_opt, y_opt, opt_returns, feature_names, fee, m1_prec_opt
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
    cal_returns = returns_all[idx_cal].copy()

    # ┏━━━━━━━━━━ Direction-aware returns ━━━━━━━━━━┓
    if direction.lower() == "down":
        opt_returns = -opt_returns
        cal_returns = -cal_returns

    # ┏━━━━━━━━━━ Drop NaN returns from Val-Opt ━━━━━━━━━━┓
    valid_opt = ~np.isnan(opt_returns)
    if not valid_opt.all():
        X_opt       = X_opt[valid_opt]
        y_opt       = y_opt[valid_opt]
        opt_returns = opt_returns[valid_opt]

    # ┏━━━━━━━━━━ M1 precision on Val-Opt (or merged Val for nocal) ━━━━━━━━━━┓
    # Computed on both windows here; caller picks the right one based on nocal flag.
    m1_prec_opt = None
    m1_prec_merged = None
    try:
        m1_pred = dataset.get("m1_pred_labels") if isinstance(dataset, dict) else getattr(dataset, "m1_pred_labels", None)
        m1_true = dataset.get("m1_true_labels") if isinstance(dataset, dict) else getattr(dataset, "m1_true_labels", None)
        if m1_pred is not None and m1_true is not None:
            if isinstance(m1_pred, torch.Tensor): m1_pred = m1_pred.numpy()
            if isinstance(m1_true, torch.Tensor): m1_true = m1_true.numpy()
            from sklearn.metrics import precision_score
            for idx_slice, attr_name in [(idx_opt, "m1_prec_opt"),
                                          (np.concatenate([idx_cal, idx_opt]), "m1_prec_merged")]:
                p = m1_pred[idx_slice]; t = m1_true[idx_slice]
                ok = ~np.isnan(p) & ~np.isnan(t)
                if ok.sum() > 0:
                    val = float(precision_score(t[ok].astype(int), p[ok].astype(int), zero_division=0))
                    if attr_name == "m1_prec_opt":
                        m1_prec_opt = val
                    else:
                        m1_prec_merged = val
    except Exception:
        pass

    return X_train, y_train, X_cal, y_cal, X_opt, y_opt, opt_returns, cal_returns, feature_names, fee, m1_prec_opt, m1_prec_merged


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
                      seed: int = 42,
                      m1_precision: float = None,
                      nocal: bool = False,
                      cal_returns: np.ndarray = None):
    """Create an Optuna objective closure.

    Standard (nocal=False):
      1. Fit model on Train
      2. Calibrate on Val-Cal (isotonic regression)
      3. Optimize threshold on calibrated Val-Opt
      4. Return utility score as the objective

    NoCal (nocal=True):
      1. Fit model on Train
      2. Skip calibration — use raw probs directly
      3. Optimize threshold on merged Val-Cal + Val-Opt (same as kronos_tree nocal)
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

    # ┏━━━━━━━━━━ Precompute merged Val window for nocal ━━━━━━━━━━┓
    if nocal and cal_returns is not None:
        _merged_y       = np.concatenate([y_cal,       y_opt])
        _merged_returns = np.concatenate([cal_returns,  opt_returns])
    else:
        _merged_y       = None
        _merged_returns = None

    def objective(trial: optuna.Trial) -> float:
        # ┏━━━━━━━━━━ Suggest hyperparameters ━━━━━━━━━━┓
        params = suggest_fn(trial)

        # ┏━━━━━━━━━━ Build and fit model ━━━━━━━━━━┓
        try:
            model = _build_model_from_params(model_name, params, seed=seed)
            # ┏━━━━━━━━━━ TabM gets the merged Val window for early stopping ━━━━━━━━━━┓
            # Same chronological tail used downstream by the threshold
            # optimiser; matches the protocol applied to AutoGluon.
            fit_kwargs = {}
            if model_name == "tabm":
                if nocal and _merged_y is not None:
                    fit_kwargs["X_eval"] = np.concatenate([X_cal_s, X_opt_s])
                    fit_kwargs["y_eval"] = _merged_y
                else:
                    fit_kwargs["X_eval"] = X_opt_s
                    fit_kwargs["y_eval"] = y_opt
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(X_train_s, y_train, **fit_kwargs)
        except Exception as e:
            print(f"  [Trial {trial.number}] Model fit failed: {e}")
            return float("-inf")

        # ┏━━━━━━━━━━ Threshold optimization ━━━━━━━━━━┓
        try:
            if nocal and _merged_y is not None:
                # ┏━━━━━━━━━━ NoCal: raw probs on merged Val-Cal + Val-Opt (mirrors kronos_tree nocal) ━━━━━━━━━━┓
                raw_cal_probs = model.predict_proba(X_cal_s)[:, 1]
                raw_opt_probs = model.predict_proba(X_opt_s)[:, 1]
                merged_probs  = np.concatenate([raw_cal_probs, raw_opt_probs])
                op = _find_best_utility_threshold(merged_probs,
                                                  _merged_returns,
                                                  fee          = fee,
                                                  labels       = _merged_y,
                                                  m1_precision = m1_precision)
            else:
                # ┏━━━━━━━━━━ Standard: calibrate on Val-Cal, sweep on calibrated Val-Opt ━━━━━━━━━━┓
                raw_cal_probs = model.predict_proba(X_cal_s)[:, 1]
                calib = calibrate_probabilities(raw_cal_probs, y_cal)
                calibrator = calib["calibrator"]
                raw_opt_probs = model.predict_proba(X_opt_s)[:, 1]
                cal_opt_probs = calibrator.predict(raw_opt_probs)
                op = _find_best_utility_threshold(cal_opt_probs,
                                                  opt_returns,
                                                  fee          = fee,
                                                  labels       = y_opt,
                                                  m1_precision = m1_precision)
        except Exception as e:
            print(f"  [Trial {trial.number}] Threshold optimization failed: {e}")
            return float("-inf")

        # ┏━━━━━━━━━━ Extract metrics ━━━━━━━━━━┓
        utility   = op.get("utility", float("-inf"))
        threshold = op.get("threshold", 0.5)
        coverage  = op.get("coverage", 1.0)
        sel_mean        = op.get("mean_ret", 0.0)
        threshold_source = op.get("threshold_source", "unknown")

        # ┏━━━━━━━━━━ Compute selective precision on eval window ━━━━━━━━━━┓
        _eval_probs  = merged_probs   if (nocal and _merged_y is not None) else cal_opt_probs
        _eval_labels = _merged_y      if (nocal and _merged_y is not None) else y_opt
        sel_mask = _eval_probs >= threshold
        n_sel = int(sel_mask.sum())
        sel_prec = float(_eval_labels[sel_mask].mean()) if n_sel > 0 else 0.0

        # ┏━━━━━━━━━━ Store trial attributes ━━━━━━━━━━┓
        trial.set_user_attr("threshold", float(threshold))
        trial.set_user_attr("threshold_source", threshold_source)
        trial.set_user_attr("coverage", float(coverage))
        trial.set_user_attr("sel_precision", float(sel_prec))
        trial.set_user_attr("sel_mean_return", float(sel_mean))
        trial.set_user_attr("n_selected", int(n_sel))
        trial.set_user_attr("n_opt_total", int(len(_eval_labels)))

        return utility

    return objective
