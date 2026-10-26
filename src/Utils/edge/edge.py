"""
Edge Analysis — Cross-Validation Evidence
========================================
Two modes, both writing a per-sample CSV that the regime analysis consumes:
  --mode cpcv     : Combinatorial Purged Cross-Validation (Lopez de Prado)
                    N=6 datetime blocks, k=2 test → C(6,2)=15 splits → 5 paths
  --mode cvexpand : expanding-window purged CV — same datetime blocks and time-based
                    purge, but walk-forward only (no combinatorics)

Supported models: rf, xgboost, autogluon, tabpfn, tabpfn_ft

Usage:
  python -m Utils.edge --cache path/to/multi_cache.pt --mode cpcv
  python -m Utils.edge --cache path/to/multi_cache.pt --mode cpcv --model tabpfn_ft
  python -m Utils.edge --cache path/to/multi_cache.pt --mode cvexpand
"""
import math
import argparse, sys, json
from pathlib import Path
from itertools import combinations
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

_SRC = Path(__file__).resolve().parent.parent.parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

# ┏━━━━━━━━━━ Imports from Data Preprocessing ━━━━━━━━━━┓
from Utils.data import resolve_feature_names

# ┏━━━━━━━━━━ Imports from Selective Classification ━━━━━━━━━━┓
from Utils.selective_classification import _find_best_utility_threshold, calibrate_probabilities

# ┏━━━━━━━━━━ Imports from Models ━━━━━━━━━━┓
from Utils.classifier import _build_tree_model, MODELS_NO_SCALING

# ┏━━━━━━━━━━ Imports from Backtest ━━━━━━━━━━┓
from Utils.backtest import (_annualization_factor,
                            _build_spread_equity,
                            _calc_drawdown,
                            _calc_sharpe,
                            _equity_horizon_returns,
                            _load_raw_close_prices,
                            _plot_path_equity)

# ┏━━━━━━━━━━ Imports from Utils ━━━━━━━━━━┓
from Utils.utils import (_load_config,
                         _infer_direction,
                         _load_multi_cache,
                         _class_names,
                         m1_output_bucket,
                         m1_display_label,
                         _load_best_params)

# ┏━━━━━━━━━━ Imports from TS Cross-Validation ━━━━━━━━━━┓
from Utils.ts_cross_validation import (_gran_to_timedelta,
                                       _build_datetime_blocks,
                                       _assign_blocks,
                                       _generate_cpcv_splits,
                                       _reconstruct_paths,
                                       compute_embargo_splits as _compute_embargo_splits,
                                       CAL_SPLIT_RATIO)

# ┏━━━━━━━━━━ Imports from Edge Plots ━━━━━━━━━━┓
from .plots import (_plot_edge_curves,
                    _plot_summary_boxplots,
                    _plot_split_matrix,
                    _plot_path_boxplots,
                    _plot_cross_gran_cpcv)


# ┏━━━━━━━━━ Fixed seed for CPCV — variance measures regime sensitivity, not model noise ━━━━━━━━━━┓
EDGE_SEED = 42

# ┏━━━━━━━━━━ Calibration split ratios ━━━━━━━━━━┓
# [XGBoost, TabpPFN or others]
# First CAL_SPLIT_RATIO of Val (Calibration Set ~ Val-Cal) — imported from ts_cross_validation.
# Remainder = Val-Opt (Threshold Optimization Set).
#
# [Random Forest]
# First CPCV_OOB_CAL_RATIO of OOB Train = Val-Cal; remainder = Val-Opt (CPCV only).
CPCV_OOB_CAL_RATIO = 0.40



# ┏━━━━━━━━━━ Metrics to plot ━━━━━━━━━━┓
METRICS_TO_PLOT = [("accuracy",      "Accuracy (@0.5)"),
                   ("sel_accuracy",  "Selective Accuracy"),
                   ("precision",     "Precision (@0.5)"),
                   ("sel_precision", "Selective Precision"),
                   ("mean_ret",      "Mean Ret (@0.5)"),
                   ("sel_mean_ret",  "Selective Mean Ret"),
                   ("sel_coverage",  "Selective Coverage")]


# ┏━━━━━━━━━━ Embargo helper: 4-way split with purge at all boundaries ━━━━━━━━━━┓
def _compute_embargo_splits(dates_valid, train_end, val_end, horizon, granularity):
    """Compute indices for Train / Val-Cal / Val-Opt / Test with embargo gaps.

    Purge = horizon x bar_width at each boundary to prevent label leakage.
    Cal/Opt split is determined by CAL_SPLIT_RATIO applied to the Val window.

    Returns dict with keys: idx_train, idx_cal, idx_opt, idx_test, cal_end, purge_td
    """
    # ┏━━━━━━━━━━ Granularity to timedelta ━━━━━━━━━━┓
    bar_td = _gran_to_timedelta(granularity)
    purge_td = bar_td * horizon

    # ┏━━━━━━━━━━ Train and Validation End Dates ━━━━━━━━━━┓
    t_train_end = pd.Timestamp(train_end)
    t_val_end   = pd.Timestamp(val_end)

    # ┏━━━━━━━━━━ Val-Cal End = 40% of the Val window ━━━━━━━━━━┓
    val_duration = t_val_end - t_train_end
    t_cal_end = t_train_end + CAL_SPLIT_RATIO * val_duration

    # ┏━━━━━━━━━━ Boundaries with embargo ━━━━━━━━━━┓
    #   Train:   date <= train_end
    #   (purge): train_end < date <= train_end + purge
    #   Val-Cal:     train_end + purge < date <= cal_end
    #   (purge): cal_end < date <= cal_end + purge
    #   Val-Opt:     cal_end + purge < date <= val_end
    #   (purge): val_end < date <= val_end + purge
    #   Test:    date > val_end + purge
    
    # ┏━━━━━━━━━━ Split indices ━━━━━━━━━━┓
    idx_train, idx_cal, idx_opt, idx_test = [], [], [], []
    for i, d in enumerate(dates_valid):
        # ┏━━━━━━━━━━ Train ━━━━━━━━━━┓
        if d <= t_train_end:
            idx_train.append(i)
        
        # ┏━━━━━━━━━━ Purge ━━━━━━━━━━┓
        elif d <= t_train_end + purge_td:
            continue  # purge zone
        
        # ┏━━━━━━━━━━ Val-Cal ━━━━━━━━━━┓
        elif d <= t_cal_end:
            idx_cal.append(i)
        
        # ┏━━━━━━━━━━ Purge ━━━━━━━━━━┓
        elif d <= t_cal_end + purge_td:
            continue  # purge zone
        
        # ┏━━━━━━━━━━ Val-Opt ━━━━━━━━━━┓
        elif d <= t_val_end:
            idx_opt.append(i)
        
        # ┏━━━━━━━━━━ Purge ━━━━━━━━━━┓
        elif d <= t_val_end + purge_td:
            continue  # purge zone
        else:
            idx_test.append(i)

    return {"idx_train": np.array(idx_train),
            "idx_cal":   np.array(idx_cal),
            "idx_opt":   np.array(idx_opt),
            "idx_test":  np.array(idx_test),
            "cal_end":   t_cal_end,
            "purge_td":  purge_td}


# ┏━━━━━━━━━━ Build model for edge analysis ━━━━━━━━━━┓
def _build_edge_model(model_key: str,
                      n_samples: int,
                      seed: int = EDGE_SEED,
                      class_weight_ratio: float = 1.0,
                      best_params: dict | None = None,
                      feature_names=None):
    """Build a model via _build_tree_model, then override random_state for seed control.

    For TabPFN and TabICL, ``best_params`` from HPO are forwarded so CPCV splits
    reuse the tuned hyperparameters instead of model defaults.
    AutoGluon is NOT built here — it is handled separately in _run_cpcv_split
    via ``fit_with_hyperparameters`` to avoid the 3600s search per split.
    """
    # ┏━━━━━━━━━━ Build the classifier ━━━━━━━━━━┓
    model = _build_tree_model(model_key,
                              n_samples          = n_samples,
                              class_weight_ratio = class_weight_ratio,
                              feature_names      = feature_names,
                              params             = best_params)

    # ┏━━━━━━━━━━ Override seed for reproducibility / seed-trial variance measurement ━━━━━━━━━━┓
    if hasattr(model, "random_state"):
        model.random_state = seed

    # ┏━━━━━━━━━━ Propagate seed to internal sklearn/xgb classifier (already constructed) ━━━━━━━━━━┓
    if hasattr(model, "_clf") and hasattr(model._clf, "random_state"):
        model._clf.random_state = seed

    # ┏━━━━━━━━━━ RF also needs oob_score for CPCV threshold estimation ━━━━━━━━━━┓
    if model_key == "rf":
        if hasattr(model, "_clf"):
            model._clf.oob_score = True
        model.oob_score = True
    return model


# ┏━━━━━━━━━━ M1 baselines Metrics ━━━━━━━━━━┓
def _compute_m1_baselines(sub: dict,
                          dates_all: list,
                          train_end: str,
                          val_end: str,
                          returns_valid: np.ndarray,
                          idx_val: np.ndarray,
                          idx_test: np.ndarray,
                          direction: str,
                          fee: float) -> dict:
    """Compute M1 accuracy, precision, and mean return on val/test splits.

    M1 acc/prec use RAW indices (all windows, including NaN meta-labels)
    to match m2_pipeline behavior. M1 mean return uses the valid-filtered
    returns at idx_val/idx_test.
    """
    # ┏━━━━━━━━━━ Get M1 predictions and true labels ━━━━━━━━━━┓
    m1_pred = sub.get("m1_pred_labels")
    m1_true = sub.get("m1_true_labels")
    if hasattr(sub, "m1_pred_labels"):
        m1_pred = sub.m1_pred_labels
        m1_true = sub.m1_true_labels

    # ┏━━━━━━━━━━ Initialize result ━━━━━━━━━━┓
    baselines = {"val":  {"m1_acc": None, "m1_prec": None, "m1_mean_ret": None},
                 "test": {"m1_acc": None, "m1_prec": None, "m1_mean_ret": None}}

    # ┏━━━━━━━━━━ M1 acc/prec: use ALL windows (raw, no NaN meta-label filter) ━━━━━━━━━━┓
    if m1_pred is not None and m1_true is not None:
        # ┏━━━━━━━━━━ Convert M1 predictions and true labels to numpy arrays ━━━━━━━━━━┓
        if isinstance(m1_pred, torch.Tensor):
            m1_pred = m1_pred.numpy()
        if isinstance(m1_true, torch.Tensor):
            m1_true = m1_true.numpy()

        # ┏━━━━━━━━━━ Get raw indices for validation and test sets ━━━━━━━━━━┓
        t_train_end = pd.Timestamp(train_end)
        t_val_end   = pd.Timestamp(val_end)
        idx_val_raw, idx_test_raw = [], []
        for i, d in enumerate(dates_all):
            if d <= t_train_end:
                continue
            elif d <= t_val_end:
                idx_val_raw.append(i)
            else:
                idx_test_raw.append(i)

        # ┏━━━━━━━━━━ Compute M1 acc/prec on raw indices ━━━━━━━━━━┓
        for split_name, idx_raw in [("val", idx_val_raw), ("test", idx_test_raw)]:
            if len(idx_raw) == 0:
                continue
            p = m1_pred[idx_raw]
            t = m1_true[idx_raw]
            ok = ~np.isnan(p) & ~np.isnan(t)
            if ok.sum() > 0:
                baselines[split_name]["m1_acc"]  = float(accuracy_score(t[ok], p[ok]))
                baselines[split_name]["m1_prec"] = float(precision_score(t[ok], p[ok], zero_division=0))

    # ┏━━━━━━━━━━ M1 mean return: avg net return of all M1 trades (valid-filtered) ━━━━━━━━━━┓
    for split_name, idx in [("val", idx_val), ("test", idx_test)]:
        split_rets = returns_valid[idx].copy()
        if direction.lower() == "down":
            split_rets = -split_rets
        baselines[split_name]["m1_mean_ret"] = float(np.nanmean(split_rets - fee))

    return baselines


# ┏━━━━━━━━━━ M1 baselines Metrics on Raw Indices ━━━━━━━━━━┓
def _compute_m1_baseline_on_raw_indices(sub: dict, idx_raw: list) -> dict:
    """Compute M1 acc/prec on a specific set of raw (unfiltered) indices."""
    # ┏━━━━━━━━━━ Get M1 predictions and true labels ━━━━━━━━━━┓
    m1_pred = sub.get("m1_pred_labels")
    m1_true = sub.get("m1_true_labels")
    if hasattr(sub, "m1_pred_labels"):
        m1_pred = sub.m1_pred_labels
        m1_true = sub.m1_true_labels

    # ┏━━━━━━━━━━ Initialize result ━━━━━━━━━━┓
    result = {"m1_acc": None, "m1_prec": None}
    if m1_pred is None or m1_true is None or len(idx_raw) == 0:
        return result
    if isinstance(m1_pred, torch.Tensor):
        m1_pred = m1_pred.numpy()
    if isinstance(m1_true, torch.Tensor):
        m1_true = m1_true.numpy()

    # ┏━━━━━━━━━━ Extract M1 predictions and true labels on raw indices ━━━━━━━━━━┓
    p = m1_pred[idx_raw]
    t = m1_true[idx_raw]
    ok = ~np.isnan(p) & ~np.isnan(t)

    # ┏━━━━━━━━━━ Compute M1 acc/prec on raw indices ━━━━━━━━━━┓
    if ok.sum() > 0:
        result["m1_acc"]  = float(accuracy_score(t[ok], p[ok]))
        result["m1_prec"] = float(precision_score(t[ok], p[ok], zero_division=0))
    return result


# ┏━━━━━━━━━━ Helper to run CPCV split (with & without OOB-calibrated threshold) ━━━━━━━━━━┓
def _run_cpcv_split(eng,
                    labels,
                    returns,
                    asset_ids,
                    asset_map,
                    idx_train,
                    idx_test,
                    direction,
                    fee,
                    model_key          = "rf",
                    dates_valid        = None,
                    best_params        = None,
                    feature_names      = None,
                    nocal              = False):
    """Train on train, calibrate + sweep threshold on training OOB, evaluate on test.

    RF path (OOB available):
        1. model.fit(full X_train)
        2. Split OOB probs → Val-cal (first 40%) + Val-Opt (last 60%)
        3. isotonic.fit(oob_cal, y_cal)
        4. τ* = sweep(isotonic(oob_opt), returns_opt)
        5. Apply isotonic + τ* to test

    Non-RF path (no OOB):
        1. Split train → fit (60%), Val-cal (20%), Val-Opt (20%)
        2. model.fit(X_fit)
        3. isotonic.fit(predict(X_cal), y_cal)
        4. τ* = sweep(isotonic(predict(X_opt)), returns_opt)
        5. Apply isotonic + τ* to test
    """
    # ┏━━━━━━━━━━ Extract data ━━━━━━━━━━┓
    X_train_all = eng[idx_train]
    y_train_all = labels[idx_train].astype(int)
    X_test  = eng[idx_test]
    y_test  = labels[idx_test].astype(int)
    train_returns_all = returns[idx_train].copy()
    test_returns  = returns[idx_test].copy()

    test_assets = [asset_map.get(int(a), str(a)) for a in asset_ids[idx_test]]

    # ┏━━━━━━━━━━ Apply direction to returns ━━━━━━━━━━┓
    if direction.lower() == "down":
        train_returns_all = -train_returns_all
        test_returns  = -test_returns

    # ┏━━━━━━━━━━ Scale features (skip for TabPFN) ━━━━━━━━━━┓
    if model_key not in MODELS_NO_SCALING:
        scaler = StandardScaler()
        X_train_all = scaler.fit_transform(X_train_all)
        X_test  = scaler.transform(X_test)

    # ┏━━━━━━━━━━ Build class weights ━━━━━━━━━━┓
    n_pos = int(y_train_all.sum())
    n_neg = len(y_train_all) - n_pos
    cw_ratio = n_neg / max(n_pos, 1)

    # ┏━━━━━━━━━━ RF path: OOB-calibrated threshold ━━━━━━━━━━┓
    if model_key == "rf":
        # ┏━━━━━━━━━━ Build model ━━━━━━━━━━┓
        model = _build_edge_model(model_key,
                                  n_samples          = len(y_train_all),
                                  seed               = EDGE_SEED,
                                  class_weight_ratio = cw_ratio,
                                  best_params        = best_params)

        # ┏━━━━━━━━━━ Fit model ━━━━━━━━━━┓
        model.fit(X_train_all, y_train_all)

        # ┏━━━━━━━━━━ OOB probs for all training samples ━━━━━━━━━━┓
        if hasattr(model, "oob_decision_function_"):
            oob_probs = model.oob_decision_function_[:, 1]
        else:
            oob_probs = model.predict_proba(X_train_all)[:, 1]  # fallback
        
        # ┏━━━━━━━━━━ NoCal or Calibrated ━━━━━━━━━━┓
        if nocal:
            # ┏━━━━━━━━━━ NoCal: identity calibrator, sweep τ* on all OOB probs directly (no Cal/Opt split) ━━━━━━━━━━┓
            from Utils.selective_classification.calibration import _IdentityCalibrator
            calibrator = _IdentityCalibrator()
            op = _find_best_utility_threshold(oob_probs, train_returns_all, fee=fee, labels=y_train_all)
        else:
            # ┏━━━━━━━━━━ Chronological split of OOB into Val-cal (40%) / Val-Opt (60%) ━━━━━━━━━━┓
            n_train = len(idx_train)
            n_cal = int(n_train * CPCV_OOB_CAL_RATIO)
            cal_mask = np.zeros(n_train, dtype=bool)
            cal_mask[:n_cal] = True
            opt_mask = ~cal_mask

            # ┏━━━━━━━━━━ Fit isotonic on cal OOB ━━━━━━━━━━┓
            _cal = calibrate_probabilities(oob_probs[cal_mask], y_train_all[cal_mask])
            calibrator = _cal["calibrator"]

            # ┏━━━━━━━━━━ Sweep threshold on calibrated opt OOB ━━━━━━━━━━┓
            opt_probs_cal = calibrator.predict(oob_probs[opt_mask])
            op = _find_best_utility_threshold(opt_probs_cal, train_returns_all[opt_mask], fee=fee, labels=y_train_all[opt_mask])

    # ┏━━━━━━━━━━ AutoGluon path: reuse Phase-1 best model hyperparameters ━━━━━━━━━━┓
    elif model_key == "autogluon":
        from Utils.classifier.autogluon_classifier import AutoGluon

        n_train = len(idx_train)
        if nocal:
            # ┏━━━━━━━━━━ NoCal: split 75% fit / 25% opt (Val-Cal merged into Val-Opt → no cal slot) ━━━━━━━━━━┓
            n_fit   = int(n_train * 0.75)
            X_fit   = X_train_all[:n_fit]; y_fit = y_train_all[:n_fit]
            X_opt   = X_train_all[n_fit:]; y_opt = y_train_all[n_fit:]
            opt_rets = train_returns_all[n_fit:]
        else:
            # ┏━━━━━━━━━━ Split train → fit (60%), Val-cal (20%), Val-Opt (20%) ━━━━━━━━━━┓
            n_fit   = int(n_train * 0.60)
            n_cal   = int(n_train * 0.20)
            X_fit    = X_train_all[:n_fit]; y_fit = y_train_all[:n_fit]
            X_cal    = X_train_all[n_fit:n_fit+n_cal]; y_cal = y_train_all[n_fit:n_fit+n_cal]
            X_opt    = X_train_all[n_fit+n_cal:]; y_opt = y_train_all[n_fit+n_cal:]
            opt_rets = train_returns_all[n_fit+n_cal:]

        # ┏━━━━━━━━━━ Build AutoGluon classifier ━━━━━━━━━━┓
        model = AutoGluon(feature_names=feature_names, class_weight_ratio=cw_ratio)

        model.fit(X_fit, y_fit)

        # ┏━━━━━━━━━━ Calibrate on cal or use identity ━━━━━━━━━━┓
        if nocal:
            # ┏━━━━━━━━━━ NoCal: identity calibrator ━━━━━━━━━━┓
            from Utils.selective_classification.calibration import _IdentityCalibrator
            calibrator = _IdentityCalibrator()
            op = _find_best_utility_threshold(model.predict_proba(X_opt)[:, 1],
                                              opt_rets, fee=fee, labels=y_opt)
        else:
            # ┏━━━━━━━━━━ Calibrate on cal ━━━━━━━━━━┓
            cal_probs_raw = model.predict_proba(X_cal)[:, 1]
            _cal = calibrate_probabilities(cal_probs_raw, y_cal)
            calibrator = _cal["calibrator"]

            # ┏━━━━━━━━━━ Sweep threshold on calibrated opt ━━━━━━━━━━┓
            opt_probs_cal = calibrator.predict(model.predict_proba(X_opt)[:, 1])
            op = _find_best_utility_threshold(opt_probs_cal, opt_rets, fee=fee, labels=y_opt)

    # ┏━━━━━━━━━━ Non-RF / Non-AutoGluon path: 60/20/20 physical split ━━━━━━━━━━┓
    else:
        n_train = len(idx_train)
        if nocal:
            # ┏━━━━━━━━━━ NoCal: 75% fit / 25% opt (no cal slot) ━━━━━━━━━━┓
            n_fit   = int(n_train * 0.75)
            X_fit   = X_train_all[:n_fit]; y_fit = y_train_all[:n_fit]
            X_opt   = X_train_all[n_fit:]; y_opt = y_train_all[n_fit:]
            opt_rets = train_returns_all[n_fit:]
        else:
            # ┏━━━━━━━━━━ Split train → fit (60%), Val-cal (20%), Val-Opt (20%) ━━━━━━━━━━┓
            n_fit   = int(n_train * 0.60)
            n_cal   = int(n_train * 0.20)
            X_fit    = X_train_all[:n_fit]; y_fit = y_train_all[:n_fit]
            X_cal    = X_train_all[n_fit:n_fit+n_cal]; y_cal = y_train_all[n_fit:n_fit+n_cal]
            X_opt    = X_train_all[n_fit+n_cal:]; y_opt = y_train_all[n_fit+n_cal:]
            opt_rets = train_returns_all[n_fit+n_cal:]

        # ┏━━━━━━━━━━ Build model with HPO best_params if available ━━━━━━━━━━┓
        model = _build_edge_model(model_key, n_fit, seed=EDGE_SEED,
                                  class_weight_ratio=cw_ratio, best_params=best_params,
                                  feature_names=feature_names)

        # ┏━━━━━━━━━━ Fit model (TabM gets the opt slice for early stopping) ━━━━━━━━━━┓
        if model_key == "tabm":
            model.fit(X_fit, y_fit, X_eval=X_opt, y_eval=y_opt)
        else:
            model.fit(X_fit, y_fit)

        if nocal:
            # ┏━━━━━━━━━━ NoCal: identity calibrator ━━━━━━━━━━┓
            from Utils.selective_classification.calibration import _IdentityCalibrator
            calibrator = _IdentityCalibrator()
            op = _find_best_utility_threshold(model.predict_proba(X_opt)[:, 1],
                                              opt_rets, fee=fee, labels=y_opt)
        else:
            # ┏━━━━━━━━━━ Calibrate on cal ━━━━━━━━━━┓
            cal_probs_raw = model.predict_proba(X_cal)[:, 1]
            _cal = calibrate_probabilities(cal_probs_raw, y_cal)
            calibrator = _cal["calibrator"]

            # ┏━━━━━━━━━━ Sweep threshold on calibrated opt ━━━━━━━━━━┓
            opt_probs_cal = calibrator.predict(model.predict_proba(X_opt)[:, 1])
            op = _find_best_utility_threshold(opt_probs_cal, opt_rets, fee=fee, labels=y_opt)

    thr = op["threshold"]

    # ┏━━━━━━━━━━ Evaluate on calibrated test probs ━━━━━━━━━━┓
    preds = model.predict(X_test)
    raw_test_probs = model.predict_proba(X_test)[:, 1]
    cal_test_probs = calibrator.predict(raw_test_probs)
    sel = cal_test_probs >= thr

    return {"idx_test": idx_test,
            "y_test": y_test,
            "preds": preds,
            "probs": cal_test_probs,
            "sel": sel,
            "threshold": thr,
            "test_returns": test_returns,
            "test_assets": np.array(test_assets),
            "n_sel": int(sel.sum())}


# ┏━━━━━━━━━━ Helper to compute path metrics ━━━━━━━━━━┓
def _compute_path_metrics(path_results, fee):
    """Compute stitched CPCV-path metrics from concatenated split outputs."""
    # ┏━━━━━━━━━━ Concatenate split results ━━━━━━━━━━┓
    y_all     = np.concatenate([r["y_test"] for r in path_results])
    preds_all = np.concatenate([r["preds"] for r in path_results])
    probs_all = np.concatenate([r["probs"] for r in path_results])
    sel_all   = np.concatenate([r["sel"] for r in path_results]).astype(bool)
    rets_all  = np.concatenate([r["test_returns"] for r in path_results])
    asts_all  = np.concatenate([r["test_assets"] for r in path_results])

    # ┏━━━━━━━━━━ Compute metrics ━━━━━━━━━━┓
    n_samples  = len(y_all)
    n_selected = int(sel_all.sum())
    pred_mask  = preds_all == 1
    n_pred     = int(pred_mask.sum())
    
    # ┏━━━━━━━━━━ Compute net returns ━━━━━━━━━━┓
    pred_net = rets_all[pred_mask] - fee if n_pred > 0 else np.array([0.0])
    sel_net  = rets_all[sel_all] - fee if n_selected > 0 else np.array([0.0])
    m1_net   = rets_all - fee if n_samples > 0 else np.array([0.0])

    # ┏━━━━━━━━━━ Compute precision and accuracy ━━━━━━━━━━┓
    sel_precision = float((y_all[sel_all] == 1).sum()) / n_selected if n_selected > 0 else 0.0
    sel_accuracy = float(accuracy_score(y_all, sel_all.astype(int))) if n_samples > 0 else 0.0

    return {
        "accuracy": float(accuracy_score(y_all, preds_all)) if n_samples > 0 else 0.0,
        "precision": float(precision_score(y_all, preds_all, zero_division=0)) if n_samples > 0 else 0.0,
        "recall": float(recall_score(y_all, preds_all, zero_division=0)) if n_samples > 0 else 0.0,
        "f1": float(f1_score(y_all, preds_all, zero_division=0)) if n_samples > 0 else 0.0,
        "sel_accuracy": sel_accuracy,
        "sel_precision": sel_precision,
        "sel_coverage": n_selected / n_samples if n_samples > 0 else 0.0,
        "mean_ret": float(np.nanmean(pred_net)),
        "sel_mean_ret": float(np.nanmean(sel_net)),
        "m1_mean_ret": float(np.nanmean(m1_net)),
        "n_samples": n_samples,
        "n_selected": n_selected,
        "_y": y_all.astype(int),
        "_preds": preds_all.astype(int),
        "_probs": probs_all,
        "_sel": sel_all.astype(int),
        "_rets": rets_all,
        "_assets": asts_all,
    }



# ┏━━━━━━━━━━ Per-sample persistence (regime-composition analysis) ━━━━━━━━━━┓
def _write_sample_level_csv(metrics_list, dates_by_unit, out_path, unit_key="path"):
    """Persist one row per evaluated sample: unit id, date, asset, label, pred, prob, sel, ret.

    The path/split metrics dicts already carry per-sample arrays (_preds/_probs/_sel/
    _rets/_assets) and the runners already track the matching dates, but until now
    both were consumed only by the plots and then discarded. The regime-composition
    analysis needs to join each sample to a dated regime label, which the aggregated
    per-path CSV cannot support. Writing them out here keeps that analysis a pure
    read-side consumer instead of forcing it to re-run the CV.
    """
    rows = []
    for ui, (pm, unit_dates) in enumerate(zip(metrics_list, dates_by_unit)):
        n = len(pm["_rets"])
        if len(unit_dates) != n:
            # Misalignment would silently corrupt every downstream regime join.
            print(f"  [WARN] {unit_key} {ui+1}: {len(unit_dates)} dates vs {n} samples; skipping")
            continue
        rows.append(pd.DataFrame({
            unit_key: ui + 1,
            "date":   pd.to_datetime(list(unit_dates)),
            "asset":  pm["_assets"],
            "y_true": pm["_y"] if "_y" in pm else np.nan,
            "pred":   pm["_preds"],
            "prob":   pm["_probs"],
            "sel":    pm["_sel"],
            "ret":    pm["_rets"],
        }))
    if rows:
        pd.concat(rows, ignore_index=True).to_csv(out_path, index=False)
        print(f"  [saved] per-sample -> {out_path.name}")


# ┏━━━━━━━━━━ Expanding-window purged CV: main runner ━━━━━━━━━━┓
def run_cvexpand_analysis(cache_path, config, output_root, n_blocks=6, m2_name="rf", direction="up", granularity="1d", nocal=False):
    """Expanding-window CV with time-based purge (no combinatorics).

    Reuses the SAME datetime-block construction and purge-by-timedelta logic as CPCV
    (_build_datetime_blocks / _assign_blocks, purge_td = horizon * bar_width), but
    instead of every C(n_blocks, k_test) combination, only walks forward: for block
    i = 1..n_blocks-1, train = blocks [0..i-1] (purged near the boundary with block i),
    test = block i alone. This mirrors classic expanding-window walk-forward CV while
    keeping the same leakage protection CPCV uses at each train/test boundary.

    Unlike CPCV there is no "path" reconstruction: each split already IS one full
    train/test partition, not a combinatorial fragment.
    """
    # ┏━━━━━━━━━━ Initialize variables ━━━━━━━━━━┓
    fee     = config["evaluation"]["fee_per_trade"]
    horizon = config["data"]["load"]["forecast_horizon"]

    # ┏━━━━━━━━━━ Load multi cache ━━━━━━━━━━┓
    print(f"\n[edge-cvexpand] Loading cache: {cache_path}")
    multi = _load_multi_cache(cache_path)
    print(f"[edge-cvexpand] Direction: {direction} | Granularities: {multi.grans}")
    print(f"[edge-cvexpand] Expanding-window CV: N={n_blocks} blocks -> {n_blocks - 1} splits")
    print(f"[edge-cvexpand] Fee: {fee} | Horizon: {horizon}")
    print(f"[edge-cvexpand] Output: {output_root}\n")

    print(f"\n{'='*60}")
    print(f"[edge-cvexpand] {granularity} — {direction.upper()}")
    print(f"{'='*60}")

    # ┏━━━━━━━━━━ Extract data ━━━━━━━━━━┓
    sub = multi.sub[granularity]
    eng = sub["eng_features"]
    if isinstance(eng, torch.Tensor): eng = eng.numpy()
    labels = sub["labels"]
    if isinstance(labels, torch.Tensor): labels = labels.numpy()
    returns = sub["returns"]
    if isinstance(returns, torch.Tensor): returns = returns.numpy()
    asset_ids = sub["asset_ids"]
    if isinstance(asset_ids, torch.Tensor): asset_ids = asset_ids.numpy()
    asset_map = sub.get("asset_map", {})
    if not isinstance(asset_map, dict) and hasattr(sub, "asset_map"): asset_map = sub.asset_map
    dates_all_raw = sub["dates"]

    # ┏━━━━━━━━━━ CTTS branch: replace eng with raw close windows ━━━━━━━━━━┓
    if m2_name == "ctts":
        from Utils.classifier.ctts.ctts_features import extract_close_windows
        from Utils.data import GRAN_SEQ_LEN
        _ctts_seq = min(90, GRAN_SEQ_LEN.get(granularity, 90))
        eng = extract_close_windows(sub, seq_len=_ctts_seq)
        print(f"  [CTTS] Using raw close windows: shape {eng.shape} (seq_len={_ctts_seq})")

    # ┏━━━━━━━━━━ Filter valid samples ━━━━━━━━━━┓
    valid = ~np.isnan(labels)
    eng, labels, returns, asset_ids = eng[valid], labels[valid], returns[valid], asset_ids[valid]
    dates_valid = [dates_all_raw[i] for i in range(len(dates_all_raw)) if valid[i]]

    # ┏━━━━━━━━━━ CRITICAL: Exclude Test data — cvexpand only ever sees Train+Val ━━━━━━━━━━┓
    t_val_end = pd.Timestamp(config["data"]["split"]["val_end"])
    # Lower bound as well as upper: a prediction must not be *dated* before the
    # configured start_date. Context windows may reach further back (the model needs
    # history), but the first evaluated sample belongs to the declared window --
    # otherwise the evaluated span silently depends on granularity (1h reached 37
    # days earlier than 1d), and the reported window does not match the data.
    t_start = pd.Timestamp(config["data"]["split"]["start_date"])
    cvx_mask = np.array([(d >= t_start) and (d <= t_val_end) for d in dates_valid])
    n_before = int(sum(1 for d in dates_valid if d < t_start))
    n_excluded = int((~cvx_mask).sum())
    eng = eng[cvx_mask]
    labels = labels[cvx_mask]
    returns = returns[cvx_mask]
    asset_ids = asset_ids[cvx_mask]
    dates_valid = [d for d, m in zip(dates_valid, cvx_mask) if m]

    print(f"  Samples: {len(labels):,} (excluded {n_excluded:,} outside "
          f"{t_start.strftime('%Y-%m-%d')}..{t_val_end.strftime('%Y-%m-%d')}; "
          f"{n_before:,} before start_date)")
    print(f"  Features: {eng.shape[1]}")

    # ┏━━━━━━━━━━ Build datetime blocks (SAME construction as CPCV) ━━━━━━━━━━┓
    boundaries = _build_datetime_blocks(dates_valid, n_blocks)
    block_ids = _assign_blocks(dates_valid, boundaries)
    for b in range(n_blocks):
        b_size = int((block_ids == b).sum())
        print(f"    Block {b}: {boundaries[b][0].strftime('%Y-%m-%d')} → "
              f"{boundaries[b][1].strftime('%Y-%m-%d')} ({b_size:,} samples)")
    unassigned = int((block_ids == -1).sum())
    if unassigned > 0:
        print(f"    WARNING: {unassigned} samples unassigned")

    # ┏━━━━━━━━━━ Purge window (SAME as CPCV: horizon x bar width) ━━━━━━━━━━┓
    bar_td = _gran_to_timedelta(granularity)
    purge_td = bar_td * horizon
    print(f"  Purge: {purge_td} (= {horizon} x {bar_td})")

    # ┏━━━━━━━━━━ Generate expanding-window splits ━━━━━━━━━━┓
    # Split i: train = blocks [0..i-1] (right edge purged near block i's start), test = block i.
    # No left-side purge is needed on the test block itself — nothing after it is ever used
    # as train in an expanding scheme, unlike CPCV's combinatorial train blocks on both sides.
    dates_arr = np.asarray(dates_valid)
    splits = []
    for i in range(1, n_blocks):
        idx_test = np.where(block_ids == i)[0]
        idx_train_raw = np.where(np.isin(block_ids, list(range(i))))[0]
        if len(idx_test) == 0 or len(idx_train_raw) == 0:
            continue

        test_start, _test_end = boundaries[i]
        purged_mask = np.zeros(len(dates_arr), dtype=bool)
        for j in idx_train_raw:
            t = dates_arr[j]
            if (test_start - purge_td) <= t < test_start:
                purged_mask[j] = True

        idx_train = idx_train_raw[~purged_mask[idx_train_raw]]
        splits.append({"test_block": i, "idx_train": idx_train, "idx_test": idx_test,
                       "idx_purged": idx_train_raw[purged_mask[idx_train_raw]]})

    avg_purged = np.mean([len(s["idx_purged"]) for s in splits]) if splits else 0.0
    avg_train  = np.mean([len(s["idx_train"])  for s in splits]) if splits else 0.0
    avg_test   = np.mean([len(s["idx_test"])   for s in splits]) if splits else 0.0
    print(f"  {len(splits)} splits | Avg: train={avg_train:.0f}, test={avg_test:.0f}, purged={avg_purged:.0f}")

    if not splits:
        print(f"  [SKIP] No valid expanding-window splits")
        return

    # ┏━━━━━━━━━━ Create output directory ━━━━━━━━━━┓
    gran_dir = output_root / direction.upper() / granularity
    gran_dir.mkdir(parents=True, exist_ok=True)

    # ┏━━━━━━━━━━ Load HPO params ━━━━━━━━━━┓
    _best_params = _load_best_params(config, m2_name, direction, granularity)
    _feature_names = list(sub.get("feature_names", [])) if hasattr(sub, "feature_names") or "feature_names" in sub else None

    # ┏━━━━━━━━━━ Run splits (reuses the SAME per-split fit/calibrate/eval as CPCV) ━━━━━━━━━━┓
    split_metrics_list, dates_by_split = [], []
    for si, sp in enumerate(splits):
        result = _run_cpcv_split(eng, labels, returns, asset_ids, asset_map,
                                 sp["idx_train"], sp["idx_test"],
                                 direction, fee,
                                 model_key    = m2_name,
                                 best_params  = _best_params,
                                 feature_names = _feature_names,
                                 nocal        = nocal)
        acc = accuracy_score(result["y_test"], result["preds"])
        print(f"    Split {si+1:2d}/{len(splits)}: test_block={sp['test_block']} "
              f"n_test={len(sp['idx_test']):,} acc={acc:.4f} thr={result['threshold']:.3f}")

        # Each split IS the reportable unit directly (no path stitching needed).
        pm = _compute_path_metrics([{
            "idx_test": result["idx_test"], "y_test": result["y_test"],
            "preds": result["preds"], "probs": result["probs"], "sel": result["sel"],
            "threshold": result["threshold"], "test_returns": result["test_returns"],
            "test_assets": result["test_assets"], "n_sel": int(result["sel"].sum()),
        }], fee)
        split_metrics_list.append(pm)
        dates_by_split.append([dates_valid[idx] for idx in sp["idx_test"]])
        print(f"    Split {si+1}: n={pm['n_samples']:,} acc={pm['accuracy']:.4f} "
              f"prec={pm['precision']:.4f} sel_prec={pm['sel_precision']:.4f} "
              f"cov={pm['sel_coverage']:.3f} sel_μ_ret={pm['sel_mean_ret']:+.5f}")

    # ┏━━━━━━━━━━ M1 baseline on the raw (train+val) timeline ━━━━━━━━━━┓
    m1_baselines = _compute_m1_baseline_on_raw_indices(sub, list(range(len(dates_all_raw))))
    if m1_baselines["m1_acc"] is not None:
        print(f"\n  M1 Baseline: acc={m1_baselines['m1_acc']:.4f}  prec={m1_baselines['m1_prec']:.4f}")

    # ┏━━━━━━━━━━ Plots (same helpers as CPCV; "path" = "split" here) ━━━━━━━━━━┓
    sharpes, total_rets = _plot_path_equity(split_metrics_list,
                                            dates_by_split,
                                            gran_dir / "cvexpand_equity.png",
                                            granularity,
                                            direction,
                                            fee,
                                            horizon,
                                            granularity,
                                            config,
                                            unit_label="Split")
    _plot_path_boxplots(split_metrics_list,
                        m1_baselines,
                        gran_dir / "cvexpand_boxplots.png",
                        granularity,
                        direction,
                        unit_label="Splits",
                        scheme_label="Expanding-Window CV")

    # ┏━━━━━━━━━━ CSV ━━━━━━━━━━┓
    rows = []
    for si, pm in enumerate(split_metrics_list):
        row = {"split": si + 1, "test_block": splits[si]["test_block"]}
        for k in ["accuracy", "precision", "recall", "f1", "sel_accuracy", "sel_precision",
                   "sel_coverage", "mean_ret", "sel_mean_ret", "m1_mean_ret", "n_samples", "n_selected"]:
            row[k] = pm[k]
        if si < len(sharpes):
            row["sharpe"] = sharpes[si]
            row["total_ret_pct"] = total_rets[si]
        rows.append(row)
    pd.DataFrame(rows).to_csv(gran_dir / "cvexpand_splits.csv", index=False)
    _write_sample_level_csv(split_metrics_list, dates_by_split,
                            gran_dir / "cvexpand_samples.csv", unit_key="split")

    # ┏━━━━━━━━━━ Summary ━━━━━━━━━━┓
    accs    = np.array([pm["accuracy"]      for pm in split_metrics_list])
    precs   = np.array([pm["precision"]     for pm in split_metrics_list])
    s_precs = np.array([pm["sel_precision"] for pm in split_metrics_list])
    s_accs  = np.array([pm["sel_accuracy"]  for pm in split_metrics_list])
    s_covs  = np.array([pm["sel_coverage"]  for pm in split_metrics_list])
    s_rets  = np.array([pm["sel_mean_ret"]  for pm in split_metrics_list])

    summary = {"granularity": granularity, "direction": direction,
               "n_blocks": n_blocks, "n_splits": len(splits), "n_features": eng.shape[1],
               "purge_window": str(purge_td),
               "m1_baseline_acc": m1_baselines.get("m1_acc"),
               "m1_baseline_prec": m1_baselines.get("m1_prec"),
               "split_acc_mean": float(np.mean(accs)), "split_acc_std": float(np.std(accs)),
               "split_prec_mean": float(np.mean(precs)), "split_prec_std": float(np.std(precs)),
               "split_sel_acc_mean": float(np.mean(s_accs)), "split_sel_acc_std": float(np.std(s_accs)),
               "split_sel_prec_mean": float(np.mean(s_precs)), "split_sel_prec_std": float(np.std(s_precs)),
               "split_sel_cov_mean": float(np.mean(s_covs)), "split_sel_cov_std": float(np.std(s_covs)),
               "split_sel_ret_mean": float(np.mean(s_rets)), "split_sel_ret_std": float(np.std(s_rets)),
               "split_sharpes": sharpes, "split_total_rets": total_rets}

    print(f"\n  ┌─ {granularity} {direction.upper()} Expanding-CV Summary ─────────────┐")
    if m1_baselines["m1_acc"] is not None:
        print(f"  │ M1 Baseline: acc={m1_baselines['m1_acc']:.4f}  prec={m1_baselines['m1_prec']:.4f}")
    print(f"  │ Split Acc:       {summary['split_acc_mean']:.4f} ± {summary['split_acc_std']:.4f}")
    print(f"  │ Split Prec:      {summary['split_prec_mean']:.4f} ± {summary['split_prec_std']:.4f}")
    print(f"  │ Split Sel Prec:  {summary['split_sel_prec_mean']:.4f} ± {summary['split_sel_prec_std']:.4f}")
    print(f"  │ Split Sel Cov:   {summary['split_sel_cov_mean']:.4f} ± {summary['split_sel_cov_std']:.4f}")
    print(f"  │ Split Sel μ-Ret: {summary['split_sel_ret_mean']:+.5f} ± {summary['split_sel_ret_std']:.5f}")
    print(f"  └─────────────────────────────────────────┘")
    print(f"  Saved to: {gran_dir}")

    summary_path = output_root / direction.upper() / f"edge_summary_cvexpand_{granularity}.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[edge-cvexpand] Summary: {summary_path}")
    print(f"\n[edge-cvexpand] Done.")


# ┏━━━━━━━━━━ CPCV: main runner ━━━━━━━━━━┓
def run_cpcv_analysis(cache_path, config, output_root, n_blocks=6, k_test=2, m2_name="rf", direction="up", granularity="1d", nocal=False):
    # ┏━━━━━━━━━━ Initialize variables ━━━━━━━━━━┓
    fee       = config["evaluation"]["fee_per_trade"]
    horizon   = config["data"]["load"]["forecast_horizon"]
    
    # Paths are C(N-1, k-1), matching _reconstruct_paths — NOT n_blocks // k_test,
    # which under-reported (e.g. 6//2=3 while C(5,1)=5 paths are actually built).
    n_paths  = math.comb(n_blocks - 1, k_test - 1)
    n_splits = len(list(combinations(range(n_blocks), k_test)))
    summary_all = {}
    
    # ┏━━━━━━━━━━ Load multi cache ━━━━━━━━━━┓
    print(f"\n[edge-cpcv] Loading cache: {cache_path}")
    multi = _load_multi_cache(cache_path)
    print(f"[edge-cpcv] Direction: {direction} | Granularities: {multi.grans}")
    print(f"[edge-cpcv] CPCV: N={n_blocks}, k={k_test} → C({n_blocks},{k_test})={n_splits} splits → {n_paths} paths")
    print(f"[edge-cpcv] Fee: {fee} | Horizon: {horizon}")
    print(f"[edge-cpcv] Output: {output_root}\n")

    print(f"\n{'='*60}")
    print(f"[edge-cpcv] {granularity} — {direction.upper()}")
    print(f"{'='*60}")

    # ┏━━━━━━━━━━ Extract data ━━━━━━━━━━┓
    sub = multi.sub[granularity]
    eng = sub["eng_features"]
    if isinstance(eng, torch.Tensor): eng = eng.numpy()
    labels = sub["labels"]
    if isinstance(labels, torch.Tensor): labels = labels.numpy()
    returns = sub["returns"]
    if isinstance(returns, torch.Tensor): returns = returns.numpy()
    asset_ids = sub["asset_ids"]
    if isinstance(asset_ids, torch.Tensor): asset_ids = asset_ids.numpy()
    asset_map = sub.get("asset_map", {})
    if not isinstance(asset_map, dict) and hasattr(sub, "asset_map"): asset_map = sub.asset_map
    dates_all_raw = sub["dates"]

    # ┏━━━━━━━━━━ CTTS branch: replace eng with raw close windows ━━━━━━━━━━┓
    if m2_name == "ctts":
        from Utils.classifier.ctts.ctts_features import extract_close_windows
        from Utils.data import GRAN_SEQ_LEN
        _ctts_seq = min(90, GRAN_SEQ_LEN.get(granularity, 90))
        eng = extract_close_windows(sub, seq_len=_ctts_seq)
        print(f"  [CTTS] Using raw close windows: shape {eng.shape} (seq_len={_ctts_seq})")

    # ┏━━━━━━━━━━ Filter valid samples ━━━━━━━━━━┓
    valid = ~np.isnan(labels)
    eng, labels, returns, asset_ids = eng[valid], labels[valid], returns[valid], asset_ids[valid]
    dates_valid = [dates_all_raw[i] for i in range(len(dates_all_raw)) if valid[i]]
    valid_to_raw = [i for i in range(len(dates_all_raw)) if valid[i]]

    # ┏━━━━━━━━━━ CRITICAL: Exclude Test data from CPCV ━━━━━━━━━━┓
    # CPCV must only use Train+Val data. Test period is completely invisible.
    t_val_end = pd.Timestamp(config["data"]["split"]["val_end"])
    # See the CVExpand site: bound the window below by start_date as well, so the
    # first prediction belongs to the declared evaluation window. Context windows
    # may still reach back before it.
    t_start = pd.Timestamp(config["data"]["split"]["start_date"])
    cpcv_mask = np.array([(d >= t_start) and (d <= t_val_end) for d in dates_valid])
    n_before = int(sum(1 for d in dates_valid if d < t_start))
    n_excluded = int((~cpcv_mask).sum())
    eng = eng[cpcv_mask]
    labels = labels[cpcv_mask]
    returns = returns[cpcv_mask]
    asset_ids = asset_ids[cpcv_mask]
    dates_valid = [d for d, m in zip(dates_valid, cpcv_mask) if m]

    print(f"  Samples: {len(labels):,} (excluded {n_excluded:,} outside "
          f"{t_start.strftime('%Y-%m-%d')}..{t_val_end.strftime('%Y-%m-%d')}; "
          f"{n_before:,} before start_date)")
    print(f"  Features: {eng.shape[1]}")

    # ┏━━━━━━━━━━ Build datetime blocks ━━━━━━━━━━┓
    boundaries = _build_datetime_blocks(dates_valid, n_blocks)
    block_ids = _assign_blocks(dates_valid, boundaries)
    for b in range(n_blocks):
        b_size = int((block_ids == b).sum())
        print(f"    Block {b}: {boundaries[b][0].strftime('%Y-%m-%d')} → "
              f"{boundaries[b][1].strftime('%Y-%m-%d')} ({b_size:,} samples)")
    unassigned = int((block_ids == -1).sum())
    if unassigned > 0:
        print(f"    WARNING: {unassigned} samples unassigned")

    # ┏━━━━━━━━━━ Purge window ━━━━━━━━━━┓
    bar_td = _gran_to_timedelta(granularity)
    purge_td = bar_td * horizon
    print(f"  Purge: {purge_td} (= {horizon} x {bar_td})")

    # ┏━━━━━━━━━━ Generate splits ━━━━━━━━━━┓
    splits = _generate_cpcv_splits(n_blocks, k_test, block_ids, dates_valid, purge_td, boundaries)
    avg_purged = np.mean([len(s["idx_purged"]) for s in splits])
    avg_train  = np.mean([len(s["idx_train"])  for s in splits])
    avg_test   = np.mean([len(s["idx_test"])   for s in splits])
    print(f"  {len(splits)} splits | Avg: train={avg_train:.0f}, test={avg_test:.0f}, purged={avg_purged:.0f}")

    # ┏━━━━━━━━━━ Create output directory ━━━━━━━━━━┓
    gran_dir = output_root / direction.upper() / granularity
    gran_dir.mkdir(parents=True, exist_ok=True)

    # ┏━━━━━━━━━━ Reconstruct paths ━━━━━━━━━━┓
    paths = _reconstruct_paths(splits, n_blocks, k_test)
    print(f"\n  Reconstructed {len(paths)} paths")

    # ┏━━━━━━━━━━ Plot split matrix CPCV ━━━━━━━━━━┓
    _plot_split_matrix(splits,
                       paths,
                       n_blocks,
                       boundaries,
                       purge_td,
                       gran_dir / "cpcv_split_matrix.png",
                       granularity,
                       direction)

    # ┏━━━━━━━━━━ Load HPO params for this (m2, direction, granularity) ━━━━━━━━━━┓
    # TabPFN / TabICL / RF: load from Utils.hpo best_params.json
    # AutoGluon: no param reuse (WeightedEnsemble can't be reconstructed via hyperparameters API)
    _best_params = _load_best_params(config, m2_name, direction, granularity)

    _feature_names = list(sub.get("feature_names", [])) if hasattr(sub, "feature_names") or "feature_names" in sub else None

    # ┏━━━━━━━━━━ Run splits ━━━━━━━━━━┓
    split_results = []
    for si, sp in enumerate(splits):
        result = _run_cpcv_split(eng, labels, returns, asset_ids, asset_map,
                                 sp["idx_train"], sp["idx_test"],
                                 direction, fee,
                                 model_key          = m2_name,
                                 best_params        = _best_params,
                                 feature_names      = _feature_names,
                                 nocal              = nocal)
        split_results.append(result)
        if (si + 1) % 5 == 0 or si == 0:
            acc = accuracy_score(result["y_test"], result["preds"])
            print(f"    Split {si+1:2d}/{len(splits)}: test_blocks={sp['test_blocks']} "
                  f"n_test={len(sp['idx_test']):,} acc={acc:.4f} thr={result['threshold']:.3f}")

    # ┏━━━━━━━━━━ Compute path metrics ━━━━━━━━━━┓
    path_metrics_list, dates_by_path = [], []
    
    # ┏━━━━━━━━━━ Loop through paths ━━━━━━━━━━┓
    for pi, path in enumerate(paths):
        # ┏━━━━━━━━━━ Initialize path results ━━━━━━━━━━┓
        path_split_results, path_dates = [], []

        # ┏━━━━━━━━━━ Loop through splits in path ━━━━━━━━━━┓
        for entry in sorted(path, key=lambda e: e["block"]):
            # ┏━━━━━━━━━━ Get split results ━━━━━━━━━━┓
            sr = split_results[entry["split_idx"]]
            
            # ┏━━━━━━━━━━ Get block mask ━━━━━━━━━━┓
            block_mask = block_ids[sr["idx_test"]] == entry["block"]
            if block_mask.sum() == 0:
                continue
            
            # ┏━━━━━━━━━━ Append path split results ━━━━━━━━━━┓
            path_split_results.append({"idx_test": sr["idx_test"][block_mask],
                                       "y_test": sr["y_test"][block_mask],
                                       "preds": sr["preds"][block_mask],
                                       "probs": sr["probs"][block_mask],
                                       "sel": sr["sel"][block_mask],
                                       "threshold": sr["threshold"],
                                       "test_returns": sr["test_returns"][block_mask],
                                       "test_assets": sr["test_assets"][block_mask],
                                       "n_sel": int(sr["sel"][block_mask].sum())})
            
            # ┏━━━━━━━━━━ Append path dates ━━━━━━━━━━┓
            for idx in sr["idx_test"][block_mask]:
                path_dates.append(dates_valid[idx])
        
        # ┏━━━━━━━━━━ Compute path metrics ━━━━━━━━━━┓
        if not path_split_results:
            continue
        pm = _compute_path_metrics(path_split_results, fee)
        path_metrics_list.append(pm)
        dates_by_path.append(path_dates)
        
        # ┏━━━━━━━━━━ Print path metrics ━━━━━━━━━━┓
        print(f"    Path {pi+1}: n={pm['n_samples']:,} acc={pm['accuracy']:.4f} "
              f"prec={pm['precision']:.4f} sel_prec={pm['sel_precision']:.4f} "
              f"cov={pm['sel_coverage']:.3f} sel_μ_ret={pm['sel_mean_ret']:+.5f}")

    if not path_metrics_list:
        print(f"  [SKIP] No valid paths")
        return

    # ┏━━━━━━━━━━ M1 baselines on stitched path raw indices ━━━━━━━━━━┓
    # Since the concatenated sum of all paths is exactly the entire raw timeline,
    # the M1 baseline for the stitched paths is the M1 baseline over all raw dates.
    m1_baselines = _compute_m1_baseline_on_raw_indices(sub, list(range(len(dates_all_raw))))
    if m1_baselines["m1_acc"] is not None:
        print(f"\n  M1 Baseline (stitched): acc={m1_baselines['m1_acc']:.4f}  prec={m1_baselines['m1_prec']:.4f}")

    # ┏━━━━━━━━━━ Plots ━━━━━━━━━━┓
    # Equity curve
    sharpes, total_rets = _plot_path_equity(path_metrics_list,
                                            dates_by_path,
                                            gran_dir / "cpcv_path_equity.png",
                                            granularity,
                                            direction,
                                            fee,
                                            horizon,
                                            granularity,
                                            config)
    # Boxplots
    _plot_path_boxplots(path_metrics_list,
                        m1_baselines,
                        gran_dir / "cpcv_path_boxplots.png",
                        granularity,
                        direction)

    # ┏━━━━━━━━━━ CSV ━━━━━━━━━━┓
    rows = []
    for pi, pm in enumerate(path_metrics_list):
        row = {"path": pi + 1}
        for k in ["accuracy", "precision", "recall", "f1", "sel_accuracy", "sel_precision",
                   "sel_coverage", "mean_ret", "sel_mean_ret", "m1_mean_ret", "n_samples", "n_selected"]:
            row[k] = pm[k]
        if pi < len(sharpes):
            row["sharpe"] = sharpes[pi]
            row["total_ret_pct"] = total_rets[pi]
        rows.append(row)
    pd.DataFrame(rows).to_csv(gran_dir / "cpcv_paths.csv", index=False)
    _write_sample_level_csv(path_metrics_list, dates_by_path,
                            gran_dir / "cpcv_samples.csv", unit_key="path")

    # ┏━━━━━━━━━━ Summary ━━━━━━━━━━┓
    accs    = np.array([pm["accuracy"]      for pm in path_metrics_list])
    precs   = np.array([pm["precision"]     for pm in path_metrics_list])
    s_precs = np.array([pm["sel_precision"] for pm in path_metrics_list])
    s_accs  = np.array([pm["sel_accuracy"]  for pm in path_metrics_list])
    s_covs  = np.array([pm["sel_coverage"]  for pm in path_metrics_list])
    s_rets  = np.array([pm["sel_mean_ret"]  for pm in path_metrics_list])
    worst_sharpe = min(sharpes) if sharpes else 0.0

    # ┏━━━━━━━━━━ Descriptive path statistics ━━━━━━━━━━┓
    # Reported, not gated: the deployment verdict this used to emit has been removed —
    # the regime analysis (Phase 5) is what the results are read from now.
    path_total_rets_arr = np.array(total_rets) if total_rets else np.array([0.0])
    frac_profitable = float(np.mean(path_total_rets_arr > 0))
    median_sharpe   = float(np.median(sharpes)) if sharpes else 0.0

    # ┏━━━━━━━━━━ Summary dict ━━━━━━━━━━┓
    summary = {"granularity": granularity, "direction": direction,
               "n_blocks": n_blocks, "k_test": k_test, "n_splits": len(splits),
               "n_paths": len(path_metrics_list), "n_features": eng.shape[1],
               "purge_window": str(purge_td),
               "m1_baseline_acc": m1_baselines.get("m1_acc"),
               "m1_baseline_prec": m1_baselines.get("m1_prec"),
               "path_acc_mean": float(np.mean(accs)), "path_acc_std": float(np.std(accs)),
               "path_prec_mean": float(np.mean(precs)), "path_prec_std": float(np.std(precs)),
               "path_sel_acc_mean": float(np.mean(s_accs)), "path_sel_acc_std": float(np.std(s_accs)),
               "path_sel_prec_mean": float(np.mean(s_precs)), "path_sel_prec_std": float(np.std(s_precs)),
               "path_sel_cov_mean": float(np.mean(s_covs)), "path_sel_cov_std": float(np.std(s_covs)),
               "path_sel_ret_mean": float(np.mean(s_rets)), "path_sel_ret_std": float(np.std(s_rets)),
               "worst_path_sharpe": worst_sharpe,
               "path_sharpes": sharpes, "path_total_rets": total_rets,
               "median_path_sharpe":  median_sharpe,
               "frac_profitable":     frac_profitable}

    summary_all[granularity] = summary

    # ┏━━━━━━━━━━ Print summary ━━━━━━━━━━┓
    print(f"\n  ┌─ {granularity} {direction.upper()} CPCV Summary ─────────────┐")
    if m1_baselines["m1_acc"] is not None:
        print(f"  │ M1 Baseline: acc={m1_baselines['m1_acc']:.4f}  prec={m1_baselines['m1_prec']:.4f}")
    print(f"  │ Path Acc:       {summary['path_acc_mean']:.4f} ± {summary['path_acc_std']:.4f}")
    print(f"  │ Path Prec:      {summary['path_prec_mean']:.4f} ± {summary['path_prec_std']:.4f}")
    print(f"  │ Path Sel Prec:  {summary['path_sel_prec_mean']:.4f} ± {summary['path_sel_prec_std']:.4f}")
    print(f"  │ Path Sel Cov:   {summary['path_sel_cov_mean']:.4f} ± {summary['path_sel_cov_std']:.4f}")
    print(f"  │ Path Sel μ-Ret: {summary['path_sel_ret_mean']:+.5f} ± {summary['path_sel_ret_std']:.5f}")
    print(f"  │ Median SR:      {median_sharpe:.2f}")
    print(f"  │ Frac Profitable:{frac_profitable:.0%}")
    print(f"  └─────────────────────────────────────────┘")
    print(f"  Saved to: {gran_dir}")

    # ┏━━━━━━━━━━ Save summary ━━━━━━━━━━┓
    if summary_all:
        summary_path = output_root / direction.upper() / f"edge_summary_{granularity}.json"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        with open(summary_path, "w") as f:
            json.dump(summary_all, f, indent=2)
        print(f"\n[edge-cpcv] Summary: {summary_path}")
        _plot_cross_gran_cpcv(summary_all, output_root / direction.upper() / "edge_cross_gran.png", direction)
    print(f"\n[edge-cpcv] Done.")





# ┏━━━━━━━━━━ CLI ━━━━━━━━━━┓
def main():
    # ┏━━━━━━━━━━ Parse arguments ━━━━━━━━━━┓
    parser = argparse.ArgumentParser(description="Edge Analysis — cross-validation evidence via CPCV and expanding-window CV")
    parser.add_argument("--cache_path", type=str, default=None, help="Explicit path to dataset cache .pt")
    parser.add_argument("--config", type=_load_config, help="Experiment config (path to .yaml or inline JSON string)", required=True)
    parser.add_argument("--mode", type=str, choices=["cpcv", "cvexpand"], required=True,
                        help="'cpcv' = Combinatorial Purged CV; "
                             "'cvexpand' = expanding-window purged CV, same datetime blocks and "
                             "time-based purge as cpcv but walk-forward only, no combinatorics")
    parser.add_argument("--phase", type=str, help="Experimental Phase", required=True)
    parser.add_argument("--m2", type=str, help="M2 model to use", required=True)
    parser.add_argument("--direction", type=str, help="Direction to use", required=True)
    parser.add_argument("--granularity", type=str, help="Granularity to use", required=True)
    parser.add_argument("--nocal", action="store_true",
                        help="Disable probability calibration. Merges Val-Cal into Val-Opt and "
                             "sweeps τ* on raw probabilities. Outputs land under Edge_NoCal/.")

    args = parser.parse_args()

    # ┏━━━━━━━━━━ Auto-resolve cache path from config if not provided ━━━━━━━━━━┓
    if args.cache_path is None:
        from Utils.utils import _resolve_caches, _infer_direction
        try:
            caches = _resolve_caches(args.config, explicit=None)
            args.cache_path = caches.get(args.direction.lower())
            if args.cache_path:
                print(f"[edge] Auto-resolved cache ({args.direction}): {Path(args.cache_path).name}")
            else:
                print(f"[edge] WARNING: could not auto-resolve cache for direction={args.direction}")
        except Exception as e:
            print(f"[edge] WARNING: cache auto-resolution failed: {e}")

    # ┏━━━━━━━━━━ Auto-set nocal from config if not explicitly passed ━━━━━━━━━━┓
    if not args.nocal:
        _thres = args.config.get("runtime", {}).get("training", {}).get("thres", "utility")
        if _thres == "utility_nocal":
            args.nocal = True

    # ┏━━━━━━━━━━ Output directory (includes model name) ━━━━━━━━━━┓
    # When --nocal is set, outputs go under Edge_NoCal/ so they don't collide with the
    # calibrated results already produced under Edge/.
    _edge_folder = "Edge_NoCal" if args.nocal else "Edge"
    output_root = Path(args.config["paths"]["output_root"]) / "Analysis" / _edge_folder / args.config["experiment"]["m1"].capitalize() / args.m2

    # ┏━━━━━━━━━━ Run analysis ━━━━━━━━━━┓
    if args.mode == "cpcv":
        run_cpcv_analysis(args.cache_path,
                          args.config,
                          output_root,
                          n_blocks=args.config["runtime"][args.phase]["n_blocks"],
                          k_test=args.config["runtime"][args.phase]["k_test"],
                          m2_name=args.m2,
                          direction=args.direction,
                          granularity=args.granularity,
                          nocal=args.nocal)

    elif args.mode == "cvexpand":
        run_cvexpand_analysis(args.cache_path,
                              args.config,
                              output_root,
                              n_blocks=args.config["runtime"][args.phase]["n_blocks"],
                              m2_name=args.m2,
                              direction=args.direction,
                              granularity=args.granularity,
                              nocal=args.nocal)
    else:
        print(f"Invalid mode: {args.mode}")


if __name__ == "__main__":
    main()

