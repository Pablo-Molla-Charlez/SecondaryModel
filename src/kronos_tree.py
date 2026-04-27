"""
Kronos Tree — M2 Meta-Labeling worker (training + combined phases).

Invocation contract:
    CLI args (--m2, --direction, --granularity, --phase) select WHICH slice to
    run. The YAML file passed to --config defines HOW a slice is built (data
    signature, splits, threshold/HPO knobs, output root). config.experiment.*
    lists are read only by Utils/experiments.py, which explodes them into
    per-slice invocations of this module.

    Example (direct invocation):
        python kronos_tree.py --config config.yaml --phase training \\
            --m2 rf --direction up --granularity 1d

    `--config` accepts a YAML file path. On first run `_resolve_caches` will
    build the multi-gran cache for both directions automatically.
"""

import argparse, hashlib, sys, json, pickle
from pathlib import Path
import yaml
import numpy as np
import pandas as pd
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))

# ┏━━━━━━━━━━ Pipeline Data Preprocessing ━━━━━━━━━━┓
from Utils.data import (ENG_FEATURE_NAMES,
                        ENG_FEATURE_GROUPS,
                        resolve_feature_names,
                        split_by_global_time,
                        load_dataset_from_config,
                        prepare_multi_asset_dataset,
                        prepare_multi_gran_dataset,
                        GRAN_SEQ_LEN)

# ┏━━━━━━━━━━ Financial Backtesting ━━━━━━━━━━┓
from Utils.backtest import (_annualization_factor,
                            _build_spread_equity,
                            _calc_drawdown,
                            _calc_sharpe,
                            _equity_horizon_returns,
                            _load_raw_close_prices,
                            run_feature_backtest,
                            run_combined_backtest)

# ┏━━━━━━━━━━ Comparison of results between models and granularities ━━━━━━━━━━┓
from Utils.backtest import (GRAN_ORDER,
                            run_comparison,
                            run_paradigm_comparison)

# ┏━━━━━━━━━━ Feature analysis ━━━━━━━━━━┓
from Utils.feature_selection import (_plot_prob_distribution,
                                     plot_class_distributions,
                                     plot_correlation_heatmap,
                                     plot_mutual_information,
                                     plot_confusion_matrix,
                                     plot_ocp_threshold_evolution,
                                     plot_pointbiserial,
                                     plot_temporal_risk_coverage_curve,
                                     plot_temporal_risk_coverage_curve_final,
                                     plot_tree_importance,
                                     compute_top_features,
                                     run_feature_selection,
                                     plot_selective_return_distribution)

# ┏━━━━━━━━━━ Online Conformal Prediction ━━━━━━━━━━┓
from Utils.ocp import (_ocp_threshold_to_op,
                       _run_saocp_online,
                       _run_cost_deferral_online,
                       calib_window_for_gran,
                       plot_mondrian_diagnostics)

# ┏━━━━━━━━━━ Utility-based Selective Classification [risk-coverage analysis] ━━━━━━━━━━┓   
from Utils.selective_classification import (_find_best_utility_threshold,
                                            collect_risk_coverage_curve,
                                            calibrate_probabilities)

# ┏━━━━━━━━━━ Utils ━━━━━━━━━━┓
from Utils.utils import (NumpyJSONEncoder,
                         m1_display_label as _m1_display_label,
                         m1_output_bucket as _m1_output_bucket,
                         model_label as _model_label,
                         _safe_json,
                         _load_config,
                         _build_cache_from_config,
                         _resolve_caches,
                         _filter_dataset_by_granularity,
                         _class_names,
                         _infer_direction,
                         _load_multi_cache,
                         _load_best_params)
# ┏━━━━━━━━━━ Models ━━━━━━━━━━┓
from Utils.classifier import (MODEL_CHOICES,
                              MODELS_NO_SCALING,
                              _build_tree_model,
                              _save_final_model)
from Utils.classifier.factory import (_AG_TIME_LIMIT,
                                      _AG_PRESETS)


def _build_dataframe(dataset) -> tuple[pd.DataFrame, np.ndarray]:
    # ┏━━━━━━━━━━ Convert eng_features to numpy ━━━━━━━━━━┓
    eng = dataset["eng_features"] if isinstance(dataset, dict) else dataset.eng_features
    if isinstance(eng, torch.Tensor):
        eng = eng.numpy()

    # ┏━━━━━━━━━━ Get labels ━━━━━━━━━━┓
    labels = dataset["labels"] if isinstance(dataset, dict) else dataset.labels
    
    # ┏━━━━━━━━━━ Convert labels to numpy ━━━━━━━━━━┓
    if isinstance(labels, torch.Tensor):
        labels = labels.numpy()
    
    # ┏━━━━━━━━━━ Filter out NaN labels ━━━━━━━━━━┓
    valid = ~np.isnan(labels)
    eng = eng[valid]
    labels = labels[valid].astype(int)
    
    # ┏━━━━━━━━━━ Create dataframe ━━━━━━━━━━┓
    df = pd.DataFrame(eng, columns=resolve_feature_names(eng.shape[1]))
    return df, labels


# ------------------------------------------------------------------------------
# 
# ------------------------------------------------------------------------------
def temporal_eval(dataset: dict,
                  feature_cols: list,
                  save_dir: Path,
                  class_names: list,
                  meta_mode: str,
                  split_indices: tuple,
                  direction: str = "up",
                  fee: float = 0.0,
                  model_name: str = "rf",
                  desc: str = "all features", file_prefix: str = "8_all",
                  thres_mode: str = "utility", ocp_alpha: float = 0.10,
                  forecast_horizon: int = 1,
                  split_indices_raw: tuple = None,
                  granularity: str = "1d",
                  ocp_costs: tuple = (0.0, 10.0, 2.0),
                  ocp_window_days: int = 25,
                  best_params: dict | None = None) -> dict:
    """Train on Train, calibrate on Val-Cal, sweep threshold on Val-Opt, evaluate on Test.

    4-way split: Train / Val-Cal / Val-Opt / Test with embargo gaps.
    Calibration → Threshold → Evaluation all operate in calibrated probability space.

    Args:
        split_indices: (idx_train, idx_cal, idx_opt, idx_test) with embargo.
        direction: 'up' or 'down' — used to flip returns for short strategies.
        fee: per-trade fee (decimal, e.g. 0.002 for 0.2%).
    """
    # ┏━━━━━━━━━━ Load Data ━━━━━━━━━━┓
    mlabel = _model_label(model_name)
    _is_dict = isinstance(dataset, dict)
    eng = dataset["eng_features"] if _is_dict else dataset.eng_features
    if isinstance(eng, torch.Tensor):
        eng = eng.numpy()
    labels = dataset["labels"] if _is_dict else dataset.labels
    if isinstance(labels, torch.Tensor):
        labels = labels.numpy()
    returns_all = dataset["returns"] if _is_dict else dataset.returns
    if isinstance(returns_all, torch.Tensor):
        returns_all = returns_all.numpy()
    
    # ┏━━━━━━━━━━ Split Data (4-way: Train / Cal / Opt / Test) ━━━━━━━━━━┓
    idx_train, idx_cal, idx_opt, idx_test = split_indices
    
    # ┏━━━━━━━━━━ M1 Metrics ━━━━━━━━━━┓
    m1_acc_val, m1_prec_val = None, None
    m1_acc_test, m1_prec_test = None, None
    m1_pred_all = None
    m1_true_all = None

    # ┏━━━━━━━━━━ M1 Predictions and True Labels ━━━━━━━━━━┓
    if split_indices_raw is not None:
        from sklearn.metrics import accuracy_score, precision_score
        idx_train_raw, idx_val_raw, idx_test_raw = split_indices_raw

        m1_pred_all = dataset.m1_pred_labels if hasattr(dataset, 'm1_pred_labels') else dataset.get('m1_pred_labels')
        m1_true_all = dataset.m1_true_labels if hasattr(dataset, 'm1_true_labels') else dataset.get('m1_true_labels')

        if m1_pred_all is not None and m1_true_all is not None:
            if isinstance(m1_pred_all, torch.Tensor): m1_pred_all = m1_pred_all.numpy()
            if isinstance(m1_true_all, torch.Tensor): m1_true_all = m1_true_all.numpy()

            # ┏━━━━━━━━━━ Calculate M1 Precision and Accuracy ━━━━━━━━━━┓
            def _calc_m1(idx_raw):
                p, t = m1_pred_all[idx_raw], m1_true_all[idx_raw]
                valid = frozenset(np.where(~np.isnan(p) & ~np.isnan(t))[0])
                if len(valid) > 0:
                    v_idx = list(valid)
                    return accuracy_score(t[v_idx], p[v_idx]), precision_score(t[v_idx], p[v_idx], zero_division=0)
                return None, None

            m1_acc_val, m1_prec_val = _calc_m1(idx_val_raw)
            m1_acc_test, m1_prec_test = _calc_m1(idx_test_raw)

    
    # ┏━━━━━━━━━━ Feature Selection ━━━━━━━━━━┓
    all_names = resolve_feature_names(eng.shape[1])
    col_indices = [all_names.index(c) for c in feature_cols]
    
    # ┏━━━━━━━━━━ Load Train, Val-Cal, Val-Opt, and Test data ━━━━━━━━━━┓
    X_train = eng[idx_train][:, col_indices]
    y_train = labels[idx_train].astype(int)
    X_cal = eng[idx_cal][:, col_indices]
    y_cal = labels[idx_cal].astype(int)
    X_opt = eng[idx_opt][:, col_indices]
    y_opt = labels[idx_opt].astype(int)
    X_test = eng[idx_test][:, col_indices]
    y_test = labels[idx_test].astype(int)
    train_returns = returns_all[idx_train]
    opt_returns = returns_all[idx_opt].copy()
    test_returns = returns_all[idx_test].copy()
    
    # ┏━━━━━━━━━━ Direction-aware returns ━━━━━━━━━━┓
    if direction.lower() == "down":
        opt_returns = -opt_returns
        test_returns = -test_returns
    
    # ┏━━━━━━━━━━ Drop NaN-return rows before eval ━━━━━━━━━━┓
    # Val-Opt
    _opt_valid = ~np.isnan(opt_returns)
    if not _opt_valid.all():
        print(f"  [temporal_eval] WARNING: {(~_opt_valid).sum()} opt rows with NaN returns dropped")
        X_opt = X_opt[_opt_valid]
        y_opt = y_opt[_opt_valid]
        opt_returns = opt_returns[_opt_valid]
        idx_opt = idx_opt[_opt_valid]
    
    # Test
    _test_valid = ~np.isnan(test_returns)
    if not _test_valid.all():
        print(f"  [temporal_eval] WARNING: {(~_test_valid).sum()} test rows with NaN returns dropped")
        X_test = X_test[_test_valid]
        y_test = y_test[_test_valid]
        test_returns = test_returns[_test_valid]
        idx_test = idx_test[_test_valid]
    
    # ┏━━━━━━━━━━ Standardize Features ━━━━━━━━━━┓
    scaler = StandardScaler()
    if model_name not in MODELS_NO_SCALING:
        X_train = scaler.fit_transform(X_train)
        X_cal = scaler.transform(X_cal)
        X_opt = scaler.transform(X_opt)
        X_test = scaler.transform(X_test)
    else:
        scaler.fit(X_train)  # kept for cache compatibility; data stays raw
    
    # ┏━━━━━━━━━━ Class Weights ━━━━━━━━━━┓
    n_pos = int((y_train == 1).sum())
    n_neg = int((y_train == 0).sum())
    cw_ratio = n_neg / max(n_pos, 1)
    
    # ┏━━━━━━━━━━ Build Tree Model + Fit on Train ━━━━━━━━━━┓
    if best_params:
        print(f"    Using HPO best_params for {model_name} ({granularity}/{direction}): {best_params}")
    model = _build_tree_model(model_name,
                              len(y_train),
                              cw_ratio,
                              feature_names=feature_cols,
                              presets=_AG_PRESETS,
                              params=best_params)
    model.fit(X_train, y_train)
    
    # ┏━━━━━━━━━━ Autogluon Leaderboard and Model Info ━━━━━━━━━━┓
    if model_name == "autogluon":
        model.leaderboard()
        model.model_info(save_dir)
    
    results = {}
    val_thresholds = {}
    val_op = None
    
    # ┏━━━━━━━━━━ Pre-compute OCP probs (warm-up) ━━━━━━━━━━┓
    _is_ocp = thres_mode.startswith("OCP")
    # For OCP, use combined Val-Cal+Val-Opt as the "val" warm-up set
    X_val_combined = np.vstack([X_cal, X_opt])
    y_val_combined = np.concatenate([y_cal, y_opt])
    _val_probs_ocp = model.predict_proba(X_val_combined)[:, 1] if _is_ocp else None
    
    # ┏━━━━━━━━━━ Calibration branch (utility_nocal skips it; merges Val-Cal + Val-Opt) ━━━━━━━━━━┓
    _nocal = (thres_mode == "utility_nocal")

    if _nocal:
        # ┏━━━━━━━━━━ No calibrator — use raw probs directly on a merged validation set ━━━━━━━━━━┓
        # Val-Cal + Val-Opt are concatenated into a single optimisation window.
        from Utils.selective_classification.calibration import _IdentityCalibrator
        _calibrator   = _IdentityCalibrator()
        _raw_cal_probs = model.predict_proba(X_cal)[:, 1]
        _raw_opt_probs = model.predict_proba(X_opt)[:, 1]

        # ┏━━━━━━━━━━ Merge Cal + Opt → single "val" window; sweep τ* on raw probs over it ━━━━━━━━━━┓
        _merged_probs   = np.concatenate([_raw_cal_probs, _raw_opt_probs])
        _merged_y       = np.concatenate([y_cal,           y_opt])
        _cal_returns    = returns_all[idx_cal].copy()
        if direction.lower() == "down":
            _cal_returns = -_cal_returns
        _merged_returns = np.concatenate([_cal_returns, opt_returns])

        # ┏━━━━━━━━━━ Sweeps threshold on raw probabilities over the merged Val-Cal + Val-Opt window ━━━━━━━━━━┓
        # M1 precision floor: m1_prec_val is computed on idx_val_raw (full val =
        # Val-Cal + Val-Opt), which exactly matches the optimizer window in nocal mode.
        op = _find_best_utility_threshold(_merged_probs, _merged_returns, fee=fee,
                                          labels=_merged_y, m1_precision=m1_prec_val)
        val_op = op
        val_thresholds["thr"] = op["threshold"]

        # ┏━━━━━━━━━━ Calibrated views (identity) so downstream code paths stay unchanged ━━━━━━━━━━┓
        _cal_opt_probs  = _raw_opt_probs
        _raw_test_probs = model.predict_proba(X_test)[:, 1]
        _cal_test_probs = _raw_test_probs
        
        # ┏━━━━━━━━━━ Store merged dataset so the Val plot can compute U(τ) on the same N the optimizer used ━━━━━━━━━━┓
        _opt_plot_probs   = _merged_probs
        _opt_plot_y       = _merged_y
        _opt_plot_returns = _merged_returns

        print(f"    [utility_nocal] Calibration skipped — Val-Cal + Val-Opt merged "
              f"(n={len(_merged_y)}) for threshold sweep on raw probs.")
        print(f"    Threshold τ={op['threshold']:.3f} (swept on merged raw Val, n={len(_merged_y)})")
    else:
        # ┏━━━━━━━━━━ Calibrate on Val-Cal ━━━━━━━━━━┓
        _raw_cal_probs = model.predict_proba(X_cal)[:, 1]
        _calib = calibrate_probabilities(_raw_cal_probs, y_cal)
        _calibrator = _calib["calibrator"]

        # ┏━━━━━━━━━━ Sweep threshold on calibrated Val-Opt ━━━━━━━━━━┓
        _raw_opt_probs = model.predict_proba(X_opt)[:, 1]
        _cal_opt_probs = _calibrator.predict(_raw_opt_probs)
        # M1 precision floor: m1_prec_val is computed on the full val window
        # (Val-Cal + Val-Opt). Optimizer here runs on Val-Opt only — using the
        # full-val M1 precision is a close, well-defined proxy.
        op = _find_best_utility_threshold(_cal_opt_probs, opt_returns, fee=fee,
                                          labels=y_opt, m1_precision=m1_prec_val)
        val_op = op
        val_thresholds["thr"] = op["threshold"]

        # ┏━━━━━━━━━━ Apply calibration to Test ━━━━━━━━━━┓
        _raw_test_probs = model.predict_proba(X_test)[:, 1]
        _cal_test_probs = _calibrator.predict(_raw_test_probs)
        print(f"    Calibration (isotonic on Cal n={len(y_cal)}):")
        print(
            f"      Opt range  [{_raw_opt_probs.min():.3f}, {_raw_opt_probs.max():.3f}] → [{_cal_opt_probs.min():.3f}, {_cal_opt_probs.max():.3f}]")
        print(
            f"      Test range [{_raw_test_probs.min():.3f}, {_raw_test_probs.max():.3f}] → [{_cal_test_probs.min():.3f}, {_cal_test_probs.max():.3f}]")
        print(f"    Threshold τ={op['threshold']:.3f} (swept on calibrated Opt, n={len(y_opt)})")
        
        # ┏━━━━━━━━━━ Calibrated mode: optimizer dataset = Val-Opt = the plotted Val split → no separate dataset needed ━━━━━━━━━━┓
        _opt_plot_probs   = None
        _opt_plot_y       = None
        _opt_plot_returns = None

    # ┏━━━━━━━━━━ Export Raw & Calibrated Probs (mirroring HPO Phase 0) ━━━━━━━━━━┓
    probs_path = save_dir / "best_probs.npz"
    np.savez(probs_path,
             cal_probs_raw = _raw_cal_probs,
             cal_probs_cal = _calibrator.predict(_raw_cal_probs),
             y_cal         = y_cal,
             opt_probs_raw = _raw_opt_probs,
             opt_probs_cal = _cal_opt_probs,
             y_opt         = y_opt)
    print(f"  [probs] Saved raw+cal probs: {probs_path.name}")
    
    
    # ┏━━━━━━━━━━ Extract dates for SAOCP ━━━━━━━━━━┓
    _ds_dates = dataset["dates"] if _is_dict else dataset.dates
    _opt_dates = np.array([_ds_dates[j] for j in idx_opt])
    _test_dates = np.array([_ds_dates[j] for j in idx_test])
    _val_dates_ocp = np.array([_ds_dates[j] for j in np.concatenate([idx_cal, idx_opt])]) if _is_ocp else None
    _test_dates_ocp = _test_dates if _is_ocp else None
    
    # ┏━━━━━━━━━━ Loop through evaluation splits ━━━━━━━━━━┓
    for split_name, X_split, y_split, split_rets in [("Val", X_opt, y_opt, opt_returns),
                                                     ("Test", X_test, y_test, test_returns)]:
        
        # ┏━━━━━━━━━━ Predictions and Probabilities ━━━━━━━━━━┓
        preds = model.predict(X_split)
        probs = model.predict_proba(X_split)[:, 1]  # P(class=1)
        prec_val = round(float(precision_score(y_split, preds, zero_division=0)), 4)
        n_pred_pos = int((preds == 1).sum())
        
        # ┏━━━━━━━━━━ Metrics ━━━━━━━━━━┓
        metrics = {"accuracy": round(float(accuracy_score(y_split, preds)), 4),
                   "precision": prec_val,
                   "recall": round(float(recall_score(y_split, preds, zero_division=0)), 4),
                   "f1_score": round(float(f1_score(y_split, preds, zero_division=0)), 4),
                   "coverage": round(n_pred_pos / len(y_split), 4) if len(y_split) > 0 else 0,
                   "risk": round(1 - prec_val, 4),
                   "baseline": round(int((y_split == 1).sum()) / len(y_split), 4) if len(y_split) > 0 else 0}
        results[split_name] = metrics
        
        # ┏━━━━━━━━━━ M1 Metrics ━━━━━━━━━━┓
        m1_a = m1_acc_test if split_name == "Test" else m1_acc_val
        m1_p = m1_prec_test if split_name == "Test" else m1_prec_val
        
        # ┏━━━━━━━━━━ Confusion Matrix ━━━━━━━━━━┓
        cm_path = save_dir / f"{file_prefix}_{split_name}_CM.png"
        plot_confusion_matrix(y_split,
                              preds,
                              classes=class_names,
                              save_path=str(cm_path),
                              title=f"{mlabel} {split_name} (@thr=0.5) | {desc}",
                              meta_mode=meta_mode,
                              m1_acc=m1_a,
                              m1_prec=m1_p)
        
        # ┏━━━━━━━━━━ Probability Distribution ━━━━━━━━━━┓
        _plot_prob_distribution(y_split,
                                probs,
                                class_names,
                                save_dir,
                                file_prefix=f"{file_prefix}_{split_name}_Prob_Dist",
                                title=f"{mlabel} {split_name} ({desc})")
        
        # ┏━━━━━━━━━━ Risk-coverage curve (all in calibrated space) ━━━━━━━━━━┓
        if split_name == "Test" and not _is_ocp:
            score_probs = _cal_test_probs
        elif split_name == "Val" and not _is_ocp:
            score_probs = _cal_opt_probs
        else:
            score_probs = probs
        
        # ┏━━━━━━━━━━ Collect Risk-Coverage Curve ━━━━━━━━━━┓
        curve = collect_risk_coverage_curve(y_true=y_split, y_score=score_probs)
        
        # ┏━━━━━━━━━━ Val: threshold already computed above, just record metrics ━━━━━━━━━━┓
        if split_name == "Val":
            sel_val = _cal_opt_probs >= op["threshold"]
            n_sel_val = int(sel_val.sum())
            err_val = int((y_split[sel_val] == 0).sum()) if n_sel_val > 0 else 0
            op["risk"] = err_val / max(n_sel_val, 1)
        else:
            if _is_ocp:
                # ┏━━━━━━━━━━ OCP dispatch: choose variant ━━━━━━━━━━┓
                _cw = calib_window_for_gran(granularity, ocp_window_days)
                c_FN, c_FP, c_DEF = ocp_costs
                
                # ┏━━━━━━━━━━ Cost-aware deferral (vanilla or Mondrian) ━━━━━━━━━━┓
                if thres_mode in ("OCP-cost", "OCP-cost-mondrian"):
                    test_s_hats, test_approved_ocp, val_s_hats, conf_stats = _run_cost_deferral_online(_val_probs_ocp,
                                                                                                       y_val_combined,
                                                                                                       probs,
                                                                                                       y_split,
                                                                                                       alpha=ocp_alpha,
                                                                                                       test_dates=_test_dates_ocp,
                                                                                                       forecast_horizon=forecast_horizon,
                                                                                                       val_dates=_val_dates_ocp,
                                                                                                       calib_window=_cw,
                                                                                                       c_FP=c_FP,
                                                                                                       c_FN=c_FN,
                                                                                                       c_DEF=c_DEF,
                                                                                                       mondrian=(
                                                                                                               thres_mode == "OCP-cost-mondrian"),
                                                                                                       test_returns=split_rets)
                
                # ┏━━━━━━━━━━ Windowed SAOCP ━━━━━━━━━━┓
                elif thres_mode == "OCP-W":
                    test_s_hats, test_approved_ocp, val_s_hats, conf_stats = _run_saocp_online(_val_probs_ocp,
                                                                                               y_val_combined,
                                                                                               probs,
                                                                                               y_split,
                                                                                               alpha=ocp_alpha,
                                                                                               test_dates=_test_dates_ocp,
                                                                                               forecast_horizon=forecast_horizon,
                                                                                               val_dates=_val_dates_ocp,
                                                                                               calib_window=_cw)
                # ┏━━━━━━━━━━ Original OCP (unbounded history) ━━━━━━━━━━┓
                else:
                    test_s_hats, test_approved_ocp, val_s_hats, conf_stats = _run_saocp_online(_val_probs_ocp,
                                                                                               y_val_combined,
                                                                                               probs,
                                                                                               y_split,
                                                                                               alpha=ocp_alpha,
                                                                                               test_dates=_test_dates_ocp,
                                                                                               forecast_horizon=forecast_horizon,
                                                                                               val_dates=_val_dates_ocp)
                # ┏━━━━━━━━━━ Convert OCP thresholds to operating point ━━━━━━━━━━┓
                op = _ocp_threshold_to_op(probs, y_split, split_rets,
                                          test_approved_ocp, test_s_hats, fee,
                                          conformal_stats=conf_stats)
                op["threshold_source"] = thres_mode  # e.g. OCP, OCP-W, OCP-cost, OCP-cost-mondrian
                
                # ┏━━━━━━━━━━ Store for backtest ━━━━━━━━━━┓    
                val_op["_ocp_test_approved"] = test_approved_ocp
                val_op["_ocp_test_thresholds"] = test_s_hats
                val_op["_ocp_val_thresholds"] = val_s_hats
                cc = conf_stats["conformal_coverage"]
                mode_tag = thres_mode
                print(f"    {mode_tag}: α={ocp_alpha}, median τ={op['threshold']:.3f}, "
                      f"cov={op['coverage']:.1%}, μ={op['mean_ret'] * 100:+.3f}% | "
                      f"Conformal cov={cc:.1%} (target≥{1 - ocp_alpha:.0%}) | "
                      f"Sets: {{1}}={conf_stats['n_set_1']} {{0}}={conf_stats['n_set_0']} "
                      f"{{0,1}}={conf_stats['n_set_both']} {{}}={conf_stats['n_set_empty']}")
                if "tau_trajectory" in conf_stats:
                    tau_std = float(np.std(conf_stats["tau_trajectory"]))
                    print(f"    Cost params: c_FN={c_FN}, c_FP={c_FP}, c_DEF={c_DEF} | "
                          f"τ* std={tau_std:.4f}")
                
                # ┏━━━━━━━━━━ Save OCP diagnostics npz for offline analysis ━━━━━━━━━━┓
                np.savez_compressed(save_dir / f"{file_prefix}_{split_name}_ocp_diagnostics.npz",
                                    test_s_hats=test_s_hats,
                                    val_s_hats=val_s_hats,
                                    val_probs=_val_probs_ocp,
                                    val_labels=y_val_combined.astype(int),
                                    alpha=np.array([ocp_alpha]),
                                    **({"tau_trajectory": conf_stats[
                                        "tau_trajectory"]} if "tau_trajectory" in conf_stats else {}))
                
                # ┏━━━━━━━━━━ Mondrian / cost-deferral regime diagnostics ━━━━━━━━━━┓
                if split_name == "Test" and "mondrian_diag" in conf_stats:
                    plot_mondrian_diagnostics(conf_stats,
                                              save_dir,
                                              gran_label=granularity,
                                              thres_mode=thres_mode)
            else:
                # ┏━━━━━━━━━━ Utility: apply fixed Val threshold to test (calibrated test probs) ━━━━━━━━━━┓
                thr = val_thresholds["thr"]
                sel = _cal_test_probs >= thr
                n_sel = int(sel.sum())
                err = int((y_split[sel] == 0).sum()) if n_sel > 0 else 0
                net_rets_test = split_rets[sel] - fee if n_sel > 0 else np.array([0.0])
                mu_test = float(np.nanmean(net_rets_test))
                sigma_test = float(np.nanstd(net_rets_test, ddof=1)) if n_sel > 1 else 0.0
                t_test = mu_test / sigma_test * np.sqrt(n_sel) if sigma_test > 0 else 0.0
                
                # ┏━━━━━━━━━━ Operating Threshold ━━━━━━━━━━┓
                op = {"threshold": thr,
                      "coverage": n_sel / len(y_split),
                      "risk": err / max(n_sel, 1),
                      "selected_count": n_sel,
                      "constraint_satisfied": val_op.get("constraint_satisfied", True),
                      "threshold_source": val_op.get("threshold_source", thres_mode),
                      "mean_ret": mu_test,
                      "t_stat": t_test}
        
        # ┏━━━━━━━━━━ Plot risk-coverage with return overlay (business-level) ━━━━━━━━━━┓
        rc_path = save_dir / f"{file_prefix}_{split_name}_Risk_Coverage.png"
        rc_final_path = save_dir / f"{file_prefix}_{split_name}_Risk_Coverage_final.png"
        
        # ┏━━━━━━━━━━ Plot risk-coverage (paper-level) ━━━━━━━━━━┓
        plot_temporal_risk_coverage_curve_final(save_path         = rc_final_path,
                                                curve             = curve,
                                                probs             = score_probs,
                                                y_true            = y_split,
                                                split_rets        = split_rets,
                                                fee               = fee,
                                                op                = op,
                                                split_name        = split_name,
                                                model_label       = mlabel,
                                                thres_mode        = thres_mode,
                                                ocp_alpha         = ocp_alpha,
                                                val_threshold     = val_thresholds.get("thr"),
                                                val_op            = val_op,
                                                is_ocp            = _is_ocp,
                                                test_approved_ocp = test_approved_ocp if (split_name == "Test" and _is_ocp) else None,
                                                opt_probs         = _opt_plot_probs if split_name == "Val" else None,
                                                opt_y             = _opt_plot_y if split_name == "Val" else None,
                                                opt_rets          = _opt_plot_returns if split_name == "Val" else None,
                                                direction         = direction,
                                                granularity       = granularity,
                                                m1_precision      = m1_prec_val if split_name == "Val" else m1_prec_test)

        # ┏━━━━━━━━━━ Plot temporal risk coverage curve (basic) ━━━━━━━━━━┓
        plot_temporal_risk_coverage_curve(save_path         = rc_path,
                                          curve             = curve,
                                          probs             = score_probs,
                                          y_true            = y_split,
                                          split_rets        = split_rets,
                                          fee               = fee,
                                          op                = op,
                                          split_name        = split_name,
                                          model_label       = mlabel,
                                          thres_mode        = thres_mode,
                                          ocp_alpha         = ocp_alpha,
                                          val_threshold     = val_thresholds.get("thr"),
                                          val_op            = val_op,
                                          is_ocp            = _is_ocp,
                                          test_approved_ocp = test_approved_ocp if (split_name == "Test" and _is_ocp) else None,
                                          direction         = direction,
                                          granularity       = granularity)
        
        # ┏━━━━━━━━━━ OCP threshold evolution plot (test only) ━━━━━━━━━━┓
        if split_name == "Test" and _is_ocp:
            thr_evo_path = save_dir / f"{file_prefix}_Test_OCP_Threshold_Evolution.png"
            plot_ocp_threshold_evolution(save_path=thr_evo_path,
                                         test_s_hats=test_s_hats,
                                         utility_threshold=val_thresholds["thr"],
                                         model_label=mlabel,
                                         thres_mode=thres_mode,
                                         ocp_alpha=ocp_alpha,
                                         conformal_coverage=op.get("conformal_coverage", 0),
                                         n_set_1=op.get("n_set_1", 0),
                                         n_set_0=op.get("n_set_0", 0),
                                         n_set_both=op.get("n_set_both", 0),
                                         n_set_empty=op.get("n_set_empty", 0),
                                         split_name=split_name)
        
        # ┏━━━━━━━━━━ Print and store selective metrics + confusion matrix ━━━━━━━━━━┓
        thr_sel = op["threshold"]
        if split_name == "Test" and _is_ocp:
            sel = test_approved_ocp
        else:
            sel = (score_probs >= thr_sel)
        sel_preds = sel.astype(int)
        sel_true = y_split
        
        # ┏━━━━━━━━━━ Information for Confusion Matrix ━━━━━━━━━━┓
        sel_cm_path = save_dir / f"{file_prefix}_{split_name}_Selective_CM.png"
        thr_src = op.get("threshold_source", thres_mode)
        if _is_ocp and split_name == "Test":
            cc = op.get("conformal_coverage", 0)
            sel_title = (f"{mlabel} {split_name} selective @thr={thr_sel:.3f} ({thr_src})\n"
                         f"Conformal Cov={cc:.1%} (target≥{1 - ocp_alpha:.0%})")
        else:
            sel_title = f"{mlabel} {split_name} selective @thr={thr_sel:.3f} ({thr_src})"
        
        # ┏━━━━━━━━━━ Plot Test Confusion Matrix ━━━━━━━━━━┓
        plot_confusion_matrix(sel_true,
                              sel_preds,
                              classes=class_names,
                              save_path=str(sel_cm_path),
                              title=sel_title,
                              meta_mode=meta_mode,
                              is_selective=True,
                              m1_acc=m1_a,
                              m1_prec=m1_p)
        
        n_sel = int(sel.sum())
        risk = int((y_split[sel] == 0).sum()) / max(n_sel, 1) if n_sel > 0 else 0
        print(f"    {split_name}: acc={metrics['accuracy']:.3f} prec={metrics['precision']:.3f} "
              f"rec={metrics['recall']:.3f} f1={metrics['f1_score']:.3f} "
              f"| selective @thr={thr_sel:.3f} ({thr_src}): "
              f"cov={op['coverage']:.1%} t-stat={op['t_stat']:.2f} "
              f"μ={op['mean_ret'] * 100:+.3f}% n={op['selected_count']}")
        
        # ┏━━━━━━━━━━ Selective Metrics ━━━━━━━━━━┓
        sel_dict = {"threshold": round(thr_sel, 4),
                    "coverage": round(op["coverage"], 4),
                    "risk": round(risk, 4),
                    "precision": round(1 - risk, 4),
                    "selected_count": op["selected_count"],
                    "threshold_source": thr_src,
                    "constraint_satisfied": op["constraint_satisfied"],
                    "mean_ret": round(op["mean_ret"], 6),
                    "t_stat": round(op["t_stat"], 4)}
        
        # ┏━━━━━━━━━━ OCP Metrics ━━━━━━━━━━┓
        if _is_ocp and split_name == "Test":
            sel_dict["ocp"] = {"alpha": ocp_alpha,
                               "conformal_coverage": round(op.get("conformal_coverage", 0), 4),
                               "target_coverage": round(1 - ocp_alpha, 4),
                               "guarantee_met": op.get("conformal_coverage", 0) >= (1 - ocp_alpha),
                               "n_set_1_trade": op.get("n_set_1", 0),
                               "n_set_0_dont_trade": op.get("n_set_0", 0),
                               "n_set_both_abstain": op.get("n_set_both", 0),
                               "n_set_empty_abstain": op.get("n_set_empty", 0)}
            
            if conf_stats.get("regime_stats"):
                sel_dict["regime_stats"] = conf_stats["regime_stats"]
        results[f"{split_name}_selective"] = sel_dict
        
        # ┏━━━━━━━━━━ M2 Selective Return Distribution (Test only) ━━━━━━━━━━┓
        if split_name == "Test":
            ret_dist_path = save_dir / f"{file_prefix}_Test_Selective_RetDist.png"
            plot_selective_return_distribution(
                test_returns=split_rets,
                test_labels=y_split,
                m2_approved=sel,
                save_path=ret_dist_path,
                fee=fee,
                direction=direction,
                granularity=granularity,
                model_label=mlabel)
    
    artifacts = {"model": model,
                 "scaler": scaler,
                 "col_indices": col_indices,
                 "val_op": val_op,
                 "calibrator": _calibrator}
    
    return results, artifacts


# ------------------------------------------------------------------------------
# Run analysis for a Single M2 (each granularity independently)
# ------------------------------------------------------------------------------
def run_analysis(cache_path: Path, direction: str, mode: str, granularity: str, output_root: Path,
                 train_end: str = None, val_end: str = None, model_name: str = "rf",
                 cfg: dict = None, run_top5: bool = True,
                 run_features: bool = True, thres_mode: str = "utility",
                 ocp_alpha: float = 0.10,
                 ocp_costs: tuple = (0.0, 10.0, 2.0),
                 ocp_window_days: int = 25,
                 best_params: dict | None = None):
    """Run full feature analysis for one direction/granularity.

    Args:
        dataset_override: if provided, use this dict instead of loading cache_path.
                          Used when iterating sub-granularities from a multi cache.
    """
    
    # ┏━━━━━━━━━━ Load dataset ━━━━━━━━━━┓
    mlabel = _model_label(model_name)
    class_names = _class_names(direction, mode)
    
    # ┏━━━━━━━━━━ Determine thresholding folder based on mode ━━━━━━━━━━┓
    if thres_mode.startswith("OCP"):
        thres_folder = thres_mode
    elif thres_mode == "utility_nocal":
        thres_folder = "Utility_Score_NoCal"
    else:
        thres_folder = "Utility_Score"
    
    # ┏━━━━━━━━━━ Create save directory ━━━━━━━━━━┓
    save_dir = Path(output_root) / cfg["experiment"]["m1"].capitalize() / model_name / direction.upper() / thres_folder / f"{granularity}_{mode}"
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # ┏━━━━━━━━━━ Print header ━━━━━━━━━━┓
    print(f"\n{'=' * 60}")
    print(f"[kronos_tree] Direction: {direction} | Mode: {mode} | Granularity: {granularity}")
    print(f"[kronos_tree] Model: {mlabel} | Classes: {class_names}")
    print(f"[kronos_tree] Cache: {cache_path.name}")
    print(f"[kronos_tree] Output: {save_dir}")
    print(f"{'=' * 60}\n")
    
    # ┏━━━━━━━━━━ Load dataset ━━━━━━━━━━┓
    dataset = torch.load(cache_path, weights_only=False)

    # ┏━━━━━━━━━━ Check if cache contains eng_features (engineered features) ━━━━━━━━━━┓
    _eng_check = dataset.get("eng_features") if isinstance(dataset, dict) else getattr(dataset, "eng_features", None)
    if _eng_check is None:
        print("ERROR: Cache does not contain 'eng_features'. Rebuild cache with engineered features enabled.")
        return

    # ┏━━━━━━━━━━ Filter multi-gran cache to the requested granularity ━━━━━━━━━━┓
    dataset = _filter_dataset_by_granularity(dataset, granularity)
    
    # ┏━━━━━━━━━━ Build dataframe ━━━━━━━━━━┓
    df_all, labels_all = _build_dataframe(dataset)
    
    # ┏━━━━━━━━━━ Get train-only subset for feature analysis (no val/test leakage in feature selection) ━━━━━━━━━━┓
    df_val_fs = None  # validation DataFrame for MDA/SHAP/LIME (None if no split)
    labels_val_fs = None
    if train_end and val_end:
        # ┏━━━━━━━━━━ Split by global time ━━━━━━━━━━┓
        idx_train, _, idx_val, _ = split_by_global_time(dataset, train_end=train_end, val_end=val_end)
        
        # ┏━━━━━━━━━━ Convert eng_features to numpy ━━━━━━━━━━┓
        _eng_f = dataset["eng_features"] if isinstance(dataset, dict) else dataset.eng_features
        _lbl_f = dataset["labels"] if isinstance(dataset, dict) else dataset.labels
        eng_raw = _eng_f.numpy() if isinstance(_eng_f, torch.Tensor) else _eng_f
        labels_raw = _lbl_f.numpy() if isinstance(_lbl_f, torch.Tensor) else _lbl_f
        
        # ┏━━━━━━━━━━ Build dataframe ━━━━━━━━━━┓
        _feat_names = resolve_feature_names(eng_raw.shape[1])
        df_train = pd.DataFrame(eng_raw[idx_train], columns=_feat_names)
        labels_train = labels_raw[idx_train].astype(int)
        
        # ┏━━━━━━━━━━ Validation set for MDA/SHAP/LIME feature selection ━━━━━━━━━━┓
        if len(idx_val) > 0:
            df_val_fs = pd.DataFrame(eng_raw[idx_val], columns=_feat_names)
            labels_val_fs = labels_raw[idx_val].astype(int)
        
        print(f"[kronos_tree] Total samples: {len(df_all)} | Train-only for feature analysis: {len(df_train)}"
              f" | Val for FS: {len(idx_val)}")
    else:
        # ┏━━━━━━━━━━ No temporal split dates, using all data ━━━━━━━━━━┓
        df_train = df_all
        labels_train = labels_all
        print(f"[kronos_tree] Samples: {len(df_all)} (no temporal split dates, using all data)")
    
    # ┏━━━━━━━━━━ Print train statistics ━━━━━━━━━━┓
    print(
        f"[kronos_tree] Train: {len(df_train)} (class 0: {(labels_train == 0).sum()}, class 1: {(labels_train == 1).sum()})")
    print(f"[kronos_tree] Features: {len(df_train.columns)}\n")
    
    # ┏━━━━━━━━━━ Drop features that are all-NaN or zero-variance (based on train) ━━━━━━━━━━┓
    to_drop = [c for c in df_train.columns if df_train[c].isna().all() or df_train[c].std() == 0]
    if to_drop:
        print(f"[kronos_tree] Dropping zero-variance/all-NaN features: {to_drop}\n")
        df_train = df_train.drop(columns=to_drop)
    
    # ┏━━━━━━━━━━ Steps 1-6: feature analysis (correlation, MI, importance, rank aggregation) ━━━━━━━━━━┓
    tree_metrics = {}
    tree_top5_metrics = {}
    top_result = {}
    top5 = []
    
    if run_features:
        # ┏━━━━━━━━━━ Create features analysis directory ━━━━━━━━━━┓
        features_dir = save_dir / "Features_Analysis"
        features_dir.mkdir(parents=True, exist_ok=True)
        
        # ┏━━━━━━━━━━ Plot correlation heatmap ━━━━━━━━━━┓
        plot_correlation_heatmap(df_train, features_dir)
        
        # ┏━━━━━━━━━━ Plot pointbiserial ━━━━━━━━━━┓
        pb_scores = plot_pointbiserial(df_train, labels_train, class_names, features_dir)
        
        # ┏━━━━━━━━━━ Plot class distributions ━━━━━━━━━━┓
        plot_class_distributions(df_train, labels_train, class_names, features_dir)
        
        # ┏━━━━━━━━━━ Plot mutual information ━━━━━━━━━━┓
        mi_scores = plot_mutual_information(df_train, labels_train, features_dir)
        
        # ┏━━━━━━━━━━ Plot tree importance ━━━━━━━━━━┓
        tree_scores, tree_metrics = plot_tree_importance(df_train,
                                                         labels_train,
                                                         features_dir,
                                                         model_name=model_name,
                                                         class_names=class_names,
                                                         meta_mode=mode,
                                                         model_builder=lambda name,
                                                                              n_samples,
                                                                              ratio: _build_tree_model(name, n_samples,
                                                                                                       ratio),
                                                         model_labeler=_model_label)
        
        # ┏━━━━━━━━━━ Compute top features ━━━━━━━━━━┓
        corr_matrix = df_train.corr()
        top_result = compute_top_features(pb_scores, mi_scores, tree_scores, corr_matrix, features_dir)
        top5 = top_result["top_5_features"]
        
        # ┏━━━━━━━━━━ Tree-model on top-5 only (train only) ━━━━━━━━━━┓
        if run_top5:
            # ┏━━━━━━━━━━ Create top features directory ━━━━━━━━━━┓
            top_features_dir = save_dir / "Top_Features"
            top_features_dir.mkdir(parents=True, exist_ok=True)
            
            # ┏━━━━━━━━━━ Build dataframe with top-5 features ━━━━━━━━━━┓
            df_top5 = df_train[top5]
            
            # ┏━━━━━━━━━━ Plot tree importance on top-5 features ━━━━━━━━━━┓
            _, tree_top5_metrics = plot_tree_importance(df_top5,
                                                        labels_train,
                                                        top_features_dir,
                                                        model_name=model_name,
                                                        step_label="7/9",
                                                        desc="top-5 only",
                                                        file_prefix="7_top5_feature_importance",
                                                        class_names=class_names,
                                                        meta_mode=mode,
                                                        model_builder=lambda name, n_samples, ratio: _build_tree_model(
                                                            name, n_samples, ratio),
                                                        model_labeler=_model_label)
        
        # ┏━━━━━━━━━━ MDA / SHAP / LIME Feature Selection (requires val split) ━━━━━━━━━━┓
        fs_result = {}
        fs_cfg = cfg.get("data", {}).get("features", {}).get("feature_selection", {}) if cfg else {}
        fs_methods = fs_cfg.get("methods", ["mda", "shap", "lime"])
        fs_top_k = fs_cfg.get("top_k", None)
        run_fs = fs_cfg.get("enabled", True)
        
        if run_fs and df_val_fs is not None and len(df_val_fs) > 0:
            fs_dir = features_dir / "Feature_Selection"
            fs_dir.mkdir(parents=True, exist_ok=True)
            fs_result = run_feature_selection(df_train=df_train,
                                              labels_train=labels_train,
                                              df_val=df_val_fs,
                                              labels_val=labels_val_fs,
                                              save_dir=fs_dir,
                                              model_builder=lambda name, n_samples, ratio: _build_tree_model(name,
                                                                                                             n_samples,
                                                                                                             ratio),
                                              model_name=model_name,
                                              fs_methods=fs_methods,
                                              top_k=fs_top_k,
                                              verbose=True)
        elif run_fs:
            print("  [FS] Skipping MDA/SHAP/LIME — no validation split available (need train_end + val_end).")
    else:
        print(f"  [--features=false] Skipping feature analysis (steps 1-7)")
    
    # ┏━━━━━━━━━━ Steps 8-9: temporal train/val/test evaluation ━━━━━━━━━━┓
    temporal_all = {}
    temporal_top5 = {}
    backtest_all = {}
    backtest_top5 = {}
    if train_end and val_end:
        ((idx_train_t, idx_train_raw),
         (idx_meta_t, idx_meta_raw),
         (idx_val_t, idx_val_raw),
         (idx_test_t, idx_test_raw)) = split_by_global_time(dataset,
                                                            train_end=train_end,
                                                            val_end=val_end,
                                                            return_raw=True)
        
        # ┏━━━━━━━━━━ Compute 4-way split with embargo ━━━━━━━━━━┓
        from Utils.edge import _compute_embargo_splits, _gran_to_timedelta
        
        # ┏━━━━━━━━━━ Get Dates and Labels ━━━━━━━━━━┓
        fh = int(cfg.get("data", {}).get("load", {}).get("forecast_horizon", 7)) if cfg else 7
        dates_all = dataset["dates"] if isinstance(dataset, dict) else dataset.dates
        labels_all = dataset["labels"] if isinstance(dataset, dict) else dataset.labels
        if isinstance(labels_all, torch.Tensor): labels_all = labels_all.numpy()
        valid_mask = ~np.isnan(labels_all)
        dates_valid = [dates_all[i] for i in range(len(dates_all)) if valid_mask[i]]
        
        # ┏━━━━━━━━━━ Compute Embargo Splits ━━━━━━━━━━┓
        embargo = _compute_embargo_splits(dates_valid, train_end, val_end, fh, granularity)
        
        # ┏━━━━━━━━━━ Map embargo indices (into valid-only) to full-dataset indices ━━━━━━━━━━┓
        valid_indices = np.where(valid_mask)[0]
        idx_train_e = valid_indices[embargo["idx_train"]]
        idx_cal_e = valid_indices[embargo["idx_cal"]]
        idx_opt_e = valid_indices[embargo["idx_opt"]]
        idx_test_e = valid_indices[embargo["idx_test"]]
        split_indices = (idx_train_e, idx_cal_e, idx_opt_e, idx_test_e)
        split_indices_raw = (idx_train_raw, idx_val_raw, idx_test_raw)
        
        # ┏━━━━━━━━━━ Get Purge and Cal End ━━━━━━━━━━┓
        purge_td = embargo["purge_td"]
        cal_end = embargo["cal_end"]
        
        # ┏━━━━━━━━━━ Print Split Info ━━━━━━━━━━┓
        print(
            f"    4-way split: train={len(idx_train_e):,}  cal={len(idx_cal_e):,}  opt={len(idx_opt_e):,}  test={len(idx_test_e):,}")
        print(f"    Cal end: {cal_end.strftime('%Y-%m-%d')}  |  Purge: {purge_td}")
        
        # ┏━━━━━━━━━━ Fee per trade ━━━━━━━━━━┓
        fee = cfg.get("evaluation", {}).get("fee_per_trade", 0.0) if cfg is not None else 0.0
        
        # ┏━━━━━━━━━━ Active features ━━━━━━━━━━┓
        active_features = list(df_train.columns)
        print(f"\n  [8/9] {mlabel} temporal split ({len(active_features)} feats, train<={train_end}, val<={val_end}):")
        
        # ┏━━━━━━━━━━ Temporal evaluation on all features ━━━━━━━━━━┓
        temporal_all, artifacts_all = temporal_eval(dataset,
                                                    active_features,
                                                    save_dir,
                                                    class_names=class_names,
                                                    meta_mode=mode,
                                                    split_indices=split_indices,
                                                    direction=direction,
                                                    fee=fee,
                                                    model_name=model_name,
                                                    desc="all features",
                                                    file_prefix="8_all",
                                                    thres_mode=thres_mode,
                                                    ocp_alpha=ocp_alpha,
                                                    forecast_horizon=fh,
                                                    split_indices_raw=split_indices_raw,
                                                    granularity=granularity,
                                                    ocp_costs=ocp_costs,
                                                    ocp_window_days=ocp_window_days,
                                                    best_params=best_params)

        # ┏━━━━━━━━━━ Persist final production model (train+cal+opt fitted model used for Test) ━━━━━━━━━━┓
        _save_final_model(artifacts_all,
                          save_dir=save_dir / "final_model",
                          model_name=model_name,
                          features_used=active_features,
                          best_params=best_params,
                          meta={"granularity":     granularity,
                                "direction":       direction,
                                "meta_label_mode": mode,
                                "thres_mode":      thres_mode,
                                "m1":              cfg.get("data", {}).get("load", {}).get("m1") if cfg else None})

        # TODO: Change the selection when Till creates pipeline
        if run_top5:
            # ┏━━━━━━━━━━ Create top features directory ━━━━━━━━━━┓
            top_features_dir = save_dir / "Top_Features"
            top_features_dir.mkdir(parents=True, exist_ok=True)
            print(f"  [9/9] {mlabel} temporal split (top-5, train<={train_end}, val<={val_end}):")
            
            # ┏━━━━━━━━━━ Temporal evaluation on top-5 features ━━━━━━━━━━┓
            temporal_top5, artifacts_top5 = temporal_eval(dataset,
                                                          top5,
                                                          top_features_dir,
                                                          class_names=class_names,
                                                          meta_mode=mode,
                                                          split_indices=split_indices,
                                                          direction=direction,
                                                          fee=fee,
                                                          model_name=model_name,
                                                          desc="top-5 only",
                                                          file_prefix="9_top5",
                                                          thres_mode=thres_mode,
                                                          ocp_alpha=ocp_alpha,
                                                          forecast_horizon=fh,
                                                          split_indices_raw=split_indices_raw,
                                                          granularity=granularity,
                                                          ocp_costs=ocp_costs,
                                                          ocp_window_days=ocp_window_days,
                                                          best_params=best_params)
        
        # ┏━━━━━━━━━━ Financial Backtest (Equity Curves + ROI Reports) ━━━━━━━━━━┓
        if cfg is not None:
            print(f"\n  [10/11] {mlabel} financial backtest (all features):")
            
            # ┏━━━━━━━━━━ Financial backtest on all features ━━━━━━━━━━┓
            backtest_all = run_feature_backtest(dataset,
                                                split_indices,
                                                artifacts_all,
                                                cfg,
                                                save_dir,
                                                class_names=class_names,
                                                meta_mode=mode,
                                                granularity=granularity,
                                                direction=direction,
                                                model_name=model_name,
                                                desc="all features",
                                                file_prefix="10_backtest_all",
                                                fee=fee,
                                                thres_mode=thres_mode,
                                                ocp_alpha=ocp_alpha)
            
            if run_top5:
                print(f"\n  [11/11] {mlabel} financial backtest (top-5):")
                
                # ┏━━━━━━━━━━ Financial backtest on top-5 features ━━━━━━━━━━┓
                backtest_top5 = run_feature_backtest(dataset,
                                                     split_indices,
                                                     artifacts_top5,
                                                     cfg,
                                                     top_features_dir,
                                                     class_names=class_names,
                                                     meta_mode=mode,
                                                     granularity=granularity,
                                                     direction=direction,
                                                     model_name=model_name,
                                                     desc="top-5 only",
                                                     file_prefix="11_backtest_top5",
                                                     fee=fee,
                                                     thres_mode=thres_mode,
                                                     ocp_alpha=ocp_alpha)
    
    # ┏━━━━━━━━━━ Save summary JSON ━━━━━━━━━━┓
    summary = {"cache": str(cache_path),
               "m1_model": cfg.get("data", {}).get("load", {}).get("m1", "kronos") if cfg else "kronos",
               "granularity": granularity,
               "direction": direction,
               "meta_label_mode": mode,
               "model": model_name,
               "n_samples_total": len(df_all),
               "n_samples_train": len(df_train),
               "n_features": len(df_train.columns),
               "class_distribution_train": {class_names[0]: int((labels_train == 0).sum()),
                                            class_names[1]: int((labels_train == 1).sum())},
               "features_used": list(df_train.columns),
               "dropped_features": to_drop,
               f"{model_name}_all_features_5fold_cv": tree_metrics,
               f"{model_name}_top5_features_5fold_cv": tree_top5_metrics,
               f"{model_name}_temporal_all_features": temporal_all,
               f"{model_name}_temporal_top5_features": temporal_top5,
               f"{model_name}_backtest_all_features": backtest_all,
               f"{model_name}_backtest_top5_features": backtest_top5,
               **top_result}
    
    # ┏━━━━━━━━━━ Save summary JSON ━━━━━━━━━━┓
    with open(save_dir / "analysis_summary.json", "w") as f:
        json.dump(summary, f, indent=2, cls=NumpyJSONEncoder)
    
    # TODO save model -- currently only the unified M2 is saved
    
    print(f"\nDone ({direction}). Outputs in: {save_dir}")


# ------------------------------------------------------------------------------
# Run analysis for Unified M2 (all granularities together but separate evaluation)
# ------------------------------------------------------------------------------
def run_unified_analysis(cache_path: Path,
                         multi: bool,
                         direction: str,
                         mode: str,
                         output_root: Path,
                         train_end: str,
                         val_end: str,
                         model_name: str = "rf",
                         cfg: dict = None,
                         thres_mode: str = "utility",
                         ocp_alpha: float = 0.10,
                         ocp_costs: tuple = (0.0, 10.0, 2.0),
                         ocp_window_days: int = 25):
    """Train ONE model on all granularities (with gran-balancing weights), evaluate per-gran."""
    
    # ┏━━━━━━━━━━ Models and Path Setup ━━━━━━━━━━┓
    mlabel = _model_label(model_name)
    class_names = _class_names(direction, mode)
    
    # ┏━━━━━━━━━━ Determine thresholding folder based on mode ━━━━━━━━━━┓
    if thres_mode.startswith("OCP"):
        thres_folder = thres_mode
    elif thres_mode == "utility_nocal":
        thres_folder = "Utility_Score_NoCal"
    else:
        thres_folder = "Utility_Score"
    save_dir = output_root / _m1_output_bucket(
        cfg) / model_name / direction.upper() / thres_folder / f"unified_{mode}"
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # ┏━━━━━━━━━━ Fee and Horizon ━━━━━━━━━━┓
    fee = cfg.get("evaluation", {}).get("fee_per_trade", 0.0) if cfg else 0.0
    horizon = int(cfg.get("data", {}).get("load", {}).get("forecast_horizon", 7))
    
    # ┏━━━━━━━━━━ Print Info ━━━━━━━━━━┓
    print(f"\n{'=' * 60}")
    print(f"[kronos_tree] UNIFIED MODEL — Direction: {direction} | Mode: {mode}")
    print(f"[kronos_tree] Granularities: {multi.grans}")
    print(f"[kronos_tree] Model: {mlabel} | Classes: {class_names}")
    print(f"[kronos_tree] Output: {save_dir}")
    print(f"{'=' * 60}\n")
    
    # ┏━━━━━━━━━━ Collect data from all granularities with one-hot encoding ━━━━━━━━━━┓
    gran_list = list(multi.grans)
    gran_onehot_names = [f"is_{g}" for g in gran_list]
    
    # ┏━━━━━━━━━━ Initialize lists ━━━━━━━━━━┓
    all_X_train, all_y_train, all_w_train = [], [], []
    per_gran_data = {}  # for per-gran evaluation later
    
    # ┏━━━━━━━━━━ Loop through granularities ━━━━━━━━━━┓
    for gi, gran in enumerate(gran_list):
        # ┏━━━━━━━━━━ Get data for this granularity ━━━━━━━━━━┓
        sub = multi.sub[gran]
        eng = sub["eng_features"].numpy() if isinstance(sub["eng_features"], torch.Tensor) else sub["eng_features"]
        labels = sub["labels"].numpy() if isinstance(sub["labels"], torch.Tensor) else sub["labels"]
        returns = sub["returns"].numpy() if isinstance(sub["returns"], torch.Tensor) else sub["returns"]
        asset_ids = sub["asset_ids"].numpy() if isinstance(sub["asset_ids"], torch.Tensor) else sub["asset_ids"]
        
        # ┏━━━━━━━━━━ Split data into train, val, test ━━━━━━━━━━┓
        idx_train, _, idx_val, idx_test = split_by_global_time(sub, train_end=train_end, val_end=val_end)
        
        # ┏━━━━━━━━━━ One-hot for this granularity ━━━━━━━━━━┓
        onehot = np.zeros((len(eng), len(gran_list)), dtype=np.float32)
        onehot[:, gi] = 1.0
        X_full = np.hstack([eng, onehot])
        
        # ┏━━━━━━━━━━ Train data ━━━━━━━━━━┓
        X_tr = X_full[idx_train]
        y_tr = labels[idx_train].astype(int)
        valid_tr = ~np.isnan(y_tr)  # Remove NaN values
        X_tr, y_tr = X_tr[valid_tr], y_tr[valid_tr]
        
        # ┏━━━━━━━━━━ Granularity-balancing weight: 1/N_gran so each gran contributes equally ━━━━━━━━━━┓
        n_gran_train = len(y_tr)
        w_tr = np.full(n_gran_train, 1.0 / max(n_gran_train, 1), dtype=np.float64)
        
        # ┏━━━━━━━━━━ Append to lists ━━━━━━━━━━┓
        all_X_train.append(X_tr)
        all_y_train.append(y_tr)
        all_w_train.append(w_tr)
        
        # ┏━━━━━━━━━━ Per-gran M1 labels (for Stage-A precision floor) ━━━━━━━━━━┓
        _m1p = sub.get("m1_pred_labels", None)
        _m1t = sub.get("m1_true_labels", None)
        if isinstance(_m1p, torch.Tensor): _m1p = _m1p.numpy()
        if isinstance(_m1t, torch.Tensor): _m1t = _m1t.numpy()

        # ┏━━━━━━━━━━ Store per-gran splits for evaluation ━━━━━━━━━━┓
        per_gran_data[gran] = {"X_full": X_full,
                               "labels": labels,
                               "returns": returns,
                               "asset_ids": asset_ids,
                               "dates": sub["dates"],
                               "asset_map": sub.get("asset_map", {}),
                               "idx_train": idx_train,
                               "idx_val": idx_val,
                               "idx_test": idx_test,
                               "n_train": n_gran_train,
                               "m1_pred_labels": _m1p,
                               "m1_true_labels": _m1t}
        
        print(f"  {gran}: Train={n_gran_train}  Val={len(idx_val)}  Test={len(idx_test)}")
    
    # ┏━━━━━━━━━━ Concatenate all granularities ━━━━━━━━━━┓
    X_train_all = np.vstack(all_X_train)
    y_train_all = np.concatenate(all_y_train)
    w_train_all = np.concatenate(all_w_train)
    
    print(f"\n  Unified train set: {len(y_train_all)} samples "
          f"(class 0: {(y_train_all == 0).sum()}, class 1: {(y_train_all == 1).sum()})")
    
    # ┏━━━━━━━━━━ Feature names: eng features + one-hot gran columns ━━━━━━━━━━┓
    n_eng = X_train_all.shape[1] - len(gran_onehot_names)
    _eng_names = resolve_feature_names(n_eng)
    feature_names = _eng_names + gran_onehot_names
    
    # ┏━━━━━━━━━━ Drop zero-variance / all-NaN features, impute, scale ━━━━━━━━━━┓
    scaler = StandardScaler()
    if model_name not in MODELS_NO_SCALING:
        X_train_all[:, :n_eng] = scaler.fit_transform(X_train_all[:, :n_eng])
    else:
        scaler.fit(X_train_all[:, :n_eng])  # kept for cache compatibility; data stays raw
    
    # ┏━━━━━━━━━━ Train unified model ━━━━━━━━━━┓
    n_pos = int((y_train_all == 1).sum())
    n_neg = int((y_train_all == 0).sum())
    cw_ratio = n_neg / max(n_pos, 1)  # Class weight ratio
    
    print(f"\n  Training unified {mlabel} ({len(feature_names)} features)...")
    model = _build_tree_model(model_name,
                              len(y_train_all),
                              cw_ratio,
                              feature_names=feature_names,
                              presets=_AG_PRESETS)
    model.fit(X_train_all, y_train_all, sample_weight=w_train_all)
    print(f"  Training complete.\n")
    
    # ┏━━━━━━━━━━ Save AutoGluon model info ━━━━━━━━━━┓
    if model_name == "autogluon":
        model.leaderboard()
        model.model_info(save_dir)
    
    # ┏━━━━━━━━━━ Summary of per-granularity evaluation ━━━━━━━━━━┓
    summary = {"cache": str(cache_path),
               "m1_model": cfg.get("data", {}).get("load", {}).get("m1", "kronos") if cfg else "kronos",
               "direction": direction,
               "meta_label_mode": mode,
               "model": model_name,
               "mode": "unified",
               "granularities": gran_list,
               "n_train_total": len(y_train_all),
               "feature_names": feature_names,
               "per_gran": {}}
    
    # ┏━━━━━━━━━━ Loop through granularities ━━━━━━━━━━┓
    for gran in gran_list:
        # ┏━━━━━━━━━━ Get data for this granularity ━━━━━━━━━━┓
        gd = per_gran_data[gran]
        gran_dir = save_dir / gran
        gran_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"  {'─' * 50}")
        print(f"  Evaluating {gran}...")
        
        # ┏━━━━━━━━━━ Prepare validation data ━━━━━━━━━━┓
        X_val_raw = gd["X_full"][gd["idx_val"]].copy()
        y_val = gd["labels"][gd["idx_val"]].astype(int)
        val_returns = gd["returns"][gd["idx_val"]].copy()
        val_dates_raw = [gd["dates"][i] for i in gd["idx_val"]]
        
        # ┏━━━━━━━━━━ Prepare test data ━━━━━━━━━━┓
        X_test_raw = gd["X_full"][gd["idx_test"]].copy()
        y_test = gd["labels"][gd["idx_test"]].astype(int)
        test_returns = gd["returns"][gd["idx_test"]].copy()
        test_asset_ids = gd["asset_ids"][gd["idx_test"]]
        test_dates_raw = [gd["dates"][i] for i in gd["idx_test"]]
        test_assets = [gd["asset_map"].get(int(aid), str(aid)) for aid in test_asset_ids]
        
        # ┏━━━━━━━━━━ Direction-aware returns ━━━━━━━━━━┓
        if direction.lower() == "down":
            val_returns = -val_returns
            test_returns = -test_returns
        
        # ┏━━━━━━━━━━ Impute + scale (eng features only, one-hot stays as-is) ━━━━━━━━━━┓
        if model_name not in MODELS_NO_SCALING:
            for X in [X_val_raw, X_test_raw]:
                X[:, :n_eng] = scaler.transform(X[:, :n_eng])
        
        # ┏━━━━━━━━━━ Predict ━━━━━━━━━━┓
        val_preds = model.predict(X_val_raw)
        val_probs = model.predict_proba(X_val_raw)[:, 1]
        test_preds = model.predict(X_test_raw)
        test_probs = model.predict_proba(X_test_raw)[:, 1]
        
        # ┏━━━━━━━━━━ Calibrate probabilities (isotonic on val → apply to test only) ━━━━━━━━━━┓
        if thres_mode == "utility_nocal":
            from Utils.selective_classification.calibration import _IdentityCalibrator
            _calibrator_u = _IdentityCalibrator()
            test_probs_cal = test_probs
            print(f"    [utility_nocal] Calibration skipped — raw test probs used directly.")
        else:
            _calib_u = calibrate_probabilities(val_probs, y_val, test_probs)
            test_probs_cal = _calib_u["test_calibrated"]
            _calibrator_u = _calib_u["calibrator"]
            print(f"    Calibration (isotonic): test [{test_probs.min():.3f},{test_probs.max():.3f}] → "
                  f"[{test_probs_cal.min():.3f},{test_probs_cal.max():.3f}]")
        
        # ┏━━━━━━━━━━ Pre-selective confusion matrices + metrics (Val & Test) ━━━━━━━━━━┓
        presel_metrics = {}
        for split_name, y_split, preds, probs in [("Val", y_val, val_preds, val_probs),
                                                  ("Test", y_test, test_preds, test_probs)]:
            # ┏━━━━━━━━━━ Save confusion matrix ━━━━━━━━━━┓
            cm_path = gran_dir / f"{split_name}_CM.png"
            plot_confusion_matrix(y_split,
                                  preds,
                                  classes=class_names,
                                  save_path=str(cm_path),
                                  title=f"Unified {mlabel} {gran} {split_name}",
                                  meta_mode=mode)
            
            # ┏━━━━━━━━━━ Calculate metrics ━━━━━━━━━━┓
            prec_val = float(precision_score(y_split, preds, zero_division=0))
            n_pred_pos = int((preds == 1).sum())
            presel_metrics[split_name] = {"accuracy": round(float(accuracy_score(y_split, preds)), 4),
                                          "precision": round(prec_val, 4),
                                          "recall": round(float(recall_score(y_split, preds, zero_division=0)), 4),
                                          "f1_score": round(float(f1_score(y_split, preds, zero_division=0)), 4),
                                          "coverage": round(n_pred_pos / len(y_split), 4) if len(y_split) > 0 else 0,
                                          "risk": round(1 - prec_val, 4),
                                          "baseline": round(int((y_split == 1).sum()) / len(y_split), 4) if len(
                                              y_split) > 0 else 0}
        
        # ┏━━━━━━━━━━ M1 Precision Floor on Val (per-gran) ━━━━━━━━━━┓
        _m1p_g = gd.get("m1_pred_labels")
        _m1t_g = gd.get("m1_true_labels")
        _m1_prec_val = None
        if _m1p_g is not None and _m1t_g is not None:
            try:
                p_v = _m1p_g[gd["idx_val"]]
                t_v = _m1t_g[gd["idx_val"]]
                _valid_v = ~np.isnan(p_v) & ~np.isnan(t_v)
                if _valid_v.any():
                    _m1_prec_val = float(precision_score(t_v[_valid_v].astype(int),
                                                         p_v[_valid_v].astype(int),
                                                         zero_division=0))
            except Exception:
                _m1_prec_val = None

        # ┏━━━━━━━━━━ Threshold optimization on val (standard utility) ━━━━━━━━━━┓
        val_op = _find_best_utility_threshold(val_probs, val_returns, fee=fee, labels=y_val,
                                              m1_precision=_m1_prec_val)
        sel_val = val_probs >= val_op["threshold"]
        n_sel_val = int(sel_val.sum())
        err_val = int((y_val[sel_val] == 0).sum()) if n_sel_val > 0 else 0
        val_op["risk"] = err_val / max(n_sel_val, 1)
        threshold = val_op["threshold"]
        
        # ┏━━━━━━━━━━ OCP: run SAOCP warm-up on val, adapt on test (delayed feedback) ━━━━━━━━━━┓
        _is_ocp_u = thres_mode.startswith("OCP")
        if _is_ocp_u:
            # ┏━━━━━━━━━━ Determine calibration window and Costs ━━━━━━━━━━┓
            _cw = calib_window_for_gran(gran, ocp_window_days)
            c_FN, c_FP, c_DEF = ocp_costs
            
            # ┏━━━━━━━━━━ Run OCP ━━━━━━━━━━┓
            if thres_mode in ("OCP-cost", "OCP-cost-mondrian"):
                test_s_hats, test_approved_ocp, val_s_hats, conf_stats = _run_cost_deferral_online(val_probs,
                                                                                                   y_val,
                                                                                                   test_probs,
                                                                                                   y_test,
                                                                                                   alpha=ocp_alpha,
                                                                                                   test_dates=test_dates_raw,
                                                                                                   forecast_horizon=horizon,
                                                                                                   val_dates=val_dates_raw,
                                                                                                   calib_window=_cw,
                                                                                                   c_FP=c_FP,
                                                                                                   c_FN=c_FN,
                                                                                                   c_DEF=c_DEF,
                                                                                                   mondrian=(
                                                                                                           thres_mode == "OCP-cost-mondrian"),
                                                                                                   test_returns=test_returns)
            # ┏━━━━━━━━━━ Run SAOCP with adapting calibration window of residuals ━━━━━━━━━━┓
            elif thres_mode == "OCP-W":
                test_s_hats, test_approved_ocp, val_s_hats, conf_stats = _run_saocp_online(val_probs,
                                                                                           y_val,
                                                                                           test_probs,
                                                                                           y_test,
                                                                                           alpha=ocp_alpha,
                                                                                           test_dates=test_dates_raw,
                                                                                           forecast_horizon=horizon,
                                                                                           val_dates=val_dates_raw,
                                                                                           calib_window=_cw)
            # ┏━━━━━━━━━━ Run SAOCP with fixed calibration window ━━━━━━━━━━┓
            else:
                test_s_hats, test_approved_ocp, val_s_hats, conf_stats = _run_saocp_online(val_probs,
                                                                                           y_val,
                                                                                           test_probs,
                                                                                           y_test,
                                                                                           alpha=ocp_alpha,
                                                                                           test_dates=test_dates_raw,
                                                                                           forecast_horizon=horizon,
                                                                                           val_dates=val_dates_raw)
            
            # ┏━━━━━━━━━━ Convert OCP outputs to threshold-based operation ━━━━━━━━━━┓
            ocp_op = _ocp_threshold_to_op(test_probs,
                                          y_test,
                                          test_returns,
                                          test_approved_ocp,
                                          test_s_hats,
                                          fee,
                                          conformal_stats=conf_stats)
            ocp_op["threshold_source"] = thres_mode
            cc = conf_stats["conformal_coverage"]
            mode_tag = thres_mode
            
            # ┏━━━━━━━━━━ Print OCP results ━━━━━━━━━━┓
            print(f"    {mode_tag} ({gran}): α={ocp_alpha}, median τ={ocp_op['threshold']:.3f}, "
                  f"cov={ocp_op['coverage']:.1%}, μ={ocp_op['mean_ret'] * 100:+.3f}% | "
                  f"Conformal cov={cc:.1%} (target≥{1 - ocp_alpha:.0%}) | "
                  f"Sets: {{1}}={conf_stats['n_set_1']} {{0}}={conf_stats['n_set_0']} "
                  f"{{0,1}}={conf_stats['n_set_both']} {{}}={conf_stats['n_set_empty']}")
            if "tau_trajectory" in conf_stats:
                tau_std = float(np.std(conf_stats["tau_trajectory"]))
                print(f"    Cost params: c_FN={c_FN}, c_FP={c_FP}, c_DEF={c_DEF} | τ* std={tau_std:.4f}")
        
        # ┏━━━━━━━━━━ Post-selective confusion matrices (Val & Test) ━━━━━━━━━━┓
        for split_name, y_split, probs_cal in [("Val", y_val, val_probs), ("Test", y_test, test_probs_cal)]:
            # ┏━━━━━━━━━━ Determine selection method ━━━━━━━━━━┓
            if split_name == "Test" and _is_ocp_u:
                sel = test_approved_ocp
                thr_source = ocp_op.get("threshold_source", "OCP-SAOCP")
            else:
                sel = probs_cal >= threshold
                thr_source = val_op.get("threshold_source", "Utility-Opt") if split_name == "Val" else val_op.get(
                    "threshold_source", "Val-Utility")
            
            # ┏━━━━━━━━━━ Save confusion matrix ━━━━━━━━━━┓
            sel_true = y_split
            sel_preds = sel.astype(int)
            sel_cm_path = gran_dir / f"{split_name}_Selective_CM.png"
            thr_display = ocp_op["threshold"] if (split_name == "Test" and _is_ocp_u) else threshold
            
            # ┏━━━━━━━━━━ Determine title ━━━━━━━━━━┓
            if _is_ocp_u and split_name == "Test":
                cc = ocp_op.get("conformal_coverage", 0)
                sel_title = (f"Unified {mlabel} {gran} {split_name} selective @thr={thr_display:.3f} ({thr_source})\n"
                             f"Conformal Cov={cc:.1%} (target≥{1 - ocp_alpha:.0%})")
            else:
                sel_title = f"Unified {mlabel} {gran} {split_name} selective @thr={thr_display:.3f} ({thr_source})"
            
            # ┏━━━━━━━━━━ Plot confusion matrix ━━━━━━━━━━┓
            plot_confusion_matrix(sel_true,
                                  sel_preds,
                                  classes=class_names,
                                  save_path=str(sel_cm_path),
                                  title=sel_title,
                                  meta_mode=mode,
                                  is_selective=True)
        
        # ┏━━━━━━━━━━ Apply threshold on test ━━━━━━━━━━┓
        if _is_ocp_u:
            m2_approved = test_approved_ocp
        else:
            m2_approved = test_probs_cal >= threshold
        net_returns = test_returns - fee
        
        # ┏━━━━━━━━━━ Trade DataFrame ━━━━━━━━━━┓
        df_trades = pd.DataFrame({"date": pd.to_datetime(test_dates_raw),
                                  "asset": test_assets,
                                  "return": net_returns,
                                  "label": y_test,
                                  "m2_approved": m2_approved,
                                  "m2_prob_raw": test_probs,
                                  "m2_prob": test_probs_cal})
        
        df_trades = df_trades.dropna(subset=["return"]).reset_index(drop=True)
        m2_approved = df_trades["m2_approved"].values
        test_probs = df_trades["m2_prob"].values
        test_returns = df_trades["return"].values + fee
        y_test = df_trades["label"].values
        
        # ┏━━━━━━━━━━ Save trades CSV for diagnostics ━━━━━━━━━━┓
        trades_dump_u = df_trades.copy()
        trades_dump_u["direction"] = direction
        trades_dump_u["return_pct"] = trades_dump_u["return"] * 100
        trades_dump_u.to_csv(gran_dir / "backtest_trades.csv",
                             index=False,
                             float_format="%.6f")
        
        # ┏━━━━━━━━━━ Save OCP diagnostics npz (thresholds + val probs for re-run) ━━━━━━━━━━┓
        if _is_ocp_u:
            np.savez_compressed(gran_dir / "ocp_diagnostics.npz",
                                test_s_hats=test_s_hats,
                                val_s_hats=val_s_hats,
                                val_probs=val_probs,
                                val_labels=y_val.astype(int),
                                alpha=np.array([ocp_alpha]))
            
            # ┏━━━━━━━━━━ Mondrian / cost-deferral regime diagnostics ━━━━━━━━━━┓
            if "mondrian_diag" in conf_stats:
                plot_mondrian_diagnostics(conf_stats,
                                          gran_dir,
                                          gran_label=gran,
                                          thres_mode=thres_mode)
        
        m2_df = df_trades[df_trades["m2_approved"]]
        test_start = df_trades["date"].min()
        test_end = df_trades["date"].max()
        
        # ┏━━━━━━━━━━ OCP threshold evolution plot ━━━━━━━━━━┓
        if _is_ocp_u:
            fig_thr, ax_thr = plt.subplots(figsize=(10, 4), facecolor="white")
            ax_thr.set_facecolor("#FAFAFA")
            eff_tau = np.maximum(test_s_hats, 1.0 - test_s_hats)
            ax_thr.plot(eff_tau, color="#8B008B", linewidth=0.8, alpha=0.9, label=f"τ_t ({thres_mode})")
            ax_thr.axhline(y=threshold, color="#34495E", linestyle="--", linewidth=1.2, alpha=0.7,
                           label=f"τ Utility = {threshold:.3f}")
            ax_thr.axhline(y=0.5, color="#BDC3C7", linestyle=":", linewidth=0.8, alpha=0.6)
            ax_thr.set_xlabel("Test sample index", fontsize=10)
            ax_thr.set_ylabel("Threshold τ_t", fontsize=10)
            cc = ocp_op.get("conformal_coverage", 0)
            n1 = ocp_op.get("n_set_1", 0)
            n0 = ocp_op.get("n_set_0", 0)
            nb = ocp_op.get("n_set_both", 0)
            ne = ocp_op.get("n_set_empty", 0)
            ax_thr.set_title(f"{thres_mode} Threshold Evolution  |  Unified {gran}  |  {mlabel}  (α={ocp_alpha})\n"
                             f"Conformal Cov={cc:.1%} (target≥{1 - ocp_alpha:.0%})  |  "
                             f"{{1}}={n1}  {{0}}={n0}  {{0,1}}={nb}  {{}}={ne}",
                             fontsize=11,
                             fontweight="bold",
                             color="#2C3E50")
            ax_thr.legend(fontsize=8, loc="upper right")
            ax_thr.set_ylim(0.4, 1.0)
            ax_thr.grid(True, alpha=0.3)
            fig_thr.tight_layout()
            thr_evo_path = gran_dir / "OCP_Threshold_Evolution.png"
            fig_thr.savefig(str(thr_evo_path), dpi=200, facecolor="white")
            plt.close(fig_thr)
        
        # ┏━━━━━━━━━━ Statistics of SAOCP ━━━━━━━━━━┓
        n_total = len(y_test)
        n_approved = int(m2_approved.sum())
        n_m2_good = int(((m2_approved) & (y_test == 1)).sum())
        m2_wr = n_m2_good / n_approved * 100 if n_approved > 0 else 0
        m1_good = int((y_test == 1).sum())
        m1_wr = m1_good / n_total * 100 if n_total > 0 else 0
        execution_rate = n_approved / n_total * 100 if n_total > 0 else 0
        
        # ┏━━━━━━━━━━ Test selective metrics (use OCP mask when applicable) ━━━━━━━━━━┓
        if _is_ocp_u:
            sel_test = m2_approved
        else:
            sel_test = test_probs >= threshold
        n_sel_test = int(sel_test.sum())
        err_test = int((y_test[sel_test] == 0).sum()) if n_sel_test > 0 else 0
        net_rets_test = test_returns[sel_test] - fee if n_sel_test > 0 else np.array([0.0])
        mu_test = float(np.nanmean(net_rets_test))
        sigma_test = float(np.nanstd(net_rets_test, ddof=1)) if n_sel_test > 1 else 0.0
        t_test = mu_test / sigma_test * np.sqrt(n_sel_test) if sigma_test > 0 else 0.0
        
        # ┏━━━━━━━━━━ Equity curves ━━━━━━━━━━┓
        raw_close = _load_raw_close_prices(cfg, gran, direction=direction)
        has_bh = len(raw_close) > 0
        if has_bh:
            raw_close = raw_close[(raw_close["date"] >= test_start) & (raw_close["date"] <= test_end)]
            bh_pivot = raw_close.pivot_table(index="date", columns="asset", values="close")
            bh_first = bh_pivot.iloc[0]
            bh_equity = (bh_pivot / bh_first).mean(axis=1)
            full_idx = bh_equity.index
        else:
            full_idx = pd.DatetimeIndex(sorted(df_trades["date"].unique()))
        
        # ┏━━━━━━━━━━ Equity curves for M1 & M2 ━━━━━━━━━━┓
        m1_equity, _ = _build_spread_equity(df_trades, full_idx, horizon)
        m2_equity, _ = _build_spread_equity(m2_df, full_idx, horizon)
        
        # ┏━━━━━━━━━━ Sharpe: horizon-length non-overlapping returns from equity curve ━━━━━━━━━━┓
        ann_bar = _annualization_factor(gran)
        ann_horizon = np.sqrt(ann_bar ** 2 / horizon)
        
        m2_name = f"M2 {mlabel} unified"
        m1_name = f"{_m1_display_label(cfg)} (all trades)"
        bh_name = "Buy & Hold"
        
        # ┏━━━━━━━━━━ Results from Backtest: Sharpe Ratio, Max Drawdown, Total Return ━━━━━━━━━━┓
        strats = {}
        for name, eq, tdf in [(m2_name, m2_equity, m2_df), (m1_name, m1_equity, df_trades)]:
            h_rets = _equity_horizon_returns(eq, horizon) if len(eq) > horizon else np.array([])
            strats[name] = {"total_ret": (eq.iloc[-1] - 1) * 100 if len(eq) > 0 else 0,
                            "mdd": _calc_drawdown(eq.values) * 100 if len(eq) > 0 else 0,
                            "sharpe": _calc_sharpe(h_rets, ann_horizon)}
        if has_bh:
            bh_h_rets = _equity_horizon_returns(bh_equity, horizon) if len(bh_equity) > horizon else np.array([])
            strats[bh_name] = {"total_ret": (bh_equity.iloc[-1] - 1) * 100,
                               "mdd": _calc_drawdown(bh_equity.values) * 100,
                               "sharpe": _calc_sharpe(bh_h_rets, ann_horizon)}
        
        # ┏━━━━━━━━━━ Plot equity curve ━━━━━━━━━━┓
        if _is_ocp_u:
            ocp_median_tau = float(np.median(np.maximum(test_s_hats, 1.0 - test_s_hats)))
            constraint_tag = "OCP Adaptive"
            thr_display_eq = ocp_median_tau
        else:
            constraint_tag = "Utility-Opt" if val_op["constraint_satisfied"] else "fallback"
            thr_display_eq = threshold
        fee_tag = f" fee={fee * 100:.2f}%" if fee > 0 else ""
        direction_label = direction.upper()
        
        fig, ax = plt.subplots(figsize=(14, 6))
        ax.plot((m2_equity - 1) * 100,
                label=f"{m2_name} (SR: {strats[m2_name]['sharpe']:.2f}, Exec: {execution_rate:.1f}%)", color="green",
                linewidth=3.0)
        ax.plot((m1_equity - 1) * 100, label=f"{m1_name} (SR: {strats[m1_name]['sharpe']:.2f})", color="blue",
                alpha=0.6, linewidth=2.0)
        if has_bh:
            ax.plot((bh_equity - 1) * 100, label=f"{bh_name} (SR: {strats[bh_name]['sharpe']:.2f})", color="gray",
                    linestyle="--", linewidth=1.5)
        ax.axhline(0, color="black", linewidth=0.5)
        ax.set_title(f"UNIFIED {mlabel} {gran.upper()}+{direction_label}+{mode.upper()} "
                     f"thr={thr_display_eq:.3f} ({constraint_tag}){fee_tag}", fontsize=12)
        ax.set_ylabel("Cumulative Return (%)")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(gran_dir / f"backtest_equity_curve.png", dpi=200)
        plt.close(fig)
        
        # ┏━━━━━━━━━━ ROI report ━━━━━━━━━━┓
        avg_ret_approved = m2_df["return"].mean() * 100 if len(m2_df) > 0 else 0
        avg_ret_rejected = df_trades[~df_trades["m2_approved"]]["return"].mean() * 100 if n_total > n_approved else 0
        edge = avg_ret_approved - avg_ret_rejected
        
        roi_lines = [
            "=" * 60,
            f"FINANCIAL BACKTEST: UNIFIED {mlabel} {gran.upper()} {direction_label} {mode.upper()}",
            f"Period: {test_start.strftime('%Y-%m-%d')} to {test_end.strftime('%Y-%m-%d')}",
            f"Threshold: {thr_display_eq:.4f} ({constraint_tag}) | Fee: {fee * 100:.3f}%",
            "=" * 60,
            f"Total Test Trades:     {n_total}",
            f"{_m1_display_label(cfg)} Baseline Win-Rate:  {m1_wr:.1f}% ({m1_good}/{n_total})",
            "-" * 60,
            f"M2 Approved Trades:    {n_approved} ({execution_rate:.1f}% execution)",
            f"M2 Rejected Trades:    {n_total - n_approved}",
            f"M2 Win-Rate:           {m2_wr:.1f}% ({n_m2_good}/{n_approved})",
            "-" * 60,
            f"Avg Return APPROVED:   {avg_ret_approved:+.3f}%",
            f"Avg Return REJECTED:   {avg_ret_rejected:+.3f}%",
            f"M2 Edge: Approved trades yield {edge:.3f}% more per trade",
            "=" * 60,
            f"{'Strategy':<27} {'Total Ret':>10} {'MaxDD':>8} {'Sharpe':>8}",
            "-" * 60,
        ]
        
        # ┏━━━━━━━━━━ Add strategy results to ROI report ━━━━━━━━━━┓
        for sname in [m2_name, m1_name, bh_name]:
            if sname in strats:
                s = strats[sname]
                roi_lines.append(f"{sname:<27} {s['total_ret']:>+9.2f}% {s['mdd']:>+7.2f}% {s['sharpe']:>7.2f}")
        roi_lines.append("=" * 60)
        
        # ┏━━━━━━━━━━ Save ROI report to file ━━━━━━━━━━┓
        roi_text = "\n".join(roi_lines)
        with open(gran_dir / "backtest_ROI.txt", "w") as f:
            f.write(roi_text)
        print(roi_text)
        
        # ┏━━━━━━━━━━ Store per-gran summary ━━━━━━━━━━┓
        summary["per_gran"][gran] = {
            "n_train": gd["n_train"],
            "n_val": len(gd["idx_val"]),
            "n_test": len(gd["idx_test"]),
            "threshold": threshold,
            "constraint_satisfied": val_op["constraint_satisfied"],
            "Val": presel_metrics.get("Val", {}),
            "Test": presel_metrics.get("Test", {}),
            "val_selective": {"coverage": val_op["coverage"],
                              "risk": val_op["risk"],
                              "precision": 1 - val_op["risk"],
                              "mean_ret": val_op["mean_ret"],
                              "t_stat": val_op["t_stat"],
                              "selected_count": val_op["selected_count"]},
            "test_selective": {"coverage": n_sel_test / n_total if n_total > 0 else 0,
                               "risk": err_test / max(n_sel_test, 1),
                               "precision": 1 - err_test / max(n_sel_test, 1),
                               "mean_ret": mu_test,
                               "t_stat": t_test,
                               "selected_count": n_sel_test,
                               **({"ocp": {
                                   "alpha": ocp_alpha,
                                   "conformal_coverage": round(ocp_op.get("conformal_coverage", 0), 4),
                                   "target_coverage": round(1 - ocp_alpha, 4),
                                   "guarantee_met": ocp_op.get("conformal_coverage", 0) >= (1 - ocp_alpha),
                                   "n_set_1_trade": ocp_op.get("n_set_1", 0),
                                   "n_set_0_dont_trade": ocp_op.get("n_set_0", 0),
                                   "n_set_both_abstain": ocp_op.get("n_set_both", 0),
                                   "n_set_empty_abstain": ocp_op.get("n_set_empty", 0),
                                   **({"regime_stats": conf_stats["regime_stats"]} if conf_stats.get(
                                       "regime_stats") else {})}} if _is_ocp_u else {})},
            "backtest": {"execution_rate": execution_rate,
                         "n_total_trades": n_total,
                         "n_m2_trades": n_approved,
                         "m1_win_rate": m1_wr,
                         "m2_win_rate": m2_wr,
                         "m2_total_return": strats[m2_name]["total_ret"],
                         "m1_total_return": strats[m1_name]["total_ret"],
                         "bh_total_return": strats.get(bh_name, {}).get("total_ret", None),
                         "m2_sharpe": strats[m2_name]["sharpe"],
                         "m1_sharpe": strats[m1_name]["sharpe"],
                         "bh_sharpe": strats.get(bh_name, {}).get("sharpe", None),
                         "m2_max_drawdown": strats[m2_name]["mdd"],
                         "m1_max_drawdown": strats[m1_name]["mdd"],
                         "bh_max_drawdown": strats.get(bh_name, {}).get("mdd", None),
                         "fee": fee,
                         },
        }
    
    # ┏━━━━━━━━━━ Save unified summary ━━━━━━━━━━┓
    with open(save_dir / "unified_summary.json", "w") as f:
        json.dump(summary, f, indent=2, cls=NumpyJSONEncoder)
    
    # ┏━━━━━━━━━━ Save model artifacts ━━━━━━━━━━┓
    artifacts = {"model": model,
                 "scaler": scaler,
                 "feature_names": feature_names,
                 "gran_list": gran_list,
                 "thresholds": {g: summary["per_gran"][g]["threshold"] for g in gran_list},
                 "direction": direction,
                 "mode": mode,
                 "train_end": train_end,
                 "val_end": val_end}
    
    # ┏━━━━━━━━━━ Save unified model ━━━━━━━━━━┓
    with open(save_dir / "unified_model.pkl", "wb") as f:
        pickle.dump(artifacts, f)
    print(f"\n  Model saved to {save_dir / 'unified_model.pkl'}")
    print(f"\nDone (unified {direction}). Outputs in: {save_dir}")


def main():
    # ┏━━━━━━━━━━ Parse Arguments ━━━━━━━━━━┓
    parser = argparse.ArgumentParser(
        description="Kronos Tree — M2 Meta-Labeling with RF/XGBoost/AutoGluon")  # TODO change description
    parser.add_argument("--cache_path", type=str, default=None, help="Explicit path to dataset cache .pt")
    parser.add_argument("--config", type=_load_config, help="Experiment config", required=True)
    parser.add_argument("--phase", type=str, help="Experimental Phase", required=True)
    parser.add_argument("--m2", type=str, help="M2 model to use", required=True)
    parser.add_argument("--direction", type=str, help="Direction to use", required=True)
    parser.add_argument("--granularity", type=str, help="Granularity to use", required=True)

    args = parser.parse_args()
    
    # ┏━━━━━━━━━━ Parse cost vector ━━━━━━━━━━┓
    _cost_parts = args.config["runtime"][args.phase]["ocp_costs"]
    ocp_costs = (_cost_parts[0], _cost_parts[1], _cost_parts[2])  # (c_FN, c_FP, c_DEF)

    # ┏━━━━━━━━━━ Phase 1: Training ━━━━━━━━━━┓
    if args.phase == "training":
        # ┏━━━━━━━━━━ Analysis Comparison between Models ━━━━━━━━━━┓ # NOTE this is not actively used?!
        if args.config["runtime"][args.phase]["paradigm_comparison"]:
            run_paradigm_comparison(args.config["runtime"][args.phase]["paradigm_comparison"])
            print(f"\nParadigm comparison complete.")
            return

        # ┏━━━━━━━━━━ Analysis Comparison between all-grans vs per-gran ━━━━━━━━━━┓ # NOTE this is not actively used?!
        if args.config["runtime"][args.phase]["comparison"]:
            per_gran_dir = Path(args.config["runtime"][args.phase]["comparison"][0])
            unified_dir = Path(args.config["runtime"][args.phase]["comparison"][1])
            run_comparison(per_gran_dir, unified_dir)
            print(f"\nComparison complete.")
            return

        # ┏━━━━━━━━━━ Unified model ━━━━━━━━━━┓ # NOTE this is phase 1
        # One model trained on all granularities, evaluated per-gran
        if args.config["runtime"][args.phase]["all_grans"]:
            # ┏━━━━━━━━━━ Resolve caches (builds both directions if missing) ━━━━━━━━━━┓
            direction_caches = _resolve_caches(args.config, args.cache_path)
            if args.direction not in direction_caches:
                raise RuntimeError(f"No cache resolved for direction='{args.direction}'. "
                                   f"Available: {sorted(direction_caches.keys())}")
            cache_path = direction_caches[args.direction]

            # ┏━━━━━━━━━━ Load Cache ━━━━━━━━━━┓
            print(f"[kronos_tree] Loading multi-gran cache for UNIFIED model: {cache_path.name}")
            multi = _load_multi_cache(cache_path)

            # ┏━━━━━━━━━━ Run Analysis for each granularity ━━━━━━━━━━┓
            run_unified_analysis(cache_path,
                                 multi,
                                 args.direction,
                                 args.config['data']['load']['meta_label_mode'],
                                 args.config['paths']['output_root'],
                                 train_end=args.config['data']['split']['train_end'],
                                 val_end=args.config['data']['split']['val_end'],
                                 model_name=args.m2,
                                 cfg=args.config,
                                 thres_mode=args.config['runtime'][args.phase]["thres"],
                                 ocp_alpha=args.config['runtime'][args.phase]["ocp_alpha"],
                                 ocp_costs=ocp_costs,
                                 ocp_window_days=args.config['runtime'][args.phase]["ocp_window_days"])
            return
        # ┏━━━━━━━━━━ Single granularity — run ONLY the CLI-specified direction ━━━━━━━━━━┓
        else:
            # Build/discover both direction caches once (so the other is ready for future runs),
            # but only run analysis for args.direction.
            direction_caches = _resolve_caches(args.config, args.cache_path)
            if args.direction not in direction_caches:
                raise RuntimeError(f"No cache resolved for direction='{args.direction}'. "
                                   f"Available: {sorted(direction_caches.keys())}")

            cache_path = direction_caches[args.direction]
            best_params = _load_best_params(args.config, args.m2, args.direction, args.granularity)
            run_analysis(cache_path=cache_path,
                         direction=args.direction,
                         mode=args.config['data']['load']['meta_label_mode'],
                         granularity=args.granularity,
                         output_root=args.config['paths']['output_root'],
                         train_end=args.config['data']['split']['train_end'],
                         val_end=args.config['data']['split']['val_end'],
                         model_name=args.m2,
                         cfg=args.config,
                         run_top5=args.config['runtime'][args.phase]["top5"],
                         run_features=args.config['runtime'][args.phase]["run_features"],
                         thres_mode=args.config['runtime'][args.phase]["thres"],
                         ocp_alpha=args.config['runtime'][args.phase]["ocp_alpha"],
                         ocp_costs=ocp_costs,
                         ocp_window_days=args.config['runtime'][args.phase]["ocp_window_days"],
                         best_params=best_params)
            print(f"\nAnalysis complete ({args.m2}/{args.direction}/{args.granularity}).")
            return

    # ┏━━━━━━━━━━ Phase 3: Combined ━━━━━━━━━━┓
    if args.phase == "combined":
        # ┏━━━━━━━━━━ Combined UP+DOWN Backtest ━━━━━━━━━━┓
        combined_pair = args.config["runtime"]["combined"]["combined_backtest"]
        if combined_pair:
            # ┏━━━━━━━━━━ Save into the model directory (parent of the UP/DN folders) under Backtest_UP_DN ━━━━━━━━━━┓
            up_path = Path(combined_pair[0])
            dn_path = Path(combined_pair[1])
            save_combined = up_path.parent.parent / "Backtest_UP_DN"

            # ┏━━━━━━━━━━ Run combined backtest ━━━━━━━━━━┓
            run_combined_backtest(up_path, dn_path, save_combined, args.config, granularity=args.granularity,
                                  model_name=args.m2)
            print(f"\nCombined UP+DOWN backtest complete. Outputs in: {save_combined}")
            return


if __name__ == "__main__":
    main()
