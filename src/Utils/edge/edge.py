"""
Edge Analysis — Model Stability & Regime Sensitivity
=====================================================
Two modes:
  --mode seeds  : 100 trials with different seeds on a static train/val/test split
  --mode cpcv   : Combinatorial Purged Cross-Validation (Lopez de Prado)
                  N=6 datetime blocks, k=2 test → C(6,2)=15 splits → 5 paths

Supported models: rf (randforest), xgboost, autogluon, tabpfn, tabpfn_ft

Usage:
  python -m Utils.edge --cache path/to/multi_cache.pt --mode seeds
  python -m Utils.edge --cache path/to/multi_cache.pt --mode seeds --model xgboost
  python -m Utils.edge --cache path/to/multi_cache.pt --mode cpcv
  python -m Utils.edge --cache path/to/multi_cache.pt --mode cpcv --model tabpfn_ft
  python -m Utils.edge --cache path/to/multi_cache.pt --mode cpcv --n-blocks 8 --k-test 2
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
                         m1_display_label)

# ┏━━━━━━━━━━ Imports from TS Cross-Validation ━━━━━━━━━━┓
from Utils.ts_cross_validation import (_gran_to_timedelta,
                                       _build_datetime_blocks,
                                       _assign_blocks,
                                       _generate_cpcv_splits,
                                       _reconstruct_paths,
                                       compute_embargo_splits as _compute_embargo_splits,
                                       compute_seeds_embargo_splits as _compute_seeds_embargo_splits,
                                       CAL_SPLIT_RATIO)

from Utils.edge.plots import (_plot_split_matrix,
                              _plot_path_boxplots,
                              _plot_cross_gran_cpcv)
from .plots import (
    _plot_edge_curves,
    _plot_summary_boxplots,
    _plot_cross_gran_seeds,
    _plot_split_matrix,
    _plot_path_boxplots,
    _plot_cross_gran_cpcv,
)


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

# ┏━━━━━━━━━━ CLI model name → models.py model key ━━━━━━━━━━┓
# _CLI_TO_MODEL_KEY = {"randforest": "rf",
#                      "xgboost":    "xgboost",
#                      "autogluon":  "autogluon",
#                      "tabpfn":     "tabpfn",
#                      "tabpfn_ft":  "tabpfn_ft",
#                      "tabicl":     "tabicl"}


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

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Shared: M1 baselines and model builder
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# ┏━━━━━━━━━━ Build model for edge analysis ━━━━━━━━━━┓
def _build_edge_model(model_key: str, 
                      n_samples: int, 
                      seed: int = EDGE_SEED,
                      class_weight_ratio: float = 1.0):
    """Build a model via _build_tree_model, then override random_state for seed control."""
    model = _build_tree_model(model_key, n_samples, class_weight_ratio=class_weight_ratio)
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
    to match kronos_tree behavior. M1 mean return uses the valid-filtered
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




# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ██████  MODE: SEEDS  ██████
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# ┏━━━━━━━━━━ Run single trial [How noisy is the model under different seeds?] ━━━━━━━━━━┓
def _run_single_trial(eng, labels, returns, split_indices, direction, fee, seed, granularity, horizon,
                      model_key="rf"):
    """Train one model with a given seed, return metrics.

    4-way split: Train / Val-Cal / Val-Opt / Val-Eval (all carved inside
    train+val — the real TEST window is never referenced here; embargo gaps
    are handled upstream by ``compute_seeds_embargo_splits``).
    Order: Train → Calibrate on Cal → Sweep τ on calibrated Opt → Evaluate on calibrated Val-Eval.

    The result key remains ``"test"`` (in the dict) to keep downstream plot
    code unchanged; it means "evaluation set of this trial" = Val-Eval, NOT
    the real test window.
    """
    # ┏━━━━━━━━━━ Get 4-way split indices (Val-Eval is the per-trial hold-out) ━━━━━━━━━━┓
    idx_train, idx_cal, idx_opt, idx_val_eval = split_indices

    # ┏━━━━━━━━━━ Extract data for each split ━━━━━━━━━━┓
    X_train = eng[idx_train];     y_train = labels[idx_train].astype(int)
    X_cal   = eng[idx_cal];       y_cal   = labels[idx_cal].astype(int)
    X_opt   = eng[idx_opt];       y_opt   = labels[idx_opt].astype(int)
    X_test  = eng[idx_val_eval];  y_test  = labels[idx_val_eval].astype(int)   # Val-Eval acts as "test" within this trial
    opt_returns  = returns[idx_opt].copy()
    test_returns = returns[idx_val_eval].copy()

    # ┏━━━━━━━━━━ Adjust returns based on direction ━━━━━━━━━━┓
    if direction.lower() == "down":
        opt_returns  = -opt_returns
        test_returns = -test_returns

    ann_horizon = np.sqrt(_annualization_factor(granularity) ** 2 / max(horizon, 1))

    # ┏━━━━━━━━━━ Scale features (skip for TabPFN) ━━━━━━━━━━┓
    if model_key not in MODELS_NO_SCALING:  # TODO this is very questionable to only scale for some models?! Why is that?
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_cal   = scaler.transform(X_cal)
        X_opt   = scaler.transform(X_opt)
        X_test  = scaler.transform(X_test)

    # ┏━━━━━━━━━━ Step 1: Build and train model ━━━━━━━━━━┓
    n_pos = int(y_train.sum())
    n_neg = len(y_train) - n_pos
    cw_ratio = n_neg / max(n_pos, 1)
    model = _build_edge_model(model_key, len(y_train), seed=seed, class_weight_ratio=cw_ratio)
    model.fit(X_train, y_train)

    # ┏━━━━━━━━━━ Step 2: Calibrate on Val-Cal ━━━━━━━━━━┓
    cal_probs_raw = model.predict_proba(X_cal)[:, 1]
    _cal = calibrate_probabilities(cal_probs_raw, y_cal)
    calibrator = _cal["calibrator"]

    # ┏━━━━━━━━━━ Step 3: Sweep threshold on calibrated Val-Opt ━━━━━━━━━━┓
    opt_probs_raw = model.predict_proba(X_opt)[:, 1]
    opt_probs_cal = calibrator.predict(opt_probs_raw)
    op = _find_best_utility_threshold(opt_probs_cal, opt_returns, fee=fee, labels=y_opt)
    val_thr = op["threshold"]

    # ┏━━━━━━━━━━ Step 4: Evaluate on calibrated Test ━━━━━━━━━━┓
    test_preds = model.predict(X_test)
    test_probs_raw = model.predict_proba(X_test)[:, 1]
    test_probs_cal = calibrator.predict(test_probs_raw)

    # ┏━━━━━━━━━━ Base classifier metrics (predict @0.5, no selective threshold) ━━━━━━━━━━┓
    acc  = accuracy_score(y_test, test_preds)
    prec = precision_score(y_test, test_preds, zero_division=0)
    rec  = recall_score(y_test, test_preds, zero_division=0)
    f1   = f1_score(y_test, test_preds, zero_division=0)

    # ┏━━━━━━━━━━ Selective metrics (threshold on calibrated test probs) ━━━━━━━━━━┓
    sel = test_probs_cal >= val_thr
    n_sel = int(sel.sum())
    sel_acc  = accuracy_score(y_test, sel.astype(int))
    sel_prec = float((y_test[sel] == 1).sum()) / max(n_sel, 1) if n_sel > 0 else 0.0
    coverage = n_sel / len(y_test) if len(y_test) > 0 else 0.0
    net_rets = test_returns[sel] - fee if n_sel > 0 else np.array([0.0])
    mean_ret_sel = float(np.nanmean(net_rets))
    sel_sharpe = _calc_sharpe(net_rets, ann_horizon) if n_sel > 1 else 0.0

    # ┏━━━━━━━━━━ Base classifier metrics (predict @0.5, no selective threshold) ━━━━━━━━━━┓
    base_pred_mask = test_preds == 1
    n_base_pred = int(base_pred_mask.sum())
    base_net_rets = test_returns[base_pred_mask] - fee if n_base_pred > 0 else np.array([0.0])
    mean_ret = float(np.nanmean(base_net_rets))

    # ┏━━━━━━━━━━ Store results (test only — no val metrics needed) ━━━━━━━━━━┓
    result = {"test": {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1,
                       "sel_accuracy": sel_acc, "sel_precision": sel_prec,
                       "sel_coverage": coverage, "sel_threshold": val_thr,
                       "mean_ret": mean_ret, "sel_mean_ret": mean_ret_sel,
                       "sel_sharpe": sel_sharpe}}
    return result



# ┏━━━━━━━━━━ Seeds: main runner ━━━━━━━━━━┓
def run_seeds_analysis(cache_path, config, output_root, n_trials=100, m2_name="randforest", direction="up", granularity="1d"):
    # ┏━━━━━━━━━━ Extract config ━━━━━━━━━━┓
    train_end  = config["data"]["split"]["train_end"]
    val_end    = config["data"]["split"]["val_end"]
    fee        = config["evaluation"]["fee_per_trade"]
    horizon    = config["data"]["load"]["forecast_horizon"]
    
    # model_key  = _CLI_TO_MODEL_KEY.get(model_name, model_name)

    # ┏━━━━━━━━━━ Load cache ━━━━━━━━━━┓
    print(f"\n[edge-seeds] Loading cache: {cache_path}")
    multi = _load_multi_cache(cache_path)
    print(f"[edge-seeds] Direction: {direction} | Granularities: {multi.grans}")
    print(f"[edge-seeds] Trials: {n_trials} | Fee: {fee}")
    print(f"[edge-seeds] Output: {output_root}\n")

    # ┏━━━━━━━━━━ Initialize summary ━━━━━━━━━━┓
    summary_all = {}

    print(f"\n{'='*60}")
    print(f"[edge-seeds] {granularity} — {direction.upper()} — {n_trials} trials")
    print(f"{'='*60}")

    # ┏━━━━━━━━━━ Extract data ━━━━━━━━━━┓
    sub = multi.sub[granularity]
    eng = sub["eng_features"]
    if isinstance(eng, torch.Tensor): eng = eng.numpy()
    labels = sub["labels"]
    if isinstance(labels, torch.Tensor): labels = labels.numpy()
    returns = sub["returns"]
    if isinstance(returns, torch.Tensor): returns = returns.numpy()

    # ┏━━━━━━━━━━ Filter valid samples ━━━━━━━━━━┓
    valid = ~np.isnan(labels)
    eng, labels, returns = eng[valid], labels[valid], returns[valid]

    # ┏━━━━━━━━━━ Get dates ━━━━━━━━━━┓
    dates_all = sub["dates"]
    dates_valid = [dates_all[i] for i in range(len(dates_all)) if valid[i]]

    # ┏━━━━━━━━━━ 4-way split with embargo — INSIDE train+val only (no TEST leakage) ━━━━━━━━━━┓
    # The real TEST window (dates > val_end) is never exposed during the
    # seeds-convergence experiment. Per-seed noise is measured on Val-Eval,
    # a held-out slice carved from the tail of the train+val timeline.
    splits = _compute_seeds_embargo_splits(dates_valid, train_end, val_end, horizon, gran)
    idx_train    = splits["idx_train"]
    idx_cal      = splits["idx_cal"]
    idx_opt      = splits["idx_opt"]
    idx_val_eval = splits["idx_val_eval"]
    purge_td     = splits["purge_td"]
    b            = splits["boundaries"]

    # ┏━━━━━━━━━━ Check for empty splits ━━━━━━━━━━┓
    if any(len(s) == 0 for s in [idx_train, idx_cal, idx_opt, idx_val_eval]):
        print(f"  [SKIP] Empty split (train={len(idx_train)} cal={len(idx_cal)} "
                f"opt={len(idx_opt)} val_eval={len(idx_val_eval)})")
        continue

    # ┏━━━━━━━━━━ Print split info ━━━━━━━━━━┓
    print(f"  Splits: train={len(idx_train):,}  cal={len(idx_cal):,}  "
            f"opt={len(idx_opt):,}  val_eval={len(idx_val_eval):,}  (TEST window excluded)")
    print(f"  Boundaries: train→{b['train_end'].strftime('%Y-%m-%d')} | "
            f"cal→{b['cal_end'].strftime('%Y-%m-%d')} | "
            f"opt→{b['opt_end'].strftime('%Y-%m-%d')} | "
            f"val_eval→{b['val_end'].strftime('%Y-%m-%d')}  |  Purge: {purge_td}")
    print(f"  Features: {eng.shape[1]}")

    # ┏━━━━━━━━━━ Compute M1 baselines on Val-Eval (NOT on real test) ━━━━━━━━━━┓
    # Build raw-index list that matches idx_val_eval in the valid-filtered space.
    m1_pred = sub.get("m1_pred_labels")
    m1_true = sub.get("m1_true_labels")
    if isinstance(m1_pred, torch.Tensor): m1_pred = m1_pred.numpy()
    if isinstance(m1_true, torch.Tensor): m1_true = m1_true.numpy()
    # Map valid-filtered Val-Eval indices back to raw cache indices
    valid_to_raw = np.flatnonzero(valid)
    idx_val_eval_raw = valid_to_raw[idx_val_eval].tolist() if len(idx_val_eval) else []

    m1_acc_test = m1_prec_test = None
    if m1_pred is not None and m1_true is not None and len(idx_val_eval_raw) > 0:
        p = m1_pred[idx_val_eval_raw]
        t = m1_true[idx_val_eval_raw]
        ok = ~np.isnan(p) & ~np.isnan(t)
        if ok.sum() > 0:
            m1_acc_test  = float(accuracy_score(t[ok], p[ok]))
            m1_prec_test = float(precision_score(t[ok], p[ok], zero_division=0))

    rets_val_eval = returns[idx_val_eval].copy()
    if direction.lower() == "down":
        rets_val_eval = -rets_val_eval
    m1_ret_test = float(np.nanmean(rets_val_eval - fee)) if len(rets_val_eval) > 0 else 0.0

    # ┏━━━━━━━━━━ Keep the dict shape expected by plot helpers ━━━━━━━━━━┓
    m1_baselines = {"val":  {"m1_acc": None, "m1_prec": None, "m1_mean_ret": None},
                    "test": {"m1_acc":  m1_acc_test,
                                "m1_prec": m1_prec_test,
                                "m1_mean_ret": m1_ret_test}}

    # ┏━━━━━━━━━━ Print M1 baselines (on Val-Eval) ━━━━━━━━━━┓
    if m1_acc_test is not None:
        print(f"  M1 Baseline Val-Eval: acc={m1_acc_test:.4f}  "
                f"prec={m1_prec_test:.4f}  mean_ret={m1_ret_test:.6f}")
    else:
        print(f"  M1 Baselines — not available | Mean Ret Val-Eval: {m1_ret_test:.6f}")

    # ┏━━━━━━━━━━ Run trials (Val-Eval stands in for 'test' within each trial) ━━━━━━━━━━┓
    all_trials = []
    for trial_i in range(n_trials):
        seed = trial_i * 7 + 1
        result = _run_single_trial(eng, labels, returns,
                                    (idx_train, idx_cal, idx_opt, idx_val_eval),
                                    direction, fee, seed, gran, horizon,
                                    model_key=model_key)
        all_trials.append(result)
        if (trial_i + 1) % 10 == 0 or trial_i == 0:
            t = result["test"]
            print(f"    Trial {trial_i+1:3d}/{n_trials}: test_acc={t['accuracy']:.4f}  "
                    f"sel_prec={t['sel_precision']:.4f}  sel_μ={t['sel_mean_ret']:+.5f}  τ={t['sel_threshold']:.3f}")

    # ┏━━━━━━━━━━ Create output directory ━━━━━━━━━━┓
    gran_dir = output_root / direction.upper() / granularity
    gran_dir.mkdir(parents=True, exist_ok=True)

    # ┏━━━━━━━━━━ Plot ━━━━━━━━━━┓
    _plot_edge_curves(all_trials, "test", gran_dir / "edge_test.png", granularity, direction, n_trials, m1_baselines)
    _plot_summary_boxplots(all_trials, gran_dir / "edge_summary.png", granularity, direction, n_trials, m1_baselines)

    # ┏━━━━━━━━━━ Save trial results ━━━━━━━━━━┓
    rows = []
    for i, t in enumerate(all_trials):
        row = {"trial": i, "seed": i * 7 + 1}
        for k, v in t["test"].items():
            row[f"test_{k}"] = v
        rows.append(row)
    pd.DataFrame(rows).to_csv(gran_dir / "edge_trials.csv", index=False)

    # ┏━━━━━━━━━━ Compute summary statistics ━━━━━━━━━━┓
    test_accs      = np.array([t["test"]["accuracy"]      for t in all_trials])
    test_precs     = np.array([t["test"]["precision"]     for t in all_trials])
    test_sel_accs  = np.array([t["test"]["sel_accuracy"]  for t in all_trials])
    test_sel_precs = np.array([t["test"]["sel_precision"] for t in all_trials])
    test_sel_covs  = np.array([t["test"]["sel_coverage"]  for t in all_trials])
    test_sel_rets  = np.array([t["test"]["sel_mean_ret"]  for t in all_trials])
    test_sel_shps  = np.array([t["test"]["sel_sharpe"]    for t in all_trials])
    test_thresholds = np.array([t["test"]["sel_threshold"] for t in all_trials])

    # ┏━━━━━━━━━━ Threshold distribution ━━━━━━━━━━┓
    median_threshold = float(np.median(test_thresholds))
    threshold_std    = float(np.std(test_thresholds))

    # ┏━━━━━━━━━━ Profitability and Sharpe gates ━━━━━━━━━━┓
    frac_profitable = float(np.mean(test_sel_rets > 0))
    mean_sharpe     = float(np.mean(test_sel_shps))
    std_sharpe      = float(np.std(test_sel_shps, ddof=1)) if len(test_sel_shps) > 1 else 0.0
    sharpe_ci_lower = mean_sharpe - 1.96 * std_sharpe / np.sqrt(max(len(test_sel_shps), 1))

    # ┏━━━━━━━━━━ Edge vs M1 ━━━━━━━━━━┓
    acc_edge_test  = float(np.mean(test_accs) - m1_acc_test)   if m1_acc_test  else None
    prec_edge_test = float(np.mean(test_precs) - m1_prec_test) if m1_prec_test else None

    # ┏━━━━━━━━━━ Create summary dictionary ━━━━━━━━━━┓
    summary = {
        "granularity": gran, "direction": direction, "n_trials": n_trials,
        "n_train": len(idx_train), "n_cal": len(idx_cal),
        "n_opt": len(idx_opt), "n_val_eval": len(idx_val_eval),
        "eval_split": "val_eval (carved inside train+val; real TEST excluded)",
        "n_features": eng.shape[1],
        "train_end":    str(b["train_end"]),
        "cal_end":      str(b["cal_end"]),
        "opt_end":      str(b["opt_end"]),
        "val_eval_end": str(b["val_end"]),
        "purge": str(purge_td),
        "m1_baseline_val_eval_acc":  m1_acc_test,
        "m1_baseline_val_eval_prec": m1_prec_test,
        # Legacy keys retained for downstream plot compatibility:
        "m1_baseline_test_acc": m1_acc_test, "m1_baseline_test_prec": m1_prec_test,
        "test_acc_mean": float(np.mean(test_accs)), "test_acc_std": float(np.std(test_accs)),
        "test_prec_mean": float(np.mean(test_precs)), "test_prec_std": float(np.std(test_precs)),
        "test_sel_acc_mean": float(np.mean(test_sel_accs)), "test_sel_acc_std": float(np.std(test_sel_accs)),
        "test_sel_prec_mean": float(np.mean(test_sel_precs)), "test_sel_prec_std": float(np.std(test_sel_precs)),
        "test_sel_cov_mean": float(np.mean(test_sel_covs)), "test_sel_cov_std": float(np.std(test_sel_covs)),
        "test_sel_ret_mean": float(np.mean(test_sel_rets)), "test_sel_ret_std": float(np.std(test_sel_rets)),
        "median_threshold": median_threshold, "threshold_std": threshold_std,
        "frac_profitable": frac_profitable,
        "mean_sharpe": mean_sharpe, "sharpe_ci_lower": sharpe_ci_lower,
        "acc_edge_test": acc_edge_test, "prec_edge_test": prec_edge_test,
    }

    # ┏━━━━━━━━━━ Save summary ━━━━━━━━━━┓
    summary_all[granularity] = summary

    # ┏━━━━━━━━━━ Print summary ━━━━━━━━━━┓
    print(f"\n  ┌─ {granularity} {direction.upper()} Seeds Summary ──────────────┐")
    if m1_acc_test is not None:
        print(f"  │ M1 Baseline Test: acc={m1_acc_test:.4f}  prec={m1_prec_test:.4f}")
    print(f"  │ M2 Test Acc:      {summary['test_acc_mean']:.4f} ± {summary['test_acc_std']:.4f}" + (f"  (edge: {acc_edge_test:+.4f})" if acc_edge_test else ""))
    print(f"  │ M2 Test Prec:     {summary['test_prec_mean']:.4f} ± {summary['test_prec_std']:.4f}" + (f"  (edge: {prec_edge_test:+.4f})" if prec_edge_test else ""))
    print(f"  │ Sel Mean Ret:     {summary['test_sel_ret_mean']:+.5f} ± {summary['test_sel_ret_std']:.5f}")
    print(f"  │ Frac Profitable:  {frac_profitable:.0%}")
    print(f"  │ Sharpe:           {mean_sharpe:.2f} (CI lower: {sharpe_ci_lower:.2f})")
    print(f"  │ Median Threshold: {median_threshold:.3f} ± {threshold_std:.3f}")
    gate_pass = frac_profitable > 0.70 and sharpe_ci_lower > 0
    print(f"  │ Gate: {'PASS ✓' if gate_pass else 'FAIL ✗'}  (>70% profitable AND CI>0)")
    print(f"  └──────────────────────────────────────┘")
    print(f"  Saved to: {gran_dir}")

    # ┏━━━━━━━━━━ Save summary ━━━━━━━━━━┓  
    if summary_all:
        summary_path = output_root / direction.upper() / f"edge_summary_{granularity}.json"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        with open(summary_path, "w") as f:
            json.dump(summary_all, f, indent=2)
        print(f"\n[edge-seeds] Summary: {summary_path}")
    print(f"\n[edge-seeds] Done.")



# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ██████  MODE: CPCV  ██████
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Block/path/purge helpers (_gran_to_timedelta, _build_datetime_blocks,
# _assign_blocks, _generate_cpcv_splits, _reconstruct_paths) are imported
# from Utils.ts_cross_validation at the top of this module — single source
# of truth shared with the CombinatorialPurgedCV class.


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
                    model_key   = "rf", 
                    dates_valid = None):
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
        model = _build_edge_model(model_key, len(y_train_all), seed=EDGE_SEED, class_weight_ratio=cw_ratio)
        
        # ┏━━━━━━━━━━ Fit model ━━━━━━━━━━┓
        model.fit(X_train_all, y_train_all)

        # ┏━━━━━━━━━━ OOB probs for all training samples ━━━━━━━━━━┓
        if hasattr(model, "oob_decision_function_"):
            oob_probs = model.oob_decision_function_[:, 1]
        else:
            oob_probs = model.predict_proba(X_train_all)[:, 1]  # fallback

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

    # ┏━━━━━━━━━━ Non-RF path: 60/20/20 physical split ━━━━━━━━━━┓
    else:
        # ┏━━━━━━━━━━ Split train → fit (60%), Val-cal (20%), Val-Opt (20%) ━━━━━━━━━━┓
        n_train = len(idx_train)
        n_fit   = int(n_train * 0.60)
        n_cal   = int(n_train * 0.20)
        
        X_fit    = X_train_all[:n_fit]; y_fit = y_train_all[:n_fit]
        X_cal    = X_train_all[n_fit:n_fit+n_cal]; y_cal = y_train_all[n_fit:n_fit+n_cal]
        X_opt    = X_train_all[n_fit+n_cal:]; y_opt = y_train_all[n_fit+n_cal:]
        opt_rets = train_returns_all[n_fit+n_cal:]

        # ┏━━━━━━━━━━ Build model ━━━━━━━━━━┓
        model = _build_edge_model(model_key, n_fit, seed=EDGE_SEED, class_weight_ratio=cw_ratio)
        
        # ┏━━━━━━━━━━ Fit model ━━━━━━━━━━┓
        model.fit(X_fit, y_fit)

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
        "_preds": preds_all.astype(int),
        "_probs": probs_all,
        "_sel": sel_all.astype(int),
        "_rets": rets_all,
        "_assets": asts_all,
    }



# ┏━━━━━━━━━━ CPCV: main runner ━━━━━━━━━━┓
def run_cpcv_analysis(cache_path, config, output_root, n_blocks=6, k_test=2, m2_name="randforest", direction="up", granularity="1d"):
    # ┏━━━━━━━━━━ Initialize variables ━━━━━━━━━━┓
    fee       = config["evaluation"]["fee_per_trade"]
    horizon   = config["data"]["load"]["forecast_horizon"]
    
    n_paths  = n_blocks // k_test
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

    # ┏━━━━━━━━━━ Filter valid samples ━━━━━━━━━━┓
    valid = ~np.isnan(labels)
    eng, labels, returns, asset_ids = eng[valid], labels[valid], returns[valid], asset_ids[valid]
    dates_valid = [dates_all_raw[i] for i in range(len(dates_all_raw)) if valid[i]]
    valid_to_raw = [i for i in range(len(dates_all_raw)) if valid[i]]

    # ┏━━━━━━━━━━ CRITICAL: Exclude Test data from CPCV ━━━━━━━━━━┓
    # CPCV must only use Train+Val data. Test period is completely invisible.
    t_val_end = pd.Timestamp(config["data"]["split"]["val_end"])
    cpcv_mask = np.array([d <= t_val_end for d in dates_valid])
    n_excluded = int((~cpcv_mask).sum())
    eng = eng[cpcv_mask]
    labels = labels[cpcv_mask]
    returns = returns[cpcv_mask]
    asset_ids = asset_ids[cpcv_mask]
    dates_valid = [d for d, m in zip(dates_valid, cpcv_mask) if m]

    print(f"  Samples: {len(labels):,} (excluded {n_excluded:,} test samples after {t_val_end.strftime('%Y-%m-%d')})")
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
    gran_dir = output_root / config["experiment"]["m1"] / m2_name / direction.upper() / granularity
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

    # ┏━━━━━━━━━━ Run splits ━━━━━━━━━━┓
    split_results = []
    for si, sp in enumerate(splits):
        result = _run_cpcv_split(eng, labels, returns, asset_ids, asset_map, sp["idx_train"], sp["idx_test"], direction, fee,
                                 model_key=m2_name)
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
                   "sel_coverage", "sel_mean_ret", "m1_mean_ret", "n_samples", "n_selected"]:
            row[k] = pm[k]
        if pi < len(sharpes):
            row["sharpe"] = sharpes[pi]
            row["total_ret_pct"] = total_rets[pi]
        rows.append(row)
    pd.DataFrame(rows).to_csv(gran_dir / "cpcv_paths.csv", index=False)

    # ┏━━━━━━━━━━ Summary ━━━━━━━━━━┓
    accs    = np.array([pm["accuracy"]      for pm in path_metrics_list])
    precs   = np.array([pm["precision"]     for pm in path_metrics_list])
    s_precs = np.array([pm["sel_precision"] for pm in path_metrics_list])
    s_accs  = np.array([pm["sel_accuracy"]  for pm in path_metrics_list])
    s_covs  = np.array([pm["sel_coverage"]  for pm in path_metrics_list])
    s_rets  = np.array([pm["sel_mean_ret"]  for pm in path_metrics_list])
    worst_sharpe = min(sharpes) if sharpes else 0.0

    # ┏━━━━━━━━━━ Summary ━━━━━━━━━━┓
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
               "path_sharpes": sharpes, "path_total_rets": total_rets}

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
    print(f"  │ Worst Path SR:  {worst_sharpe:.2f}")
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


# ┏━━━━━━━━━━ Compute edge convergence score ━━━━━━━━━━┓
def compute_edge_convergence_score(cache_path, config, direction="up", model_name="randforest", granularity="1d"):
    """2-stage convergence score: CPCV (60%) + Seeds (40%).

    Hard gates (AND logic — both must pass for GREEN):
        CPCV:  median path Sharpe Ratio > 0  AND  >60% paths profitable
        Seeds: >70% seeds profitable  AND  Sharpe Confidence Interval lower > 0
    """
    # ┏━━━━━━━━━━ Load Cache ━━━━━━━━━━┓
    print(f"\n[edge-convergence] Loading cache: {cache_path}")
    multi = _load_multi_cache(cache_path)
    
    # ┏━━━━━━━━━━ Get Granularities ━━━━━━━━━━┓
    edge_root = Path(config["paths"]["output_root"]) / "Analysis" / "Edge" / config["experiment"]["m1"].capitalize() / model_name
    dir_path = edge_root / direction.upper()
    if not dir_path.exists():
        print(f"  [SKIP] Directory not found: {dir_path}")
        return
    grans = sorted([g for g in multi.sub.keys() if (dir_path / g).exists()])

    print(f"\n{'='*80}")
    print(f"[edge-convergence] 2-Stage Convergence Score — {direction.upper()} — {model_name}")
    print(f"{'='*80}")

    # ┏━━━━━━━━━━ Initialize Weights and Results ━━━━━━━━━━┓
    w_cpcv, w_seeds = 0.60, 0.40
    convergence_results = {}

    # ┏━━━━━━━━━━ Get Paths ━━━━━━━━━━┓
    gran_dir = dir_path / granularity
    cpcv_csv = gran_dir / "cpcv_paths.csv"
    seeds_csv = gran_dir / "edge_trials.csv"

    # ┏━━━━━━━━━━ Stage 1: CPCV ━━━━━━━━━━┓
    cpcv_score = 0.0
    cpcv_raw = {"fraction_profitable": None,
                "median_path_sharpe": None,
                "passes_gate": False}
    cpcv_detail = "Missing CPCV results"
    
    # ┏━━━━━━━━━━ Load CPCV Results ━━━━━━━━━━┓
    if cpcv_csv.exists():
        df_cpcv = pd.read_csv(cpcv_csv)

        # ┏━━━━━━━━━━ Calculate CPCV Score ━━━━━━━━━━┓
        if "sharpe" in df_cpcv.columns:
            total_ret_col = "total_ret_pct" if "total_ret_pct" in df_cpcv.columns else "sel_mean_ret"
            path_rets = df_cpcv[total_ret_col].dropna().values
            path_sharpes = df_cpcv["sharpe"].dropna().values

            # ┏━━━━━━━━━━ Check if CPCV Results Exist ━━━━━━━━━━┓
            if len(path_rets) > 0 and len(path_sharpes) > 0:
                frac_profitable = float(np.mean(path_rets > 0))
                median_sharpe = float(np.median(path_sharpes))
                cpcv_score = frac_profitable * float(np.clip(median_sharpe, 0.0, 1.0))
                
                # ┏━━━━━━━━━━ CPCV Raw ━━━━━━━━━━┓
                cpcv_raw = {"fraction_profitable": frac_profitable,
                            "median_path_sharpe":  median_sharpe,
                            "passes_gate":         bool(median_sharpe > 0 and frac_profitable > 0.60)}
                
                # ┏━━━━━━━━━━ CPCV Detail ━━━━━━━━━━┓
                cpcv_detail = (f"profitable = {frac_profitable:.0%} "
                               f"median_SR  = {median_sharpe:.2f} "
                               f"worst_ret  = {float(np.min(path_rets)):+.2f}%")

    # ┏━━━━━━━━━━ Stage 2: Seeds ━━━━━━━━━━┓
    seeds_score = 0.0
    seeds_raw = {"fraction_profitable": None,
                 "mean_sharpe":         None,
                 "sharpe_ci_lower":     None,
                 "median_threshold":    None,
                 "threshold_std":       None,
                 "passes_gate":         False}
    seeds_detail = "missing seeds results"
    
    # ┏━━━━━━━━━━ Load Seeds Results ━━━━━━━━━━┓
    if seeds_csv.exists():
        df_seeds = pd.read_csv(seeds_csv)

        # ┏━━━━━━━━━━ Calculate Seeds Score ━━━━━━━━━━┓
        if "test_sel_mean_ret" in df_seeds.columns and "test_sel_sharpe" in df_seeds.columns:
            seed_rets = df_seeds["test_sel_mean_ret"].dropna().values
            seed_sharpes = df_seeds["test_sel_sharpe"].dropna().values
            seed_thresholds = df_seeds["test_sel_threshold"].dropna().values if "test_sel_threshold" in df_seeds.columns else np.array([])
            
            # ┏━━━━━━━━━━ Check if Seeds Results Exist ━━━━━━━━━━┓
            if len(seed_rets) > 0 and len(seed_sharpes) > 0:
                frac_profitable = float(np.mean(seed_rets > 0))
                mean_sharpe = float(np.mean(seed_sharpes))
                std_sharpe = float(np.std(seed_sharpes, ddof=1)) if len(seed_sharpes) > 1 else 0.0
                sharpe_ci_lower = mean_sharpe - 1.96 * std_sharpe / np.sqrt(max(len(seed_sharpes), 1))
                cv_sharpe = std_sharpe / abs(mean_sharpe) if abs(mean_sharpe) > 1e-9 else 1.0
                stability = float(np.clip(1.0 - cv_sharpe, 0.0, 1.0))
                seeds_score = frac_profitable * stability
                median_thr = float(np.median(seed_thresholds)) if len(seed_thresholds) > 0 else None
                thr_std = float(np.std(seed_thresholds)) if len(seed_thresholds) > 0 else None
                
                # ┏━━━━━━━━━━ Seeds Raw ━━━━━━━━━━┓
                seeds_raw = {"fraction_profitable": frac_profitable,
                             "mean_sharpe":         mean_sharpe,
                             "sharpe_ci_lower":     sharpe_ci_lower,
                             "cv_sharpe":           cv_sharpe,
                             "median_threshold":    median_thr,
                             "threshold_std":       thr_std,
                             "passes_gate":         bool(frac_profitable > 0.70 and sharpe_ci_lower > 0)}

                # ┏━━━━━━━━━━ Seeds Detail ━━━━━━━━━━┓
                seeds_detail = (f"profitable = {frac_profitable:.0%} "
                                f"mean_SR    = {mean_sharpe:.2f} "
                                f"CI_low     = {sharpe_ci_lower:.2f} "
                                f"CV         = {cv_sharpe:.2f} "
                                f"τ_med      = {median_thr:.3f}" if median_thr else
                                f"profitable = {frac_profitable:.0%} "
                                f"mean_SR    = {mean_sharpe:.2f} "
                                f"CI_low     = {sharpe_ci_lower:.2f}")

    # ┏━━━━━━━━━━ Combined score + verdict ━━━━━━━━━━┓
    final_score = w_cpcv * cpcv_score + w_seeds * seeds_score
    
    # ┏━━━━━━━━━━ Verdict ━━━━━━━━━━┓
    cpcv_pass = cpcv_raw.get("passes_gate", False)
    seeds_pass = seeds_raw.get("passes_gate", False)
    if cpcv_pass and seeds_pass:
        verdict = "GREEN"
    elif cpcv_pass or seeds_pass:
        verdict = "AMBER"
    else:
        verdict = "RED"

    # ┏━━━━━━━━━━ Convergence Results ━━━━━━━━━━┓
    convergence_results[granularity] = {"weights":      {"cpcv": w_cpcv, "seeds": w_seeds},
                                 "cpcv":         {**cpcv_raw, "score": round(cpcv_score, 4)},
                                 "seeds":        {**seeds_raw, "score": round(seeds_score, 4)},
                                 "final_score":  round(final_score, 4),
                                 "verdict":      verdict,
                                 "gates_passed": {"cpcv": cpcv_pass, "seeds": seeds_pass}}

    icon = {"GREEN": "✓", "AMBER": "~", "RED": "✗"}[verdict]
    print(f"\n  [{icon}] {granularity:4s} — Score: {final_score:.3f} ({verdict})")
    print(f"      1. CPCV  (w={w_cpcv}): {cpcv_score:.3f}  | {cpcv_detail}")
    print(f"      2. Seeds (w={w_seeds}): {seeds_score:.3f}  | {seeds_detail}")

    # ┏━━━━━━━━━━ Save Convergence Results ━━━━━━━━━━┓
    if convergence_results:
        out_path = dir_path / f"convergence_scores_{granularity}.json"
        with open(out_path, "w") as f:
            json.dump(convergence_results, f, indent=2)
        print(f"\n[edge-convergence] Saved: {out_path}")

        # ┏━━━━━━━━━━ Print Convergence Results ━━━━━━━━━━┓
        print(f"\n  {'Gran':>5s} | {'CPCV':>6s} | {'Seeds':>6s} | {'FINAL':>6s} | Verdict")
        print(f"  {'─'*5} | {'─'*6} | {'─'*6} | {'─'*6} | {'─'*7}")
        for g, r in convergence_results.items():
            print(f"  {g:>5s} | {r['cpcv']['score']:6.3f} | {r['seeds']['score']:6.3f} | "
                  f"{r['final_score']:6.3f} | {r['verdict']}")

    print(f"\n[edge-convergence] Done.\n")



# ┏━━━━━━━━━━ CLI ━━━━━━━━━━┓
def main():
    # ┏━━━━━━━━━━ Parse arguments ━━━━━━━━━━┓
    # _VALID_CLI_MODELS = tuple(_CLI_TO_MODEL_KEY.keys())
    parser = argparse.ArgumentParser(description="Edge Analysis — Model stability (seeds) or regime sensitivity (CPCV)")  # TODO adjust description
    parser.add_argument("--cache_path", type=str, default=None, help="Explicit path to dataset cache .pt")
    parser.add_argument("--config", type=json.loads, help="Experiment config", required=True)
    parser.add_argument("--mode", type=str, choices=["seeds", "cpcv", "convergence"], required=True, help="'seeds' = 100 seed trials; 'cpcv' = Combinatorial Purged CV; convergence = Compute the 3-stage Edge Convergence Score from pre-calculated results")
    parser.add_argument("--phase", type=str, help="Experimental Phase", required=True)
    parser.add_argument("--m2", type=str, help="M2 model to use", required=True)
    parser.add_argument("--direction", type=str, help="Direction to use", required=True)
    parser.add_argument("--granularity", type=str, help="Granularity to use", required=True)

    args = parser.parse_args()

    # ┏━━━━━━━━━━ Output directory (includes model name) ━━━━━━━━━━┓
    output_root = Path(args.config["paths"]["output_root"]) / "Analysis" / "Edge" / args.config["experiment"]["m1"].capitalize() / args.m2

    # ┏━━━━━━━━━━ Run analysis ━━━━━━━━━━┓
    if args.mode == "convergence":
        compute_edge_convergence_score(args.cache_path,
                                       args.config,
                                       direction=args.direction,
                                       model_name=args.m2,
                                       granularity=args.granularity)
    elif args.mode == "seeds":
        run_seeds_analysis(args.cache_path,
                           args.config,
                           output_root,
                           n_trials=args.config["runtime"][args.phase]["n_trials"],
                           m2_name=args.m2,
                           direction=args.direction,
                           granularity=args.granularity)
    elif args.mode == "cpcv":
        run_cpcv_analysis(args.cache_path,
                          args.config,
                          output_root,
                          n_blocks=args.config["runtime"][args.phase]["n_blocks"],
                          k_test=args.config["runtime"][args.phase]["k_test"],
                          m2_name=args.m2,
                          direction=args.direction,
                          granularity=args.granularity)
    else:
        print(f"Invalid mode: {args.mode}")


if __name__ == "__main__":
    main()

