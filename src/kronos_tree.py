"""
Kronos Tree — M2 Meta-Labeling with RF/XGBoost/AutoGluon
=========================================================
Feature analysis, temporal evaluation, and financial backtesting for the M2 meta-labeling system.

Modes:
  --per-gran   Per-granularity models (one model per granularity, current default)
  --all-grans  Unified model trained on all granularities, evaluated per-gran

Usage:
  python Utils/kronos_tree.py --per-gran --cache path/to/multi.pt            # per-gran models (default)
  python Utils/kronos_tree.py --per-gran --cache path/to/multi.pt --features false --top5 false  # skip feature analysis
  python Utils/kronos_tree.py --all-grans --cache path/to/multi.pt           # unified multi-gran model
  python Utils/kronos_tree.py --cache path/to/cache.pt                       # single-gran auto-detect
  python Utils/kronos_tree.py --model xgboost --cache path/to/cache.pt       # use XGBoost
  python Utils/kronos_tree.py --model autogluon --cache path/to/cache.pt     # use AutoGluon ensemble
"""

import argparse, hashlib, sys, json, pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent))

# ┏━━━━━━━━━━ Pipeline Data Preprocessing ━━━━━━━━━━┓
from Utils.data_preprocessing import (ENG_FEATURE_NAMES, ENG_FEATURE_GROUPS, split_by_global_time,
                                      load_dataset_from_config, prepare_multi_asset_dataset,
                                      prepare_multi_gran_dataset, GRAN_SEQ_LEN)

# ┏━━━━━━━━━━ Financial Backtesting ━━━━━━━━━━┓
from Utils.backtest import (_annualization_factor, 
                            _build_spread_equity, 
                            _calc_drawdown,
                            _calc_sharpe, 
                            _equity_horizon_returns, 
                            _load_raw_close_prices,
                            run_feature_backtest)

# ┏━━━━━━━━━━ Comparison of results between models and granularities ━━━━━━━━━━┓
from Utils.comparison import (GRAN_ORDER, 
                              run_comparison, 
                              run_paradigm_comparison)

# ┏━━━━━━━━━━ Feature analysis ━━━━━━━━━━┓
from Utils.features import (_plot_prob_distribution,
                            plot_class_distributions,
                            plot_correlation_heatmap,
                            plot_mutual_information,
                            plot_confusion_matrix, 
                            plot_pointbiserial,
                            plot_tree_importance)

# ┏━━━━━━━━━━ Online Conformal Prediction ━━━━━━━━━━┓
from Utils.saocp import (_ocp_threshold_to_op, 
                         _run_saocp_online)

# ┏━━━━━━━━━━ Utility-based Selective Classification [risk-coverage analysis] ━━━━━━━━━━┓   
from Utils.selective_classification import (_find_best_utility_threshold, 
                                            collect_risk_coverage_curve)

# ┏━━━━━━━━━━ Utils ━━━━━━━━━━┓
from Utils.utils import (NumpyJSONEncoder, 
                         model_label as _model_label, 
                         _safe_json, 
                         _load_config, 
                         _build_cache_from_config,
                         _resolve_caches,
                         _class_names,
                         _infer_direction,
                         _load_multi_cache)


import yaml

# ┏━━━━━━━━━━ Model choices ━━━━━━━━━━┓
MODEL_CHOICES = ("rf", "xgboost", "autogluon")

# ┏━━━━━━━━━━ AutoGluon parameters ━━━━━━━━━━┓
_AG_TIME_LIMIT = 300          # overridden by --ag-time-limit
_AG_PRESETS = "best_quality"  # overridden by --ag-presets


# ┏━━━━━━━━━━ AutoGluon wrapper ━━━━━━━━━━┓
class _AutoGluonWrapper:
    """Sklearn-compatible wrapper around AutoGluon TabularPredictor."""

    def __init__(self, feature_names=None, time_limit=300, presets="best_quality", class_weight_ratio=1.0):
        self.feature_names = feature_names
        self.time_limit = time_limit
        self.presets = presets
        self.class_weight_ratio = class_weight_ratio
        self._predictor = None
        self._label_col = "__target__"
        self._weight_col = "__sample_weight__"

    def _to_df(self, X):
        cols = self.feature_names if self.feature_names else [f"f{i}" for i in range(X.shape[1])]
        return pd.DataFrame(X, columns=cols)

    def fit(self, X, y, sample_weight=None):
        from autogluon.tabular import TabularPredictor
        import tempfile

        df = self._to_df(X)
        df[self._label_col] = y

        ag_ctor_kwargs = {}
        if sample_weight is not None:
            df[self._weight_col] = sample_weight
            ag_ctor_kwargs["sample_weight"] = self._weight_col

        save_dir = tempfile.mkdtemp(prefix="ag_model_")
        self._predictor = TabularPredictor(
            label=self._label_col,
            path=save_dir,
            eval_metric="f1",
            verbosity=1,
            **ag_ctor_kwargs,
        ).fit(
            train_data=df,
            time_limit=self.time_limit,
            presets=self.presets,
        )
        self._train_df = df  # kept for feature_importance
        return self

    def predict(self, X):
        df = self._to_df(X)
        return self._predictor.predict(df).values.astype(int)

    def predict_proba(self, X):
        df = self._to_df(X)
        proba = self._predictor.predict_proba(df)
        return proba.values

    @property
    def feature_importances_(self):
        imp = self._predictor.feature_importance(data=self._train_df, silent=True)
        return imp["importance"].values

    def leaderboard(self):
        """Print and return the AutoGluon model leaderboard."""
        return self._predictor.leaderboard(silent=False)

    def model_info(self, save_dir):
        """Save detailed model info (leaderboard + per-model hyperparams) to files."""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        # 1. Leaderboard CSV
        lb = self._predictor.leaderboard(silent=True)
        lb_path = save_dir / "ag_leaderboard.csv"
        lb.to_csv(lb_path, index=False)
        print(f"  AutoGluon leaderboard saved to {lb_path}")

        # 2. Per-model detailed info JSON
        full_info = self._predictor.info()
        model_info_dict = full_info.get("model_info", {})

        export = {
            "presets": self.presets,
            "time_limit": self.time_limit,
            "eval_metric": "f1",
            "num_models_trained": len(model_info_dict),
            "models": {},
        }
        for name, info in model_info_dict.items():
            export["models"][name] = {
                "model_type": str(info.get("model_type", "N/A")),
                "hyperparameters": {k: _safe_json(v) for k, v in info.get("hyperparameters", {}).items()},
                "num_features": info.get("num_features", None),
                "stack_level": info.get("stacker_info", {}).get("stacker_level", 0)
                               if isinstance(info.get("stacker_info"), dict) else 0,
                "fit_time": round(info.get("fit_time", 0), 2),
                "pred_time_val": round(info.get("pred_time_val", 0), 4),
                "val_score": round(info.get("val_score", 0), 4),
                "children": info.get("children_info", {}).get("children", [])
                            if isinstance(info.get("children_info"), dict) else [],
            }

        info_path = save_dir / "ag_model_info.json"
        with open(info_path, "w") as f:
            json.dump(export, f, indent=2, default=str)
        print(f"  AutoGluon model info saved to {info_path}")
        return export

    def save_to(self, path):
        """Persist the AutoGluon predictor to a permanent directory."""
        import shutil
        dest = Path(path) / "ag_model"
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(self._predictor.path, str(dest))
        print(f"  AutoGluon model saved to {dest}")
        return dest


# ┏━━━━━━━━━━ Build Tree Model [Random Forest, XGBoost, AutoGluon] ━━━━━━━━━━┓
def _build_tree_model(model_name: str, n_samples: int, class_weight_ratio: float = 1.0,
                      feature_names=None, time_limit=None, presets="best_quality"):
    """Return a scikit-learn-compatible classifier.

    Args:
        model_name: 'rf', 'xgboost', or 'autogluon'
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
        return XGBClassifier(n_estimators       = 300,
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
        return _AutoGluonWrapper(feature_names      = feature_names,
                                 time_limit         = time_limit if time_limit is not None else _AG_TIME_LIMIT,
                                 presets            = presets,
                                 class_weight_ratio = class_weight_ratio)
    else:
        raise ValueError(f"Unknown model: {model_name}. Choose from {MODEL_CHOICES}")


def _build_dataframe(dataset: dict) -> tuple[pd.DataFrame, np.ndarray]:
    # ┏━━━━━━━━━━ Convert eng_features to numpy ━━━━━━━━━━┓
    eng = dataset["eng_features"]
    if isinstance(eng, torch.Tensor):
        eng = eng.numpy()
    labels = dataset["labels"]

    # ┏━━━━━━━━━━ Convert labels to numpy ━━━━━━━━━━┓
    if isinstance(labels, torch.Tensor):
        labels = labels.numpy()

    # ┏━━━━━━━━━━ Filter out NaN labels ━━━━━━━━━━┓
    valid = ~np.isnan(labels)
    eng = eng[valid]
    labels = labels[valid].astype(int)

    # ┏━━━━━━━━━━ Create dataframe ━━━━━━━━━━┓
    df = pd.DataFrame(eng, columns=ENG_FEATURE_NAMES)
    return df, labels


def temporal_eval(dataset: dict, 
                  feature_cols: list, 
                  save_dir: Path,
                  class_names: list, 
                  meta_mode: str,
                  split_indices: tuple,
                  direction: str = "up",
                  fee: float = 0.0,
                  model_name: str = "rf",
                  desc: str = "all features", file_prefix: str = "8_temporal_all",
                  thres_mode: str = "utility", ocp_alpha: float = 0.10,
                  forecast_horizon: int = 1,      
                  split_indices_raw: tuple = None) -> dict:
    """Train model on train split, evaluate on val and test. Plot confusion matrices.

    Threshold selection uses financial utility (t-statistic of net returns)
    on the validation set instead of a fixed classification-risk budget.

    Args:
        split_indices: (idx_train, idx_val, idx_test) pre-computed from split_by_global_time.
        direction: 'up' or 'down' — used to flip returns for short strategies.
        fee: per-trade fee (decimal, e.g. 0.002 for 0.2%).
    """
    mlabel = _model_label(model_name)

    eng = dataset["eng_features"]
    if isinstance(eng, torch.Tensor):
        eng = eng.numpy()
    labels = dataset["labels"]
    if isinstance(labels, torch.Tensor):
        labels = labels.numpy()
    returns_all = dataset["returns"]
    if isinstance(returns_all, torch.Tensor):
        returns_all = returns_all.numpy()

    idx_train, idx_val, idx_test = split_indices

    m1_acc_val, m1_prec_val = None, None
    m1_acc_test, m1_prec_test = None, None
    if split_indices_raw is not None:
        from sklearn.metrics import accuracy_score, precision_score
        idx_train_raw, idx_val_raw, idx_test_raw = split_indices_raw
        
        m1_pred_all = dataset.m1_pred_labels if hasattr(dataset, 'm1_pred_labels') else dataset.get('m1_pred_labels')
        m1_true_all = dataset.m1_true_labels if hasattr(dataset, 'm1_true_labels') else dataset.get('m1_true_labels')

        if m1_pred_all is not None and m1_true_all is not None:
            if isinstance(m1_pred_all, torch.Tensor): m1_pred_all = m1_pred_all.numpy()
            if isinstance(m1_true_all, torch.Tensor): m1_true_all = m1_true_all.numpy()

            def _calc_m1(idx_raw):
                p, t = m1_pred_all[idx_raw], m1_true_all[idx_raw]
                valid = frozenset(np.where(~np.isnan(p) & ~np.isnan(t))[0])
                if len(valid) > 0:
                    v_idx = list(valid)
                    return accuracy_score(t[v_idx], p[v_idx]), precision_score(t[v_idx], p[v_idx], zero_division=0)
                return None, None

            m1_acc_val, m1_prec_val = _calc_m1(idx_val_raw)
            m1_acc_test, m1_prec_test = _calc_m1(idx_test_raw)

    all_names = list(ENG_FEATURE_NAMES)
    col_indices = [all_names.index(c) for c in feature_cols]

    X_train = eng[idx_train][:, col_indices]
    y_train = labels[idx_train].astype(int)
    X_val   = eng[idx_val][:, col_indices]
    y_val   = labels[idx_val].astype(int)
    X_test  = eng[idx_test][:, col_indices]
    y_test  = labels[idx_test].astype(int)

    # Direction-aware returns for utility calculation
    val_returns = returns_all[idx_val].copy()
    test_returns = returns_all[idx_test].copy()
    if direction.lower() == "down":
        val_returns = -val_returns
        test_returns = -test_returns

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val   = scaler.transform(X_val)
    X_test  = scaler.transform(X_test)

    n_pos = int((y_train == 1).sum())
    n_neg = int((y_train == 0).sum())
    cw_ratio = n_neg / max(n_pos, 1)

    model = _build_tree_model(model_name, len(y_train), cw_ratio,
                              feature_names=feature_cols,
                              presets=_AG_PRESETS)
    model.fit(X_train, y_train)

    if model_name == "autogluon":
        model.leaderboard()
        model.model_info(save_dir)
        model.save_to(save_dir)

    results = {}
    val_thresholds = {}
    val_op = None

    # Pre-compute val probs for OCP warm-up (needed before test loop)
    _val_probs_ocp = model.predict_proba(X_val)[:, 1] if thres_mode == "OCP" else None
    # Extract dates for delayed feedback in SAOCP (both val and test)
    _val_dates_ocp  = [dataset["dates"][j] for j in idx_val]  if thres_mode == "OCP" else None
    _test_dates_ocp = [dataset["dates"][j] for j in idx_test] if thres_mode == "OCP" else None

    for split_name, X_split, y_split, split_rets in [("Val", X_val, y_val, val_returns),
                                                       ("Test", X_test, y_test, test_returns)]:
        preds = model.predict(X_split)
        probs = model.predict_proba(X_split)[:, 1]  # P(class=1)
        prec_val = round(float(precision_score(y_split, preds, zero_division=0)), 4)
        n_pred_pos = int((preds == 1).sum())
        metrics = {
            "accuracy":  round(float(accuracy_score(y_split, preds)), 4),
            "precision": prec_val,
            "recall":    round(float(recall_score(y_split, preds, zero_division=0)), 4),
            "f1_score":  round(float(f1_score(y_split, preds, zero_division=0)), 4),
            "coverage":  round(n_pred_pos / len(y_split), 4) if len(y_split) > 0 else 0,
            "risk":      round(1 - prec_val, 4),
            "baseline":  round(int((y_split == 1).sum()) / len(y_split), 4) if len(y_split) > 0 else 0,
        }
        results[split_name] = metrics

        m1_a = m1_acc_test if split_name == "Test" else m1_acc_val
        m1_p = m1_prec_test if split_name == "Test" else m1_prec_val

        cm_path = save_dir / f"{file_prefix}_{split_name}_CM.png"
        plot_confusion_matrix(y_split, preds, classes=class_names,
                              save_path=str(cm_path),
                              title=f"{mlabel} {split_name} (@thr=0.5) | {desc}",
                              meta_mode=meta_mode,
                              m1_acc=m1_a, m1_prec=m1_p)

        _plot_prob_distribution(y_split, probs, class_names, save_dir,
                                file_prefix=f"{file_prefix}_{split_name}_Prob_Dist",
                                title=f"{mlabel} {split_name} ({desc})")

        # Risk-coverage curve (always computed on own data for visualization)
        curve = collect_risk_coverage_curve(y_true=y_split, y_score=probs)

        if split_name == "Val":
            # Financial utility threshold optimization on Val (always computed)
            op = _find_best_utility_threshold(probs, split_rets, fee=fee)
            sel_val = probs >= op["threshold"]
            n_sel_val = int(sel_val.sum())
            err_val = int((y_split[sel_val] == 0).sum()) if n_sel_val > 0 else 0
            op["risk"] = err_val / max(n_sel_val, 1)
            val_thresholds["thr"] = op["threshold"]
            val_op = op
        else:
            if thres_mode == "OCP":
                # OCP: warm up on val, adapt on test (delayed feedback)
                test_s_hats, test_approved_ocp, val_s_hats, conf_stats = _run_saocp_online(
                    _val_probs_ocp, y_val, probs, y_split, alpha=ocp_alpha,
                    test_dates=_test_dates_ocp,
                    forecast_horizon=forecast_horizon,
                    val_dates=_val_dates_ocp)
                op = _ocp_threshold_to_op(probs, y_split, split_rets,
                                          test_approved_ocp, test_s_hats, fee,
                                          conformal_stats=conf_stats)
                # Store for backtest
                val_op["_ocp_test_approved"] = test_approved_ocp
                val_op["_ocp_test_thresholds"] = test_s_hats
                val_op["_ocp_val_thresholds"] = val_s_hats
                cc = conf_stats["conformal_coverage"]
                print(f"    OCP: α={ocp_alpha}, median τ={op['threshold']:.3f}, "
                      f"cov={op['coverage']:.1%}, μ={op['mean_ret']*100:+.3f}% | "
                      f"Conformal cov={cc:.1%} (target≥{1-ocp_alpha:.0%}) | "
                      f"Sets: {{1}}={conf_stats['n_set_1']} {{0}}={conf_stats['n_set_0']} "
                      f"{{0,1}}={conf_stats['n_set_both']} {{}}={conf_stats['n_set_empty']}")
                # Save OCP diagnostics npz for offline analysis
                np.savez_compressed(
                    save_dir / f"{file_prefix}_{split_name}_ocp_diagnostics.npz",
                    test_s_hats=test_s_hats,
                    val_s_hats=val_s_hats,
                    val_probs=_val_probs_ocp,
                    val_labels=y_val.astype(int),
                    alpha=np.array([ocp_alpha]),
                )
            else:
                # Utility: apply fixed Val threshold to test
                thr = val_thresholds["thr"]
                sel = probs >= thr
                n_sel = int(sel.sum())
                err = int((y_split[sel] == 0).sum()) if n_sel > 0 else 0
                net_rets_test = split_rets[sel] - fee if n_sel > 0 else np.array([0.0])
                mu_test = float(np.nanmean(net_rets_test))
                sigma_test = float(np.nanstd(net_rets_test, ddof=1)) if n_sel > 1 else 0.0
                t_test = mu_test / sigma_test * np.sqrt(n_sel) if sigma_test > 0 else 0.0
                op = {"threshold": thr, "coverage": n_sel / len(y_split),
                      "risk": err / max(n_sel, 1), "selected_count": n_sel,
                      "constraint_satisfied": True,
                      "mean_ret": mu_test, "t_stat": t_test}

        # ── Plot risk-coverage with return overlay (business-level) ──
        rc_path = save_dir / f"{file_prefix}_{split_name}_Risk_Coverage.png"

        # Colors
        C_RISK   = "#1B4F72"   # deep navy for risk curve
        C_RET    = "#1E8449"   # forest green for positive mean return
        C_RET_N  = "#8B0000"   # dark red for negative mean return
        C_WIN    = "#1E8449"   # lighter green for mean win
        C_LOSS   = "#8B0000"   # burgundy for mean loss
        C_FILL   = "#E8F8F5"   # very light teal for win/loss gap fill
        C_FILLR  = "#FADBD8"   # very light pink for loss side
        C_OP     = "#8B008B"   # dark magenta for operating point
        C_GRID   = "#D5D8DC"   # subtle grey for grid
        C_THR05  = "#34495E"   # darker grey for thr=0.5 line

        fig_rc, ax_rc = plt.subplots(figsize=(10, 6.5), facecolor="white")
        ax_rc.set_facecolor("#FAFAFA")

        # ── Smooth risk curve via PCHIP ──
        thrs = curve["thresholds"]
        covs = curve["coverage"]
        risks_raw = curve["risk"]
        order = np.argsort(covs)
        covs_s, risks_s = covs[order], risks_raw[order]
        umask = np.concatenate(([True], np.diff(covs_s) != 0))
        cov_u, risk_u = covs_s[umask], risks_s[umask]
        fmask = np.isfinite(risk_u)
        cov_u, risk_u = cov_u[fmask], risk_u[fmask]
        if cov_u.size >= 2:
            from scipy.interpolate import PchipInterpolator
            grid_cov = np.linspace(cov_u.min(), cov_u.max(), 300)
            risk_smooth = PchipInterpolator(cov_u, risk_u, extrapolate=False)(grid_cov)
            valid_r = np.isfinite(risk_smooth)
            grid_cov, risk_smooth = grid_cov[valid_r], risk_smooth[valid_r]
        else:
            grid_cov, risk_smooth = cov_u, risk_u

        ax_rc.plot(grid_cov, risk_smooth, color=C_RISK, linewidth=2.2, label="Risk (Error Rate)", zorder=3)
        ax_rc.set_xlabel("Coverage", fontsize=11, fontweight="bold", color="black", labelpad=8)
        ax_rc.set_ylabel("Risk (Error Rate)", fontsize=11, fontweight="bold", color="black", labelpad=8)
        ax_rc.tick_params(axis="x", colors="black", labelcolor="black", labelsize=9, width=1.5)
        ax_rc.tick_params(axis="y", colors="black", labelcolor="black", labelsize=9, width=1.5)
        
        for spine in ax_rc.spines.values():
            spine.set_color("black")
            spine.set_linewidth(1.5)
        
        plt.setp(ax_rc.get_xticklabels(), fontweight="bold")
        plt.setp(ax_rc.get_yticklabels(), fontweight="bold")

        ax_rc.set_xlim(-0.02, 1.02)
        ax_rc.grid(True, which="major", color=C_GRID, linewidth=0.6, alpha=0.7)
        ax_rc.set_axisbelow(True)

        # ── Compute mean return curves at each threshold ──
        mean_rets = np.full_like(thrs, np.nan)
        mean_win_rets = np.full_like(thrs, np.nan)
        mean_lose_rets = np.full_like(thrs, np.nan)
        for _i, _thr in enumerate(thrs):
            _sel = probs >= _thr
            _n = int(_sel.sum())
            if _n >= 2:
                _net = split_rets[_sel] - fee
                _labels = y_split[_sel]
                mean_rets[_i] = float(np.nanmean(_net))
                _winners = _net[_labels == 1]
                _losers = _net[_labels == 0]
                if len(_winners) >= 1:
                    mean_win_rets[_i] = float(np.nanmean(_winners))
                if len(_losers) >= 1:
                    mean_lose_rets[_i] = float(np.nanmean(_losers))

        ax_ret = ax_rc.twinx()
        valid   = ~np.isnan(mean_rets)
        valid_w = ~np.isnan(mean_win_rets)
        valid_l = ~np.isnan(mean_lose_rets)

        # Shaded fill between mean win and mean loss
        both_valid = valid_w & valid_l
        if both_valid.any():
            ax_ret.fill_between(covs[both_valid],
                                mean_win_rets[both_valid] * 100,
                                mean_lose_rets[both_valid] * 100,
                                alpha=0.06, color=C_WIN, zorder=1, label="_nolegend_")

        # Dynamically colored return curves
        def plot_dynamic_return(ax, x, y, lw, ls, alpha, label, zorder):
            if len(x) > 1:
                from matplotlib.collections import LineCollection
                points = np.array([x, y]).T.reshape(-1, 1, 2)
                segments = np.concatenate([points[:-1], points[1:]], axis=1)
                y_mids = segments[:, :, 1].mean(axis=1)
                seg_colors = [C_RET if ym >= 0 else C_RET_N for ym in y_mids]
                lc = LineCollection(segments, colors=seg_colors, linewidth=lw, linestyles=ls, alpha=alpha, zorder=zorder)
                ax.add_collection(lc)
                if label and label != "_nolegend_":
                    ax.plot([], [], color=C_RET, linewidth=lw, linestyle=ls, label=label)
            else:
                c = C_RET if y[0] >= 0 else C_RET_N
                ax.plot(x, y, color=c, linewidth=lw, linestyle=ls, alpha=alpha, label=label, zorder=zorder)

        plot_dynamic_return(ax_ret, covs[valid], mean_rets[valid] * 100, 2.0, "-", 0.9, "Mean Return", 3)
        plot_dynamic_return(ax_ret, covs[valid_w], mean_win_rets[valid_w] * 100, 1.0, ":", 0.8, "_nolegend_", 2)
        plot_dynamic_return(ax_ret, covs[valid_l], mean_lose_rets[valid_l] * 100, 1.0, ":", 0.8, "_nolegend_", 2)
        ax_ret.axhline(y=0, color=C_RET, linestyle=":", alpha=0.35, linewidth=1.0)
        ax_ret.set_ylabel("Return (%)", fontsize=11, fontweight="bold", color="black", labelpad=8)
        ax_ret.tick_params(axis="y", colors="black", labelcolor="black", labelsize=9, width=1.5)
        
        for spine in ax_ret.spines.values():
            spine.set_color("black")
            spine.set_linewidth(1.5)
            
        plt.setp(ax_ret.get_yticklabels(), fontweight="bold")

        # ── τ = 0.5 vertical line + label at intersection ──
        idx_05 = np.argmin(np.abs(thrs - 0.5))
        cov_05 = covs[idx_05]
        risk_05 = risks_raw[idx_05]

        # ── Operating point ──
        thr_source = op.get("threshold_source", "OCP-SAOCP" if thres_mode == "OCP" else ("Val-Utility" if split_name == "Test" else "Utility-Opt"))
        op_cov = op["coverage"]
        op_risk = op.get("risk", 0)

        # Do not show baseline if it physically overlaps the operating point on the plot
        show_baseline = abs(op_cov - cov_05) > 0.02 and abs(op["threshold"] - 0.5) > 0.01

        if show_baseline:
            ax_rc.axvline(x=cov_05, color=C_THR05, linestyle='--', alpha=0.7, linewidth=1.8)
            ax_rc.scatter([cov_05], [risk_05], color=C_THR05, marker="o", s=40,
                          edgecolors="white", linewidths=1.0, zorder=5)
            ax_rc.annotate(f"τ=0.50", xy=(cov_05, risk_05), xytext=(3, 5),
                           textcoords="offset points", fontsize=7, color=C_THR05,
                           fontweight="bold", zorder=10,
                           bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=C_THR05,
                                     alpha=0.8, lw=0.6))

        # Vertical line at operating point
        ax_rc.axvline(x=op_cov, color=C_OP, linestyle='--', alpha=0.7, linewidth=1.8)

        # Diamond marker on risk curve + τ̂ label
        ax_rc.scatter([op_cov], [op_risk], color=C_OP, marker="D", s=40,
                      edgecolors="white", linewidths=1.0, zorder=6)
        ax_rc.annotate(f"$\\hat{{\\tau}}$={op['threshold']:.3f}",
                       xy=(op_cov, op_risk), xytext=(3, 6),
                       textcoords="offset points", fontsize=7.5, color=C_OP,
                       fontweight="bold", zorder=10,
                       bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=C_OP,
                                 alpha=0.85, lw=0.6))

        # Intersections at operating point — use actual selection mask
        mr_val = op["mean_ret"] * 100
        if split_name == "Test" and thres_mode == "OCP":
            _sel_op = test_approved_ocp
        else:
            _sel_op = probs >= op["threshold"]
        _n_op = int(_sel_op.sum())
        if _n_op >= 2:
            _net_op = split_rets[_sel_op] - fee
            _lab_op = y_split[_sel_op]
            _w_op = _net_op[_lab_op == 1]
            _l_op = _net_op[_lab_op == 0]
            mw_val = float(np.nanmean(_w_op)) * 100 if len(_w_op) >= 1 else None
            ml_val = float(np.nanmean(_l_op)) * 100 if len(_l_op) >= 1 else None
        else:
            mw_val, ml_val = None, None

        # Intersections at baseline (τ=0.50)
        mr_05 = mean_rets[idx_05] * 100 if not np.isnan(mean_rets[idx_05]) else None
        mw_05 = mean_win_rets[idx_05] * 100 if not np.isnan(mean_win_rets[idx_05]) else None
        ml_05 = mean_lose_rets[idx_05] * 100 if not np.isnan(mean_lose_rets[idx_05]) else None

        # Function to space out text properly based on value ranking
        def _get_staggered_offsets(val_dict):
            valid = {k: v for k, v in val_dict.items() if v is not None}
            s_keys = sorted(valid.keys(), key=lambda k: valid[k])
            if len(s_keys) == 3: return {s_keys[0]: (3, -8), s_keys[1]: (3, 0), s_keys[2]: (3, 8)}
            elif len(s_keys) == 2: return {s_keys[0]: (3, -5), s_keys[1]: (3, 5)}
            elif len(s_keys) == 1: return {s_keys[0]: (3, 0)}
            return {}

        # Plot baseline (τ=0.50) intersections
        if show_baseline:
            offs_05 = _get_staggered_offsets({"mw": mw_05, "ml": ml_05, "mr": mr_05})
            if mw_05 is not None:
                c_mw = C_RET if mw_05 >= 0 else C_RET_N
                ax_ret.scatter([cov_05], [mw_05], color=c_mw, marker="o", s=25,
                               edgecolors="white", linewidths=0.6, zorder=5, alpha=0.9)
                ax_ret.annotate(f"{mw_05:+.2f}%", xy=(cov_05, mw_05), xytext=offs_05["mw"],
                                textcoords="offset points", fontsize=7.5, color=c_mw,
                                fontweight="bold", zorder=10)
            if ml_05 is not None:
                c_ml = C_RET if ml_05 >= 0 else C_RET_N
                ax_ret.scatter([cov_05], [ml_05], color=c_ml, marker="o", s=25,
                               edgecolors="white", linewidths=0.6, zorder=5, alpha=0.9)
                ax_ret.annotate(f"{ml_05:+.2f}%", xy=(cov_05, ml_05), xytext=offs_05["ml"],
                                textcoords="offset points", fontsize=7.5, color=c_ml,
                                fontweight="bold", zorder=10)
            if mr_05 is not None:
                mc_05 = C_RET if mr_05 >= 0 else C_RET_N
                ax_ret.scatter([cov_05], [mr_05], color=mc_05, marker="o", s=35,
                               edgecolors="white", linewidths=0.8, zorder=5)
                ax_ret.annotate(f"{mr_05:+.2f}%", xy=(cov_05, mr_05), xytext=offs_05["mr"],
                                textcoords="offset points", fontsize=7.5, color=mc_05,
                                fontweight="bold", zorder=10)

        # Plot operating point (τ̂) intersections
        offs_op = _get_staggered_offsets({"mw": mw_val, "ml": ml_val, "mr": mr_val})
        if mr_val is not None:
            mc_val = C_RET if mr_val >= 0 else C_RET_N
            ax_ret.scatter([op_cov], [mr_val], color=mc_val, marker="D", s=40,
                           edgecolors="white", linewidths=1.0, zorder=7)
            ax_ret.annotate(f"{mr_val:+.2f}%", xy=(op_cov, mr_val), xytext=offs_op["mr"],
                            textcoords="offset points", fontsize=7.5, color=mc_val,
                            fontweight="bold", zorder=10)
        if mw_val is not None:
            c_mw = C_RET if mw_val >= 0 else C_RET_N
            ax_ret.scatter([op_cov], [mw_val], color=c_mw, marker="o", s=35,
                           edgecolors="white", linewidths=0.8, zorder=6)
            ax_ret.annotate(f"{mw_val:+.2f}%", xy=(op_cov, mw_val), xytext=offs_op["mw"],
                            textcoords="offset points", fontsize=7.5, color=c_mw,
                            fontweight="bold", zorder=10)
        if ml_val is not None:
            c_ml = C_RET if ml_val >= 0 else C_RET_N
            ax_ret.scatter([op_cov], [ml_val], color=c_ml, marker="o", s=35,
                           edgecolors="white", linewidths=0.8, zorder=6)
            ax_ret.annotate(f"{ml_val:+.2f}%", xy=(op_cov, ml_val), xytext=offs_op["ml"],
                            textcoords="offset points", fontsize=7.5, color=c_ml,
                            fontweight="bold", zorder=10)

        # ── Utility reference point on Test when using OCP ──
        C_UTIL_REF = "#E67E22"  # bold orange for utility reference
        _util_ref_plotted = False
        if split_name == "Test" and thres_mode == "OCP" and val_op.get("constraint_satisfied", False):
            _util_thr = val_thresholds["thr"]
            _util_sel = probs >= _util_thr
            _util_n = int(_util_sel.sum())
            _util_cov = _util_n / len(y_split) if len(y_split) > 0 else 0
            _util_risk = int((y_split[_util_sel] == 0).sum()) / max(_util_n, 1) if _util_n > 0 else 0
            # Return metrics at utility threshold
            if _util_n >= 2:
                _util_net = split_rets[_util_sel] - fee
                _util_lab = y_split[_util_sel]
                _util_mr = float(np.nanmean(_util_net)) * 100
                _util_w = _util_net[_util_lab == 1]
                _util_l = _util_net[_util_lab == 0]
                _util_mw = float(np.nanmean(_util_w)) * 100 if len(_util_w) >= 1 else None
                _util_ml = float(np.nanmean(_util_l)) * 100 if len(_util_l) >= 1 else None
            else:
                _util_mr, _util_mw, _util_ml = 0, None, None

            # Vertical line
            ax_rc.axvline(x=_util_cov, color=C_UTIL_REF, linestyle='--', alpha=0.7, linewidth=1.5)
            # Diamond on risk curve
            ax_rc.scatter([_util_cov], [_util_risk], color=C_UTIL_REF, marker="D", s=45,
                          edgecolors="white", linewidths=1.0, zorder=6)
            ax_rc.annotate(f"τ_util={_util_thr:.3f}",
                           xy=(_util_cov, _util_risk), xytext=(5, -12),
                           textcoords="offset points", fontsize=7.5, color=C_UTIL_REF,
                           fontweight="bold", zorder=10,
                           bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=C_UTIL_REF,
                                     alpha=0.85, lw=0.6))
            # Return intersections
            _util_offs = _get_staggered_offsets({"mw": _util_mw, "ml": _util_ml, "mr": _util_mr})
            if _util_mr is not None:
                _mc = C_RET if _util_mr >= 0 else C_RET_N
                ax_ret.scatter([_util_cov], [_util_mr], color=C_UTIL_REF, marker="s", s=35,
                               edgecolors="white", linewidths=0.8, zorder=6)
                ax_ret.annotate(f"{_util_mr:+.2f}%", xy=(_util_cov, _util_mr),
                                xytext=_util_offs.get("mr", (6, 8)),
                                textcoords="offset points", fontsize=7.5, color=C_UTIL_REF,
                                fontweight="bold", zorder=10)
            if _util_mw is not None:
                ax_ret.scatter([_util_cov], [_util_mw], color=C_UTIL_REF, marker="o", s=25,
                               edgecolors="white", linewidths=0.6, zorder=5, alpha=0.9)
                ax_ret.annotate(f"{_util_mw:+.2f}%", xy=(_util_cov, _util_mw),
                                xytext=_util_offs.get("mw", (6, -6)),
                                textcoords="offset points", fontsize=7.5, color=C_UTIL_REF,
                                fontweight="bold", zorder=10)
            if _util_ml is not None:
                ax_ret.scatter([_util_cov], [_util_ml], color=C_UTIL_REF, marker="o", s=25,
                               edgecolors="white", linewidths=0.6, zorder=5, alpha=0.9)
                ax_ret.annotate(f"{_util_ml:+.2f}%", xy=(_util_cov, _util_ml),
                                xytext=_util_offs.get("ml", (6, -6)),
                                textcoords="offset points", fontsize=7.5, color=C_UTIL_REF,
                                fontweight="bold", zorder=10)
            _util_ref_plotted = True

        # ── Title ──
        ax_rc.set_title(f"Coverage-Risk  |  {split_name}  |  {mlabel}",
                        fontsize=13, fontweight="bold", color="#2C3E50", pad=12)

        # ── Layout ──
        fig_rc.tight_layout()
        n_legend_rows = 3 if _util_ref_plotted else 2
        fig_rc.subplots_adjust(bottom=0.22 + (0.05 if _util_ref_plotted else 0.0))

        # ── Unified legend ──
        from matplotlib.lines import Line2D

        card_edge = C_RET if op["constraint_satisfied"] else C_OP
        legend_prop = {"size": 8}

        # Row 1 (TOP): Avg Win, Avg Loss, τ̂ (OCP or utility)
        row1_handles, row1_labels = [], []
        if mw_val is not None:
            c_mw = C_RET if mw_val >= 0 else C_RET_N
            row1_handles.append(Line2D([], [], color=c_mw, linewidth=1.0,
                                       linestyle=":", marker="o", markersize=4,
                                       markeredgecolor="white", markeredgewidth=0.6))
            row1_labels.append(f"Avg Win = {mw_val:+.2f}%")
        if ml_val is not None:
            c_ml = C_RET if ml_val >= 0 else C_RET_N
            row1_handles.append(Line2D([], [], color=c_ml, linewidth=1.0,
                                       linestyle=":", marker="o", markersize=4,
                                       markeredgecolor="white", markeredgewidth=0.6))
            row1_labels.append(f"Avg Loss = {ml_val:+.2f}%")
        row1_handles.append(Line2D([], [], color=C_OP, marker="D", markersize=5,
                                   linestyle="--", linewidth=1.5, alpha=0.8,
                                   markeredgecolor="white", markeredgewidth=0.8))
        row1_labels.append(f"τ̂ = {op['threshold']:.3f}  ({thr_source})   "
                           f"Cov = {op['coverage']:.1%}   N = {op['selected_count']}   "
                           f"t = {op['t_stat']:.1f}")

        leg_top = fig_rc.legend(row1_handles, row1_labels,
                                loc="lower center", ncol=len(row1_handles), prop=legend_prop,
                                frameon=True, framealpha=0.92, edgecolor="none",
                                fancybox=True,
                                bbox_to_anchor=(0.5, 0.065 + (0.05 if _util_ref_plotted else 0.0)),
                                handlelength=2.5, handletextpad=0.8)

        # Row 2: Risk, Mean Return
        row2_handles, row2_labels = [], []
        op_risk_val = op.get("risk", 0)
        row2_handles.append(Line2D([], [], color=C_RISK, linewidth=2.2))
        row2_labels.append(f"Risk = {op_risk_val:.1%}")
        mr_legend_color = C_RET if op['mean_ret'] >= 0 else C_RET_N
        row2_handles.append(Line2D([], [], color=mr_legend_color, linewidth=2.0))
        row2_labels.append(f"Mean Ret = {op['mean_ret']*100:+.2f}%")

        fig_rc.legend(row2_handles, row2_labels,
                      loc="lower center", ncol=2, prop=legend_prop,
                      frameon=True, framealpha=0.92, edgecolor="none",
                      fancybox=True,
                      bbox_to_anchor=(0.5, 0.015 + (0.05 if _util_ref_plotted else 0.0)),
                      handlelength=2.5, handletextpad=0.8)

        # Row 3 (only OCP): Utility reference
        if _util_ref_plotted:
            row3_handles, row3_labels = [], []
            row3_handles.append(Line2D([], [], color=C_UTIL_REF, marker="D", markersize=5,
                                       linestyle="--", linewidth=1.5, alpha=0.8,
                                       markeredgecolor="white", markeredgewidth=0.8))
            _util_t_stat = val_op.get("t_stat", 0)
            row3_labels.append(f"τ_util = {_util_thr:.3f}  (Val-Utility)   "
                               f"Cov = {_util_cov:.1%}   N = {_util_n}   "
                               f"Risk = {_util_risk:.1%}   MeanRet = {_util_mr:+.2f}%")
            fig_rc.legend(row3_handles, row3_labels,
                          loc="lower center", ncol=1, prop=legend_prop,
                          frameon=True, framealpha=0.92, edgecolor="none",
                          fancybox=True, bbox_to_anchor=(0.5, 0.015),
                          handlelength=2.5, handletextpad=0.8)

        fig_rc.savefig(str(rc_path), dpi=200, facecolor="white")
        plt.close(fig_rc)

        # ── OCP threshold evolution plot (test only) ──
        if split_name == "Test" and thres_mode == "OCP":
            fig_thr, ax_thr = plt.subplots(figsize=(10, 4), facecolor="white")
            ax_thr.set_facecolor("#FAFAFA")
            # Effective threshold: τ_t = max(ŝ_t, 1 - ŝ_t)
            eff_tau = np.maximum(test_s_hats, 1.0 - test_s_hats)
            ax_thr.plot(eff_tau, color="#8B008B", linewidth=0.8, alpha=0.9, label="τ_t (OCP)")
            ax_thr.axhline(y=val_thresholds["thr"], color="#34495E", linestyle="--",
                           linewidth=1.2, alpha=0.7, label=f"τ Utility = {val_thresholds['thr']:.3f}")
            ax_thr.axhline(y=0.5, color="#BDC3C7", linestyle=":", linewidth=0.8, alpha=0.6)
            ax_thr.set_xlabel("Test sample index", fontsize=10)
            ax_thr.set_ylabel("Threshold τ_t", fontsize=10)
            cc = op.get("conformal_coverage", 0)
            n1 = op.get("n_set_1", 0)
            n0 = op.get("n_set_0", 0)
            nb = op.get("n_set_both", 0)
            ne = op.get("n_set_empty", 0)
            ax_thr.set_title(
                f"OCP Threshold Evolution  |  Test  |  {mlabel}  (α={ocp_alpha})\n"
                f"Conformal Cov={cc:.1%} (target≥{1-ocp_alpha:.0%})  |  "
                f"{{1}}={n1}  {{0}}={n0}  {{0,1}}={nb}  {{}}={ne}",
                fontsize=11, fontweight="bold", color="#2C3E50")
            ax_thr.legend(fontsize=8, loc="upper right")
            ax_thr.set_ylim(0.4, 1.0)
            ax_thr.grid(True, alpha=0.3)
            fig_thr.tight_layout()
            thr_evo_path = save_dir / f"{file_prefix}_Test_OCP_Threshold_Evolution.png"
            fig_thr.savefig(str(thr_evo_path), dpi=200, facecolor="white")
            plt.close(fig_thr)

        # Print and store selective metrics + confusion matrix
        thr_sel = op["threshold"]
        if split_name == "Test" and thres_mode == "OCP":
            sel = test_approved_ocp
        else:
            sel = probs >= thr_sel
        sel_preds = sel.astype(int)
        sel_true = y_split

        sel_cm_path = save_dir / f"{file_prefix}_{split_name}_Selective_CM.png"
        if thres_mode == "OCP" and split_name == "Test":
            cc = op.get("conformal_coverage", 0)
            sel_title = (f"{mlabel} {split_name} selective @thr={thr_sel:.3f} (OCP Median-Adaptive)\n"
                         f"Conformal Cov={cc:.1%} (target≥{1-ocp_alpha:.0%})")
        else:
            sel_title = f"{mlabel} {split_name} selective @thr={thr_sel:.3f} ({thr_source})"
        plot_confusion_matrix(sel_true, sel_preds, classes=class_names,
                              save_path=str(sel_cm_path),
                              title=sel_title,
                              meta_mode=meta_mode,
                              is_selective=True,
                              m1_acc=m1_a, m1_prec=m1_p)

        n_sel = int(sel.sum())
        risk = int((y_split[sel] == 0).sum()) / max(n_sel, 1) if n_sel > 0 else 0
        print(f"    {split_name}: acc={metrics['accuracy']:.3f} prec={metrics['precision']:.3f} "
              f"rec={metrics['recall']:.3f} f1={metrics['f1_score']:.3f} "
              f"| selective @thr={thr_sel:.3f} ({'OCP Median-Adaptive' if thres_mode == 'OCP' and split_name == 'Test' else thr_source}): "
              f"cov={op['coverage']:.1%} t-stat={op['t_stat']:.2f} "
              f"μ={op['mean_ret']*100:+.3f}% n={op['selected_count']}")
        sel_dict = {
            "threshold": round(thr_sel, 4),
            "coverage":  round(op["coverage"], 4),
            "risk":      round(risk, 4),
            "precision": round(1 - risk, 4),
            "selected_count": op["selected_count"],
            "threshold_source": thr_source,
            "constraint_satisfied": op["constraint_satisfied"],
            "mean_ret": round(op["mean_ret"], 6),
            "t_stat": round(op["t_stat"], 4),
        }
        if thres_mode == "OCP" and split_name == "Test":
            sel_dict["ocp"] = {
                "alpha": ocp_alpha,
                "conformal_coverage": round(op.get("conformal_coverage", 0), 4),
                "target_coverage": round(1 - ocp_alpha, 4),
                "guarantee_met": op.get("conformal_coverage", 0) >= (1 - ocp_alpha),
                "n_set_1_trade": op.get("n_set_1", 0),
                "n_set_0_dont_trade": op.get("n_set_0", 0),
                "n_set_both_abstain": op.get("n_set_both", 0),
                "n_set_empty_abstain": op.get("n_set_empty", 0),
            }
        results[f"{split_name}_selective"] = sel_dict

    artifacts = {
        "model": model,
        "scaler": scaler,
        "col_indices": col_indices,
        "val_op": val_op,
    }
    return results, artifacts


# ------------------------------------------------------------------------------
# Run analysis for a single direction
# ------------------------------------------------------------------------------
def run_analysis(cache_path: Path, direction: str, mode: str, granularity: str, output_root: Path,
                  train_end: str = None, val_end: str = None, model_name: str = "rf",
                  dataset_override: dict = None, cfg: dict = None, run_top5: bool = True,
                  run_features: bool = True, thres_mode: str = "utility",
                  ocp_alpha: float = 0.10):
    """Run full feature analysis for one direction/granularity.

    Args:
        dataset_override: if provided, use this dict instead of loading cache_path.
                          Used when iterating sub-granularities from a multi cache.
    """
    mlabel = _model_label(model_name)
    class_names = _class_names(direction, mode)
    model_folder = {"rf": "randforest", "xgboost": "xgboost", "autogluon": "autogluon"}[model_name]
    save_dir = output_root / "Kronos" / model_folder / f"{granularity}_{direction}_{mode}"
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"[kronos_tree] Direction: {direction} | Mode: {mode} | Granularity: {granularity}")
    print(f"[kronos_tree] Model: {mlabel} | Classes: {class_names}")
    print(f"[kronos_tree] Cache: {cache_path.name}")
    print(f"[kronos_tree] Output: {save_dir}")
    print(f"{'='*60}\n")

    dataset = dataset_override if dataset_override is not None else torch.load(cache_path, weights_only=False)

    if "eng_features" not in dataset or dataset["eng_features"] is None:
        print("ERROR: Cache does not contain 'eng_features'. Rebuild cache with engineered features enabled.")
        return

    df_all, labels_all = _build_dataframe(dataset)

    # Get train-only subset for feature analysis (no val/test leakage in feature selection)
    if train_end and val_end:
        idx_train, _, _, _ = split_by_global_time(dataset, train_end=train_end, val_end=val_end)
        eng_raw = dataset["eng_features"].numpy() if isinstance(dataset["eng_features"], torch.Tensor) else dataset["eng_features"]
        labels_raw = dataset["labels"].numpy() if isinstance(dataset["labels"], torch.Tensor) else dataset["labels"]
        df_train = pd.DataFrame(eng_raw[idx_train], columns=ENG_FEATURE_NAMES)
        labels_train = labels_raw[idx_train].astype(int)
        print(f"[kronos_tree] Total samples: {len(df_all)} | Train-only for feature analysis: {len(df_train)}")
    else:
        df_train = df_all
        labels_train = labels_all
        print(f"[kronos_tree] Samples: {len(df_all)} (no temporal split dates, using all data)")

    print(f"[kronos_tree] Train: {len(df_train)} (class 0: {(labels_train==0).sum()}, class 1: {(labels_train==1).sum()})")
    print(f"[kronos_tree] Features: {len(df_train.columns)}\n")

    # Drop features that are all-NaN or zero-variance (based on train)
    to_drop = [c for c in df_train.columns if df_train[c].isna().all() or df_train[c].std() == 0]
    if to_drop:
        print(f"[kronos_tree] Dropping zero-variance/all-NaN features: {to_drop}\n")
        df_train = df_train.drop(columns=to_drop)

    # Steps 1-6: feature analysis (correlation, MI, importance, rank aggregation)
    tree_metrics = {}
    tree_top5_metrics = {}
    top_result = {}
    top5 = []

    if run_features:
        plot_correlation_heatmap(df_train, save_dir)
        pb_scores = plot_pointbiserial(df_train, labels_train, class_names, save_dir)
        plot_class_distributions(df_train, labels_train, class_names, save_dir)
        mi_scores = plot_mutual_information(df_train, labels_train, save_dir)

        tree_scores, tree_metrics = plot_tree_importance(
            df_train, labels_train, save_dir, model_name=model_name,
            class_names=class_names, meta_mode=mode,
            model_builder=lambda name, n_samples, ratio: _build_tree_model(name, n_samples, ratio),
            model_labeler=_model_label)

        corr_matrix = df_train.corr()
        top_result = compute_top_features(pb_scores, mi_scores, tree_scores, corr_matrix, save_dir)
        top5 = top_result["top_5_features"]

        # Step 7: tree-model on top-5 only (train only)
        if run_top5:
            df_top5 = df_train[top5]
            _, tree_top5_metrics = plot_tree_importance(
                df_top5, labels_train, save_dir, model_name=model_name,
                step_label="7/9", desc="top-5 only", file_prefix="7_top5_feature_importance",
                class_names=class_names, meta_mode=mode,
                model_builder=lambda name, n_samples, ratio: _build_tree_model(name, n_samples, ratio),
                model_labeler=_model_label)
    else:
        print(f"  [--features=false] Skipping feature analysis (steps 1-7)")

    # Steps 8-9: temporal train/val/test evaluation
    temporal_all = {}
    temporal_top5 = {}
    backtest_all = {}
    backtest_top5 = {}
    if train_end and val_end:
        ((idx_train_t, idx_train_raw), 
         (idx_meta_t, idx_meta_raw), 
         (idx_val_t, idx_val_raw), 
         (idx_test_t, idx_test_raw)) = split_by_global_time(
            dataset, train_end=train_end, val_end=val_end, return_raw=True)
            
        split_indices = (idx_train_t, idx_val_t, idx_test_t)
        split_indices_raw = (idx_train_raw, idx_val_raw, idx_test_raw)
        fee = cfg.get("evaluation", {}).get("fee_per_trade", 0.0) if cfg is not None else 0.0

        fh = int(cfg.get("data", {}).get("load", {}).get("forecast_horizon", 7)) if cfg else 7
        active_features = list(df_train.columns)
        print(f"\n  [8/9] {mlabel} temporal split ({len(active_features)} feats, train<={train_end}, val<={val_end}):")
        temporal_all, artifacts_all = temporal_eval(dataset, active_features, save_dir,
                                                    class_names=class_names, meta_mode=mode,
                                                    split_indices=split_indices,
                                                    direction=direction, fee=fee,
                                                    model_name=model_name,
                                                    desc="all features", file_prefix="8_temporal_all",
                                                    thres_mode=thres_mode, ocp_alpha=ocp_alpha,
                                                    forecast_horizon=fh, split_indices_raw=split_indices_raw)
        if run_top5:
            print(f"  [9/9] {mlabel} temporal split (top-5, train<={train_end}, val<={val_end}):")
            temporal_top5, artifacts_top5 = temporal_eval(dataset, top5, save_dir,
                                                          class_names=class_names, meta_mode=mode,
                                                          split_indices=split_indices,
                                                          direction=direction, fee=fee,
                                                          model_name=model_name,
                                                          desc="top-5 only", file_prefix="9_temporal_top5",
                                                          thres_mode=thres_mode, ocp_alpha=ocp_alpha,
                                                          forecast_horizon=fh, split_indices_raw=split_indices_raw)

        # Steps 10-11: financial backtest (equity curves + ROI reports)
        if cfg is not None:

            print(f"\n  [10/11] {mlabel} financial backtest (all features):")
            backtest_all = run_feature_backtest(
                dataset, split_indices, artifacts_all, cfg, save_dir,
                class_names=class_names, meta_mode=mode, granularity=granularity,
                direction=direction, model_name=model_name, desc="all features",
                file_prefix="10_backtest_all", fee=fee,
                thres_mode=thres_mode, ocp_alpha=ocp_alpha)

            if run_top5:
                print(f"\n  [11/11] {mlabel} financial backtest (top-5):")
                backtest_top5 = run_feature_backtest(
                    dataset, split_indices, artifacts_top5, cfg, save_dir,
                    class_names=class_names, meta_mode=mode, granularity=granularity,
                    direction=direction, model_name=model_name, desc="top-5 only",
                    file_prefix="11_backtest_top5", fee=fee,
                    thres_mode=thres_mode, ocp_alpha=ocp_alpha)

    # Save summary JSON
    summary = {
        "cache": str(cache_path),
        "granularity": granularity,
        "direction": direction,
        "meta_label_mode": mode,
        "model": model_name,
        "n_samples_total": len(df_all),
        "n_samples_train": len(df_train),
        "n_features": len(df_train.columns),
        "class_distribution_train": {class_names[0]: int((labels_train == 0).sum()), class_names[1]: int((labels_train == 1).sum())},
        "features_used": list(df_train.columns),
        "dropped_features": to_drop,
        f"{model_name}_all_features_5fold_cv": tree_metrics,
        f"{model_name}_top5_features_5fold_cv": tree_top5_metrics,
        f"{model_name}_temporal_all_features": temporal_all,
        f"{model_name}_temporal_top5_features": temporal_top5,
        f"{model_name}_backtest_all_features": backtest_all,
        f"{model_name}_backtest_top5_features": backtest_top5,
        **top_result,
    }
    with open(save_dir / "analysis_summary.json", "w") as f:
        json.dump(summary, f, indent=2, cls=NumpyJSONEncoder)

    print(f"\nDone ({direction}). Outputs in: {save_dir}")


def run_unified_analysis(cache_path: Path, multi, direction: str, mode: str, output_root: Path,
                         train_end: str, val_end: str, model_name: str = "rf", cfg: dict = None,
                         thres_mode: str = "utility", ocp_alpha: float = 0.10):
    """Train ONE model on all granularities (with gran-balancing weights), evaluate per-gran."""
    mlabel = _model_label(model_name)
    class_names = _class_names(direction, mode)
    model_folder = {"rf": "randforest", "xgboost": "xgboost", "autogluon": "autogluon"}[model_name]
    save_dir = output_root / "Kronos" / model_folder / f"unified_{direction}_{mode}"
    save_dir.mkdir(parents=True, exist_ok=True)

    fee = cfg.get("evaluation", {}).get("fee_per_trade", 0.0) if cfg else 0.0
    horizon = int(cfg.get("data", {}).get("load", {}).get("forecast_horizon", 7))

    print(f"\n{'='*60}")
    print(f"[kronos_tree] UNIFIED MODEL — Direction: {direction} | Mode: {mode}")
    print(f"[kronos_tree] Granularities: {multi.grans}")
    print(f"[kronos_tree] Model: {mlabel} | Classes: {class_names}")
    print(f"[kronos_tree] Output: {save_dir}")
    print(f"{'='*60}\n")

    # ------------------------------------------------------------------
    # 1. Collect data from all granularities with one-hot encoding
    # ------------------------------------------------------------------
    gran_list = list(multi.grans)
    gran_onehot_names = [f"is_{g}" for g in gran_list]

    all_X_train, all_y_train, all_w_train = [], [], []
    per_gran_data = {}  # for per-gran evaluation later

    for gi, gran in enumerate(gran_list):
        sub = multi.sub[gran]
        eng = sub["eng_features"].numpy() if isinstance(sub["eng_features"], torch.Tensor) else sub["eng_features"]
        labels = sub["labels"].numpy() if isinstance(sub["labels"], torch.Tensor) else sub["labels"]
        returns = sub["returns"].numpy() if isinstance(sub["returns"], torch.Tensor) else sub["returns"]
        asset_ids = sub["asset_ids"].numpy() if isinstance(sub["asset_ids"], torch.Tensor) else sub["asset_ids"]

        idx_train, _, idx_val, idx_test = split_by_global_time(sub, train_end=train_end, val_end=val_end)

        # One-hot for this granularity
        onehot = np.zeros((len(eng), len(gran_list)), dtype=np.float32)
        onehot[:, gi] = 1.0
        X_full = np.hstack([eng, onehot])

        # Train data
        X_tr = X_full[idx_train]
        y_tr = labels[idx_train].astype(int)
        valid_tr = ~np.isnan(y_tr)
        X_tr, y_tr = X_tr[valid_tr], y_tr[valid_tr]

        # Granularity-balancing weight: 1/N_gran so each gran contributes equally
        n_gran_train = len(y_tr)
        w_tr = np.full(n_gran_train, 1.0 / max(n_gran_train, 1), dtype=np.float64)

        all_X_train.append(X_tr)
        all_y_train.append(y_tr)
        all_w_train.append(w_tr)

        # Store per-gran splits for evaluation
        per_gran_data[gran] = {
            "X_full": X_full,
            "labels": labels,
            "returns": returns,
            "asset_ids": asset_ids,
            "dates": sub["dates"],
            "asset_map": sub.get("asset_map", {}),
            "idx_train": idx_train,
            "idx_val": idx_val,
            "idx_test": idx_test,
            "n_train": n_gran_train,
        }

        print(f"  {gran}: train={n_gran_train}  val={len(idx_val)}  test={len(idx_test)}")

    # Concatenate all granularities
    X_train_all = np.vstack(all_X_train)
    y_train_all = np.concatenate(all_y_train)
    w_train_all = np.concatenate(all_w_train)

    print(f"\n  Unified train set: {len(y_train_all)} samples "
          f"(class 0: {(y_train_all==0).sum()}, class 1: {(y_train_all==1).sum()})")

    # Feature names: 23 eng features + one-hot gran columns
    feature_names = list(ENG_FEATURE_NAMES) + gran_onehot_names

    # ------------------------------------------------------------------
    # 2. Drop zero-variance / all-NaN features, impute, scale
    # ------------------------------------------------------------------
    n_eng = len(ENG_FEATURE_NAMES)

    scaler = StandardScaler()
    # Only scale the 23 eng features, not the one-hot columns
    X_train_all[:, :n_eng] = scaler.fit_transform(X_train_all[:, :n_eng])

    # ------------------------------------------------------------------
    # 3. Train unified model
    # ------------------------------------------------------------------
    n_pos = int((y_train_all == 1).sum())
    n_neg = int((y_train_all == 0).sum())
    cw_ratio = n_neg / max(n_pos, 1)

    print(f"\n  Training unified {mlabel} ({len(feature_names)} features)...")
    model = _build_tree_model(model_name, len(y_train_all), cw_ratio,
                              feature_names=feature_names,
                              presets=_AG_PRESETS)
    model.fit(X_train_all, y_train_all, sample_weight=w_train_all)
    print(f"  Training complete.\n")

    if model_name == "autogluon":
        model.leaderboard()
        model.model_info(save_dir)
        model.save_to(save_dir)

    # ------------------------------------------------------------------
    # 4. Per-granularity evaluation: threshold on val, backtest on test
    # ------------------------------------------------------------------
    summary = {
        "cache": str(cache_path),
        "direction": direction,
        "meta_label_mode": mode,
        "model": model_name,
        "mode": "unified",
        "granularities": gran_list,
        "n_train_total": len(y_train_all),
        "feature_names": feature_names,
        "per_gran": {},
    }

    for gran in gran_list:
        gd = per_gran_data[gran]
        gran_dir = save_dir / gran
        gran_dir.mkdir(parents=True, exist_ok=True)

        print(f"  {'─'*50}")
        print(f"  Evaluating {gran}...")

        # Prepare val/test data
        X_val_raw = gd["X_full"][gd["idx_val"]].copy()
        y_val = gd["labels"][gd["idx_val"]].astype(int)
        val_returns = gd["returns"][gd["idx_val"]].copy()

        X_test_raw = gd["X_full"][gd["idx_test"]].copy()
        y_test = gd["labels"][gd["idx_test"]].astype(int)
        test_returns = gd["returns"][gd["idx_test"]].copy()
        test_asset_ids = gd["asset_ids"][gd["idx_test"]]
        val_dates_raw  = [gd["dates"][i] for i in gd["idx_val"]]
        test_dates_raw = [gd["dates"][i] for i in gd["idx_test"]]
        test_assets = [gd["asset_map"].get(int(aid), str(aid)) for aid in test_asset_ids]

        # Direction-aware returns
        if direction.lower() == "down":
            val_returns = -val_returns
            test_returns = -test_returns

        # Impute + scale (eng features only, one-hot stays as-is)
        for X in [X_val_raw, X_test_raw]:
            X[:, :n_eng] = scaler.transform(X[:, :n_eng])

        # Predict
        val_preds = model.predict(X_val_raw)
        val_probs = model.predict_proba(X_val_raw)[:, 1]
        test_preds = model.predict(X_test_raw)
        test_probs = model.predict_proba(X_test_raw)[:, 1]

        # ── Pre-selective confusion matrices + metrics (Val & Test) ──
        presel_metrics = {}
        for split_name, y_split, preds, probs in [("Val", y_val, val_preds, val_probs),
                                                    ("Test", y_test, test_preds, test_probs)]:
            cm_path = gran_dir / f"{split_name}_CM.png"
            plot_confusion_matrix(y_split, preds, classes=class_names, save_path=str(cm_path),
                                  title=f"Unified {mlabel} {gran} {split_name}", meta_mode=mode)
            prec_val = float(precision_score(y_split, preds, zero_division=0))
            n_pred_pos = int((preds == 1).sum())
            presel_metrics[split_name] = {
                "accuracy": round(float(accuracy_score(y_split, preds)), 4),
                "precision": round(prec_val, 4),
                "recall": round(float(recall_score(y_split, preds, zero_division=0)), 4),
                "f1_score": round(float(f1_score(y_split, preds, zero_division=0)), 4),
                "coverage": round(n_pred_pos / len(y_split), 4) if len(y_split) > 0 else 0,
                "risk": round(1 - prec_val, 4),
                "baseline": round(int((y_split == 1).sum()) / len(y_split), 4) if len(y_split) > 0 else 0,
            }

        # Threshold optimization on val (always computed)
        val_op = _find_best_utility_threshold(val_probs, val_returns, fee=fee)
        sel_val = val_probs >= val_op["threshold"]
        n_sel_val = int(sel_val.sum())
        err_val = int((y_val[sel_val] == 0).sum()) if n_sel_val > 0 else 0
        val_op["risk"] = err_val / max(n_sel_val, 1)

        threshold = val_op["threshold"]

        # OCP: run SAOCP warm-up on val, adapt on test (delayed feedback)
        if thres_mode == "OCP":
            test_s_hats, test_approved_ocp, val_s_hats, conf_stats = _run_saocp_online(
                val_probs, y_val, test_probs, y_test, alpha=ocp_alpha,
                test_dates=test_dates_raw, forecast_horizon=horizon,
                val_dates=val_dates_raw)
            ocp_op = _ocp_threshold_to_op(test_probs, y_test, test_returns,
                                          test_approved_ocp, test_s_hats, fee,
                                          conformal_stats=conf_stats)
            cc = conf_stats["conformal_coverage"]
            print(f"    OCP ({gran}): α={ocp_alpha}, median τ={ocp_op['threshold']:.3f}, "
                  f"cov={ocp_op['coverage']:.1%}, μ={ocp_op['mean_ret']*100:+.3f}% | "
                  f"Conformal cov={cc:.1%} (target≥{1-ocp_alpha:.0%}) | "
                  f"Sets: {{1}}={conf_stats['n_set_1']} {{0}}={conf_stats['n_set_0']} "
                  f"{{0,1}}={conf_stats['n_set_both']} {{}}={conf_stats['n_set_empty']}")

        # ── Post-selective confusion matrices (Val & Test) ──
        for split_name, y_split, probs in [("Val", y_val, val_probs),
                                            ("Test", y_test, test_probs)]:
            if split_name == "Test" and thres_mode == "OCP":
                sel = test_approved_ocp
                thr_source = "OCP-SAOCP"
            else:
                sel = probs >= threshold
                thr_source = "Utility-Opt" if split_name == "Val" else "Val-Utility"
            sel_true = y_split
            sel_preds = sel.astype(int)
            sel_cm_path = gran_dir / f"{split_name}_Selective_CM.png"
            thr_display = ocp_op["threshold"] if (split_name == "Test" and thres_mode == "OCP") else threshold
            if thres_mode == "OCP" and split_name == "Test":
                cc = ocp_op.get("conformal_coverage", 0)
                sel_title = (f"Unified {mlabel} {gran} {split_name} selective @thr={thr_display:.3f} (OCP Median-Adaptive)\n"
                             f"Conformal Cov={cc:.1%} (target≥{1-ocp_alpha:.0%})")
            else:
                sel_title = f"Unified {mlabel} {gran} {split_name} selective @thr={thr_display:.3f} ({thr_source})"
            plot_confusion_matrix(sel_true, sel_preds, classes=class_names,
                                  save_path=str(sel_cm_path),
                                  title=sel_title,
                                  meta_mode=mode,
                                  is_selective=True)

        # Apply threshold on test
        if thres_mode == "OCP":
            m2_approved = test_approved_ocp
        else:
            m2_approved = test_probs >= threshold
        net_returns = test_returns - fee

        # Trade DataFrame
        df_trades = pd.DataFrame({
            "date": pd.to_datetime(test_dates_raw),
            "asset": test_assets,
            "return": net_returns,
            "label": y_test,
            "m2_approved": m2_approved,
            "m2_prob": test_probs,
        })
        df_trades = df_trades.dropna(subset=["return"]).reset_index(drop=True)
        m2_approved = df_trades["m2_approved"].values
        test_probs = df_trades["m2_prob"].values
        test_returns = df_trades["return"].values + fee
        y_test = df_trades["label"].values

        # Save trades CSV for diagnostics
        trades_dump_u = df_trades.copy()
        trades_dump_u["direction"] = direction
        trades_dump_u["return_pct"] = trades_dump_u["return"] * 100
        trades_dump_u.to_csv(gran_dir / "backtest_trades.csv",
                             index=False, float_format="%.6f")

        # Save OCP diagnostics npz (thresholds + val probs for re-run)
        if thres_mode == "OCP":
            np.savez_compressed(
                gran_dir / "ocp_diagnostics.npz",
                test_s_hats=test_s_hats,
                val_s_hats=val_s_hats,
                val_probs=val_probs,
                val_labels=y_val.astype(int),
                alpha=np.array([ocp_alpha]),
            )

        m2_df = df_trades[df_trades["m2_approved"]]
        test_start = df_trades["date"].min()
        test_end = df_trades["date"].max()

        # OCP threshold evolution plot
        if thres_mode == "OCP":
            fig_thr, ax_thr = plt.subplots(figsize=(10, 4), facecolor="white")
            ax_thr.set_facecolor("#FAFAFA")
            eff_tau = np.maximum(test_s_hats, 1.0 - test_s_hats)
            ax_thr.plot(eff_tau, color="#8B008B", linewidth=0.8, alpha=0.9, label="τ_t (OCP)")
            ax_thr.axhline(y=threshold, color="#34495E", linestyle="--",
                           linewidth=1.2, alpha=0.7, label=f"τ Utility = {threshold:.3f}")
            ax_thr.axhline(y=0.5, color="#BDC3C7", linestyle=":", linewidth=0.8, alpha=0.6)
            ax_thr.set_xlabel("Test sample index", fontsize=10)
            ax_thr.set_ylabel("Threshold τ_t", fontsize=10)
            cc = ocp_op.get("conformal_coverage", 0)
            n1 = ocp_op.get("n_set_1", 0)
            n0 = ocp_op.get("n_set_0", 0)
            nb = ocp_op.get("n_set_both", 0)
            ne = ocp_op.get("n_set_empty", 0)
            ax_thr.set_title(
                f"OCP Threshold Evolution  |  Unified {gran}  |  {mlabel}  (α={ocp_alpha})\n"
                f"Conformal Cov={cc:.1%} (target≥{1-ocp_alpha:.0%})  |  "
                f"{{1}}={n1}  {{0}}={n0}  {{0,1}}={nb}  {{}}={ne}",
                fontsize=11, fontweight="bold", color="#2C3E50")
            ax_thr.legend(fontsize=8, loc="upper right")
            ax_thr.set_ylim(0.4, 1.0)
            ax_thr.grid(True, alpha=0.3)
            fig_thr.tight_layout()
            thr_evo_path = gran_dir / "OCP_Threshold_Evolution.png"
            fig_thr.savefig(str(thr_evo_path), dpi=200, facecolor="white")
            plt.close(fig_thr)

        # Stats
        n_total = len(y_test)
        n_approved = int(m2_approved.sum())
        n_m2_good = int(((m2_approved) & (y_test == 1)).sum())
        m2_wr = n_m2_good / n_approved * 100 if n_approved > 0 else 0
        m1_good = int((y_test == 1).sum())
        m1_wr = m1_good / n_total * 100 if n_total > 0 else 0
        execution_rate = n_approved / n_total * 100 if n_total > 0 else 0

        # Test selective metrics (use OCP mask when applicable)
        if thres_mode == "OCP":
            sel_test = m2_approved
        else:
            sel_test = test_probs >= threshold
        n_sel_test = int(sel_test.sum())
        err_test = int((y_test[sel_test] == 0).sum()) if n_sel_test > 0 else 0
        net_rets_test = test_returns[sel_test] - fee if n_sel_test > 0 else np.array([0.0])
        mu_test = float(np.nanmean(net_rets_test))
        sigma_test = float(np.nanstd(net_rets_test, ddof=1)) if n_sel_test > 1 else 0.0
        t_test = mu_test / sigma_test * np.sqrt(n_sel_test) if sigma_test > 0 else 0.0

        # Equity curves
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

        m1_equity, _ = _build_spread_equity(df_trades, full_idx, horizon)
        m2_equity, _ = _build_spread_equity(m2_df, full_idx, horizon)

        # Sharpe: horizon-length non-overlapping returns from equity curve
        ann_bar = _annualization_factor(gran)
        ann_horizon = np.sqrt(ann_bar ** 2 / horizon)

        m2_name = f"M2 {mlabel} unified"
        m1_name = "M1 Kronos (all trades)"
        bh_name = "Buy & Hold"

        strats = {}
        for name, eq, tdf in [(m2_name, m2_equity, m2_df), (m1_name, m1_equity, df_trades)]:
            h_rets = _equity_horizon_returns(eq, horizon) if len(eq) > horizon else np.array([])
            strats[name] = {
                "total_ret": (eq.iloc[-1] - 1) * 100 if len(eq) > 0 else 0,
                "mdd": _calc_drawdown(eq.values) * 100 if len(eq) > 0 else 0,
                "sharpe": _calc_sharpe(h_rets, ann_horizon),
            }
        if has_bh:
            bh_h_rets = _equity_horizon_returns(bh_equity, horizon) if len(bh_equity) > horizon else np.array([])
            strats[bh_name] = {
                "total_ret": (bh_equity.iloc[-1] - 1) * 100,
                "mdd": _calc_drawdown(bh_equity.values) * 100,
                "sharpe": _calc_sharpe(bh_h_rets, ann_horizon),
            }

        # ── Plot equity curve ──
        if thres_mode == "OCP":
            ocp_median_tau = float(np.median(np.maximum(test_s_hats, 1.0 - test_s_hats)))
            constraint_tag = "OCP Adaptive"
            thr_display_eq = ocp_median_tau
        else:
            constraint_tag = "Utility-Opt" if val_op["constraint_satisfied"] else "fallback"
            thr_display_eq = threshold
        fee_tag = f" fee={fee*100:.2f}%" if fee > 0 else ""
        direction_label = direction.upper()

        fig, ax = plt.subplots(figsize=(14, 6))
        ax.plot((m2_equity - 1) * 100,
                label=f"{m2_name} (SR: {strats[m2_name]['sharpe']:.2f}, Exec: {execution_rate:.1f}%)",
                color="green", linewidth=3.0)
        ax.plot((m1_equity - 1) * 100,
                label=f"{m1_name} (SR: {strats[m1_name]['sharpe']:.2f})",
                color="blue", alpha=0.6, linewidth=2.0)
        if has_bh:
            ax.plot((bh_equity - 1) * 100,
                    label=f"{bh_name} (SR: {strats[bh_name]['sharpe']:.2f})",
                    color="gray", linestyle="--", linewidth=1.5)
        ax.axhline(0, color="black", linewidth=0.5)
        ax.set_title(f"UNIFIED {mlabel} {gran.upper()}+{direction_label}+{mode.upper()} "
                     f"thr={thr_display_eq:.3f} ({constraint_tag}){fee_tag}", fontsize=12)
        ax.set_ylabel("Cumulative Return (%)")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(gran_dir / f"backtest_equity_curve.png", dpi=200)
        plt.close(fig)

        # ── ROI report ──
        avg_ret_approved = m2_df["return"].mean() * 100 if len(m2_df) > 0 else 0
        avg_ret_rejected = df_trades[~df_trades["m2_approved"]]["return"].mean() * 100 if n_total > n_approved else 0
        edge = avg_ret_approved - avg_ret_rejected

        roi_lines = [
            "=" * 60,
            f"FINANCIAL BACKTEST: UNIFIED {mlabel} {gran.upper()} {direction_label} {mode.upper()}",
            f"Period: {test_start.strftime('%Y-%m-%d')} to {test_end.strftime('%Y-%m-%d')}",
            f"Threshold: {thr_display_eq:.4f} ({constraint_tag}) | Fee: {fee*100:.3f}%",
            "=" * 60,
            f"Total Test Trades:     {n_total}",
            f"M1 Baseline Win-Rate:  {m1_wr:.1f}% ({m1_good}/{n_total})",
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
        for sname in [m2_name, m1_name, bh_name]:
            if sname in strats:
                s = strats[sname]
                roi_lines.append(f"{sname:<27} {s['total_ret']:>+9.2f}% {s['mdd']:>+7.2f}% {s['sharpe']:>7.2f}")
        roi_lines.append("=" * 60)

        roi_text = "\n".join(roi_lines)
        with open(gran_dir / "backtest_ROI.txt", "w") as f:
            f.write(roi_text)
        print(roi_text)

        # Store per-gran summary
        summary["per_gran"][gran] = {
            "n_train": gd["n_train"],
            "n_val": len(gd["idx_val"]),
            "n_test": len(gd["idx_test"]),
            "threshold": threshold,
            "constraint_satisfied": val_op["constraint_satisfied"],
            "Val": presel_metrics.get("Val", {}),
            "Test": presel_metrics.get("Test", {}),
            "val_selective": {
                "coverage": val_op["coverage"],
                "risk": val_op["risk"],
                "precision": 1 - val_op["risk"],
                "mean_ret": val_op["mean_ret"],
                "t_stat": val_op["t_stat"],
                "selected_count": val_op["selected_count"],
            },
            "test_selective": {
                "coverage": n_sel_test / n_total if n_total > 0 else 0,
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
                }} if thres_mode == "OCP" else {}),
            },
            "backtest": {
                "execution_rate": execution_rate,
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

    # ------------------------------------------------------------------
    # 5. Save unified summary
    # ------------------------------------------------------------------
    with open(save_dir / "unified_summary.json", "w") as f:
        json.dump(summary, f, indent=2, cls=NumpyJSONEncoder)

    # Save model artifacts
    artifacts = {
        "model": model,
        "scaler": scaler,
        "feature_names": feature_names,
        "gran_list": gran_list,
        "thresholds": {g: summary["per_gran"][g]["threshold"] for g in gran_list},
        "direction": direction,
        "mode": mode,
        "train_end": train_end,
        "val_end": val_end,
    }
    with open(save_dir / "unified_model.pkl", "wb") as f:
        pickle.dump(artifacts, f)
    print(f"\n  Model saved to {save_dir / 'unified_model.pkl'}")

    print(f"\nDone (unified {direction}). Outputs in: {save_dir}")


def main():
    # ┏━━━━━━━━━━ Parse Arguments ━━━━━━━━━━┓
    parser = argparse.ArgumentParser(description="Kronos Tree — M2 Meta-Labeling with RF/XGBoost/AutoGluon")
    parser.add_argument("--cache",         type=str, default=None, help="Explicit path to dataset cache .pt")
    parser.add_argument("--config",        type=str, default="config.yaml", help="Path to config.yaml")
    
    # ┏━━━━━━━━━━ Model Selection ━━━━━━━━━━┓
    parser.add_argument("--model",         type=str, default="rf", choices=MODEL_CHOICES, help="Classifier: 'rf' (Random Forest), 'xgboost' (XGBoost), or 'autogluon' (AutoGluon)")
    parser.add_argument("--ag-time-limit", type=int, default=300, help="AutoGluon time limit per fit in seconds (default: 300)")
    parser.add_argument("--ag-presets",    type=str, default="best_quality", choices=["best_quality", "high_quality", "good_quality", "medium_quality"], help="AutoGluon model preset (default: best_quality)")

    # ┏━━━━━━━━━━ Mode Selection [all granularities vs per-granularity] ━━━━━━━━━━┓
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument("--per-gran", action="store_true", help="Per-granularity models from a multi-gran cache (one model per gran)")
    mode_group.add_argument("--all-grans", action="store_true", help="Unified model trained on all granularities, evaluated per-gran")
    
    # ┏━━━━━━━━━━ Analysis Comparison between Models ━━━━━━━━━━┓
    mode_group.add_argument("--comparison", nargs=2, metavar=("PER_GRAN_DIR", "UNIFIED_DIR"), help="Build comparison table from existing per-gran and unified result dirs")
    mode_group.add_argument("--paradigm-comparison", nargs="+", metavar="DIR", help="Cross-paradigm comparison: pass 2+ result dirs (e.g. autogluon_7_fees randforest_7_fees randforest_OCP)")

    # ┏━━━━━━━━━━ Threshold Selection [Online Conformal Prediction] ━━━━━━━━━━┓
    parser.add_argument("--thres", type=str, default="utility", choices=["utility", "OCP"], help="Threshold selection: 'utility' (financial utility on val) or 'OCP' (SAOCP online conformal)")
    parser.add_argument("--ocp-alpha", type=float, default=0.10, help="OCP target miscoverage rate (default: 0.10 → 90%% coverage target)")
    
    # ┏━━━━━━━━━━ Feature Analysis ━━━━━━━━━━┓
    parser.add_argument("--top5", type=str, default="true", choices=["true", "false"], help="Whether to run top-5 feature analysis/backtest (default: true)")
    parser.add_argument("--features", type=str, default="true", choices=["true", "false"], help="Whether to run feature analysis (correlation, MI, importance, rank agg). Default: true")
    args = parser.parse_args()
    args.top5 = args.top5.lower() == "true"
    args.features = args.features.lower() == "true"
    if args.top5 and not args.features:
        parser.error("--top5 true requires --features true (top5 depends on feature ranking)")

    global _AG_TIME_LIMIT, _AG_PRESETS
    _AG_TIME_LIMIT = args.ag_time_limit
    _AG_PRESETS = args.ag_presets

    # ┏━━━━━━━━━━ Load Config ━━━━━━━━━━┓
    cfg         = _load_config(args.config)
    mode        = cfg["data"]["load"].get("meta_label_mode", "og").lower()
    output_root = Path(cfg["paths"]["output_root"])
    split_cfg   = cfg["data"]["split"]
    train_end   = split_cfg.get("train_end")
    val_end     = split_cfg.get("val_end")

    # ┏━━━━━━━━━━ Analysis Comparison between Models ━━━━━━━━━━┓
    if args.paradigm_comparison:
        run_paradigm_comparison(args.paradigm_comparison)
        print(f"\nParadigm comparison complete.")
        return

    # ┏━━━━━━━━━━ Analysis Comparison between all-grans vs per-gran ━━━━━━━━━━┓
    if args.comparison:
        per_gran_dir = Path(args.comparison[0])
        unified_dir = Path(args.comparison[1])
        run_comparison(per_gran_dir, unified_dir)
        print(f"\nComparison complete.")
        return

    # ┏━━━━━━━━━━ Per-granularity models ━━━━━━━━━━┓
    # One independent model per granularity
    if args.per_gran:
        # ┏━━━━━━━━━━ Load Cache ━━━━━━━━━━┓
        if args.cache:
            cache_path = Path(args.cache)
            if not cache_path.exists():
                raise FileNotFoundError(f"Cache not found: {cache_path}")
        else:
            # ┏━━━━━━━━━━ Auto-build multi-gran cache from config ━━━━━━━━━━┓
            cache_path, _ = _build_cache_from_config(cfg)
        print(f"[kronos_tree] Loading multi-gran cache: {cache_path.name}")
        multi = _load_multi_cache(cache_path)
        direction = _infer_direction(cache_path)

        # ┏━━━━━━━━━━ Run Analysis for each granularity ━━━━━━━━━━┓
        for gran in multi.grans:
            sub = multi.sub[gran]
            run_analysis(cache_path, 
                         direction, 
                         mode, 
                         gran, 
                         output_root,
                         train_end        = train_end, 
                         val_end          = val_end,
                         model_name       = args.model, 
                         dataset_override = sub, 
                         cfg              = cfg,
                         run_top5         = args.top5, 
                         run_features     = args.features,
                         thres_mode       = args.thres, 
                         ocp_alpha        = args.ocp_alpha)

    # ┏━━━━━━━━━━ Unified model ━━━━━━━━━━┓
    # One model trained on all granularities, evaluated per-gran
    elif args.all_grans:
        # ┏━━━━━━━━━━ Load Cache ━━━━━━━━━━┓
        if args.cache:
            cache_path = Path(args.cache)
            if not cache_path.exists():
                raise FileNotFoundError(f"Cache not found: {cache_path}")
        else:
            # ┏━━━━━━━━━━ Auto-build multi-gran cache from config ━━━━━━━━━━┓
            cache_path, _ = _build_cache_from_config(cfg)
        
        # ┏━━━━━━━━━━ Load Cache ━━━━━━━━━━┓
        print(f"[kronos_tree] Loading multi-gran cache for UNIFIED model: {cache_path.name}")
        multi = _load_multi_cache(cache_path)
        direction = _infer_direction(cache_path)

        # ┏━━━━━━━━━━ Run Analysis for each granularity ━━━━━━━━━━┓
        run_unified_analysis(cache_path, 
                             multi, 
                             direction, 
                             mode, 
                             output_root,
                             train_end  = train_end, 
                             val_end    = val_end,
                             model_name = args.model, 
                             cfg        = cfg,
                             thres_mode = args.thres, 
                             ocp_alpha  = args.ocp_alpha)

    else:
        # ┏━━━━━━━━━━ Single granularity (auto-detect from config) ━━━━━━━━━━┓
        granularity = cfg["data"]["load"]["granularity"]
        direction_caches = _resolve_caches(cfg, args.cache)

        for direction, cache_path in sorted(direction_caches.items()):
            run_analysis(cache_path, 
                         direction, 
                         mode, 
                         granularity, 
                         output_root,
                         train_end    = train_end, 
                         val_end      = val_end,
                         model_name   = args.model, 
                         cfg          = cfg,
                         run_top5     = args.top5, 
                         run_features = args.features,
                         thres_mode   = args.thres, 
                         ocp_alpha    = args.ocp_alpha)

    print(f"\nAll analyses complete.")


if __name__ == "__main__":
    main()
