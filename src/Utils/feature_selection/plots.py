"""Feature analysis and plotting helpers extracted from m2_pipeline.py.

This module only contains feature-level diagnostics and the final
risk-coverage plot. Meta-label diagnostics and M2 result-aggregation
plots live in ``Utils.analysis``. CPCV/edge plots live in
``Utils.edge.plots``.
"""

import warnings
import re
import glob
import os
import json
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.scale as mscale
import matplotlib.transforms as mtransforms
import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path
from scipy import stats
from typing import Dict, List, Optional, Any, Union, Tuple

from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import (accuracy_score, f1_score, precision_score, recall_score,
                             precision_recall_fscore_support, confusion_matrix,
                             ConfusionMatrixDisplay, fbeta_score, matthews_corrcoef)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

from Utils.utils import model_label
from Utils.data import _get_from_dataset, get_dynamic_ret_limits, MultiGranDataset


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Rank Aggregation → Top-K feature selection
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CORR_THRESHOLD = 0.85
TOP_K = 5

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MDA / SHAP / LIME  FEATURE RANKING
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
_RNG = np.random.default_rng(42)


__all__ = [
    "plot_correlation_heatmap",
    "plot_pointbiserial",
    "plot_mutual_information",
    "plot_confusion_matrix",
    "plot_temporal_risk_coverage_curve_final",
    "plot_performance_over_n_features",
]


# ┏━━━━━━━━━━ Correlation heatmap matrix of features ━━━━━━━━━━┓
def plot_correlation_heatmap(df: pd.DataFrame, save_dir: Path):
    """Plot correlation heatmap of features."""
    # ┏━━━━━━━━━━ Correlation Matrix ━━━━━━━━━━┓
    corr = df.corr()
    n = len(corr)
    
    # ┏━━━━━━━━━━ Plot Heatmap ━━━━━━━━━━┓
    fig, ax = plt.subplots(figsize=(max(10, n * 0.55), max(8, n * 0.45)))
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    sns.heatmap(corr,
                mask       = mask,
                annot      = True,
                fmt        = ".2f",
                cmap       = "RdBu_r",
                center     = 0,
                vmin       = -1,
                vmax       = 1,
                square     = True,
                linewidths = 0.5,
                cbar_kws   = {"shrink": 0.8},
                ax         = ax,
                annot_kws  = {"size": 7})
    ax.set_title("Engineered Features — Correlation Matrix", fontsize=13, pad=12)
    plt.tight_layout()

    # ┏━━━━━━━━━━ Save Heatmap ━━━━━━━━━━┓
    fig.savefig(save_dir / "1_feature_correlation_heatmap.png", dpi=200)
    plt.close(fig)
    print("  [1/9] Correlation heatmap saved")



# ┏━━━━━━━━━━ Point-Biserial Correlation ━━━━━━━━━━┓
def plot_pointbiserial(df: pd.DataFrame, labels: np.ndarray, class_names: list, save_dir: Path) -> dict:
    """Returns {feature: abs_correlation}."""
    
    # ┏━━━━━━━━━━ Calculate Correlations ━━━━━━━━━━┓
    results = []
    for col in df.columns:
        vals = df[col].values
        valid = ~np.isnan(vals)
        if valid.sum() < 10:
            results.append({"feature": col, "correlation": 0.0, "p_value": 1.0})
            continue
        r, p = stats.pointbiserialr(labels[valid], vals[valid])
        results.append({"feature": col, "correlation": r, "p_value": p})

    # ┏━━━━━━━━━━ Sort Results ━━━━━━━━━━┓
    res_df = pd.DataFrame(results).sort_values("correlation", key=abs, ascending=True)

    # ┏━━━━━━━━━━ Plot Bar Chart ━━━━━━━━━━┓
    fig, ax = plt.subplots(figsize=(8, max(5, len(res_df) * 0.35)))
    colors = ["#e74c3c" if r < 0 else "#2ecc71" for r in res_df["correlation"]]
    bars = ax.barh(res_df["feature"],
                   res_df["correlation"],
                   color     = colors,
                   edgecolor = "k",
                   linewidth = 0.4)

    # ┏━━━━━━━━━━ Add Significance Markers ━━━━━━━━━━┓
    for bar, (_, row) in zip(bars, res_df.iterrows()):
        if row["p_value"] < 0.001:
            marker = "***"
        elif row["p_value"] < 0.01:
            marker = "**"
        elif row["p_value"] < 0.05:
            marker = "*"
        else:
            marker = ""
        if marker:
            x = bar.get_width()
            ax.text(x + 0.005 * np.sign(x),
                    bar.get_y() + bar.get_height() / 2,
                    marker,
                    va = "center",
                    ha = "left" if x >= 0 else "right",
                    fontsize = 8)

    # ┏━━━━━━━━━━ Add Zero Line ━━━━━━━━━━┓
    ax.axvline(0, color="k", linewidth=0.8)
    ax.set_xlabel("Point-Biserial Correlation")
    ax.set_title(f"Feature ↔ Target ({class_names[1]}) Correlation", fontsize=12, pad=10)
    plt.tight_layout()

    # ┏━━━━━━━━━━ Save Bar Chart ━━━━━━━━━━┓
    fig.savefig(save_dir / "2_pointbiserial_correlation.png", dpi=200)
    plt.close(fig)

    # ┏━━━━━━━━━━ Save Results ━━━━━━━━━━┓
    res_df.to_csv(save_dir / "2_pointbiserial_correlation.csv", index=False, float_format="%.6f")
    print("  [2/9] Point-biserial correlation saved")

    return {row["feature"]: abs(row["correlation"]) for _, row in res_df.iterrows()}



# ┏━━━━━━━━━━ Mutual Information ━━━━━━━━━━┓
def plot_mutual_information(df: pd.DataFrame, labels: np.ndarray, save_dir: Path) -> dict:
    """Returns {feature: mi_score}."""
    # ┏━━━━━━━━━━ Copy Data ━━━━━━━━━━┓
    X = df.values.copy()

    # ┏━━━━━━━━━━ Calculate Mutual Information ━━━━━━━━━━┓
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mi = mutual_info_classif(X, labels, discrete_features=False, random_state=42, n_neighbors=5)

    # ┏━━━━━━━━━━ Sort Results ━━━━━━━━━━┓
    mi_df = pd.DataFrame({"feature": df.columns, "mutual_info": mi}).sort_values("mutual_info", ascending=True)

    # ┏━━━━━━━━━━ Plot Bar Chart ━━━━━━━━━━┓
    fig, ax = plt.subplots(figsize=(8, max(5, len(mi_df) * 0.35)))
    ax.barh(mi_df["feature"], mi_df["mutual_info"], color="#9b59b6", edgecolor="k", linewidth=0.4)
    ax.set_xlabel("Mutual Information (nats)")
    ax.set_title("Feature → Target: Mutual Information", fontsize=12, pad=10)
    plt.tight_layout()

    # ┏━━━━━━━━━━ Save Bar Chart ━━━━━━━━━━┓
    fig.savefig(save_dir / "4_mutual_information.png", dpi=200)
    plt.close(fig)

    # ┏━━━━━━━━━━ Save Results ━━━━━━━━━━┓
    mi_df.to_csv(save_dir / "4_mutual_information.csv", index=False, float_format="%.6f")
    print("  [4/9] Mutual information saved")

    return {row["feature"]: row["mutual_info"] for _, row in mi_df.iterrows()}



# ┏━━━━━━━━━━ Confusion Matrix ━━━━━━━━━━┓
def plot_confusion_matrix(targets: np.ndarray,
                          preds: np.ndarray,
                          classes: list,
                          save_path: str,
                          title: str = "Confusion Matrix",
                          meta_mode: str = "og",
                          baseline_targets: Optional[np.ndarray] = None,
                          m1_prec: Optional[float] = None,
                          m1_acc: Optional[float] = None,
                          is_selective: bool = False):
    """
    Plot and save confusion matrix with win-rate and baseline annotations.
    
    Args:
        targets: Ground truth labels
        preds: Predicted labels
        classes: List of class names
        save_path: Path to save the plot
        title: Plot title
        meta_mode: 'fp', 'tp', or 'og' — determines the "interested" class
        baseline_targets: Full dataset labels for baseline win-rate
        m1_prec: Optional explicit M1 precision
        m1_acc: Optional explicit M1 accuracy
        is_selective: Flag indicating this is a selective confusion matrix
    """
    cm = confusion_matrix(targets, preds, labels=list(range(len(classes))))
    
    # ┏━━━━━━━━━━ Calculate macro metrics ━━━━━━━━━━┓
    unique_labels = np.unique(targets)
    is_multiclass = len(unique_labels) > 2 or (len(unique_labels) == 2 and not np.array_equal(sorted(unique_labels), [0, 1]))
    avg_mode = "weighted" if is_multiclass else "macro"
    precision_m, recall_m, f1_m, _ = precision_recall_fscore_support(targets, preds, average=avg_mode, zero_division=0)
    metrics = {
        "accuracy":  accuracy_score(targets, preds),
        "precision": precision_m,
        "recall":    recall_m,
        "f1":        f1_m,
        "fbeta":     fbeta_score(targets, preds, beta=0.3, average=avg_mode, zero_division=0),
        "mcc":       matthews_corrcoef(targets, preds),
    }

    # ┏━━━━━━━━━━ Win-Rate (precision of the interested class) ━━━━━━━━━━┓
    interested_class = 0 if meta_mode == 'fp' else 1
    all_labels = list(range(len(classes)))
    per_class_prec = precision_recall_fscore_support(targets, preds, labels=all_labels, average=None, zero_division=0)[0]
    win_rate = per_class_prec[interested_class]

    # ┏━━━━━━━━━━ Baseline win-rate ━━━━━━━━━━┓
    base = baseline_targets if baseline_targets is not None else targets
    baseline_wr = (base == interested_class).sum() / len(base) if len(base) > 0 else 0.0

    # ┏━━━━━━━━━━ Coverage ━━━━━━━━━━┓
    is_selective_plot = is_selective or (baseline_targets is not None)
    if baseline_targets is not None and len(targets) < len(baseline_targets):
        coverage = len(targets) / len(baseline_targets) if len(baseline_targets) > 0 else 0.0
    else:
        coverage = (preds == interested_class).sum() / len(preds) if len(preds) > 0 else 0.0

    # ┏━━━━━━━━━━ Risk & Title ━━━━━━━━━━┓
    risk = 1.0 - win_rate
    
    val_m1_prec = m1_prec if m1_prec is not None else baseline_wr
    val_m1_acc = m1_acc if m1_acc is not None else baseline_wr

    if is_selective_plot:
        title_with_metrics = (f"{title}\n"
                              f"M2 Acc: {metrics['accuracy']:.3f} | WinRate: {win_rate:.3f} | Risk: {risk:.3f} | Cov: {coverage:.1%}\n"
                              f"M1 Acc: {val_m1_acc:.3f} | Prec: {val_m1_prec:.3f}")
    else:
        title_with_metrics = (f"{title}\n"
                              f"Acc: {metrics['accuracy']:.3f} | Prec: {metrics['precision']:.3f} | Rec: {metrics['recall']:.3f} | F1: {metrics['f1']:.3f} | Cov: {coverage:.1%}\n"
                              f"Risk: {risk:.3f} | WinRate: {win_rate:.3f} | M1 Prec: {val_m1_prec:.3f} | M1 Acc: {val_m1_acc:.3f}")

    # ┏━━━━━━━━━━ Plot Confusion Matrix ━━━━━━━━━━┓
    plt.figure(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    # Match color to split: "Val"/"Validation" as split indicator → Oranges, otherwise Blues
    # Avoid false positives like "val-tuned" (threshold source)
    is_val = bool(re.search(r'(?:^|[\s(])val(?:idation)?(?:[\s_)]|$)', title, re.IGNORECASE))
    cmap = 'Oranges' if is_val else 'Blues'
    disp.plot(cmap=cmap, values_format='d', ax=plt.gca())
    plt.title(title_with_metrics)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()



# ┏━━━━━━━━━━ Piecewise-Linear Axis Scale (used by risk-coverage plot) ━━━━━━━━━━┓
class _PiecewiseLinearScale(mscale.ScaleBase):
    name = 'piecewise_linear'

    def __init__(self, axis, **kwargs):
        super().__init__(axis)
        self.x_nodes = kwargs.pop('x_nodes', [0.0, 0.5, 1.0])
        self.y_nodes = kwargs.pop('y_nodes', [0.0, 0.5, 1.0])

    def get_transform(self):
        return self._PiecewiseTransform(self.x_nodes, self.y_nodes)

    def set_default_locators_and_formatters(self, axis):
        pass

    class _PiecewiseTransform(mtransforms.Transform):
        input_dims = output_dims = 1
        is_separable = has_inverse = True

        def __init__(self, x_nodes, y_nodes):
            super().__init__()
            self.x_nodes = np.array(x_nodes, dtype=float)
            self.y_nodes = np.array(y_nodes, dtype=float)

        def transform_non_affine(self, a):
            return np.interp(a, self.x_nodes, self.y_nodes)

        def inverted(self):
            return _PiecewiseLinearScale._InvertedTransform(self.x_nodes, self.y_nodes)

    class _InvertedTransform(mtransforms.Transform):
        input_dims = output_dims = 1
        is_separable = has_inverse = True

        def __init__(self, x_nodes, y_nodes):
            super().__init__()
            self.x_nodes = np.array(x_nodes, dtype=float)
            self.y_nodes = np.array(y_nodes, dtype=float)

        def transform_non_affine(self, a):
            return np.interp(a, self.y_nodes, self.x_nodes)

        def inverted(self):
            return _PiecewiseLinearScale._PiecewiseTransform(self.x_nodes, self.y_nodes)

mscale.register_scale(_PiecewiseLinearScale)


# ┏━━━━━━━━━━ Temporal Risk-Coverage Plot ━━━━━━━━━━┓
def plot_temporal_risk_coverage_curve_final(save_path: Path,
                                            curve: dict,
                                            probs: np.ndarray,
                                            y_true: np.ndarray,
                                            split_rets: np.ndarray,
                                            fee: float,
                                            op: dict,
                                            split_name: str,
                                            model_label: str,
                                            thres_mode: str,
                                            val_threshold: Optional[float] = None,
                                            val_op: Optional[dict] = None,
                                            cov_min: float = 0.05,
                                            cov_star: float = 0.15,
                                            t_min: float = 1.0,
                                            n_prior: int = 50,
                                            opt_probs: Optional[np.ndarray] = None,
                                            opt_y: Optional[np.ndarray] = None,
                                            opt_rets: Optional[np.ndarray] = None,
                                            direction: str = "",
                                            granularity: str = "",
                                            m1_precision: Optional[float] = None):
    """Risk-coverage plot that visualizes the Stage-A optimization problem.

    Adds on top of `plot_temporal_risk_coverage_curve`:
      * Forbidden zone (Cov < cov_min) shaded red/hatched.
      * Quadratic-penalty zone (cov_min ≤ Cov < cov_star) shaded orange/hatched.
      * Baseline risk floor from M2 τ=0.5 precision: τ̂ must land below it.
      * Stage-A utility curve U(τ) = t_reg x cov_factor on a third y-axis,
        drawn only where Stage-A hard constraints hold (so the feasible region
        is reinforced visually). A gold star marks argmax U.
      * Two reference utility curves: t_reg (no penalty) and t_reg x min(1, cov/cov*)
        (linear penalty), each with its own argmax star, for visual comparison.
    """
    thrs = curve["thresholds"]
    covs = curve["coverage"]
    risks_raw = curve["risk"]

    # ┏━━━━━━━━━━ Colors ━━━━━━━━━━┓
    c_risk     = "#1B4F72"
    c_ret      = "#1E8449"
    c_ret_neg  = "#8B0000"
    c_win      = "#1E8449"
    c_op       = "#8B008B"
    c_grid     = "#D5D8DC"
    c_thr05    = "#34495E"
    c_util_ref = "#E67E22"
    c_forbid   = "#C0392B"
    c_penalty  = "#E67E22"
    c_baseline = "#C0392B"  # red — Risk_ceil
    c_util     = "#B7950B"

    fig_rc, ax_rc = plt.subplots(figsize=(10.5, 6.8), facecolor="white")
    ax_rc.set_facecolor("#FAFAFA")
    x1, y1 = cov_min, cov_min  # [0, C_min] uses natural 1:1 scale
    # Zoom up to the M2@τ=0.5 baseline coverage — all optimizer-relevant action is left of this.
    _cov_05_zoom = float((np.asarray(probs) >= 0.50).sum()) / max(len(probs), 1)
    x2 = max(_cov_05_zoom + 0.02, cov_star + 0.02)  # at least past C*, always past baseline
    if x2 <= x1 + 1e-5: x2 = min(1.0, x1 + 0.1)
    x2 = min(x2, 0.95)
    y2 = min(0.80, y1 + 0.50)  # zoomed region takes up this fraction of axis width
    
    x_nodes, y_nodes = [0.0, x1], [0.0, y1]
    if x2 < 1.0 - 1e-5:
        x_nodes.append(x2); y_nodes.append(y2)
    x_nodes.append(1.0); y_nodes.append(1.0)
    
    ax_rc.set_xscale('piecewise_linear', x_nodes=x_nodes, y_nodes=y_nodes)

    # Alphas per zone. Forbidden: curves are not drawn at all (handled via mask).
    # Penalty: curves drawn with medium alpha so hatches still show through.
    ALPHA_OK = 1.00
    ALPHA_PEN = 0.55
    ALPHA_FOR = 0.0   # forbidden: do not plot

    def _zone_alpha(x):
        return ALPHA_FOR if x < cov_min else (ALPHA_PEN if x < cov_star else ALPHA_OK)

    # ┏━━━━━━━━━━ Feasibility zones (behind everything) ━━━━━━━━━━┓
    # Forbidden zone: white background + stronger red hatch, no fill.
    ax_rc.axvspan(0.0, cov_min, facecolor="white", edgecolor=c_forbid,
                  hatch="///", linewidth=0.0, alpha=1.0, zorder=0)
    ax_rc.axvspan(cov_min, cov_star, color=c_penalty, alpha=0.10, hatch="..",
                  edgecolor=c_penalty, linewidth=0.0, zorder=0)
    ax_rc.axvline(x=cov_min, color=c_forbid, linestyle=":", linewidth=1.2, alpha=0.8, zorder=1)
    ax_rc.axvline(x=cov_star, color=c_penalty, linestyle=":", linewidth=1.2, alpha=0.8, zorder=1)

    # ┏━━━━━━━━━━ Smooth risk curve ━━━━━━━━━━┓
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

    def _plot_alpha_masked(ax, x, y, base_color, lw, ls, zorder):
        """Plot a line in two alpha tiers (penalty/OK). Forbidden zone is skipped."""
        if len(x) < 2: return
        x = np.asarray(x); y = np.asarray(y)
        is_increasing = x[0] < x[-1]
        
        m_pen = (x >= cov_min) & (x <= cov_star)
        idx_pen = np.where(m_pen)[0]
        if len(idx_pen) > 0:
            if is_increasing and idx_pen[-1] < len(x) - 1:
                idx_pen = np.append(idx_pen, idx_pen[-1] + 1)
            elif not is_increasing and idx_pen[0] > 0:
                idx_pen = np.insert(idx_pen, 0, idx_pen[0] - 1)
            if len(idx_pen) >= 2:
                ax.plot(x[idx_pen], y[idx_pen], color=base_color, linewidth=lw, linestyle=ls, alpha=ALPHA_PEN, zorder=zorder)
                
        m_ok = (x >= cov_star)
        idx_ok = np.where(m_ok)[0]
        if len(idx_ok) > 0:
            if is_increasing and idx_ok[0] > 0:
                idx_ok = np.insert(idx_ok, 0, idx_ok[0] - 1)
            elif not is_increasing and idx_ok[-1] < len(x) - 1:
                idx_ok = np.append(idx_ok, idx_ok[-1] + 1)
            if len(idx_ok) >= 2:
                ax.plot(x[idx_ok], y[idx_ok], color=base_color, linewidth=lw, linestyle=ls, alpha=ALPHA_OK, zorder=zorder)

    _plot_alpha_masked(ax_rc, grid_cov, risk_smooth, c_risk, 2.2, "-", 4)
    ax_rc.set_xlabel("Coverage", fontsize=15, fontweight="bold", color="black", labelpad=-2)
    ax_rc.set_ylabel("Risk", fontsize=15, fontweight="bold", color="black", labelpad=8)
    ax_rc.tick_params(axis="x", colors="black", labelcolor="black", labelsize=13, width=1.5)
    ax_rc.tick_params(axis="y", colors="black", labelcolor="black", labelsize=13, width=1.5)
    ax_rc.yaxis.set_major_locator(plt.MaxNLocator(5))
    for spine in ax_rc.spines.values():
        spine.set_color("black")
        spine.set_linewidth(1.5)
    plt.setp(ax_rc.get_xticklabels(), fontweight="bold")
    plt.setp(ax_rc.get_yticklabels(), fontweight="bold")
    ax_rc.set_xlim(-0.02, 1.02)
    ax_rc.grid(True, which="major", color=c_grid, linewidth=0.6, alpha=0.7)
    ax_rc.set_axisbelow(True)

    xticks_cur = [t for t in ax_rc.get_xticks() if 0.0 <= t <= 1.0]
    op_cov_tmp = op.get("coverage", None)
    
    special_ticks = []
    special_labels = {}
    special_colors = {}
    
    special_ticks.append(cov_min)
    special_labels[cov_min] = r"$C_{min}$"
    special_colors[cov_min] = c_forbid
    
    if abs(cov_star - cov_min) > 0.001:
        special_ticks.append(cov_star)
        special_labels[cov_star] = r"$C^{*}$"
        special_colors[cov_star] = c_penalty
    else:
        special_labels[cov_min] = r"$C_{min}/C^{*}$"
        
    if op_cov_tmp is not None and 0.0 <= op_cov_tmp <= 1.0:
        if abs(op_cov_tmp - cov_min) < 0.001:
            special_labels[cov_min] += "\n" + r"$\tau^{*}$" + f"({op_cov_tmp:.2f})"
            special_colors[cov_min] = c_op
        else:
            # Tick stays exactly at op_cov_tmp (where the vertical line is)
            special_ticks.append(op_cov_tmp)
            special_labels[op_cov_tmp] = f"{op_cov_tmp:.2f}"
            special_colors[op_cov_tmp] = c_op
    final_ticks = list(special_ticks)
    for t in xticks_cur:
        clash_tol = 0.005 if cov_min <= t <= cov_star else 0.025
        if not any(abs(t - st) < clash_tol for st in special_ticks):
            final_ticks.append(t)
    final_ticks.append(1.0)
            
    final_ticks = sorted(set(final_ticks))
    ax_rc.set_xticks(final_ticks)
    
    xtick_labels = []
    for t in final_ticks:
        if t in special_labels:
            xtick_labels.append(special_labels[t])
        else:
            xtick_labels.append(f"{t:.1f}")
    ax_rc.set_xticklabels(xtick_labels)
    
    for lbl, t in zip(ax_rc.get_xticklabels(), final_ticks):
        if t in special_colors:
            lbl.set_color(special_colors[t])
            lbl.set_fontweight("bold")
            lbl.set_fontsize(14)
            if special_colors[t] == c_op:
                lbl.set_fontsize(12)
            if t == op_cov_tmp:
                lbl.set_ha('left')
    ax_rc.set_xlim(0.0, 1.0)

    # ┏━━━━━━━━━━ Per-threshold mean returns (plotted-split dataset) ━━━━━━━━━━┓
    mean_rets = np.full_like(thrs, np.nan)

    labels_int = np.asarray(y_true).astype(int)
    N_total = len(probs)

    for i, thr in enumerate(thrs):
        sel = probs >= thr
        n = int(sel.sum())
        if n < 2:
            continue
        net = split_rets[sel] - fee
        lab = labels_int[sel]
        mu = float(np.nanmean(net))
        mean_rets[i] = mu
    # ┏━━━━━━━━━━ Stage-A utility — use the optimizer's dataset when provided ━━━━━━━━━━┓
    # For nocal mode, opt_probs/opt_y/opt_rets = merged Val-Cal+Val-Opt (same N as optimizer).
    # For calibrated mode they are None, so we fall back to the plotted-split dataset.
    _u_probs  = np.asarray(opt_probs)  if opt_probs  is not None else probs
    _u_y      = np.asarray(opt_y).astype(int) if opt_y is not None else labels_int
    _u_rets   = np.asarray(opt_rets)   if opt_rets   is not None else split_rets
    _u_N      = len(_u_probs)

    min_trades = max(50, int(cov_min * _u_N))
    all_net_base = _u_rets[_u_probs >= 0.50] - fee
    base_var = float(np.nanvar(all_net_base, ddof=1)) if len(all_net_base) > 1 else 1.0
    if base_var <= 0:
        base_var = 1.0
    sel_05 = _u_probs >= 0.50
    prec_argmax = float(_u_y[sel_05].mean()) if sel_05.sum() > 0 else 0.0
    # Stage-A precision floor mirrors the optimizer: max(M2@τ=0.5, M1 precision).
    prec_floor_A = prec_argmax
    if m1_precision is not None and not (isinstance(m1_precision, float) and np.isnan(m1_precision)):
        prec_floor_A = max(prec_floor_A, float(m1_precision))

    # Precision of M2@τ=0.5 on the PLOTTED split (used for the visual risk-floor line).
    _plot_sel_05 = np.asarray(probs) >= 0.50
    _plot_prec_argmax = float(labels_int[_plot_sel_05].mean()) if _plot_sel_05.sum() > 0 else 0.0

    # ┏━━━━━━━━━━ Hybrid Threshold Grid (mirrors optimizer) ━━━━━━━━━━┓
    # Union of (a) unique probabilities >= 0.50 emitted by M2 and (b) a dense
    # linspace from 0.50 to 0.95. Sorted and deduplicated. No median start.
    pos_probs = _u_probs[_u_probs >= 0.0]
    linspace_grid = np.linspace(0.0, 0.95, 1000)
    _u_thr_grid = np.unique(np.concatenate([pos_probs, linspace_grid]))
    _u_thr_grid = _u_thr_grid[(_u_thr_grid >= 0.0) & (_u_thr_grid <= 0.95)]

    _u_covs   = np.full(len(_u_thr_grid), np.nan)
    utilities = np.full(len(_u_thr_grid), np.nan)  # quadratic penalty (current optimizer)

    for i, thr in enumerate(_u_thr_grid):
        sel = _u_probs >= thr
        n = int(sel.sum())
        if n < min_trades:
            continue
        cov = n / _u_N
        net = _u_rets[sel] - fee
        mu = float(np.nanmean(net))
        sample_var = float(np.nanvar(net, ddof=1)) if n > 1 else base_var
        shrinkage = n_prior / (n + n_prior)
        reg_var = (1 - shrinkage) * sample_var + shrinkage * base_var
        reg_std = np.sqrt(max(reg_var, 1e-12))
        if reg_std <= 0:
            continue
        t_reg = mu / reg_std * np.sqrt(n)
        cov_factor_quad = 1.0 if cov >= cov_star else (cov / cov_star) ** 2
        utilities[i] = t_reg * cov_factor_quad
        _u_covs[i] = cov

    # ┏━━━━━━━━━━ Return axis (right, primary) ━━━━━━━━━━┓
    ax_ret = ax_rc.twinx()
    valid = ~np.isnan(mean_rets) & (covs >= cov_min)

    def _plot_dynamic_return(ax, x, y, lw, ls, base_alpha, label, zorder):
        if len(x) > 1:
            from matplotlib.collections import LineCollection
            from matplotlib.colors import to_rgba
            x = np.asarray(x); y = np.asarray(y)
            points = np.array([x, y]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            y_mids = segments[:, :, 1].mean(axis=1)
            x_mids = segments[:, :, 0].mean(axis=1)
            keep = []
            seg_colors = []
            for idx, (ym, xm) in enumerate(zip(y_mids, x_mids)):
                if xm < cov_min:
                    continue  # skip forbidden zone entirely
                col = c_ret if ym >= 0 else c_ret_neg
                seg_colors.append(to_rgba(col, alpha=_zone_alpha(xm) * base_alpha))
                keep.append(idx)
            if keep:
                lc = LineCollection(segments[keep], colors=seg_colors, linewidth=lw,
                                    linestyles=ls, zorder=zorder)
                ax.add_collection(lc)
            if label and label != "_nolegend_":
                ax.plot([], [], color=c_ret, linewidth=lw, linestyle=ls, label=label)
        elif len(x) == 1:
            color = c_ret if y[0] >= 0 else c_ret_neg
            ax.plot(x, y, color=color, linewidth=lw, linestyle=ls, alpha=base_alpha, label=label, zorder=zorder)

    _plot_dynamic_return(ax_ret, covs[valid], mean_rets[valid] * 100, 2.0, "-", 0.95, "Mean Return", 3)

    ax_ret.axhline(y=0, color=c_ret, linestyle=":", alpha=0.35, linewidth=1.0)
    # Hide the Return axis visual elements: keep the axis (for data scaling) but
    # suppress label, ticks, and spine so the right side is owned by utility.
    ax_ret.set_ylabel("")
    ax_ret.tick_params(axis="y", which="both", left=False, right=False,
                       labelleft=False, labelright=False)
    for spine in ax_ret.spines.values():
        spine.set_visible(False)

    # ┏━━━━━━━━━━ Utility axis (right, primary on right side) ━━━━━━━━━━┓
    ax_util = ax_rc.twinx()
    ax_util.set_frame_on(True)
    ax_util.patch.set_visible(False)
    for spine_name, spine in ax_util.spines.items():
        spine.set_visible(spine_name == "right")
        spine.set_color(c_util)
        spine.set_linewidth(1.5)
    # ┏━━━━━━━━━━ Helper: dedup + smooth a (cov, utility) curve and plot it ━━━━━━━━━━┓
    def _draw_util_curve(util_arr, color, linestyle, linewidth, star_size,
                         label, alpha=1.0, zorder_curve=3, zorder_star=9,
                         show_star=True, pin_x=None):
        valid = ~np.isnan(util_arr)
        if not valid.any():
            return None
        xv = _u_covs[valid]
        yv = util_arr[valid]
        order = np.argsort(xv)
        xv_s = xv[order]; yv_s = yv[order]
        uniq_cov, inv = np.unique(xv_s, return_inverse=True)
        uniq_util = np.full_like(uniq_cov, -np.inf, dtype=float)
        for k, v in zip(inv, yv_s):
            if v > uniq_util[k]:
                uniq_util[k] = v
        xc = uniq_cov; yc = uniq_util
        if xc.size >= 3:
            try:
                from scipy.interpolate import PchipInterpolator
                grid_xc = np.linspace(xc.min(), xc.max(), 300)
                yc_smooth = PchipInterpolator(xc, yc, extrapolate=False)(grid_xc)
                mvalid = np.isfinite(yc_smooth)
                xp, yp = grid_xc[mvalid], yc_smooth[mvalid]
            except Exception:
                xp, yp = xc, yc
        else:
            xp, yp = xc, yc
        # Draw as one continuous line — no segment split needed for the curve itself.
        # The zone shading underneath already conveys the penalty region visually.
        if xp.size >= 2:
            ax_util.plot(xp, yp, color=color, linestyle=linestyle,
                         linewidth=linewidth, alpha=alpha, zorder=zorder_curve,
                         label=label)
        sxy = None
        if show_star and yc.size > 0:
            if pin_x is not None:
                # Pin star to the optimizer's chosen coverage (interpolated on the curve)
                star_y = float(np.interp(pin_x, xc, yc))
                sxy = (float(pin_x), star_y)
            else:
                i_max = int(np.argmax(yc))
                sxy = (float(xc[i_max]), float(yc[i_max]))
            ax_util.scatter([sxy[0]], [sxy[1]], color=color, marker="*",
                            s=star_size, edgecolors="white", linewidths=1.2,
                            zorder=zorder_star)
        return sxy

    # ┏━━━━━━━━━━ Utility curve (quadratic penalty — current optimizer) ━━━━━━━━━━┓
    # Star pinned to the optimizer's chosen coverage so it's always vertically
    # aligned with the τ̂ vertical line, not the unconstrained plot maximum.
    star_xy_main = _draw_util_curve(utilities, c_util, "--", 1.8, 200,
                                    r"Risk-Profitability Score",
                                    alpha=1.0, zorder_curve=3, zorder_star=9,
                                    pin_x=op["coverage"])
    star_xy = None
    ax_util.set_ylabel("Risk-Profitability Score",
                       fontsize=15, fontweight="bold", color=c_util, labelpad=8)
    ax_util.tick_params(axis="y", colors=c_util, labelcolor=c_util, labelsize=13, width=1.2)
    ax_util.yaxis.set_major_locator(plt.MaxNLocator(5))
    plt.setp(ax_util.get_yticklabels(), fontweight="bold")

    # ┏━━━━━━━━━━ Baseline precision floor (segment, skips forbidden zone) ━━━━━━━━━━┓
    risk_floor = 1.0 - max(_plot_prec_argmax, float(m1_precision) if m1_precision is not None and not (isinstance(m1_precision, float) and m1_precision != m1_precision) else 0.0)
    ax_rc.plot([cov_min, 1.0], [risk_floor, risk_floor],
               color=c_baseline, linestyle="-.", linewidth=1.6, alpha=0.9, zorder=3)

    # ┏━━━━━━━━━━ Operating points (τ=0.5 and τ̂) ━━━━━━━━━━┓
    idx_05 = int(np.argmin(np.abs(thrs - 0.5)))
    cov_05 = covs[idx_05]
    risk_05 = np.interp(cov_05, grid_cov, risk_smooth) if grid_cov.size >= 2 else risks_raw[idx_05]
    op_cov = op["coverage"]
    op_risk = np.interp(op_cov, grid_cov, risk_smooth) if grid_cov.size >= 2 else op.get("risk", 0)
    thr_source = op.get("threshold_source") or ("Val-Utility" if split_name == "Test" else "Utility-Opt")
    show_baseline = abs(op_cov - cov_05) > 0.02 and abs(op["threshold"] - 0.5) > 0.01
    if show_baseline:
        ax_rc.axvline(x=cov_05, color=c_thr05, linestyle="--", alpha=0.7, linewidth=1.8)
        ax_rc.scatter([cov_05], [risk_05], color=c_thr05, marker="o", s=40,
                      edgecolors="white", linewidths=1.0, zorder=5)

    aop = _zone_alpha(op_cov)  # kept for axvline only
    ax_rc.axvline(x=op_cov, color=c_op, linestyle="--", alpha=0.7, linewidth=1.8)
    ax_rc.scatter([op_cov], [op_risk], color=c_op, marker="D", s=50,
                  edgecolors="white", linewidths=1.0, zorder=6)
    ax_rc.annotate(rf"$\tau^*$={op['threshold']:.3f}", xy=(op_cov, op_risk), xytext=(3, 6),
                   textcoords="offset points", fontsize=12, color=c_op, fontweight="bold", zorder=10,
                   bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=c_op, alpha=0.9, lw=0.6))

    # Return annotations at τ̂
    mr_val = op["mean_ret"] * 100
    # Use the optimizer's dataset (opt_probs/opt_rets/opt_y) when provided so that
    # mr/mw/ml annotations match op["mean_ret"] exactly (same N, same population).
    _ann_probs  = _u_probs   if opt_probs is not None else probs
    _ann_rets   = _u_rets    if opt_rets  is not None else split_rets
    _ann_labels = _u_y       if opt_y     is not None else labels_int
    sel_op = (_ann_probs >= op["threshold"])
    n_op = int(sel_op.sum())
    # Interpolate exact y-coordinates from the return curves at op_cov so markers
    # land precisely on top of each line.
    def _interp_on_curve(x_target, mask, values):
        xs = covs[mask]
        ys = values[mask]
        if len(xs) < 2:
            return None
        order = np.argsort(xs)
        xs, ys = xs[order], ys[order]
        if x_target < xs[0] or x_target > xs[-1]:
            return None
        return float(np.interp(x_target, xs, ys))

    # Dot y-positions: always interpolated from the plotted curves so dots sit exactly
    # on top of the visible lines regardless of which dataset was used for labels.
    # when opt_rets is provided; otherwise also use the curve interpolation.
    mr_dot = _interp_on_curve(op_cov, valid,   mean_rets      * 100)
    if opt_rets is None:
        # Calibrated mode: labels and dots both come from the plotted curves.
        if mr_dot is not None: mr_val = mr_dot

    def _get_staggered_offsets(val_dict):
        valid_vals = {k: v for k, v in val_dict.items() if v is not None}
        s_keys = sorted(valid_vals.keys(), key=lambda k: valid_vals[k])
        if len(s_keys) == 3:
            return {s_keys[0]: (3, -8), s_keys[1]: (3, 0), s_keys[2]: (3, 8)}
        if len(s_keys) == 2:
            return {s_keys[0]: (3, -5), s_keys[1]: (3, 5)}
        if len(s_keys) == 1:
            return {s_keys[0]: (3, 0)}
        return {}

    if mr_dot is not None and mr_val is not None:
        color = c_ret if mr_val >= 0 else c_ret_neg
        ax_ret.scatter([op_cov], [mr_dot], color=color, marker="D", s=40,
                       edgecolors="white", linewidths=1.0, zorder=7)
        ax_ret.annotate(f"{mr_val:+.2f}%", xy=(op_cov, mr_dot), xytext=(3, 6),
                        textcoords="offset points", fontsize=12, color=color,
                        fontweight="bold", zorder=10)



    # ┏━━━━━━━━━━ Title ━━━━━━━━━━┓
    _model_display = {"RF": "Random Forest", "rf": "Random Forest"}.get(model_label, model_label)
    _split_display = {"Val": "Validation", "val": "Validation"}.get(split_name, split_name)
    _dir_gran = f"  |  {direction.upper()}  {granularity}" if direction or granularity else ""
    # title removed

    # ┏━━━━━━━━━━ Single legend: 3 rows × 4 columns ━━━━━━━━━━┓
    # Stats (Prec, Cov, μ/t) are embedded as multi-line text in col-4 entries
    # so no phantom handle-space is wasted on invisible patches.
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    import matplotlib.colors as mcolors
    _prec_str = rf"Prec$_{{\hat{{τ}}}}$={op.get('precision', float('nan'))*100:.1f}%"
    _cov_str  = rf"Cov$_{{\hat{{τ}}}}$={op['coverage']*100:.1f}% (N={op.get('selected_count', 0)})"
    _mut_str  = rf"$t$-statistic: t={op.get('t_stat', 0):.2f}"
    handles = [
        # ── Column 1: Curves ──────────────────────────────────────────
        Line2D([], [], color=c_util, marker="*", markersize=15, linestyle="None",
               markeredgecolor="white", markeredgewidth=1.0, label=r"Max. Risk-Profitability Score"),
        Line2D([], [], color=c_util, linewidth=2.0, linestyle="--", label=r"Risk-Profitability Score"),
        Line2D([], [], color=c_risk, linewidth=2.2, label="Risk-Coverage Curve"),

        # ── Column 2: Zones ───────────────────────────────────────────
        Line2D([], [], color=c_ret, linewidth=1.8, linestyle="-", label=rf"Mean net return $\mu$={op['mean_ret']*100:+.2f}% (i)"),
        Line2D([], [], color=c_op, marker="D", markersize=8, linestyle="--",
               markeredgecolor="white", markeredgewidth=0.8,
               label=rf"$\tau^*={op['threshold']:.3f}$, {_mut_str} (ii)"),
        Line2D([], [], color=c_baseline, linewidth=1.6, linestyle="-.", label=r"$Risk_{ceil}$ (iii)"),

        # ── Column 3: Zones + baseline τ ──────────────────────────────
        Patch(facecolor="white", alpha=1.0, hatch="///", edgecolor=c_forbid,
              label=r"Coverage: $\mathcal{C} < \mathcal{C}_{min}$ (iv)"),
        Patch(facecolor=c_penalty, alpha=0.20, hatch="..", edgecolor=c_penalty,
              label=r"$(\mathcal{C}/\mathcal{C}^{*})^{2} < 1$"),
        Line2D([], [], color=c_thr05, linewidth=1.8, linestyle="--", label=r"$\tau=0.5$ (Baseline)"),
    ]

    # Center the legend on the main axes span [left=0.08, right=0.92].
    _leg_cx = (0.08 + 0.92) / 2
    leg = fig_rc.legend(handles=handles, loc="lower center",
                        bbox_to_anchor=(_leg_cx -0.005, -0.015), ncol=3,
                        prop={"size": 14}, frameon=True, framealpha=0.95,
                        edgecolor="#BDC3C7", fancybox=True,
                        handlelength=2.4, handletextpad=0.6,
                        columnspacing=1.2, borderpad=0.6)
    leg.set_zorder(20)
    fig_rc.tight_layout()
    fig_rc.subplots_adjust(left=0.08, bottom=0.22, right=0.92, top=0.97)

    fig_rc.savefig(str(save_path), dpi=500, facecolor="white")
    plt.close(fig_rc)


# ┏━━━━━━━━━━ Performance over n features ━━━━━━━━━━┓
def plot_performance_over_n_features(base_dir:         str = "/home/till/PycharmProjects/Secondary-Model/src/Output",
                                     m1:               str = "kronos",
                                     m2:               str = "rf",
                                     direction:        str = "up",
                                     granularity:      str = "1d",
                                     meta_label_mode:  str = "tp",
                                     scoring:          str = "accuracy",
                                     cv_strategy:      str = "CombinatorialPurgedEmbargoCV",
                                     n_splits:         int = 10) -> None:

    # ┏━━━━━━━━━━ Set up search directory and file mask ━━━━━━━━━━┓
    search_dir = (f"{base_dir}/"
                  f"{m1.capitalize()}/"
                  f"{m2}/"
                  f"{direction.upper()}/"
                  f"interpretability/"
                  f"feature_selection/"
                  f"{granularity}_{meta_label_mode}")

    # ┏━━━━━━━━━━ File mask ━━━━━━━━━━┓
    file_mask = f"*_features_{scoring}_{cv_strategy}_{n_splits}_cached.csv"

    # ┏━━━━━━━━━━ Sorted files ━━━━━━━━━━┓
    files = sorted(glob.glob(f"{search_dir}/{file_mask}"),
                   key = lambda x: int(re.search(r"^(\d+)_features", os.path.basename(x)).group(1)))

    # ┏━━━━━━━━━━ Lists to store performance metrics ━━━━━━━━━━┓
    n_features = []
    val_mean = []
    val_std = []
    test_mean = []
    test_std = []

    # ┏━━━━━━━━━━ Extract performance metrics from each file ━━━━━━━━━━┓
    for file in files:
        # ┏━━━━━━━━━━ Read the CSV file ━━━━━━━━━━┓
        df = pd.read_csv(file)

        # ┏━━━━━━━━━━ Extract the number of features ━━━━━━━━━━┓
        file_name = os.path.basename(file)
        n_feature = int(file_name.split("_")[0])

        # ┏━━━━━━━━━━ Find the best index ━━━━━━━━━━┓
        best_idx = df['mean_val_scoring'].argmax()

        # ┏━━━━━━━━━━ Append the performance metrics ━━━━━━━━━━┓
        n_features.append(n_feature)
        val_mean.append(df['mean_val_scoring'].iloc[best_idx])
        val_std.append(df['std_val_scoring'].iloc[best_idx])
        test_mean.append(df['mean_test_scoring'].iloc[best_idx])
        test_std.append(df['std_test_scoring'].iloc[best_idx])

    # ┏━━━━━━━━━━ Convert lists to numpy arrays ━━━━━━━━━━┓
    n_features = np.array(n_features)
    val_mean = np.array(val_mean)
    val_std = np.array(val_std)
    test_mean = np.array(test_mean)
    test_std = np.array(test_std)

    # ┏━━━━━━━━━━ Plotting ━━━━━━━━━━┓
    fig, ax = plt.subplots(figsize=(12, 5))

    # ┏━━━━━━━━━━ Plot validation set ━━━━━━━━━━┓
    ax.plot(n_features, val_mean, label="Validation", marker="o")
    ax.fill_between(n_features, val_mean - val_std, val_mean + val_std, alpha=0.2)

    # ┏━━━━━━━━━━ Plot test set ━━━━━━━━━━┓
    ax.plot(n_features, test_mean, label="Test", marker="o")
    ax.fill_between(n_features, test_mean - test_std, test_mean + test_std, alpha=0.2)

    # ┏━━━━━━━━━━ Set labels and title ━━━━━━━━━━┓
    ax.set_xlabel("Number of Features")
    ax.set_ylabel("Scoring")
    ax.set_title(f"M1={m1} | M2={m2} | time frame={granularity} | direction={direction} | meta label mode={meta_label_mode}")
    ax.legend()
    ax.grid(True)
    plt.tight_layout()

    # ┏━━━━━━━━━━ Save plot ━━━━━━━━━━┓
    plt.savefig(f"{search_dir}/strategy={cv_strategy}_scoring={scoring}_n_splits={n_splits}_min_max={1}_{len(files)}_summary_plot.pdf")
    plt.close()

