"""Feature analysis and plotting helpers extracted from kronos_tree.py."""

import warnings
import re
import glob
import os
import json
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path
from scipy import stats
from typing import Dict, List, Optional, Any, Union

# ┏━━━━━━━━━━ Feature Selection Methods ━━━━━━━━━━┓
from sklearn.feature_selection import mutual_info_classif

# ┏━━━━━━━━━━ SkLearn Metrics ━━━━━━━━━━┓
from sklearn.metrics import (accuracy_score, f1_score, precision_score, recall_score,
                             precision_recall_fscore_support, confusion_matrix, 
                             ConfusionMatrixDisplay, fbeta_score, matthews_corrcoef)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

# ┏━━━━━━━━━━ Utils ━━━━━━━━━━┓
from Utils.utils import model_label

# ┏━━━━━━━━━━ Data ━━━━━━━━━━┓
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
    "plot_class_distributions",
    "plot_mutual_information",
    "plot_tree_importance",
    "_plot_prob_distribution",
    "plot_class_distribution",
    "plot_meta_label_returns_histogram",
    "plot_prediction_returns_histogram",
    "plot_m1_prediction_returns_histogram",
    "plot_confusion_matrix",
    "plot_temporal_risk_coverage_curve",
    "plot_temporal_risk_coverage_curve_final",
    "plot_ocp_threshold_evolution",
    "plot_selective_return_distribution",
    "plot_asset_correlation",
    "plot_dataset_size_distribution",
    "plot_return_quality_distribution",
    "plot_best_m2_per_gran",
    "plot_results_radar_focused",
    "plot_kronos_down_combined",
]





# ┏━━━━━━━━━━ Feature Correlation Heatmap ━━━━━━━━━━┓
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



# ┏━━━━━━━━━━ Class Distributions ━━━━━━━━━━┓
def plot_class_distributions(df: pd.DataFrame, labels: np.ndarray, class_names: list, save_dir: Path):
    """Plot class distributions of features."""
    # ┏━━━━━━━━━━ Copy Data ━━━━━━━━━━┓
    plot_df = df.copy()
    plot_df["class"] = [class_names[l] for l in labels]

    # ┏━━━━━━━━━━ Calculate Dimensions ━━━━━━━━━━┓
    n_features = len(df.columns)
    n_cols = 4
    n_rows = int(np.ceil(n_features / n_cols))

    # ┏━━━━━━━━━━ Create Subplots ━━━━━━━━━━┓
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3.5, n_rows * 3))
    axes = axes.flatten()

    # ┏━━━━━━━━━━ Plot Violin Plots ━━━━━━━━━━┓
    for i, col in enumerate(df.columns):
        ax = axes[i]
        sns.violinplot(data      = plot_df,
                       x         = "class",
                       y         = col,
                       hue       = "class",
                       ax        = ax,
                       inner     = "quartile",
                       palette   = ["#3498db", "#e74c3c"],
                       linewidth = 0.8,
                       cut       = 0,
                       legend    = False)
        ax.set_title(col, fontsize=9, fontweight="bold")
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.tick_params(labelsize=7)

    for j in range(n_features, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Feature Distributions by Class", fontsize=13, y=1.01)
    plt.tight_layout()

    # ┏━━━━━━━━━━ Save Violin Plots ━━━━━━━━━━┓
    fig.savefig(save_dir / "3_class_distributions_violin.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  [3/9] Class-conditional violin plots saved")



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



# ┏━━━━━━━━━━ Tree Importance ━━━━━━━━━━┓
def plot_tree_importance(df:           pd.DataFrame,
                         labels:       np.ndarray,
                         save_dir:     Path,
                         model_name:   str = "rf",
                         step_label:   str = "5/9",
                         desc:         str = "all features",
                         file_prefix:  str = "5_feature_importance",
                         class_names:  list = None,
                         meta_mode:    str = "tp",
                         model_builder     = None,
                         model_labeler     = None):
    """Returns ({feature: importance}, cv_metrics)."""
    # ┏━━━━━━━━━━ Build Model ━━━━━━━━━━┓
    if model_builder is None:
        raise ValueError("model_builder must be provided to plot_tree_importance")
    builder = model_builder
    labeler = model_labeler or model_label
    mlabel = labeler(model_name)

    # ┏━━━━━━━━━━ Copy Data ━━━━━━━━━━┓
    X = df.values.copy()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # ┏━━━━━━━━━━ Calculate Class Weights ━━━━━━━━━━┓
    n_pos = int((labels == 1).sum())
    n_neg = int((labels == 0).sum())
    cw_ratio = n_neg / max(n_pos, 1)

    # ┏━━━━━━━━━━ Initialize Model ━━━━━━━━━━┓
    model = builder(model_name, len(labels), cw_ratio)
    model.fit(X_scaled, labels)
    importances = model.feature_importances_

    # ┏━━━━━━━━━━ Sort Results ━━━━━━━━━━┓
    imp_df = pd.DataFrame({"feature": df.columns, "importance": importances}).sort_values("importance", ascending=True)

    # ┏━━━━━━━━━━ Plot Bar Chart ━━━━━━━━━━┓
    fig, ax = plt.subplots(figsize=(8, max(5, len(imp_df) * 0.35)))
    ax.barh(imp_df["feature"], imp_df["importance"], color="#e67e22", edgecolor="k", linewidth=0.4)
    ax.set_xlabel("Feature Importance")
    ax.set_title(f"{mlabel} Feature Importance ({desc})", fontsize=12, pad=10)
    plt.tight_layout()
    
    # ┏━━━━━━━━━━ Save Bar Chart ━━━━━━━━━━┓
    fig.savefig(save_dir / f"{file_prefix}.png", dpi=200)
    plt.close(fig)

    # ┏━━━━━━━━━━ Save Results ━━━━━━━━━━┓
    imp_df.to_csv(save_dir / f"{file_prefix}.csv", index=False, float_format="%.6f")

    # ┏━━━━━━━━━━ Time Series Cross-Validation ━━━━━━━━━━┓
    tscv = TimeSeriesSplit(n_splits=5)
    cv_preds = np.full_like(labels, fill_value=-1)
    for train_idx, val_idx in tscv.split(X_scaled):
        m_cv = builder(model_name,
                       len(train_idx),
                       (labels[train_idx] == 0).sum() / max((labels[train_idx] == 1).sum(), 1))
        m_cv.fit(X_scaled[train_idx], labels[train_idx])
        cv_preds[val_idx] = m_cv.predict(X_scaled[val_idx])
    scored_mask = cv_preds >= 0

    # ┏━━━━━━━━━━ Calculate CV Metrics ━━━━━━━━━━┓
    cv_metrics = {"accuracy":  round(float(accuracy_score(labels[scored_mask], cv_preds[scored_mask])), 4),
                  "precision": round(float(precision_score(labels[scored_mask], cv_preds[scored_mask], zero_division=0)), 4),
                  "recall":    round(float(recall_score(labels[scored_mask], cv_preds[scored_mask], zero_division=0)), 4),
                  "f1_score":  round(float(f1_score(labels[scored_mask], cv_preds[scored_mask], zero_division=0)), 4)}
    
    # ┏━━━━━━━━━━ Print CV Metrics ━━━━━━━━━━┓
    n_feats = df.shape[1]
    print(f"  [{step_label}] {mlabel} ({desc}, {n_feats} feats) 5-fold chrono CV: "
          f"acc={cv_metrics['accuracy']:.3f} prec={cv_metrics['precision']:.3f} "
          f"rec={cv_metrics['recall']:.3f} f1={cv_metrics['f1_score']:.3f}")

    # ┏━━━━━━━━━━ Plot Confusion Matrix (5-fold Chrono CV diagnostic) ━━━━━━━━━━┓
    if class_names is not None:
        cm_path = save_dir / f"{file_prefix}_CM.png"
        plot_confusion_matrix(labels[scored_mask],
                              cv_preds[scored_mask],
                              classes=class_names,
                              save_path=str(cm_path),
                              title=f"{mlabel} — 5-Fold Chronological CV ({desc})\n[Diagnostic: feature importance sanity check, not a hold-out eval]",
                              meta_mode=meta_mode)

    return {row["feature"]: row["importance"] for _, row in imp_df.iterrows()}, cv_metrics



# ┏━━━━━━━━━━ Plot Probability Distribution ━━━━━━━━━━┓
def _plot_prob_distribution(y_true:      np.ndarray,
                            probs:       np.ndarray,
                            class_names: list,
                            save_dir:    Path,
                            file_prefix: str,
                            title:       str):
    """Plot histogram + KDE of predicted P(class=1) split by true label."""
    
    # ┏━━━━━━━━━━ Epsilon and Logit Difference ━━━━━━━━━━┓
    eps = 1e-7
    p_clipped = np.clip(probs, eps, 1 - eps)
    logit_diff = np.log(p_clipped / (1 - p_clipped))

    # ┏━━━━━━━━━━ Create DataFrame ━━━━━━━━━━┓
    df_plot = pd.DataFrame({"prob": probs,
                            "logit_diff": logit_diff,
                            "class": [class_names[int(y)] for y in y_true]})
    hue_order = [class_names[0], class_names[1]]

    # ┏━━━━━━━━━━ Plot Histograms ━━━━━━━━━━┓
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # ┏━━━━━━━━━━ Plot Probability Distribution ━━━━━━━━━━┓
    sns.histplot(data      = df_plot,
                 x         = "prob",
                 hue       = "class",
                 hue_order = hue_order,
                 bins      = 50,
                 kde       = True,
                 ax        = axes[0],
                 palette   = "Set1",
                 alpha     = 0.6)
    axes[0].set_title(f"{title} — P(class 1)")
    axes[0].set_xlabel("Probability of Class 1")
    axes[0].set_ylabel("Count")
    axes[0].set_xlim(0, 1)

    # ┏━━━━━━━━━━ Plot Logit Difference ━━━━━━━━━━┓
    sns.histplot(data      = df_plot,
                 x         = "logit_diff",
                 hue       = "class",
                 hue_order = hue_order,
                 bins      = 50,
                 kde       = True,
                 ax        = axes[1],
                 palette   = "Set1",
                 alpha     = 0.6)
    axes[1].set_title(f"{title} — Logit Difference")
    axes[1].set_xlabel("log(p / (1-p))")
    axes[1].set_ylabel("Count")
    plt.tight_layout()

    # ┏━━━━━━━━━━ Save Plot ━━━━━━━━━━┓
    fig.savefig(save_dir / f"{file_prefix}.png", dpi=200)
    plt.close(fig)



# ┏━━━━━━━━━━ Plot Class Distribution ━━━━━━━━━━┓
def plot_class_distribution(dataset: Dict[str, torch.Tensor],
                            idx_train: List[int],
                            idx_meta: List[int],
                            idx_val: List[int],
                            idx_test: List[int],
                            save_path: Optional[Union[str, Path]] = None,
                            show: bool = False,
                            title_suffix: str = "",
                            meta_mode: str = None) -> None:
    """
    Plot meta-label class distribution across train/meta/val/test splits.
    
    Creates a grouped bar chart showing class balance.
    
    Parameters
    ----------
    dataset : Dict
        Output from prepare_multi_asset_dataset
    idx_train, idx_meta, idx_val, idx_test : List[int]
        Split indices
    save_path : Path, optional
        If provided, save the figure
    show : bool
        Whether to display the plot
    meta_mode : str, optional
        The meta_label_mode used (e.g., 'og', 'fp', 'tp') to accurately label the x-axis.
    """
    print(f"\n┏━━━━━━━━━━ Plotting Class Distribution (Train/Val/Test) ━━━━━━━━━━┓")
    # ┏━━━━━━━━━━ Import matplotlib ━━━━━━━━━━┓
    import matplotlib.pyplot as plt
    
    # ┏━━━━━━━━━━ Extract labels ━━━━━━━━━━┓
    from Utils.data import MultiGranDataset
    labels = dataset.labels if isinstance(dataset, MultiGranDataset) else dataset['labels']
    
    # ┏━━━━━━━━━━ Get class counts ━━━━━━━━━━┓
    def get_class_counts(indices):
        split_labels = labels[indices].numpy()
        unique = np.unique(split_labels[~np.isnan(split_labels)])
        n_nan = int(np.isnan(split_labels).sum())
        
        # Check if 3-class (UP/FLAT/DN) - usually 0.0, 1.0, 2.0
        # If max label is 2, treat as 3-class even if some are missing in this split
        if (len(unique) > 0 and unique.max() > 1.0) or (len(unique) > 2):
            # 3-Class Logic: 0=UP, 1=FLAT, 2=DN (as per ground_truth)
            n_up   = int((split_labels == 0.0).sum())
            n_flat = int((split_labels == 1.0).sum())
            n_dn   = int((split_labels == 2.0).sum())
            return {'UP': n_up, 'FLAT': n_flat, 'DN': n_dn, 'NaN': n_nan}
        else:
            if meta_mode:
                m_mode = meta_mode.lower()
                if m_mode == 'fp':
                    lbl_1, lbl_0 = '1 (FP)', '0 (TP)'
                elif m_mode == 'tp':
                    lbl_1, lbl_0 = '1 (TP)', '0 (FP)'
                else: # 'og'
                    lbl_1, lbl_0 = '1 (Success)', '0 (Fail)'
            else:
                lbl_1, lbl_0 = '1', '0'
                
            n_pos = int((split_labels == 1.0).sum())
            n_neg = int((split_labels == 0.0).sum())
            return {lbl_1: n_pos, lbl_0: n_neg, 'NaN': n_nan}
    
    # ┏━━━━━━━━━━ Get class counts for each split ━━━━━━━━━━┓
    train_counts = get_class_counts(idx_train)
    val_counts   = get_class_counts(idx_val)
    test_counts  = get_class_counts(idx_test)
    
    # ┏━━━━━━━━━━ Create figure ━━━━━━━━━━┓
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    splits = ['Train', 'Validation', 'Test']
    all_counts = [train_counts, val_counts, test_counts]
    colors_map = {'UP': '#4CAF50', 'FLAT': '#FFC107', 'DN': '#F44336', # Green, Amber, Red
                  '1 (TP)': '#4CAF50', '0 (FP)': '#F44336',
                  '1 (FP)': '#4CAF50', '0 (TP)': '#F44336',
                  '1 (Success)': '#4CAF50', '0 (Fail)': '#F44336',
                  '1': '#4CAF50', '0': '#F44336',
                  'NaN': '#9E9E9E'}
    
    # ┏━━━━━━━━━━ Iterate Over Splits ━━━━━━━━━━┓
    for ax, split_name, counts_dict in zip(axes, splits, all_counts):
        total = sum(v for k, v in counts_dict.items() if k != 'NaN') + counts_dict['NaN']
        
        # ┏━━━━━━━━━━ Determine categories present ━━━━━━━━━━┓
        if 'UP' in counts_dict:
            categories = ['UP', 'FLAT', 'DN', 'NaN']
        else:
            categories = [c for c in counts_dict.keys() if c != 'NaN'] + ['NaN']
        
        # ┏━━━━━━━━━━ Get values and colors ━━━━━━━━━━┓
        values = [counts_dict.get(c, 0) for c in categories]
        bar_colors = [colors_map.get(c, '#9E9E9E') for c in categories]
        
        # ┏━━━━━━━━━━ Plot bars ━━━━━━━━━━┓
        bars = ax.bar(categories, values, color=bar_colors)
        ax.set_title(f'{split_name}')
        ax.set_ylabel('Count')
        
        # ┏━━━━━━━━━━ Add total count ━━━━━━━━━━┓
        ax.text(0.95, 0.95, f'Total: {total:,}', 
                transform=ax.transAxes, 
                ha='right', va='top', 
                fontsize=10, fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
        
        # ┏━━━━━━━━━━ Add labels ━━━━━━━━━━┓
        max_height = 0
        for bar, count in zip(bars, values):
            height = bar.get_height()
            max_height = max(max_height, height)
            if total > 0:
                pct = 100 * count / total
            else:
                pct = 0
            
            # ┏━━━━━━━━━━ Label placement logic ━━━━━━━━━━┓
            if height > (total * 0.05):
                y_pos = height - (total * 0.01)
                va = 'top'
                color = 'white' if bar.get_facecolor() != (1.0, 1.0, 1.0, 1.0) else 'black'
            else:
                y_pos = height + (total * 0.01)
                va = 'bottom'
                color = 'black'
                
            ax.text(bar.get_x() + bar.get_width()/2., y_pos,
                    f'{count:,}\n({pct:.1f}%)',
                    ha='center', va=va, fontsize=9, color=color, fontweight='bold')

        # ┏━━━━━━━━━━ Ensure upper bound is at least 1.0 to avoid singular transformation warning ━━━━━━━━━━┓
        upper_limit = max(max_height, total) * 1.15
        if upper_limit <= 0:
            upper_limit = 1.0
        ax.set_ylim(0, upper_limit)
    
    title = 'Meta-Label Class Distribution'
    if title_suffix:
        title += f": {title_suffix}"
    fig.suptitle(title, fontsize=16, fontweight='bold', y=0.98)
    
    # ┏━━━━━━━━━━ Adjust layout to prevent overlap between suptitle and ax titles ━━━━━━━━━━┓
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)

    # ┏━━━━━━━━━━ Save plot if save_path is provided ━━━━━━━━━━┓
    if save_path:
        save_path = Path(save_path)
        if len(save_path.parts) == 1:
            output_dir = Path("Output/Analysis")
            output_dir.mkdir(parents=True, exist_ok=True)
            save_path = output_dir / save_path
        else:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[plot_class_distribution] Saved to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close(fig)



# ┏━━━━━━━━━━ Plot Meta-Label Returns Histogram ━━━━━━━━━━┓
def plot_meta_label_returns_histogram(dataset: Dict[str, Any], 
                                      indices: List[int], 
                                      save_path: str,
                                      title_suffix: str = "",
                                      all_indices: Optional[List[int]] = None):
    """
    Plots a histogram of the actual continuous returns for the given indices,
    separated by meta_label (0 vs 1). Analyzes what percentage increases
    are captured by meta_label = 1.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import os
    from typing import Any, Optional

    returns_all = _get_from_dataset(dataset, 'returns')
    labels_all  = _get_from_dataset(dataset, 'labels')

    if returns_all is None:
        print("[WARN] 'returns' not found in dataset. Cannot plot returns histogram.")
        return

    # ┏━━━━━━━━━━ Setup Data & Dynamic Limits ━━━━━━━━━━┓
    fee = 0.20 # 0.2% round-trip fee
    
    # ┏━━━━━━━━━━ Foreground data ━━━━━━━━━━┓
    labels = labels_all[indices].numpy()
    returns = returns_all[indices].numpy() * 100.0
    valid = ~np.isnan(returns)
    labels = labels[valid]
    returns = returns[valid]
    ret_1 = returns[labels == 1]
    ret_0 = returns[labels == 0]
    
    # ┏━━━━━━━━━━ Background data (if requested) ━━━━━━━━━━┓
    all_valid = None
    if all_indices is not None:
        all_rets = returns_all[all_indices].numpy() * 100.0
        all_valid = all_rets[~np.isnan(all_rets)]
        
    # ┏━━━━━━━━━━ Compute limits over ALL data to be plotted ━━━━━━━━━━┓
    data_to_limit = [ret_0, ret_1]
    if all_valid is not None:
        data_to_limit.append(all_valid)
        
    low, high = get_dynamic_ret_limits(data_to_limit)
    step = 0.1 if high <= 10 else 0.2
    bins = np.arange(low, high + step, step)

    # ┏━━━━━━━━━━ Setup Plot ━━━━━━━━━━┓
    fig, ax1 = plt.subplots(figsize=(12, 7))
    # ┏━━━━━━━━━━ Background (All Samples) ━━━━━━━━━━┓
    if all_valid is not None:
        ax2 = ax1.twinx()
        ax2.hist(all_valid, bins=bins, alpha=0.2, color='black', label=f'Complete Dataset (N={len(all_valid):,})', zorder=1)
        ax2.set_ylabel("Number of Windows (Full Dataset)", color='black', alpha=0.6)
        ax2.tick_params(axis='y', labelcolor='black', colors='dimgrey')

    # ┏━━━━━━━━━━ Foreground (Selective Indices) ━━━━━━━━━━┓
    ax1.hist(ret_0, bins=bins, alpha=0.5, color='red', label=f'Meta-Label 0 (N={len(ret_0):,}, Mean={np.mean(ret_0):.2f}%)', zorder=2)
    ax1.hist(ret_1, bins=bins, alpha=0.5, color='green', label=f'Meta-Label 1 (N={len(ret_1):,}, Mean={np.mean(ret_1):.2f}%)', zorder=3)
    
    ax1.axvline(x=0, color='black', linestyle='--', alpha=0.5)
    ax1.axvline(x=fee, color='magenta', linestyle=':', alpha=0.7, label=f'Fee Break-Even (±{fee}%)')
    ax1.axvline(x=-fee, color='magenta', linestyle=':', alpha=0.7)
    
    if len(ret_0) > 0:
        ax1.axvline(x=np.mean(ret_0), color='firebrick', linestyle='-', alpha=0.8, label=f'Mean 0')
    if len(ret_1) > 0:
        ax1.axvline(x=np.mean(ret_1), color='darkgreen', linestyle='-', alpha=0.8, label=f'Mean 1')
    
    ax1.set_title(f"Ground Truth Returns Distribution by Meta-Label {title_suffix}")
    ax1.set_xlabel("Ground Truth Return (%)")
    ax1.set_ylabel("Number of Windows (Filtered)", color='black')
    ax1.set_xlim(low, high)
    
    # ┏━━━━━━━━━━ Combine legends from both axes ━━━━━━━━━━┓
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    if all_indices is not None:
        lines_2, labels_2 = ax2.get_legend_handles_labels()
        ax1.legend(lines_2 + lines_1, labels_2 + labels_1, loc='upper left')
    else:
        ax1.legend(loc='upper left')
        
    ax1.grid(True, alpha=0.3)
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f"[plot_meta_label_returns_histogram] Saved to {save_path}")



# ┏━━━━━━━━━━ Plot Prediction Returns Histogram ━━━━━━━━━━┓
def plot_prediction_returns_histogram(dataset: Dict[str, Any], 
                                      indices: List[int], 
                                      preds: np.ndarray,
                                      interested_class: int,
                                      save_path: str,
                                      title_suffix: str = ""):
    """
    Plots a histogram comparing the actual returns of the model's positive predictions
    against the ground-truth actual returns for the interested class.
    Shows fee boundaries.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import os
    from typing import Any

    returns_all = _get_from_dataset(dataset, 'returns')
    labels_all  = _get_from_dataset(dataset, 'labels')

    if returns_all is None:
        print("[WARN] 'returns' not found in dataset. Cannot plot prediction returns histogram.")
        return

    # ┏━━━━━━━━━━ Extract Data ━━━━━━━━━━┓
    fee = 0.20 # 0.2% round-trip fee
    returns = returns_all[indices].numpy() * 100.0
    labels = labels_all[indices].numpy()
    
    # ┏━━━━━━━━━━ Filter NaNs ━━━━━━━━━━┓
    valid = ~np.isnan(returns)
    labels = labels[valid]
    returns = returns[valid]
    
    # ┏━━━━━━━━━━ Ensure preds align with valid returns if dataset was filtered ━━━━━━━━━━┓
    if len(preds) == len(indices):
        preds = preds[valid]
    else:
        print("[WARN] Predictions length does not match indices length. Cannot plot prediction returns.")
        return

    # ┏━━━━━━━━━━ Segregate Returns ━━━━━━━━━━┓
    # Ground Truth: Actual labels matching interested class
    gt_returns = returns[labels == interested_class]
    
    # Predicted: Model predicted the interested class
    pred_returns = returns[preds == interested_class]
    
    # ┏━━━━━━━━━━ Setup Plot ━━━━━━━━━━┓
    fig, ax1 = plt.subplots(figsize=(12, 7))
    ax2 = ax1.twinx()
    
    # ┏━━━━━━━━━━ Dynamic Limits & Binning ━━━━━━━━━━┓
    low, high = get_dynamic_ret_limits([gt_returns, pred_returns])
    step = 0.1 if high <= 10 else 0.2
    bins = np.arange(low, high + step, step)
    
    # ┏━━━━━━━━━━ Plot Ground Truth (Background on Ax2 to avoid dwarfing predictions) ━━━━━━━━━━┓
    ax2.hist(gt_returns, 
             bins   = bins, 
             alpha  = 0.15, 
             color  = 'blue', 
             label  = f'Ground Truth Class {interested_class} (N={len(gt_returns):,}, Mean={np.mean(gt_returns):.2f}%)', 
             zorder = 1)
    ax2.set_ylabel("Number of Windows (Ground Truth)", color='blue', alpha=0.6)
    ax2.tick_params(axis='y', labelcolor='blue', colors='blue')
    
    # ┏━━━━━━━━━━ Plot Predictions (Foreground on Ax1) ━━━━━━━━━━┓
    overlap = np.sum((preds == interested_class) & (labels == interested_class))
    overlap_pct = (overlap / len(pred_returns) * 100.0) if len(pred_returns) > 0 else 0.0
    label_text = f'Predicted Class {interested_class} (N={len(pred_returns):,}, Mean={np.mean(pred_returns) if len(pred_returns)>0 else 0:.2f}%)\nOverlap (True Positives): {overlap:,} ({overlap_pct:.1f}%)'
    
    ax1.hist(pred_returns, 
             bins   = bins, 
             alpha  = 0.6, 
             color  = 'darkorange', 
             label  = label_text, 
             zorder = 2)
    ax1.set_ylabel("Number of Windows (Predicted)", color='darkorange')
    
    ax1.axvline(x=0, color='black', linestyle='--', alpha=0.5)
    ax1.axvline(x=fee, color='magenta', linestyle=':', alpha=0.7, label=f'Fee Break-Even (±{fee}%)')
    ax1.axvline(x=-fee, color='magenta', linestyle=':', alpha=0.7)
    
    if len(gt_returns) > 0:
        ax1.axvline(x=np.mean(gt_returns), color='navy', linestyle='-', alpha=0.8, label=f'Ground Truth Mean')
    if len(pred_returns) > 0:
        ax1.axvline(x=np.mean(pred_returns), color='orangered', linestyle='-', alpha=0.8, label=f'Predicted Mean')
    
    ax1.set_title(f"Prediction vs Ground-Truth Actual Returns {title_suffix}")
    ax1.set_xlabel("Actual Return (%)")
    ax1.set_xlim(low, high)
    
    # ┏━━━━━━━━━━ Combine legends ━━━━━━━━━━┓
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_2 + lines_1, labels_2 + labels_1, loc='upper left')
    
    ax1.grid(True, alpha=0.3)
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f"[plot_prediction_returns_histogram] Saved to {save_path}")



# ┏━━━━━━━━━━ Plot M1 Prediction Returns Histogram ━━━━━━━━━━┓
def plot_m1_prediction_returns_histogram(dataset: Dict[str, Any], 
                                         indices: List[int], 
                                         interested_class: int,
                                         save_path: str,
                                         meta_label_mode: str = "OG",
                                         split_name:      str = "val",
                                         direction:       str = "up",
                                         title_suffix:    str = ""):
    """
    Plots a two-subplot analysis for M1 predictions:
    1. Signal Quality: M1's actual returns vs Ground Truth actual returns.
    2. Calibration: Window-by-window scatter plot of M1's prediction vs Outcome.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import os
    from typing import Any

    returns_all = _get_from_dataset(dataset, 'returns')
    labels_all  = _get_from_dataset(dataset, 'labels')
    m1_pred_rets_all = _get_from_dataset(dataset, 'm1_pred_returns')
    m1_pred_labs_all = _get_from_dataset(dataset, 'm1_pred_labels')

    if returns_all is None or m1_pred_rets_all is None:
        print("[WARN] Required keys missing in dataset for M1 returns histogram.")
        return

    # ┏━━━━━━━━━━ Extract Data ━━━━━━━━━━┓
    fee = 0.20
    returns = returns_all[indices].numpy() * 100.0
    labels = labels_all[indices].numpy()
    m1_pred_returns = m1_pred_rets_all[indices].numpy() * 100.0
    m1_pred_labels = m1_pred_labs_all[indices].numpy()
    
    valid = ~np.isnan(returns) & ~np.isnan(m1_pred_returns)
    labels, returns, m1_pred_returns, m1_pred_labels = labels[valid], returns[valid], m1_pred_returns[valid], m1_pred_labels[valid]

    # ┏━━━━━━━━━━ Colors Setup ━━━━━━━━━━┓
    gt_color = 'green'
    if split_name.lower().startswith('val'):
        pred_color = 'darkorange'
    elif split_name.lower().startswith('test'):
        pred_color = 'dodgerblue'
    else:
        pred_color = 'purple'

    # ┏━━━━━━━━━━ Setup Data ━━━━━━━━━━┓
    # Ground Truth Reference (Actual labels matching interested class)
    gt_actual = returns[labels == interested_class]
    
    # M1 Signals (Actual returns when M1 signaled)
    m1_actual = returns[m1_pred_labels == interested_class] 

    # ┏━━━━━━━━━━ Plotting ━━━━━━━━━━┓
    fig = plt.figure(figsize=(12, 16))
    from matplotlib import gridspec
    gs = gridspec.GridSpec(2, 1, height_ratios=[1, 1], figure=fig)
    
    ax_b = fig.add_subplot(gs[0, 0])
    ax_c = fig.add_subplot(gs[1, 0])
    
    # ┏━━━━━━━━━━ Dynamic Limits & Binning ━━━━━━━━━━┓
    low, high = get_dynamic_ret_limits([gt_actual, m1_actual])
    step = 0.1 if high <= 10 else 0.2
    bins = np.arange(low, high + step, step)
    
    # ┏━━━━━━━━━━ Subplot 1: Signal Quality (Actual Returns) ━━━━━━━━━━┓
    ax_b2 = ax_b.twinx()
    ax_b2.hist(gt_actual, bins=bins, alpha=0.35, color=gt_color, label='Ground Truth Returns', zorder=1)
    
    # Calculate Hit Rates and Overlaps
    overlap = np.sum((m1_pred_labels == interested_class) & (labels == interested_class))
    overlap_pct = (overlap / len(m1_actual) * 100.0) if len(m1_actual) > 0 else 0.0
    
    if len(m1_actual) > 0:
        hit_rate = (m1_actual > 0).mean() * 100.0
        net_hit_rate = (m1_actual > fee).mean() * 100.0
        m1_label = f'Distribution Returns of M1 Predictions (N={len(m1_actual):,})\n' + \
                   f'Hit Rate (>0%): {hit_rate:.1f}% | Net Hit Rate (>{fee}%): {net_hit_rate:.1f}%\n' + \
                   f'Overlap (True Positives): {overlap:,} ({overlap_pct:.1f}%)'
    else:
        m1_label = f'Distribution Returns of M1 Predictions (N=0)\nOverlap (True Positives): {overlap:,} (0.0%)'

    ax_b.hist(m1_actual, bins=bins, alpha=0.6, color=pred_color, label=m1_label, zorder=2)
    
    ax_b.set_title(f"A: M1 {direction.upper()} Signal Realization | Predicted vs Ground Truth Returns Distribution ({meta_label_mode})")
    ax_b.set_ylabel("Number of Windows (Predicted M1 Return)", color=pred_color)
    ax_b2.set_ylabel("Number of Windows (Ground Truth)", color=gt_color, alpha=0.5)

    # ┏━━━━━━━━━━ Subplot 2: Window-by-Window Scatter ━━━━━━━━━━┓
    scatter_mask = (m1_pred_labels == interested_class)
    x_scatter = m1_pred_returns[scatter_mask]
    y_scatter = returns[scatter_mask]
    
    if len(x_scatter) > 1:
        corr = np.corrcoef(x_scatter, y_scatter)[0, 1]
        ax_c.scatter(x_scatter, y_scatter, alpha=0.3, color=pred_color, s=15, 
                     label=f'Window Pearson Correlation (N={len(x_scatter):,}, R={corr:.3f})')
        
        # Add y=x line
        lims = [
            np.min([ax_c.get_xlim(), ax_c.get_ylim()]),
            np.max([ax_c.get_xlim(), ax_c.get_ylim()]),
        ]
        ax_c.plot(lims, lims, 'k--', alpha=0.5, label='Perfect Calibration (y=x)')
        
        # Regression trendline
        m, b = np.polyfit(x_scatter, y_scatter, 1)
        ax_c.plot(x_scatter, m*x_scatter + b, color='red', alpha=0.8, linewidth=2, label=f'Trend (Slope={m:.2f})')

    ax_c.set_title(f"B: M1 {direction.upper()} Magnitude Calibration | Predicted vs. Ground Truth Market Return")
    ax_c.set_xlabel("M1 Predicted Return (%)")
    ax_c.set_ylabel("Ground Truth Return (%)")
    ax_c.legend(loc='upper left', fontsize='small')
    ax_c.grid(True, alpha=0.3)

    # ┏━━━━━━━━━━ Decorations & Fee Lines ━━━━━━━━━━┓
    ax_b.axvline(x=0, color='black', linestyle='--', alpha=0.3)
    ax_b.axvline(x=fee, color='magenta', linestyle=':', alpha=0.7, label=f'Fee (±{fee}%)')
    ax_b.axvline(x=-fee, color='magenta', linestyle=':', alpha=0.7)
    ax_b.set_xlabel("Return (%)")
    ax_b.set_xlim(low, high)
    ax_b.grid(True, alpha=0.2)
    
    # ┏━━━━━━━━━━ Scatter Axis limits zoom - dynamically centered around 0 with buffer ━━━━━━━━━━┓
    s_low, s_high = get_dynamic_ret_limits([x_scatter, y_scatter], min_buffer=5.0)
    ax_c.set_xlim(s_low, s_high)
    ax_c.set_ylim(s_low, s_high)

    # ┏━━━━━━━━━━ Combine legends for neatness ━━━━━━━━━━┓
    h3, l3 = ax_b.get_legend_handles_labels()
    h4, l4 = ax_b2.get_legend_handles_labels()
    ax_b.legend(h4 + h3, l4 + l3, loc='upper left', fontsize='small')

    # ┏━━━━━━━━━━ Title and Layout ━━━━━━━━━━┓
    fig.suptitle(f"M1 Performance Analysis & Calibration {title_suffix}", fontsize=18, fontweight='bold')
    plt.tight_layout(rect=[0, 0.02, 1, 0.96])
    
    # ┏━━━━━━━━━━ Save Figure ━━━━━━━━━━┓
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f"[plot_m1_prediction_returns_histogram] Saved Double-plot to {save_path}")



# ┏━━━━━━━━━━ Plot Confusion Matrix ━━━━━━━━━━┓
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



# ┏━━━━━━━━━━ Temporal Risk-Coverage Plot ━━━━━━━━━━┓
def plot_temporal_risk_coverage_curve(save_path: Path,
                                      curve: dict,
                                      probs: np.ndarray,
                                      y_true: np.ndarray,
                                      split_rets: np.ndarray,
                                      fee: float,
                                      op: dict,
                                      split_name: str,
                                      model_label: str,
                                      thres_mode: str,
                                      ocp_alpha: float,
                                      val_threshold: Optional[float] = None,
                                      val_op: Optional[dict] = None,
                                      is_ocp: bool = False,
                                      test_approved_ocp: Optional[np.ndarray] = None,
                                      direction: str = "",
                                      granularity: str = ""):
    """Plot risk-coverage with return overlays for temporal_eval."""
    thrs = curve["thresholds"]
    covs = curve["coverage"]
    risks_raw = curve["risk"]

    # ┏━━━━━━━━━━ Colors ━━━━━━━━━━┓
    c_risk     = "#1B4F72" # deep navy for risk curve
    c_ret      = "#1E8449" # forest green for positive mean return
    c_ret_neg  = "#8B0000" # dark red for negative mean return
    c_win      = "#1E8449" # lighter green for mean win
    c_op       = "#8B008B" # dark magenta for operating point
    c_grid     = "#D5D8DC" # subtle grey for grid
    c_thr05    = "#34495E" # darker grey for thr=0.5 line
    c_util_ref = "#E67E22" # orange for utility reference

    fig_rc, ax_rc = plt.subplots(figsize=(10, 6.5), facecolor="white")
    ax_rc.set_facecolor("#FAFAFA")

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

    ax_rc.plot(grid_cov, risk_smooth, color=c_risk, linewidth=2.2, label="Risk (Error Rate)", zorder=3)
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
    ax_rc.grid(True, which="major", color=c_grid, linewidth=0.6, alpha=0.7)
    ax_rc.set_axisbelow(True)

    mean_rets = np.full_like(thrs, np.nan)
    mean_win_rets = np.full_like(thrs, np.nan)
    mean_lose_rets = np.full_like(thrs, np.nan)
    for i, thr in enumerate(thrs):
        sel = probs >= thr
        if int(sel.sum()) >= 2:
            net = split_rets[sel] - fee
            labels = y_true[sel]
            mean_rets[i] = float(np.nanmean(net))
            winners = net[labels == 1]
            losers = net[labels == 0]
            if len(winners) >= 1:
                mean_win_rets[i] = float(np.nanmean(winners))
            if len(losers) >= 1:
                mean_lose_rets[i] = float(np.nanmean(losers))

    ax_ret = ax_rc.twinx()
    valid = ~np.isnan(mean_rets)
    valid_w = ~np.isnan(mean_win_rets)
    valid_l = ~np.isnan(mean_lose_rets)
    both_valid = valid_w & valid_l
    if both_valid.any():
        ax_ret.fill_between(covs[both_valid],
                            mean_win_rets[both_valid] * 100,
                            mean_lose_rets[both_valid] * 100,
                            alpha=0.06, color=c_win, zorder=1, label="_nolegend_")

    def _plot_dynamic_return(ax, x, y, lw, ls, alpha, label, zorder):
        if len(x) > 1:
            from matplotlib.collections import LineCollection
            points = np.array([x, y]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            y_mids = segments[:, :, 1].mean(axis=1)
            seg_colors = [c_ret if ym >= 0 else c_ret_neg for ym in y_mids]
            lc = LineCollection(segments, colors=seg_colors, linewidth=lw, linestyles=ls, alpha=alpha, zorder=zorder)
            ax.add_collection(lc)
            if label and label != "_nolegend_":
                ax.plot([], [], color=c_ret, linewidth=lw, linestyle=ls, label=label)
        elif len(x) == 1:
            color = c_ret if y[0] >= 0 else c_ret_neg
            ax.plot(x, y, color=color, linewidth=lw, linestyle=ls, alpha=alpha, label=label, zorder=zorder)

    _plot_dynamic_return(ax_ret, covs[valid], mean_rets[valid] * 100, 2.0, "-", 0.9, "Mean Return", 3)
    ax_ret.axhline(y=0, color=c_ret, linestyle=":", alpha=0.35, linewidth=1.0)
    ax_ret.set_ylabel("Return (%)", fontsize=11, fontweight="bold", color="black", labelpad=8)
    ax_ret.tick_params(axis="y", colors="black", labelcolor="black", labelsize=9, width=1.5)
    for spine in ax_ret.spines.values():
        spine.set_color("black")
        spine.set_linewidth(1.5)
    plt.setp(ax_ret.get_yticklabels(), fontweight="bold")

    idx_05 = np.argmin(np.abs(thrs - 0.5))
    cov_05 = covs[idx_05]
    risk_05 = risks_raw[idx_05]
    op_cov = op["coverage"]
    op_risk = op.get("risk", 0)
    thr_source = op.get("threshold_source") or ("OCP-SAOCP" if is_ocp else ("Val-Utility" if split_name == "Test" else "Utility-Opt"))
    show_baseline = abs(op_cov - cov_05) > 0.02 and abs(op["threshold"] - 0.5) > 0.01
    if show_baseline:
        ax_rc.axvline(x=cov_05, color=c_thr05, linestyle="--", alpha=0.7, linewidth=1.8)
        ax_rc.scatter([cov_05], [risk_05], color=c_thr05, marker="o", s=40, edgecolors="white", linewidths=1.0, zorder=5)
        ax_rc.annotate("τ=0.50", xy=(cov_05, risk_05), xytext=(3, 5), textcoords="offset points",
                       fontsize=7, color=c_thr05, fontweight="bold", zorder=10,
                       bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=c_thr05, alpha=0.8, lw=0.6))

    ax_rc.axvline(x=op_cov, color=c_op, linestyle="--", alpha=0.7, linewidth=1.8)
    ax_rc.scatter([op_cov], [op_risk], color=c_op, marker="D", s=40, edgecolors="white", linewidths=1.0, zorder=6)
    ax_rc.annotate(f"$\\hat{{\\tau}}$={op['threshold']:.3f}", xy=(op_cov, op_risk), xytext=(3, 6),
                   textcoords="offset points", fontsize=7.5, color=c_op, fontweight="bold", zorder=10,
                   bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=c_op, alpha=0.85, lw=0.6))

    mr_val = op["mean_ret"] * 100
    sel_op = test_approved_ocp if split_name == "Test" and is_ocp and test_approved_ocp is not None else (probs >= op["threshold"])
    n_op = int(sel_op.sum())
    if n_op >= 2:
        net_op = split_rets[sel_op] - fee
        lab_op = y_true[sel_op]
        w_op = net_op[lab_op == 1]
        l_op = net_op[lab_op == 0]
        mw_val = float(np.nanmean(w_op)) * 100 if len(w_op) >= 1 else None
        ml_val = float(np.nanmean(l_op)) * 100 if len(l_op) >= 1 else None
    else:
        mw_val, ml_val = None, None

    mr_05 = mean_rets[idx_05] * 100 if not np.isnan(mean_rets[idx_05]) else None
    mw_05 = mean_win_rets[idx_05] * 100 if not np.isnan(mean_win_rets[idx_05]) else None
    ml_05 = mean_lose_rets[idx_05] * 100 if not np.isnan(mean_lose_rets[idx_05]) else None

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

    if show_baseline:
        offs_05 = _get_staggered_offsets({"mw": mw_05, "ml": ml_05, "mr": mr_05})
        for key, value in {"mw": mw_05, "ml": ml_05, "mr": mr_05}.items():
            if value is None:
                continue
            color = c_ret if value >= 0 else c_ret_neg
            ax_ret.scatter([cov_05], [value], color=color, marker="o", s=35 if key == "mr" else 25,
                           edgecolors="white", linewidths=0.8 if key == "mr" else 0.6, zorder=5)
            ax_ret.annotate(f"{value:+.2f}%", xy=(cov_05, value), xytext=offs_05[key],
                            textcoords="offset points", fontsize=7.5, color=color,
                            fontweight="bold", zorder=10)

    offs_op = _get_staggered_offsets({"mw": mw_val, "ml": ml_val, "mr": mr_val})
    for key, value in {"mr": mr_val, "mw": mw_val, "ml": ml_val}.items():
        if value is None:
            continue
        color = c_ret if value >= 0 else c_ret_neg
        ax_ret.scatter([op_cov], [value], color=color, marker="D" if key == "mr" else "o",
                       s=40 if key == "mr" else 35,
                       edgecolors="white", linewidths=1.0 if key == "mr" else 0.8,
                       zorder=7 if key == "mr" else 6)
        ax_ret.annotate(f"{value:+.2f}%", xy=(op_cov, value), xytext=offs_op[key],
                        textcoords="offset points", fontsize=7.5, color=color,
                        fontweight="bold", zorder=10)

    util_ref_plotted = False
    if split_name == "Test" and is_ocp and val_op is not None and val_threshold is not None and val_op.get("constraint_satisfied", False):
        util_sel = probs >= val_threshold
        util_n = int(util_sel.sum())
        util_cov = util_n / len(y_true) if len(y_true) > 0 else 0
        util_risk = int((y_true[util_sel] == 0).sum()) / max(util_n, 1) if util_n > 0 else 0
        if util_n >= 2:
            util_net = split_rets[util_sel] - fee
            util_lab = y_true[util_sel]
            util_mr = float(np.nanmean(util_net)) * 100
            util_w = util_net[util_lab == 1]
            util_l = util_net[util_lab == 0]
            util_mw = float(np.nanmean(util_w)) * 100 if len(util_w) >= 1 else None
            util_ml = float(np.nanmean(util_l)) * 100 if len(util_l) >= 1 else None
        else:
            util_mr, util_mw, util_ml = 0, None, None

        ax_rc.axvline(x=util_cov, color=c_util_ref, linestyle="--", alpha=0.7, linewidth=1.5)
        ax_rc.scatter([util_cov], [util_risk], color=c_util_ref, marker="D", s=45, edgecolors="white", linewidths=1.0, zorder=6)
        ax_rc.annotate(f"τ_util={val_threshold:.3f}", xy=(util_cov, util_risk), xytext=(5, -12),
                       textcoords="offset points", fontsize=7.5, color=c_util_ref, fontweight="bold", zorder=10,
                       bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=c_util_ref, alpha=0.85, lw=0.6))
        util_offs = _get_staggered_offsets({"mw": util_mw, "ml": util_ml, "mr": util_mr})
        for key, value in {"mr": util_mr, "mw": util_mw, "ml": util_ml}.items():
            if value is None:
                continue
            ax_ret.scatter([util_cov], [value], color=c_util_ref, marker="s" if key == "mr" else "o",
                           s=35 if key == "mr" else 25,
                           edgecolors="white", linewidths=0.8 if key == "mr" else 0.6,
                           zorder=6 if key == "mr" else 5)
            ax_ret.annotate(f"{value:+.2f}%", xy=(util_cov, value), xytext=util_offs.get(key, (6, 0)),
                            textcoords="offset points", fontsize=7.5, color=c_util_ref,
                            fontweight="bold", zorder=10)
        util_ref_plotted = True

    _split_display = {"Val": "Validation", "val": "Validation"}.get(split_name, split_name)
    _dir_gran = f"  |  {direction.upper()}  {granularity}" if direction or granularity else ""
    ax_rc.set_title(f"Risk-Profitability  |  {_split_display}  |  {model_label}{_dir_gran}",
                    fontsize=13, fontweight="bold", color="#2C3E50", pad=12)
    fig_rc.tight_layout()
    fig_rc.subplots_adjust(bottom=0.22 + (0.05 if util_ref_plotted else 0.0))

    from matplotlib.lines import Line2D
    legend_prop = {"size": 8}
    row1_handles, row1_labels = [], []
    if mw_val is not None:
        color = c_ret if mw_val >= 0 else c_ret_neg
        row1_handles.append(Line2D([], [], color=color, linewidth=1.0, linestyle=":", marker="o", markersize=4, markeredgecolor="white", markeredgewidth=0.6))
        row1_labels.append(f"Avg Win = {mw_val:+.2f}%")
    if ml_val is not None:
        color = c_ret if ml_val >= 0 else c_ret_neg
        row1_handles.append(Line2D([], [], color=color, linewidth=1.0, linestyle=":", marker="o", markersize=4, markeredgecolor="white", markeredgewidth=0.6))
        row1_labels.append(f"Avg Loss = {ml_val:+.2f}%")
    row1_handles.append(Line2D([], [], color=c_op, marker="D", markersize=5, linestyle="--", linewidth=1.5, alpha=0.8, markeredgecolor="white", markeredgewidth=0.8))
    row1_labels.append(f"τ̂ = {op['threshold']:.3f}  ({thr_source})   Cov = {op['coverage']:.1%}   N = {op['selected_count']}   t = {op['t_stat']:.1f}")
    fig_rc.legend(row1_handles, row1_labels,
                  loc="lower center", ncol=len(row1_handles), prop=legend_prop,
                  frameon=True, framealpha=0.92, edgecolor="none", fancybox=True,
                  bbox_to_anchor=(0.5, 0.065 + (0.05 if util_ref_plotted else 0.0)),
                  handlelength=2.5, handletextpad=0.8)

    row2_handles = [Line2D([], [], color=c_risk, linewidth=2.2),
                    Line2D([], [], color=c_ret if op["mean_ret"] >= 0 else c_ret_neg, linewidth=2.0)]
    row2_labels = [f"Risk = {op.get('risk', 0):.1%}", f"Mean Ret = {op['mean_ret']*100:+.2f}%"]
    fig_rc.legend(row2_handles, row2_labels,
                  loc="lower center", ncol=2, prop=legend_prop,
                  frameon=True, framealpha=0.92, edgecolor="none", fancybox=True,
                  bbox_to_anchor=(0.5, 0.015 + (0.05 if util_ref_plotted else 0.0)),
                  handlelength=2.5, handletextpad=0.8)

    if util_ref_plotted:
        row3_handles = [Line2D([], [], color=c_util_ref, marker="D", markersize=5, linestyle="--", linewidth=1.5, alpha=0.8, markeredgecolor="white", markeredgewidth=0.8)]
        row3_labels = [f"τ_util = {val_threshold:.3f}  (Val-Utility)   Cov = {util_cov:.1%}   N = {util_n}   Risk = {util_risk:.1%}   MeanRet = {util_mr:+.2f}%"]
        fig_rc.legend(row3_handles, row3_labels,
                      loc="lower center", ncol=1, prop=legend_prop,
                      frameon=True, framealpha=0.92, edgecolor="none", fancybox=True,
                      bbox_to_anchor=(0.5, 0.015), handlelength=2.5, handletextpad=0.8)

    fig_rc.savefig(str(save_path), dpi=200, facecolor="white")
    plt.close(fig_rc)

import matplotlib.scale as mscale
import matplotlib.transforms as mtransforms

class PiecewiseLinearScale(mscale.ScaleBase):
    name = 'piecewise_linear'

    def __init__(self, axis, **kwargs):
        super().__init__(axis)
        self.x_nodes = kwargs.pop('x_nodes', [0.0, 0.5, 1.0])
        self.y_nodes = kwargs.pop('y_nodes', [0.0, 0.5, 1.0])

    def get_transform(self):
        return self.PiecewiseTransform(self.x_nodes, self.y_nodes)

    def set_default_locators_and_formatters(self, axis):
        pass

    class PiecewiseTransform(mtransforms.Transform):
        input_dims = 1
        output_dims = 1
        is_separable = True
        has_inverse = True

        def __init__(self, x_nodes, y_nodes):
            mtransforms.Transform.__init__(self)
            self.x_nodes = np.array(x_nodes, dtype=float)
            self.y_nodes = np.array(y_nodes, dtype=float)

        def transform_non_affine(self, a):
            return np.interp(a, self.x_nodes, self.y_nodes)

        def inverted(self):
            return PiecewiseLinearScale.InvertedPiecewiseTransform(self.x_nodes, self.y_nodes)

    class InvertedPiecewiseTransform(mtransforms.Transform):
        input_dims = 1
        output_dims = 1
        is_separable = True
        has_inverse = True

        def __init__(self, x_nodes, y_nodes):
            mtransforms.Transform.__init__(self)
            self.x_nodes = np.array(x_nodes, dtype=float)
            self.y_nodes = np.array(y_nodes, dtype=float)

        def transform_non_affine(self, a):
            return np.interp(a, self.y_nodes, self.x_nodes)

        def inverted(self):
            return PiecewiseLinearScale.PiecewiseTransform(self.x_nodes, self.y_nodes)

mscale.register_scale(PiecewiseLinearScale)

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
                                            ocp_alpha: float,
                                            val_threshold: Optional[float] = None,
                                            val_op: Optional[dict] = None,
                                            is_ocp: bool = False,
                                            test_approved_ocp: Optional[np.ndarray] = None,
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
            special_labels[cov_min] += "\n" + r"$\hat{\tau}$" + f"({op_cov_tmp:.2f})"
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
    thr_source = op.get("threshold_source") or ("OCP-SAOCP" if is_ocp else ("Val-Utility" if split_name == "Test" else "Utility-Opt"))
    show_baseline = abs(op_cov - cov_05) > 0.02 and abs(op["threshold"] - 0.5) > 0.01
    if show_baseline:
        ax_rc.axvline(x=cov_05, color=c_thr05, linestyle="--", alpha=0.7, linewidth=1.8)
        ax_rc.scatter([cov_05], [risk_05], color=c_thr05, marker="o", s=40,
                      edgecolors="white", linewidths=1.0, zorder=5)

    aop = _zone_alpha(op_cov)  # kept for axvline only
    ax_rc.axvline(x=op_cov, color=c_op, linestyle="--", alpha=0.7, linewidth=1.8)
    ax_rc.scatter([op_cov], [op_risk], color=c_op, marker="D", s=50,
                  edgecolors="white", linewidths=1.0, zorder=6)
    ax_rc.annotate(f"$\\hat{{\\tau}}$={op['threshold']:.3f}", xy=(op_cov, op_risk), xytext=(3, 6),
                   textcoords="offset points", fontsize=12, color=c_op, fontweight="bold", zorder=10,
                   bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=c_op, alpha=0.9, lw=0.6))

    # Return annotations at τ̂
    mr_val = op["mean_ret"] * 100
    # Use the optimizer's dataset (opt_probs/opt_rets/opt_y) when provided so that
    # mr/mw/ml annotations match op["mean_ret"] exactly (same N, same population).
    _ann_probs  = _u_probs   if opt_probs is not None else probs
    _ann_rets   = _u_rets    if opt_rets  is not None else split_rets
    _ann_labels = _u_y       if opt_y     is not None else labels_int
    sel_op = test_approved_ocp if split_name == "Test" and is_ocp and test_approved_ocp is not None else (_ann_probs >= op["threshold"])
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
               label=rf"$\hat{{\tau}}$={op['threshold']:.3f}, {_mut_str} (ii)"),
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


# ┏━━━━━━━━━━ OCP Threshold Evolution ━━━━━━━━━━┓
def plot_ocp_threshold_evolution(save_path: Path,
                                 test_s_hats: np.ndarray,
                                 utility_threshold: float,
                                 model_label: str,
                                 thres_mode: str,
                                 ocp_alpha: float,
                                 conformal_coverage: float,
                                 n_set_1: int,
                                 n_set_0: int,
                                 n_set_both: int,
                                 n_set_empty: int,
                                 split_name: str = "Test"):
    """Plot OCP threshold evolution for temporal_eval."""
    fig_thr, ax_thr = plt.subplots(figsize=(10, 4), facecolor="white")
    ax_thr.set_facecolor("#FAFAFA")
    eff_tau = np.maximum(test_s_hats, 1.0 - test_s_hats)
    ax_thr.plot(eff_tau, color="#8B008B", linewidth=0.8, alpha=0.9, label=f"τ_t ({thres_mode})")
    ax_thr.axhline(y=utility_threshold, color="#34495E", linestyle="--", linewidth=1.2, alpha=0.7, label=f"τ Utility = {utility_threshold:.3f}")
    ax_thr.axhline(y=0.5, color="#BDC3C7", linestyle=":", linewidth=0.8, alpha=0.6)
    ax_thr.set_xlabel("Test sample index", fontsize=10)
    ax_thr.set_ylabel("Threshold τ_t", fontsize=10)
    ax_thr.set_title(f"{thres_mode} Threshold Evolution  |  {split_name}  |  {model_label}  (α={ocp_alpha})\n"
                     f"Conformal Cov={conformal_coverage:.1%} (target≥{1-ocp_alpha:.0%})  |  "
                     f"{{1}}={n_set_1}  {{0}}={n_set_0}  {{0,1}}={n_set_both}  {{}}={n_set_empty}",
                     fontsize=11, fontweight="bold", color="#2C3E50")
    ax_thr.legend(fontsize=8, loc="upper right")
    ax_thr.set_ylim(0.4, 1.0)
    ax_thr.grid(True, alpha=0.3)
    fig_thr.tight_layout()
    fig_thr.savefig(str(save_path), dpi=200, facecolor="white")
    plt.close(fig_thr)


# ┏━━━━━━━━━━ M2 Selective Return Distribution (TP/FP vs M1) ━━━━━━━━━━┓
def plot_selective_return_distribution(test_returns: np.ndarray,
                                       test_labels: np.ndarray,
                                       m2_approved: np.ndarray,
                                       save_path,
                                       fee: float = 0.002,
                                       direction: str = "up",
                                       granularity: str = "1d",
                                       model_label: str = "RF"):
    """Histogram of M2-selected trade returns split into TPs (green) and FPs (red),
    with the full M1 distribution shown as a background layer in cyan.

    Parameters
    ----------
    test_returns : np.ndarray
        Net returns (already fee-adjusted and direction-flipped) for the test set.
    test_labels : np.ndarray
        Binary ground-truth labels (1 = TP, 0 = FP) for the test set.
    m2_approved : np.ndarray (bool)
        Boolean mask of M2-approved trades.
    save_path : str or Path
        Where to write the PNG.
    fee : float
        Fee per trade (for display only — returns should already be net).
    direction : str
        "up" or "down".
    granularity : str
        Granularity label for the title.
    model_label : str
        M2 model label (e.g. "RF", "XGB").
    """
    import matplotlib.patches as mpatches

    # ┏━━━━━━━━━━ Convert to numpy arrays with canonical dtypes ━━━━━━━━━━┓
    test_returns = np.asarray(test_returns, dtype=float)
    test_labels  = np.asarray(test_labels, dtype=int)
    m2_approved  = np.asarray(m2_approved, dtype=bool)

    # ┏━━━━━━━━━━ Separate M2-approved trades into TPs and FPs ━━━━━━━━━━┓
    sel_mask   = m2_approved
    sel_rets   = test_returns[sel_mask] * 100
    sel_labels = test_labels[sel_mask]
    tp_rets    = sel_rets[sel_labels == 1]
    fp_rets    = sel_rets[sel_labels == 0]

    # ┏━━━━━━━━━━ Full M1 distribution (all trades, approved and rejected) ━━━━━━━━━━┓
    m1_rets = test_returns * 100

    # ┏━━━━━━━━━━ Dynamic bin width based on the observed return range ━━━━━━━━━━┓
    all_combined = np.concatenate([m1_rets, sel_rets]) if len(sel_rets) > 0 else m1_rets
    low  = np.percentile(all_combined, 1) if len(all_combined) > 0 else -10
    high = np.percentile(all_combined, 99) if len(all_combined) > 0 else 10
    step = 0.15 if (high - low) < 20 else 0.3
    bins = np.arange(low, high + step, step)

    fig, ax = plt.subplots(figsize=(13, 7))

    # ┏━━━━━━━━━━ Layer 1: M1 full distribution (navy, background) ━━━━━━━━━━┓
    ax.hist(m1_rets, bins=bins, alpha=0.25, color="navy", edgecolor="midnightblue",
            linewidth=0.4, label=f"M1 Complete Test (N={len(m1_rets):,}, "
                                 f"\u03BC={np.mean(m1_rets):.2f}%)", zorder=1)

    # ┏━━━━━━━━━━ Layer 2: M2 FPs (red, foreground) ━━━━━━━━━━┓
    if len(fp_rets) > 0:
        ax.hist(fp_rets, bins=bins, alpha=0.65, color="#e74c3c", edgecolor="#c0392b",
                linewidth=0.4, label=f"M2 FP (n={len(fp_rets):,}, "
                                     f"\u03BC={np.mean(fp_rets):.2f}%)", zorder=2)

    # ┏━━━━━━━━━━ Layer 3: M2 TPs (green, foreground) ━━━━━━━━━━┓
    if len(tp_rets) > 0:
        ax.hist(tp_rets, bins=bins, alpha=0.65, color="#2ecc71", edgecolor="#27ae60",
                linewidth=0.4, label=f"M2 TP (n={len(tp_rets):,}, "
                                     f"\u03BC={np.mean(tp_rets):.2f}%)", zorder=3)

    # ┏━━━━━━━━━━ Mean lines ━━━━━━━━━━┓
    if len(m1_rets) > 0:
        ax.axvline(np.mean(m1_rets), color="navy", linestyle="--", linewidth=1.5,
                   alpha=0.8, label=f"M1 Mean ({np.mean(m1_rets):.2f}%)")
    if len(tp_rets) > 0:
        ax.axvline(np.mean(tp_rets), color="#27ae60", linestyle="-", linewidth=2.0,
                   alpha=0.9, label=f"Mean TP ({np.mean(tp_rets):.2f}%)")
    if len(fp_rets) > 0:
        ax.axvline(np.mean(fp_rets), color="#c0392b", linestyle="-", linewidth=2.0,
                   alpha=0.9, label=f"Mean FP ({np.mean(fp_rets):.2f}%)")

    # ┏━━━━━━━━━━ Zero and fee lines ━━━━━━━━━━┓
    ax.axvline(0, color="black", linestyle="--", alpha=0.4, linewidth=0.8)
    fee_pct = fee * 100
    ax.axvline(fee_pct,  color="#8B008B", linestyle="--", alpha=0.8, linewidth=1.8,
               label=f"Fee break-even (\u00B1{fee_pct:.2f}%)")
    ax.axvline(-fee_pct, color="#8B008B", linestyle="--", alpha=0.8, linewidth=1.8)

    # ┏━━━━━━━━━━ Title and labels ━━━━━━━━━━┓
    n_sel = int(sel_mask.sum())
    cov = n_sel / len(test_returns) * 100 if len(test_returns) > 0 else 0
    prec = len(tp_rets) / n_sel * 100 if n_sel > 0 else 0
    ax.set_title(f"M2 {model_label} Selective Return Distribution — "
                 f"{granularity.upper()} {direction.upper()} TP\n"
                 f"Coverage: {cov:.1f}% ({n_sel:,}/{len(test_returns):,})  |  "
                 f"Precision: {prec:.1f}%",
                 fontsize=12, fontweight="bold")
    ax.set_xlabel("Return (%)")
    ax.set_ylabel("Number of Trades")
    ax.set_xlim(low, high)
    ax.legend(loc="upper left", fontsize=9, framealpha=0.9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(save_path), bbox_inches="tight", dpi=200)
    plt.close(fig)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CROSS-ASSET CORRELATION PLOTS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def plot_asset_correlation(pearson_corr:  pd.DataFrame,
                           spearman_corr: pd.DataFrame,
                           lag_matrix:    pd.DataFrame,
                           pivot:         pd.DataFrame,
                           sig:           dict,
                           save_dir:      Path,
                           gran:          str = "",
                           direction:     str = "") -> None:
    """Produce four publication-quality figures for the cross-asset correlation analysis.

    Figures saved
    -------------
    1. ``asset_corr_pearson.png``   — annotated Pearson correlation heatmap
    2. ``asset_corr_clustermap.png`` — seaborn clustermap (hierarchical clustering)
    3. ``asset_corr_lag.png``       — lead-lag matrix heatmap (peak cross-corr lag)
    4. ``asset_corr_rolling.png``   — rolling 30-bar mean pairwise correlation

    Parameters
    ----------
    pearson_corr : pd.DataFrame
        (AxA) Pearson correlation matrix.
    spearman_corr : pd.DataFrame
        (AxA) Spearman correlation matrix.
    lag_matrix : pd.DataFrame
        (AxA) peak cross-correlation lag matrix (integer bars).
    pivot : pd.DataFrame
        (TxA) return matrix used to build the rolling plot.
    sig : dict
        Output of ``_permutation_significance`` — used for subtitle annotation.
    save_dir : Path
        Directory to write PNG files.
    gran : str
        Granularity label for titles.
    direction : str
        Direction label for titles.
    """
    save_dir = Path(save_dir)
    tag = f"{gran} {direction}".strip() or "all"
    A   = pearson_corr.shape[0]

    # ┏━━━━━━━━━━ Figure 1 — Pearson + Spearman heatmaps (side by side) ━━━━━━━━━━┓
    fig, axes = plt.subplots(1, 2, figsize=(max(14, A * 0.9), max(6, A * 0.7)))

    for ax, corr, title in zip(axes, [pearson_corr, spearman_corr], ["Pearson", "Spearman"]):
        # ┏━━━━━━━━━━ Render correlation matrix as coloured image ━━━━━━━━━━┓
        im = ax.imshow(corr.values, vmin=-1, vmax=1, cmap="RdYlGn", aspect="auto")
        ax.set_xticks(range(A))
        ax.set_yticks(range(A))
        ax.set_xticklabels(corr.columns, rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(corr.index, fontsize=8)
        # ┏━━━━━━━━━━ Annotate each cell with its correlation value ━━━━━━━━━━┓
        for i in range(A):
            for j in range(A):
                v = corr.values[i, j]
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        fontsize=6, color="black" if abs(v) < 0.7 else "white")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title(f"{title} Correlation — {tag}", fontsize=11, fontweight="bold")

    # ┏━━━━━━━━━━ Shared suptitle with permutation-test digest ━━━━━━━━━━┓
    mean_r  = sig["observed_mean_r"]
    p_value = sig["p_value"]
    z_score = sig["z_score"]
    fig.suptitle(
        f"Cross-Asset Return Correlation  |  mean Pearson r = {mean_r:+.3f}  "
        f"p = {p_value:.4f}  z = {z_score:.1f}",
        fontsize=10, y=1.01,
    )
    plt.tight_layout()
    fig.savefig(save_dir / "asset_corr_pearson.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ┏━━━━━━━━━━ Figure 2 — Hierarchical clustermap (Ward linkage) ━━━━━━━━━━┓
    cg = sns.clustermap(pearson_corr,
                        cmap  = "RdYlGn",
                        vmin  = -1,
                        vmax  = 1,
                        annot = A <= 25,
                        fmt=".2f",
                        annot_kws={"size": 6},
                        linewidths=0.3,
                        figsize=(max(10, A * 0.8), max(8, A * 0.8)))
    cg.ax_heatmap.set_title(f"Clustered Pearson Correlation — {tag}\n"
                            f"(Ward linkage on correlation distance)",
                            fontsize=10, fontweight="bold", pad=12)
    cg.figure.savefig(save_dir / "asset_corr_clustermap.png", dpi=150, bbox_inches="tight")
    plt.close(cg.figure)

    # ┏━━━━━━━━━━ Figure 3 — Lead-lag matrix ━━━━━━━━━━┓
    fig, ax = plt.subplots(figsize=(max(8, A * 0.7), max(6, A * 0.6)))
    abs_max = max(abs(lag_matrix.values.min()), abs(lag_matrix.values.max()), 1)
    im = ax.imshow(lag_matrix.values, vmin=-abs_max, vmax=abs_max, cmap="coolwarm", aspect="auto")
    ax.set_xticks(range(A))
    ax.set_yticks(range(A))
    ax.set_xticklabels(lag_matrix.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(lag_matrix.index, fontsize=8)
    # ┏━━━━━━━━━━ Annotate each cell with its integer lag value ━━━━━━━━━━┓
    for i in range(A):
        for j in range(A):
            v = int(lag_matrix.values[i, j])
            ax.text(j, i, str(v), ha="center", va="center", fontsize=7,
                    color="black" if abs(v) < abs_max * 0.6 else "white")
    plt.colorbar(im, ax=ax, label="Lag (bars, + = row leads column)", fraction=0.046, pad=0.04)
    ax.set_xlabel("Target asset (j)")
    ax.set_ylabel("Leading asset (i)")
    ax.set_title(f"Peak Cross-Correlation Lag — {tag}", fontsize=11, fontweight="bold")
    plt.tight_layout()
    fig.savefig(save_dir / "asset_corr_lag.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ┏━━━━━━━━━━ Figure 4 — Rolling mean pairwise correlation ━━━━━━━━━━┓
    window = min(30, max(5, len(pivot) // 10))
    ret    = pivot.fillna(0.0)
    T      = len(ret)
    dates  = ret.index.tolist()

    # ┏━━━━━━━━━━ Compute rolling mean off-diagonal correlation ━━━━━━━━━━┓
    off_diag       = ~np.eye(A, dtype=bool)
    rolling_mean_r = []
    for t in range(window, T + 1):
        chunk = ret.iloc[t - window:t].values
        c = np.corrcoef(chunk.T)
        rolling_mean_r.append(float(c[off_diag].mean()))

    roll_dates = dates[window - 1:]

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(roll_dates, rolling_mean_r, color="#2b5797", lw=1.5, label=f"Rolling {window}-bar mean r")
    ax.axhline(mean_r, color="red", ls="--", lw=1, alpha=0.7, label=f"Full-period mean r={mean_r:+.3f}")
    ax.axhline(0, color="black", ls="-", lw=0.5, alpha=0.3)
    
    # ┏━━━━━━━━━━ Shade positive and negative correlation regimes ━━━━━━━━━━┓
    ax.fill_between(roll_dates, rolling_mean_r, 0,
                    where=[r > 0  for r in rolling_mean_r], alpha=0.15, color="green", label="Positive regime")
    ax.fill_between(roll_dates, rolling_mean_r, 0,
                    where=[r <= 0 for r in rolling_mean_r], alpha=0.15, color="red",   label="Negative regime")
    ax.set_xlabel("Date")
    ax.set_ylabel("Mean pairwise Pearson r")
    ax.set_title(f"Rolling Cross-Asset Correlation — {tag}  "
                 f"(window={window} bars, p={p_value:.4f})",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(save_dir / "asset_corr_rolling.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Dataset size & TP/FP distribution across M1 models × granularities × directions
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def plot_dataset_size_distribution(cache_roots: Optional[Dict[str, Path]] = None,
                                   save_dir: Optional[Path] = None) -> pd.DataFrame:
    """Summarise and plot TP vs FP window counts per (M1 model, granularity, direction).

    For every MultiGranDataset cache in the provided roots, the binary
    `meta_label` is interpreted as:
        label == 1  →  True Positive  (M1 prediction aligned with realised direction)
        label == 0  →  False Positive (M1 prediction contradicted by outcome)

      1. ``size_stacked_counts.png``    — grouped-and-stacked bars of absolute TP/FP
                                          counts per granularity, split by direction,
                                          one panel per M1 model.
      2. ``size_tp_rate.png``           — TP-rate (base-rate) heatmap across models
                                          × granularities × directions.
      3. ``size_total_windows.png``     — total window count per model/gran/direction
                                          (log scale) — dataset-size comparison.

    Returns the long-format DataFrame of counts for downstream use.
    """

    # ┏━━━━━━━━━━ Default cache roots ━━━━━━━━━━┓
    if cache_roots is None:
        base = Path(__file__).resolve().parents[2] / "Output"
        cache_roots = {"Kronos":   base / "Kronos"   / "cache",
                       "Fincast":  base / "Fincast"  / "cache",
                       "Chronos2": base / "Chronos2" / "cache",
                       "Tirex":    base / "Tirex"    / "cache"}

    # ┏━━━━━━━━━━ Default save dir ━━━━━━━━━━┓
    if save_dir is None:
        save_dir = Path(__file__).resolve().parents[2] / "Output" / "Analysis" / "Size"
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # ┏━━━━━━━━━━ Direction extractor ━━━━━━━━━━┓
    _DIR_RE = re.compile(r"_fee_(up|down)_", re.IGNORECASE)
    def _infer_direction(p: Path) -> Optional[str]:
        m = _DIR_RE.search(p.name)
        return m.group(1).lower() if m else None

    # ┏━━━━━━━━━━ Collect counts ━━━━━━━━━━┓
    rows: List[Dict[str, Any]] = []
    for m1_name, cdir in cache_roots.items():
        cdir = Path(cdir)
        if not cdir.is_dir():
            warnings.warn(f"[size-plot] Cache dir missing for {m1_name}: {cdir}")
            continue
        for pt in sorted(cdir.glob("*.pt")):
            direction = _infer_direction(pt)
            if direction is None:
                continue
            try:
                ds = torch.load(pt, weights_only=False, map_location="cpu")
            except Exception as e:
                warnings.warn(f"[size-plot] Failed to load {pt.name}: {e}")
                continue
            grans = getattr(ds, "grans", None) or (list(ds.keys()) if isinstance(ds, dict) else None)
            if not grans:
                warnings.warn(f"[size-plot] No grans found in {pt.name}")
                continue
            for g in grans:
                sub = ds.sub[g] if hasattr(ds, "sub") else ds[g]
                labels = sub["labels"] if isinstance(sub, dict) else getattr(sub, "labels", None)
                if labels is None:
                    continue
                lab = np.asarray(labels.cpu() if hasattr(labels, "cpu") else labels).ravel()
                lab = lab[~np.isnan(lab)] if lab.dtype.kind == "f" else lab
                lab = lab.astype(int)
                n_tp = int((lab == 1).sum())
                n_fp = int((lab == 0).sum())
                rows.append({"m1_model":    m1_name,
                             "granularity": g,
                             "direction":   direction.upper(),
                             "n_tp":        n_tp,
                             "n_fp":        n_fp,
                             "n_total":     n_tp + n_fp,
                             "tp_rate":     n_tp / max(n_tp + n_fp, 1)})

    if not rows:
        warnings.warn("[size-plot] No data collected — aborting.")
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    # ┏━━━━━━━━━━ Persist raw counts ━━━━━━━━━━┓
    df.sort_values(["m1_model", "direction", "granularity"]).to_csv(save_dir / "size_counts.csv",
                                                                    index=False)

    # ┏━━━━━━━━━━ Ordering helpers (fixed canonical order, no 15m) ━━━━━━━━━━┓
    CANONICAL_GRANS = ["30m", "1h", "2h", "4h", "6h", "8h", "12h", "1d"]
    grans_present = [g for g in CANONICAL_GRANS if g in df["granularity"].unique().tolist()]
    # Filter dataframe to only those grans (drops 15m and anything else outside canonical set)
    df = df[df["granularity"].isin(grans_present)].copy()
    
    # Order models exactly as requested rather than strictly alphabetically
    _found = df["m1_model"].unique().tolist()
    pref_order = ["Chronos2", "Tirex", "Fincast", "Kronos"]
    models_present = [m for m in pref_order if m in _found] + sorted([m for m in _found if m not in pref_order])
    directions = ["UP", "DOWN"]

    # ┏━━━━━━━━━━ Academic style ━━━━━━━━━━┓
    plt.rcParams.update({"font.family":      "DejaVu Sans",
                         "axes.titlesize":   11,
                         "axes.labelsize":   10,
                         "xtick.labelsize":  9,
                         "ytick.labelsize":  9,
                         "legend.fontsize":  9,
                         "axes.spines.top":   False,
                         "axes.spines.right": False})

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Figure 1 — grouped stacked bars: TP/FP counts per gran × direction
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # ┏━━━━━━━━━━ Set up figure ━━━━━━━━━━┓
    n_models = len(models_present)
    fig, axes = plt.subplots(nrows   = n_models,
                             ncols   = 1,
                             figsize = (max(9, 0.8 * len(grans_present) * 2 + 2), 3.0 * n_models),
                             sharex  = True)
    if n_models == 1:
        axes = [axes]

    # ┏━━━━━━━━━━ Colors and bar width ━━━━━━━━━━┓
    TP_COLOR = "#2E7D32"
    FP_COLOR = "#C62828"
    bar_w = 0.38
    x = np.arange(len(grans_present))

    # ┏━━━━━━━━━━ Plot bars ━━━━━━━━━━┓
    for ax, m1 in zip(axes, models_present):
        sub = df[df["m1_model"] == m1]
        for i, d in enumerate(directions):
            offset = (i - 0.5) * bar_w
            sub_d = sub[sub["direction"] == d].set_index("granularity").reindex(grans_present)
            n_fp = sub_d["n_fp"].fillna(0).to_numpy()
            n_tp = sub_d["n_tp"].fillna(0).to_numpy()
            hatch = "" if d == "UP" else "///"
            ax.bar(x + offset, n_fp, bar_w,
                   color=FP_COLOR, edgecolor="black", linewidth=0.5,
                   hatch=hatch, label=f"FP ({d})" if ax is axes[0] else None)
            ax.bar(x + offset, n_tp, bar_w, bottom=n_fp,
                   color=TP_COLOR, edgecolor="black", linewidth=0.5,
                   hatch=hatch, label=f"TP ({d})" if ax is axes[0] else None)
            totals = n_fp + n_tp
            for xi, tot in zip(x + offset, totals):
                if tot > 0:
                    # Provide a significantly higher vertical offset to lift the labels off the top of the bars
                    ax.text(xi, tot + (sub["n_tp"].max() + sub["n_fp"].max()) * 0.06, 
                            f"{int(tot):,}",
                            ha="center", va="bottom", fontsize=7)
        
        # Add extra headroom so labels aren't clipped
        ax.relim()
        ax.autoscale_view()
        ax.set_ylim(0, ax.get_ylim()[1] * 1.18)
        
        ax.set_title(f"M1 = {m1}", loc="left", fontweight="bold")
        ax.set_ylabel("Window count")
        ax.set_yscale("function", functions=(np.sqrt, lambda x: x**2))
        ax.set_xticks(x)
        ax.set_xticklabels(grans_present)
        ax.grid(axis="y", linestyle=":", alpha=0.4)

    # ┏━━━━━━━━━━ Final touches ━━━━━━━━━━┓
    axes[-1].set_xlabel("Granularity")
    axes[0].legend(loc="upper right", ncol=2, frameon=True, framealpha=0.9)
    fig.suptitle("Dataset Distribution — True Positives vs False Positives Windows\nper M1 Model, Granularity and Direction", fontsize=13, fontweight="bold", y=0.995)
    fig.tight_layout()
    
    # ┏━━━━━━━━━━ Save figure ━━━━━━━━━━┓
    fig.savefig(save_dir / "size_stacked_counts.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Figure 2 — TP-rate heatmap (class balance)
    # Single combined matrix: rows = models, columns = (gran, direction)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Figure 2 — TP-rate heatmap (class balance)
    # Single combined matrix: rows = models, columns = (gran, direction)
    # One shared colorbar on the right symmetrically extracted.
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # ┏━━━━━━━━━━ Set up figure ━━━━━━━━━━┓
    fig, axes_h = plt.subplots(nrows   = 1,
                               ncols   = len(directions),
                               figsize = (max(7, 0.75 * len(grans_present) + 1) * len(directions), 0.6 * n_models + 2.5),
                               sharey  = True,
                               layout  = "constrained")
    if len(directions) == 1:
        axes_h = [axes_h]
    
    # ┏━━━━━━━━━━ Plot heatmaps ━━━━━━━━━━┓
    for i, (ax, d) in enumerate(zip(axes_h, directions)):
        mat = (df[df["direction"] == d]
               .pivot(index="m1_model", columns="granularity", values="tp_rate")
               .reindex(index=models_present, columns=grans_present))
        
        # Don't let seaborn allocate the colorbar so we can do it symmetrically later
        sns.heatmap(mat, ax=ax, annot=True, fmt=".2%", cmap="RdYlGn",
                    vmin=0.40, vmax=0.55, center=0.50,
                    square=True,
                    cbar=False,
                    linewidths=0.4, linecolor="white",
                    annot_kws={"size": 9})
        
        ax.set_title(f"Direction: {d}", fontweight="bold")
        
        # Add a bit of separation for Granularity as requested
        ax.set_xlabel("Granularity", labelpad=12)
        ax.set_ylabel("M1 model" if i == 0 else "")
        if i > 0:
            ax.tick_params(left=False, labelleft=False)
            
    # Extract the image from the first axes to supply to the colorbar
    mappable = axes_h[0].collections[0]
    # layout="constrained" flawlessly integrates this shared colorbar without overlapping
    cbar = fig.colorbar(mappable, ax=axes_h, shrink=0.65, pad=0.02, fraction=0.05)
    cbar.set_label("TP rate")
    
    # Pull title slightly closer to the tables (from y=1.02 -> y=0.97)
    fig.suptitle("True Positive Rate of Meta-Labels across M1 Models, Granularities and Directions", fontsize=12, fontweight="bold", y=0.96)

    # ┏━━━━━━━━━━ Save figure ━━━━━━━━━━┓
    fig.savefig(save_dir / "size_tp_rate.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Figure 3 — stacked FP (dim) + TP (full) bars, model colours kept.
    # FP segment: alpha=0.35 (same model colour, washed out)
    # TP segment: alpha=1.0  (full model colour, stacked on top)
    # White rotated labels inside each segment; total in black above bar.
    # Two-row layout splits fine (30m–4h) from coarse (6h–1d) grans.
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    n_series = n_models * len(directions)
    inner_w  = 0.92 / n_series
    FP_ALPHA = 0.35   # washed-out model colour → FP segment
    TP_ALPHA = 1.00   # full model colour        → TP segment

    row_grans = [[g for g in ["30m", "1h", "2h", "4h"] if g in grans_present],
                 [g for g in ["6h", "8h", "12h", "1d"] if g in grans_present]]
    row_grans = [rg for rg in row_grans if rg]

    fig, axes = plt.subplots(nrows=len(row_grans), ncols=1,
                             figsize=(20, 7 * len(row_grans)), sharey=False)
    if len(row_grans) == 1:
        axes = [axes]

    palette = sns.color_palette("Set2", n_colors=n_models)

    for row_idx, r_grans in enumerate(row_grans):
        ax   = axes[row_idx]
        x    = np.arange(len(r_grans))
        bar_meta = []   # (xpos_arr, fp_arr, tp_arr) for annotation pass

        # ┏━━━━━━━━━━ Draw stacked FP + TP bars ━━━━━━━━━━┓
        for mi, m1 in enumerate(models_present):
            for di, d in enumerate(directions):
                sub    = (df[(df["m1_model"] == m1) & (df["direction"] == d)]
                          .set_index("granularity").reindex(r_grans))
                fp_arr = sub["n_fp"].fillna(0).to_numpy()
                tp_arr = sub["n_tp"].fillna(0).to_numpy()
                offset = (mi * len(directions) + di - (n_series - 1) / 2) * inner_w
                hatch  = "" if d == "UP" else "///"
                col    = palette[mi]
                lbl    = f"{m1} — {d}" if row_idx == 0 else ""

                # FP — dim alpha, no label duplication
                ax.bar(x + offset, fp_arr, inner_w * 0.92,
                       color=col, alpha=FP_ALPHA,
                       edgecolor="black", linewidth=0.45,
                       hatch=hatch, label=lbl)
                # TP — full alpha stacked on top (no label; colour shared)
                ax.bar(x + offset, tp_arr, inner_w * 0.92, bottom=fp_arr,
                       color=col, alpha=TP_ALPHA,
                       edgecolor="black", linewidth=0.45,
                       hatch=hatch)
                bar_meta.append((x + offset, fp_arr, tp_arr))

        # ┏━━━━━━━━━━ Establish ylim before annotations ━━━━━━━━━━┓
        ax.relim(); ax.autoscale_view()
        ymax = ax.get_ylim()[1]
        ax.set_ylim(0, ymax * 1.10)   # Giant headroom canopy to protect long exported text!

        # ┏━━━━━━━━━━ Annotate: white labels inside, black total above ━━━━━━━━━━┓
        min_seg = ymax * 0.020   # strict 2% height check; if it's smaller, it exports its text outside
        for (xpos_arr, fp_arr, tp_arr) in bar_meta:
            for xp, fp, tp in zip(xpos_arr, fp_arr, tp_arr):
                total = fp + tp
                if total == 0:
                    continue
                
                # Base floating total text
                total_str = f"{int(total):,}"
                needs_export = False
                
                # FP label — centred vertically inside FP segment
                if fp >= min_seg:
                    ax.text(xp, fp / 2, f"{int(fp):,}",
                            ha="center", va="center",
                            fontsize=5.5, fontweight="bold",
                            color="white", rotation=25)
                else:
                    needs_export = True

                # TP label — centred vertically inside TP segment
                if tp >= min_seg:
                    ax.text(xp, fp + tp / 2, f"{int(tp):,}",
                            ha="center", va="center",
                            fontsize=5.5, fontweight="bold",
                            color="white", rotation=25)
                else:
                    needs_export = True

                # Fallback purely for scientifically microscopic chunks!
                if needs_export:
                    total_str += f"  (TP:{int(tp):,} | FP:{int(fp):,})"

                # Total above bar (90-degree spine so it can safely grow vertically indefinitely!)
                ax.text(xp, total + ymax * 0.02, total_str,
                        ha="center", va="bottom",
                        fontsize=7.2, color="black", rotation=28)

        ax.set_xticks(x)
        ax.set_xticklabels(r_grans)
        ax.set_ylabel("Window count")
        # Linear scaling is MANDATORY for stacked bar charts. Non-linear scaling (e.g. sqrt) 
        # destroys area proportionality because f(A+B) - f(A) != f(B). This ensures a 24k 
        # segment looks identically sized to a 22k segment regardless of where it stacks!
        ax.grid(axis="y", linestyle=":", alpha=0.4)
        if row_idx == 0:
            ax.set_title("Dataset Size per M1 Model, Granularity and Direction",
                         fontweight="bold", fontsize=15)

    axes[-1].set_xlabel("Granularity")

    # ┏━━━━━━━━━━ Legend: model colours + opacity key ━━━━━━━━━━┓
    from matplotlib.patches import Patch
    from matplotlib.lines   import Line2D
    
    # ┏━━━━━━━━━━ Matplotlib populates legends DOWN columns first! ━━━━━━━━━━┓
    # To get Row 1 = [UP] and Row 2 = [DOWN], we must pair them adjacently!
    all_handles = []
    sep = Line2D([0], [0], color="none", label="")
    
    for mi, m1 in enumerate(models_present):
        all_handles.append(Patch(facecolor=palette[mi], edgecolor="black", linewidth=0.5, hatch="", label=f"{m1} UP"))
        all_handles.append(Patch(facecolor=palette[mi], edgecolor="black", linewidth=0.5, hatch="///", label=f"{m1} DOWN"))

    tp_handle = Patch(facecolor="grey", alpha=TP_ALPHA, edgecolor="black", linewidth=0.5, label="TP")
    fp_handle = Patch(facecolor="grey", alpha=FP_ALPHA, edgecolor="black", linewidth=0.5, label="FP")
    
    # Insert visual separation column, then the TP/FP column
    all_handles.extend([sep, sep, tp_handle, fp_handle])

    fig.legend(handles=all_handles,
               ncol=(len(models_present) + 2),
               loc="lower center", bbox_to_anchor=(0.5, 0.0),
               frameon=False, fontsize=12)
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    fig.savefig(save_dir / "size_total_windows.png", dpi=400, bbox_inches="tight")
    plt.close(fig)

    print(f"[size-plot] Wrote {len(df)} rows and 3 figures to {save_dir}")
    return df


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Return quality: TP vs FP return distributions per model × gran × direction
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def plot_return_quality_distribution(cache_roots: Optional[Dict[str, Path]] = None,
                                     save_dir: Optional[Path] = None,
                                     clip_pct: float = 0.2) -> pd.DataFrame:
    """Split-violin plot of TP vs FP returns per M1 model x granularity x direction.

    Layout: rows = M1 models, columns = granularities (canonical order, no 15m).
    Within each cell: one split violin per direction (UP / DOWN).
    Left half = FP returns (red), right half = TP returns (green).
    A horizontal reference line marks zero return.
    Median is shown as a white dot; IQR box overlaid.

    Parameters
    ----------
    cache_roots : dict {model_name: Path}, optional
        Paths to cache dirs. Defaults to Output/{Kronos,Fincast,Chronos2,Tirex}/cache.
    save_dir : Path, optional
        Where to write the figure. Defaults to Output/Analysis/Quality.
    clip_pct : float
        Y-axis is clipped to [-clip_pct, +clip_pct] to suppress extreme outliers.
        Default 0.15 (±15 %).

    Returns the long-format DataFrame used for plotting.
    """
    import warnings

    # ┏━━━━━━━━━━ Default paths ━━━━━━━━━━┓
    if cache_roots is None:
        base = Path(__file__).resolve().parents[2] / "Output"
        cache_roots = {"Kronos":   base / "Kronos"   / "cache",
                       "Fincast":  base / "Fincast"  / "cache",
                       "Chronos2": base / "Chronos2" / "cache",
                       "Tirex":    base / "Tirex"    / "cache"}
    if save_dir is None:
        save_dir = Path(__file__).resolve().parents[2] / "Output" / "Analysis" / "Quality"
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # ┏━━━━━━━━━━ Direction regex ━━━━━━━━━━┓
    _DIR_RE = re.compile(r"_fee_(up|down)_", re.IGNORECASE)
    def _infer_dir(p: Path) -> Optional[str]:
        m = _DIR_RE.search(p.name)
        return m.group(1).upper() if m else None

    # ┏━━━━━━━━━━ Collect returns ━━━━━━━━━━┓
    rows: List[Dict[str, Any]] = []
    for m1_name, cdir in cache_roots.items():
        cdir = Path(cdir)
        if not cdir.is_dir():
            warnings.warn(f"[quality-plot] Missing cache dir: {cdir}")
            continue
        for pt in sorted(cdir.glob("*.pt")):
            direction = _infer_dir(pt)
            if direction is None:
                continue
            try:
                ds = torch.load(pt, weights_only=False, map_location="cpu")
            except Exception as e:
                warnings.warn(f"[quality-plot] Failed to load {pt.name}: {e}")
                continue
            grans = getattr(ds, "grans", None)
            if not grans:
                continue
            for g in grans:
                sub     = ds.sub[g]
                labels  = np.asarray(sub["labels"].cpu()).ravel()
                returns = np.asarray(sub["returns"].cpu()).ravel()
                valid   = ~(np.isnan(labels) | np.isnan(returns))
                lab     = labels[valid].astype(int)
                ret     = returns[valid]
                for lv, lname in ((1, "TP"), (0, "FP")):
                    mask = lab == lv
                    if mask.sum() == 0:
                        continue
                    for r in ret[mask]:
                        rows.append({"m1_model":    m1_name,
                                     "granularity": g,
                                     "direction":   direction,
                                     "label":       lname,
                                     "return":      float(r)})

    if not rows:
        warnings.warn("[quality-plot] No data — aborting.")
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    # ┏━━━━━━━━━━ Canonical ordering (no 15m) ━━━━━━━━━━┓
    CANONICAL_GRANS = ["30m", "1h", "2h", "4h", "6h", "8h", "12h", "1d"]
    grans_present  = [g for g in CANONICAL_GRANS if g in df["granularity"].unique()]
    df = df[df["granularity"].isin(grans_present)].copy()
    models_present = sorted(df["m1_model"].unique())
    directions     = ["UP", "DOWN"]

    n_rows = len(models_present)
    n_cols = len(grans_present)

    # ┏━━━━━━━━━━ Colour palette ━━━━━━━━━━┓
    TP_COLOR = "#2E7D32"   # dark green
    FP_COLOR = "#C62828"   # dark red
    DIR_PAL  = {"UP": "black", "DOWN": "black"}   # neutral x-ticks

    # ┏━━━━━━━━━━ Academic rcParams ━━━━━━━━━━┓
    plt.rcParams.update({"font.family":       "DejaVu Sans",
                          "axes.titlesize":    8,
                          "axes.labelsize":    7,
                          "xtick.labelsize":   7,
                          "ytick.labelsize":   6.5,
                          "legend.fontsize":   9,
                          "axes.spines.top":   False,
                          "axes.spines.right": False})

    fig, axes = plt.subplots(nrows   = n_rows,
                             ncols   = n_cols,
                             figsize = (2.6 * n_cols, 3.2 * n_rows),
                             sharey  = "row",
                             sharex  = False)
    # Ensure 2-D array even for single row/col
    if n_rows == 1:
        axes = axes[np.newaxis, :]
    if n_cols == 1:
        axes = axes[:, np.newaxis]

    for ri, m1 in enumerate(models_present):
        for ci, gran in enumerate(grans_present):
            ax  = axes[ri, ci]
            sub = df[(df["m1_model"] == m1) & (df["granularity"] == gran)]

            # ┏━━━━━━━━━━ Build per-direction split-violin data ━━━━━━━━━━┓
            # seaborn split violin: x=direction, y=return, hue=label, split=True
            sub_plot = sub[sub["direction"].isin(directions)].copy()
            sub_plot["direction"] = pd.Categorical(sub_plot["direction"],
                                                   categories=directions, ordered=True)
            sub_plot["label"]     = pd.Categorical(sub_plot["label"],
                                                   categories=["FP", "TP"], ordered=True)

            if sub_plot.empty:
                ax.set_visible(False)
                continue

            # ┏━━━━━━━━━━ Clip returns for display ━━━━━━━━━━┓
            sub_plot["return_clipped"] = sub_plot["return"].clip(-clip_pct, clip_pct)

            # ┏━━━━━━━━━━ Split violin ━━━━━━━━━━┓
            try:
                sns.violinplot(data    = sub_plot,
                               x       = "direction",
                               y       = "return_clipped",
                               hue     = "label",
                               split   = True,
                               order   = directions,
                               hue_order = ["FP", "TP"],
                               palette = {"FP": FP_COLOR, "TP": TP_COLOR},
                               inner   = None,
                               linewidth = 0.6,
                               cut     = 0,
                               ax      = ax,
                               legend  = False)
            except Exception:
                # Fallback: plain box if violin KDE fails (too few points)
                sns.boxplot(data    = sub_plot,
                            x       = "direction",
                            y       = "return_clipped",
                            hue     = "label",
                            order   = directions,
                            hue_order = ["FP", "TP"],
                            palette = {"FP": FP_COLOR, "TP": TP_COLOR},
                            linewidth = 0.6,
                            ax      = ax,
                            legend  = False)

            # ┏━━━━━━━━━━ IQR box + median dot + median label overlay ━━━━━━━━━━┓
            # Violin halves (x-centre of each half relative to violin x-position):
            #   FP = −0.22  (left half),  TP = +0.22  (right half)
            # Label placement — all labels point INWARD (toward the gap between violins):
            #   UP  violin (di=0): TP label on the left  of its dot (ha="right")
            #                      FP label on the right of its dot (ha="left")
            #   DOWN violin (di=1): FP label on the right of its dot (ha="left")
            #                       TP label on the left  of its dot (ha="right")
            # → TP always ha="right", FP always ha="left" → labels face inward, no overlap.
            half_offsets = {"FP": -0.22, "TP":  0.22}
            # IQR box half-width = 0.045; label sits just outside the box edge
            BOX_HW = 0.055   # slightly beyond the box edge
            text_side = {"FP": +4.5, "TP": -4.5}   # FP: text to the right (+), TP: to the left (−)
            text_ha   = {"FP": "left",  "TP": "right"}
            label_colors = {"FP": FP_COLOR, "TP": TP_COLOR}
            for di, d in enumerate(directions):
                for lname, x_off in half_offsets.items():
                    seg = sub_plot[(sub_plot["direction"] == d) &
                                   (sub_plot["label"] == lname)]["return_clipped"]
                    if len(seg) < 4:
                        continue
                    # Raw median for the displayed number (unclipped)
                    seg_raw = sub_plot[(sub_plot["direction"] == d) &
                                       (sub_plot["label"] == lname)]["return"]
                    q1, _, q3 = np.percentile(seg, [25, 50, 75])
                    med_raw  = float(np.median(seg_raw))
                    med_disp = float(np.median(seg))
                    xc = di + x_off
                    # IQR box
                    ax.add_patch(plt.Rectangle((xc - 0.045, q1), 0.09, q3 - q1,
                                               facecolor="white", edgecolor="black",
                                               linewidth=0.5, zorder=3))
                    # Median dot
                    ax.plot(xc, med_disp, "o", color="white",
                            markersize=3.5, zorder=4,
                            markeredgecolor="black", markeredgewidth=0.5)
                    # Median value label — anchored inward from the IQR box edge
                    sign  = "+" if med_raw >= 0 else ""
                    label = f"{sign}{med_raw*100:.2f}%"
                    tx = xc + text_side[lname] * BOX_HW
                    ax.text(tx, med_disp, label,
                            ha=text_ha[lname], va="center",
                            fontsize=5.0, fontweight="bold",
                            color=label_colors[lname], zorder=5)

            # ┏━━━━━━━━━━ Reference lines: zero + fee break-even ━━━━━━━━━━┓
            FEE = 0.002   # 0.2% round-trip (0.1% taker + 0.1% maker)
            ax.axhline(0,    color="black",   linewidth=0.7, linestyle="--", alpha=0.50, zorder=2)
            ax.axhline( FEE, color="purple", linewidth=0.7, linestyle=":",  alpha=0.75, zorder=2)
            ax.axhline(-FEE, color="purple", linewidth=0.7, linestyle=":",  alpha=0.75, zorder=2)

            # ┏━━━━━━━━━━ Cell formatting ━━━━━━━━━━┓
            ax.set_ylim(-clip_pct * 1.05, clip_pct * 1.05)
            ax.set_xlabel("")
            ax.set_xticks([0, 1])
            ax.set_xticklabels(directions, fontsize=7)
            # Colour x-tick labels by direction
            for tick, d in zip(ax.get_xticklabels(), directions):
                tick.set_color(DIR_PAL[d])
                tick.set_fontweight("bold")
            ax.grid(axis="y", linestyle=":", alpha=0.35, linewidth=0.5)

            # Column header (gran) on top row only
            if ri == 0:
                ax.set_title(gran, fontweight="bold", fontsize=9, pad=4)
            # Y-label (model) on leftmost column only
            if ci == 0:
                ax.set_ylabel(f"{m1}\nReturn", fontsize=8, fontweight="bold")
            else:
                ax.set_ylabel("")
            # Hide redundant y-ticks on non-left cells (sharey handles scale)
            if ci > 0:
                ax.tick_params(labelleft=False)

    # ┏━━━━━━━━━━ Global title ━━━━━━━━━━┓
    fig.suptitle("Return Quality of True Positives vs False Positives per M1 Model, Granularity and Direction\n"
                 f"Y-Axis Clipped to ±{int(clip_pct*100)}%",
                 fontsize=11, fontweight="bold", y=1.01)

    # ┏━━━━━━━━━━ Shared legend ━━━━━━━━━━┓
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    legend_handles = [
        Patch(facecolor=FP_COLOR, edgecolor="black", linewidth=0.6, label="FP"),
        Patch(facecolor=TP_COLOR, edgecolor="black", linewidth=0.6, label="TP"),
        Patch(facecolor="white",  edgecolor="black", linewidth=0.6, label="IQR box (25th-75th pct)"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="white",
               markeredgecolor="black", markeredgewidth=0.5,
               markersize=7, label="Median (Value %)"),
        Line2D([0], [0], color="black",   linewidth=0.9, linestyle="--",
               alpha=0.6, label="Zero return"),
        Line2D([0], [0], color="purple", linewidth=0.9, linestyle=":",
               alpha=0.8, label="Fee break-even (±0.2%)"),
    ]
    fig.legend(handles=legend_handles, ncol=6,
               loc="lower center", bbox_to_anchor=(0.5, -0.01),
               frameon=False, fontsize=10)

    fig.tight_layout(rect=[0, 0.03, 1, 1])
    out_path = save_dir / "return_quality_distribution.png"
    fig.savefig(out_path, dpi=400, bbox_inches="tight")
    plt.close(fig)
    print(f"[quality-plot] Saved → {out_path}  ({len(df):,} return observations)")
    return df


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Performance over number of features
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
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


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CPCV Edge Convergence — data loading helper
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def _load_cpcv_records(edge_root: str,
                       output_root: str | None = None) -> list[dict]:
    """Pair every edge_summary_*.json with its analysis_summary.json.

    Returns a list of records with the three filter inputs:
        - frac_profitable    (C1: regime-sensitivity, fraction of CPCV paths > 0)
        - median_path_sharpe (C2: median Sharpe across the 5 paths)
        - val_mean_ret       (C3: chronological Val_selective.mean_ret)
    plus the two outcome ground-truths used by the heatmaps:
        - val_mean_ret  (outcome on Val)
        - test_mean_ret (outcome on Test)
    """
    import json
    import os
    import glob
    import numpy as np

    # ┏━━━━━━━━━━ Convert edge_root and output_root to absolute paths ━━━━━━━━━━┓
    edge_root = os.path.abspath(edge_root)

    # ┏━━━━━━━━━━ If output_root is not provided, derive it from edge_root ━━━━━━━━━━┓
    if output_root is None:
        output_root = os.path.dirname(edge_root.rstrip(os.sep))
        output_root = os.path.dirname(output_root)  # …/Output
    output_root = os.path.abspath(output_root)

    # ┏━━━━━━━━━━ Dictionary mapping model names to their respective keys ━━━━━━━━━━┓
    M2_KEY = {"rf": "rf_temporal_all_features",
              "autogluon": "autogluon_temporal_all_features",
              "tabpfn": "tabpfn_temporal_all_features",
              "tabicl": "tabicl_temporal_all_features",
              "tabm": "tabm_temporal_all_features",}

    # ┏━━━━━━━━━━ Initialize the records list ━━━━━━━━━━┓
    records = []

    # ┏━━━━━━━━━━ Pattern to find all edge_summary_*.json files ━━━━━━━━━━┓
    pattern = os.path.join(edge_root, "**", "edge_summary_*.json")
    
    # ┏━━━━━━━━━━ Iterate over all edge_summary_*.json files ━━━━━━━━━━┓
    for fpath in glob.glob(pattern, recursive=True):
        # ┏━━━━━━━━━━ Load the edge_summary_*.json file ━━━━━━━━━━┓
        try:
            with open(fpath) as f:
                edge_data = json.load(f)
        except Exception:
            continue

        # ┏━━━━━━━━━━ Extract the relative path ━━━━━━━━━━┓
        rel   = os.path.relpath(fpath, edge_root)
        parts = rel.split(os.sep)
        if len(parts) < 4:
            continue
        m1, m2, direction = parts[0], parts[1], parts[2]

        # ┏━━━━━━━━━━ Iterate over all granularities ━━━━━━━━━━┓
        for gran, entry in edge_data.items():
            sharpes = entry.get("path_sharpes")
            frac_p  = entry.get("frac_profitable")
            med_sr  = entry.get("median_path_sharpe")
            if sharpes is None or frac_p is None or med_sr is None:
                continue
            sharpes = np.asarray(sharpes, dtype=float)
            sharpes = sharpes[~np.isnan(sharpes)]
            if sharpes.size == 0:
                continue

            # ┏━━━━━━━━━━ Pair with analysis_summary.json ━━━━━━━━━━┓
            ana_path = os.path.join(output_root, m1, m2, direction,
                                    "Utility_Score_NoCal", f"{gran}_tp", "analysis_summary.json")
            val_mean_ret  = None
            test_mean_ret = None

            # ┏━━━━━━━━━━ If analysis_summary.json exists, extract the performance metrics ━━━━━━━━━━┓
            val_mean_ret = test_mean_ret = None
            val_f1       = test_f1       = None
            if os.path.exists(ana_path):
                try:
                    with open(ana_path) as f:
                        ana = json.load(f)
                    block    = ana.get(M2_KEY.get(m2, ""), {})
                    val_sel  = block.get("Val_selective",  {}) or {}
                    test_sel = block.get("Test_selective", {}) or {}
                    val_blk  = block.get("Val",  {}) or {}
                    test_blk = block.get("Test", {}) or {}
                    val_mean_ret  = val_sel.get("mean_ret")
                    test_mean_ret = test_sel.get("mean_ret")
                    val_f1        = val_blk.get("f1_score")
                    test_f1       = test_blk.get("f1_score")
                except Exception:
                    pass

            # ┏━━━━━━━━━━ Append the performance metrics to the records list ━━━━━━━━━━┓
            records.append({"m1":              m1,
                            "m2":              m2,
                            "direction":       direction,
                            "gran":            gran,
                            "frac_profitable": float(frac_p),
                            "median_sharpe":   float(med_sr),
                            "val_mean_ret":    None if val_mean_ret  is None else float(val_mean_ret),
                            "test_mean_ret":   None if test_mean_ret is None else float(test_mean_ret),
                            "val_f1":          None if val_f1        is None else float(val_f1),
                            "test_f1":         None if test_f1       is None else float(test_f1),
                            "path_sharpes":    sharpes.tolist()})
    return records


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CPCV Edge — Constraint-trigger bar plot
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def plot_cpcv_constraint_bars(edge_root: str = "/home/pablo/M2_DS/Secondary-Model/src/Output/Analysis/Edge_NoCal",
                              output_dir: str | None = None,
                              tau_fp: float = 0.6,
                              tau_sr: float = 0.0) -> str:
    """3-circle Venn diagram for the three CPCV constraints.

        C1 := frac_profitable    >= tau_fp   (default 0.6)
        C2 := median_path_sharpe >= tau_sr   (default 0.0)
        C3 := val_mean_ret       >  0

    Circles are drawn at fixed equal-size positions so every region is
    clearly readable regardless of the actual subset counts. Annotated
    counts point to each region with arrows.
    """
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.patches import Circle
    from matplotlib.colors import to_rgba

    # ┏━━━━━━━━━━ Setup the output directory ━━━━━━━━━━┓
    edge_root = os.path.abspath(edge_root)
    if output_dir is None:
        output_dir = edge_root
    os.makedirs(output_dir, exist_ok=True)

    # ┏━━━━━━━━━━ Load the CPCV records ━━━━━━━━━━┓
    records = _load_cpcv_records(edge_root)
    records = [r for r in records if r["val_mean_ret"] is not None]
    N = len(records)
    if N == 0:
        print(f"[plot_cpcv_constraint_venn] No records with val_mean_ret found")
        return ""

    # ┏━━━━━━━━━━ Extract the performance metrics ━━━━━━━━━━┓
    fp = np.array([r["frac_profitable"] for r in records])
    sr = np.array([r["median_sharpe"]   for r in records])
    mr = np.array([r["val_mean_ret"]    for r in records])

    # ┏━━━━━━━━━━ Apply the constraints ━━━━━━━━━━┓
    c1 = fp >= tau_fp
    c2 = sr >= tau_sr
    c3 = mr >  0.0

    # ┏━━━━━━━━━━ Compute the 7 mutually-exclusive region counts ━━━━━━━━━━┓
    Abc = int(( c1 & ~c2 & ~c3).sum())   # only C1
    aBc = int((~c1 &  c2 & ~c3).sum())   # only C2
    ABc = int(( c1 &  c2 & ~c3).sum())   # C1 ∧ C2, not C3
    abC = int((~c1 & ~c2 &  c3).sum())   # only C3
    AbC = int(( c1 & ~c2 &  c3).sum())   # C1 ∧ C3, not C2
    aBC = int((~c1 &  c2 &  c3).sum())   # C2 ∧ C3, not C1
    ABC = int(( c1 &  c2 &  c3).sum())   # all three
    none = int((~c1 & ~c2 & ~c3).sum())  # outside all
    assert Abc+aBc+ABc+abC+AbC+aBC+ABC+none == N

    # ┏━━━━━━━━━━ Fixed circle geometry — equal radii, standard triangle layout ━━━━━━━━━━┓
    R   = 1.0          # circle radius
    sep = 0.75         # distance between circle centres (< R so they overlap)
    # C1: top-left, C2: top-right, C3: bottom-centre
    cx1, cy1 = -sep / 2,  sep * np.sqrt(3) / 4 + 0.1
    cx2, cy2 =  sep / 2,  sep * np.sqrt(3) / 4 + 0.1
    cx3, cy3 =  0.0,     -sep * np.sqrt(3) / 4 - 0.05

    COLORS = {"c1": "#6baed6", "c2": "#74c476", "c3": "#fd8d3c"}
    ALPHA  = 0.38

    fig, ax = plt.subplots(figsize=(10, 9))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.set_aspect("equal")
    ax.axis("off")

    # ┏━━━━━━━━━━ Draw the three circles ━━━━━━━━━━┓
    for (cx, cy), col in [(( cx1, cy1), COLORS["c1"]),
                           (( cx2, cy2), COLORS["c2"]),
                           (( cx3, cy3), COLORS["c3"])]:
        patch = Circle((cx, cy), R, color=col, alpha=ALPHA, zorder=1)
        ax.add_patch(patch)
        edge = Circle((cx, cy), R, fill=False, edgecolor=col,
                      linewidth=2.5, zorder=2)
        ax.add_patch(edge)

    # ┏━━━━━━━━━━ Circle title labels outside the diagram ━━━━━━━━━━┓
    set_labels = [(f"C1\nfrac_prof ≥ {tau_fp:.1f}", cx1 - 0.85, cy1 + 0.80, COLORS["c1"]),
                  (f"C2\nmed_SR ≥ {tau_sr:.1f}",    cx2 + 0.85, cy2 + 0.80, COLORS["c2"]),
                  (f"C3\nval_mean_ret > 0",          cx3,        cy3 - 1.05, COLORS["c3"])]

    # ┏━━━━━━━━━━ Label text styling ━━━━━━━━━━┓
    for text, tx, ty, col in set_labels:
        ax.text(tx, ty, text, ha="center", va="center", fontsize=12,
                fontweight="bold", color=col,
                bbox=dict(boxstyle="round,pad=0.3", fc="white",
                          ec=col, lw=1.8, alpha=0.95), zorder=5)

    # ┏━━━━━━━━━━ Region label positions + arrow targets ━━━━━━━━━━┓
    # Each entry: (count, label_x, label_y, arrow_target_x, arrow_target_y, colour)
    # Arrow targets are approximate centroids of each Venn region.
    ARROWPROPS = dict(arrowstyle="-|>", color="#555555", lw=1.4)

    def _fmt(n):
        return f"{n}\n({100*n/N:.1f}%)"

    # ┏━━━━━━━━━━ Analytically-computed region centroids ━━━━━━━━━━┓
    def _excl(cx, cy, o1, o2, push=0.55):
        """Centre of exclusive region: push away from the other two circles."""
        ox, oy = (o1[0]+o2[0])/2, (o1[1]+o2[1])/2
        dx, dy = cx-ox, cy-oy
        n = np.sqrt(dx**2+dy**2) or 1.0
        return cx+push*dx/n, cy+push*dy/n

    def _pair(ca, cb, o, push=0.18):
        """Centre of pairwise intersection: midpoint pushed away from third circle."""
        mx, my = (ca[0]+cb[0])/2, (ca[1]+cb[1])/2
        dx, dy = mx-o[0], my-o[1]
        n = np.sqrt(dx**2+dy**2) or 1.0
        return mx+push*dx/n, my+push*dy/n

    t_c1  = _excl(cx1, cy1, (cx2,cy2), (cx3,cy3))
    t_c2  = _excl(cx2, cy2, (cx1,cy1), (cx3,cy3))
    t_c3  = _excl(cx3, cy3, (cx1,cy1), (cx2,cy2))
    t_c12 = _pair((cx1,cy1), (cx2,cy2), (cx3,cy3))
    t_c13 = _pair((cx1,cy1), (cx3,cy3), (cx2,cy2))
    t_c23 = _pair((cx2,cy2), (cx3,cy3), (cx1,cy1))
    t_all = ((cx1+cx2+cx3)/3, (cy1+cy2+cy3)/3)

    regions = [
        # count  label_x  label_y   arrow_x      arrow_y      colour
        (Abc, -2.30,  1.40,  t_c1[0],  t_c1[1],  COLORS["c1"]),   # only C1
        (aBc,  2.30,  1.40,  t_c2[0],  t_c2[1],  COLORS["c2"]),   # only C2
        (abC,  0.00, -2.30,  t_c3[0],  t_c3[1],  COLORS["c3"]),   # only C3
        (ABc,  0.00,  2.20,  t_c12[0], t_c12[1], "#2171b5"),       # C1∧C2 only
        (AbC, -2.30, -1.40,  t_c13[0], t_c13[1], "#d94801"),       # C1∧C3 only
        (aBC,  2.30, -1.40,  t_c23[0], t_c23[1], "#238b45"),       # C2∧C3 only
        (ABC, -2.30,  0.00,  t_all[0], t_all[1], "#006d2c"),       # C1∧C2∧C3
    ]

    region_labels = ["only C1", "only C2", "only C3", "C1∧C2", "C1∧C3", "C2∧C3", "C1∧C2∧C3"]

    # ┏━━━━━━━━━━ Region labels and arrows ━━━━━━━━━━┓
    for (cnt, lx, ly, ax_, ay_, col), rlbl in zip(regions, region_labels):
        ax.annotate(_fmt(cnt), xy=(ax_, ay_), xytext=(lx, ly),
                    ha="center", va="center", fontsize=10.5, fontweight="bold",
                    color="black", bbox=dict(boxstyle="round,pad=0.35", fc="white",
                    ec=col, lw=1.8, alpha=0.97), arrowprops=ARROWPROPS, zorder=6)

    # ┏━━━━━━━━━━ "None" box — bottom right, no arrow needed ━━━━━━━━━━┓
    ax.text(2.2, -1.8, f"None\n{none} ({100*none/N:.1f}%)",
            ha="center", va="center", fontsize=10.5, fontweight="bold",
            color="#666666",
            bbox=dict(boxstyle="round,pad=0.35", fc="#f5f5f5",
                      ec="#aaaaaa", lw=1.6, alpha=0.97), zorder=6)

    ax.set_xlim(-3.0, 3.0)
    ax.set_ylim(-2.8, 2.8)

    ax.set_title(f"CPCV 3-constraint Venn diagram  |  N = {N} configurations | "
                 f"C1: frac_profitable ≥ {tau_fp:.2f} | "
                 f"C2: median_path_sharpe ≥ {tau_sr:.2f} | "
                 f"C3: val_mean_ret > 0",
                 fontsize=12, fontweight="bold", pad=14)

    plt.tight_layout()
    out_path = os.path.join(output_dir, "cpcv_constraint_venn.png")
    plt.savefig(out_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"[plot_cpcv_constraint_venn] {N} configs -> {out_path}")
    return out_path


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CPCV Edge Convergence Heatmap (Val + Test)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def plot_cpcv_edge_heatmap(edge_root: str = "/home/pablo/M2_DS/Secondary-Model/src/Output/Analysis/Edge_NoCal",
                            output_dir=None) -> str:
    """Informed-accuracy heatmap for the 3-constraint CPCV filtering logic.

    Three filter constraints (all evaluated on validation data — no leakage):
        C1 := frac_profitable        >= tau_fp   (regime sensitivity)
        C2 := median_path_sharpe     >= tau_sr   (median CPCV Sharpe)
        C3 := Val_selective.mean_ret >  0        (chronological-val signal)

    For every (tau_sr, tau_fp) pair on the heatmap, each config is classified
    against an actual outcome (Val.mean_ret>0 or Test.mean_ret>0) into one of
    four mutually-exclusive categories:

        TP : C1 and C2 and C3 all pass AND outcome > 0  (filter trusted, was right)
        TN : at least one of {C1,C2,C3} fails AND outcome <= 0 (filter rejected, avoided loss)
        FN : at least one fails AND outcome > 0 (missed profit)
        FP : all three pass  AND  outcome <= 0 (worst case: deployed a loser)

    Accuracy = (TP + TN) / (TP + TN + FN + FP) = (TP + TN) / N_total.

    Two heatmaps are produced (same filter, different outcome labels):
        cpcv_edge_heatmap_val.png   (outcome = Val_selective.mean_ret > 0)
        cpcv_edge_heatmap_test.png  (outcome = Test_selective.mean_ret > 0)
    """
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    from matplotlib.colors import LinearSegmentedColormap

    # ┏━━━━━━━━━━ Convert edge_root and output_dir to absolute paths ━━━━━━━━━━┓
    edge_root = os.path.abspath(edge_root)
    if output_dir is None:
        output_dir = edge_root
    os.makedirs(output_dir, exist_ok=True)

    # ┏━━━━━━━━━━ Load CPCV records ━━━━━━━━━━┓
    records = _load_cpcv_records(edge_root)
    records = [r for r in records if r["val_mean_ret"] is not None]
    if not records:
        print(f"[plot_cpcv_edge_heatmap] No records with val_mean_ret found under {edge_root}")
        return ""

    # ┏━━━━━━━━━━ Extract feature arrays ━━━━━━━━━━┓
    fp = np.array([r["frac_profitable"] for r in records], dtype=float)
    sr = np.array([r["median_sharpe"]   for r in records], dtype=float)
    vmr = np.array([r["val_mean_ret"]   for r in records], dtype=float)
    tmr = np.array([(r["test_mean_ret"] if r["test_mean_ret"] is not None else np.nan) for r in records], dtype=float)

    # ┏━━━━━━━━━━ Define thresholds for the heatmap grid ━━━━━━━━━━┓
    sr_step = 0.5
    sr_hi   = max(sr_step, float(np.ceil(sr.max() / sr_step) * sr_step))
    x_thresholds = np.arange(0.0, sr_hi + sr_step * 0.5, sr_step)
    y_thresholds = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    n_x, n_y = len(x_thresholds), len(y_thresholds)

    # ┏━━━━━━━━━━ Print summary statistics ━━━━━━━━━━┓
    print(f"[plot_cpcv_edge_heatmap] {len(records)} configs loaded (val_mean_ret available)")
    print(f"  median_sharpe  : [{sr.min():.2f}, {sr.max():.2f}]")
    print(f"  frac_profitable: [{fp.min():.2f}, {fp.max():.2f}]")
    print(f"  val_mean_ret   : [{vmr.min():.4f}, {vmr.max():.4f}]")
    if np.isfinite(tmr).any():
        print(f"  test_mean_ret  : [{np.nanmin(tmr):.4f}, {np.nanmax(tmr):.4f}]  (N={int(np.isfinite(tmr).sum())})")

    # ┏━━━━━━━━━━ Define color map ━━━━━━━━━━┓
    cmap = LinearSegmentedColormap.from_list(
        "acc_cmap",
        ["#67000d", "#a50f15", "#cb181d", "#fb6a4a", "#fcae91", "#fee5d9",
         "#ffffff",
         "#e5f5e0", "#a1d99b", "#41ab5d", "#238b45", "#005a32", "#00441b"],
        N=256)

    # ┏━━━━━━━━━━ Iterate over splits (val and test) ━━━━━━━━━━┓
    out_paths = {}
    for split_name, outcome_arr in (("val", vmr), ("test", tmr)):
        valid = np.isfinite(outcome_arr)
        N = int(valid.sum())
        
        # ┏━━━━━━━━━━ Skip if no valid outcomes ━━━━━━━━━━┓
        if N == 0:
            print(f"[plot_cpcv_edge_heatmap] No valid {split_name} outcomes - skipping")
            continue

        # ┏━━━━━━━━━━ Filter records for current split ━━━━━━━━━━┓
        fp_v   = fp[valid]
        sr_v   = sr[valid]
        c3_v   = vmr[valid] > 0.0
        out_v  = outcome_arr[valid] > 0.0

        # ┏━━━━━━━━━━ Initialize grids ━━━━━━━━━━┓
        accuracy  = np.zeros((n_y, n_x), dtype=float)
        precision = np.full((n_y, n_x), np.nan, dtype=float)
        tp_grid   = np.zeros((n_y, n_x), dtype=int)
        tn_grid   = np.zeros((n_y, n_x), dtype=int)
        fp_grid   = np.zeros((n_y, n_x), dtype=int)
        fn_grid   = np.zeros((n_y, n_x), dtype=int)

        # ┏━━━━━━━━━━ Compute TP, TN, FP, FN for each grid cell ━━━━━━━━━━┓
        for yi, tau_fp in enumerate(y_thresholds):
            for xi, tau_sr in enumerate(x_thresholds):
                # ┏━━━━━━━━━━ Apply Filters ━━━━━━━━━━┓
                c1 = fp_v >= tau_fp
                c2 = sr_v >= tau_sr
                all_pass = c1 & c2 & c3_v

                # ┏━━━━━━━━━━ Compute counts ━━━━━━━━━━┓
                tp = int(np.sum( all_pass &  out_v))
                fp_ = int(np.sum( all_pass & ~out_v))
                fn = int(np.sum(~all_pass &  out_v))
                tn = int(np.sum(~all_pass & ~out_v))

                # ┏━━━━━━━━━━ Store counts and metrics ━━━━━━━━━━┓
                tp_grid[yi, xi] = tp
                fp_grid[yi, xi] = fp_
                fn_grid[yi, xi] = fn
                tn_grid[yi, xi] = tn
                accuracy[yi, xi] = (tp + tn) / N
                precision[yi, xi] = (tp / (tp + fp_)) if (tp + fp_) > 0 else np.nan

        # ┏━━━━━━━━━━ F1 grid — F1 of the triggering logic (same TP/FP/FN as above) ━━━━━━━━━━┓
        # F1 = 2*TP / (2*TP + FP + FN)  — harmonic mean of precision and recall
        # of the filter itself, treating "outcome > 0" as the positive class.
        f1_grid = np.full((n_y, n_x), np.nan, dtype=float)
        for yi in range(n_y):
            for xi in range(n_x):
                tp = tp_grid[yi, xi]
                fp_ = fp_grid[yi, xi]
                fn = fn_grid[yi, xi]
                denom = 2 * tp + fp_ + fn
                if denom > 0:
                    f1_grid[yi, xi] = (2 * tp) / denom

        # ┏━━━━━━━━━━ Three side-by-side subplots ━━━━━━━━━━┓
        fig, (ax_acc, ax_prec, ax_f1) = plt.subplots(
            1, 3, figsize=(max(30, 2.85 * n_x + 9), max(6, 0.85 * n_y + 2)))
        fig.patch.set_facecolor("white")

        # ┏━━━━━━━━━━ Set suptitle ━━━━━━━━━━┓
        outcome_lbl = "Val_selective.mean_ret > 0" if split_name == "val" else "Test_selective.mean_ret > 0"
        fig.suptitle(f"CPCV 3-Constraint Filter — {split_name.upper()} outcome\n"
                     f"C1: frac_prof≥τ_FP   C2: med_SR≥τ_SR   C3: val_mean_ret>0    "
                     f"|   outcome = {outcome_lbl}   |   N = {N} configs",
                     fontsize   = 12,
                     fontweight = "bold",
                     y          = 0.99)

        # ┏━━━━━━━━━━ Extent ━━━━━━━━━━┓
        extent = [x_thresholds[0]  - sr_step / 2,
                  x_thresholds[-1] + sr_step / 2,
                  y_thresholds[0]  - 0.1,
                  y_thresholds[-1] + 0.1]

        # ┏━━━━━━━━━━ Per-grid color limits anchored to actual min/max for visual contrast. ━━━━━━━━━━┓
        def _clim(arr):
            vals = arr[np.isfinite(arr)]
            if vals.size == 0:
                return 0.0, 1.0
            lo, hi = float(vals.min()), float(vals.max())
            if hi - lo < 1e-9:
                lo = max(0.0, lo - 0.05); hi = min(1.0, hi + 0.05)
            return lo, hi

        acc_lo,  acc_hi  = _clim(accuracy)
        prec_lo, prec_hi = _clim(precision)

        # ┏━━━━━━━━━━ Subplot 1 — Accuracy ━━━━━━━━━━┓
        im_a = ax_acc.imshow(accuracy,
                             origin  = "lower",
                             cmap    = cmap,
                             vmin    = acc_lo,
                             vmax    = acc_hi,
                             aspect  = "auto",
                             extent  = extent)
        
        # ┏━━━━━━━━━━ Iterate over grid cells ━━━━━━━━━━┓
        for yi in range(n_y):
            for xi in range(n_x):
                acc = accuracy[yi, xi]
                tp, tn = tp_grid[yi, xi], tn_grid[yi, xi]
                fp_, fn = fp_grid[yi, xi], fn_grid[yi, xi]
                cx, cy  = x_thresholds[xi], y_thresholds[yi]
                rel = (acc - acc_lo) / max(1e-9, acc_hi - acc_lo)
                text_col = "white" if (rel < 0.30 or rel > 0.85) else "black"
                ax_acc.text(cx, cy + 0.035, f"{acc*100:.0f}%",
                            ha="center", va="center",
                            fontsize=10, color=text_col, fontweight="bold")
                ax_acc.text(cx, cy - 0.035,
                            f"TP={tp} TN={tn}\nFP={fp_} FN={fn}",
                            ha="center", va="center",
                            fontsize=6.8, color=text_col, alpha=0.85)

        # ┏━━━━━━━━━━ Set ticks and labels for the accuracy subplot ━━━━━━━━━━┓
        ax_acc.set_xticks(x_thresholds)
        ax_acc.set_xticklabels([f"{v:.1f}" for v in x_thresholds], fontsize=9)
        ax_acc.set_yticks(y_thresholds)
        ax_acc.set_yticklabels([f"{v:.1f}" for v in y_thresholds], fontsize=10)
        ax_acc.set_xlabel("τ_SR  (median path Sharpe threshold, C2)", fontsize=11, fontweight="bold", labelpad=8)
        ax_acc.set_ylabel("τ_FP  (frac. profitable paths threshold, C1)", fontsize=11, fontweight="bold", labelpad=8)
        ax_acc.set_title(f"Accuracy = (TP+TN) / N [{acc_lo*100:.0f}% - {acc_hi*100:.0f}%]", fontsize=11, fontweight="bold", pad=8)
        
        # ┏━━━━━━━━━━ Draw grid lines ━━━━━━━━━━┓
        for xv in x_thresholds - sr_step / 2:
            ax_acc.axvline(xv, color="white", lw=0.8, alpha=0.6)
        ax_acc.axvline(x_thresholds[-1] + sr_step / 2, color="white", lw=0.8, alpha=0.6)
        for yv in y_thresholds:
            ax_acc.axhline(yv - 0.1, color="white", lw=0.8, alpha=0.6)
        ax_acc.axhline(y_thresholds[-1] + 0.1, color="white", lw=0.8, alpha=0.6)

        # ┏━━━━━━━━━━ Add colorbar ━━━━━━━━━━┓
        cbar_a = fig.colorbar(im_a, ax=ax_acc, fraction=0.038, pad=0.02)
        cbar_a.set_label("Accuracy", fontsize=9)
        cbar_a.ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0, decimals=0))
        cbar_a.ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=5))
        cbar_a.ax.tick_params(labelsize=8)

        # ┏━━━━━━━━━━ Highlight best accuracy cell ━━━━━━━━━━┓
        best_yi_a, best_xi_a = np.unravel_index(np.argmax(accuracy), accuracy.shape)
        ax_acc.add_patch(plt.Rectangle((x_thresholds[best_xi_a] - sr_step / 2, y_thresholds[best_yi_a] - 0.1),
                                        sr_step, 0.2, fill=False, edgecolor="black", lw=2.2))

        # ┏━━━━━━━━━━ Subplot 2 — Precision ━━━━━━━━━━┓
        # Precision = TP / (TP + FP):  among configs the filter accepts, how
        # many actually deliver a positive outcome?  This is the "deployment
        # trust" signal — it directly answers "if I deploy when the filter
        # fires, what fraction will be winners?".
        im_p = ax_prec.imshow(precision,
                              origin = "lower",
                              cmap   = cmap,
                              vmin   = prec_lo,
                              vmax   = prec_hi,
                              aspect = "auto",
                              extent = extent)

        # ┏━━━━━━━━━━ Iterate over grid cells ━━━━━━━━━━┓
        for yi in range(n_y):
            for xi in range(n_x):
                prec = precision[yi, xi]
                tp, fp_ = tp_grid[yi, xi], fp_grid[yi, xi]
                cx, cy  = x_thresholds[xi], y_thresholds[yi]
                if not np.isfinite(prec):
                    ax_prec.text(cx, cy, "n/a", ha="center", va="center", fontsize=8, color="#999999")
                    continue
                rel = (prec - prec_lo) / max(1e-9, prec_hi - prec_lo)
                text_col = "white" if (rel < 0.30 or rel > 0.85) else "black"
                ax_prec.text(cx, cy + 0.035, f"{prec*100:.0f}%",
                             ha="center", va="center",
                             fontsize=10, color=text_col, fontweight="bold")
                ax_prec.text(cx, cy - 0.035,
                             f"TP={tp} FP={fp_}",
                             ha="center", va="center",
                             fontsize=6.8, color=text_col, alpha=0.85)

        # ┏━━━━━━━━━━ Set ticks and labels for the precision subplot ━━━━━━━━━━┓
        ax_prec.set_xticks(x_thresholds)
        ax_prec.set_xticklabels([f"{v:.1f}" for v in x_thresholds], fontsize=9)
        ax_prec.set_yticks(y_thresholds)
        ax_prec.set_yticklabels([f"{v:.1f}" for v in y_thresholds], fontsize=10)
        ax_prec.set_xlabel("τ_SR  (median path Sharpe threshold, C2)", fontsize=11, fontweight="bold", labelpad=8)
        ax_prec.set_ylabel("τ_FP  (frac. profitable paths threshold, C1)", fontsize=11, fontweight="bold", labelpad=8)
        ax_prec.set_title(f"Precision = TP / (TP+FP) [{prec_lo*100:.0f}% - {prec_hi*100:.0f}%]", fontsize=11, fontweight="bold", pad=8)

        # ┏━━━━━━━━━━ Draw grid lines ━━━━━━━━━━┓
        for xv in x_thresholds - sr_step / 2:
            ax_prec.axvline(xv, color="white", lw=0.8, alpha=0.6)
        ax_prec.axvline(x_thresholds[-1] + sr_step / 2, color="white", lw=0.8, alpha=0.6)
        for yv in y_thresholds:
            ax_prec.axhline(yv - 0.1, color="white", lw=0.8, alpha=0.6)
        ax_prec.axhline(y_thresholds[-1] + 0.1, color="white", lw=0.8, alpha=0.6)

        # ┏━━━━━━━━━━ Add colorbar ━━━━━━━━━━┓
        cbar_p = fig.colorbar(im_p, ax=ax_prec, fraction=0.038, pad=0.02)
        cbar_p.set_label("Precision", fontsize=9)
        cbar_p.ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0, decimals=0))
        cbar_p.ax.tick_params(labelsize=8)

        # ┏━━━━━━━━━━ Highlight best precision cell ━━━━━━━━━━┓
        prec_for_best = np.where(np.isfinite(precision), precision, -np.inf)
        best_yi_p, best_xi_p = np.unravel_index(np.argmax(prec_for_best), precision.shape)
        if np.isfinite(precision[best_yi_p, best_xi_p]):
            ax_prec.add_patch(plt.Rectangle((x_thresholds[best_xi_p] - sr_step / 2, y_thresholds[best_yi_p] - 0.1),
                                             sr_step, 0.2, fill=False, edgecolor="black", lw=2.2))

        # ┏━━━━━━━━━━ Subplot 3 — Mean F1 of passing configs ━━━━━━━━━━┓
        # Each cell shows the mean F1 score (at τ=0.5) averaged over all
        # configs that pass C1∧C2∧C3 at that threshold pair. This answers:
        # "among the configs the filter selects for deployment, how good
        # is their raw discriminative ability?"
        f1_lo, f1_hi = _clim(f1_grid)
        f1_cmap = LinearSegmentedColormap.from_list(
            "f1_cmap",
            ["#fcfbfd", "#dadaeb", "#9e9ac8", "#6a51a3", "#3f007d"], N=256)

        im_f1 = ax_f1.imshow(f1_grid, origin="lower", cmap=f1_cmap,
                             vmin=f1_lo, vmax=f1_hi, aspect="auto", extent=extent)

        for yi in range(n_y):
            for xi in range(n_x):
                f1v = f1_grid[yi, xi]
                cx, cy = x_thresholds[xi], y_thresholds[yi]
                if not np.isfinite(f1v):
                    ax_f1.text(cx, cy, "n/a", ha="center", va="center",
                               fontsize=8, color="#999999")
                    continue
                rel = (f1v - f1_lo) / max(1e-9, f1_hi - f1_lo)
                text_col = "white" if rel > 0.55 else "black"
                ax_f1.text(cx, cy, f"{f1v:.3f}",
                           ha="center", va="center",
                           fontsize=10, color=text_col, fontweight="bold")

        ax_f1.set_xticks(x_thresholds)
        ax_f1.set_xticklabels([f"{v:.1f}" for v in x_thresholds], fontsize=9)
        ax_f1.set_yticks(y_thresholds)
        ax_f1.set_yticklabels([f"{v:.1f}" for v in y_thresholds], fontsize=10)
        ax_f1.set_xlabel("τ_SR  (median path Sharpe threshold, C2)",
                         fontsize=11, fontweight="bold", labelpad=8)
        ax_f1.set_ylabel("τ_FP  (frac. profitable paths threshold, C1)",
                         fontsize=11, fontweight="bold", labelpad=8)
        f1_split_lbl = "Val" if split_name == "val" else "Test"
        ax_f1.set_title(f"F1 = 2·TP / (2·TP+FP+FN)  [{f1_lo:.3f} – {f1_hi:.3f}]",
                        fontsize=11, fontweight="bold", pad=8)

        for xv in x_thresholds - sr_step / 2:
            ax_f1.axvline(xv, color="white", lw=0.8, alpha=0.6)
        ax_f1.axvline(x_thresholds[-1] + sr_step / 2, color="white", lw=0.8, alpha=0.6)
        for yv in y_thresholds:
            ax_f1.axhline(yv - 0.1, color="white", lw=0.8, alpha=0.6)
        ax_f1.axhline(y_thresholds[-1] + 0.1, color="white", lw=0.8, alpha=0.6)

        cbar_f1 = fig.colorbar(im_f1, ax=ax_f1, fraction=0.038, pad=0.02)
        cbar_f1.set_label("F1 score (filter logic)", fontsize=9)
        cbar_f1.ax.tick_params(labelsize=8)

        # Best F1 cell highlight
        f1_for_best = np.where(np.isfinite(f1_grid), f1_grid, -np.inf)
        best_yi_f, best_xi_f = np.unravel_index(np.argmax(f1_for_best), f1_grid.shape)
        if np.isfinite(f1_grid[best_yi_f, best_xi_f]):
            ax_f1.add_patch(plt.Rectangle(
                (x_thresholds[best_xi_f] - sr_step / 2, y_thresholds[best_yi_f] - 0.1),
                sr_step, 0.2, fill=False, edgecolor="black", lw=2.2))

        # ┏━━━━━━━━━━ Adjust layout and save figure ━━━━━━━━━━┓
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        out_path = os.path.join(output_dir, f"cpcv_edge_heatmap_{split_name}.png")
        plt.savefig(out_path, dpi=180, bbox_inches="tight", facecolor="white")
        plt.close()
        
        # ┏━━━━━━━━━━ Summary message ━━━━━━━━━━┓
        bx_a, by_a = x_thresholds[best_xi_a], y_thresholds[best_yi_a]
        msg = (f"[plot_cpcv_edge_heatmap] {split_name.upper()}: "
               f"best acc = {accuracy[best_yi_a, best_xi_a]*100:.1f}% "
               f"@(τ_SR={bx_a:.1f}, τ_FP={by_a:.1f})")
        if np.isfinite(precision[best_yi_p, best_xi_p]):
            bx_p, by_p = x_thresholds[best_xi_p], y_thresholds[best_yi_p]
            msg += (f"   |   best prec = {precision[best_yi_p, best_xi_p]*100:.1f}% "
                    f"@(τ_SR={bx_p:.1f}, τ_FP={by_p:.1f})")
        msg += f"  -> {out_path}"
        print(msg)
        out_paths[split_name] = out_path

    return output_dir


def plot_results_radar(data_root: str = "/home/pablo/M2_DS/Secondary-Model/src/Output",
                       output_dir: str = "/home/pablo/M2_DS/Secondary-Model/src/Output/Analysis/Results",
                       metric: str = "prec_delta",
                       tau_sr: float = 1.5,
                       tau_fp: float = 0.8,
                       cv_max: float = None,
                       require_constraint: bool = False):
    """Radar / spider chart of M2 results.

    Layout  : 4 rows (M1) x 2 cols (direction) = 8 subplots.
    Axes    : 8 spokes, one per granularity (1d → 30m).
    Polygons: one per M2 model, colour-coded.
    Values  : mean of `metric` across all 20 assets for that (M1, M2, direction, gran).
              Missing cells are set to 0 so the polygon closes cleanly.

    metric options: 'prec_delta' | 'm2_return'
    Saved as: results_radar_{metric}.png
    """
    import os
    import json
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from pathlib import Path

    M1_MODELS  = ["Tirex", "Chronos2", "Fincast", "Kronos"]
    M1_LABELS  = {"Tirex": "TiRex", "Chronos2": "Chronos-2",
                  "Fincast": "Fincast", "Kronos": "Kronos"}
    M2_MODELS  = ["rf", "autogluon", "tabpfn", "tabicl", "ctts"]
    M2_LABELS  = {"rf": "Random Forest", "autogluon": "AutoGluon",
                  "tabpfn": "TabPFN", "tabicl": "TabICL", "ctts": "CTTS"}
    GRAN_ORDER = ["1d", "12h", "8h", "6h", "4h", "2h", "1h", "30m"]
    DIRECTIONS = ["UP", "DOWN"]
    GRAN_DIR   = {g: f"{g}_tp" for g in GRAN_ORDER}
    M2_KEYS    = {
        "rf":        "rf_backtest_all_features",
        "autogluon": "autogluon_backtest_all_features",
        "tabpfn":    "tabpfn_backtest_all_features",
        "tabicl":    "tabicl_backtest_all_features",
        "ctts":      "ctts_backtest_all_features",
    }

    # Muted palette consistent with best_m2_per_gran.png and focused radar
    M2_COLORS = {
        "rf":        "#7FB069",   # muted green
        "autogluon": "#E89A4F",   # muted orange
        "tabpfn":    "#6FA8DC",   # muted blue
        "tabicl":    "#C28EC9",   # muted purple
        "ctts":      "#D97374",   # muted red
    }
    M2_LS = {
        "rf": "-", "autogluon": "--", "tabpfn": "-.",
        "tabicl": ":", "ctts": (0, (3, 1, 1, 1)),
    }

    os.makedirs(output_dir, exist_ok=True)
    data_root = Path(data_root)
    edge_root  = data_root / "Analysis" / "Edge_NoCal"

    VERDICT_DOT = {"GREEN": "#2ca02c", "RED": "#d62728"}

    # ── load backtest metrics ────────────────────────────────────────────────
    vals: dict = {}
    for m1 in M1_MODELS:
        vals[m1] = {}
        for m2 in M2_MODELS:
            vals[m1][m2] = {}
            key = M2_KEYS[m2]
            for direction in DIRECTIONS:
                vals[m1][m2][direction] = {}
                for gran, gran_dir in GRAN_DIR.items():
                    path = (data_root / m1 / m2 / direction
                            / "Utility_Score_NoCal" / gran_dir / "analysis_summary.json")
                    try:
                        b = json.load(open(path)).get(key, {})
                        if not b or b.get("m2_win_rate") is None:
                            raise ValueError
                        if metric == "prec_delta":
                            # plot full M2 precision (not delta) — units: %
                            v = b["m2_win_rate"]
                        else:
                            v = b.get("m2_total_return", 0.0)
                        vals[m1][m2][direction][gran] = v
                    except Exception:
                        vals[m1][m2][direction][gran] = None

    # ── load convergence verdicts (recomputed with tau_sr / tau_fp / C3) ──────
    # C1: frac_profitable >= tau_fp
    # C2: median_path_sharpe >= tau_sr
    # C3: val_mean_ret > 0  (from backtest analysis_summary)
    # GREEN = all 3 pass, AMBER = C1+C2 pass but C3 fails OR exactly one of C1/C2,
    # RED   = neither C1 nor C2 passes
    verdicts: dict = {}
    for m1 in M1_MODELS:
        verdicts[m1] = {}
        for m2 in M2_MODELS:
            verdicts[m1][m2] = {}
            for direction in DIRECTIONS:
                verdicts[m1][m2][direction] = {}
                for gran, gran_dir in GRAN_DIR.items():
                    path_edge = (edge_root / m1 / m2 / direction
                                 / f"edge_summary_{gran}.json")
                    path_bt   = (data_root / m1 / m2 / direction
                                 / "Utility_Score_NoCal" / gran_dir
                                 / "analysis_summary.json")
                    try:
                        entry  = json.load(open(path_edge)).get(gran, {})
                        # GREEN = constraint_satisfied AND CV<0.5; RED otherwise
                        try:
                            bt     = json.load(open(path_bt))
                            tkey_v = f"{m2}_temporal_all_features"
                            constr = bool(bt.get(tkey_v, {}).get("Val_selective", {})
                                         .get("constraint_satisfied", False))
                        except Exception:
                            constr = False
                        p_e = np.array(entry.get("path_total_rets", []), dtype=float)
                        cv_e = float(np.std(p_e)/(abs(np.mean(p_e))+1e-6)) if len(p_e)>1 else 99.0
                        v = "GREEN" if (constr and cv_e < 0.5) else "RED"
                        verdicts[m1][m2][direction][gran] = v
                    except Exception:
                        verdicts[m1][m2][direction][gran] = None

    # ── load m1_baseline_prec per (m1, direction, gran) ─────────────────────
    # Use the first available M2 model's edge summary (m1 baseline is the same)
    m1_prec: dict = {}
    for m1 in M1_MODELS:
        m1_prec[m1] = {}
        for direction in DIRECTIONS:
            m1_prec[m1][direction] = {}
            for gran in GRAN_ORDER:
                val = None
                for m2 in M2_MODELS:
                    path = (edge_root / m1 / m2 / direction
                            / f"edge_summary_{gran}.json")
                    try:
                        entry = json.load(open(path)).get(gran, {})
                        val = entry.get("m1_baseline_prec")
                        if val is not None:
                            break
                    except Exception:
                        continue
                m1_prec[m1][direction][gran] = val

    # ── Optional reliability filters ─────────────────────────────────────
    #   * cv_max:             keep only cells with CPCV CV < cv_max
    #   * require_constraint: keep only cells with constraint_satisfied=True
    if cv_max is not None or require_constraint:
        n_kept = n_dropped = 0
        for m1 in M1_MODELS:
            for m2 in M2_MODELS:
                for direction in DIRECTIONS:
                    for gran, gran_dir in GRAN_DIR.items():
                        if vals[m1][m2][direction][gran] is None:
                            continue
                        # CV from edge summary
                        if cv_max is not None:
                            path_edge = (edge_root / m1 / m2 / direction
                                         / f"edge_summary_{gran}.json")
                            try:
                                entry = json.load(open(path_edge)).get(gran, {})
                                p_e = np.array(entry.get("path_total_rets", []), dtype=float)
                                cv_e = float(np.std(p_e)/(abs(np.mean(p_e))+1e-6)) if len(p_e) > 1 else 99.0
                            except Exception:
                                cv_e = 99.0
                        else:
                            cv_e = 0.0
                        # constraint_satisfied from backtest analysis_summary
                        if require_constraint:
                            path_bt = (data_root / m1 / m2 / direction
                                       / "Utility_Score_NoCal" / gran_dir
                                       / "analysis_summary.json")
                            try:
                                bt = json.load(open(path_bt))
                                tkey_v = f"{m2}_temporal_all_features"
                                constr = bool(bt.get(tkey_v, {}).get("Val_selective", {})
                                               .get("constraint_satisfied", False))
                            except Exception:
                                constr = False
                        else:
                            constr = True
                        drop = (cv_max is not None and cv_e >= cv_max) or \
                               (require_constraint and not constr)
                        if drop:
                            vals[m1][m2][direction][gran] = None
                            n_dropped += 1
                        else:
                            n_kept += 1
        tag = []
        if cv_max is not None:        tag.append(f"CV<{cv_max}")
        if require_constraint:        tag.append("constr=True")
        print(f"[plot_results_radar] {' & '.join(tag)}: kept {n_kept}, dropped {n_dropped}")

    # ── figure layout: 2 rows (UP/DOWN) × 4 cols (M1) + right legend ─────────
    # Layout: 2 rows × 5 cols where col 4 is a narrow legend axes
    n_data_cols = len(M1_MODELS)
    fig = plt.figure(figsize=(8.0 * n_data_cols, 8.5 * 2 + 1.8), dpi=180)
    fig.patch.set_facecolor("white")
    # 2 data rows + 1 legend row at bottom
    gs = fig.add_gridspec(3, n_data_cols,
                          height_ratios=[1, 1, 0.05],
                          hspace=0.08, wspace=0.15)

    N = len(GRAN_ORDER)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]
    # Index of 8h spoke for value annotations
    spoke_8h = GRAN_ORDER.index("8h")

    metric_label = "M2 Precision (%)" if metric == "prec_delta" else "M2 Return (%)"
    use_zone_bg  = (metric == "m2_return")

    # ── global radial limits (5th–95th percentile) ───────────────────────────
    global_all = [vals[m1][m2][direction][g]
                  for m1 in M1_MODELS for m2 in M2_MODELS
                  for direction in DIRECTIONS for g in GRAN_ORDER
                  if vals[m1][m2][direction][g] is not None]
    p05 = float(np.percentile(global_all,  5)) if global_all else -1.0
    p95 = float(np.percentile(global_all, 95)) if global_all else  1.0
    g_vmin = min(0.0, p05) * 1.08
    g_vmax = p95 * 1.15

    # ── sqrt radial transform helpers ─────────────────────────────────────────
    # Applies signed-sqrt: f(x) = sign(x) * sqrt(|x|)
    # This zooms into values near zero while preserving outer information.
    def _r(x):
        """Signed sqrt transform for radial axis."""
        return float(np.sign(x) * np.sqrt(abs(x)))

    def _r_inv(y):
        """Inverse: y -> sign(y)*y^2"""
        return float(np.sign(y) * y ** 2)

    r_vmin = _r(g_vmin)
    r_vmax = _r(g_vmax)

    def _draw_spider(ax, m1, direction):
        """Draw one spider subplot using signed-sqrt radial scale."""
        # Compute local limits from this subplot's data only
        local_vals = [vals[m1][m2][direction][g]
                      for m2 in M2_MODELS for g in GRAN_ORDER
                      if vals[m1][m2][direction][g] is not None]
        if metric == "prec_delta":
            # Precision metric (units: %). Zoom into the meaningful 30–95 band:
            # values below ~30% are extremely rare and squash the visual range.
            if local_vals:
                lp05 = float(np.percentile(local_vals,  5))
                lp95 = float(np.percentile(local_vals, 95))
                l_vmin = max(30.0, lp05 - 5.0)
                l_vmax = min(98.0, lp95 + 5.0)
            else:
                l_vmin, l_vmax = 30.0, 90.0
            # Ensure the 50% reference ring sits inside the chart area
            if l_vmin > 50.0: l_vmin = 50.0 - 5.0
            if l_vmax < 55.0: l_vmax = 60.0
        else:
            if local_vals:
                lp05 = float(np.percentile(local_vals,  5))
                lp95 = float(np.percentile(local_vals, 95))
                l_vmin = min(0.0, lp05) * 1.08
                l_vmax = lp95 * 1.15
            else:
                l_vmin, l_vmax = g_vmin, g_vmax
            # Always leave a visible red (loss) band even when all surviving values
            # are positive.
            if l_vmin >= 0.0:
                l_vmin = -max(0.05 * max(l_vmax, 1.0), 0.5)
        vmin = _r(l_vmin)
        vmax = _r(l_vmax)

        # ── axis cosmetics ────────────────────────────────────────────────
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(GRAN_ORDER, fontsize=10.5, fontweight="semibold",
                           color="#222222")
        ax.tick_params(axis='x', pad=4)
        ax.set_facecolor("white")
        ax.spines["polar"].set_visible(True)
        ax.spines["polar"].set_color("#777777")
        ax.spines["polar"].set_linewidth(1.0)
        ax.grid(color="#aaaaaa", linewidth=0.8, linestyle="-", alpha=1.0)
        # Disable default y-grid (concentric rings) — we draw them manually with a gap
        ax.yaxis.grid(False)
        ax.set_ylim(vmin, vmax)
        # For the precision metric, shift the radial origin so vmin sits at the
        # plot CENTRE — eliminates the empty "donut hole" caused by zooming into
        # the meaningful 30–95 % band.
        if metric == "prec_delta":
            ax.set_rorigin(vmin)

        # Draw zone shading FIRST (before any early-return) so even subplots
        # with no surviving points still display the green/red reference rings.
        ang_full = np.linspace(0, 2 * np.pi, 300)
        r_zero   = _r(0.0)   # 0 in transformed space

        if use_zone_bg:
            ax_vmax = ax.get_ylim()[1]
            ax_vmin = ax.get_ylim()[0]
            ax.fill_between(ang_full, np.full(300, r_zero), np.full(300, ax_vmax),
                            color="#a8d5b5", alpha=0.45, zorder=0)
            if ax_vmin < r_zero:
                ax.fill_between(ang_full, np.full(300, ax_vmin), np.full(300, r_zero),
                                color="#f0a8a8", alpha=0.55, zorder=0)
            ax.plot(ang_full, np.full(300, r_zero),
                    color="#777777", linewidth=1.0, linestyle="-", zorder=1)

        all_vals = [vals[m1][m2][direction][g]
                    for m2 in M2_MODELS for g in GRAN_ORDER
                    if vals[m1][m2][direction][g] is not None]
        if not all_vals:
            return
        else:
            # Precision metric: red zone = below 50%, green zone = ≥ 50%.
            # No amber (break-even) ring.
            r_50 = _r(50.0)
            if vmin < r_50:
                ax.fill_between(ang_full, np.full(300, vmin), np.full(300, r_50),
                                color="#f0a8a8", alpha=0.45, zorder=0)
            if r_50 < vmax:
                ax.fill_between(ang_full, np.full(300, r_50), np.full(300, vmax),
                                color="#a8d5b5", alpha=0.45, zorder=0)
            ax.plot(ang_full, np.full(300, r_50),
                    color="#777777", linewidth=1.0, linestyle="-", zorder=1)

        # ── concentric ring ticks equally spaced in transformed (radial) space ──
        n_rings = 5
        # Equal spacing in transformed space → equal visual ring gaps
        r_ticks    = np.linspace(vmin, vmax, n_rings + 2)[1:-1].tolist()
        orig_ticks = [_r_inv(r) for r in r_ticks]   # back to original for labels
        ax.set_yticks(r_ticks)
        ax.set_yticklabels([""] * len(r_ticks))

        ang_8h   = angles[spoke_8h]
        # Place ring labels in the gap between 30m (last) and 1d (first) spokes
        ang_30m  = angles[GRAN_ORDER.index("30m")]
        ang_1d   = angles[GRAN_ORDER.index("1d")]
        # wrap-around gap: average the two angles accounting for circular wrap
        ang_gap  = (ang_30m + ang_1d + 2 * np.pi) / 2.0

        # Custom concentric rings drawn as arcs with an angular gap at ang_gap;
        # label sits in the gap so the ring appears interrupted by the number (---79---).
        _gap_half = np.deg2rad(3.5)   # tighter gap
        ang_arc = np.linspace(ang_gap + _gap_half,
                              ang_gap + 2 * np.pi - _gap_half, 300)
        _r_span = r_ticks[-1] - r_ticks[0] if len(r_ticks) > 1 else 1.0
        for i, (rv_orig, rv_r) in enumerate(zip(orig_ticks, r_ticks)):
            ax.plot(ang_arc, np.full_like(ang_arc, rv_r),
                    color="#aaaaaa", linewidth=0.8, linestyle="-",
                    alpha=1.0, zorder=1)
            # Pull the outermost label slightly inward so it doesn't kiss the spine
            r_label = rv_r - 0.04 * _r_span if i == len(r_ticks) - 1 else rv_r
            ax.annotate(f"{rv_orig:.0f}",
                        xy=(ang_gap, r_label),
                        fontsize=9, color="#222222", fontweight="bold",
                        ha="center", va="center", zorder=9)

        # ── polygons ──────────────────────────────────────────────────────
        for m2 in M2_MODELS:
            raw  = [vals[m1][m2][direction][g] for g in GRAN_ORDER]
            data = [_r(float(np.clip(v if v is not None else 0.0, l_vmin, l_vmax)))
                    for v in raw]
            data_closed = data + data[:1]
            ax.plot(angles, data_closed,
                    color=M2_COLORS[m2], linestyle=M2_LS[m2],
                    linewidth=2.0, zorder=3, solid_capstyle="round")
            # polygon fill removed — background zones carry the meaning

            # (verdict dots removed — reliability is conveyed elsewhere)


        dir_arrow = r"$\uparrow$" if direction == "UP" else r"$\downarrow$"
        ax.set_title(f"{M1_LABELS[m1]}  {dir_arrow}",
                     fontsize=16, fontweight="bold", pad=20, color="#111111")

    # ── draw all subplots: row=direction, col=M1 ─────────────────────────────
    for ri, direction in enumerate(DIRECTIONS):
        for ci, m1 in enumerate(M1_MODELS):
            ax = fig.add_subplot(gs[ri, ci], polar=True)
            _draw_spider(ax, m1, direction)

        pass  # row label removed

    # ── horizontal legend — figure-level, aligned with spider chart columns ──
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    # Invisible axes in legend row just to hold the row height
    leg_ax = fig.add_subplot(gs[2, :])
    leg_ax.set_axis_off()

    m2_handles = [
        Line2D([0], [0], color=M2_COLORS[m2], linewidth=2.5,
               linestyle=M2_LS[m2], label=M2_LABELS[m2])
        for m2 in M2_MODELS
    ]
    zone_handles = [
        Patch(facecolor="#a8d5b5", edgecolor="#aaaaaa", alpha=0.8,
              label="M2 Return $> 0$ (Profitable)"),
        Patch(facecolor="#f0a8a8", edgecolor="#aaaaaa", alpha=0.8,
              label="M2 Return $< 0$ (Loss)"),
    ]
    all_handles = m2_handles + [Line2D([], [], visible=False)] + zone_handles

    # Get the x-extent of the spider chart area from the first and last subplot
    fig.canvas.draw()   # force layout so axes positions are known
    ax_left  = fig.axes[0]   # top-left spider (UP, Tirex)
    ax_right = fig.axes[3]   # top-right spider (UP, Kronos)
    inv = fig.transFigure.inverted()
    x0 = inv.transform(ax_left.transAxes.transform([0, 0]))[0]
    x1 = inv.transform(ax_right.transAxes.transform([1, 0]))[0]
    x_center = (x0 + x1) / 2

    # y position: centre of the legend axes row
    leg_bbox = inv.transform(leg_ax.transAxes.transform([0.5, 0.5]))
    y_center = leg_bbox[1]

    leg = fig.legend(handles=all_handles,
                     loc="center",
                     bbox_to_anchor=(x_center, y_center),
                     bbox_transform=fig.transFigure,
                     fontsize=18, frameon=True, framealpha=0.95,
                     edgecolor="#cccccc",
                     ncol=len(all_handles), handlelength=2.4, handleheight=1.6,
                     markerscale=1.8,
                     borderpad=0.8, labelspacing=0.2, columnspacing=0.8)
    leg.get_frame().set_linewidth(0.8)


    suffix = ""
    if cv_max is not None:        suffix += f"_cv{str(cv_max).replace('.','p')}"
    if require_constraint:        suffix += "_constr"
    fname = f"results_radar_{metric}{suffix}.png"
    out_path = os.path.join(output_dir, fname)
    fig.savefig(out_path, dpi=200, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close(fig)
    print(f"[plot_results_radar] {out_path}")
    return out_path


if __name__ == "__main__":
    plot_cpcv_constraint_bars()
    plot_cpcv_edge_heatmap()
    plot_results_3d()
    plot_results_3d(metric="m2_return")
    plot_results_radar()
    plot_results_radar(metric="m2_return")


def plot_results_radar_focused(
    data_root: str = "/home/pablo/M2_DS/Secondary-Model/src/Output",
    output_dir: str = "/home/pablo/M2_DS/Secondary-Model/src/Output/Analysis/Results",
    m1: str = "Kronos",
    m2_models: tuple = ("rf", "tabpfn", "ctts"),
    metric: str = "m2_return",
):
    """Two side-by-side spider charts for ONE M1 (UP and DOWN), with only the
    selected ``m2_models`` plotted. Colours match ``plot_best_m2_per_gran`` so
    the figure is consistent across the paper.

    Layout:  1 row × 2 cols  (col 0: UP, col 1: DOWN)
    Saved as: results_radar_focused_{m1}_{metric}.png
    """
    import os, json
    from pathlib import Path
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    M2_LABELS = {"rf": "Random Forest", "autogluon": "AutoGluon",
                 "tabpfn": "TabPFN", "tabicl": "TabICL", "ctts": "CTTS"}
    GRAN_ORDER = ["1d", "12h", "8h", "6h", "4h", "2h", "1h", "30m"]
    DIRECTIONS = ["UP", "DOWN"]
    GRAN_DIR   = {g: f"{g}_tp" for g in GRAN_ORDER}
    M2_KEYS    = {m2: f"{m2}_backtest_all_features" for m2 in
                  ("rf", "autogluon", "tabpfn", "tabicl", "ctts")}

    # ── Palette consistent with plot_best_m2_per_gran ────────────────────
    M2_COLORS = {
        "rf":        "#7FB069",   # muted green
        "autogluon": "#E89A4F",   # muted orange
        "tabpfn":    "#6FA8DC",   # muted blue
        "tabicl":    "#C28EC9",   # muted purple
        "ctts":      "#D97374",   # muted red
    }
    M2_LS = {"rf": "-", "autogluon": "--", "tabpfn": "-",
             "tabicl": ":",  "ctts": "-"}

    VERDICT_DOT = {"GREEN": "#2ca02c", "RED": "#d62728"}

    os.makedirs(output_dir, exist_ok=True)
    data_root = Path(data_root)
    edge_root = data_root / "Analysis" / "Edge_NoCal"

    # ── Load values for the selected M1 × M2s only ──────────────────────
    vals: dict = {m2: {d: {} for d in DIRECTIONS} for m2 in m2_models}
    for m2 in m2_models:
        key = M2_KEYS[m2]
        for direction in DIRECTIONS:
            for gran, gd in GRAN_DIR.items():
                path = (data_root / m1 / m2 / direction
                        / "Utility_Score_NoCal" / gd / "analysis_summary.json")
                v = None
                try:
                    b = json.load(open(path)).get(key, {})
                    if b and b.get("m2_win_rate") is not None:
                        if metric == "prec_delta":
                            v = b["m2_win_rate"] - b["m1_win_rate"]
                        else:
                            v = b.get("m2_total_return", 0.0)
                except Exception:
                    v = None
                vals[m2][direction][gran] = v

    # ── Reliability verdicts (CV<0.5 AND constraint_satisfied) ──────────
    verdicts: dict = {m2: {d: {} for d in DIRECTIONS} for m2 in m2_models}
    for m2 in m2_models:
        for direction in DIRECTIONS:
            for gran, gd in GRAN_DIR.items():
                path_edge = (edge_root / m1 / m2 / direction
                             / f"edge_summary_{gran}.json")
                path_bt   = (data_root / m1 / m2 / direction
                             / "Utility_Score_NoCal" / gd
                             / "analysis_summary.json")
                v = None
                try:
                    entry = json.load(open(path_edge)).get(gran, {})
                    try:
                        bt = json.load(open(path_bt))
                        tkey_v = f"{m2}_temporal_all_features"
                        constr = bool(bt.get(tkey_v, {}).get("Val_selective", {})
                                       .get("constraint_satisfied", False))
                    except Exception:
                        constr = False
                    p_e = np.array(entry.get("path_total_rets", []), dtype=float)
                    cv_e = float(np.std(p_e) / (abs(np.mean(p_e)) + 1e-6)) if len(p_e) > 1 else 99.0
                    v = "GREEN" if (constr and cv_e < 0.5) else "RED"
                except Exception:
                    v = None
                verdicts[m2][direction][gran] = v

    # ── signed-sqrt radial transform ───────────────────────────────────
    def _r(x):     return float(np.sign(x) * np.sqrt(abs(x)))
    def _r_inv(y): return float(np.sign(y) * y * y)

    # ── figure ─────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(13, 7.6), dpi=180)
    fig.patch.set_facecolor("white")
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 0.10],
                          hspace=0.05, wspace=0.18)

    N = len(GRAN_ORDER)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]
    use_zone_bg = (metric == "m2_return")

    def _draw_one(ax, direction):
        local = [vals[m2][direction][g]
                 for m2 in m2_models for g in GRAN_ORDER
                 if vals[m2][direction][g] is not None]
        if local:
            lp05 = float(np.percentile(local, 5))
            lp95 = float(np.percentile(local, 95))
            l_vmin = min(0.0, lp05) * 1.08
            l_vmax = lp95 * 1.15
        else:
            l_vmin, l_vmax = -1.0, 1.0
        vmin = _r(l_vmin); vmax = _r(l_vmax)

        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(GRAN_ORDER, fontsize=11, fontweight="semibold",
                           color="#222222")
        ax.tick_params(axis="x", pad=6)
        ax.set_facecolor("white")
        ax.spines["polar"].set_visible(True)
        ax.spines["polar"].set_color("#777777")
        ax.spines["polar"].set_linewidth(1.0)
        ax.grid(color="#aaaaaa", linewidth=0.8, linestyle="-", alpha=1.0)
        ax.yaxis.grid(False)
        ax.set_ylim(vmin, vmax)

        ang_full = np.linspace(0, 2 * np.pi, 300)
        r_zero = _r(0.0)

        if use_zone_bg:
            ax_vmax = ax.get_ylim()[1]; ax_vmin = ax.get_ylim()[0]
            ax.fill_between(ang_full, np.full(300, r_zero), np.full(300, ax_vmax),
                            color="#a8d5b5", alpha=0.45, zorder=0)
            if ax_vmin < r_zero:
                ax.fill_between(ang_full, np.full(300, ax_vmin), np.full(300, r_zero),
                                color="#f0a8a8", alpha=0.55, zorder=0)
            ax.plot(ang_full, np.full(300, r_zero), color="#777777",
                    linewidth=0.9, linestyle="-", zorder=1)

        # Concentric ring labels with arc-gap (---79--- effect)
        n_rings = 5
        r_ticks = np.linspace(vmin, vmax, n_rings + 2)[1:-1].tolist()
        # Snap the ring closest to the zone boundary onto it exactly,
        # so the visible label reads "0" (return) / "50" (precision)
        # instead of e.g. "−1" / "49".
        if abs(r_zone - vmin) > 1e-9 and abs(r_zone - vmax) > 1e-9:
            i_snap = int(np.argmin([abs(rt - r_zone) for rt in r_ticks]))
            r_ticks[i_snap] = r_zone
        orig_ticks = [_r_inv(r) for r in r_ticks]
        ax.set_yticks(r_ticks)
        ax.set_yticklabels([""] * len(r_ticks))
        ang_30m = angles[GRAN_ORDER.index("30m")]
        ang_1d  = angles[GRAN_ORDER.index("1d")]
        ang_gap = (ang_30m + ang_1d + 2 * np.pi) / 2.0
        gap_half = np.deg2rad(3.5)
        ang_arc = np.linspace(ang_gap + gap_half,
                              ang_gap + 2 * np.pi - gap_half, 300)
        r_span = r_ticks[-1] - r_ticks[0] if len(r_ticks) > 1 else 1.0
        for i, (rv_orig, rv_r) in enumerate(zip(orig_ticks, r_ticks)):
            ax.plot(ang_arc, np.full_like(ang_arc, rv_r),
                    color="#aaaaaa", linewidth=0.8, linestyle="-",
                    alpha=1.0, zorder=1)
            r_label = rv_r - 0.03 * r_span if i == len(r_ticks) - 1 else rv_r
            ax.annotate(f"{rv_orig:.0f}", xy=(ang_gap, r_label),
                        fontsize=9, color="#222222", fontweight="bold",
                        ha="center", va="center", zorder=9)

        # Polygons
        for m2 in m2_models:
            raw = [vals[m2][direction][g] for g in GRAN_ORDER]
            data = [_r(float(np.clip(v if v is not None else 0.0, l_vmin, l_vmax)))
                    for v in raw]
            data_closed = data + data[:1]
            ax.plot(angles, data_closed, color=M2_COLORS[m2],
                    linestyle=M2_LS[m2], linewidth=2.4, zorder=3,
                    solid_capstyle="round")

        dir_arrow = r"$\uparrow$" if direction == "UP" else r"$\downarrow$"
        ax.set_title(f"{m1}  {dir_arrow}",
                     fontsize=15, fontweight="bold", pad=18, color="#111111")

    for ci, dr in enumerate(DIRECTIONS):
        ax = fig.add_subplot(gs[0, ci], polar=True)
        _draw_one(ax, dr)

    # ── Legend (one row, bottom) ────────────────────────────────────────
    leg_ax = fig.add_subplot(gs[1, :]); leg_ax.set_axis_off()
    m2_handles = [Line2D([0], [0], color=M2_COLORS[m2], linewidth=2.6,
                         linestyle=M2_LS[m2], label=M2_LABELS[m2])
                  for m2 in m2_models]
    zone_handles = []
    if use_zone_bg:
        zone_handles = [
            Patch(facecolor="#a8d5b5", edgecolor="#aaaaaa", alpha=0.8,
                  label=r"M2 Return $> 0$ (Profitable)"),
            Patch(facecolor="#f0a8a8", edgecolor="#aaaaaa", alpha=0.8,
                  label=r"M2 Return $< 0$ (Loss)"),
        ]
    handles = m2_handles + zone_handles

    fig.canvas.draw()
    inv = fig.transFigure.inverted()
    ax_l = fig.axes[0]; ax_r = fig.axes[1]
    x0 = inv.transform(ax_l.transAxes.transform([0, 0]))[0]
    x1 = inv.transform(ax_r.transAxes.transform([1, 0]))[0]
    x_center = (x0 + x1) / 2
    leg_bbox = inv.transform(leg_ax.transAxes.transform([0.5, 0.5]))
    y_center = leg_bbox[1]

    fig.legend(handles=handles, loc="center",
               bbox_to_anchor=(x_center, y_center),
               bbox_transform=fig.transFigure,
               fontsize=12, frameon=True, framealpha=0.95,
               edgecolor="#cccccc", ncol=len(handles),
               handlelength=2.4, handleheight=1.4, markerscale=1.4,
               borderpad=0.6, columnspacing=1.4)

    fname = f"results_radar_focused_{m1}_{metric}.png"
    out = os.path.join(output_dir, fname)
    fig.savefig(out, dpi=200, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close(fig)
    print(f"[plot_results_radar_focused] {out}")
    return out


def plot_kronos_down_combined(
    data_root: str = "/home/pablo/M2_DS/Secondary-Model/src/Output",
    output_dir: str = "/home/pablo/M2_DS/Secondary-Model/src/Output/Analysis/Results",
    m2_models: tuple = ("rf", "tabpfn", "ctts"),
):
    """One figure with two side-by-side spider charts for **Kronos / DOWN**:

      • LEFT  — M2 total return (%):  green > 0, red < 0
      • RIGHT — M2 precision    (%):  green ≥ 50, red < 50

    Three M2 models drawn (default: rf, tabpfn, ctts) using the muted palette
    consistent with ``best_m2_per_gran.png``. A single shared bottom legend.
    """
    import os, json
    from pathlib import Path
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    M1, DIR = "Kronos", "DOWN"
    M2_LABELS = {"rf": "Random Forest", "autogluon": "AutoGluon",
                 "tabpfn": "TabPFN", "tabicl": "TabICL", "ctts": "CTTS"}
    GRAN_ORDER = ["1d", "12h", "8h", "6h", "4h", "2h", "1h", "30m"]
    M2_KEYS    = {m2: f"{m2}_backtest_all_features"
                  for m2 in ("rf", "autogluon", "tabpfn", "tabicl", "ctts")}

    M2_COLORS = {
        "rf":        "#7FB069",   # muted green
        "autogluon": "#E89A4F",   # muted orange
        "tabpfn":    "#6FA8DC",   # muted blue
        "tabicl":    "#C28EC9",   # muted purple
        "ctts":      "#D97374",   # muted red
    }
    M2_LS = {"rf": "-", "autogluon": "--", "tabpfn": "-",
             "tabicl": ":", "ctts": "-"}

    os.makedirs(output_dir, exist_ok=True)
    data_root = Path(data_root)

    # ── load values for both metrics ─────────────────────────────────────
    vals_ret  = {m2: {} for m2 in m2_models}
    vals_prec = {m2: {} for m2 in m2_models}
    for m2 in m2_models:
        key = M2_KEYS[m2]
        for g in GRAN_ORDER:
            p = (data_root / M1 / m2 / DIR / "Utility_Score_NoCal"
                 / f"{g}_tp" / "analysis_summary.json")
            r_v = pr_v = None
            try:
                b = json.load(open(p)).get(key, {})
                if b and b.get("m2_win_rate") is not None:
                    r_v  = b.get("m2_total_return", 0.0)
                    pr_v = b["m2_win_rate"]            # already in %
            except Exception:
                pass
            vals_ret[m2][g]  = r_v
            vals_prec[m2][g] = pr_v

    # ── signed-sqrt radial transform ────────────────────────────────────
    def _r(x):     return float(np.sign(x) * np.sqrt(abs(x)))
    def _r_inv(y): return float(np.sign(y) * y * y)

    fig = plt.figure(figsize=(13, 7.6), dpi=180)
    fig.patch.set_facecolor("white")
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 0.10],
                          hspace=0.05, wspace=0.18,
                          left=0.06, right=0.94)

    N = len(GRAN_ORDER)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    # Fraction of each chart's radial extent (in transformed sqrt space) to
    # reserve for the inner RED zone. Keeping this constant across both
    # subplots makes their red discs visually similar in size.
    RED_FRAC = 0.35

    def _draw(ax, vals: dict, title: str, kind: str):
        """kind ∈ {"return", "precision"}"""
        local = [vals[m2][g] for m2 in m2_models for g in GRAN_ORDER
                 if vals[m2][g] is not None]
        if kind == "return":
            if local:
                lp95 = float(np.percentile(local, 95))
                l_vmax = lp95 * 1.15
            else:
                l_vmax = 1.0
            zone_orig = 0.0
            r_zone    = _r(zone_orig)
            r_vmax    = _r(l_vmax)
            # Force the red zone to occupy RED_FRAC of the radial extent in
            # transformed (sqrt) space.  Solve for l_vmin such that
            # (r_zone − r_vmin) / (r_vmax − r_vmin) == RED_FRAC.
            r_vmin    = r_zone - RED_FRAC / (1.0 - RED_FRAC) * (r_vmax - r_zone)
            l_vmin    = _r_inv(r_vmin)
        else:  # precision (units: %)
            if local:
                lp95 = float(np.percentile(local, 95))
                l_vmax = min(98.0, lp95 + 5.0)
            else:
                l_vmax = 90.0
            if l_vmax < 55.0: l_vmax = 60.0
            zone_orig = 50.0
            r_zone    = _r(zone_orig)
            r_vmax    = _r(l_vmax)
            r_vmin    = r_zone - RED_FRAC / (1.0 - RED_FRAC) * (r_vmax - r_zone)
            l_vmin    = _r_inv(r_vmin)

        vmin = _r(l_vmin); vmax = _r(l_vmax)

        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(GRAN_ORDER, fontsize=15, fontweight="semibold",
                           color="#222222")
        ax.tick_params(axis="x", pad=6)
        ax.set_facecolor("white")
        ax.spines["polar"].set_visible(True)
        ax.spines["polar"].set_color("#777777")
        ax.spines["polar"].set_linewidth(1.0)
        ax.grid(color="#aaaaaa", linewidth=0.8, linestyle="-", alpha=1.0)
        ax.yaxis.grid(False)
        ax.set_ylim(vmin, vmax)
        # Always apply set_rorigin: keeps the inner red zone proportionally
        # equal across both subplots (controlled by RED_FRAC).
        ax.set_rorigin(vmin)

        ang_full = np.linspace(0, 2 * np.pi, 300)
        ax_vmin, ax_vmax = ax.get_ylim()
        if ax_vmin < r_zone:
            ax.fill_between(ang_full, np.full(300, ax_vmin), np.full(300, r_zone),
                            color="#f0a8a8", alpha=0.55, zorder=0)
        if r_zone < ax_vmax:
            ax.fill_between(ang_full, np.full(300, r_zone), np.full(300, ax_vmax),
                            color="#a8d5b5", alpha=0.45, zorder=0)
        ax.plot(ang_full, np.full(300, r_zone),
                color="#777777", linewidth=1.0, linestyle="-", zorder=1)

        # Concentric rings drawn manually with arc-gap (---79--- effect)
        n_rings = 5
        r_ticks = np.linspace(vmin, vmax, n_rings + 2)[1:-1].tolist()
        # Snap the ring closest to the zone boundary onto it exactly so the
        # label reads "0" (return) / "50" (precision).
        if abs(r_zone - vmin) > 1e-9 and abs(r_zone - vmax) > 1e-9:
            i_snap = int(np.argmin([abs(rt - r_zone) for rt in r_ticks]))
            r_ticks[i_snap] = r_zone
        orig_ticks = [_r_inv(r) for r in r_ticks]
        ax.set_yticks(r_ticks)
        ax.set_yticklabels([""] * len(r_ticks))

        ang_30m = angles[GRAN_ORDER.index("30m")]
        ang_1d  = angles[GRAN_ORDER.index("1d")]
        ang_gap = (ang_30m + ang_1d + 2 * np.pi) / 2.0
        gap_half = np.deg2rad(3.5)
        ang_arc = np.linspace(ang_gap + gap_half,
                              ang_gap + 2 * np.pi - gap_half, 300)
        r_span = r_ticks[-1] - r_ticks[0] if len(r_ticks) > 1 else 1.0
        seen_labels = set()
        for i, (rv_orig, rv_r) in enumerate(zip(orig_ticks, r_ticks)):
            ax.plot(ang_arc, np.full_like(ang_arc, rv_r),
                    color="#aaaaaa", linewidth=0.8, linestyle="-",
                    alpha=1.0, zorder=1)
            # Avoid "-0" rendering and duplicate "0" labels when multiple
            # ticks round to the same integer (e.g. −0.4 and the snapped 0).
            label_v = 0.0 if abs(rv_orig) < 0.5 else rv_orig
            label_str = f"{label_v:.0f}"
            if label_str in seen_labels:
                continue
            seen_labels.add(label_str)
            r_label = rv_r - 0.04 * r_span if i == len(r_ticks) - 1 else rv_r
            ax.annotate(label_str, xy=(ang_gap, r_label),
                        fontsize=15, color="#222222", fontweight="bold",
                        ha="center", va="center", zorder=9)

        # Polygons
        for m2 in m2_models:
            raw = [vals[m2][g] for g in GRAN_ORDER]
            data = [_r(float(np.clip(v if v is not None else 0.0, l_vmin, l_vmax)))
                    for v in raw]
            data_closed = data + data[:1]
            ax.plot(angles, data_closed, color=M2_COLORS[m2],
                    linestyle=M2_LS[m2], linewidth=2.4, zorder=3,
                    solid_capstyle="round")

        ax.set_title(title, fontsize=15, fontweight="bold",
                     pad=18, color="#111111")

    ax_ret  = fig.add_subplot(gs[0, 0], polar=True)
    ax_prec = fig.add_subplot(gs[0, 1], polar=True)
    _draw(ax_ret,  vals_ret,  "M2 Total Return (%)", kind="return")
    _draw(ax_prec, vals_prec, "M2 Precision (%)",    kind="precision")

    # Shared centred label sitting between the two individual subplot titles
    fig.canvas.draw()
    inv = fig.transFigure.inverted()
    # Use the top of the left axes as the y reference so the label sits
    # just below the subplot titles and is clearly between them.
    top_y = inv.transform(ax_ret.transAxes.transform([0.5, 1.0]))[1]
    FS = 15   # unified font size for all text in the figure

    fig.text(0.5, top_y + 0.065, rf"{M1}  $\downarrow$",
             ha="center", va="bottom", fontsize=15, fontweight="bold",
             color="#111111")

    # ── shared bottom legend — 2 rows: M2 models (top) / zone conditions (bottom) ──
    leg_ax = fig.add_subplot(gs[1, :]); leg_ax.set_axis_off()
    m2_handles = [Line2D([0], [0], color=M2_COLORS[m2], linewidth=2.6,
                         linestyle=M2_LS[m2], label=M2_LABELS[m2])
                  for m2 in m2_models]
    zone_handles = [
        Patch(facecolor="#a8d5b5", edgecolor="#aaaaaa", alpha=0.85,
              label=r"M2 Return $> 0$  /  Precision $\geq$ 50%"),
        Patch(facecolor="#f0a8a8", edgecolor="#aaaaaa", alpha=0.85,
              label=r"M2 Return $< 0$  /  Precision $<$ 50%"),
    ]
    # Row 1: M2 model lines; Row 2: zone patches — achieved with ncol=max(len)
    # so both rows are centred independently.
    handles = m2_handles + zone_handles

    fig.canvas.draw()
    inv = fig.transFigure.inverted()
    leg_bbox = inv.transform(leg_ax.transAxes.transform([0.5, 0.5]))
    y_center = leg_bbox[1]

    # Two separate legends stacked vertically so each row is exact:
    #   Row 1 (top):    Random Forest | TabPFN | CTTS
    #   Row 2 (bottom): M2 Return>0… | M2 Return<0…
    leg_kw = dict(bbox_transform=fig.transFigure, fontsize=FS,
                  frameon=False, handlelength=2.4, handleheight=1.4,
                  markerscale=1.4, borderpad=0.4, columnspacing=1.4)
    leg1 = fig.legend(handles=m2_handles, loc="center",
                      bbox_to_anchor=(0.5, y_center + 0.01),
                      ncol=len(m2_handles), **leg_kw)
    fig.add_artist(leg1)
    fig.legend(handles=zone_handles, loc="center",
               bbox_to_anchor=(0.5, y_center - 0.04),
               ncol=len(zone_handles), **leg_kw)

    out = os.path.join(output_dir, "results_kronos_down_combined.png")
    fig.savefig(out, dpi=200, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close(fig)
    print(f"[plot_kronos_down_combined] {out}")
    return out


def plot_scenario_matrices(data_root: str = "/home/pablo/M2_DS/Secondary-Model/src/Output",
                           output_dir: str = "/home/pablo/M2_DS/Secondary-Model/src/Output/Analysis/Results",
                           edge_root: str = "/home/pablo/M2_DS/Secondary-Model/src/Output/Analysis/Edge_NoCal"):
    """Generate two 2x2 matrices counting outcomes of precision and profitability scenarios."""
    import os
    import json
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from pathlib import Path
    try:
        from Utils.backtest.engine import _build_spread_equity
    except ImportError:
        _build_spread_equity = None

    M1_MODELS  = ["Tirex", "Chronos2", "Fincast", "Kronos"]
    M2_MODELS  = ["rf", "autogluon", "tabpfn", "tabicl", "ctts"]
    GRAN_ORDER = ["1d", "12h", "8h", "6h", "4h", "2h", "1h", "30m"]
    DIRECTIONS = ["UP", "DOWN"]

    M2_KEYS = {
        "rf":        "rf_backtest_all_features",
        "autogluon": "autogluon_backtest_all_features",
        "tabpfn":    "tabpfn_backtest_all_features",
        "tabicl":    "tabicl_backtest_all_features",
        "ctts":      "ctts_backtest_all_features",
    }
    M2_TEMP_KEYS = {
        "rf":        "rf_temporal_all_features",
        "autogluon": "autogluon_temporal_all_features",
        "tabpfn":    "tabpfn_temporal_all_features",
        "tabicl":    "tabicl_temporal_all_features",
        "ctts":      "ctts_temporal_all_features",
    }

    os.makedirs(output_dir, exist_ok=True)
    data_path = Path(data_root)
    edge_path = Path(edge_root)

    mat1 = np.zeros((2, 2), dtype=int)
    mat2 = np.zeros((2, 2), dtype=int)

    for m1 in M1_MODELS:
        for m2 in M2_MODELS:
            key_back = M2_KEYS[m2]
            key_temp = M2_TEMP_KEYS[m2]
            for direction in DIRECTIONS:
                for gran in GRAN_ORDER:
                    gran_dir = f"{gran}_tp"
                    p_base = data_path / m1 / m2 / direction / "Utility_Score_NoCal" / gran_dir
                    bp = p_base / "analysis_summary.json"
                    ep = edge_path / m1 / m2 / direction / f"edge_summary_{gran}.json"
                    trades_csv = p_base / "10_backtest_all_trades.csv"

                    if not bp.exists() or not trades_csv.exists():
                        continue
                    
                    try:
                        bt_ = json.load(open(bp))
                        b_ = bt_.get(key_back, {})
                        t_ = bt_.get(key_temp, {})
                        if not b_ or not t_:
                            continue
                        
                        test_perf = t_.get("Test", {})
                        test_sel_perf = t_.get("Test_selective", {})
                        
                        if not test_perf or not test_sel_perf:
                            continue

                        # Matrix 1: Precision
                        prec_05 = test_perf.get("precision", 0)
                        baseline_05 = test_perf.get("baseline", 0)
                        prec_sel = test_sel_perf.get("precision", 0)
                        
                        if prec_05 > baseline_05:
                            mat1[0, 0] += 1
                        else:
                            mat1[0, 1] += 1
                            
                        if prec_sel > baseline_05:
                            mat1[1, 0] += 1
                        else:
                            mat1[1, 1] += 1

                        # Reliability check
                        constr = bool(test_sel_perf.get("constraint_satisfied", False))
                        cv_val = 99.0
                        if ep.exists():
                            edge_data = json.load(open(ep)).get(gran, {})
                            p_rets = np.array(edge_data.get('path_total_rets', []), dtype=float)
                            if len(p_rets) > 1:
                                cv_val = float(np.std(p_rets) / (abs(np.mean(p_rets)) + 1e-6))
                        green = constr and (cv_val < 0.5)

                        # Matrix 2: Profitability
                        m1_ret = b_.get("m1_total_return", 0)
                        m2_ret_sel = b_.get("m2_total_return", 0)
                        
                        m2_ret_05 = m1_ret
                        if _build_spread_equity is not None:
                            df = pd.read_csv(trades_csv)
                            df['date'] = pd.to_datetime(df['date'])
                            df_05 = df[df['m2_prob'] >= 0.5].copy()
                            df_05['m2_approved'] = True
                            if len(df_05) > 0:
                                timeline = pd.DatetimeIndex(sorted(df["date"].unique()))
                                eq_05, _ = _build_spread_equity(df_05, timeline, 7)
                                m2_ret_05 = (eq_05.iloc[-1] - 1) * 100
                            else:
                                m2_ret_05 = 0.0

                        if m2_ret_05 > m1_ret:
                            mat2[0, 1] += 1
                            if green:
                                mat2[0, 0] += 1
                                
                        if m2_ret_sel > m1_ret:
                            mat2[1, 1] += 1
                            if green:
                                mat2[1, 0] += 1
                                
                    except Exception as e:
                        print(f"Error processing {bp}: {e}")
                        continue

    fig, axes = plt.subplots(1, 2, figsize=(15, 6), facecolor='white')
    
    def draw_matrix(ax, mat, row_labels, col_labels, title):
        cax = ax.matshow(mat, cmap='Blues', alpha=0.8)
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                val = mat[i, j]
                color = 'white' if val > np.max(mat)/2 else 'black'
                ax.text(j, i, str(val), va='center', ha='center',
                        fontsize=20, fontweight='bold', color=color)
        
        ax.set_xticks(range(len(col_labels)))
        ax.set_yticks(range(len(row_labels)))
        ax.set_xticklabels(col_labels, fontsize=12, fontweight='bold')
        ax.set_yticklabels(row_labels, fontsize=12, fontweight='bold', rotation=90, va='center')
        ax.xaxis.set_ticks_position('bottom')
        ax.set_title(title, pad=20, fontsize=15, fontweight='bold', color='#2c3e50')
        ax.tick_params(axis='both', which='both', length=0)

    draw_matrix(axes[0], mat1, 
                ["Baseline $\\tau=0.5$", "Optimized $\\hat{\\tau}$"], 
                ["Precision > $M_1$ Base", "Precision $\\leq$ $M_1$ Base"],
                "Scenario 1: Precision ($M_2$ vs $M_1$)")
                
    draw_matrix(axes[1], mat2, 
                ["Baseline $\\tau=0.5$", "Optimized $\\hat{\\tau}$"], 
                ["With Reliability\nAnalysis", "Without Reliability\nAnalysis"],
                "Scenario 2: Profitability ($M_2$ Return > $M_1$ Return)")
                
    plt.tight_layout()
    out_path = Path(output_dir) / "scenario_matrices.png"
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved scenario matrices to {out_path}")



# ════════════════════════════════════════════════════════════════════════════
# ┏━━━━━━━━━━ CPCV / Results matrices analysis ━━━━━━━━━━┓
# ════════════════════════════════════════════════════════════════════════════
# Three callables (also exposed via run_confusion_combined.py):
#   - plot_cpcv_filter_confusion(...)      bar chart of filter combinations
#   - compute_tab_vs_ctts_comparison(...)  tab vs CTTS counts -> JSON
#   - plot_results_matrices(...)           3×4 results matrices (this file)
# ════════════════════════════════════════════════════════════════════════════

_M1_LIST   = ["Tirex", "Chronos2", "Fincast", "Kronos"]
_M2_LIST   = ["rf", "autogluon", "tabpfn", "tabicl", "ctts"]
_DIRS_LIST = ["UP", "DOWN"]
_GRANS_LIST = ["1d", "12h", "8h", "6h", "4h", "2h", "1h", "30m"]


def _load_cpcv_records(edge_root: Path, bt_root: Path):
    """Load all (M1,M2,dir,gran) records with constraint_satisfied=True for CPCV analysis.

    Used by plot_cpcv_filter_confusion and compute_tab_vs_ctts_comparison.
    Returns list of dicts with fp, med_sr, mean_sr, cv, path_mean, val_mean_ret,
    val_tstat, val_constr, test_pos, val_pos, m1, m2, dir, gran.
    """
    records = []
    for m1 in _M1_LIST:
        for m2 in _M2_LIST:
            for d in _DIRS_LIST:
                for g in _GRANS_LIST:
                    ep = edge_root / m1 / m2 / d / f"edge_summary_{g}.json"
                    bp = bt_root / m1 / m2 / d / "Utility_Score_NoCal" / f"{g}_tp" / "analysis_summary.json"
                    try:
                        entry = json.load(open(ep)).get(g, {})
                        bt    = json.load(open(bp))
                        tkey  = f"{m2}_temporal_all_features"
                        bkey  = f"{m2}_backtest_all_features"
                        val_sel  = bt[tkey]["Val_selective"]
                        val_ret  = val_sel["mean_ret"]
                        test_ret = bt[bkey]["m2_total_return"]
                        if val_ret is None or test_ret is None: continue
                        if not val_sel.get("constraint_satisfied", False): continue
                        p   = np.array(entry.get("path_total_rets", []), dtype=float)
                        srs = np.array(entry.get("path_sharpes", []),    dtype=float)
                        cv        = float(np.std(p) / (abs(np.mean(p)) + 1e-6)) if len(p) > 1 else 99.0
                        mean_sr   = float(np.mean(srs)) if len(srs) > 0 else -99.0
                        path_mean = float(np.mean(p))   if len(p)   > 0 else -99.0
                        records.append({
                            "val_pos":      int(val_ret > 0),
                            "test_pos":     int(test_ret > 0),
                            "fp":           entry.get("frac_profitable", 0),
                            "med_sr":       entry.get("median_path_sharpe", -99),
                            "mean_sr":      mean_sr,
                            "pp_mean":      entry.get("path_sel_prec_mean", 0),
                            "pp_std":       entry.get("path_sel_prec_std", 99),
                            "cv":           cv,
                            "path_mean":    path_mean,
                            "val_mean_ret": val_ret,
                            "val_tstat":    val_sel.get("t_stat", 0),
                            "val_constr":   1,
                            "test_ret":     test_ret,
                            "test_sharpe":  bt[bkey].get("m2_sharpe", None),
                            "m1": m1, "m2": m2, "dir": d, "gran": g,
                        })
                    except Exception:
                        pass
    return records


def plot_cpcv_filter_confusion(edge_root: Path, bt_root: Path, save_path: Path):
    """Top-30 filter combinations bar chart for CPCV reliability analysis.

    Saves results_matrices_summary-style PNG showing TP/FP/FN/TN bars and
    precision/recall/accuracy lines for the top-30 filter combinations
    (out of singles + pairs + triples) ranked by test precision.
    """
    from itertools import combinations
    records = _load_cpcv_records(edge_root, bt_root)
    N = len(records)
    print(f"[plot_cpcv_filter_confusion] N={N}")

    base_conditions = [
        ("fp≥0.6",      lambda r: r["fp"]          >= 0.6),
        ("fp≥0.8",      lambda r: r["fp"]          >= 0.8),
        ("meanSR>0.5",  lambda r: r["mean_sr"]     > 0.5),
        ("meanSR>1.0",  lambda r: r["mean_sr"]     > 1.0),
        ("meanSR≥1.5",  lambda r: r["mean_sr"]     >= 1.5),
        ("medSR>0.5",   lambda r: r["med_sr"]      > 0.5),
        ("medSR>1.0",   lambda r: r["med_sr"]      > 1.0),
        ("medSR≥1.5",   lambda r: r["med_sr"]      >= 1.5),
        ("CV<1.0",      lambda r: r["cv"]          < 1.0),
        ("CV<0.5",      lambda r: r["cv"]          < 0.5),
        ("prec≥0.52",   lambda r: r["pp_mean"]     >= 0.52),
        ("pathMean>0",  lambda r: r["path_mean"]   > 0),
        ("valRet>0",    lambda r: r["val_mean_ret"] > 0),
        ("tStat>1.5",   lambda r: r["val_tstat"]   > 1.5),
        ("tStat>2",     lambda r: r["val_tstat"]   > 2),
        ("tStat>3",     lambda r: r["val_tstat"]   > 3),
        ("constr=True", lambda r: r["val_constr"]  == 1),
    ]
    redundant_groups = [
        {"fp≥0.6", "fp≥0.8"},
        {"meanSR>0.5", "meanSR>1.0", "meanSR≥1.5"},
        {"medSR>0.5", "medSR>1.0", "medSR≥1.5"},
        {"tStat>1.5", "tStat>2", "tStat>3"},
        {"CV<1.0", "CV<0.5"},
        {"constr=True", "valRet>0"},
        {"constr=True", "tStat>1.5"},
        {"constr=True", "tStat>2"},
    ]
    def _has_red(names):
        s = set(names)
        return any(len(g & s) > 1 for g in redundant_groups)

    def _combine(fns):
        return lambda r: all(fn(r) for fn in fns)

    all_filters = [("Baseline\n(no filter)", lambda r: True)]
    for name, fn in base_conditions:
        all_filters.append((name, fn))
    for (n1, f1), (n2, f2) in combinations(base_conditions, 2):
        if not _has_red([n1, n2]):
            all_filters.append((f"{n1} &\n{n2}", _combine([f1, f2])))
    for (n1, f1), (n2, f2), (n3, f3) in combinations(base_conditions, 3):
        if not _has_red([n1, n2, n3]):
            all_filters.append((f"{n1} &\n{n2} & {n3}", _combine([f1, f2, f3])))

    def _stats(filters, key):
        TPs, FPs, FNs, TNs = [], [], [], []
        for _, fn in filters:
            sel = [r for r in records if fn(r)]
            rej = [r for r in records if not fn(r)]
            TPs.append(sum(r[key] == 1 for r in sel))
            FPs.append(sum(r[key] == 0 for r in sel))
            FNs.append(sum(r[key] == 1 for r in rej))
            TNs.append(sum(r[key] == 0 for r in rej))
        return TPs, FPs, FNs, TNs

    TPs_t, FPs_t, FNs_t, TNs_t = _stats(all_filters, "test_pos")
    precs_all = [TP / (TP + FP) if (TP + FP) > 0 else -1 for TP, FP in zip(TPs_t, FPs_t)]
    ranked = [0] + sorted(range(1, len(all_filters)), key=lambda i: -precs_all[i])
    TOP_N = 30
    keep = ranked[: TOP_N + 1]
    filters = [all_filters[i] for i in keep]
    # Pin CV<0.5 & fp≥0.8 at the end
    filters.append(("CV<0.5 &\nfp≥0.8", lambda r: r["cv"] < 0.5 and r["fp"] >= 0.8))

    splits = [("VAL", "val_pos", "Val_selective mean_ret > 0"),
              ("TEST", "test_pos", "m2_total_return > 0")]

    fig, axes = plt.subplots(2, 1, figsize=(26, 16), dpi=160)
    fig.patch.set_facecolor("white")

    for ax, (split, out_key, split_label) in zip(axes, splits):
        n_pos = sum(r[out_key] for r in records)
        n_neg = N - n_pos
        bar_w = 0.18
        TPs, FPs, FNs, TNs = _stats(filters, out_key)
        x = np.arange(len(filters))

        TPs_r = np.array(TPs) / N; FPs_r = np.array(FPs) / N
        FNs_r = np.array(FNs) / N; TNs_r = np.array(TNs) / N

        b1 = ax.bar(x - 1.5*bar_w, TPs_r, bar_w, label="TP: selected & profitable",
                    color="#2ca02c", edgecolor="white")
        b2 = ax.bar(x - 0.5*bar_w, FPs_r, bar_w, label="FP: selected & NOT profitable",
                    color="#d62728", edgecolor="white")
        b3 = ax.bar(x + 0.5*bar_w, TNs_r, bar_w, label="TN: rejected & NOT profitable",
                    color="#1f77b4", edgecolor="white")
        b4 = ax.bar(x + 1.5*bar_w, FNs_r, bar_w, label="FN: rejected & profitable (missed)",
                    color="#ff7f0e", edgecolor="white")

        for bars, vals in [(b1, TPs), (b2, FPs), (b3, TNs), (b4, FNs)]:
            for bar, v in zip(bars, vals):
                if v > 0:
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
                            str(v), ha="center", va="bottom", fontsize=7,
                            fontweight="bold", color="#111111")

        ax2 = ax.twinx()
        precisions = [TP/(TP+FP) if (TP+FP) > 0 else np.nan for TP, FP in zip(TPs, FPs)]
        recalls    = [TP/(TP+FN) if (TP+FN) > 0 else np.nan for TP, FN in zip(TPs, FNs)]
        accs       = [(TP+TN)/N  if N > 0       else np.nan for TP, TN in zip(TPs, TNs)]
        ax2.plot(x, precisions, "D--", color="#9467bd", lw=1.8, ms=6,
                 label="Precision TP/(TP+FP)", zorder=5)
        ax2.plot(x, recalls,    "s--", color="#8c564b", lw=1.8, ms=6,
                 label="Recall TP/(TP+FN)",    zorder=5)
        ax2.plot(x, accs,       "^--", color="#17becf", lw=1.8, ms=6,
                 label="Accuracy (TP+TN)/N",   zorder=5)
        for xi, (p, r_, a) in enumerate(zip(precisions, recalls, accs)):
            if np.isfinite(p):
                ax2.text(xi - 0.22, p + 0.02, f"{p:.0%}", fontsize=6.5,
                         color="#9467bd", ha="center", fontweight="bold")
            if np.isfinite(r_):
                ax2.text(xi + 0.0, r_ - 0.05, f"{r_:.0%}", fontsize=6.5,
                         color="#8c564b", ha="center", fontweight="bold")
            if np.isfinite(a):
                ax2.text(xi + 0.22, a + 0.02, f"{a:.0%}", fontsize=6.5,
                         color="#17becf", ha="center", fontweight="bold")
        ax2.set_ylim(0, 1.15)
        ax2.set_ylabel("Precision / Recall / Accuracy", fontsize=10)
        ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
        ax2.spines["top"].set_visible(False)

        ax.set_xticks(x)
        ax.set_xticklabels([f[0] for f in filters], fontsize=8, rotation=35, ha="right")
        ax.set_ylabel(f"Fraction of all configs (N={N})", fontsize=10)
        ax.set_ylim(0, 0.75)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_facecolor("#fafafa")
        ax.grid(axis="y", color="#dddddd", lw=0.6, zorder=0)
        ax.axvspan(-0.5, 0.5, color="#ffffcc", alpha=0.5, zorder=0)

        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, loc="upper right", fontsize=8, framealpha=0.9, ncol=3)

        ax.text(0.01, 0.97,
                f"Actually profitable: {n_pos}/{N} ({n_pos/N:.1%})  |  "
                f"Actually not profitable: {n_neg}/{N} ({n_neg/N:.1%})",
                transform=ax.transAxes, fontsize=8.5, va="top",
                bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                          edgecolor="#aaaaaa", alpha=0.9))
        ax.set_title(f"► {split} split  —  Profitable = {split_label}",
                     fontsize=11, fontweight="bold", pad=8)

    fig.suptitle(
        f"CPCV Filter Confusion Analysis  |  Top {TOP_N} filters by TEST precision "
        f"(out of {len(all_filters)})  |  N={N} configs  |  Ranked left→right",
        fontsize=11, fontweight="bold", y=1.01)

    plt.tight_layout(h_pad=4.0)
    fig.savefig(str(save_path), dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[plot_cpcv_filter_confusion] Saved -> {save_path}")


def compute_tab_vs_ctts_comparison(bt_root: Path, edge_root: Path, save_path: Path):
    """Tab vs CTTS Δ Precision and reliability-aware return comparison → JSON."""
    def _load(m1, m2, d, g):
        bp = bt_root / m1 / m2 / d / "Utility_Score_NoCal" / f"{g}_tp" / "analysis_summary.json"
        ep = edge_root / m1 / m2 / d / f"edge_summary_{g}.json"
        try:
            bt_   = json.load(open(bp))
            entry = json.load(open(ep)).get(g, {})
            b = bt_[f"{m2}_backtest_all_features"]
            t = bt_[f"{m2}_temporal_all_features"]
            val_sel = t["Val_selective"]
            constr  = bool(val_sel.get("constraint_satisfied", False))
            p       = np.array(entry.get("path_total_rets", []), dtype=float)
            cv      = float(np.std(p) / (abs(np.mean(p)) + 1e-6)) if len(p) > 1 else 99.0
            return {"prec_delta": b["m2_win_rate"] - b["m1_win_rate"],
                    "m2_return":  b["m2_total_return"],
                    "green":      constr and cv < 1.0}
        except Exception:
            return None

    TAB = ["rf", "autogluon", "tabpfn", "tabicl"]
    t1_tab_wins = t1_ctts_wins = t1_tie = t1_total = 0
    t2_scA_tab = t2_scA_ctts = t2_scA_tie = 0
    t2_scB = t2_scC = t2_neither = 0

    for m1 in _M1_LIST:
        for d in _DIRS_LIST:
            for g in _GRANS_LIST:
                ctts = _load(m1, "ctts", d, g)
                tabs = {m2: _load(m1, m2, d, g) for m2 in TAB}
                tab_deltas = [v["prec_delta"] for v in tabs.values() if v is not None]
                ctts_delta = ctts["prec_delta"] if ctts else None
                if tab_deltas and ctts_delta is not None:
                    best_tab = max(tab_deltas)
                    t1_total += 1
                    if best_tab > ctts_delta:   t1_tab_wins  += 1
                    elif best_tab < ctts_delta: t1_ctts_wins += 1
                    else:                       t1_tie       += 1

                green_tabs = {m2: v for m2, v in tabs.items() if v is not None and v["green"]}
                ctts_green = ctts is not None and ctts["green"]
                if ctts_green and green_tabs:
                    best_tab_ret = max(v["m2_return"] for v in green_tabs.values())
                    ctts_ret     = ctts["m2_return"]
                    if   best_tab_ret > ctts_ret: t2_scA_tab  += 1
                    elif best_tab_ret < ctts_ret: t2_scA_ctts += 1
                    else:                         t2_scA_tie  += 1
                elif not ctts_green and green_tabs: t2_scB     += 1
                elif ctts_green and not green_tabs: t2_scC     += 1
                else:                               t2_neither += 1

    out = {
        "table1": {
            "total":        t1_total,
            "tab_wins":     t1_tab_wins,
            "ctts_wins":    t1_ctts_wins,
            "tie":          t1_tie,
            "tab_win_pct":  round(t1_tab_wins / t1_total, 4) if t1_total else 0,
            "ctts_win_pct": round(t1_ctts_wins / t1_total, 4) if t1_total else 0,
        },
        "table2": {
            "total":              len(_M1_LIST) * len(_DIRS_LIST) * len(_GRANS_LIST),
            "scA_tab_wins":       t2_scA_tab,
            "scA_ctts_wins":      t2_scA_ctts,
            "scA_tie":            t2_scA_tie,
            "scB_ctts_red_tab_green": t2_scB,
            "scC_ctts_green_no_tab":  t2_scC,
            "neither_green":      t2_neither,
            "total_tab_wins":     t2_scA_tab + t2_scB,
            "total_ctts_wins":    t2_scA_ctts + t2_scC,
        },
    }
    json.dump(out, open(save_path, "w"), indent=2)
    print(f"[compute_tab_vs_ctts_comparison] Saved -> {save_path}")
    print(f"  Table 1: tab={t1_tab_wins} ctts={t1_ctts_wins} (tie={t1_tie}, total={t1_total})")
    print(f"  Table 2: tab={out['table2']['total_tab_wins']} ctts={out['table2']['total_ctts_wins']}")
    return out


def _compute_metrics_at_threshold(csv_path: Path, threshold: float, fee: float = 0.002,
                                  horizon: int = 7):
    """Re-evaluate metrics at any threshold from 10_backtest_all_trades.csv.

    Reuses _build_spread_equity and _equity_horizon_returns/_calc_sharpe from backtest engine.
    Returns dict with: m2_total_return (%), m2_sharpe, avg_app (%), avg_rej (%), n_app, n_rej.
    The 'return' column in the CSV is already net-of-fee; threshold filters
    by m2_prob >= threshold.
    """
    from Utils.backtest.engine import _build_spread_equity, _equity_horizon_returns, _calc_sharpe, _annualization_factor  # type: ignore

    df = pd.read_csv(csv_path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.dropna(subset=["return"]).reset_index(drop=True)

    approved = df["m2_prob"].values >= threshold
    n_app = int(approved.sum())
    n_rej = int((~approved).sum())

    avg_app = float(df.loc[approved, "return"].mean()) * 100 if n_app > 0 else 0.0
    avg_rej = float(df.loc[~approved, "return"].mean()) * 100 if n_rej > 0 else 0.0

    # Profitable / losing among the approved trades (return is already net-of-fee)
    if n_app > 0:
        app_rets = df.loc[approved, "return"].values
        win_mask = app_rets > 0
        loss_mask = ~win_mask
        n_app_win  = int(win_mask.sum())
        n_app_loss = int(loss_mask.sum())
        avg_app_win  = float(app_rets[win_mask].mean())  * 100 if n_app_win  > 0 else 0.0
        avg_app_loss = float(app_rets[loss_mask].mean()) * 100 if n_app_loss > 0 else 0.0
    else:
        n_app_win = n_app_loss = 0
        avg_app_win = avg_app_loss = 0.0

    full_idx = pd.DatetimeIndex(sorted(df["date"].unique()))
    appr_df = df[approved].copy()
    m2_total_return = 0.0
    m2_sharpe = float("nan")
    if len(appr_df) > 0:
        equity, _ = _build_spread_equity(appr_df, full_idx, horizon)
        if len(equity) > 0:
            m2_total_return = float((equity.iloc[-1] - 1) * 100)
            # infer granularity from date spacing for annualisation
            if len(full_idx) > 1:
                delta_h = (full_idx[1] - full_idx[0]).total_seconds() / 3600
                if   delta_h >= 24*7: gran_str = "1w"
                elif delta_h >= 24:   gran_str = "1d"
                elif delta_h >= 12:   gran_str = "12h"
                elif delta_h >= 8:    gran_str = "8h"
                elif delta_h >= 6:    gran_str = "6h"
                elif delta_h >= 4:    gran_str = "4h"
                elif delta_h >= 2:    gran_str = "2h"
                elif delta_h >= 1:    gran_str = "1h"
                else:                 gran_str = "30m"
                ann = _annualization_factor(gran_str)
                ann_h = np.sqrt(ann ** 2 / horizon)
                h_rets = _equity_horizon_returns(equity, horizon)
                if len(h_rets) > 0:
                    m2_sharpe = float(_calc_sharpe(h_rets, ann_h))

    return {
        "m2_total_return": m2_total_return,
        "m2_sharpe":       m2_sharpe,
        "avg_app":         avg_app,
        "avg_rej":         avg_rej,
        "n_app":           n_app,
        "n_rej":           n_rej,
        "n_app_win":       n_app_win,
        "n_app_loss":      n_app_loss,
        "avg_app_win":     avg_app_win,
        "avg_app_loss":    avg_app_loss,
    }


def _walk_results_configs(bt_root: Path, edge_root: Path):
    """Yield per-config dicts with all metrics needed by plot_results_matrices.

    Per (M1, M2, dir, gran) config emits:
      - M1 baseline precision and total return
      - M2 precision at τ=0.5 (Test.precision) and τ̂ (Test_selective.precision)
      - M2 total return at τ=0.5 (recomputed) and τ̂ (from JSON)
      - Avg approved/rejected return at τ=0.5 (recomputed) and τ̂ (from ROI.txt)
      - constraint_satisfied flag
      - CV (from edge_summary path_total_rets)
    Skips configs without all required files.
    """
    for m1 in _M1_LIST:
        for m2 in _M2_LIST:
            tkey = f"{m2}_temporal_all_features"
            bkey = f"{m2}_backtest_all_features"
            for d in _DIRS_LIST:
                for g in _GRANS_LIST:
                    bp  = bt_root / m1 / m2 / d / "Utility_Score_NoCal" / f"{g}_tp" / "analysis_summary.json"
                    csv = bt_root / m1 / m2 / d / "Utility_Score_NoCal" / f"{g}_tp" / "10_backtest_all_trades.csv"
                    roi = bt_root / m1 / m2 / d / "Utility_Score_NoCal" / f"{g}_tp" / "10_backtest_all_ROI.txt"
                    ep  = edge_root / m1 / m2 / d / f"edge_summary_{g}.json"
                    if not (bp.exists() and csv.exists()):
                        continue
                    try:
                        bt_data = json.load(open(bp))
                        b = bt_data.get(bkey, {})
                        t = bt_data.get(tkey, {})
                        if not b or not t:
                            continue
                        val_sel = t.get("Val_selective", {})
                        constr  = bool(val_sel.get("constraint_satisfied", False))
                        m1_prec = float(t.get("Test", {}).get("baseline", float("nan")))
                        m2_prec_05  = float(t.get("Test", {}).get("precision", float("nan")))
                        m2_prec_tau = float(t.get("Test_selective", {}).get("precision", float("nan")))
                        m1_total_ret = float(b.get("m1_total_return", float("nan")))
                        m2_total_ret_tau = float(b.get("m2_total_return", float("nan")))
                        m2_sharpe_tau    = float(b.get("m2_sharpe", float("nan")))
                        fee = float(b.get("fee", 0.002))

                        # CV from CPCV
                        cv = 99.0
                        if ep.exists():
                            try:
                                entry = json.load(open(ep)).get(g, {})
                                p = np.array(entry.get("path_total_rets", []), dtype=float)
                                if len(p) > 1:
                                    cv = float(np.std(p) / (abs(np.mean(p)) + 1e-6))
                            except Exception:
                                pass

                        # Recompute τ=0.5 metrics from CSV
                        m05 = _compute_metrics_at_threshold(csv, 0.5, fee=fee, horizon=7)

                        # τ̂ approved/rejected averages: parse from ROI.txt if present
                        avg_app_tau = avg_rej_tau = None
                        if roi.exists():
                            try:
                                roi_txt = roi.read_text()
                                import re as _re
                                m_app = _re.search(r"Avg Return APPROVED:\s+([+-]?[\d.]+)%", roi_txt)
                                m_rej = _re.search(r"Avg Return REJECTED:\s+([+-]?[\d.]+)%", roi_txt)
                                if m_app: avg_app_tau = float(m_app.group(1))
                                if m_rej: avg_rej_tau = float(m_rej.group(1))
                            except Exception:
                                pass
                        # Fallback: recompute at τ̂ if ROI parse failed
                        if avg_app_tau is None or avg_rej_tau is None:
                            try:
                                tau_hat = float(b.get("threshold", 0.5))
                                m_tau = _compute_metrics_at_threshold(csv, tau_hat, fee=fee, horizon=7)
                                avg_app_tau = m_tau["avg_app"]
                                avg_rej_tau = m_tau["avg_rej"]
                            except Exception:
                                continue

                        # Val metrics
                        val_m1_prec      = float(t.get("Val", {}).get("baseline", float("nan")))
                        val_m2_prec_05   = float(t.get("Val", {}).get("precision", float("nan")))
                        val_m2_prec_tau  = float(t.get("Val_selective", {}).get("precision", float("nan")))
                        val_mean_ret_tau = float(t.get("Val_selective", {}).get("mean_ret", float("nan")))

                        yield {
                            "m1": m1, "m2": m2, "dir": d, "gran": g,
                            "m1_prec":           m1_prec,
                            "m2_prec_05":        m2_prec_05,
                            "m2_prec_tau":       m2_prec_tau,
                            "m1_total_ret":      m1_total_ret,
                            "m2_total_ret_05":   m05["m2_total_return"],
                            "m2_sharpe_05":      m05["m2_sharpe"],
                            "m2_sharpe_tau":     m2_sharpe_tau,
                            "m2_total_ret_tau":  m2_total_ret_tau,
                            "avg_app_05":        m05["avg_app"],
                            "avg_rej_05":        m05["avg_rej"],
                            "avg_app_tau":       avg_app_tau,
                            "avg_rej_tau":       avg_rej_tau,
                            "constr":            constr,
                            "cv":                cv,
                            "reliable":          constr and (cv < 0.5),
                            # validation split
                            "val_m1_prec":       val_m1_prec,
                            "val_m2_prec_05":    val_m2_prec_05,
                            "val_m2_prec_tau":   val_m2_prec_tau,
                            "val_mean_ret_tau":  val_mean_ret_tau,   # Val_selective mean_ret at τ̂
                        }
                    except Exception as e:
                        print(f"[walk] skip {m1}/{m2}/{d}/{g}: {e}")
                        continue


def build_metrics_dict(bt_root: Path, save_path: Path) -> dict:
    """Build and save a structured metrics dict with the schema:

        metrics[m1][m2][direction][granularity] = {
            "threshold_found":                  bool,
            "precision_without_threshold":      float,   # Test.precision at τ=0.5
            "precision_with_threshold":         float,   # Test_selective.precision at τ̂
            "mean_return_without_threshold":    float,   # M2 total return recomputed at τ=0.5
            "mean_return_with_threshold":       float,   # M2 total return at τ̂ (stored)
            "sharpe_ratio_without_threshold":   float,   # Sharpe recomputed at τ=0.5
            "sharpe_ratio_with_threshold":      float,   # Sharpe at τ̂ (stored)
            "n_trades_without_threshold":       int,     # trades approved at τ=0.5
            "n_trades_with_threshold":          int,     # trades approved at τ̂
            "n_total_suggested_trades_by_m1":   int,     # all M1 trades in test
        }
    """
    metrics = {m1: {m2: {d: {} for d in _DIRS_LIST} for m2 in _M2_LIST} for m1 in _M1_LIST}

    for m1 in _M1_LIST:
        for m2 in _M2_LIST:
            bkey = f"{m2}_backtest_all_features"
            tkey = f"{m2}_temporal_all_features"
            for d in _DIRS_LIST:
                for g in _GRANS_LIST:
                    bp  = bt_root / m1 / m2 / d / "Utility_Score_NoCal" / f"{g}_tp" / "analysis_summary.json"
                    csv = bt_root / m1 / m2 / d / "Utility_Score_NoCal" / f"{g}_tp" / "10_backtest_all_trades.csv"
                    entry = {}
                    try:
                        bt_data = json.load(open(bp))
                        b = bt_data.get(bkey, {})
                        t = bt_data.get(tkey, {})
                        fee = float(b.get("fee", 0.002))

                        # τ̂ fields (stored)
                        entry["threshold_found"]               = bool(b.get("constraint_satisfied", False))
                        entry["precision_without_threshold"]   = float(t.get("Test", {}).get("precision", float("nan")))
                        entry["precision_with_threshold"]      = float(t.get("Test_selective", {}).get("precision", float("nan")))
                        entry["mean_return_with_threshold"]    = float(b.get("m2_total_return", float("nan")))
                        entry["sharpe_ratio_with_threshold"]   = float(b.get("m2_sharpe", float("nan")))
                        entry["n_trades_with_threshold"]       = int(b.get("n_m2_trades", 0))
                        entry["n_total_suggested_trades_by_m1"] = int(b.get("n_total_trades", 0))

                        # τ=0.5 fields (recomputed from CSV)
                        if csv.exists():
                            m05 = _compute_metrics_at_threshold(csv, 0.5, fee=fee, horizon=7)
                            entry["mean_return_without_threshold"]  = m05["m2_total_return"]
                            entry["sharpe_ratio_without_threshold"] = m05["m2_sharpe"]
                            entry["n_trades_without_threshold"]     = m05["n_app"]
                        else:
                            entry["mean_return_without_threshold"]  = float("nan")
                            entry["sharpe_ratio_without_threshold"] = float("nan")
                            entry["n_trades_without_threshold"]     = 0

                    except Exception as e:
                        print(f"[build_metrics_dict] skip {m1}/{m2}/{d}/{g}: {e}")
                        entry = {
                            "threshold_found": None,
                            "precision_without_threshold": float("nan"),
                            "precision_with_threshold": float("nan"),
                            "mean_return_without_threshold": float("nan"),
                            "mean_return_with_threshold": float("nan"),
                            "sharpe_ratio_without_threshold": float("nan"),
                            "sharpe_ratio_with_threshold": float("nan"),
                            "n_trades_without_threshold": 0,
                            "n_trades_with_threshold": 0,
                            "n_total_suggested_trades_by_m1": 0,
                        }
                    metrics[m1][m2][d][g] = entry

    import pickle as _pickle
    with open(save_path, "wb") as f:
        _pickle.dump(metrics, f)
    print(f"[build_metrics_dict] Saved -> {save_path}")
    return metrics


def build_reliability_metrics_dict(bt_root: Path, edge_root: Path, save_path: Path) -> dict:
    """Build and save a reliability-filtered metrics dict with schema:

        metrics[m1][m2][direction][granularity] = {
            "threshold_found":                  bool,   # True if CV<0.5 AND constraint_satisfied
            "precision_without_threshold":      float,  # Test.precision at τ=0.5
            "mean_return_without_threshold":    float,  # M2 total return at τ=0.5 (recomputed)
            "sharpe_ratio_without_threshold":   float,  # Sharpe at τ=0.5 (recomputed)
            "n_trades_without_threshold":       int,    # trades approved at τ=0.5
            "n_total_suggested_trades_by_m1":   int,    # all M1 trades in test
        }
    Only _without_threshold fields are filled (τ=0.5). The reliability flag replaces
    the threshold optimisation flag.
    """
    metrics = {m1: {m2: {d: {} for d in _DIRS_LIST} for m2 in _M2_LIST} for m1 in _M1_LIST}

    for m1 in _M1_LIST:
        for m2 in _M2_LIST:
            bkey = f"{m2}_backtest_all_features"
            tkey = f"{m2}_temporal_all_features"
            for d in _DIRS_LIST:
                for g in _GRANS_LIST:
                    bp  = bt_root / m1 / m2 / d / "Utility_Score_NoCal" / f"{g}_tp" / "analysis_summary.json"
                    csv = bt_root / m1 / m2 / d / "Utility_Score_NoCal" / f"{g}_tp" / "10_backtest_all_trades.csv"
                    ep  = edge_root / m1 / m2 / d / f"edge_summary_{g}.json"
                    entry = {}
                    try:
                        bt_data = json.load(open(bp))
                        b = bt_data.get(bkey, {})
                        t = bt_data.get(tkey, {})
                        fee = float(b.get("fee", 0.002))

                        # Reliability flag: CV < 0.5 AND constraint_satisfied
                        constr = bool(bt_data.get(tkey, {}).get("Val_selective", {}).get("constraint_satisfied", False))
                        cv = 99.0
                        if ep.exists():
                            edge_entry = json.load(open(ep)).get(g, {})
                            p = np.array(edge_entry.get("path_total_rets", []), dtype=float)
                            if len(p) > 1:
                                cv = float(np.std(p) / (abs(np.mean(p)) + 1e-6))
                        reliable = constr and cv < 0.5

                        entry["threshold_found"]                 = reliable
                        entry["precision_without_threshold"]     = float(t.get("Test", {}).get("precision", float("nan")))
                        entry["n_total_suggested_trades_by_m1"]  = int(b.get("n_total_trades", 0))

                        if csv.exists():
                            m05 = _compute_metrics_at_threshold(csv, 0.5, fee=fee, horizon=7)
                            entry["mean_return_without_threshold"]  = m05["m2_total_return"]
                            entry["sharpe_ratio_without_threshold"] = m05["m2_sharpe"]
                            entry["n_trades_without_threshold"]     = m05["n_app"]
                        else:
                            entry["mean_return_without_threshold"]  = float("nan")
                            entry["sharpe_ratio_without_threshold"] = float("nan")
                            entry["n_trades_without_threshold"]     = 0

                    except Exception as e:
                        print(f"[build_reliability_metrics_dict] skip {m1}/{m2}/{d}/{g}: {e}")
                        entry = {
                            "threshold_found": None,
                            "precision_without_threshold": float("nan"),
                            "mean_return_without_threshold": float("nan"),
                            "sharpe_ratio_without_threshold": float("nan"),
                            "n_trades_without_threshold": 0,
                            "n_total_suggested_trades_by_m1": 0,
                        }
                    metrics[m1][m2][d][g] = entry

    import pickle as _pickle
    with open(save_path, "wb") as f:
        _pickle.dump(metrics, f)
    print(f"[build_reliability_metrics_dict] Saved -> {save_path}")
    return metrics


def build_combined_metrics_dict(bt_root: Path, edge_root: Path, save_path: Path) -> dict:
    """Single unified dict for all 320 configs with separate reliability and threshold flags.

    Source convention
    -----------------
    Each key carries a prefix indicating which model it describes:
      • ``m1_*``  → metric of the upstream forecaster M1 (no reliability filter).
                    M1 emits a directional signal for every test bar; these
                    metrics are computed on the full set of M1 trades.
      • ``m2_*``  → metric of the downstream selective classifier M2 evaluated
                    at a specific decision threshold τ (suffix ``_tau05`` for
                    τ=0.5, ``_tauhat`` for the optimisation-derived τ̂). M2
                    selects a subset of M1 trades to actually execute.
      • no prefix → meta-flags / diagnostics about the M2 calibration itself
                    (constraint, reliability, CV).

    metrics[m1][m2][direction][granularity] = {
        # ── Calibration meta-flags (M2 fit diagnostics) ───────────────────
        "constraint_satisfied":  bool,   # τ̂ optimiser found a valid threshold
                                         # (Stage-A constraints satisfied on Val)
        "reliable":              bool,   # M2 CPCV stability flag: CV < 0.5
                                         # (independent of threshold convergence)
        "cv":                    float,  # M2 CPCV coefficient of variation =
                                         # std(path_total_rets) / |mean(path_total_rets)|

        # ── M1 baseline (unfiltered, all trades) ──────────────────────────
        "m1_n_trades":           int,    # # of M1 directional trades in test split
        "m1_precision":          float,  # M1 baseline precision = win-rate of M1 trades (∈ [0,1])
        "m1_total_return":       float,  # M1 equity-curve total return (%) over test
        "m1_sharpe":             float,  # M1 Sharpe ratio over test

        # ── M2 @ τ=0.5 (default decision threshold, no calibration) ───────
        # Always filled when the trades CSV exists.
        "m2_precision_tau05":         float,  # M2 selective precision at τ=0.5
        "m2_total_return_tau05":      float,  # M2 equity-curve total return (%), τ=0.5
        "m2_sharpe_tau05":            float,  # M2 Sharpe at τ=0.5 (via engine._calc_sharpe)
        "m2_n_trades_tau05":          int,    # # of M2-approved trades at τ=0.5
        "m2_n_profitable_trades_tau05":    int,    # of approved trades, # with net return > 0
        "m2_n_losing_trades_tau05":        int,    # of approved trades, # with net return ≤ 0
        "m2_mean_return_profitable_tau05": float,  # mean net return (%) over profitable approved trades
        "m2_mean_return_losing_tau05":     float,  # mean net return (%) over losing approved trades

        # ── M2 @ τ̂ (calibrated threshold from Stage-A optimiser) ──────────
        # Filled only when constraint_satisfied=True; NaN/0 otherwise.
        "m2_precision_tauhat":          float, # M2 selective precision at τ̂
        "m2_total_return_tauhat":       float, # M2 equity-curve total return (%), τ̂
        "m2_sharpe_tauhat":             float, # M2 Sharpe at τ̂
        "m2_n_trades_tauhat":           int,   # # of M2-approved trades at τ̂
        "m2_n_profitable_trades_tauhat":    int,    # of approved trades, # with net return > 0
        "m2_n_losing_trades_tauhat":        int,    # of approved trades, # with net return ≤ 0
        "m2_mean_return_profitable_tauhat": float,  # mean net return (%) over profitable approved trades
        "m2_mean_return_losing_tauhat":     float,  # mean net return (%) over losing approved trades
    }
    All 320 configs present. ``m2_*_tauhat`` fields are NaN when
    constraint_satisfied=False.
    """
    metrics = {m1: {m2: {d: {} for d in _DIRS_LIST} for m2 in _M2_LIST} for m1 in _M1_LIST}
    nan = float("nan")

    total = included = 0
    for m1 in _M1_LIST:
        for m2 in _M2_LIST:
            bkey = f"{m2}_backtest_all_features"
            tkey = f"{m2}_temporal_all_features"
            for d in _DIRS_LIST:
                for g in _GRANS_LIST:
                    total += 1
                    bp  = bt_root / m1 / m2 / d / "Utility_Score_NoCal" / f"{g}_tp" / "analysis_summary.json"
                    csv = bt_root / m1 / m2 / d / "Utility_Score_NoCal" / f"{g}_tp" / "10_backtest_all_trades.csv"
                    ep  = edge_root / m1 / m2 / d / f"edge_summary_{g}.json"
                    entry = {
                        # M2 calibration diagnostics
                        "constraint_satisfied":     False,
                        "reliable":                 False,
                        "cv":                       nan,
                        # M1 baseline
                        "m1_n_trades":              0,
                        "m1_precision":             nan,
                        "m1_total_return":          nan,
                        "m1_sharpe":                nan,
                        # M2 @ τ=0.5
                        "m2_precision_tau05":            nan,
                        "m2_total_return_tau05":         nan,
                        "m2_sharpe_tau05":               nan,
                        "m2_n_trades_tau05":             0,
                        "m2_n_profitable_trades_tau05":    0,
                        "m2_n_losing_trades_tau05":        0,
                        "m2_mean_return_profitable_tau05": nan,
                        "m2_mean_return_losing_tau05":     nan,
                        # M2 @ τ̂
                        "m2_precision_tauhat":           nan,
                        "m2_total_return_tauhat":        nan,
                        "m2_sharpe_tauhat":              nan,
                        "m2_n_trades_tauhat":            0,
                        "m2_n_profitable_trades_tauhat":    0,
                        "m2_n_losing_trades_tauhat":        0,
                        "m2_mean_return_profitable_tauhat": nan,
                        "m2_mean_return_losing_tauhat":     nan,
                    }
                    try:
                        bt_data = json.load(open(bp))
                        b = bt_data.get(bkey, {})
                        t = bt_data.get(tkey, {})
                        fee = float(b.get("fee", 0.002))

                        # ── Flags ──────────────────────────────────────────
                        constr = bool(t.get("Val_selective", {}).get("constraint_satisfied", False))
                        cv = 99.0
                        if ep.exists():
                            p = np.array(json.load(open(ep)).get(g, {}).get("path_total_rets", []), dtype=float)
                            if len(p) > 1:
                                cv = float(np.std(p) / (abs(np.mean(p)) + 1e-6))

                        entry["constraint_satisfied"] = constr
                        entry["reliable"]             = cv < 0.5   # independent of constraint
                        entry["cv"]                   = round(cv, 6)
                        entry["m1_n_trades"]     = int(b.get("n_total_trades", 0))
                        entry["m1_total_return"] = float(b.get("m1_total_return", nan))
                        entry["m1_sharpe"]       = float(b.get("m1_sharpe", nan))
                        # m1_win_rate is stored in % (e.g. 45.35) — convert to fraction
                        _wr = b.get("m1_win_rate", nan)
                        try:
                            entry["m1_precision"] = float(_wr) / 100.0 if _wr == _wr else nan
                        except Exception:
                            entry["m1_precision"] = nan

                        # ── M2 @ τ=0.5 ─────────────────────────────────────
                        entry["m2_precision_tau05"] = float(t.get("Test", {}).get("precision", nan))
                        if csv.exists():
                            m05 = _compute_metrics_at_threshold(csv, 0.5, fee=fee, horizon=7)
                            entry["m2_total_return_tau05"]        = m05["m2_total_return"]
                            entry["m2_sharpe_tau05"]              = m05["m2_sharpe"]
                            entry["m2_n_trades_tau05"]            = m05["n_app"]
                            entry["m2_n_profitable_trades_tau05"]    = m05["n_app_win"]
                            entry["m2_n_losing_trades_tau05"]        = m05["n_app_loss"]
                            entry["m2_mean_return_profitable_tau05"] = m05["avg_app_win"]
                            entry["m2_mean_return_losing_tau05"]     = m05["avg_app_loss"]

                        # ── M2 @ τ̂ (only when constraint_satisfied) ───────
                        if constr:
                            entry["m2_precision_tauhat"]    = float(t.get("Test_selective", {}).get("precision", nan))
                            entry["m2_total_return_tauhat"] = float(b.get("m2_total_return", nan))
                            entry["m2_sharpe_tauhat"]       = float(b.get("m2_sharpe", nan))
                            entry["m2_n_trades_tauhat"]     = int(b.get("n_m2_trades", 0))
                            if csv.exists():
                                tau_hat = float(b.get("threshold", 0.5))
                                m_th = _compute_metrics_at_threshold(csv, tau_hat, fee=fee, horizon=7)
                                entry["m2_n_profitable_trades_tauhat"]    = m_th["n_app_win"]
                                entry["m2_n_losing_trades_tauhat"]        = m_th["n_app_loss"]
                                entry["m2_mean_return_profitable_tauhat"] = m_th["avg_app_win"]
                                entry["m2_mean_return_losing_tauhat"]     = m_th["avg_app_loss"]

                        included += 1
                    except Exception as e:
                        print(f"[build_combined_metrics_dict] skip {m1}/{m2}/{d}/{g}: {e}")

                    metrics[m1][m2][d][g] = entry

    import pickle as _pickle
    with open(save_path, "wb") as f:
        _pickle.dump(metrics, f)

    constr_n  = sum(metrics[m1][m2][d][g]["constraint_satisfied"]
                    for m1 in metrics for m2 in metrics[m1]
                    for d in metrics[m1][m2] for g in metrics[m1][m2][d])
    reliable_n = sum(metrics[m1][m2][d][g]["reliable"]
                     for m1 in metrics for m2 in metrics[m1]
                     for d in metrics[m1][m2] for g in metrics[m1][m2][d])
    both_n = sum(1 for m1 in metrics for m2 in metrics[m1]
                 for d in metrics[m1][m2] for g in metrics[m1][m2][d]
                 if metrics[m1][m2][d][g]["constraint_satisfied"] and metrics[m1][m2][d][g]["reliable"])
    print(f"[build_combined_metrics_dict] Saved -> {save_path}")
    print(f"  Total={total}  loaded={included}  constraint_satisfied={constr_n}  reliable(CV<0.5)={reliable_n}  both={both_n}")
    return metrics


def build_threshold_and_reliability_dict(bt_root: Path, edge_root: Path, save_path: Path) -> dict:
    """Build and save a dict restricted to configs that BOTH have a valid τ̂ AND pass reliability.

    Includes only configs where constraint_satisfied=True AND CV<0.5.
    All fields (with and without threshold) are filled.

        metrics[m1][m2][direction][granularity] = {
            "threshold_found":                  bool,   # always True (filter guarantees it)
            "reliable":                         bool,   # always True (filter guarantees it)
            "precision_without_threshold":      float,
            "precision_with_threshold":         float,
            "mean_return_without_threshold":    float,
            "mean_return_with_threshold":       float,
            "sharpe_ratio_without_threshold":   float,
            "sharpe_ratio_with_threshold":      float,
            "n_trades_without_threshold":       int,
            "n_trades_with_threshold":          int,
            "n_total_suggested_trades_by_m1":   int,
        }
    Configs not passing both conditions are omitted entirely.
    """
    metrics = {m1: {m2: {d: {} for d in _DIRS_LIST} for m2 in _M2_LIST} for m1 in _M1_LIST}

    included = 0
    for m1 in _M1_LIST:
        for m2 in _M2_LIST:
            bkey = f"{m2}_backtest_all_features"
            tkey = f"{m2}_temporal_all_features"
            for d in _DIRS_LIST:
                for g in _GRANS_LIST:
                    bp  = bt_root / m1 / m2 / d / "Utility_Score_NoCal" / f"{g}_tp" / "analysis_summary.json"
                    csv = bt_root / m1 / m2 / d / "Utility_Score_NoCal" / f"{g}_tp" / "10_backtest_all_trades.csv"
                    ep  = edge_root / m1 / m2 / d / f"edge_summary_{g}.json"
                    try:
                        bt_data = json.load(open(bp))
                        b = bt_data.get(bkey, {})
                        t = bt_data.get(tkey, {})
                        fee = float(b.get("fee", 0.002))

                        constr = bool(t.get("Val_selective", {}).get("constraint_satisfied", False))
                        cv = 99.0
                        if ep.exists():
                            p = np.array(json.load(open(ep)).get(g, {}).get("path_total_rets", []), dtype=float)
                            if len(p) > 1:
                                cv = float(np.std(p) / (abs(np.mean(p)) + 1e-6))

                        if not (constr and cv < 0.5):
                            continue   # skip — does not pass both conditions

                        entry = {
                            "threshold_found":               True,
                            "reliable":                      True,
                            "precision_without_threshold":   float(t.get("Test", {}).get("precision", float("nan"))),
                            "precision_with_threshold":      float(t.get("Test_selective", {}).get("precision", float("nan"))),
                            "mean_return_with_threshold":    float(b.get("m2_total_return", float("nan"))),
                            "sharpe_ratio_with_threshold":   float(b.get("m2_sharpe", float("nan"))),
                            "n_trades_with_threshold":       int(b.get("n_m2_trades", 0)),
                            "n_total_suggested_trades_by_m1": int(b.get("n_total_trades", 0)),
                        }
                        if csv.exists():
                            m05 = _compute_metrics_at_threshold(csv, 0.5, fee=fee, horizon=7)
                            entry["mean_return_without_threshold"]  = m05["m2_total_return"]
                            entry["sharpe_ratio_without_threshold"] = m05["m2_sharpe"]
                            entry["n_trades_without_threshold"]     = m05["n_app"]
                        else:
                            entry["mean_return_without_threshold"]  = float("nan")
                            entry["sharpe_ratio_without_threshold"] = float("nan")
                            entry["n_trades_without_threshold"]     = 0

                        metrics[m1][m2][d][g] = entry
                        included += 1

                    except Exception as e:
                        print(f"[build_threshold_and_reliability_dict] skip {m1}/{m2}/{d}/{g}: {e}")

    import pickle as _pickle
    with open(save_path, "wb") as f:
        _pickle.dump(metrics, f)
    print(f"[build_threshold_and_reliability_dict] Saved -> {save_path}  ({included} configs included)")
    return metrics


def plot_results_matrices(bt_root: Path, edge_root: Path, save_path: Path):
    """Generate a 1×3 panel of summary matrices (precision, profitability, ROI).

    Each matrix is 2×2 with axes:
      rows: τ=0.5  vs  τ̂ (optimized selective threshold)
      cols: without reliability filter  vs  with reliability filter (CV<0.5 & constraint)

    Matrix 1 (Precision):    cell = % configs where M2 precision > M1 baseline
    Matrix 2 (Profitability): cell = % configs where M2 total return > M1 total return
    Matrix 3 (Approved vs Rejected): cell = (avg_app, avg_rej, % configs where avg_app > avg_rej)
    """
    print(f"[plot_results_matrices] Walking configs...")
    configs = list(_walk_results_configs(bt_root, edge_root))
    print(f"[plot_results_matrices] Loaded {len(configs)} configs")
    if not configs:
        print("[plot_results_matrices] No configs — abort")
        return

    # ── Matrix 1: Precision ──────────────────────────────────────────
    def _mat1_cell(thr_key, reliab):
        pop = [c for c in configs if (c["reliable"] if reliab else True)]
        if thr_key == "tau":
            pop = [c for c in pop if c["constr"]]  # τ̂ only meaningful when constraint satisfied
        if not pop: return (0, 0, 0.0)
        prec_key = "m2_prec_05" if thr_key == "05" else "m2_prec_tau"
        wins = sum(1 for c in pop
                   if not np.isnan(c[prec_key]) and not np.isnan(c["m1_prec"])
                   and c[prec_key] > c["m1_prec"])
        return (wins, len(pop), wins / len(pop))

    # ── Matrix 2: Profitability ──────────────────────────────────────
    def _mat2_cell(thr_key, reliab):
        pop = [c for c in configs if (c["reliable"] if reliab else True)]
        if thr_key == "tau":
            pop = [c for c in pop if c["constr"]]
        if not pop: return (0, 0, 0.0)
        ret_key = "m2_total_ret_05" if thr_key == "05" else "m2_total_ret_tau"
        wins = sum(1 for c in pop
                   if not np.isnan(c[ret_key]) and not np.isnan(c["m1_total_ret"])
                   and c[ret_key] > c["m1_total_ret"])
        return (wins, len(pop), wins / len(pop))

    # ── Matrix 3: Approved vs Rejected ───────────────────────────────
    def _mat3_cell(thr_key, reliab):
        pop = [c for c in configs if (c["reliable"] if reliab else True)]
        if thr_key == "tau":
            pop = [c for c in pop if c["constr"]]
        if not pop: return (np.nan, np.nan, 0, 0, 0.0)
        app_key = "avg_app_05" if thr_key == "05" else "avg_app_tau"
        rej_key = "avg_rej_05" if thr_key == "05" else "avg_rej_tau"
        apps = [c[app_key] for c in pop if c[app_key] is not None and not (isinstance(c[app_key], float) and np.isnan(c[app_key]))]
        rejs = [c[rej_key] for c in pop if c[rej_key] is not None and not (isinstance(c[rej_key], float) and np.isnan(c[rej_key]))]
        avg_app = float(np.mean(apps)) if apps else np.nan
        avg_rej = float(np.mean(rejs)) if rejs else np.nan
        wins = sum(1 for c in pop
                   if c[app_key] is not None and c[rej_key] is not None
                   and not (isinstance(c[app_key], float) and np.isnan(c[app_key]))
                   and not (isinstance(c[rej_key], float) and np.isnan(c[rej_key]))
                   and c[app_key] > c[rej_key])
        return (avg_app, avg_rej, wins, len(pop), wins / len(pop))

    row_labels = [r"$\tau = 0.5$", r"$\hat{\tau}$ (optimized)"]
    col_labels = ["No reliability\nfilter", "Reliability filter\n(CV < 0.5 & constr.)"]

    fig = plt.figure(figsize=(30, 13), dpi=160)
    fig.patch.set_facecolor("white")
    # 2 rows × 4 cols; col 2 (Matrix 3) given extra width via width_ratios
    # row 0: Matrix 1(Test) | Matrix 2(Test) | Matrix 3(Test) | Matrix 4(Test)
    # row 1: Matrix 5(Val)  | Matrix 6(Val)  | Matrix 7(Test) | Matrix 8(Test)
    from matplotlib.gridspec import GridSpec
    gs = GridSpec(2, 4, figure=fig, hspace=0.50, wspace=0.35,
                  width_ratios=[1, 1, 1.4, 1])
    ax_t1  = fig.add_subplot(gs[0, 0])   # Matrix 1 (Test): precision vs M1
    ax_t2  = fig.add_subplot(gs[0, 1])   # Matrix 2 (Test): M2 return > M1 return
    ax_t3  = fig.add_subplot(gs[0, 2])   # Matrix 3 (Test): approved vs rejected
    ax_t4  = fig.add_subplot(gs[0, 3])   # Matrix 4 (Test): CV confusion
    ax_v1  = fig.add_subplot(gs[1, 0])   # Matrix 5 (Val):  precision vs M1
    ax_v2  = fig.add_subplot(gs[1, 1])   # Matrix 6 (Val):  val mean ret > 0 at τ̂
    ax_t7  = fig.add_subplot(gs[1, 2])   # Matrix 7 (Test): M2 return > 0
    ax_t8  = fig.add_subplot(gs[1, 3])   # Matrix 8 (Test): mean return / mean sharpe
    axes = [ax_t1, ax_t2, ax_t3, ax_t4]   # backward-compat for existing code

    # ── Helper to render a 2×2 matrix with given cell-value texts and colours ──
    def _draw_matrix(ax, cells, title, cmap_name="RdYlGn",
                     value_for_color=lambda c: c["frac"], vmin=0, vmax=1, fmt_text=None):
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xticklabels(col_labels, fontsize=11, fontweight="bold")
        ax.set_yticklabels(row_labels, fontsize=11, fontweight="bold")
        ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
        ax.set_xlim(-0.5, 1.5); ax.set_ylim(-0.5, 1.5)
        ax.invert_yaxis()
        cmap = plt.get_cmap(cmap_name)
        for (i, j), c in cells.items():
            v = value_for_color(c)
            color = cmap((v - vmin) / max(vmax - vmin, 1e-9))
            ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1,
                                       facecolor=color, edgecolor="black", linewidth=1.5))
            txt = fmt_text(c) if fmt_text else f"{c['frac']:.0%}"
            ax.text(j, i, txt, ha="center", va="center",
                    fontsize=12, fontweight="bold", color="black",
                    linespacing=1.4)
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.tick_params(length=0)

    # Matrix 1 (Test)
    cells1 = {}
    for i, thr in enumerate(["05", "tau"]):
        for j, rel in enumerate([False, True]):
            w, n, frac = _mat1_cell(thr, rel)
            cells1[(i, j)] = {"wins": w, "n": n, "frac": frac}
    _draw_matrix(axes[0], cells1,
                 "Matrix 1 (Test): Precision\n% configs with $M_2$ Prec > $M_1$ baseline",
                 fmt_text=lambda c: f"{c['frac']:.1%}\n({c['wins']}/{c['n']})")

    # Matrix 2 (Test)
    cells2 = {}
    for i, thr in enumerate(["05", "tau"]):
        for j, rel in enumerate([False, True]):
            w, n, frac = _mat2_cell(thr, rel)
            cells2[(i, j)] = {"wins": w, "n": n, "frac": frac}
    _draw_matrix(axes[1], cells2,
                 "Matrix 2 (Test): Profitability\n% configs with $M_2$ Return > $M_1$ Return",
                 fmt_text=lambda c: f"{c['frac']:.1%}\n({c['wins']}/{c['n']})")

    # Matrix 3 (Test) — approved vs rejected
    cells3 = {}
    for i, thr in enumerate(["05", "tau"]):
        for j, rel in enumerate([False, True]):
            avg_app, avg_rej, w, n, frac = _mat3_cell(thr, rel)
            cells3[(i, j)] = {"avg_app": avg_app, "avg_rej": avg_rej,
                              "wins": w, "n": n, "frac": frac}
    def _mat3_fmt(c):
        return (f"App: {c['avg_app']:+.2f}%\n"
                f"Rej: {c['avg_rej']:+.2f}%\n"
                f"App>Rej: {c['frac']:.1%} ({c['wins']}/{c['n']})")
    _draw_matrix(axes[2], cells3,
                 "Matrix 3 (Test): Approved vs Rejected\nMean returns across configs",
                 fmt_text=_mat3_fmt)
    # Slightly smaller font for Matrix 3 to fit 3-line cell text
    for txt in axes[2].texts:
        txt.set_fontsize(10.5)

    # Matrix 4 — CV<0.5 confusion matrix on CPCV-eligible configs (constr=True)
    cpcv_records = _load_cpcv_records(edge_root, bt_root)
    N_cpcv = len(cpcv_records)
    cv_fn  = lambda r: r["cv"] < 0.5
    sel_c  = [r for r in cpcv_records if cv_fn(r)]
    rej_c  = [r for r in cpcv_records if not cv_fn(r)]
    TP_c = sum(r["test_pos"] == 1 for r in sel_c)
    FP_c = sum(r["test_pos"] == 0 for r in sel_c)
    TN_c = sum(r["test_pos"] == 0 for r in rej_c)
    FN_c = sum(r["test_pos"] == 1 for r in rej_c)
    base_pos_c = sum(r["test_pos"] for r in cpcv_records)
    prec_c = TP_c / (TP_c + FP_c) if (TP_c + FP_c) > 0 else 0.0
    acc_c  = (TP_c + TN_c) / N_cpcv if N_cpcv > 0 else 0.0

    ax4 = axes[3]
    # Draw 2×2 confusion matrix: rows=Predicted (Selected/Rejected), cols=Actual (Profitable/Not)
    cm_labels_x = ["Profitable\n(Actual)", "Not Profitable\n(Actual)"]
    cm_labels_y = ["Selected\n(CV<0.5)", "Rejected\n(CV≥0.5)"]
    cm_vals = [[TP_c, FP_c], [FN_c, TN_c]]
    cm_fracs= [[TP_c/N_cpcv, FP_c/N_cpcv], [FN_c/N_cpcv, TN_c/N_cpcv]]
    cm_colors = [["#2ca02c", "#d62728"], ["#ff7f0e", "#1f77b4"]]  # TP=green FP=red FN=orange TN=blue
    cm_labels_abbr = [["TP", "FP"], ["FN", "TN"]]

    ax4.set_xlim(-0.5, 1.5); ax4.set_ylim(-0.5, 1.5)
    ax4.invert_yaxis()
    for i in range(2):
        for j in range(2):
            ax4.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1,
                                        facecolor=cm_colors[i][j], edgecolor="black",
                                        linewidth=1.5, alpha=0.75))
            ax4.text(j, i - 0.18, cm_labels_abbr[i][j],
                     ha="center", va="center", fontsize=14, fontweight="bold", color="white")
            ax4.text(j, i + 0.08, f"{cm_vals[i][j]}",
                     ha="center", va="center", fontsize=13, fontweight="bold", color="white")
            ax4.text(j, i + 0.30, f"({cm_fracs[i][j]:.1%})",
                     ha="center", va="center", fontsize=10, color="white")

    ax4.set_xticks([0, 1]); ax4.set_yticks([0, 1])
    ax4.set_xticklabels(cm_labels_x, fontsize=10, fontweight="bold")
    ax4.set_yticklabels(cm_labels_y, fontsize=10, fontweight="bold")
    for sp in ax4.spines.values():
        sp.set_visible(False)
    ax4.tick_params(length=0)
    ax4.set_title(
        f"Matrix 4 (Test): CV<0.5 Confusion  |  N={N_cpcv} (constr=True)\n"
        f"Base={base_pos_c/N_cpcv:.1%} profitable  |  "
        f"Prec={prec_c:.1%}  |  Acc={acc_c:.1%}  |  Lift={prec_c - base_pos_c/N_cpcv:+.1%}pp",
        fontsize=11, fontweight="bold", pad=10)

    # ── Val Matrix 1: Precision ─────────────────────────────────────────
    def _vmat1_cell(thr_key, reliab):
        pop = [c for c in configs if (c["reliable"] if reliab else True)]
        if thr_key == "tau":
            pop = [c for c in pop if c["constr"]]
        if not pop: return (0, 0, 0.0)
        prec_key = "val_m2_prec_05" if thr_key == "05" else "val_m2_prec_tau"
        wins = sum(1 for c in pop
                   if not np.isnan(c[prec_key]) and not np.isnan(c["val_m1_prec"])
                   and c[prec_key] > c["val_m1_prec"])
        return (wins, len(pop), wins / len(pop))

    # ── Val Matrix 2: Profitability (Val_selective mean_ret > 0 for τ̂; N/A for τ=0.5) ──
    def _vmat2_cell(thr_key, reliab):
        # τ̂: use Val_selective.mean_ret > 0 (stored val net return at optimized threshold)
        # τ=0.5: val trades CSV not stored → return None (N/A)
        if thr_key == "05":
            return None
        pop = [c for c in configs if (c["reliable"] if reliab else True)]
        pop = [c for c in pop if c["constr"]]
        if not pop: return (0, 0, 0.0)
        wins = sum(1 for c in pop
                   if not np.isnan(c["val_mean_ret_tau"]) and c["val_mean_ret_tau"] > 0)
        return (wins, len(pop), wins / len(pop))

    vcells1 = {}
    for i, thr in enumerate(["05", "tau"]):
        for j, rel in enumerate([False, True]):
            w, n, frac = _vmat1_cell(thr, rel)
            vcells1[(i, j)] = {"wins": w, "n": n, "frac": frac}
    _draw_matrix(ax_v1, vcells1,
                 "Matrix 5 (Val): Precision\n% configs with $M_2$ Prec > $M_1$ baseline",
                 fmt_text=lambda c: f"{c['frac']:.1%}\n({c['wins']}/{c['n']})")

    vcells2 = {}
    for i, thr in enumerate(["05", "tau"]):
        for j, rel in enumerate([False, True]):
            res = _vmat2_cell(thr, rel)
            if res is None:
                vcells2[(i, j)] = {"wins": 0, "n": 0, "frac": float("nan"), "na": True}
            else:
                w, n, frac = res
                vcells2[(i, j)] = {"wins": w, "n": n, "frac": frac, "na": False}

    def _vmat2_fmt(c):
        if c.get("na"):
            return "N/A\n(val trades\nnot cached)"
        return f"{c['frac']:.1%}\n({c['wins']}/{c['n']})"

    def _vmat2_color(c):
        if c.get("na"): return 0.5
        return c["frac"]

    _draw_matrix(ax_v2, vcells2,
                 "Matrix 6 (Val): Profitability\n% configs with val mean ret > 0 at $\\hat{\\tau}$",
                 cmap_name="RdYlGn",
                 value_for_color=_vmat2_color,
                 fmt_text=_vmat2_fmt)

    # Matrix 7 (Test): % configs where M2 total return > 0 (absolute profitability)
    def _mat7_cell(thr_key, reliab):
        pop = [c for c in configs if (c["reliable"] if reliab else True)]
        if thr_key == "tau":
            pop = [c for c in pop if c["constr"]]
        if not pop: return (0, 0, 0.0)
        ret_key = "m2_total_ret_05" if thr_key == "05" else "m2_total_ret_tau"
        wins = sum(1 for c in pop
                   if not np.isnan(c[ret_key]) and c[ret_key] > 0)
        return (wins, len(pop), wins / len(pop))

    cells7 = {}
    for i, thr in enumerate(["05", "tau"]):
        for j, rel in enumerate([False, True]):
            w, n, frac = _mat7_cell(thr, rel)
            cells7[(i, j)] = {"wins": w, "n": n, "frac": frac}
    _draw_matrix(ax_t7, cells7,
                 "Matrix 7 (Test): M2 Profitable\n% configs with $M_2$ Return > 0",
                 fmt_text=lambda c: f"{c['frac']:.1%}\n({c['wins']}/{c['n']})")

    # Label rows
    fig.text(0.01, 0.73, "TEST", va="center", ha="left", fontsize=13,
             fontweight="bold", color="#1a1a1a", rotation=90)
    fig.text(0.01, 0.28, "VAL", va="center", ha="left", fontsize=13,
             fontweight="bold", color="#555555", rotation=90)

    # ── Matrix 8 (Test): Mean Return & Mean Sharpe per cell ─────────────────
    def _mat8_cell(thr_key, reliab):
        pop = [c for c in configs if (c["reliable"] if reliab else True)]
        if thr_key == "tau":
            pop = [c for c in pop if c["constr"]]
        if not pop: return (float("nan"), float("nan"), [])
        ret_key = "m2_total_ret_05" if thr_key == "05" else "m2_total_ret_tau"
        sr_key  = "m2_sharpe_05"    if thr_key == "05" else "m2_sharpe_tau"
        rets    = [c[ret_key] for c in pop if not np.isnan(c[ret_key])]
        sharpes = [c[sr_key]  for c in pop if not np.isnan(c[sr_key])]
        mean_ret = float(np.mean(rets))    if rets    else float("nan")
        mean_sr  = float(np.mean(sharpes)) if sharpes else float("nan")
        # store per-config detail for pickle
        details = [{"m1": c["m1"], "m2": c["m2"], "dir": c["dir"], "gran": c["gran"],
                    "total_ret": c[ret_key], "sharpe": c[sr_key],
                    "constr": c["constr"], "reliable": c["reliable"]}
                   for c in pop]
        return (mean_ret, mean_sr, details)

    cells8 = {}
    m8_details = {}
    for i, thr in enumerate(["05", "tau"]):
        for j, rel in enumerate([False, True]):
            mr, ms, dets = _mat8_cell(thr, rel)
            ratio = mr / ms if (np.isfinite(mr) and np.isfinite(ms) and abs(ms) > 1e-9) else float("nan")
            cells8[(i, j)] = {"mean_ret": mr, "mean_sr": ms, "ratio": ratio}
            m8_details[f"thr={'05' if thr=='05' else 'tau'}_rel={'yes' if rel else 'no'}"] = dets

    # Draw Matrix 8 — cell content = mean_ret / mean_sharpe ratio
    ax_t8.set_xlim(-0.5, 1.5); ax_t8.set_ylim(-0.5, 1.5)
    ax_t8.invert_yaxis()
    cmap8 = plt.get_cmap("RdYlGn")
    all_ratios = [v["ratio"] for v in cells8.values() if np.isfinite(v["ratio"])]
    vmin8 = min(all_ratios) if all_ratios else -1
    vmax8 = max(all_ratios) if all_ratios else  1
    for (i, j), c in cells8.items():
        ratio_val = c["ratio"]
        norm_v = (ratio_val - vmin8) / max(vmax8 - vmin8, 1e-9) if np.isfinite(ratio_val) else 0.5
        color = cmap8(np.clip(norm_v, 0, 1))
        ax_t8.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1,
                                       facecolor=color, edgecolor="black", linewidth=1.5))
        ratio_str = f"{ratio_val:+.2f}" if np.isfinite(ratio_val) else "N/A"
        ax_t8.text(j, i, ratio_str,
                   ha="center", va="center", fontsize=14, fontweight="bold", color="black")
    ax_t8.set_xticks([0, 1]); ax_t8.set_yticks([0, 1])
    ax_t8.set_xticklabels(col_labels, fontsize=11, fontweight="bold")
    ax_t8.set_yticklabels(row_labels, fontsize=11, fontweight="bold")
    for sp in ax_t8.spines.values(): sp.set_visible(False)
    ax_t8.tick_params(length=0)
    ax_t8.set_title("Matrix 8 (Test): Return / Sharpe\n"
                    r"Mean total return $\div$ mean Sharpe ratio",
                    fontsize=13, fontweight="bold", pad=12)

    # ── Assemble and save all matrix results to pickle ───────────────────────
    import pickle as _pickle

    def _cell_dict(wins, n, frac):
        return {"wins": wins, "n": n, "frac": round(frac, 4)}

    pickle_data = {
        "N_configs": len(configs),
        "matrix1_test_precision": {
            f"thr={'05' if t=='05' else 'tau'}_rel={'yes' if r else 'no'}":
                _cell_dict(*_mat1_cell(t, r))
            for t in ["05","tau"] for r in [False, True]},
        "matrix2_test_profitability_vs_m1": {
            f"thr={'05' if t=='05' else 'tau'}_rel={'yes' if r else 'no'}":
                _cell_dict(*_mat2_cell(t, r))
            for t in ["05","tau"] for r in [False, True]},
        "matrix3_test_app_vs_rej": {
            f"thr={'05' if t=='05' else 'tau'}_rel={'yes' if r else 'no'}": {
                "avg_app": round(avg_app, 4), "avg_rej": round(avg_rej, 4),
                "wins": w, "n": n, "frac": round(frac, 4)}
            for t in ["05","tau"] for r in [False, True]
            for avg_app, avg_rej, w, n, frac in [_mat3_cell(t, r)]},
        "matrix4_test_cv_confusion": {
            "N": N_cpcv, "base_profitable": round(base_pos_c / N_cpcv, 4),
            "TP": TP_c, "FP": FP_c, "TN": TN_c, "FN": FN_c,
            "precision": round(prec_c, 4), "accuracy": round(acc_c, 4),
            "lift": round(prec_c - base_pos_c / N_cpcv, 4)},
        "matrix5_val_precision": {
            f"thr={'05' if t=='05' else 'tau'}_rel={'yes' if r else 'no'}":
                _cell_dict(*_vmat1_cell(t, r))
            for t in ["05","tau"] for r in [False, True]},
        "matrix6_val_profitability": {
            f"thr=tau_rel={'yes' if r else 'no'}": (lambda res: _cell_dict(*res) if res else {"na": True})(_vmat2_cell("tau", r))
            for r in [False, True]},
        "matrix7_test_m2_profitable": {
            f"thr={'05' if t=='05' else 'tau'}_rel={'yes' if r else 'no'}":
                _cell_dict(*_mat7_cell(t, r))
            for t in ["05","tau"] for r in [False, True]},
        "matrix8_test_return_over_sharpe": {
            f"thr={'05' if i==0 else 'tau'}_rel={'no' if j==0 else 'yes'}": {
                "mean_ret":   round(v["mean_ret"], 4) if np.isfinite(v["mean_ret"]) else None,
                "mean_sr":    round(v["mean_sr"],  4) if np.isfinite(v["mean_sr"])  else None,
                "ratio":      round(v["ratio"],    4) if np.isfinite(v["ratio"])    else None}
            for (i, j), v in cells8.items()},
        "matrix8_individual_configs": m8_details,
        "all_configs": [
            {k: (float(v) if isinstance(v, float) and not np.isnan(v) else
                 (None if isinstance(v, float) and np.isnan(v) else v))
             for k, v in c.items()}
            for c in configs],
    }

    pkl_path = save_path.parent / "results_matrices_data.pkl"
    with open(pkl_path, "wb") as _f:
        _pickle.dump(pickle_data, _f)
    print(f"[plot_results_matrices] Pickle -> {pkl_path}")

    fig.suptitle(
        f"Results Matrices Summary  |  N={len(configs)} configs across "
        f"{len(_M1_LIST)} M1 × {len(_M2_LIST)} M2 × 2 directions × {len(_GRANS_LIST)} granularities",
        fontsize=14, fontweight="bold", y=1.02)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(save_path), dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[plot_results_matrices] Saved -> {save_path}")



def plot_best_m2_per_gran(metrics_pkl: Path,
                          save_path: Path,
                          metric: str = "precision",
                          which_threshold: str = "tauhat") -> None:
    """For every (M1, granularity, direction) cell, pick the best-performing M2
    by ``metric`` and visualise three quantities side-by-side:

      • Precision bar (solid colour, primary y-axis)
      • Coverage bar (hatched ``--``, secondary y-axis)
      • ΔPrecision overlay (white + ``////`` hatch, on top of precision)

    Two rows: top = UP, bottom = DOWN. M1 letters (T/C/F/K) are drawn only on
    the top row; the column structure carries to the bottom row by alignment.

    Parameters
    ----------
    metrics_pkl : Path
        Path to ``metrics_combined_dict.pkl`` produced by
        :func:`build_combined_metrics_dict`.
    save_path : Path
        Output PNG path.
    metric : {"precision"}
        Metric used to rank M2 candidates within each (M1, gran, direction) cell.
    which_threshold : {"tauhat", "tau05"}
        Whether to use the calibrated τ̂ metrics (falls back to τ=0.5 if the
        config did not converge) or the raw τ=0.5 metrics.
    """
    import pickle as _pickle
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D

    with open(metrics_pkl, "rb") as f:
        metrics = _pickle.load(f)

    M1_LIST    = ["Tirex", "Chronos2", "Fincast", "Kronos"]
    M1_LETTER  = {"Tirex": "T", "Chronos2": "C", "Fincast": "F", "Kronos": "K"}
    M2_LIST    = ["rf", "autogluon", "tabpfn", "tabicl", "ctts"]
    M2_LABELS  = {"rf": "Random Forest", "autogluon": "AutoGluon",
                  "tabpfn": "TabPFN", "tabicl": "TabICL", "ctts": "CTTS"}
    GRANS      = ["1d", "12h", "8h", "6h", "4h", "2h", "1h", "30m"]
    DIRS       = ["UP", "DOWN"]

    # Smoother, well-separated palette (muted)
    M2_COLOR = {
        "rf":        "#7FB069",   # muted green
        "autogluon": "#E89A4F",   # muted orange
        "tabpfn":    "#6FA8DC",   # muted blue
        "tabicl":    "#C28EC9",   # muted purple
        "ctts":      "#D97374",   # muted red
    }

    # ── Helper: extract (precision, coverage, m1_precision) for one cell ──
    # Reject degenerate cells (n_trades == 0): a precision computed over zero
    # approved trades is not a meaningful 1.0 — it is undefined. We fall back
    # from τ̂ → τ=0.5 in that case, and skip the cell entirely if both fail.
    def _cell(m1: str, m2: str, dr: str, g: str) -> tuple:
        e = metrics.get(m1, {}).get(m2, {}).get(dr, {}).get(g, None)
        if e is None:
            return None
        prec = float("nan"); n_m2 = 0
        if which_threshold == "tauhat" and e.get("constraint_satisfied", False):
            n_t = int(e.get("m2_n_trades_tauhat", 0))
            if n_t > 0:
                prec = e.get("m2_precision_tauhat", float("nan"))
                n_m2 = n_t
        if n_m2 == 0:  # fall back to τ=0.5
            n_5 = int(e.get("m2_n_trades_tau05", 0))
            if n_5 > 0:
                prec = e.get("m2_precision_tau05", float("nan"))
                n_m2 = n_5
        if n_m2 == 0:
            return None
        n_m1 = e.get("m1_n_trades", 0)
        cov  = (n_m2 / n_m1) if n_m1 > 0 else float("nan")
        m1_prec = e.get("m1_precision", float("nan"))
        return prec, cov, m1_prec

    def _is_nan(x) -> bool:
        return isinstance(x, float) and x != x

    def _delta(prec: float, m1_prec: float) -> float:
        """Δ = M2 precision − M1 baseline precision (only positive segment shown)."""
        if _is_nan(prec) or _is_nan(m1_prec):
            return 0.0
        return float(max(prec - m1_prec, 0.0))

    # ── Typography (single unified font size) ─────────────────────────────
    FS = 10  # used for ALL labels, ticks, legend, M1 letters
    plt.rcParams["hatch.linewidth"] = 1.4
    plt.rcParams["hatch.color"]     = "#111111"

    # ── Geometry ──────────────────────────────────────────────────────────
    # For each (M1, gran) cell we draw a doublet: precision-bar then
    # coverage-bar, strictly side-by-side and non-overlapping.
    bar_w      = 0.34          # individual bar width (precision = coverage)
    pair_pitch = 2 * bar_w + 0.04   # horizontal span of one (prec, cov) doublet
    gap_in_grp = pair_pitch + 0.10  # spacing between consecutive M1 doublets
    grp_gap    = 0.50               # extra space between granularity groups
    group_w    = (len(M1_LIST) - 1) * gap_in_grp + pair_pitch
    x_gran     = np.arange(len(GRANS)) * (group_w + grp_gap)

    fig, axes = plt.subplots(2, 1, figsize=(15, 5.6),
                             sharex=True, facecolor="white")
    axes_right = []

    for row_idx, dr in enumerate(DIRS):
        ax  = axes[row_idx]
        ax2 = ax.twinx()
        axes_right.append(ax2)

        # Compute per-row maxima for nice y-limits
        row_prec_max = 0.5
        row_cov_max  = 0.0
        for m1 in M1_LIST:
            for g in GRANS:
                # Find best M2 for this cell
                best = None
                best_v = -np.inf
                for m2 in M2_LIST:
                    c = _cell(m1, m2, dr, g)
                    if c is None: continue
                    pv = c[0]
                    if pv is not None and not (isinstance(pv, float) and pv != pv) and pv > best_v:
                        best_v, best = pv, c
                if best is None: continue
                prec, cov, _ = best
                if prec == prec: row_prec_max = max(row_prec_max, prec)
                if cov  == cov:  row_cov_max  = max(row_cov_max,  cov)

        row_prec_max = max(row_prec_max, 0.51)
        row_cov_max  = max(row_cov_max,  0.05)
        # Both axes always span [0, 1] — never truncated.
        head_prec = 1.0
        head_cov  = 1.0
        ax.set_ylim(0, head_prec)
        ax2.set_ylim(0, head_cov)

        # Alternating grey background per granularity group — span the FULL
        # doublet width (precision + coverage) for every M1 in the group.
        for i, gx in enumerate(x_gran):
            if i % 2 == 1:
                left  = gx - 0.10
                right = gx + (len(M1_LIST) - 1) * gap_in_grp + pair_pitch + 0.10
                ax.axvspan(left, right, color="#F0F0F0", zorder=0)

        # 0.5 reference line on precision (left, no-skill baseline)
        ax.axhline(0.5, color="#888888", linewidth=0.7,
                   linestyle=":", alpha=0.8, zorder=1)

        # Plot bars
        for j, m1 in enumerate(M1_LIST):
            for g_idx, g in enumerate(GRANS):
                # Find best M2
                best_m2 = None; best_v = -np.inf; best_cell = None
                for m2 in M2_LIST:
                    c = _cell(m1, m2, dr, g)
                    if c is None: continue
                    pv = c[0]
                    if pv is not None and not (isinstance(pv, float) and pv != pv) and pv > best_v:
                        best_v, best_m2, best_cell = pv, m2, c
                if best_m2 is None: continue

                prec, cov, m1_prec = best_cell
                delta = _delta(prec, m1_prec)
                col = M2_COLOR[best_m2]
                # Doublet origin (precision bar at x_p, coverage bar at x_c)
                doublet_x0 = x_gran[g_idx] + j * gap_in_grp
                x_p = doublet_x0 + bar_w / 2
                x_c = doublet_x0 + bar_w + 0.04 + bar_w / 2

                # Precision bar (total height = M2 precision). The bar is split:
                #   • bottom segment [0, M1 baseline] in solid colour
                #   • top segment [M1 baseline, M2 precision] hatched //// to
                #     visualise Δ = M2 − M1 (the lift contributed by M2)
                # No stacking on top — total height never exceeds M2 precision.
                m1_base = prec - delta if delta > 0 else prec
                # solid base up to M1 baseline (or full bar if Δ <= 0)
                ax.bar(x_p, m1_base, width=bar_w, color=col,
                       edgecolor="#222222", linewidth=0.7, zorder=3)
                # Δ band (hatched) inside the bar, between M1 and M2
                if delta > 0:
                    ax.bar(x_p, delta, width=bar_w, bottom=m1_base,
                           facecolor=col, edgecolor="#222222",
                           linewidth=0.7, hatch="////", alpha=0.95, zorder=3)
                # coverage bar (right axis) — strictly to the right of prec
                ax2.bar(x_c, cov, width=bar_w, color=col,
                        edgecolor="#222222", linewidth=0.5,
                        hatch="---", alpha=0.85, zorder=2)

                # M1 letter — directly above the precision bar (BOTH rows)
                ax.text(x_p, min(prec + 0.015, 0.985), M1_LETTER[m1],
                        ha="center", va="bottom", fontsize=FS,
                        color="#333333", zorder=5)

        # Cosmetics
        ax.set_ylabel("Precision", fontsize=FS, fontweight="bold")
        ax2.set_ylabel("Coverage", fontsize=FS, fontweight="bold",
                       color="#333333")
        ax.tick_params(axis="y", labelsize=FS)
        ax2.tick_params(axis="y", labelsize=FS, colors="#333333")
        # UP/DOWN tag inside each subplot, top-RIGHT
        # On UP, place the cheatsheet first (further left) and the UP tag at
        # the far right so they sit side-by-side without overlap.
        if row_idx == 0:
            ax.text(0.005, 0.96,
                    "T = Tirex   C = Chronos2   F = Fincast   K = Kronos",
                    transform=ax.transAxes, ha="left", va="top",
                    fontsize=FS, color="#222222",
                    bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                              edgecolor="#888888", linewidth=0.6, alpha=0.95),
                    zorder=10)
        ax.text(0.99, 0.96, dr, transform=ax.transAxes,
                ha="right", va="top", fontsize=FS, fontweight="bold",
                color="#222222",
                bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                          edgecolor="#888888", linewidth=0.6, alpha=0.95),
                zorder=10)
        for spine in ("top",):
            ax.spines[spine].set_visible(False)
            ax2.spines[spine].set_visible(False)
        ax.set_axisbelow(True)
        ax.grid(axis="y", color="#DDDDDD", linewidth=0.5, alpha=0.7, zorder=0)

    # ── x-axis on bottom row only ─────────────────────────────────────────
    # Centre the x-tick under the full doublet span, not just precision
    full_grp_w = (len(M1_LIST) - 1) * gap_in_grp + pair_pitch
    for ax_row in axes:
        ax_row.set_xticks(x_gran + full_grp_w / 2)
        ax_row.set_xticklabels(GRANS, fontsize=FS, fontweight="bold")
        ax_row.set_xlim(x_gran[0] - 0.25, x_gran[-1] + full_grp_w + 0.10)
        ax_row.tick_params(axis="x", labelsize=FS, labelbottom=True)
    axes[1].set_xlabel("Temporal Granularity", fontsize=FS, fontweight="bold")

    # ── Single combined legend at the BOTTOM (one row) ────────────────────
    # Order: M2 colour swatches → M1 Precision → M2 Coverage → Δ Precision.
    # Last three patches use a white background.
    handles = (
        [Patch(facecolor=M2_COLOR[m2], edgecolor="#222222",
               linewidth=0.7, label=M2_LABELS[m2]) for m2 in M2_LIST]
        + [
            Patch(facecolor="white", edgecolor="#222222",
                  label="M1 Precision"),
            Patch(facecolor="white", edgecolor="#222222", hatch="---",
                  label="M2 Coverage"),
            Patch(facecolor="white", edgecolor="#222222", hatch="////",
                  label=r"$\Delta$ Precision"),
        ]
    )

    fig.legend(handles=handles, loc="lower center",
               bbox_to_anchor=(0.5, 0.005),
               ncol=len(handles), frameon=True, framealpha=0.95,
               edgecolor="#BDC3C7", fontsize=FS,
               handlelength=2.0, handletextpad=0.6, columnspacing=1.4,
               borderpad=0.6)

    fig.subplots_adjust(top=0.96, hspace=0.20, left=0.06, right=0.95, bottom=0.17)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(save_path), dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[plot_best_m2_per_gran] Saved -> {save_path}")
