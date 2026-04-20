"""Feature analysis and plotting helpers extracted from kronos_tree.py."""

import warnings
import re
import glob
import os
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
    "plot_ocp_threshold_evolution",
    "plot_selective_return_distribution",
    "plot_asset_correlation",
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
                                      test_approved_ocp: Optional[np.ndarray] = None):
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
    _plot_dynamic_return(ax_ret, covs[valid_w], mean_win_rets[valid_w] * 100, 1.0, ":", 0.8, "_nolegend_", 2)
    _plot_dynamic_return(ax_ret, covs[valid_l], mean_lose_rets[valid_l] * 100, 1.0, ":", 0.8, "_nolegend_", 2)
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

    ax_rc.set_title(f"Coverage-Risk  |  {split_name}  |  {model_label}",
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


def plot_performance_over_n_features(base_dir: str ="/home/till/PycharmProjects/Secondary-Model/src/Output",
                                     m1: str = "kronos",
                                     m2: str = "randforest",
                                     direction: str = "up",
                                     granularity: str = "1d",
                                     meta_label_mode: str = "tp",
                                     scoring: str = "accuracy",
                                     cv_strategy: str = "CombinatorialPurgedEmbargoCV",
                                     n_splits: int = 10) -> None:
    # 1_features_accuracy_CombinatorialPurgedEmbargoCV_10_cached.csv

    search_dir = (f"{base_dir}/"
                  f"{m1.capitalize()}/"
                  f"{m2}/"
                  f"{direction.upper()}/"
                  f"interpretability/"
                  f"feature_selection/"
                  f"{granularity}_{meta_label_mode}")
    file_mask = f"*_features_{scoring}_{cv_strategy}_{n_splits}_cached.csv"
    # print(search_dir)
    # files = os.listdir(search_dir)
    # for file in files:
    #     print(file)
    # print(file_mask)
    files = sorted(
        glob.glob(f"{search_dir}/{file_mask}"),
        key=lambda x: int(re.search(r"^(\d+)_features", os.path.basename(x)).group(1))
    )

    n_features = []
    val_mean = []
    val_std = []
    test_mean = []
    test_std = []

    for file in files:
        df = pd.read_csv(file)
        file_name = os.path.basename(file)
        n_feature = int(file_name.split("_")[0])

        best_idx = df['mean_val_scoring'].argmax()
        n_features.append(n_feature)
        val_mean.append(df['mean_val_scoring'].iloc[best_idx])
        val_std.append(df['std_val_scoring'].iloc[best_idx])
        test_mean.append(df['mean_test_scoring'].iloc[best_idx])
        test_std.append(df['std_test_scoring'].iloc[best_idx])

    n_features = np.array(n_features)
    val_mean = np.array(val_mean)
    val_std = np.array(val_std)
    test_mean = np.array(test_mean)
    test_std = np.array(test_std)

    # plot
    fig, ax = plt.subplots(figsize=(12, 5))

    ax.plot(n_features, val_mean, label="Validation", marker="o")
    ax.fill_between(n_features, val_mean - val_std, val_mean + val_std, alpha=0.2)

    ax.plot(n_features, test_mean, label="Test", marker="o")
    ax.fill_between(n_features, test_mean - test_std, test_mean + test_std, alpha=0.2)

    ax.set_xlabel("Number of Features")
    ax.set_ylabel("Scoring")
    ax.set_title(f"M1={m1} | M2={m2} | time frame={granularity} | direction={direction} | meta label mode={meta_label_mode}")
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    # plt.show()
    plt.savefig(f"{search_dir}/strategy={cv_strategy}_scoring={scoring}_n_splits={n_splits}_min_max={1}_{len(files)}_summary_plot.pdf")
    plt.close()