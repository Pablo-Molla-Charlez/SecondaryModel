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
    "plot_temporal_risk_coverage_curve_final",
    "plot_ocp_threshold_evolution",
    "plot_selective_return_distribution",
    "plot_asset_correlation",
    "plot_dataset_size_distribution",
    "plot_return_quality_distribution",
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
      * Stage-A utility curve U(τ) = t_reg × cov_factor on a third y-axis,
        drawn only where Stage-A hard constraints hold (so the feasible region
        is reinforced visually). A gold star marks argmax U.
      * Two reference utility curves: t_reg (no penalty) and t_reg × min(1, cov/cov*)
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
    c_baseline = "#7D3C98"
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
    ax_rc.set_xlabel("Coverage", fontsize=11, fontweight="bold", color="black", labelpad=8)
    ax_rc.set_ylabel("Risk", fontsize=11, fontweight="bold", color="black", labelpad=8)
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
            if special_colors[t] == c_op:
                lbl.set_fontsize(8)
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
                       fontsize=11, fontweight="bold", color=c_util, labelpad=8)
    ax_util.tick_params(axis="y", colors=c_util, labelcolor=c_util, labelsize=8, width=1.2)
    plt.setp(ax_util.get_yticklabels(), fontweight="bold")

    # ┏━━━━━━━━━━ Baseline precision floor (segment, skips forbidden zone) ━━━━━━━━━━┓
    risk_floor = 1.0 - max(_plot_prec_argmax, float(m1_precision) if m1_precision is not None and not (isinstance(m1_precision, float) and m1_precision != m1_precision) else 0.0)
    ax_rc.plot([cov_min, 1.0], [risk_floor, risk_floor],
               color=c_baseline, linestyle="-.", linewidth=1.6, alpha=0.9, zorder=3)
    # Inline label at the right end of the line, just below it.
    ax_rc.annotate(r"$Risk_{Floor}$",
                   xy=(1.0, risk_floor), xytext=(-4, 8), textcoords="offset points",
                   fontsize=9, color=c_baseline, fontweight="bold", ha="right", va="top",
                   zorder=11)

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
        ax_rc.annotate("τ=0.50", xy=(cov_05, risk_05), xytext=(3, 5), textcoords="offset points",
                       fontsize=7, color=c_thr05, fontweight="bold", zorder=10,
                       bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=c_thr05, alpha=0.9, lw=0.6))

    aop = _zone_alpha(op_cov)  # kept for axvline only
    ax_rc.axvline(x=op_cov, color=c_op, linestyle="--", alpha=0.7, linewidth=1.8)
    ax_rc.scatter([op_cov], [op_risk], color=c_op, marker="D", s=50,
                  edgecolors="white", linewidths=1.0, zorder=6)
    ax_rc.annotate(f"$\\hat{{\\tau}}$={op['threshold']:.3f}", xy=(op_cov, op_risk), xytext=(3, 6),
                   textcoords="offset points", fontsize=7.5, color=c_op, fontweight="bold", zorder=10,
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
                        textcoords="offset points", fontsize=7.5, color=color,
                        fontweight="bold", zorder=10)



    # ┏━━━━━━━━━━ Title ━━━━━━━━━━┓
    _model_display = {"RF": "Random Forest", "rf": "Random Forest"}.get(model_label, model_label)
    _split_display = {"Val": "Validation", "val": "Validation"}.get(split_name, split_name)
    _dir_gran = f"  |  {direction.upper()}  {granularity}" if direction or granularity else ""
    ax_rc.set_title(f"Profitability-Risk  |  {_split_display}  |  {_model_display}{_dir_gran}",
                    fontsize=13, fontweight="bold", color="#2C3E50", pad=12)

    # ┏━━━━━━━━━━ Single legend: 3 rows × 4 columns ━━━━━━━━━━┓
    # Stats (Prec, Cov, μ/t) are embedded as multi-line text in col-4 entries
    # so no phantom handle-space is wasted on invisible patches.
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    _prec_str = rf"Prec$_{{\hat{{τ}}}}$={op.get('precision', float('nan'))*100:.1f}%"
    _cov_str  = rf"Cov$_{{\hat{{τ}}}}$={op['coverage']*100:.1f}% (N={op.get('selected_count', 0)})"
    _mut_str  = rf"$\mu$={op['mean_ret']*100:+.2f}%, t={op.get('t_stat', 0):.2f}"
    handles = [
        # ── Column 1: Curves ──────────────────────────────────────────
        Line2D([], [], color=c_util, linewidth=1.8, linestyle="--", label=r"Risk-Profitability Score"),
        Line2D([], [], color=c_ret, linewidth=2.0, label="Mean Net Return"),
        Line2D([], [], color=c_risk, linewidth=2.2, label="Risk-Coverage Curve"),
        # ── Column 2: Zones ───────────────────────────────────────────
        Patch(facecolor=c_forbid, alpha=0.30, hatch="//", edgecolor=c_forbid,
              label=r"Forbidden: $\mathrm{Cov} < C_{min}$"),
        Patch(facecolor=c_penalty, alpha=0.20, hatch="..", edgecolor=c_penalty,
              label=r"Quadratic Penalty: $\mathrm{Cov} < C^{*}$"),
        Line2D([], [], color=c_baseline, linewidth=1.6, linestyle="-.", label=r"Risk Floor: $Risk_{floor}$"),
        # ── Column 3: Operating points + stats ────────────────────────
        Line2D([], [], color=c_util, marker="*", markersize=13, linestyle="None",
               markeredgecolor="white", markeredgewidth=1.0,
               label=rf"Risk-Profitability Score   {_prec_str}"),
        Line2D([], [], color=c_op, marker="D", markersize=7, linestyle="--",
               markeredgecolor="white", markeredgewidth=0.8,
               label=rf"$\hat{{\tau}}$={op['threshold']:.3f}   ({thr_source})   {_cov_str}"),
        Line2D([], [], color=c_thr05, marker="o", markersize=6, linestyle="--",
               markeredgecolor="white", markeredgewidth=0.8,
               label=rf"$\tau$=0.5 (Baseline)            {_mut_str}"),
    ]

    # Center the legend on the main axes span [left=0.08, right=0.92].
    _leg_cx = (0.08 + 0.92) / 2
    leg = fig_rc.legend(handles=handles, loc="lower center",
                        bbox_to_anchor=(_leg_cx, 0.01), ncol=3,
                        prop={"size": 8.5}, frameon=True, framealpha=0.95,
                        edgecolor="#BDC3C7", fancybox=True,
                        handlelength=2.2, handletextpad=0.6,
                        columnspacing=1.2, borderpad=0.6)
    leg.set_zorder(20)
    fig_rc.tight_layout()
    fig_rc.subplots_adjust(left=0.08, bottom=0.22, right=0.92, top=0.93)

    fig_rc.savefig(str(save_path), dpi=500, facecolor="white")
    plt.close(fig_rc)


def plot_temporal_risk_coverage_curve_final_copy(save_path: Path,
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
                                            granularity: str = ""):
    """Risk-coverage plot that visualizes the Stage-A optimization problem.

    Adds on top of `plot_temporal_risk_coverage_curve`:
      * Forbidden zone (Cov < cov_min) shaded red/hatched.
      * Quadratic-penalty zone (cov_min ≤ Cov < cov_star) shaded orange/hatched.
      * Baseline risk floor from M2 τ=0.5 precision: τ̂ must land below it.
      * Stage-A utility curve U(τ) = t_reg × cov_factor on a third y-axis,
        drawn only where Stage-A hard constraints hold (so the feasible region
        is reinforced visually). A gold star marks argmax U.
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
    c_baseline = "#7D3C98"
    c_util     = "#B7950B"

    fig_rc, ax_rc = plt.subplots(figsize=(10.5, 6.8), facecolor="white")
    ax_rc.set_facecolor("#FAFAFA")
    x1, y1 = cov_min, cov_min  # [0, C_min] uses natural 1:1 scale
    x2 = cov_star
    if x2 <= x1 + 1e-5: x2 = min(1.0, x1 + 0.1)
    y2 = min(0.95, y1 + 0.3)  # [C_min, 2*C*] gets warped to take up 75%
    
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
    ax_rc.set_xlabel("Coverage", fontsize=11, fontweight="bold", color="black", labelpad=8)
    ax_rc.set_ylabel("Risk", fontsize=11, fontweight="bold", color="black", labelpad=8)
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
        elif abs(op_cov_tmp - cov_star) < 0.001:
            special_labels[cov_star] += "\n" + r"$\hat{\tau}$" + f"({op_cov_tmp:.2f})"
            special_colors[cov_star] = c_op
        else:
            special_ticks.append(op_cov_tmp)
            special_labels[op_cov_tmp] = rf"{op_cov_tmp:.2f}"
            special_colors[op_cov_tmp] = c_op

    final_ticks = list(special_ticks)
    for t in xticks_cur:
        clash_tol = 0.005 if cov_min <= t <= cov_star else 0.025
        if not any(abs(t - st) < clash_tol for st in special_ticks):
            final_ticks.append(t)
            
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
            if special_colors[t] == c_op:
                lbl.set_fontsize(8)
    ax_rc.set_xlim(0.0, 1.0)

    # ┏━━━━━━━━━━ Per-threshold mean returns (plotted-split dataset) ━━━━━━━━━━┓
    mean_rets = np.full_like(thrs, np.nan)
    mean_win_rets = np.full_like(thrs, np.nan)
    mean_lose_rets = np.full_like(thrs, np.nan)

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
        winners = net[lab == 1]
        losers = net[lab == 0]
        if len(winners) >= 1:
            mean_win_rets[i] = float(np.nanmean(winners))
        if len(losers) >= 1:
            mean_lose_rets[i] = float(np.nanmean(losers))

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

    # Mirror optimizer grid: grid_lo = clamp(median(pos_probs), 0.50, 0.85)
    pos_probs = _u_probs[_u_probs >= 0.50]
    if pos_probs.size >= max(min_trades * 2, 20):
        grid_lo = min(max(float(np.median(pos_probs)), 0.50), 0.85)
    else:
        grid_lo = 0.50

    # Evaluate U(τ) on a fine grid (1000 pts) over the optimizer's dataset — smooth by construction.
    _u_thr_grid = np.linspace(grid_lo, 0.95, 1000)
    _u_covs     = np.full(len(_u_thr_grid), np.nan)
    utilities   = np.full(len(_u_thr_grid), np.nan)

    for i, thr in enumerate(_u_thr_grid):
        sel = _u_probs >= thr
        n = int(sel.sum())
        if n < min_trades:
            continue
        cov = n / _u_N
        if cov < cov_min:
            continue
        net = _u_rets[sel] - fee
        lab = _u_y[sel]
        mu = float(np.nanmean(net))
        if mu <= 0:
            continue
        prec_thr = float(lab.mean())
        if prec_thr < prec_argmax:
            continue
        sample_var = float(np.nanvar(net, ddof=1)) if n > 1 else base_var
        shrinkage = n_prior / (n + n_prior)
        reg_var = (1 - shrinkage) * sample_var + shrinkage * base_var
        reg_std = np.sqrt(max(reg_var, 1e-12))
        if reg_std <= 0:
            continue
        t_reg = mu / reg_std * np.sqrt(n)
        if t_reg < t_min:
            continue
        cov_factor = 1.0 if cov >= cov_star else (cov / cov_star) ** 2
        utilities[i] = t_reg * cov_factor
        _u_covs[i] = cov

    # ┏━━━━━━━━━━ Return axis (right, primary) ━━━━━━━━━━┓
    ax_ret = ax_rc.twinx()
    valid = ~np.isnan(mean_rets) & (covs >= cov_min)
    valid_w = ~np.isnan(mean_win_rets) & (covs >= cov_min)
    valid_l = ~np.isnan(mean_lose_rets) & (covs >= cov_min)
    # Profitability zone: shaded light-green band between Mean Win Return and
    # Mean Net Return (the "above-average winners" region).
    profit_band = valid & valid_w
    if profit_band.any():
        ax_ret.fill_between(covs[profit_band],
                            mean_win_rets[profit_band] * 100,
                            mean_rets[profit_band] * 100,
                            alpha=0.13, color=c_win, zorder=1, label="_nolegend_")

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
    _plot_dynamic_return(ax_ret, covs[valid_w], mean_win_rets[valid_w] * 100, 1.0, ":", 0.85, "_nolegend_", 2)
    _plot_dynamic_return(ax_ret, covs[valid_l], mean_lose_rets[valid_l] * 100, 1.0, ":", 0.85, "_nolegend_", 2)

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
    valid_u = ~np.isnan(utilities)
    star_xy = None
    if valid_u.any():
        xu_raw = _u_covs[valid_u]
        yu_raw = utilities[valid_u]
        order_u = np.argsort(xu_raw)
        xu_sorted = xu_raw[order_u]
        yu_sorted = yu_raw[order_u]
        # Deduplicate equal coverages: keep the MAX utility at each unique coverage.
        uniq_cov, inv = np.unique(xu_sorted, return_inverse=True)
        uniq_util = np.full_like(uniq_cov, -np.inf, dtype=float)
        for k, v in zip(inv, yu_sorted):
            if v > uniq_util[k]:
                uniq_util[k] = v
        xu = uniq_cov
        yu = uniq_util
        # Smooth with PCHIP if enough points
        if xu.size >= 3:
            try:
                from scipy.interpolate import PchipInterpolator
                grid_xu = np.linspace(xu.min(), xu.max(), 300)
                yu_smooth = PchipInterpolator(xu, yu, extrapolate=False)(grid_xu)
                mvalid = np.isfinite(yu_smooth)
                xu_plot, yu_plot = grid_xu[mvalid], yu_smooth[mvalid]
            except Exception:
                xu_plot, yu_plot = xu, yu
        else:
            xu_plot, yu_plot = xu, yu
        # Utility curve: full alpha in both penalty and OK zones (user request:
        # max alpha in the penalty zone for the utility curve specifically).
        for lo, hi in [(cov_min, cov_star), (cov_star, np.inf)]:
            m = (xu_plot >= lo) & (xu_plot <= hi)
            if m.sum() >= 2:
                ax_util.plot(xu_plot[m], yu_plot[m], color=c_util, linestyle="--",
                             linewidth=1.8, alpha=1.0, zorder=3)
        # Gold star: argmax U on the deduped grid (satisfies all Stage-A gates)
        i_max = int(np.argmax(yu))
        star_xy_main = (float(xu[i_max]), float(yu[i_max]))
        ax_util.scatter([star_xy_main[0]], [star_xy_main[1]], color=c_util,
                        marker="*", s=200, edgecolors="white", linewidths=1.2,
                        zorder=9)
    ax_util.set_ylabel("Risk-Profitability Score",
                       fontsize=11, fontweight="bold", color=c_util, labelpad=8)
    ax_util.tick_params(axis="y", colors=c_util, labelcolor=c_util, labelsize=8, width=1.2)
    plt.setp(ax_util.get_yticklabels(), fontweight="bold")

    # ┏━━━━━━━━━━ Baseline precision floor (segment, skips forbidden zone) ━━━━━━━━━━┓
    risk_floor = 1.0 - prec_argmax
    ax_rc.plot([cov_min, 1.0], [risk_floor, risk_floor],
               color=c_baseline, linestyle="-.", linewidth=1.6, alpha=0.9, zorder=3)
    # Inline label at the right end of the line, just below it.
    ax_rc.annotate(r"$Risk_{M2@0.5}$",
                   xy=(1.0, risk_floor), xytext=(-4, -10), textcoords="offset points",
                   fontsize=9, color=c_baseline, fontweight="bold", ha="right", va="top",
                   zorder=11)

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
        ax_rc.annotate("τ=0.50", xy=(cov_05, risk_05), xytext=(3, 5), textcoords="offset points",
                       fontsize=7, color=c_thr05, fontweight="bold", zorder=10,
                       bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=c_thr05, alpha=0.9, lw=0.6))

    aop = _zone_alpha(op_cov)  # kept for axvline only
    ax_rc.axvline(x=op_cov, color=c_op, linestyle="--", alpha=0.7, linewidth=1.8)
    ax_rc.scatter([op_cov], [op_risk], color=c_op, marker="D", s=50,
                  edgecolors="white", linewidths=1.0, zorder=6)
    ax_rc.annotate(f"$\\hat{{\\tau}}$={op['threshold']:.3f}", xy=(op_cov, op_risk), xytext=(3, 6),
                   textcoords="offset points", fontsize=7.5, color=c_op, fontweight="bold", zorder=10,
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
    if n_op >= 2:
        net_op = _ann_rets[sel_op] - fee
        lab_op = _ann_labels[sel_op]
        w_op = net_op[lab_op == 1]
        l_op = net_op[lab_op == 0]
        mw_val = float(np.nanmean(w_op)) * 100 if len(w_op) >= 1 else None
        ml_val = float(np.nanmean(l_op)) * 100 if len(l_op) >= 1 else None
    else:
        mw_val, ml_val = None, None

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
    # Label values: already set from the optimizer's dataset (mr_val/mw_val/ml_val)
    # when opt_rets is provided; otherwise also use the curve interpolation.
    mr_dot = _interp_on_curve(op_cov, valid,   mean_rets      * 100)
    mw_dot = _interp_on_curve(op_cov, valid_w, mean_win_rets  * 100)
    ml_dot = _interp_on_curve(op_cov, valid_l, mean_lose_rets * 100)
    if opt_rets is None:
        # Calibrated mode: labels and dots both come from the plotted curves.
        if mr_dot is not None: mr_val = mr_dot
        if mw_dot is not None: mw_val = mw_dot
        if ml_dot is not None: ml_val = ml_dot

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

    _dot_positions = {"mr": mr_dot, "mw": mw_dot, "ml": ml_dot}
    _label_values  = {"mr": mr_val, "mw": mw_val, "ml": ml_val}
    offs_op = _get_staggered_offsets({k: v for k, v in _dot_positions.items() if v is not None})
    for key in ("mr", "mw", "ml"):
        dot_y   = _dot_positions[key]
        label_v = _label_values[key]
        if dot_y is None or label_v is None:
            continue
        color = c_ret if label_v >= 0 else c_ret_neg
        ax_ret.scatter([op_cov], [dot_y], color=color, marker="D" if key == "mr" else "o",
                       s=40 if key == "mr" else 30,
                       edgecolors="white", linewidths=1.0 if key == "mr" else 0.7,
                       zorder=7 if key == "mr" else 6)
        ax_ret.annotate(f"{label_v:+.2f}%", xy=(op_cov, dot_y), xytext=offs_op.get(key, (3, 0)),
                        textcoords="offset points", fontsize=7.5, color=color,
                        fontweight="bold", zorder=10)



    # ┏━━━━━━━━━━ Title ━━━━━━━━━━┓
    _model_display = {"RF": "Random Forest", "rf": "Random Forest"}.get(model_label, model_label)
    _split_display = {"Val": "Validation", "val": "Validation"}.get(split_name, split_name)
    _dir_gran = f"  |  {direction.upper()}  {granularity}" if direction or granularity else ""
    ax_rc.set_title(f"Profitability-Risk  |  {_split_display}  |  {_model_display}{_dir_gran}",
                    fontsize=13, fontweight="bold", color="#2C3E50", pad=12)

    # ┏━━━━━━━━━━ Single legend: 3 rows × 4 columns ━━━━━━━━━━┓
    # Stats (Prec, Cov, μ/t) are embedded as multi-line text in col-4 entries
    # so no phantom handle-space is wasted on invisible patches.
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    _prec_str = rf"Prec$_{{\hat{{τ}}}}$={op.get('precision', float('nan'))*100:.1f}%"
    _cov_str  = rf"Cov$_{{\hat{{τ}}}}$={op['coverage']*100:.1f}% (N={op.get('selected_count', 0)})"
    _mut_str  = rf"$\mu$={op['mean_ret']*100:+.2f}%, t={op.get('t_stat', 0):.2f}"
    handles = [
        # ── Column 1: Returns ─────────────────────────────────────────
        Line2D([], [], color=c_ret, linewidth=2.0, label="Mean Net Return"),
        Line2D([], [], color=c_ret, linewidth=1.0, linestyle=":", label="Mean Win Return"),
        Line2D([], [], color=c_ret_neg, linewidth=1.0, linestyle=":", label="Mean Loss Return"),
        # ── Column 2: Risk / Utility curves ───────────────────────────
        Line2D([], [], color=c_risk, linewidth=2.2, label="Risk-Coverage Curve"),
        Line2D([], [], color=c_util, linewidth=1.8, linestyle="--", label="Risk-Profitability Score"),
        Line2D([], [], color=c_baseline, linewidth=1.6, linestyle="-.", label=r"Risk Floor: $Risk_{M2@0.5}$"),
        # ── Column 3: Zones ───────────────────────────────────────────
        Patch(facecolor=c_forbid, alpha=0.30, hatch="//", edgecolor=c_forbid,
              label=r"Forbidden: $\mathrm{Cov} < C_{min}$"),
        Patch(facecolor=c_penalty, alpha=0.20, hatch="..", edgecolor=c_penalty,
              label=r"Quadratic Penalty: $\mathrm{Cov} < C^{*}$"),
        Patch(facecolor=c_win, alpha=0.20, edgecolor="none", label="Profitability Zone"),
        # ── Column 4: Operating points + stats inline ─────────────────
        Line2D([], [], color=c_util, marker="*", markersize=13, linestyle="None",
               markeredgecolor="white", markeredgewidth=1.0,
               label=rf"Max Risk-Prof Score       {_prec_str}"),
        Line2D([], [], color=c_op, marker="D", markersize=7, linestyle="--",
               markeredgecolor="white", markeredgewidth=0.8,
               label=rf"$\hat{{\tau}}$={op['threshold']:.3f} ({thr_source})     {_cov_str}"),
        Line2D([], [], color=c_thr05, marker="o", markersize=6, linestyle="--",
               markeredgecolor="white", markeredgewidth=0.8,
               label=rf"$\tau$=0.5 (Baseline)            {_mut_str}"),
    ]
    # Center the legend on the main axes span [left=0.08, right=0.92].
    _leg_cx = (0.08 + 0.92) / 2
    leg = fig_rc.legend(handles=handles, loc="lower center",
                        bbox_to_anchor=(_leg_cx, 0.01), ncol=4,
                        prop={"size": 8.5}, frameon=True, framealpha=0.95,
                        edgecolor="#BDC3C7", fancybox=True,
                        handlelength=2.2, handletextpad=0.6,
                        columnspacing=1.2, borderpad=0.6)
    leg.set_zorder(20)
    fig_rc.tight_layout()
    fig_rc.subplots_adjust(left=0.08, bottom=0.22, right=0.92, top=0.93)

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
# performance over number of features
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
            if os.path.exists(ana_path):
                try:
                    with open(ana_path) as f:
                        ana = json.load(f)
                    block = ana.get(M2_KEY.get(m2, ""), {})
                    val_sel  = block.get("Val_selective",  {}) or {}
                    test_sel = block.get("Test_selective", {}) or {}
                    val_mean_ret  = val_sel.get("mean_ret")
                    test_mean_ret = test_sel.get("mean_ret")
                except Exception:
                    pass

            # ┏━━━━━━━━━━ Append the performance metrics to the records list ━━━━━━━━━━┓
            records.append({"m1":              m1,
                            "m2":              m2,
                            "direction":       direction,
                            "gran":            gran,
                            "frac_profitable": float(frac_p),
                            "median_sharpe":   float(med_sr),
                            "val_mean_ret":    None if val_mean_ret is None else float(val_mean_ret),
                            "test_mean_ret":   None if test_mean_ret is None else float(test_mean_ret),
                            "path_sharpes":    sharpes.tolist()})
    return records


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CPCV Edge — Constraint-trigger bar plot
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def plot_cpcv_constraint_bars(edge_root: str = "/home/pablo/M2_DS/Secondary-Model/src/Output/Analysis/Edge_NoCal",
                              output_dir: str | None = None,
                              tau_fp: float = 0.6,
                              tau_sr: float = 0.0) -> str:
    """Bar plot: how often each constraint (and combination) triggers.

        C1 := frac_profitable    >= tau_fp   (default 0.6)
        C2 := median_path_sharpe >= tau_sr   (default 0.0)
        C3 := val_mean_ret       >  0

    7 bars (inclusive counts — each config can contribute to multiple bars):
        C1, C2, C3, C1∧C2, C1∧C3, C2∧C3, C1∧C2∧C3
    """
    import os
    import numpy as np
    import matplotlib.pyplot as plt

    # ┏━━━━━━━━━━ Convert edge_root and output_dir to absolute paths ━━━━━━━━━━┓
    edge_root = os.path.abspath(edge_root)
    if output_dir is None:
        output_dir = edge_root
    os.makedirs(output_dir, exist_ok=True)

    # ┏━━━━━━━━━━ Load CPCV records ━━━━━━━━━━┓
    records = _load_cpcv_records(edge_root)
    
    # ┏━━━━━━━━━━ Filter records with val_mean_ret ━━━━━━━━━━┓
    records = [r for r in records if r["val_mean_ret"] is not None]
    N = len(records)
    if N == 0:
        print(f"[plot_cpcv_constraint_bars] No records with val_mean_ret found")
        return ""

    # ┏━━━━━━━━━━ Extract feature arrays ━━━━━━━━━━┓
    fp = np.array([r["frac_profitable"] for r in records])
    sr = np.array([r["median_sharpe"]   for r in records])
    mr = np.array([r["val_mean_ret"]    for r in records])

    # ┏━━━━━━━━━━ Define constraints ━━━━━━━━━━┓
    c1 = fp >= tau_fp
    c2 = sr >= tau_sr
    c3 = mr >  0.0

    # ┏━━━━━━━━━━ Define bars, labels, include counts and percentages, for plot ━━━━━━━━━━┓
    bars = [("C1\n(frac_prof≥{:.1f})".format(tau_fp), int(c1.sum())),
            ("C2\n(med_SR≥{:.1f})".format(tau_sr),    int(c2.sum())),
            ("C3\n(val_mean_ret>0)",                  int(c3.sum())),
            ("C1∧C2",                                 int((c1 & c2).sum())),
            ("C1∧C3",                                 int((c1 & c3).sum())),
            ("C2∧C3",                                 int((c2 & c3).sum())),
            ("C1∧C2∧C3",                              int((c1 & c2 & c3).sum()))]
    
    labels  = [b[0] for b in bars]
    counts  = [b[1] for b in bars]
    pcts    = [100.0 * c / N for c in counts]

    # ┏━━━━━━━━━━ Set colors for bars (single-constraint bars: blue;  pairs: orange;  triple: green) ━━━━━━━━━━┓
    colours = ["#3182bd"] * 3 + ["#fd8d3c"] * 3 + ["#31a354"]

    # ┏━━━━━━━━━━ Create figure and axes, and draw bars ━━━━━━━━━━┓
    fig, ax = plt.subplots(figsize=(11, 5.5))
    fig.patch.set_facecolor("white")
    bars_obj = ax.bar(labels, counts, color=colours, edgecolor="black", linewidth=0.7)

    # ┏━━━━━━━━━━ Set text labels (counts + percentages) on top of each bar ━━━━━━━━━━┓
    for rect, c, p in zip(bars_obj, counts, pcts):
        ax.text(rect.get_x() + rect.get_width() / 2,
                rect.get_height() + max(counts) * 0.012,
                f"{c}\n({p:.1f}%)",
                ha="center", va="bottom", fontsize=9, fontweight="bold")

    # ┏━━━━━━━━━━ Set y-axis label, title, and limits ━━━━━━━━━━┓
    ax.set_ylabel(f"# configurations (out of {N})", fontsize=11, fontweight="bold")
    ax.set_title(f"CPCV constraint-trigger frequency  |  {N} configurations\n"
                 f"C1: frac_profitable ≥ {tau_fp:.2f}    "
                 f"C2: median_path_sharpe ≥ {tau_sr:.2f}    "
                 f"C3: val_mean_ret > 0",
                 fontsize   = 12, 
                 fontweight = "bold", 
                 pad        = 10)
    
    ax.set_ylim(0, max(counts) * 1.18)
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    ax.set_axisbelow(True)
    plt.xticks(fontsize=9)
    plt.tight_layout()

    # ┏━━━━━━━━━━ Save figure ━━━━━━━━━━┓
    out_path = os.path.join(output_dir, "cpcv_constraint_bars.png")
    plt.savefig(out_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"[plot_cpcv_constraint_bars] {N} configs -> {out_path}")
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

        # ┏━━━━━━━━━━ Two side-by-side subplots ━━━━━━━━━━┓
        fig, (ax_acc, ax_prec) = plt.subplots(1, 2, figsize=(max(20, 1.9 * n_x + 6), max(6, 0.85 * n_y + 2)))
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


if __name__ == "__main__":
    plot_cpcv_constraint_bars()
    plot_cpcv_edge_heatmap()
