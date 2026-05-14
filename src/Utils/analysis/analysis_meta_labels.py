"""Auto-split from Utils/feature_selection/plots.py during 2026-05-14 cleanup."""

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
from Utils.utils import model_label
from typing import Dict, List, Optional, Any, Union, Tuple
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from Utils.data import _get_from_dataset, get_dynamic_ret_limits, MultiGranDataset
from sklearn.metrics import (accuracy_score, f1_score, precision_score, recall_score,
                             precision_recall_fscore_support, confusion_matrix,
                             ConfusionMatrixDisplay, fbeta_score, matthews_corrcoef)


# ┏━━━━━━━━━━ Plot class distributions ━━━━━━━━━━┓
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


# ┏━━━━━━━━━━ Plot M1 Meta-Label Returns Histogram ━━━━━━━━━━┓
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


# ┏━━━━━━━━━━ Histograms of Euclidean Distance TP UP vs TP DOWN ━━━━━━━━━━┓
def plot_up_down_meta_label_distance_histograms(up_cache_path: Union[str, Path],
                                                down_cache_path: Union[str, Path],
                                                save_dir: Union[str, Path],
                                                train_end: str = "2025-05-30",
                                                val_end: str = "2025-10-01",
                                                granularities: Optional[List[str]] = None,
                                                max_pairs: int = 250_000,
                                                random_state: int = 42,
                                                bins: int = 80,
                                                x_quantile: float = 0.75,
                                                standardize: bool = True) -> pd.DataFrame:
    """Plot UP-vs-DOWN label-1 feature-vector Euclidean distances by split.

    The output has one figure per granularity, with Train, Val, and Test
    histograms side by side. Distances are computed from sampled UP-positive x
    DOWN-positive pairs within the same granularity/split. This estimates the
    feature-space overlap between TP UP and TP DOWN windows without
    materializing the full Cartesian product for fine granularities.
    """
    import matplotlib.pyplot as plt

    up_cache_path = Path(up_cache_path)
    down_cache_path = Path(down_cache_path)
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    up_multi = torch.load(up_cache_path, weights_only=False, map_location="cpu")
    down_multi = torch.load(down_cache_path, weights_only=False, map_location="cpu")
    if not hasattr(up_multi, "sub") or not hasattr(down_multi, "sub"):
        raise ValueError("Both cache files must be multi-granularity caches with a .sub attribute.")

    present = [g for g in getattr(up_multi, "grans", []) if g in getattr(down_multi, "grans", [])]
    if granularities is None:
        granularities = present
    else:
        missing = [g for g in granularities if g not in present]
        if missing:
            raise ValueError(f"Granularities not present in both caches: {missing}. Present: {present}")

    train_cut = pd.Timestamp(train_end)
    val_cut = pd.Timestamp(val_end)
    if not 0.0 < x_quantile <= 1.0:
        raise ValueError(f"x_quantile must be in (0, 1], got {x_quantile}")

    rng = np.random.default_rng(random_state)
    summary_rows = []

    def _np(x):
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
        return np.asarray(x)

    def _split_mask(dates, split_name):
        ts = pd.to_datetime(pd.Series(dates))
        if split_name == "Train":
            return (ts <= train_cut).to_numpy()
        if split_name == "Val":
            return ((ts > train_cut) & (ts <= val_cut)).to_numpy()
        return (ts > val_cut).to_numpy()

    def _positive_vectors(sub, split_name):
        x = _np(sub["eng_features"]).astype(float, copy=False)
        y = _np(sub["labels"]).astype(float, copy=False)
        dates = list(sub["dates"])
        mask = _split_mask(dates, split_name) & np.isfinite(y) & (y == 1)
        mask &= np.isfinite(x).all(axis=1)
        idx = np.where(mask)[0]
        return x[idx]

    def _maybe_standardize(up_x, down_x):
        if not standardize or len(up_x) == 0 or len(down_x) == 0:
            return up_x, down_x
        both = np.vstack([up_x, down_x])
        mu = np.nanmean(both, axis=0)
        sigma = np.nanstd(both, axis=0)
        sigma[sigma == 0] = 1.0
        return (up_x - mu) / sigma, (down_x - mu) / sigma

    def _sampled_pair_distances(up_x, down_x):
        up_x, down_x = _maybe_standardize(up_x, down_x)
        n_up, n_down = len(up_x), len(down_x)
        if n_up == 0 or n_down == 0:
            return np.array([], dtype=float), 0, False

        total_pairs = n_up * n_down
        sampled = total_pairs > max_pairs
        n_pairs = min(total_pairs, max_pairs)
        if sampled:
            ui = rng.integers(0, n_up, size=n_pairs)
            di = rng.integers(0, n_down, size=n_pairs)
            return np.linalg.norm(up_x[ui] - down_x[di], axis=1), total_pairs, True

        chunks = []
        chunk_size = max(1, max_pairs // max(n_down, 1))
        for start in range(0, n_up, chunk_size):
            diff = up_x[start:start + chunk_size, None, :] - down_x[None, :, :]
            chunks.append(np.linalg.norm(diff, axis=2).ravel())
        return np.concatenate(chunks), total_pairs, False

    for gran in granularities:
        split_payload = []
        for split_name in ["Train", "Val", "Test"]:
            up_x = _positive_vectors(up_multi.sub[gran], split_name)
            down_x = _positive_vectors(down_multi.sub[gran], split_name)
            dists, possible_pairs, sampled = _sampled_pair_distances(up_x, down_x)
            split_payload.append((split_name, up_x, down_x, dists, possible_pairs, sampled))

        finite_for_xlim = [d for _, _, _, d, _, _ in split_payload if len(d) > 0]
        if finite_for_xlim:
            all_distances = np.concatenate(finite_for_xlim)
            display_upper = float(np.quantile(all_distances, x_quantile))
            if not np.isfinite(display_upper) or display_upper <= 0:
                display_upper = float(np.max(all_distances))
            if not np.isfinite(display_upper) or display_upper <= 0:
                display_upper = 1.0
        else:
            display_upper = 1.0

        hist_bins = np.linspace(0, display_upper, bins + 1)
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle(
            f"UP vs DOWN Meta-Label=1 Feature Distances | {gran} | sampled pairs | x <= p{int(round(x_quantile * 100))}",
            fontsize=15,
            fontweight="bold")

        for ax, (split_name, up_x, down_x, dists, possible_pairs, sampled) in zip(axes, split_payload):
            row = {"granularity": gran,
                   "split": split_name,
                   "pair_mode": "sampled_all_pairs",
                   "n_up_label_1": int(len(up_x)),
                   "n_down_label_1": int(len(down_x)),
                   "possible_pairs": int(possible_pairs),
                   "n_distances_plotted": int(len(dists)),
                   "sampled": bool(sampled),
                   "x_quantile": float(x_quantile),
                   "x_display_upper": float(display_upper),
                   "standardize": bool(standardize)}

            if len(dists) > 0:
                shown = dists[dists <= display_upper]
                n_clipped = int(len(dists) - len(shown))
                ax.hist(shown, bins=hist_bins, color="#2563eb", alpha=0.72, edgecolor="white", linewidth=0.35)
                mean = float(np.mean(dists))
                median = float(np.median(dists))
                p05, p95 = np.percentile(dists, [5, 95])
                if mean <= display_upper:
                    ax.axvline(mean, color="#dc2626", linewidth=1.5, label=f"Mean {mean:.3f}")
                if median <= display_upper:
                    ax.axvline(median, color="#111827", linestyle="--", linewidth=1.2, label=f"Median {median:.3f}")
                if n_clipped:
                    ax.text(0.98, 0.88, f">{display_upper:.2f}: {n_clipped:,}",
                            ha="right", va="top", transform=ax.transAxes, fontsize=8,
                            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#9ca3af", alpha=0.85, lw=0.6))
                handles, labels = ax.get_legend_handles_labels()
                if handles:
                    ax.legend(handles, labels, loc="upper right", fontsize=8)
                ax.set_xlim(0, display_upper)
                row.update({"distance_mean": mean,
                            "distance_median": median,
                            "distance_std": float(np.std(dists)),
                            "distance_min": float(np.min(dists)),
                            "distance_p05": float(p05),
                            "distance_p95": float(p95),
                            "distance_max": float(np.max(dists)),
                            "n_distances_shown": int(len(shown)),
                            "n_distances_clipped": n_clipped})
            else:
                ax.text(0.5, 0.5, "No comparable pairs", ha="center", va="center", transform=ax.transAxes)
                row.update({"distance_mean": np.nan,
                            "distance_median": np.nan,
                            "distance_std": np.nan,
                            "distance_min": np.nan,
                            "distance_p05": np.nan,
                            "distance_p95": np.nan,
                            "distance_max": np.nan,
                            "n_distances_shown": 0,
                            "n_distances_clipped": 0})

            sample_note = "sampled" if sampled else "exact"
            ax.set_title(
                f"{split_name}\nUP={len(up_x):,} DOWN={len(down_x):,} pairs={len(dists):,} ({sample_note})",
                fontsize=10)
            ax.set_xlabel("Euclidean distance")
            ax.set_ylabel("Count")
            ax.grid(True, alpha=0.25)
            summary_rows.append(row)

        fig.tight_layout(rect=[0, 0, 1, 0.92])
        suffix = "_zscore" if standardize else ""
        out_path = save_dir / f"up_down_meta_label_1_distance_{gran}_sampled_all_pairs{suffix}.png"
        fig.savefig(out_path, bbox_inches="tight", dpi=160)
        plt.close(fig)
        print(f"[plot_up_down_meta_label_distance_histograms] Saved -> {out_path}")

    summary = pd.DataFrame(summary_rows)
    summary_path = save_dir / f"up_down_meta_label_distance_summary_sampled_all_pairs{'_zscore' if standardize else ''}.csv"
    summary.to_csv(summary_path, index=False)
    print(f"[plot_up_down_meta_label_distance_histograms] Summary -> {summary_path}")
    return summary


# ┏━━━━━━━━━━ Plot Asset Price Correlation Matrix of Top 20 Crypto-Assets ━━━━━━━━━━┓
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


# ┏━━━━━━━━━━ Meta-Label Dataset size & TP/FP distribution across M1 models × granularities × directions ━━━━━━━━━━┓
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
                                          x granularities x directions.
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


# ┏━━━━━━━━━━ Return quality of M1 Meta-labels: TP vs FP return distributions per model × gran × direction ━━━━━━━━━━┓
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

    # ┏━━━━━━━━━━ Export summary JSON ━━━━━━━━━━┓
    import json
    summary: dict = {}
    for m1 in df["m1_model"].unique():
        summary[m1] = {}
        for gran in df["granularity"].unique():
            summary[m1][gran] = {}
            for direction in df["direction"].unique():
                summary[m1][gran][direction] = {}
                for label in ("TP", "FP"):
                    seg = df[
                        (df["m1_model"] == m1) &
                        (df["granularity"] == gran) &
                        (df["direction"] == direction) &
                        (df["label"] == label)
                    ]["return"]
                    if len(seg) == 0:
                        summary[m1][gran][direction][label] = None
                        continue
                    summary[m1][gran][direction][label] = {
                        "n":      int(len(seg)),
                        "median": round(float(np.median(seg)) * 100, 4),
                        "mean":   round(float(np.mean(seg)) * 100, 4),
                        "q25":    round(float(np.percentile(seg, 25)) * 100, 4),
                        "q75":    round(float(np.percentile(seg, 75)) * 100, 4),
                        "std":    round(float(np.std(seg)) * 100, 4),
                    }

    # Aggregate across all granularities per (model, direction, label)
    agg: dict = {"by_model_direction": {}}
    for m1 in df["m1_model"].unique():
        agg["by_model_direction"][m1] = {}
        for direction in df["direction"].unique():
            agg["by_model_direction"][m1][direction] = {}
            for label in ("TP", "FP"):
                seg = df[
                    (df["m1_model"] == m1) &
                    (df["direction"] == direction) &
                    (df["label"] == label)
                ]["return"]
                if len(seg) == 0:
                    agg["by_model_direction"][m1][direction][label] = None
                    continue
                agg["by_model_direction"][m1][direction][label] = {
                    "n":      int(len(seg)),
                    "median": round(float(np.median(seg)) * 100, 4),
                    "mean":   round(float(np.mean(seg)) * 100, 4),
                    "q25":    round(float(np.percentile(seg, 25)) * 100, 4),
                    "q75":    round(float(np.percentile(seg, 75)) * 100, 4),
                    "std":    round(float(np.std(seg)) * 100, 4),
                }

    full_json = {"per_model_gran_dir": summary, "aggregate": agg}
    json_path = save_dir / "return_quality_distribution.json"
    with open(json_path, "w") as jf:
        json.dump(full_json, jf, indent=2)
    print(f"[quality-plot] JSON saved → {json_path}")

    return df


# ┏━━━━━━━━━━ M1 Meta-labels Returns over Directions (All grans aggregated) ━━━━━━━━━━┓
def plot_aggregate_return_quality_violins(data: dict, out_dir: Path):
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd

    agg_data = data.get("aggregate", {}).get("by_model_direction", {})
    MAX_SAMPLES = 5000
    records = []
    models = ["Kronos", "Fincast", "Chronos2", "Tirex"]

    for model in models:
        if model not in agg_data:
            continue
        for direction in ["UP", "DOWN"]:
            for pred_type in ["TP", "FP"]:
                stats = agg_data[model][direction][pred_type]
                n = min(stats["n"], MAX_SAMPLES)
                mean = stats["mean"]
                std = stats["std"]
                
                np.random.seed(42 + hash(model) % 1000)
                samples = np.random.normal(loc=mean, scale=std, size=n)
                
                for val in samples:
                    records.append({
                        "Model": model,
                        "Direction": direction,
                        "Prediction": pred_type,
                        "Return (%)": val
                    })

    if not records:
        return
    df = pd.DataFrame(records)

    fig, axes = plt.subplots(2, 1, figsize=(14, 12), sharey=True)
    for i, direction in enumerate(["UP", "DOWN"]):
        ax = axes[i]
        subset = df[df["Direction"] == direction]
        
        sns.violinplot(
            data=subset, 
            x="Model", 
            y="Return (%)", 
            hue="Prediction",
            split=True,
            inner="quartile",
            ax=ax,
            palette={"TP": "#2ecc71", "FP": "#e74c3c"}
        )
        
        ax.set_title(f"{direction} Trades - Aggregate Return Quality Distribution", fontsize=14, fontweight='bold')
        ax.set_xlabel("M1 Model", fontsize=12)
        ax.set_ylabel("Return (%)", fontsize=12)
        ax.grid(axis='y', alpha=0.3)
        ax.legend(title="Prediction Type", loc="upper left")

    plt.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "aggregate_return_quality_violins.png"
    fig.savefig(out_file, dpi=300, bbox_inches='tight')
    plt.close(fig)


# ┏━━━━━━━━━━ Aggregate Return Quality Boxplots (All models aggregated over granularities) ━━━━━━━━━━┓
def plot_aggregate_all_models(data: dict, out_dir: Path):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    agg_data = data.get("aggregate", {}).get("by_model_direction", {})
    models = ["Kronos", "Fincast", "Chronos2", "Tirex"]

    fig, axes = plt.subplots(2, 1, figsize=(16, 12), sharey=True)
    
    for i, direction in enumerate(["UP", "DOWN"]):
        ax = axes[i]
        box_stats = []
        positions = []
        x_pos = 1
        
        for model in models:
            if model not in agg_data:
                continue
                
            for pred_type in ["TP", "FP"]:
                stats = agg_data[model][direction][pred_type]
                q1 = stats["q25"]
                med = stats["median"]
                q3 = stats["q75"]
                mean = stats["mean"]
                
                iqr = q3 - q1
                whislo = q1 - 1.5 * iqr
                whishi = q3 + 1.5 * iqr
                
                box_stats.append({
                    'label': f"{model}\n({pred_type})",
                    'mean': mean,
                    'med': med,
                    'q1': q1,
                    'q3': q3,
                    'whislo': whislo,
                    'whishi': whishi,
                    'fliers': [],
                })
                positions.append(x_pos)
                x_pos += 1
            x_pos += 0.5

        if not box_stats:
            continue

        bplot = ax.bxp(box_stats, positions=positions, showmeans=True, meanline=True, 
                       patch_artist=True, widths=0.6, showfliers=False)
        
        for j, patch in enumerate(bplot['boxes']):
            pred_type = "TP" if j % 2 == 0 else "FP"
            color = "#2ecc71" if pred_type == "TP" else "#e74c3c"
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
            patch.set_edgecolor('black')
            
        for median_line in bplot['medians']:
            median_line.set_color('black')
            median_line.set_linewidth(2)
            
        for mean_line in bplot['means']:
            mean_line.set_color('blue')
            mean_line.set_linewidth(2)
            mean_line.set_linestyle('--')
            
        for j, stat in enumerate(box_stats):
            pos = positions[j]
            ax.text(pos + 0.35, stat['med'], f"Med: {stat['med']:.2f}%", 
                    ha='left', va='center', size=10, color='black', weight='bold',
                    bbox=dict(facecolor='white', alpha=0.5, edgecolor='none', pad=1))
            ax.text(pos + 0.35, stat['mean'], f"μ: {stat['mean']:.2f}%", 
                    ha='left', va='center', size=10, color='blue', weight='bold',
                    bbox=dict(facecolor='white', alpha=0.5, edgecolor='none', pad=1))
                    
        ax.set_title(f"{direction} Trades - Aggregate Return Distribution", fontsize=14, fontweight='bold')
        ax.set_ylabel("Return (%)", fontsize=12)
        ax.grid(axis='y', alpha=0.3)
        
        legend_elements = [
            Patch(facecolor='#2ecc71', alpha=0.6, edgecolor='black', label='True Positives (TP)'),
            Patch(facecolor='#e74c3c', alpha=0.6, edgecolor='black', label='False Positives (FP)'),
            plt.Line2D([0], [0], color='black', lw=2, label='Median'),
            plt.Line2D([0], [0], color='blue', lw=2, linestyle='--', label='Mean')
        ]
        ax.legend(handles=legend_elements, loc='upper left', ncol=4, framealpha=0.8)

        y_max = ax.get_ylim()[1]
        x_pos_idx = 0
        for model in models:
            if model not in agg_data:
                continue
            tp_mean = agg_data[model][direction]["TP"]["mean"]
            fp_mean = agg_data[model][direction]["FP"]["mean"]
            tp_std = agg_data[model][direction]["TP"]["std"]
            fp_std = agg_data[model][direction]["FP"]["std"]
            tp_n = agg_data[model][direction]["TP"]["n"]
            fp_n = agg_data[model][direction]["FP"]["n"]
            
            p_win = tp_n / (tp_n + fp_n) if (tp_n + fp_n) > 0 else 0
            net_reward = max(0, abs(tp_mean) - 0.20)
            net_risk = abs(fp_mean) + 0.20
            rr = net_reward / net_risk if net_risk != 0 else 0
            
            ev = (p_win * net_reward) - ((1 - p_win) * net_risk)
            e_x2_tp = (tp_std ** 2) + (net_reward ** 2)
            e_x2_fp = (fp_std ** 2) + ((-net_risk) ** 2)
            e_x2 = (p_win * e_x2_tp) + ((1 - p_win) * e_x2_fp)
            total_var = max(0, e_x2 - (ev ** 2))
            cagr = ev - (total_var / 200)
            
            center_pos = (positions[x_pos_idx] + positions[x_pos_idx+1]) / 2
            y_frac = 0.85 if direction == "UP" else 0.75
            ax.text(center_pos, y_max * y_frac, f"Reward/Risk: {rr:.2f}",
                    ha='center', va='top', size=9, color='purple', weight='bold',
                    bbox=dict(facecolor='white', alpha=0.8, edgecolor='purple', boxstyle='round,pad=0.3'))
            x_pos_idx += 2

    plt.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "aggregate_return_quality_boxplot.png"
    fig.savefig(out_file, dpi=300, bbox_inches='tight')
    plt.close(fig)


# ┏━━━━━━━━━━ Granularity Return Quality Boxplots (All granularities analysis) ━━━━━━━━━━┓
def plot_all_models_granularities(data: dict, out_dir: Path):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    if "per_model_gran_dir" not in data:
        return
        
    models_data = data["per_model_gran_dir"]
    models = ["Kronos", "Fincast", "Chronos2", "Tirex"]
    grans = ["1d", "12h", "8h", "6h", "4h", "2h", "1h", "30m"]
    
    fig, axes = plt.subplots(2, 1, figsize=(40, 16), sharey=True)
    
    for i, direction in enumerate(["UP", "DOWN"]):
        ax = axes[i]
        box_stats = []
        positions = []
        x_pos = 1
        
        for gran in grans:
            for model in models:
                if model not in models_data or gran not in models_data[model] or direction not in models_data[model][gran]:
                    continue
                    
                for pred_type in ["TP", "FP"]:
                    stats = models_data[model][gran][direction][pred_type]
                    q1 = stats["q25"]
                    med = stats["median"]
                    q3 = stats["q75"]
                    mean = stats["mean"]
                    
                    iqr = q3 - q1
                    whislo = q1 - 1.5 * iqr
                    whishi = q3 + 1.5 * iqr
                    
                    box_stats.append({
                        'label': f"{gran}\n{model}\n({pred_type})",
                        'mean': mean,
                        'med': med,
                        'q1': q1,
                        'q3': q3,
                        'whislo': whislo,
                        'whishi': whishi,
                        'fliers': [],
                    })
                    positions.append(x_pos)
                    x_pos += 1
                x_pos += 0.3
            x_pos += 1.2

        if not box_stats:
            continue

        bplot = ax.bxp(box_stats, positions=positions, showmeans=True, meanline=True, 
                       patch_artist=True, widths=0.5, showfliers=False)
        
        for j, patch in enumerate(bplot['boxes']):
            pred_type = "TP" if j % 2 == 0 else "FP"
            color = "#2ecc71" if pred_type == "TP" else "#e74c3c"
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
            patch.set_edgecolor('black')
            
        for median_line in bplot['medians']:
            median_line.set_color('black')
            median_line.set_linewidth(2)
            
        for mean_line in bplot['means']:
            mean_line.set_color('blue')
            mean_line.set_linewidth(2)
            mean_line.set_linestyle('--')
            
        for j, stat in enumerate(box_stats):
            pos = positions[j]
            ax.text(pos + 0.28, stat['med'], f"M: {stat['med']:.1f}%", 
                    ha='left', va='center', size=7, color='black', weight='bold',
                    bbox=dict(facecolor='white', alpha=0.5, edgecolor='none', pad=0.5))
            ax.text(pos + 0.28, stat['mean'], f"μ: {stat['mean']:.1f}%", 
                    ha='left', va='center', size=7, color='blue', weight='bold',
                    bbox=dict(facecolor='white', alpha=0.5, edgecolor='none', pad=0.5))
                    
        ax.set_title(f"All Models {direction} Trades - Return Distribution by Granularity", fontsize=16, fontweight='bold')
        ax.set_ylabel("Return (%)", fontsize=14)
        ax.grid(axis='y', alpha=0.3)
        ax.tick_params(axis='x', labelsize=8)
        
        legend_elements = [
            Patch(facecolor='#2ecc71', alpha=0.6, edgecolor='black', label='True Positives (TP)'),
            Patch(facecolor='#e74c3c', alpha=0.6, edgecolor='black', label='False Positives (FP)'),
            plt.Line2D([0], [0], color='black', lw=2, label='Median'),
            plt.Line2D([0], [0], color='blue', lw=2, linestyle='--', label='Mean')
        ]
        ax.legend(handles=legend_elements, loc='upper left', fontsize=12)

        y_max = ax.get_ylim()[1]
        x_pos_idx = 0
        for gran in grans:
            for model in models:
                if model not in models_data or gran not in models_data[model] or direction not in models_data[model][gran]:
                    continue
                
                tp_mean = models_data[model][gran][direction]["TP"]["mean"]
                fp_mean = models_data[model][gran][direction]["FP"]["mean"]
                tp_std = models_data[model][gran][direction]["TP"]["std"]
                fp_std = models_data[model][gran][direction]["FP"]["std"]
                tp_n = models_data[model][gran][direction]["TP"]["n"]
                fp_n = models_data[model][gran][direction]["FP"]["n"]
                
                p_win = tp_n / (tp_n + fp_n) if (tp_n + fp_n) > 0 else 0
                net_reward = max(0, abs(tp_mean) - 0.20)
                net_risk = abs(fp_mean) + 0.20
                rr = net_reward / net_risk if net_risk != 0 else 0
                
                ev = (p_win * net_reward) - ((1 - p_win) * net_risk)
                e_x2_tp = (tp_std ** 2) + (net_reward ** 2)
                e_x2_fp = (fp_std ** 2) + ((-net_risk) ** 2)
                e_x2 = (p_win * e_x2_tp) + ((1 - p_win) * e_x2_fp)
                total_var = max(0, e_x2 - (ev ** 2))
                cagr = ev - (total_var / 200)
                
                center_pos = (positions[x_pos_idx] + positions[x_pos_idx+1]) / 2
                ax.text(center_pos, y_max * 0.85, f"R/R: {rr:.2f}\nEV: {ev:.2f}\nC: {cagr:.1f}", 
                        ha='center', va='top', size=5, color='purple', weight='bold',
                        bbox=dict(facecolor='white', alpha=0.8, edgecolor='purple', boxstyle='round,pad=0.2', linewidth=0.5))
                x_pos_idx += 2

    plt.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "all_models_granularities_return_boxplot.png"
    fig.savefig(out_file, dpi=300, bbox_inches='tight')
    plt.close(fig)


# ┏━━━━━━━━━━ Plot Prediction Returns Histogram (M1 model preds vs ground truth) ━━━━━━━━━━┓
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

