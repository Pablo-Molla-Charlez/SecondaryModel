"""
Edge Analysis — Model Stability & Regime Sensitivity
=====================================================
Two modes:
  --mode seeds  : 100 trials with different seeds on a static train/val/test split
  --mode cpcv   : Combinatorial Purged Cross-Validation (Lopez de Prado)
                  N=6 datetime blocks, k=2 test → C(6,2)=15 splits → 5 paths

Supported models: rf, xgboost, autogluon, tabpfn, tabpfn_ft

Usage:
  python -m Utils.edge --cache path/to/multi_cache.pt --mode seeds
  python -m Utils.edge --cache path/to/multi_cache.pt --mode seeds --model xgboost
  python -m Utils.edge --cache path/to/multi_cache.pt --mode cpcv
  python -m Utils.edge --cache path/to/multi_cache.pt --mode cpcv --model tabpfn_ft
  python -m Utils.edge --cache path/to/multi_cache.pt --mode cpcv --n-blocks 8 --k-test 2
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

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

# ┏━━━━━━━━━ Fixed seed for CPCV — variance measures regime sensitivity, not model noise ━━━━━━━━━━┓
EDGE_SEED = 42

# ┏━━━━━━━━━━ Calibration split ratios ━━━━━━━━━━┓
# [XGBoost, TabpPFN or others] 
# First 40% of Val (Calibration Set ~ Val-Cal)
# Last 60% (Threshold Optimization Set ~ Val-Opt)
CAL_SPLIT_RATIO = 0.40

# [Random Forest] 
# First 40% of OOB Train (Calibration Set ~ Val-Cal)
# Last 60% (Threshold Optimization Set ~ Val-Opt) (CPCV only)
CPCV_OOB_CAL_RATIO = 0.40

# ┏━━━━━━━━━━ Metrics to plot ━━━━━━━━━━┓
METRICS_TO_PLOT = [("accuracy",      "Accuracy (@0.5)"),
                   ("sel_accuracy",  "Selective Accuracy"),
                   ("precision",     "Precision (@0.5)"),
                   ("sel_precision", "Selective Precision"),
                   ("mean_ret",      "Mean Ret (@0.5)"),
                   ("sel_mean_ret",  "Selective Mean Ret"),
                   ("sel_coverage",  "Selective Coverage")]

# ┏━━━━━━━━━━ Result-grid constants (M1 × M2 × dir × gran) ━━━━━━━━━━┓
_M1_LIST    = ["Tirex", "Chronos2", "Fincast", "Kronos"]
_M2_LIST    = ["rf", "autogluon", "tabpfn", "tabicl", "ctts"]
_DIRS_LIST  = ["UP", "DOWN"]
_GRANS_LIST = ["1d", "12h", "8h", "6h", "4h", "2h", "1h", "30m"]

if __name__ == "__main__":
    main()


# Re-export ALL names (including _private helpers) from .edge
from . import edge as _topic_mod  # noqa: F401
globals().update({k: v for k, v in vars(_topic_mod).items() if not k.startswith('__')})



# ┏━━━━━━━━━━ Plot edge curves ━━━━━━━━━━┓
def _plot_edge_curves(all_trials, split_name, save_path, gran, direction, n_trials, m1_baselines=None):
    # ┏━━━━━━━━━━ Extract metrics data ━━━━━━━━━━┓
    metrics_data = {}
    for m_key, _ in METRICS_TO_PLOT:
        metrics_data[m_key] = np.array([t[split_name][m_key] for t in all_trials])

    # ┏━━━━━━━━━━ Extract M1 baselines ━━━━━━━━━━┓
    m1_split = m1_baselines.get(split_name, {}) if m1_baselines else {}
    m1_baseline_for = {}
    if m1_split.get("m1_acc") is not None:
        m1_baseline_for["accuracy"]  = ("M1 Acc",  m1_split["m1_acc"])
    if m1_split.get("m1_prec") is not None:
        m1_baseline_for["precision"]     = ("M1 Prec", m1_split["m1_prec"])
        m1_baseline_for["sel_precision"] = ("M1 Prec", m1_split["m1_prec"])
    if m1_split.get("m1_mean_ret") is not None:
        m1_baseline_for["sel_mean_ret"] = ("M1 Mean Ret (all trades)", m1_split["m1_mean_ret"])

    # ┏━━━━━━━━━━ Create plots ━━━━━━━━━━┓
    fig, axes = plt.subplots(len(METRICS_TO_PLOT), 1, figsize=(10, 3.2 * len(METRICS_TO_PLOT)))
    if len(METRICS_TO_PLOT) == 1:
        axes = [axes]

    # ┏━━━━━━━━━━ Plot each metric ━━━━━━━━━━┓
    for ax, (m_key, m_label) in zip(axes, METRICS_TO_PLOT):
        # ┏━━━━━━━━━━ Extract metrics data ━━━━━━━━━━┓
        vals = metrics_data[m_key]
        mean, std, median = np.mean(vals), np.std(vals), np.median(vals)

        # ┏━━━━━━━━━━ Plot each metric ━━━━━━━━━━┓
        ax.hist(vals, bins=min(30, n_trials // 3 + 1), alpha=0.5, color="steelblue", edgecolor="white", density=True, label="Trial distribution")
        ax.axvline(mean, color="navy", lw=2, label=f"M2 Mean: {mean:.4f} (Solid)")
        ax.axvspan(mean - std, mean + std, alpha=0.15, color="navy", label=f"M2 ±1σ: {std:.4f} (Shaded)")
        ax.axvline(median, color="purple", lw=1.5, ls="--", label=f"M2 Median: {median:.4f} (Dashed)")

        # ┏━━━━━━━━━━ Plot M1 baselines ━━━━━━━━━━┓
        if m_key in m1_baseline_for:
            bl_label, bl_val = m1_baseline_for[m_key]
            ax.axvline(bl_val, color="red", lw=2, ls=":", label=f"{bl_label}: {bl_val:.4f} (Dotted)")

        # ┏━━━━━━━━━━ Set labels and title ━━━━━━━━━━┓
        ax.set_xlabel(m_label)
        ax.set_ylabel("Density")
        ax.legend(fontsize=9, loc="upper right")
        ax.set_title(f"{m_label} ({split_name.upper()})")

    # ┏━━━━━━━━━━ Set title and save ━━━━━━━━━━┓
    fig.suptitle(f"Edge Analysis — {gran} {direction.upper()} | {split_name.upper()} | {n_trials} trials", fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)



# ┏━━━━━━━━━━ Plot summary boxplots (test only — no val in seed trials) ━━━━━━━━━━┓
def _plot_summary_boxplots(all_trials, save_path, gran, direction, n_trials, m1_baselines=None):
    # ┏━━━━━━━━━━ Define key metrics and labels ━━━━━━━━━━┓
    key_metrics = ["accuracy", "sel_accuracy", "precision", "sel_precision", "mean_ret", "sel_mean_ret", "sel_coverage"]
    labels      = ["Accuracy (@0.5)", "Selective Accuracy", "Precision (@0.5)", "Selective Precision", "Mean Ret (@0.5)", "Selective Mean Ret", "Selective Coverage"]

    TEST_COLOR = "royalblue"

    # ┏━━━━━━━━━━ Extract M1 baselines ━━━━━━━━━━┓
    m1_map = {}
    if m1_baselines:
        m1_test = m1_baselines.get("test", {})
        if m1_test.get("m1_acc")  is not None:
            m1_map["accuracy"]      = m1_test["m1_acc"]
            m1_map["sel_accuracy"]  = m1_test["m1_acc"]
        if m1_test.get("m1_prec") is not None:
            m1_map["precision"]     = m1_test["m1_prec"]
            m1_map["sel_precision"] = m1_test["m1_prec"]
        if m1_test.get("m1_mean_ret") is not None:
            m1_map["mean_ret"]      = m1_test["m1_mean_ret"]
            m1_map["sel_mean_ret"]  = m1_test["m1_mean_ret"]

    # ┏━━━━━━━━━━ Create boxplots ━━━━━━━━━━┓
    fig, axes = plt.subplots(1, len(key_metrics), figsize=(3.2 * len(key_metrics), 5))
  
    # ┏━━━━━━━━━━ Plot each metric ━━━━━━━━━━┓
    for ax, m_key, label in zip(axes, key_metrics, labels):
        test_vals = [t["test"][m_key] for t in all_trials]
        bp = ax.boxplot([test_vals], labels=["Test"], patch_artist=True, widths=0.5)
        bp["boxes"][0].set_facecolor(TEST_COLOR); bp["boxes"][0].set_alpha(0.5)

        if m_key in m1_map:
            ax.axhline(m1_map[m_key], color="red", lw=2, ls="--", label=f"M1={m1_map[m_key]:.4f}")
            ax.legend(fontsize=7, loc="upper right")

        ax.set_title(label, fontsize=10)
        ax.grid(axis="y", alpha=0.3)

    # ┏━━━━━━━━━━ Set suptitle and save ━━━━━━━━━━┓
    fig.suptitle(f"Edge Summary — {gran} {direction.upper()} | {n_trials} trials", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)



# ┏━━━━━━━━━━ Plot cross-granularity seeds ━━━━━━━━━━┓
def _plot_cross_gran_seeds(summary, save_path, direction, n_trials):
    # ┏━━━━━━━━━━ Extract granularities ━━━━━━━━━━┓
    grans = list(summary.keys())
    
    # ┏━━━━━━━━━━ Create figure and axes ━━━━━━━━━━┓
    fig, axes = plt.subplots(2, 2, figsize=(max(12, len(grans) * 3), 10))
    axes = axes.flatten()

    # ┏━━━━━━━━━━ Define metrics to plot ━━━━━━━━━━┓
    metrics_to_plot = [("val_acc_mean", "test_acc_mean", "val_acc_std", "test_acc_std",
                        "m1_baseline_val_acc", "m1_baseline_test_acc", "Accuracy (@0.5)"),
                        ("val_prec_mean", "test_prec_mean", "val_prec_std", "test_prec_std",
                        "m1_baseline_val_prec", "m1_baseline_test_prec", "Precision (@0.5)"),
                        ("val_sel_acc_mean", "test_sel_acc_mean", "val_sel_acc_std", "test_sel_acc_std",
                        "m1_baseline_val_acc", "m1_baseline_test_acc", "Selective Accuracy"),
                        ("val_sel_prec_mean", "test_sel_prec_mean", "val_sel_prec_std", "test_sel_prec_std",
                        "m1_baseline_val_prec", "m1_baseline_test_prec", "Selective Precision")]

    # ┏━━━━━━━━━━ Plot each metric ━━━━━━━━━━┓
    for ax, metric in zip(axes, metrics_to_plot):
        val_k, test_k, val_std_k, test_std_k, m1_val_k, m1_test_k, label = metric
        val_means  = [summary[g].get(val_k, 0) for g in grans]
        test_means = [summary[g].get(test_k, 0) for g in grans]
        val_stds   = [summary[g].get(val_std_k, 0) for g in grans]
        test_stds  = [summary[g].get(test_std_k, 0) for g in grans]
        m1_vals    = [summary[g].get(m1_val_k) for g in grans]
        m1_tests   = [summary[g].get(m1_test_k) for g in grans]

        # ┏━━━━━━━━━━ Plot bars ━━━━━━━━━━┓
        x = np.arange(len(grans))
        w = 0.3
        ax.bar(x - w/2, val_means,  w, yerr=val_stds,  label="M2 Val",  color="darkorange", capsize=4, edgecolor="white", alpha=0.7)
        ax.bar(x + w/2, test_means, w, yerr=test_stds, label="M2 Test", color="royalblue",  capsize=4, edgecolor="white", alpha=0.7)

        # ┏━━━━━━━━━━ Plot M1 baselines ━━━━━━━━━━┓
        for i in range(len(grans)):
            if m1_vals[i] is not None:
                ax.hlines(m1_vals[i],  x[i] - 0.4, x[i] + 0.4, colors="darkorange", linestyles="--", lw=2, label="M1 Val" if i == 0 else "")
            if m1_tests[i] is not None:
                ax.hlines(m1_tests[i], x[i] - 0.4, x[i] + 0.4, colors="royalblue",  linestyles="--", lw=2, label="M1 Test" if i == 0 else "")

        # ┏━━━━━━━━━━ Set legend and title ━━━━━━━━━━┓
        ax.set_xticks(x)
        ax.set_xticklabels(grans, fontsize=10)
        ax.set_ylabel(label)
        ax.set_title(label, fontsize=12)
        ax.legend(loc="lower right", fontsize=8)
        ax.grid(axis="y", alpha=0.3)

    # ┏━━━━━━━━━━ Set suptitle and save ━━━━━━━━━━┓
    fig.suptitle(f"Edge Analysis — {direction.upper()} — {n_trials} trials per granularity", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)



# ┏━━━━━━━━━━ Plot CPCV split matrix ━━━━━━━━━━┓
def _plot_split_matrix(splits, paths, n_blocks, boundaries, purge_td, save_path, gran, direction):
    """Visualize the Train/Test split for each CPCV split with path labels and professional styling."""
    import matplotlib.patches as mpatches
    
    # ┏━━━━━━━━━━ Number of splits and paths ━━━━━━━━━━┓
    n_splits = len(splits)
    n_paths = len(paths)
    
    # ┏━━━━━━━━━━ Create figure and axes ━━━━━━━━━━┓
    fig, ax = plt.subplots(figsize=(8, 0.35 * n_splits + 2.5))

    # ┏━━━━━━━━━━ Design System ━━━━━━━━━━┓
    COLOR_TEST  = "royalblue" 
    COLOR_TRAIN = "darkorange"
    COLOR_PURGE = "#ffffff"
    HATCH_PURGE = "XX"
    EDGE_COLOR  = "#7F8C8D"
    
    # ┏━━━━━━━━━━ Build map: (split_idx, block_idx) -> list(path_indices) ━━━━━━━━━━┓
    block_path_map = {}
    for pi, p_entries in enumerate(paths):
        for entry in p_entries:
            key = (entry["split_idx"], entry["block"])
            if key not in block_path_map:
                block_path_map[key] = []
            block_path_map[key].append(pi + 1)

    # ┏━━━━━━━━━━ Plot each split ━━━━━━━━━━┓
    for i, s in enumerate(splits):
        test_blocks = set(s["test_blocks"])
        
        # ┏━━━━━━━━━━ Add subtle horizontal guide line for the split ━━━━━━━━━━┓
        ax.axhline(i, color="#f8f8f8", lw=0.5, zorder=0)

        # ┏━━━━━━━━━━ Plot each block ━━━━━━━━━━┓
        for b in range(n_blocks):
            # ┏━━━━━━━━━━ Determine if block is in test set ━━━━━━━━━━┓
            is_test = b in test_blocks
            color = COLOR_TEST if is_test else COLOR_TRAIN
            
            # ┏━━━━━━━━━━ Base rectangle ━━━━━━━━━━┓
            rect = mpatches.Rectangle((b, i - 0.42), 1.0, 0.85, color=color, ec="white", lw=0.5, alpha=0.9, zorder=2)
            ax.add_patch(rect)

            # ┏━━━━━━━━━━ Path labels ━━━━━━━━━━┓
            if is_test:
                paths_using_this = block_path_map.get((i, b), [])
                if paths_using_this:
                    label = ", ".join([f"Path {p}" for p in paths_using_this])
                    ax.text(b + 0.5, i, label, color="black", ha="center", va="center", 
                            fontsize=8, fontweight="bold", zorder=3)

            # ┏━━━━━━━━━━ Purge indicators ━━━━━━━━━━┓
            p_width = 0.05 
            # Left boundary
            if b > 0 and is_test != ((b - 1) in test_blocks):
                ax.add_patch(mpatches.Rectangle((b - p_width/2, i - 0.42), 
                                                width     = p_width, 
                                                height    = 0.85, 
                                                facecolor = COLOR_PURGE, 
                                                hatch     = HATCH_PURGE, 
                                                edgecolor = EDGE_COLOR, 
                                                lw        = 0.5, 
                                                zorder    = 4))
            # Right boundary
            if b < n_blocks - 1 and is_test != ((b + 1) in test_blocks):
                ax.add_patch(mpatches.Rectangle((b + 1 - p_width/2, i - 0.42), 
                                                width     = p_width, 
                                                height    = 0.85, 
                                                facecolor = COLOR_PURGE, 
                                                hatch     = HATCH_PURGE, 
                                                edgecolor = EDGE_COLOR, 
                                                lw        = 0.5, 
                                                zorder    = 4))

    # ┏━━━━━━━━━━ Formatting ━━━━━━━━━━┓
    ax.set_xlim(0, n_blocks)
    ax.set_ylim(-1, n_splits)
    
    # ┏━━━━━━━━━━ X-Axis: Chronological dates (aligned to block starts + final end date) ━━━━━━━━━━┓
    block_dates = [b[0].strftime("%Y-%m-%d") for b in boundaries]
    block_dates.append(boundaries[-1][1].strftime("%Y-%m-%d"))
    ax.set_xticks(np.arange(n_blocks + 1))
    ax.set_xticklabels(block_dates, rotation=45, ha="right", fontsize=9, color="#2C3E50")
    
    # ┏━━━━━━━━━━ Y-Axis: Split labels ━━━━━━━━━━┓
    ax.set_yticks(np.arange(n_splits))
    ax.set_yticklabels([f"Split {i+1}" for i in range(n_splits)], fontsize=9, color="#2C3E50")
    ax.invert_yaxis()
    
    # ┏━━━━━━━━━━ Remove clutter ━━━━━━━━━━┓
    for spine in ["top", "right", "bottom", "left"]:
        ax.spines[spine].set_visible(False)
    ax.tick_params(axis="both", length=0) 

    # ┏━━━━━━━━━━ Title & Subtitle (Fixed Path Count) ━━━━━━━━━━┓
    # Title
    plt.text(x = 0.5, 
             y = 1.12, 
             s   = f"CPCV Split Strategy — {gran} {direction.upper()}", 
             transform   = ax.transAxes, 
             fontsize    = 16, 
             fontweight  = "bold", 
             ha          = "center", 
             color       = "#2C3E50")
    
    # Subtitle
    plt.text(x = 0.5, 
             y = 1.05, 
             s   = f"Purge Window: {purge_td} | N={n_blocks} Blocks/Split | {n_paths} Chronological Paths", 
             transform   = ax.transAxes, 
             fontsize    = 11, 
             ha          = "center", 
             color       = "#7F8C8D")

    # ┏━━━━━━━━━━ Legend ━━━━━━━━━━┓
    train_p = mpatches.Patch(color=COLOR_TRAIN, label="Training Set")
    test_p  = mpatches.Patch(color=COLOR_TEST,  label="Test Set")
    purge_p = mpatches.Patch(facecolor=COLOR_PURGE, hatch=HATCH_PURGE, edgecolor=EDGE_COLOR, label="Purged/Embargoed")
    legend = ax.legend(handles        = [train_p, test_p, purge_p], 
                       loc            = "upper center", 
                       bbox_to_anchor = (0.5, -0.18), 
                       ncol           = 3, 
                       frameon        = False, 
                       fontsize       = 10)
    for text in legend.get_texts():
        text.set_color("#2C3E50")

    # ┏━━━━━━━━━━ Save figure ━━━━━━━━━━┓
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)



# ┏━━━━━━━━━━ Plot path boxplots ━━━━━━━━━━┓
def _plot_path_boxplots(path_metrics, m1_baselines, save_path, gran, direction):
    """Boxplots of metrics across CPCV paths."""

    # ┏━━━━━━━━━━ Initialize variables ━━━━━━━━━━┓
    metrics_cfg = [("accuracy", "Accuracy (@0.5)"), 
                   ("sel_accuracy", "Selective Accuracy"),
                   ("precision", "Precision (@0.5)"), 
                   ("sel_precision", "Selective Precision"),
                   ("mean_ret", "Mean Ret (@0.5)"), 
                   ("sel_mean_ret", "Selective Mean Ret"),
                   ("sel_coverage", "Selective Coverage")]

    # ┏━━━━━━━━━━ Create figure ━━━━━━━━━━┓
    fig, axes = plt.subplots(1, len(metrics_cfg), figsize=(3 * len(metrics_cfg), 5))
    
    # ┏━━━━━━━━━━ Plot boxplots ━━━━━━━━━━┓
    for ax, (m_key, label) in zip(axes, metrics_cfg):
        # ┏━━━━━━━━━━ Extract values ━━━━━━━━━━┓
        vals = [pm[m_key] for pm in path_metrics]
        
        # ┏━━━━━━━━━━ Create boxplot ━━━━━━━━━━┓
        bp = ax.boxplot([vals], labels=["Paths"], patch_artist=True, widths=0.5)
        bp["boxes"][0].set_facecolor("cornflowerblue")

        # ┏━━━━━━━━━━ Add M1 baseline ━━━━━━━━━━┓
        if m_key == "accuracy" and m1_baselines.get("m1_acc") is not None:
            ax.axhline(m1_baselines["m1_acc"], color="red", lw=1.5, ls="--", label=f"M1={m1_baselines['m1_acc']:.3f}")
            ax.legend(fontsize=7, loc="upper right")
        elif m_key == "sel_accuracy" and m1_baselines.get("m1_acc") is not None:
            ax.axhline(m1_baselines["m1_acc"], color="red", lw=1.5, ls="--", label=f"M1={m1_baselines['m1_acc']:.3f}")
            ax.legend(fontsize=7, loc="upper right")
        elif m_key == "precision" and m1_baselines.get("m1_prec") is not None:
            ax.axhline(m1_baselines["m1_prec"], color="red", lw=1.5, ls="--", label=f"M1={m1_baselines['m1_prec']:.3f}")
            ax.legend(fontsize=7, loc="upper right")
        elif m_key == "sel_precision" and m1_baselines.get("m1_prec") is not None:
            ax.axhline(m1_baselines["m1_prec"], color="red", lw=1.5, ls="--",
                       label=f"M1={m1_baselines['m1_prec']:.3f}")
            ax.legend(fontsize=7, loc="upper right")
        elif m_key in ["sel_mean_ret", "mean_ret"]:
            m1_ret = np.mean([pm["m1_mean_ret"] for pm in path_metrics])
            ax.axhline(m1_ret, color="red", lw=1.5, ls="--", label=f"M1={m1_ret:.4f}")
            ax.legend(fontsize=7, loc="upper right")

        # ┏━━━━━━━━━━ Add individual points ━━━━━━━━━━┓
        ax.scatter([1] * len(vals), vals, color="navy", s=30, zorder=5, alpha=0.7)
        ax.set_title(label, fontsize=9); ax.grid(axis="y", alpha=0.3)

    # ┏━━━━━━━━━━ Set title and save figure ━━━━━━━━━━┓
    fig.suptitle(f"CPCV Path Distribution — {gran} {direction.upper()} | {len(path_metrics)} Chronological Paths", fontsize=12, fontweight="bold")
    fig.tight_layout()

    # ┏━━━━━━━━━━ Save figure ━━━━━━━━━━┓
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)



# ┏━━━━━━━━━━ Plot cross-granularity CPCV ━━━━━━━━━━┓
def _plot_cross_gran_cpcv(summary, save_path, direction):
    # ┏━━━━━━━━━━ Initialize variables ━━━━━━━━━━┓
    grans = list(summary.keys())
    metrics_cfg = [
        ("path_acc_mean",      "path_acc_std",      "m1_baseline_acc",  "Accuracy (@0.5)"),
        ("path_prec_mean",     "path_prec_std",     "m1_baseline_prec", "Precision (@0.5)"),
        ("path_sel_prec_mean", "path_sel_prec_std", "m1_baseline_prec", "Selective Precision"),
        ("path_sel_ret_mean",  "path_sel_ret_std",  None,               "Selective Mean Ret")]

    # ┏━━━━━━━━━━ Create figure ━━━━━━━━━━┓
    fig, axes = plt.subplots(2, 2, figsize=(max(12, len(grans) * 2.5), 9))
    
    # ┏━━━━━━━━━━ Plot metrics ━━━━━━━━━━┓
    for ax, (mean_k, std_k, m1_k, label) in zip(axes.flatten(), metrics_cfg):
        means = [summary[g].get(mean_k, 0) for g in grans]
        stds  = [summary[g].get(std_k, 0)  for g in grans]
        m1s   = [summary[g].get(m1_k) for g in grans] if m1_k else [None] * len(grans)
        x = np.arange(len(grans))
        ax.bar(x, means, 0.5, yerr=stds, color="cornflowerblue", capsize=4, edgecolor="white", alpha=0.8, label="M2 CPCV Paths")
        
        # ┏━━━━━━━━━━ Add M1 baseline ━━━━━━━━━━┓
        for i, m1 in enumerate(m1s):
            if m1 is not None:
                ax.hlines(m1, x[i] - 0.35, x[i] + 0.35, colors="red", linestyles="--", lw=2, label="M1 Baseline" if i == 0 else "")
        
        # ┏━━━━━━━━━━ Set ticks, labels and title ━━━━━━━━━━┓
        ax.set_xticks(x); ax.set_xticklabels(grans, fontsize=9)
        ax.set_ylabel(label); ax.set_title(label, fontsize=11)
        ax.legend(fontsize=7, loc="lower right"); ax.grid(axis="y", alpha=0.3)
    
    # ┏━━━━━━━━━━ Set super title and save figure ━━━━━━━━━━┓
    fig.suptitle(f"CPCV Edge — {direction.upper()} — Cross-Granularity", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ┏━━━━━━━━━━ Load full CPCV records from edge results ━━━━━━━━━━┓
def _load_cpcv_records_full(edge_root: str,
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


# ┏━━━━━━━━━━ CPCV Edge Convergence Heatmap (Val + Test) ━━━━━━━━━━┓
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
    records = _load_cpcv_records_full(edge_root)
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



# ┏━━━━━━━━━━ Load CPCV records with constraint_satisfied=True ━━━━━━━━━━┓
def _load_cpcv_records_constraint(edge_root: Path, bt_root: Path):
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


# ┏━━━━━━━━━━ Plot filter confusion ━━━━━━━━━━┓
def plot_cpcv_filter_confusion(edge_root: Path, bt_root: Path, save_path: Path):
    """Top-30 filter combinations bar chart for CPCV reliability analysis.

    Saves results_matrices_summary-style PNG showing TP/FP/FN/TN bars and
    precision/recall/accuracy lines for the top-30 filter combinations
    (out of singles + pairs + triples) ranked by test precision.
    """
    from itertools import combinations
    records = _load_cpcv_records_constraint(edge_root, bt_root)
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


# ┏━━━━━━━━━━ Compute Tabular Models vs CTTS Comparison ━━━━━━━━━━┓
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


# ┏━━━━━━━━━━ Plot CV Sweep ━━━━━━━━━━┓
def plot_cv_sweep(metrics_pkl: Path,
                  save_path: Path,
                  n_grid: int = 2000) -> None:
    """CV-threshold sweep comparing four pipeline variants.

    For each CV threshold τ ∈ [0, 1] the function partitions all 320 configs
    from ``metrics_combined_dict.pkl`` into:

      • M1 only            — all configs (baseline, constant).
      • M1 + M2 vanilla    — configs where τ=0.5 metrics exist (no reliability filter).
      • M1 + M2 RA         — configs with cv < τ (reliability-aware, τ=0.5 threshold).
      • M1 + M2 τ̂ + RA    — configs with cv < τ AND constraint_satisfied (full pipeline).

    Left y-axis : mean total return across surviving configs.
    Right y-axis: number of surviving configurations.
    A red dashed vertical marks the argmax of the full-pipeline curve.
    """
    import pickle as _pickle

    # ── palette consistent with paper ────────────────────────────────────
    C = {
        "m1":     "#6FA8DC",   # muted blue
        "vanilla":"#7FB069",   # muted green
        "ra":     "#E89A4F",   # muted orange
        "full":   "#C28EC9",   # muted purple
    }

    with open(metrics_pkl, "rb") as f:
        metrics = _pickle.load(f)

    # Flatten all configs into a list once
    all_cfgs = [
        metrics[m1][m2][d][g]
        for m1 in metrics
        for m2 in metrics[m1]
        for d  in metrics[m1][m2]
        for g  in metrics[m1][m2][d]
    ]

    nan = float("nan")
    cv_grid = np.linspace(0.0, 1.0, n_grid)

    ret_m1_all      = []
    ret_vanilla_all = []
    ret_ra_all      = []
    ret_full_all    = []
    cnt_ra_all      = []
    cnt_full_all    = []

    # M1 baseline and vanilla are constant (no cv filter)
    m1_rets      = [c["m1_total_return"]    for c in all_cfgs
                    if isinstance(c["m1_total_return"], float)
                    and c["m1_total_return"] == c["m1_total_return"]]
    vanilla_rets = [c["m2_total_return_tau05"] for c in all_cfgs
                    if isinstance(c["m2_sharpe_tau05"], float)
                    and c["m2_sharpe_tau05"] == c["m2_sharpe_tau05"]]
    m1_mean      = float(np.mean(m1_rets))      if m1_rets      else nan
    vanilla_mean = float(np.mean(vanilla_rets)) if vanilla_rets else nan

    for t in cv_grid:
        ra_rets   = [c["m2_total_return_tau05"]  for c in all_cfgs
                     if c["cv"] < t
                     and isinstance(c["m2_total_return_tau05"], float)
                     and c["m2_total_return_tau05"] == c["m2_total_return_tau05"]]
        full_rets = [c["m2_total_return_tauhat"] for c in all_cfgs
                     if c["cv"] < t and c["constraint_satisfied"]
                     and isinstance(c["m2_total_return_tauhat"], float)
                     and c["m2_total_return_tauhat"] == c["m2_total_return_tauhat"]]
        ret_m1_all.append(m1_mean)
        ret_vanilla_all.append(vanilla_mean)
        ret_ra_all.append(float(np.mean(ra_rets))   if ra_rets   else nan)
        ret_full_all.append(float(np.mean(full_rets)) if full_rets else nan)
        cnt_ra_all.append(len(ra_rets))
        cnt_full_all.append(len(full_rets))

    ret_m1_all      = np.array(ret_m1_all)
    ret_vanilla_all = np.array(ret_vanilla_all)
    ret_ra_all      = np.array(ret_ra_all)
    ret_full_all    = np.array(ret_full_all)
    cnt_ra_all      = np.array(cnt_ra_all)
    cnt_full_all    = np.array(cnt_full_all)

    # argmax of full pipeline
    valid_full = np.isfinite(ret_full_all)
    i_max = int(np.nanargmax(ret_full_all)) if valid_full.any() else 0
    t_star = float(cv_grid[i_max])
    r_star = float(ret_full_all[i_max])

    # ── plot ──────────────────────────────────────────────────────────────
    FS = 16
    fig, ax1 = plt.subplots(figsize=(14,7), dpi=180, facecolor="white")
    ax1.set_facecolor("#FAFAFA")

    ax1.plot(cv_grid, ret_m1_all,      color=C["m1"],     linewidth=2.0,
             label=rf"M1 Baseline  ($\mu$ = {m1_mean:+.1f}%)")
    ax1.plot(cv_grid, ret_vanilla_all, color=C["vanilla"], linewidth=2.0,
             label=rf"M1+M2 [$\tau=0.5$, No CV]  ($\mu$ = {vanilla_mean:+.1f}%)")
    ax1.plot(cv_grid, ret_ra_all,      color=C["ra"],      linewidth=2.0,
             label=r"M1+M2 [$\tau=0.5$, CV]")
    ax1.plot(cv_grid, ret_full_all,    color=C["full"],    linewidth=2.0,
             label=r"M1+M2 [$\tau^*$, CV]")
    ax1.axvline(t_star, color="#C0392B", linestyle="--", linewidth=1.4,
                label=rf"Optimal $\alpha^* = {t_star:.3f}$  ($\mu$ = {r_star:+.1f}%)")
    ax1.scatter([t_star], [r_star], color="#C0392B", marker="D",
                s=48, edgecolors="white", linewidths=1.0, zorder=5)
    ax1.annotate(r"$\alpha^*$", xy=(t_star, 0), xycoords=("data", "axes fraction"),
                 xytext=(0, -5), textcoords="offset points",
                 ha="center", va="top", color="#C0392B", fontsize=FS+2, fontweight="bold")

    ax1.set_xlabel(r"CV threshold  $\alpha$", fontsize=FS, fontweight="bold", color="black")
    ax1.set_ylabel("Mean Return (%)",
                   fontsize=FS, fontweight="bold", color="black")
    ax1.tick_params(axis="both", labelsize=FS, colors="black")
    ax1.set_xlim(0.0, 1.0)
    ax1.set_ylim(bottom=-40)
    ax1.grid(True, color="#DDDDDD", linewidth=0.5, alpha=0.8)
    ax1.spines[["top"]].set_visible(False)

    # right axis — config counts
    ax2 = ax1.twinx()
    ax2.plot(cv_grid, cnt_ra_all,   color=C["ra"],   linewidth=1.4,
             linestyle=":", label=r"№ Configurations [$\tau=0.5$]")
    ax2.plot(cv_grid, cnt_full_all, color=C["full"], linewidth=1.4,
             linestyle=":", label=r"№ Configurations [$\tau^*$, CV]")
    ax2.set_ylabel("№ of Configurations",
                   fontsize=FS, fontweight="bold", color="black", labelpad=13)
    ax2.tick_params(axis="y", labelsize=FS, colors="black")
    ax2.spines[["top"]].set_visible(False)

    # combined legend
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    
    leg = fig.legend(h1 + h2, l1 + l2, loc="lower center", bbox_to_anchor=(0.5, -0.012),
                     ncol=3, prop={"size": FS}, frameon=True, framealpha=0.95,
                     edgecolor="#BDC3C7", fancybox=True,
                     handlelength=2.4, handletextpad=0.6,
                     columnspacing=1.2, borderpad=0.6)
    leg.set_zorder(20)

    fig.tight_layout()
    fig.subplots_adjust(bottom=0.28)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(save_path), dpi=200, facecolor="white")
    plt.close(fig)
    print(f"[plot_cv_sweep] Saved -> {save_path}  "
          f"(CV* = {t_star:.3f}, mean ret = {r_star:+.1f}%)")



