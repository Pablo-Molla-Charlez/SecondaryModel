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

# ┏━━━━━━━━━━ CLI model name → models.py model key ━━━━━━━━━━┓
_CLI_TO_MODEL_KEY = {"randforest": "rf",
                     "xgboost":    "xgboost",
                     "autogluon":  "autogluon",
                     "tabpfn":     "tabpfn",
                     "tabpfn_ft":  "tabpfn_ft",
                     "tabicl":     "tabicl"}


# ┏━━━━━━━━━━ Metrics to plot ━━━━━━━━━━┓
METRICS_TO_PLOT = [("accuracy",      "Accuracy (@0.5)"),
                   ("sel_accuracy",  "Selective Accuracy"),
                   ("precision",     "Precision (@0.5)"),
                   ("sel_precision", "Selective Precision"),
                   ("mean_ret",      "Mean Ret (@0.5)"),
                   ("sel_mean_ret",  "Selective Mean Ret"),
                   ("sel_coverage",  "Selective Coverage")]


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
