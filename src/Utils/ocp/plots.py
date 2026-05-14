"""Reusable OCP and SAOCP helpers for Kronos M2 analyses.

Includes three threshold-selection modes for online conformal prediction:

  1. "_run_saocp_online"  — Original SAOCP (optionally windowed).
  2. "_run_cost_deferral_online" — Cost-aware deferral (vanilla):
     dynamically re-optimises τ* on a rolling window by minimising
     an explicit expected cost  L(τ) = c_FP·FP + c_FN·FN + c_DEF·DEF.
  3. "_run_cost_deferral_online" with "mondrian=True" — Mondrian
     cost-aware deferral: computes separate τ* per volatility regime
     (low-vol / high-vol) using realized volatility of recent returns.

All functions return the same 4-tuple
    "(test_thresholds, test_approved, val_thresholds, conformal_stats)"
so they are drop-in replacements for one another.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional


# ┏━━━━━━━━━━ Granularity → candles per day mapping ━━━━━━━━━━┓
_CANDLES_PER_DAY = {"1d": 1, "12h": 2, "8h": 3, "6h": 4, "4h": 6,
                    "2h": 12, "1h": 24, "30m": 48, "15m": 96, "5m": 288}


# ┏━━━━━━━━━━ Cost-Aware Deferral (Vanilla + Mondrian) ━━━━━━━━━━┓
_TAU_GRID = np.arange(0.50, 0.96, 0.01)  # search grid for deferral threshold


__all__ = ["plot_mondrian_diagnostics", "plot_ocp_threshold_evolution"]


def plot_mondrian_diagnostics(conformal_stats, save_dir, gran_label="", thres_mode="OCP-cost-mondrian"):
    """Generate 3 diagnostic plots for Mondrian cost-aware deferral.

    Plots saved to save_dir:
      1. RV time series with median split line
      2. τ* per regime (low-vol vs high-vol) over time
      3. Rolling win rate per regime (does the split separate easy/hard?)

    Parameters
    ----------
    conformal_stats : dict
        Output from _run_cost_deferral_online with mondrian=True.
    save_dir : str or Path
        Directory to save the plot files.
    gran_label : str
        Granularity label for titles (e.g. "4h").
    thres_mode : str
        Threshold mode label for titles.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from pathlib import Path

    diag = conformal_stats.get("mondrian_diag")
    if diag is None:
        return

    # ┏━━━━━━━━━━ Extract Information for realized volatility-based regime analysis ━━━━━━━━━━┓
    save_dir = Path(save_dir)
    test_rv      = diag["test_rv"]
    regimes      = diag["regime_assignments"]
    tau_low_t    = diag["tau_low_trajectory"]    # None for non-Mondrian
    tau_high_t   = diag["tau_high_trajectory"]   # None for non-Mondrian
    is_mondrian  = diag.get("is_mondrian", False)
    median_rv_t  = diag["median_rv_trajectory"]
    labels       = diag["test_labels"]
    tau_traj     = conformal_stats["tau_trajectory"]
    n = len(test_rv)
    idx = np.arange(n)

    cost_p = conformal_stats.get("cost_params", {})
    cost_tag = f"c_FP={cost_p.get('c_FP','?')}, c_DEF={cost_p.get('c_DEF','?')}"

    # ┏━━━━━━━━━━ Plot 1: RV time series with median split ━━━━━━━━━━┓
    fig, ax = plt.subplots(figsize=(12, 4), facecolor="white")
    ax.set_facecolor("#FAFAFA")
    valid = ~np.isnan(test_rv)
    ax.plot(idx[valid], test_rv[valid], color="#2980B9", linewidth=0.6, alpha=0.8, label="Realized Volatility")
    valid_med = ~np.isnan(median_rv_t)
    if valid_med.any():
        ax.plot(idx[valid_med], median_rv_t[valid_med], color="#E74C3C", linewidth=1.2,
                linestyle="--", alpha=0.9, label="Rolling median (split)")
    
    # ┏━━━━━━━━━━ Shade Regimes: Low vs High ━━━━━━━━━━┓
    rv_max = float(np.nanmax(test_rv)) if valid.any() else 1.0
    low = regimes == 0
    high = regimes == 1
    if low.any():
        ax.fill_between(idx, 0, rv_max, where=low, alpha=0.08, color="#27AE60", label="Low-vol regime")
    if high.any():
        ax.fill_between(idx, 0, rv_max, where=high, alpha=0.08, color="#E74C3C", label="High-vol regime")
    ax.set_xlabel("Test sample index", fontsize=10)
    ax.set_ylabel("Realized Volatility", fontsize=10)
    ax.set_title(f"Realized Volatility & Regime Assignment  |  {gran_label}  |  {thres_mode}\n{cost_tag}", fontsize=11, fontweight="bold", color="#2C3E50")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    # ┏━━━━━━━━━━ Save Plot ━━━━━━━━━━┓
    fig.savefig(str(save_dir / "mondrian_rv_regimes.png"), dpi=200, facecolor="white")
    plt.close(fig)

    # ┏━━━━━━━━━━ Plot 2: τ* per regime over time (only meaningful for Mondrian) ━━━━━━━━━━┓
    fig, ax = plt.subplots(figsize=(12, 4), facecolor="white")
    ax.set_facecolor("#FAFAFA")
    if is_mondrian:
        ax.plot(idx, tau_traj, color="#8B008B", linewidth=0.4, alpha=0.25, label="τ* applied (per-sample)")
    else:
        ax.plot(idx, tau_traj, color="#8B008B", linewidth=0.8, alpha=0.7, label="τ* applied")
    if is_mondrian and tau_low_t is not None and tau_high_t is not None:
        valid_low = ~np.isnan(tau_low_t)
        valid_high = ~np.isnan(tau_high_t)
        if valid_high.any():
            ax.plot(idx[valid_high], tau_high_t[valid_high], color="#E74C3C", linewidth=1.2, alpha=0.9, label="τ* high-vol")
        if valid_low.any():
            ax.plot(idx[valid_low], tau_low_t[valid_low], color="#27AE60", linewidth=1.2, alpha=0.9,
                    linestyle="--", label="τ* low-vol")
    ax.axhline(y=0.5, color="#BDC3C7", linestyle=":", linewidth=0.8, alpha=0.6)
    ax.set_xlabel("Test sample index", fontsize=10)
    ax.set_ylabel("Deferral threshold τ*", fontsize=10)
    title_2 = "Per-Regime τ* Evolution" if is_mondrian else "τ* Evolution (global, no regime split)"
    ax.set_title(f"{title_2}  |  {gran_label}  |  {thres_mode}\n{cost_tag}", fontsize=11, fontweight="bold", color="#2C3E50")
    ax.legend(fontsize=8, loc="upper right")
    ax.set_ylim(0.45, 1.0)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    # ┏━━━━━━━━━━ Save Plot ━━━━━━━━━━┓
    fig.savefig(str(save_dir / "mondrian_tau_regimes.png"), dpi=200, facecolor="white")
    plt.close(fig)

    # ┏━━━━━━━━━━ Plot 3: Rolling win rate per regime ━━━━━━━━━━┓
    rolling_w = min(200, n // 5) if n > 50 else max(10, n // 3)
    fig, ax = plt.subplots(figsize=(12, 4), facecolor="white")
    ax.set_facecolor("#FAFAFA")

    # ┏━━━━━━━━━━ Overall Rolling Window Win-Rate ━━━━━━━━━━┓
    wins = (labels == 1).astype(float)
    overall_wr = pd.Series(wins).rolling(rolling_w, min_periods=20).mean()
    ax.plot(idx, overall_wr.values, color="#8B008B", linewidth=1.0, alpha=0.6, label=f"Overall WR (roll={rolling_w})")

    # ┏━━━━━━━━━━ Per-regime WR: compute only within regime windows ━━━━━━━━━━┓
    for regime_val, regime_name, color in [(0, "Low-vol", "#27AE60"), (1, "High-vol", "#E74C3C")]:
        mask = regimes == regime_val
        if mask.sum() < 20:
            continue
        # ┏━━━━━━━━━━ Rolling WR within regime samples only ━━━━━━━━━━┓
        regime_wins = np.where(mask, wins, np.nan)
        regime_wr = pd.Series(regime_wins).rolling(rolling_w, min_periods=20).mean()
        ax.plot(idx, regime_wr.values, color=color, linewidth=1.2, alpha=0.9, label=f"{regime_name} WR")

    # ┏━━━━━━━━━━ Summary stats in text box ━━━━━━━━━━┓
    low_mask = regimes == 0
    high_mask = regimes == 1
    low_wr = labels[low_mask].mean() * 100 if low_mask.sum() > 0 else 0
    high_wr = labels[high_mask].mean() * 100 if high_mask.sum() > 0 else 0
    overall_wr_val = labels.mean() * 100
    n_low = int(low_mask.sum())
    n_high = int(high_mask.sum())
    stats_text = (f"Low-vol: WR={low_wr:.1f}% (n={n_low})\n"
                  f"High-vol: WR={high_wr:.1f}% (n={n_high})\n"
                  f"Overall: WR={overall_wr_val:.1f}% (n={n})\n"
                  f"Δ WR = {low_wr - high_wr:+.1f}pp")
    ax.text(0.02, 0.97, stats_text, transform=ax.transAxes, fontsize=8, verticalalignment="top", fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="#BDC3C7", alpha=0.9))

    ax.axhline(y=0.5, color="#BDC3C7", linestyle=":", linewidth=0.8, alpha=0.6)
    ax.set_xlabel("Test sample index", fontsize=10)
    ax.set_ylabel("Win Rate", fontsize=10)
    ax.set_title(f"Win Rate by Volatility Regime  |  {gran_label}  |  {thres_mode}\n"
                 f"{cost_tag}  |  Does the regime split separate easy/hard periods?",
                 fontsize=11, fontweight="bold", color="#2C3E50")
    ax.legend(fontsize=8, loc="upper right")
    ax.set_ylim(0.2, 0.8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    # ┏━━━━━━━━━━ Save Plot ━━━━━━━━━━┓
    fig.savefig(str(save_dir / "mondrian_wr_regimes.png"), dpi=200, facecolor="white")
    plt.close(fig)


# ━━━━━━━━━━ Migrated from Utils/feature_selection/plots.py (2026-05-14) ━━━━━━━━━━
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

