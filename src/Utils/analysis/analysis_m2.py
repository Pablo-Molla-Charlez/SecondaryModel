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


# ┏━━━━━━━━━━ Result-grid constants (M1 × M2 × dir × gran) ━━━━━━━━━━┓
_M1_LIST    = ["Tirex", "Chronos2", "Fincast", "Kronos"]
_M2_LIST    = ["rf", "autogluon", "tabpfn", "tabicl", "ctts"]
_DIRS_LIST  = ["UP", "DOWN"]
_GRANS_LIST = ["1d", "12h", "8h", "6h", "4h", "2h", "1h", "30m"]


# ┏━━━━━━━━━━ Plot M2 Prediction Returns Histogram ━━━━━━━━━━┓
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


# ┏━━━━━━━━━━ Plot M2 Results Radar Chart ━━━━━━━━━━┓
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


# ┏━━━━━━━━━━ Plot M2 Results Radar FocusedChart ━━━━━━━━━━┓
def plot_results_radar_focused(data_root: str = "/home/pablo/M2_DS/Secondary-Model/src/Output",
                               output_dir: str = "/home/pablo/M2_DS/Secondary-Model/src/Output/Analysis/Results",
                               m1: str = "Kronos",
                               m2_models: tuple = ("rf", "tabpfn", "ctts"),
                               metric: str = "m2_return"):
    """Two side-by-side spider charts for ONE M1 (UP and DOWN), with only the
    selected ``m2_models`` plotted. Colours match ``plot_best_m2_per_gran`` so
    the figure is consistent across the paper.

    Layout:  1 row x 2 cols  (col 0: UP, col 1: DOWN)
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


# ┏━━━━━━━━━━ Plot M2 Results Radar Chart of Precision and Profitability ━━━━━━━━━━┓
def plot_kronos_down_combined(data_root: str = "/home/pablo/M2_DS/Secondary-Model/src/Output",
                              output_dir: str = "/home/pablo/M2_DS/Secondary-Model/src/Output/Analysis/Results",
                              m2_models: tuple = ("rf", "tabpfn", "ctts")):
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


# ┏━━━━━━━━━━ Compute Metrics at a given Threshold ━━━━━━━━━━┓
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


# ┏━━━━━━━━━━ Walk Through Results Configs ━━━━━━━━━━┓
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


# ┏━━━━━━━━━━ Build Combined Metrics Dict ━━━━━━━━━━┓
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


# ┏━━━━━━━━━━ Plot M2 Performance Matrices ━━━━━━━━━━┓
def plot_results_matrices(bt_root: Path, edge_root: Path, save_path: Path):
    """Generate a 1x3 panel of summary matrices (precision, profitability, ROI).

    Each matrix is 2x2 with axes:
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
        f"{len(_M1_LIST)} M1 x {len(_M2_LIST)} M2 x 2 directions x {len(_GRANS_LIST)} granularities",
        fontsize=14, fontweight="bold", y=1.02)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(save_path), dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[plot_results_matrices] Saved -> {save_path}")


# ┏━━━━━━━━━━ Plot Best M2 Per Granularity (Selected Metrics) ━━━━━━━━━━┓
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


# ┏━━━━━━━━━━ Heatmap of M2 predicted-associated returns ━━━━━━━━━━┓
def plot_return_heatmap(metrics_pkl: Path,
                        save_path: Path,
                        constrained: bool = True) -> None:
    """2-row x 4-column grid of heatmaps showing M2 test total return.

    Layout
    ------
    * Row 0 (top)    : UP direction  — columns: Tirex, Chronos2, Fincast, Kronos
    * Row 1 (bottom) : DOWN direction — same column order
    * Each heatmap   : y-axis = granularity (1d … 30m), x-axis = M2 model
    * Shared diverging colorbar on the far right, spanning both rows.
    * Missing / unsatisfied cells rendered in light grey.

    Parameters
    ----------
    metrics_pkl : Path
        Path to ``metrics_combined_dict.pkl``.
    save_path : Path
        Output PNG path.
    constrained : bool, default True
        If True  — uses ``m2_total_return_tauhat`` and only shows cells where
                   ``constraint_satisfied`` is True (reliability-filtered).
        If False — uses ``m2_total_return_tau05`` (baseline τ=0.5) and shows
                   every cell regardless of constraint status.
    In both cases the colorbar is calibrated to the values that are actually
    displayed, so the colour scale is never diluted by off-screen extremes.
    """
    import pickle as _pickle
    import numpy as _np
    import matplotlib.pyplot as _plt
    import matplotlib.colors as _mcolors
    from matplotlib.cm import ScalarMappable

    with open(metrics_pkl, "rb") as fh:
        metrics = _pickle.load(fh)

    M1_LIST   = ["Tirex", "Chronos2", "Fincast", "Kronos"]
    M2_LIST   = ["rf", "tabpfn", "tabicl", "autogluon", "ctts"]
    M2_LABELS = ["RF", "TPFN", "TICL", "AG", "CTTS"]
    GRANS     = ["1d", "12h", "8h", "6h", "4h", "2h", "1h", "30m"]
    DIRS      = ["UP", "DOWN"]

    FS = 11   # unified font size — axis labels are additionally bolded

    # ── helper: extract a value for one (m1, m2, dir, gran) cell ───────────
    def _get(m1, m2, dr, g):
        e = metrics.get(m1, {}).get(m2, {}).get(dr, {}).get(g, {})
        if not e:
            return _np.nan, False
        satisfied = e.get("constraint_satisfied", False)
        if constrained:
            # Only show cells where the constraint was satisfied, using optimal τ̂
            if not satisfied:
                return _np.nan, satisfied
            v = e.get("m2_total_return_tauhat", _np.nan)
        else:
            # Show everything: use τ̂ where constraint was satisfied, τ=0.5 otherwise
            if satisfied:
                v = e.get("m2_total_return_tauhat", _np.nan)
            else:
                v = e.get("m2_total_return_tau05", _np.nan)
        val = float(v) if (v is not None and _np.isfinite(float(v))) else _np.nan
        return val, satisfied

    # ── build all data matrices first so the colorscale sees only displayed vals
    data_grids = {}
    sat_grids = {}
    all_shown = []
    for dr in DIRS:
        for m1 in M1_LIST:
            mat = _np.full((len(GRANS), len(M2_LIST)), _np.nan)
            sat_mat = _np.full((len(GRANS), len(M2_LIST)), False, dtype=bool)
            for gi, g in enumerate(GRANS):
                for mi, m2 in enumerate(M2_LIST):
                    v, sat = _get(m1, m2, dr, g)
                    mat[gi, mi] = v
                    sat_mat[gi, mi] = sat
                    if _np.isfinite(v):
                        all_shown.append(v)
            data_grids[(dr, m1)] = mat
            sat_grids[(dr, m1)] = sat_mat

    if not all_shown:
        print("[plot_return_heatmap] No finite values found — aborting.")
        return

    # Create a custom colormap that physically covers the true min/max of the data
    # but the gradient is strictly clamped between -50 and 100
    grad_vmin, grad_vmax = -50.0, 100.0
    actual_vmin = min(min(all_shown), grad_vmin)
    actual_vmax = max(max(all_shown), grad_vmax)
    
    span = actual_vmax - actual_vmin
    p_n50 = (grad_vmin - actual_vmin) / span
    p_0   = (0.0 - actual_vmin) / span
    p_100 = (grad_vmax - actual_vmin) / span
    
    base_cmap = _plt.get_cmap("RdYlGn")
    positions, colors = [], []
    
    # From actual min up to -50, use the solid dark red
    positions.extend([0.0, p_n50])
    colors.extend([base_cmap(0.0), base_cmap(0.0)])
    
    # From -50 to 0, sample the lower half of RdYlGn
    n_samples = 50
    for i in range(1, n_samples):
        frac = i / n_samples
        positions.append(p_n50 + frac * (p_0 - p_n50))
        colors.append(base_cmap(frac * 0.5))
        
    positions.append(p_0)
    colors.append(base_cmap(0.5))
    
    # From 0 to 100, sample the upper half of RdYlGn
    for i in range(1, n_samples):
        frac = i / n_samples
        positions.append(p_0 + frac * (p_100 - p_0))
        colors.append(base_cmap(0.5 + frac * 0.5))
        
    # From 100 up to actual max, use the solid dark green
    positions.extend([p_100, 1.0])
    colors.extend([base_cmap(1.0), base_cmap(1.0)])
    
    cmap = _mcolors.LinearSegmentedColormap.from_list("custom_clamped", list(zip(positions, colors)))
    norm = _mcolors.Normalize(vmin=actual_vmin, vmax=actual_vmax)

    # ── figure ───────────────────────────────────────────────────────────────
    fig_w = 4.5 * len(M1_LIST) 
    fig_h = 3.0 * len(DIRS)
    fig, axes = _plt.subplots(
        len(DIRS), len(M1_LIST),
        figsize=(fig_w, fig_h),
        facecolor="white",
        constrained_layout=False,
    )
    fig.patch.set_facecolor("white")

    # ── render each heatmap ──────────────────────────────────────────────────
    for row_idx, dr in enumerate(DIRS):
        for col_idx, m1 in enumerate(M1_LIST):
            ax   = axes[row_idx][col_idx]
            data = data_grids[(dr, m1)]

            sat_mat = sat_grids[(dr, m1)]

            # Grey background layer (fills NaN cells)
            ax.imshow(_np.zeros_like(data), cmap=_plt.cm.Greys,
                      vmin=0, vmax=1, aspect="auto", alpha=0.15)

            # Draw heatmap via RGBA so we can individually control alpha transparency
            safe_data = _np.nan_to_num(data, nan=0.0)
            rgba = cmap(norm(safe_data))
            
            # Base alpha: 0 for NaN, 1 for valid
            alpha_mask = _np.where(_np.isnan(data), 0.0, 1.0)
            
            # Fade out UNSATISFIED cells in the unconstrained plot to emphasize the ones that satisfied the constraints
            if not constrained:
                fade_mask = ~sat_mat & ~_np.isnan(data)
                alpha_mask = _np.where(fade_mask, 0.6, alpha_mask)
                
            rgba[..., 3] = alpha_mask
            ax.imshow(rgba, aspect="auto")

            # Cell annotations
            for gi in range(len(GRANS)):
                for mi in range(len(M2_LIST)):
                    val = data[gi, mi]
                    if _np.isfinite(val):
                        # Approximate the text contrast check based on gradient
                        clamped_val = max(grad_vmin, min(grad_vmax, val))
                        if clamped_val < 0:
                            val_normed = 0.5 * (clamped_val - grad_vmin) / (0 - grad_vmin)
                        else:
                            val_normed = 0.5 + 0.5 * (clamped_val - 0) / (grad_vmax - 0)
                            
                        txt_color = "black" if abs(val_normed - 0.5) < 0.35 else "white"
                        
                        # Apply fading to the text if the cell is faded (i.e. constraint failed)
                        if not constrained and not sat_mat[gi, mi]:
                            ax.text(mi, gi, f"{val:.1f}%", ha="center", va="center",
                                    fontsize=FS - 1, color=txt_color, alpha=0.4)
                        else:
                            ax.text(mi, gi, f"{val:.1f}%", ha="center", va="center",
                                    fontsize=FS - 1, color=txt_color)
                    else:
                        ax.text(mi, gi, "N/A",
                                ha="center", va="center",
                                fontsize=FS - 2, color="#888888")

            # Ticks
            ax.set_xticks(range(len(M2_LIST)))
            ax.set_yticks(range(len(GRANS)))

            if row_idx == len(DIRS) - 1:
                ax.set_xticklabels(M2_LABELS, fontsize=FS, fontweight="bold",
                                   rotation=0, ha="center")
            else:
                ax.set_xticklabels([])

            if col_idx == 0:
                ax.set_yticklabels(GRANS, fontsize=FS, fontweight="bold", va="center")
            else:
                ax.set_yticklabels([])

            ax.tick_params(axis="both", which="both", length=0)

            # Column title (top row only)
            if row_idx == 0:
                ax.set_title(rf"{m1}", fontsize=FS, fontweight="bold", pad=6)

            # Row label on rightmost column
            if col_idx == len(M1_LIST) - 1:
                ax.yaxis.set_label_position("right")
                ax.set_ylabel(f"{dr}", fontsize=FS, fontweight="bold",
                              rotation=270, labelpad=8, va="bottom")

            # Minor grid lines between cells
            ax.set_xticks(_np.arange(-0.5, len(M2_LIST), 1), minor=True)
            ax.set_yticks(_np.arange(-0.5, len(GRANS), 1), minor=True)
            ax.grid(which="minor", color="white", linewidth=1.2)
            ax.tick_params(which="minor", bottom=False, left=False)

    # ── shared colorbar ──────────────────────────────────────────────────────
    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    
    # Create a completely independent axes for the colorbar to guarantee no overlap
    # [left, bottom, width, height] in figure coordinates (0 to 1)
    # Made it slightly thinner for a sleeker look
    cax = fig.add_axes([0.91, 0.15, 0.012, 0.70])
    cbar = fig.colorbar(sm, cax=cax)
    
    # Add max/min values explicitly to the colorbar ticks, plus more intermediate values for the profitable side
    default_ticks = [-50, 0, 50, 100]
    new_ticks = list(default_ticks)
    if actual_vmin < -55:
        new_ticks.insert(0, int(actual_vmin))
        
    curr_tick = 150
    while curr_tick <= actual_vmax - 20: # -20 to prevent text overlap with the actual_vmax tick
        new_ticks.append(curr_tick)
        curr_tick += 50
        
    if actual_vmax > 105:
        new_ticks.append(int(actual_vmax))
    cbar.set_ticks(new_ticks)
    
    # Professional styling for the colorbar
    cbar.outline.set_edgecolor('#555555')
    cbar.outline.set_linewidth(0.8)
    cbar.ax.tick_params(labelsize=FS, size=0)  # remove protruding tick lines
    
    cbar_label = (r"Total Return (%) — $\tau^*$ [constrained]"
                  if constrained else
                  r"Total Return (%)")
    cbar.set_label(cbar_label, fontsize=FS, fontweight="bold", labelpad=6)

    tag = "constrained" if constrained else "unconstrained"
    

    # Force the heatmaps to stop at 83% of the figure width, giving the row labels
    # and the colorbar at 88% full unobstructed space.
    _plt.subplots_adjust(left=0.06, right=0.88, wspace=0.06, hspace=0.10, bottom=0.15, top=0.90)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(save_path), dpi=200, bbox_inches="tight", facecolor="white")
    _plt.close(fig)
    print(f"[plot_return_heatmap] Saved -> {save_path}  (constrained={constrained})")


# ┏━━━━━━━━━━ Plot M2 Predicted-associated returns vs Precision-Profitability ━━━━━━━━━━┓
def plot_selective_classification_vs_profitability(res_mat, selective_cls_metrics, save_path: Path):
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import pickle
    from scipy.interpolate import griddata
    from scipy.spatial import ConvexHull
    from matplotlib.colors import TwoSlopeNorm, ListedColormap
    from matplotlib.path import Path as MplPath
    from matplotlib.ticker import FormatStrFormatter, MaxNLocator
    from pathlib import Path as _Path

    # Load data if they are paths
    def _load_data(obj):
        if isinstance(obj, (str, _Path)):
            with open(obj, "rb") as f:
                return pickle.load(f)
        return obj

    res_mat = _load_data(res_mat)
    selective_cls_metrics = _load_data(selective_cls_metrics)

    common_font_size = 20
    plt.rcParams.update({
        'font.size': common_font_size,
        'axes.labelsize': common_font_size,
        'axes.titlesize': common_font_size,
        'xtick.labelsize': common_font_size,
        'ytick.labelsize': common_font_size,
        'legend.fontsize': common_font_size,
    })

    m1s = ["Kronos"]
    m2s = ["rf", "autogluon", "tabpfn", "tabicl", "ctts"]
    directions = ["UP", "DOWN"]
    granularity = ["1d", "12h", "8h", "6h", "4h", "2h", "1h", "30m"]

    dfs = {}
    all_z = []
    all_y = []

    for direction in directions:
        rows = []
        for gran in granularity:
            for m1 in m1s:
                for m2 in m2s:
                    try:
                        # Try to find the config in res_mat
                        cfg = None
                        if m1 in res_mat and m2 in res_mat[m1] and direction in res_mat[m1][m2] and gran in res_mat[m1][m2][direction]:
                            cfg = res_mat[m1][m2][direction][gran]
                        
                        # Try to find the config in selective_cls_metrics (with or without "metrics" key)
                        cfg_sel = None
                        source = selective_cls_metrics
                        if isinstance(source, dict) and "metrics" in source: source = source["metrics"]
                        
                        if isinstance(source, dict) and m1 in source and m2 in source[m1] and direction in source[m1][m2] and gran in source[m1][m2][direction]:
                            cfg_sel = source[m1][m2][direction][gran]

                        # Extract values with fallbacks
                        if cfg is None and cfg_sel is not None: cfg = cfg_sel
                        if cfg_sel is None and cfg is not None: cfg_sel = cfg
                        
                        if cfg is None: continue
                        
                        total_return = cfg.get("m2_total_return_tau05", cfg.get("total_return", 0))
                        test_prec = cfg_sel.get("test_precision", cfg_sel.get("m2_precision_tau05", 0))
                        
                        # Handle precision being 0-1 vs 0-100
                        if 0 < test_prec <= 1.0: test_prec *= 100
                        
                        if test_prec == 0:
                            continue
                            
                        # Coverage calculation
                        n_trades = cfg_sel.get("coverage", None)
                        if n_trades is None:
                            m2_n = cfg_sel.get("m2_n_trades_tau05", 0)
                            m1_n = cfg_sel.get("m1_n_trades", 1)
                            n_trades = (m2_n / m1_n) * 100 if m1_n > 0 else 0
                        else:
                            # If coverage is already a fraction, convert to %
                            if n_trades <= 1.0: n_trades *= 100

                        rows.append({
                            "M1": m1,
                            "Granularity": gran,
                            "M2": m2,
                            "m2_total_return": total_return,
                            "n_trades": n_trades,
                            "test_prec": test_prec
                        })
                    except:
                        continue

        df = pd.DataFrame(rows)
        if df.empty:
            dfs[direction] = df
            continue
            
        dfs[direction] = df
        all_z.extend(df["m2_total_return"].values)
        all_y.extend(df["n_trades"].values)

    if not all_z or not all_y:
        print("No valid data for selective classification plot.")
        return

    vmin = np.min(all_z)
    vmax = np.max(all_z)
    if vmin >= 0: vmin = -1e-5
    if vmax <= 0: vmax = 1e-5
    
    norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
    max_trades = np.max(all_y)

    fig, axes = plt.subplots(1, 2, figsize=(14, 7), sharey=True)
    fig.subplots_adjust(bottom=0.34, right=0.86, wspace=0.12)

    contour_ref = None
    selected = [0, 2, 3, 4, 7]
    unique_grans = granularity
    
    # Custom colormap injection: replace the 4h color with magenta
    base_cmap = plt.get_cmap("tab10")
    color_list = [base_cmap(i) for i in range(len(unique_grans))]
    gran_to_idx = {g: i for i, g in enumerate(unique_grans)}
    
    if "4h" in gran_to_idx:
        color_list[gran_to_idx["4h"]] = plt.matplotlib.colors.to_rgba("magenta")
    
    if "6h" in gran_to_idx:
        color_list[gran_to_idx["6h"]] = plt.matplotlib.colors.to_rgba("cyan")
    cmap = ListedColormap(color_list)

    selected_grans = [unique_grans[i] for i in selected]
    legend_handles = [
        plt.Line2D([0], [0], marker='o', color='w', label=g,
                   markerfacecolor=cmap(gran_to_idx[g]), markeredgecolor='black', markersize=8)
        for g in selected_grans
    ]
    loss_boundary_color = "#6A3D9A"
    profit_boundary_color = "#1F4E79"
    precision_plot_min = 40
    precision_plot_max = 80

    def _thresholds_for_df(df):
        if df.empty:
            return None, None
        x_vals = df["test_prec"].values
        z_vals = df["m2_total_return"].values
        unique_precision = np.sort(np.unique(x_vals))
        left_candidates = [p for p in unique_precision if np.all(z_vals[x_vals <= p] < 0)]
        right_candidates = [p for p in unique_precision if np.all(z_vals[x_vals >= p] > 0)]
        left_threshold = left_candidates[-1] if left_candidates else None
        right_threshold = right_candidates[0] if right_candidates else None
        return left_threshold, right_threshold

    thresholds_by_direction = {
        direction: _thresholds_for_df(dfs.get(direction, pd.DataFrame()))
        for direction in directions
    }
    threshold_display_start = 48
    threshold_display_end = 53

    def _make_precision_warp(left_threshold, right_threshold):
        if left_threshold is None or right_threshold is None or right_threshold <= left_threshold:
            left_threshold = threshold_display_start
            right_threshold = threshold_display_end
        pre_factor = (threshold_display_start - precision_plot_min) / max(left_threshold - precision_plot_min, 1e-12)
        mid_factor = (threshold_display_end - threshold_display_start) / max(right_threshold - left_threshold, 1e-12)
        post_factor = (precision_plot_max - threshold_display_end) / max(precision_plot_max - right_threshold, 1e-12)

        def _warp_precision(values):
            is_scalar = np.isscalar(values)
            values = np.asarray(values)
            warped = np.piecewise(
                values,
                [
                    values <= left_threshold,
                    (values > left_threshold) & (values <= right_threshold),
                    values > right_threshold,
                ],
                [
                    lambda val: precision_plot_min + (val - precision_plot_min) * pre_factor,
                    lambda val: threshold_display_start + (val - left_threshold) * mid_factor,
                    lambda val: threshold_display_end + (val - right_threshold) * post_factor,
                ],
            )
            return float(warped) if is_scalar else warped

        return _warp_precision

    for idx, (ax, direction) in enumerate(zip(axes, directions)):
        df = dfs[direction]
        if df.empty:
            ax.set_title(direction)
            continue

        x = df["test_prec"].values
        y = df["n_trades"].values
        z = df["m2_total_return"].values
        gran_labels = df["Granularity"].values
        left_threshold, right_threshold = thresholds_by_direction[direction]
        warp_precision = _make_precision_warp(left_threshold, right_threshold)

        xi = np.linspace(x.min(), x.max(), 100)
        yi = np.linspace(y.min(), y.max(), 100)
        X, Y = np.meshgrid(xi, yi)
        X_plot = warp_precision(X)
        Z = griddata((x, y), z, (X, Y), method='linear')
        if np.isnan(Z).any():
            Z_nearest = griddata((x, y), z, (X, Y), method='nearest')
            Z = np.where(np.isnan(Z), Z_nearest, Z)
        if len(x) >= 3:
            points = np.column_stack((warp_precision(x), y))
            try:
                hull = ConvexHull(points)
                hull_path = MplPath(points[hull.vertices])
                grid_points = np.column_stack((X_plot.ravel(), Y.ravel()))
                outside_hull = ~hull_path.contains_points(grid_points, radius=1e-9)
                Z = np.ma.masked_where(outside_hull.reshape(X.shape), Z)
            except Exception:
                pass

        contour = ax.contourf(X_plot, Y, Z, levels=np.linspace(vmin, vmax, 100), cmap="RdYlGn", norm=norm)
        contour_ref = contour

        gran_idx = np.array([gran_to_idx[g] for g in gran_labels])
        mask = np.isin(gran_idx, selected)

        ax.scatter(warp_precision(x[mask]), y[mask], c=gran_idx[mask], cmap=cmap, s=70, edgecolors='black', alpha=0.8)

        if left_threshold is not None:
            x1 = left_threshold
            ax.axvline(warp_precision(x1), color=loss_boundary_color, linestyle='--', linewidth=2.4, alpha=0.95)

        if right_threshold is not None:
            x1 = right_threshold
            ax.axvline(warp_precision(x1), color=profit_boundary_color, linestyle='--', linewidth=2.4, alpha=0.95)

        ax.set_title(direction)
        ax.set_xlabel("Precision (%)")
        ax.xaxis.set_label_coords(0.5, -0.17)
        base_ticks = [40, 80, 60, 65, 70, 75, 80] if direction == "UP" else [40, 45, 60, 65, 70, 75, 80]
        direction_threshold_values = [
            round(value, 1)
            for value in thresholds_by_direction[direction]
            if value is not None
        ]
        tick_values = np.array(sorted(set(base_ticks + direction_threshold_values)))
        ax.set_xlim(warp_precision(precision_plot_min), warp_precision(precision_plot_max))
        ax.set_xticks(warp_precision(tick_values))
        ax.set_xticklabels([f"{tick:g}" for tick in tick_values])
        for tick_label, tick in zip(ax.get_xticklabels(), tick_values):
            tick_label.set_ha("center")
            tick_label.set_rotation_mode("anchor")
            threshold_match = [
                role
                for left, right in [thresholds_by_direction[direction]]
                for role, value in (("left", left), ("right", right))
                if value is not None and np.isclose(round(value, 1), tick)
            ]
            if "left" in threshold_match:
                tick_label.set_color(loss_boundary_color)
                tick_label.set_rotation(35)
                tick_label.set_ha("center")
                tick_label.set_y(-0.025)
            if "right" in threshold_match:
                tick_label.set_color(profit_boundary_color)
                tick_label.set_rotation(35)
                tick_label.set_ha("center")
                tick_label.set_y(-0.025)
        ax.set_ylim(-5, max_trades + 5)
        ax.grid(True, linestyle='--', linewidth=0.6, color='0.75', alpha=0.7)
        ax.set_axisbelow(True)
        
    axes[0].set_ylabel("Coverage (%)")
    fig.legend(handles=legend_handles, loc='lower center',
               bbox_to_anchor=(0.5, 0.025), ncol=len(selected_grans),
               frameon=True, handletextpad=0.25, columnspacing=1.25, borderaxespad=2.5)

    if contour_ref:
        cbar = fig.colorbar(contour_ref, ax=axes, orientation='vertical', fraction=0.045, pad=0.02)
        cbar.set_label("Total Return (%)", labelpad=10)
        cbar.ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        cbar.ax.yaxis.set_major_formatter(FormatStrFormatter("%d"))

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)



# ┏━━━━━━━━━━ Fold regime composition (CPCV / CVExpand / Test) ━━━━━━━━━━┓
# Motivation: CPCV and expanding-window CV rank direction×granularity configs in the
# opposite order to the held-out test period — both favour UP (which loses on test)
# and disfavour DOWN (which wins on test). The hypothesis under test here is that the
# divergence is a *regime-mix* artifact: validation covers a mixed/rising market while
# the test window is a sustained bear. These helpers join each evaluated sample to a
# dated Sparse-Jump-Model regime label so fold composition and regime-conditional
# performance can be measured rather than assumed.

_JM_GLOBAL_ROOT = Path("/home/pablo/M2_DS/SecondaryModel/src/Output/Jump_Models/_global")


def _find_jm_fit_dir(granularity: str, root: Path = _JM_GLOBAL_ROOT) -> Optional[Path]:
    """Locate the pooled K=2 fit directory for a granularity.

    Fits are named ``sparse_K2_lam<lambda>`` where lambda is BIC-selected per
    granularity, so the exact directory name is not known ahead of time.
    """
    gran_dir = root / granularity
    if not gran_dir.is_dir():
        return None
    cands = sorted(d for d in gran_dir.glob("sparse_K2_lam*") if (d / "labels").is_dir())
    if not cands:
        return None
    if len(cands) > 1:
        # Several lambdas fitted for one granularity: prefer the newest.
        cands = sorted(cands, key=lambda d: d.stat().st_mtime)
    return cands[-1]


def load_regime_labels(granularity: str,
                       root: Path = _JM_GLOBAL_ROOT) -> Tuple[pd.DataFrame, dict]:
    """Load per-asset causal regime labels for one granularity.

    Returns ``(labels_df, meta)`` where labels_df has columns
    ``[asset, date, regime, split]``. Labels are the *causal* (online) ones: at each
    bar the regime is inferred from information available up to that bar, so joining
    them to trading samples introduces no lookahead.
    """
    fit_dir = _find_jm_fit_dir(granularity, root)
    if fit_dir is None:
        raise FileNotFoundError(
            f"No pooled K=2 jump-model fit for granularity '{granularity}' under {root}. "
            f"Fit one first (Utils/regime/scripts/fit_all_granularities.sh).")
    meta = json.loads((fit_dir / "meta.json").read_text())
    frames = []
    for csv in sorted((fit_dir / "labels").glob("*.csv")):
        d = pd.read_csv(csv, parse_dates=["date"])
        d["asset"] = csv.stem
        frames.append(d[["asset", "date", "regime", "split"]])
    if not frames:
        raise FileNotFoundError(f"No label CSVs under {fit_dir / 'labels'}")
    return pd.concat(frames, ignore_index=True), meta


def _regime_direction_map(granularity: str,
                          labels: pd.DataFrame,
                          raw_root: Path = None) -> Dict[int, str]:
    """Map integer regime ids to 'Bearish'/'Bullish'.

    The fit already guarantees this ordering: RegimeJumpModel relabels states by
    each regime's CAUSAL mean return, so id 0 is the lowest-mean-return (most
    bearish) state and id K-1 the highest (see Utils/regime/model.py,
    ``_return_ordered_remap``). The saved labels are therefore already ordered, and
    this function only names them — it deliberately does NOT re-derive the ordering
    from prices, which would introduce a second, subtly different source of truth
    (raw pct_change over all bars vs the model's aligned return series over
    causally-labelled bars) and could silently contradict the fit.

    Names match ``Utils/regime/regime_breakdown._regime_label_map`` so this table is
    directly comparable to the per-regime breakdowns already reported elsewhere.

    ``raw_root`` is accepted and ignored; kept so callers need not special-case it.
    """
    present = sorted(int(r) for r in labels["regime"].dropna().unique())
    if not present:
        return {}
    k = max(present) + 1
    if k == 2:
        names = {0: "Bearish", 1: "Bullish"}
    elif k == 3:
        names = {0: "Bearish", 1: "Sideways", 2: "Bullish"}
    else:
        names = {i: f"regime {i}" for i in range(k)}
    return {r: names[r] for r in present if r in names}


def verify_regime_ordering(granularity: str,
                           root: Path = _JM_GLOBAL_ROOT,
                           raw_root: Path = Path("/home/pablo/M2_DS/SecondaryModel/src/Data_MLA/Raw"),
                           ) -> pd.DataFrame:
    """Check that saved labels really are return-ordered (id 0 = most bearish).

    The naming above trusts the fit's ordering guarantee rather than re-deriving it.
    That trust is worth auditing once per fit, so this recomputes realised mean and
    cumulative return per regime from raw closes and reports them. A correct fit
    yields mean return increasing with regime id.
    """
    labels, _ = load_regime_labels(granularity, root)
    parts = []
    for asset, grp in labels.groupby("asset", sort=False):
        csv = Path(raw_root) / granularity / f"{asset}.csv"
        if not csv.exists():
            continue
        px = (pd.read_csv(csv, parse_dates=["date"], usecols=["date", "close"])
                .set_index("date").sort_index())
        j = grp.set_index("date").join(px["close"].pct_change().rename("ret"), how="inner")
        parts.append(j[["regime", "ret"]])
    if not parts:
        raise ValueError(f"No raw closes matched labels for {granularity}")
    allr = pd.concat(parts).dropna()
    out = (allr.groupby("regime")["ret"]
                .agg(n="size", mean_ret="mean",
                     cum_ret=lambda r: float(np.expm1(np.log1p(r).sum())))
                .reset_index())
    out["name"] = out["regime"].map(_regime_direction_map(granularity, labels))
    out["ordering_ok"] = out["mean_ret"].is_monotonic_increasing
    return out


# ┏━━━━━━━━━━ Output layout for the regime analysis ━━━━━━━━━━┓
# One clearly named root, partitioned M1 / M2 / period / analysis. The period split is
# the important one: `train_val/` holds everything measured on data the models were
# fitted on (CPCV paths, expanding-window folds) and `test/` holds only the held-out
# period. A claim of generalisation may cite `test/` alone; `train_val/` establishes
# that an effect is real and replicates, not that it transfers.
#
# Two analyses straddle both periods (the 4-scenario table has one TEST scenario plus
# three validation ones; the significance suite runs bootstrap on TEST but Wilcoxon and
# permutation on validation), so those are split by ROW into the two folders rather than
# filed under whichever period dominates.
REGIME_ANALYSIS_ROOT = Path("/home/pablo/M2_DS/SecondaryModel/src/Output/Analysis/RegimeAnalysis")

REGIME_SUBDIRS = {
    "market_vs_filter": "market_vs_filter",
    "granularity": "granularity",
    "cv_test_transfer": "cv_test_transfer",
    "scenarios":    "bull_vs_bear_scenarios",       # which regime wins (descriptive)
    "significance": "bull_vs_bear_significance",    # wilcoxon / permutation / bootstrap
    "test_quarter": "bull_vs_bear_by_quarter",      # held-out gap, quarterly breakdown
    "m2_vs_m1":     "m2_minus_m1_by_regime",        # M2's gain over M1, within regime
    "robust_significance": "robust_significance",    # tests B/C/E/G on the regime gap
    "composition":  "regime_composition_by_split",  # model-independent: %bull/bear
}


def regime_out_dir(analysis: str, m1: str | None = None, m2: str | None = None,
                   period: str | None = None, root: Path = REGIME_ANALYSIS_ROOT) -> Path:
    """Resolve (and create) the output directory for one regime analysis.

    ``period`` is "train_val" or "test"; omit it for model-independent analyses.
    """
    sub = REGIME_SUBDIRS.get(analysis, analysis)
    if m1 is None:
        d = Path(root) / sub
    elif period is None:
        d = Path(root) / m1 / m2 / sub
    else:
        d = Path(root) / m1 / m2 / period / sub
    d.mkdir(parents=True, exist_ok=True)
    return d


def _write_by_granularity(df: pd.DataFrame, out_dir: Path, verbose: bool = True) -> None:
    """Write one CSV per granularity plus a combined all_granularities.csv."""
    if df.empty:
        return
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "all_granularities.csv", index=False)
    if "granularity" in df.columns:
        for g, part in df.groupby("granularity"):
            part.to_csv(out_dir / f"{g}.csv", index=False)
    if verbose:
        print(f"  [saved] {out_dir}")


def _write_split_by_period(df: pd.DataFrame, analysis: str, m1: str, m2: str,
                           test_mask: "pd.Series", verbose: bool = True) -> None:
    """Route TEST-period rows and train+val rows to their own folders."""
    for period, part in (("test", df[test_mask]), ("train_val", df[~test_mask])):
        if len(part):
            _write_by_granularity(part, regime_out_dir(analysis, m1, m2, period), verbose)


def regime_split_table(granularities: Optional[List[str]] = None,
                       root: Path = _JM_GLOBAL_ROOT,
                       raw_root: Path = Path("/home/pablo/M2_DS/SecondaryModel/src/Data_MLA/Raw"),
                       save_path: Optional[Path] = None) -> pd.DataFrame:
    """Table A — regime share (%) of train / val / test, per granularity.

    Establishes the baseline claim the fold analysis rests on: whether the three
    config-defined periods differ in regime mix at all, and by how much.
    """
    grans = granularities or ["1d", "12h", "8h", "6h", "4h", "2h", "1h", "30m"]
    rows = []
    for g in grans:
        try:
            labels, meta = load_regime_labels(g, root)
        except FileNotFoundError:
            continue
        name_of = _regime_direction_map(g, labels, raw_root)
        labels = labels.assign(regime_name=labels["regime"].map(name_of))
        for split in ("train", "val", "test"):
            part = labels[labels["split"] == split]
            if part.empty:
                continue
            share = part["regime_name"].value_counts(normalize=True)
            rows.append({
                "granularity": g,
                "split": split,
                "n_bars": len(part),
                "bull_pct": 100 * float(share.get("Bullish", 0.0)),
                "bear_pct": 100 * float(share.get("Bearish", 0.0)),
                "lambda": meta.get("jump_penalty"),
            })
    out = pd.DataFrame(rows)
    if not out.empty:
        d = Path(save_path) if save_path is not None else regime_out_dir("composition")
        _write_by_granularity(out, d, verbose=False)
    return out


def _load_fold_samples(edge_dir: Path, scheme: str) -> Optional[pd.DataFrame]:
    """Read the per-sample CSV written by the CPCV / cvexpand runners."""
    fname, unit = (("cpcv_samples.csv", "path") if scheme == "cpcv"
                   else ("cvexpand_samples.csv", "split"))
    p = edge_dir / fname
    if not p.exists():
        return None
    d = pd.read_csv(p, parse_dates=["date"])
    return d.rename(columns={unit: "fold"}).assign(scheme=scheme)


def _attach_regimes(samples: pd.DataFrame,
                    granularity: str,
                    root: Path,
                    raw_root: Path) -> pd.DataFrame:
    """Join samples to their (asset, date) causal regime label.

    Matching is exact on (asset, date): the jump model is fitted at the same
    granularity as the samples, so no resampling or forward-fill is involved.
    """
    labels, _ = load_regime_labels(granularity, root)
    name_of = _regime_direction_map(granularity, labels, raw_root)
    labels = labels.assign(regime_name=labels["regime"].map(name_of))
    merged = samples.merge(labels[["asset", "date", "regime_name", "split"]],
                           on=["asset", "date"], how="left")
    return merged


def _perf_by_regime(df: pd.DataFrame, fee: float = 0.002) -> pd.DataFrame:
    """Per-regime precision / return / coverage for one set of evaluated samples.

    ``@0.5`` rows use the raw M2 decision (``pred``); ``sel`` rows use the
    utility-optimal threshold the runner already applied (``sel``).
    """
    out = []
    for name, g in df.groupby("regime_name", dropna=True):
        pred = g["pred"].astype(bool)
        sel = g["sel"].astype(bool)
        out.append({
            "regime": name,
            "n": len(g),
            "share_pct": np.nan,  # filled by caller against its own total
            "prec_05": float(g.loc[pred, "y_true"].mean()) if pred.any() else np.nan,
            "prec_sel": float(g.loc[sel, "y_true"].mean()) if sel.any() else np.nan,
            "ret_05": float((g.loc[pred, "ret"] - fee).mean()) if pred.any() else np.nan,
            "ret_sel": float((g.loc[sel, "ret"] - fee).mean()) if sel.any() else np.nan,
            "coverage": float(sel.mean()),
        })
    res = pd.DataFrame(out)
    if not res.empty:
        res["share_pct"] = 100 * res["n"] / res["n"].sum()
    return res


def _reweight(perf: pd.DataFrame, target_shares: Dict[str, float],
              cols=("prec_05", "prec_sel", "ret_05", "ret_sel")) -> Dict[str, float]:
    """Recombine per-regime metrics at another period's regime proportions.

    This is the counterfactual: *if the validation folds had contained the test
    period's mix of bull and bear bars, what would they have reported?* Weights are
    renormalised over the regimes actually present, so a fold missing one regime
    degrades to its single-regime value rather than silently reporting a partial sum.
    """
    present = perf.dropna(subset=["regime"])
    w = np.array([target_shares.get(r, 0.0) for r in present["regime"]], dtype=float)
    if w.sum() <= 0:
        return {c: np.nan for c in cols}
    w = w / w.sum()
    res = {}
    for c in cols:
        v = present[c].to_numpy(dtype=float)
        m = np.isfinite(v)
        res[c] = float(np.sum(w[m] * v[m]) / w[m].sum()) if m.any() else np.nan
    return res


def analyze_fold_regime_composition(
        edge_root: Path = Path("/home/pablo/M2_DS/SecondaryModel/src/Output/Analysis/Edge_NoCal"),
        test_root: Path = Path("/home/pablo/M2_DS/SecondaryModel/src/Output"),
        jm_root: Path = _JM_GLOBAL_ROOT,
        raw_root: Path = Path("/home/pablo/M2_DS/SecondaryModel/src/Data_MLA/Raw"),
        m1: str = "Kronos",
        m2: str = "rf",
        directions: Tuple[str, ...] = ("UP", "DOWN"),
        granularities: Optional[List[str]] = None,
        fee: float = 0.002,
        save_dir: Optional[Path] = None,
        verbose: bool = True) -> Dict[str, pd.DataFrame]:
    """Tables A/B/C linking fold regime composition to CPCV-vs-test divergence.

    Returns a dict of DataFrames:
        ``splits``      Table A — regime share of train/val/test per granularity.
        ``folds``       Table B — regime share per CPCV path and CVExpand split.
        ``conditional`` Table C — regime-conditional precision/return per scheme,
                        plus CPCV/CVExpand metrics reweighted to the *test* period's
                        regime mix, alongside the actual test values.

    Table C is the one that answers the question: if ``cpcv_reweighted`` moves toward
    ``test_actual`` relative to ``cpcv_raw``, the validation/test disagreement is
    explained by regime mix; whatever gap remains is not.
    """
    grans = granularities or ["1d", "12h", "8h", "6h", "4h", "2h", "1h"]
    split_tbl = regime_split_table(grans, jm_root, raw_root)

    fold_rows, cond_rows = [], []
    for direction in directions:
        for g in grans:
            edge_dir = edge_root / m1 / m2 / direction / g
            if not edge_dir.is_dir():
                continue
            try:
                test_shares = _test_regime_shares(g, jm_root, raw_root)
            except FileNotFoundError:
                if verbose:
                    print(f"  [skip] no jump model for {g}")
                continue

            for scheme in ("cpcv", "cvexpand"):
                samples = _load_fold_samples(edge_dir, scheme)
                if samples is None:
                    if verbose:
                        print(f"  [skip] {direction}/{g}: no {scheme} per-sample CSV "
                              f"(re-run the runner to emit it)")
                    continue
                merged = _attach_regimes(samples, g, jm_root, raw_root)
                unmatched = merged["regime_name"].isna().mean()
                if unmatched > 0.01 and verbose:
                    print(f"  [warn] {direction}/{g}/{scheme}: "
                          f"{100*unmatched:.1f}% samples unmatched to a regime")

                # ── Table B: composition per fold ────────────────────────────
                for fold, fg in merged.groupby("fold"):
                    share = fg["regime_name"].value_counts(normalize=True)
                    fold_rows.append({
                        "direction": direction, "granularity": g, "scheme": scheme,
                        "fold": int(fold), "n": len(fg),
                        "bull_pct": 100 * float(share.get("Bullish", 0.0)),
                        "bear_pct": 100 * float(share.get("Bearish", 0.0)),
                    })

                # ── Table C: regime-conditional + reweighted ─────────────────
                perf = _perf_by_regime(merged, fee=fee)
                raw_all = _perf_by_regime(merged.assign(regime_name="ALL"), fee=fee)
                rw = _reweight(perf, test_shares)
                for c in ("prec_05", "prec_sel", "ret_05", "ret_sel"):
                    cond_rows.append({
                        "direction": direction, "granularity": g, "scheme": scheme,
                        "metric": c,
                        "raw": float(raw_all[c].iloc[0]) if not raw_all.empty else np.nan,
                        "reweighted_to_test": rw[c],
                        **{f"{r.lower()[:4]}_{c}": float(perf.loc[perf.regime == r, c].iloc[0])
                           if (perf.regime == r).any() else np.nan
                           for r in ("Bullish", "Bearish")},
                    })

    folds = pd.DataFrame(fold_rows)
    cond = pd.DataFrame(cond_rows)
    if save_dir is not None:
        save_dir = Path(save_dir); save_dir.mkdir(parents=True, exist_ok=True)
        split_tbl.to_csv(save_dir / "regime_table_A_splits.csv", index=False)
        folds.to_csv(save_dir / "regime_table_B_folds.csv", index=False)
        cond.to_csv(save_dir / "regime_table_C_conditional.csv", index=False)
        if verbose:
            print(f"[regime-composition] saved 3 tables -> {save_dir}")
    return {"splits": split_tbl, "folds": folds, "conditional": cond}


def _test_regime_shares(granularity: str, jm_root: Path, raw_root: Path) -> Dict[str, float]:
    """Regime proportions of the held-out test split, used as reweighting targets."""
    labels, _ = load_regime_labels(granularity, jm_root)
    name_of = _regime_direction_map(granularity, labels, raw_root)
    test = labels[labels["split"] == "test"]
    share = test["regime"].map(name_of).value_counts(normalize=True)
    return {k: float(v) for k, v in share.items()}


def temporal_decay_table(edge_root: Path = Path("/home/pablo/M2_DS/SecondaryModel/src/Output/Analysis/Edge_NoCal"),
                         m1: str = "Kronos",
                         m2: str = "rf",
                         directions: Tuple[str, ...] = ("UP", "DOWN"),
                         granularities: Optional[List[str]] = None,
                         fee: float = 0.002,
                         save_path: Optional[Path] = None) -> Dict[str, pd.DataFrame]:
    """Per-fold return vs fold recency, for expanding-window CV and CPCV.

    Tests whether the strategy's edge decays over the validation window. This is the
    alternative to the regime-composition explanation: fold regime mix turned out to
    account for <1% of the validation/test gap (see analyze_fold_regime_composition),
    whereas fold *recency* tracks it closely.

    The two schemes are expected to behave differently by construction. CVExpand folds
    are chronological, so a decaying edge shows up as a downward trend in fold index.
    CPCV paths deliberately recombine blocks out of order, so the same decay is averaged
    away and its fold index carries no time information — which is precisely why CPCV
    reports a stationary-looking edge on non-stationary data.

    Returns ``{"per_fold": ..., "trend": ...}``; the trend frame carries one Spearman
    rho per (direction, granularity, scheme).
    """
    from scipy.stats import spearmanr

    grans = granularities or ["1d", "12h", "8h", "6h", "4h", "2h", "1h"]
    fold_rows, trend_rows = [], []
    for direction in directions:
        for g in grans:
            edge_dir = edge_root / m1 / m2 / direction / g
            for scheme in ("cvexpand", "cpcv"):
                s = _load_fold_samples(edge_dir, scheme)
                if s is None:
                    continue
                per = s.groupby("fold").apply(
                    lambda x: pd.Series({
                        "ret_05": float((x.loc[x["pred"] == 1, "ret"] - fee).mean()),
                        "ret_sel": float((x.loc[x["sel"] == 1, "ret"] - fee).mean()),
                        "prec_05": float(x.loc[x["pred"] == 1, "y_true"].mean()),
                        "n": len(x),
                        "date_end": x["date"].max(),
                    }), include_groups=False)
                for f, row in per.iterrows():
                    fold_rows.append({"direction": direction, "granularity": g,
                                      "scheme": scheme, "fold": int(f), **row.to_dict()})
                if len(per) > 2:
                    rho, pval = spearmanr(per.index.astype(int), per["ret_05"])
                    trend_rows.append({"direction": direction, "granularity": g,
                                       "scheme": scheme, "spearman_rho": float(rho),
                                       "p_value": float(pval), "n_folds": len(per)})
    out = {"per_fold": pd.DataFrame(fold_rows), "trend": pd.DataFrame(trend_rows)}
    if save_path is not None:
        save_path = Path(save_path); save_path.mkdir(parents=True, exist_ok=True)
        out["per_fold"].to_csv(save_path / "regime_table_D_per_fold.csv", index=False)
        out["trend"].to_csv(save_path / "regime_table_D_trend.csv", index=False)
    return out


# ┏━━━━━━━━━━ 4-scenario regime analysis (reusable across every M1 × M2) ━━━━━━━━━━┓
# Every scenario answers the same question — does this M2's edge depend on market
# regime? — but over a different evaluation and a different time grouping:
#
#   1 static (TEST)      held-out test trades; single period, no grouping needed
#   2 quarters (VAL)     CVExpand samples grouped by calendar quarter
#   3 folds (VAL)        CVExpand samples grouped by fold (folds ARE chronological)
#   4 cpcv (VAL)         CPCV samples grouped by (path x quarter)
#
# The grouping is not cosmetic. Regime share swings hugely between periods and
# correlates with how good that period was (r = -0.81 between a quarter's bull-share
# and its return), so a pooled average measures WHEN each regime occurred rather than
# the regime itself. Scenario 4 is the subtle one: CPCV paths are proper chronological
# walks, but each path still spans the ENTIRE timeline, so averaging within a path
# reproduces the pooled number exactly (-0.0247 vs -0.0246 on Kronos/rf/DOWN/1d) and
# inherits its bias. Adding the quarter dimension controls for time while preserving
# the 5-path replication that CPCV exists to provide.

def regime_scenarios(m1: str = "Kronos",
                     m2: str = "rf",
                     directions: Tuple[str, ...] = ("UP", "DOWN"),
                     granularities: Optional[List[str]] = None,
                     fee: float = 0.002,
                     jm_root: Path = _JM_GLOBAL_ROOT,
                     raw_root: Path = Path("/home/pablo/M2_DS/SecondaryModel/src/Data_MLA/Raw"),
                     edge_root: Path = Path("/home/pablo/M2_DS/SecondaryModel/src/Output/Analysis/Edge_NoCal"),
                     test_root: Path = Path("/home/pablo/M2_DS/SecondaryModel/src/Output"),
                     thres_folder: str = "Utility_Score_NoCal",
                     save_path: Optional[Path] = None,
                     verbose: bool = True) -> pd.DataFrame:
    """Run the 4-scenario regime analysis for one M1 x M2 pair.

    Returns one row per (direction, granularity, scenario) with the per-regime mean
    return of predicted trades, the bull-minus-bear gap, and selective precision.
    A regime effect is credible when scenario 1 (TEST) agrees with 2, 3 and 4.
    """
    grans = granularities or ["1d", "12h", "8h", "6h", "4h", "2h", "1h"]
    rows = []

    def _cells(df, group_cols, value="v"):
        """Per-regime means inside each group, then averaged over groups equally."""
        out = {}
        for reg in ("Bullish", "Bearish"):
            sub = df[df["regime_name"] == reg]
            if sub.empty:
                out[reg] = np.nan
                continue
            out[reg] = sub.groupby(group_cols)[value].mean().mean() if group_cols else sub[value].mean()
        return out

    for direction in directions:
        for g in grans:
            try:
                labels, _ = load_regime_labels(g, jm_root)
            except FileNotFoundError:
                if verbose:
                    print(f"  [skip] no jump model for {g}")
                continue
            name_of = _regime_direction_map(g, labels)
            labels = labels.assign(regime_name=labels["regime"].map(name_of))
            lab = labels[["asset", "date", "regime_name"]]

            # ── Scenario 1: static pipeline, held-out TEST ────────────────────
            tp = (test_root / m1 / m2 / direction / thres_folder / f"{g}_tp"
                  / "10_backtest_cash_trades.csv")
            if tp.exists():
                t = pd.read_csv(tp, parse_dates=["date"]).merge(lab, on=["asset", "date"], how="left")
                t = t[t["regime_name"].notna()].copy()
                t["v"] = t["return"]  # held-out trades are already net of the fee
                c = _cells(t, None)
                sel = t[t["m2_approved"]] if "m2_approved" in t else t.iloc[0:0]
                rows.append({"m1": m1, "m2": m2, "direction": direction, "granularity": g,
                             "scenario": "1_static_TEST", "bull": c["Bullish"], "bear": c["Bearish"],
                             "gap": c["Bullish"] - c["Bearish"], "n": len(t),
                             "sel_prec_bull": sel.loc[sel.regime_name == "Bullish", "label"].mean()
                                              if len(sel) else np.nan,
                             "sel_prec_bear": sel.loc[sel.regime_name == "Bearish", "label"].mean()
                                              if len(sel) else np.nan})

            # ── Scenarios 2-4: validation-window CV samples ───────────────────
            for scheme, scen, group in (("cvexpand", "2_quarters_VAL", ["q"]),
                                        ("cvexpand", "3_folds_VAL", ["fold"]),
                                        ("cpcv", "4_cpcv_path_x_quarter_VAL", ["fold", "q"])):
                s = _load_fold_samples(edge_root / m1 / m2 / direction / g, scheme)
                if s is None:
                    continue
                m = s.merge(lab, on=["asset", "date"], how="left")
                m = m[m["regime_name"].notna()].copy()
                m["v"] = m["ret"] - fee
                m["q"] = m["date"].dt.to_period("Q")
                p = m[m["pred"] == 1]
                if p.empty:
                    continue
                if scen.startswith("4"):
                    # Gap inside each (path, quarter), averaged over quarters then paths.
                    gaps = []
                    for _, gg in p.groupby("fold"):
                        qq = gg.groupby(["q", "regime_name"])["v"].mean().unstack().dropna()
                        if len(qq) and {"Bullish", "Bearish"}.issubset(qq.columns):
                            gaps.append((qq["Bullish"] - qq["Bearish"]).mean())
                    gap = float(np.nanmean(gaps)) if gaps else np.nan
                    c = _cells(p, ["fold", "q"])
                else:
                    c = _cells(p, group)
                    gap = c["Bullish"] - c["Bearish"]
                # Selective precision must use the SAME grouping as the returns above:
                # pooling it would reintroduce exactly the time confound the grouping exists
                # to remove (and would make scenarios 2 and 3 report identical values).
                sel = m[m["sel"] == 1]
                sp = _cells(sel, ["fold", "q"] if scen.startswith("4") else group, value="y_true")
                rows.append({"m1": m1, "m2": m2, "direction": direction, "granularity": g,
                             "scenario": scen, "bull": c["Bullish"], "bear": c["Bearish"],
                             "gap": gap, "n": len(p),
                             "sel_prec_bull": sp["Bullish"], "sel_prec_bear": sp["Bearish"]})

    out = pd.DataFrame(rows)
    if not out.empty:
        out["winner"] = np.where(out["gap"] > 0, "BULL", "BEAR")
    if not out.empty:
        if save_path is not None:
            _write_by_granularity(out, Path(save_path), verbose=verbose)
        else:
            _write_split_by_period(out, "scenarios", m1, m2,
                                   out["scenario"].str.contains("TEST"), verbose)
    return out


# ┏━━━━━━━━━━ Significance tests for the regime effect ━━━━━━━━━━┓
# Reviewers R5-W2 and R3 asked explicitly for paired comparisons and statistical
# significance. The obvious test — a t-test over per-trade returns — is invalid here:
# a 7-bar horizon means consecutive trades share most of their price path, so the
# effective sample size is far below n (at 1h, adjacent bars of the 63-day feature
# correlate at 0.9996). All three tests below are built to respect that dependence.

def _test_trades_with_regime(m1, m2, direction, granularity, thres_folder, test_root, jm_root):
    """Held-out test trades joined to their causal regime label, sorted by date."""
    tp = (Path(test_root) / m1 / m2 / direction / thres_folder
          / f"{granularity}_tp" / "10_backtest_cash_trades.csv")
    if not tp.exists():
        return None
    labels, _ = load_regime_labels(granularity, jm_root)
    name_of = _regime_direction_map(granularity, labels)
    labels = labels.assign(regime_name=labels["regime"].map(name_of))
    t = (pd.read_csv(tp, parse_dates=["date"])
           .merge(labels[["asset", "date", "regime_name"]], on=["asset", "date"], how="left"))
    t = t[t["regime_name"].notna()].sort_values("date").copy()
    return t if len(t) else None


def _block_boot(df, statfn, n_boot=1000, block_days=7, seed=42):
    """Moving-block bootstrap over calendar days; returns (lo, hi, p, n_draws).

    Blocks are cut on days rather than rows so a block always spans at least the
    forecast horizon in time, keeping overlapping trades together. Resampling
    individual trades would treat a 7-bar horizon's worth of shared price path as
    independent evidence and return an interval several times too narrow.
    """
    rng = np.random.default_rng(seed)
    day = df["date"].dt.normalize()
    days = np.array(sorted(day.unique()))
    starts = np.arange(0, len(days) - block_days + 1)
    if len(starts) < 2:
        return None
    by = {d: g for d, g in df.groupby(day)}
    n = max(1, len(days) // block_days)
    out = []
    for _ in range(n_boot):
        picks = rng.choice(starts, size=n, replace=True)
        parts = [by[d] for s in picks for d in days[s:s + block_days] if d in by]
        if not parts:
            continue
        v = statfn(pd.concat(parts, ignore_index=True))
        if v is not None and np.isfinite(v):
            out.append(float(v))
    if len(out) < 50:
        return None
    a = np.asarray(out)
    lo, hi = np.percentile(a, [2.5, 97.5])
    p = 2 * min((a <= 0).mean(), (a >= 0).mean())
    return float(lo), float(hi), float(p), int(len(a))


def test_gap_by_quarter(m1="Kronos", m2="rf", directions=("UP", "DOWN"),
                        granularities=None, fee=0.002, n_boot=1000,
                        test_root=Path("/home/pablo/M2_DS/SecondaryModel/src/Output"),
                        thres_folder="Utility_Score_NoCal", jm_root=_JM_GLOBAL_ROOT,
                        save_path=None, verbose=True) -> pd.DataFrame:
    """Held-out bull-minus-bear gap, reported OVERALL and per calendar quarter.

    The quarterly rows are a descriptive robustness check, not a second significance
    test: they show whether the overall effect is spread across the test window or
    leans on one quarter. Watch `n_bull` — the test period is ~84% bearish, so a
    quarterly slice at coarse granularity can fall to a handful of trades, and a
    "significant" CI built on those is meaningless.
    """
    grans = granularities or ["1d", "12h", "8h", "6h", "4h", "2h", "1h"]
    rows = []
    for direction in directions:
        for g in grans:
            try:
                t = _test_trades_with_regime(m1, m2, direction, g, thres_folder, test_root, jm_root)
            except FileNotFoundError:
                continue
            if t is None:
                continue

            for metric in ("precision", "return"):
                def stat(v, _m=metric):
                    a = v[v["m2_approved"]] if "m2_approved" in v else v
                    col = a["label"].astype(float) if _m == "precision" else a["return"]  # already net of the fee
                    b = col[a["regime_name"] == "Bullish"]
                    e = col[a["regime_name"] == "Bearish"]
                    return float(b.mean() - e.mean()) if len(b) and len(e) else None

                for scope, sub in [("OVERALL", t)] + [(str(q), s) for q, s in
                                                      t.groupby(t["date"].dt.to_period("Q"))]:
                    ap = sub[sub["m2_approved"]] if "m2_approved" in sub else sub
                    r = _block_boot(sub, stat, n_boot=n_boot)
                    rows.append({
                        "m1": m1, "m2": m2, "direction": direction, "granularity": g,
                        "metric": metric, "scope": scope,
                        "n_bull": int((ap["regime_name"] == "Bullish").sum()),
                        "n_bear": int((ap["regime_name"] == "Bearish").sum()),
                        "gap": stat(sub),
                        "ci_low": r[0] if r else np.nan, "ci_high": r[1] if r else np.nan,
                        "p_value": r[2] if r else np.nan,
                        "excludes_zero": bool(r and (r[0] > 0 or r[1] < 0)),
                    })
    out = pd.DataFrame(rows)
    if not out.empty:
        d = Path(save_path) if save_path is not None else regime_out_dir("test_quarter", m1, m2, "test")
        _write_by_granularity(out, d, verbose=verbose)
    return out


def m2_vs_m1_by_regime(m1="Kronos", m2="rf", directions=("UP", "DOWN"),
                       granularities=None, fee=0.002, n_boot=1000,
                       test_root=Path("/home/pablo/M2_DS/SecondaryModel/src/Output"),
                       thres_folder="Utility_Score_NoCal", jm_root=_JM_GLOBAL_ROOT,
                       save_path=None, verbose=True) -> pd.DataFrame:
    """M2's gain over M1 computed SEPARATELY inside each regime, on held-out test.

    Pooled over the whole test period this comparison is not significant, because the
    window is ~84% bearish and the gain concentrates in bullish bars — pooling dilutes
    it. Splitting by regime first recovers the effect, and keeps the comparison paired:
    M1 and M2 are scored on the same bars within the same bootstrap resample, so
    market-wide noise cancels instead of swamping the difference.
    """
    grans = granularities or ["1d", "12h", "8h", "6h", "4h", "2h", "1h"]
    rows = []
    for direction in directions:
        for g in grans:
            try:
                t = _test_trades_with_regime(m1, m2, direction, g, thres_folder, test_root, jm_root)
            except FileNotFoundError:
                continue
            if t is None or "m2_approved" not in t:
                continue
            for metric in ("precision", "return"):
                for regime in ("Bullish", "Bearish"):
                    def stat(v, _m=metric, _r=regime):
                        s = v[v["regime_name"] == _r]
                        ap = s[s["m2_approved"]]
                        if not len(s) or not len(ap):
                            return None
                        if _m == "precision":
                            return float(ap["label"].mean() - s["label"].mean())
                        return float(ap["return"].mean() - s["return"].mean())  # already net of the fee

                    sub = t[t["regime_name"] == regime]
                    r = _block_boot(t, stat, n_boot=n_boot)
                    # A signal is COUNTER-trend when M1's direction opposes the prevailing
                    # regime (UP called in a bearish market, DOWN called in a bullish one).
                    # That is where M1 is least reliable and so where a reliability filter
                    # has the most to remove — the single mechanism behind both directions'
                    # regime preference, rather than two unrelated findings.
                    aligned = ((direction.upper() == "UP" and regime == "Bullish") or
                               (direction.upper() == "DOWN" and regime == "Bearish"))
                    rows.append({
                        "m1": m1, "m2": m2, "direction": direction, "granularity": g,
                        "metric": metric, "regime": regime,
                        "alignment": "with_trend" if aligned else "counter_trend",
                        "n": int(len(sub)), "n_approved": int(sub["m2_approved"].sum()),
                        "delta": stat(t),
                        "ci_low": r[0] if r else np.nan, "ci_high": r[1] if r else np.nan,
                        "p_value": r[2] if r else np.nan,
                        "excludes_zero": bool(r and (r[0] > 0 or r[1] < 0)),
                    })
    out = pd.DataFrame(rows)
    if not out.empty:
        d = Path(save_path) if save_path is not None else regime_out_dir("m2_vs_m1", m1, m2, "test")
        _write_by_granularity(out, d, verbose=verbose)
    return out


def permutation_regime_test(m1="Kronos", m2="rf", direction="DOWN", granularity="1d",
                            scheme="cpcv", fee=0.002, metric="return", n_perm=1000, seed=42,
                            edge_root=Path("/home/pablo/M2_DS/SecondaryModel/src/Output/Analysis/Edge_NoCal"),
                            jm_root=_JM_GLOBAL_ROOT) -> dict:
    """Permutation test against a composition null: is the gap more than Simpson's paradox?

    Regime labels are shuffled *within each quarter*, so every quarter keeps its exact
    bull/bear counts (2024Q4 really is 3614 bull vs 642 bear) while any genuine
    regime->return link is destroyed. The null distribution therefore already contains
    the composition imbalance that produced our earlier false "M2 DOWN wins in bearish"
    result; beating it means the effect is not that artifact.

    Shuffling across the whole period instead would break the imbalance and give a
    falsely easy null.
    """
    rng = np.random.default_rng(seed)
    s = _load_fold_samples(edge_root / m1 / m2 / direction / granularity, scheme)
    if s is None:
        return {"error": "no per-sample CSV"}
    m = _attach_regimes(s, granularity, jm_root, Path("Data_MLA/Raw"))
    m = m[m["regime_name"].notna()].copy()
    m["q"] = m["date"].dt.to_period("Q")
    if metric == "precision":
        p = m[m["pred"] == 1].assign(v=lambda d: d["y_true"].astype(float)).copy()
    else:
        p = m[m["pred"] == 1].assign(v=lambda d: d["ret"] - fee).copy()
    if p.empty:
        return {"error": "no predicted trades"}

    group = ["fold", "q"] if scheme == "cpcv" else ["fold"]

    def _gap(labels_col):
        d = p.assign(_r=labels_col)
        cell = d.groupby(group + ["_r"])["v"].mean().unstack()
        if not {"Bullish", "Bearish"}.issubset(cell.columns):
            return np.nan
        cell = cell.dropna()
        return float((cell["Bullish"] - cell["Bearish"]).mean()) if len(cell) else np.nan

    observed = _gap(p["regime_name"].to_numpy())
    if not np.isfinite(observed):
        return {"error": "observed gap undefined"}

    q_idx = [np.flatnonzero(p["q"].to_numpy() == q) for q in p["q"].unique()]
    base = p["regime_name"].to_numpy()
    null = []
    for _ in range(n_perm):
        shuffled = base.copy()
        for idx in q_idx:                      # permute inside each quarter only
            shuffled[idx] = rng.permutation(shuffled[idx])
        g = _gap(shuffled)
        if np.isfinite(g):
            null.append(g)
    if len(null) < 50:
        return {"error": f"only {len(null)} usable permutations"}
    null = np.asarray(null)
    # Two-sided, with the +1 correction so p is never exactly 0.
    pval = (np.sum(np.abs(null) >= abs(observed)) + 1) / (len(null) + 1)
    return {"test": "permutation", "scheme": scheme, "metric": metric, "observed_gap": observed,
            "null_mean": float(null.mean()), "null_std": float(null.std()),
            "n_perm": int(len(null)), "p_value": float(pval)}


def permutation_regime_sweep(m1: str = "Kronos", m2: str = "rf",
                             directions=("DOWN", "UP"),
                             granularities=None,
                             schemes=("cpcv", "cvexpand"),
                             metrics=("precision", "return"),
                             fee: float = 0.002, n_perm: int = 1000, seed: int = 42,
                             edge_root: Path = Path(
                                 "/home/pablo/M2_DS/SecondaryModel/src/Output/Analysis/Edge_NoCal"),
                             jm_root: Path = _JM_GLOBAL_ROOT,
                             save: bool = True, verbose: bool = True) -> pd.DataFrame:
    """Run the composition-null permutation test over the whole train+val grid.

    ``permutation_regime_test`` answers one (granularity, direction, scheme, metric)
    cell; this sweeps the grid and writes the table the paper's cross-validated
    figure reads. It is the only significance test applied to the CV folds -- the
    earlier per-cell Wilcoxon saw 5-6 observations and could not resolve anything,
    so it was dropped rather than reported alongside a test that can.

    Grouping differs by scheme and that is deliberate: a CPCV path spans every
    quarter, so its cells are fold x quarter, while expanding-window folds are
    already chronologically disjoint and a fold is itself a period.
    """
    grans = list(granularities) if granularities else ["1d", "12h", "8h", "6h",
                                                       "4h", "2h", "1h", "30m"]
    rows = []
    for direction in directions:
        for g in grans:
            for scheme in schemes:
                for metric in metrics:
                    res = permutation_regime_test(
                        m1=m1, m2=m2, direction=direction, granularity=g,
                        scheme=scheme, fee=fee, metric=metric, n_perm=n_perm,
                        seed=seed, edge_root=edge_root, jm_root=jm_root)
                    if "error" in res:
                        continue
                    rows.append({"m1": m1, "m2": m2, "direction": direction,
                                 "granularity": g, **res})
    out = pd.DataFrame(rows)
    if save and not out.empty:
        d = regime_out_dir("significance", m1, m2, "train_val")
        _write_by_granularity(out, d, verbose=verbose)
    if verbose:
        n_sig = int((out["p_value"] < 0.05).sum()) if len(out) else 0
        print(f"  [permutation_regime_sweep] {len(out)} cells, {n_sig} with p<0.05")
    return out


# ┏━━━━━━━━━━ Regime-conditional cash backtest ━━━━━━━━━━┓
# The headline claim is that M2 does better in counter-trend bars. That is measured
# per trade; this asks the portfolio question instead — what would the cash curve look
# like if you only took the counter-trend trades? Equity runs over the FULL test
# timeline in every variant, so the excluded regime shows up as flat cash and the
# totals stay directly comparable to the unconditional backtest.

def regime_conditional_backtest(m1: str = "Kronos", m2: str = "rf",
                                directions=("DOWN", "UP"),
                                granularities=None,
                                test_root: Path = Path(
                                    "/home/pablo/M2_DS/SecondaryModel/src/Output"),
                                thres_folder: str = "Utility_Score_NoCal",
                                jm_root: Path = _JM_GLOBAL_ROOT,
                                horizon: int = 7,
                                write_json: bool = True,
                                verbose: bool = True) -> pd.DataFrame:
    """Cash-engine metrics for counter-trend / with-trend / all trades.

    Reuses ``_build_cash_equity`` and the same Sharpe/drawdown helpers the production
    backtest uses, so a row here is directly comparable to 10_backtest_cash_ROI.txt.
    Results are merged into each config's analysis_summary.json under the key
    ``<m2>_backtest_regime_split``; the existing keys are left untouched.
    """
    from Utils.backtest.engine import (_build_cash_equity, _calc_sharpe,
                                       _calc_drawdown, _equity_horizon_returns,
                                       _annualization_factor)

    grans = list(granularities) if granularities else ["1d", "12h", "8h", "6h",
                                                       "4h", "2h", "1h"]
    rows = []
    for direction in directions:
        counter = "Bullish" if direction.upper() == "DOWN" else "Bearish"
        for g in grans:
            cfg_dir = Path(test_root) / m1 / m2 / direction / thres_folder / f"{g}_tp"
            tp = cfg_dir / "10_backtest_cash_trades.csv"
            if not tp.exists():
                continue
            t = pd.read_csv(tp, parse_dates=["date"])
            labels, _ = load_regime_labels(g, jm_root)
            name_of = _regime_direction_map(g, labels)
            lab = labels.assign(regime_name=labels["regime"].map(name_of))[
                ["asset", "date", "regime_name"]]
            t = t.merge(lab, on=["asset", "date"], how="left")
            t = t[t["regime_name"].notna()].copy()
            if t.empty:
                continue

            # One timeline for every variant — this is what makes the excluded
            # regime read as flat cash rather than a compressed series.
            full_idx = pd.DatetimeIndex(sorted(t["date"].unique()))
            ann_bar = _annualization_factor(g)
            ann_horizon = np.sqrt(ann_bar ** 2 / horizon)

            approved = t[t["m2_approved"]] if "m2_approved" in t else t
            subsets = {
                "all":           approved,
                "counter_trend": approved[approved["regime_name"] == counter],
                "with_trend":    approved[approved["regime_name"] != counter],
            }
            for scope, sub in subsets.items():
                eq, _ = _build_cash_equity(sub, full_idx, horizon)
                h = (_equity_horizon_returns(eq, horizon)
                     if len(eq) > horizon else np.array([]))
                rows.append({
                    "m1": m1, "m2": m2, "direction": direction, "granularity": g,
                    "scope": scope, "regime": counter if scope == "counter_trend" else
                             ("not " + counter if scope == "with_trend" else "both"),
                    "n_trades": int(len(sub)),
                    "win_rate": float(sub["label"].mean() * 100) if len(sub) else np.nan,
                    "total_return_pct": float((eq.iloc[-1] - 1) * 100) if len(eq) else 0.0,
                    "max_drawdown_pct": float(_calc_drawdown(eq.values) * 100) if len(eq) else 0.0,
                    "sharpe": float(_calc_sharpe(h, ann_horizon)),
                })

            if write_json:
                jp = cfg_dir / "analysis_summary.json"
                try:
                    blob = json.loads(jp.read_text()) if jp.exists() else {}
                except (OSError, json.JSONDecodeError):
                    blob = {}
                here = [r for r in rows if r["granularity"] == g
                        and r["direction"] == direction]
                blob[f"{m2}_backtest_regime_split"] = {
                    r["scope"]: {k: r[k] for k in
                                 ("regime", "n_trades", "win_rate", "total_return_pct",
                                  "max_drawdown_pct", "sharpe")}
                    for r in here}
                jp.write_text(json.dumps(blob, indent=2))

    out = pd.DataFrame(rows)
    if verbose and not out.empty:
        n_cfg = out.groupby(["direction", "granularity"]).ngroups
        print(f"  [regime_conditional_backtest] {m1}/{m2}: {n_cfg} configs, "
              f"{len(out)} rows written")
    return out


# ┏━━━━━━━━━━ Robust significance suite (tests B, C, E, G) ━━━━━━━━━━┓
# Earlier per-cell designs (one Wilcoxon or bootstrap per granularity) saw only 5-6
# observations each and could not resolve the effect. These four slice the SAME held-out
# trades into more independent units and answer four distinct objections:
#
#   B  by (granularity, quarter)  -> "it is one lucky quarter"
#   C  by asset                   -> "it is just Bitcoin"
#   E  permutation, per-asset     -> "it is a regime-composition artifact"
#   G  bootstrap over assets      -> "how big is it, with what interval"
#
# A naive per-trade test is invalid here: a 7-bar horizon makes consecutive trades share
# most of their price path, so every design below pairs or resamples at a level that
# respects that overlap.

def _regime_trades(m1, m2, direction, granularity, thres_folder, test_root, jm_root, fee):
    """Held-out M2-approved trades joined to their causal regime label."""
    tp = (Path(test_root) / m1 / m2 / direction / thres_folder
          / f"{granularity}_tp" / "10_backtest_cash_trades.csv")
    if not tp.exists():
        return None
    labels, _ = load_regime_labels(granularity, jm_root)
    name_of = _regime_direction_map(granularity, labels)
    lab = labels.assign(regime_name=labels["regime"].map(name_of))[
        ["asset", "date", "regime_name"]]
    t = pd.read_csv(tp, parse_dates=["date"]).merge(lab, on=["asset", "date"], how="left")
    t = t[t["regime_name"].notna()].copy()
    if "m2_approved" in t:
        t = t[t["m2_approved"]]
    if t.empty:
        return None
    t["net"] = t["return"]  # the cash engine already charged the fee
    t["q"] = t["date"].dt.to_period("Q")
    t["granularity"] = granularity
    return t


def _regime_trades_cpcv(m1, m2, direction, granularity, edge_root, jm_root, fee,
                        raw_root=Path("Data_MLA/Raw"), dedup=False):
    """CPCV M2-selected samples joined to their causal regime label.

    Same columns as ``_regime_trades`` so the tests below run unchanged. Every
    reconstructed path is kept: each is a separate out-of-fold evaluation.
    """
    f = Path(edge_root) / m1 / m2 / direction / granularity / "cpcv_samples.csv"
    if not f.exists():
        return None
    s = pd.read_csv(f, parse_dates=["date"])
    t = _attach_regimes(s, granularity, jm_root, raw_root)
    t = t[t["regime_name"].notna() & t["sel"].astype(bool)].copy()
    if t.empty:
        return None
    t = t.rename(columns={"y_true": "label", "ret": "return"})
    if dedup:
        # One row per bar: the paths share its outcome, so repeating it would count
        # one market move as several trades. ``w`` keeps the share of paths selecting it.
        n_paths = s["path"].nunique()
        t = (t.groupby(["asset", "date"], as_index=False)
               .agg(label=("label", "first"), **{"return": ("return", "first")},
                    regime_name=("regime_name", "first"), w=("path", "size")))
        t["w"] = t["w"] / n_paths
    t["net"] = t["return"] - fee
    t["q"] = t["date"].dt.to_period("Q")
    t["granularity"] = granularity
    return t


def _cell_gaps(df, keys, counter_regime, min_n=5):
    """Bull-minus-bear gap per group, keeping only groups with min_n trades each side."""
    other = "Bearish" if counter_regime == "Bullish" else "Bullish"
    out = []
    for k, g in df.groupby(keys, observed=True):
        c = g[g["regime_name"] == counter_regime]
        w = g[g["regime_name"] == other]
        if len(c) >= min_n and len(w) >= min_n:
            if "w" in g:
                # weighted rows (de-duplicated CPCV bars): same means as the pooled rows
                avg = lambda x, col: float(np.average(x[col], weights=x["w"]))
            else:
                avg = lambda x, col: float(x[col].mean())
            rec = {"key": k, "n_counter": len(c), "n_with": len(w),
                   "prec_gap": avg(c, "label") - avg(w, "label"),
                   "ret_gap": avg(c, "net") - avg(w, "net")}
            out.append(rec)
    return pd.DataFrame(out)


def robust_significance_suite(m1="Kronos", m2="rf", direction="DOWN",
                              granularities=None, fee=0.002,
                              n_perm=1000, n_boot=10000, min_n=5, seed=42,
                              test_root=Path("/home/pablo/M2_DS/SecondaryModel/src/Output"),
                              thres_folder="Utility_Score_NoCal",
                              jm_root=_JM_GLOBAL_ROOT,
                              save_path=None, verbose=True, window="test",
                              edge_root=Path("Output/Analysis/Edge_NoCal"),
                              dedup=True) -> pd.DataFrame:
    """Tests B, C, E and G on the counter-trend regime gap for one direction.

    ``window`` selects the held-out trades ("test") or the CPCV out-of-fold
    samples ("cpcv"). For CPCV, ``dedup`` keeps one weighted row per bar, since
    the five paths share each bar's outcome; it has no effect on "test".

    ``direction`` fixes which regime is counter-trend: M2 DOWN is counter-trend in
    Bullish bars, M2 UP in Bearish ones. The two directions are NOT pooled — doing so
    dilutes whichever direction carries the effect with the other's noise.
    """
    from scipy.stats import wilcoxon

    rng = np.random.default_rng(seed)
    grans = granularities or ["1d", "12h", "8h", "6h", "4h", "2h", "1h"]
    counter = "Bullish" if direction.upper() == "DOWN" else "Bearish"

    frames = []
    for g in grans:
        try:
            if window == "cpcv":
                t = _regime_trades_cpcv(m1, m2, direction, g, edge_root, jm_root, fee,
                                        dedup=dedup)
            else:
                t = _regime_trades(m1, m2, direction, g, thres_folder, test_root, jm_root, fee)
        except FileNotFoundError:
            continue
        if t is not None:
            frames.append(t)
    if not frames:
        return pd.DataFrame([{"error": "no trades found"}])
    A = pd.concat(frames, ignore_index=True)

    rows = []

    def _wilcox(vals, test, metric, unit, extra=None):
        v = np.asarray([x for x in vals if np.isfinite(x)])
        rec = {"m1": m1, "m2": m2, "direction": direction, "test": test,
               "metric": metric, "unit": unit, "n": len(v),
               "mean_gap": float(v.mean()) if len(v) else np.nan,
               "median_gap": float(np.median(v)) if len(v) else np.nan,
               "n_positive": int((v > 0).sum()) if len(v) else 0}
        if len(v) >= 6 and not np.allclose(v, 0):
            stat, p = wilcoxon(v)
            rec["statistic"], rec["p_value"] = float(stat), float(p)
        else:
            rec["error"] = f"only {len(v)} usable observations"
        if extra:
            rec.update(extra)
        rows.append(rec)

    # ── B: one observation per (granularity, quarter) ──────────────────────
    b = _cell_gaps(A, ["granularity", "q"], counter, min_n=20)
    for metric, col in (("precision", "prec_gap"), ("return", "ret_gap")):
        _wilcox(b[col] if len(b) else [], "B_by_gran_quarter", metric, "(granularity, quarter)")

    # ── C: one observation per asset (gaps averaged over granularities) ────
    # Averaging first matters: one coin measured at 3 granularities is one fact, not
    # three, and leaving the cells unaveraged inflates n with repeated assets.
    cell = _cell_gaps(A, ["asset", "granularity"], counter, min_n=min_n)
    per_asset = pd.DataFrame()
    if len(cell):
        cell["asset"] = cell["key"].map(lambda k: k[0])
        per_asset = cell.groupby("asset")[["prec_gap", "ret_gap"]].mean()
    for metric, col in (("precision", "prec_gap"), ("return", "ret_gap")):
        _wilcox(per_asset[col] if len(per_asset) else [], "C_by_asset", metric, "asset")

    # ── E: permutation against a regime-composition null ───────────────────
    # Labels are shuffled WITHIN each (asset, quarter), so every cell keeps its exact
    # bull/bear counts and its returns; only the regime->outcome link is destroyed. The
    # null therefore already contains the composition imbalance that can manufacture a
    # spurious gap, and beating it rules that explanation out.
    if len(per_asset):
        idx = A.groupby(["asset", "q"], observed=True).indices
        base = A["regime_name"].to_numpy()
        for metric, col in (("precision", "prec_gap"), ("return", "ret_gap")):
            obs = float(per_asset[col].mean())
            null = []
            for _ in range(n_perm):
                sh = base.copy()
                for _, ii in idx.items():
                    sh[ii] = rng.permutation(sh[ii])
                gg = _cell_gaps(A.assign(regime_name=sh), ["asset", "granularity"],
                                counter, min_n=min_n)
                if len(gg):
                    gg["asset"] = gg["key"].map(lambda k: k[0])
                    null.append(gg.groupby("asset")[col].mean().mean())
            null = np.asarray([x for x in null if np.isfinite(x)])
            rec = {"m1": m1, "m2": m2, "direction": direction, "test": "E_permutation",
                   "metric": metric, "unit": "1000 label shuffles", "n": len(null),
                   "mean_gap": obs}
            if len(null) >= 50:
                rec["null_mean"] = float(null.mean())
                rec["null_std"] = float(null.std())
                # Distance from the null's centre, not from zero: the composition
                # null is not centred on zero, which is the point of the test.
                mu = null.mean()
                rec["p_value"] = float((np.sum(np.abs(null - mu) >= abs(obs - mu)) + 1)
                                       / (len(null) + 1))
                rec["p_value_vs_zero"] = float((np.sum(np.abs(null) >= abs(obs)) + 1)
                                               / (len(null) + 1))
            else:
                rec["error"] = f"only {len(null)} usable permutations"
            rows.append(rec)

        # ── G: bootstrap over assets -> effect size with interval ──────────
        for metric, col in (("precision", "prec_gap"), ("return", "ret_gap")):
            v = per_asset[col].dropna().to_numpy()
            if len(v) >= 6:
                bs = [rng.choice(v, size=len(v), replace=True).mean() for _ in range(n_boot)]
                lo, hi = np.percentile(bs, [2.5, 97.5])
                rows.append({"m1": m1, "m2": m2, "direction": direction,
                             "test": "G_bootstrap_assets", "metric": metric,
                             "unit": "asset", "n": len(v), "mean_gap": float(v.mean()),
                             "ci_low": float(lo), "ci_high": float(hi),
                             "excludes_zero": bool(lo > 0 or hi < 0)})

    out = pd.DataFrame(rows)
    if not out.empty:
        d = Path(save_path) if save_path is not None else regime_out_dir(
            "robust_significance", m1, m2, "train_val" if window == "cpcv" else "test")
        d.mkdir(parents=True, exist_ok=True)
        # Each call covers one direction, so the filenames carry it: writing to a
        # shared name made a later UP run silently destroy the DOWN results.
        tag = direction.upper()
        out.to_csv(d / f"all_granularities_{tag}.csv", index=False)
        merged = [pd.read_csv(f) for f in sorted(d.glob("all_granularities_*.csv"))]
        pd.concat(merged, ignore_index=True).to_csv(d / "all_granularities.csv",
                                                    index=False)
        if len(cell):
            cell.to_csv(d / f"per_asset_granularity_cells_{tag}.csv", index=False)
        if len(b):
            b.to_csv(d / f"per_gran_quarter_cells_{tag}.csv", index=False)
        if verbose:
            print(f"  [saved] {d}")
    return out


# ════════════════════════════════════════════════════════════════════════════
#  Counter-trend evidence figure  (tornado + significance lanes)
# ════════════════════════════════════════════════════════════════════════════
# One row per backbone x M2. Left/right of a shared spine: the return and
# precision advantage of counter-trend over with-trend bars, each with its 95%
# bootstrap CI. Right panel: the same effect seen by three tests that fail
# differently, so agreement across them is the claim.
#
# Reads the per-cell CSVs written by ``robust_significance_suite``:
#   RegimeAnalysis/{M1}/{m2}/test/robust_significance/all_granularities_{DIR}.csv

_CT_COL = {"Kronos": "#1F6FB4", "Fincast": "#2E8B57",
           "Chronos2": "#8A6BBE", "Tirex": "#D08A2E"}
_CT_NICE = {"Kronos": "Kronos", "Fincast": "Fincast",
            "Chronos2": "Chronos-2", "Tirex": "Tirex"}
_CT_M2N = {"rf": "Random Forest", "tabicl": "TabICL", "tabpfn": "TabPFN",
           "autogluon": "AutoGluon", "ctts": "CTTS"}
_CT_ORDER = {"Kronos": 0, "Fincast": 1, "Chronos2": 2, "Tirex": 3}
_CT_PFLOOR = 1e-4          # permutation p-values are floored at 1/(n_perm+1)


def _ct_load(direction: str, root: Path = REGIME_ANALYSIS_ROOT) -> tuple:
    """Collect the per-cell G/E/C/B results for one direction, both metrics.

    Returns ``(precision_df, return_df)`` with identical row order. Gaps are
    rescaled to percentage points (precision) and percent per trade (return).
    """
    files = sorted(Path(root).glob(
        f"*/*/test/robust_significance/all_granularities_{direction.upper()}.csv"))
    if not files:
        raise FileNotFoundError(
            f"no robust_significance results for {direction} under {root}; "
            "run robust_significance_suite() first")
    A = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    A = A[A["direction"].str.upper() == direction.upper()]

    def _one(metric: str) -> pd.DataFrame:
        rows = []
        for (m1, m2), g in A[A["metric"] == metric].groupby(["m1", "m2"]):
            r = {"m1": m1, "m2": m2}
            for test, pre in (("B_by_gran_quarter", "B"), ("C_by_asset", "C")):
                s = g[g["test"] == test]
                r[pre + "_p"] = float(s["p_value"].iloc[0]) if len(s) else np.nan
            # composition null: its p-value is floored, so carry the standardised
            # distance instead - how many sigma the observed gap sits outside it.
            s = g[g["test"] == "E_permutation"]
            if len(s) and np.isfinite(s["null_std"].iloc[0]) and s["null_std"].iloc[0] > 0:
                r["E_z"] = float((s["mean_gap"].iloc[0] - s["null_mean"].iloc[0])
                                 / s["null_std"].iloc[0])
            else:
                r["E_z"] = np.nan
            s = g[g["test"] == "G_bootstrap_assets"]
            if len(s):
                r.update(gap=float(s["mean_gap"].iloc[0]) * 100.0,
                         lo=float(s["ci_low"].iloc[0]) * 100.0,
                         hi=float(s["ci_high"].iloc[0]) * 100.0,
                         excl0=bool(s["excludes_zero"].iloc[0]))
            rows.append(r)
        return pd.DataFrame(rows)

    P, R = _one("precision"), _one("return")
    P["_o"] = P["m1"].map(_CT_ORDER)
    P = P.sort_values(["_o", "gap"], ascending=[True, False]).reset_index(drop=True)
    R = R.set_index(["m1", "m2"]).loc[list(zip(P["m1"], P["m2"]))].reset_index()
    return P, R


def plot_ct_significance_two_panel(root: Path = REGIME_ANALYSIS_ROOT,
                                   save_path=None, dpi: int = 600,
                                   backbones=("Kronos", "Fincast", "Chronos2", "Tirex"),
                                   m2_models=("rf", "tabpfn", "autogluon", "tabicl", "ctts")) -> Path:
    """Effect size of the counter-trend gap for every backbone-estimator pair, two windows.

    Left: up predictions on the cross-validated window (train_val/robust_significance);
    right: down predictions on the held-out window (test/robust_significance). Each row is
    one pair, ordered within its backbone by the held-out down precision gap. Precision
    gaps are read outward to the left of zero and return gaps to the right; markers are the
    mean per-asset gap with its bootstrap 95% interval (test G), hollow and dashed when the
    interval includes zero.
    """
    from matplotlib.lines import Line2D
    M2N = {"rf": "Random Forest", "tabpfn": "TabPFN", "autogluon": "AutoGluon",
           "tabicl": "TabICL", "ctts": "CTTS"}
    PMAX, RMAX = 26.0, 3.0

    def _g(b, m, sub, d, metric):
        df = pd.read_csv(Path(root) / b / m / sub / "robust_significance" / f"all_granularities_{d}.csv")
        r = df[(df["test"] == "G_bootstrap_assets") & (df["metric"] == metric)].iloc[0]
        return 100 * r["mean_gap"], 100 * r["ci_low"], 100 * r["ci_high"]

    rows = []
    for b in backbones:
        order = sorted(m2_models, key=lambda m: -_g(b, m, "test", "DOWN", "precision")[0])
        rows += [(b, m) for m in order]
    panels = (("train_val", "UP", "CPCV  \u00b7  M2 UP"), ("test", "DOWN", "Held-out  \u00b7  M2 DOWN"))

    fig, axes = plt.subplots(1, 2, figsize=(8.98, 5.51), sharey=True)
    y = np.arange(len(rows))[::-1]
    for ax, (sub, d, title) in zip(axes, panels):
        for yi, (b, m) in zip(y, rows):
            c = _GR_COL[b]
            for metric, scale in (("precision", -1.0 / PMAX), ("return", 1.0 / RMAX)):
                mu, lo, hi = _g(b, m, sub, d, metric)
                inc = lo <= 0 <= hi
                ax.plot([lo * scale, hi * scale], [yi, yi], color=c, lw=1.2,
                        ls=(0, (3, 2)) if inc else "-", zorder=3)
                for e in (lo, hi):
                    ax.plot([e * scale] * 2, [yi - 0.18, yi + 0.18], color=c, lw=1.2, zorder=3)
                ax.plot(mu * scale, yi, marker="s", ms=6, color=c,
                        mfc="white" if inc else c, mew=1.2, zorder=4)
        for k in range(1, len(backbones)):
            ax.axhline(len(rows) - 5 * k - 0.5, color="#DDDDDD", lw=0.6, zorder=0)
        ax.axvline(0, color="black", lw=0.8, zorder=2)
        ax.set_xlim(-1, 1)
        ticks = [-1, -0.75, -0.5, -0.25, 0, 1 / 3, 2 / 3, 1]
        ax.set_xticks(ticks)
        ax.set_xticklabels(["+26", "+19.5", "+13", "+6.5", "0", "+1", "+2", "+3"], fontsize=8)
        ax.set_title(title, fontsize=9, fontweight="bold")
        ax.text(-0.5, -0.085, "Precision gap (pp)", transform=ax.get_xaxis_transform(), ha="center", fontsize=8)
        ax.text(0.5, -0.085, "Return gap (pp)", transform=ax.get_xaxis_transform(), ha="center", fontsize=8)
        for sp in ("top", "right", "left"):
            ax.spines[sp].set_visible(False)
        ax.tick_params(axis="y", length=0)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels([M2N[m] for _, m in rows], fontsize=8)
    for k, b in enumerate(backbones):
        axes[0].text(-0.36, len(rows) - 5 * k - 3, _GR_NICE.get(b, b), transform=axes[0].get_yaxis_transform(),
                     rotation=90, va="center", ha="center", fontsize=8, fontweight="bold", color=_GR_COL[b])
    h = [Line2D([], [], color=_GR_COL[b], marker="s", ms=6, lw=0) for b in backbones]
    h += [Line2D([], [], color="#666666", marker="s", ms=6, lw=1.2),
          Line2D([], [], color="#666666", marker="s", ms=6, mfc="white", lw=1.2, ls=(0, (3, 2)))]
    fig.subplots_adjust(left=0.17, right=0.98, top=0.9, bottom=0.2, wspace=0.12)
    # anchored just below the x labels rather than at the figure foot
    fig.legend(h, [_GR_NICE.get(b, b) for b in backbones] + ["95% CI excludes zero", "95% CI includes zero"],
               loc="upper center", bbox_to_anchor=(0.575, 0.105), ncol=6, fontsize=8,
               frameon=True, handlelength=1.6, columnspacing=1.2, handletextpad=0.5)
    save_path = Path(save_path) if save_path else Path("CT_Significance_TwoPanel.png")
    fig.savefig(save_path, dpi=dpi)
    plt.close(fig)
    return save_path


def plot_counter_trend_evidence(direction: str = "DOWN",
                                save_path: str | Path | None = None,
                                root: Path = REGIME_ANALYSIS_ROOT,
                                dpi: int = 600, verbose: bool = True) -> Path:
    """Tornado of counter-trend effect sizes beside the three significance tests.

    ``direction`` selects which regime counts as counter-trend: M2 DOWN is
    counter-trend in Bullish bars, M2 UP in Bearish ones. Writes a PDF (for
    LaTeX) and a PNG, and returns the PDF path.
    """
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    P, R = _ct_load(direction, root)
    N = len(P)
    y = np.arange(N)[::-1]

    MM = 1 / 25.4
    FS = 7.0            # one size for every text element
    AXLW = 0.7          # one weight for every axis line, horizontal and vertical
    BLACK, GREY, DK, RULE, GRID = "#111111", "#666666", "#333333", "#DDDDDD", "#BBBBBB"

    XP, TP = 26.0, [0, 6.5, 13, 19.5, 26]       # precision (pp)
    XR, TR = 2.0, [0, 0.5, 1.0, 1.5, 2.0]       # return (% per trade)
    W, H = 202, 140
    ML, PW, G2, W3 = 40, 44, 7, 66
    BOT, TOP = 36, 19
    PH = H - TOP - BOT

    with plt.rc_context({"font.family": "DejaVu Sans", "axes.linewidth": AXLW,
                         "xtick.major.width": AXLW, "hatch.linewidth": 0.7}):
        fig = plt.figure(figsize=(W * MM, H * MM))
        aR = fig.add_axes([ML / W, BOT / H, PW / W, PH / H])
        aP = fig.add_axes([(ML + PW) / W, BOT / H, PW / W, PH / H])
        a3 = fig.add_axes([(ML + 2 * PW + G2) / W, BOT / H, W3 / W, PH / H])

        def _forest(ax, df, xmax, ticks, invert, xlabel):
            for i, r in df.iterrows():
                col, yy, ex = _CT_COL[r["m1"]], y[i], bool(r["excl0"])
                lo_c, hi_c = max(r["lo"], 0.0), min(r["hi"], xmax)
                if hi_c > lo_c:
                    ax.plot([lo_c, hi_c], [yy, yy], color=col, lw=1.2,
                            ls="-" if ex else (0, (3, 2)), zorder=3,
                            solid_capstyle="butt")
                if r["lo"] > 0:
                    ax.plot([r["lo"]] * 2, [yy - 0.17, yy + 0.17], color=col, lw=0.8, zorder=3)
                if r["hi"] < xmax:
                    ax.plot([r["hi"]] * 2, [yy - 0.17, yy + 0.17], color=col, lw=0.8, zorder=3)
                ax.plot(r["gap"], yy, marker="s", ms=4.8, mec=col, mew=1.0,
                        mfc=col if ex else "#FFFFFF", zorder=4)
            ax.set_xlim(*((xmax, 0) if invert else (0, xmax)))
            ax.set_ylim(-0.8, N - 1 + 0.8)
            ax.set_yticks([])
            ax.set_xticks(ticks)
            ax.set_xticklabels([f"+{t:g}" if t else "0" for t in ticks], fontsize=FS)
            ax.set_xlabel(xlabel, fontsize=FS, labelpad=3)
            for sp in ("top", "right", "left"):
                ax.spines[sp].set_visible(False)

        _forest(aR, R, XR, TR, True, "Return gap (pp)")
        _forest(aP, P, XP, TP, False, "Precision gap (pp)")
        # one shared spine at the join; both panels' own left/right spines are off
        for _a in (aR, aP):
            _a.spines["left"].set_visible(False)
            _a.spines["right"].set_visible(False)
        aP.axvline(0, color=BLACK, lw=AXLW, ls="-", zorder=6, clip_on=False)

        tr = aR.get_yaxis_transform()
        for i, r in P.iterrows():
            aR.text(-0.048, y[i], _CT_M2N.get(r["m2"], r["m2"]), transform=tr,
                    ha="right", va="center", fontsize=FS, color=DK, clip_on=False)
        for bi, m1 in enumerate(["Kronos", "Fincast", "Chronos2", "Tirex"]):
            yc = (y[bi * 5] + y[bi * 5 + 4]) / 2.0
            aR.text(-0.555, yc, _CT_NICE[m1], transform=tr, ha="center", va="center",
                    rotation=90, fontsize=FS, fontweight="bold",
                    color=_CT_COL[m1], clip_on=False)

        # ── significance lanes ────────────────────────────────────────────
        # Two lanes are p-values; the third is sigma beyond the shuffled null,
        # because that test's p-value is floored. Both map onto a common
        # "evidence" axis whose gridlines mark .05 / .01 / .001, using the
        # normal equivalents 1.96 / 3.29 / 6 sigma for the null lane.
        VMAX = -np.log10(_CT_PFLOOR)
        ZLO, ZHI = 1.0, 14.0
        PAD, INSET = 0.34, 0.035
        BAR = 1.0 - PAD
        LANES = [("Across quarters", "B_p", "p"), ("Across assets", "C_p", "p"),
                 ("Reg. Permutations", "E_z", "z")]
        LEVELS = [(0.05, 1.96, ".05"), (0.01, 3.29, ""), (0.001, 6.00, ".001")]
        ZLAB = {1.96: "2σ", 3.29: "", 6.00: "6σ"}

        def _pos(val, kind):
            if kind == "p":
                return -np.log10(max(float(val), _CT_PFLOOR)) / VMAX * BAR
            v = min(max(float(val), ZLO), ZHI)
            return np.log(v / ZLO) / np.log(ZHI / ZLO) * BAR

        a3.set_xlim(0, len(LANES))
        a3.set_ylim(-0.8, N - 1 + 0.8)
        ticks, labels = [], []
        for j, (lab, col, kind) in enumerate(LANES):
            x0 = j + INSET
            for pv, zv, lb in LEVELS:
                v = _pos(pv if kind == "p" else zv, kind)
                a3.plot([x0 + v] * 2, [-0.55, N - 1 + 0.55], color=GRID,
                        lw=AXLW, ls=(0, (3, 2)), zorder=1)
                tick_lab = ZLAB.get(zv, "") if kind == "z" else lb
                if tick_lab:
                    ticks.append(x0 + v)
                    labels.append(tick_lab)
            for i in range(N):
                c0, yy = _CT_COL[P["m1"][i]], y[i]
                pv_, rv_ = P[col][i], R[col][i]
                ok_p = (pv_ <= 0.05) if kind == "p" else (pv_ >= 1.96)
                ok_r = (rv_ <= 0.05) if kind == "p" else (rv_ >= 1.96)
                # precision: solid fill.  return: hatched outline - a different
                # texture, so it cannot be confused with "not significant".
                # the outlined bar's edge straddles its own left edge, so half the
                # stroke would spill past the lane axis: start it a hair to the right
                inset = 0.004
                a3.barh(yy + 0.15, _pos(pv_, kind), height=0.27, left=x0, color=c0,
                        alpha=0.95 if ok_p else 0.15, edgecolor="none", zorder=3)
                a3.barh(yy - 0.15, max(_pos(rv_, kind) - inset, 0.0), height=0.27,
                        left=x0 + inset, facecolor="#FFFFFF", edgecolor=c0,
                        lw=1.3 if ok_r else 0.7, alpha=1.0 if ok_r else 0.32,
                        hatch="/////", zorder=3)
            a3.text(x0 + BAR / 2, N - 1 + 0.85, lab, ha="center", va="bottom",
                    fontsize=FS, color=DK)
        # lane axes drawn last and unclipped: on the panel edge half the stroke
        # would otherwise be cut, making lane 0 render thinner than the rest.
        for j in range(len(LANES)):
            a3.plot([j + INSET] * 2, [-0.55, N - 1 + 0.55], color=BLACK, lw=AXLW,
                    zorder=20, solid_capstyle="butt", clip_on=False)
        a3.set_yticks([])
        a3.set_xticks(ticks)
        a3.set_xticklabels(labels, fontsize=FS, color="#888888")
        a3.tick_params(axis="x", length=2, pad=1.0, colors="#888888")
        for sp in ("top", "right", "left", "bottom"):
            a3.spines[sp].set_visible(False)
        for j in range(len(LANES)):
            a3.plot([j + INSET, j + INSET + BAR], [-0.80, -0.80], color=BLACK,
                    lw=AXLW, zorder=5, clip_on=False, solid_capstyle="butt")
        a3.set_xlabel("evidence against chance\n(p-value; σ beyond the null)",
                      fontsize=FS, labelpad=3, color=DK, linespacing=1.4)

        for b in range(1, 4):
            yy = y[b * 5 - 1] - 0.5
            for ax in (aR, aP):
                ax.plot([0, 1], [yy, yy], transform=ax.get_yaxis_transform(),
                        color=RULE, lw=0.5, zorder=0, clip_on=False)
            for j in range(len(LANES)):
                a3.plot([j + INSET, j + INSET + BAR], [yy, yy], color=RULE,
                        lw=0.5, zorder=0)

        h1 = [Line2D([], [], color=GREY, lw=1.2, ls="-", marker="s", ms=4.8,
                     mfc=GREY, mec=GREY, markevery=[1]),
              Line2D([], [], color=GREY, lw=1.2, ls=(0, (2.2, 1.8)), marker="s",
                     ms=4.8, mfc="#FFFFFF", mec=GREY, markevery=[1])]
        fig.legend(h1, ["95% CI excludes zero", "95% CI includes zero"],
                   loc="center", bbox_to_anchor=((ML + PW) / W, 0.118), ncol=1,
                   fontsize=FS, frameon=True, handlelength=3.4, handletextpad=0.7,
                   labelspacing=0.5, borderpad=0.55, edgecolor="#CCCCCC",
                   facecolor="#FFFFFF")
        h2 = [Patch(facecolor=GREY, alpha=0.95, edgecolor="none"),
              Patch(facecolor="#FFFFFF", edgecolor=GREY, lw=1.3, hatch="/////"),
              Patch(facecolor=GREY, alpha=0.15, edgecolor="none"),
              Line2D([], [], color=GRID, lw=AXLW, ls=(0, (3, 2)))]
        # centred on the drawn lanes, not on the axes box (they are inset)
        lane_mid = (INSET + (len(LANES) - 1) + INSET + BAR) / 2 / len(LANES)
        fig.legend(h2, ["precision", "return", "no significance", "thresholds"],
                   loc="center",
                   bbox_to_anchor=((ML + 2 * PW + G2 + W3 * lane_mid) / W, 0.118),
                   ncol=2, fontsize=FS, frameon=True, handlelength=1.5,
                   columnspacing=1.2, labelspacing=0.5, borderpad=0.55,
                   edgecolor="#CCCCCC", facecolor="#FFFFFF")
        fig.text(0.44, 0.016,
                 f"Counter-trend vs with-trend bars, M2 {direction.upper()}, "
                 "held-out test period", ha="center", va="center",
                 fontsize=FS, style="italic", color=GREY)

        if save_path is None:
            save_path = regime_out_dir("robust_significance") / \
                f"counter_trend_evidence_{direction.upper()}.pdf"
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path)
        fig.savefig(save_path.with_suffix(".png"), dpi=dpi)
        plt.close(fig)

    if verbose:
        n_p = int(P["excl0"].sum())
        n_r = int(R["excl0"].sum())
        print(f"  [saved] {save_path}")
        print(f"          M2 {direction.upper()}: precision CIs excluding zero "
              f"{n_p}/{N}, return {n_r}/{N}, mean gap {P['gap'].mean():+.1f} pp")
    return save_path


# ════════════════════════════════════════════════════════════════════════════
#  CV → test transfer figure  (counter-trend gap by granularity)
# ════════════════════════════════════════════════════════════════════════════
# Four columns (backbones) x two rows (precision, return). Each panel overlays
# the counter-trend minus with-trend gap as measured by CPCV, by CVExpand, and on
# the held-out test window, with a +/-1 std band over the five secondary models.
#
# The claim is transfer, not prediction: the pair-level CV<->test correlation is
# NEGATIVE (-0.48), so never draw this as a scatter with a fitted line. What the
# figure shows is that the gap is present and similarly shaped in all three
# designs, and that only the financially-pretrained backbones carry the
# granularity structure through to held-out data.

_CVT_CP = "#1F6FB4"   # CPCV      - primary CV evidence (5 chronological paths)
_CVT_CX = "#7FB3DC"   # CVExpand  - corroborating
_CVT_TE = "#B03030"   # held-out test


def _cvt_regime_mix(m1: str, directions=("DOWN",), granularities=None,
                    m2_models=("rf", "tabicl", "tabpfn", "autogluon", "ctts"),
                    edge_root: Path = Path("Output/Analysis/Edge_NoCal"),
                    jm_root: Path = _JM_GLOBAL_ROOT,
                    raw_root: Path = Path("Data_MLA/Raw")) -> Dict[str, float]:
    """Bullish share of the bars each design actually evaluates, in percent.

    This is NOT the static split composition. The validation *window* is ~80%
    bearish, but CPCV and CVExpand score across the whole train+val timeline, so
    the bars they evaluate are far more balanced (~43% bullish on Kronos). The
    share is backbone-specific because each backbone emits signals on different
    bars, so it must be reported per column rather than as one global figure.
    """
    grans = granularities or ["1d", "12h", "8h", "6h", "4h", "2h", "1h"]
    out: Dict[str, List[float]] = {"cpcv": [], "cvexpand": []}
    for g in grans:
        for scheme, fname, key in (("cpcv", "cpcv_samples.csv", "path"),
                                   ("cvexpand", "cvexpand_samples.csv", "split")):
            vals = []
            for d in directions:
                for m2 in m2_models:
                    f = Path(edge_root) / m1 / m2 / d.upper() / g / fname
                    if not f.exists():
                        continue
                    s = pd.read_csv(f, parse_dates=["date"])
                    # the regime join brings its own "split" column; rename first
                    s = s.rename(columns={key: "_fold"})
                    m = _attach_regimes(s, g, jm_root, raw_root)
                    m = m.dropna(subset=["regime_name"])
                    if len(m):
                        vals.append(float((m["regime_name"] == "Bullish").mean() * 100))
            if vals:
                out[scheme].append(float(np.mean(vals)))
    mix = {k: float(np.mean(v)) for k, v in out.items() if v}

    test_vals = []
    for g in grans:
        for d in directions:
            for m2 in m2_models:
                try:
                    t = _test_trades_with_regime(m1, m2, d.upper(), g,
                                                 "Utility_Score_NoCal",
                                                 Path("Output"), jm_root)
                except Exception:
                    t = None
                if t is not None and len(t):
                    test_vals.append(float((t["regime_name"] == "Bullish").mean() * 100))
    if test_vals:
        mix["test"] = float(np.mean(test_vals))
    return mix


def plot_cv_to_test_transfer(direction: str = "DOWN",
                             backbones=("Kronos", "Fincast", "Chronos2", "Tirex"),
                             granularities=None,
                             save_path: str | Path | None = None,
                             root: Path = REGIME_ANALYSIS_ROOT,
                             show_regime_mix: bool = True,
                             jm_root: Path = _JM_GLOBAL_ROOT,
                             dpi: int = 600, verbose: bool = True) -> Path:
    """Counter-trend gap by granularity under CPCV, CVExpand and the held-out test.

    Set ``show_regime_mix=False`` to skip the per-column bullish-share strip; it
    re-reads every CV sample file and is the slow part of this function.
    """
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    G = granularities or ["1d", "12h", "8h", "6h", "4h", "2h", "1h"]
    NICE = {"Kronos": "Kronos", "Fincast": "FinCast",
            "Chronos2": "Chronos-2", "Tirex": "Tirex"}

    cv = pd.concat([pd.read_csv(f) for f in sorted(Path(root).glob(
        "*/*/train_val/bull_vs_bear_significance/all_granularities.csv"))],
        ignore_index=True)
    cv = cv.drop_duplicates(["m1", "m2", "direction", "granularity",
                             "scheme", "metric"])
    cv = cv[cv["direction"] == direction.upper()]
    # CV files store a raw bull-minus-bear gap for both directions; M2 UP's
    # hypothesis is bear-minus-bull, so its sign is flipped to match the test
    # files, which are already oriented by the counter-trend regime (Table 5).
    sign = -1.0 if direction.upper() == "UP" else 1.0
    cv["gap"] = cv["observed_gap"] * 100.0 * sign

    te = pd.concat([pd.read_csv(f) for f in sorted(Path(root).glob(
        "*/*/test/bull_vs_bear_by_quarter/all_granularities.csv"))],
        ignore_index=True)
    te = te[te["scope"] == "OVERALL"].drop_duplicates(
        ["m1", "m2", "direction", "granularity", "metric"])
    te = te[te["direction"] == direction.upper()]
    te["gap"] = te["gap"] * 100.0 * sign

    def _curve(m1, metric, src, scheme=None):
        if src == "cv":
            q = cv[(cv["m1"] == m1) & (cv["metric"] == metric)
                   & (cv["scheme"] == scheme)]
        else:
            q = te[(te["m1"] == m1) & (te["metric"] == metric)]
        g = q.groupby("granularity")["gap"]
        return g.mean().reindex(G), g.std().reindex(G)

    mixes = {}
    if show_regime_mix:
        for m1 in backbones:
            try:
                mixes[m1] = _cvt_regime_mix(m1, (direction,), G, jm_root=jm_root)
            except Exception:
                mixes[m1] = {}

    MM = 1 / 25.4
    FS, AXLW = 7.0, 0.7
    GREY, DK = "#666666", "#333333"
    W, H = 190, 116
    ML, PW, GX, BOT, TOP, GY = 17, 38, 5.5, 26, 15, 7
    PH = (H - TOP - BOT - GY) / 2
    YL = {"precision": (-11, 31, [-10, 0, 10, 20, 30]),
          "return": (-2.2, 6.0, [-2, 0, 2, 4, 6])}
    x = np.arange(len(G))

    with plt.rc_context({"font.family": "DejaVu Sans", "axes.linewidth": AXLW,
                         "xtick.major.width": AXLW, "ytick.major.width": AXLW}):
        fig = plt.figure(figsize=(W * MM, H * MM))
        for i, metric in enumerate(("precision", "return")):
            lo, hi, ticks = YL[metric]
            for j, m1 in enumerate(backbones):
                ax = fig.add_axes([(ML + j * (PW + GX)) / W,
                                   (BOT + (1 - i) * (PH + GY)) / H, PW / W, PH / H])
                # grey dashed, so it never reads as the axis spine
                ax.axhline(0, color="#999999", lw=AXLW, ls=(0, (4, 3)), zorder=4)
                for src, sch, col, lw, ls, mk, z, al in (
                        ("cv", "cpcv", _CVT_CP, 1.7, "-", "o", 6, 0.20),
                        ("cv", "cvexpand", _CVT_CX, 1.2, "-", "^", 5, 0.16),
                        ("test", None, _CVT_TE, 1.5, (0, (4, 2)), "s", 7, 0.14)):
                    m, sd = _curve(m1, metric, src, sch)
                    ax.fill_between(x, (m - sd).values, (m + sd).values,
                                    color=col, alpha=al, lw=0, zorder=z - 4)
                    ax.plot(x, m.values, color=col, lw=lw, ls=ls, marker=mk,
                            ms=3.4, mfc=col, mec="none", zorder=z)
                ax.set_xlim(-0.4, len(G) - 0.6)
                ax.set_xticks(x)
                ax.set_xticklabels(G if i == 1 else [""] * len(G), fontsize=FS - 0.8)
                ax.set_ylim(lo, hi)
                ax.set_yticks(ticks)
                ax.set_yticklabels([f"{t:g}" for t in ticks] if j == 0
                                   else [""] * len(ticks), fontsize=FS)
                for sp in ("top", "right"):
                    ax.spines[sp].set_visible(False)
                if i == 0:
                    ax.set_title(NICE.get(m1, m1), fontsize=FS, pad=13,
                                 fontweight="bold", color=DK)
                    mx = mixes.get(m1, {})
                    if mx.get("cpcv") is not None and mx.get("test") is not None:
                        # each share is coloured like the curve it describes
                        parts = [("Bullish bars  ", GREY, "normal"),
                                 (f"{mx['cpcv']:.0f}%", _CVT_CP, "bold"),
                                 ("  ·  ", GREY, "normal"),
                                 (f"{mx.get('cvexpand', float('nan')):.0f}%",
                                  _CVT_CX, "bold"),
                                 ("  ·  ", GREY, "normal"),
                                 (f"{mx['test']:.0f}%", _CVT_TE, "bold")]
                        rend = fig.canvas.get_renderer()
                        arts, widths = [], []
                        for txt, col, fw in parts:
                            t = ax.text(0, 1.015, txt, transform=ax.transAxes,
                                        ha="left", va="bottom", fontsize=FS - 1.2,
                                        color=col, fontweight=fw)
                            bb = t.get_window_extent(renderer=rend).transformed(
                                ax.transAxes.inverted())
                            arts.append(t)
                            widths.append(bb.width)
                        xc = 0.5 - sum(widths) / 2
                        for t, ww in zip(arts, widths):
                            t.set_x(xc)
                            xc += ww
                if j == 0:
                    ax.set_ylabel("Precision gap (pp)" if metric == "precision"
                                  else "Return gap (pp)", fontsize=FS)

        hh = [Line2D([], [], color=_CVT_CP, lw=1.7, marker="o", ms=3.4),
              Line2D([], [], color=_CVT_CX, lw=1.2, marker="^", ms=3.4),
              Line2D([], [], color=_CVT_TE, lw=1.5, ls=(0, (4, 2)), marker="s", ms=3.4),
              Patch(facecolor=GREY, alpha=0.22, edgecolor="none")]
        ll = ["CPCV  (5 chronological paths)", "CVExpand  (5 chronological folds)",
              "Held-out test", "Shaded band (±1 std over M2s)"]
        fig.legend(hh, ll, loc="center",
                   bbox_to_anchor=((ML + 2 * PW + 1.5 * GX) / W, 0.095), ncol=4,
                   fontsize=FS, frameon=True, handlelength=1.9, columnspacing=1.5,
                   borderpad=0.55, edgecolor="#CCCCCC", facecolor="#FFFFFF")
        tail = ("· percentages above each column give the bullish share of the "
                "bars evaluated (CPCV · CVExpand · test)"
                if show_regime_mix else "")
        fig.text(0.5, 0.018, f"Counter-trend minus with-trend, M2 "
                 f"{direction.upper()} {tail}", ha="center", va="center",
                 fontsize=FS - 0.5, style="italic", color=GREY)

        if save_path is None:
            save_path = regime_out_dir("cv_test_transfer") / \
                f"cv_to_test_transfer_{direction.upper()}.pdf"
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path)
        fig.savefig(save_path.with_suffix(".png"), dpi=dpi)
        plt.close(fig)

    if verbose:
        for metric in ("precision", "return"):
            row = []
            for src, sch in (("cv", "cpcv"), ("cv", "cvexpand"), ("test", None)):
                vals = [_curve(m1, metric, src, sch)[0] for m1 in backbones]
                row.append(float(np.nanmean([v.mean() for v in vals])))
            print(f"          {metric:9s} CPCV {row[0]:+5.2f}  "
                  f"CVExpand {row[1]:+5.2f}  test {row[2]:+5.2f}")
        print(f"  [saved] {save_path}")
    return save_path


# ════════════════════════════════════════════════════════════════════════════
#  Granularity figure  (A1: coarse bars dominate fine ones)
# ════════════════════════════════════════════════════════════════════════════
# Two rows (selective precision, portfolio return) x two columns (M2 DOWN, M2 UP).
# Every one of the 280 configurations is plotted as a dot, coloured and slotted by
# backbone; the box is the interquartile range and the bar the median over the 20
# backbone x M2 combinations at that granularity.
#
# The predecessor figure showed only the BEST M2 per cell, selected on the test
# window, which is exactly reviewer R5-W2's complaint. Nothing is selected here.
#
# No significance test is drawn: the 140 cells per direction share backbones,
# assets and overlapping 7-bar trades, so they are not independent and any
# p-value would be optimistic. The coarse/fine means carry the claim instead.

_GR_COL = {"Kronos": "#1F6FB4", "Fincast": "#2E8B57",
           "Chronos2": "#8A6BBE", "Tirex": "#D08A2E"}
_GR_NICE = {"Kronos": "Kronos", "Fincast": "FinCast",
            "Chronos2": "Chronos-2", "Tirex": "Tirex"}


def plot_granularity_performance(perf_grid: str | Path,
                                 save_path: str | Path | None = None,
                                 coarse=("1d", "12h", "8h"),
                                 granularities=None,
                                 band_style: str = "bracket",
                                 dpi: int = 600, verbose: bool = True) -> Path:
    """Selective precision and portfolio return by granularity, all cells shown.

    ``band_style="text"`` writes the coarse/fine means as bold headers; ``"bracket"``
    shades the coarse band, writes each band's mean as a small tag, and names
    the bands once with brackets above the top row.

    ``perf_grid`` is a CSV with one row per (m1, m2, dir, gran) carrying
    ``m2_prec`` and ``m2_ret`` on the held-out window.
    """
    from matplotlib.lines import Line2D

    G = list(granularities or ["1d", "12h", "8h", "6h", "4h", "2h", "1h"])
    COARSE = set(coarse)
    M1S = ["Kronos", "Fincast", "Chronos2", "Tirex"]
    d = pd.read_csv(perf_grid)

    MM = 1 / 25.4
    FS, AXLW = 8.0, 0.7
    GREY, DK, BOX = "#666666", "#333333", "#555555"
    W, H = 186, 126
    # GY (row gap), GX (column gap) and B (legend band) tightened to save
    # vertical and horizontal space in the paper without moving any element.
    ML, R, T, B, GX, GY = 22, 6, 12, 22, 11, 7
    pw = (W - ML - R - GX) / 2
    ph = (H - T - B - GY) / 2
    x = np.arange(len(G))
    # each backbone owns a fixed horizontal slot, so dot position is informative
    SLOT = {m: (-0.255 + 0.17 * i) for i, m in enumerate(M1S)}
    PREC_YL, PREC_TK = (31, 92), [30, 40, 50, 60, 70, 80]
    RET_YL, RET_TK = (-72, 54), [-60, -40, -20, 0, 20, 40]
    # spine and divider stop here, clear of the Coarse/Fine labels
    SPINE_TOP = {"m2_prec": 84.0, "m2_ret": 40.0}

    with plt.rc_context({"font.family": "DejaVu Sans", "axes.linewidth": AXLW,
                         "xtick.major.width": AXLW, "ytick.major.width": AXLW}):
        fig = plt.figure(figsize=(W * MM, H * MM))
        spec = [(0, "DOWN", "m2_prec", PREC_YL, PREC_TK, 50, "{:.1f}%"),
                (1, "UP", "m2_prec", PREC_YL, PREC_TK, 50, "{:.1f}%"),
                (2, "DOWN", "m2_ret", RET_YL, RET_TK, 0, "{:+.1f}%"),
                (3, "UP", "m2_ret", RET_YL, RET_TK, 0, "{:+.1f}%")]
        for k, dirn, col, ylim, ticks, ref, fmt in spec:
            i, j = divmod(k, 2)
            ax = fig.add_axes([(ML + j * (pw + GX)) / W,
                               (B + (1 - i) * (ph + GY)) / H, pw / W, ph / H])
            sub = d[d["dir"] == dirn]
            ax.axhline(ref, color="#999999", lw=AXLW, ls=(0, (4, 3)), zorder=2)
            for xi, g in enumerate(G):
                v = sub[sub["gran"] == g]
                for m1 in M1S:
                    w = v[v["m1"] == m1][col].values
                    off = (np.linspace(-0.055, 0.055, len(w)) if len(w) > 1
                           else np.zeros(len(w)))
                    ax.scatter(xi + SLOT[m1] + off, w, s=5.5, color=_GR_COL[m1],
                               alpha=0.85, lw=0, zorder=4)
                allv = v[col].values
                q1, md, q3 = np.percentile(allv, [25, 50, 75])
                ax.add_patch(plt.Rectangle((xi - 0.34, q1), 0.68, q3 - q1,
                                           facecolor="none", edgecolor=BOX,
                                           lw=0.7, zorder=5))
                ax.plot([xi - 0.34, xi + 0.34], [md, md], color=BOX, lw=1.6,
                        zorder=6, solid_capstyle="butt")
            ax.set_xlim(-0.66, len(G) - 0.34)
            ax.set_xticks(x)
            ax.set_xticklabels(G, fontsize=FS)
            if col == "m2_ret":
                # symlog keeps the dense region honest while fitting the -67% tail
                ax.set_yscale("symlog", linthresh=60, linscale=1.0)
            ax.set_ylim(*ylim)
            ax.set_yticks(ticks)
            ax.set_yticklabels([f"{t:g}" for t in ticks], fontsize=FS)
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
            top = SPINE_TOP[col]
            ax.spines["left"].set_bounds(ylim[0], top)
            tr = ax.get_xaxis_transform()
            lab_y = (top - ylim[0]) / (ylim[1] - ylim[0]) + 0.022
            co = sub[sub["gran"].isin(COARSE)][col]
            fi = sub[~sub["gran"].isin(COARSE)][col]
            if band_style == "bracket":
                nc = len(coarse)
                ax.add_patch(plt.Rectangle((-0.66, 0), nc - 0.5 + 0.66, 1, transform=tr,
                                           facecolor="#EEF3F8", lw=0, zorder=0))
                for (x0, x1), vals in (((-0.5, nc - 0.5), co), ((nc - 0.5, len(G) - 0.5), fi)):
                    ax.text((x0 + x1) / 2, lab_y - 0.01, f"mean {fmt.format(vals.mean())}",
                            transform=tr, ha="center", va="bottom", fontsize=FS - 0.5,
                            color=DK, zorder=8)
                if i == 0:
                    yb = lab_y + 0.085
                    for x0, x1, lab in ((-0.4, nc - 0.6, f"Coarse ({G[0]}–{G[nc - 1]})"),
                                        (nc - 0.4, len(G) - 0.6, f"Fine ({G[nc]}–{G[-1]})")):
                        ax.plot([x0, x0, x1, x1], [yb - 0.018, yb, yb, yb - 0.018], color=GREY,
                                lw=0.6, transform=tr, clip_on=False)
                        ax.text((x0 + x1) / 2, yb + 0.012, lab, transform=tr, ha="center",
                                va="bottom", fontsize=FS - 0.5, color=GREY)
            else:
                ax.plot([len(coarse) - 0.5] * 2, [ylim[0], top], color="#CCCCCC",
                        lw=0.7, zorder=1, clip_on=False)
                for xpos, lab, vals in ((1.0, "Coarse", co), (4.5, "Fine", fi)):
                    ax.text(xpos, lab_y, f"{lab} {fmt.format(vals.mean())}",
                            transform=tr, ha="center", va="bottom", fontsize=FS,
                            fontweight="bold", color=DK)
            if i == 0 and band_style == "bracket":
                ax.text(2.75, lab_y + 0.2, f"M2 {dirn}", transform=tr,
                        ha="center", va="bottom", fontsize=FS,
                        fontweight="bold", color=DK)
            elif i == 0:
                ax.text(2.75, lab_y + 0.082, f"M2 {dirn}", transform=tr,
                        ha="center", va="bottom", fontsize=FS,
                        fontweight="bold", color=DK)
            if j == 0:
                ax.set_ylabel("Selective precision (%)" if col == "m2_prec"
                              else "Portfolio return (%)", fontsize=FS)
                # one fixed offset, so both labels share a vertical line however
                # wide the tick labels of each row are
                ax.yaxis.set_label_coords(-0.12, 0.5)
            if i == 1:
                ax.set_xlabel("Bar granularity", fontsize=FS, labelpad=6)

        h = [Line2D([], [], color=_GR_COL[m], lw=0, marker="o", ms=3.4) for m in M1S]
        h += [Line2D([], [], color=BOX, lw=1.6)]
        labs = [_GR_NICE[m] for m in M1S] + ["Median and interquartile"]
        fig.legend(h, labs,
                   loc="center", bbox_to_anchor=((ML + pw + GX / 2) / W, 0.055),
                   ncol=len(h), fontsize=FS, frameon=True, handlelength=1.2,
                   columnspacing=0.9, handletextpad=0.45, borderpad=0.5,
                   edgecolor="#CCCCCC", facecolor="#FFFFFF")
        if save_path is None:
            save_path = regime_out_dir("granularity") / "granularity_performance.pdf"
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path)
        fig.savefig(save_path.with_suffix(".png"), dpi=dpi)
        plt.close(fig)

    if verbose:
        for dirn in ("DOWN", "UP"):
            s = d[d["dir"] == dirn]
            for col, nm in (("m2_prec", "precision"), ("m2_ret", "return")):
                co = s[s["gran"].isin(COARSE)][col]
                fi = s[~s["gran"].isin(COARSE)][col]
                print(f"          M2 {dirn:5s} {nm:9s} coarse {co.mean():+6.2f}  "
                      f"fine {fi.mean():+6.2f}")
        print(f"  [saved] {save_path}")
    return save_path


# ════════════════════════════════════════════════════════════════════════════
#  Reliability-model family figure  (A2: tabular beats time-series deep learning)
# ════════════════════════════════════════════════════════════════════════════
# Two rows (gain in selective precision over M1, portfolio return) x two columns
# (M2 DOWN, M2 UP). Five secondary-model families on x, every cell plotted as a
# dot coloured by backbone - the same visual grammar as the granularity figure.
#
# All four backbones are kept deliberately. A2 is a FAMILY claim (tabular beats
# time-series DL) and is stronger the more backbones it survives: CTTS ranks last
# of five on delta-precision on 4/4 backbones. Scoping to Kronos would cut the
# evidence from 280 cells to 70 for no gain. The separate "best M2 given Kronos"
# claim is a configuration claim and lives in tab:portfolio_performance.
#
# WHY M2 UP IS SHOWN: on M2 UP return CTTS ranks 2nd of five on Tirex and high
# generally, which looks like a counterexample until you see that every M2 UP
# configuration loses money. CTTS trades least, so it bleeds least. Showing the
# panel makes that artifact visible instead of leaving a reviewer to find it.

_M2_FAM = ["tabicl", "autogluon", "tabpfn", "rf", "ctts"]
_M2_LBL = {"rf": "Random\nForest", "tabicl": "TabICL", "tabpfn": "TabPFN",
           "autogluon": "AutoGluon", "ctts": "CTTS"}


def plot_reliability_family(perf_grid: str | Path,
                            save_path: str | Path | None = None,
                            band=("1d", "12h", "8h"),
                            restrict_to_band: bool = False,
                            dpi: int = 600, verbose: bool = True) -> Path:
    """Gain over M1 and portfolio return by secondary-model family.

    Defaults to all seven granularities. A2 holds at every one of them - CTTS
    improves on M1 in only 28/56 cells against 49-52 for the tabular models - so
    restricting to the profitable band (``restrict_to_band=True``) would sharpen
    CTTS's mean but narrow a claim that does not need narrowing.
    """
    from matplotlib.lines import Line2D

    d = pd.read_csv(perf_grid)
    d["dprec"] = d["m2_prec"] - d["m1_prec"]
    if restrict_to_band:
        d = d[d["gran"].isin(band)]
    M1S = ["Kronos", "Fincast", "Chronos2", "Tirex"]

    MM = 1 / 25.4
    FS, AXLW = 7.0, 0.7
    GREY, DK, BOX = "#666666", "#333333", "#555555"
    W, H = 186, 152
    ML, R, T, B, GX, GY = 21, 6, 15, 30, 16, 13
    pw = (W - ML - R - GX) / 2
    ph = (H - T - B - GY) / 2
    x = np.arange(len(_M2_FAM))
    SLOT = {m: (-0.255 + 0.17 * i) for i, m in enumerate(M1S)}
    DP_YL, DP_TK = (-22, 46), [-20, -10, 0, 10, 20, 30]
    RET_YL, RET_TK = (-72, 54), [-60, -40, -20, 0, 20, 40]
    SPINE_TOP = {"dprec": 30.0, "m2_ret": 40.0}

    with plt.rc_context({"font.family": "DejaVu Sans", "axes.linewidth": AXLW,
                         "xtick.major.width": AXLW, "ytick.major.width": AXLW}):
        fig = plt.figure(figsize=(W * MM, H * MM))
        spec = [(0, "DOWN", "dprec", DP_YL, DP_TK, 0, "{:+.1f}%"),
                (1, "UP", "dprec", DP_YL, DP_TK, 0, "{:+.1f}%"),
                (2, "DOWN", "m2_ret", RET_YL, RET_TK, 0, "{:+.1f}%"),
                (3, "UP", "m2_ret", RET_YL, RET_TK, 0, "{:+.1f}%")]
        for k, dirn, col, ylim, ticks, ref, fmt in spec:
            i, j = divmod(k, 2)
            ax = fig.add_axes([(ML + j * (pw + GX)) / W,
                               (B + (1 - i) * (ph + GY)) / H, pw / W, ph / H])
            sub = d[d["dir"] == dirn]
            ax.axhline(ref, color="#999999", lw=AXLW, ls=(0, (4, 3)), zorder=2)
            for xi, m2 in enumerate(_M2_FAM):
                v = sub[sub["m2"] == m2]
                for m1 in M1S:
                    w = v[v["m1"] == m1][col].values
                    off = (np.linspace(-0.055, 0.055, len(w)) if len(w) > 1
                           else np.zeros(len(w)))
                    ax.scatter(xi + SLOT[m1] + off, w, s=5.5, color=_GR_COL[m1],
                               alpha=0.85, lw=0, zorder=4)
                allv = v[col].values
                q1, md, q3 = np.percentile(allv, [25, 50, 75])
                ax.add_patch(plt.Rectangle((xi - 0.34, q1), 0.68, q3 - q1,
                                           facecolor="none", edgecolor=BOX,
                                           lw=0.7, zorder=5))
                ax.plot([xi - 0.34, xi + 0.34], [md, md], color=BOX, lw=1.6,
                        zorder=6, solid_capstyle="butt")
                lab_y0 = (SPINE_TOP[col] - ylim[0]) / (ylim[1] - ylim[0]) + 0.030
                c = "#B03030" if m2 == "ctts" else DK
                ax.text(xi, lab_y0, fmt.format(allv.mean()),
                        transform=ax.get_xaxis_transform(), ha="center",
                        va="bottom", fontsize=FS - 0.8, fontweight="bold", color=c)
                if col == "dprec":
                    ax.text(xi, lab_y0 + 0.058, f"{int((allv > 0).sum())}/{len(allv)}",
                            transform=ax.get_xaxis_transform(), ha="center",
                            va="bottom", fontsize=FS - 1.5, color=c)
            ax.set_xlim(-0.66, len(_M2_FAM) - 0.34)
            ax.set_xticks(x)
            ax.set_xticklabels([_M2_LBL[m] for m in _M2_FAM], fontsize=FS - 0.8)
            if col == "m2_ret":
                ax.set_yscale("symlog", linthresh=60, linscale=1.0)
            ax.set_ylim(*ylim)
            ax.set_yticks(ticks)
            ax.set_yticklabels([f"{t:g}" for t in ticks], fontsize=FS)
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
            top = SPINE_TOP[col]
            ax.spines["left"].set_bounds(ylim[0], top)
            # CTTS sits in its own lane on the right, marked off from the tabular four
            ax.plot([len(_M2_FAM) - 1.5] * 2, [ylim[0], top], color="#CCCCCC",
                    lw=0.7, zorder=1, clip_on=False)
            if i == 0:
                lab_y = (top - ylim[0]) / (ylim[1] - ylim[0]) + 0.030
                ax.text((len(_M2_FAM) - 1) / 2,
                        lab_y + (0.175 if col == "dprec" else 0.115), f"M2 {dirn}",
                        transform=ax.get_xaxis_transform(), ha="center",
                        va="bottom", fontsize=FS, fontweight="bold", color=DK)
            if j == 0:
                # row 1 is a GAIN over the same configuration's M1; row 2 is the
                # ABSOLUTE level. Return is deliberately not shown as a gain: on
                # M2 UP that would rank CNNT best (+35 pp, 28/28) purely because
                # M1 UP loses ~70% and CNNT trades least - the loss-mitigation
                # overclaim the roadmap forbids.
                ax.set_ylabel("Gain in Selective Precision (%)"
                              if col == "dprec"
                              else "Portfolio return (%)", fontsize=FS)

        h = [Line2D([], [], color=_GR_COL[m], lw=0, marker="o", ms=3.4) for m in M1S]
        h += [Line2D([], [], color=BOX, lw=1.6)]
        fig.legend(h, [_GR_NICE[m] for m in M1S]
                   + ["Median and interquartile over all cells"],
                   loc="center", bbox_to_anchor=((ML + pw + GX / 2) / W, 0.055),
                   ncol=5, fontsize=FS, frameon=True, handlelength=1.2,
                   columnspacing=0.9, handletextpad=0.45, borderpad=0.5,
                   edgecolor="#CCCCCC", facecolor="#FFFFFF")
        scope = ("1d–12h–8h only" if restrict_to_band
                 else "all seven granularities")
        fig.text((ML + pw + GX / 2) / W, 0.020,
                 f"Top row: gain over M1, with the count of cells improving on it · "
                 f"bottom row: absolute return · {scope} · no model is selected",
                 ha="center", va="center", fontsize=FS - 0.5, style="italic",
                 color=GREY)

        if save_path is None:
            save_path = regime_out_dir("granularity") / "reliability_family.pdf"
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path)
        fig.savefig(save_path.with_suffix(".png"), dpi=dpi)
        plt.close(fig)

    if verbose:
        for dirn in ("DOWN", "UP"):
            s = d[d["dir"] == dirn]
            r = s.groupby("m2")["dprec"].mean().rank(ascending=False)
            print(f"          M2 {dirn}: " + "  ".join(
                f"{m}={s[s['m2'] == m]['dprec'].mean():+.2f}" for m in _M2_FAM))
            print(f"                   CTTS rank on Δprecision: {int(r['ctts'])}/5")
        print(f"  [saved] {save_path}")
    return save_path


# ════════════════════════════════════════════════════════════════════════════
#  Risk/return trade-off figure  (A7: return and drawdown improve together)
# ════════════════════════════════════════════════════════════════════════════
# One dot per configuration: change in portfolio return (x) against reduction in
# maximum drawdown (y), both relative to the same configuration's own M1.
#
# The quadrants carry the claim. Upper-RIGHT is the honest result - the filter
# earns more AND draws down less. Upper-LEFT is the mechanical case the roadmap
# insists on disclosing: a filter that trades less is necessarily exposed less,
# so a drawdown improvement there costs return. Reporting only "M2 reduces
# drawdown in 277/280 cells" would hide that split.
#
# Sharpe is drawn as a BINARY outline, not marker area: area is judged poorly by
# eye, and the fact worth showing is that 34 of 140 cells lose risk-adjusted
# return even where a raw metric improves.

def plot_risk_return_tradeoff(perf_grid: str | Path,
                              direction: str = "DOWN",
                              save_path: str | Path | None = None,
                              dpi: int = 600, verbose: bool = True) -> Path:
    """Change in return vs reduction in drawdown, one dot per configuration."""
    from matplotlib.lines import Line2D

    d = pd.read_csv(perf_grid)
    s = d[d["dir"] == direction.upper()].copy()
    s["dret"] = s["m2_ret"] - s["m1_ret"]
    s["dmdd"] = s["m2_mdd"] - s["m1_mdd"]
    s["dsr"] = s["m2_sr"] - s["m1_sr"]
    M1S = ["Kronos", "Fincast", "Chronos2", "Tirex"]

    MM = 1 / 25.4
    FS, AXLW = 7.0, 0.7
    GREY, DK = "#666666", "#333333"
    W, H = 92, 90
    L, R, T, B = 17, 5, 4, 20
    XL, YL = (-52, 84), (-17, 76)

    with plt.rc_context({"font.family": "DejaVu Sans", "axes.linewidth": AXLW,
                         "xtick.major.width": AXLW, "ytick.major.width": AXLW}):
        fig = plt.figure(figsize=(W * MM, H * MM))
        ax = fig.add_axes([L / W, B / H, (W - L - R) / W, (H - T - B) / H])
        ax.add_patch(plt.Rectangle((0, 0), XL[1], YL[1], facecolor="#2E8B57",
                                   alpha=0.055, lw=0, zorder=0))
        ax.add_patch(plt.Rectangle((XL[0], 0), -XL[0], YL[1], facecolor="#B03030",
                                   alpha=0.06, lw=0, zorder=0))
        ax.axhline(0, color="#999999", lw=AXLW, ls=(0, (4, 3)), zorder=2)
        ax.axvline(0, color="#999999", lw=AXLW, ls=(0, (4, 3)), zorder=2)
        for m1 in M1S:
            v = s[s["m1"] == m1]
            up, dn = v[v["dsr"] > 0], v[v["dsr"] <= 0]
            ax.scatter(up["dret"], up["dmdd"], s=13, color=_GR_COL[m1],
                       alpha=0.75, lw=0, zorder=4)
            ax.scatter(dn["dret"], dn["dmdd"], s=20, facecolor="none",
                       edgecolor=_GR_COL[m1], lw=1.0, zorder=5)
        ax.set_xlim(*XL)
        ax.set_ylim(*YL)
        ax.set_xticks([-40, -20, 0, 20, 40, 60, 80])
        ax.set_yticks([0, 20, 40, 60])
        ax.set_xticklabels([f"{t:g}" for t in ax.get_xticks()], fontsize=FS)
        ax.set_yticklabels([f"{t:g}" for t in ax.get_yticks()], fontsize=FS)
        ax.set_xlabel("Cumulative return, M1+M2 minus M1 (pp)", fontsize=FS)
        ax.set_ylabel("Maximum drawdown, M1+M2 minus M1 (pp)", fontsize=FS)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)

        n_both = int(((s["dret"] > 0) & (s["dmdd"] > 0)).sum())
        n_mech = int(((s["dret"] <= 0) & (s["dmdd"] > 0)).sum())
        n_sr = int((s["dsr"] <= 0).sum())
        n = len(s)
        # placed in DATA coords so each label sits inside its own shaded quadrant
        ax.text(XL[1] - 2.5, 2.5, f"Higher return,\nsmaller drawdown\n{n_both}/{n}",
                ha="right", va="bottom", fontsize=FS - 0.5, fontweight="bold",
                color="#2E8B57", linespacing=1.35, zorder=6)
        ax.text(XL[0] + 2.5, YL[1] - 2.0, f"Lower return,\nsmaller drawdown\n{n_mech}/{n}",
                ha="left", va="top", fontsize=FS - 0.5, fontweight="bold",
                color="#B03030", linespacing=1.35, zorder=6)

        h = [Line2D([], [], color=_GR_COL[m], lw=0, marker="o", ms=3.4) for m in M1S]
        h += [Line2D([], [], color=GREY, lw=0, marker="o", ms=3.4),
              Line2D([], [], color=GREY, lw=0, marker="o", ms=3.8, mfc="none", mew=1.0)]
        fig.legend(h, [_GR_NICE[m] for m in M1S]
                   + ["Sharpe ratio improvement", "Sharpe ratio decline"],
                   # anchored a fixed 9 mm below the axes, so the gap to the x
                   # label does not drift when the figure height changes
                   loc="upper center",
                   bbox_to_anchor=((L + (W - L - R) / 2) / W, (B - 9) / H),
                   ncol=3, fontsize=FS - 0.5, frameon=True, handlelength=1.0,
                   columnspacing=1.0, handletextpad=0.4, borderpad=0.5,
                   labelspacing=0.45, edgecolor="#CCCCCC", facecolor="#FFFFFF")

        if save_path is None:
            save_path = regime_out_dir("granularity") / \
                f"risk_return_tradeoff_{direction.upper()}.pdf"
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path)
        fig.savefig(save_path.with_suffix(".png"), dpi=dpi)
        plt.close(fig)

    if verbose:
        print(f"          both improve {n_both}/{n} · lower return "
              f"{n_mech}/{n} · Sharpe declines {n_sr}/{n}")
        print(f"  [saved] {save_path}")
    return save_path


# ════════════════════════════════════════════════════════════════════════════
#  Market-vs-filter figure  (the direction flip is the market's; the gap is ours)
# ════════════════════════════════════════════════════════════════════════════
# Four rows (backbones) x two columns (selective precision, return per trade).
# Each panel runs CPCV -> held-out test. Per direction: a solid line for
# counter-trend bars, a dashed line for with-trend bars, the shaded band between
# them being the counter-trend advantage, and a dotted line in the same colour
# for the UNFILTERED forecaster. A strip under each panel gives the bullish share
# of the bars that design actually evaluates, per backbone.
#
# This figure answers R5-W3 by conceding its premise. The reviewer asks whether
# M2 DOWN's strength is methodological or regime-driven; the dotted lines show
# the direction flip happening with no filter at all (28/28 cells), so the paper
# never defends it. What the filter contributes is the shaded gap, which stays
# positive in 16/16 backbone x direction x period cells despite the bullish share
# falling by 20-33 points between the two designs.
#
# Everything is computed WITHIN a time group before averaging - CPCV inside each
# (path, quarter), test inside each calendar quarter - per the anti-pooling rule
# in CLAUDE.md. Pooling across time reverses the sign of this effect.

_MV_M1C = "#9A9A9A"


def _mv_collect(m1: str, direction: str, granularities, m2_models,
                edge_root: Path, test_root: Path, jm_root: Path,
                raw_root: Path, fee: float = 0.002, min_n: int = 5) -> Dict[str, float]:
    """Counter-trend / with-trend levels for one (backbone, direction), both periods.

    Returns selective precision and mean net return for M2 in each regime, plus
    the unfiltered M1 level, for CPCV and for the held-out test.
    """
    counter = "Bullish" if direction.upper() == "DOWN" else "Bearish"
    out: Dict[str, float] = {}

    acc = {k: [] for k in ("pc", "pw", "rc", "rw", "m1p", "m1r")}
    for g in granularities:
        for m2 in m2_models:
            f = Path(edge_root) / m1 / m2 / direction.upper() / g / "cpcv_samples.csv"
            if not f.exists():
                continue
            s = pd.read_csv(f, parse_dates=["date"]).rename(columns={"path": "_fold"})
            d = _attach_regimes(s, g, jm_root, raw_root).dropna(subset=["regime_name"])
            if not len(d):
                continue
            d["_q"] = pd.PeriodIndex(d["date"], freq="Q").astype(str)
            d["_net"] = d["ret"] - fee
            per = {k: [] for k in acc}
            # gap inside each (path, quarter); averaged over quarters then paths
            for _, gr in d.groupby(["_fold", "_q"]):
                c = gr[gr["regime_name"] == counter]
                w = gr[gr["regime_name"] != counter]
                if len(c) < min_n or len(w) < min_n:
                    continue
                cs, ws = c[c["sel"].astype(bool)], w[w["sel"].astype(bool)]
                if len(cs) and len(ws):
                    per["pc"].append(cs["y_true"].mean() * 100)
                    per["pw"].append(ws["y_true"].mean() * 100)
                    per["rc"].append(cs["_net"].mean() * 100)
                    per["rw"].append(ws["_net"].mean() * 100)
                both = pd.concat([c, w])
                per["m1p"].append(both["y_true"].mean() * 100)
                per["m1r"].append(both["_net"].mean() * 100)
            for k, v in per.items():
                if v:
                    acc[k].append(float(np.mean(v)))
    for k, v in acc.items():
        out[f"cpcv_{k}"] = float(np.mean(v)) if v else np.nan

    acc = {k: [] for k in ("pc", "pw", "rc", "rw", "m1p", "m1r")}
    for g in granularities:
        for m2 in m2_models:
            try:
                t = _test_trades_with_regime(m1, m2, direction.upper(), g,
                                             "Utility_Score_NoCal", test_root, jm_root)
            except Exception:
                t = None
            if t is None or not len(t):
                continue
            t = t.copy()
            t["_q"] = pd.PeriodIndex(pd.to_datetime(t["date"]), freq="Q").astype(str)
            t["_net"] = t["return"]  # held-out trades are already net of the fee
            per = {k: [] for k in acc}
            for _, gr in t.groupby("_q"):
                c = gr[gr["regime_name"] == counter]
                w = gr[gr["regime_name"] != counter]
                if len(c) < min_n or len(w) < min_n:
                    continue
                cs = c[c["m2_approved"].astype(bool)]
                ws = w[w["m2_approved"].astype(bool)]
                if len(cs) and len(ws):
                    per["pc"].append(cs["label"].mean() * 100)
                    per["pw"].append(ws["label"].mean() * 100)
                    per["rc"].append(cs["_net"].mean() * 100)
                    per["rw"].append(ws["_net"].mean() * 100)
                both = pd.concat([c, w])
                per["m1p"].append(both["label"].mean() * 100)
                per["m1r"].append(both["_net"].mean() * 100)
            for k, v in per.items():
                if v:
                    acc[k].append(float(np.mean(v)))
    for k, v in acc.items():
        out[f"test_{k}"] = float(np.mean(v)) if v else np.nan
    return out


def selection_quality_table(backbones=("Kronos", "Fincast", "Chronos2", "Tirex"),
                            m2_models=("rf", "tabpfn", "tabicl", "autogluon", "ctts"),
                            granularities=None,
                            edge_root: Path = Path("Output/Analysis/Edge_NoCal"),
                            test_root: Path = Path("Output"),
                            thres_folder: str = "Utility_Score_NoCal",
                            save_path=Path("Output/Analysis/SelectionQuality/selection_quality.csv"),
                            verbose: bool = True) -> pd.DataFrame:
    """Forecaster alone vs. forecaster + estimator, cross-validated and held-out (Table N5).

    CPCV comes from ``cpcv_paths.csv``: per-path selective precision and mean
    return (already net of the fee), averaged over paths, then over the estimators
    and granularities. The held-out window comes from ``10_backtest_cash_trades.csv``,
    whose returns the cash engine has already charged the fee on, so no fee is
    subtracted here. A cell clears break-even when its mean net return per trade is
    positive, which is the break-even precision condition of the paper.
    ``dprec_*`` is the held-out precision gain of each estimator over its forecaster,
    with the number of granularities in which it is positive.
    """
    G = list(granularities or ["1d", "12h", "8h", "6h", "4h", "2h", "1h"])
    rows = []
    for b in backbones:
        for d in ("DOWN", "UP"):
            cv1p, cv1r, cv2p, cv2r, te1p, te1r, te2p, te2r = ([] for _ in range(8))
            gain = {m: [] for m in m2_models}
            for g in G:
                for i, m in enumerate(m2_models):
                    p = pd.read_csv(Path(edge_root) / b / m / d / g / "cpcv_paths.csv")
                    cv2p.append(p["sel_precision"].mean())
                    cv2r.append(p["sel_mean_ret"].mean())
                    t = pd.read_csv(Path(test_root) / b / m / d / thres_folder / f"{g}_tp"
                                    / "10_backtest_cash_trades.csv")
                    a = t[t["m2_approved"].astype(bool)]
                    te2p.append(a["label"].mean()); te2r.append(a["return"].mean())
                    gain[m].append(a["label"].mean() - t["label"].mean())
                    if i == 0:
                        s = pd.read_csv(Path(edge_root) / b / m / d / g / "cpcv_samples.csv")
                        cv1p.append(s.groupby("path")["y_true"].mean().mean())
                        cv1r.append(p["m1_mean_ret"].mean())
                        te1p.append(t["label"].mean()); te1r.append(t["return"].mean())
            row = {"m1": b, "dir": d,
                   "cpcv_m1_prec": 100 * np.mean(cv1p), "cpcv_m2_prec": 100 * np.mean(cv2p),
                   "cpcv_m1_ret": 100 * np.mean(cv1r), "cpcv_m2_ret": 100 * np.mean(cv2r),
                   "cpcv_m1_be": 100 * np.mean(np.asarray(cv1r) > 0),
                   "cpcv_m2_be": 100 * np.mean(np.asarray(cv2r) > 0),
                   "test_m1_prec": 100 * np.mean(te1p), "test_m2_prec": 100 * np.mean(te2p),
                   "test_m1_ret": 100 * np.mean(te1r), "test_m2_ret": 100 * np.mean(te2r),
                   "test_m1_be": 100 * np.mean(np.asarray(te1r) > 0),
                   "test_m2_be": 100 * np.mean(np.asarray(te2r) > 0)}
            for m in m2_models:
                row[f"dprec_{m}"] = 100 * np.mean(gain[m])
                row[f"npos_{m}"] = int((np.asarray(gain[m]) > 0).sum())
            rows.append(row)
    out = pd.DataFrame(rows)
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(save_path, index=False)
        if verbose:
            print(f"  [saved] {save_path}")
    return out


def plot_market_vs_filter(backbones=("Kronos", "Fincast", "Chronos2", "Tirex"),
                          granularities=None,
                          m2_models=("rf", "tabicl", "tabpfn", "autogluon", "ctts"),
                          save_path: str | Path | None = None,
                          edge_root: Path = Path("Output/Analysis/Edge_NoCal"),
                          test_root: Path = Path("Output"),
                          jm_root: Path = _JM_GLOBAL_ROOT,
                          raw_root: Path = Path("Data_MLA/Raw"),
                          dpi: int = 600, verbose: bool = True) -> Path:
    """CPCV -> test, per backbone: the market moves the levels, the filter holds the gap."""
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    G = list(granularities or ["1d", "12h", "8h", "6h", "4h", "2h", "1h"])
    DAT, MIX = {}, {}
    for b in backbones:
        for d in ("DOWN", "UP"):
            DAT[(b, d)] = _mv_collect(b, d, G, m2_models, edge_root, test_root,
                                      jm_root, raw_root)
        try:
            mx = _cvt_regime_mix(b, ("DOWN",), G, m2_models, edge_root, jm_root, raw_root)
            MIX[b] = (mx.get("cpcv", np.nan), mx.get("test", np.nan))
        except Exception:
            MIX[b] = (np.nan, np.nan)

    MM = 1 / 25.4
    FS, AXLW = 8.5, 0.8
    GREY, DK = "#666666", "#333333"
    DNC, UPC = "#B03030", "#2E8B57"
    # 2 rows (metric) x 4 cols (backbone): wide and short, so the figure spans
    # the text width instead of a full page.
    # GY must still clear the regime-mix bars and their "% bullish" labels, which
    # hang ~0.20 axes units below each panel; Bm now only has to seat the legend,
    # the italic strapline below it having been dropped.
    # The regime-mix strips hang under the bottom row only, so GY just separates
    # the two rows.
    W, H = 210, 98
    L, R, T, Bm, GX, GY = 16, 7, 11, 23, 11, 6
    pw = (W - L - R - 3 * GX) / 4
    ph = (H - T - Bm - GY) / 2
    x = [0, 1]
    SPEC = [("p", 50.0, (31, 73), [40, 50, 60], "Selective\nprecision (%)", "{:+.1f}"),
            ("r", 0.0, (-2.2, 2.5), [-2, -1, 0, 1, 2], "Return per\ntrade (%)", "{:+.2f}")]

    with plt.rc_context({"font.family": "DejaVu Sans", "axes.linewidth": AXLW,
                         "xtick.major.width": AXLW, "ytick.major.width": AXLW}):
        fig = plt.figure(figsize=(W * MM, H * MM))
        for r, b in enumerate(backbones):
            for k, (m, ref, ylim, ticks, ylab, fmt) in enumerate(SPEC):
                ax = fig.add_axes([(L + r * (pw + GX)) / W,
                                   (Bm + (1 - k) * (ph + GY)) / H, pw / W, ph / H])
                ax.axhline(ref, color="#999999", lw=AXLW, ls=(0, (4, 3)), zorder=2)
                kc, kw = ("pc", "pw") if m == "p" else ("rc", "rw")
                km1 = "m1p" if m == "p" else "m1r"
                for d, c in (("DOWN", DNC), ("UP", UPC)):
                    v = DAT[(b, d)]
                    ct = [v[f"cpcv_{kc}"], v[f"test_{kc}"]]
                    wt = [v[f"cpcv_{kw}"], v[f"test_{kw}"]]
                    m1v = [v[f"cpcv_{km1}"], v[f"test_{km1}"]]
                    ax.plot(x, m1v, color=c, lw=1.9, ls=(0, (1.2, 1.4)), marker="s",
                            ms=5.2, mfc=c, mec="#FFFFFF", mew=0.6, alpha=1.0, zorder=7)
                    for k2 in (0, 1):
                        ax.plot([x[k2]] * 2, [wt[k2], ct[k2]], color=c, lw=9,
                                alpha=0.15, solid_capstyle="butt", zorder=3)
                        # labels outside the data span so they never sit on a band
                        # Labels go OUTBOARD of each band (left of CPCV, right of
                        # test) so they never sit on the crossing lines. The wider
                        # 2x4 canvas leaves room for this; the x-limits below are
                        # padded to keep them inside the axes.
                        # When the two directions' bands nearly coincide (Tirex,
                        # precision) their midpoints collide, so push the labels
                        # apart vertically in that case only.
                        ymid = (ct[k2] + wt[k2]) / 2
                        omid = None
                        od = "UP" if d == "DOWN" else "DOWN"
                        ov = DAT[(b, od)]
                        omid = (ov[f"{'cpcv' if k2 == 0 else 'test'}_{kc}"]
                                + ov[f"{'cpcv' if k2 == 0 else 'test'}_{kw}"]) / 2
                        span = ylim[1] - ylim[0]
                        dy = 0.0
                        if abs(ymid - omid) < 0.075 * span:
                            dy = 7.0 if d == "DOWN" else -7.0
                        ax.annotate(fmt.format(ct[k2] - wt[k2]), (x[k2], ymid),
                                    textcoords="offset points",
                                    xytext=(-8 if k2 == 0 else 8, dy),
                                    ha="right" if k2 == 0 else "left", va="center",
                                    fontsize=FS - 1.0, fontweight="bold", color=c,
                                    zorder=8)
                    ax.plot(x, ct, color=c, lw=2.3, marker="o", ms=5.8, mfc=c,
                            mec="none", zorder=6)
                    ax.plot(x, wt, color=c, lw=2.3, ls=(0, (3, 2)), marker="o",
                            ms=5.8, mfc="#FFFFFF", mec=c, mew=1.5, zorder=6)
                ax.set_xlim(-0.78, 1.78)
                ax.set_xticks(x)
                ax.set_xticklabels(["", ""])
                ax.set_ylim(*ylim)
                ax.set_yticks(ticks)
                ax.set_yticklabels([f"{t:g}" for t in ticks], fontsize=FS)
                ax.spines["left"].set_bounds(*ylim)
                for sp in ("top", "right"):
                    ax.spines[sp].set_visible(False)
                if k == 0 and r == 0:
                    ax.set_title("", fontsize=FS)
                if r == 0:
                    ax.text(-0.29, 0.5, ylab, transform=ax.transAxes,
                            rotation=90, ha="center", va="center", fontsize=FS + 0.3,
                            linespacing=1.05,
                            fontweight="bold", color=DK)
                if k == 0:
                    # just clear of the CPCV/Test labels, which sit at 1.012
                    ax.text(0.5, 1.15, _GR_NICE.get(b, b), transform=ax.transAxes,
                            ha="center", va="bottom", fontsize=FS + 0.6,
                            fontweight="bold", color=DK)
                tr = ax.get_xaxis_transform()
                cvp, tep = MIX[b]
                for k2, mu in ((0, cvp), (1, tep)):
                    if k == 0:
                        ax.text(x[k2], 1.012, "CPCV" if k2 == 0 else "Test",
                                transform=tr, ha="center", va="bottom",
                                fontsize=FS - 0.8, color=DK)
                        continue
                    if not np.isfinite(mu):
                        continue
                    y0, hgt = -0.165, 0.075
                    ax.add_patch(plt.Rectangle((x[k2] - 0.19, y0), 0.38 * mu / 100, hgt,
                                               transform=tr, facecolor=UPC, alpha=0.42,
                                               lw=0, clip_on=False, zorder=5))
                    ax.add_patch(plt.Rectangle((x[k2] - 0.19 + 0.38 * mu / 100, y0),
                                               0.38 * (1 - mu / 100), hgt, transform=tr,
                                               facecolor=DNC, alpha=0.42, lw=0,
                                               clip_on=False, zorder=5))
                    ax.text(x[k2], y0 - 0.030, f"{mu:.0f}%\nbullish", transform=tr,
                            ha="center", va="top", fontsize=FS - 1.0, color=DK,
                            linespacing=1.0)

        hh = [Line2D([], [], color=GREY, lw=1.3, ls=(0, (1, 1.8)), marker="s", ms=4.0),
              Patch(facecolor=DNC), Patch(facecolor=UPC),
              Line2D([], [], color=GREY, lw=2.3, marker="o", ms=5.2, mfc=GREY, mec="none"),
              Line2D([], [], color=GREY, lw=2.3, ls=(0, (3, 2)), marker="o", ms=5.2,
                     mfc="#FFFFFF", mec=GREY, mew=1.5)]
        # anchored a fixed distance under the bottom row rather than at a
        # fraction of the canvas, so the gap holds if the height changes
        fig.legend(hh, ["Forecaster M1", "M2 DOWN", "M2 UP", "Counter-trend bars",
                        "With-trend bars"], loc="upper center",
                   bbox_to_anchor=(0.5, (Bm - 12.5) / H),
                   ncol=5, fontsize=FS - 0.5, frameon=True, handlelength=1.8,
                   columnspacing=1.5, borderpad=0.6, edgecolor="#CCCCCC",
                   facecolor="#FFFFFF")

        if save_path is None:
            save_path = regime_out_dir("market_vs_filter") / "market_vs_filter.pdf"
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path)
        fig.savefig(save_path.with_suffix(".png"), dpi=dpi)
        plt.close(fig)

    if verbose:
        n_pos = 0
        for b in backbones:
            for d in ("DOWN", "UP"):
                v = DAT[(b, d)]
                for per in ("cpcv", "test"):
                    if v[f"{per}_pc"] - v[f"{per}_pw"] > 0:
                        n_pos += 1
        print(f"          counter-trend gap positive in {n_pos}/"
              f"{2 * 2 * len(backbones)} backbone x direction x period cells")
        print(f"  [saved] {save_path}")
    return save_path


# ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
# ┃  Equity curves: M1 vs M1+M2 vs Buy & Hold, held-out and cross-validated ┃
# ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
# Reuses the paper's own cash engine verbatim, so the curves reproduce the
# published totals exactly:
#   * M1  = every proposed trade, flat-sized   (flat_size=True is REQUIRED --
#           otherwise M1 borrows M2's probabilities and stops being a
#           model-free yardstick; see _build_cash_equity's docstring)
#   * M2  = the accepted subset, confidence-sized
#   * B&H = equal-weight over the 20 assets, each normalised to its own first
#           price inside the window.
#
# Two window conventions matter and are handled explicitly:
#   1. The simulation timeline is the PRICE index of the panel's own window
#      (engine.py sets full_idx = bh_equity.index). Using trade dates instead
#      changes how many bars the horizon spans and shifts the M1 curve.
#   2. The benchmark is pinned to the configured split dates, not to each
#      panel's first sample. CPCV samples reach back before start_date for
#      model context (1h by 37 days), and the held-out set carries a few
#      trades stamped on val_end itself; without the bound the same benchmark
#      appears to depend on sampling granularity.

_EQ_C_M2   = "#1F6FB4"   # the framework
_EQ_C_M1   = "#C0392B"   # forecaster alone
_EQ_C_BH   = "#E8873A"   # buy & hold (orange: reads clearly over the teal
                         # with-trend exposure band on the DOWN row)
_EQ_C_CT   = "#8E44AD"   # counter-trend open positions (exposure band)
_EQ_C_WT   = "#4A9E9E"   # with-trend open positions (exposure band)
_EQ_GRID   = "#E8E8E8"
_EQ_FS     = 11.0
# The exposure band occupies this fraction of a panel's height, drawn behind the
# curves on a twin axis. Heights are square-rooted: with-trend exposure peaks
# several times higher than counter-trend, and on a linear scale those peaks
# flatten the counter-trend component into an unreadable sliver.
_EQ_BAND_H = 0.30


def _strip_pct(label: str) -> str:
    """Drop a trailing "  +12.3%" from a curve label: the returns are annotated
    per panel, so the shared legend only needs the series name."""
    import re as _re
    return _re.sub(r"\s{2,}[+-][\d.]+%$", "", label)


def _eq_style(ax, title=None, fs=_EQ_FS):
    ax.grid(True, color=_EQ_GRID, lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.axhline(0, color="#333333", lw=0.8, zorder=2)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(labelsize=fs - 0.5)
    if title:
        ax.set_title(title, fontsize=fs + 1, fontweight="bold", pad=6)


def plot_equity_curves(m1: str = "Kronos",
                       m2: str = "rf",
                       grans=("1d", "12h", "8h"),
                       window: str = "held-out",
                       save_path=None,
                       src_root: Path = Path("/home/pablo/M2_DS/SecondaryModel/src"),
                       thres_folder: str = "Utility_Score_NoCal",
                       horizon: int = 7,
                       fee: float = 0.002,
                       start_dates: dict = None,
                       dpi: int = 600,
                       exposure_band=None,
                       show_stats: bool = False,
                       regime_root: Path = None,
                       verbose: bool = True) -> Path:
    """Equity for one forecaster/estimator pair: rows are directions, columns granularities.

    ``regime_root`` selects the regime labels for the exposure bands (default the
    stored fits; Output/Jump_Models/_global_warmup holds labels extended back to
    the window start, used for plotting only).

    ``show_stats`` writes the Sharpe ratio and maximum drawdown of M1 and M1+M2
    under each curve's end-of-curve return, in the curve's colour, computed with
    the engine's own helpers and annualisation as in the published tables.

    ``window`` is ``"held-out"`` (one curve per series) or ``"cpcv"`` (the median
    of the five reconstructed paths with their min-max band). Every panel shares
    one vertical scale so the two directions are directly comparable, and the
    direction is written once per row, rotated on the right-hand edge, instead of
    being repeated in every panel title.

    ``exposure_band`` names the direction ("DOWN" / "UP") whose row carries a
    volume-style band of the positions open at each bar, split into counter- and
    with-trend. Alignment is defined against the prevailing regime, so the band
    belongs on the direction that regime favours: DOWN on the held-out window,
    UP under cross-validation. ``None`` disables it.
    """
    import matplotlib.dates as mdates
    from matplotlib.patches import Patch
    from Utils.backtest.engine import (_build_cash_equity, _load_raw_close_prices,
                                       _calc_drawdown, _calc_sharpe,
                                       _annualization_factor, _equity_horizon_returns)
    import copy as _copy
    import yaml as _yaml

    grans = list(grans)
    # Name the actual models in the legend rather than the generic M1/M2 roles.
    _M2_NICE = {"rf": "Random Forest", "tabpfn": "TabPFN", "tabicl": "TabICL",
                "autogluon": "AutoGluon", "ctts": "CTTS"}
    _M1_NICE = {"kronos": "Kronos", "fincast": "FinCast",
                "chronos2": "Chronos-2", "tirex": "Tirex"}
    _n1 = _M1_NICE.get(m1.lower(), m1)
    _n2 = _M2_NICE.get(m2.lower(), m2)
    # Generic roles in the legend; the caption names the actual backbone and
    # estimator, so the same figure reads identically across backbones.
    _lab_pair, _lab_solo = "$M1$+$M2$", "$M1$ alone"

    cfg_base = _yaml.safe_load(open(src_root / "config.yaml"))
    split = cfg_base["data"]["split"]
    if start_dates is None:
        # the test split opens the day after val_end; the cross-validated window is
        # the configured start_date .. val_end. Both come from config.yaml so the
        # benchmark always matches the window the paper declares.
        _test_open = (pd.Timestamp(split["val_end"]) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        start_dates = {"held-out": (_test_open, split["end_date"]),
                       "cpcv":     (split["start_date"], split["val_end"])}
    w0, w1 = (pd.Timestamp(x) for x in start_dates[window])

    def _cfg():
        c = _copy.deepcopy(cfg_base)
        c["experiment"]["m1"] = c["data"]["load"]["m1"] = m1.lower()
        c["paths"]["csv_dir"] = str(src_root / "Data_MLA" / m1.capitalize() / "Crypto/TP/horizon_7")
        return c

    # one shared benchmark for the whole figure
    bh_eq = None
    # Always daily closes, so the benchmark is one number across every figure
    # (built from grans[0] it read -35.8% in the coarse set and -34.3% in the fine).
    raw = _load_raw_close_prices(_cfg(), "1d", direction="down")
    if len(raw):
        raw = raw[(raw["date"] >= w0) & (raw["date"] <= w1)]
        piv = raw.pivot_table(index="date", columns="asset", values="close")
        if len(piv):
            first = piv.apply(lambda c: c.dropna().iloc[0] if c.notna().any() else np.nan)
            bh_eq = (piv / first).mean(axis=1)

    # taller panels than a naive width*n keeps 4-column figures legible
    fig, axes = plt.subplots(2, len(grans),
                             figsize=(4.05 * len(grans), 3.3 * 2 + 0.4),
                             sharex="col", sharey=True)
    axes = np.atleast_2d(axes)
    ylo, yhi = np.inf, -np.inf
    records = []
    band_data = {}
    shared_handles, shared_labels, handles = [], [], []

    def _end_labels(ax, panel_ends, ylo, yhi):
        # Each value sits at the height its own curve ends at, in data
        # coordinates, so it reads as that curve's label. Where two curves
        # finish close together the labels would collide, so they are
        # nudged apart by a minimum separation while keeping their order.
        lo_, hi_ = ylo - 0.04 * (yhi - ylo), yhi + 0.04 * (yhi - ylo)
        span = hi_ - lo_
        min_gap = 0.075 * span
        pts = sorted(panel_ends, key=lambda t: -t[0])
        ys = [t[0] for t in pts]
        for i in range(1, len(ys)):
            if ys[i - 1] - ys[i] < min_gap:
                ys[i] = ys[i - 1] - min_gap
        for (val, col, _), yy in zip(pts, ys):
            ax.annotate("%+.1f%%" % val, xy=(1.015, yy),
                        xycoords=("axes fraction", "data"),
                        color=col, fontsize=_EQ_FS - 1.0, fontweight="bold",
                        ha="left", va="center", annotation_clip=False)

    stat_rows = {}
    for r, d in enumerate(("DOWN", "UP")):
        for c, g in enumerate(grans):
            ax = axes[r, c]
            # Sharpe of non-overlapping horizon returns, annualised as the engine
            # does for the published tables: sqrt(bars per year / horizon).
            ann = _annualization_factor(g) / np.sqrt(horizon)
            series = []
            panel_ends = []
            if window == "held-out":
                tp = (src_root / "Output" / m1 / m2 / d / thres_folder /
                      f"{g}_tp" / "10_backtest_cash_trades.csv")
                if not tp.exists():
                    ax.set_visible(False); continue
                # All held-out trades are kept, exactly as the published backtest
                # (10_backtest_cash_ROI.txt) and Tables 19/20 use them. A handful
                # at 2h/1h carry a 2025-12-31 timestamp: they are after val_end
                # (midnight), so legitimately test data. The x-axis ticks come from
                # the benchmark window, so the panels still read from Jan 2026.
                tr = pd.read_csv(tp, parse_dates=["date"]).sort_values("date")
                if len(tr) < 3:
                    ax.set_visible(False); continue
                rt = _load_raw_close_prices(_cfg(), g, direction=d.lower())
                tl = None
                if len(rt):
                    # The timeline must run `horizon` bars past the LAST ENTRY, not stop at
                    # it: a trade entered on the final bar resolves `horizon` bars later, and
                    # clipping at the entry silently drops that tail of P&L from the curve
                    # (~14d at 1d, ~5d at 8h) while the benchmark still runs to end_date.
                    # Per-trade returns in Table 20 always included it; only the plot did not.
                    bars = pd.DatetimeIndex(sorted(rt["date"].unique()))
                    last = bars.searchsorted(tr["date"].max(), side="left")
                    rt = rt[(rt["date"] >= tr["date"].min()) &
                            (rt["date"] <= bars[min(last + horizon, len(bars) - 1)])]
                    pv = rt.pivot_table(index="date", columns="asset", values="close")
                    if len(pv):
                        tl = pv.index
                if tl is None:
                    tl = pd.DatetimeIndex(sorted(tr["date"].unique()))
                # open positions marked to market at the bar closes
                _px = pv if tl is not None and len(rt) else None
                m2_eq, _ = _build_cash_equity(tr[tr["m2_approved"].astype(bool)], tl, horizon,
                                              close=_px, fee=fee, direction=d)
                m1_eq, _ = _build_cash_equity(tr, tl, horizon, flat_size=True,
                                              close=_px, fee=fee, direction=d)
                series = [(m2_eq, _EQ_C_M2, "-", 1.6, _lab_pair),
                          (m1_eq, _EQ_C_M1, "-", 1.3, _lab_solo)]
                # Counter-trend exposure, DOWN only: the alignment is defined
                # against the prevailing regime, so on UP it would compare a
                # different (and far more common) regime and read as a much
                # larger band for a weaker effect.
                if exposure_band and exposure_band.upper() in (d, "BOTH"):
                    _lb, _ = load_regime_labels(g, root=regime_root or _JM_GLOBAL_ROOT)
                    _nm = _regime_direction_map(g, _lb)
                    _ln = _lb.assign(regime_name=_lb["regime"].map(_nm))[
                        ["asset", "date", "regime_name"]]
                    _tj = tr.merge(_ln, on=["asset", "date"], how="left")
                    _tj = _tj[_tj["regime_name"].notna()]
                    _tj = _tj[_tj["m2_approved"].astype(bool)]
                    _counter = "Bullish" if d == "DOWN" else "Bearish"
                    _bm = {dt: i for i, dt in enumerate(tl)}
                    _oc, _ow = np.zeros(len(tl)), np.zeros(len(tl))
                    for _dt, _rn in zip(_tj["date"].values,
                                        _tj["regime_name"].values):
                        _i = _bm.get(pd.Timestamp(_dt))
                        if _i is None:
                            continue
                        _j = min(_i + horizon, len(tl))
                        (_oc if _rn == _counter else _ow)[_i:_j] += 1.0
                    band_data[(r, c)] = (tl, _oc, _ow)
            else:
                f = (src_root / "Output/Analysis/Edge_NoCal" / m1 / m2 / d / g /
                     "cpcv_samples.csv")
                if not f.exists():
                    ax.set_visible(False); continue
                s = pd.read_csv(f, parse_dates=["date"]).sort_values("date")
                s = s[s["date"] >= w0]
                if s.empty or "path" not in s.columns:
                    ax.set_visible(False); continue
                tl = pd.DatetimeIndex(sorted(s["date"].unique()))
                # bar closes over the CV window, to mark open positions to market
                _rc = _load_raw_close_prices(_cfg(), g, direction=d.lower())
                _px = _rc[(_rc["date"] >= tl[0]) & (_rc["date"] <= tl[-1])].pivot_table(
                    index="date", columns="asset", values="close")
                curves = {}
                for p_, gg in s.groupby("path"):
                    sel = gg[gg["sel"] == 1].copy()
                    if len(sel) < 5:
                        continue
                    sel = sel.rename(columns={"prob": "m2_prob"})
                    sel["return"] = sel["ret"] - fee
                    curves[p_], _ = _build_cash_equity(sel, tl, horizon, close=_px, fee=fee,
                                                       direction=d)
                if not curves:
                    ax.set_visible(False); continue
                M = pd.concat([e.rename(k) for k, e in sorted(curves.items())], axis=1).ffill()
                med = M.median(axis=1)
                ax.fill_between(M.index, (M.min(axis=1) - 1) * 100, (M.max(axis=1) - 1) * 100,
                                color=_EQ_C_M2, alpha=.18, lw=0, zorder=3,
                                label="%s  range of %d paths" % (_lab_pair, M.shape[1]))
                df_m1 = s.copy(); df_m1["return"] = df_m1["ret"] - fee
                m1_eq, _ = _build_cash_equity(df_m1, tl, horizon, flat_size=True, close=_px,
                                              fee=fee, direction=d)
                series = [(med, _EQ_C_M2, "-", 1.6, "%s  median" % _lab_pair),
                          (m1_eq, _EQ_C_M1, "-", 1.3, _lab_solo)]
                # Exposure for the band is taken from the path whose final
                # equity is the median one, so the bars belong to the same walk
                # the plotted median curve ends on rather than to all five at
                # once (which would multiply the position counts by five).
                if exposure_band and exposure_band.upper() in (d, "BOTH"):
                    _ends = M.iloc[-1].sort_values()
                    _pmed = _ends.index[len(_ends) // 2]
                    _sp = s[(s["path"] == _pmed) & (s["sel"] == 1)]
                    _lb, _ = load_regime_labels(g, root=regime_root or _JM_GLOBAL_ROOT)
                    _nm = _regime_direction_map(g, _lb)
                    _ln = _lb.assign(regime_name=_lb["regime"].map(_nm))[
                        ["asset", "date", "regime_name"]]
                    _tj = _sp.merge(_ln, on=["asset", "date"], how="left")
                    _tj = _tj[_tj["regime_name"].notna()]
                    _counter = "Bullish" if d == "DOWN" else "Bearish"
                    _bm = {dt: i for i, dt in enumerate(tl)}
                    _oc, _ow = np.zeros(len(tl)), np.zeros(len(tl))
                    for _dt, _rn in zip(_tj["date"].values,
                                        _tj["regime_name"].values):
                        _i = _bm.get(pd.Timestamp(_dt))
                        if _i is None:
                            continue
                        _j = min(_i + horizon, len(tl))
                        (_oc if _rn == _counter else _ow)[_i:_j] += 1.0
                    band_data[(r, c)] = (tl, _oc, _ow)
                ylo = min(ylo, float((M.min(axis=1) - 1).min() * 100))
                yhi = max(yhi, float((M.max(axis=1) - 1).max() * 100))

            for eq, col, ls, lw, lab in series:
                hr = _equity_horizon_returns(eq, horizon) if len(eq) > horizon else np.array([])
                ret = (eq.iloc[-1] - 1) * 100
                ax.plot(eq.index, (eq - 1) * 100, color=col, lw=lw, ls=ls, zorder=5,
                        label="%s  %+.1f%%" % (lab, ret))
                records.append(dict(window=window, direction=d, gran=g, series=lab,
                                    ret=ret, mdd=_calc_drawdown(eq.values) * 100,
                                    sr=_calc_sharpe(hr, ann)))
                panel_ends.append((ret, col, (records[-1]["sr"], records[-1]["mdd"])
                                   if show_stats else None))
                ylo = min(ylo, float((eq - 1).min() * 100))
                yhi = max(yhi, float((eq - 1).max() * 100))
            if bh_eq is not None and len(bh_eq):
                ax.plot(bh_eq.index, (bh_eq - 1) * 100, color=_EQ_C_BH, lw=1.3, ls="-",
                        zorder=4, label="Buy & Hold  %+.1f%%" % ((bh_eq.iloc[-1] - 1) * 100))
                panel_ends.append((float((bh_eq.iloc[-1] - 1) * 100), _EQ_C_BH, None))
                ylo = min(ylo, float((bh_eq - 1).min() * 100))
                yhi = max(yhi, float((bh_eq - 1).max() * 100))

            # Cumulative returns for this panel, written just right of the axes in
            # each curve's own colour and at the shared font size. They are stacked
            # in value order with a fixed vertical step, so they cannot overlap.
            if panel_ends:
                _end_labels(ax, panel_ends, ylo, yhi)
                if show_stats:
                    # M1 first, as in the legend
                    stat_rows[(r, c)] = [t_[2] + (t_[1],) for t_ in panel_ends
                                         if t_[2] is not None][::-1]

            # granularity only: the direction is written once per row, on the right
            _eq_style(ax, g if r == 0 else None)
            # One shared legend for the whole figure (added after the loop); each
            # panel instead carries its own cumulative returns, written at the end
            # of each curve in that curve's colour so they cannot be confused.
            if not handles:
                handles, labels_ = ax.get_legend_handles_labels()
                shared_handles.extend(handles)
                shared_labels.extend([_strip_pct(l) for l in labels_])
            if c == 0:
                ax.set_ylabel("Cumulative return (%)", fontsize=_EQ_FS)
            # Direction written as a light watermark in the centre of every panel,
            # behind the curves and bands.
            ax.text(0.5, 0.5, "%s predictions" % d.capitalize(), transform=ax.transAxes, ha="center",
                    va="center", fontsize=_EQ_FS + 3, fontweight="bold",
                    color="#E3E3E3", zorder=0.5)
            ref = bh_eq if bh_eq is not None else None
            if ref is not None and len(ref):
                t0, t1 = mdates.date2num(ref.index[0]), mdates.date2num(ref.index[-1])
                ax.set_xticks(np.linspace(t0, t1, 3))
                ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
            for lb in ax.get_xticklabels():
                lb.set_rotation(0); lb.set_ha("center")

    pad = 0.04 * (yhi - ylo)
    # ONE shared y-axis for both rows, running from the lowest point reached by
    # any plotted curve to the highest. The exposure band is fitted INSIDE the
    # bottom of that range rather than added beneath it: extending the axis to
    # make room would push the floor past every curve and leave the unbanded row
    # with a large empty margin.
    _floor = ylo - pad

    _top = yhi + pad
    for ax in axes.ravel():
        if ax.get_visible():
            ax.set_ylim(_floor, _top)
    _band_span = _EQ_BAND_H * (_top - _floor) if band_data else 0.0
    # Exposure bands are drawn last, once the shared y-limits are settled: each
    # sits on a twin axis behind its panel, scaled so the tallest bar in the
    # figure fills _EQ_BAND_H of the panel height. Heights are square-rooted, so
    # the bars show shape and composition, never a readable magnitude.
    if band_data:
        _vmax = max(float((oc + ow).max()) for _, oc, ow in band_data.values())
        for (r, c), (tl_, oc_, ow_) in band_data.items():
            ax = axes[r, c]
            axv = ax.twinx()
            axv.set_zorder(ax.get_zorder() - 1)
            ax.patch.set_visible(False)
            _ct_h = np.sqrt(oc_)
            _tot_h = np.sqrt(oc_ + ow_)
            # The twin axis spans the whole panel, with the bars confined to the
            # reserved strip at its foot: the tallest bar in the figure exactly
            # fills that strip.
            _panel = _top - _floor
            axv.set_ylim(0, np.sqrt(_vmax) * _panel / _band_span)
            _x = mdates.date2num(tl_)
            _w = (_x[1] - _x[0]) if len(_x) > 1 else 1.0
            axv.bar(_x, _tot_h - _ct_h, bottom=_ct_h, width=_w, color=_EQ_C_WT,
                    lw=0, alpha=0.40, zorder=1)
            axv.plot(_x, _tot_h, color=_EQ_C_WT, lw=0.9, zorder=2)
            axv.bar(_x, _ct_h, width=_w, color=_EQ_C_CT, lw=0, alpha=0.30, zorder=3)
            axv.plot(_x, _ct_h, color=_EQ_C_CT, lw=0.9, zorder=4)
            axv.set_yticks([])
            for _sp in ("top", "left", "right"):
                axv.spines[_sp].set_visible(False)
        shared_handles.extend([
            Patch(facecolor=_EQ_C_CT, edgecolor=_EQ_C_CT, alpha=0.30, linewidth=1.0),
            Patch(facecolor=_EQ_C_WT, edgecolor=_EQ_C_WT, alpha=0.40, linewidth=1.0)])
        shared_labels.extend(["$M2$ CT open positions",
                              "$M2$ WT open positions"])

    fig.tight_layout(h_pad=1.6, w_pad=1.9, rect=(0, 0.055, 1, 1))

    # SR / MDD: one row at the top inside each panel, drawn in the background:
    # above the grid and watermark (its white backing hides the gridlines behind
    # the text) but below every curve and band, which pass over it.
    # Added after the layout, so no axes moves and the y-range is unchanged.
    _leg_y = 0.052
    for (r, c), rows_ in stat_rows.items():
        ax = axes[r, c]
        kw_ = dict(fontsize=_EQ_FS - 1.5, annotation_clip=False, zorder=1.5)
        prev_ = ax.annotate("SR / MDD", xy=(0.02, 0.975), xycoords="axes fraction",
                            ha="left", va="top", color="#555555", **kw_)
        arts_ = [prev_]
        # each value is anchored to the right edge of the previous text, so the
        # gaps hold whatever the length of the numbers
        for sr_, mdd_, col_ in rows_:
            prev_ = ax.annotate("%.2f / %.1f%%" % (sr_, mdd_), xy=(1, 0), xycoords=prev_,
                                xytext=(9, 0), textcoords="offset points", ha="left",
                                va="bottom", fontweight="bold", color=col_, **kw_)
            arts_.append(prev_)
        _rnd = fig.canvas.get_renderer()
        _inv = ax.transAxes.inverted()
        _bb = [_inv.transform(a_.get_window_extent(_rnd)) for a_ in arts_]
        x0_, y0_ = min(b_[0][0] for b_ in _bb), min(b_[0][1] for b_ in _bb)
        x1_, y1_ = max(b_[1][0] for b_ in _bb), max(b_[1][1] for b_ in _bb)
        ax.add_patch(plt.Rectangle((x0_ - 0.012, y0_ - 0.012), x1_ - x0_ + 0.024,
                                   y1_ - y0_ + 0.024, transform=ax.transAxes,
                                   facecolor="white", edgecolor="none", alpha=0.85,
                                   zorder=1.4, clip_on=False))

    # A single legend under the bottom row, horizontally centred on the middle of
    # the grid: under the 12h column with three columns, and midway between 4h and
    # 2h with four. Placed below the x labels with clear air between them.
    if shared_handles:
        _b = [axes[1, k].get_position() for k in range(len(grans))]
        if len(grans) % 2:                      # odd -> centre of the middle panel
            _cx = (_b[len(grans) // 2].x0 + _b[len(grans) // 2].x1) / 2.0
        else:                                   # even -> seam between the two middles
            _cx = (_b[len(grans) // 2 - 1].x1 + _b[len(grans) // 2].x0) / 2.0
        # With the exposure bands the legend carries five entries. Matplotlib
        # fills columns top-to-bottom, so to read as "curves on row 1, bands
        # centred on row 2" the entries are interleaved into column order and
        # the short row is padded with an invisible handle at each end.
        # One row: M1 alone, M1+M2 (its CPCV range entry first, when present),
        # Buy & Hold, then the two exposure-band entries.
        if band_data:
            def _rank(lbl):
                if "open positions" in lbl:
                    return 5 if "WT" in lbl else 4
                if lbl.startswith("$M1$ alone"):
                    return 0
                if lbl.startswith("Buy & Hold"):
                    return 3
                return 1 if "range" in lbl else 2
            _pairs = sorted(zip(shared_handles, shared_labels),
                            key=lambda t: _rank(t[1]))
            shared_handles = [h_ for h_, _ in _pairs]
            shared_labels = [l_ for _, l_ in _pairs]
        # A four-panel figure is wider and prints smaller at the same text width,
        # so its legend is scaled by the panel count to print like a three-panel one.
        _lfs = (_EQ_FS - 0.5) * max(len(grans), 3) / 3.0
        fig.legend(shared_handles, shared_labels, loc="upper center",
                   bbox_to_anchor=(_cx, _leg_y), bbox_transform=fig.transFigure,
                   ncol=len(shared_labels), fontsize=_lfs, frameon=True,
                   framealpha=.95, edgecolor="#CCCCCC", handlelength=1.8,
                   columnspacing=1.6, borderpad=0.5)

    if save_path is None:
        save_path = Path("equity_%s.png" % window.replace("-", ""))
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=dpi, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    if verbose:
        print(f"  [saved] {save_path}")
        for rec in records:
            print(f"      {rec['window']:9} {rec['direction']:4} {rec['gran']:3} "
                  f"{rec['series']:22} ret={rec['ret']:+8.2f} mdd={rec['mdd']:+8.2f} sr={rec['sr']:+6.2f}")
    return save_path


# ════════════════════════════════════════════════════════════════════════════
#  Reliability estimators: mean bar + backbone spread
# ════════════════════════════════════════════════════════════════════════════
_BW_TAB, _BW_SEQ = "#C7D9E8", "#F5C4C4"
_BW_MK = {"Kronos": "o", "Fincast": "s", "Chronos2": "^", "Tirex": "D"}


def plot_reliability_barwhisker(perf_grid: str | Path,
                                m2_models=("tabicl", "autogluon", "tabpfn", "rf", "ctts"),
                                backbones=("Kronos", "Fincast", "Chronos2", "Tirex"),
                                save_path: str | Path | None = None,
                                dpi: int = 600, verbose: bool = True) -> Path:
    """Per estimator: mean bar over backbones, with one marker per backbone.

    Two metric rows (precision gain, portfolio return) by two directions. A tight
    cluster of markers means the estimator behaves the same whichever forecaster
    it filters; a spread straddling zero is the coin flip described in the text.

    Axis limits are derived from the data rather than hardcoded, so no backbone
    marker can fall outside its panel. Each metric row shares one scale across
    the two directions; $M2$ UP portfolio return keeps its own, since it lies
    entirely below zero and sharing would flatten the other panels.
    """
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    M2, M1 = list(m2_models), list(backbones)
    LBL = {"tabicl": "TabICL", "autogluon": "AutoGluon", "tabpfn": "TabPFN",
           "rf": "Random\nForest", "ctts": "CTTS"}
    d = pd.read_csv(perf_grid)
    d["dprec"] = d["m2_prec"] - d["m1_prec"]
    SPEC = [("dprec", "Precision gain (pp)"), ("m2_ret", "Portfolio return (%)")]

    def _cell(col, dr, m1, m2):
        x = d[(d["dir"] == dr) & (d["m1"] == m1) & (d["m2"] == m2)]
        return x[col].mean()

    def _lim(col, dirs, padf=0.10):
        v = [_cell(col, dr, m1, m2) for dr in dirs for m2 in M2 for m1 in M1]
        lo, hi = min(v), max(v)
        pad = (hi - lo) * padf
        return min(lo - pad, 0.0), hi + pad

    _pr, _rd, _ru = _lim("dprec", ["DOWN", "UP"]), _lim("m2_ret", ["DOWN"]), \
        _lim("m2_ret", ["UP"])
    YL = {(0, 0): _pr, (0, 1): _pr, (1, 0): _rd, (1, 1): (_ru[0], 1.5)}

    MM, FS, AXLW = 1 / 25.4, 7.5, 0.7
    W, H = 176, 92

    with plt.rc_context({"font.family": "DejaVu Sans", "axes.linewidth": AXLW,
                         "xtick.major.width": AXLW, "ytick.major.width": AXLW}):
        fig, axes = plt.subplots(2, 2, figsize=(W * MM, H * MM))
        for r, (col, ylabel) in enumerate(SPEC):
            for c, dr in enumerate(("DOWN", "UP")):
                ax = axes[r][c]
                lo, hi = YL[(r, c)]
                for i, m2 in enumerate(M2):
                    per = [_cell(col, dr, m1, m2) for m1 in M1]
                    mean = sum(per) / len(per)
                    ax.bar(i, mean, width=0.58,
                           color=_BW_TAB,
                           edgecolor="#9AA4AE", linewidth=0.5, zorder=2)
                    # Markers within ~2.5% of the axis range would overprint, so
                    # each such cluster is spread horizontally; all stay visible.
                    order = sorted(range(len(per)), key=lambda k: per[k])
                    tol, slot = (hi - lo) * 0.025, {}
                    cluster, last = [], None
                    for k in order:
                        if last is not None and abs(per[k] - last) > tol:
                            for n, kk in enumerate(cluster):
                                slot[kk] = (n, len(cluster))
                            cluster = []
                        cluster.append(k)
                        last = per[k]
                    for n, kk in enumerate(cluster):
                        slot[kk] = (n, len(cluster))
                    for k, (m1, v) in enumerate(zip(M1, per)):
                        n, tot = slot[k]
                        dx = 0.0 if tot == 1 else (n - (tot - 1) / 2) * 0.115
                        ax.scatter([i + dx], [v], s=30, marker=_BW_MK[m1],
                                   color=_GR_COL[m1], edgecolor="white",
                                   linewidths=0.55, zorder=5)

                ax.axhline(0, color="#555555", lw=AXLW, ls=(0, (4, 3)), zorder=3)
                ax.set_xticks(range(len(M2)))
                ax.set_xticklabels([LBL.get(m, m) for m in M2], fontsize=FS - 0.8)
                ax.set_xlim(-0.62, len(M2) - 0.38)
                ax.set_ylim(lo, hi)
                # one label per row: the right column shares the same scale
                if c == 0:
                    ax.set_ylabel(ylabel, fontsize=FS - 0.3, fontweight="bold")
                ax.grid(axis="y", color="#EAEAEA", lw=0.5, zorder=0)
                ax.set_axisbelow(True)
                for sp in ("top", "right"):
                    ax.spines[sp].set_visible(False)
                ax.spines["left"].set_bounds(lo, hi)
                ax.tick_params(labelsize=FS - 0.8, length=2.5)
                if r == 0:
                    ax.set_title(f"M2 {dr}", fontsize=FS + 1.5,
                                 fontweight="bold", pad=9)

        # The two rows' tick labels differ in width ("10" vs "-10"), so each
        # y-label would otherwise sit at its own offset and the pair would read
        # as misaligned even though the panels are not.
        fig.align_ylabels([axes[0][0], axes[1][0]])

        # one bar colour for all five estimators: the figure compares them on
        # equal terms, without splitting them into model families
        handles = [Patch(facecolor=_BW_TAB, edgecolor="#9AA4AE", linewidth=0.5)]
        handles += [Line2D([], [], marker=_BW_MK[m], ls="none", ms=5.5,
                           color=_GR_COL[m], markeredgecolor="white",
                           markeredgewidth=0.55) for m in M1]
        labels = ["Mean over backbones"] + [_GR_NICE.get(m, m) for m in M1]
        fig.legend(handles, labels, loc="lower center", ncol=5, fontsize=FS - 0.5,
                   frameon=True, edgecolor="#CCCCCC", facecolor="#FFFFFF",
                   bbox_to_anchor=(0.537, 0.010), handlelength=1.5,
                   columnspacing=1.6, handletextpad=0.5, borderpad=0.5)
        fig.subplots_adjust(left=0.082, right=0.995, top=0.900, bottom=0.195,
                            wspace=0.115, hspace=0.46)

        if save_path is None:
            save_path = regime_out_dir("granularity") / "reliability_barwhisker.pdf"
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, facecolor="white")
        fig.savefig(save_path.with_suffix(".png"), dpi=dpi, facecolor="white")
        plt.close(fig)

    if verbose:
        for (r, c), (lo, hi) in YL.items():
            print(f"          panel ({r},{c}) ylim = [{lo:.2f}, {hi:.2f}]")
        print(f"  [saved] {save_path}")
    return save_path


# ════════════════════════════════════════════════════════════════════════════
#  Threshold sensitivity: held-out outcome as the validation tau* is shifted
# ════════════════════════════════════════════════════════════════════════════
_TAU_GRANS = ["1d", "12h", "8h", "6h", "4h", "2h", "1h"]
# Coarse to fine: dark blue (1d) and cyan (12h), then towards warm hues, ending
# in orange (2h) and red (1h). Neighbouring granularities alternate in lightness
# so every pair stays distinguishable for full-colour readers (OKLab dE >= 15
# over all pairs); each granularity also gets its own marker, which keeps them
# apart under red-green colour vision deficiency, where hue alone cannot.
_TAU_COL = {"1d": "#08306B", "12h": "#00A0DC", "8h": "#006D5B", "6h": "#8DB600",
            "4h": "#9A6400", "2h": "#FF8C1A", "1h": "#B2182B"}
_TAU_MK = {"1d": "o", "12h": "s", "8h": "^", "6h": "D", "4h": "v", "2h": "P", "1h": "X"}
_TAU_LS = {g: ("--" if g in ("2h", "1h") else "-") for g in _TAU_GRANS}
_TAU_NICE = {"rf": "Random Forest", "tabpfn": "TabPFN", "tabicl": "TabICL"}


def plot_threshold_sensitivity(sweep_csvs,
                               m2_models=("rf", "tabpfn", "tabicl"),
                               save_path=None, band: float = 0.02,
                               ct_grans=("1d", "12h", "8h"), dpi: int = 600,
                               verbose: bool = True) -> Path:
    """Held-out down predictions as tau* is shifted by delta, one row per estimator.

    Columns: cumulative return, coverage, and counter- vs with-trend selective
    precision at the coarse granularities. tau* is the validation-selected value
    and is never re-selected on held-out data; delta = 0 reproduces the published
    backtest. The sweep CSVs come from the cash-engine re-scoring of the stored
    per-signal probabilities (columns m2, gran, delta, cum_ret, coverage,
    ct_prec, wt_prec).
    """
    from matplotlib.lines import Line2D

    d = pd.concat([pd.read_csv(p) for p in sweep_csvs], ignore_index=True)
    M2 = [m for m in m2_models if m in set(d["m2"])]
    FS = _EQ_FS + 2.5
    fig, axes = plt.subplots(len(M2), 3, figsize=(13.5, 3.15 * len(M2) + 0.6),
                             sharex=True)
    axes = np.atleast_2d(axes)

    for r, m2 in enumerate(M2):
        x = d[d["m2"] == m2]
        for c in range(3):
            ax = axes[r, c]
            ax.axvline(0, color="#333333", lw=0.9, ls=":", zorder=1)
            _eq_style(ax)
            ax.tick_params(labelsize=FS - 1)
        for g in _TAU_GRANS:
            s = x[x["gran"] == g].sort_values("delta")
            if s.empty:
                continue
            kw = dict(color=_TAU_COL[g], ls=_TAU_LS[g], lw=1.9, zorder=3,
                      marker=_TAU_MK[g], ms=4.5, mec="white", mew=0.5)
            axes[r, 0].plot(s["delta"], s["cum_ret"], **kw)
            axes[r, 1].plot(s["delta"], s["coverage"], **kw)
        for g, ls in zip(ct_grans, ("-", "--", ":")):
            s = x[x["gran"] == g].sort_values("delta")
            axes[r, 2].plot(s["delta"], s["ct_prec"], color=_EQ_C_CT, ls=ls, lw=1.6,
                            marker=_TAU_MK[g], ms=3.8, mec="white", mew=0.4, zorder=3)
            axes[r, 2].plot(s["delta"], s["wt_prec"], color=_EQ_C_WT, ls=ls, lw=1.6,
                            marker=_TAU_MK[g], ms=3.8, mec="white", mew=0.4, zorder=3)
        axes[r, 0].axhline(0, color="#333333", lw=0.8, zorder=2)
        axes[r, 0].set_ylabel("%s\nCumulative return (%%)" % _TAU_NICE.get(m2, m2),
                              fontsize=FS)
        axes[r, 1].set_ylabel("Coverage (%)", fontsize=FS)
        axes[r, 2].set_ylabel("Selective precision (%)", fontsize=FS)
        if r == 0:
            for c, t in enumerate(("Cumulative return", "Coverage",
                                   "Counter- vs with-trend precision")):
                axes[r, c].set_title(t, fontsize=FS + 1, fontweight="bold", pad=6)
    for c in range(3):
        axes[-1, c].set_xlabel(r"$\delta = \tau - \tau^{*}$", fontsize=FS)
    # shared y within each column, so the three estimators read on one scale
    for c in range(3):
        lo = min(a.get_ylim()[0] for a in axes[:, c])
        hi = max(a.get_ylim()[1] for a in axes[:, c])
        if c == 2:
            # precision never approaches zero; start the axis where the data lives
            lo = max(lo, 40.0)
        for a in axes[:, c]:
            a.set_ylim(lo, hi)

    def _h(g, color, ls):
        return Line2D([], [], color=color, ls=ls, lw=1.9, marker=_TAU_MK[g],
                      ms=5.5, mec="white", mew=0.5)
    _ctls = dict(zip(ct_grans, ("-", "--", ":")))
    _n1 = 3  # row 1: 1d 12h 8h; row 2: 6h 4h 2h 1h
    left_rows = [([_h(g, _TAU_COL[g], _TAU_LS[g]) for g in _TAU_GRANS[:_n1]], list(_TAU_GRANS[:_n1])),
                 ([_h(g, _TAU_COL[g], _TAU_LS[g]) for g in _TAU_GRANS[_n1:]], list(_TAU_GRANS[_n1:]))]
    right_rows = [([_h(g, "#555555", _ctls[g]) for g in ct_grans], list(ct_grans)),
                  ([Line2D([], [], color=_EQ_C_CT, lw=1.8), Line2D([], [], color=_EQ_C_WT, lw=1.8)],
                   ["Counter-trend", "With-trend"])]
    fig.tight_layout(rect=(0, 0.11, 1, 1), h_pad=1.2, w_pad=1.6)

    # each box = two frameless one-row legends stacked, with one frame drawn around both
    from matplotlib.patches import FancyBboxPatch
    def _box(rows, cx):
        legs, y = [], 0.082
        for hh, ll in rows:
            lg = fig.legend(hh, ll, loc="upper center", ncol=len(ll), fontsize=FS - 0.5,
                            bbox_to_anchor=(cx, y), frameon=False, borderaxespad=0,
                            handlelength=2.4, columnspacing=1.4, borderpad=0.15)
            legs.append(lg); y -= 0.034
        fig.canvas.draw()
        inv = fig.transFigure.inverted()
        bb = [inv.transform_bbox(lg.get_window_extent()) for lg in legs]
        x0 = min(b.x0 for b in bb); x1 = max(b.x1 for b in bb)
        y0 = min(b.y0 for b in bb); y1 = max(b.y1 for b in bb)
        px, py = 0.008, 0.005
        fig.patches.append(FancyBboxPatch((x0 - px, y0 - py), x1 - x0 + 2 * px, y1 - y0 + 2 * py,
                                          boxstyle="round,pad=0,rounding_size=0.004",
                                          transform=fig.transFigure, fc="none",
                                          ec="#CCCCCC", lw=0.8, zorder=0))
    pl0, pl1, pr = axes[-1, 0].get_position(), axes[-1, 1].get_position(), axes[-1, 2].get_position()
    _box(left_rows, (pl0.x0 + pl1.x1) / 2)
    _box(right_rows, (pr.x0 + pr.x1) / 2)

    save_path = Path(save_path) if save_path else Path("threshold_sensitivity.png")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=dpi, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    if verbose:
        print(f"  [saved] {save_path}")
    return save_path
