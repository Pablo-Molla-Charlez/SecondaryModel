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
                  "tabpfn": "TabPFN", "tabicl": "TabICL", "ctts": "CNNT"}
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

