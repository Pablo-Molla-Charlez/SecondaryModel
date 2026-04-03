#!/usr/bin/env python3
"""
OCP Diagnostic Analysis
=======================
Runs 6 diagnostic tests on OCP (SAOCP) results to verify correctness
and measure the value added by adaptive thresholding.

Tests:
  1. Fixed-threshold comparison (median OCP τ as fixed)
  2. Random selection baseline (bootstrap)
  3. Shuffled labels sanity check
  4. Rolling conformal coverage
  5. Trade overlap with utility threshold
  6. Probability calibration (reliability diagram)

Usage:
  python ocp_analysis.py --folder <path_to_results_folder> [--mode separate|unified]

  --folder : path to e.g. .../randforest/8h_down_tp  (separate)
             or            .../randforest/unified_down_tp  (unified)
  --mode   : auto-detected from folder name if not provided
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd
import torch


# ---------------------------------------------------------------------------
# SAOCP re-run — canonical implementation lives in Utils.saocp
# ---------------------------------------------------------------------------
from Utils.saocp import _ocp_conformity_score, _run_saocp_online  # noqa: F401


def _run_saocp(val_probs, val_labels, test_probs, test_labels, alpha=0.10,
               test_dates=None, forecast_horizon=1):
    """Thin wrapper around _run_saocp_online returning (s_hats, approved, covered)."""
    s_hats, approved, _, conf_stats = _run_saocp_online(
        val_probs, val_labels, test_probs, test_labels,
        alpha=alpha, test_dates=test_dates, forecast_horizon=forecast_horizon)
    return s_hats, approved, conf_stats["covered"]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_separate(folder: Path):
    """Load data for a per-granularity (separate) results folder."""
    summary_path = folder / "analysis_summary.json"
    with open(summary_path) as f:
        summary = json.load(f)

    # Find trades CSV (pattern: 10_backtest_all_trades.csv)
    csv_candidates = sorted(folder.glob("*_trades.csv"))
    if not csv_candidates:
        raise FileNotFoundError(f"No trades CSV found in {folder}")
    trades_csv = csv_candidates[0]

    df = pd.read_csv(trades_csv)
    df["date"] = pd.to_datetime(df["date"])

    # Extract info from summary
    temp = summary.get("rf_temporal_all_features", {})
    test_sel = temp.get("Test_selective", {})
    val_sel = temp.get("Val_selective", {})
    ocp_info = test_sel.get("ocp", {})

    gran = summary.get("granularity", "?")
    direction = summary.get("direction", "?")

    # Try loading npz diagnostics
    npz_candidates = sorted(folder.glob("*_ocp_diagnostics.npz"))
    npz_data = None
    if npz_candidates:
        npz_data = dict(np.load(npz_candidates[0], allow_pickle=True))

    return {
        "gran": gran,
        "direction": direction,
        "df": df,
        "summary": summary,
        "test_selective": test_sel,
        "val_selective": val_sel,
        "ocp_info": ocp_info,
        "npz": npz_data,
        "alpha": ocp_info.get("alpha", 0.10),
        "utility_threshold": val_sel.get("threshold", 0.5),
    }


def load_unified(folder: Path):
    """Load data for a unified results folder. Returns a dict per granularity."""
    summary_path = folder / "unified_summary.json"
    with open(summary_path) as f:
        summary = json.load(f)

    direction = summary.get("direction", "?")
    per_gran = summary.get("per_gran", {})
    results = {}

    for gran, gdata in per_gran.items():
        gran_dir = folder / gran
        csv_candidates = sorted(gran_dir.glob("*_trades.csv")) if gran_dir.exists() else []
        if not csv_candidates:
            print(f"  [SKIP] {gran}: no trades CSV found")
            continue

        df = pd.read_csv(csv_candidates[0])
        df["date"] = pd.to_datetime(df["date"])

        test_sel = gdata.get("test_selective", {})
        val_sel = gdata.get("val_selective", {})
        ocp_info = test_sel.get("ocp", {})

        npz_candidates = sorted(gran_dir.glob("*ocp_diagnostics.npz"))
        npz_data = None
        if npz_candidates:
            npz_data = dict(np.load(npz_candidates[0], allow_pickle=True))

        results[gran] = {
            "gran": gran,
            "direction": direction,
            "df": df,
            "summary": summary,
            "test_selective": test_sel,
            "val_selective": val_sel,
            "ocp_info": ocp_info,
            "npz": npz_data,
            "alpha": ocp_info.get("alpha", 0.10),
            "utility_threshold": gdata.get("threshold", 0.5),
        }

    return results


# ---------------------------------------------------------------------------
# Test 1: Fixed-threshold comparison
# ---------------------------------------------------------------------------
def test_fixed_threshold(data, fee=0.002):
    """Compare OCP adaptive threshold vs its own median applied as fixed."""
    df = data["df"]
    probs = df["m2_prob"].values
    labels = df["label"].values
    returns = df["return"].values
    ocp_approved = df["m2_approved"].values

    # OCP results (from stored data)
    n_total = len(df)
    n_ocp = int(ocp_approved.sum())
    ocp_risk = float((labels[ocp_approved] == 0).sum() / max(n_ocp, 1))
    ocp_ret = float(returns[ocp_approved].mean()) if n_ocp > 0 else 0.0

    # Median OCP threshold
    ocp_info = data["ocp_info"]
    median_tau = data["test_selective"].get("threshold", 0.5)

    # Fixed threshold = median tau
    fixed_approved = probs > median_tau
    n_fixed = int(fixed_approved.sum())
    fixed_risk = float((labels[fixed_approved] == 0).sum() / max(n_fixed, 1)) if n_fixed > 0 else 0.0
    fixed_ret = float(returns[fixed_approved].mean()) if n_fixed > 0 else 0.0

    # Also try utility threshold
    util_thr = data["utility_threshold"]
    util_approved = probs >= util_thr
    n_util = int(util_approved.sum())
    util_risk = float((labels[util_approved] == 0).sum() / max(n_util, 1)) if n_util > 0 else 0.0
    util_ret = float(returns[util_approved].mean()) if n_util > 0 else 0.0

    return {
        "OCP_adaptive": {"n": n_ocp, "cov": n_ocp / n_total, "risk": ocp_risk, "mean_ret": ocp_ret},
        "Fixed_median_tau": {"n": n_fixed, "cov": n_fixed / n_total, "risk": fixed_risk,
                             "mean_ret": fixed_ret, "threshold": median_tau},
        "Utility_threshold": {"n": n_util, "cov": n_util / n_total, "risk": util_risk,
                              "mean_ret": util_ret, "threshold": util_thr},
    }


# ---------------------------------------------------------------------------
# Test 2: Random selection baseline
# ---------------------------------------------------------------------------
def test_random_baseline(data, n_bootstrap=1000, seed=42):
    """Bootstrap: randomly select same fraction as OCP, compare returns."""
    df = data["df"]
    returns = df["return"].values
    labels = df["label"].values
    ocp_approved = df["m2_approved"].values
    n_total = len(df)
    n_ocp = int(ocp_approved.sum())

    if n_ocp == 0 or n_total == 0:
        return {"ocp_ret": 0, "random_mean": 0, "random_std": 0, "percentile": 0, "p_value": 1.0}

    ocp_ret = float(returns[ocp_approved].mean())
    ocp_risk = float((labels[ocp_approved] == 0).sum() / n_ocp)

    rng = np.random.RandomState(seed)
    random_rets = np.zeros(n_bootstrap)
    random_risks = np.zeros(n_bootstrap)
    for b in range(n_bootstrap):
        idx = rng.choice(n_total, size=n_ocp, replace=False)
        random_rets[b] = returns[idx].mean()
        random_risks[b] = (labels[idx] == 0).sum() / n_ocp

    percentile = float((random_rets < ocp_ret).mean() * 100)
    p_value = float((random_rets >= ocp_ret).mean())

    return {
        "n_ocp": n_ocp,
        "ocp_ret": ocp_ret,
        "ocp_risk": ocp_risk,
        "random_ret_mean": float(random_rets.mean()),
        "random_ret_std": float(random_rets.std()),
        "random_risk_mean": float(random_risks.mean()),
        "percentile": percentile,
        "p_value": p_value,
        "random_rets": random_rets,  # for plotting
    }


# ---------------------------------------------------------------------------
# Test 3: Shuffled labels sanity check
# ---------------------------------------------------------------------------
def test_shuffled_labels(data, n_shuffles=5, seed=42):
    """Re-run SAOCP with permuted test labels. Compare vs real."""
    df = data["df"]
    probs = df["m2_prob"].values
    labels = df["label"].values
    returns = df["return"].values
    alpha = data["alpha"]

    # Get val data
    npz = data["npz"]
    if npz is None:
        print("    [SKIP] No npz diagnostics — cannot re-run SAOCP (need val probs)")
        return None

    val_probs = npz["val_probs"]
    val_labels = npz["val_labels"]

    # Real run
    _, real_approved, real_covered = _run_saocp(val_probs, val_labels, probs, labels, alpha)
    n_real = int(real_approved.sum())
    real_ret = float(returns[real_approved].mean()) if n_real > 0 else 0.0
    real_risk = float((labels[real_approved] == 0).sum() / max(n_real, 1))
    real_ccov = float(real_covered.mean())

    rng = np.random.RandomState(seed)
    shuffled_results = []
    for s in range(n_shuffles):
        shuffled_labels = labels.copy()
        rng.shuffle(shuffled_labels)
        _, shuf_approved, shuf_covered = _run_saocp(
            val_probs, val_labels, probs, shuffled_labels, alpha)
        n_shuf = int(shuf_approved.sum())
        shuf_ret = float(returns[shuf_approved].mean()) if n_shuf > 0 else 0.0
        shuf_risk = float((labels[shuf_approved] == 0).sum() / max(n_shuf, 1))
        shuf_ccov = float(shuf_covered.mean())
        shuffled_results.append({
            "n": n_shuf, "ret": shuf_ret, "risk": shuf_risk, "conf_cov": shuf_ccov
        })

    return {
        "real": {"n": n_real, "ret": real_ret, "risk": real_risk, "conf_cov": real_ccov},
        "shuffled": shuffled_results,
    }


# ---------------------------------------------------------------------------
# Test 4: Rolling conformal coverage
# ---------------------------------------------------------------------------
def test_rolling_coverage(data, window=200):
    """Compute conformal coverage in sliding windows."""
    npz = data["npz"]
    if npz is None:
        # Re-derive from trades CSV + re-running SAOCP
        df = data["df"]
        probs = df["m2_prob"].values
        labels = df["label"].values
        alpha = data["alpha"]
        # Without val warm-up, just run on test alone
        print("    [WARN] No npz — running SAOCP without val warm-up (results approximate)")
        s_hats, _, covered = _run_saocp(np.array([]), np.array([]),
                                        probs, labels, alpha)
        return {"covered": covered, "s_hats": s_hats, "window": window, "no_warmup": True}

    val_probs = npz["val_probs"]
    val_labels = npz["val_labels"]
    df = data["df"]
    probs = df["m2_prob"].values
    labels = df["label"].values
    alpha = data["alpha"]

    s_hats, _, covered = _run_saocp(val_probs, val_labels, probs, labels, alpha)

    return {"covered": covered, "s_hats": s_hats, "window": window, "no_warmup": False}


# ---------------------------------------------------------------------------
# Test 5: Trade overlap with utility threshold
# ---------------------------------------------------------------------------
def test_trade_overlap(data):
    """Compute overlap between OCP-selected and utility-selected trades."""
    df = data["df"]
    probs = df["m2_prob"].values
    ocp_mask = df["m2_approved"].values
    util_thr = data["utility_threshold"]
    util_mask = probs >= util_thr

    n_ocp = int(ocp_mask.sum())
    n_util = int(util_mask.sum())
    n_both = int((ocp_mask & util_mask).sum())
    n_union = int((ocp_mask | util_mask).sum())
    jaccard = n_both / max(n_union, 1)

    # OCP-only and utility-only trades
    ocp_only = ocp_mask & ~util_mask
    util_only = util_mask & ~ocp_mask

    labels = df["label"].values
    returns = df["return"].values

    def _stats(mask):
        n = int(mask.sum())
        if n == 0:
            return {"n": 0, "risk": 0, "mean_ret": 0, "win_rate": 0}
        return {
            "n": n,
            "risk": float((labels[mask] == 0).sum() / n),
            "mean_ret": float(returns[mask].mean()),
            "win_rate": float((labels[mask] == 1).sum() / n),
        }

    return {
        "n_ocp": n_ocp,
        "n_util": n_util,
        "n_intersection": n_both,
        "n_union": n_union,
        "jaccard": jaccard,
        "overlap_pct_of_ocp": n_both / max(n_ocp, 1),
        "overlap_pct_of_util": n_both / max(n_util, 1),
        "ocp_only": _stats(ocp_only),
        "util_only": _stats(util_only),
        "intersection": _stats(ocp_mask & util_mask),
        "utility_threshold": util_thr,
    }


# ---------------------------------------------------------------------------
# Test 6: Probability calibration (reliability diagram)
# ---------------------------------------------------------------------------
def test_calibration(data, n_bins=10):
    """Reliability diagram: predicted probability vs observed frequency."""
    df = data["df"]
    probs = df["m2_prob"].values
    labels = df["label"].values

    bin_edges = np.linspace(0, 1, n_bins + 1)
    bins = []
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        mask = (probs >= lo) & (probs < hi) if i < n_bins - 1 else (probs >= lo) & (probs <= hi)
        n = int(mask.sum())
        if n == 0:
            continue
        mean_pred = float(probs[mask].mean())
        mean_actual = float(labels[mask].mean())
        bins.append({
            "lo": lo, "hi": hi, "n": n,
            "mean_pred": mean_pred, "mean_actual": mean_actual,
        })

    # ECE (Expected Calibration Error)
    total = len(probs)
    ece = sum(b["n"] / total * abs(b["mean_pred"] - b["mean_actual"]) for b in bins)

    return {"bins": bins, "ece": ece}


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_all(results, save_dir, gran_label):
    """Generate all diagnostic plots."""
    save_dir.mkdir(parents=True, exist_ok=True)

    # --- Test 1: Fixed threshold comparison (bar chart) ---
    t1 = results.get("test1_fixed_threshold")
    if t1:
        fig, axes = plt.subplots(1, 3, figsize=(12, 4), facecolor="white")
        methods = list(t1.keys())
        colors = ["#8B008B", "#E67E22", "#3498DB"]
        for i, metric in enumerate(["cov", "risk", "mean_ret"]):
            vals = [t1[m][metric] for m in methods]
            labels_m = [m.replace("_", "\n") for m in methods]
            axes[i].bar(labels_m, vals, color=colors, alpha=0.8, edgecolor="black", linewidth=0.5)
            axes[i].set_title({"cov": "Coverage", "risk": "Risk", "mean_ret": "Mean Return"}[metric],
                             fontsize=11, fontweight="bold")
            axes[i].grid(axis="y", alpha=0.3)
            for j, v in enumerate(vals):
                axes[i].text(j, v, f"{v:.4f}", ha="center", va="bottom", fontsize=8)
        fig.suptitle(f"Test 1: Fixed Threshold Comparison — {gran_label}", fontsize=13, fontweight="bold")
        fig.tight_layout()
        fig.savefig(save_dir / "diag_1_fixed_threshold.png", dpi=200)
        plt.close(fig)

    # --- Test 2: Random baseline (histogram) ---
    t2 = results.get("test2_random_baseline")
    if t2 and "random_rets" in t2:
        fig, ax = plt.subplots(figsize=(8, 4), facecolor="white")
        ax.hist(t2["random_rets"], bins=50, color="#BDC3C7", edgecolor="black",
                linewidth=0.3, alpha=0.8, label="Random selection")
        ax.axvline(t2["ocp_ret"], color="#8B008B", linewidth=2.5,
                   label=f"OCP return = {t2['ocp_ret']:.6f}")
        ax.axvline(t2["random_ret_mean"], color="#E67E22", linewidth=1.5, linestyle="--",
                   label=f"Random mean = {t2['random_ret_mean']:.6f}")
        ax.set_xlabel("Mean Return", fontsize=10)
        ax.set_ylabel("Count", fontsize=10)
        ax.set_title(f"Test 2: Random Selection Baseline — {gran_label}\n"
                     f"OCP at {t2['percentile']:.1f}th percentile (p={t2['p_value']:.4f})",
                     fontsize=11, fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(save_dir / "diag_2_random_baseline.png", dpi=200)
        plt.close(fig)

    # --- Test 3: Shuffled labels ---
    t3 = results.get("test3_shuffled_labels")
    if t3:
        fig, axes = plt.subplots(1, 3, figsize=(12, 4), facecolor="white")
        real = t3["real"]
        shuf_list = t3["shuffled"]
        metrics = [("n", "Trades Selected"), ("ret", "Mean Return"), ("risk", "Risk")]
        for i, (key, label) in enumerate(metrics):
            real_val = real[key]
            shuf_vals = [s[key] for s in shuf_list]
            axes[i].bar(["Real"] + [f"Shuf_{j+1}" for j in range(len(shuf_vals))],
                       [real_val] + shuf_vals,
                       color=["#8B008B"] + ["#BDC3C7"] * len(shuf_vals),
                       alpha=0.8, edgecolor="black", linewidth=0.5)
            axes[i].set_title(label, fontsize=11, fontweight="bold")
            axes[i].grid(axis="y", alpha=0.3)
        fig.suptitle(f"Test 3: Shuffled Labels Sanity Check — {gran_label}", fontsize=13, fontweight="bold")
        fig.tight_layout()
        fig.savefig(save_dir / "diag_3_shuffled_labels.png", dpi=200)
        plt.close(fig)

    # --- Test 4: Rolling conformal coverage ---
    t4 = results.get("test4_rolling_coverage")
    if t4 is not None and "covered" in t4:
        covered = t4["covered"]
        window = t4["window"]
        alpha = results.get("alpha", 0.10)

        # Rolling mean
        if len(covered) >= window:
            rolling = np.convolve(covered, np.ones(window) / window, mode="valid")
        else:
            rolling = np.array([covered.mean()])

        fig, axes = plt.subplots(2, 1, figsize=(12, 7), facecolor="white",
                                 gridspec_kw={"height_ratios": [2, 1]})
        # Top: rolling coverage
        ax = axes[0]
        ax.plot(rolling, color="#8B008B", linewidth=1.0, alpha=0.9)
        ax.axhline(1 - alpha, color="#27AE60", linewidth=1.5, linestyle="--",
                   label=f"Target = {1-alpha:.0%}")
        ax.axhline(covered.mean(), color="#E67E22", linewidth=1.2, linestyle=":",
                   label=f"Overall = {covered.mean():.1%}")
        ax.fill_between(range(len(rolling)), 1 - alpha - 0.05, 1 - alpha + 0.05,
                        color="#27AE60", alpha=0.1)
        ax.set_ylabel("Conformal Coverage", fontsize=10)
        ax.set_title(f"Test 4: Rolling Conformal Coverage (w={window}) — {gran_label}",
                     fontsize=11, fontweight="bold")
        ax.legend(fontsize=9)
        ax.set_ylim(0.5, 1.05)
        ax.grid(alpha=0.3)

        # Bottom: threshold evolution
        if "s_hats" in t4:
            ax2 = axes[1]
            eff_tau = np.maximum(t4["s_hats"], 1.0 - t4["s_hats"])
            ax2.plot(eff_tau, color="#8B008B", linewidth=0.6, alpha=0.8)
            ax2.axhline(0.5, color="#BDC3C7", linewidth=0.8, linestyle=":")
            ax2.set_ylabel("τ_t", fontsize=10)
            ax2.set_xlabel("Test sample index", fontsize=10)
            ax2.grid(alpha=0.3)

        fig.tight_layout()
        fig.savefig(save_dir / "diag_4_rolling_coverage.png", dpi=200)
        plt.close(fig)

    # --- Test 5: Trade overlap ---
    t5 = results.get("test5_trade_overlap")
    if t5:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4), facecolor="white")

        # Venn-style bar
        ax = axes[0]
        ocp_only_n = t5["ocp_only"]["n"]
        util_only_n = t5["util_only"]["n"]
        inter_n = t5["n_intersection"]
        ax.barh(["OCP only", "Intersection", "Utility only"],
                [ocp_only_n, inter_n, util_only_n],
                color=["#8B008B", "#27AE60", "#E67E22"], alpha=0.8,
                edgecolor="black", linewidth=0.5)
        ax.set_xlabel("Number of trades", fontsize=10)
        ax.set_title(f"Trade Overlap (Jaccard={t5['jaccard']:.3f})", fontsize=11, fontweight="bold")
        for i, v in enumerate([ocp_only_n, inter_n, util_only_n]):
            ax.text(v + 1, i, str(v), va="center", fontsize=9)
        ax.grid(axis="x", alpha=0.3)

        # Win rate comparison
        ax2 = axes[1]
        groups = ["OCP only", "Intersection", "Utility only"]
        wr = [t5["ocp_only"]["win_rate"], t5["intersection"]["win_rate"],
              t5["util_only"]["win_rate"]]
        ret = [t5["ocp_only"]["mean_ret"], t5["intersection"]["mean_ret"],
               t5["util_only"]["mean_ret"]]
        x = np.arange(len(groups))
        w = 0.35
        bars1 = ax2.bar(x - w/2, wr, w, label="Win Rate", color="#3498DB", alpha=0.8,
                        edgecolor="black", linewidth=0.5)
        ax2_r = ax2.twinx()
        bars2 = ax2_r.bar(x + w/2, [r * 100 for r in ret], w, label="Mean Ret %",
                          color="#E74C3C", alpha=0.8, edgecolor="black", linewidth=0.5)
        ax2.set_xticks(x)
        ax2.set_xticklabels(groups, fontsize=8)
        ax2.set_ylabel("Win Rate", fontsize=9, color="#3498DB")
        ax2_r.set_ylabel("Mean Return (%)", fontsize=9, color="#E74C3C")
        ax2.set_title("Quality by Group", fontsize=11, fontweight="bold")
        ax2.grid(axis="y", alpha=0.3)

        fig.suptitle(f"Test 5: Trade Overlap — {gran_label}", fontsize=13, fontweight="bold")
        fig.tight_layout()
        fig.savefig(save_dir / "diag_5_trade_overlap.png", dpi=200)
        plt.close(fig)

    # --- Test 6: Calibration ---
    t6 = results.get("test6_calibration")
    if t6:
        bins = t6["bins"]
        fig, ax = plt.subplots(figsize=(6, 6), facecolor="white")
        ax.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.5, label="Perfect calibration")
        preds = [b["mean_pred"] for b in bins]
        actuals = [b["mean_actual"] for b in bins]
        sizes = [b["n"] for b in bins]
        max_size = max(sizes) if sizes else 1
        ax.scatter(preds, actuals, s=[s / max_size * 300 + 20 for s in sizes],
                   c="#8B008B", alpha=0.7, edgecolors="black", linewidth=0.5)
        for b in bins:
            ax.annotate(f"n={b['n']}", (b["mean_pred"], b["mean_actual"]),
                       fontsize=6, ha="center", va="bottom",
                       xytext=(0, 5), textcoords="offset points")
        ax.set_xlabel("Mean Predicted Probability", fontsize=10)
        ax.set_ylabel("Observed Frequency (Win Rate)", fontsize=10)
        ax.set_title(f"Test 6: Calibration — {gran_label}\nECE = {t6['ece']:.4f}",
                     fontsize=11, fontweight="bold")
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=9)
        fig.tight_layout()
        fig.savefig(save_dir / "diag_6_calibration.png", dpi=200)
        plt.close(fig)


# ---------------------------------------------------------------------------
# Run all tests for one granularity
# ---------------------------------------------------------------------------
def run_all_tests(data, save_dir, gran_label):
    """Run all 6 diagnostic tests and produce plots + JSON summary."""
    save_dir.mkdir(parents=True, exist_ok=True)
    results = {"alpha": data["alpha"]}

    print(f"\n  === {gran_label} ===")

    # Test 1
    print("  [1/6] Fixed threshold comparison...")
    t1 = test_fixed_threshold(data)
    results["test1_fixed_threshold"] = t1
    for method, vals in t1.items():
        print(f"    {method}: n={vals['n']}, cov={vals['cov']:.4f}, "
              f"risk={vals['risk']:.4f}, ret={vals['mean_ret']:.6f}")

    # Test 2
    print("  [2/6] Random selection baseline...")
    t2 = test_random_baseline(data)
    results["test2_random_baseline"] = {k: v for k, v in t2.items() if k != "random_rets"}
    print(f"    OCP return: {t2['ocp_ret']:.6f} | Random mean: {t2['random_ret_mean']:.6f} "
          f"| Percentile: {t2['percentile']:.1f}% | p-value: {t2['p_value']:.4f}")

    # Test 3
    print("  [3/6] Shuffled labels sanity check...")
    t3 = test_shuffled_labels(data)
    results["test3_shuffled_labels"] = t3
    if t3:
        print(f"    Real: n={t3['real']['n']}, ret={t3['real']['ret']:.6f}, "
              f"risk={t3['real']['risk']:.4f}, ccov={t3['real']['conf_cov']:.4f}")
        for i, s in enumerate(t3["shuffled"]):
            print(f"    Shuf_{i+1}: n={s['n']}, ret={s['ret']:.6f}, "
                  f"risk={s['risk']:.4f}, ccov={s['conf_cov']:.4f}")

    # Test 4
    print("  [4/6] Rolling conformal coverage...")
    t4 = test_rolling_coverage(data)
    results["test4_rolling_coverage"] = (
        {"window": t4["window"], "no_warmup": t4.get("no_warmup", False),
         "overall_ccov": float(t4["covered"].mean()),
         "min_rolling": float(np.convolve(t4["covered"],
                              np.ones(t4["window"]) / t4["window"],
                              mode="valid").min()) if len(t4["covered"]) >= t4["window"] else 0,
         "max_rolling": float(np.convolve(t4["covered"],
                              np.ones(t4["window"]) / t4["window"],
                              mode="valid").max()) if len(t4["covered"]) >= t4["window"] else 0,
        } if t4 else None
    )
    if t4:
        r = results["test4_rolling_coverage"]
        print(f"    Overall ccov: {r['overall_ccov']:.4f} | "
              f"Rolling range: [{r['min_rolling']:.4f}, {r['max_rolling']:.4f}]")

    # Test 5
    print("  [5/6] Trade overlap with utility threshold...")
    t5 = test_trade_overlap(data)
    results["test5_trade_overlap"] = t5
    print(f"    Jaccard: {t5['jaccard']:.3f} | "
          f"OCP∩Util: {t5['n_intersection']} | "
          f"OCP-only: {t5['ocp_only']['n']} (WR={t5['ocp_only']['win_rate']:.3f}) | "
          f"Util-only: {t5['util_only']['n']} (WR={t5['util_only']['win_rate']:.3f})")

    # Test 6
    print("  [6/6] Probability calibration...")
    t6 = test_calibration(data)
    results["test6_calibration"] = t6
    print(f"    ECE: {t6['ece']:.4f}")

    # Plot
    plot_all({**results, "alpha": data["alpha"]},
             save_dir, gran_label)
    # Keep random_rets for plotting
    results["test2_random_baseline"]["random_rets"] = t2.get("random_rets")
    plot_all(results, save_dir, gran_label)

    # Save JSON (remove non-serializable arrays)
    json_results = {}
    for k, v in results.items():
        if k == "test2_random_baseline" and isinstance(v, dict):
            json_results[k] = {kk: vv for kk, vv in v.items() if kk != "random_rets"}
        elif k == "test4_rolling_coverage":
            json_results[k] = v
        elif k == "test6_calibration":
            json_results[k] = v
        else:
            json_results[k] = v

    with open(save_dir / "ocp_diagnostics.json", "w") as f:
        json.dump(json_results, f, indent=2, default=str)

    return results


# ---------------------------------------------------------------------------
# Analysis 7: Candlestick + tau_t + volume per asset
# ---------------------------------------------------------------------------
def _load_ohlcv_for_gran(cache_path, gran, direction="up"):
    """Load OHLCV data from Kronos cache for a given granularity.
    cache_path can be a .pt file or a directory containing .pt files."""
    cache_p = Path(cache_path).resolve()
    # If directory, pick the right .pt based on direction
    if cache_p.is_dir():
        candidates = sorted(cache_p.glob(f"*_{direction}_*.pt"))
        if not candidates:
            candidates = sorted(cache_p.glob("*.pt"))
        if not candidates:
            raise FileNotFoundError(f"No .pt files found in {cache_p}")
        cache_p = candidates[0]
    # Ensure the project root is on sys.path so torch can unpickle Utils classes
    # Cache lives at .../src/Output/Kronos/cache/xxx.pt → go up 4 levels
    project_root = cache_p.parent.parent.parent.parent  # .../src/
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    dataset = torch.load(str(cache_p), map_location="cpu", weights_only=False)
    sub = dataset.sub[gran]
    ohlcv = sub["ohlcv"]  # (n_samples, 5, 75)
    dates = sub["dates"]
    asset_ids = sub["asset_ids"]
    asset_map = dataset.asset_map  # {id: name}
    return ohlcv, dates, asset_ids, asset_map


def _build_ohlcv_df(ohlcv, dates, asset_ids, asset_map, asset_name, test_start):
    """Build a DataFrame with O,H,L,C,V for one asset in the test period."""
    inv_map = {v: k for k, v in asset_map.items()}
    aid = inv_map[asset_name]
    mask = asset_ids == aid
    idx = mask.nonzero(as_tuple=True)[0]
    rows = []
    for i in idx:
        d = dates[i]
        if d < test_start:
            continue
        bar = ohlcv[i, :, -1]  # last lookback bar = current bar
        rows.append({
            "date": d,
            "open": float(bar[0]),
            "high": float(bar[1]),
            "low": float(bar[2]),
            "close": float(bar[3]),
            "volume": float(bar[4]),
        })
    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)


def _resolve_util_threshold(save_dir, gran):
    """Find the validation utility threshold from analysis/unified summary."""
    # Per-gran separate
    summary_path = save_dir.parent / "analysis_summary.json"
    if summary_path.exists():
        with open(summary_path) as f:
            s = json.load(f)
        return s.get("rf_temporal_all_features", {}).get("Val_selective", {}).get("threshold")
    # Unified
    summary_path = save_dir.parent.parent / "unified_summary.json"
    if summary_path.exists():
        with open(summary_path) as f:
            s = json.load(f)
        return s.get("per_gran", {}).get(gran, {}).get("threshold")
    return None


def _select_window(df_candle, asset_trades, max_bars=300, min_bars=120):
    """Pick the most interesting subperiod if there are too many bars.
    Returns (start_idx, end_idx) into df_candle, or (0, len) if small enough."""
    n = len(df_candle)
    if n <= max_bars:
        return 0, n
    # Find densest trading-activity window: rolling count of approved trades
    window_size = min(max_bars, n)
    candle_dates = df_candle["date"].values
    approved_dates = set(
        asset_trades.loc[asset_trades["m2_approved"], "date"].values
    )
    # Binary array: 1 if any approved trade at this candle date
    activity = np.array([1 if d in approved_dates else 0 for d in candle_dates])
    # Rolling sum
    if len(activity) > window_size:
        rolling_sum = np.convolve(activity, np.ones(window_size), mode="valid")
        best_start = int(np.argmax(rolling_sum))
    else:
        best_start = 0
    return best_start, min(best_start + window_size, n)


def analysis_candlestick(cache_path, gran, direction, trades_df, npz_path,
                         assets, save_dir, test_start="2025-10-01"):
    """Analysis 7: Candlestick+volume + tau_t + regime per asset."""
    save_dir.mkdir(parents=True, exist_ok=True)
    test_start = pd.Timestamp(test_start)

    # Style (matching risk-coverage plots)
    C_GRID   = "#D5D8DC"
    C_TP     = "#1E8449"
    C_FP     = "#8B0000"
    C_TP_BG  = "#E8F8F5"
    C_FP_BG  = "#FADBD8"
    C_UP     = "#1B5E20"
    C_DN     = "#B71C1C"
    C_TAU    = "#8B008B"
    C_PROB   = "#95A5A6"
    C_TREND  = "#1E8449"
    C_RANGE  = "#E67E22"
    C_UTIL   = "#E67E22"

    ohlcv, dates, asset_ids, asset_map = _load_ohlcv_for_gran(cache_path, gran, direction)

    npz = np.load(npz_path, allow_pickle=True)
    test_s_hats = npz["test_s_hats"]
    test_tau = np.maximum(test_s_hats, 1.0 - test_s_hats)

    trades_df = trades_df.copy()
    trades_df["date"] = pd.to_datetime(trades_df["date"])
    trades_df = trades_df.sort_values("date").reset_index(drop=True)
    trades_df["tau_t"] = test_tau

    # Resolve utility threshold
    util_thr = _resolve_util_threshold(save_dir, gran)

    from matplotlib.lines import Line2D

    for asset_name in assets:
        print(f"    Plotting {asset_name} ({gran} {direction})...")
        df_candle_full = _build_ohlcv_df(ohlcv, dates, asset_ids, asset_map,
                                          asset_name, test_start)
        if df_candle_full.empty:
            print(f"      [SKIP] No OHLCV data for {asset_name}")
            continue

        asset_trades = trades_df[trades_df["asset"] == asset_name].copy()

        # Smart window selection for dense granularities
        start_idx, end_idx = _select_window(df_candle_full, asset_trades)
        df_candle = df_candle_full.iloc[start_idx:end_idx].copy().reset_index(drop=True)
        if start_idx > 0 or end_idx < len(df_candle_full):
            d0 = df_candle["date"].iloc[0].strftime("%Y-%m-%d")
            d1 = df_candle["date"].iloc[-1].strftime("%Y-%m-%d")
            print(f"      Windowed: {len(df_candle)} bars ({d0} → {d1})")

        # Regime
        df_candle["ret_sign"] = (df_candle["close"] - df_candle["open"]).apply(
            lambda x: 1 if x >= 0 else 0)
        df_candle["regime"] = df_candle["ret_sign"].rolling(20, min_periods=10).mean()

        # Filter trades to window dates
        window_dates = set(df_candle["date"].values)
        asset_trades_win = asset_trades[asset_trades["date"].isin(window_dates)].copy()

        # Also filter tau to window dates for the tau line
        trades_in_window = trades_df[trades_df["date"].isin(window_dates)]

        dates_num = mdates.date2num(df_candle["date"])
        bar_w = float(np.median(np.diff(dates_num))) * 0.6 if len(dates_num) > 1 else 0.2

        # --- Figure: 3 subplots ---
        fig, axes = plt.subplots(3, 1, figsize=(16, 10), facecolor="white",
                                  gridspec_kw={"height_ratios": [5, 2.5, 1]},
                                  sharex=True)
        for ax in axes:
            ax.set_facecolor("#FAFAFA")
            for spine in ax.spines.values():
                spine.set_color("black")
                spine.set_linewidth(1.2)

        # === Panel 1: Candlesticks + Volume (right Y) + trade bands ===
        ax_price = axes[0]
        y_lo_price = df_candle["low"].min()
        y_hi_price = df_candle["high"].max()
        price_range = y_hi_price - y_lo_price
        band_w = bar_w * 1.5

        # Trade approval background bands
        for _, trade in asset_trades_win.iterrows():
            if not trade["m2_approved"]:
                continue
            d_num = mdates.date2num(trade["date"])
            bg = C_TP_BG if trade["label"] == 1 else C_FP_BG
            rect = Rectangle((d_num - band_w / 2, y_lo_price - price_range * 0.02),
                              band_w, price_range * 1.04,
                              facecolor=bg, alpha=0.7, edgecolor="none", zorder=1)
            ax_price.add_patch(rect)

        # Candlesticks
        for _, row in df_candle.iterrows():
            d = mdates.date2num(row["date"])
            o, h, l, c = row["open"], row["high"], row["low"], row["close"]
            color = C_UP if c >= o else C_DN
            ax_price.plot([d, d], [l, h], color=color, linewidth=0.7, zorder=4)
            body_h = max(abs(c - o), (h - l) * 0.01)
            rect = Rectangle((d - bar_w / 2, min(o, c)), bar_w, body_h,
                              facecolor=color, edgecolor=color, linewidth=0.3, zorder=5)
            ax_price.add_patch(rect)

        ax_price.set_xlim(dates_num[0] - bar_w * 2, dates_num[-1] + bar_w * 2)
        margin = price_range * 0.05
        ax_price.set_ylim(y_lo_price - margin, y_hi_price + margin)
        ax_price.set_ylabel("Price", fontsize=11, fontweight="bold", labelpad=8)
        ax_price.tick_params(axis="y", labelsize=9, width=1.2)
        plt.setp(ax_price.get_yticklabels(), fontweight="bold")
        ax_price.grid(True, which="major", color=C_GRID, linewidth=0.6, alpha=0.7)
        ax_price.set_axisbelow(True)

        # Volume on right Y-axis (bottom 20%, no overlap with candles)
        ax_vol = ax_price.twinx()
        vol_colors = [C_UP if c >= o else C_DN
                      for c, o in zip(df_candle["close"], df_candle["open"])]
        ax_vol.bar(dates_num, df_candle["volume"], width=bar_w, color=vol_colors,
                   alpha=0.55, zorder=0)
        ax_vol.set_ylabel("Volume", fontsize=10, fontweight="bold", color="#566573",
                          labelpad=8)
        ax_vol.tick_params(axis="y", labelsize=8, colors="#566573")
        ax_vol.set_ylim(0, df_candle["volume"].max() * 5.0)
        ax_vol.spines["right"].set_color("#566573")
        ax_vol.spines["right"].set_linewidth(0.8)

        ax_price.set_title(
            f"{asset_name} — {gran} {direction.upper()} — OCP Trade Selection",
            fontsize=14, fontweight="bold", pad=12)
        legend_handles = [
            Rectangle((0, 0), 1, 1, facecolor=C_TP_BG, alpha=0.7,
                       edgecolor=C_TP, linewidth=0.8, label="Approved (TP)"),
            Rectangle((0, 0), 1, 1, facecolor=C_FP_BG, alpha=0.7,
                       edgecolor=C_FP, linewidth=0.8, label="Approved (FP)"),
        ]
        ax_price.legend(handles=legend_handles, loc="upper left", fontsize=9,
                         framealpha=0.9, edgecolor=C_GRID)

        # === Panel 2: tau_t + utility threshold + asset RF probabilities ===
        ax_tau = axes[1]
        date_tau = trades_in_window.groupby("date")["tau_t"].mean()
        ax_tau.plot(date_tau.index, date_tau.values, color=C_TAU,
                    linewidth=1.5, alpha=0.9, label="τ_t (OCP)", zorder=3)

        # Utility threshold as dashed orange reference
        if util_thr is not None:
            ax_tau.axhline(util_thr, color=C_UTIL, linewidth=1.5, linestyle="--",
                           alpha=0.8, label=f"τ Utility = {util_thr:.3f}", zorder=2)

        if not asset_trades_win.empty:
            # All predictions as darker grey dots
            ax_tau.scatter(asset_trades_win["date"], asset_trades_win["m2_prob"],
                          c="#566573", s=10, alpha=0.5, zorder=2,
                          label="p (not selected)")
            # Approved trades: green for TP, softer red for FP
            approved = asset_trades_win[asset_trades_win["m2_approved"]]
            if not approved.empty:
                tp_mask = approved["label"] == 1
                fp_mask = approved["label"] == 0
                if tp_mask.any():
                    ax_tau.scatter(approved.loc[tp_mask, "date"],
                                  approved.loc[tp_mask, "m2_prob"],
                                  c=C_TP, s=30, alpha=0.85, zorder=4,
                                  edgecolors="black", linewidths=0.4,
                                  label="p (approved TP)")
                if fp_mask.any():
                    ax_tau.scatter(approved.loc[fp_mask, "date"],
                                  approved.loc[fp_mask, "m2_prob"],
                                  c="#E74C3C", s=30, alpha=0.7, zorder=4,
                                  edgecolors="black", linewidths=0.4,
                                  label="p (approved FP)")

        ax_tau.axhline(0.5, color=C_GRID, linewidth=0.8, linestyle=":")
        ax_tau.set_ylabel("Probability / τ_t", fontsize=11, fontweight="bold", labelpad=8)
        ax_tau.set_ylim(0.35, 1.0)
        ax_tau.tick_params(axis="y", labelsize=9, width=1.2)
        plt.setp(ax_tau.get_yticklabels(), fontweight="bold")
        ax_tau.legend(fontsize=8, loc="upper right", framealpha=0.9, edgecolor=C_GRID,
                      ncol=2)
        ax_tau.grid(True, which="major", color=C_GRID, linewidth=0.6, alpha=0.7)
        ax_tau.set_axisbelow(True)

        # === Panel 3: Regime indicator ===
        ax_regime = axes[2]
        regime_vals = df_candle["regime"].values
        ax_regime.fill_between(dates_num, 0.5, regime_vals,
                               where=regime_vals >= 0.5, color=C_TREND,
                               alpha=0.25, interpolate=True, label="Trending")
        ax_regime.fill_between(dates_num, 0.5, regime_vals,
                               where=regime_vals < 0.5, color=C_RANGE,
                               alpha=0.25, interpolate=True, label="Ranging")
        ax_regime.plot(dates_num, regime_vals, color="#34495E",
                       linewidth=0.9, alpha=0.8)
        ax_regime.axhline(0.5, color=C_GRID, linewidth=0.8, linestyle=":")
        ax_regime.set_ylabel("Regime", fontsize=10, fontweight="bold", labelpad=8)
        ax_regime.set_xlabel("Date", fontsize=11, fontweight="bold", labelpad=8)
        ax_regime.set_ylim(0.0, 1.0)
        ax_regime.tick_params(axis="both", labelsize=9, width=1.2)
        plt.setp(ax_regime.get_xticklabels(), fontweight="bold")
        ax_regime.legend(fontsize=9, loc="upper right", framealpha=0.9, edgecolor=C_GRID)
        ax_regime.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
        ax_regime.xaxis.set_major_locator(mdates.AutoDateLocator())
        ax_regime.grid(True, which="major", color=C_GRID, linewidth=0.6, alpha=0.7)
        ax_regime.set_axisbelow(True)

        plt.setp(ax_regime.xaxis.get_majorticklabels(), rotation=30, ha="right")
        fig.tight_layout()
        fig.savefig(save_dir / f"diag_7_candlestick_{asset_name}_{gran}_{direction}.png",
                    dpi=200, bbox_inches="tight")
        plt.close(fig)


# ---------------------------------------------------------------------------
# Analysis 8: Regime-conditioned statistics
# ---------------------------------------------------------------------------
def analysis_regime_stats(trades_df, npz_path, gran, direction, save_dir,
                          cache_path=None, window=20):
    """Compute stats split by trending/ranging using rolling return sign consistency."""
    save_dir.mkdir(parents=True, exist_ok=True)

    trades_df = trades_df.copy()
    trades_df["date"] = pd.to_datetime(trades_df["date"])
    trades_df = trades_df.sort_values("date").reset_index(drop=True)

    npz = np.load(npz_path, allow_pickle=True)
    test_s_hats = npz["test_s_hats"]
    test_tau = np.maximum(test_s_hats, 1.0 - test_s_hats)
    trades_df["tau_t"] = test_tau

    # Compute regime per date-bar: take all assets at each date,
    # determine whether the bar was up or down, compute rolling consistency
    date_groups = trades_df.groupby("date")
    date_stats = []
    for dt, grp in date_groups:
        # Net direction: fraction of assets where return > 0
        frac_up = (grp["return"] > 0).mean()
        date_stats.append({
            "date": dt,
            "frac_up": frac_up,
            "bar_up": 1 if frac_up >= 0.5 else 0,
            "mean_tau": grp["tau_t"].mean(),
            "n_approved": grp["m2_approved"].sum(),
            "n_total": len(grp),
        })
    date_df = pd.DataFrame(date_stats).sort_values("date").reset_index(drop=True)

    # Rolling return sign consistency
    date_df["regime"] = date_df["bar_up"].rolling(window, min_periods=window // 2).mean()
    date_df = date_df.dropna(subset=["regime"])

    # Split into terciles
    tercile_edges = date_df["regime"].quantile([1 / 3, 2 / 3]).values
    date_df["regime_label"] = pd.cut(
        date_df["regime"],
        bins=[-0.01, tercile_edges[0], tercile_edges[1], 1.01],
        labels=["Ranging", "Neutral", "Trending"]
    )

    # Also binary split at median
    median_regime = date_df["regime"].median()

    # Merge regime back to trades
    date_regime_map = dict(zip(date_df["date"], date_df["regime_label"]))
    date_regime_raw = dict(zip(date_df["date"], date_df["regime"]))
    trades_df["regime_label"] = trades_df["date"].map(date_regime_map)
    trades_df["regime_raw"] = trades_df["date"].map(date_regime_raw)
    trades_with_regime = trades_df.dropna(subset=["regime_label"])

    # Compute stats per tercile
    tercile_stats = []
    for label in ["Ranging", "Neutral", "Trending"]:
        subset = trades_with_regime[trades_with_regime["regime_label"] == label]
        n = len(subset)
        if n == 0:
            continue
        approved = subset[subset["m2_approved"]]
        n_app = len(approved)
        win_rate = float((approved["label"] == 1).mean()) if n_app > 0 else 0.0
        mean_ret = float(approved["return"].mean()) if n_app > 0 else 0.0
        mean_tau = float(subset["tau_t"].mean())
        tercile_stats.append({
            "regime": label,
            "n_bars": n,
            "mean_tau": mean_tau,
            "trading_cov": n_app / n if n > 0 else 0.0,
            "n_trades": n_app,
            "win_rate": win_rate,
            "mean_ret": mean_ret,
        })

    # Print table
    print(f"\n    Regime-Conditioned Statistics ({gran} {direction}, window={window}):")
    print(f"    {'Regime':<10} {'N_bars':<8} {'Mean_τ':<8} {'Trd_Cov':<9} {'N_trades':<10} {'Win_Rate':<10} {'Mean_Ret':<10}")
    print(f"    {'-'*65}")
    for s in tercile_stats:
        print(f"    {s['regime']:<10} {s['n_bars']:<8} {s['mean_tau']:<8.4f} "
              f"{s['trading_cov']:<9.4f} {s['n_trades']:<10} "
              f"{s['win_rate']:<10.4f} {s['mean_ret']:<10.6f}")

    # --- Plot ---
    fig, axes = plt.subplots(1, 4, figsize=(16, 5), facecolor="white")
    regimes = [s["regime"] for s in tercile_stats]
    colors_regime = {"Ranging": "#FF9800", "Neutral": "#90A4AE", "Trending": "#4CAF50"}
    bar_colors = [colors_regime.get(r, "#999") for r in regimes]

    for i, (metric, title, fmt) in enumerate([
        ("mean_tau", "Mean τ_t", ".4f"),
        ("trading_cov", "Trading Coverage", ".3f"),
        ("win_rate", "Win Rate", ".3f"),
        ("mean_ret", "Mean Return", ".5f"),
    ]):
        vals = [s[metric] for s in tercile_stats]
        axes[i].bar(regimes, vals, color=bar_colors, alpha=0.8,
                    edgecolor="black", linewidth=0.5)
        axes[i].set_title(title, fontsize=11, fontweight="bold")
        axes[i].grid(axis="y", alpha=0.3)
        for j, v in enumerate(vals):
            axes[i].text(j, v, f"{v:{fmt}}", ha="center", va="bottom", fontsize=8)

    fig.suptitle(f"Analysis 8: Regime-Conditioned Stats — {gran} {direction.upper()}",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(save_dir / f"diag_8_regime_stats_{gran}_{direction}.png",
                dpi=200, bbox_inches="tight")
    plt.close(fig)

    # Save JSON
    result = {
        "window": window,
        "median_regime": float(median_regime),
        "tercile_edges": [float(x) for x in tercile_edges],
        "tercile_stats": tercile_stats,
    }
    with open(save_dir / f"diag_8_regime_stats_{gran}_{direction}.json", "w") as f:
        json.dump(result, f, indent=2)

    return result


# ---------------------------------------------------------------------------
# Analysis 9: Rolling precision + return, OCP vs Utility
# ---------------------------------------------------------------------------
def analysis_rolling_precision(trades_df, gran, direction, save_dir,
                                window=100):
    """Rolling win rate and mean return for OCP vs Utility, all assets combined."""
    save_dir.mkdir(parents=True, exist_ok=True)

    trades_df = trades_df.copy()
    trades_df["date"] = pd.to_datetime(trades_df["date"])
    trades_df = trades_df.sort_values("date").reset_index(drop=True)

    # Utility threshold from summary (already in trades_df via m2_approved for OCP)
    # We need the utility threshold — read from the analysis summary
    util_thr = None
    summary_path = save_dir.parent / "analysis_summary.json"
    if summary_path.exists():
        with open(summary_path) as f:
            summary = json.load(f)
        val_sel = summary.get("rf_temporal_all_features", {}).get("Val_selective", {})
        util_thr = val_sel.get("threshold", 0.5)
    else:
        # Try unified
        summary_path = save_dir.parent.parent / "unified_summary.json"
        if summary_path.exists():
            with open(summary_path) as f:
                summary = json.load(f)
            pg = summary.get("per_gran", {}).get(gran, {})
            util_thr = pg.get("threshold", 0.5)

    if util_thr is None:
        print(f"    [SKIP] Cannot find utility threshold for {gran} {direction}")
        return None

    # OCP-approved trades (sorted by date)
    ocp_trades = trades_df[trades_df["m2_approved"]].copy().reset_index(drop=True)

    # Utility-approved trades
    util_mask = trades_df["m2_prob"] >= util_thr
    util_trades = trades_df[util_mask].copy().reset_index(drop=True)

    if len(ocp_trades) < window or len(util_trades) < window:
        print(f"    [SKIP] Not enough trades for rolling window "
              f"(OCP={len(ocp_trades)}, Util={len(util_trades)}, window={window})")
        return None

    # Rolling stats for OCP
    ocp_rolling_wr = ocp_trades["label"].rolling(window).mean()
    ocp_rolling_ret = ocp_trades["return"].rolling(window).mean()

    # Rolling stats for Utility
    util_rolling_wr = util_trades["label"].rolling(window).mean()
    util_rolling_ret = util_trades["return"].rolling(window).mean()

    # Style (matching risk-coverage plots)
    C_GRID = "#D5D8DC"
    C_OCP = "#8B008B"
    C_UTIL = "#E67E22"
    C_FILL_OCP = "#E8D5F5"
    C_FILL_UTIL = "#FDEBD0"

    # Compute overall fixed win rate and mean return for utility
    util_overall_wr = float(util_trades["label"].mean())
    util_overall_ret = float(util_trades["return"].mean()) * 100
    ocp_overall_wr = float(ocp_trades["label"].mean())
    ocp_overall_ret = float(ocp_trades["return"].mean()) * 100

    # --- Plot: 2 subplots stacked ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), facecolor="white", sharex=False)
    for ax in (ax1, ax2):
        ax.set_facecolor("#FAFAFA")
        for spine in ax.spines.values():
            spine.set_color("black")
            spine.set_linewidth(1.2)

    # === Top: Rolling win rate ===
    ax1.plot(ocp_trades["date"], ocp_rolling_wr, color=C_OCP,
             linewidth=2.0, alpha=0.9, label=f"OCP (n={len(ocp_trades)})", zorder=3)
    ax1.plot(util_trades["date"], util_rolling_wr, color=C_UTIL,
             linewidth=1.5, alpha=0.7, linestyle="--",
             label=f"Utility τ={util_thr:.3f} (n={len(util_trades)})", zorder=2)
    # Fixed reference lines
    ax1.axhline(ocp_overall_wr, color=C_OCP, linewidth=0.8, linestyle=":",
                alpha=0.5, zorder=1)
    ax1.axhline(util_overall_wr, color=C_UTIL, linewidth=0.8, linestyle=":",
                alpha=0.5, zorder=1)
    ax1.axhline(0.5, color=C_GRID, linewidth=0.8, linestyle="-", alpha=0.4)

    ax1.set_ylabel("Rolling Win Rate", fontsize=11, fontweight="bold", labelpad=8)
    ax1.set_title(f"Rolling Precision & Return — {gran} {direction.upper()} "
                  f"(window = {window} trades)", fontsize=13, fontweight="bold", pad=10)
    ax1.legend(fontsize=9, loc="lower left", framealpha=0.9, edgecolor=C_GRID)
    ax1.grid(True, which="major", color=C_GRID, linewidth=0.6, alpha=0.7)
    ax1.set_axisbelow(True)
    ax1.set_ylim(0.3, 1.0)
    ax1.tick_params(axis="both", labelsize=9, width=1.2)
    plt.setp(ax1.get_yticklabels(), fontweight="bold")
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))

    # === Bottom: Rolling mean return ===
    ax2.plot(ocp_trades["date"], ocp_rolling_ret * 100, color=C_OCP,
             linewidth=2.0, alpha=0.9, label="OCP Mean Return (%)", zorder=3)
    ax2.plot(util_trades["date"], util_rolling_ret * 100, color=C_UTIL,
             linewidth=1.5, alpha=0.7, linestyle="--",
             label="Utility Mean Return (%)", zorder=2)
    # Fill between OCP return and zero: green above, red below
    ocp_ret_pct = ocp_rolling_ret.values * 100
    ocp_dates_arr = ocp_trades["date"].values
    valid_mask = ~np.isnan(ocp_ret_pct)
    ax2.fill_between(ocp_dates_arr[valid_mask], 0, ocp_ret_pct[valid_mask],
                     where=ocp_ret_pct[valid_mask] >= 0,
                     color="#27AE60", alpha=0.25, zorder=1)
    ax2.fill_between(ocp_dates_arr[valid_mask], 0, ocp_ret_pct[valid_mask],
                     where=ocp_ret_pct[valid_mask] < 0,
                     color="#E74C3C", alpha=0.25, zorder=1)
    # Fixed reference lines
    ax2.axhline(ocp_overall_ret, color=C_OCP, linewidth=0.8, linestyle=":",
                alpha=0.5, zorder=1)
    ax2.axhline(util_overall_ret, color=C_UTIL, linewidth=0.8, linestyle=":",
                alpha=0.5, zorder=1)
    ax2.axhline(0.0, color=C_GRID, linewidth=0.8, linestyle="-", alpha=0.4)

    ax2.set_ylabel("Rolling Mean Return (%)", fontsize=11, fontweight="bold", labelpad=8)
    ax2.set_xlabel("Date", fontsize=11, fontweight="bold", labelpad=8)
    ax2.legend(fontsize=9, loc="lower left", framealpha=0.9, edgecolor=C_GRID)
    ax2.grid(True, which="major", color=C_GRID, linewidth=0.6, alpha=0.7)
    ax2.set_axisbelow(True)
    ax2.tick_params(axis="both", labelsize=9, width=1.2)
    plt.setp(ax2.get_xticklabels(), fontweight="bold")
    plt.setp(ax2.get_yticklabels(), fontweight="bold")
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))

    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=30, ha="right")
    fig.tight_layout()
    fig.savefig(save_dir / f"diag_9_rolling_precision_{gran}_{direction}.png",
                dpi=200, bbox_inches="tight")
    plt.close(fig)

    # Summary stats
    result = {
        "window": window,
        "ocp_n_trades": len(ocp_trades),
        "util_n_trades": len(util_trades),
        "util_threshold": util_thr,
        "ocp_wr_mean": float(ocp_rolling_wr.dropna().mean()),
        "ocp_wr_std": float(ocp_rolling_wr.dropna().std()),
        "ocp_wr_min": float(ocp_rolling_wr.dropna().min()),
        "ocp_wr_max": float(ocp_rolling_wr.dropna().max()),
        "util_wr_mean": float(util_rolling_wr.dropna().mean()),
        "util_wr_std": float(util_rolling_wr.dropna().std()),
        "util_wr_min": float(util_rolling_wr.dropna().min()),
        "util_wr_max": float(util_rolling_wr.dropna().max()),
        "ocp_ret_mean": float(ocp_rolling_ret.dropna().mean()),
        "ocp_ret_std": float(ocp_rolling_ret.dropna().std()),
        "util_ret_mean": float(util_rolling_ret.dropna().mean()),
        "util_ret_std": float(util_rolling_ret.dropna().std()),
    }

    print(f"\n    Rolling Precision ({gran} {direction}, w={window}):")
    print(f"      OCP:  WR={result['ocp_wr_mean']:.3f}±{result['ocp_wr_std']:.3f} "
          f"[{result['ocp_wr_min']:.3f}, {result['ocp_wr_max']:.3f}]  "
          f"Ret={result['ocp_ret_mean']*100:.3f}%±{result['ocp_ret_std']*100:.3f}%")
    print(f"      Util: WR={result['util_wr_mean']:.3f}±{result['util_wr_std']:.3f} "
          f"[{result['util_wr_min']:.3f}, {result['util_wr_max']:.3f}]  "
          f"Ret={result['util_ret_mean']*100:.3f}%±{result['util_ret_std']*100:.3f}%")

    with open(save_dir / f"diag_9_rolling_precision_{gran}_{direction}.json", "w") as f:
        json.dump(result, f, indent=2)

    return result


# ---------------------------------------------------------------------------
# Runner for new analyses (7-9)
# ---------------------------------------------------------------------------
def run_interpretability(folder, cache_path, assets, grans_filter=None,
                         mode="separate"):
    """Run analyses 7-9 for the given folder."""
    folder = Path(folder)

    def _run_for_one(gran, direction, df, npz_path, diag_dir):
        # Analysis 7: Candlestick
        if cache_path and Path(cache_path).exists():
            print(f"\n  [7] Candlestick analysis ({gran} {direction})...")
            analysis_candlestick(cache_path, gran, direction, df, npz_path,
                                 assets, diag_dir)
        else:
            print(f"\n  [7] SKIP candlestick (no cache file)")

        # Analysis 8: Regime stats
        print(f"\n  [8] Regime-conditioned stats ({gran} {direction})...")
        analysis_regime_stats(df, npz_path, gran, direction, diag_dir,
                              cache_path=cache_path)

        # Analysis 9: Rolling precision
        print(f"\n  [9] Rolling precision ({gran} {direction})...")
        analysis_rolling_precision(df, gran, direction, diag_dir)

    if mode == "separate":
        # Support both single per-gran folder and parent folder
        if (folder / "analysis_summary.json").exists():
            sub_folders = [folder]
        else:
            sub_folders = sorted([
                p.parent for p in folder.glob("*/analysis_summary.json")
            ])
            if grans_filter:
                sub_folders = [
                    sf for sf in sub_folders
                    if any(g in sf.name for g in grans_filter)
                ]

        for sub_folder in sub_folders:
            data = load_separate(sub_folder)
            gran = data["gran"]
            direction = data["direction"]
            df = data["df"]

            npz_candidates = sorted(sub_folder.glob("*_ocp_diagnostics.npz"))
            if not npz_candidates:
                print(f"  [SKIP] {sub_folder.name}: no npz diagnostics")
                continue
            npz_path = npz_candidates[0]
            diag_dir = sub_folder / "ocp_diagnostics"
            _run_for_one(gran, direction, df, npz_path, diag_dir)

    else:  # unified
        all_data = load_unified(folder)
        if not all_data:
            print("ERROR: No granularities with trades CSV found")
            return

        for gran, data in sorted(all_data.items()):
            if grans_filter and gran not in grans_filter:
                continue
            ocp_info = data.get("ocp_info", {})
            if not ocp_info:
                print(f"  [SKIP] {gran}: no OCP data")
                continue

            direction = data["direction"]
            df = data["df"]
            gran_dir = folder / gran
            npz_candidates = sorted(gran_dir.glob("*ocp_diagnostics.npz"))
            if not npz_candidates:
                print(f"  [SKIP] {gran}: no npz diagnostics")
                continue
            npz_path = npz_candidates[0]
            diag_dir = gran_dir / "ocp_diagnostics"
            _run_for_one(gran, direction, df, npz_path, diag_dir)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="OCP Diagnostic Analysis")
    parser.add_argument("--folder", type=str, required=True,
                        help="Path to results folder (separate or unified)")
    parser.add_argument("--mode", type=str, choices=["separate", "unified"], default=None,
                        help="Mode (auto-detected if not provided)")
    parser.add_argument("--grans", type=str, default=None,
                        help="Comma-separated granularities to analyze (unified only, default: all)")
    parser.add_argument("--analyses", type=str, default="1-6",
                        help="Which analyses to run: '1-6' (original diagnostics), "
                             "'7-9' (interpretability), 'all' (both)")
    parser.add_argument("--cache", type=str, default=None,
                        help="Path to Kronos cache .pt file (required for analysis 7)")
    parser.add_argument("--assets", type=str,
                        default="BTCUSDT,ETHUSDT,SOLUSDT,XRPUSDT,DOGEUSDT",
                        help="Comma-separated asset names for candlestick plots")
    args = parser.parse_args()

    folder = Path(args.folder)
    if not folder.exists():
        print(f"ERROR: {folder} does not exist")
        sys.exit(1)

    # Auto-detect mode
    mode = args.mode
    if mode is None:
        if (folder / "unified_summary.json").exists():
            mode = "unified"
        elif (folder / "analysis_summary.json").exists():
            mode = "separate"
        else:
            # Check if parent folder contains per-gran sub-folders
            sub_summaries = list(folder.glob("*/analysis_summary.json"))
            if sub_summaries:
                mode = "separate"
            else:
                print("ERROR: Cannot auto-detect mode. Use --mode separate|unified")
                sys.exit(1)

    # Parse analysis range
    run_original = False
    run_interp = False
    if args.analyses == "all":
        run_original = True
        run_interp = True
    elif args.analyses == "1-6":
        run_original = True
    elif args.analyses == "7-9":
        run_interp = True
    else:
        print(f"ERROR: Unknown --analyses value: {args.analyses}")
        sys.exit(1)

    print(f"Mode: {mode}")
    print(f"Folder: {folder}")
    print(f"Analyses: {args.analyses}")

    grans_filter = args.grans.split(",") if args.grans else None
    assets = args.assets.split(",")

    # --- Collect per-gran sub-folders for separate mode ---
    def _collect_separate_folders(base_folder, grans_filter):
        """Return list of per-gran sub-folders to process."""
        if (base_folder / "analysis_summary.json").exists():
            return [base_folder]
        sub_folders = sorted([
            p.parent for p in base_folder.glob("*/analysis_summary.json")
        ])
        if grans_filter:
            sub_folders = [
                sf for sf in sub_folders
                if any(g in sf.name for g in grans_filter)
            ]
        return sub_folders

    # --- Original diagnostics (1-6) ---
    if run_original:
        print("\n--- Running original diagnostics (1-6) ---")
        if mode == "separate":
            for sub_folder in _collect_separate_folders(folder, grans_filter):
                data = load_separate(sub_folder)
                ocp_info = data.get("ocp_info", {})
                if not ocp_info:
                    print(f"  [SKIP] {sub_folder.name}: no OCP data")
                    continue
                gran_label = f"{data['gran']}_{data['direction']}"
                diag_dir = sub_folder / "ocp_diagnostics"
                run_all_tests(data, diag_dir, gran_label)
        else:
            all_data = load_unified(folder)
            if not all_data:
                print("ERROR: No granularities with trades CSV found")
                sys.exit(1)
            for gran, data in sorted(all_data.items()):
                if grans_filter and gran not in grans_filter:
                    continue
                ocp_info = data.get("ocp_info", {})
                if not ocp_info:
                    print(f"  [SKIP] {gran}: no OCP data")
                    continue
                gran_label = f"unified_{gran}_{data['direction']}"
                diag_dir = folder / gran / "ocp_diagnostics"
                run_all_tests(data, diag_dir, gran_label)

    # --- Interpretability analyses (7-9) ---
    if run_interp:
        print("\n--- Running interpretability analyses (7-9) ---")

        # Auto-detect cache path if not provided
        cache_path = args.cache
        if cache_path is None:
            # Try to find from analysis_summary or unified_summary
            if mode == "separate":
                summary_path = folder / "analysis_summary.json"
            else:
                summary_path = folder / "unified_summary.json"
            if summary_path.exists():
                with open(summary_path) as f:
                    summary = json.load(f)
                cache_path = summary.get("cache", None)

        if cache_path is None or not Path(cache_path).exists():
            print(f"WARNING: Cache file not found ({cache_path}). "
                  f"Analysis 7 (candlestick) will be skipped.")

        run_interpretability(folder, cache_path, assets,
                             grans_filter=grans_filter, mode=mode)

    print("\nDone.")


if __name__ == "__main__":
    main()
