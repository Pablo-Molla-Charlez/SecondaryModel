#!/usr/bin/env python3
"""
OCP Theoretical Experiments
===========================
DEPRECATED: kept separate from `ocp/_analysis_impl.py` because the delayed-
feedback setting breaks the intended coverage-theory story for the current M2
pipeline. Use `python -m Utils.ocp.analysis` for practical diagnostics.

Empirical experiments to support the model-aware precision bound:

    rho >= f(alpha, AUROC, p)

which is strictly tighter than the marginal bound rho >= 1 - alpha/pi
when AUROC > 0.5.

Experiments:
  A. AUROC Ablation — Train RF models of varying quality, run SAOCP on each,
     plot AUROC vs OCP precision to show the model-aware surplus.
  B. Score-Precision Monotonicity — Bin model scores into deciles, compute
     actual TP rate per bin. If monotonically increasing, SAOCP's threshold
     mechanism provably improves precision.
  C. Threshold Trajectory vs Regime — Plot SAOCP's adaptive threshold over
     time alongside regime labels to show the adaptation mechanism.
  D. Conditional Coverage by Regime — Check if the marginal guarantee
     P(y in C_t) >= 1 - alpha holds conditionally per regime.
  E. Precision-Coverage Frontier — For each AUROC level, plot the attainable
     precision at each coverage level, showing higher AUROC shifts the frontier.
  F. Bi-Normal Score Distribution Check — Test whether conformity scores follow
     the bi-normal assumption needed for the model-aware bound.

Usage:
  python -m Utils.ocp.theory --cache Output/Kronos/cache/multi_7_fee_up_*.pt \\
                              --cache Output/Kronos/cache/multi_7_fee_down_*.pt \\
                              [--grans 1d,6h] [--experiments A,B,C,D,E]

  # Run all experiments on all granularities for both directions:
  python -m Utils.ocp.theory --cache Output/Kronos/cache/multi_7_fee_up_*.pt \\
                              --cache Output/Kronos/cache/multi_7_fee_down_*.pt
"""

import argparse
import json
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy import stats as scipy_stats
from scipy.interpolate import PchipInterpolator
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from scipy.stats import shapiro, kstest, norm
from sklearn.preprocessing import StandardScaler

# -- project imports -----------------------------------------------------------
from Utils.data import split_by_global_time
from Utils.utils import _safe_json, _load_config, _infer_direction, _load_multi_cache

warnings.filterwarnings("ignore", category=FutureWarning)

# -- constants -----------------------------------------------------------------
OUTPUT_ROOT = Path(__file__).resolve().parent.parent / "Output" / "Analysis" / "Theory"
RF_PARAMS = dict(n_estimators=500, max_depth=6, min_samples_leaf=20,
                 random_state=42, n_jobs=-1, class_weight="balanced")
ALPHA = 0.10
SEED = 42


# ==============================================================================
# Helpers
# ==============================================================================
def _extract_split_data(dataset, train_end, val_end, direction, fee=0.002,
                        forecast_horizon=7):
    """Extract X_train, y_train, X_val, y_val, X_test, y_test, returns from a dataset dict."""
    eng = dataset["eng_features"]
    if isinstance(eng, torch.Tensor):
        eng = eng.numpy()
    labels = dataset["labels"]
    if isinstance(labels, torch.Tensor):
        labels = labels.numpy()
    returns = dataset["returns"]
    if isinstance(returns, torch.Tensor):
        returns = returns.numpy()

    idx_train, _, idx_val, idx_test = split_by_global_time(
        dataset, train_end=train_end, val_end=val_end)

    X_train = eng[idx_train]
    y_train = labels[idx_train].astype(int)
    X_val = eng[idx_val]
    y_val = labels[idx_val].astype(int)
    X_test = eng[idx_test]
    y_test = labels[idx_test].astype(int)

    ret_val = returns[idx_val].copy()
    ret_test = returns[idx_test].copy()
    if direction.lower() == "down":
        ret_val = -ret_val
        ret_test = -ret_test

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    # Test dates for delayed SAOCP feedback
    test_dates = [dataset["dates"][i] for i in idx_test]

    return {
        "X_train": X_train, "y_train": y_train,
        "X_val": X_val, "y_val": y_val,
        "X_test": X_test, "y_test": y_test,
        "ret_val": ret_val, "ret_test": ret_test,
        "scaler": scaler, "fee": fee,
        "test_dates": test_dates,
        "forecast_horizon": forecast_horizon,
    }


# -- SAOCP: canonical implementation in Utils.saocp --------------------------
from Utils.ocp.saocp import _ocp_conformity_score, _run_saocp_online  # noqa: F401


def _run_saocp(val_probs, val_labels, test_probs, test_labels, alpha=0.10,
               test_dates=None, forecast_horizon=1):
    """Thin wrapper returning (s_hats, approved, covered, pred_sets)."""
    s_hats, approved, _, conf_stats = _run_saocp_online(
        val_probs, val_labels, test_probs, test_labels,
        alpha=alpha, test_dates=test_dates, forecast_horizon=forecast_horizon)
    return s_hats, approved, conf_stats["covered"], conf_stats["pred_sets"]


def _train_and_evaluate(X_train, y_train, X_val, y_val, X_test, y_test,
                        feature_mask=None, noise_std=0.0, permute_labels=False,
                        alpha=0.10, test_dates=None, forecast_horizon=1):
    """Train RF, run SAOCP, return metrics dict."""
    rng = np.random.RandomState(SEED)

    # Apply feature mask (subset of features)
    if feature_mask is not None:
        X_tr = X_train[:, feature_mask]
        X_v = X_val[:, feature_mask]
        X_te = X_test[:, feature_mask]
    else:
        X_tr = X_train.copy()
        X_v = X_val.copy()
        X_te = X_test.copy()

    # Add noise
    if noise_std > 0:
        X_tr = X_tr + rng.randn(*X_tr.shape) * noise_std
        X_v = X_v + rng.randn(*X_v.shape) * noise_std
        X_te = X_te + rng.randn(*X_te.shape) * noise_std

    # Permute labels (destroys signal)
    y_tr = y_train.copy()
    if permute_labels:
        rng.shuffle(y_tr)

    # Train
    model = RandomForestClassifier(**RF_PARAMS)
    model.fit(X_tr, y_tr)

    # Predict
    val_probs = model.predict_proba(X_v)[:, 1]
    test_probs = model.predict_proba(X_te)[:, 1]

    # AUROC
    try:
        auroc_test = roc_auc_score(y_test, test_probs)
    except ValueError:
        auroc_test = 0.5

    # Run SAOCP (with delayed feedback if forecast_horizon > 1)
    s_hats, approved, covered, pred_sets = _run_saocp(
        val_probs, y_val, test_probs, y_test, alpha=alpha,
        test_dates=test_dates, forecast_horizon=forecast_horizon)

    n_total = len(y_test)
    n_approved = int(approved.sum())
    pi = n_approved / n_total if n_total > 0 else 0.0

    if n_approved > 0:
        precision = float((y_test[approved] == 1).mean())
        mean_ret_raw = None  # caller can compute with returns
    else:
        precision = np.nan
        mean_ret_raw = np.nan

    # Conformal coverage
    conf_cov = float(covered.mean()) if n_total > 0 else 0.0

    # Marginal bound
    if pi > 0:
        bound_marginal = max(0.0, 1.0 - alpha / pi)
    else:
        bound_marginal = 0.0

    return {
        "auroc": auroc_test,
        "pi": pi,
        "precision": precision,
        "conf_coverage": conf_cov,
        "n_approved": n_approved,
        "n_total": n_total,
        "bound_marginal": bound_marginal,
        "s_hats": s_hats,
        "approved": approved,
        "covered": covered,
        "pred_sets": pred_sets,
        "test_probs": test_probs,
        "val_probs": val_probs,
    }

# ==============================================================================
# Experiment A: AUROC Ablation
# ==============================================================================

def experiment_A(data, gran, direction, save_dir):
    """Train RF models of varying quality, run SAOCP, plot AUROC vs precision."""
    print(f"\n  [Exp A] AUROC Ablation — {gran} {direction}")
    save_dir.mkdir(parents=True, exist_ok=True)

    X_train = data["X_train"]
    y_train = data["y_train"]
    X_val = data["X_val"]
    y_val = data["y_val"]
    X_test = data["X_test"]
    y_test = data["y_test"]
    ret_test = data["ret_test"]
    fee = data["fee"]
    test_dates = data.get("test_dates")
    fh = data.get("forecast_horizon", 7)
    n_features = X_train.shape[1]

    # Define ablation configurations: (label, feature_mask, noise_std, permute)
    rng = np.random.RandomState(SEED)
    configs = []

    # 1. Permuted labels (AUROC ~0.5 baseline)
    configs.append(("Permuted labels", None, 0.0, True))

    # 2. Random 3 features
    idx_3 = sorted(rng.choice(n_features, size=min(3, n_features), replace=False))
    configs.append(("3 random features", idx_3, 0.0, False))

    # 3. Random 5 features
    idx_5 = sorted(rng.choice(n_features, size=min(5, n_features), replace=False))
    configs.append(("5 random features", idx_5, 0.0, False))

    # 4. Random 10 features
    idx_10 = sorted(rng.choice(n_features, size=min(10, n_features), replace=False))
    configs.append(("10 random features", idx_10, 0.0, False))

    # 5. All features + heavy noise
    configs.append(("All + noise(1.0)", None, 1.0, False))

    # 6. All features + moderate noise
    configs.append(("All + noise(0.5)", None, 0.5, False))

    # 7. All features (full model)
    configs.append(("Full model (23 feats)", None, 0.0, False))

    results = []
    for label, feat_mask, noise_std, permute in configs:
        print(f"    Training: {label} ...", end=" ", flush=True)
        res = _train_and_evaluate(
            X_train, y_train, X_val, y_val, X_test, y_test,
            feature_mask=feat_mask, noise_std=noise_std,
            permute_labels=permute, alpha=ALPHA,
            test_dates=test_dates, forecast_horizon=fh)

        # Compute mean return for approved trades
        if res["n_approved"] > 0:
            mean_ret = float((ret_test[res["approved"]] - fee).mean())
        else:
            mean_ret = np.nan

        row = {
            "label": label,
            "auroc": res["auroc"],
            "pi": res["pi"],
            "precision": res["precision"],
            "bound_marginal": res["bound_marginal"],
            "surplus": res["precision"] - res["bound_marginal"] if not np.isnan(res["precision"]) else np.nan,
            "conf_coverage": res["conf_coverage"],
            "n_approved": res["n_approved"],
            "n_total": res["n_total"],
            "mean_ret": mean_ret,
        }
        results.append(row)
        print(f"AUROC={res['auroc']:.3f}, pi={res['pi']:.3f}, "
              f"prec={res['precision']:.3f}, bound={res['bound_marginal']:.3f}")

    # --- Plot ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), facecolor="white")

    aurocs = [r["auroc"] for r in results]
    precisions = [r["precision"] for r in results]
    bounds = [r["bound_marginal"] for r in results]
    surpluses = [r["surplus"] for r in results]
    labels_short = [r["label"] for r in results]

    # Panel 1: AUROC vs Precision
    ax = axes[0]
    ax.scatter(aurocs, precisions, s=80, c="#1B4F72", zorder=5, label="Actual precision")
    ax.scatter(aurocs, bounds, s=60, c="#E74C3C", marker="^", zorder=4, label="Marginal bound 1-alpha/pi")
    for i, lab in enumerate(labels_short):
        ax.annotate(lab, (aurocs[i], precisions[i]), fontsize=6.5,
                    xytext=(5, 5), textcoords="offset points", alpha=0.8)
    ax.set_xlabel("AUROC", fontsize=11, fontweight="bold")
    ax.set_ylabel("Precision", fontsize=11, fontweight="bold")
    ax.set_title("AUROC vs OCP Precision", fontsize=12, fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel 2: AUROC vs Surplus (precision - bound)
    ax = axes[1]
    valid = [i for i in range(len(surpluses)) if not np.isnan(surpluses[i])]
    ax.bar([labels_short[i] for i in valid], [surpluses[i] for i in valid],
           color=["#2ECC71" if surpluses[i] > 0 else "#E74C3C" for i in valid],
           edgecolor="black", linewidth=0.5)
    ax.set_ylabel("Precision Surplus (rho - bound)", fontsize=10, fontweight="bold")
    ax.set_title("Precision Surplus by Model Quality", fontsize=12, fontweight="bold")
    ax.tick_params(axis="x", rotation=45, labelsize=7)
    ax.grid(axis="y", alpha=0.3)
    ax.axhline(0, color="black", linewidth=0.8)

    # Panel 3: AUROC vs Coverage (pi)
    ax = axes[2]
    pis = [r["pi"] for r in results]
    ax.scatter(aurocs, pis, s=80, c="#8E44AD", zorder=5)
    for i, lab in enumerate(labels_short):
        ax.annotate(lab, (aurocs[i], pis[i]), fontsize=6.5,
                    xytext=(5, 5), textcoords="offset points", alpha=0.8)
    ax.set_xlabel("AUROC", fontsize=11, fontweight="bold")
    ax.set_ylabel("pi = P(C_t = {1})", fontsize=11, fontweight="bold")
    ax.set_title("AUROC vs Trading Probability", fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3)

    fig.suptitle(f"Experiment A: AUROC Ablation — {gran} {direction.upper()}",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(save_dir / f"expA_auroc_ablation_{gran}_{direction}.png",
                dpi=200, bbox_inches="tight")
    plt.close(fig)

    # Save JSON
    with open(save_dir / f"expA_auroc_ablation_{gran}_{direction}.json", "w") as f:
        json.dump(_safe_json(results), f, indent=2)

    print(f"    Saved to {save_dir.name}/")
    return results


# ==============================================================================
# Experiment B: Score-Precision Monotonicity
# ==============================================================================

def experiment_B(data, gran, direction, save_dir):
    """Bin model scores into deciles, compute actual TP rate per bin."""
    print(f"\n  [Exp B] Score-Precision Monotonicity — {gran} {direction}")
    save_dir.mkdir(parents=True, exist_ok=True)

    X_train = data["X_train"]
    y_train = data["y_train"]
    X_test = data["X_test"]
    y_test = data["y_test"]

    # Train full model
    model = RandomForestClassifier(**RF_PARAMS)
    model.fit(X_train, y_train)
    test_probs = model.predict_proba(X_test)[:, 1]

    # Bin into deciles
    n_bins = 10
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    bin_stats = []
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        if i == n_bins - 1:
            mask = (test_probs >= lo) & (test_probs <= hi)
        else:
            mask = (test_probs >= lo) & (test_probs < hi)
        n_in_bin = int(mask.sum())
        if n_in_bin == 0:
            tp_rate = np.nan
        else:
            tp_rate = float((y_test[mask] == 1).mean())
        bin_stats.append({
            "bin": f"[{lo:.1f}, {hi:.1f})",
            "center": float(bin_centers[i]),
            "n_samples": n_in_bin,
            "tp_rate": tp_rate,
            "mean_prob": float(test_probs[mask].mean()) if n_in_bin > 0 else np.nan,
        })

    # Check monotonicity
    valid_rates = [b["tp_rate"] for b in bin_stats if not np.isnan(b["tp_rate"])]
    is_monotone = all(a <= b for a, b in zip(valid_rates, valid_rates[1:]))

    # --- Plot ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), facecolor="white")

    # Panel 1: TP rate per bin
    ax = axes[0]
    centers = [b["center"] for b in bin_stats]
    tp_rates = [b["tp_rate"] for b in bin_stats]
    counts = [b["n_samples"] for b in bin_stats]

    colors = ["#2ECC71" if not np.isnan(t) else "#BDC3C7" for t in tp_rates]
    bars = ax.bar(centers, [t if not np.isnan(t) else 0 for t in tp_rates],
                  width=0.08, color=colors, edgecolor="black", linewidth=0.5, alpha=0.8)
    # Perfect calibration line
    ax.plot([0, 1], [0, 1], "--", color="#E74C3C", linewidth=1.5, label="Perfect calibration")
    ax.set_xlabel("Model Score Bin", fontsize=11, fontweight="bold")
    ax.set_ylabel("P(y=1 | score in bin)", fontsize=11, fontweight="bold")
    ax.set_title("Score-Precision Relationship", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    mono_text = "MONOTONE" if is_monotone else "NOT monotone"
    ax.text(0.05, 0.95, mono_text, transform=ax.transAxes, fontsize=10,
            fontweight="bold", va="top",
            color="#2ECC71" if is_monotone else "#E74C3C")

    # Annotate counts
    for i, (c, t, n) in enumerate(zip(centers, tp_rates, counts)):
        if not np.isnan(t):
            ax.text(c, t + 0.02, f"n={n}", ha="center", fontsize=6.5, alpha=0.7)

    # Panel 2: Score distribution by class
    ax = axes[1]
    ax.hist(test_probs[y_test == 0], bins=50, alpha=0.6, color="#E74C3C",
            label="y=0 (No TP)", density=True, edgecolor="black", linewidth=0.3)
    ax.hist(test_probs[y_test == 1], bins=50, alpha=0.6, color="#2ECC71",
            label="y=1 (TP)", density=True, edgecolor="black", linewidth=0.3)
    ax.set_xlabel("Model Score P(y=1)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Density", fontsize=11, fontweight="bold")
    ax.set_title("Score Distributions by Class", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    fig.suptitle(f"Experiment B: Score-Precision Monotonicity — {gran} {direction.upper()}",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(save_dir / f"expB_score_precision_{gran}_{direction}.png",
                dpi=200, bbox_inches="tight")
    plt.close(fig)

    # Save JSON
    result = {
        "is_monotone": is_monotone,
        "bins": _safe_json(bin_stats),
        "auroc": float(roc_auc_score(y_test, test_probs)),
    }
    with open(save_dir / f"expB_score_precision_{gran}_{direction}.json", "w") as f:
        json.dump(result, f, indent=2)

    print(f"    Monotone: {is_monotone} | Saved to {save_dir.name}/")
    return result


# ==============================================================================
# Experiment C: Threshold Trajectory vs Regime
# ==============================================================================

def experiment_C(data, gran, direction, save_dir):
    """Plot SAOCP's adaptive threshold tau_t over time alongside regime labels."""
    print(f"\n  [Exp C] Threshold Trajectory vs Regime — {gran} {direction}")
    save_dir.mkdir(parents=True, exist_ok=True)

    X_train = data["X_train"]
    y_train = data["y_train"]
    X_val = data["X_val"]
    y_val = data["y_val"]
    X_test = data["X_test"]
    y_test = data["y_test"]
    ret_test = data["ret_test"]
    test_dates = data.get("test_dates")
    fh = data.get("forecast_horizon", 7)

    # Train full model
    model = RandomForestClassifier(**RF_PARAMS)
    model.fit(X_train, y_train)
    val_probs = model.predict_proba(X_val)[:, 1]
    test_probs = model.predict_proba(X_test)[:, 1]

    # Run SAOCP
    s_hats, approved, covered, pred_sets = _run_saocp(
        val_probs, y_val, test_probs, y_test, alpha=ALPHA,
        test_dates=test_dates, forecast_horizon=fh)
    tau_t = np.maximum(s_hats, 1.0 - s_hats)

    # Regime: rolling win rate over window=20 test samples
    window = 20
    rolling_correct = pd.Series(y_test).rolling(window, min_periods=window // 2).mean().values

    # Classify regime: use rolling return direction consistency
    # For simplicity, use rolling fraction of TP (y=1) as proxy
    regime_labels = np.full(len(y_test), "Neutral", dtype=object)
    valid_mask = ~np.isnan(rolling_correct)
    if valid_mask.sum() > 0:
        terciles = np.nanpercentile(rolling_correct[valid_mask], [33, 67])
        for i in range(len(y_test)):
            if np.isnan(rolling_correct[i]):
                continue
            if rolling_correct[i] <= terciles[0]:
                regime_labels[i] = "Adverse"
            elif rolling_correct[i] >= terciles[1]:
                regime_labels[i] = "Favorable"

    # Correlation between tau_t and regime
    regime_numeric = np.where(regime_labels == "Adverse", 0,
                              np.where(regime_labels == "Favorable", 2, 1)).astype(float)
    valid_both = valid_mask & np.isfinite(tau_t)
    if valid_both.sum() > 2:
        corr_tau_regime, p_val_corr = scipy_stats.spearmanr(
            tau_t[valid_both], regime_numeric[valid_both])
    else:
        corr_tau_regime, p_val_corr = np.nan, np.nan

    # --- Plot ---
    fig, axes = plt.subplots(3, 1, figsize=(16, 10), facecolor="white",
                             sharex=True, gridspec_kw={"height_ratios": [2, 1, 1]})

    t_idx = np.arange(len(y_test))

    # Panel 1: tau_t and trade events
    ax = axes[0]
    ax.plot(t_idx, tau_t, color="#1B4F72", linewidth=1.0, alpha=0.8, label="tau_t (SAOCP threshold)")
    # Mark approved trades
    trade_idx = np.where(approved)[0]
    trade_correct = y_test[approved] == 1
    ax.scatter(trade_idx[trade_correct], tau_t[trade_idx[trade_correct]],
               color="#2ECC71", s=15, zorder=5, alpha=0.7, label="Correct trade")
    ax.scatter(trade_idx[~trade_correct], tau_t[trade_idx[~trade_correct]],
               color="#E74C3C", s=15, zorder=5, alpha=0.7, label="Wrong trade")
    ax.set_ylabel("tau_t", fontsize=11, fontweight="bold")
    ax.set_title(f"SAOCP Threshold Trajectory — {gran} {direction.upper()}", fontsize=12, fontweight="bold")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.3)

    # Panel 2: Regime
    ax = axes[1]
    colors_map = {"Adverse": "#E74C3C", "Neutral": "#F5B041", "Favorable": "#2ECC71"}
    for regime in ["Adverse", "Neutral", "Favorable"]:
        mask = regime_labels == regime
        if mask.sum() > 0:
            ax.scatter(t_idx[mask], np.ones(mask.sum()) * (0 if regime == "Adverse" else
                       1 if regime == "Neutral" else 2),
                       color=colors_map[regime], s=3, alpha=0.6, label=regime)
    ax.set_ylabel("Regime", fontsize=10, fontweight="bold")
    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels(["Adverse", "Neutral", "Favorable"], fontsize=8)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.3)

    # Panel 3: Rolling precision
    rolling_prec = pd.Series((y_test == 1).astype(float)).rolling(window, min_periods=window // 2).mean().values
    ax = axes[2]
    ax.plot(t_idx, rolling_prec, color="#8E44AD", linewidth=1.0, alpha=0.8,
            label=f"Rolling TP rate (w={window})")
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.set_xlabel("Test Sample Index", fontsize=11, fontweight="bold")
    ax.set_ylabel("TP Rate", fontsize=10, fontweight="bold")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_dir / f"expC_threshold_trajectory_{gran}_{direction}.png",
                dpi=200, bbox_inches="tight")
    plt.close(fig)

    # Save JSON
    result = {
        "corr_tau_regime": float(corr_tau_regime) if not np.isnan(corr_tau_regime) else None,
        "p_value": float(p_val_corr) if not np.isnan(p_val_corr) else None,
        "n_approved": int(approved.sum()),
        "n_total": len(y_test),
        "mean_tau": float(tau_t.mean()),
        "std_tau": float(tau_t.std()),
        "regime_window": window,
    }
    with open(save_dir / f"expC_threshold_trajectory_{gran}_{direction}.json", "w") as f:
        json.dump(result, f, indent=2)

    print(f"    Corr(tau, regime)={corr_tau_regime:.3f} (p={p_val_corr:.4f}) | "
          f"Saved to {save_dir.name}/")
    return result


# ==============================================================================
# Experiment D: Conditional Coverage by Regime
# ==============================================================================

def experiment_D(data, gran, direction, save_dir):
    """Check if conformal coverage holds conditionally by regime."""
    print(f"\n  [Exp D] Conditional Coverage by Regime — {gran} {direction}")
    save_dir.mkdir(parents=True, exist_ok=True)

    X_train = data["X_train"]
    y_train = data["y_train"]
    X_val = data["X_val"]
    y_val = data["y_val"]
    X_test = data["X_test"]
    y_test = data["y_test"]
    ret_test = data["ret_test"]
    test_dates = data.get("test_dates")
    fh = data.get("forecast_horizon", 7)

    # Train full model
    model = RandomForestClassifier(**RF_PARAMS)
    model.fit(X_train, y_train)
    val_probs = model.predict_proba(X_val)[:, 1]
    test_probs = model.predict_proba(X_test)[:, 1]

    # Run SAOCP
    s_hats, approved, covered, pred_sets = _run_saocp(
        val_probs, y_val, test_probs, y_test, alpha=ALPHA,
        test_dates=test_dates, forecast_horizon=fh)

    # Regime classification using rolling return consistency
    # Use test returns to classify regime
    window = 20
    rolling_ret_sign = pd.Series((ret_test > 0).astype(float)).rolling(
        window, min_periods=window // 2).mean().values

    # Tercile-based regime split
    valid_mask = ~np.isnan(rolling_ret_sign)
    regime_labels = np.full(len(y_test), "Neutral", dtype=object)

    if valid_mask.sum() > 0:
        terciles = np.nanpercentile(rolling_ret_sign[valid_mask], [33, 67])

        for i in range(len(y_test)):
            if not valid_mask[i]:
                continue
            if direction.lower() == "down":
                # For DOWN: low frac_up = trending down = favorable
                if rolling_ret_sign[i] <= terciles[0]:
                    regime_labels[i] = "Trending"
                elif rolling_ret_sign[i] >= terciles[1]:
                    regime_labels[i] = "Ranging"
            else:
                # For UP: high frac_up = trending up = favorable
                if rolling_ret_sign[i] <= terciles[0]:
                    regime_labels[i] = "Ranging"
                elif rolling_ret_sign[i] >= terciles[1]:
                    regime_labels[i] = "Trending"

    # Compute stats per regime
    regime_stats = []
    for regime in ["Ranging", "Neutral", "Trending"]:
        mask = (regime_labels == regime) & valid_mask
        n = int(mask.sum())
        if n == 0:
            continue

        cov_in_regime = float(covered[mask].mean())
        n_approved_regime = int(approved[mask].sum())
        if n_approved_regime > 0:
            prec_regime = float((y_test[mask & approved] == 1).sum() / n_approved_regime)
        else:
            prec_regime = np.nan

        pi_regime = n_approved_regime / n if n > 0 else 0.0

        regime_stats.append({
            "regime": regime,
            "n_samples": n,
            "coverage_P_y_in_C": cov_in_regime,
            "target_coverage": 1.0 - ALPHA,
            "guarantee_met": cov_in_regime >= 1.0 - ALPHA - 0.05,  # 5% tolerance
            "n_approved": n_approved_regime,
            "pi": pi_regime,
            "precision": prec_regime,
        })

    # Print table
    print(f"    {'Regime':<12} {'N':<8} {'Cov(y in C)':<14} {'Target':<10} {'Met?':<8} "
          f"{'N_trade':<10} {'pi':<8} {'Precision':<10}")
    print(f"    {'-'*80}")
    for s in regime_stats:
        met_str = "YES" if s["guarantee_met"] else "NO"
        print(f"    {s['regime']:<12} {s['n_samples']:<8} {s['coverage_P_y_in_C']:<14.4f} "
              f"{s['target_coverage']:<10.2f} {met_str:<8} "
              f"{s['n_approved']:<10} {s['pi']:<8.4f} "
              f"{s['precision']:<10.4f}" if not np.isnan(s['precision'])
              else f"    {s['regime']:<12} {s['n_samples']:<8} {s['coverage_P_y_in_C']:<14.4f} "
              f"{s['target_coverage']:<10.2f} {met_str:<8} "
              f"{s['n_approved']:<10} {s['pi']:<8.4f} {'N/A':<10}")

    # Overall
    overall_cov = float(covered[valid_mask].mean()) if valid_mask.sum() > 0 else np.nan
    print(f"\n    Overall coverage: {overall_cov:.4f} (target >= {1-ALPHA:.2f})")

    # --- Plot ---
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), facecolor="white")
    regimes = [s["regime"] for s in regime_stats]
    colors_regime = {"Ranging": "#FF9800", "Neutral": "#90A4AE", "Trending": "#4CAF50"}

    # Panel 1: Coverage by regime
    ax = axes[0]
    covs = [s["coverage_P_y_in_C"] for s in regime_stats]
    bar_colors = [colors_regime.get(r, "#999") for r in regimes]
    ax.bar(regimes, covs, color=bar_colors, edgecolor="black", linewidth=0.5, alpha=0.8)
    ax.axhline(1 - ALPHA, color="#E74C3C", linestyle="--", linewidth=1.5,
               label=f"Target = {1-ALPHA:.1f}")
    ax.set_ylabel("P(y in C_t)", fontsize=10, fontweight="bold")
    ax.set_title("Conformal Coverage by Regime", fontsize=11, fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    for i, v in enumerate(covs):
        ax.text(i, v + 0.01, f"{v:.3f}", ha="center", fontsize=9)

    # Panel 2: Precision by regime
    ax = axes[1]
    precs = [s["precision"] if not np.isnan(s["precision"]) else 0 for s in regime_stats]
    ax.bar(regimes, precs, color=bar_colors, edgecolor="black", linewidth=0.5, alpha=0.8)
    ax.set_ylabel("Precision", fontsize=10, fontweight="bold")
    ax.set_title("OCP Precision by Regime", fontsize=11, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    for i, v in enumerate(precs):
        ax.text(i, v + 0.01, f"{v:.3f}", ha="center", fontsize=9)

    # Panel 3: Trading probability (pi) by regime
    ax = axes[2]
    pis = [s["pi"] for s in regime_stats]
    ax.bar(regimes, pis, color=bar_colors, edgecolor="black", linewidth=0.5, alpha=0.8)
    ax.set_ylabel("pi = P(C_t = {1})", fontsize=10, fontweight="bold")
    ax.set_title("Trading Probability by Regime", fontsize=11, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    for i, v in enumerate(pis):
        ax.text(i, v + 0.005, f"{v:.3f}", ha="center", fontsize=9)

    fig.suptitle(f"Experiment D: Conditional Coverage — {gran} {direction.upper()}",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(save_dir / f"expD_conditional_coverage_{gran}_{direction}.png",
                dpi=200, bbox_inches="tight")
    plt.close(fig)

    # Save JSON
    result = {
        "alpha": ALPHA,
        "overall_coverage": float(overall_cov) if not np.isnan(overall_cov) else None,
        "regime_window": window,
        "regime_stats": _safe_json(regime_stats),
    }
    with open(save_dir / f"expD_conditional_coverage_{gran}_{direction}.json", "w") as f:
        json.dump(result, f, indent=2)

    print(f"    Saved to {save_dir.name}/")
    return result


# ==============================================================================
# Experiment E: Precision-Coverage Frontier
# ==============================================================================

def experiment_E(data, gran, direction, save_dir):
    """Plot precision-coverage frontier for multiple AUROC levels."""
    print(f"\n  [Exp E] Precision-Coverage Frontier — {gran} {direction}")
    save_dir.mkdir(parents=True, exist_ok=True)

    X_train = data["X_train"]
    y_train = data["y_train"]
    X_test = data["X_test"]
    y_test = data["y_test"]
    n_features = X_train.shape[1]

    rng = np.random.RandomState(SEED)

    # Define model configs (subset of Experiment A)
    configs = [
        ("3 features", sorted(rng.choice(n_features, size=min(3, n_features), replace=False)), 0.0),
        ("10 features", sorted(rng.choice(n_features, size=min(10, n_features), replace=False)), 0.0),
        ("All + noise(0.5)", None, 0.5),
        ("Full model", None, 0.0),
    ]

    # --- Plot ---
    fig, ax = plt.subplots(1, 1, figsize=(10, 7), facecolor="white")
    colors = ["#E74C3C", "#F39C12", "#3498DB", "#1B4F72"]
    frontier_data = []

    for idx, (label, feat_mask, noise_std) in enumerate(configs):
        # Train model
        if feat_mask is not None:
            X_tr = X_train[:, feat_mask]
            X_te = X_test[:, feat_mask]
        else:
            X_tr = X_train.copy()
            X_te = X_test.copy()

        if noise_std > 0:
            X_tr = X_tr + rng.randn(*X_tr.shape) * noise_std
            X_te = X_te + rng.randn(*X_te.shape) * noise_std

        model = RandomForestClassifier(**RF_PARAMS)
        model.fit(X_tr, y_train)
        test_probs = model.predict_proba(X_te)[:, 1]

        try:
            auroc = roc_auc_score(y_test, test_probs)
        except ValueError:
            auroc = 0.5

        # Sweep thresholds to build precision-coverage frontier
        thresholds = np.linspace(0.3, 0.95, 200)
        coverages = []
        precisions = []

        for thr in thresholds:
            sel = test_probs >= thr
            n_sel = int(sel.sum())
            cov = n_sel / len(y_test)
            if n_sel >= 5:
                prec = float((y_test[sel] == 1).mean())
            else:
                prec = np.nan
            coverages.append(cov)
            precisions.append(prec)

        coverages = np.array(coverages)
        precisions = np.array(precisions)

        # Smooth for plotting
        valid = np.isfinite(precisions) & (coverages > 0)
        if valid.sum() >= 2:
            order = np.argsort(coverages[valid])
            cov_s = coverages[valid][order]
            prec_s = precisions[valid][order]
            # Remove duplicate coverages
            umask = np.concatenate(([True], np.diff(cov_s) > 0))
            cov_u, prec_u = cov_s[umask], prec_s[umask]
            if len(cov_u) >= 2:
                grid = np.linspace(cov_u.min(), cov_u.max(), 200)
                interp = PchipInterpolator(cov_u, prec_u, extrapolate=False)(grid)
                v2 = np.isfinite(interp)
                ax.plot(grid[v2], interp[v2], color=colors[idx], linewidth=2.0,
                        label=f"{label} (AUROC={auroc:.3f})")

        frontier_data.append({
            "label": label,
            "auroc": float(auroc),
            "n_threshold_points": int(valid.sum()),
        })

    # Marginal bound line: precision = 1 - alpha/coverage
    cov_grid = np.linspace(ALPHA + 0.01, 1.0, 200)
    bound_line = 1.0 - ALPHA / cov_grid
    ax.plot(cov_grid, bound_line, "--", color="#999999", linewidth=1.5,
            label=f"Marginal bound: 1 - {ALPHA}/pi")

    ax.set_xlabel("Coverage (pi)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Precision (rho)", fontsize=12, fontweight="bold")
    ax.set_title(f"Experiment E: Precision-Coverage Frontier — {gran} {direction.upper()}",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=9, loc="lower left")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)

    fig.tight_layout()
    fig.savefig(save_dir / f"expE_precision_frontier_{gran}_{direction}.png",
                dpi=200, bbox_inches="tight")
    plt.close(fig)

    # Save JSON
    with open(save_dir / f"expE_precision_frontier_{gran}_{direction}.json", "w") as f:
        json.dump(_safe_json(frontier_data), f, indent=2)

    print(f"    Saved to {save_dir.name}/")
    return frontier_data


# ==============================================================================
# Experiment F: Bi-Normal Score Distribution Check
# ==============================================================================

def experiment_F(data, gran, direction, save_dir):
    """Check whether conformity scores follow the bi-normal assumption.

    The model-aware bound assumes class-conditional scores ~ N(mu_k, sigma_k).
    We test this by:
      1. Fitting Gaussians to s|y=0 and s|y=1 on test data
      2. KS test for normality of each class
      3. Variance ratio (equal-variance assumption)
      4. Visual: histograms + fitted Gaussians + Q-Q plots
    """
    print(f"\n  [Exp F] Bi-Normal Score Distribution Check — {gran} {direction}")
    save_dir.mkdir(parents=True, exist_ok=True)

    X_train = data["X_train"]
    y_train = data["y_train"]
    X_val = data["X_val"]
    y_val = data["y_val"]
    X_test = data["X_test"]
    y_test = data["y_test"]

    # Train full model
    model = RandomForestClassifier(**RF_PARAMS)
    model.fit(X_train, y_train)
    test_probs = model.predict_proba(X_test)[:, 1]

    # Conformity scores: s = 1-p if y=1, s = p if y=0
    scores = np.where(y_test == 1, 1.0 - test_probs, test_probs)

    s0 = scores[y_test == 0]  # scores for class 0
    s1 = scores[y_test == 1]  # scores for class 1

    result = {"gran": gran, "direction": direction}

    for label, s_k in [("class0", s0), ("class1", s1)]:
        mu, sigma = np.mean(s_k), np.std(s_k, ddof=1)

        # KS test against fitted normal
        ks_stat, ks_p = kstest(s_k, 'norm', args=(mu, sigma))

        # Shapiro-Wilk (on subsample if too large — limit 5000)
        if len(s_k) > 5000:
            rng = np.random.RandomState(SEED)
            sw_stat, sw_p = shapiro(rng.choice(s_k, 5000, replace=False))
        else:
            sw_stat, sw_p = shapiro(s_k)

        # Skewness and kurtosis
        skew = float(scipy_stats.skew(s_k))
        kurt = float(scipy_stats.kurtosis(s_k))  # excess kurtosis (0 = normal)

        result[label] = {
            "n": int(len(s_k)),
            "mean": float(mu),
            "std": float(sigma),
            "skewness": skew,
            "excess_kurtosis": kurt,
            "ks_stat": float(ks_stat),
            "ks_pvalue": float(ks_p),
            "shapiro_stat": float(sw_stat),
            "shapiro_pvalue": float(sw_p),
        }
        print(f"    {label}: mu={mu:.4f}, sigma={sigma:.4f}, "
              f"skew={skew:.3f}, kurt={kurt:.3f}, "
              f"KS_p={ks_p:.4f}, Shapiro_p={sw_p:.4f}")

    # Variance ratio
    var_ratio = (np.var(s0, ddof=1) / np.var(s1, ddof=1)) if np.var(s1, ddof=1) > 0 else np.nan
    result["variance_ratio"] = float(var_ratio)
    result["equal_variance"] = bool(0.5 <= var_ratio <= 2.0) if not np.isnan(var_ratio) else False

    # Separation (d')
    pooled_std = np.sqrt((np.var(s0, ddof=1) + np.var(s1, ddof=1)) / 2.0)
    d_prime = abs(np.mean(s0) - np.mean(s1)) / pooled_std if pooled_std > 0 else 0.0
    result["d_prime"] = float(d_prime)

    print(f"    Var ratio: {var_ratio:.3f} ({'OK' if result['equal_variance'] else 'UNEQUAL'}), "
          f"d'={d_prime:.3f}")

    # --- Plot: 2x2 grid ---
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), facecolor="white")

    # Panel 1: Histogram of s|y=0 with fitted Gaussian
    ax = axes[0, 0]
    ax.hist(s0, bins=50, density=True, alpha=0.6, color="#3498DB", edgecolor="white", label="s | y=0")
    x_grid = np.linspace(s0.min(), s0.max(), 200)
    mu0, sig0 = np.mean(s0), np.std(s0, ddof=1)
    ax.plot(x_grid, norm.pdf(x_grid, mu0, sig0), color="#1B4F72", linewidth=2,
            label=f"N({mu0:.3f}, {sig0:.3f})")
    ax.set_title("Score Distribution | y=0", fontsize=11, fontweight="bold")
    ax.set_xlabel("Conformity Score", fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Panel 2: Histogram of s|y=1 with fitted Gaussian
    ax = axes[0, 1]
    ax.hist(s1, bins=50, density=True, alpha=0.6, color="#E74C3C", edgecolor="white", label="s | y=1")
    x_grid = np.linspace(s1.min(), s1.max(), 200)
    mu1, sig1 = np.mean(s1), np.std(s1, ddof=1)
    ax.plot(x_grid, norm.pdf(x_grid, mu1, sig1), color="#922B21", linewidth=2,
            label=f"N({mu1:.3f}, {sig1:.3f})")
    ax.set_title("Score Distribution | y=1", fontsize=11, fontweight="bold")
    ax.set_xlabel("Conformity Score", fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Panel 3: Q-Q plot for s|y=0
    ax = axes[1, 0]
    scipy_stats.probplot(s0, dist="norm", plot=ax)
    ax.set_title("Q-Q Plot | y=0", fontsize=11, fontweight="bold")
    ax.grid(True, alpha=0.3)

    # Panel 4: Q-Q plot for s|y=1
    ax = axes[1, 1]
    scipy_stats.probplot(s1, dist="norm", plot=ax)
    ax.set_title("Q-Q Plot | y=1", fontsize=11, fontweight="bold")
    ax.grid(True, alpha=0.3)

    fig.suptitle(f"Experiment F: Bi-Normal Check — {gran} {direction.upper()}\n"
                 f"d'={d_prime:.3f}, Var ratio={var_ratio:.3f}",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(save_dir / f"expF_binormal_check_{gran}_{direction}.png",
                dpi=200, bbox_inches="tight")
    plt.close(fig)

    # Save JSON
    with open(save_dir / f"expF_binormal_check_{gran}_{direction}.json", "w") as f:
        json.dump(_safe_json(result), f, indent=2)

    print(f"    Saved to {save_dir.name}/")
    return result


# ==============================================================================
# Experiment G: Bi-Normal Bound vs Actual Precision (Robustness Check)
# ==============================================================================

def experiment_G(data, gran, direction, save_dir):
    """Check if the model-aware precision bound is conservative despite bi-normal
    assumption being violated (Experiment F showed it doesn't hold).

    The bi-normal model-aware bound is:

        rho >= p * Phi_bar(Phi_inv(alpha/(1-p)) - d') / (p * Phi_bar(Phi_inv(alpha/(1-p)) - d') + alpha)

    where d' = sqrt(2) * Phi_inv(AUROC) is the discriminability index.

    If bound <= actual precision: the bound is CONSERVATIVE (safe to use despite
    violated assumptions — Proposal 2 holds).
    If bound > actual precision: the bound is ANTI-CONSERVATIVE (unsafe — cannot
    use this theorem without valid assumptions).

    We compute this for 7 model variants (same as Experiment A) to check across
    different AUROC levels.
    """
    print(f"\n  [Exp G] Bi-Normal Bound vs Actual Precision — {gran} {direction}")
    save_dir.mkdir(parents=True, exist_ok=True)

    X_train = data["X_train"]
    y_train = data["y_train"]
    X_val = data["X_val"]
    y_val = data["y_val"]
    X_test = data["X_test"]
    y_test = data["y_test"]
    test_dates = data.get("test_dates")
    fh = data.get("forecast_horizon", 7)
    n_features = X_train.shape[1]

    rng = np.random.RandomState(SEED)

    # Class prior
    p = float(y_train.mean())

    # Model variants (same as Experiment A)
    configs = [
        ("Permuted labels", None, 0.0, True),
        ("3 random features", sorted(rng.choice(n_features, size=min(3, n_features), replace=False)), 0.0, False),
        ("5 random features", sorted(rng.choice(n_features, size=min(5, n_features), replace=False)), 0.0, False),
        ("10 random features", sorted(rng.choice(n_features, size=min(10, n_features), replace=False)), 0.0, False),
        ("All + noise(1.0)", None, 1.0, False),
        ("All + noise(0.5)", None, 0.5, False),
        ("Full model (23 feats)", None, 0.0, False),
    ]

    results = []
    for label, feat_mask, noise_std, permute in configs:
        # Prepare data
        if feat_mask is not None:
            X_tr = X_train[:, feat_mask]
            X_v = X_val[:, feat_mask]
            X_te = X_test[:, feat_mask]
        else:
            X_tr = X_train.copy()
            X_v = X_val.copy()
            X_te = X_test.copy()

        if noise_std > 0:
            X_tr = X_tr + rng.randn(*X_tr.shape) * noise_std
            X_v = X_v + rng.randn(*X_v.shape) * noise_std
            X_te = X_te + rng.randn(*X_te.shape) * noise_std

        y_tr = y_train.copy()
        if permute:
            y_tr = rng.permutation(y_tr)

        # Train and get probs
        model = RandomForestClassifier(**RF_PARAMS)
        model.fit(X_tr, y_tr)
        val_probs = model.predict_proba(X_v)[:, 1]
        test_probs = model.predict_proba(X_te)[:, 1]

        try:
            auroc = roc_auc_score(y_test, test_probs)
        except ValueError:
            auroc = 0.5

        # Run SAOCP to get actual precision
        _, approved, _, _ = _run_saocp(val_probs, y_val, test_probs, y_test, alpha=ALPHA,
                                       test_dates=test_dates, forecast_horizon=fh)

        n_approved = int(approved.sum())
        if n_approved > 0:
            actual_precision = float((y_test[approved] == 1).mean())
            pi = n_approved / len(y_test)
        else:
            actual_precision = np.nan
            pi = 0.0

        # Compute bi-normal bound
        from scipy.stats import norm as norm_dist
        d_prime = np.sqrt(2) * norm_dist.ppf(auroc) if 0 < auroc < 1 else 0.0

        if p < 1.0 and d_prime != 0:
            fpr_max = ALPHA / (1.0 - p)
            if fpr_max >= 1.0:
                # Bound is vacuous
                binormal_bound = 0.0
            else:
                phi_inv_fpr = norm_dist.ppf(fpr_max)
                tpr_min = norm_dist.sf(phi_inv_fpr - d_prime)  # sf = 1 - cdf = Phi_bar
                binormal_bound = (p * tpr_min) / (p * tpr_min + ALPHA)
        else:
            binormal_bound = 0.0

        # Marginal bound for comparison
        marginal_bound = max(1.0 - ALPHA / pi, 0.0) if pi > 0 else 0.0

        # Is the bi-normal bound conservative?
        if not np.isnan(actual_precision):
            is_conservative = bool(actual_precision >= binormal_bound)
        else:
            is_conservative = None

        row = {
            "label": label,
            "auroc": float(auroc),
            "d_prime": float(d_prime),
            "p_prior": float(p),
            "pi": float(pi),
            "n_approved": n_approved,
            "actual_precision": float(actual_precision) if not np.isnan(actual_precision) else None,
            "binormal_bound": float(binormal_bound),
            "marginal_bound": float(marginal_bound),
            "is_conservative": is_conservative,
            "gap_actual_minus_bound": float(actual_precision - binormal_bound) if not np.isnan(actual_precision) else None,
        }
        results.append(row)

        status = "SAFE" if is_conservative else ("ANTI-CONSERVATIVE" if is_conservative is False else "N/A")
        print(f"    {label:<25s} AUROC={auroc:.3f} d'={d_prime:.3f} | "
              f"actual={actual_precision:.3f} binormal_bound={binormal_bound:.3f} "
              f"marginal={marginal_bound:.3f} → {status}"
              if not np.isnan(actual_precision) else
              f"    {label:<25s} AUROC={auroc:.3f} | no trades")

    # Count conservative vs anti-conservative
    n_safe = sum(1 for r in results if r["is_conservative"] is True)
    n_anti = sum(1 for r in results if r["is_conservative"] is False)
    n_na = sum(1 for r in results if r["is_conservative"] is None)
    print(f"\n    Summary: {n_safe} conservative, {n_anti} anti-conservative, {n_na} N/A")

    # --- Plot ---
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), facecolor="white")

    valid = [r for r in results if r["actual_precision"] is not None]
    aurocs = [r["auroc"] for r in valid]
    actuals = [r["actual_precision"] for r in valid]
    bi_bounds = [r["binormal_bound"] for r in valid]
    marg_bounds = [r["marginal_bound"] for r in valid]
    labels_list = [r["label"] for r in valid]

    # Panel 1: Actual vs Bi-normal bound
    ax = axes[0]
    ax.scatter(aurocs, actuals, color="#2ECC71", s=80, zorder=5, edgecolors="black",
               linewidth=0.5, label="Actual precision")
    ax.scatter(aurocs, bi_bounds, color="#E74C3C", s=80, zorder=5, marker="^",
               edgecolors="black", linewidth=0.5, label="Bi-normal bound")
    ax.scatter(aurocs, marg_bounds, color="#95A5A6", s=50, zorder=4, marker="s",
               edgecolors="black", linewidth=0.5, label="Marginal bound", alpha=0.6)

    # Connect actual to bound
    for i in range(len(aurocs)):
        color = "#2ECC71" if actuals[i] >= bi_bounds[i] else "#E74C3C"
        ax.plot([aurocs[i], aurocs[i]], [actuals[i], bi_bounds[i]],
                color=color, linewidth=1.5, alpha=0.5)

    ax.set_xlabel("AUROC", fontsize=11, fontweight="bold")
    ax.set_ylabel("Precision", fontsize=11, fontweight="bold")
    ax.set_title("Actual Precision vs Bi-Normal Bound", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Panel 2: Gap (actual - bound) vs AUROC
    ax = axes[1]
    gaps = [r["gap_actual_minus_bound"] for r in valid]
    colors_gap = ["#2ECC71" if g >= 0 else "#E74C3C" for g in gaps]
    ax.bar(range(len(valid)), gaps, color=colors_gap, edgecolor="black", linewidth=0.5, alpha=0.8)
    ax.axhline(0, color="black", linewidth=1)
    ax.set_xticks(range(len(valid)))
    ax.set_xticklabels([f"{l}\n(A={a:.2f})" for l, a in zip(labels_list, aurocs)],
                       fontsize=7, rotation=45, ha="right")
    ax.set_ylabel("Actual − Bi-normal Bound", fontsize=11, fontweight="bold")
    ax.set_title("Gap: Positive = Conservative (Safe)", fontsize=12, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)

    fig.suptitle(f"Experiment G: Bi-Normal Bound Robustness — {gran} {direction.upper()}\n"
                 f"Conservative: {n_safe}/{len(valid)}, Anti-conservative: {n_anti}/{len(valid)}",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(save_dir / f"expG_bound_robustness_{gran}_{direction}.png",
                dpi=200, bbox_inches="tight")
    plt.close(fig)

    # Save JSON
    with open(save_dir / f"expG_bound_robustness_{gran}_{direction}.json", "w") as f:
        json.dump(_safe_json(results), f, indent=2)

    print(f"    Saved to {save_dir.name}/")
    return results


# ==============================================================================
# Experiment H: MLR Validation & MLR Precision Bound
# ==============================================================================

def experiment_H(data, gran, direction, save_dir):
    """Validate the Monotone Likelihood Ratio (MLR) assumption and test the
    MLR-based precision bound.

    The MLR assumption states that f_1(s)/f_0(s) is non-decreasing in s, where
    f_1, f_0 are the class-conditional score densities. Under MLR, the ROC curve
    is concave, and a key lemma gives R(x) >= 2Ax for x in [0, 1/(2A)].

    This leads to a model-aware precision bound that is:
      - Distribution-free (only needs MLR, no Gaussianity)
      - Independent of trading coverage pi
      - Complementary to the marginal bound

    Theorem (MLR Precision Bound):
        rho >= 2Ap / (2Ap + (1-p))
    where A = AUROC, p = class prior.

    Experiment H checks:
      1. Whether MLR holds empirically (likelihood ratio f_1/f_0 increasing)
      2. Whether the MLR bound is conservative (below actual precision)
      3. Comparison of MLR bound vs marginal bound vs bi-normal bound
    """
    print(f"\n  [Exp H] MLR Validation & Precision Bound — {gran} {direction}")
    save_dir.mkdir(parents=True, exist_ok=True)

    X_train = data["X_train"]
    y_train = data["y_train"]
    X_val = data["X_val"]
    y_val = data["y_val"]
    X_test = data["X_test"]
    y_test = data["y_test"]
    test_dates = data.get("test_dates")
    fh = data.get("forecast_horizon", 7)
    n_features = X_train.shape[1]

    rng = np.random.RandomState(SEED)
    p = float(y_train.mean())
    baseline_winrate = float(y_test.mean())  # M1 win rate (test class prior)

    # --- Part 1: Check MLR on full model ---
    model_full = RandomForestClassifier(**RF_PARAMS)
    model_full.fit(X_train, y_train)
    test_probs_full = model_full.predict_proba(X_test)[:, 1]

    # Bin scores into deciles, compute empirical f_1/f_0 ratio per bin
    n_bins = 10
    bin_edges = np.linspace(0, 1, n_bins + 1)
    lr_bins = []
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        in_bin = (test_probs_full >= lo) & (test_probs_full < hi)
        if i == n_bins - 1:
            in_bin = (test_probs_full >= lo) & (test_probs_full <= hi)

        n0_bin = int((in_bin & (y_test == 0)).sum())
        n1_bin = int((in_bin & (y_test == 1)).sum())
        n_total_0 = int((y_test == 0).sum())
        n_total_1 = int((y_test == 1).sum())

        f0 = n0_bin / n_total_0 if n_total_0 > 0 else 0
        f1 = n1_bin / n_total_1 if n_total_1 > 0 else 0
        lr = f1 / f0 if f0 > 0 else (float('inf') if f1 > 0 else 1.0)

        lr_bins.append({
            "bin": f"[{lo:.2f}, {hi:.2f})",
            "n_class0": n0_bin,
            "n_class1": n1_bin,
            "f0": float(f0),
            "f1": float(f1),
            "likelihood_ratio": float(lr) if lr != float('inf') else None,
        })

    # Check monotonicity of likelihood ratio
    lr_values = [b["likelihood_ratio"] for b in lr_bins if b["likelihood_ratio"] is not None]
    n_increasing = sum(1 for i in range(1, len(lr_values)) if lr_values[i] >= lr_values[i - 1])
    n_pairs = len(lr_values) - 1
    mlr_holds = n_increasing >= n_pairs * 0.8  # Allow 1-2 violations
    mlr_fraction = n_increasing / n_pairs if n_pairs > 0 else 0

    print(f"    MLR check: {n_increasing}/{n_pairs} consecutive pairs increasing "
          f"({mlr_fraction:.0%}) → {'HOLDS' if mlr_holds else 'VIOLATED'}")

    # --- Part 2: MLR bound vs actual precision for model variants ---
    configs = [
        ("Permuted labels", None, 0.0, True),
        ("3 random features", sorted(rng.choice(n_features, size=min(3, n_features), replace=False)), 0.0, False),
        ("5 random features", sorted(rng.choice(n_features, size=min(5, n_features), replace=False)), 0.0, False),
        ("10 random features", sorted(rng.choice(n_features, size=min(10, n_features), replace=False)), 0.0, False),
        ("All + noise(1.0)", None, 1.0, False),
        ("All + noise(0.5)", None, 0.5, False),
        ("Full model (23 feats)", None, 0.0, False),
    ]

    bound_results = []
    roc_curves = []  # collect (label, fpr_arr, tpr_arr, auroc) for ROC plot
    for label, feat_mask, noise_std, permute in configs:
        if feat_mask is not None:
            X_tr = X_train[:, feat_mask]
            X_v = X_val[:, feat_mask]
            X_te = X_test[:, feat_mask]
        else:
            X_tr = X_train.copy()
            X_v = X_val.copy()
            X_te = X_test.copy()

        if noise_std > 0:
            X_tr = X_tr + rng.randn(*X_tr.shape) * noise_std
            X_v = X_v + rng.randn(*X_v.shape) * noise_std
            X_te = X_te + rng.randn(*X_te.shape) * noise_std

        y_tr = y_train.copy()
        if permute:
            y_tr = rng.permutation(y_tr)

        model = RandomForestClassifier(**RF_PARAMS)
        model.fit(X_tr, y_tr)
        val_probs = model.predict_proba(X_v)[:, 1]
        test_probs = model.predict_proba(X_te)[:, 1]

        try:
            auroc = roc_auc_score(y_test, test_probs)
        except ValueError:
            auroc = 0.5

        # Collect ROC curve for concavity plot
        from sklearn.metrics import roc_curve as sk_roc_curve
        fpr_arr, tpr_arr, _ = sk_roc_curve(y_test, test_probs)
        roc_curves.append((label, fpr_arr, tpr_arr, auroc))

        # Run SAOCP
        _, approved, _, _ = _run_saocp(val_probs, y_val, test_probs, y_test, alpha=ALPHA,
                                       test_dates=test_dates, forecast_horizon=fh)
        n_approved = int(approved.sum())
        if n_approved > 0:
            actual_prec = float((y_test[approved] == 1).mean())
            pi = n_approved / len(y_test)
        else:
            actual_prec = np.nan
            pi = 0.0

        # MLR bound: 2Ap / (2Ap + (1-p))
        mlr_bound = (2 * auroc * p) / (2 * auroc * p + (1 - p))

        # Marginal bound: 1 - alpha/pi
        marginal_bound = max(1.0 - ALPHA / pi, 0.0) if pi > 0 else 0.0

        # Bi-normal bound (from Experiment G)
        from scipy.stats import norm as norm_dist
        d_prime = np.sqrt(2) * norm_dist.ppf(auroc) if 0 < auroc < 1 else 0.0
        if p < 1.0 and d_prime != 0:
            fpr_max = ALPHA / (1.0 - p)
            if fpr_max < 1.0:
                tpr_min = norm_dist.sf(norm_dist.ppf(fpr_max) - d_prime)
                binormal_bound = (p * tpr_min) / (p * tpr_min + ALPHA)
            else:
                binormal_bound = 0.0
        else:
            binormal_bound = 0.0

        is_mlr_conservative = bool(actual_prec >= mlr_bound) if not np.isnan(actual_prec) else None

        row = {
            "label": label,
            "auroc": float(auroc),
            "p_prior": float(p),
            "pi": float(pi),
            "n_approved": n_approved,
            "actual_precision": float(actual_prec) if not np.isnan(actual_prec) else None,
            "mlr_bound": float(mlr_bound),
            "marginal_bound": float(marginal_bound),
            "binormal_bound": float(binormal_bound),
            "is_mlr_conservative": is_mlr_conservative,
            "mlr_gap": float(actual_prec - mlr_bound) if not np.isnan(actual_prec) else None,
        }
        bound_results.append(row)

        status = "SAFE" if is_mlr_conservative else ("FAIL" if is_mlr_conservative is False else "N/A")
        if not np.isnan(actual_prec):
            print(f"    {label:<25s} A={auroc:.3f} | actual={actual_prec:.3f} "
                  f"MLR={mlr_bound:.3f} marg={marginal_bound:.3f} biN={binormal_bound:.3f} → {status}")
        else:
            print(f"    {label:<25s} A={auroc:.3f} | no trades")

    # Count
    n_safe = sum(1 for r in bound_results if r["is_mlr_conservative"] is True)
    n_fail = sum(1 for r in bound_results if r["is_mlr_conservative"] is False)
    n_total = n_safe + n_fail
    print(f"\n    MLR bound: {n_safe}/{n_total} conservative")

    # --- Plot: 3 panels ---
    fig, axes = plt.subplots(1, 3, figsize=(20, 6), facecolor="white")

    # Panel 1: Likelihood ratio by score bin (MLR check)
    ax = axes[0]
    bin_centers = [(bin_edges[i] + bin_edges[i + 1]) / 2 for i in range(n_bins)]
    lr_plot = [b["likelihood_ratio"] if b["likelihood_ratio"] is not None else 0 for b in lr_bins]
    colors_lr = ["#2ECC71" if i == 0 or lr_plot[i] >= lr_plot[i - 1] else "#E74C3C"
                 for i in range(len(lr_plot))]
    ax.bar(bin_centers, lr_plot, width=0.08, color=colors_lr, edgecolor="black",
           linewidth=0.5, alpha=0.8)
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.set_xlabel("Score bin center", fontsize=10, fontweight="bold")
    ax.set_ylabel("Likelihood Ratio f₁(s)/f₀(s)", fontsize=10, fontweight="bold")
    ax.set_title(f"MLR Check: {n_increasing}/{n_pairs} increasing\n"
                 f"({'MLR HOLDS' if mlr_holds else 'MLR VIOLATED'})",
                 fontsize=11, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)

    # Panel 2: Three bounds vs actual precision
    ax = axes[1]
    valid = [r for r in bound_results if r["actual_precision"] is not None]
    x_pos = range(len(valid))
    labels_short = [r["label"].replace(" features", "f").replace("random ", "")
                    .replace("Full model (23 feats)", "Full")
                    .replace("Permuted labels", "Permuted")
                    .replace("All + noise", "Noise") for r in valid]

    ax.scatter(x_pos, [r["actual_precision"] for r in valid], color="#2ECC71", s=100,
               zorder=5, edgecolors="black", linewidth=0.5, label="Actual", marker="o")
    ax.scatter(x_pos, [r["mlr_bound"] for r in valid], color="#3498DB", s=80,
               zorder=5, edgecolors="black", linewidth=0.5, label="MLR bound", marker="^")
    ax.scatter(x_pos, [r["marginal_bound"] for r in valid], color="#95A5A6", s=60,
               zorder=4, edgecolors="black", linewidth=0.5, label="Marginal", marker="s")
    ax.scatter(x_pos, [r["binormal_bound"] for r in valid], color="#E74C3C", s=60,
               zorder=4, edgecolors="black", linewidth=0.5, label="Bi-normal", marker="D")
    # M1 baseline win rate
    ax.axhline(baseline_winrate, color="#FF6F00", linestyle="--", linewidth=2.0,
               alpha=0.8, label=f"M1 baseline = {baseline_winrate:.3f}", zorder=3)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels_short, fontsize=7, rotation=45, ha="right")
    ax.set_ylabel("Precision", fontsize=10, fontweight="bold")
    ax.set_title("All Bounds vs Actual Precision", fontsize=11, fontweight="bold")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(axis="y", alpha=0.3)

    # Panel 3: Gap (actual - MLR bound)
    ax = axes[2]
    gaps = [r["mlr_gap"] for r in valid]
    colors_gap = ["#2ECC71" if g >= 0 else "#E74C3C" for g in gaps]
    ax.bar(x_pos, gaps, color=colors_gap, edgecolor="black", linewidth=0.5, alpha=0.8)
    ax.axhline(0, color="black", linewidth=1)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels_short, fontsize=7, rotation=45, ha="right")
    ax.set_ylabel("Actual − MLR Bound", fontsize=10, fontweight="bold")
    ax.set_title("Gap: Positive = Conservative (Safe)", fontsize=11, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)

    fig.suptitle(f"Experiment H: MLR Bound — {gran} {direction.upper()} (p={p:.3f})\n"
                 f"MLR bound = 2Ap/(2Ap+1−p) | Conservative: {n_safe}/{n_total}",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(save_dir / f"expH_mlr_bound_{gran}_{direction}.png",
                dpi=200, bbox_inches="tight")
    plt.close(fig)

    # --- ROC Curves plot (concavity check) — 2 panels ---
    fig_roc, (ax_all, ax_detail) = plt.subplots(1, 2, figsize=(16, 8), facecolor="white")
    cmap_roc = plt.cm.viridis(np.linspace(0.1, 0.9, len(roc_curves)))
    x_max = ALPHA / (1.0 - p)

    # ---- Panel 1: All ROC curves ----
    for idx, (lbl, fpr_arr, tpr_arr, auc_val) in enumerate(roc_curves):
        short = (lbl.replace(" features", "f").replace("random ", "")
                 .replace("Full model (23 feats)", "Full")
                 .replace("Permuted labels", "Permuted")
                 .replace("All + noise", "Noise"))
        lw = 2.5 if "Full" in lbl else (1.5 if "Permuted" in lbl else 1.8)
        ls = "--" if "Permuted" in lbl else "-"
        ax_all.plot(fpr_arr, tpr_arr, color=cmap_roc[idx], linewidth=lw,
                    linestyle=ls, label=f"{short} (A={auc_val:.3f})", alpha=0.85)
    ax_all.plot([0, 1], [0, 1], color="gray", linestyle=":", linewidth=1.0,
                alpha=0.6, label="Random (A=0.5)")
    auroc_full = roc_curves[-1][3]
    x_lin = np.linspace(0, min(1.0, 1.0 / (2 * auroc_full)), 200)
    ax_all.plot(x_lin, 2 * auroc_full * x_lin, color="#E74C3C", linestyle="--",
                linewidth=1.5, alpha=0.7, label=f"y=2A·x (A={auroc_full:.3f})")
    ax_all.axvline(x_max, color="#FF6F00", linestyle="-.", linewidth=1.5,
                   alpha=0.7, label=f"x_max = α/(1−p) = {x_max:.3f}")
    ax_all.set_xlim(-0.02, 1.02)
    ax_all.set_ylim(-0.02, 1.02)
    ax_all.set_xlabel("FPR (False Positive Rate)", fontsize=11, fontweight="bold")
    ax_all.set_ylabel("TPR (True Positive Rate)", fontsize=11, fontweight="bold")
    ax_all.set_title(f"All ROC Curves — {gran} {direction.upper()}\n"
                     f"Concavity Check", fontsize=12, fontweight="bold")
    ax_all.legend(fontsize=7, loc="lower right")
    ax_all.grid(alpha=0.3)

    # ---- Panel 2: Permuted + Best AUROC + Full, with x_max intersections ----
    # Identify the 3 curves: Permuted (idx 0), Full (idx -1), Best AUROC
    permuted_curve = roc_curves[0]   # (label, fpr, tpr, auroc)
    full_curve = roc_curves[-1]
    best_idx = int(np.argmax([rc[3] for rc in roc_curves]))
    best_curve = roc_curves[best_idx]

    # Build list of curves to plot (deduplicate if best == full)
    detail_curves = [
        (permuted_curve, "#9B59B6", "--", "Permuted"),
        (full_curve, "#2ECC71", "-", "Full"),
    ]
    if best_idx != len(roc_curves) - 1 and best_idx != 0:
        best_short = (best_curve[0].replace(" features", "f").replace("random ", "")
                      .replace("All + noise", "Noise"))
        detail_curves.append((best_curve, "#3498DB", "-", best_short))

    # Plot the selected curves
    for (lbl, fpr_arr, tpr_arr, auc_val), color, ls, short in detail_curves:
        ax_detail.plot(fpr_arr, tpr_arr, color=color, linewidth=2.5,
                       linestyle=ls, label=f"{short} (A={auc_val:.3f})", alpha=0.9)

    # Diagonal and y=2Ax line
    ax_detail.plot([0, 1], [0, 1], color="gray", linestyle=":", linewidth=1.0, alpha=0.6)
    ax_detail.plot(x_lin, 2 * auroc_full * x_lin, color="#E74C3C", linestyle="--",
                   linewidth=1.5, alpha=0.7, label=f"y=2A·x (A={auroc_full:.3f})")

    # Vertical line at x_max
    ax_detail.axvline(x_max, color="#FF6F00", linestyle="-.", linewidth=1.5,
                      alpha=0.7, label=f"x_max = {x_max:.3f}")

    # Compute and annotate intersections at x_max
    y_linear = 2 * auroc_full * x_max  # y=2Ax at x_max
    intersections = [("2A·x", y_linear, "#E74C3C")]
    for (lbl, fpr_arr, tpr_arr, auc_val), color, ls, short in detail_curves:
        tpr_at_xmax = float(np.interp(x_max, fpr_arr, tpr_arr))
        intersections.append((short, tpr_at_xmax, color))

    # Draw horizontal dashed lines from y-axis to intersection points + markers
    for i, (name, y_val, color) in enumerate(intersections):
        ax_detail.plot(x_max, y_val, "o", color=color, markersize=9, zorder=10,
                       markeredgecolor="black", markeredgewidth=0.8)
        ax_detail.hlines(y_val, 0, x_max, colors=color, linestyles=":",
                         linewidth=1.0, alpha=0.6)
        # Annotate next to the dot (right side, staggered vertically)
        ax_detail.annotate(f"{name}: {y_val:.3f}",
                           xy=(x_max, y_val), xytext=(8, -6 + i * 12),
                           textcoords="offset points", fontsize=8, fontweight="bold",
                           color=color, ha="left", va="center",
                           bbox=dict(boxstyle="round,pad=0.2", fc="white",
                                     ec=color, alpha=0.85),
                           arrowprops=dict(arrowstyle="-", color=color,
                                          lw=0.8, alpha=0.6))

    ax_detail.set_xlim(-0.02, 1.02)
    ax_detail.set_ylim(-0.02, 1.02)
    ax_detail.set_xlabel("FPR (False Positive Rate)", fontsize=11, fontweight="bold")
    ax_detail.set_ylabel("TPR (True Positive Rate)", fontsize=11, fontweight="bold")
    ax_detail.set_title(f"Key Curves at x_max | R(x_max) intersections",
                        fontsize=12, fontweight="bold")
    ax_detail.legend(fontsize=8, loc="lower right")
    ax_detail.grid(alpha=0.3)

    fig_roc.suptitle(f"Experiment H: ROC Curves — {gran} {direction.upper()} "
                     f"(p={p:.3f}, x_max={x_max:.3f})",
                     fontsize=14, fontweight="bold", y=1.02)
    fig_roc.tight_layout()
    fig_roc.savefig(save_dir / f"expH_roc_curves_{gran}_{direction}.png",
                    dpi=200, bbox_inches="tight")
    plt.close(fig_roc)

    # Save JSON
    result = {
        "gran": gran,
        "direction": direction,
        "p_prior": float(p),
        "baseline_winrate": float(baseline_winrate),
        "mlr_check": {
            "n_increasing_pairs": n_increasing,
            "n_total_pairs": n_pairs,
            "fraction_increasing": float(mlr_fraction),
            "mlr_holds": bool(mlr_holds),
            "lr_bins": _safe_json(lr_bins),
        },
        "bound_comparison": _safe_json(bound_results),
        "summary": {
            "n_conservative": n_safe,
            "n_anti_conservative": n_fail,
            "n_total": n_total,
        },
    }
    with open(save_dir / f"expH_mlr_bound_{gran}_{direction}.json", "w") as f:
        json.dump(result, f, indent=2)

    print(f"    Saved to {save_dir.name}/")
    return result


# ==============================================================================
# Experiment I: Model-Specific ROC Bound (Tightest, α-dependent)
# ==============================================================================

def experiment_I(data, gran, direction, save_dir):
    """Compute the model-specific precision bound using the actual ROC curve.

    Proposition: Under MLR, SAOCP achieves:
        rho >= p * R(alpha/(1-p)) / [p * R(alpha/(1-p)) + alpha]

    where R(x) = TPR at FPR = x is read directly from the model's ROC curve.
    This is the TIGHTEST possible bound because it uses the actual ROC curve
    rather than a worst-case approximation (like 2Ax in the MLR bound).

    This bound IS alpha-dependent — it captures SAOCP's concentration effect:
    lower alpha forces operation at lower FPR, where R(x)/x is higher (concave
    ROC), yielding higher guaranteed precision.

    Compared to:
      - Marginal bound (1 - alpha/pi): distribution-free but vacuous when pi ~ alpha
      - MLR bound (2Ap/(2Ap+1-p)): model-aware but alpha-independent (FPR cancels)
      - Bi-normal bound: anti-conservative (violated assumption)
      - ROC-specific bound: tightest, alpha-dependent, model-specific
    """
    print(f"\n  [Exp I] Model-Specific ROC Bound — {gran} {direction}")
    save_dir.mkdir(parents=True, exist_ok=True)

    X_train = data["X_train"]
    y_train = data["y_train"]
    X_val = data["X_val"]
    y_val = data["y_val"]
    X_test = data["X_test"]
    y_test = data["y_test"]
    test_dates = data.get("test_dates")
    fh = data.get("forecast_horizon", 1)
    n_features = X_train.shape[1]

    rng = np.random.RandomState(SEED)
    p = float(y_train.mean())
    baseline_winrate = float(y_test.mean())
    x_max = ALPHA / (1.0 - p)  # FPR constraint from conformal guarantee

    # Model variants
    configs = [
        ("Permuted labels", None, 0.0, True),
        ("3 random features", sorted(rng.choice(n_features, size=min(3, n_features), replace=False)), 0.0, False),
        ("5 random features", sorted(rng.choice(n_features, size=min(5, n_features), replace=False)), 0.0, False),
        ("10 random features", sorted(rng.choice(n_features, size=min(10, n_features), replace=False)), 0.0, False),
        ("All + noise(1.0)", None, 1.0, False),
        ("All + noise(0.5)", None, 0.5, False),
        ("Full model (23 feats)", None, 0.0, False),
    ]

    results = []
    for label, feat_mask, noise_std, permute in configs:
        if feat_mask is not None:
            X_tr = X_train[:, feat_mask]
            X_v = X_val[:, feat_mask]
            X_te = X_test[:, feat_mask]
        else:
            X_tr = X_train.copy()
            X_v = X_val.copy()
            X_te = X_test.copy()

        if noise_std > 0:
            X_tr = X_tr + rng.randn(*X_tr.shape) * noise_std
            X_v = X_v + rng.randn(*X_v.shape) * noise_std
            X_te = X_te + rng.randn(*X_te.shape) * noise_std

        y_tr = y_train.copy()
        if permute:
            y_tr = rng.permutation(y_tr)

        model = RandomForestClassifier(**RF_PARAMS)
        model.fit(X_tr, y_tr)
        val_probs = model.predict_proba(X_v)[:, 1]
        test_probs = model.predict_proba(X_te)[:, 1]

        try:
            auroc = roc_auc_score(y_test, test_probs)
        except ValueError:
            auroc = 0.5

        # Compute R(x_max) from the actual ROC curve
        from sklearn.metrics import roc_curve
        fpr_arr, tpr_arr, _ = roc_curve(y_test, test_probs)
        # Interpolate TPR at FPR = x_max
        tpr_at_xmax = float(np.interp(x_max, fpr_arr, tpr_arr))

        # ROC-specific bound: p * R(x_max) / (p * R(x_max) + alpha)
        roc_bound = (p * tpr_at_xmax) / (p * tpr_at_xmax + ALPHA) if tpr_at_xmax > 0 else 0.0

        # Also compute R'(x_max) — slope of ROC at x_max (for the critic's analysis)
        # Use finite difference around x_max
        dx = 0.005
        tpr_lo = float(np.interp(max(x_max - dx, 0), fpr_arr, tpr_arr))
        tpr_hi = float(np.interp(min(x_max + dx, 1), fpr_arr, tpr_arr))
        roc_slope = (tpr_hi - tpr_lo) / (2 * dx) if dx > 0 else 0.0

        # R(x)/x = average slope from origin
        avg_slope = tpr_at_xmax / x_max if x_max > 0 else 0.0

        # Run SAOCP for actual precision
        _, approved, _, _ = _run_saocp(val_probs, y_val, test_probs, y_test, alpha=ALPHA,
                                       test_dates=test_dates, forecast_horizon=fh)
        n_approved = int(approved.sum())
        if n_approved > 0:
            actual_prec = float((y_test[approved] == 1).mean())
            pi = n_approved / len(y_test)
        else:
            actual_prec = np.nan
            pi = 0.0

        # Other bounds for comparison
        mlr_bound = (2 * auroc * p) / (2 * auroc * p + (1 - p))
        marginal_bound = max(1.0 - ALPHA / pi, 0.0) if pi > 0 else 0.0

        is_roc_conservative = bool(actual_prec >= roc_bound) if not np.isnan(actual_prec) else None

        row = {
            "label": label,
            "auroc": float(auroc),
            "p_prior": float(p),
            "pi": float(pi),
            "n_approved": n_approved,
            "x_max_fpr": float(x_max),
            "tpr_at_xmax": float(tpr_at_xmax),
            "roc_slope_at_xmax": float(roc_slope),
            "avg_slope_from_origin": float(avg_slope),
            "actual_precision": float(actual_prec) if not np.isnan(actual_prec) else None,
            "roc_bound": float(roc_bound),
            "mlr_bound": float(mlr_bound),
            "marginal_bound": float(marginal_bound),
            "is_roc_conservative": is_roc_conservative,
            "roc_gap": float(actual_prec - roc_bound) if not np.isnan(actual_prec) else None,
        }
        results.append(row)

        status = "SAFE" if is_roc_conservative else ("FAIL" if is_roc_conservative is False else "N/A")
        if not np.isnan(actual_prec):
            print(f"    {label:<25s} A={auroc:.3f} R({x_max:.3f})={tpr_at_xmax:.3f} "
                  f"R'={roc_slope:.1f} | actual={actual_prec:.3f} "
                  f"ROC={roc_bound:.3f} MLR={mlr_bound:.3f} marg={marginal_bound:.3f} → {status}")
        else:
            print(f"    {label:<25s} A={auroc:.3f} | no trades")

    n_safe = sum(1 for r in results if r["is_roc_conservative"] is True)
    n_fail = sum(1 for r in results if r["is_roc_conservative"] is False)
    n_total = n_safe + n_fail
    print(f"\n    ROC bound: {n_safe}/{n_total} conservative | baseline={baseline_winrate:.3f}")

    # --- Plot: 2 panels ---
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), facecolor="white")

    valid = [r for r in results if r["actual_precision"] is not None]
    x_pos = list(range(len(valid)))
    labels_short = [r["label"].replace(" features", "f").replace("random ", "")
                    .replace("Full model (23 feats)", "Full")
                    .replace("Permuted labels", "Permuted")
                    .replace("All + noise", "Noise") for r in valid]

    # Panel 1: All bounds comparison (including ROC-specific)
    ax = axes[0]
    ax.scatter(x_pos, [r["actual_precision"] for r in valid], color="#2ECC71", s=120,
               zorder=6, edgecolors="black", linewidth=0.5, label="Actual ρ", marker="o")
    ax.scatter(x_pos, [r["roc_bound"] for r in valid], color="#9B59B6", s=90,
               zorder=5, edgecolors="black", linewidth=0.5, label="ROC bound (Prop.)", marker="P")
    ax.scatter(x_pos, [r["mlr_bound"] for r in valid], color="#3498DB", s=70,
               zorder=5, edgecolors="black", linewidth=0.5, label="MLR bound (Thm 2)", marker="^")
    ax.scatter(x_pos, [r["marginal_bound"] for r in valid], color="#95A5A6", s=50,
               zorder=4, edgecolors="black", linewidth=0.5, label="Marginal (Thm 1)", marker="s")
    # Baseline
    ax.axhline(baseline_winrate, color="#FF6F00", linestyle="--", linewidth=2.0,
               alpha=0.8, label=f"M1 baseline = {baseline_winrate:.3f}", zorder=3)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels_short, fontsize=7, rotation=45, ha="right")
    ax.set_ylabel("Precision", fontsize=11, fontweight="bold")
    ax.set_title("All Bounds vs Actual Precision", fontsize=12, fontweight="bold")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, 1.05)

    # Panel 2: Gaps for all three bounds
    ax = axes[1]
    width = 0.25
    x_arr = np.array(x_pos)
    roc_gaps = [r["roc_gap"] if r["roc_gap"] is not None else 0 for r in valid]
    mlr_gaps = [r["actual_precision"] - r["mlr_bound"] if r["actual_precision"] is not None else 0 for r in valid]
    marg_gaps = [r["actual_precision"] - r["marginal_bound"] if r["actual_precision"] is not None else 0 for r in valid]

    ax.bar(x_arr - width, marg_gaps, width, color="#95A5A6", edgecolor="black",
           linewidth=0.3, alpha=0.8, label="Actual − Marginal")
    ax.bar(x_arr, mlr_gaps, width, color="#3498DB", edgecolor="black",
           linewidth=0.3, alpha=0.8, label="Actual − MLR")
    ax.bar(x_arr + width, roc_gaps, width, color="#9B59B6", edgecolor="black",
           linewidth=0.3, alpha=0.8, label="Actual − ROC")
    ax.axhline(0, color="black", linewidth=1)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels_short, fontsize=7, rotation=45, ha="right")
    ax.set_ylabel("Actual − Bound (positive = safe)", fontsize=10, fontweight="bold")
    ax.set_title("Gap Comparison: Tighter = Smaller Gap", fontsize=12, fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    fig.suptitle(f"Experiment I: ROC-Specific Bound — {gran} {direction.upper()} "
                 f"(p={p:.3f}, baseline={baseline_winrate:.3f})\n"
                 f"ROC bound = p·R(α/(1−p)) / [p·R(α/(1−p)) + α] | "
                 f"Conservative: {n_safe}/{n_total}",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(save_dir / f"expI_roc_bound_{gran}_{direction}.png",
                dpi=200, bbox_inches="tight")
    plt.close(fig)

    # Save JSON
    out = {
        "gran": gran,
        "direction": direction,
        "p_prior": float(p),
        "baseline_winrate": float(baseline_winrate),
        "alpha": ALPHA,
        "x_max_fpr": float(x_max),
        "bound_comparison": _safe_json(results),
        "summary": {
            "n_conservative": n_safe,
            "n_anti_conservative": n_fail,
            "n_total": n_total,
        },
    }
    with open(save_dir / f"expI_roc_bound_{gran}_{direction}.json", "w") as f:
        json.dump(out, f, indent=2)

    print(f"    Saved to {save_dir.name}/")
    return out


# ==============================================================================
# Cross-config summary table
# ==============================================================================

def generate_summary_table(all_results, save_dir):
    """Generate a summary table across all configs for Experiment A."""
    save_dir.mkdir(parents=True, exist_ok=True)

    # Collect all Experiment A rows
    rows = []
    for key, exp_results in all_results.items():
        gran, direction = key
        if "A" not in exp_results:
            continue
        for r in exp_results["A"]:
            rows.append({
                "gran": gran,
                "direction": direction,
                "model": r["label"],
                "auroc": r["auroc"],
                "pi": r["pi"],
                "precision": r["precision"],
                "bound": r["bound_marginal"],
                "surplus": r["surplus"],
                "n_trades": r["n_approved"],
            })

    if not rows:
        return

    df = pd.DataFrame(rows)

    # Pivot: full model only, across all configs
    full = df[df["model"] == "Full model (23 feats)"].copy()
    if len(full) > 0:
        full["config"] = full["gran"] + "_" + full["direction"]
        print("\n" + "="*80)
        print("SUMMARY: Full Model across all configurations")
        print("="*80)
        print(f"{'Config':<16} {'AUROC':<8} {'pi':<8} {'Precision':<12} "
              f"{'Bound':<8} {'Surplus':<10} {'N_trades':<10}")
        print("-"*72)
        for _, row in full.iterrows():
            print(f"{row['config']:<16} {row['auroc']:<8.3f} {row['pi']:<8.3f} "
                  f"{row['precision']:<12.4f} {row['bound']:<8.3f} "
                  f"{row['surplus']:<10.4f} {row['n_trades']:<10}")

    # Save full table
    df.to_csv(save_dir / "summary_all_experiments.csv", index=False)

    # Save full model summary JSON
    with open(save_dir / "summary_full_model.json", "w") as f:
        json.dump(_safe_json(full.to_dict("records") if len(full) > 0 else []), f, indent=2)

    print(f"\n  Summary saved to {save_dir}/")


# ==============================================================================
# Main
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="OCP Theoretical Experiments — empirical support for model-aware precision bound")
    parser.add_argument("--cache", type=str, action="append", required=True,
                        help="Path(s) to multi-gran cache .pt files (use multiple --cache for up+down)")
    parser.add_argument("--grans", type=str, default=None,
                        help="Comma-separated granularities to process (default: all)")
    parser.add_argument("--experiments", type=str, default="A,B,C,D,E,F,G,H,I",
                        help="Comma-separated experiments to run (default: A,B,C,D,E,F,G,H,I)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output root (default: Output/Analysis/Theory)")
    args = parser.parse_args()

    # We use a hardcoded path relative to the script because of its standalone nature.
    cfg = _load_config(str(Path(__file__).resolve().parent.parent / "config.yaml"))
    train_end = cfg["data"]["split"]["train_end"]
    val_end = cfg["data"]["split"]["val_end"]
    fee = cfg.get("evaluation", {}).get("fee_per_trade", 0.002)

    grans_filter = set(args.grans.split(",")) if args.grans else None
    experiments = set(args.experiments.upper().split(","))
    output_root = Path(args.output) if args.output else OUTPUT_ROOT

    print(f"[ocp_theory] Experiments: {sorted(experiments)}")
    print(f"[ocp_theory] Output: {output_root}")
    print(f"[ocp_theory] Train end: {train_end} | Val end: {val_end} | Fee: {fee}")

    all_results = {}

    for cache_str in args.cache:
        cache_path = Path(cache_str).resolve()
        if not cache_path.exists():
            print(f"  [SKIP] Cache not found: {cache_path}")
            continue

        direction = _infer_direction(cache_path)
        print(f"\n{'='*70}")
        print(f"[ocp_theory] Loading cache: {cache_path.name} (direction={direction})")
        print(f"{'='*70}")

        multi = _load_multi_cache(cache_path)

        for gran in multi.grans:
            if grans_filter and gran not in grans_filter:
                continue

            print(f"\n--- {gran} {direction} ---")
            sub = multi.sub[gran]

            data = _extract_split_data(sub, train_end, val_end, direction, fee)
            print(f"  Train: {len(data['X_train'])} | Val: {len(data['X_val'])} | "
                  f"Test: {len(data['X_test'])} | Features: {data['X_train'].shape[1]}")

            key = (gran, direction)
            all_results[key] = {}

            if "A" in experiments:
                all_results[key]["A"] = experiment_A(
                    data, gran, direction, output_root / "ExperimentA")

            if "B" in experiments:
                all_results[key]["B"] = experiment_B(
                    data, gran, direction, output_root / "ExperimentB")

            if "C" in experiments:
                all_results[key]["C"] = experiment_C(
                    data, gran, direction, output_root / "ExperimentC")

            if "D" in experiments:
                all_results[key]["D"] = experiment_D(
                    data, gran, direction, output_root / "ExperimentD")

            if "E" in experiments:
                all_results[key]["E"] = experiment_E(
                    data, gran, direction, output_root / "ExperimentE")

            if "F" in experiments:
                all_results[key]["F"] = experiment_F(
                    data, gran, direction, output_root / "ExperimentF")

            if "G" in experiments:
                all_results[key]["G"] = experiment_G(
                    data, gran, direction, output_root / "ExperimentG")

            if "H" in experiments:
                all_results[key]["H"] = experiment_H(
                    data, gran, direction, output_root / "ExperimentH")

            if "I" in experiments:
                all_results[key]["I"] = experiment_I(
                    data, gran, direction, output_root / "ExperimentI")

    # Cross-config summary
    if "A" in experiments:
        generate_summary_table(all_results, output_root)

    print(f"\n{'='*70}")
    print(f"[ocp_theory] All experiments complete. Output: {output_root}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
