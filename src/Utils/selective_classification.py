"""Selective-classification utilities for kronos_tree."""
import numpy as np

# ┏━━━━━━━━━━ Collect Risk-Coverage Curve ━━━━━━━━━━┓
def collect_risk_coverage_curve(y_true,
                                y_score,
                                thresholds = None,
                                empty_selection_value = np.nan,
                                include_error_counts: bool = False):
  
    """Return coverage and risk arrays for the given score thresholds."""
    
    # ┏━━━━━━━━━━ Format Change ━━━━━━━━━━┓
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)

    # ┏━━━━━━━━━━ Safe-Check Conditions ━━━━━━━━━━┓
    if y_true.shape[0] != y_score.shape[0]:
        raise ValueError("y_true and y_score must have the same length")
    total = y_score.shape[0]
    if total == 0:
        raise ValueError("Inputs must contain at least one sample")
    
    # ┏━━━━━━━━━━ Format Change ━━━━━━━━━━┓
    if thresholds is None:
        thresholds = np.sort(np.unique(y_score))[::-1]
    else:
        thresholds = np.asarray(thresholds)

    # ┏━━━━━━━━━━ Definition of Empty Variables ━━━━━━━━━━┓
    coverages       = np.empty(thresholds.shape, dtype=float)
    risks           = np.empty(thresholds.shape, dtype=float)
    selected_counts = np.empty(thresholds.shape, dtype=int)
    error_counts    = np.empty(thresholds.shape, dtype=int) if include_error_counts else None

    # ┏━━━━━━━━━━ Computing Coverage & Risk Pairs and Error ━━━━━━━━━━┓
    for idx, threshold in enumerate(thresholds):
        selected = y_score >= threshold
        selected_count = int(np.sum(selected))

        coverages[idx] = selected_count / total
        selected_counts[idx] = selected_count

        if selected_count == 0:
            risks[idx] = empty_selection_value
            if include_error_counts:
                error_counts[idx] = 0
        else:
            errors = (y_true[selected] == 0)
            err_count = int(np.sum(errors))
            risks[idx] = err_count / selected_count
            if include_error_counts:
                error_counts[idx] = err_count
        
    # ┏━━━━━━━━━━ Dict to store Curve's Data ━━━━━━━━━━┓
    curve = {"thresholds": thresholds,
             "coverage": coverages,
             "risk": risks,
             "selected_count": selected_counts}
    
    # ┏━━━━━━━━━━ Curve's Additional Data ━━━━━━━━━━┓
    if include_error_counts:
        curve["error_count"] = error_counts

    return curve


# ┏━━━━━━━━━━ Find Best Utility Threshold ━━━━━━━━━━┓
def _find_best_utility_threshold(probs: np.ndarray,
                                 returns: np.ndarray,
                                 fee: float = 0.0,
                                 n_prior: int = 50,
                                 labels: np.ndarray = None) -> dict:
    """Pick the validation threshold that maximizes a regularized utility score.
    Selection must beat the argmax (thr=0.50) baseline on precision when labels
    are provided, otherwise the optimizer refuses to commit and abstains.

    TO BE IMPROVED/STUDIED IN DETAIL.
    """
    # ┏━━━━━━━━━━ Format Change ━━━━━━━━━━┓
    probs = np.asarray(probs, dtype=float)
    returns = np.asarray(returns, dtype=float)
    N_total = len(probs)
    min_trades = max(50, int(0.005 * N_total))

    # ┏━━━━━━━━━━ Base Variance ━━━━━━━━━━┓
    all_net = returns[probs >= 0.50] - fee
    base_var = float(np.nanvar(all_net, ddof=1)) if len(all_net) > 1 else 1.0
    if base_var <= 0:
        base_var = 1.0

    # ┏━━━━━━━━━━ Precision Floor ━━━━━━━━━━┓
    if labels is not None:
        labels = np.asarray(labels).astype(int)
        sel_argmax = probs >= 0.50
        n_argmax = int(sel_argmax.sum())
        prec_argmax = float(labels[sel_argmax].mean()) if n_argmax > 0 else 0.0
    else:
        prec_argmax = 0.0

    # ┏━━━━━━━━━━ Grid Initialization ━━━━━━━━━━┓
    # Start the grid above the median of the positive-side probs so we actually
    # explore the discriminative regime instead of trivially re-selecting all.
    pos_probs = probs[probs >= 0.50]
    if pos_probs.size >= max(min_trades * 2, 20):
        grid_lo = float(np.median(pos_probs))
        grid_lo = min(max(grid_lo, 0.50), 0.85)
    else:
        grid_lo = 0.50
    thr_grid = np.linspace(grid_lo, 0.95, 200)

    best = {"threshold": 0.50,
            "utility": -np.inf,
            "coverage": 1.0,
            "selected_count": len(probs),
            "constraint_satisfied": True,
            "mean_ret": 0.0,
            "t_stat": 0.0}

    # ┏━━━━━━━━━━ Iterate through thresholds ━━━━━━━━━━┓
    for thr in thr_grid:
        sel = probs >= thr
        n = int(sel.sum())
        if n < min_trades:
            continue
        net_rets = returns[sel] - fee
        mu = float(np.nanmean(net_rets))
        if mu <= 0:
            continue
        if labels is not None:
            prec_thr = float(labels[sel].mean())
            if prec_thr < prec_argmax:
                continue
        sample_var = float(np.nanvar(net_rets, ddof=1))
        shrinkage = n_prior / (n + n_prior)
        reg_var = (1 - shrinkage) * sample_var + shrinkage * base_var
        reg_std = np.sqrt(reg_var)
        if reg_std <= 0:
            continue
        t_reg = mu / reg_std * np.sqrt(n)
        
        # ┏━━━━━━━━━━ Concave Penalty on High Coverage ━━━━━━━━━━┓
        cov = n / N_total
        cov_pen = (1.0 - cov) ** 0.5
        utility = t_reg * cov_pen
        if utility > best["utility"]:
            best = {"threshold": float(thr),
                    "utility": float(utility),
                    "coverage": cov,
                    "selected_count": n,
                    "constraint_satisfied": True,
                    "mean_ret": mu,
                    "t_stat": float(t_reg)}

    # ┏━━━━━━━━━━ If no threshold found, find one that satisfies constraint ━━━━━━━━━━┓
    if best["utility"] == -np.inf:
        cov_penalty = 0.10
        sel_base = probs >= 0.50
        cov_base = max(int(sel_base.sum()) / N_total, 1e-9)

        best_score = np.inf
        for thr in np.linspace(0.50, 0.90, 200):
            sel = probs >= thr
            n = int(sel.sum())
            if n < min_trades:
                continue
            n_err = int((returns[sel] < fee).sum())
            risk = n_err / n
            cov = n / N_total
            score = risk + cov_penalty * (1 - cov / cov_base)
            if score < best_score:
                best_score = score
                net_rets = returns[sel] - fee
                mu = float(np.nanmean(net_rets))
                sample_var = float(np.nanvar(net_rets, ddof=1))
                shrinkage = n_prior / (n + n_prior)
                reg_var = (1 - shrinkage) * sample_var + shrinkage * base_var
                reg_std = np.sqrt(max(reg_var, 1e-12))
                t_val = mu / reg_std * np.sqrt(n)
                best = {"threshold":            float(thr),
                        "utility":              0.0,
                        "coverage":             cov,
                        "selected_count":       n,
                        "constraint_satisfied": False,
                        "threshold_source":     "risk-fallback",
                        "mean_ret":             mu,
                        "t_stat":               float(t_val)}

        if best["utility"] == -np.inf:
            sel_50 = probs >= 0.50
            n_50 = int(sel_50.sum())
            net_50 = returns[sel_50] - fee if n_50 > 0 else np.array([0.0])
            best = {"threshold":            0.50,
                    "utility":              0.0,
                    "coverage":             n_50 / N_total if N_total > 0 else 0.0,
                    "selected_count":       n_50,
                    "constraint_satisfied": False,
                    "mean_ret":             float(np.nanmean(net_50)),
                    "t_stat":               0.0}

    return best


# ┏━━━━━━━━━━ Identity calibrator (fallback when isotonic degenerates) ━━━━━━━━━━┓
class _IdentityCalibrator:
    """No-op calibrator: predict(x) = clip(x, 0, 1). Used when isotonic collapses."""
    def fit(self, X, y):
        return self
    def predict(self, X):
        return np.clip(np.asarray(X, dtype=float), 0.0, 1.0)


# ┏━━━━━━━━━━ Calibrate Probabilities (Isotonic Regression + fallback) ━━━━━━━━━━┓
def calibrate_probabilities(val_probs: np.ndarray,
                            val_labels: np.ndarray,
                            test_probs: np.ndarray = None,
                            min_unique_out: int = 5,
                            min_range_out: float = 0.10):
    """Calibrate predicted probabilities using isotonic regression with safety fallback.

    Fits isotonic regression on (val_probs, val_labels). If the fitted calibrator
    is degenerate on its own training probs — fewer than ``min_unique_out`` distinct
    output values, or an output range smaller than ``min_range_out`` — the calibrator
    is replaced with an identity mapping so downstream threshold sweeps remain valid.
    This protects against small Cal sets where isotonic regression collapses all
    probabilities to a single value.

    Returns a dict with calibrated arrays and the fitted calibrator.
    """
    from sklearn.isotonic import IsotonicRegression

    # ┏━━━━━━━━━━ Format Change ━━━━━━━━━━┓
    val_probs  = np.asarray(val_probs, dtype=float)
    val_labels = np.asarray(val_labels, dtype=float)

    # ┏━━━━━━━━━━ Isotonic Regression ━━━━━━━━━━┓
    iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
    
    # ┏━━━━━━━━━━ Fit Calibrator ━━━━━━━━━━┓
    iso.fit(val_probs, val_labels)

    # ┏━━━━━━━━━━ Predict Calibrated Values ━━━━━━━━━━┓
    val_cal = iso.predict(val_probs)

    # ┏━━━━━━━━━━ Check for Degeneracy ━━━━━━━━━━┓
    n_unique = int(np.unique(np.round(val_cal, 6)).size)
    out_range = float(val_cal.max() - val_cal.min()) if val_cal.size else 0.0
    degenerate = n_unique < min_unique_out or out_range < min_range_out
    if degenerate:
        print(f"    [calibrate] WARNING: isotonic degenerated "
              f"(unique={n_unique}, range={out_range:.3f}) — falling back to identity.")
        iso = _IdentityCalibrator()
        val_cal = iso.predict(val_probs)

    test_cal = None
    if test_probs is not None:
        test_probs = np.asarray(test_probs, dtype=float)
        test_cal = iso.predict(test_probs)

    return {"val_calibrated": val_cal,
            "test_calibrated": test_cal,
            "calibrator": iso}


__all__ = [
    "collect_risk_coverage_curve",
    "_find_best_utility_threshold",
    "calibrate_probabilities",
]
