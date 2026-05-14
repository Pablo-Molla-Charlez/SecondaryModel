"""Risk/coverage curve collection and utility-optimal threshold search."""
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


def _find_best_utility_threshold(probs: np.ndarray,
                                 returns: np.ndarray,
                                 fee: float = 0.0,
                                 n_prior: int = 50,
                                 labels: np.ndarray = None,
                                 cov_min: float = 0.05,
                                 cov_star: float = 0.15,
                                 t_min: float = 1.0,
                                 m1_precision: float = None) -> dict:
    """Search for the utility-optimal threshold on the validation set.

    Runs a single-stage grid search (Stage A). If no threshold satisfies all
    constraints (coverage ≥ cov_min, mu > 0, precision ≥ baseline, t_reg ≥
    t_min), returns τ = 0.50 with constraint_satisfied = False.
    """
    probs   = np.asarray(probs,   dtype=float)
    returns = np.asarray(returns, dtype=float)
    N_total = len(probs)
    min_trades = max(50, int(cov_min * N_total))

    # Base variance for shrinkage prior
    all_net  = returns[probs >= 0.50] - fee
    base_var = float(np.nanvar(all_net, ddof=1)) if len(all_net) > 1 else 1.0
    if base_var <= 0:
        base_var = 1.0

    if labels is not None:
        labels      = np.asarray(labels).astype(int)
        sel_base    = probs >= 0.50
        n_base      = int(sel_base.sum())
        prec_base   = float(labels[sel_base].mean()) if n_base > 0 else 0.0
    else:
        prec_base = 0.0

    # Precision floor: max of M2@τ=0.5 and optional M1 precision
    prec_floor = prec_base
    if m1_precision is not None and not np.isnan(m1_precision):
        prec_floor = max(prec_floor, float(m1_precision))

    # Hybrid threshold grid: actual emitted probabilities + dense linspace
    thr_grid = np.unique(np.concatenate([probs[probs >= 0.50],
                                         np.linspace(0.50, 0.95, 200)]))
    thr_grid = thr_grid[(thr_grid >= 0.50) & (thr_grid <= 0.95)]

    best_utility = -np.inf
    best = None

    for thr in thr_grid:
        sel = probs >= thr
        n   = int(sel.sum())
        if n < min_trades:
            continue
        cov = n / N_total
        if cov < cov_min:
            continue
        net_rets = returns[sel] - fee
        mu = float(np.nanmean(net_rets))
        if mu <= 0:
            continue
        if labels is not None and float(labels[sel].mean()) < prec_floor:
            continue
        sample_var = float(np.nanvar(net_rets, ddof=1))
        shrinkage  = n_prior / (n + n_prior)
        reg_var    = (1 - shrinkage) * sample_var + shrinkage * base_var
        reg_std    = np.sqrt(max(reg_var, 1e-12))
        t_reg      = mu / reg_std * np.sqrt(n)
        if t_reg < t_min:
            continue
        cov_factor = 1.0 if cov >= cov_star else (cov / cov_star) ** 2
        utility    = t_reg * cov_factor
        if utility > best_utility:
            best_utility = utility
            best = {"threshold":            float(thr),
                    "utility":              float(utility),
                    "coverage":             cov,
                    "selected_count":       n,
                    "constraint_satisfied": True,
                    "threshold_source":     "Utility-Opt",
                    "mean_ret":             mu,
                    "t_stat":               float(t_reg),
                    "precision":            float(labels[sel].mean()) if labels is not None else float("nan")}

    # Constraint not satisfied — default to τ = 0.5
    if best is None:
        sel = probs >= 0.50
        n   = int(sel.sum())
        net = returns[sel] - fee
        mu  = float(np.nanmean(net)) if n > 0 else 0.0
        best = {"threshold":            0.50,
                "utility":              0.0,
                "coverage":             n / N_total if N_total > 0 else 0.0,
                "selected_count":       n,
                "constraint_satisfied": False,
                "threshold_source":     "Baseline",
                "mean_ret":             mu,
                "t_stat":               0.0,
                "precision":            float(labels[sel].mean()) if (labels is not None and n > 0) else 0.0}

    return best


__all__ = ["collect_risk_coverage_curve", "_find_best_utility_threshold"]