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


# ┏━━━━━━━━━━ Threshold Cascade (v2) ━━━━━━━━━━┓
# Stages (each fires only if all previous stages found nothing):
#   A. Utility-Opt        — strict: cov ≥ cov_min, mu > 0, precision ≥ baseline, t_reg ≥ t_min
#                           objective: t_reg * quadratic coverage penalty anchored at cov_star
#   B. Precision-Coverage — relaxes t_min and cov_min but keeps mu > 0; maximizes
#                           precision × coverage (expected-hit rate) with a precision floor.
#                           Skipped entirely when no threshold in [0.5, 0.9] has mu > 0.
#   C. Risk-Fallback      — no return requirement; minimizes error rate + coverage penalty.
#                           Skipped if every threshold in [0.5, 0.9] has n < min_trades.
#   D. Baseline           — last resort: τ = 0.50, no constraints.
#   E. Baseline-Override  — post-selection guard (runs after any of A–D): if the τ=0.50
#                           baseline is ≥ baseline_override_cov_ratio × wider AND at
#                           least as precise as the chosen pick, override with the baseline.
#                           Not applied when the chosen stage is already Baseline.
def _find_best_utility_threshold(probs: np.ndarray,
                                 returns: np.ndarray,
                                 fee: float = 0.0,
                                 n_prior: int = 50,
                                 labels: np.ndarray = None,
                                 cov_min: float = 0.05,
                                 cov_star: float = 0.15,
                                 t_min: float = 1.0,
                                 baseline_override_cov_ratio: float = 10.0) -> dict:
    """Pick a validation threshold via a five-stage cascade (see module docstring).

    Parameters
    ----------
    probs, returns, labels
        Calibrated scores, per-sample net-of-fee returns, binary labels.
    fee
        Round-trip fee subtracted from returns during utility scoring.
    n_prior
        Shrinkage prior for the regularized variance used in the t-stat.
    cov_min
        Hard coverage floor for Stage A. No Stage-A pick may cover less than this.
    cov_star
        Coverage anchor at which the Stage-A penalty becomes zero (sweet spot).
    t_min
        Minimum regularized t-stat required for Stage A.
    baseline_override_cov_ratio
        Stage E triggers when the raw τ=0.50 baseline has precision ≥ op.precision
        AND coverage ≥ this ratio x op.coverage.
    """
    # ┏━━━━━━━━━━ Format Change ━━━━━━━━━━┓
    probs = np.asarray(probs, dtype=float)
    returns = np.asarray(returns, dtype=float)
    N_total = len(probs)
    min_trades = max(50, int(cov_min * N_total))

    # ┏━━━━━━━━━━ Base Variance (for shrinkage) ━━━━━━━━━━┓
    all_net = returns[probs >= 0.50] - fee
    base_var = float(np.nanvar(all_net, ddof=1)) if len(all_net) > 1 else 1.0
    if base_var <= 0:
        base_var = 1.0

    # ┏━━━━━━━━━━ Baseline (τ=0.50) Precision & Coverage ━━━━━━━━━━┓
    baseline_thr = 0.50
    if labels is not None:
        labels = np.asarray(labels).astype(int)
        sel_argmax = probs >= 0.50
        n_argmax = int(sel_argmax.sum())
        prec_argmax = float(labels[sel_argmax].mean()) if n_argmax > 0 else 0.0
        cov_argmax = n_argmax / N_total if N_total > 0 else 0.0
    else:
        prec_argmax = 0.0
        cov_argmax = 0.0
        n_argmax = 0

    # ┏━━━━━━━━━━ Helper: build op dict from a threshold ━━━━━━━━━━┓
    def _op_from_threshold(thr: float, source: str, constraint: bool, utility_val: float = 0.0):
        sel = probs >= thr
        n = int(sel.sum())
        if n == 0:
            return {"threshold":            float(thr),
                    "utility":              float(utility_val),
                    "coverage":             0.0,
                    "selected_count":       0,
                    "constraint_satisfied": constraint,
                    "threshold_source":     source,
                    "mean_ret":             0.0,
                    "t_stat":               0.0,
                    "precision":            0.0}
        net = returns[sel] - fee
        mu = float(np.nanmean(net))
        sample_var = float(np.nanvar(net, ddof=1)) if n > 1 else base_var
        shrinkage = n_prior / (n + n_prior)
        reg_var = (1 - shrinkage) * sample_var + shrinkage * base_var
        reg_std = np.sqrt(max(reg_var, 1e-12))
        t_val = mu / reg_std * np.sqrt(n) if reg_std > 0 else 0.0
        prec = float(labels[sel].mean()) if labels is not None else float("nan")
        return {"threshold":            float(thr),
                "utility":              float(utility_val),
                "coverage":             n / N_total,
                "selected_count":       n,
                "constraint_satisfied": constraint,
                "threshold_source":     source,
                "mean_ret":             mu,
                "t_stat":               float(t_val),
                "precision":            prec}

    # ┏━━━━━━━━━━ Grid Initialization ━━━━━━━━━━┓
    pos_probs = probs[probs >= 0.50]
    if pos_probs.size >= max(min_trades * 2, 20):
        grid_lo = float(np.median(pos_probs))
        grid_lo = min(max(grid_lo, 0.50), 0.85)
    else:
        grid_lo = 0.50
    thr_grid = np.linspace(grid_lo, 0.95, 200)

    best = {"threshold": 0.50, "utility": -np.inf}

    # ┏━━━━━━━━━━ Stage A — Utility-Opt (strict) ━━━━━━━━━━┓
    for thr in thr_grid:
        sel = probs >= thr
        n = int(sel.sum())
        if n < min_trades:
            continue
        cov = n / N_total
        if cov < cov_min:
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
        if t_reg < t_min:
            continue

        # ┏━━━━━━━━━━ Sweet-spot Coverage Penalty ━━━━━━━━━━┓
        # Full reward at or above cov_star; strong quadratic penalty below it.
        if cov >= cov_star:
            cov_factor = 1.0
        else:
            cov_factor = (cov / cov_star) ** 2
        utility = t_reg * cov_factor

        if utility > best["utility"]:
            best = {"threshold": float(thr),
                    "utility": float(utility),
                    "coverage": cov,
                    "selected_count": n,
                    "constraint_satisfied": True,
                    "threshold_source": "Utility-Opt",
                    "mean_ret": mu,
                    "t_stat": float(t_reg),
                    "precision": float(labels[sel].mean()) if labels is not None else float("nan")}

    # ┏━━━━━━━━━━ Stage B — Precision-Coverage Pareto ━━━━━━━━━━┓
    # Stage A already required mu > 0 AND t_reg ≥ t_min AND cov ≥ cov_min AND
    # precision ≥ baseline. Stage B relaxes t_min and cov_min but keeps mu > 0
    # — positive return must exist somewhere, just with insufficient statistical
    # power to satisfy Stage A. When the whole [0.5, 0.9] space has mu ≤ 0,
    # Stage B finds nothing and we fall through to Stage C (Risk-Fallback).
    if best["utility"] == -np.inf and labels is not None:
        slack = 0.02
        prec_floor = max(0.0, prec_argmax - slack)
        floor_B = max(min_trades, int(0.10 * N_total))
        best_B = {"score": -np.inf}
        for thr in np.linspace(0.50, 0.90, 200):
            sel = probs >= thr
            n = int(sel.sum())
            if n < floor_B:
                continue
            if float(np.nanmean(returns[sel] - fee)) <= 0:
                continue
            prec = float(labels[sel].mean())
            if prec < prec_floor:
                continue
            cov = n / N_total
            score = prec * cov  # expected-hit rate
            if score > best_B["score"]:
                best_B = {"score": score, "thr": float(thr)}
        if best_B["score"] != -np.inf:
            op = _op_from_threshold(best_B["thr"],
                                    source="Precision-Coverage",
                                    constraint=False,
                                    utility_val=0.0)
            best = {**op, "utility": 0.0}

    # ┏━━━━━━━━━━ Stage C — Risk-Fallback ━━━━━━━━━━┓
    if best["utility"] == -np.inf:
        cov_penalty = 0.10
        sel_base = probs >= 0.50
        cov_base = max(int(sel_base.sum()) / N_total, 1e-9)

        best_score = np.inf
        best_C = None
        for thr in np.linspace(0.50, 0.90, 200):
            sel = probs >= thr
            n = int(sel.sum())
            if n < min_trades:
                continue
            if float(np.nanmean(returns[sel] - fee)) <= 0:
                continue
            n_err = int((returns[sel] < fee).sum())
            risk = n_err / n
            cov = n / N_total
            score = risk + cov_penalty * (1 - cov / cov_base)
            if score < best_score:
                best_score = score
                best_C = float(thr)
        if best_C is not None:
            op = _op_from_threshold(best_C,
                                    source="Risk-Fallback",
                                    constraint=False,
                                    utility_val=0.0)
            best = {**op, "utility": 0.0}

    # ┏━━━━━━━━━━ Stage D — Baseline (data-anchored) ━━━━━━━━━━┓
    if best["utility"] == -np.inf:
        op = _op_from_threshold(baseline_thr,
                                source="Baseline",
                                constraint=False,
                                utility_val=0.0)
        best = {**op, "utility": 0.0}

    # ┏━━━━━━━━━━ Stage E — Baseline-Override Guard ━━━━━━━━━━┓
    # If the raw τ=0.50 baseline has precision ≥ op.precision AND coverage much
    # wider than op.coverage, replace op with the baseline. Prevents narrow
    # fluke-tail picks from beating a solid wide edge.
    if labels is not None and n_argmax > 0 and best["threshold_source"] != "Baseline":
        op_cov = best.get("coverage", 0.0)
        op_prec = best.get("precision", 0.0)
        if op_prec != op_prec:  # NaN guard
            op_prec = 0.0
        triggers = (op_cov > 0
                    and cov_argmax >= baseline_override_cov_ratio * op_cov
                    and prec_argmax >= op_prec)
        if triggers:
            op = _op_from_threshold(baseline_thr,
                                    source="Baseline-Override",
                                    constraint=False,
                                    utility_val=0.0)
            best = {**op, "utility": 0.0}

    return best


__all__ = ["collect_risk_coverage_curve", "_find_best_utility_threshold"]