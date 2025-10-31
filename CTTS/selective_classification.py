import json
import numpy as np
from scipy.stats import beta
import matplotlib.pyplot as plt
from typing import Any, Dict, Iterable, Optional

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


def coverage_at_risk(y_true = None,
                     y_score = None,
                     *,  
                     max_risk,
                     thresholds = None,
                     empty_selection_value = np.nan,
                     min_coverage: float = 0.0,
                     min_selected: int = 0,
                     curve: Optional[Dict[str, np.ndarray]] = None):

    """Return the best threshold whose risk is ≤ max_risk.
    The max_risk, is defined by the user, in the config.yaml file.

    Notes: 
    1. Coverage is the fraction of samples whose score ≥ threshold.
    2. Risk is the error rate among the selected samples when predicting 1 for score ≥ threshold.    
    """
    
    # ┏━━━━━━━━━━ Risk Interval ━━━━━━━━━━┓
    if not 0.0 <= max_risk <= 1.0:
        raise ValueError("max_risk must be within [0, 1]")

    # ┏━━━━━━━━━━ Recompute Curve if not Provided ━━━━━━━━━━┓
    if curve is None:
        if y_true is None or y_score is None:
            raise ValueError("Provide either `curve` or y_true/y_score arrays")
        curve = collect_risk_coverage_curve(y_true=y_true,
                                            y_score=y_score,
                                            thresholds=thresholds,
                                            empty_selection_value=empty_selection_value)

    elif thresholds is not None:
        raise ValueError("When `curve` is supplied, `thresholds` must be omitted")

    # ┏━━━━━━━━━━ Extract Information Curve ━━━━━━━━━━┓
    risks = curve["risk"]
    coverages = curve["coverage"]
    selected_counts = curve["selected_count"]
    thresholds_arr = curve["thresholds"]

    # ┏━━━━━━━━━━ Mask to filter Conditions ━━━━━━━━━━┓
    valid_mask = (~np.isnan(risks) 
                & (risks <= max_risk)
                & (coverages >= min_coverage) 
                & (selected_counts >= min_selected))

    # ┏━━━━━━━━━━ All Conditions Satisfied [Minimum Selected & Finite Risk & Min Coverage & Max Risk] ━━━━━━━━━━┓
    if np.any(valid_mask):
        best_idx = int(np.argmax(coverages[valid_mask])) # Among valid thresholds, find the one with maximum coverage
        valid_indices = np.flatnonzero(valid_mask)       # Get the actual indices of valid entries
        chosen_idx = valid_indices[best_idx]             # Map back to original indices

        return {"threshold":            float(thresholds_arr[chosen_idx]),
                "coverage":             float(coverages[chosen_idx]),
                "risk":                 float(risks[chosen_idx]),
                "selected_count":       int(selected_counts[chosen_idx]),
                "constraint_satisfied": True}

    # ┏━━━━━━━━━━ Minimum Conditions Satisfied [Only Minimum Selected & Finite Risk] ━━━━━━━━━━┓
    # If no thresholds satisfy the risk budget + coverage constraints, fall back to the
    # non-empty point with minimal risk and mark it as an infeasible solution.
    fallback_mask = (~np.isnan(risks)) & (selected_counts >= max(min_selected, 1))
    if np.any(fallback_mask):
        fallback_indices = np.flatnonzero(fallback_mask)
        min_risk_idx = fallback_indices[int(np.argmin(risks[fallback_mask]))]
        
        return {"threshold":            float(thresholds_arr[min_risk_idx]),
                "coverage":             float(coverages[min_risk_idx]),
                "risk":                 float(risks[min_risk_idx]),
                "selected_count":       int(selected_counts[min_risk_idx]),
                "constraint_satisfied": False}

    # ┏━━━━━━━━━━ No Condition Satisfied ━━━━━━━━━━┓
    return {"threshold": float(thresholds_arr.max()) if thresholds_arr.size else np.nan,
            "coverage": 0.0,
            "risk": np.nan,
            "selected_count": 0,
            "constraint_satisfied": False}


def area_under_risk_coverage(y_true = None,
                             y_score = None,
                             thresholds = None,
                             empty_selection_value = np.nan,
                             curve: Optional[Dict[str, np.ndarray]] = None):

    """Compute AURC either from provided curve or raw inputs."""
    
    # ┏━━━━━━━━━━ Recompute Curve if not Provided ━━━━━━━━━━┓
    if curve is None:
        if y_true is None or y_score is None:
            raise ValueError("Provide either `curve` or y_true/y_score arrays")
        curve = collect_risk_coverage_curve(y_true = y_true,
                                            y_score = y_score,
                                            thresholds = thresholds,
                                            empty_selection_value = empty_selection_value)

    # ┏━━━━━━━━━━ Extract Information Curve ━━━━━━━━━━┓
    coverage = np.asarray(curve["coverage"])
    risk = np.asarray(curve["risk"])

    # ┏━━━━━━━━━━ Mask to filter Conditions ━━━━━━━━━━┓
    valid_mask = (~np.isnan(coverage)) & (~np.isnan(risk))
    coverage = coverage[valid_mask]
    risk = risk[valid_mask]

    # ┏━━━━━━━━━━ Minimum Requirement for AUR&C Curve ━━━━━━━━━━┓
    # AURC is defined as the integral of risk with respect to coverage (coverage is on the x‑axis).
    # For a correct trapezoidal integration (np.trapezoid(y, x)), the x values should be monotonic; 
    # Sorting ensures coverage is nondecreasing.
    if coverage.size < 2:
        return 0.0

    # ┏━━━━━━━━━━ Sorting Coverages & Risk for computing AUR&C Curve ━━━━━━━━━━┓
    order = np.argsort(coverage)
    coverage_sorted = coverage[order]
    risk_sorted = risk[order]

    # ┏━━━━━━━━━━ Area of Risk & Coverage Curve ━━━━━━━━━━┓
    area = float(np.trapezoid(risk_sorted, coverage_sorted))
    return area


def plot_coverage_risk_curve(y_true = None,
                             y_score = None,
                             *,
                             curve: Optional[Dict[str, np.ndarray]] = None,
                             thresholds = None,
                             empty_selection_value = 0.0,
                             label: Optional[str] = None,
                             ax: Optional[plt.Axes] = None,
                             save_path: Optional[str] = None,
                             show: bool = True):

    """Minimal coverage-risk plot."""

    # ┏━━━━━━━━━━ Recompute Curve if not Provided ━━━━━━━━━━┓
    if curve is None:
        if y_true is None or y_score is None:
            raise ValueError("Provide either `curve` or y_true/y_score arrays")
        
        # ┏━━━━━━━━━━ Computing Curve ━━━━━━━━━━┓
        curve = collect_risk_coverage_curve(y_true = y_true,
                                            y_score = y_score,
                                            thresholds = thresholds,
                                            empty_selection_value = empty_selection_value)

    # ┏━━━━━━━━━━ Extract Coverage & Risk from Curve ━━━━━━━━━━┓
    coverage = np.asarray(curve["coverage"], dtype=float)
    risk = np.asarray(curve["risk"], dtype=float)

    # ┏━━━━━━━━━━ Sort Coverage then Risk ━━━━━━━━━━┓
    order = np.argsort(coverage)
    coverage_sorted = coverage[order]
    risk_sorted = risk[order]

    # ┏━━━━━━━━━━ Create Skeleton Plot ━━━━━━━━━━┓
    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
        created_fig = True
    else:
        fig = ax.figure

    # ┏━━━━━━━━━━ Legend of Plot ━━━━━━━━━━┓
    ax.plot(coverage_sorted, risk_sorted, marker="o", label=label)
    ax.set_xlabel("Coverage")
    ax.set_ylabel("Risk (Error Rate)")
    ax.set_title("Coverage-Risk Curve")
    ax.grid(True, which="both", linestyle="--", alpha=0.5)

    handles, labels = ax.get_legend_handles_labels()
    filtered = [(h, l) for h, l in zip(handles, labels) if l]
    if filtered:
        ax.legend(*zip(*filtered))

    fig.tight_layout()
    
    # ┏━━━━━━━━━━ Save Plot ━━━━━━━━━━┓
    if save_path:
        fig.savefig(save_path)
    if created_fig:
        if show:
            plt.show()
        else:
            plt.close(fig)


def _to_jsonable(value: Any) -> Any:
    "Recursive helper to ensure metrics can be written to JSON without manual conversions."
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, dict):
        return {k: _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    return value


def save_metrics(metrics: Dict[str, Any], save_path: str, convert_numpy: bool = True) -> None:
    """Persist metrics to JSON, optionally converting numpy types."""
    
    # ┏━━━━━━━━━━ From Numpy to JSON ━━━━━━━━━━┓
    if convert_numpy:
        payload = _to_jsonable(metrics)
    else:
        payload = metrics

    with open(save_path, "w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=4)