"""Selective-classification utilities (package).

Split from the previous ``_impl.py`` monolith into:

- :mod:`thresholds` — risk/coverage curve + utility-optimal threshold search
- :mod:`calibration` — isotonic calibrator + identity fallback
"""
from Utils.selective_classification.thresholds import (
    collect_risk_coverage_curve,
    _find_best_utility_threshold,
)
from Utils.selective_classification.calibration import (
    _IdentityCalibrator,
    calibrate_probabilities,
)

__all__ = [
    "collect_risk_coverage_curve",
    "_find_best_utility_threshold",
    "_IdentityCalibrator",
    "calibrate_probabilities",
]
