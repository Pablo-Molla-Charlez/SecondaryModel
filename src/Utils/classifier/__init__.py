"""M2 classifier package — BaseClassifier + concrete wrappers + factory.

Concrete wrappers are imported defensively: any of them may bring in heavy
optional dependencies (skops, tabpfn, tabicl, autogluon, xgboost, ...). If one
of those is missing, the corresponding symbol is set to ``None`` so callers
can import the package and only fail when they actually use the missing
class.
"""
from Utils.classifier._classifier import BaseClassifier

try:
    from Utils.classifier.random_forest_classifier import RFClassifier
except Exception:
    RFClassifier = None  # type: ignore

try:
    from Utils.classifier.tabpfn_classifier import TabPFN
except Exception:
    TabPFN = None  # type: ignore

try:
    from Utils.classifier.tabicl_classifier import TabICL
except Exception:
    TabICL = None  # type: ignore

# Optional — only importable when the fine-tuning extra is installed.
try:
    from Utils.classifier.tabpfn_finetuned_classifier import TabPFNFineTuned
except Exception:  # pragma: no cover — degraded gracefully when TabPFN-FT deps are absent
    TabPFNFineTuned = None  # type: ignore

try:
    from Utils.classifier.autogluon_classifier import AutoGluon, AutogluonClassifier
except Exception:  # pragma: no cover
    AutoGluon = None  # type: ignore
    AutogluonClassifier = None  # type: ignore

# Factory — never fails on import (xgboost + sklearn are hard deps).
from Utils.classifier.factory import (
    _build_tree_model,
    MODEL_CHOICES,
    MODELS_NO_SCALING,
    _TABPFN_MAX_ROWS,
    _AG_TIME_LIMIT,
    _AG_PRESETS,
)

__all__ = [
    "BaseClassifier",
    "RFClassifier",
    "TabPFN",
    "TabICL",
    "TabPFNFineTuned",
    "AutoGluon",
    "AutogluonClassifier",
    "_build_tree_model",
    "MODEL_CHOICES",
    "MODELS_NO_SCALING",
    "_TABPFN_MAX_ROWS",
    "_AG_TIME_LIMIT",
    "_AG_PRESETS",
]
