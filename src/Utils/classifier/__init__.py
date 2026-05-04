"""M2 classifier package — BaseClassifier + concrete wrappers + factory.

Concrete wrappers are imported defensively: any of them may bring in heavy
optional dependencies (skops, tabpfn, tabicl, autogluon, xgboost, ...). If one
of those is missing, the corresponding symbol is set to ``None`` so callers
can import the package and only fail when they actually use the missing
class.
"""
from Utils.classifier._classifier import BaseClassifier

# ┏━━━━━━━━━━ Random Forest Classifier ━━━━━━━━━━┓
try:
    from Utils.classifier.random_forest_classifier import RFClassifier
except Exception:
    RFClassifier = None

# ┏━━━━━━━━━━ TabPFN Classifier ━━━━━━━━━━┓
try:
    from Utils.classifier.tabpfn_classifier import TabPFN
except Exception:
    TabPFN = None

# ┏━━━━━━━━━━ TabICL Classifier ━━━━━━━━━━┓
try:
    from Utils.classifier.tabicl_classifier import TabICL
except Exception:
    TabICL = None

# ┏━━━━━━━━━━ TabM Classifier ━━━━━━━━━━┓
try:
    from Utils.classifier.tabm_classifier import TabMClassifier
except Exception:
    TabMClassifier = None

# ┏━━━━━━━━━━ CTTS Classifier ━━━━━━━━━━┓
try:
    from Utils.classifier.ctts_classifier import CTTSClassifier
except Exception:
    CTTSClassifier = None

# ┏━━━━━━━━━━ TabPFN Fine-Tuned Classifier ━━━━━━━━━━┓
try:
    from Utils.classifier.tabpfn_finetuned_classifier import TabPFNFineTuned
except Exception:
    TabPFNFineTuned = None

# ┏━━━━━━━━━━ AutoGluon Classifier ━━━━━━━━━━┓
try:
    from Utils.classifier.autogluon_classifier import AutoGluon, AutogluonClassifier
except Exception:
    AutoGluon = None
    AutogluonClassifier = None

# ┏━━━━━━━━━━ Factory ━━━━━━━━━━┓
from Utils.classifier.factory import (_build_tree_model,
                                      _save_final_model,
                                      MODEL_CHOICES,
                                      MODELS_NO_SCALING,
                                      _TABPFN_MAX_ROWS,
                                      _AG_TIME_LIMIT,
                                      _AG_PRESETS)

__all__ = [
    "BaseClassifier",
    "RFClassifier",
    "TabPFN",
    "TabICL",
    "TabMClassifier",
    "CTTSClassifier",
    "TabPFNFineTuned",
    "AutoGluon",
    "AutogluonClassifier",
    "_build_tree_model",
    "_save_final_model",
    "MODEL_CHOICES",
    "MODELS_NO_SCALING",
    "_TABPFN_MAX_ROWS",
    "_AG_TIME_LIMIT",
    "_AG_PRESETS",
]
