"""Utils package — public API re-exports.

This file pins the hot-path public API against the new package locations.
External callers can import every ``hot symbol`` directly from ``Utils``
without knowing which submodule it lives in.
"""

# ┏━━━━━━━━━━ Unpickle-compat aliases ━━━━━━━━━━┓
import sys as _sys
from Utils import data as _data
from Utils import classifier as _classifier
from Utils import feature_selection as _feature_selection
from Utils import backtest as _backtest
from Utils import ocp as _ocp
from Utils import hpo as _hpo
from Utils import edge as _edge

# ┏━━━━━━━━━━ Legacy module aliases ━━━━━━━━━━┓
_sys.modules.setdefault("Utils.data_preprocessing", _data)
_sys.modules.setdefault("Utils.models",             _classifier)
_sys.modules.setdefault("Utils.features",           _feature_selection)
_sys.modules.setdefault("Utils.comparison",         _backtest)
_sys.modules.setdefault("Utils.saocp",              _ocp)
_sys.modules.setdefault("Utils.ocp_analysis",       _ocp)
_sys.modules.setdefault("Utils.ocp_theory",         _ocp)
_sys.modules.setdefault("Utils.HPO",                _hpo)

# ┏━━━━━━━━━━ Cleanup ━━━━━━━━━━┓
del _sys, _data, _classifier, _feature_selection, _backtest, _ocp, _hpo, _edge

# ┏━━━━━━━━━━ Data hot symbols ━━━━━━━━━━┓
from Utils.data import (split_by_global_time,
                        ENG_FEATURE_NAMES,
                        ENG_FEATURE_GROUPS,
                        GRAN_SEQ_LEN,
                        MultiGranDataset,
                        load_dataset_from_config,
                        resolve_feature_names,
                        prepare_multi_asset_dataset,
                        prepare_multi_gran_dataset)

# ┏━━━━━━━━━━ Utils hot symbols ━━━━━━━━━━┓
from Utils.utils import (_load_multi_cache,
                         model_label,
                         _load_config,
                         _infer_direction,
                         _safe_json,
                         seed_everything,
                         m1_output_bucket,
                         m1_display_label,
                         _load_best_params,
                         HPO_SUPPORTED_M2)

# ┏━━━━━━━━━━ Selective-classification hot symbols ━━━━━━━━━━┓
from Utils.selective_classification import (_find_best_utility_threshold,
                                            calibrate_probabilities,
                                            collect_risk_coverage_curve)

# ┏━━━━━━━━━━ Model hot symbols ━━━━━━━━━━┓
from Utils.classifier import (MODELS_NO_SCALING,
                              MODEL_CHOICES,
                              _build_tree_model)

# ┏━━━━━━━━━━ Feature-selection / plotting hot symbols ━━━━━━━━━━┓
from Utils.feature_selection import (plot_confusion_matrix,
                                     run_feature_selection)

# ┏━━━━━━━━━━ Time-series CV (CPCV default is mode="datetime") ━━━━━━━━━━┓
from Utils.ts_cross_validation import (BaseTimeSeriesCV,
                                       CombinatorialPurgedCV,
                                       PurgedEmbargoTimeSeriesCV,
                                       SklearnTimeSeriesCV,
                                       compute_embargo_splits)

# ┏━━━━━━━━━━ Classifier package (wrappers) ━━━━━━━━━━┓
from Utils.classifier import BaseClassifier

__all__ = [
    # ┏━━━━━━━━━━ Data ━━━━━━━━━━┓
    "split_by_global_time",
    "ENG_FEATURE_NAMES",
    "ENG_FEATURE_GROUPS",
    "GRAN_SEQ_LEN",
    "MultiGranDataset",
    "load_dataset_from_config",
    "resolve_feature_names",
    "prepare_multi_asset_dataset",
    "prepare_multi_gran_dataset",
    
    # ┏━━━━━━━━━━ Utils ━━━━━━━━━━┓
    "_load_multi_cache",
    "model_label",
    "_load_config",
    "_infer_direction",
    "_safe_json",
    "seed_everything",
    "m1_output_bucket",
    "m1_display_label",
    "_load_best_params",
    "HPO_SUPPORTED_M2",
    
    # ┏━━━━━━━━━━ Selective classification ━━━━━━━━━━┓
    "_find_best_utility_threshold",
    "calibrate_probabilities",
    "collect_risk_coverage_curve",
    
    # ┏━━━━━━━━━━ Models ━━━━━━━━━━┓
    "MODELS_NO_SCALING",
    "MODEL_CHOICES",
    "_build_tree_model",
    
    # ┏━━━━━━━━━━ Feature selection ━━━━━━━━━━┓
    "plot_confusion_matrix",
    "run_feature_selection",
    
    # ┏━━━━━━━━━━ CV ━━━━━━━━━━┓
    "BaseTimeSeriesCV",
    "CombinatorialPurgedCV",
    "PurgedEmbargoTimeSeriesCV",
    "SklearnTimeSeriesCV",
    "compute_embargo_splits",
    
    # ┏━━━━━━━━━━ Classifiers ━━━━━━━━━━┓
    "BaseClassifier",
]
