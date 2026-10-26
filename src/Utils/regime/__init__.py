"""Regime detection via pooled (global) Sparse Jump Models.

This is a fitting-only subset of the upstream regime package: enough to fit one
global JM per granularity and emit per-asset causal regime labels, which the
fold-composition analysis in ``Utils.analysis.analysis_m2`` consumes. The
per-asset entrypoint, the financial (Sharpe) tuner and the breakdown reporting
from upstream are deliberately absent — nothing here needs them.

Modules:
    paths                    — shared filesystem roots (src/, Data_MLA/Raw, Output/Jump_Models)
    data_access              — raw OHLCV(A) loading
    config                   — config/path resolution + train/val/test windowing
    features                 — regime feature builders
    model                    — RegimeJumpModel wrapper + typed fit results
    stat_tuning              — modified-BIC scoring primitives
    bic_selection            — BIC grid-search helpers (extracted from upstream train_jump_model)
    storage                  — persistence for fitted models and regime labels
    train_global_jump_model  — pooled global fit entrypoint

Public names are re-exported lazily (PEP 562) so a light import does not eagerly
pull ``jumpmodels`` (imported by ``model``).
"""

from __future__ import annotations

import importlib

_EXPORTS = {
    # paths
    "SRC_ROOT": "paths",
    "RAW_ROOT": "paths",
    "JUMP_MODELS_ROOT": "paths",
    "DEFAULT_CONFIG_PATH": "paths",
    # raw data access
    "load_ohlcv": "data_access",
    "clear_ohlcv_cache": "data_access",
    # config / split windowing
    "load_regime_config": "config",
    "load_regime_model_defaults": "config",
    "load_regime_scope": "config",
    "resolve_raw_data_dir": "config",
    "resolve_output_root": "config",
    "load_split_config": "config",
    "clip_ohlcv_to_split_window": "config",
    "split_frame_by_config": "config",
    # features
    "build_regime_features": "features",
    "compute_log_returns": "features",
    "BARS_PER_DAY": "features",
    "FEATURE_SETS": "features",
    # model
    "RegimeJumpModel": "model",
    "RegimeFitResult": "model",
    "FitArtifacts": "model",
    "JM_VARIANTS": "model",
    "is_sparse_variant": "model",
    "variant_is_continuous": "model",
    # statistical / BIC tuning
    "DEFAULT_K_GRID": "stat_tuning",
    "DEFAULT_LAMBDA_GRID": "stat_tuning",
    "BicScore": "stat_tuning",
    "best_bic_candidate": "stat_tuning",
    "count_jumps": "stat_tuning",
    "default_kappa_grid": "stat_tuning",
    "select_bic_hyperparameters": "stat_tuning",
    # storage
    "global_fit_dir": "storage",
    "save_global_fit": "storage",
    "load_global_labels": "storage",
    "load_global_model": "storage",
    # pooled global fit entrypoint
    "fit_global_jump_model": "train_global_jump_model",
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str):
    """Resolve a public name to its owning submodule on first access (PEP 562)."""
    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = importlib.import_module(f"{__name__}.{module_name}")
    value = getattr(module, name)
    globals()[name] = value  # cache so subsequent lookups skip __getattr__
    return value


def __dir__():
    return __all__
