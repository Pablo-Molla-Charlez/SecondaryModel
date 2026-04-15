"""Hyperparameter optimization (package).

Split from the previous ``_impl.py`` monolith into:

- :mod:`runner` — ``run_hpo`` / ``run_hpo_single`` / CLI ``main``
- :mod:`objectives` — per-granularity dataset loader, split preparation,
  objective-function factory
- :mod:`search_spaces` — Optuna ``suggest_*`` helpers + model builder
- :mod:`main` — thin re-export for legacy ``Utils.hpo.main`` import
"""
from Utils.hpo.runner import run_hpo, run_hpo_single, main
from Utils.hpo.objectives import (
    _create_objective,
    _prepare_splits,
    _load_dataset_for_gran,
)
from Utils.hpo.search_spaces import (
    _suggest_rf,
    _suggest_tabpfn,
    _suggest_tabicl,
    _build_model_from_params,
)