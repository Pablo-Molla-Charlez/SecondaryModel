"""Hyperparameter optimization (package).

Submodules:
- :mod:`runner` — ``run_hpo`` / ``run_hpo_single`` / CLI ``main``
- :mod:`objectives` — per-granularity dataset loader, split preparation,
  objective-function factory
- :mod:`search_spaces` — Optuna ``suggest_*`` helpers + model builder

CLI entrypoint: ``python -m Utils.hpo ...``
"""
# ┏━━━━━━━━━━ Runner ━━━━━━━━━━┓
from Utils.hpo.runner import run_hpo, run_hpo_single, main

# ┏━━━━━━━━━━ Objectives ━━━━━━━━━━┓
from Utils.hpo.objectives import (_create_objective,
                                  _prepare_splits,
                                  _load_dataset_for_gran)

# ┏━━━━━━━━━━ Search Spaces ━━━━━━━━━━┓
from Utils.hpo.search_spaces import (_suggest_rf,
                                     _suggest_tabpfn,
                                     _suggest_tabicl,
                                     _build_model_from_params)