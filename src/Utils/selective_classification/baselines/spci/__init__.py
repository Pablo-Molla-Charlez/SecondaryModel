"""M1+SPCI baseline — wrapper plus the vendored upstream SPCI implementation.

``scpi_selective.py`` is this project's gate; the remaining modules (``SPCI_class.py``,
``utils_SPCI.py``, ``data.py``, ``utils_EnbPI.py``, ``PI_class_EnbPI.py``) are the
upstream `hamrel-cxu/SPCI-code` source, used with the author's permission and kept
here unmodified. They import each other by bare module name, so ``scpi_selective.py`` puts this
directory on ``sys.path`` before importing them.

Extra third-party requirements (see ``requirements.txt``, mirrored into the repo-level
``requirements.txt``): ``sklearn_quantile``, ``skranger``, ``statsmodels``.

Citation: Xu & Xie, "Sequential Predictive Conformal Inference for Time Series",
ICML 2023, PMLR 202:38707-38727.
"""
