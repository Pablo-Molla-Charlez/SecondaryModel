"""Kumo Tabular classifier — BaseClassifier-compliant wrapper.

Kumo Tabular (NVIDIA, released 2026-09-29) is an in-context tabular
foundation model pretrained only on synthetic tables sampled from
structural causal models — the same family as TabPFN/TabICL, but ranked
#1 on TabArena (ELO 1950) at release, ahead of tuned GBTs, AutoGluon, and
other tabular foundation models per NVIDIA's own benchmark. Unlike
TabPFN-3.5/Mitra, its weights are openly licensed (OpenMDW-1.1, permits
commercial use) -- cleaner for a thesis than the non-commercial foundation
models already wired in here.

Reference: https://huggingface.co/blog/nvidia/kumo-tabular
Code: https://github.com/NVIDIA/structured-data-models

UNVERIFIED -- read, not run. This wrapper was written by reading NVIDIA's
own AutoGluon-integration glue in that repo's benchmark/tabular/model.py
(SDMKumoTabularModel._fit / ._predict_proba), not by installing and
exercising the package (no local GPU here). The fit/predict call shape
(sdm.TableTensor, model.fit(x=..., y=..., recipe=...), model.predict(...)
returning a structured output reconstructed into a (n, n_classes) array)
is taken directly from that source, not guessed -- but it has never
actually run. Sanity-check before trusting any number out of this:
predict_proba(X).sum(axis=1) should be ~1 for every row, and the kumo
entry in slurm/m2-compare-trial.sh's version/import check should pass
cleanly before the fit itself is trusted.
"""
from __future__ import annotations

import pickle
import warnings
from typing import Optional, Union

import numpy as np
import pandas as pd

from Utils.classifier._classifier import BaseClassifier

# Soft row-count guard, mirroring _TABPFN_MAX_ROWS / _TABICL_MAX_ROWS in
# factory.py. M2's actual training cohort (fit20, ~2k-6k rows/side) is far
# below this -- it exists for safety, not because we expect to hit it.
_KUMO_MAX_ROWS = 50_000


def _resolve_device(prefer: str = "cuda") -> str:
    import torch
    if prefer == "cuda" and torch.cuda.is_available():
        return "cuda"
    return "cpu"


class KumoTabularClassifier(BaseClassifier):
    """Sklearn-compatible wrapper around ``sdm.models.KumoTabular``.

    Parameters
    ----------
    size : str
        "small" (~28M params), "medium" (~62M), or "large" (~215M, the
        upstream default). Smaller may generalise better with limited data
        and is cheaper to run -- M2's fit20 cohort is small enough that
        "small"/"medium" are worth comparing against "large", not just
        assumed worse.
    num_estimators : int | None
        Ensemble size for the in-context predictor. None keeps Kumo's own
        default (do not assume a specific value without checking the
        installed version's default).
    device : str | None
        None auto-resolves to CUDA if available, else CPU (Kumo is
        GPU-native but the upstream README does not state it refuses CPU).
    """

    def __init__(self,
                 size:           str          = "large",
                 num_estimators: Optional[int] = None,
                 random_state:   int          = 42,
                 device:         Optional[str] = None) -> None:
        super().__init__(random_state)
        self.size           = size
        self.num_estimators = num_estimators
        self.random_state   = random_state
        self.device         = device if device is not None else _resolve_device()
        self._model         = None
        self._stypes        = None
        self._recipe        = None
        self._feature_cols  = None
        self._target_name   = "__m2_target__"
        self.classes_       = None
        self.n_features_in_ = None

    # ┏━━━━━━━━━━ Fit ━━━━━━━━━━┓
    def fit(self, X_train: Union[np.ndarray, pd.DataFrame], y_train: np.ndarray):
        import torch
        import sdm

        X = X_train if isinstance(X_train, pd.DataFrame) else pd.DataFrame(
            np.asarray(X_train, dtype=np.float32))
        # sdm.infer_stypes requires string column names -- pd.DataFrame(ndarray)
        # without columns= gives an integer RangeIndex, which fails its
        # isinstance(name, str) assertion. Caught by actually running this
        # (2026-10-26): AssertionError inside sdm.stype.infer_stypes.
        X.columns = [f"f{i}" for i in range(X.shape[1])]
        y = np.asarray(y_train)

        if len(X) > _KUMO_MAX_ROWS:
            warnings.warn(f"KumoTabular: training set has {len(X):,} rows "
                          f"(soft guard {_KUMO_MAX_ROWS:,}). Randomly sub-sampling.")
            rng = np.random.default_rng(self.random_state)
            idx = rng.choice(len(X), _KUMO_MAX_ROWS, replace=False)
            X, y = X.iloc[idx].reset_index(drop=True), y[idx]

        self._feature_cols  = list(X.columns)
        self.n_features_in_ = len(self._feature_cols)
        self.classes_       = np.unique(y)

        device = torch.device(self.device)
        self._model = sdm.models.KumoTabular(size=self.size, device=device)

        self._stypes = sdm.infer_stypes(X)
        x_context = sdm.TableTensor.from_pandas(df=X, stypes=self._stypes, device=device)

        y_df = pd.DataFrame({self._target_name: y}).astype("category")
        y_context = sdm.TableTensor.from_pandas(
            df=y_df, stypes={self._target_name: "categorical"}, device=device)

        self._recipe = self._model.default_recipe()

        generator = torch.Generator(device).manual_seed(self.random_state)
        self._model.fit(x=x_context,
                        y=y_context,
                        recipe=self._recipe,
                        num_estimators=self.num_estimators,
                        generator=generator)
        return self

    # ┏━━━━━━━━━━ Predict Probabilities ━━━━━━━━━━┓
    def predict_proba(self, X_test: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        import torch
        import sdm

        if self._model is None:
            raise AttributeError("The model has not been fitted yet.")

        X = X_test if isinstance(X_test, pd.DataFrame) else pd.DataFrame(
            np.asarray(X_test, dtype=np.float32), columns=self._feature_cols)

        device = torch.device(self.device)
        x_query = sdm.TableTensor.from_pandas(df=X, stypes=self._stypes, device=device)

        with torch.inference_mode():
            out = self._model.predict(x_query)

        # Correction (2026-10-26, verified by actually running this against
        # sdm==0.2.1): `out.numerical_columns` -- what NVIDIA's own
        # benchmark/tabular/model.py AutoGluon glue reads -- does not exist
        # on this version's TableTensor. `out.to_pandas()` does, and gives a
        # plain (n, n_classes) frame with the class labels as column names
        # (category codes from the y_df.astype("category") in fit, e.g.
        # "0"/"1"), already summing to 1 per row. Index explicitly by
        # self.classes_ rather than assuming column order matches it.
        proba_df = out.to_pandas()
        probabilities = np.column_stack(
            [proba_df[str(cls)].to_numpy(dtype=np.float32) for cls in self.classes_])

        # Defensive renormalisation: we don't have NVIDIA's
        # _convert_proba_to_unified_form helper, so guard against rows that
        # don't already sum to ~1 rather than silently returning something
        # that isn't a valid probability distribution.
        row_sums = probabilities.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        return probabilities / row_sums

    # ┏━━━━━━━━━━ Predict ━━━━━━━━━━┓
    def predict(self, X_test: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        proba = self.predict_proba(X_test)
        return self.classes_[np.argmax(proba, axis=1)]

    # ┏━━━━━━━━━━ Feature Importance (uniform fallback -- Kumo has no native one) ━━━━━━━━━━┓
    @property
    def feature_importances_(self):
        n_feat = self.n_features_in_ or 1
        return np.ones(n_feat) / n_feat

    # ┏━━━━━━━━━━ Get Parameters ━━━━━━━━━━┓
    def get_params(self, deep: bool = True) -> dict:
        return {"size":           self.size,
                "num_estimators": self.num_estimators,
                "random_state":   self.random_state,
                "device":         self.device}

    # ┏━━━━━━━━━━ Save / Load ━━━━━━━━━━┓
    # sdm's own native serialization API is not confirmed from reading the
    # source alone -- pickling the wrapper's state is the safe fallback
    # factory.py already uses for any classifier whose save_model fails.
    def save_model(self, model_path: str) -> None:
        if self._model is None:
            raise AttributeError("The model has not been fitted yet.")
        with open(f"{model_path}.pkl", "wb") as f:
            pickle.dump({"model":        self._model,
                        "stypes":       self._stypes,
                        "recipe":       self._recipe,
                        "feature_cols": self._feature_cols,
                        "classes_":     self.classes_}, f)

    def load_model(self, model_path: str) -> None:
        with open(f"{model_path}.pkl", "rb") as f:
            state = pickle.load(f)
        self._model         = state["model"]
        self._stypes        = state["stypes"]
        self._recipe        = state["recipe"]
        self._feature_cols  = state["feature_cols"]
        self.classes_       = state["classes_"]
