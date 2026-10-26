"""Persistence for fitted Jump Models and their regime labels (Stage 0).

Layout under Output/Jump_Models/:

  Output/Jump_Models/<asset>/<granularity>/<variant>_K<k>_lam<λ>/
    ├── model.pkl          # the fitted RegimeJumpModel (scaler + jumpmodels obj)
    ├── labels.csv         # date, regime, split, p_regime_0..K-1 (causal)
    ├── labels_insample.csv# date, regime, split, p_regime_* (train in-sample)
    └── meta.json          # variant, K, λ, feature_set, half-lives, n_shifts, dates

Stage 2 reads `labels.csv` (the CAUSAL labels) by asset + granularity to pick
context windows whose regime matches a target. The in-sample labels are kept
separately and clearly marked exploration-only so they never leak into M2.
"""

from __future__ import annotations

import json
import pickle  # nosec B403 — only used on first-party fits this process wrote
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd

from Utils.regime.model import is_sparse_variant
from Utils.regime.paths import JUMP_MODELS_ROOT

if TYPE_CHECKING:
    from Utils.regime.model import RegimeJumpModel


def _num_slug(value: float) -> str:
    """Compact filesystem token for numeric hyperparameters."""
    return f"{float(value):g}".replace("-", "m").replace(".", "p")


def fit_dir(asset: str, granularity: str, variant: str,
            n_components: int, jump_penalty: float,
            root: Path | None = None,
            max_feats: float | None = None) -> Path:
    """Canonical directory for one (asset, gran, variant, K, λ) fit."""
    root = Path(root) if root is not None else JUMP_MODELS_ROOT
    # λ formatted compactly and filesystem-safely (e.g. 50.0 → 50, 22.5 → 22p5)
    lam = _num_slug(jump_penalty)
    slug = f"{variant}_K{n_components}_lam{lam}"
    return root / asset / granularity / slug


def global_fit_dir(granularity: str,
                   variant: str,
                   n_components: int,
                   jump_penalty: float,
                   root: Path | None = None,
                   max_feats: float | None = None) -> Path:
    """Canonical directory for one pooled global JM fit."""
    root = Path(root) if root is not None else JUMP_MODELS_ROOT
    lam = _num_slug(jump_penalty)
    slug = f"{variant}_K{n_components}_lam{lam}"
    return root / "_global" / granularity / slug


def save_fit(asset: str,
             granularity: str,
             model: "RegimeJumpModel",    # fitted
             causal_labels: pd.Series,
             causal_proba: pd.DataFrame,
             insample_labels: pd.Series,
             insample_proba: pd.DataFrame,
             meta: dict[str, Any],
             causal_split: pd.Series | None = None,
             insample_split: pd.Series | None = None,
             root: Path | None = None) -> Path:
    """Persist model + causal/in-sample labels + metadata. Returns the dir."""
    d = fit_dir(asset, granularity, model.variant, model.n_components,
                model.jump_penalty,
                max_feats=(model.max_feats if is_sparse_variant(model.variant) else None),
                root=root)
    d.mkdir(parents=True, exist_ok=True)

    # ── model (pickle): carries the scaler and the jumpmodels object ──
    with open(d / "model.pkl", "wb") as f:
        pickle.dump(model, f)

    # ── causal labels (what Stage 2 / M2 consume) ──
    causal = causal_proba.copy()
    causal.insert(0, "regime", causal_labels.reindex(causal.index).values)
    if causal_split is not None:
        causal.insert(1, "split", causal_split.reindex(causal.index).values)
    causal.index.name = "date"
    causal.to_csv(d / "labels.csv")

    # ── in-sample labels (exploration / visualization only) ──
    insmp = insample_proba.copy()
    insmp.insert(0, "regime", insample_labels.reindex(insmp.index).values)
    if insample_split is not None:
        insmp.insert(1, "split", insample_split.reindex(insmp.index).values)
    insmp.index.name = "date"
    insmp.to_csv(d / "labels_insample.csv")

    # ── metadata ──
    with open(d / "meta.json", "w") as f:
        json.dump(meta, f, indent=2, default=str)

    return d


def load_labels(asset: str, granularity: str, variant: str,
                n_components: int, jump_penalty: float,
                causal: bool = True, root: Path | None = None,
                max_feats: float | None = None) -> pd.DataFrame:
    """Load the regime label table for a fit. `causal=True` → labels.csv."""
    d = fit_dir(asset, granularity, variant, n_components, jump_penalty,
                root=root, max_feats=max_feats)
    fname = "labels.csv" if causal else "labels_insample.csv"
    path = d / fname
    if not path.exists():
        raise FileNotFoundError(f"No regime labels at {path}. Fit the JM first.")
    df = pd.read_csv(path, parse_dates=["date"]).set_index("date")
    return df


def load_model(asset: str, granularity: str, variant: str,
               n_components: int, jump_penalty: float,
               root: Path | None = None,
               max_feats: float | None = None) -> "RegimeJumpModel":
    """Load the fitted RegimeJumpModel object."""
    d = fit_dir(asset, granularity, variant, n_components, jump_penalty,
                root=root, max_feats=max_feats)
    path = d / "model.pkl"
    if not path.exists():
        raise FileNotFoundError(f"No fitted model at {path}. Fit the JM first.")
    # Trusted input: load_model only reads model.pkl files that save_fit wrote
    # under this repo's Output/ tree (no untrusted/remote pickles).
    with open(path, "rb") as f:
        return pickle.load(f)  # nosec B301 — first-party artifact written by save_fit


def load_metadata(asset: str, granularity: str, variant: str,
                  n_components: int, jump_penalty: float,
                  root: Path | None = None,
                  max_feats: float | None = None) -> dict[str, Any]:
    """Load the meta.json payload for a persisted fit."""
    d = fit_dir(asset, granularity, variant, n_components, jump_penalty,
                root=root, max_feats=max_feats)
    path = d / "meta.json"
    if not path.exists():
        raise FileNotFoundError(f"No regime metadata at {path}. Fit the JM first.")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_global_fit(granularity: str,
                    model: "RegimeJumpModel",
                    causal_by_asset: dict[str, tuple[pd.Series, pd.DataFrame, pd.Series | None]],
                    insample_by_asset: dict[str, tuple[pd.Series, pd.DataFrame, pd.Series | None]],
                    meta: dict[str, Any],
                    root: Path | None = None) -> Path:
    """Persist a pooled model plus per-asset causal/in-sample label files."""
    d = global_fit_dir(
        granularity,
        model.variant,
        model.n_components,
        model.jump_penalty,
        max_feats=(model.max_feats if is_sparse_variant(model.variant) else None),
        root=root,
    )
    d.mkdir(parents=True, exist_ok=True)
    (d / "labels").mkdir(exist_ok=True)
    (d / "labels_insample").mkdir(exist_ok=True)

    with open(d / "model.pkl", "wb") as f:
        pickle.dump(model, f)

    for asset, (labels, proba, split) in causal_by_asset.items():
        df = proba.copy()
        df.insert(0, "regime", labels.reindex(df.index).values)
        if split is not None:
            df.insert(1, "split", split.reindex(df.index).values)
        df.index.name = "date"
        df.to_csv(d / "labels" / f"{asset}.csv")

    for asset, (labels, proba, split) in insample_by_asset.items():
        df = proba.copy()
        df.insert(0, "regime", labels.reindex(df.index).values)
        if split is not None:
            df.insert(1, "split", split.reindex(df.index).values)
        df.index.name = "date"
        df.to_csv(d / "labels_insample" / f"{asset}.csv")

    with open(d / "meta.json", "w") as f:
        json.dump(meta, f, indent=2, default=str)
    return d


def load_global_model(granularity: str,
                      variant: str,
                      n_components: int,
                      jump_penalty: float,
                      root: Path | None = None,
                      max_feats: float | None = None) -> "RegimeJumpModel":
    """Load the pooled global RegimeJumpModel object."""
    d = global_fit_dir(granularity, variant, n_components, jump_penalty,
                       root=root, max_feats=max_feats)
    path = d / "model.pkl"
    if not path.exists():
        raise FileNotFoundError(f"No fitted global model at {path}. Fit the global JM first.")
    with open(path, "rb") as f:
        return pickle.load(f)  # nosec B301 — first-party artifact written by save_global_fit


def load_global_labels(granularity: str,
                       variant: str,
                       n_components: int,
                       jump_penalty: float,
                       asset: str,
                       causal: bool = True,
                       root: Path | None = None,
                       max_feats: float | None = None) -> pd.DataFrame:
    """Load one asset's label table from a pooled global fit."""
    d = global_fit_dir(granularity, variant, n_components, jump_penalty,
                       root=root, max_feats=max_feats)
    folder = "labels" if causal else "labels_insample"
    path = d / folder / f"{asset}.csv"
    if not path.exists():
        raise FileNotFoundError(f"No global regime labels at {path}. Fit the global JM first.")
    return pd.read_csv(path, parse_dates=["date"]).set_index("date")


__all__ = [
    "JUMP_MODELS_ROOT",
    "fit_dir",
    "global_fit_dir",
    "save_fit",
    "save_global_fit",
    "load_labels",
    "load_global_labels",
    "load_model",
    "load_global_model",
    "load_metadata",
]
