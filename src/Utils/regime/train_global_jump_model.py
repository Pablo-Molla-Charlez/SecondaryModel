"""Fit one pooled Jump Model across assets and emit per-asset labels.

The global JM is a shared estimator trained on concatenated feature frames, with
asset groups passed into fitting so the E-step and jump counts reset at asset
boundaries. It still produces one causal label/probability sequence per asset.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from Utils.regime import storage
from Utils.regime.config import (
    SplitConfig,
    clip_ohlcv_to_split_window,
    load_regime_config,
    load_regime_model_defaults,
    load_split_config,
    resolve_output_root,
    resolve_raw_data_dir,
    split_frame_by_config,
)
from Utils.regime.data_access import load_ohlcv
from Utils.regime.features import BARS_PER_DAY, FEATURE_SETS, build_regime_features, compute_log_returns
from Utils.regime.model import RegimeJumpModel, is_sparse_variant
from Utils.regime.stat_tuning import (
    DEFAULT_BIC_ACTIVE_THRESHOLD,
    DEFAULT_BIC_A0,
    DEFAULT_BIC_K0,
    DEFAULT_BIC_KBAR,
    DEFAULT_K_GRID,
    DEFAULT_LAMBDA_GRID,
    count_jumps,
    default_bic_a0,
)
from Utils.regime.paths import DEFAULT_CONFIG_PATH
from Utils.regime.bic_selection import (
    DEFAULT_CONTINUOUS_K_GRID,
    _hard_bic_selection_variant,
    _resolve_bic_k_grid,
    _select_by_modified_bic,
    _validate_continuous_k_cap,
)


@dataclass(frozen=True)
class AssetRegimeData:
    asset: str
    features: pd.DataFrame
    log_returns: pd.Series
    features_by_split: dict[str, pd.DataFrame]
    returns_by_split: dict[str, pd.Series]


def _discover_assets(raw_root: Path, granularity: str) -> list[str]:
    d = raw_root / granularity
    if not d.is_dir():
        return []
    return sorted(p.stem for p in d.glob("*.csv") if p.is_file())


def _prepare_asset_data(asset: str,
                        granularity: str,
                        *,
                        raw_root: Path,
                        split_cfg: SplitConfig,
                        feature_set: str,
                        explicit: bool,
                        verbose: bool) -> AssetRegimeData | None:
    try:
        ohlcv = clip_ohlcv_to_split_window(
            load_ohlcv(asset, granularity, root=raw_root), split_cfg)
        if ohlcv.empty:
            raise ValueError("empty OHLCV inside split window")
        feats = build_regime_features(
            ohlcv,
            granularity,
            feature_set=feature_set,
            asset=asset,
            raw_root=raw_root,
        )
        if feats.empty:
            raise ValueError("empty regime feature frame")
        log_ret = compute_log_returns(ohlcv["close"]).reindex(feats.index)
        valid = (
            np.isfinite(feats.to_numpy(dtype=float)).all(axis=1)
            & np.isfinite(log_ret.to_numpy(dtype=float))
        )
        if not bool(valid.all()):
            feats = feats.loc[valid]
            log_ret = log_ret.loc[valid]
        if feats.empty:
            raise ValueError("no finite regime feature/return rows")
        features_by_split = split_frame_by_config(feats, split_cfg)
        returns_by_split = split_frame_by_config(log_ret, split_cfg)
        if features_by_split["train"].empty:
            raise ValueError("empty train feature split")
        if features_by_split["test"].empty:
            raise ValueError("empty test feature split")
        return AssetRegimeData(
            asset=asset,
            features=feats,
            log_returns=log_ret,
            features_by_split=features_by_split,
            returns_by_split=returns_by_split,
        )
    except Exception:
        if explicit:
            raise
        if verbose:
            print(f"[global-regime] skipping auto-discovered {asset}: no usable features")
        return None


def _concat_asset_frames(parts: list[tuple[str, pd.DataFrame, pd.Series]]
                         ) -> tuple[pd.DataFrame, pd.Series, np.ndarray]:
    X_parts, ret_parts, groups = [], [], []
    for asset, feats, returns in parts:
        X_parts.append(feats)
        ret_parts.append(returns.reindex(feats.index))
        groups.extend([asset] * len(feats))
    return pd.concat(X_parts), pd.concat(ret_parts), np.asarray(groups, dtype=object)


def _split_membership(index: pd.Index, split_cfg: SplitConfig) -> pd.Series:
    return pd.Series(
        np.where(index <= pd.Timestamp(split_cfg.train_end), "train", "val"),
        index=index,
        name="split",
    )


def _assemble_pooled_fit_window(
    asset_data: Sequence[AssetRegimeData],
    *,
    select_bic: bool,
) -> tuple[pd.DataFrame, pd.Series, np.ndarray, dict[str, pd.DataFrame], list[str]]:
    """Validate feature alignment and pool the per-asset fit windows.

    Every asset must share the first asset's feature columns. The fit window is
    Train+Val under BIC selection (the criterion is self-penalizing) and Train
    only otherwise. Returns the pooled ``(features, returns, groups)``, the
    per-asset fit features, and the shared feature column order.
    """
    expected_columns = list(asset_data[0].features.columns)
    for data in asset_data[1:]:
        cols = list(data.features.columns)
        if cols != expected_columns:
            missing = sorted(set(expected_columns) - set(cols))
            extra = sorted(set(cols) - set(expected_columns))
            raise ValueError(
                f"Feature columns for {data.asset} do not match {asset_data[0].asset}: "
                f"missing={missing}, extra={extra}"
            )

    fit_parts = []
    fit_feats_by_asset = {}
    for data in asset_data:
        if select_bic:
            feats_fit = pd.concat([data.features_by_split["train"], data.features_by_split["val"]])
            ret_fit = pd.concat([data.returns_by_split["train"], data.returns_by_split["val"]])
        else:
            feats_fit = data.features_by_split["train"]
            ret_fit = data.returns_by_split["train"]
        fit_feats_by_asset[data.asset] = feats_fit
        fit_parts.append((data.asset, feats_fit, ret_fit))
    feats_fit_global, ret_fit_global, groups_fit = _concat_asset_frames(fit_parts)
    return feats_fit_global, ret_fit_global, groups_fit, fit_feats_by_asset, expected_columns


def _label_assets(jm: RegimeJumpModel,
                  asset_data: Sequence[AssetRegimeData],
                  fit_feats_by_asset: dict[str, pd.DataFrame],
                  chosen_k: int,
                  split_cfg) -> tuple[dict, dict, dict, int, int]:
    """Emit per-asset causal (online) and in-sample labels/probabilities + meta.

    Returns ``(causal_by_asset, insample_by_asset, per_asset_meta,
    total_causal_jumps, total_insample_jumps)``. Causal labels are inferred
    split-by-split with the online predictor; in-sample labels use the pooled fit
    features. Empty splits yield correctly-shaped empty label/proba frames.
    """
    causal_by_asset: dict[str, tuple[pd.Series, pd.DataFrame, pd.Series | None]] = {}
    insample_by_asset: dict[str, tuple[pd.Series, pd.DataFrame, pd.Series | None]] = {}
    per_asset_meta = {}
    total_causal_jumps = 0
    total_insample_jumps = 0
    for data in asset_data:
        label_parts, proba_parts, split_parts = [], [], []
        jump_by_split, counts_by_split = {}, {}
        for split_name in ("train", "val", "test"):
            feats_split = data.features_by_split[split_name]
            if feats_split.empty:
                labels = pd.Series(dtype=int, index=feats_split.index, name="regime")
                proba = pd.DataFrame(
                    index=feats_split.index,
                    columns=[f"p_regime_{i}" for i in range(chosen_k)],
                    dtype=float,
                )
            else:
                labels = jm.predict_online(feats_split)
                proba = jm.predict_proba_online(feats_split)
            label_parts.append(labels)
            proba_parts.append(proba)
            split_parts.append(pd.Series(split_name, index=labels.index, name="split"))
            jump_by_split[split_name] = count_jumps(labels)
            counts_by_split[split_name] = labels.value_counts().sort_index().to_dict()
        causal_labels = pd.concat(label_parts).sort_index()
        causal_proba = pd.concat(proba_parts).sort_index()
        causal_split = pd.concat(split_parts).reindex(causal_labels.index)
        causal_jumps = count_jumps(causal_labels)
        total_causal_jumps += causal_jumps
        causal_by_asset[data.asset] = (causal_labels, causal_proba, causal_split)

        asset_fit_feats = fit_feats_by_asset[data.asset]
        insample_labels = jm.predict(asset_fit_feats)
        insample_proba = jm.predict_proba(asset_fit_feats)
        insample_split = _split_membership(asset_fit_feats.index, split_cfg)
        insample_jumps = count_jumps(insample_labels)
        total_insample_jumps += insample_jumps
        insample_by_asset[data.asset] = (insample_labels, insample_proba, insample_split)

        per_asset_meta[data.asset] = {
            "n_bars": int(len(data.features)),
            "n_bars_by_split": {name: int(len(part)) for name, part in data.features_by_split.items()},
            "n_shifts_causal": int(causal_jumps),
            "n_shifts_causal_by_split": jump_by_split,
            "n_shifts_insample": int(insample_jumps),
            "regime_counts_causal": causal_labels.value_counts().sort_index().to_dict(),
            "regime_counts_causal_by_split": counts_by_split,
            "regime_counts_insample": insample_labels.value_counts().sort_index().to_dict(),
        }
    return (
        causal_by_asset,
        insample_by_asset,
        per_asset_meta,
        total_causal_jumps,
        total_insample_jumps,
    )


def fit_global_jump_model(*,
                          granularity: str = "4h",
                          assets: Sequence[str] | None = None,
                          variant: str | None = None,
                          k: int | None = None,
                          feature_set: str = "combined",
                          fixed_lambda: float | None = None,
                          select_bic: bool = False,
                          select_financial: bool = False,
                          k_grid: Sequence[int] | None = None,
                          lambda_grid: Sequence[float] | None = None,
                          kappa_grid: Sequence[float] | None = None,
                          kappa_grid_size: int = 8,
                          bic_k0: int = DEFAULT_BIC_K0,
                          bic_a0: int | None = DEFAULT_BIC_A0,
                          bic_kbar: int = DEFAULT_BIC_KBAR,
                          bic_active_threshold: float = DEFAULT_BIC_ACTIVE_THRESHOLD,
                          max_feats: float | None = None,
                          config_path: str | Path | None = DEFAULT_CONFIG_PATH,
                          start_date: str | None = None,
                          train_end: str | None = None,
                          val_end: str | None = None,
                          end_date: str | None = None,
                          output_root: str | Path | None = None,
                          raw_root: str | Path | None = None,
                          verbose: bool = True) -> dict[str, object]:
    if select_financial:
        raise ValueError("--select-financial is not supported for global JM v1")
    cfg, config_file = load_regime_config(config_path)
    split_cfg = load_split_config(
        config_path=config_path,
        start_date=start_date,
        train_end=train_end,
        val_end=val_end,
        end_date=end_date,
    )
    regime_defaults = load_regime_model_defaults(cfg)
    variant = str(variant or regime_defaults["variant"])
    selection_variant = _hard_bic_selection_variant(variant) if select_bic else variant
    fixed_lambda = (
        float(fixed_lambda)
        if fixed_lambda is not None
        else float(regime_defaults["jump_penalty"])
    )
    chosen_k_default = int(k if k is not None else regime_defaults["k"])
    if select_bic:
        resolved_k_grid = _resolve_bic_k_grid(k_grid, variant=variant)
    else:
        _validate_continuous_k_cap(variant, chosen_k_default)
        resolved_k_grid = ()
    resolved_lambda_grid = tuple(float(x) for x in (lambda_grid or DEFAULT_LAMBDA_GRID))
    if not is_sparse_variant(selection_variant) and kappa_grid is not None:
        raise ValueError("kappa_grid is only valid with a sparse variant")
    if kappa_grid is not None and not select_bic:
        raise ValueError("kappa_grid is only valid with --select-bic")

    raw_data_root = resolve_raw_data_dir(
        config_path=config_path,
        override=raw_root,
        cfg=cfg,
        config_file=config_file,
    )
    jump_models_root = resolve_output_root(
        config_path=config_path,
        override=output_root,
        cfg=cfg,
        config_file=config_file,
    ) / "Jump_Models"

    explicit_assets = assets is not None and len(assets) > 0
    selected_assets = list(assets) if explicit_assets else _discover_assets(raw_data_root, granularity)
    if not selected_assets:
        raise ValueError(f"No assets found under {raw_data_root / granularity}")

    asset_data = [
        data for asset in selected_assets
        if (data := _prepare_asset_data(
            asset,
            granularity,
            raw_root=raw_data_root,
            split_cfg=split_cfg,
            feature_set=feature_set,
            explicit=explicit_assets,
            verbose=verbose,
        )) is not None
    ]
    if not asset_data:
        raise ValueError(f"No usable assets for global JM at {granularity}")

    (
        feats_fit_global,
        ret_fit_global,
        groups_fit,
        fit_feats_by_asset,
        expected_columns,
    ) = _assemble_pooled_fit_window(asset_data, select_bic=select_bic)
    fit_p = int(feats_fit_global.shape[1])
    resolved_max_feats = (
        float(np.sqrt(fit_p)) if is_sparse_variant(variant) and max_feats is None else max_feats
    )
    effective_bic_a0 = (
        default_bic_a0(fit_p)
        if is_sparse_variant(selection_variant) and bic_a0 is None
        else bic_a0
    )
    if granularity not in BARS_PER_DAY:
        raise ValueError(f"Unknown granularity '{granularity}'. Known: {list(BARS_PER_DAY)}")

    if verbose:
        print(f"[global-regime] assets={len(asset_data)} granularity={granularity}")
        print(f"[global-regime] pooled fit rows={len(feats_fit_global):,} features={fit_p}")

    bic_scores: list = []
    if select_bic:
        chosen_k, chosen_lambda, chosen_max_feats, bic_scores = _select_by_modified_bic(
            variant=selection_variant,
            feats_fit=feats_fit_global,
            ret_fit=ret_fit_global,
            groups_fit=groups_fit,
            fit_p=fit_p,
            resolved_max_feats=resolved_max_feats,
            k_grid=resolved_k_grid,
            lambda_grid=resolved_lambda_grid,
            kappa_grid=kappa_grid,
            kappa_grid_size=kappa_grid_size,
            bic_k0=bic_k0,
            bic_a0=effective_bic_a0,
            bic_kbar=bic_kbar,
            bic_active_threshold=bic_active_threshold,
            verbose=verbose,
        )
    else:
        chosen_k = chosen_k_default
        chosen_lambda = fixed_lambda
        chosen_max_feats = resolved_max_feats
        if verbose:
            print(f"[global-regime] using fixed K={chosen_k}, lambda={chosen_lambda:g}")

    _validate_continuous_k_cap(variant, chosen_k)
    jm = RegimeJumpModel(
        variant=variant,
        n_components=chosen_k,
        jump_penalty=chosen_lambda,
        max_feats=chosen_max_feats,
    )
    jm.fit(feats_fit_global, ret_fit_global, groups=groups_fit)

    (
        causal_by_asset,
        insample_by_asset,
        per_asset_meta,
        total_causal_jumps,
        total_insample_jumps,
    ) = _label_assets(jm, asset_data, fit_feats_by_asset, chosen_k, split_cfg)

    meta = {
        "scope": "global",
        "assets": [data.asset for data in asset_data],
        "granularity": granularity,
        "variant": variant,
        "n_components": int(chosen_k),
        "feature_set": feature_set,
        "feature_names": expected_columns,
        "jump_penalty": float(chosen_lambda),
        "max_feats": chosen_max_feats if is_sparse_variant(variant) else None,
        "hyperparameter_selection": "bic" if select_bic else "fixed",
        "hyperparameter_selection_variant": selection_variant if select_bic else None,
        "fit_scope": "train+val" if select_bic else "train",
        "k_grid": list(resolved_k_grid) if select_bic else None,
        "lambda_grid": list(resolved_lambda_grid),
        "bic_k0": bic_k0,
        "bic_a0": effective_bic_a0,
        "bic_kbar": bic_kbar,
        "bic_active_threshold": bic_active_threshold,
        "bic_scores": [s.to_meta_dict() for s in bic_scores],
        "bars_per_day": BARS_PER_DAY.get(granularity),
        "n_assets": int(len(asset_data)),
        "n_bars_fit": int(len(feats_fit_global)),
        "n_shifts_insample": int(total_insample_jumps),
        "n_shifts_causal": int(total_causal_jumps),
        "per_asset": per_asset_meta,
        "selected_features": jm.selected_features if is_sparse_variant(variant) else None,
        "config_path": split_cfg.config_path,
        "output_root": str(jump_models_root.parent),
        "raw_data_root": str(raw_data_root),
        "start_date": split_cfg.start_date,
        "train_end": split_cfg.train_end,
        "val_end": split_cfg.val_end,
        "end_date": split_cfg.end_date,
    }

    out_dir = storage.save_global_fit(
        granularity=granularity,
        model=jm,
        causal_by_asset=causal_by_asset,
        insample_by_asset=insample_by_asset,
        meta=meta,
        root=jump_models_root,
    )
    meta["output_dir"] = str(out_dir)
    if verbose:
        print(f"[global-regime] saved -> {out_dir}")
    return meta


def main() -> None:
    p = argparse.ArgumentParser(description="Fit one pooled global Jump Model across assets.")
    p.add_argument("--granularity", default="4h")
    p.add_argument("--assets", nargs="+", default=None,
                   help="asset symbols; default discovers all CSVs under Data_MLA/Raw/<granularity>/")
    p.add_argument("--variant", default=None,
                   choices=["standard", "continuous", "sparse", "sparse_continuous"])
    p.add_argument("--k", type=int, default=None)
    p.add_argument("--feature-set", default="combined", choices=FEATURE_SETS)
    p.add_argument("--fixed-lambda", type=float, default=None)
    selector = p.add_mutually_exclusive_group()
    selector.add_argument("--select-bic", action="store_true")
    selector.add_argument("--select-financial", action="store_true",
                          help="rejected for global JM v1")
    p.add_argument("--k-grid", type=int, nargs="+", default=None,
                   help=f"K grid for --select-bic. Defaults to {list(DEFAULT_K_GRID)} "
                        f"for hard variants and {list(DEFAULT_CONTINUOUS_K_GRID)} for continuous variants.")
    p.add_argument("--lambda-grid", type=float, nargs="+", default=None,
                   help=f"lambda grid for --select-bic. Defaults to {list(DEFAULT_LAMBDA_GRID)}")
    p.add_argument("--kappa-grid", type=float, nargs="+", default=None)
    p.add_argument("--kappa-grid-size", type=int, default=8)
    p.add_argument("--bic-k0", type=int, default=DEFAULT_BIC_K0)
    p.add_argument("--bic-a0", type=int, default=DEFAULT_BIC_A0)
    p.add_argument("--bic-kbar", type=int, default=DEFAULT_BIC_KBAR)
    p.add_argument("--bic-active-threshold", type=float, default=DEFAULT_BIC_ACTIVE_THRESHOLD)
    p.add_argument("--max-feats", type=float, default=None)
    p.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    p.add_argument("--start-date", default=None)
    p.add_argument("--train-end", default=None)
    p.add_argument("--val-end", default=None)
    p.add_argument("--end-date", default=None)
    p.add_argument("--output-root", default=None)
    p.add_argument("--raw-root", default=None)
    args = p.parse_args()

    summary = fit_global_jump_model(
        granularity=args.granularity,
        assets=args.assets,
        variant=args.variant,
        k=args.k,
        feature_set=args.feature_set,
        fixed_lambda=args.fixed_lambda,
        select_bic=args.select_bic,
        select_financial=args.select_financial,
        k_grid=args.k_grid,
        lambda_grid=args.lambda_grid,
        kappa_grid=args.kappa_grid,
        kappa_grid_size=args.kappa_grid_size,
        bic_k0=args.bic_k0,
        bic_a0=args.bic_a0,
        bic_kbar=args.bic_kbar,
        bic_active_threshold=args.bic_active_threshold,
        max_feats=args.max_feats,
        config_path=args.config,
        start_date=args.start_date,
        train_end=args.train_end,
        val_end=args.val_end,
        end_date=args.end_date,
        output_root=args.output_root,
        raw_root=args.raw_root,
    )
    print("\n" + json.dumps({k: v for k, v in summary.items() if k != "bic_scores"},
                            indent=2, default=str))


__all__ = ["fit_global_jump_model", "main"]


if __name__ == "__main__":
    main()
