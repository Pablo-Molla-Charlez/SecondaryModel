from typing import Tuple
import pandas as pd
import numpy as np
import torch

from Utils.utils import _load_multi_cache
from Utils.data_preprocessing import split_by_global_time, ENG_FEATURE_NAMES


def load_tabular_dataset_from_cache_to_DataFrame(
        cache_path: str,
        gran: str,
        train_end: str = "2025-05-30",
        val_end: str = "2025-10-01",
) -> Tuple[pd.DataFrame, np.ndarray, pd.DataFrame, np.ndarray]:
    """
    Convenience function for loading the tabular dataset from cache
    Args:
        cache_path: path to cache directory (specify the experimental setting, m1, etc.); (without .pt)
        gran: the desired granularity ("1d", "12h", etc.)
        train_end: Defaults to "2025-05-30"
        val_end:  Defaults to "2025-10-01"

    Returns:
        X_analysis: pd.DataFrame
        y_analysis: np.ndarray
        X_test: pd.DataFrame
        y_test: np.ndarray
    """
    multi = _load_multi_cache(f'{cache_path}.pt')

    sub = multi.sub[gran]
    idx_train, _, idx_val, idx_test = split_by_global_time(sub, train_end=train_end, val_end=val_end)
    eng_raw = sub["eng_features"].numpy() if isinstance(sub["eng_features"], torch.Tensor) else sub["eng_features"]
    labels_raw = sub["labels"].numpy() if isinstance(sub["labels"], torch.Tensor) else sub["labels"]
    
    # include other metadata
    returns = sub["returns"].numpy() if isinstance(sub["returns"], torch.Tensor) else sub["returns"]
    asset_ids = sub["asset_ids"].numpy() if isinstance(sub["asset_ids"], torch.Tensor) else sub["asset_ids"]
    ## Dont know what this is supposed to mean
    # if not isinstance(asset_map, dict) and hasattr(sub, "asset_map"): asset_map = sub.asset_map
    asset_map = sub.get("asset_map", {}) if isinstance(sub["asset_map"], dict) else sub.asset_map
    
    X_train = pd.DataFrame(eng_raw[idx_train], columns=ENG_FEATURE_NAMES, index=[sub["dates"][i] for i in idx_train])
    y_train = labels_raw[idx_train].astype(int)
    returns_train = pd.DataFrame(returns[idx_train], columns=["returns"], index=[sub["dates"][i] for i in idx_train])
    asset_ids_train = pd.DataFrame(asset_ids[idx_train], columns=["asset_id"], index=[sub["dates"][i] for i in idx_train])

    X_val = pd.DataFrame(eng_raw[idx_val], columns=ENG_FEATURE_NAMES, index=[sub["dates"][i] for i in idx_val])
    y_val = labels_raw[idx_val].astype(int)
    returns_val = pd.DataFrame(returns[idx_val], columns=["returns"], index=[sub["dates"][i] for i in idx_val])
    asset_ids_val = pd.DataFrame(asset_ids[idx_val], columns=["asset_id"], index=[sub["dates"][i] for i in idx_val])

    X_test = pd.DataFrame(eng_raw[idx_test], columns=ENG_FEATURE_NAMES, index=[sub["dates"][i] for i in idx_test])
    y_test = labels_raw[idx_test].astype(int)
    returns_test = pd.DataFrame(returns[idx_test], columns=["returns"], index=[sub["dates"][i] for i in idx_test])
    asset_ids_test = pd.DataFrame(asset_ids[idx_test], columns=["asset_id"], index=[sub["dates"][i] for i in idx_test])
    
    # merge X_train and X_val
    X_analysis = pd.concat([X_train, X_val], axis=0).sort_index()
    y_analysis = np.concatenate([y_train, y_val])
    returns_analysis = pd.concat([returns_train, returns_val], axis=0).sort_index()
    asset_ids_analysis = pd.concat([asset_ids_train, asset_ids_val], axis=0).sort_index()

    print(asset_map)
    print(f"train: {X_train.shape} | {y_train.shape} | {returns_train.shape} | {asset_ids_train.shape}")
    print(f"val: {X_val.shape} | {y_val.shape} | {returns_val.shape} | {asset_ids_val.shape}")
    print(f"test: {X_test.shape} | {y_test.shape} | {returns_test.shape} | {asset_ids_test.shape}")
    print(f"Done loading ....")
    print(f"\n\n\n")
    print(returns_analysis.head())
    print(f"\n\n\n")
    print(asset_ids_analysis.head())
    print(f"\n\n\n")
    print(X_analysis.head())
    
    return X_analysis, y_analysis, X_test, y_test, returns_analysis, asset_ids_analysis, returns_test, asset_ids_test, asset_map
