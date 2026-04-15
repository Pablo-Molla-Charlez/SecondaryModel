from typing import Tuple
import pandas as pd
import numpy as np
import torch

from Utils.utils import _load_multi_cache
from Utils.data_preprocessing import split_by_global_time, ENG_FEATURE_NAMES


def load_tabular_dataset_from_cache_to_DataFrame(cache_path: str,
                                                 gran: str,
                                                 train_end: str = "2025-05-30",
                                                 val_end: str = "2025-10-01") -> Tuple[pd.DataFrame, np.ndarray, pd.DataFrame, np.ndarray]:
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
    # ┏━━━━━━━━━━ Load Multi-Granularity Dataset ━━━━━━━━━━┓
    multi = _load_multi_cache(f'{cache_path}.pt')

    # ┏━━━━━━━━━━ Extract Granularity ━━━━━━━━━━┓
    sub = multi.sub[gran]
    
    # ┏━━━━━━━━━━ Split by global time ━━━━━━━━━━┓
    idx_train, _, idx_val, idx_test = split_by_global_time(sub, train_end=train_end, val_end=val_end)
    eng_raw = sub["eng_features"].numpy() if isinstance(sub["eng_features"], torch.Tensor) else sub["eng_features"]
    labels_raw = sub["labels"].numpy() if isinstance(sub["labels"], torch.Tensor) else sub["labels"]

    # ┏━━━━━━━━━━ DataFrames for Train, Validation and Test ━━━━━━━━━━┓
    X_train = pd.DataFrame(eng_raw[idx_train], columns=ENG_FEATURE_NAMES, index=[sub["dates"][i] for i in idx_train])
    y_train = labels_raw[idx_train].astype(int)
    X_val = pd.DataFrame(eng_raw[idx_val], columns=ENG_FEATURE_NAMES, index=[sub["dates"][i] for i in idx_val])
    y_val = labels_raw[idx_val].astype(int)
    X_test = pd.DataFrame(eng_raw[idx_test], columns=ENG_FEATURE_NAMES, index=[sub["dates"][i] for i in idx_test])
    y_test = labels_raw[idx_test].astype(int)

    # ┏━━━━━━━━━━ Merge X_train and X_val ━━━━━━━━━━┓
    X_analysis = pd.concat([X_train, X_val], axis=0).reset_index(drop=True)
    y_analysis = np.concatenate([y_train, y_val])

    return X_analysis, y_analysis, X_test, y_test
