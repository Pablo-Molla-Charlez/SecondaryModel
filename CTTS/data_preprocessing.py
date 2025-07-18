import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from torch.utils.data import TensorDataset, Subset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from typing import List, Tuple, Union


def merge_meta_targets(asset_type: str,
                       asset: str,
                       data_dir: str,
                       output_dir: str = None,
                       set_index: bool = True) -> pd.DataFrame:
    """
    Load and merge 'down' and 'up' meta_target CSVs for a given asset type and asset.

    Parameters
    ----------
    asset_type : str
        Either 'crypto' or 'equities'.
    asset : str
        Asset symbol, e.g. 'BTC' or 'SPY'.
    data_dir : str
        Directory containing CSVs named like '{ASSET}_down.csv' and '{ASSET}_up.csv'.
    output_dir : str, optional
        If given, merged CSV is saved here under '{ASSET}_merge.csv'.
    set_index : bool, default True
        Whether to set the 'date' column as index on the returned DataFrame.

    Returns
    -------
    pd.DataFrame
        The merged DataFrame, with columns ['close', 'isTP_DN', 'isTP_UP'].
    """
    atype = asset_type.upper()
    sym   = asset.upper()
    dn_path = os.path.join(data_dir, f"{sym}_down.csv")
    up_path = os.path.join(data_dir, f"{sym}_up.csv")

    # 1) Down side
    df_dn = (
        pd.read_csv(dn_path, parse_dates=["date"])
          .loc[:, ["date", "close", "meta_target"]]
          .rename(columns={"meta_target": "isTP_DN"})
    )

    # 2) Up side
    df_up = (
        pd.read_csv(up_path, parse_dates=["date"])
          .loc[:, ["date", "meta_target"]]
          .rename(columns={"meta_target": "isTP_UP"})
    )

    # 3) Merge outer on date
    df = (
        pd.merge(df_dn, df_up, on="date", how="outer")
          .sort_values("date")
          .reset_index(drop=True)
    )

    # 4) Optional index
    if set_index:
        df = df.set_index("date")

    # 5) Optional save
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        out_path = os.path.join(output_dir, f"{sym}_merge.csv")
        df.to_csv(out_path)

    return df
    

def prepare_dataset(df: pd.DataFrame,
                    seq_len: int = 90) -> TensorDataset:
    "Convert raw DF → sliding-window TensorDataset (no splits)."
    # 1) Scale & extract
    df = df.copy()
    df['close_scaled'] = MinMaxScaler().fit_transform(df[['close']])
    values = df['close_scaled'].to_numpy(dtype=np.float32)
    y_up   = df['isTP_UP'].fillna(0).to_numpy(dtype=np.int64)
    y_dn   = df['isTP_DN'].fillna(0).to_numpy(dtype=np.int64)

    n = len(df)
    N = n - seq_len + 1
    
    # 2) Build sliding windows
    X    = np.zeros((N, 1, seq_len), dtype=np.float32)
    Y_up = np.zeros((N,), dtype=np.int64)
    Y_dn = np.zeros((N,), dtype=np.int64)
    for i in range(N):
        start, end     = i, i + seq_len
        X[i, 0, :]     = values[start:end]
        Y_up[i]        = y_up[end-1]
        Y_dn[i]        = y_dn[end-1]

    # 3) Return torch tensors
    return TensorDataset(torch.from_numpy(X),
                         torch.from_numpy(Y_up),
                         torch.from_numpy(Y_dn))


def build_loaders(ds: TensorDataset,
                  cross_validation: bool,
                  target:           str,
                  props:            float,
                  train_frac:       float,
                  val_frac:         float,
                  test_frac:        float,
                  batch_size:       int,
                  loss_type:        str,
                  device:           torch.device
                  ) -> Tuple[List[Tuple[DataLoader, DataLoader, nn.Module]],
                             DataLoader]:
    """
    Build train/val/test DataLoaders and loss criterion, with or without cross-validation.

    Returns:
      folds:       List of (train_loader, val_loader, criterion)
      val_loader:  Fixed (outer) validation set
      test_loader: Fixed test set
    """

    N = len(ds)

    # Compute split indices
    n_train = int(train_frac * N)
    n_val   = int(val_frac   * N)
    n_test  = N - n_train - n_val

    idx_val  = list(range(n_train, n_train + n_val))
    idx_test = list(range(n_train + n_val, N))

    val_loader_outer  = DataLoader(Subset(ds, idx_val),  batch_size=batch_size, shuffle=False)
    test_loader       = DataLoader(Subset(ds, idx_test), batch_size=batch_size, shuffle=False)

    folds = []

    if not cross_validation:
        # ┏━━━━━━━━━━ Single Fold (No CV) ━━━━━━━━━━┓
        idx_train = list(range(0, n_train))
        train_loader = DataLoader(Subset(ds, idx_train), batch_size=batch_size, shuffle=False)

        y_up_tr = ds.tensors[1][idx_train]
        y_dn_tr = ds.tensors[2][idx_train]
        
        # ┏━━━━━━━━━━ Compute class-weights on TRAIN only ━━━━━━━━━━┓
        if loss_type == 'cross_entropy':
            cu = torch.bincount(y_up_tr, minlength=2).float()
            cd = torch.bincount(y_dn_tr, minlength=2).float()
            tu, td = cu.sum(), cd.sum()
            w_up = tu / (cu + 1e-8)
            w_dn = td / (cd + 1e-8)
        elif loss_type == 'bce':
            pu, nu = y_up_tr.sum().float(), y_up_tr.numel() - y_up_tr.sum().float()
            pd, nd = y_dn_tr.sum().float(), y_dn_tr.numel() - y_dn_tr.sum().float()
            w_up = nu / (pu + 1e-8)
            w_dn = nd / (pd + 1e-8)
        else:
            raise ValueError("loss_type must be 'bce' or 'cross_entropy'")

        w_up = w_up.to(device)
        w_dn = w_dn.to(device)

        # ┏━━━━━━━━━━ Build both criteria, then pick the one for `target` ━━━━━━━━━━┓
        if loss_type == 'bce':
            crit_up = nn.BCEWithLogitsLoss() # crit_up = nn.BCEWithLogitsLoss(pos_weight=w_up)
            crit_dn = nn.BCEWithLogitsLoss(pos_weight=w_dn)
        else:
            crit_up = nn.CrossEntropyLoss()
            crit_dn = nn.CrossEntropyLoss(weight=w_dn)

        crit_up = crit_up.to(device)
        crit_dn = crit_dn.to(device)
        criterion = crit_up if target.upper() == 'UP' else crit_dn

        folds.append((train_loader, val_loader_outer, criterion))

    else:
        # ┏━━━━━━━━━━ Multiple Folds (CV) ━━━━━━━━━━┓
        p_start = props
        
        # CV applied through out training and original validation samples
        n_pool = n_train + n_val
        fold_idx = 1
        train_end = int(n_pool * p_start)
        val_window = n_val

        while train_end + val_window <= n_pool:
            start_tr, end_tr = 0, train_end
            start_va, end_va = train_end, train_end + val_window

            idx_train_fold = list(range(0, train_end))
            idx_val_fold   = list(range(train_end, train_end + val_window))

            assert max(idx_train_fold) <= min(idx_val_fold), "Train/Val overlap!"
            assert max(idx_val_fold)   < n_pool, "Val fold exceeds pool boundary!"

            train_loader = DataLoader(Subset(ds, idx_train_fold), batch_size=batch_size, shuffle=False)
            val_loader   = DataLoader(Subset(ds, idx_val_fold),   batch_size=batch_size, shuffle=False)

            y_up_tr = ds.tensors[1][idx_train_fold]
            y_dn_tr = ds.tensors[2][idx_train_fold]

            # ┏━━━━━━━━━━ Compute class-weights on TRAIN Fold only ━━━━━━━━━━┓
            if loss_type == 'cross_entropy':
                cu = torch.bincount(y_up_tr, minlength=2).float()
                cd = torch.bincount(y_dn_tr, minlength=2).float()
                tu, td = cu.sum(), cd.sum()
                w_up = tu / (cu + 1e-8)
                w_dn = td / (cd + 1e-8)
            elif loss_type == 'bce':
                pu, nu = y_up_tr.sum().float(), y_up_tr.numel() - y_up_tr.sum().float()
                pd, nd = y_dn_tr.sum().float(), y_dn_tr.numel() - y_dn_tr.sum().float()
                w_up = nu / (pu + 1e-8)
                w_dn = nd / (pd + 1e-8)
            else:
                raise ValueError("loss_type must be 'bce' or 'cross_entropy'")

            w_up = w_up.to(device)
            w_dn = w_dn.to(device)

            # ┏━━━━━━━━━━ Build both criteria, then pick the one for `target` ━━━━━━━━━━┓
            if loss_type == 'bce':
                crit_up = nn.BCEWithLogitsLoss()
                crit_dn = nn.BCEWithLogitsLoss(pos_weight=w_dn)
            else:
                crit_up = nn.CrossEntropyLoss()
                crit_dn = nn.CrossEntropyLoss(weight=w_dn)

            crit_up = crit_up.to(device)
            crit_dn = crit_dn.to(device)
            criterion = crit_up if target.upper() == 'UP' else crit_dn

            folds.append((train_loader, val_loader, criterion))

            fold_idx += 1
            train_end += val_window  # Expand training window
        
        # Handle leftover samples in the pool:
        if train_end < n_pool:
            # Last fold should mirror original split sizes
            start_tr, end_tr = 0, n_train
            start_va, end_va = n_train, n_train + n_val

            idx_tr = list(range(start_tr, end_tr))
            idx_va = list(range(start_va, end_va))

            train_loader = DataLoader(Subset(ds, idx_tr),
                                      batch_size=batch_size,
                                      shuffle=False)
            val_loader   = DataLoader(Subset(ds, idx_va),
                                      batch_size=batch_size,
                                      shuffle=False)

            # ┏━━━━━━━━━━ Compute class-weights on TRAIN Fold only ━━━━━━━━━━┓
            if loss_type == 'cross_entropy':
                cu = torch.bincount(y_up_tr, minlength=2).float()
                cd = torch.bincount(y_dn_tr, minlength=2).float()
                tu, td = cu.sum(), cd.sum()
                w_up = tu / (cu + 1e-8)
                w_dn = td / (cd + 1e-8)
            elif loss_type == 'bce':
                pu, nu = y_up_tr.sum().float(), y_up_tr.numel() - y_up_tr.sum().float()
                pd, nd = y_dn_tr.sum().float(), y_dn_tr.numel() - y_dn_tr.sum().float()
                w_up = nu / (pu + 1e-8)
                w_dn = nd / (pd + 1e-8)
            else:
                raise ValueError("loss_type must be 'bce' or 'cross_entropy'")

            w_up = w_up.to(device)
            w_dn = w_dn.to(device)

            # ┏━━━━━━━━━━ Build both criteria, then pick the one for `target` ━━━━━━━━━━┓
            if loss_type == 'bce':
                crit_up = nn.BCEWithLogitsLoss()
                crit_dn = nn.BCEWithLogitsLoss(pos_weight=w_dn)
            else:
                crit_up = nn.CrossEntropyLoss()
                crit_dn = nn.CrossEntropyLoss(weight=w_dn)

            crit_up = crit_up.to(device)
            crit_dn = crit_dn.to(device)
            criterion = crit_up if target.upper() == 'UP' else crit_dn

            folds.append((train_loader, val_loader, criterion))

    return folds, test_loader


