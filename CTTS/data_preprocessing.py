import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

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


class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.5, alpha: float = 0.25, reduction: str = "mean"):
        """
        gamma: focusing parameter
        alpha: class-balance weight for the positive class
        reduction: 'none' | 'mean' | 'sum'
        """
        super().__init__()
        self.gamma    = gamma
        self.alpha    = alpha
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor):
        """
        logits: shape (N, C) for multi-class or (N,) for binary (pre-activation)
        targets: shape (N,) with {0,1} for binary or class indices for multi-class
        """
        if logits.dim() > 1:
            # ┏━━━━━━━━━━ Multi-class ━━━━━━━━━━┓
            probs   = F.softmax(logits, dim=1)
            p_t     = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
            # ┏━━━━━━━━━━ Per-sample alpha_t ━━━━━━━━━━┓
            alpha_t = torch.where(targets == 1,
                                  self.alpha,
                                  1 - self.alpha).to(logits.device)
            # ┏━━━━━━━━━━ Per-sample cross-entropy (no reduction) ━━━━━━━━━━┓
            ce      = F.nll_loss(torch.log(probs),
                                 targets,
                                 reduction="none")
        else:
            # ┏━━━━━━━━━━ Binary ━━━━━━━━━━┓
            probs   = torch.sigmoid(logits)
            p_t     = probs * targets + (1 - probs) * (1 - targets)
            alpha_t = targets * self.alpha + (1 - targets) * (1 - self.alpha)
            # ┏━━━━━━━━━━ Per-sample BCE (no reduction) ━━━━━━━━━━┓
            ce      = F.binary_cross_entropy_with_logits(logits,
                                                         targets.float(),
                                                         reduction="none")

        # ┏━━━━━━━━━━ Focal factor & Final per-sample loss ━━━━━━━━━━┓
        focal_factor = (1 - p_t).pow(self.gamma)
        loss = alpha_t * focal_factor * ce

        # ┏━━━━━━━━━━ Reduce  ━━━━━━━━━━┓
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss  # 'none'



def make_criteria(loss_type: str,
                  w_up: torch.Tensor,
                  w_dn: torch.Tensor,
                  device: torch.device,
                  focal_gamma: float = 2.5,
                  focal_alpha: float = 0.25,
                 ) -> Tuple[nn.Module, nn.Module]:
    """
    Build both 'UP' and 'DN' criteria, given:
    - loss_type ∈ {'bce','cross_entropy','focal'}
    - w_up, w_dn: positive-class weights (scalar tensors)
    """
    # ┏━━━━━━━━━━ BCE ━━━━━━━━━━┓
    if loss_type == 'bce':
        crit_up = nn.BCEWithLogitsLoss()
        crit_dn = nn.BCEWithLogitsLoss(pos_weight = w_dn)

    # ┏━━━━━━━━━━ Cross‐Entropy ━━━━━━━━━━┓
    elif loss_type == 'cross_entropy':
        crit_up = nn.CrossEntropyLoss()
        crit_dn = nn.CrossEntropyLoss(weight = w_dn)

    # ┏━━━━━━━━━━ Focal Loss ━━━━━━━━━━┓
    elif loss_type == 'focal':
        # ┏━━━━━━━━━━ Select the positive‐class weight (index 1) if vector, else tensor itself ━━━━━━━━━━┓
        up_pos = w_up[1] if w_up.numel() > 1 else w_up
        dn_pos = w_dn[1] if w_dn.numel() > 1 else w_dn
        total = up_pos + dn_pos
        # ┏━━━━━━━━━━ Normalize into scalars in [0,1]  ━━━━━━━━━━┓
        alpha_up = (up_pos/total).item() if focal_alpha is None else focal_alpha
        alpha_dn = (dn_pos/total).item() if focal_alpha is None else focal_alpha
        
        # ┏━━━━━━━━━━ Criterion UP ━━━━━━━━━━┓
        crit_up = FocalLoss(
            gamma = focal_gamma,
            alpha = alpha_up,
            reduction = 'mean')

        # ┏━━━━━━━━━━ Criterion DN ━━━━━━━━━━┓
        crit_dn = FocalLoss(
            gamma = focal_gamma,
            alpha = alpha_dn,
            reduction = 'mean')
    else:
        raise ValueError(f"Unknown loss_type: {loss_type!r}")

    return crit_up.to(device), crit_dn.to(device)



def build_loaders(ds: TensorDataset,
                  cross_validation: bool,
                  target:           str,
                  props:            float,
                  train_frac:       float,
                  val_frac:         float,
                  test_frac:        float,
                  batch_size:       int,
                  loss_type:        str,
                  focal_gamma:      float,
                  focal_alpha:      float,
                  device:           torch.device
                  ) -> Tuple[List[Tuple[DataLoader, DataLoader, nn.Module]],
                             DataLoader, DataLoader]:
    """
    Build train/val/test DataLoaders and loss criterion, with or without cross-validation.

    Returns:
      folds:       List of (train_loader, val_loader, criterion)
      val_loader:  Fixed (outer) validation set
      test_loader: Fixed test set
    """

    N = len(ds)

    # ┏━━━━━━━━━━ Compute split indices ━━━━━━━━━━┓
    n_train = int(train_frac * N)
    n_val   = int(val_frac   * N)
    n_test  = N - n_train - n_val

    idx_val  = list(range(n_train, n_train + n_val))
    idx_test = list(range(n_train + n_val, N))

    val_loader_outer  = DataLoader(Subset(ds, idx_val),
                                   batch_size = batch_size,
                                   shuffle    = False)

    test_loader       = DataLoader(Subset(ds, idx_test),
                                   batch_size = batch_size,
                                   shuffle    = False)
    folds = []

    # ┏━━━━━━━━━━ Seed for all train‐set shuffles ━━━━━━━━━━┓
    _seed = 42


    if not cross_validation:
        # ┏━━━━━━━━━━ Single Fold (No CV) ━━━━━━━━━━┓
        idx_train = list(range(0, n_train))
        
        # ┏━━━━━━━━━━ Extract targets on Train only ━━━━━━━━━━┓
        y_up_tr = ds.tensors[1][idx_train]
        y_dn_tr = ds.tensors[2][idx_train]
        
        # ┏━━━━━━━━━━ Compute class-weights on TRAIN only ━━━━━━━━━━┓
        if loss_type in ('cross_entropy', 'focal'):
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
            raise ValueError("loss_type must be 'bce', 'cross_entropy' or 'focal")

        w_up = w_up.to(device)
        w_dn = w_dn.to(device)

        # ┏━━━━━━━━━━ Build a standard shuffled DataLoader ━━━━━━━━━━┓
        train_loader = DataLoader(Subset(ds, idx_train),
                                  batch_size = batch_size,
                                  shuffle    = True,
                                  generator  = torch.Generator().manual_seed(_seed))

        

        # ┏━━━━━━━━━━ Build both criteria, then pick the one for target ━━━━━━━━━━┓
        crit_up, crit_dn = make_criteria(loss_type, 
                                         w_up, 
                                         w_dn, 
                                         device, 
                                         focal_gamma, 
                                         focal_alpha)

        criterion = crit_up if target.upper() == 'UP' else crit_dn

        # ┏━━━━━━━━━━ Append datasets and criterion ━━━━━━━━━━┓
        folds.append((train_loader, val_loader_outer, criterion))

    else:
        # ┏━━━━━━━━━━ Multiple Folds (CV) ━━━━━━━━━━┓
        p_start = props 
        n_pool = n_train + n_val
        train_end = int(n_pool * p_start)
        val_window = n_val

        while train_end + val_window <= n_pool:
            idx_train_fold = list(range(0, train_end))
            idx_val_fold   = list(range(train_end, train_end + val_window))

            # ┏━━━━━━━━━━ Sanity Check ━━━━━━━━━━┓
            assert max(idx_train_fold) <= min(idx_val_fold), "Train/Val overlap!"
            assert max(idx_val_fold)   < n_pool, "Val fold exceeds pool boundary!"

            # ┏━━━━━━━━━━ Extract targets on Train only ━━━━━━━━━━┓
            y_up_tr = ds.tensors[1][idx_train_fold]
            y_dn_tr = ds.tensors[2][idx_train_fold]

            # ┏━━━━━━━━━━ Compute class-weights on TRAIN Fold only ━━━━━━━━━━┓
            if loss_type in ('cross_entropy', 'focal'):
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
                raise ValueError("loss_type must be 'bce', 'cross_entropy' or 'focal")

            w_up = w_up.to(device)
            w_dn = w_dn.to(device)

            # ┏━━━━━━━━━━ Standard shuffled DataLoader for this fold ━━━━━━━━━━┓
            train_loader = DataLoader(Subset(ds, idx_train_fold),
                                      batch_size = batch_size,
                                      shuffle    = True,
                                      generator  = torch.Generator().manual_seed(_seed))
            
            val_loader   = DataLoader(Subset(ds, idx_val_fold),
                                      batch_size = batch_size,
                                      shuffle    = False)

            # ┏━━━━━━━━━━ Build both criteria, then pick the one for target ━━━━━━━━━━┓
            crit_up, crit_dn = make_criteria(loss_type, 
                                             w_up, 
                                             w_dn, 
                                             device, 
                                             focal_gamma, 
                                             focal_alpha)

            criterion = crit_up if target.upper() == 'UP' else crit_dn

            # ┏━━━━━━━━━━ Append datasets and criterion ━━━━━━━━━━┓
            folds.append((train_loader, val_loader, criterion))

            # ┏━━━━━━━━━━ Expand training window ━━━━━━━━━━┓
            train_end += val_window 
        
        # ┏━━━━━━━━━━ Handle leftover samples in the pool ━━━━━━━━━━┓
        if train_end < n_pool:
            # ┏━━━━━━━━━━ Last fold should mirror original split sizes ━━━━━━━━━━┓
            idx_tr = list(range(0, n_train))
            idx_va = list(range(n_train, n_train + n_val))

            # ┏━━━━━━━━━━ Extract targets on last train set ━━━━━━━━━━┓
            y_up_tr = ds.tensors[1][idx_tr]
            y_dn_tr = ds.tensors[2][idx_tr]

            # ┏━━━━━━━━━━ Compute class-weights on TRAIN Fold only ━━━━━━━━━━┓
            if loss_type in ('cross_entropy', 'focal'):
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
                raise ValueError("loss_type must be 'bce', 'cross_entropy' or 'focal")

            w_up = w_up.to(device)
            w_dn = w_dn.to(device)

            # ┏━━━━━━━━━━ Standard shuffled DataLoader for last fold ━━━━━━━━━━┓
            train_loader = DataLoader(Subset(ds, idx_tr),
                                      batch_size = batch_size,
                                      shuffle    = True,
                                      generator  = torch.Generator().manual_seed(_seed))

            val_loader   = DataLoader(Subset(ds, idx_va),
                                      batch_size = batch_size,
                                      shuffle    = False)

            # ┏━━━━━━━━━━ Build both criteria, then pick the one for target ━━━━━━━━━━┓
            crit_up, crit_dn = make_criteria(loss_type, 
                                             w_up, 
                                             w_dn, 
                                             device, 
                                             focal_gamma, 
                                             focal_alpha)

            criterion = crit_up if target.upper() == 'UP' else crit_dn

            # ┏━━━━━━━━━━ Append datasets and criterion ━━━━━━━━━━┓
            folds.append((train_loader, val_loader, criterion))

    return folds, test_loader