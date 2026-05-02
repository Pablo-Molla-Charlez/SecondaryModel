"""TabM classifier — BaseClassifier-compliant wrapper.

TabM (Gorishniy et al., ICLR 2025) is a parameter-efficient ensemble of MLPs
("tabular MLP-Mixer") that is the current state-of-the-art deep-learning
baseline on standard tabular benchmarks. We use it here as the deep-learning
M2 baseline against which the tabular foundation models (TabPFN, TabICL),
classical Random Forest, and AutoGluon are compared.

Reference: https://github.com/yandex-research/tabm

Implementation notes
--------------------
The TabM upstream package exposes a ``torch.nn.Module``; there is no
sklearn-compatible ``.fit()``. We therefore wrap a complete training loop
(AdamW + early stopping) here.

The k ensemble members must be trained *independently* — i.e. the loss is the
mean of per-member cross-entropies, NOT the cross-entropy of the averaged
logits. This is the key prescription in the TabM paper and is honoured below.

At inference the per-member softmaxes are averaged.

Optionally a ``PiecewiseLinearEmbeddings`` numerical embedding (configurable
via ``n_bins`` and ``d_embedding``) is enabled to match the second TabM
variant in the paper.
"""

from __future__ import annotations

import math
import os
import pickle
import numpy as np
import pandas as pd
from typing import Optional, Union
from Utils.classifier._classifier import BaseClassifier

# ┏━━━━━━━━━━ Resolve Device ━━━━━━━━━━┓
def _resolve_device(prefer: str = "cuda") -> str:
    import torch
    if prefer == "cuda" and torch.cuda.is_available():
        return "cuda"
    return "cpu"


class TabMClassifier(BaseClassifier):
    """Sklearn-compatible wrapper around the TabM tabular DL baseline.

    Parameters mirror the search space proposed in the TabM paper / GitHub:

        - k             : number of MLP ensemble members  (paper default: 32)
        - n_blocks      : MLP depth                       (1..5)
        - d_block       : MLP width                       (64..1024 step 16)
        - lr            : AdamW learning rate             (1e-4..5e-3 log)
        - weight_decay  : AdamW weight decay              ({0} ∪ [1e-4..1e-1] log)
        - n_bins        : PLR embedding bins              (None → no embedding)
        - d_embedding   : PLR embedding dim per feature   (8..32 step 4)

    Training:
        - AdamW + cross-entropy
        - Early stopping on a held-out 15% slice of the training set
          (random split, fixed seed) using val accuracy
        - Patience = 16 epochs, max 200 epochs, batch size 256
    """

    def __init__(self,
                 k:             int   = 32,
                 n_blocks:      int   = 2,
                 d_block:       int   = 256,
                 lr:            float = 2e-3,
                 weight_decay:  float = 3e-4,
                 dropout:       float = 0.1,
                 arch_type:     str   = "tabm",
                 n_bins:        Optional[int] = None,
                 d_embedding:   Optional[int] = None,
                 batch_size:    int   = 256,
                 max_epochs:    int   = 200,
                 patience:      int   = 16,
                 val_fraction:  float = 0.25,
                 random_state:  int   = 42,
                 device:        str   = "cuda") -> None:
        super().__init__(random_state)
        self.k             = k
        self.n_blocks      = n_blocks
        self.d_block       = d_block
        self.lr            = lr
        self.weight_decay  = weight_decay
        self.dropout       = dropout
        self.arch_type     = arch_type
        self.n_bins        = n_bins
        self.d_embedding   = d_embedding
        self.batch_size    = batch_size
        self.max_epochs    = max_epochs
        self.patience      = patience
        self.val_fraction  = val_fraction
        self.random_state  = random_state
        self.device        = _resolve_device(device)

        self._model        = None
        self._n_features   = None
        self._n_classes    = None
        self.classes_      = None
        self._best_state   = None  # state_dict at best val epoch

    # ┏━━━━━━━━━━ Fit ━━━━━━━━━━┓
    def fit(self,
            X_train:      Union[np.ndarray, pd.DataFrame],
            y_train:      np.ndarray,
            sample_weight = None,
            X_eval:       Optional[Union[np.ndarray, pd.DataFrame]] = None,
            y_eval:       Optional[np.ndarray] = None):
        """Fit TabM with AdamW + early stopping.

        Parameters
        ----------
        X_train, y_train
            Training features and binary labels. The full window is used for
            gradient updates.
        X_eval, y_eval
            Optional held-out window used as the early-stopping signal.
            When provided, training stops when val accuracy plateaus and the
            weights from the best epoch are restored. The pipeline passes
            the same chronological tail used downstream by the threshold
            optimiser, matching the protocol applied to AutoGluon.
        sample_weight
            Unused — accepted only for sklearn-API compatibility.

        Fallback when ``X_eval`` is None
        --------------------------------
        Take the final ``val_fraction`` slice of ``X_train`` chronologically
        (no shuffle) so no random row leaks into the early-stopping signal.
        """
        import torch
        from tabm import TabM

        # ┏━━━━━━━━━━ Preprocessing ━━━━━━━━━━┓
        X = np.asarray(X_train, dtype=np.float32)
        y = np.asarray(y_train).astype(np.int64)

        self._n_features    = int(X.shape[1])
        self.n_features_in_ = self._n_features
        self.classes_       = np.unique(y)
        self._n_classes     = int(self.classes_.size)

        # ┏━━━━━━━━━━ Optional PiecewiseLinearEmbeddings ━━━━━━━━━━┓
        num_embeddings = None
        if self.n_bins is not None and self.d_embedding is not None:
            from rtdl_num_embeddings import (PiecewiseLinearEmbeddings, compute_bins)
        
            # ┏━━━━━━━━━━ Compute bins ━━━━━━━━━━┓
            X_t = torch.from_numpy(X)
            bins = compute_bins(X_t, n_bins=int(self.n_bins))
            
            # ┏━━━━━━━━━━ Compute embeddings ━━━━━━━━━━┓
            num_embeddings = PiecewiseLinearEmbeddings(bins,
                                                       d_embedding = int(self.d_embedding),
                                                       activation  = False,
                                                       version     = "B")

        # ┏━━━━━━━━━━ Build TabM ━━━━━━━━━━┓
        torch.manual_seed(self.random_state)
        torch.cuda.manual_seed_all(self.random_state)
        self._model = TabM.make(n_num_features    = self._n_features,
                                cat_cardinalities = None,
                                d_out             = self._n_classes,
                                num_embeddings    = num_embeddings,
                                arch_type         = self.arch_type,
                                k                 = int(self.k),
                                n_blocks          = int(self.n_blocks),
                                d_block           = int(self.d_block),
                                dropout           = float(self.dropout)).to(self.device)

        # ┏━━━━━━━━━━ Resolve early-stopping window ━━━━━━━━━━┓
        # Priority 1: explicit (X_eval, y_eval) from the caller — this is the
        # merged Val window in Phase 0/1, or the 25% chronological opt slice
        # in Phase 2 (CPCV).  Train on the entire X_train when this is set.
        # Priority 2: chronological tail of X_train (size val_fraction).
        if X_eval is not None and y_eval is not None:
            X_tr, y_tr = X, y
            X_va = np.asarray(X_eval, dtype=np.float32)
            y_va = np.asarray(y_eval).astype(np.int64)
        else:
            N = X.shape[0]
            n_val = max(1, int(round(N * self.val_fraction)))
            cut = N - n_val
            X_tr, y_tr = X[:cut], y[:cut]
            X_va, y_va = X[cut:], y[cut:]

        X_tr_t = torch.from_numpy(X_tr).to(self.device)
        y_tr_t = torch.from_numpy(y_tr).to(self.device)
        X_va_t = torch.from_numpy(X_va).to(self.device)
        y_va_t = torch.from_numpy(y_va).to(self.device)

        # ┏━━━━━━━━━━ Optimizer ━━━━━━━━━━┓
        optim = torch.optim.AdamW(self._model.parameters(),
                                  lr           = float(self.lr),
                                  weight_decay = float(self.weight_decay))

        # ┏━━━━━━━━━━ Loss function ━━━━━━━━━━┓
        loss_fn = torch.nn.CrossEntropyLoss()

        # ┏━━━━━━━━━━ Training loop with early stopping ━━━━━━━━━━┓
        best_val_acc = -1.0
        best_epoch   = 0
        epochs_no_improve = 0
        n_train = X_tr_t.shape[0]
        bs = int(self.batch_size)
        steps_per_epoch = max(1, math.ceil(n_train / bs))

        # ┏━━━━━━━━━━ Loop through epochs ━━━━━━━━━━┓
        for epoch in range(int(self.max_epochs)):

            # ┏━━━━━━━━━━ Shuffle training data ━━━━━━━━━━┓
            self._model.train()
            perm = torch.randperm(n_train, device=self.device)
            
            # ┏━━━━━━━━━━ Train model with early stopping ━━━━━━━━━━┓
            for s in range(steps_per_epoch):
                lo = s * bs
                hi = min(n_train, lo + bs)
                bidx = perm[lo:hi]
                xb = X_tr_t[bidx]
                yb = y_tr_t[bidx]

                # ┏━━━━━━━━━━ Zero gradients ━━━━━━━━━━┓
                optim.zero_grad(set_to_none=True)

                # ┏━━━━━━━━━━ Forward pass ━━━━━━━━━━┓
                logits = self._model(x_num=xb)            # (B, k, C)
                B, k, C = logits.shape
                
                # ┏━━━━━━━━━━ Train k members independently — mean of per-member CE ━━━━━━━━━━┓
                loss = loss_fn(logits.reshape(B * k, C),
                               yb.repeat_interleave(k))

                # ┏━━━━━━━━━━ Backpropagation ━━━━━━━━━━┓
                loss.backward()
                optim.step()

            # ┏━━━━━━━━━━ Validation eval ━━━━━━━━━━┓
            self._model.eval()
            with torch.no_grad():
                v_logits = self._model(x_num=X_va_t)       # (Nv, k, C)
                v_probs  = torch.softmax(v_logits, dim=-1).mean(dim=1)
                v_pred   = v_probs.argmax(dim=-1)
                v_acc    = (v_pred == y_va_t).float().mean().item()

            # ┏━━━━━━━━━━ Update best model and patience ━━━━━━━━━━┓
            if v_acc > best_val_acc + 1e-6:
                best_val_acc = v_acc
                best_epoch   = epoch
                epochs_no_improve = 0
                
                # ┏━━━━━━━━━━ Snapshot model state on CPU to keep GPU memory bounded ━━━━━━━━━━┓
                self._best_state = {k_: v_.detach().cpu().clone()
                                    for k_, v_ in self._model.state_dict().items()}
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= int(self.patience):
                    break

        # ┏━━━━━━━━━━ Restore best epoch weights ━━━━━━━━━━┓
        if self._best_state is not None:
            self._model.load_state_dict(self._best_state)
        self._model.eval()
        return self

    # ┏━━━━━━━━━━ predict_proba ━━━━━━━━━━┓
    def predict_proba(self, X_test: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        if self._model is None:
            raise AttributeError("The model has not been fitted yet.")
        import torch
        X = np.asarray(X_test, dtype=np.float32)
        x = torch.from_numpy(X).to(self.device)
        self._model.eval()
        bs = max(1024, int(self.batch_size))
        chunks = []
        with torch.no_grad():
            for i in range(0, x.shape[0], bs):
                xb = x[i:i + bs]
                logits = self._model(x_num=xb)              # (B, k, C)
                probs  = torch.softmax(logits, dim=-1).mean(dim=1)
                chunks.append(probs.cpu().numpy())
        return np.concatenate(chunks, axis=0).astype(np.float64)

    # ┏━━━━━━━━━━ predict ━━━━━━━━━━┓
    def predict(self, X_test: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        probs = self.predict_proba(X_test)
        idx   = probs.argmax(axis=1)
        if self.classes_ is None:
            return idx
        return self.classes_[idx]

    # ┏━━━━━━━━━━ Feature importance (uniform fallback) ━━━━━━━━━━┓
    @property
    def feature_importances_(self):
        n_feat = getattr(self, "n_features_in_", 1)
        return np.ones(n_feat) / max(1, n_feat)

    # ┏━━━━━━━━━━ get_params ━━━━━━━━━━┓
    def get_params(self, deep: bool = True) -> dict:
        return {"k":            self.k,
                "n_blocks":     self.n_blocks,
                "d_block":      self.d_block,
                "lr":           self.lr,
                "weight_decay": self.weight_decay,
                "dropout":      self.dropout,
                "arch_type":    self.arch_type,
                "n_bins":       self.n_bins,
                "d_embedding":  self.d_embedding,
                "batch_size":   self.batch_size,
                "max_epochs":   self.max_epochs,
                "patience":     self.patience,
                "val_fraction": self.val_fraction,
                "random_state": self.random_state,
                "device":       self.device}

    # ┏━━━━━━━━━━ Save Model ━━━━━━━━━━┓
    def save_model(self, model_path: str) -> None:
        if self._model is None:
            raise AttributeError("The model has not been fitted yet.")
        import torch
        path = f"{model_path}.pt"
        torch.save({"state_dict":   self._model.state_dict(),
                    "params":       self.get_params(),
                    "n_features":   self._n_features,
                    "n_classes":    self._n_classes,
                    "classes_":     None if self.classes_ is None else self.classes_.tolist(),}, path)

    # ┏━━━━━━━━━━ Load Model ━━━━━━━━━━┓
    def load_model(self, model_path: str) -> None:
        import torch
        from tabm import TabM
        path = f"{model_path}.pt"
        payload = torch.load(path, map_location=self.device, weights_only=False)
        params = payload["params"]
        for k_, v_ in params.items(): setattr(self, k_, v_)
        self._n_features = payload["n_features"]
        self._n_classes  = payload["n_classes"]
        self.classes_    = (np.array(payload["classes_"])if payload.get("classes_") is not None else None)

        # ┏━━━━━━━━━━ Rebuild model skeleton ━━━━━━━━━━┓ 
        # embeddings are not persisted; on reload we disable them. 
        # Save/load is used for inference of the production model fitted by Phase 1, 
        # where Phase 2 (CPCV) refits anyway.
        self._model = TabM.make(n_num_features = self._n_features,
                                cat_cardinalities = None,
                                d_out          = self._n_classes,
                                num_embeddings = None,
                                arch_type      = self.arch_type,
                                k              = int(self.k),
                                n_blocks       = int(self.n_blocks),
                                d_block        = int(self.d_block),
                                dropout        = float(self.dropout)).to(self.device)

        self._model.load_state_dict(payload["state_dict"])
        self._model.eval()
