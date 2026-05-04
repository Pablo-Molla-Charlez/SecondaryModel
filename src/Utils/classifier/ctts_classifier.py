"""CTTSClassifier — BaseClassifier-compliant wrapper around CTTSModel.

Mirrors the TabMClassifier pattern: a self-contained training loop with
AdamW + early stopping, batched inference, and save/load via state_dict.

The CTTS architecture (CNN → Transformer → Attention-pooled MLP head) is
imported from a local copy at ``Utils.classifier.ctts.model`` to avoid
import-path fragility with the external Meta-Labeling-CTTS repository.

Input contract:
    ``X_train`` has shape ``(N, seq_len)`` — the output of
    ``extract_close_windows()`` in ``ctts_features.py``.
    Internally reshaped to ``(N, 1, seq_len)`` for CTTSModel.forward().
"""

from __future__ import annotations

import math
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


class CTTSClassifier(BaseClassifier):
    """Sklearn-compatible wrapper around the CTTS time-series classifier.

    Default hyperparameters match the CTTS paper configuration.

    Parameters
    ----------
    seq_len : int
        Context window length.  Must match the ``seq_len`` used in
        ``extract_close_windows()``.  Granularity-adaptive: callers pass
        ``min(90, GRAN_SEQ_LEN[gran])`` so shorter granularities avoid
        excessive zero-padding.
    """

    def __init__(self,
                 seq_len:         int   = 90,
                 cnn_embed_dim:   list  = None,
                 cnn_kernel:      list  = None,
                 cnn_stride:      list  = None,
                 p_pos_drop:      float = 0.1,
                 trans_heads:     int   = 4,
                 trans_layers:    int   = 2,
                 trans_ff:        int   = 256,
                 trans_dropout:   float = 0.1,
                 trans_activ:     str   = "gelu",
                 mlp_hidden:      int   = 128,
                 mlp_dropout:     float = 0.1,
                 mlp_activ:       str   = "gelu",
                 mlp_pooling:     str   = "attention",
                 num_classes:     int   = 2,
                 padding:         bool  = True,
                 lr:              float = 1e-4,
                 weight_decay:    float = 1e-4,
                 max_epochs:      int   = 100,
                 patience:        int   = 15,
                 batch_size:      int   = 128,
                 val_fraction:    float = 0.25,
                 random_state:    int   = 42,
                 device:          str   = "cuda") -> None:
        super().__init__(random_state)
        self.seq_len       = seq_len
        self.cnn_embed_dim = cnn_embed_dim if cnn_embed_dim is not None else [64, 128]
        self.cnn_kernel    = cnn_kernel    if cnn_kernel    is not None else [7, 5]
        self.cnn_stride    = cnn_stride    if cnn_stride    is not None else [2, 1]
        self.p_pos_drop    = p_pos_drop
        self.trans_heads   = trans_heads
        self.trans_layers  = trans_layers
        self.trans_ff      = trans_ff
        self.trans_dropout = trans_dropout
        self.trans_activ   = trans_activ
        self.mlp_hidden    = mlp_hidden
        self.mlp_dropout   = mlp_dropout
        self.mlp_activ     = mlp_activ
        self.mlp_pooling   = mlp_pooling
        self.num_classes   = num_classes
        self.padding       = padding
        self.lr            = lr
        self.weight_decay  = weight_decay
        self.max_epochs    = max_epochs
        self.patience      = patience
        self.batch_size    = batch_size
        self.val_fraction  = val_fraction
        self.random_state  = random_state
        self.device        = _resolve_device(device)

        self._model      = None
        self._best_state = None
        self.classes_    = None

    # ┏━━━━━━━━━━ Build CTTSModel ━━━━━━━━━━┓
    def _build_model(self):
        from Utils.classifier.ctts.model import CTTSModel
        return CTTSModel(cnn_embed_dim = list(self.cnn_embed_dim),
                         cnn_kernel    = list(self.cnn_kernel),
                         cnn_stride    = list(self.cnn_stride),
                         p_pos_drop    = float(self.p_pos_drop),
                         nb_features   = 1,
                         trans_heads   = int(self.trans_heads),
                         trans_layers  = int(self.trans_layers),
                         trans_ff      = int(self.trans_ff),
                         trans_dropout = float(self.trans_dropout),
                         trans_activ   = str(self.trans_activ),
                         mlp_hidden    = int(self.mlp_hidden),
                         mlp_dropout   = float(self.mlp_dropout),
                         mlp_activ     = str(self.mlp_activ),
                         mlp_pooling   = str(self.mlp_pooling),
                         num_classes   = int(self.num_classes),
                         padding       = bool(self.padding),
                         context_len   = int(self.seq_len))

    # ┏━━━━━━━━━━ Fit ━━━━━━━━━━┓
    def fit(self,
            X_train:      Union[np.ndarray, pd.DataFrame],
            y_train:      np.ndarray,
            sample_weight = None,
            X_eval:       Optional[Union[np.ndarray, pd.DataFrame]] = None,
            y_eval:       Optional[np.ndarray] = None):
        """Fit CTTS with AdamW + early stopping.

        Parameters
        ----------
        X_train : ndarray of shape (N, seq_len)
            MinMax-scaled close windows from ``extract_close_windows()``.
        y_train : ndarray of shape (N,)
            Binary labels.
        X_eval, y_eval : optional
            When provided, used as the early-stopping signal (val accuracy).
            Otherwise, the final ``val_fraction`` chronological slice of
            X_train is used.
        """
        import torch

        # ┏━━━━━━━━━━ Preprocessing ━━━━━━━━━━┓
        X = np.asarray(X_train, dtype=np.float32)
        y = np.asarray(y_train).astype(np.int64)
        self.n_features_in_ = int(X.shape[1])
        self.classes_        = np.unique(y)

        # ┏━━━━━━━━━━ Resolve early-stopping window ━━━━━━━━━━┓
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

        # ┏━━━━━━━━━━ Reshape to (N, 1, seq_len) ━━━━━━━━━━┓
        X_tr_t = torch.from_numpy(X_tr[:, np.newaxis, :]).to(self.device)
        y_tr_t = torch.from_numpy(y_tr).to(self.device)
        X_va_t = torch.from_numpy(X_va[:, np.newaxis, :]).to(self.device)
        y_va_t = torch.from_numpy(y_va).to(self.device)

        # ┏━━━━━━━━━━ Build model ━━━━━━━━━━┓
        torch.manual_seed(self.random_state)
        torch.cuda.manual_seed_all(self.random_state)
        self._model = self._build_model().to(self.device)

        # ┏━━━━━━━━━━ Optimizer + loss ━━━━━━━━━━┓
        optim = torch.optim.AdamW(self._model.parameters(),
                                  lr=float(self.lr),
                                  weight_decay=float(self.weight_decay))
        loss_fn = torch.nn.CrossEntropyLoss()

        # ┏━━━━━━━━━━ Training loop with early stopping ━━━━━━━━━━┓
        best_val_acc = -1.0
        epochs_no_improve = 0
        n_train = X_tr_t.shape[0]
        bs = int(self.batch_size)
        steps_per_epoch = max(1, math.ceil(n_train / bs))

        for epoch in range(int(self.max_epochs)):
            # ┏━━━━━━━━━━ Train ━━━━━━━━━━┓
            self._model.train()
            perm = torch.randperm(n_train, device=self.device)
            for s in range(steps_per_epoch):
                lo = s * bs
                hi = min(n_train, lo + bs)
                bidx = perm[lo:hi]
                xb = X_tr_t[bidx]
                yb = y_tr_t[bidx]
                optim.zero_grad(set_to_none=True)
                logits = self._model(xb)          # (B, num_classes)
                loss = loss_fn(logits, yb)
                loss.backward()
                optim.step()

            # ┏━━━━━━━━━━ Validation eval ━━━━━━━━━━┓
            self._model.eval()
            with torch.no_grad():
                v_logits = self._model(X_va_t)
                v_pred = v_logits.argmax(dim=-1)
                v_acc = (v_pred == y_va_t).float().mean().item()

            # ┏━━━━━━━━━━ Early stopping ━━━━━━━━━━┓
            if v_acc > best_val_acc + 1e-6:
                best_val_acc = v_acc
                epochs_no_improve = 0
                self._best_state = {k: v.detach().cpu().clone()
                                    for k, v in self._model.state_dict().items()}
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= int(self.patience):
                    break

        # ┏━━━━━━━━━━ Restore best weights ━━━━━━━━━━┓
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
        x = torch.from_numpy(X[:, np.newaxis, :]).to(self.device)  # (N, 1, seq_len)
        self._model.eval()
        bs = max(1024, int(self.batch_size))
        chunks = []
        with torch.no_grad():
            for i in range(0, x.shape[0], bs):
                xb = x[i:i + bs]
                logits = self._model(xb)                    # (B, num_classes)
                probs = torch.softmax(logits, dim=-1)
                chunks.append(probs.cpu().numpy())
        return np.concatenate(chunks, axis=0).astype(np.float64)

    # ┏━━━━━━━━━━ predict ━━━━━━━━━━┓
    def predict(self, X_test: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        probs = self.predict_proba(X_test)
        idx = probs.argmax(axis=1)
        if self.classes_ is None:
            return idx
        return self.classes_[idx]

    # ┏━━━━━━━━━━ Feature importance (uniform — CTTS has no feature importance) ━━━━━━━━━━┓
    @property
    def feature_importances_(self):
        n_feat = getattr(self, "n_features_in_", 1)
        return np.ones(n_feat) / max(1, n_feat)

    # ┏━━━━━━━━━━ get_params ━━━━━━━━━━┓
    def get_params(self, deep: bool = True) -> dict:
        return {"seq_len":       self.seq_len,
                "cnn_embed_dim": list(self.cnn_embed_dim),
                "cnn_kernel":    list(self.cnn_kernel),
                "cnn_stride":    list(self.cnn_stride),
                "p_pos_drop":    self.p_pos_drop,
                "trans_heads":   self.trans_heads,
                "trans_layers":  self.trans_layers,
                "trans_ff":      self.trans_ff,
                "trans_dropout": self.trans_dropout,
                "trans_activ":   self.trans_activ,
                "mlp_hidden":    self.mlp_hidden,
                "mlp_dropout":   self.mlp_dropout,
                "mlp_activ":     self.mlp_activ,
                "mlp_pooling":   self.mlp_pooling,
                "num_classes":   self.num_classes,
                "padding":       self.padding,
                "lr":            self.lr,
                "weight_decay":  self.weight_decay,
                "max_epochs":    self.max_epochs,
                "patience":      self.patience,
                "batch_size":    self.batch_size,
                "val_fraction":  self.val_fraction,
                "random_state":  self.random_state,
                "device":        self.device}

    # ┏━━━━━━━━━━ Save Model ━━━━━━━━━━┓
    def save_model(self, model_path: str) -> None:
        if self._model is None:
            raise AttributeError("The model has not been fitted yet.")
        import torch
        path = f"{model_path}.pt"
        torch.save({"state_dict": self._model.state_dict(),
                    "params":     self.get_params(),
                    "classes_":   None if self.classes_ is None else self.classes_.tolist()}, path)

    # ┏━━━━━━━━━━ Load Model ━━━━━━━━━━┓
    def load_model(self, model_path: str) -> None:
        import torch
        path = f"{model_path}.pt"
        payload = torch.load(path, map_location=self.device, weights_only=False)
        params = payload["params"]
        for k, v in params.items():
            setattr(self, k, v)
        self.classes_ = (np.array(payload["classes_"])
                         if payload.get("classes_") is not None else None)

        # ┏━━━━━━━━━━ Rebuild model skeleton and load weights ━━━━━━━━━━┓
        self._model = self._build_model().to(self.device)
        self._model.load_state_dict(payload["state_dict"])
        self._model.eval()
