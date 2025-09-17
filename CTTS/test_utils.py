import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from pathlib import Path
from paths import dataset_path
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    fbeta_score
)


def get_preds_and_targets(model: torch.nn.Module,
                          loader: torch.utils.data.DataLoader,
                          device: torch.device,
                          task_name,
                          loss_type
    ):
    """
    Run 'model' on 'loader' and return (preds, targets) numpy lists.
    - BCE mode: logits → sigmoid → thresh 0.5 → preds in {0,1}, targets float
    - cross_entropy mode: logits → softmax → argmax → preds in {0…C-1}, targets long
    """
    model.eval()
    is_bce = loss_type == "bce"
    all_preds, all_targets = [], []
    with torch.no_grad():
        for xb, y_up, y_dn in loader:
            xb = xb.to(device)
            # ┏━━━━━━━━━━ Select & cast target ━━━━━━━━━━┓
            y = (y_up if task_name == "UP" else y_dn).to(device)
            y = y.float() if is_bce else y.long()

            logits = model(xb)
            # ┏━━━━━━━━━━ Binary Cross-Entropy ━━━━━━━━━━┓
            if is_bce:
                # ┏━━━━━━━━━━ Assume logits shape [B] or [B,1] ━━━━━━━━━━┓
                probs = torch.sigmoid(logits.squeeze(1))
                preds = (probs > 0.5).long()
            
            # ┏━━━━━━━━━━ Cross-Entropy or Focal ━━━━━━━━━━┓
            else:
                preds = logits.softmax(dim=1).argmax(dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(y.cpu().numpy())

    return np.array(all_preds), np.array(all_targets)


def plot_cm_with_metrics(preds, targets, labels, title, out_dir, cmap="Oranges"):
    """
    preds, targets : array-like of shape (n_samples,)
    labels         : tuple of display labels, e.g. ("No_TP","TP")
    title          : str, e.g. "UP — Test"
    out_dir        : pathlib.Path or str where to save the PNG
    cmap           : Matplotlib colormap name
    """
    # ┏━━━━━━━━━━ 1) Compute metrics ━━━━━━━━━━┓
    cm    = confusion_matrix(targets, preds)
    acc   = accuracy_score(targets, preds)
    prec  = precision_score(targets, preds, zero_division = 0)
    rec   = recall_score(targets, preds)
    f1    = f1_score(targets, preds)
    fbeta = fbeta_score(targets, preds, beta = 0.9, zero_division = 0)

    # ┏━━━━━━━━━━  2) Plot ━━━━━━━━━━┓
    fig, ax = plt.subplots(figsize=(4, 4))
    disp = ConfusionMatrixDisplay(cm, display_labels=labels)
    disp.plot(cmap=cmap, ax=ax, colorbar=False)
    ax.set_title(title)

    # ┏━━━━━━━━━━  3) Annotate metrics ━━━━━━━━━━┓
    textstr = (
        f"Accuracy : {acc:.2f}\n"
        f"Precision: {prec:.2f}\n"
        f"Recall   : {rec:.2f}\n"
        f"F1 Score : {f1:.2f}\n"
        f"F-Beta-Score: {fbeta:.2f}"
    )
    ax.text(
        1.05, 0.6, textstr,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='white', edgecolor='gray')
    )
    plt.tight_layout()

    # ┏━━━━━━━━━━  4) Save to disk ━━━━━━━━━━┓
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    # ┏━━━━━━━━━━ 5) Sanitize filename ━━━━━━━━━━┓
    fname = title.replace(" — ", "_") + ".png"
    fig.savefig(out_path / fname, dpi=150)
    plt.close(fig)

    print(f"Saved confusion matrix to {out_path / fname}")


def export_predictions(df_asset,
                       dataset_tensor,
                       tpreds: np.ndarray,
                       cfg: dict,
                       checkpoint_dir: Path,
                       tprobs: np.ndarray = None) -> Path:
    """
    Export a CSV with test-timeline dates aligned to CTTS predictions.

    Parameters
    - df_asset: DataFrame returned by merge_meta_targets (indexed by 'date').
    - dataset_tensor: TensorDataset used to build loaders (len(...) equals number of windows).
    - tpreds: numpy array of test predictions (ordered as test_loader iteration).
    - cfg: full training config dict (reads 'splits', 'sequence_length', 'dataset', 'training_mode').
    - checkpoint_dir: Path where to save the CSV.

    Returns
    - Path to the written CSV.
    """
    # ┏━━━━━━━━━━ Recompute split sizes to recover test indices ━━━━━━━━━━┓
    N_windows   = len(dataset_tensor)
    n_train_win = int(cfg["splits"]["train"] * N_windows)
    n_val_win   = int(cfg["splits"]["val"]   * N_windows)
    idx_test_ds = list(range(n_train_win + n_val_win, N_windows))

    # ┏━━━━━━━━━━ Map each dataset index k to its corresponding end-of-window date: df.index[k + seq_len - 1] ━━━━━━━━━━┓
    seq_len = cfg["sequence_length"]
    all_dates = df_asset.index
    mapped_dates = [all_dates[k + seq_len - 1] for k in idx_test_ds]

    # ┏━━━━━━━━━━ Sanity check ━━━━━━━━━━┓
    if len(mapped_dates) != len(tpreds):
        print(f"[WARN] Date/predictions length mismatch: dates={len(mapped_dates)} preds={len(tpreds)}")

    # ┏━━━━━━━━━━ Start with DataFrame of test dates ━━━━━━━━━━┓
    base_df = pd.DataFrame({
        "date": mapped_dates[:len(tpreds)]
    })

    # ┏━━━━━━━━━━ Load M1 CSVs (UP and DOWN) ━━━━━━━━━━┓
    provider = cfg["dataset"]["source"]
    market   = cfg["dataset"]["type"].capitalize()
    symbol   = cfg["dataset"]["symbol"]
    up_path  = dataset_path(provider, market, symbol, "up")
    dn_path  = dataset_path(provider, market, symbol, "down")

    up_df = pd.read_csv(up_path, parse_dates=["date"]) if up_path.exists() else pd.DataFrame(columns=["date"])
    dn_df = pd.read_csv(dn_path, parse_dates=["date"]) if dn_path.exists() else pd.DataFrame(columns=["date"])

    # ┏━━━━━━━━━━ Select and rename necessary columns ━━━━━━━━━━┓
    up_keep = [c for c in ["date", "prediction", "ground_truth", "pred", "pred_proba", "lab", "meta_target", "meta_label"] if c in up_df.columns]
    dn_keep = [c for c in ["date", "prediction", "ground_truth", "pred", "pred_proba", "lab", "meta_target", "meta_label"] if c in dn_df.columns]
    up_sel = up_df.loc[:, up_keep].rename(columns={
        "pred":        "m1_pred_up",
        "pred_proba":  "m1_pred_proba_up",
        "meta_target": "meta_label_up",
        "meta_label":  "meta_label_up",
        "lab":         "lab_up",
        "ground_truth": "ground_truth_up",
        "prediction":  "numerical_prediction_up",
    }) if len(up_keep) else pd.DataFrame(columns=["date"]) 

    dn_sel = dn_df.loc[:, dn_keep].rename(columns={
        "pred":        "m1_pred_down",
        "pred_proba":  "m1_pred_proba_dn",
        "meta_target": "meta_label_dn",
        "meta_label":  "meta_label_dn",
        "lab":         "lab_dn",
        "ground_truth":"ground_truth_dn",
        "prediction":  "numerical_prediction_dn",
    }) if len(dn_keep) else pd.DataFrame(columns=["date"]) 

    # ┏━━━━━━━━━━ Merge on test dates ━━━━━━━━━━┓
    merged = (
        base_df
        .merge(up_sel, on="date", how="left")
        .merge(dn_sel, on="date", how="left")
    )

    # ┏━━━━━━━━━━ Coalesce 'prediction' and 'ground_truth' ━━━━━━━━━━┓
    # ┏━━━━━━━━━━ Task-specific columns ━━━━━━━━━━┓
    task = cfg["training_mode"]["optuna_task"].upper()
    task_lower = task.lower()

    pred_series_up = merged.get("numerical_prediction_up")
    pred_series_dn = merged.get("numerical_prediction_dn")
    gt_series_up   = merged.get("ground_truth_up")
    gt_series_dn   = merged.get("ground_truth_dn")

    task_pred_col = f"numerical_prediction_{task_lower}"
    task_gt_col   = f"ground_truth_{task_lower}"

    # Use the series that matches the active task, fall back to whichever exists.
    if isinstance(merged.get(task_pred_col), pd.Series):
        merged["prediction"] = merged[task_pred_col]
    elif isinstance(pred_series_up, pd.Series) or isinstance(pred_series_dn, pd.Series):
        merged["prediction"] = (
            (pred_series_up if isinstance(pred_series_up, pd.Series) else pd.Series([pd.NA] * len(merged), index=merged.index))
            .combine_first(pred_series_dn if isinstance(pred_series_dn, pd.Series) else pd.Series([pd.NA] * len(merged), index=merged.index))
        )
    else:
        merged["prediction"] = pd.Series([pd.NA] * len(merged), index=merged.index)

    if isinstance(merged.get(task_gt_col), pd.Series):
        merged["ground_truth"] = merged[task_gt_col]
    elif isinstance(gt_series_up, pd.Series) or isinstance(gt_series_dn, pd.Series):
        merged["ground_truth"] = (
            (gt_series_up if isinstance(gt_series_up, pd.Series) else pd.Series([pd.NA] * len(merged), index=merged.index))
            .combine_first(gt_series_dn if isinstance(gt_series_dn, pd.Series) else pd.Series([pd.NA] * len(merged), index=merged.index))
        )
    else:
        merged["ground_truth"] = pd.Series([pd.NA] * len(merged), index=merged.index)

    if task == "UP":
        merged["meta_label"] = merged.get("meta_label_up")
        merged["lab"]        = merged.get("lab_up")
    else:
        merged["meta_label"] = merged.get("meta_label_dn")
        merged["lab"]        = merged.get("lab_dn")

    # ┏━━━━━━━━━━ Include CTTS (M2) discrete predictions ━━━━━━━━━━┓
    pred_col = f"m2_pred_{task_lower}"
    merged[pred_col] = pd.Series(np.asarray(tpreds).astype(int), index=merged.index)
    # Optionally include M2 probability for the active task
    if tprobs is not None:
        prob_col = f"m2_prob_{task_lower}"
        merged[prob_col] = pd.Series(tprobs, index=merged.index)

    # ┏━━━━━━━━━━ Final column set (at minimum) ━━━━━━━━━━┓
    final_cols = [
        "date",
        "prediction",
        "m1_pred_up",
        "m1_pred_down",
        "m1_pred_proba_up",
        "m1_pred_proba_dn",
        "meta_label",
        "ground_truth",
        "lab",
        pred_col,
    ]
    # Include probability column for this task only if present
    prob_col = f"m2_prob_{task_lower}"
    if prob_col in merged.columns:
        final_cols.append(prob_col)
    for col in final_cols:
        if col not in merged.columns:
            merged[col] = pd.NA

    out_df = merged[final_cols].copy()
    out_df["date"] = pd.to_datetime(out_df["date"], errors="coerce").dt.strftime("%Y-%m-%d")

    # ┏━━━━━━━━━━ Write with requested naming ━━━━━━━━━━┓
    out_path = checkpoint_dir / f"{symbol}_{task}_predictions.csv"
    out_df.to_csv(out_path, index=False)
    print(f"Saved enriched predictions to {out_path}")
    return out_path


def get_preds_probs(model: torch.nn.Module,
                    loader: torch.utils.data.DataLoader,
                    device: torch.device,
                    task_name: str,
                    loss_type: str):
    """
    Run 'model' on 'loader' and return (preds, probs, targets) as numpy arrays.
    - BCE: probs = sigmoid(logits) in [0,1]
    - cross_entropy/focal: probs = softmax(logits)[:, 1] (prob of class 1)
    """
    model.eval()
    is_bce = loss_type == "bce"
    all_preds, all_probs, all_targets = [], [], []
    task_name = task_name.upper()

    with torch.no_grad():
        for xb, y_up, y_dn in loader:
            xb = xb.to(device)

            # Select appropriate target depending on active task
            y = y_up if task_name == "UP" else y_dn
            y = y.to(device)
            y = y.float() if is_bce else y.long()

            logits = model(xb)
            if is_bce:
                # logits shape [B] or [B,1]
                logits = logits.squeeze(1)
                probs  = torch.sigmoid(logits)
                preds  = (probs > 0.5).long()
            else:
                sm    = torch.softmax(logits, dim=1)
                probs = sm[:, 1]              # prob of positive class
                preds = sm.argmax(dim=1).long()

            all_probs.extend(probs.detach().cpu().numpy().tolist())
            all_preds.extend(preds.detach().cpu().numpy().tolist())
            all_targets.extend(y.detach().cpu().numpy().tolist())

    target_dtype = np.float32 if is_bce else np.int64
    return (
        np.array(all_preds),
        np.array(all_probs, dtype=np.float32),
        np.array(all_targets, dtype=target_dtype),
    )


def plot_meta_labeling_consensus(cfg: dict,
                                 checkpoint_dir: Path) -> Path:
    """
    Build a consensus confusion matrix using the exported predictions CSV and save
    it alongside the CSV as 'meta_labeling_results.png'. Consensus equals 1 only
    when the M1 prediction and CTTS prediction agree on 1 for the active task.
    """
    task = cfg["training_mode"]["optuna_task"].upper()
    symbol = cfg["dataset"]["symbol"]
    csv_path = Path(checkpoint_dir) / f"{symbol}_{task}_predictions.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Predictions CSV not found at {csv_path}")

    df = pd.read_csv(csv_path)
    task_lower = task.lower()
    m1_suffix = "down" if task == "DN" else "up"
    m1_col = f"m1_pred_{m1_suffix}"
    m2_col = f"m2_pred_{task_lower}"

    required_cols = {m1_col, m2_col, "lab"}
    missing = required_cols - set(df.columns)
    if missing:
        raise KeyError(f"Missing required columns in predictions CSV: {sorted(missing)}")

    m1_preds = df[m1_col].fillna(0).astype(int)
    m2_preds = df[m2_col].fillna(0).astype(int)
    targets  = df["lab"].fillna(0).astype(int)

    consensus_preds = ((m1_preds == 1) & (m2_preds == 1)).astype(int)

    labels = (f"No_TP_{task}", f"TP_{task}")
    cm = confusion_matrix(targets, consensus_preds, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(4, 4))
    disp = ConfusionMatrixDisplay(cm, display_labels=labels)
    disp.plot(cmap="Purples", ax=ax, colorbar=False)
    ax.set_title(f"{task} — Meta-Label Consensus")
    ax.set_xlabel("Predicted Consensus")
    ax.set_ylabel("True Label")

    acc  = accuracy_score(targets, consensus_preds)
    prec = precision_score(targets, consensus_preds, zero_division=0)
    rec  = recall_score(targets, consensus_preds, zero_division=0)
    f1   = f1_score(targets, consensus_preds, zero_division=0)
    textstr = (
        f"Accuracy : {acc:.2f}\n"
        f"Precision: {prec:.2f}\n"
        f"Recall   : {rec:.2f}\n"
        f"F1 Score : {f1:.2f}"
    )
    ax.text(
        1.05, 0.6, textstr,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='white', edgecolor='gray')
    )
    fig.tight_layout()

    out_path = csv_path.parent / "Meta_Labeling_Results.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved meta-labeling consensus confusion matrix to {out_path}")
    return out_path
