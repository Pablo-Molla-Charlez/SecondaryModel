import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
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
    fbeta = fbeta_score(targets, preds, beta = 1.5, zero_division = 0)

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