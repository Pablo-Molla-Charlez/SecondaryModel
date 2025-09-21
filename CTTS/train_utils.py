import os
import torch
import torch.nn as nn
import numpy as np
import random
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, fbeta_score
from torch.amp import GradScaler, autocast
from optim_utils import step_scheduler


def task_features(cfg: dict, prefix: str, task: str) -> list:
    """Return the feature list for the given task, falling back to shared entries."""
    key = f"{prefix}_{task.lower()}"
    if key in cfg:
        values = cfg[key]
    else:
        raise KeyError(f"Missing '{key}' in configuration for task '{task}'.")
    if not isinstance(values, list) or not values:
        raise ValueError(f"Configuration key '{key}' must be a non-empty list.")
    return values

# ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
# ┃ INITIALIZATION OF SEEDS                                               ┃
# ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
# ┃ EPOCH LOOPS: TRAINING & TESTING                                       ┃
# ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
def epoch_loop(model, 
               loader, 
               criterion, 
               device, 
               optimizer, 
               task_name,
               bce_thr,
               amp = True,
               clip_grad = 1.0,
               beta = 1.15
               ):
    """
    One epoch over 'loader' considering:
    - If 'optimizer' is given → training; otherwise → evaluation.
    
    - amp: Automatic Mixed Precision is a PyTorch feature (torch.cuda.amp)
      that automatically uses 16-bit floats where it's numerically safe, 
      while keeping 32-bit math for sensitive ops.
    
    - clip_grad: Clip_grad keeps the L2-norm of the gradient vector below a threshold 
      (clip_grad_norm_(… , max_norm=1.0) by default). If the loss ever spikes (common
      when Optuna tries a bad hyper-param set), the step is tamed instead of blowing
      the weights to NaNs.
    
    - Inspects criterion type to switch between BCE and CE.
    
    Returns: (mean_loss, acc, prec, rec, f1)
    """
    # ┏━━━━━━━━━━ AMP ━━━━━━━━━━┓
    amp = amp and (device.type == 'cuda')
    scaler = GradScaler('cuda') if amp else None

    # ┏━━━━━━━━━━ Training or Evaluation Mode ━━━━━━━━━━┓
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    # ┏━━━━━━━━━━ Storage of losses, predictions and targets ━━━━━━━━━━┓
    losses, preds_cpu, targets_cpu = [], [], []

    with torch.set_grad_enabled(is_train):
        # Main Loop
        for xb, y_up, y_dn in loader:
            # For faster transfer
            xb = xb.to(device, non_blocking=True)
            targ = (y_up if task_name == "UP" else y_dn).to(device, non_blocking=True)
            
            # ┏━━━━━━━━━━ Forward Pass (+ Autocast) + Predictions ━━━━━━━━━━┓
            with autocast('cuda', enabled = amp):
                if isinstance(criterion, nn.BCEWithLogitsLoss):
                        logits = model(xb).squeeze(1)
                        loss   = criterion(logits, targ.float())
                        pred   = (torch.sigmoid(logits) > bce_thr).long()
                else:
                    logits = model(xb)
                    loss   = criterion(logits, targ.long())
                    pred   = logits.argmax(dim=1)
            



            # ┏━━━━━━━━━━ Back-prop ━━━━━━━━━━┓
            if is_train:
                # Cheaper
                optimizer.zero_grad(set_to_none=True)
                
                if amp:
                    # ┏━━━━━━━━━━ Scale + Backward ━━━━━━━━━━┓
                    scaler.scale(loss).backward()

                    # ┏━━━━━━━━━━ Unscale if we want gradient clipping ━━━━━━━━━━┓
                    if clip_grad is not None:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)

                    # ┏━━━━━━━━━━ Step & Update ━━━━━━━━━━┓
                    scaler.step(optimizer)
                    scaler.update()

                else:
                    # ┏━━━━━━━━━━ Pure FP32 ━━━━━━━━━━┓
                    loss.backward()
                    if clip_grad is not None:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
                    optimizer.step()
            
            # ┏━━━━━━━━━━ Store Losses, Predictions and Targets ━━━━━━━━━━┓
            losses.append(loss.detach().item())
            preds_cpu.append(pred.detach().cpu())
            targets_cpu.append(targ.detach().cpu())

    # ┏━━━━━━━━━━ Format of Predictions, Targets and Mean Loss ━━━━━━━━━━┓
    preds  = torch.cat(preds_cpu).numpy()
    targs  = torch.cat(targets_cpu).numpy()
    mean_loss = float(np.mean(losses))

    # ┏━━━━━━━━━━ Compute Metrics ━━━━━━━━━━┓
    acc   = accuracy_score(targs, preds)
    prec  = precision_score(targs, preds, zero_division=0)
    rec   = recall_score(targs, preds, zero_division=0)
    f1    = f1_score(targs, preds, zero_division=0)
    fbeta = fbeta_score(targs, preds, beta = beta, zero_division = 0)

    return mean_loss, acc, prec, rec, f1, fbeta


# ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
# ┃ MODEL TRAIN                                                           ┃
# ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
def model_train(model, 
                optimizer, 
                criterion, 
                early_stopper,
                train_loader, 
                val_loader, 
                writer, 
                device,
                MAX_EPOCHS, 
                task_name,
                scheduler,
                bce_thr,
                beta
                ):
    """
    Full training loop with early stopping on validation loss.
    Logs train/val losses to TensorBoard as '{task_name}/Loss/{Train,Val}'.
    """
    # ┏━━━━━━━━━━ Storage of Losses ━━━━━━━━━━┓
    train_losses = []
    val_losses   = []

    # ┏━━━━━━━━━━ Header ━━━━━━━━━━┓
    header = f"▶️▶️▶️ Starting {task_name} training for {MAX_EPOCHS} epochs ◀️◀️◀️"
    print("\n" + "="*len(header))
    print(header)
    print("="*len(header) + "\n")

    for epoch in range(1, MAX_EPOCHS+1):
        # ┏━━━━━━━━━━ Training ━━━━━━━━━━┓
        tr_loss, tr_acc, tr_prec, tr_rec, tr_f1, tr_fbeta = epoch_loop(model, 
                                                             train_loader, 
                                                             criterion, 
                                                             device, 
                                                             optimizer, 
                                                             task_name,
                                                             bce_thr,
                                                             amp = True,
                                                             clip_grad = 1.0,
                                                             beta = beta
                                                  )

        # ┏━━━━━━━━━━ Validation ━━━━━━━━━━┓
        val_loss, val_acc, val_prec, val_rec, val_f1, val_fbeta = epoch_loop(model, 
                                                                  val_loader, 
                                                                  criterion, 
                                                                  device, 
                                                                  None, 
                                                                  task_name,
                                                                  bce_thr,
                                                                  amp = True,
                                                                  clip_grad = 0.0,
                                                                  beta = beta
                                                       )
        
        # ┏━━━━━━━━━━ Record losses ━━━━━━━━━━┓
        train_losses.append(tr_loss)
        val_losses.append(val_loss)

        # ┏━━━━━━━━━━ Pretty print ━━━━━━━━━━┓
        # print(f"{'-'*60}")
        # print(f" Epoch {epoch:2d}/{MAX_EPOCHS:2d} | {task_name}")
        # print(f"{'-'*60}")
        # print(f"{'Phase':<10}{'Loss':>8}{'Acc':>8}{'Prec':>8}{'Rec':>8}{'F1':>8}")
        # print(f"{'Train':<10}{tr_loss:>8.4f}{tr_acc:>8.4f}{tr_prec:>8.4f}{tr_rec:>8.4f}{tr_f1:>8.4f}")
        # print(f"{'Valid':<10}{val_loss:>8.4f}{val_acc:>8.4f}{val_prec:>8.4f}{val_rec:>8.4f}{val_f1:>8.4f}")
        # print()  # blank line

        # ┏━━━━━━━━━━ TensorBoard logging ━━━━━━━━━━┓
        writer.add_scalar(f"{task_name}/Loss/Train", tr_loss, epoch)
        writer.add_scalar(f"{task_name}/Loss/Val",   val_loss, epoch)

        # ┏━━━━━━━━━━ LR-scheduler step ━━━━━━━━━━┓
        step_scheduler(scheduler, epoch, val_loss)

        # ┏━━━━━━━━━━ Early-stopping ━━━━━━━━━━┓
        early_stopper(val_loss, model)
        if early_stopper.early_stop:
            print(f"🚨 Early stopping triggered at epoch {epoch} 🚨\n")
            break
    
    # ┏━━━━━━━━━━ Compute and print averages ━━━━━━━━━━┓
    avg_train = sum(train_losses) / len(train_losses)
    avg_val   = sum(val_losses)   / len(val_losses)
    print(f"   ▶︎ Average Train Loss: {avg_train:.4f}")
    print(f"   ▶︎ Average Val   Loss: {avg_val:.4f}\n")

    # ┏━━━━━━━━━━ Footer ━━━━━━━━━━┓
    footer = f"▶️▶️▶️ Finished {task_name} training for {MAX_EPOCHS} epochs ◀️◀️◀️"
    print("="*len(footer))
    print(footer)
    print("="*len(footer) + "\n")

    return val_loss


# ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
# ┃ MODEL TEST                                                            ┃
# ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
def model_test(model, 
               criterion, 
               test_loader, 
               writer, 
               device,
               step, 
               task_name,
               bce_thr,
               beta
            ):
    """
    Run one pass over 'test_loader', print metrics, and log test-loss at given 'step'.
    """ 
    # ┏━━━━━━━━━━ Header ━━━━━━━━━━┓
    header = f"▶️▶️▶️ Starting {task_name} Testing ◀️◀️◀️"
    print("\n" + "="*len(header))
    print(header)
    print("="*len(header) + "\n")
    
    # ┏━━━━━━━━━━ Testing ━━━━━━━━━━┓
    loss, acc, prec, rec, f1, fbeta = epoch_loop(model, 
                                          test_loader, 
                                          criterion, 
                                          device, 
                                          None, 
                                          task_name,
                                          bce_thr,
                                          amp = True,
                                          clip_grad = 1.0,
                                          beta = beta
                                          )

    #  ━━━━━━━━━━┓ Pretty test banner ━━━━━━━━━━┓
    print("#"*23)
    print(f"🌟 {task_name} TEST RESULTS 🌟")
    print("#"*23)
    print(f"{'Metric':<10}{'Value':>10}")
    print(f"{'Loss':<10}{loss:>10.4f}")
    print(f"{'Accuracy':<10}{acc:>10.4f}")
    print(f"{'Precision':<10}{prec:>10.4f}")
    print(f"{'Recall':<10}{rec:>10.4f}")
    print(f"{'F1':<10}{f1:>10.4f}")
    print("#"*23 + "\n")
    
    # ┏━━━━━━━━━━ Footer ━━━━━━━━━━┓
    footer = f"▶️▶️▶️ Finished {task_name} Testing ◀️◀️◀️"
    print("="*len(footer))
    print(footer)
    print("="*len(footer) + "\n")

    return loss, acc, prec, rec, f1, fbeta
