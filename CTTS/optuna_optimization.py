#!/usr/bin/env python
# ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
# ┃ 1. IMPORTS                                                            ┃
# ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
import os
import json
import argparse
import random
import warnings
import numpy  as np
import pandas as pd
import torch
import copy
import yaml
import subprocess
import optuna
from pathlib import Path
from typing import List
from torch.utils.data import TensorDataset, Subset, DataLoader
from torch.utils.tensorboard import SummaryWriter

# ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
# ┃ 2. SPECIFIC CLASSES & FUNCTIONS IMPORTS                               ┃
# ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
from model import CTTSModel, EarlyStopping
from train_utils import epoch_loop, seed_everything
from optim_utils import make_scheduler, step_scheduler, get_optimizer
from data_preprocessing import prepare_dataset, build_loaders, merge_meta_targets
from paths import dataset_path
from test_utils import get_preds_and_targets, plot_cm_with_metrics

# ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
# ┃ 3. CONFIG & CLIENT ARGUMENTS                                          ┃
# ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
# ┏━━━━━━━━━━ 3.a) Root ━━━━━━━━━━┓
base = Path(__file__).parent
cfg  = yaml.safe_load(open(base / "config.yaml", "r"))

# ┏━━━━━━━━━━ 3.b) For the training hyper-parameters, i.e. learning rate, batch size, etc. ━━━━━━━━━━┓
train_cfg = {"UP": cfg["train_up"], "DN": cfg["train_dn"]}

# ┏━━━━━━━━━━ 3.c) Fixed Constants ━━━━━━━━━━┓
seq_len         = cfg["sequence_length"]
column_features = cfg["column_features"]
train_frac      = cfg["splits"]["train"]
val_frac        = cfg["splits"]["val"]
test_frac       = cfg["splits"]["test"]

# ┏━━━━━━━━━━ 3.d) Cross-Validation ━━━━━━━━━━┓
cross_validation = cfg["training_mode"]["cross_validation"] # Set it to true
cross_val_props  = cfg["training_mode"]["cross_val_props"]
optuna_task      = cfg["training_mode"]["optuna_task"]
num_classes      = cfg["training_mode"]["num_classes"]
padding          = cfg["training_mode"]["padding"]

# ┏━━━━━━━━━━ 3.e) Client ━━━━━━━━━━┓
cli = argparse.ArgumentParser()
cli.add_argument("--trials",  type=int, default=500, help="Optuna trials")    # 500 different parameter combinations
cli.add_argument("--seed",    type=int, default=42,  help="Global random seed")
cli.add_argument("--device",  type=str, default= 'cuda', help="'cuda', 'cpu' or 'mps'.  If omitted → auto-detect")
ARGS = cli.parse_args()


# ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
# ┃ 4. REPRODUCIBILITY & DEVICE                                           ┃
# ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
# ┏━━━━━━━━━━ Select GPU if available, otherwise CPU ━━━━━━━━━━┓
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on {DEVICE}")
# For Personal Mac
# device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

# ┏━━━━━━━━━━ Reproducibility ━━━━━━━━━━┓
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
seed_everything(1493583942)


# ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
# ┃ 5. CROSS-VALIDATION RUN                                               ┃
# ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
def cv_folds(ds, params, props, optuna_task, trial_number: int) -> dict[str, float]:
    # ┏━━━━━━━━━━ 5.a) Build Tensors ━━━━━━━━━━┓
    seed_everything(1493583942)
    folds, test_loader = build_loaders(ds,
                                       cross_validation = True,                    # Must be True
                                       target           = optuna_task,             # "DN" otherwise
                                       props            = props,                   # Fixed
                                       train_frac       = train_frac,              # Fixed
                                       val_frac         = val_frac,                # Fixed
                                       test_frac        = test_frac,               # Fixed
                                       batch_size       = params["batch_size"],
                                       loss_type        = params["loss_function"],
                                       focal_gamma      = params["focal_gamma"],
                                       focal_alpha      = params["focal_alpha"],
                                       device           = DEVICE)
    
    # ┏━━━━━━━━━━ 5.b) Storage of Validation Losses and Metrics (each fold) ━━━━━━━━━━┓
    n_folds        = len(folds)
    test_precision = None
    fold_losses    = []
    fold_accs      = []
    fold_precs     = []
    fold_f1s       = []
    fold_recs      = []
    fold_fbetas    = []
    
    # ┏━━━━━━━━━━ Prepare directory for this Trial's Confusion Matrices (Val & Test) & TensorBoard ━━━━━━━━━━┓
    trial_dir = base / cfg["paths"]["output_root"] / "Optuna" / f"Trial_{trial_number}"
    ckpt_dir  = trial_dir / "checkpoints"
    cm_dir    = trial_dir / "confusion_matrices"
    for d in (ckpt_dir, cm_dir):
        d.mkdir(parents=True, exist_ok=True)

    # ┏━━━━━━━━━━ TensorBoard Directory (holding the writer here once we hit the last fold) ━━━━━━━━━━┓
    tb_trial_dir = base / cfg["paths"]["output_root"] / "Optuna" / "Tensorboard"
    writer = None

    # ┏━━━━━━━━━━ 5.c) Model Instantiation, Train & Validation per fold ━━━━━━━━━━┓
    for fold_idx, (train_ld, val_ld, criterion) in enumerate(folds):
        # ┏━━━━━━━━━━ If this is the last fold, create a TensorBoard writer ━━━━━━━━━━┓
        if fold_idx == n_folds - 1:
            writer = SummaryWriter(tb_trial_dir / f"Trial_{trial_number}")

        # ┏━━━━━━━━━━ Model Instantiation per fold ━━━━━━━━━━┓
        seed_everything(1493583942)
        model = CTTSModel(
                        # ┏━━━━━━━━━━ 1. CNN Architecture ━━━━━━━━━━┓
                        cnn_embed_dim = params["cnn_embed_dim"],
                        cnn_kernel    = params["cnn_kernel"],
                        cnn_stride    = params["cnn_stride"],
                        p_pos_drop    = params["p_pos_drop"],
                        nb_features  = len(column_features),
                        
                        # ┏━━━━━━━━━━ 2. Transformer Architecture ━━━━━━━━━━┓
                        trans_heads   = params["heads"],
                        trans_layers  = params["layers"],
                        trans_ff      = params['ffn_dim'] * params['cnn_embed_dim'][-1],
                        trans_dropout = params["dropout"],
                        trans_activ   = params["activation"],
                        
                        # ┏━━━━━━━━━━ 3. MLP Architecture ━━━━━━━━━━┓
                        mlp_hidden    = params["mlp_hidden"],
                        mlp_dropout   = params["mlp_dropout"],
                        mlp_activ     = params["mlp_activation"],
                        mlp_pooling   = params["mlp_pooling"],
                        
                        # ┏━━━━━━━━━━ 4. Other Parameters ━━━━━━━━━━┓
                        context_len   = seq_len,                                              # Fixed
                        padding       = padding,                                              # Fixed
                        num_classes   = 1 if params["loss_function"] == "bce" else 2          # bce | cross_entropy | focal

                        ).to(DEVICE)

        # ┏━━━━━━━━━━ Optimizer ━━━━━━━━━━┓
        optimizer = get_optimizer(model.parameters(), params)
        
        # ┏━━━━━━━━━━ Learning Rate Scheduler ━━━━━━━━━━┓
        max_epochs = train_cfg[optuna_task]["max_epochs"]
        scheduler = make_scheduler(optimizer, params, max_epochs)
        
        # ┏━━━━━━━━━━ Early Stopping Instantiation ━━━━━━━━━━┓
        stop = EarlyStopping(patience   = train_cfg[optuna_task]["patience"], 
                             verbose    = False,
                             delta      = 1e-4,
                             path       = None,    # No model saves
                             just_count = True)

        # ┏━━━━━━━━━━ Best Metrics (temporary) ━━━━━━━━━━┓
        best_loss  = float('inf')
        best_acc   = 0.0
        best_prec  = 0.0
        best_rec   = 0.0
        best_f1    = 0.0
        best_fbeta = 0.0
        best_state = None            # ← store best weights in memory

        # ┏━━━━━━━━━━ Train & Validation ━━━━━━━━━━┓
        for epoch in range(max_epochs):
            # ┏━━━━━━━━━━ Train ━━━━━━━━━━┓
            train_loss, _, _, _, _, _ = epoch_loop(model, 
                                                   train_ld, 
                                                   criterion, 
                                                   DEVICE, 
                                                   optimizer, 
                                                   task_name    = optuna_task,
                                                   bce_thr      = 0.5,
                                                   amp          = True,
                                                   clip_grad    = 1.0,
                                                   beta         = train_cfg[optuna_task]["fbeta"])
            
            # ┏━━━━━━━━━━ Validation ━━━━━━━━━━┓
            val_loss, vacc, vprec, vrec, vf1, vfbeta = epoch_loop(model, 
                                                                  val_ld,   
                                                                  criterion, 
                                                                  DEVICE, 
                                                                  optimizer = None,
                                                                  task_name = optuna_task,
                                                                  bce_thr   = 0.5, 
                                                                  amp       = True, 
                                                                  clip_grad = 0.0,
                                                                  beta      = train_cfg[optuna_task]["fbeta"])
                        
            # ┏━━━━━━━━━━ Log to TensorBoard if last fold ━━━━━━━━━━┓
            if writer is not None:
                writer.add_scalar("Loss/train",      train_loss, epoch)
                writer.add_scalar("Loss/validation", val_loss  , epoch)

            # ┏━━━━━━━━━━ Early Stopping ━━━━━━━━━━┓
            stop(val_loss, model)
            step_scheduler(scheduler, epoch, val_loss)
            if stop.early_stop:
                break
            
            # ┏━━━━━━━━━━ If this epoch is the best so far, record its metrics ━━━━━━━━━━┓
            if val_loss < best_loss:
                best_loss = val_loss
                best_acc  = vacc
                best_prec = vprec
                best_f1   = vf1
                best_rec  = vrec
                best_fbeta  = vfbeta
                best_state = copy.deepcopy(model.state_dict())
        
        # ┏━━━━━━━━━━ Close writer after last fold training ━━━━━━━━━━┓
        if writer is not None:
            writer.close()
            writer = None

        # ┏━━━━━━━━━━ Store Best Validation Loss and Metrics (per fold) ━━━━━━━━━━┓
        fold_losses.append(best_loss)
        fold_accs.append(best_acc)
        fold_precs.append(best_prec)
        fold_f1s.append(best_f1)
        fold_recs.append(best_rec)
        fold_fbetas.append(best_fbeta)

        # ┏━━━━━━━━━━ Store Best Precision in Test ━━━━━━━━━━┓
        if fold_idx == n_folds - 1:
            
            # ┏━━━━━━━━━━ Reload the best state into model ━━━━━━━━━━┓
            model.load_state_dict(best_state)

            # ┏━━━━━━━━━━ Save that best state‐dict to disk ━━━━━━━━━━┓
            ckpt_path = ckpt_dir / f"{trial_number}_{optuna_task}_best.pt"
            torch.save(best_state, ckpt_path)

            # ┏━━━━━━━━━━ Original split's Model Evaluation on Validation Set & Confusion Matrix ━━━━━━━━━━┓
            vpreds, vtargets = get_preds_and_targets(model, val_ld, DEVICE, optuna_task, params["loss_function"])
            plot_cm_with_metrics(vpreds, 
                                 vtargets,
                                 labels  = (f"No_TP_{optuna_task}", f"TP_{optuna_task}"),
                                 title   = f"{optuna_task} — Validation",
                                 out_dir = cm_dir,
                                 cmap    = "Oranges"
                            )
            
            # ┏━━━━━━━━━━ Original split's Model Evaluation on Test Set & Confusion Matrix ━━━━━━━━━━┓
            with torch.no_grad():
                t_loss, t_acc, t_prec, t_rec, t_f1, t_fbeta = epoch_loop(model, 
                                                                test_loader, 
                                                                criterion, 
                                                                DEVICE,
                                                                optimizer = None, 
                                                                task_name = optuna_task, 
                                                                bce_thr   = 0.5, 
                                                                amp       = True, 
                                                                clip_grad = 0.0,
                                                                beta      = train_cfg[optuna_task]["fbeta"])
            test_accuracy  = t_acc
            test_precision = t_prec
            test_recall    = t_rec
            test_f1        = t_f1
            test_fbeta     = t_fbeta

            # ┏━━━━━━━━━━ Test Confusion Matrix  ━━━━━━━━━━┓
            tpreds, ttargets = get_preds_and_targets(model, test_loader, DEVICE, optuna_task, params["loss_function"])
            plot_cm_with_metrics(tpreds, 
                                 ttargets,
                                 labels  = (f"No_TP_{optuna_task}", f"TP_{optuna_task}"),
                                 title   = f"{optuna_task} — Test",
                                 out_dir = cm_dir,
                                 cmap    = "Blues"
                            )

        
        # ┏━━━━━━━━━━ Empty Caché ━━━━━━━━━━┓
        torch.cuda.empty_cache()

    return {"mean_val_loss":       float(np.mean(fold_losses)),
            "mean_val_accuracy":   float(np.mean(fold_accs)),
            "mean_val_precision":  float(np.mean(fold_precs)),
            "mean_val_rec":        float(np.mean(fold_recs)),
            "mean_val_f1":         float(np.mean(fold_f1s)),
            "mean_val_fbeta":      float(np.mean(fold_fbetas)),
            
            "best_val":            best_loss,
            "best_acc":            best_acc,
            "best_prec":           best_prec,
            "best_rec":            best_rec,
            "best_f1":             best_f1,
            "best_fbeta":          best_fbeta,

            "test_acc":            test_accuracy,
            "test_prec":           test_precision,
            "test_rec":            test_recall,
            "test_f1":             test_f1,
            "test_fbeta":          test_fbeta
            }

# ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
# ┃ 6. OPTUNA OBJECTIVE                                                   ┃
# ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
def objective(trial: optuna.Trial, dataset: TensorDataset, props: List[float], optuna_task) -> float:
    
    # ┏━━━━━━━━━━ Multiple Convolutions for CNN Architecture ━━━━━━━━━━┓
    n_convs = trial.suggest_int("n_convs", 1, 2)                                                                                                # Optimized
    cnn_embed_dim = [trial.suggest_categorical(f"cnn_embed_dim_{i}", [64, 128, 256]) for i in range(n_convs)]
    cnn_kernel    = [trial.suggest_categorical(f"cnn_kernel_{i}",    [4, 8]) for i in range(n_convs)]
    cnn_stride    = [trial.suggest_categorical(f"cnn_stride_{i}",    [1, 2, 4]) for i in range(n_convs)]

    # ┏━━━━━━━━━━ Parameters to Optimize by Optuna ━━━━━━━━━━┓
    params = {
        # ┏━━━━━━━━━━ 1. CNN Architecture ━━━━━━━━━━┓
        "cnn_embed_dim":  cnn_embed_dim,                                                          # Embedding dimension
        "cnn_kernel":     cnn_kernel,                                                             # CNN kernel size
        "cnn_stride":     cnn_stride,                                                             # CNN stride
        "p_pos_drop":     trial.suggest_float("p_pos_drop", 0.0, 0.5),                            # Positional-encoding dropout probability


        # ┏━━━━━━━━━━ 2. Transformer Architecture ━━━━━━━━━━┓
        "heads":      trial.suggest_categorical("heads", [2, 4]),                                  # Attention heads
        "layers":     trial.suggest_int("layers", 2, 4),                                           # Number of transformer layers        
        "ffn_dim":    trial.suggest_categorical("ffn_dim", [4, 8]),                                # Feed-forward network multiplier (× embed)
        "dropout":    trial.suggest_float("dropout", 0.0, 0.5),                                    # Transformer dropout probability
        "activation": trial.suggest_categorical("activation", ["gelu", 
                                                                "relu", 
                                                                "silu", 
                                                                "mish"]),
        
        # ┏━━━━━━━━━━ 3. MLP Architecture ━━━━━━━━━━┓
        "mlp_hidden":     trial.suggest_categorical("mlp_hidden", [128, 256, 512]),               # MLP hidden size
        "mlp_dropout":    trial.suggest_float("mlp_dropout", 0.0, 0.5),                            # MLP dropout probability
        "mlp_pooling":    trial.suggest_categorical("mlp_pooling", ["attention", "meanmax"]),      # MLP pooling mechanism
        "mlp_activation": trial.suggest_categorical("mlp_activation", ["gelu",                     # MLP activation function
                                                                        "relu", 
                                                                        "silu", 
                                                                        "mish"]),              
        
        # ┏━━━━━━━━━━ Optimizer ━━━━━━━━━━┓
        "optimizer":  trial.suggest_categorical("optimizer", ["adagrad",                            # Optimizer Name
                                                              #"rmsprop", 
                                                              #"adam", 
                                                              "adamw", 
                                                              "adabelief", 
                                                              "ranger"]),
        "lr":            trial.suggest_float("lr", 1e-6, 1e-2, log=True),                           # Learning rate
        "weight_decay":  trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True),                 # Weight decay

        # ┏━━━━━━━━━━ Optimizers tuning ━━━━━━━━━━┓
        "eps":              1e-16, # trial.suggest_float("eps", 1e-8, 1e-4, log=True),              # Default Value
        "lr_decay":         trial.suggest_float("lr_decay", 1e-6, 1e-1, log=True),
        # "rms_alpha":       trial.suggest_float("rms_alpha", 0.8, 0.999),
        # "rms_momentum":    trial.suggest_float("rms_momentum", 0.0, 0.9),
        "lookahead_k":      6,     # trial.suggest_int("lookahead_k", 1, 10),                       # Default Value
        "lookahead_alpha":  0.5,   # trial.suggest_float("lookahead_alpha", 0.1, 0.9),              # Default Value                                           
        "weight_decouple":  True,  # trial.suggest_categorical("weight_decouple", [True, False]),   # Default Value
        "rectify":          False, # trial.suggest_categorical("rectify", [True, False]),           # Default Value

        # ┏━━━━━━━━━━ Hyperparameters ━━━━━━━━━━┓
        "batch_size":      2 ** trial.suggest_int("batch_pow", 5, 8),                               # Batch size (2**batch_pow)
        "loss_function":   trial.suggest_categorical("loss_function", ["bce",
                                                                       "focal", 
                                                                       "cross_entropy"]),           # Loss type
        # ┏━━━━━━━━━━ Focal Loss ━━━━━━━━━━┓
        "focal_gamma":     trial.suggest_float("focal_gamma", 0.5, 5.0),
        "focal_alpha":     (2.954 / (2.954 + 1.1514)),                                              # Default Value

        # ┏━━━━━━━━━━ Scheduler ━━━━━━━━━━┓
        "sch_name":        trial.suggest_categorical("sch_name", [#"none",                          # Scheduler Name
                                                                  #"power", 
                                                                  #"linear", 
                                                                  #"cosine", 
                                                                  # "plateau", 
                                                                  "linear_plateau", 
                                                                  "cosine_plateau"]),

        "warmup_epochs":    5,                                                                      # Default Value [Warm-up epochs (10% of max_epochs)]
        "plateau_patience": trial.suggest_int("plateau_patience", 5, 20),                           # Plateau patience
        "plateau_factor":   trial.suggest_float("plateau_factor", 0.2, 0.5),                        # Learning-rate reduction factor
        #"power_s":         trial.suggest_float("power_s", 0.2, 0.5),                               # Power Scheduling epoch divider
        #"power_c":         trial.suggest_float("power_c", 0.2, 0.5),                               # Power Scheduling exponent
    }

    # ┏━━━━━━━━━━ Betas (Optimizer) & Packing ━━━━━━━━━━┓
    beta1     = trial.suggest_float("beta1", 0.9, 0.95, step=0.05)
    beta2     = trial.suggest_float("beta2", 0.98, 0.999, step=0.019)
    params["betas"] = (beta1, beta2)

    # ┏━━━━━━━━━━ CV_Folds Call ━━━━━━━━━━┓
    seed_everything(1493583942)
    metrics = cv_folds(dataset, params, props, optuna_task, trial.number)

    # ┏━━━━━━━━━━ Filter Results ━━━━━━━━━━┓
    mean_val_loss = metrics["mean_val_loss"]
    test_prec = metrics["test_prec"]
    test_rec = metrics["test_rec"]
    if mean_val_loss > 0.7:
        raise optuna.TrialPruned()
    
    # ┏━━━━━━━━━━ Record metrics on the Trial ━━━━━━━━━━┓
    trial.set_user_attr("mean_val_loss",       metrics["mean_val_loss"])
    trial.set_user_attr("mean_val_precision",  metrics["mean_val_precision"])
    trial.set_user_attr("mean_val_accuracy",   metrics["mean_val_accuracy"])
    trial.set_user_attr("mean_val_rec",        metrics["mean_val_rec"])
    trial.set_user_attr("mean_val_f1",         metrics["mean_val_f1"])
    trial.set_user_attr("mean_val_fbeta",      metrics["mean_val_fbeta"])

    trial.set_user_attr("best_val",       metrics["best_val"])
    trial.set_user_attr("best_acc",       metrics["best_acc"])
    trial.set_user_attr("best_prec",      metrics["best_prec"])
    trial.set_user_attr("best_rec",       metrics["best_rec"])
    trial.set_user_attr("best_f1",        metrics["best_f1"])
    trial.set_user_attr("best_fbeta",     metrics["best_fbeta"])

    trial.set_user_attr("test_acc",       metrics["test_acc"])
    trial.set_user_attr("test_prec",      metrics["test_prec"])
    trial.set_user_attr("test_rec",       metrics["test_rec"])
    trial.set_user_attr("test_f1",        metrics["test_f1"])
    trial.set_user_attr("test_fbeta",     metrics["test_fbeta"])
              
    return metrics["mean_val_loss"], metrics["best_fbeta"]


# ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
# ┃ 7. MAIN FUNCTION                                                      ┃
# ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
if __name__ == "__main__":
    # ┏━━━━━━━━━━ 7.a) Omit Warnings & CSV Paths ━━━━━━━━━━┓
    warnings.filterwarnings("ignore", category=UserWarning)
    csv_path = dataset_path(cfg["dataset"]["source"],
                            cfg["dataset"]["type"].capitalize(),
                            cfg["dataset"]["symbol"],
                            "up")

    # ┏━━━━━━━━━━ 7.b) Reading CSV ━━━━━━━━━━┓
    df_asset = merge_meta_targets(asset_type      = cfg["dataset"]["type"],
                                  asset           = cfg["dataset"]["symbol"],
                                  data_dir        = str(Path(csv_path).parent),
                                  output_dir      = str(Path(csv_path).parent),
                                  column_features = cfg['column_features']
                                )

    # ┏━━━━━━━━━━ 7.c) Preparing Data ━━━━━━━━━━┓        
    dataset = prepare_dataset(df_asset, 
                              seq_len          = cfg["sequence_length"],
                              column_features  = cfg['column_features'],
                              context_features = cfg['context_features'])
    
    # ┏━━━━━━━━━━ 7.d) Prepare Optuna Storage ━━━━━━━━━━┓
    base = Path(__file__).parent
    db_path = base / "optuna_study.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)                                                # Make sure the directory exists
    storage_url = f"sqlite:///{db_path.resolve()}"                                                   # Must include the file name

    # ┏━━━━━━━━━━ 7.e) Instantiation of Optuna & Optimization ━━━━━━━━━━┓
    study = optuna.create_study(study_name     = "CTTS_Study",
                                storage        = storage_url,
                                directions     = ["minimize", "maximize"],                           # Minimize or maximize the objective
                                pruner         = optuna.pruners.MedianPruner(n_startup_trials=10),   # Stops unpromising trials early (
                                load_if_exists = False)                                              # to save time) by comparing with median 
                                                                                                     # of 10 completed trials.
                   
                                                                                                
    # ┏━━━━━━━━━━ 7.f) Optuna Optimization ━━━━━━━━━━┓
    study.optimize(lambda trial: objective(trial,
                            dataset,
                            cross_val_props,
                            optuna_task),
                   n_trials = ARGS.trials,
                   show_progress_bar = True)
    
    # ┏━━━━━━━━━━ 7.g) JSON Storage with Pareto-optimal trials ━━━━━━━━━━┓
    pareto = []
    for t in study.best_trials:
        if t.state.is_finished():
            pareto.append({
                "number":              t.number,
                "values":              t.values,

                "mean_val_loss":       t.user_attrs.get("mean_val_loss"),
                "mean_val_precision":  t.user_attrs.get("mean_val_precision"),
                "mean_val_accuracy":   t.user_attrs.get("mean_val_accuracy"),
                "mean_val_f1":         t.user_attrs.get("mean_val_f1"),
                "mean_val_fbeta":      t.user_attrs.get("mean_val_fbeta"),

                "best_val":            t.user_attrs.get("best_val"),
                "best_acc":            t.user_attrs.get("best_acc"),
                "best_prec":           t.user_attrs.get("best_prec"),
                "best_rec":            t.user_attrs.get("best_rec"),
                "best_f1":             t.user_attrs.get("best_f1"),
                "best_fbeta":          t.user_attrs.get("best_fbeta"),

                "test_acc":            t.user_attrs.get("test_acc"),
                "test_prec":           t.user_attrs.get("test_prec"),
                "test_rec":            t.user_attrs.get("test_rec"),
                "test_f1":             t.user_attrs.get("test_f1"),
                "test_fbeta":          t.user_attrs.get("test_fbeta"),

                "params":              t.params,
            })

    out_path = base / "optuna_pareto.json"
    with open(out_path, "w") as f:
        json.dump(pareto, f, indent=2)

    print(f"\nSaved {len(pareto)} Pareto-optimal trials to {out_path}.")
