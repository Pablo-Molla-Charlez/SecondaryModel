#!/usr/bin/env python
# ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
# ┃ 1. IMPORTS                                                            ┃
# ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
import os
import argparse
import random
import warnings
import datetime
import numpy  as np
import pandas as pd
import torch
import copy
import yaml
import subprocess
import optuna
from optuna import logging as optuna_logging

from pathlib import Path
from typing import List, Optional
from optuna_utils import build_candidate_config, export_pareto_configs, feature_map
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
config_path = base / "config.yaml"
cfg  = yaml.safe_load(open(config_path, "r"))

# ┏━━━━━━━━━━ 3.b) For the training hyper-parameters, i.e. learning rate, batch size, etc. ━━━━━━━━━━┓
train_cfg = {"UP": cfg["train_up"], "DN": cfg["train_dn"]}

# ┏━━━━━━━━━━ 3.c) Fixed Constants ━━━━━━━━━━┓
seq_len = cfg["sequence_length"]
COLUMN_FEATURES = feature_map(cfg, "column_features")
CONTEXT_FEATURES = feature_map(cfg, "context_features")
train_frac       = cfg["splits"]["train"]
val_frac         = cfg["splits"]["val"]
test_frac        = cfg["splits"]["test"]

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

# ┏━━━━━━━━━━ Reproducibility ━━━━━━━━━━┓
seed_everything(1493583942)


# ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
# ┃ 5. CROSS-VALIDATION RUN                                               ┃
# ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
def cv_folds(ds, params, props, task, trial_number: int, run_id: str, trial: Optional[optuna.Trial] = None) -> dict[str, float]:
    task = task.upper()
    task_columns = COLUMN_FEATURES[task]
    task_context = CONTEXT_FEATURES[task]
    # ┏━━━━━━━━━━ 5.a) Build Tensors ━━━━━━━━━━┓
    seed_everything(1493583942)
    folds, test_loader = build_loaders(ds,
                                       cross_validation = True,                    # Must be True
                                       target           = task,                    # "DN" otherwise
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
    optuna_task_dir = base / cfg["paths"]["output_root"] / "Optuna" / f"Task_{task}"
    trial_dir = optuna_task_dir / run_id / f"Trial_{trial_number}"
    ckpt_dir  = trial_dir / "checkpoints"
    cm_dir    = trial_dir / "confusion_matrices"
    for d in (ckpt_dir, cm_dir):
        d.mkdir(parents=True, exist_ok=True)

    # ┏━━━━━━━━━━ TensorBoard Directory (holding the writer here once we hit the last fold) ━━━━━━━━━━┓
    tb_trial_dir = base / cfg["paths"]["output_root"] / "Optuna" / "Tensorboard" / f"Task_{task}" / run_id
    tb_trial_dir.mkdir(parents=True, exist_ok=True)
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
            nb_features   = len(task_columns),

            # ┏━━━━━━━━━━ 2. Transformer Architecture ━━━━━━━━━━┓
            trans_heads   = params["heads"],
            trans_layers  = params["layers"],
            trans_ff      = params["ffn_dim"] * params["cnn_embed_dim"][-1],
            trans_dropout = params["dropout"],
            trans_activ   = params["activation"],

            # ┏━━━━━━━━━━ 3. MLP Architecture ━━━━━━━━━━┓
            mlp_hidden    = params["mlp_hidden"],
            mlp_dropout   = params["mlp_dropout"],
            mlp_activ     = params["mlp_activation"],
            mlp_pooling   = params["mlp_pooling"],

            # ┏━━━━━━━━━━ 4. Other Parameters ━━━━━━━━━━┓
            context_len   = seq_len + len(task_context),  # sequence length plus appended context tokens
            padding       = padding,
            num_classes   = 1 if params["loss_function"] == "bce" else 2,
        ).to(DEVICE)

        # ┏━━━━━━━━━━ Optimizer ━━━━━━━━━━┓
        optimizer = get_optimizer(model.parameters(), params)
        
        # ┏━━━━━━━━━━ Learning Rate Scheduler ━━━━━━━━━━┓
        max_epochs = train_cfg[task]["max_epochs"]
        scheduler = make_scheduler(optimizer, params, max_epochs)
        
        # ┏━━━━━━━━━━ Early Stopping Instantiation ━━━━━━━━━━┓
        stop = EarlyStopping(patience   = train_cfg[task]["patience"], 
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
        try:
            for epoch in range(max_epochs):
                # ┏━━━━━━━━━━ Train ━━━━━━━━━━┓
                train_loss, _, _, _, _, _ = epoch_loop(model, 
                                                       train_ld, 
                                                       criterion, 
                                                       DEVICE, 
                                                       optimizer, 
                                                       task_name    = task,
                                                       bce_thr      = 0.5,
                                                       amp          = True,
                                                       clip_grad    = 1.0,
                                                       beta         = train_cfg[task]["fbeta"])
                
                # ┏━━━━━━━━━━ Validation ━━━━━━━━━━┓
                val_loss, vacc, vprec, vrec, vf1, vfbeta = epoch_loop(model, 
                                                                      val_ld,   
                                                                      criterion, 
                                                                      DEVICE, 
                                                                      optimizer = None,
                                                                      task_name = task,
                                                                      bce_thr   = 0.5, 
                                                                      amp       = True, 
                                                                      clip_grad = 0.0,
                                                                      beta      = train_cfg[task]["fbeta"])
                            
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
        except RuntimeError as err:
            if writer is not None:
                writer.close()
                writer = None
            err_lower = str(err).lower()
            if (
                "cuda" in err_lower
                or "cublas" in err_lower
                or "invalid target index" in err_lower
            ):
                torch.cuda.empty_cache()
                if trial is not None:
                    reason = (
                        "invalid target index"
                        if "invalid target index" in err_lower
                        else "CUDA failure"
                    )
                    raise optuna.TrialPruned(f"Pruned due to {reason}: {err}")
            raise
        
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
            ckpt_path = ckpt_dir / f"{trial_number}_{task}_best.pt"
            torch.save(best_state, ckpt_path)

            # ┏━━━━━━━━━━ Original split's Model Evaluation on Validation Set & Confusion Matrix ━━━━━━━━━━┓
            vpreds, vtargets = get_preds_and_targets(model, val_ld, DEVICE, task, params["loss_function"])
            plot_cm_with_metrics(vpreds, 
                                 vtargets,
                                 labels  = (f"No_TP_{task}", f"TP_{task}"),
                                 title   = f"{task} — Validation",
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
                                                                task_name = task, 
                                                                bce_thr   = 0.5, 
                                                                amp       = True, 
                                                                clip_grad = 0.0,
                                                                beta      = train_cfg[task]["fbeta"])
            test_accuracy  = t_acc
            test_precision = t_prec
            test_recall    = t_rec
            test_f1        = t_f1
            test_fbeta     = t_fbeta

            # ┏━━━━━━━━━━ Test Confusion Matrix  ━━━━━━━━━━┓
            tpreds, ttargets = get_preds_and_targets(model, test_loader, DEVICE, task, params["loss_function"])
            plot_cm_with_metrics(tpreds, 
                                 ttargets,
                                 labels  = (f"No_TP_{task}", f"TP_{task}"),
                                 title   = f"{task} — Test",
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
def objective(trial: optuna.Trial, dataset: TensorDataset, props: List[float], task: str, run_id: str) -> float:
    task = task.upper()
    
    # ┏━━━━━━━━━━ Multiple Convolutions for CNN Architecture ━━━━━━━━━━┓
    model_key = f"model_{task.lower()}"
    configured_blocks = cfg[model_key].get("cnn_blocks")
    if isinstance(configured_blocks, int) and configured_blocks >= 1:
        n_convs = configured_blocks
    else:
        max_convs = len(cfg[model_key]["cnn_embed_dim"])
        n_convs = trial.suggest_int("n_convs", 1, max_convs)
    
    cnn_embed_dim = [trial.suggest_categorical(f"cnn_embed_dim_{i}", [64, 128, 256]) for i in range(n_convs)]
    cnn_kernel    = [trial.suggest_categorical(f"cnn_kernel_{i}",    [4, 8]) for i in range(n_convs)]
    cnn_stride    = [trial.suggest_categorical(f"cnn_stride_{i}",    [1, 2, 4]) for i in range(n_convs)]

    trial.set_user_attr("n_convs", n_convs)
    trial.set_user_attr("cnn_embed_dim_list", cnn_embed_dim)
    trial.set_user_attr("cnn_kernel_list", cnn_kernel)
    trial.set_user_attr("cnn_stride_list", cnn_stride)

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
                                                                "silu"]),
        
        # ┏━━━━━━━━━━ 3. MLP Architecture ━━━━━━━━━━┓
        "mlp_hidden":     trial.suggest_categorical("mlp_hidden", [128, 256, 512]),               # MLP hidden size
        "mlp_dropout":    trial.suggest_float("mlp_dropout", 0.0, 0.5),                            # MLP dropout probability
        "mlp_pooling":    trial.suggest_categorical("mlp_pooling", ["attention", "meanmax"]),      # MLP pooling mechanism
        "mlp_activation": trial.suggest_categorical("mlp_activation", ["gelu",                     # MLP activation function
                                                                        "relu", 
                                                                        #"silu", 
                                                                        "mish"]),              
        
        # ┏━━━━━━━━━━ Optimizer ━━━━━━━━━━┓
        "optimizer":  trial.suggest_categorical("optimizer", ["adagrad",                            # Optimizer Name
                                                              #"rmsprop", 
                                                              "adam", 
                                                              "adamw", 
                                                              #"adabelief", 
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
        #"weight_decouple":  True,  # trial.suggest_categorical("weight_decouple", [True, False]),   # Default Value
        #"rectify":          False, # trial.suggest_categorical("rectify", [True, False]),           # Default Value

        # ┏━━━━━━━━━━ Hyperparameters ━━━━━━━━━━┓
        "batch_size":      2 ** trial.suggest_int("batch_pow", 5, 8),                               # Batch size (2**batch_pow)
        "loss_function":   trial.suggest_categorical("loss_function", ["bce",
                                                                       #"focal", 
                                                                       "cross_entropy"]),           # Loss type
        # ┏━━━━━━━━━━ Focal Loss ━━━━━━━━━━┓
        "focal_gamma":     3.3771842197375523,
        "focal_alpha":    (2.954 / (2.954 + 1.1514)),                                              # Default Value

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
    metrics = cv_folds(dataset, params, props, task, trial.number, run_id, trial = trial)

    # ┏━━━━━━━━━━ Filter Results ━━━━━━━━━━┓
    mean_val_loss = metrics["mean_val_loss"]
    best_prec = metrics["best_prec"]
    best_fbeta = metrics["best_fbeta"]
    if mean_val_loss > 0.7 or best_fbeta < 0.35:
       raise optuna.TrialPruned()
    
    # ┏━━━━━━━━━━ Record metrics on the Trial ━━━━━━━━━━┓
    trial.set_user_attr("task",                task)
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
# ┃ 7. OPTUNA PER TASK                                                    ┃
# ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
def run_optuna_for_task(dataset: TensorDataset,
                        props: List[float],
                        task: str) -> optuna.study.Study:
    """Launch an Optuna study for the provided task ("UP" | "DN")."""
    
    task = task.upper()
    if task not in train_cfg:
        raise ValueError(f"Unknown task '{task}'. Expected one of: {list(train_cfg.keys())}")

    # ┏━━━━━━━━━━ 1) Prepare Optuna Storage ━━━━━━━━━━┓
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"run_{timestamp}"
    run_root = base / cfg["paths"]["output_root"] / "Optuna" / f"Task_{task}" / run_id
    run_root.mkdir(parents=True, exist_ok=True)
    db_path = run_root / f"optuna_study_{task}_{timestamp}.db"
    storage_url = f"sqlite:///{db_path.resolve()}"
    study_name = f"CTTS_Study_{task}_{timestamp}"

    # ┏━━━━━━━━━━ 2) Instantiation of Optuna & Optimization ━━━━━━━━━━┓
    print(f"\n▶ Running Optuna study for task {task} with {ARGS.trials} trials (run id: {run_id})")
    study = optuna.create_study(
        study_name     = study_name,
        storage        = storage_url,
        directions     = ["minimize", "maximize"],
        pruner         = optuna.pruners.MedianPruner(n_startup_trials = 20),
        load_if_exists = False,
    )

    # ┏━━━━━━━━━━ 3) Optuna Optimization ━━━━━━━━━━┓
    study.optimize(lambda trial: objective(trial,
                            dataset,
                            props,
                            task,
                            run_id),
                   n_trials = ARGS.trials,
                   show_progress_bar = True)

    # ┏━━━━━━━━━━ 4) Export Pareto-optimal configs ━━━━━━━━━━┓
    pareto_trials = [t for t in study.best_trials if t.state.is_finished()]
    saved = export_pareto_configs(config_path, cfg, pareto_trials, run_root, task)
    print(f"Saved {saved} Pareto-optimal configs to {run_root / 'Optuna_Pareto_Candidates'}.")

    return study


# ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
# ┃ 8. MAIN FUNCTION                                                      ┃
# ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
if __name__ == "__main__":
    # ┏━━━━━━━━━━ 7.a) Omit Warnings & CSV Paths ━━━━━━━━━━┓
    warnings.filterwarnings("ignore", category=UserWarning)
    csv_path = dataset_path(cfg["dataset"]["source"],
                            cfg["dataset"]["type"].capitalize(),
                            cfg["dataset"]["symbol"],
                            "up")

    # ┏━━━━━━━━━━ 7.b) Extracting Column Features ━━━━━━━━━━┓
    all_columns = []
    for feat_list in COLUMN_FEATURES.values():
        for feat in feat_list:
            if feat not in all_columns:
                all_columns.append(feat)

    # ┏━━━━━━━━━━ 7.c) Reading CSV ━━━━━━━━━━┓
    df_asset = merge_meta_targets(asset_type      = cfg["dataset"]["type"],
                                  asset           = cfg["dataset"]["symbol"],
                                  data_dir        = str(Path(csv_path).parent),
                                  output_dir      = str(Path(csv_path).parent),
                                  column_features = all_columns)

    # ┏━━━━━━━━━━ 7.d) Preparing Data ━━━━━━━━━━┓        
    default_task = optuna_task.upper()
    tasks = [default_task]
    if cfg["training_mode"].get("optuna_both", True):
        opposite_task = "DN" if default_task == "UP" else "UP"
        if opposite_task not in tasks:
            tasks.append(opposite_task)

    datasets = {task: prepare_dataset(
                      df_asset,
                      seq_len = cfg["sequence_length"],
                      column_features = COLUMN_FEATURES[task],
                      context_features = CONTEXT_FEATURES[task]) for task in tasks}

    # ┏━━━━━━━━━━ 7.e) Running BOTH (UP/DOWN) Optuna Studies ━━━━━━━━━━┓
    studies = {}
    for task in tasks:
        studies[task] = run_optuna_for_task(datasets[task], cross_val_props, task)

    print("\n Completed Optuna optimization for tasks:")
    for task, study in studies.items():
        if study.best_trials:
            best = study.best_trials[0]
            print(f"  • {task}: Objectives={best.values} (study: {study.study_name})")
        else:
            print(f"  • {task}: No completed trials")
