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

# ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
# ┃ 2. SPECIFIC CLASSES & FUNCTIONS IMPORTS                               ┃
# ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
from model import CTTSModel, EarlyStopping
from train_utils import epoch_loop, init_seeds
from optim_utils import make_scheduler, step_scheduler, get_optimizer
from data_preprocessing import prepare_dataset, build_loaders
from paths import dataset_path

# ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
# ┃ 3. CONFIG & CLIENT ARGUMENTS                                          ┃
# ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
# ┏━━━━━━━━━━ 3.a) Root ━━━━━━━━━━┓
base = Path(__file__).parent
cfg  = yaml.safe_load(open(base / "config.yaml", "r"))

# ┏━━━━━━━━━━ 3.b) For the training hyper-parameters, i.e. learning rate, batch size, etc. ━━━━━━━━━━┓
train_cfg = {"UP": cfg["train_up"], "DN": cfg["train_dn"]}

# ┏━━━━━━━━━━ 3.c) Fixed Constants ━━━━━━━━━━┓
seq_len    = cfg["sequence_length"]
train_frac = cfg["splits"]["train"]
val_frac   = cfg["splits"]["val"]
test_frac  = cfg["splits"]["test"]

# ┏━━━━━━━━━━ 3.d) Cross-Validation ━━━━━━━━━━┓
cross_validation = cfg["training_mode"]["cross_validation"] # Set it to true
cross_val_props = cfg["training_mode"]["cross_val_props"]
optuna_task = cfg["training_mode"]["optuna_task"]
num_classes = cfg["training_mode"]["num_classes"]
padding = cfg["training_mode"]["padding"]

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
init_seeds(42, force_cuda_deterministic = True)


# ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
# ┃ 5. CROSS-VALIDATION RUN                                               ┃
# ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
def cv_folds(ds, params, props, optuna_task) -> float:
    # ┏━━━━━━━━━━ 5.a) Build Tensors ━━━━━━━━━━┓
    folds, test_loader = build_loaders(ds,
                                       cross_validation = cross_validation,        # Must be True
                                       target           = optuna_task,             # "DN" otherwise
                                       props            = props,                   # Fixed
                                       train_frac       = train_frac,              # Fixed
                                       val_frac         = val_frac,                # Fixed
                                       test_frac        = test_frac,               # Fixed
                                       batch_size       = params["batch_size"],
                                       loss_type        = params["loss_function"],
                                       device           = DEVICE)
    
    # ┏━━━━━━━━━━ 5.b) Storage of Validation Losses and Metrics (each fold) ━━━━━━━━━━┓
    n_folds        = len(folds)
    test_precision = None
    fold_losses    = []
    fold_accs      = []
    fold_precs     = []
    fold_f1s       = []
    
    # ┏━━━━━━━━━━ 5.c) Model Instantiation, Train & Validation per fold ━━━━━━━━━━┓
    for fold_idx, (train_ld, val_ld, criterion) in enumerate(folds):
        # ┏━━━━━━━━━━ Model Instantiation per fold ━━━━━━━━━━┓
        model = CTTSModel(
                        # ┏━━━━━━━━━━ 1. CNN Architecture ━━━━━━━━━━┓
                        cnn_embed_dim = params["cnn_embed_dim"],
                        cnn_kernel    = params["cnn_kernel"],
                        cnn_stride    = params["cnn_stride"],
                        p_pos_drop    = params["p_pos_drop"],
                        
                        # ┏━━━━━━━━━━ 2. Transformer Architecture ━━━━━━━━━━┓
                        trans_heads   = params["heads"],
                        trans_layers  = params["layers"],
                        trans_ff      = params['ffn_dim'] * params['cnn_embed_dim'],
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
                        num_classes   = 1 if params["loss_function"] == "bce" else 2          # Fixed (~ cross_entropy)

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
        best_f1    = 0.0
        best_state = None            # ← store best weights in memory

        # ┏━━━━━━━━━━ Train & Validation ━━━━━━━━━━┓
        for epoch in range(max_epochs):
            # ┏━━━━━━━━━━ Train ━━━━━━━━━━┓
            epoch_loop(model, 
                       train_ld, 
                       criterion, 
                       DEVICE, 
                       optimizer, 
                       optuna_task, 
                       bce_thr = 0.5,
                       amp = True,
                       clip_grad = 1.0)
            
            # ┏━━━━━━━━━━ Validation ━━━━━━━━━━┓
            vloss, vacc, vprec, vrec, vf1 = epoch_loop(model, 
                                                       val_ld,   
                                                       criterion, 
                                                       DEVICE, 
                                                       optimizer = None,
                                                       task_name = optuna_task,
                                                       bce_thr = 0.5, 
                                                       amp = True, 
                                                       clip_grad = 0.0)
            
            # ┏━━━━━━━━━━ Early Stopping ━━━━━━━━━━┓
            stop(vloss, model)
            step_scheduler(scheduler, epoch, vloss)
            if stop.early_stop:
                break
            
            # ┏━━━━━━━━━━ If this epoch is the best so far, record its metrics ━━━━━━━━━━┓
            if vloss < best_loss:
                best_loss = vloss
                best_acc  = vacc
                best_prec = vprec
                best_f1   = vf1
                best_state = copy.deepcopy(model.state_dict())
        
        # ┏━━━━━━━━━━ Store Best Precision in Test ━━━━━━━━━━┓
        if fold_idx == n_folds - 1:
            
            model.load_state_dict(best_state)
            # ┏━━━━━━━━━━ Original split's Model Evaluation on Test Set ━━━━━━━━━━┓
            with torch.no_grad():
                t_loss, t_acc, t_prec, t_rec, t_f1 = epoch_loop(model, 
                                                                test_loader, 
                                                                criterion, 
                                                                DEVICE,
                                                                optimizer = None, 
                                                                task_name = optuna_task, 
                                                                bce_thr   = 0.5, 
                                                                amp       = False, 
                                                                clip_grad = 0.0)
            test_precision = t_prec

        # ┏━━━━━━━━━━ Store Best Validation Loss and Metrics (per fold) ━━━━━━━━━━┓
        fold_losses.append(best_loss)
        fold_accs.append(best_acc)
        fold_precs.append(best_prec)
        fold_f1s.append(best_f1)

        # ┏━━━━━━━━━━ Empty Caché ━━━━━━━━━━┓
        torch.cuda.empty_cache()

    return {"val_loss":       float(np.mean(fold_losses)),
            "val_accuracy":   float(np.mean(fold_accs)),
            "val_precision":  float(np.mean(fold_precs)),
            "val_f1":         float(np.mean(fold_f1s)),
            "test_precision": test_precision}

# ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
# ┃ 6. OPTUNA OBJECTIVE                                                   ┃
# ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
def objective(trial: optuna.Trial, dataset: TensorDataset, props: List[float], optuna_task) -> float:
    # ┏━━━━━━━━━━ Parameters to Optimize by Optuna ━━━━━━━━━━┓
    params = {
        # ┏━━━━━━━━━━ 1. CNN Architecture ━━━━━━━━━━┓
        "cnn_embed_dim":  trial.suggest_categorical("cnn_embed_dim", [64, 128, 256]),         # Embedding dimension
        "cnn_kernel":     trial.suggest_categorical("cnn_kernel", [8, 16, 32]),               # CNN kernel size
        "cnn_stride":     trial.suggest_categorical("cnn_stride", [1, 2, 4]),                 # CNN stride
        "p_pos_drop":     trial.suggest_float("p_pos_drop", 0.0, 0.5),                        # Positional-encoding dropout probability


        # ┏━━━━━━━━━━ 2. Transformer Architecture ━━━━━━━━━━┓
        "heads":      trial.suggest_categorical("heads", [2, 4, 8]),                          # Attention heads
        "layers":     trial.suggest_int("layers", 2, 6),                                      # Number of transformer layers        
        "ffn_dim":    trial.suggest_categorical("ffn_dim", [2, 4, 8]),                        # Feed-forward network multiplier (× embed)
        "dropout":    trial.suggest_float("dropout", 0.0, 0.5),                               # Transformer dropout probability
        "activation": trial.suggest_categorical("activation", ["gelu", 
                                                                "relu", 
                                                                "silu", 
                                                                "mish"]),
        
        # ┏━━━━━━━━━━ 3. MLP Architecture ━━━━━━━━━━┓
        "mlp_hidden":     trial.suggest_categorical("mlp_hidden", [256, 512, 768]),           # MLP hidden size
        "mlp_dropout":    trial.suggest_float("mlp_dropout", 0.0, 0.5),                       # MLP dropout probability
        "mlp_pooling":    trial.suggest_categorical("mlp_pooling", ["attention", "meanmax"]), # MLP pooling mechanism
        "mlp_activation": trial.suggest_categorical("mlp_activation", ["gelu",                # MLP activation function
                                                                        "relu", 
                                                                        "silu", 
                                                                        "mish"]),              
    

        # ┏━━━━━━━━━━ Hyperparameters ━━━━━━━━━━┓
        "batch_size":    2 ** trial.suggest_int("batch_pow", 5, 8),                           # Batch size (2**batch_pow)
        "lr":            trial.suggest_float("lr", 1e-16, 1e-2, log=True),                    # Learning rate
        "weight_decay":  trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True),           # Weight decay
        "loss_function": "cross_entropy",                                                           # Loss type
        #"bce_thr":        (trial.suggest_float("thr", 0.1, 0.7)                              # BCE threshold (only for BCE loss)
                            #if loss_name == "bce" else 0.5),
        
        
        # ┏━━━━━━━━━━ Scheduler ━━━━━━━━━━┓
        "sch_name":          trial.suggest_categorical("sch_name", ["none",                         # Scheduler Name
                                                            "power", 
                                                            "linear", 
                                                            "cosine", 
                                                            "plateau", 
                                                            "linear_plateau", 
                                                            "cosine_plateau"]),
        "warmup_epochs":     trial.suggest_int("warmup_epochs", 9, 10),                             # Warm-up epochs
        "plateau_patience":  trial.suggest_int("plateau_patience", 5, 20),                      # Plateau patience
        "plateau_factor":    trial.suggest_float("plateau_factor", 0.2, 0.5),                   # Learning-rate reduction factor
        "power_s":           trial.suggest_float("power_s", 0.2, 0.5),                          # Power Scheduling epoch divider
        "power_c":           trial.suggest_float("power_c", 0.2, 0.5),                          # Power Scheduling exponent
        

        # ┏━━━━━━━━━━ Optimizer ━━━━━━━━━━┓
        "optimizer":  trial.suggest_categorical("optimizer", ["adagrad",                         # Optimizer Name
                                                            "rmsprop", 
                                                            "adam", 
                                                            "adamw", 
                                                            "adabelief", 
                                                            "ranger"]),
        # ┏━━━━━━━━━━ Optimizer tuning ━━━━━━━━━━┓
        "eps":             trial.suggest_float("eps", 1e-8, 1e-4, log=True),
        "lr_decay":        trial.suggest_float("lr_decay", 1e-6, 1e-1, log=True),
        "rms_alpha":           trial.suggest_float("rms_alpha", 0.8, 0.999),
        "rms_momentum":        trial.suggest_float("rms_momentum", 0.0, 0.9),
        "weight_decouple": trial.suggest_categorical("weight_decouple", [True, False]),
        "rectify":         trial.suggest_categorical("rectify", [True, False]),
        "lookahead_k":     trial.suggest_int("lookahead_k", 1, 10),
        "lookahead_alpha": trial.suggest_float("lookahead_alpha", 0.1, 0.9),
    }

    # ┏━━━━━━━━━━ Betas (Optimizer) & Packing ━━━━━━━━━━┓
    beta1     = trial.suggest_float("beta1", 0.9, 0.95, step=0.05)
    beta2     = trial.suggest_float("beta2", 0.98, 0.999, step=0.019)
    params["betas"] = (beta1, beta2)

    # ┏━━━━━━━━━━ CV_Folds Call ━━━━━━━━━━┓
    metrics = cv_folds(dataset, params, props, optuna_task)
    
    # ┏━━━━━━━━━━ Record metrics on the Trial ━━━━━━━━━━┓
    trial.set_user_attr("val_loss",       metrics["val_loss"])
    trial.set_user_attr("val_precision",  metrics["val_precision"])
    trial.set_user_attr("test_precision", metrics["test_precision"])
    trial.set_user_attr("val_accuracy",   metrics["val_accuracy"])
    trial.set_user_attr("val_f1",         metrics["val_f1"])
    
    return metrics["val_loss"], metrics["test_precision"]


# ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
# ┃ 7. MAIN FUNCTION                                                      ┃
# ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
if __name__ == "__main__":
    # ┏━━━━━━━━━━ 7.a) Omit Warnings ━━━━━━━━━━┓
    warnings.filterwarnings("ignore", category=UserWarning)

    # ┏━━━━━━━━━━ 7.b) Reading CSV ━━━━━━━━━━┓
    df_spy = pd.read_csv(dataset_path("Bolt", "Equities", "SPY", "merge"), parse_dates=["date"])
    dataset = prepare_dataset(df_spy, seq_len=seq_len)
    
    # ┏━━━━━━━━━━ 7.c) Prepare Optuna Storage ━━━━━━━━━━┓
    base = Path(__file__).parent
    db_path = base / "optuna_study.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)      # make sure the directory exists
    storage_url = f"sqlite:///{db_path.resolve()}"         # must include the file name

    # ┏━━━━━━━━━━ 7.d) Instantiation of Optuna & Optimization ━━━━━━━━━━┓
    study = optuna.create_study(study_name     = "CTTS_Study_Big",
                                storage        = storage_url,
                                directions     = ["minimize", "maximize"],                           # Minimize or maximize the objective
                                pruner         = optuna.pruners.MedianPruner(n_startup_trials=10),   # Stops unpromising trials early (
                                load_if_exists = False)                                               # to save time) by comparing with median 
                                                                                                     # of 10 completed trials.
                   
                                                                                                
    # ┏━━━━━━━━━━ 7.d) Optuna Optimization ━━━━━━━━━━┓
    study.optimize(lambda trial: objective(trial,
                            dataset,
                            cross_val_props,
                            optuna_task),
                   n_trials = ARGS.trials,
                   show_progress_bar = True)
    
    # ┏━━━━━━━━━━ 7.e) JSON Storage with Pareto-optimal trials ━━━━━━━━━━┓
    pareto = []
    for t in study.best_trials:
        if t.state.is_finished():
            pareto.append({
                "number":         t.number,
                "values":         t.values,
                "val_loss":       t.user_attrs.get("val_loss"),
                "val_precision":  t.user_attrs.get("val_precision"),
                "test_precision": t.user_attrs.get("test_precision"),
                "val_accuracy":   t.user_attrs.get("val_accuracy"),
                "val_f1":         t.user_attrs.get("val_f1"),
                "params":         t.params,
            })

    out_path = base / "optuna_pareto.json"
    with open(out_path, "w") as f:
        json.dump(pareto, f, indent=2)

    print(f"\nSaved {len(pareto)} Pareto-optimal trials to {out_path}.")


    # ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
    # ┃ 8) LAUNCH TRAINING ON ALL PARETO TRIALS                       ┃
    # ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
    # ┏━━━━━━━━━━ Call train_trials.py ━━━━━━━━━━┓
    result = subprocess.run(
        ["python3", str(base / "train_trials.py")],
        check=False,
        capture_output=True,
        text=True,
    )

    # ┏━━━━━━━━━━ Safety Check ━━━━━━━━━━┓
    if result.returncode != 0:
        print("🚨 train_all.py failed with exit code", result.returncode)
        print("STDOUT:\n", result.stdout)
        print("STDERR:\n", result.stderr)
    else:
        print("✅ train_all.py finished successfully.")


    
    