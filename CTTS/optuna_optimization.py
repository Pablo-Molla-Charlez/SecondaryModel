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
from tqdm.auto import tqdm

from pathlib import Path
from typing import List, Optional, Dict, Any
from torch.utils.data import TensorDataset
from torch.utils.tensorboard import SummaryWriter
from paths import dataset_path

# ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
# ┃ 2. SPECIFIC CLASSES & FUNCTIONS IMPORTS                               ┃
# ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛

# ┏━━━━━━━━━━ Training utils ━━━━━━━━━━┓
from model import CTTSModel, EarlyStopping
from Utils.train_utils import (epoch_loop, 
                              seed_everything, 
                              select_threshold_fbeta, 
                              evaluate_threshold,
                              build_scores,
                              build_m1_window)

# ┏━━━━━━━━━━ Optimization utils ━━━━━━━━━━┓
from Utils.optim_utils import (make_scheduler, 
                               step_scheduler, 
                               get_optimizer)

# ┏━━━━━━━━━━ Data Preprocessing utils ━━━━━━━━━━┓
from data_preprocessing import (prepare_dataset, 
                                build_loaders, 
                                merge_meta_targets)

# ┏━━━━━━━━━━ Evaluation utils ━━━━━━━━━━┓
from Utils.test_utils import plot_cm_with_metrics

# ┏━━━━━━━━━━ Optuna utils ━━━━━━━━━━┓
from Utils.optuna_utils import (feature_map,
                                export_pareto_configs,
                                parse_optuna_objectives)

# ┏━━━━━━━━━━ Selective Classification ━━━━━━━━━━┓
from selective_classification import (collect_risk_coverage_curve,
                                      area_under_risk_coverage,
                                      plot_coverage_risk_curve,
                                      save_metrics,
                                      coverage_at_risk)

# ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
# ┃ 3. CONFIG & CLIENT ARGUMENTS                                          ┃
# ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
# ┏━━━━━━━━━━ 3.a) Root ━━━━━━━━━━┓
base = Path(__file__).parent
config_path = base / "config.yaml"
cfg  = yaml.safe_load(open(config_path, "r"))

# ┏━━━━━━━━━━ 3.b) Optuna Objectives ━━━━━━━━━━┓
optuna_objectives = parse_optuna_objectives(cfg["training_mode"]["optuna_objectives"])
optuna_directions = [direction for direction, _ in optuna_objectives]
optuna_metrics    = [metric for _, metric in optuna_objectives]

# ┏━━━━━━━━━━ 3.c) For the training hyper-parameters, i.e. learning rate, batch size, etc. ━━━━━━━━━━┓
train_cfg = {"UP": cfg["train_up"], "DN": cfg["train_dn"]}
provider = cfg["dataset"]["source"].capitalize()
symbol = cfg["dataset"]["symbol"].upper()

# ┏━━━━━━━━━━ 3.d) Fixed Constants ━━━━━━━━━━┓
seq_len          = cfg["sequence_length"]
COLUMN_FEATURES  = feature_map(cfg, "column_features")
CONTEXT_FEATURES = feature_map(cfg, "context_features")
train_frac       = cfg["splits"]["train"]
val_frac         = cfg["splits"]["val"]
test_frac        = cfg["splits"]["test"]

# ┏━━━━━━━━━━ 3.e) Cross-Validation ━━━━━━━━━━┓
cv_opt             = cfg["training_mode"]["cv_optuna"]
cross_val_props    = cfg["training_mode"]["cross_val_props"]
optuna_task        = cfg["training_mode"]["optuna_task"]
num_classes        = cfg["training_mode"]["num_classes"]
padding            = cfg["training_mode"]["padding"]
granularity_optuna = cfg["training_mode"]["granularity_optuna"]
granularity_slug   = granularity_optuna.replace(" ", "").replace("-", "").lower()

# ┏━━━━━━━━━━ 3.f) Meta-Label Modes ━━━━━━━━━━┓
meta_label_optuna = cfg["training_mode"]["meta_label_optuna"]
meta_suffix_optuna = "FP" if meta_label_optuna == "fp" else "TP"
meta_dir_suffix = "og" if meta_label_optuna == "original" else meta_label_optuna
granularity_slug_with_meta = f"{granularity_slug}_{meta_dir_suffix}"

# ┏━━━━━━━━━━ 3.g) Coverage Configuration ━━━━━━━━━━┓
threshold_cfg      = cfg["training_mode"].get("threshold", {})
policy             = threshold_cfg["policy"].lower()
alpha_cfg          = float(threshold_cfg["alpha"])
fbeta_cfg          = float(threshold_cfg["fbeta"])
gating_mode        = threshold_cfg["gating"].lower()
min_coverage_cfg   = float(threshold_cfg["min_coverage"])
min_selected_cfg   = float(threshold_cfg["min_selected_count"])
floor_policy       = float(threshold_cfg["floor_policy"])

# ┏━━━━━━━━━━ 3.h) Client ━━━━━━━━━━┓
cli = argparse.ArgumentParser()
cli.add_argument("--trials",  type = int, default = 500,    help = "Optuna trials")    # 500 different parameter combinations
cli.add_argument("--seed",    type = int, default = 42,     help = "Global random seed")
cli.add_argument("--device",  type = str, default = 'cuda', help = "'cuda', 'cpu' or 'mps'.  If omitted → auto-detect")
ARGS = cli.parse_args()

# ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
# ┃ 4. REPRODUCIBILITY & DEVICE                                           ┃
# ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
# ┏━━━━━━━━━━ Select GPU if available, otherwise CPU ━━━━━━━━━━┓
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\nDevice: {DEVICE}")

# ┏━━━━━━━━━━ Reproducibility ━━━━━━━━━━┓
seed_everything(1493583942)

# ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
# ┃ 5. CROSS-VALIDATION RUN                                               ┃
# ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
def cv_folds(ds, 
             params, 
             props, 
             task, 
             trial_number: int, 
             run_id: str, 
             trial: Optional[optuna.Trial] = None,
             m1_window: Optional[np.ndarray] = None) -> dict[str, float]:

    # ┏━━━━━━━━━━ Task-specific Features ━━━━━━━━━━┓
    task = task.upper()
    task_columns = COLUMN_FEATURES[task]
    task_context = CONTEXT_FEATURES[task]
    cm_labels = (f'No_{meta_suffix_optuna}_{task}', f'{meta_suffix_optuna}_{task}')

    # ┏━━━━━━━━━━ 5.a) Build Tensors ━━━━━━━━━━┓
    seed_everything(1493583942)
    folds, test_loader = build_loaders(ds,
                                       cross_validation = cv_opt,                  # Must be True
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
    n_folds            = len(folds)
    fold_losses        = []
    fold_accs          = []
    fold_precs         = []
    fold_f1s           = []
    fold_recs          = []
    fold_fbetas        = []
    # val_thresholds     = []
    # val_metrics        = []
    # val_aurcs          = []
    # last_val_curve     = None
    # last_val_preds     = None
    # last_val_targets   = None
    # last_val_metrics   = None
    # last_tau           = None
    # last_val_aurc      = None
    # last_fold_metric   = None
    
    # ┏━━━━━━━━━━ Prepare directory for this Trial's Confusion Matrices (Val & Test) & TensorBoard ━━━━━━━━━━┓
    optuna_task_dir = (base / cfg["paths"]["output_root"] / "Optuna" / provider
                       / symbol / task / granularity_slug_with_meta)
    trial_dir = optuna_task_dir / run_id / f"Trial_{trial_number}"
    ckpt_dir  = trial_dir / "checkpoints"
    cm_dir    = trial_dir / "confusion_matrices"

    # ┏━━━━━━━━━━ TensorBoard Directory (holding the writer here once we hit the last fold) ━━━━━━━━━━┓
    tb_trial_dir = (base / cfg["paths"]["output_root"] / "Optuna" / "Tensorboard"
                    / provider / symbol / task / granularity_slug_with_meta / run_id)
    tb_trial_dir.mkdir(parents = True, exist_ok = True)
    writer = None

    # ┏━━━━━━━━━━ Warn if M1 window length does not match dataset length ━━━━━━━━━━┓
    if m1_window is not None and len(m1_window) != len(ds):
        print(f"[WARN] M1 window length ({len(m1_window)}) does not match dataset length ({len(ds)}). "
              "Risk-budget alignment may be incorrect.")

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
            context_len   = seq_len + len(task_context),  # Sequence length plus appended context tokens
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
        epoch_iter = tqdm(range(max_epochs),
                          desc          = f"Trial Nb {trial_number}, Task: {task}, Fold {fold_idx + 1}/{n_folds}",
                          position      = 1,
                          leave         = False,
                          dynamic_ncols = True)

        for epoch in epoch_iter:
            # ┏━━━━━━━━━━ Train ━━━━━━━━━━┓
            train_result = epoch_loop(model,
                                      train_ld,
                                      criterion,
                                      DEVICE,
                                      mode        = "train",
                                      task_name   = task,
                                      optimizer   = optimizer,
                                      bce_thr     = 0.5,
                                      amp         = True,
                                      clip_grad   = 1.0,
                                      beta        = train_cfg[task]["fbeta"],
                                      return_raw  = False)
            
            # ┏━━━━━━━━━━ Train Loss ━━━━━━━━━━┓
            train_loss = train_result["loss"]
            
            # ┏━━━━━━━━━━ Validation ━━━━━━━━━━┓
            val_result = epoch_loop(model,
                                    val_ld,
                                    criterion,
                                    DEVICE,
                                    mode       = "val",
                                    task_name  = task,
                                    optimizer  = None,
                                    bce_thr    = 0.5,
                                    amp        = False,
                                    clip_grad  = 0.0,
                                    beta       = train_cfg[task]["fbeta"],
                                    return_raw = True)
            
            # ┏━━━━━━━━━━ Validation Loss & Metrics ━━━━━━━━━━┓
            val_loss = val_result["loss"]
            vacc     = val_result["acc"]
            vprec    = val_result["prec"]
            vrec     = val_result["rec"]
            vf1      = val_result["f1"]
            vfbeta   = val_result["fbeta"]
            vtargets = val_result["targets"]
            vprobs   = val_result["probs"]
                        
            # ┏━━━━━━━━━━ Log to TensorBoard if last fold ━━━━━━━━━━┓
            if writer is not None:
                writer.add_scalar("Loss/train",      train_loss, epoch)
                writer.add_scalar("Loss/validation", val_loss  , epoch)
            
            epoch_iter.set_postfix(train=f"{train_loss:.4f}", val=f"{val_loss:.4f}")

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
        
        epoch_iter.close()

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
            assert best_state is not None, "No Model saved during Training"
            model.load_state_dict(best_state)

            # ┏━━━━━━━━━━ Original split's Model Evaluation on Validation Set & Confusion Matrix ━━━━━━━━━━┓
            val_eval = epoch_loop(model,
                                  val_ld,
                                  criterion,
                                  DEVICE,
                                  mode        = "val",
                                  task_name   = task,
                                  optimizer   = None,
                                  bce_thr     = 0.5,
                                  amp         = False,
                                  clip_grad   = 0.0,
                                  beta        = train_cfg[task]["fbeta"],
                                  return_raw  = True)
            
            # ┏━━━━━━━━━━ Extraction of raw predictions & probabilities ━━━━━━━━━━┓
            vpreds_raw = val_eval["preds"]
            vtargets   = val_eval["targets"]
            vprobs     = val_eval.get("probs")
            if vprobs is None:
                raise ValueError("Validation probabilities were not produced; ensure the criterion supports probability extraction.")

            # ┏━━━━━━━━━━ Adapt raw probabilities to Gating Policy & Threshold Uniqueness with endpoints ━━━━━━━━━━┓ 
            val_scores = build_scores(vprobs, gating_mode)

            """

            HERE IT MIGHT NOT WORK FOR TP AND FP SINCE IT'S INDEX BASED AND NOT TIME-BASED.

            """
            # ┏━━━━━━━━━━ Aligning Validation Indices for M1 Masking (Aware of Subsets) ━━━━━━━━━━┓
            if hasattr(val_ld.dataset, "indices"):
                val_indices = np.asarray(val_ld.dataset.indices, dtype=int)
            else:
                print("[WARN] Validation dataset has no 'indices' attribute; "
                        "falling back to np.arange(len(scores)). Alignment with M1 window may be incorrect.")
                val_indices = np.arange(val_scores.shape[0], dtype=int)

            # ┏━━━━━━━━━━ Extracting Validation Indices for M1 Masking (Aware of Subsets) ━━━━━━━━━━┓
            val_m1_mask = None
            if m1_window is not None:
                val_m1_mask = m1_window[val_indices].astype(bool)

            """

            HERE IT MIGHT NOT WORK FOR TP AND FP SINCE IT'S INDEX BASED AND NOT TIME-BASED.

            """
            # ┏━━━━━━━━━━ Sanity Check ━━━━━━━━━━┓
            if val_m1_mask is None:
                raise ValueError("M1 mask unavailable for risk_budget policy during validation.")

            # ┏━━━━━━━━━━ Align M1 Window to M2 Validation Subset ━━━━━━━━━━┓
            if not val_m1_mask.any():
                # ┏━━━━━━━━━━ Protects against empty Val splits ━━━━━━━━━━┓
                raise ValueError("No M1-positive samples in validation split for risk_budget policy.")
            
            # ┏━━━━━━━━━━ Apply mask to both targets (GT) and scores (probs) to keep them aligned ━━━━━━━━━━┓
            val_targets_policy = vtargets[val_m1_mask]
            val_scores_policy  = val_scores[val_m1_mask]

            # ┏━━━━━━━━━━ Sanity Alignment Check ━━━━━━━━━━┓
            if val_targets_policy.shape[0] != val_scores_policy.shape[0]:
                raise AssertionError(f"Validation targets ({val_targets_policy.shape[0]}) and scores ({val_scores_policy.shape[0]}) mismatch.")
            
            # ┏━━━━━━━━━━ Gating Policy Probabilities & Threshold Uniqueness with endpoints ━━━━━━━━━━┓ 
            finite_scores = val_scores_policy[np.isfinite(val_scores_policy)]
            if finite_scores.size == 0:
                # No usable scores, at least create a trivial curve
                thresholds_unique = np.array([0.0, 1.0])
                print("[WARN] No finite validation scores available; using trivial thresholds {0, 1}.")
            else:
                thresholds_unique = np.unique(finite_scores)
                thresholds_unique = np.unique(np.concatenate([thresholds_unique, np.array([0.0, 1.0])]))
 
            # ┏━━━━━━━━━━ Risk/Coverage Curve Creation with Sweeping Thresholds ━━━━━━━━━━┓ 
            val_curve = collect_risk_coverage_curve(y_true = val_targets_policy,
                                                    y_score = val_scores_policy, 
                                                    thresholds = thresholds_unique, 
                                                    include_error_counts = True)

            # ┏━━━━━━━━━━ Area under the validation risk–coverage curve ━━━━━━━━━━┓ 
            val_aurc = area_under_risk_coverage(curve = val_curve)
        
            # ┏━━━━━━━━━━ Threshold Policy Selection ━━━━━━━━━━┓ 
            risk_constraints_flag = True
            if policy == "risk_budget":
                # ┏━━━━━━━━━━ Coverage associated to user-defined Risk ━━━━━━━━━━┓ 
                selection = coverage_at_risk(curve        = val_curve,
                                             max_risk     = alpha_cfg,
                                             min_coverage = min_coverage_cfg,
                                             min_selected = min_selected_cfg)
                    
                # ┏━━━━━━━━━━ Best Threshold (according to Policy) ━━━━━━━━━━┓ 
                selected_tau = float(selection["threshold"])
                best_metric = float(selection["coverage"])
                
                # ┏━━━━━━━━━━ Risk & Coverage corresponding metrics to Selected Threshold ━━━━━━━━━━┓ 
                selection_metric = {"best_risk": selection["risk"],
                                    "best_coverage": selection["coverage"],
                                    "risk_constraint_satisfied": bool(selection["constraint_satisfied"])}
                risk_constraints_flag = bool(selection["constraint_satisfied"])

                # ┏━━━━━━━━━━ Pruning Filter ━━━━━━━━━━┓ 
                if not risk_constraints_flag:                    
                    return {"Policy":              policy,
                            "mean_val_loss":       float(np.mean(fold_losses)),
                            "best_metric":         best_metric,
                            "Risk_Constraints":    selection["constraint_satisfied"]}
            
                
            elif policy == "f_beta":
                # ┏━━━━━━━━━━ Optimized Threshold with its FBeta Metric ━━━━━━━━━━┓ 
                selected_tau, best_metric = select_threshold_fbeta(val_targets_policy,
                                                                   val_scores_policy,
                                                                   thresholds_unique,
                                                                   fbeta_cfg)

            else:
                raise ValueError(f"Unknown threshold policy: {policy}")

            # ┏━━━━━━━━━━ Optimized/Best Threshold for Validation Predictions [Last Fold] ━━━━━━━━━━┓ 
            eval_val = evaluate_threshold(val_targets_policy, val_scores_policy, selected_tau)
            val_preds_tau = eval_val.pop("predictions")

            # ┏━━━━━━━━━━ Pruning of Trial [Validation Conditions] ━━━━━━━━━━┓
            mean_val_loss = float(np.mean(fold_losses))
            if mean_val_loss <= 0.7 and best_metric >= floor_policy:
                # ┏━━━━━━━━━━ Folders Creation ━━━━━━━━━━┓
                for d in (ckpt_dir, cm_dir):
                    d.mkdir(parents = True, exist_ok = True)
                
                # ┏━━━━━━━━━━ Save that best state‐dict to disk ━━━━━━━━━━┓
                ckpt_path = ckpt_dir / f"{trial_number}_{task}_best.pt"
                torch.save(best_state, ckpt_path)

                # ┏━━━━━━━━━━ Validation Confusion Matrix & Metrics [Not Optimized Threshold] ━━━━━━━━━━┓
                plot_cm_with_metrics(vpreds_raw[val_m1_mask],
                                     vtargets[val_m1_mask],
                                     labels         = cm_labels,
                                     title          = f'M2_{task} — Val',
                                     out_dir        = cm_dir,
                                     best_threshold = None,
                                     cmap           = "Oranges")

                # ┏━━━━━━━━━━ Validation Confusion Matrix & Metrics [Optimized Threshold] ━━━━━━━━━━┓
                plot_cm_with_metrics(val_preds_tau, 
                                     vtargets[val_m1_mask],
                                     labels         = cm_labels,
                                     title          = f'M2_{task} — Val',
                                     out_dir        = cm_dir,
                                     best_threshold = selected_tau,
                                     cmap           = "Greens")

                # ┏━━━━━━━━━━ Plotting Risk & Coverage Curve ━━━━━━━━━━┓
                val_rc_png = cm_dir / f"M1+M2_{task}_Val_RiskCoverage.png"
                plot_coverage_risk_curve(curve = val_curve, 
                                         label = f"Gating: {gating_mode}", 
                                         save_path = str(val_rc_png), 
                                         show = False,
                                         highlight_point = (eval_val["coverage"], eval_val["risk"]),
                                         highlight_text = f"τ* = {selected_tau:.3f}")
                
                # ┏━━━━━━━━━━ Summary of Risk & Coverage Analysis ━━━━━━━━━━┓
                curve_rows = [{"Threshold": float(t),
                               "Coverage": float(c),
                               "Risk": float(r),
                               "Selected_Count": int(s),
                               "Error_Count": int(e) if "error_count" in val_curve else None} for t, c, r, s, e in zip(val_curve["thresholds"],
                                                                                                                       val_curve["coverage"],
                                                                                                                       val_curve["risk"],
                                                                                                                       val_curve["selected_count"],
                                                                                                                       val_curve.get("error_count", [0] * len(val_curve["thresholds"])))
                              ]

                payload = {"Dataset":                "Validation",
                           "Task":                   task,
                           "Policy":                 policy,
                           "Gating":                 gating_mode,
                           "Val_AURC":               val_aurc,
                           "Val_Tau":                selected_tau,
                           "Val_Coverage@Tau":       eval_val["coverage"],
                           "Val_Risk@Tau":           eval_val["risk"],
                           "Val_Selected_Count@Tau": eval_val["selected_count"]}
            
                # ┏━━━━━━━━━━ Adding Data to Summary of Risk & Coverage Analysis ━━━━━━━━━━┓
                if policy == "risk_budget":
                    payload["Alpha"] = alpha_cfg
                    payload.update(selection_metric)
                    payload["Min_Coverage"] = min_coverage_cfg
                    payload["Min_Selected_Count"] = min_selected_cfg
                    payload["Risk_Constraints"] = risk_constraints_flag
                
                # ┏━━━━━━━━━━ Adding Data to Summary of FBeta Analysis ━━━━━━━━━━┓
                elif policy == "f_beta":
                    payload["FBeta"]     = fbeta_cfg
                    payload["FBeta@Tau"] = best_metric

                # ┏━━━━━━━━━━ Original split's Model Evaluation on Test Set & Confusion Matrix ━━━━━━━━━━┓
                test_eval = epoch_loop(model,
                                      test_loader,
                                      criterion,
                                      DEVICE,
                                      mode = "test",
                                      task_name = task,
                                      optimizer = None,
                                      bce_thr = 0.5,
                                      amp = False,
                                      clip_grad = 0.0,
                                      beta = train_cfg[task]["fbeta"],
                                      return_raw = True)

                # ┏━━━━━━━━━━ Test Predictions & Probabilities [Not Optimized Threshold] ━━━━━━━━━━┓
                tpreds   = test_eval["preds"]
                ttargets = test_eval["targets"]
                tprobs   = test_eval.get("probs")
                if tprobs is None:
                    raise ValueError("Test probabilities were not produced; ensure the criterion supports probability extraction.")

                # ┏━━━━━━━━━━ Risk & Coverage Scores [Optimized Threshold] ━━━━━━━━━━┓
                test_scores = build_scores(tprobs, gating_mode)

                """

                HERE IT MIGHT NOW WORK FOR TP AND FP SINCE IT'S INDEX BASED AND NOT TIME-BASED.

                """

                # ┏━━━━━━━━━━ Aligning Test Indices for M1 Masking (Aware of Subsets) ━━━━━━━━━━┓
                if hasattr(test_loader.dataset, "indices"):
                    test_indices = np.asarray(test_loader.dataset.indices, dtype=int)
                else:
                    print("[WARN] Test dataset has no 'indices' attribute; "
                          "falling back to np.arange(len(scores)). Alignment with M1 window may be incorrect.")
                    test_indices = np.arange(test_scores.shape[0], dtype=int)
                
                if test_indices.size != test_scores.shape[0]:
                    # ┏━━━━━━━━━━ Protects against different lengths ━━━━━━━━━━┓
                    raise ValueError("Test loader indices do not match the number of test scores."
                                     f" Expected {test_scores.shape[0]} entries, got {test_indices.size}.")

                """

                HERE IT MIGHT NOW WORK FOR TP AND FP SINCE IT'S INDEX BASED AND NOT TIME-BASED.

                """

                # ┏━━━━━━━━━━ M1 Predictions Aligned to Test Set & Test Mask ━━━━━━━━━━┓
                m1_test_preds = (m1_window[test_indices].astype(int) if m1_window is not None else np.zeros_like(ttargets))
                test_m1_mask = m1_test_preds.astype(bool) if m1_window is not None else None

                # ┏━━━━━━━━━━ Sanity Check ━━━━━━━━━━┓
                if test_m1_mask is None:
                    raise ValueError("M1 mask unavailable for risk_budget policy during test.")
                if not test_m1_mask.any():
                    raise ValueError("No M1-positive samples in test split for risk_budget policy.")

                # ┏━━━━━━━━━━ Apply mask to both targets (GT) and scores (probs) to keep them aligned ━━━━━━━━━━┓
                ttargets_policy = ttargets[test_m1_mask]
                test_scores_policy = test_scores[test_m1_mask]

                 # ┏━━━━━━━━━━ Final safety: targets and scores must stay aligned ━━━━━━━━━━┓
                if ttargets_policy.shape[0] != test_scores_policy.shape[0]:
                    raise AssertionError(f"Test targets ({ttargets_policy.shape[0]}) and scores ({test_scores_policy.shape[0]}) mismatch.")

                # ┏━━━━━━━━━━ Optimized/Best Threshold for Test Predictions ━━━━━━━━━━┓ 
                eval_test_tau = evaluate_threshold(ttargets_policy, test_scores_policy, selected_tau)
                test_preds_tau = eval_test_tau.pop("predictions")

                # ┏━━━━━━━━━━ Ensure predictions length matches test length ━━━━━━━━━━┓
                if test_preds_tau.shape[0] != ttargets[test_m1_mask].shape[0]:
                    raise AssertionError("Length of test predictions at τ does not match test targets.")
                
                # ┏━━━━━━━━━━ Test Metrics ━━━━━━━━━━┓
                payload["Test_Coverage@Tau"]       = eval_test_tau["coverage"]
                payload["Test_Risk@Tau"]           = eval_test_tau["risk"]
                payload["Test_Selected_Count@Tau"] = eval_test_tau["selected_count"]
                
                # ┏━━━━━━━━━━ Validation Risk & Coverage Curves ━━━━━━━━━━┓
                payload["Val_Curves"] = curve_rows

                # ┏━━━━━━━━━━ Path & Save Summary ━━━━━━━━━━┓
                payload_json = cm_dir / f"M2_{task}_R&C_Analysis.json"
                save_metrics(payload, str(payload_json))

                # ┏━━━━━━━━━━ Test Confusion Matrix & Metrics [Not Optimized Threshold] ━━━━━━━━━━┓
                plot_cm_with_metrics(tpreds[test_m1_mask],
                                     ttargets[test_m1_mask],
                                     labels         = cm_labels,
                                     title          = f'M2_{task} — Test',
                                     out_dir        = cm_dir,
                                     best_threshold = None,
                                     cmap           = "Blues")
                
                # ┏━━━━━━━━━━ Test Confusion Matrix & Metrics [Optimized Threshold] ━━━━━━━━━━┓
                plot_cm_with_metrics(test_preds_tau,
                                     ttargets[test_m1_mask],
                                     labels         = cm_labels,
                                     title          = f'M2_{task} — Test',
                                     out_dir        = cm_dir,
                                     best_threshold = selected_tau,
                                     cmap           = "Purples")

                # ┏━━━━━━━━━━ Empty Caché ━━━━━━━━━━┓
                torch.cuda.empty_cache()

                if policy == "f_beta":
                    return {"Policy":           policy,
                            "mean_val_loss":    float(np.mean(fold_losses)),
                            "best_metric":      best_metric}
                else:
                    return {"Policy":           policy,
                            "mean_val_loss":    float(np.mean(fold_losses)),
                            "best_metric":      best_metric,
                            "Risk_Constraints": risk_constraints_flag}   
            
            # ┏━━━━━━━━━━ Non-Evaluated Trial ━━━━━━━━━━┓
            else:
                # ┏━━━━━━━━━━ Empty Caché ━━━━━━━━━━┓
                torch.cuda.empty_cache()
                
                if policy == "f_beta":
                    return {"Policy":           policy,
                            "mean_val_loss":    float(np.mean(fold_losses)),
                            "best_metric":      best_metric}
                else:
                    return {"Policy":           policy,
                            "mean_val_loss":    float(np.mean(fold_losses)),
                            "best_metric":      best_metric,
                            "Risk_Constraints": risk_constraints_flag}   



# ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
# ┃ 6. OPTUNA OBJECTIVE                                                   ┃
# ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
def objective(trial: optuna.Trial, 
              dataset: TensorDataset, 
              props: List[float], 
              task: str, 
              run_id: str,
              m1_window: Optional[np.ndarray]) -> float:

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
    beta1 = trial.suggest_float("beta1", 0.9, 0.95, step = 0.05)
    beta2 = trial.suggest_float("beta2", 0.98, 0.999, step = 0.019)
    params["betas"] = (beta1, beta2)

    # ┏━━━━━━━━━━ CV_Folds Call ━━━━━━━━━━┓
    seed_everything(1493583942)
    metrics = cv_folds(dataset, 
                       params, 
                       props, 
                       task, 
                       trial.number, 
                       run_id, 
                       trial = trial, 
                       m1_window = m1_window)

    # ┏━━━━━━━━━━ Filter Results ━━━━━━━━━━┓
    if metrics["Policy"] == "risk_budget" and not metrics["Risk_Constraints"]:
        print("\nRisk Constraints: ", metrics["Risk_Constraints"])
        raise optuna.TrialPruned()
    
    elif metrics["mean_val_loss"] > 0.7 or metrics["best_metric"] < floor_policy:
        print(f"\nMean Val Loss {metrics['mean_val_loss']:.3f}: ",
              metrics["mean_val_loss"] <= 0.7,
              f"Cov {metrics['best_metric']:.3f}> Floor {floor_policy:.3f}: ",
              metrics["best_metric"] >= floor_policy)
        raise optuna.TrialPruned()

    
    # ┏━━━━━━━━━━ Record metrics on the Trial ━━━━━━━━━━┓
    trial.set_user_attr("task",          task)
    trial.set_user_attr("mean_val_loss", metrics["mean_val_loss"])
    trial.set_user_attr("best_metric",   metrics["best_metric"])
    print("\n")
              
    # ┏━━━━━━━━━━ Extract Metric(s) to Optimize ━━━━━━━━━━┓
    objective_values: List[float] = []
    for metric_name in optuna_metrics:
        if metric_name not in metrics:
            raise KeyError(f"Metric '{metric_name}' requested in optuna_objectives is not available from the trial metrics.")
        objective_values.append(metrics[metric_name])
    
    # ┏━━━━━━━━━━ Optuna Single Objective (First) ━━━━━━━━━━┓
    if len(objective_values) == 1:
        return objective_values[0]

    # ┏━━━━━━━━━━ Optuna Double Objective ━━━━━━━━━━┓
    return tuple(objective_values)


# ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
# ┃ 7. OPTUNA PER TASK                                                    ┃
# ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
def run_optuna_for_task(dataset: TensorDataset,
                        props: List[float],
                        task: str,
                        m1_window: Optional[np.ndarray]) -> optuna.study.Study:

    """Launch an Optuna study for the provided task ("UP" | "DN")."""
    
    task = task.upper()
    if task not in train_cfg:
        raise ValueError(f"Unknown task '{task}'. Expected one of: {list(train_cfg.keys())}")

    # ┏━━━━━━━━━━ 1) Prepare Optuna Storage ━━━━━━━━━━┓
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"run_{timestamp}"
    run_root = (base / cfg["paths"]["output_root"] / "Optuna" / provider
                / symbol / task / granularity_slug_with_meta / run_id)
    run_root.mkdir(parents = True, exist_ok = True)
    db_path = run_root / f"optuna_study_{task}_{timestamp}.db"
    storage_url = f"sqlite:///{db_path.resolve()}"
    study_name = f"CTTS_Study_{task}_{timestamp}"

    # ┏━━━━━━━━━━ 2) Instantiation of Optuna Parameters ━━━━━━━━━━┓
    print(f"\n▶ Running Optuna study for task {task} with {ARGS.trials} trials (run id: {run_id})")
    study_kwargs = dict(study_name     = study_name,
                        storage        = storage_url,
                        pruner         = optuna.pruners.MedianPruner(n_startup_trials = 20),
                        load_if_exists = False)
    
    # ┏━━━━━━━━━━ Optuna Directions [Single and Dual] ━━━━━━━━━━┓
    if len(optuna_directions) == 1:
        # ┏━━━━━━━━━━ SingleOptimization ━━━━━━━━━━┓
        study = optuna.create_study(direction = optuna_directions[0], 
                                    **study_kwargs)
    else:
        # ┏━━━━━━━━━━ Dual Optimization ━━━━━━━━━━┓
        study = optuna.create_study(directions = optuna_directions, 
                                    **study_kwargs)

    # ┏━━━━━━━━━━ 3) Optuna Optimization ━━━━━━━━━━┓
    study.optimize(lambda trial: objective(trial,
                                           dataset,
                                           props,
                                           task,
                                           run_id,
                                           m1_window),
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
                            "up",
                            granularity = granularity_optuna,
                            meta_label_mode = meta_label_optuna)

    # ┏━━━━━━━━━━ 7.b) Extracting Column and Context Features ━━━━━━━━━━┓
    # ┏━━━━━━━━━━ 7.b.1) Extracting Column ━━━━━━━━━━┓
    all_columns = []
    for feat_list in COLUMN_FEATURES.values():
        for feat in feat_list:
            if feat not in all_columns:
                all_columns.append(feat)
    
    # ┏━━━━━━━━━━ 7.b.2) Extracting Column ━━━━━━━━━━┓
    all_context = []
    for feat_list in CONTEXT_FEATURES.values():
        for feat in feat_list:
            if feat not in all_context:
                all_context.append(feat)

    # ┏━━━━━━━━━━ 7.c) Reading CSV ━━━━━━━━━━┓
    df_asset = merge_meta_targets(asset_type       = cfg["dataset"]["type"],
                                  asset            = cfg["dataset"]["symbol"],
                                  data_dir         = str(Path(csv_path).parent),
                                  output_dir       = str(Path(csv_path).parent),
                                  column_features  = all_columns,
                                  context_features = all_context,
                                  meta_label_mode  = meta_label_optuna)

    # ┏━━━━━━━━━━ 7.d) Preparing Data ━━━━━━━━━━┓        
    default_task = optuna_task.upper()
    tasks = [default_task]
    if cfg["training_mode"]["optuna_both"]:
        opposite_task = "DN" if default_task == "UP" else "UP"
        if opposite_task not in tasks:
            tasks.append(opposite_task)

    datasets = {task: prepare_dataset(df_asset,
                                      seq_len = cfg["sequence_length"],
                                      column_features = COLUMN_FEATURES[task],
                                      context_features = CONTEXT_FEATURES[task],
                                      meta_label_mode = meta_label_optuna,
                                      task = task) for task in tasks}
    
    # ┏━━━━━━━━━━ 7.e) Extracting M1 Positive Predictions ━━━━━━━━━━┓
    m1_windows = {}
    for task in tasks:
        total_samples = len(datasets[task])
        try:
            m1_windows[task] = build_m1_window(df_asset, task, cfg["sequence_length"], total_samples)
        except Exception as exc:
            print(f"[WARN] Unable to build M1 window for task {task}: {exc}")
            m1_windows[task] = None

    # ┏━━━━━━━━━━ 7.e) Running BOTH (UP/DOWN) Optuna Studies ━━━━━━━━━━┓
    studies = {}
    for task in tasks:
        studies[task] = run_optuna_for_task(datasets[task], cross_val_props, task, m1_windows.get(task))

    print("\n Completed Optuna optimization for tasks:")
    for task, study in studies.items():
        if study.best_trials:
            best = study.best_trials[0]
            print(f"  • {task}: Objectives={best.values} (study: {study.study_name})")
        else:
            print(f"  • {task}: No completed trials")
