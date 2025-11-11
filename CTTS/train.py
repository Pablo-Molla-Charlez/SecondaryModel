# ┏━━━━━━━━━━ General Imports ━━━━━━━━━━┓
import os
import argparse
import datetime
import yaml
import torch
import copy
import warnings
import numpy as np
from pathlib import Path
from tqdm.auto import tqdm
from torch.utils.data import TensorDataset
from torch.utils.tensorboard import SummaryWriter

# ┏━━━━━━━━━━ Data Preprocessing utils ━━━━━━━━━━┓
from paths import dataset_path
from data_preprocessing import (merge_meta_targets, 
                                build_loaders, 
                                prepare_dataset, 
                                count_meta_targets)

# ┏━━━━━━━━━━ Optimization utils ━━━━━━━━━━┓
from Utils.optim_utils import (get_optimizer, 
                               make_scheduler, 
                               step_scheduler)

# ┏━━━━━━━━━━ Training utils ━━━━━━━━━━┓
from Utils.train_utils import (epoch_loop,
                               seed_everything,
                               task_features,
                               build_scores,
                               select_threshold_fbeta,
                               evaluate_threshold)

# ┏━━━━━━━━━━ Model Class ━━━━━━━━━━┓
from model import CTTSModel, EarlyStopping

# ┏━━━━━━━━━━ Evaluation utils ━━━━━━━━━━┓
from Utils.test_utils import (plot_cm_with_metrics, 
                              export_predictions, 
                              plot_meta_labeling_consensus)

# ┏━━━━━━━━━━ Selective Classification ━━━━━━━━━━┓
from selective_classification import (save_metrics,
                                      coverage_at_risk,
                                      collect_risk_coverage_curve,
                                      area_under_risk_coverage,
                                      plot_coverage_risk_curve)

# ┏━━━━━━━━━━ Ignoring Warning Message ━━━━━━━━━━┓
os.environ.setdefault("TORCH_CUDA_NVML_DISABLE_WARNING", "1")
warnings.filterwarnings("ignore", message="Can't initialize NVML")


def run_training(cfg: dict) -> None:
    # ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
    # ┃ 1) CONFIG & ASSET SELECTION                                           ┃
    # ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
    # ┏━━━━━━━━━━ 1.a) For the model architecture parameters i.e. CNN, Transformer, MLP ━━━━━━━━━━┓
    architecture_cfg = {"UP": cfg["model_up"], "DN": cfg["model_dn"]}

    # ┏━━━━━━━━━━ 1.b) For the training hyper-parameters, i.e. learning rate, batch size, etc. ━━━━━━━━━━┓
    train_cfg = {"UP": cfg["train_up"], "DN": cfg["train_dn"]}

    # ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
    # ┃ 2) CONFIGURATION PARAMETERS & DIRECTORY STRUCTURE                     ┃
    # ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
    # ┏━━━━━━━━━━ 2.a) Data Source & Asset Configuration ━━━━━━━━━━┓
    granularity_usual = cfg["training_mode"]["granularity_usual"]
    granularity_slug  = granularity_usual.replace(" ", "").replace("-", "").lower()
    loss_type         = cfg["training_mode"]["loss_function"].lower()
    provider          = cfg["dataset"]["source"].capitalize()
    cv_usual          = cfg["training_mode"]["cv_usual"]
    meta_label_usual  = cfg["training_mode"]["meta_label_usual"].lower()
    meta_suffix       = "FP" if meta_label_usual == "fp" else "TP"
    meta_dir_suffix   = "og" if meta_label_usual == "original" else meta_label_usual
    granularity_slug_with_meta = f"{granularity_slug}_{meta_dir_suffix}"

    # ┏━━━━━━━━━━ 2.b) Selective Classification Configuration ━━━━━━━━━━┓
    threshold_cfg    = cfg["training_mode"].get("threshold", {})
    policy           = threshold_cfg["policy"].lower()
    alpha_cfg        = float(threshold_cfg["alpha"])
    fbeta_cfg        = float(threshold_cfg["fbeta"])
    gating_mode      = threshold_cfg["gating"].lower()
    min_coverage_cfg = float(threshold_cfg["min_coverage"])
    min_selected_cfg = float(threshold_cfg["min_selected_count"])

    # ┏━━━━━━━━━━ 2.c) Data Paths ━━━━━━━━━━┓
    csv_path = dataset_path(cfg["dataset"]["source"],
                            cfg["dataset"]["type"].capitalize(),
                            cfg["dataset"]["symbol"],
                            "up",
                            granularity = granularity_usual,
                            meta_label_mode = meta_label_usual)


    # ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
    # ┃ 3) DEVICE & SEEDS                                                     ┃
    # ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
    # ┏━━━━━━━━━━ Select GPU if available, otherwise CPU ━━━━━━━━━━┓
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    # For Personal Mac
    # device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

    # ┏━━━━━━━━━━ Reproducibility ━━━━━━━━━━┓
    seed_everything(1493583942)   


    # ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
    # ┃ 4) DATA PREPARATION                                                   ┃
    # ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
    #cfg.setdefault("training_mode", {})
    normal_task = cfg["training_mode"]["normal_task"].upper()

    # ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
    # ┃ 5) DATALOADERS & CRITERION                                            ┃
    # ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
    for task in [normal_task]:
        # ┏━━━━━━━━━━ Accessing Configuration per Task ━━━━━━━━━━┓
        model_cfg = architecture_cfg[task]
        trainer_cfg = train_cfg[task]
        cm_labels = (f'No_{meta_suffix}_{task}', f'{meta_suffix}_{task}')

        # ┏━━━━━━━━━━ Extracting Column & Context Features ━━━━━━━━━━┓
        column_features = task_features(cfg, "column_features", task)
        context_features = task_features(cfg, "context_features", task)

        # ┏━━━━━━━━━━ Merge of Data [UP & DN CSVs from M1] ━━━━━━━━━━┓
        df_asset = merge_meta_targets(asset_type       = cfg["dataset"]["type"],
                                      asset            = cfg["dataset"]["symbol"],
                                      data_dir         = str(Path(csv_path).parent),
                                      output_dir       = str(Path(csv_path).parent),
                                      column_features  = column_features,
                                      context_features = context_features,
                                      meta_label_mode  = meta_label_usual)
        

        # ┏━━━━━━━━━━ Counting Meta-Labels [Positives & Negatives] ━━━━━━━━━━┓
        meta_columns = (f"is{meta_suffix}_UP", f"is{meta_suffix}_DN")
        meta_up_col, meta_dn_col = meta_columns
        meta_counts = count_meta_targets(df_asset, columns = meta_columns, task = task)
        print("Meta-Label counts:")
        for column, stats in meta_counts.items():
            formatted = ", ".join(f"{label}: {count}" for label, count in stats.items())
            print(f"  {column}: {formatted}")
        
        # ┏━━━━━━━━━━ 5.a) Preparation of Dataloaders and Training/Validation/Testing Splits with corresponding Criterion ━━━━━━━━━━┓
        # ┏━━━━━━━━━━ Preparation of Dataloaders ━━━━━━━━━━┓
        dataset_tensor = prepare_dataset(df_asset, 
                                         seq_len          = cfg["sequence_length"],
                                         column_features  = column_features,
                                         context_features = context_features,
                                         meta_label_mode  = meta_label_usual,
                                         task             = task)


        # ┏━━━━━━━━━━ 5.b) Splits and Criterion ━━━━━━━━━━┓
        seed_everything(1493583942)  # Re-seed for reproducibility
        print(f"\n────────── {task} model ──────────")
        train_folds, test_loader = build_loaders(ds               = dataset_tensor,
                                                 cross_validation = cv_usual,
                                                 target           = task,
                                                 props            = cfg["training_mode"]["cross_val_props"],
                                                 train_frac       = cfg["splits"]["train"],
                                                 val_frac         = cfg["splits"]["val"],
                                                 test_frac        = cfg["splits"]["test"],
                                                 batch_size       = trainer_cfg["batch_size"],
                                                 loss_type        = loss_type,
                                                 focal_gamma      = cfg["training_mode"]["focal_gamma"],
                                                 focal_alpha      = cfg["training_mode"]["focal_alpha"],
                                                 device           = device)
        
        # ┏━━━━━━━━━━ 5.c) Get the model parameters ━━━━━━━━━━┓
        model_kwargs = dict(
            # ┏━━━━━━━━━━ CNN Parameters ━━━━━━━━━━┓
            cnn_embed_dim = model_cfg["cnn_embed_dim"],
            cnn_kernel    = model_cfg["cnn_kernel"],
            cnn_stride    = model_cfg["cnn_stride"],
            p_pos_drop    = model_cfg["p_pos_drop"],
            nb_features   = len(column_features),
            
            # ┏━━━━━━━━━━ Transformer Parameters ━━━━━━━━━━┓
            trans_heads   = model_cfg["transformer"]["heads"],
            trans_ff      = model_cfg["transformer"]["ffn_dim"] * model_cfg["cnn_embed_dim"][-1],
            trans_layers  = model_cfg["transformer"]["layers"],
            trans_dropout = model_cfg["transformer"]["dropout"],
            trans_activ   = model_cfg["transformer"]["activation"],
            
            # ┏━━━━━━━━━━ Classifier Parameters ━━━━━━━━━━┓
            mlp_hidden    = model_cfg["classifier"]["mlp_hidden"],
            mlp_dropout   = model_cfg["classifier"]["mlp_dropout"],
            mlp_activ     = model_cfg["classifier"]["mlp_activation"],
            mlp_pooling   = model_cfg["classifier"]["mlp_pooling"],
            

            # ┏━━━━━━━━━━ Training Mode Parameters ━━━━━━━━━━┓
            num_classes   = 1 if cfg["training_mode"]["loss_function"] == "bce" else 2,
            padding       = cfg["training_mode"]["padding"],
            context_len   = cfg["sequence_length"] + len(context_features)
        )
        
        # ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
        # ┃ 6) LOOP OVER FOLDS: MODEL, OPTIMIZER, SCHEDULER & EARLY STOPPING      ┃
        # ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
        n_folds        = len(train_folds)
        fold_losses    = []

        # ┏━━━━━━━━━━ 6.a) Store validation losses & Paths ━━━━━━━━━━┓
        val_losses_all_folds = []
        task_root = (Path(cfg["paths"]["output_root"]) / "Usual" / provider
                                                                / cfg["dataset"]["symbol"] 
                                                               / task 
                                                              / granularity_slug_with_meta)
        
        tb_root = (Path(cfg["paths"]["output_root"]) / "Usual" / "Tensorboard" / provider 
                                                                              / cfg["dataset"]["symbol"])
        # ┏━━━━━━━━━━ Main Loop for Training & Validation ━━━━━━━━━━┓
        for fold_idx, (train_loader, val_loader, crit) in enumerate(train_folds):
            # ┏━━━━━━━━━━ 6.b) Checkpoints & Runs and Tensorboard Folders Creation ━━━━━━━━━━┓
            run_stamp = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
            checkpoint_dir = task_root / f"Run_{run_stamp}"
            tensorboard_dir = tb_root / f"{task}" / granularity_slug_with_meta / f"Run_{run_stamp}"

            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            tensorboard_dir.mkdir(parents=True, exist_ok=True)

            # ┏━━━━━━━━━━ 6.c) Instantiate the model (fresh for each fold) ━━━━━━━━━━┓
            seed_everything(1493583942)
            model = CTTSModel(**model_kwargs).to(device)

            # ┏━━━━━━━━━━ 6.d) Optimizer ━━━━━━━━━━┓
            optim_cfg = trainer_cfg
            optimizer = get_optimizer(model.parameters(), optim_cfg)

            # ┏━━━━━━━━━━ 6.e) Scheduler (optional - remove it when scheduler chosen) ━━━━━━━━━━┓
            sch_cfg = trainer_cfg['scheduler']
            scheduler = make_scheduler(optimizer, sch_cfg, trainer_cfg["max_epochs"])

            # ┏━━━━━━━━━━ 6.f) Early Stoppings ━━━━━━━━━━┓
            checkpoint_path = checkpoint_dir / f"{task}_best.pt"
            stopper = EarlyStopping(patience   = trainer_cfg["patience"],
                                    verbose    = False,
                                    delta      = 1e-4,
                                    path       = str(checkpoint_path),
                                    just_count = True)
            
            # ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
            # ┃ 7) TENSORBOARD WRITERS                                                ┃
            # ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
            # ┏━━━━━━━━━━ TensorBoard Terminal Command  ━━━━━━━━━━┓
            # ┏━━━━━━━━━━ 1) ls ~/miniconda3/envs/CTTS/bin/tensorboard  ━━━━━━━━━━┓
            # ┏━━━━━━━━━━ 2) ~/miniconda3/envs/CTTS/bin/tensorboard --logdir Output/runs  ━━━━━━━━━━┓
            writer = SummaryWriter(str(tensorboard_dir))
            
            # ┏━━━━━━━━━━ Best Metrics (temporary) ━━━━━━━━━━┓
            best_loss  = float('inf')
            best_acc   = 0.0
            best_prec  = 0.0
            best_rec   = 0.0
            best_f1    = 0.0
            best_fbeta = 0.0
            best_state = None        


            # ┏━━━━━━━━━━ Train & Validation ━━━━━━━━━━┓
            epoch_iter = tqdm(range(trainer_cfg["max_epochs"]),
                              desc          = f"🌀 {task} Fold {fold_idx + 1}/{n_folds}",
                              leave         = True,
                              dynamic_ncols = True)

            for epoch in epoch_iter:
                # ┏━━━━━━━━━━ Train ━━━━━━━━━━┓
                train_result = epoch_loop(model,
                                          train_loader,
                                          crit,
                                          device,
                                          mode = "train",
                                          task_name = task,
                                          optimizer = optimizer,
                                          bce_thr = 0.5,
                                          amp = True,
                                          clip_grad = 1.0,
                                          beta = trainer_cfg["fbeta"],
                                          return_raw = False)
                
                # ┏━━━━━━━━━━ Train Loss ━━━━━━━━━━┓
                train_loss = train_result["loss"]
                
                # ┏━━━━━━━━━━ Validation ━━━━━━━━━━┓
                val_result = epoch_loop(model,
                                        val_loader,
                                        crit,
                                        device,
                                        mode = "val",
                                        task_name = task,
                                        optimizer = None,
                                        bce_thr = 0.5,
                                        amp = True,
                                        clip_grad = 0.0,
                                        beta = trainer_cfg["fbeta"],
                                        return_raw = False)
                
                # ┏━━━━━━━━━━ Validation Loss & Metrics ━━━━━━━━━━┓
                val_loss = val_result["loss"]
                vacc = val_result["acc"]
                vprec = val_result["prec"]
                vrec = val_result["rec"]
                vf1 = val_result["f1"]
                vfbeta = val_result["fbeta"]
                            
                # ┏━━━━━━━━━━ Log to TensorBoard if last fold ━━━━━━━━━━┓
                if writer is not None:
                    writer.add_scalar("Loss/train",      train_loss, epoch)
                    writer.add_scalar("Loss/validation", val_loss  , epoch)
                
                epoch_iter.set_postfix(train = f"{train_loss:.4f}", val = f"{val_loss:.4f}")

                # ┏━━━━━━━━━━ Early Stopping ━━━━━━━━━━┓
                stopper(val_loss, model)
                step_scheduler(scheduler, epoch, val_loss)
                if stopper.early_stop:
                    break
                
                # ┏━━━━━━━━━━ If this epoch is the best so far, record its metrics ━━━━━━━━━━┓
                if val_loss < best_loss:
                    best_loss  = val_loss
                    best_acc   = vacc
                    best_prec  = vprec
                    best_f1    = vf1
                    best_rec   = vrec
                    best_fbeta = vfbeta
                    best_state = copy.deepcopy(model.state_dict())
            
            epoch_iter.close()
            
            # ┏━━━━━━━━━━ Close writer after last fold training ━━━━━━━━━━┓
            if writer is not None:
                writer.close()
                writer = None

            # ┏━━━━━━━━━━ Store Best Validation Loss (per fold) ━━━━━━━━━━┓
            fold_losses.append(best_loss)

            # ┏━━━━━━━━━━ Store Best Precision in Test ━━━━━━━━━━┓
            if fold_idx == n_folds - 1:
                # ┏━━━━━━━━━━ Reload the best state into model ━━━━━━━━━━┓
                assert best_state is not None, "No Model saved during Training"
                model.load_state_dict(best_state)

                # ┏━━━━━━━━━━ Save that best state-dict to disk ━━━━━━━━━━┓
                ckpt_path = checkpoint_dir / f'{task}_best.pt'
                torch.save(best_state, ckpt_path)

                # ┏━━━━━━━━━━ Validation evaluation (raw outputs cached) ━━━━━━━━━━┓
                val_eval = epoch_loop(model,
                                      val_loader,
                                      crit,
                                      device,
                                      mode = "val",
                                      task_name = task,
                                      optimizer = None,
                                      bce_thr = 0.5,
                                      amp = True,
                                      clip_grad = 0.0,
                                      beta = trainer_cfg["fbeta"],
                                      return_raw = True)
                
                # ┏━━━━━━━━━━ Extraction of raw predictions & probabilities ━━━━━━━━━━┓
                vpreds_raw = val_eval["preds"]
                vtargets   = val_eval["targets"]
                vprobs     = val_eval.get("probs")
                if vprobs is None:
                    raise ValueError("Validation probabilities were not produced; ensure the criterion supports probability extraction.")

                # ┏━━━━━━━━━━ Adapt raw probabilities to Gating Policy & Threshold Uniqueness with endpoints ━━━━━━━━━━┓ 
                val_scores        = build_scores(vprobs, gating_mode)
                finite_scores     = val_scores[np.isfinite(val_scores)]
                thresholds_unique = np.unique(finite_scores)
                thresholds_unique = np.unique(np.concatenate([thresholds_unique, np.array([0.0, 1.0])]))
                
                # ┏━━━━━━━━━━ Risk/Coverage Curve Creation with Sweeping Thresholds ━━━━━━━━━━┓ 
                val_curve = collect_risk_coverage_curve(y_true = vtargets,
                                                        y_score = val_scores, 
                                                        thresholds = thresholds_unique, 
                                                        include_error_counts = True)

                # ┏━━━━━━━━━━ Area under the validation risk–coverage curve ━━━━━━━━━━┓ 
                val_aurc = area_under_risk_coverage(curve = val_curve)

                # ┏━━━━━━━━━━ Threshold Policy Selection ━━━━━━━━━━┓
                if policy == "risk_budget":
                    # ┏━━━━━━━━━━ Coverage associated to user-defined Risk ━━━━━━━━━━┓ 
                    selection = coverage_at_risk(curve        = val_curve,
                                                 max_risk     = alpha_cfg,
                                                 min_coverage = min_coverage_cfg,
                                                 min_selected = min_selected_cfg)
                     
                    # ┏━━━━━━━━━━ Best Threshold (according to Policy) ━━━━━━━━━━┓ 
                    selected_tau = float(selection["threshold"])
                    
                    # ┏━━━━━━━━━━ Risk & Coverage corresponding metrics to Selected Threshold ━━━━━━━━━━┓ 
                    selection_metric = {"achieved_risk": selection["risk"],
                                        "achieved_coverage": selection["coverage"],
                                        "risk_constraint_satisfied": bool(selection["constraint_satisfied"])}
                    
                    if not selection_metric["risk_constraint_satisfied"]:
                        print(f"[WARN] Risk Budget infeasible at Alpha - {alpha_cfg:.3f}. Then, using lowest-risk fallback threshold.")

                elif policy == "f_beta":
                    # ┏━━━━━━━━━━ Optimized Threshold with its FBeta Metric ━━━━━━━━━━┓ 
                    selected_tau, best_metric = select_threshold_fbeta(vtargets,
                                                                       val_scores,
                                                                       thresholds_unique,
                                                                       fbeta_cfg)

                else:
                    raise ValueError(f"Unknown threshold policy: {policy}")

                # ┏━━━━━━━━━━ Optimized/Best Threshold for Validation Predictions [Last Fold] ━━━━━━━━━━┓ 
                eval_val = evaluate_threshold(vtargets, val_scores, selected_tau)
                val_preds_tau = eval_val.pop("predictions")

                # ┏━━━━━━━━━━ Validation Confusion Matrix & Metrics [Not Optimized Threshold] ━━━━━━━━━━┓
                plot_cm_with_metrics(vpreds_raw,
                                     vtargets,
                                     labels         = cm_labels,
                                     title          = f'M2_{task} — Val',
                                     out_dir        = checkpoint_dir,
                                     best_threshold = None,
                                     cmap           = "Oranges")

                # ┏━━━━━━━━━━ Validation Confusion Matrix & Metrics [Optimized Threshold] ━━━━━━━━━━┓
                plot_cm_with_metrics(val_preds_tau, 
                                     vtargets,
                                     labels         = cm_labels,
                                     title          = f'M2_{task} — Val',
                                     out_dir        = checkpoint_dir,
                                     best_threshold = selected_tau,
                                     cmap           = "Greens")

                # ┏━━━━━━━━━━ Plotting Risk & Coverage Curve ━━━━━━━━━━┓
                val_rc_png = checkpoint_dir / f"M2_{task}_Val_RiskCoverage.png"
                plot_coverage_risk_curve(curve = val_curve, 
                                         label = f"Gating: {gating_mode}", 
                                         save_path = str(val_rc_png), 
                                         show = False)
                
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

                payload = {"Dataset":            "Validation",
                           "Task":               task,
                           "Policy":             policy,
                           "Gating":             gating_mode,
                           "Val_AURC":           val_aurc,
                           "Val_Tau":            selected_tau,
                           "Val_Coverage@Tau":   eval_val["coverage"],
                           "Val_Risk@Tau":       eval_val["risk"],
                           "Selected_Count@Tau": eval_val["selected_count"]}
                
                # ┏━━━━━━━━━━ Adding Data to Summary of Risk & Coverage Analysis ━━━━━━━━━━┓
                if policy == "risk_budget":
                    payload["Alpha"] = alpha_cfg
                    payload.update(selection_metric)
                    payload["Min_Coverage"] = min_coverage_cfg
                    payload["Min_Selected_Count"] = min_selected_cfg
                
                # ┏━━━━━━━━━━ Adding Data to Summary of FBeta Analysis ━━━━━━━━━━┓
                elif policy == "f_beta":
                    payload["FBeta"]     = fbeta_cfg
                    payload["FBeta@Tau"] = best_metric

                # ┏━━━━━━━━━━ Original split's Model Evaluation on Test Set & Confusion Matrix ━━━━━━━━━━┓
                test_eval = epoch_loop(model,
                                       test_loader,
                                       crit,
                                       device,
                                       mode = "test",
                                       task_name = task,
                                       optimizer = None,
                                       bce_thr = 0.5,
                                       amp = True,
                                       clip_grad = 0.0,
                                       beta = trainer_cfg["fbeta"],
                                       return_raw = True)

                # ┏━━━━━━━━━━ Test Predictions & Probabilities [Not Optimized Threshold] ━━━━━━━━━━┓
                tpreds   = test_eval["preds"]
                ttargets = test_eval["targets"]
                tprobs   = test_eval.get("probs")
                if tprobs is None:
                    raise ValueError("Test probabilities were not produced; ensure the criterion supports probability extraction.")

                # ┏━━━━━━━━━━ Risk & Coverage Scores [Optimized Threshold] ━━━━━━━━━━┓
                test_scores = build_scores(tprobs, gating_mode)
                eval_test_tau = evaluate_threshold(ttargets, test_scores, selected_tau)
                test_preds_tau = eval_test_tau.pop("predictions")

                # ┏━━━━━━━━━━ Test Metrics ━━━━━━━━━━┓
                payload["Test_Coverage@Tau"]       = eval_test_tau["coverage"]
                payload["Test_Risk@Tau"]           = eval_test_tau["risk"]
                payload["Test_Selected_Count@Tau"] = eval_test_tau["selected_count"]
                
                # ┏━━━━━━━━━━ Validation Risk & Coverage Curves ━━━━━━━━━━┓
                payload["Val_Curves"] = curve_rows

                # ┏━━━━━━━━━━ Path & Save Summary ━━━━━━━━━━┓
                payload_json = checkpoint_dir / f"M2_{task}_R&C_Analysis.json"
                save_metrics(payload, str(payload_json))

                # ┏━━━━━━━━━━ Test Confusion Matrix & Metrics [Not Optimized Threshold] ━━━━━━━━━━┓
                plot_cm_with_metrics(tpreds,
                                     ttargets,
                                     labels         = cm_labels,
                                     title          = f'M2_{task} — Test',
                                     out_dir        = checkpoint_dir,
                                     best_threshold = None,
                                     cmap           = "Blues")
                
                # ┏━━━━━━━━━━ Test Confusion Matrix & Metrics [Optimized Threshold] ━━━━━━━━━━┓
                plot_cm_with_metrics(test_preds_tau,
                                     ttargets,
                                     labels         = cm_labels,
                                     title          = f'M2_{task} — Test',
                                     out_dir        = checkpoint_dir,
                                     best_threshold = selected_tau,
                                     cmap           = "Purples")

                # ┏━━━━━━━━━━ Export into CSV Test Predictions ━━━━━━━━━━┓
                print("\nEnriched Results: ")
                export_predictions(df_asset        = df_asset,
                                   dataset_tensor  = dataset_tensor,
                                   tpreds          = tpreds,
                                   tpreds_tau      = test_preds_tau,
                                   cfg             = cfg,
                                   checkpoint_dir  = checkpoint_dir,
                                   tprobs          = tprobs,
                                   meta_label_mode = meta_label_usual)

                # ┏━━━━━━━━━━ Meta-Labeling Consensus ━━━━━━━━━━┓
                plot_meta_labeling_consensus(cfg = cfg,
                                             checkpoint_dir = checkpoint_dir,
                                             best_threshold = selected_tau)

            # ┏━━━━━━━━━━ Empty Caché ━━━━━━━━━━┓
            torch.cuda.empty_cache()
                        

def main():
    parser = argparse.ArgumentParser(description = "Train CTTS model")
    parser.add_argument("--config",
                        type    = str,
                        default = "config_10.yaml",
                        help    = "Path to the YAML configuration file")
    args = parser.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = Path(__file__).parent / cfg_path

    if not cfg_path.exists():
        raise FileNotFoundError(f"Could not find config file at {cfg_path}")

    with cfg_path.open("r") as fh:
        cfg = yaml.safe_load(fh)

    run_training(cfg)


if __name__ == "__main__":
    main()
