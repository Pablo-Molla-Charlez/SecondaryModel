import os
import datetime
import yaml
import torch
import copy
import pandas

from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from paths import dataset_path

from data_preprocessing import merge_meta_targets, build_loaders, prepare_dataset
from model import CTTSModel, EarlyStopping

from train_utils import epoch_loop, seed_everything
from test_utils import get_preds_and_targets, plot_cm_with_metrics
from optim_utils import get_optimizer,  make_scheduler, step_scheduler


def main():
    # ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
    # ┃ 1) CONFIG & ASSET SELECTION                                           ┃
    # ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
    base = Path(__file__).parent
    cfg  = yaml.safe_load(open(base / "config.yaml", "r"))

    # Architectural and Hyper & Paramenters Config
    # ┏━━━━━━━━━━ 1.a) For the model architecture parameters i.e. CNN, Transformer, MLP ━━━━━━━━━━┓
    architecture_cfg = {"UP": cfg["model_up"], "DN": cfg["model_dn"]}

    # ┏━━━━━━━━━━ 1.b) For the training hyper-parameters, i.e. learning rate, batch size, etc. ━━━━━━━━━━┓
    train_cfg = {"UP": cfg["train_up"], "DN": cfg["train_dn"]}

    # ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
    # ┃ 2) DIRECTORY STRUCTURE: Data Source and Data Preparation              ┃
    # ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
    # ┏━━━━━━━━━━ Which data‐source & asset to train? ━━━━━━━━━━┓
    csv_path = dataset_path(cfg["dataset"]["source"],
                            cfg["dataset"]["type"].capitalize(),
                            cfg["dataset"]["symbol"],
                            "up")


    # ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
    # ┃ 3) DEVICE & SEEDS                                                     ┃
    # ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
    # ┏━━━━━━━━━━ Select GPU if available, otherwise CPU ━━━━━━━━━━┓
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on {device}")
    # For Personal Mac
    # device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

    # ┏━━━━━━━━━━ Reproducibility ━━━━━━━━━━┓
    #os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    #os.environ['PYTHONHASHSEED'] = str(1493583942)
    #init_seeds(1493583942, force_cuda_deterministic = True)
    seed_everything(1493583942)   


    # ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
    # ┃ 4) DATA PREPARATION                                                   ┃
    # ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
    df_asset = merge_meta_targets(asset_type      = cfg["dataset"]["type"],
                                  asset           = cfg["dataset"]["symbol"],
                                  data_dir        = str(Path(csv_path).parent),
                                  output_dir      = str(Path(csv_path).parent),
                                  column_features = cfg['column_features']
                                )

    
    # ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
    # ┃ 5) DATALOADERS & CRITERION                                            ┃
    # ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
    for task in [(cfg["training_mode"]["optuna_task"])]: # Replace by [("UP", "DN")] for both
        print(f"\n────────── {task} model ──────────")
        model_cfg = architecture_cfg[task]
        trainer_cfg = train_cfg[task]

        # ┏━━━━━━━━━━ 5.a) Preparation of Dataloaders and Training/Validation/Testing Splits with corresponding Criterion ━━━━━━━━━━┓
        # ┏━━━━━━━━━━ Preparation of Dataloaders ━━━━━━━━━━┓
        dataset_tensor = prepare_dataset(df_asset, 
                                  seq_len          = cfg["sequence_length"],
                                  column_features  = cfg["column_features"],
                                  context_features = cfg["context_features"])
        
        # ┏━━━━━━━━━━ 5.b) Splits and Criterion ━━━━━━━━━━┓
        #init_seeds(1493583942, force_cuda_deterministic = True)
        seed_everything(1493583942)  # Re-seed for reproducibility
        train_folds, test_loader = build_loaders(ds               = dataset_tensor,
                                                 cross_validation = False,
                                                 target           = task,
                                                 props            = cfg["training_mode"]["cross_val_props"],  # list only used if Cross-Validation
                                                 train_frac       = cfg["splits"]["train"],
                                                 val_frac         = cfg["splits"]["val"],
                                                 test_frac        = cfg["splits"]["test"],
                                                 batch_size       = trainer_cfg["batch_size"],
                                                 loss_type        = cfg["training_mode"]["loss_function"],
                                                 focal_gamma      = cfg["training_mode"]["focal_gamma"],
                                                 focal_alpha      = cfg["training_mode"]["focal_alpha"],
                                                 device           = device
                                   )
        
        # ┏━━━━━━━━━━ 5.c) Get the model parameters ━━━━━━━━━━┓
        model_kwargs = dict(
            # ┏━━━━━━━━━━ CNN Parameters ━━━━━━━━━━┓
            cnn_embed_dim = model_cfg["cnn_embed_dim"],
            cnn_kernel    = model_cfg["cnn_kernel"],
            cnn_stride    = model_cfg["cnn_stride"],
            p_pos_drop    = model_cfg["p_pos_drop"],
            nb_features   = len(cfg['column_features']),
            
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
            context_len   = cfg["sequence_length"]
        )
        
        # ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
        # ┃ 6) LOOP OVER FOLDS: MODEL, OPTIMIZER, SCHEDULER & EARLY STOPPING      ┃
        # ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
        n_folds        = len(train_folds)
        test_precision = None
        fold_losses    = []
        fold_accs      = []
        fold_precs     = []
        fold_f1s       = []
        fold_recs      = []
        fold_fbetas    = []

        # ┏━━━━━━━━━━ Store validation losses ━━━━━━━━━━┓
        val_losses_all_folds = []
        for fold_idx, (train_loader, val_loader, crit) in enumerate(train_folds):
            print(f"\n🌀 Fold {fold_idx + 1} / {len(train_folds)}")

            # ┏━━━━━━━━━━ 6.a) Instantiate the model (fresh for each fold) ━━━━━━━━━━┓
            #init_seeds(1493583942, force_cuda_deterministic = True)
            seed_everything(1493583942)
            model = CTTSModel(**model_kwargs).to(device)

            # ┏━━━━━━━━━━ 6.b) Optimizer ━━━━━━━━━━┓
            optim_cfg = trainer_cfg
            optimizer = get_optimizer(model.parameters(), optim_cfg)

            # ┏━━━━━━━━━━ 6.c) Scheduler (optional - remove it when scheduler chosen) ━━━━━━━━━━┓
            sch_cfg = trainer_cfg['scheduler']
            scheduler = make_scheduler(optimizer, sch_cfg, trainer_cfg["max_epochs"])

            # ┏━━━━━━━━━━ 6.d) Checkpoints Early Stoppings ━━━━━━━━━━┓
            checkpoint_dir  = (Path(cfg["paths"]["output_root"]) / "Usual" / cfg["dataset"]["symbol"]) / "Run"
            
            # ┏━━━━━━━━━━ 6.e) Create all directories if they do not exist ━━━━━━━━━━┓
            for path in [checkpoint_dir]:
                path.mkdir(parents=True, exist_ok=True)
            
            # ┏━━━━━━━━━━ 6.f) Early Stoppings ━━━━━━━━━━┓
            checkpoint_path = checkpoint_dir / f"{task}_best.pt"
            stopper = EarlyStopping(patience   = trainer_cfg["patience"],
                                    verbose    = False,
                                    delta      = 1e-4,
                                    path       = str(checkpoint_path),
                                    just_count = True
                        )
            
            # ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
            # ┃ 7) TENSORBOARD WRITERS                                                ┃
            # ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
            # ┏━━━━━━━━━━ TensorBoard Terminal Command  ━━━━━━━━━━┓
            # ┏━━━━━━━━━━ 1) ls ~/miniconda3/envs/CTTS/bin/tensorboard  ━━━━━━━━━━┓
            # ┏━━━━━━━━━━ 2) ~/miniconda3/envs/CTTS/bin/tensorboard --logdir Output/runs  ━━━━━━━━━━┓
            run_stamp = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
            tb_dir_fold = Path(cfg["paths"]["output_root"]) / "Usual" / "Tensorboard" / cfg["dataset"]["symbol"] / f"{task}_{run_stamp}"
            writer = SummaryWriter(tb_dir_fold)
            
            # ┏━━━━━━━━━━ Best Metrics (temporary) ━━━━━━━━━━┓
            #best_state = copy.deepcopy(model.state_dict()) 
            best_loss  = float('inf')
            best_acc   = 0.0
            best_prec  = 0.0
            best_rec   = 0.0
            best_f1    = 0.0
            best_fbeta = 0.0
            best_state = None        


            # ┏━━━━━━━━━━ Train & Validation ━━━━━━━━━━┓
            for epoch in range(trainer_cfg["max_epochs"]):
                # ┏━━━━━━━━━━ Train ━━━━━━━━━━┓
                train_loss, _, _, _, _, _ = epoch_loop(model, 
                                                    train_loader, 
                                                    crit, 
                                                    device, 
                                                    optimizer, 
                                                    task_name    = cfg["training_mode"]["optuna_task"],
                                                    bce_thr      = 0.5,
                                                    amp          = True,
                                                    clip_grad    = 1.0,
                                                    beta         = trainer_cfg["fbeta"])
                
                # ┏━━━━━━━━━━ Validation ━━━━━━━━━━┓
                val_loss, vacc, vprec, vrec, vf1, vfbeta = epoch_loop(model, 
                                                                    val_loader,   
                                                                    crit, 
                                                                    device, 
                                                                    optimizer = None,
                                                                    task_name = cfg["training_mode"]["optuna_task"],
                                                                    bce_thr   = 0.5, 
                                                                    amp       = True, 
                                                                    clip_grad = 0.0,
                                                                    beta      = trainer_cfg["fbeta"])
                
                            
                # ┏━━━━━━━━━━ Log to TensorBoard if last fold ━━━━━━━━━━┓
                if writer is not None:
                    writer.add_scalar("Loss/train",      train_loss, epoch)
                    writer.add_scalar("Loss/validation", val_loss  , epoch)

                # ┏━━━━━━━━━━ Early Stopping ━━━━━━━━━━┓
                stopper(val_loss, model)
                step_scheduler(scheduler, epoch, val_loss)
                if stopper.early_stop:
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
                # if best_state is None:
                #     print("[WARN] No valid checkpoint found – using last‑epoch weights")
                # else:
                #     model.load_state_dict(best_state)

                
                
                # ┏━━━━━━━━━━ Reload the best state into model ━━━━━━━━━━┓
                model.load_state_dict(best_state)

                # ┏━━━━━━━━━━ Save that best state‐dict to disk ━━━━━━━━━━┓
                ckpt_path = checkpoint_dir / f'{cfg["training_mode"]["optuna_task"]}_best.pt'
                torch.save(best_state, ckpt_path)

                # ┏━━━━━━━━━━ Original split's Model Evaluation on Validation Set & Confusion Matrix ━━━━━━━━━━┓
                vpreds, vtargets = get_preds_and_targets(model, val_loader, device, cfg["training_mode"]["optuna_task"], cfg["training_mode"]["loss_function"])
                plot_cm_with_metrics(vpreds, 
                                    vtargets,
                                    labels  = (f'No_TP_{cfg["training_mode"]["optuna_task"]}', f'TP_{cfg["training_mode"]["optuna_task"]}'),
                                    title   = f'{cfg["training_mode"]["optuna_task"]} — Validation',
                                    out_dir = checkpoint_dir,
                                    cmap    = "Oranges"
                                )
                
                # ┏━━━━━━━━━━ Original split's Model Evaluation on Test Set & Confusion Matrix ━━━━━━━━━━┓
                with torch.no_grad():
                    t_loss, t_acc, t_prec, t_rec, t_f1, t_fbeta = epoch_loop(model, 
                                                                    test_loader, 
                                                                    crit, 
                                                                    device,
                                                                    optimizer = None, 
                                                                    task_name = cfg["training_mode"]["optuna_task"], 
                                                                    bce_thr   = 0.5, 
                                                                    amp       = True, 
                                                                    clip_grad = 0.0,
                                                                    beta      = trainer_cfg["fbeta"])
                test_accuracy  = t_acc
                test_precision = t_prec
                test_recall    = t_rec
                test_f1        = t_f1
                test_fbeta     = t_fbeta

                # ┏━━━━━━━━━━ Test Confusion Matrix  ━━━━━━━━━━┓
                tpreds, ttargets = get_preds_and_targets(model, test_loader, device, cfg["training_mode"]["optuna_task"], cfg["training_mode"]["loss_function"])
                plot_cm_with_metrics(tpreds, 
                                    ttargets,
                                    labels  = (f'No_TP_{cfg["training_mode"]["optuna_task"]}', f'TP_{cfg["training_mode"]["optuna_task"]}'),
                                    title   = f'{cfg["training_mode"]["optuna_task"]} — Test M1+M2',
                                    out_dir = checkpoint_dir,
                                    cmap    = "Blues"
                                )

                # ┏━━━━━━━━━━ Export test timeline dates + CTTS predictions ━━━━━━━━━━┓
                try:
                    # Recompute split sizes to recover test indices
                    N_windows   = len(dataset_tensor)
                    n_train_win = int(cfg["splits"]["train"] * N_windows)
                    n_val_win   = int(cfg["splits"]["val"]   * N_windows)
                    # idx_test are dataset indices [n_train+n_val, ..., N_windows-1]
                    idx_test_ds = list(range(n_train_win + n_val_win, N_windows))

                    # Map each dataset index k to its corresponding end-of-window date: df.index[k + seq_len - 1]
                    seq_len = cfg["sequence_length"]
                    # df_asset index was set to 'date' in merge_meta_targets
                    all_dates = df_asset.index
                    mapped_dates = [all_dates[k + seq_len - 1] for k in idx_test_ds]

                    # Ensure date format as YYYY-MM-DD
                    date_strs = [d.strftime("%Y-%m-%d") for d in mapped_dates]

                    # Sanity: align lengths
                    if len(date_strs) != len(tpreds):
                        print(f"[WARN] Date/predictions length mismatch: dates={len(date_strs)} preds={len(tpreds)}")

                    # Build output DataFrame and write CSV
                    col_name = f"M2_Pred_{cfg['training_mode']['optuna_task']}"
                    out_df = pd.DataFrame({
                        "date": date_strs[:len(tpreds)],
                        col_name: list(map(int, tpreds.tolist()))
                    }) if pd is not None else None

                    # Fallback to manual CSV write if pandas unavailable
                    out_path = checkpoint_dir / f"{cfg['dataset']['symbol']}_{cfg['training_mode']['optuna_task']}_ctts_predictions.csv"
                    if out_df is not None:
                        out_df.to_csv(out_path, index=False)
                    else:
                        with open(out_path, "w") as f:
                            f.write(f"date,{col_name}\n")
                            for d, p in zip(date_strs, tpreds.tolist()):
                                f.write(f"{d},{int(p)}\n")

                    print(f"Saved CTTS test predictions to {out_path}")
                except Exception as e:
                    print(f"[ERROR] Failed to export CTTS predictions CSV: {e}")

            
            # ┏━━━━━━━━━━ Empty Caché ━━━━━━━━━━┓
            torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
