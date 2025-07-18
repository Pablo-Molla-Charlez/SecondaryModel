import os
import datetime
import yaml
import torch

from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from paths import dataset_path

from data_preprocessing import merge_meta_targets, build_loaders, prepare_dataset
from model import CTTSModel, EarlyStopping

from train_utils import init_seeds, model_train, model_test
from test_utils import get_preds_and_targets, plot_cm_with_metrics
from optim_utils import get_optimizer,  make_scheduler


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

    # ┏━━━━━━━━━━ 1.c) Training with Cross-Validation or without ━━━━━━━━━━┓
    cross_validation = cfg["training_mode"]["cross_validation"]
    if cross_validation:
        cross_val_props = cfg["training_mode"]["cross_val_props"]



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
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    init_seeds(42, force_cuda_deterministic = False)


    # ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
    # ┃ 4) DATA PREPARATION                                                   ┃
    # ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
    df_asset = merge_meta_targets(asset_type = cfg["dataset"]["type"],
                                  asset       = cfg["dataset"]["symbol"],
                                  data_dir    = str(Path(csv_path).parent),
                                  output_dir  = str(Path(csv_path).parent)
                                )


    # ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
    # ┃ 5) DATALOADERS & CRITERION                                            ┃
    # ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
    for task in [("UP")]: # Replace by [("UP", "DN")] for both
        print(f"\n────────── {task} model ──────────")
        model_cfg = architecture_cfg[task]
        trainer_cfg = train_cfg[task]

        # ┏━━━━━━━━━━ 5.a) Preparation of Dataloaders and Training/Validation/Testing Splits with corresponding Criterion ━━━━━━━━━━┓
        # Preparation of Dataloaders
        dataset_tensor = prepare_dataset(df_asset, seq_len=cfg["sequence_length"])
        
        # ┏━━━━━━━━━━ 5.b) Splits and Criterion ━━━━━━━━━━┓
        train_folds, test_loader = build_loaders(ds               = dataset_tensor,
                                                 cross_validation = cfg["training_mode"]["cross_validation"],
                                                 target           = task,
                                                 props            = cfg["training_mode"]["cross_val_props"],  # list only used if Cross-Validation
                                                 train_frac       = cfg["splits"]["train"],
                                                 val_frac         = cfg["splits"]["val"],
                                                 test_frac        = cfg["splits"]["test"],
                                                 batch_size       = trainer_cfg["batch_size"],
                                                 loss_type        = cfg["training_mode"]["loss_function"],
                                                 device           = device
                                   )   

        
        # ┏━━━━━━━━━━ 5.c) Get the model parameters ━━━━━━━━━━┓
        model_kwargs = dict(
            # ┏━━━━━━━━━━ CNN Parameters ━━━━━━━━━━┓
            cnn_embed_dim = model_cfg["cnn_embed_dim"],
            cnn_kernel    = model_cfg["cnn_kernel"],
            cnn_stride    = model_cfg["cnn_stride"],
            p_pos_drop    = model_cfg["p_pos_drop"],
            
            # ┏━━━━━━━━━━ Transformer Parameters ━━━━━━━━━━┓
            trans_heads   = model_cfg["transformer"]["heads"],
            trans_ff      = model_cfg["transformer"]["ffn_dim"],
            trans_layers  = model_cfg["transformer"]["layers"],
            trans_dropout = model_cfg["transformer"]["dropout"],
            trans_activ   = model_cfg["transformer"]["activation"],
            
            # ┏━━━━━━━━━━ Classifier Parameters ━━━━━━━━━━┓
            mlp_hidden    = model_cfg["classifier"]["mlp_hidden"],
            mlp_dropout   = model_cfg["classifier"]["mlp_dropout"],
            mlp_activ     = model_cfg["classifier"]["mlp_activation"],
            mlp_pooling   = model_cfg["classifier"]["mlp_pooling"],
            

            # ┏━━━━━━━━━━ Training Mode Parameters ━━━━━━━━━━┓
            num_classes   = cfg["training_mode"]["num_classes"],
            padding       = cfg["training_mode"]["padding"],
            context_len   = cfg["sequence_length"]
        )
        
        # ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
        # ┃ 6) LOOP OVER FOLDS: MODEL, OPTIMIZER, SCHEDULER & EARLY STOPPING      ┃
        # ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
        
        # Store validation losses
        val_losses_all_folds = []
        for fold_idx, (train_loader, val_loader, crit) in enumerate(train_folds):

            print(f"\n🌀 Fold {fold_idx + 1} / {len(train_folds)}")

            # ┏━━━━━━━━━━ 6.a) Instantiate the model (fresh for each fold) ━━━━━━━━━━┓
            model = CTTSModel(**model_kwargs).to(device)

            # ┏━━━━━━━━━━ 6.b) Optimizer ━━━━━━━━━━┓
            optim_cfg = trainer_cfg
            optimizer = get_optimizer(model.parameters(), optim_cfg)

            # ┏━━━━━━━━━━ 6.c) Scheduler (optional - remove it when scheduler chosen) ━━━━━━━━━━┓
            sch_cfg = trainer_cfg['scheduler']
            scheduler = make_scheduler(optimizer, sch_cfg, trainer_cfg["max_epochs"])

            # ┏━━━━━━━━━━ 6.d) Checkpoints Early Stoppings ━━━━━━━━━━┓
            fold_suffix = f"fold_{fold_idx + 1}" if cross_validation else "no_cv"
            checkpoint_dir  = (Path(cfg["paths"]["output_root"]) / "checkpoints" / cfg["dataset"]["symbol"])
            checkpoint_CV_dir = checkpoint_dir / "CV"
            checkpoint_No_CV_dir = checkpoint_dir / "No_CV"
            # Create all directories if they do not exist
            for path in [checkpoint_dir, checkpoint_CV_dir, checkpoint_No_CV_dir]:
                path.mkdir(parents=True, exist_ok=True)
            
            # ┏━━━━━━━━━━ 6.e) Early Stoppings ━━━━━━━━━━┓
            checkpoint_path = checkpoint_CV_dir / f"{task}_{fold_suffix}_best.pt" if cross_validation else checkpoint_No_CV_dir / f"{task}_best.pt"
            stopper = EarlyStopping(patience   = trainer_cfg["patience"],
                                    verbose    = True,
                                    delta      = 1e-4,
                                    path       = str(checkpoint_path),
                                    just_count = False
                        )
            
            # ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
            # ┃ 7) TENSORBOARD WRITERS                                                ┃
            # ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
            # TensorBoard Terminal Command: 1) ls ~/miniconda3/envs/CTTS/bin/tensorboard 2) ~/miniconda3/envs/CTTS/bin/tensorboard --logdir Output/runs
            run_stamp = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
            tb_dir_fold = Path(cfg["paths"]["output_root"]) / "runs" / cfg["dataset"]["symbol"] / "CV" / f"{task}_{fold_suffix}_{run_stamp}" if cross_validation else Path(cfg["paths"]["output_root"]) / "runs" / cfg["dataset"]["symbol"] / "No_CV" / f"{task}_{run_stamp}"
            writer = SummaryWriter(tb_dir_fold)
            
            
            # ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
            # ┃ 8) MODEL TRAINING                                                     ┃
            # ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
            # ┏━━━━━━━━━━ 8.a) Training ━━━━━━━━━━┓
            val_loss = model_train(model, 
                                   optimizer, 
                                   crit, 
                                   stopper,
                                   train_loader, 
                                   val_loader, 
                                   writer, 
                                   device,
                                   MAX_EPOCHS = trainer_cfg["max_epochs"],
                                   task_name  = task,
                                   scheduler  = scheduler,
                                   bce_thr    = trainer_cfg["bce_thr"]
                       )
            
            # ┏━━━━━━━━━━ 8.b) Store Val Losses ━━━━━━━━━━┓
            val_losses_all_folds.append(val_loss)            

            # ┏━━━━━━━━━━ 8.c) Save final model info from last fold ━━━━━━━━━━┓
            if fold_idx == len(train_folds) - 1:
                final_model = model
                final_criterion = crit
                final_checkpoint = stopper.path
                final_writer = writer

                # Final fold's validation loader for Confusion Matrix and metric purposes
                final_val_loader = val_loader

        #  ┏━━━━━━━━━━ 8.d) Average Validation Losses from Cross-Validation ━━━━━━━━━━┓
        if cross_validation:
            print("🌟 Validation Results 🌟")
            for i, v in enumerate(val_losses_all_folds, 1):
                print(f"   Fold {i}: {v:.4f}")
            print(f"   📊 Average Validation Loss: {sum(val_losses_all_folds) / len(val_losses_all_folds):.4f}")

        
        # ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
        # ┃ 9) LOAD MODEL & WEIGHTS         ┃
        # ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
        # ┏━━━━━━━━━━ Reload best weights from last fold (CV) or only model (no CV) ━━━━━━━━━━┓
        if Path(final_checkpoint).exists():
            final_model.load_state_dict(torch.load(final_checkpoint, map_location=device))
        else:
            print(f"[WARN] No checkpoint found at {final_checkpoint} - testing last-epoch weights")


        # ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
        # ┃ 10) MODEL TESTING               ┃
        # ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
        # ┏━━━━━━━━━━ Run testing on final test_loader ━━━━━━━━━━┓
        model_test(final_model,
                   final_criterion,
                   test_loader,
                   final_writer,
                   device,
                   step      = trainer_cfg["max_epochs"],
                   task_name = task,
                   bce_thr   = trainer_cfg["bce_thr"]
                )

        # ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
        # ┃ 11) PLOT & SAVE ALL CONFUSION ┃
        # ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
        # ┏━━━━━━━━━━ Confusion Matrices with metrics ━━━━━━━━━━┓
        confusion_matrices_dir = checkpoint_CV_dir / "confusion_matrices" if cross_validation else checkpoint_No_CV_dir / "confusion_matrices"
        for phase, loader, color in [("Validation", final_val_loader, "Oranges"), ("Test", test_loader, "Blues")]:
            preds, targets = get_preds_and_targets(final_model, loader, device, task, loss_type=cfg["training_mode"]["loss_function"])
            plot_cm_with_metrics(
                preds, targets,
                labels  = (f"No_TP_{task}",f"TP_{task}"),
                title   = f"{task} — {phase}",
                out_dir = confusion_matrices_dir,
                cmap    = color
            )


if __name__ == "__main__":
    main()