import os
import datetime
import yaml
import torch
import json

from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from paths import dataset_path

from data_preprocessing import merge_meta_targets, build_loaders, prepare_dataset
from model import CTTSModel, EarlyStopping

from train_utils import init_seeds, model_train, model_test
from test_utils import get_preds_and_targets, plot_cm_with_metrics
from optim_utils import get_optimizer,  make_scheduler

# ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
# ┃ LOAD OPTUNA BEST TRIALS & COMMON CONFIG PARAMETERS                    ┃
# ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
def load_experiments(config_path: Path, pareto_path: Path):
    # ┏━━━━━━━━━━ 1.a) Load config.yaml and select the fixed parameters ━━━━━━━━━━┓
    cfg = yaml.safe_load(open(config_path, "r"))
    fixed = {
        # ┏━━━━━━━━━━━━━━━ General paths (Fixed Parameters) ━━━━━━━━━━━━━━━━┓
        "paths":    cfg["paths"],
        "dataset":  cfg["dataset"],
        "splits":   cfg["splits"],

        # ┏━━━━━━━━━━━━━━ Training Mode (Fixed Parameters) ━━━━━━━━━━━━━━━┓
        "task_cfg": {
            "loss_function": cfg["training_mode"]["loss_function"],
            "num_classes":   cfg["training_mode"]["num_classes"],
            "padding":       cfg["training_mode"]["padding"],
        },

        # ┏━━━━━━━━━━ Basic (Fixed Parameters) ━━━━━━━━┓
        "train_up": {
            "max_epochs": cfg["train_up"]["max_epochs"],
            "patience":   cfg["train_up"]["patience"],
            "bce_thr":    cfg["train_up"]["bce_thr"],  
        },

        # ┏━━━━━━━━━━ Sequence Length (Fixed Parameters) ━━━━━━━━┓
        "sequence_length":    cfg["sequence_length"],
    }

    # ┏━━━━━━━━━━ 1.b) Load Pareto JSON ━━━━━━━━━━┓
    pareto = json.load(open(pareto_path, "r"))
    experiments = []
    for entry in pareto:
        num = entry["number"]
        p   = entry["params"]
        betas = (p.pop("beta1"), p.pop("beta2"))

        # ┏━━━━━━━━━━ Build model_cfg for CTTSModel() Constructor (New Parameters) ━━━━━━━━━━┓
        model_cfg = {
            # ┏━━━━━━━━━━ CNN Architecture ━━━━━━━━━━┓
            "cnn_embed_dim": p["cnn_embed_dim"],
            "cnn_kernel":    p["cnn_kernel"],
            "cnn_stride":    p["cnn_stride"],
            "p_pos_drop":    p["p_pos_drop"],
            
            # ┏━━━━━━━━━━ Transformer Architecture ━━━━━━━━━━┓
            "trans_heads":   p["heads"],
            "trans_layers":  p["layers"],
            "trans_ff":      p["ffn_dim"] * p["cnn_embed_dim"],
            "trans_dropout": p["dropout"],
            "trans_activ":   p["activation"],
            
            # ┏━━━━━━━━━━ MLP Architecture ━━━━━━━━━━┓
            "mlp_hidden":    p["mlp_hidden"],
            "mlp_dropout":   p["mlp_dropout"],
            "mlp_pooling":   p["mlp_pooling"],
            "mlp_activ":     p["mlp_activation"],
            
            # ┏━━━━━━━━━━ Task settings ━━━━━━━━━━┓
            "num_classes":   fixed["task_cfg"]["num_classes"],                  # Fixed
            "padding":       fixed["task_cfg"]["padding"],                      # Fixed
            "context_len":   fixed["sequence_length"],                          # Fixed
        }

        # ┏━━━━━━━━━━ Build trainer_cfg (New Parameters) ━━━━━━━━━━┓
        trainer_cfg = {
            # ┏━━━━━━━━━━ Hyperparameters (New Parameters) ━━━━━━━━━━┓
            "batch_size":    2 ** p["batch_pow"],
            "lr":            p["lr"],
            "weight_decay":  p["weight_decay"],
            "optimizer":     p["optimizer"],
            "eps":           p["eps"],
            "betas":         betas,
            "lr_decay":      p["lr_decay"],
            
            # ┏━━━━━━━━━━ Learning Rate Scheduler (New Parameters) ━━━━━━━━━━┓
            "sch_name":        p["sch_name"],
            "warmup_epochs":   p["warmup_epochs"],
            "plateau_patience": p["plateau_patience"],
            "plateau_factor":   p["plateau_factor"],
            "power_s":          p["power_s"],
            "power_c":          p["power_c"],

            # ┏━━━━━━━━━━ Optimizer (New Parameters) ━━━━━━━━━━┓
            "alpha":            p.get("rms_alpha"),
            "momentum":         p.get("rms_momentum"),
            "weight_decouple":  p.get("weight_decouple"),
            "rectify":          p.get("rectify"),
            "lookahead_k":      p.get("lookahead_k"),
            "lookahead_alpha":  p.get("lookahead_alpha"),

            # ┏━━━━━━━━━━ Extra (Fixed Parameters) ━━━━━━━━━━┓
            **fixed["train_up"]                                                 # Fixed
        }

        # ┏━━━━━━━━━━ Trial Metrics ━━━━━━━━━━┓
        experiments.append({
            "trial_number":   num,
            "model_cfg":      model_cfg,
            "trainer_cfg":    trainer_cfg,
            "val_loss":       entry["val_loss"],
            "val_precision":  entry["val_precision"],
            "test_precision": entry.get("test_precision"),
        })

    return fixed, experiments


def main():
    # ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
    # ┃ 1) CONFIG & ASSET SELECTION                                           ┃
    # ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
    base = Path(__file__).parent
    fixed, experiments = load_experiments(base / "config.yaml", base / "optuna_pareto.json")

    # ┏━━━━━━━━━━━━━━━ Main Loop (Iteration per Trial) ━━━━━━━━━━┓
    for exp in experiments:
        num = exp["trial_number"]

        # ┏━━━━━━━━━━━━━━━ (Per-) Trial output directory ━━━━━━━━━━┓
        run_stamp = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        tb_dir = Path(fixed["paths"]["output_root"]) / "runs" / fixed["dataset"]["symbol"] / f"Trial_{num}_{run_stamp}"
        ckpt_dir = Path(fixed["paths"]["output_root"]) / "checkpoints" / fixed["dataset"]["symbol"] / f"Trial_{num}"
        tb_dir.mkdir(parents=True, exist_ok=True)
        ckpt_dir.mkdir(parents=True, exist_ok=True)


        # ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
        # ┃ 2) DIRECTORY STRUCTURE: Data Source and Data Preparation              ┃
        # ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
        # ┏━━━━━━━━━━ Which data-source & asset to train? ━━━━━━━━━━┓
        csv_path = dataset_path(fixed["dataset"]["source"],
                                fixed["dataset"]["type"].capitalize(),
                                fixed["dataset"]["symbol"],
                                "merge")
        

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
        init_seeds(42, force_cuda_deterministic = True)

        
        # ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
        # ┃ 4) DATA PREPARATION                                                   ┃
        # ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
        df_asset = merge_meta_targets(asset_type = fixed["dataset"]["type"],
                                      asset       = fixed["dataset"]["symbol"],
                                      data_dir    = str(Path(csv_path).parent),
                                      output_dir  = str(Path(csv_path).parent)
                                    )


        # ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
        # ┃ 5) DATALOADERS & CRITERION                                            ┃
        # ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
        for task in [("UP")]: # Replace by [("UP", "DN")] for both
            print(f"\n────────── {task} model ──────────")
            model_cfg   = exp["model_cfg"]
            trainer_cfg = exp["trainer_cfg"]

            # ┏━━━━━━━━━━ 5.a) Preparation of Dataloaders ━━━━━━━━━━┓
            dataset_tensor = prepare_dataset(df_asset, seq_len = fixed["sequence_length"])
            
            # ┏━━━━━━━━━━ 5.b) Training/Validation/Testing Splits and Criterion ━━━━━━━━━━┓
            train_folds, test_loader = build_loaders(ds               = dataset_tensor,
                                                     cross_validation = False,
                                                     target           = task,
                                                     props            = 0.4,                                # Only for Cross-Validation
                                                     train_frac       = fixed["splits"]["train"],
                                                     val_frac         = fixed["splits"]["val"],
                                                     test_frac        = fixed["splits"]["test"],
                                                     batch_size       = trainer_cfg["batch_size"],
                                                     loss_type        = fixed["task_cfg"]["loss_function"],
                                                     device           = device
                                                    )   

            
            # ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
            # ┃ 6) LOOP OVER FOLDS (SiNGLE FOLD): MODEL, OPTIMIZER, SCHEDULER         ┃
            # ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
            # ┏━━━━━━━━━━ Single fold because we don't use Cross-Validation ━━━━━━━━━━┓
            for fold_idx, (train_loader, val_loader, crit) in enumerate(train_folds):
                print(f"\n🌀 Fold {fold_idx + 1} / {len(train_folds)}")

                # ┏━━━━━━━━━━ 6.a) Instantiate the model (fresh for each fold) ━━━━━━━━━━┓
                model = CTTSModel(**model_cfg).to(device)
                
                # ┏━━━━━━━━━━ 6.b) Optimizer ━━━━━━━━━━┓
                optim_cfg = trainer_cfg
                optimizer = get_optimizer(model.parameters(), optim_cfg)

                # ┏━━━━━━━━━━ 6.c) Scheduler (optional - remove it when scheduler chosen) ━━━━━━━━━━┓
                sch_cfg = trainer_cfg
                scheduler = make_scheduler(optimizer, sch_cfg, sch_cfg["max_epochs"])
                
                # ┏━━━━━━━━━━ 6.d) Early Stoppings ━━━━━━━━━━┓
                stopper = EarlyStopping(patience   = trainer_cfg["patience"],
                                        verbose    = True,
                                        delta      = 1e-4,
                                        path       = str(ckpt_dir / f"{task}_best.pt"),
                                        just_count = False)
                
                # ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
                # ┃ 7) TENSORBOARD WRITERS                                                ┃
                # ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
                # TensorBoard Terminal Command: 1) ls ~/miniconda3/envs/CTTS/bin/tensorboard 2) ~/miniconda3/envs/CTTS/bin/tensorboard --logdir Output/runs
                writer = SummaryWriter(tb_dir / f"{task}")
                
                
                # ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
                # ┃ 8) MODEL TRAINING                                                     ┃
                # ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
                # ┏━━━━━━━━━━ 8.a) Training ━━━━━━━━━━┓
                _ = model_train(model, 
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
                                bce_thr    = trainer_cfg["bce_thr"])

                # ┏━━━━━━━━━━ 8.b) Save final model info from last fold ━━━━━━━━━━┓
                # Last fold trains the model with original splits: 70% (Train) + 15% (Val) + 15% (Test)
                if fold_idx == len(train_folds) - 1:
                    final_model = model
                    final_criterion = crit
                    final_checkpoint = stopper.path
                    final_writer = writer
                    final_val_loader = val_loader
            

            # ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
            # ┃ 9) LOAD MODEL & WEIGHTS         ┃
            # ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
            # ┏━━━━━━━━━━ Reload best weights from last fold ━━━━━━━━━━┓
            if Path(final_checkpoint).exists():
                final_model.load_state_dict(torch.load(final_checkpoint, map_location=device))
            else:
                print(f"[WARN] No checkpoint found at {final_checkpoint} - testing last-epoch weights")

            
            # ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
            # ┃ 10) MODEL TESTING               ┃
            # ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
            # ┏━━━━━━━━━━ Run testing on final model ━━━━━━━━━━┓
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
            confusion_matrices_dir = ckpt_dir / "confusion_matrices"
            for phase, loader, color in [("Validation", final_val_loader, "Oranges"), ("Test", test_loader, "Blues")]:
                preds, targets = get_preds_and_targets(final_model, loader, device, task, loss_type = fixed["task_cfg"]["loss_function"])
                plot_cm_with_metrics(
                    preds, targets,
                    labels  = (f"No_TP_{task}",f"TP_{task}"),
                    title   = f"{task} — {phase}",
                    out_dir = confusion_matrices_dir,
                    cmap    = color
                )

if __name__ == "__main__":
    main()