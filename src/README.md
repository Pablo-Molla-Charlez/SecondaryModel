# Bi-FAST Framework

A PyTorch implementation of the **Bi-FAST Framework** (Bi-level Flow Adaptation with Sampling Techniques) for time-series classification. This version (`src`) features a streamlined 4-phase Hyperparameter Optimization (HPO) pipeline.

## 📂 Project Structure

| File | Description |
|------|-------------|
| `config.yaml` | Central configuration for data, model, and HPO ranges. |
| `kronos_clas.py` | **Standalone** Kronos classifier: frozen backbone + trainable head + clean trainer with TensorBoard. |
| `HPO/hpo_utils.py` | Shared utilities for Optuna studies, expanding window CV, and config overrides. |
| `HPO/global_exploration.py` | **Phase 1A**: Global search on a single Train/Val split (Pruning ENABLED). |
| `HPO/fine_exploration.py` | **Phase 1B**: Local search around best Phase 1A params using 5-Fold Expanding Window CV (Pruning DISABLED). |
| `HPO/inflow_exploration.py` | **Phase 2**: Freezes Main Model ($\theta^*$), searches IN-Flow ($\phi$) using Bi-Level Expanding Window CV. |
| `HPO/meta_exploration.py` | **Phase 3**: Freezes Main ($\theta^*$), searches Meta-Controller ($\psi$) using Bi-Level Expanding Window CV. |
| `HPO/kronos_exploration.py` | **Standalone** Kronos HPO: searches head architecture, LR, loss type with per-trial TensorBoard. |
| `s2_model.py` | Core M2 architecture: SecondaryModel, RoPE, Patching, Fusion Transformer. |
| `Bi_FAST/` | Modules for IN-Flow, Meta-Controller, and Bi-Level Trainer. |

---

## 🚀 HPO Pipeline Overview

Run the phases in order. Each phase loads the best parameters from the previous one.

| Phase | Script | Goal | Parameters Searched | Data Split | Modules Optimized | Optimization |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **1.A Global** | `HPO/global_exploration.py` | Find rough architecture baselines of Main Model ($\theta$) | `d_model`, `n_heads`, `n_layers`, `lr`, `gamma`, `dropout` | **Train / Val**<br>(Single Split: 60/20) | **Main Model Only** ($\theta$) | **TPE + MedianPruner**<br>on Main Model ($\theta$) |
| **1.B Fine** | `HPO/fine_exploration.py` | Refine parameters carefully of Main Model ($\theta$) | Narrowed ranges around Phase 1.A best. | **Expanding Rolling Window CV: 5 Folds of Train / Val** | **Main Model Only** ($\theta$) | **TPE (No Pruning + Full Training on Folds)**<br>on Main Model ($\theta$) |
| **2.A IN-Flow (Global)** | `HPO/inflow_exploration.py --phase 2a` | Rough search for IN-Flow ($\phi$) | `flow_depth`, `hidden_dim`, `meta_lr` | **Expanding Rolling Window CV: 3 Folds of Train / Val** | **IN-Flow** ($\phi$) + **Fixed Main Model** ($\theta^*$) | **Coordinate Descent**<br>TPE+MedianPruner on Fixed Main Model Best Config ($\theta^*$) + IN-Flow ($\phi$) (Pruning ENABLED) |
| **2.B IN-Flow (Fine)** | `HPO/inflow_exploration.py --phase 2b` | Refine IN-Flow ($\phi$) carefully | Narrowed ranges around Phase 2.A best. | **Expanding Rolling Window CV: 5 Folds of Train / Val** | **IN-Flow** ($\phi$) + **Fixed Main Model** ($\theta^*$) | **Coordinate Descent**<br>TPE (No Pruning) on Fixed Main Model Best Config ($\theta^*$) + IN-Flow ($\phi$) |
| **3.A Meta (Global)** | `HPO/meta_exploration.py --phase 3a` | Rough search for Meta-Controller ($\psi$) | `lambda_barrier`, `meta_lr`, `hidden_dim`, `layers` | **Bi-Level Expanding Rolling Window CV: 3 Folds of Train / Meta / Val** | **Meta** ($\psi$) + **Main** ($\theta^*$) + **IN-Flow** ($\phi$) | **Bi-Level with TPE+MedianPruner**<br> on Fixed Main Model ($\theta^*$). IN-Flow ($\phi$) can be frozen or trained with Main Model ($\theta^*$). |
| **3.B Meta (Fine)** | `HPO/meta_exploration.py --phase 3b` | Refine Meta-Controller ($\psi$) carefully | Narrowed ranges around Phase 3.A best. | **Bi-Level Expanding Rolling Window CV: 5 Folds of Train / Meta / Val** | **Meta** ($\psi$) + **Main** ($\theta^*$) + **IN-Flow** ($\phi$) | **Bi-Level with TPE**<br>on Fixed Main Model ($\theta^*$). IN-Flow ($\phi$) can be frozen or trained altogether with Meta-Controller. |

### ❄️ Freezing & Symbiosis (Phase 2 & 3)

#### Phase 2: IN-Flow Exploration 🌊
By default, the Main Model is **Frozen** ❄️ ($\theta^*$ fixed) to provide a stable target 🎯 for the IN-Flow module to adapt to.
- **Frozen (Default)**: Best for initial IN-Flow search. Gradients still flow through the frozen model to guide the IN-Flow network.
- **Unfrozen (`--unfreeze-main`)**: Useful if you suspect the Main Model needs to adapt its weights to the new IN-Flow distribution immediately.

#### Phase 3: Meta-Controller Exploration 🧠
By default, the IN-Flow module is **Unfrozen** ($\phi$ learnable) to allow fine-tuning alongside the Meta-Controller.
- **Unfrozen (Default)**: Best for final performance. The IN-Flow adapts to the weighting policy.
- **Frozen (`--freeze-inflow`)**: Useful if you want to strictly isolate the Meta-Controller's effect.

---

## 🏛️ Kronos Classifier (Standalone)

The Kronos Foundation Model integration is now **fully separated** from the Bi-FAST pipeline into its own standalone module (`kronos_clas.py`). This allows independent development, debugging, and HPO of the classification head without any BiFAST machinery.

### Architecture
- **Frozen Backbone** ❄️: Pre-trained Kronos from HuggingFace (`NeoQuasar/Kronos-base`). Extracts context embeddings `(B, T, D)` from tokenized time-series.
- **Trainable Head** 🔥: Lightweight MLP (`ClassificationHead`): `[Linear → GELU → LayerNorm → Dropout] × N → Linear`. Optimized independently via `kronos_exploration.py`.
- **Clean Trainer**: `KronosTrainer` with AdamW, cosine annealing, early stopping, and full TensorBoard logging (all train/val metrics per epoch).

### HPO (`HPO/kronos_exploration.py`)
- **Search Space**: `head_hidden_dim`, `head_layers`, `head_dropout`, `lr`, `loss_type` (CE vs Focal), `gamma`.
- **Per-trial TensorBoard**: Each trial logs to its own TensorBoard directory.
- **Dataset Caching**: Tokenized dataset is cached to disk for fast re-use across trials.

### Usage
```bash
# Standalone training
python kronos_clas.py --config config.yaml

# HPO search over classification head
python HPO/kronos_exploration.py --n-trials 100 --max-epochs 50 --patience 15
```

## 🛠️ Usage

### 1. Global Exploration
```bash
python HPO/global_exploration.py --n-trials 400 --study-name Phase1A_Global_Exploration --db Output/HPO/Phase1A_Global_Exploration.db
```
### 2. Fine Exploration (Baseline)
```bash
python HPO/fine_exploration.py --n-trials 200 --n-folds 3 --study-name Phase1B_Fine_Exploration --db Output/HPO/Phase1B_Fine_Exploration.db \
    --best-from Output/HPO/Phase1A_Global_Exploration_best_params.json
```

### 3. IN-Flow Exploration
```bash
# Phase 2.A Global
python HPO/inflow_exploration.py --phase 2a --n-trials 200 --n-folds 3 --study-name Phase2A_Inflow_Exploration --db Output/HPO/Phase2A_Inflow_Exploration.db \
    --best-from Output/HPO/Phase1B_Fine_Exploration_best_params.json

# Phase 2.B Fine
python HPO/inflow_exploration.py --phase 2b --n-trials 100 --n-folds 5 --study-name Phase2B_Inflow_Exploration --db Output/HPO/Phase2B_Inflow_Exploration.db \
    --best-from Output/HPO/Phase2A_Inflow_Exploration_best_params.json
```

### 4. Meta-Controller Exploration
```bash
# Phase 3.A Global (Symbiotic)
python HPO/meta_exploration.py --phase 3a --n-trials 200 --n-folds 3 --study-name Phase3A_Meta_Controller --db Output/HPO/Phase3A_Meta_Controller.db \
    --best-from Output/HPO/Phase2B_Inflow_Exploration_best_params.json \
    --checkpoint Output/HPO/checkpoints/Phase2B_Inflow_Exploration_best_params.pt

# Phase 3.B Fine (Symbiotic)
python HPO/meta_exploration.py --phase 3b --n-trials 100 --n-folds 5 --study-name Phase3B_Meta_Controller --db Output/HPO/Phase3B_Meta_Controller.db \
    --best-from Output/HPO/Phase3A_Meta_Controller_best_params.json \
    --checkpoint Output/HPO/checkpoints/Phase2B_Inflow_Exploration_best_params.pt
```

## ⚙️ Configuration (`config.yaml`)

### **1. Paths (`paths`)**
Defines where data is stored and where results go.
- `csv_dir`: Directory containing the processed CSV files (e.g., `1h_og`).
- `output_root`: Root directory for all HPO databases, checkpoints, and logs.

### **2. Data Loading (`data.load`)**
Controls how the raw CSVs are ingested.
- `symbol`: List of assets to load (e.g., `["BTC", "ETH"]`).
- `start_date` / `end_date`: Filters the dataset to a specific range.
- `target_col`: The column to predict (e.g., `meta_label`).
- `granularity`: Timeframe of the data (e.g., `1h`).

### **3. Data Splits (`data.split`)**
Defines how the dataset is partitioned for **Global Exploration** and determining sizes for **Expanding Window CV**.
- `train`: **0.60** (60%) - Used for training models in all phases.
- `meta`: **0.10** (10%) - Used ONLY in Phase 3 for the **Meta-Controller's held-out set**. It acts as a proxy for test performance to optimize the weighting policy ($\psi$).
- `val`: **0.20** (20%) - Used for validation (early stopping, hyperparameter scoring) in all phases.
- `context_length`: **48** - The number of past timesteps ($T$) fed into the model (e.g., 48 hours).

### **4. Features (`data.features`)**
Specifies input channels.
- `input`: OHLCV columns (Open, High, Low, Close, Volume).
- `extrinsic`: Indicators like RSI, MACD, Bollinger Bands that bypass the main transformer branch (fed via Cross-Attention).

### **5. Main Model (`main_model`)**
Base architecture parameters (refined in Phase 1).
- `d_model`, `n_heads`, `n_layers`, `dropout`, `ffn_mult`: Transformer dimensions.
- `patch_size`: **24** - Number of timesteps per patch. With `context_length=48`, this creates $48/24 = 2$ patch tokens.

### **6. Kronos Model (`kronos_model`)**
Settings for the standalone Kronos classifier (`kronos_clas.py`). These are used by `kronos_clas.py` and `HPO/kronos_exploration.py` — **not** by the BiFAST pipeline.
- `enabled`: Legacy flag (ignored by BiFAST pipeline after separation).
- `model_path`: HuggingFace ID (e.g., `NeoQuasar/Kronos-base`).
- `tokenizer_path`: HuggingFace ID for the tokenizer.
- `head_hidden_dim`: Hidden layer size for the classification head.
- `head_dropout`: Dropout rate for the head.
- `head_layers`: Number of MLP layers.

### **7. IN-Flow Module (`in_flow`)**
Configuration for the Normalizing Flow (Phase 2).
- `type`: `revin` (Phase 1 Baseline) or `in_flow` (Phase 2+).
- `flow_depth`: Number of affine coupling layers.
- `hidden_dim`: Hidden dimension of the flow network.

### **7. Meta-Controller (`meta_controller`)**
Configuration for the Bi-Level Optimization policy (Phase 3).
- `enabled`: `true`/`false`.
- `lambda_barrier`: Coefficient for the barrier penalty (prevents weights from collapsing to 0).
- `hidden_dim`, `temperature`: Architecture of the meta-network.

### **8. Training (`training`)**
Hyperparameters for the optimization loop.
- `batch_size`: Number of samples per batch (e.g., 64).
- `learning_rate`: Baseline LR for the Main Model.
- `meta_lr`: Learning rate for Meta-Controller and IN-Flow (outer loop).
- `max_epochs`: Maximum training duration.
- `patience`: Early stopping counter.
- `loss.type`: `focal_loss` (handles class imbalance) or `cross_entropy`.

## ⚡ Performance Optimization

### Dataset Caching 💾
To speed up HPO, the pipeline automatically caches the **tokenized** dataset.
- **Cache Location**: `Output/HPO/cache/dataset_{hash}.pt`
- **Mechanism**: A unique SHA256 hash is generated from your `config.yaml` (data paths, splits, task type).
    - **First Run**: Computes tokens (slow ~minutes) -> Saves to Cache.
    - **Subsequent Runs**: Loads from Cache (fast ~seconds).
- **Invalidation**: Changing critical config parameters (e.g., `start_date`, `context_length`) automatically generates a new hash and triggers a fresh computation.

---


