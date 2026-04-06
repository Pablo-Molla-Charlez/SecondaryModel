# Secondary-Model

<p align="center">
  <img src="https://img.shields.io/badge/Kronos-M2%20Tree%20Stack-0f766e?style=for-the-badge" alt="Kronos M2 Tree Stack" />
  <img src="https://img.shields.io/badge/Models-Utils%2Fmodels.py-1d4ed8?style=for-the-badge" alt="Models Utils models py" />
  <img src="https://img.shields.io/badge/Config-config.yaml-2563eb?style=for-the-badge" alt="Config config yaml" />
  <img src="https://img.shields.io/badge/Outputs-src%2FOutput-f59e0b?style=for-the-badge" alt="Outputs src Output" />
  <img src="https://img.shields.io/badge/OCP-SAOCP%20Diagnostics-7c3aed?style=for-the-badge" alt="OCP SAOCP Diagnostics" />
</p>

> Current `src/` workspace for the Secondary Model of the Meta-Labeling architecture, on top of financial foundation models like Kronos or Fincast.
> This README documents the code that is actually present today: the modular tree-based M2 stack around `kronos_tree.py` and `Utils/`.

<table>
  <tr>
    <td bgcolor="#ccfbf1"><strong>Main Entry</strong><br /><code>kronos_tree.py</code></td>
    <td bgcolor="#dbeafe"><strong>Model Registry</strong><br /><code>Utils/models.py</code></td>
    <td bgcolor="#dbeafe"><strong>Primary Config</strong><br /><code>config.yaml</code></td>
    <td bgcolor="#fef3c7"><strong>Outputs</strong><br /><code>src/Output/</code></td>
    <td bgcolor="#ede9fe"><strong>OCP Diagnostics</strong><br /><code>Utils/ocp_analysis.py</code></td>
  </tr>
</table>

---

## Visual Overview

<p>
  <img src="https://img.shields.io/badge/Data-Preprocessing-0f766e?style=flat-square" alt="Data Preprocessing" />
  <img src="https://img.shields.io/badge/Models-RF%20%7C%20XGBoost%20%7C%20AutoGluon-2563eb?style=flat-square" alt="Models" />
  <img src="https://img.shields.io/badge/Factory-Utils%2Fmodels.py-1d4ed8?style=flat-square" alt="Factory Utils models py" />
  <img src="https://img.shields.io/badge/Selection-Utility%20or%20SAOCP-f59e0b?style=flat-square" alt="Selection" />
  <img src="https://img.shields.io/badge/Reports-Backtests%20%7C%20Comparisons%20%7C%20OCP-7c3aed?style=flat-square" alt="Reports" />
</p>

```mermaid
%%{init: {'theme': 'base', 'themeVariables': { 'primaryColor': '#0f766e', 'primaryBorderColor': '#115e59', 'primaryTextColor': '#ffffff', 'secondaryColor': '#f59e0b', 'tertiaryColor': '#dbeafe', 'lineColor': '#0f172a', 'background': '#ffffff'}}}%%
flowchart LR
    A[CSV Market Data<br/>multi-asset / multi-granularity] --> B[Utils/data_preprocessing.py]
    B --> C[M1 / Kronos Signals<br/>labels, returns, dates, engineered features]
    C --> D[kronos_tree.py<br/>pipeline orchestration]
    D --> K[Utils/models.py<br/>model factory and AutoGluon wrapper]
    K --> E[Selective Classification<br/>utility threshold or SAOCP]
    E --> F[Feature Plots]
    E --> G[Temporal Evaluation]
    E --> H[Backtests]
    E --> I[Comparison Tables]
    E --> J[OCP Diagnostics]
    classDef input fill:#0f766e,stroke:#115e59,color:#ffffff,stroke-width:2px;
    classDef model fill:#2563eb,stroke:#1d4ed8,color:#ffffff,stroke-width:2px;
    classDef select fill:#f59e0b,stroke:#d97706,color:#111827,stroke-width:2px;
    classDef report fill:#7c3aed,stroke:#6d28d9,color:#ffffff,stroke-width:2px;
    class A,B,C input;
    class D,K model;
    class E select;
    class F,G,H,I,J report;
    linkStyle default stroke:#0f172a,stroke-width:5px;
```

```mermaid
%%{init: {'theme': 'base', 'themeVariables': { 'primaryColor': '#2563eb', 'primaryBorderColor': '#1d4ed8', 'primaryTextColor': '#ffffff', 'secondaryColor': '#f59e0b', 'tertiaryColor': '#dcfce7', 'lineColor': '#0f172a', 'background': '#ffffff'}}}%%
flowchart TD
    A[config.yaml] --> B[paths]
    A --> C[data.load]
    A --> D[data.split]
    A --> E[data.features]
    A --> G[evaluation.fee_per_trade]
    classDef root fill:#2563eb,stroke:#1d4ed8,color:#ffffff,stroke-width:2px;
    classDef group fill:#0f766e,stroke:#115e59,color:#ffffff,stroke-width:2px;
    classDef detail fill:#f59e0b,stroke:#d97706,color:#111827,stroke-width:2px;
    class A root;
    class B,C,D,E group;
    class F,G detail;
    linkStyle 0,1,2,3,4,5 stroke:#2563eb,stroke-width:6px;
```

---

## Codebase Description

<p>
  <img src="https://img.shields.io/badge/Focus-Current%20Code%20Only-0f766e?style=flat-square" alt="Focus Current Code Only" />
</p>

The active `src/` tree is centered on `kronos_tree.py`, which drives the current M2 research workflow for tree-based meta-label filtering on top of Kronos/Fincast signals.

The latest refactor also split model-definition logic into `Utils/models.py`, so model construction is no longer described as living inside `kronos_tree.py`.

It supports:

- Random Forest, XGBoost, and AutoGluon classifiers
- Centralized model construction through `Utils/models.py`
- Feature diagnostics and ranking
- Temporal validation and test evaluation
- Utility-threshold and SAOCP selection
- Backtests and equity curves
- Separate-vs-unified (per-gran vs all-grans) comparison tables
- Practical OCP diagnostics

---

## Current Project Map

<p>
  <img src="https://img.shields.io/badge/Core-kronos_tree.py-2563eb?style=flat-square" alt="Core kronos_tree py" />
  <img src="https://img.shields.io/badge/Models-Utils%2Fmodels.py-1d4ed8?style=flat-square" alt="Models Utils models py" />
  <img src="https://img.shields.io/badge/Utilities-Utils%2F-0f766e?style=flat-square" alt="Utilities Utils" />
  <img src="https://img.shields.io/badge/Data-Data_MLA-f59e0b?style=flat-square" alt="Data Data MLA" />
</p>

| Path | Role |
| --- | --- |
| `config.yaml` | Main runtime configuration for paths, dates, selected engineered features, forecast horizon, and fees. |
| `kronos_tree.py` | Main M2 analysis entrypoint and the only primary CLI in this folder; orchestrates training, evaluation, selection, backtesting, and reporting. |
| `Utils/models.py` | Central model factory for `rf`, `xgboost`, and `autogluon`, including the sklearn-compatible AutoGluon wrapper and model-info export helpers. |
| `Utils/data_preprocessing.py` | Dataset loading, multi-asset assembly, multi-granularity wrapping, chronological splitting, and feature plumbing. |
| `Utils/features.py` | Feature plots, feature ranking, confusion matrices, return histograms, and probability diagnostics. |
| `Utils/selective_classification.py` | Risk-coverage utilities, plotting, metrics export, and utility-threshold search. |
| `Utils/saocp.py` | Online Conformal Prediction (OCP) / Strongly Adaptive Online Conformal Prediction (SAOCP) logic, including delayed-feedback online helpers. |
| `Utils/backtest.py` | Backtest helpers, equity construction, Sharpe / drawdown, and reporting. |
| `Utils/comparison.py` | Separate-vs-unified and cross-paradigm comparison builders. |
| `Utils/ocp_analysis.py` | Practical OCP diagnostics for completed result folders. |
| `Utils/ocp_theory.py` | OCP theory-oriented experiments kept separate from the main analysis path. |
| `Data_MLA/` | Kronos-oriented dataset assets, technical indicator computation, and meta-label conversion utilities. |

---

## Run Guide

<p>
  <img src="https://img.shields.io/badge/Run%20Modes-Single%20%7C%20Per--Gran%20%7C%20Unified-0f766e?style=flat-square" alt="Run Modes" />
  <img src="https://img.shields.io/badge/Model%20Choice-rf%20%7C%20xgboost%20%7C%20autogluon-2563eb?style=flat-square" alt="Model Choice" />
  <img src="https://img.shields.io/badge/Threshold-utility%20or%20OCP-f59e0b?style=flat-square" alt="Threshold" />
</p>

### Working Directory

Run the commands below from:

```bash
cd /home/pablo/M2_DS/Secondary-Model/src
```

### `kronos_tree.py`: Main CLI

`kronos_tree.py` is the real entrypoint. It has four mutually exclusive modes:

| Mode | Command shape | What it does |
| --- | --- | --- |
| single-granularity | `python kronos_tree.py [flags]` | Uses a single-granularity config or cache. |
| per-granularity | `python kronos_tree.py --per-gran [flags]` | Trains one model per granularity from a multi-gran cache. |
| unified | `python kronos_tree.py --all-grans [flags]` | Trains one unified model across all granularities, then reports per granularity. |
| comparison-only | `python kronos_tree.py --comparison ...` or `--paradigm-comparison ...` | Builds comparison artifacts from already completed result folders. |

### `kronos_tree.py`: Command Cookbook

The current `config.yaml` uses `granularity: "all"`, so the normal choices for this repository right now are `--per-gran` and `--all-grans`.

| Use case | Command |
| --- | --- |
| Show the full CLI help | `python kronos_tree.py --help` |
| Single-granularity run with a single-gran config | `python kronos_tree.py --config your_single_gran_config.yaml` |
| Per-granularity run using the current config | `python kronos_tree.py --config config.yaml --per-gran` |
| Unified multi-granularity run using the current config | `python kronos_tree.py --config config.yaml --all-grans` |
| Per-granularity run with an explicit cache | `python kronos_tree.py --config config.yaml --per-gran --cache Output/Kronos/cache/your_multi_cache.pt` |
| Unified run with an explicit cache | `python kronos_tree.py --config config.yaml --all-grans --cache Output/Kronos/cache/your_multi_cache.pt` |
| Per-granularity Random Forest with utility threshold | `python kronos_tree.py --config config.yaml --per-gran --model rf --thres utility` |
| Per-granularity Random Forest with SAOCP | `python kronos_tree.py --config config.yaml --per-gran --model rf --thres OCP --ocp-alpha 0.10` |
| Unified XGBoost run | `python kronos_tree.py --config config.yaml --all-grans --model xgboost --thres utility` |
| Unified AutoGluon run | `python kronos_tree.py --config config.yaml --all-grans --model autogluon --ag-time-limit 900 --ag-presets high_quality` |
| Disable feature analysis completely | `python kronos_tree.py --config config.yaml --per-gran --features false --top5 false` |
| Build separate-vs-unified comparison tables | `python kronos_tree.py --comparison Output/Kronos/randforest Output/Kronos/randforest/unified_down_tp` |
| Build cross-paradigm comparison tables | `python kronos_tree.py --paradigm-comparison Output/Kronos/randforest Output/Kronos/xgboost Output/Kronos/autogluon` |

### `kronos_tree.py`: Flag Reference

| Flag | Values | Meaning |
| --- | --- | --- |
| `--cache` | path to `.pt` | Use an explicit dataset cache instead of resolving it only from `config.yaml`. |
| `--config` | path to YAML | Config file path. Default: `config.yaml`. |
| `--model` | `rf`, `xgboost`, `autogluon` | Selects the classifier family. |
| `--ag-time-limit` | integer seconds | AutoGluon fit time limit per training call. |
| `--ag-presets` | `best_quality`, `high_quality`, `good_quality`, `medium_quality` | AutoGluon preset bundle. |
| `--per-gran` | flag | Train one model per granularity. |
| `--all-grans` | flag | Train one model on all granularities together. |
| `--comparison` | `PER_GRAN_DIR UNIFIED_DIR` | Build comparison outputs from two finished result directories. |
| `--paradigm-comparison` | `DIR DIR ...` | Compare two or more completed paradigms side by side. |
| `--thres` | `utility`, `OCP` | Use validation-set utility thresholding or SAOCP. |
| `--ocp-alpha` | float | Target miscoverage for OCP. `0.10` means a nominal 90% coverage target. |
| `--top5` | `true`, `false` | Whether to run top-5 feature analysis and top-5 backtests. |
| `--features` | `true`, `false` | Whether to run feature analysis at all. |

Important constraint:

- `--top5 true` requires `--features true`
- The actual model objects used by `--model` are now built in `Utils/models.py`

### `features.py`: No Standalone CLI

`Utils/features.py` is a support module, not a script with its own CLI. In normal usage it is reached indirectly through `kronos_tree.py` when feature analysis is enabled.

### `comparison.py`: No Standalone CLI

`Utils/comparison.py` is also a library module. The usual way to use it is through:

- `python kronos_tree.py --comparison ...`
- `python kronos_tree.py --paradigm-comparison ...`

There is no standalone `python Utils/comparison.py ...` workflow documented for normal use.

---

## Outputs

<p>
  <img src="https://img.shields.io/badge/Artifact%20Root-src%2FOutput-f59e0b?style=flat-square" alt="Artifact Root" />
  <img src="https://img.shields.io/badge/Active-Kronos-0f766e?style=flat-square" alt="Active Kronos" />
  <img src="https://img.shields.io/badge/Legacy-Analysis-64748b?style=flat-square" alt="Legacy Analysis" />
</p>

Current output root:

```text
src/Output/
```

Current on-disk hierarchy:

```text
src/Output/
├── Analysis/
│   ├── Theory/
│   │   ├── ExperimentA/
│   │   ├── ExperimentB/
│   │   ├── ExperimentC/
│   │   ├── ExperimentD/
│   │   ├── ExperimentE/
│   │   ├── ExperimentF/
│   │   ├── ExperimentG/
│   │   ├── ExperimentH/
│   │   └── ExperimentI/
│   └── Uncertainty/
│       ├── All/
│       ├── Per_Granularity/
│       └── Probe/
└── Kronos/
    ├── autogluon/
    │   ├── DOWN/
    │   │   ├── OCP/
    │   │   └── Utility_Score/
    │   └── UP/
    │       ├── OCP/
    │       └── Utility_Score/
    ├── cache/
    └── randforest/
        ├── DOWN/
        │   ├── OCP/
        │   └── Utility_Score/
        └── UP/
            ├── OCP/
            └── Utility_Score/
```

How to read this structure:

- `src/Output/Kronos/` is the active result tree for the current M2 workflow.
- `src/Output/Kronos/cache/` stores dataset caches used by `kronos_tree.py`.
- `src/Output/Kronos/autogluon/` and `src/Output/Kronos/randforest/` currently hold model-family result folders split by `UP` and `DOWN`.
- `src/Output/Analysis/` keeps older theory and uncertainty-study outputs that are still present on disk but are not the main target of the current Kronos tree workflow.
- Additional model-family folders, such as `xgboost/`, appear here when those runs are generated.

---

## Exact Current `config.yaml`

<p>
  <img src="https://img.shields.io/badge/Config-Exact%20File%20Snapshot-2563eb?style=flat-square" alt="Config Exact File Snapshot" />
  <img src="https://img.shields.io/badge/Current%20Direction-down-ef4444?style=flat-square" alt="Current Direction down" />
  <img src="https://img.shields.io/badge/Granularity-all-0f766e?style=flat-square" alt="Granularity all" />
  <img src="https://img.shields.io/badge/Horizon-7-f59e0b?style=flat-square" alt="Horizon 7" />
</p>

The block below is the current file exactly as it exists today.

```yaml
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Kronos Tree Configuration
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# ┏━━━━━━━━━━ Paths ━━━━━━━━━━┓
paths:
  csv_dir: "/home/pablo/M2_DS/Secondary-Model/src/Data_MLA/Kronos/Crypto/TP/horizon_7"
  output_root: "/home/pablo/M2_DS/Secondary-Model/src/Output"

# ┏━━━━━━━━━━ Data Configuration ━━━━━━━━━━┓
data:
  load:
    symbol:          null          # or null for multi-asset or ["BTC", "ETH", "XRP", ...]
    target_col:      "meta_label"  # "meta_label" or "close" or "ground_truth"
    meta_label_mode: "tp"          # "fp" or "tp" or "og"
    direction:       "down"        # "up" or "down"
    granularity:     "all"         # "1d", "4h", etc. or "all" for multi-granularity
    forecast_horizon: 7
    m1: "fincast"

  # ┏━━━━━━━━━━ Data Splits ━━━━━━━━━━┓
  split:
    start_date: "2024-07-01"
    train_end:  "2025-05-30"
    val_end:    "2025-10-01"
    end_date:   "2026-01-25"
    context_length: 90

  # ┏━━━━━━━━━━ Features ━━━━━━━━━━┓
  features:
    input: ["open", "high", "low", "close", "volume"]

    # ┏━━━━━━━━━━ Engineered Window Features ━━━━━━━━━━┓
    engineered_features:
      selected: [bb_pctb_last, rsi_last, roc_5_last, roc_20_last, atr_norm_last]

# ┏━━━━━━━━━━ Evaluation ━━━━━━━━━━┓
evaluation:
  fee_per_trade: 0.002
```

---

## `config.yaml` Parameter Meanings

<p>
  <img src="https://img.shields.io/badge/paths-I%2FO-0f766e?style=flat-square" alt="paths IO" />
  <img src="https://img.shields.io/badge/data.load-target%20setup-2563eb?style=flat-square" alt="data load target setup" />
  <img src="https://img.shields.io/badge/data.split-time%20boundaries-f59e0b?style=flat-square" alt="data split time boundaries" />
  <img src="https://img.shields.io/badge/data.features-feature%20surface-7c3aed?style=flat-square" alt="data features feature surface" />
</p>

### `paths`

| Key | Current value | Meaning |
| --- | --- | --- |
| `paths.csv_dir` | `/home/pablo/M2_DS/Secondary-Model/src/Data_MLA/Kronos/Crypto/TP/horizon_7` | Root directory containing the processed Kronos CSV files consumed by the M2 pipeline. |
| `paths.output_root` | `/home/pablo/M2_DS/Secondary-Model/src/Output` | Base output directory. Artifacts are then written under the configured M1 bucket, e.g. `Output/Fincast` or `Output/Kronos`. |

### `data.load`

| Key | Current value | Meaning |
| --- | --- | --- |
| `data.load.symbol` | `null` | `null` means multi-asset loading. If set to a symbol or symbol list, loading becomes asset-specific. |
| `data.load.target_col` | `meta_label` | Which target column the M2 classifier learns to predict. |
| `data.load.meta_label_mode` | `tp` | Which meta-label variant to use. `tp` is the current active setup. |
| `data.load.direction` | `down` | Trade direction for the labeling and evaluation path. |
| `data.load.granularity` | `all` | Multi-granularity mode. This is why the main run modes for the current config are `--per-gran` and `--all-grans`. |
| `data.load.forecast_horizon` | `7` | Prediction horizon used by the M2 pipeline. It also matters for return alignment, backtesting, and delayed-feedback OCP logic. |
| `data.load.m1` | `fincast` | Declares which M1 model generated the upstream signals so caches, outputs, and reporting can be grouped under the correct output bucket. |

### `data.split`

| Key | Current value | Meaning |
| --- | --- | --- |
| `data.split.start_date` | `2024-07-01` | Earliest date included when building the dataset windows. |
| `data.split.train_end` | `2025-05-30` | End of the training segment. Samples after this date move to later splits. |
| `data.split.val_end` | `2025-10-01` | End of the validation segment. Samples after this date move to the test segment. |
| `data.split.end_date` | `2026-01-25` | Final date admitted into the dataset. |
| `data.split.context_length` | `90` | Number of timesteps per lookback window used during dataset construction. |

### `data.features`

| Key | Current value | Meaning |
| --- | --- | --- |
| `data.features.input` | `["open", "high", "low", "close", "volume"]` | Raw market columns used as the base inputs. |
| `data.features.engineered_features.selected` | `[bb_pctb_last, rsi_last, roc_5_last, roc_20_last, atr_norm_last]` | Engineered window-level features exposed to the tree model and the feature-analysis utilities. |

### `evaluation`

| Key | Current value | Meaning |
| --- | --- | --- |
| `evaluation.fee_per_trade` | `0.002` | Transaction fee assumption used when computing selective-trading utility and backtest metrics. |

---

## Reporting and Diagnostics

<p>
  <img src="https://img.shields.io/badge/Comparison-Separate%20vs%20Unified-2563eb?style=flat-square" alt="Comparison Separate vs Unified" />
  <img src="https://img.shields.io/badge/OCP-Practical%20Diagnostics-7c3aed?style=flat-square" alt="OCP Practical Diagnostics" />
  <img src="https://img.shields.io/badge/OCP-Theory-ef4444?style=flat-square" alt="OCP Theory" />
</p>

### Comparison Utilities

`Utils/comparison.py` builds the polished summary tables and CSV exports for:

- separate vs unified model structure
- validation and test performance panels
- backtest comparisons
- paradigm-level side-by-side reports

### OCP Diagnostics

`Utils/ocp_analysis.py` is the practical diagnostic entrypoint for completed OCP runs.

Usage:

```bash
python Utils/ocp_analysis.py --folder Output/Kronos/randforest/8h_down_tp
python Utils/ocp_analysis.py --folder Output/Kronos/randforest/unified_down_tp --mode unified
```

It currently covers:

- fixed-threshold comparison
- random baseline checks
- shuffled-label sanity checks
- rolling conformal coverage
- trade overlap versus utility threshold
- probability calibration inspection

### Theory File Status

`Utils/ocp_theory.py` is still present, but it is not the main path for current practical analysis. For active OCP validation work, use `Utils/ocp_analysis.py`.

---

## Practical Notes

<p>
  <img src="https://img.shields.io/badge/Use-Current%20Code-0f766e?style=flat-square" alt="Use Current Code" />
</p>

- The canonical output location for run results is `src/Output/Kronos/`.
- `kronos_tree.py` is the main CLI.
- `Utils/features.py` and `Utils/comparison.py` are callable modules, not standalone command-line programs.
- If you are trying to understand the current M2 stack, focus on `config.yaml`, `kronos_tree.py`, and `Utils/`.

---

## One-Line Summary

This repository is a modular M2 research workspace for tree-based meta-label filtering, selective-classification tooling, SAOCP diagnostics, backtesting, and comparison reporting, all driven by the current `config.yaml`.
