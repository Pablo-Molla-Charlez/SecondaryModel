#!/usr/bin/env python
"""
Batch runner to estimate the mean/stdev of CTTS metrics over many random seeds.

- Launches `train.py` repeatedly with different seeds.
- Captures the metrics JSON produced per run.
- Aggregates mean/std tables and produces a bar chart with error bars.
"""
from __future__ import annotations

import sys
import json
import yaml
import random
import shutil
import argparse
import subprocess
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from datetime import datetime
from typing import Dict, Iterable, List, Sequence

# ┏━━━━━━━━━━ Metrics to Study ━━━━━━━━━━┓
SUMMARY_KEYS: Sequence[str] = (# M2 Validation Results: Classification [w/o Optimized Threshold]
                               "Val_Accuracy",
                               "Val_Precision",
                               "Val_Recall",
                               "Val_F1",
                               "Val_FBeta",

                               # M2 Validation Results: Classification [Optimized Threshold]
                               "Val_Accuracy@Tau",
                               "Val_Precision@Tau",
                               "Val_Recall@Tau",
                               "Val_F1@Tau",
                               "Val_FBeta@Tau",

                               # M2 Validation Results: Risk & Coverage
                               "Val_AURC",
                               "Val_Tau",
                               "Val_Coverage@Tau",
                               "Val_Risk@Tau",
                               "Selected_Count@Tau",
                               
                               # M2 Test Results: Classification [w/o Optimized Threshold]
                               "Test_Accuracy",
                               "Test_Precision",
                               "Test_Recall",
                               "Test_F1",
                               "Test_FBeta",

                               # M2 Test Results: Classification [Optimized Threshold]
                               "Test_Accuracy@Tau",
                               "Test_Precision@Tau",
                               "Test_Recall@Tau",
                               "Test_F1@Tau",
                               "Test_FBeta@Tau",

                               # M2 Test Results: Risk & Coverage
                               "Test_Coverage@Tau",
                               "Test_Risk@Tau",
                               "Test_Selected_Count@Tau",
                               
                               # M1+M2 Consensus Results [w/o Optimized Threshold]
                               "M1+M2_Accuracy",
                               "M1+M2_Precision",
                               "M1+M2_Recall",
                               "M1+M2_F1",
                               "M1+M2_FBeta",

                               # M1+M2 Consensus Results [Optimized Threshold]
                               "M1+M2_Accuracy@Tau",
                               "M1+M2_Precision@Tau",
                               "M1+M2_Recall@Tau",
                               "M1+M2_F1@Tau",
                               "M1+M2_FBeta@Tau")

CLASSIFICATION_PLOTS = ({"slug": "_Val_Classification",
                         "title": "M2 Validation",
                         "columns": ("Val_Accuracy",
                                     "Val_Precision",
                                     "Val_Recall",
                                     "Val_F1",
                                     "Val_FBeta"),
                         "ylim": (0.0, 1.0)},

                        {"slug": "_Val_Classification_tau",
                         "title": "M2 Validation (@tau)",
                         "columns": ("Val_Accuracy@Tau",
                                     "Val_Precision@Tau",
                                     "Val_Recall@Tau",
                                     "Val_F1@Tau",
                                     "Val_FBeta@Tau"),
                         "ylim": (0.0, 1.0)},

                        {"slug": "_Test_Classification",
                         "title": "M2 Test",
                         "columns": ("Test_Accuracy",
                                     "Test_Precision",
                                "Test_Recall",
                                "Test_F1",
                                "Test_FBeta"),
                        "ylim": (0.0, 1.0)},

                        {"slug": "_Test_Classification_tau",
                         "title": "M2 Test (@tau)",
                         "columns": ("Test_Accuracy@Tau",
                                     "Test_Precision@Tau",
                                     "Test_Recall@Tau",
                                     "Test_F1@Tau",
                                     "Test_FBeta@Tau"),
                                     "ylim": (0.0, 1.0)}
                                     ,
                        {"slug": "_M1+M2_Consensus",
                         "title": "M1+M2 Consensus",
                         "columns": ("M1+M2_Accuracy",
                                     "M1+M2_Precision",
                                     "M1+M2_Recall",
                                     "M1+M2_F1",
                                     "M1+M2_FBeta"),
                         "ylim": (0.0, 1.0)},

                        {"slug": "_M1+M2_Consensus_tau",
                         "title": "M1+M2 Consensus (@tau)",
                         "columns": ("M1+M2_Accuracy@Tau",
                                     "M1+M2_Precision@Tau",
                                     "M1+M2_Recall@Tau",
                                     "M1+M2_F1@Tau",
                                     "M1+M2_FBeta@Tau"),
                         "ylim": (0.0, 1.0)})

RISK_COVERAGE_PLOTS = ({"slug": "_Val_Risk_Coverage",
                        "title": "M2 Validation Risk & Coverage",
                        "value_cols": ("Val_AURC",
                                        "Val_Tau",
                                        "Val_Coverage@Tau",
                                        "Val_Risk@Tau"),
                        "count_cols": ("Selected_Count@Tau",)},
    
                        {"slug": "_Test_Risk_Coverage",
                        "title": "M2 Test Risk & Coverage",
                        "value_cols": ("Test_Coverage@Tau",
                                        "Test_Risk@Tau"),
                        "count_cols": ("Test_Selected_Count@Tau")})


# ┏━━━━━━━━━━ 1. Helper Function: Cleaning Name ━━━━━━━━━━┓
def _granularity_slug(text: str) -> str:
    return text.replace(" ", "").replace("-", "").lower()

# ┏━━━━━━━━━━ 2. Helper Function: Extracting Data Path Information ━━━━━━━━━━┓
def _task_root(cfg: Dict) -> Path:
    # ┏━━━━━━━━━━ Extracting Asset Information ━━━━━━━━━━┓
    provider    = cfg["dataset"]["source"].capitalize()
    symbol      = cfg["dataset"]["symbol"]
    task        = cfg["training_mode"]["normal_task"].upper()
    granularity = cfg["training_mode"]["granularity_usual"]
    
    # ┏━━━━━━━━━━ Extracting Meta-Label Mode ━━━━━━━━━━┓
    meta_mode   = cfg["training_mode"]["meta_label_usual"].lower()
    meta_dir_suffix = "og" if meta_mode == "original" else meta_mode
    granularity_slug_with_meta = f"{_granularity_slug(granularity)}_{meta_dir_suffix}"
    
    return (Path(cfg["paths"]["output_root"])
            / "Usual"
            / provider
            / symbol
            / task
            / granularity_slug_with_meta)

# ┏━━━━━━━━━━ 4. Helper Function: Global Path ━━━━━━━━━━┓
def _list_run_dirs(root: Path) -> set[Path]:
    if not root.exists():
        return set()
    return set(root.glob("Run_*"))

# ┏━━━━━━━━━━ 5. Helper Function: Random Seed ━━━━━━━━━━┓
def _choose_seed(index: int, base_seed: int | None, rng: random.Random) -> int:
    # ┏━━━━━━━━━━ If not None, the new seed is the addition, otherwise Random ━━━━━━━━━━┓ 
    if base_seed is not None:
        return base_seed + index
    return rng.randrange(0, 2**32 - 1)

# ┏━━━━━━━━━━ 6. Helper Function: Random Seed ━━━━━━━━━━┓
def _bar_with_error(ax: plt.Axes,
                    df: pd.DataFrame,
                    columns: Sequence[str],
                    ylim: tuple[float, float] | None = None,
                    ylabel: str = "Value") -> None:
    
    positions = range(len(columns))
    means = df[columns].mean()
    stds = df[columns].std()
    ax.bar(positions, means.values, yerr=stds.values, capsize=4)
    ax.set_xticks(list(positions))
    ax.set_xticklabels(columns, rotation=45, ha="right")
    ax.set_ylabel(ylabel)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.grid(axis="y", linestyle="--", alpha=0.2)

# ┏━━━━━━━━━━ 7. Helper Function: Random Seed ━━━━━━━━━━┓
def _plot_metric_group(df: pd.DataFrame,
                       columns: Sequence[str],
                       title: str,
                       out_path: Path,
                       ylim: tuple[float, float] | None = None) -> bool:
    available = [col for col in columns if col in df.columns]
    if not available:
        print(f"[Study] Skipping {title}: metrics missing.")
        return False
    fig_width = max(8.0, 0.9 * len(available))
    fig, ax = plt.subplots(figsize=(fig_width, 4.5))
    _bar_with_error(ax, df, available, ylim=ylim)
    ax.set_title(f"{title} mean ± std")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[Study] Saved summary plot to {out_path}")
    return True

# ┏━━━━━━━━━━ 8. Helper Function: Random Seed ━━━━━━━━━━┓
def _plot_risk_coverage_group(df: pd.DataFrame,
                              value_cols: Sequence[str],
                              count_cols: Sequence[str],
                              title: str,
                              out_path: Path) -> bool:
    value_available = [col for col in value_cols if col in df.columns]
    count_available = [col for col in count_cols if col in df.columns]
    sections: list[tuple[str, list[str]]] = []
    if value_available:
        sections.append(("value", value_available))
    if count_available:
        sections.append(("count", count_available))
    if not sections:
        print(f"[Study] Skipping {title}: metrics missing.")
        return False

    max_cols = max(len(cols) for _, cols in sections)
    fig, axes = plt.subplots(len(sections), 1,
                             figsize=(max(8.0, 0.9 * max_cols), 4.0 * len(sections)))
    if len(sections) == 1:
        axes = [axes]

    for ax, (section_type, cols) in zip(axes, sections):
        ylim = (0.0, 1.0) if section_type == "value" else None
        ylabel = "Value" if section_type == "value" else "Count"
        suffix = " (probability metrics)" if section_type == "value" else " (counts)"
        _bar_with_error(ax, df, cols, ylim=ylim, ylabel=ylabel)
        ax.set_title(f"{title}{suffix}")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[Study] Saved summary plot to {out_path}")
    return True


def run_study(config_path: Path,
              runs: int,
              output_dir: Path,
              python_exe: str,
              base_seed: int | None = None) -> None:

    # ┏━━━━━━━━━━ Parent Directory and Path of Train.py ━━━━━━━━━━┓ 
    script_dir = Path(__file__).resolve().parent
    train_script = script_dir / "train.py"
    
    # ┏━━━━━━━━━━ Loading Config.yaml ━━━━━━━━━━┓
    cfg = yaml.safe_load(config_path.read_text())
    task = cfg["training_mode"]["normal_task"].upper()
    
    # ┏━━━━━━━━━━ Data Path ━━━━━━━━━━┓
    task_root = _task_root(cfg)

    # ┏━━━━━━━━━━ Statistical Study Path Storage & Creation of Folders ━━━━━━━━━━┓
    timestamp = datetime.now().strftime("Study_%Y%m%d_%H%M%S")
    study_dir = output_dir / timestamp
    logs_dir = study_dir / "Logs"
    study_dir.mkdir(parents = True, exist_ok = True)
    logs_dir.mkdir(parents = True, exist_ok = True)

    # ┏━━━━━━━━━━ Main Loop: Running Train+Val+Test a number "runs" times ━━━━━━━━━━┓
    rng = random.Random()
    rows: List[Dict] = []
    print(f"\nStatistical Study: {runs} Runs.")
    for idx in range(1, runs + 1):
        # ┏━━━━━━━━━━ Seed Selection ━━━━━━━━━━┓
        seed = _choose_seed(idx - 1, base_seed, rng)
        print(f"\n[Run {idx}/{runs}] with seed = {seed}.")

        # ┏━━━━━━━━━━ To locate Runs in "Usual" Folder BEFORE running a trial ━━━━━━━━━━┓
        before = _list_run_dirs(task_root)
        
        # ┏━━━━━━━━━━ To store logs for each trial ━━━━━━━━━━┓
        log_path = logs_dir / f"run_{idx:03d}.log"
        with log_path.open("w") as log_file:
            subprocess.run(
                [python_exe, str(train_script), "--config", str(config_path), "--seed",   str(seed)],
                cwd    = script_dir,
                check  = True,                # Statistical study stops immediately when any training run fails or crashes
                stdout = log_file,            # Storing stdout (“standard output”) ~ Information appearing on Terminal.
                stderr = subprocess.STDOUT,   # Storing stderr (“standard error)   ~ Errors appearing on Terminal.
            )

        # ┏━━━━━━━━━━ To locate Runs in "Usual" Folder AFTER running a trial ━━━━━━━━━━┓
        after = _list_run_dirs(task_root)
        
        # ┏━━━━━━━━━━ The difference provides the new trial run from which extract results ━━━━━━━━━━┓
        new_dirs = after - before
        
        # ┏━━━━━━━━━━ Safety Check ━━━━━━━━━━┓
        if not new_dirs:
            raise RuntimeError("Could not locate the newly created Run_* directory. "
                               "Ensure output_root points to the expected location.")
                
        # ┏━━━━━━━━━━ Extracting Paths: New Run and R&C Analysis JSON ━━━━━━━━━━┓
        run_dir = max(new_dirs, key = lambda p: p.stat().st_mtime)
        metrics_path = run_dir / f"M2_{task}_R&C_Analysis.json"
        
        # ┏━━━━━━━━━━ Safety Check ━━━━━━━━━━┓
        if not metrics_path.exists():
            raise FileNotFoundError(f"Metrics file not found at {metrics_path}")

        # ┏━━━━━━━━━━ Extracting R&C Analysis Metrics ━━━━━━━━━━┓
        with metrics_path.open("r") as fh:
            metrics = json.load(fh)

        row = {"run": idx, "seed": seed, "run_dir": str(run_dir)}
        for key in SUMMARY_KEYS:
            row[key] = metrics.get(key)
        rows.append(row)
        
        # ┏━━━━━━━━━━ Copying metrics from JSON to study folder ━━━━━━━━━━┓
        shutil.copy(metrics_path, study_dir / f"Run_{idx:03d}_metrics.json")

    
    # ┏━━━━━━━━━━ Creating DataFrame & CSV with metrics ━━━━━━━━━━┓
    df = pd.DataFrame(rows)
    runs_csv = study_dir / "Stats_Complete_Metrics.csv"
    df.to_csv(runs_csv, index = False)
    print(f"\n[Study] Saved per-run metrics to {runs_csv}")
    
    metrics_cols = [k for k in SUMMARY_KEYS if k in df.columns]
    if metrics_cols:
        # ┏━━━━━━━━━━ Creating Statistical Study CSV ━━━━━━━━━━┓
        summary = df[metrics_cols].agg(["mean", "std"])
        summary_csv = study_dir / "Stats_Study.csv"
        summary.to_csv(summary_csv)
        print(f"[Study] Saved summary stats to {summary_csv}")

        # ┏━━━━━━━━━━ Classification Summary Plots ━━━━━━━━━━┓
        for plot_cfg in CLASSIFICATION_PLOTS:
            _plot_metric_group(df,
                               columns  = plot_cfg["columns"],
                               title    = plot_cfg["title"],
                               out_path = study_dir / f"{plot_cfg['slug']}.png",
                               ylim     = plot_cfg["ylim"])

        # ┏━━━━━━━━━━ Risk & Coverage Summary Plots ━━━━━━━━━━┓
        for plot_cfg in RISK_COVERAGE_PLOTS:
            _plot_risk_coverage_group(df,
                                      value_cols = plot_cfg["value_cols"],
                                      count_cols = plot_cfg["count_cols"],
                                      title      = plot_cfg["title"],
                                      out_path   = study_dir / f"{plot_cfg['slug']}.png")
    else:
        print("[Study] No numeric metrics captured for aggregation.")

    seeds_txt = study_dir / "Stats_Seeds.txt"
    with seeds_txt.open("w") as fh:
        for row in rows:
            fh.write(f"Run={row['run']}\tseed={row['seed']}\tpath={row['run_dir']}\n")
    print(f"[Study] Recorded seeds in {seeds_txt}")
    


def parse_args() -> argparse.Namespace:
    
    parser = argparse.ArgumentParser(description = "Run CTTS multiple times and aggregate metrics.")

    # ┏━━━━━━━━━━ Config Argument ━━━━━━━━━━┓
    parser.add_argument("--config",
                        type    = str,
                        default = "config.yaml",
                        help    = "Path to the CTTS config.yaml to reuse.")

    # ┏━━━━━━━━━━ Run Argument ━━━━━━━━━━┓
    parser.add_argument("--runs",
                        type    = int,
                        default = 10,
                        help    = "Number of independent runs to execute.")

    # ┏━━━━━━━━━━ Output Directory Argument ━━━━━━━━━━┓
    parser.add_argument("--output-dir",
                        type    = str,
                        default = "Statistical_Studies",
                        help    = "Directory where study artifacts will be stored.")

    # ┏━━━━━━━━━━ Python Argument (use default) ━━━━━━━━━━┓
    parser.add_argument("--python",
                        type    = str,
                        default = sys.executable,
                        help    = "Python interpreter used for launching train.py")

    # ┏━━━━━━━━━━ Seed Argument ━━━━━━━━━━┓
    parser.add_argument("--base-seed",
                        type    = int,
                        default = None,
                        help    = "Optional base seed; actual seeds become base+run_index. "
                        "If omitted, seeds are random.")
    
    return parser.parse_args()


def main() -> None:
    # ┏━━━━━━━━━━ Extracting Arguments & Paths ━━━━━━━━━━┓
    args = parse_args()
    config_path = Path(args.config)
    script_dir = Path(__file__).resolve().parent
    
    # ┏━━━━━━━━━━ Config Path ━━━━━━━━━━┓
    if not config_path.is_absolute():
        config_path = script_dir / config_path
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found at {config_path}")

    # ┏━━━━━━━━━━ Output Directory Path ━━━━━━━━━━┓
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = script_dir / output_dir

    # ┏━━━━━━━━━━ Run Statistical Study ━━━━━━━━━━┓
    run_study(config_path = config_path,
              runs        = args.runs,
              output_dir  = output_dir,
              python_exe  = args.python,
              base_seed   = args.base_seed)


if __name__ == "__main__":
    main()
