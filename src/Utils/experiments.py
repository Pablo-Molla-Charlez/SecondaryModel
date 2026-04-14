"""Experiment orchestrator — runs the full M2 pipeline for all models and directions.

This script automates the sequential execution of:
  1. Per-granularity training (kronos_tree.py --per-gran) for each model × direction
  2. Edge convergence protocol (seeds → cpcv → convergence) for each model × direction
  3. Combined UP+DOWN backtest (kronos_tree.py --combined-backtest) for each model

Usage:
    python Utils/experiments.py --config config_kronos.yaml
    python Utils/experiments.py --config config_kronos.yaml --models rf tabicl
    python Utils/experiments.py --config config_kronos.yaml --skip-training --skip-edge
    python Utils/experiments.py --config config_kronos.yaml --edge-trials 50 --edge-blocks 6
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path


# ┏━━━━━━━━━━ Constants ━━━━━━━━━━┓
ALL_MODELS = ["rf", "xgboost", "autogluon", "tabpfn", "tabpfn_ft", "tabicl"]
DIRECTIONS = ["up", "down"]

# ┏━━━━━━━━━━ CLI model name → output folder name (must match kronos_tree.py mapping) ━━━━━━━━━━┓
_MODEL_TO_FOLDER = {"rf":        "randforest",
                    "xgboost":   "xgboost",
                    "autogluon": "autogluon",
                    "tabpfn":    "tabpfn",
                    "tabpfn_ft": "tabpfn_ft",
                    "tabicl":    "tabicl"}

# ┏━━━━━━━━━━ CLI model name → edge.py --model arg ━━━━━━━━━━┓
_MODEL_TO_EDGE_CLI = {"rf":        "randforest",
                      "xgboost":   "xgboost",
                      "autogluon": "autogluon",
                      "tabpfn":    "tabpfn",
                      "tabpfn_ft": "tabpfn_ft",
                      "tabicl":    "tabicl"}

# ┏━━━━━━━━━━ Run subprocess ━━━━━━━━━━┓
def _run(cmd: list[str], label: str):
    """Run a subprocess and stream output. Returns True on success."""
    print(f"\n{'='*70}")
    print(f"[experiments] {label}")
    print(f"  CMD: {' '.join(cmd)}")
    print(f"{'='*70}\n")
    t0 = time.time()
    result = subprocess.run(cmd, cwd=str(Path(__file__).resolve().parent.parent))
    elapsed = time.time() - t0
    status = "OK" if result.returncode == 0 else f"FAIL (rc={result.returncode})"
    print(f"\n[experiments] {label} — {status} ({elapsed:.0f}s)")
    return result.returncode == 0

# ┏━━━━━━━━━━ Find cache ━━━━━━━━━━┓
def _find_cache(cfg_path: str, direction: str) -> str | None:
    """Resolve the multi-gran cache path for a direction from config."""
    import yaml
    
    # ┏━━━━━━━━━━ Load config ━━━━━━━━━━┓
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)
    output_root = Path(cfg["paths"]["output_root"])
    
    # ┏━━━━━━━━━━ Search all cache dirs for matching direction ━━━━━━━━━━┓
    for cache_dir in sorted(output_root.glob("**/cache*")):
        for pt in sorted(cache_dir.glob(f"multi_*_{direction}_*.pt")):
            return str(pt)
    return None

# ┏━━━━━━━━━━ Run experiments ━━━━━━━━━━┓
def run_experiments(config: str,
                    models: list[str],
                    skip_training: bool = False,
                    skip_edge: bool = False,
                    skip_combined: bool = False,
                    edge_trials: int = 100,
                    edge_blocks: int = 6,
                    edge_k_test: int = 2,
                    features: bool = True,
                    top5: bool = True):
    """Execute the full experiment suite."""
    python = sys.executable
    cfg_path = config
    results = {}

    # ┏━━━━━━━━━━ Phase 1: Training (per-gran) ━━━━━━━━━━┓
    if not skip_training:
        print(f"\n{'#'*70}")
        print(f"# PHASE 1: Per-Granularity Training")
        print(f"# Models: {models}")
        print(f"# Config: {cfg_path}")
        print(f"{'#'*70}")

        for model in models:
            for direction in DIRECTIONS:
                cache = _find_cache(cfg_path, direction)
                if cache is None:
                    print(f"  [SKIP] No cache found for direction={direction}")
                    continue

                label = f"Train {model.upper()} {direction.upper()}"
                cmd = [python, "kronos_tree.py",
                       "--config", cfg_path,
                       "--cache", cache,
                       "--per-gran",
                       "--model", model,
                       "--features", "true" if features else "false",           # TODO: TILL Feature Selection Pipeline
                       "--top5", "true" if (top5 and features) else "false"]    # TODO: TILL Feature Selection Pipeline
                ok = _run(cmd, label)
                results[label] = ok

    # ┏━━━━━━━━━━ Phase 2: Edge Convergence ━━━━━━━━━━┓
    if not skip_edge:
        print(f"\n{'#'*70}")
        print(f"# PHASE 2: Edge Convergence Protocol")
        print(f"# Models: {models}  |  trials={edge_trials}  blocks={edge_blocks}")
        print(f"{'#'*70}")

        for model in models:
            edge_cli = _MODEL_TO_EDGE_CLI[model]
            for direction in DIRECTIONS:
                cache = _find_cache(cfg_path, direction)
                if cache is None:
                    print(f"  [SKIP] No cache for direction={direction}")
                    continue

                # ┏━━━━━━━━━━ Seeds ━━━━━━━━━━┓
                label = f"Edge Seeds {model.upper()} {direction.upper()}"
                cmd = [python, "Utils/edge.py",
                       "--config", cfg_path,
                       "--cache", cache,
                       "--mode", "seeds",
                       "--trials", str(edge_trials),
                       "--model", edge_cli]
                ok = _run(cmd, label)
                results[label] = ok

                # ┏━━━━━━━━━━ CPCV ━━━━━━━━━━┓
                label = f"Edge CPCV {model.upper()} {direction.upper()}"
                cmd = [python, "Utils/edge.py",
                       "--config", cfg_path,
                       "--cache", cache,
                       "--mode", "cpcv",
                       "--n-blocks", str(edge_blocks),
                       "--k-test", str(edge_k_test),
                       "--model", edge_cli]
                ok = _run(cmd, label)
                results[label] = ok

                # ┏━━━━━━━━━━ Convergence score ━━━━━━━━━━┓
                label = f"Edge Convergence {model.upper()} {direction.upper()}"
                cmd = [python, "Utils/edge.py",
                       "--config", cfg_path,
                       "--cache", cache,
                       "--convergence",
                       "--model", edge_cli]
                ok = _run(cmd, label)
                results[label] = ok

    # ┏━━━━━━━━━━ Phase 3: Combined UP+DOWN Backtest ━━━━━━━━━━┓
    if not skip_combined:
        import yaml
        
        # ┏━━━━━━━━━━ Print header ━━━━━━━━━━┓
        print(f"\n{'#'*70}")
        print(f"# PHASE 3: Combined UP+DOWN Backtests")
        print(f"{'#'*70}")

        # ┏━━━━━━━━━━ Load config ━━━━━━━━━━┓
        with open(cfg_path, "r") as f:
            cfg = yaml.safe_load(f)

        # ┏━━━━━━━━━━ Determine M1 bucket ━━━━━━━━━━┓
        m1 = cfg.get("data", {}).get("load", {}).get("m1", "kronos").lower()
        m1_bucket = {"kronos": "Kronos", "fincast": "Fincast"}.get(m1, m1.capitalize())
        output_root = Path(cfg["paths"]["output_root"]) / m1_bucket

        # ┏━━━━━━━━━━ Run combined backtest for each model ━━━━━━━━━━┓
        for model in models:
            folder = _MODEL_TO_FOLDER[model]
            up_dir = output_root / folder / "UP" / "Utility_Score"
            dn_dir = output_root / folder / "DOWN" / "Utility_Score"

            if not up_dir.exists() or not dn_dir.exists():
                print(f"  [SKIP] Missing UP or DOWN results for {model}: {up_dir}, {dn_dir}")
                continue

            # ┏━━━━━━━━━━ Run combined backtest ━━━━━━━━━━┓
            label = f"Combined Backtest {model.upper()}"
            cmd = [python, "kronos_tree.py",
                   "--config", cfg_path,
                   "--model", model,
                   "--combined-backtest", str(up_dir), str(dn_dir)]
            ok = _run(cmd, label)
            results[label] = ok

    # ┏━━━━━━━━━━ Summary ━━━━━━━━━━┓
    print(f"\n{'='*70}")
    print(f"EXPERIMENT SUMMARY")
    print(f"{'='*70}")
    for label, ok in results.items():
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {label}")
    n_pass = sum(results.values())
    n_total = len(results)
    print(f"\n  {n_pass}/{n_total} steps completed successfully.")
    print(f"{'='*70}")


def main():
    parser = argparse.ArgumentParser(description="Experiment orchestrator — full M2 pipeline for all models and directions")
    
    # ┏━━━━━━━━━━ Arguments ━━━━━━━━━━┓
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    parser.add_argument("--models", nargs="+", default=ALL_MODELS, choices=ALL_MODELS, help=f"Models to run (default: all). Choices: {ALL_MODELS}")
    parser.add_argument("--skip-training", action="store_true",      help="Skip Phase 1 (training)")
    parser.add_argument("--skip-edge",     action="store_true",      help="Skip Phase 2 (edge convergence)")
    parser.add_argument("--skip-combined", action="store_true",      help="Skip Phase 3 (combined backtest)")
    parser.add_argument("--edge-trials",   type=int, default=100,    help="Seed trials for edge analysis")
    parser.add_argument("--edge-blocks",   type=int, default=6,      help="CPCV blocks")
    parser.add_argument("--edge-k-test",   type=int, default=2,      help="CPCV test blocks per split")
    parser.add_argument("--features",      type=str, default="true", choices=["true", "false"], help="Run feature analysis during training")
    parser.add_argument("--top5",          type=str, default="true", choices=["true", "false"], help="Run top-5 feature analysis during training")
    args = parser.parse_args()

    # ┏━━━━━━━━━━ Run experiments ━━━━━━━━━━┓
    run_experiments(config        = args.config,
                    models        = args.models,
                    skip_training = args.skip_training,
                    skip_edge     = args.skip_edge,
                    skip_combined = args.skip_combined,
                    edge_trials   = args.edge_trials,
                    edge_blocks   = args.edge_blocks,
                    edge_k_test   = args.edge_k_test,
                    features      = args.features.lower() == "true",
                    top5          = args.top5.lower() == "true")


if __name__ == "__main__":
    main()
