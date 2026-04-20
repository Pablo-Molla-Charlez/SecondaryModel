"""Experiment orchestrator — runs the full M2 pipeline from a single config.yaml.

Iterates over the (m2 × direction × granularity) cross product defined in
`experiment:` and dispatches subprocesses for each enabled phase in `runtime.skip`:
  1. Per-granularity training      → kronos_tree.py         (phase=training)
  2. Edge convergence protocol     → python -m Utils.edge   (phase=edge)
  3. Combined UP+DOWN backtest     → kronos_tree.py         (phase=combined)
  4. Feature selection experiment  → feature_selection_experiment.py (phase=feature_selection)

Usage:
    python Utils/experiments.py --config config.yaml
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path
import yaml
import json
import os

# experiments.py lives in src/Utils/ — insert src/ so "import Utils" works
# regardless of the working directory, matching every other entry-point script.
_SRC = Path(__file__).resolve().parent.parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from Utils.utils import _load_config


# ┏━━━━━━━━━━ Run subprocess ━━━━━━━━━━┓
def _run(cmd: list[str], label: str):
    """Run a subprocess and stream output. Returns True on success."""
    print(f"\n{'=' * 70}")
    print(f"[experiments] {label}")
    print(f"  CMD: {' '.join(cmd)}")
    print(f"{'=' * 70}\n")
    t0 = time.time()
    result = subprocess.run(cmd, cwd=str(Path(__file__).resolve().parent.parent))
    elapsed = time.time() - t0
    status = "OK" if result.returncode == 0 else f"FAIL (rc={result.returncode})"
    print(f"\n[experiments] {label} — {status} ({elapsed:.0f}s)")
    return result.returncode == 0


# ┏━━━━━━━━━━ Find cache ━━━━━━━━━━┓
def _find_cache(cfg_path: str, direction: str, m1: str = None) -> str | None:
    """Resolve the multi-gran cache path for a direction from config."""
    
    # ┏━━━━━━━━━━ Search the specific M1 cache dir ━━━━━━━━━━┓
    cache_dir = Path(cfg_path) / m1.capitalize() / "cache"
    if not cache_dir.exists():
        return None
    
    for pt in sorted(cache_dir.glob(f"multi_7_fee_{direction}_*.pt")):
        return str(pt)
    
    return None


# ┏━━━━━━━━━━ Run experiments ━━━━━━━━━━┓
def run_experiments(config: dict):
    """Execute the full experiment suite."""
    
    python = sys.executable
    results = {}
    
    # ┏━━━━━━━━━━ Phase 1: Training (per-gran) ━━━━━━━━━━┓
    if not config["runtime"]['skip']["training"]:
        print(f"\n{'#' * 70}")
        print(f"# PHASE 1: Per-Granularity Training")
        print(f"# M1 Model: {config['experiment']['m1']}")
        print(f"# M2 Models: {config['experiment']['m2']}")
        print(f"# Directions: {config['experiment']['direction']}")
        print(f"# Granularities: {config['experiment']['granularity']}")
        print(f"{'#' * 70}")
        
        for m2 in config['experiment']['m2']:
            for direction in config['experiment']['direction']:
                for granularity in config['experiment']['granularity']:
                    cache_path = _find_cache(config["paths"]["output_root"], direction, m1=config['experiment']['m1'])
                    if cache_path is None:
                        print(f"  [SKIP] No cache found for M1={config['experiment']['m1']} and direction={direction}")
                        continue
                    
                    label = f"Train {m2.upper()} {direction.upper()} {granularity.upper()}"
                    cmd = [python, "kronos_tree.py",
                           "--cache_path", cache_path,
                           "--config", json.dumps(config),
                           "--phase", "training",
                           "--m2", m2,
                           "--direction", direction,
                           "--granularity", granularity]
                    ok = _run(cmd, label)
                    results[label] = ok
    
    # ┏━━━━━━━━━━ Phase 2: Edge Convergence ━━━━━━━━━━┓
    if not config["runtime"]["skip"]['edge']:
        print(f"\n{'#' * 70}")
        print(f"# PHASE 2: Edge Convergence Protocol")
        print(f"# M1 Model: {config['experiment']['m1']}")
        print(f"# M2 Models: {config['experiment']['m2']}")
        print(f"# Directions: {config['experiment']['direction']}")
        print(f"# Granularities: {config['experiment']['granularity']}")
        print(f"# Trials: {config['runtime']['edge']['n_trials']}")
        print(f"# Blocks: {config['runtime']['edge']['n_blocks']}")
        print(f"{'#' * 70}")
        
        for m2 in config['experiment']['m2']:
            for direction in config['experiment']['direction']:
                for granularity in config['experiment']['granularity']:
                    cache_path = _find_cache(config["paths"]["output_root"], direction, m1=config['experiment']['m1'])
                    if cache_path is None:
                        print(f"  [SKIP] No cache for direction={direction}")
                        continue
                    
                    # ┏━━━━━━━━━━ Seeds ━━━━━━━━━━┓
                    label = f"Edge Seeds {m2.upper()} {direction.upper()}"
                    cmd = [python, "-m", "Utils.edge",  # TODO why is this not via kronos_tree.py?
                           "--config", json.dumps(config),
                           "--cache_path", cache_path,
                           "--mode", "seeds",
                           '--phase', 'edge',
                           '--m2', m2,
                           '--direction', direction,
                           '--granularity', granularity]
                    ok = _run(cmd, label)
                    results[label] = ok
                    
                    # ┏━━━━━━━━━━ CPCV ━━━━━━━━━━┓
                    label = f"Edge CPCV {m2.upper()} {direction.upper()}"
                    cmd = [python, "-m", "Utils.edge",  # TODO why is this not via kronos_tree.py?
                           "--config", json.dumps(config),
                           "--cache_path", cache_path,
                           "--mode", "cpcv",
                           '--phase', 'edge',
                           '--m2', m2,
                           '--direction', direction,
                           '--granularity', granularity]
                    ok = _run(cmd, label)
                    results[label] = ok
                    
                    # ┏━━━━━━━━━━ Convergence score ━━━━━━━━━━┓
                    label = f"Edge Convergence {m2.upper()} {direction.upper()}"
                    cmd = [python, "-m", "Utils.edge",
                           "--config", json.dumps(config),
                           "--cache_path", cache_path,
                           "--mode", "convergence",
                           '--phase', 'edge',
                           '--m2', m2,
                           '--direction', direction,
                           '--granularity', granularity]
                    ok = _run(cmd, label)
                    results[label] = ok
    
    # ┏━━━━━━━━━━ Phase 3: Combined UP+DOWN Backtest ━━━━━━━━━━┓
    if not config["runtime"]["skip"]['combined']:
        # ┏━━━━━━━━━━ Print header ━━━━━━━━━━┓
        print(f"\n{'#' * 70}")
        print(f"# PHASE 3: Combined UP+DOWN Backtests")
        print(f"{'#' * 70}")
        
        output_root = Path(config["paths"]["output_root"]) / config["experiment"]["m1"].capitalize()
        
        for m2 in config['experiment']['m2']:
            for granularity in config['experiment']['granularity']:
                # TODO this is not required here!?
                # cache_path = _find_cache(config["paths"]["output_root"], direction, m1=config['experiment']['m1'])
                # if cache_path is None:
                #     print(f"  [SKIP] No cache found for M1={config['experiment']['m1']} and direction={direction}")
                #     continue
                
                config["runtime"]["combined"]["combined_backtest"][0] = str(output_root / m2 / "UP" / "Utility_Score")
                config["runtime"]["combined"]["combined_backtest"][1] = str(output_root / m2 / "DOWN" / "Utility_Score")
                if not os.path.exists(config["runtime"]["combined"]["combined_backtest"][0]) or not \
                    os.path.exists(config["runtime"]["combined"]["combined_backtest"][1]):
                    print(
                        f"  [SKIP] Missing UP or DOWN results for {m2}: {config["runtime"]["combined"]["combined_backtest"][1]}, {config["runtime"]["combined"]["combined_backtest"][1]}")
                    continue
                # ┏━━━━━━━━━━ Run combined backtest ━━━━━━━━━━┓
                label = f"Combined Backtest {m2.upper()} {granularity.upper()}"
                cmd = [python, "kronos_tree.py",
                       "--cache_path", "not_needed_here",
                       "--config", json.dumps(config),
                       "--phase", "combined",
                       "--m2", m2,
                       "--direction", "not_needed_here",
                       "--granularity", granularity]
                # "--combined-backtest", str(up_dir), str(dn_dir)]
                ok = _run(cmd, label)
                results[label] = ok
    
    # ┏━━━━━━━━━━ Phase 4: Feature selection ━━━━━━━━━━┓
    if not config["runtime"]["skip"]['feature_selection']:
        # ┏━━━━━━━━━━ Print header ━━━━━━━━━━┓
        print(f"\n{'#' * 70}")
        print(f"# PHASE 4: Feature Selection Analysis")
        print(f"# M1 Model: {config['experiment']['m1']}")
        print(f"# M2 Models: {config['experiment']['m2']}")
        print(f"# Directions: {config['experiment']['direction']}")
        print(f"# Granularities: {config['experiment']['granularity']}")
        print(f"{'#' * 70}")
        
        for m2 in config['experiment']['m2']:
            for direction in config['experiment']['direction']:
                for granularity in config['experiment']['granularity']:
                    cache_path = _find_cache(config["paths"]["output_root"], direction, m1=config['experiment']['m1'])
                    if cache_path is None:
                        print(f"  [SKIP] No cache found for M1={config['experiment']['m1']} and direction={direction}")
                        continue
                    
                    label = f"Train {m2.upper()} {direction.upper()} {granularity.upper()}"
                    cmd = [python, "feature_selection_experiment.py",
                           "--cache_path", cache_path,
                           "--config", json.dumps(config),
                           "--phase", "feature_selection",
                           "--m2", m2,
                           "--direction", direction,
                           "--granularity", granularity]
                    ok = _run(cmd, label)
                    results[label] = ok
    
    # ┏━━━━━━━━━━ Summary ━━━━━━━━━━┓
    print(f"\n{'=' * 70}")
    print(f"EXPERIMENT SUMMARY")
    print(f"{'=' * 70}")
    for label, ok in results.items():
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {label}")
    n_pass = sum(results.values())
    n_total = len(results)
    print(f"\n  {n_pass}/{n_total} steps completed successfully.")
    print(f"{'=' * 70}")


def main():
    parser = argparse.ArgumentParser(
        description="Experiment orchestrator — full M2 pipeline for all models and directions")
    
    # ┏━━━━━━━━━━ Arguments ━━━━━━━━━━┓
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    
    args = parser.parse_args()
    
    config_path = Path(args.config)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    
    with config_path.open("r") as f:
        cfg = yaml.safe_load(f) or {}
    
    print(f"\n{'=' * 70}")
    print(f"EXPERIMENT CONFIG")
    print(yaml.dump(cfg, sort_keys=False))
    print(f"{'=' * 70}")
    
    # ┏━━━━━━━━━━ Run experiments ━━━━━━━━━━┓
    run_experiments(config=cfg)


if __name__ == "__main__":
    main()
