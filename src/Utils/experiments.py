"""Experiment orchestrator — runs the full M2 pipeline from a single config.yaml.

Iterates over the (m2 x direction x granularity) cross product defined in
`experiment:` and dispatches subprocesses for each enabled phase in `runtime.skip`:
  0. Hyperparameter optimization   → python -m Utils.hpo    (phase=hpo; rf/tabpfn/tabicl/tabm only)
  1. Per-granularity training      → m2_pipeline.py         (phase=training)
  2. Edge convergence protocol     → python -m Utils.edge   (phase=edge)
  3. Combined UP+DOWN backtest     → m2_pipeline.py         (phase=combined)
  4. Feature selection experiment  → Utils/feature_selection_experiment.py (phase=feature_selection)

Usage:
    python Utils/experiments.py --config config.yaml
"""

import argparse
import pandas as pd
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

from Utils.utils import _load_config, HPO_SUPPORTED_M2

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
    
    for pt in sorted(cache_dir.glob(f"multi_{m1.lower()}_7_fee_{direction}_*.pt")):
        return str(pt)
    
    return None


# ┏━━━━━━━━━━ Run experiments ━━━━━━━━━━┓
def run_experiments(config: dict, config_path: str):
    """Execute the full experiment suite."""

    python = sys.executable
    results = {}

    # ┏━━━━━━━━━━ Phase 0: Hyperparameter Optimization ━━━━━━━━━━┓
    # Runs per (m2, direction) over ALL configured granularities in one invocation.
    # Results land in Output/{M1}/HPO/{m2}/{DIR}/{gran}/best_params.json and are
    # picked up by Phase 1 training via Utils.utils._load_best_params.
    if not config["runtime"]["skip"].get("hpo", True):
        print(f"\n{'#' * 70}")
        print(f"# PHASE 0: Hyperparameter Optimization")
        print(f"# M1 Model: {config['experiment']['m1']}")
        print(f"# M2 Models: {config['experiment']['m2']}")
        print(f"# Directions: {config['experiment']['direction']}")
        print(f"# Granularities: {config['experiment']['granularity']}")
        print(f"# Trials: {config['runtime']['hpo']['n_trials']}")
        print(f"{'#' * 70}")

        # ┏━━━━━━━━━━ Loop through M2 models ━━━━━━━━━━┓
        for m2 in config['experiment']['m2']:

            # ┏━━━━━━━━━━ Skip if HPO not supported for m2 ━━━━━━━━━━┓
            if m2 not in HPO_SUPPORTED_M2:
                print(f"  [SKIP] HPO not supported for m2={m2} (supported: {sorted(HPO_SUPPORTED_M2)}) — will use defaults")
                continue

            # ┏━━━━━━━━━━ Loop through directions ━━━━━━━━━━┓
            for direction in config['experiment']['direction']:

                # ┏━━━━━━━━━━ Find cache path ━━━━━━━━━━┓
                cache_path = _find_cache(config["paths"]["output_root"], direction, m1=config['experiment']['m1'])
                if cache_path is None:
                    print(f"  [SKIP] No cache found for M1={config['experiment']['m1']} and direction={direction}")
                    continue

                # ┏━━━━━━━━━━ Build command for HPO ━━━━━━━━━━┓
                label = f"HPO {m2.upper()} {direction.upper()}"
                cmd = [python, "-m", "Utils.hpo",
                       "--config",     config_path,
                       "--cache",      cache_path,
                       "--models",     m2,
                       "--directions", direction,
                       "--grans",      *config['experiment']['granularity'],
                       "--n-trials",   str(config['runtime']['hpo']['n_trials']),
                       "--seed",       str(config['runtime']['hpo']['seed'])]
                ok = _run(cmd, label)
                results[label] = ok

    # ┏━━━━━━━━━━ Phase 1: Training (per-gran) ━━━━━━━━━━┓
    if not config["runtime"]['skip']["training"]:
        print(f"\n{'#' * 70}")
        print(f"# PHASE 1: Per-Granularity Training")
        print(f"# M1 Model: {config['experiment']['m1']}")
        print(f"# M2 Models: {config['experiment']['m2']}")
        print(f"# Directions: {config['experiment']['direction']}")
        print(f"# Granularities: {config['experiment']['granularity']}")
        print(f"{'#' * 70}")
        
        mode = config["data"]["load"]["meta_label_mode"]   # e.g. "tp"
        # Mirror the folder layout chosen inside m2_pipeline.py::run_analysis.
        _training_thres = config["runtime"]["training"].get("thres", "utility")
        
        # ┏━━━━━━━━━━ Determine thresholding folder based on mode ━━━━━━━━━━┓
        if _training_thres == "utility_nocal":
            thres_folder = "Utility_Score_NoCal"
        else:
            thres_folder = "Utility_Score"

        # ┏━━━━━━━━━━ Loop through M2 models ━━━━━━━━━━┓
        for m2 in config['experiment']['m2']:
            # ┏━━━━━━━━━━ Loop through directions ━━━━━━━━━━┓
            for direction in config['experiment']['direction']:
                # ┏━━━━━━━━━━ Loop through granularities ━━━━━━━━━━┓
                for granularity in config['experiment']['granularity']:
                    cache_path = _find_cache(config["paths"]["output_root"], direction, m1=config['experiment']['m1'])
                    
                    # ┏━━━━━━━━━━ Skip if no cache found ━━━━━━━━━━┓
                    if cache_path is None:
                        print(f"  [SKIP] No cache found for M1={config['experiment']['m1']} and direction={direction}")
                        continue

                    # ┏━━━━━━━━━━ Per-gran skip: analysis_summary.json exists → done ━━━━━━━━━━┓
                    train_dir = (Path(config["paths"]["output_root"])
                                 / config["experiment"]["m1"].capitalize()
                                 / m2 / direction.upper() / thres_folder
                                 / f"{granularity}_{mode}")
                    summary_file = train_dir / "analysis_summary.json"
                    if summary_file.exists():
                        print(f"  [SKIP] Training already done: {m2.upper()} {direction.upper()} {granularity} "
                              f"({summary_file})")
                        results[f"Train {m2.upper()} {direction.upper()} {granularity.upper()}"] = True
                        continue

                    label = f"Train {m2.upper()} {direction.upper()} {granularity.upper()}"
                    cmd = [python, "m2_pipeline.py",
                           "--cache_path", cache_path,
                           "--config", config_path,
                           "--phase", "training",
                           "--m2", m2,
                           "--direction", direction,
                           "--granularity", granularity]
                    ok = _run(cmd, label)
                    results[label] = ok
    
    # ┏━━━━━━━━━━ Phase 2: Cross-validation evidence (CPCV + expanding-window) ━━━━━━━━━━┓
    # Runs the two CV schemes whose per-sample outputs Phase 5 consumes:
    #   cpcv     -> cpcv_paths.csv     + cpcv_samples.csv
    #   cvexpand -> cvexpand_splits.csv + cvexpand_samples.csv
    # The old GREEN/AMBER/RED verdict gate is not consulted anywhere: it answered a
    # deployment question, whereas what the paper needs is the regime evidence built
    # in Phase 5. The seeds/convergence sub-steps are gone too: their model list had
    # been empty for a while, so they never executed.
    if not config["runtime"]["skip"]['edge']:
        # Threshold-selection mode chosen for the training phase also drives edge:
        # utility       → Edge/ folder (isotonic calibration in CPCV splits)
        # utility_nocal → Edge_NoCal/ folder (no calibration; raw τ* sweep)
        _thres_mode = config["runtime"]["training"].get("thres", "utility")
        _edge_nocal = (_thres_mode == "utility_nocal")
        _edge_folder = "Edge_NoCal" if _edge_nocal else "Edge"

        print(f"\n{'#' * 70}")
        print(f"# PHASE 2: Cross-Validation Evidence (CPCV + Expanding-Window)")
        print(f"# M1 Model: {config['experiment']['m1']}")
        print(f"# M2 Models: {config['experiment']['m2']}")
        print(f"# Directions: {config['experiment']['direction']}")
        print(f"# Granularities: {config['experiment']['granularity']}")
        print(f"# Blocks: {config['runtime']['edge']['n_blocks']}")
        print(f"# Threshold mode: {_thres_mode}  →  outputs in Analysis/{_edge_folder}/")
        print(f"{'#' * 70}")

        for m2 in config['experiment']['m2']:
            for direction in config['experiment']['direction']:
                for granularity in config['experiment']['granularity']:
                    cache_path = _find_cache(config["paths"]["output_root"], direction, m1=config['experiment']['m1'])
                    if cache_path is None:
                        print(f"  [SKIP] No cache for direction={direction}")
                        continue

                    edge_root = (Path(config["paths"]["output_root"])
                                 / "Analysis" / _edge_folder
                                 / config["experiment"]["m1"].capitalize()
                                 / m2)
                    gran_dir = edge_root / direction.upper() / granularity

                    # Sentinel is the per-SAMPLE csv, not the aggregate: an older run
                    # produced cpcv_paths.csv without cpcv_samples.csv, and Phase 5
                    # cannot work from the aggregate alone.
                    for mode, sentinel in (("cpcv", "cpcv_samples.csv"),
                                           ("cvexpand", "cvexpand_samples.csv")):
                        label = f"Edge {mode.upper()} {m2.upper()} {direction.upper()} {granularity}"
                        if (gran_dir / sentinel).exists():
                            print(f"  [SKIP] already done: {label}")
                            results[label] = True
                            continue
                        cmd = [python, "-m", "Utils.edge",
                               "--config", json.dumps(config),
                               "--cache_path", cache_path,
                               "--mode", mode,
                               "--phase", "edge",
                               "--m2", m2,
                               "--direction", direction,
                               "--granularity", granularity]
                        if _edge_nocal:
                            cmd.append("--nocal")
                        results[label] = _run(cmd, label)

    # ┏━━━━━━━━━━ Phase 3: Combined UP+DOWN Backtest ━━━━━━━━━━┓
    if not config["runtime"]["skip"]['combined']:
        # ┏━━━━━━━━━━ Print header ━━━━━━━━━━┓
        print(f"\n{'#' * 70}")
        print(f"# PHASE 3: Combined UP+DOWN Backtests")
        print(f"{'#' * 70}")
        
        output_root = Path(config["paths"]["output_root"]) / config["experiment"]["m1"].capitalize()
        
        for m2 in config['experiment']['m2']:
            for granularity in config['experiment']['granularity']:
                # ┏━━━━━━━━━━ Set combined backtest paths ━━━━━━━━━━┓
                _combined_thres_folder = "Utility_Score_NoCal" if config["runtime"]["training"].get("thres") == "utility_nocal" else "Utility_Score"
                config["runtime"]["combined"]["combined_backtest"][0] = str(output_root / m2 / "UP" / _combined_thres_folder)
                config["runtime"]["combined"]["combined_backtest"][1] = str(output_root / m2 / "DOWN" / _combined_thres_folder)
                
                # ┏━━━━━━━━━━ Check if combined backtest is already done ━━━━━━━━━━┓
                if not os.path.exists(config["runtime"]["combined"]["combined_backtest"][0]) or not \
                    os.path.exists(config["runtime"]["combined"]["combined_backtest"][1]):
                    up_path  = config["runtime"]["combined"]["combined_backtest"][0]
                    dn_path  = config["runtime"]["combined"]["combined_backtest"][1]
                    print(f"  [SKIP] Missing UP or DOWN results for {m2}: {up_path}, {dn_path}")
                    continue
                
                # ┏━━━━━━━━━━ Run combined backtest ━━━━━━━━━━┓
                label = f"Combined Backtest {m2.upper()} {granularity.upper()}"
                cmd = [python, "m2_pipeline.py",
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
                    cmd = [python, "Utils/feature_selection_experiment.py",
                           "--cache_path", cache_path,
                           "--config", config_path,
                           "--phase", "feature_selection",
                           "--m2", m2,
                           "--direction", direction,
                           "--granularity", granularity]
                    ok = _run(cmd, label)
                    results[label] = ok
    
    # ┏━━━━━━━━━━ Phase 5: 4-scenario regime analysis ━━━━━━━━━━┓
    # Consumes what Phases 1 and 2 wrote — the static test-period trades plus the
    # per-sample CV outputs — and joins every evaluated bar to its causal SJM regime
    # label. Pure read-side aggregation (no refitting), so it is cheap; the cost sits
    # entirely in having the jump models fitted beforehand.
    #
    # The four scenarios differ only in how bars are grouped before averaging, and that
    # grouping is the whole point: regime share correlates strongly with how good each
    # period was, so a pooled average measures WHEN a regime occurred rather than the
    # regime itself. See CLAUDE.md for the worked example.
    if not config["runtime"]["skip"].get("regime", False):
        print(f"\n{'#' * 70}")
        print(f"# PHASE 5: Regime Analysis (4 scenarios)")
        print(f"# M1 Model: {config['experiment']['m1']}")
        print(f"# M2 Models: {config['experiment']['m2']}")
        print(f"{'#' * 70}")

        _thres_mode = config["runtime"]["training"].get("thres", "utility")
        _edge_folder = "Edge_NoCal" if _thres_mode == "utility_nocal" else "Edge"
        _thres_folder = "Utility_Score_NoCal" if _thres_mode == "utility_nocal" else "Utility_Score"
        out_root = Path(config["paths"]["output_root"])
        m1_cap = config["experiment"]["m1"].capitalize()

        try:
            from Utils.analysis.analysis_m2 import (regime_scenarios, regime_split_table,
                                                    test_gap_by_quarter, m2_vs_m1_by_regime,
                                                    robust_significance_suite,
                                                    permutation_regime_sweep)
        except Exception as exc:
            print(f"  [SKIP] cannot import regime analysis: {exc}")
        else:
            # Each analysis resolves its own structured output dir
            # (Analysis/RegimeAnalysis/<M1>/<M2>/<NN_analysis_name>/).

            # Table A describes the data splits themselves, so it is M1/M2-independent.
            label = "Regime Table A (split composition)"
            try:
                tbl = regime_split_table(list(config["experiment"]["granularity"]),)
                print(f"  [OK] {label}: {len(tbl)} rows")
                results[label] = True
            except Exception as exc:
                print(f"  [FAIL] {label}: {exc}")
                results[label] = False

            for m2 in config["experiment"]["m2"]:
                label = f"Regime Scenarios {m2.upper()}"
                try:
                    df = regime_scenarios(
                        m1=m1_cap,
                        m2=m2,
                        directions=tuple(d.upper() for d in config["experiment"]["direction"]),
                        granularities=list(config["experiment"]["granularity"]),
                        fee=config["evaluation"]["fee_per_trade"],
                        edge_root=out_root / "Analysis" / _edge_folder,
                        test_root=out_root,
                        thres_folder=_thres_folder,
                        verbose=False)
                    if df.empty:
                        print(f"  [WARN] {label}: no rows (missing SJM fits or CV samples?)")
                        results[label] = False
                    else:
                        tally = df.groupby(["direction", "winner"]).size().to_dict()
                        print(f"  [OK] {label}: {len(df)} cells  {tally}")
                        results[label] = True
                except Exception as exc:
                    print(f"  [FAIL] {label}: {exc}")
                    results[label] = False

                # Significance for the regime effect: reviewers R5-W2 and R3 asked for
                # paired comparisons and statistical validation by name.
                # Tests B/C/E/G on the counter-trend regime gap. Run per direction:
                # pooling M2 UP and M2 DOWN dilutes whichever one carries the effect.
                label = f"Regime Significance {m2.upper()}"
                try:
                    parts = []
                    for _dir in config["experiment"]["direction"]:
                        parts.append(robust_significance_suite(
                            m1=m1_cap, m2=m2, direction=_dir.upper(),
                            granularities=list(config["experiment"]["granularity"]),
                            fee=config["evaluation"]["fee_per_trade"],
                            test_root=out_root, thres_folder=_thres_folder,
                            verbose=False))
                    sig = pd.concat([p for p in parts if len(p)], ignore_index=True) \
                          if any(len(p) for p in parts) else pd.DataFrame()
                    nsig = int((sig["p_value"] < 0.05).sum()) if "p_value" in sig else 0
                    print(f"  [OK] {label}: {len(sig)} tests ({nsig} with p<0.05)")
                    results[label] = True
                except Exception as exc:
                    print(f"  [FAIL] {label}: {exc}")
                    results[label] = False

                # Cross-validated significance: the composition-null permutation test
                # over every (granularity, direction, scheme) cell of the CV folds.
                # This is the train+val counterpart to the held-out suite above, and
                # the only test applied there -- a per-cell Wilcoxon on 5-6 folds
                # cannot resolve the effect and was removed rather than reported.
                label = f"Regime CV-Permutation {m2.upper()}"
                try:
                    cv = permutation_regime_sweep(
                        m1=m1_cap, m2=m2,
                        directions=tuple(d.upper() for d in config["experiment"]["direction"]),
                        granularities=list(config["experiment"]["granularity"]),
                        fee=config["evaluation"]["fee_per_trade"],
                        edge_root=out_root / "Analysis" / _edge_folder,
                        verbose=False)
                    nsig = int((cv["p_value"] < 0.05).sum()) if len(cv) else 0
                    print(f"  [OK] {label}: {len(cv)} cells ({nsig} with p<0.05)")
                    results[label] = True
                except Exception as exc:
                    print(f"  [FAIL] {label}: {exc}")
                    results[label] = False

                # Held-out gap per quarter: is the effect spread over the test window
                # or carried by one quarter? Descriptive, not a second significance test.
                label = f"Regime TestGap-by-Quarter {m2.upper()}"
                try:
                    q = test_gap_by_quarter(
                        m1=m1_cap, m2=m2,
                        directions=tuple(d.upper() for d in config["experiment"]["direction"]),
                        granularities=list(config["experiment"]["granularity"]),
                        fee=config["evaluation"]["fee_per_trade"],
                        test_root=out_root, thres_folder=_thres_folder,
                        verbose=False)
                    print(f"  [OK] {label}: {len(q)} rows")
                    results[label] = True
                except Exception as exc:
                    print(f"  [FAIL] {label}: {exc}")
                    results[label] = False

                # M2's gain over M1 within each regime — the pooled version is diluted
                # by an ~84% bearish test window.
                label = f"Regime M2-vs-M1 {m2.upper()}"
                try:
                    v = m2_vs_m1_by_regime(
                        m1=m1_cap, m2=m2,
                        directions=tuple(d.upper() for d in config["experiment"]["direction"]),
                        granularities=list(config["experiment"]["granularity"]),
                        fee=config["evaluation"]["fee_per_trade"],
                        test_root=out_root, thres_folder=_thres_folder,
                        verbose=False)
                    nsig = int(v["excludes_zero"].sum()) if len(v) else 0
                    print(f"  [OK] {label}: {len(v)} rows ({nsig} with CI excluding zero)")
                    results[label] = True
                except Exception as exc:
                    print(f"  [FAIL] {label}: {exc}")
                    results[label] = False
                    v = None

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

    # ┏━━━━━━━━━━ Config validation ━━━━━━━━━━┓
    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    
    # ┏━━━━━━━━━━ Load config ━━━━━━━━━━┓
    with config_path.open("r") as f:
        cfg = yaml.safe_load(f) or {}
    
    # ┏━━━━━━━━━━ Print config ━━━━━━━━━━┓
    print(f"\n{'=' * 70}")
    print(f"EXPERIMENT CONFIG")
    print(yaml.dump(cfg, sort_keys=False))
    print(f"{'=' * 70}")
    
    # ┏━━━━━━━━━━ Run experiments ━━━━━━━━━━┓
    run_experiments(config=cfg, config_path=str(config_path))


if __name__ == "__main__":
    main()
