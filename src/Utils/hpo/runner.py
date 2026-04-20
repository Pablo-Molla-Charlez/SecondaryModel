"""HPO orchestrator + CLI entrypoint."""
import argparse
import json
import time
import optuna
from pathlib import Path

# ┏━━━━━━━━━━ Utils ━━━━━━━━━━┓
from Utils.utils import (_load_config,
                         _resolve_caches,
                         _load_multi_cache,
                         _infer_direction,
                         m1_output_bucket)

from Utils.hpo.objectives import (_load_dataset_for_gran,
                                  _prepare_splits,
                                  _create_objective)

# ┏━━━━━━━━━━ Constants ━━━━━━━━━━┓
ALL_GRANS  = ["1d", "12h", "8h", "6h", "4h", "2h", "1h", "30m"]
DIRECTIONS = ["up", "down"]
HPO_MODELS = ["rf", "tabpfn", "tabicl"]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Run HPO for one (model, direction, granularity) configuration
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def run_hpo_single(model_name: str,
                   direction: str,
                   granularity: str,
                   cfg: dict,
                   multi_cache,
                   output_root: Path,
                   n_trials: int = 100,
                   seed: int = 42) -> dict | None:
    """Run HPO for a single (model, direction, granularity) configuration."""
    # ┏━━━━━━━━━━ Output directory ━━━━━━━━━━┓
    m1_bucket = m1_output_bucket(cfg)
    out_dir = output_root / m1_bucket / "HPO" / model_name / direction.upper() / granularity
    out_dir.mkdir(parents=True, exist_ok=True)

    # ┏━━━━━━━━━━ Skip if already completed ━━━━━━━━━━┓
    best_path = out_dir / "best_params.json"
    if best_path.exists():
        print(f"  [SKIP] {model_name.upper()} {direction.upper()} {granularity} — best_params.json exists")
        with open(best_path) as f:
            return json.load(f)

    print(f"\n{'='*60}")
    print(f"  HPO: {model_name.upper()} | {direction.upper()} | {granularity}")
    print(f"  Output: {out_dir}")
    print(f"{'='*60}")

    # ┏━━━━━━━━━━ Load single-granularity dataset ━━━━━━━━━━┓
    try:
        dataset = _load_dataset_for_gran(multi_cache, granularity)
    except ValueError as e:
        print(f"  [SKIP] {e}")
        return None

    # ┏━━━━━━━━━━ Prepare splits ━━━━━━━━━━┓
    try:
        (X_train, y_train,
         X_cal, y_cal,
         X_opt, y_opt,
         opt_returns,
         feature_names,
         fee) = _prepare_splits(dataset, cfg, granularity, direction)
    except Exception as e:
        print(f"  [SKIP] Split preparation failed: {e}")
        return None

    print(f"  Train: {len(y_train):,}  Cal: {len(y_cal):,}  Opt: {len(y_opt):,}")
    print(f"  Fee: {fee}  Direction: {direction}")

    if len(y_train) < 100 or len(y_cal) < 20 or len(y_opt) < 50:
        print(f"  [SKIP] Insufficient data for HPO")
        return None

    # ┏━━━━━━━━━━ Create Optuna study ━━━━━━━━━━┓
    db_path = out_dir / "optuna_study.db"
    study_name = f"HPO_{model_name}_{direction}_{granularity}"
    storage = f"sqlite:///{db_path}"

    study = optuna.create_study(study_name     = study_name,
                                storage        = storage,
                                direction      = "maximize",
                                load_if_exists = True,
                                sampler        = optuna.samplers.TPESampler(seed=seed),
                                pruner         = optuna.pruners.NopPruner())

    # ┏━━━━━━━━━━ Create objective ━━━━━━━━━━┓
    objective = _create_objective(model_name  = model_name,
                                  X_train     = X_train,
                                  y_train     = y_train,
                                  X_cal       = X_cal,
                                  y_cal       = y_cal,
                                  X_opt       = X_opt,
                                  y_opt       = y_opt,
                                  opt_returns = opt_returns,
                                  fee         = fee,
                                  seed        = seed)

    # ┏━━━━━━━━━━ Remaining trials ━━━━━━━━━━┓
    completed = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    remaining = max(0, n_trials - completed)
    if remaining == 0:
        print(f"  Already completed {completed} trials — loading best.")
    else:
        print(f"  Running {remaining} trials ({completed} already completed)...")
        t0 = time.time()
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study.optimize(objective, n_trials=remaining, show_progress_bar=True)
        elapsed = time.time() - t0
        print(f"  Completed in {elapsed:.0f}s")

    # ┏━━━━━━━━━━ Extract best ━━━━━━━━━━┓
    best = study.best_trial
    result = {"model":       model_name,
              "direction":   direction,
              "granularity": granularity,
              "best_trial":  best.number,
              "best_utility": best.value,
              "best_params": best.params,
              "best_metrics": {k: v for k, v in best.user_attrs.items()},
              "n_trials":    len(study.trials)}

    # ┏━━━━━━━━━━ Save best params ━━━━━━━━━━┓
    with open(best_path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"  Best utility: {best.value:.4f}")
    print(f"  Best params: {best.params}")
    print(f"  Saved: {best_path}")

    # ┏━━━━━━━━━━ Save study history ━━━━━━━━━━┓
    history_path = out_dir / "study_history.csv"
    df = study.trials_dataframe()
    df.to_csv(history_path, index=False)

    return result


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Orchestrator: run HPO for all combinations
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def run_hpo(config: str,
            models: list[str],
            directions: list[str],
            grans: list[str],
            n_trials: int = 100,
            seed: int = 42,
            cache: str | None = None):
    """Run HPO for all (model x direction x granularity) combinations."""
    # ┏━━━━━━━━━━ Load config ━━━━━━━━━━┓
    cfg = _load_config(config)
    output_root = Path(cfg["paths"]["output_root"])

    # ┏━━━━━━━━━━ Print header ━━━━━━━━━━┓
    print(f"\n{'#'*70}")
    print(f"# M2 Hyperparameter Optimization")
    print(f"# Models:     {models}")
    print(f"# Directions: {directions}")
    print(f"# Grans:      {grans}")
    print(f"# Trials:     {n_trials}")
    print(f"# Config:     {config}")
    if cache:
        print(f"# Cache:      {cache}")
    print(f"{'#'*70}")

    # ┏━━━━━━━━━━ Load caches per direction ━━━━━━━━━━┓
    caches = {}
    if cache:
        cache_path = Path(cache)
        if cache_path.is_dir():
            # ┏━━━━━━━━━━ Directory mode ━━━━━━━━━━┓
            for direction in directions:
                candidates = sorted(cache_path.glob(f"*_{direction}_*.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
                if candidates:
                    print(f"\n  Loading multi-cache for {direction.upper()}: {candidates[0].name}")
                    caches[direction] = _load_multi_cache(candidates[0])
                else:
                    print(f"  [WARN] No {direction} cache found in {cache_path}")
        elif cache_path.is_file():
            # ┏━━━━━━━━━━ Single file mode: infer direction from filename ━━━━━━━━━━┓
            direction = _infer_direction(cache_path)
            print(f"\n  Loading multi-cache for {direction.upper()}: {cache_path.name}")
            caches[direction] = _load_multi_cache(cache_path)
        else:
            raise FileNotFoundError(f"Cache path not found: {cache_path}")
    else:
        for direction in directions:
            cache_map = _resolve_caches(cfg, explicit=None)
            if direction in cache_map:
                print(f"\n  Loading multi-cache for {direction.upper()}: {cache_map[direction].name}")
                caches[direction] = _load_multi_cache(cache_map[direction])
            else:
                print(f"  [WARN] No cache found for direction={direction}")

    # ┏━━━━━━━━━━ Run HPO for each combination ━━━━━━━━━━┓
    all_results = []
    total = len(models) * len(directions) * len(grans)
    i = 0

    # ┏━━━━━━━━━━ Run HPO for each combination ━━━━━━━━━━┓
    for model_name in models:
        for direction in directions:
            if direction not in caches:
                print(f"  [SKIP] No cache for direction={direction}")
                continue

            # ┏━━━━━━━━━━ Get cache for this direction ━━━━━━━━━━┓
            multi_cache = caches[direction]
            available_grans = list(multi_cache.sub.keys()) if hasattr(multi_cache, "sub") else []

            # ┏━━━━━━━━━━ Run HPO for each granularity ━━━━━━━━━━┓
            for gran in grans:
                i += 1
                print(f"\n  [{i}/{total}] {model_name.upper()} {direction.upper()} {gran}")

                if gran not in available_grans:
                    print(f"    [SKIP] Granularity {gran} not in cache (available: {available_grans})")
                    continue

                # ┏━━━━━━━━━━ Run HPO for single combination ━━━━━━━━━━┓
                result = run_hpo_single(model_name  = model_name,
                                        direction   = direction,
                                        granularity = gran,
                                        cfg         = cfg,
                                        multi_cache = multi_cache,
                                        output_root = output_root,
                                        n_trials    = n_trials,
                                        seed        = seed)

                if result:
                    all_results.append(result)

    # ┏━━━━━━━━━━ Summary ━━━━━━━━━━┓
    print(f"\n{'='*70}")
    print(f"HPO SUMMARY")
    print(f"{'='*70}")
    print(f"{'Model':<10} {'Dir':<6} {'Gran':<6} {'Utility':>10} {'Prec':>8} {'Cov':>8} {'Source':<18} {'Thr':>8}")
    print(f"{'-'*78}")
    for r in all_results:
        m = r["best_metrics"]
        print(f"{r['model']:<10} {r['direction']:<6} {r['granularity']:<6} "
              f"{r['best_utility']:>10.4f} "
              f"{m.get('sel_precision', 0):>8.3f} "
              f"{m.get('coverage', 0):>8.3f} "
              f"{m.get('threshold_source', 'unknown'):<18} "
              f"{m.get('threshold', 0.5):>8.3f}")
    print(f"{'='*70}")
    print(f"Completed {len(all_results)}/{total} configurations.")

    # ┏━━━━━━━━━━ Save global summary ━━━━━━━━━━┓
    m1_bucket = m1_output_bucket(cfg)
    summary_path = output_root / m1_bucket / "HPO" / "hpo_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"Summary saved: {summary_path}")

    return all_results


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CLI
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def main():
    # ┏━━━━━━━━━━ Parse arguments ━━━━━━━━━━┓
    parser = argparse.ArgumentParser(description="M2 Hyperparameter Optimization (RF, TabPFN, TabICL)")

    parser.add_argument("--config",     type=str,  required=True, help="Path to config YAML (e.g. config.yaml)")
    parser.add_argument("--models",     nargs="+", default=HPO_MODELS, choices=HPO_MODELS, help=f"Models to optimize (default: {HPO_MODELS})")
    parser.add_argument("--directions", nargs="+", default=DIRECTIONS,choices=DIRECTIONS, help="Directions to optimize (default: up down)")
    parser.add_argument("--grans",      nargs="+", default=ALL_GRANS, help=f"Granularities to optimize (default: {ALL_GRANS})")
    parser.add_argument("--n-trials",   type=int,  default=100, help="Number of Optuna trials per configuration (default: 100)")
    parser.add_argument("--seed",       type=int,  default=42, help="Random seed (default: 42)")
    parser.add_argument("--cache",      type=str,  default=None, help="Path to cache .pt file or directory containing up/down caches (skips auto-detection / rebuilding)")

    args = parser.parse_args()

    # ┏━━━━━━━━━━ Run HPO ━━━━━━━━━━┓
    run_hpo(config     = args.config,
            models     = args.models,
            directions = args.directions,
            grans      = args.grans,
            n_trials   = args.n_trials,
            seed       = args.seed,
            cache      = args.cache)


if __name__ == "__main__":
    main()
