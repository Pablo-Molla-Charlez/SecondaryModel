# M2 model comparison — EPFL SLURM jobs

Compares three M2 backends on the fit20 cohort (20 assets, both UP/DOWN directions):
`xgboost` (GBM baseline), `tabpfn` (TabPFN), and `autogluon` configured to reach **Mitra**
(and TabPFNv2/TabICL) as base learners. Same features, same splits, same target for every
model — only the model family changes, so the comparison is actually fair.

All three backends were already wired into `src/Utils/classifier/factory.py` as `--m2`
choices before this change. The one addition made for this comparison: `--ag-presets` /
`--ag-time-limit` CLI flags on `m2_pipeline.py`, which were already documented in
`factory.py`'s `_AG_PRESETS`/`_AG_TIME_LIMIT` comments ("overridden by --ag-presets") but
never actually implemented. Without them, `autogluon` was hardcoded to `best_quality`,
which uses AutoGluon's `"zeroshot"` hyperparameter set — **that excludes every foundation
model, including Mitra**. The flags are optional and default to the old hardcoded values,
so every existing invocation (Pablo's included) behaves identically unless you pass them.
See the diff on `src/m2_pipeline.py` — it's small and additive.

## Before you run anything

1. **EPFL cluster access.** These scripts use generic `#SBATCH --partition=gpu` /
   `--qos=normal|debug` directives. Once you know the actual partition/qos names on your
   cluster (SCITAS Jed/Izar, RCP, etc.), edit the `#SBATCH` headers accordingly.

2. **Stage the training data — two hops.** `Data_MLA/Kronos/Crypto/TP/horizon_7/`
   (referenced by `config.yaml`'s `paths.csv_dir`) is gitignored and lives on **gpu8**
   (`/home/pablo/M2_DS/SecondaryModel/src/Data_MLA/Kronos/Crypto/TP/horizon_7/`, confirmed
   2026-10-26 — **not gpu7**, which turned out to be a freshly provisioned machine with
   nothing staged on it yet).
   - **Hop 1 (done):** pulled onto this machine with `slurm/stage-data.sh`
     (341M, 280 files, via the `gpu8-pablo` SSH alias already in `~/.ssh/config`).
   - **Hop 2 (you do this once you have cluster access):** EPFL compute/login nodes can't
     reach `gpu8-pablo` with your laptop's keys, so rsync your now-local
     `src/Data_MLA/Kronos/Crypto/TP/horizon_7/` onto EPFL scratch storage, e.g.:
     ```bash
     rsync -avz src/Data_MLA/Kronos/Crypto/TP/horizon_7/ \
       <your-epfl-cluster-login>:<project-path>/SecondaryModel/src/Data_MLA/Kronos/Crypto/TP/horizon_7/
     ```
   Both scripts check for the data on the cluster side and exit immediately with
   instructions if it's missing. Everything downstream (cache building, in
   `_resolve_caches`) runs automatically once the raw CSVs are there.

3. **Don't trust "TabPFN-2.5" or "Mitra" until the trial job confirms it.** Two things
   are genuinely unverified until you run on the actual cluster:
   - `tabpfn==7.1.0` is pinned in `pyproject.toml`, but the `tabpfn` PyPI package's
     version number does not map 1:1 onto Prior Labs' "2.5" marketing name. The trial
     script prints `tabpfn.__version__` and the `TabPFNClassifier` signature — check it
     matches what you expect before trusting a full run's numbers.
   - AutoGluon preset names shift between releases. The trial script checks that
     `AG_PRESET` (default `tabarena`) actually exists in the pinned AutoGluon 1.5.0 and
     prints what it maps to. If it's missing, try `extreme_quality` instead
     (`sbatch slurm/m2-compare-trial.sh --export=AG_PRESET=extreme_quality`) — both
     reach Mitra/TabPFNv2/TabICL; `best_quality` (the old default) does not.

## Usage

```bash
# 1. Trial: one direction (up), short AutoGluon time budget, fails fast on bad config.
sbatch slurm/m2-compare-trial.sh

# Read the trial's .out log — specifically the version/preset checks near the top
# and whether all three models fit without error — before doing anything else.

# 2. Full: both directions, full AutoGluon time budget (3600s/fit by default).
sbatch slurm/m2-compare-full.sh

# Override granularity or AutoGluon settings via --export:
sbatch slurm/m2-compare-full.sh --export=GRANULARITY=4h
sbatch slurm/m2-compare-full.sh --export=AG_PRESET=extreme_quality,AG_TIME_LIMIT=7200
```

Both scripts assume they're submitted from the `SecondaryModel/` repo root (the
`PROJECT_DIR` resolution falls back to `$SLURM_SUBMIT_DIR`, matching `sbatch`'s default).

## Reading results

Each `(model, direction)` slice writes to
`src/Output/Kronos/<model>/<UP|DOWN>/Utility_Score*/<granularity>_tp/analysis_summary.json`
— the same layout `run_analysis()` always uses, so results sit next to anything Pablo's
own runs have already produced for `rf`/other backbones. The full script's final section
lists the summaries it produced. Per-model/direction stdout is also saved under
`slurm/logs/{trial,full}-<job-id>/`.

## What this does *not* cover

- **xgboost ≠ LightGBM.** The earlier thesis-level TBM diagnostics (in
  `nii-thesis/report/triple_barrier/analysis/benchmark.py`) used LightGBM as the GBM
  baseline. This repo's standalone-tree baseline is XGBoost (`rf` is a random forest, not
  a boosted-tree baseline) — close enough as "a tuned GBM" for the comparison's purpose,
  but don't cite it as "the same LightGBM baseline from the TBM deck" without saying so.
- **No `tabicl` comparison yet.** `tabicl` is also a valid `--m2` choice already wired in
  `factory.py` if you want to add it to `MODELS=(...)` in either script — it wasn't
  included here to keep the first comparison to three models, per the earlier discussion.
- **Granularity defaults to `1d`** (matches the fit20 cohort's scope from the Prendinger
  brief). Nothing stops you from sweeping other granularities, but each one is a separate
  config/data signature — don't assume a `1d` result generalizes.
