# M2 model comparison — EPFL SLURM jobs

Compares four M2 backends on the fit20 cohort (20 assets, both UP/DOWN directions):
`xgboost` (GBM baseline), `tabpfn`, `autogluon` configured to reach **Mitra** (and
TabPFNv2/TabICL) as base learners, and `kumo` (NVIDIA Kumo Tabular). Same features, same
splits, same target for every model — only the model family changes, so the comparison is
actually fair.

`xgboost`/`tabpfn`/`autogluon` were already wired into `src/Utils/classifier/factory.py` as
`--m2` choices before this PR. `kumo` is new (`kumo_tabular_classifier.py` + a `factory.py`
registration). The other addition: `--ag-presets`/`--ag-time-limit` CLI flags on
`m2_pipeline.py`, already documented in `factory.py`'s `_AG_PRESETS`/`_AG_TIME_LIMIT`
comments ("overridden by --ag-presets") but never actually implemented. Without them,
`autogluon` was hardcoded to `best_quality`, which uses AutoGluon's `"zeroshot"`
hyperparameter set — **that excludes every foundation model, including Mitra**. Both flags
are optional and default to the old hardcoded values, so every existing invocation
(Pablo's included) behaves identically unless you pass them. See the diffs on
`src/m2_pipeline.py` / `src/Utils/classifier/factory.py` — small and additive.

## Environment note — conda, not uv

This branch is based on **`upstream/main`**, not `elie/sota`. That matters here
specifically: `elie/sota` has a later commit migrating the repo to `uv` + `pyproject.toml`
+ `mise`, but that commit isn't part of `upstream/main` yet, so **this PR's base still uses
conda + `requirements.txt`**, exactly as README.md's own "Setup" section documents
(`conda create -n S2 python=3.11`, `pip install -r requirements.txt`). Both SLURM scripts
below use conda accordingly — don't "fix" them to `uv sync` without separately bringing
that migration commit into this PR first (that's a bigger, separate scope decision).

## Before you run anything

1. **EPFL cluster access.** These scripts use generic `#SBATCH --partition=gpu` /
   `--qos=normal|debug` directives, and try `module load miniconda3` / `anaconda3` for
   conda (common names, not guaranteed). Once you know your cluster's actual
   partition/qos/module names (SCITAS Jed/Izar, RCP, etc.), edit accordingly.

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

3. **Don't trust "TabPFN-2.5", "Mitra", or "kumo" until the trial job confirms them.**
   Three things are genuinely unverified until you run on the actual cluster:
   - `tabpfn==7.1.0` is pinned in `requirements.txt`. **Correction (2026-10-26):** per
     `PriorLabs/TabPFN`'s current README, a bare `TabPFNClassifier()` now defaults to
     **TabPFN-3.5**, not 2.5 — "To use our default TabPFN-3.5 model: `TabPFNClassifier()`".
     TabPFN-3.5's weights are non-commercial-licensed (fine for a thesis), and it raises
     the CPU-only row cap to 5,000 (vs 1,000 on older versions) — comfortably covers the
     fit20 cohort's DOWN direction (~5,868 rows) with margin to spare, UP (~2,002) easily.
     This repo's `tabpfn_classifier.py` wrapper doesn't pin a `ModelVersion`, so it almost
     certainly already gets 3.5 — **strictly newer than what we assumed when deciding on
     "TabPFN-2.5."** The trial script still prints `tabpfn.__version__` and the
     `TabPFNClassifier` signature — confirm which generation you actually got before
     citing a number, don't assume this note is still accurate by the time you read it.
   - AutoGluon preset names shift between releases, and `requirements.txt` pins
     `autogluon.tabular` **unversioned** (unlike `elie/sota`'s `pyproject.toml`, which pins
     `==1.5.0` — this branch doesn't have that pin, so you may get a different release).
     The trial script checks that `AG_PRESET` (default `tabarena`) actually exists in
     whatever got installed and prints what it maps to. If it's missing, try
     `extreme_quality` instead (`sbatch slurm/m2-compare-trial.sh --export=AG_PRESET=extreme_quality`)
     — both reach Mitra/TabPFNv2/TabICL; `best_quality` (the old default) does not.
   - **`kumo` (`kumo_tabular_classifier.py`) has never actually been run.** It was written
     by reading NVIDIA's own AutoGluon-integration source
     (`NVIDIA/structured-data-models`'s `benchmark/tabular/model.py`,
     `SDMKumoTabularModel._fit`/`._predict_proba`) to get the `sdm.TableTensor`/
     `model.fit(x=..., y=..., recipe=...)`/`model.predict(...)` call shape right, not by
     installing and exercising the package (no local GPU here). The trial script's step 6
     checks the import and prints `sdm.models.KumoTabular`; the real test is whether the
     fit/predict call in the pipeline itself succeeds and produces sane probabilities. If
     `kumo`'s precision/coverage numbers in the trial log look wrong, the probability
     reconstruction in `predict_proba` (around the `out.numerical_columns` indexing) is
     the first place to look — it has no `_convert_proba_to_unified_form` equivalent to
     fall back on, just a defensive renormalisation.

## Usage

```bash
# 1. Trial: one direction (up), short AutoGluon time budget, fails fast on bad config.
sbatch slurm/m2-compare-trial.sh

# Read the trial's .out log — specifically the version/preset checks near the top,
# the kumo import check, and whether all four models fit without error — before
# doing anything else.

# 2. Full: both directions, full AutoGluon time budget (3600s/fit by default).
sbatch slurm/m2-compare-full.sh

# Override granularity, AutoGluon settings, or the conda env name via --export:
sbatch slurm/m2-compare-full.sh --export=GRANULARITY=4h
sbatch slurm/m2-compare-full.sh --export=AG_PRESET=extreme_quality,AG_TIME_LIMIT=7200
sbatch slurm/m2-compare-full.sh --export=CONDA_ENV=my_existing_env
```

Both scripts assume they're submitted from the `SecondaryModel/` repo root (the
`PROJECT_DIR` resolution falls back to `$SLURM_SUBMIT_DIR`, matching `sbatch`'s default).
`CONDA_ENV` (default `S2_m2compare`) is created once by whichever script runs first and
reused by the other — run the trial first so the full job doesn't pay the install cost cold.

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
  `factory.py` if you want to add it to `MODELS=(...)` in either script.
- **Granularity defaults to `1d`** (matches the fit20 cohort's scope from the Prendinger
  brief). Nothing stops you from sweeping other granularities, but each one is a separate
  config/data signature — don't assume a `1d` result generalizes.
- **Kumo Tabular's `size` is fixed at `"large"` (215M params) by default.** `small`/
  `medium` are cheaper and may generalise better on a dataset this small (fit20 is
  thousands of rows, not the millions Kumo's large size was benchmarked on) — worth a
  sweep once the default run is confirmed to work at all, via
  `_load_best_params`'s HPO path or a manual `params={"size": "small"}` override in
  `factory.py`'s kumo branch.
