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
   - AutoGluon preset names and contents shift between releases, and
     `requirements.txt` pins `autogluon.tabular` **unversioned**. Verified locally
     2026-10-26 by actually installing it: you get **1.6.3**, not the `1.5.0` this was
     originally researched against, and the old `PRESETS_CONFIGS` import doesn't exist
     in 1.6.x (replaced by `tabular_presets_dict`) — the trial script handles both.
     **Correction: `extreme_quality` is NOT an equivalent fallback for Mitra.** Verified
     by importing the actual hyperparameter config in 1.6.3: `tabarena`
     (`zeroshot_2025_tabfm`) includes `REALTABPFN-V2, TABM, TABICL, GBM, CAT, XGB, MITRA`.
     `extreme_quality` (`commercial_2026_08_05`) is `NORI, TABICL, TABDPT-TURBO, GBM, CAT,
     REALMLP` — **no Mitra, no TabPFN at all**. `noncommercial` gets `TABPFN-3` but still
     no Mitra. If `tabarena` isn't available on whatever AutoGluon version you end up
     with, there is no drop-in replacement that still reaches Mitra — the trial script's
     preset check prints each preset's actual model-family list so you can see this
     directly rather than trusting this note after the next AutoGluon release moves
     things again.
   - **`kumo` (`kumo_tabular_classifier.py`) — now actually run locally (CPU, small-scale,
     see "Local validation" below) and confirmed working end-to-end**, including a real
     backtest. Two real bugs from the initial NVIDIA-ported draft were found and fixed in
     the process: integer column names broke `sdm.infer_stypes` (needs strings), and
     `out.numerical_columns` (from NVIDIA's own AutoGluon glue) doesn't exist on this
     installed `sdm==0.2.1` — `out.to_pandas()` does, and is simpler. Still not run on
     GPU or at the full dataset scale (all 20 assets × both directions), so don't treat
     this as fully validated — just no longer "read, not run."

## Local validation (2026-10-26)

Ran the actual pipeline locally (CPU, macOS, Python 3.11 venv + `pip install -r
requirements.txt` — a conda-equivalent substitute for smoke-testing, not a cluster run) to
check whether these scripts would actually work before spending real cluster time on them.
They mostly didn't, at first — six real, concrete bugs found and fixed along the way:

1. `config.yaml` had **hardcoded `/home/pablo/M2_DS/Secondary-Model/...` absolute paths**
   for `csv_dir`/`output_root` — nothing could run from any location but Pablo's exact
   gpu8 checkout. Fixed to relative paths (`Data_MLA/...`, `Output`).
2. `xgboost` segfaulted — the classic macOS "duplicate OpenMP runtime" crash (`torch` and
   `xgboost` each bundle their own `libomp`). Fixed with `KMP_DUPLICATE_LIB_OK=TRUE`,
   added to both scripts; harmless on Linux where this rarely triggers.
3. AutoGluon's old `PRESETS_CONFIGS` import is gone in the actually-installed 1.6.3
   (`requirements.txt` pins it unversioned). Fixed the trial script's check to handle
   both the old and new API — and in the process **found the `extreme_quality`-reaches-
   Mitra claim in this README was wrong**; corrected above.
4. `tabpfn` now requires an explicit `TABPFN_TOKEN` (license acceptance from
   ux.priorlabs.ai) — it tries a browser login, which fails outright on a headless node.
   Added a precondition check to both scripts. **Still blocked**: nobody has run `tabpfn`
   end-to-end in this validation pass without one.
5. `autogluon --ag-presets tabarena` crashed on CPU here, but AutoGluon itself warned the
   Mitra/TabPFNv2 base learner "attends over the training context for every prediction,"
   is "typically an order of magnitude slower" on CPU, and "may appear to hang" — **this
   one is genuinely inconclusive locally**, not confirmed broken, since there's no GPU on
   this machine to test the path AutoGluon actually expects.
6. `kumo_tabular_classifier.py` had two real bugs (see above) — both found by actually
   running it, both fixed, now confirmed working end-to-end including a real backtest.

**What actually ran clean, on the real data (fit20 `up`/`1d` slice, single run, no
significance testing — do not over-read these numbers, same caveat as everywhere else in
this thesis):**

| Model | Total Ret | MaxDD | Sharpe |
|---|---|---|---|
| M2 xgboost | +5.00% | -12.45% | 0.85 |
| M2 kumo | +14.21% | -14.94% | 1.42 |
| M1 Kronos (baseline, all trades) | +1.80% | -13.57% | 0.56 |
| Buy & Hold | -36.21% | -42.17% | -2.68 |

Kumo's number is the most eye-catching thing in this PR and the least justified to
repeat anywhere — it's one run, one direction, one granularity, on CPU, with the default
`size="large"`. Confirms the wrapper works, nothing about whether kumo is actually better.

**Not re-triggered after the first attempt crashed:** `_resolve_caches` builds both
`up`/`down` caches on first run regardless of which `--direction` you asked for, and
building the `down` cache (all 7 granularities, ~1M rows) raw-segfaulted (no Python
traceback) partway through — most likely memory pressure from building the full
multi-granularity dataset twice on a resource-constrained laptop, not something SLURM's
requested 16-32GB should hit, but genuinely unconfirmed. If the trial job dies here with
no traceback, that's the first thing to suspect; `--cache_path` (pointing at an
already-built cache) skips this entirely if you need to isolate it.

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
