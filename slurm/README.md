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

## Environment — uv, not conda

This branch now carries the `uv` + `pyproject.toml` + `mise` migration (ported from
`elie/sota`'s tooling commit, adapted to this PR's actual module set — the SPCI/regime-only
dependencies that migration also pinned aren't here, since those source files live only on
`elie/sota`). Both SLURM scripts use `uv sync --locked` + `uv run`, matching README.md's
"Setup" section. Neither script assumes `uv` is preinstalled on the compute node: each
tries `module load uv` first, falls back to the official installer
(`curl -LsSf https://astral.sh/uv/install.sh | sh`) if that's not available either, and
exits with a clear message only if both fail.

## Before you run anything

1. **EPFL cluster access.** These scripts use generic `#SBATCH --partition=gpu` /
   `--qos=normal|debug` directives, and try `module load uv` (common name, not
   guaranteed — falls back to installing uv directly if the module doesn't exist). Once
   you know your cluster's actual partition/qos/module names (SCITAS Jed/Izar, RCP, etc.),
   edit accordingly.

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
   - `tabpfn==7.1.0` is pinned in `pyproject.toml`. **Correction (2026-10-26):** per
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
   - AutoGluon preset names and contents shift between releases. `pyproject.toml` now
     pins `autogluon-tabular==1.5.0` (matching the CTTS training env — a trained bundle
     unpickles against it), but the old, unpinned `requirements.txt` resolved to **1.6.3**
     during earlier local testing, where the old `PRESETS_CONFIGS` import doesn't exist
     (replaced by `tabular_presets_dict`) — the trial script handles both either way.
     **Correction: `extreme_quality` is NOT an equivalent fallback for Mitra.** Verified
     by importing the actual hyperparameter config in 1.6.3: `tabarena`
     (`zeroshot_2025_tabfm`) includes `REALTABPFN-V2, TABM, TABICL, GBM, CAT, XGB, MITRA`.
     `extreme_quality` (`commercial_2026_08_05`) is `NORI, TABICL, TABDPT-TURBO, GBM, CAT,
     REALMLP` — **no Mitra, no TabPFN at all**. `noncommercial` gets `TABPFN-3` but still
     no Mitra. If `tabarena` isn't available on whatever AutoGluon version you end up
     with, there is no drop-in replacement that still reaches Mitra — the trial script's
     preset check prints each preset's actual model-family list so you can see this
     directly rather than trusting this note after the next AutoGluon release moves
     things again. **Update (2026-10-26, full trial run):** `tabarena` was previously
     flagged "inconclusive" on CPU after a standalone attempt crashed. Running it through
     the actual trial script end-to-end (10-minute `--ag-time-limit`, real fit20 data) now
     completed cleanly with a real backtest (Sharpe 0.86) — the earlier crash doesn't
     reproduce under these conditions. Still worth re-confirming on the actual GPU cluster
     (AutoGluon itself still warns Mitra/TabPFNv2 is much slower on CPU), but this is no
     longer a known blocker.
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
   Added a precondition check to both scripts. **Unblocked (2026-10-26, see below)**:
   now confirmed running end-to-end with a real token.
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
| M2 autogluon (tabarena/Mitra, 600s budget) | +4.59% | -13.11% | 0.86 |
| M2 tabpfn | +7.61% | -12.48% | 1.23 |
| M2 kumo | +14.21% | -14.94% | 1.42 |
| M1 Kronos (baseline, all trades) | +1.80% | -13.57% | 0.56 |
| Buy & Hold | -36.21% | -42.17% | -2.68 |

Kumo's number is the most eye-catching thing in this PR and the least justified to
repeat anywhere — it's one run, one direction, one granularity, on CPU, with the default
`size="large"`. Confirms the wrapper works, nothing about whether kumo is actually better.
**All four models now have at least one confirmed clean end-to-end run** (see "TabPFN
unblocked" below for how the last one, `tabpfn`, got there).

**Not re-triggered after the first attempt crashed:** `_resolve_caches` builds both
`up`/`down` caches on first run regardless of which `--direction` you asked for, and
building the `down` cache (all 7 granularities, ~1M rows) raw-segfaulted (no Python
traceback) partway through — most likely memory pressure from building the full
multi-granularity dataset twice on a resource-constrained laptop, not something SLURM's
requested 16-32GB should hit, but genuinely unconfirmed. If the trial job dies here with
no traceback, that's the first thing to suspect; `--cache_path` (pointing at an
already-built cache) skips this entirely if you need to isolate it. (Both caches now exist
locally from later runs and the crash hasn't recurred, including after the OpenMP fix
below — still not root-caused, just no longer blocking.)

## uv migration + further local validation (2026-10-26, same day)

Ported the `uv` + `pyproject.toml` + `mise` + ruff/pytest/CI tooling from `elie/sota`
onto this branch (that migration commit was previously out of scope — see git history —
now explicitly brought in). Adapted rather than copied: dropped the SPCI/regime-only
dependencies (`statsmodels`, `jumpmodels`, `sklearn-quantile`, `skranger`) since those
source files don't exist outside `elie/sota`, and added `structured-data-models` (kumo).
Actually running the result (not just editing config) surfaced four more real bugs:

1. **Two modules couldn't import at all**: `feature_selection_experiment.py` imported the
   long-deleted `Interpretability` package (unused — only referenced in a comment); 
   `ts_cross_validation/plotting.py` used `from src.Utils...` instead of `from Utils...`.
   Both are one-line fixes, caught by the import-smoke-test (`pytest -m slow`, new in this
   migration) rather than by ruff, since ruff doesn't execute imports.
2. **The trial script's version-check crashed**: `tabicl.__version__` and
   `autogluon.tabular.__version__` don't exist on the installed versions
   (`AttributeError`, `No package metadata found`). Switched to
   `importlib.metadata.version(...)`, which works for every package checked.
3. **`module purge` isn't guarded** — fails outright with `command not found` outside an
   Lmod cluster (e.g. testing the script locally on a laptop). Guarded with
   `command -v module`, a no-op anywhere Lmod exists.
4. **One model failing killed the whole trial/full loop** (`set -e` + a non-zero exit from
   any `m2_pipeline.py` invocation inside the `for` loop aborts the script immediately) —
   caught by actually inducing a failure (a bad `TABPFN_TOKEN`) and watching autogluon/kumo
   never run. Both scripts now catch each model's exit status, keep going, and report a
   summary with a non-zero exit only at the end if anything failed — the entire point of a
   cheap trial is seeing which of the four models are fine, not just the first one.

Ruff (`select = ["F", "E9", "B", "PLE"]`) found ~32 further pre-existing issues in Pablo's
analysis/plotting code (missing `import json`, a dead `if __name__: main()`, an undefined
`r_zone` in a radar plot, a few more) — left as-is, documented in `pyproject.toml`'s
per-file-ignores, since fixing them is a separate bug-fix pass, not this PR's scope.

## TabPFN unblocked (2026-10-26)

Two separate things were wrong, found by actually trying a real token rather than
continuing to treat `tabpfn` as abstractly "blocked":

1. **The first token tried was the wrong kind.** PriorLabs' dashboard has two unrelated
   credentials: an **API Key** (under API → API Keys, prefixed `tabpfn_sk_...`) for
   `tabpfn-client`, their hosted/cloud inference package — a different package than the
   one this repo uses — and an **Access Token** (Account → Settings, at `/account`) for
   the local `tabpfn` package's license-verification flow, which is what `TABPFN_TOKEN`
   actually checks (`tabpfn/browser_auth.py`'s `verify_token()` calls
   `{api_url}/protected/` with the token as a bearer; an API Key gets a 401 there, which
   looks identical to "browser login failed" because the code falls back to trying a
   browser login when the cached token is rejected). Confirmed by reading that module's
   source rather than guessing. Licenses had already been accepted on the account, which
   ruled out the license step itself and pointed at the token.
2. **The fix that actually worked**: run the pipeline once, interactively, in a real
   terminal (not through a tool call — the browser-login flow in `try_browser_login()`
   explicitly checks `sys.stdin.isatty()` and refuses to run otherwise) with no
   `TABPFN_TOKEN` set at all. That opened a browser, logged in against the
   already-accepted licenses, and cached a JWT to `~/.cache/tabpfn/auth_token` — which
   `get_cached_token()` reads automatically on every later run, no env var needed, on
   *this* machine. For the actual headless EPFL cluster, the real Access Token (not an
   API Key) from `/account` still needs to be exported as `TABPFN_TOKEN` there, since a
   compute node can't open a browser.
3. **A second, unrelated bug surfaced right after**, once the license check passed:
   `tabpfn_classifier.py` hardcoded `device="cuda"`, which raised `AssertionError: Torch
   not compiled with CUDA enabled` on this Mac. Changed the default to `"auto"` — TabPFN's
   own `TabPFNClassifier` already supports `device="auto"` natively (confirmed from its
   `__init__` signature), so this defers to the library's own resolution instead of
   duplicating it, and is a no-op change on the actual GPU cluster (`"auto"` resolves to
   `cuda` there too).

With both fixed, `tabpfn` now runs end-to-end on real data — Sharpe 1.23 (Total Ret
+7.61%, MaxDD -12.48%), added to the table above.

## Usage

```bash
# 1. Trial: one direction (up), short AutoGluon time budget, fails fast on bad config.
sbatch slurm/m2-compare-trial.sh

# Read the trial's .out log — specifically the version/preset checks near the top,
# the kumo import check, and whether all four models fit without error — before
# doing anything else.

# 2. Full: both directions, full AutoGluon time budget (3600s/fit by default).
sbatch slurm/m2-compare-full.sh

# Override granularity or AutoGluon settings via --export:
sbatch slurm/m2-compare-full.sh --export=GRANULARITY=4h
sbatch slurm/m2-compare-full.sh --export=AG_PRESET=extreme_quality,AG_TIME_LIMIT=7200
```

Both scripts assume they're submitted from the `SecondaryModel/` repo root (the
`PROJECT_DIR` resolution falls back to `$SLURM_SUBMIT_DIR`, matching `sbatch`'s default).
The locked `.venv` (from `uv.lock`) is created once by whichever script runs first and
reused by the other via `uv sync --locked` — run the trial first so the full job doesn't
pay the install cost cold. Each script also tolerates a model failing mid-run: it keeps
going through the rest, then reports which (if any) failed and exits non-zero only then.

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
