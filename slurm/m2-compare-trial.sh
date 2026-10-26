#!/bin/bash
#SBATCH --job-name=m2-compare-trial
#SBATCH --partition=gpu
#SBATCH --qos=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:45:00
#SBATCH --output=slurm/logs/m2-compare-trial-%j.out
#SBATCH --error=slurm/logs/m2-compare-trial-%j.err

# ============================================================================
# M2 model-comparison trial — xgboost / tabpfn / autogluon(tabarena) / kumo
# on a single direction+granularity slice, before committing to the full
# sweep.
#
# Confirms: environment builds, data is staged, TabPFN/AutoGluon/Kumo load
# what we think they load, and nothing crashes — fast and cheap.
#
# Environment: uv + pyproject.toml/uv.lock (Python 3.11), locked. See
# mise.toml / README.md's "Setup" section — this replaced the old conda +
# requirements.txt setup.
#
# Usage:
#   sbatch slurm/m2-compare-trial.sh
#   sbatch slurm/m2-compare-trial.sh --export=GRANULARITY=4h
# ============================================================================

set -euo pipefail

DIRECTION="up"
GRANULARITY="${GRANULARITY:-1d}"
MODELS=("xgboost" "tabpfn" "autogluon" "kumo")
AG_PRESET="${AG_PRESET:-tabarena}"   # AutoGluon preset that includes Mitra/TabPFNv2/TabICL as base learners
AG_TIME_LIMIT_TRIAL=600              # short budget per fit, just to prove it runs
PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"

cd "$PROJECT_DIR"
mkdir -p slurm/logs

# ---------------------------------------------------------------------------
# 1. uv — prefer a cluster module if one exists; otherwise install it
#    locally (official installer, no sudo/root needed, lands in
#    $HOME/.local/bin). Pinned version is in mise.toml ([tools] uv = "0.12").
# ---------------------------------------------------------------------------
# No-op (and `command -v module` guarded) outside an Lmod cluster, e.g. when
# testing this script locally on a laptop.
command -v module &> /dev/null && module purge
if ! command -v uv &> /dev/null; then
    command -v module &> /dev/null && module load uv 2>/dev/null
fi
if ! command -v uv &> /dev/null; then
    echo "[$(date)] No 'uv' on PATH or as a module — installing it locally."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi
if ! command -v uv &> /dev/null; then
    echo "[$(date)] ERROR: could not find or install 'uv'. Load your cluster's uv module"
    echo "  (check 'module avail uv') or install it manually before sbatch."
    exit 1
fi
echo "[$(date)] uv: $(uv --version)"

# ---------------------------------------------------------------------------
# 2. Locked environment (Python 3.11, CTTS-equivalent ML stack incl. Kumo).
#    `uv sync --locked` creates/reuses .venv from uv.lock, so re-running a
#    trial doesn't reinstall the whole ML stack every time. torch resolves
#    to the CUDA 12.8 build on Linux per pyproject.toml's [tool.uv.sources].
# ---------------------------------------------------------------------------
uv sync --locked

export HF_HOME="${SCRATCH:-/tmp}/.cache/huggingface"

# Verified locally 2026-10-26: without this, xgboost segfaults on macOS
# (torch and xgboost both bundle libomp, and the duplicate runtime init
# crashes the process). Harmless no-op on Linux where this rarely triggers,
# but cheap enough to always set.
export KMP_DUPLICATE_LIB_OK=TRUE

# Verified locally 2026-10-26: tabpfn now gates model downloads behind an
# explicit license acceptance (tries a browser login, which fails outright
# on a headless node) unless TABPFN_TOKEN is set. Get one from
# https://ux.priorlabs.ai and export it before submitting, or every tabpfn
# fit in this job will fail with TabPFNLicenseError.
if [ -z "${TABPFN_TOKEN:-}" ]; then
    echo "[$(date)] ERROR: TABPFN_TOKEN is not set."
    echo "  tabpfn will fail to download its model on this headless node without it."
    echo "  Get the ACCESS TOKEN (not an API Key -- that's for the unrelated tabpfn-client"
    echo "  package) from https://ux.priorlabs.ai -> Account -> Settings, then:"
    echo "  export TABPFN_TOKEN=... before sbatch,"
    echo "  or sbatch --export=ALL,TABPFN_TOKEN=... slurm/m2-compare-trial.sh"
    exit 1
fi

# ---------------------------------------------------------------------------
# 3. Data precondition — Data_MLA/* is gitignored and lives on gpu8 (the lab
#    GPU server), not in this checkout. Fail fast with a clear message rather
#    than letting the pipeline die deep inside cache-building.
# ---------------------------------------------------------------------------
DATA_DIR="${PROJECT_DIR}/src/Data_MLA/Kronos/Crypto/TP/horizon_7"
if [ ! -d "$DATA_DIR" ] || [ -z "$(find "$DATA_DIR" -maxdepth 2 -type f -name '*.csv' -print -quit 2>/dev/null)" ]; then
    echo "[$(date)] ERROR: no CSVs found under $DATA_DIR"
    echo "  This data is gitignored and lives on gpu8 (confirmed 2026-10-26, NOT gpu7 — that"
    echo "  machine is freshly provisioned with nothing staged on it)."
    echo "  On a machine that can reach gpu8-pablo (your laptop, not an EPFL compute node):"
    echo "    slurm/stage-data.sh"
    echo "  Then rsync your local Data_MLA/Kronos/Crypto/TP/horizon_7/ onto EPFL cluster"
    echo "  storage (scratch) before submitting this job — EPFL nodes can't reach gpu8"
    echo "  with your laptop's keys, so this is a second, separate hop."
    exit 1
fi
echo "[$(date)] Data OK: $(find "$DATA_DIR" -name '*.csv' | wc -l) CSVs under $DATA_DIR"

# ---------------------------------------------------------------------------
# 4. Sanity-check what we're actually loading before trusting any numbers.
#    tabpfn's PyPI package version does not map 1:1 onto Prior Labs' version
#    names — per PriorLabs/TabPFN's current README, a bare TabPFNClassifier()
#    now defaults to TabPFN-3.5, not the "2.5" we originally assumed. Confirm
#    here, don't assume either number.
# ---------------------------------------------------------------------------
echo "[$(date)] Checking installed model-backend versions ..."
uv run python -c "
import importlib.metadata as metadata
import tabpfn, tabicl, autogluon.tabular, xgboost  # noqa: F401 -- import-success is the check
# tabicl and autogluon.tabular don't expose __version__ (verified 2026-10-26:
# AttributeError / no package metadata under the bare 'autogluon' name) --
# importlib.metadata is the only version source that works for every package here.
print(f'  tabpfn       {metadata.version(\"tabpfn\")}')
print(f'  tabicl       {metadata.version(\"tabicl\")}')
print(f'  autogluon    {metadata.version(\"autogluon.tabular\")}')
print(f'  xgboost      {metadata.version(\"xgboost\")}')
try:
    from tabpfn import TabPFNClassifier
    import inspect
    print(f'  TabPFNClassifier defaults: {inspect.signature(TabPFNClassifier.__init__)}')
except Exception as e:
    print(f'  (could not introspect TabPFNClassifier: {e})')
"
echo "[$(date)] If this isn't the TabPFN generation you expect, stop and check the pin in pyproject.toml"
echo "  before trusting the sweep below."

# ---------------------------------------------------------------------------
# 5. Confirm the AutoGluon preset we want actually exists in this version
#    AND actually includes Mitra -- verified by running this locally
#    (2026-10-26) against the locked autogluon-tabular==1.5.0: the old
#    `PRESETS_CONFIGS` import no longer exists in 1.6.x (replaced by
#    tabular_presets_dict) — this script tries both. Only "tabarena"
#    (zeroshot_2025_tabfm) has Mitra (REALTABPFN-V2, TABM, TABICL, GBM, CAT,
#    XGB, MITRA) -- confirmed by actually importing the hyperparameter
#    config, not guessed from docs. "noncommercial" has TabPFN-3 but still
#    no Mitra, if you want that instead.
# ---------------------------------------------------------------------------
echo "[$(date)] Checking AutoGluon preset '$AG_PRESET' is known to this install ..."
uv run python -c "
preset = '$AG_PRESET'
try:
    from autogluon.tabular.configs.presets_configs import PRESETS_CONFIGS as presets
except ImportError:
    from autogluon.tabular.configs.presets_configs import tabular_presets_dict as presets
if preset not in presets:
    print(f'  WARNING: preset {preset!r} not found. Available: {sorted(presets)}')
    print('  No fallback reaches Mitra except tabarena itself -- extreme_quality/noncommercial')
    print('  do NOT include Mitra (checked against the locked AutoGluon version).')
else:
    cfg = presets[preset]
    print(f'  OK: {preset!r} -> {cfg}')
    try:
        from autogluon.tabular.configs.hyperparameter_configs import get_hyperparameter_config
        families = list(get_hyperparameter_config(cfg['hyperparameters']).keys())
        print(f'  Model families in this preset: {families}')
        print(f'  Includes Mitra: {\"MITRA\" in families}')
    except Exception as e:
        print(f'  (could not resolve hyperparameter families: {e})')
"

# ---------------------------------------------------------------------------
# 6. Kumo Tabular import check — this wrapper (kumo_tabular_classifier.py)
#    is validated locally on CPU (macOS) against the fit20 cohort; this is
#    the first time it will run on an actual GPU, so check the import first.
# ---------------------------------------------------------------------------
echo "[$(date)] Checking structured-data-models (Kumo Tabular) import ..."
uv run python -c "
import sdm
print(f'  sdm (structured-data-models) {getattr(sdm, \"__version__\", \"unknown\")}')
print(f'  sdm.models.KumoTabular: {sdm.models.KumoTabular}')
" || echo "  WARNING: sdm import failed — 'kumo' will fail below. Check 'uv sync --locked' output above."

# ---------------------------------------------------------------------------
# 7. Run one short fit per model on direction=up, granularity=$GRANULARITY
# ---------------------------------------------------------------------------
cd src
RESULTS_ROOT="${PROJECT_DIR}/slurm/logs/trial-${SLURM_JOB_ID:-local}"
mkdir -p "$RESULTS_ROOT"

# One model failing (e.g. a bad/missing token) must not hide whether the
# other three are fine -- that's the whole point of a cheap trial before the
# multi-hour full sweep. Collect failures, keep going, exit non-zero at the
# end if any model failed (verified 2026-10-26: without this, `set -e` killed
# the loop outright on tabpfn's license error and autogluon/kumo never ran).
FAILED_MODELS=()
for MODEL in "${MODELS[@]}"; do
    echo ""
    echo "============================================"
    echo "  [$(date)] Trial fit: model=$MODEL direction=$DIRECTION granularity=$GRANULARITY"
    echo "============================================"

    EXTRA_ARGS=()
    if [ "$MODEL" = "autogluon" ]; then
        EXTRA_ARGS=(--ag-presets "$AG_PRESET" --ag-time-limit "$AG_TIME_LIMIT_TRIAL")
    fi

    set +e
    uv run --project .. python m2_pipeline.py \
        --config config.yaml \
        --phase training \
        --m2 "$MODEL" \
        --direction "$DIRECTION" \
        --granularity "$GRANULARITY" \
        "${EXTRA_ARGS[@]}" \
        2>&1 | tee "${RESULTS_ROOT}/${MODEL}.log"
    MODEL_STATUS=${PIPESTATUS[0]}
    set -e

    if [ "$MODEL_STATUS" -ne 0 ]; then
        echo "[$(date)] FAILED: model=$MODEL (exit $MODEL_STATUS) -- see ${RESULTS_ROOT}/${MODEL}.log"
        FAILED_MODELS+=("$MODEL")
    else
        echo "[$(date)] OK: model=$MODEL"
    fi
done

echo ""
echo "[$(date)] Trial complete. Logs: $RESULTS_ROOT"
echo "Pay particular attention to the 'kumo' log above — cross-check its probability-sum"
echo "sanity (predict_proba(X).sum(axis=1) should be ~1 for every row) if its precision/"
echo "coverage numbers look off on this GPU vs. the local CPU run."
if [ ${#FAILED_MODELS[@]} -eq 0 ]; then
    echo "All models fit cleanly. If the version/preset checks above look right, run:"
    echo "  sbatch slurm/m2-compare-full.sh"
else
    echo "FAILED: ${FAILED_MODELS[*]} -- fix these before running the full sweep."
    exit 1
fi
