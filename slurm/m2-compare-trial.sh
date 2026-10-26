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
# Environment: this branch is based on upstream/main, which predates
# Elie's uv/pyproject.toml migration (that's on elie/sota, not here) — it
# still uses conda + requirements.txt, per README.md's own "Setup" section.
# Don't "fix" this to uv without also bringing that migration into the PR.
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
CONDA_ENV="${CONDA_ENV:-S2_m2compare}"
PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"

cd "$PROJECT_DIR"
mkdir -p slurm/logs

# ---------------------------------------------------------------------------
# 1. Load conda — try the common module name, else require it preinstalled.
#    (EPFL cluster module names vary; `module load miniconda3` is a common
#    one but not guaranteed. If this fails, load whatever your cluster's
#    conda module is actually called before submitting.)
# ---------------------------------------------------------------------------
module purge
if ! command -v conda &> /dev/null; then
    module load miniconda3 2>/dev/null || module load anaconda3 2>/dev/null || true
fi
if ! command -v conda &> /dev/null; then
    echo "[$(date)] ERROR: no 'conda' on PATH and no miniconda3/anaconda3 module found."
    echo "  Load your cluster's conda module (check 'module avail conda' / 'module avail anaconda')"
    echo "  and either edit this script's module-load line or run 'module load <name>' before sbatch."
    exit 1
fi
eval "$(conda shell.bash hook)"

# ---------------------------------------------------------------------------
# 2. Create the env once, then reuse it (so re-running a trial doesn't
#    reinstall the whole ML stack every time).
# ---------------------------------------------------------------------------
if ! conda env list | grep -qE "^\s*${CONDA_ENV}\s"; then
    echo "[$(date)] Creating conda env '$CONDA_ENV' (python=3.11, per README.md's Setup section) ..."
    conda create -n "$CONDA_ENV" python=3.11 -y
fi
conda activate "$CONDA_ENV"
pip install -r requirements.txt

export HF_HOME="${SCRATCH:-/tmp}/.cache/huggingface"

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
python -c "
import tabpfn, tabicl, autogluon.tabular, xgboost
print(f'  tabpfn       {tabpfn.__version__}')
print(f'  tabicl       {tabicl.__version__}')
print(f'  autogluon    {autogluon.tabular.__version__}')
print(f'  xgboost      {xgboost.__version__}')
try:
    from tabpfn import TabPFNClassifier
    import inspect
    print(f'  TabPFNClassifier defaults: {inspect.signature(TabPFNClassifier.__init__)}')
except Exception as e:
    print(f'  (could not introspect TabPFNClassifier: {e})')
"
echo "[$(date)] If this isn't the TabPFN generation you expect, stop and check the pin in requirements.txt"
echo "  before trusting the sweep below."

# ---------------------------------------------------------------------------
# 5. Confirm the AutoGluon preset we want actually exists in this version
#    (preset names have shifted across AutoGluon releases, and
#    requirements.txt pins autogluon.tabular unversioned — don't assume
#    which release you got).
# ---------------------------------------------------------------------------
echo "[$(date)] Checking AutoGluon preset '$AG_PRESET' is known to this install ..."
python -c "
from autogluon.tabular.configs.presets_configs import PRESETS_CONFIGS
preset = '$AG_PRESET'
if preset not in PRESETS_CONFIGS:
    print(f'  WARNING: preset {preset!r} not found. Available: {sorted(PRESETS_CONFIGS)}')
    print('  Falling back to extreme_quality if present, else best_quality.')
else:
    print(f'  OK: {preset!r} -> {PRESETS_CONFIGS[preset]}')
"

# ---------------------------------------------------------------------------
# 6. Kumo Tabular import check — this wrapper (kumo_tabular_classifier.py)
#    was written by reading NVIDIA's source, never actually run (no local
#    GPU here). This is the single most important check in this script.
# ---------------------------------------------------------------------------
echo "[$(date)] Checking structured-data-models (Kumo Tabular) import ..."
python -c "
import sdm
print(f'  sdm (structured-data-models) {getattr(sdm, \"__version__\", \"unknown\")}')
print(f'  sdm.models.KumoTabular: {sdm.models.KumoTabular}')
" || echo "  WARNING: sdm import failed — 'kumo' will fail below. Check requirements.txt installed cleanly."

# ---------------------------------------------------------------------------
# 7. Run one short fit per model on direction=up, granularity=$GRANULARITY
# ---------------------------------------------------------------------------
cd src
RESULTS_ROOT="${PROJECT_DIR}/slurm/logs/trial-${SLURM_JOB_ID:-local}"
mkdir -p "$RESULTS_ROOT"

for MODEL in "${MODELS[@]}"; do
    echo ""
    echo "============================================"
    echo "  [$(date)] Trial fit: model=$MODEL direction=$DIRECTION granularity=$GRANULARITY"
    echo "============================================"

    EXTRA_ARGS=()
    if [ "$MODEL" = "autogluon" ]; then
        EXTRA_ARGS=(--ag-presets "$AG_PRESET" --ag-time-limit "$AG_TIME_LIMIT_TRIAL")
    fi

    python m2_pipeline.py \
        --config config.yaml \
        --phase training \
        --m2 "$MODEL" \
        --direction "$DIRECTION" \
        --granularity "$GRANULARITY" \
        "${EXTRA_ARGS[@]}" \
        2>&1 | tee "${RESULTS_ROOT}/${MODEL}.log"
done

echo ""
echo "[$(date)] Trial complete. Logs: $RESULTS_ROOT"
echo "Pay particular attention to the 'kumo' log above — that wrapper is unverified until"
echo "this trial actually runs it. The pipeline doesn't print a probability-sum check"
echo "itself; if kumo's precision/coverage numbers look off, that's the first thing to"
echo "verify by hand (predict_proba(X).sum(axis=1) should be ~1 for every row)."
echo "If every model fit cleanly and the version/preset checks above look right, run:"
echo "  sbatch slurm/m2-compare-full.sh"
