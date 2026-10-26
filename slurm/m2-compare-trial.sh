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
# M2 model-comparison trial — xgboost / tabpfn / autogluon(tabarena) on a
# single direction+granularity slice, before committing to the full sweep.
#
# Confirms: environment builds, data is staged, TabPFN/AutoGluon load the
# generations we think they load, and nothing crashes — fast and cheap.
#
# Usage:
#   sbatch slurm/m2-compare-trial.sh
#   sbatch slurm/m2-compare-trial.sh --export=GRANULARITY=4h
# ============================================================================

set -euo pipefail

DIRECTION="up"
GRANULARITY="${GRANULARITY:-1d}"
MODELS=("xgboost" "tabpfn" "autogluon")
AG_PRESET="${AG_PRESET:-tabarena}"   # AutoGluon preset that includes Mitra/TabPFNv2/TabICL as base learners
AG_TIME_LIMIT_TRIAL=600              # short budget per fit, just to prove it runs
PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"

cd "$PROJECT_DIR"
mkdir -p slurm/logs

# ---------------------------------------------------------------------------
# 1. Clean environment — let uv manage Python/CUDA, avoid module conflicts
# ---------------------------------------------------------------------------
module purge

# ---------------------------------------------------------------------------
# 2. Install dependencies (uv manages Python 3.11, CUDA 12.8 torch on Linux)
# ---------------------------------------------------------------------------
export PATH="$HOME/.local/bin:$PATH"
if ! command -v uv &> /dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi
uv sync --locked

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
#    tabpfn's PyPI package version does not map 1:1 onto Prior Labs' "2.5"
#    marketing name — confirm here, don't assume.
# ---------------------------------------------------------------------------
echo "[$(date)] Checking installed model-backend versions ..."
uv run python -c "
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
echo "[$(date)] If this isn't the TabPFN generation you expect, stop and check the pin in pyproject.toml"
echo "  before trusting the sweep below."

# ---------------------------------------------------------------------------
# 5. Confirm the AutoGluon preset we want actually exists in this version
#    (preset names have shifted across AutoGluon releases — don't assume).
# ---------------------------------------------------------------------------
echo "[$(date)] Checking AutoGluon preset '$AG_PRESET' is known to this install ..."
uv run python -c "
from autogluon.tabular.configs.presets_configs import PRESETS_CONFIGS
preset = '$AG_PRESET'
if preset not in PRESETS_CONFIGS:
    print(f'  WARNING: preset {preset!r} not found. Available: {sorted(PRESETS_CONFIGS)}')
    print('  Falling back to extreme_quality if present, else best_quality.')
else:
    print(f'  OK: {preset!r} -> {PRESETS_CONFIGS[preset]}')
"

# ---------------------------------------------------------------------------
# 6. Run one short fit per model on direction=up, granularity=$GRANULARITY
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

    uv run python m2_pipeline.py \
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
echo "If every model fit cleanly and the version/preset checks above look right, run:"
echo "  sbatch slurm/m2-compare-full.sh"
