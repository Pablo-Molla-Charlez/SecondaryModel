#!/bin/bash
#SBATCH --job-name=m2-compare-full
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=slurm/logs/m2-compare-full-%j.out
#SBATCH --error=slurm/logs/m2-compare-full-%j.err

# ============================================================================
# M2 model-comparison full run — xgboost (GBM baseline) / tabpfn / autogluon
# (AG_PRESET, default "tabarena" to reach Mitra + TabPFNv2 + TabICL as base
# learners) across both directions, on the fit20 cohort's granularity (1d by
# default). Run slurm/m2-compare-trial.sh first and read its output before
# submitting this.
#
# Same features, same splits, same k/target for every model — only the model
# family changes between runs, so the comparison is actually fair.
#
# Usage:
#   sbatch slurm/m2-compare-full.sh
#   sbatch slurm/m2-compare-full.sh --export=GRANULARITY=4h
#   sbatch slurm/m2-compare-full.sh --export=AG_PRESET=extreme_quality
# ============================================================================

set -euo pipefail

DIRECTIONS=("up" "down")
GRANULARITY="${GRANULARITY:-1d}"
MODELS=("xgboost" "tabpfn" "autogluon")
AG_PRESET="${AG_PRESET:-tabarena}"
AG_TIME_LIMIT="${AG_TIME_LIMIT:-3600}"   # matches factory.py's _AG_TIME_LIMIT default; raise if autogluon times out in the trial
PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"

cd "$PROJECT_DIR"
mkdir -p slurm/logs

# ---------------------------------------------------------------------------
# 1. Clean environment
# ---------------------------------------------------------------------------
module purge

# ---------------------------------------------------------------------------
# 2. Install dependencies
# ---------------------------------------------------------------------------
export PATH="$HOME/.local/bin:$PATH"
if ! command -v uv &> /dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi
uv sync --locked

export HF_HOME="${SCRATCH:-/tmp}/.cache/huggingface"

# ---------------------------------------------------------------------------
# 3. Data precondition — same check as the trial script. Don't start a
#    6-hour job only to fail in the first minute.
# ---------------------------------------------------------------------------
DATA_DIR="${PROJECT_DIR}/src/Data_MLA/Kronos/Crypto/TP/horizon_7"
if [ ! -d "$DATA_DIR" ] || [ -z "$(find "$DATA_DIR" -maxdepth 2 -type f -name '*.csv' -print -quit 2>/dev/null)" ]; then
    echo "[$(date)] ERROR: no CSVs found under $DATA_DIR"
    echo "  Stage it (local machine: slurm/stage-data.sh pulls from gpu8-pablo), then rsync it"
    echo "  onto EPFL cluster storage before submitting this job — see slurm/README.md."
    exit 1
fi
echo "[$(date)] Data OK: $(find "$DATA_DIR" -name '*.csv' | wc -l) CSVs under $DATA_DIR"

# ---------------------------------------------------------------------------
# 4. Run the full model x direction grid at this granularity
# ---------------------------------------------------------------------------
cd src
RESULTS_ROOT="${PROJECT_DIR}/slurm/logs/full-${SLURM_JOB_ID:-local}"
mkdir -p "$RESULTS_ROOT"

TOTAL=$((${#MODELS[@]} * ${#DIRECTIONS[@]}))
DONE=0

for MODEL in "${MODELS[@]}"; do
    for DIRECTION in "${DIRECTIONS[@]}"; do
        DONE=$((DONE + 1))
        echo ""
        echo "============================================"
        echo "  [$(date)] [$DONE/$TOTAL] model=$MODEL direction=$DIRECTION granularity=$GRANULARITY"
        echo "============================================"

        EXTRA_ARGS=()
        if [ "$MODEL" = "autogluon" ]; then
            EXTRA_ARGS=(--ag-presets "$AG_PRESET" --ag-time-limit "$AG_TIME_LIMIT")
        fi

        uv run python m2_pipeline.py \
            --config config.yaml \
            --phase training \
            --m2 "$MODEL" \
            --direction "$DIRECTION" \
            --granularity "$GRANULARITY" \
            "${EXTRA_ARGS[@]}" \
            2>&1 | tee "${RESULTS_ROOT}/${MODEL}_${DIRECTION}.log"
    done
done

# ---------------------------------------------------------------------------
# 5. Final report — point at where each model's analysis_summary.json landed
#    (Output/<M1>/<model>/<DIRECTION>/.../analysis_summary.json per
#    run_analysis()'s save_dir convention in m2_pipeline.py)
# ---------------------------------------------------------------------------
echo ""
echo "============================================"
echo "  FULL COMPARISON RESULTS"
echo "============================================"
echo "Logs:    $RESULTS_ROOT"
echo "Outputs: ${PROJECT_DIR}/src/Output/Kronos/<model>/<UP|DOWN>/Utility_Score*/<gran>_tp/analysis_summary.json"
echo "Found this run's summaries:"
find "${PROJECT_DIR}/src/Output/Kronos" -name "analysis_summary.json" -newer "$RESULTS_ROOT" 2>/dev/null | sort || true
echo ""
echo "[$(date)] Full comparison complete."
