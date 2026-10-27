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
# learners) / kumo (NVIDIA Kumo Tabular) across both directions, on the
# fit20 cohort's granularity (1d by default). Run slurm/m2-compare-trial.sh
# first and read its output before submitting this — the kumo wrapper in
# particular has never actually been run.
#
# Same features, same splits, same k/target for every model — only the model
# family changes between runs, so the comparison is actually fair.
#
# Environment: uv + pyproject.toml/uv.lock (Python 3.11), locked. See
# mise.toml / README.md's "Setup" section.
#
# Usage:
#   sbatch slurm/m2-compare-full.sh
#   sbatch slurm/m2-compare-full.sh --export=GRANULARITY=4h
#   sbatch slurm/m2-compare-full.sh --export=AG_PRESET=extreme_quality
#   sbatch slurm/m2-compare-full.sh --export=ALL,CONFIG=tb_pilot/configs/tb_ref.yaml,DATA_DIR=.../TB/horizon_7
# ============================================================================

set -euo pipefail

DIRECTIONS=("up" "down")
GRANULARITY="${GRANULARITY:-1d}"
MODELS=("xgboost" "tabpfn" "autogluon" "kumo")
AG_PRESET="${AG_PRESET:-tabarena}"
AG_TIME_LIMIT="${AG_TIME_LIMIT:-3600}"   # matches factory.py's _AG_TIME_LIMIT default; raise if autogluon times out in the trial
PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
CONFIG="${CONFIG:-config.yaml}"          # relative to src/ (same as --config always was); the TB pilot passes tb_pilot/configs/<mode>.yaml

cd "$PROJECT_DIR"
mkdir -p slurm/logs

# ---------------------------------------------------------------------------
# 1. uv — same approach as the trial script. uv.lock already has everything
#    resolved from that run, so this is just `uv sync --locked` (fast, no
#    re-resolution) if the trial ran on this node/scratch already.
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
    echo "  or sbatch --export=ALL,TABPFN_TOKEN=... slurm/m2-compare-full.sh"
    exit 1
fi

# ---------------------------------------------------------------------------
# 3. Data precondition — same check as the trial script. Don't start a
#    6-hour job only to fail in the first minute.
# ---------------------------------------------------------------------------
DATA_DIR="${DATA_DIR:-${PROJECT_DIR}/src/Data_MLA/Kronos/Crypto/TP/horizon_7}"
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
# One model/direction failing partway through a multi-hour sweep must not
# throw away every other combination that would have finished fine.
FAILED_RUNS=()

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

        set +e
        uv run --project .. python m2_pipeline.py \
            --config "$CONFIG" \
            --phase training \
            --m2 "$MODEL" \
            --direction "$DIRECTION" \
            --granularity "$GRANULARITY" \
            "${EXTRA_ARGS[@]}" \
            2>&1 | tee "${RESULTS_ROOT}/${MODEL}_${DIRECTION}.log"
        RUN_STATUS=${PIPESTATUS[0]}
        set -e

        if [ "$RUN_STATUS" -ne 0 ]; then
            echo "[$(date)] FAILED: model=$MODEL direction=$DIRECTION (exit $RUN_STATUS)"
            FAILED_RUNS+=("${MODEL}_${DIRECTION}")
        fi
    done
done

# ---------------------------------------------------------------------------
# 5. Final report — point at where each model's analysis_summary.json landed
#    (<output_root>/<M1>/<model>/<DIRECTION>/.../<gran>_<mode>/analysis_summary.json
#    per run_analysis()'s save_dir convention in m2_pipeline.py). output_root and
#    meta_label_mode are read back out of $CONFIG rather than assumed, since the
#    TB pilot's per-mode configs override both (tb_pilot_plan.md item 5).
# ---------------------------------------------------------------------------
OUTPUT_ROOT="$(grep -m1 'output_root:' "$CONFIG" | sed -E 's/^[^:]*:[[:space:]]*"?([^"#]+)"?.*/\1/' | xargs)"
OUTPUT_ROOT="${OUTPUT_ROOT:-Output}"
MODE="$(grep -m1 'meta_label_mode:' "$CONFIG" | sed -E 's/^[^:]*:[[:space:]]*"?([^"#]+)"?.*/\1/' | xargs)"
MODE="${MODE:-tp}"

echo ""
echo "============================================"
echo "  FULL COMPARISON RESULTS"
echo "============================================"
echo "Logs:    $RESULTS_ROOT"
echo "Outputs: ${PROJECT_DIR}/src/${OUTPUT_ROOT}/Kronos/<model>/<UP|DOWN>/Utility_Score*/<gran>_${MODE}/analysis_summary.json"
echo "Found this run's summaries:"
find "${PROJECT_DIR}/src/${OUTPUT_ROOT}/Kronos" -name "analysis_summary.json" -newer "$RESULTS_ROOT" 2>/dev/null | sort || true
echo ""
if [ ${#FAILED_RUNS[@]} -eq 0 ]; then
    echo "[$(date)] Full comparison complete. All $TOTAL runs succeeded."
else
    echo "[$(date)] Full comparison finished with failures: ${FAILED_RUNS[*]}"
    echo "  (${#FAILED_RUNS[@]}/$TOTAL failed — see their logs under $RESULTS_ROOT)"
    exit 1
fi
