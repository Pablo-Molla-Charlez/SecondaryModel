#!/bin/bash
# ============================================================================
# TB-target pilot launcher (tb_pilot_plan.md Phase 4, item 10).
#
# Unlike the other two scripts here, this one is NOT meant to be sbatch'd
# itself — run it directly on the login node (`slurm/tb-pilot.sh`). It
# submits two chained jobs:
#
#   1. prepare — runs the generator (Data_MLA/meta_label_tb.py), then
#      make_configs, then build_caches, in that order, once. This is what
#      keeps the sweep's array tasks from racing to build the same mode's
#      label tree / config / dataset cache the first time they touch it
#      (tb_pilot_plan.md F5).
#   2. sweep   — an array job, `--dependency=afterok` on the prepare job,
#      covering every (mode, model) pair (default 5 modes x 4 models,
#      `--array=0-19%16`; the %16 matches this allocation's max-16-GPU cap).
#      Each task runs both directions with the same flags
#      m2-compare-full.sh uses (AG tabarena, 3600s): 40 fits total, about
#      10 GPU-hours for AutoGluon. All 4 backends run on every mode, since
#      k is a property of the target, not of a backend.
#
# The sweep task discovers its mode names from tb_pilot/configs/*.yaml at
# run time (sorted), rather than recomputing Data_MLA/meta_label_tb.py's
# mode_name() rounding in bash — one less place for the two to drift apart.
#
# Usage:
#   slurm/tb-pilot.sh
#   KS="0.5 1.0 1.5 2.0" GRANULARITY=1d slurm/tb-pilot.sh
# ============================================================================

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_DIR"
mkdir -p slurm/logs

KS="${KS:-0.5 1.0 1.5 2.0}"
read -r -a KS_ARR <<< "$KS"
GRANULARITY="${GRANULARITY:-1d}"
TP_SRC="${TP_SRC:-Data_MLA/Kronos/Crypto/TP/horizon_7/${GRANULARITY}_tp}"
TB_DST="${TB_DST:-Data_MLA/Kronos/Crypto/TB/horizon_7}"
AG_PRESET="${AG_PRESET:-tabarena}"
AG_TIME_LIMIT="${AG_TIME_LIMIT:-3600}"
MODELS=("xgboost" "tabpfn" "autogluon" "kumo")

N_MODES=$((1 + ${#KS_ARR[@]}))   # +1 for the --ref arm, always included
N_MODELS=${#MODELS[@]}
N_TASKS=$((N_MODES * N_MODELS))

RUN_ID="tb-pilot-$(date +%Y%m%d-%H%M%S)"
JOB_DIR="${PROJECT_DIR}/slurm/logs/${RUN_ID}"
mkdir -p "$JOB_DIR"
echo "[$(date)] ks: ${KS_ARR[*]} (+ tb_ref)  |  models: ${MODELS[*]}"
echo "[$(date)] ${N_MODES} modes x ${N_MODELS} models = ${N_TASKS} array tasks (2 directions each)"
echo "[$(date)] job scripts + logs under: $JOB_DIR"

# ---------------------------------------------------------------------------
# 1. Prepare job — generator, make_configs, build_caches, in that order.
#    Same resource shape as m2-compare-trial.sh's job class (short, cheap);
#    --gres=gpu:1 is unused here but kept for consistency with the only
#    resource request confirmed to work on this cluster's "gpu" partition.
# ---------------------------------------------------------------------------
PREPARE_SCRIPT="${JOB_DIR}/prepare.sh"
cat > "$PREPARE_SCRIPT" <<EOF
#!/bin/bash
#SBATCH --job-name=tb-pilot-prepare
#SBATCH --partition=gpu
#SBATCH --qos=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:45:00
#SBATCH --output=${JOB_DIR}/prepare-%j.out
#SBATCH --error=${JOB_DIR}/prepare-%j.err

set -euo pipefail
cd "${PROJECT_DIR}"
command -v module &> /dev/null && module purge
if ! command -v uv &> /dev/null; then
    command -v module &> /dev/null && module load uv 2>/dev/null
fi
if ! command -v uv &> /dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="\$HOME/.local/bin:\$PATH"
fi
uv sync --locked
export KMP_DUPLICATE_LIB_OK=TRUE

cd src
echo "[\$(date)] [1/3] generating TB label trees: ${TP_SRC} -> ${TB_DST}"
uv run --project .. python -m Data_MLA.meta_label_tb \\
    --src "${TP_SRC}" --dst "${TB_DST}" \\
    --ref --ks ${KS} --config config.yaml

echo "[\$(date)] [2/3] writing per-mode configs"
uv run --project .. python -m tb_pilot.make_configs \\
    --tb-root "${TB_DST}" --gran "${GRANULARITY}" --ref --ks ${KS}

N_WRITTEN=\$(ls tb_pilot/configs/*.yaml 2>/dev/null | wc -l | tr -d ' ')
if [ "\$N_WRITTEN" -ne ${N_MODES} ]; then
    echo "[\$(date)] ERROR: expected ${N_MODES} mode configs, make_configs wrote \$N_WRITTEN"
    exit 1
fi

echo "[\$(date)] [3/3] pre-building dataset caches"
uv run --project .. python -m tb_pilot.build_caches --configs-dir tb_pilot/configs
echo "[\$(date)] prepare done."
EOF
chmod +x "$PREPARE_SCRIPT"

echo "[$(date)] submitting prepare job ..."
PREPARE_JOBID="$(sbatch --parsable "$PREPARE_SCRIPT")"
echo "[$(date)] prepare job: $PREPARE_JOBID"

# ---------------------------------------------------------------------------
# 2. Sweep array job — N_MODES x N_MODELS, each task runs both directions.
# ---------------------------------------------------------------------------
SWEEP_SCRIPT="${JOB_DIR}/sweep.sh"
cat > "$SWEEP_SCRIPT" <<EOF
#!/bin/bash
#SBATCH --job-name=tb-pilot-sweep
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --array=0-$((N_TASKS - 1))%16
#SBATCH --output=${JOB_DIR}/sweep-%A_%a.out
#SBATCH --error=${JOB_DIR}/sweep-%A_%a.err

set -euo pipefail
cd "${PROJECT_DIR}"
command -v module &> /dev/null && module purge
if ! command -v uv &> /dev/null; then
    command -v module &> /dev/null && module load uv 2>/dev/null
fi
if ! command -v uv &> /dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="\$HOME/.local/bin:\$PATH"
fi
uv sync --locked
export HF_HOME="\${SCRATCH:-/tmp}/.cache/huggingface"
export KMP_DUPLICATE_LIB_OK=TRUE
# tabpfn's get_cached_token() checks TABPFN_TOKEN, then ~/.cache/tabpfn/auth_token, then
# ~/.tabpfn/token (tabpfn/browser_auth.py) -- a cached file works with no env var at all,
# so only fail here if none of the three are present.
if [ -z "\${TABPFN_TOKEN:-}" ] && [ ! -s "\$HOME/.cache/tabpfn/auth_token" ] && [ ! -s "\$HOME/.tabpfn/token" ]; then
    echo "[\$(date)] ERROR: no TABPFN_TOKEN and no cached token at ~/.cache/tabpfn/auth_token or ~/.tabpfn/token."
    echo "  See m2-compare-full.sh for how to get one."
    exit 1
fi

cd src
mapfile -t MODES < <(ls tb_pilot/configs/*.yaml | xargs -n1 basename | sed 's/\.yaml\$//' | sort)
if [ "\${#MODES[@]}" -ne ${N_MODES} ]; then
    echo "[\$(date)] ERROR: expected ${N_MODES} mode configs under tb_pilot/configs, found \${#MODES[@]}"
    echo "  (did the prepare job finish, or did KS change since it ran?)"
    exit 1
fi
MODELS=(${MODELS[*]})

MODE_IDX=\$((SLURM_ARRAY_TASK_ID / ${N_MODELS}))
MODEL_IDX=\$((SLURM_ARRAY_TASK_ID % ${N_MODELS}))
MODE="\${MODES[\$MODE_IDX]}"
MODEL="\${MODELS[\$MODEL_IDX]}"
echo "[\$(date)] task \$SLURM_ARRAY_TASK_ID -> mode=\$MODE model=\$MODEL"

for DIRECTION in up down; do
    EXTRA_ARGS=()
    if [ "\$MODEL" = "autogluon" ]; then
        EXTRA_ARGS=(--ag-presets "${AG_PRESET}" --ag-time-limit "${AG_TIME_LIMIT}")
    fi
    echo ""
    echo "[\$(date)] mode=\$MODE model=\$MODEL direction=\$DIRECTION granularity=${GRANULARITY}"
    uv run --project .. python m2_pipeline.py \\
        --config "tb_pilot/configs/\${MODE}.yaml" \\
        --phase training \\
        --m2 "\$MODEL" \\
        --direction "\$DIRECTION" \\
        --granularity "${GRANULARITY}" \\
        "\${EXTRA_ARGS[@]}"
done
EOF
chmod +x "$SWEEP_SCRIPT"

echo "[$(date)] submitting sweep array job (dependency: afterok:$PREPARE_JOBID) ..."
SWEEP_JOBID="$(sbatch --parsable --dependency="afterok:${PREPARE_JOBID}" "$SWEEP_SCRIPT")"
echo "[$(date)] sweep job: $SWEEP_JOBID (array 0-$((N_TASKS - 1))%16)"
echo ""
echo "Prepare log: ${JOB_DIR}/prepare-${PREPARE_JOBID}.out"
echo "Sweep logs:  ${JOB_DIR}/sweep-${SWEEP_JOBID}_<task>.out"
echo "squeue -u \$USER   # to watch both"
