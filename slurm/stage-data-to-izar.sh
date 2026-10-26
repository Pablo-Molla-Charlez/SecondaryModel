#!/bin/bash
# ============================================================================
# Hop 2: push the already-staged Data_MLA/Kronos/Crypto/TP/horizon_7 dataset
# from this laptop onto Izar's /scratch (EPFL compute nodes can't reach
# gpu8-pablo directly, so this machine is the relay — see slurm/README.md).
#
# Run slurm/stage-data.sh FIRST if you haven't (that's hop 1, gpu8 -> here).
#
# Access: SCITAS ticket RITM0523861, hpc-masters group, account live
# 2026-10-27. Login host izar.hpc.epfl.ch (GASPAR username). First connection
# needs the EPFL network or VPN, and you'll be prompted to accept the host
# key the first time.
#
# Usage:
#   slurm/stage-data-to-izar.sh <gaspar-username>
#   slurm/stage-data-to-izar.sh <gaspar-username> /scratch/<gaspar-username>/custom/path
# ============================================================================

set -euo pipefail

if [ -z "${1:-}" ]; then
    echo "Usage: $0 <gaspar-username> [remote-base-path]"
    echo "  e.g.: $0 ebruno"
    exit 1
fi

GASPAR="$1"
HOST="izar.hpc.epfl.ch"
REMOTE_BASE="${2:-/scratch/${GASPAR}/SecondaryModel}"
REMOTE_DST="${REMOTE_BASE}/src/Data_MLA/Kronos/Crypto/TP/horizon_7/"

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
SRC="${PROJECT_DIR}/src/Data_MLA/Kronos/Crypto/TP/horizon_7/"

if [ ! -d "$SRC" ] || [ -z "$(find "$SRC" -maxdepth 2 -type f -name '*.csv' -print -quit 2>/dev/null)" ]; then
    echo "[$(date)] ERROR: no CSVs found under $SRC"
    echo "  Run slurm/stage-data.sh first (hop 1: gpu8 -> this laptop)."
    exit 1
fi
echo "[$(date)] Local data OK: $(find "$SRC" -name '*.csv' | wc -l) CSVs under $SRC"

echo "[$(date)] Creating ${REMOTE_DST} on ${HOST} ..."
ssh "${GASPAR}@${HOST}" "mkdir -p '${REMOTE_DST}'"

echo "[$(date)] Pushing $SRC -> ${GASPAR}@${HOST}:${REMOTE_DST}"
rsync -avz --stats "$SRC" "${GASPAR}@${HOST}:${REMOTE_DST}"

echo ""
echo "[$(date)] Done. Remote data root for config.yaml / the SLURM scripts: ${REMOTE_BASE}"
echo "Reminder: Izar's /scratch auto-deletes files older than 30 days (SCITAS policy)"
echo "-- fine for an active pilot, don't treat it as durable storage."
