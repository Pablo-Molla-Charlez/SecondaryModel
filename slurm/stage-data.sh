#!/bin/bash
# ============================================================================
# Pull the Kronos/Crypto/TP/horizon_7 dataset from Pablo's gpu8 checkout into
# this repo's gitignored Data_MLA/ (local run only — NOT a SLURM job).
#
# Source confirmed 2026-10-26: gpu8-pablo (136.187.100.86, via the `pablo`
# account, key ~/.ssh/id_ed25519_nii — already in ~/.ssh/config).
# gpu7 was a dead end: elie's and pablo's accounts there are both freshly
# provisioned with nothing staged (/data and /storage empty but lost+found).
#
# This only gets the data onto YOUR machine. EPFL cluster nodes can't reach
# gpu8-pablo with your laptop's keys, so getting it onto the cluster is a
# second hop you do once you have access — see slurm/README.md.
#
# Usage:
#   slurm/stage-data.sh
# ============================================================================

set -euo pipefail

SRC="gpu8-pablo:/home/pablo/M2_DS/SecondaryModel/src/Data_MLA/Kronos/Crypto/TP/horizon_7/"
PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
DST="${PROJECT_DIR}/src/Data_MLA/Kronos/Crypto/TP/horizon_7/"

mkdir -p "$DST"
echo "[$(date)] Pulling $SRC -> $DST"
rsync -avz --stats "$SRC" "$DST"

echo ""
echo "[$(date)] Done: $(find "$DST" -type f | wc -l) files, $(du -sh "$DST" | cut -f1)"
