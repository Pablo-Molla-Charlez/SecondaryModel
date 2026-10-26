#!/usr/bin/env bash
# Re-run CPCV + expanding-window CV for Kronos/RF to emit the per-sample CSVs.
#
# The runners already computed per-sample dates/probs/returns but only fed them to
# the plots; cpcv_samples.csv / cvexpand_samples.csv persist them so the regime
# composition analysis can join each evaluated sample to a dated regime label
# without re-running the CV again. Metrics are unchanged — this reproduces the
# existing sweep and additionally writes the per-sample file.
set -u

SRC=/home/pablo/M2_DS/SecondaryModel/src
PY=/home/pablo/miniconda3/envs/CTTS/bin/python
LOG="$SRC/Output/Analysis/Edge_NoCal/_rerun_logs"
mkdir -p "$LOG"

export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4

UP_CACHE="$SRC/Output/Kronos/cache/multi_kronos_7_fee_up_10533bfac3.pt"
DOWN_CACHE="$SRC/Output/Kronos/cache/multi_kronos_7_fee_down_09fe3f5e4d.pt"
GRANS=${GRANS:-"1d 12h 8h 6h 4h 2h 1h"}

for dir in up down; do
  cache=$UP_CACHE; [ "$dir" = "down" ] && cache=$DOWN_CACHE
  for gran in $GRANS; do
    for mode in cpcv cvexpand; do
      f="$LOG/${mode}_${dir}_${gran}.log"
      echo "=== [$(date '+%F %T')] $mode $dir $gran"
      nice -n 10 "$PY" -m Utils.edge \
        --cache_path "$cache" --config "$SRC/config.yaml" \
        --mode "$mode" --phase edge --m2 rf \
        --direction "$dir" --granularity "$gran" --nocal \
        > "$f" 2>&1 || echo "    FAILED (see $f)"
    done
  done
done
echo "=== [$(date '+%F %T')] rerun complete"
