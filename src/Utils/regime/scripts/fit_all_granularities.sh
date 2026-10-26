#!/usr/bin/env bash
# Fit one pooled global Sparse Jump Model per granularity (K pinned to 2).
#
# K=2 is fixed by design: the fold-composition analysis needs a bull/bear split,
# so BIC searches lambda and kappa only. The lambda grid reproduces the original
# 1d fit (sparse_K2_lam20) rather than the package default, so every granularity
# is selected under the same criterion the paper's 1d fit already used.
#
# Runs coarse -> fine so the low-granularity tables are usable before 1h/30min
# (which have 24x/48x the bars) finish. Sequential by design: each fit already
# saturates its core, and the box is shared with a long-running Kronos forecast.
set -u

SRC=/home/pablo/M2_DS/SecondaryModel/src
PY=/home/pablo/miniconda3/envs/CTTS/bin/python
LOGDIR="$SRC/Output/Jump_Models/_logs"
mkdir -p "$LOGDIR"

# Keep BLAS single-threaded: these fits are run one at a time and the machine is
# shared with a CPU-bound Kronos forecast. Letting each fit spawn 24 BLAS threads
# would starve it for no speedup.
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

# 30min is excluded by default: no CPCV/expanding-CV results exist at that
# granularity to compose against, and its fit alone costs ~2x the other seven
# combined. Pass GRANS explicitly to include it.
GRANS=${GRANS:-"1d 12h 8h 6h 4h 2h 1h"}
LAMBDA_GRID=${LAMBDA_GRID:-"5 10 20 50 100 200"}
# Sparse-BIC reference active-feature count. The package default derives A0 from the
# feature width (0.25*46 -> 12), which rejects 46 of 48 candidates and leaves a nearly
# degenerate search. The paper's 1d fit used A0=3, under which the whole grid competes;
# keep that so every granularity is selected on the same footing as the published fit.
BIC_A0=${BIC_A0:-3}

for gran in $GRANS; do
  log="$LOGDIR/fit_${gran}.log"
  echo "=== [$(date '+%F %T')] fitting granularity=$gran -> $log"

  nice -n 10 "$PY" -m Utils.regime.train_global_jump_model \
      --granularity "$gran" \
      --select-bic --k-grid 2 \
      --lambda-grid $LAMBDA_GRID \
      --bic-a0 "$BIC_A0" \
      --raw-root "$raw_root" \
      > "$log" 2>&1
  status=$?
  if [ $status -eq 0 ]; then
    echo "    [$(date '+%F %T')] OK  $gran -> $(grep -o '\"output_dir\": \"[^\"]*\"' "$log" | tail -1)"
  else
    echo "    [$(date '+%F %T')] FAILED $gran (exit $status); see $log"
  fi
done
echo "=== [$(date '+%F %T')] all granularities done"
