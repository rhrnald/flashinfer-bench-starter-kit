#!/usr/bin/env bash
set -euo pipefail

cd /home/snu_avq1/workspace/chaewon/flashinfer-bench-starter-kit

PYTHON=${PYTHON:-/home/snu_avq1/miniconda3/envs/fi-bench/bin/python}
export FIB_DATASET_PATH=${FIB_DATASET_PATH:-/home/snu_avq1/workspace/mlsys26-contest}
export FIB_MOE_COMM_USE_TMA=${FIB_MOE_COMM_USE_TMA:-1}
export FIB_MOE_DIRECT_TMA_SW128=${FIB_MOE_DIRECT_TMA_SW128:-1}
export FIB_MOE_DIRECT_B_SW128=${FIB_MOE_DIRECT_B_SW128:-1}
export FIB_TCGEN_ACCUM=${FIB_TCGEN_ACCUM:-tf32_scale}

#WORKLOAD_INDEXES=${WORKLOAD_INDEXES:-0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18}
WORKLOAD_INDEXES=${WORKLOAD_INDEXES:-1}
WARMUP_RUNS=${WARMUP_RUNS:-1}
ITERATIONS=${ITERATIONS:-10}
NUM_TRIALS=${NUM_TRIALS:-1}
SEED=${SEED:-1234}
LOG_FILE=${LOG_FILE:-log_submit_all.txt}
REQUIRED_MATCHED_RATIO=${REQUIRED_MATCHED_RATIO:-0.8}

"${PYTHON}" scripts/run_local_single.py \
    --impl direct \
    --compare-target baseline \
    --warmup-runs "${WARMUP_RUNS}" \
    --iterations "${ITERATIONS}" \
    --num-trials "${NUM_TRIALS}" \
    --workload-indexes "${WORKLOAD_INDEXES}" \
    --seed "${SEED}" \
    --required-matched-ratio "${REQUIRED_MATCHED_RATIO}" \
    2>&1 | tee "${LOG_FILE}"
