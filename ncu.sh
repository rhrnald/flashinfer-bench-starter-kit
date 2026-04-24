#!/usr/bin/env bash
set -euo pipefail

cd /home/snu_avq1/workspace/chaewon/flashinfer-bench-starter-kit

PYTHON=/home/snu_avq1/miniconda3/envs/fi-bench/bin/python
export FIB_DATASET_PATH=/home/snu_avq1/workspace/mlsys26-contest

WORKLOAD_INDEXES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18
REPORT_NAME=${1:-ncu_flashinfer_baseline_and_direct_19x1}
if (($# > 0)); then
    shift
fi

ncu --set full \
    --target-processes all \
    --clock-control none \
    -f \
    -o "${REPORT_NAME}" \
    "$@" \
    "${PYTHON}" scripts/run_local_single.py \
        --impl direct \
        --compare-target baseline \
        --warmup-runs 0 \
        --iterations 1 \
        --num-trials 1 \
        --correctness-only \
        --workload-indexes "${WORKLOAD_INDEXES}" \
        --seed 1234

echo "NCU report written to ${REPORT_NAME}.ncu-rep"
