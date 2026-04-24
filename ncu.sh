#!/usr/bin/env bash
set -euo pipefail

cd /home/snu_avq1/workspace/chaewon/flashinfer-bench-starter-kit

PYTHON=/home/snu_avq1/miniconda3/envs/fi-bench/bin/python
export FIB_DATASET_PATH=/home/snu_avq1/workspace/mlsys26-contest
export FIB_MOE_COMM_USE_TMA=1
export FIB_MOE_DIRECT_TMA_SW128=1
export FIB_MOE_DIRECT_B_SW128=1
export FIB_MOE_COMM_H_TILES=8
export FIB_TCGEN_ACCUM=${FIB_TCGEN_ACCUM:-tf32_scale}

WORKLOAD_INDEXES=${WORKLOAD_INDEXES:-1}
REPORT_NAME=${REPORT_NAME:-ex}

if [[ -x /engrid/ensh/gpubin/ctn_gcsudo ]]; then
    NCU_CMD=(/engrid/ensh/gpubin/ctn_gcsudo ncu)
elif command -v gcsudo >/dev/null 2>&1; then
    NCU_CMD=(gcsudo ncu)
else
    NCU_CMD=(ncu)
fi

"${NCU_CMD[@]}" --set full \
    --target-processes all \
    --clock-control none \
    --kernel-name-base function \
    --kernel-name regex:step1_gemm1_swiglu_direct_kernel.* \
    --launch-count 2 \
    -f \
    -o "${REPORT_NAME}" \
    "$@" \
    "${PYTHON}" scripts/run_local_single.py \
        --impl direct \
        --compare-target baseline \
        --warmup-runs 0 \
        --iterations 1 \
        --num-trials 1 \
        --workload-indexes "${WORKLOAD_INDEXES}" \
        --seed 1234

echo "NCU report written to ${REPORT_NAME}.ncu-rep"
