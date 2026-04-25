#!/usr/bin/env bash
set -euo pipefail

cd /home/snu_avq1/workspace/chaewon/flashinfer-bench-starter-kit

FIB_DATASET_PATH=${FIB_DATASET_PATH:-/home/snu_avq1/workspace/mlsys26-contest}
WARMUP_RUNS=${WARMUP_RUNS:-1}
OUTPUT_PREFIX=${OUTPUT_PREFIX:-ncu_baseline}
WORKLOAD_INDEXES=${WORKLOAD_INDEXES:-all}
PYTHON_BIN=${PYTHON_BIN:-/home/snu_avq1/miniconda3/envs/fi-bench/bin/python}

export FIB_DATASET_PATH

timestamp=$(date +%Y%m%d_%H%M%S)
report_dir="${OUTPUT_PREFIX}_${timestamp}"
mkdir -p "${report_dir}"

workload_lines=$(
  "${PYTHON_BIN}" - <<'PY'
from flashinfer_bench import TraceSet
from scripts.run_local_single import DEFAULT_COMPARE_SOLUTION, get_trace_set_path, load_solution_json

solution = load_solution_json(DEFAULT_COMPARE_SOLUTION)
trace = TraceSet.from_path(get_trace_set_path())
workloads = trace.workloads[solution.definition]

for idx, entry in enumerate(workloads):
    print(f"{idx} {entry.workload.uuid}")
PY
)

if [[ "${WORKLOAD_INDEXES}" == "all" ]]; then
  selected_lines="${workload_lines}"
else
  selected_lines=$(printf '%s\n' "${workload_lines}" | awk -v wanted="${WORKLOAD_INDEXES}" '
    BEGIN {
      split(wanted, arr, ",")
      for (i in arr) keep[arr[i]] = 1
    }
    keep[$1] { print }
  ')
fi

printf '%s\n' "${selected_lines}" > "${report_dir}/workloads.txt"

while read -r workload_index workload_uuid; do
  [[ -z "${workload_index:-}" ]] && continue
  short_uuid=${workload_uuid%%-*}
  output_base="${report_dir}/w${workload_index}_${short_uuid}"

  echo "[ncu_baseline] profiling workload ${workload_index} ${workload_uuid}"

  WORKLOAD_INDEX="${workload_index}" \
  WARMUP_RUNS="${WARMUP_RUNS}" \
  ncu --set full \
      --target-processes all \
      --profile-from-start off \
      --clock-control none \
      -o "${output_base}" \
      "${PYTHON_BIN}" - <<'PY'
import os
import torch

from flashinfer_bench import BenchmarkConfig, TraceSet
from flashinfer_bench.bench.evaluators import resolve_evaluator
from flashinfer_bench.bench.evaluators.utils import allocate_outputs, normalize_result
from flashinfer_bench.compile import BuilderRegistry
from scripts.run_local_single import (
    DEFAULT_COMPARE_SOLUTION,
    ensure_tvm_ffi_cuda_arch,
    ensure_tvm_ffi_link_cuda_driver,
    get_trace_set_path,
    load_solution_json,
    set_reproducible_seed,
)


def run_once(runnable, definition, inputs, device):
    if runnable.metadata.destination_passing_style:
        out = allocate_outputs(definition, inputs, device)
        with torch.no_grad():
            runnable(*inputs, *out)
    else:
        with torch.no_grad():
            result = runnable(*inputs)
        normalize_result(definition, result, device)
    torch.cuda.synchronize(device)


def main():
    device = "cuda:0"
    workload_index = int(os.environ["WORKLOAD_INDEX"])
    warmup_runs = int(os.environ["WARMUP_RUNS"])

    set_reproducible_seed(1234)

    solution = load_solution_json(DEFAULT_COMPARE_SOLUTION)
    ensure_tvm_ffi_cuda_arch(solution)
    ensure_tvm_ffi_link_cuda_driver(solution)

    trace_set = TraceSet.from_path(get_trace_set_path())
    definition = trace_set.definitions[solution.definition]
    workload = trace_set.workloads[solution.definition][workload_index].workload
    evaluator_cls = resolve_evaluator(definition)

    cfg = BenchmarkConfig(
        warmup_runs=0,
        iterations=1,
        num_trials=1,
        profile_baseline=False,
    )

    baseline = evaluator_cls.build_baseline(
        definition=definition,
        workload=workload,
        cfg=cfg,
        device=device,
        trace_set_root=trace_set.root,
    )

    runnable = BuilderRegistry.get_instance().build(definition, solution)

    for _ in range(warmup_runs):
        for inp in baseline.inputs:
            run_once(runnable, definition, inp, device)

    cudart = torch.cuda.cudart()
    cudart.cudaProfilerStart()
    for inp in baseline.inputs:
        run_once(runnable, definition, inp, device)
    cudart.cudaProfilerStop()


if __name__ == "__main__":
    main()
PY
done <<< "${selected_lines}"

tar -czf "${report_dir}.tar.gz" "${report_dir}"
echo "[ncu_baseline] archive written to ${report_dir}.tar.gz"
