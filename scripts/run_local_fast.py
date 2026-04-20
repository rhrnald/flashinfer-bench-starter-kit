"""
FlashInfer-Bench Local Fast Benchmark Runner.

Packs solution from source and runs a quick local benchmark with reduced
iterations/workloads for rapid iteration.
"""

import os
import sys
from pathlib import Path

import torch

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from flashinfer_bench import Benchmark, BenchmarkConfig, Solution, TraceSet
from scripts.pack_solution import pack_solution


def ensure_tvm_ffi_cuda_arch(solution: Solution) -> None:
    """Force TVM_FFI_CUDA_ARCH_LIST from current visible GPU."""
    if solution.spec.language.value != "cuda":
        return
    if not torch.cuda.is_available() or torch.cuda.device_count() == 0:
        return
    major, minor = torch.cuda.get_device_capability(0)
    # Blackwell family kernels in this repo require the "a" target
    # (compute_100a/sm_100a), not plain sm_100.
    suffix = "a" if major >= 10 else ""
    arch = f"{major}.{minor}{suffix}"
    prev = os.environ.get("TVM_FFI_CUDA_ARCH_LIST")
    if prev and prev != arch:
        print(f"Overriding TVM_FFI_CUDA_ARCH_LIST={prev} -> {arch}")
    os.environ["TVM_FFI_CUDA_ARCH_LIST"] = arch
    print(f"Using TVM_FFI_CUDA_ARCH_LIST={arch}")


def ensure_tvm_ffi_link_cuda_driver(solution: Solution) -> None:
    """Ensure tvm_ffi links against CUDA driver symbols (e.g. cuTensorMapEncodeTiled)."""
    if solution.spec.language.value != "cuda":
        return
    try:
        import tvm_ffi.cpp  # type: ignore
    except Exception as exc:
        print(f"Skip tvm_ffi -lcuda patch: {exc}")
        return

    if getattr(tvm_ffi.cpp.build, "_fib_lcuda_patched", False):
        return

    original_build = tvm_ffi.cpp.build

    def build_with_cuda_driver(*args, **kwargs):
        ldflags = list(kwargs.get("extra_ldflags") or [])
        for flag in ("-L/usr/local/cuda/lib64/stubs", "-lcuda"):
            if flag not in ldflags:
                ldflags.append(flag)
        kwargs["extra_ldflags"] = ldflags
        return original_build(*args, **kwargs)

    build_with_cuda_driver._fib_lcuda_patched = True  # type: ignore[attr-defined]
    tvm_ffi.cpp.build = build_with_cuda_driver


def get_trace_set_path() -> str:
    """Get trace set path from environment variable."""
    path = os.environ.get("FIB_DATASET_PATH")
    if not path:
        raise EnvironmentError(
            "FIB_DATASET_PATH environment variable not set. "
            "Please set it to the path of your flashinfer-trace dataset."
        )
    return path


def run_benchmark(solution: Solution) -> dict:
    """Run benchmark locally in fast mode and return results."""
    config = BenchmarkConfig(warmup_runs=1, iterations=10, num_trials=1)
    max_workloads = 3

    trace_set_path = get_trace_set_path()
    trace_set = TraceSet.from_path(trace_set_path)

    if solution.definition not in trace_set.definitions:
        raise ValueError(f"Definition '{solution.definition}' not found in trace set")

    definition = trace_set.definitions[solution.definition]
    workloads = trace_set.workloads.get(solution.definition, [])[:max_workloads]

    if not workloads:
        raise ValueError(f"No workloads found for definition '{solution.definition}'")

    bench_trace_set = TraceSet(
        root=trace_set.root,
        definitions={definition.name: definition},
        solutions={definition.name: [solution]},
        workloads={definition.name: workloads},
        traces={definition.name: []},
    )

    benchmark = Benchmark(bench_trace_set, config)
    result_trace_set = benchmark.run_all(dump_traces=True)

    traces = result_trace_set.traces.get(definition.name, [])
    results = {definition.name: {}}

    for trace in traces:
        if trace.evaluation:
            entry = {
                "status": trace.evaluation.status.value,
                "solution": trace.solution,
            }
            if trace.evaluation.performance:
                entry["latency_ms"] = trace.evaluation.performance.latency_ms
                entry["reference_latency_ms"] = trace.evaluation.performance.reference_latency_ms
                entry["speedup_factor"] = trace.evaluation.performance.speedup_factor
            if trace.evaluation.correctness:
                entry["max_abs_error"] = trace.evaluation.correctness.max_absolute_error
                entry["max_rel_error"] = trace.evaluation.correctness.max_relative_error
            results[definition.name][trace.workload.uuid] = entry

    return results


def print_results(results: dict):
    """Print benchmark results in a formatted way."""
    for def_name, traces in results.items():
        print(f"\n{def_name}:")
        for workload_uuid, result in traces.items():
            status = result.get("status")
            print(f"  Workload {workload_uuid[:8]}...: {status}", end="")

            if result.get("latency_ms") is not None:
                print(f" | {result['latency_ms']:.3f} ms", end="")

            if result.get("speedup_factor") is not None:
                print(f" | {result['speedup_factor']:.2f}x speedup", end="")

            if result.get("max_abs_error") is not None:
                abs_err = result["max_abs_error"]
                rel_err = result.get("max_rel_error", 0)
                print(f" | abs_err={abs_err:.2e}, rel_err={rel_err:.2e}", end="")

            print()


def main():
    """Pack solution and run fast benchmark."""
    print("Packing solution from source files...")
    solution_path = pack_solution()

    print("\nLoading solution...")
    solution = Solution.model_validate_json(solution_path.read_text())
    print(f"Loaded: {solution.name} ({solution.definition})")
    ensure_tvm_ffi_cuda_arch(solution)
    ensure_tvm_ffi_link_cuda_driver(solution)

    print("\nRunning benchmark (fast mode: warmup=1, iter=10, trials=1, workloads=3)...")
    results = run_benchmark(solution)

    if not results:
        print("No results returned!")
        return

    print_results(results)


if __name__ == "__main__":
    main()
