"""
FlashInfer-Bench Modal Cloud Benchmark Runner.

Automatically packs the solution from source files and runs benchmarks
on a Modal GPU (B200 by default).

Setup (one-time):
    modal setup
    modal volume create flashinfer-trace
    modal volume put flashinfer-trace /path/to/flashinfer-trace/

Usage:
    # Full B200 benchmark (default: all workloads, warmup=3, iter=100, trials=5):
    modal run scripts/run_modal.py

    # Smoke test on B200 (1 workload, minimal iters, for compile + correctness check):
    modal run scripts/run_modal.py -- --gpu B200 --max-workloads 1 \\
        --warmup-runs 1 --iterations 3 --num-trials 1

    # Cheaper correctness sweep on A10G:
    modal run scripts/run_modal.py -- --gpu A10G --max-workloads 3 --iterations 10
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import modal

app = modal.App("flashinfer-bench")

trace_volume = modal.Volume.from_name("flashinfer-trace", create_if_missing=True)
TRACE_SET_PATH = "/data"

# CUDA 12.8+ devel image: ships nvcc and headers (flashinfer-bench's CUDA
# builder shells out to nvcc). 12.8 is the minimum that supports sm_100 (B200).
image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.12"
    )
    .env({"CUDA_HOME": "/usr/local/cuda"})
    .pip_install("flashinfer-bench", "torch", "triton", "numpy")
)


def _ensure_cuda_arch() -> None:
    """Set TVM_FFI_CUDA_ARCH_LIST from the container's visible GPU.

    flashinfer-bench's TVM-FFI builder reads this to pick which SM to compile
    for. Without it the build falls back to a list that may not include sm_100.
    """
    import os

    if os.environ.get("TVM_FFI_CUDA_ARCH_LIST"):
        return
    import torch

    if not torch.cuda.is_available() or torch.cuda.device_count() == 0:
        return
    major, minor = torch.cuda.get_device_capability(0)
    # Blackwell TCGEN05/CUTLASS SM100 kernels require the family-specific
    # "a" target (compute_100a/sm_100a), not plain sm_100.
    suffix = "a" if major >= 10 else ""
    os.environ["TVM_FFI_CUDA_ARCH_LIST"] = f"{major}.{minor}{suffix}"
    print(f"[run_modal] TVM_FFI_CUDA_ARCH_LIST={major}.{minor}{suffix}")


def _ensure_flashinfer_include_paths() -> None:
    """Expose FlashInfer's vendored headers to TVM-FFI/nvcc.

    The contest CUDA builder does not process Solution.spec.dependencies for
    tvm-ffi builds. For experiments that directly include FlashInfer's CUTLASS
    SM100 groupwise GEMM headers, CPATH is the least invasive way to make the
    vendored `flashinfer/` and `cutlass/` headers visible.
    """
    import os
    from pathlib import Path

    try:
        import flashinfer
    except Exception as exc:
        print(f"[run_modal] flashinfer import failed; CUTLASS headers unavailable: {exc}")
        return

    base = Path(flashinfer.__file__).resolve().parent / "data"
    paths = [
        base / "include",
        base / "cutlass" / "include",
        base / "cutlass" / "tools" / "util" / "include",
    ]
    existing = [p for p in paths if p.exists()]
    if not existing:
        return

    for var in ("CPATH", "CPLUS_INCLUDE_PATH"):
        prior = os.environ.get(var, "")
        pieces = [str(p) for p in existing]
        if prior:
            pieces.append(prior)
        os.environ[var] = ":".join(pieces)
    print("[run_modal] FlashInfer include paths:", ", ".join(str(p) for p in existing))

    import tvm_ffi.cpp

    if getattr(tvm_ffi.cpp.build, "_fib_flashinfer_patched", False):
        return

    original_build = tvm_ffi.cpp.build

    def build_with_flashinfer_headers(*args, **kwargs):
        inc = list(kwargs.get("extra_include_paths") or [])
        for p in existing:
            sp = str(p)
            if sp not in inc:
                inc.append(sp)
        kwargs["extra_include_paths"] = inc

        cuda_flags = list(kwargs.get("extra_cuda_cflags") or [])
        for flag in (
            "-DCUTLASS_ENABLE_GDC_FOR_SM100=1",
            "-DFLASHINFER_ENABLE_FP8_E8M0",
            "-DFLASHINFER_ENABLE_FP4_E2M1",
        ):
            if flag not in cuda_flags:
                cuda_flags.append(flag)
        kwargs["extra_cuda_cflags"] = cuda_flags
        return original_build(*args, **kwargs)

    build_with_flashinfer_headers._fib_flashinfer_patched = True
    tvm_ffi.cpp.build = build_with_flashinfer_headers


def _extract_results(definition_name, result_trace_set) -> dict:
    traces = result_trace_set.traces.get(definition_name, [])
    results = {definition_name: {}}
    for trace in traces:
        if not trace.evaluation:
            continue
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
        results[definition_name][trace.workload.uuid] = entry
    return results


def _apply_env(env: dict) -> dict:
    """Set env vars, return snapshot of previous values for restore."""
    import os

    saved = {}
    for k, v in (env or {}).items():
        saved[k] = os.environ.get(k)
        os.environ[k] = v
    return saved


def _restore_env(env: dict, saved: dict) -> None:
    import os

    for k in (env or {}):
        prior = saved.get(k)
        if prior is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = prior


def _run_impl(solution: Solution, config: BenchmarkConfig, max_workloads: int,
              env_vars: dict | None = None) -> dict:
    """Shared benchmark body. `max_workloads=0` means all workloads.

    `env_vars` is applied before the build+run so runtime-read env flags like
    `FIB_MOE_GROUPED` / `FIB_MOE_PROFILE` reach the CUDA kernel process.
    """
    from flashinfer_bench import Benchmark, TraceSet
    from flashinfer_bench.compile import BuildError, BuilderRegistry

    _ensure_cuda_arch()
    _ensure_flashinfer_include_paths()
    saved_env = _apply_env(env_vars or {})

    try:
        trace_set = TraceSet.from_path(TRACE_SET_PATH)

        if solution.definition not in trace_set.definitions:
            raise ValueError(f"Definition '{solution.definition}' not found in trace set")

        definition = trace_set.definitions[solution.definition]
        workloads = trace_set.workloads.get(solution.definition, [])
        if max_workloads > 0:
            workloads = workloads[:max_workloads]

        if not workloads:
            raise ValueError(f"No workloads found for definition '{solution.definition}'")

        # Pre-build once so compile failures surface with the real nvcc/compiler
        # message instead of an opaque COMPILE_ERROR status from Benchmark's workers.
        try:
            BuilderRegistry.get_instance().build(definition, solution)
        except BuildError as e:
            return {
                definition.name: {
                    "__build_error__": {
                        "status": "COMPILE_ERROR",
                        "error": str(e),
                    }
                }
            }

        bench_trace_set = TraceSet(
            root=trace_set.root,
            definitions={definition.name: definition},
            solutions={definition.name: [solution]},
            workloads={definition.name: workloads},
            traces={definition.name: []},
        )

        benchmark = Benchmark(bench_trace_set, config)
        result_trace_set = benchmark.run_all(dump_traces=True)
        return _extract_results(definition.name, result_trace_set)
    finally:
        _restore_env(env_vars or {}, saved_env)


def _run_sweep_impl(solution: Solution, config: BenchmarkConfig, max_workloads: int,
                    variants: list) -> dict:
    """Run many env-var variants in a single container so nvcc compiles once.

    `variants` is a list of {"name": str, "env": dict}. Returns
    {variant_name: {definition_name: {workload_uuid: entry, ...}}}.
    """
    from flashinfer_bench import Benchmark, TraceSet
    from flashinfer_bench.compile import BuildError, BuilderRegistry

    _ensure_cuda_arch()
    _ensure_flashinfer_include_paths()

    trace_set = TraceSet.from_path(TRACE_SET_PATH)
    if solution.definition not in trace_set.definitions:
        raise ValueError(f"Definition '{solution.definition}' not found in trace set")

    definition = trace_set.definitions[solution.definition]
    workloads = trace_set.workloads.get(solution.definition, [])
    if max_workloads > 0:
        workloads = workloads[:max_workloads]
    if not workloads:
        raise ValueError(f"No workloads found for definition '{solution.definition}'")

    try:
        BuilderRegistry.get_instance().build(definition, solution)
    except BuildError as e:
        return {
            "__build_error__": {
                "status": "COMPILE_ERROR",
                "error": str(e),
            }
        }

    bench_trace_set = TraceSet(
        root=trace_set.root,
        definitions={definition.name: definition},
        solutions={definition.name: [solution]},
        workloads={definition.name: workloads},
        traces={definition.name: []},
    )

    all_results = {}
    for variant in variants:
        name = variant["name"]
        env = variant.get("env", {})
        saved = _apply_env(env)
        try:
            # Re-instantiate per variant so any worker-side state gets a clean start.
            benchmark = Benchmark(bench_trace_set, config)
            result_trace_set = benchmark.run_all(dump_traces=True)
            all_results[name] = _extract_results(definition.name, result_trace_set)
        finally:
            _restore_env(env, saved)
    return all_results


# Modal requires `@app.function` at module scope, so GPU cannot be overridden
# via CLI on a single function. Register one runner per supported GPU type and
# dispatch in `main()`. Add a new entry here if you need another GPU.
_COMMON = dict(image=image, timeout=3600, volumes={TRACE_SET_PATH: trace_volume})


@app.function(**_COMMON, gpu="B200:1")
def run_b200(solution: Solution, config: BenchmarkConfig, max_workloads: int,
             env_vars: dict | None = None) -> dict:
    return _run_impl(solution, config, max_workloads, env_vars)


@app.function(**_COMMON, gpu="H100:1")
def run_h100(solution: Solution, config: BenchmarkConfig, max_workloads: int,
             env_vars: dict | None = None) -> dict:
    return _run_impl(solution, config, max_workloads, env_vars)


@app.function(**_COMMON, gpu="A100:1")
def run_a100(solution: Solution, config: BenchmarkConfig, max_workloads: int,
             env_vars: dict | None = None) -> dict:
    return _run_impl(solution, config, max_workloads, env_vars)


@app.function(**_COMMON, gpu="A10G:1")
def run_a10g(solution: Solution, config: BenchmarkConfig, max_workloads: int,
             env_vars: dict | None = None) -> dict:
    return _run_impl(solution, config, max_workloads, env_vars)


@app.function(**_COMMON, gpu="L4:1")
def run_l4(solution: Solution, config: BenchmarkConfig, max_workloads: int,
           env_vars: dict | None = None) -> dict:
    return _run_impl(solution, config, max_workloads, env_vars)


# Sweep variants share one container so nvcc compile cost (~1–2 min) is paid once.
@app.function(**_COMMON, gpu="B200:1")
def sweep_b200(solution: Solution, config: BenchmarkConfig, max_workloads: int,
               variants: list) -> dict:
    return _run_sweep_impl(solution, config, max_workloads, variants)


@app.function(**_COMMON, gpu="H100:1")
def sweep_h100(solution: Solution, config: BenchmarkConfig, max_workloads: int,
               variants: list) -> dict:
    return _run_sweep_impl(solution, config, max_workloads, variants)


@app.function(**_COMMON, gpu="A100:1")
def sweep_a100(solution: Solution, config: BenchmarkConfig, max_workloads: int,
               variants: list) -> dict:
    return _run_sweep_impl(solution, config, max_workloads, variants)


@app.function(**_COMMON, gpu="A10G:1")
def sweep_a10g(solution: Solution, config: BenchmarkConfig, max_workloads: int,
               variants: list) -> dict:
    return _run_sweep_impl(solution, config, max_workloads, variants)


@app.function(**_COMMON, gpu="L4:1")
def sweep_l4(solution: Solution, config: BenchmarkConfig, max_workloads: int,
             variants: list) -> dict:
    return _run_sweep_impl(solution, config, max_workloads, variants)


_GPU_RUNNERS = {
    "B200": run_b200,
    "H100": run_h100,
    "A100": run_a100,
    "A10G": run_a10g,
    "L4": run_l4,
}

_GPU_SWEEPERS = {
    "B200": sweep_b200,
    "H100": sweep_h100,
    "A100": sweep_a100,
    "A10G": sweep_a10g,
    "L4": sweep_l4,
}


def print_results(results: dict):
    """Print benchmark results in a formatted way."""
    for def_name, traces in results.items():
        print(f"\n{def_name}:")
        build_err = traces.get("__build_error__")
        if build_err is not None:
            print(f"  COMPILE_ERROR:\n{build_err['error']}")
            continue
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


@app.local_entrypoint()
def main(
    gpu: str = "B200",
    max_workloads: int = 0,
    warmup_runs: int = 3,
    iterations: int = 100,
    num_trials: int = 5,
):
    """Pack solution and run benchmark on Modal.

    Args are exposed as CLI flags via `modal run ... -- --flag value`.
    `max_workloads=0` means "all workloads".
    """
    from scripts.pack_solution import pack_solution
    from flashinfer_bench import BenchmarkConfig, Solution

    print("Packing solution from source files...")
    solution_path = pack_solution()

    print("\nLoading solution...")
    solution = Solution.model_validate_json(solution_path.read_text())
    print(f"Loaded: {solution.name} ({solution.definition})")

    config = BenchmarkConfig(
        warmup_runs=warmup_runs,
        iterations=iterations,
        num_trials=num_trials,
    )

    workloads_label = "all" if max_workloads == 0 else str(max_workloads)
    print(
        f"\nRunning on Modal {gpu} (workloads={workloads_label}, "
        f"warmup={warmup_runs}, iter={iterations}, trials={num_trials})..."
    )

    gpu_key = gpu.upper()
    if gpu_key not in _GPU_RUNNERS:
        raise ValueError(
            f"Unsupported --gpu {gpu!r}; choose from {sorted(_GPU_RUNNERS)}"
        )
    fn = _GPU_RUNNERS[gpu_key]
    results = fn.remote(solution, config, max_workloads)

    if not results:
        print("No results returned!")
        return

    print_results(results)
