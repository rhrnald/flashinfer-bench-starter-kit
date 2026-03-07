"""
FlashInfer-Bench local single-process runner.

This runner avoids Benchmark's multiprocessing workers and evaluates
reference/solution directly in the current process.
"""

import argparse
import os
import sys
import time
import traceback
from pathlib import Path

import torch

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from flashinfer_bench import BenchmarkConfig, Solution, TraceSet
from flashinfer_bench.bench.evaluators import resolve_evaluator
from flashinfer_bench.bench.utils import make_eval
from flashinfer_bench.compile import BuildError, BuilderRegistry
from flashinfer_bench.data import EvaluationStatus
from scripts.pack_solution import pack_solution


def ensure_tvm_ffi_cuda_arch(solution: Solution) -> None:
    """Set TVM_FFI_CUDA_ARCH_LIST from current visible GPU when not explicitly set."""
    if solution.spec.language.value != "cuda":
        return
    if os.environ.get("TVM_FFI_CUDA_ARCH_LIST"):
        return
    if not torch.cuda.is_available() or torch.cuda.device_count() == 0:
        return
    major, minor = torch.cuda.get_device_capability(0)
    arch = f"{major}.{minor}"
    os.environ["TVM_FFI_CUDA_ARCH_LIST"] = arch
    print(f"Using TVM_FFI_CUDA_ARCH_LIST={arch}")


def get_trace_set_path() -> str:
    path = os.environ.get("FIB_DATASET_PATH")
    if not path:
        raise EnvironmentError(
            "FIB_DATASET_PATH environment variable not set. "
            "Please set it to the path of your flashinfer-trace dataset."
        )
    return path


def run_single(solution: Solution, cfg: BenchmarkConfig, max_workloads: int, device: str) -> dict:
    trace_set = TraceSet.from_path(get_trace_set_path())
    if solution.definition not in trace_set.definitions:
        raise ValueError(f"Definition '{solution.definition}' not found in trace set")

    definition = trace_set.definitions[solution.definition]
    workloads = trace_set.workloads.get(solution.definition, [])[:max_workloads]
    if not workloads:
        raise ValueError(f"No workloads found for definition '{solution.definition}'")

    evaluator_cls = resolve_evaluator(definition)
    registry = BuilderRegistry.get_instance()

    try:
        sol_runnable = registry.build(definition, solution)
    except BuildError:
        # Mirror bench behavior: if build fails, mark all workloads compile error.
        results = {definition.name: {}}
        for w in workloads:
            results[definition.name][w.workload.uuid] = {
                "status": EvaluationStatus.COMPILE_ERROR.value,
                "solution": solution.name,
            }
        return results

    results = {definition.name: {}}

    for w in workloads:
        workload = w.workload
        log_path = str(Path(cfg.log_dir) / f"{solution.name}_{workload.uuid}_{int(time.time())}.log")
        try:
            baseline = evaluator_cls.build_baseline(
                definition=definition,
                workload=workload,
                cfg=cfg,
                device=device,
                trace_set_root=trace_set.root,
            )
            evaluation = evaluator_cls.evaluate(
                definition=definition,
                sol_runnable=sol_runnable,
                inputs=baseline.inputs,
                ref_outputs=baseline.outputs,
                ref_mean_latency_ms=baseline.mean_latency_ms,
                cfg=cfg,
                log_path=log_path,
                device=device,
            )
        except BuildError as e:
            print(f"[{solution.name}] BuildError on workload {workload.uuid}: {e}", file=sys.stderr)
            evaluation = make_eval(
                status=EvaluationStatus.COMPILE_ERROR,
                device=device,
                log_path=log_path,
            )
        except Exception as e:
            print(
                f"[{solution.name}] Runtime exception on workload {workload.uuid}: {type(e).__name__}: {e}",
                file=sys.stderr,
            )
            print(traceback.format_exc(), file=sys.stderr)
            evaluation = make_eval(
                status=EvaluationStatus.RUNTIME_ERROR,
                device=device,
                log_path=log_path,
            )

        entry = {
            "status": evaluation.status.value,
            "solution": solution.name,
        }

        if evaluation.performance is not None:
            entry["latency_ms"] = evaluation.performance.latency_ms
            entry["reference_latency_ms"] = evaluation.performance.reference_latency_ms
            entry["speedup_factor"] = evaluation.performance.speedup_factor

        if evaluation.correctness is not None:
            entry["max_abs_error"] = evaluation.correctness.max_absolute_error
            entry["max_rel_error"] = evaluation.correctness.max_relative_error

        results[definition.name][workload.uuid] = entry

    return results


def print_results(results: dict):
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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Single-process local benchmark runner")
    p.add_argument("--warmup-runs", type=int, default=1)
    p.add_argument("--iterations", type=int, default=10)
    p.add_argument("--num-trials", type=int, default=1)
    p.add_argument("--max-workloads", type=int, default=3)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--rtol", type=float, default=1e-2)
    p.add_argument("--atol", type=float, default=1e-2)
    return p.parse_args()


def main():
    args = parse_args()

    print("Packing solution from source files...")
    solution_path = pack_solution()

    print("\nLoading solution...")
    solution = Solution.model_validate_json(solution_path.read_text())
    print(f"Loaded: {solution.name} ({solution.definition})")
    ensure_tvm_ffi_cuda_arch(solution)

    cfg = BenchmarkConfig(
        warmup_runs=args.warmup_runs,
        iterations=args.iterations,
        num_trials=args.num_trials,
        rtol=float(args.rtol),
        atol=float(args.atol),
    )

    print(
        "\nRunning benchmark (single-process mode: "
        f"warmup={args.warmup_runs}, iter={args.iterations}, "
        f"trials={args.num_trials}, workloads={args.max_workloads}, device={args.device})..."
    )

    results = run_single(solution, cfg, args.max_workloads, args.device)
    print_results(results)


if __name__ == "__main__":
    main()
