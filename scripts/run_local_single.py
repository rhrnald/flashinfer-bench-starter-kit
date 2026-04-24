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
from dataclasses import replace
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


MOE_DEFINITION = "moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048"


def apply_definition_defaults(
    definition_name: str,
    cfg: BenchmarkConfig,
    *,
    rtol: float | None = None,
    atol: float | None = None,
    required_matched_ratio: float | None = None,
) -> BenchmarkConfig:
    if definition_name != MOE_DEFINITION:
        return replace(
            cfg,
            rtol=cfg.rtol if rtol is None else float(rtol),
            atol=cfg.atol if atol is None else float(atol),
            required_matched_ratio=(
                cfg.required_matched_ratio
                if required_matched_ratio is None
                else float(required_matched_ratio)
            ),
        )

    return replace(
        cfg,
        rtol=0.3 if rtol is None else float(rtol),
        atol=1.0 if atol is None else float(atol),
        required_matched_ratio=0.9
        if required_matched_ratio is None
        else float(required_matched_ratio),
    )


def ensure_flashinfer_cutlass_sm100_group_scheduler_patch(cutlass_include: Path) -> None:
    """Patch FlashInfer's vendored CUTLASS SM100 grouped scheduler callback bug."""
    header = cutlass_include / "cutlass" / "gemm" / "kernel" / "sm100_tile_scheduler_group.hpp"
    if not header.exists():
        return

    text = header.read_text()
    if "struct IdentityCallback" in text:
        return

    original = text
    text = text.replace(
        "  using CLCResponse = WorkTileInfo;\n  \n"
        "  static constexpr bool IsDynamicPersistent = UnderlyingScheduler::IsDynamicPersistent;\n",
        "  using CLCResponse = WorkTileInfo;\n\n"
        "  struct IdentityCallback {\n"
        "    CUTLASS_DEVICE\n"
        "    WorkTileInfo operator()(WorkTileInfo info) const {\n"
        "      return info;\n"
        "    }\n"
        "  };\n"
        "  \n"
        "  static constexpr bool IsDynamicPersistent = UnderlyingScheduler::IsDynamicPersistent;\n",
    )
    text = text.replace(
        "  template <typename ClusterShape, typename CallbackBeforeCommit = WorkTileInfo(*)(WorkTileInfo)>\n"
        "  CUTLASS_DEVICE\n"
        "  auto\n"
        "  initial_work_tile_info(ClusterShape cluster_shape, CallbackBeforeCommit callback_before_commit = [] (WorkTileInfo info) { return info;}) {\n",
        "  template <typename ClusterShape, typename CallbackBeforeCommit = IdentityCallback>\n"
        "  CUTLASS_DEVICE\n"
        "  auto\n"
        "  initial_work_tile_info(ClusterShape cluster_shape, CallbackBeforeCommit callback_before_commit = {}) {\n",
    )
    text = text.replace(
        "  template <typename CLCPipeline, typename CLCPipelineState, typename CallbackBeforeCommit = WorkTileInfo(*)(WorkTileInfo)>\n"
        "  CUTLASS_DEVICE\n"
        "  auto\n"
        "  advance_to_next_work(\n"
        "    CLCPipeline& clc_pipeline,\n"
        "    CLCPipelineState clc_pipe_producer_state,\n"
        "    uint32_t advance_count = 1,\n"
        "    CallbackBeforeCommit callback_before_commit = [] (WorkTileInfo info) { return info;}) {\n",
        "  template <typename CLCPipeline, typename CLCPipelineState, typename CallbackBeforeCommit = IdentityCallback>\n"
        "  CUTLASS_DEVICE\n"
        "  auto\n"
        "  advance_to_next_work(\n"
        "    CLCPipeline& clc_pipeline,\n"
        "    CLCPipelineState clc_pipe_producer_state,\n"
        "    uint32_t advance_count = 1,\n"
        "    CallbackBeforeCommit callback_before_commit = {}) {\n",
    )

    if text == original:
        print(
            f"[run_local_single] CUTLASS SM100 grouped scheduler patch not applied: unexpected header shape at {header}",
            file=sys.stderr,
        )
        return
    header.write_text(text)


def ensure_tvm_ffi_cuda_arch(solution: Solution) -> None:
    """Set TVM_FFI_CUDA_ARCH_LIST from current visible GPU when not explicitly set."""
    if solution.spec.language.value != "cuda":
        return
    if os.environ.get("TVM_FFI_CUDA_ARCH_LIST"):
        return
    if not torch.cuda.is_available() or torch.cuda.device_count() == 0:
        return
    major, minor = torch.cuda.get_device_capability(0)
    suffix = "a" if major >= 10 else ""
    arch = f"{major}.{minor}{suffix}"
    os.environ["TVM_FFI_CUDA_ARCH_LIST"] = arch
    print(f"Using TVM_FFI_CUDA_ARCH_LIST={arch}")


def ensure_cutlass_include_paths() -> None:
    """Expose FlashInfer's vendored CUTLASS headers to local TVM-FFI builds."""
    try:
        import flashinfer
        import tvm_ffi.cpp
    except Exception as exc:
        print(f"[run_local_single] CUTLASS include setup skipped: {exc}", file=sys.stderr)
        return

    base = Path(flashinfer.__file__).resolve().parent / "data"
    paths = [
        base / "include",
        base / "cutlass" / "include",
        base / "cutlass" / "tools" / "util" / "include",
    ]
    existing = [str(p) for p in paths if p.exists()]
    if not existing:
        return
    ensure_flashinfer_cutlass_sm100_group_scheduler_patch(base / "cutlass" / "include")

    for var in ("CPATH", "CPLUS_INCLUDE_PATH"):
        prior = os.environ.get(var, "")
        pieces = existing + ([prior] if prior else [])
        os.environ[var] = ":".join(pieces)

    if getattr(tvm_ffi.cpp.build, "_fib_cutlass_patched", False):
        return

    original_build = tvm_ffi.cpp.build

    def build_with_cutlass_headers(*args, **kwargs):
        inc = list(kwargs.get("extra_include_paths") or [])
        for path in existing:
            if path not in inc:
                inc.append(path)
        kwargs["extra_include_paths"] = inc

        cuda_flags = list(kwargs.get("extra_cuda_cflags") or [])
        for flag in ("-DCUTLASS_ENABLE_GDC_FOR_SM100=1",):
            if flag not in cuda_flags:
                cuda_flags.append(flag)
        kwargs["extra_cuda_cflags"] = cuda_flags
        return original_build(*args, **kwargs)

    build_with_cutlass_headers._fib_cutlass_patched = True
    tvm_ffi.cpp.build = build_with_cutlass_headers


def get_trace_set_path() -> str:
    path = os.environ.get("FIB_DATASET_PATH")
    if not path:
        raise EnvironmentError(
            "FIB_DATASET_PATH environment variable not set. "
            "Please set it to the path of your flashinfer-trace dataset."
        )
    return path


def run_single(
    solution: Solution,
    cfg: BenchmarkConfig,
    max_workloads: int,
    device: str,
    workload_start: int = 0,
) -> dict:
    trace_set = TraceSet.from_path(get_trace_set_path())
    if solution.definition not in trace_set.definitions:
        raise ValueError(f"Definition '{solution.definition}' not found in trace set")

    definition = trace_set.definitions[solution.definition]
    all_workloads = trace_set.workloads.get(solution.definition, [])
    if workload_start < 0:
        raise ValueError("--workload-start must be non-negative")
    workloads = all_workloads[workload_start : workload_start + max_workloads]
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
            if evaluation.correctness.extra is not None:
                entry["matched_ratio"] = evaluation.correctness.extra.get("matched_ratio")

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
            if result.get("matched_ratio") is not None:
                print(f" | matched={result['matched_ratio']:.4f}", end="")

            print()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Single-process local benchmark runner")
    p.add_argument("--warmup-runs", type=int, default=1)
    p.add_argument("--iterations", type=int, default=10)
    p.add_argument("--num-trials", type=int, default=1)
    p.add_argument("--max-workloads", type=int, default=3)
    p.add_argument("--workload-start", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--rtol", type=float, default=None)
    p.add_argument("--atol", type=float, default=None)
    p.add_argument("--required-matched-ratio", type=float, default=None)
    return p.parse_args()


def main():
    args = parse_args()

    print("Packing solution from source files...")
    solution_path = pack_solution()

    print("\nLoading solution...")
    solution = Solution.model_validate_json(solution_path.read_text())
    print(f"Loaded: {solution.name} ({solution.definition})")
    ensure_tvm_ffi_cuda_arch(solution)
    ensure_cutlass_include_paths()

    cfg = apply_definition_defaults(
        solution.definition,
        BenchmarkConfig(
        warmup_runs=args.warmup_runs,
        iterations=args.iterations,
        num_trials=args.num_trials,
        rtol=1e-2,
        atol=1e-2,
        ),
        rtol=args.rtol,
        atol=args.atol,
        required_matched_ratio=args.required_matched_ratio,
    )

    print(
        "\nRunning benchmark (single-process mode: "
        f"warmup={args.warmup_runs}, iter={args.iterations}, "
        f"trials={args.num_trials}, workload_start={args.workload_start}, "
        f"workloads={args.max_workloads}, device={args.device})..."
    )

    results = run_single(solution, cfg, args.max_workloads, args.device, args.workload_start)
    print_results(results)


if __name__ == "__main__":
    main()
