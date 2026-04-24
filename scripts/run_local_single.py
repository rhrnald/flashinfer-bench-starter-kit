"""
FlashInfer-Bench local single-process runner.

This runner avoids Benchmark's multiprocessing workers and evaluates
reference/baseline/solution directly in the current process.
"""

import argparse
import os
import random
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import torch
import safetensors.torch as st

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
TMP_ROOT = Path.home() / "tmp"
TMP_ROOT.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("TMPDIR", str(TMP_ROOT))
sys.path.insert(0, str(PROJECT_ROOT))

from flashinfer_bench import BenchmarkConfig, Solution, TraceSet
from flashinfer_bench.bench.evaluators import resolve_evaluator
from flashinfer_bench.bench.evaluators.utils import allocate_outputs, normalize_result
from flashinfer_bench.bench.timing import time_runnable
from flashinfer_bench.bench.utils import make_eval
from flashinfer_bench.compile import BuildError, BuilderRegistry
from flashinfer_bench.data import EvaluationStatus
from scripts.pack_solution import pack_solution

DEFAULT_COMPARE_SOLUTION = (
    PROJECT_ROOT.parents[1]
    / "mlsys26-contest"
    / "solutions"
    / "baseline"
    / "moe"
    / "moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048"
    / "flashinfer_wrapper_9sdjf3.json"
)


def set_reproducible_seed(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except Exception:
        pass
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


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
        cflags = list(kwargs.get("extra_cflags") or [])
        if "-O3" not in cflags:
            cflags.append("-O3")
        kwargs["extra_cflags"] = cflags

        cuda_flags = list(kwargs.get("extra_cuda_cflags") or [])
        if "-O3" not in cuda_flags:
            cuda_flags.append("-O3")
        kwargs["extra_cuda_cflags"] = cuda_flags

        ldflags = list(kwargs.get("extra_ldflags") or [])
        for flag in ("-L/usr/local/cuda/lib64/stubs", "-lcuda"):
            if flag not in ldflags:
                ldflags.append(flag)
        kwargs["extra_ldflags"] = ldflags

        include_paths = list(kwargs.get("extra_include_paths") or [])
        for path in (
            PROJECT_ROOT.parents[1] / "cutlass" / "include",
            PROJECT_ROOT.parents[1] / "cutlass" / "tools" / "util" / "include",
        ):
            spath = str(path)
            if spath not in include_paths:
                include_paths.append(spath)
        kwargs["extra_include_paths"] = include_paths
        return original_build(*args, **kwargs)

    build_with_cuda_driver._fib_lcuda_patched = True  # type: ignore[attr-defined]
    tvm_ffi.cpp.build = build_with_cuda_driver


def get_trace_set_path() -> str:
    path = os.environ.get("FIB_DATASET_PATH")
    if not path:
        raise EnvironmentError(
            "FIB_DATASET_PATH environment variable not set. "
            "Please set it to the path of your flashinfer-trace dataset."
        )
    return path


def describe_workload_inputs(trace_set_root: Path, workload) -> str:
    cu_info = workload.inputs.get("cu_seqlens") if hasattr(workload, "inputs") else None
    axes = getattr(workload, "axes", {}) or {}
    if not cu_info:
        total_seq_len = axes.get("total_seq_len")
        num_seqs = axes.get("num_seqs")
        parts = []
        if num_seqs is not None:
            parts.append(f"num_seqs={num_seqs}")
        if total_seq_len is not None:
            parts.append(f"total_seq_len={total_seq_len}")
        return ', '.join(parts)

    try:
        rel_path = cu_info.path if hasattr(cu_info, 'path') else cu_info['path']
        tensor_key = cu_info.tensor_key if hasattr(cu_info, 'tensor_key') else cu_info['tensor_key']
        spath = (trace_set_root / rel_path).resolve()
        vals = st.load_file(str(spath))[tensor_key].cpu().tolist()
        lengths = [vals[i + 1] - vals[i] for i in range(len(vals) - 1)]
        return f"num_seqs={len(lengths)}, seq_lengths={lengths}"
    except Exception as e:
        total_seq_len = axes.get("total_seq_len")
        num_seqs = axes.get("num_seqs")
        parts = []
        if num_seqs is not None:
            parts.append(f"num_seqs={num_seqs}")
        if total_seq_len is not None:
            parts.append(f"total_seq_len={total_seq_len}")
        parts.append(f"cu_seqlens_load_failed={type(e).__name__}")
        return ', '.join(parts)


def load_solution_json(path: str | Path) -> Solution:
    solution_path = Path(path).expanduser().resolve()
    if not solution_path.exists():
        raise FileNotFoundError(f"Solution JSON not found: {solution_path}")
    return Solution.model_validate_json(solution_path.read_text())


def collect_outputs_and_latency(
    definition,
    runnable,
    inputs: list[list[Any]],
    cfg: BenchmarkConfig,
    device: str,
) -> tuple[list[list[torch.Tensor]], float]:
    outputs: list[list[torch.Tensor]] = []
    latencies: list[float] = []
    is_dps = runnable.metadata.destination_passing_style

    for inp in inputs:
        if is_dps:
            out = allocate_outputs(definition, inp, device)
            with torch.no_grad():
                runnable(*inp, *out)
            args = list(inp) + out
        else:
            with torch.no_grad():
                result = runnable(*inp)
            out = normalize_result(definition, result, device)
            args = list(inp)

        torch.cuda.synchronize(device)
        outputs.append(out)

        if cfg.profile_baseline:
            latencies.append(time_runnable(runnable, args, cfg.warmup_runs, cfg.iterations, device))

    mean_latency_ms = sum(latencies) / float(len(latencies)) if latencies else 0.0
    return outputs, mean_latency_ms


def run_single(
    solution: Solution,
    compare_solution: Solution | None,
    cfg: BenchmarkConfig,
    max_workloads: int,
    device: str,
    correctness_only: bool = False,
    workload_indexes: list[int] | None = None,
) -> dict:
    trace_set = TraceSet.from_path(get_trace_set_path())
    if solution.definition not in trace_set.definitions:
        raise ValueError(f"Definition '{solution.definition}' not found in trace set")
    if compare_solution is not None and compare_solution.definition != solution.definition:
        raise ValueError(
            f"Comparison solution definition '{compare_solution.definition}' does not match "
            f"target definition '{solution.definition}'"
        )

    definition = trace_set.definitions[solution.definition]
    all_workloads = trace_set.workloads.get(solution.definition, [])
    if workload_indexes:
        workloads = [all_workloads[i] for i in workload_indexes if 0 <= i < len(all_workloads)]
    else:
        workloads = all_workloads[:max_workloads]
    if not workloads:
        raise ValueError(f"No workloads found for definition '{solution.definition}'")

    evaluator_cls = resolve_evaluator(definition)
    registry = BuilderRegistry.get_instance()

    try:
        sol_runnable = registry.build(definition, solution)
        compare_runnable = (
            registry.build(definition, compare_solution) if compare_solution is not None else None
        )
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

    total_workloads = len(workloads)

    for idx, w in enumerate(workloads, start=1):
        workload = w.workload
        workload_desc = describe_workload_inputs(trace_set.root, workload)
        print(f"[{idx}/{total_workloads}] Starting workload {workload.uuid[:8]}... {workload_desc}", flush=True)
        workload_t0 = time.time()
        log_path = str(Path(cfg.log_dir) / f"{solution.name}_{workload.uuid}_{int(time.time())}.log")
        extra_entry = {}
        try:
            baseline = evaluator_cls.build_baseline(
                definition=definition,
                workload=workload,
                cfg=cfg,
                device=device,
                trace_set_root=trace_set.root,
            )
            if compare_runnable is None:
                ref_outputs = baseline.outputs
                ref_mean_latency_ms = baseline.mean_latency_ms
            else:
                ref_outputs, ref_mean_latency_ms = collect_outputs_and_latency(
                    definition=definition,
                    runnable=compare_runnable,
                    inputs=baseline.inputs,
                    cfg=cfg,
                    device=device,
                )
            correctness, early_eval = evaluator_cls.check_correctness(
                definition=definition,
                sol_runnable=sol_runnable,
                inputs=baseline.inputs,
                ref_outputs=ref_outputs,
                cfg=cfg,
                log_path=log_path,
                device=device,
            )
            performance = None

            if correctness_only:
                evaluation = early_eval or make_eval(
                    status=EvaluationStatus.PASSED,
                    device=device,
                    log_path=log_path,
                    correctness=correctness,
                )
            elif (
                early_eval is not None
                and early_eval.status == EvaluationStatus.INCORRECT_NUMERICAL
            ):
                performance, perf_eval = evaluator_cls.eval_performance(
                    definition=definition,
                    sol_runnable=sol_runnable,
                    inputs=baseline.inputs,
                    ref_mean_latency_ms=ref_mean_latency_ms,
                    cfg=cfg,
                    log_path=log_path,
                    device=device,
                )
                evaluation = perf_eval or early_eval
                if performance is not None:
                    extra_entry["latency_ms"] = performance.latency_ms
                    extra_entry["reference_latency_ms"] = performance.reference_latency_ms
                    extra_entry["speedup_factor"] = performance.speedup_factor
            elif early_eval is not None:
                evaluation = early_eval
            else:
                performance, perf_eval = evaluator_cls.eval_performance(
                    definition=definition,
                    sol_runnable=sol_runnable,
                    inputs=baseline.inputs,
                    ref_mean_latency_ms=ref_mean_latency_ms,
                    cfg=cfg,
                    log_path=log_path,
                    device=device,
                )
                evaluation = perf_eval or make_eval(
                    status=EvaluationStatus.PASSED,
                    device=device,
                    log_path=log_path,
                    correctness=correctness,
                    performance=performance,
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
        elif extra_entry:
            entry.update(extra_entry)

        if evaluation.correctness is not None:
            entry["max_abs_error"] = evaluation.correctness.max_absolute_error
            entry["max_rel_error"] = evaluation.correctness.max_relative_error

        results[definition.name][workload.uuid] = entry
        elapsed_s = time.time() - workload_t0
        print(
            f"[{idx}/{total_workloads}] Finished workload {workload.uuid[:8]}... "
            f"status={entry['status']} elapsed={elapsed_s:.2f}s",
            flush=True,
        )

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
    p.add_argument(
        "--impl",
        choices=("reference", "direct"),
        default="reference",
        help="MoE expert implementation to run",
    )
    p.add_argument("--warmup-runs", type=int, default=1)
    p.add_argument("--iterations", type=int, default=3)
    p.add_argument("--num-trials", type=int, default=1)
    p.add_argument("--max-workloads", type=int, default=3)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--workload-indexes", type=str, default="", help="Comma-separated 0-based workload indexes")
    p.add_argument(
        "--correctness-only",
        action="store_true",
        help="Skip performance measurement and only run correctness validation",
    )
    p.add_argument(
        "--compare-target",
        choices=("baseline", "reference"),
        default="baseline",
        help="Use the baseline solution JSON or builtin definition.reference as comparison target",
    )
    p.add_argument(
        "--compare-solution",
        type=str,
        default=str(DEFAULT_COMPARE_SOLUTION),
        help="Path to solution JSON used as correctness/performance reference when compare-target=baseline",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=1234,
        help="Random seed for generated workload inputs",
    )
    p.add_argument("--rtol", type=float, default=0.3)
    p.add_argument("--atol", type=float, default=1.0)
    p.add_argument("--required-matched-ratio", type=float, default=0.9)
    return p.parse_args()


def main():
    args = parse_args()
    set_reproducible_seed(args.seed)
    os.environ["FIB_MOE_IMPL"] = args.impl
    print(f"Selected implementation: {args.impl}")
    print(f"Using random seed={args.seed}")

    print("Packing solution from source files...")
    solution_path = pack_solution()

    print("\nLoading solution...")
    solution = Solution.model_validate_json(solution_path.read_text())
    print(f"Loaded: {solution.name} ({solution.definition})")
    ensure_tvm_ffi_cuda_arch(solution)
    ensure_tvm_ffi_link_cuda_driver(solution)

    compare_solution = None
    if args.compare_target == "baseline":
        print("\nLoading comparison solution...")
        compare_solution = load_solution_json(args.compare_solution)
        print(f"Loaded comparison: {compare_solution.name} ({compare_solution.definition})")
        ensure_tvm_ffi_cuda_arch(compare_solution)
        ensure_tvm_ffi_link_cuda_driver(compare_solution)
    else:
        print("\nUsing builtin definition.reference for comparison")

    cfg = BenchmarkConfig(
        warmup_runs=args.warmup_runs,
        iterations=args.iterations,
        num_trials=args.num_trials,
        rtol=float(args.rtol),
        atol=float(args.atol),
        required_matched_ratio=(
            float(args.required_matched_ratio)
            if args.required_matched_ratio is not None
            else None
        ),
        profile_baseline=not args.correctness_only,
    )

    print(
        "\nRunning benchmark (single-process mode: "
        f"warmup={args.warmup_runs}, iter={args.iterations}, "
        f"trials={args.num_trials}, workloads={args.max_workloads}, device={args.device}, "
        f"compare={compare_solution.name if compare_solution is not None else 'definition.reference'}, "
        f"correctness_only={args.correctness_only})..."
    )

    workload_indexes = [int(x) for x in args.workload_indexes.split(',') if x.strip()] or None
    results = run_single(
        solution,
        compare_solution,
        cfg,
        args.max_workloads,
        args.device,
        args.correctness_only,
        workload_indexes,
    )
    print_results(results)


if __name__ == "__main__":
    main()
