"""
FlashInfer-Bench A/B sweep harness on top of Modal.

Runs several env-flag variants of the same solution in one Modal container so
nvcc compile cost is paid once. Emits CSV + markdown under
`experiments/<timestamp>_<gpu>/` for reproducible comparison.

Usage (entrypoint must be qualified since the app has other entrypoints):
    # Default variant sweep on A10G, 1 workload:
    modal run scripts/sweep.py::sweep --gpu A10G --max-workloads 1 \
        --warmup-runs 1 --iterations 5 --num-trials 1

    # Explicit variants via JSON path:
    modal run scripts/sweep.py::sweep --gpu B200 --max-workloads 1 \
        --variants-json my_variants.json

The JSON file must parse to `[{"name": "...", "env": {"FIB_...": "1", ...}}, ...]`.
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from flashinfer_bench import BenchmarkConfig, Solution  # noqa: E402
from scripts.run_modal import _GPU_SWEEPERS, app  # noqa: E402 — re-use registered Modal app


DEFAULT_VARIANTS = [
    # Default (grouped) is the shipped path as of 2026-04-18. Legacy opt-out is
    # retained so we can still A/B the dense-mask path against grouped.
    {"name": "default", "env": {}},
    {"name": "tc", "env": {"FIB_MOE_TC": "1"}},
    {"name": "tc_gemm1_only", "env": {"FIB_MOE_TC": "1", "FIB_MOE_TC_GEMM1_ONLY": "1"}},
    {"name": "tc_transpose_b_probe", "env": {"FIB_MOE_TC": "1", "FIB_MOE_TC_TRANSPOSE_B": "1"}},
    {"name": "legacy", "env": {"FIB_MOE_LEGACY": "1"}},
    {"name": "default_profile", "env": {"FIB_MOE_PROFILE": "1"}},
    {"name": "tc_profile", "env": {"FIB_MOE_TC": "1", "FIB_MOE_PROFILE": "1"}},
    {"name": "legacy_profile", "env": {"FIB_MOE_LEGACY": "1", "FIB_MOE_PROFILE": "1"}},
]

PRECISION_VARIANTS = [
    {"name": "baseline_profile", "env": {"FIB_MOE_PROFILE": "1"}},
    {
        "name": "hidden_dequant_bf16",
        "env": {
            "FIB_MOE_PROFILE": "1",
            "FIB_MOE_PREC_STAGE": "hidden_dequant",
            "FIB_MOE_PREC_MODE": "bf16",
        },
    },
    {
        "name": "gemm1_output_bf16",
        "env": {
            "FIB_MOE_PROFILE": "1",
            "FIB_MOE_PREC_STAGE": "gemm1_output",
            "FIB_MOE_PREC_MODE": "bf16",
        },
    },
    {
        "name": "swiglu_output_bf16",
        "env": {
            "FIB_MOE_PROFILE": "1",
            "FIB_MOE_PREC_STAGE": "swiglu_output",
            "FIB_MOE_PREC_MODE": "bf16",
        },
    },
    {
        "name": "gemm2_operands_f16",
        "env": {
            "FIB_MOE_PROFILE": "1",
            "FIB_MOE_PREC_STAGE": "gemm2_operands",
            "FIB_MOE_PREC_MODE": "f16",
        },
    },
    {
        "name": "out_accumulator_bf16",
        "env": {
            "FIB_MOE_PROFILE": "1",
            "FIB_MOE_PREC_STAGE": "out_accumulator",
            "FIB_MOE_PREC_MODE": "bf16",
        },
    },
]


def _experiments_root() -> Path:
    return PROJECT_ROOT / "experiments"


def _short(uuid: str) -> str:
    return uuid[:8] if uuid else "-"


def _write_csv(out_path: Path, rows: list[dict]) -> None:
    import csv

    fieldnames = [
        "variant",
        "workload",
        "status",
        "latency_ms",
        "reference_latency_ms",
        "speedup_factor",
        "max_abs_error",
        "max_rel_error",
    ]
    with out_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_summary(out_path: Path, rows: list[dict], variants: list[dict], meta: dict) -> None:
    # Group by (variant, workload) to produce a compact table.
    lines = []
    lines.append(f"# Sweep {meta['timestamp']}  ({meta['gpu']})")
    lines.append("")
    lines.append("| metric | value |")
    lines.append("|---|---|")
    for k, v in meta.items():
        lines.append(f"| {k} | {v} |")
    lines.append("")

    lines.append("## Variants")
    lines.append("")
    for v in variants:
        env_str = ", ".join(f"`{k}={val}`" for k, val in v.get("env", {}).items()) or "_(none)_"
        lines.append(f"- **{v['name']}**: {env_str}")
    lines.append("")

    lines.append("## Results")
    lines.append("")
    lines.append(
        "| variant | workload | status | latency_ms | ref_ms | speedup | abs_err | rel_err |"
    )
    lines.append("|---|---|---|---|---|---|---|---|")
    for r in rows:
        lines.append(
            "| {variant} | {workload} | {status} | {lat} | {ref} | {sp} | {ae} | {re} |".format(
                variant=r.get("variant", "-"),
                workload=_short(r.get("workload", "-")),
                status=r.get("status", "-"),
                lat=_fmt(r.get("latency_ms"), ".3f"),
                ref=_fmt(r.get("reference_latency_ms"), ".3f"),
                sp=_fmt(r.get("speedup_factor"), ".2f") + ("x" if r.get("speedup_factor") is not None else ""),
                ae=_fmt(r.get("max_abs_error"), ".2e"),
                re=_fmt(r.get("max_rel_error"), ".2e"),
            )
        )
    lines.append("")

    # Per-variant average speedup (only over PASSED rows with a speedup_factor).
    lines.append("## Speedup by variant (PASSED rows only)")
    lines.append("")
    lines.append("| variant | n | mean speedup | min speedup | max speedup |")
    lines.append("|---|---|---|---|---|")
    variant_to_sp: dict[str, list[float]] = {}
    for r in rows:
        if r.get("status") != "PASSED":
            continue
        sp = r.get("speedup_factor")
        if sp is None:
            continue
        variant_to_sp.setdefault(r["variant"], []).append(float(sp))
    for v in variants:
        sps = variant_to_sp.get(v["name"], [])
        if not sps:
            lines.append(f"| {v['name']} | 0 | - | - | - |")
            continue
        mean = sum(sps) / len(sps)
        lines.append(
            f"| {v['name']} | {len(sps)} | {mean:.3f}x | {min(sps):.3f}x | {max(sps):.3f}x |"
        )
    lines.append("")

    out_path.write_text("\n".join(lines))


def _fmt(val, spec: str) -> str:
    if val is None:
        return "-"
    try:
        return f"{float(val):{spec}}"
    except (TypeError, ValueError):
        return str(val)


def _flatten(results_by_variant: dict) -> list[dict]:
    rows = []
    for variant_name, per_def in results_by_variant.items():
        if variant_name == "__build_error__":
            # Top-level build failure — shared by all variants.
            rows.append({
                "variant": "(build)",
                "workload": "-",
                "status": per_def.get("status", "COMPILE_ERROR"),
                "latency_ms": None,
                "reference_latency_ms": None,
                "speedup_factor": None,
                "max_abs_error": None,
                "max_rel_error": None,
                "error": per_def.get("error", ""),
            })
            continue
        for _def_name, workloads in per_def.items():
            for workload_uuid, entry in workloads.items():
                row = {"variant": variant_name, "workload": workload_uuid}
                row.update({
                    "status": entry.get("status"),
                    "latency_ms": entry.get("latency_ms"),
                    "reference_latency_ms": entry.get("reference_latency_ms"),
                    "speedup_factor": entry.get("speedup_factor"),
                    "max_abs_error": entry.get("max_abs_error"),
                    "max_rel_error": entry.get("max_rel_error"),
                })
                rows.append(row)
    return rows


@app.local_entrypoint()
def sweep(
    gpu: str = "A10G",
    max_workloads: int = 1,
    warmup_runs: int = 1,
    iterations: int = 5,
    num_trials: int = 1,
    variants_json: str = "",
    preset: str = "default",
    label: str = "",
):
    """Pack solution and sweep variants on Modal. Outputs CSV + markdown."""
    from scripts.pack_solution import pack_solution

    if variants_json:
        variants = json.loads(Path(variants_json).read_text())
    elif preset == "precision":
        variants = PRECISION_VARIANTS
    elif preset == "default":
        variants = DEFAULT_VARIANTS
    else:
        raise ValueError("Unsupported --preset; choose from default, precision")

    print("Packing solution from source files...")
    solution_path = pack_solution()

    solution = Solution.model_validate_json(solution_path.read_text())
    print(f"Loaded: {solution.name} ({solution.definition})")

    config = BenchmarkConfig(
        warmup_runs=warmup_runs,
        iterations=iterations,
        num_trials=num_trials,
    )

    gpu_key = gpu.upper()
    if gpu_key not in _GPU_SWEEPERS:
        raise ValueError(f"Unsupported --gpu {gpu!r}; choose from {sorted(_GPU_SWEEPERS)}")

    workloads_label = "all" if max_workloads == 0 else str(max_workloads)
    print(
        f"\nSweeping on Modal {gpu} (workloads={workloads_label}, variants={len(variants)}, "
        f"warmup={warmup_runs}, iter={iterations}, trials={num_trials})..."
    )
    for v in variants:
        env_str = ", ".join(f"{k}={val}" for k, val in v.get("env", {}).items()) or "(none)"
        print(f"  - {v['name']}: {env_str}")

    fn = _GPU_SWEEPERS[gpu_key]
    results_by_variant = fn.remote(solution, config, max_workloads, variants)

    if not results_by_variant:
        print("No results returned!")
        return

    # Check for a top-level build error.
    if "__build_error__" in results_by_variant:
        err = results_by_variant["__build_error__"]
        print(f"\nCOMPILE_ERROR (shared across variants):\n{err.get('error', '')}")
        return

    rows = _flatten(results_by_variant)

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    suffix = f"_{label}" if label else ""
    out_dir = _experiments_root() / f"{ts}_{gpu_key}{suffix}"
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / "results.csv"
    md_path = out_dir / "summary.md"
    raw_path = out_dir / "raw.json"

    meta = {
        "timestamp": ts,
        "gpu": gpu_key,
        "max_workloads": workloads_label,
        "warmup_runs": warmup_runs,
        "iterations": iterations,
        "num_trials": num_trials,
        "n_variants": len(variants),
        "solution": solution.name,
        "definition": solution.definition,
    }

    _write_csv(csv_path, rows)
    _write_summary(md_path, rows, variants, meta)
    raw_path.write_text(json.dumps(results_by_variant, indent=2, default=str))

    print(f"\nWrote {csv_path.relative_to(PROJECT_ROOT)}")
    print(f"Wrote {md_path.relative_to(PROJECT_ROOT)}")
    print(f"Wrote {raw_path.relative_to(PROJECT_ROOT)}")
    print(f"\n--- {md_path.name} preview ---")
    print(md_path.read_text())
