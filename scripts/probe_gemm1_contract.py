"""
Probe the FlashInfer SM100 FP8 group-GEMM contract for the MoE GEMM1 shape.

This is intentionally separate from the full solution path.  The current
FIB_MOE_TC prototype compiles on B200, but GEMM1-only is numerically wrong.
This script runs on Modal B200, recreates benchmark inputs with the same
flashinfer-bench utilities, and compares candidate FlashInfer group GEMM input
layouts against the scalar PyTorch GEMM1 reference.

Usage:
    modal run scripts/probe_gemm1_contract.py::probe

Optional:
    modal run scripts/probe_gemm1_contract.py::probe --workload-index 3 --rows 32
"""

from __future__ import annotations

import csv
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import modal

app = modal.App("flashinfer-gemm1-contract-probe")
trace_volume = modal.Volume.from_name("flashinfer-trace", create_if_missing=True)
TRACE_SET_PATH = "/data"
image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.12"
    )
    .env({"CUDA_HOME": "/usr/local/cuda"})
    .pip_install("flashinfer-bench", "torch", "triton", "numpy", "safetensors")
)
_COMMON = dict(image=image, timeout=3600, volumes={TRACE_SET_PATH: trace_volume})

DEFINITION = "moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048"
BLOCK = 128
HIDDEN = 7168
GEMM1_OUT = 4096
HIDDEN_BLOCKS = HIDDEN // BLOCK
GEMM1_OUT_BLOCKS = GEMM1_OUT // BLOCK


def _ensure_cuda_arch() -> None:
    import os

    import torch

    if not torch.cuda.is_available() or torch.cuda.device_count() == 0:
        return
    major, minor = torch.cuda.get_device_capability(0)
    suffix = "a" if major >= 10 else ""
    arch = f"{major}.{minor}{suffix}"
    prev = os.environ.get("TVM_FFI_CUDA_ARCH_LIST")
    if prev and prev != arch:
        print(f"[probe_gemm1_contract] overriding TVM_FFI_CUDA_ARCH_LIST={prev} -> {arch}")
    os.environ["TVM_FFI_CUDA_ARCH_LIST"] = arch
    print(f"[probe_gemm1_contract] TVM_FFI_CUDA_ARCH_LIST={arch}")


def _fmt(v: Any, spec: str = ".4g") -> str:
    if v is None:
        return "-"
    if isinstance(v, str):
        return v
    try:
        return f"{float(v):{spec}}"
    except (TypeError, ValueError):
        return str(v)


def _short_uuid(value: str) -> str:
    return value[:8] if value else "-"


def _write_outputs(out_dir: Path, rows: list[dict], meta: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    fields = [
        "rank",
        "ok",
        "mode",
        "b_layout",
        "sign_mode",
        "mma_sm",
        "out_dtype",
        "max_abs",
        "max_rel",
        "mean_abs",
        "matched_ratio",
        "error",
    ]
    with (out_dir / "gemm1_contract_candidates.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    lines = []
    lines.append(f"# GEMM1 Contract Probe {meta['timestamp']}")
    lines.append("")
    lines.append("| metric | value |")
    lines.append("|---|---|")
    for k, v in meta.items():
        lines.append(f"| {k} | {v} |")
    lines.append("")
    lines.append("## Best Candidates")
    lines.append("")
    lines.append("| rank | ok | mode | b_layout | sign_mode | mma_sm | dtype | max_abs | max_rel | mean_abs | matched | error |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for row in rows[:20]:
        lines.append(
            "| {rank} | {ok} | {mode} | {b_layout} | {sign_mode} | {mma_sm} | {dtype} | "
            "{max_abs} | {max_rel} | {mean_abs} | {matched} | {error} |".format(
                rank=row.get("rank", "-"),
                ok=row.get("ok", "-"),
                mode=row.get("mode", "-"),
                b_layout=row.get("b_layout", "-"),
                sign_mode=row.get("sign_mode", "-"),
                mma_sm=row.get("mma_sm", "-"),
                dtype=row.get("out_dtype", "-"),
                max_abs=_fmt(row.get("max_abs"), ".4e"),
                max_rel=_fmt(row.get("max_rel"), ".4e"),
                mean_abs=_fmt(row.get("mean_abs"), ".4e"),
                matched=_fmt(row.get("matched_ratio"), ".6f"),
                error=str(row.get("error") or "").replace("|", "/")[:160],
            )
        )
    lines.append("")
    lines.append("A candidate is promising only if `matched_ratio` is close to 1.0 under the contest tolerance.")
    (out_dir / "summary.md").write_text("\n".join(lines))


def _probe_impl(workload_index: int, rows: int, expert: int | None, seed: int) -> dict:
    import torch
    from flashinfer.gemm import group_gemm_fp8_nt_groupwise
    from flashinfer_bench import TraceSet
    from flashinfer_bench.bench.config import BenchmarkConfig
    from flashinfer_bench.bench.utils import gen_inputs, load_safetensors

    _ensure_cuda_arch()
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    trace_set = TraceSet.from_path(TRACE_SET_PATH)
    definition = trace_set.definitions[DEFINITION]
    traces = trace_set.workloads[DEFINITION]
    if workload_index < 0 or workload_index >= len(traces):
        raise ValueError(f"workload_index must be in [0, {len(traces) - 1}], got {workload_index}")

    trace = traces[workload_index]
    workload = trace.workload
    safe = load_safetensors(definition, workload, trace_set.root)
    values = gen_inputs(definition, workload, "cuda:0", safe)
    inputs = dict(zip(definition.inputs.keys(), values))

    hidden = inputs["hidden_states"]
    hidden_scale = inputs["hidden_states_scale"].to(torch.float32)
    w13_all = inputs["gemm1_weights"]
    s13_all = inputs["gemm1_weights_scale"].to(torch.float32)

    seq_len = int(hidden.shape[0])
    m = min(int(rows), seq_len)
    padded_m = (m + 3) & ~3
    if padded_m <= 0:
        raise ValueError("rows must be positive")

    le = int(expert if expert is not None else 0)
    if le < 0 or le >= int(w13_all.shape[0]):
        raise ValueError(f"expert must be in [0, {int(w13_all.shape[0]) - 1}], got {le}")

    token_idx = torch.arange(m, device="cuda:0", dtype=torch.long)

    a = torch.zeros((padded_m, HIDDEN), dtype=hidden.dtype, device="cuda:0")
    a[:m] = hidden.index_select(0, token_idx)
    a_scale_k = torch.zeros((padded_m, HIDDEN_BLOCKS), dtype=torch.float32, device="cuda:0")
    a_scale_k[:m] = hidden_scale[:, :m].t().contiguous()
    # Non-real padded rows get scale 1.0 to avoid accidental NaNs in kernels.
    if padded_m > m:
        a_scale_k[m:] = 1.0

    b_nk = w13_all[le].contiguous()
    b_scale_k = s13_all[le].contiguous()  # [n_blocks, k_blocks]

    # PyTorch scalar reference under the contest semantics.
    a_ref = a[:m].to(torch.float32) * a_scale_k[:m].repeat_interleave(BLOCK, dim=1)
    b_ref = b_nk.to(torch.float32) * b_scale_k.repeat_interleave(BLOCK, dim=0).repeat_interleave(BLOCK, dim=1)
    ref = a_ref.matmul(b_ref.t()).to(torch.float32)
    ref_bf16 = ref.to(torch.bfloat16).to(torch.float32)

    cfg = BenchmarkConfig()
    atol = float(cfg.atol)
    rtol = float(cfg.rtol)

    def fold_a_sign(x: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
        signs = torch.sign(scales)
        signs = torch.where(signs == 0, torch.ones_like(signs), signs)
        y = x.to(torch.float32).view(padded_m, HIDDEN_BLOCKS, BLOCK)
        y = y * signs[:, :, None]
        return y.reshape(padded_m, HIDDEN).to(torch.float8_e4m3fn).contiguous()

    def fold_b_sign(x: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
        signs = torch.sign(scales)
        signs = torch.where(signs == 0, torch.ones_like(signs), signs)
        y = x.to(torch.float32).view(GEMM1_OUT_BLOCKS, BLOCK, HIDDEN_BLOCKS, BLOCK)
        y = y * signs[:, None, :, None]
        return y.reshape(GEMM1_OUT, HIDDEN).to(torch.float8_e4m3fn).contiguous()

    sign_variants = []
    sign_variants.append(("raw", a, a_scale_k, b_nk, b_scale_k))
    sign_variants.append(("abs_scale_only", a, a_scale_k.abs(), b_nk, b_scale_k.abs()))
    sign_variants.append((
        "fold_operand_signs",
        fold_a_sign(a, a_scale_k),
        a_scale_k.abs(),
        fold_b_sign(b_nk, b_scale_k),
        b_scale_k.abs(),
    ))

    candidates = []
    for sign_name, a_in, a_scale_in_k, b_in_nk, b_scale_in_k in sign_variants:
        a_scale_by_mode = {
            "K": a_scale_in_k.contiguous(),
            "MN": a_scale_in_k.t().contiguous(),
        }
        b_scale_by_mode = {
            "K": b_scale_in_k.unsqueeze(0).contiguous(),
            "MN": b_scale_in_k.t().unsqueeze(0).contiguous(),
        }
        b_by_layout = {
            "rowmajor_physical": b_in_nk.contiguous().unsqueeze(0),
            # Shape is still [1, N, K], but physical storage is column-major
            # because the inner 2D tensor has stride (1, N).
            "colmajor_physical": b_in_nk.t().contiguous().t().unsqueeze(0),
        }
        for mode in ("MN", "K"):
            for b_layout, b_arg in b_by_layout.items():
                for mma_sm in (1, 2):
                    for out_dtype in (torch.bfloat16, torch.float16):
                        candidates.append({
                            "mode": mode,
                            "b_layout": b_layout,
                            "sign_mode": sign_name,
                            "mma_sm": mma_sm,
                            "out_dtype": out_dtype,
                            "a": a_in,
                            "b": b_arg,
                            "a_scale": a_scale_by_mode[mode],
                            "b_scale": b_scale_by_mode[mode],
                        })

    m_indptr = torch.tensor([0, padded_m], dtype=torch.int32, device="cuda:0")
    results = []
    for cand in candidates:
        row = {
            "ok": False,
            "mode": cand["mode"],
            "b_layout": cand["b_layout"],
            "sign_mode": cand["sign_mode"],
            "mma_sm": cand["mma_sm"],
            "out_dtype": str(cand["out_dtype"]).replace("torch.", ""),
            "max_abs": None,
            "max_rel": None,
            "mean_abs": None,
            "matched_ratio": 0.0,
            "error": "",
        }
        try:
            out = group_gemm_fp8_nt_groupwise(
                cand["a"],
                cand["b"],
                cand["a_scale"],
                cand["b_scale"],
                m_indptr,
                scale_granularity_mnk=(1, BLOCK, BLOCK),
                scale_major_mode=cand["mode"],
                mma_sm=int(cand["mma_sm"]),
                out_dtype=cand["out_dtype"],
            )
            torch.cuda.synchronize()
            got = out[:m].to(torch.float32)
            target = ref_bf16 if cand["out_dtype"] is torch.bfloat16 else ref.to(torch.float16).to(torch.float32)
            diff = (got - target).abs()
            rel = diff / (target.abs() + 1e-8)
            exceeds = (diff > atol) & (rel > rtol)
            matched = 1.0 - (float(exceeds.sum().item()) / float(exceeds.numel()))
            row.update({
                "ok": True,
                "max_abs": float(diff.max().item()),
                "max_rel": float(rel.max().item()),
                "mean_abs": float(diff.mean().item()),
                "matched_ratio": matched,
            })
        except Exception as exc:
            row["error"] = f"{type(exc).__name__}: {exc}"
        results.append(row)

    def sort_key(r: dict) -> tuple:
        ok_rank = 0 if r.get("ok") else 1
        matched_rank = -float(r.get("matched_ratio") or 0.0)
        max_abs = float(r.get("max_abs") if r.get("max_abs") is not None else 1e30)
        return (ok_rank, matched_rank, max_abs)

    results.sort(key=sort_key)
    for i, row in enumerate(results, start=1):
        row["rank"] = i

    meta = {
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "definition": DEFINITION,
        "workload_index": workload_index,
        "workload_uuid": trace.workload.uuid,
        "workload_short": _short_uuid(trace.workload.uuid),
        "seq_len": seq_len,
        "rows": m,
        "padded_rows": padded_m,
        "expert": le,
        "seed": seed,
        "atol": atol,
        "rtol": rtol,
        "device": torch.cuda.get_device_name(0),
        "capability": ".".join(str(x) for x in torch.cuda.get_device_capability(0)),
    }
    return {"meta": meta, "rows": results}


@app.function(**_COMMON, gpu="B200:1")
def probe_b200(workload_index: int, rows: int, expert: int | None, seed: int) -> dict:
    return _probe_impl(workload_index, rows, expert, seed)


@app.local_entrypoint()
def probe(workload_index: int = 0, rows: int = 7, expert: int = -1, seed: int = 1234, label: str = ""):
    expert_arg = None if expert < 0 else expert
    result = probe_b200.remote(workload_index, rows, expert_arg, seed)

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    suffix = f"_{label}" if label else ""
    out_dir = PROJECT_ROOT / "experiments" / f"{ts}_B200_gemm1_contract_probe{suffix}"
    _write_outputs(out_dir, result["rows"], result["meta"])

    print(f"Wrote {out_dir / 'summary.md'}")
    print(f"Wrote {out_dir / 'gemm1_contract_candidates.csv'}")
    print("")
    print("Top candidates:")
    for row in result["rows"][:8]:
        print(
            "#{rank:02d} ok={ok} mode={mode} b={b_layout} sign={sign_mode} "
            "mma={mma_sm} dtype={out_dtype} matched={matched} max_abs={max_abs} "
            "max_rel={max_rel} err={error}".format(
                rank=row.get("rank", 0),
                ok=row.get("ok"),
                mode=row.get("mode"),
                b_layout=row.get("b_layout"),
                sign_mode=row.get("sign_mode"),
                mma_sm=row.get("mma_sm"),
                out_dtype=row.get("out_dtype"),
                matched=_fmt(row.get("matched_ratio"), ".6f"),
                max_abs=_fmt(row.get("max_abs"), ".4e"),
                max_rel=_fmt(row.get("max_rel"), ".4e"),
                error=(row.get("error") or "")[:120],
            )
        )
