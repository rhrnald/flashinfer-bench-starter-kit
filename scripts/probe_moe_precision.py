"""
Probe whether low-precision GEMM1 output is enough for final MoE correctness.

The FlashInfer TC GEMM1 oracle can match GEMM1 itself within tolerance, but the
full MoE output may amplify small G1 errors through SwiGLU and GEMM2.  This
Modal B200 script runs a PyTorch oracle for one workload and compares:

  - fp32_g1: reference-style GEMM1
  - bf16_g1: GEMM1 rounded to BF16 before SwiGLU
  - f16_g1: GEMM1 rounded to FP16 before SwiGLU

Usage:
    modal run scripts/probe_moe_precision.py::probe
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

app = modal.App("flashinfer-moe-precision-probe")
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
INTERMEDIATE = 2048
TOP_K = 8
N_GROUP = 8
TOPK_GROUP = 4
E_LOCAL = 32


def _fmt(v: Any, spec: str = ".4g") -> str:
    if v is None:
        return "-"
    try:
        return f"{float(v):{spec}}"
    except (TypeError, ValueError):
        return str(v)


def _write_outputs(out_dir: Path, rows: list[dict], meta: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fields = ["variant", "max_abs", "max_rel", "mean_abs", "matched_ratio", "selected_experts"]
    with (out_dir / "precision_candidates.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    lines = [f"# MoE Precision Probe {meta['timestamp']}", ""]
    lines.append("| metric | value |")
    lines.append("|---|---|")
    for k, v in meta.items():
        lines.append(f"| {k} | {v} |")
    lines.append("")
    lines.append("| variant | max_abs | max_rel | mean_abs | matched | selected_experts |")
    lines.append("|---|---|---|---|---|---|")
    for r in rows:
        lines.append(
            f"| {r['variant']} | {_fmt(r['max_abs'], '.4e')} | {_fmt(r['max_rel'], '.4e')} | "
            f"{_fmt(r['mean_abs'], '.4e')} | {_fmt(r['matched_ratio'], '.6f')} | "
            f"{r['selected_experts']} |"
        )
    (out_dir / "summary.md").write_text("\n".join(lines))


def _route(inputs: dict, routed_scaling_factor: float):
    import torch

    logits = inputs["routing_logits"].to(torch.float32)
    bias = inputs["routing_bias"].to(torch.float32).reshape(-1)
    t, e_global = logits.shape
    group_size = e_global // N_GROUP
    s = torch.sigmoid(logits)
    s_wb = s + bias
    grouped = s_wb.view(t, N_GROUP, group_size)
    top2_vals, _ = torch.topk(grouped, k=2, dim=2, largest=True, sorted=False)
    group_scores = top2_vals.sum(dim=2)
    _, group_idx = torch.topk(group_scores, k=TOPK_GROUP, dim=1, largest=True, sorted=False)
    group_mask = torch.zeros_like(group_scores)
    group_mask.scatter_(1, group_idx, 1.0)
    score_mask = group_mask.unsqueeze(2).expand(t, N_GROUP, group_size).reshape(t, e_global)
    scores_pruned = s_wb.masked_fill(score_mask == 0, torch.finfo(torch.float32).min)
    _, topk_idx = torch.topk(scores_pruned, k=TOP_K, dim=1, largest=True, sorted=False)
    mask = torch.zeros_like(s)
    mask.scatter_(1, topk_idx, 1.0)
    weights = s * mask
    weights = (weights / (weights.sum(dim=1, keepdim=True) + 1e-20)) * routed_scaling_factor
    return topk_idx, weights


def _run_moe(inputs: dict, g1_round: str):
    import torch

    hidden = inputs["hidden_states"]
    hidden_scale = inputs["hidden_states_scale"].to(torch.float32)
    w13 = inputs["gemm1_weights"]
    s13 = inputs["gemm1_weights_scale"].to(torch.float32)
    w2 = inputs["gemm2_weights"]
    s2 = inputs["gemm2_weights_scale"].to(torch.float32)
    local_offset = int(inputs["local_expert_offset"])
    routed_scaling = float(inputs["routed_scaling_factor"])
    t = int(hidden.shape[0])

    a_scale = hidden_scale.t().contiguous()
    a = hidden.to(torch.float32) * a_scale.repeat_interleave(BLOCK, dim=1)
    topk_idx, weights = _route(inputs, routed_scaling)

    out = torch.zeros((t, HIDDEN), dtype=torch.float32, device=hidden.device)
    selected_experts = 0
    for le in range(E_LOCAL):
        ge = local_offset + le
        selected = (topk_idx == ge).any(dim=1)
        if not selected.any():
            continue
        selected_experts += 1
        tok = torch.nonzero(selected, as_tuple=False).squeeze(1)
        a_e = a.index_select(0, tok)
        w13_scale = s13[le].repeat_interleave(BLOCK, dim=0).repeat_interleave(BLOCK, dim=1)
        w13_e = w13[le].to(torch.float32) * w13_scale
        g1 = a_e.matmul(w13_e.t())
        if g1_round == "bf16":
            g1 = g1.to(torch.bfloat16).to(torch.float32)
        elif g1_round == "f16":
            g1 = g1.to(torch.float16).to(torch.float32)
        x1 = g1[:, :INTERMEDIATE]
        x2 = g1[:, INTERMEDIATE:]
        c = x1 * torch.nn.functional.silu(x2)
        w2_scale = s2[le].repeat_interleave(BLOCK, dim=0).repeat_interleave(BLOCK, dim=1)
        w2_e = w2[le].to(torch.float32) * w2_scale
        o = c.matmul(w2_e.t())
        w_tok = weights.index_select(0, tok)[:, ge]
        out.index_add_(0, tok, o * w_tok.unsqueeze(1))
    return out.to(torch.bfloat16), selected_experts


def _probe_impl(workload_index: int, seed: int) -> dict:
    import torch
    from flashinfer_bench import TraceSet
    from flashinfer_bench.bench.config import BenchmarkConfig
    from flashinfer_bench.bench.utils import gen_inputs, load_safetensors

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    trace_set = TraceSet.from_path(TRACE_SET_PATH)
    definition = trace_set.definitions[DEFINITION]
    trace = trace_set.workloads[DEFINITION][workload_index]
    safe = load_safetensors(definition, trace.workload, trace_set.root)
    values = gen_inputs(definition, trace.workload, "cuda:0", safe)
    inputs = dict(zip(definition.inputs.keys(), values))
    cfg = BenchmarkConfig()
    atol = float(cfg.atol)
    rtol = float(cfg.rtol)

    ref, selected = _run_moe(inputs, "fp32")
    rows = []
    for variant in ("fp32", "bf16", "f16"):
        got, selected_experts = (ref, selected) if variant == "fp32" else _run_moe(inputs, variant)
        diff = (got.to(torch.float32) - ref.to(torch.float32)).abs()
        rel = diff / (ref.to(torch.float32).abs() + 1e-8)
        exceeds = (diff > atol) & (rel > rtol)
        matched = 1.0 - float(exceeds.sum().item()) / float(exceeds.numel())
        rows.append({
            "variant": variant,
            "max_abs": float(diff.max().item()),
            "max_rel": float(rel.max().item()),
            "mean_abs": float(diff.mean().item()),
            "matched_ratio": matched,
            "selected_experts": selected_experts,
        })

    meta = {
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "definition": DEFINITION,
        "workload_index": workload_index,
        "workload_uuid": trace.workload.uuid,
        "seq_len": int(inputs["hidden_states"].shape[0]),
        "seed": seed,
        "atol": atol,
        "rtol": rtol,
        "device": torch.cuda.get_device_name(0),
    }
    return {"meta": meta, "rows": rows}


@app.function(**_COMMON, gpu="B200:1")
def probe_b200(workload_index: int, seed: int) -> dict:
    return _probe_impl(workload_index, seed)


@app.local_entrypoint()
def probe(workload_index: int = 0, seed: int = 1234, label: str = ""):
    result = probe_b200.remote(workload_index, seed)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    suffix = f"_{label}" if label else ""
    out_dir = PROJECT_ROOT / "experiments" / f"{ts}_B200_moe_precision_probe{suffix}"
    _write_outputs(out_dir, result["rows"], result["meta"])
    print(f"Wrote {out_dir / 'summary.md'}")
    for row in result["rows"]:
        print(
            "{variant}: matched={matched} max_abs={max_abs} max_rel={max_rel} mean_abs={mean_abs}".format(
                variant=row["variant"],
                matched=_fmt(row["matched_ratio"], ".6f"),
                max_abs=_fmt(row["max_abs"], ".4e"),
                max_rel=_fmt(row["max_rel"], ".4e"),
                mean_abs=_fmt(row["mean_abs"], ".4e"),
            )
        )
