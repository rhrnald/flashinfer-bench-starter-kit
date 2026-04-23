"""
Search MoE precision-reduction candidates without changing the CUDA baseline.

This script runs a PyTorch oracle over a representative workload panel and
evaluates stage-scoped precision reductions one at a time, then cumulatively.
The search is declarative: candidates are generated from stage/mode/scale/scope
metadata rather than hardcoded for one hypothesis.

Usage:
    modal run scripts/probe_moe_precision.py::probe
"""

from __future__ import annotations

import csv
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import modal

app = modal.App("flashinfer-moe-precision-search")
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
DEFAULT_PANEL_SIZE = 8
DEFAULT_ATOL = 1.0
DEFAULT_RTOL = 0.3
DEFAULT_REQUIRED_MATCHED_RATIO = 0.9
STRICT_ATOL = 0.01
STRICT_RTOL = 0.01

KNOWN_PANEL_UUIDS = [
    "b8f4f012",
    "1a4c6ba1",
    "5e8dc11c",
    "58a34f27",
]


@dataclass(frozen=True)
class Candidate:
    stage: str
    mode: str
    scale_mode: str = "none"
    scope: str = "storage_compute"

    @property
    def name(self) -> str:
        parts = [self.stage, self.mode]
        if self.scale_mode != "none":
            parts.append(self.scale_mode)
        if self.scope != "storage_compute":
            parts.append(self.scope)
        return "__".join(parts)


def _fmt(v: Any, spec: str = ".4g") -> str:
    if v is None:
        return "-"
    try:
        return f"{float(v):{spec}}"
    except (TypeError, ValueError):
        return str(v)


def _canonical_short_uuid(v: str) -> str:
    return v.split("-")[0]


def _mode_rank(mode: str) -> int:
    order = {"fp32": 0, "bf16": 1, "f16": 2, "fp8": 3}
    return order.get(mode, 99)


def _round_tensor(x, mode: str):
    import torch

    if mode == "fp32":
        return x.to(torch.float32)
    if mode == "bf16":
        return x.to(torch.bfloat16).to(torch.float32)
    if mode == "f16":
        return x.to(torch.float16).to(torch.float32)
    raise ValueError(f"Unsupported round mode {mode!r}")


def _quantize_fp8_scaled(x, scale_mode: str, block: int = BLOCK):
    import torch

    ax = x.abs()
    if scale_mode == "tensor":
        scale = ax.max().clamp_min(1.0e-8) / 448.0
        return (x / scale).to(torch.float8_e4m3fn).to(torch.float32) * scale
    if scale_mode == "row":
        if x.ndim < 2:
            scale = ax.max().clamp_min(1.0e-8) / 448.0
            return (x / scale).to(torch.float8_e4m3fn).to(torch.float32) * scale
        scale = ax.max(dim=-1, keepdim=True).values.clamp_min(1.0e-8) / 448.0
        return (x / scale).to(torch.float8_e4m3fn).to(torch.float32) * scale
    if scale_mode == "block":
        if x.shape[-1] % block != 0:
            raise ValueError(f"Block quantization requires trailing dim divisible by {block}")
        shape = x.shape
        xb = x.reshape(*shape[:-1], shape[-1] // block, block)
        scale = xb.abs().amax(dim=-1, keepdim=True).clamp_min(1.0e-8) / 448.0
        qb = (xb / scale).to(torch.float8_e4m3fn).to(torch.float32) * scale
        return qb.reshape(shape)
    raise ValueError(f"Unsupported FP8 scale_mode {scale_mode!r}")


def _apply_precision(x, mode: str, scale_mode: str):
    if mode == "fp32":
        return x
    if mode in ("bf16", "f16"):
        return _round_tensor(x, mode)
    if mode == "fp8":
        return _quantize_fp8_scaled(x, scale_mode)
    raise ValueError(f"Unsupported precision mode {mode!r}")


def _accumulate_blockwise(a, b_t, block_terms: int, acc_mode: str):
    import torch

    chunks = []
    for i0 in range(0, a.shape[1], block_terms):
        i1 = min(i0 + block_terms, a.shape[1])
        chunks.append(a[:, i0:i1].matmul(b_t[i0:i1, :]))
    acc = torch.zeros_like(chunks[0], dtype=torch.float32)
    for chunk in chunks:
        acc = acc + chunk
        if acc_mode != "fp32":
            acc = _round_tensor(acc, acc_mode)
    return acc


def _classify_failure(
    matched_ratio: float,
    max_rel: float,
    max_abs: float,
    *,
    pass_threshold: float,
) -> str:
    if matched_ratio >= pass_threshold:
        return "pass"
    if matched_ratio < 0.50:
        return "catastrophic_saturation"
    if max_rel > 5.0:
        return "catastrophic_outlier"
    if matched_ratio < pass_threshold:
        return "global_drift"
    return "localized_outlier"


def _candidate_table() -> list[Candidate]:
    table = [Candidate("baseline", "fp32")]
    for stage, modes in (
        ("routing_math", ("bf16", "f16")),
        ("routing_weights", ("bf16", "f16")),
        ("hidden_dequant", ("bf16", "f16")),
        ("gemm1_operands", ("bf16", "f16")),
        ("gemm1_accumulator", ("bf16", "f16")),
        ("gemm1_output", ("bf16", "f16")),
        ("swiglu_input", ("bf16", "f16")),
        ("gemm2_operands", ("bf16", "f16")),
        ("gemm2_accumulator", ("bf16", "f16")),
        ("out_accumulator", ("bf16", "f16")),
    ):
        for mode in modes:
            table.append(Candidate(stage, mode))
    table.extend(
        [
            Candidate("swiglu_output", "bf16"),
            Candidate("swiglu_output", "f16"),
            Candidate("swiglu_output", "fp8", "tensor"),
            Candidate("swiglu_output", "fp8", "row"),
            Candidate("swiglu_output", "fp8", "block"),
        ]
    )
    return table


def _combined_candidates(survivors: list[Candidate]) -> list[Candidate]:
    chosen: dict[str, Candidate] = {}
    for cand in survivors:
        if cand.stage in chosen:
            prev = chosen[cand.stage]
            cur_key = (_mode_rank(cand.mode), cand.scale_mode)
            prev_key = (_mode_rank(prev.mode), prev.scale_mode)
            if cur_key > prev_key:
                chosen[cand.stage] = cand
        else:
            chosen[cand.stage] = cand
    return [
        chosen[stage]
        for stage in sorted(chosen.keys())
        if stage != "baseline"
    ]


def _write_outputs(out_dir: Path, rows: list[dict], meta: dict, stage_summary: list[dict], frontier: list[dict]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fields = [
        "candidate",
        "search_phase",
        "workload_index",
        "workload_uuid",
        "seq_len",
        "stage",
        "mode",
        "scale_mode",
        "scope",
        "max_abs",
        "max_rel",
        "mean_abs",
        "matched_ratio_contest",
        "matched_ratio_strict",
        "selected_experts",
        "failure_label",
    ]
    with (out_dir / "precision_candidates.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    (out_dir / "stage_summary.json").write_text(json.dumps(stage_summary, indent=2))
    (out_dir / "safe_frontier.json").write_text(json.dumps(frontier, indent=2))

    lines = [f"# MoE Precision Search {meta['timestamp']}", ""]
    lines.append("| metric | value |")
    lines.append("|---|---|")
    for k, v in meta.items():
        lines.append(f"| {k} | {v} |")
    lines.append("")

    lines.append("## Survivor Summary")
    lines.append("")
    lines.append("| category | count | candidates |")
    lines.append("|---|---:|---|")
    lines.append(
        f"| contest_safe_single_stage | {meta['contest_safe_survivor_count']} | {meta['contest_safe_survivors']} |"
    )
    lines.append(
        f"| strict_safe_single_stage | {meta['strict_safe_survivor_count']} | {meta['strict_safe_survivors']} |"
    )
    lines.append(
        f"| contest_only_single_stage | {meta['contest_only_survivor_count']} | {meta['contest_only_survivors']} |"
    )
    lines.append("")

    lines.append("## Stage Summary")
    lines.append("")
    lines.append("| stage | best_safe_mode | scale | worst_matched_contest | worst_matched_strict | worst_rel | status |")
    lines.append("|---|---|---|---:|---:|---:|---|")
    for row in stage_summary:
        lines.append(
            f"| {row['stage']} | {row['best_safe_mode']} | {row['scale_mode']} | "
            f"{_fmt(row['worst_matched_ratio_contest'], '.6f')} | {_fmt(row['worst_matched_ratio_strict'], '.6f')} | "
            f"{_fmt(row['worst_max_rel'], '.4e')} | "
            f"{row['status']} |"
        )
    lines.append("")

    lines.append("## Cumulative Safe Frontier")
    lines.append("")
    lines.append("| order | candidate | worst_matched_contest | worst_matched_strict | worst_rel | status |")
    lines.append("|---|---|---:|---:|---:|---|")
    for i, row in enumerate(frontier, start=1):
        lines.append(
            f"| {i} | {row['candidate']} | {_fmt(row['worst_matched_ratio_contest'], '.6f')} | "
            f"{_fmt(row['worst_matched_ratio_strict'], '.6f')} | "
            f"{_fmt(row['worst_max_rel'], '.4e')} | {row['status']} |"
        )
    lines.append("")

    lines.append("## Sampled Results")
    lines.append("")
    lines.append("| candidate | phase | workload | seq_len | matched_contest | matched_strict | max_abs | max_rel | failure |")
    lines.append("|---|---|---|---:|---:|---:|---:|---:|---|")
    for row in rows:
        lines.append(
            f"| {row['candidate']} | {row['search_phase']} | "
            f"{_canonical_short_uuid(row['workload_uuid'])} | {row['seq_len']} | "
            f"{_fmt(row['matched_ratio_contest'], '.6f')} | {_fmt(row['matched_ratio_strict'], '.6f')} | "
            f"{_fmt(row['max_abs'], '.4e')} | "
            f"{_fmt(row['max_rel'], '.4e')} | {row['failure_label']} |"
        )
    (out_dir / "summary.md").write_text("\n".join(lines))


def _route(inputs: dict, routed_scaling_factor: float, candidate: Candidate):
    import torch

    logits = inputs["routing_logits"].to(torch.float32)
    bias = inputs["routing_bias"].to(torch.float32).reshape(-1)
    if candidate.stage == "routing_math":
        logits = _apply_precision(logits, candidate.mode, candidate.scale_mode)
        bias = _apply_precision(bias, candidate.mode, candidate.scale_mode)
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
    if candidate.stage == "routing_weights":
        weights = _apply_precision(weights, candidate.mode, candidate.scale_mode)
    return topk_idx, weights


def _run_gemm(a_e, w_e, candidate: Candidate, op_stage: str, acc_stage: str):
    if candidate.stage == op_stage:
        a_e = _apply_precision(a_e, candidate.mode, candidate.scale_mode)
        w_e = _apply_precision(w_e, candidate.mode, candidate.scale_mode)
    if candidate.stage == acc_stage:
        return _accumulate_blockwise(a_e, w_e.t().contiguous(), BLOCK, candidate.mode)
    return a_e.matmul(w_e.t())


def _run_moe(inputs: dict, candidate: Candidate):
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
    if candidate.stage == "hidden_dequant":
        a = _apply_precision(a, candidate.mode, candidate.scale_mode)
    topk_idx, weights = _route(inputs, routed_scaling, candidate)

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
        g1 = _run_gemm(a_e, w13_e, candidate, "gemm1_operands", "gemm1_accumulator")
        if candidate.stage == "gemm1_output":
            g1 = _apply_precision(g1, candidate.mode, candidate.scale_mode)
        x1 = g1[:, :INTERMEDIATE]
        x2 = g1[:, INTERMEDIATE:]
        if candidate.stage == "swiglu_input":
            x1 = _apply_precision(x1, candidate.mode, candidate.scale_mode)
            x2 = _apply_precision(x2, candidate.mode, candidate.scale_mode)
        c = x1 * torch.nn.functional.silu(x2)
        if candidate.stage == "swiglu_output":
            c = _apply_precision(c, candidate.mode, candidate.scale_mode)
        w2_scale = s2[le].repeat_interleave(BLOCK, dim=0).repeat_interleave(BLOCK, dim=1)
        w2_e = w2[le].to(torch.float32) * w2_scale
        o = _run_gemm(c, w2_e, candidate, "gemm2_operands", "gemm2_accumulator")
        w_tok = weights.index_select(0, tok)[:, ge]
        contrib = o * w_tok.unsqueeze(1)
        if candidate.stage == "out_accumulator":
            current = out.index_select(0, tok) + contrib
            current = _apply_precision(current, candidate.mode, candidate.scale_mode)
            out.index_copy_(0, tok, current)
        else:
            out.index_add_(0, tok, contrib)
    out_bf16 = out.to(torch.bfloat16).to(torch.float32)
    if candidate.stage == "final_output":
        out_bf16 = _apply_precision(out_bf16, candidate.mode, candidate.scale_mode)
    return out_bf16.to(torch.bfloat16), selected_experts


def _panel_indices(workloads: list, panel_size: int) -> list[int]:
    indexed = list(enumerate(workloads))
    by_uuid = {_canonical_short_uuid(w.workload.uuid): i for i, w in indexed}
    picked: list[int] = []
    for short in KNOWN_PANEL_UUIDS:
        idx = by_uuid.get(short)
        if idx is not None and idx not in picked:
            picked.append(idx)

    seq_sorted = indexed
    anchors = [0, len(indexed) // 4, len(indexed) // 2, (3 * len(indexed)) // 4, len(indexed) - 1]
    for pos in anchors:
        if not indexed:
            break
        idx = seq_sorted[max(0, min(pos, len(seq_sorted) - 1))][0]
        if idx not in picked:
            picked.append(idx)

    for idx, _ in indexed:
        if len(picked) >= panel_size:
            break
        if idx not in picked:
            picked.append(idx)
    return sorted(picked[:panel_size])


def _aggregate_candidate(rows: Iterable[dict]) -> dict:
    rows = list(rows)
    worst_matched_contest = min(r["matched_ratio_contest"] for r in rows)
    worst_matched_strict = min(r["matched_ratio_strict"] for r in rows)
    worst_rel = max(r["max_rel"] for r in rows)
    worst_abs = max(r["max_abs"] for r in rows)
    return {
        "worst_matched_ratio_contest": worst_matched_contest,
        "worst_matched_ratio_strict": worst_matched_strict,
        "worst_max_rel": worst_rel,
        "worst_max_abs": worst_abs,
        "status": "safe",
    }


def _matched_ratio(diff, rel, *, atol: float, rtol: float) -> float:
    exceeds = (diff > atol) & (rel > rtol)
    return 1.0 - float(exceeds.sum().item()) / float(exceeds.numel())


def _probe_impl(seed: int, panel_size: int, atol: float, rtol: float, required_matched_ratio: float) -> dict:
    import torch
    from flashinfer_bench import TraceSet
    from flashinfer_bench.bench.utils import gen_inputs, load_safetensors

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    trace_set = TraceSet.from_path(TRACE_SET_PATH)
    definition = trace_set.definitions[DEFINITION]
    workloads = trace_set.workloads[DEFINITION]
    panel = _panel_indices(workloads, panel_size)

    rows: list[dict] = []
    candidates = _candidate_table()
    candidate_agg: dict[str, dict] = {}

    for workload_index in panel:
        trace = workloads[workload_index]
        safe = load_safetensors(definition, trace.workload, trace_set.root)
        values = gen_inputs(definition, trace.workload, "cuda:0", safe)
        inputs = dict(zip(definition.inputs.keys(), values))

        ref, _ = _run_moe(inputs, Candidate("baseline", "fp32"))
        for cand in candidates:
            got, selected_experts = (ref, 0) if cand.stage == "baseline" else _run_moe(inputs, cand)
            diff = (got.to(torch.float32) - ref.to(torch.float32)).abs()
            rel = diff / (ref.to(torch.float32).abs() + 1e-8)
            matched_contest = _matched_ratio(diff, rel, atol=atol, rtol=rtol)
            matched_strict = _matched_ratio(diff, rel, atol=STRICT_ATOL, rtol=STRICT_RTOL)
            row = {
                "candidate": cand.name,
                "search_phase": "single_stage",
                "workload_index": workload_index,
                "workload_uuid": trace.workload.uuid,
                "seq_len": int(inputs["hidden_states"].shape[0]),
                "stage": cand.stage,
                "mode": cand.mode,
                "scale_mode": cand.scale_mode,
                "scope": cand.scope,
                "max_abs": float(diff.max().item()),
                "max_rel": float(rel.max().item()),
                "mean_abs": float(diff.mean().item()),
                "matched_ratio_contest": matched_contest,
                "matched_ratio_strict": matched_strict,
                "selected_experts": selected_experts,
                "failure_label": _classify_failure(
                    matched_contest,
                    float(rel.max().item()),
                    float(diff.max().item()),
                    pass_threshold=required_matched_ratio,
                ),
            }
            rows.append(row)

    for cand in candidates:
        cand_rows = [r for r in rows if r["candidate"] == cand.name and r["search_phase"] == "single_stage"]
        agg = _aggregate_candidate(cand_rows)
        agg.update({
            "candidate": cand.name,
            "stage": cand.stage,
            "mode": cand.mode,
            "scale_mode": cand.scale_mode,
            "scope": cand.scope,
        })
        agg["status"] = "safe" if agg["worst_matched_ratio_contest"] >= required_matched_ratio else "unsafe"
        candidate_agg[cand.name] = agg

    survivors = [
        cand for cand in candidates
        if cand.stage != "baseline" and candidate_agg[cand.name]["status"] == "safe"
    ]
    strict_survivors = [
        cand
        for cand in candidates
        if cand.stage != "baseline"
        and candidate_agg[cand.name]["worst_matched_ratio_strict"] >= required_matched_ratio
    ]
    strict_survivor_names = {cand.name for cand in strict_survivors}
    contest_only_survivors = [cand for cand in survivors if cand.name not in strict_survivor_names]
    survivors.sort(
        key=lambda cand: (
            candidate_agg[cand.name]["worst_matched_ratio_contest"],
            -candidate_agg[cand.name]["worst_max_rel"],
            -_mode_rank(cand.mode),
        ),
        reverse=True,
    )

    frontier_rows: list[dict] = []
    combined = _combined_candidates(survivors)
    active: list[Candidate] = []
    for cand in combined:
        active.append(cand)
        combined_name = "+".join(c.name for c in active)
        staged_rows = []
        for workload_index in panel:
            trace = workloads[workload_index]
            safe = load_safetensors(definition, trace.workload, trace_set.root)
            values = gen_inputs(definition, trace.workload, "cuda:0", safe)
            inputs = dict(zip(definition.inputs.keys(), values))
            ref, _ = _run_moe(inputs, Candidate("baseline", "fp32"))
            if len(active) == 1:
                got, selected_experts = _run_moe(inputs, active[0])
            else:
                got, selected_experts = _run_moe_with_candidates(inputs, active)
            diff = (got.to(torch.float32) - ref.to(torch.float32)).abs()
            rel = diff / (ref.to(torch.float32).abs() + 1e-8)
            matched_contest = _matched_ratio(diff, rel, atol=atol, rtol=rtol)
            matched_strict = _matched_ratio(diff, rel, atol=STRICT_ATOL, rtol=STRICT_RTOL)
            row = {
                "candidate": combined_name,
                "search_phase": "cumulative",
                "workload_index": workload_index,
                "workload_uuid": trace.workload.uuid,
                "seq_len": int(inputs["hidden_states"].shape[0]),
                "stage": "+".join(c.stage for c in active),
                "mode": "+".join(c.mode for c in active),
                "scale_mode": "+".join(c.scale_mode for c in active),
                "scope": "+".join(c.scope for c in active),
                "max_abs": float(diff.max().item()),
                "max_rel": float(rel.max().item()),
                "mean_abs": float(diff.mean().item()),
                "matched_ratio_contest": matched_contest,
                "matched_ratio_strict": matched_strict,
                "selected_experts": selected_experts,
                "failure_label": _classify_failure(
                    matched_contest,
                    float(rel.max().item()),
                    float(diff.max().item()),
                    pass_threshold=required_matched_ratio,
                ),
            }
            rows.append(row)
            staged_rows.append(row)
        agg = _aggregate_candidate(staged_rows)
        agg["candidate"] = combined_name
        agg["status"] = "safe" if agg["worst_matched_ratio_contest"] >= required_matched_ratio else "unsafe"
        frontier_rows.append(agg)
        if agg["status"] != "safe":
            break

    stage_summary: list[dict] = []
    stages = sorted({cand.stage for cand in candidates if cand.stage != "baseline"})
    for stage in stages:
        stage_cands = [candidate_agg[c.name] for c in candidates if c.stage == stage]
        safe_stage = [r for r in stage_cands if r["status"] == "safe"]
        if safe_stage:
            safe_stage.sort(
                key=lambda r: (_mode_rank(r["mode"]), r["scale_mode"]),
                reverse=True,
            )
            best = safe_stage[0]
            stage_summary.append(
                {
                    "stage": stage,
                    "best_safe_mode": best["mode"],
                    "scale_mode": best["scale_mode"],
                    "scope": best["scope"],
                    "worst_matched_ratio_contest": best["worst_matched_ratio_contest"],
                    "worst_matched_ratio_strict": best["worst_matched_ratio_strict"],
                    "worst_max_rel": best["worst_max_rel"],
                    "status": "safe",
                }
            )
        else:
            first = min(stage_cands, key=lambda r: r["worst_matched_ratio_contest"])
            stage_summary.append(
                {
                    "stage": stage,
                    "best_safe_mode": "-",
                    "scale_mode": "-",
                    "scope": "-",
                    "worst_matched_ratio_contest": first["worst_matched_ratio_contest"],
                    "worst_matched_ratio_strict": first["worst_matched_ratio_strict"],
                    "worst_max_rel": first["worst_max_rel"],
                    "status": "unsafe",
                }
            )

    meta = {
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "definition": DEFINITION,
        "seed": seed,
        "panel_size": len(panel),
        "panel_indices": ",".join(str(i) for i in panel),
        "atol": atol,
        "rtol": rtol,
        "required_matched_ratio": required_matched_ratio,
        "strict_atol": STRICT_ATOL,
        "strict_rtol": STRICT_RTOL,
        "device": torch.cuda.get_device_name(0),
        "n_candidates": len(candidates),
        "n_survivors": len(survivors),
        "contest_safe_survivor_count": len(survivors),
        "contest_safe_survivors": ", ".join(c.name for c in survivors) or "-",
        "strict_safe_survivor_count": len(strict_survivors),
        "strict_safe_survivors": ", ".join(c.name for c in strict_survivors) or "-",
        "contest_only_survivor_count": len(contest_only_survivors),
        "contest_only_survivors": ", ".join(c.name for c in contest_only_survivors) or "-",
    }
    return {
        "meta": meta,
        "rows": rows,
        "stage_summary": stage_summary,
        "frontier": frontier_rows,
    }


def _run_moe_with_candidates(inputs: dict, candidates: list[Candidate]):
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
    cand_by_stage = {c.stage: c for c in candidates}

    a_scale = hidden_scale.t().contiguous()
    a = hidden.to(torch.float32) * a_scale.repeat_interleave(BLOCK, dim=1)
    if "hidden_dequant" in cand_by_stage:
        c = cand_by_stage["hidden_dequant"]
        a = _apply_precision(a, c.mode, c.scale_mode)
    topk_idx, weights = _route(inputs, routed_scaling, cand_by_stage.get("routing_math", cand_by_stage.get("routing_weights", Candidate("baseline", "fp32"))))
    if "routing_weights" in cand_by_stage:
        c = cand_by_stage["routing_weights"]
        weights = _apply_precision(weights, c.mode, c.scale_mode)

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
        g1 = _run_gemm(a_e, w13_e, cand_by_stage.get("gemm1_operands", cand_by_stage.get("gemm1_accumulator", Candidate("baseline", "fp32"))), "gemm1_operands", "gemm1_accumulator")
        if "gemm1_output" in cand_by_stage:
            c = cand_by_stage["gemm1_output"]
            g1 = _apply_precision(g1, c.mode, c.scale_mode)
        x1 = g1[:, :INTERMEDIATE]
        x2 = g1[:, INTERMEDIATE:]
        if "swiglu_input" in cand_by_stage:
            c = cand_by_stage["swiglu_input"]
            x1 = _apply_precision(x1, c.mode, c.scale_mode)
            x2 = _apply_precision(x2, c.mode, c.scale_mode)
        c_act = x1 * torch.nn.functional.silu(x2)
        if "swiglu_output" in cand_by_stage:
            c = cand_by_stage["swiglu_output"]
            c_act = _apply_precision(c_act, c.mode, c.scale_mode)
        w2_scale = s2[le].repeat_interleave(BLOCK, dim=0).repeat_interleave(BLOCK, dim=1)
        w2_e = w2[le].to(torch.float32) * w2_scale
        o = _run_gemm(c_act, w2_e, cand_by_stage.get("gemm2_operands", cand_by_stage.get("gemm2_accumulator", Candidate("baseline", "fp32"))), "gemm2_operands", "gemm2_accumulator")
        w_tok = weights.index_select(0, tok)[:, ge]
        contrib = o * w_tok.unsqueeze(1)
        if "out_accumulator" in cand_by_stage:
            c = cand_by_stage["out_accumulator"]
            current = out.index_select(0, tok) + contrib
            current = _apply_precision(current, c.mode, c.scale_mode)
            out.index_copy_(0, tok, current)
        else:
            out.index_add_(0, tok, contrib)
    out_bf16 = out.to(torch.bfloat16).to(torch.float32)
    if "final_output" in cand_by_stage:
        c = cand_by_stage["final_output"]
        out_bf16 = _apply_precision(out_bf16, c.mode, c.scale_mode)
    return out_bf16.to(torch.bfloat16), selected_experts


@app.function(**_COMMON, gpu="B200:1")
def probe_b200(seed: int, panel_size: int, atol: float, rtol: float, required_matched_ratio: float) -> dict:
    return _probe_impl(seed, panel_size, atol, rtol, required_matched_ratio)


@app.local_entrypoint()
def probe(
    seed: int = 1234,
    panel_size: int = DEFAULT_PANEL_SIZE,
    label: str = "",
    atol: float = DEFAULT_ATOL,
    rtol: float = DEFAULT_RTOL,
    required_matched_ratio: float = DEFAULT_REQUIRED_MATCHED_RATIO,
):
    result = probe_b200.remote(seed, panel_size, atol, rtol, required_matched_ratio)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    suffix = f"_{label}" if label else ""
    out_dir = PROJECT_ROOT / "experiments" / f"{ts}_B200_moe_precision_search{suffix}"
    _write_outputs(
        out_dir,
        result["rows"],
        result["meta"],
        result["stage_summary"],
        result["frontier"],
    )
    print(f"Wrote {out_dir / 'summary.md'}")
    print(f"Stage summary: {out_dir / 'stage_summary.json'}")
    print(f"Cumulative frontier: {out_dir / 'safe_frontier.json'}")
