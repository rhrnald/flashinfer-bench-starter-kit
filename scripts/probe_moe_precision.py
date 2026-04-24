"""
Search MoE precision-reduction candidates without changing the CUDA baseline.

This script runs a PyTorch oracle over a representative workload panel and
evaluates stage-scoped precision reductions one at a time, then cumulatively.
The search is declarative: candidates are generated from stage/mode/scale/scope
metadata rather than hardcoded for one hypothesis.

Important:
    This is oracle-only evidence. It does not validate the real CUDA kernel path.
    The primary pass/fail gate matches the contest MoE tolerance
    (`atol=1`, `rtol=0.3`, `required_matched_ratio=0.9`). The stricter
    `0.01/0.01` metric is reported as a shadow check only.

Usage:
    modal run scripts/probe_moe_precision.py::probe
"""

from __future__ import annotations

import csv
import itertools
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
DEFAULT_PANEL_SIZE = 19
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

BF16_F16_STAGES = (
    "hidden_dequant",
    "gemm1_operands",
    "gemm1_accumulator",
    "gemm1_output",
    "swiglu_input",
    "swiglu_output",
    "gemm2_operands",
    "gemm2_accumulator",
    "out_accumulator",
)

DEFAULT_PAIRWISE_SHORTLIST = (
    "hidden_dequant__f16",
    "gemm1_operands__f16",
    "gemm1_accumulator__f16",
    "gemm1_output__f16",
    "swiglu_output__f16",
    "gemm2_accumulator__bf16",
    "out_accumulator__bf16",
)

DEFAULT_STRESS_CANDIDATES = (
    "gemm2_accumulator__bf16",
)

DEFAULT_FP8_TARGET_STAGES = (
    "hidden_dequant",
    "gemm1_operands",
    "gemm1_output",
    "swiglu_input",
    "swiglu_output",
    "gemm2_operands",
    "gemm2_accumulator",
    "out_accumulator",
)
DEFAULT_PROMOTE_TOP_K_PER_STAGE = 1
DEFAULT_ENABLE_KERNEL_VALIDATION = True
DEFAULT_KERNEL_VALIDATION_LIMIT = 5
EVIDENCE_SCOPE = "oracle_only"
KERNEL_EVIDENCE_SCOPE = "kernel_validated"
KERNEL_UNSUPPORTED_SCOPE = "kernel_not_supported"


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


@dataclass(frozen=True)
class EvalConfig:
    name: str
    candidates: tuple[Candidate, ...]
    search_phase: str

    @property
    def stage(self) -> str:
        if not self.candidates:
            return "baseline"
        return "+".join(c.stage for c in self.candidates)

    @property
    def mode(self) -> str:
        if not self.candidates:
            return "fp32"
        return "+".join(c.mode for c in self.candidates)

    @property
    def scale_mode(self) -> str:
        if not self.candidates:
            return "none"
        return "+".join(c.scale_mode for c in self.candidates)

    @property
    def scope(self) -> str:
        if not self.candidates:
            return "storage_compute"
        return "+".join(c.scope for c in self.candidates)


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


def _scale_rank(scale_mode: str) -> int:
    order = {"none": 0, "tensor": 1, "row": 2, "block": 3}
    return order.get(scale_mode, -1)


def _pipeline_stage_rank(stage: str) -> int:
    order = {
        "hidden_dequant": 0,
        "gemm1_operands": 1,
        "gemm1_output": 2,
        "swiglu_input": 3,
        "swiglu_output": 4,
        "gemm2_operands": 5,
        "gemm2_accumulator": 6,
        "out_accumulator": 7,
    }
    return order.get(stage, 999)


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
            if acc_mode in ("bf16", "f16"):
                acc = _round_tensor(acc, acc_mode)
            else:
                acc = _apply_precision(acc, acc_mode, "tensor")
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


def _build_bf16_f16_candidates() -> list[Candidate]:
    table = [Candidate("baseline", "fp32")]
    for stage in BF16_F16_STAGES:
        for mode in ("bf16", "f16"):
            table.append(Candidate(stage, mode))
    return table


def _build_fp8_candidates(stages: Iterable[str]) -> list[Candidate]:
    table: list[Candidate] = []
    for stage in stages:
        table.extend(
            [
                Candidate(stage, "fp8", "tensor"),
                Candidate(stage, "fp8", "row"),
                Candidate(stage, "fp8", "block"),
            ]
        )
    return table


def _pick_top_fp8_candidates(
    candidate_aggs: dict[str, dict],
    fp8_candidates: list[Candidate],
    top_k_per_stage: int,
) -> list[Candidate]:
    chosen: list[Candidate] = []
    stages = sorted({cand.stage for cand in fp8_candidates}, key=_pipeline_stage_rank)
    for stage in stages:
        stage_cands = []
        for cand in fp8_candidates:
            if cand.stage != stage:
                continue
            agg = candidate_aggs.get(cand.name)
            if agg is None or agg["status"] != "safe":
                continue
            stage_cands.append((cand, agg))
        stage_cands.sort(
            key=lambda item: (
                item[1]["worst_matched_ratio_contest"],
                item[1]["worst_matched_ratio_strict"],
                -item[1]["worst_max_rel"],
                _scale_rank(item[0].scale_mode),
            ),
            reverse=True,
        )
        chosen.extend(cand for cand, _ in stage_cands[:top_k_per_stage])
    return chosen


def _candidate_from_name(name: str) -> Candidate:
    parts = name.split("__")
    if len(parts) < 2:
        raise ValueError(f"Invalid candidate name {name!r}")
    stage = parts[0]
    mode = parts[1]
    scale_mode = "none"
    scope = "storage_compute"
    if len(parts) >= 3:
        if parts[2] in {"tensor", "row", "block"}:
            scale_mode = parts[2]
            if len(parts) >= 4:
                scope = parts[3]
        else:
            scope = parts[2]
    return Candidate(stage, mode, scale_mode=scale_mode, scope=scope)


def _parse_targeted_config_specs(config_specs: str) -> list[tuple[str, tuple[Candidate, ...]]]:
    configs: list[tuple[str, tuple[Candidate, ...]]] = []
    for raw_spec in config_specs.split(","):
        spec = raw_spec.strip()
        if not spec:
            continue
        if "=" in spec:
            config_name, candidate_blob = spec.split("=", 1)
            config_name = config_name.strip()
        else:
            config_name = spec
            candidate_blob = spec
        candidates = tuple(_candidate_from_name(part.strip()) for part in candidate_blob.split("+") if part.strip())
        if not candidates:
            raise ValueError(f"Targeted config {spec!r} did not include any candidates")
        configs.append((config_name, candidates))
    if not configs:
        raise ValueError("No targeted configs were provided")
    return configs


def _combined_candidates(survivors: list[Candidate]) -> list[Candidate]:
    return [
        cand
        for cand in sorted(survivors, key=lambda c: (_pipeline_stage_rank(c.stage), c.name))
        if cand.stage != "baseline"
    ]


def _make_eval_config(candidates: Iterable[Candidate], search_phase: str, name: str | None = None) -> EvalConfig:
    cands = tuple(candidates)
    if name is None:
        name = "baseline__fp32" if not cands else "+".join(c.name for c in cands)
    return EvalConfig(name=name, candidates=cands, search_phase=search_phase)


def _candidate_by_name(candidates: Iterable[Candidate]) -> dict[str, Candidate]:
    return {c.name: c for c in candidates}


def _write_outputs(
    out_dir: Path,
    rows: list[dict],
    meta: dict,
    stage_summary: list[dict],
    frontier: list[dict],
    extra_json: dict[str, Any] | None = None,
) -> None:
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
    if extra_json:
        for filename, payload in extra_json.items():
            (out_dir / filename).write_text(json.dumps(payload, indent=2))

    lines = [f"# MoE Precision Search {meta['timestamp']}", ""]
    lines.append("| metric | value |")
    lines.append("|---|---|")
    for k, v in meta.items():
        lines.append(f"| {k} | {v} |")
    lines.append("")

    lines.append("## Evidence Scope")
    lines.append("")
    lines.append("| field | value |")
    lines.append("|---|---|")
    lines.append(f"| evidence_scope | {meta.get('evidence_scope', EVIDENCE_SCOPE)} |")
    lines.append(f"| kernel_validated_candidates | {meta.get('kernel_validated_candidates', '-')} |")
    lines.append("")

    lines.append("## Oracle Survivor Summary")
    lines.append("")
    lines.append("| category | count | candidates |")
    lines.append("|---|---:|---|")
    lines.append(
        f"| contest_safe_oracle_single_stage | {meta['contest_safe_survivor_count']} | {meta['contest_safe_survivors']} |"
    )
    lines.append(
        f"| strict_safe_oracle_single_stage | {meta['strict_safe_survivor_count']} | {meta['strict_safe_survivors']} |"
    )
    lines.append(
        f"| contest_only_oracle_single_stage | {meta['contest_only_survivor_count']} | {meta['contest_only_survivors']} |"
    )
    lines.append("")

    lines.append("## Stage Summary")
    lines.append("")
    lines.append("| stage | best_contest_safe_mode | scale | worst_matched_contest | worst_matched_strict | worst_rel | contest_status | strict_status | evidence |")
    lines.append("|---|---|---|---:|---:|---:|---|---|---|")
    for row in stage_summary:
        lines.append(
            f"| {row['stage']} | {row['best_safe_mode']} | {row['scale_mode']} | "
            f"{_fmt(row['worst_matched_ratio_contest'], '.6f')} | {_fmt(row['worst_matched_ratio_strict'], '.6f')} | "
            f"{_fmt(row['worst_max_rel'], '.4e')} | "
            f"{row['contest_status']} | {row['strict_status']} | {row['evidence_scope']} |"
        )
    lines.append("")

    lines.append("## Contest-Safe Oracle Frontier")
    lines.append("")
    lines.append("| order | candidate | worst_matched_contest | worst_matched_strict | worst_rel | contest_status | strict_status | evidence | kept |")
    lines.append("|---|---|---:|---:|---:|---|---|---|---|")
    for i, row in enumerate(frontier, start=1):
        kept = "yes" if row.get("kept", row.get("status") == "safe") else "no"
        lines.append(
            f"| {i} | {row['candidate']} | {_fmt(row['worst_matched_ratio_contest'], '.6f')} | "
            f"{_fmt(row['worst_matched_ratio_strict'], '.6f')} | "
            f"{_fmt(row['worst_max_rel'], '.4e')} | {row['contest_status']} | {row['strict_status']} | {row['evidence_scope']} | {kept} |"
        )
    lines.append("")

    if extra_json and "margin_summary.json" in extra_json:
        margin_summary = extra_json["margin_summary.json"]
        lines.append("## BF16/F16 Margin")
        lines.append("")
        lines.append("| stage | preferred | bf16_contest | f16_contest | bf16_strict | f16_strict | |")
        lines.append("|---|---|---:|---:|---:|---:|---|")
        for row in margin_summary:
            lines.append(
                f"| {row['stage']} | {row['preferred_mode']} | "
                f"{_fmt(row['bf16_contest'], '.6f')} | {_fmt(row['f16_contest'], '.6f')} | "
                f"{_fmt(row['bf16_strict'], '.6f')} | {_fmt(row['f16_strict'], '.6f')} | |"
            )
        lines.append("")

    if extra_json and "pairwise_summary.json" in extra_json:
        pairwise_summary = extra_json["pairwise_summary.json"]
        lines.append("## Pairwise Summary")
        lines.append("")
        lines.append("| pair | worst_matched_contest | worst_matched_strict | worst_rel | contest_status | strict_status | evidence |")
        lines.append("|---|---:|---:|---:|---|---|---|")
        for row in pairwise_summary:
            lines.append(
                f"| {row['candidate']} | {_fmt(row['worst_matched_ratio_contest'], '.6f')} | "
                f"{_fmt(row['worst_matched_ratio_strict'], '.6f')} | "
                f"{_fmt(row['worst_max_rel'], '.4e')} | {row['contest_status']} | {row['strict_status']} | {row['evidence_scope']} |"
            )
        lines.append("")

    if extra_json and "stress_summary.json" in extra_json:
        stress_summary = extra_json["stress_summary.json"]
        lines.append("## Stress Summary")
        lines.append("")
        lines.append("| candidate | worst_workload | worst_seq_len | worst_matched_contest | worst_matched_strict | contest_status | strict_status | evidence |")
        lines.append("|---|---|---:|---:|---:|---|---|---|")
        for row in stress_summary:
            lines.append(
                f"| {row['candidate']} | {row['worst_workload']} | {row['worst_seq_len']} | "
                f"{_fmt(row['worst_matched_ratio_contest'], '.6f')} | {_fmt(row['worst_matched_ratio_strict'], '.6f')} | "
                f"{row['contest_status']} | {row['strict_status']} | {row['evidence_scope']} |"
            )
        lines.append("")

    if extra_json and "promotion_summary.json" in extra_json:
        promotion = extra_json["promotion_summary.json"]
        lines.append("## Promotion Summary")
        lines.append("")
        lines.append("| category | candidates |")
        lines.append("|---|---|")
        for key, value in promotion.items():
            if isinstance(value, list):
                lines.append(f"| {key} | {', '.join(value) or '-'} |")
        lines.append("")

    if extra_json and "kernel_validation_summary.json" in extra_json:
        kernel_summary = extra_json["kernel_validation_summary.json"]
        lines.append("## Kernel Validation Summary")
        lines.append("")
        lines.append("| candidate | validation_status | workloads_passed | total_workloads | worst_max_abs | worst_max_rel | evidence | notes |")
        lines.append("|---|---|---:|---:|---:|---:|---|---|")
        for row in kernel_summary:
            lines.append(
                f"| {row['candidate']} | {row['validation_status']} | "
                f"{row.get('workloads_passed', '-')} | {row.get('total_workloads', '-')} | "
                f"{_fmt(row.get('worst_max_abs'), '.4e')} | {_fmt(row.get('worst_max_rel'), '.4e')} | "
                f"{row.get('evidence_scope', '-')} | {row.get('notes', '-')} |"
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


def _write_targeted_outputs(out_dir: Path, rows: list[dict], meta: dict[str, Any], config_summary: list[dict]) -> None:
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
    (out_dir / "config_summary.json").write_text(json.dumps(config_summary, indent=2))

    lines = [f"# MoE Targeted Precision Search {meta['timestamp']}", ""]
    lines.append("| metric | value |")
    lines.append("|---|---|")
    for k, v in meta.items():
        lines.append(f"| {k} | {v} |")
    lines.append("")
    lines.append("## Config Summary")
    lines.append("")
    lines.append("| config | worst_matched_contest | worst_matched_strict | worst_max_rel | contest_status | strict_status |")
    lines.append("|---|---:|---:|---:|---|---|")
    for row in config_summary:
        lines.append(
            f"| {row['candidate']} | {_fmt(row['worst_matched_ratio_contest'], '.6f')} | "
            f"{_fmt(row['worst_matched_ratio_strict'], '.6f')} | {_fmt(row['worst_max_rel'], '.4e')} | "
            f"{row['contest_status']} | {row['strict_status']} |"
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
            f"{_fmt(row['max_abs'], '.4e')} | {_fmt(row['max_rel'], '.4e')} | {row['failure_label']} |"
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
        "evidence_scope": EVIDENCE_SCOPE,
    }


def _matched_ratio(diff, rel, *, atol: float, rtol: float) -> float:
    exceeds = (diff > atol) & (rel > rtol)
    return 1.0 - float(exceeds.sum().item()) / float(exceeds.numel())


def _load_panel_records(trace_set, definition, workloads, panel: list[int]) -> list[tuple[int, Any]]:
    return [(idx, workloads[idx]) for idx in panel]


def _evaluate_configs(
    panel_records: list[tuple[int, Any]],
    definition,
    trace_set,
    configs: list[EvalConfig],
    *,
    atol: float,
    rtol: float,
    required_matched_ratio: float,
) -> list[dict]:
    import torch

    rows: list[dict] = []
    from flashinfer_bench.bench.utils import gen_inputs, load_safetensors

    for workload_index, trace in panel_records:
        safe = load_safetensors(definition, trace.workload, trace_set.root)
        values = gen_inputs(definition, trace.workload, "cuda:0", safe)
        inputs = dict(zip(definition.inputs.keys(), values))
        ref, _ = _run_moe(inputs, Candidate("baseline", "fp32"))
        for cfg in configs:
            if not cfg.candidates:
                got, selected_experts = ref, 0
            elif len(cfg.candidates) == 1:
                got, selected_experts = _run_moe(inputs, cfg.candidates[0])
            else:
                got, selected_experts = _run_moe_with_candidates(inputs, list(cfg.candidates))
            diff = (got.to(torch.float32) - ref.to(torch.float32)).abs()
            rel = diff / (ref.to(torch.float32).abs() + 1e-8)
            matched_contest = _matched_ratio(diff, rel, atol=atol, rtol=rtol)
            matched_strict = _matched_ratio(diff, rel, atol=STRICT_ATOL, rtol=STRICT_RTOL)
            row = {
                "candidate": cfg.name,
                "search_phase": cfg.search_phase,
                "workload_index": workload_index,
                "workload_uuid": trace.workload.uuid,
                "seq_len": int(inputs["hidden_states"].shape[0]),
                "stage": cfg.stage,
                "mode": cfg.mode,
                "scale_mode": cfg.scale_mode,
                "scope": cfg.scope,
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
    return rows


def _aggregate_configs(
    rows: list[dict],
    configs: list[EvalConfig],
    *,
    required_matched_ratio: float,
) -> dict[str, dict]:
    aggs: dict[str, dict] = {}
    for cfg in configs:
        cfg_rows = [r for r in rows if r["candidate"] == cfg.name]
        agg = _aggregate_candidate(cfg_rows)
        agg.update(
            {
                "candidate": cfg.name,
                "stage": cfg.stage,
                "mode": cfg.mode,
                "scale_mode": cfg.scale_mode,
                "scope": cfg.scope,
            }
        )
        contest_safe = agg["worst_matched_ratio_contest"] >= required_matched_ratio
        strict_safe = agg["worst_matched_ratio_strict"] >= required_matched_ratio
        agg["status"] = "safe" if contest_safe else "unsafe"
        agg["contest_status"] = "contest_safe" if contest_safe else "contest_unsafe"
        agg["strict_status"] = "strict_safe" if strict_safe else "strict_unsafe"
        aggs[cfg.name] = agg
    return aggs


def _build_stage_summary(candidate_aggs: dict[str, dict], candidates: list[Candidate]) -> list[dict]:
    stage_summary: list[dict] = []
    stages = sorted({cand.stage for cand in candidates if cand.stage != "baseline"})
    for stage in stages:
        stage_cands = [candidate_aggs[c.name] for c in candidates if c.stage == stage and c.name in candidate_aggs]
        if not stage_cands:
            continue
        safe_stage = [r for r in stage_cands if r["status"] == "safe"]
        if safe_stage:
            safe_stage.sort(key=lambda r: (_mode_rank(r["mode"]), r["scale_mode"]), reverse=True)
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
                    "contest_status": best["contest_status"],
                    "strict_status": best["strict_status"],
                    "evidence_scope": best["evidence_scope"],
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
                    "contest_status": first["contest_status"],
                    "strict_status": first["strict_status"],
                    "evidence_scope": first["evidence_scope"],
                }
            )
    return stage_summary


def _build_margin_summary(candidate_aggs: dict[str, dict], stages: Iterable[str]) -> list[dict]:
    summary: list[dict] = []
    for stage in stages:
        bf16 = candidate_aggs.get(Candidate(stage, "bf16").name)
        f16 = candidate_aggs.get(Candidate(stage, "f16").name)
        if bf16 is None or f16 is None:
            continue
        preferred = "f16" if f16["worst_matched_ratio_contest"] >= bf16["worst_matched_ratio_contest"] else "bf16"
        summary.append(
            {
                "stage": stage,
                "preferred_mode": preferred,
                "bf16_contest": bf16["worst_matched_ratio_contest"],
                "f16_contest": f16["worst_matched_ratio_contest"],
                "bf16_strict": bf16["worst_matched_ratio_strict"],
                "f16_strict": f16["worst_matched_ratio_strict"],
                "bf16_margin": bf16["worst_matched_ratio_contest"] - DEFAULT_REQUIRED_MATCHED_RATIO,
                "f16_margin": f16["worst_matched_ratio_contest"] - DEFAULT_REQUIRED_MATCHED_RATIO,
            }
        )
    return summary


def _build_pairwise_shortlist(candidate_aggs: dict[str, dict]) -> list[Candidate]:
    names = [name for name in DEFAULT_PAIRWISE_SHORTLIST if candidate_aggs.get(name, {}).get("status") == "safe"]
    return [_candidate_by_name(_build_bf16_f16_candidates())[name] for name in names]


def _build_pairwise_configs(shortlist: list[Candidate]) -> list[EvalConfig]:
    configs: list[EvalConfig] = []
    for left, right in itertools.combinations(shortlist, 2):
        configs.append(_make_eval_config((left, right), "pairwise"))
    return configs


def _build_stress_summary(rows: list[dict], aggs: dict[str, dict]) -> list[dict]:
    summary: list[dict] = []
    for candidate, agg in aggs.items():
        cand_rows = [r for r in rows if r["candidate"] == candidate]
        worst = min(cand_rows, key=lambda r: (r["matched_ratio_contest"], r["matched_ratio_strict"]))
        summary.append(
            {
                "candidate": candidate,
                "worst_workload": _canonical_short_uuid(worst["workload_uuid"]),
                "worst_seq_len": worst["seq_len"],
                "worst_matched_ratio_contest": agg["worst_matched_ratio_contest"],
                "worst_matched_ratio_strict": agg["worst_matched_ratio_strict"],
                "status": agg["status"],
                "contest_status": agg["contest_status"],
                "strict_status": agg["strict_status"],
                "evidence_scope": agg["evidence_scope"],
            }
        )
    summary.sort(key=lambda r: (r["worst_matched_ratio_contest"], r["worst_matched_ratio_strict"]))
    return summary


def _kernel_stage_supported(cand: Candidate) -> tuple[bool, str]:
    if cand.stage == "hidden_dequant" and cand.mode == "fp8":
        return False, "top-level hidden_dequant kernel only implements bf16/f16 rounding"
    if cand.stage in {
        "gemm1_operands",
        "gemm1_output",
        "swiglu_input",
        "swiglu_output",
        "gemm2_operands",
        "gemm2_accumulator",
        "out_accumulator",
    }:
        return True, ""
    return False, "kernel precision hook is not implemented for this stage"


def _kernel_config_supported(cfg: EvalConfig) -> tuple[bool, str]:
    unsupported: list[str] = []
    for cand in cfg.candidates:
        ok, reason = _kernel_stage_supported(cand)
        if not ok:
            unsupported.append(f"{cand.name}: {reason}")
    if unsupported:
        return False, "; ".join(unsupported)
    return True, ""


def _build_kernel_validation_configs(
    contest_candidates: list[Candidate],
    contest_analysis: dict[str, Any],
    *,
    limit: int,
) -> list[EvalConfig]:
    configs: list[EvalConfig] = []
    seen: set[str] = set()
    promoted_fp8 = [cand for cand in contest_candidates if cand.mode == "fp8"]
    for cand in promoted_fp8:
        cfg = _make_eval_config((cand,), "kernel_validation")
        if cfg.name not in seen:
            configs.append(cfg)
            seen.add(cfg.name)
    active = contest_analysis.get("active_candidates", [])
    if active:
        cfg = _make_eval_config(tuple(active), "kernel_validation")
        if cfg.name not in seen:
            configs.append(cfg)
            seen.add(cfg.name)
    return configs[:limit]


def _run_kernel_validation(
    configs: list[EvalConfig],
    *,
    atol: float,
    rtol: float,
    required_matched_ratio: float,
) -> list[dict]:
    try:
        from flashinfer_bench import Solution
        from scripts.pack_solution import pack_solution
        from scripts.run_modal import _run_impl
    except Exception as exc:
        return [
            {
                "candidate": cfg.name,
                "validation_status": "kernel_not_supported",
                "workloads_passed": 0,
                "total_workloads": 19,
                "worst_max_abs": None,
                "worst_max_rel": None,
                "evidence_scope": KERNEL_UNSUPPORTED_SCOPE,
                "notes": f"kernel validation helper unavailable in Modal worker: {type(exc).__name__}",
            }
            for cfg in configs
        ]

    rows: list[dict] = []
    solution_path = pack_solution(PROJECT_ROOT / ".tmp_kernel_validation_solution.json")
    solution = Solution.model_validate_json(solution_path.read_text())
    try:
        import os
        from flashinfer_bench import BenchmarkConfig

        for cfg in configs:
            supported, reason = _kernel_config_supported(cfg)
            if not supported:
                rows.append(
                    {
                        "candidate": cfg.name,
                        "validation_status": "kernel_not_supported",
                        "workloads_passed": 0,
                        "total_workloads": 19,
                        "worst_max_abs": None,
                        "worst_max_rel": None,
                        "evidence_scope": KERNEL_UNSUPPORTED_SCOPE,
                        "notes": reason,
                    }
                )
                continue

            env_vars = {
                "FIB_MOE_TC": "1",
                "FIB_MOE_PREC_STAGE": cfg.candidates[0].stage if len(cfg.candidates) == 1 else "",
                "FIB_MOE_PREC_MODE": cfg.candidates[0].mode if len(cfg.candidates) == 1 else "",
                "FIB_MOE_PREC_SCALE": cfg.candidates[0].scale_mode if len(cfg.candidates) == 1 else "none",
                "FIB_MOE_PREC_SCOPE": cfg.candidates[0].scope if len(cfg.candidates) == 1 else "storage_compute",
            }
            if len(cfg.candidates) > 1:
                rows.append(
                    {
                        "candidate": cfg.name,
                        "validation_status": "kernel_not_supported",
                        "workloads_passed": 0,
                        "total_workloads": 19,
                        "worst_max_abs": None,
                        "worst_max_rel": None,
                        "evidence_scope": KERNEL_UNSUPPORTED_SCOPE,
                        "notes": "kernel validation only supports single-stage candidates in this pass",
                    }
                )
                continue

            os.environ.pop("FIB_MOE_LEGACY", None)
            config = BenchmarkConfig(
                warmup_runs=1,
                iterations=3,
                num_trials=1,
                atol=atol,
                rtol=rtol,
                required_matched_ratio=required_matched_ratio,
                use_isolated_runner=False,
                profile_baseline=False,
            )
            result = _run_impl(solution, config, max_workloads=19, env_vars=env_vars)
            traces = result.get(solution.definition, {})
            workload_rows = [v for k, v in traces.items() if not k.startswith("__")]
            passed = sum(1 for v in workload_rows if v.get("status") == "CORRECT")
            worst_abs = max((float(v.get("max_abs_error", 0.0)) for v in workload_rows), default=0.0)
            worst_rel = max((float(v.get("max_rel_error", 0.0)) for v in workload_rows), default=0.0)
            status = "kernel_validated" if passed == len(workload_rows) and workload_rows else "kernel_failed"
            rows.append(
                {
                    "candidate": cfg.name,
                    "validation_status": status,
                    "workloads_passed": passed,
                    "total_workloads": len(workload_rows),
                    "worst_max_abs": worst_abs,
                    "worst_max_rel": worst_rel,
                    "evidence_scope": KERNEL_EVIDENCE_SCOPE if status == "kernel_validated" else "kernel_failed",
                    "notes": "validated through flashinfer-bench benchmark run with FIB_MOE_TC=1",
                }
            )
    finally:
        try:
            solution_path.unlink(missing_ok=True)
        except Exception:
            pass
    return rows


def _analyze_candidates(
    panel_records: list[tuple[int, Any]],
    definition,
    trace_set,
    candidates: list[Candidate],
    *,
    atol: float,
    rtol: float,
    required_matched_ratio: float,
) -> dict[str, Any]:
    configs = [_make_eval_config((), "single_stage", "baseline__fp32")]
    configs.extend(_make_eval_config((cand,), "single_stage", cand.name) for cand in candidates if cand.stage != "baseline")
    rows = _evaluate_configs(
        panel_records,
        definition,
        trace_set,
        configs,
        atol=atol,
        rtol=rtol,
        required_matched_ratio=required_matched_ratio,
    )
    candidate_aggs = _aggregate_configs(rows, configs, required_matched_ratio=required_matched_ratio)
    survivors = [cand for cand in candidates if cand.stage != "baseline" and candidate_aggs.get(cand.name, {}).get("status") == "safe"]
    strict_survivors = [
        cand for cand in candidates
        if cand.stage != "baseline" and candidate_aggs.get(cand.name, {}).get("worst_matched_ratio_strict", 0.0) >= required_matched_ratio
    ]
    strict_survivor_names = {cand.name for cand in strict_survivors}
    contest_only_survivors = [cand for cand in survivors if cand.name not in strict_survivor_names]
    survivors.sort(
        key=lambda cand: (
            candidate_aggs[cand.name]["worst_matched_ratio_contest"],
            -candidate_aggs[cand.name]["worst_max_rel"],
            -_mode_rank(cand.mode),
        ),
        reverse=True,
    )

    frontier_rows: list[dict] = []
    active: list[Candidate] = []
    for cand in _combined_candidates(survivors):
        trial = active + [cand]
        cfg = _make_eval_config(trial, "cumulative")
        staged_rows = _evaluate_configs(
            panel_records,
            definition,
            trace_set,
            [cfg],
            atol=atol,
            rtol=rtol,
            required_matched_ratio=required_matched_ratio,
        )
        rows.extend(staged_rows)
        agg = _aggregate_configs(staged_rows, [cfg], required_matched_ratio=required_matched_ratio)[cfg.name]
        agg["kept"] = agg["status"] == "safe"
        frontier_rows.append(agg)
        if agg["kept"]:
            active.append(cand)

    stage_summary = _build_stage_summary(candidate_aggs, candidates)
    margin_summary = _build_margin_summary(candidate_aggs, BF16_F16_STAGES)

    pairwise_shortlist = _build_pairwise_shortlist(candidate_aggs)
    pairwise_configs = _build_pairwise_configs(pairwise_shortlist)
    pairwise_rows = _evaluate_configs(
        panel_records,
        definition,
        trace_set,
        pairwise_configs,
        atol=atol,
        rtol=rtol,
        required_matched_ratio=required_matched_ratio,
    ) if pairwise_configs else []
    pairwise_aggs = _aggregate_configs(pairwise_rows, pairwise_configs, required_matched_ratio=required_matched_ratio) if pairwise_configs else {}
    pairwise_summary = list(pairwise_aggs.values())
    pairwise_summary.sort(key=lambda r: (r["worst_matched_ratio_contest"], r["worst_matched_ratio_strict"]))

    stress_names = [name for name in DEFAULT_STRESS_CANDIDATES if name in candidate_aggs]
    stress_summary = _build_stress_summary(rows, {name: candidate_aggs[name] for name in stress_names})

    promotion_summary = {
        "contest_safe_oracle_candidates": [c.name for c in survivors],
        "strict_safe_oracle_candidates": [c.name for c in strict_survivors],
        "kernel_validated_candidates": [],
        "pairwise_shortlist": [c.name for c in pairwise_shortlist],
    }

    return {
        "rows": rows,
        "candidate_aggs": candidate_aggs,
        "stage_summary": stage_summary,
        "frontier": frontier_rows,
        "margin_summary": margin_summary,
        "pairwise_summary": pairwise_summary,
        "stress_summary": stress_summary,
        "survivors": survivors,
        "strict_survivors": strict_survivors,
        "contest_only_survivors": contest_only_survivors,
        "promotion_summary": promotion_summary,
        "active_candidates": active,
    }


def _analyze_eval_configs(
    panel_records: list[tuple[int, Any]],
    definition,
    trace_set,
    configs: list[EvalConfig],
    *,
    atol: float,
    rtol: float,
    required_matched_ratio: float,
) -> dict[str, Any]:
    rows = _evaluate_configs(
        panel_records,
        definition,
        trace_set,
        configs,
        atol=atol,
        rtol=rtol,
        required_matched_ratio=required_matched_ratio,
    )
    config_aggs = _aggregate_configs(rows, configs, required_matched_ratio=required_matched_ratio)
    config_summary = list(config_aggs.values())
    config_summary.sort(
        key=lambda row: (
            row["worst_matched_ratio_contest"],
            row["worst_matched_ratio_strict"],
            -row["worst_max_rel"],
        ),
        reverse=True,
    )
    return {
        "rows": rows,
        "config_summary": config_summary,
        "config_aggs": config_aggs,
    }


def _candidate_from_stage_row(row: dict) -> Candidate | None:
    if row["status"] != "safe" or row["best_safe_mode"] == "-":
        return None
    scale_mode = row["scale_mode"] if row["scale_mode"] != "-" else "none"
    return Candidate(row["stage"], row["best_safe_mode"], scale_mode=scale_mode)


def _promoted_contest_candidates(
    stage1_analysis: dict[str, Any],
    fp8_analysis: dict[str, Any],
    *,
    promote_top_k_per_stage: int,
) -> list[Candidate]:
    promoted: dict[str, Candidate] = {}
    for row in stage1_analysis["stage_summary"]:
        cand = _candidate_from_stage_row(row)
        if cand is not None:
            promoted[cand.stage] = cand
    for cand in _pick_top_fp8_candidates(
        fp8_analysis["candidate_aggs"],
        fp8_analysis["candidates"],
        promote_top_k_per_stage,
    ):
        promoted[cand.stage] = cand
    return [promoted[stage] for stage in sorted(promoted, key=_pipeline_stage_rank)]


def _staged_probe_impl(
    seed: int,
    stage1_panel_size: int,
    contest_panel_size: int,
    atol: float,
    rtol: float,
    required_matched_ratio: float,
    promote_top_k_per_stage: int,
    enable_kernel_validation: bool,
    kernel_validation_limit: int,
) -> dict:
    import torch
    from flashinfer_bench import TraceSet

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    trace_set = TraceSet.from_path(TRACE_SET_PATH)
    definition = trace_set.definitions[DEFINITION]
    workloads = trace_set.workloads[DEFINITION]

    stage1_panel = _panel_indices(workloads, stage1_panel_size)
    contest_panel = _panel_indices(workloads, min(contest_panel_size, len(workloads)))
    stage1_records = _load_panel_records(trace_set, definition, workloads, stage1_panel)
    contest_records = _load_panel_records(trace_set, definition, workloads, contest_panel)

    stage1_candidates = _build_bf16_f16_candidates()
    stage1_analysis = _analyze_candidates(
        stage1_records,
        definition,
        trace_set,
        stage1_candidates,
        atol=atol,
        rtol=rtol,
        required_matched_ratio=required_matched_ratio,
    )

    fp8_targets = list(DEFAULT_FP8_TARGET_STAGES)
    fp8_candidates = _build_fp8_candidates(fp8_targets)
    fp8_analysis = _analyze_candidates(
        stage1_records,
        definition,
        trace_set,
        [Candidate("baseline", "fp32"), *fp8_candidates],
        atol=atol,
        rtol=rtol,
        required_matched_ratio=required_matched_ratio,
    )
    fp8_analysis["candidates"] = fp8_candidates

    contest_candidates = _promoted_contest_candidates(
        stage1_analysis,
        fp8_analysis,
        promote_top_k_per_stage=promote_top_k_per_stage,
    )
    contest_analysis = _analyze_candidates(
        contest_records,
        definition,
        trace_set,
        [Candidate("baseline", "fp32"), *contest_candidates],
        atol=atol,
        rtol=rtol,
        required_matched_ratio=required_matched_ratio,
    )
    kernel_validation_summary = (
        _run_kernel_validation(
            _build_kernel_validation_configs(
                contest_candidates,
                contest_analysis,
                limit=kernel_validation_limit,
            ),
            atol=atol,
            rtol=rtol,
            required_matched_ratio=required_matched_ratio,
        )
        if enable_kernel_validation
        else []
    )

    base_meta = {
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "definition": DEFINITION,
        "seed": seed,
        "atol": atol,
        "rtol": rtol,
        "required_matched_ratio": required_matched_ratio,
        "strict_atol": STRICT_ATOL,
        "strict_rtol": STRICT_RTOL,
        "device": torch.cuda.get_device_name(0),
        "evidence_scope": EVIDENCE_SCOPE,
        "kernel_validated_candidates": "-",
    }
    return {
        "stage1": {
            **stage1_analysis,
            "meta": {
                **base_meta,
                "run_stage": "stage1_bf16_f16",
                "panel_size": len(stage1_panel),
                "panel_indices": ",".join(str(i) for i in stage1_panel),
                "n_candidates": len(stage1_candidates),
                "n_survivors": len(stage1_analysis["survivors"]),
                "contest_safe_survivor_count": len(stage1_analysis["survivors"]),
                "contest_safe_survivors": ", ".join(c.name for c in stage1_analysis["survivors"]) or "-",
                "strict_safe_survivor_count": len(stage1_analysis["strict_survivors"]),
                "strict_safe_survivors": ", ".join(c.name for c in stage1_analysis["strict_survivors"]) or "-",
                "contest_only_survivor_count": len(stage1_analysis["contest_only_survivors"]),
                "contest_only_survivors": ", ".join(c.name for c in stage1_analysis["contest_only_survivors"]) or "-",
                "promote_top_k_per_stage": promote_top_k_per_stage,
            },
        },
        "fp8": {
            **fp8_analysis,
            "kernel_validation_summary": [],
            "meta": {
                **base_meta,
                "run_stage": "stage1_fp8_followup",
                "panel_size": len(stage1_panel),
                "panel_indices": ",".join(str(i) for i in stage1_panel),
                "fp8_target_stages": ",".join(fp8_targets) or "-",
                "n_candidates": len(fp8_candidates),
                "n_survivors": len(fp8_analysis["survivors"]),
                "contest_safe_survivor_count": len(fp8_analysis["survivors"]),
                "contest_safe_survivors": ", ".join(c.name for c in fp8_analysis["survivors"]) or "-",
                "strict_safe_survivor_count": len(fp8_analysis["strict_survivors"]),
                "strict_safe_survivors": ", ".join(c.name for c in fp8_analysis["strict_survivors"]) or "-",
                "contest_only_survivor_count": len(fp8_analysis["contest_only_survivors"]),
                "contest_only_survivors": ", ".join(c.name for c in fp8_analysis["contest_only_survivors"]) or "-",
                "promote_top_k_per_stage": promote_top_k_per_stage,
            },
        },
        "contest": {
            **contest_analysis,
            "kernel_validation_summary": kernel_validation_summary,
            "meta": {
                **base_meta,
                "run_stage": "contest_panel",
                "panel_size": len(contest_panel),
                "panel_indices": ",".join(str(i) for i in contest_panel),
                "n_candidates": len(contest_candidates),
                "n_survivors": len(contest_analysis["survivors"]),
                "contest_safe_survivor_count": len(contest_analysis["survivors"]),
                "contest_safe_survivors": ", ".join(c.name for c in contest_analysis["survivors"]) or "-",
                "strict_safe_survivor_count": len(contest_analysis["strict_survivors"]),
                "strict_safe_survivors": ", ".join(c.name for c in contest_analysis["strict_survivors"]) or "-",
                "contest_only_survivor_count": len(contest_analysis["contest_only_survivors"]),
                "contest_only_survivors": ", ".join(c.name for c in contest_analysis["contest_only_survivors"]) or "-",
                "promote_top_k_per_stage": promote_top_k_per_stage,
                "kernel_validation_enabled": str(enable_kernel_validation).lower(),
                "kernel_validation_limit": kernel_validation_limit,
            },
        },
    }


def _targeted_probe_impl(
    seed: int,
    contest_panel_size: int,
    config_specs: str,
    atol: float,
    rtol: float,
    required_matched_ratio: float,
) -> dict:
    import torch
    from flashinfer_bench import TraceSet

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    trace_set = TraceSet.from_path(TRACE_SET_PATH)
    definition = trace_set.definitions[DEFINITION]
    workloads = trace_set.workloads[DEFINITION]
    contest_panel = _panel_indices(workloads, min(contest_panel_size, len(workloads)))
    contest_records = _load_panel_records(trace_set, definition, workloads, contest_panel)

    parsed_configs = _parse_targeted_config_specs(config_specs)
    eval_configs = [_make_eval_config(candidates, "targeted", config_name) for config_name, candidates in parsed_configs]
    analysis = _analyze_eval_configs(
        contest_records,
        definition,
        trace_set,
        eval_configs,
        atol=atol,
        rtol=rtol,
        required_matched_ratio=required_matched_ratio,
    )
    return {
        **analysis,
        "meta": {
            "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "definition": DEFINITION,
            "seed": seed,
            "atol": atol,
            "rtol": rtol,
            "required_matched_ratio": required_matched_ratio,
            "strict_atol": STRICT_ATOL,
            "strict_rtol": STRICT_RTOL,
            "panel_size": len(contest_panel),
            "panel_indices": ",".join(str(i) for i in contest_panel),
            "targeted_configs": ", ".join(name for name, _ in parsed_configs),
            "evidence_scope": EVIDENCE_SCOPE,
        },
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
def probe_b200(
    seed: int,
    stage1_panel_size: int,
    contest_panel_size: int,
    atol: float,
    rtol: float,
    required_matched_ratio: float,
    promote_top_k_per_stage: int,
    enable_kernel_validation: bool,
    kernel_validation_limit: int,
) -> dict:
    return _staged_probe_impl(
        seed,
        stage1_panel_size,
        contest_panel_size,
        atol,
        rtol,
        required_matched_ratio,
        promote_top_k_per_stage,
        enable_kernel_validation,
        kernel_validation_limit,
    )


@app.function(**_COMMON, gpu="B200:1")
def probe_targeted_b200(
    seed: int,
    contest_panel_size: int,
    config_specs: str,
    atol: float,
    rtol: float,
    required_matched_ratio: float,
) -> dict:
    return _targeted_probe_impl(
        seed,
        contest_panel_size,
        config_specs,
        atol,
        rtol,
        required_matched_ratio,
    )


@app.local_entrypoint()
def probe(
    seed: int = 1234,
    stage1_panel_size: int = DEFAULT_PANEL_SIZE,
    contest_panel_size: int = 19,
    label: str = "",
    atol: float = DEFAULT_ATOL,
    rtol: float = DEFAULT_RTOL,
    required_matched_ratio: float = DEFAULT_REQUIRED_MATCHED_RATIO,
    promote_top_k_per_stage: int = DEFAULT_PROMOTE_TOP_K_PER_STAGE,
    enable_kernel_validation: bool = DEFAULT_ENABLE_KERNEL_VALIDATION,
    kernel_validation_limit: int = DEFAULT_KERNEL_VALIDATION_LIMIT,
):
    result = probe_b200.remote(
        seed,
        stage1_panel_size,
        contest_panel_size,
        atol,
        rtol,
        required_matched_ratio,
        promote_top_k_per_stage,
        enable_kernel_validation,
        kernel_validation_limit,
    )
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    suffix = f"_{label}" if label else ""
    root_dir = PROJECT_ROOT / "experiments" / f"{ts}_B200_moe_precision_staged{suffix}"
    root_dir.mkdir(parents=True, exist_ok=True)
    for key in ("stage1", "fp8", "contest"):
        run = result[key]
        out_dir = root_dir / key
        _write_outputs(
            out_dir,
            run["rows"],
            run["meta"],
            run["stage_summary"],
            run["frontier"],
            extra_json={
                "pairwise_summary.json": run["pairwise_summary"],
                "margin_summary.json": run["margin_summary"],
                "stress_summary.json": run["stress_summary"],
                "promotion_summary.json": run["promotion_summary"],
                "kernel_validation_summary.json": run.get("kernel_validation_summary", []),
            },
        )
        print(f"Wrote {out_dir / 'summary.md'}")


@app.local_entrypoint()
def probe_targeted(
    config_specs: str,
    seed: int = 1234,
    contest_panel_size: int = 19,
    label: str = "",
    atol: float = DEFAULT_ATOL,
    rtol: float = DEFAULT_RTOL,
    required_matched_ratio: float = DEFAULT_REQUIRED_MATCHED_RATIO,
):
    result = probe_targeted_b200.remote(
        seed,
        contest_panel_size,
        config_specs,
        atol,
        rtol,
        required_matched_ratio,
    )
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    suffix = f"_{label}" if label else ""
    out_dir = PROJECT_ROOT / "experiments" / f"{ts}_B200_moe_precision_targeted{suffix}"
    _write_targeted_outputs(out_dir, result["rows"], result["meta"], result["config_summary"])
    print(f"Wrote {out_dir / 'summary.md'}")
