#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import safetensors.torch as st
import torch

from flashinfer_bench import TraceSet


K_NUM_EXPERTS = 256
K_NUM_GROUPS = 8
K_GROUP_SIZE = 32
K_TOPK_GROUPS = 4
K_TOPK = 8


def bf16_to_float_tensor(t: torch.Tensor) -> torch.Tensor:
    if t.dtype == torch.bfloat16 or t.dtype == torch.float32:
        return t.float()
    if t.dtype == torch.uint16:
        return t.view(torch.bfloat16).float()
    return t.float()


def routing_topk_indices(logits: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    s = torch.sigmoid(logits.float())
    b = bf16_to_float_tensor(bias)
    score = s + b

    group_scores = []
    for g in range(K_NUM_GROUPS):
        base = g * K_GROUP_SIZE
        group_vals = score[:, base : base + K_GROUP_SIZE]
        top2 = torch.topk(group_vals, k=2, dim=-1).values
        group_scores.append((top2[:, 0] + top2[:, 1]).unsqueeze(-1))
    group_scores_t = torch.cat(group_scores, dim=-1)

    kept_groups = torch.topk(group_scores_t, k=K_TOPK_GROUPS, dim=-1).indices
    keep_mask = torch.zeros_like(group_scores_t, dtype=torch.bool)
    keep_mask.scatter_(1, kept_groups, True)

    full_keep = torch.zeros_like(score, dtype=torch.bool)
    for g in range(K_NUM_GROUPS):
        base = g * K_GROUP_SIZE
        full_keep[:, base : base + K_GROUP_SIZE] = keep_mask[:, g].unsqueeze(-1)

    masked_score = torch.where(full_keep, score, torch.full_like(score, -torch.inf))
    return torch.topk(masked_score, k=K_TOPK, dim=-1).indices


def main() -> None:
    parser = argparse.ArgumentParser(description="Export per-workload local expert t_valid CSV.")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("/home/snu_avq1/workspace/mlsys26-contest"),
        help="TraceSet dataset root.",
    )
    parser.add_argument(
        "--definition",
        default="moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048",
        help="Definition name.",
    )
    parser.add_argument("--local-expert-offset", type=int, default=32)
    parser.add_argument("--num-local-experts", type=int, default=32)
    parser.add_argument("--num-workloads", type=int, default=19)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/home/snu_avq1/workspace/chaewon/flashinfer-bench-starter-kit/tvalid_by_workload.csv"),
    )
    args = parser.parse_args()

    trace_set = TraceSet.from_path(str(args.dataset_root))
    workloads = trace_set.workloads[args.definition][: args.num_workloads]

    columns = [f"w{i}" for i in range(len(workloads))]
    counts_by_workload: list[list[int]] = []

    for wrapped in workloads:
        workload = wrapped.workload
        rl = workload.inputs["routing_logits"]
        rb = workload.inputs["routing_bias"]
        logits = st.load_file(str((args.dataset_root / rl.path).resolve()))[rl.tensor_key]
        bias = st.load_file(str((args.dataset_root / rb.path).resolve()))[rb.tensor_key]

        topk = routing_topk_indices(logits, bias)
        local_counts = [0 for _ in range(args.num_local_experts)]
        for ge in topk.reshape(-1).tolist():
            le = ge - args.local_expert_offset
            if 0 <= le < args.num_local_experts:
                local_counts[le] += 1
        counts_by_workload.append(local_counts)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["local_expert_idx", *columns])
        for le in range(args.num_local_experts):
            writer.writerow([le, *[counts_by_workload[w][le] for w in range(len(workloads))]])
        writer.writerow(["nonzero_local_experts", *[
            sum(1 for v in counts_by_workload[w] if v > 0) for w in range(len(workloads))
        ]])
        writer.writerow(["total_local_routed", *[
            sum(counts_by_workload[w]) for w in range(len(workloads))
        ]])

    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
