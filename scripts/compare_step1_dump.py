"""
Compare dumped direct Step1 compact output against a PyTorch oracle.

Usage:
    FIB_DATASET_PATH=/path/to/mlsys26-contest \
    /home/snu_avq1/miniconda3/envs/fi-bench/bin/python scripts/compare_step1_dump.py \
      --dump-dir /tmp/fib_step1_dump \
      --workload-index 0
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from flashinfer_bench import TraceSet
from flashinfer_bench.bench.utils import gen_inputs, load_safetensors


DEFINITION = "moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048"
BLOCK = 128
HIDDEN = 7168
INTERMEDIATE = 2048
E_LOCAL = 32


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare Step1 dump against oracle")
    p.add_argument("--dump-dir", type=Path, required=True)
    p.add_argument("--workload-index", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda:0")
    return p.parse_args()


def get_trace_set_path() -> str:
    path = os.environ.get("FIB_DATASET_PATH")
    if not path:
        raise EnvironmentError("FIB_DATASET_PATH environment variable not set")
    return path


def load_inputs(workload_index: int, device: str) -> dict[str, torch.Tensor | int | float]:
    trace_set = TraceSet.from_path(get_trace_set_path())
    definition = trace_set.definitions[DEFINITION]
    trace = trace_set.workloads[DEFINITION][workload_index]
    safe = load_safetensors(definition, trace.workload, trace_set.root)
    values = gen_inputs(definition, trace.workload, device=device, safe_tensors=safe)
    return dict(zip(definition.inputs.keys(), values, strict=True))


def repeat_block_scales(scale: torch.Tensor) -> torch.Tensor:
    return scale.repeat_interleave(BLOCK, dim=0).repeat_interleave(BLOCK, dim=1)


def compute_oracle_step1(inputs: dict, token_ids: np.ndarray, expert_offsets: np.ndarray, device: str) -> torch.Tensor:
    hidden = inputs["hidden_states"].to(device)
    hidden_scale = inputs["hidden_states_scale"].to(torch.float32).to(device)
    gemm1_weights = inputs["gemm1_weights"].to(device)
    gemm1_weights_scale = inputs["gemm1_weights_scale"].to(torch.float32).to(device)

    hidden_f32 = hidden.to(torch.float32) * hidden_scale.t().contiguous().repeat_interleave(BLOCK, dim=1)
    out = torch.empty((int(token_ids.shape[0]), INTERMEDIATE), dtype=torch.float32, device=device)

    token_ids_t = torch.from_numpy(token_ids).to(device=device, dtype=torch.int64)
    expert_offsets_t = torch.from_numpy(expert_offsets).to(device=device, dtype=torch.int64)

    for expert in range(E_LOCAL):
        start = int(expert_offsets_t[expert].item())
        end = int(expert_offsets_t[expert + 1].item())
        if start >= end:
            continue
        tok = token_ids_t[start:end]
        a_e = hidden_f32.index_select(0, tok)
        w13_e = gemm1_weights[expert].to(torch.float32) * repeat_block_scales(gemm1_weights_scale[expert])
        g1 = a_e.matmul(w13_e.t())
        gate = g1[:, :INTERMEDIATE]
        up = g1[:, INTERMEDIATE:]
        out[start:end] = torch.nn.functional.silu(gate) * up
    return out


def main() -> None:
    args = parse_args()
    dump_dir = args.dump_dir.resolve()

    token_ids = np.fromfile(dump_dir / "permuted_token_ids.bin", dtype=np.int32)
    expert_offsets = np.fromfile(dump_dir / "expert_offsets.bin", dtype=np.int32)
    c_perm = np.fromfile(dump_dir / "c_perm.bin", dtype=np.float32).reshape(-1, INTERMEDIATE)

    inputs = load_inputs(args.workload_index, args.device)
    oracle = compute_oracle_step1(inputs, token_ids, expert_offsets, args.device).cpu().numpy()

    diff = np.abs(c_perm - oracle)
    rel = diff / (np.abs(oracle) + 1e-8)
    print(f"rows={c_perm.shape[0]} intermediate={c_perm.shape[1]}")
    print(f"max_abs={diff.max():.6e}")
    print(f"max_rel={rel.max():.6e}")
    print(f"mean_abs={diff.mean():.6e}")


if __name__ == "__main__":
    main()
