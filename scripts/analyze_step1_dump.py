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
    p = argparse.ArgumentParser(description="Analyze dumped Step1 output against oracle")
    p.add_argument("--dump-dir", type=Path, required=True)
    p.add_argument("--workload-index", type=int, required=True)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--topk", type=int, default=12)
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


def compute_oracle(inputs: dict, token_ids: np.ndarray, expert_offsets: np.ndarray, device: str, mode: int) -> torch.Tensor:
    hidden = inputs["hidden_states"].to(device)
    hidden_scale = inputs["hidden_states_scale"].to(torch.float32).to(device)
    gemm1_weights = inputs["gemm1_weights"].to(device)
    gemm1_weights_scale = inputs["gemm1_weights_scale"].to(torch.float32).to(device)

    hidden_unscaled = hidden.to(torch.float32)
    hidden_f32 = hidden_unscaled * hidden_scale.t().contiguous().repeat_interleave(BLOCK, dim=1)
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
        w13_unscaled = gemm1_weights[expert].to(torch.float32)
        w13_scaled = w13_unscaled * repeat_block_scales(gemm1_weights_scale[expert])
        if mode in (3, 4):
            a_e = hidden_unscaled.index_select(0, tok)
            g1 = a_e.matmul(w13_unscaled.t())
        else:
            g1 = a_e.matmul(w13_scaled.t())
        gate = g1[:, :INTERMEDIATE]
        up = g1[:, INTERMEDIATE:]
        if mode in (1, 3):
            out[start:end] = gate
        elif mode in (2, 4):
            out[start:end] = up
        else:
            out[start:end] = gate * torch.nn.functional.silu(up)
    return out


def read_debug_mode(dump_dir: Path) -> int:
    meta = (dump_dir / "meta.txt").read_text()
    for line in meta.splitlines():
        if line.startswith("debug_output_mode="):
            return int(line.split("=", 1)[1])
    return 0


def main() -> None:
    args = parse_args()
    dump_dir = args.dump_dir.resolve()
    mode = read_debug_mode(dump_dir)

    token_ids = np.fromfile(dump_dir / "permuted_token_ids.bin", dtype=np.int32)
    expert_offsets = np.fromfile(dump_dir / "expert_offsets.bin", dtype=np.int32)
    c_perm = np.fromfile(dump_dir / "c_perm.bin", dtype=np.float32).reshape(-1, INTERMEDIATE)

    inputs = load_inputs(args.workload_index, args.device)
    oracle = compute_oracle(inputs, token_ids, expert_offsets, args.device, mode).cpu().numpy()

    diff = np.abs(c_perm - oracle)
    rel = diff / (np.abs(oracle) + 1e-8)
    mode_name = {0: "swiglu", 1: "gate", 2: "up", 3: "gate_unscaled", 4: "up_unscaled"}.get(mode, f"unknown({mode})")
    print(f"mode={mode_name} rows={c_perm.shape[0]} intermediate={c_perm.shape[1]}")
    print(f"max_abs={diff.max():.6e}")
    print(f"max_rel={rel.max():.6e}")
    print(f"mean_abs={diff.mean():.6e}")

    topk = args.topk
    expert_idx = np.searchsorted(expert_offsets, np.arange(c_perm.shape[0]), side="right") - 1
    for row in range(c_perm.shape[0]):
        row_diff = diff[row]
        worst = np.argsort(-row_diff)[:topk]
        corr = np.corrcoef(c_perm[row], oracle[row])[0, 1] if np.std(c_perm[row]) and np.std(oracle[row]) else 0.0
        print(f"\nrow={row} expert={int(expert_idx[row])} row_max_abs={row_diff.max():.6e} corr={corr:.6f}")
        for j in worst.tolist():
            print(
                f"  col={j} got={c_perm[row,j]:.6e} ref={oracle[row,j]:.6e} diff={c_perm[row,j]-oracle[row,j]:.6e}"
            )


if __name__ == "__main__":
    main()
