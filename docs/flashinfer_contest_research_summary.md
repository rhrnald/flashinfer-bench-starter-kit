# FlashInfer Contest Research Summary

Date: 2026-04-18

This note summarizes the project investigation, current repo state, external references, and recommended next steps for the FlashInfer AI Kernel Generation Contest work. It is written to be useful for teammates who want the high-level picture before touching CUDA.

## Project Goal

The goal is to score well in the FlashInfer AI Kernel Generation Contest at MLSys 2026 by submitting fast, correct GPU kernels for modern LLM operators on NVIDIA Blackwell B200.

Judging is based on:

- Correctness against FlashInfer-Bench reference outputs.
- Latency speedup versus the reference implementation.
- Final target hardware: NVIDIA B200 via Modal.
- Local iteration can happen on any CUDA GPU for non-SM100-specific code.

The repo currently targets the `fused_moe` track:

```text
moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048
```

## Current Repo State

Important files:

- `config.toml`: currently selects the fused MoE definition.
- `solution/cuda/moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048.cu`: main MoE implementation.
- `solution/cuda/mxfp_gemm_module.{cu,h}`: modular MXFP GEMM abstraction.
- `scripts/run_modal.py`: Modal runner for cloud GPU benchmarks.
- `scripts/run_local.py`, `scripts/run_local_fast.py`, `scripts/run_local_single.py`: local benchmark/debug runners.
- `config_presets/`: quick config switches for MoE, GDN, and sparse-attention definitions.

Kernel status:

| Area | Status | Notes |
| --- | --- | --- |
| MoE | GPU correctness baseline | Full GPU pipeline exists, but final performance path is not there yet. |
| MXFP GEMM module | Modular baseline | Clear swap point for future B200 tensor-core/CUTLASS/DeepGEMM work. |
| GDN decode | Optimized enough to keep | One block per `(batch, v-head, state-row)`, shared Q/K, warp reductions. |
| GDN prefill | Naive | Correctness-first recurrence, not optimized. |
| Sparse attention | Stub | Zero-fill / placeholder behavior only. |
| Triton path | Empty | CUDA is the active path. |

## Current MoE Pipeline

The MoE CUDA path is structured as:

```text
routing_kernel
  -> dequant_hidden_kernel
  -> per-expert DeviceMxfpGemmModule::RunExpert
       GEMM1 -> SwiGLU -> GEMM2 accumulate
  -> f32_to_bf16_kernel
```

The implementation currently writes:

```text
local_weight[T, 32]
expert_used[32]
```

Then the host copies `expert_used` back and loops over local experts, launching work per used expert.

This is a good correctness/debug baseline, but not a final-performance MoE design. The main issues are:

- Host synchronization and device-to-host copy in the hot path.
- Per-expert launch structure instead of grouped GEMM.
- Dense `local_weight[T, 32]` does not match the metadata layout expected by grouped MoE GEMM implementations.
- Stage timing uses `cudaStreamSynchronize`, which is useful for debugging but distorts final benchmark latency.
- Workspace allocation/free still happens inside the invocation path.

## Recommended Direction

The most important design change is not making the routing math slightly faster. It is making routing produce metadata that a grouped GEMM path can consume.

Recommended target routing output:

```text
expert_counts[32]
expert_offsets[33]
permuted_token_ids[num_routed_tokens]
permuted_weights[num_routed_tokens]
optional token/expert reverse map for combine
```

This would let the GEMM side treat each expert as a compact token range and move toward one grouped-GEMM style launch rather than a host loop over experts.

Suggested work split:

- Routing/metadata and experiment harness: good area for readable, maintainable support code.
- B200 tensor-core GEMM: best handled by the CUDA/CUTLASS specialist.
- Keep the current dense `local_weight` path as a reference until the grouped path passes correctness.

## Modal / Experiment Harness

Modal setup status from the investigation:

- Modal CLI installed and authenticated.
- `flashinfer-trace` volume created.
- `mlsys26-contest` dataset uploaded to the Modal volume.
- Local dataset path: `/home/ainta/mlsys26-contest`.

Useful shell environment:

```bash
export FIB_DATASET_PATH=/home/ainta/mlsys26-contest
export PATH="$HOME/.local/bin:$PATH"
```

Recommended Modal usage pattern:

1. Run a very small B200 smoke test first.
2. Use cheaper GPUs for non-SM100 correctness checks when possible.
3. Only run full B200 sweeps after compile/correctness is known to work.

Smoke-test goal:

```text
Modal image build
solution pack/load
CUDA compile on B200 / sm_100
one workload correctness
```

Example smoke command if `run_modal.py` supports the flags:

```bash
modal run scripts/run_modal.py -- --gpu B200 --max-workloads 1 --warmup-runs 1 --iterations 3 --num-trials 1
```

The next useful harness feature is CSV/JSON output with:

```text
definition
workload_uuid
status
latency_ms
reference_latency_ms
speedup_factor
max_abs_error
max_rel_error
gpu
warmup_runs
iterations
num_trials
```

If MoE stage timing stays enabled, parse `[moe_step_timing]` into:

```text
routing_ms
dequant_ms
expert_ms
out_ms
total_ms
```

## External References

### CUTLASS / Blackwell

NVIDIA CUTLASS Blackwell SM100 docs are directly relevant because B200 uses SM100-class tensor cores and supports `tcgen05.mma`, including block-scaled narrow-precision MMA.

Reference:

- https://docs.nvidia.com/cutlass/latest/media/docs/cpp/blackwell_functionality.html

Why it matters:

- CUTLASS has Blackwell block-scaled GEMM examples.
- It supports MXF8/MXF6/MXF4-style narrow precision paths.
- It is a better first implementation target than hand-writing `tcgen05` from scratch.

Caveat:

- The current dataset exposes FP32 scale tensors. Some SM100 block-scaled paths expect specific packed scale-factor formats/layouts, so a conversion or fused preprocessing path may be needed.

### DeepGEMM

DeepGEMM is useful as a reference for fast FP8/MoE GEMM structure.

Reference:

- https://github.com/deepseek-ai/DeepGEMM

Why it matters:

- Provides FP8 GEMM and grouped MoE GEMM ideas.
- Supports SM90/SM100 in recent versions.
- Documents contiguous and masked grouped GEMM layouts that map conceptually to MoE routing outputs.

Caveat:

- It is not a drop-in replacement for the FlashInfer-Bench solution interface.
- Scale layout and input packing requirements may differ from this contest dataset.

### GPU MODE

GPU MODE is useful in two different ways.

Learning resources:

- https://github.com/gpu-mode/lectures
- https://github.com/gpu-mode/resource-stream

Most relevant lecture topics:

- Profiling CUDA kernels.
- CUDA performance checklist.
- Compute and memory architecture.
- CUTLASS.
- Tensor cores.
- Fused kernels.
- FlashInfer.
- CuTE / CuTeDSL.

Competition/reference data:

- https://github.com/gpu-mode/reference-kernels
- https://huggingface.co/datasets/GPUMODE/kernelbot-data

Why the KernelBot dataset matters:

- It contains real GPU kernel competition submissions.
- Relevant problem names include `fp8-gemm`, `moe`, `mxfp4-mm`, `moe-mxfp4`, `mla-decode`, and `mixed-mla`.
- These are useful for studying routing, grouped expert layouts, low-precision GEMM patterns, and submission organization.

Caveat:

- Many submissions target AMD hardware and may use HIP/Triton patterns that do not translate directly to CUDA/B200.
- Use as design inspiration, not copy-paste source.
- Check license and contest rules before reusing code.

## Highest-Value Next Steps

1. Confirm the current MoE baseline compiles and passes one workload on B200.
2. Add or verify safe Modal flags: `--gpu`, `--max-workloads`, `--warmup-runs`, `--iterations`, `--num-trials`.
3. Add CSV/JSON result output for repeatable sweeps.
4. Gate expensive MoE timing synchronizations behind an env var such as `FIB_MOE_PROFILE=1`.
5. Keep current dense-routing path as a reference.
6. Add an experimental routing metadata path for grouped GEMM:

   ```text
   expert_counts
   expert_offsets
   permuted_token_ids
   permuted_weights
   ```

7. Let the GEMM work start from CUTLASS SM100 block-scaled examples or DeepGEMM-style grouped GEMM ideas.
8. Only hand-write `tcgen05` where library paths cannot express the needed fusion/layout.

## Suggested Team Alignment Questions

Before large B200 spending, align on:

- Should routing output remain `local_weight[T,32]`, or move to compact grouped-GEMM metadata?
- Which grouped GEMM backend is the first target: CUTLASS, DeepGEMM-style code, or custom CUDA?
- What exact workload set and tolerance were used to claim FP8 validation passes?
- Should profiling synchronizations be disabled in benchmark mode?
- Who owns routing metadata, GEMM backend, and benchmark harness?

Recommended split:

- Routing metadata and harness: readable support code.
- B200 tensor-core GEMM: CUDA/CUTLASS specialist work.
- Current baseline: keep as a correctness reference until the new path is validated.
