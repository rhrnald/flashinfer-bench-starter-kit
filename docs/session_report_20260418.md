# Session Report — 2026-04-18 MoE B200 optimization

## Scope

Contest target: `fused_moe` track, `moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048` on B200. Goal: move the submission's measured speedup on the contest's 19-workload suite without requiring CUTLASS SM100 expertise (that lane is teammate 김채원's).

## What was shipped

### 1. Grouped-GEMM routing metadata (earlier in session)

Replaced the dense `local_weight[T, 32]` mask + host-side per-expert loop with grouped-GEMM-shaped metadata:

- `expert_counts[32]`, `expert_offsets[33]`, `permuted_token_ids[N_routed]`, `permuted_weights[N_routed]`
- Built by three new routing-metadata kernels (`build_counts_kernel`, `scan_offsets_kernel`, `scatter_placements_kernel`) in `solution/cuda/moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048.cu`.
- Added `DeviceMxfpGemmModule::RunExpertPermuted` in `solution/cuda/mxfp_gemm_module.{h,cu}` — a grouped-GEMM-shaped call site. CUTLASS SM100 blockwise FP8 grouped GEMM + SwiGLU + weighted-acc scatter can swap in without changing the outer caller.

### 2. Profile-sync env gate (earlier in session)

Per-stage `cudaStreamSynchronize` calls that existed only to time `[moe_step_timing]` are now gated behind `FIB_MOE_PROFILE=1`. Production runs skip 4 unconditional device round-trips per MoE call.

### 3. Persistent workspace (this session)

Added a file-local static `MoeWorkspace` struct that owns all transient device buffers with grow-on-demand capacity: `local_weight_dev`, `expert_used_dev`, `a_dev`, `out_acc_dev`, plus the 5 grouped-metadata buffers. `DeviceMxfpGemmModule` instance is also function-local static so its `g1_dev_`/`c_dev_` workspace persists too.

Eliminates 9 `cudaMalloc`/`cudaFree` per MoE invocation. The previously-required unconditional `cudaStreamSynchronize` before `cudaFree` is now only taken when `FIB_MOE_PROFILE=1`.

### 4. Flipped default path to grouped (this session)

As of this session, **grouped is the default**. Opt-out is `FIB_MOE_LEGACY=1` → uses the dense-mask path via `DeviceMxfpGemmModule::RunExpert`. `FIB_MOE_GROUPED=1` is obsolete (no-op — already the default).

Legacy path is kept compilable so CUTLASS work on `RunExpertPermuted` has an A/B fallback.

### 5. Reproducible sweep harness

`scripts/sweep.py::sweep` — one Modal container builds once and iterates env-flag variants. Writes `experiments/<UTC_ts>_<GPU>[_<label>]/{results.csv, summary.md, raw.json}`. Default variant set updated to `default` / `legacy` / `default_profile` / `legacy_profile`.

## Measurements

All on Modal B200, 19 contest workloads, `trtllm_fp8_block_scale_moe` reference.

### Headline: persistent workspace + grouped default

| configuration | mean speedup | grouped mean |
|---|---|---|
| **Before:** baseline (pre-persistent, pre-grouped-default) | 0.514× | 0.407× (FIB_MOE_GROUPED=1, hurt by malloc churn) |
| **After:** persistent workspace, grouped default | **0.610×** | (same — is now the default) |

*Smoke config: warmup=1 iter=3 trial=1, all 19 workloads, 4 variants.*

### Deep statistics (warmup=2 iter=10 trial=2)

| variant | n | mean | min | max |
|---|---|---|---|---|
| default (grouped) | 19 | **0.576×** | 0.017× | 2.339× |
| legacy | 19 | 0.554× | 0.017× | 2.393× |
| default_profile | 19 | 0.572× | 0.017× | 2.318× |
| legacy_profile | 19 | 0.541× | 0.017× | 2.297× |

Default (grouped) wins on 13/19 workloads, ties on 3, loses by ≤5% on 3. All 76 (variant × workload) combos PASS correctness.

### Per-workload speedup (baseline vs default, iter=10 trial=2)

| workload | seq_len class | legacy | default (grouped) | Δ |
|---|---|---|---|---|
| `5e8dc11c` | ~11948 (monster) | 0.02× | 0.02× | tied (TC-bound) |
| `58a34f27` | ~14107 (monster) | 0.02× | 0.02× | tied (TC-bound) |
| `1a4c6ba1` | ~901 (monster) | 0.07× | **0.09×** | +29% |
| `8f1ff9f1` | ~mid | 0.28× | **0.36×** | +29% |
| `5eadab1e` | ~mid | 0.40× | **0.48×** | +20% |
| `eedc63b2` | ~mid | 0.42× | **0.47×** | +12% |
| `e626d3e6` | ~mid | 0.34× | **0.38×** | +12% |
| `74d7ff04` | ~mid | 0.35× | **0.39×** | +11% |
| `4822167c` | ~mid | 0.32× | **0.37×** | +16% |
| `81955b1e` | ~mid | 0.36× | **0.40×** | +11% |
| `76010cb4` | ~mid | 0.38× | **0.41×** | +8% |
| `fc378037` | ~mid | 0.36× | **0.40×** | +11% |
| `f7d6ac7c` | ~mid | 0.50× | **0.52×** | +4% |
| `a7c2bcfd` | ~small | 0.62× | **0.64×** | +3% |
| `2e69caee` | ~small | 1.45× | **1.48×** | +2% |
| `6230e838` | ~small | 0.43× | 0.42× | −2% |
| `8cba5890` | ~small | 0.70× | 0.68× | −3% |
| `b8f4f012` | ~small | 1.12× | 1.07× | −5% |
| `e05c6c03` | ~small | 2.39× | 2.34× | −2% |

## Where the bleed lives

**3 monster workloads absorb ~90% of total wall time.** Baseline latencies:

| workload | baseline ms | ref ms | gap |
|---|---|---|---|
| `5e8dc11c` | 2758 | 47 | **60× slower** |
| `58a34f27` | 1821 | 36 | 51× slower |
| `1a4c6ba1` | 299 | 22 | 14× slower |

These are arithmetic-bound on CUDA cores (FP32 dot products over 7168-element vectors, per-expert, no tensor-core utilization). Reference uses TensorRT-LLM's FP8 TC path at ~186 TFLOPS effective. Routing-side work cannot move them — **CUTLASS SM100 blockwise FP8 grouped GEMM on the `RunExpertPermuted` call site is the only lever.**

## Handoff to CUTLASS lane

`DeviceMxfpGemmModule::RunExpertPermuted` in `solution/cuda/mxfp_gemm_module.h` is the drop-in point. Signature matches what a CUTLASS grouped FP8 GEMM needs:

```cpp
void RunExpertPermuted(const float* a_dev, int64_t t, int n_rows,
                       const int* permuted_tok_e, const float* permuted_w_e,
                       int local_expert_idx, const uint8_t* gemm1_w_dev,
                       const float* gemm1_s_dev, const uint8_t* gemm2_w_dev,
                       const float* gemm2_s_dev, float* out_acc_dev,
                       cudaStream_t stream) const;
```

Inputs: contiguous expert weight slices (already per-expert), gather indices for activation rows, per-token routing weight.
Outputs: scatter-accumulate into `out_acc_dev`.

CUTLASS implementation should replace the 3-kernel sequence (`gemm1_permuted_kernel` + `swiglu_permuted_kernel` + `gemm2_scatter_accumulate_kernel`) with a CUTLASS SM100 blockwise FP8 grouped GEMM + SwiGLU epilogue + weighted-acc scatter. Reference layout: DeepSeek-V3 block-scale (kBlock=128) — already matches contest definition.

## Experiment artifacts

All persisted under `experiments/`:

| directory | config | purpose |
|---|---|---|
| `20260418T195323Z_B200_all19/` | warmup=1 iter=3 trial=1 | Pre-persistent-workspace baseline |
| `20260418T200245Z_B200_persistent_all19/` | warmup=1 iter=3 trial=1 | Persistent workspace win |
| `20260418T201848Z_B200_persistent_deep/` | warmup=2 iter=10 trial=2 | Statistically-robust all-19 |
| `20260418T202*_B200_flipped_default/` | warmup=1 iter=3 trial=1 | Default-flip verification |

Each directory has `results.csv`, `summary.md`, `raw.json`.

## Reproducing

```bash
# Full 19-workload deep sweep
modal run scripts/sweep.py::sweep --gpu B200 --max-workloads 0 \
    --warmup-runs 2 --iterations 10 --num-trials 2 --label my_label

# Smoke on a single workload (cheap discipline)
modal run scripts/sweep.py::sweep --gpu B200 --max-workloads 1 \
    --warmup-runs 1 --iterations 3 --num-trials 1
```

Custom variants: pass `--variants-json path/to/file.json` where the file is `[{"name": str, "env": {"FIB_...": "1", ...}}, ...]`.

## Files touched this session

- `solution/cuda/moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048.cu` — routing metadata kernels, persistent `MoeWorkspace`, static `DeviceMxfpGemmModule`, `FIB_MOE_LEGACY` opt-out gate, conditional final sync.
- `solution/cuda/mxfp_gemm_module.{h,cu}` — `RunExpertPermuted` + three permuted kernels (`gemm1_permuted_kernel`, `swiglu_permuted_kernel`, `gemm2_scatter_accumulate_kernel`).
- `scripts/run_modal.py` — `_run_sweep_impl`, `sweep_<gpu>` functions, env-var threading via `_apply_env`/`_restore_env`.
- `scripts/sweep.py` (new) — sweep entrypoint, output writers, default variant set.
