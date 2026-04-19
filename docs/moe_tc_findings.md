# MoE TC Findings

Date: 2026-04-18

## Result

The current FlashInfer `group_gemm_fp8_nt_groupwise` wrapper is useful for
probing the B200 FP8 contract, but it is not enough as the final MoE path when
GEMM1 is materialized to BF16/FP16 before SwiGLU.

Best next direction:

1. Keep the current grouped-routing/default CUDA path as the passing baseline.
2. Do not spend more time trying to make `GEMM1 -> BF16/FP16 tensor -> SwiGLU`
   pass full MoE tolerance.
3. Implement a custom CUTLASS/SM100 path that fuses GEMM1 accumulator handling
   with SwiGLU and FP8 quantization, so the post-GEMM nonlinearity sees higher
   precision than the public FlashInfer wrapper exposes.

## Evidence

GEMM1 contract probe:

- `experiments/20260418T220755Z_B200_gemm1_contract_probe_selfcontained/`
- `experiments/20260418T222216Z_B200_gemm1_contract_probe_rows1/`

Findings:

- B physical storage should stay as the contest `[N,K]` row-major tensor for
  this wrapper path.
- Both `MN` and `K` scale modes can match GEMM1 in the Python FlashInfer API
  when the corresponding scale layout is supplied.
- Tiny expert batches are not the blocker: `rows=1`, padded to 4, still matched
  GEMM1 with `matched_ratio=1.0`.

Precision sensitivity probe:

- `experiments/20260418T222541Z_B200_moe_precision_probe_first/`

On workload `b8f4f012` (`seq_len=7`), a PyTorch oracle shows:

| variant | matched_ratio | max_abs | max_rel |
|---|---:|---:|---:|
| fp32 GEMM1 | 1.000000 | 0 | 0 |
| BF16 GEMM1 before SwiGLU | 0.958187 | 2048 | 8.4689 |
| FP16 GEMM1 before SwiGLU | 0.994938 | 2048 | 2.1754 |

Since the benchmark requires full matched ratio under `rtol=1e-2`, even FP16
GEMM1 materialization is not sufficient.

Final smoke after the FP16/MN TC experiment:

- `experiments/20260418T222351Z_B200_tc_f16_decode_fix/`

The default path still passes, but all `FIB_MOE_TC=1` variants are
`INCORRECT_NUMERICAL`. This is expected from the precision probe above.

## Practical Next Step

The next implementation should stop using the wrapper as the production path.
Use it only as an oracle for layout. The production kernel needs one of:

- CUTLASS SM100 grouped GEMM with a custom epilogue/visitor that computes
  `x1 * silu(x2)` before writing the intermediate, ideally quantizing the SwiGLU
  output to FP8 for GEMM2.
- A dedicated two-stage kernel where GEMM1 fragments/accumulators are consumed
  before any BF16/FP16 materialization. This is harder, but it targets the actual
  numerical blocker.
