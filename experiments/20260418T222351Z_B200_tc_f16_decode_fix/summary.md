# Sweep 20260418T222351Z  (B200)

| metric | value |
|---|---|
| timestamp | 20260418T222351Z |
| gpu | B200 |
| max_workloads | 1 |
| warmup_runs | 1 |
| iterations | 3 |
| num_trials | 1 |
| n_variants | 8 |
| solution | moe_fp8_cpu_ref_cuda |
| definition | moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048 |

## Variants

- **default**: _(none)_
- **tc**: `FIB_MOE_TC=1`
- **tc_gemm1_only**: `FIB_MOE_TC=1`, `FIB_MOE_TC_GEMM1_ONLY=1`
- **tc_transpose_b_probe**: `FIB_MOE_TC=1`, `FIB_MOE_TC_TRANSPOSE_B=1`
- **legacy**: `FIB_MOE_LEGACY=1`
- **default_profile**: `FIB_MOE_PROFILE=1`
- **tc_profile**: `FIB_MOE_TC=1`, `FIB_MOE_PROFILE=1`
- **legacy_profile**: `FIB_MOE_LEGACY=1`, `FIB_MOE_PROFILE=1`

## Results

| variant | workload | status | latency_ms | ref_ms | speedup | abs_err | rel_err |
|---|---|---|---|---|---|---|---|
| default | b8f4f012 | PASSED | 10.755 | 12.192 | 1.13x | 5.12e+02 | 6.33e-03 |
| tc | b8f4f012 | INCORRECT_NUMERICAL | - | - | - | 5.06e+05 | 1.00e+00 |
| tc_gemm1_only | b8f4f012 | INCORRECT_NUMERICAL | - | - | - | 5.65e+05 | 1.00e+00 |
| tc_transpose_b_probe | b8f4f012 | INCORRECT_NUMERICAL | - | - | - | 5.53e+05 | 1.00e+00 |
| legacy | b8f4f012 | PASSED | 10.855 | 11.676 | 1.08x | 2.56e+02 | 1.38e-02 |
| default_profile | b8f4f012 | PASSED | 11.081 | 11.727 | 1.06x | 1.02e+03 | 7.58e-03 |
| tc_profile | b8f4f012 | INCORRECT_NUMERICAL | - | - | - | 5.57e+05 | 1.00e+00 |
| legacy_profile | b8f4f012 | PASSED | 10.939 | 11.735 | 1.07x | 2.56e+02 | 7.69e-03 |

## Speedup by variant (PASSED rows only)

| variant | n | mean speedup | min speedup | max speedup |
|---|---|---|---|---|
| default | 1 | 1.134x | 1.134x | 1.134x |
| tc | 0 | - | - | - |
| tc_gemm1_only | 0 | - | - | - |
| tc_transpose_b_probe | 0 | - | - | - |
| legacy | 1 | 1.076x | 1.076x | 1.076x |
| default_profile | 1 | 1.058x | 1.058x | 1.058x |
| tc_profile | 0 | - | - | - |
| legacy_profile | 1 | 1.073x | 1.073x | 1.073x |
