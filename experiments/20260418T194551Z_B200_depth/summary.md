# Sweep 20260418T194551Z  (B200)

| metric | value |
|---|---|
| timestamp | 20260418T194551Z |
| gpu | B200 |
| max_workloads | 3 |
| warmup_runs | 3 |
| iterations | 20 |
| num_trials | 3 |
| n_variants | 4 |
| solution | moe_fp8_cpu_ref_cuda |
| definition | moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048 |

## Variants

- **baseline**: _(none)_
- **profile_only**: `FIB_MOE_PROFILE=1`
- **grouped**: `FIB_MOE_GROUPED=1`
- **grouped_profile**: `FIB_MOE_GROUPED=1`, `FIB_MOE_PROFILE=1`

## Results

| variant | workload | status | latency_ms | ref_ms | speedup | abs_err | rel_err |
|---|---|---|---|---|---|---|---|
| baseline | b8f4f012 | PASSED | 14.492 | 11.483 | 0.79x | 5.12e+02 | 3.99e-01 |
| baseline | e05c6c03 | PASSED | 6.231 | 10.859 | 1.74x | 5.12e+02 | 1.48e-02 |
| baseline | 6230e838 | PASSED | 37.539 | 13.723 | 0.37x | 1.02e+03 | 1.14e-01 |
| profile_only | b8f4f012 | PASSED | 13.148 | 11.358 | 0.86x | 2.56e+02 | 4.44e-01 |
| profile_only | e05c6c03 | PASSED | 7.447 | 10.810 | 1.45x | 1.02e+03 | 3.11e-02 |
| profile_only | 6230e838 | PASSED | 43.240 | 13.601 | 0.31x | 1.02e+03 | 4.39e-01 |
| grouped | b8f4f012 | PASSED | 13.280 | 11.325 | 0.85x | 2.56e+02 | 3.09e-02 |
| grouped | e05c6c03 | PASSED | 6.281 | 10.870 | 1.73x | 1.28e+02 | 2.23e-02 |
| grouped | 6230e838 | PASSED | 36.759 | 13.591 | 0.37x | 1.02e+03 | 6.44e+00 |
| grouped_profile | b8f4f012 | PASSED | 13.129 | 11.380 | 0.87x | 1.02e+03 | 3.14e-02 |
| grouped_profile | e05c6c03 | PASSED | 7.620 | 10.824 | 1.42x | 1.02e+03 | 1.65e-02 |
| grouped_profile | 6230e838 | PASSED | 36.160 | 13.686 | 0.38x | 1.02e+03 | 2.58e-02 |

## Speedup by variant (PASSED rows only)

| variant | n | mean speedup | min speedup | max speedup |
|---|---|---|---|---|
| baseline | 3 | 0.967x | 0.366x | 1.743x |
| profile_only | 3 | 0.877x | 0.315x | 1.452x |
| grouped | 3 | 0.984x | 0.370x | 1.731x |
| grouped_profile | 3 | 0.889x | 0.378x | 1.420x |
