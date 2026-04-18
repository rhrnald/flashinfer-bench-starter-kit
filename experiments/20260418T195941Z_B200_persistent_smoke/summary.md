# Sweep 20260418T195941Z  (B200)

| metric | value |
|---|---|
| timestamp | 20260418T195941Z |
| gpu | B200 |
| max_workloads | 1 |
| warmup_runs | 1 |
| iterations | 3 |
| num_trials | 1 |
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
| baseline | b8f4f012 | PASSED | 10.863 | 11.949 | 1.10x | 1.02e+03 | 2.12e-02 |
| profile_only | b8f4f012 | PASSED | 10.953 | 11.930 | 1.09x | 5.12e+02 | 7.52e-03 |
| grouped | b8f4f012 | PASSED | 10.759 | 11.857 | 1.10x | 6.40e+01 | 2.62e-02 |
| grouped_profile | b8f4f012 | PASSED | 10.853 | 11.839 | 1.09x | 5.12e+02 | 1.85e-02 |

## Speedup by variant (PASSED rows only)

| variant | n | mean speedup | min speedup | max speedup |
|---|---|---|---|---|
| baseline | 1 | 1.100x | 1.100x | 1.100x |
| profile_only | 1 | 1.089x | 1.089x | 1.089x |
| grouped | 1 | 1.102x | 1.102x | 1.102x |
| grouped_profile | 1 | 1.091x | 1.091x | 1.091x |
