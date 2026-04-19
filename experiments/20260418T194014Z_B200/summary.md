# Sweep 20260418T194014Z  (B200)

| metric | value |
|---|---|
| timestamp | 20260418T194014Z |
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
| baseline | b8f4f012 | PASSED | 11.766 | 12.861 | 1.09x | 2.56e+02 | 6.76e-03 |
| profile_only | b8f4f012 | PASSED | 11.641 | 13.288 | 1.14x | 2.56e+02 | 1.19e-02 |
| grouped | b8f4f012 | PASSED | 11.798 | 11.817 | 1.00x | 3.20e+01 | 7.35e-03 |
| grouped_profile | b8f4f012 | PASSED | 14.796 | 11.847 | 0.80x | 2.56e+02 | 9.43e-03 |

## Speedup by variant (PASSED rows only)

| variant | n | mean speedup | min speedup | max speedup |
|---|---|---|---|---|
| baseline | 1 | 1.093x | 1.093x | 1.093x |
| profile_only | 1 | 1.141x | 1.141x | 1.141x |
| grouped | 1 | 1.002x | 1.002x | 1.002x |
| grouped_profile | 1 | 0.801x | 0.801x | 0.801x |
