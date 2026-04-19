# Sweep 20260418T195323Z  (B200)

| metric | value |
|---|---|
| timestamp | 20260418T195323Z |
| gpu | B200 |
| max_workloads | all |
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
| baseline | b8f4f012 | PASSED | 11.600 | 11.873 | 1.02x | 2.56e+02 | 1.16e-02 |
| baseline | e05c6c03 | PASSED | 5.205 | 10.973 | 2.11x | 2.56e+02 | 6.06e-03 |
| baseline | 6230e838 | PASSED | 35.767 | 14.351 | 0.40x | 1.02e+03 | 3.91e-01 |
| baseline | 8f1ff9f1 | PASSED | 61.519 | 16.146 | 0.26x | 1.02e+03 | 3.65e-01 |
| baseline | 1a4c6ba1 | PASSED | 301.589 | 20.890 | 0.07x | 4.10e+03 | 7.50e+00 |
| baseline | a7c2bcfd | PASSED | 21.102 | 12.505 | 0.59x | 2.56e+02 | 1.79e-02 |
| baseline | 2e69caee | PASSED | 8.384 | 11.502 | 1.37x | 1.02e+03 | 7.75e-03 |
| baseline | 8cba5890 | PASSED | 19.592 | 12.271 | 0.63x | 1.02e+03 | 1.69e-02 |
| baseline | 5e8dc11c | PASSED | 2764.769 | 45.936 | 0.02x | 4.10e+03 | 3.24e+07 |
| baseline | 58a34f27 | PASSED | 1828.473 | 35.881 | 0.02x | 2.05e+03 | 1.37e+07 |
| baseline | 5eadab1e | PASSED | 35.364 | 13.830 | 0.39x | 2.05e+03 | 1.42e-01 |
| baseline | eedc63b2 | PASSED | 33.668 | 13.579 | 0.40x | 5.12e+02 | 3.10e-02 |
| baseline | e626d3e6 | PASSED | 46.877 | 15.531 | 0.33x | 1.02e+03 | 4.43e-02 |
| baseline | 74d7ff04 | PASSED | 44.274 | 15.042 | 0.34x | 5.12e+02 | 1.64e-01 |
| baseline | 4822167c | PASSED | 48.301 | 14.951 | 0.31x | 1.02e+03 | 1.13e-01 |
| baseline | 81955b1e | PASSED | 41.952 | 14.494 | 0.35x | 5.12e+02 | 1.89e-01 |
| baseline | 76010cb4 | PASSED | 38.710 | 14.331 | 0.37x | 1.02e+03 | 2.79e-01 |
| baseline | fc378037 | PASSED | 46.834 | 14.704 | 0.31x | 1.02e+03 | 1.14e-01 |
| baseline | f7d6ac7c | PASSED | 28.018 | 13.251 | 0.47x | 5.12e+02 | 6.72e-02 |
| profile_only | b8f4f012 | PASSED | 12.455 | 11.515 | 0.92x | 2.56e+02 | 7.35e-03 |
| profile_only | e05c6c03 | PASSED | 16.962 | 10.952 | 0.65x | 1.28e+02 | 7.46e-03 |
| profile_only | 6230e838 | PASSED | 35.918 | 13.774 | 0.38x | 1.02e+03 | 4.78e-02 |
| profile_only | 8f1ff9f1 | PASSED | 59.862 | 16.153 | 0.27x | 1.02e+03 | 2.31e-01 |
| profile_only | 1a4c6ba1 | PASSED | 306.493 | 20.960 | 0.07x | 2.05e+03 | 2.40e+00 |
| profile_only | a7c2bcfd | PASSED | 25.357 | 12.533 | 0.49x | 5.12e+02 | 2.45e-02 |
| profile_only | 2e69caee | PASSED | 9.879 | 11.455 | 1.16x | 5.12e+02 | 1.23e-02 |
| profile_only | 8cba5890 | PASSED | 19.461 | 12.302 | 0.63x | 2.56e+02 | 1.01e-01 |
| profile_only | 5e8dc11c | PASSED | 2762.868 | 45.646 | 0.02x | 2.05e+03 | 2.11e+02 |
| profile_only | 58a34f27 | PASSED | 1824.258 | 35.848 | 0.02x | 4.10e+03 | 2.09e+02 |
| profile_only | 5eadab1e | PASSED | 35.596 | 13.902 | 0.39x | 1.02e+03 | 1.24e+01 |
| profile_only | eedc63b2 | PASSED | 34.043 | 13.614 | 0.40x | 1.02e+03 | 4.23e+00 |
| profile_only | e626d3e6 | PASSED | 52.925 | 15.635 | 0.30x | 1.02e+03 | 2.14e-01 |
| profile_only | 74d7ff04 | PASSED | 48.521 | 15.043 | 0.31x | 1.02e+03 | 2.29e-01 |
| profile_only | 4822167c | PASSED | 48.313 | 15.032 | 0.31x | 1.02e+03 | 2.19e-01 |
| profile_only | 81955b1e | PASSED | 42.187 | 14.542 | 0.34x | 1.02e+03 | 5.26e-02 |
| profile_only | 76010cb4 | PASSED | 38.645 | 14.285 | 0.37x | 1.02e+03 | 1.44e-01 |
| profile_only | fc378037 | PASSED | 42.030 | 14.585 | 0.35x | 2.05e+03 | 1.38e-01 |
| profile_only | f7d6ac7c | PASSED | 28.064 | 13.191 | 0.47x | 1.02e+03 | 5.71e-02 |
| grouped | b8f4f012 | PASSED | 11.568 | 11.540 | 1.00x | 2.56e+02 | 1.44e-01 |
| grouped | e05c6c03 | PASSED | 66.115 | 10.960 | 0.17x | 2.56e+02 | 7.35e-03 |
| grouped | 6230e838 | PASSED | 54.114 | 13.858 | 0.26x | 1.02e+03 | 3.78e-02 |
| grouped | 8f1ff9f1 | PASSED | 49.217 | 16.094 | 0.33x | 1.02e+03 | 5.68e-01 |
| grouped | 1a4c6ba1 | PASSED | 249.029 | 20.920 | 0.08x | 2.05e+03 | 8.36e+00 |
| grouped | a7c2bcfd | PASSED | 27.796 | 12.592 | 0.45x | 5.12e+02 | 2.93e-02 |
| grouped | 2e69caee | PASSED | 10.073 | 11.490 | 1.14x | 1.28e+02 | 1.10e-02 |
| grouped | 8cba5890 | PASSED | 19.218 | 12.312 | 0.64x | 2.56e+02 | 3.55e-02 |
| grouped | 5e8dc11c | PASSED | 2734.355 | 45.691 | 0.02x | 2.05e+03 | 6.80e+01 |
| grouped | 58a34f27 | PASSED | 1787.412 | 35.896 | 0.02x | 4.10e+03 | 3.01e+01 |
| grouped | 5eadab1e | PASSED | 30.004 | 13.846 | 0.46x | 1.02e+03 | 1.13e-01 |
| grouped | eedc63b2 | PASSED | 31.356 | 13.643 | 0.44x | 1.02e+03 | 2.86e-01 |
| grouped | e626d3e6 | PASSED | 42.234 | 15.509 | 0.37x | 1.02e+03 | 3.56e+00 |
| grouped | 74d7ff04 | PASSED | 39.870 | 14.909 | 0.37x | 1.02e+03 | 7.45e-02 |
| grouped | 4822167c | PASSED | 41.992 | 15.040 | 0.36x | 5.12e+02 | 2.05e-01 |
| grouped | 81955b1e | PASSED | 38.205 | 14.745 | 0.39x | 1.02e+03 | 3.73e-01 |
| grouped | 76010cb4 | PASSED | 36.371 | 14.248 | 0.39x | 5.12e+02 | 1.33e-01 |
| grouped | fc378037 | PASSED | 38.027 | 14.680 | 0.39x | 1.02e+03 | 3.87e-02 |
| grouped | f7d6ac7c | PASSED | 27.840 | 13.190 | 0.47x | 1.02e+03 | 2.74e-01 |
| grouped_profile | b8f4f012 | PASSED | 13.458 | 11.535 | 0.86x | 5.12e+02 | 7.14e-03 |
| grouped_profile | e05c6c03 | PASSED | 7.281 | 10.941 | 1.50x | 2.56e+02 | 1.12e-02 |
| grouped_profile | 6230e838 | PASSED | 34.752 | 13.856 | 0.40x | 5.12e+02 | 1.43e-01 |
| grouped_profile | 8f1ff9f1 | PASSED | 47.040 | 16.102 | 0.34x | 1.02e+03 | 1.63e-01 |
| grouped_profile | 1a4c6ba1 | PASSED | 248.953 | 20.897 | 0.08x | 2.05e+03 | 2.71e+00 |
| grouped_profile | a7c2bcfd | PASSED | 20.881 | 12.742 | 0.61x | 5.12e+02 | 4.02e-02 |
| grouped_profile | 2e69caee | PASSED | 8.360 | 11.413 | 1.37x | 5.12e+02 | 7.69e-03 |
| grouped_profile | 8cba5890 | PASSED | 19.269 | 12.310 | 0.64x | 5.12e+02 | 5.73e-02 |
| grouped_profile | 5e8dc11c | PASSED | 2734.250 | 45.565 | 0.02x | 2.05e+03 | 3.43e+01 |
| grouped_profile | 58a34f27 | PASSED | 1793.776 | 35.916 | 0.02x | 2.05e+03 | 9.55e+01 |
| grouped_profile | 5eadab1e | PASSED | 30.212 | 13.828 | 0.46x | 5.12e+02 | 2.77e-01 |
| grouped_profile | eedc63b2 | PASSED | 42.753 | 13.616 | 0.32x | 1.02e+03 | 3.03e-02 |
| grouped_profile | e626d3e6 | PASSED | 43.196 | 15.552 | 0.36x | 1.02e+03 | 1.02e+00 |
| grouped_profile | 74d7ff04 | PASSED | 41.007 | 14.941 | 0.36x | 1.02e+03 | 1.20e+01 |
| grouped_profile | 4822167c | PASSED | 42.870 | 14.940 | 0.35x | 1.02e+03 | 7.09e-01 |
| grouped_profile | 81955b1e | PASSED | 48.162 | 14.610 | 0.30x | 1.02e+03 | 1.11e-01 |
| grouped_profile | 76010cb4 | PASSED | 36.656 | 14.461 | 0.39x | 1.02e+03 | 2.77e+00 |
| grouped_profile | fc378037 | PASSED | 39.027 | 14.687 | 0.38x | 1.02e+03 | 5.52e-01 |
| grouped_profile | f7d6ac7c | PASSED | 28.236 | 13.328 | 0.47x | 2.56e+02 | 2.79e-02 |

## Speedup by variant (PASSED rows only)

| variant | n | mean speedup | min speedup | max speedup |
|---|---|---|---|---|
| baseline | 19 | 0.514x | 0.017x | 2.108x |
| profile_only | 19 | 0.413x | 0.017x | 1.160x |
| grouped | 19 | 0.407x | 0.017x | 1.141x |
| grouped_profile | 19 | 0.486x | 0.017x | 1.503x |
