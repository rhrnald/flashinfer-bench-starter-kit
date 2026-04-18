# Sweep 20260418T200245Z  (B200)

| metric | value |
|---|---|
| timestamp | 20260418T200245Z |
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
| baseline | b8f4f012 | PASSED | 10.825 | 12.203 | 1.13x | 6.40e+01 | 1.96e-02 |
| baseline | e05c6c03 | PASSED | 4.695 | 11.034 | 2.35x | 2.00e+00 | 4.85e-03 |
| baseline | 6230e838 | PASSED | 34.636 | 14.148 | 0.41x | 2.56e+02 | 5.48e-02 |
| baseline | 8f1ff9f1 | PASSED | 58.024 | 15.922 | 0.27x | 1.02e+03 | 1.46e-01 |
| baseline | 1a4c6ba1 | PASSED | 298.725 | 20.824 | 0.07x | 2.05e+03 | 1.42e+01 |
| baseline | a7c2bcfd | PASSED | 20.219 | 12.490 | 0.62x | 5.12e+02 | 7.24e-02 |
| baseline | 2e69caee | PASSED | 7.903 | 11.443 | 1.45x | 5.12e+02 | 8.40e-02 |
| baseline | 8cba5890 | PASSED | 18.491 | 12.203 | 0.66x | 1.02e+03 | 4.55e-02 |
| baseline | 5e8dc11c | PASSED | 2754.118 | 45.222 | 0.02x | 2.05e+03 | 4.16e+07 |
| baseline | 58a34f27 | PASSED | 1817.554 | 35.747 | 0.02x | 4.10e+03 | 2.65e+01 |
| baseline | 5eadab1e | PASSED | 34.092 | 13.698 | 0.40x | 1.02e+03 | 3.11e-02 |
| baseline | eedc63b2 | PASSED | 32.164 | 13.538 | 0.42x | 5.12e+02 | 9.86e-02 |
| baseline | e626d3e6 | PASSED | 45.539 | 15.456 | 0.34x | 1.02e+03 | 4.06e-01 |
| baseline | 74d7ff04 | PASSED | 42.678 | 14.788 | 0.35x | 1.02e+03 | 2.42e-01 |
| baseline | 4822167c | PASSED | 47.174 | 14.913 | 0.32x | 1.02e+03 | 8.28e-02 |
| baseline | 81955b1e | PASSED | 40.615 | 14.463 | 0.36x | 5.12e+02 | 1.58e-01 |
| baseline | 76010cb4 | PASSED | 37.298 | 14.168 | 0.38x | 5.12e+02 | 5.74e-01 |
| baseline | fc378037 | PASSED | 40.754 | 14.531 | 0.36x | 2.05e+03 | 3.25e-02 |
| baseline | f7d6ac7c | PASSED | 26.656 | 13.236 | 0.50x | 1.02e+03 | 1.39e-02 |
| profile_only | b8f4f012 | PASSED | 10.959 | 11.600 | 1.06x | 1.28e+02 | 1.06e-01 |
| profile_only | e05c6c03 | PASSED | 4.887 | 10.933 | 2.24x | 3.20e+01 | 5.95e-03 |
| profile_only | 6230e838 | PASSED | 35.242 | 13.833 | 0.39x | 1.02e+03 | 2.78e-02 |
| profile_only | 8f1ff9f1 | PASSED | 58.780 | 16.047 | 0.27x | 1.02e+03 | 1.74e-01 |
| profile_only | 1a4c6ba1 | PASSED | 301.674 | 20.952 | 0.07x | 2.05e+03 | 4.26e+07 |
| profile_only | a7c2bcfd | PASSED | 20.330 | 12.492 | 0.61x | 5.12e+02 | 1.20e-02 |
| profile_only | 2e69caee | PASSED | 8.793 | 11.456 | 1.30x | 5.12e+02 | 7.75e-03 |
| profile_only | 8cba5890 | PASSED | 18.762 | 12.243 | 0.65x | 2.56e+02 | 6.11e-02 |
| profile_only | 5e8dc11c | PASSED | 2755.081 | 45.347 | 0.02x | 2.05e+03 | 9.30e+01 |
| profile_only | 58a34f27 | PASSED | 1817.123 | 35.838 | 0.02x | 4.10e+03 | 4.84e+07 |
| profile_only | 5eadab1e | PASSED | 34.423 | 13.678 | 0.40x | 5.12e+02 | 8.59e-02 |
| profile_only | eedc63b2 | PASSED | 32.442 | 13.402 | 0.41x | 1.02e+03 | 1.94e+00 |
| profile_only | e626d3e6 | PASSED | 45.811 | 16.086 | 0.35x | 1.02e+03 | 9.46e-01 |
| profile_only | 74d7ff04 | PASSED | 42.723 | 14.833 | 0.35x | 2.05e+03 | 6.57e-02 |
| profile_only | 4822167c | PASSED | 47.274 | 15.109 | 0.32x | 1.02e+03 | 2.36e-01 |
| profile_only | 81955b1e | PASSED | 40.657 | 14.593 | 0.36x | 1.02e+03 | 1.14e-01 |
| profile_only | 76010cb4 | PASSED | 37.627 | 14.166 | 0.38x | 1.02e+03 | 6.07e-02 |
| profile_only | fc378037 | PASSED | 40.968 | 14.534 | 0.35x | 5.12e+02 | 7.43e-02 |
| profile_only | f7d6ac7c | PASSED | 26.727 | 13.061 | 0.49x | 1.02e+03 | 2.17e-01 |
| grouped | b8f4f012 | PASSED | 10.803 | 11.449 | 1.06x | 1.28e+02 | 7.69e-03 |
| grouped | e05c6c03 | PASSED | 4.692 | 10.996 | 2.34x | 2.56e+02 | 7.58e-03 |
| grouped | 6230e838 | PASSED | 33.446 | 13.694 | 0.41x | 2.05e+03 | 4.27e-01 |
| grouped | 8f1ff9f1 | PASSED | 44.472 | 15.893 | 0.36x | 1.02e+03 | 2.11e-01 |
| grouped | 1a4c6ba1 | PASSED | 245.018 | 20.859 | 0.09x | 2.05e+03 | 3.00e+00 |
| grouped | a7c2bcfd | PASSED | 19.768 | 12.510 | 0.63x | 5.12e+02 | 1.22e-01 |
| grouped | 2e69caee | PASSED | 7.747 | 11.341 | 1.46x | 5.12e+02 | 7.52e-02 |
| grouped | 8cba5890 | PASSED | 18.293 | 12.201 | 0.67x | 2.56e+02 | 4.93e-02 |
| grouped | 5e8dc11c | PASSED | 2725.388 | 45.311 | 0.02x | 2.05e+03 | 1.50e+07 |
| grouped | 58a34f27 | PASSED | 1780.683 | 35.706 | 0.02x | 2.05e+03 | 5.70e+01 |
| grouped | 5eadab1e | PASSED | 28.706 | 13.705 | 0.48x | 1.02e+03 | 1.11e-01 |
| grouped | eedc63b2 | PASSED | 29.358 | 13.505 | 0.46x | 1.02e+03 | 5.48e-02 |
| grouped | e626d3e6 | PASSED | 40.580 | 15.356 | 0.38x | 1.02e+03 | 1.38e-01 |
| grouped | 74d7ff04 | PASSED | 38.609 | 14.754 | 0.38x | 2.05e+03 | 7.41e-02 |
| grouped | 4822167c | PASSED | 40.721 | 14.799 | 0.36x | 1.02e+03 | 1.10e-01 |
| grouped | 81955b1e | PASSED | 36.778 | 14.430 | 0.39x | 2.05e+03 | 2.05e-01 |
| grouped | 76010cb4 | PASSED | 35.116 | 14.153 | 0.40x | 1.02e+03 | 2.91e-01 |
| grouped | fc378037 | PASSED | 36.738 | 14.506 | 0.39x | 5.12e+02 | 7.32e-02 |
| grouped | f7d6ac7c | PASSED | 25.913 | 13.151 | 0.51x | 1.02e+03 | 5.16e-01 |
| grouped_profile | b8f4f012 | PASSED | 11.371 | 11.507 | 1.01x | 5.12e+02 | 7.46e-03 |
| grouped_profile | e05c6c03 | PASSED | 4.779 | 10.913 | 2.28x | 8.00e+00 | 4.84e-02 |
| grouped_profile | 6230e838 | PASSED | 33.563 | 13.674 | 0.41x | 1.02e+03 | 1.06e-01 |
| grouped_profile | 8f1ff9f1 | PASSED | 44.568 | 15.899 | 0.36x | 1.02e+03 | 5.92e-01 |
| grouped_profile | 1a4c6ba1 | PASSED | 245.255 | 20.875 | 0.09x | 2.05e+03 | 1.69e+02 |
| grouped_profile | a7c2bcfd | PASSED | 19.857 | 12.507 | 0.63x | 5.12e+02 | 8.61e-02 |
| grouped_profile | 2e69caee | PASSED | 7.947 | 11.331 | 1.43x | 5.12e+02 | 1.26e-02 |
| grouped_profile | 8cba5890 | PASSED | 18.498 | 12.183 | 0.66x | 5.12e+02 | 5.30e-02 |
| grouped_profile | 5e8dc11c | PASSED | 2726.316 | 45.388 | 0.02x | 2.05e+03 | 2.03e+06 |
| grouped_profile | 58a34f27 | PASSED | 1780.885 | 35.678 | 0.02x | 2.05e+03 | 2.55e+01 |
| grouped_profile | 5eadab1e | PASSED | 28.898 | 13.692 | 0.47x | 1.02e+03 | 1.45e-01 |
| grouped_profile | eedc63b2 | PASSED | 29.556 | 13.487 | 0.46x | 1.02e+03 | 2.24e-01 |
| grouped_profile | e626d3e6 | PASSED | 40.671 | 15.786 | 0.39x | 1.02e+03 | 7.66e-01 |
| grouped_profile | 74d7ff04 | PASSED | 38.801 | 14.830 | 0.38x | 5.12e+02 | 9.09e-01 |
| grouped_profile | 4822167c | PASSED | 40.810 | 16.313 | 0.40x | 1.02e+03 | 5.00e+00 |
| grouped_profile | 81955b1e | PASSED | 37.766 | 14.933 | 0.40x | 1.02e+03 | 7.88e-02 |
| grouped_profile | 76010cb4 | PASSED | 35.712 | 14.902 | 0.42x | 1.02e+03 | 1.21e-01 |
| grouped_profile | fc378037 | PASSED | 36.952 | 15.337 | 0.42x | 5.12e+02 | 1.27e+00 |
| grouped_profile | f7d6ac7c | PASSED | 25.973 | 13.664 | 0.53x | 5.12e+02 | 6.19e-02 |

## Speedup by variant (PASSED rows only)

| variant | n | mean speedup | min speedup | max speedup |
|---|---|---|---|---|
| baseline | 19 | 0.548x | 0.016x | 2.350x |
| profile_only | 19 | 0.529x | 0.016x | 2.237x |
| grouped | 19 | 0.569x | 0.017x | 2.343x |
| grouped_profile | 19 | 0.566x | 0.017x | 2.284x |
