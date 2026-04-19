# Sweep 20260418T201848Z  (B200)

| metric | value |
|---|---|
| timestamp | 20260418T201848Z |
| gpu | B200 |
| max_workloads | all |
| warmup_runs | 2 |
| iterations | 10 |
| num_trials | 2 |
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
| baseline | b8f4f012 | PASSED | 10.833 | 12.141 | 1.12x | 5.12e+02 | 5.94e-02 |
| baseline | e05c6c03 | PASSED | 4.698 | 11.243 | 2.39x | 2.56e+02 | 2.01e-02 |
| baseline | 6230e838 | PASSED | 34.712 | 14.869 | 0.43x | 1.02e+03 | 8.29e-01 |
| baseline | 8f1ff9f1 | PASSED | 58.448 | 16.463 | 0.28x | 1.02e+03 | 4.07e+01 |
| baseline | 1a4c6ba1 | PASSED | 299.356 | 22.374 | 0.07x | 2.05e+03 | 1.42e+01 |
| baseline | a7c2bcfd | PASSED | 20.256 | 12.554 | 0.62x | 5.12e+02 | 3.54e-01 |
| baseline | 2e69caee | PASSED | 7.909 | 11.433 | 1.45x | 1.02e+03 | 5.62e-02 |
| baseline | 8cba5890 | PASSED | 18.519 | 12.946 | 0.70x | 5.12e+02 | 7.75e-02 |
| baseline | 5e8dc11c | PASSED | 2758.502 | 46.979 | 0.02x | 4.10e+03 | 1.66e+07 |
| baseline | 58a34f27 | PASSED | 1821.083 | 36.429 | 0.02x | 4.10e+03 | 9.47e+06 |
| baseline | 5eadab1e | PASSED | 34.069 | 13.772 | 0.40x | 1.02e+03 | 8.00e-02 |
| baseline | eedc63b2 | PASSED | 32.424 | 13.593 | 0.42x | 1.02e+03 | 7.57e-01 |
| baseline | e626d3e6 | PASSED | 45.612 | 15.410 | 0.34x | 2.05e+03 | 1.16e-01 |
| baseline | 74d7ff04 | PASSED | 42.723 | 14.889 | 0.35x | 1.02e+03 | 5.34e-01 |
| baseline | 4822167c | PASSED | 46.916 | 15.032 | 0.32x | 1.02e+03 | 5.28e-01 |
| baseline | 81955b1e | PASSED | 40.525 | 14.570 | 0.36x | 2.05e+03 | 8.39e-02 |
| baseline | 76010cb4 | PASSED | 37.381 | 14.274 | 0.38x | 1.02e+03 | 3.96e-01 |
| baseline | fc378037 | PASSED | 40.563 | 14.615 | 0.36x | 1.02e+03 | 3.68e-01 |
| baseline | f7d6ac7c | PASSED | 26.646 | 13.280 | 0.50x | 5.12e+02 | 5.31e-01 |
| profile_only | b8f4f012 | PASSED | 10.938 | 11.548 | 1.06x | 5.12e+02 | 7.87e-03 |
| profile_only | e05c6c03 | PASSED | 4.761 | 10.938 | 2.30x | 1.02e+03 | 2.14e-02 |
| profile_only | 6230e838 | PASSED | 34.767 | 13.858 | 0.40x | 1.02e+03 | 2.54e-01 |
| profile_only | 8f1ff9f1 | PASSED | 59.002 | 15.987 | 0.27x | 2.05e+03 | 8.58e-01 |
| profile_only | 1a4c6ba1 | PASSED | 300.117 | 20.794 | 0.07x | 2.05e+03 | 9.44e+00 |
| profile_only | a7c2bcfd | PASSED | 20.420 | 12.600 | 0.62x | 1.02e+03 | 2.85e-01 |
| profile_only | 2e69caee | PASSED | 8.005 | 11.437 | 1.43x | 2.56e+02 | 3.24e-02 |
| profile_only | 8cba5890 | PASSED | 18.692 | 12.336 | 0.66x | 5.12e+02 | 8.79e-02 |
| profile_only | 5e8dc11c | PASSED | 2759.194 | 45.808 | 0.02x | 2.05e+03 | 1.95e+02 |
| profile_only | 58a34f27 | PASSED | 1821.373 | 36.138 | 0.02x | 2.05e+03 | 1.02e+07 |
| profile_only | 5eadab1e | PASSED | 34.141 | 13.795 | 0.40x | 1.02e+03 | 2.04e-01 |
| profile_only | eedc63b2 | PASSED | 32.464 | 13.790 | 0.42x | 1.02e+03 | 3.24e-01 |
| profile_only | e626d3e6 | PASSED | 45.721 | 15.521 | 0.34x | 1.02e+03 | 6.15e-01 |
| profile_only | 74d7ff04 | PASSED | 42.830 | 15.078 | 0.35x | 1.02e+03 | 9.92e-02 |
| profile_only | 4822167c | PASSED | 47.083 | 15.228 | 0.32x | 1.02e+03 | 4.30e-01 |
| profile_only | 81955b1e | PASSED | 40.723 | 14.747 | 0.36x | 1.02e+03 | 1.15e-01 |
| profile_only | 76010cb4 | PASSED | 37.602 | 14.441 | 0.38x | 1.02e+03 | 2.55e-01 |
| profile_only | fc378037 | PASSED | 40.935 | 14.903 | 0.36x | 1.02e+03 | 3.56e-01 |
| profile_only | f7d6ac7c | PASSED | 26.774 | 13.379 | 0.50x | 2.05e+03 | 2.01e-02 |
| grouped | b8f4f012 | PASSED | 10.780 | 11.576 | 1.07x | 2.56e+02 | 6.25e-02 |
| grouped | e05c6c03 | PASSED | 4.684 | 10.956 | 2.34x | 3.20e+01 | 6.33e-03 |
| grouped | 6230e838 | PASSED | 33.473 | 14.043 | 0.42x | 1.02e+03 | 1.12e-01 |
| grouped | 8f1ff9f1 | PASSED | 44.526 | 16.063 | 0.36x | 1.02e+03 | 2.91e-01 |
| grouped | 1a4c6ba1 | PASSED | 246.730 | 20.994 | 0.09x | 2.05e+03 | 3.99e+00 |
| grouped | a7c2bcfd | PASSED | 19.818 | 12.692 | 0.64x | 1.02e+03 | 1.57e-01 |
| grouped | 2e69caee | PASSED | 7.749 | 11.482 | 1.48x | 5.12e+02 | 5.60e-02 |
| grouped | 8cba5890 | PASSED | 18.341 | 12.415 | 0.68x | 2.05e+03 | 7.45e-02 |
| grouped | 5e8dc11c | PASSED | 2729.059 | 45.981 | 0.02x | 4.10e+03 | 1.08e+07 |
| grouped | 58a34f27 | PASSED | 1785.487 | 36.016 | 0.02x | 2.05e+03 | 1.56e+06 |
| grouped | 5eadab1e | PASSED | 28.722 | 13.883 | 0.48x | 1.02e+03 | 3.31e-01 |
| grouped | eedc63b2 | PASSED | 29.359 | 13.752 | 0.47x | 1.02e+03 | 1.46e-01 |
| grouped | e626d3e6 | PASSED | 40.590 | 15.545 | 0.38x | 2.05e+03 | 9.39e-01 |
| grouped | 74d7ff04 | PASSED | 38.631 | 15.070 | 0.39x | 1.02e+03 | 1.23e-01 |
| grouped | 4822167c | PASSED | 40.719 | 15.234 | 0.37x | 1.02e+03 | 5.15e-01 |
| grouped | 81955b1e | PASSED | 36.789 | 14.706 | 0.40x | 1.02e+03 | 4.52e-02 |
| grouped | 76010cb4 | PASSED | 35.115 | 14.475 | 0.41x | 1.02e+03 | 1.94e+00 |
| grouped | fc378037 | PASSED | 36.787 | 14.775 | 0.40x | 1.02e+03 | 5.90e-01 |
| grouped | f7d6ac7c | PASSED | 25.897 | 13.413 | 0.52x | 5.12e+02 | 2.65e-02 |
| grouped_profile | b8f4f012 | PASSED | 10.974 | 11.667 | 1.06x | 2.56e+02 | 7.52e-03 |
| grouped_profile | e05c6c03 | PASSED | 4.743 | 10.996 | 2.32x | 1.02e+03 | 1.00e-01 |
| grouped_profile | 6230e838 | PASSED | 33.570 | 14.043 | 0.42x | 1.02e+03 | 4.79e-02 |
| grouped_profile | 8f1ff9f1 | PASSED | 44.587 | 16.102 | 0.36x | 2.05e+03 | 2.27e-01 |
| grouped_profile | 1a4c6ba1 | PASSED | 246.707 | 20.950 | 0.08x | 2.05e+03 | 3.42e+02 |
| grouped_profile | a7c2bcfd | PASSED | 19.917 | 12.842 | 0.64x | 1.02e+03 | 1.23e-01 |
| grouped_profile | 2e69caee | PASSED | 7.805 | 11.567 | 1.48x | 1.02e+03 | 3.77e-01 |
| grouped_profile | 8cba5890 | PASSED | 18.356 | 12.560 | 0.68x | 1.02e+03 | 2.19e-02 |
| grouped_profile | 5e8dc11c | PASSED | 2729.660 | 45.968 | 0.02x | 4.10e+03 | 1.27e+02 |
| grouped_profile | 58a34f27 | PASSED | 1786.298 | 36.086 | 0.02x | 4.10e+03 | 1.37e+07 |
| grouped_profile | 5eadab1e | PASSED | 28.784 | 13.636 | 0.47x | 1.02e+03 | 1.52e-01 |
| grouped_profile | eedc63b2 | PASSED | 29.466 | 13.685 | 0.46x | 1.02e+03 | 6.09e-01 |
| grouped_profile | e626d3e6 | PASSED | 40.706 | 15.502 | 0.38x | 2.05e+03 | 6.40e-02 |
| grouped_profile | 74d7ff04 | PASSED | 38.890 | 14.853 | 0.38x | 1.02e+03 | 4.61e-01 |
| grouped_profile | 4822167c | PASSED | 40.855 | 14.955 | 0.37x | 1.02e+03 | 2.94e-01 |
| grouped_profile | 81955b1e | PASSED | 36.896 | 14.552 | 0.39x | 1.02e+03 | 4.92e-01 |
| grouped_profile | 76010cb4 | PASSED | 35.333 | 14.270 | 0.40x | 5.12e+02 | 1.53e+00 |
| grouped_profile | fc378037 | PASSED | 37.058 | 14.625 | 0.39x | 1.02e+03 | 1.78e-01 |
| grouped_profile | f7d6ac7c | PASSED | 25.959 | 13.282 | 0.51x | 5.12e+02 | 7.57e-02 |

## Speedup by variant (PASSED rows only)

| variant | n | mean speedup | min speedup | max speedup |
|---|---|---|---|---|
| baseline | 19 | 0.554x | 0.017x | 2.393x |
| profile_only | 19 | 0.541x | 0.017x | 2.297x |
| grouped | 19 | 0.576x | 0.017x | 2.339x |
| grouped_profile | 19 | 0.572x | 0.017x | 2.318x |
