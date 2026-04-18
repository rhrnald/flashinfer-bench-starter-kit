# Sweep 20260418T202247Z  (B200)

| metric | value |
|---|---|
| timestamp | 20260418T202247Z |
| gpu | B200 |
| max_workloads | all |
| warmup_runs | 1 |
| iterations | 3 |
| num_trials | 1 |
| n_variants | 4 |
| solution | moe_fp8_cpu_ref_cuda |
| definition | moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048 |

## Variants

- **default**: _(none)_
- **legacy**: `FIB_MOE_LEGACY=1`
- **default_profile**: `FIB_MOE_PROFILE=1`
- **legacy_profile**: `FIB_MOE_LEGACY=1`, `FIB_MOE_PROFILE=1`

## Results

| variant | workload | status | latency_ms | ref_ms | speedup | abs_err | rel_err |
|---|---|---|---|---|---|---|---|
| default | b8f4f012 | PASSED | 10.742 | 12.268 | 1.14x | 5.12e+02 | 7.58e-03 |
| default | e05c6c03 | PASSED | 4.677 | 11.291 | 2.41x | 1.60e+01 | 8.16e-03 |
| default | 6230e838 | PASSED | 33.433 | 15.432 | 0.46x | 1.02e+03 | 8.85e-03 |
| default | 8f1ff9f1 | PASSED | 44.479 | 16.405 | 0.37x | 1.02e+03 | 4.59e-01 |
| default | 1a4c6ba1 | PASSED | 245.438 | 21.346 | 0.09x | 2.05e+03 | 1.56e+01 |
| default | a7c2bcfd | PASSED | 19.766 | 13.175 | 0.67x | 2.56e+02 | 2.93e-02 |
| default | 2e69caee | PASSED | 7.762 | 11.660 | 1.50x | 5.12e+02 | 1.77e-01 |
| default | 8cba5890 | PASSED | 18.271 | 13.113 | 0.72x | 2.56e+02 | 2.63e-02 |
| default | 5e8dc11c | PASSED | 2734.466 | 46.347 | 0.02x | 2.05e+03 | 4.47e+01 |
| default | 58a34f27 | PASSED | 1781.613 | 36.607 | 0.02x | 4.10e+03 | 4.05e+01 |
| default | 5eadab1e | PASSED | 28.683 | 16.664 | 0.58x | 1.02e+03 | 3.72e-02 |
| default | eedc63b2 | PASSED | 29.329 | 14.265 | 0.49x | 5.12e+02 | 1.50e+00 |
| default | e626d3e6 | PASSED | 40.557 | 15.882 | 0.39x | 1.02e+03 | 4.72e-02 |
| default | 74d7ff04 | PASSED | 38.565 | 15.652 | 0.41x | 1.02e+03 | 4.97e-01 |
| default | 4822167c | PASSED | 40.683 | 18.833 | 0.46x | 1.02e+03 | 1.13e-01 |
| default | 81955b1e | PASSED | 36.735 | 17.854 | 0.49x | 5.12e+02 | 1.05e-01 |
| default | 76010cb4 | PASSED | 35.066 | 14.998 | 0.43x | 1.02e+03 | 5.97e-01 |
| default | fc378037 | PASSED | 36.717 | 15.061 | 0.41x | 1.02e+03 | 9.40e-02 |
| default | f7d6ac7c | PASSED | 25.851 | 13.785 | 0.53x | 5.12e+02 | 8.41e-01 |
| legacy | b8f4f012 | PASSED | 10.840 | 11.842 | 1.09x | 2.56e+02 | 2.66e-02 |
| legacy | e05c6c03 | PASSED | 4.672 | 11.186 | 2.39x | 1.28e+02 | 7.14e-03 |
| legacy | 6230e838 | PASSED | 34.598 | 14.463 | 0.42x | 1.02e+03 | 1.16e-01 |
| legacy | 8f1ff9f1 | PASSED | 57.217 | 16.484 | 0.29x | 1.02e+03 | 2.35e-01 |
| legacy | 1a4c6ba1 | PASSED | 298.341 | 22.153 | 0.07x | 2.05e+03 | 4.40e+00 |
| legacy | a7c2bcfd | PASSED | 20.215 | 13.112 | 0.65x | 1.02e+03 | 7.14e-02 |
| legacy | 2e69caee | PASSED | 7.902 | 11.715 | 1.48x | 5.12e+02 | 3.50e-02 |
| legacy | 8cba5890 | PASSED | 18.492 | 12.812 | 0.69x | 5.12e+02 | 3.47e-02 |
| legacy | 5e8dc11c | PASSED | 2756.286 | 46.068 | 0.02x | 2.05e+03 | 1.77e+07 |
| legacy | 58a34f27 | PASSED | 1817.705 | 36.240 | 0.02x | 2.05e+03 | 6.06e+01 |
| legacy | 5eadab1e | PASSED | 34.005 | 14.212 | 0.42x | 1.02e+03 | 1.07e-01 |
| legacy | eedc63b2 | PASSED | 32.398 | 14.240 | 0.44x | 1.02e+03 | 6.93e-01 |
| legacy | e626d3e6 | PASSED | 45.425 | 15.761 | 0.35x | 5.12e+02 | 2.33e-01 |
| legacy | 74d7ff04 | PASSED | 42.700 | 15.289 | 0.36x | 1.02e+03 | 4.85e-01 |
| legacy | 4822167c | PASSED | 46.906 | 15.448 | 0.33x | 1.02e+03 | 2.97e-01 |
| legacy | 81955b1e | PASSED | 40.351 | 14.920 | 0.37x | 1.02e+03 | 3.08e-01 |
| legacy | 76010cb4 | PASSED | 37.283 | 14.871 | 0.40x | 5.12e+02 | 1.01e+00 |
| legacy | fc378037 | PASSED | 40.614 | 15.007 | 0.37x | 1.02e+03 | 1.03e-01 |
| legacy | f7d6ac7c | PASSED | 26.648 | 13.692 | 0.51x | 1.02e+03 | 3.66e-01 |
| default_profile | b8f4f012 | PASSED | 11.056 | 12.232 | 1.11x | 1.28e+02 | 8.40e-03 |
| default_profile | e05c6c03 | PASSED | 4.776 | 11.196 | 2.34x | 1.60e+01 | 8.58e-01 |
| default_profile | 6230e838 | PASSED | 33.716 | 14.632 | 0.43x | 1.02e+03 | 2.27e-02 |
| default_profile | 8f1ff9f1 | PASSED | 44.667 | 16.365 | 0.37x | 1.02e+03 | 1.61e-01 |
| default_profile | 1a4c6ba1 | PASSED | 245.370 | 21.167 | 0.09x | 2.05e+03 | 4.30e+00 |
| default_profile | a7c2bcfd | PASSED | 19.826 | 12.989 | 0.66x | 5.12e+02 | 8.11e-02 |
| default_profile | 2e69caee | PASSED | 7.813 | 11.630 | 1.49x | 2.56e+02 | 1.02e-01 |
| default_profile | 8cba5890 | PASSED | 18.348 | 12.844 | 0.70x | 1.02e+03 | 3.18e-02 |
| default_profile | 5e8dc11c | PASSED | 2726.293 | 46.248 | 0.02x | 2.05e+03 | 3.89e+01 |
| default_profile | 58a34f27 | PASSED | 1781.084 | 36.297 | 0.02x | 2.05e+03 | 4.37e+02 |
| default_profile | 5eadab1e | PASSED | 28.906 | 14.193 | 0.49x | 2.05e+03 | 1.65e-01 |
| default_profile | eedc63b2 | PASSED | 29.442 | 14.171 | 0.48x | 5.12e+02 | 6.52e-02 |
| default_profile | e626d3e6 | PASSED | 40.676 | 15.940 | 0.39x | 1.02e+03 | 1.12e-01 |
| default_profile | 74d7ff04 | PASSED | 38.713 | 15.399 | 0.40x | 1.02e+03 | 2.55e-01 |
| default_profile | 4822167c | PASSED | 40.770 | 15.624 | 0.38x | 1.02e+03 | 9.93e-01 |
| default_profile | 81955b1e | PASSED | 36.813 | 15.305 | 0.42x | 1.02e+03 | 1.12e+00 |
| default_profile | 76010cb4 | PASSED | 35.170 | 15.079 | 0.43x | 1.02e+03 | 6.74e-01 |
| default_profile | fc378037 | PASSED | 36.820 | 15.175 | 0.41x | 1.02e+03 | 1.42e-01 |
| default_profile | f7d6ac7c | PASSED | 25.958 | 13.841 | 0.53x | 1.02e+03 | 3.17e-02 |
| legacy_profile | b8f4f012 | PASSED | 11.057 | 11.910 | 1.08x | 5.12e+02 | 5.81e-03 |
| legacy_profile | e05c6c03 | PASSED | 4.817 | 11.231 | 2.33x | 8.00e+00 | 7.35e-03 |
| legacy_profile | 6230e838 | PASSED | 34.709 | 14.695 | 0.42x | 5.12e+02 | 4.48e-01 |
| legacy_profile | 8f1ff9f1 | PASSED | 57.979 | 16.430 | 0.28x | 1.02e+03 | 2.62e-01 |
| legacy_profile | 1a4c6ba1 | PASSED | 299.751 | 21.213 | 0.07x | 2.05e+03 | 1.04e+01 |
| legacy_profile | a7c2bcfd | PASSED | 20.368 | 13.084 | 0.64x | 2.56e+02 | 6.70e-02 |
| legacy_profile | 2e69caee | PASSED | 7.980 | 11.677 | 1.46x | 1.02e+03 | 1.59e-02 |
| legacy_profile | 8cba5890 | PASSED | 18.563 | 12.806 | 0.69x | 5.12e+02 | 5.19e-02 |
| legacy_profile | 5e8dc11c | PASSED | 2757.942 | 46.285 | 0.02x | 2.05e+03 | 1.52e+02 |
| legacy_profile | 58a34f27 | PASSED | 1822.284 | 36.239 | 0.02x | 2.05e+03 | 2.15e+01 |
| legacy_profile | 5eadab1e | PASSED | 34.384 | 14.418 | 0.42x | 1.02e+03 | 3.81e-01 |
| legacy_profile | eedc63b2 | PASSED | 32.655 | 14.376 | 0.44x | 1.02e+03 | 1.86e-01 |
| legacy_profile | e626d3e6 | PASSED | 45.842 | 15.961 | 0.35x | 1.02e+03 | 1.46e+00 |
| legacy_profile | 74d7ff04 | PASSED | 42.883 | 16.224 | 0.38x | 1.02e+03 | 8.78e-02 |
| legacy_profile | 4822167c | PASSED | 47.168 | 15.756 | 0.33x | 1.02e+03 | 1.16e-01 |
| legacy_profile | 81955b1e | PASSED | 40.781 | 15.205 | 0.37x | 2.05e+03 | 1.43e-01 |
| legacy_profile | 76010cb4 | PASSED | 37.340 | 15.800 | 0.42x | 1.02e+03 | 1.89e-01 |
| legacy_profile | fc378037 | PASSED | 40.741 | 15.418 | 0.38x | 1.02e+03 | 1.28e-01 |
| legacy_profile | f7d6ac7c | PASSED | 26.869 | 13.899 | 0.52x | 1.02e+03 | 4.70e-02 |

## Speedup by variant (PASSED rows only)

| variant | n | mean speedup | min speedup | max speedup |
|---|---|---|---|---|
| default | 19 | 0.610x | 0.017x | 2.414x |
| legacy | 19 | 0.562x | 0.017x | 2.395x |
| default_profile | 19 | 0.587x | 0.017x | 2.344x |
| legacy_profile | 19 | 0.559x | 0.017x | 2.332x |
