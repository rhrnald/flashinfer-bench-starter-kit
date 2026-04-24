# MoE Precision Search 2026-04-23T05:02:57Z

| metric | value |
|---|---|
| timestamp | 2026-04-23T05:02:57Z |
| definition | moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048 |
| seed | 4321 |
| atol | 1.0 |
| rtol | 0.3 |
| required_matched_ratio | 0.9 |
| strict_atol | 0.01 |
| strict_rtol | 0.01 |
| device | NVIDIA B200 |
| run_stage | contest_panel |
| panel_size | 19 |
| panel_indices | 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18 |
| n_candidates | 9 |
| n_survivors | 9 |
| contest_safe_survivor_count | 9 |
| contest_safe_survivors | gemm1_operands__f16, gemm1_output__f16, swiglu_input__f16, out_accumulator__bf16, gemm1_accumulator__f16, gemm2_operands__bf16, gemm2_accumulator__bf16, swiglu_output__fp8__block, hidden_dequant__fp8__block |
| strict_safe_survivor_count | 5 |
| strict_safe_survivors | gemm1_accumulator__f16, gemm1_operands__f16, gemm1_output__f16, out_accumulator__bf16, swiglu_input__f16 |
| contest_only_survivor_count | 4 |
| contest_only_survivors | gemm2_accumulator__bf16, gemm2_operands__bf16, hidden_dequant__fp8__block, swiglu_output__fp8__block |

## Survivor Summary

| category | count | candidates |
|---|---:|---|
| contest_safe_single_stage | 9 | gemm1_operands__f16, gemm1_output__f16, swiglu_input__f16, out_accumulator__bf16, gemm1_accumulator__f16, gemm2_operands__bf16, gemm2_accumulator__bf16, swiglu_output__fp8__block, hidden_dequant__fp8__block |
| strict_safe_single_stage | 5 | gemm1_accumulator__f16, gemm1_operands__f16, gemm1_output__f16, out_accumulator__bf16, swiglu_input__f16 |
| contest_only_single_stage | 4 | gemm2_accumulator__bf16, gemm2_operands__bf16, hidden_dequant__fp8__block, swiglu_output__fp8__block |

## Stage Summary

| stage | best_safe_mode | scale | worst_matched_contest | worst_matched_strict | worst_rel | status |
|---|---|---|---:|---:|---:|---|
| gemm1_accumulator | f16 | none | 0.997930 | 0.901367 | 1.4400e+10 | safe |
| gemm1_operands | f16 | none | 0.999429 | 0.975446 | 1.9250e+09 | safe |
| gemm1_output | f16 | none | 0.999302 | 0.980329 | 2.1750e+09 | safe |
| gemm2_accumulator | bf16 | none | 0.993025 | 0.747070 | 8.4500e+09 | safe |
| gemm2_operands | bf16 | none | 0.995954 | 0.847238 | 4.9000e+09 | safe |
| hidden_dequant | fp8 | block | 0.927037 | 0.180385 | 6.9600e+10 | safe |
| out_accumulator | bf16 | none | 0.998326 | 0.933733 | 7.9000e+09 | safe |
| swiglu_input | f16 | none | 0.999302 | 0.980329 | 2.1750e+09 | safe |
| swiglu_output | fp8 | block | 0.954520 | 0.256278 | 2.9120e+11 | safe |

## Cumulative Safe Frontier

| order | candidate | worst_matched_contest | worst_matched_strict | worst_rel | status | kept |
|---|---|---:|---:|---:|---|---|
| 1 | gemm1_accumulator__f16 | 0.996931 | 0.903739 | 6.7500e+09 | safe | yes |
| 2 | gemm1_accumulator__f16+gemm1_operands__f16 | 0.999372 | 0.973075 | 2.1739e+04 | safe | yes |
| 3 | gemm1_accumulator__f16+gemm1_operands__f16+gemm1_output__f16 | 0.999310 | 0.964844 | 1.8242e+05 | safe | yes |
| 4 | gemm1_accumulator__f16+gemm1_operands__f16+gemm1_output__f16+gemm2_accumulator__bf16 | 0.992885 | 0.747768 | 1.4177e+05 | safe | yes |
| 5 | gemm1_accumulator__f16+gemm1_operands__f16+gemm1_output__f16+gemm2_accumulator__bf16+gemm2_operands__bf16 | 0.994978 | 0.840541 | 3.8000e+09 | safe | yes |
| 6 | gemm1_accumulator__f16+gemm1_operands__f16+gemm1_output__f16+gemm2_accumulator__bf16+gemm2_operands__bf16+hidden_dequant__fp8__block | 0.923549 | 0.162946 | 2.9440e+11 | safe | yes |
| 7 | gemm1_accumulator__f16+gemm1_operands__f16+gemm1_output__f16+gemm2_accumulator__bf16+gemm2_operands__bf16+hidden_dequant__fp8__block+out_accumulator__bf16 | 0.925781 | 0.160854 | 1.1120e+11 | safe | yes |
| 8 | gemm1_accumulator__f16+gemm1_operands__f16+gemm1_output__f16+gemm2_accumulator__bf16+gemm2_operands__bf16+hidden_dequant__fp8__block+out_accumulator__bf16+swiglu_input__f16 | 0.917271 | 0.148996 | 1.5925e+06 | safe | yes |
| 9 | gemm1_accumulator__f16+gemm1_operands__f16+gemm1_output__f16+gemm2_accumulator__bf16+gemm2_operands__bf16+hidden_dequant__fp8__block+out_accumulator__bf16+swiglu_input__f16+swiglu_output__fp8__block | 0.902483 | 0.143136 | 3.8720e+11 | safe | yes |

## BF16/F16 Margin

| stage | preferred | bf16_contest | f16_contest | bf16_strict | f16_strict | |
|---|---|---:|---:|---:|---:|---|

## Pairwise Summary

| pair | worst_matched_contest | worst_matched_strict | worst_rel | status |
|---|---:|---:|---:|---|
| gemm1_output__f16+gemm2_accumulator__bf16 | 0.989955 | 0.754325 | 6.8400e+10 | safe |
| gemm1_accumulator__f16+gemm2_accumulator__bf16 | 0.990792 | 0.728237 | 1.1120e+11 | safe |
| gemm2_accumulator__bf16+out_accumulator__bf16 | 0.991350 | 0.745815 | 7.4800e+10 | safe |
| gemm1_operands__f16+gemm2_accumulator__bf16 | 0.991350 | 0.756557 | 8.3200e+10 | safe |
| gemm1_accumulator__f16+out_accumulator__bf16 | 0.995536 | 0.875419 | 2.5300e+10 | safe |
| gemm1_accumulator__f16+gemm1_output__f16 | 0.995536 | 0.891881 | 1.2900e+10 | safe |
| gemm1_operands__f16+out_accumulator__bf16 | 0.997210 | 0.935128 | 1.4700e+10 | safe |
| gemm1_output__f16+out_accumulator__bf16 | 0.997628 | 0.939174 | 1.4700e+10 | safe |
| gemm1_operands__f16+gemm1_output__f16 | 0.998605 | 0.963309 | 9.2000e+09 | safe |
| gemm1_operands__f16+gemm1_accumulator__f16 | 0.999023 | 0.971959 | 6.1750e+09 | safe |

## Stress Summary

| candidate | worst_workload | worst_seq_len | worst_matched_contest | worst_matched_strict | status |
|---|---|---:|---:|---:|---|
| gemm2_accumulator__bf16 | e05c6c03 | 1 | 0.993025 | 0.747070 | safe |

## Promotion Summary

| category | candidates |
|---|---|
| bf16_f16_survivors | gemm1_operands__f16, gemm1_output__f16, swiglu_input__f16, out_accumulator__bf16, gemm1_accumulator__f16, gemm2_operands__bf16, gemm2_accumulator__bf16, swiglu_output__fp8__block, hidden_dequant__fp8__block |
| strict_survivors | gemm1_accumulator__f16, gemm1_operands__f16, gemm1_output__f16, out_accumulator__bf16, swiglu_input__f16 |
| pairwise_shortlist | gemm1_operands__f16, gemm1_accumulator__f16, gemm1_output__f16, gemm2_accumulator__bf16, out_accumulator__bf16 |

## Sampled Results

| candidate | phase | workload | seq_len | matched_contest | matched_strict | max_abs | max_rel | failure |
|---|---|---|---:|---:|---:|---:|---:|---|
| baseline__fp32 | single_stage | b8f4f012 | 7 | 1.000000 | 1.000000 | 0.0000e+00 | 0.0000e+00 | pass |
| gemm1_accumulator__f16 | single_stage | b8f4f012 | 7 | 0.998924 | 0.970703 | 2.0480e+03 | 9.8858e+00 | pass |
| gemm1_operands__f16 | single_stage | b8f4f012 | 7 | 0.999741 | 0.991709 | 2.0480e+03 | 3.7854e+00 | pass |
| gemm1_output__f16 | single_stage | b8f4f012 | 7 | 0.999781 | 0.994041 | 1.0240e+03 | 9.4521e-01 | pass |
| gemm2_accumulator__bf16 | single_stage | b8f4f012 | 7 | 0.997509 | 0.930465 | 4.0960e+03 | 9.9530e+01 | pass |
| gemm2_operands__bf16 | single_stage | b8f4f012 | 7 | 0.998246 | 0.955995 | 2.0480e+03 | 9.7200e+00 | pass |
| hidden_dequant__fp8__block | single_stage | b8f4f012 | 7 | 0.978456 | 0.763612 | 2.2528e+04 | 7.7986e+02 | pass |
| out_accumulator__bf16 | single_stage | b8f4f012 | 7 | 0.999223 | 0.979951 | 2.0480e+03 | 3.8800e+00 | pass |
| swiglu_input__f16 | single_stage | b8f4f012 | 7 | 0.999781 | 0.994041 | 1.0240e+03 | 9.4521e-01 | pass |
| swiglu_output__fp8__block | single_stage | b8f4f012 | 7 | 0.985810 | 0.789581 | 1.1456e+04 | 1.4521e+02 | pass |
| baseline__fp32 | single_stage | e05c6c03 | 1 | 1.000000 | 1.000000 | 0.0000e+00 | 0.0000e+00 | pass |
| gemm1_accumulator__f16 | single_stage | e05c6c03 | 1 | 0.998326 | 0.901367 | 2.0480e+03 | 1.0475e+01 | pass |
| gemm1_operands__f16 | single_stage | e05c6c03 | 1 | 0.999442 | 0.975446 | 2.0480e+03 | 1.2775e+01 | pass |
| gemm1_output__f16 | single_stage | e05c6c03 | 1 | 0.999302 | 0.980329 | 2.0480e+03 | 2.3647e+00 | pass |
| gemm2_accumulator__bf16 | single_stage | e05c6c03 | 1 | 0.993025 | 0.747070 | 6.1440e+03 | 9.1821e+01 | pass |
| gemm2_operands__bf16 | single_stage | e05c6c03 | 1 | 0.995954 | 0.847238 | 4.0960e+03 | 1.8619e+01 | pass |
| hidden_dequant__fp8__block | single_stage | e05c6c03 | 1 | 0.927037 | 0.180385 | 1.9200e+04 | 1.8995e+02 | pass |
| out_accumulator__bf16 | single_stage | e05c6c03 | 1 | 0.998326 | 0.933733 | 4.0960e+03 | 1.4152e+01 | pass |
| swiglu_input__f16 | single_stage | e05c6c03 | 1 | 0.999302 | 0.980329 | 2.0480e+03 | 2.3647e+00 | pass |
| swiglu_output__fp8__block | single_stage | e05c6c03 | 1 | 0.954520 | 0.256278 | 1.1776e+04 | 1.6263e+02 | pass |
| baseline__fp32 | single_stage | 6230e838 | 32 | 1.000000 | 1.000000 | 0.0000e+00 | 0.0000e+00 | pass |
| gemm1_accumulator__f16 | single_stage | 6230e838 | 32 | 0.998574 | 0.956286 | 2.0480e+03 | 1.0657e+03 | pass |
| gemm1_operands__f16 | single_stage | 6230e838 | 32 | 0.999651 | 0.988477 | 2.0480e+03 | 2.1963e+03 | pass |
| gemm1_output__f16 | single_stage | 6230e838 | 32 | 0.999721 | 0.991991 | 2.0480e+03 | 1.4923e+03 | pass |
| gemm2_accumulator__bf16 | single_stage | 6230e838 | 32 | 0.996674 | 0.899279 | 5.1200e+03 | 7.1670e+03 | pass |
| gemm2_operands__bf16 | single_stage | 6230e838 | 32 | 0.998082 | 0.937317 | 4.0960e+03 | 3.3503e+03 | pass |
| hidden_dequant__fp8__block | single_stage | 6230e838 | 32 | 0.968122 | 0.661259 | 2.4832e+04 | 2.4477e+04 | pass |
| out_accumulator__bf16 | single_stage | 6230e838 | 32 | 0.999254 | 0.976144 | 4.0960e+03 | 7.2024e+02 | pass |
| swiglu_input__f16 | single_stage | 6230e838 | 32 | 0.999721 | 0.991991 | 2.0480e+03 | 1.4923e+03 | pass |
| swiglu_output__fp8__block | single_stage | 6230e838 | 32 | 0.979723 | 0.696886 | 1.4336e+04 | 1.8348e+04 | pass |
| baseline__fp32 | single_stage | 8f1ff9f1 | 80 | 1.000000 | 1.000000 | 0.0000e+00 | 0.0000e+00 | pass |
| gemm1_accumulator__f16 | single_stage | 8f1ff9f1 | 80 | 0.998017 | 0.938841 | 2.0480e+03 | 1.4198e+02 | pass |
| gemm1_operands__f16 | single_stage | 8f1ff9f1 | 80 | 0.999466 | 0.983723 | 4.0960e+03 | 8.8730e+01 | pass |
| gemm1_output__f16 | single_stage | 8f1ff9f1 | 80 | 0.999627 | 0.988227 | 2.0480e+03 | 8.9607e+01 | pass |
| gemm2_accumulator__bf16 | single_stage | 8f1ff9f1 | 80 | 0.995778 | 0.862270 | 4.8640e+03 | 1.7614e+03 | pass |
| gemm2_operands__bf16 | single_stage | 8f1ff9f1 | 80 | 0.997028 | 0.909610 | 4.0960e+03 | 1.5177e+03 | pass |
| hidden_dequant__fp8__block | single_stage | 8f1ff9f1 | 80 | 0.954776 | 0.500277 | 2.3040e+04 | 1.2019e+04 | pass |
| out_accumulator__bf16 | single_stage | 8f1ff9f1 | 80 | 0.999208 | 0.974486 | 4.0960e+03 | 4.9574e+02 | pass |
| swiglu_input__f16 | single_stage | 8f1ff9f1 | 80 | 0.999627 | 0.988227 | 2.0480e+03 | 8.9607e+01 | pass |
| swiglu_output__fp8__block | single_stage | 8f1ff9f1 | 80 | 0.970286 | 0.555387 | 1.6384e+04 | 2.1693e+03 | pass |
| baseline__fp32 | single_stage | 1a4c6ba1 | 901 | 1.000000 | 1.000000 | 0.0000e+00 | 0.0000e+00 | pass |
| gemm1_accumulator__f16 | single_stage | 1a4c6ba1 | 901 | 0.997930 | 0.935821 | 4.0960e+03 | 3.3791e+04 | pass |
| gemm1_operands__f16 | single_stage | 1a4c6ba1 | 901 | 0.999429 | 0.982654 | 4.0960e+03 | 8.2673e+02 | pass |
| gemm1_output__f16 | single_stage | 1a4c6ba1 | 901 | 0.999604 | 0.987785 | 4.0960e+03 | 6.7850e+03 | pass |
| gemm2_accumulator__bf16 | single_stage | 1a4c6ba1 | 901 | 0.995514 | 0.853785 | 8.1920e+03 | 9.0113e+04 | pass |
| gemm2_operands__bf16 | single_stage | 1a4c6ba1 | 901 | 0.996876 | 0.904590 | 4.0960e+03 | 6.0417e+04 | pass |
| hidden_dequant__fp8__block | single_stage | 1a4c6ba1 | 901 | 0.952050 | 0.474596 | 3.2768e+04 | 8.5196e+05 | pass |
| out_accumulator__bf16 | single_stage | 1a4c6ba1 | 901 | 0.999100 | 0.971148 | 4.0960e+03 | 3.3290e+03 | pass |
| swiglu_input__f16 | single_stage | 1a4c6ba1 | 901 | 0.999604 | 0.987785 | 4.0960e+03 | 6.7850e+03 | pass |
| swiglu_output__fp8__block | single_stage | 1a4c6ba1 | 901 | 0.969260 | 0.533420 | 2.0992e+04 | 2.9491e+05 | pass |
| baseline__fp32 | single_stage | a7c2bcfd | 16 | 1.000000 | 1.000000 | 0.0000e+00 | 0.0000e+00 | pass |
| gemm1_accumulator__f16 | single_stage | a7c2bcfd | 16 | 0.998134 | 0.939575 | 4.0960e+03 | 1.4871e+02 | pass |
| gemm1_operands__f16 | single_stage | a7c2bcfd | 16 | 0.999512 | 0.982788 | 2.0480e+03 | 1.3444e+01 | pass |
| gemm1_output__f16 | single_stage | a7c2bcfd | 16 | 0.999634 | 0.988351 | 2.0480e+03 | 1.1448e+01 | pass |
| gemm2_accumulator__bf16 | single_stage | a7c2bcfd | 16 | 0.995928 | 0.861686 | 6.1440e+03 | 1.2601e+02 | pass |
| gemm2_operands__bf16 | single_stage | a7c2bcfd | 16 | 0.997035 | 0.907244 | 4.0960e+03 | 6.4712e+01 | pass |
| hidden_dequant__fp8__block | single_stage | a7c2bcfd | 16 | 0.953212 | 0.478882 | 2.5600e+04 | 7.4355e+02 | pass |
| out_accumulator__bf16 | single_stage | a7c2bcfd | 16 | 0.999355 | 0.980652 | 4.0960e+03 | 1.7959e+01 | pass |
| swiglu_input__f16 | single_stage | a7c2bcfd | 16 | 0.999634 | 0.988351 | 2.0480e+03 | 1.1448e+01 | pass |
| swiglu_output__fp8__block | single_stage | a7c2bcfd | 16 | 0.969962 | 0.534040 | 1.8688e+04 | 6.4170e+02 | pass |
| baseline__fp32 | single_stage | 2e69caee | 15 | 1.000000 | 1.000000 | 0.0000e+00 | 0.0000e+00 | pass |
| gemm1_accumulator__f16 | single_stage | 2e69caee | 15 | 0.999098 | 0.975084 | 2.0480e+03 | 2.3241e+02 | pass |
| gemm1_operands__f16 | single_stage | 2e69caee | 15 | 0.999767 | 0.992950 | 1.0240e+03 | 5.3471e+01 | pass |
| gemm1_output__f16 | single_stage | 2e69caee | 15 | 0.999851 | 0.995033 | 2.0480e+03 | 4.8176e+01 | pass |
| gemm2_accumulator__bf16 | single_stage | 2e69caee | 15 | 0.998233 | 0.938876 | 4.0960e+03 | 5.7500e+02 | pass |
| gemm2_operands__bf16 | single_stage | 2e69caee | 15 | 0.998735 | 0.959505 | 2.0480e+03 | 4.2396e+01 | pass |
| hidden_dequant__fp8__block | single_stage | 2e69caee | 15 | 0.979650 | 0.778088 | 1.7920e+04 | 5.1511e+03 | pass |
| out_accumulator__bf16 | single_stage | 2e69caee | 15 | 0.999693 | 0.988086 | 2.0480e+03 | 3.1824e+01 | pass |
| swiglu_input__f16 | single_stage | 2e69caee | 15 | 0.999851 | 0.995033 | 2.0480e+03 | 4.8176e+01 | pass |
| swiglu_output__fp8__block | single_stage | 2e69caee | 15 | 0.987230 | 0.802102 | 1.1264e+04 | 7.9501e+03 | pass |
| baseline__fp32 | single_stage | 8cba5890 | 14 | 1.000000 | 1.000000 | 0.0000e+00 | 0.0000e+00 | pass |
| gemm1_accumulator__f16 | single_stage | 8cba5890 | 14 | 0.998734 | 0.957250 | 2.0480e+03 | 5.9905e+02 | pass |
| gemm1_operands__f16 | single_stage | 8cba5890 | 14 | 0.999651 | 0.988401 | 2.0480e+03 | 7.3218e+01 | pass |
| gemm1_output__f16 | single_stage | 8cba5890 | 14 | 0.999831 | 0.992417 | 2.0480e+03 | 8.2748e+01 | pass |
| gemm2_accumulator__bf16 | single_stage | 8cba5890 | 14 | 0.997001 | 0.899534 | 4.0960e+03 | 4.9164e+02 | pass |
| gemm2_operands__bf16 | single_stage | 8cba5890 | 14 | 0.998037 | 0.935846 | 2.0480e+03 | 2.0834e+03 | pass |
| hidden_dequant__fp8__block | single_stage | 8cba5890 | 14 | 0.968072 | 0.643395 | 1.4848e+04 | 1.2872e+04 | pass |
| out_accumulator__bf16 | single_stage | 8cba5890 | 14 | 0.999432 | 0.979642 | 2.0480e+03 | 3.3660e+02 | pass |
| swiglu_input__f16 | single_stage | 8cba5890 | 14 | 0.999831 | 0.992417 | 2.0480e+03 | 8.2748e+01 | pass |
| swiglu_output__fp8__block | single_stage | 8cba5890 | 14 | 0.979422 | 0.684082 | 1.0496e+04 | 1.4664e+04 | pass |
| baseline__fp32 | single_stage | 5e8dc11c | 14107 | 1.000000 | 1.000000 | 0.0000e+00 | 0.0000e+00 | pass |
| gemm1_accumulator__f16 | single_stage | 5e8dc11c | 14107 | 0.998120 | 0.941496 | 4.0960e+03 | 1.4400e+10 | pass |
| gemm1_operands__f16 | single_stage | 5e8dc11c | 14107 | 0.999499 | 0.984240 | 4.0960e+03 | 1.9250e+09 | pass |
| gemm1_output__f16 | single_stage | 5e8dc11c | 14107 | 0.999647 | 0.988935 | 4.0960e+03 | 2.1750e+09 | pass |
| gemm2_accumulator__bf16 | single_stage | 5e8dc11c | 14107 | 0.996190 | 0.871167 | 1.2288e+04 | 8.4500e+09 | pass |
| gemm2_operands__bf16 | single_stage | 5e8dc11c | 14107 | 0.997194 | 0.913642 | 4.0960e+03 | 4.9000e+09 | pass |
| hidden_dequant__fp8__block | single_stage | 5e8dc11c | 14107 | 0.956358 | 0.522933 | 4.1984e+04 | 6.9600e+10 | pass |
| out_accumulator__bf16 | single_stage | 5e8dc11c | 14107 | 0.999365 | 0.979642 | 8.1920e+03 | 7.9000e+09 | pass |
| swiglu_input__f16 | single_stage | 5e8dc11c | 14107 | 0.999647 | 0.988935 | 4.0960e+03 | 2.1750e+09 | pass |
| swiglu_output__fp8__block | single_stage | 5e8dc11c | 14107 | 0.972227 | 0.576540 | 2.7456e+04 | 2.9120e+11 | pass |
| baseline__fp32 | single_stage | 58a34f27 | 11948 | 1.000000 | 1.000000 | 0.0000e+00 | 0.0000e+00 | pass |
| gemm1_accumulator__f16 | single_stage | 58a34f27 | 11948 | 0.998562 | 0.955238 | 4.0960e+03 | 6.7653e+05 | pass |
| gemm1_operands__f16 | single_stage | 58a34f27 | 11948 | 0.999617 | 0.987924 | 4.0960e+03 | 4.3639e+05 | pass |
| gemm1_output__f16 | single_stage | 58a34f27 | 11948 | 0.999731 | 0.991503 | 4.0960e+03 | 5.1967e+04 | pass |
| gemm2_accumulator__bf16 | single_stage | 58a34f27 | 11948 | 0.997076 | 0.901311 | 8.7040e+03 | 1.1207e+06 | pass |
| gemm2_operands__bf16 | single_stage | 58a34f27 | 11948 | 0.997848 | 0.933881 | 8.1920e+03 | 1.4357e+06 | pass |
| hidden_dequant__fp8__block | single_stage | 58a34f27 | 11948 | 0.966563 | 0.634549 | 4.5056e+04 | 2.4789e+07 | pass |
| out_accumulator__bf16 | single_stage | 58a34f27 | 11948 | 0.999510 | 0.984303 | 4.0960e+03 | 1.9711e+04 | pass |
| swiglu_input__f16 | single_stage | 58a34f27 | 11948 | 0.999731 | 0.991503 | 4.0960e+03 | 5.1967e+04 | pass |
| swiglu_output__fp8__block | single_stage | 58a34f27 | 11948 | 0.978726 | 0.675617 | 2.4576e+04 | 1.8592e+07 | pass |
| baseline__fp32 | single_stage | 5eadab1e | 62 | 1.000000 | 1.000000 | 0.0000e+00 | 0.0000e+00 | pass |
| gemm1_accumulator__f16 | single_stage | 5eadab1e | 62 | 0.998571 | 0.951552 | 2.0480e+03 | 1.1925e+02 | pass |
| gemm1_operands__f16 | single_stage | 5eadab1e | 62 | 0.999615 | 0.986967 | 2.0480e+03 | 3.6797e+01 | pass |
| gemm1_output__f16 | single_stage | 5eadab1e | 62 | 0.999748 | 0.991074 | 2.0480e+03 | 3.0096e+01 | pass |
| gemm2_accumulator__bf16 | single_stage | 5eadab1e | 62 | 0.997082 | 0.895391 | 4.0960e+03 | 2.3331e+02 | pass |
| gemm2_operands__bf16 | single_stage | 5eadab1e | 62 | 0.997901 | 0.929091 | 2.0480e+03 | 8.2251e+02 | pass |
| hidden_dequant__fp8__block | single_stage | 5eadab1e | 62 | 0.963836 | 0.608932 | 2.4576e+04 | 4.0256e+03 | pass |
| out_accumulator__bf16 | single_stage | 5eadab1e | 62 | 0.999548 | 0.983842 | 2.0480e+03 | 3.3462e+01 | pass |
| swiglu_input__f16 | single_stage | 5eadab1e | 62 | 0.999748 | 0.991074 | 2.0480e+03 | 3.0096e+01 | pass |
| swiglu_output__fp8__block | single_stage | 5eadab1e | 62 | 0.976956 | 0.652438 | 1.3312e+04 | 2.8683e+03 | pass |
| baseline__fp32 | single_stage | eedc63b2 | 59 | 1.000000 | 1.000000 | 0.0000e+00 | 0.0000e+00 | pass |
| gemm1_accumulator__f16 | single_stage | eedc63b2 | 59 | 0.998796 | 0.960408 | 2.0480e+03 | 3.2758e+03 | pass |
| gemm1_operands__f16 | single_stage | eedc63b2 | 59 | 0.999655 | 0.989062 | 2.0480e+03 | 1.1516e+03 | pass |
| gemm1_output__f16 | single_stage | eedc63b2 | 59 | 0.999754 | 0.992372 | 2.0480e+03 | 8.8516e+01 | pass |
| gemm2_accumulator__bf16 | single_stage | eedc63b2 | 59 | 0.997517 | 0.914441 | 6.1440e+03 | 3.9155e+02 | pass |
| gemm2_operands__bf16 | single_stage | eedc63b2 | 59 | 0.998094 | 0.941402 | 2.0480e+03 | 3.9426e+03 | pass |
| hidden_dequant__fp8__block | single_stage | eedc63b2 | 59 | 0.969972 | 0.674854 | 2.8416e+04 | 2.3165e+04 | pass |
| out_accumulator__bf16 | single_stage | eedc63b2 | 59 | 0.999624 | 0.987851 | 2.0480e+03 | 4.1364e+01 | pass |
| swiglu_input__f16 | single_stage | eedc63b2 | 59 | 0.999754 | 0.992372 | 2.0480e+03 | 8.8516e+01 | pass |
| swiglu_output__fp8__block | single_stage | eedc63b2 | 59 | 0.981032 | 0.712162 | 1.2800e+04 | 4.4504e+04 | pass |
| baseline__fp32 | single_stage | e626d3e6 | 58 | 1.000000 | 1.000000 | 0.0000e+00 | 0.0000e+00 | pass |
| gemm1_accumulator__f16 | single_stage | e626d3e6 | 58 | 0.998071 | 0.939619 | 4.0960e+03 | 3.2059e+01 | pass |
| gemm1_operands__f16 | single_stage | e626d3e6 | 58 | 0.999550 | 0.983769 | 4.0960e+03 | 9.4000e+00 | pass |
| gemm1_output__f16 | single_stage | e626d3e6 | 58 | 0.999682 | 0.988618 | 2.0480e+03 | 1.4378e+01 | pass |
| gemm2_accumulator__bf16 | single_stage | e626d3e6 | 58 | 0.995976 | 0.863580 | 7.1680e+03 | 1.6649e+02 | pass |
| gemm2_operands__bf16 | single_stage | e626d3e6 | 58 | 0.997179 | 0.911289 | 4.0960e+03 | 8.1086e+01 | pass |
| hidden_dequant__fp8__block | single_stage | e626d3e6 | 58 | 0.955085 | 0.511517 | 2.7648e+04 | 1.6962e+03 | pass |
| out_accumulator__bf16 | single_stage | e626d3e6 | 58 | 0.999199 | 0.973842 | 4.0960e+03 | 3.6181e+01 | pass |
| swiglu_input__f16 | single_stage | e626d3e6 | 58 | 0.999682 | 0.988618 | 2.0480e+03 | 1.4378e+01 | pass |
| swiglu_output__fp8__block | single_stage | e626d3e6 | 58 | 0.971610 | 0.565038 | 1.7920e+04 | 1.1620e+03 | pass |
| baseline__fp32 | single_stage | 74d7ff04 | 57 | 1.000000 | 1.000000 | 0.0000e+00 | 0.0000e+00 | pass |
| gemm1_accumulator__f16 | single_stage | 74d7ff04 | 57 | 0.998274 | 0.947312 | 2.0480e+03 | 7.3962e+02 | pass |
| gemm1_operands__f16 | single_stage | 74d7ff04 | 57 | 0.999552 | 0.986005 | 2.0480e+03 | 3.3054e+02 | pass |
| gemm1_output__f16 | single_stage | 74d7ff04 | 57 | 0.999692 | 0.990469 | 2.0480e+03 | 2.5411e+01 | pass |
| gemm2_accumulator__bf16 | single_stage | 74d7ff04 | 57 | 0.996659 | 0.888510 | 4.7360e+03 | 1.5118e+03 | pass |
| gemm2_operands__bf16 | single_stage | 74d7ff04 | 57 | 0.997570 | 0.926139 | 4.0960e+03 | 1.0362e+03 | pass |
| hidden_dequant__fp8__block | single_stage | 74d7ff04 | 57 | 0.961799 | 0.589220 | 2.2528e+04 | 1.9351e+03 | pass |
| out_accumulator__bf16 | single_stage | 74d7ff04 | 57 | 0.999383 | 0.981161 | 4.0960e+03 | 2.1995e+02 | pass |
| swiglu_input__f16 | single_stage | 74d7ff04 | 57 | 0.999692 | 0.990469 | 2.0480e+03 | 2.5411e+01 | pass |
| swiglu_output__fp8__block | single_stage | 74d7ff04 | 57 | 0.975970 | 0.635914 | 1.5360e+04 | 9.4464e+03 | pass |
| baseline__fp32 | single_stage | 4822167c | 56 | 1.000000 | 1.000000 | 0.0000e+00 | 0.0000e+00 | pass |
| gemm1_accumulator__f16 | single_stage | 4822167c | 56 | 0.998144 | 0.938673 | 2.0480e+03 | 4.3752e+01 | pass |
| gemm1_operands__f16 | single_stage | 4822167c | 56 | 0.999492 | 0.983371 | 2.0480e+03 | 1.4200e+01 | pass |
| gemm1_output__f16 | single_stage | 4822167c | 56 | 0.999636 | 0.988326 | 2.0480e+03 | 2.5510e+01 | pass |
| gemm2_accumulator__bf16 | single_stage | 4822167c | 56 | 0.995715 | 0.861393 | 4.0960e+03 | 2.4139e+02 | pass |
| gemm2_operands__bf16 | single_stage | 4822167c | 56 | 0.997035 | 0.908345 | 2.0480e+03 | 4.0735e+02 | pass |
| hidden_dequant__fp8__block | single_stage | 4822167c | 56 | 0.953576 | 0.494176 | 2.0480e+04 | 3.2084e+03 | pass |
| out_accumulator__bf16 | single_stage | 4822167c | 56 | 0.999270 | 0.975481 | 2.0480e+03 | 7.7561e+01 | pass |
| swiglu_input__f16 | single_stage | 4822167c | 56 | 0.999636 | 0.988326 | 2.0480e+03 | 2.5510e+01 | pass |
| swiglu_output__fp8__block | single_stage | 4822167c | 56 | 0.970693 | 0.550649 | 1.3312e+04 | 1.5485e+03 | pass |
| baseline__fp32 | single_stage | 81955b1e | 55 | 1.000000 | 1.000000 | 0.0000e+00 | 0.0000e+00 | pass |
| gemm1_accumulator__f16 | single_stage | 81955b1e | 55 | 0.998448 | 0.952242 | 4.0960e+03 | 4.9794e+02 | pass |
| gemm1_operands__f16 | single_stage | 81955b1e | 55 | 0.999584 | 0.986889 | 4.0960e+03 | 3.8124e+02 | pass |
| gemm1_output__f16 | single_stage | 81955b1e | 55 | 0.999708 | 0.990828 | 2.0480e+03 | 2.5888e+02 | pass |
| gemm2_accumulator__bf16 | single_stage | 81955b1e | 55 | 0.996850 | 0.895072 | 4.0960e+03 | 8.1041e+02 | pass |
| gemm2_operands__bf16 | single_stage | 81955b1e | 55 | 0.997722 | 0.928868 | 2.0480e+03 | 5.1476e+02 | pass |
| hidden_dequant__fp8__block | single_stage | 81955b1e | 55 | 0.963938 | 0.606585 | 2.1504e+04 | 3.4215e+04 | pass |
| out_accumulator__bf16 | single_stage | 81955b1e | 55 | 0.999465 | 0.982120 | 4.0960e+03 | 5.8183e+01 | pass |
| swiglu_input__f16 | single_stage | 81955b1e | 55 | 0.999708 | 0.990828 | 2.0480e+03 | 2.5888e+02 | pass |
| swiglu_output__fp8__block | single_stage | 81955b1e | 55 | 0.977189 | 0.650525 | 1.4336e+04 | 4.7576e+03 | pass |
| baseline__fp32 | single_stage | 76010cb4 | 54 | 1.000000 | 1.000000 | 0.0000e+00 | 0.0000e+00 | pass |
| gemm1_accumulator__f16 | single_stage | 76010cb4 | 54 | 0.998675 | 0.959726 | 2.0480e+03 | 5.7615e+01 | pass |
| gemm1_operands__f16 | single_stage | 76010cb4 | 54 | 0.999641 | 0.989129 | 2.0480e+03 | 5.0477e+01 | pass |
| gemm1_output__f16 | single_stage | 76010cb4 | 54 | 0.999731 | 0.992503 | 2.0480e+03 | 1.6108e+01 | pass |
| gemm2_accumulator__bf16 | single_stage | 76010cb4 | 54 | 0.997336 | 0.912148 | 6.1440e+03 | 7.2368e+02 | pass |
| gemm2_operands__bf16 | single_stage | 76010cb4 | 54 | 0.998065 | 0.941179 | 2.0480e+03 | 1.3491e+02 | pass |
| hidden_dequant__fp8__block | single_stage | 76010cb4 | 54 | 0.969683 | 0.675197 | 2.0864e+04 | 1.9919e+03 | pass |
| out_accumulator__bf16 | single_stage | 76010cb4 | 54 | 0.999569 | 0.985385 | 2.0480e+03 | 3.7923e+01 | pass |
| swiglu_input__f16 | single_stage | 76010cb4 | 54 | 0.999731 | 0.992503 | 2.0480e+03 | 1.6108e+01 | pass |
| swiglu_output__fp8__block | single_stage | 76010cb4 | 54 | 0.981047 | 0.712707 | 1.2242e+04 | 2.0628e+03 | pass |
| baseline__fp32 | single_stage | fc378037 | 53 | 1.000000 | 1.000000 | 0.0000e+00 | 0.0000e+00 | pass |
| gemm1_accumulator__f16 | single_stage | fc378037 | 53 | 0.998489 | 0.954128 | 4.0960e+03 | 1.5543e+03 | pass |
| gemm1_operands__f16 | single_stage | fc378037 | 53 | 0.999624 | 0.987313 | 2.0480e+03 | 5.7247e+01 | pass |
| gemm1_output__f16 | single_stage | fc378037 | 53 | 0.999703 | 0.991219 | 2.0480e+03 | 1.1606e+02 | pass |
| gemm2_accumulator__bf16 | single_stage | fc378037 | 53 | 0.996876 | 0.896321 | 6.1440e+03 | 4.9305e+02 | pass |
| gemm2_operands__bf16 | single_stage | fc378037 | 53 | 0.997694 | 0.931225 | 4.0960e+03 | 9.5021e+02 | pass |
| hidden_dequant__fp8__block | single_stage | fc378037 | 53 | 0.965162 | 0.622183 | 2.6112e+04 | 1.4707e+03 | pass |
| out_accumulator__bf16 | single_stage | fc378037 | 53 | 0.999413 | 0.981637 | 4.0960e+03 | 1.3491e+02 | pass |
| swiglu_input__f16 | single_stage | fc378037 | 53 | 0.999703 | 0.991219 | 2.0480e+03 | 1.1606e+02 | pass |
| swiglu_output__fp8__block | single_stage | fc378037 | 53 | 0.978063 | 0.664068 | 1.8432e+04 | 4.7885e+03 | pass |
| baseline__fp32 | single_stage | f7d6ac7c | 52 | 1.000000 | 1.000000 | 0.0000e+00 | 0.0000e+00 | pass |
| gemm1_accumulator__f16 | single_stage | f7d6ac7c | 52 | 0.999107 | 0.970523 | 2.0480e+03 | 1.1134e+02 | pass |
| gemm1_operands__f16 | single_stage | f7d6ac7c | 52 | 0.999753 | 0.991992 | 2.0480e+03 | 1.1390e+01 | pass |
| gemm1_output__f16 | single_stage | f7d6ac7c | 52 | 0.999836 | 0.994505 | 2.0480e+03 | 1.3184e+01 | pass |
| gemm2_accumulator__bf16 | single_stage | f7d6ac7c | 52 | 0.998253 | 0.937347 | 4.0960e+03 | 3.5843e+02 | pass |
| gemm2_operands__bf16 | single_stage | f7d6ac7c | 52 | 0.998578 | 0.956277 | 2.0480e+03 | 1.0189e+02 | pass |
| hidden_dequant__fp8__block | single_stage | f7d6ac7c | 52 | 0.978419 | 0.760707 | 1.9456e+04 | 1.8441e+03 | pass |
| out_accumulator__bf16 | single_stage | f7d6ac7c | 52 | 0.999761 | 0.991670 | 2.0480e+03 | 6.6509e+01 | pass |
| swiglu_input__f16 | single_stage | f7d6ac7c | 52 | 0.999836 | 0.994505 | 2.0480e+03 | 1.3184e+01 | pass |
| swiglu_output__fp8__block | single_stage | f7d6ac7c | 52 | 0.986025 | 0.787665 | 1.1264e+04 | 1.0138e+03 | pass |
| gemm1_accumulator__f16 | cumulative | b8f4f012 | 7 | 0.999143 | 0.971720 | 4.0960e+03 | 5.3765e+00 | pass |
| gemm1_accumulator__f16 | cumulative | e05c6c03 | 1 | 0.996931 | 0.903739 | 2.0480e+03 | 2.3422e+00 | pass |
| gemm1_accumulator__f16 | cumulative | 6230e838 | 32 | 0.998544 | 0.958335 | 2.0480e+03 | 7.9108e+01 | pass |
| gemm1_accumulator__f16 | cumulative | 8f1ff9f1 | 80 | 0.998000 | 0.939645 | 4.0960e+03 | 6.9622e+03 | pass |
| gemm1_accumulator__f16 | cumulative | 1a4c6ba1 | 901 | 0.997949 | 0.935502 | 4.0960e+03 | 2.5984e+03 | pass |
| gemm1_accumulator__f16 | cumulative | a7c2bcfd | 16 | 0.997899 | 0.935407 | 2.0480e+03 | 1.3403e+02 | pass |
| gemm1_accumulator__f16 | cumulative | 2e69caee | 15 | 0.999219 | 0.971298 | 4.0960e+03 | 4.9854e+01 | pass |
| gemm1_accumulator__f16 | cumulative | 8cba5890 | 14 | 0.998565 | 0.954809 | 2.0480e+03 | 4.5596e+01 | pass |
| gemm1_accumulator__f16 | cumulative | 5e8dc11c | 14107 | 0.998126 | 0.941529 | 4.0960e+03 | 5.2481e+04 | pass |
| gemm1_accumulator__f16 | cumulative | 58a34f27 | 11948 | 0.998559 | 0.955332 | 4.0960e+03 | 6.7500e+09 | pass |
| gemm1_accumulator__f16 | cumulative | 5eadab1e | 62 | 0.998515 | 0.953125 | 4.0960e+03 | 4.7000e+01 | pass |
| gemm1_accumulator__f16 | cumulative | eedc63b2 | 59 | 0.998685 | 0.959169 | 2.0480e+03 | 3.1518e+02 | pass |
| gemm1_accumulator__f16 | cumulative | e626d3e6 | 58 | 0.998081 | 0.939884 | 4.0960e+03 | 2.0470e+03 | pass |
| gemm1_accumulator__f16 | cumulative | 74d7ff04 | 57 | 0.998566 | 0.949855 | 4.0960e+03 | 2.0608e+03 | pass |
| gemm1_accumulator__f16 | cumulative | 4822167c | 56 | 0.997907 | 0.938240 | 4.0960e+03 | 3.4500e+02 | pass |
| gemm1_accumulator__f16 | cumulative | 81955b1e | 55 | 0.998407 | 0.951565 | 4.0960e+03 | 3.9959e+02 | pass |
| gemm1_accumulator__f16 | cumulative | 76010cb4 | 54 | 0.998701 | 0.959289 | 4.0960e+03 | 1.5500e+02 | pass |
| gemm1_accumulator__f16 | cumulative | fc378037 | 53 | 0.998510 | 0.955047 | 4.0960e+03 | 1.9015e+02 | pass |
| gemm1_accumulator__f16 | cumulative | f7d6ac7c | 52 | 0.999029 | 0.971741 | 2.0480e+03 | 5.7182e+01 | pass |
| gemm1_accumulator__f16+gemm1_operands__f16 | cumulative | b8f4f012 | 7 | 0.999641 | 0.991629 | 2.0480e+03 | 3.1395e+01 | pass |
| gemm1_accumulator__f16+gemm1_operands__f16 | cumulative | e05c6c03 | 1 | 0.999442 | 0.973075 | 2.0480e+03 | 1.1610e+00 | pass |
| gemm1_accumulator__f16+gemm1_operands__f16 | cumulative | 6230e838 | 32 | 0.999656 | 0.988848 | 2.0480e+03 | 2.8745e+01 | pass |
| gemm1_accumulator__f16+gemm1_operands__f16 | cumulative | 8f1ff9f1 | 80 | 0.999404 | 0.983578 | 4.0960e+03 | 2.0645e+02 | pass |
| gemm1_accumulator__f16+gemm1_operands__f16 | cumulative | 1a4c6ba1 | 901 | 0.999437 | 0.982604 | 4.0960e+03 | 1.6863e+03 | pass |
| gemm1_accumulator__f16+gemm1_operands__f16 | cumulative | a7c2bcfd | 16 | 0.999372 | 0.982910 | 2.0480e+03 | 2.0920e+01 | pass |
| gemm1_accumulator__f16+gemm1_operands__f16 | cumulative | 2e69caee | 15 | 0.999842 | 0.993015 | 4.0960e+03 | 5.6400e+00 | pass |
| gemm1_accumulator__f16+gemm1_operands__f16 | cumulative | 8cba5890 | 14 | 0.999542 | 0.988710 | 2.0480e+03 | 1.4479e+01 | pass |
| gemm1_accumulator__f16+gemm1_operands__f16 | cumulative | 5e8dc11c | 14107 | 0.999499 | 0.984253 | 4.0960e+03 | 2.1739e+04 | pass |
| gemm1_accumulator__f16+gemm1_operands__f16 | cumulative | 58a34f27 | 11948 | 0.999617 | 0.987910 | 4.0960e+03 | 7.6508e+03 | pass |
| gemm1_accumulator__f16+gemm1_operands__f16 | cumulative | 5eadab1e | 62 | 0.999620 | 0.987154 | 2.0480e+03 | 1.4652e+02 | pass |
| gemm1_accumulator__f16+gemm1_operands__f16 | cumulative | eedc63b2 | 59 | 0.999719 | 0.989270 | 2.0480e+03 | 7.6740e+01 | pass |
| gemm1_accumulator__f16+gemm1_operands__f16 | cumulative | e626d3e6 | 58 | 0.999536 | 0.983680 | 4.0960e+03 | 2.4874e+02 | pass |
| gemm1_accumulator__f16+gemm1_operands__f16 | cumulative | 74d7ff04 | 57 | 0.999542 | 0.986600 | 2.0480e+03 | 5.7180e+02 | pass |
| gemm1_accumulator__f16+gemm1_operands__f16 | cumulative | 4822167c | 56 | 0.999452 | 0.983060 | 2.0480e+03 | 7.2650e+03 | pass |
| gemm1_accumulator__f16+gemm1_operands__f16 | cumulative | 81955b1e | 55 | 0.999566 | 0.986846 | 2.0480e+03 | 2.1898e+01 | pass |
| gemm1_accumulator__f16+gemm1_operands__f16 | cumulative | 76010cb4 | 54 | 0.999628 | 0.989297 | 2.0480e+03 | 1.2120e+01 | pass |
| gemm1_accumulator__f16+gemm1_operands__f16 | cumulative | fc378037 | 53 | 0.999605 | 0.987747 | 2.0480e+03 | 2.6141e+01 | pass |
| gemm1_accumulator__f16+gemm1_operands__f16 | cumulative | f7d6ac7c | 52 | 0.999761 | 0.991941 | 2.0480e+03 | 2.8576e+01 | pass |
| gemm1_accumulator__f16+gemm1_operands__f16+gemm1_output__f16 | cumulative | b8f4f012 | 7 | 0.999721 | 0.990633 | 2.0480e+03 | 2.7663e+01 | pass |
| gemm1_accumulator__f16+gemm1_operands__f16+gemm1_output__f16 | cumulative | e05c6c03 | 1 | 0.999442 | 0.964844 | 1.0240e+03 | 6.4402e-01 | pass |
| gemm1_accumulator__f16+gemm1_operands__f16+gemm1_output__f16 | cumulative | 6230e838 | 32 | 0.999599 | 0.986210 | 2.0480e+03 | 7.5936e+01 | pass |
| gemm1_accumulator__f16+gemm1_operands__f16+gemm1_output__f16 | cumulative | 8f1ff9f1 | 80 | 0.999370 | 0.979832 | 2.0480e+03 | 9.7987e+01 | pass |
| gemm1_accumulator__f16+gemm1_operands__f16+gemm1_output__f16 | cumulative | 1a4c6ba1 | 901 | 0.999310 | 0.978679 | 4.0960e+03 | 1.0390e+03 | pass |
| gemm1_accumulator__f16+gemm1_operands__f16+gemm1_output__f16 | cumulative | a7c2bcfd | 16 | 0.999329 | 0.978123 | 2.0480e+03 | 1.0699e+02 | pass |
| gemm1_accumulator__f16+gemm1_operands__f16+gemm1_output__f16 | cumulative | 2e69caee | 15 | 0.999767 | 0.991109 | 2.0480e+03 | 3.4224e+00 | pass |
| gemm1_accumulator__f16+gemm1_operands__f16+gemm1_output__f16 | cumulative | 8cba5890 | 14 | 0.999601 | 0.985461 | 2.0480e+03 | 2.4585e+01 | pass |
| gemm1_accumulator__f16+gemm1_operands__f16+gemm1_output__f16 | cumulative | 5e8dc11c | 14107 | 0.999385 | 0.980716 | 4.0960e+03 | 1.8242e+05 | pass |
| gemm1_accumulator__f16+gemm1_operands__f16+gemm1_output__f16 | cumulative | 58a34f27 | 11948 | 0.999531 | 0.985249 | 4.0960e+03 | 3.5428e+04 | pass |
| gemm1_accumulator__f16+gemm1_operands__f16+gemm1_output__f16 | cumulative | 5eadab1e | 62 | 0.999507 | 0.984551 | 2.0480e+03 | 6.8368e+01 | pass |
| gemm1_accumulator__f16+gemm1_operands__f16+gemm1_output__f16 | cumulative | eedc63b2 | 59 | 0.999577 | 0.987026 | 2.0480e+03 | 1.6223e+03 | pass |
| gemm1_accumulator__f16+gemm1_operands__f16+gemm1_output__f16 | cumulative | e626d3e6 | 58 | 0.999382 | 0.980435 | 2.0480e+03 | 3.1000e+01 | pass |
| gemm1_accumulator__f16+gemm1_operands__f16+gemm1_output__f16 | cumulative | 74d7ff04 | 57 | 0.999501 | 0.983558 | 2.0480e+03 | 3.0618e+02 | pass |
| gemm1_accumulator__f16+gemm1_operands__f16+gemm1_output__f16 | cumulative | 4822167c | 56 | 0.999375 | 0.979928 | 2.0480e+03 | 1.6658e+02 | pass |
| gemm1_accumulator__f16+gemm1_operands__f16+gemm1_output__f16 | cumulative | 81955b1e | 55 | 0.999465 | 0.984466 | 2.0480e+03 | 4.0639e+01 | pass |
| gemm1_accumulator__f16+gemm1_operands__f16+gemm1_output__f16 | cumulative | 76010cb4 | 54 | 0.999548 | 0.986990 | 2.0480e+03 | 2.0353e+04 | pass |
| gemm1_accumulator__f16+gemm1_operands__f16+gemm1_output__f16 | cumulative | fc378037 | 53 | 0.999516 | 0.984854 | 2.0480e+03 | 1.8419e+02 | pass |
| gemm1_accumulator__f16+gemm1_operands__f16+gemm1_output__f16 | cumulative | f7d6ac7c | 52 | 0.999657 | 0.990417 | 2.0480e+03 | 1.4594e+02 | pass |
| gemm1_accumulator__f16+gemm1_operands__f16+gemm1_output__f16+gemm2_accumulator__bf16 | cumulative | b8f4f012 | 7 | 0.997927 | 0.927097 | 4.0960e+03 | 5.6208e+01 | pass |
| gemm1_accumulator__f16+gemm1_operands__f16+gemm1_output__f16+gemm2_accumulator__bf16 | cumulative | e05c6c03 | 1 | 0.992885 | 0.747768 | 3.0720e+03 | 2.7776e+01 | pass |
| gemm1_accumulator__f16+gemm1_operands__f16+gemm1_output__f16+gemm2_accumulator__bf16 | cumulative | 6230e838 | 32 | 0.996765 | 0.898690 | 4.0960e+03 | 1.0331e+03 | pass |
| gemm1_accumulator__f16+gemm1_operands__f16+gemm1_output__f16+gemm2_accumulator__bf16 | cumulative | 8f1ff9f1 | 80 | 0.995818 | 0.861239 | 8.1920e+03 | 3.9273e+03 | pass |
| gemm1_accumulator__f16+gemm1_operands__f16+gemm1_output__f16+gemm2_accumulator__bf16 | cumulative | 1a4c6ba1 | 901 | 0.995399 | 0.851060 | 8.1920e+03 | 4.1814e+04 | pass |
| gemm1_accumulator__f16+gemm1_operands__f16+gemm1_output__f16+gemm2_accumulator__bf16 | cumulative | a7c2bcfd | 16 | 0.995989 | 0.859968 | 4.0960e+03 | 5.7388e+02 | pass |
| gemm1_accumulator__f16+gemm1_operands__f16+gemm1_output__f16+gemm2_accumulator__bf16 | cumulative | 2e69caee | 15 | 0.998112 | 0.936988 | 4.0960e+03 | 4.5911e+01 | pass |
| gemm1_accumulator__f16+gemm1_operands__f16+gemm1_output__f16+gemm2_accumulator__bf16 | cumulative | 8cba5890 | 14 | 0.996921 | 0.897421 | 4.0960e+03 | 1.8743e+02 | pass |
| gemm1_accumulator__f16+gemm1_operands__f16+gemm1_output__f16+gemm2_accumulator__bf16 | cumulative | 5e8dc11c | 14107 | 0.996129 | 0.869934 | 1.2288e+04 | 1.4177e+05 | pass |
| gemm1_accumulator__f16+gemm1_operands__f16+gemm1_output__f16+gemm2_accumulator__bf16 | cumulative | 58a34f27 | 11948 | 0.997047 | 0.900624 | 1.2288e+04 | 4.8129e+04 | pass |
| gemm1_accumulator__f16+gemm1_operands__f16+gemm1_output__f16+gemm2_accumulator__bf16 | cumulative | 5eadab1e | 62 | 0.996848 | 0.894288 | 4.6080e+03 | 1.0277e+02 | pass |
| gemm1_accumulator__f16+gemm1_operands__f16+gemm1_output__f16+gemm2_accumulator__bf16 | cumulative | eedc63b2 | 59 | 0.997323 | 0.913306 | 6.1440e+03 | 4.3885e+02 | pass |
| gemm1_accumulator__f16+gemm1_operands__f16+gemm1_output__f16+gemm2_accumulator__bf16 | cumulative | e626d3e6 | 58 | 0.995668 | 0.861059 | 6.1440e+03 | 1.7614e+04 | pass |
| gemm1_accumulator__f16+gemm1_operands__f16+gemm1_output__f16+gemm2_accumulator__bf16 | cumulative | 74d7ff04 | 57 | 0.996593 | 0.886885 | 4.0960e+03 | 8.3461e+02 | pass |
| gemm1_accumulator__f16+gemm1_operands__f16+gemm1_output__f16+gemm2_accumulator__bf16 | cumulative | 4822167c | 56 | 0.995730 | 0.858775 | 4.0960e+03 | 4.5789e+02 | pass |
| gemm1_accumulator__f16+gemm1_operands__f16+gemm1_output__f16+gemm2_accumulator__bf16 | cumulative | 81955b1e | 55 | 0.996690 | 0.892000 | 5.1200e+03 | 1.7883e+02 | pass |
| gemm1_accumulator__f16+gemm1_operands__f16+gemm1_output__f16+gemm2_accumulator__bf16 | cumulative | 76010cb4 | 54 | 0.997388 | 0.912272 | 6.1440e+03 | 1.2648e+02 | pass |
| gemm1_accumulator__f16+gemm1_operands__f16+gemm1_output__f16+gemm2_accumulator__bf16 | cumulative | fc378037 | 53 | 0.996812 | 0.894823 | 6.1440e+03 | 1.3526e+02 | pass |
| gemm1_accumulator__f16+gemm1_operands__f16+gemm1_output__f16+gemm2_accumulator__bf16 | cumulative | f7d6ac7c | 52 | 0.998039 | 0.936553 | 4.0960e+03 | 2.3102e+02 | pass |
| gemm1_accumulator__f16+gemm1_operands__f16+gemm1_output__f16+gemm2_accumulator__bf16+gemm2_operands__bf16 | cumulative | b8f4f012 | 7 | 0.998744 | 0.957151 | 2.0480e+03 | 1.5127e+03 | pass |
| gemm1_accumulator__f16+gemm1_operands__f16+gemm1_output__f16+gemm2_accumulator__bf16+gemm2_operands__bf16 | cumulative | e05c6c03 | 1 | 0.994978 | 0.840541 | 2.0480e+03 | 3.2022e+00 | pass |
| gemm1_accumulator__f16+gemm1_operands__f16+gemm1_output__f16+gemm2_accumulator__bf16+gemm2_operands__bf16 | cumulative | 6230e838 | 32 | 0.997750 | 0.937587 | 4.0960e+03 | 3.3670e+02 | pass |
| gemm1_accumulator__f16+gemm1_operands__f16+gemm1_output__f16+gemm2_accumulator__bf16+gemm2_operands__bf16 | cumulative | 8f1ff9f1 | 80 | 0.996987 | 0.906975 | 4.0960e+03 | 6.7959e+02 | pass |
| gemm1_accumulator__f16+gemm1_operands__f16+gemm1_output__f16+gemm2_accumulator__bf16+gemm2_operands__bf16 | cumulative | 1a4c6ba1 | 901 | 0.996789 | 0.902125 | 4.0960e+03 | 3.7210e+03 | pass |
| gemm1_accumulator__f16+gemm1_operands__f16+gemm1_output__f16+gemm2_accumulator__bf16+gemm2_operands__bf16 | cumulative | a7c2bcfd | 16 | 0.996791 | 0.902605 | 2.0480e+03 | 3.5118e+01 | pass |
| gemm1_accumulator__f16+gemm1_operands__f16+gemm1_output__f16+gemm2_accumulator__bf16+gemm2_operands__bf16 | cumulative | 2e69caee | 15 | 0.998484 | 0.959040 | 2.0480e+03 | 9.5140e+01 | pass |
| gemm1_accumulator__f16+gemm1_operands__f16+gemm1_output__f16+gemm2_accumulator__bf16+gemm2_operands__bf16 | cumulative | 8cba5890 | 14 | 0.997579 | 0.933225 | 2.0480e+03 | 5.2241e+01 | pass |
| gemm1_accumulator__f16+gemm1_operands__f16+gemm1_output__f16+gemm2_accumulator__bf16+gemm2_operands__bf16 | cumulative | 5e8dc11c | 14107 | 0.997130 | 0.911782 | 4.0960e+03 | 2.3961e+05 | pass |
| gemm1_accumulator__f16+gemm1_operands__f16+gemm1_output__f16+gemm2_accumulator__bf16+gemm2_operands__bf16 | cumulative | 58a34f27 | 11948 | 0.997785 | 0.932220 | 4.0960e+03 | 3.8000e+09 | pass |
| gemm1_accumulator__f16+gemm1_operands__f16+gemm1_output__f16+gemm2_accumulator__bf16+gemm2_operands__bf16 | cumulative | 5eadab1e | 62 | 0.997784 | 0.927440 | 2.0480e+03 | 9.7536e+01 | pass |
| gemm1_accumulator__f16+gemm1_operands__f16+gemm1_output__f16+gemm2_accumulator__bf16+gemm2_operands__bf16 | cumulative | eedc63b2 | 59 | 0.998033 | 0.940125 | 2.0480e+03 | 7.5376e+02 | pass |
| gemm1_accumulator__f16+gemm1_operands__f16+gemm1_output__f16+gemm2_accumulator__bf16+gemm2_operands__bf16 | cumulative | e626d3e6 | 58 | 0.997171 | 0.909836 | 4.0960e+03 | 3.1563e+02 | pass |
| gemm1_accumulator__f16+gemm1_operands__f16+gemm1_output__f16+gemm2_accumulator__bf16+gemm2_operands__bf16 | cumulative | 74d7ff04 | 57 | 0.997459 | 0.923400 | 4.0960e+03 | 2.3106e+02 | pass |
| gemm1_accumulator__f16+gemm1_operands__f16+gemm1_output__f16+gemm2_accumulator__bf16+gemm2_operands__bf16 | cumulative | 4822167c | 56 | 0.996953 | 0.906434 | 2.0480e+03 | 2.5500e+02 | pass |
| gemm1_accumulator__f16+gemm1_operands__f16+gemm1_output__f16+gemm2_accumulator__bf16+gemm2_operands__bf16 | cumulative | 81955b1e | 55 | 0.997717 | 0.927445 | 4.0960e+03 | 1.1268e+03 | pass |
| gemm1_accumulator__f16+gemm1_operands__f16+gemm1_output__f16+gemm2_accumulator__bf16+gemm2_operands__bf16 | cumulative | 76010cb4 | 54 | 0.997985 | 0.939432 | 4.0960e+03 | 1.5179e+02 | pass |
| gemm1_accumulator__f16+gemm1_operands__f16+gemm1_output__f16+gemm2_accumulator__bf16+gemm2_operands__bf16 | cumulative | fc378037 | 53 | 0.997728 | 0.930388 | 4.0960e+03 | 7.4707e+01 | pass |
| gemm1_accumulator__f16+gemm1_operands__f16+gemm1_output__f16+gemm2_accumulator__bf16+gemm2_operands__bf16 | cumulative | f7d6ac7c | 52 | 0.998519 | 0.955515 | 2.0480e+03 | 9.6706e+01 | pass |
| gemm1_accumulator__f16+gemm1_operands__f16+gemm1_output__f16+gemm2_accumulator__bf16+gemm2_operands__bf16+hidden_dequant__fp8__block | cumulative | b8f4f012 | 7 | 0.977519 | 0.758430 | 2.1376e+04 | 1.5527e+03 | pass |
| gemm1_accumulator__f16+gemm1_operands__f16+gemm1_output__f16+gemm2_accumulator__bf16+gemm2_operands__bf16+hidden_dequant__fp8__block | cumulative | e05c6c03 | 1 | 0.923549 | 0.162946 | 1.3824e+04 | 3.6906e+01 | pass |
| gemm1_accumulator__f16+gemm1_operands__f16+gemm1_output__f16+gemm2_accumulator__bf16+gemm2_operands__bf16+hidden_dequant__fp8__block | cumulative | 6230e838 | 32 | 0.968951 | 0.660331 | 3.2768e+04 | 1.5106e+03 | pass |
| gemm1_accumulator__f16+gemm1_operands__f16+gemm1_output__f16+gemm2_accumulator__bf16+gemm2_operands__bf16+hidden_dequant__fp8__block | cumulative | 8f1ff9f1 | 80 | 0.953934 | 0.498650 | 2.3424e+04 | 3.5302e+05 | pass |
| gemm1_accumulator__f16+gemm1_operands__f16+gemm1_output__f16+gemm2_accumulator__bf16+gemm2_operands__bf16+hidden_dequant__fp8__block | cumulative | 1a4c6ba1 | 901 | 0.951803 | 0.474065 | 2.9696e+04 | 5.9528e+05 | pass |
| gemm1_accumulator__f16+gemm1_operands__f16+gemm1_output__f16+gemm2_accumulator__bf16+gemm2_operands__bf16+hidden_dequant__fp8__block | cumulative | a7c2bcfd | 16 | 0.952340 | 0.477347 | 2.2528e+04 | 5.7457e+03 | pass |
| gemm1_accumulator__f16+gemm1_operands__f16+gemm1_output__f16+gemm2_accumulator__bf16+gemm2_operands__bf16+hidden_dequant__fp8__block | cumulative | 2e69caee | 15 | 0.980599 | 0.778451 | 2.1504e+04 | 1.6385e+04 | pass |
| gemm1_accumulator__f16+gemm1_operands__f16+gemm1_output__f16+gemm2_accumulator__bf16+gemm2_operands__bf16+hidden_dequant__fp8__block | cumulative | 8cba5890 | 14 | 0.968003 | 0.644252 | 2.1504e+04 | 7.9494e+02 | pass |
| gemm1_accumulator__f16+gemm1_operands__f16+gemm1_output__f16+gemm2_accumulator__bf16+gemm2_operands__bf16+hidden_dequant__fp8__block | cumulative | 5e8dc11c | 14107 | 0.956221 | 0.522753 | 4.5056e+04 | 7.6256e+06 | pass |
| gemm1_accumulator__f16+gemm1_operands__f16+gemm1_output__f16+gemm2_accumulator__bf16+gemm2_operands__bf16+hidden_dequant__fp8__block | cumulative | 58a34f27 | 11948 | 0.966437 | 0.634270 | 4.0960e+04 | 2.9440e+11 | pass |
| gemm1_accumulator__f16+gemm1_operands__f16+gemm1_output__f16+gemm2_accumulator__bf16+gemm2_operands__bf16+hidden_dequant__fp8__block | cumulative | 5eadab1e | 62 | 0.964472 | 0.610277 | 2.1504e+04 | 8.8212e+03 | pass |
| gemm1_accumulator__f16+gemm1_operands__f16+gemm1_output__f16+gemm2_accumulator__bf16+gemm2_operands__bf16+hidden_dequant__fp8__block | cumulative | eedc63b2 | 59 | 0.970750 | 0.675188 | 1.9072e+04 | 3.3647e+03 | pass |
| gemm1_accumulator__f16+gemm1_operands__f16+gemm1_output__f16+gemm2_accumulator__bf16+gemm2_operands__bf16+hidden_dequant__fp8__block | cumulative | e626d3e6 | 58 | 0.955571 | 0.512051 | 2.7648e+04 | 6.6395e+03 | pass |
| gemm1_accumulator__f16+gemm1_operands__f16+gemm1_output__f16+gemm2_accumulator__bf16+gemm2_operands__bf16+hidden_dequant__fp8__block | cumulative | 74d7ff04 | 57 | 0.962098 | 0.590277 | 2.8672e+04 | 1.0389e+04 | pass |
| gemm1_accumulator__f16+gemm1_operands__f16+gemm1_output__f16+gemm2_accumulator__bf16+gemm2_operands__bf16+hidden_dequant__fp8__block | cumulative | 4822167c | 56 | 0.953454 | 0.493331 | 2.2528e+04 | 9.2069e+03 | pass |
| gemm1_accumulator__f16+gemm1_operands__f16+gemm1_output__f16+gemm2_accumulator__bf16+gemm2_operands__bf16+hidden_dequant__fp8__block | cumulative | 81955b1e | 55 | 0.963654 | 0.605190 | 2.9696e+04 | 1.3914e+04 | pass |
| gemm1_accumulator__f16+gemm1_operands__f16+gemm1_output__f16+gemm2_accumulator__bf16+gemm2_operands__bf16+hidden_dequant__fp8__block | cumulative | 76010cb4 | 54 | 0.969910 | 0.674838 | 3.4816e+04 | 2.3008e+04 | pass |
| gemm1_accumulator__f16+gemm1_operands__f16+gemm1_output__f16+gemm2_accumulator__bf16+gemm2_operands__bf16+hidden_dequant__fp8__block | cumulative | fc378037 | 53 | 0.965604 | 0.622241 | 2.2528e+04 | 7.3485e+03 | pass |
| gemm1_accumulator__f16+gemm1_operands__f16+gemm1_output__f16+gemm2_accumulator__bf16+gemm2_operands__bf16+hidden_dequant__fp8__block | cumulative | f7d6ac7c | 52 | 0.977383 | 0.758572 | 2.0736e+04 | 1.0967e+03 | pass |
| gemm1_accumulator__f16+gemm1_operands__f16+gemm1_output__f16+gemm2_accumulator__bf16+gemm2_operands__bf16+hidden_dequant__fp8__block+out_accumulator__bf16 | cumulative | b8f4f012 | 7 | 0.978635 | 0.759546 | 2.0480e+04 | 2.3012e+02 | pass |
| gemm1_accumulator__f16+gemm1_operands__f16+gemm1_output__f16+gemm2_accumulator__bf16+gemm2_operands__bf16+hidden_dequant__fp8__block+out_accumulator__bf16 | cumulative | e05c6c03 | 1 | 0.925781 | 0.160854 | 1.1264e+04 | 4.5700e+02 | pass |
| gemm1_accumulator__f16+gemm1_operands__f16+gemm1_output__f16+gemm2_accumulator__bf16+gemm2_operands__bf16+hidden_dequant__fp8__block+out_accumulator__bf16 | cumulative | 6230e838 | 32 | 0.969526 | 0.660579 | 3.0720e+04 | 4.8036e+03 | pass |
| gemm1_accumulator__f16+gemm1_operands__f16+gemm1_output__f16+gemm2_accumulator__bf16+gemm2_operands__bf16+hidden_dequant__fp8__block+out_accumulator__bf16 | cumulative | 8f1ff9f1 | 80 | 0.954184 | 0.499667 | 2.1568e+04 | 3.4305e+04 | pass |
| gemm1_accumulator__f16+gemm1_operands__f16+gemm1_output__f16+gemm2_accumulator__bf16+gemm2_operands__bf16+hidden_dequant__fp8__block+out_accumulator__bf16 | cumulative | 1a4c6ba1 | 901 | 0.951962 | 0.474342 | 3.2768e+04 | 1.1196e+05 | pass |
| gemm1_accumulator__f16+gemm1_operands__f16+gemm1_output__f16+gemm2_accumulator__bf16+gemm2_operands__bf16+hidden_dequant__fp8__block+out_accumulator__bf16 | cumulative | a7c2bcfd | 16 | 0.952157 | 0.477679 | 2.0160e+04 | 6.6916e+02 | pass |
| gemm1_accumulator__f16+gemm1_operands__f16+gemm1_output__f16+gemm2_accumulator__bf16+gemm2_operands__bf16+hidden_dequant__fp8__block+out_accumulator__bf16 | cumulative | 2e69caee | 15 | 0.979501 | 0.777632 | 1.8432e+04 | 3.8245e+02 | pass |
| gemm1_accumulator__f16+gemm1_operands__f16+gemm1_output__f16+gemm2_accumulator__bf16+gemm2_operands__bf16+hidden_dequant__fp8__block+out_accumulator__bf16 | cumulative | 8cba5890 | 14 | 0.966717 | 0.643066 | 2.6112e+04 | 1.9689e+03 | pass |
| gemm1_accumulator__f16+gemm1_operands__f16+gemm1_output__f16+gemm2_accumulator__bf16+gemm2_operands__bf16+hidden_dequant__fp8__block+out_accumulator__bf16 | cumulative | 5e8dc11c | 14107 | 0.956244 | 0.522708 | 4.6080e+04 | 1.1120e+11 | pass |
| gemm1_accumulator__f16+gemm1_operands__f16+gemm1_output__f16+gemm2_accumulator__bf16+gemm2_operands__bf16+hidden_dequant__fp8__block+out_accumulator__bf16 | cumulative | 58a34f27 | 11948 | 0.966504 | 0.634342 | 3.6864e+04 | 3.6372e+06 | pass |
| gemm1_accumulator__f16+gemm1_operands__f16+gemm1_output__f16+gemm2_accumulator__bf16+gemm2_operands__bf16+hidden_dequant__fp8__block+out_accumulator__bf16 | cumulative | 5eadab1e | 62 | 0.963851 | 0.608245 | 2.1504e+04 | 1.0443e+03 | pass |
| gemm1_accumulator__f16+gemm1_operands__f16+gemm1_output__f16+gemm2_accumulator__bf16+gemm2_operands__bf16+hidden_dequant__fp8__block+out_accumulator__bf16 | cumulative | eedc63b2 | 59 | 0.970422 | 0.674741 | 2.0480e+04 | 3.1437e+03 | pass |
| gemm1_accumulator__f16+gemm1_operands__f16+gemm1_output__f16+gemm2_accumulator__bf16+gemm2_operands__bf16+hidden_dequant__fp8__block+out_accumulator__bf16 | cumulative | e626d3e6 | 58 | 0.954828 | 0.509944 | 2.2528e+04 | 5.0816e+03 | pass |
| gemm1_accumulator__f16+gemm1_operands__f16+gemm1_output__f16+gemm2_accumulator__bf16+gemm2_operands__bf16+hidden_dequant__fp8__block+out_accumulator__bf16 | cumulative | 74d7ff04 | 57 | 0.962609 | 0.590243 | 2.2528e+04 | 8.9791e+03 | pass |
| gemm1_accumulator__f16+gemm1_operands__f16+gemm1_output__f16+gemm2_accumulator__bf16+gemm2_operands__bf16+hidden_dequant__fp8__block+out_accumulator__bf16 | cumulative | 4822167c | 56 | 0.953165 | 0.493994 | 2.7648e+04 | 6.6708e+03 | pass |
| gemm1_accumulator__f16+gemm1_operands__f16+gemm1_output__f16+gemm2_accumulator__bf16+gemm2_operands__bf16+hidden_dequant__fp8__block+out_accumulator__bf16 | cumulative | 81955b1e | 55 | 0.963931 | 0.605695 | 2.4704e+04 | 3.2503e+03 | pass |
| gemm1_accumulator__f16+gemm1_operands__f16+gemm1_output__f16+gemm2_accumulator__bf16+gemm2_operands__bf16+hidden_dequant__fp8__block+out_accumulator__bf16 | cumulative | 76010cb4 | 54 | 0.970347 | 0.676104 | 2.0544e+04 | 7.1670e+03 | pass |
| gemm1_accumulator__f16+gemm1_operands__f16+gemm1_output__f16+gemm2_accumulator__bf16+gemm2_operands__bf16+hidden_dequant__fp8__block+out_accumulator__bf16 | cumulative | fc378037 | 53 | 0.965707 | 0.623368 | 3.6864e+04 | 1.7087e+04 | pass |
| gemm1_accumulator__f16+gemm1_operands__f16+gemm1_output__f16+gemm2_accumulator__bf16+gemm2_operands__bf16+hidden_dequant__fp8__block+out_accumulator__bf16 | cumulative | f7d6ac7c | 52 | 0.977636 | 0.758360 | 2.2528e+04 | 5.7905e+03 | pass |
| gemm1_accumulator__f16+gemm1_operands__f16+gemm1_output__f16+gemm2_accumulator__bf16+gemm2_operands__bf16+hidden_dequant__fp8__block+out_accumulator__bf16+swiglu_input__f16 | cumulative | b8f4f012 | 7 | 0.979572 | 0.765206 | 1.7664e+04 | 1.1733e+03 | pass |
| gemm1_accumulator__f16+gemm1_operands__f16+gemm1_output__f16+gemm2_accumulator__bf16+gemm2_operands__bf16+hidden_dequant__fp8__block+out_accumulator__bf16+swiglu_input__f16 | cumulative | e05c6c03 | 1 | 0.917271 | 0.148996 | 1.4848e+04 | 1.9423e+02 | pass |
| gemm1_accumulator__f16+gemm1_operands__f16+gemm1_output__f16+gemm2_accumulator__bf16+gemm2_operands__bf16+hidden_dequant__fp8__block+out_accumulator__bf16+swiglu_input__f16 | cumulative | 6230e838 | 32 | 0.968711 | 0.661277 | 2.0736e+04 | 1.0893e+04 | pass |
| gemm1_accumulator__f16+gemm1_operands__f16+gemm1_output__f16+gemm2_accumulator__bf16+gemm2_operands__bf16+hidden_dequant__fp8__block+out_accumulator__bf16+swiglu_input__f16 | cumulative | 8f1ff9f1 | 80 | 0.954365 | 0.499890 | 3.2768e+04 | 1.2850e+04 | pass |
| gemm1_accumulator__f16+gemm1_operands__f16+gemm1_output__f16+gemm2_accumulator__bf16+gemm2_operands__bf16+hidden_dequant__fp8__block+out_accumulator__bf16+swiglu_input__f16 | cumulative | 1a4c6ba1 | 901 | 0.951947 | 0.474350 | 3.2768e+04 | 3.6640e+05 | pass |
| gemm1_accumulator__f16+gemm1_operands__f16+gemm1_output__f16+gemm2_accumulator__bf16+gemm2_operands__bf16+hidden_dequant__fp8__block+out_accumulator__bf16+swiglu_input__f16 | cumulative | a7c2bcfd | 16 | 0.952550 | 0.477147 | 1.6384e+04 | 5.2919e+03 | pass |
| gemm1_accumulator__f16+gemm1_operands__f16+gemm1_output__f16+gemm2_accumulator__bf16+gemm2_operands__bf16+hidden_dequant__fp8__block+out_accumulator__bf16+swiglu_input__f16 | cumulative | 2e69caee | 15 | 0.980394 | 0.776907 | 2.0480e+04 | 2.5002e+03 | pass |
| gemm1_accumulator__f16+gemm1_operands__f16+gemm1_output__f16+gemm2_accumulator__bf16+gemm2_operands__bf16+hidden_dequant__fp8__block+out_accumulator__bf16+swiglu_input__f16 | cumulative | 8cba5890 | 14 | 0.966837 | 0.642807 | 2.5088e+04 | 1.2111e+04 | pass |
| gemm1_accumulator__f16+gemm1_operands__f16+gemm1_output__f16+gemm2_accumulator__bf16+gemm2_operands__bf16+hidden_dequant__fp8__block+out_accumulator__bf16+swiglu_input__f16 | cumulative | 5e8dc11c | 14107 | 0.956222 | 0.522744 | 4.5056e+04 | 1.0954e+06 | pass |
| gemm1_accumulator__f16+gemm1_operands__f16+gemm1_output__f16+gemm2_accumulator__bf16+gemm2_operands__bf16+hidden_dequant__fp8__block+out_accumulator__bf16+swiglu_input__f16 | cumulative | 58a34f27 | 11948 | 0.966469 | 0.634373 | 3.9936e+04 | 1.5925e+06 | pass |
| gemm1_accumulator__f16+gemm1_operands__f16+gemm1_output__f16+gemm2_accumulator__bf16+gemm2_operands__bf16+hidden_dequant__fp8__block+out_accumulator__bf16+swiglu_input__f16 | cumulative | 5eadab1e | 62 | 0.964592 | 0.610383 | 2.4576e+04 | 8.2247e+03 | pass |
| gemm1_accumulator__f16+gemm1_operands__f16+gemm1_output__f16+gemm2_accumulator__bf16+gemm2_operands__bf16+hidden_dequant__fp8__block+out_accumulator__bf16+swiglu_input__f16 | cumulative | eedc63b2 | 59 | 0.970339 | 0.675323 | 2.5600e+04 | 6.0157e+04 | pass |
| gemm1_accumulator__f16+gemm1_operands__f16+gemm1_output__f16+gemm2_accumulator__bf16+gemm2_operands__bf16+hidden_dequant__fp8__block+out_accumulator__bf16+swiglu_input__f16 | cumulative | e626d3e6 | 58 | 0.955275 | 0.510689 | 2.3552e+04 | 1.1738e+04 | pass |
| gemm1_accumulator__f16+gemm1_operands__f16+gemm1_output__f16+gemm2_accumulator__bf16+gemm2_operands__bf16+hidden_dequant__fp8__block+out_accumulator__bf16+swiglu_input__f16 | cumulative | 74d7ff04 | 57 | 0.962333 | 0.589376 | 2.3552e+04 | 4.5030e+03 | pass |
| gemm1_accumulator__f16+gemm1_operands__f16+gemm1_output__f16+gemm2_accumulator__bf16+gemm2_operands__bf16+hidden_dequant__fp8__block+out_accumulator__bf16+swiglu_input__f16 | cumulative | 4822167c | 56 | 0.953451 | 0.492673 | 2.2320e+04 | 9.9996e+03 | pass |
| gemm1_accumulator__f16+gemm1_operands__f16+gemm1_output__f16+gemm2_accumulator__bf16+gemm2_operands__bf16+hidden_dequant__fp8__block+out_accumulator__bf16+swiglu_input__f16 | cumulative | 81955b1e | 55 | 0.964702 | 0.605908 | 2.1504e+04 | 1.6886e+03 | pass |
| gemm1_accumulator__f16+gemm1_operands__f16+gemm1_output__f16+gemm2_accumulator__bf16+gemm2_operands__bf16+hidden_dequant__fp8__block+out_accumulator__bf16+swiglu_input__f16 | cumulative | 76010cb4 | 54 | 0.969879 | 0.675401 | 2.2016e+04 | 3.2007e+03 | pass |
| gemm1_accumulator__f16+gemm1_operands__f16+gemm1_output__f16+gemm2_accumulator__bf16+gemm2_operands__bf16+hidden_dequant__fp8__block+out_accumulator__bf16+swiglu_input__f16 | cumulative | fc378037 | 53 | 0.964941 | 0.621823 | 2.0864e+04 | 5.0471e+03 | pass |
| gemm1_accumulator__f16+gemm1_operands__f16+gemm1_output__f16+gemm2_accumulator__bf16+gemm2_operands__bf16+hidden_dequant__fp8__block+out_accumulator__bf16+swiglu_input__f16 | cumulative | f7d6ac7c | 52 | 0.977933 | 0.760074 | 2.2912e+04 | 2.2937e+04 | pass |
| gemm1_accumulator__f16+gemm1_operands__f16+gemm1_output__f16+gemm2_accumulator__bf16+gemm2_operands__bf16+hidden_dequant__fp8__block+out_accumulator__bf16+swiglu_input__f16+swiglu_output__fp8__block | cumulative | b8f4f012 | 7 | 0.974410 | 0.756796 | 2.0480e+04 | 3.7927e+02 | pass |
| gemm1_accumulator__f16+gemm1_operands__f16+gemm1_output__f16+gemm2_accumulator__bf16+gemm2_operands__bf16+hidden_dequant__fp8__block+out_accumulator__bf16+swiglu_input__f16+swiglu_output__fp8__block | cumulative | e05c6c03 | 1 | 0.902483 | 0.143136 | 1.6672e+04 | 2.0228e+03 | pass |
| gemm1_accumulator__f16+gemm1_operands__f16+gemm1_output__f16+gemm2_accumulator__bf16+gemm2_operands__bf16+hidden_dequant__fp8__block+out_accumulator__bf16+swiglu_input__f16+swiglu_output__fp8__block | cumulative | 6230e838 | 32 | 0.962895 | 0.650552 | 3.5008e+04 | 5.1626e+04 | pass |
| gemm1_accumulator__f16+gemm1_operands__f16+gemm1_output__f16+gemm2_accumulator__bf16+gemm2_operands__bf16+hidden_dequant__fp8__block+out_accumulator__bf16+swiglu_input__f16+swiglu_output__fp8__block | cumulative | 8f1ff9f1 | 80 | 0.945764 | 0.485353 | 2.7648e+04 | 3.0621e+04 | pass |
| gemm1_accumulator__f16+gemm1_operands__f16+gemm1_output__f16+gemm2_accumulator__bf16+gemm2_operands__bf16+hidden_dequant__fp8__block+out_accumulator__bf16+swiglu_input__f16+swiglu_output__fp8__block | cumulative | 1a4c6ba1 | 901 | 0.943107 | 0.459289 | 3.6864e+04 | 5.3651e+04 | pass |
| gemm1_accumulator__f16+gemm1_operands__f16+gemm1_output__f16+gemm2_accumulator__bf16+gemm2_operands__bf16+hidden_dequant__fp8__block+out_accumulator__bf16+swiglu_input__f16+swiglu_output__fp8__block | cumulative | a7c2bcfd | 16 | 0.945321 | 0.467268 | 3.3024e+04 | 1.7497e+03 | pass |
| gemm1_accumulator__f16+gemm1_operands__f16+gemm1_output__f16+gemm2_accumulator__bf16+gemm2_operands__bf16+hidden_dequant__fp8__block+out_accumulator__bf16+swiglu_input__f16+swiglu_output__fp8__block | cumulative | 2e69caee | 15 | 0.975409 | 0.770787 | 2.4576e+04 | 5.0525e+02 | pass |
| gemm1_accumulator__f16+gemm1_operands__f16+gemm1_output__f16+gemm2_accumulator__bf16+gemm2_operands__bf16+hidden_dequant__fp8__block+out_accumulator__bf16+swiglu_input__f16+swiglu_output__fp8__block | cumulative | 8cba5890 | 14 | 0.961605 | 0.633560 | 2.2528e+04 | 2.3243e+03 | pass |
| gemm1_accumulator__f16+gemm1_operands__f16+gemm1_output__f16+gemm2_accumulator__bf16+gemm2_operands__bf16+hidden_dequant__fp8__block+out_accumulator__bf16+swiglu_input__f16+swiglu_output__fp8__block | cumulative | 5e8dc11c | 14107 | 0.948472 | 0.509235 | 5.3248e+04 | 3.4952e+05 | pass |
| gemm1_accumulator__f16+gemm1_operands__f16+gemm1_output__f16+gemm2_accumulator__bf16+gemm2_operands__bf16+hidden_dequant__fp8__block+out_accumulator__bf16+swiglu_input__f16+swiglu_output__fp8__block | cumulative | 58a34f27 | 11948 | 0.960473 | 0.623856 | 4.3520e+04 | 3.8720e+11 | pass |
| gemm1_accumulator__f16+gemm1_operands__f16+gemm1_output__f16+gemm2_accumulator__bf16+gemm2_operands__bf16+hidden_dequant__fp8__block+out_accumulator__bf16+swiglu_input__f16+swiglu_output__fp8__block | cumulative | 5eadab1e | 62 | 0.958496 | 0.599686 | 2.5600e+04 | 7.6790e+03 | pass |
| gemm1_accumulator__f16+gemm1_operands__f16+gemm1_output__f16+gemm2_accumulator__bf16+gemm2_operands__bf16+hidden_dequant__fp8__block+out_accumulator__bf16+swiglu_input__f16+swiglu_output__fp8__block | cumulative | eedc63b2 | 59 | 0.964870 | 0.665803 | 2.4576e+04 | 1.1841e+05 | pass |
| gemm1_accumulator__f16+gemm1_operands__f16+gemm1_output__f16+gemm2_accumulator__bf16+gemm2_operands__bf16+hidden_dequant__fp8__block+out_accumulator__bf16+swiglu_input__f16+swiglu_output__fp8__block | cumulative | e626d3e6 | 58 | 0.947374 | 0.496584 | 3.7888e+04 | 4.8586e+04 | pass |
| gemm1_accumulator__f16+gemm1_operands__f16+gemm1_output__f16+gemm2_accumulator__bf16+gemm2_operands__bf16+hidden_dequant__fp8__block+out_accumulator__bf16+swiglu_input__f16+swiglu_output__fp8__block | cumulative | 74d7ff04 | 57 | 0.955710 | 0.578252 | 2.5856e+04 | 6.9777e+03 | pass |
| gemm1_accumulator__f16+gemm1_operands__f16+gemm1_output__f16+gemm2_accumulator__bf16+gemm2_operands__bf16+hidden_dequant__fp8__block+out_accumulator__bf16+swiglu_input__f16+swiglu_output__fp8__block | cumulative | 4822167c | 56 | 0.946065 | 0.479398 | 2.2528e+04 | 5.0422e+03 | pass |
| gemm1_accumulator__f16+gemm1_operands__f16+gemm1_output__f16+gemm2_accumulator__bf16+gemm2_operands__bf16+hidden_dequant__fp8__block+out_accumulator__bf16+swiglu_input__f16+swiglu_output__fp8__block | cumulative | 81955b1e | 55 | 0.958031 | 0.595188 | 2.7032e+04 | 1.8328e+03 | pass |
| gemm1_accumulator__f16+gemm1_operands__f16+gemm1_output__f16+gemm2_accumulator__bf16+gemm2_operands__bf16+hidden_dequant__fp8__block+out_accumulator__bf16+swiglu_input__f16+swiglu_output__fp8__block | cumulative | 76010cb4 | 54 | 0.964988 | 0.665026 | 3.1744e+04 | 3.2150e+03 | pass |
| gemm1_accumulator__f16+gemm1_operands__f16+gemm1_output__f16+gemm2_accumulator__bf16+gemm2_operands__bf16+hidden_dequant__fp8__block+out_accumulator__bf16+swiglu_input__f16+swiglu_output__fp8__block | cumulative | fc378037 | 53 | 0.959266 | 0.611136 | 2.7648e+04 | 2.1726e+03 | pass |
| gemm1_accumulator__f16+gemm1_operands__f16+gemm1_output__f16+gemm2_accumulator__bf16+gemm2_operands__bf16+hidden_dequant__fp8__block+out_accumulator__bf16+swiglu_input__f16+swiglu_output__fp8__block | cumulative | f7d6ac7c | 52 | 0.974529 | 0.752932 | 2.0736e+04 | 1.5614e+04 | pass |