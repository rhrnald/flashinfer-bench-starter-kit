# MoE Precision Search 2026-04-23T03:06:06Z

| metric | value |
|---|---|
| timestamp | 2026-04-23T03:06:06Z |
| definition | moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048 |
| seed | 1234 |
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
| contest_safe_survivors | gemm1_output__f16, swiglu_input__f16, gemm1_operands__f16, out_accumulator__bf16, gemm1_accumulator__f16, gemm2_accumulator__bf16, swiglu_output__fp8__block, hidden_dequant__fp8__block, gemm2_operands__f16 |
| strict_safe_survivor_count | 6 |
| strict_safe_survivors | gemm1_accumulator__f16, gemm1_operands__f16, gemm1_output__f16, gemm2_operands__f16, out_accumulator__bf16, swiglu_input__f16 |
| contest_only_survivor_count | 3 |
| contest_only_survivors | gemm2_accumulator__bf16, hidden_dequant__fp8__block, swiglu_output__fp8__block |

## Survivor Summary

| category | count | candidates |
|---|---:|---|
| contest_safe_single_stage | 9 | gemm1_output__f16, swiglu_input__f16, gemm1_operands__f16, out_accumulator__bf16, gemm1_accumulator__f16, gemm2_accumulator__bf16, swiglu_output__fp8__block, hidden_dequant__fp8__block, gemm2_operands__f16 |
| strict_safe_single_stage | 6 | gemm1_accumulator__f16, gemm1_operands__f16, gemm1_output__f16, gemm2_operands__f16, out_accumulator__bf16, swiglu_input__f16 |
| contest_only_single_stage | 3 | gemm2_accumulator__bf16, hidden_dequant__fp8__block, swiglu_output__fp8__block |

## Stage Summary

| stage | best_safe_mode | scale | worst_matched_contest | worst_matched_strict | worst_rel | status |
|---|---|---|---:|---:|---:|---|
| gemm1_accumulator | f16 | none | 0.997210 | 0.902065 | 2.0200e+10 | safe |
| gemm1_operands | f16 | none | 0.998884 | 0.971959 | 3.5750e+09 | safe |
| gemm1_output | f16 | none | 0.999302 | 0.980469 | 1.9750e+09 | safe |
| gemm2_accumulator | bf16 | none | 0.992885 | 0.760742 | 1.9700e+10 | safe |
| gemm2_operands | f16 | none | 0.922128 | 0.913127 | 7.5202e+00 | safe |
| hidden_dequant | fp8 | block | 0.924805 | 0.164062 | 3.3920e+11 | safe |
| out_accumulator | bf16 | none | 0.997489 | 0.941546 | 4.9750e+09 | safe |
| swiglu_input | f16 | none | 0.999302 | 0.980469 | 1.9750e+09 | safe |
| swiglu_output | fp8 | block | 0.955357 | 0.280831 | 3.5200e+11 | safe |

## Cumulative Safe Frontier

| order | candidate | worst_matched_contest | worst_matched_strict | worst_rel | status |
|---|---|---:|---:|---:|---|
| 1 | gemm1_accumulator__f16 | 0.996791 | 0.901925 | 1.1172e+08 | safe |
| 2 | gemm1_accumulator__f16+gemm1_operands__f16 | 0.999302 | 0.974609 | 1.2202e+04 | safe |
| 3 | gemm1_accumulator__f16+gemm1_operands__f16+gemm1_output__f16 | 0.999023 | 0.963588 | 1.2250e+09 | safe |
| 4 | gemm1_accumulator__f16+gemm1_operands__f16+gemm1_output__f16+gemm2_accumulator__bf16 | 0.990234 | 0.747907 | 1.5213e+05 | safe |
| 5 | gemm1_accumulator__f16+gemm1_operands__f16+gemm1_output__f16+gemm2_accumulator__bf16+gemm2_operands__f16 | 0.874555 | 0.855983 | 3.4964e+02 | unsafe |

## BF16/F16 Margin

| stage | preferred | bf16_contest | f16_contest | bf16_strict | f16_strict | |
|---|---|---:|---:|---:|---:|---|

## Pairwise Summary

| pair | worst_matched_contest | worst_matched_strict | worst_rel | status |
|---|---:|---:|---:|---|
| gemm1_accumulator__f16+gemm2_accumulator__bf16 | 0.989816 | 0.731027 | 8.1162e+04 | safe |
| gemm1_output__f16+gemm2_accumulator__bf16 | 0.990653 | 0.747907 | 7.9148e+04 | safe |
| gemm2_accumulator__bf16+out_accumulator__bf16 | 0.990932 | 0.733817 | 8.6709e+04 | safe |
| gemm1_operands__f16+gemm2_accumulator__bf16 | 0.990932 | 0.741350 | 7.8139e+04 | safe |
| gemm1_accumulator__f16+out_accumulator__bf16 | 0.995257 | 0.885324 | 4.9919e+04 | safe |
| gemm1_accumulator__f16+gemm1_output__f16 | 0.996652 | 0.903599 | 3.9935e+04 | safe |
| gemm1_operands__f16+out_accumulator__bf16 | 0.997210 | 0.929967 | 5.9647e+04 | safe |
| gemm1_output__f16+out_accumulator__bf16 | 0.997210 | 0.932338 | 5.2223e+04 | safe |
| gemm1_operands__f16+gemm1_output__f16 | 0.998326 | 0.966657 | 1.5530e+04 | safe |
| gemm1_operands__f16+gemm1_accumulator__f16 | 0.998884 | 0.973772 | 1.6554e+04 | safe |

## Stress Summary

| candidate | worst_workload | worst_seq_len | worst_matched_contest | worst_matched_strict | status |
|---|---|---:|---:|---:|---|
| gemm2_accumulator__bf16 | e05c6c03 | 1 | 0.992885 | 0.760742 | safe |

## Promotion Summary

| category | candidates |
|---|---|
| bf16_f16_survivors | gemm1_output__f16, swiglu_input__f16, gemm1_operands__f16, out_accumulator__bf16, gemm1_accumulator__f16, gemm2_accumulator__bf16, swiglu_output__fp8__block, hidden_dequant__fp8__block, gemm2_operands__f16 |
| strict_survivors | gemm1_accumulator__f16, gemm1_operands__f16, gemm1_output__f16, gemm2_operands__f16, out_accumulator__bf16, swiglu_input__f16 |
| pairwise_shortlist | gemm1_operands__f16, gemm1_accumulator__f16, gemm1_output__f16, gemm2_accumulator__bf16, out_accumulator__bf16 |

## Sampled Results

| candidate | phase | workload | seq_len | matched_contest | matched_strict | max_abs | max_rel | failure |
|---|---|---|---:|---:|---:|---:|---:|---|
| baseline__fp32 | single_stage | b8f4f012 | 7 | 1.000000 | 1.000000 | 0.0000e+00 | 0.0000e+00 | pass |
| gemm1_accumulator__f16 | single_stage | b8f4f012 | 7 | 0.998924 | 0.969746 | 2.0480e+03 | 8.6306e+01 | pass |
| gemm1_operands__f16 | single_stage | b8f4f012 | 7 | 0.999681 | 0.992068 | 1.0240e+03 | 1.0387e+01 | pass |
| gemm1_output__f16 | single_stage | b8f4f012 | 7 | 0.999721 | 0.994200 | 1.0240e+03 | 1.1763e+01 | pass |
| gemm2_accumulator__bf16 | single_stage | b8f4f012 | 7 | 0.997907 | 0.928731 | 3.0720e+03 | 1.0902e+02 | pass |
| gemm2_operands__f16 | single_stage | b8f4f012 | 7 | 0.999801 | 0.994141 | 1.0240e+03 | 7.5202e+00 | pass |
| hidden_dequant__fp8__block | single_stage | b8f4f012 | 7 | 0.978755 | 0.760802 | 1.5104e+04 | 1.4033e+02 | pass |
| out_accumulator__bf16 | single_stage | b8f4f012 | 7 | 0.999322 | 0.980309 | 2.0480e+03 | 1.7867e+01 | pass |
| swiglu_input__f16 | single_stage | b8f4f012 | 7 | 0.999721 | 0.994200 | 1.0240e+03 | 1.1763e+01 | pass |
| swiglu_output__fp8__block | single_stage | b8f4f012 | 7 | 0.986029 | 0.788963 | 8.6400e+03 | 9.3621e+02 | pass |
| baseline__fp32 | single_stage | e05c6c03 | 1 | 1.000000 | 1.000000 | 0.0000e+00 | 0.0000e+00 | pass |
| gemm1_accumulator__f16 | single_stage | e05c6c03 | 1 | 0.997210 | 0.902065 | 1.0240e+03 | 9.3122e+00 | pass |
| gemm1_operands__f16 | single_stage | e05c6c03 | 1 | 0.998884 | 0.971959 | 1.0240e+03 | 2.7409e+00 | pass |
| gemm1_output__f16 | single_stage | e05c6c03 | 1 | 0.999302 | 0.980469 | 1.0240e+03 | 1.4000e+00 | pass |
| gemm2_accumulator__bf16 | single_stage | e05c6c03 | 1 | 0.992885 | 0.760742 | 2.0480e+03 | 1.3722e+01 | pass |
| gemm2_operands__f16 | single_stage | e05c6c03 | 1 | 0.999023 | 0.982143 | 1.0240e+03 | 2.9793e+00 | pass |
| hidden_dequant__fp8__block | single_stage | e05c6c03 | 1 | 0.924805 | 0.164062 | 1.3584e+04 | 3.0245e+02 | pass |
| out_accumulator__bf16 | single_stage | e05c6c03 | 1 | 0.997489 | 0.941546 | 2.0480e+03 | 8.1415e+00 | pass |
| swiglu_input__f16 | single_stage | e05c6c03 | 1 | 0.999302 | 0.980469 | 1.0240e+03 | 1.4000e+00 | pass |
| swiglu_output__fp8__block | single_stage | e05c6c03 | 1 | 0.955357 | 0.280831 | 6.6560e+03 | 1.9831e+02 | pass |
| baseline__fp32 | single_stage | 6230e838 | 32 | 1.000000 | 1.000000 | 0.0000e+00 | 0.0000e+00 | pass |
| gemm1_accumulator__f16 | single_stage | 6230e838 | 32 | 0.998744 | 0.956887 | 2.0480e+03 | 2.1706e+02 | pass |
| gemm1_operands__f16 | single_stage | 6230e838 | 32 | 0.999651 | 0.988857 | 2.0480e+03 | 2.7368e+01 | pass |
| gemm1_output__f16 | single_stage | 6230e838 | 32 | 0.999760 | 0.992196 | 2.0480e+03 | 3.8120e+01 | pass |
| gemm2_accumulator__bf16 | single_stage | 6230e838 | 32 | 0.996953 | 0.901489 | 5.1200e+03 | 6.0111e+02 | pass |
| gemm2_operands__f16 | single_stage | 6230e838 | 32 | 0.968515 | 0.961775 | nan | nan | pass |
| hidden_dequant__fp8__block | single_stage | 6230e838 | 32 | 0.969317 | 0.661809 | 2.0480e+04 | 1.1795e+04 | pass |
| out_accumulator__bf16 | single_stage | 6230e838 | 32 | 0.999333 | 0.976680 | 2.0480e+03 | 5.6519e+01 | pass |
| swiglu_input__f16 | single_stage | 6230e838 | 32 | 0.999760 | 0.992196 | 2.0480e+03 | 3.8120e+01 | pass |
| swiglu_output__fp8__block | single_stage | 6230e838 | 32 | 0.980046 | 0.700213 | 1.2800e+04 | 1.5145e+03 | pass |
| baseline__fp32 | single_stage | 8f1ff9f1 | 80 | 1.000000 | 1.000000 | 0.0000e+00 | 0.0000e+00 | pass |
| gemm1_accumulator__f16 | single_stage | 8f1ff9f1 | 80 | 0.998028 | 0.939179 | 4.0960e+03 | 1.3992e+03 | pass |
| gemm1_operands__f16 | single_stage | 8f1ff9f1 | 80 | 0.999456 | 0.983529 | 2.0480e+03 | 1.4008e+03 | pass |
| gemm1_output__f16 | single_stage | 8f1ff9f1 | 80 | 0.999627 | 0.988398 | 2.0480e+03 | 1.1224e+03 | pass |
| gemm2_accumulator__bf16 | single_stage | 8f1ff9f1 | 80 | 0.995776 | 0.862516 | 4.0960e+03 | 2.8026e+03 | pass |
| gemm2_operands__f16 | single_stage | 8f1ff9f1 | 80 | 0.948941 | 0.938902 | nan | nan | pass |
| hidden_dequant__fp8__block | single_stage | 8f1ff9f1 | 80 | 0.954400 | 0.498809 | 2.8672e+04 | 1.9378e+04 | pass |
| out_accumulator__bf16 | single_stage | 8f1ff9f1 | 80 | 0.999210 | 0.974758 | 4.0960e+03 | 9.2654e+02 | pass |
| swiglu_input__f16 | single_stage | 8f1ff9f1 | 80 | 0.999627 | 0.988398 | 2.0480e+03 | 1.1224e+03 | pass |
| swiglu_output__fp8__block | single_stage | 8f1ff9f1 | 80 | 0.970837 | 0.555613 | 1.6384e+04 | 5.7584e+04 | pass |
| baseline__fp32 | single_stage | 1a4c6ba1 | 901 | 1.000000 | 1.000000 | 0.0000e+00 | 0.0000e+00 | pass |
| gemm1_accumulator__f16 | single_stage | 1a4c6ba1 | 901 | 0.997930 | 0.935792 | 4.0960e+03 | 2.0875e+09 | pass |
| gemm1_operands__f16 | single_stage | 1a4c6ba1 | 901 | 0.999442 | 0.982676 | 4.0960e+03 | 1.8750e+09 | pass |
| gemm1_output__f16 | single_stage | 1a4c6ba1 | 901 | 0.999606 | 0.987757 | 4.0960e+03 | 1.8000e+09 | pass |
| gemm2_accumulator__bf16 | single_stage | 1a4c6ba1 | 901 | 0.995513 | 0.853187 | 8.1920e+03 | 1.3300e+10 | pass |
| gemm2_operands__f16 | single_stage | 1a4c6ba1 | 901 | 0.955566 | 0.944830 | nan | nan | pass |
| hidden_dequant__fp8__block | single_stage | 1a4c6ba1 | 901 | 0.952012 | 0.474592 | 3.8912e+04 | 1.7040e+11 | pass |
| out_accumulator__bf16 | single_stage | 1a4c6ba1 | 901 | 0.999105 | 0.971200 | 4.0960e+03 | 4.9750e+09 | pass |
| swiglu_input__f16 | single_stage | 1a4c6ba1 | 901 | 0.999606 | 0.987757 | 4.0960e+03 | 1.8000e+09 | pass |
| swiglu_output__fp8__block | single_stage | 1a4c6ba1 | 901 | 0.969383 | 0.533301 | 1.9456e+04 | 1.8640e+11 | pass |
| baseline__fp32 | single_stage | a7c2bcfd | 16 | 1.000000 | 1.000000 | 0.0000e+00 | 0.0000e+00 | pass |
| gemm1_accumulator__f16 | single_stage | a7c2bcfd | 16 | 0.997934 | 0.936166 | 2.0480e+03 | 5.8467e+01 | pass |
| gemm1_operands__f16 | single_stage | a7c2bcfd | 16 | 0.999486 | 0.983102 | 2.0480e+03 | 8.2667e+00 | pass |
| gemm1_output__f16 | single_stage | a7c2bcfd | 16 | 0.999660 | 0.987287 | 2.0480e+03 | 8.7308e+00 | pass |
| gemm2_accumulator__bf16 | single_stage | a7c2bcfd | 16 | 0.995823 | 0.861886 | 4.0960e+03 | 6.5377e+02 | pass |
| gemm2_operands__f16 | single_stage | a7c2bcfd | 16 | 0.937291 | 0.926313 | nan | nan | pass |
| hidden_dequant__fp8__block | single_stage | a7c2bcfd | 16 | 0.954180 | 0.482997 | 1.6384e+04 | 1.0102e+03 | pass |
| out_accumulator__bf16 | single_stage | a7c2bcfd | 16 | 0.999337 | 0.979998 | 2.0480e+03 | 5.1000e+01 | pass |
| swiglu_input__f16 | single_stage | a7c2bcfd | 16 | 0.999660 | 0.987287 | 2.0480e+03 | 8.7308e+00 | pass |
| swiglu_output__fp8__block | single_stage | a7c2bcfd | 16 | 0.968898 | 0.536028 | 1.0504e+04 | 3.0135e+02 | pass |
| baseline__fp32 | single_stage | 2e69caee | 15 | 1.000000 | 1.000000 | 0.0000e+00 | 0.0000e+00 | pass |
| gemm1_accumulator__f16 | single_stage | 2e69caee | 15 | 0.999200 | 0.973847 | 2.0480e+03 | 1.4405e+01 | pass |
| gemm1_operands__f16 | single_stage | 2e69caee | 15 | 0.999823 | 0.992494 | 2.0480e+03 | 5.1302e+00 | pass |
| gemm1_output__f16 | single_stage | 2e69caee | 15 | 0.999870 | 0.994857 | 2.0480e+03 | 6.7674e+00 | pass |
| gemm2_accumulator__bf16 | single_stage | 2e69caee | 15 | 0.998140 | 0.936505 | 6.1440e+03 | 3.3000e+01 | pass |
| gemm2_operands__f16 | single_stage | 2e69caee | 15 | 0.999870 | 0.995071 | 2.0480e+03 | 2.0884e+00 | pass |
| hidden_dequant__fp8__block | single_stage | 2e69caee | 15 | 0.979046 | 0.777939 | 1.9456e+04 | 1.0754e+03 | pass |
| out_accumulator__bf16 | single_stage | 2e69caee | 15 | 0.999684 | 0.988467 | 2.0480e+03 | 2.5366e+00 | pass |
| swiglu_input__f16 | single_stage | 2e69caee | 15 | 0.999870 | 0.994857 | 2.0480e+03 | 6.7674e+00 | pass |
| swiglu_output__fp8__block | single_stage | 2e69caee | 15 | 0.986868 | 0.801535 | 1.4336e+04 | 1.5498e+02 | pass |
| baseline__fp32 | single_stage | 8cba5890 | 14 | 1.000000 | 1.000000 | 0.0000e+00 | 0.0000e+00 | pass |
| gemm1_accumulator__f16 | single_stage | 8cba5890 | 14 | 0.998635 | 0.953563 | 4.0960e+03 | 2.1081e+01 | pass |
| gemm1_operands__f16 | single_stage | 8cba5890 | 14 | 0.999731 | 0.987345 | 2.0480e+03 | 2.6533e+00 | pass |
| gemm1_output__f16 | single_stage | 8cba5890 | 14 | 0.999831 | 0.991470 | 2.0480e+03 | 9.0000e+00 | pass |
| gemm2_accumulator__bf16 | single_stage | 8cba5890 | 14 | 0.996732 | 0.895169 | 4.0960e+03 | 1.0752e+02 | pass |
| gemm2_operands__f16 | single_stage | 8cba5890 | 14 | 0.928462 | 0.921427 | nan | nan | pass |
| hidden_dequant__fp8__block | single_stage | 8cba5890 | 14 | 0.966508 | 0.642229 | 2.3040e+04 | 5.4286e+02 | pass |
| out_accumulator__bf16 | single_stage | 8cba5890 | 14 | 0.999362 | 0.978645 | 4.0960e+03 | 2.5880e+01 | pass |
| swiglu_input__f16 | single_stage | 8cba5890 | 14 | 0.999831 | 0.991470 | 2.0480e+03 | 9.0000e+00 | pass |
| swiglu_output__fp8__block | single_stage | 8cba5890 | 14 | 0.978217 | 0.684341 | 1.3824e+04 | 3.2152e+02 | pass |
| baseline__fp32 | single_stage | 5e8dc11c | 14107 | 1.000000 | 1.000000 | 0.0000e+00 | 0.0000e+00 | pass |
| gemm1_accumulator__f16 | single_stage | 5e8dc11c | 14107 | 0.998125 | 0.941683 | 4.0960e+03 | 2.0200e+10 | pass |
| gemm1_operands__f16 | single_stage | 5e8dc11c | 14107 | 0.999501 | 0.984235 | 4.0960e+03 | 2.7250e+09 | pass |
| gemm1_output__f16 | single_stage | 5e8dc11c | 14107 | 0.999647 | 0.988906 | 4.0960e+03 | 1.6375e+09 | pass |
| gemm2_accumulator__bf16 | single_stage | 5e8dc11c | 14107 | 0.996196 | 0.872125 | 1.2288e+04 | 7.3000e+09 | pass |
| gemm2_operands__f16 | single_stage | 5e8dc11c | 14107 | 0.958488 | 0.948656 | nan | nan | pass |
| hidden_dequant__fp8__block | single_stage | 5e8dc11c | 14107 | 0.956378 | 0.523060 | 3.8400e+04 | 3.3920e+11 | pass |
| out_accumulator__bf16 | single_stage | 5e8dc11c | 14107 | 0.999368 | 0.979632 | 8.1920e+03 | 4.7500e+09 | pass |
| swiglu_input__f16 | single_stage | 5e8dc11c | 14107 | 0.999647 | 0.988906 | 4.0960e+03 | 1.6375e+09 | pass |
| swiglu_output__fp8__block | single_stage | 5e8dc11c | 14107 | 0.972197 | 0.576605 | 2.7648e+04 | 3.5200e+11 | pass |
| baseline__fp32 | single_stage | 58a34f27 | 11948 | 1.000000 | 1.000000 | 0.0000e+00 | 0.0000e+00 | pass |
| gemm1_accumulator__f16 | single_stage | 58a34f27 | 11948 | 0.998552 | 0.955031 | 4.0960e+03 | 3.3750e+09 | pass |
| gemm1_operands__f16 | single_stage | 58a34f27 | 11948 | 0.999616 | 0.987916 | 4.0960e+03 | 3.5750e+09 | pass |
| gemm1_output__f16 | single_stage | 58a34f27 | 11948 | 0.999731 | 0.991490 | 4.0960e+03 | 1.9750e+09 | pass |
| gemm2_accumulator__bf16 | single_stage | 58a34f27 | 11948 | 0.997085 | 0.901614 | 8.1920e+03 | 1.9700e+10 | pass |
| gemm2_operands__f16 | single_stage | 58a34f27 | 11948 | 0.971004 | 0.963427 | nan | nan | pass |
| hidden_dequant__fp8__block | single_stage | 58a34f27 | 11948 | 0.966554 | 0.634463 | 3.4816e+04 | 2.2400e+11 | pass |
| out_accumulator__bf16 | single_stage | 58a34f27 | 11948 | 0.999511 | 0.984196 | 4.0960e+03 | 3.2500e+09 | pass |
| swiglu_input__f16 | single_stage | 58a34f27 | 11948 | 0.999731 | 0.991490 | 4.0960e+03 | 1.9750e+09 | pass |
| swiglu_output__fp8__block | single_stage | 58a34f27 | 11948 | 0.978696 | 0.675599 | 2.0480e+04 | 8.2400e+10 | pass |
| baseline__fp32 | single_stage | 5eadab1e | 62 | 1.000000 | 1.000000 | 0.0000e+00 | 0.0000e+00 | pass |
| gemm1_accumulator__f16 | single_stage | 5eadab1e | 62 | 0.998553 | 0.951651 | 2.0480e+03 | 2.4826e+02 | pass |
| gemm1_operands__f16 | single_stage | 5eadab1e | 62 | 0.999624 | 0.986888 | 2.0480e+03 | 1.5197e+01 | pass |
| gemm1_output__f16 | single_stage | 5eadab1e | 62 | 0.999734 | 0.990914 | 2.0480e+03 | 8.3818e+00 | pass |
| gemm2_accumulator__bf16 | single_stage | 5eadab1e | 62 | 0.997043 | 0.895706 | 6.1440e+03 | 1.7620e+02 | pass |
| gemm2_operands__f16 | single_stage | 5eadab1e | 62 | 0.943247 | 0.935538 | nan | nan | pass |
| hidden_dequant__fp8__block | single_stage | 5eadab1e | 62 | 0.964562 | 0.610205 | 2.6624e+04 | 1.6881e+03 | pass |
| out_accumulator__bf16 | single_stage | 5eadab1e | 62 | 0.999518 | 0.984206 | 4.0960e+03 | 5.0000e+01 | pass |
| swiglu_input__f16 | single_stage | 5eadab1e | 62 | 0.999734 | 0.990914 | 2.0480e+03 | 8.3818e+00 | pass |
| swiglu_output__fp8__block | single_stage | 5eadab1e | 62 | 0.977517 | 0.653561 | 1.4848e+04 | 1.1002e+03 | pass |
| baseline__fp32 | single_stage | eedc63b2 | 59 | 1.000000 | 1.000000 | 0.0000e+00 | 0.0000e+00 | pass |
| gemm1_accumulator__f16 | single_stage | eedc63b2 | 59 | 0.998702 | 0.959753 | 4.0960e+03 | 7.5800e+01 | pass |
| gemm1_operands__f16 | single_stage | eedc63b2 | 59 | 0.999702 | 0.989225 | 2.0480e+03 | 8.1643e+00 | pass |
| gemm1_output__f16 | single_stage | eedc63b2 | 59 | 0.999773 | 0.992599 | 2.0480e+03 | 8.9500e+00 | pass |
| gemm2_accumulator__bf16 | single_stage | eedc63b2 | 59 | 0.997543 | 0.914890 | 6.1440e+03 | 5.6469e+01 | pass |
| gemm2_operands__f16 | single_stage | eedc63b2 | 59 | 0.944771 | 0.938666 | nan | nan | pass |
| hidden_dequant__fp8__block | single_stage | eedc63b2 | 59 | 0.970318 | 0.674968 | 2.5728e+04 | 3.1089e+03 | pass |
| out_accumulator__bf16 | single_stage | eedc63b2 | 59 | 0.999622 | 0.987773 | 4.0960e+03 | 2.0368e+01 | pass |
| swiglu_input__f16 | single_stage | eedc63b2 | 59 | 0.999773 | 0.992599 | 2.0480e+03 | 8.9500e+00 | pass |
| swiglu_output__fp8__block | single_stage | eedc63b2 | 59 | 0.980897 | 0.711611 | 1.6384e+04 | 4.2406e+02 | pass |
| baseline__fp32 | single_stage | e626d3e6 | 58 | 1.000000 | 1.000000 | 0.0000e+00 | 0.0000e+00 | pass |
| gemm1_accumulator__f16 | single_stage | e626d3e6 | 58 | 0.998069 | 0.939367 | 4.0960e+03 | 1.1857e+03 | pass |
| gemm1_operands__f16 | single_stage | e626d3e6 | 58 | 0.999473 | 0.983848 | 4.0960e+03 | 9.2835e+01 | pass |
| gemm1_output__f16 | single_stage | e626d3e6 | 58 | 0.999603 | 0.988464 | 4.0960e+03 | 2.0788e+02 | pass |
| gemm2_accumulator__bf16 | single_stage | e626d3e6 | 58 | 0.995803 | 0.864311 | 8.1920e+03 | 4.0236e+03 | pass |
| gemm2_operands__f16 | single_stage | e626d3e6 | 58 | 0.922128 | 0.913127 | nan | nan | pass |
| hidden_dequant__fp8__block | single_stage | e626d3e6 | 58 | 0.954826 | 0.510754 | 3.6864e+04 | 5.4487e+04 | pass |
| out_accumulator__bf16 | single_stage | e626d3e6 | 58 | 0.999175 | 0.973537 | 4.0960e+03 | 9.7000e+01 | pass |
| swiglu_input__f16 | single_stage | e626d3e6 | 58 | 0.999603 | 0.988464 | 4.0960e+03 | 2.0788e+02 | pass |
| swiglu_output__fp8__block | single_stage | e626d3e6 | 58 | 0.971398 | 0.566678 | 2.6112e+04 | 2.6673e+04 | pass |
| baseline__fp32 | single_stage | 74d7ff04 | 57 | 1.000000 | 1.000000 | 0.0000e+00 | 0.0000e+00 | pass |
| gemm1_accumulator__f16 | single_stage | 74d7ff04 | 57 | 0.998419 | 0.949875 | 2.0480e+03 | 9.5553e+02 | pass |
| gemm1_operands__f16 | single_stage | 74d7ff04 | 57 | 0.999572 | 0.986585 | 2.0480e+03 | 3.0698e+02 | pass |
| gemm1_output__f16 | single_stage | 74d7ff04 | 57 | 0.999677 | 0.990643 | 2.0480e+03 | 1.2900e+02 | pass |
| gemm2_accumulator__bf16 | single_stage | 74d7ff04 | 57 | 0.996897 | 0.889913 | 4.0960e+03 | 2.3030e+03 | pass |
| gemm2_operands__f16 | single_stage | 74d7ff04 | 57 | 0.982206 | 0.973454 | nan | nan | pass |
| hidden_dequant__fp8__block | single_stage | 74d7ff04 | 57 | 0.962776 | 0.589983 | 2.1248e+04 | 3.9625e+04 | pass |
| out_accumulator__bf16 | single_stage | 74d7ff04 | 57 | 0.999493 | 0.981169 | 4.0960e+03 | 5.1741e+01 | pass |
| swiglu_input__f16 | single_stage | 74d7ff04 | 57 | 0.999677 | 0.990643 | 2.0480e+03 | 1.2900e+02 | pass |
| swiglu_output__fp8__block | single_stage | 74d7ff04 | 57 | 0.976308 | 0.636041 | 1.3568e+04 | 1.6258e+04 | pass |
| baseline__fp32 | single_stage | 4822167c | 56 | 1.000000 | 1.000000 | 0.0000e+00 | 0.0000e+00 | pass |
| gemm1_accumulator__f16 | single_stage | 4822167c | 56 | 0.997990 | 0.938183 | 2.0480e+03 | 4.6370e+02 | pass |
| gemm1_operands__f16 | single_stage | 4822167c | 56 | 0.999454 | 0.983045 | 2.0480e+03 | 2.3172e+01 | pass |
| gemm1_output__f16 | single_stage | 4822167c | 56 | 0.999547 | 0.988349 | 2.0480e+03 | 1.4419e+01 | pass |
| gemm2_accumulator__bf16 | single_stage | 4822167c | 56 | 0.995713 | 0.862237 | 4.0960e+03 | 8.6161e+02 | pass |
| gemm2_operands__f16 | single_stage | 4822167c | 56 | 0.954944 | 0.944588 | nan | nan | pass |
| hidden_dequant__fp8__block | single_stage | 4822167c | 56 | 0.953382 | 0.493919 | 2.2528e+04 | 2.9617e+03 | pass |
| out_accumulator__bf16 | single_stage | 4822167c | 56 | 0.999210 | 0.975606 | 4.0960e+03 | 1.7655e+02 | pass |
| swiglu_input__f16 | single_stage | 4822167c | 56 | 0.999547 | 0.988349 | 2.0480e+03 | 1.4419e+01 | pass |
| swiglu_output__fp8__block | single_stage | 4822167c | 56 | 0.970295 | 0.551140 | 1.5360e+04 | 5.2768e+03 | pass |
| baseline__fp32 | single_stage | 81955b1e | 55 | 1.000000 | 1.000000 | 0.0000e+00 | 0.0000e+00 | pass |
| gemm1_accumulator__f16 | single_stage | 81955b1e | 55 | 0.998557 | 0.952113 | 2.0480e+03 | 1.1958e+02 | pass |
| gemm1_operands__f16 | single_stage | 81955b1e | 55 | 0.999637 | 0.987330 | 2.0480e+03 | 1.0596e+02 | pass |
| gemm1_output__f16 | single_stage | 81955b1e | 55 | 0.999721 | 0.990810 | 2.0480e+03 | 7.9080e+01 | pass |
| gemm2_accumulator__bf16 | single_stage | 81955b1e | 55 | 0.996931 | 0.894082 | 4.0960e+03 | 4.3865e+02 | pass |
| gemm2_operands__f16 | single_stage | 81955b1e | 55 | 0.981590 | 0.973103 | nan | nan | pass |
| hidden_dequant__fp8__block | single_stage | 81955b1e | 55 | 0.964197 | 0.606311 | 2.2528e+04 | 1.5053e+03 | pass |
| out_accumulator__bf16 | single_stage | 81955b1e | 55 | 0.999477 | 0.982394 | 2.0480e+03 | 4.3667e+01 | pass |
| swiglu_input__f16 | single_stage | 81955b1e | 55 | 0.999721 | 0.990810 | 2.0480e+03 | 7.9080e+01 | pass |
| swiglu_output__fp8__block | single_stage | 81955b1e | 55 | 0.977230 | 0.651025 | 1.4336e+04 | 4.3428e+03 | pass |
| baseline__fp32 | single_stage | 76010cb4 | 54 | 1.000000 | 1.000000 | 0.0000e+00 | 0.0000e+00 | pass |
| gemm1_accumulator__f16 | single_stage | 76010cb4 | 54 | 0.998708 | 0.960772 | 4.0960e+03 | 5.5534e+03 | pass |
| gemm1_operands__f16 | single_stage | 76010cb4 | 54 | 0.999669 | 0.988909 | 2.0480e+03 | 1.8518e+02 | pass |
| gemm1_output__f16 | single_stage | 76010cb4 | 54 | 0.999760 | 0.992286 | 2.0480e+03 | 2.0361e+02 | pass |
| gemm2_accumulator__bf16 | single_stage | 76010cb4 | 54 | 0.997354 | 0.913967 | 6.1440e+03 | 8.2330e+02 | pass |
| gemm2_operands__f16 | single_stage | 76010cb4 | 54 | 0.981252 | 0.974232 | nan | nan | pass |
| hidden_dequant__fp8__block | single_stage | 76010cb4 | 54 | 0.969982 | 0.676042 | 2.8672e+04 | 7.7453e+04 | pass |
| out_accumulator__bf16 | single_stage | 76010cb4 | 54 | 0.999571 | 0.985636 | 2.0480e+03 | 6.6385e+01 | pass |
| swiglu_input__f16 | single_stage | 76010cb4 | 54 | 0.999760 | 0.992286 | 2.0480e+03 | 2.0361e+02 | pass |
| swiglu_output__fp8__block | single_stage | 76010cb4 | 54 | 0.981016 | 0.713177 | 1.7408e+04 | 1.9536e+04 | pass |
| baseline__fp32 | single_stage | fc378037 | 53 | 1.000000 | 1.000000 | 0.0000e+00 | 0.0000e+00 | pass |
| gemm1_accumulator__f16 | single_stage | fc378037 | 53 | 0.998479 | 0.953957 | 4.0960e+03 | 2.9302e+03 | pass |
| gemm1_operands__f16 | single_stage | fc378037 | 53 | 0.999610 | 0.987726 | 2.0480e+03 | 1.3002e+03 | pass |
| gemm1_output__f16 | single_stage | fc378037 | 53 | 0.999679 | 0.991098 | 2.0480e+03 | 1.0934e+03 | pass |
| gemm2_accumulator__bf16 | single_stage | fc378037 | 53 | 0.996847 | 0.896869 | 8.1920e+03 | 1.1713e+03 | pass |
| gemm2_operands__f16 | single_stage | fc378037 | 53 | 0.962038 | 0.954417 | nan | nan | pass |
| hidden_dequant__fp8__block | single_stage | fc378037 | 53 | 0.965654 | 0.622749 | 2.6624e+04 | 1.7403e+04 | pass |
| out_accumulator__bf16 | single_stage | fc378037 | 53 | 0.999397 | 0.981766 | 4.0960e+03 | 4.0714e+03 | pass |
| swiglu_input__f16 | single_stage | fc378037 | 53 | 0.999679 | 0.991098 | 2.0480e+03 | 1.0934e+03 | pass |
| swiglu_output__fp8__block | single_stage | fc378037 | 53 | 0.978918 | 0.667890 | 1.7408e+04 | 3.0107e+04 | pass |
| baseline__fp32 | single_stage | f7d6ac7c | 52 | 1.000000 | 1.000000 | 0.0000e+00 | 0.0000e+00 | pass |
| gemm1_accumulator__f16 | single_stage | f7d6ac7c | 52 | 0.999115 | 0.970714 | 2.0480e+03 | 3.2467e+01 | pass |
| gemm1_operands__f16 | single_stage | f7d6ac7c | 52 | 0.999769 | 0.992118 | 2.0480e+03 | 9.1000e+00 | pass |
| gemm1_output__f16 | single_stage | f7d6ac7c | 52 | 0.999818 | 0.994387 | 2.0480e+03 | 6.4516e+00 | pass |
| gemm2_accumulator__bf16 | single_stage | f7d6ac7c | 52 | 0.998208 | 0.936051 | 4.6080e+03 | 5.1400e+01 | pass |
| gemm2_operands__f16 | single_stage | f7d6ac7c | 52 | 0.961431 | 0.956822 | nan | nan | pass |
| hidden_dequant__fp8__block | single_stage | f7d6ac7c | 52 | 0.977456 | 0.757947 | 2.0480e+04 | 8.0616e+02 | pass |
| out_accumulator__bf16 | single_stage | f7d6ac7c | 52 | 0.999748 | 0.991635 | 2.0480e+03 | 1.5742e+01 | pass |
| swiglu_input__f16 | single_stage | f7d6ac7c | 52 | 0.999818 | 0.994387 | 2.0480e+03 | 6.4516e+00 | pass |
| swiglu_output__fp8__block | single_stage | f7d6ac7c | 52 | 0.986366 | 0.788242 | 1.4848e+04 | 4.1060e+02 | pass |
| gemm1_accumulator__f16 | cumulative | b8f4f012 | 7 | 0.998884 | 0.971042 | 2.0480e+03 | 1.5390e+01 | pass |
| gemm1_accumulator__f16 | cumulative | e05c6c03 | 1 | 0.996791 | 0.901925 | 2.0480e+03 | 7.1786e+00 | pass |
| gemm1_accumulator__f16 | cumulative | 6230e838 | 32 | 0.998666 | 0.957890 | 2.0480e+03 | 3.2416e+02 | pass |
| gemm1_accumulator__f16 | cumulative | 8f1ff9f1 | 80 | 0.998024 | 0.938299 | 4.0960e+03 | 1.0204e+03 | pass |
| gemm1_accumulator__f16 | cumulative | 1a4c6ba1 | 901 | 0.997903 | 0.935057 | 4.0960e+03 | 1.1216e+04 | pass |
| gemm1_accumulator__f16 | cumulative | a7c2bcfd | 16 | 0.997986 | 0.935477 | 4.0960e+03 | 4.8612e+01 | pass |
| gemm1_accumulator__f16 | cumulative | 2e69caee | 15 | 0.999154 | 0.974730 | 2.0480e+03 | 1.0482e+01 | pass |
| gemm1_accumulator__f16 | cumulative | 8cba5890 | 14 | 0.998645 | 0.956513 | 2.0480e+03 | 1.5119e+02 | pass |
| gemm1_accumulator__f16 | cumulative | 5e8dc11c | 14107 | 0.998119 | 0.941357 | 4.0960e+03 | 1.1172e+08 | pass |
| gemm1_accumulator__f16 | cumulative | 58a34f27 | 11948 | 0.998562 | 0.955233 | 4.0960e+03 | 1.3845e+05 | pass |
| gemm1_accumulator__f16 | cumulative | 5eadab1e | 62 | 0.998546 | 0.951986 | 4.0960e+03 | 3.6893e+03 | pass |
| gemm1_accumulator__f16 | cumulative | eedc63b2 | 59 | 0.998761 | 0.960876 | 2.0480e+03 | 1.8402e+02 | pass |
| gemm1_accumulator__f16 | cumulative | e626d3e6 | 58 | 0.998162 | 0.938732 | 4.0960e+03 | 1.3964e+02 | pass |
| gemm1_accumulator__f16 | cumulative | 74d7ff04 | 57 | 0.998446 | 0.949921 | 4.0960e+03 | 1.3471e+02 | pass |
| gemm1_accumulator__f16 | cumulative | 4822167c | 56 | 0.998139 | 0.938516 | 2.0480e+03 | 2.7196e+02 | pass |
| gemm1_accumulator__f16 | cumulative | 81955b1e | 55 | 0.998481 | 0.953186 | 2.0480e+03 | 1.5276e+04 | pass |
| gemm1_accumulator__f16 | cumulative | 76010cb4 | 54 | 0.998760 | 0.961310 | 4.0960e+03 | 9.2260e+02 | pass |
| gemm1_accumulator__f16 | cumulative | fc378037 | 53 | 0.998523 | 0.952351 | 2.0480e+03 | 8.0231e+01 | pass |
| gemm1_accumulator__f16 | cumulative | f7d6ac7c | 52 | 0.999141 | 0.971613 | 2.0480e+03 | 3.0291e+02 | pass |
| gemm1_accumulator__f16+gemm1_operands__f16 | cumulative | b8f4f012 | 7 | 0.999721 | 0.991629 | 2.0480e+03 | 1.8178e+01 | pass |
| gemm1_accumulator__f16+gemm1_operands__f16 | cumulative | e05c6c03 | 1 | 0.999302 | 0.974609 | 2.0480e+03 | 2.4667e+00 | pass |
| gemm1_accumulator__f16+gemm1_operands__f16 | cumulative | 6230e838 | 32 | 0.999717 | 0.988852 | 2.0480e+03 | 2.8754e+01 | pass |
| gemm1_accumulator__f16+gemm1_operands__f16 | cumulative | 8f1ff9f1 | 80 | 0.999454 | 0.983418 | 4.0960e+03 | 5.5303e+01 | pass |
| gemm1_accumulator__f16+gemm1_operands__f16 | cumulative | 1a4c6ba1 | 901 | 0.999449 | 0.982694 | 4.0960e+03 | 2.9770e+03 | pass |
| gemm1_accumulator__f16+gemm1_operands__f16 | cumulative | a7c2bcfd | 16 | 0.999651 | 0.982657 | 2.0480e+03 | 1.7393e+02 | pass |
| gemm1_accumulator__f16+gemm1_operands__f16 | cumulative | 2e69caee | 15 | 0.999693 | 0.992374 | 2.0480e+03 | 6.5376e+00 | pass |
| gemm1_accumulator__f16+gemm1_operands__f16 | cumulative | 8cba5890 | 14 | 0.999611 | 0.988750 | 2.0480e+03 | 3.1250e+00 | pass |
| gemm1_accumulator__f16+gemm1_operands__f16 | cumulative | 5e8dc11c | 14107 | 0.999499 | 0.984265 | 8.1920e+03 | 7.8898e+03 | pass |
| gemm1_accumulator__f16+gemm1_operands__f16 | cumulative | 58a34f27 | 11948 | 0.999614 | 0.987918 | 4.0960e+03 | 1.2202e+04 | pass |
| gemm1_accumulator__f16+gemm1_operands__f16 | cumulative | 5eadab1e | 62 | 0.999602 | 0.987145 | 2.0480e+03 | 4.4294e+01 | pass |
| gemm1_accumulator__f16+gemm1_operands__f16 | cumulative | eedc63b2 | 59 | 0.999678 | 0.989331 | 2.0480e+03 | 3.4744e+02 | pass |
| gemm1_accumulator__f16+gemm1_operands__f16 | cumulative | e626d3e6 | 58 | 0.999490 | 0.983716 | 2.0480e+03 | 3.5036e+01 | pass |
| gemm1_accumulator__f16+gemm1_operands__f16 | cumulative | 74d7ff04 | 57 | 0.999594 | 0.986661 | 2.0480e+03 | 6.1977e+02 | pass |
| gemm1_accumulator__f16+gemm1_operands__f16 | cumulative | 4822167c | 56 | 0.999539 | 0.983595 | 2.0480e+03 | 5.2981e+01 | pass |
| gemm1_accumulator__f16+gemm1_operands__f16 | cumulative | 81955b1e | 55 | 0.999609 | 0.987267 | 4.0960e+03 | 4.1878e+02 | pass |
| gemm1_accumulator__f16+gemm1_operands__f16 | cumulative | 76010cb4 | 54 | 0.999625 | 0.989170 | 2.0480e+03 | 7.3035e+01 | pass |
| gemm1_accumulator__f16+gemm1_operands__f16 | cumulative | fc378037 | 53 | 0.999576 | 0.987486 | 2.0480e+03 | 1.3200e+02 | pass |
| gemm1_accumulator__f16+gemm1_operands__f16 | cumulative | f7d6ac7c | 52 | 0.999737 | 0.992268 | 2.0480e+03 | 2.3927e+01 | pass |
| gemm1_accumulator__f16+gemm1_operands__f16+gemm1_output__f16 | cumulative | b8f4f012 | 7 | 0.999721 | 0.990812 | 2.0480e+03 | 1.3817e+01 | pass |
| gemm1_accumulator__f16+gemm1_operands__f16+gemm1_output__f16 | cumulative | e05c6c03 | 1 | 0.999023 | 0.963588 | 1.0240e+03 | 7.4151e+00 | pass |
| gemm1_accumulator__f16+gemm1_operands__f16+gemm1_output__f16 | cumulative | 6230e838 | 32 | 0.999581 | 0.986302 | 4.0960e+03 | 8.5224e+00 | pass |
| gemm1_accumulator__f16+gemm1_operands__f16+gemm1_output__f16 | cumulative | 8f1ff9f1 | 80 | 0.999405 | 0.979848 | 2.0480e+03 | 2.8326e+02 | pass |
| gemm1_accumulator__f16+gemm1_operands__f16+gemm1_output__f16 | cumulative | 1a4c6ba1 | 901 | 0.999314 | 0.978735 | 4.0960e+03 | 2.1777e+03 | pass |
| gemm1_accumulator__f16+gemm1_operands__f16+gemm1_output__f16 | cumulative | a7c2bcfd | 16 | 0.999311 | 0.979126 | 2.0480e+03 | 1.0553e+02 | pass |
| gemm1_accumulator__f16+gemm1_operands__f16+gemm1_output__f16 | cumulative | 2e69caee | 15 | 0.999721 | 0.991006 | 2.0480e+03 | 7.1772e+00 | pass |
| gemm1_accumulator__f16+gemm1_operands__f16+gemm1_output__f16 | cumulative | 8cba5890 | 14 | 0.999512 | 0.985660 | 2.0480e+03 | 2.5000e+01 | pass |
| gemm1_accumulator__f16+gemm1_operands__f16+gemm1_output__f16 | cumulative | 5e8dc11c | 14107 | 0.999388 | 0.980734 | 4.0960e+03 | 5.5040e+04 | pass |
| gemm1_accumulator__f16+gemm1_operands__f16+gemm1_output__f16 | cumulative | 58a34f27 | 11948 | 0.999528 | 0.985227 | 4.0960e+03 | 4.5046e+03 | pass |
| gemm1_accumulator__f16+gemm1_operands__f16+gemm1_output__f16 | cumulative | 5eadab1e | 62 | 0.999541 | 0.984352 | 2.0480e+03 | 5.4257e+01 | pass |
| gemm1_accumulator__f16+gemm1_operands__f16+gemm1_output__f16 | cumulative | eedc63b2 | 59 | 0.999577 | 0.987085 | 2.0480e+03 | 3.4074e+03 | pass |
| gemm1_accumulator__f16+gemm1_operands__f16+gemm1_output__f16 | cumulative | e626d3e6 | 58 | 0.999449 | 0.980548 | 4.0960e+03 | 1.9098e+02 | pass |
| gemm1_accumulator__f16+gemm1_operands__f16+gemm1_output__f16 | cumulative | 74d7ff04 | 57 | 0.999459 | 0.983636 | 2.0480e+03 | 6.6806e+02 | pass |
| gemm1_accumulator__f16+gemm1_operands__f16+gemm1_output__f16 | cumulative | 4822167c | 56 | 0.999340 | 0.979607 | 2.0480e+03 | 1.7859e+02 | pass |
| gemm1_accumulator__f16+gemm1_operands__f16+gemm1_output__f16 | cumulative | 81955b1e | 55 | 0.999477 | 0.984106 | 2.0480e+03 | 1.2250e+09 | pass |
| gemm1_accumulator__f16+gemm1_operands__f16+gemm1_output__f16 | cumulative | 76010cb4 | 54 | 0.999620 | 0.986834 | 2.0480e+03 | 1.6516e+02 | pass |
| gemm1_accumulator__f16+gemm1_operands__f16+gemm1_output__f16 | cumulative | fc378037 | 53 | 0.999510 | 0.984720 | 2.0480e+03 | 2.9852e+01 | pass |
| gemm1_accumulator__f16+gemm1_operands__f16+gemm1_output__f16 | cumulative | f7d6ac7c | 52 | 0.999694 | 0.990422 | 2.0480e+03 | 9.2924e+00 | pass |
| gemm1_accumulator__f16+gemm1_operands__f16+gemm1_output__f16+gemm2_accumulator__bf16 | cumulative | b8f4f012 | 7 | 0.997828 | 0.924845 | 4.0960e+03 | 1.3621e+02 | pass |
| gemm1_accumulator__f16+gemm1_operands__f16+gemm1_output__f16+gemm2_accumulator__bf16 | cumulative | e05c6c03 | 1 | 0.990234 | 0.747907 | 4.0960e+03 | 6.5929e+02 | pass |
| gemm1_accumulator__f16+gemm1_operands__f16+gemm1_output__f16+gemm2_accumulator__bf16 | cumulative | 6230e838 | 32 | 0.996700 | 0.900016 | 4.1920e+03 | 1.3517e+02 | pass |
| gemm1_accumulator__f16+gemm1_operands__f16+gemm1_output__f16+gemm2_accumulator__bf16 | cumulative | 8f1ff9f1 | 80 | 0.995691 | 0.859923 | 6.1440e+03 | 1.8737e+02 | pass |
| gemm1_accumulator__f16+gemm1_operands__f16+gemm1_output__f16+gemm2_accumulator__bf16 | cumulative | 1a4c6ba1 | 901 | 0.995476 | 0.851011 | 8.1920e+03 | 5.5317e+03 | pass |
| gemm1_accumulator__f16+gemm1_operands__f16+gemm1_output__f16+gemm2_accumulator__bf16 | cumulative | a7c2bcfd | 16 | 0.995858 | 0.855974 | 4.0960e+03 | 3.4385e+02 | pass |
| gemm1_accumulator__f16+gemm1_operands__f16+gemm1_output__f16+gemm2_accumulator__bf16 | cumulative | 2e69caee | 15 | 0.997982 | 0.936198 | 4.0960e+03 | 4.5138e+01 | pass |
| gemm1_accumulator__f16+gemm1_operands__f16+gemm1_output__f16+gemm2_accumulator__bf16 | cumulative | 8cba5890 | 14 | 0.996771 | 0.895508 | 4.0960e+03 | 2.0678e+02 | pass |
| gemm1_accumulator__f16+gemm1_operands__f16+gemm1_output__f16+gemm2_accumulator__bf16 | cumulative | 5e8dc11c | 14107 | 0.996153 | 0.870359 | 1.2288e+04 | 4.1869e+04 | pass |
| gemm1_accumulator__f16+gemm1_operands__f16+gemm1_output__f16+gemm2_accumulator__bf16 | cumulative | 58a34f27 | 11948 | 0.997035 | 0.900138 | 8.1920e+03 | 1.5213e+05 | pass |
| gemm1_accumulator__f16+gemm1_operands__f16+gemm1_output__f16+gemm2_accumulator__bf16 | cumulative | 5eadab1e | 62 | 0.996940 | 0.895357 | 6.1440e+03 | 1.9300e+02 | pass |
| gemm1_accumulator__f16+gemm1_operands__f16+gemm1_output__f16+gemm2_accumulator__bf16 | cumulative | eedc63b2 | 59 | 0.997486 | 0.912568 | 5.1200e+03 | 1.1540e+02 | pass |
| gemm1_accumulator__f16+gemm1_operands__f16+gemm1_output__f16+gemm2_accumulator__bf16 | cumulative | e626d3e6 | 58 | 0.995817 | 0.862011 | 8.1920e+03 | 1.5204e+02 | pass |
| gemm1_accumulator__f16+gemm1_operands__f16+gemm1_output__f16+gemm2_accumulator__bf16 | cumulative | 74d7ff04 | 57 | 0.996804 | 0.887194 | 5.1200e+03 | 1.6456e+03 | pass |
| gemm1_accumulator__f16+gemm1_operands__f16+gemm1_output__f16+gemm2_accumulator__bf16 | cumulative | 4822167c | 56 | 0.995842 | 0.859661 | 5.3760e+03 | 1.2856e+03 | pass |
| gemm1_accumulator__f16+gemm1_operands__f16+gemm1_output__f16+gemm2_accumulator__bf16 | cumulative | 81955b1e | 55 | 0.996903 | 0.891898 | 6.1440e+03 | 2.2428e+02 | pass |
| gemm1_accumulator__f16+gemm1_operands__f16+gemm1_output__f16+gemm2_accumulator__bf16 | cumulative | 76010cb4 | 54 | 0.997460 | 0.912187 | 6.1440e+03 | 6.0493e+03 | pass |
| gemm1_accumulator__f16+gemm1_operands__f16+gemm1_output__f16+gemm2_accumulator__bf16 | cumulative | fc378037 | 53 | 0.996826 | 0.895495 | 8.1920e+03 | 3.2607e+03 | pass |
| gemm1_accumulator__f16+gemm1_operands__f16+gemm1_output__f16+gemm2_accumulator__bf16 | cumulative | f7d6ac7c | 52 | 0.998237 | 0.935351 | 4.0960e+03 | 1.1669e+02 | pass |
| gemm1_accumulator__f16+gemm1_operands__f16+gemm1_output__f16+gemm2_accumulator__bf16+gemm2_operands__f16 | cumulative | b8f4f012 | 7 | 0.999641 | 0.988859 | 2.0480e+03 | 3.4964e+02 | pass |
| gemm1_accumulator__f16+gemm1_operands__f16+gemm1_output__f16+gemm2_accumulator__bf16+gemm2_operands__f16 | cumulative | e05c6c03 | 1 | 0.999721 | 0.962891 | 2.0480e+03 | 6.5873e-01 | pass |
| gemm1_accumulator__f16+gemm1_operands__f16+gemm1_output__f16+gemm2_accumulator__bf16+gemm2_operands__f16 | cumulative | 6230e838 | 32 | 0.999529 | 0.983795 | 2.0480e+03 | 9.2714e+01 | pass |
| gemm1_accumulator__f16+gemm1_operands__f16+gemm1_output__f16+gemm2_accumulator__bf16+gemm2_operands__f16 | cumulative | 8f1ff9f1 | 80 | 0.952359 | 0.931794 | nan | nan | pass |
| gemm1_accumulator__f16+gemm1_operands__f16+gemm1_output__f16+gemm2_accumulator__bf16+gemm2_operands__f16 | cumulative | 1a4c6ba1 | 901 | 0.942020 | 0.920586 | nan | nan | pass |
| gemm1_accumulator__f16+gemm1_operands__f16+gemm1_output__f16+gemm2_accumulator__bf16+gemm2_operands__f16 | cumulative | a7c2bcfd | 16 | 0.874555 | 0.855983 | nan | nan | global_drift |
| gemm1_accumulator__f16+gemm1_operands__f16+gemm1_output__f16+gemm2_accumulator__bf16+gemm2_operands__f16 | cumulative | 2e69caee | 15 | 0.933082 | 0.925270 | nan | nan | pass |
| gemm1_accumulator__f16+gemm1_operands__f16+gemm1_output__f16+gemm2_accumulator__bf16+gemm2_operands__f16 | cumulative | 8cba5890 | 14 | 0.999382 | 0.984206 | 2.0480e+03 | 1.6205e+01 | pass |
| gemm1_accumulator__f16+gemm1_operands__f16+gemm1_output__f16+gemm2_accumulator__bf16+gemm2_operands__f16 | cumulative | 5e8dc11c | 14107 | 0.957165 | 0.937421 | nan | nan | pass |
| gemm1_accumulator__f16+gemm1_operands__f16+gemm1_output__f16+gemm2_accumulator__bf16+gemm2_operands__f16 | cumulative | 58a34f27 | 11948 | 0.967788 | 0.952654 | nan | nan | pass |
| gemm1_accumulator__f16+gemm1_operands__f16+gemm1_output__f16+gemm2_accumulator__bf16+gemm2_operands__f16 | cumulative | 5eadab1e | 62 | 0.983335 | 0.966192 | nan | nan | pass |
| gemm1_accumulator__f16+gemm1_operands__f16+gemm1_output__f16+gemm2_accumulator__bf16+gemm2_operands__f16 | cumulative | eedc63b2 | 59 | 0.948715 | 0.935772 | nan | nan | pass |
| gemm1_accumulator__f16+gemm1_operands__f16+gemm1_output__f16+gemm2_accumulator__bf16+gemm2_operands__f16 | cumulative | e626d3e6 | 58 | 0.999276 | 0.977549 | 4.0960e+03 | 5.8797e+01 | pass |
| gemm1_accumulator__f16+gemm1_operands__f16+gemm1_output__f16+gemm2_accumulator__bf16+gemm2_operands__f16 | cumulative | 74d7ff04 | 57 | 0.955675 | 0.938790 | nan | nan | pass |
| gemm1_accumulator__f16+gemm1_operands__f16+gemm1_output__f16+gemm2_accumulator__bf16+gemm2_operands__f16 | cumulative | 4822167c | 56 | 0.981488 | 0.959373 | nan | nan | pass |
| gemm1_accumulator__f16+gemm1_operands__f16+gemm1_output__f16+gemm2_accumulator__bf16+gemm2_operands__f16 | cumulative | 81955b1e | 55 | 0.999424 | 0.981884 | 2.0480e+03 | 1.5244e+02 | pass |
| gemm1_accumulator__f16+gemm1_operands__f16+gemm1_output__f16+gemm2_accumulator__bf16+gemm2_operands__f16 | cumulative | 76010cb4 | 54 | 0.962550 | 0.949108 | nan | nan | pass |
| gemm1_accumulator__f16+gemm1_operands__f16+gemm1_output__f16+gemm2_accumulator__bf16+gemm2_operands__f16 | cumulative | fc378037 | 53 | 0.961814 | 0.946171 | nan | nan | pass |
| gemm1_accumulator__f16+gemm1_operands__f16+gemm1_output__f16+gemm2_accumulator__bf16+gemm2_operands__f16 | cumulative | f7d6ac7c | 52 | 0.980485 | 0.970443 | nan | nan | pass |