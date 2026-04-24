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
| run_stage | stage1_fp8_followup |
| panel_size | 19 |
| panel_indices | 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18 |
| fp8_target_stages | hidden_dequant,swiglu_output |
| n_candidates | 6 |
| n_survivors | 6 |
| contest_safe_survivor_count | 6 |
| contest_safe_survivors | swiglu_output__fp8__block, swiglu_output__fp8__row, swiglu_output__fp8__tensor, hidden_dequant__fp8__block, hidden_dequant__fp8__row, hidden_dequant__fp8__tensor |
| strict_safe_survivor_count | 0 |
| strict_safe_survivors | - |
| contest_only_survivor_count | 6 |
| contest_only_survivors | hidden_dequant__fp8__tensor, hidden_dequant__fp8__row, hidden_dequant__fp8__block, swiglu_output__fp8__tensor, swiglu_output__fp8__row, swiglu_output__fp8__block |

## Survivor Summary

| category | count | candidates |
|---|---:|---|
| contest_safe_single_stage | 6 | swiglu_output__fp8__block, swiglu_output__fp8__row, swiglu_output__fp8__tensor, hidden_dequant__fp8__block, hidden_dequant__fp8__row, hidden_dequant__fp8__tensor |
| strict_safe_single_stage | 0 | - |
| contest_only_single_stage | 6 | hidden_dequant__fp8__tensor, hidden_dequant__fp8__row, hidden_dequant__fp8__block, swiglu_output__fp8__tensor, swiglu_output__fp8__row, swiglu_output__fp8__block |

## Stage Summary

| stage | best_safe_mode | scale | worst_matched_contest | worst_matched_strict | worst_rel | status |
|---|---|---|---:|---:|---:|---|
| hidden_dequant | fp8 | tensor | 0.924805 | 0.168108 | 3.6800e+11 | safe |
| swiglu_output | fp8 | tensor | 0.942941 | 0.226981 | 3.5040e+11 | safe |

## Cumulative Safe Frontier

| order | candidate | worst_matched_contest | worst_matched_strict | worst_rel | status | kept |
|---|---|---:|---:|---:|---|---|
| 1 | hidden_dequant__fp8__tensor | 0.920898 | 0.163504 | 1.0857e+06 | safe | yes |
| 2 | hidden_dequant__fp8__tensor+swiglu_output__fp8__tensor | 0.900251 | 0.135045 | 1.7500e+10 | safe | yes |

## BF16/F16 Margin

| stage | preferred | bf16_contest | f16_contest | bf16_strict | f16_strict | |
|---|---|---:|---:|---:|---:|---|

## Pairwise Summary

| pair | worst_matched_contest | worst_matched_strict | worst_rel | status |
|---|---:|---:|---:|---|

## Stress Summary

| candidate | worst_workload | worst_seq_len | worst_matched_contest | worst_matched_strict | status |
|---|---|---:|---:|---:|---|

## Promotion Summary

| category | candidates |
|---|---|
| bf16_f16_survivors | swiglu_output__fp8__block, swiglu_output__fp8__row, swiglu_output__fp8__tensor, hidden_dequant__fp8__block, hidden_dequant__fp8__row, hidden_dequant__fp8__tensor |
| strict_survivors | - |
| pairwise_shortlist | - |

## Sampled Results

| candidate | phase | workload | seq_len | matched_contest | matched_strict | max_abs | max_rel | failure |
|---|---|---|---:|---:|---:|---:|---:|---|
| baseline__fp32 | single_stage | b8f4f012 | 7 | 1.000000 | 1.000000 | 0.0000e+00 | 0.0000e+00 | pass |
| hidden_dequant__fp8__tensor | single_stage | b8f4f012 | 7 | 0.977419 | 0.760702 | 2.1504e+04 | 6.7300e+02 | pass |
| hidden_dequant__fp8__row | single_stage | b8f4f012 | 7 | 0.977938 | 0.761898 | 2.6240e+04 | 5.8780e+02 | pass |
| hidden_dequant__fp8__block | single_stage | b8f4f012 | 7 | 0.979173 | 0.762875 | 2.2016e+04 | 5.8780e+02 | pass |
| swiglu_output__fp8__tensor | single_stage | b8f4f012 | 7 | 0.984734 | 0.780891 | 1.6384e+04 | 1.1466e+03 | pass |
| swiglu_output__fp8__row | single_stage | b8f4f012 | 7 | 0.984734 | 0.780891 | 1.6384e+04 | 1.1466e+03 | pass |
| swiglu_output__fp8__block | single_stage | b8f4f012 | 7 | 0.986109 | 0.789581 | 1.6064e+04 | 4.3100e+02 | pass |
| baseline__fp32 | single_stage | e05c6c03 | 1 | 1.000000 | 1.000000 | 0.0000e+00 | 0.0000e+00 | pass |
| hidden_dequant__fp8__tensor | single_stage | e05c6c03 | 1 | 0.924805 | 0.168108 | 1.7664e+04 | 5.7073e+02 | pass |
| hidden_dequant__fp8__row | single_stage | e05c6c03 | 1 | 0.924805 | 0.168108 | 1.7664e+04 | 5.7073e+02 | pass |
| hidden_dequant__fp8__block | single_stage | e05c6c03 | 1 | 0.929967 | 0.170619 | 1.3312e+04 | 8.2600e+01 | pass |
| swiglu_output__fp8__tensor | single_stage | e05c6c03 | 1 | 0.942941 | 0.226981 | 1.0240e+04 | 2.6353e+02 | pass |
| swiglu_output__fp8__row | single_stage | e05c6c03 | 1 | 0.942941 | 0.226981 | 1.0240e+04 | 2.6353e+02 | pass |
| swiglu_output__fp8__block | single_stage | e05c6c03 | 1 | 0.951451 | 0.257533 | 9.2160e+03 | 7.6700e+02 | pass |
| baseline__fp32 | single_stage | 6230e838 | 32 | 1.000000 | 1.000000 | 0.0000e+00 | 0.0000e+00 | pass |
| hidden_dequant__fp8__tensor | single_stage | 6230e838 | 32 | 0.968523 | 0.659925 | 2.2528e+04 | 1.4211e+04 | pass |
| hidden_dequant__fp8__row | single_stage | 6230e838 | 32 | 0.969195 | 0.661155 | 2.2528e+04 | 2.8644e+03 | pass |
| hidden_dequant__fp8__block | single_stage | 6230e838 | 32 | 0.969339 | 0.660278 | 2.2528e+04 | 4.7176e+03 | pass |
| swiglu_output__fp8__tensor | single_stage | 6230e838 | 32 | 0.978141 | 0.688538 | 1.5872e+04 | 9.6805e+03 | pass |
| swiglu_output__fp8__row | single_stage | 6230e838 | 32 | 0.978341 | 0.688642 | 1.7408e+04 | 5.8017e+03 | pass |
| swiglu_output__fp8__block | single_stage | 6230e838 | 32 | 0.980556 | 0.699428 | 1.5360e+04 | 1.7740e+03 | pass |
| baseline__fp32 | single_stage | 8f1ff9f1 | 80 | 1.000000 | 1.000000 | 0.0000e+00 | 0.0000e+00 | pass |
| hidden_dequant__fp8__tensor | single_stage | 8f1ff9f1 | 80 | 0.953392 | 0.498295 | 2.5600e+04 | 1.3829e+04 | pass |
| hidden_dequant__fp8__row | single_stage | 8f1ff9f1 | 80 | 0.952785 | 0.499194 | 2.4064e+04 | 5.8911e+03 | pass |
| hidden_dequant__fp8__block | single_stage | 8f1ff9f1 | 80 | 0.954609 | 0.500256 | 2.1120e+04 | 5.8127e+03 | pass |
| swiglu_output__fp8__tensor | single_stage | 8f1ff9f1 | 80 | 0.966818 | 0.537720 | 2.2016e+04 | 2.8276e+03 | pass |
| swiglu_output__fp8__row | single_stage | 8f1ff9f1 | 80 | 0.967397 | 0.539835 | 1.7920e+04 | 2.0271e+03 | pass |
| swiglu_output__fp8__block | single_stage | 8f1ff9f1 | 80 | 0.970902 | 0.555816 | 1.5360e+04 | 1.5138e+03 | pass |
| baseline__fp32 | single_stage | 1a4c6ba1 | 901 | 1.000000 | 1.000000 | 0.0000e+00 | 0.0000e+00 | pass |
| hidden_dequant__fp8__tensor | single_stage | 1a4c6ba1 | 901 | 0.950303 | 0.473245 | 3.6864e+04 | 5.9711e+05 | pass |
| hidden_dequant__fp8__row | single_stage | 1a4c6ba1 | 901 | 0.950376 | 0.473458 | 4.0960e+04 | 4.3205e+05 | pass |
| hidden_dequant__fp8__block | single_stage | 1a4c6ba1 | 901 | 0.951833 | 0.474438 | 3.3520e+04 | 6.2138e+05 | pass |
| swiglu_output__fp8__tensor | single_stage | 1a4c6ba1 | 901 | 0.964874 | 0.514124 | 2.4576e+04 | 1.2622e+05 | pass |
| swiglu_output__fp8__row | single_stage | 1a4c6ba1 | 901 | 0.965720 | 0.517788 | 2.4576e+04 | 4.2478e+04 | pass |
| swiglu_output__fp8__block | single_stage | 1a4c6ba1 | 901 | 0.969237 | 0.533564 | 2.2528e+04 | 7.0998e+04 | pass |
| baseline__fp32 | single_stage | a7c2bcfd | 16 | 1.000000 | 1.000000 | 0.0000e+00 | 0.0000e+00 | pass |
| hidden_dequant__fp8__tensor | single_stage | a7c2bcfd | 16 | 0.950675 | 0.475839 | 1.8944e+04 | 9.1501e+02 | pass |
| hidden_dequant__fp8__row | single_stage | a7c2bcfd | 16 | 0.950718 | 0.475699 | 1.6896e+04 | 1.1539e+03 | pass |
| hidden_dequant__fp8__block | single_stage | a7c2bcfd | 16 | 0.952262 | 0.479780 | 2.0480e+04 | 1.1449e+03 | pass |
| swiglu_output__fp8__tensor | single_stage | a7c2bcfd | 16 | 0.965594 | 0.520142 | 1.4336e+04 | 5.7514e+02 | pass |
| swiglu_output__fp8__row | single_stage | a7c2bcfd | 16 | 0.965881 | 0.519871 | 1.5360e+04 | 5.9347e+02 | pass |
| swiglu_output__fp8__block | single_stage | a7c2bcfd | 16 | 0.969308 | 0.535200 | 1.0752e+04 | 6.5989e+02 | pass |
| baseline__fp32 | single_stage | 2e69caee | 15 | 1.000000 | 1.000000 | 0.0000e+00 | 0.0000e+00 | pass |
| hidden_dequant__fp8__tensor | single_stage | 2e69caee | 15 | 0.978404 | 0.775326 | 3.0208e+04 | 6.7609e+02 | pass |
| hidden_dequant__fp8__row | single_stage | 2e69caee | 15 | 0.978358 | 0.777920 | 2.4576e+04 | 4.9092e+02 | pass |
| hidden_dequant__fp8__block | single_stage | 2e69caee | 15 | 0.979818 | 0.777065 | 1.9840e+04 | 9.8284e+02 | pass |
| swiglu_output__fp8__tensor | single_stage | 2e69caee | 15 | 0.984691 | 0.793313 | 1.6384e+04 | 3.4735e+02 | pass |
| swiglu_output__fp8__row | single_stage | 2e69caee | 15 | 0.985286 | 0.796735 | 1.4336e+04 | 3.6542e+02 | pass |
| swiglu_output__fp8__block | single_stage | 2e69caee | 15 | 0.986542 | 0.801925 | 1.2800e+04 | 6.5938e+02 | pass |
| baseline__fp32 | single_stage | 8cba5890 | 14 | 1.000000 | 1.000000 | 0.0000e+00 | 0.0000e+00 | pass |
| hidden_dequant__fp8__tensor | single_stage | 8cba5890 | 14 | 0.966687 | 0.642050 | 2.6624e+04 | 9.6646e+02 | pass |
| hidden_dequant__fp8__row | single_stage | 8cba5890 | 14 | 0.965990 | 0.640994 | 2.6624e+04 | 6.8576e+02 | pass |
| hidden_dequant__fp8__block | single_stage | 8cba5890 | 14 | 0.966877 | 0.641661 | 2.4576e+04 | 2.7632e+03 | pass |
| swiglu_output__fp8__tensor | single_stage | 8cba5890 | 14 | 0.976543 | 0.672164 | 1.5360e+04 | 2.5174e+02 | pass |
| swiglu_output__fp8__row | single_stage | 8cba5890 | 14 | 0.976303 | 0.671247 | 1.5360e+04 | 2.5174e+02 | pass |
| swiglu_output__fp8__block | single_stage | 8cba5890 | 14 | 0.979044 | 0.682607 | 1.6384e+04 | 3.0569e+02 | pass |
| baseline__fp32 | single_stage | 5e8dc11c | 14107 | 1.000000 | 1.000000 | 0.0000e+00 | 0.0000e+00 | pass |
| hidden_dequant__fp8__tensor | single_stage | 5e8dc11c | 14107 | 0.955134 | 0.521966 | 4.7104e+04 | 1.5431e+07 | pass |
| hidden_dequant__fp8__row | single_stage | 5e8dc11c | 14107 | 0.955019 | 0.522165 | 5.7472e+04 | 1.0223e+08 | pass |
| hidden_dequant__fp8__block | single_stage | 5e8dc11c | 14107 | 0.956327 | 0.522983 | 4.5056e+04 | 3.7372e+07 | pass |
| swiglu_output__fp8__tensor | single_stage | 5e8dc11c | 14107 | 0.968207 | 0.558884 | 3.4816e+04 | 2.5799e+07 | pass |
| swiglu_output__fp8__row | single_stage | 5e8dc11c | 14107 | 0.968855 | 0.561838 | 3.5456e+04 | 9.1922e+05 | pass |
| swiglu_output__fp8__block | single_stage | 5e8dc11c | 14107 | 0.972228 | 0.576576 | 2.8672e+04 | 4.2917e+07 | pass |
| baseline__fp32 | single_stage | 58a34f27 | 11948 | 1.000000 | 1.000000 | 0.0000e+00 | 0.0000e+00 | pass |
| hidden_dequant__fp8__tensor | single_stage | 58a34f27 | 11948 | 0.965592 | 0.633690 | 4.7104e+04 | 1.2729e+06 | pass |
| hidden_dequant__fp8__row | single_stage | 58a34f27 | 11948 | 0.965473 | 0.633793 | 3.4816e+04 | 1.1271e+06 | pass |
| hidden_dequant__fp8__block | single_stage | 58a34f27 | 11948 | 0.966576 | 0.634484 | 3.6864e+04 | 3.7478e+05 | pass |
| swiglu_output__fp8__tensor | single_stage | 58a34f27 | 11948 | 0.975646 | 0.662155 | 2.8672e+04 | 7.9624e+05 | pass |
| swiglu_output__fp8__row | single_stage | 58a34f27 | 11948 | 0.976178 | 0.664315 | 2.6624e+04 | 3.0139e+05 | pass |
| swiglu_output__fp8__block | single_stage | 58a34f27 | 11948 | 0.978740 | 0.675566 | 2.4576e+04 | 9.7007e+05 | pass |
| baseline__fp32 | single_stage | 5eadab1e | 62 | 1.000000 | 1.000000 | 0.0000e+00 | 0.0000e+00 | pass |
| hidden_dequant__fp8__tensor | single_stage | 5eadab1e | 62 | 0.963453 | 0.608740 | 2.6624e+04 | 1.8022e+05 | pass |
| hidden_dequant__fp8__row | single_stage | 5eadab1e | 62 | 0.963617 | 0.609771 | 1.9968e+04 | 3.8707e+05 | pass |
| hidden_dequant__fp8__block | single_stage | 5eadab1e | 62 | 0.964986 | 0.610534 | 2.0992e+04 | 2.6419e+05 | pass |
| swiglu_output__fp8__tensor | single_stage | 5eadab1e | 62 | 0.974261 | 0.640080 | 1.4336e+04 | 4.8947e+05 | pass |
| swiglu_output__fp8__row | single_stage | 5eadab1e | 62 | 0.974888 | 0.642290 | 1.4336e+04 | 9.4208e+05 | pass |
| swiglu_output__fp8__block | single_stage | 5eadab1e | 62 | 0.977382 | 0.653201 | 1.3824e+04 | 1.1878e+06 | pass |
| baseline__fp32 | single_stage | eedc63b2 | 59 | 1.000000 | 1.000000 | 0.0000e+00 | 0.0000e+00 | pass |
| hidden_dequant__fp8__tensor | single_stage | eedc63b2 | 59 | 0.969717 | 0.674535 | 2.1504e+04 | 6.6199e+03 | pass |
| hidden_dequant__fp8__row | single_stage | eedc63b2 | 59 | 0.969604 | 0.675774 | 2.4576e+04 | 7.2932e+03 | pass |
| hidden_dequant__fp8__block | single_stage | eedc63b2 | 59 | 0.970642 | 0.675767 | 2.2528e+04 | 1.4206e+04 | pass |
| swiglu_output__fp8__tensor | single_stage | eedc63b2 | 59 | 0.978721 | 0.701174 | 2.4064e+04 | 1.7080e+04 | pass |
| swiglu_output__fp8__row | single_stage | eedc63b2 | 59 | 0.978913 | 0.701605 | 2.4064e+04 | 1.8126e+04 | pass |
| swiglu_output__fp8__block | single_stage | eedc63b2 | 59 | 0.981337 | 0.711309 | 1.6384e+04 | 1.6211e+04 | pass |
| baseline__fp32 | single_stage | e626d3e6 | 58 | 1.000000 | 1.000000 | 0.0000e+00 | 0.0000e+00 | pass |
| hidden_dequant__fp8__tensor | single_stage | e626d3e6 | 58 | 0.953289 | 0.510336 | 2.9184e+04 | 3.6800e+11 | pass |
| hidden_dequant__fp8__row | single_stage | e626d3e6 | 58 | 0.953548 | 0.508967 | 2.5984e+04 | 4.4000e+10 | pass |
| hidden_dequant__fp8__block | single_stage | e626d3e6 | 58 | 0.954729 | 0.510567 | 2.8672e+04 | 3.4400e+11 | pass |
| swiglu_output__fp8__tensor | single_stage | e626d3e6 | 58 | 0.967114 | 0.548155 | 2.0224e+04 | 3.5040e+11 | pass |
| swiglu_output__fp8__row | single_stage | e626d3e6 | 58 | 0.967646 | 0.550942 | 2.0480e+04 | 5.2000e+10 | pass |
| swiglu_output__fp8__block | single_stage | e626d3e6 | 58 | 0.971463 | 0.565742 | 1.4848e+04 | 8.8000e+10 | pass |
| baseline__fp32 | single_stage | 74d7ff04 | 57 | 1.000000 | 1.000000 | 0.0000e+00 | 0.0000e+00 | pass |
| hidden_dequant__fp8__tensor | single_stage | 74d7ff04 | 57 | 0.962186 | 0.591021 | 2.6624e+04 | 2.7618e+03 | pass |
| hidden_dequant__fp8__row | single_stage | 74d7ff04 | 57 | 0.961924 | 0.591016 | 2.2528e+04 | 1.7560e+03 | pass |
| hidden_dequant__fp8__block | single_stage | 74d7ff04 | 57 | 0.962007 | 0.590015 | 2.5600e+04 | 1.0464e+03 | pass |
| swiglu_output__fp8__tensor | single_stage | 74d7ff04 | 57 | 0.972894 | 0.622107 | 1.7664e+04 | 1.8738e+03 | pass |
| swiglu_output__fp8__row | single_stage | 74d7ff04 | 57 | 0.973430 | 0.624165 | 1.9456e+04 | 1.4833e+03 | pass |
| swiglu_output__fp8__block | single_stage | 74d7ff04 | 57 | 0.975867 | 0.635564 | 1.5360e+04 | 1.9019e+03 | pass |
| baseline__fp32 | single_stage | 4822167c | 56 | 1.000000 | 1.000000 | 0.0000e+00 | 0.0000e+00 | pass |
| hidden_dequant__fp8__tensor | single_stage | 4822167c | 56 | 0.951969 | 0.492001 | 2.2656e+04 | 6.0747e+03 | pass |
| hidden_dequant__fp8__row | single_stage | 4822167c | 56 | 0.952457 | 0.492905 | 2.0224e+04 | 7.7473e+03 | pass |
| hidden_dequant__fp8__block | single_stage | 4822167c | 56 | 0.953175 | 0.493251 | 2.4064e+04 | 2.6305e+03 | pass |
| swiglu_output__fp8__tensor | single_stage | 4822167c | 56 | 0.966441 | 0.533417 | 1.8624e+04 | 7.9199e+03 | pass |
| swiglu_output__fp8__row | single_stage | 4822167c | 56 | 0.967111 | 0.535719 | 1.7408e+04 | 6.1109e+03 | pass |
| swiglu_output__fp8__block | single_stage | 4822167c | 56 | 0.970459 | 0.551588 | 1.7152e+04 | 5.0507e+03 | pass |
| baseline__fp32 | single_stage | 81955b1e | 55 | 1.000000 | 1.000000 | 0.0000e+00 | 0.0000e+00 | pass |
| hidden_dequant__fp8__tensor | single_stage | 81955b1e | 55 | 0.962934 | 0.604239 | 2.1136e+04 | 4.0140e+04 | pass |
| hidden_dequant__fp8__row | single_stage | 81955b1e | 55 | 0.962703 | 0.604794 | 2.5600e+04 | 1.2015e+05 | pass |
| hidden_dequant__fp8__block | single_stage | 81955b1e | 55 | 0.964139 | 0.605859 | 2.0480e+04 | 1.1741e+04 | pass |
| swiglu_output__fp8__tensor | single_stage | 81955b1e | 55 | 0.973965 | 0.636399 | 1.4336e+04 | 1.8296e+04 | pass |
| swiglu_output__fp8__row | single_stage | 81955b1e | 55 | 0.974419 | 0.638177 | 1.4336e+04 | 1.8296e+04 | pass |
| swiglu_output__fp8__block | single_stage | 81955b1e | 55 | 0.976573 | 0.649191 | 1.2576e+04 | 3.8774e+04 | pass |
| baseline__fp32 | single_stage | 76010cb4 | 54 | 1.000000 | 1.000000 | 0.0000e+00 | 0.0000e+00 | pass |
| hidden_dequant__fp8__tensor | single_stage | 76010cb4 | 54 | 0.969238 | 0.675239 | 3.0720e+04 | 5.9052e+04 | pass |
| hidden_dequant__fp8__row | single_stage | 76010cb4 | 54 | 0.969388 | 0.674595 | 2.8672e+04 | 2.2188e+04 | pass |
| hidden_dequant__fp8__block | single_stage | 76010cb4 | 54 | 0.970329 | 0.675125 | 3.6864e+04 | 8.8748e+04 | pass |
| swiglu_output__fp8__tensor | single_stage | 76010cb4 | 54 | 0.978335 | 0.702094 | 2.4576e+04 | 6.2465e+04 | pass |
| swiglu_output__fp8__row | single_stage | 76010cb4 | 54 | 0.978541 | 0.702120 | 2.4576e+04 | 5.0175e+04 | pass |
| swiglu_output__fp8__block | single_stage | 76010cb4 | 54 | 0.981329 | 0.712614 | 1.8432e+04 | 4.0106e+04 | pass |
| baseline__fp32 | single_stage | fc378037 | 53 | 1.000000 | 1.000000 | 0.0000e+00 | 0.0000e+00 | pass |
| hidden_dequant__fp8__tensor | single_stage | fc378037 | 53 | 0.964452 | 0.620970 | 2.6496e+04 | 3.6389e+03 | pass |
| hidden_dequant__fp8__row | single_stage | fc378037 | 53 | 0.964123 | 0.620907 | 2.4064e+04 | 5.8755e+03 | pass |
| hidden_dequant__fp8__block | single_stage | fc378037 | 53 | 0.965189 | 0.621865 | 2.1760e+04 | 4.9459e+03 | pass |
| swiglu_output__fp8__tensor | single_stage | fc378037 | 53 | 0.974365 | 0.650504 | 1.4848e+04 | 2.9889e+03 | pass |
| swiglu_output__fp8__row | single_stage | fc378037 | 53 | 0.975067 | 0.651817 | 1.6384e+04 | 1.2753e+03 | pass |
| swiglu_output__fp8__block | single_stage | fc378037 | 53 | 0.978058 | 0.664560 | 1.6384e+04 | 4.2751e+03 | pass |
| baseline__fp32 | single_stage | f7d6ac7c | 52 | 1.000000 | 1.000000 | 0.0000e+00 | 0.0000e+00 | pass |
| hidden_dequant__fp8__tensor | single_stage | f7d6ac7c | 52 | 0.977950 | 0.760841 | 1.5872e+04 | 6.9780e+04 | pass |
| hidden_dequant__fp8__row | single_stage | f7d6ac7c | 52 | 0.978051 | 0.760144 | 1.7024e+04 | 7.6191e+05 | pass |
| hidden_dequant__fp8__block | single_stage | f7d6ac7c | 52 | 0.978282 | 0.760007 | 1.6896e+04 | 3.8096e+05 | pass |
| swiglu_output__fp8__tensor | single_stage | f7d6ac7c | 52 | 0.984128 | 0.778012 | 1.1264e+04 | 1.6879e+05 | pass |
| swiglu_output__fp8__row | single_stage | f7d6ac7c | 52 | 0.984227 | 0.777875 | 1.1648e+04 | 1.6879e+05 | pass |
| swiglu_output__fp8__block | single_stage | f7d6ac7c | 52 | 0.985722 | 0.785419 | 1.0240e+04 | 4.0359e+05 | pass |
| hidden_dequant__fp8__tensor | cumulative | b8f4f012 | 7 | 0.976881 | 0.761041 | 1.5872e+04 | 6.9886e+02 | pass |
| hidden_dequant__fp8__tensor | cumulative | e05c6c03 | 1 | 0.920898 | 0.163504 | 1.3312e+04 | 4.0860e+02 | pass |
| hidden_dequant__fp8__tensor | cumulative | 6230e838 | 32 | 0.968266 | 0.661630 | 2.8672e+04 | 1.3457e+04 | pass |
| hidden_dequant__fp8__tensor | cumulative | 8f1ff9f1 | 80 | 0.952980 | 0.499262 | 3.2000e+04 | 5.8009e+04 | pass |
| hidden_dequant__fp8__tensor | cumulative | 1a4c6ba1 | 901 | 0.950381 | 0.473296 | 3.9680e+04 | 1.1551e+05 | pass |
| hidden_dequant__fp8__tensor | cumulative | a7c2bcfd | 16 | 0.952201 | 0.481262 | 1.9968e+04 | 1.7430e+03 | pass |
| hidden_dequant__fp8__tensor | cumulative | 2e69caee | 15 | 0.976953 | 0.773186 | 1.7984e+04 | 3.1302e+03 | pass |
| hidden_dequant__fp8__tensor | cumulative | 8cba5890 | 14 | 0.966089 | 0.640844 | 2.1504e+04 | 6.1239e+02 | pass |
| hidden_dequant__fp8__tensor | cumulative | 5e8dc11c | 14107 | 0.955083 | 0.521961 | 4.0960e+04 | 4.2325e+05 | pass |
| hidden_dequant__fp8__tensor | cumulative | 58a34f27 | 11948 | 0.965631 | 0.633739 | 4.8128e+04 | 1.0857e+06 | pass |
| hidden_dequant__fp8__tensor | cumulative | 5eadab1e | 62 | 0.962992 | 0.608133 | 2.5600e+04 | 1.1634e+04 | pass |
| hidden_dequant__fp8__tensor | cumulative | eedc63b2 | 59 | 0.969528 | 0.675086 | 2.3552e+04 | 2.3013e+03 | pass |
| hidden_dequant__fp8__tensor | cumulative | e626d3e6 | 58 | 0.954359 | 0.510336 | 2.5664e+04 | 2.9852e+03 | pass |
| hidden_dequant__fp8__tensor | cumulative | 74d7ff04 | 57 | 0.961794 | 0.590497 | 2.7648e+04 | 1.6295e+04 | pass |
| hidden_dequant__fp8__tensor | cumulative | 4822167c | 56 | 0.953586 | 0.494019 | 2.3040e+04 | 3.1676e+03 | pass |
| hidden_dequant__fp8__tensor | cumulative | 81955b1e | 55 | 0.963390 | 0.605763 | 2.5039e+04 | 1.8374e+03 | pass |
| hidden_dequant__fp8__tensor | cumulative | 76010cb4 | 54 | 0.969324 | 0.675766 | 2.4576e+04 | 3.1469e+03 | pass |
| hidden_dequant__fp8__tensor | cumulative | fc378037 | 53 | 0.964615 | 0.622052 | 2.4768e+04 | 2.1652e+03 | pass |
| hidden_dequant__fp8__tensor | cumulative | f7d6ac7c | 52 | 0.977461 | 0.759610 | 1.9504e+04 | 3.8662e+03 | pass |
| hidden_dequant__fp8__tensor+swiglu_output__fp8__tensor | cumulative | b8f4f012 | 7 | 0.973812 | 0.754604 | 2.9088e+04 | 4.0400e+04 | pass |
| hidden_dequant__fp8__tensor+swiglu_output__fp8__tensor | cumulative | e05c6c03 | 1 | 0.900251 | 0.135045 | 1.5872e+04 | 4.9756e+02 | pass |
| hidden_dequant__fp8__tensor+swiglu_output__fp8__tensor | cumulative | 6230e838 | 32 | 0.961735 | 0.649392 | 2.3552e+04 | 2.8027e+04 | pass |
| hidden_dequant__fp8__tensor+swiglu_output__fp8__tensor | cumulative | 8f1ff9f1 | 80 | 0.942540 | 0.481451 | 3.0720e+04 | 8.8313e+03 | pass |
| hidden_dequant__fp8__tensor+swiglu_output__fp8__tensor | cumulative | 1a4c6ba1 | 901 | 0.939682 | 0.455182 | 4.9152e+04 | 9.9296e+04 | pass |
| hidden_dequant__fp8__tensor+swiglu_output__fp8__tensor | cumulative | a7c2bcfd | 16 | 0.940552 | 0.459394 | 3.2256e+04 | 3.0506e+03 | pass |
| hidden_dequant__fp8__tensor+swiglu_output__fp8__tensor | cumulative | 2e69caee | 15 | 0.975344 | 0.769773 | 1.8432e+04 | 5.9957e+02 | pass |
| hidden_dequant__fp8__tensor+swiglu_output__fp8__tensor | cumulative | 8cba5890 | 14 | 0.960260 | 0.629564 | 2.2528e+04 | 2.3730e+03 | pass |
| hidden_dequant__fp8__tensor+swiglu_output__fp8__tensor | cumulative | 5e8dc11c | 14107 | 0.945170 | 0.505444 | 5.6320e+04 | 2.3265e+06 | pass |
| hidden_dequant__fp8__tensor+swiglu_output__fp8__tensor | cumulative | 58a34f27 | 11948 | 0.957959 | 0.620970 | 5.3248e+04 | 1.7500e+10 | pass |
| hidden_dequant__fp8__tensor+swiglu_output__fp8__tensor | cumulative | 5eadab1e | 62 | 0.955602 | 0.596284 | 3.4816e+04 | 1.0853e+04 | pass |
| hidden_dequant__fp8__tensor+swiglu_output__fp8__tensor | cumulative | eedc63b2 | 59 | 0.962791 | 0.662963 | 3.2768e+04 | 1.7426e+04 | pass |
| hidden_dequant__fp8__tensor+swiglu_output__fp8__tensor | cumulative | e626d3e6 | 58 | 0.944148 | 0.493482 | 2.9952e+04 | 1.5667e+05 | pass |
| hidden_dequant__fp8__tensor+swiglu_output__fp8__tensor | cumulative | 74d7ff04 | 57 | 0.953277 | 0.575908 | 3.1744e+04 | 5.8524e+03 | pass |
| hidden_dequant__fp8__tensor+swiglu_output__fp8__tensor | cumulative | 4822167c | 56 | 0.940970 | 0.473977 | 2.4576e+04 | 6.4514e+03 | pass |
| hidden_dequant__fp8__tensor+swiglu_output__fp8__tensor | cumulative | 81955b1e | 55 | 0.955202 | 0.590884 | 2.8672e+04 | 2.8130e+03 | pass |
| hidden_dequant__fp8__tensor+swiglu_output__fp8__tensor | cumulative | 76010cb4 | 54 | 0.963118 | 0.664579 | 2.8160e+04 | 1.9116e+04 | pass |
| hidden_dequant__fp8__tensor+swiglu_output__fp8__tensor | cumulative | fc378037 | 53 | 0.956800 | 0.608983 | 3.8912e+04 | 9.3499e+03 | pass |
| hidden_dequant__fp8__tensor+swiglu_output__fp8__tensor | cumulative | f7d6ac7c | 52 | 0.972340 | 0.750663 | 2.0480e+04 | 2.1571e+04 | pass |