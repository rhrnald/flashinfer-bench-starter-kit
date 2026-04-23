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
| run_stage | stage1_fp8_followup |
| panel_size | 8 |
| panel_indices | 0,1,2,4,8,9,14,18 |
| fp8_target_stages | hidden_dequant,swiglu_output |
| n_candidates | 6 |
| n_survivors | 6 |
| contest_safe_survivor_count | 6 |
| contest_safe_survivors | swiglu_output__fp8__block, swiglu_output__fp8__tensor, swiglu_output__fp8__row, hidden_dequant__fp8__block, hidden_dequant__fp8__tensor, hidden_dequant__fp8__row |
| strict_safe_survivor_count | 0 |
| strict_safe_survivors | - |
| contest_only_survivor_count | 6 |
| contest_only_survivors | hidden_dequant__fp8__tensor, hidden_dequant__fp8__row, hidden_dequant__fp8__block, swiglu_output__fp8__tensor, swiglu_output__fp8__row, swiglu_output__fp8__block |

## Survivor Summary

| category | count | candidates |
|---|---:|---|
| contest_safe_single_stage | 6 | swiglu_output__fp8__block, swiglu_output__fp8__tensor, swiglu_output__fp8__row, hidden_dequant__fp8__block, hidden_dequant__fp8__tensor, hidden_dequant__fp8__row |
| strict_safe_single_stage | 0 | - |
| contest_only_single_stage | 6 | hidden_dequant__fp8__tensor, hidden_dequant__fp8__row, hidden_dequant__fp8__block, swiglu_output__fp8__tensor, swiglu_output__fp8__row, swiglu_output__fp8__block |

## Stage Summary

| stage | best_safe_mode | scale | worst_matched_contest | worst_matched_strict | worst_rel | status |
|---|---|---|---:|---:|---:|---|
| hidden_dequant | fp8 | tensor | 0.912667 | 0.148019 | 3.5635e+06 | safe |
| swiglu_output | fp8 | tensor | 0.936802 | 0.222098 | 2.1602e+06 | safe |

## Cumulative Safe Frontier

| order | candidate | worst_matched_contest | worst_matched_strict | worst_rel | status |
|---|---|---:|---:|---:|---|
| 1 | hidden_dequant__fp8__tensor | 0.928990 | 0.179688 | 1.5366e+07 | safe |
| 2 | hidden_dequant__fp8__tensor+swiglu_output__fp8__tensor | 0.903041 | 0.136858 | 2.0644e+06 | safe |

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
| bf16_f16_survivors | swiglu_output__fp8__block, swiglu_output__fp8__tensor, swiglu_output__fp8__row, hidden_dequant__fp8__block, hidden_dequant__fp8__tensor, hidden_dequant__fp8__row |
| strict_survivors | - |
| pairwise_shortlist | - |

## Sampled Results

| candidate | phase | workload | seq_len | matched_contest | matched_strict | max_abs | max_rel | failure |
|---|---|---|---:|---:|---:|---:|---:|---|
| baseline__fp32 | single_stage | b8f4f012 | 7 | 1.000000 | 1.000000 | 0.0000e+00 | 0.0000e+00 | pass |
| hidden_dequant__fp8__tensor | single_stage | b8f4f012 | 7 | 0.976562 | 0.759945 | 2.3040e+04 | 2.7904e+02 | pass |
| hidden_dequant__fp8__row | single_stage | b8f4f012 | 7 | 0.977519 | 0.764190 | 2.3040e+04 | 2.1462e+02 | pass |
| hidden_dequant__fp8__block | single_stage | b8f4f012 | 7 | 0.978296 | 0.761938 | 2.1696e+04 | 1.6274e+02 | pass |
| swiglu_output__fp8__tensor | single_stage | b8f4f012 | 7 | 0.983618 | 0.779317 | 1.7408e+04 | 3.8587e+01 | pass |
| swiglu_output__fp8__row | single_stage | b8f4f012 | 7 | 0.983618 | 0.779317 | 1.7408e+04 | 3.8587e+01 | pass |
| swiglu_output__fp8__block | single_stage | b8f4f012 | 7 | 0.985391 | 0.789581 | 1.5232e+04 | 8.1970e+01 | pass |
| baseline__fp32 | single_stage | e05c6c03 | 1 | 1.000000 | 1.000000 | 0.0000e+00 | 0.0000e+00 | pass |
| hidden_dequant__fp8__tensor | single_stage | e05c6c03 | 1 | 0.912667 | 0.148019 | 1.4336e+04 | 5.0336e+02 | pass |
| hidden_dequant__fp8__row | single_stage | e05c6c03 | 1 | 0.912667 | 0.148019 | 1.4336e+04 | 5.0336e+02 | pass |
| hidden_dequant__fp8__block | single_stage | e05c6c03 | 1 | 0.929408 | 0.179269 | 1.2668e+04 | 4.0092e+02 | pass |
| swiglu_output__fp8__tensor | single_stage | e05c6c03 | 1 | 0.936802 | 0.222098 | 1.0368e+04 | 7.9350e+01 | pass |
| swiglu_output__fp8__row | single_stage | e05c6c03 | 1 | 0.936802 | 0.222098 | 1.0368e+04 | 7.9350e+01 | pass |
| swiglu_output__fp8__block | single_stage | e05c6c03 | 1 | 0.948661 | 0.270089 | 7.6800e+03 | 1.2318e+02 | pass |
| baseline__fp32 | single_stage | 6230e838 | 32 | 1.000000 | 1.000000 | 0.0000e+00 | 0.0000e+00 | pass |
| hidden_dequant__fp8__tensor | single_stage | 6230e838 | 32 | 0.968135 | 0.660858 | 2.4064e+04 | 3.5259e+03 | pass |
| hidden_dequant__fp8__row | single_stage | 6230e838 | 32 | 0.967499 | 0.660326 | 2.5344e+04 | 1.7319e+03 | pass |
| hidden_dequant__fp8__block | single_stage | 6230e838 | 32 | 0.968772 | 0.662000 | 2.2528e+04 | 3.0710e+03 | pass |
| swiglu_output__fp8__tensor | single_stage | 6230e838 | 32 | 0.977997 | 0.688664 | 1.7368e+04 | 1.1922e+03 | pass |
| swiglu_output__fp8__row | single_stage | 6230e838 | 32 | 0.978149 | 0.690430 | 1.7368e+04 | 9.7784e+02 | pass |
| swiglu_output__fp8__block | single_stage | 6230e838 | 32 | 0.980028 | 0.698417 | 1.3524e+04 | 1.4504e+03 | pass |
| baseline__fp32 | single_stage | 1a4c6ba1 | 901 | 1.000000 | 1.000000 | 0.0000e+00 | 0.0000e+00 | pass |
| hidden_dequant__fp8__tensor | single_stage | 1a4c6ba1 | 901 | 0.950431 | 0.473044 | 3.5840e+04 | 2.7197e+05 | pass |
| hidden_dequant__fp8__row | single_stage | 1a4c6ba1 | 901 | 0.950454 | 0.473224 | 3.2768e+04 | 1.8159e+05 | pass |
| hidden_dequant__fp8__block | single_stage | 1a4c6ba1 | 901 | 0.951895 | 0.474277 | 3.1808e+04 | 2.0890e+05 | pass |
| swiglu_output__fp8__tensor | single_stage | 1a4c6ba1 | 901 | 0.965061 | 0.514560 | 2.1568e+04 | 1.5769e+05 | pass |
| swiglu_output__fp8__row | single_stage | 1a4c6ba1 | 901 | 0.965738 | 0.517172 | 2.4576e+04 | 7.9463e+04 | pass |
| swiglu_output__fp8__block | single_stage | 1a4c6ba1 | 901 | 0.969357 | 0.533076 | 2.4576e+04 | 9.7076e+04 | pass |
| baseline__fp32 | single_stage | 5e8dc11c | 14107 | 1.000000 | 1.000000 | 0.0000e+00 | 0.0000e+00 | pass |
| hidden_dequant__fp8__tensor | single_stage | 5e8dc11c | 14107 | 0.955158 | 0.522060 | 5.3912e+04 | 1.3844e+06 | pass |
| hidden_dequant__fp8__row | single_stage | 5e8dc11c | 14107 | 0.955082 | 0.522291 | 4.5824e+04 | 1.5810e+06 | pass |
| hidden_dequant__fp8__block | single_stage | 5e8dc11c | 14107 | 0.956388 | 0.523028 | 4.0960e+04 | 1.0845e+06 | pass |
| swiglu_output__fp8__tensor | single_stage | 5e8dc11c | 14107 | 0.968257 | 0.559060 | 4.3008e+04 | 6.3488e+05 | pass |
| swiglu_output__fp8__row | single_stage | 5e8dc11c | 14107 | 0.968874 | 0.561729 | 3.6864e+04 | 3.6484e+06 | pass |
| swiglu_output__fp8__block | single_stage | 5e8dc11c | 14107 | 0.972245 | 0.576692 | 3.4048e+04 | 3.1241e+05 | pass |
| baseline__fp32 | single_stage | 58a34f27 | 11948 | 1.000000 | 1.000000 | 0.0000e+00 | 0.0000e+00 | pass |
| hidden_dequant__fp8__tensor | single_stage | 58a34f27 | 11948 | 0.965614 | 0.633766 | 4.5056e+04 | 3.3423e+06 | pass |
| hidden_dequant__fp8__row | single_stage | 58a34f27 | 11948 | 0.965532 | 0.633859 | 5.1200e+04 | 4.1943e+06 | pass |
| hidden_dequant__fp8__block | single_stage | 58a34f27 | 11948 | 0.966566 | 0.634534 | 4.5056e+04 | 5.3739e+05 | pass |
| swiglu_output__fp8__tensor | single_stage | 58a34f27 | 11948 | 0.975635 | 0.662091 | 4.5056e+04 | 1.8268e+06 | pass |
| swiglu_output__fp8__row | single_stage | 58a34f27 | 11948 | 0.976161 | 0.664205 | 2.9696e+04 | 4.3827e+05 | pass |
| swiglu_output__fp8__block | single_stage | 58a34f27 | 11948 | 0.978712 | 0.675637 | 2.6624e+04 | 1.5810e+06 | pass |
| baseline__fp32 | single_stage | 4822167c | 56 | 1.000000 | 1.000000 | 0.0000e+00 | 0.0000e+00 | pass |
| hidden_dequant__fp8__tensor | single_stage | 4822167c | 56 | 0.952487 | 0.493269 | 2.2528e+04 | 3.5635e+06 | pass |
| hidden_dequant__fp8__row | single_stage | 4822167c | 56 | 0.952751 | 0.493725 | 2.5600e+04 | 3.5635e+06 | pass |
| hidden_dequant__fp8__block | single_stage | 4822167c | 56 | 0.953905 | 0.495523 | 2.3552e+04 | 3.1063e+06 | pass |
| swiglu_output__fp8__tensor | single_stage | 4822167c | 56 | 0.966892 | 0.533407 | 1.6384e+04 | 2.1602e+06 | pass |
| swiglu_output__fp8__row | single_stage | 4822167c | 56 | 0.967021 | 0.536434 | 1.6384e+04 | 1.7345e+06 | pass |
| swiglu_output__fp8__block | single_stage | 4822167c | 56 | 0.970549 | 0.550315 | 1.8176e+04 | 8.9679e+04 | pass |
| baseline__fp32 | single_stage | f7d6ac7c | 52 | 1.000000 | 1.000000 | 0.0000e+00 | 0.0000e+00 | pass |
| hidden_dequant__fp8__tensor | single_stage | f7d6ac7c | 52 | 0.977697 | 0.759476 | 2.1504e+04 | 4.3769e+03 | pass |
| hidden_dequant__fp8__row | single_stage | f7d6ac7c | 52 | 0.977394 | 0.758617 | 1.7920e+04 | 2.6191e+03 | pass |
| hidden_dequant__fp8__block | single_stage | f7d6ac7c | 52 | 0.978339 | 0.759527 | 2.3552e+04 | 1.3423e+03 | pass |
| swiglu_output__fp8__tensor | single_stage | f7d6ac7c | 52 | 0.984270 | 0.778135 | 1.3312e+04 | 2.8679e+03 | pass |
| swiglu_output__fp8__row | single_stage | f7d6ac7c | 52 | 0.984410 | 0.779463 | 1.3312e+04 | 4.1447e+03 | pass |
| swiglu_output__fp8__block | single_stage | f7d6ac7c | 52 | 0.986232 | 0.787061 | 1.1520e+04 | 4.4764e+03 | pass |
| hidden_dequant__fp8__tensor | cumulative | b8f4f012 | 7 | 0.975825 | 0.759327 | 1.8976e+04 | 1.6265e+03 | pass |
| hidden_dequant__fp8__tensor | cumulative | e05c6c03 | 1 | 0.928990 | 0.179688 | 1.3440e+04 | 1.3625e+02 | pass |
| hidden_dequant__fp8__tensor | cumulative | 6230e838 | 32 | 0.967904 | 0.660915 | 2.1504e+04 | 2.9825e+03 | pass |
| hidden_dequant__fp8__tensor | cumulative | 1a4c6ba1 | 901 | 0.950565 | 0.473198 | 3.4816e+04 | 1.6727e+05 | pass |
| hidden_dequant__fp8__tensor | cumulative | 5e8dc11c | 14107 | 0.955086 | 0.521999 | 5.1200e+04 | 1.5366e+07 | pass |
| hidden_dequant__fp8__tensor | cumulative | 58a34f27 | 11948 | 0.965570 | 0.633672 | 4.3392e+04 | 9.1750e+05 | pass |
| hidden_dequant__fp8__tensor | cumulative | 4822167c | 56 | 0.952928 | 0.493266 | 2.3552e+04 | 6.0486e+03 | pass |
| hidden_dequant__fp8__tensor | cumulative | f7d6ac7c | 52 | 0.977432 | 0.758829 | 1.7920e+04 | 8.4368e+03 | pass |
| hidden_dequant__fp8__tensor+swiglu_output__fp8__tensor | cumulative | b8f4f012 | 7 | 0.974171 | 0.755780 | 2.8672e+04 | 9.0253e+02 | pass |
| hidden_dequant__fp8__tensor+swiglu_output__fp8__tensor | cumulative | e05c6c03 | 1 | 0.903041 | 0.136858 | 2.7136e+04 | 6.5308e+02 | pass |
| hidden_dequant__fp8__tensor+swiglu_output__fp8__tensor | cumulative | 6230e838 | 32 | 0.960567 | 0.649484 | 2.6624e+04 | 1.3296e+03 | pass |
| hidden_dequant__fp8__tensor+swiglu_output__fp8__tensor | cumulative | 1a4c6ba1 | 901 | 0.939616 | 0.455154 | 5.7344e+04 | 6.1144e+04 | pass |
| hidden_dequant__fp8__tensor+swiglu_output__fp8__tensor | cumulative | 5e8dc11c | 14107 | 0.945148 | 0.505354 | 5.1712e+04 | 2.0644e+06 | pass |
| hidden_dequant__fp8__tensor+swiglu_output__fp8__tensor | cumulative | 58a34f27 | 11948 | 0.958004 | 0.621064 | 4.5056e+04 | 7.4247e+05 | pass |
| hidden_dequant__fp8__tensor+swiglu_output__fp8__tensor | cumulative | 4822167c | 56 | 0.942174 | 0.475561 | 4.3716e+04 | 8.1009e+04 | pass |
| hidden_dequant__fp8__tensor+swiglu_output__fp8__tensor | cumulative | f7d6ac7c | 52 | 0.973311 | 0.751637 | 2.8672e+04 | 3.1990e+03 | pass |