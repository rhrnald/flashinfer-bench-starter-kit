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
| run_stage | stage1_bf16_f16 |
| panel_size | 19 |
| panel_indices | 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18 |
| n_candidates | 19 |
| n_survivors | 14 |
| contest_safe_survivor_count | 14 |
| contest_safe_survivors | hidden_dequant__f16, gemm1_output__f16, swiglu_input__f16, gemm1_operands__f16, out_accumulator__bf16, swiglu_output__bf16, gemm1_accumulator__f16, gemm1_output__bf16, swiglu_input__bf16, hidden_dequant__bf16, gemm2_operands__bf16, gemm1_operands__bf16, gemm2_accumulator__bf16, gemm1_accumulator__bf16 |
| strict_safe_survivor_count | 5 |
| strict_safe_survivors | hidden_dequant__f16, gemm1_operands__f16, gemm1_output__f16, swiglu_input__f16, out_accumulator__bf16 |
| contest_only_survivor_count | 9 |
| contest_only_survivors | hidden_dequant__bf16, gemm1_operands__bf16, gemm1_accumulator__bf16, gemm1_accumulator__f16, gemm1_output__bf16, swiglu_input__bf16, swiglu_output__bf16, gemm2_operands__bf16, gemm2_accumulator__bf16 |

## Survivor Summary

| category | count | candidates |
|---|---:|---|
| contest_safe_single_stage | 14 | hidden_dequant__f16, gemm1_output__f16, swiglu_input__f16, gemm1_operands__f16, out_accumulator__bf16, swiglu_output__bf16, gemm1_accumulator__f16, gemm1_output__bf16, swiglu_input__bf16, hidden_dequant__bf16, gemm2_operands__bf16, gemm1_operands__bf16, gemm2_accumulator__bf16, gemm1_accumulator__bf16 |
| strict_safe_single_stage | 5 | hidden_dequant__f16, gemm1_operands__f16, gemm1_output__f16, swiglu_input__f16, out_accumulator__bf16 |
| contest_only_single_stage | 9 | hidden_dequant__bf16, gemm1_operands__bf16, gemm1_accumulator__bf16, gemm1_accumulator__f16, gemm1_output__bf16, swiglu_input__bf16, swiglu_output__bf16, gemm2_operands__bf16, gemm2_accumulator__bf16 |

## Stage Summary

| stage | best_safe_mode | scale | worst_matched_contest | worst_matched_strict | worst_rel | status |
|---|---|---|---:|---:|---:|---|
| gemm1_accumulator | f16 | none | 0.996094 | 0.892578 | 4.2500e+09 | safe |
| gemm1_operands | f16 | none | 0.999163 | 0.967355 | 1.3000e+09 | safe |
| gemm1_output | f16 | none | 0.999302 | 0.979353 | 1.2000e+09 | safe |
| gemm2_accumulator | bf16 | none | 0.990932 | 0.738560 | 1.4800e+10 | safe |
| gemm2_operands | bf16 | none | 0.994559 | 0.850307 | 7.3500e+09 | safe |
| hidden_dequant | f16 | none | 0.999442 | 0.976842 | 1.5562e+09 | safe |
| out_accumulator | bf16 | none | 0.998326 | 0.929688 | 3.3250e+09 | safe |
| swiglu_input | f16 | none | 0.999302 | 0.979353 | 1.2000e+09 | safe |
| swiglu_output | bf16 | none | 0.996931 | 0.889090 | 9.4000e+09 | safe |

## Cumulative Safe Frontier

| order | candidate | worst_matched_contest | worst_matched_strict | worst_rel | status | kept |
|---|---|---:|---:|---:|---|---|
| 1 | gemm1_accumulator__f16 | 0.997349 | 0.898019 | 6.6053e+04 | safe | yes |
| 2 | gemm1_accumulator__f16+gemm1_operands__f16 | 0.998744 | 0.972656 | 1.6385e+04 | safe | yes |
| 3 | gemm1_accumulator__f16+gemm1_operands__f16+gemm1_output__f16 | 0.999302 | 0.964844 | 5.3986e+05 | safe | yes |
| 4 | gemm1_accumulator__f16+gemm1_operands__f16+gemm1_output__f16+gemm2_accumulator__bf16 | 0.989955 | 0.732840 | 4.0403e+05 | safe | yes |
| 5 | gemm1_accumulator__f16+gemm1_operands__f16+gemm1_output__f16+gemm2_accumulator__bf16+gemm2_operands__bf16 | 0.994838 | 0.846959 | 8.2500e+09 | safe | yes |
| 6 | gemm1_accumulator__f16+gemm1_operands__f16+gemm1_output__f16+gemm2_accumulator__bf16+gemm2_operands__bf16+hidden_dequant__f16 | 0.996669 | 0.851144 | 3.9000e+10 | safe | yes |
| 7 | gemm1_accumulator__f16+gemm1_operands__f16+gemm1_output__f16+gemm2_accumulator__bf16+gemm2_operands__bf16+hidden_dequant__f16+out_accumulator__bf16 | 0.994559 | 0.833984 | 1.2390e+05 | safe | yes |
| 8 | gemm1_accumulator__f16+gemm1_operands__f16+gemm1_output__f16+gemm2_accumulator__bf16+gemm2_operands__bf16+hidden_dequant__f16+out_accumulator__bf16+swiglu_input__f16 | 0.995117 | 0.836356 | 1.0251e+05 | safe | yes |
| 9 | gemm1_accumulator__f16+gemm1_operands__f16+gemm1_output__f16+gemm2_accumulator__bf16+gemm2_operands__bf16+hidden_dequant__f16+out_accumulator__bf16+swiglu_input__f16+swiglu_output__bf16 | 0.995815 | 0.823940 | 1.8800e+10 | safe | yes |

## BF16/F16 Margin

| stage | preferred | bf16_contest | f16_contest | bf16_strict | f16_strict | |
|---|---|---:|---:|---:|---:|---|
| hidden_dequant | f16 | 0.994559 | 0.999442 | 0.858398 | 0.976842 | |
| gemm1_operands | f16 | 0.992188 | 0.999163 | 0.796038 | 0.967355 | |
| gemm1_accumulator | f16 | 0.970564 | 0.996094 | 0.427176 | 0.892578 | |
| gemm1_output | f16 | 0.994699 | 0.999302 | 0.848354 | 0.979353 | |
| swiglu_input | f16 | 0.994699 | 0.999302 | 0.848354 | 0.979353 | |
| swiglu_output | bf16 | 0.996931 | 0.896458 | 0.889090 | 0.890084 | |
| gemm2_operands | bf16 | 0.994559 | 0.896386 | 0.850307 | 0.887404 | |
| gemm2_accumulator | bf16 | 0.990932 | 0.594587 | 0.738560 | 0.594230 | |
| out_accumulator | bf16 | 0.998326 | 0.428292 | 0.929688 | 0.423410 | |

## Pairwise Summary

| pair | worst_matched_contest | worst_matched_strict | worst_rel | status |
|---|---:|---:|---:|---|
| gemm1_accumulator__f16+gemm2_accumulator__bf16 | 0.991071 | 0.729353 | 3.5600e+10 | safe |
| gemm1_output__f16+gemm2_accumulator__bf16 | 0.991490 | 0.752651 | 1.9600e+10 | safe |
| hidden_dequant__f16+gemm2_accumulator__bf16 | 0.991490 | 0.753767 | 1.9600e+10 | safe |
| gemm2_accumulator__bf16+out_accumulator__bf16 | 0.991769 | 0.743025 | 3.7200e+10 | safe |
| gemm1_operands__f16+gemm2_accumulator__bf16 | 0.992048 | 0.753209 | 1.9600e+10 | safe |
| gemm1_accumulator__f16+out_accumulator__bf16 | 0.995536 | 0.879325 | 1.4562e+09 | safe |
| hidden_dequant__f16+gemm1_accumulator__f16 | 0.996233 | 0.895229 | 6.4500e+09 | safe |
| gemm1_accumulator__f16+gemm1_output__f16 | 0.997210 | 0.898019 | 4.3750e+09 | safe |
| gemm1_operands__f16+out_accumulator__bf16 | 0.997210 | 0.934291 | 5.3125e+08 | safe |
| hidden_dequant__f16+out_accumulator__bf16 | 0.997768 | 0.937500 | 8.7500e+08 | safe |
| gemm1_output__f16+out_accumulator__bf16 | 0.997768 | 0.938058 | 2.4531e+08 | safe |
| hidden_dequant__f16+gemm1_output__f16 | 0.998465 | 0.971401 | 2.8250e+09 | safe |
| gemm1_operands__f16+gemm1_output__f16 | 0.998744 | 0.968052 | 8.3000e+09 | safe |
| hidden_dequant__f16+gemm1_operands__f16 | 0.999023 | 0.974191 | 1.1187e+09 | safe |
| gemm1_operands__f16+gemm1_accumulator__f16 | 0.999023 | 0.974191 | 1.1187e+09 | safe |

## Stress Summary

| candidate | worst_workload | worst_seq_len | worst_matched_contest | worst_matched_strict | status |
|---|---|---:|---:|---:|---|
| gemm2_accumulator__bf16 | e05c6c03 | 1 | 0.990932 | 0.738560 | safe |

## Promotion Summary

| category | candidates |
|---|---|
| bf16_f16_survivors | hidden_dequant__f16, gemm1_output__f16, swiglu_input__f16, gemm1_operands__f16, out_accumulator__bf16, swiglu_output__bf16, gemm1_accumulator__f16, gemm1_output__bf16, swiglu_input__bf16, hidden_dequant__bf16, gemm2_operands__bf16, gemm1_operands__bf16, gemm2_accumulator__bf16, gemm1_accumulator__bf16 |
| strict_survivors | hidden_dequant__f16, gemm1_operands__f16, gemm1_output__f16, swiglu_input__f16, out_accumulator__bf16 |
| pairwise_shortlist | hidden_dequant__f16, gemm1_operands__f16, gemm1_accumulator__f16, gemm1_output__f16, gemm2_accumulator__bf16, out_accumulator__bf16 |

## Sampled Results

| candidate | phase | workload | seq_len | matched_contest | matched_strict | max_abs | max_rel | failure |
|---|---|---|---:|---:|---:|---:|---:|---|
| baseline__fp32 | single_stage | b8f4f012 | 7 | 1.000000 | 1.000000 | 0.0000e+00 | 0.0000e+00 | pass |
| hidden_dequant__bf16 | single_stage | b8f4f012 | 7 | 0.998465 | 0.957211 | 2.0480e+03 | 2.2106e+01 | pass |
| hidden_dequant__f16 | single_stage | b8f4f012 | 7 | 0.999761 | 0.994300 | 2.0480e+03 | 3.9825e+00 | pass |
| gemm1_operands__bf16 | single_stage | b8f4f012 | 7 | 0.997768 | 0.938815 | 2.0480e+03 | 4.2544e+01 | pass |
| gemm1_operands__f16 | single_stage | b8f4f012 | 7 | 0.999661 | 0.992387 | 2.0480e+03 | 5.8936e+00 | pass |
| gemm1_accumulator__bf16 | single_stage | b8f4f012 | 7 | 0.993084 | 0.838289 | 7.6800e+03 | 1.7167e+02 | pass |
| gemm1_accumulator__f16 | single_stage | b8f4f012 | 7 | 0.998944 | 0.971102 | 2.0480e+03 | 2.9602e+01 | pass |
| gemm1_output__bf16 | single_stage | b8f4f012 | 7 | 0.998705 | 0.958207 | 2.0480e+03 | 5.9823e+01 | pass |
| gemm1_output__f16 | single_stage | b8f4f012 | 7 | 0.999741 | 0.995097 | 2.0480e+03 | 2.7021e+00 | pass |
| swiglu_input__bf16 | single_stage | b8f4f012 | 7 | 0.998705 | 0.958207 | 2.0480e+03 | 5.9823e+01 | pass |
| swiglu_input__f16 | single_stage | b8f4f012 | 7 | 0.999741 | 0.995097 | 2.0480e+03 | 2.7021e+00 | pass |
| swiglu_output__bf16 | single_stage | b8f4f012 | 7 | 0.998744 | 0.969906 | 2.0480e+03 | 2.2830e+01 | pass |
| swiglu_output__f16 | single_stage | b8f4f012 | 7 | 0.999821 | 0.996134 | 2.0480e+03 | 8.0780e+00 | pass |
| gemm2_operands__bf16 | single_stage | b8f4f012 | 7 | 0.998326 | 0.957510 | 2.0480e+03 | 6.0277e+01 | pass |
| gemm2_operands__f16 | single_stage | b8f4f012 | 7 | 0.999701 | 0.994539 | 2.0480e+03 | 6.8298e+00 | pass |
| gemm2_accumulator__bf16 | single_stage | b8f4f012 | 7 | 0.997648 | 0.929110 | 4.0960e+03 | 4.7281e+01 | pass |
| gemm2_accumulator__f16 | single_stage | b8f4f012 | 7 | 0.913584 | 0.913584 | nan | nan | pass |
| out_accumulator__bf16 | single_stage | b8f4f012 | 7 | 0.999422 | 0.981067 | 2.0480e+03 | 2.3779e+01 | pass |
| out_accumulator__f16 | single_stage | b8f4f012 | 7 | 0.792909 | 0.791932 | inf | inf | catastrophic_outlier |
| baseline__fp32 | single_stage | e05c6c03 | 1 | 1.000000 | 1.000000 | 0.0000e+00 | 0.0000e+00 | pass |
| hidden_dequant__bf16 | single_stage | e05c6c03 | 1 | 0.994559 | 0.858398 | 2.0480e+03 | 1.9272e+01 | pass |
| hidden_dequant__f16 | single_stage | e05c6c03 | 1 | 0.999442 | 0.976842 | 1.0240e+03 | 4.4412e+00 | pass |
| gemm1_operands__bf16 | single_stage | e05c6c03 | 1 | 0.992188 | 0.796038 | 2.0480e+03 | 1.9848e+01 | pass |
| gemm1_operands__f16 | single_stage | e05c6c03 | 1 | 0.999163 | 0.967355 | 1.0240e+03 | 2.5000e+00 | pass |
| gemm1_accumulator__bf16 | single_stage | e05c6c03 | 1 | 0.970564 | 0.427176 | 4.8640e+03 | 2.7040e+02 | pass |
| gemm1_accumulator__f16 | single_stage | e05c6c03 | 1 | 0.996094 | 0.892578 | 2.0480e+03 | 2.4288e+01 | pass |
| gemm1_output__bf16 | single_stage | e05c6c03 | 1 | 0.994699 | 0.848354 | 2.0480e+03 | 9.4118e+00 | pass |
| gemm1_output__f16 | single_stage | e05c6c03 | 1 | 0.999302 | 0.979353 | 1.0240e+03 | 4.6230e+00 | pass |
| swiglu_input__bf16 | single_stage | e05c6c03 | 1 | 0.994699 | 0.848354 | 2.0480e+03 | 9.4118e+00 | pass |
| swiglu_input__f16 | single_stage | e05c6c03 | 1 | 0.999302 | 0.979353 | 1.0240e+03 | 4.6230e+00 | pass |
| swiglu_output__bf16 | single_stage | e05c6c03 | 1 | 0.996931 | 0.889090 | 2.0480e+03 | 4.1880e+01 | pass |
| swiglu_output__f16 | single_stage | e05c6c03 | 1 | 0.999581 | 0.985073 | 1.0240e+03 | 3.5340e+00 | pass |
| gemm2_operands__bf16 | single_stage | e05c6c03 | 1 | 0.994559 | 0.850307 | 2.0480e+03 | 7.2037e+01 | pass |
| gemm2_operands__f16 | single_stage | e05c6c03 | 1 | 0.999163 | 0.980190 | 1.0240e+03 | 6.9895e+00 | pass |
| gemm2_accumulator__bf16 | single_stage | e05c6c03 | 1 | 0.990932 | 0.738560 | 2.0480e+03 | 3.6059e+01 | pass |
| gemm2_accumulator__f16 | single_stage | e05c6c03 | 1 | 0.686942 | 0.686802 | nan | nan | global_drift |
| out_accumulator__bf16 | single_stage | e05c6c03 | 1 | 0.998326 | 0.929688 | 2.0480e+03 | 1.4916e+01 | pass |
| out_accumulator__f16 | single_stage | e05c6c03 | 1 | 0.428292 | 0.423410 | inf | inf | catastrophic_saturation |
| baseline__fp32 | single_stage | 6230e838 | 32 | 1.000000 | 1.000000 | 0.0000e+00 | 0.0000e+00 | pass |
| hidden_dequant__bf16 | single_stage | 6230e838 | 32 | 0.997981 | 0.937705 | 2.0480e+03 | 7.6429e+01 | pass |
| hidden_dequant__f16 | single_stage | 6230e838 | 32 | 0.999756 | 0.991991 | 2.0480e+03 | 2.5095e+01 | pass |
| gemm1_operands__bf16 | single_stage | 6230e838 | 32 | 0.996996 | 0.912711 | 2.0480e+03 | 1.4703e+02 | pass |
| gemm1_operands__f16 | single_stage | 6230e838 | 32 | 0.999634 | 0.988687 | 2.0480e+03 | 1.7952e+01 | pass |
| gemm1_accumulator__bf16 | single_stage | 6230e838 | 32 | 0.989192 | 0.769919 | 6.9120e+03 | 3.6071e+02 | pass |
| gemm1_accumulator__f16 | single_stage | 6230e838 | 32 | 0.998649 | 0.959516 | 2.0480e+03 | 4.9451e+01 | pass |
| gemm1_output__bf16 | single_stage | 6230e838 | 32 | 0.997881 | 0.937382 | 2.0480e+03 | 1.3816e+02 | pass |
| gemm1_output__f16 | single_stage | 6230e838 | 32 | 0.999760 | 0.991813 | 2.0480e+03 | 1.3958e+01 | pass |
| swiglu_input__bf16 | single_stage | 6230e838 | 32 | 0.997881 | 0.937382 | 2.0480e+03 | 1.3816e+02 | pass |
| swiglu_input__f16 | single_stage | 6230e838 | 32 | 0.999760 | 0.991813 | 2.0480e+03 | 1.3958e+01 | pass |
| swiglu_output__bf16 | single_stage | 6230e838 | 32 | 0.998492 | 0.956604 | 2.0480e+03 | 2.8519e+02 | pass |
| swiglu_output__f16 | single_stage | 6230e838 | 32 | 0.968602 | 0.963959 | nan | nan | pass |
| gemm2_operands__bf16 | single_stage | 6230e838 | 32 | 0.997925 | 0.938542 | 2.0480e+03 | 2.8913e+02 | pass |
| gemm2_operands__f16 | single_stage | 6230e838 | 32 | 0.968545 | 0.962324 | nan | nan | pass |
| gemm2_accumulator__bf16 | single_stage | 6230e838 | 32 | 0.996922 | 0.900957 | 6.1440e+03 | 2.2612e+02 | pass |
| gemm2_accumulator__f16 | single_stage | 6230e838 | 32 | 0.835231 | 0.835201 | nan | nan | global_drift |
| out_accumulator__bf16 | single_stage | 6230e838 | 32 | 0.999285 | 0.976763 | 2.0480e+03 | 2.8718e+01 | pass |
| out_accumulator__f16 | single_stage | 6230e838 | 32 | 0.767875 | 0.766257 | inf | inf | catastrophic_outlier |
| baseline__fp32 | single_stage | 8f1ff9f1 | 80 | 1.000000 | 1.000000 | 0.0000e+00 | 0.0000e+00 | pass |
| hidden_dequant__bf16 | single_stage | 8f1ff9f1 | 80 | 0.997131 | 0.909567 | 2.0480e+03 | 4.9282e+02 | pass |
| hidden_dequant__f16 | single_stage | 8f1ff9f1 | 80 | 0.999630 | 0.988600 | 2.0480e+03 | 6.0163e+01 | pass |
| gemm1_operands__bf16 | single_stage | 8f1ff9f1 | 80 | 0.995944 | 0.875255 | 2.0480e+03 | 6.3807e+02 | pass |
| gemm1_operands__f16 | single_stage | 8f1ff9f1 | 80 | 0.999489 | 0.983688 | 2.0480e+03 | 6.8044e+01 | pass |
| gemm1_accumulator__bf16 | single_stage | 8f1ff9f1 | 80 | 0.984450 | 0.656267 | 8.7040e+03 | 3.8781e+03 | pass |
| gemm1_accumulator__f16 | single_stage | 8f1ff9f1 | 80 | 0.998083 | 0.938058 | 2.0480e+03 | 4.8950e+02 | pass |
| gemm1_output__bf16 | single_stage | 8f1ff9f1 | 80 | 0.997124 | 0.909943 | 2.0480e+03 | 5.8917e+02 | pass |
| gemm1_output__f16 | single_stage | 8f1ff9f1 | 80 | 0.999625 | 0.988478 | 2.0480e+03 | 4.7122e+01 | pass |
| swiglu_input__bf16 | single_stage | 8f1ff9f1 | 80 | 0.997124 | 0.909943 | 2.0480e+03 | 5.8917e+02 | pass |
| swiglu_input__f16 | single_stage | 8f1ff9f1 | 80 | 0.999625 | 0.988478 | 2.0480e+03 | 4.7122e+01 | pass |
| swiglu_output__bf16 | single_stage | 8f1ff9f1 | 80 | 0.997981 | 0.934340 | 2.0480e+03 | 4.1011e+02 | pass |
| swiglu_output__f16 | single_stage | 8f1ff9f1 | 80 | 0.987228 | 0.979417 | nan | nan | pass |
| gemm2_operands__bf16 | single_stage | 8f1ff9f1 | 80 | 0.997049 | 0.909473 | 2.0480e+03 | 1.0835e+03 | pass |
| gemm2_operands__f16 | single_stage | 8f1ff9f1 | 80 | 0.987125 | 0.976156 | nan | nan | pass |
| gemm2_accumulator__bf16 | single_stage | 8f1ff9f1 | 80 | 0.995773 | 0.862186 | 6.1440e+03 | 3.9575e+02 | pass |
| gemm2_accumulator__f16 | single_stage | 8f1ff9f1 | 80 | 0.674390 | 0.674114 | nan | nan | global_drift |
| out_accumulator__bf16 | single_stage | 8f1ff9f1 | 80 | 0.999234 | 0.974656 | 2.0480e+03 | 7.8938e+01 | pass |
| out_accumulator__f16 | single_stage | 8f1ff9f1 | 80 | 0.685331 | 0.683644 | inf | inf | catastrophic_outlier |
| baseline__fp32 | single_stage | 1a4c6ba1 | 901 | 1.000000 | 1.000000 | 0.0000e+00 | 0.0000e+00 | pass |
| hidden_dequant__bf16 | single_stage | 1a4c6ba1 | 901 | 0.996856 | 0.904040 | 4.0960e+03 | 8.3210e+03 | pass |
| hidden_dequant__f16 | single_stage | 1a4c6ba1 | 901 | 0.999618 | 0.987695 | 4.0960e+03 | 6.9058e+02 | pass |
| gemm1_operands__bf16 | single_stage | 1a4c6ba1 | 901 | 0.995549 | 0.867001 | 4.0960e+03 | 3.5841e+04 | pass |
| gemm1_operands__f16 | single_stage | 1a4c6ba1 | 901 | 0.999446 | 0.982532 | 4.0960e+03 | 2.0659e+03 | pass |
| gemm1_accumulator__bf16 | single_stage | 1a4c6ba1 | 901 | 0.983768 | 0.642964 | 1.2288e+04 | 1.6384e+05 | pass |
| gemm1_accumulator__f16 | single_stage | 1a4c6ba1 | 901 | 0.997915 | 0.935801 | 4.0960e+03 | 3.6419e+03 | pass |
| gemm1_output__bf16 | single_stage | 1a4c6ba1 | 901 | 0.996879 | 0.904326 | 4.0960e+03 | 2.1375e+04 | pass |
| gemm1_output__f16 | single_stage | 1a4c6ba1 | 901 | 0.999619 | 0.987766 | 4.0960e+03 | 2.9910e+03 | pass |
| swiglu_input__bf16 | single_stage | 1a4c6ba1 | 901 | 0.996879 | 0.904326 | 4.0960e+03 | 2.1375e+04 | pass |
| swiglu_input__f16 | single_stage | 1a4c6ba1 | 901 | 0.999619 | 0.987766 | 4.0960e+03 | 2.9910e+03 | pass |
| swiglu_output__bf16 | single_stage | 1a4c6ba1 | 901 | 0.997801 | 0.931630 | 4.0960e+03 | 2.7979e+03 | pass |
| swiglu_output__f16 | single_stage | 1a4c6ba1 | 901 | 0.955282 | 0.947552 | nan | nan | pass |
| gemm2_operands__bf16 | single_stage | 1a4c6ba1 | 901 | 0.996875 | 0.904788 | 4.0960e+03 | 5.5039e+04 | pass |
| gemm2_operands__f16 | single_stage | 1a4c6ba1 | 901 | 0.955180 | 0.944272 | nan | nan | pass |
| gemm2_accumulator__bf16 | single_stage | 1a4c6ba1 | 901 | 0.995440 | 0.852118 | 8.1920e+03 | 3.7121e+04 | pass |
| gemm2_accumulator__f16 | single_stage | 1a4c6ba1 | 901 | 0.678667 | 0.678407 | nan | nan | global_drift |
| out_accumulator__bf16 | single_stage | 1a4c6ba1 | 901 | 0.999100 | 0.971110 | 4.0960e+03 | 2.7510e+03 | pass |
| out_accumulator__f16 | single_stage | 1a4c6ba1 | 901 | 0.684182 | 0.682074 | inf | inf | catastrophic_outlier |
| baseline__fp32 | single_stage | a7c2bcfd | 16 | 1.000000 | 1.000000 | 0.0000e+00 | 0.0000e+00 | pass |
| hidden_dequant__bf16 | single_stage | a7c2bcfd | 16 | 0.996905 | 0.903460 | 2.0480e+03 | 1.4341e+02 | pass |
| hidden_dequant__f16 | single_stage | a7c2bcfd | 16 | 0.999503 | 0.988002 | 2.0480e+03 | 1.2304e+01 | pass |
| gemm1_operands__bf16 | single_stage | a7c2bcfd | 16 | 0.995658 | 0.867937 | 2.0480e+03 | 2.7541e+02 | pass |
| gemm1_operands__f16 | single_stage | a7c2bcfd | 16 | 0.999372 | 0.982945 | 2.0480e+03 | 2.1522e+01 | pass |
| gemm1_accumulator__bf16 | single_stage | a7c2bcfd | 16 | 0.983459 | 0.640215 | 7.1680e+03 | 6.4524e+02 | pass |
| gemm1_accumulator__f16 | single_stage | a7c2bcfd | 16 | 0.998117 | 0.938110 | 2.0480e+03 | 4.0205e+01 | pass |
| gemm1_output__bf16 | single_stage | a7c2bcfd | 16 | 0.997140 | 0.905465 | 2.0480e+03 | 6.4097e+01 | pass |
| gemm1_output__f16 | single_stage | a7c2bcfd | 16 | 0.999564 | 0.988342 | 2.0480e+03 | 2.5667e+01 | pass |
| swiglu_input__bf16 | single_stage | a7c2bcfd | 16 | 0.997140 | 0.905465 | 2.0480e+03 | 6.4097e+01 | pass |
| swiglu_input__f16 | single_stage | a7c2bcfd | 16 | 0.999564 | 0.988342 | 2.0480e+03 | 2.5667e+01 | pass |
| swiglu_output__bf16 | single_stage | a7c2bcfd | 16 | 0.997934 | 0.931876 | 2.0480e+03 | 6.9174e+01 | pass |
| swiglu_output__f16 | single_stage | a7c2bcfd | 16 | 0.999747 | 0.991804 | 2.0480e+03 | 6.7681e+00 | pass |
| gemm2_operands__bf16 | single_stage | a7c2bcfd | 16 | 0.996835 | 0.906686 | 2.0480e+03 | 1.8836e+02 | pass |
| gemm2_operands__f16 | single_stage | a7c2bcfd | 16 | 0.999538 | 0.988578 | 2.0480e+03 | 9.8406e+00 | pass |
| gemm2_accumulator__bf16 | single_stage | a7c2bcfd | 16 | 0.995911 | 0.860709 | 4.0960e+03 | 2.4852e+02 | pass |
| gemm2_accumulator__f16 | single_stage | a7c2bcfd | 16 | 0.594587 | 0.594230 | nan | nan | global_drift |
| out_accumulator__bf16 | single_stage | a7c2bcfd | 16 | 0.999390 | 0.981149 | 4.0960e+03 | 1.4989e+02 | pass |
| out_accumulator__f16 | single_stage | a7c2bcfd | 16 | 0.758780 | 0.757289 | inf | inf | catastrophic_outlier |
| baseline__fp32 | single_stage | 2e69caee | 15 | 1.000000 | 1.000000 | 0.0000e+00 | 0.0000e+00 | pass |
| hidden_dequant__bf16 | single_stage | 2e69caee | 15 | 0.998744 | 0.956045 | 2.0480e+03 | 4.6124e+01 | pass |
| hidden_dequant__f16 | single_stage | 2e69caee | 15 | 0.999860 | 0.995219 | 2.0480e+03 | 1.8539e+00 | pass |
| gemm1_operands__bf16 | single_stage | 2e69caee | 15 | 0.998177 | 0.940923 | 2.0480e+03 | 8.4171e+01 | pass |
| gemm1_operands__f16 | single_stage | 2e69caee | 15 | 0.999814 | 0.993090 | 2.0480e+03 | 1.5325e+00 | pass |
| gemm1_accumulator__bf16 | single_stage | 2e69caee | 15 | 0.993741 | 0.850149 | 6.6560e+03 | 1.6993e+02 | pass |
| gemm1_accumulator__f16 | single_stage | 2e69caee | 15 | 0.999191 | 0.971596 | 2.0480e+03 | 2.0512e+01 | pass |
| gemm1_output__bf16 | single_stage | 2e69caee | 15 | 0.998754 | 0.958715 | 2.0480e+03 | 1.8522e+01 | pass |
| gemm1_output__f16 | single_stage | 2e69caee | 15 | 0.999842 | 0.995461 | 2.0480e+03 | 2.6364e+00 | pass |
| swiglu_input__bf16 | single_stage | 2e69caee | 15 | 0.998754 | 0.958715 | 2.0480e+03 | 1.8522e+01 | pass |
| swiglu_input__f16 | single_stage | 2e69caee | 15 | 0.999842 | 0.995461 | 2.0480e+03 | 2.6364e+00 | pass |
| swiglu_output__bf16 | single_stage | 2e69caee | 15 | 0.999247 | 0.971754 | 2.0480e+03 | 1.0268e+01 | pass |
| swiglu_output__f16 | single_stage | 2e69caee | 15 | 0.933315 | 0.930943 | nan | nan | pass |
| gemm2_operands__bf16 | single_stage | 2e69caee | 15 | 0.998986 | 0.960342 | 4.0960e+03 | 2.4364e+01 | pass |
| gemm2_operands__f16 | single_stage | 2e69caee | 15 | 0.933259 | 0.929697 | nan | nan | pass |
| gemm2_accumulator__bf16 | single_stage | 2e69caee | 15 | 0.998354 | 0.938653 | 6.1440e+03 | 4.6795e+01 | pass |
| gemm2_accumulator__f16 | single_stage | 2e69caee | 15 | 0.873428 | 0.873289 | nan | nan | global_drift |
| out_accumulator__bf16 | single_stage | 2e69caee | 15 | 0.999674 | 0.989062 | 4.0960e+03 | 3.3659e+00 | pass |
| out_accumulator__f16 | single_stage | 2e69caee | 15 | 0.841815 | 0.841350 | inf | inf | catastrophic_outlier |
| baseline__fp32 | single_stage | 8cba5890 | 14 | 1.000000 | 1.000000 | 0.0000e+00 | 0.0000e+00 | pass |
| hidden_dequant__bf16 | single_stage | 8cba5890 | 14 | 0.997718 | 0.934072 | 2.0480e+03 | 1.5691e+02 | pass |
| hidden_dequant__f16 | single_stage | 8cba5890 | 14 | 0.999641 | 0.991619 | 2.0480e+03 | 2.1935e+01 | pass |
| gemm1_operands__bf16 | single_stage | 8cba5890 | 14 | 0.996841 | 0.909319 | 2.0480e+03 | 4.4451e+02 | pass |
| gemm1_operands__f16 | single_stage | 8cba5890 | 14 | 0.999562 | 0.988192 | 2.0480e+03 | 3.0584e+01 | pass |
| gemm1_accumulator__bf16 | single_stage | 8cba5890 | 14 | 0.988710 | 0.758799 | 5.1200e+03 | 8.2219e+02 | pass |
| gemm1_accumulator__f16 | single_stage | 8cba5890 | 14 | 0.998665 | 0.958048 | 2.0480e+03 | 1.1504e+02 | pass |
| gemm1_output__bf16 | single_stage | 8cba5890 | 14 | 0.998067 | 0.935955 | 2.0480e+03 | 2.1178e+02 | pass |
| gemm1_output__f16 | single_stage | 8cba5890 | 14 | 0.999691 | 0.991420 | 2.0480e+03 | 7.0981e+01 | pass |
| swiglu_input__bf16 | single_stage | 8cba5890 | 14 | 0.998067 | 0.935955 | 2.0480e+03 | 2.1178e+02 | pass |
| swiglu_input__f16 | single_stage | 8cba5890 | 14 | 0.999691 | 0.991420 | 2.0480e+03 | 7.0981e+01 | pass |
| swiglu_output__bf16 | single_stage | 8cba5890 | 14 | 0.998455 | 0.955148 | 2.0480e+03 | 1.9638e+02 | pass |
| swiglu_output__f16 | single_stage | 8cba5890 | 14 | 0.928412 | 0.923639 | nan | nan | pass |
| gemm2_operands__bf16 | single_stage | 8cba5890 | 14 | 0.997828 | 0.937062 | 2.0480e+03 | 1.0988e+02 | pass |
| gemm2_operands__f16 | single_stage | 8cba5890 | 14 | 0.928342 | 0.921945 | nan | nan | pass |
| gemm2_accumulator__bf16 | single_stage | 8cba5890 | 14 | 0.996831 | 0.897610 | 4.0960e+03 | 3.1442e+02 | pass |
| gemm2_accumulator__f16 | single_stage | 8cba5890 | 14 | 0.795908 | 0.795709 | nan | nan | global_drift |
| out_accumulator__bf16 | single_stage | 8cba5890 | 14 | 0.999233 | 0.979004 | 2.0480e+03 | 8.5857e+01 | pass |
| out_accumulator__f16 | single_stage | 8cba5890 | 14 | 0.802206 | 0.800403 | inf | inf | catastrophic_outlier |
| baseline__fp32 | single_stage | 5e8dc11c | 14107 | 1.000000 | 1.000000 | 0.0000e+00 | 0.0000e+00 | pass |
| hidden_dequant__bf16 | single_stage | 5e8dc11c | 14107 | 0.997176 | 0.912928 | 8.1920e+03 | 4.7924e+04 | pass |
| hidden_dequant__f16 | single_stage | 5e8dc11c | 14107 | 0.999648 | 0.988900 | 4.0960e+03 | 2.3006e+04 | pass |
| gemm1_operands__bf16 | single_stage | 5e8dc11c | 14107 | 0.995999 | 0.879362 | 8.1920e+03 | 3.1606e+05 | pass |
| gemm1_operands__f16 | single_stage | 5e8dc11c | 14107 | 0.999500 | 0.984267 | 4.0960e+03 | 1.9713e+04 | pass |
| gemm1_accumulator__bf16 | single_stage | 5e8dc11c | 14107 | 0.985257 | 0.675835 | 2.2528e+04 | 6.9533e+05 | pass |
| gemm1_accumulator__f16 | single_stage | 5e8dc11c | 14107 | 0.998126 | 0.941560 | 4.0960e+03 | 6.0422e+04 | pass |
| gemm1_output__bf16 | single_stage | 5e8dc11c | 14107 | 0.997186 | 0.913225 | 4.0960e+03 | 1.5497e+05 | pass |
| gemm1_output__f16 | single_stage | 5e8dc11c | 14107 | 0.999647 | 0.988923 | 4.0960e+03 | 1.3226e+04 | pass |
| swiglu_input__bf16 | single_stage | 5e8dc11c | 14107 | 0.997186 | 0.913225 | 4.0960e+03 | 1.5497e+05 | pass |
| swiglu_input__f16 | single_stage | 5e8dc11c | 14107 | 0.999647 | 0.988923 | 4.0960e+03 | 1.3226e+04 | pass |
| swiglu_output__bf16 | single_stage | 5e8dc11c | 14107 | 0.998009 | 0.938025 | 8.1920e+03 | 1.6589e+05 | pass |
| swiglu_output__f16 | single_stage | 5e8dc11c | 14107 | 0.961845 | 0.954828 | nan | nan | pass |
| gemm2_operands__bf16 | single_stage | 5e8dc11c | 14107 | 0.997199 | 0.913685 | 8.1920e+03 | 1.9800e+05 | pass |
| gemm2_operands__f16 | single_stage | 5e8dc11c | 14107 | 0.961752 | 0.951883 | nan | nan | pass |
| gemm2_accumulator__bf16 | single_stage | 5e8dc11c | 14107 | 0.996198 | 0.871886 | 1.2288e+04 | 7.9223e+04 | pass |
| gemm2_accumulator__f16 | single_stage | 5e8dc11c | 14107 | 0.655234 | 0.654870 | nan | nan | global_drift |
| out_accumulator__bf16 | single_stage | 5e8dc11c | 14107 | 0.999371 | 0.979634 | 8.1920e+03 | 1.4465e+04 | pass |
| out_accumulator__f16 | single_stage | 5e8dc11c | 14107 | 0.745060 | 0.743502 | inf | inf | catastrophic_outlier |
| baseline__fp32 | single_stage | 58a34f27 | 11948 | 1.000000 | 1.000000 | 0.0000e+00 | 0.0000e+00 | pass |
| hidden_dequant__bf16 | single_stage | 58a34f27 | 11948 | 0.997839 | 0.933328 | 4.0960e+03 | 6.5500e+09 | pass |
| hidden_dequant__f16 | single_stage | 58a34f27 | 11948 | 0.999733 | 0.991482 | 4.0960e+03 | 1.5562e+09 | pass |
| gemm1_operands__bf16 | single_stage | 58a34f27 | 11948 | 0.996936 | 0.907295 | 4.0960e+03 | 2.8500e+09 | pass |
| gemm1_operands__f16 | single_stage | 58a34f27 | 11948 | 0.999617 | 0.987920 | 4.0960e+03 | 1.3000e+09 | pass |
| gemm1_accumulator__bf16 | single_stage | 58a34f27 | 11948 | 0.988743 | 0.751769 | 1.4336e+04 | 2.9760e+11 | pass |
| gemm1_accumulator__f16 | single_stage | 58a34f27 | 11948 | 0.998572 | 0.955326 | 4.0960e+03 | 4.2500e+09 | pass |
| gemm1_output__bf16 | single_stage | 58a34f27 | 11948 | 0.997851 | 0.933534 | 4.0960e+03 | 1.6200e+10 | pass |
| gemm1_output__f16 | single_stage | 58a34f27 | 11948 | 0.999731 | 0.991501 | 4.0960e+03 | 1.2000e+09 | pass |
| swiglu_input__bf16 | single_stage | 58a34f27 | 11948 | 0.997851 | 0.933534 | 4.0960e+03 | 1.6200e+10 | pass |
| swiglu_input__f16 | single_stage | 58a34f27 | 11948 | 0.999731 | 0.991501 | 4.0960e+03 | 1.2000e+09 | pass |
| swiglu_output__bf16 | single_stage | 58a34f27 | 11948 | 0.998482 | 0.952469 | 4.0960e+03 | 9.4000e+09 | pass |
| swiglu_output__f16 | single_stage | 58a34f27 | 11948 | 0.966038 | 0.960724 | nan | nan | pass |
| gemm2_operands__bf16 | single_stage | 58a34f27 | 11948 | 0.997857 | 0.933779 | 4.0960e+03 | 7.3500e+09 | pass |
| gemm2_operands__f16 | single_stage | 58a34f27 | 11948 | 0.965968 | 0.958492 | nan | nan | pass |
| gemm2_accumulator__bf16 | single_stage | 58a34f27 | 11948 | 0.997079 | 0.901454 | 1.2288e+04 | 1.4800e+10 | pass |
| gemm2_accumulator__f16 | single_stage | 58a34f27 | 11948 | 0.735056 | 0.734768 | nan | nan | global_drift |
| out_accumulator__bf16 | single_stage | 58a34f27 | 11948 | 0.999513 | 0.984253 | 4.0960e+03 | 3.3250e+09 | pass |
| out_accumulator__f16 | single_stage | 58a34f27 | 11948 | 0.800582 | 0.799383 | inf | inf | catastrophic_outlier |
| baseline__fp32 | single_stage | 5eadab1e | 62 | 1.000000 | 1.000000 | 0.0000e+00 | 0.0000e+00 | pass |
| hidden_dequant__bf16 | single_stage | 5eadab1e | 62 | 0.997554 | 0.927813 | 4.0960e+03 | 5.5140e+02 | pass |
| hidden_dequant__f16 | single_stage | 5eadab1e | 62 | 0.999667 | 0.990691 | 2.0480e+03 | 6.7400e+01 | pass |
| gemm1_operands__bf16 | single_stage | 5eadab1e | 62 | 0.996566 | 0.900310 | 2.0480e+03 | 1.2042e+03 | pass |
| gemm1_operands__f16 | single_stage | 5eadab1e | 62 | 0.999552 | 0.986780 | 2.0480e+03 | 1.4420e+02 | pass |
| gemm1_accumulator__bf16 | single_stage | 5eadab1e | 62 | 0.987757 | 0.735475 | 7.3440e+03 | 7.2714e+03 | pass |
| gemm1_accumulator__f16 | single_stage | 5eadab1e | 62 | 0.998382 | 0.952421 | 2.0480e+03 | 3.4460e+02 | pass |
| gemm1_output__bf16 | single_stage | 5eadab1e | 62 | 0.997689 | 0.929552 | 4.0960e+03 | 2.8580e+02 | pass |
| gemm1_output__f16 | single_stage | 5eadab1e | 62 | 0.999689 | 0.990795 | 2.0480e+03 | 2.6620e+01 | pass |
| swiglu_input__bf16 | single_stage | 5eadab1e | 62 | 0.997689 | 0.929552 | 4.0960e+03 | 2.8580e+02 | pass |
| swiglu_input__f16 | single_stage | 5eadab1e | 62 | 0.999689 | 0.990795 | 2.0480e+03 | 2.6620e+01 | pass |
| swiglu_output__bf16 | single_stage | 5eadab1e | 62 | 0.998261 | 0.949219 | 2.0480e+03 | 4.1180e+02 | pass |
| swiglu_output__f16 | single_stage | 5eadab1e | 62 | 0.991749 | 0.985536 | nan | nan | pass |
| gemm2_operands__bf16 | single_stage | 5eadab1e | 62 | 0.997604 | 0.929681 | 4.0960e+03 | 4.0220e+02 | pass |
| gemm2_operands__f16 | single_stage | 5eadab1e | 62 | 0.991659 | 0.983117 | nan | nan | pass |
| gemm2_accumulator__bf16 | single_stage | 5eadab1e | 62 | 0.996785 | 0.895632 | 4.0960e+03 | 1.0678e+03 | pass |
| gemm2_accumulator__f16 | single_stage | 5eadab1e | 62 | 0.706498 | 0.706048 | nan | nan | global_drift |
| out_accumulator__bf16 | single_stage | 5eadab1e | 62 | 0.999489 | 0.984240 | 2.0480e+03 | 6.6488e+01 | pass |
| out_accumulator__f16 | single_stage | 5eadab1e | 62 | 0.810308 | 0.809174 | inf | inf | catastrophic_outlier |
| baseline__fp32 | single_stage | eedc63b2 | 59 | 1.000000 | 1.000000 | 0.0000e+00 | 0.0000e+00 | pass |
| hidden_dequant__bf16 | single_stage | eedc63b2 | 59 | 0.998127 | 0.941342 | 4.0960e+03 | 1.4931e+02 | pass |
| hidden_dequant__f16 | single_stage | eedc63b2 | 59 | 0.999759 | 0.992641 | 2.0480e+03 | 9.2105e+00 | pass |
| gemm1_operands__bf16 | single_stage | eedc63b2 | 59 | 0.997212 | 0.917243 | 4.0960e+03 | 2.9375e+02 | pass |
| gemm1_operands__f16 | single_stage | eedc63b2 | 59 | 0.999676 | 0.989404 | 2.0480e+03 | 9.7636e+00 | pass |
| gemm1_accumulator__bf16 | single_stage | eedc63b2 | 59 | 0.990062 | 0.778479 | 9.2160e+03 | 5.9833e+02 | pass |
| gemm1_accumulator__f16 | single_stage | eedc63b2 | 59 | 0.998796 | 0.960209 | 2.0480e+03 | 4.6211e+01 | pass |
| gemm1_output__bf16 | single_stage | eedc63b2 | 59 | 0.998042 | 0.940661 | 2.0480e+03 | 2.3367e+02 | pass |
| gemm1_output__f16 | single_stage | eedc63b2 | 59 | 0.999780 | 0.992540 | 2.0480e+03 | 9.2807e+00 | pass |
| swiglu_input__bf16 | single_stage | eedc63b2 | 59 | 0.998042 | 0.940661 | 2.0480e+03 | 2.3367e+02 | pass |
| swiglu_input__f16 | single_stage | eedc63b2 | 59 | 0.999780 | 0.992540 | 2.0480e+03 | 9.2807e+00 | pass |
| swiglu_output__bf16 | single_stage | eedc63b2 | 59 | 0.998638 | 0.957417 | 4.0960e+03 | 2.0872e+02 | pass |
| swiglu_output__f16 | single_stage | eedc63b2 | 59 | 0.974501 | 0.969972 | nan | nan | pass |
| gemm2_operands__bf16 | single_stage | eedc63b2 | 59 | 0.998089 | 0.941080 | 4.0960e+03 | 2.5475e+02 | pass |
| gemm2_operands__f16 | single_stage | eedc63b2 | 59 | 0.974434 | 0.967759 | nan | nan | pass |
| gemm2_accumulator__bf16 | single_stage | eedc63b2 | 59 | 0.997531 | 0.913263 | 5.6320e+03 | 1.5302e+02 | pass |
| gemm2_accumulator__f16 | single_stage | eedc63b2 | 59 | 0.743223 | 0.742982 | nan | nan | global_drift |
| out_accumulator__bf16 | single_stage | eedc63b2 | 59 | 0.999633 | 0.988182 | 4.0960e+03 | 9.2193e+01 | pass |
| out_accumulator__f16 | single_stage | eedc63b2 | 59 | 0.828737 | 0.827688 | inf | inf | catastrophic_outlier |
| baseline__fp32 | single_stage | e626d3e6 | 58 | 1.000000 | 1.000000 | 0.0000e+00 | 0.0000e+00 | pass |
| hidden_dequant__bf16 | single_stage | e626d3e6 | 58 | 0.997104 | 0.911219 | 4.0960e+03 | 7.8798e+02 | pass |
| hidden_dequant__f16 | single_stage | e626d3e6 | 58 | 0.999644 | 0.988481 | 2.0480e+03 | 3.1644e+02 | pass |
| gemm1_operands__bf16 | single_stage | e626d3e6 | 58 | 0.995860 | 0.876128 | 4.0960e+03 | 2.8192e+03 | pass |
| gemm1_operands__f16 | single_stage | e626d3e6 | 58 | 0.999507 | 0.983742 | 2.0480e+03 | 7.0556e+02 | pass |
| gemm1_accumulator__bf16 | single_stage | e626d3e6 | 58 | 0.985118 | 0.668214 | 9.7280e+03 | 8.2601e+03 | pass |
| gemm1_accumulator__f16 | single_stage | e626d3e6 | 58 | 0.998143 | 0.940309 | 4.0960e+03 | 6.5936e+03 | pass |
| gemm1_output__bf16 | single_stage | e626d3e6 | 58 | 0.997087 | 0.911188 | 4.0960e+03 | 4.1372e+02 | pass |
| gemm1_output__f16 | single_stage | e626d3e6 | 58 | 0.999642 | 0.988640 | 4.0960e+03 | 5.2580e+02 | pass |
| swiglu_input__bf16 | single_stage | e626d3e6 | 58 | 0.997087 | 0.911188 | 4.0960e+03 | 4.1372e+02 | pass |
| swiglu_input__f16 | single_stage | e626d3e6 | 58 | 0.999642 | 0.988640 | 4.0960e+03 | 5.2580e+02 | pass |
| swiglu_output__bf16 | single_stage | e626d3e6 | 58 | 0.998056 | 0.936451 | 4.0960e+03 | 3.3392e+03 | pass |
| swiglu_output__f16 | single_stage | e626d3e6 | 58 | 0.896458 | 0.890084 | nan | nan | global_drift |
| gemm2_operands__bf16 | single_stage | e626d3e6 | 58 | 0.997200 | 0.911965 | 4.0960e+03 | 9.4628e+03 | pass |
| gemm2_operands__f16 | single_stage | e626d3e6 | 58 | 0.896386 | 0.887404 | nan | nan | global_drift |
| gemm2_accumulator__bf16 | single_stage | e626d3e6 | 58 | 0.995743 | 0.862793 | 8.1920e+03 | 9.0122e+03 | pass |
| gemm2_accumulator__f16 | single_stage | e626d3e6 | 58 | 0.702584 | 0.702189 | nan | nan | global_drift |
| out_accumulator__bf16 | single_stage | e626d3e6 | 58 | 0.999204 | 0.973686 | 4.0960e+03 | 3.6669e+03 | pass |
| out_accumulator__f16 | single_stage | e626d3e6 | 58 | 0.704823 | 0.702812 | inf | inf | catastrophic_outlier |
| baseline__fp32 | single_stage | 74d7ff04 | 57 | 1.000000 | 1.000000 | 0.0000e+00 | 0.0000e+00 | pass |
| hidden_dequant__bf16 | single_stage | 74d7ff04 | 57 | 0.997641 | 0.926427 | 4.0960e+03 | 5.7422e+02 | pass |
| hidden_dequant__f16 | single_stage | 74d7ff04 | 57 | 0.999704 | 0.990810 | 2.0480e+03 | 1.7696e+01 | pass |
| gemm1_operands__bf16 | single_stage | 74d7ff04 | 57 | 0.996715 | 0.897591 | 4.0960e+03 | 7.9126e+02 | pass |
| gemm1_operands__f16 | single_stage | 74d7ff04 | 57 | 0.999537 | 0.986874 | 2.0480e+03 | 3.2217e+01 | pass |
| gemm1_accumulator__bf16 | single_stage | 74d7ff04 | 57 | 0.987672 | 0.721496 | 8.1920e+03 | 1.7131e+03 | pass |
| gemm1_accumulator__f16 | single_stage | 74d7ff04 | 57 | 0.998365 | 0.950447 | 4.0960e+03 | 8.8088e+01 | pass |
| gemm1_output__bf16 | single_stage | 74d7ff04 | 57 | 0.997589 | 0.926119 | 4.0960e+03 | 5.7422e+02 | pass |
| gemm1_output__f16 | single_stage | 74d7ff04 | 57 | 0.999684 | 0.990846 | 2.0480e+03 | 5.5000e+01 | pass |
| swiglu_input__bf16 | single_stage | 74d7ff04 | 57 | 0.997589 | 0.926119 | 4.0960e+03 | 5.7422e+02 | pass |
| swiglu_input__f16 | single_stage | 74d7ff04 | 57 | 0.999684 | 0.990846 | 2.0480e+03 | 5.5000e+01 | pass |
| swiglu_output__bf16 | single_stage | 74d7ff04 | 57 | 0.998284 | 0.947545 | 2.0480e+03 | 4.3865e+02 | pass |
| swiglu_output__f16 | single_stage | 74d7ff04 | 57 | 0.982265 | 0.976154 | nan | nan | pass |
| gemm2_operands__bf16 | single_stage | 74d7ff04 | 57 | 0.997621 | 0.926503 | 4.0960e+03 | 5.1657e+02 | pass |
| gemm2_operands__f16 | single_stage | 74d7ff04 | 57 | 0.982155 | 0.973449 | nan | nan | pass |
| gemm2_accumulator__bf16 | single_stage | 74d7ff04 | 57 | 0.996703 | 0.889269 | 6.1440e+03 | 5.3804e+02 | pass |
| gemm2_accumulator__f16 | single_stage | 74d7ff04 | 57 | 0.720209 | 0.719935 | nan | nan | global_drift |
| out_accumulator__bf16 | single_stage | 74d7ff04 | 57 | 0.999398 | 0.981629 | 4.0960e+03 | 8.5779e+01 | pass |
| out_accumulator__f16 | single_stage | 74d7ff04 | 57 | 0.744522 | 0.743504 | inf | inf | catastrophic_outlier |
| baseline__fp32 | single_stage | 4822167c | 56 | 1.000000 | 1.000000 | 0.0000e+00 | 0.0000e+00 | pass |
| hidden_dequant__bf16 | single_stage | 4822167c | 56 | 0.997008 | 0.905894 | 2.0480e+03 | 1.9102e+02 | pass |
| hidden_dequant__f16 | single_stage | 4822167c | 56 | 0.999606 | 0.987843 | 2.0480e+03 | 5.6732e+01 | pass |
| gemm1_operands__bf16 | single_stage | 4822167c | 56 | 0.995551 | 0.869701 | 2.0480e+03 | 2.3125e+02 | pass |
| gemm1_operands__f16 | single_stage | 4822167c | 56 | 0.999410 | 0.982883 | 2.0480e+03 | 8.9164e+01 | pass |
| gemm1_accumulator__bf16 | single_stage | 4822167c | 56 | 0.984131 | 0.654740 | 6.4000e+03 | 2.3324e+03 | pass |
| gemm1_accumulator__f16 | single_stage | 4822167c | 56 | 0.997920 | 0.937458 | 4.0960e+03 | 1.4430e+02 | pass |
| gemm1_output__bf16 | single_stage | 4822167c | 56 | 0.996856 | 0.908049 | 2.0480e+03 | 9.6166e+02 | pass |
| gemm1_output__f16 | single_stage | 4822167c | 56 | 0.999559 | 0.988224 | 2.0480e+03 | 4.1742e+01 | pass |
| swiglu_input__bf16 | single_stage | 4822167c | 56 | 0.996856 | 0.908049 | 2.0480e+03 | 9.6166e+02 | pass |
| swiglu_input__f16 | single_stage | 4822167c | 56 | 0.999559 | 0.988224 | 2.0480e+03 | 4.1742e+01 | pass |
| swiglu_output__bf16 | single_stage | 4822167c | 56 | 0.997793 | 0.933796 | 4.0960e+03 | 1.0900e+02 | pass |
| swiglu_output__f16 | single_stage | 4822167c | 56 | 0.928337 | 0.921175 | nan | nan | pass |
| gemm2_operands__bf16 | single_stage | 4822167c | 56 | 0.996903 | 0.907949 | 4.0960e+03 | 1.8900e+02 | pass |
| gemm2_operands__f16 | single_stage | 4822167c | 56 | 0.928267 | 0.918335 | nan | nan | pass |
| gemm2_accumulator__bf16 | single_stage | 4822167c | 56 | 0.995685 | 0.861869 | 4.0960e+03 | 4.7933e+02 | pass |
| gemm2_accumulator__f16 | single_stage | 4822167c | 56 | 0.652311 | 0.651958 | nan | nan | global_drift |
| out_accumulator__bf16 | single_stage | 4822167c | 56 | 0.999200 | 0.975857 | 4.0960e+03 | 1.0589e+02 | pass |
| out_accumulator__f16 | single_stage | 4822167c | 56 | 0.736278 | 0.734407 | inf | inf | catastrophic_outlier |
| baseline__fp32 | single_stage | 81955b1e | 55 | 1.000000 | 1.000000 | 0.0000e+00 | 0.0000e+00 | pass |
| hidden_dequant__bf16 | single_stage | 81955b1e | 55 | 0.997677 | 0.929520 | 4.0960e+03 | 9.0622e+02 | pass |
| hidden_dequant__f16 | single_stage | 81955b1e | 55 | 0.999691 | 0.990922 | 2.0480e+03 | 1.3156e+02 | pass |
| gemm1_operands__bf16 | single_stage | 81955b1e | 55 | 0.996723 | 0.900670 | 4.0960e+03 | 1.4914e+03 | pass |
| gemm1_operands__f16 | single_stage | 81955b1e | 55 | 0.999581 | 0.987165 | 2.0480e+03 | 1.1006e+02 | pass |
| gemm1_accumulator__bf16 | single_stage | 81955b1e | 55 | 0.988154 | 0.735341 | 8.7040e+03 | 2.7492e+03 | pass |
| gemm1_accumulator__f16 | single_stage | 81955b1e | 55 | 0.998402 | 0.952171 | 2.0480e+03 | 6.9122e+02 | pass |
| gemm1_output__bf16 | single_stage | 81955b1e | 55 | 0.997644 | 0.928533 | 4.0960e+03 | 1.1479e+03 | pass |
| gemm1_output__f16 | single_stage | 81955b1e | 55 | 0.999678 | 0.990681 | 4.0960e+03 | 1.0239e+02 | pass |
| swiglu_input__bf16 | single_stage | 81955b1e | 55 | 0.997644 | 0.928533 | 4.0960e+03 | 1.1479e+03 | pass |
| swiglu_input__f16 | single_stage | 81955b1e | 55 | 0.999678 | 0.990681 | 4.0960e+03 | 1.0239e+02 | pass |
| swiglu_output__bf16 | single_stage | 81955b1e | 55 | 0.998280 | 0.948326 | 2.0480e+03 | 4.1679e+02 | pass |
| swiglu_output__f16 | single_stage | 81955b1e | 55 | 0.958934 | 0.953389 | nan | nan | pass |
| gemm2_operands__bf16 | single_stage | 81955b1e | 55 | 0.997671 | 0.928336 | 4.0960e+03 | 6.8029e+02 | pass |
| gemm2_operands__f16 | single_stage | 81955b1e | 55 | 0.958873 | 0.951200 | nan | nan | pass |
| gemm2_accumulator__bf16 | single_stage | 81955b1e | 55 | 0.996779 | 0.892865 | 5.1200e+03 | 1.1649e+03 | pass |
| gemm2_accumulator__f16 | single_stage | 81955b1e | 55 | 0.728417 | 0.728143 | nan | nan | global_drift |
| out_accumulator__bf16 | single_stage | 81955b1e | 55 | 0.999450 | 0.981950 | 4.0960e+03 | 2.0526e+02 | pass |
| out_accumulator__f16 | single_stage | 81955b1e | 55 | 0.761105 | 0.759953 | inf | inf | catastrophic_outlier |
| baseline__fp32 | single_stage | 76010cb4 | 54 | 1.000000 | 1.000000 | 0.0000e+00 | 0.0000e+00 | pass |
| hidden_dequant__bf16 | single_stage | 76010cb4 | 54 | 0.998006 | 0.940533 | 4.0960e+03 | 7.7654e+03 | pass |
| hidden_dequant__f16 | single_stage | 76010cb4 | 54 | 0.999765 | 0.992467 | 2.0480e+03 | 2.8735e+03 | pass |
| gemm1_operands__bf16 | single_stage | 76010cb4 | 54 | 0.997215 | 0.917377 | 4.0960e+03 | 2.4681e+04 | pass |
| gemm1_operands__f16 | single_stage | 76010cb4 | 54 | 0.999656 | 0.989423 | 2.0480e+03 | 3.9374e+03 | pass |
| gemm1_accumulator__bf16 | single_stage | 76010cb4 | 54 | 0.989547 | 0.777349 | 8.1920e+03 | 5.7024e+04 | pass |
| gemm1_accumulator__f16 | single_stage | 76010cb4 | 54 | 0.998685 | 0.959858 | 4.0960e+03 | 2.7873e+04 | pass |
| gemm1_output__bf16 | single_stage | 76010cb4 | 54 | 0.997972 | 0.940587 | 2.0480e+03 | 2.7873e+04 | pass |
| gemm1_output__f16 | single_stage | 76010cb4 | 54 | 0.999762 | 0.992412 | 2.0480e+03 | 3.1794e+03 | pass |
| swiglu_input__bf16 | single_stage | 76010cb4 | 54 | 0.997972 | 0.940587 | 2.0480e+03 | 2.7873e+04 | pass |
| swiglu_input__f16 | single_stage | 76010cb4 | 54 | 0.999762 | 0.992412 | 2.0480e+03 | 3.1794e+03 | pass |
| swiglu_output__bf16 | single_stage | 76010cb4 | 54 | 0.998489 | 0.956812 | 2.0480e+03 | 2.0747e+04 | pass |
| swiglu_output__f16 | single_stage | 76010cb4 | 54 | 0.962847 | 0.958124 | nan | nan | pass |
| gemm2_operands__bf16 | single_stage | 76010cb4 | 54 | 0.997902 | 0.940518 | 2.0480e+03 | 2.5497e+03 | pass |
| gemm2_operands__f16 | single_stage | 76010cb4 | 54 | 0.962769 | 0.956145 | nan | nan | pass |
| gemm2_accumulator__bf16 | single_stage | 76010cb4 | 54 | 0.997336 | 0.911471 | 6.1440e+03 | 1.0478e+04 | pass |
| gemm2_accumulator__f16 | single_stage | 76010cb4 | 54 | 0.775949 | 0.775654 | nan | nan | global_drift |
| out_accumulator__bf16 | single_stage | 76010cb4 | 54 | 0.999556 | 0.985597 | 4.0960e+03 | 3.0644e+02 | pass |
| out_accumulator__f16 | single_stage | 76010cb4 | 54 | 0.797521 | 0.796487 | inf | inf | catastrophic_outlier |
| baseline__fp32 | single_stage | fc378037 | 53 | 1.000000 | 1.000000 | 0.0000e+00 | 0.0000e+00 | pass |
| hidden_dequant__bf16 | single_stage | fc378037 | 53 | 0.997670 | 0.929624 | 4.0960e+03 | 2.8395e+02 | pass |
| hidden_dequant__f16 | single_stage | fc378037 | 53 | 0.999710 | 0.991042 | 2.0480e+03 | 5.3812e+01 | pass |
| gemm1_operands__bf16 | single_stage | fc378037 | 53 | 0.996857 | 0.903713 | 4.0960e+03 | 7.1941e+02 | pass |
| gemm1_operands__f16 | single_stage | fc378037 | 53 | 0.999618 | 0.987347 | 2.0480e+03 | 3.5021e+01 | pass |
| gemm1_accumulator__bf16 | single_stage | fc378037 | 53 | 0.988242 | 0.743580 | 9.2160e+03 | 1.0331e+03 | pass |
| gemm1_accumulator__f16 | single_stage | fc378037 | 53 | 0.998594 | 0.954275 | 4.0960e+03 | 2.1843e+02 | pass |
| gemm1_output__bf16 | single_stage | fc378037 | 53 | 0.997692 | 0.932125 | 4.0960e+03 | 1.7327e+02 | pass |
| gemm1_output__f16 | single_stage | fc378037 | 53 | 0.999739 | 0.991040 | 4.0960e+03 | 4.1220e+01 | pass |
| swiglu_input__bf16 | single_stage | fc378037 | 53 | 0.997692 | 0.932125 | 4.0960e+03 | 1.7327e+02 | pass |
| swiglu_input__f16 | single_stage | fc378037 | 53 | 0.999739 | 0.991040 | 4.0960e+03 | 4.1220e+01 | pass |
| swiglu_output__bf16 | single_stage | fc378037 | 53 | 0.998402 | 0.950548 | 4.0960e+03 | 2.9755e+02 | pass |
| swiglu_output__f16 | single_stage | fc378037 | 53 | 0.943262 | 0.937774 | nan | nan | pass |
| gemm2_operands__bf16 | single_stage | fc378037 | 53 | 0.997784 | 0.930927 | 4.0960e+03 | 2.9629e+02 | pass |
| gemm2_operands__f16 | single_stage | fc378037 | 53 | 0.943191 | 0.935668 | nan | nan | pass |
| gemm2_accumulator__bf16 | single_stage | fc378037 | 53 | 0.996923 | 0.897169 | 6.1440e+03 | 2.4441e+02 | pass |
| gemm2_accumulator__f16 | single_stage | fc378037 | 53 | 0.750790 | 0.750526 | nan | nan | global_drift |
| out_accumulator__bf16 | single_stage | fc378037 | 53 | 0.999452 | 0.981711 | 4.0960e+03 | 3.1999e+02 | pass |
| out_accumulator__f16 | single_stage | fc378037 | 53 | 0.782277 | 0.781047 | inf | inf | catastrophic_outlier |
| baseline__fp32 | single_stage | f7d6ac7c | 52 | 1.000000 | 1.000000 | 0.0000e+00 | 0.0000e+00 | pass |
| hidden_dequant__bf16 | single_stage | f7d6ac7c | 52 | 0.998476 | 0.955336 | 2.0480e+03 | 1.2654e+02 | pass |
| hidden_dequant__f16 | single_stage | f7d6ac7c | 52 | 0.999826 | 0.994253 | 2.0480e+03 | 3.3298e+01 | pass |
| gemm1_operands__bf16 | single_stage | f7d6ac7c | 52 | 0.997926 | 0.938047 | 2.0480e+03 | 2.0654e+02 | pass |
| gemm1_operands__f16 | single_stage | f7d6ac7c | 52 | 0.999753 | 0.991908 | 2.0480e+03 | 4.9210e+01 | pass |
| gemm1_accumulator__bf16 | single_stage | f7d6ac7c | 52 | 0.992579 | 0.835768 | 6.1440e+03 | 9.0054e+02 | pass |
| gemm1_accumulator__f16 | single_stage | f7d6ac7c | 52 | 0.998994 | 0.969480 | 2.0480e+03 | 4.7162e+01 | pass |
| gemm1_output__bf16 | single_stage | f7d6ac7c | 52 | 0.998524 | 0.956135 | 2.0480e+03 | 2.7822e+02 | pass |
| gemm1_output__f16 | single_stage | f7d6ac7c | 52 | 0.999826 | 0.994567 | 2.0480e+03 | 8.7692e+00 | pass |
| swiglu_input__bf16 | single_stage | f7d6ac7c | 52 | 0.998524 | 0.956135 | 2.0480e+03 | 2.7822e+02 | pass |
| swiglu_input__f16 | single_stage | f7d6ac7c | 52 | 0.999826 | 0.994567 | 2.0480e+03 | 8.7692e+00 | pass |
| swiglu_output__bf16 | single_stage | f7d6ac7c | 52 | 0.998951 | 0.968938 | 2.0480e+03 | 3.9136e+02 | pass |
| swiglu_output__f16 | single_stage | f7d6ac7c | 52 | 0.999893 | 0.996150 | 2.0480e+03 | 3.3829e+01 | pass |
| gemm2_operands__bf16 | single_stage | f7d6ac7c | 52 | 0.998559 | 0.956395 | 2.0480e+03 | 6.4029e+02 | pass |
| gemm2_operands__f16 | single_stage | f7d6ac7c | 52 | 0.999826 | 0.994505 | 2.0480e+03 | 5.4514e+01 | pass |
| gemm2_accumulator__bf16 | single_stage | f7d6ac7c | 52 | 0.998184 | 0.935794 | 4.0960e+03 | 3.5960e+02 | pass |
| gemm2_accumulator__f16 | single_stage | f7d6ac7c | 52 | 0.809369 | 0.809031 | nan | nan | global_drift |
| out_accumulator__bf16 | single_stage | f7d6ac7c | 52 | 0.999737 | 0.991533 | 2.0480e+03 | 6.2312e+01 | pass |
| out_accumulator__f16 | single_stage | f7d6ac7c | 52 | 0.879381 | 0.878705 | inf | inf | catastrophic_outlier |
| gemm1_accumulator__f16 | cumulative | b8f4f012 | 7 | 0.999183 | 0.971460 | 2.0480e+03 | 3.3276e+01 | pass |
| gemm1_accumulator__f16 | cumulative | e05c6c03 | 1 | 0.997349 | 0.898019 | 2.0480e+03 | 1.9684e+00 | pass |
| gemm1_accumulator__f16 | cumulative | 6230e838 | 32 | 0.998784 | 0.958670 | 4.0960e+03 | 2.1941e+02 | pass |
| gemm1_accumulator__f16 | cumulative | 8f1ff9f1 | 80 | 0.997974 | 0.938138 | 4.0960e+03 | 1.1906e+03 | pass |
| gemm1_accumulator__f16 | cumulative | 1a4c6ba1 | 901 | 0.997945 | 0.935820 | 4.0960e+03 | 3.5885e+03 | pass |
| gemm1_accumulator__f16 | cumulative | a7c2bcfd | 16 | 0.997872 | 0.936663 | 2.0480e+03 | 4.4053e+01 | pass |
| gemm1_accumulator__f16 | cumulative | 2e69caee | 15 | 0.999061 | 0.973382 | 2.0480e+03 | 2.1800e+01 | pass |
| gemm1_accumulator__f16 | cumulative | 8cba5890 | 14 | 0.998485 | 0.954620 | 2.0480e+03 | 9.8173e+01 | pass |
| gemm1_accumulator__f16 | cumulative | 5e8dc11c | 14107 | 0.998131 | 0.941555 | 4.0960e+03 | 1.8561e+04 | pass |
| gemm1_accumulator__f16 | cumulative | 58a34f27 | 11948 | 0.998560 | 0.955137 | 4.0960e+03 | 6.6053e+04 | pass |
| gemm1_accumulator__f16 | cumulative | 5eadab1e | 62 | 0.998441 | 0.952511 | 4.0960e+03 | 1.5063e+02 | pass |
| gemm1_accumulator__f16 | cumulative | eedc63b2 | 59 | 0.998740 | 0.960051 | 2.0480e+03 | 1.1014e+03 | pass |
| gemm1_accumulator__f16 | cumulative | e626d3e6 | 58 | 0.998090 | 0.940026 | 4.0960e+03 | 8.9519e+01 | pass |
| gemm1_accumulator__f16 | cumulative | 74d7ff04 | 57 | 0.998348 | 0.949598 | 2.0480e+03 | 1.0970e+03 | pass |
| gemm1_accumulator__f16 | cumulative | 4822167c | 56 | 0.998092 | 0.938820 | 4.0960e+03 | 2.4874e+02 | pass |
| gemm1_accumulator__f16 | cumulative | 81955b1e | 55 | 0.998389 | 0.951689 | 2.0480e+03 | 5.0315e+02 | pass |
| gemm1_accumulator__f16 | cumulative | 76010cb4 | 54 | 0.998732 | 0.959842 | 4.0960e+03 | 5.1068e+01 | pass |
| gemm1_accumulator__f16 | cumulative | fc378037 | 53 | 0.998489 | 0.954891 | 4.0960e+03 | 6.8424e+01 | pass |
| gemm1_accumulator__f16 | cumulative | f7d6ac7c | 52 | 0.999115 | 0.970585 | 2.0480e+03 | 1.9407e+02 | pass |
| gemm1_accumulator__f16+gemm1_operands__f16 | cumulative | b8f4f012 | 7 | 0.999681 | 0.991749 | 2.0480e+03 | 5.8772e+00 | pass |
| gemm1_accumulator__f16+gemm1_operands__f16 | cumulative | e05c6c03 | 1 | 0.998744 | 0.972656 | 2.0480e+03 | 1.5390e+01 | pass |
| gemm1_accumulator__f16+gemm1_operands__f16 | cumulative | 6230e838 | 32 | 0.999586 | 0.988595 | 2.0480e+03 | 3.8876e+02 | pass |
| gemm1_accumulator__f16+gemm1_operands__f16 | cumulative | 8f1ff9f1 | 80 | 0.999468 | 0.983423 | 2.0480e+03 | 8.4117e+01 | pass |
| gemm1_accumulator__f16+gemm1_operands__f16 | cumulative | 1a4c6ba1 | 901 | 0.999448 | 0.982700 | 4.0960e+03 | 6.5270e+03 | pass |
| gemm1_accumulator__f16+gemm1_operands__f16 | cumulative | a7c2bcfd | 16 | 0.999433 | 0.983320 | 2.0480e+03 | 2.1348e+02 | pass |
| gemm1_accumulator__f16+gemm1_operands__f16 | cumulative | 2e69caee | 15 | 0.999795 | 0.992476 | 2.0480e+03 | 3.1509e+00 | pass |
| gemm1_accumulator__f16+gemm1_operands__f16 | cumulative | 8cba5890 | 14 | 0.999671 | 0.988281 | 2.0480e+03 | 7.7843e+00 | pass |
| gemm1_accumulator__f16+gemm1_operands__f16 | cumulative | 5e8dc11c | 14107 | 0.999499 | 0.984223 | 4.0960e+03 | 1.6385e+04 | pass |
| gemm1_accumulator__f16+gemm1_operands__f16 | cumulative | 58a34f27 | 11948 | 0.999615 | 0.987910 | 4.0960e+03 | 9.2810e+03 | pass |
| gemm1_accumulator__f16+gemm1_operands__f16 | cumulative | 5eadab1e | 62 | 0.999586 | 0.987149 | 2.0480e+03 | 6.5432e+01 | pass |
| gemm1_accumulator__f16+gemm1_operands__f16 | cumulative | eedc63b2 | 59 | 0.999626 | 0.989348 | 4.0960e+03 | 4.3478e+02 | pass |
| gemm1_accumulator__f16+gemm1_operands__f16 | cumulative | e626d3e6 | 58 | 0.999480 | 0.983923 | 4.0960e+03 | 1.2617e+02 | pass |
| gemm1_accumulator__f16+gemm1_operands__f16 | cumulative | 74d7ff04 | 57 | 0.999638 | 0.986590 | 2.0480e+03 | 3.1186e+02 | pass |
| gemm1_accumulator__f16+gemm1_operands__f16 | cumulative | 4822167c | 56 | 0.999432 | 0.983525 | 2.0480e+03 | 9.2538e+01 | pass |
| gemm1_accumulator__f16+gemm1_operands__f16 | cumulative | 81955b1e | 55 | 0.999521 | 0.986998 | 4.0960e+03 | 3.5870e+01 | pass |
| gemm1_accumulator__f16+gemm1_operands__f16 | cumulative | 76010cb4 | 54 | 0.999631 | 0.989227 | 2.0480e+03 | 9.0476e+01 | pass |
| gemm1_accumulator__f16+gemm1_operands__f16 | cumulative | fc378037 | 53 | 0.999560 | 0.987497 | 2.0480e+03 | 1.0806e+03 | pass |
| gemm1_accumulator__f16+gemm1_operands__f16 | cumulative | f7d6ac7c | 52 | 0.999756 | 0.991844 | 2.0480e+03 | 1.3899e+01 | pass |
| gemm1_accumulator__f16+gemm1_operands__f16+gemm1_output__f16 | cumulative | b8f4f012 | 7 | 0.999761 | 0.990015 | 2.0480e+03 | 4.6500e+01 | pass |
| gemm1_accumulator__f16+gemm1_operands__f16+gemm1_output__f16 | cumulative | e05c6c03 | 1 | 0.999302 | 0.964844 | 2.0480e+03 | 8.1946e+00 | pass |
| gemm1_accumulator__f16+gemm1_operands__f16+gemm1_output__f16 | cumulative | 6230e838 | 32 | 0.999581 | 0.986747 | 2.0480e+03 | 2.3933e+02 | pass |
| gemm1_accumulator__f16+gemm1_operands__f16+gemm1_output__f16 | cumulative | 8f1ff9f1 | 80 | 0.999376 | 0.979749 | 2.0480e+03 | 1.5305e+02 | pass |
| gemm1_accumulator__f16+gemm1_operands__f16+gemm1_output__f16 | cumulative | 1a4c6ba1 | 901 | 0.999330 | 0.978785 | 4.0960e+03 | 5.9349e+02 | pass |
| gemm1_accumulator__f16+gemm1_operands__f16+gemm1_output__f16 | cumulative | a7c2bcfd | 16 | 0.999390 | 0.979056 | 2.0480e+03 | 3.8371e+02 | pass |
| gemm1_accumulator__f16+gemm1_operands__f16+gemm1_output__f16 | cumulative | 2e69caee | 15 | 0.999730 | 0.990727 | 2.0480e+03 | 2.2333e+01 | pass |
| gemm1_accumulator__f16+gemm1_operands__f16+gemm1_output__f16 | cumulative | 8cba5890 | 14 | 0.999502 | 0.985900 | 2.0480e+03 | 2.9634e+01 | pass |
| gemm1_accumulator__f16+gemm1_operands__f16+gemm1_output__f16 | cumulative | 5e8dc11c | 14107 | 0.999389 | 0.980731 | 4.0960e+03 | 1.7663e+04 | pass |
| gemm1_accumulator__f16+gemm1_operands__f16+gemm1_output__f16 | cumulative | 58a34f27 | 11948 | 0.999528 | 0.985233 | 4.0960e+03 | 5.3986e+05 | pass |
| gemm1_accumulator__f16+gemm1_operands__f16+gemm1_output__f16 | cumulative | 5eadab1e | 62 | 0.999464 | 0.984022 | 2.0480e+03 | 7.5626e+01 | pass |
| gemm1_accumulator__f16+gemm1_operands__f16+gemm1_output__f16 | cumulative | eedc63b2 | 59 | 0.999577 | 0.986841 | 2.0480e+03 | 9.7065e+01 | pass |
| gemm1_accumulator__f16+gemm1_operands__f16+gemm1_output__f16 | cumulative | e626d3e6 | 58 | 0.999370 | 0.979911 | 4.0960e+03 | 4.5133e+01 | pass |
| gemm1_accumulator__f16+gemm1_operands__f16+gemm1_output__f16 | cumulative | 74d7ff04 | 57 | 0.999488 | 0.983548 | 2.0480e+03 | 8.0568e+01 | pass |
| gemm1_accumulator__f16+gemm1_operands__f16+gemm1_output__f16 | cumulative | 4822167c | 56 | 0.999347 | 0.979313 | 2.0480e+03 | 4.8624e+01 | pass |
| gemm1_accumulator__f16+gemm1_operands__f16+gemm1_output__f16 | cumulative | 81955b1e | 55 | 0.999450 | 0.984038 | 2.0480e+03 | 9.6016e+01 | pass |
| gemm1_accumulator__f16+gemm1_operands__f16+gemm1_output__f16 | cumulative | 76010cb4 | 54 | 0.999607 | 0.986726 | 2.0480e+03 | 1.1633e+02 | pass |
| gemm1_accumulator__f16+gemm1_operands__f16+gemm1_output__f16 | cumulative | fc378037 | 53 | 0.999495 | 0.984615 | 4.0960e+03 | 2.7245e+02 | pass |
| gemm1_accumulator__f16+gemm1_operands__f16+gemm1_output__f16 | cumulative | f7d6ac7c | 52 | 0.999635 | 0.990014 | 2.0480e+03 | 1.4614e+02 | pass |
| gemm1_accumulator__f16+gemm1_operands__f16+gemm1_output__f16+gemm2_accumulator__bf16 | cumulative | b8f4f012 | 7 | 0.997947 | 0.927754 | 4.0960e+03 | 9.6179e+02 | pass |
| gemm1_accumulator__f16+gemm1_operands__f16+gemm1_output__f16+gemm2_accumulator__bf16 | cumulative | e05c6c03 | 1 | 0.989955 | 0.732840 | 6.1440e+03 | 3.0461e+01 | pass |
| gemm1_accumulator__f16+gemm1_operands__f16+gemm1_output__f16+gemm2_accumulator__bf16 | cumulative | 6230e838 | 32 | 0.996831 | 0.900569 | 5.1200e+03 | 1.7561e+02 | pass |
| gemm1_accumulator__f16+gemm1_operands__f16+gemm1_output__f16+gemm2_accumulator__bf16 | cumulative | 8f1ff9f1 | 80 | 0.995761 | 0.861079 | 6.1440e+03 | 1.8578e+02 | pass |
| gemm1_accumulator__f16+gemm1_operands__f16+gemm1_output__f16+gemm2_accumulator__bf16 | cumulative | 1a4c6ba1 | 901 | 0.995490 | 0.851314 | 8.1920e+03 | 5.6630e+03 | pass |
| gemm1_accumulator__f16+gemm1_operands__f16+gemm1_output__f16+gemm2_accumulator__bf16 | cumulative | a7c2bcfd | 16 | 0.995701 | 0.860099 | 4.0960e+03 | 4.5361e+02 | pass |
| gemm1_accumulator__f16+gemm1_operands__f16+gemm1_output__f16+gemm2_accumulator__bf16 | cumulative | 2e69caee | 15 | 0.998093 | 0.937528 | 4.0960e+03 | 6.8534e+01 | pass |
| gemm1_accumulator__f16+gemm1_operands__f16+gemm1_output__f16+gemm2_accumulator__bf16 | cumulative | 8cba5890 | 14 | 0.996672 | 0.895847 | 4.0960e+03 | 1.9155e+02 | pass |
| gemm1_accumulator__f16+gemm1_operands__f16+gemm1_output__f16+gemm2_accumulator__bf16 | cumulative | 5e8dc11c | 14107 | 0.996117 | 0.869372 | 1.6384e+04 | 4.0403e+05 | pass |
| gemm1_accumulator__f16+gemm1_operands__f16+gemm1_output__f16+gemm2_accumulator__bf16 | cumulative | 58a34f27 | 11948 | 0.997031 | 0.900446 | 1.6384e+04 | 1.5613e+05 | pass |
| gemm1_accumulator__f16+gemm1_operands__f16+gemm1_output__f16+gemm2_accumulator__bf16 | cumulative | 5eadab1e | 62 | 0.996886 | 0.895094 | 8.1920e+03 | 2.5700e+02 | pass |
| gemm1_accumulator__f16+gemm1_operands__f16+gemm1_output__f16+gemm2_accumulator__bf16 | cumulative | eedc63b2 | 59 | 0.997285 | 0.912573 | 6.1440e+03 | 1.8732e+02 | pass |
| gemm1_accumulator__f16+gemm1_operands__f16+gemm1_output__f16+gemm2_accumulator__bf16 | cumulative | e626d3e6 | 58 | 0.995613 | 0.862940 | 6.1440e+03 | 2.0208e+03 | pass |
| gemm1_accumulator__f16+gemm1_operands__f16+gemm1_output__f16+gemm2_accumulator__bf16 | cumulative | 74d7ff04 | 57 | 0.996463 | 0.887708 | 6.1440e+03 | 2.7275e+03 | pass |
| gemm1_accumulator__f16+gemm1_operands__f16+gemm1_output__f16+gemm2_accumulator__bf16 | cumulative | 4822167c | 56 | 0.995852 | 0.859649 | 6.1440e+03 | 2.5700e+02 | pass |
| gemm1_accumulator__f16+gemm1_operands__f16+gemm1_output__f16+gemm2_accumulator__bf16 | cumulative | 81955b1e | 55 | 0.996936 | 0.891378 | 8.1920e+03 | 1.4902e+02 | pass |
| gemm1_accumulator__f16+gemm1_operands__f16+gemm1_output__f16+gemm2_accumulator__bf16 | cumulative | 76010cb4 | 54 | 0.997285 | 0.910412 | 6.1440e+03 | 1.0558e+03 | pass |
| gemm1_accumulator__f16+gemm1_operands__f16+gemm1_output__f16+gemm2_accumulator__bf16 | cumulative | fc378037 | 53 | 0.996949 | 0.896450 | 8.1920e+03 | 2.4941e+02 | pass |
| gemm1_accumulator__f16+gemm1_operands__f16+gemm1_output__f16+gemm2_accumulator__bf16 | cumulative | f7d6ac7c | 52 | 0.998079 | 0.935013 | 4.0960e+03 | 1.2900e+02 | pass |
| gemm1_accumulator__f16+gemm1_operands__f16+gemm1_output__f16+gemm2_accumulator__bf16+gemm2_operands__bf16 | cumulative | b8f4f012 | 7 | 0.998625 | 0.956473 | 2.0480e+03 | 2.1855e+01 | pass |
| gemm1_accumulator__f16+gemm1_operands__f16+gemm1_output__f16+gemm2_accumulator__bf16+gemm2_operands__bf16 | cumulative | e05c6c03 | 1 | 0.994838 | 0.846959 | 2.0480e+03 | 4.7544e+00 | pass |
| gemm1_accumulator__f16+gemm1_operands__f16+gemm1_output__f16+gemm2_accumulator__bf16+gemm2_operands__bf16 | cumulative | 6230e838 | 32 | 0.998003 | 0.936794 | 4.0960e+03 | 1.2042e+03 | pass |
| gemm1_accumulator__f16+gemm1_operands__f16+gemm1_output__f16+gemm2_accumulator__bf16+gemm2_operands__bf16 | cumulative | 8f1ff9f1 | 80 | 0.996985 | 0.907844 | 2.0480e+03 | 9.8959e+01 | pass |
| gemm1_accumulator__f16+gemm1_operands__f16+gemm1_output__f16+gemm2_accumulator__bf16+gemm2_operands__bf16 | cumulative | 1a4c6ba1 | 901 | 0.996835 | 0.902303 | 4.0960e+03 | 5.5477e+03 | pass |
| gemm1_accumulator__f16+gemm1_operands__f16+gemm1_output__f16+gemm2_accumulator__bf16+gemm2_operands__bf16 | cumulative | a7c2bcfd | 16 | 0.996966 | 0.904018 | 2.0480e+03 | 2.9488e+02 | pass |
| gemm1_accumulator__f16+gemm1_operands__f16+gemm1_output__f16+gemm2_accumulator__bf16+gemm2_operands__bf16 | cumulative | 2e69caee | 15 | 0.998819 | 0.958863 | 2.0480e+03 | 2.5626e+01 | pass |
| gemm1_accumulator__f16+gemm1_operands__f16+gemm1_output__f16+gemm2_accumulator__bf16+gemm2_operands__bf16 | cumulative | 8cba5890 | 14 | 0.997658 | 0.933105 | 2.0480e+03 | 1.1242e+02 | pass |
| gemm1_accumulator__f16+gemm1_operands__f16+gemm1_output__f16+gemm2_accumulator__bf16+gemm2_operands__bf16 | cumulative | 5e8dc11c | 14107 | 0.997120 | 0.911469 | 4.0960e+03 | 8.2500e+09 | pass |
| gemm1_accumulator__f16+gemm1_operands__f16+gemm1_output__f16+gemm2_accumulator__bf16+gemm2_operands__bf16 | cumulative | 58a34f27 | 11948 | 0.997807 | 0.932369 | 4.0960e+03 | 3.9468e+05 | pass |
| gemm1_accumulator__f16+gemm1_operands__f16+gemm1_output__f16+gemm2_accumulator__bf16+gemm2_operands__bf16 | cumulative | 5eadab1e | 62 | 0.997698 | 0.927791 | 4.0960e+03 | 1.7436e+04 | pass |
| gemm1_accumulator__f16+gemm1_operands__f16+gemm1_output__f16+gemm2_accumulator__bf16+gemm2_operands__bf16 | cumulative | eedc63b2 | 59 | 0.998061 | 0.939510 | 4.0960e+03 | 1.4603e+02 | pass |
| gemm1_accumulator__f16+gemm1_operands__f16+gemm1_output__f16+gemm2_accumulator__bf16+gemm2_operands__bf16 | cumulative | e626d3e6 | 58 | 0.997087 | 0.909163 | 4.0960e+03 | 6.0202e+02 | pass |
| gemm1_accumulator__f16+gemm1_operands__f16+gemm1_output__f16+gemm2_accumulator__bf16+gemm2_operands__bf16 | cumulative | 74d7ff04 | 57 | 0.997467 | 0.924462 | 2.0480e+03 | 8.7905e+01 | pass |
| gemm1_accumulator__f16+gemm1_operands__f16+gemm1_output__f16+gemm2_accumulator__bf16+gemm2_operands__bf16 | cumulative | 4822167c | 56 | 0.996976 | 0.905782 | 4.0960e+03 | 8.8312e+02 | pass |
| gemm1_accumulator__f16+gemm1_operands__f16+gemm1_output__f16+gemm2_accumulator__bf16+gemm2_operands__bf16 | cumulative | 81955b1e | 55 | 0.997527 | 0.926887 | 4.0960e+03 | 1.1646e+02 | pass |
| gemm1_accumulator__f16+gemm1_operands__f16+gemm1_output__f16+gemm2_accumulator__bf16+gemm2_operands__bf16 | cumulative | 76010cb4 | 54 | 0.998042 | 0.940189 | 2.0480e+03 | 5.5567e+02 | pass |
| gemm1_accumulator__f16+gemm1_operands__f16+gemm1_output__f16+gemm2_accumulator__bf16+gemm2_operands__bf16 | cumulative | fc378037 | 53 | 0.997670 | 0.929764 | 4.0960e+03 | 4.6680e+02 | pass |
| gemm1_accumulator__f16+gemm1_operands__f16+gemm1_output__f16+gemm2_accumulator__bf16+gemm2_operands__bf16 | cumulative | f7d6ac7c | 52 | 0.998516 | 0.956285 | 2.0480e+03 | 2.7712e+02 | pass |
| gemm1_accumulator__f16+gemm1_operands__f16+gemm1_output__f16+gemm2_accumulator__bf16+gemm2_operands__bf16+hidden_dequant__f16 | cumulative | b8f4f012 | 7 | 0.998625 | 0.955796 | 2.0480e+03 | 3.9000e+01 | pass |
| gemm1_accumulator__f16+gemm1_operands__f16+gemm1_output__f16+gemm2_accumulator__bf16+gemm2_operands__bf16+hidden_dequant__f16 | cumulative | e05c6c03 | 1 | 0.997349 | 0.851144 | 2.0480e+03 | 1.8000e+00 | pass |
| gemm1_accumulator__f16+gemm1_operands__f16+gemm1_output__f16+gemm2_accumulator__bf16+gemm2_operands__bf16+hidden_dequant__f16 | cumulative | 6230e838 | 32 | 0.997968 | 0.936484 | 4.0960e+03 | 5.8460e+02 | pass |
| gemm1_accumulator__f16+gemm1_operands__f16+gemm1_output__f16+gemm2_accumulator__bf16+gemm2_operands__bf16+hidden_dequant__f16 | cumulative | 8f1ff9f1 | 80 | 0.997049 | 0.907813 | 4.0960e+03 | 3.7336e+02 | pass |
| gemm1_accumulator__f16+gemm1_operands__f16+gemm1_output__f16+gemm2_accumulator__bf16+gemm2_operands__bf16+hidden_dequant__f16 | cumulative | 1a4c6ba1 | 901 | 0.996800 | 0.902809 | 4.0960e+03 | 6.1695e+04 | pass |
| gemm1_accumulator__f16+gemm1_operands__f16+gemm1_output__f16+gemm2_accumulator__bf16+gemm2_operands__bf16+hidden_dequant__f16 | cumulative | a7c2bcfd | 16 | 0.996669 | 0.904925 | 2.0480e+03 | 1.1009e+02 | pass |
| gemm1_accumulator__f16+gemm1_operands__f16+gemm1_output__f16+gemm2_accumulator__bf16+gemm2_operands__bf16+hidden_dequant__f16 | cumulative | 2e69caee | 15 | 0.998754 | 0.960612 | 2.0480e+03 | 7.3209e+02 | pass |
| gemm1_accumulator__f16+gemm1_operands__f16+gemm1_output__f16+gemm2_accumulator__bf16+gemm2_operands__bf16+hidden_dequant__f16 | cumulative | 8cba5890 | 14 | 0.997808 | 0.933664 | 2.0480e+03 | 6.0257e+01 | pass |
| gemm1_accumulator__f16+gemm1_operands__f16+gemm1_output__f16+gemm2_accumulator__bf16+gemm2_operands__bf16+hidden_dequant__f16 | cumulative | 5e8dc11c | 14107 | 0.997123 | 0.911571 | 4.0960e+03 | 3.9000e+10 | pass |
| gemm1_accumulator__f16+gemm1_operands__f16+gemm1_output__f16+gemm2_accumulator__bf16+gemm2_operands__bf16+hidden_dequant__f16 | cumulative | 58a34f27 | 11948 | 0.997802 | 0.932358 | 8.1920e+03 | 7.3679e+03 | pass |
| gemm1_accumulator__f16+gemm1_operands__f16+gemm1_output__f16+gemm2_accumulator__bf16+gemm2_operands__bf16+hidden_dequant__f16 | cumulative | 5eadab1e | 62 | 0.997574 | 0.928225 | 4.0960e+03 | 1.4700e+02 | pass |
| gemm1_accumulator__f16+gemm1_operands__f16+gemm1_output__f16+gemm2_accumulator__bf16+gemm2_operands__bf16+hidden_dequant__f16 | cumulative | eedc63b2 | 59 | 0.997955 | 0.939231 | 4.0960e+03 | 2.7729e+02 | pass |
| gemm1_accumulator__f16+gemm1_operands__f16+gemm1_output__f16+gemm2_accumulator__bf16+gemm2_operands__bf16+hidden_dequant__f16 | cumulative | e626d3e6 | 58 | 0.997001 | 0.909259 | 2.0480e+03 | 2.7393e+04 | pass |
| gemm1_accumulator__f16+gemm1_operands__f16+gemm1_output__f16+gemm2_accumulator__bf16+gemm2_operands__bf16+hidden_dequant__f16 | cumulative | 74d7ff04 | 57 | 0.997533 | 0.924100 | 2.0480e+03 | 1.4875e+02 | pass |
| gemm1_accumulator__f16+gemm1_operands__f16+gemm1_output__f16+gemm2_accumulator__bf16+gemm2_operands__bf16+hidden_dequant__f16 | cumulative | 4822167c | 56 | 0.997008 | 0.907351 | 4.0960e+03 | 9.2810e+03 | pass |
| gemm1_accumulator__f16+gemm1_operands__f16+gemm1_output__f16+gemm2_accumulator__bf16+gemm2_operands__bf16+hidden_dequant__f16 | cumulative | 81955b1e | 55 | 0.997633 | 0.926476 | 2.0480e+03 | 1.7984e+02 | pass |
| gemm1_accumulator__f16+gemm1_operands__f16+gemm1_output__f16+gemm2_accumulator__bf16+gemm2_operands__bf16+hidden_dequant__f16 | cumulative | 76010cb4 | 54 | 0.997951 | 0.939430 | 2.0480e+03 | 1.2259e+02 | pass |
| gemm1_accumulator__f16+gemm1_operands__f16+gemm1_output__f16+gemm2_accumulator__bf16+gemm2_operands__bf16+hidden_dequant__f16 | cumulative | fc378037 | 53 | 0.997713 | 0.931319 | 4.0960e+03 | 1.7907e+02 | pass |
| gemm1_accumulator__f16+gemm1_operands__f16+gemm1_output__f16+gemm2_accumulator__bf16+gemm2_operands__bf16+hidden_dequant__f16 | cumulative | f7d6ac7c | 52 | 0.998613 | 0.955835 | 2.0480e+03 | 3.6633e+01 | pass |
| gemm1_accumulator__f16+gemm1_operands__f16+gemm1_output__f16+gemm2_accumulator__bf16+gemm2_operands__bf16+hidden_dequant__f16+out_accumulator__bf16 | cumulative | b8f4f012 | 7 | 0.998346 | 0.949279 | 4.0960e+03 | 4.7223e+01 | pass |
| gemm1_accumulator__f16+gemm1_operands__f16+gemm1_output__f16+gemm2_accumulator__bf16+gemm2_operands__bf16+hidden_dequant__f16+out_accumulator__bf16 | cumulative | e05c6c03 | 1 | 0.994559 | 0.833984 | 2.0480e+03 | 1.8524e+00 | pass |
| gemm1_accumulator__f16+gemm1_operands__f16+gemm1_output__f16+gemm2_accumulator__bf16+gemm2_operands__bf16+hidden_dequant__f16+out_accumulator__bf16 | cumulative | 6230e838 | 32 | 0.997729 | 0.931170 | 4.0960e+03 | 8.9134e+02 | pass |
| gemm1_accumulator__f16+gemm1_operands__f16+gemm1_output__f16+gemm2_accumulator__bf16+gemm2_operands__bf16+hidden_dequant__f16+out_accumulator__bf16 | cumulative | 8f1ff9f1 | 80 | 0.996655 | 0.900725 | 4.0960e+03 | 2.0490e+03 | pass |
| gemm1_accumulator__f16+gemm1_operands__f16+gemm1_output__f16+gemm2_accumulator__bf16+gemm2_operands__bf16+hidden_dequant__f16+out_accumulator__bf16 | cumulative | 1a4c6ba1 | 901 | 0.996603 | 0.895256 | 4.0960e+03 | 9.0870e+03 | pass |
| gemm1_accumulator__f16+gemm1_operands__f16+gemm1_output__f16+gemm2_accumulator__bf16+gemm2_operands__bf16+hidden_dequant__f16+out_accumulator__bf16 | cumulative | a7c2bcfd | 16 | 0.996748 | 0.897592 | 2.0480e+03 | 1.0023e+02 | pass |
| gemm1_accumulator__f16+gemm1_operands__f16+gemm1_output__f16+gemm2_accumulator__bf16+gemm2_operands__bf16+hidden_dequant__f16+out_accumulator__bf16 | cumulative | 2e69caee | 15 | 0.998605 | 0.956687 | 4.0960e+03 | 2.3923e+01 | pass |
| gemm1_accumulator__f16+gemm1_operands__f16+gemm1_output__f16+gemm2_accumulator__bf16+gemm2_operands__bf16+hidden_dequant__f16+out_accumulator__bf16 | cumulative | 8cba5890 | 14 | 0.997539 | 0.929379 | 2.0480e+03 | 3.5740e+02 | pass |
| gemm1_accumulator__f16+gemm1_operands__f16+gemm1_output__f16+gemm2_accumulator__bf16+gemm2_operands__bf16+hidden_dequant__f16+out_accumulator__bf16 | cumulative | 5e8dc11c | 14107 | 0.996999 | 0.906970 | 8.1920e+03 | 1.2390e+05 | pass |
| gemm1_accumulator__f16+gemm1_operands__f16+gemm1_output__f16+gemm2_accumulator__bf16+gemm2_operands__bf16+hidden_dequant__f16+out_accumulator__bf16 | cumulative | 58a34f27 | 11948 | 0.997704 | 0.928604 | 6.1440e+03 | 8.2156e+04 | pass |
| gemm1_accumulator__f16+gemm1_operands__f16+gemm1_output__f16+gemm2_accumulator__bf16+gemm2_operands__bf16+hidden_dequant__f16+out_accumulator__bf16 | cumulative | 5eadab1e | 62 | 0.997545 | 0.923770 | 2.0480e+03 | 1.0374e+03 | pass |
| gemm1_accumulator__f16+gemm1_operands__f16+gemm1_output__f16+gemm2_accumulator__bf16+gemm2_operands__bf16+hidden_dequant__f16+out_accumulator__bf16 | cumulative | eedc63b2 | 59 | 0.998071 | 0.937481 | 4.0960e+03 | 4.8241e+02 | pass |
| gemm1_accumulator__f16+gemm1_operands__f16+gemm1_output__f16+gemm2_accumulator__bf16+gemm2_operands__bf16+hidden_dequant__f16+out_accumulator__bf16 | cumulative | e626d3e6 | 58 | 0.996900 | 0.903157 | 4.0960e+03 | 1.9454e+02 | pass |
| gemm1_accumulator__f16+gemm1_operands__f16+gemm1_output__f16+gemm2_accumulator__bf16+gemm2_operands__bf16+hidden_dequant__f16+out_accumulator__bf16 | cumulative | 74d7ff04 | 57 | 0.997320 | 0.919109 | 4.0960e+03 | 4.5951e+02 | pass |
| gemm1_accumulator__f16+gemm1_operands__f16+gemm1_output__f16+gemm2_accumulator__bf16+gemm2_operands__bf16+hidden_dequant__f16+out_accumulator__bf16 | cumulative | 4822167c | 56 | 0.996682 | 0.899197 | 2.0480e+03 | 4.8707e+02 | pass |
| gemm1_accumulator__f16+gemm1_operands__f16+gemm1_output__f16+gemm2_accumulator__bf16+gemm2_operands__bf16+hidden_dequant__f16+out_accumulator__bf16 | cumulative | 81955b1e | 55 | 0.997466 | 0.922783 | 4.0960e+03 | 4.3117e+02 | pass |
| gemm1_accumulator__f16+gemm1_operands__f16+gemm1_output__f16+gemm2_accumulator__bf16+gemm2_operands__bf16+hidden_dequant__f16+out_accumulator__bf16 | cumulative | 76010cb4 | 54 | 0.997851 | 0.936642 | 4.0960e+03 | 2.9129e+02 | pass |
| gemm1_accumulator__f16+gemm1_operands__f16+gemm1_output__f16+gemm2_accumulator__bf16+gemm2_operands__bf16+hidden_dequant__f16+out_accumulator__bf16 | cumulative | fc378037 | 53 | 0.997557 | 0.925413 | 4.0960e+03 | 4.7137e+03 | pass |
| gemm1_accumulator__f16+gemm1_operands__f16+gemm1_output__f16+gemm2_accumulator__bf16+gemm2_operands__bf16+hidden_dequant__f16+out_accumulator__bf16 | cumulative | f7d6ac7c | 52 | 0.998573 | 0.954281 | 2.0480e+03 | 1.8408e+03 | pass |
| gemm1_accumulator__f16+gemm1_operands__f16+gemm1_output__f16+gemm2_accumulator__bf16+gemm2_operands__bf16+hidden_dequant__f16+out_accumulator__bf16+swiglu_input__f16 | cumulative | b8f4f012 | 7 | 0.998346 | 0.947943 | 2.0480e+03 | 6.4576e+01 | pass |
| gemm1_accumulator__f16+gemm1_operands__f16+gemm1_output__f16+gemm2_accumulator__bf16+gemm2_operands__bf16+hidden_dequant__f16+out_accumulator__bf16+swiglu_input__f16 | cumulative | e05c6c03 | 1 | 0.995117 | 0.836356 | 2.0480e+03 | 2.3000e+01 | pass |
| gemm1_accumulator__f16+gemm1_operands__f16+gemm1_output__f16+gemm2_accumulator__bf16+gemm2_operands__bf16+hidden_dequant__f16+out_accumulator__bf16+swiglu_input__f16 | cumulative | 6230e838 | 32 | 0.997877 | 0.930869 | 4.0960e+03 | 9.6193e+01 | pass |
| gemm1_accumulator__f16+gemm1_operands__f16+gemm1_output__f16+gemm2_accumulator__bf16+gemm2_operands__bf16+hidden_dequant__f16+out_accumulator__bf16+swiglu_input__f16 | cumulative | 8f1ff9f1 | 80 | 0.996788 | 0.900445 | 4.0960e+03 | 2.6625e+04 | pass |
| gemm1_accumulator__f16+gemm1_operands__f16+gemm1_output__f16+gemm2_accumulator__bf16+gemm2_operands__bf16+hidden_dequant__f16+out_accumulator__bf16+swiglu_input__f16 | cumulative | 1a4c6ba1 | 901 | 0.996616 | 0.895254 | 4.0960e+03 | 4.0970e+03 | pass |
| gemm1_accumulator__f16+gemm1_operands__f16+gemm1_output__f16+gemm2_accumulator__bf16+gemm2_operands__bf16+hidden_dequant__f16+out_accumulator__bf16+swiglu_input__f16 | cumulative | a7c2bcfd | 16 | 0.996704 | 0.897661 | 2.0480e+03 | 1.4049e+02 | pass |
| gemm1_accumulator__f16+gemm1_operands__f16+gemm1_output__f16+gemm2_accumulator__bf16+gemm2_operands__bf16+hidden_dequant__f16+out_accumulator__bf16+swiglu_input__f16 | cumulative | 2e69caee | 15 | 0.998586 | 0.954697 | 4.0960e+03 | 3.2224e+01 | pass |
| gemm1_accumulator__f16+gemm1_operands__f16+gemm1_output__f16+gemm2_accumulator__bf16+gemm2_operands__bf16+hidden_dequant__f16+out_accumulator__bf16+swiglu_input__f16 | cumulative | 8cba5890 | 14 | 0.997758 | 0.927824 | 2.0480e+03 | 5.6464e+01 | pass |
| gemm1_accumulator__f16+gemm1_operands__f16+gemm1_output__f16+gemm2_accumulator__bf16+gemm2_operands__bf16+hidden_dequant__f16+out_accumulator__bf16+swiglu_input__f16 | cumulative | 5e8dc11c | 14107 | 0.997003 | 0.906933 | 8.1920e+03 | 1.0251e+05 | pass |
| gemm1_accumulator__f16+gemm1_operands__f16+gemm1_output__f16+gemm2_accumulator__bf16+gemm2_operands__bf16+hidden_dequant__f16+out_accumulator__bf16+swiglu_input__f16 | cumulative | 58a34f27 | 11948 | 0.997700 | 0.928701 | 8.1920e+03 | 3.2769e+04 | pass |
| gemm1_accumulator__f16+gemm1_operands__f16+gemm1_output__f16+gemm2_accumulator__bf16+gemm2_operands__bf16+hidden_dequant__f16+out_accumulator__bf16+swiglu_input__f16 | cumulative | 5eadab1e | 62 | 0.997536 | 0.923212 | 4.0960e+03 | 1.7580e+03 | pass |
| gemm1_accumulator__f16+gemm1_operands__f16+gemm1_output__f16+gemm2_accumulator__bf16+gemm2_operands__bf16+hidden_dequant__f16+out_accumulator__bf16+swiglu_input__f16 | cumulative | eedc63b2 | 59 | 0.997978 | 0.936897 | 4.0960e+03 | 1.7223e+02 | pass |
| gemm1_accumulator__f16+gemm1_operands__f16+gemm1_output__f16+gemm2_accumulator__bf16+gemm2_operands__bf16+hidden_dequant__f16+out_accumulator__bf16+swiglu_input__f16 | cumulative | e626d3e6 | 58 | 0.996753 | 0.902476 | 4.0960e+03 | 2.2791e+02 | pass |
| gemm1_accumulator__f16+gemm1_operands__f16+gemm1_output__f16+gemm2_accumulator__bf16+gemm2_operands__bf16+hidden_dequant__f16+out_accumulator__bf16+swiglu_input__f16 | cumulative | 74d7ff04 | 57 | 0.997401 | 0.918314 | 4.0960e+03 | 8.7351e+03 | pass |
| gemm1_accumulator__f16+gemm1_operands__f16+gemm1_output__f16+gemm2_accumulator__bf16+gemm2_operands__bf16+hidden_dequant__f16+out_accumulator__bf16+swiglu_input__f16 | cumulative | 4822167c | 56 | 0.996749 | 0.901108 | 4.0960e+03 | 1.5625e+02 | pass |
| gemm1_accumulator__f16+gemm1_operands__f16+gemm1_output__f16+gemm2_accumulator__bf16+gemm2_operands__bf16+hidden_dequant__f16+out_accumulator__bf16+swiglu_input__f16 | cumulative | 81955b1e | 55 | 0.997451 | 0.922481 | 2.0480e+03 | 1.9007e+03 | pass |
| gemm1_accumulator__f16+gemm1_operands__f16+gemm1_output__f16+gemm2_accumulator__bf16+gemm2_operands__bf16+hidden_dequant__f16+out_accumulator__bf16+swiglu_input__f16 | cumulative | 76010cb4 | 54 | 0.997951 | 0.936684 | 4.0960e+03 | 1.1606e+03 | pass |
| gemm1_accumulator__f16+gemm1_operands__f16+gemm1_output__f16+gemm2_accumulator__bf16+gemm2_operands__bf16+hidden_dequant__f16+out_accumulator__bf16+swiglu_input__f16 | cumulative | fc378037 | 53 | 0.997623 | 0.925347 | 4.0960e+03 | 2.3581e+02 | pass |
| gemm1_accumulator__f16+gemm1_operands__f16+gemm1_output__f16+gemm2_accumulator__bf16+gemm2_operands__bf16+hidden_dequant__f16+out_accumulator__bf16+swiglu_input__f16 | cumulative | f7d6ac7c | 52 | 0.998481 | 0.954155 | 2.0480e+03 | 1.2700e+02 | pass |
| gemm1_accumulator__f16+gemm1_operands__f16+gemm1_output__f16+gemm2_accumulator__bf16+gemm2_operands__bf16+hidden_dequant__f16+out_accumulator__bf16+swiglu_input__f16+swiglu_output__bf16 | cumulative | b8f4f012 | 7 | 0.998406 | 0.947246 | 2.0480e+03 | 1.2241e+01 | pass |
| gemm1_accumulator__f16+gemm1_operands__f16+gemm1_output__f16+gemm2_accumulator__bf16+gemm2_operands__bf16+hidden_dequant__f16+out_accumulator__bf16+swiglu_input__f16+swiglu_output__bf16 | cumulative | e05c6c03 | 1 | 0.995815 | 0.823940 | 2.0480e+03 | 3.2431e+00 | pass |
| gemm1_accumulator__f16+gemm1_operands__f16+gemm1_output__f16+gemm2_accumulator__bf16+gemm2_operands__bf16+hidden_dequant__f16+out_accumulator__bf16+swiglu_input__f16+swiglu_output__bf16 | cumulative | 6230e838 | 32 | 0.997947 | 0.931671 | 4.0960e+03 | 6.7243e+04 | pass |
| gemm1_accumulator__f16+gemm1_operands__f16+gemm1_output__f16+gemm2_accumulator__bf16+gemm2_operands__bf16+hidden_dequant__f16+out_accumulator__bf16+swiglu_input__f16+swiglu_output__bf16 | cumulative | 8f1ff9f1 | 80 | 0.996858 | 0.900718 | 4.0960e+03 | 2.9914e+02 | pass |
| gemm1_accumulator__f16+gemm1_operands__f16+gemm1_output__f16+gemm2_accumulator__bf16+gemm2_operands__bf16+hidden_dequant__f16+out_accumulator__bf16+swiglu_input__f16+swiglu_output__bf16 | cumulative | 1a4c6ba1 | 901 | 0.996627 | 0.895273 | 4.0960e+03 | 1.8800e+10 | pass |
| gemm1_accumulator__f16+gemm1_operands__f16+gemm1_output__f16+gemm2_accumulator__bf16+gemm2_operands__bf16+hidden_dequant__f16+out_accumulator__bf16+swiglu_input__f16+swiglu_output__bf16 | cumulative | a7c2bcfd | 16 | 0.996634 | 0.900051 | 2.0480e+03 | 7.8748e+01 | pass |
| gemm1_accumulator__f16+gemm1_operands__f16+gemm1_output__f16+gemm2_accumulator__bf16+gemm2_operands__bf16+hidden_dequant__f16+out_accumulator__bf16+swiglu_input__f16+swiglu_output__bf16 | cumulative | 2e69caee | 15 | 0.998428 | 0.955125 | 4.0960e+03 | 4.7522e+01 | pass |
| gemm1_accumulator__f16+gemm1_operands__f16+gemm1_output__f16+gemm2_accumulator__bf16+gemm2_operands__bf16+hidden_dequant__f16+out_accumulator__bf16+swiglu_input__f16+swiglu_output__bf16 | cumulative | 8cba5890 | 14 | 0.997479 | 0.930156 | 2.0480e+03 | 5.0470e+01 | pass |
| gemm1_accumulator__f16+gemm1_operands__f16+gemm1_output__f16+gemm2_accumulator__bf16+gemm2_operands__bf16+hidden_dequant__f16+out_accumulator__bf16+swiglu_input__f16+swiglu_output__bf16 | cumulative | 5e8dc11c | 14107 | 0.996997 | 0.907011 | 8.1920e+03 | 1.9661e+05 | pass |
| gemm1_accumulator__f16+gemm1_operands__f16+gemm1_output__f16+gemm2_accumulator__bf16+gemm2_operands__bf16+hidden_dequant__f16+out_accumulator__bf16+swiglu_input__f16+swiglu_output__bf16 | cumulative | 58a34f27 | 11948 | 0.997701 | 0.928807 | 8.1920e+03 | 3.7889e+04 | pass |
| gemm1_accumulator__f16+gemm1_operands__f16+gemm1_output__f16+gemm2_accumulator__bf16+gemm2_operands__bf16+hidden_dequant__f16+out_accumulator__bf16+swiglu_input__f16+swiglu_output__bf16 | cumulative | 5eadab1e | 62 | 0.997473 | 0.924301 | 4.0960e+03 | 2.5930e+02 | pass |
| gemm1_accumulator__f16+gemm1_operands__f16+gemm1_output__f16+gemm2_accumulator__bf16+gemm2_operands__bf16+hidden_dequant__f16+out_accumulator__bf16+swiglu_input__f16+swiglu_output__bf16 | cumulative | eedc63b2 | 59 | 0.997962 | 0.936486 | 4.0960e+03 | 6.1365e+02 | pass |
| gemm1_accumulator__f16+gemm1_operands__f16+gemm1_output__f16+gemm2_accumulator__bf16+gemm2_operands__bf16+hidden_dequant__f16+out_accumulator__bf16+swiglu_input__f16+swiglu_output__bf16 | cumulative | e626d3e6 | 58 | 0.996907 | 0.902878 | 4.0960e+03 | 5.6255e+02 | pass |
| gemm1_accumulator__f16+gemm1_operands__f16+gemm1_output__f16+gemm2_accumulator__bf16+gemm2_operands__bf16+hidden_dequant__f16+out_accumulator__bf16+swiglu_input__f16+swiglu_output__bf16 | cumulative | 74d7ff04 | 57 | 0.997472 | 0.918774 | 4.0960e+03 | 3.3326e+02 | pass |
| gemm1_accumulator__f16+gemm1_operands__f16+gemm1_output__f16+gemm2_accumulator__bf16+gemm2_operands__bf16+hidden_dequant__f16+out_accumulator__bf16+swiglu_input__f16+swiglu_output__bf16 | cumulative | 4822167c | 56 | 0.996719 | 0.901003 | 2.0480e+03 | 7.3087e+02 | pass |
| gemm1_accumulator__f16+gemm1_operands__f16+gemm1_output__f16+gemm2_accumulator__bf16+gemm2_operands__bf16+hidden_dequant__f16+out_accumulator__bf16+swiglu_input__f16+swiglu_output__bf16 | cumulative | 81955b1e | 55 | 0.997565 | 0.922727 | 4.0960e+03 | 1.1910e+03 | pass |
| gemm1_accumulator__f16+gemm1_operands__f16+gemm1_output__f16+gemm2_accumulator__bf16+gemm2_operands__bf16+hidden_dequant__f16+out_accumulator__bf16+swiglu_input__f16+swiglu_output__bf16 | cumulative | 76010cb4 | 54 | 0.997995 | 0.936384 | 2.0480e+03 | 4.0019e+03 | pass |
| gemm1_accumulator__f16+gemm1_operands__f16+gemm1_output__f16+gemm2_accumulator__bf16+gemm2_operands__bf16+hidden_dequant__f16+out_accumulator__bf16+swiglu_input__f16+swiglu_output__bf16 | cumulative | fc378037 | 53 | 0.997660 | 0.925518 | 2.0480e+03 | 1.3991e+02 | pass |
| gemm1_accumulator__f16+gemm1_operands__f16+gemm1_output__f16+gemm2_accumulator__bf16+gemm2_operands__bf16+hidden_dequant__f16+out_accumulator__bf16+swiglu_input__f16+swiglu_output__bf16 | cumulative | f7d6ac7c | 52 | 0.998591 | 0.954126 | 2.0480e+03 | 3.4714e+02 | pass |