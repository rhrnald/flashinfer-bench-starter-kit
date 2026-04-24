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
| run_stage | stage1_bf16_f16 |
| panel_size | 8 |
| panel_indices | 0,1,2,4,8,9,14,18 |
| n_candidates | 19 |
| n_survivors | 16 |
| contest_safe_survivor_count | 16 |
| contest_safe_survivors | gemm1_output__f16, swiglu_input__f16, gemm1_operands__f16, hidden_dequant__f16, out_accumulator__bf16, swiglu_output__bf16, gemm1_accumulator__f16, hidden_dequant__bf16, gemm1_output__bf16, swiglu_input__bf16, gemm2_operands__bf16, gemm1_operands__bf16, gemm2_accumulator__bf16, gemm1_accumulator__bf16, swiglu_output__f16, gemm2_operands__f16 |
| strict_safe_survivor_count | 7 |
| strict_safe_survivors | hidden_dequant__f16, gemm1_operands__f16, gemm1_output__f16, swiglu_input__f16, swiglu_output__f16, gemm2_operands__f16, out_accumulator__bf16 |
| contest_only_survivor_count | 9 |
| contest_only_survivors | hidden_dequant__bf16, gemm1_operands__bf16, gemm1_accumulator__bf16, gemm1_accumulator__f16, gemm1_output__bf16, swiglu_input__bf16, swiglu_output__bf16, gemm2_operands__bf16, gemm2_accumulator__bf16 |

## Survivor Summary

| category | count | candidates |
|---|---:|---|
| contest_safe_single_stage | 16 | gemm1_output__f16, swiglu_input__f16, gemm1_operands__f16, hidden_dequant__f16, out_accumulator__bf16, swiglu_output__bf16, gemm1_accumulator__f16, hidden_dequant__bf16, gemm1_output__bf16, swiglu_input__bf16, gemm2_operands__bf16, gemm1_operands__bf16, gemm2_accumulator__bf16, gemm1_accumulator__bf16, swiglu_output__f16, gemm2_operands__f16 |
| strict_safe_single_stage | 7 | hidden_dequant__f16, gemm1_operands__f16, gemm1_output__f16, swiglu_input__f16, swiglu_output__f16, gemm2_operands__f16, out_accumulator__bf16 |
| contest_only_single_stage | 9 | hidden_dequant__bf16, gemm1_operands__bf16, gemm1_accumulator__bf16, gemm1_accumulator__f16, gemm1_output__bf16, swiglu_input__bf16, swiglu_output__bf16, gemm2_operands__bf16, gemm2_accumulator__bf16 |

## Stage Summary

| stage | best_safe_mode | scale | worst_matched_contest | worst_matched_strict | worst_rel | status |
|---|---|---|---:|---:|---:|---|
| gemm1_accumulator | f16 | none | 0.996233 | 0.893834 | 1.6000e+10 | safe |
| gemm1_operands | f16 | none | 0.999442 | 0.975167 | 3.3500e+09 | safe |
| gemm1_output | f16 | none | 0.999599 | 0.982143 | 8.3750e+08 | safe |
| gemm2_accumulator | bf16 | none | 0.991908 | 0.747210 | 1.2438e+09 | safe |
| gemm2_operands | f16 | none | 0.956146 | 0.946328 | 1.4422e+00 | safe |
| hidden_dequant | f16 | none | 0.999302 | 0.983538 | 1.8750e+09 | safe |
| out_accumulator | bf16 | none | 0.998465 | 0.935965 | 2.2800e+10 | safe |
| swiglu_input | f16 | none | 0.999599 | 0.982143 | 8.3750e+08 | safe |
| swiglu_output | f16 | none | 0.956238 | 0.949300 | 2.6140e+00 | safe |

## Cumulative Safe Frontier

| order | candidate | worst_matched_contest | worst_matched_strict | worst_rel | status |
|---|---|---:|---:|---:|---|
| 1 | gemm1_accumulator__f16 | 0.995675 | 0.905413 | 4.0729e+04 | safe |
| 2 | gemm1_accumulator__f16+gemm1_operands__f16 | 0.999302 | 0.968471 | 8.0500e+09 | safe |
| 3 | gemm1_accumulator__f16+gemm1_operands__f16+gemm1_output__f16 | 0.999300 | 0.965402 | 1.0100e+10 | safe |
| 4 | gemm1_accumulator__f16+gemm1_operands__f16+gemm1_output__f16+gemm2_accumulator__bf16 | 0.991071 | 0.763811 | 1.4200e+10 | safe |
| 5 | gemm1_accumulator__f16+gemm1_operands__f16+gemm1_output__f16+gemm2_accumulator__bf16+gemm2_operands__f16 | 0.821189 | 0.821189 | nan | unsafe |

## BF16/F16 Margin

| stage | preferred | bf16_contest | f16_contest | bf16_strict | f16_strict | |
|---|---|---:|---:|---:|---:|---|
| hidden_dequant | f16 | 0.995396 | 0.999302 | 0.847377 | 0.983538 | |
| gemm1_operands | f16 | 0.994001 | 0.999442 | 0.788086 | 0.975167 | |
| gemm1_accumulator | f16 | 0.977260 | 0.996233 | 0.422294 | 0.893834 | |
| gemm1_output | f16 | 0.994420 | 0.999599 | 0.847935 | 0.982143 | |
| swiglu_input | f16 | 0.994420 | 0.999599 | 0.847935 | 0.982143 | |
| swiglu_output | bf16 | 0.996512 | 0.956238 | 0.892997 | 0.949300 | |
| gemm2_operands | bf16 | 0.994141 | 0.956146 | 0.842773 | 0.946328 | |
| gemm2_accumulator | bf16 | 0.991908 | 0.626395 | 0.747210 | 0.625837 | |
| out_accumulator | bf16 | 0.998465 | 0.580218 | 0.935965 | 0.573382 | |

## Pairwise Summary

| pair | worst_matched_contest | worst_matched_strict | worst_rel | status |
|---|---:|---:|---:|---|
| swiglu_output__f16+gemm2_accumulator__bf16 | 0.936244 | 0.750558 | 3.6972e+01 | safe |
| gemm1_accumulator__f16+swiglu_output__f16 | 0.936907 | 0.887067 | 2.3990e+01 | safe |
| swiglu_output__f16+out_accumulator__bf16 | 0.937385 | 0.919455 | 1.3738e+01 | safe |
| gemm1_operands__f16+swiglu_output__f16 | 0.937385 | 0.930217 | 8.4586e+00 | safe |
| hidden_dequant__f16+swiglu_output__f16 | 0.937409 | 0.932518 | 5.0000e+00 | safe |
| gemm1_output__f16+swiglu_output__f16 | 0.937414 | 0.932510 | 3.4812e+00 | safe |
| gemm1_accumulator__f16+gemm2_accumulator__bf16 | 0.989258 | 0.724191 | 5.3247e+04 | safe |
| gemm1_output__f16+gemm2_accumulator__bf16 | 0.990653 | 0.751116 | 8.7041e+04 | safe |
| gemm1_operands__f16+gemm2_accumulator__bf16 | 0.990932 | 0.750558 | 8.7041e+04 | safe |
| gemm2_accumulator__bf16+out_accumulator__bf16 | 0.991071 | 0.743304 | 1.0240e+05 | safe |
| hidden_dequant__f16+gemm2_accumulator__bf16 | 0.991629 | 0.751674 | 8.7041e+04 | safe |
| gemm1_accumulator__f16+out_accumulator__bf16 | 0.995536 | 0.870954 | 7.8164e+04 | safe |
| gemm1_accumulator__f16+gemm1_output__f16 | 0.996094 | 0.887974 | 7.7140e+04 | safe |
| hidden_dequant__f16+gemm1_accumulator__f16 | 0.996512 | 0.891602 | 8.4308e+04 | safe |
| gemm1_operands__f16+out_accumulator__bf16 | 0.997489 | 0.929129 | 3.2425e+04 | safe |
| hidden_dequant__f16+out_accumulator__bf16 | 0.997489 | 0.933315 | 3.2426e+04 | safe |
| gemm1_output__f16+out_accumulator__bf16 | 0.997768 | 0.931641 | 3.4132e+04 | safe |
| gemm1_operands__f16+gemm1_output__f16 | 0.998744 | 0.964007 | 1.0836e+04 | safe |
| hidden_dequant__f16+gemm1_output__f16 | 0.998884 | 0.970703 | 2.5428e+04 | safe |
| hidden_dequant__f16+gemm1_operands__f16 | 0.999163 | 0.970285 | 2.2356e+04 | safe |
| gemm1_operands__f16+gemm1_accumulator__f16 | 0.999163 | 0.970285 | 2.2356e+04 | safe |

## Stress Summary

| candidate | worst_workload | worst_seq_len | worst_matched_contest | worst_matched_strict | status |
|---|---|---:|---:|---:|---|
| gemm2_accumulator__bf16 | e05c6c03 | 1 | 0.991908 | 0.747210 | safe |

## Promotion Summary

| category | candidates |
|---|---|
| bf16_f16_survivors | gemm1_output__f16, swiglu_input__f16, gemm1_operands__f16, hidden_dequant__f16, out_accumulator__bf16, swiglu_output__bf16, gemm1_accumulator__f16, hidden_dequant__bf16, gemm1_output__bf16, swiglu_input__bf16, gemm2_operands__bf16, gemm1_operands__bf16, gemm2_accumulator__bf16, gemm1_accumulator__bf16, swiglu_output__f16, gemm2_operands__f16 |
| strict_survivors | hidden_dequant__f16, gemm1_operands__f16, gemm1_output__f16, swiglu_input__f16, swiglu_output__f16, gemm2_operands__f16, out_accumulator__bf16 |
| pairwise_shortlist | hidden_dequant__f16, gemm1_operands__f16, gemm1_accumulator__f16, gemm1_output__f16, swiglu_output__f16, gemm2_accumulator__bf16, out_accumulator__bf16 |

## Sampled Results

| candidate | phase | workload | seq_len | matched_contest | matched_strict | max_abs | max_rel | failure |
|---|---|---|---:|---:|---:|---:|---:|---|
| baseline__fp32 | single_stage | b8f4f012 | 7 | 1.000000 | 1.000000 | 0.0000e+00 | 0.0000e+00 | pass |
| hidden_dequant__bf16 | single_stage | b8f4f012 | 7 | 0.998705 | 0.959084 | 2.0480e+03 | 3.7772e+01 | pass |
| hidden_dequant__f16 | single_stage | b8f4f012 | 7 | 0.999960 | 0.995018 | 2.0480e+03 | 1.2807e+00 | pass |
| gemm1_operands__bf16 | single_stage | b8f4f012 | 7 | 0.997788 | 0.941805 | 2.3040e+03 | 1.3281e+01 | pass |
| gemm1_operands__f16 | single_stage | b8f4f012 | 7 | 0.999841 | 0.992506 | 2.0480e+03 | 6.8947e+00 | pass |
| gemm1_accumulator__bf16 | single_stage | b8f4f012 | 7 | 0.991829 | 0.832270 | 8.1920e+03 | 4.3016e+02 | pass |
| gemm1_accumulator__f16 | single_stage | b8f4f012 | 7 | 0.999023 | 0.968511 | 2.0480e+03 | 2.6228e+01 | pass |
| gemm1_output__bf16 | single_stage | b8f4f012 | 7 | 0.998645 | 0.958187 | 2.0480e+03 | 8.4689e+00 | pass |
| gemm1_output__f16 | single_stage | b8f4f012 | 7 | 0.999880 | 0.994938 | 2.0480e+03 | 2.1754e+00 | pass |
| swiglu_input__bf16 | single_stage | b8f4f012 | 7 | 0.998645 | 0.958187 | 2.0480e+03 | 8.4689e+00 | pass |
| swiglu_input__f16 | single_stage | b8f4f012 | 7 | 0.999880 | 0.994938 | 2.0480e+03 | 2.1754e+00 | pass |
| swiglu_output__bf16 | single_stage | b8f4f012 | 7 | 0.999143 | 0.968949 | 2.0480e+03 | 5.2333e+01 | pass |
| swiglu_output__f16 | single_stage | b8f4f012 | 7 | 0.999940 | 0.996154 | 2.0480e+03 | 2.6140e+00 | pass |
| gemm2_operands__bf16 | single_stage | b8f4f012 | 7 | 0.998585 | 0.957091 | 2.0480e+03 | 8.9386e+01 | pass |
| gemm2_operands__f16 | single_stage | b8f4f012 | 7 | 0.999860 | 0.994699 | 2.0480e+03 | 1.1617e+00 | pass |
| gemm2_accumulator__bf16 | single_stage | b8f4f012 | 7 | 0.997708 | 0.930505 | 4.0960e+03 | 1.6825e+01 | pass |
| gemm2_accumulator__f16 | single_stage | b8f4f012 | 7 | 0.914521 | 0.914521 | nan | nan | pass |
| out_accumulator__bf16 | single_stage | b8f4f012 | 7 | 0.999542 | 0.981704 | 3.0720e+03 | 1.3351e+01 | pass |
| out_accumulator__f16 | single_stage | b8f4f012 | 7 | 0.796297 | 0.795360 | inf | inf | catastrophic_outlier |
| baseline__fp32 | single_stage | e05c6c03 | 1 | 1.000000 | 1.000000 | 0.0000e+00 | 0.0000e+00 | pass |
| hidden_dequant__bf16 | single_stage | e05c6c03 | 1 | 0.995396 | 0.847377 | 2.0480e+03 | 9.7638e+00 | pass |
| hidden_dequant__f16 | single_stage | e05c6c03 | 1 | 0.999302 | 0.983538 | 1.0240e+03 | 1.5528e+00 | pass |
| gemm1_operands__bf16 | single_stage | e05c6c03 | 1 | 0.994001 | 0.788086 | 1.0240e+03 | 7.6056e+00 | pass |
| gemm1_operands__f16 | single_stage | e05c6c03 | 1 | 0.999442 | 0.975167 | 1.0240e+03 | 1.6935e+00 | pass |
| gemm1_accumulator__bf16 | single_stage | e05c6c03 | 1 | 0.977260 | 0.422294 | 4.4800e+03 | 1.8480e+01 | pass |
| gemm1_accumulator__f16 | single_stage | e05c6c03 | 1 | 0.996233 | 0.893834 | 2.0480e+03 | 4.4815e+00 | pass |
| gemm1_output__bf16 | single_stage | e05c6c03 | 1 | 0.994420 | 0.847935 | 2.0480e+03 | 4.2961e+00 | pass |
| gemm1_output__f16 | single_stage | e05c6c03 | 1 | 0.999721 | 0.982143 | 1.0240e+03 | 5.2885e-01 | pass |
| swiglu_input__bf16 | single_stage | e05c6c03 | 1 | 0.994420 | 0.847935 | 2.0480e+03 | 4.2961e+00 | pass |
| swiglu_input__f16 | single_stage | e05c6c03 | 1 | 0.999721 | 0.982143 | 1.0240e+03 | 5.2885e-01 | pass |
| swiglu_output__bf16 | single_stage | e05c6c03 | 1 | 0.996512 | 0.892997 | 2.0480e+03 | 3.7350e+00 | pass |
| swiglu_output__f16 | single_stage | e05c6c03 | 1 | 0.999581 | 0.986049 | 1.0240e+03 | 8.8148e-01 | pass |
| gemm2_operands__bf16 | single_stage | e05c6c03 | 1 | 0.994141 | 0.842773 | 2.0480e+03 | 5.5126e+00 | pass |
| gemm2_operands__f16 | single_stage | e05c6c03 | 1 | 0.999442 | 0.981585 | 1.0240e+03 | 1.4422e+00 | pass |
| gemm2_accumulator__bf16 | single_stage | e05c6c03 | 1 | 0.991908 | 0.747210 | 3.0720e+03 | 9.5366e+00 | pass |
| gemm2_accumulator__f16 | single_stage | e05c6c03 | 1 | 0.626395 | 0.625837 | nan | nan | global_drift |
| out_accumulator__bf16 | single_stage | e05c6c03 | 1 | 0.998465 | 0.935965 | 2.0480e+03 | 4.3568e+00 | pass |
| out_accumulator__f16 | single_stage | e05c6c03 | 1 | 0.580218 | 0.573382 | inf | inf | catastrophic_outlier |
| baseline__fp32 | single_stage | 6230e838 | 32 | 1.000000 | 1.000000 | 0.0000e+00 | 0.0000e+00 | pass |
| hidden_dequant__bf16 | single_stage | 6230e838 | 32 | 0.998047 | 0.937509 | 2.0480e+03 | 3.6249e+01 | pass |
| hidden_dequant__f16 | single_stage | 6230e838 | 32 | 0.999765 | 0.992314 | 2.0480e+03 | 8.4452e+00 | pass |
| gemm1_operands__bf16 | single_stage | 6230e838 | 32 | 0.997158 | 0.914298 | 4.0960e+03 | 6.5000e+01 | pass |
| gemm1_operands__f16 | single_stage | 6230e838 | 32 | 0.999664 | 0.988918 | 2.0480e+03 | 6.4971e+00 | pass |
| gemm1_accumulator__bf16 | single_stage | 6230e838 | 32 | 0.989376 | 0.770133 | 8.1920e+03 | 3.4112e+02 | pass |
| gemm1_accumulator__f16 | single_stage | 6230e838 | 32 | 0.998696 | 0.957785 | 2.0480e+03 | 2.6644e+01 | pass |
| gemm1_output__bf16 | single_stage | 6230e838 | 32 | 0.998095 | 0.938237 | 2.0480e+03 | 9.3968e+01 | pass |
| gemm1_output__f16 | single_stage | 6230e838 | 32 | 0.999813 | 0.992209 | 2.0480e+03 | 4.9058e+00 | pass |
| swiglu_input__bf16 | single_stage | 6230e838 | 32 | 0.998095 | 0.938237 | 2.0480e+03 | 9.3968e+01 | pass |
| swiglu_input__f16 | single_stage | 6230e838 | 32 | 0.999813 | 0.992209 | 2.0480e+03 | 4.9058e+00 | pass |
| swiglu_output__bf16 | single_stage | 6230e838 | 32 | 0.998696 | 0.956730 | 2.0480e+03 | 3.4622e+01 | pass |
| swiglu_output__f16 | single_stage | 6230e838 | 32 | 0.968602 | 0.964068 | nan | nan | pass |
| gemm2_operands__bf16 | single_stage | 6230e838 | 32 | 0.998160 | 0.938869 | 4.0960e+03 | 4.3765e+01 | pass |
| gemm2_operands__f16 | single_stage | 6230e838 | 32 | 0.968554 | 0.961984 | nan | nan | pass |
| gemm2_accumulator__bf16 | single_stage | 6230e838 | 32 | 0.996953 | 0.901441 | 7.1680e+03 | 1.1406e+02 | pass |
| gemm2_accumulator__f16 | single_stage | 6230e838 | 32 | 0.838824 | 0.838771 | nan | nan | global_drift |
| out_accumulator__bf16 | single_stage | 6230e838 | 32 | 0.999350 | 0.977147 | 4.0960e+03 | 1.5922e+01 | pass |
| out_accumulator__f16 | single_stage | 6230e838 | 32 | 0.754591 | 0.753100 | inf | inf | catastrophic_outlier |
| baseline__fp32 | single_stage | 1a4c6ba1 | 901 | 1.000000 | 1.000000 | 0.0000e+00 | 0.0000e+00 | pass |
| hidden_dequant__bf16 | single_stage | 1a4c6ba1 | 901 | 0.996886 | 0.904165 | 4.0960e+03 | 7.6107e+03 | pass |
| hidden_dequant__f16 | single_stage | 1a4c6ba1 | 901 | 0.999611 | 0.987738 | 4.0960e+03 | 2.3371e+03 | pass |
| gemm1_operands__bf16 | single_stage | 1a4c6ba1 | 901 | 0.995581 | 0.866546 | 4.0960e+03 | 7.5083e+03 | pass |
| gemm1_operands__f16 | single_stage | 1a4c6ba1 | 901 | 0.999453 | 0.982620 | 4.0960e+03 | 3.4465e+03 | pass |
| gemm1_accumulator__bf16 | single_stage | 1a4c6ba1 | 901 | 0.983733 | 0.641791 | 1.2288e+04 | 4.5243e+04 | pass |
| gemm1_accumulator__f16 | single_stage | 1a4c6ba1 | 901 | 0.997906 | 0.935476 | 4.0960e+03 | 1.4151e+04 | pass |
| gemm1_output__bf16 | single_stage | 1a4c6ba1 | 901 | 0.996882 | 0.904181 | 4.0960e+03 | 1.6997e+04 | pass |
| gemm1_output__f16 | single_stage | 1a4c6ba1 | 901 | 0.999602 | 0.987789 | 4.0960e+03 | 1.3799e+03 | pass |
| swiglu_input__bf16 | single_stage | 1a4c6ba1 | 901 | 0.996882 | 0.904181 | 4.0960e+03 | 1.6997e+04 | pass |
| swiglu_input__f16 | single_stage | 1a4c6ba1 | 901 | 0.999602 | 0.987789 | 4.0960e+03 | 1.3799e+03 | pass |
| swiglu_output__bf16 | single_stage | 1a4c6ba1 | 901 | 0.997782 | 0.931706 | 4.0960e+03 | 2.9535e+03 | pass |
| swiglu_output__f16 | single_stage | 1a4c6ba1 | 901 | 0.957309 | 0.949594 | nan | nan | pass |
| gemm2_operands__bf16 | single_stage | 1a4c6ba1 | 901 | 0.996916 | 0.904637 | 4.0960e+03 | 7.1670e+03 | pass |
| gemm2_operands__f16 | single_stage | 1a4c6ba1 | 901 | 0.957204 | 0.946328 | nan | nan | pass |
| gemm2_accumulator__bf16 | single_stage | 1a4c6ba1 | 901 | 0.995476 | 0.853328 | 8.1920e+03 | 2.1131e+04 | pass |
| gemm2_accumulator__f16 | single_stage | 1a4c6ba1 | 901 | 0.678651 | 0.678374 | nan | nan | global_drift |
| out_accumulator__bf16 | single_stage | 1a4c6ba1 | 901 | 0.999103 | 0.971092 | 4.0960e+03 | 1.5094e+03 | pass |
| out_accumulator__f16 | single_stage | 1a4c6ba1 | 901 | 0.686777 | 0.684688 | inf | inf | catastrophic_outlier |
| baseline__fp32 | single_stage | 5e8dc11c | 14107 | 1.000000 | 1.000000 | 0.0000e+00 | 0.0000e+00 | pass |
| hidden_dequant__bf16 | single_stage | 5e8dc11c | 14107 | 0.997161 | 0.912810 | 4.0960e+03 | 9.8047e+05 | pass |
| hidden_dequant__f16 | single_stage | 5e8dc11c | 14107 | 0.999649 | 0.988893 | 4.0960e+03 | 5.5690e+03 | pass |
| gemm1_operands__bf16 | single_stage | 5e8dc11c | 14107 | 0.995973 | 0.878870 | 4.0960e+03 | 4.4598e+05 | pass |
| gemm1_operands__f16 | single_stage | 5e8dc11c | 14107 | 0.999499 | 0.984239 | 4.0960e+03 | 9.5750e+04 | pass |
| gemm1_accumulator__bf16 | single_stage | 5e8dc11c | 14107 | 0.985277 | 0.675917 | 1.6384e+04 | 1.5320e+06 | pass |
| gemm1_accumulator__f16 | single_stage | 5e8dc11c | 14107 | 0.998125 | 0.941638 | 8.1920e+03 | 2.1363e+05 | pass |
| gemm1_output__bf16 | single_stage | 5e8dc11c | 14107 | 0.997178 | 0.913153 | 4.0960e+03 | 2.3490e+05 | pass |
| gemm1_output__f16 | single_stage | 5e8dc11c | 14107 | 0.999648 | 0.988924 | 4.0960e+03 | 6.8938e+04 | pass |
| swiglu_input__bf16 | single_stage | 5e8dc11c | 14107 | 0.997178 | 0.913153 | 4.0960e+03 | 2.3490e+05 | pass |
| swiglu_input__f16 | single_stage | 5e8dc11c | 14107 | 0.999648 | 0.988924 | 4.0960e+03 | 6.8938e+04 | pass |
| swiglu_output__bf16 | single_stage | 5e8dc11c | 14107 | 0.998007 | 0.938007 | 4.0960e+03 | 2.6725e+05 | pass |
| swiglu_output__f16 | single_stage | 5e8dc11c | 14107 | 0.956238 | 0.949300 | nan | nan | pass |
| gemm2_operands__bf16 | single_stage | 5e8dc11c | 14107 | 0.997198 | 0.913606 | 4.0960e+03 | 1.6001e+05 | pass |
| gemm2_operands__f16 | single_stage | 5e8dc11c | 14107 | 0.956146 | 0.946378 | nan | nan | pass |
| gemm2_accumulator__bf16 | single_stage | 5e8dc11c | 14107 | 0.996196 | 0.871353 | 1.2288e+04 | 4.5279e+05 | pass |
| gemm2_accumulator__f16 | single_stage | 5e8dc11c | 14107 | 0.655088 | 0.654711 | nan | nan | global_drift |
| out_accumulator__bf16 | single_stage | 5e8dc11c | 14107 | 0.999363 | 0.979636 | 8.1920e+03 | 2.6241e+04 | pass |
| out_accumulator__f16 | single_stage | 5e8dc11c | 14107 | 0.747705 | 0.746125 | inf | inf | catastrophic_outlier |
| baseline__fp32 | single_stage | 58a34f27 | 11948 | 1.000000 | 1.000000 | 0.0000e+00 | 0.0000e+00 | pass |
| hidden_dequant__bf16 | single_stage | 58a34f27 | 11948 | 0.997835 | 0.933303 | 8.1920e+03 | 2.3625e+09 | pass |
| hidden_dequant__f16 | single_stage | 58a34f27 | 11948 | 0.999729 | 0.991484 | 4.0960e+03 | 1.8750e+09 | pass |
| gemm1_operands__bf16 | single_stage | 58a34f27 | 11948 | 0.996917 | 0.907284 | 4.0960e+03 | 7.7000e+09 | pass |
| gemm1_operands__f16 | single_stage | 58a34f27 | 11948 | 0.999617 | 0.987919 | 4.0960e+03 | 3.3500e+09 | pass |
| gemm1_accumulator__bf16 | single_stage | 58a34f27 | 11948 | 0.988732 | 0.751603 | 1.6384e+04 | 1.7680e+11 | pass |
| gemm1_accumulator__f16 | single_stage | 58a34f27 | 11948 | 0.998566 | 0.955282 | 8.1920e+03 | 1.6000e+10 | pass |
| gemm1_output__bf16 | single_stage | 58a34f27 | 11948 | 0.997845 | 0.933583 | 8.1920e+03 | 6.4500e+09 | pass |
| gemm1_output__f16 | single_stage | 58a34f27 | 11948 | 0.999730 | 0.991511 | 4.0960e+03 | 8.3750e+08 | pass |
| swiglu_input__bf16 | single_stage | 58a34f27 | 11948 | 0.997845 | 0.933583 | 8.1920e+03 | 6.4500e+09 | pass |
| swiglu_input__f16 | single_stage | 58a34f27 | 11948 | 0.999730 | 0.991511 | 4.0960e+03 | 8.3750e+08 | pass |
| swiglu_output__bf16 | single_stage | 58a34f27 | 11948 | 0.998477 | 0.952491 | 8.1920e+03 | 1.8000e+09 | pass |
| swiglu_output__f16 | single_stage | 58a34f27 | 11948 | 0.966158 | 0.960856 | nan | nan | pass |
| gemm2_operands__bf16 | single_stage | 58a34f27 | 11948 | 0.997856 | 0.933802 | 4.0960e+03 | 1.2500e+10 | pass |
| gemm2_operands__f16 | single_stage | 58a34f27 | 11948 | 0.966088 | 0.958616 | nan | nan | pass |
| gemm2_accumulator__bf16 | single_stage | 58a34f27 | 11948 | 0.997077 | 0.901170 | 1.2288e+04 | 1.2438e+09 | pass |
| gemm2_accumulator__f16 | single_stage | 58a34f27 | 11948 | 0.734762 | 0.734490 | nan | nan | global_drift |
| out_accumulator__bf16 | single_stage | 58a34f27 | 11948 | 0.999510 | 0.984290 | 4.0960e+03 | 2.2800e+10 | pass |
| out_accumulator__f16 | single_stage | 58a34f27 | 11948 | 0.799730 | 0.798540 | inf | inf | catastrophic_outlier |
| baseline__fp32 | single_stage | 4822167c | 56 | 1.000000 | 1.000000 | 0.0000e+00 | 0.0000e+00 | pass |
| hidden_dequant__bf16 | single_stage | 4822167c | 56 | 0.996961 | 0.907540 | 4.0960e+03 | 7.2740e+02 | pass |
| hidden_dequant__f16 | single_stage | 4822167c | 56 | 0.999631 | 0.988192 | 2.0480e+03 | 3.2060e+02 | pass |
| gemm1_operands__bf16 | single_stage | 4822167c | 56 | 0.995690 | 0.871149 | 4.0960e+03 | 1.4602e+03 | pass |
| gemm1_operands__f16 | single_stage | 4822167c | 56 | 0.999472 | 0.983281 | 2.0480e+03 | 2.0860e+02 | pass |
| gemm1_accumulator__bf16 | single_stage | 4822167c | 56 | 0.984467 | 0.656228 | 8.4480e+03 | 3.3837e+03 | pass |
| gemm1_accumulator__f16 | single_stage | 4822167c | 56 | 0.998057 | 0.938048 | 2.0480e+03 | 6.6780e+02 | pass |
| gemm1_output__bf16 | single_stage | 4822167c | 56 | 0.996991 | 0.907585 | 4.0960e+03 | 1.3046e+03 | pass |
| gemm1_output__f16 | single_stage | 4822167c | 56 | 0.999599 | 0.988695 | 2.0480e+03 | 1.2287e+02 | pass |
| swiglu_input__bf16 | single_stage | 4822167c | 56 | 0.996991 | 0.907585 | 4.0960e+03 | 1.3046e+03 | pass |
| swiglu_input__f16 | single_stage | 4822167c | 56 | 0.999599 | 0.988695 | 2.0480e+03 | 1.2287e+02 | pass |
| swiglu_output__bf16 | single_stage | 4822167c | 56 | 0.997925 | 0.935335 | 4.0960e+03 | 1.9210e+03 | pass |
| swiglu_output__f16 | single_stage | 4822167c | 56 | 0.973115 | 0.965778 | nan | nan | pass |
| gemm2_operands__bf16 | single_stage | 4822167c | 56 | 0.997043 | 0.909516 | 4.0960e+03 | 2.7658e+03 | pass |
| gemm2_operands__f16 | single_stage | 4822167c | 56 | 0.972983 | 0.962642 | nan | nan | pass |
| gemm2_accumulator__bf16 | single_stage | 4822167c | 56 | 0.995974 | 0.861722 | 4.3520e+03 | 1.1456e+03 | pass |
| gemm2_accumulator__f16 | single_stage | 4822167c | 56 | 0.660582 | 0.660121 | nan | nan | global_drift |
| out_accumulator__bf16 | single_stage | 4822167c | 56 | 0.999230 | 0.975688 | 4.0960e+03 | 3.7980e+02 | pass |
| out_accumulator__f16 | single_stage | 4822167c | 56 | 0.739076 | 0.737357 | inf | inf | catastrophic_outlier |
| baseline__fp32 | single_stage | f7d6ac7c | 52 | 1.000000 | 1.000000 | 0.0000e+00 | 0.0000e+00 | pass |
| hidden_dequant__bf16 | single_stage | f7d6ac7c | 52 | 0.998473 | 0.956202 | 2.0480e+03 | 1.3664e+02 | pass |
| hidden_dequant__f16 | single_stage | f7d6ac7c | 52 | 0.999761 | 0.994471 | 2.0480e+03 | 1.3800e+01 | pass |
| gemm1_operands__bf16 | single_stage | f7d6ac7c | 52 | 0.997905 | 0.938224 | 2.0480e+03 | 2.5373e+02 | pass |
| gemm1_operands__f16 | single_stage | f7d6ac7c | 52 | 0.999700 | 0.992115 | 2.0480e+03 | 8.0709e+00 | pass |
| gemm1_accumulator__bf16 | single_stage | f7d6ac7c | 52 | 0.992330 | 0.834135 | 7.1680e+03 | 2.2043e+02 | pass |
| gemm1_accumulator__f16 | single_stage | f7d6ac7c | 52 | 0.998986 | 0.970014 | 2.0480e+03 | 4.3896e+01 | pass |
| gemm1_output__bf16 | single_stage | f7d6ac7c | 52 | 0.998417 | 0.957957 | 2.0480e+03 | 9.9070e+01 | pass |
| gemm1_output__f16 | single_stage | f7d6ac7c | 52 | 0.999785 | 0.994226 | 2.0480e+03 | 1.6183e+01 | pass |
| swiglu_input__bf16 | single_stage | f7d6ac7c | 52 | 0.998417 | 0.957957 | 2.0480e+03 | 9.9070e+01 | pass |
| swiglu_input__f16 | single_stage | f7d6ac7c | 52 | 0.999785 | 0.994226 | 2.0480e+03 | 1.6183e+01 | pass |
| swiglu_output__bf16 | single_stage | f7d6ac7c | 52 | 0.998962 | 0.969351 | 2.0480e+03 | 7.0960e+01 | pass |
| swiglu_output__f16 | single_stage | f7d6ac7c | 52 | 0.980630 | 0.977161 | nan | nan | pass |
| gemm2_operands__bf16 | single_stage | f7d6ac7c | 52 | 0.998503 | 0.957329 | 2.0480e+03 | 5.5374e+01 | pass |
| gemm2_operands__f16 | single_stage | f7d6ac7c | 52 | 0.980592 | 0.975640 | nan | nan | pass |
| gemm2_accumulator__bf16 | single_stage | f7d6ac7c | 52 | 0.998098 | 0.937613 | 4.0960e+03 | 1.0726e+02 | pass |
| gemm2_accumulator__f16 | single_stage | f7d6ac7c | 52 | 0.805991 | 0.805739 | nan | nan | global_drift |
| out_accumulator__bf16 | single_stage | f7d6ac7c | 52 | 0.999705 | 0.991498 | 2.0480e+03 | 1.3402e+01 | pass |
| out_accumulator__f16 | single_stage | f7d6ac7c | 52 | 0.881745 | 0.881010 | inf | inf | catastrophic_outlier |
| gemm1_accumulator__f16 | cumulative | b8f4f012 | 7 | 0.999163 | 0.971919 | 2.0480e+03 | 6.6422e+01 | pass |
| gemm1_accumulator__f16 | cumulative | e05c6c03 | 1 | 0.995675 | 0.905413 | 2.0480e+03 | 1.1557e+02 | pass |
| gemm1_accumulator__f16 | cumulative | 6230e838 | 32 | 0.998548 | 0.957393 | 4.0960e+03 | 1.6438e+02 | pass |
| gemm1_accumulator__f16 | cumulative | 1a4c6ba1 | 901 | 0.997949 | 0.935806 | 4.0960e+03 | 5.1807e+03 | pass |
| gemm1_accumulator__f16 | cumulative | 5e8dc11c | 14107 | 0.998123 | 0.941586 | 4.0960e+03 | 2.7145e+04 | pass |
| gemm1_accumulator__f16 | cumulative | 58a34f27 | 11948 | 0.998566 | 0.955216 | 4.0960e+03 | 4.0729e+04 | pass |
| gemm1_accumulator__f16 | cumulative | 4822167c | 56 | 0.998079 | 0.938175 | 2.0480e+03 | 2.9742e+02 | pass |
| gemm1_accumulator__f16 | cumulative | f7d6ac7c | 52 | 0.999082 | 0.971014 | 2.0480e+03 | 1.6046e+02 | pass |
| gemm1_accumulator__f16+gemm1_operands__f16 | cumulative | b8f4f012 | 7 | 0.999801 | 0.991988 | 2.0480e+03 | 6.9385e+00 | pass |
| gemm1_accumulator__f16+gemm1_operands__f16 | cumulative | e05c6c03 | 1 | 0.999302 | 0.968471 | 1.0240e+03 | 1.7186e+00 | pass |
| gemm1_accumulator__f16+gemm1_operands__f16 | cumulative | 6230e838 | 32 | 0.999643 | 0.989114 | 2.0480e+03 | 6.4317e+01 | pass |
| gemm1_accumulator__f16+gemm1_operands__f16 | cumulative | 1a4c6ba1 | 901 | 0.999448 | 0.982652 | 4.0960e+03 | 6.1603e+02 | pass |
| gemm1_accumulator__f16+gemm1_operands__f16 | cumulative | 5e8dc11c | 14107 | 0.999497 | 0.984232 | 4.0960e+03 | 8.0500e+09 | pass |
| gemm1_accumulator__f16+gemm1_operands__f16 | cumulative | 58a34f27 | 11948 | 0.999615 | 0.987910 | 4.0960e+03 | 6.5500e+09 | pass |
| gemm1_accumulator__f16+gemm1_operands__f16 | cumulative | 4822167c | 56 | 0.999469 | 0.983451 | 2.0480e+03 | 3.6470e+03 | pass |
| gemm1_accumulator__f16+gemm1_operands__f16 | cumulative | f7d6ac7c | 52 | 0.999761 | 0.992209 | 2.0480e+03 | 8.4847e+01 | pass |
| gemm1_accumulator__f16+gemm1_operands__f16+gemm1_output__f16 | cumulative | b8f4f012 | 7 | 0.999741 | 0.990673 | 2.0480e+03 | 2.2458e+01 | pass |
| gemm1_accumulator__f16+gemm1_operands__f16+gemm1_output__f16 | cumulative | e05c6c03 | 1 | 0.999442 | 0.965402 | 1.0240e+03 | 1.0101e+00 | pass |
| gemm1_accumulator__f16+gemm1_operands__f16+gemm1_output__f16 | cumulative | 6230e838 | 32 | 0.999555 | 0.986529 | 2.0480e+03 | 2.1633e+04 | pass |
| gemm1_accumulator__f16+gemm1_operands__f16+gemm1_output__f16 | cumulative | 1a4c6ba1 | 901 | 0.999341 | 0.978717 | 4.0960e+03 | 1.4687e+09 | pass |
| gemm1_accumulator__f16+gemm1_operands__f16+gemm1_output__f16 | cumulative | 5e8dc11c | 14107 | 0.999387 | 0.980768 | 4.0960e+03 | 4.1985e+04 | pass |
| gemm1_accumulator__f16+gemm1_operands__f16+gemm1_output__f16 | cumulative | 58a34f27 | 11948 | 0.999529 | 0.985214 | 4.0960e+03 | 1.0100e+10 | pass |
| gemm1_accumulator__f16+gemm1_operands__f16+gemm1_output__f16 | cumulative | 4822167c | 56 | 0.999300 | 0.979290 | 2.0480e+03 | 3.2300e+02 | pass |
| gemm1_accumulator__f16+gemm1_operands__f16+gemm1_output__f16 | cumulative | f7d6ac7c | 52 | 0.999697 | 0.990108 | 2.0480e+03 | 1.9251e+02 | pass |
| gemm1_accumulator__f16+gemm1_operands__f16+gemm1_output__f16+gemm2_accumulator__bf16 | cumulative | b8f4f012 | 7 | 0.997688 | 0.927236 | 5.6320e+03 | 8.3772e+01 | pass |
| gemm1_accumulator__f16+gemm1_operands__f16+gemm1_output__f16+gemm2_accumulator__bf16 | cumulative | e05c6c03 | 1 | 0.991071 | 0.763811 | 3.0720e+03 | 1.3732e+01 | pass |
| gemm1_accumulator__f16+gemm1_operands__f16+gemm1_output__f16+gemm2_accumulator__bf16 | cumulative | 6230e838 | 32 | 0.996634 | 0.899397 | 4.0960e+03 | 1.5367e+02 | pass |
| gemm1_accumulator__f16+gemm1_operands__f16+gemm1_output__f16+gemm2_accumulator__bf16 | cumulative | 1a4c6ba1 | 901 | 0.995400 | 0.851389 | 8.1920e+03 | 1.5231e+04 | pass |
| gemm1_accumulator__f16+gemm1_operands__f16+gemm1_output__f16+gemm2_accumulator__bf16 | cumulative | 5e8dc11c | 14107 | 0.996143 | 0.869755 | 1.0240e+04 | 1.4200e+10 | pass |
| gemm1_accumulator__f16+gemm1_operands__f16+gemm1_output__f16+gemm2_accumulator__bf16 | cumulative | 58a34f27 | 11948 | 0.997048 | 0.900988 | 1.2288e+04 | 1.2900e+10 | pass |
| gemm1_accumulator__f16+gemm1_operands__f16+gemm1_output__f16+gemm2_accumulator__bf16 | cumulative | 4822167c | 56 | 0.995673 | 0.859475 | 6.1440e+03 | 6.0662e+03 | pass |
| gemm1_accumulator__f16+gemm1_operands__f16+gemm1_output__f16+gemm2_accumulator__bf16 | cumulative | f7d6ac7c | 52 | 0.998127 | 0.935606 | 4.8640e+03 | 3.1945e+02 | pass |
| gemm1_accumulator__f16+gemm1_operands__f16+gemm1_output__f16+gemm2_accumulator__bf16+gemm2_operands__f16 | cumulative | b8f4f012 | 7 | 0.821189 | 0.821189 | nan | nan | global_drift |
| gemm1_accumulator__f16+gemm1_operands__f16+gemm1_output__f16+gemm2_accumulator__bf16+gemm2_operands__f16 | cumulative | e05c6c03 | 1 | 0.999163 | 0.958147 | 2.0480e+03 | 2.0109e+00 | pass |
| gemm1_accumulator__f16+gemm1_operands__f16+gemm1_output__f16+gemm2_accumulator__bf16+gemm2_operands__f16 | cumulative | 6230e838 | 32 | 0.999446 | 0.984249 | 4.0960e+03 | 2.8922e+02 | pass |
| gemm1_accumulator__f16+gemm1_operands__f16+gemm1_output__f16+gemm2_accumulator__bf16+gemm2_operands__f16 | cumulative | 1a4c6ba1 | 901 | 0.939976 | 0.918740 | nan | nan | pass |
| gemm1_accumulator__f16+gemm1_operands__f16+gemm1_output__f16+gemm2_accumulator__bf16+gemm2_operands__f16 | cumulative | 5e8dc11c | 14107 | 0.957362 | 0.937584 | nan | nan | pass |
| gemm1_accumulator__f16+gemm1_operands__f16+gemm1_output__f16+gemm2_accumulator__bf16+gemm2_operands__f16 | cumulative | 58a34f27 | 11948 | 0.967108 | 0.952006 | nan | nan | pass |
| gemm1_accumulator__f16+gemm1_operands__f16+gemm1_output__f16+gemm2_accumulator__bf16+gemm2_operands__f16 | cumulative | 4822167c | 56 | 0.954525 | 0.933175 | nan | nan | pass |
| gemm1_accumulator__f16+gemm1_operands__f16+gemm1_output__f16+gemm2_accumulator__bf16+gemm2_operands__f16 | cumulative | f7d6ac7c | 52 | 0.980466 | 0.970513 | nan | nan | pass |