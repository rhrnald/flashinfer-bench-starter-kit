# MoE Precision Search 2026-04-23T18:56:06Z

| metric | value |
|---|---|
| timestamp | 2026-04-23T18:56:06Z |
| definition | moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048 |
| seed | 1234 |
| atol | 1.0 |
| rtol | 0.3 |
| required_matched_ratio | 0.9 |
| strict_atol | 0.01 |
| strict_rtol | 0.01 |
| device | NVIDIA B200 |
| evidence_scope | oracle_only |
| kernel_validated_candidates | - |
| run_stage | stage1_bf16_f16 |
| panel_size | 19 |
| panel_indices | 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18 |
| n_candidates | 19 |
| n_survivors | 16 |
| contest_safe_survivor_count | 16 |
| contest_safe_survivors | gemm1_output__f16, swiglu_input__f16, gemm1_operands__f16, hidden_dequant__f16, out_accumulator__bf16, swiglu_output__bf16, gemm1_accumulator__f16, hidden_dequant__bf16, gemm1_output__bf16, swiglu_input__bf16, gemm2_operands__bf16, gemm1_operands__bf16, gemm2_accumulator__bf16, gemm1_accumulator__bf16, swiglu_output__f16, gemm2_operands__f16 |
| strict_safe_survivor_count | 7 |
| strict_safe_survivors | hidden_dequant__f16, gemm1_operands__f16, gemm1_output__f16, swiglu_input__f16, swiglu_output__f16, gemm2_operands__f16, out_accumulator__bf16 |
| contest_only_survivor_count | 9 |
| contest_only_survivors | hidden_dequant__bf16, gemm1_operands__bf16, gemm1_accumulator__bf16, gemm1_accumulator__f16, gemm1_output__bf16, swiglu_input__bf16, swiglu_output__bf16, gemm2_operands__bf16, gemm2_accumulator__bf16 |
| promote_top_k_per_stage | 1 |

## Evidence Scope

| field | value |
|---|---|
| evidence_scope | oracle_only |
| kernel_validated_candidates | - |

## Oracle Survivor Summary

| category | count | candidates |
|---|---:|---|
| contest_safe_oracle_single_stage | 16 | gemm1_output__f16, swiglu_input__f16, gemm1_operands__f16, hidden_dequant__f16, out_accumulator__bf16, swiglu_output__bf16, gemm1_accumulator__f16, hidden_dequant__bf16, gemm1_output__bf16, swiglu_input__bf16, gemm2_operands__bf16, gemm1_operands__bf16, gemm2_accumulator__bf16, gemm1_accumulator__bf16, swiglu_output__f16, gemm2_operands__f16 |
| strict_safe_oracle_single_stage | 7 | hidden_dequant__f16, gemm1_operands__f16, gemm1_output__f16, swiglu_input__f16, swiglu_output__f16, gemm2_operands__f16, out_accumulator__bf16 |
| contest_only_oracle_single_stage | 9 | hidden_dequant__bf16, gemm1_operands__bf16, gemm1_accumulator__bf16, gemm1_accumulator__f16, gemm1_output__bf16, swiglu_input__bf16, swiglu_output__bf16, gemm2_operands__bf16, gemm2_accumulator__bf16 |

## Stage Summary

| stage | best_contest_safe_mode | scale | worst_matched_contest | worst_matched_strict | worst_rel | contest_status | strict_status | evidence |
|---|---|---|---:|---:|---:|---|---|---|
| gemm1_accumulator | f16 | none | 0.996233 | 0.893834 | 4.5750e+09 | contest_safe | strict_unsafe | oracle_only |
| gemm1_operands | f16 | none | 0.999442 | 0.975167 | 4.2500e+09 | contest_safe | strict_safe | oracle_only |
| gemm1_output | f16 | none | 0.999607 | 0.982143 | 2.8875e+09 | contest_safe | strict_safe | oracle_only |
| gemm2_accumulator | bf16 | none | 0.991908 | 0.747210 | 1.1920e+11 | contest_safe | strict_unsafe | oracle_only |
| gemm2_operands | f16 | none | 0.918485 | 0.908690 | 8.4035e+00 | contest_safe | strict_safe | oracle_only |
| hidden_dequant | f16 | none | 0.999302 | 0.983538 | 2.3250e+09 | contest_safe | strict_safe | oracle_only |
| out_accumulator | bf16 | none | 0.998465 | 0.935965 | 2.5000e+10 | contest_safe | strict_safe | oracle_only |
| swiglu_input | f16 | none | 0.999607 | 0.982143 | 2.8875e+09 | contest_safe | strict_safe | oracle_only |
| swiglu_output | f16 | none | 0.918558 | 0.911602 | 7.2807e+00 | contest_safe | strict_safe | oracle_only |

## Contest-Safe Oracle Frontier

| order | candidate | worst_matched_contest | worst_matched_strict | worst_rel | contest_status | strict_status | evidence | kept |
|---|---|---:|---:|---:|---|---|---|---|
| 1 | hidden_dequant__bf16 | 0.994978 | 0.846122 | 1.0547e+05 | contest_safe | strict_unsafe | oracle_only | yes |
| 2 | hidden_dequant__bf16+hidden_dequant__f16 | 0.999302 | 0.975028 | 1.7665e+04 | contest_safe | strict_safe | oracle_only | yes |
| 3 | hidden_dequant__bf16+hidden_dequant__f16+gemm1_operands__bf16 | 0.993164 | 0.785993 | 6.1200e+10 | contest_safe | strict_unsafe | oracle_only | yes |
| 4 | hidden_dequant__bf16+hidden_dequant__f16+gemm1_operands__bf16+gemm1_operands__f16 | 0.998465 | 0.967634 | 2.3937e+04 | contest_safe | strict_safe | oracle_only | yes |
| 5 | hidden_dequant__bf16+hidden_dequant__f16+gemm1_operands__bf16+gemm1_operands__f16+gemm1_output__bf16 | 0.994280 | 0.830357 | 2.6094e+08 | contest_safe | strict_unsafe | oracle_only | yes |
| 6 | hidden_dequant__bf16+hidden_dequant__f16+gemm1_operands__bf16+gemm1_operands__f16+gemm1_output__bf16+gemm1_output__f16 | 0.999294 | 0.969029 | 9.0000e+09 | contest_safe | strict_safe | oracle_only | yes |
| 7 | hidden_dequant__bf16+hidden_dequant__f16+gemm1_operands__bf16+gemm1_operands__f16+gemm1_output__bf16+gemm1_output__f16+swiglu_input__bf16 | 0.994280 | 0.837612 | 3.0542e+04 | contest_safe | strict_unsafe | oracle_only | yes |
| 8 | hidden_dequant__bf16+hidden_dequant__f16+gemm1_operands__bf16+gemm1_operands__f16+gemm1_output__bf16+gemm1_output__f16+swiglu_input__bf16+swiglu_input__f16 | 0.999163 | 0.970006 | 2.8906e+08 | contest_safe | strict_safe | oracle_only | yes |
| 9 | hidden_dequant__bf16+hidden_dequant__f16+gemm1_operands__bf16+gemm1_operands__f16+gemm1_output__bf16+gemm1_output__f16+swiglu_input__bf16+swiglu_input__f16+swiglu_output__bf16 | 0.996094 | 0.889090 | 3.8145e+04 | contest_safe | strict_unsafe | oracle_only | yes |
| 10 | hidden_dequant__bf16+hidden_dequant__f16+gemm1_operands__bf16+gemm1_operands__f16+gemm1_output__bf16+gemm1_output__f16+swiglu_input__bf16+swiglu_input__f16+swiglu_output__bf16+swiglu_output__f16 | 0.930481 | 0.911900 | 8.0577e+01 | contest_safe | strict_safe | oracle_only | yes |
| 11 | hidden_dequant__bf16+hidden_dequant__f16+gemm1_operands__bf16+gemm1_operands__f16+gemm1_output__bf16+gemm1_output__f16+swiglu_input__bf16+swiglu_input__f16+swiglu_output__bf16+swiglu_output__f16+gemm2_operands__bf16 | 0.872628 | 0.796413 | 5.1100e+02 | contest_unsafe | strict_unsafe | oracle_only | no |
| 12 | hidden_dequant__bf16+hidden_dequant__f16+gemm1_operands__bf16+gemm1_operands__f16+gemm1_output__bf16+gemm1_output__f16+swiglu_input__bf16+swiglu_input__f16+swiglu_output__bf16+swiglu_output__f16+gemm2_operands__f16 | 0.000837 | 0.000837 | nan | contest_unsafe | strict_unsafe | oracle_only | no |
| 13 | hidden_dequant__bf16+hidden_dequant__f16+gemm1_operands__bf16+gemm1_operands__f16+gemm1_output__bf16+gemm1_output__f16+swiglu_input__bf16+swiglu_input__f16+swiglu_output__bf16+swiglu_output__f16+gemm2_accumulator__bf16 | 0.784977 | 0.743862 | nan | contest_unsafe | strict_unsafe | oracle_only | no |
| 14 | hidden_dequant__bf16+hidden_dequant__f16+gemm1_operands__bf16+gemm1_operands__f16+gemm1_output__bf16+gemm1_output__f16+swiglu_input__bf16+swiglu_input__f16+swiglu_output__bf16+swiglu_output__f16+out_accumulator__bf16 | 0.856904 | 0.849091 | nan | contest_unsafe | strict_unsafe | oracle_only | no |
| 15 | hidden_dequant__bf16+hidden_dequant__f16+gemm1_operands__bf16+gemm1_operands__f16+gemm1_output__bf16+gemm1_output__f16+swiglu_input__bf16+swiglu_input__f16+swiglu_output__bf16+swiglu_output__f16+gemm1_accumulator__bf16 | 0.877499 | 0.861253 | 8.2843e+02 | contest_unsafe | strict_unsafe | oracle_only | no |
| 16 | hidden_dequant__bf16+hidden_dequant__f16+gemm1_operands__bf16+gemm1_operands__f16+gemm1_output__bf16+gemm1_output__f16+swiglu_input__bf16+swiglu_input__f16+swiglu_output__bf16+swiglu_output__f16+gemm1_accumulator__f16 | 0.857083 | 0.851782 | nan | contest_unsafe | strict_unsafe | oracle_only | no |

## BF16/F16 Margin

| stage | preferred | bf16_contest | f16_contest | bf16_strict | f16_strict | |
|---|---|---:|---:|---:|---:|---|
| hidden_dequant | f16 | 0.995396 | 0.999302 | 0.847377 | 0.983538 | |
| gemm1_operands | f16 | 0.994001 | 0.999442 | 0.788086 | 0.975167 | |
| gemm1_accumulator | f16 | 0.977260 | 0.996233 | 0.422294 | 0.893834 | |
| gemm1_output | f16 | 0.994420 | 0.999607 | 0.847935 | 0.982143 | |
| swiglu_input | f16 | 0.994420 | 0.999607 | 0.847935 | 0.982143 | |
| swiglu_output | bf16 | 0.996512 | 0.918558 | 0.892997 | 0.911602 | |
| gemm2_operands | bf16 | 0.994141 | 0.918485 | 0.842773 | 0.908690 | |
| gemm2_accumulator | bf16 | 0.991908 | 0.595494 | 0.747210 | 0.594918 | |
| out_accumulator | bf16 | 0.998465 | 0.580218 | 0.935965 | 0.573382 | |

## Pairwise Summary

| pair | worst_matched_contest | worst_matched_strict | worst_rel | contest_status | strict_status | evidence |
|---|---:|---:|---:|---|---|---|
| swiglu_output__f16+gemm2_accumulator__bf16 | 0.856226 | 0.750558 | nan | contest_unsafe | strict_unsafe | oracle_only |
| gemm1_accumulator__f16+swiglu_output__f16 | 0.856844 | 0.841598 | nan | contest_unsafe | strict_unsafe | oracle_only |
| swiglu_output__f16+out_accumulator__bf16 | 0.857083 | 0.850885 | nan | contest_unsafe | strict_unsafe | oracle_only |
| hidden_dequant__f16+swiglu_output__f16 | 0.857103 | 0.853934 | nan | contest_unsafe | strict_unsafe | oracle_only |
| gemm1_output__f16+swiglu_output__f16 | 0.857103 | 0.854333 | nan | contest_unsafe | strict_unsafe | oracle_only |
| gemm1_operands__f16+swiglu_output__f16 | 0.857203 | 0.852778 | nan | contest_unsafe | strict_unsafe | oracle_only |
| gemm1_accumulator__f16+gemm2_accumulator__bf16 | 0.990513 | 0.738002 | 3.1744e+05 | contest_safe | strict_unsafe | oracle_only |
| gemm1_output__f16+gemm2_accumulator__bf16 | 0.991908 | 0.752232 | 1.7408e+05 | contest_safe | strict_unsafe | oracle_only |
| gemm2_accumulator__bf16+out_accumulator__bf16 | 0.992188 | 0.744420 | 1.9388e+05 | contest_safe | strict_unsafe | oracle_only |
| gemm1_operands__f16+gemm2_accumulator__bf16 | 0.992467 | 0.748326 | 1.5770e+05 | contest_safe | strict_unsafe | oracle_only |
| hidden_dequant__f16+gemm2_accumulator__bf16 | 0.993025 | 0.748326 | 1.5770e+05 | contest_safe | strict_unsafe | oracle_only |
| hidden_dequant__f16+gemm1_accumulator__f16 | 0.996094 | 0.898856 | 1.5565e+05 | contest_safe | strict_unsafe | oracle_only |
| gemm1_accumulator__f16+out_accumulator__bf16 | 0.996791 | 0.876674 | 1.5974e+05 | contest_safe | strict_unsafe | oracle_only |
| gemm1_accumulator__f16+gemm1_output__f16 | 0.996931 | 0.893973 | 1.6282e+05 | contest_safe | strict_unsafe | oracle_only |
| gemm1_output__f16+out_accumulator__bf16 | 0.997628 | 0.930106 | 5.1542e+04 | contest_safe | strict_safe | oracle_only |
| gemm1_operands__f16+out_accumulator__bf16 | 0.997768 | 0.926618 | 5.4273e+04 | contest_safe | strict_safe | oracle_only |
| hidden_dequant__f16+out_accumulator__bf16 | 0.997907 | 0.931083 | 3.2596e+04 | contest_safe | strict_safe | oracle_only |
| gemm1_operands__f16+gemm1_output__f16 | 0.998465 | 0.964007 | 1.8945e+04 | contest_safe | strict_safe | oracle_only |
| hidden_dequant__f16+gemm1_output__f16 | 0.998884 | 0.974051 | 1.9028e+04 | contest_safe | strict_safe | oracle_only |
| hidden_dequant__f16+gemm1_operands__f16 | 0.999023 | 0.973912 | 2.2271e+04 | contest_safe | strict_safe | oracle_only |
| gemm1_operands__f16+gemm1_accumulator__f16 | 0.999023 | 0.973912 | 2.2271e+04 | contest_safe | strict_safe | oracle_only |

## Stress Summary

| candidate | worst_workload | worst_seq_len | worst_matched_contest | worst_matched_strict | contest_status | strict_status | evidence |
|---|---|---:|---:|---:|---|---|---|
| gemm2_accumulator__bf16 | e05c6c03 | 1 | 0.991908 | 0.747210 | contest_safe | strict_unsafe | oracle_only |

## Promotion Summary

| category | candidates |
|---|---|
| contest_safe_oracle_candidates | gemm1_output__f16, swiglu_input__f16, gemm1_operands__f16, hidden_dequant__f16, out_accumulator__bf16, swiglu_output__bf16, gemm1_accumulator__f16, hidden_dequant__bf16, gemm1_output__bf16, swiglu_input__bf16, gemm2_operands__bf16, gemm1_operands__bf16, gemm2_accumulator__bf16, gemm1_accumulator__bf16, swiglu_output__f16, gemm2_operands__f16 |
| strict_safe_oracle_candidates | hidden_dequant__f16, gemm1_operands__f16, gemm1_output__f16, swiglu_input__f16, swiglu_output__f16, gemm2_operands__f16, out_accumulator__bf16 |
| kernel_validated_candidates | - |
| pairwise_shortlist | hidden_dequant__f16, gemm1_operands__f16, gemm1_accumulator__f16, gemm1_output__f16, swiglu_output__f16, gemm2_accumulator__bf16, out_accumulator__bf16 |

## Kernel Validation Summary

| candidate | validation_status | workloads_passed | total_workloads | worst_max_abs | worst_max_rel | evidence | notes |
|---|---|---:|---:|---:|---:|---|---|

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
| baseline__fp32 | single_stage | 8f1ff9f1 | 80 | 1.000000 | 1.000000 | 0.0000e+00 | 0.0000e+00 | pass |
| hidden_dequant__bf16 | single_stage | 8f1ff9f1 | 80 | 0.997013 | 0.908437 | 4.0960e+03 | 3.5552e+02 | pass |
| hidden_dequant__f16 | single_stage | 8f1ff9f1 | 80 | 0.999651 | 0.988102 | 2.0480e+03 | 3.5635e+01 | pass |
| gemm1_operands__bf16 | single_stage | 8f1ff9f1 | 80 | 0.995792 | 0.874030 | 4.0960e+03 | 5.2440e+02 | pass |
| gemm1_operands__f16 | single_stage | 8f1ff9f1 | 80 | 0.999472 | 0.983332 | 4.0960e+03 | 2.5974e+01 | pass |
| gemm1_accumulator__bf16 | single_stage | 8f1ff9f1 | 80 | 0.984487 | 0.662927 | 8.1920e+03 | 1.5859e+03 | pass |
| gemm1_accumulator__f16 | single_stage | 8f1ff9f1 | 80 | 0.998003 | 0.938600 | 2.0480e+03 | 1.8128e+02 | pass |
| gemm1_output__bf16 | single_stage | 8f1ff9f1 | 80 | 0.997002 | 0.908254 | 2.0480e+03 | 1.2775e+02 | pass |
| gemm1_output__f16 | single_stage | 8f1ff9f1 | 80 | 0.999629 | 0.988318 | 2.0480e+03 | 5.1262e+01 | pass |
| swiglu_input__bf16 | single_stage | 8f1ff9f1 | 80 | 0.997002 | 0.908254 | 2.0480e+03 | 1.2775e+02 | pass |
| swiglu_input__f16 | single_stage | 8f1ff9f1 | 80 | 0.999629 | 0.988318 | 2.0480e+03 | 5.1262e+01 | pass |
| swiglu_output__bf16 | single_stage | 8f1ff9f1 | 80 | 0.997845 | 0.933515 | 4.0960e+03 | 1.4912e+02 | pass |
| swiglu_output__f16 | single_stage | 8f1ff9f1 | 80 | 0.918558 | 0.911602 | nan | nan | pass |
| gemm2_operands__bf16 | single_stage | 8f1ff9f1 | 80 | 0.996957 | 0.908404 | 4.0960e+03 | 1.2910e+02 | pass |
| gemm2_operands__f16 | single_stage | 8f1ff9f1 | 80 | 0.918485 | 0.908690 | nan | nan | pass |
| gemm2_accumulator__bf16 | single_stage | 8f1ff9f1 | 80 | 0.995694 | 0.861855 | 6.1440e+03 | 3.7429e+02 | pass |
| gemm2_accumulator__f16 | single_stage | 8f1ff9f1 | 80 | 0.676358 | 0.675863 | nan | nan | global_drift |
| out_accumulator__bf16 | single_stage | 8f1ff9f1 | 80 | 0.999187 | 0.974205 | 4.0960e+03 | 5.3891e+01 | pass |
| out_accumulator__f16 | single_stage | 8f1ff9f1 | 80 | 0.705186 | 0.703430 | inf | inf | catastrophic_outlier |
| baseline__fp32 | single_stage | 1a4c6ba1 | 901 | 1.000000 | 1.000000 | 0.0000e+00 | 0.0000e+00 | pass |
| hidden_dequant__bf16 | single_stage | 1a4c6ba1 | 901 | 0.996848 | 0.904143 | 4.0960e+03 | 2.6839e+03 | pass |
| hidden_dequant__f16 | single_stage | 1a4c6ba1 | 901 | 0.999616 | 0.987766 | 4.0960e+03 | 2.2440e+02 | pass |
| gemm1_operands__bf16 | single_stage | 1a4c6ba1 | 901 | 0.995560 | 0.866779 | 4.0960e+03 | 5.8992e+03 | pass |
| gemm1_operands__f16 | single_stage | 1a4c6ba1 | 901 | 0.999447 | 0.982585 | 4.0960e+03 | 4.7516e+02 | pass |
| gemm1_accumulator__bf16 | single_stage | 1a4c6ba1 | 901 | 0.983707 | 0.641867 | 1.2288e+04 | 2.0713e+04 | pass |
| gemm1_accumulator__f16 | single_stage | 1a4c6ba1 | 901 | 0.997890 | 0.935232 | 4.0960e+03 | 1.5452e+03 | pass |
| gemm1_output__bf16 | single_stage | 1a4c6ba1 | 901 | 0.996843 | 0.904212 | 4.0960e+03 | 2.7248e+03 | pass |
| gemm1_output__f16 | single_stage | 1a4c6ba1 | 901 | 0.999607 | 0.987776 | 4.0960e+03 | 5.9748e+02 | pass |
| swiglu_input__bf16 | single_stage | 1a4c6ba1 | 901 | 0.996843 | 0.904212 | 4.0960e+03 | 2.7248e+03 | pass |
| swiglu_input__f16 | single_stage | 1a4c6ba1 | 901 | 0.999607 | 0.987776 | 4.0960e+03 | 5.9748e+02 | pass |
| swiglu_output__bf16 | single_stage | 1a4c6ba1 | 901 | 0.997788 | 0.931699 | 4.0960e+03 | 2.8068e+03 | pass |
| swiglu_output__f16 | single_stage | 1a4c6ba1 | 901 | 0.940720 | 0.933261 | nan | nan | pass |
| gemm2_operands__bf16 | single_stage | 1a4c6ba1 | 901 | 0.996878 | 0.904725 | 4.0960e+03 | 2.7863e+03 | pass |
| gemm2_operands__f16 | single_stage | 1a4c6ba1 | 901 | 0.940615 | 0.930149 | nan | nan | pass |
| gemm2_accumulator__bf16 | single_stage | 1a4c6ba1 | 901 | 0.995520 | 0.853443 | 1.6384e+04 | 2.8252e+03 | pass |
| gemm2_accumulator__f16 | single_stage | 1a4c6ba1 | 901 | 0.679173 | 0.678883 | nan | nan | global_drift |
| out_accumulator__bf16 | single_stage | 1a4c6ba1 | 901 | 0.999084 | 0.971159 | 4.0960e+03 | 8.8281e+02 | pass |
| out_accumulator__f16 | single_stage | 1a4c6ba1 | 901 | 0.685288 | 0.683195 | inf | inf | catastrophic_outlier |
| baseline__fp32 | single_stage | a7c2bcfd | 16 | 1.000000 | 1.000000 | 0.0000e+00 | 0.0000e+00 | pass |
| hidden_dequant__bf16 | single_stage | a7c2bcfd | 16 | 0.997114 | 0.906110 | 2.0480e+03 | 3.4816e+01 | pass |
| hidden_dequant__f16 | single_stage | a7c2bcfd | 16 | 0.999704 | 0.988098 | 2.0480e+03 | 5.7349e+00 | pass |
| gemm1_operands__bf16 | single_stage | a7c2bcfd | 16 | 0.995780 | 0.868731 | 2.0480e+03 | 4.1240e+01 | pass |
| gemm1_operands__f16 | single_stage | a7c2bcfd | 16 | 0.999651 | 0.983329 | 2.0480e+03 | 1.6563e+01 | pass |
| gemm1_accumulator__bf16 | single_stage | a7c2bcfd | 16 | 0.984270 | 0.645700 | 6.1440e+03 | 5.7053e+02 | pass |
| gemm1_accumulator__f16 | single_stage | a7c2bcfd | 16 | 0.997977 | 0.935268 | 2.0480e+03 | 6.7679e+01 | pass |
| gemm1_output__bf16 | single_stage | a7c2bcfd | 16 | 0.997062 | 0.908325 | 2.0480e+03 | 3.3595e+01 | pass |
| gemm1_output__f16 | single_stage | a7c2bcfd | 16 | 0.999686 | 0.988717 | 2.0480e+03 | 6.2624e+00 | pass |
| swiglu_input__bf16 | single_stage | a7c2bcfd | 16 | 0.997062 | 0.908325 | 2.0480e+03 | 3.3595e+01 | pass |
| swiglu_input__f16 | single_stage | a7c2bcfd | 16 | 0.999686 | 0.988717 | 2.0480e+03 | 6.2624e+00 | pass |
| swiglu_output__bf16 | single_stage | a7c2bcfd | 16 | 0.997968 | 0.930603 | 2.0480e+03 | 5.7940e+01 | pass |
| swiglu_output__f16 | single_stage | a7c2bcfd | 16 | 0.937369 | 0.930028 | nan | nan | pass |
| gemm2_operands__bf16 | single_stage | a7c2bcfd | 16 | 0.996878 | 0.905361 | 2.0480e+03 | 3.6901e+01 | pass |
| gemm2_operands__f16 | single_stage | a7c2bcfd | 16 | 0.937334 | 0.926740 | nan | nan | pass |
| gemm2_accumulator__bf16 | single_stage | a7c2bcfd | 16 | 0.996320 | 0.859157 | 4.0960e+03 | 6.3000e+01 | pass |
| gemm2_accumulator__f16 | single_stage | a7c2bcfd | 16 | 0.595494 | 0.594918 | nan | nan | global_drift |
| out_accumulator__bf16 | single_stage | a7c2bcfd | 16 | 0.999442 | 0.979440 | 2.0480e+03 | 1.0526e+01 | pass |
| out_accumulator__f16 | single_stage | a7c2bcfd | 16 | 0.772391 | 0.770979 | inf | inf | catastrophic_outlier |
| baseline__fp32 | single_stage | 2e69caee | 15 | 1.000000 | 1.000000 | 0.0000e+00 | 0.0000e+00 | pass |
| hidden_dequant__bf16 | single_stage | 2e69caee | 15 | 0.998679 | 0.958026 | 2.0480e+03 | 1.2103e+02 | pass |
| hidden_dequant__f16 | single_stage | 2e69caee | 15 | 0.999851 | 0.994829 | 2.0480e+03 | 1.7203e+01 | pass |
| gemm1_operands__bf16 | single_stage | 2e69caee | 15 | 0.998075 | 0.941416 | 2.0480e+03 | 1.3785e+02 | pass |
| gemm1_operands__f16 | single_stage | 2e69caee | 15 | 0.999786 | 0.992587 | 2.0480e+03 | 3.5091e+00 | pass |
| gemm1_accumulator__bf16 | single_stage | 2e69caee | 15 | 0.993517 | 0.849628 | 6.1440e+03 | 7.2394e+01 | pass |
| gemm1_accumulator__f16 | single_stage | 2e69caee | 15 | 0.999135 | 0.973531 | 2.0480e+03 | 7.7644e+01 | pass |
| gemm1_output__bf16 | single_stage | 2e69caee | 15 | 0.998735 | 0.959161 | 2.0480e+03 | 1.9734e+02 | pass |
| gemm1_output__f16 | single_stage | 2e69caee | 15 | 0.999860 | 0.995099 | 2.0480e+03 | 2.3237e+01 | pass |
| swiglu_input__bf16 | single_stage | 2e69caee | 15 | 0.998735 | 0.959161 | 2.0480e+03 | 1.9734e+02 | pass |
| swiglu_input__f16 | single_stage | 2e69caee | 15 | 0.999860 | 0.995099 | 2.0480e+03 | 2.3237e+01 | pass |
| swiglu_output__bf16 | single_stage | 2e69caee | 15 | 0.999163 | 0.971652 | 2.0480e+03 | 5.1068e+01 | pass |
| swiglu_output__f16 | single_stage | 2e69caee | 15 | 0.999879 | 0.996484 | 2.0480e+03 | 2.2712e+00 | pass |
| gemm2_operands__bf16 | single_stage | 2e69caee | 15 | 0.998772 | 0.959328 | 2.0480e+03 | 7.5475e+01 | pass |
| gemm2_operands__f16 | single_stage | 2e69caee | 15 | 0.999870 | 0.995136 | 2.0480e+03 | 2.8136e+00 | pass |
| gemm2_accumulator__bf16 | single_stage | 2e69caee | 15 | 0.998177 | 0.938318 | 4.0960e+03 | 4.5102e+01 | pass |
| gemm2_accumulator__f16 | single_stage | 2e69caee | 15 | 0.868052 | 0.867997 | nan | nan | global_drift |
| out_accumulator__bf16 | single_stage | 2e69caee | 15 | 0.999674 | 0.988477 | 2.0480e+03 | 3.3983e+01 | pass |
| out_accumulator__f16 | single_stage | 2e69caee | 15 | 0.854529 | 0.853813 | inf | inf | catastrophic_outlier |
| baseline__fp32 | single_stage | 8cba5890 | 14 | 1.000000 | 1.000000 | 0.0000e+00 | 0.0000e+00 | pass |
| hidden_dequant__bf16 | single_stage | 8cba5890 | 14 | 0.997758 | 0.930624 | 2.0480e+03 | 4.7222e+01 | pass |
| hidden_dequant__f16 | single_stage | 8cba5890 | 14 | 0.999781 | 0.991879 | 2.0480e+03 | 5.1637e+00 | pass |
| gemm1_operands__bf16 | single_stage | 8cba5890 | 14 | 0.996861 | 0.904476 | 2.0480e+03 | 8.3051e+01 | pass |
| gemm1_operands__f16 | single_stage | 8cba5890 | 14 | 0.999711 | 0.988530 | 2.0480e+03 | 1.2602e+01 | pass |
| gemm1_accumulator__bf16 | single_stage | 8cba5890 | 14 | 0.989347 | 0.756128 | 6.1440e+03 | 2.7197e+02 | pass |
| gemm1_accumulator__f16 | single_stage | 8cba5890 | 14 | 0.998655 | 0.956304 | 2.0480e+03 | 2.4026e+01 | pass |
| gemm1_output__bf16 | single_stage | 8cba5890 | 14 | 0.997927 | 0.937849 | 2.0480e+03 | 3.5865e+01 | pass |
| gemm1_output__f16 | single_stage | 8cba5890 | 14 | 0.999711 | 0.992377 | 2.0480e+03 | 5.2906e+00 | pass |
| swiglu_input__bf16 | single_stage | 8cba5890 | 14 | 0.997927 | 0.937849 | 2.0480e+03 | 3.5865e+01 | pass |
| swiglu_input__f16 | single_stage | 8cba5890 | 14 | 0.999711 | 0.992377 | 2.0480e+03 | 5.2906e+00 | pass |
| swiglu_output__bf16 | single_stage | 8cba5890 | 14 | 0.998565 | 0.954022 | 2.0480e+03 | 1.6404e+01 | pass |
| swiglu_output__f16 | single_stage | 8cba5890 | 14 | 0.999781 | 0.994250 | 2.0480e+03 | 7.2807e+00 | pass |
| gemm2_operands__bf16 | single_stage | 8cba5890 | 14 | 0.998107 | 0.935567 | 2.0480e+03 | 3.4928e+01 | pass |
| gemm2_operands__f16 | single_stage | 8cba5890 | 14 | 0.999721 | 0.992178 | 2.0480e+03 | 8.4035e+00 | pass |
| gemm2_accumulator__bf16 | single_stage | 8cba5890 | 14 | 0.996901 | 0.901666 | 4.0960e+03 | 6.2626e+01 | pass |
| gemm2_accumulator__f16 | single_stage | 8cba5890 | 14 | 0.803083 | 0.803013 | nan | nan | global_drift |
| out_accumulator__bf16 | single_stage | 8cba5890 | 14 | 0.999372 | 0.979452 | 2.0480e+03 | 1.3017e+01 | pass |
| out_accumulator__f16 | single_stage | 8cba5890 | 14 | 0.773687 | 0.771963 | inf | inf | catastrophic_outlier |
| baseline__fp32 | single_stage | 5e8dc11c | 14107 | 1.000000 | 1.000000 | 0.0000e+00 | 0.0000e+00 | pass |
| hidden_dequant__bf16 | single_stage | 5e8dc11c | 14107 | 0.997167 | 0.912999 | 4.0960e+03 | 1.3300e+10 | pass |
| hidden_dequant__f16 | single_stage | 5e8dc11c | 14107 | 0.999645 | 0.988876 | 4.0960e+03 | 2.3250e+09 | pass |
| gemm1_operands__bf16 | single_stage | 5e8dc11c | 14107 | 0.995988 | 0.879246 | 4.0960e+03 | 1.6800e+10 | pass |
| gemm1_operands__f16 | single_stage | 5e8dc11c | 14107 | 0.999499 | 0.984227 | 4.0960e+03 | 4.2500e+09 | pass |
| gemm1_accumulator__bf16 | single_stage | 5e8dc11c | 14107 | 0.985253 | 0.675658 | 1.7024e+04 | 1.7800e+10 | pass |
| gemm1_accumulator__f16 | single_stage | 5e8dc11c | 14107 | 0.998125 | 0.941616 | 4.0960e+03 | 4.5750e+09 | pass |
| gemm1_output__bf16 | single_stage | 5e8dc11c | 14107 | 0.997176 | 0.913283 | 4.0960e+03 | 3.6500e+09 | pass |
| gemm1_output__f16 | single_stage | 5e8dc11c | 14107 | 0.999649 | 0.988922 | 4.0960e+03 | 2.8875e+09 | pass |
| swiglu_input__bf16 | single_stage | 5e8dc11c | 14107 | 0.997176 | 0.913283 | 4.0960e+03 | 3.6500e+09 | pass |
| swiglu_input__f16 | single_stage | 5e8dc11c | 14107 | 0.999649 | 0.988922 | 4.0960e+03 | 2.8875e+09 | pass |
| swiglu_output__bf16 | single_stage | 5e8dc11c | 14107 | 0.998004 | 0.937990 | 4.0960e+03 | 2.2300e+10 | pass |
| swiglu_output__f16 | single_stage | 5e8dc11c | 14107 | 0.957047 | 0.950101 | nan | nan | pass |
| gemm2_operands__bf16 | single_stage | 5e8dc11c | 14107 | 0.997196 | 0.913683 | 4.0960e+03 | 2.8200e+10 | pass |
| gemm2_operands__f16 | single_stage | 5e8dc11c | 14107 | 0.956954 | 0.947171 | nan | nan | pass |
| gemm2_accumulator__bf16 | single_stage | 5e8dc11c | 14107 | 0.996187 | 0.871506 | 1.2288e+04 | 1.1920e+11 | pass |
| gemm2_accumulator__f16 | single_stage | 5e8dc11c | 14107 | 0.654792 | 0.654404 | nan | nan | global_drift |
| out_accumulator__bf16 | single_stage | 5e8dc11c | 14107 | 0.999367 | 0.979593 | 8.1920e+03 | 2.5000e+10 | pass |
| out_accumulator__f16 | single_stage | 5e8dc11c | 14107 | 0.747794 | 0.746215 | inf | inf | catastrophic_outlier |
| baseline__fp32 | single_stage | 58a34f27 | 11948 | 1.000000 | 1.000000 | 0.0000e+00 | 0.0000e+00 | pass |
| hidden_dequant__bf16 | single_stage | 58a34f27 | 11948 | 0.997833 | 0.933396 | 4.0960e+03 | 1.8477e+05 | pass |
| hidden_dequant__f16 | single_stage | 58a34f27 | 11948 | 0.999727 | 0.991449 | 4.0960e+03 | 1.2197e+04 | pass |
| gemm1_operands__bf16 | single_stage | 58a34f27 | 11948 | 0.996918 | 0.907440 | 8.1920e+03 | 2.1017e+05 | pass |
| gemm1_operands__f16 | single_stage | 58a34f27 | 11948 | 0.999612 | 0.987917 | 4.0960e+03 | 2.4683e+04 | pass |
| gemm1_accumulator__bf16 | single_stage | 58a34f27 | 11948 | 0.988714 | 0.751771 | 1.5360e+04 | 9.6542e+05 | pass |
| gemm1_accumulator__f16 | single_stage | 58a34f27 | 11948 | 0.998554 | 0.955301 | 4.0960e+03 | 3.7243e+04 | pass |
| gemm1_output__bf16 | single_stage | 58a34f27 | 11948 | 0.997831 | 0.933506 | 8.1920e+03 | 8.8919e+04 | pass |
| gemm1_output__f16 | single_stage | 58a34f27 | 11948 | 0.999728 | 0.991498 | 4.0960e+03 | 9.0614e+03 | pass |
| swiglu_input__bf16 | single_stage | 58a34f27 | 11948 | 0.997831 | 0.933506 | 8.1920e+03 | 8.8919e+04 | pass |
| swiglu_input__f16 | single_stage | 58a34f27 | 11948 | 0.999728 | 0.991498 | 4.0960e+03 | 9.0614e+03 | pass |
| swiglu_output__bf16 | single_stage | 58a34f27 | 11948 | 0.998472 | 0.952457 | 4.0960e+03 | 1.3222e+05 | pass |
| swiglu_output__f16 | single_stage | 58a34f27 | 11948 | 0.964211 | 0.958905 | nan | nan | pass |
| gemm2_operands__bf16 | single_stage | 58a34f27 | 11948 | 0.997837 | 0.933808 | 4.0960e+03 | 1.1375e+05 | pass |
| gemm2_operands__f16 | single_stage | 58a34f27 | 11948 | 0.964141 | 0.956687 | nan | nan | pass |
| gemm2_accumulator__bf16 | single_stage | 58a34f27 | 11948 | 0.997064 | 0.900904 | 8.1920e+03 | 2.8263e+04 | pass |
| gemm2_accumulator__f16 | single_stage | 58a34f27 | 11948 | 0.734831 | 0.734559 | nan | nan | global_drift |
| out_accumulator__bf16 | single_stage | 58a34f27 | 11948 | 0.999511 | 0.984267 | 8.1920e+03 | 1.2082e+04 | pass |
| out_accumulator__f16 | single_stage | 58a34f27 | 11948 | 0.798905 | 0.797715 | inf | inf | catastrophic_outlier |
| baseline__fp32 | single_stage | 5eadab1e | 62 | 1.000000 | 1.000000 | 0.0000e+00 | 0.0000e+00 | pass |
| hidden_dequant__bf16 | single_stage | 5eadab1e | 62 | 0.997633 | 0.928828 | 4.0960e+03 | 2.7297e+03 | pass |
| hidden_dequant__f16 | single_stage | 5eadab1e | 62 | 0.999712 | 0.990729 | 2.0480e+03 | 4.5967e+02 | pass |
| gemm1_operands__bf16 | single_stage | 5eadab1e | 62 | 0.996710 | 0.901599 | 4.0960e+03 | 2.7012e+03 | pass |
| gemm1_operands__f16 | single_stage | 5eadab1e | 62 | 0.999559 | 0.987104 | 2.0480e+03 | 3.7233e+02 | pass |
| gemm1_accumulator__bf16 | single_stage | 5eadab1e | 62 | 0.987881 | 0.735579 | 8.1920e+03 | 1.1206e+04 | pass |
| gemm1_accumulator__f16 | single_stage | 5eadab1e | 62 | 0.998389 | 0.951010 | 4.0960e+03 | 1.3163e+03 | pass |
| gemm1_output__bf16 | single_stage | 5eadab1e | 62 | 0.997613 | 0.929282 | 2.0480e+03 | 1.4997e+03 | pass |
| gemm1_output__f16 | single_stage | 5eadab1e | 62 | 0.999696 | 0.990909 | 2.0480e+03 | 2.2122e+02 | pass |
| swiglu_input__bf16 | single_stage | 5eadab1e | 62 | 0.997613 | 0.929282 | 2.0480e+03 | 1.4997e+03 | pass |
| swiglu_input__f16 | single_stage | 5eadab1e | 62 | 0.999696 | 0.990909 | 2.0480e+03 | 2.2122e+02 | pass |
| swiglu_output__bf16 | single_stage | 5eadab1e | 62 | 0.998276 | 0.949063 | 2.0480e+03 | 1.0819e+03 | pass |
| swiglu_output__f16 | single_stage | 5eadab1e | 62 | 0.975608 | 0.969731 | nan | nan | pass |
| gemm2_operands__bf16 | single_stage | 5eadab1e | 62 | 0.997759 | 0.929753 | 4.0960e+03 | 3.5850e+03 | pass |
| gemm2_operands__f16 | single_stage | 5eadab1e | 62 | 0.975521 | 0.967303 | nan | nan | pass |
| gemm2_accumulator__bf16 | single_stage | 5eadab1e | 62 | 0.996944 | 0.896815 | 8.1920e+03 | 2.6017e+03 | pass |
| gemm2_accumulator__f16 | single_stage | 5eadab1e | 62 | 0.707578 | 0.707229 | nan | nan | global_drift |
| out_accumulator__bf16 | single_stage | 5eadab1e | 62 | 0.999525 | 0.984265 | 4.0960e+03 | 1.3359e+03 | pass |
| out_accumulator__f16 | single_stage | 5eadab1e | 62 | 0.796470 | 0.795417 | inf | inf | catastrophic_outlier |
| baseline__fp32 | single_stage | eedc63b2 | 59 | 1.000000 | 1.000000 | 0.0000e+00 | 0.0000e+00 | pass |
| hidden_dequant__bf16 | single_stage | eedc63b2 | 59 | 0.998033 | 0.940572 | 4.0960e+03 | 3.0695e+02 | pass |
| hidden_dequant__f16 | single_stage | eedc63b2 | 59 | 0.999738 | 0.992334 | 4.0960e+03 | 1.7767e+01 | pass |
| gemm1_operands__bf16 | single_stage | eedc63b2 | 59 | 0.997212 | 0.917054 | 4.0960e+03 | 2.2990e+02 | pass |
| gemm1_operands__f16 | single_stage | eedc63b2 | 59 | 0.999622 | 0.988932 | 4.0960e+03 | 2.1611e+01 | pass |
| gemm1_accumulator__bf16 | single_stage | eedc63b2 | 59 | 0.989920 | 0.778935 | 1.0240e+04 | 9.0811e+02 | pass |
| gemm1_accumulator__f16 | single_stage | eedc63b2 | 59 | 0.998662 | 0.959505 | 4.0960e+03 | 1.8730e+02 | pass |
| gemm1_output__bf16 | single_stage | eedc63b2 | 59 | 0.998011 | 0.939380 | 4.0960e+03 | 2.9265e+02 | pass |
| gemm1_output__f16 | single_stage | eedc63b2 | 59 | 0.999723 | 0.992355 | 4.0960e+03 | 3.7905e+01 | pass |
| swiglu_input__bf16 | single_stage | eedc63b2 | 59 | 0.998011 | 0.939380 | 4.0960e+03 | 2.9265e+02 | pass |
| swiglu_input__f16 | single_stage | eedc63b2 | 59 | 0.999723 | 0.992355 | 4.0960e+03 | 3.7905e+01 | pass |
| swiglu_output__bf16 | single_stage | eedc63b2 | 59 | 0.998617 | 0.957577 | 4.0960e+03 | 2.1129e+02 | pass |
| swiglu_output__f16 | single_stage | eedc63b2 | 59 | 0.965946 | 0.961278 | nan | nan | pass |
| gemm2_operands__bf16 | single_stage | eedc63b2 | 59 | 0.998189 | 0.940948 | 4.0960e+03 | 1.6384e+02 | pass |
| gemm2_operands__f16 | single_stage | eedc63b2 | 59 | 0.965905 | 0.959528 | nan | nan | pass |
| gemm2_accumulator__bf16 | single_stage | eedc63b2 | 59 | 0.997583 | 0.914715 | 6.1440e+03 | 2.5750e+02 | pass |
| gemm2_accumulator__f16 | single_stage | eedc63b2 | 59 | 0.744824 | 0.744533 | nan | nan | global_drift |
| out_accumulator__bf16 | single_stage | eedc63b2 | 59 | 0.999638 | 0.988040 | 4.0960e+03 | 5.2200e+01 | pass |
| out_accumulator__f16 | single_stage | eedc63b2 | 59 | 0.830946 | 0.830002 | inf | inf | catastrophic_outlier |
| baseline__fp32 | single_stage | e626d3e6 | 58 | 1.000000 | 1.000000 | 0.0000e+00 | 0.0000e+00 | pass |
| hidden_dequant__bf16 | single_stage | e626d3e6 | 58 | 0.997265 | 0.911482 | 4.0960e+03 | 5.1822e+02 | pass |
| hidden_dequant__f16 | single_stage | e626d3e6 | 58 | 0.999615 | 0.988892 | 4.0960e+03 | 1.0401e+02 | pass |
| gemm1_operands__bf16 | single_stage | e626d3e6 | 58 | 0.996075 | 0.876378 | 4.0960e+03 | 6.5929e+02 | pass |
| gemm1_operands__f16 | single_stage | e626d3e6 | 58 | 0.999490 | 0.984041 | 4.0960e+03 | 1.1519e+02 | pass |
| gemm1_accumulator__bf16 | single_stage | e626d3e6 | 58 | 0.984560 | 0.664281 | 9.4720e+03 | 1.5474e+03 | pass |
| gemm1_accumulator__f16 | single_stage | e626d3e6 | 58 | 0.998093 | 0.940235 | 4.0960e+03 | 5.8756e+02 | pass |
| gemm1_output__bf16 | single_stage | e626d3e6 | 58 | 0.997205 | 0.911556 | 4.0960e+03 | 3.9862e+02 | pass |
| gemm1_output__f16 | single_stage | e626d3e6 | 58 | 0.999656 | 0.988587 | 4.0960e+03 | 1.9114e+02 | pass |
| swiglu_input__bf16 | single_stage | e626d3e6 | 58 | 0.997205 | 0.911556 | 4.0960e+03 | 3.9862e+02 | pass |
| swiglu_input__f16 | single_stage | e626d3e6 | 58 | 0.999656 | 0.988587 | 4.0960e+03 | 1.9114e+02 | pass |
| swiglu_output__bf16 | single_stage | e626d3e6 | 58 | 0.997982 | 0.936540 | 4.0960e+03 | 4.8104e+02 | pass |
| swiglu_output__f16 | single_stage | e626d3e6 | 58 | 0.948076 | 0.941245 | nan | nan | pass |
| gemm2_operands__bf16 | single_stage | e626d3e6 | 58 | 0.997044 | 0.911258 | 4.0960e+03 | 3.9121e+02 | pass |
| gemm2_operands__f16 | single_stage | e626d3e6 | 58 | 0.947999 | 0.938339 | nan | nan | pass |
| gemm2_accumulator__bf16 | single_stage | e626d3e6 | 58 | 0.995767 | 0.861097 | 1.2288e+04 | 1.1791e+03 | pass |
| gemm2_accumulator__f16 | single_stage | e626d3e6 | 58 | 0.692849 | 0.692614 | nan | nan | global_drift |
| out_accumulator__bf16 | single_stage | e626d3e6 | 58 | 0.999247 | 0.973926 | 4.0960e+03 | 2.1279e+02 | pass |
| out_accumulator__f16 | single_stage | e626d3e6 | 58 | 0.726161 | 0.724140 | inf | inf | catastrophic_outlier |
| baseline__fp32 | single_stage | 74d7ff04 | 57 | 1.000000 | 1.000000 | 0.0000e+00 | 0.0000e+00 | pass |
| hidden_dequant__bf16 | single_stage | 74d7ff04 | 57 | 0.997526 | 0.924829 | 2.0480e+03 | 3.0946e+03 | pass |
| hidden_dequant__f16 | single_stage | 74d7ff04 | 57 | 0.999716 | 0.990374 | 4.0960e+03 | 2.0749e+03 | pass |
| gemm1_operands__bf16 | single_stage | 74d7ff04 | 57 | 0.996595 | 0.896142 | 4.0960e+03 | 1.2306e+04 | pass |
| gemm1_operands__f16 | single_stage | 74d7ff04 | 57 | 0.999579 | 0.986592 | 4.0960e+03 | 1.5996e+03 | pass |
| gemm1_accumulator__bf16 | single_stage | 74d7ff04 | 57 | 0.986964 | 0.718033 | 8.1920e+03 | 1.6385e+04 | pass |
| gemm1_accumulator__f16 | single_stage | 74d7ff04 | 57 | 0.998270 | 0.949383 | 4.0960e+03 | 1.0094e+04 | pass |
| gemm1_output__bf16 | single_stage | 74d7ff04 | 57 | 0.997413 | 0.924854 | 4.0960e+03 | 7.5708e+03 | pass |
| gemm1_output__f16 | single_stage | 74d7ff04 | 57 | 0.999706 | 0.990322 | 4.0960e+03 | 1.0634e+02 | pass |
| swiglu_input__bf16 | single_stage | 74d7ff04 | 57 | 0.997413 | 0.924854 | 4.0960e+03 | 7.5708e+03 | pass |
| swiglu_input__f16 | single_stage | 74d7ff04 | 57 | 0.999706 | 0.990322 | 4.0960e+03 | 1.0634e+02 | pass |
| swiglu_output__bf16 | single_stage | 74d7ff04 | 57 | 0.998172 | 0.946573 | 4.0960e+03 | 1.2693e+03 | pass |
| swiglu_output__f16 | single_stage | 74d7ff04 | 57 | 0.929722 | 0.924139 | nan | nan | pass |
| gemm2_operands__bf16 | single_stage | 74d7ff04 | 57 | 0.997540 | 0.925502 | 4.0960e+03 | 1.3038e+03 | pass |
| gemm2_operands__f16 | single_stage | 74d7ff04 | 57 | 0.929651 | 0.921873 | nan | nan | pass |
| gemm2_accumulator__bf16 | single_stage | 74d7ff04 | 57 | 0.996613 | 0.888941 | 8.1920e+03 | 1.0014e+03 | pass |
| gemm2_accumulator__f16 | single_stage | 74d7ff04 | 57 | 0.715480 | 0.715199 | nan | nan | global_drift |
| out_accumulator__bf16 | single_stage | 74d7ff04 | 57 | 0.999344 | 0.981492 | 4.0960e+03 | 3.2755e+02 | pass |
| out_accumulator__f16 | single_stage | 74d7ff04 | 57 | 0.757274 | 0.756026 | inf | inf | catastrophic_outlier |
| baseline__fp32 | single_stage | 4822167c | 56 | 1.000000 | 1.000000 | 0.0000e+00 | 0.0000e+00 | pass |
| hidden_dequant__bf16 | single_stage | 4822167c | 56 | 0.997135 | 0.908101 | 2.0480e+03 | 4.5488e+02 | pass |
| hidden_dequant__f16 | single_stage | 4822167c | 56 | 0.999664 | 0.988289 | 2.0480e+03 | 3.9694e+01 | pass |
| gemm1_operands__bf16 | single_stage | 4822167c | 56 | 0.995780 | 0.872459 | 2.0480e+03 | 8.4461e+02 | pass |
| gemm1_operands__f16 | single_stage | 4822167c | 56 | 0.999524 | 0.983403 | 2.0480e+03 | 5.9449e+01 | pass |
| gemm1_accumulator__bf16 | single_stage | 4822167c | 56 | 0.984240 | 0.656711 | 9.2160e+03 | 9.1889e+02 | pass |
| gemm1_accumulator__f16 | single_stage | 4822167c | 56 | 0.998119 | 0.937662 | 4.0960e+03 | 6.2773e+02 | pass |
| gemm1_output__bf16 | single_stage | 4822167c | 56 | 0.997153 | 0.908400 | 2.0480e+03 | 2.0991e+02 | pass |
| gemm1_output__f16 | single_stage | 4822167c | 56 | 0.999659 | 0.988289 | 2.0480e+03 | 6.7245e+01 | pass |
| swiglu_input__bf16 | single_stage | 4822167c | 56 | 0.997153 | 0.908400 | 2.0480e+03 | 2.0991e+02 | pass |
| swiglu_input__f16 | single_stage | 4822167c | 56 | 0.999659 | 0.988289 | 2.0480e+03 | 6.7245e+01 | pass |
| swiglu_output__bf16 | single_stage | 4822167c | 56 | 0.997992 | 0.934812 | 4.0960e+03 | 2.5435e+02 | pass |
| swiglu_output__f16 | single_stage | 4822167c | 56 | 0.981929 | 0.974076 | nan | nan | pass |
| gemm2_operands__bf16 | single_stage | 4822167c | 56 | 0.997175 | 0.909155 | 4.0960e+03 | 3.5104e+02 | pass |
| gemm2_operands__f16 | single_stage | 4822167c | 56 | 0.981859 | 0.970902 | nan | nan | pass |
| gemm2_accumulator__bf16 | single_stage | 4822167c | 56 | 0.995957 | 0.861341 | 4.0960e+03 | 1.1817e+03 | pass |
| gemm2_accumulator__f16 | single_stage | 4822167c | 56 | 0.652371 | 0.652090 | nan | nan | global_drift |
| out_accumulator__bf16 | single_stage | 4822167c | 56 | 0.999315 | 0.975972 | 2.0480e+03 | 6.1606e+01 | pass |
| out_accumulator__f16 | single_stage | 4822167c | 56 | 0.734826 | 0.733012 | inf | inf | catastrophic_outlier |
| baseline__fp32 | single_stage | 81955b1e | 55 | 1.000000 | 1.000000 | 0.0000e+00 | 0.0000e+00 | pass |
| hidden_dequant__bf16 | single_stage | 81955b1e | 55 | 0.997623 | 0.927062 | 2.0480e+03 | 1.1646e+02 | pass |
| hidden_dequant__f16 | single_stage | 81955b1e | 55 | 0.999703 | 0.990813 | 2.0480e+03 | 8.7857e+01 | pass |
| gemm1_operands__bf16 | single_stage | 81955b1e | 55 | 0.996672 | 0.899074 | 2.0480e+03 | 5.1100e+02 | pass |
| gemm1_operands__f16 | single_stage | 81955b1e | 55 | 0.999581 | 0.986967 | 2.0480e+03 | 1.0081e+02 | pass |
| gemm1_accumulator__bf16 | single_stage | 81955b1e | 55 | 0.987477 | 0.729190 | 9.2160e+03 | 1.9880e+03 | pass |
| gemm1_accumulator__f16 | single_stage | 81955b1e | 55 | 0.998364 | 0.951124 | 2.0480e+03 | 5.9833e+02 | pass |
| gemm1_output__bf16 | single_stage | 81955b1e | 55 | 0.997567 | 0.928051 | 2.0480e+03 | 1.9757e+02 | pass |
| gemm1_output__f16 | single_stage | 81955b1e | 55 | 0.999685 | 0.990704 | 2.0480e+03 | 6.6048e+01 | pass |
| swiglu_input__bf16 | single_stage | 81955b1e | 55 | 0.997567 | 0.928051 | 2.0480e+03 | 1.9757e+02 | pass |
| swiglu_input__f16 | single_stage | 81955b1e | 55 | 0.999685 | 0.990704 | 2.0480e+03 | 6.6048e+01 | pass |
| swiglu_output__bf16 | single_stage | 81955b1e | 55 | 0.998326 | 0.949158 | 2.0480e+03 | 9.7687e+01 | pass |
| swiglu_output__f16 | single_stage | 81955b1e | 55 | 0.981623 | 0.975632 | nan | nan | pass |
| gemm2_operands__bf16 | single_stage | 81955b1e | 55 | 0.997608 | 0.928219 | 2.0480e+03 | 2.3184e+02 | pass |
| gemm2_operands__f16 | single_stage | 81955b1e | 55 | 0.981542 | 0.972869 | nan | nan | pass |
| gemm2_accumulator__bf16 | single_stage | 81955b1e | 55 | 0.996784 | 0.892814 | 4.0960e+03 | 2.9150e+02 | pass |
| gemm2_accumulator__f16 | single_stage | 81955b1e | 55 | 0.726256 | 0.725906 | nan | nan | global_drift |
| out_accumulator__bf16 | single_stage | 81955b1e | 55 | 0.999394 | 0.981940 | 2.0480e+03 | 6.3680e+01 | pass |
| out_accumulator__f16 | single_stage | 81955b1e | 55 | 0.784332 | 0.783023 | inf | inf | catastrophic_outlier |
| baseline__fp32 | single_stage | 76010cb4 | 54 | 1.000000 | 1.000000 | 0.0000e+00 | 0.0000e+00 | pass |
| hidden_dequant__bf16 | single_stage | 76010cb4 | 54 | 0.998070 | 0.941667 | 4.0960e+03 | 4.3536e+02 | pass |
| hidden_dequant__f16 | single_stage | 76010cb4 | 54 | 0.999786 | 0.992472 | 4.0960e+03 | 8.3770e+00 | pass |
| gemm1_operands__bf16 | single_stage | 76010cb4 | 54 | 0.997228 | 0.918687 | 4.0960e+03 | 3.4987e+02 | pass |
| gemm1_operands__f16 | single_stage | 76010cb4 | 54 | 0.999667 | 0.989286 | 4.0960e+03 | 9.1639e+00 | pass |
| gemm1_accumulator__bf16 | single_stage | 76010cb4 | 54 | 0.990033 | 0.781284 | 1.2288e+04 | 1.1677e+03 | pass |
| gemm1_accumulator__f16 | single_stage | 76010cb4 | 54 | 0.998786 | 0.961359 | 4.0960e+03 | 1.0796e+02 | pass |
| gemm1_output__bf16 | single_stage | 76010cb4 | 54 | 0.998047 | 0.940652 | 4.0960e+03 | 2.0034e+02 | pass |
| gemm1_output__f16 | single_stage | 76010cb4 | 54 | 0.999798 | 0.992461 | 2.0480e+03 | 5.2082e+01 | pass |
| swiglu_input__bf16 | single_stage | 76010cb4 | 54 | 0.998047 | 0.940652 | 4.0960e+03 | 2.0034e+02 | pass |
| swiglu_input__f16 | single_stage | 76010cb4 | 54 | 0.999798 | 0.992461 | 2.0480e+03 | 5.2082e+01 | pass |
| swiglu_output__bf16 | single_stage | 76010cb4 | 54 | 0.998675 | 0.957804 | 4.0960e+03 | 8.6273e+01 | pass |
| swiglu_output__f16 | single_stage | 76010cb4 | 54 | 0.976767 | 0.972090 | nan | nan | pass |
| gemm2_operands__bf16 | single_stage | 76010cb4 | 54 | 0.998070 | 0.941130 | 4.0960e+03 | 1.4393e+02 | pass |
| gemm2_operands__f16 | single_stage | 76010cb4 | 54 | 0.976715 | 0.970161 | nan | nan | pass |
| gemm2_accumulator__bf16 | single_stage | 76010cb4 | 54 | 0.997298 | 0.911991 | 6.1440e+03 | 1.8171e+02 | pass |
| gemm2_accumulator__f16 | single_stage | 76010cb4 | 54 | 0.775757 | 0.775530 | nan | nan | global_drift |
| out_accumulator__bf16 | single_stage | 76010cb4 | 54 | 0.999532 | 0.985277 | 4.0960e+03 | 6.1149e+01 | pass |
| out_accumulator__f16 | single_stage | 76010cb4 | 54 | 0.799562 | 0.798485 | inf | inf | catastrophic_outlier |
| baseline__fp32 | single_stage | fc378037 | 53 | 1.000000 | 1.000000 | 0.0000e+00 | 0.0000e+00 | pass |
| hidden_dequant__bf16 | single_stage | fc378037 | 53 | 0.997799 | 0.931080 | 4.0960e+03 | 1.3057e+02 | pass |
| hidden_dequant__f16 | single_stage | fc378037 | 53 | 0.999766 | 0.991227 | 2.0480e+03 | 1.1356e+01 | pass |
| gemm1_operands__bf16 | single_stage | fc378037 | 53 | 0.996839 | 0.904700 | 4.0960e+03 | 1.2040e+02 | pass |
| gemm1_operands__f16 | single_stage | fc378037 | 53 | 0.999642 | 0.987589 | 2.0480e+03 | 2.3157e+01 | pass |
| gemm1_accumulator__bf16 | single_stage | fc378037 | 53 | 0.988466 | 0.745272 | 8.1920e+03 | 5.9153e+02 | pass |
| gemm1_accumulator__f16 | single_stage | fc378037 | 53 | 0.998579 | 0.954662 | 2.0480e+03 | 8.1941e+01 | pass |
| gemm1_output__bf16 | single_stage | fc378037 | 53 | 0.997807 | 0.932501 | 4.0960e+03 | 2.0674e+02 | pass |
| gemm1_output__f16 | single_stage | fc378037 | 53 | 0.999732 | 0.991300 | 2.0480e+03 | 1.2350e+01 | pass |
| swiglu_input__bf16 | single_stage | fc378037 | 53 | 0.997807 | 0.932501 | 4.0960e+03 | 2.0674e+02 | pass |
| swiglu_input__f16 | single_stage | fc378037 | 53 | 0.999732 | 0.991300 | 2.0480e+03 | 1.2350e+01 | pass |
| swiglu_output__bf16 | single_stage | fc378037 | 53 | 0.998373 | 0.950911 | 2.0480e+03 | 5.8922e+01 | pass |
| swiglu_output__f16 | single_stage | fc378037 | 53 | 0.952775 | 0.947574 | nan | nan | pass |
| gemm2_operands__bf16 | single_stage | fc378037 | 53 | 0.997776 | 0.931338 | 2.0480e+03 | 6.6951e+01 | pass |
| gemm2_operands__f16 | single_stage | fc378037 | 53 | 0.952701 | 0.945344 | nan | nan | pass |
| gemm2_accumulator__bf16 | single_stage | fc378037 | 53 | 0.996852 | 0.897703 | 6.1440e+03 | 1.1217e+02 | pass |
| gemm2_accumulator__f16 | single_stage | fc378037 | 53 | 0.752382 | 0.752119 | nan | nan | global_drift |
| out_accumulator__bf16 | single_stage | fc378037 | 53 | 0.999458 | 0.981924 | 4.0960e+03 | 1.0692e+02 | pass |
| out_accumulator__f16 | single_stage | fc378037 | 53 | 0.777794 | 0.776544 | inf | inf | catastrophic_outlier |
| baseline__fp32 | single_stage | f7d6ac7c | 52 | 1.000000 | 1.000000 | 0.0000e+00 | 0.0000e+00 | pass |
| hidden_dequant__bf16 | single_stage | f7d6ac7c | 52 | 0.998586 | 0.956328 | 2.0480e+03 | 4.7222e+01 | pass |
| hidden_dequant__f16 | single_stage | f7d6ac7c | 52 | 0.999831 | 0.994374 | 2.0480e+03 | 1.1978e+01 | pass |
| gemm1_operands__bf16 | single_stage | f7d6ac7c | 52 | 0.997830 | 0.938793 | 2.0480e+03 | 1.1715e+02 | pass |
| gemm1_operands__f16 | single_stage | f7d6ac7c | 52 | 0.999759 | 0.992112 | 2.0480e+03 | 2.7683e+01 | pass |
| gemm1_accumulator__bf16 | single_stage | f7d6ac7c | 52 | 0.992346 | 0.835793 | 7.1680e+03 | 7.8461e+02 | pass |
| gemm1_accumulator__f16 | single_stage | f7d6ac7c | 52 | 0.999042 | 0.970623 | 2.0480e+03 | 5.8366e+01 | pass |
| gemm1_output__bf16 | single_stage | f7d6ac7c | 52 | 0.998565 | 0.956527 | 2.0480e+03 | 8.6073e+01 | pass |
| gemm1_output__f16 | single_stage | f7d6ac7c | 52 | 0.999842 | 0.994229 | 2.0480e+03 | 1.0122e+01 | pass |
| swiglu_input__bf16 | single_stage | f7d6ac7c | 52 | 0.998565 | 0.956527 | 2.0480e+03 | 8.6073e+01 | pass |
| swiglu_input__f16 | single_stage | f7d6ac7c | 52 | 0.999842 | 0.994229 | 2.0480e+03 | 1.0122e+01 | pass |
| swiglu_output__bf16 | single_stage | f7d6ac7c | 52 | 0.998946 | 0.968471 | 2.0480e+03 | 9.1196e+01 | pass |
| swiglu_output__f16 | single_stage | f7d6ac7c | 52 | 0.980662 | 0.976941 | nan | nan | pass |
| gemm2_operands__bf16 | single_stage | f7d6ac7c | 52 | 0.998495 | 0.956154 | 2.0480e+03 | 1.2879e+02 | pass |
| gemm2_operands__f16 | single_stage | f7d6ac7c | 52 | 0.980616 | 0.975573 | nan | nan | pass |
| gemm2_accumulator__bf16 | single_stage | f7d6ac7c | 52 | 0.998007 | 0.936282 | 4.0960e+03 | 1.5907e+02 | pass |
| gemm2_accumulator__f16 | single_stage | f7d6ac7c | 52 | 0.806716 | 0.806472 | nan | nan | global_drift |
| out_accumulator__bf16 | single_stage | f7d6ac7c | 52 | 0.999726 | 0.991350 | 2.0480e+03 | 6.6902e+01 | pass |
| out_accumulator__f16 | single_stage | f7d6ac7c | 52 | 0.878469 | 0.877779 | inf | inf | catastrophic_outlier |
| hidden_dequant__bf16 | cumulative | b8f4f012 | 7 | 0.998426 | 0.954899 | 2.0480e+03 | 7.5255e+00 | pass |
| hidden_dequant__bf16 | cumulative | e05c6c03 | 1 | 0.994978 | 0.846122 | 1.0240e+03 | 1.7339e+01 | pass |
| hidden_dequant__bf16 | cumulative | 6230e838 | 32 | 0.997973 | 0.938660 | 2.0480e+03 | 4.3069e+02 | pass |
| hidden_dequant__bf16 | cumulative | 8f1ff9f1 | 80 | 0.997121 | 0.908780 | 4.0960e+03 | 6.8700e+02 | pass |
| hidden_dequant__bf16 | cumulative | 1a4c6ba1 | 901 | 0.996876 | 0.904096 | 4.0960e+03 | 4.0438e+03 | pass |
| hidden_dequant__bf16 | cumulative | a7c2bcfd | 16 | 0.996861 | 0.902344 | 4.0960e+03 | 2.6884e+02 | pass |
| hidden_dequant__bf16 | cumulative | 2e69caee | 15 | 0.998596 | 0.960184 | 4.0960e+03 | 1.1014e+03 | pass |
| hidden_dequant__bf16 | cumulative | 8cba5890 | 14 | 0.997907 | 0.937161 | 2.0480e+03 | 4.7884e+01 | pass |
| hidden_dequant__bf16 | cumulative | 5e8dc11c | 14107 | 0.997169 | 0.912949 | 4.0960e+03 | 4.2037e+04 | pass |
| hidden_dequant__bf16 | cumulative | 58a34f27 | 11948 | 0.997835 | 0.933302 | 4.0960e+03 | 1.0547e+05 | pass |
| hidden_dequant__bf16 | cumulative | 5eadab1e | 62 | 0.997534 | 0.929577 | 2.0480e+03 | 1.2041e+02 | pass |
| hidden_dequant__bf16 | cumulative | eedc63b2 | 59 | 0.998184 | 0.939990 | 2.0480e+03 | 5.0952e+02 | pass |
| hidden_dequant__bf16 | cumulative | e626d3e6 | 58 | 0.997104 | 0.911429 | 4.0960e+03 | 8.9130e+01 | pass |
| hidden_dequant__bf16 | cumulative | 74d7ff04 | 57 | 0.997575 | 0.923774 | 4.0960e+03 | 2.7817e+02 | pass |
| hidden_dequant__bf16 | cumulative | 4822167c | 56 | 0.996844 | 0.907583 | 2.0480e+03 | 6.0121e+02 | pass |
| hidden_dequant__bf16 | cumulative | 81955b1e | 55 | 0.997669 | 0.929028 | 4.0960e+03 | 6.8456e+02 | pass |
| hidden_dequant__bf16 | cumulative | 76010cb4 | 54 | 0.998021 | 0.939910 | 4.0960e+03 | 2.1374e+02 | pass |
| hidden_dequant__bf16 | cumulative | fc378037 | 53 | 0.997752 | 0.930722 | 4.0960e+03 | 8.1936e+02 | pass |
| hidden_dequant__bf16 | cumulative | f7d6ac7c | 52 | 0.998659 | 0.957133 | 2.0480e+03 | 1.1884e+02 | pass |
| hidden_dequant__bf16+hidden_dequant__f16 | cumulative | b8f4f012 | 7 | 0.999900 | 0.994758 | 2.0480e+03 | 5.4632e+00 | pass |
| hidden_dequant__bf16+hidden_dequant__f16 | cumulative | e05c6c03 | 1 | 0.999302 | 0.975028 | 1.0240e+03 | 4.5663e+00 | pass |
| hidden_dequant__bf16+hidden_dequant__f16 | cumulative | 6230e838 | 32 | 0.999708 | 0.992706 | 2.0480e+03 | 3.0784e+01 | pass |
| hidden_dequant__bf16+hidden_dequant__f16 | cumulative | 8f1ff9f1 | 80 | 0.999660 | 0.988391 | 4.0960e+03 | 5.2478e+01 | pass |
| hidden_dequant__bf16+hidden_dequant__f16 | cumulative | 1a4c6ba1 | 901 | 0.999621 | 0.987655 | 4.0960e+03 | 1.2010e+03 | pass |
| hidden_dequant__bf16+hidden_dequant__f16 | cumulative | a7c2bcfd | 16 | 0.999651 | 0.987392 | 2.0480e+03 | 6.4404e+01 | pass |
| hidden_dequant__bf16+hidden_dequant__f16 | cumulative | 2e69caee | 15 | 0.999823 | 0.994996 | 2.0480e+03 | 7.6175e+01 | pass |
| hidden_dequant__bf16+hidden_dequant__f16 | cumulative | 8cba5890 | 14 | 0.999731 | 0.991699 | 2.0480e+03 | 4.4107e+00 | pass |
| hidden_dequant__bf16+hidden_dequant__f16 | cumulative | 5e8dc11c | 14107 | 0.999650 | 0.988895 | 4.0960e+03 | 1.3297e+04 | pass |
| hidden_dequant__bf16+hidden_dequant__f16 | cumulative | 58a34f27 | 11948 | 0.999729 | 0.991481 | 4.0960e+03 | 1.7665e+04 | pass |
| hidden_dequant__bf16+hidden_dequant__f16 | cumulative | 5eadab1e | 62 | 0.999683 | 0.990889 | 2.0480e+03 | 2.8804e+01 | pass |
| hidden_dequant__bf16+hidden_dequant__f16 | cumulative | eedc63b2 | 59 | 0.999740 | 0.992469 | 2.0480e+03 | 9.3415e+01 | pass |
| hidden_dequant__bf16+hidden_dequant__f16 | cumulative | e626d3e6 | 58 | 0.999644 | 0.988729 | 2.0480e+03 | 2.6942e+01 | pass |
| hidden_dequant__bf16+hidden_dequant__f16 | cumulative | 74d7ff04 | 57 | 0.999645 | 0.990313 | 2.0480e+03 | 4.7000e+01 | pass |
| hidden_dequant__bf16+hidden_dequant__f16 | cumulative | 4822167c | 56 | 0.999579 | 0.987746 | 2.0480e+03 | 4.9500e+01 | pass |
| hidden_dequant__bf16+hidden_dequant__f16 | cumulative | 81955b1e | 55 | 0.999744 | 0.990846 | 2.0480e+03 | 4.5024e+01 | pass |
| hidden_dequant__bf16+hidden_dequant__f16 | cumulative | 76010cb4 | 54 | 0.999739 | 0.992461 | 2.0480e+03 | 1.8038e+01 | pass |
| hidden_dequant__bf16+hidden_dequant__f16 | cumulative | fc378037 | 53 | 0.999739 | 0.990898 | 2.0480e+03 | 8.3051e+01 | pass |
| hidden_dequant__bf16+hidden_dequant__f16 | cumulative | f7d6ac7c | 52 | 0.999818 | 0.994487 | 2.0480e+03 | 1.0268e+01 | pass |
| hidden_dequant__bf16+hidden_dequant__f16+gemm1_operands__bf16 | cumulative | b8f4f012 | 7 | 0.997648 | 0.937241 | 2.0480e+03 | 3.6205e+02 | pass |
| hidden_dequant__bf16+hidden_dequant__f16+gemm1_operands__bf16 | cumulative | e05c6c03 | 1 | 0.993164 | 0.785993 | 2.0480e+03 | 1.1060e+01 | pass |
| hidden_dequant__bf16+hidden_dequant__f16+gemm1_operands__bf16 | cumulative | 6230e838 | 32 | 0.997083 | 0.914647 | 4.0960e+03 | 1.2466e+02 | pass |
| hidden_dequant__bf16+hidden_dequant__f16+gemm1_operands__bf16 | cumulative | 8f1ff9f1 | 80 | 0.995806 | 0.871470 | 4.0960e+03 | 4.7165e+02 | pass |
| hidden_dequant__bf16+hidden_dequant__f16+gemm1_operands__bf16 | cumulative | 1a4c6ba1 | 901 | 0.995533 | 0.865524 | 4.0960e+03 | 5.1199e+04 | pass |
| hidden_dequant__bf16+hidden_dequant__f16+gemm1_operands__bf16 | cumulative | a7c2bcfd | 16 | 0.995326 | 0.868286 | 2.0480e+03 | 1.8527e+02 | pass |
| hidden_dequant__bf16+hidden_dequant__f16+gemm1_operands__bf16 | cumulative | 2e69caee | 15 | 0.998261 | 0.943369 | 4.0960e+03 | 2.7521e+02 | pass |
| hidden_dequant__bf16+hidden_dequant__f16+gemm1_operands__bf16 | cumulative | 8cba5890 | 14 | 0.997319 | 0.907695 | 2.0480e+03 | 1.2734e+02 | pass |
| hidden_dequant__bf16+hidden_dequant__f16+gemm1_operands__bf16 | cumulative | 5e8dc11c | 14107 | 0.995945 | 0.877816 | 8.1920e+03 | 2.0700e+10 | pass |
| hidden_dequant__bf16+hidden_dequant__f16+gemm1_operands__bf16 | cumulative | 58a34f27 | 11948 | 0.996892 | 0.906176 | 4.0960e+03 | 6.1200e+10 | pass |
| hidden_dequant__bf16+hidden_dequant__f16+gemm1_operands__bf16 | cumulative | 5eadab1e | 62 | 0.996650 | 0.899268 | 4.0960e+03 | 1.7579e+03 | pass |
| hidden_dequant__bf16+hidden_dequant__f16+gemm1_operands__bf16 | cumulative | eedc63b2 | 59 | 0.997165 | 0.915429 | 2.0480e+03 | 2.6399e+02 | pass |
| hidden_dequant__bf16+hidden_dequant__f16+gemm1_operands__bf16 | cumulative | e626d3e6 | 58 | 0.996053 | 0.875637 | 4.0960e+03 | 2.1460e+03 | pass |
| hidden_dequant__bf16+hidden_dequant__f16+gemm1_operands__bf16 | cumulative | 74d7ff04 | 57 | 0.996478 | 0.894930 | 2.5600e+03 | 1.2823e+03 | pass |
| hidden_dequant__bf16+hidden_dequant__f16+gemm1_operands__bf16 | cumulative | 4822167c | 56 | 0.995655 | 0.869425 | 4.0960e+03 | 1.3120e+03 | pass |
| hidden_dequant__bf16+hidden_dequant__f16+gemm1_operands__bf16 | cumulative | 81955b1e | 55 | 0.996756 | 0.901159 | 2.1120e+03 | 2.6202e+02 | pass |
| hidden_dequant__bf16+hidden_dequant__f16+gemm1_operands__bf16 | cumulative | 76010cb4 | 54 | 0.997295 | 0.916563 | 4.0960e+03 | 7.4285e+02 | pass |
| hidden_dequant__bf16+hidden_dequant__f16+gemm1_operands__bf16 | cumulative | fc378037 | 53 | 0.996744 | 0.904860 | 4.0960e+03 | 1.2271e+03 | pass |
| hidden_dequant__bf16+hidden_dequant__f16+gemm1_operands__bf16 | cumulative | f7d6ac7c | 52 | 0.998101 | 0.938849 | 2.0480e+03 | 4.2645e+02 | pass |
| hidden_dequant__bf16+hidden_dequant__f16+gemm1_operands__bf16+gemm1_operands__f16 | cumulative | b8f4f012 | 7 | 0.999661 | 0.991331 | 2.0480e+03 | 1.2566e+01 | pass |
| hidden_dequant__bf16+hidden_dequant__f16+gemm1_operands__bf16+gemm1_operands__f16 | cumulative | e05c6c03 | 1 | 0.998465 | 0.967634 | 1.0240e+03 | 1.7870e+01 | pass |
| hidden_dequant__bf16+hidden_dequant__f16+gemm1_operands__bf16+gemm1_operands__f16 | cumulative | 6230e838 | 32 | 0.999669 | 0.988770 | 2.0480e+03 | 1.6368e+01 | pass |
| hidden_dequant__bf16+hidden_dequant__f16+gemm1_operands__bf16+gemm1_operands__f16 | cumulative | 8f1ff9f1 | 80 | 0.999428 | 0.983595 | 4.0960e+03 | 2.4700e+02 | pass |
| hidden_dequant__bf16+hidden_dequant__f16+gemm1_operands__bf16+gemm1_operands__f16 | cumulative | 1a4c6ba1 | 901 | 0.999449 | 0.982653 | 4.0960e+03 | 1.3046e+03 | pass |
| hidden_dequant__bf16+hidden_dequant__f16+gemm1_operands__bf16+gemm1_operands__f16 | cumulative | a7c2bcfd | 16 | 0.999425 | 0.983259 | 2.0480e+03 | 2.6930e+01 | pass |
| hidden_dequant__bf16+hidden_dequant__f16+gemm1_operands__bf16+gemm1_operands__f16 | cumulative | 2e69caee | 15 | 0.999767 | 0.992857 | 2.0480e+03 | 1.2574e+01 | pass |
| hidden_dequant__bf16+hidden_dequant__f16+gemm1_operands__bf16+gemm1_operands__f16 | cumulative | 8cba5890 | 14 | 0.999641 | 0.988839 | 2.0480e+03 | 7.7103e+00 | pass |
| hidden_dequant__bf16+hidden_dequant__f16+gemm1_operands__bf16+gemm1_operands__f16 | cumulative | 5e8dc11c | 14107 | 0.999498 | 0.984239 | 8.1920e+03 | 1.1374e+04 | pass |
| hidden_dequant__bf16+hidden_dequant__f16+gemm1_operands__bf16+gemm1_operands__f16 | cumulative | 58a34f27 | 11948 | 0.999617 | 0.987909 | 4.0960e+03 | 2.3937e+04 | pass |
| hidden_dequant__bf16+hidden_dequant__f16+gemm1_operands__bf16+gemm1_operands__f16 | cumulative | 5eadab1e | 62 | 0.999613 | 0.986987 | 2.0480e+03 | 1.9875e+01 | pass |
| hidden_dequant__bf16+hidden_dequant__f16+gemm1_operands__bf16+gemm1_operands__f16 | cumulative | eedc63b2 | 59 | 0.999591 | 0.989144 | 2.0480e+03 | 1.0187e+02 | pass |
| hidden_dequant__bf16+hidden_dequant__f16+gemm1_operands__bf16+gemm1_operands__f16 | cumulative | e626d3e6 | 58 | 0.999485 | 0.983740 | 2.0480e+03 | 5.3473e+01 | pass |
| hidden_dequant__bf16+hidden_dequant__f16+gemm1_operands__bf16+gemm1_operands__f16 | cumulative | 74d7ff04 | 57 | 0.999591 | 0.986495 | 2.0480e+03 | 2.5929e+01 | pass |
| hidden_dequant__bf16+hidden_dequant__f16+gemm1_operands__bf16+gemm1_operands__f16 | cumulative | 4822167c | 56 | 0.999422 | 0.983321 | 2.0480e+03 | 1.2452e+01 | pass |
| hidden_dequant__bf16+hidden_dequant__f16+gemm1_operands__bf16+gemm1_operands__f16 | cumulative | 81955b1e | 55 | 0.999561 | 0.986830 | 2.0480e+03 | 1.0927e+02 | pass |
| hidden_dequant__bf16+hidden_dequant__f16+gemm1_operands__bf16+gemm1_operands__f16 | cumulative | 76010cb4 | 54 | 0.999693 | 0.989521 | 2.0480e+03 | 1.4508e+02 | pass |
| hidden_dequant__bf16+hidden_dequant__f16+gemm1_operands__bf16+gemm1_operands__f16 | cumulative | fc378037 | 53 | 0.999600 | 0.987597 | 2.0480e+03 | 5.9602e+01 | pass |
| hidden_dequant__bf16+hidden_dequant__f16+gemm1_operands__bf16+gemm1_operands__f16 | cumulative | f7d6ac7c | 52 | 0.999740 | 0.991965 | 2.0480e+03 | 3.6299e+01 | pass |
| hidden_dequant__bf16+hidden_dequant__f16+gemm1_operands__bf16+gemm1_operands__f16+gemm1_output__bf16 | cumulative | b8f4f012 | 7 | 0.998744 | 0.960300 | 2.0480e+03 | 1.7370e+01 | pass |
| hidden_dequant__bf16+hidden_dequant__f16+gemm1_operands__bf16+gemm1_operands__f16+gemm1_output__bf16 | cumulative | e05c6c03 | 1 | 0.994280 | 0.830357 | 2.0480e+03 | 2.8604e+01 | pass |
| hidden_dequant__bf16+hidden_dequant__f16+gemm1_operands__bf16+gemm1_operands__f16+gemm1_output__bf16 | cumulative | 6230e838 | 32 | 0.997929 | 0.938463 | 4.0960e+03 | 4.2567e+02 | pass |
| hidden_dequant__bf16+hidden_dequant__f16+gemm1_operands__bf16+gemm1_operands__f16+gemm1_output__bf16 | cumulative | 8f1ff9f1 | 80 | 0.996957 | 0.907443 | 2.0480e+03 | 1.2238e+03 | pass |
| hidden_dequant__bf16+hidden_dequant__f16+gemm1_operands__bf16+gemm1_operands__f16+gemm1_output__bf16 | cumulative | 1a4c6ba1 | 901 | 0.996877 | 0.903098 | 4.0960e+03 | 3.0393e+04 | pass |
| hidden_dequant__bf16+hidden_dequant__f16+gemm1_operands__bf16+gemm1_operands__f16+gemm1_output__bf16 | cumulative | a7c2bcfd | 16 | 0.996809 | 0.905770 | 2.0480e+03 | 6.5547e+01 | pass |
| hidden_dequant__bf16+hidden_dequant__f16+gemm1_operands__bf16+gemm1_operands__f16+gemm1_output__bf16 | cumulative | 2e69caee | 15 | 0.998614 | 0.958715 | 4.0960e+03 | 1.5578e+02 | pass |
| hidden_dequant__bf16+hidden_dequant__f16+gemm1_operands__bf16+gemm1_operands__f16+gemm1_output__bf16 | cumulative | 8cba5890 | 14 | 0.997977 | 0.932787 | 2.0480e+03 | 3.2371e+01 | pass |
| hidden_dequant__bf16+hidden_dequant__f16+gemm1_operands__bf16+gemm1_operands__f16+gemm1_output__bf16 | cumulative | 5e8dc11c | 14107 | 0.997124 | 0.911850 | 8.1920e+03 | 1.1280e+05 | pass |
| hidden_dequant__bf16+hidden_dequant__f16+gemm1_operands__bf16+gemm1_operands__f16+gemm1_output__bf16 | cumulative | 58a34f27 | 11948 | 0.997798 | 0.932502 | 4.0960e+03 | 2.6094e+08 | pass |
| hidden_dequant__bf16+hidden_dequant__f16+gemm1_operands__bf16+gemm1_operands__f16+gemm1_output__bf16 | cumulative | 5eadab1e | 62 | 0.997556 | 0.927325 | 4.0960e+03 | 1.0319e+03 | pass |
| hidden_dequant__bf16+hidden_dequant__f16+gemm1_operands__bf16+gemm1_operands__f16+gemm1_output__bf16 | cumulative | eedc63b2 | 59 | 0.998023 | 0.941319 | 4.0960e+03 | 3.9535e+02 | pass |
| hidden_dequant__bf16+hidden_dequant__f16+gemm1_operands__bf16+gemm1_operands__f16+gemm1_output__bf16 | cumulative | e626d3e6 | 58 | 0.997121 | 0.909709 | 4.0960e+03 | 5.2491e+02 | pass |
| hidden_dequant__bf16+hidden_dequant__f16+gemm1_operands__bf16+gemm1_operands__f16+gemm1_output__bf16 | cumulative | 74d7ff04 | 57 | 0.997516 | 0.923498 | 2.0480e+03 | 1.1811e+03 | pass |
| hidden_dequant__bf16+hidden_dequant__f16+gemm1_operands__bf16+gemm1_operands__f16+gemm1_output__bf16 | cumulative | 4822167c | 56 | 0.996881 | 0.905874 | 2.0480e+03 | 4.5003e+02 | pass |
| hidden_dequant__bf16+hidden_dequant__f16+gemm1_operands__bf16+gemm1_operands__f16+gemm1_output__bf16 | cumulative | 81955b1e | 55 | 0.997664 | 0.927747 | 2.0480e+03 | 1.5258e+02 | pass |
| hidden_dequant__bf16+hidden_dequant__f16+gemm1_operands__bf16+gemm1_operands__f16+gemm1_output__bf16 | cumulative | 76010cb4 | 54 | 0.998148 | 0.940195 | 4.0960e+03 | 2.9132e+02 | pass |
| hidden_dequant__bf16+hidden_dequant__f16+gemm1_operands__bf16+gemm1_operands__f16+gemm1_output__bf16 | cumulative | fc378037 | 53 | 0.997718 | 0.930577 | 2.0480e+03 | 1.2833e+03 | pass |
| hidden_dequant__bf16+hidden_dequant__f16+gemm1_operands__bf16+gemm1_operands__f16+gemm1_output__bf16 | cumulative | f7d6ac7c | 52 | 0.998506 | 0.955365 | 2.0480e+03 | 2.6030e+03 | pass |
| hidden_dequant__bf16+hidden_dequant__f16+gemm1_operands__bf16+gemm1_operands__f16+gemm1_output__bf16+gemm1_output__f16 | cumulative | b8f4f012 | 7 | 0.999581 | 0.989716 | 2.0480e+03 | 5.8455e+00 | pass |
| hidden_dequant__bf16+hidden_dequant__f16+gemm1_operands__bf16+gemm1_operands__f16+gemm1_output__bf16+gemm1_output__f16 | cumulative | e05c6c03 | 1 | 0.999302 | 0.969029 | 2.0480e+03 | 4.1810e+00 | pass |
| hidden_dequant__bf16+hidden_dequant__f16+gemm1_operands__bf16+gemm1_operands__f16+gemm1_output__bf16+gemm1_output__f16 | cumulative | 6230e838 | 32 | 0.999529 | 0.986389 | 2.0480e+03 | 3.5053e+01 | pass |
| hidden_dequant__bf16+hidden_dequant__f16+gemm1_operands__bf16+gemm1_operands__f16+gemm1_output__bf16+gemm1_output__f16 | cumulative | 8f1ff9f1 | 80 | 0.999388 | 0.979986 | 2.0480e+03 | 4.3324e+01 | pass |
| hidden_dequant__bf16+hidden_dequant__f16+gemm1_operands__bf16+gemm1_operands__f16+gemm1_output__bf16+gemm1_output__f16 | cumulative | 1a4c6ba1 | 901 | 0.999321 | 0.978755 | 4.0960e+03 | 1.2926e+03 | pass |
| hidden_dequant__bf16+hidden_dequant__f16+gemm1_operands__bf16+gemm1_operands__f16+gemm1_output__bf16+gemm1_output__f16 | cumulative | a7c2bcfd | 16 | 0.999294 | 0.980434 | 2.0480e+03 | 2.1698e+01 | pass |
| hidden_dequant__bf16+hidden_dequant__f16+gemm1_operands__bf16+gemm1_operands__f16+gemm1_output__bf16+gemm1_output__f16 | cumulative | 2e69caee | 15 | 0.999712 | 0.991313 | 2.0480e+03 | 6.3000e+01 | pass |
| hidden_dequant__bf16+hidden_dequant__f16+gemm1_operands__bf16+gemm1_operands__f16+gemm1_output__bf16+gemm1_output__f16 | cumulative | 8cba5890 | 14 | 0.999522 | 0.984644 | 2.0480e+03 | 4.1872e+00 | pass |
| hidden_dequant__bf16+hidden_dequant__f16+gemm1_operands__bf16+gemm1_operands__f16+gemm1_output__bf16+gemm1_output__f16 | cumulative | 5e8dc11c | 14107 | 0.999388 | 0.980738 | 4.0960e+03 | 9.0000e+09 | pass |
| hidden_dequant__bf16+hidden_dequant__f16+gemm1_operands__bf16+gemm1_operands__f16+gemm1_output__bf16+gemm1_output__f16 | cumulative | 58a34f27 | 11948 | 0.999533 | 0.985241 | 4.0960e+03 | 4.7000e+09 | pass |
| hidden_dequant__bf16+hidden_dequant__f16+gemm1_operands__bf16+gemm1_operands__f16+gemm1_output__bf16+gemm1_output__f16 | cumulative | 5eadab1e | 62 | 0.999491 | 0.984283 | 2.0480e+03 | 5.1409e+01 | pass |
| hidden_dequant__bf16+hidden_dequant__f16+gemm1_operands__bf16+gemm1_operands__f16+gemm1_output__bf16+gemm1_output__f16 | cumulative | eedc63b2 | 59 | 0.999527 | 0.987030 | 4.0960e+03 | 3.2592e+02 | pass |
| hidden_dequant__bf16+hidden_dequant__f16+gemm1_operands__bf16+gemm1_operands__f16+gemm1_output__bf16+gemm1_output__f16 | cumulative | e626d3e6 | 58 | 0.999351 | 0.980274 | 2.0480e+03 | 1.5995e+02 | pass |
| hidden_dequant__bf16+hidden_dequant__f16+gemm1_operands__bf16+gemm1_operands__f16+gemm1_output__bf16+gemm1_output__f16 | cumulative | 74d7ff04 | 57 | 0.999506 | 0.983770 | 2.0480e+03 | 1.4813e+02 | pass |
| hidden_dequant__bf16+hidden_dequant__f16+gemm1_operands__bf16+gemm1_operands__f16+gemm1_output__bf16+gemm1_output__f16 | cumulative | 4822167c | 56 | 0.999330 | 0.979457 | 4.0960e+03 | 7.3254e+02 | pass |
| hidden_dequant__bf16+hidden_dequant__f16+gemm1_operands__bf16+gemm1_operands__f16+gemm1_output__bf16+gemm1_output__f16 | cumulative | 81955b1e | 55 | 0.999472 | 0.983781 | 2.0480e+03 | 1.8973e+01 | pass |
| hidden_dequant__bf16+hidden_dequant__f16+gemm1_operands__bf16+gemm1_operands__f16+gemm1_output__bf16+gemm1_output__f16 | cumulative | 76010cb4 | 54 | 0.999566 | 0.987005 | 2.0480e+03 | 2.5246e+01 | pass |
| hidden_dequant__bf16+hidden_dequant__f16+gemm1_operands__bf16+gemm1_operands__f16+gemm1_output__bf16+gemm1_output__f16 | cumulative | fc378037 | 53 | 0.999495 | 0.984436 | 2.0480e+03 | 1.4789e+02 | pass |
| hidden_dequant__bf16+hidden_dequant__f16+gemm1_operands__bf16+gemm1_operands__f16+gemm1_output__bf16+gemm1_output__f16 | cumulative | f7d6ac7c | 52 | 0.999753 | 0.990398 | 2.0480e+03 | 2.8320e+01 | pass |
| hidden_dequant__bf16+hidden_dequant__f16+gemm1_operands__bf16+gemm1_operands__f16+gemm1_output__bf16+gemm1_output__f16+swiglu_input__bf16 | cumulative | b8f4f012 | 7 | 0.998964 | 0.955576 | 2.0480e+03 | 2.0829e+01 | pass |
| hidden_dequant__bf16+hidden_dequant__f16+gemm1_operands__bf16+gemm1_operands__f16+gemm1_output__bf16+gemm1_output__f16+swiglu_input__bf16 | cumulative | e05c6c03 | 1 | 0.994280 | 0.837612 | 2.0480e+03 | 1.7748e+01 | pass |
| hidden_dequant__bf16+hidden_dequant__f16+gemm1_operands__bf16+gemm1_operands__f16+gemm1_output__bf16+gemm1_output__f16+swiglu_input__bf16 | cumulative | 6230e838 | 32 | 0.997942 | 0.935669 | 4.0960e+03 | 1.2399e+02 | pass |
| hidden_dequant__bf16+hidden_dequant__f16+gemm1_operands__bf16+gemm1_operands__f16+gemm1_output__bf16+gemm1_output__f16+swiglu_input__bf16 | cumulative | 8f1ff9f1 | 80 | 0.996994 | 0.905186 | 4.0960e+03 | 8.1910e+03 | pass |
| hidden_dequant__bf16+hidden_dequant__f16+gemm1_operands__bf16+gemm1_operands__f16+gemm1_output__bf16+gemm1_output__f16+swiglu_input__bf16 | cumulative | 1a4c6ba1 | 901 | 0.996769 | 0.900580 | 4.0960e+03 | 1.2361e+04 | pass |
| hidden_dequant__bf16+hidden_dequant__f16+gemm1_operands__bf16+gemm1_operands__f16+gemm1_output__bf16+gemm1_output__f16+swiglu_input__bf16 | cumulative | a7c2bcfd | 16 | 0.996861 | 0.904733 | 2.0480e+03 | 8.9594e+01 | pass |
| hidden_dequant__bf16+hidden_dequant__f16+gemm1_operands__bf16+gemm1_operands__f16+gemm1_output__bf16+gemm1_output__f16+swiglu_input__bf16 | cumulative | 2e69caee | 15 | 0.998717 | 0.959356 | 2.0480e+03 | 8.9141e+01 | pass |
| hidden_dequant__bf16+hidden_dequant__f16+gemm1_operands__bf16+gemm1_operands__f16+gemm1_output__bf16+gemm1_output__f16+swiglu_input__bf16 | cumulative | 8cba5890 | 14 | 0.997748 | 0.930056 | 2.0480e+03 | 5.1124e+01 | pass |
| hidden_dequant__bf16+hidden_dequant__f16+gemm1_operands__bf16+gemm1_operands__f16+gemm1_output__bf16+gemm1_output__f16+swiglu_input__bf16 | cumulative | 5e8dc11c | 14107 | 0.997076 | 0.910101 | 8.1920e+03 | 3.0542e+04 | pass |
| hidden_dequant__bf16+hidden_dequant__f16+gemm1_operands__bf16+gemm1_operands__f16+gemm1_output__bf16+gemm1_output__f16+swiglu_input__bf16 | cumulative | 58a34f27 | 11948 | 0.997763 | 0.931204 | 4.0960e+03 | 1.2213e+04 | pass |
| hidden_dequant__bf16+hidden_dequant__f16+gemm1_operands__bf16+gemm1_operands__f16+gemm1_output__bf16+gemm1_output__f16+swiglu_input__bf16 | cumulative | 5eadab1e | 62 | 0.997640 | 0.925516 | 2.0480e+03 | 8.4000e+01 | pass |
| hidden_dequant__bf16+hidden_dequant__f16+gemm1_operands__bf16+gemm1_operands__f16+gemm1_output__bf16+gemm1_output__f16+swiglu_input__bf16 | cumulative | eedc63b2 | 59 | 0.998004 | 0.937982 | 2.0480e+03 | 3.1035e+02 | pass |
| hidden_dequant__bf16+hidden_dequant__f16+gemm1_operands__bf16+gemm1_operands__f16+gemm1_output__bf16+gemm1_output__f16+swiglu_input__bf16 | cumulative | e626d3e6 | 58 | 0.997053 | 0.906936 | 4.0960e+03 | 5.5405e+02 | pass |
| hidden_dequant__bf16+hidden_dequant__f16+gemm1_operands__bf16+gemm1_operands__f16+gemm1_output__bf16+gemm1_output__f16+swiglu_input__bf16 | cumulative | 74d7ff04 | 57 | 0.997543 | 0.923390 | 4.0960e+03 | 5.0386e+02 | pass |
| hidden_dequant__bf16+hidden_dequant__f16+gemm1_operands__bf16+gemm1_operands__f16+gemm1_output__bf16+gemm1_output__f16+swiglu_input__bf16 | cumulative | 4822167c | 56 | 0.996916 | 0.904514 | 4.0960e+03 | 2.3330e+02 | pass |
| hidden_dequant__bf16+hidden_dequant__f16+gemm1_operands__bf16+gemm1_operands__f16+gemm1_output__bf16+gemm1_output__f16+swiglu_input__bf16 | cumulative | 81955b1e | 55 | 0.997540 | 0.926471 | 2.0480e+03 | 2.3942e+02 | pass |
| hidden_dequant__bf16+hidden_dequant__f16+gemm1_operands__bf16+gemm1_operands__f16+gemm1_output__bf16+gemm1_output__f16+swiglu_input__bf16 | cumulative | 76010cb4 | 54 | 0.998099 | 0.939311 | 2.0480e+03 | 2.3284e+03 | pass |
| hidden_dequant__bf16+hidden_dequant__f16+gemm1_operands__bf16+gemm1_operands__f16+gemm1_output__bf16+gemm1_output__f16+swiglu_input__bf16 | cumulative | fc378037 | 53 | 0.997736 | 0.928385 | 4.0960e+03 | 1.0150e+03 | pass |
| hidden_dequant__bf16+hidden_dequant__f16+gemm1_operands__bf16+gemm1_operands__f16+gemm1_output__bf16+gemm1_output__f16+swiglu_input__bf16 | cumulative | f7d6ac7c | 52 | 0.998546 | 0.954576 | 2.0480e+03 | 9.2308e+01 | pass |
| hidden_dequant__bf16+hidden_dequant__f16+gemm1_operands__bf16+gemm1_operands__f16+gemm1_output__bf16+gemm1_output__f16+swiglu_input__bf16+swiglu_input__f16 | cumulative | b8f4f012 | 7 | 0.999681 | 0.990753 | 2.0480e+03 | 6.6951e+01 | pass |
| hidden_dequant__bf16+hidden_dequant__f16+gemm1_operands__bf16+gemm1_operands__f16+gemm1_output__bf16+gemm1_output__f16+swiglu_input__bf16+swiglu_input__f16 | cumulative | e05c6c03 | 1 | 0.999163 | 0.970006 | 2.0480e+03 | 1.6905e+00 | pass |
| hidden_dequant__bf16+hidden_dequant__f16+gemm1_operands__bf16+gemm1_operands__f16+gemm1_output__bf16+gemm1_output__f16+swiglu_input__bf16+swiglu_input__f16 | cumulative | 6230e838 | 32 | 0.999503 | 0.986389 | 2.0480e+03 | 4.3492e+01 | pass |
| hidden_dequant__bf16+hidden_dequant__f16+gemm1_operands__bf16+gemm1_operands__f16+gemm1_output__bf16+gemm1_output__f16+swiglu_input__bf16+swiglu_input__f16 | cumulative | 8f1ff9f1 | 80 | 0.999386 | 0.979586 | 2.0480e+03 | 5.4145e+01 | pass |
| hidden_dequant__bf16+hidden_dequant__f16+gemm1_operands__bf16+gemm1_operands__f16+gemm1_output__bf16+gemm1_output__f16+swiglu_input__bf16+swiglu_input__f16 | cumulative | 1a4c6ba1 | 901 | 0.999320 | 0.978724 | 4.0960e+03 | 2.3599e+03 | pass |
| hidden_dequant__bf16+hidden_dequant__f16+gemm1_operands__bf16+gemm1_operands__f16+gemm1_output__bf16+gemm1_output__f16+swiglu_input__bf16+swiglu_input__f16 | cumulative | a7c2bcfd | 16 | 0.999285 | 0.977870 | 2.0480e+03 | 1.7581e+01 | pass |
| hidden_dequant__bf16+hidden_dequant__f16+gemm1_operands__bf16+gemm1_operands__f16+gemm1_output__bf16+gemm1_output__f16+swiglu_input__bf16+swiglu_input__f16 | cumulative | 2e69caee | 15 | 0.999656 | 0.991248 | 2.0480e+03 | 2.4732e+01 | pass |
| hidden_dequant__bf16+hidden_dequant__f16+gemm1_operands__bf16+gemm1_operands__f16+gemm1_output__bf16+gemm1_output__f16+swiglu_input__bf16+swiglu_input__f16 | cumulative | 8cba5890 | 14 | 0.999552 | 0.985641 | 2.0480e+03 | 3.1842e+01 | pass |
| hidden_dequant__bf16+hidden_dequant__f16+gemm1_operands__bf16+gemm1_operands__f16+gemm1_output__bf16+gemm1_output__f16+swiglu_input__bf16+swiglu_input__f16 | cumulative | 5e8dc11c | 14107 | 0.999386 | 0.980723 | 4.0960e+03 | 2.8906e+08 | pass |
| hidden_dequant__bf16+hidden_dequant__f16+gemm1_operands__bf16+gemm1_operands__f16+gemm1_output__bf16+gemm1_output__f16+swiglu_input__bf16+swiglu_input__f16 | cumulative | 58a34f27 | 11948 | 0.999528 | 0.985249 | 4.0960e+03 | 1.9455e+04 | pass |
| hidden_dequant__bf16+hidden_dequant__f16+gemm1_operands__bf16+gemm1_operands__f16+gemm1_output__bf16+gemm1_output__f16+swiglu_input__bf16+swiglu_input__f16 | cumulative | 5eadab1e | 62 | 0.999545 | 0.983911 | 4.0960e+03 | 8.6843e+01 | pass |
| hidden_dequant__bf16+hidden_dequant__f16+gemm1_operands__bf16+gemm1_operands__f16+gemm1_output__bf16+gemm1_output__f16+swiglu_input__bf16+swiglu_input__f16 | cumulative | eedc63b2 | 59 | 0.999584 | 0.987023 | 2.0480e+03 | 3.3519e+01 | pass |
| hidden_dequant__bf16+hidden_dequant__f16+gemm1_operands__bf16+gemm1_operands__f16+gemm1_output__bf16+gemm1_output__f16+swiglu_input__bf16+swiglu_input__f16 | cumulative | e626d3e6 | 58 | 0.999408 | 0.980399 | 4.0960e+03 | 2.9431e+01 | pass |
| hidden_dequant__bf16+hidden_dequant__f16+gemm1_operands__bf16+gemm1_operands__f16+gemm1_output__bf16+gemm1_output__f16+swiglu_input__bf16+swiglu_input__f16 | cumulative | 74d7ff04 | 57 | 0.999464 | 0.983489 | 2.0480e+03 | 1.8005e+02 | pass |
| hidden_dequant__bf16+hidden_dequant__f16+gemm1_operands__bf16+gemm1_operands__f16+gemm1_output__bf16+gemm1_output__f16+swiglu_input__bf16+swiglu_input__f16 | cumulative | 4822167c | 56 | 0.999360 | 0.979198 | 2.0480e+03 | 6.1688e+02 | pass |
| hidden_dequant__bf16+hidden_dequant__f16+gemm1_operands__bf16+gemm1_operands__f16+gemm1_output__bf16+gemm1_output__f16+swiglu_input__bf16+swiglu_input__f16 | cumulative | 81955b1e | 55 | 0.999533 | 0.984190 | 2.0480e+03 | 1.7738e+02 | pass |
| hidden_dequant__bf16+hidden_dequant__f16+gemm1_operands__bf16+gemm1_operands__f16+gemm1_output__bf16+gemm1_output__f16+swiglu_input__bf16+swiglu_input__f16 | cumulative | 76010cb4 | 54 | 0.999602 | 0.986889 | 2.0480e+03 | 1.1408e+02 | pass |
| hidden_dequant__bf16+hidden_dequant__f16+gemm1_operands__bf16+gemm1_operands__f16+gemm1_output__bf16+gemm1_output__f16+swiglu_input__bf16+swiglu_input__f16 | cumulative | fc378037 | 53 | 0.999434 | 0.984836 | 2.0480e+03 | 2.2503e+01 | pass |
| hidden_dequant__bf16+hidden_dequant__f16+gemm1_operands__bf16+gemm1_operands__f16+gemm1_output__bf16+gemm1_output__f16+swiglu_input__bf16+swiglu_input__f16 | cumulative | f7d6ac7c | 52 | 0.999718 | 0.989786 | 2.0480e+03 | 1.3798e+01 | pass |
| hidden_dequant__bf16+hidden_dequant__f16+gemm1_operands__bf16+gemm1_operands__f16+gemm1_output__bf16+gemm1_output__f16+swiglu_input__bf16+swiglu_input__f16+swiglu_output__bf16 | cumulative | b8f4f012 | 7 | 0.998984 | 0.968371 | 2.0480e+03 | 1.0725e+02 | pass |
| hidden_dequant__bf16+hidden_dequant__f16+gemm1_operands__bf16+gemm1_operands__f16+gemm1_output__bf16+gemm1_output__f16+swiglu_input__bf16+swiglu_input__f16+swiglu_output__bf16 | cumulative | e05c6c03 | 1 | 0.996094 | 0.889090 | 2.0480e+03 | 2.1557e+01 | pass |
| hidden_dequant__bf16+hidden_dequant__f16+gemm1_operands__bf16+gemm1_operands__f16+gemm1_output__bf16+gemm1_output__f16+swiglu_input__bf16+swiglu_input__f16+swiglu_output__bf16 | cumulative | 6230e838 | 32 | 0.998396 | 0.953600 | 2.0480e+03 | 8.7871e+02 | pass |
| hidden_dequant__bf16+hidden_dequant__f16+gemm1_operands__bf16+gemm1_operands__f16+gemm1_output__bf16+gemm1_output__f16+swiglu_input__bf16+swiglu_input__f16+swiglu_output__bf16 | cumulative | 8f1ff9f1 | 80 | 0.997738 | 0.931189 | 2.0480e+03 | 4.0918e+02 | pass |
| hidden_dequant__bf16+hidden_dequant__f16+gemm1_operands__bf16+gemm1_operands__f16+gemm1_output__bf16+gemm1_output__f16+swiglu_input__bf16+swiglu_input__f16+swiglu_output__bf16 | cumulative | 1a4c6ba1 | 901 | 0.997723 | 0.928292 | 4.0960e+03 | 4.3106e+03 | pass |
| hidden_dequant__bf16+hidden_dequant__f16+gemm1_operands__bf16+gemm1_operands__f16+gemm1_output__bf16+gemm1_output__f16+swiglu_input__bf16+swiglu_input__f16+swiglu_output__bf16 | cumulative | a7c2bcfd | 16 | 0.997655 | 0.927987 | 2.0480e+03 | 8.1386e+01 | pass |
| hidden_dequant__bf16+hidden_dequant__f16+gemm1_operands__bf16+gemm1_operands__f16+gemm1_output__bf16+gemm1_output__f16+swiglu_input__bf16+swiglu_input__f16+swiglu_output__bf16 | cumulative | 2e69caee | 15 | 0.998958 | 0.970582 | 2.0480e+03 | 1.2657e+01 | pass |
| hidden_dequant__bf16+hidden_dequant__f16+gemm1_operands__bf16+gemm1_operands__f16+gemm1_output__bf16+gemm1_output__f16+swiglu_input__bf16+swiglu_input__f16+swiglu_output__bf16 | cumulative | 8cba5890 | 14 | 0.998575 | 0.951780 | 2.0480e+03 | 2.0661e+02 | pass |
| hidden_dequant__bf16+hidden_dequant__f16+gemm1_operands__bf16+gemm1_operands__f16+gemm1_output__bf16+gemm1_output__f16+swiglu_input__bf16+swiglu_input__f16+swiglu_output__bf16 | cumulative | 5e8dc11c | 14107 | 0.997904 | 0.935140 | 4.0960e+03 | 3.8145e+04 | pass |
| hidden_dequant__bf16+hidden_dequant__f16+gemm1_operands__bf16+gemm1_operands__f16+gemm1_output__bf16+gemm1_output__f16+swiglu_input__bf16+swiglu_input__f16+swiglu_output__bf16 | cumulative | 58a34f27 | 11948 | 0.998403 | 0.950351 | 4.0960e+03 | 1.6212e+04 | pass |
| hidden_dequant__bf16+hidden_dequant__f16+gemm1_operands__bf16+gemm1_operands__f16+gemm1_output__bf16+gemm1_output__f16+swiglu_input__bf16+swiglu_input__f16+swiglu_output__bf16 | cumulative | 5eadab1e | 62 | 0.998328 | 0.947394 | 4.0960e+03 | 4.2300e+02 | pass |
| hidden_dequant__bf16+hidden_dequant__f16+gemm1_operands__bf16+gemm1_operands__f16+gemm1_output__bf16+gemm1_output__f16+swiglu_input__bf16+swiglu_input__f16+swiglu_output__bf16 | cumulative | eedc63b2 | 59 | 0.998527 | 0.955556 | 4.0960e+03 | 8.4440e+02 | pass |
| hidden_dequant__bf16+hidden_dequant__f16+gemm1_operands__bf16+gemm1_operands__f16+gemm1_output__bf16+gemm1_output__f16+swiglu_input__bf16+swiglu_input__f16+swiglu_output__bf16 | cumulative | e626d3e6 | 58 | 0.997756 | 0.933555 | 4.0960e+03 | 2.6945e+02 | pass |
| hidden_dequant__bf16+hidden_dequant__f16+gemm1_operands__bf16+gemm1_operands__f16+gemm1_output__bf16+gemm1_output__f16+swiglu_input__bf16+swiglu_input__f16+swiglu_output__bf16 | cumulative | 74d7ff04 | 57 | 0.998319 | 0.944084 | 4.0960e+03 | 9.7192e+01 | pass |
| hidden_dequant__bf16+hidden_dequant__f16+gemm1_operands__bf16+gemm1_operands__f16+gemm1_output__bf16+gemm1_output__f16+swiglu_input__bf16+swiglu_input__f16+swiglu_output__bf16 | cumulative | 4822167c | 56 | 0.997713 | 0.931421 | 2.0480e+03 | 6.7252e+02 | pass |
| hidden_dequant__bf16+hidden_dequant__f16+gemm1_operands__bf16+gemm1_operands__f16+gemm1_output__bf16+gemm1_output__f16+swiglu_input__bf16+swiglu_input__f16+swiglu_output__bf16 | cumulative | 81955b1e | 55 | 0.998369 | 0.946685 | 4.0960e+03 | 4.9533e+02 | pass |
| hidden_dequant__bf16+hidden_dequant__f16+gemm1_operands__bf16+gemm1_operands__f16+gemm1_output__bf16+gemm1_output__f16+swiglu_input__bf16+swiglu_input__f16+swiglu_output__bf16 | cumulative | 76010cb4 | 54 | 0.998646 | 0.956579 | 4.0960e+03 | 3.1854e+02 | pass |
| hidden_dequant__bf16+hidden_dequant__f16+gemm1_operands__bf16+gemm1_operands__f16+gemm1_output__bf16+gemm1_output__f16+swiglu_input__bf16+swiglu_input__f16+swiglu_output__bf16 | cumulative | fc378037 | 53 | 0.998352 | 0.947916 | 4.0960e+03 | 1.3680e+02 | pass |
| hidden_dequant__bf16+hidden_dequant__f16+gemm1_operands__bf16+gemm1_operands__f16+gemm1_output__bf16+gemm1_output__f16+swiglu_input__bf16+swiglu_input__f16+swiglu_output__bf16 | cumulative | f7d6ac7c | 52 | 0.999015 | 0.966633 | 2.0480e+03 | 1.1745e+03 | pass |
| hidden_dequant__bf16+hidden_dequant__f16+gemm1_operands__bf16+gemm1_operands__f16+gemm1_output__bf16+gemm1_output__f16+swiglu_input__bf16+swiglu_input__f16+swiglu_output__bf16+swiglu_output__f16 | cumulative | b8f4f012 | 7 | 0.999721 | 0.989178 | 2.0480e+03 | 5.7358e+00 | pass |
| hidden_dequant__bf16+hidden_dequant__f16+gemm1_operands__bf16+gemm1_operands__f16+gemm1_output__bf16+gemm1_output__f16+swiglu_input__bf16+swiglu_input__f16+swiglu_output__bf16+swiglu_output__f16 | cumulative | e05c6c03 | 1 | 0.999023 | 0.967076 | 2.0480e+03 | 1.0735e+00 | pass |
| hidden_dequant__bf16+hidden_dequant__f16+gemm1_operands__bf16+gemm1_operands__f16+gemm1_output__bf16+gemm1_output__f16+swiglu_input__bf16+swiglu_input__f16+swiglu_output__bf16+swiglu_output__f16 | cumulative | 6230e838 | 32 | 0.952781 | 0.941271 | nan | nan | pass |
| hidden_dequant__bf16+hidden_dequant__f16+gemm1_operands__bf16+gemm1_operands__f16+gemm1_output__bf16+gemm1_output__f16+swiglu_input__bf16+swiglu_input__f16+swiglu_output__bf16+swiglu_output__f16 | cumulative | 8f1ff9f1 | 80 | 0.952485 | 0.933311 | nan | nan | pass |
| hidden_dequant__bf16+hidden_dequant__f16+gemm1_operands__bf16+gemm1_operands__f16+gemm1_output__bf16+gemm1_output__f16+swiglu_input__bf16+swiglu_input__f16+swiglu_output__bf16+swiglu_output__f16 | cumulative | 1a4c6ba1 | 901 | 0.944839 | 0.924875 | nan | nan | pass |
| hidden_dequant__bf16+hidden_dequant__f16+gemm1_operands__bf16+gemm1_operands__f16+gemm1_output__bf16+gemm1_output__f16+swiglu_input__bf16+swiglu_input__f16+swiglu_output__bf16+swiglu_output__f16 | cumulative | a7c2bcfd | 16 | 0.999329 | 0.977609 | 2.0480e+03 | 7.4175e+01 | pass |
| hidden_dequant__bf16+hidden_dequant__f16+gemm1_operands__bf16+gemm1_operands__f16+gemm1_output__bf16+gemm1_output__f16+swiglu_input__bf16+swiglu_input__f16+swiglu_output__bf16+swiglu_output__f16 | cumulative | 2e69caee | 15 | 0.999851 | 0.990858 | 2.0480e+03 | 8.0577e+01 | pass |
| hidden_dequant__bf16+hidden_dequant__f16+gemm1_operands__bf16+gemm1_operands__f16+gemm1_output__bf16+gemm1_output__f16+swiglu_input__bf16+swiglu_input__f16+swiglu_output__bf16+swiglu_output__f16 | cumulative | 8cba5890 | 14 | 0.999472 | 0.984903 | 1.0240e+03 | 1.9000e+01 | pass |
| hidden_dequant__bf16+hidden_dequant__f16+gemm1_operands__bf16+gemm1_operands__f16+gemm1_output__bf16+gemm1_output__f16+swiglu_input__bf16+swiglu_input__f16+swiglu_output__bf16+swiglu_output__f16 | cumulative | 5e8dc11c | 14107 | 0.961306 | 0.942641 | nan | nan | pass |
| hidden_dequant__bf16+hidden_dequant__f16+gemm1_operands__bf16+gemm1_operands__f16+gemm1_output__bf16+gemm1_output__f16+swiglu_input__bf16+swiglu_input__f16+swiglu_output__bf16+swiglu_output__f16 | cumulative | 58a34f27 | 11948 | 0.964767 | 0.950698 | nan | nan | pass |
| hidden_dequant__bf16+hidden_dequant__f16+gemm1_operands__bf16+gemm1_operands__f16+gemm1_output__bf16+gemm1_output__f16+swiglu_input__bf16+swiglu_input__f16+swiglu_output__bf16+swiglu_output__f16 | cumulative | 5eadab1e | 62 | 0.967254 | 0.951728 | nan | nan | pass |
| hidden_dequant__bf16+hidden_dequant__f16+gemm1_operands__bf16+gemm1_operands__f16+gemm1_output__bf16+gemm1_output__f16+swiglu_input__bf16+swiglu_input__f16+swiglu_output__bf16+swiglu_output__f16 | cumulative | eedc63b2 | 59 | 0.948862 | 0.936989 | nan | nan | pass |
| hidden_dequant__bf16+hidden_dequant__f16+gemm1_operands__bf16+gemm1_operands__f16+gemm1_output__bf16+gemm1_output__f16+swiglu_input__bf16+swiglu_input__f16+swiglu_output__bf16+swiglu_output__f16 | cumulative | e626d3e6 | 58 | 0.930481 | 0.911900 | nan | nan | pass |
| hidden_dequant__bf16+hidden_dequant__f16+gemm1_operands__bf16+gemm1_operands__f16+gemm1_output__bf16+gemm1_output__f16+swiglu_input__bf16+swiglu_input__f16+swiglu_output__bf16+swiglu_output__f16 | cumulative | 74d7ff04 | 57 | 0.946933 | 0.931770 | nan | nan | pass |
| hidden_dequant__bf16+hidden_dequant__f16+gemm1_operands__bf16+gemm1_operands__f16+gemm1_output__bf16+gemm1_output__f16+swiglu_input__bf16+swiglu_input__f16+swiglu_output__bf16+swiglu_output__f16 | cumulative | 4822167c | 56 | 0.941329 | 0.922478 | nan | nan | pass |
| hidden_dequant__bf16+hidden_dequant__f16+gemm1_operands__bf16+gemm1_operands__f16+gemm1_output__bf16+gemm1_output__f16+swiglu_input__bf16+swiglu_input__f16+swiglu_output__bf16+swiglu_output__f16 | cumulative | 81955b1e | 55 | 0.963114 | 0.947202 | nan | nan | pass |
| hidden_dequant__bf16+hidden_dequant__f16+gemm1_operands__bf16+gemm1_operands__f16+gemm1_output__bf16+gemm1_output__f16+swiglu_input__bf16+swiglu_input__f16+swiglu_output__bf16+swiglu_output__f16 | cumulative | 76010cb4 | 54 | 0.953308 | 0.941569 | nan | nan | pass |
| hidden_dequant__bf16+hidden_dequant__f16+gemm1_operands__bf16+gemm1_operands__f16+gemm1_output__bf16+gemm1_output__f16+swiglu_input__bf16+swiglu_input__f16+swiglu_output__bf16+swiglu_output__f16 | cumulative | fc378037 | 53 | 0.957302 | 0.943591 | nan | nan | pass |
| hidden_dequant__bf16+hidden_dequant__f16+gemm1_operands__bf16+gemm1_operands__f16+gemm1_output__bf16+gemm1_output__f16+swiglu_input__bf16+swiglu_input__f16+swiglu_output__bf16+swiglu_output__f16 | cumulative | f7d6ac7c | 52 | 0.980488 | 0.970580 | nan | nan | pass |
| hidden_dequant__bf16+hidden_dequant__f16+gemm1_operands__bf16+gemm1_operands__f16+gemm1_output__bf16+gemm1_output__f16+swiglu_input__bf16+swiglu_input__f16+swiglu_output__bf16+swiglu_output__f16+gemm2_operands__bf16 | cumulative | b8f4f012 | 7 | 0.998705 | 0.954839 | 4.0960e+03 | 1.6519e+02 | pass |
| hidden_dequant__bf16+hidden_dequant__f16+gemm1_operands__bf16+gemm1_operands__f16+gemm1_output__bf16+gemm1_output__f16+swiglu_input__bf16+swiglu_input__f16+swiglu_output__bf16+swiglu_output__f16+gemm2_operands__bf16 | cumulative | e05c6c03 | 1 | 0.995536 | 0.849051 | 1.0240e+03 | 5.1078e+01 | pass |
| hidden_dequant__bf16+hidden_dequant__f16+gemm1_operands__bf16+gemm1_operands__f16+gemm1_output__bf16+gemm1_output__f16+swiglu_input__bf16+swiglu_input__f16+swiglu_output__bf16+swiglu_output__f16+gemm2_operands__bf16 | cumulative | 6230e838 | 32 | 0.966897 | 0.910047 | nan | nan | pass |
| hidden_dequant__bf16+hidden_dequant__f16+gemm1_operands__bf16+gemm1_operands__f16+gemm1_output__bf16+gemm1_output__f16+swiglu_input__bf16+swiglu_input__f16+swiglu_output__bf16+swiglu_output__f16+gemm2_operands__bf16 | cumulative | 8f1ff9f1 | 80 | 0.990806 | 0.901957 | nan | nan | pass |
| hidden_dequant__bf16+hidden_dequant__f16+gemm1_operands__bf16+gemm1_operands__f16+gemm1_output__bf16+gemm1_output__f16+swiglu_input__bf16+swiglu_input__f16+swiglu_output__bf16+swiglu_output__f16+gemm2_operands__bf16 | cumulative | 1a4c6ba1 | 901 | 0.938730 | 0.853108 | nan | nan | pass |
| hidden_dequant__bf16+hidden_dequant__f16+gemm1_operands__bf16+gemm1_operands__f16+gemm1_output__bf16+gemm1_output__f16+swiglu_input__bf16+swiglu_input__f16+swiglu_output__bf16+swiglu_output__f16+gemm2_operands__bf16 | cumulative | a7c2bcfd | 16 | 0.872628 | 0.796413 | nan | nan | global_drift |
| hidden_dequant__bf16+hidden_dequant__f16+gemm1_operands__bf16+gemm1_operands__f16+gemm1_output__bf16+gemm1_output__f16+swiglu_input__bf16+swiglu_input__f16+swiglu_output__bf16+swiglu_output__f16+gemm2_operands__bf16 | cumulative | 2e69caee | 15 | 0.998614 | 0.957478 | 2.0480e+03 | 4.9918e+01 | pass |
| hidden_dequant__bf16+hidden_dequant__f16+gemm1_operands__bf16+gemm1_operands__f16+gemm1_output__bf16+gemm1_output__f16+swiglu_input__bf16+swiglu_input__f16+swiglu_output__bf16+swiglu_output__f16+gemm2_operands__bf16 | cumulative | 8cba5890 | 14 | 0.997937 | 0.931491 | 2.0480e+03 | 4.1308e+01 | pass |
| hidden_dequant__bf16+hidden_dequant__f16+gemm1_operands__bf16+gemm1_operands__f16+gemm1_output__bf16+gemm1_output__f16+swiglu_input__bf16+swiglu_input__f16+swiglu_output__bf16+swiglu_output__f16+gemm2_operands__bf16 | cumulative | 5e8dc11c | 14107 | 0.958908 | 0.879175 | nan | nan | pass |
| hidden_dequant__bf16+hidden_dequant__f16+gemm1_operands__bf16+gemm1_operands__f16+gemm1_output__bf16+gemm1_output__f16+swiglu_input__bf16+swiglu_input__f16+swiglu_output__bf16+swiglu_output__f16+gemm2_operands__bf16 | cumulative | 58a34f27 | 11948 | 0.964367 | 0.903895 | nan | nan | pass |
| hidden_dequant__bf16+hidden_dequant__f16+gemm1_operands__bf16+gemm1_operands__f16+gemm1_output__bf16+gemm1_output__f16+swiglu_input__bf16+swiglu_input__f16+swiglu_output__bf16+swiglu_output__f16+gemm2_operands__bf16 | cumulative | 5eadab1e | 62 | 0.997523 | 0.926823 | 2.0480e+03 | 5.1100e+02 | pass |
| hidden_dequant__bf16+hidden_dequant__f16+gemm1_operands__bf16+gemm1_operands__f16+gemm1_output__bf16+gemm1_output__f16+swiglu_input__bf16+swiglu_input__f16+swiglu_output__bf16+swiglu_output__f16+gemm2_operands__bf16 | cumulative | eedc63b2 | 59 | 0.972692 | 0.918595 | nan | nan | pass |
| hidden_dequant__bf16+hidden_dequant__f16+gemm1_operands__bf16+gemm1_operands__f16+gemm1_output__bf16+gemm1_output__f16+swiglu_input__bf16+swiglu_input__f16+swiglu_output__bf16+swiglu_output__f16+gemm2_operands__bf16 | cumulative | e626d3e6 | 58 | 0.945649 | 0.864751 | nan | nan | pass |
| hidden_dequant__bf16+hidden_dequant__f16+gemm1_operands__bf16+gemm1_operands__f16+gemm1_output__bf16+gemm1_output__f16+swiglu_input__bf16+swiglu_input__f16+swiglu_output__bf16+swiglu_output__f16+gemm2_operands__bf16 | cumulative | 74d7ff04 | 57 | 0.945406 | 0.882051 | nan | nan | pass |
| hidden_dequant__bf16+hidden_dequant__f16+gemm1_operands__bf16+gemm1_operands__f16+gemm1_output__bf16+gemm1_output__f16+swiglu_input__bf16+swiglu_input__f16+swiglu_output__bf16+swiglu_output__f16+gemm2_operands__bf16 | cumulative | 4822167c | 56 | 0.970367 | 0.884028 | nan | nan | pass |
| hidden_dequant__bf16+hidden_dequant__f16+gemm1_operands__bf16+gemm1_operands__f16+gemm1_output__bf16+gemm1_output__f16+swiglu_input__bf16+swiglu_input__f16+swiglu_output__bf16+swiglu_output__f16+gemm2_operands__bf16 | cumulative | 81955b1e | 55 | 0.979454 | 0.910428 | nan | nan | pass |
| hidden_dequant__bf16+hidden_dequant__f16+gemm1_operands__bf16+gemm1_operands__f16+gemm1_output__bf16+gemm1_output__f16+swiglu_input__bf16+swiglu_input__f16+swiglu_output__bf16+swiglu_output__f16+gemm2_operands__bf16 | cumulative | 76010cb4 | 54 | 0.979448 | 0.923503 | nan | nan | pass |
| hidden_dequant__bf16+hidden_dequant__f16+gemm1_operands__bf16+gemm1_operands__f16+gemm1_output__bf16+gemm1_output__f16+swiglu_input__bf16+swiglu_input__f16+swiglu_output__bf16+swiglu_output__f16+gemm2_operands__bf16 | cumulative | fc378037 | 53 | 0.932238 | 0.875790 | nan | nan | pass |
| hidden_dequant__bf16+hidden_dequant__f16+gemm1_operands__bf16+gemm1_operands__f16+gemm1_output__bf16+gemm1_output__f16+swiglu_input__bf16+swiglu_input__f16+swiglu_output__bf16+swiglu_output__f16+gemm2_operands__bf16 | cumulative | f7d6ac7c | 52 | 0.979345 | 0.939233 | nan | nan | pass |
| hidden_dequant__bf16+hidden_dequant__f16+gemm1_operands__bf16+gemm1_operands__f16+gemm1_output__bf16+gemm1_output__f16+swiglu_input__bf16+swiglu_input__f16+swiglu_output__bf16+swiglu_output__f16+gemm2_operands__f16 | cumulative | b8f4f012 | 7 | 0.857043 | 0.851921 | nan | nan | global_drift |
| hidden_dequant__bf16+hidden_dequant__f16+gemm1_operands__bf16+gemm1_operands__f16+gemm1_output__bf16+gemm1_output__f16+swiglu_input__bf16+swiglu_input__f16+swiglu_output__bf16+swiglu_output__f16+gemm2_operands__f16 | cumulative | e05c6c03 | 1 | 0.000837 | 0.000837 | nan | nan | catastrophic_saturation |
| hidden_dequant__bf16+hidden_dequant__f16+gemm1_operands__bf16+gemm1_operands__f16+gemm1_output__bf16+gemm1_output__f16+swiglu_input__bf16+swiglu_input__f16+swiglu_output__bf16+swiglu_output__f16+gemm2_operands__f16 | cumulative | 6230e838 | 32 | 0.968310 | 0.954346 | nan | nan | pass |
| hidden_dequant__bf16+hidden_dequant__f16+gemm1_operands__bf16+gemm1_operands__f16+gemm1_output__bf16+gemm1_output__f16+swiglu_input__bf16+swiglu_input__f16+swiglu_output__bf16+swiglu_output__f16+gemm2_operands__f16 | cumulative | 8f1ff9f1 | 80 | 0.942193 | 0.922695 | nan | nan | pass |
| hidden_dequant__bf16+hidden_dequant__f16+gemm1_operands__bf16+gemm1_operands__f16+gemm1_output__bf16+gemm1_output__f16+swiglu_input__bf16+swiglu_input__f16+swiglu_output__bf16+swiglu_output__f16+gemm2_operands__f16 | cumulative | 1a4c6ba1 | 901 | 0.950473 | 0.929073 | nan | nan | pass |
| hidden_dequant__bf16+hidden_dequant__f16+gemm1_operands__bf16+gemm1_operands__f16+gemm1_output__bf16+gemm1_output__f16+swiglu_input__bf16+swiglu_input__f16+swiglu_output__bf16+swiglu_output__f16+gemm2_operands__f16 | cumulative | a7c2bcfd | 16 | 0.999233 | 0.975621 | 2.0480e+03 | 3.2062e+01 | pass |
| hidden_dequant__bf16+hidden_dequant__f16+gemm1_operands__bf16+gemm1_operands__f16+gemm1_output__bf16+gemm1_output__f16+swiglu_input__bf16+swiglu_input__f16+swiglu_output__bf16+swiglu_output__f16+gemm2_operands__f16 | cumulative | 2e69caee | 15 | 0.966797 | 0.959477 | nan | nan | pass |
| hidden_dequant__bf16+hidden_dequant__f16+gemm1_operands__bf16+gemm1_operands__f16+gemm1_output__bf16+gemm1_output__f16+swiglu_input__bf16+swiglu_input__f16+swiglu_output__bf16+swiglu_output__f16+gemm2_operands__f16 | cumulative | 8cba5890 | 14 | 0.928183 | 0.915667 | nan | nan | pass |
| hidden_dequant__bf16+hidden_dequant__f16+gemm1_operands__bf16+gemm1_operands__f16+gemm1_output__bf16+gemm1_output__f16+swiglu_input__bf16+swiglu_input__f16+swiglu_output__bf16+swiglu_output__f16+gemm2_operands__f16 | cumulative | 5e8dc11c | 14107 | 0.956692 | 0.936978 | nan | nan | pass |
| hidden_dequant__bf16+hidden_dequant__f16+gemm1_operands__bf16+gemm1_operands__f16+gemm1_output__bf16+gemm1_output__f16+swiglu_input__bf16+swiglu_input__f16+swiglu_output__bf16+swiglu_output__f16+gemm2_operands__f16 | cumulative | 58a34f27 | 11948 | 0.963149 | 0.948186 | nan | nan | pass |
| hidden_dequant__bf16+hidden_dequant__f16+gemm1_operands__bf16+gemm1_operands__f16+gemm1_output__bf16+gemm1_output__f16+swiglu_input__bf16+swiglu_input__f16+swiglu_output__bf16+swiglu_output__f16+gemm2_operands__f16 | cumulative | 5eadab1e | 62 | 0.963080 | 0.948199 | nan | nan | pass |
| hidden_dequant__bf16+hidden_dequant__f16+gemm1_operands__bf16+gemm1_operands__f16+gemm1_output__bf16+gemm1_output__f16+swiglu_input__bf16+swiglu_input__f16+swiglu_output__bf16+swiglu_output__f16+gemm2_operands__f16 | cumulative | eedc63b2 | 59 | 0.982611 | 0.968788 | nan | nan | pass |
| hidden_dequant__bf16+hidden_dequant__f16+gemm1_operands__bf16+gemm1_operands__f16+gemm1_output__bf16+gemm1_output__f16+swiglu_input__bf16+swiglu_input__f16+swiglu_output__bf16+swiglu_output__f16+gemm2_operands__f16 | cumulative | e626d3e6 | 58 | 0.964841 | 0.944389 | nan | nan | pass |
| hidden_dequant__bf16+hidden_dequant__f16+gemm1_operands__bf16+gemm1_operands__f16+gemm1_output__bf16+gemm1_output__f16+swiglu_input__bf16+swiglu_input__f16+swiglu_output__bf16+swiglu_output__f16+gemm2_operands__f16 | cumulative | 74d7ff04 | 57 | 0.938371 | 0.923466 | nan | nan | pass |
| hidden_dequant__bf16+hidden_dequant__f16+gemm1_operands__bf16+gemm1_operands__f16+gemm1_output__bf16+gemm1_output__f16+swiglu_input__bf16+swiglu_input__f16+swiglu_output__bf16+swiglu_output__f16+gemm2_operands__f16 | cumulative | 4822167c | 56 | 0.945721 | 0.924795 | nan | nan | pass |
| hidden_dequant__bf16+hidden_dequant__f16+gemm1_operands__bf16+gemm1_operands__f16+gemm1_output__bf16+gemm1_output__f16+swiglu_input__bf16+swiglu_input__f16+swiglu_output__bf16+swiglu_output__f16+gemm2_operands__f16 | cumulative | 81955b1e | 55 | 0.944960 | 0.929452 | nan | nan | pass |
| hidden_dequant__bf16+hidden_dequant__f16+gemm1_operands__bf16+gemm1_operands__f16+gemm1_output__bf16+gemm1_output__f16+swiglu_input__bf16+swiglu_input__f16+swiglu_output__bf16+swiglu_output__f16+gemm2_operands__f16 | cumulative | 76010cb4 | 54 | 0.999452 | 0.984799 | 4.0960e+03 | 4.6082e+01 | pass |
| hidden_dequant__bf16+hidden_dequant__f16+gemm1_operands__bf16+gemm1_operands__f16+gemm1_output__bf16+gemm1_output__f16+swiglu_input__bf16+swiglu_input__f16+swiglu_output__bf16+swiglu_output__f16+gemm2_operands__f16 | cumulative | fc378037 | 53 | 0.961717 | 0.945989 | nan | nan | pass |
| hidden_dequant__bf16+hidden_dequant__f16+gemm1_operands__bf16+gemm1_operands__f16+gemm1_output__bf16+gemm1_output__f16+swiglu_input__bf16+swiglu_input__f16+swiglu_output__bf16+swiglu_output__f16+gemm2_operands__f16 | cumulative | f7d6ac7c | 52 | 0.999614 | 0.989022 | 2.0480e+03 | 1.0912e+02 | pass |
| hidden_dequant__bf16+hidden_dequant__f16+gemm1_operands__bf16+gemm1_operands__f16+gemm1_output__bf16+gemm1_output__f16+swiglu_input__bf16+swiglu_input__f16+swiglu_output__bf16+swiglu_output__f16+gemm2_accumulator__bf16 | cumulative | b8f4f012 | 7 | 0.784977 | 0.784977 | nan | nan | global_drift |
| hidden_dequant__bf16+hidden_dequant__f16+gemm1_operands__bf16+gemm1_operands__f16+gemm1_output__bf16+gemm1_output__f16+swiglu_input__bf16+swiglu_input__f16+swiglu_output__bf16+swiglu_output__f16+gemm2_accumulator__bf16 | cumulative | e05c6c03 | 1 | 0.991350 | 0.743862 | 3.5840e+03 | 1.1916e+01 | pass |
| hidden_dequant__bf16+hidden_dequant__f16+gemm1_operands__bf16+gemm1_operands__f16+gemm1_output__bf16+gemm1_output__f16+swiglu_input__bf16+swiglu_input__f16+swiglu_output__bf16+swiglu_output__f16+gemm2_accumulator__bf16 | cumulative | 6230e838 | 32 | 0.934984 | 0.853415 | nan | nan | pass |
| hidden_dequant__bf16+hidden_dequant__f16+gemm1_operands__bf16+gemm1_operands__f16+gemm1_output__bf16+gemm1_output__f16+swiglu_input__bf16+swiglu_input__f16+swiglu_output__bf16+swiglu_output__f16+gemm2_accumulator__bf16 | cumulative | 8f1ff9f1 | 80 | 0.946205 | 0.820715 | nan | nan | pass |
| hidden_dequant__bf16+hidden_dequant__f16+gemm1_operands__bf16+gemm1_operands__f16+gemm1_output__bf16+gemm1_output__f16+swiglu_input__bf16+swiglu_input__f16+swiglu_output__bf16+swiglu_output__f16+gemm2_accumulator__bf16 | cumulative | 1a4c6ba1 | 901 | 0.962409 | 0.825236 | nan | nan | pass |
| hidden_dequant__bf16+hidden_dequant__f16+gemm1_operands__bf16+gemm1_operands__f16+gemm1_output__bf16+gemm1_output__f16+swiglu_input__bf16+swiglu_input__f16+swiglu_output__bf16+swiglu_output__f16+gemm2_accumulator__bf16 | cumulative | a7c2bcfd | 16 | 0.933986 | 0.810425 | nan | nan | pass |
| hidden_dequant__bf16+hidden_dequant__f16+gemm1_operands__bf16+gemm1_operands__f16+gemm1_output__bf16+gemm1_output__f16+swiglu_input__bf16+swiglu_input__f16+swiglu_output__bf16+swiglu_output__f16+gemm2_accumulator__bf16 | cumulative | 2e69caee | 15 | 0.998214 | 0.937193 | 4.0960e+03 | 2.6827e+01 | pass |
| hidden_dequant__bf16+hidden_dequant__f16+gemm1_operands__bf16+gemm1_operands__f16+gemm1_output__bf16+gemm1_output__f16+swiglu_input__bf16+swiglu_input__f16+swiglu_output__bf16+swiglu_output__f16+gemm2_accumulator__bf16 | cumulative | 8cba5890 | 14 | 0.925711 | 0.839176 | nan | nan | pass |
| hidden_dequant__bf16+hidden_dequant__f16+gemm1_operands__bf16+gemm1_operands__f16+gemm1_output__bf16+gemm1_output__f16+swiglu_input__bf16+swiglu_input__f16+swiglu_output__bf16+swiglu_output__f16+gemm2_accumulator__bf16 | cumulative | 5e8dc11c | 14107 | 0.954766 | 0.838380 | nan | nan | pass |
| hidden_dequant__bf16+hidden_dequant__f16+gemm1_operands__bf16+gemm1_operands__f16+gemm1_output__bf16+gemm1_output__f16+swiglu_input__bf16+swiglu_input__f16+swiglu_output__bf16+swiglu_output__f16+gemm2_accumulator__bf16 | cumulative | 58a34f27 | 11948 | 0.964445 | 0.875118 | nan | nan | pass |
| hidden_dequant__bf16+hidden_dequant__f16+gemm1_operands__bf16+gemm1_operands__f16+gemm1_output__bf16+gemm1_output__f16+swiglu_input__bf16+swiglu_input__f16+swiglu_output__bf16+swiglu_output__f16+gemm2_accumulator__bf16 | cumulative | 5eadab1e | 62 | 0.964860 | 0.869575 | nan | nan | pass |
| hidden_dequant__bf16+hidden_dequant__f16+gemm1_operands__bf16+gemm1_operands__f16+gemm1_output__bf16+gemm1_output__f16+swiglu_input__bf16+swiglu_input__f16+swiglu_output__bf16+swiglu_output__f16+gemm2_accumulator__bf16 | cumulative | eedc63b2 | 59 | 0.963834 | 0.887130 | nan | nan | pass |
| hidden_dequant__bf16+hidden_dequant__f16+gemm1_operands__bf16+gemm1_operands__f16+gemm1_output__bf16+gemm1_output__f16+swiglu_input__bf16+swiglu_input__f16+swiglu_output__bf16+swiglu_output__f16+gemm2_accumulator__bf16 | cumulative | e626d3e6 | 58 | 0.927470 | 0.815879 | nan | nan | pass |
| hidden_dequant__bf16+hidden_dequant__f16+gemm1_operands__bf16+gemm1_operands__f16+gemm1_output__bf16+gemm1_output__f16+swiglu_input__bf16+swiglu_input__f16+swiglu_output__bf16+swiglu_output__f16+gemm2_accumulator__bf16 | cumulative | 74d7ff04 | 57 | 0.944329 | 0.846486 | nan | nan | pass |
| hidden_dequant__bf16+hidden_dequant__f16+gemm1_operands__bf16+gemm1_operands__f16+gemm1_output__bf16+gemm1_output__f16+swiglu_input__bf16+swiglu_input__f16+swiglu_output__bf16+swiglu_output__f16+gemm2_accumulator__bf16 | cumulative | 4822167c | 56 | 0.969273 | 0.840427 | nan | nan | pass |
| hidden_dequant__bf16+hidden_dequant__f16+gemm1_operands__bf16+gemm1_operands__f16+gemm1_output__bf16+gemm1_output__f16+swiglu_input__bf16+swiglu_input__f16+swiglu_output__bf16+swiglu_output__f16+gemm2_accumulator__bf16 | cumulative | 81955b1e | 55 | 0.933619 | 0.844752 | nan | nan | pass |
| hidden_dequant__bf16+hidden_dequant__f16+gemm1_operands__bf16+gemm1_operands__f16+gemm1_output__bf16+gemm1_output__f16+swiglu_input__bf16+swiglu_input__f16+swiglu_output__bf16+swiglu_output__f16+gemm2_accumulator__bf16 | cumulative | 76010cb4 | 54 | 0.997357 | 0.911611 | 6.1440e+03 | 1.6302e+04 | pass |
| hidden_dequant__bf16+hidden_dequant__f16+gemm1_operands__bf16+gemm1_operands__f16+gemm1_output__bf16+gemm1_output__f16+swiglu_input__bf16+swiglu_input__f16+swiglu_output__bf16+swiglu_output__f16+gemm2_accumulator__bf16 | cumulative | fc378037 | 53 | 0.922062 | 0.838383 | nan | nan | pass |
| hidden_dequant__bf16+hidden_dequant__f16+gemm1_operands__bf16+gemm1_operands__f16+gemm1_output__bf16+gemm1_output__f16+swiglu_input__bf16+swiglu_input__f16+swiglu_output__bf16+swiglu_output__f16+gemm2_accumulator__bf16 | cumulative | f7d6ac7c | 52 | 0.960058 | 0.906644 | nan | nan | pass |
| hidden_dequant__bf16+hidden_dequant__f16+gemm1_operands__bf16+gemm1_operands__f16+gemm1_output__bf16+gemm1_output__f16+swiglu_input__bf16+swiglu_input__f16+swiglu_output__bf16+swiglu_output__f16+out_accumulator__bf16 | cumulative | b8f4f012 | 7 | 0.856904 | 0.849091 | nan | nan | global_drift |
| hidden_dequant__bf16+hidden_dequant__f16+gemm1_operands__bf16+gemm1_operands__f16+gemm1_output__bf16+gemm1_output__f16+swiglu_input__bf16+swiglu_input__f16+swiglu_output__bf16+swiglu_output__f16+out_accumulator__bf16 | cumulative | e05c6c03 | 1 | 0.998605 | 0.922294 | 1.0240e+03 | 3.3452e+00 | pass |
| hidden_dequant__bf16+hidden_dequant__f16+gemm1_operands__bf16+gemm1_operands__f16+gemm1_output__bf16+gemm1_output__f16+swiglu_input__bf16+swiglu_input__f16+swiglu_output__bf16+swiglu_output__f16+out_accumulator__bf16 | cumulative | 6230e838 | 32 | 0.936833 | 0.913954 | nan | nan | pass |
| hidden_dequant__bf16+hidden_dequant__f16+gemm1_operands__bf16+gemm1_operands__f16+gemm1_output__bf16+gemm1_output__f16+swiglu_input__bf16+swiglu_input__f16+swiglu_output__bf16+swiglu_output__f16+out_accumulator__bf16 | cumulative | 8f1ff9f1 | 80 | 0.973854 | 0.940208 | nan | nan | pass |
| hidden_dequant__bf16+hidden_dequant__f16+gemm1_operands__bf16+gemm1_operands__f16+gemm1_output__bf16+gemm1_output__f16+swiglu_input__bf16+swiglu_input__f16+swiglu_output__bf16+swiglu_output__f16+out_accumulator__bf16 | cumulative | 1a4c6ba1 | 901 | 0.934944 | 0.900766 | nan | nan | pass |
| hidden_dequant__bf16+hidden_dequant__f16+gemm1_operands__bf16+gemm1_operands__f16+gemm1_output__bf16+gemm1_output__f16+swiglu_input__bf16+swiglu_input__f16+swiglu_output__bf16+swiglu_output__f16+out_accumulator__bf16 | cumulative | a7c2bcfd | 16 | 0.936698 | 0.906887 | nan | nan | pass |
| hidden_dequant__bf16+hidden_dequant__f16+gemm1_operands__bf16+gemm1_operands__f16+gemm1_output__bf16+gemm1_output__f16+swiglu_input__bf16+swiglu_input__f16+swiglu_output__bf16+swiglu_output__f16+out_accumulator__bf16 | cumulative | 2e69caee | 15 | 0.999386 | 0.983529 | 2.0480e+03 | 2.5837e+02 | pass |
| hidden_dequant__bf16+hidden_dequant__f16+gemm1_operands__bf16+gemm1_operands__f16+gemm1_output__bf16+gemm1_output__f16+swiglu_input__bf16+swiglu_input__f16+swiglu_output__bf16+swiglu_output__f16+out_accumulator__bf16 | cumulative | 8cba5890 | 14 | 0.999093 | 0.972875 | 2.0480e+03 | 1.1240e+01 | pass |
| hidden_dequant__bf16+hidden_dequant__f16+gemm1_operands__bf16+gemm1_operands__f16+gemm1_output__bf16+gemm1_output__f16+swiglu_input__bf16+swiglu_input__f16+swiglu_output__bf16+swiglu_output__f16+out_accumulator__bf16 | cumulative | 5e8dc11c | 14107 | 0.959388 | 0.930467 | nan | nan | pass |
| hidden_dequant__bf16+hidden_dequant__f16+gemm1_operands__bf16+gemm1_operands__f16+gemm1_output__bf16+gemm1_output__f16+swiglu_input__bf16+swiglu_input__f16+swiglu_output__bf16+swiglu_output__f16+out_accumulator__bf16 | cumulative | 58a34f27 | 11948 | 0.966321 | 0.944248 | nan | nan | pass |
| hidden_dequant__bf16+hidden_dequant__f16+gemm1_operands__bf16+gemm1_operands__f16+gemm1_output__bf16+gemm1_output__f16+swiglu_input__bf16+swiglu_input__f16+swiglu_output__bf16+swiglu_output__f16+out_accumulator__bf16 | cumulative | 5eadab1e | 62 | 0.983065 | 0.957679 | nan | nan | pass |
| hidden_dequant__bf16+hidden_dequant__f16+gemm1_operands__bf16+gemm1_operands__f16+gemm1_output__bf16+gemm1_output__f16+swiglu_input__bf16+swiglu_input__f16+swiglu_output__bf16+swiglu_output__f16+out_accumulator__bf16 | cumulative | eedc63b2 | 59 | 0.982455 | 0.963264 | nan | nan | pass |
| hidden_dequant__bf16+hidden_dequant__f16+gemm1_operands__bf16+gemm1_operands__f16+gemm1_output__bf16+gemm1_output__f16+swiglu_input__bf16+swiglu_input__f16+swiglu_output__bf16+swiglu_output__f16+out_accumulator__bf16 | cumulative | e626d3e6 | 58 | 0.947299 | 0.915176 | nan | nan | pass |
| hidden_dequant__bf16+hidden_dequant__f16+gemm1_operands__bf16+gemm1_operands__f16+gemm1_output__bf16+gemm1_output__f16+swiglu_input__bf16+swiglu_input__f16+swiglu_output__bf16+swiglu_output__f16+out_accumulator__bf16 | cumulative | 74d7ff04 | 57 | 0.942319 | 0.918226 | nan | nan | pass |
| hidden_dequant__bf16+hidden_dequant__f16+gemm1_operands__bf16+gemm1_operands__f16+gemm1_output__bf16+gemm1_output__f16+swiglu_input__bf16+swiglu_input__f16+swiglu_output__bf16+swiglu_output__f16+out_accumulator__bf16 | cumulative | 4822167c | 56 | 0.963150 | 0.930166 | nan | nan | pass |
| hidden_dequant__bf16+hidden_dequant__f16+gemm1_operands__bf16+gemm1_operands__f16+gemm1_output__bf16+gemm1_output__f16+swiglu_input__bf16+swiglu_input__f16+swiglu_output__bf16+swiglu_output__f16+out_accumulator__bf16 | cumulative | 81955b1e | 55 | 0.944968 | 0.923410 | nan | nan | pass |
| hidden_dequant__bf16+hidden_dequant__f16+gemm1_operands__bf16+gemm1_operands__f16+gemm1_output__bf16+gemm1_output__f16+swiglu_input__bf16+swiglu_input__f16+swiglu_output__bf16+swiglu_output__f16+out_accumulator__bf16 | cumulative | 76010cb4 | 54 | 0.999212 | 0.977263 | 4.0960e+03 | 7.0953e+01 | pass |
| hidden_dequant__bf16+hidden_dequant__f16+gemm1_operands__bf16+gemm1_operands__f16+gemm1_output__bf16+gemm1_output__f16+swiglu_input__bf16+swiglu_input__f16+swiglu_output__bf16+swiglu_output__f16+out_accumulator__bf16 | cumulative | fc378037 | 53 | 0.942659 | 0.919072 | nan | nan | pass |
| hidden_dequant__bf16+hidden_dequant__f16+gemm1_operands__bf16+gemm1_operands__f16+gemm1_output__bf16+gemm1_output__f16+swiglu_input__bf16+swiglu_input__f16+swiglu_output__bf16+swiglu_output__f16+out_accumulator__bf16 | cumulative | f7d6ac7c | 52 | 0.970765 | 0.958746 | nan | nan | pass |
| hidden_dequant__bf16+hidden_dequant__f16+gemm1_operands__bf16+gemm1_operands__f16+gemm1_output__bf16+gemm1_output__f16+swiglu_input__bf16+swiglu_input__f16+swiglu_output__bf16+swiglu_output__f16+gemm1_accumulator__bf16 | cumulative | b8f4f012 | 7 | 0.999581 | 0.989497 | 1.0240e+03 | 8.2843e+02 | pass |
| hidden_dequant__bf16+hidden_dequant__f16+gemm1_operands__bf16+gemm1_operands__f16+gemm1_output__bf16+gemm1_output__f16+swiglu_input__bf16+swiglu_input__f16+swiglu_output__bf16+swiglu_output__f16+gemm1_accumulator__bf16 | cumulative | e05c6c03 | 1 | 0.998465 | 0.966518 | 2.0480e+03 | 1.2755e+01 | pass |
| hidden_dequant__bf16+hidden_dequant__f16+gemm1_operands__bf16+gemm1_operands__f16+gemm1_output__bf16+gemm1_output__f16+swiglu_input__bf16+swiglu_input__f16+swiglu_output__bf16+swiglu_output__f16+gemm1_accumulator__bf16 | cumulative | 6230e838 | 32 | 0.968349 | 0.955231 | nan | nan | pass |
| hidden_dequant__bf16+hidden_dequant__f16+gemm1_operands__bf16+gemm1_operands__f16+gemm1_output__bf16+gemm1_output__f16+swiglu_input__bf16+swiglu_input__f16+swiglu_output__bf16+swiglu_output__f16+gemm1_accumulator__bf16 | cumulative | 8f1ff9f1 | 80 | 0.877499 | 0.861253 | nan | nan | global_drift |
| hidden_dequant__bf16+hidden_dequant__f16+gemm1_operands__bf16+gemm1_operands__f16+gemm1_output__bf16+gemm1_output__f16+swiglu_input__bf16+swiglu_input__f16+swiglu_output__bf16+swiglu_output__f16+gemm1_accumulator__bf16 | cumulative | 1a4c6ba1 | 901 | 0.937621 | 0.917850 | nan | nan | pass |
| hidden_dequant__bf16+hidden_dequant__f16+gemm1_operands__bf16+gemm1_operands__f16+gemm1_output__bf16+gemm1_output__f16+swiglu_input__bf16+swiglu_input__f16+swiglu_output__bf16+swiglu_output__f16+gemm1_accumulator__bf16 | cumulative | a7c2bcfd | 16 | 0.936986 | 0.917916 | nan | nan | pass |
| hidden_dequant__bf16+hidden_dequant__f16+gemm1_operands__bf16+gemm1_operands__f16+gemm1_output__bf16+gemm1_output__f16+swiglu_input__bf16+swiglu_input__f16+swiglu_output__bf16+swiglu_output__f16+gemm1_accumulator__bf16 | cumulative | 2e69caee | 15 | 0.933110 | 0.926116 | nan | nan | pass |
| hidden_dequant__bf16+hidden_dequant__f16+gemm1_operands__bf16+gemm1_operands__f16+gemm1_output__bf16+gemm1_output__f16+swiglu_input__bf16+swiglu_input__f16+swiglu_output__bf16+swiglu_output__f16+gemm1_accumulator__bf16 | cumulative | 8cba5890 | 14 | 0.999522 | 0.984086 | 2.0480e+03 | 9.7078e+01 | pass |
| hidden_dequant__bf16+hidden_dequant__f16+gemm1_operands__bf16+gemm1_operands__f16+gemm1_output__bf16+gemm1_output__f16+swiglu_input__bf16+swiglu_input__f16+swiglu_output__bf16+swiglu_output__f16+gemm1_accumulator__bf16 | cumulative | 5e8dc11c | 14107 | 0.958455 | 0.939917 | nan | nan | pass |
| hidden_dequant__bf16+hidden_dequant__f16+gemm1_operands__bf16+gemm1_operands__f16+gemm1_output__bf16+gemm1_output__f16+swiglu_input__bf16+swiglu_input__f16+swiglu_output__bf16+swiglu_output__f16+gemm1_accumulator__bf16 | cumulative | 58a34f27 | 11948 | 0.968400 | 0.954157 | nan | nan | pass |
| hidden_dequant__bf16+hidden_dequant__f16+gemm1_operands__bf16+gemm1_operands__f16+gemm1_output__bf16+gemm1_output__f16+swiglu_input__bf16+swiglu_input__f16+swiglu_output__bf16+swiglu_output__f16+gemm1_accumulator__bf16 | cumulative | 5eadab1e | 62 | 0.963215 | 0.948526 | nan | nan | pass |
| hidden_dequant__bf16+hidden_dequant__f16+gemm1_operands__bf16+gemm1_operands__f16+gemm1_output__bf16+gemm1_output__f16+swiglu_input__bf16+swiglu_input__f16+swiglu_output__bf16+swiglu_output__f16+gemm1_accumulator__bf16 | cumulative | eedc63b2 | 59 | 0.999596 | 0.985768 | 2.0480e+03 | 6.2907e+01 | pass |
| hidden_dequant__bf16+hidden_dequant__f16+gemm1_operands__bf16+gemm1_operands__f16+gemm1_output__bf16+gemm1_output__f16+swiglu_input__bf16+swiglu_input__f16+swiglu_output__bf16+swiglu_output__f16+gemm1_accumulator__bf16 | cumulative | e626d3e6 | 58 | 0.904744 | 0.888843 | nan | nan | pass |
| hidden_dequant__bf16+hidden_dequant__f16+gemm1_operands__bf16+gemm1_operands__f16+gemm1_output__bf16+gemm1_output__f16+swiglu_input__bf16+swiglu_input__f16+swiglu_output__bf16+swiglu_output__f16+gemm1_accumulator__bf16 | cumulative | 74d7ff04 | 57 | 0.977500 | 0.960822 | nan | nan | pass |
| hidden_dequant__bf16+hidden_dequant__f16+gemm1_operands__bf16+gemm1_operands__f16+gemm1_output__bf16+gemm1_output__f16+swiglu_input__bf16+swiglu_input__f16+swiglu_output__bf16+swiglu_output__f16+gemm1_accumulator__bf16 | cumulative | 4822167c | 56 | 0.910151 | 0.891881 | nan | nan | pass |
| hidden_dequant__bf16+hidden_dequant__f16+gemm1_operands__bf16+gemm1_operands__f16+gemm1_output__bf16+gemm1_output__f16+swiglu_input__bf16+swiglu_input__f16+swiglu_output__bf16+swiglu_output__f16+gemm1_accumulator__bf16 | cumulative | 81955b1e | 55 | 0.926857 | 0.912640 | nan | nan | pass |
| hidden_dequant__bf16+hidden_dequant__f16+gemm1_operands__bf16+gemm1_operands__f16+gemm1_output__bf16+gemm1_output__f16+swiglu_input__bf16+swiglu_input__f16+swiglu_output__bf16+swiglu_output__f16+gemm1_accumulator__bf16 | cumulative | 76010cb4 | 54 | 0.980985 | 0.968096 | nan | nan | pass |
| hidden_dequant__bf16+hidden_dequant__f16+gemm1_operands__bf16+gemm1_operands__f16+gemm1_output__bf16+gemm1_output__f16+swiglu_input__bf16+swiglu_input__f16+swiglu_output__bf16+swiglu_output__f16+gemm1_accumulator__bf16 | cumulative | fc378037 | 53 | 0.980666 | 0.965178 | nan | nan | pass |
| hidden_dequant__bf16+hidden_dequant__f16+gemm1_operands__bf16+gemm1_operands__f16+gemm1_output__bf16+gemm1_output__f16+swiglu_input__bf16+swiglu_input__f16+swiglu_output__bf16+swiglu_output__f16+gemm1_accumulator__bf16 | cumulative | f7d6ac7c | 52 | 0.980458 | 0.971462 | nan | nan | pass |
| hidden_dequant__bf16+hidden_dequant__f16+gemm1_operands__bf16+gemm1_operands__f16+gemm1_output__bf16+gemm1_output__f16+swiglu_input__bf16+swiglu_input__f16+swiglu_output__bf16+swiglu_output__f16+gemm1_accumulator__f16 | cumulative | b8f4f012 | 7 | 0.857083 | 0.851782 | nan | nan | global_drift |
| hidden_dequant__bf16+hidden_dequant__f16+gemm1_operands__bf16+gemm1_operands__f16+gemm1_output__bf16+gemm1_output__f16+swiglu_input__bf16+swiglu_input__f16+swiglu_output__bf16+swiglu_output__f16+gemm1_accumulator__f16 | cumulative | e05c6c03 | 1 | 0.999023 | 0.964704 | 1.0240e+03 | 4.0827e+00 | pass |
| hidden_dequant__bf16+hidden_dequant__f16+gemm1_operands__bf16+gemm1_operands__f16+gemm1_output__bf16+gemm1_output__f16+swiglu_input__bf16+swiglu_input__f16+swiglu_output__bf16+swiglu_output__f16+gemm1_accumulator__f16 | cumulative | 6230e838 | 32 | 0.983616 | 0.970254 | nan | nan | pass |
| hidden_dequant__bf16+hidden_dequant__f16+gemm1_operands__bf16+gemm1_operands__f16+gemm1_output__bf16+gemm1_output__f16+swiglu_input__bf16+swiglu_input__f16+swiglu_output__bf16+swiglu_output__f16+gemm1_accumulator__f16 | cumulative | 8f1ff9f1 | 80 | 0.961905 | 0.942008 | nan | nan | pass |
| hidden_dequant__bf16+hidden_dequant__f16+gemm1_operands__bf16+gemm1_operands__f16+gemm1_output__bf16+gemm1_output__f16+swiglu_input__bf16+swiglu_input__f16+swiglu_output__bf16+swiglu_output__f16+gemm1_accumulator__f16 | cumulative | 1a4c6ba1 | 901 | 0.945966 | 0.925896 | nan | nan | pass |
| hidden_dequant__bf16+hidden_dequant__f16+gemm1_operands__bf16+gemm1_operands__f16+gemm1_output__bf16+gemm1_output__f16+swiglu_input__bf16+swiglu_input__f16+swiglu_output__bf16+swiglu_output__f16+gemm1_accumulator__f16 | cumulative | a7c2bcfd | 16 | 0.936820 | 0.917498 | nan | nan | pass |
| hidden_dequant__bf16+hidden_dequant__f16+gemm1_operands__bf16+gemm1_operands__f16+gemm1_output__bf16+gemm1_output__f16+swiglu_input__bf16+swiglu_input__f16+swiglu_output__bf16+swiglu_output__f16+gemm1_accumulator__f16 | cumulative | 2e69caee | 15 | 0.999554 | 0.990355 | 2.0480e+03 | 2.8496e+01 | pass |
| hidden_dequant__bf16+hidden_dequant__f16+gemm1_operands__bf16+gemm1_operands__f16+gemm1_output__bf16+gemm1_output__f16+swiglu_input__bf16+swiglu_input__f16+swiglu_output__bf16+swiglu_output__f16+gemm1_accumulator__f16 | cumulative | 8cba5890 | 14 | 0.963249 | 0.950305 | nan | nan | pass |
| hidden_dequant__bf16+hidden_dequant__f16+gemm1_operands__bf16+gemm1_operands__f16+gemm1_output__bf16+gemm1_output__f16+swiglu_input__bf16+swiglu_input__f16+swiglu_output__bf16+swiglu_output__f16+gemm1_accumulator__f16 | cumulative | 5e8dc11c | 14107 | 0.958190 | 0.939640 | nan | nan | pass |
| hidden_dequant__bf16+hidden_dequant__f16+gemm1_operands__bf16+gemm1_operands__f16+gemm1_output__bf16+gemm1_output__f16+swiglu_input__bf16+swiglu_input__f16+swiglu_output__bf16+swiglu_output__f16+gemm1_accumulator__f16 | cumulative | 58a34f27 | 11948 | 0.966072 | 0.951964 | nan | nan | pass |
| hidden_dequant__bf16+hidden_dequant__f16+gemm1_operands__bf16+gemm1_operands__f16+gemm1_output__bf16+gemm1_output__f16+swiglu_input__bf16+swiglu_input__f16+swiglu_output__bf16+swiglu_output__f16+gemm1_accumulator__f16 | cumulative | 5eadab1e | 62 | 0.935047 | 0.920658 | nan | nan | pass |
| hidden_dequant__bf16+hidden_dequant__f16+gemm1_operands__bf16+gemm1_operands__f16+gemm1_output__bf16+gemm1_output__f16+swiglu_input__bf16+swiglu_input__f16+swiglu_output__bf16+swiglu_output__f16+gemm1_accumulator__f16 | cumulative | eedc63b2 | 59 | 0.965678 | 0.952983 | nan | nan | pass |
| hidden_dequant__bf16+hidden_dequant__f16+gemm1_operands__bf16+gemm1_operands__f16+gemm1_output__bf16+gemm1_output__f16+swiglu_input__bf16+swiglu_input__f16+swiglu_output__bf16+swiglu_output__f16+gemm1_accumulator__f16 | cumulative | e626d3e6 | 58 | 0.919715 | 0.902834 | nan | nan | pass |
| hidden_dequant__bf16+hidden_dequant__f16+gemm1_operands__bf16+gemm1_operands__f16+gemm1_output__bf16+gemm1_output__f16+swiglu_input__bf16+swiglu_input__f16+swiglu_output__bf16+swiglu_output__f16+gemm1_accumulator__f16 | cumulative | 74d7ff04 | 57 | 0.964349 | 0.948920 | nan | nan | pass |
| hidden_dequant__bf16+hidden_dequant__f16+gemm1_operands__bf16+gemm1_operands__f16+gemm1_output__bf16+gemm1_output__f16+swiglu_input__bf16+swiglu_input__f16+swiglu_output__bf16+swiglu_output__f16+gemm1_accumulator__f16 | cumulative | 4822167c | 56 | 0.981508 | 0.960698 | nan | nan | pass |
| hidden_dequant__bf16+hidden_dequant__f16+gemm1_operands__bf16+gemm1_operands__f16+gemm1_output__bf16+gemm1_output__f16+swiglu_input__bf16+swiglu_input__f16+swiglu_output__bf16+swiglu_output__f16+gemm1_accumulator__f16 | cumulative | 81955b1e | 55 | 0.981293 | 0.965465 | nan | nan | pass |
| hidden_dequant__bf16+hidden_dequant__f16+gemm1_operands__bf16+gemm1_operands__f16+gemm1_output__bf16+gemm1_output__f16+swiglu_input__bf16+swiglu_input__f16+swiglu_output__bf16+swiglu_output__f16+gemm1_accumulator__f16 | cumulative | 76010cb4 | 54 | 0.953239 | 0.941701 | nan | nan | pass |
| hidden_dequant__bf16+hidden_dequant__f16+gemm1_operands__bf16+gemm1_operands__f16+gemm1_output__bf16+gemm1_output__f16+swiglu_input__bf16+swiglu_input__f16+swiglu_output__bf16+swiglu_output__f16+gemm1_accumulator__f16 | cumulative | fc378037 | 53 | 0.943022 | 0.928943 | nan | nan | pass |
| hidden_dequant__bf16+hidden_dequant__f16+gemm1_operands__bf16+gemm1_operands__f16+gemm1_output__bf16+gemm1_output__f16+swiglu_input__bf16+swiglu_input__f16+swiglu_output__bf16+swiglu_output__f16+gemm1_accumulator__f16 | cumulative | f7d6ac7c | 52 | 0.980437 | 0.970695 | nan | nan | pass |