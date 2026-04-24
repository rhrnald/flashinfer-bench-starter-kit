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
| run_stage | contest_panel |
| panel_size | 19 |
| panel_indices | 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18 |
| n_candidates | 9 |
| n_survivors | 9 |
| contest_safe_survivor_count | 9 |
| contest_safe_survivors | gemm1_operands__f16, gemm1_accumulator__f16, gemm2_accumulator__bf16, out_accumulator__fp8__tensor, swiglu_output__fp8__block, swiglu_input__fp8__block, gemm2_operands__fp8__block, hidden_dequant__fp8__block, gemm1_output__fp8__tensor |
| strict_safe_survivor_count | 1 |
| strict_safe_survivors | gemm1_operands__f16 |
| contest_only_survivor_count | 8 |
| contest_only_survivors | hidden_dequant__fp8__block, gemm1_output__fp8__tensor, swiglu_input__fp8__block, swiglu_output__fp8__block, gemm2_operands__fp8__block, gemm2_accumulator__bf16, out_accumulator__fp8__tensor, gemm1_accumulator__f16 |
| promote_top_k_per_stage | 1 |
| kernel_validation_enabled | true |
| kernel_validation_limit | 5 |

## Evidence Scope

| field | value |
|---|---|
| evidence_scope | oracle_only |
| kernel_validated_candidates | - |

## Oracle Survivor Summary

| category | count | candidates |
|---|---:|---|
| contest_safe_oracle_single_stage | 9 | gemm1_operands__f16, gemm1_accumulator__f16, gemm2_accumulator__bf16, out_accumulator__fp8__tensor, swiglu_output__fp8__block, swiglu_input__fp8__block, gemm2_operands__fp8__block, hidden_dequant__fp8__block, gemm1_output__fp8__tensor |
| strict_safe_oracle_single_stage | 1 | gemm1_operands__f16 |
| contest_only_oracle_single_stage | 8 | hidden_dequant__fp8__block, gemm1_output__fp8__tensor, swiglu_input__fp8__block, swiglu_output__fp8__block, gemm2_operands__fp8__block, gemm2_accumulator__bf16, out_accumulator__fp8__tensor, gemm1_accumulator__f16 |

## Stage Summary

| stage | best_contest_safe_mode | scale | worst_matched_contest | worst_matched_strict | worst_rel | contest_status | strict_status | evidence |
|---|---|---|---:|---:|---:|---|---|---|
| gemm1_accumulator | f16 | none | 0.996652 | 0.887974 | 1.1650e+10 | contest_safe | strict_unsafe | oracle_only |
| gemm1_operands | f16 | none | 0.998744 | 0.971401 | 8.8500e+09 | contest_safe | strict_safe | oracle_only |
| gemm1_output | fp8 | tensor | 0.913225 | 0.152204 | 3.8240e+11 | contest_safe | strict_unsafe | oracle_only |
| gemm2_accumulator | bf16 | none | 0.990932 | 0.738560 | 2.6600e+10 | contest_safe | strict_unsafe | oracle_only |
| gemm2_operands | fp8 | block | 0.923549 | 0.172712 | 3.6640e+11 | contest_safe | strict_unsafe | oracle_only |
| hidden_dequant | fp8 | block | 0.916574 | 0.148577 | 4.6080e+11 | contest_safe | strict_unsafe | oracle_only |
| out_accumulator | fp8 | tensor | 0.968192 | 0.167411 | 3.2320e+11 | contest_safe | strict_unsafe | oracle_only |
| swiglu_input | fp8 | block | 0.926200 | 0.171177 | 2.3200e+11 | contest_safe | strict_unsafe | oracle_only |
| swiglu_output | fp8 | block | 0.947545 | 0.243443 | 3.6640e+11 | contest_safe | strict_unsafe | oracle_only |

## Contest-Safe Oracle Frontier

| order | candidate | worst_matched_contest | worst_matched_strict | worst_rel | contest_status | strict_status | evidence | kept |
|---|---|---:|---:|---:|---|---|---|---|
| 1 | hidden_dequant__fp8__block | 0.932896 | 0.189314 | 5.3440e+11 | contest_safe | strict_unsafe | oracle_only | yes |
| 2 | hidden_dequant__fp8__block+gemm1_operands__f16 | 0.921177 | 0.152204 | 7.7005e+05 | contest_safe | strict_unsafe | oracle_only | yes |
| 3 | hidden_dequant__fp8__block+gemm1_operands__f16+gemm1_output__fp8__tensor | 0.901367 | 0.116769 | 2.3101e+06 | contest_safe | strict_unsafe | oracle_only | yes |
| 4 | hidden_dequant__fp8__block+gemm1_operands__f16+gemm1_output__fp8__tensor+swiglu_input__fp8__block | 0.868164 | 0.105887 | 1.7792e+12 | contest_unsafe | strict_unsafe | oracle_only | no |
| 5 | hidden_dequant__fp8__block+gemm1_operands__f16+gemm1_output__fp8__tensor+swiglu_output__fp8__block | 0.881696 | 0.108259 | 5.2800e+11 | contest_unsafe | strict_unsafe | oracle_only | no |
| 6 | hidden_dequant__fp8__block+gemm1_operands__f16+gemm1_output__fp8__tensor+gemm2_operands__fp8__block | 0.876116 | 0.098354 | 1.8000e+11 | contest_unsafe | strict_unsafe | oracle_only | no |
| 7 | hidden_dequant__fp8__block+gemm1_operands__f16+gemm1_output__fp8__tensor+gemm2_accumulator__bf16 | 0.892299 | 0.104353 | 2.9491e+06 | contest_unsafe | strict_unsafe | oracle_only | no |
| 8 | hidden_dequant__fp8__block+gemm1_operands__f16+gemm1_output__fp8__tensor+out_accumulator__fp8__tensor | 0.885882 | 0.100725 | 6.2913e+06 | contest_unsafe | strict_unsafe | oracle_only | no |
| 9 | hidden_dequant__fp8__block+gemm1_operands__f16+gemm1_output__fp8__tensor+gemm1_accumulator__f16 | 0.891741 | 0.132952 | 8.0640e+11 | contest_unsafe | strict_unsafe | oracle_only | no |

## BF16/F16 Margin

| stage | preferred | bf16_contest | f16_contest | bf16_strict | f16_strict | |
|---|---|---:|---:|---:|---:|---|

## Pairwise Summary

| pair | worst_matched_contest | worst_matched_strict | worst_rel | contest_status | strict_status | evidence |
|---|---:|---:|---:|---|---|---|
| gemm1_operands__f16+gemm2_accumulator__bf16 | 0.992048 | 0.739816 | 4.5750e+09 | contest_safe | strict_unsafe | oracle_only |
| gemm1_accumulator__f16+gemm2_accumulator__bf16 | 0.992746 | 0.715402 | 1.9100e+10 | contest_safe | strict_unsafe | oracle_only |
| gemm1_operands__f16+gemm1_accumulator__f16 | 0.999426 | 0.972098 | 3.8750e+09 | contest_safe | strict_safe | oracle_only |

## Stress Summary

| candidate | worst_workload | worst_seq_len | worst_matched_contest | worst_matched_strict | contest_status | strict_status | evidence |
|---|---|---:|---:|---:|---|---|---|
| gemm2_accumulator__bf16 | e05c6c03 | 1 | 0.990932 | 0.738560 | contest_safe | strict_unsafe | oracle_only |

## Promotion Summary

| category | candidates |
|---|---|
| contest_safe_oracle_candidates | gemm1_operands__f16, gemm1_accumulator__f16, gemm2_accumulator__bf16, out_accumulator__fp8__tensor, swiglu_output__fp8__block, swiglu_input__fp8__block, gemm2_operands__fp8__block, hidden_dequant__fp8__block, gemm1_output__fp8__tensor |
| strict_safe_oracle_candidates | gemm1_operands__f16 |
| kernel_validated_candidates | - |
| pairwise_shortlist | gemm1_operands__f16, gemm1_accumulator__f16, gemm2_accumulator__bf16 |

## Kernel Validation Summary

| candidate | validation_status | workloads_passed | total_workloads | worst_max_abs | worst_max_rel | evidence | notes |
|---|---|---:|---:|---:|---:|---|---|
| hidden_dequant__fp8__block | kernel_not_supported | 0 | 19 | - | - | kernel_not_supported | kernel validation helper unavailable in Modal worker: ModuleNotFoundError |
| gemm1_output__fp8__tensor | kernel_not_supported | 0 | 19 | - | - | kernel_not_supported | kernel validation helper unavailable in Modal worker: ModuleNotFoundError |
| swiglu_input__fp8__block | kernel_not_supported | 0 | 19 | - | - | kernel_not_supported | kernel validation helper unavailable in Modal worker: ModuleNotFoundError |
| swiglu_output__fp8__block | kernel_not_supported | 0 | 19 | - | - | kernel_not_supported | kernel validation helper unavailable in Modal worker: ModuleNotFoundError |
| gemm2_operands__fp8__block | kernel_not_supported | 0 | 19 | - | - | kernel_not_supported | kernel validation helper unavailable in Modal worker: ModuleNotFoundError |

## Sampled Results

| candidate | phase | workload | seq_len | matched_contest | matched_strict | max_abs | max_rel | failure |
|---|---|---|---:|---:|---:|---:|---:|---|
| baseline__fp32 | single_stage | b8f4f012 | 7 | 1.000000 | 1.000000 | 0.0000e+00 | 0.0000e+00 | pass |
| hidden_dequant__fp8__block | single_stage | b8f4f012 | 7 | 0.979193 | 0.762197 | 1.5360e+04 | 2.6209e+03 | pass |
| gemm1_operands__f16 | single_stage | b8f4f012 | 7 | 0.999562 | 0.992746 | 2.0480e+03 | 3.6978e+01 | pass |
| gemm1_output__fp8__tensor | single_stage | b8f4f012 | 7 | 0.979094 | 0.762855 | 1.5360e+04 | 3.0730e+03 | pass |
| swiglu_input__fp8__block | single_stage | b8f4f012 | 7 | 0.978555 | 0.763413 | 1.6384e+04 | 1.5305e+04 | pass |
| swiglu_output__fp8__block | single_stage | b8f4f012 | 7 | 0.987105 | 0.788445 | 9.4720e+03 | 5.2448e+03 | pass |
| gemm2_operands__fp8__block | single_stage | b8f4f012 | 7 | 0.979492 | 0.765366 | 1.2544e+04 | 9.2733e+03 | pass |
| gemm2_accumulator__bf16 | single_stage | b8f4f012 | 7 | 0.997608 | 0.929249 | 3.0720e+03 | 7.2680e+02 | pass |
| out_accumulator__fp8__tensor | single_stage | b8f4f012 | 7 | 0.990772 | 0.760563 | 3.4816e+04 | 2.3978e+03 | pass |
| gemm1_accumulator__f16 | single_stage | b8f4f012 | 7 | 0.999083 | 0.971221 | 2.0480e+03 | 9.5242e+01 | pass |
| baseline__fp32 | single_stage | e05c6c03 | 1 | 1.000000 | 1.000000 | 0.0000e+00 | 0.0000e+00 | pass |
| hidden_dequant__fp8__block | single_stage | e05c6c03 | 1 | 0.916574 | 0.148577 | 1.2288e+04 | 1.8726e+03 | pass |
| gemm1_operands__f16 | single_stage | e05c6c03 | 1 | 0.998744 | 0.971401 | 1.0240e+03 | 1.0177e+01 | pass |
| gemm1_output__fp8__tensor | single_stage | e05c6c03 | 1 | 0.913225 | 0.152204 | 1.5872e+04 | 2.8885e+02 | pass |
| swiglu_input__fp8__block | single_stage | e05c6c03 | 1 | 0.926200 | 0.171177 | 1.3312e+04 | 3.9990e+03 | pass |
| swiglu_output__fp8__block | single_stage | e05c6c03 | 1 | 0.947545 | 0.243443 | 9.3120e+03 | 1.2631e+03 | pass |
| gemm2_operands__fp8__block | single_stage | e05c6c03 | 1 | 0.923549 | 0.172712 | 1.4336e+04 | 1.5668e+03 | pass |
| gemm2_accumulator__bf16 | single_stage | e05c6c03 | 1 | 0.990932 | 0.738560 | 2.0480e+03 | 5.9874e+02 | pass |
| out_accumulator__fp8__tensor | single_stage | e05c6c03 | 1 | 0.968192 | 0.167411 | 2.3552e+04 | 2.6357e+02 | pass |
| gemm1_accumulator__f16 | single_stage | e05c6c03 | 1 | 0.996652 | 0.887974 | 1.0240e+03 | 5.1976e+01 | pass |
| baseline__fp32 | single_stage | 6230e838 | 32 | 1.000000 | 1.000000 | 0.0000e+00 | 0.0000e+00 | pass |
| hidden_dequant__fp8__block | single_stage | 6230e838 | 32 | 0.968846 | 0.661298 | 2.0480e+04 | 2.0938e+03 | pass |
| gemm1_operands__f16 | single_stage | 6230e838 | 32 | 0.999647 | 0.988451 | 2.0480e+03 | 3.4169e+01 | pass |
| gemm1_output__fp8__tensor | single_stage | 6230e838 | 32 | 0.967808 | 0.661194 | 2.7648e+04 | 1.2141e+03 | pass |
| swiglu_input__fp8__block | single_stage | 6230e838 | 32 | 0.969753 | 0.663853 | 2.4064e+04 | 1.1510e+03 | pass |
| swiglu_output__fp8__block | single_stage | 6230e838 | 32 | 0.979518 | 0.699345 | 1.3312e+04 | 1.6542e+03 | pass |
| gemm2_operands__fp8__block | single_stage | 6230e838 | 32 | 0.970354 | 0.666395 | 2.2528e+04 | 2.6756e+03 | pass |
| gemm2_accumulator__bf16 | single_stage | 6230e838 | 32 | 0.996857 | 0.900657 | 6.1440e+03 | 1.6540e+02 | pass |
| out_accumulator__fp8__tensor | single_stage | 6230e838 | 32 | 0.988268 | 0.665963 | 4.5056e+04 | 1.0059e+03 | pass |
| gemm1_accumulator__f16 | single_stage | 6230e838 | 32 | 0.998683 | 0.958553 | 4.0960e+03 | 2.5427e+01 | pass |
| baseline__fp32 | single_stage | 8f1ff9f1 | 80 | 1.000000 | 1.000000 | 0.0000e+00 | 0.0000e+00 | pass |
| hidden_dequant__fp8__block | single_stage | 8f1ff9f1 | 80 | 0.953796 | 0.499388 | 2.6624e+04 | 1.3158e+04 | pass |
| gemm1_operands__f16 | single_stage | 8f1ff9f1 | 80 | 0.999479 | 0.983313 | 4.0960e+03 | 7.6899e+01 | pass |
| gemm1_output__fp8__tensor | single_stage | 8f1ff9f1 | 80 | 0.953320 | 0.499808 | 3.0464e+04 | 4.8811e+03 | pass |
| swiglu_input__fp8__block | single_stage | 8f1ff9f1 | 80 | 0.954124 | 0.502963 | 2.6624e+04 | 4.6928e+03 | pass |
| swiglu_output__fp8__block | single_stage | 8f1ff9f1 | 80 | 0.970979 | 0.556187 | 1.5360e+04 | 3.0297e+03 | pass |
| gemm2_operands__fp8__block | single_stage | 8f1ff9f1 | 80 | 0.956827 | 0.506846 | 2.3936e+04 | 5.9560e+03 | pass |
| gemm2_accumulator__bf16 | single_stage | 8f1ff9f1 | 80 | 0.995846 | 0.862238 | 6.1440e+03 | 1.7284e+03 | pass |
| out_accumulator__fp8__tensor | single_stage | 8f1ff9f1 | 80 | 0.987343 | 0.514802 | 5.1200e+04 | 1.0799e+03 | pass |
| gemm1_accumulator__f16 | single_stage | 8f1ff9f1 | 80 | 0.998052 | 0.937655 | 4.0960e+03 | 5.1817e+02 | pass |
| baseline__fp32 | single_stage | 1a4c6ba1 | 901 | 1.000000 | 1.000000 | 0.0000e+00 | 0.0000e+00 | pass |
| hidden_dequant__fp8__block | single_stage | 1a4c6ba1 | 901 | 0.952114 | 0.474334 | 3.6864e+04 | 2.2938e+05 | pass |
| gemm1_operands__f16 | single_stage | 1a4c6ba1 | 901 | 0.999465 | 0.982654 | 4.0960e+03 | 7.1304e+03 | pass |
| gemm1_output__fp8__tensor | single_stage | 1a4c6ba1 | 901 | 0.950562 | 0.473877 | 4.6080e+04 | 1.8125e+05 | pass |
| swiglu_input__fp8__block | single_stage | 1a4c6ba1 | 901 | 0.952063 | 0.477246 | 4.6720e+04 | 5.0322e+05 | pass |
| swiglu_output__fp8__block | single_stage | 1a4c6ba1 | 901 | 0.969435 | 0.533277 | 2.9952e+04 | 2.3991e+05 | pass |
| gemm2_operands__fp8__block | single_stage | 1a4c6ba1 | 901 | 0.954472 | 0.481826 | 3.8912e+04 | 3.0662e+05 | pass |
| gemm2_accumulator__bf16 | single_stage | 1a4c6ba1 | 901 | 0.995523 | 0.852626 | 8.1920e+03 | 5.2077e+04 | pass |
| out_accumulator__fp8__tensor | single_stage | 1a4c6ba1 | 901 | 0.985465 | 0.487116 | 7.3728e+04 | 1.2756e+05 | pass |
| gemm1_accumulator__f16 | single_stage | 1a4c6ba1 | 901 | 0.997929 | 0.935601 | 4.0960e+03 | 2.8525e+04 | pass |
| baseline__fp32 | single_stage | a7c2bcfd | 16 | 1.000000 | 1.000000 | 0.0000e+00 | 0.0000e+00 | pass |
| hidden_dequant__fp8__block | single_stage | a7c2bcfd | 16 | 0.951869 | 0.478777 | 2.0480e+04 | 3.9497e+02 | pass |
| gemm1_operands__f16 | single_stage | a7c2bcfd | 16 | 0.999486 | 0.983128 | 2.0480e+03 | 5.7692e+00 | pass |
| gemm1_output__fp8__tensor | single_stage | a7c2bcfd | 16 | 0.951024 | 0.480835 | 2.4576e+04 | 3.7315e+02 | pass |
| swiglu_input__fp8__block | single_stage | a7c2bcfd | 16 | 0.950169 | 0.480948 | 2.0992e+04 | 9.8362e+02 | pass |
| swiglu_output__fp8__block | single_stage | a7c2bcfd | 16 | 0.969535 | 0.539621 | 1.4848e+04 | 8.6287e+02 | pass |
| gemm2_operands__fp8__block | single_stage | a7c2bcfd | 16 | 0.953953 | 0.485308 | 2.2528e+04 | 1.3344e+03 | pass |
| gemm2_accumulator__bf16 | single_stage | a7c2bcfd | 16 | 0.996172 | 0.860997 | 8.1920e+03 | 8.4733e+01 | pass |
| out_accumulator__fp8__tensor | single_stage | a7c2bcfd | 16 | 0.990226 | 0.501081 | 4.0960e+04 | 1.8253e+02 | pass |
| gemm1_accumulator__f16 | single_stage | a7c2bcfd | 16 | 0.997977 | 0.935782 | 2.0480e+03 | 2.9662e+01 | pass |
| baseline__fp32 | single_stage | 2e69caee | 15 | 1.000000 | 1.000000 | 0.0000e+00 | 0.0000e+00 | pass |
| hidden_dequant__fp8__block | single_stage | 2e69caee | 15 | 0.979474 | 0.775028 | 1.6768e+04 | 1.8632e+02 | pass |
| gemm1_operands__f16 | single_stage | 2e69caee | 15 | 0.999805 | 0.992587 | 2.0480e+03 | 3.6761e+00 | pass |
| gemm1_output__fp8__tensor | single_stage | 2e69caee | 15 | 0.979436 | 0.777362 | 1.7920e+04 | 3.1542e+02 | pass |
| swiglu_input__fp8__block | single_stage | 2e69caee | 15 | 0.979976 | 0.779334 | 1.7264e+04 | 2.5320e+02 | pass |
| swiglu_output__fp8__block | single_stage | 2e69caee | 15 | 0.987240 | 0.802409 | 1.3312e+04 | 1.5449e+02 | pass |
| gemm2_operands__fp8__block | single_stage | 2e69caee | 15 | 0.980813 | 0.780199 | 2.0480e+04 | 2.4211e+02 | pass |
| gemm2_accumulator__bf16 | single_stage | 2e69caee | 15 | 0.998270 | 0.938560 | 4.0960e+03 | 1.1648e+01 | pass |
| out_accumulator__fp8__tensor | single_stage | 2e69caee | 15 | 0.994224 | 0.783417 | 3.0720e+04 | 4.4932e+01 | pass |
| gemm1_accumulator__f16 | single_stage | 2e69caee | 15 | 0.999358 | 0.972805 | 2.0480e+03 | 1.2606e+01 | pass |
| baseline__fp32 | single_stage | 8cba5890 | 14 | 1.000000 | 1.000000 | 0.0000e+00 | 0.0000e+00 | pass |
| hidden_dequant__fp8__block | single_stage | 8cba5890 | 14 | 0.966916 | 0.642748 | 1.4336e+04 | 1.6327e+03 | pass |
| gemm1_operands__f16 | single_stage | 8cba5890 | 14 | 0.999502 | 0.987634 | 2.0480e+03 | 4.3397e+01 | pass |
| gemm1_output__fp8__tensor | single_stage | 8cba5890 | 14 | 0.965691 | 0.643834 | 1.7824e+04 | 3.3020e+03 | pass |
| swiglu_input__fp8__block | single_stage | 8cba5890 | 14 | 0.966438 | 0.643595 | 1.4336e+04 | 2.4033e+03 | pass |
| swiglu_output__fp8__block | single_stage | 8cba5890 | 14 | 0.978555 | 0.684112 | 1.0240e+04 | 7.9492e+02 | pass |
| gemm2_operands__fp8__block | single_stage | 8cba5890 | 14 | 0.968909 | 0.648477 | 1.3776e+04 | 1.2778e+03 | pass |
| gemm2_accumulator__bf16 | single_stage | 8cba5890 | 14 | 0.996393 | 0.897889 | 4.0960e+03 | 1.7341e+02 | pass |
| out_accumulator__fp8__tensor | single_stage | 8cba5890 | 14 | 0.989198 | 0.650540 | 2.7648e+04 | 4.9356e+02 | pass |
| gemm1_accumulator__f16 | single_stage | 8cba5890 | 14 | 0.998236 | 0.954092 | 2.0480e+03 | 5.2038e+01 | pass |
| baseline__fp32 | single_stage | 5e8dc11c | 14107 | 1.000000 | 1.000000 | 0.0000e+00 | 0.0000e+00 | pass |
| hidden_dequant__fp8__block | single_stage | 5e8dc11c | 14107 | 0.956342 | 0.522949 | 4.5056e+04 | 4.6080e+11 | pass |
| gemm1_operands__f16 | single_stage | 5e8dc11c | 14107 | 0.999496 | 0.984198 | 4.0960e+03 | 8.8500e+09 | pass |
| gemm1_output__fp8__tensor | single_stage | 5e8dc11c | 14107 | 0.955228 | 0.523059 | 4.5056e+04 | 3.8240e+11 | pass |
| swiglu_input__fp8__block | single_stage | 5e8dc11c | 14107 | 0.956443 | 0.525656 | 4.0960e+04 | 2.3200e+11 | pass |
| swiglu_output__fp8__block | single_stage | 5e8dc11c | 14107 | 0.972249 | 0.576657 | 3.2768e+04 | 3.6640e+11 | pass |
| gemm2_operands__fp8__block | single_stage | 5e8dc11c | 14107 | 0.958636 | 0.529770 | 4.0448e+04 | 3.6640e+11 | pass |
| gemm2_accumulator__bf16 | single_stage | 5e8dc11c | 14107 | 0.996198 | 0.871490 | 1.0240e+04 | 2.6600e+10 | pass |
| out_accumulator__fp8__tensor | single_stage | 5e8dc11c | 14107 | 0.989692 | 0.540876 | 1.4746e+05 | 3.2320e+11 | pass |
| gemm1_accumulator__f16 | single_stage | 5e8dc11c | 14107 | 0.998124 | 0.941684 | 4.0960e+03 | 1.1650e+10 | pass |
| baseline__fp32 | single_stage | 58a34f27 | 11948 | 1.000000 | 1.000000 | 0.0000e+00 | 0.0000e+00 | pass |
| hidden_dequant__fp8__block | single_stage | 58a34f27 | 11948 | 0.966532 | 0.634486 | 3.8912e+04 | 2.4759e+06 | pass |
| gemm1_operands__f16 | single_stage | 58a34f27 | 11948 | 0.999617 | 0.987872 | 4.0960e+03 | 6.8696e+03 | pass |
| gemm1_output__fp8__tensor | single_stage | 58a34f27 | 11948 | 0.965666 | 0.634549 | 4.1984e+04 | 2.8277e+06 | pass |
| swiglu_input__fp8__block | single_stage | 58a34f27 | 11948 | 0.966576 | 0.636483 | 3.6864e+04 | 9.2979e+05 | pass |
| swiglu_output__fp8__block | single_stage | 58a34f27 | 11948 | 0.978679 | 0.675613 | 2.8672e+04 | 5.1063e+05 | pass |
| gemm2_operands__fp8__block | single_stage | 58a34f27 | 11948 | 0.968267 | 0.639728 | 4.3008e+04 | 5.3794e+05 | pass |
| gemm2_accumulator__bf16 | single_stage | 58a34f27 | 11948 | 0.997081 | 0.902015 | 1.2288e+04 | 5.8367e+04 | pass |
| out_accumulator__fp8__tensor | single_stage | 58a34f27 | 11948 | 0.992020 | 0.647626 | 9.4208e+04 | 2.9696e+05 | pass |
| gemm1_accumulator__f16 | single_stage | 58a34f27 | 11948 | 0.998562 | 0.955148 | 4.0960e+03 | 2.3522e+04 | pass |
| baseline__fp32 | single_stage | 5eadab1e | 62 | 1.000000 | 1.000000 | 0.0000e+00 | 0.0000e+00 | pass |
| hidden_dequant__fp8__block | single_stage | 5eadab1e | 62 | 0.964297 | 0.610408 | 2.4576e+04 | 1.8591e+03 | pass |
| gemm1_operands__f16 | single_stage | 5eadab1e | 62 | 0.999617 | 0.987091 | 2.0480e+03 | 5.7365e+01 | pass |
| gemm1_output__fp8__tensor | single_stage | 5eadab1e | 62 | 0.963343 | 0.609857 | 2.7648e+04 | 1.8153e+03 | pass |
| swiglu_input__fp8__block | single_stage | 5eadab1e | 62 | 0.964211 | 0.611357 | 2.6624e+04 | 1.7636e+03 | pass |
| swiglu_output__fp8__block | single_stage | 5eadab1e | 62 | 0.976992 | 0.653579 | 1.4208e+04 | 1.3553e+03 | pass |
| gemm2_operands__fp8__block | single_stage | 5eadab1e | 62 | 0.966050 | 0.616249 | 2.2528e+04 | 1.6876e+03 | pass |
| gemm2_accumulator__bf16 | single_stage | 5eadab1e | 62 | 0.997021 | 0.896253 | 6.1440e+03 | 1.4850e+02 | pass |
| out_accumulator__fp8__tensor | single_stage | 5eadab1e | 62 | 0.991992 | 0.625720 | 4.7104e+04 | 4.2823e+02 | pass |
| gemm1_accumulator__f16 | single_stage | 5eadab1e | 62 | 0.998517 | 0.952470 | 2.0480e+03 | 8.9583e+01 | pass |
| baseline__fp32 | single_stage | eedc63b2 | 59 | 1.000000 | 1.000000 | 0.0000e+00 | 0.0000e+00 | pass |
| hidden_dequant__fp8__block | single_stage | eedc63b2 | 59 | 0.970147 | 0.674923 | 2.2784e+04 | 1.5133e+04 | pass |
| gemm1_operands__f16 | single_stage | eedc63b2 | 59 | 0.999643 | 0.988962 | 2.0480e+03 | 7.7667e+01 | pass |
| gemm1_output__fp8__tensor | single_stage | eedc63b2 | 59 | 0.969814 | 0.675658 | 2.1536e+04 | 7.9086e+03 | pass |
| swiglu_input__fp8__block | single_stage | eedc63b2 | 59 | 0.970131 | 0.677091 | 1.8432e+04 | 3.1137e+03 | pass |
| swiglu_output__fp8__block | single_stage | eedc63b2 | 59 | 0.980646 | 0.711311 | 1.2288e+04 | 2.1039e+03 | pass |
| gemm2_operands__fp8__block | single_stage | eedc63b2 | 59 | 0.971630 | 0.680037 | 1.8432e+04 | 2.7439e+03 | pass |
| gemm2_accumulator__bf16 | single_stage | eedc63b2 | 59 | 0.997498 | 0.914079 | 6.1440e+03 | 3.3678e+02 | pass |
| out_accumulator__fp8__tensor | single_stage | eedc63b2 | 59 | 0.993696 | 0.687661 | 3.0720e+04 | 1.2221e+03 | pass |
| gemm1_accumulator__f16 | single_stage | eedc63b2 | 59 | 0.998728 | 0.960361 | 2.0480e+03 | 3.2456e+02 | pass |
| baseline__fp32 | single_stage | e626d3e6 | 58 | 1.000000 | 1.000000 | 0.0000e+00 | 0.0000e+00 | pass |
| hidden_dequant__fp8__block | single_stage | e626d3e6 | 58 | 0.955648 | 0.511517 | 2.6624e+04 | 1.9332e+03 | pass |
| gemm1_operands__f16 | single_stage | e626d3e6 | 58 | 0.999476 | 0.983658 | 4.0960e+03 | 1.0490e+02 | pass |
| gemm1_output__fp8__tensor | single_stage | e626d3e6 | 58 | 0.954241 | 0.511079 | 3.3152e+04 | 6.5113e+03 | pass |
| swiglu_input__fp8__block | single_stage | e626d3e6 | 58 | 0.955566 | 0.513597 | 2.7648e+04 | 3.2277e+03 | pass |
| swiglu_output__fp8__block | single_stage | e626d3e6 | 58 | 0.971531 | 0.565038 | 1.5360e+04 | 1.7175e+03 | pass |
| gemm2_operands__fp8__block | single_stage | e626d3e6 | 58 | 0.957661 | 0.517713 | 2.4576e+04 | 3.8023e+03 | pass |
| gemm2_accumulator__bf16 | single_stage | e626d3e6 | 58 | 0.995711 | 0.862160 | 6.1440e+03 | 5.4783e+02 | pass |
| out_accumulator__fp8__tensor | single_stage | e626d3e6 | 58 | 0.986612 | 0.523514 | 5.7344e+04 | 1.3117e+03 | pass |
| gemm1_accumulator__f16 | single_stage | e626d3e6 | 58 | 0.998085 | 0.940656 | 4.0960e+03 | 1.5602e+02 | pass |
| baseline__fp32 | single_stage | 74d7ff04 | 57 | 1.000000 | 1.000000 | 0.0000e+00 | 0.0000e+00 | pass |
| hidden_dequant__fp8__block | single_stage | 74d7ff04 | 57 | 0.963064 | 0.589540 | 2.4576e+04 | 2.0057e+04 | pass |
| gemm1_operands__f16 | single_stage | 74d7ff04 | 57 | 0.999621 | 0.986507 | 2.0480e+03 | 1.4219e+02 | pass |
| gemm1_output__fp8__tensor | single_stage | 74d7ff04 | 57 | 0.961403 | 0.590025 | 2.5600e+04 | 2.7008e+04 | pass |
| swiglu_input__fp8__block | single_stage | 74d7ff04 | 57 | 0.962844 | 0.592279 | 2.4576e+04 | 1.6780e+04 | pass |
| swiglu_output__fp8__block | single_stage | 74d7ff04 | 57 | 0.976489 | 0.636190 | 1.6384e+04 | 7.2000e+03 | pass |
| gemm2_operands__fp8__block | single_stage | 74d7ff04 | 57 | 0.964699 | 0.595781 | 2.2528e+04 | 7.0511e+03 | pass |
| gemm2_accumulator__bf16 | single_stage | 74d7ff04 | 57 | 0.996801 | 0.889754 | 6.1440e+03 | 1.3539e+03 | pass |
| out_accumulator__fp8__tensor | single_stage | 74d7ff04 | 57 | 0.990770 | 0.604940 | 4.0960e+04 | 6.4057e+03 | pass |
| gemm1_accumulator__f16 | single_stage | 74d7ff04 | 57 | 0.998480 | 0.949050 | 4.0960e+03 | 1.8951e+02 | pass |
| baseline__fp32 | single_stage | 4822167c | 56 | 1.000000 | 1.000000 | 0.0000e+00 | 0.0000e+00 | pass |
| hidden_dequant__fp8__block | single_stage | 4822167c | 56 | 0.953895 | 0.494093 | 1.9968e+04 | 2.7869e+03 | pass |
| gemm1_operands__f16 | single_stage | 4822167c | 56 | 0.999452 | 0.983525 | 2.0480e+03 | 2.9126e+01 | pass |
| gemm1_output__fp8__tensor | single_stage | 4822167c | 56 | 0.953125 | 0.495625 | 2.2528e+04 | 4.4670e+03 | pass |
| swiglu_input__fp8__block | single_stage | 4822167c | 56 | 0.953658 | 0.496841 | 1.9456e+04 | 3.3168e+03 | pass |
| swiglu_output__fp8__block | single_stage | 4822167c | 56 | 0.971107 | 0.551678 | 1.2288e+04 | 2.0094e+03 | pass |
| gemm2_operands__fp8__block | single_stage | 4822167c | 56 | 0.956511 | 0.501188 | 2.0736e+04 | 5.0992e+03 | pass |
| gemm2_accumulator__bf16 | single_stage | 4822167c | 56 | 0.995865 | 0.860374 | 4.6080e+03 | 5.9412e+02 | pass |
| out_accumulator__fp8__tensor | single_stage | 4822167c | 56 | 0.988022 | 0.510742 | 4.0960e+04 | 9.4920e+02 | pass |
| gemm1_accumulator__f16 | single_stage | 4822167c | 56 | 0.997922 | 0.937767 | 4.0960e+03 | 1.3856e+02 | pass |
| baseline__fp32 | single_stage | 81955b1e | 55 | 1.000000 | 1.000000 | 0.0000e+00 | 0.0000e+00 | pass |
| hidden_dequant__fp8__block | single_stage | 81955b1e | 55 | 0.963606 | 0.605246 | 2.5600e+04 | 1.2796e+04 | pass |
| gemm1_operands__f16 | single_stage | 81955b1e | 55 | 0.999566 | 0.986665 | 2.0480e+03 | 3.9000e+01 | pass |
| gemm1_output__fp8__tensor | single_stage | 81955b1e | 55 | 0.962880 | 0.606169 | 2.1504e+04 | 2.6709e+03 | pass |
| swiglu_input__fp8__block | single_stage | 81955b1e | 55 | 0.964697 | 0.608995 | 2.4576e+04 | 2.1066e+04 | pass |
| swiglu_output__fp8__block | single_stage | 81955b1e | 55 | 0.977118 | 0.650132 | 1.4480e+04 | 5.1873e+03 | pass |
| gemm2_operands__fp8__block | single_stage | 81955b1e | 55 | 0.966115 | 0.611917 | 2.2528e+04 | 6.9837e+03 | pass |
| gemm2_accumulator__bf16 | single_stage | 81955b1e | 55 | 0.996713 | 0.892538 | 4.0960e+03 | 2.1933e+03 | pass |
| out_accumulator__fp8__tensor | single_stage | 81955b1e | 55 | 0.990828 | 0.619191 | 4.0960e+04 | 1.4814e+03 | pass |
| gemm1_accumulator__f16 | single_stage | 81955b1e | 55 | 0.998453 | 0.951448 | 2.0480e+03 | 5.0612e+02 | pass |
| baseline__fp32 | single_stage | 76010cb4 | 54 | 1.000000 | 1.000000 | 0.0000e+00 | 0.0000e+00 | pass |
| hidden_dequant__fp8__block | single_stage | 76010cb4 | 54 | 0.970442 | 0.675913 | 2.2528e+04 | 2.1904e+03 | pass |
| gemm1_operands__f16 | single_stage | 76010cb4 | 54 | 0.999729 | 0.989312 | 2.0480e+03 | 4.9097e+01 | pass |
| gemm1_output__fp8__tensor | single_stage | 76010cb4 | 54 | 0.969652 | 0.675551 | 2.4576e+04 | 1.7091e+03 | pass |
| swiglu_input__fp8__block | single_stage | 76010cb4 | 54 | 0.970437 | 0.677845 | 2.5600e+04 | 1.3200e+03 | pass |
| swiglu_output__fp8__block | single_stage | 76010cb4 | 54 | 0.981285 | 0.711860 | 1.4336e+04 | 1.3640e+03 | pass |
| gemm2_operands__fp8__block | single_stage | 76010cb4 | 54 | 0.971902 | 0.680085 | 2.0480e+04 | 1.4326e+03 | pass |
| gemm2_accumulator__bf16 | single_stage | 76010cb4 | 54 | 0.997411 | 0.912112 | 6.1440e+03 | 4.3532e+02 | pass |
| out_accumulator__fp8__tensor | single_stage | 76010cb4 | 54 | 0.992888 | 0.687291 | 3.6864e+04 | 6.7190e+02 | pass |
| gemm1_accumulator__f16 | single_stage | 76010cb4 | 54 | 0.998879 | 0.960046 | 2.0480e+03 | 9.6779e+01 | pass |
| baseline__fp32 | single_stage | fc378037 | 53 | 1.000000 | 1.000000 | 0.0000e+00 | 0.0000e+00 | pass |
| hidden_dequant__fp8__block | single_stage | fc378037 | 53 | 0.964594 | 0.621181 | 2.6112e+04 | 6.8586e+03 | pass |
| gemm1_operands__f16 | single_stage | fc378037 | 53 | 0.999595 | 0.987168 | 2.0480e+03 | 7.9265e+01 | pass |
| gemm1_output__fp8__tensor | single_stage | fc378037 | 53 | 0.964531 | 0.622449 | 4.0960e+04 | 2.5918e+03 | pass |
| swiglu_input__fp8__block | single_stage | fc378037 | 53 | 0.965325 | 0.625153 | 3.1744e+04 | 3.0811e+03 | pass |
| swiglu_output__fp8__block | single_stage | fc378037 | 53 | 0.977776 | 0.664855 | 1.6384e+04 | 1.7139e+03 | pass |
| gemm2_operands__fp8__block | single_stage | fc378037 | 53 | 0.966615 | 0.627801 | 2.5600e+04 | 3.6015e+03 | pass |
| gemm2_accumulator__bf16 | single_stage | fc378037 | 53 | 0.996844 | 0.896224 | 6.1440e+03 | 3.4753e+02 | pass |
| out_accumulator__fp8__tensor | single_stage | fc378037 | 53 | 0.990621 | 0.633399 | 5.7344e+04 | 8.1527e+02 | pass |
| gemm1_accumulator__f16 | single_stage | fc378037 | 53 | 0.998379 | 0.953307 | 4.0960e+03 | 1.6127e+02 | pass |
| baseline__fp32 | single_stage | f7d6ac7c | 52 | 1.000000 | 1.000000 | 0.0000e+00 | 0.0000e+00 | pass |
| hidden_dequant__fp8__block | single_stage | f7d6ac7c | 52 | 0.977840 | 0.758875 | 2.0480e+04 | 4.4604e+03 | pass |
| gemm1_operands__f16 | single_stage | f7d6ac7c | 52 | 0.999804 | 0.992131 | 2.0480e+03 | 2.2290e+01 | pass |
| gemm1_output__fp8__tensor | single_stage | f7d6ac7c | 52 | 0.977461 | 0.759452 | 2.5600e+04 | 7.1009e+03 | pass |
| swiglu_input__fp8__block | single_stage | f7d6ac7c | 52 | 0.978113 | 0.760195 | 1.8688e+04 | 4.6235e+03 | pass |
| swiglu_output__fp8__block | single_stage | f7d6ac7c | 52 | 0.985993 | 0.787002 | 1.0752e+04 | 1.2955e+03 | pass |
| gemm2_operands__fp8__block | single_stage | f7d6ac7c | 52 | 0.978803 | 0.763374 | 1.7920e+04 | 1.3285e+03 | pass |
| gemm2_accumulator__bf16 | single_stage | f7d6ac7c | 52 | 0.998251 | 0.937063 | 6.1440e+03 | 1.5894e+02 | pass |
| out_accumulator__fp8__tensor | single_stage | f7d6ac7c | 52 | 0.995707 | 0.770089 | 2.8672e+04 | 3.2712e+03 | pass |
| gemm1_accumulator__f16 | single_stage | f7d6ac7c | 52 | 0.999096 | 0.969633 | 2.0480e+03 | 1.0165e+02 | pass |
| hidden_dequant__fp8__block | cumulative | b8f4f012 | 7 | 0.978476 | 0.763174 | 1.7408e+04 | 4.8691e+02 | pass |
| hidden_dequant__fp8__block | cumulative | e05c6c03 | 1 | 0.932896 | 0.189314 | 1.4336e+04 | 1.5859e+02 | pass |
| hidden_dequant__fp8__block | cumulative | 6230e838 | 32 | 0.969138 | 0.660880 | 2.1504e+04 | 6.4445e+04 | pass |
| hidden_dequant__fp8__block | cumulative | 8f1ff9f1 | 80 | 0.953990 | 0.499432 | 2.4576e+04 | 1.5864e+04 | pass |
| hidden_dequant__fp8__block | cumulative | 1a4c6ba1 | 901 | 0.951972 | 0.474318 | 3.5840e+04 | 2.4160e+11 | pass |
| hidden_dequant__fp8__block | cumulative | a7c2bcfd | 16 | 0.952279 | 0.478149 | 1.8432e+04 | 1.1067e+03 | pass |
| hidden_dequant__fp8__block | cumulative | 2e69caee | 15 | 0.979622 | 0.777167 | 2.0736e+04 | 5.0279e+03 | pass |
| hidden_dequant__fp8__block | cumulative | 8cba5890 | 14 | 0.968710 | 0.644192 | 1.7408e+04 | 7.8669e+02 | pass |
| hidden_dequant__fp8__block | cumulative | 5e8dc11c | 14107 | 0.956304 | 0.522971 | 4.7104e+04 | 5.3440e+11 | pass |
| hidden_dequant__fp8__block | cumulative | 58a34f27 | 11948 | 0.966528 | 0.634460 | 3.7632e+04 | 2.9111e+05 | pass |
| hidden_dequant__fp8__block | cumulative | 5eadab1e | 62 | 0.964250 | 0.609868 | 2.4320e+04 | 2.9412e+03 | pass |
| hidden_dequant__fp8__block | cumulative | eedc63b2 | 59 | 0.970074 | 0.675169 | 2.2528e+04 | 2.3924e+03 | pass |
| hidden_dequant__fp8__block | cumulative | e626d3e6 | 58 | 0.955441 | 0.510278 | 2.3296e+04 | 5.4623e+03 | pass |
| hidden_dequant__fp8__block | cumulative | 74d7ff04 | 57 | 0.962798 | 0.590664 | 2.2528e+04 | 7.5519e+03 | pass |
| hidden_dequant__fp8__block | cumulative | 4822167c | 56 | 0.953481 | 0.492728 | 2.0480e+04 | 1.1850e+04 | pass |
| hidden_dequant__fp8__block | cumulative | 81955b1e | 55 | 0.964108 | 0.605811 | 2.3552e+04 | 1.3097e+03 | pass |
| hidden_dequant__fp8__block | cumulative | 76010cb4 | 54 | 0.970104 | 0.674998 | 2.4640e+04 | 8.8171e+03 | pass |
| hidden_dequant__fp8__block | cumulative | fc378037 | 53 | 0.965302 | 0.623036 | 2.5216e+04 | 3.5389e+06 | pass |
| hidden_dequant__fp8__block | cumulative | f7d6ac7c | 52 | 0.977464 | 0.760109 | 1.8432e+04 | 1.5673e+03 | pass |
| hidden_dequant__fp8__block+gemm1_operands__f16 | cumulative | b8f4f012 | 7 | 0.979273 | 0.761579 | 2.4576e+04 | 2.1608e+02 | pass |
| hidden_dequant__fp8__block+gemm1_operands__f16 | cumulative | e05c6c03 | 1 | 0.921177 | 0.152204 | 1.5488e+04 | 1.5697e+03 | pass |
| hidden_dequant__fp8__block+gemm1_operands__f16 | cumulative | 6230e838 | 32 | 0.968850 | 0.661198 | 2.4576e+04 | 3.4259e+03 | pass |
| hidden_dequant__fp8__block+gemm1_operands__f16 | cumulative | 8f1ff9f1 | 80 | 0.954089 | 0.499864 | 2.8672e+04 | 8.7122e+03 | pass |
| hidden_dequant__fp8__block+gemm1_operands__f16 | cumulative | 1a4c6ba1 | 901 | 0.951919 | 0.474413 | 3.0848e+04 | 3.2067e+04 | pass |
| hidden_dequant__fp8__block+gemm1_operands__f16 | cumulative | a7c2bcfd | 16 | 0.952218 | 0.478856 | 2.1504e+04 | 5.3591e+03 | pass |
| hidden_dequant__fp8__block+gemm1_operands__f16 | cumulative | 2e69caee | 15 | 0.978841 | 0.779064 | 2.4576e+04 | 1.9599e+03 | pass |
| hidden_dequant__fp8__block+gemm1_operands__f16 | cumulative | 8cba5890 | 14 | 0.966837 | 0.641382 | 1.9456e+04 | 9.9318e+03 | pass |
| hidden_dequant__fp8__block+gemm1_operands__f16 | cumulative | 5e8dc11c | 14107 | 0.956400 | 0.523065 | 3.8912e+04 | 4.2598e+05 | pass |
| hidden_dequant__fp8__block+gemm1_operands__f16 | cumulative | 58a34f27 | 11948 | 0.966620 | 0.634585 | 3.7888e+04 | 7.7005e+05 | pass |
| hidden_dequant__fp8__block+gemm1_operands__f16 | cumulative | 5eadab1e | 62 | 0.963984 | 0.610408 | 2.3552e+04 | 9.0001e+03 | pass |
| hidden_dequant__fp8__block+gemm1_operands__f16 | cumulative | eedc63b2 | 59 | 0.970708 | 0.674448 | 2.4576e+04 | 3.1271e+03 | pass |
| hidden_dequant__fp8__block+gemm1_operands__f16 | cumulative | e626d3e6 | 58 | 0.955530 | 0.512813 | 2.6624e+04 | 7.6692e+03 | pass |
| hidden_dequant__fp8__block+gemm1_operands__f16 | cumulative | 74d7ff04 | 57 | 0.962594 | 0.590272 | 2.4576e+04 | 1.5684e+05 | pass |
| hidden_dequant__fp8__block+gemm1_operands__f16 | cumulative | 4822167c | 56 | 0.954463 | 0.494803 | 2.0992e+04 | 4.4027e+03 | pass |
| hidden_dequant__fp8__block+gemm1_operands__f16 | cumulative | 81955b1e | 55 | 0.964212 | 0.606653 | 2.2784e+04 | 1.6060e+03 | pass |
| hidden_dequant__fp8__block+gemm1_operands__f16 | cumulative | 76010cb4 | 54 | 0.970416 | 0.675662 | 2.2528e+04 | 9.0404e+03 | pass |
| hidden_dequant__fp8__block+gemm1_operands__f16 | cumulative | fc378037 | 53 | 0.965425 | 0.622076 | 3.4816e+04 | 4.5422e+03 | pass |
| hidden_dequant__fp8__block+gemm1_operands__f16 | cumulative | f7d6ac7c | 52 | 0.978271 | 0.760372 | 1.8432e+04 | 1.1959e+04 | pass |
| hidden_dequant__fp8__block+gemm1_operands__f16+gemm1_output__fp8__tensor | cumulative | b8f4f012 | 7 | 0.969308 | 0.748824 | 1.9136e+04 | 1.0239e+04 | pass |
| hidden_dequant__fp8__block+gemm1_operands__f16+gemm1_output__fp8__tensor | cumulative | e05c6c03 | 1 | 0.901367 | 0.116769 | 1.9968e+04 | 2.9700e+02 | pass |
| hidden_dequant__fp8__block+gemm1_operands__f16+gemm1_output__fp8__tensor | cumulative | 6230e838 | 32 | 0.956390 | 0.642617 | 3.0720e+04 | 1.2810e+03 | pass |
| hidden_dequant__fp8__block+gemm1_operands__f16+gemm1_output__fp8__tensor | cumulative | 8f1ff9f1 | 80 | 0.934867 | 0.471582 | 3.7888e+04 | 9.2689e+03 | pass |
| hidden_dequant__fp8__block+gemm1_operands__f16+gemm1_output__fp8__tensor | cumulative | 1a4c6ba1 | 901 | 0.931413 | 0.444282 | 5.2480e+04 | 2.3101e+06 | pass |
| hidden_dequant__fp8__block+gemm1_operands__f16+gemm1_output__fp8__tensor | cumulative | a7c2bcfd | 16 | 0.930926 | 0.448748 | 3.0208e+04 | 1.1037e+04 | pass |
| hidden_dequant__fp8__block+gemm1_operands__f16+gemm1_output__fp8__tensor | cumulative | 2e69caee | 15 | 0.969568 | 0.763858 | 2.6696e+04 | 2.5590e+03 | pass |
| hidden_dequant__fp8__block+gemm1_operands__f16+gemm1_output__fp8__tensor | cumulative | 8cba5890 | 14 | 0.953623 | 0.619918 | 2.3040e+04 | 1.3022e+03 | pass |
| hidden_dequant__fp8__block+gemm1_operands__f16+gemm1_output__fp8__tensor | cumulative | 5e8dc11c | 14107 | 0.937953 | 0.495678 | 6.5536e+04 | 6.7830e+05 | pass |
| hidden_dequant__fp8__block+gemm1_operands__f16+gemm1_output__fp8__tensor | cumulative | 58a34f27 | 11948 | 0.952459 | 0.613581 | 5.6320e+04 | 5.4722e+05 | pass |
| hidden_dequant__fp8__block+gemm1_operands__f16+gemm1_output__fp8__tensor | cumulative | 5eadab1e | 62 | 0.949518 | 0.587740 | 2.7648e+04 | 4.8187e+04 | pass |
| hidden_dequant__fp8__block+gemm1_operands__f16+gemm1_output__fp8__tensor | cumulative | eedc63b2 | 59 | 0.957348 | 0.656560 | 2.7648e+04 | 2.9991e+03 | pass |
| hidden_dequant__fp8__block+gemm1_operands__f16+gemm1_output__fp8__tensor | cumulative | e626d3e6 | 58 | 0.936617 | 0.483177 | 3.7376e+04 | 1.5826e+04 | pass |
| hidden_dequant__fp8__block+gemm1_operands__f16+gemm1_output__fp8__tensor | cumulative | 74d7ff04 | 57 | 0.946331 | 0.567361 | 2.7648e+04 | 1.9035e+03 | pass |
| hidden_dequant__fp8__block+gemm1_operands__f16+gemm1_output__fp8__tensor | cumulative | 4822167c | 56 | 0.934969 | 0.464956 | 3.7888e+04 | 2.0470e+03 | pass |
| hidden_dequant__fp8__block+gemm1_operands__f16+gemm1_output__fp8__tensor | cumulative | 81955b1e | 55 | 0.948942 | 0.582729 | 3.6352e+04 | 1.2815e+04 | pass |
| hidden_dequant__fp8__block+gemm1_operands__f16+gemm1_output__fp8__tensor | cumulative | 76010cb4 | 54 | 0.957879 | 0.656888 | 4.5056e+04 | 1.2856e+04 | pass |
| hidden_dequant__fp8__block+gemm1_operands__f16+gemm1_output__fp8__tensor | cumulative | fc378037 | 53 | 0.951227 | 0.601060 | 3.9936e+04 | 5.0052e+03 | pass |
| hidden_dequant__fp8__block+gemm1_operands__f16+gemm1_output__fp8__tensor | cumulative | f7d6ac7c | 52 | 0.968734 | 0.745498 | 3.3280e+04 | 2.0490e+03 | pass |
| hidden_dequant__fp8__block+gemm1_operands__f16+gemm1_output__fp8__tensor+swiglu_input__fp8__block | cumulative | b8f4f012 | 7 | 0.961914 | 0.739836 | 4.0960e+04 | 4.4238e+02 | pass |
| hidden_dequant__fp8__block+gemm1_operands__f16+gemm1_output__fp8__tensor+swiglu_input__fp8__block | cumulative | e05c6c03 | 1 | 0.868164 | 0.105887 | 3.3024e+04 | 5.4287e+02 | catastrophic_outlier |
| hidden_dequant__fp8__block+gemm1_operands__f16+gemm1_output__fp8__tensor+swiglu_input__fp8__block | cumulative | 6230e838 | 32 | 0.945326 | 0.631509 | 4.0960e+04 | 3.0562e+03 | pass |
| hidden_dequant__fp8__block+gemm1_operands__f16+gemm1_output__fp8__tensor+swiglu_input__fp8__block | cumulative | 8f1ff9f1 | 80 | 0.921795 | 0.457162 | 4.0960e+04 | 2.0832e+04 | pass |
| hidden_dequant__fp8__block+gemm1_operands__f16+gemm1_output__fp8__tensor+swiglu_input__fp8__block | cumulative | 1a4c6ba1 | 901 | 0.917462 | 0.430260 | 6.2464e+04 | 1.4745e+05 | pass |
| hidden_dequant__fp8__block+gemm1_operands__f16+gemm1_output__fp8__tensor+swiglu_input__fp8__block | cumulative | a7c2bcfd | 16 | 0.919268 | 0.434614 | 2.6624e+04 | 7.2939e+03 | pass |
| hidden_dequant__fp8__block+gemm1_operands__f16+gemm1_output__fp8__tensor+swiglu_input__fp8__block | cumulative | 2e69caee | 15 | 0.966592 | 0.758129 | 3.1744e+04 | 3.7636e+02 | pass |
| hidden_dequant__fp8__block+gemm1_operands__f16+gemm1_output__fp8__tensor+swiglu_input__fp8__block | cumulative | 8cba5890 | 14 | 0.944147 | 0.613142 | 3.6864e+04 | 5.0199e+03 | pass |
| hidden_dequant__fp8__block+gemm1_operands__f16+gemm1_output__fp8__tensor+swiglu_input__fp8__block | cumulative | 5e8dc11c | 14107 | 0.924690 | 0.482676 | 8.1920e+04 | 1.7792e+12 | pass |
| hidden_dequant__fp8__block+gemm1_operands__f16+gemm1_output__fp8__tensor+swiglu_input__fp8__block | cumulative | 58a34f27 | 11948 | 0.942444 | 0.603730 | 6.4052e+04 | 2.5280e+11 | pass |
| hidden_dequant__fp8__block+gemm1_operands__f16+gemm1_output__fp8__tensor+swiglu_input__fp8__block | cumulative | 5eadab1e | 62 | 0.938420 | 0.577310 | 5.0816e+04 | 2.8562e+03 | pass |
| hidden_dequant__fp8__block+gemm1_operands__f16+gemm1_output__fp8__tensor+swiglu_input__fp8__block | cumulative | eedc63b2 | 59 | 0.948651 | 0.648452 | 3.3792e+04 | 2.4420e+03 | pass |
| hidden_dequant__fp8__block+gemm1_operands__f16+gemm1_output__fp8__tensor+swiglu_input__fp8__block | cumulative | e626d3e6 | 58 | 0.922248 | 0.469746 | 6.2336e+04 | 4.3940e+04 | pass |
| hidden_dequant__fp8__block+gemm1_operands__f16+gemm1_output__fp8__tensor+swiglu_input__fp8__block | cumulative | 74d7ff04 | 57 | 0.934477 | 0.555684 | 4.2496e+04 | 1.3432e+04 | pass |
| hidden_dequant__fp8__block+gemm1_operands__f16+gemm1_output__fp8__tensor+swiglu_input__fp8__block | cumulative | 4822167c | 56 | 0.919917 | 0.451167 | 3.7888e+04 | 7.7536e+03 | pass |
| hidden_dequant__fp8__block+gemm1_operands__f16+gemm1_output__fp8__tensor+swiglu_input__fp8__block | cumulative | 81955b1e | 55 | 0.937766 | 0.573354 | 4.1984e+04 | 6.3025e+03 | pass |
| hidden_dequant__fp8__block+gemm1_operands__f16+gemm1_output__fp8__tensor+swiglu_input__fp8__block | cumulative | 76010cb4 | 54 | 0.948666 | 0.647567 | 4.5056e+04 | 2.9159e+04 | pass |
| hidden_dequant__fp8__block+gemm1_operands__f16+gemm1_output__fp8__tensor+swiglu_input__fp8__block | cumulative | fc378037 | 53 | 0.941704 | 0.590491 | 5.1200e+04 | 1.0442e+06 | pass |
| hidden_dequant__fp8__block+gemm1_operands__f16+gemm1_output__fp8__tensor+swiglu_input__fp8__block | cumulative | f7d6ac7c | 52 | 0.962322 | 0.739271 | 3.4816e+04 | 2.9121e+03 | pass |
| hidden_dequant__fp8__block+gemm1_operands__f16+gemm1_output__fp8__tensor+swiglu_output__fp8__block | cumulative | b8f4f012 | 7 | 0.964724 | 0.743044 | 4.9152e+04 | 1.8530e+02 | pass |
| hidden_dequant__fp8__block+gemm1_operands__f16+gemm1_output__fp8__tensor+swiglu_output__fp8__block | cumulative | e05c6c03 | 1 | 0.881696 | 0.108259 | 3.0720e+04 | 1.5965e+02 | catastrophic_outlier |
| hidden_dequant__fp8__block+gemm1_operands__f16+gemm1_output__fp8__tensor+swiglu_output__fp8__block | cumulative | 6230e838 | 32 | 0.953038 | 0.637830 | 3.7888e+04 | 2.0938e+03 | pass |
| hidden_dequant__fp8__block+gemm1_operands__f16+gemm1_output__fp8__tensor+swiglu_output__fp8__block | cumulative | 8f1ff9f1 | 80 | 0.928924 | 0.465229 | 3.8912e+04 | 1.1368e+04 | pass |
| hidden_dequant__fp8__block+gemm1_operands__f16+gemm1_output__fp8__tensor+swiglu_output__fp8__block | cumulative | 1a4c6ba1 | 901 | 0.925139 | 0.437765 | 4.7104e+04 | 5.2800e+11 | pass |
| hidden_dequant__fp8__block+gemm1_operands__f16+gemm1_output__fp8__tensor+swiglu_output__fp8__block | cumulative | a7c2bcfd | 16 | 0.927080 | 0.442540 | 2.9184e+04 | 8.2376e+04 | pass |
| hidden_dequant__fp8__block+gemm1_operands__f16+gemm1_output__fp8__tensor+swiglu_output__fp8__block | cumulative | 2e69caee | 15 | 0.967262 | 0.762742 | 2.3808e+04 | 1.3329e+03 | pass |
| hidden_dequant__fp8__block+gemm1_operands__f16+gemm1_output__fp8__tensor+swiglu_output__fp8__block | cumulative | 8cba5890 | 14 | 0.948850 | 0.619081 | 2.9312e+04 | 1.1264e+03 | pass |
| hidden_dequant__fp8__block+gemm1_operands__f16+gemm1_output__fp8__tensor+swiglu_output__fp8__block | cumulative | 5e8dc11c | 14107 | 0.932079 | 0.489670 | 7.2704e+04 | 1.1700e+10 | pass |
| hidden_dequant__fp8__block+gemm1_operands__f16+gemm1_output__fp8__tensor+swiglu_output__fp8__block | cumulative | 58a34f27 | 11948 | 0.947981 | 0.608941 | 6.0416e+04 | 6.4700e+06 | pass |
| hidden_dequant__fp8__block+gemm1_operands__f16+gemm1_output__fp8__tensor+swiglu_output__fp8__block | cumulative | 5eadab1e | 62 | 0.943654 | 0.582592 | 3.6864e+04 | 1.6597e+03 | pass |
| hidden_dequant__fp8__block+gemm1_operands__f16+gemm1_output__fp8__tensor+swiglu_output__fp8__block | cumulative | eedc63b2 | 59 | 0.954520 | 0.652691 | 3.2768e+04 | 1.2886e+04 | pass |
| hidden_dequant__fp8__block+gemm1_operands__f16+gemm1_output__fp8__tensor+swiglu_output__fp8__block | cumulative | e626d3e6 | 58 | 0.929873 | 0.476981 | 4.6592e+04 | 6.3497e+03 | pass |
| hidden_dequant__fp8__block+gemm1_operands__f16+gemm1_output__fp8__tensor+swiglu_output__fp8__block | cumulative | 74d7ff04 | 57 | 0.942390 | 0.561626 | 3.4816e+04 | 4.4765e+03 | pass |
| hidden_dequant__fp8__block+gemm1_operands__f16+gemm1_output__fp8__tensor+swiglu_output__fp8__block | cumulative | 4822167c | 56 | 0.928011 | 0.459463 | 3.3024e+04 | 7.0005e+04 | pass |
| hidden_dequant__fp8__block+gemm1_operands__f16+gemm1_output__fp8__tensor+swiglu_output__fp8__block | cumulative | 81955b1e | 55 | 0.944240 | 0.578660 | 3.5712e+04 | 1.2353e+04 | pass |
| hidden_dequant__fp8__block+gemm1_operands__f16+gemm1_output__fp8__tensor+swiglu_output__fp8__block | cumulative | 76010cb4 | 54 | 0.954003 | 0.653426 | 3.3792e+04 | 5.3392e+03 | pass |
| hidden_dequant__fp8__block+gemm1_operands__f16+gemm1_output__fp8__tensor+swiglu_output__fp8__block | cumulative | fc378037 | 53 | 0.944928 | 0.595835 | 4.5440e+04 | 7.5718e+03 | pass |
| hidden_dequant__fp8__block+gemm1_operands__f16+gemm1_output__fp8__tensor+swiglu_output__fp8__block | cumulative | f7d6ac7c | 52 | 0.966067 | 0.742949 | 3.0720e+04 | 5.5622e+03 | pass |
| hidden_dequant__fp8__block+gemm1_operands__f16+gemm1_output__fp8__tensor+gemm2_operands__fp8__block | cumulative | b8f4f012 | 7 | 0.963708 | 0.741809 | 2.6112e+04 | 1.2582e+04 | pass |
| hidden_dequant__fp8__block+gemm1_operands__f16+gemm1_output__fp8__tensor+gemm2_operands__fp8__block | cumulative | e05c6c03 | 1 | 0.876116 | 0.098354 | 2.3552e+04 | 9.9776e+01 | catastrophic_outlier |
| hidden_dequant__fp8__block+gemm1_operands__f16+gemm1_output__fp8__tensor+gemm2_operands__fp8__block | cumulative | 6230e838 | 32 | 0.946285 | 0.632760 | 3.8048e+04 | 1.1377e+04 | pass |
| hidden_dequant__fp8__block+gemm1_operands__f16+gemm1_output__fp8__tensor+gemm2_operands__fp8__block | cumulative | 8f1ff9f1 | 80 | 0.922473 | 0.458098 | 3.6352e+04 | 1.4511e+05 | pass |
| hidden_dequant__fp8__block+gemm1_operands__f16+gemm1_output__fp8__tensor+gemm2_operands__fp8__block | cumulative | 1a4c6ba1 | 901 | 0.918235 | 0.431058 | 5.9392e+04 | 4.5344e+05 | pass |
| hidden_dequant__fp8__block+gemm1_operands__f16+gemm1_output__fp8__tensor+gemm2_operands__fp8__block | cumulative | a7c2bcfd | 16 | 0.920672 | 0.438224 | 3.4816e+04 | 3.8913e+04 | pass |
| hidden_dequant__fp8__block+gemm1_operands__f16+gemm1_output__fp8__tensor+gemm2_operands__fp8__block | cumulative | 2e69caee | 15 | 0.964937 | 0.759552 | 4.5056e+04 | 2.4420e+02 | pass |
| hidden_dequant__fp8__block+gemm1_operands__f16+gemm1_output__fp8__tensor+gemm2_operands__fp8__block | cumulative | 8cba5890 | 14 | 0.942752 | 0.613700 | 4.2496e+04 | 1.6088e+03 | pass |
| hidden_dequant__fp8__block+gemm1_operands__f16+gemm1_output__fp8__tensor+gemm2_operands__fp8__block | cumulative | 5e8dc11c | 14107 | 0.925895 | 0.483698 | 9.2160e+04 | 2.3314e+06 | pass |
| hidden_dequant__fp8__block+gemm1_operands__f16+gemm1_output__fp8__tensor+gemm2_operands__fp8__block | cumulative | 58a34f27 | 11948 | 0.943210 | 0.604425 | 1.0701e+05 | 1.8000e+11 | pass |
| hidden_dequant__fp8__block+gemm1_operands__f16+gemm1_output__fp8__tensor+gemm2_operands__fp8__block | cumulative | 5eadab1e | 62 | 0.939793 | 0.578485 | 3.8912e+04 | 4.8886e+03 | pass |
| hidden_dequant__fp8__block+gemm1_operands__f16+gemm1_output__fp8__tensor+gemm2_operands__fp8__block | cumulative | eedc63b2 | 59 | 0.949819 | 0.647891 | 3.8912e+04 | 8.9377e+03 | pass |
| hidden_dequant__fp8__block+gemm1_operands__f16+gemm1_output__fp8__tensor+gemm2_operands__fp8__block | cumulative | e626d3e6 | 58 | 0.923819 | 0.471225 | 5.4304e+04 | 1.5980e+04 | pass |
| hidden_dequant__fp8__block+gemm1_operands__f16+gemm1_output__fp8__tensor+gemm2_operands__fp8__block | cumulative | 74d7ff04 | 57 | 0.935841 | 0.557201 | 4.1984e+04 | 2.5745e+04 | pass |
| hidden_dequant__fp8__block+gemm1_operands__f16+gemm1_output__fp8__tensor+gemm2_operands__fp8__block | cumulative | 4822167c | 56 | 0.921937 | 0.453172 | 3.5840e+04 | 1.5124e+05 | pass |
| hidden_dequant__fp8__block+gemm1_operands__f16+gemm1_output__fp8__tensor+gemm2_operands__fp8__block | cumulative | 81955b1e | 55 | 0.938428 | 0.573678 | 4.3008e+04 | 6.0455e+03 | pass |
| hidden_dequant__fp8__block+gemm1_operands__f16+gemm1_output__fp8__tensor+gemm2_operands__fp8__block | cumulative | 76010cb4 | 54 | 0.950025 | 0.649693 | 4.7488e+04 | 7.8738e+03 | pass |
| hidden_dequant__fp8__block+gemm1_operands__f16+gemm1_output__fp8__tensor+gemm2_operands__fp8__block | cumulative | fc378037 | 53 | 0.941367 | 0.591176 | 3.8400e+04 | 1.7709e+04 | pass |
| hidden_dequant__fp8__block+gemm1_operands__f16+gemm1_output__fp8__tensor+gemm2_operands__fp8__block | cumulative | f7d6ac7c | 52 | 0.962279 | 0.739590 | 3.0720e+04 | 2.2529e+04 | pass |
| hidden_dequant__fp8__block+gemm1_operands__f16+gemm1_output__fp8__tensor+gemm2_accumulator__bf16 | cumulative | b8f4f012 | 7 | 0.971062 | 0.748964 | 4.3008e+04 | 1.3070e+02 | pass |
| hidden_dequant__fp8__block+gemm1_operands__f16+gemm1_output__fp8__tensor+gemm2_accumulator__bf16 | cumulative | e05c6c03 | 1 | 0.892299 | 0.104353 | 1.8432e+04 | 2.8481e+02 | catastrophic_outlier |
| hidden_dequant__fp8__block+gemm1_operands__f16+gemm1_output__fp8__tensor+gemm2_accumulator__bf16 | cumulative | 6230e838 | 32 | 0.955750 | 0.642473 | 3.2768e+04 | 8.8127e+03 | pass |
| hidden_dequant__fp8__block+gemm1_operands__f16+gemm1_output__fp8__tensor+gemm2_accumulator__bf16 | cumulative | 8f1ff9f1 | 80 | 0.934145 | 0.471474 | 3.7888e+04 | 5.9402e+03 | pass |
| hidden_dequant__fp8__block+gemm1_operands__f16+gemm1_output__fp8__tensor+gemm2_accumulator__bf16 | cumulative | 1a4c6ba1 | 901 | 0.931353 | 0.443828 | 4.6080e+04 | 1.6454e+05 | pass |
| hidden_dequant__fp8__block+gemm1_operands__f16+gemm1_output__fp8__tensor+gemm2_accumulator__bf16 | cumulative | a7c2bcfd | 16 | 0.934326 | 0.451494 | 3.2768e+04 | 2.2766e+03 | pass |
| hidden_dequant__fp8__block+gemm1_operands__f16+gemm1_output__fp8__tensor+gemm2_accumulator__bf16 | cumulative | 2e69caee | 15 | 0.971363 | 0.764574 | 2.8832e+04 | 2.8170e+02 | pass |
| hidden_dequant__fp8__block+gemm1_operands__f16+gemm1_output__fp8__tensor+gemm2_accumulator__bf16 | cumulative | 8cba5890 | 14 | 0.954430 | 0.622768 | 2.5856e+04 | 1.4642e+03 | pass |
| hidden_dequant__fp8__block+gemm1_operands__f16+gemm1_output__fp8__tensor+gemm2_accumulator__bf16 | cumulative | 5e8dc11c | 14107 | 0.937822 | 0.495339 | 6.7584e+04 | 8.1167e+05 | pass |
| hidden_dequant__fp8__block+gemm1_operands__f16+gemm1_output__fp8__tensor+gemm2_accumulator__bf16 | cumulative | 58a34f27 | 11948 | 0.952346 | 0.613270 | 4.9152e+04 | 2.9491e+06 | pass |
| hidden_dequant__fp8__block+gemm1_operands__f16+gemm1_output__fp8__tensor+gemm2_accumulator__bf16 | cumulative | 5eadab1e | 62 | 0.949142 | 0.587020 | 2.9696e+04 | 1.2135e+04 | pass |
| hidden_dequant__fp8__block+gemm1_operands__f16+gemm1_output__fp8__tensor+gemm2_accumulator__bf16 | cumulative | eedc63b2 | 59 | 0.958266 | 0.656392 | 3.0848e+04 | 5.2611e+03 | pass |
| hidden_dequant__fp8__block+gemm1_operands__f16+gemm1_output__fp8__tensor+gemm2_accumulator__bf16 | cumulative | e626d3e6 | 58 | 0.936357 | 0.483160 | 3.4816e+04 | 3.6954e+03 | pass |
| hidden_dequant__fp8__block+gemm1_operands__f16+gemm1_output__fp8__tensor+gemm2_accumulator__bf16 | cumulative | 74d7ff04 | 57 | 0.946394 | 0.567412 | 3.2768e+04 | 3.5336e+03 | pass |
| hidden_dequant__fp8__block+gemm1_operands__f16+gemm1_output__fp8__tensor+gemm2_accumulator__bf16 | cumulative | 4822167c | 56 | 0.934107 | 0.464744 | 2.7264e+04 | 2.4087e+04 | pass |
| hidden_dequant__fp8__block+gemm1_operands__f16+gemm1_output__fp8__tensor+gemm2_accumulator__bf16 | cumulative | 81955b1e | 55 | 0.948681 | 0.583269 | 2.7648e+04 | 8.4755e+03 | pass |
| hidden_dequant__fp8__block+gemm1_operands__f16+gemm1_output__fp8__tensor+gemm2_accumulator__bf16 | cumulative | 76010cb4 | 54 | 0.958093 | 0.656834 | 3.2256e+04 | 1.6501e+04 | pass |
| hidden_dequant__fp8__block+gemm1_operands__f16+gemm1_output__fp8__tensor+gemm2_accumulator__bf16 | cumulative | fc378037 | 53 | 0.950614 | 0.599936 | 4.1984e+04 | 1.6911e+03 | pass |
| hidden_dequant__fp8__block+gemm1_operands__f16+gemm1_output__fp8__tensor+gemm2_accumulator__bf16 | cumulative | f7d6ac7c | 52 | 0.968321 | 0.744929 | 2.8672e+04 | 2.2819e+03 | pass |
| hidden_dequant__fp8__block+gemm1_operands__f16+gemm1_output__fp8__tensor+out_accumulator__fp8__tensor | cumulative | b8f4f012 | 7 | 0.966279 | 0.740374 | 4.5056e+04 | 5.0563e+02 | pass |
| hidden_dequant__fp8__block+gemm1_operands__f16+gemm1_output__fp8__tensor+out_accumulator__fp8__tensor | cumulative | e05c6c03 | 1 | 0.885882 | 0.100725 | 2.9696e+04 | 7.4421e+02 | catastrophic_outlier |
| hidden_dequant__fp8__block+gemm1_operands__f16+gemm1_output__fp8__tensor+out_accumulator__fp8__tensor | cumulative | 6230e838 | 32 | 0.954394 | 0.631553 | 5.7344e+04 | 3.3180e+03 | pass |
| hidden_dequant__fp8__block+gemm1_operands__f16+gemm1_output__fp8__tensor+out_accumulator__fp8__tensor | cumulative | 8f1ff9f1 | 80 | 0.932500 | 0.456463 | 6.1440e+04 | 1.8257e+05 | pass |
| hidden_dequant__fp8__block+gemm1_operands__f16+gemm1_output__fp8__tensor+out_accumulator__fp8__tensor | cumulative | 1a4c6ba1 | 901 | 0.928634 | 0.428951 | 7.3728e+04 | 9.6195e+04 | pass |
| hidden_dequant__fp8__block+gemm1_operands__f16+gemm1_output__fp8__tensor+out_accumulator__fp8__tensor | cumulative | a7c2bcfd | 16 | 0.929164 | 0.434483 | 3.0720e+04 | 1.4346e+03 | pass |
| hidden_dequant__fp8__block+gemm1_operands__f16+gemm1_output__fp8__tensor+out_accumulator__fp8__tensor | cumulative | 2e69caee | 15 | 0.968955 | 0.757952 | 4.5056e+04 | 2.3690e+03 | pass |
| hidden_dequant__fp8__block+gemm1_operands__f16+gemm1_output__fp8__tensor+out_accumulator__fp8__tensor | cumulative | 8cba5890 | 14 | 0.951770 | 0.610840 | 3.2768e+04 | 2.7876e+03 | pass |
| hidden_dequant__fp8__block+gemm1_operands__f16+gemm1_output__fp8__tensor+out_accumulator__fp8__tensor | cumulative | 5e8dc11c | 14107 | 0.935919 | 0.482164 | 1.3107e+05 | 3.5801e+06 | pass |
| hidden_dequant__fp8__block+gemm1_operands__f16+gemm1_output__fp8__tensor+out_accumulator__fp8__tensor | cumulative | 58a34f27 | 11948 | 0.950743 | 0.603348 | 8.6016e+04 | 6.2913e+06 | pass |
| hidden_dequant__fp8__block+gemm1_operands__f16+gemm1_output__fp8__tensor+out_accumulator__fp8__tensor | cumulative | 5eadab1e | 62 | 0.948773 | 0.577956 | 5.5296e+04 | 1.8044e+04 | pass |
| hidden_dequant__fp8__block+gemm1_operands__f16+gemm1_output__fp8__tensor+out_accumulator__fp8__tensor | cumulative | eedc63b2 | 59 | 0.957102 | 0.647253 | 5.3248e+04 | 6.3726e+03 | pass |
| hidden_dequant__fp8__block+gemm1_operands__f16+gemm1_output__fp8__tensor+out_accumulator__fp8__tensor | cumulative | e626d3e6 | 58 | 0.933817 | 0.468644 | 6.5536e+04 | 1.5485e+04 | pass |
| hidden_dequant__fp8__block+gemm1_operands__f16+gemm1_output__fp8__tensor+out_accumulator__fp8__tensor | cumulative | 74d7ff04 | 57 | 0.944390 | 0.554822 | 5.1200e+04 | 3.9287e+03 | pass |
| hidden_dequant__fp8__block+gemm1_operands__f16+gemm1_output__fp8__tensor+out_accumulator__fp8__tensor | cumulative | 4822167c | 56 | 0.931130 | 0.450457 | 6.1440e+04 | 1.7792e+04 | pass |
| hidden_dequant__fp8__block+gemm1_operands__f16+gemm1_output__fp8__tensor+out_accumulator__fp8__tensor | cumulative | 81955b1e | 55 | 0.946198 | 0.570863 | 6.1440e+04 | 3.1003e+03 | pass |
| hidden_dequant__fp8__block+gemm1_operands__f16+gemm1_output__fp8__tensor+out_accumulator__fp8__tensor | cumulative | 76010cb4 | 54 | 0.956967 | 0.648386 | 5.7344e+04 | 1.6204e+04 | pass |
| hidden_dequant__fp8__block+gemm1_operands__f16+gemm1_output__fp8__tensor+out_accumulator__fp8__tensor | cumulative | fc378037 | 53 | 0.949495 | 0.589965 | 6.1440e+04 | 3.2033e+04 | pass |
| hidden_dequant__fp8__block+gemm1_operands__f16+gemm1_output__fp8__tensor+out_accumulator__fp8__tensor | cumulative | f7d6ac7c | 52 | 0.968012 | 0.739328 | 3.4816e+04 | 2.6998e+03 | pass |
| hidden_dequant__fp8__block+gemm1_operands__f16+gemm1_output__fp8__tensor+gemm1_accumulator__f16 | cumulative | b8f4f012 | 7 | 0.970045 | 0.748924 | 2.3552e+04 | 1.7121e+02 | pass |
| hidden_dequant__fp8__block+gemm1_operands__f16+gemm1_output__fp8__tensor+gemm1_accumulator__f16 | cumulative | e05c6c03 | 1 | 0.891741 | 0.132952 | 2.0480e+04 | 4.0432e+02 | catastrophic_outlier |
| hidden_dequant__fp8__block+gemm1_operands__f16+gemm1_output__fp8__tensor+gemm1_accumulator__f16 | cumulative | 6230e838 | 32 | 0.955366 | 0.642365 | 3.1744e+04 | 8.0640e+11 | pass |
| hidden_dequant__fp8__block+gemm1_operands__f16+gemm1_output__fp8__tensor+gemm1_accumulator__f16 | cumulative | 8f1ff9f1 | 80 | 0.934741 | 0.471655 | 3.4816e+04 | 5.8820e+04 | pass |
| hidden_dequant__fp8__block+gemm1_operands__f16+gemm1_output__fp8__tensor+gemm1_accumulator__f16 | cumulative | 1a4c6ba1 | 901 | 0.931539 | 0.444227 | 5.9392e+04 | 2.1571e+06 | pass |
| hidden_dequant__fp8__block+gemm1_operands__f16+gemm1_output__fp8__tensor+gemm1_accumulator__f16 | cumulative | a7c2bcfd | 16 | 0.932844 | 0.449951 | 2.5536e+04 | 1.5032e+03 | pass |
| hidden_dequant__fp8__block+gemm1_operands__f16+gemm1_output__fp8__tensor+gemm1_accumulator__f16 | cumulative | 2e69caee | 15 | 0.970973 | 0.764890 | 3.6096e+04 | 1.4323e+03 | pass |
| hidden_dequant__fp8__block+gemm1_operands__f16+gemm1_output__fp8__tensor+gemm1_accumulator__f16 | cumulative | 8cba5890 | 14 | 0.953613 | 0.622449 | 1.8160e+04 | 8.2526e+03 | pass |
| hidden_dequant__fp8__block+gemm1_operands__f16+gemm1_output__fp8__tensor+gemm1_accumulator__f16 | cumulative | 5e8dc11c | 14107 | 0.937948 | 0.495624 | 6.7584e+04 | 6.4400e+10 | pass |
| hidden_dequant__fp8__block+gemm1_operands__f16+gemm1_output__fp8__tensor+gemm1_accumulator__f16 | cumulative | 58a34f27 | 11948 | 0.952410 | 0.613499 | 5.3248e+04 | 7.9360e+11 | pass |
| hidden_dequant__fp8__block+gemm1_operands__f16+gemm1_output__fp8__tensor+gemm1_accumulator__f16 | cumulative | 5eadab1e | 62 | 0.949070 | 0.587549 | 3.7376e+04 | 4.6337e+04 | pass |
| hidden_dequant__fp8__block+gemm1_operands__f16+gemm1_output__fp8__tensor+gemm1_accumulator__f16 | cumulative | eedc63b2 | 59 | 0.957890 | 0.657217 | 3.1232e+04 | 3.6291e+03 | pass |
| hidden_dequant__fp8__block+gemm1_operands__f16+gemm1_output__fp8__tensor+gemm1_accumulator__f16 | cumulative | e626d3e6 | 58 | 0.935989 | 0.483367 | 3.6864e+04 | 5.3271e+03 | pass |
| hidden_dequant__fp8__block+gemm1_operands__f16+gemm1_output__fp8__tensor+gemm1_accumulator__f16 | cumulative | 74d7ff04 | 57 | 0.946906 | 0.567341 | 3.6864e+04 | 4.9328e+03 | pass |
| hidden_dequant__fp8__block+gemm1_operands__f16+gemm1_output__fp8__tensor+gemm1_accumulator__f16 | cumulative | 4822167c | 56 | 0.934276 | 0.464978 | 3.0208e+04 | 1.1026e+04 | pass |
| hidden_dequant__fp8__block+gemm1_operands__f16+gemm1_output__fp8__tensor+gemm1_accumulator__f16 | cumulative | 81955b1e | 55 | 0.948861 | 0.582962 | 3.6864e+04 | 5.0097e+03 | pass |
| hidden_dequant__fp8__block+gemm1_operands__f16+gemm1_output__fp8__tensor+gemm1_accumulator__f16 | cumulative | 76010cb4 | 54 | 0.957377 | 0.656849 | 3.1488e+04 | 5.1930e+04 | pass |
| hidden_dequant__fp8__block+gemm1_operands__f16+gemm1_output__fp8__tensor+gemm1_accumulator__f16 | cumulative | fc378037 | 53 | 0.950445 | 0.600665 | 3.8912e+04 | 4.3144e+05 | pass |
| hidden_dequant__fp8__block+gemm1_operands__f16+gemm1_output__fp8__tensor+gemm1_accumulator__f16 | cumulative | f7d6ac7c | 52 | 0.968975 | 0.745587 | 3.0720e+04 | 2.6863e+03 | pass |