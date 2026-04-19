# GEMM1 Contract Probe 2026-04-18T22:07:55Z

| metric | value |
|---|---|
| timestamp | 2026-04-18T22:07:55Z |
| definition | moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048 |
| workload_index | 0 |
| workload_uuid | b8f4f012-a32e-4356-b4e1-7665b3d598af |
| workload_short | b8f4f012 |
| seq_len | 7 |
| rows | 7 |
| padded_rows | 8 |
| expert | 0 |
| seed | 1234 |
| atol | 0.01 |
| rtol | 0.01 |
| device | NVIDIA B200 |
| capability | 10.0 |

## Best Candidates

| rank | ok | mode | b_layout | sign_mode | mma_sm | dtype | max_abs | max_rel | mean_abs | matched | error |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | True | MN | rowmajor_physical | raw | 1 | float16 | 1.2500e-01 | 1.5474e-02 | 3.5800e-05 | 1.000000 |  |
| 2 | True | MN | rowmajor_physical | raw | 2 | float16 | 1.2500e-01 | 1.5474e-02 | 3.5800e-05 | 1.000000 |  |
| 3 | True | K | rowmajor_physical | raw | 1 | float16 | 1.2500e-01 | 1.5474e-02 | 3.5800e-05 | 1.000000 |  |
| 4 | True | K | rowmajor_physical | raw | 2 | float16 | 1.2500e-01 | 1.5474e-02 | 3.5800e-05 | 1.000000 |  |
| 5 | True | MN | rowmajor_physical | fold_operand_signs | 1 | float16 | 1.2500e-01 | 1.5474e-02 | 3.5800e-05 | 1.000000 |  |
| 6 | True | MN | rowmajor_physical | fold_operand_signs | 2 | float16 | 1.2500e-01 | 1.5474e-02 | 3.5800e-05 | 1.000000 |  |
| 7 | True | K | rowmajor_physical | fold_operand_signs | 1 | float16 | 1.2500e-01 | 1.5474e-02 | 3.5800e-05 | 1.000000 |  |
| 8 | True | K | rowmajor_physical | fold_operand_signs | 2 | float16 | 1.2500e-01 | 1.5474e-02 | 3.5800e-05 | 1.000000 |  |
| 9 | True | MN | rowmajor_physical | raw | 1 | bfloat16 | 2.5000e-01 | 1.5504e-02 | 1.0862e-05 | 1.000000 |  |
| 10 | True | MN | rowmajor_physical | raw | 2 | bfloat16 | 2.5000e-01 | 1.5504e-02 | 1.0862e-05 | 1.000000 |  |
| 11 | True | K | rowmajor_physical | raw | 1 | bfloat16 | 2.5000e-01 | 1.5504e-02 | 1.0862e-05 | 1.000000 |  |
| 12 | True | K | rowmajor_physical | raw | 2 | bfloat16 | 2.5000e-01 | 1.5504e-02 | 1.0862e-05 | 1.000000 |  |
| 13 | True | MN | rowmajor_physical | fold_operand_signs | 1 | bfloat16 | 2.5000e-01 | 1.5504e-02 | 1.0862e-05 | 1.000000 |  |
| 14 | True | MN | rowmajor_physical | fold_operand_signs | 2 | bfloat16 | 2.5000e-01 | 1.5504e-02 | 1.0862e-05 | 1.000000 |  |
| 15 | True | K | rowmajor_physical | fold_operand_signs | 1 | bfloat16 | 2.5000e-01 | 1.5504e-02 | 1.0862e-05 | 1.000000 |  |
| 16 | True | K | rowmajor_physical | fold_operand_signs | 2 | bfloat16 | 2.5000e-01 | 1.5504e-02 | 1.0862e-05 | 1.000000 |  |
| 17 | True | MN | colmajor_physical | raw | 1 | float16 | 5.7625e+02 | 4.2910e+04 | 8.6848e+01 | 0.003592 |  |
| 18 | True | MN | colmajor_physical | raw | 2 | float16 | 5.7625e+02 | 4.2910e+04 | 8.6848e+01 | 0.003592 |  |
| 19 | True | K | colmajor_physical | raw | 1 | float16 | 5.7625e+02 | 4.2910e+04 | 8.6848e+01 | 0.003592 |  |
| 20 | True | K | colmajor_physical | raw | 2 | float16 | 5.7625e+02 | 4.2910e+04 | 8.6848e+01 | 0.003592 |  |

A candidate is promising only if `matched_ratio` is close to 1.0 under the contest tolerance.