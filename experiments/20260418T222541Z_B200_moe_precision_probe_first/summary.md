# MoE Precision Probe 2026-04-18T22:25:40Z

| metric | value |
|---|---|
| timestamp | 2026-04-18T22:25:40Z |
| definition | moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048 |
| workload_index | 0 |
| workload_uuid | b8f4f012-a32e-4356-b4e1-7665b3d598af |
| seq_len | 7 |
| seed | 1234 |
| atol | 0.01 |
| rtol | 0.01 |
| device | NVIDIA B200 |

| variant | max_abs | max_rel | mean_abs | matched | selected_experts |
|---|---|---|---|---|---|
| fp32 | 0.0000e+00 | 0.0000e+00 | 0.0000e+00 | 1.000000 | 7 |
| bf16 | 2.0480e+03 | 8.4689e+00 | 5.8845e+01 | 0.958187 | 7 |
| f16 | 2.0480e+03 | 2.1754e+00 | 7.6044e+00 | 0.994938 | 7 |