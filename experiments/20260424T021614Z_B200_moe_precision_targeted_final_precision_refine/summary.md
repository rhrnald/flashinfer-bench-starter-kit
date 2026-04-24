# MoE Targeted Precision Search 2026-04-24T02:16:14Z

| metric | value |
|---|---|
| timestamp | 2026-04-24T02:16:14Z |
| definition | moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048 |
| seed | 1234 |
| atol | 1.0 |
| rtol | 0.3 |
| required_matched_ratio | 0.9 |
| strict_atol | 0.01 |
| strict_rtol | 0.01 |
| panel_size | 19 |
| panel_indices | 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18 |
| targeted_configs | post_g1_chain, post_g1_no_outacc, post_g1_no_g2in, post_g1_plus_hidden |
| evidence_scope | oracle_only |

## Config Summary

| config | worst_matched_contest | worst_matched_strict | worst_max_rel | contest_status | strict_status |
|---|---:|---:|---:|---|---|
| post_g1_no_g2in | 0.939035 | 0.130720 | 2.9760e+11 | contest_safe | strict_unsafe |
| post_g1_no_outacc | 0.931362 | 0.182199 | 7.4560e+11 | contest_safe | strict_unsafe |
| post_g1_chain | 0.921177 | 0.110770 | 6.7520e+11 | contest_safe | strict_unsafe |
| post_g1_plus_hidden | 0.889369 | 0.090123 | 5.2800e+11 | contest_unsafe | strict_unsafe |

## Sampled Results

| candidate | phase | workload | seq_len | matched_contest | matched_strict | max_abs | max_rel | failure |
|---|---|---|---:|---:|---:|---:|---:|---|
| post_g1_chain | targeted | b8f4f012 | 7 | 0.977021 | 0.747250 | 5.1200e+04 | 4.8405e+02 | pass |
| post_g1_no_outacc | targeted | b8f4f012 | 7 | 0.979951 | 0.767020 | 2.4064e+04 | 1.3052e+02 | pass |
| post_g1_no_g2in | targeted | b8f4f012 | 7 | 0.983159 | 0.752671 | 4.7104e+04 | 1.3753e+03 | pass |
| post_g1_plus_hidden | targeted | b8f4f012 | 7 | 0.968531 | 0.740932 | 6.1440e+04 | 1.3733e+03 | pass |
| post_g1_chain | targeted | e05c6c03 | 1 | 0.921177 | 0.110770 | 2.2528e+04 | 1.9268e+02 | pass |
| post_g1_no_outacc | targeted | e05c6c03 | 1 | 0.931362 | 0.182199 | 1.3824e+04 | 1.4251e+02 | pass |
| post_g1_no_g2in | targeted | e05c6c03 | 1 | 0.939035 | 0.130720 | 2.1504e+04 | 4.3060e+01 | pass |
| post_g1_plus_hidden | targeted | e05c6c03 | 1 | 0.889369 | 0.090123 | 1.8816e+04 | 5.9938e+01 | catastrophic_outlier |
| post_g1_chain | targeted | 6230e838 | 32 | 0.967159 | 0.640111 | 5.1200e+04 | 6.4788e+02 | pass |
| post_g1_no_outacc | targeted | 6230e838 | 32 | 0.970394 | 0.666351 | 2.5600e+04 | 5.9008e+02 | pass |
| post_g1_no_g2in | targeted | 6230e838 | 32 | 0.975826 | 0.647805 | 5.1200e+04 | 6.5304e+02 | pass |
| post_g1_plus_hidden | targeted | 6230e838 | 32 | 0.955213 | 0.631662 | 5.5296e+04 | 2.1151e+03 | pass |
| post_g1_chain | targeted | 8f1ff9f1 | 80 | 0.952877 | 0.471929 | 5.9392e+04 | 2.0042e+03 | pass |
| post_g1_no_outacc | targeted | 8f1ff9f1 | 80 | 0.956449 | 0.506508 | 2.2528e+04 | 1.0923e+03 | pass |
| post_g1_no_g2in | targeted | 8f1ff9f1 | 80 | 0.966155 | 0.486164 | 6.1440e+04 | 2.3814e+03 | pass |
| post_g1_plus_hidden | targeted | 8f1ff9f1 | 80 | 0.934729 | 0.458587 | 6.9632e+04 | 9.6084e+03 | pass |
| post_g1_chain | targeted | 1a4c6ba1 | 901 | 0.950516 | 0.445138 | 7.3728e+04 | 3.7355e+04 | pass |
| post_g1_no_outacc | targeted | 1a4c6ba1 | 901 | 0.954356 | 0.482016 | 3.2768e+04 | 2.7033e+04 | pass |
| post_g1_no_g2in | targeted | 1a4c6ba1 | 901 | 0.964361 | 0.458340 | 7.7824e+04 | 1.9004e+04 | pass |
| post_g1_plus_hidden | targeted | 1a4c6ba1 | 901 | 0.931311 | 0.429319 | 8.1920e+04 | 5.5167e+04 | pass |
| post_g1_chain | targeted | a7c2bcfd | 16 | 0.952114 | 0.453265 | 3.4816e+04 | 1.9422e+03 | pass |
| post_g1_no_outacc | targeted | a7c2bcfd | 16 | 0.954817 | 0.488743 | 1.6896e+04 | 1.7136e+03 | pass |
| post_g1_no_g2in | targeted | a7c2bcfd | 16 | 0.966378 | 0.468619 | 2.8672e+04 | 1.1468e+03 | pass |
| post_g1_plus_hidden | targeted | a7c2bcfd | 16 | 0.931893 | 0.434413 | 3.4816e+04 | 8.3725e+02 | pass |
| post_g1_chain | targeted | 2e69caee | 15 | 0.979315 | 0.765392 | 4.0960e+04 | 9.3822e+02 | pass |
| post_g1_no_outacc | targeted | 2e69caee | 15 | 0.980813 | 0.780701 | 1.8944e+04 | 4.0236e+02 | pass |
| post_g1_no_g2in | targeted | 2e69caee | 15 | 0.985407 | 0.771205 | 3.4816e+04 | 7.2361e+02 | pass |
| post_g1_plus_hidden | targeted | 2e69caee | 15 | 0.971401 | 0.759338 | 4.9152e+04 | 2.7711e+02 | pass |
| post_g1_chain | targeted | 8cba5890 | 14 | 0.966558 | 0.621991 | 3.0720e+04 | 1.1508e+03 | pass |
| post_g1_no_outacc | targeted | 8cba5890 | 14 | 0.969079 | 0.648517 | 1.5360e+04 | 1.1088e+03 | pass |
| post_g1_no_g2in | targeted | 8cba5890 | 14 | 0.975566 | 0.630680 | 2.8672e+04 | 3.4028e+02 | pass |
| post_g1_plus_hidden | targeted | 8cba5890 | 14 | 0.952607 | 0.611976 | 3.1744e+04 | 2.5640e+03 | pass |
| post_g1_chain | targeted | 5e8dc11c | 14107 | 0.956008 | 0.497295 | 1.3926e+05 | 6.7520e+11 | pass |
| post_g1_no_outacc | targeted | 5e8dc11c | 14107 | 0.958644 | 0.529779 | 5.0176e+04 | 7.4560e+11 | pass |
| post_g1_no_g2in | targeted | 5e8dc11c | 14107 | 0.968773 | 0.510976 | 1.3926e+05 | 2.9760e+11 | pass |
| post_g1_plus_hidden | targeted | 5e8dc11c | 14107 | 0.938279 | 0.483698 | 1.2698e+05 | 5.2800e+11 | pass |
| post_g1_chain | targeted | 58a34f27 | 11948 | 0.966237 | 0.615195 | 9.8304e+04 | 1.6999e+06 | pass |
| post_g1_no_outacc | targeted | 58a34f27 | 11948 | 0.968280 | 0.639716 | 3.6864e+04 | 1.6629e+06 | pass |
| post_g1_no_g2in | targeted | 58a34f27 | 11948 | 0.975999 | 0.625176 | 8.6016e+04 | 1.7091e+06 | pass |
| post_g1_plus_hidden | targeted | 58a34f27 | 11948 | 0.952729 | 0.604101 | 9.0112e+04 | 2.7715e+06 | pass |
| post_g1_chain | targeted | 5eadab1e | 62 | 0.963847 | 0.588748 | 4.9152e+04 | 1.5749e+04 | pass |
| post_g1_no_outacc | targeted | 5eadab1e | 62 | 0.965913 | 0.615570 | 2.5600e+04 | 1.6244e+04 | pass |
| post_g1_no_g2in | targeted | 5eadab1e | 62 | 0.974436 | 0.599895 | 5.1200e+04 | 3.3907e+04 | pass |
| post_g1_plus_hidden | targeted | 5eadab1e | 62 | 0.949424 | 0.577508 | 5.2224e+04 | 2.2600e+04 | pass |
| post_g1_chain | targeted | eedc63b2 | 59 | 0.970058 | 0.657324 | 5.9392e+04 | 6.1048e+03 | pass |
| post_g1_no_outacc | targeted | eedc63b2 | 59 | 0.971557 | 0.679628 | 2.8672e+04 | 5.8639e+03 | pass |
| post_g1_no_g2in | targeted | eedc63b2 | 59 | 0.979069 | 0.667125 | 5.5296e+04 | 2.8521e+03 | pass |
| post_g1_plus_hidden | targeted | eedc63b2 | 59 | 0.957984 | 0.647641 | 5.9392e+04 | 4.9119e+03 | pass |
| post_g1_chain | targeted | e626d3e6 | 58 | 0.954412 | 0.484690 | 6.1440e+04 | 3.5184e+03 | pass |
| post_g1_no_outacc | targeted | e626d3e6 | 58 | 0.957784 | 0.518860 | 3.1744e+04 | 4.6319e+03 | pass |
| post_g1_no_g2in | targeted | e626d3e6 | 58 | 0.967319 | 0.496791 | 6.9632e+04 | 8.3521e+03 | pass |
| post_g1_plus_hidden | targeted | e626d3e6 | 58 | 0.936968 | 0.470092 | 7.7824e+04 | 1.9008e+04 | pass |
| post_g1_chain | targeted | 74d7ff04 | 57 | 0.961853 | 0.567385 | 5.5296e+04 | 1.5264e+05 | pass |
| post_g1_no_outacc | targeted | 74d7ff04 | 57 | 0.964572 | 0.596134 | 2.4576e+04 | 1.5264e+05 | pass |
| post_g1_no_g2in | targeted | 74d7ff04 | 57 | 0.972761 | 0.578913 | 4.7104e+04 | 6.1664e+04 | pass |
| post_g1_plus_hidden | targeted | 74d7ff04 | 57 | 0.947175 | 0.556910 | 5.2224e+04 | 6.1111e+04 | pass |
| post_g1_chain | targeted | 4822167c | 56 | 0.953499 | 0.466774 | 4.0960e+04 | 1.3923e+04 | pass |
| post_g1_no_outacc | targeted | 4822167c | 56 | 0.956869 | 0.500864 | 2.0480e+04 | 1.3499e+04 | pass |
| post_g1_no_g2in | targeted | 4822167c | 56 | 0.966730 | 0.480329 | 4.5056e+04 | 7.0045e+03 | pass |
| post_g1_plus_hidden | targeted | 4822167c | 56 | 0.934483 | 0.452226 | 5.3248e+04 | 8.7981e+03 | pass |
| post_g1_chain | targeted | 81955b1e | 55 | 0.963223 | 0.583523 | 3.6864e+04 | 2.5283e+03 | pass |
| post_g1_no_outacc | targeted | 81955b1e | 55 | 0.965643 | 0.611209 | 1.6384e+04 | 2.2523e+03 | pass |
| post_g1_no_g2in | targeted | 81955b1e | 55 | 0.973574 | 0.595351 | 3.8912e+04 | 4.8752e+03 | pass |
| post_g1_plus_hidden | targeted | 81955b1e | 55 | 0.948397 | 0.572103 | 4.1984e+04 | 2.1138e+03 | pass |
| post_g1_chain | targeted | 76010cb4 | 54 | 0.969409 | 0.658250 | 5.3248e+04 | 3.6605e+03 | pass |
| post_g1_no_outacc | targeted | 76010cb4 | 54 | 0.971496 | 0.680651 | 2.6624e+04 | 3.7613e+03 | pass |
| post_g1_no_g2in | targeted | 76010cb4 | 54 | 0.978257 | 0.667294 | 6.1440e+04 | 2.9284e+03 | pass |
| post_g1_plus_hidden | targeted | 76010cb4 | 54 | 0.957778 | 0.648303 | 5.7344e+04 | 5.7334e+03 | pass |
| post_g1_chain | targeted | fc378037 | 53 | 0.965102 | 0.602034 | 5.5296e+04 | 1.9586e+03 | pass |
| post_g1_no_outacc | targeted | fc378037 | 53 | 0.967650 | 0.628206 | 2.2272e+04 | 3.8761e+03 | pass |
| post_g1_no_g2in | targeted | fc378037 | 53 | 0.974828 | 0.612284 | 5.1200e+04 | 2.9927e+03 | pass |
| post_g1_plus_hidden | targeted | fc378037 | 53 | 0.951153 | 0.590933 | 6.4512e+04 | 6.7855e+03 | pass |
| post_g1_chain | targeted | f7d6ac7c | 52 | 0.977700 | 0.746923 | 3.8912e+04 | 2.1742e+03 | pass |
| post_g1_no_outacc | targeted | f7d6ac7c | 52 | 0.978770 | 0.761711 | 2.0480e+04 | 2.2879e+03 | pass |
| post_g1_no_g2in | targeted | f7d6ac7c | 52 | 0.984337 | 0.753257 | 3.4816e+04 | 1.2241e+03 | pass |
| post_g1_plus_hidden | targeted | f7d6ac7c | 52 | 0.968337 | 0.739301 | 4.5056e+04 | 1.8971e+03 | pass |