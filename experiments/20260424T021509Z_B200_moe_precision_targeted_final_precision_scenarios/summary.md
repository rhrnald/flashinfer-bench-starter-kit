# MoE Targeted Precision Search 2026-04-24T02:15:08Z

| metric | value |
|---|---|
| timestamp | 2026-04-24T02:15:08Z |
| definition | moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048 |
| seed | 1234 |
| atol | 1.0 |
| rtol | 0.3 |
| required_matched_ratio | 0.9 |
| strict_atol | 0.01 |
| strict_rtol | 0.01 |
| panel_size | 19 |
| panel_indices | 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18 |
| targeted_configs | best_recipe, swap_g1out_for_swiglu, swap_g1out_for_g2in, swap_g1out_for_outacc, post_g1_chain |
| evidence_scope | oracle_only |

## Config Summary

| config | worst_matched_contest | worst_matched_strict | worst_max_rel | contest_status | strict_status |
|---|---:|---:|---:|---|---|
| post_g1_chain | 0.921177 | 0.110770 | 6.7520e+11 | contest_safe | strict_unsafe |
| swap_g1out_for_outacc | 0.916155 | 0.108398 | 2.3200e+11 | contest_safe | strict_unsafe |
| swap_g1out_for_swiglu | 0.911412 | 0.140206 | 3.1520e+11 | contest_safe | strict_unsafe |
| best_recipe | 0.896345 | 0.119559 | 3.7440e+11 | contest_unsafe | strict_unsafe |
| swap_g1out_for_g2in | 0.894252 | 0.114955 | 6.4640e+11 | contest_unsafe | strict_unsafe |

## Sampled Results

| candidate | phase | workload | seq_len | matched_contest | matched_strict | max_abs | max_rel | failure |
|---|---|---|---:|---:|---:|---:|---:|---|
| best_recipe | targeted | b8f4f012 | 7 | 0.968650 | 0.746512 | 2.8672e+04 | 1.2565e+03 | pass |
| swap_g1out_for_swiglu | targeted | b8f4f012 | 7 | 0.973473 | 0.754165 | 2.5600e+04 | 1.0766e+02 | pass |
| swap_g1out_for_g2in | targeted | b8f4f012 | 7 | 0.970424 | 0.749183 | 2.7648e+04 | 3.4482e+02 | pass |
| swap_g1out_for_outacc | targeted | b8f4f012 | 7 | 0.974809 | 0.746911 | 4.7104e+04 | 1.1957e+03 | pass |
| post_g1_chain | targeted | b8f4f012 | 7 | 0.977021 | 0.747250 | 5.1200e+04 | 4.8405e+02 | pass |
| best_recipe | targeted | e05c6c03 | 1 | 0.896345 | 0.119559 | 1.5360e+04 | 7.3964e+01 | catastrophic_outlier |
| swap_g1out_for_swiglu | targeted | e05c6c03 | 1 | 0.911412 | 0.140206 | 1.7312e+04 | 6.2630e+01 | pass |
| swap_g1out_for_g2in | targeted | e05c6c03 | 1 | 0.894252 | 0.114955 | 2.0256e+04 | 6.3359e+01 | catastrophic_outlier |
| swap_g1out_for_outacc | targeted | e05c6c03 | 1 | 0.916155 | 0.108398 | 2.0480e+04 | 4.2857e+01 | pass |
| post_g1_chain | targeted | e05c6c03 | 1 | 0.921177 | 0.110770 | 2.2528e+04 | 1.9268e+02 | pass |
| best_recipe | targeted | 6230e838 | 32 | 0.955518 | 0.642439 | 3.4816e+04 | 1.1912e+03 | pass |
| swap_g1out_for_swiglu | targeted | 6230e838 | 32 | 0.963279 | 0.651415 | 2.6624e+04 | 1.9263e+03 | pass |
| swap_g1out_for_g2in | targeted | 6230e838 | 32 | 0.957886 | 0.643289 | 3.0720e+04 | 2.0207e+03 | pass |
| swap_g1out_for_outacc | targeted | 6230e838 | 32 | 0.965498 | 0.638048 | 6.3488e+04 | 2.1056e+03 | pass |
| post_g1_chain | targeted | 6230e838 | 32 | 0.967159 | 0.640111 | 5.1200e+04 | 6.4788e+02 | pass |
| best_recipe | targeted | 8f1ff9f1 | 80 | 0.934502 | 0.471268 | 3.2768e+04 | 1.5184e+04 | pass |
| swap_g1out_for_swiglu | targeted | 8f1ff9f1 | 80 | 0.945890 | 0.485587 | 2.8672e+04 | 1.1324e+04 | pass |
| swap_g1out_for_g2in | targeted | 8f1ff9f1 | 80 | 0.937435 | 0.473312 | 3.3792e+04 | 9.8228e+03 | pass |
| swap_g1out_for_outacc | targeted | 8f1ff9f1 | 80 | 0.950867 | 0.470210 | 6.7584e+04 | 9.6084e+03 | pass |
| post_g1_chain | targeted | 8f1ff9f1 | 80 | 0.952877 | 0.471929 | 5.9392e+04 | 2.0042e+03 | pass |
| best_recipe | targeted | 1a4c6ba1 | 901 | 0.931616 | 0.444204 | 5.7344e+04 | 5.2471e+04 | pass |
| swap_g1out_for_swiglu | targeted | 1a4c6ba1 | 901 | 0.943350 | 0.459486 | 3.6864e+04 | 5.7671e+04 | pass |
| swap_g1out_for_g2in | targeted | 1a4c6ba1 | 901 | 0.934331 | 0.446145 | 5.3248e+04 | 5.7343e+04 | pass |
| swap_g1out_for_outacc | targeted | 1a4c6ba1 | 901 | 0.948199 | 0.442705 | 7.7824e+04 | 4.1761e+04 | pass |
| post_g1_chain | targeted | 1a4c6ba1 | 901 | 0.950516 | 0.445138 | 7.3728e+04 | 3.7355e+04 | pass |
| best_recipe | targeted | a7c2bcfd | 16 | 0.933262 | 0.449925 | 2.4576e+04 | 4.5258e+02 | pass |
| swap_g1out_for_swiglu | targeted | a7c2bcfd | 16 | 0.943787 | 0.463379 | 2.2568e+04 | 6.3639e+02 | pass |
| swap_g1out_for_g2in | targeted | a7c2bcfd | 16 | 0.933681 | 0.450099 | 2.6624e+04 | 6.1549e+02 | pass |
| swap_g1out_for_outacc | targeted | a7c2bcfd | 16 | 0.951050 | 0.447780 | 3.0720e+04 | 4.7617e+02 | pass |
| post_g1_chain | targeted | a7c2bcfd | 16 | 0.952114 | 0.453265 | 3.4816e+04 | 1.9422e+03 | pass |
| best_recipe | targeted | 2e69caee | 15 | 0.971029 | 0.765365 | 2.9696e+04 | 2.0904e+03 | pass |
| swap_g1out_for_swiglu | targeted | 2e69caee | 15 | 0.976097 | 0.772786 | 2.2784e+04 | 9.0151e+02 | pass |
| swap_g1out_for_g2in | targeted | 2e69caee | 15 | 0.972526 | 0.765783 | 2.7648e+04 | 6.0412e+02 | pass |
| swap_g1out_for_outacc | targeted | 2e69caee | 15 | 0.978209 | 0.764174 | 4.7104e+04 | 3.4702e+03 | pass |
| post_g1_chain | targeted | 2e69caee | 15 | 0.979315 | 0.765392 | 4.0960e+04 | 9.3822e+02 | pass |
| best_recipe | targeted | 8cba5890 | 14 | 0.953972 | 0.622120 | 2.8928e+04 | 1.6298e+03 | pass |
| swap_g1out_for_swiglu | targeted | 8cba5890 | 14 | 0.961814 | 0.631577 | 2.3040e+04 | 1.0789e+03 | pass |
| swap_g1out_for_g2in | targeted | 8cba5890 | 14 | 0.954670 | 0.623027 | 2.6112e+04 | 2.4682e+03 | pass |
| swap_g1out_for_outacc | targeted | 8cba5890 | 14 | 0.965073 | 0.619340 | 3.2768e+04 | 6.4474e+02 | pass |
| post_g1_chain | targeted | 8cba5890 | 14 | 0.966558 | 0.621991 | 3.0720e+04 | 1.1508e+03 | pass |
| best_recipe | targeted | 5e8dc11c | 14107 | 0.938003 | 0.495702 | 7.3728e+04 | 3.7440e+11 | pass |
| swap_g1out_for_swiglu | targeted | 5e8dc11c | 14107 | 0.948564 | 0.509462 | 6.2464e+04 | 3.1520e+11 | pass |
| swap_g1out_for_g2in | targeted | 5e8dc11c | 14107 | 0.940410 | 0.497340 | 6.1952e+04 | 6.4640e+11 | pass |
| swap_g1out_for_outacc | targeted | 5e8dc11c | 14107 | 0.953820 | 0.495482 | 1.4746e+05 | 2.3200e+11 | pass |
| post_g1_chain | targeted | 5e8dc11c | 14107 | 0.956008 | 0.497295 | 1.3926e+05 | 6.7520e+11 | pass |
| best_recipe | targeted | 58a34f27 | 11948 | 0.952447 | 0.613611 | 6.1440e+04 | 1.2749e+06 | pass |
| swap_g1out_for_swiglu | targeted | 58a34f27 | 11948 | 0.960579 | 0.624106 | 5.8112e+04 | 2.9194e+06 | pass |
| swap_g1out_for_g2in | targeted | 58a34f27 | 11948 | 0.954339 | 0.614862 | 6.4512e+04 | 2.9378e+06 | pass |
| swap_g1out_for_outacc | targeted | 58a34f27 | 11948 | 0.964591 | 0.613415 | 9.4208e+04 | 2.4944e+06 | pass |
| post_g1_chain | targeted | 58a34f27 | 11948 | 0.966237 | 0.615195 | 9.8304e+04 | 1.6999e+06 | pass |
| best_recipe | targeted | 5eadab1e | 62 | 0.949399 | 0.587873 | 4.0960e+04 | 1.0810e+04 | pass |
| swap_g1out_for_swiglu | targeted | 5eadab1e | 62 | 0.958102 | 0.599760 | 3.0720e+04 | 2.8218e+04 | pass |
| swap_g1out_for_g2in | targeted | 5eadab1e | 62 | 0.951212 | 0.589612 | 3.5856e+04 | 2.2035e+04 | pass |
| swap_g1out_for_outacc | targeted | 5eadab1e | 62 | 0.962456 | 0.586410 | 5.1200e+04 | 7.6790e+03 | pass |
| post_g1_chain | targeted | 5eadab1e | 62 | 0.963847 | 0.588748 | 4.9152e+04 | 1.5749e+04 | pass |
| best_recipe | targeted | eedc63b2 | 59 | 0.957334 | 0.656931 | 4.6080e+04 | 7.0350e+03 | pass |
| swap_g1out_for_swiglu | targeted | eedc63b2 | 59 | 0.964792 | 0.666075 | 3.1744e+04 | 4.9977e+03 | pass |
| swap_g1out_for_g2in | targeted | eedc63b2 | 59 | 0.959325 | 0.657378 | 4.7104e+04 | 4.7832e+03 | pass |
| swap_g1out_for_outacc | targeted | eedc63b2 | 59 | 0.968880 | 0.655210 | 5.5296e+04 | 5.3623e+03 | pass |
| post_g1_chain | targeted | eedc63b2 | 59 | 0.970058 | 0.657324 | 5.9392e+04 | 6.1048e+03 | pass |
| best_recipe | targeted | e626d3e6 | 58 | 0.936961 | 0.482504 | 5.0176e+04 | 8.7761e+03 | pass |
| swap_g1out_for_swiglu | targeted | e626d3e6 | 58 | 0.947374 | 0.498145 | 3.7888e+04 | 1.0577e+04 | pass |
| swap_g1out_for_g2in | targeted | e626d3e6 | 58 | 0.939273 | 0.484808 | 4.1984e+04 | 2.1155e+04 | pass |
| swap_g1out_for_outacc | targeted | e626d3e6 | 58 | 0.952329 | 0.482326 | 6.1440e+04 | 8.4109e+03 | pass |
| post_g1_chain | targeted | e626d3e6 | 58 | 0.954412 | 0.484690 | 6.1440e+04 | 3.5184e+03 | pass |
| best_recipe | targeted | 74d7ff04 | 57 | 0.946517 | 0.567297 | 3.7888e+04 | 3.2905e+04 | pass |
| swap_g1out_for_swiglu | targeted | 74d7ff04 | 57 | 0.956578 | 0.579549 | 2.8672e+04 | 3.1110e+04 | pass |
| swap_g1out_for_g2in | targeted | 74d7ff04 | 57 | 0.949446 | 0.569468 | 3.2256e+04 | 6.3599e+04 | pass |
| swap_g1out_for_outacc | targeted | 74d7ff04 | 57 | 0.960443 | 0.567559 | 4.5056e+04 | 3.0557e+04 | pass |
| post_g1_chain | targeted | 74d7ff04 | 57 | 0.961853 | 0.567385 | 5.5296e+04 | 1.5264e+05 | pass |
| best_recipe | targeted | 4822167c | 56 | 0.934795 | 0.465195 | 2.8160e+04 | 1.2310e+04 | pass |
| swap_g1out_for_swiglu | targeted | 4822167c | 56 | 0.945649 | 0.479554 | 2.4064e+04 | 9.2170e+03 | pass |
| swap_g1out_for_g2in | targeted | 4822167c | 56 | 0.937136 | 0.467098 | 2.7136e+04 | 9.8221e+03 | pass |
| swap_g1out_for_outacc | targeted | 4822167c | 56 | 0.950791 | 0.464268 | 5.5296e+04 | 7.1225e+03 | pass |
| post_g1_chain | targeted | 4822167c | 56 | 0.953499 | 0.466774 | 4.0960e+04 | 1.3923e+04 | pass |
| best_recipe | targeted | 81955b1e | 55 | 0.948321 | 0.582348 | 3.0272e+04 | 1.0385e+04 | pass |
| swap_g1out_for_swiglu | targeted | 81955b1e | 55 | 0.956942 | 0.593755 | 2.4576e+04 | 6.0942e+03 | pass |
| swap_g1out_for_g2in | targeted | 81955b1e | 55 | 0.950332 | 0.583497 | 2.6624e+04 | 1.4147e+03 | pass |
| swap_g1out_for_outacc | targeted | 81955b1e | 55 | 0.960976 | 0.582211 | 3.2768e+04 | 5.4847e+03 | pass |
| post_g1_chain | targeted | 81955b1e | 55 | 0.963223 | 0.583523 | 3.6864e+04 | 2.5283e+03 | pass |
| best_recipe | targeted | 76010cb4 | 54 | 0.957933 | 0.657759 | 3.8912e+04 | 3.1071e+03 | pass |
| swap_g1out_for_swiglu | targeted | 76010cb4 | 54 | 0.964815 | 0.667116 | 3.0720e+04 | 2.9023e+03 | pass |
| swap_g1out_for_g2in | targeted | 76010cb4 | 54 | 0.959243 | 0.658319 | 3.3792e+04 | 4.7937e+03 | pass |
| swap_g1out_for_outacc | targeted | 76010cb4 | 54 | 0.968507 | 0.656648 | 6.1440e+04 | 3.2999e+03 | pass |
| post_g1_chain | targeted | 76010cb4 | 54 | 0.969409 | 0.658250 | 5.3248e+04 | 3.6605e+03 | pass |
| best_recipe | targeted | fc378037 | 53 | 0.950606 | 0.600278 | 4.1856e+04 | 7.3075e+03 | pass |
| swap_g1out_for_swiglu | targeted | fc378037 | 53 | 0.959608 | 0.612170 | 3.4816e+04 | 8.0304e+03 | pass |
| swap_g1out_for_g2in | targeted | fc378037 | 53 | 0.953128 | 0.602063 | 3.8912e+04 | 6.5044e+03 | pass |
| swap_g1out_for_outacc | targeted | fc378037 | 53 | 0.963070 | 0.600381 | 5.9392e+04 | 5.2194e+03 | pass |
| post_g1_chain | targeted | fc378037 | 53 | 0.965102 | 0.602034 | 5.5296e+04 | 1.9586e+03 | pass |
| best_recipe | targeted | f7d6ac7c | 52 | 0.967934 | 0.745796 | 3.1744e+04 | 3.9701e+03 | pass |
| swap_g1out_for_swiglu | targeted | f7d6ac7c | 52 | 0.973255 | 0.752297 | 2.5088e+04 | 3.6954e+03 | pass |
| swap_g1out_for_g2in | targeted | f7d6ac7c | 52 | 0.969348 | 0.746268 | 2.7904e+04 | 3.5455e+03 | pass |
| swap_g1out_for_outacc | targeted | f7d6ac7c | 52 | 0.976214 | 0.745259 | 3.4816e+04 | 1.2557e+03 | pass |
| post_g1_chain | targeted | f7d6ac7c | 52 | 0.977700 | 0.746923 | 3.8912e+04 | 2.1742e+03 | pass |