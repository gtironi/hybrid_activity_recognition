# PatchTST Ablations Summary

| run | acc | macro_f1 | done |
|---|---:|---:|---|
| patchtst_hybrid_AcTBeCalf_topk_k_05_d128_l3_h4_p8_s8_ep150_bs2000_lr1e-3_s42 | 0.9143 | 0.7967 | yes |
| patchtst_hybrid_AcTBeCalf_topk_k_07_d128_l3_h4_p8_s8_ep150_bs2000_lr1e-3_s42 | 0.8592 | 0.7049 | yes |
| patchtst_hybrid_AcTBeCalf_topk_k_10_d128_l3_h4_p8_s8_ep150_bs2000_lr1e-3_s42 | 0.8350 | 0.6274 | yes |
| patchtst_hybrid_AcTBeCalf_topk_k_13_d128_l3_h4_p8_s8_ep150_bs2000_lr1e-3_s42 | 0.7874 | 0.5694 | yes |
| patchtst_hybrid_AcTBeCalf_tamanho_modelo_d64_l2_h2_p4_s4_ep150_bs516_lr1e-3_s42 | 0.8218 | 0.5484 | yes |
| patchtst_hybrid_AcTBeCalf_tamanho_modelo_d64_l2_h4_p4_s4_ep150_bs516_lr1e-3_s42 | 0.8162 | 0.5359 | yes |
| patchtst_hybrid_AcTBeCalf_tamanho_modelo_d128_l3_h2_p8_s8_ep150_bs516_lr1e-3_s42 | 0.8140 | 0.5309 | yes |
| patchtst_hybrid_AcTBeCalf_tamanho_modelo_d256_l3_h4_p8_s8_ep150_bs516_lr1e-3_s42 | 0.8060 | 0.5281 | yes |
| patchtst_hybrid_AcTBeCalf_tamanho_modelo_d128_l2_h4_p8_s8_ep150_bs516_lr1e-3_s42 | 0.8095 | 0.5242 | yes |
| patchtst_hybrid_AcTBeCalf_tamanho_modelo_d128_l2_h2_p4_s4_ep150_bs516_lr1e-3_s42 | 0.8076 | 0.5240 | yes |
| patchtst_hybrid_AcTBeCalf_tamanho_modelo_d128_l3_h4_p4_s4_ep150_bs516_lr1e-3_s42 | 0.7950 | 0.5118 | yes |
| patchtst_hybrid_AcTBeCalf_tamanho_modelo_d64_l3_h2_p8_s8_ep150_bs516_lr1e-3_s42 | 0.8156 | 0.5064 | yes |
| patchtst_hybrid_AcTBeCalf_tamanho_modelo_d64_l3_h4_p8_s8_ep150_bs516_lr1e-3_s42 | 0.8140 | 0.5026 | yes |
| patchtst_hybrid_AcTBeCalf_tamanho_modelo_d256_l2_h2_p8_s8_ep150_bs516_lr1e-3_s42 | 0.8110 | 0.5020 | yes |
| patchtst_hybrid_AcTBeCalf_tamanho_modelo_d256_l3_h2_p4_s4_ep150_bs516_lr1e-3_s42 | 0.8121 | 0.4990 | yes |
| patchtst_hybrid_AcTBeCalf_topk_k_15_d128_l3_h4_p8_s8_ep150_bs2000_lr1e-3_s42 | 0.7605 | 0.4955 | yes |
| patchtst_hybrid_AcTBeCalf_tamanho_modelo_d256_l2_h4_p4_s4_ep150_bs516_lr1e-3_s42 | 0.8084 | 0.4933 | yes |
| patchtst_hybrid_AcTBeCalf_tamanho_modelo_d64_l2_h2_p4_s4_ep150_bs2000_lr1e-3_s42 | 0.7822 | 0.4852 | yes |
| patchtst_hybrid_AcTBeCalf_tamanho_modelo_d128_l2_h2_p4_s4_ep150_bs2000_lr1e-3_s42 | 0.7592 | 0.4599 | yes |
| patchtst_hybrid_AcTBeCalf_tamanho_modelo_d64_l2_h4_p4_s4_ep150_bs2000_lr1e-3_s42 | 0.7772 | 0.4538 | yes |
| patchtst_hybrid_AcTBeCalf_tamanho_modelo_d256_l2_h4_p4_s4_ep150_bs2000_lr1e-3_s42 | 0.7576 | 0.4483 | yes |
| patchtst_hybrid_AcTBeCalf_normalizacao_per_window_zscore_d128_l3_h4_p8_s8_ep150_bs2000_lr1e-3_s42 | 0.7437 | 0.4409 | yes |
| patchtst_hybrid_AcTBeCalf_tamanho_modelo_d64_l3_h4_p8_s8_ep150_bs2000_lr1e-3_s42 | 0.7447 | 0.4403 | yes |
| patchtst_hybrid_AcTBeCalf_tamanho_modelo_d256_l3_h2_p4_s4_ep150_bs2000_lr1e-3_s42 | 0.7685 | 0.4388 | yes |
| patchtst_hybrid_AcTBeCalf_tamanho_modelo_d128_l3_h2_p8_s8_ep150_bs2000_lr1e-3_s42 | 0.7624 | 0.4340 | yes |
| patchtst_hybrid_AcTBeCalf_tamanho_modelo_d64_l3_h2_p8_s8_ep150_bs2000_lr1e-3_s42 | 0.7498 | 0.4230 | yes |
| patchtst_hybrid_AcTBeCalf_tamanho_modelo_d256_l2_h2_p8_s8_ep150_bs2000_lr1e-3_s42 | 0.7430 | 0.4129 | yes |
| patchtst_hybrid_AcTBeCalf_mixing_proxy_d128_l3_h4_p8_s8_ep150_bs2000_lr1e-3_s42 | 0.7472 | 0.4086 | yes |
| patchtst_hybrid_AcTBeCalf_normalizacao_global_pipeline_d128_l3_h4_p8_s8_ep150_bs2000_lr1e-3_s42 | 0.7472 | 0.4086 | yes |
| patchtst_hybrid_AcTBeCalf_topk_k_18_d128_l3_h4_p8_s8_ep150_bs2000_lr1e-3_s42 | 0.7472 | 0.4086 | yes |
| patchtst_hybrid_AcTBeCalf_tamanho_modelo_d256_l3_h4_p8_s8_ep150_bs2000_lr1e-3_s42 | 0.7365 | 0.4056 | yes |
| patchtst_hybrid_AcTBeCalf_tamanho_modelo_d128_l3_h4_p4_s4_ep150_bs2000_lr1e-3_s42 | 0.7292 | 0.3949 | yes |
| patchtst_hybrid_AcTBeCalf_tamanho_modelo_d128_l2_h4_p8_s8_ep150_bs2000_lr1e-3_s42 | 0.7193 | 0.3934 | yes |
| patchtst_deep_only_AcTBeCalf_nopre_d128_l3_h4_p8_s8_ep150_bs256_lr1e-3_s42 | 0.3930 | 0.0767 | yes |
| patchtst_deep_only_AcTBeCalf_nopre_d128_l3_h4_p8_s8_ep150_bs256_lr1e-3_s123 | 0.3563 | 0.0755 | yes |
| patchtst_deep_only_AcTBeCalf_nopre_d128_l3_h4_p8_s8_ep150_bs516_lr1e-3_s456 | 0.3650 | 0.0749 | yes |
| patchtst_deep_only_AcTBeCalf_nopre_d128_l3_h4_p8_s8_ep150_bs256_lr1e-3_s456 | 0.4079 | 0.0703 | yes |
| patchtst_deep_only_AcTBeCalf_nopre_d128_l3_h4_p8_s8_ep150_bs516_lr1e-3_s42 | 0.3708 | 0.0687 | yes |
| patchtst_deep_only_AcTBeCalf_nopre_d128_l3_h4_p8_s8_ep150_bs516_lr1e-3_s123 | 0.1926 | 0.0346 | yes |
| patchtst_deep_only_AcTBeCalf_mixing_proxy_d128_l3_h4_p8_s8_ep150_bs2000_lr1e-3_s42 | 0.1500 | 0.0322 | yes |
| patchtst_pretrain_AcTBeCalf_mixing_proxy_d128_l3_h4_p8_s8_s42 |  |  | yes |
| patchtst_pretrain_AcTBeCalf_normalizacao_global_pipeline_d128_l3_h4_p8_s8_s42 |  |  | yes |
| patchtst_pretrain_AcTBeCalf_normalizacao_per_window_zscore_d128_l3_h4_p8_s8_s42 |  |  | yes |
| patchtst_pretrain_AcTBeCalf_tamanho_modelo_d128_l2_h2_p4_s4_s42 |  |  | yes |
| patchtst_pretrain_AcTBeCalf_tamanho_modelo_d128_l2_h4_p8_s8_s42 |  |  | yes |
| patchtst_pretrain_AcTBeCalf_tamanho_modelo_d128_l3_h2_p8_s8_s42 |  |  | yes |
| patchtst_pretrain_AcTBeCalf_tamanho_modelo_d128_l3_h4_p4_s4_s42 |  |  | yes |
| patchtst_pretrain_AcTBeCalf_tamanho_modelo_d256_l2_h2_p8_s8_s42 |  |  | yes |
| patchtst_pretrain_AcTBeCalf_tamanho_modelo_d256_l2_h4_p4_s4_s42 |  |  | yes |
| patchtst_pretrain_AcTBeCalf_tamanho_modelo_d256_l3_h2_p4_s4_s42 |  |  | yes |
| patchtst_pretrain_AcTBeCalf_tamanho_modelo_d256_l3_h4_p8_s8_s42 |  |  | yes |
| patchtst_pretrain_AcTBeCalf_tamanho_modelo_d64_l2_h2_p4_s4_s42 |  |  | yes |
| patchtst_pretrain_AcTBeCalf_tamanho_modelo_d64_l2_h4_p4_s4_s42 |  |  | yes |
| patchtst_pretrain_AcTBeCalf_tamanho_modelo_d64_l3_h2_p8_s8_s42 |  |  | yes |
| patchtst_pretrain_AcTBeCalf_tamanho_modelo_d64_l3_h4_p8_s8_s42 |  |  | yes |
| patchtst_pretrain_AcTBeCalf_topk_k_05_d128_l3_h4_p8_s8_s42 |  |  | yes |
| patchtst_pretrain_AcTBeCalf_topk_k_07_d128_l3_h4_p8_s8_s42 |  |  | yes |
| patchtst_pretrain_AcTBeCalf_topk_k_10_d128_l3_h4_p8_s8_s42 |  |  | yes |
| patchtst_pretrain_AcTBeCalf_topk_k_13_d128_l3_h4_p8_s8_s42 |  |  | yes |
| patchtst_pretrain_AcTBeCalf_topk_k_15_d128_l3_h4_p8_s8_s42 |  |  | yes |
| patchtst_pretrain_AcTBeCalf_topk_k_18_d128_l3_h4_p8_s8_s42 |  |  | yes |
