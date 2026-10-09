| release             | quant   |   flash_kb |   subgraph_tensor_kb |   peak_op_io_kb |   non_int_tensors |   int_tensor_% |   p99_latency_ms |   f1_clean_% |   f1_pocket_occluded_% |   f1_appliance_noise_3db_% |   bg_fpr_% |   max_slice_bg_fpr_% | gate_status          |
|:--------------------|:--------|-----------:|---------------------:|----------------:|------------------:|---------------:|-----------------:|-------------:|-----------------------:|---------------------------:|-----------:|---------------------:|:---------------------|
| v1_baseline         | fp32    |      64.08 |               587.94 |             320 |                20 |            4.8 |            0.378 |        83.54 |                  46.62 |                      25.34 |      33.33 |                   88 | BLOCKED (HW + SLICE) |
| v1_baseline         | fp16    |      36.04 |               617.76 |             320 |                30 |            3.2 |            0.364 |        83.54 |                  46.62 |                      25.34 |      33.33 |                   88 | BLOCKED (HW + SLICE) |
| v1_baseline         | int8    |      22.98 |               147.33 |              80 |                 0 |          100   |            0.157 |        82.77 |                  46.62 |                      25.51 |      30.67 |                   88 | BLOCKED (SLICE F1)   |
| v2_robust_augmented | fp32    |      64.08 |               587.94 |             320 |                20 |            4.8 |            1.323 |        90.9  |                  79.47 |                      63.69 |       4    |                    8 | BLOCKED (HW)         |
| v2_robust_augmented | fp16    |      36.04 |               617.76 |             320 |                30 |            3.2 |            0.329 |        90.9  |                  79.47 |                      63.69 |       4    |                    8 | BLOCKED (HW)         |
| v2_robust_augmented | int8    |      22.98 |               147.33 |              80 |                 0 |          100   |            0.128 |        89.03 |                  82.62 |                      63.69 |       4    |                    8 | SHIP (PASS)          |

### Source-Clustered Bootstrap 95% CIs & Per-Slice BG FPR (Fold 5: N=50 clips across 42 `src_file` sources)

| locked_fold5_slice   | v1_int8_95%_ci   | v2_int8_95%_ci   | paired_delta_f1_%   | paired_delta_95%_ci   | v1_bg_fpr_%   | v2_bg_fpr_%   |
|:---------------------|:-----------------|:-----------------|:--------------------|:----------------------|:--------------|:--------------|
| clean                | [69.62%, 93.25%] | [81.88%, 96.85%] | +6.26%              | [-4.78%, +21.45%]     | 4.00%         | 8.00%         |
| pocket_occluded      | [33.46%, 74.41%] | [72.76%, 96.34%] | +36.00%             | [+14.64%, +48.58%]    | 88.00%        | 0.00%         |
| appliance_noise_3db  | [13.70%, 33.16%] | [50.79%, 79.56%] | +38.18%             | [+23.17%, +57.90%]    | 0.00%         | 4.00%         |

### Live CI ARM64 Benchmark (`benchmark_model`)

- **Verified `.tflite` SHA-256**: `f54324477dabc02c50189e8825763c6420a2752d95fc9583e5464bac299d7829`
- **Raw ARM64 CPU (1 Thread)**: avg `0.301 ms`, p95 `0.310 ms` (3262 runs, `AllocateTensors=256.0 KB`, process RSS delta `3.27 MB`)
- **ARM64 XNNPACK Delegate (1 Thread)**: avg `0.172 ms`, p95 `0.177 ms` (5725 runs, delegate_applied=True, `AllocateTensors=512.0 KB`, process RSS delta `4.15 MB`)
