| release             | quant   |   flash_kb |   subgraph_tensor_kb |   peak_op_io_kb |   non_int_tensors |   int_tensor_% |   p99_latency_ms |   f1_clean_% |   f1_pocket_occluded_% |   f1_appliance_noise_3db_% |   bg_fpr_% |   max_slice_bg_fpr_% | gate_status          |
|:--------------------|:--------|-----------:|---------------------:|----------------:|------------------:|---------------:|-----------------:|-------------:|-----------------------:|---------------------------:|-----------:|---------------------:|:---------------------|
| v1_baseline         | fp32    |      64.08 |               587.94 |             320 |                20 |            4.8 |            2.162 |        83.54 |                  46.62 |                      25.34 |      33.33 |                   88 | BLOCKED (HW + SLICE) |
| v1_baseline         | fp16    |      36.04 |               617.76 |             320 |                30 |            3.2 |            0.325 |        83.54 |                  46.62 |                      25.34 |      33.33 |                   88 | BLOCKED (HW + SLICE) |
| v1_baseline         | int8    |      22.98 |               147.33 |              80 |                 0 |          100   |            0.125 |        82.77 |                  46.62 |                      25.51 |      30.67 |                   88 | BLOCKED (SLICE F1)   |
| v2_robust_augmented | fp32    |      64.08 |               587.94 |             320 |                20 |            4.8 |            0.34  |        92.18 |                  83.52 |                      63.69 |       5.33 |                    8 | BLOCKED (HW)         |
| v2_robust_augmented | fp16    |      36.04 |               617.76 |             320 |                30 |            3.2 |            0.329 |        92.18 |                  83.52 |                      63.69 |       5.33 |                    8 | BLOCKED (HW)         |
| v2_robust_augmented | int8    |      22.98 |               147.33 |              80 |                 0 |          100   |            0.114 |        92.18 |                  85.02 |                      60.64 |       4    |                    8 | SHIP (PASS)          |

### Source-Clustered Bootstrap 95% CIs & Per-Slice BG FPR (Fold 5: N=50 clips across 42 `src_file` sources)

| locked_fold5_slice   | v1_int8_95%_ci   | v2_int8_95%_ci   | paired_delta_f1_%   | paired_delta_95%_ci   | v1_bg_fpr_%   | v2_bg_fpr_%   |
|:---------------------|:-----------------|:-----------------|:--------------------|:----------------------|:--------------|:--------------|
| clean                | [69.62%, 93.25%] | [85.17%, 98.40%] | +9.41%              | [-1.59%, +23.80%]     | 4.00%         | 8.00%         |
| pocket_occluded      | [33.46%, 74.41%] | [76.14%, 98.26%] | +38.40%             | [+18.23%, +49.53%]    | 88.00%        | 0.00%         |
| appliance_noise_3db  | [13.70%, 33.16%] | [46.79%, 76.58%] | +35.13%             | [+18.68%, +55.80%]    | 0.00%         | 4.00%         |

### Live CI ARM64 Benchmark (`benchmark_model`)

- **Verified `.tflite` SHA-256**: `0a2d18b832711b56d1e5356ecae25d50d8829822ae4a007bc0eeea6934cbf0cf`
- **Raw ARM64 CPU (1 Thread)**: avg `0.301 ms`, p95 `0.310 ms` (3262 runs, `AllocateTensors=256.0 KB`, process RSS delta `3.27 MB`)
- **ARM64 XNNPACK Delegate (1 Thread)**: avg `0.172 ms`, p95 `0.177 ms` (5725 runs, delegate_applied=True, `AllocateTensors=512.0 KB`, process RSS delta `4.15 MB`)
