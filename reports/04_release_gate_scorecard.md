| release             | quant   |   flash_kb |   subgraph_tensor_kb |   peak_op_io_kb |   non_int_tensors |   int_tensor_% |   p99_latency_ms |   f1_clean_% |   f1_pocket_occluded_% |   f1_appliance_noise_3db_% |   bg_fpr_% |   max_slice_bg_fpr_% | gate_status          |
|:--------------------|:--------|-----------:|---------------------:|----------------:|------------------:|---------------:|-----------------:|-------------:|-----------------------:|---------------------------:|-----------:|---------------------:|:---------------------|
| v1_baseline         | fp32    |      64.08 |               587.94 |             320 |                20 |            4.8 |            0.361 |        79.49 |                  46.62 |                      24.33 |      36    |                   88 | BLOCKED (HW + SLICE) |
| v1_baseline         | fp16    |      36.04 |               617.76 |             320 |                30 |            3.2 |            0.345 |        79.49 |                  46.62 |                      24.33 |      36    |                   88 | BLOCKED (HW + SLICE) |
| v1_baseline         | int8    |      22.98 |               147.33 |              80 |                 0 |          100   |            0.152 |        80.69 |                  46.62 |                      24.33 |      34.67 |                   88 | BLOCKED (SLICE F1)   |
| v2_robust_augmented | fp32    |      64.08 |               587.94 |             320 |                20 |            4.8 |            0.311 |        90.35 |                  82.62 |                      55.21 |       2.67 |                    4 | BLOCKED (HW)         |
| v2_robust_augmented | fp16    |      36.04 |               617.76 |             320 |                30 |            3.2 |            0.332 |        90.35 |                  82.62 |                      55.21 |       2.67 |                    4 | BLOCKED (HW)         |
| v2_robust_augmented | int8    |      22.98 |               147.33 |              80 |                 0 |          100   |            0.111 |        86.7  |                  82.62 |                      55.21 |       4    |                    8 | SHIP (PASS)          |

### Source-Clustered Bootstrap 95% CIs & Per-Slice BG FPR (Fold 5: N=50 clips across 42 `src_file` sources)

| locked_fold5_slice   | v1_int8_95%_ci   | v2_int8_95%_ci   | paired_delta_f1_%   | paired_delta_95%_ci   | v1_bg_fpr_%   | v2_bg_fpr_%   |
|:---------------------|:-----------------|:-----------------|:--------------------|:----------------------|:--------------|:--------------|
| clean                | [68.80%, 92.87%] | [74.42%, 96.37%] | +6.01%              | [-12.59%, +21.35%]    | 12.00%        | 8.00%         |
| pocket_occluded      | [33.46%, 74.41%] | [72.76%, 96.34%] | +36.00%             | [+14.64%, +48.58%]    | 88.00%        | 0.00%         |
| appliance_noise_3db  | [14.05%, 32.45%] | [43.58%, 68.05%] | +30.88%             | [+18.40%, +45.15%]    | 4.00%         | 4.00%         |

### Cached CI Reference ARM64 Telemetry (`reports/05_arm64_hardware_telemetry_int8.txt`)

- **Verified `.tflite` SHA-256**: `32be051b97adf504969b472e3c86b0f4474de7456f0426b853da2a2975d33362`
- **Raw ARM64 CPU (1 Thread)**: avg `0.301 ms`, p95 `0.310 ms` (3262 runs, `AllocateTensors=256.0 KB`, process RSS delta `3.27 MB`)
- **ARM64 XNNPACK Delegate (1 Thread)**: avg `0.172 ms`, p95 `0.177 ms` (5725 runs, delegate_applied=True, `AllocateTensors=512.0 KB`, process RSS delta `4.15 MB`)
