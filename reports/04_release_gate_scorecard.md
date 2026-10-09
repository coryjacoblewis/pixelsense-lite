| release             | quant   |   flash_kb |   subgraph_tensor_kb |   peak_op_io_kb |   non_int_tensors |   int_tensor_% |   p99_latency_ms |   f1_clean_% |   f1_pocket_occluded_% |   f1_appliance_noise_3db_% |   bg_fpr_% |   max_slice_bg_fpr_% | gate_status          |
|:--------------------|:--------|-----------:|---------------------:|----------------:|------------------:|---------------:|-----------------:|-------------:|-----------------------:|---------------------------:|-----------:|---------------------:|:---------------------|
| v1_baseline         | fp32    |      64.08 |               587.94 |             320 |                20 |            4.8 |            0.344 |        83.54 |                  46.62 |                      25.34 |      33.33 |                   88 | BLOCKED (HW + SLICE) |
| v1_baseline         | fp16    |      36.04 |               617.76 |             320 |                30 |            3.2 |            0.359 |        83.54 |                  46.62 |                      25.34 |      33.33 |                   88 | BLOCKED (HW + SLICE) |
| v1_baseline         | int8    |      22.98 |               147.33 |              80 |                 0 |          100   |            0.145 |        82.77 |                  46.62 |                      25.51 |      30.67 |                   88 | BLOCKED (SLICE F1)   |
| v2_robust_augmented | fp32    |      64.08 |               587.94 |             320 |                20 |            4.8 |            0.319 |        88.47 |                  79.47 |                      59.42 |       2.67 |                    4 | BLOCKED (HW)         |
| v2_robust_augmented | fp16    |      36.04 |               617.76 |             320 |                30 |            3.2 |            0.315 |        88.47 |                  79.47 |                      59.42 |       2.67 |                    4 | BLOCKED (HW)         |
| v2_robust_augmented | int8    |      22.98 |               147.33 |              80 |                 0 |          100   |            0.12  |        88.47 |                  82.62 |                      59.42 |       2.67 |                    4 | SHIP (PASS)          |

### Source-Clustered Bootstrap 95% CIs & Per-Slice BG FPR (Fold 5: N=50 clips across 42 `src_file` sources)

| locked_fold5_slice   | v1_int8_95%_ci   | v2_int8_95%_ci   | paired_delta_f1_%   | paired_delta_95%_ci   | v1_bg_fpr_%   | v2_bg_fpr_%   |
|:---------------------|:-----------------|:-----------------|:--------------------|:----------------------|:--------------|:--------------|
| clean                | [69.62%, 93.25%] | [80.14%, 96.37%] | +5.70%              | [-4.78%, +20.20%]     | 4.00%         | 4.00%         |
| pocket_occluded      | [33.46%, 74.41%] | [72.76%, 96.34%] | +36.00%             | [+14.64%, +48.58%]    | 88.00%        | 0.00%         |
| appliance_noise_3db  | [13.70%, 33.16%] | [48.77%, 74.62%] | +33.91%             | [+20.40%, +52.22%]    | 0.00%         | 4.00%         |

### Live CI ARM64 Benchmark (`benchmark_model`)

- **Verified `.tflite` SHA-256**: `1e5ddab618a1c612764656dbc2dda23bfc3ecbdfae9b013fdd5129a5d1e3f4a0`
- **Raw ARM64 CPU (1 Thread)**: avg `0.301 ms`, p95 `0.310 ms` (3262 runs, `AllocateTensors=256.0 KB`, process RSS delta `3.27 MB`)
- **ARM64 XNNPACK Delegate (1 Thread)**: avg `0.172 ms`, p95 `0.177 ms` (5725 runs, delegate_applied=True, `AllocateTensors=512.0 KB`, process RSS delta `4.15 MB`)
