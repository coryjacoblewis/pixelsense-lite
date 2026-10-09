# Model Card & Data Card: PixelSense-Lite (`v2_robust_augmented`)

**Artifact:** [`models/v2_robust_augmented/model_int8.tflite`](../models/v2_robust_augmented/model_int8.tflite)

---

## 1. Model Card (`pixelsense_lite_cnn_int8`)

| Attribute | Specification |
| :--- | :--- |
| **Architecture** | 3-Layer 2D CNN (`Conv2D(16)` -> `MaxPool2D` -> `Conv2D(32)` -> `MaxPool2D` -> `Conv2D(32)` -> `GlobalAveragePooling2D` -> `Dense(32)` -> `Softmax(5)`) |
| **Frontend & I/O** | Blind 2.0s peak-energy window @ 16 kHz (`n_fft=512, hop=500`, 64-band Log-Mel in `[0, 1]`); `int8[1, 64, 64, 1]` -> `int8[1, 5]` (`background_noise`, `coughing`, `snoring`, `siren`, `crying_baby`) |
| **Subgraph Ops** | `TFLITE_BUILTINS_INT8` (`CONV_2D v3`, `MAX_POOL_2D v2`, `MEAN v2`, `FULLY_CONNECTED v4`, `SOFTMAX v2`); 21 tensors (`17` `int8`, `4` `int32` bias, `0` non-integer tensors) |
| **Prior & Weighting** | Trained directly under natural empirical class priors (53.6% `background_noise` in Folds 1–3, 75/140 clips) with OOF noisy-label down-weighting (`w = 0.35`), avoiding `class_weight='balanced'` prior distortion and requiring zero post-hoc threshold tuning |
| **Footprint** | 22.98 KB (23,536 B) `.tflite` flatbuffer (`<= 45 KB`); 147.33 KB subgraph tensor descriptors (`<= 160 KB`); 80.00 KB peak op I/O (`conv1`, `<= 100 KB`); 256.0 KB raw CPU (`3.27 MB` process RSS delta) / 512.0 KB XNNPACK (`4.15 MB` process RSS delta) `AllocateTensors` session-init heap (`<= 512 KB`) |
| **ARM64 Latency** | 172.38 µs avg / 177 µs p95 (`<= 1.0 ms` `MAX_ARM64_P95_LATENCY_MS`) on Linux `aarch64` XNNPACK (`benchmark_model`, 1 thread, 5,725 runs on `ubuntu-24.04-arm`, SHA-256 verified) |
| **Scope** | Benchmark evaluation for ARM64 application-processor audio classification (`clean`, `pocket_occluded`, `appliance_noise_3db`). Non-medical / non-safety-critical. |

---

## 2. Data Card (`ESC-50 Audited Subset`)

| Field | Details |
| :--- | :--- |
| **Source Corpus** | [ESC-50](https://github.com/karolpiczak/ESC-50) (Piczak, 2015; Freesound.org `CC-BY 3.0` / `CC0` / `CC-BY-NC`). |
| **Ingested Subset** | 320 clips (5.0s, 44.1 kHz WAV resampled to 16 kHz mono) across 8 categories: 233 `PASS`, 87 `QUARANTINE` (75 clipping saturation, 10 DC offset, 2 dead air). |
| **3-Way Partition** | Disjoint by `src_file`: **Folds 1–3** Train (`N = 140 -> 700` views in `v2_robust_augmented`), **Fold 4** Val & PSI Audit (`N = 43`/slice), **Fold 5** Locked Test (`N = 50`/slice, 42 `src_file` sources, 25 BG clips). See [`docs/data_collection_sop.md`](./data_collection_sop.md). |
| **Limitations** | Degraded slices (`pocket_occluded`, `appliance_noise_3db`) use synthetic DSP transforms on ESC-50 clips and OOF label-noise screening uses an automated RF probe rather than human relabelers; production deployment requires multi-room hardware capture and human annotation. |
