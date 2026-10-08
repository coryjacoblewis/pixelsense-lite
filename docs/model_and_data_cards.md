# Model Card & Data Card: PixelSense-Lite (`v2_data_flywheel`)

**Artifact:** [`models/v2_data_flywheel/model_int8.tflite`](../models/v2_data_flywheel/model_int8.tflite)

---

## 1. Model Card (`pixelsense_lite_cnn_int8`)

| Attribute | Specification |
| :--- | :--- |
| **Architecture** | 3-Layer 2D CNN (`Conv2D(16)` -> `MaxPool2D` -> `Conv2D(32)` -> `MaxPool2D` -> `Conv2D(32)` -> `GlobalAveragePooling2D` -> `Dense(32)` -> `Softmax(5)`) |
| **I/O Tensors** | `int8[1, 64, 64, 1]` (2.0s @ 16 kHz, 64-band Log-Mel) -> `int8[1, 5]` (`background_noise`, `coughing`, `snoring`, `siren`, `crying_baby`) |
| **Subgraph Ops** | `TFLITE_BUILTINS_INT8` (`CONV_2D v3`, `MAX_POOL_2D v2`, `MEAN v2`, `FULLY_CONNECTED v4`, `SOFTMAX v2`); `0` float fallback tensors |
| **Memory & Power** | `22.98 KB` (`23,536 B`) Flash; `147.33 KB` static subgraph tensor sum; `0.001%` 24h DSP battery drain (@ 43,200 runs/day) |
| **Intended Scope** | On-device ambient audio event classification (`clean`, `pocket_occluded`, `appliance_noise_3db`). Not a medical or life-safety device. |

### Reference Harness Notes
- **Frontend**: Window-peak normalized Log-Mel (`librosa.power_to_db(ref=np.max)`).
- **Arena Accounting**: `04_release_gate.py` sums static subgraph tensor descriptors (`147.33 KB` `int8`) rather than live activation buffer reuse.
- **Evaluation**: Held-out `Fold 5` (`N = 49` clips/slice); see [`README.md`](../README.md) and [`reports/04_release_gate_scorecard.md`](../reports/04_release_gate_scorecard.md).

---

## 2. Data Card (`ESC-50 Audited Subset`)

| Field | Details |
| :--- | :--- |
| **Upstream Corpus** | [ESC-50](https://github.com/karolpiczak/ESC-50) (Piczak, 2015), sourced from Freesound.org (`CC-BY 3.0` / `CC0` / `CC-BY-NC`). |
| **Ingested Subset** | 320 recordings (5.0s, 44.1 kHz WAV resampled to 16 kHz mono) across 8 categories (`230 PASS`, `90 QUARANTINED`). |
| **Partitioning** | `Folds 1-4` (`181` clean clips -> `905` views in `v2`) isolated by `src_file` from `Fold 5` (`49` clips/slice). See [`docs/data_collection_sop.md`](./data_collection_sop.md). |
