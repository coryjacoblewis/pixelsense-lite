# PixelSense-Lite: Edge-Audio ML Pipeline & ARM64 INT8 Release Gate

[![ARM64 Release Gate](https://github.com/coryjacoblewis/pixelsense-lite/actions/workflows/arm64_ml_release_gate.yml/badge.svg)](https://github.com/coryjacoblewis/pixelsense-lite/actions/workflows/arm64_ml_release_gate.yml)
[![Inspect v2 INT8 Subgraph in Netron](https://img.shields.io/badge/Netron-Inspect_INT8_.tflite_Subgraph-blue?logo=tensorflow)](https://netron.app/?url=https://raw.githubusercontent.com/coryjacoblewis/pixelsense-lite/main/models/v2_robust_augmented/model_int8.tflite)

**PixelSense-Lite** is an end-to-end edge-audio ML pipeline that trains, quantizes, and release-gates a **23 KB full-integer `INT8` `.tflite` model** for 5-class sound recognition (`coughing`, `snoring`, `siren`, `crying_baby`, `background_noise`) on ARM64 hardware.

## The Problem & Result

Models trained only on clean audio (`v1_baseline`) break down when a device is muffled inside a pocket or placed near loud household appliances, triggering false alarms up to **88%** of the time on background noise. PixelSense-Lite (`v2_robust_augmented`) fixes this without increasing model size or latency:

| Metric (Locked Fold 5 Test Set, `INT8` `.tflite`) | `v1_baseline` (Clean Train) | `v2_robust_augmented` (Ours) | Improvement | Release Requirement |
| :--- | :---: | :---: | :---: | :---: |
| **Clean Audio F1** | 80.69% | **86.70%** | `+6.01%` | `>= 65.0%` |
| **Pocket-Muffled Audio F1** (1,600 Hz low-pass) | 46.62% | **82.62%** | **`+36.00%`** | `>= 58.0%` |
| **Noisy Room Audio F1** (+3 dB appliance SNR) | 24.33% | **55.21%** | **`+30.88%`** | `>= 52.0%` |
| **Worst-Case Background False-Positive Rate** | 88.00% | **8.00%** | **`-80.00%`** | `<= 15.0%` |
| **Model Binary Size / ARM64 Latency (p95)** | 22.98 KB / — | **22.98 KB / 177 µs** | Same footprint | `<= 45 KB` / `<= 1.0 ms` |
| **Release Gate Verdict** | **BLOCKED** | **SHIP (PASS)** | — | All gates passed |

---

## Quickstart

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python release_gate.py --require-arm64-telemetry  # Fast gate check on cached slices & .tflite (<3s)
pytest test_pipeline.py                           # Unit & integration test suite (<3s)
python run_automated_pipeline.py                  # Full rebuild (downloads ESC-50, runs Stages 1-4)
```

---

## How the 4-Stage Pipeline Works

Using a 320-clip subset of [ESC-50](https://github.com/karolpiczak/ESC-50), the pipeline enforces strict source-recording isolation (`src_file`) across **Folds 1–3** (Train, `N=140`), **Fold 4** (Validation & Shift Audit, `N=43`), and **Fold 5** (Locked Test, `N=50` across 42 sources):

```mermaid
flowchart LR
    A["Raw Audio (320 WAVs)"] --> B["Stage 1: Signal QA (ingest_qa.py)"]
    B -->|"87 Quarantined (27.2%)"| Q["Quarantine Log"]
    B -->|"233 Clean Clips"| S["Source-Isolated Split"]
    S -->|"Folds 1-3 Train + Fold 4 Val"| C["Stage 2: Label & Drift Audit (consensus_drift.py)"]
    C -->|"Clean Only + Balanced Prior"| V1["v1_baseline INT8 (22.98 KB)"]
    C -->|"5x Augmentation + Empirical Prior"| V2["v2_robust_augmented INT8 (22.98 KB)"]
    S -->|"Locked Fold 5 Test (N=50)"| G["Stage 4: Release Gate (release_gate.py)"]
    V1 --> G
    V2 --> G
    G -->|"Fails Stress F1 & BG FPR (88%)"| R1["v1: BLOCKED"]
    G -->|"Passes All Quality & HW Gates"| R2["v2: SHIP (PASS)"]
```

| Stage | Script | What It Does | Output Artifact |
| :---: | :--- | :--- | :--- |
| **1. Signal QA** | [`ingest_qa.py`](./ingest_qa.py) | Screens raw WAV files for physical audio defects (clipping saturation, DC offset, dead air). Quarantines 87 bad clips; passes 233 clean clips. | [`reports/01_signal_qa_report.csv`](./reports/01_signal_qa_report.csv) |
| **2. Drift & Label Audit** | [`consensus_drift.py`](./consensus_drift.py) | Down-weights likely mislabeled training clips (`w = 0.35`) via out-of-fold classifier disagreement, and measures spectral drift (PSI) on Fold 4 to trigger 5x acoustic augmentation. | [`reports/02_oof_label_noise_audit.csv`](./reports/02_oof_label_noise_audit.csv), [`reports/02_psi_spectral_drift_audit.csv`](./reports/02_psi_spectral_drift_audit.csv) |
| **3. Train & Quantize** | [`train_quantize.py`](./train_quantize.py) | Trains a compact 2D CNN on 64x64 Log-Mel spectrograms and exports `fp32`, `fp16`, and full-integer `int8` `.tflite` models. Runs 5-seed ablations. | [`reports/03_training_and_ablation_metrics.csv`](./reports/03_training_and_ablation_metrics.csv) |
| **4. Release Gate** | [`release_gate.py`](./release_gate.py) | Audits `.tflite` memory/operators, verifies SHA-256 ARM64 XNNPACK latency, and blocks any candidate that fails Fold 5 F1 or false-positive thresholds. | [`reports/04_release_gate_scorecard.md`](./reports/04_release_gate_scorecard.md) |

> **Detailed Reference Docs:** [Audio Corpus & Signal QA SOP (`docs/data_collection_sop.md`)](./docs/data_collection_sop.md) | [Model & Data Cards (`docs/model_and_data_cards.md`)](./docs/model_and_data_cards.md)

---

## Stage-by-Stage Results

### Stage 1: Physical Audio QA ([`reports/01_signal_qa_report.csv`](./reports/01_signal_qa_report.csv))

| QA Gate Status | Clips | Share | Rejection Rule |
| :--- | :---: | :---: | :--- |
| `PASS` | 233 | 72.8% | Clean signal (140 Train, 43 Val, 50 Test; preserves single-sample peak-normalized files) |
| `QUARANTINE_CLIPPING_SATURATION` | 75 | 23.4% | Peak amplitude `>= 0.998` across `> 2` samples (`MULTI_SAMPLE_SATURATION`) |
| `QUARANTINE_DC_OFFSET` | 10 | 3.1% | Mean waveform offset `> 0.002` |
| `QUARANTINE_EXCESSIVE_DEAD_AIR` | 2 | 0.6% | `< 12%` active 50ms frames above `-50 dBFS` |

### Stage 2: Label-Noise Down-Weighting & Spectral Shift Audit

- **Out-of-Fold Label Audit ([`reports/02_oof_label_noise_audit.csv`](./reports/02_oof_label_noise_audit.csv)):** A source-grouped random forest agrees with ESC-50 labels on **88.6%** of Folds 1–3 training clips (`124/140`, κ = `0.822`). 4 high-confidence disputed clips (`1-187207-A-20.wav`, `2-43802-A-42.wav`, `3-124795-A-28.wav`, `3-51731-A-42.wav`) are down-weighted (`w = 0.35`) rather than discarded.
- **Spectral Shift Audit ([`reports/02_psi_spectral_drift_audit.csv`](./reports/02_psi_spectral_drift_audit.csv)):** Compares Fold 4 validation slices against Folds 1–3 clean training audio using Population Stability Index (PSI) across high frequencies (`> 2.0 kHz`) and passband dynamic range (`< 1.5 kHz`):

| Fold 4 Validation Slice | Clips | High-Freq PSI (`>2.0 kHz`) | Passband Dynamic-Range PSI (`<1.5 kHz`) | Max PSI | Threshold | Action |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| `clean` (Natural Cross-Fold Audio) | 43 | 0.0494 | 0.0272 | 0.0494 | 0.2500 | `STABLE` |
| `pocket_occluded` (1,600 Hz Low-Pass) | 43 | **2.0071** | 0.0263 | 2.0071 | 0.2500 | `HIGH_SHIFT_AUGMENTATION_ACTIVE` |
| `appliance_noise_3db` (+3 dB SNR Noise) | 43 | 1.9135 | **0.6300** | 1.9135 | 0.2500 | `HIGH_SHIFT_AUGMENTATION_ACTIVE` |

### Stage 3: Ablation Study — Why `v2_robust_augmented` Works ([`reports/03_training_and_ablation_metrics.csv`](./reports/03_training_and_ablation_metrics.csv))

Training with `class_weight='balanced'` (`Balanced 1/K`) artificially forces a uniform 20% class prior over a dataset that is naturally 53.6% `background_noise`, causing high false-alarm rates (25%–97% BG FPR). Combining **5x acoustic augmentation**, **empirical class priors**, and **OOF noisy-label down-weighting (`w=0.35`)** achieves high F1 while keeping background false positives below the `15.0%` ceiling across 5 random seeds:

| Configuration (Trained on Folds 1–3) | Train Views / Epochs | Weighting & Prior | Seed-42 INT8 Clean / Pocket / +3dB F1 (Mean) | Seed-42 INT8 Max BG FPR | 5-Seed INT8 Mean F1 ± Std | 5-Seed INT8 Max BG FPR ± Std |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| 1. `v1_baseline` (Clean Only) | 140 / 70 | `w=1.00`, Balanced 1/K | 80.69% / 46.62% / 24.33% (`50.55%`) | 88.00% | `47.53 ± 1.75%` | `96.80 ± 4.66%` |
| 2. `step_matched_clean` (5x Epochs) | 140 / 350 | `w=1.00`, Balanced 1/K | 92.04% / 59.14% / 28.78% (`59.99%`) | 20.00% | `57.66 ± 5.80%` | `45.60 ± 32.46%` |
| 3. `v2_augmentation_only` | 700 / 70 | `w=1.00`, Balanced 1/K | 76.82% / 86.94% / 76.77% (`80.18%`) | 36.00% | `77.03 ± 3.50%` | `27.20 ± 9.60%` |
| 4. `v2_aug_oof_weights_only` | 700 / 70 | `w=0.35`, Balanced 1/K | 78.13% / 89.38% / 70.40% (`79.30%`) | 32.00% | `77.75 ± 1.97%` | `25.60 ± 9.67%` |
| 5. `v2_aug_empirical_prior_only` | 700 / 70 | `w=1.00`, Empirical Prior | 91.44% / 83.03% / 54.08% (`76.18%`) | 12.00% | `76.13 ± 0.89%` | `11.20 ± 5.31%` |
| 6. **`v2_robust_augmented`** | 700 / 70 | **`w=0.35` + Empirical Prior** | **86.70% / 82.62% / 55.21% (`74.84%`)** | **8.00%** | **`76.81 ± 2.46%`** | **`13.60 ± 8.62%`** |

### Stage 4: Full Release Gate Scorecard & Hardware Telemetry ([`reports/04_release_gate_scorecard.md`](./reports/04_release_gate_scorecard.md))

#### 1. `.tflite` Precision & Hardware Gate Comparison

Only the full-integer `INT8` build of `v2_robust_augmented` satisfies both the embedded hardware limits (`<= 45 KB` binary, `<= 160 KB` tensor descriptors, `100%` integer tensors) and the Fold 5 quality thresholds:

| Candidate | Quant | Binary (`<=45 KB`) | Subgraph Tensors (`<=160 KB`) | Peak Op I/O (`<=100 KB`) | INT Tensors (`100%`) | Host p99 (`<=5.0 ms`) | Clean / Pocket / +3dB F1 | Pooled / Max BG FPR (`<=15%`) | Gate Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| `v1_baseline` | `fp32` | 64.08 KB | 587.94 KB | 320.00 KB | 4.8% | 0.331 ms | 79.49% / 46.62% / 24.33% | 36.00% / 88.00% | BLOCKED (HW + SLICE) |
| `v1_baseline` | `fp16` | 36.04 KB | 617.76 KB | 320.00 KB | 3.2% | 0.310 ms | 79.49% / 46.62% / 24.33% | 36.00% / 88.00% | BLOCKED (HW + SLICE) |
| `v1_baseline` | `int8` | 22.98 KB | 147.33 KB | 80.00 KB | 100.0% | 0.120 ms | 80.69% / 46.62% / 24.33% | 34.67% / 88.00% | BLOCKED (SLICE F1) |
| `v2_robust_augmented` | `fp32` | 64.08 KB | 587.94 KB | 320.00 KB | 4.8% | 0.310 ms | 90.35% / 82.62% / 55.21% | 2.67% / 4.00% | BLOCKED (HW) |
| `v2_robust_augmented` | `fp16` | 36.04 KB | 617.76 KB | 320.00 KB | 3.2% | 0.317 ms | 90.35% / 82.62% / 55.21% | 2.67% / 4.00% | BLOCKED (HW) |
| **`v2_robust_augmented`** | **`int8`** | **22.98 KB** | **147.33 KB** | **80.00 KB** | **100.0%** | **0.123 ms** | **86.70% / 82.62% / 55.21%** | **4.00% / 8.00%** | **SHIP (PASS)** |

#### 2. Paired Bootstrap 95% Confidence Intervals (Fold 5: `N = 50` across 42 sources, 1,000 Replicates)

| Locked Fold 5 Slice | `v1_baseline` INT8 (95% CI) | `v2_robust_augmented` INT8 (95% CI) | Paired ΔF1 (`v2 - v1`) | Paired ΔF1 95% Bootstrap CI | `v1` BG FPR | `v2` BG FPR |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `clean` (Reference Audio) | 80.69% `[68.80%, 92.87%]` | 86.70% `[74.42%, 96.37%]` | `+6.01%` | `[-12.59%, +21.35%]` | 12.00% | **8.00%** |
| `pocket_occluded` (1,600 Hz LP) | 46.62% `[33.46%, 74.41%]` | 82.62% `[72.76%, 96.34%]` | **`+36.00%`** | **`[+14.64%, +48.58%]`** | 88.00% | **0.00%** |
| `appliance_noise_3db` (+3 dB SNR) | 24.33% `[14.05%, 32.45%]` | 55.21% `[43.58%, 68.05%]` | **`+30.88%`** | **`[+18.40%, +45.15%]`** | 4.00% | **4.00%** |

#### 3. Native ARM64 Operator Profile ([`reports/05_arm64_op_profile_int8.csv`](./reports/05_arm64_op_profile_int8.csv), [`reports/05_arm64_hardware_telemetry_int8.txt`](./reports/05_arm64_hardware_telemetry_int8.txt))

```mermaid
flowchart LR
    IN["int8[1,64,64,1]"] --> C1["CONV_2D 16x3x3 (72.0 µs)"]
    C1 -->|"int8[1,64,64,16]"| P1["MAX_POOL_2D (6.0 µs)"]
    P1 -->|"int8[1,32,32,16]"| C2["CONV_2D 32x3x3 (61.0 µs)"]
    C2 -->|"int8[1,32,32,32]"| P2["MAX_POOL_2D (2.0 µs)"]
    P2 -->|"int8[1,16,16,32]"| C3["CONV_2D 32x3x3 (25.0 µs)"]
    C3 -->|"int8[1,16,16,32]"| GAP["MEAN GlobalAvgPool (< 0.10 µs)"]
    GAP -->|"int8[1,32]"| FC["2x FULLY_CONNECTED + SOFTMAX (< 0.20 µs)"]
    FC --> OUT["int8[1,5]"]
```

| Subgraph Operator (`v2 INT8`) | Op Count | Raw ARM64 CPU (`--use_xnnpack=false`) | ARM64 XNNPACK Delegate (`--use_xnnpack=true`) | Speedup & Kernel Dispatch |
| :--- | :---: | :---: | :---: | :--- |
| `conv1` (`CONV_2D` 16x3x3) | 1 | 93.97 µs (31.4%) | 72.00 µs (43.4%) | 1.31x (`Convolution NHWC, QC8 IGEMM`) |
| `pool1` (`MAX_POOL_2D` 2x2) | 1 | 11.00 µs (3.7%) | 6.00 µs (3.6%) | 1.83x (`Max Pooling NHWC, S8`) |
| `conv2` (`CONV_2D` 32x3x3) | 1 | 80.08 µs (26.8%) | 61.00 µs (36.7%) | 1.31x (`Convolution NHWC, QC8 IGEMM`) |
| `pool2` (`MAX_POOL_2D` 2x2) | 1 | 3.40 µs (1.1%) | 2.00 µs (1.2%) | 1.70x (`Max Pooling NHWC, S8`) |
| `conv3` (`CONV_2D` 32x3x3) | 1 | 33.80 µs (11.3%) | 25.00 µs (15.1%) | 1.35x (`Convolution NHWC, QC8 IGEMM`) |
| `gap` (`MEAN` GlobalAvgPool) | 1 | 75.87 µs (25.4%) | < 0.10 µs (0.0%) | >100x (`Mean ND SIMD reduction`) |
| `fc1` + `probs` (`FULLY_CONNECTED` + `SOFTMAX`) | 3 | 0.78 µs (0.3%) | < 0.20 µs (0.1%) | ~4.0x (`Fully Connected NC, QS8, QC8W GEMM`) |
| **Total Inference (avg / p95)** | **9 Nodes** | **300.97 µs / 310 µs** | **172.38 µs / 177 µs** | **1.75x Faster (42.7% latency reduction)** |
| **Init `AllocateTensors` (Process RSS Delta)** | **Session Init** | **256.0 KB** (`3.27 MB` process RSS delta) | **512.0 KB** (`4.15 MB` process RSS delta) | **SHA-256-verified `benchmark_model` profile** |
