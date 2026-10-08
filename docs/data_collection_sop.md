# Audio Corpus Ingestion & Signal QA Specification

## 1. Acoustic Taxonomy & Split Matrix

| Class ID | Class Name | Source Category (`ESC-50`) | Role in Pipeline | Allocation |
| :---: | :--- | :--- | :--- | :--- |
| 0 | `background_noise` | `vacuum_cleaner`, `washing_machine`, `engine`, `rain` | Hard negatives & +3 dB SNR interferer pool | 160 clips (40 per source) |
| 1 | `coughing` | `coughing` | Target event | 40 clips |
| 2 | `snoring` | `snoring` | Target event | 40 clips |
| 3 | `siren` | `siren` | Target event | 40 clips |
| 4 | `crying_baby` | `crying_baby` | Target event | 40 clips |

- **Training Pool (Folds 1–4)**: Grouped by Freesound `src_file`. `v2_data_flywheel` applies disjoint Butterworth low-pass cutoffs (1,350 Hz, 2,100 Hz) and domestic interferer mixing (+2.0 dB, +5.5 dB SNR).
- **Held-Out Evaluation (Fold 5)**: Zero `src_file` overlap across `clean`, `pocket_occluded` (1,600 Hz 4th-order Butterworth low-pass), and `appliance_noise_3db` (+3.0 dB SNR).

---

## 2. Stage 1: Signal & Vocal-Bleed Gate (`ingest_qa.py`)

| Priority | Gate Code | Metric | Threshold |
| :---: | :--- | :--- | :--- |
| 1 (P0) | `QUARANTINE_POTENTIAL_SPEECH_PII` | Formant (300–3,400 Hz) energy ratio + 50ms syllabic crest in non-vocal background classes | Ratio > 0.62 & Crest > 2.35 |
| 2 (P1) | `QUARANTINE_ADC_PREAMP_CLIPPING` | Normalized peak amplitude `max(|x(t)|)` | >= 0.998 |
| 3 (P1) | `QUARANTINE_EXCESSIVE_DEAD_AIR` | Fraction of 50ms frames exceeding -50 dBFS RMS | < 0.12 |
| 4 (P1) | `QUARANTINE_MIC_DC_OFFSET_BIAS` | Absolute mean waveform drift `|mean(x)|` | > 0.002 |

---

## 3. Stage 2: Consensus & Drift Thresholds (`consensus_drift.py`)

- **OOF Teacher Consensus**: `StratifiedGroupKFold` grouped by `src_file` (Folds 1–4 isolated from Fold 5); target `Kappa >= 0.80`. Disagreements at confidence >= 0.65 route to `SEND_TO_EXPERT_ADJUDICATION` (`sample_weight = 0.35` in `v2`).
- **Spectral PSI Drift**: 10-bin quantile PSI across high-frequency energy (>= 4 kHz) and 25th-percentile noise floor (< 0.10 `STABLE`; 0.10–0.25 `MODERATE_DRIFT_MONITOR`; > 0.25 `CRITICAL_DRIFT_TRIGGER_FLYWHEEL`).
