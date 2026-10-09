# Audio Corpus & Signal QA SOP

## 1. Acoustic Taxonomy & 3-Way `src_file` Split

| Class ID | Class Name | `ESC-50` Source | Role | Raw Clips |
| :---: | :--- | :--- | :--- | :---: |
| 0 | `background_noise` | `vacuum_cleaner`, `washing_machine`, `engine`, `rain` | Hard negatives & SNR interferer pool | 160 (40/source) |
| 1 | `coughing` | `coughing` | Target event | 40 |
| 2 | `snoring` | `snoring` | Target event | 40 |
| 3 | `siren` | `siren` | Target event | 40 |
| 4 | `crying_baby` | `crying_baby` | Target event | 40 |

- **Folds 1–3 (Train, `N = 140`)**: When Fold 4 synthetic stress-slice PSI shift exceeds `0.25` (`augmentation_active=True`), `v2_robust_augmented` adds 5x disjoint augmentation (1,350/2,100 Hz LP; +2.0/+5.5 dB SNR) with blind 2.0s peak-energy windowing on each waveform and trains under the natural empirical class distribution (`w = 0.35` on OOF-disputed clips).
- **Fold 4 (Val & Shift Audit, `N = 43`/slice)**: Cross-fold spectral stability check (`clean`) and synthetic stress-slice PSI verification (`pocket_occluded`, `appliance_noise_3db`).
- **Fold 5 (Locked Test, `N = 50`/slice, 42 `src_file` sources, 25 BG clips)**: Evaluated across `clean`, `pocket_occluded` (1,600 Hz LP), and `appliance_noise_3db` (+3.0 dB SNR) using blind peak-energy window selection.

---

## 2. Stage 1: Physical Signal QA Policy (`ingest_qa.py`)

| Priority | Gate Code | Metric | Threshold |
| :---: | :--- | :--- | :--- |
| 1 (P0) | `QUARANTINE_CORRUPT_HEADER` | Readable PCM header, `sr > 0`, non-empty, all samples finite | Any violation |
| 2 (P1) | `QUARANTINE_CLIPPING_SATURATION` | Peak amplitude `max(abs(x(t)))` & multi-sample saturation (`MULTI_SAMPLE_SATURATION`) | Peak >= 0.998 & `clipped_samples` > 2 |
| 3 (P1) | `QUARANTINE_EXCESSIVE_DEAD_AIR` | Active 50ms frames > -50 dBFS RMS | < 0.12 |
| 4 (P1) | `QUARANTINE_DC_OFFSET` | Mean waveform drift `abs(mean(x))` | > 0.002 |

---

## 3. Stage 2: OOF Classifier-to-Label Audit & Spectral PSI (`consensus_drift.py`)

| Check | Scope | Definition | Action / Threshold |
| :--- | :--- | :--- | :--- |
| **OOF Classifier-to-Label Audit** | Folds 1–3 (`N = 140`) | 3-fold `StratifiedGroupKFold` (`src_file`); κ = 0.822 (88.6% OOF RF agreement vs. ESC-50 label; automated label-noise probe, no human relabeling) | Disagreement w/ `oof_rf_conf` >= 0.65 -> `DOWNWEIGHT_NOISY_LABEL` (`w = 0.35`) |
| **`high_freq_band_psi`** | Fold 4 vs. Folds 1–3 | 10-bin quantile PSI on mean log-Mel energy in bins `32:64` (`> 2.0 kHz`) across natural Fold 4 and synthetic DSP slices | `< 0.10` `STABLE`, `0.10–0.25` `MODERATE_SHIFT`, `> 0.25` `HIGH_SHIFT_AUGMENTATION_ACTIVE` |
| **`noise_floor_psi`** | Fold 4 vs. Folds 1–3 | 10-bin quantile PSI on frame `P95 - P25` dynamic range in bins `0:24` (`< 1.5 kHz`) | Same thresholds; invariant to per-window `ref=np.max` offsets |
