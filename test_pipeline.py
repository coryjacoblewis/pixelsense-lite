#!/usr/bin/env python3
"""Test suite for PixelSense-Lite signal QA, 3-way split isolation, blind windowing, and 5-seed INT8 release gate."""

import inspect
import os
import tempfile
import numpy as np
import pandas as pd
import tensorflow as tf
from scipy.io import wavfile

import consensus_drift
import ingest_qa
import release_gate
import train_quantize


def _audit_temp_wav(
    tmpdir: str, name: str, sr: int, data: np.ndarray, category: str
) -> dict:
    path = os.path.join(tmpdir, name)
    wavfile.write(path, sr, data)
    return ingest_qa.audit_wav_signal(path, category)


def test_signal_qa_physical_screening() -> None:
    """Verifies category-agnostic WAV signal QA across multi-sample clipping, single-sample peak norm, dead air, and DC offset."""
    sr = 16000
    t = np.linspace(0, 1.0, sr, endpoint=False)
    clean_sine = 0.45 * np.sin(2 * np.pi * 200.0 * t)

    dc_clipped = np.clip(clean_sine + 0.42, -1.0, 1.0).astype(np.float32)
    dc_clipped[:5] = 1.0
    nan_wave = clean_sine.astype(np.float32).copy()
    nan_wave[500] = np.nan

    cases = [
        ("int32.wav", (clean_sine * 2147483647.0).astype(np.int32), "rain", "PASS"),
        (
            "uint8.wav",
            np.round(clean_sine * 127.0 + 128.0).astype(np.uint8),
            "rain",
            "PASS",
        ),
        (
            "short.wav",
            (clean_sine[:20] * 32767.0).astype(np.int16),
            "coughing",
            "QUARANTINE_EXCESSIVE_DEAD_AIR",
        ),
        (
            "stereo_clip.wav",
            np.column_stack(
                [np.clip(clean_sine * 2.5, -1.0, 1.0), np.zeros_like(clean_sine)]
            ).astype(np.float32),
            "rain",
            "QUARANTINE_CLIPPING_SATURATION",
        ),
        ("nan.wav", nan_wave, "rain", "QUARANTINE_CORRUPT_HEADER"),
    ]

    with tempfile.TemporaryDirectory() as tmpdir:
        for fname, pcm, cat, expected_status in cases:
            res = _audit_temp_wav(tmpdir, fname, sr, pcm, cat)
            assert (
                res["qa_status"] == expected_status
            ), f"{fname}: got {res['qa_status']}, expected {expected_status}"

        res_dc = _audit_temp_wav(
            tmpdir, "dc_clipped.wav", sr, dc_clipped, "vacuum_cleaner"
        )
        assert res_dc["qa_status"] == "QUARANTINE_CLIPPING_SATURATION"
        assert res_dc["clipped_samples"] >= 3
        assert res_dc["clipping_severity"] == "MULTI_SAMPLE_SATURATION"
        assert "QUARANTINE_CLIPPING_SATURATION" in res_dc["all_qa_flags"]
        assert "QUARANTINE_DC_OFFSET" in res_dc["all_qa_flags"]

        # Verify single-sample 0 dBFS peak normalization is NOT falsely quarantined as clipping saturation
        single_peak = clean_sine.astype(np.float32).copy()
        single_peak[10] = 1.0
        res_peak = _audit_temp_wav(tmpdir, "peak_norm.wav", sr, single_peak, "rain")
        assert res_peak["qa_status"] == "PASS"
        assert res_peak["clipped_samples"] == 1
        assert res_peak["clipping_severity"] == "SINGLE_SAMPLE_PEAK_NORM"


def test_oof_label_noise_and_group_isolation() -> None:
    """Verifies 3-way src_file isolation (Folds 1-3, Fold 4, Fold 5) and train-only OOF audit."""
    qa_df = pd.read_csv("reports/01_signal_qa_report.csv")
    pass_df = qa_df[qa_df["qa_status"] == "PASS"].reset_index(drop=True)
    audit_df = pd.read_csv("reports/02_oof_label_noise_audit.csv")
    psi_df = pd.read_csv("reports/02_psi_spectral_drift_audit.csv")

    assert len(pass_df) == 233
    assert "QUARANTINE_MIDBAND_TRANSIENT_SPIKE" not in set(qa_df["qa_status"])

    train_sources = set(
        pass_df[pass_df["fold"].isin(consensus_drift.TRAIN_FOLDS)]["src_file"]
    )
    val_sources = set(
        pass_df[pass_df["fold"] == consensus_drift.VAL_FOLD]["src_file"]
    )
    eval_sources = set(
        pass_df[pass_df["fold"] == consensus_drift.EVAL_FOLD]["src_file"]
    )
    assert train_sources.isdisjoint(val_sources)
    assert train_sources.isdisjoint(eval_sources)
    assert val_sources.isdisjoint(eval_sources)

    # Verify OOF audit table strictly contains Folds 1-3 training clips (no fake holdout OOF rows)
    expected_train_count = int(pass_df["fold"].isin(consensus_drift.TRAIN_FOLDS).sum())
    assert len(audit_df) == expected_train_count == 140
    assert (audit_df["split_role"] == "train_pool").all()
    assert set(audit_df["fold"]).issubset(set(consensus_drift.TRAIN_FOLDS))

    # Verify PSI shift audit is strictly computed on Fold 4 validation, never Fold 5
    assert (psi_df["split"] == "fold_4_validation").all()
    assert (psi_df["val_samples"] == int((pass_df["fold"] == 4).sum())).all()

    rng = np.random.default_rng(123)
    synth_mels = rng.uniform(0.0, 1.0, size=(len(pass_df), 64, 64, 1)).astype(
        np.float32
    )
    synth_labels = np.array(
        [
            consensus_drift.CLASS_MAP[
                c if c in ingest_qa.TARGET_CLASSES else "background_noise"
            ]
            for c in pass_df["category"]
        ]
    )
    folds_arr = pass_df["fold"].to_numpy()
    src_arr = pass_df["src_file"].to_numpy()
    fnames = pass_df["filename"].tolist()
    cats = pass_df["category"].tolist()

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_audit_csv = os.path.join(tmpdir, "counterfactual_audit.csv")
        kappa_clean, df_clean = consensus_drift.run_oof_label_noise_audit(
            synth_mels,
            synth_labels,
            fnames,
            cats,
            src_arr,
            folds=folds_arr,
            out_csv=tmp_audit_csv,
        )
        poisoned_mels = synth_mels.copy()
        poisoned_labels = synth_labels.copy()
        holdout_mask = ~np.isin(folds_arr, consensus_drift.TRAIN_FOLDS)
        poisoned_mels[holdout_mask] = 1.0 - poisoned_mels[holdout_mask]
        poisoned_labels[holdout_mask] = (poisoned_labels[holdout_mask] + 3) % 5
        kappa_poisoned, df_poisoned = consensus_drift.run_oof_label_noise_audit(
            poisoned_mels,
            poisoned_labels,
            fnames,
            cats,
            src_arr,
            folds=folds_arr,
            out_csv=tmp_audit_csv,
        )

    assert kappa_clean == kappa_poisoned
    assert (
        df_clean["oof_rf_conf"].tolist() == df_poisoned["oof_rf_conf"].tolist()
        and df_clean["sample_action"].tolist() == df_poisoned["sample_action"].tolist()
    )

    assert (
        consensus_drift.EVAL_POCKET_CUTOFF_HZ
        not in consensus_drift.TRAIN_AUG_CUTOFFS_HZ
    )
    assert (
        consensus_drift.EVAL_NOISE_SNR_DB not in consensus_drift.TRAIN_AUG_SNRS_DB
    )

    w_v2 = np.load("data/golden_eval/w_train_v2.npy")
    disputed_train_count = int(
        (audit_df["sample_action"] == "DOWNWEIGHT_NOISY_LABEL").sum()
    )
    downweighted_count = int(
        np.sum(np.isclose(w_v2, consensus_drift.NOISY_LABEL_SAMPLE_WEIGHT))
    )
    assert downweighted_count == disputed_train_count * 5


def test_dsp_augmentations_and_blind_window_psi() -> None:
    """Verifies blind windowing signature, Butterworth response, SNR mixing accuracy, and orthogonal PSI."""
    sig_params = list(inspect.signature(consensus_drift.wav_to_mel).parameters.keys())
    assert sig_params == ["y"], f"Expected blind wav_to_mel(y), got {sig_params}"

    sr = 16000
    t = np.linspace(0, 2.0, sr * 2, endpoint=False, dtype=np.float32)

    tone_pass = np.sin(2 * np.pi * 300.0 * t).astype(np.float32)
    tone_stop = np.sin(2 * np.pi * 4000.0 * t).astype(np.float32)
    out_pass = consensus_drift.apply_pocket_occlusion(
        tone_pass, sr=sr, cutoff_hz=1600.0
    )
    out_stop = consensus_drift.apply_pocket_occlusion(
        tone_stop, sr=sr, cutoff_hz=1600.0
    )

    steady = slice(int(0.05 * sr), None)
    gain_pass_db = float(
        20.0
        * np.log10(
            np.sqrt(np.mean(out_pass[steady] ** 2))
            / np.sqrt(np.mean(tone_pass[steady] ** 2))
        )
    )
    gain_stop_db = float(
        20.0
        * np.log10(
            np.sqrt(np.mean(out_stop[steady] ** 2))
            / np.sqrt(np.mean(tone_stop[steady] ** 2))
        )
    )
    assert abs(gain_pass_db) < 0.5 and gain_stop_db < -25.0

    rng = np.random.default_rng(77)
    short_noise = rng.normal(0.0, 0.25, size=3700).astype(np.float32)
    for target_snr_db in (2.0, 3.0, 5.5):
        mixed = consensus_drift.mix_real_interferer(
            tone_pass, short_noise, snr_db=target_snr_db
        )
        assert mixed.shape == tone_pass.shape and mixed.dtype == np.float32
        scaled_noise = mixed - tone_pass
        measured_snr_db = float(
            20.0
            * np.log10(
                np.sqrt(np.mean(tone_pass**2)) / np.sqrt(np.mean(scaled_noise**2))
            )
        )
        assert abs(measured_snr_db - target_snr_db) < 0.01

    X_clean = np.load("data/golden_eval/X_clean.npy")
    assert consensus_drift.compute_spectral_psi(X_clean, X_clean) == (
        0.0,
        0.0,
        0.0,
        "STABLE",
    )

    # Verify orthogonal PSI behavior: passive low-pass filtering triggers HF PSI (>0.25)
    # without false-triggering passband dynamic-range noise_floor_psi (<0.10)
    psi_df = pd.read_csv("reports/02_psi_spectral_drift_audit.csv").set_index(
        "validation_slice"
    )
    assert psi_df.loc["clean", "composite_psi"] < 0.10
    assert (
        psi_df.loc["pocket_occluded", "high_freq_band_psi"] > 0.25
        and psi_df.loc["pocket_occluded", "noise_floor_psi"] < 0.10
    )
    assert psi_df.loc["appliance_noise_3db", "noise_floor_psi"] > 0.25


def test_tflite_subgraph_and_multiseed_release_gate() -> None:
    """Verifies subgraph INT8/INT32 accounting, SHA-256 telemetry binding, clustered bootstrap CIs, and gate blocking."""
    for ver in ["v1_baseline", "v2_robust_augmented"]:
        int8_path = os.path.join("models", ver, "model_int8.tflite")
        assert 0 < os.path.getsize(int8_path) <= release_gate.MAX_FLASH_KB * 1024
        interp = tf.lite.Interpreter(model_path=int8_path)
        interp.allocate_tensors()
        hw = release_gate.audit_int8_hardware_compatibility(interp)
        assert (
            hw["int8_subgraph_ready"]
            and hw["non_int_tensors"] == 0
            and hw["int8_tensor_count"] > 0
            and hw["int32_accumulator_count"] > 0
            and hw["unsupported_op_count"] == 0
            and hw["dynamic_tensor_count"] == 0
            and hw["int_tensor_%"] == 100.0
            and hw["subgraph_tensor_kb"] <= release_gate.MAX_SUBGRAPH_TENSOR_KB
            and hw["peak_op_io_kb"] == 80.0
        )

    fp16_interp = tf.lite.Interpreter(
        model_path="models/v2_robust_augmented/model_fp16.tflite"
    )
    fp16_interp.allocate_tensors()
    fp16_hw = release_gate.audit_int8_hardware_compatibility(fp16_interp)
    assert (
        not fp16_hw["int8_subgraph_ready"]
        and fp16_hw["non_int_tensors"] > 0
        and fp16_hw["unsupported_op_count"] > 0
        and fp16_hw["peak_op_io_kb"] == 320.0
    )

    # Verify source-clustered bootstrap CI & paired delta CI on full 50-clip Fold 5
    src_eval = consensus_drift.load_cached_eval_sources()
    eval_slices = release_gate.load_cached_eval_slices()
    y_clean = eval_slices["clean"][1]
    assert len(src_eval) == len(y_clean) == 50
    ci_lo, ci_hi = release_gate.bootstrap_slice_f1_ci(
        y_clean, y_clean, src_files=src_eval, n_boot=200
    )
    assert ci_lo == 100.0 and ci_hi == 100.0
    d_pt, d_lo, d_hi = release_gate.bootstrap_paired_delta_f1_ci(
        y_clean, np.zeros_like(y_clean), y_clean, src_files=src_eval, n_boot=200
    )
    assert d_pt > 70.0 and d_lo <= d_pt <= d_hi

    # Verify ARM64 benchmark_model telemetry parser, SHA-256 verification, and per-slice BG FPR ceiling
    arm64_tel = release_gate.parse_arm64_benchmark_telemetry()
    assert arm64_tel is not None
    assert (
        arm64_tel["model_sha256"]
        == release_gate.compute_file_sha256(
            "models/v2_robust_augmented/model_int8.tflite"
        )
        and arm64_tel["xnnpack_delegate_applied"]
        and arm64_tel["arm64_latency_pass"]
        and 0.0 < arm64_tel["xnnpack_p95_ms"] < arm64_tel["raw_cpu_p95_ms"]
        and arm64_tel["xnnpack_p95_ms"] <= release_gate.MAX_ARM64_P95_LATENCY_MS
        and 0.0
        <= arm64_tel["raw_cpu_allocate_tensors_kb"]
        <= release_gate.MAX_ALLOCATE_TENSORS_KB
        and 0.0
        <= arm64_tel["xnnpack_allocate_tensors_kb"]
        <= release_gate.MAX_ALLOCATE_TENSORS_KB
        and arm64_tel["raw_cpu_rss_delta_mb"] > 0.0
        and arm64_tel["xnnpack_rss_delta_mb"] > 0.0
    )
    assert release_gate.parse_arm64_benchmark_telemetry(op_csv_path="") is None
    # Verify same-byte-size model with different weights (v1_baseline int8, also 23,536 B) is rejected by SHA-256 check
    assert os.path.getsize("models/v1_baseline/model_int8.tflite") == os.path.getsize(
        "models/v2_robust_augmented/model_int8.tflite"
    )
    assert (
        release_gate.parse_arm64_benchmark_telemetry(
            expected_model_path="models/v1_baseline/model_int8.tflite"
        )
        is None
    )
    v2_eval = release_gate.evaluate_tflite_binary(
        "models/v2_robust_augmented/model_int8.tflite", eval_slices, src_files=src_eval
    )
    assert (
        v2_eval["peak_op_io_kb"] == 80.0
        <= release_gate.MAX_PEAK_OP_IO_KB
    )
    assert set(v2_eval["slice_bg_fpr"].keys()) == set(consensus_drift.SLICES)
    assert (
        round(float(np.mean(list(v2_eval["slice_bg_fpr"].values()))), 2)
        == v2_eval["bg_fpr_%"]
    )
    assert (
        v2_eval["max_slice_bg_fpr_%"]
        == round(float(max(v2_eval["slice_bg_fpr"].values())), 2)
        <= release_gate.MAX_BG_FPR_PCT
    )

    assert (
        v2_eval["f1_clean_%"] >= release_gate.SLICE_F1_THRESHOLDS["clean"]
        and v2_eval["f1_pocket_occluded_%"]
        >= release_gate.SLICE_F1_THRESHOLDS["pocket_occluded"]
        and v2_eval["f1_appliance_noise_3db_%"]
        >= release_gate.SLICE_F1_THRESHOLDS["appliance_noise_3db"]
    )

    ablat_df = pd.read_csv("reports/03_training_and_ablation_metrics.csv").set_index(
        "configuration"
    )
    expected_configs = {
        "v1_baseline",
        "step_matched_clean",
        "v2_augmentation_only",
        "v2_aug_oof_weights_only",
        "v2_aug_empirical_prior_only",
        "v2_robust_augmented",
    }
    assert set(ablat_df.index) == expected_configs
    for col in (
        "noisy_label_downweighted",
        "empirical_class_prior",
        "val_max_slice_bg_fpr_%",
        "eval_max_slice_bg_fpr_%",
        "int8_f1_macro_mean_%",
        "int8_eval_max_slice_bg_fpr_%",
        "int8_eval_f1_5seed_mean_%",
        "int8_eval_f1_5seed_std_%",
        "int8_eval_bg_fpr_5seed_mean_%",
        "int8_eval_bg_fpr_5seed_std_%",
    ):
        assert col in ablat_df.columns

    # Verify Seed-42 and 5-seed INT8 metrics satisfy release gates with cross-platform safety margins
    assert (
        v2_eval["f1_appliance_noise_3db_%"] >= 58.0
        and ablat_df.loc["v2_robust_augmented", "int8_f1_appliance_noise_3db_%"] >= 58.0
    )
    assert (
        v2_eval["max_slice_bg_fpr_%"] <= 12.0
        and ablat_df.loc["v2_robust_augmented", "int8_eval_max_slice_bg_fpr_%"] <= 12.0
    )
    assert (
        ablat_df.loc["v2_robust_augmented", "int8_eval_bg_fpr_5seed_mean_%"]
        <= 12.0  # >= 3.0% safety margin below 15.0% gate to prevent cross-OS drift failures
    )
    assert (
        ablat_df.loc["v2_robust_augmented", "int8_eval_bg_fpr_5seed_std_%"]
        <= 7.5  # Guard against high-variance bimodal seed trajectories
    )
    assert (
        ablat_df.loc["v2_robust_augmented", "int8_eval_f1_5seed_mean_%"] >= 70.0
        and ablat_df.loc["v2_robust_augmented", "int8_eval_f1_5seed_std_%"] <= 5.0
    )
    assert (
        ablat_df.loc["v2_aug_oof_weights_only", "int8_eval_bg_fpr_5seed_mean_%"]
        > release_gate.MAX_BG_FPR_PCT
    )

    blocked_caught = False
    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            release_gate.run_release_gate(
                versions=["v1_baseline"],
                eval_slices=eval_slices,
                enforce_target="v1_baseline",
                out_md=os.path.join(tmpdir, "blocked_scorecard.md"),
            )
        except RuntimeError:
            blocked_caught = True
    assert blocked_caught


def run_verification() -> None:
    print("[Stage 5] Running pipeline test suite...")
    test_signal_qa_physical_screening()
    test_oof_label_noise_and_group_isolation()
    test_dsp_augmentations_and_blind_window_psi()
    test_tflite_subgraph_and_multiseed_release_gate()
    print("[Stage 5] All pipeline checks passed.")


if __name__ == "__main__":
    run_verification()
