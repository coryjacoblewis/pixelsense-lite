#!/usr/bin/env python3
"""Test suite for PixelSense-Lite data QA, drift detection, and LiteRT release gate."""

import os
import tempfile
import numpy as np
import pandas as pd
import tensorflow as tf
from scipy.io import wavfile

import consensus_drift
import ingest_qa
import release_gate


def _synthesize_syllabic_speech_burst(sr: int = 16000) -> np.ndarray:
    """Synthesizes a 1.0s syllabic formant signal (850 Hz + 1,750 Hz pulsed at ~2 Hz)."""
    t = np.linspace(0, 1.0, sr, endpoint=False)
    carrier = 0.65 * np.sin(2 * np.pi * 850.0 * t) + 0.35 * np.sin(
        2 * np.pi * 1750.0 * t
    )
    env = np.where(((t >= 0.10) & (t < 0.22)) | ((t >= 0.55) & (t < 0.67)), 1.0, 0.02)
    return (carrier * env).astype(np.float32)


def _audit_temp_wav(
    tmpdir: str, name: str, sr: int, data: np.ndarray, category: str
) -> dict:
    path = os.path.join(tmpdir, name)
    wavfile.write(path, sr, data)
    return ingest_qa.audit_wav_signal(path, category)


def test_signal_qa_and_vocal_bleed_screening() -> None:
    """Verifies WAV signal QA defect detection across clipping, dead air, DC bias, and vocal bleed."""
    sr = 16000
    t = np.linspace(0, 1.0, sr, endpoint=False)
    clean_sine = 0.45 * np.sin(2 * np.pi * 200.0 * t)
    speech_burst = _synthesize_syllabic_speech_burst(sr)

    dc_clipped_speech = np.clip(speech_burst + 0.42, -1.0, 1.0).astype(np.float32)
    dc_clipped_speech[0] = 1.0
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
            "QUARANTINE_ADC_PREAMP_CLIPPING",
        ),
        (
            "stereo_pii.wav",
            np.column_stack([speech_burst, -speech_burst]).astype(np.float32),
            "vacuum_cleaner",
            "QUARANTINE_POTENTIAL_SPEECH_PII",
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
            tmpdir, "dc_speech.wav", sr, dc_clipped_speech, "vacuum_cleaner"
        )
        assert res_dc["qa_status"] == "QUARANTINE_POTENTIAL_SPEECH_PII"
        assert "QUARANTINE_ADC_PREAMP_CLIPPING" in res_dc["all_qa_flags"]
        assert "QUARANTINE_MIC_DC_OFFSET_BIAS" in res_dc["all_qa_flags"]


def test_oof_consensus_group_isolation() -> None:
    """Verifies zero src_file leakage across splits and Fold 5 isolation during OOF teacher training."""
    qa_df = pd.read_csv("reports/01_vendor_data_qa_report.csv")
    pass_df = qa_df[qa_df["qa_status"] == "PASS"].reset_index(drop=True)
    audit_df = pd.read_csv("reports/02_label_consensus_and_kappa_audit.csv")

    train_sources = set(pass_df[pass_df["fold"] != 5]["src_file"])
    eval_sources = set(pass_df[pass_df["fold"] == 5]["src_file"])
    assert train_sources.isdisjoint(eval_sources)

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
        _, df_clean_f5 = consensus_drift.run_annotation_consensus_audit(
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
        f5_mask = folds_arr == 5
        poisoned_mels[f5_mask] = 1.0 - poisoned_mels[f5_mask]
        poisoned_labels[f5_mask] = (poisoned_labels[f5_mask] + 3) % 5
        _, df_poisoned_f5 = consensus_drift.run_annotation_consensus_audit(
            poisoned_mels,
            poisoned_labels,
            fnames,
            cats,
            src_arr,
            folds=folds_arr,
            out_csv=tmp_audit_csv,
        )

    tr_mask = folds_arr != 5
    assert (
        df_clean_f5.loc[tr_mask, "teacher_confidence"].tolist()
        == df_poisoned_f5.loc[tr_mask, "teacher_confidence"].tolist()
        and df_clean_f5.loc[tr_mask, "routing_action"].tolist()
        == df_poisoned_f5.loc[tr_mask, "routing_action"].tolist()
    )

    assert (
        consensus_drift.EVAL_POCKET_CUTOFF_HZ
        not in consensus_drift.TRAIN_FLYWHEEL_CUTOFFS_HZ
    )
    assert (
        consensus_drift.EVAL_NOISE_SNR_DB not in consensus_drift.TRAIN_FLYWHEEL_SNRS_DB
    )

    w_v2 = np.load("data/golden_eval/w_train_v2.npy")
    train_audit = audit_df[pass_df["fold"] != 5].reset_index(drop=True)
    disputed_train_count = int(
        (train_audit["routing_action"] == "SEND_TO_EXPERT_ADJUDICATION").sum()
    )
    downweighted_count = int(
        np.sum(np.isclose(w_v2, consensus_drift.ADJUDICATION_SAMPLE_WEIGHT))
    )
    assert downweighted_count == disputed_train_count * 5


def test_dsp_augmentations_and_psi_drift() -> None:
    """Verifies Butterworth filter frequency response, SNR mixing accuracy, and PSI identity."""
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


def test_tflite_subgraph_and_release_gate() -> None:
    """Verifies subgraph INT8 compliance and release gate pass/block behavior."""
    for ver in ["v1_baseline", "v2_data_flywheel"]:
        int8_path = os.path.join("models", ver, "model_int8.tflite")
        assert 0 < os.path.getsize(int8_path) <= release_gate.MAX_FLASH_KB * 1024
        interp = tf.lite.Interpreter(model_path=int8_path)
        interp.allocate_tensors()
        hw = release_gate.audit_int8_hardware_compatibility(interp)
        assert (
            hw["dsp_delegate_ready"]
            and hw["float_fallback_count"] == 0
            and hw["unsupported_op_count"] == 0
            and hw["dynamic_tensor_count"] == 0
            and hw["int8_compliance_%"] == 100.0
            and hw["subgraph_tensor_kb"] <= release_gate.MAX_SUBGRAPH_TENSOR_KB
        )

    fp16_interp = tf.lite.Interpreter(
        model_path="models/v2_data_flywheel/model_fp16.tflite"
    )
    fp16_interp.allocate_tensors()
    fp16_hw = release_gate.audit_int8_hardware_compatibility(fp16_interp)
    assert (
        not fp16_hw["dsp_delegate_ready"]
        and fp16_hw["float_fallback_count"] > 0
        and fp16_hw["unsupported_op_count"] > 0
    )

    eval_slices = release_gate.load_cached_eval_slices()
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
    test_signal_qa_and_vocal_bleed_screening()
    test_oof_consensus_group_isolation()
    test_dsp_augmentations_and_psi_drift()
    test_tflite_subgraph_and_release_gate()
    print("[Stage 5] All pipeline checks passed.")


if __name__ == "__main__":
    run_verification()
