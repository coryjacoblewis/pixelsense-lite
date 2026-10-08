#!/usr/bin/env python3
"""Stage 5: Automated Pipeline Invariant & Subgraph Verification Suite."""

import importlib
import os
import tempfile
import numpy as np
import pandas as pd
import tensorflow as tf
from scipy.io import wavfile

ingest_mod = importlib.import_module("01_ingest_qa")
consensus_mod = importlib.import_module("02_consensus_drift")
gate_mod = importlib.import_module("04_release_gate")


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
    return ingest_mod.audit_wav_signal(path, category)


def test_privacy_precedence_and_pcm_resilience() -> None:
    """Verifies speech-PII precedence and multi-bit/stereo/corrupt PCM handling."""
    qa_df = pd.read_csv("reports/01_vendor_data_qa_report.csv")

    target_row = qa_df[qa_df["filename"] == "3-152020-C-36.wav"].iloc[0]
    assert target_row["qa_status"] == "QUARANTINE_POTENTIAL_SPEECH_PII"
    assert "QUARANTINE_ADC_PREAMP_CLIPPING" in target_row["all_qa_flags"]
    assert int((qa_df["qa_status"] == "QUARANTINE_POTENTIAL_SPEECH_PII").sum()) == 2

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
        ("uint8.wav", np.round(clean_sine * 127.0 + 128.0).astype(np.uint8), "rain", "PASS"),
        ("short.wav", (clean_sine[:20] * 32767.0).astype(np.int16), "coughing", "QUARANTINE_EXCESSIVE_DEAD_AIR"),
        ("stereo_clip.wav", np.column_stack([np.clip(clean_sine * 2.5, -1.0, 1.0), np.zeros_like(clean_sine)]).astype(np.float32), "rain", "QUARANTINE_ADC_PREAMP_CLIPPING"),
        ("stereo_pii.wav", np.column_stack([speech_burst, -speech_burst]).astype(np.float32), "vacuum_cleaner", "QUARANTINE_POTENTIAL_SPEECH_PII"),
        ("nan.wav", nan_wave, "rain", "QUARANTINE_CORRUPT_HEADER"),
    ]

    with tempfile.TemporaryDirectory() as tmpdir:
        for fname, pcm, cat, expected_status in cases:
            res = _audit_temp_wav(tmpdir, fname, sr, pcm, cat)
            assert res["qa_status"] == expected_status, f"{fname}: got {res['qa_status']}, expected {expected_status}"

        quiet_pcm = (0.0039 * np.sin(2 * np.pi * 200.0 * t) * 2147483647.0).astype(np.int32)
        res_quiet = _audit_temp_wav(tmpdir, "quiet32.wav", sr, quiet_pcm, "rain")
        assert res_quiet["peak_amplitude"] == 0.0039 and res_quiet["qa_status"] != "QUARANTINE_ADC_PREAMP_CLIPPING"

        res_dc = _audit_temp_wav(tmpdir, "dc_speech.wav", sr, dc_clipped_speech, "vacuum_cleaner")
        assert res_dc["qa_status"] == "QUARANTINE_POTENTIAL_SPEECH_PII"
        assert "QUARANTINE_ADC_PREAMP_CLIPPING" in res_dc["all_qa_flags"]
        assert "QUARANTINE_MIC_DC_OFFSET_BIAS" in res_dc["all_qa_flags"]


def test_zero_leakage_and_disjoint_flywheel() -> None:
    """Verifies zero src_file leakage, Fold 5 OOF isolation, and disjoint flywheel parameters."""
    qa_df = pd.read_csv("reports/01_vendor_data_qa_report.csv")
    pass_df = qa_df[qa_df["qa_status"] == "PASS"].reset_index(drop=True)
    audit_df = pd.read_csv("reports/02_label_consensus_and_kappa_audit.csv")

    train_sources = set(pass_df[pass_df["fold"] != 5]["src_file"])
    eval_sources = set(pass_df[pass_df["fold"] == 5]["src_file"])
    assert train_sources.isdisjoint(eval_sources)
    assert "src_file" in audit_df.columns

    rng = np.random.default_rng(123)
    synth_mels = rng.uniform(0.0, 1.0, size=(len(pass_df), 64, 64, 1)).astype(np.float32)
    synth_labels = np.array(
        [
            consensus_mod.CLASS_MAP[
                c if c in consensus_mod.TARGET_CLASSES else "background_noise"
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
        _, df_clean_f5 = consensus_mod.run_annotation_consensus_audit(
            synth_mels, synth_labels, fnames, cats, src_arr, folds=folds_arr, out_csv=tmp_audit_csv
        )
        poisoned_mels = synth_mels.copy()
        poisoned_labels = synth_labels.copy()
        f5_mask = folds_arr == 5
        poisoned_mels[f5_mask] = 1.0 - poisoned_mels[f5_mask]
        poisoned_labels[f5_mask] = (poisoned_labels[f5_mask] + 3) % 5
        _, df_poisoned_f5 = consensus_mod.run_annotation_consensus_audit(
            poisoned_mels, poisoned_labels, fnames, cats, src_arr, folds=folds_arr, out_csv=tmp_audit_csv
        )

    tr_mask = folds_arr != 5
    assert (
        df_clean_f5.loc[tr_mask, "teacher_confidence"].tolist()
        == df_poisoned_f5.loc[tr_mask, "teacher_confidence"].tolist()
        and df_clean_f5.loc[tr_mask, "routing_action"].tolist()
        == df_poisoned_f5.loc[tr_mask, "routing_action"].tolist()
    )

    assert consensus_mod.EVAL_POCKET_CUTOFF_HZ not in consensus_mod.TRAIN_FLYWHEEL_CUTOFFS_HZ
    assert consensus_mod.EVAL_NOISE_SNR_DB not in consensus_mod.TRAIN_FLYWHEEL_SNRS_DB

    w_v2 = np.load("data/golden_eval/w_train_v2.npy")
    train_audit = audit_df[pass_df["fold"] != 5].reset_index(drop=True)
    disputed_train_count = int(
        (train_audit["routing_action"] == "SEND_TO_EXPERT_ADJUDICATION").sum()
    )
    downweighted_count = int(np.sum(np.isclose(w_v2, consensus_mod.ADJUDICATION_SAMPLE_WEIGHT)))
    assert downweighted_count == disputed_train_count * 5


def test_hardware_gate_and_doc_sync() -> None:
    """Verifies subgraph int8 compatibility, release gate blocking, and doc synchronization."""
    psi_df = pd.read_csv("reports/02_psi_spectral_drift_audit.csv")
    audit_df = pd.read_csv("reports/02_label_consensus_and_kappa_audit.csv")
    assert psi_df[psi_df["evaluation_slice"] == "clean"].iloc[0]["drift_status"] == "STABLE"

    for ver in ["v1_baseline", "v2_data_flywheel"]:
        int8_path = os.path.join("models", ver, "model_int8.tflite")
        assert os.path.getsize(int8_path) == 23536
        interp = tf.lite.Interpreter(model_path=int8_path)
        interp.allocate_tensors()
        hw = gate_mod.audit_int8_hardware_compatibility(interp)
        assert (
            hw["dsp_delegate_ready"]
            and hw["float_fallback_count"] == 0
            and hw["unsupported_op_count"] == 0
            and hw["dynamic_tensor_count"] == 0
            and hw["int8_compliance_%"] == 100.0
            and hw["tensor_arena_kb"] <= gate_mod.MAX_TENSOR_ARENA_KB
        )

    fp16_interp = tf.lite.Interpreter(model_path="models/v2_data_flywheel/model_fp16.tflite")
    fp16_interp.allocate_tensors()
    fp16_hw = gate_mod.audit_int8_hardware_compatibility(fp16_interp)
    assert (
        not fp16_hw["dsp_delegate_ready"]
        and fp16_hw["float_fallback_count"] > 0
        and fp16_hw["unsupported_op_count"] > 0
    )

    with open("reports/04_release_gate_scorecard.md", "r", encoding="utf-8") as f:
        saved_scorecard_md = f.read()

    eval_slices = gate_mod.load_cached_eval_slices()
    blocked_caught = False
    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            gate_mod.run_release_gate(
                versions=["v1_baseline"],
                eval_slices=eval_slices,
                enforce_target="v1_baseline",
                out_md=os.path.join(tmpdir, "blocked_scorecard.md"),
            )
        except RuntimeError:
            blocked_caught = True
    assert blocked_caught

    assert (
        "| v2_data_flywheel | int8" in saved_scorecard_md
        and "SHIP (PASS)" in saved_scorecard_md
        and "bg_fpr_%" in saved_scorecard_md
    )

    with open("README.md", "r", encoding="utf-8") as f:
        readme_text = f.read()
    with open("docs/data_collection_sop.md", "r", encoding="utf-8") as f:
        sop_text = f.read()
    with open("docs/model_and_data_cards.md", "r", encoding="utf-8") as f:
        cards_text = f.read()

    for param in ["300", "3,400", "0.62", "2.35", "1,600", "0.25", "0.35"]:
        assert param in sop_text

    for _, row in psi_df.iterrows():
        assert f"{row['composite_psi']:.4f}" in readme_text

    disputed_clips = audit_df[
        audit_df["routing_action"] == "SEND_TO_EXPERT_ADJUDICATION"
    ]["filename"].tolist()
    for clip in disputed_clips:
        assert clip in readme_text

    assert (
        "230 PASS" in cards_text
        and "| `QUARANTINE_POTENTIAL_SPEECH_PII` | **2** |" in readme_text
    )

    if os.path.exists("data/golden_eval/X_pocket_occluded_blind.npy") and os.path.exists(
        "data/golden_eval/X_appliance_noise_3db_blind.npy"
    ):
        blind_slices = {
            "clean": eval_slices["clean"],
            "pocket_occluded": (
                np.load("data/golden_eval/X_pocket_occluded_blind.npy"),
                eval_slices["pocket_occluded"][1],
            ),
            "appliance_noise_3db": (
                np.load("data/golden_eval/X_appliance_noise_3db_blind.npy"),
                eval_slices["appliance_noise_3db"][1],
            ),
        }
        blind_res = gate_mod.evaluate_tflite_binary(
            "models/v2_data_flywheel/model_int8.tflite", blind_slices
        )
        assert (
            f"{blind_res['f1_pocket_occluded_%']:.2f}%" in readme_text
            and f"{blind_res['f1_appliance_noise_3db_%']:.2f}%" in readme_text
        )

    with open("reports/05_arm64_op_profile_int8.csv", "r", encoding="utf-8") as f:
        arm_csv_text = f.read()
    with open("reports/05_arm64_hardware_telemetry_int8.txt", "r", encoding="utf-8") as f:
        arm_txt_text = f.read()
    assert (
        "9 nodes observed" in arm_csv_text
        and "XNNPACK delegate created" in arm_txt_text
        and "Convolution (NHWC, QC8) IGEMM" in arm_txt_text
    )


def test_dsp_physics_and_drift_math() -> None:
    """Verifies Butterworth filter response, exact SNR mixing, PSI identity & duty-cycle energy."""
    sr = 16000
    t = np.linspace(0, 2.0, sr * 2, endpoint=False, dtype=np.float32)

    tone_pass = np.sin(2 * np.pi * 300.0 * t).astype(np.float32)
    tone_stop = np.sin(2 * np.pi * 4000.0 * t).astype(np.float32)
    out_pass = consensus_mod.apply_pocket_occlusion(tone_pass, sr=sr, cutoff_hz=1600.0)
    out_stop = consensus_mod.apply_pocket_occlusion(tone_stop, sr=sr, cutoff_hz=1600.0)

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
        mixed = consensus_mod.mix_real_interferer(
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
    assert consensus_mod.compute_spectral_psi(X_clean, X_clean) == (
        0.0,
        0.0,
        0.0,
        "STABLE",
    )

    batt_dsp = gate_mod.compute_daily_battery_pct(0.11, dsp_delegate_ready=True)
    batt_cpu = gate_mod.compute_daily_battery_pct(0.32, dsp_delegate_ready=False)
    assert batt_dsp <= gate_mod.MAX_DAILY_BATTERY_PCT < batt_cpu


def run_verification() -> None:
    print("[Stage 5] Running invariant & doc-sync verification suite...")
    test_privacy_precedence_and_pcm_resilience()
    test_zero_leakage_and_disjoint_flywheel()
    test_hardware_gate_and_doc_sync()
    test_dsp_physics_and_drift_math()
    print("[Stage 5] All invariant checks passed.")


if __name__ == "__main__":
    run_verification()
