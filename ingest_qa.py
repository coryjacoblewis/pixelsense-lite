#!/usr/bin/env python3
"""Stage 1: Audio corpus ingestion, signal QA, and background vocal-bleed screening."""

import os
import urllib.request
import zipfile
import numpy as np
import pandas as pd
from scipy.io import wavfile
from scipy.signal import butter, lfilter

DATA_DIR = "data"
ESC_ZIP = os.path.join(DATA_DIR, "ESC-50-master.zip")
ESC_DIR = os.path.join(DATA_DIR, "ESC-50-master")
ESC_URL = "https://github.com/karolpiczak/ESC-50/archive/master.zip"

TARGET_CLASSES = ["coughing", "snoring", "siren", "crying_baby"]
INTERFERER_CLASSES = ["vacuum_cleaner", "washing_machine", "engine", "rain"]
ALL_MODEL_CLASSES = ["background_noise"] + TARGET_CLASSES
CLASS_MAP = {name: idx for idx, name in enumerate(ALL_MODEL_CLASSES)}


def download_real_corpus() -> None:
    """Downloads and extracts the ESC-50 dataset if not already cached locally."""
    os.makedirs(DATA_DIR, exist_ok=True)
    if not os.path.exists(ESC_DIR):
        print("[Stage 1] Downloading ESC-50 corpus (~600MB)...")
        urllib.request.urlretrieve(ESC_URL, ESC_ZIP)
        with zipfile.ZipFile(ESC_ZIP, "r") as zf:
            zf.extractall(DATA_DIR)
        print("[Stage 1] Download and extraction complete.")
    else:
        print("[Stage 1] Using cached ESC-50 corpus at data/ESC-50-master.")


def scale_pcm(raw: np.ndarray) -> np.ndarray:
    """Scales integer or float PCM buffers to [-1.0, 1.0] float32."""
    if raw.dtype == np.int16:
        return raw.astype(np.float32) / 32768.0
    if raw.dtype == np.int32:
        return raw.astype(np.float32) / 2147483648.0
    if raw.dtype == np.uint8:
        return (raw.astype(np.float32) - 128.0) / 128.0
    return raw.astype(np.float32)


def to_mono(scaled: np.ndarray) -> np.ndarray:
    """Downmixes multi-channel audio to mono, falling back to dominant channel if phase-cancelled."""
    if scaled.ndim <= 1:
        return scaled
    mono_mean = np.mean(scaled, axis=1)
    ch_energies = np.sum(scaled**2, axis=0)
    max_ch_idx = int(np.argmax(ch_energies))
    if float(np.sum(mono_mean**2)) < 0.25 * float(ch_energies[max_ch_idx]):
        return scaled[:, max_ch_idx]
    return mono_mean


def _frame_rms(sig: np.ndarray, sr: int) -> np.ndarray:
    """Computes RMS energy across non-overlapping 50ms frames."""
    frame_len = max(1, int(sr * 0.05))
    n_frames = len(sig) // frame_len
    if n_frames == 0:
        return np.empty(0, dtype=np.float32)
    frames = sig[: n_frames * frame_len].reshape(n_frames, frame_len)
    return np.sqrt(np.mean(frames**2, axis=1) + 1e-12)


def compute_speech_formant_metrics(x: np.ndarray, sr: int) -> tuple[float, float]:
    """Computes 300-3,400 Hz band energy ratio and 50ms envelope crest factor."""
    x_ac = x - float(np.mean(x))
    nyq = max(0.5 * sr, 400.0)
    low = min(max(300.0 / nyq, 0.01), 0.90)
    high = max(min(3400.0 / nyq, 0.95), low + 0.02)
    b, a = butter(4, [low, high], btype="band")
    x_speech = lfilter(b, a, x_ac)

    speech_band_ratio = float(np.sum(x_speech**2)) / float(np.sum(x_ac**2) + 1e-12)
    frame_rms = _frame_rms(x_speech, sr)
    envelope_crest = (
        float(np.max(frame_rms) / (np.mean(frame_rms) + 1e-12))
        if frame_rms.size > 0
        else 1.0
    )
    return speech_band_ratio, envelope_crest


def _corrupt_header_result(sr: int = 0) -> dict:
    return {
        "qa_status": "QUARANTINE_CORRUPT_HEADER",
        "all_qa_flags": "QUARANTINE_CORRUPT_HEADER",
        "sample_rate": max(0, int(sr) if sr else 0),
        "peak_amplitude": 0.0,
        "dc_offset": 0.0,
        "active_frame_ratio": 0.0,
        "speech_band_ratio": 0.0,
        "envelope_crest": 0.0,
    }


def audit_wav_signal(filepath: str, category: str) -> dict:
    """Audits a WAV file for clipping, dead air, DC bias, and vocal bleed in background classes."""
    try:
        sr, raw = wavfile.read(filepath)
    except Exception:
        return _corrupt_header_result(0)

    if sr <= 0 or raw is None or raw.size == 0 or not np.all(np.isfinite(raw)):
        return _corrupt_header_result(sr)

    scaled = scale_pcm(raw)
    x = to_mono(scaled)

    peak_amp = float(np.max(np.abs(scaled)))
    dc_offset = float(np.max(np.abs(np.mean(scaled, axis=0))))

    frame_rms = _frame_rms(x, sr)
    active_ratio = (
        float(np.mean(20.0 * np.log10(frame_rms) > -50.0))
        if frame_rms.size > 0
        else 0.0
    )

    speech_band_ratio, envelope_crest = compute_speech_formant_metrics(x, sr)
    # Screen background/interferer recordings for transient mid-band or vocal bleed
    has_vocal_bleed = (
        category in INTERFERER_CLASSES
        and speech_band_ratio > 0.62
        and envelope_crest > 2.35
    )

    flags = []
    if has_vocal_bleed:
        flags.append("QUARANTINE_FOREGROUND_VOCAL_BLEED")
    if peak_amp >= 0.998:
        flags.append("QUARANTINE_ADC_PREAMP_CLIPPING")
    if active_ratio < 0.12:
        flags.append("QUARANTINE_EXCESSIVE_DEAD_AIR")
    if dc_offset > 0.002:
        flags.append("QUARANTINE_MIC_DC_OFFSET_BIAS")

    return {
        "qa_status": flags[0] if flags else "PASS",
        "all_qa_flags": "|".join(flags) if flags else "PASS",
        "sample_rate": int(sr),
        "peak_amplitude": round(peak_amp, 4),
        "dc_offset": round(dc_offset, 5),
        "active_frame_ratio": round(active_ratio, 3),
        "speech_band_ratio": round(speech_band_ratio, 3),
        "envelope_crest": round(envelope_crest, 3),
    }


def run_ingestion_qa() -> pd.DataFrame:
    """Runs the Stage 1 download and signal QA audit across all 320 recordings."""
    download_real_corpus()
    print("[Stage 1] Running audio ingestion QA...")

    meta_path = os.path.join(ESC_DIR, "meta", "esc50.csv")
    meta = pd.read_csv(meta_path)
    subset = meta[meta["category"].isin(TARGET_CLASSES + INTERFERER_CLASSES)].copy()

    records = []
    for _, row in subset.iterrows():
        wav_path = os.path.join(ESC_DIR, "audio", row["filename"])
        metrics = audit_wav_signal(wav_path, row["category"])
        records.append({**row.to_dict(), **metrics, "filepath": wav_path})

    qa_df = pd.DataFrame(records)
    os.makedirs("reports", exist_ok=True)
    out_csv = "reports/01_vendor_data_qa_report.csv"
    qa_df.to_csv(out_csv, index=False)

    pass_count = int((qa_df["qa_status"] == "PASS").sum())
    quarantine_count = len(qa_df) - pass_count
    print(
        f"[Stage 1] Audited {len(qa_df)} clips | PASS: {pass_count} | "
        f"QUARANTINED: {quarantine_count} ({quarantine_count / len(qa_df) * 100:.1f}%) -> {out_csv}"
    )
    return qa_df


if __name__ == "__main__":
    run_ingestion_qa()
