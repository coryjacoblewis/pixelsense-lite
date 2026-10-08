#!/usr/bin/env python3
"""Stage 1: Audio Corpus Ingestion, Physical Signal QA & Speech-Band Screening."""

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

MAX_ARCHIVE_UNCOMPRESSED_BYTES = 1_500_000_000


def download_real_corpus() -> None:
    """Downloads and extracts the ESC-50 environmental audio dataset if not cached."""
    os.makedirs(DATA_DIR, exist_ok=True)
    if not os.path.exists(ESC_DIR):
        print("[Stage 1] Downloading real ESC-50 Freesound corpus (~600MB)...")
        urllib.request.urlretrieve(ESC_URL, ESC_ZIP)
        with zipfile.ZipFile(ESC_ZIP, "r") as zf:
            abs_target = os.path.abspath(DATA_DIR)
            total_uncompressed = 0
            for info in zf.infolist():
                member_path = os.path.abspath(os.path.join(DATA_DIR, info.filename))
                if (
                    not member_path.startswith(abs_target + os.sep)
                    and member_path != abs_target
                ):
                    raise RuntimeError(
                        f"Unsafe zip path traversal blocked: {info.filename}"
                    )
                unix_mode = (info.external_attr >> 16) & 0o170000
                if unix_mode == 0o120000:
                    raise RuntimeError(
                        f"Unsafe symlink entry in zip archive blocked: {info.filename}"
                    )
                total_uncompressed += int(info.file_size)
                if total_uncompressed > MAX_ARCHIVE_UNCOMPRESSED_BYTES:
                    raise RuntimeError(
                        f"Zip archive exceeds safe uncompressed byte limit ({MAX_ARCHIVE_UNCOMPRESSED_BYTES} B)"
                    )
            zf.extractall(DATA_DIR)
        print("[Stage 1] Download and extraction complete.")
    else:
        print("[Stage 1] Cached ESC-50 corpus found at data/ESC-50-master.")


def _read_wav_bits_per_sample(filepath: str) -> int | None:
    """Reads the wBitsPerSample field from a RIFF WAV fmt chunk if present."""
    try:
        with open(filepath, "rb") as f:
            header = f.read(128)
        if len(header) >= 36 and header[:4] == b"RIFF" and header[8:12] == b"WAVE":
            idx = header.find(b"fmt ")
            if idx != -1 and idx + 24 <= len(header):
                return int.from_bytes(header[idx + 22 : idx + 24], "little")
    except Exception:
        pass
    return None


def _scale_pcm(raw: np.ndarray, bits_per_sample: int | None = None) -> np.ndarray:
    """Scales int16, int24/int32, uint8, or float PCM buffers to [-1.0, 1.0] float32."""
    if raw.dtype == np.int16:
        return raw.astype(np.float32) / 32768.0
    if raw.dtype == np.int32:
        max_val = float(np.max(np.abs(raw))) if raw.size > 0 else 0.0
        lsb_all_zero = bool(raw.size > 0 and np.all((raw & 0xFF) == 0))
        if bits_per_sample == 32:
            denom = 2147483648.0
        elif bits_per_sample == 24:
            denom = 2147483648.0 if lsb_all_zero else 8388608.0
        else:
            denom = (
                8388608.0
                if (not lsb_all_zero and 0.0 < max_val <= 8388608.0)
                else 2147483648.0
            )
        return raw.astype(np.float32) / denom
    if raw.dtype == np.uint8:
        return (raw.astype(np.float32) - 128.0) / 128.0
    return raw.astype(np.float32)


def normalize_pcm_waveform(
    raw: np.ndarray, bits_per_sample: int | None = None
) -> np.ndarray:
    """Normalizes PCM buffers to [-1.0, 1.0] mono float32."""
    x = _scale_pcm(raw, bits_per_sample)
    if x.ndim > 1:
        mono_mean = np.mean(x, axis=1)
        ch_energies = np.sum(x**2, axis=0)
        max_ch_idx = int(np.argmax(ch_energies))
        if float(np.sum(mono_mean**2)) < 0.25 * float(ch_energies[max_ch_idx]):
            x = x[:, max_ch_idx]
        else:
            x = mono_mean
    return x


def _frame_rms(sig: np.ndarray, sr: int) -> np.ndarray:
    """Computes RMS energy across non-overlapping 50ms frames."""
    frame_len = max(1, int(sr * 0.05))
    n_frames = len(sig) // frame_len
    if n_frames == 0:
        return np.empty(0, dtype=np.float32)
    frames = sig[: n_frames * frame_len].reshape(n_frames, frame_len)
    return np.sqrt(np.mean(frames**2, axis=1) + 1e-12)


def compute_speech_formant_metrics(x: np.ndarray, sr: int) -> tuple[float, float]:
    """Computes 300-3,400 Hz speech-band energy ratio and 50ms syllabic envelope crest."""
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
    """Audits a WAV file for hardware anomalies and speech-band PII (with PII precedence)."""
    try:
        sr, raw = wavfile.read(filepath)
    except Exception:
        return _corrupt_header_result(0)

    if sr <= 0 or raw is None or raw.size == 0 or not np.all(np.isfinite(raw)):
        return _corrupt_header_result(sr)

    bits_per_sample = _read_wav_bits_per_sample(filepath)
    scaled = _scale_pcm(raw, bits_per_sample=bits_per_sample)
    x = normalize_pcm_waveform(scaled)

    peak_amp = float(np.max(np.abs(scaled)))
    dc_offset = float(np.max(np.abs(np.mean(scaled, axis=0))))

    frame_rms = _frame_rms(x, sr)
    active_ratio = (
        float(np.mean(20.0 * np.log10(frame_rms) > -50.0))
        if frame_rms.size > 0
        else 0.0
    )

    speech_band_ratio, envelope_crest = compute_speech_formant_metrics(x, sr)
    is_speech_pii_risk = (
        category in INTERFERER_CLASSES
        and speech_band_ratio > 0.62
        and envelope_crest > 2.35
    )

    flags = []
    if is_speech_pii_risk:
        flags.append("QUARANTINE_POTENTIAL_SPEECH_PII")
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
    """Runs the Stage 1 download and signal/PII audit across all 320 recordings."""
    download_real_corpus()
    print("[Stage 1] Running audio ingestion QA and speech-band screening...")

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
        f"[Stage 1] Audited {len(qa_df)} clips | PASS: {pass_count} | QUARANTINED: {quarantine_count} ({quarantine_count / len(qa_df) * 100:.1f}%) -> {out_csv}"
    )
    return qa_df


if __name__ == "__main__":
    run_ingestion_qa()

