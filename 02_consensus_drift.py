#!/usr/bin/env python3
"""Stage 2: Annotation Consensus Audit (Cohen's Kappa), Golden Slices & PSI Drift."""

import importlib
import os
import librosa
import numpy as np
import pandas as pd
from scipy.signal import butter, lfilter
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import cohen_kappa_score
from sklearn.model_selection import StratifiedGroupKFold

_ingest = importlib.import_module("01_ingest_qa")
TARGET_SR = 16000
TARGET_CLASSES = _ingest.TARGET_CLASSES
INTERFERER_CLASSES = _ingest.INTERFERER_CLASSES
ALL_MODEL_CLASSES = _ingest.ALL_MODEL_CLASSES
CLASS_MAP = _ingest.CLASS_MAP

EVAL_POCKET_CUTOFF_HZ = 1600.0
EVAL_NOISE_SNR_DB = 3.0
TRAIN_FLYWHEEL_CUTOFFS_HZ = (1350.0, 2100.0)
TRAIN_FLYWHEEL_SNRS_DB = (2.0, 5.5)
ADJUDICATION_SAMPLE_WEIGHT = 0.35


def apply_pocket_occlusion(
    y: np.ndarray, sr: int = TARGET_SR, cutoff_hz: float = EVAL_POCKET_CUTOFF_HZ
) -> np.ndarray:
    """Applies a 4th-order Butterworth low-pass filter modeling fabric/pocket attenuation."""
    b, a = butter(4, cutoff_hz / (0.5 * sr), btype="low")
    return lfilter(b, a, y).astype(np.float32)


def mix_real_interferer(
    clean: np.ndarray, interferer: np.ndarray, snr_db: float = EVAL_NOISE_SNR_DB
) -> np.ndarray:
    """Mixes target audio with interferer audio at the specified SNR (dB)."""
    if len(interferer) < len(clean):
        reps = int(np.ceil(len(clean) / len(interferer)))
        interferer = np.tile(interferer, reps)
    interferer = interferer[: len(clean)]

    clean_rms = float(np.sqrt(np.mean(clean**2) + 1e-9))
    int_rms = float(np.sqrt(interferer.dot(interferer) / len(interferer) + 1e-9))
    scale = (clean_rms / (10.0 ** (snr_db / 20.0))) / int_rms
    return (clean + interferer * scale).astype(np.float32)


def wav_to_mel(y: np.ndarray, center_start: int | None = None) -> tuple[np.ndarray, int]:
    """Extracts a 2.0s window (16 kHz) and computes a normalized (64, 64, 1) Log-Mel spectrogram."""
    win = TARGET_SR * 2
    if len(y) > win:
        if center_start is None:
            step = TARGET_SR // 4
            energies = [
                float(np.sum(y[i : i + win] ** 2))
                for i in range(0, len(y) - win + 1, step)
            ]
            center_start = int(np.argmax(energies)) * step
        y_win = y[center_start : center_start + win]
    else:
        center_start = 0
        y_win = np.pad(y, (0, max(0, win - len(y))))

    mel = librosa.feature.melspectrogram(
        y=y_win, sr=TARGET_SR, n_fft=512, hop_length=500, n_mels=64
    )
    log_mel = librosa.power_to_db(mel, ref=np.max)
    if log_mel.shape[1] < 64:
        log_mel = np.pad(log_mel, ((0, 0), (0, 64 - log_mel.shape[1])), mode="edge")
    else:
        log_mel = log_mel[:, :64]
    norm_mel = np.clip((log_mel + 80.0) / 80.0, 0.0, 1.0)
    return norm_mel.astype(np.float32)[..., np.newaxis], center_start


def _build_teacher(seed: int = 99) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=250, max_depth=12, class_weight="balanced", random_state=seed
    )


def run_annotation_consensus_audit(
    X_mel: np.ndarray,
    y_human: np.ndarray,
    filenames: list[str],
    categories: list[str],
    src_files: np.ndarray,
    folds: np.ndarray | None = None,
    out_csv: str = "reports/02_label_consensus_and_kappa_audit.csv",
) -> tuple[float, pd.DataFrame]:
    """Audits crowd labels against a holdout-isolated (`src_file`-grouped) OOF spectral teacher."""
    mel_2d = X_mel.squeeze(-1)
    X_flat = np.hstack(
        [
            np.mean(mel_2d, axis=2),
            np.std(mel_2d, axis=2),
            np.max(mel_2d, axis=2),
            np.percentile(mel_2d, 75, axis=2),
            np.percentile(mel_2d, 25, axis=2),
            np.mean(np.abs(np.diff(mel_2d, axis=2)), axis=2),
        ]
    )

    oof_preds = np.zeros_like(y_human)
    oof_conf = np.zeros(len(y_human), dtype=np.float32)

    train_mask = folds != 5 if folds is not None else np.ones(len(y_human), dtype=bool)
    eval_mask = ~train_mask
    assert set(src_files[train_mask]).isdisjoint(set(src_files[eval_mask]))

    X_tr_pool = X_flat[train_mask]
    y_tr_pool = y_human[train_mask]
    src_tr_pool = src_files[train_mask]
    tr_indices = np.where(train_mask)[0]

    sgkf = StratifiedGroupKFold(n_splits=4)
    for tr_idx, va_idx in sgkf.split(X_tr_pool, y_tr_pool, groups=src_tr_pool):
        assert set(src_tr_pool[tr_idx]).isdisjoint(set(src_tr_pool[va_idx]))
        clf = _build_teacher()
        clf.fit(X_tr_pool[tr_idx], y_tr_pool[tr_idx])
        probs = clf.predict_proba(X_tr_pool[va_idx])
        oof_preds[tr_indices[va_idx]] = np.argmax(probs, axis=1)
        oof_conf[tr_indices[va_idx]] = np.max(probs, axis=1)

    if np.any(eval_mask):
        clf_eval = _build_teacher()
        clf_eval.fit(X_tr_pool, y_tr_pool)
        probs_eval = clf_eval.predict_proba(X_flat[eval_mask])
        oof_preds[eval_mask] = np.argmax(probs_eval, axis=1)
        oof_conf[eval_mask] = np.max(probs_eval, axis=1)

    kappa = float(cohen_kappa_score(y_human, oof_preds))

    audit_rows = []
    for i in range(len(y_human)):
        disagreement = (y_human[i] != oof_preds[i]) and (oof_conf[i] >= 0.65)
        audit_rows.append(
            {
                "filename": filenames[i],
                "src_file": int(src_files[i]),
                "source_category": categories[i],
                "human_vendor_label": ALL_MODEL_CLASSES[y_human[i]],
                "auto_teacher_label": ALL_MODEL_CLASSES[oof_preds[i]],
                "teacher_confidence": round(float(oof_conf[i]), 3),
                "label_agreement": bool(y_human[i] == oof_preds[i]),
                "routing_action": (
                    "SEND_TO_EXPERT_ADJUDICATION" if disagreement else "ACCEPT_LABEL"
                ),
            }
        )

    audit_df = pd.DataFrame(audit_rows)
    out_dir = os.path.dirname(out_csv)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    audit_df.to_csv(out_csv, index=False)

    adjudication_count = int(
        (audit_df["routing_action"] == "SEND_TO_EXPERT_ADJUDICATION").sum()
    )
    agreement_pct = float(audit_df["label_agreement"].mean() * 100.0)
    print(
        f"[Stage 2A] OOF Agreement: {agreement_pct:.1f}% | Kappa: {kappa:.3f} | Adjudication: {adjudication_count} -> {out_csv}"
    )
    return kappa, audit_df


def compute_spectral_psi(
    X_ref: np.ndarray, X_target: np.ndarray, bins: int = 10
) -> tuple[float, float, float, str]:
    """Computes Population Stability Index (PSI) across high-frequency band and noise floor."""
    ref_2d = X_ref.squeeze(-1)
    tgt_2d = X_target.squeeze(-1)

    ref_hf = np.mean(ref_2d[:, 32:, :], axis=2).flatten()
    tgt_hf = np.mean(tgt_2d[:, 32:, :], axis=2).flatten()

    ref_nf = np.percentile(ref_2d, 25, axis=2).flatten()
    tgt_nf = np.percentile(tgt_2d, 25, axis=2).flatten()

    def _psi_1d(r: np.ndarray, t: np.ndarray) -> float:
        quantiles = np.linspace(0, 1, bins + 1)
        edges = np.unique(np.quantile(r, quantiles))
        if len(edges) < 3:
            edges = np.linspace(
                float(np.min(r)) - 1e-5, float(np.max(r)) + 1e-5, bins + 1
            )
        edges[0], edges[-1] = -np.inf, np.inf
        r_pct = np.clip(np.histogram(r, bins=edges)[0] / float(len(r)), 1e-6, 1.0)
        t_pct = np.clip(np.histogram(t, bins=edges)[0] / float(len(t)), 1e-6, 1.0)
        return float(np.sum((t_pct - r_pct) * np.log(t_pct / r_pct)))

    hf_psi = _psi_1d(ref_hf, tgt_hf)
    nf_psi = _psi_1d(ref_nf, tgt_nf)
    composite_psi = max(hf_psi, nf_psi)

    if composite_psi > 0.25:
        status = "CRITICAL_DRIFT_TRIGGER_FLYWHEEL"
    elif composite_psi >= 0.10:
        status = "MODERATE_DRIFT_MONITOR"
    else:
        status = "STABLE"

    return round(hf_psi, 4), round(nf_psi, 4), round(composite_psi, 4), status


def run_consensus_and_drift(
    qa_csv_path: str = "reports/01_vendor_data_qa_report.csv",
) -> dict:
    """Runs OOF consensus audit, builds train/eval slices, and computes spectral PSI drift."""
    qa_df = pd.read_csv(qa_csv_path)
    clean_df = qa_df[qa_df["qa_status"] == "PASS"].copy().reset_index(drop=True)

    train_src_files = set(clean_df[clean_df["fold"] != 5]["src_file"])
    eval_src_files = set(clean_df[clean_df["fold"] == 5]["src_file"])
    assert train_src_files.isdisjoint(eval_src_files), "CRITICAL: src_file leakage across folds!"

    waveforms, all_mels, all_labels = [], [], []
    train_noises, eval_noises = [], []

    for _, row in clean_df.iterrows():
        y, _ = librosa.load(row["filepath"], sr=TARGET_SR)
        fold = int(row["fold"])
        if row["category"] in INTERFERER_CLASSES:
            (eval_noises if fold == 5 else train_noises).append(y)
            cls_name = "background_noise"
        else:
            cls_name = row["category"]
        label = CLASS_MAP[cls_name]
        mel_clean, win_start = wav_to_mel(y)

        waveforms.append((y, mel_clean, win_start, label, fold))
        all_mels.append(mel_clean)
        all_labels.append(label)

    kappa, audit_df = run_annotation_consensus_audit(
        np.array(all_mels),
        np.array(all_labels),
        clean_df["filename"].tolist(),
        clean_df["category"].tolist(),
        clean_df["src_file"].to_numpy(dtype=int),
        folds=clean_df["fold"].to_numpy(dtype=int),
    )

    X_train_v1, y_train_v1, w_train_v1 = [], [], []
    X_train_v2, y_train_v2, w_train_v2 = [], [], []
    eval_slices = {
        "clean": ([], []),
        "pocket_occluded": ([], []),
        "appliance_noise_3db": ([], []),
    }
    X_pocket_blind, X_noise_blind = [], []

    for idx, (y, mel_clean, win_start, label, fold) in enumerate(waveforms):
        is_disputed = (
            audit_df.loc[idx, "routing_action"] == "SEND_TO_EXPERT_ADJUDICATION"
        )
        flywheel_weight = ADJUDICATION_SAMPLE_WEIGHT if is_disputed else 1.0

        if fold == 5:
            noise_clip = eval_noises[idx % len(eval_noises)]
            y_pocket = apply_pocket_occlusion(y, cutoff_hz=EVAL_POCKET_CUTOFF_HZ)
            y_noisy = mix_real_interferer(y, noise_clip, snr_db=EVAL_NOISE_SNR_DB)

            for s_name, mel_arr in [
                ("clean", mel_clean),
                ("pocket_occluded", wav_to_mel(y_pocket, center_start=win_start)[0]),
                ("appliance_noise_3db", wav_to_mel(y_noisy, center_start=win_start)[0]),
            ]:
                eval_slices[s_name][0].append(mel_arr)
                eval_slices[s_name][1].append(label)

            X_pocket_blind.append(wav_to_mel(y_pocket, center_start=None)[0])
            X_noise_blind.append(wav_to_mel(y_noisy, center_start=None)[0])
        else:
            X_train_v1.append(mel_clean)
            y_train_v1.append(label)
            w_train_v1.append(1.0)

            noise_pair = (
                train_noises[idx % len(train_noises)],
                train_noises[(idx + 7) % len(train_noises)],
            )
            occ_views = [
                wav_to_mel(apply_pocket_occlusion(y, cutoff_hz=fc), center_start=win_start)[0]
                for fc in TRAIN_FLYWHEEL_CUTOFFS_HZ
            ]
            noise_views = [
                wav_to_mel(mix_real_interferer(y, nc, snr_db=snr), center_start=win_start)[0]
                for nc, snr in zip(noise_pair, TRAIN_FLYWHEEL_SNRS_DB)
            ]

            X_train_v2.extend([mel_clean, *occ_views, *noise_views])
            y_train_v2.extend([label] * 5)
            w_train_v2.extend([flywheel_weight] * 5)

    X_v1_arr, y_v1_arr = np.array(X_train_v1), np.array(y_train_v1)
    w_v1_arr = np.array(w_train_v1, dtype=np.float32)
    X_v2_arr, y_v2_arr = np.array(X_train_v2), np.array(y_train_v2)
    w_v2_arr = np.array(w_train_v2, dtype=np.float32)

    os.makedirs("data/golden_eval", exist_ok=True)
    for name, arr in [
        ("X_train_v1", X_v1_arr),
        ("y_train_v1", y_v1_arr),
        ("w_train_v1", w_v1_arr),
        ("X_train_v2", X_v2_arr),
        ("y_train_v2", y_v2_arr),
        ("w_train_v2", w_v2_arr),
        ("X_pocket_occluded_blind", np.array(X_pocket_blind)),
        ("X_appliance_noise_3db_blind", np.array(X_noise_blind)),
    ]:
        np.save(f"data/golden_eval/{name}.npy", arr)

    slice_arrays = {}
    for s_name, (X_s, y_s) in eval_slices.items():
        X_s_arr, y_s_arr = np.array(X_s), np.array(y_s)
        slice_arrays[s_name] = (X_s_arr, y_s_arr)
        np.save(f"data/golden_eval/X_{s_name}.npy", X_s_arr)
        np.save(f"data/golden_eval/y_{s_name}.npy", y_s_arr)

    psi_rows = [
        {
            "evaluation_slice": s_name,
            "eval_samples": len(X_s_arr),
            "high_freq_band_psi": hf_psi,
            "noise_floor_psi": nf_psi,
            "composite_psi": comp_psi,
            "sla_threshold": 0.25,
            "drift_status": status,
        }
        for s_name, (X_s_arr, _) in slice_arrays.items()
        for hf_psi, nf_psi, comp_psi, status in [compute_spectral_psi(X_v1_arr, X_s_arr)]
    ]

    psi_df = pd.DataFrame(psi_rows)
    psi_csv = "reports/02_psi_spectral_drift_audit.csv"
    psi_df.to_csv(psi_csv, index=False)

    print(f"[Stage 2B] PSI Drift Audit saved -> {psi_csv}")
    print(psi_df.to_string(index=False))

    return {
        "kappa": kappa,
        "audit_df": audit_df,
        "psi_df": psi_df,
        "eval_slices": slice_arrays,
    }


if __name__ == "__main__":
    run_consensus_and_drift()

