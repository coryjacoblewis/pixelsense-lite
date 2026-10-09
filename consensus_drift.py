#!/usr/bin/env python3
"""Stage 2: Out-of-fold label-noise audit, Fold 4 spectral PSI shift check, and blind-window slice generation."""

import os
import librosa
import numpy as np
import pandas as pd
from scipy.signal import butter, lfilter
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import cohen_kappa_score
from sklearn.model_selection import StratifiedGroupKFold

from ingest_qa import ALL_MODEL_CLASSES, CLASS_MAP, INTERFERER_CLASSES

TARGET_SR = 16000
TRAIN_FOLDS = (1, 2, 3)
VAL_FOLD = 4
EVAL_FOLD = 5
SLICES = ["clean", "pocket_occluded", "appliance_noise_3db"]
EVAL_POCKET_CUTOFF_HZ = 1600.0
EVAL_NOISE_SNR_DB = 3.0
TRAIN_AUG_CUTOFFS_HZ = (1350.0, 2100.0)
TRAIN_AUG_SNRS_DB = (2.0, 5.5)
NOISY_LABEL_SAMPLE_WEIGHT = 0.35


def load_cached_eval_slices() -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Loads the locked Fold 5 test slices from data/golden_eval/."""
    slices = {}
    for s in SLICES:
        X_s = np.load(f"data/golden_eval/X_{s}.npy")
        y_s = np.load(f"data/golden_eval/y_{s}.npy")
        slices[s] = (X_s, y_s)
    return slices


def load_cached_val_slices() -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Loads the Fold 4 validation slices from data/golden_eval/."""
    slices = {}
    for s in SLICES:
        X_s = np.load(f"data/golden_eval/X_val_{s}.npy")
        y_s = np.load(f"data/golden_eval/y_val_{s}.npy")
        slices[s] = (X_s, y_s)
    return slices


def load_cached_eval_sources() -> np.ndarray:
    """Loads the Fold 5 source recording IDs (src_file) for clustered bootstrap resampling."""
    return np.load("data/golden_eval/src_eval.npy")


def apply_pocket_occlusion(
    y: np.ndarray, sr: int = TARGET_SR, cutoff_hz: float = EVAL_POCKET_CUTOFF_HZ
) -> np.ndarray:
    """Applies a 4th-order Butterworth low-pass filter modeling high-frequency acoustic attenuation."""
    b, a = butter(4, cutoff_hz / (0.5 * sr), btype="low")
    return lfilter(b, a, y).astype(np.float32)


def mix_real_interferer(
    clean: np.ndarray, interferer: np.ndarray, snr_db: float = EVAL_NOISE_SNR_DB
) -> np.ndarray:
    """Mixes target audio with interferer audio at the specified 5.0s full-clip RMS SNR (dB)."""
    if len(interferer) < len(clean):
        reps = int(np.ceil(len(clean) / len(interferer)))
        interferer = np.tile(interferer, reps)
    interferer = interferer[: len(clean)]

    clean_rms = float(np.sqrt(np.mean(clean**2) + 1e-9))
    int_rms = float(np.sqrt(interferer.dot(interferer) / len(interferer) + 1e-9))
    scale = (clean_rms / (10.0 ** (snr_db / 20.0))) / int_rms
    return (clean + interferer * scale).astype(np.float32)


def wav_to_mel(y: np.ndarray) -> np.ndarray:
    """Selects the loudest 2.0s window (16 kHz) blindly from y and computes a (64, 64, 1) Log-Mel spectrogram."""
    win = TARGET_SR * 2
    if len(y) > win:
        step = TARGET_SR // 4
        energies = [
            float(np.sum(y[i : i + win] ** 2))
            for i in range(0, len(y) - win + 1, step)
        ]
        win_start = int(np.argmax(energies)) * step
        y_win = y[win_start : win_start + win]
    else:
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
    return norm_mel.astype(np.float32)[..., np.newaxis]


def _build_oof_classifier(seed: int = 99) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=250, max_depth=12, class_weight="balanced", random_state=seed
    )


def run_oof_label_noise_audit(
    X_mel: np.ndarray,
    y_esc50: np.ndarray,
    filenames: list[str],
    categories: list[str],
    src_files: np.ndarray,
    folds: np.ndarray | None = None,
    out_csv: str = "reports/02_oof_label_noise_audit.csv",
) -> tuple[float, pd.DataFrame]:
    """Audits Folds 1-3 training clips for label noise via source-grouped out-of-fold RF disagreement."""
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

    train_mask = (
        np.isin(folds, TRAIN_FOLDS)
        if folds is not None
        else np.ones(len(y_esc50), dtype=bool)
    )
    holdout_mask = ~train_mask
    if np.any(holdout_mask):
        assert set(src_files[train_mask]).isdisjoint(set(src_files[holdout_mask]))

    X_tr_pool = X_flat[train_mask]
    y_tr_pool = y_esc50[train_mask]
    src_tr_pool = src_files[train_mask]
    tr_indices = np.where(train_mask)[0]

    oof_preds = np.zeros(len(tr_indices), dtype=int)
    oof_conf = np.zeros(len(tr_indices), dtype=np.float32)

    sgkf = StratifiedGroupKFold(n_splits=len(TRAIN_FOLDS))
    for tr_idx, va_idx in sgkf.split(X_tr_pool, y_tr_pool, groups=src_tr_pool):
        assert set(src_tr_pool[tr_idx]).isdisjoint(set(src_tr_pool[va_idx]))
        clf = _build_oof_classifier()
        clf.fit(X_tr_pool[tr_idx], y_tr_pool[tr_idx])
        probs = clf.predict_proba(X_tr_pool[va_idx])
        oof_preds[va_idx] = np.argmax(probs, axis=1)
        oof_conf[va_idx] = np.max(probs, axis=1)

    kappa = float(cohen_kappa_score(y_tr_pool, oof_preds))

    audit_rows = []
    for k, orig_idx in enumerate(tr_indices):
        disagreement = (y_tr_pool[k] != oof_preds[k]) and (oof_conf[k] >= 0.65)
        fold_val = int(folds[orig_idx]) if folds is not None else 1
        sample_action = (
            "DOWNWEIGHT_NOISY_LABEL" if disagreement else "KEEP_UNIT_WEIGHT"
        )
        sample_weight = NOISY_LABEL_SAMPLE_WEIGHT if disagreement else 1.0
        audit_rows.append(
            {
                "filename": filenames[orig_idx],
                "fold": fold_val,
                "split_role": "train_pool",
                "src_file": int(src_files[orig_idx]),
                "source_category": categories[orig_idx],
                "esc50_label": ALL_MODEL_CLASSES[y_tr_pool[k]],
                "oof_rf_pred": ALL_MODEL_CLASSES[oof_preds[k]],
                "oof_rf_conf": round(float(oof_conf[k]), 3),
                "label_agreement": bool(y_tr_pool[k] == oof_preds[k]),
                "sample_action": sample_action,
                "sample_weight": sample_weight,
            }
        )

    audit_df = pd.DataFrame(audit_rows)
    out_dir = os.path.dirname(out_csv)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    audit_df.to_csv(out_csv, index=False)

    downweighted_count = int(
        (audit_df["sample_action"] == "DOWNWEIGHT_NOISY_LABEL").sum()
    )
    agreement_pct = float(audit_df["label_agreement"].mean() * 100.0)
    print(
        f"[Stage 2A] Folds 1-3 OOF RF Agreement: {agreement_pct:.1f}% | "
        f"OOF Kappa: {kappa:.3f} | Down-weighted Noisy Labels: {downweighted_count} -> {out_csv}"
    )
    return kappa, audit_df


def compute_spectral_psi(
    X_ref: np.ndarray, X_target: np.ndarray, bins: int = 10
) -> tuple[float, float, float, str]:
    """Computes orthogonal Population Stability Index (PSI) across HF band and passband dynamic range.

    - high_freq_band_psi: Mean normalized log-Mel energy in stopband bins 32:64 (> ~2.0 kHz).
    - noise_floor_psi: Temporal dynamic range (P95 - P25 across frames) in passband bins 0:24 (< ~1.5 kHz).
      Using (P95 - P25) cancels per-window ref=np.max dB offsets, and restricting to passband bins 0:24
      decouples noise-floor compression from passive 1,600 Hz low-pass filtering.
    """
    ref_2d = X_ref.squeeze(-1)
    tgt_2d = X_target.squeeze(-1)

    ref_hf = np.mean(ref_2d[:, 32:, :], axis=2).flatten()
    tgt_hf = np.mean(tgt_2d[:, 32:, :], axis=2).flatten()

    ref_nf = (
        np.percentile(ref_2d[:, :24, :], 95, axis=2)
        - np.percentile(ref_2d[:, :24, :], 25, axis=2)
    ).flatten()
    tgt_nf = (
        np.percentile(tgt_2d[:, :24, :], 95, axis=2)
        - np.percentile(tgt_2d[:, :24, :], 25, axis=2)
    ).flatten()

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
        status = "HIGH_SHIFT_AUGMENTATION_ACTIVE"
    elif composite_psi >= 0.10:
        status = "MODERATE_SHIFT"
    else:
        status = "STABLE"

    return round(hf_psi, 4), round(nf_psi, 4), round(composite_psi, 4), status


def _append_degraded_views(
    target_dict: dict[str, tuple[list[np.ndarray], list[int]]],
    y: np.ndarray,
    mel_clean: np.ndarray,
    label: int,
    noise_clip: np.ndarray,
) -> None:
    """Generates degraded views using blind 2.0s peak-energy window selection on each degraded waveform."""
    y_pocket = apply_pocket_occlusion(y, cutoff_hz=EVAL_POCKET_CUTOFF_HZ)
    y_noisy = mix_real_interferer(y, noise_clip, snr_db=EVAL_NOISE_SNR_DB)
    for s_name, mel_arr in [
        ("clean", mel_clean),
        ("pocket_occluded", wav_to_mel(y_pocket)),
        ("appliance_noise_3db", wav_to_mel(y_noisy)),
    ]:
        target_dict[s_name][0].append(mel_arr)
        target_dict[s_name][1].append(label)


def run_consensus_and_drift(
    qa_csv_path: str = "reports/01_signal_qa_report.csv",
    enable_augmentation: bool | None = None,
) -> dict:
    """Runs OOF label-noise audit on Folds 1-3, verifies Fold 4 PSI shift sensitivity, and exports blind-window slices."""
    qa_df = pd.read_csv(qa_csv_path)
    clean_df = qa_df[qa_df["qa_status"] == "PASS"].copy().reset_index(drop=True)

    train_src = set(clean_df[clean_df["fold"].isin(TRAIN_FOLDS)]["src_file"])
    val_src = set(clean_df[clean_df["fold"] == VAL_FOLD]["src_file"])
    eval_src = set(clean_df[clean_df["fold"] == EVAL_FOLD]["src_file"])
    assert train_src.isdisjoint(val_src), "src_file leakage between train and val"
    assert train_src.isdisjoint(eval_src), "src_file leakage between train and eval"
    assert val_src.isdisjoint(eval_src), "src_file leakage between val and eval"

    waveforms, all_mels, all_labels = [], [], []
    train_noises, val_noises, eval_noises = [], [], []

    for _, row in clean_df.iterrows():
        y, _ = librosa.load(row["filepath"], sr=TARGET_SR)
        fold = int(row["fold"])
        src_id = int(row["src_file"])
        if row["category"] in INTERFERER_CLASSES:
            if fold in TRAIN_FOLDS:
                train_noises.append(y)
            elif fold == VAL_FOLD:
                val_noises.append(y)
            else:
                eval_noises.append(y)
            cls_name = "background_noise"
        else:
            cls_name = row["category"]
        label = CLASS_MAP[cls_name]
        mel_clean = wav_to_mel(y)

        waveforms.append((y, mel_clean, label, fold, src_id, row["filename"]))
        all_mels.append(mel_clean)
        all_labels.append(label)

    kappa, audit_df = run_oof_label_noise_audit(
        np.array(all_mels),
        np.array(all_labels),
        clean_df["filename"].tolist(),
        clean_df["category"].tolist(),
        clean_df["src_file"].to_numpy(dtype=int),
        folds=clean_df["fold"].to_numpy(dtype=int),
    )
    weight_by_filename = dict(
        zip(audit_df["filename"], audit_df["sample_weight"])
    )

    X_train_v1, y_train_v1, w_train_v1 = [], [], []
    val_slices = {s: ([], []) for s in SLICES}
    eval_slices = {s: ([], []) for s in SLICES}
    val_sources, eval_sources = [], []

    for idx, (y, mel_clean, label, fold, src_id, _) in enumerate(waveforms):
        if fold in TRAIN_FOLDS:
            X_train_v1.append(mel_clean)
            y_train_v1.append(label)
            w_train_v1.append(1.0)
        elif fold == VAL_FOLD:
            val_sources.append(src_id)
            _append_degraded_views(
                val_slices,
                y,
                mel_clean,
                label,
                val_noises[idx % len(val_noises)],
            )
        elif fold == EVAL_FOLD:
            eval_sources.append(src_id)
            _append_degraded_views(
                eval_slices,
                y,
                mel_clean,
                label,
                eval_noises[idx % len(eval_noises)],
            )

    X_v1_arr, y_v1_arr = np.array(X_train_v1), np.array(y_train_v1)
    w_v1_arr = np.array(w_train_v1, dtype=np.float32)

    val_slice_arrays, eval_slice_arrays = {}, {}
    os.makedirs("data/golden_eval", exist_ok=True)
    for s_name in SLICES:
        X_val_s, y_val_s = np.array(val_slices[s_name][0]), np.array(val_slices[s_name][1])
        val_slice_arrays[s_name] = (X_val_s, y_val_s)
        np.save(f"data/golden_eval/X_val_{s_name}.npy", X_val_s)
        np.save(f"data/golden_eval/y_val_{s_name}.npy", y_val_s)

        X_ev_s, y_ev_s = np.array(eval_slices[s_name][0]), np.array(eval_slices[s_name][1])
        eval_slice_arrays[s_name] = (X_ev_s, y_ev_s)
        np.save(f"data/golden_eval/X_{s_name}.npy", X_ev_s)
        np.save(f"data/golden_eval/y_{s_name}.npy", y_ev_s)

    np.save("data/golden_eval/src_val.npy", np.array(val_sources, dtype=int))
    np.save("data/golden_eval/src_eval.npy", np.array(eval_sources, dtype=int))

    # Audit natural cross-fold stability (Fold 4 clean) and synthetic stress-slice PSI shift vs. Folds 1-3
    psi_rows = [
        {
            "validation_slice": s_name,
            "split": f"fold_{VAL_FOLD}_validation",
            "val_samples": len(X_val_s),
            "high_freq_band_psi": hf_psi,
            "noise_floor_psi": nf_psi,
            "composite_psi": comp_psi,
            "shift_threshold": 0.25,
            "shift_status": status,
        }
        for s_name, (X_val_s, _) in val_slice_arrays.items()
        for hf_psi, nf_psi, comp_psi, status in [
            compute_spectral_psi(X_v1_arr, X_val_s)
        ]
    ]

    psi_df = pd.DataFrame(psi_rows)
    psi_csv = "reports/02_psi_spectral_drift_audit.csv"
    psi_df.to_csv(psi_csv, index=False)

    print(f"[Stage 2B] Validation (Fold {VAL_FOLD}) Spectral Shift Audit saved -> {psi_csv}")
    print(psi_df.to_string(index=False))

    detected_shift = bool(
        (psi_df["shift_status"] == "HIGH_SHIFT_AUGMENTATION_ACTIVE").any()
    )
    augmentation_active = (
        detected_shift if enable_augmentation is None else bool(enable_augmentation)
    )

    # Build v2 training set with OOF noisy-label down-weighting and conditional 5x acoustic augmentation
    X_train_v2, y_train_v2, w_train_v2 = [], [], []
    for idx, (y, mel_clean, label, fold, _, fname) in enumerate(waveforms):
        if fold not in TRAIN_FOLDS:
            continue
        sample_w = float(weight_by_filename.get(fname, 1.0))

        if augmentation_active:
            noise_pair = (
                train_noises[idx % len(train_noises)],
                train_noises[(idx + 7) % len(train_noises)],
            )
            occ_views = [
                wav_to_mel(apply_pocket_occlusion(y, cutoff_hz=fc))
                for fc in TRAIN_AUG_CUTOFFS_HZ
            ]
            noise_views = [
                wav_to_mel(mix_real_interferer(y, nc, snr_db=snr))
                for nc, snr in zip(noise_pair, TRAIN_AUG_SNRS_DB)
            ]
            views = [mel_clean, *occ_views, *noise_views]
        else:
            views = [mel_clean]

        X_train_v2.extend(views)
        y_train_v2.extend([label] * len(views))
        w_train_v2.extend([sample_w] * len(views))

    X_v2_arr, y_v2_arr = np.array(X_train_v2), np.array(y_train_v2)
    w_v2_arr = np.array(w_train_v2, dtype=np.float32)

    for name, arr in [
        ("X_train_v1", X_v1_arr),
        ("y_train_v1", y_v1_arr),
        ("w_train_v1", w_v1_arr),
        ("X_train_v2", X_v2_arr),
        ("y_train_v2", y_v2_arr),
        ("w_train_v2", w_v2_arr),
    ]:
        np.save(f"data/golden_eval/{name}.npy", arr)

    return {
        "kappa": kappa,
        "audit_df": audit_df,
        "psi_df": psi_df,
        "augmentation_active": augmentation_active,
        "val_slices": val_slice_arrays,
        "eval_slices": eval_slice_arrays,
    }


if __name__ == "__main__":
    run_consensus_and_drift()


