#!/usr/bin/env python3
"""Stage 3: Model training on Folds 1-3, empirical vs. balanced prior ablation, and LiteRT (.tflite) quantization."""

import argparse
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import f1_score
from sklearn.utils.class_weight import compute_class_weight

from consensus_drift import load_cached_eval_slices, load_cached_val_slices

NUM_CLASSES = 5
DEFAULT_EPOCHS = 70
ABLAT_SEEDS = (42, 43, 44, 45, 46)


def build_embedded_cnn(seed: int = 42) -> tf.keras.Model:
    """Builds a compact 2D CNN (<30 KB INT8) using integer-compatible operators."""
    tf.keras.utils.set_random_seed(seed)
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(64, 64, 1), name="mel_input"),
            tf.keras.layers.Conv2D(
                16, (3, 3), activation="relu", padding="same", name="conv1"
            ),
            tf.keras.layers.MaxPooling2D((2, 2), name="pool1"),
            tf.keras.layers.Conv2D(
                32, (3, 3), activation="relu", padding="same", name="conv2"
            ),
            tf.keras.layers.MaxPooling2D((2, 2), name="pool2"),
            tf.keras.layers.Conv2D(
                32, (3, 3), activation="relu", padding="same", name="conv3"
            ),
            tf.keras.layers.GlobalAveragePooling2D(name="gap"),
            tf.keras.layers.Dense(32, activation="relu", name="fc1"),
            tf.keras.layers.Dense(NUM_CLASSES, activation="softmax", name="probs"),
        ],
        name="pixelsense_lite_cnn",
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.002),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def _compute_sample_weights(
    y: np.ndarray,
    w_noisy_label: np.ndarray | None = None,
    use_empirical_prior: bool = True,
) -> np.ndarray:
    """Computes per-sample training weights under either empirical class priors or uniform balanced priors.

    When `use_empirical_prior=False`, inverse class-frequency weights (`class_weight='balanced'`)
    shift the effective class prior from the natural Folds 1-3 distribution (53.6% background_noise)
    to a uniform 20% prior, which inflates background false positives at inference.
    When `use_empirical_prior=True`, unit class weights preserve the natural empirical prior
    directly during training without post-hoc logit threshold tuning.
    """
    if use_empirical_prior:
        base_w = np.ones(len(y), dtype=np.float32)
    else:
        classes = np.unique(y)
        weights = compute_class_weight(class_weight="balanced", classes=classes, y=y)
        cw = {int(c): float(w) for c, w in zip(classes, weights)}
        base_w = np.array([cw[int(yi)] for yi in y], dtype=np.float32)

    if w_noisy_label is None:
        return base_w
    return (base_w * np.asarray(w_noisy_label, dtype=np.float32)).astype(np.float32)


def compile_tflite_suite(
    model: tf.keras.Model, X_calib: np.ndarray, version_tag: str
) -> dict[str, str]:
    """Compiles a Keras model into FP32, FP16, and full-integer INT8 .tflite flatbuffers."""
    out_dir = os.path.join("models", version_tag)
    os.makedirs(out_dir, exist_ok=True)
    paths = {}

    def representative_dataset():
        for idx in np.linspace(0, len(X_calib) - 1, min(150, len(X_calib)), dtype=int):
            yield [X_calib[idx : idx + 1].astype(np.float32)]

    for q_name in ("fp32", "fp16", "int8"):
        conv = tf.lite.TFLiteConverter.from_keras_model(model)
        if q_name == "fp16":
            conv.optimizations = [tf.lite.Optimize.DEFAULT]
            conv.target_spec.supported_types = [tf.float16]
        elif q_name == "int8":
            conv.optimizations = [tf.lite.Optimize.DEFAULT]
            conv.representative_dataset = representative_dataset
            conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
            conv.inference_input_type = tf.int8
            conv.inference_output_type = tf.int8

        p = os.path.join(out_dir, f"model_{q_name}.tflite")
        with open(p, "wb") as f:
            f.write(conv.convert())
        paths[q_name] = p
        print(
            f"  -> [{version_tag}] {os.path.basename(p)} ({q_name}): "
            f"{os.path.getsize(p) / 1024.0:.2f} KB"
        )

    return paths


def _fit_candidate(
    suffix: str,
    epochs: int = DEFAULT_EPOCHS,
    use_noisy_label_weights: bool = True,
    use_empirical_prior: bool = True,
    seed: int = 42,
) -> tuple[tf.keras.Model, np.ndarray]:
    """Loads cached Folds 1-3 tensors for a candidate suffix ('v1' or 'v2') and trains a seeded CNN."""
    X_train = np.load(f"data/golden_eval/X_train_{suffix}.npy")
    y_train = np.load(f"data/golden_eval/y_train_{suffix}.npy")
    w_path = f"data/golden_eval/w_train_{suffix}.npy"
    w_noisy = (
        np.load(w_path)
        if (use_noisy_label_weights and os.path.exists(w_path))
        else None
    )
    model = build_embedded_cnn(seed=seed)
    model.fit(
        X_train,
        y_train,
        epochs=epochs,
        batch_size=16,
        sample_weight=_compute_sample_weights(
            y_train, w_noisy, use_empirical_prior=use_empirical_prior
        ),
        verbose=0,
    )
    return model, X_train


def train_and_export_version(
    version_tag: str, epochs: int = DEFAULT_EPOCHS
) -> dict[str, str]:
    """Loads cached Folds 1-3 tensors for a release candidate, trains, and exports .tflite."""
    config_map = {
        "v1_baseline": ("v1", False, False),
        "v2_robust_augmented": ("v2", True, True),
    }
    if version_tag not in config_map:
        raise ValueError(f"Unsupported version_tag: {version_tag}")
    suffix, use_noisy_w, use_emp_prior = config_map[version_tag]

    print(f"\n[Stage 3] Training {version_tag} on Folds 1-3 ({epochs} epochs)...")
    model, X_train = _fit_candidate(
        suffix,
        epochs=epochs,
        use_noisy_label_weights=use_noisy_w,
        use_empirical_prior=use_emp_prior,
    )
    return compile_tflite_suite(model, X_train, version_tag)


def _eval_keras_slices(
    model: tf.keras.Model, slices: dict[str, tuple[np.ndarray, np.ndarray]]
) -> tuple[dict[str, float], dict[str, float]]:
    f1s, bg_fprs = {}, {}
    for s_name, (X_s, y_s) in slices.items():
        preds = np.argmax(model.predict(X_s, verbose=0), axis=-1)
        f1s[s_name] = round(
            float(f1_score(y_s, preds, average="macro", zero_division=0) * 100.0), 2
        )
        bg_mask = y_s == 0
        bg_fprs[s_name] = round(
            float(np.mean(preds[bg_mask] != 0) * 100.0), 2
        )
    return f1s, bg_fprs


def _eval_int8_slices(
    model: tf.keras.Model,
    X_calib: np.ndarray,
    slices: dict[str, tuple[np.ndarray, np.ndarray]],
) -> tuple[dict[str, float], dict[str, float]]:
    """Compiles an in-memory INT8 .tflite flatbuffer and evaluates Macro F1 and BG FPR across slices."""
    def representative_dataset():
        for idx in np.linspace(0, len(X_calib) - 1, min(150, len(X_calib)), dtype=int):
            yield [X_calib[idx : idx + 1].astype(np.float32)]

    conv = tf.lite.TFLiteConverter.from_keras_model(model)
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    conv.representative_dataset = representative_dataset
    conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    conv.inference_input_type = tf.int8
    conv.inference_output_type = tf.int8
    tflite_buf = conv.convert()

    interp = tf.lite.Interpreter(model_content=tflite_buf)
    interp.allocate_tensors()
    in_det = interp.get_input_details()[0]
    out_det = interp.get_output_details()[0]
    scale, zp = in_det["quantization"]

    f1s, bg_fprs = {}, {}
    for s_name, (X_s, y_s) in slices.items():
        preds = []
        for i in range(len(X_s)):
            x_q = np.clip(np.round(X_s[i : i + 1] / scale + zp), -128, 127).astype(
                np.int8
            )
            interp.set_tensor(in_det["index"], x_q)
            interp.invoke()
            out = interp.get_tensor(out_det["index"])
            preds.append(int(np.argmax(out, axis=-1)[0]))
        preds_arr = np.array(preds)
        f1s[s_name] = round(
            float(f1_score(y_s, preds_arr, average="macro", zero_division=0) * 100.0),
            2,
        )
        bg_mask = y_s == 0
        bg_fprs[s_name] = round(
            float(np.mean(preds_arr[bg_mask] != 0) * 100.0), 2
        )
    return f1s, bg_fprs


def run_confounder_ablations(
    out_csv: str = "reports/03_training_and_ablation_metrics.csv",
    seeds: tuple[int, ...] = ABLAT_SEEDS,
) -> dict[str, dict[str, float]]:
    """Runs step-matched and 2x2 factorial (OOF weights x empirical vs. balanced class prior) ablations in FP32 and INT8 across 5 seeds."""
    val_slices = load_cached_val_slices()
    eval_slices = load_cached_eval_slices()

    configs = [
        ("v1_baseline", "v1", DEFAULT_EPOCHS, False, False),
        ("step_matched_clean", "v1", DEFAULT_EPOCHS * 5, False, False),
        ("v2_augmentation_only", "v2", DEFAULT_EPOCHS, False, False),
        ("v2_aug_oof_weights_only", "v2", DEFAULT_EPOCHS, True, False),
        ("v2_aug_empirical_prior_only", "v2", DEFAULT_EPOCHS, False, True),
        ("v2_robust_augmented", "v2", DEFAULT_EPOCHS, True, True),
    ]
    rows = []
    results = {}
    for name, suffix, epochs, use_noisy_w, use_emp_prior in configs:
        seed_val_means = []
        seed_eval_means = []
        seed_val_fprs = []
        seed_eval_fprs = []
        seed_int8_eval_means = []
        seed_int8_eval_fprs = []
        s42_val_metrics = {}
        s42_val_fprs = {}
        s42_eval_metrics = {}
        s42_eval_fprs = {}
        s42_int8_metrics = {}
        s42_int8_fprs = {}
        n_views = 0

        for s in seeds:
            model, X_tr = _fit_candidate(
                suffix,
                epochs=epochs,
                use_noisy_label_weights=use_noisy_w,
                use_empirical_prior=use_emp_prior,
                seed=s,
            )
            n_views = len(X_tr)
            v_f1, v_fpr = _eval_keras_slices(model, val_slices)
            e_f1, e_fpr = _eval_keras_slices(model, eval_slices)
            i_f1, i_fpr = _eval_int8_slices(model, X_tr, eval_slices)

            seed_val_means.append(float(np.mean(list(v_f1.values()))))
            seed_eval_means.append(float(np.mean(list(e_f1.values()))))
            seed_val_fprs.append(float(max(v_fpr.values())))
            seed_eval_fprs.append(float(max(e_fpr.values())))
            seed_int8_eval_means.append(float(np.mean(list(i_f1.values()))))
            seed_int8_eval_fprs.append(float(max(i_fpr.values())))

            if s == seeds[0]:
                s42_val_metrics = v_f1
                s42_val_fprs = v_fpr
                s42_eval_metrics = e_f1
                s42_eval_fprs = e_fpr
                s42_int8_metrics = i_f1
                s42_int8_fprs = i_fpr

        val_mean = round(float(np.mean(list(s42_val_metrics.values()))), 2)
        eval_mean = round(float(np.mean(list(s42_eval_metrics.values()))), 2)
        int8_mean = round(float(np.mean(list(s42_int8_metrics.values()))), 2)
        results[name] = s42_eval_metrics
        rows.append(
            {
                "configuration": name,
                "train_views": n_views,
                "epochs": epochs,
                "noisy_label_downweighted": use_noisy_w,
                "empirical_class_prior": use_emp_prior,
                "val_f1_clean_%": s42_val_metrics["clean"],
                "val_f1_pocket_occluded_%": s42_val_metrics["pocket_occluded"],
                "val_f1_appliance_noise_3db_%": s42_val_metrics["appliance_noise_3db"],
                "val_macro_f1_mean_%": val_mean,
                "val_max_slice_bg_fpr_%": round(float(max(s42_val_fprs.values())), 2),
                "f1_clean_%": s42_eval_metrics["clean"],
                "f1_pocket_occluded_%": s42_eval_metrics["pocket_occluded"],
                "f1_appliance_noise_3db_%": s42_eval_metrics["appliance_noise_3db"],
                "f1_macro_mean_%": eval_mean,
                "eval_max_slice_bg_fpr_%": round(float(max(s42_eval_fprs.values())), 2),
                "int8_f1_clean_%": s42_int8_metrics["clean"],
                "int8_f1_pocket_occluded_%": s42_int8_metrics["pocket_occluded"],
                "int8_f1_appliance_noise_3db_%": s42_int8_metrics[
                    "appliance_noise_3db"
                ],
                "int8_f1_macro_mean_%": int8_mean,
                "int8_eval_max_slice_bg_fpr_%": round(
                    float(max(s42_int8_fprs.values())), 2
                ),
                "val_f1_5seed_mean_%": round(float(np.mean(seed_val_means)), 2),
                "val_f1_5seed_std_%": round(float(np.std(seed_val_means)), 2),
                "val_bg_fpr_5seed_mean_%": round(float(np.mean(seed_val_fprs)), 2),
                "eval_f1_5seed_mean_%": round(float(np.mean(seed_eval_means)), 2),
                "eval_f1_5seed_std_%": round(float(np.std(seed_eval_means)), 2),
                "eval_bg_fpr_5seed_mean_%": round(float(np.mean(seed_eval_fprs)), 2),
                "eval_bg_fpr_5seed_std_%": round(float(np.std(seed_eval_fprs)), 2),
                "int8_eval_f1_5seed_mean_%": round(
                    float(np.mean(seed_int8_eval_means)), 2
                ),
                "int8_eval_f1_5seed_std_%": round(
                    float(np.std(seed_int8_eval_means)), 2
                ),
                "int8_eval_bg_fpr_5seed_mean_%": round(
                    float(np.mean(seed_int8_eval_fprs)), 2
                ),
                "int8_eval_bg_fpr_5seed_std_%": round(
                    float(np.std(seed_int8_eval_fprs)), 2
                ),
            }
        )

    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"[Stage 3B] Training & 5-seed INT8 ablation summary saved -> {out_csv}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train and quantize PixelSense-Lite release candidates."
    )
    parser.add_argument(
        "--version",
        choices=["v1_baseline", "v2_robust_augmented", "all"],
        default="all",
        help="Which release candidate to train and export.",
    )
    parser.add_argument(
        "--run-ablations",
        action="store_true",
        help="Run step-matched and 2x2 factorial ablations across 5 seeds.",
    )
    args = parser.parse_args()

    versions = (
        ["v1_baseline", "v2_robust_augmented"]
        if args.version == "all"
        else [args.version]
    )
    for ver in versions:
        train_and_export_version(ver)
    if args.run_ablations:
        run_confounder_ablations()


