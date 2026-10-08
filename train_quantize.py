#!/usr/bin/env python3
"""Stage 3: Model training and LiteRT (.tflite) quantization."""

import argparse
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import f1_score
from sklearn.utils.class_weight import compute_class_weight

from consensus_drift import load_cached_eval_slices

NUM_CLASSES = 5


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
    y: np.ndarray, w_adjudication: np.ndarray | None = None
) -> np.ndarray:
    """Computes balanced class weights multiplied by optional per-sample adjudication weights."""
    classes = np.unique(y)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y)
    cw = {int(c): float(w) for c, w in zip(classes, weights)}
    if w_adjudication is None:
        return np.array([cw[int(yi)] for yi in y], dtype=np.float32)
    return np.array(
        [cw[int(yi)] * float(wi) for yi, wi in zip(y, w_adjudication)],
        dtype=np.float32,
    )


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
    suffix: str, epochs: int = 65, use_adjudication_weights: bool = True
) -> tuple[tf.keras.Model, np.ndarray]:
    """Loads cached split tensors for a candidate suffix ('v1' or 'v2') and trains a seeded CNN."""
    X_train = np.load(f"data/golden_eval/X_train_{suffix}.npy")
    y_train = np.load(f"data/golden_eval/y_train_{suffix}.npy")
    w_path = f"data/golden_eval/w_train_{suffix}.npy"
    w_adj = (
        np.load(w_path)
        if (use_adjudication_weights and os.path.exists(w_path))
        else None
    )
    model = build_embedded_cnn(seed=42)
    model.fit(
        X_train,
        y_train,
        epochs=epochs,
        batch_size=16,
        sample_weight=_compute_sample_weights(y_train, w_adj),
        verbose=0,
    )
    return model, X_train


def train_and_export_version(version_tag: str, epochs: int = 65) -> dict[str, str]:
    """Loads cached split tensors for a release candidate, trains, and exports .tflite."""
    suffix = {"v1_baseline": "v1", "v2_data_flywheel": "v2"}.get(version_tag)
    if not suffix:
        raise ValueError(f"Unsupported version_tag: {version_tag}")

    print(f"\n[Stage 3] Training {version_tag} ({epochs} epochs)...")
    model, X_train = _fit_candidate(suffix, epochs=epochs, use_adjudication_weights=True)
    return compile_tflite_suite(model, X_train, version_tag)


def _eval_keras_slices(
    model: tf.keras.Model, slices: dict[str, tuple[np.ndarray, np.ndarray]]
) -> dict[str, float]:
    return {
        s_name: round(
            float(
                f1_score(
                    y_s,
                    np.argmax(model.predict(X_s, verbose=0), axis=-1),
                    average="macro",
                )
                * 100.0
            ),
            2,
        )
        for s_name, (X_s, y_s) in slices.items()
    }


def run_confounder_ablations(
    out_csv: str = "reports/03_training_and_ablation_metrics.csv",
) -> dict[str, dict[str, float]]:
    """Runs step-matched and uniform-weight ablations on Fold 5 and saves Stage 3 metrics."""
    slices = load_cached_eval_slices()

    configs = [
        ("v1_baseline", "v1", 65, True),
        ("step_matched_clean", "v1", 325, False),
        ("v2_augmentation_only", "v2", 65, False),
        ("v2_data_flywheel", "v2", 65, True),
    ]
    rows = []
    results = {}
    for name, suffix, epochs, use_adj in configs:
        model, X_tr = _fit_candidate(
            suffix, epochs=epochs, use_adjudication_weights=use_adj
        )
        metrics = _eval_keras_slices(model, slices)
        results[name] = metrics
        rows.append(
            {
                "configuration": name,
                "train_views": len(X_tr),
                "epochs": epochs,
                "adjudication_weighted": use_adj,
                "f1_clean_%": metrics["clean"],
                "f1_pocket_occluded_%": metrics["pocket_occluded"],
                "f1_appliance_noise_3db_%": metrics["appliance_noise_3db"],
            }
        )

    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"[Stage 3B] Training & ablation summary saved -> {out_csv}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train and quantize PixelSense-Lite release candidates."
    )
    parser.add_argument(
        "--version",
        choices=["v1_baseline", "v2_data_flywheel", "all"],
        default="all",
        help="Which release candidate to train and export.",
    )
    parser.add_argument(
        "--run-ablations",
        action="store_true",
        help="Run step-matched and uniform-weight ablations.",
    )
    args = parser.parse_args()

    versions = (
        ["v1_baseline", "v2_data_flywheel"]
        if args.version == "all"
        else [args.version]
    )
    for ver in versions:
        train_and_export_version(ver)
    if args.run_ablations:
        run_confounder_ablations()
