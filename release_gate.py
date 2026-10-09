#!/usr/bin/env python3
"""Stage 4: Subgraph quantization audit, source-clustered bootstrap CIs, and ARM64 release gate."""

import argparse
import hashlib
import os
import re
import sys
import time
import warnings
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import f1_score

from consensus_drift import SLICES, load_cached_eval_slices, load_cached_eval_sources

warnings.filterwarnings(
    "ignore",
    message=r".*tf\.lite\.Interpreter is deprecated.*",
    category=UserWarning,
)

MAX_FLASH_KB = 45.0
MAX_SUBGRAPH_TENSOR_KB = 160.0
MAX_PEAK_OP_IO_KB = 100.0
MAX_ALLOCATE_TENSORS_KB = 512.0
MAX_HOST_P99_LATENCY_MS = 5.0
MAX_ARM64_P95_LATENCY_MS = 1.0
MAX_P99_LATENCY_MS = MAX_HOST_P99_LATENCY_MS
ALLOWED_INT8_BUILTIN_OPS = {
    "CONV_2D",
    "MAX_POOL_2D",
    "MEAN",
    "FULLY_CONNECTED",
    "SOFTMAX",
    "DELEGATE",
}
SLICE_F1_THRESHOLDS = {
    "clean": 65.0,
    "pocket_occluded": 58.0,
    "appliance_noise_3db": 52.0,
}
MAX_BG_FPR_PCT = 15.0
DEFAULT_ARM64_TELEMETRY_PATH = "reports/05_arm64_hardware_telemetry_int8.txt"
DEFAULT_ABLATION_CSV_PATH = "reports/03_training_and_ablation_metrics.csv"


def compute_file_sha256(filepath: str) -> str:
    """Computes the hex-encoded SHA-256 digest of a file on disk."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def audit_int8_hardware_compatibility(interpreter: tf.lite.Interpreter) -> dict:
    """Audits .tflite subgraph tensors and operators for full-integer INT8 compliance.

    Note: `subgraph_tensor_kb` sums the logical buffer sizes across all subgraph tensor
    descriptors in `get_tensor_details()` (weights + intermediate activation tensors; for
    FP16 models on CPU, TFLite also materializes unpacked FP32 `DEQUANTIZE` destination
    tensors). `peak_op_io_kb` measures the largest single-operator input+output working set.
    """
    tensors = interpreter.get_tensor_details()
    non_int_dtypes = (np.float32, np.float16, np.float64, np.int64, np.int16)
    non_int_tensor_count = sum(1 for t in tensors if t["dtype"] in non_int_dtypes)
    int8_count = sum(1 for t in tensors if t["dtype"] in (np.int8, np.uint8))
    int32_bias_count = sum(1 for t in tensors if t["dtype"] == np.int32)
    int_tensor_count = int8_count + int32_bias_count
    int_tensor_ratio = int_tensor_count / float(
        max(1, int_tensor_count + non_int_tensor_count)
    )

    dynamic_count = sum(
        1
        for t in tensors
        if t.get("shape_signature") is not None
        and len(t["shape_signature"]) > 1
        and any(int(dim) < 0 for dim in t["shape_signature"][1:])
    )

    ops_details = interpreter._get_ops_details()
    unsupported_op_count = sum(
        1 for op in ops_details if op["op_name"] not in ALLOWED_INT8_BUILTIN_OPS
    )

    tensor_bytes = sum(
        int(np.prod(t["shape"])) * np.dtype(t["dtype"]).itemsize
        for t in tensors
        if t["shape"] is not None and len(t["shape"]) > 0
    )

    tensor_map = {t["index"]: t for t in tensors}
    peak_op_io_bytes = 0
    for op in ops_details:
        op_tensor_idxs = {
            int(i)
            for i in list(op.get("inputs", [])) + list(op.get("outputs", []))
            if int(i) >= 0 and int(i) in tensor_map
        }
        op_bytes = sum(
            int(np.prod(tensor_map[i]["shape"]))
            * np.dtype(tensor_map[i]["dtype"]).itemsize
            for i in op_tensor_idxs
            if tensor_map[i]["shape"] is not None and len(tensor_map[i]["shape"]) > 0
        )
        peak_op_io_bytes = max(peak_op_io_bytes, op_bytes)

    int8_subgraph_ready = (
        non_int_tensor_count == 0
        and unsupported_op_count == 0
        and dynamic_count == 0
    )

    return {
        "subgraph_tensor_kb": round(tensor_bytes / 1024.0, 2),
        "peak_op_io_kb": round(peak_op_io_bytes / 1024.0, 2),
        "total_subgraph_tensors": len(tensors),
        "int8_tensor_count": int8_count,
        "int32_accumulator_count": int32_bias_count,
        "non_int_tensors": non_int_tensor_count,
        "unsupported_op_count": unsupported_op_count,
        "dynamic_tensor_count": dynamic_count,
        "int_tensor_%": round(int_tensor_ratio * 100.0, 1),
        "int8_subgraph_ready": int8_subgraph_ready,
    }


def _build_stratified_clusters(
    y_true_arr: np.ndarray, src_files: np.ndarray | None
) -> tuple[dict[int, np.ndarray], dict[int, list[int]]]:
    """Groups sample indices by src_file cluster and stratifies clusters by primary class label."""
    if src_files is None or len(src_files) != len(y_true_arr):
        clusters = np.arange(len(y_true_arr))
        cluster_to_idx = {int(c): np.array([int(c)]) for c in clusters}
    else:
        src_arr = np.asarray(src_files)
        clusters = np.unique(src_arr)
        cluster_to_idx = {int(c): np.where(src_arr == c)[0] for c in clusters}

    cluster_label = {
        int(c): int(y_true_arr[cluster_to_idx[int(c)][0]]) for c in clusters
    }
    strata = {
        int(cls): [int(c) for c in clusters if cluster_label[int(c)] == int(cls)]
        for cls in np.unique(y_true_arr)
    }
    return cluster_to_idx, strata


def _sample_stratified_cluster_indices(
    rng: np.random.Generator,
    cluster_to_idx: dict[int, np.ndarray],
    strata: dict[int, list[int]],
) -> np.ndarray:
    sampled = []
    for cls_clusters in strata.values():
        sampled.extend(rng.choice(cls_clusters, size=len(cls_clusters), replace=True))
    return np.concatenate([cluster_to_idx[int(c)] for c in sampled])


def bootstrap_slice_f1_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    src_files: np.ndarray | None = None,
    n_boot: int = 1000,
    seed: int = 42,
) -> tuple[float, float]:
    """Computes a 95% class-stratified cluster-bootstrap CI for Macro F1 grouped by src_file."""
    rng = np.random.default_rng(seed)
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)
    cluster_to_idx, strata = _build_stratified_clusters(y_true_arr, src_files)

    scores = []
    for _ in range(n_boot):
        idx = _sample_stratified_cluster_indices(rng, cluster_to_idx, strata)
        scores.append(
            float(
                f1_score(
                    y_true_arr[idx],
                    y_pred_arr[idx],
                    average="macro",
                    zero_division=0,
                )
                * 100.0
            )
        )
    low, high = np.percentile(scores, [2.5, 97.5])
    return round(float(low), 2), round(float(high), 2)


def bootstrap_paired_delta_f1_ci(
    y_true: np.ndarray,
    preds_v1: np.ndarray,
    preds_v2: np.ndarray,
    src_files: np.ndarray | None = None,
    n_boot: int = 1000,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Computes paired Macro F1 improvement (v2 - v1) and 95% class-stratified cluster-bootstrap CI."""
    rng = np.random.default_rng(seed)
    y_true_arr = np.asarray(y_true)
    p1_arr = np.asarray(preds_v1)
    p2_arr = np.asarray(preds_v2)
    cluster_to_idx, strata = _build_stratified_clusters(y_true_arr, src_files)

    point_delta = float(
        (
            f1_score(y_true_arr, p2_arr, average="macro", zero_division=0)
            - f1_score(y_true_arr, p1_arr, average="macro", zero_division=0)
        )
        * 100.0
    )
    deltas = []
    for _ in range(n_boot):
        idx = _sample_stratified_cluster_indices(rng, cluster_to_idx, strata)
        f1_1 = f1_score(y_true_arr[idx], p1_arr[idx], average="macro", zero_division=0)
        f1_2 = f1_score(y_true_arr[idx], p2_arr[idx], average="macro", zero_division=0)
        deltas.append(float((f1_2 - f1_1) * 100.0))

    low, high = np.percentile(deltas, [2.5, 97.5])
    return round(point_delta, 2), round(float(low), 2), round(float(high), 2)


def parse_arm64_benchmark_telemetry(
    telemetry_path: str = DEFAULT_ARM64_TELEMETRY_PATH,
    op_csv_path: str = "reports/05_arm64_op_profile_int8.csv",
    expected_model_path: str | None = "models/v2_robust_augmented/model_int8.tflite",
) -> dict | None:
    """Parses Linux aarch64 benchmark_model output, verifying SHA-256 model identity, timings, and memory."""
    if not telemetry_path or not os.path.exists(telemetry_path):
        return None

    with open(telemetry_path, "r", encoding="utf-8") as f:
        text = f.read()

    pattern = re.compile(
        r"Running benchmark for at least 200 iterations.*?\n"
        r"INFO:\s*count=(\d+).*?avg=([\d.]+).*?p95=([\d.]+)",
        re.DOTALL,
    )
    matches = pattern.findall(text)
    if len(matches) < 2:
        return None

    raw_count, raw_avg_us, raw_p95_us = matches[0]
    xnn_count, xnn_avg_us, xnn_p95_us = matches[1]
    delegate_applied = "Explicitly applied XNNPACK delegate" in text

    xnn_alloc_match = re.search(
        r"AllocateTensors\s+[\d.]+\s+[\d.]+\s+[\d.]+%\s+[\d.]+%\s+([\d.]+)", text
    )
    if not xnn_alloc_match or not op_csv_path or not os.path.exists(op_csv_path):
        return None
    xnn_alloc_kb = float(xnn_alloc_match.group(1))

    with open(op_csv_path, "r", encoding="utf-8") as cf:
        csv_text = cf.read()
    raw_alloc_match = re.search(
        r"AllocateTensors,\s*[\d.]+,\s*[\d.]+,\s*[\d.]+%,\s*[\d.]+%,\s*([\d.]+)",
        csv_text,
    )
    if not raw_alloc_match:
        return None
    raw_alloc_kb = float(raw_alloc_match.group(1))

    rss_matches = re.findall(
        r"Memory footprint delta from the start of the tool \(MB\):\s*init=([\d.]+)\s+overall=([\d.]+)",
        text,
    )
    raw_rss_mb = round(float(rss_matches[0][1]), 2) if len(rss_matches) >= 1 else 0.0
    xnn_rss_mb = round(float(rss_matches[1][1]), 2) if len(rss_matches) >= 2 else 0.0

    # Verify telemetry log matches the compiled .tflite binary on disk by byte size AND SHA-256 digest
    size_mb_match = re.search(r"The input model file size \(MB\):\s*([\d.]+)", text)
    sha_match = re.search(r"Model SHA256:\s*([0-9a-fA-F]{64})", text)
    logged_sha256 = sha_match.group(1).lower() if sha_match else None
    if expected_model_path:
        if not os.path.exists(expected_model_path):
            return None
        if size_mb_match:
            logged_bytes = int(round(float(size_mb_match.group(1)) * 1_000_000))
            actual_bytes = os.path.getsize(expected_model_path)
            if abs(logged_bytes - actual_bytes) > 16:
                return None
        actual_sha256 = compute_file_sha256(expected_model_path)
        if not logged_sha256 or logged_sha256 != actual_sha256:
            return None

    xnn_p95_ms = round(float(xnn_p95_us) / 1000.0, 4)
    return {
        "model_sha256": logged_sha256,
        "raw_cpu_runs": int(raw_count),
        "raw_cpu_avg_ms": round(float(raw_avg_us) / 1000.0, 4),
        "raw_cpu_p95_ms": round(float(raw_p95_us) / 1000.0, 4),
        "raw_cpu_allocate_tensors_kb": round(raw_alloc_kb, 1),
        "raw_cpu_rss_delta_mb": raw_rss_mb,
        "xnnpack_runs": int(xnn_count),
        "xnnpack_avg_ms": round(float(xnn_avg_us) / 1000.0, 4),
        "xnnpack_p95_ms": xnn_p95_ms,
        "xnnpack_allocate_tensors_kb": round(xnn_alloc_kb, 1),
        "xnnpack_rss_delta_mb": xnn_rss_mb,
        "xnnpack_delegate_applied": delegate_applied,
        "arm64_latency_pass": (
            delegate_applied
            and xnn_p95_ms <= MAX_ARM64_P95_LATENCY_MS
            and raw_alloc_kb <= MAX_ALLOCATE_TENSORS_KB
            and xnn_alloc_kb <= MAX_ALLOCATE_TENSORS_KB
        ),
    }


def evaluate_tflite_binary(
    tflite_path: str,
    eval_slices: dict[str, tuple[np.ndarray, np.ndarray]],
    src_files: np.ndarray | None = None,
) -> dict:
    """Runs subgraph inspection and multi-slice inference on a compiled .tflite binary."""
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()

    in_det = interpreter.get_input_details()[0]
    out_det = interpreter.get_output_details()[0]

    hw_audit = audit_int8_hardware_compatibility(interpreter)

    slice_f1 = {}
    slice_ci = {}
    slice_preds = {}
    slice_bg_fpr = {}
    latencies_ms = []
    bg_false_positives = 0
    bg_total_clips = 0

    for slice_name, (X_s, y_s) in eval_slices.items():
        preds = []
        for i in range(len(X_s)):
            x = X_s[i : i + 1]
            if in_det["dtype"] == np.int8:
                scale, zp = in_det["quantization"]
                x = np.clip(np.round(x / scale + zp), -128, 127).astype(np.int8)
            else:
                x = x.astype(np.float32)

            t0 = time.perf_counter()
            interpreter.set_tensor(in_det["index"], x)
            interpreter.invoke()
            latencies_ms.append((time.perf_counter() - t0) * 1000.0)

            out = interpreter.get_tensor(out_det["index"])
            preds.append(int(np.argmax(out, axis=-1)[0]))

        preds_arr = np.array(preds)
        slice_preds[slice_name] = preds_arr
        bg_mask = y_s == 0
        s_fp = int(np.sum(preds_arr[bg_mask] != 0))
        s_bg_total = int(np.sum(bg_mask))
        bg_false_positives += s_fp
        bg_total_clips += s_bg_total
        slice_bg_fpr[slice_name] = round(
            (s_fp / float(max(1, s_bg_total))) * 100.0, 2
        )

        slice_f1[slice_name] = round(
            float(
                f1_score(y_s, preds_arr, average="macro", zero_division=0) * 100.0
            ),
            2,
        )
        slice_ci[slice_name] = bootstrap_slice_f1_ci(y_s, preds_arr, src_files=src_files)

    p99_ms = round(float(np.percentile(latencies_ms, 99)), 3)
    flash_kb = round(os.path.getsize(tflite_path) / 1024.0, 2)
    bg_fpr_pct = round(
        (bg_false_positives / float(max(1, bg_total_clips))) * 100.0, 2
    )
    max_slice_bg_fpr_pct = round(float(max(slice_bg_fpr.values())), 2)

    return {
        "flash_kb": flash_kb,
        "subgraph_tensor_kb": hw_audit["subgraph_tensor_kb"],
        "peak_op_io_kb": hw_audit["peak_op_io_kb"],
        "non_int_tensors": hw_audit["non_int_tensors"],
        "int_tensor_%": hw_audit["int_tensor_%"],
        "p99_latency_ms": p99_ms,
        "f1_clean_%": slice_f1["clean"],
        "f1_pocket_occluded_%": slice_f1["pocket_occluded"],
        "f1_appliance_noise_3db_%": slice_f1["appliance_noise_3db"],
        "bg_fpr_%": bg_fpr_pct,
        "max_slice_bg_fpr_%": max_slice_bg_fpr_pct,
        "int8_subgraph_ready": hw_audit["int8_subgraph_ready"],
        "slice_ci": slice_ci,
        "slice_preds": slice_preds,
        "slice_bg_fpr": slice_bg_fpr,
    }


def run_release_gate(
    versions: list[str] | None = None,
    eval_slices: dict[str, tuple[np.ndarray, np.ndarray]] | None = None,
    enforce_target: str | None = "v2_robust_augmented",
    out_md: str = "reports/04_release_gate_scorecard.md",
    arm64_telemetry_path: str | None = DEFAULT_ARM64_TELEMETRY_PATH,
    require_arm64_telemetry: bool = False,
    ablation_csv_path: str | None = DEFAULT_ABLATION_CSV_PATH,
) -> pd.DataFrame:
    """Evaluates .tflite binaries, checks 5-seed INT8 stability, and blocks if enforce_target fails any gate."""
    if versions is None:
        versions = ["v1_baseline", "v2_robust_augmented"]
    if eval_slices is None:
        eval_slices = load_cached_eval_slices()

    src_files = (
        load_cached_eval_sources()
        if os.path.exists("data/golden_eval/src_eval.npy")
        else None
    )
    arm64_telemetry = (
        parse_arm64_benchmark_telemetry(arm64_telemetry_path)
        if arm64_telemetry_path
        else None
    )
    if require_arm64_telemetry and arm64_telemetry is None:
        raise RuntimeError(
            f"Release Gate BLOCKED: Required ARM64 telemetry missing or invalid at {arm64_telemetry_path}"
        )

    ablation_df = (
        pd.read_csv(ablation_csv_path).set_index("configuration")
        if (ablation_csv_path and os.path.exists(ablation_csv_path))
        else None
    )

    rows = []
    int8_details = {}

    for ver in versions:
        for q_type in ["fp32", "fp16", "int8"]:
            tfl_path = os.path.join("models", ver, f"model_{q_type}.tflite")
            if not os.path.exists(tfl_path):
                continue

            res = evaluate_tflite_binary(tfl_path, eval_slices, src_files=src_files)
            int8_ready = res.pop("int8_subgraph_ready")
            slice_ci = res.pop("slice_ci")
            slice_preds = res.pop("slice_preds")
            slice_bg_fpr = res.pop("slice_bg_fpr")

            if q_type == "int8":
                int8_details[ver] = {
                    "ci": slice_ci,
                    "preds": slice_preds,
                    "bg_fpr": slice_bg_fpr,
                }

            arm64_ok = True
            if (
                ver == enforce_target
                and q_type == "int8"
                and arm64_telemetry is not None
            ):
                arm64_ok = bool(arm64_telemetry["arm64_latency_pass"])

            multiseed_ok = True
            if (
                q_type == "int8"
                and ablation_df is not None
                and ver in ablation_df.index
                and "int8_eval_bg_fpr_5seed_mean_%" in ablation_df.columns
            ):
                min_mean_f1 = float(np.mean(list(SLICE_F1_THRESHOLDS.values())))
                multiseed_ok = (
                    float(ablation_df.loc[ver, "int8_eval_f1_5seed_mean_%"])
                    >= min_mean_f1
                    and float(ablation_df.loc[ver, "int8_eval_bg_fpr_5seed_mean_%"])
                    <= MAX_BG_FPR_PCT
                )

            hw_pass = (
                res["flash_kb"] <= MAX_FLASH_KB
                and res["subgraph_tensor_kb"] <= MAX_SUBGRAPH_TENSOR_KB
                and res["peak_op_io_kb"] <= MAX_PEAK_OP_IO_KB
                and int8_ready
                and res["p99_latency_ms"] <= MAX_P99_LATENCY_MS
                and arm64_ok
            )
            quality_pass = (
                res["f1_clean_%"] >= SLICE_F1_THRESHOLDS["clean"]
                and res["f1_pocket_occluded_%"]
                >= SLICE_F1_THRESHOLDS["pocket_occluded"]
                and res["f1_appliance_noise_3db_%"]
                >= SLICE_F1_THRESHOLDS["appliance_noise_3db"]
                and res["bg_fpr_%"] <= MAX_BG_FPR_PCT
                and res["max_slice_bg_fpr_%"] <= MAX_BG_FPR_PCT
                and multiseed_ok
            )

            if hw_pass and quality_pass:
                gate_status = "SHIP (PASS)"
            elif not hw_pass and not quality_pass:
                gate_status = "BLOCKED (HW + SLICE)"
            elif not hw_pass:
                gate_status = "BLOCKED (HW)"
            else:
                gate_status = "BLOCKED (SLICE F1)"

            rows.append(
                {
                    "release": ver,
                    "quant": q_type,
                    **res,
                    "gate_status": gate_status,
                }
            )

    scorecard_df = pd.DataFrame(rows)
    out_dir = os.path.dirname(out_md)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    md_sections = [scorecard_df.to_markdown(index=False)]

    if "v1_baseline" in int8_details and "v2_robust_augmented" in int8_details:
        n_clips = len(eval_slices["clean"][1])
        n_sources = len(np.unique(src_files)) if src_files is not None else n_clips
        ci_rows = []
        for s_name in SLICES:
            y_s = eval_slices[s_name][1]
            p1 = int8_details["v1_baseline"]["preds"][s_name]
            p2 = int8_details["v2_robust_augmented"]["preds"][s_name]
            c1_lo, c1_hi = int8_details["v1_baseline"]["ci"][s_name]
            c2_lo, c2_hi = int8_details["v2_robust_augmented"]["ci"][s_name]
            d_pt, d_lo, d_hi = bootstrap_paired_delta_f1_ci(
                y_s, p1, p2, src_files=src_files
            )
            fpr1 = int8_details["v1_baseline"]["bg_fpr"][s_name]
            fpr2 = int8_details["v2_robust_augmented"]["bg_fpr"][s_name]
            ci_rows.append(
                {
                    "locked_fold5_slice": s_name,
                    "v1_int8_95%_ci": f"[{c1_lo:.2f}%, {c1_hi:.2f}%]",
                    "v2_int8_95%_ci": f"[{c2_lo:.2f}%, {c2_hi:.2f}%]",
                    "paired_delta_f1_%": f"{d_pt:+.2f}%",
                    "paired_delta_95%_ci": f"[{d_lo:+.2f}%, {d_hi:+.2f}%]",
                    "v1_bg_fpr_%": f"{fpr1:.2f}%",
                    "v2_bg_fpr_%": f"{fpr2:.2f}%",
                }
            )
        ci_df = pd.DataFrame(ci_rows)
        md_sections.append(
            f"\n### Source-Clustered Bootstrap 95% CIs & Per-Slice BG FPR (Fold 5: N={n_clips} clips across {n_sources} `src_file` sources)\n\n"
            + ci_df.to_markdown(index=False)
        )

    if arm64_telemetry is not None:
        telemetry_mode = (
            "Live CI ARM64 Benchmark (`benchmark_model`)"
            if require_arm64_telemetry
            else f"Cached CI Reference ARM64 Telemetry (`{arm64_telemetry_path}`)"
        )
        md_sections.append(
            f"\n### {telemetry_mode}\n\n"
            f"- **Verified `.tflite` SHA-256**: `{arm64_telemetry['model_sha256']}`\n"
            f"- **Raw ARM64 CPU (1 Thread)**: avg `{arm64_telemetry['raw_cpu_avg_ms']:.3f} ms`, "
            f"p95 `{arm64_telemetry['raw_cpu_p95_ms']:.3f} ms` ({arm64_telemetry['raw_cpu_runs']} runs, "
            f"`AllocateTensors={arm64_telemetry['raw_cpu_allocate_tensors_kb']:.1f} KB`, "
            f"process RSS delta `{arm64_telemetry['raw_cpu_rss_delta_mb']:.2f} MB`)\n"
            f"- **ARM64 XNNPACK Delegate (1 Thread)**: avg `{arm64_telemetry['xnnpack_avg_ms']:.3f} ms`, "
            f"p95 `{arm64_telemetry['xnnpack_p95_ms']:.3f} ms` ({arm64_telemetry['xnnpack_runs']} runs, "
            f"delegate_applied={arm64_telemetry['xnnpack_delegate_applied']}, "
            f"`AllocateTensors={arm64_telemetry['xnnpack_allocate_tensors_kb']:.1f} KB`, "
            f"process RSS delta `{arm64_telemetry['xnnpack_rss_delta_mb']:.2f} MB`)"
        )

    with open(out_md, "w", encoding="utf-8") as f:
        f.write("\n".join(md_sections) + "\n")

    print(f"[Stage 4] Release Gate Scorecard saved -> {out_md}")
    print(scorecard_df.to_markdown(index=False))

    if enforce_target and enforce_target in versions:
        target_rows = scorecard_df[
            (scorecard_df["release"] == enforce_target)
            & (scorecard_df["quant"] == "int8")
        ]
        if target_rows.empty or target_rows.iloc[0]["gate_status"] != "SHIP (PASS)":
            status_str = (
                target_rows.iloc[0]["gate_status"]
                if not target_rows.empty
                else "MISSING_BINARY"
            )
            raise RuntimeError(
                f"Release Gate BLOCKED production candidate {enforce_target} (int8): {status_str}"
            )

    return scorecard_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run PixelSense-Lite automated release gate evaluation."
    )
    parser.add_argument(
        "--versions",
        nargs="+",
        default=["v1_baseline", "v2_robust_augmented"],
        help="Release versions to evaluate.",
    )
    parser.add_argument(
        "--enforce-target",
        default="v2_robust_augmented",
        help="Release candidate whose int8 binary must pass all gates (or 'none').",
    )
    parser.add_argument(
        "--arm64-telemetry-path",
        default=DEFAULT_ARM64_TELEMETRY_PATH,
        help="Path to native ARM64 benchmark_model telemetry file.",
    )
    parser.add_argument(
        "--require-arm64-telemetry",
        action="store_true",
        help="Fail the release gate if native ARM64 telemetry is missing or invalid.",
    )
    args = parser.parse_args()
    target = None if args.enforce_target.lower() == "none" else args.enforce_target
    try:
        run_release_gate(
            versions=args.versions,
            enforce_target=target,
            arm64_telemetry_path=args.arm64_telemetry_path,
            require_arm64_telemetry=args.require_arm64_telemetry,
        )
    except RuntimeError as exc:
        print(f"\n[FATAL] {exc}", file=sys.stderr)
        sys.exit(1)
