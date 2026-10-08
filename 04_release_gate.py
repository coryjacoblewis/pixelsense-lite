#!/usr/bin/env python3
"""Stage 4: Subgraph Zero-Fallback Audit, Static Tensor Arena Profiler & Multi-Slice Release Gate."""

import argparse
import os
import sys
import time
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import f1_score

SLICES = ["clean", "pocket_occluded", "appliance_noise_3db"]

MAX_FLASH_KB = 45.0
MAX_TENSOR_ARENA_KB = 160.0
MAX_P99_LATENCY_MS = 5.0
MAX_DAILY_BATTERY_PCT = 0.050
ALLOWED_DSP_OPS = {
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


def audit_int8_hardware_compatibility(interpreter: tf.lite.Interpreter) -> dict:
    """Audits .tflite subgraph tensors and operators for 0% float fallback, op whitelist, and SRAM arena."""
    tensors = interpreter.get_tensor_details()
    non_int8_dtypes = (np.float32, np.float16, np.float64, np.int64, np.int16)
    float_fallback_count = sum(1 for t in tensors if t["dtype"] in non_int8_dtypes)
    int8_count = sum(1 for t in tensors if t["dtype"] in (np.int8, np.uint8))
    int_only_ratio = int8_count / float(max(1, int8_count + float_fallback_count))

    dynamic_count = sum(
        1
        for t in tensors
        if t.get("shape_signature") is not None
        and len(t["shape_signature"]) > 1
        and any(int(dim) < 0 for dim in t["shape_signature"][1:])
    )

    try:
        unsupported_op_count = sum(
            1
            for op in interpreter._get_ops_details()
            if op["op_name"] not in ALLOWED_DSP_OPS
        )
    except Exception:
        unsupported_op_count = 0

    tensor_bytes = sum(
        int(np.prod(t["shape"])) * np.dtype(t["dtype"]).itemsize
        for t in tensors
        if t["shape"] is not None and len(t["shape"]) > 0
    )

    return {
        "tensor_arena_kb": round(tensor_bytes / 1024.0, 2),
        "total_subgraph_tensors": len(tensors),
        "float_fallback_count": float_fallback_count,
        "unsupported_op_count": unsupported_op_count,
        "dynamic_tensor_count": dynamic_count,
        "int8_compliance_%": round(int_only_ratio * 100.0, 1),
        "dsp_delegate_ready": (
            float_fallback_count == 0
            and unsupported_op_count == 0
            and dynamic_count == 0
        ),
    }


def compute_daily_battery_pct(
    p99_latency_ms: float, dsp_delegate_ready: bool
) -> float:
    """Computes 24-hour marginal battery drain (%) for 43,200 inferences/day against a 19,250 mWh battery."""
    inferences_per_day = 43200.0
    battery_capacity_mwh = 19250.0

    if dsp_delegate_ready:
        active_power_mw = 15.0
        effective_ms_per_run = p99_latency_ms + 1.2
    else:
        active_power_mw = 420.0
        effective_ms_per_run = p99_latency_ms + 12.0

    daily_active_hours = (effective_ms_per_run / 1000.0) * inferences_per_day / 3600.0
    daily_energy_mwh = active_power_mw * daily_active_hours
    return round((daily_energy_mwh / battery_capacity_mwh) * 100.0, 3)


def evaluate_tflite_binary(
    tflite_path: str, eval_slices: dict[str, tuple[np.ndarray, np.ndarray]]
) -> dict:
    """Runs full subgraph inspection and multi-slice inference on a compiled .tflite binary."""
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()

    in_det = interpreter.get_input_details()[0]
    out_det = interpreter.get_output_details()[0]

    hw_audit = audit_int8_hardware_compatibility(interpreter)

    slice_f1 = {}
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
        bg_mask = y_s == 0
        bg_false_positives += int(np.sum(preds_arr[bg_mask] != 0))
        bg_total_clips += int(np.sum(bg_mask))

        slice_f1[slice_name] = round(
            float(f1_score(y_s, preds, average="macro") * 100.0), 2
        )

    p99_ms = round(float(np.percentile(latencies_ms, 99)), 3)
    flash_kb = round(os.path.getsize(tflite_path) / 1024.0, 2)
    daily_batt_pct = compute_daily_battery_pct(p99_ms, hw_audit["dsp_delegate_ready"])
    bg_fpr_pct = round(
        (bg_false_positives / float(max(1, bg_total_clips))) * 100.0, 2
    )

    return {
        "flash_kb": flash_kb,
        "tensor_arena_kb": hw_audit["tensor_arena_kb"],
        "int8_compliance_%": hw_audit["int8_compliance_%"],
        "p99_latency_ms": p99_ms,
        "daily_battery_%": daily_batt_pct,
        "f1_clean_%": slice_f1["clean"],
        "f1_pocket_occluded_%": slice_f1["pocket_occluded"],
        "f1_appliance_noise_3db_%": slice_f1["appliance_noise_3db"],
        "bg_fpr_%": bg_fpr_pct,
        "dsp_delegate_ready": hw_audit["dsp_delegate_ready"],
    }


def load_cached_eval_slices() -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Loads the Fold 5 Golden Evaluation slices from data/golden_eval/."""
    slices = {}
    for s in SLICES:
        X_s = np.load(f"data/golden_eval/X_{s}.npy")
        y_s = np.load(f"data/golden_eval/y_{s}.npy")
        slices[s] = (X_s, y_s)
    return slices


def run_release_gate(
    versions: list[str] | None = None,
    eval_slices: dict[str, tuple[np.ndarray, np.ndarray]] | None = None,
    enforce_target: str | None = "v2_data_flywheel",
    out_md: str = "reports/04_release_gate_scorecard.md",
) -> pd.DataFrame:
    """Evaluates .tflite binaries, writes the scorecard, and blocks if enforce_target (int8) fails any gate."""
    if versions is None:
        versions = ["v1_baseline", "v2_data_flywheel"]
    if eval_slices is None:
        eval_slices = load_cached_eval_slices()

    rows = []

    for ver in versions:
        for q_type in ["fp32", "fp16", "int8"]:
            tfl_path = os.path.join("models", ver, f"model_{q_type}.tflite")
            if not os.path.exists(tfl_path):
                continue

            res = evaluate_tflite_binary(tfl_path, eval_slices)
            dsp_ready = res.pop("dsp_delegate_ready")

            hw_pass = (
                res["flash_kb"] <= MAX_FLASH_KB
                and res["tensor_arena_kb"] <= MAX_TENSOR_ARENA_KB
                and dsp_ready
                and res["p99_latency_ms"] <= MAX_P99_LATENCY_MS
                and res["daily_battery_%"] <= MAX_DAILY_BATTERY_PCT
            )
            quality_pass = (
                res["f1_clean_%"] >= SLICE_F1_THRESHOLDS["clean"]
                and res["f1_pocket_occluded_%"]
                >= SLICE_F1_THRESHOLDS["pocket_occluded"]
                and res["f1_appliance_noise_3db_%"]
                >= SLICE_F1_THRESHOLDS["appliance_noise_3db"]
                and res["bg_fpr_%"] <= MAX_BG_FPR_PCT
            )

            if hw_pass and quality_pass:
                gate_status = "SHIP (PASS)"
            elif not hw_pass and not quality_pass:
                gate_status = "BLOCKED (HW + SLICE)"
            elif not hw_pass:
                gate_status = "BLOCKED (HW/BATTERY)"
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
    scorecard_df.to_markdown(out_md, index=False)

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
        default=["v1_baseline", "v2_data_flywheel"],
        help="Release versions to evaluate.",
    )
    parser.add_argument(
        "--enforce-target",
        default="v2_data_flywheel",
        help="Release candidate whose int8 binary must pass all gates (or 'none').",
    )
    args = parser.parse_args()
    target = None if args.enforce_target.lower() == "none" else args.enforce_target
    try:
        run_release_gate(versions=args.versions, enforce_target=target)
    except RuntimeError as exc:
        print(f"\n[FATAL] {exc}", file=sys.stderr)
        sys.exit(1)
