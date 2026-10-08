#!/usr/bin/env python3
"""End-to-End Pipeline Runner for PixelSense-Lite (Stages 1-5)."""

import importlib
import time

ingest_mod = importlib.import_module("01_ingest_qa")
consensus_mod = importlib.import_module("02_consensus_drift")
train_mod = importlib.import_module("03_train_quantize")
gate_mod = importlib.import_module("04_release_gate")
verify_mod = importlib.import_module("05_verify_pipeline")


def main() -> None:
    t_start = time.perf_counter()
    ingest_mod.run_ingestion_qa()
    stage2_data = consensus_mod.run_consensus_and_drift()

    for version_tag in ["v1_baseline", "v2_data_flywheel"]:
        train_mod.train_and_export_version(version_tag)

    gate_mod.run_release_gate(
        versions=["v1_baseline", "v2_data_flywheel"],
        eval_slices=stage2_data["eval_slices"],
        enforce_target="v2_data_flywheel",
    )

    verify_mod.run_verification()
    print(f"[Pipeline] Complete in {time.perf_counter() - t_start:.1f}s.")


if __name__ == "__main__":
    main()

