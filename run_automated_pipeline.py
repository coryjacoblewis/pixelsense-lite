#!/usr/bin/env python3
"""End-to-end pipeline runner for PixelSense-Lite."""

import time

import consensus_drift
import ingest_qa
import release_gate
import test_pipeline
import train_quantize


def main() -> None:
    t_start = time.perf_counter()
    ingest_qa.run_ingestion_qa()
    stage2_data = consensus_drift.run_consensus_and_drift()

    for version_tag in ["v1_baseline", "v2_data_flywheel"]:
        train_quantize.train_and_export_version(version_tag)

    release_gate.run_release_gate(
        versions=["v1_baseline", "v2_data_flywheel"],
        eval_slices=stage2_data["eval_slices"],
        enforce_target="v2_data_flywheel",
    )

    test_pipeline.run_verification()
    print(f"[Pipeline] Complete in {time.perf_counter() - t_start:.1f}s.")


if __name__ == "__main__":
    main()
