# Copyright (c) 2026 xangma
# SPDX-License-Identifier: MIT
"""CPU proofs for workflow metadata, detailed host intervals and trace isolation."""

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sqlite3
import zlib

import pytest


ROOT = Path(__file__).resolve().parents[1]


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


workflow = module("extract_workflow", ROOT / "benchmarks/extract_workflow.py")
old_tests = module("workflow_timeline_fixtures", ROOT / "tests/test_extract_timeline.py")


@pytest.fixture(params=("compress", "decompress"))
def capture(request, tmp_path):
    cap = old_tests.capture.__wrapped__(tmp_path)
    raw, operation = cap.telemetry, request.param
    (cap.root / "benchmarks/profile_workflow.py").write_text("# workflow harness fixture\n")
    (cap.root / "benchmarks/profile_resident.py").write_text("# immutable resident fixture\n")
    raw["harness_sha256"] = workflow.digest(cap.root / "benchmarks/profile_workflow.py")
    raw["dependencies_sha256"] = {name: workflow.digest(cap.root / name) for name in workflow.DEPENDENCY_PATHS}
    raw["arguments"].update(operation=operation, chunk_bytes=256, device=0, telemetry="process-tree")
    raw["operation"] = operation
    raw["metadata_layout"] = ["encoded_length", "status"] if operation == "compress" else ["status", "reserved_zero"]
    size = raw["arguments"]["size"]
    payload, encoded = bytes(size), zlib.compress(bytes(size), 6)
    capacity = size + 5 + 6 if operation == "compress" else size
    uploaded = payload if operation == "compress" else encoded
    raw["fixture"] = {"workload": "zeros", "output_bytes": size, "input_bytes": len(uploaded), "seed": 0,
        "chunk_bytes": 256, "output_buffer_bytes": capacity, "zlib_runtime": zlib.ZLIB_RUNTIME_VERSION,
        "raw_sha256": hashlib.sha256(payload).hexdigest(), "input_sha256": hashlib.sha256(uploaded).hexdigest(),
        "operation_input_sha256": hashlib.sha256(uploaded).hexdigest(),
        "stdlib_stream_sha256": hashlib.sha256(encoded).hexdigest() if operation == "decompress" else None,
        "stdlib_stream_bytes": len(encoded) if operation == "decompress" else None,
        "stream_sha256": hashlib.sha256(encoded).hexdigest() if operation == "decompress" else None,
        "encoded_bytes": len(encoded) if operation == "decompress" else None,
        "stdlib_level": 6 if operation == "decompress" else None}
    records = raw["worker"]["result"]["iterations"]
    for row in records:
        row.update(operation=operation, status=[len(encoded), 0] if operation == "compress" else [0, 0],
                   status_code=0, encoded_length=len(encoded), output_bytes=size, downloaded_bytes=capacity,
                   host_bytes_bytes=len(encoded) if operation == "compress" else size,
                   validation_outside_cpu_codec_guard=True,
                   validation_oracle="stdlib.zlib.decompress" if operation == "compress" else "byte_compare",
                   cpu_codec_forbidden_phases=["upload", "codec", "metadata_download", "status_check", "output_download", "host_bytes"])
    leaves = []
    for phase in raw["phase_intervals"]:
        phase["operation"] = operation
        if phase["name"] == "decode":
            phase["name"] = "codec"
        if phase["name"] == "download_check":
            step = (phase["end_ns"] - phase["start_ns"]) // 12
            for i, name in enumerate(("metadata_download", "status_check", "output_download", "host_bytes", "validate")):
                leaf = {**phase, "id": phase["id"] + ":" + name, "name": name, "parent_id": phase["id"],
                        "start_ns": phase["start_ns"] + (2 * i + 1) * step,
                        "end_ns": phase["start_ns"] + (2 * i + 2) * step}
                leaf["nvtx_label"] = "workflow:" + leaf["id"]
                leaf["nvtx_start_bracket_ns"] = [leaf["start_ns"] - 10, leaf["start_ns"] + 10]
                leaf["nvtx_end_bracket_ns"] = [leaf["end_ns"] - 10, leaf["end_ns"] + 10]
                leaves.append(leaf)
    raw["phase_intervals"].extend(leaves)
    with sqlite3.connect(cap.paths["sqlite"]) as db:
        for leaf in leaves:
            db.execute("INSERT INTO NVTX_EVENTS VALUES(?,?,?,?,?,?,?)", (old_tests.NSIGHT + leaf["start_ns"] - old_tests.MONO,
                old_tests.NSIGHT + leaf["end_ns"] - old_tests.MONO, 59, None, leaf["nvtx_label"], old_tests.global_pid(123) + 123, None))
        if operation == "compress":
            db.execute("UPDATE StringIds SET value='encode_chunks' WHERE id=1")
            db.execute("UPDATE StringIds SET value='CompressionPrefix' WHERE id=2")
    raw["worker"]["result"] = {**{key: copy.deepcopy(raw[key]) for key in
        ("source_revision", "harness_sha256", "dependencies_sha256", "source_sha256", "native_build", "environment", "fixture")},
        "complete": True, "pid": 123, "iterations": records}
    command = ["nsys", "profile", "--trace=cuda,nvtx", "--cuda-graph-trace=node", "python", "benchmarks/profile_workflow.py"]
    for name in ("operation", "size", "seed", "iterations", "warmups", "device", "chunk_bytes", "telemetry"):
        command.extend(["--" + name.replace("_", "-"), str(raw["arguments"][name])])
    command.extend(["--source-revision", raw["source_revision"]])
    cap.paths["command"].write_text(json.dumps(command))
    cap.refresh()
    cap.run = lambda: workflow.extract(cap.paths["telemetry"], cap.paths["sqlite"], cap.receipt, cap.root)
    return cap


def test_source_bound_profile_supports_both_operations_without_global_mutation(capture):
    old_mapping = dict(workflow.helpers.KERNEL_STAGES)
    before = workflow.digest(capture.paths["sqlite"])
    report = capture.run()
    assert report["profiled"] is True and report["fixture_oracle_verified"] is True
    assert report["operation"] == capture.telemetry["operation"]
    assert len(report["iteration_stats"]) == 3 and len(report["phase_stats"]) == 28
    assert report["phase_summary"]["measured"]["metadata_download"]["count"] == 2
    assert len(report["gpu_activities"]) == 7
    if report["operation"] == "compress":
        assert {row["stage"] for row in report["gpu_activities"] if row["kind"] == "kernel"} == {"encoding", "compression_prefix"}
    assert workflow.helpers.KERNEL_STAGES == old_mapping
    assert workflow.digest(capture.paths["sqlite"]) == before
    assert set(report["extractor_dependencies_sha256"]) == {"benchmarks/extract_timeline.py", "benchmarks/extract_nsight.py"}


def test_controls_accept_empty_samples_and_null_nvtx_without_cuda_zeros(capture):
    raw = capture.telemetry
    raw["samples"], raw["clock_anchor"] = [], None
    raw["arguments"]["telemetry"] = "none"
    for phase in raw["phase_intervals"]:
        phase["nvtx_label"] = None
        phase["nvtx_start_bracket_ns"] = phase["nvtx_end_bracket_ns"] = [None, None]
    capture.refresh()
    report = workflow.extract(capture.paths["telemetry"], root=capture.root)
    assert report["kind"] == "workflow_control" and report["profiled"] is False
    assert report["samples"] == report["bins"] == []
    assert report["time_origin"]["uncertainty_ns"] is None
    assert report["iteration_stats"][0]["observed_gpu_overlap_ns"] is None
    assert report["phase_stats"][0]["observed_cuda_api_overlap_ns"] is None
    assert report["phase_intervals"][0]["nvtx_start_bracket_ns"] == [None, None]


@pytest.mark.parametrize("field,value", [("status_code", False), ("byte_exact", False),
                                        ("validation_outside_cpu_codec_guard", False), ("downloaded_bytes", 1)])
def test_iteration_error_or_extent_cannot_become_complete(capture, field, value):
    capture.telemetry["worker"]["result"]["iterations"][0][field] = value
    capture.refresh()
    with pytest.raises(ValueError):
        capture.run()


def test_status_must_precede_output_download(capture):
    leaves = [row for row in capture.telemetry["phase_intervals"] if row.get("iteration_id") == "measured-000"]
    status = next(row for row in leaves if row["name"] == "status_check")
    output = next(row for row in leaves if row["name"] == "output_download")
    status["start_ns"], output["start_ns"] = output["start_ns"], status["start_ns"]
    status["end_ns"], output["end_ns"] = output["end_ns"], status["end_ns"]
    capture.refresh()
    with pytest.raises(ValueError, match="ordering"):
        capture.run()


def test_immutable_harness_dependency_hashes_are_checked(capture):
    (capture.root / "benchmarks/profile_timeline.py").write_text("changed published helper")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        capture.run()


def test_annotation_exclusive_time_and_gaps_are_unions_not_cpu_claims():
    phases = [
        {"id": "root", "name": "download_check", "parent_id": None, "operation": "compress", "iteration_id": "measured-000",
         "iteration": 0, "warmup": False, "start_ns": 0, "end_ns": 100},
        {"id": "a", "name": "output_download", "parent_id": "root", "operation": "compress", "iteration_id": "measured-000",
         "iteration": 0, "warmup": False, "start_ns": 10, "end_ns": 40},
        {"id": "b", "name": "host_bytes", "parent_id": "root", "operation": "compress", "iteration_id": "measured-000",
         "iteration": 0, "warmup": False, "start_ns": 30, "end_ns": 60}]
    stats = workflow.host_statistics(phases, {"start_ns": -10, "end_ns": 110})
    parent = stats["phase_stats"][0]
    assert parent["children_union_ns"] == 50 and parent["exclusive_annotation_ns"] == 50
    assert parent["observed_gpu_overlap_ns"] is None
    assert [row["duration_ns"] for row in stats["whole_process_unannotated_intervals"]] == [10, 10]
    assert workflow.complement(0, 100, [(10, 40), (30, 60)]) == [
        {"start_ns": 0, "end_ns": 10, "duration_ns": 10}, {"start_ns": 60, "end_ns": 100, "duration_ns": 40}]


def test_unknown_compression_kernel_still_fails_closed():
    with pytest.raises(ValueError, match="unmapped CUDA kernel"):
        workflow.kernel_stage("encode_chunks_unreviewed")
