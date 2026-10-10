# Copyright (c) 2026 xangma
# SPDX-License-Identifier: MIT
"""Portable publication markers and normalized timeline validation."""

import copy
import hashlib
import json

import pytest

from test_verify_workflow_figures import verifier

from benchmarks.export_public_report import export_report


def marker():
    return {"schema_version": 1, "kind": "portable_measurement",
            "private_evidence_sha256": hashlib.sha256(b"private evidence").hexdigest()}


def report():
    return {
        "publication": marker(),
        "artifacts": {},
        "artifact_sha256": {},
        "source_revision": "a" * 40,
        "timestamp_unit": "ns",
        "time_origin": {"uncertainty_ns": 10},
        "views": {"whole_process": {"start_ns": 0, "end_ns": 100},
                  "measured_loop": {"start_ns": 10, "end_ns": 90}},
        "phase_semantics": {"decode": {"kind": "completed"}},
        "phase_intervals": [{"id": 1, "name": "decode", "start_ns": 10, "end_ns": 90,
                             "parent_id": None}],
        "stages": ["decode"],
        "gpu_activities": [{"kind": "kernel", "stage": "decode", "start_ns": 20,
                            "end_ns": 40, "duration_ns": 20}],
        "bins": [{"start_ns": 0, "end_ns": 50, "kernel_union_ns": 20,
                  "kernel_union_fraction": 0.4,
                  "copy_bytes_completed": {"H2D": 0, "D2H": 0, "D2D": 0, "other": 0}}],
        "samples": [{"timestamp_ns": 50, "query_start_ns": 49, "query_end_ns": 51,
                     "cpu_percent": 1.0, "rss_bytes": 2, "gpu": None,
                     "metric_timestamps_ns": {"cpu_percent": 50, "rss_bytes": 50,
                                              "gpu_memory": None, "gpu_owned_memory": None}}],
        "warnings": [],
    }


def test_portable_marker_is_bound_and_raw_artifacts_are_absent():
    value = report()
    assert verifier.portable_publication(value, marker())
    with pytest.raises(ValueError, match="marker differs"):
        verifier.portable_publication(value, {**marker(), "private_evidence_sha256": "b" * 64})
    value["artifact_sha256"] = {"capture.nsys-rep": "c" * 64}
    with pytest.raises(ValueError, match="raw artifact"):
        verifier.portable_publication(value, marker())


@pytest.mark.parametrize("private", [
    {"pid": 123456789}, {"device": {"uuid": "GPU-private"}},
    {"command": ["/home/private/run"]}, {"path": "/dev/shm/private/capture"},
])
def test_portable_marker_rejects_private_operational_data(private):
    value = report()
    value["private"] = private
    with pytest.raises(ValueError, match="private operational|command receipts|redact device UUIDs|anonymized process"):
        verifier.portable_publication(value, marker())


def test_portable_timeline_validates_normalized_activity_and_sample_data():
    verifier.verify_portable_timeline(report())
    invalid = copy.deepcopy(report())
    invalid["bins"][0]["kernel_union_ns"] = 19
    with pytest.raises(ValueError, match="kernel bin"):
        verifier.verify_portable_timeline(invalid)


def test_missing_marker_cannot_be_declared_as_portable():
    with pytest.raises(ValueError, match="no matching report"):
        verifier.portable_publication({"artifacts": {}, "artifact_sha256": {}}, marker())


def test_public_export_anonymizes_nested_process_records_and_preserves_ownership_flags():
    raw = {
        "complete": True,
        "captured_processes": [
            {"globalPid": 123456789, "name": "python", "pid": 54321},
            {"globalPid": 123456790, "name": "nvcc", "pid": 54322},
        ],
        "worker": {
            "pid": 54321,
            "process_ids": [54321, 54322],
            "process_identities": [
                {"create_time": 1234.5, "first_seen_ns": 10, "pid": 54321},
                {"create_time": 1235.5, "first_seen_ns": 20, "pid": 54322},
            ],
        },
        "gpu": {"owned_pid": True},
        "queries": [{"stdout": "54321, GPU-private, 258", "command": ["nvidia-smi"]}],
        "hostname": "private-host",
        "note": "/home/xangma/capture on roni1",
    }
    exported = export_report(json.dumps(raw).encode())
    assert exported["captured_processes"] == [
        {"globalPid": 1, "name": "python", "pid": 2},
        {"globalPid": 3, "name": "nvcc", "pid": 4},
    ]
    assert exported["worker"]["process_ids"] == [2, 4]
    assert exported["worker"]["process_identities"] == [
        {"first_seen_ns": 10, "pid": 2},
        {"first_seen_ns": 20, "pid": 4},
    ]
    assert exported["gpu"]["owned_pid"] is True
    assert "queries" not in exported and "hostname" not in exported
    assert exported["note"] == "<private-path> on <capture-host>"
    verifier.portable_publication(exported, exported["publication"])
