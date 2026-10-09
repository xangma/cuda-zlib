# Copyright (c) 2026 xangma
# SPDX-License-Identifier: MIT
"""CPU-only checks for workflow metadata, guard scope and detailed phases."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import zlib

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("profile_workflow", ROOT / "benchmarks/profile_workflow.py")
workflow = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(workflow)


def check_status(status, label):
    if status:
        raise RuntimeError(f"{label}: status {status}")


def test_compression_status_precedes_extent_use():
    with pytest.raises(RuntimeError, match="status 21"):
        workflow.checked_extent("compress", [0, 21], 4096, check_status)
    with pytest.raises(ValueError, match="encoded extent"):
        workflow.checked_extent("compress", [0, 0], 4096, check_status)
    assert workflow.checked_extent("compress", [32, 0], 4096, check_status) == (32, [32, 0])


@pytest.mark.parametrize("metadata", [[7, 0], [4097, 0], [0x100000000, 0], [-1, 0]])
def test_invalid_compression_extent_is_rejected(metadata):
    with pytest.raises(ValueError):
        workflow.checked_extent("compress", metadata, 4096, check_status)


def test_decompression_status_and_reserved_word_checked_before_consuming_output():
    with pytest.raises(RuntimeError, match="status 21"):
        workflow.checked_extent("decompress", [21, 0], 4096, check_status)
    with pytest.raises(ValueError, match="unexpected decompression"):
        workflow.checked_extent("decompress", [0, 1], 4096, check_status)
    assert workflow.checked_extent("decompress", [0, 0], 4096, check_status) == (4096, [0, 0])


@pytest.mark.parametrize("metadata", [[1], [1, 0, 0], [1.5, 0]])
def test_malformed_metadata_shape_and_type_are_rejected(metadata):
    with pytest.raises((ValueError, TypeError)):
        workflow.checked_extent("compress", metadata, 4096, check_status)


def test_cpu_codec_forbidden_only_within_guard_and_restored_after_error():
    raw = b"independent compression oracle" * 100
    stream = zlib.compress(raw)
    original = zlib.decompress
    with pytest.raises(AssertionError, match="codec/transfer"):
        with workflow.forbid_cpu_codec(zlib):
            zlib.decompress(stream)
    assert zlib.decompress is original and zlib.decompress(stream) == raw
    with workflow.forbid_cpu_codec(zlib):
        with workflow.forbid_cpu_codec(zlib):
            with pytest.raises(AssertionError):
                zlib.compress(raw)
        with pytest.raises(AssertionError):
            zlib.decompress(stream)
    assert zlib.decompress is original


def test_null_nvtx_still_records_detailed_phase_tree_and_operation(capsys):
    recorder = workflow.StageRecorder(None, "test", "compress")
    with recorder.phase("measured_loop") as parent:
        with recorder.phase("download_check", parent, 0, False) as download:
            for name in ("metadata_download", "status_check", "output_download", "host_bytes", "validate"):
                with recorder.phase(name, download, 0, False):
                    pass
    lines = capsys.readouterr().err.splitlines()
    events = [workflow.parse_stage_line(line, workflow.os.getpid(), "compress") for line in lines]
    phases = workflow.helpers.phase_intervals(events)
    assert len(phases) == 7
    assert all(p["nvtx_label"] is None and p["nvtx_start_bracket_ns"] == [None, None] for p in phases)
    leaves = [p for p in phases if p["parent_id"] == download]
    assert {p["name"] for p in leaves} == {"metadata_download", "status_check", "output_download", "host_bytes", "validate"}
    assert all(p["iteration_id"] == "measured-000" and p["operation"] == "compress" for p in leaves)


def test_stage_parser_rejects_other_pid_operation_and_unknown_name():
    event = {"pid": 17, "operation": "decompress", "event": "start", "name": "codec", "timestamp_ns": 123}
    line = workflow.EVENT_PREFIX + json.dumps(event)
    assert workflow.parse_stage_line("normal warning", 17, "decompress") is None
    for pid, operation in [(18, "decompress"), (17, "compress")]:
        with pytest.raises(ValueError, match="PID/operation"):
            workflow.parse_stage_line(line, pid, operation)
    event["name"] = "untracked"
    with pytest.raises(ValueError, match="invalid workflow"):
        workflow.parse_stage_line(workflow.EVENT_PREFIX + json.dumps(event), 17, "decompress")


def rows(operation):
    return [{"operation": operation, "iteration": 0, "warmup": warmup,
             "iteration_id": "warmup-000" if warmup else "measured-000", "status_code": 0,
             "status": [32, 0] if operation == "compress" else [0, 0], "encoded_length": 32,
             "downloaded_bytes": 4096, "host_bytes_bytes": 32 if operation == "compress" else 4096,
             "byte_exact": True, "cpu_codec_forbidden": True, "validation_outside_cpu_codec_guard": True,
             "validation_oracle": "stdlib.zlib.decompress" if operation == "compress" else "byte_compare"}
            for warmup in (True, False)]


@pytest.mark.parametrize("operation", ["compress", "decompress"])
def test_iteration_proof_requires_all_warmup_measured_status_and_oracle_rows(operation):
    args = SimpleNamespace(operation=operation, warmups=1, iterations=1, size=4096)
    valid = rows(operation)
    workflow.validate_iterations({"iterations": valid}, args)
    valid[-1]["byte_exact"] = False
    with pytest.raises(ValueError, match="validation failed"):
        workflow.validate_iterations({"iterations": valid}, args)
    with pytest.raises(ValueError, match="missing validated"):
        workflow.validate_iterations({"iterations": valid[:1]}, args)


def test_source_identity_pins_own_harness_and_immutable_dependencies():
    identity = workflow.source_identity("a" * 40)
    assert identity["harness_sha256"] == workflow.sha(ROOT / "benchmarks/profile_workflow.py")
    expected = {"benchmarks/profile_timeline.py", "benchmarks/benchmark.py", "benchmarks/profile_resident.py"}
    assert set(identity["dependencies_sha256"]) == expected
    assert all(workflow.sha(ROOT / name) == value for name, value in identity["dependencies_sha256"].items())
