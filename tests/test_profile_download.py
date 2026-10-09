# Copyright (c) 2026 xangma
# SPDX-License-Identifier: MIT
"""CPU checks of bounded scheduling, status-before-copy and diagnostic metrics."""

import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
import zlib

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("profile_download", ROOT / "benchmarks/profile_download.py")
download = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(download)


@pytest.mark.parametrize("depth", [1, 2])
def test_queue_fresh_results_bounded_and_next_codec_enqueued_first(depth):
    live, events = set(), []

    def launch(index):
        assert index not in live
        live.add(index)
        assert len(live) <= depth
        events.append(("launch", index))
        return index

    def consume(index):
        events.append(("consume", index))
        live.remove(index)
        return index

    rows, maximum = download.run_queue(7, launch, consume, depth)
    assert rows == list(range(7)) and maximum == depth and not live
    assert events[:depth] == [("launch", index) for index in range(depth)]
    if depth == 2:
        assert events.index(("launch", 2)) < events.index(("consume", 1))


def test_queue_error_aborts_and_drains_only_current_and_pending():
    launched, drained = [], []

    def launch(index):
        launched.append(index)
        return index

    def consume(index):
        raise RuntimeError("bad status")

    with pytest.raises(RuntimeError, match="bad status"):
        download.run_queue(9, launch, consume, 2, drained.append)
    assert launched == [0, 1] and drained == [0, 1]


def test_phase_records_failed_scope_and_nonnegative_counters():
    row = {"phases": []}
    with pytest.raises(RuntimeError):
        with download.phase(row, "output_download"):
            raise RuntimeError("failure")
    event = row["phases"][0]
    assert not event["success"] and event["end_ns"] >= event["start_ns"]
    for name in ("wall_ns", "thread_cpu_ns", "minor_faults", "major_faults"):
        assert event[name] >= 0
    assert "thread_minor_faults" in event


def test_summary_distinguishes_batch_throughput_from_queue_latency():
    rows = [{"start_ns": start, "end_ns": end, "phases": [
        {"name": "host_bytes", "wall_ns": 4, "thread_cpu_ns": 3, "minor_faults": 2, "major_faults": 0}]
    } for start, end in ((0, 12), (1, 17))]
    result = download.summary(rows, 20, 2)
    assert result["batch_wall_per_case_ns"] == 10
    assert result["case_latency_median_ns"] == 14
    assert result["phase_metrics"]["host_bytes"]["minor_faults"]["sum"] == 4


def test_source_identity_pins_both_harness_helpers():
    identity = download.source_identity("1" * 40)
    assert identity["harness_sha256"] == download.sha(ROOT / "benchmarks/profile_download.py")
    for path in ("benchmarks/profile_workflow.py", "benchmarks/profile_timeline.py", "benchmarks/benchmark.py"):
        assert identity["dependencies_sha256"][path] == download.sha(ROOT / path)


def fake_backend(monkeypatch, operation, bad_status=False):
    """Host stand-ins exercise control flow; these do not establish CUDA behavior."""
    raw = b"independent fixture" * 16
    stream = zlib.compress(raw, 6)
    states, reads, checked = [], [], []
    monkeypatch.setitem(sys.modules, "benchmark", SimpleNamespace(make_payload=lambda *args: raw))

    class Array:
        def __init__(self, data, state, kind):
            self.data, self.state, self.kind = data, state, kind
            self.shape = data.shape

        def block_until_ready(self):
            return self

        def __array__(self, dtype=None, copy=None):
            if self.kind == "metadata":
                checked[:] = [self.state]
            else:
                assert self.state["checked"], "output read before status check"
            reads.append(self.kind)
            return np.asarray(self.data, dtype=dtype)

    def codec(*args, **kwargs):
        state = {"checked": False}
        states.append(state)
        capacity = len(raw) + ((len(raw) + 255) // 256) * 5 + 6
        data = np.frombuffer(stream, np.uint8).copy() if operation == "compress" else np.frombuffer(raw, np.uint8).copy()
        if operation == "compress":
            data = np.pad(data, (0, capacity - len(data)), constant_values=0xA5)
        status = 21 if bad_status else 0
        metadata = np.array([len(stream), status] if operation == "compress" else [status, 0], dtype=np.uint32)
        return Array(data, state, "output"), Array(metadata, state, "metadata")

    def device_put(value, sharding):
        if isinstance(value, Array):
            assert value.state["checked"], "pinned copy before status check"
            reads.append("pinned_copy")
            return Array(value.data, value.state, "pinned_output")
        return SimpleNamespace(block_until_ready=lambda: None)

    def check_status(status, label):
        if status:
            raise RuntimeError("bad status")
        checked[0]["checked"] = True

    jax = SimpleNamespace(jit=lambda fn: fn, device_put=device_put,
        block_until_ready=lambda values: tuple(value.block_until_ready() for value in values),
        sharding=SimpleNamespace(SingleDeviceSharding=lambda *a, **k: "pinned"))
    cuda = SimpleNamespace(compress_zlib_padded=codec, decompress_zlib_checked=codec)
    args = SimpleNamespace(workload="float32", size=len(raw), seed=20261008, chunk_bytes=256,
                           warmups=1, iterations=3, modes=list(download.MODES))
    return args, jax, cuda, SimpleNamespace(_check_value=check_status), states, reads, stream


@pytest.mark.parametrize("operation", ["compress", "decompress"])
def test_cpu_standins_all_modes_validate_fresh_outputs_and_padded_prefix(monkeypatch, operation):
    args, jax, cuda, codec, states, reads, stream = fake_backend(monkeypatch, operation)
    result = download.operation_report(args, operation, "device", jax, np, zlib, cuda, codec)
    assert len(states) == 13 and all(state["checked"] for state in states)
    assert len({id(state) for state in states}) == 13
    for mode in result["modes"]:
        assert mode["max_live_results"] == mode["queue_depth"]
        assert len(mode["cases"]) == 3 and len(mode["warmups"]) == 1
        for row in mode["cases"] + mode["warmups"]:
            assert row["byte_exact"] and row["codec_transfer_cpu_codec_forbidden"]
            assert row["validation_outside_cpu_codec_guard"]
            assert row["host_bytes_bytes"] == (len(stream) if operation == "compress" else args.size)
            assert row["downloaded_bytes"] == result["fixture"]["output_buffer_bytes"]
    assert result["cache_probe"]["byte_exact"] and result["cache_probe"]["excluded_from_workflows"]
    assert "pinned_copy" in reads


@pytest.mark.parametrize("operation", ["compress", "decompress"])
@pytest.mark.parametrize("mode", download.MODES)
def test_cpu_standins_failed_status_never_reads_or_pins_output(monkeypatch, operation, mode):
    args, jax, cuda, codec, states, reads, _ = fake_backend(monkeypatch, operation, bad_status=True)
    args.modes = [mode]
    with pytest.raises(RuntimeError, match="bad status"):
        download.operation_report(args, operation, "device", jax, np, zlib, cuda, codec)
    assert states and reads == ["metadata"]
