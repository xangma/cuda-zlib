# Copyright (c) 2026 xangma
# SPDX-License-Identifier: MIT
"""CPU-only methodology checks; no CUDA behavior is simulated as GPU proof."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import zlib

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("host_outputs", ROOT / "benchmarks/host_outputs.py")
host = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(host)


def test_schedule_balances_all_six_series_and_reproduces_interleaving():
    first = host.schedule(2, 12, 20261009)
    assert first == host.schedule(2, 12, 20261009)
    assert first != host.schedule(2, 12, 20261010)
    labels = {f"{backend}_{contract}" for backend, contract in host.SERIES}
    assert len(first) == 84
    for warmup, count in ((True, 2), (False, 12)):
        for iteration in range(count):
            rows = [row for row in first if row["warmup"] == warmup and row["round"] == iteration]
            assert len(rows) == 6 and {row["series"] for row in rows} == labels


@pytest.mark.parametrize("contract", host.CONTRACTS)
def test_cpu_adapters_decode_fresh_bytes_and_share_storage(contract):
    raw = b"fresh independent bytes" * 20
    stream = zlib.compress(raw)
    results = []

    def decode(value):
        assert value is stream
        data = bytes(bytearray(zlib.decompress(value)))
        results.append(data)
        return data

    call = host.consumer_call("cpu", contract, stream, len(raw), 7, None, np, SimpleNamespace(decompress=decode))
    outputs = [call(), call()]
    assert len(results) == 2 and results[0] is not results[1]
    for output, storage in zip(outputs, results):
        assert (output.base if contract == "ndarray" else output.obj if contract == "memoryview" else output) is storage
        proof = host.validate_output(output, contract, raw, np.frombuffer(raw, np.uint8), np)
        assert proof["byte_exact"] and proof["readonly"]


@pytest.mark.parametrize("contract", ("ndarray", "memoryview"))
def test_array_view_validation_never_calls_tobytes(contract):
    class NoBytes(np.ndarray):
        def tobytes(self, *args, **kwargs):
            raise AssertionError("oracle allocated bytes")

    raw = b"bounded buffer oracle" * 100
    array = np.frombuffer(raw, np.uint8).view(NoBytes)
    output = array if contract == "ndarray" else memoryview(array)
    assert host.validate_output(output, contract, raw, np.frombuffer(raw, np.uint8), np)["byte_exact"]


@pytest.mark.parametrize("contract,make", [
    ("ndarray", lambda: np.zeros(4, np.uint8)),  # Writable output.
    ("ndarray", lambda: np.frombuffer(b"1234", np.uint8).reshape(2, 2)),
    ("memoryview", lambda: memoryview(bytearray(b"1234"))),
    ("memoryview", lambda: memoryview(b"1234").cast("I")),
    ("bytes", lambda: b"1235"),
])
def test_contract_or_byte_mismatches_are_rejected(contract, make):
    with pytest.raises(ValueError):
        host.validate_output(make(), contract, b"1234", np.frombuffer(b"1234", np.uint8), np)


def test_interleaved_case_uses_host_input_fresh_calls_guard_and_post_timer_oracle(monkeypatch):
    raw = bytes(range(256)) * 2
    args = SimpleNamespace(workload="float32", seed=20261008, order_seed=20261009, samples=3, warmups=2)
    arrays, cpu_calls = [], []
    original_decode = zlib.decompress
    timing = [False]

    def cpu_decode(stream):
        cpu_calls.append(stream)
        return original_decode(stream)

    monkeypatch.setattr(zlib, "decompress", cpu_decode)

    def api(stream, size, device):
        assert type(stream) is bytes and size == len(raw) and device == 7 and timing[0]
        with pytest.raises(AssertionError, match="codec/transfer"):
            zlib.decompress(stream)
        array = np.frombuffer(bytes(bytearray(raw)), np.uint8)
        arrays.append(array)
        return array

    original_measure, original_validate = host.measure, host.validate_output

    def measure(call, row):
        timing[0] = True
        try:
            return original_measure(call, row)
        finally:
            timing[0] = False

    def validate(*args):
        assert not timing[0], "oracle was inside timer"
        return original_validate(*args)

    monkeypatch.setattr(host, "measure", measure)
    monkeypatch.setattr(host, "validate_output", validate)
    cases = []
    case = host.collect_case(args, len(raw), 7, api, np, zlib, lambda *args: raw, cases)
    assert cases == [case] and len(case["order"]) == 30
    assert len(arrays) == 15 and len({id(array) for array in arrays}) == 15
    assert len(cpu_calls) == 16  # One fixture oracle, then fifteen fresh CPU series calls.
    for series in case["series"]:
        assert len(series["samples"]) == 3 and len(series["warmups"]) == 2
        assert series["summary"] == host.summarize(series["samples"])
        if series["name"] == "cuda_bytes":
            assert series["backing_storage"] == "fresh Python bytes copied from JAX-owned pinned host storage"
        for row in series["samples"] + series["warmups"]:
            assert case["order"][row["order_index"]] == {
                "series": series["name"], "round": row["round"], "warmup": row["warmup"]}
            assert row["complete"] and row["byte_exact"] and row["readonly"]
            assert row["end_ns"] - row["start_ns"] == row["wall_ns"] >= 0
            assert all(row[key] >= 0 for key in ("thread_cpu_ns", "minor_faults", "major_faults"))


def test_failed_api_retains_metrics_partial_row_and_restores_cpu_guard(monkeypatch):
    raw = b"status failure fixture" * 20
    args = SimpleNamespace(workload="text", seed=1, order_seed=2, samples=1, warmups=1)
    original_decode = zlib.decompress
    monkeypatch.setattr(host, "schedule", lambda *args: [{"warmup": True, "round": 0, "series": "cuda_ndarray"}])

    def api(*args):
        raise RuntimeError("native status failure")

    cases = []
    with pytest.raises(RuntimeError, match="native status"):
        host.collect_case(args, len(raw), 0, api, np, zlib, lambda *args: raw, cases)
    row = next(series for series in cases[0]["series"] if series["name"] == "cuda_ndarray")["warmups"][0]
    assert not row["complete"] and row["wall_ns"] >= 0 and "byte_exact" not in row
    assert zlib.decompress is original_decode


def test_source_inventory_pins_harness_and_all_frozen_dependencies():
    identity = host.source_identity("1" * 40)
    assert identity["harness_sha256"] == host.sha(ROOT / "benchmarks/host_outputs.py")
    expected = {f"benchmarks/{name}.py" for name in ("profile_workflow", "profile_timeline", "benchmark", "profile_resident")}
    assert set(identity["dependencies_sha256"]) == expected
    assert all(identity["dependencies_sha256"][name] == host.sha(ROOT / name) for name in expected)
    assert "native/batch_decode.cuh" in identity["source_sha256"]


def test_loaded_package_must_be_the_recorded_source_tree(tmp_path):
    assert host.runtime_package(SimpleNamespace(__file__=ROOT / "src/cuda_zlib/__init__.py")) == ROOT / "src/cuda_zlib"
    with pytest.raises(ValueError, match="PYTHONPATH=src"):
        host.runtime_package(SimpleNamespace(__file__=tmp_path / "site-packages/cuda_zlib/__init__.py"))


def test_selected_uuid_uses_same_driver_ordinal_as_backend(monkeypatch):
    def get_device(pointer, ordinal):
        assert ordinal == 3
        pointer._obj.value = 17
        return 0

    def get_uuid(pointer, handle):
        assert handle == 17
        pointer._obj.bytes[:] = bytes(range(16))
        return 0

    monkeypatch.setattr(host.ctypes, "CDLL", lambda name: SimpleNamespace(
        cuInit=lambda flags: 0, cuDeviceGet=get_device, cuDeviceGetUuid=get_uuid))
    assert host.selected_gpu_uuid(SimpleNamespace(local_hardware_id=3)) == "GPU-00010203-0405-0607-0809-0a0b0c0d0e0f"


def test_snapshot_preserves_query_errors_and_foreign_gpu_processes(monkeypatch):
    def query(command, **kwargs):
        if "--query-gpu=" in command[1]:
            raise OSError("query unavailable")
        return SimpleNamespace(returncode=0, stdout=f"{host.os.getpid()}, GPU-a, 20\n991199, GPU-a, N/A\n", stderr="")

    monkeypatch.setattr(host.subprocess, "run", query)
    result = host.gpu_snapshot()
    assert result["errors"] == ["query unavailable"] and not result["gpus"]
    assert [row["owned_pid"] for row in result["compute_processes"]] == [True, False]
    assert result["compute_processes"][1]["used_memory"] == "N/A"
    assert len(result["queries"]) == 2 and result["end_ns"] >= result["start_ns"]


def test_failure_json_and_overwrite_refusal_preserve_original_file(tmp_path, monkeypatch):
    args = SimpleNamespace(output=tmp_path / "failed.json", source_revision="1" * 40)

    def identity(*args):
        raise RuntimeError("source failure")

    monkeypatch.setattr(host, "source_identity", identity)
    assert host.run(args) == 1
    contents = args.output.read_bytes()
    report = json.loads(contents)
    assert not report["complete"] and report["error"]["message"] == "source failure"
    with pytest.raises(ValueError, match="fresh output"):
        host.run(args)
    assert args.output.read_bytes() == contents
