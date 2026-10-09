# Copyright (c) 2026 xangma
# SPDX-License-Identifier: MIT
"""CPU fixtures for clocks, process/device joins and source-bound timelines."""

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sqlite3
from types import SimpleNamespace
import zlib

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("extract_timeline", ROOT / "benchmarks/extract_timeline.py")
timeline = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(timeline)
UUID = "12345678-1234-5678-9abc-def012345678"
OTHER_UUID = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
MONO, NSIGHT, MS = 1_000_000_000_000, 1000, 1_000_000


def global_pid(pid):
    return (1 << 48) + (pid << 24)


@pytest.fixture
def capture(tmp_path):
    root, directory = tmp_path / "root", tmp_path / "capture"
    package = root / "src/cuda_zlib"
    (package / "native").mkdir(parents=True)
    (root / "benchmarks").mkdir(parents=True)
    directory.mkdir()
    generated = {}
    for module, name in (("_encode_kernels.py", "encoder.cuh"), ("_decode_kernels.py", "decoder.cuh"),
                         ("_postprocess.py", "postprocess.cuh")):
        content = "// fixture " + name
        (package / module).write_text("CUDA_SOURCE = " + repr(content) + "\n")
        generated[name] = hashlib.sha256(content.encode()).hexdigest()
    for name in ("codec_ffi.cu", "batch_encode.cuh", "batch_decode.cuh"):
        (package / "native" / name).write_text("// native fixture " + name)
        generated[name] = timeline.digest(package / "native" / name)
    (root / "benchmarks/profile_timeline.py").write_text("# timeline harness fixture\n")
    (root / "benchmarks/benchmark.py").write_text(
        "def make_payload(kind, size, seed):\n    return bytes([seed & 255]) * size\n")
    identity = {"sources": generated, "architecture": "sm_89"}
    native = {"identity": identity,
              "cache_key": hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest(),
              "library_sha256": "1" * 64, "build_sha256": "2" * 64,
              "ffi_targets": [f"cuda_zlib_{kind}_sm_89" for kind in
                              ("compress", "decompress", "compress_batch", "decompress_batch")]}
    args = {"workload": "zeros", "size": 12, "seed": 0, "warmups": 1, "iterations": 2}
    raw, stream = bytes(12), zlib.compress(bytes(12), 6)
    fixture = {"workload": "zeros", "output_bytes": 12, "encoded_bytes": len(stream), "seed": 0,
               "stdlib_level": 6, "zlib_runtime": zlib.ZLIB_RUNTIME_VERSION,
               "input_sha256": hashlib.sha256(raw).hexdigest(), "stream_sha256": hashlib.sha256(stream).hexdigest()}
    phases = []

    def phase(name, begin, end, parent=None, iteration=None, warmup=None):
        item = {"id": str(len(phases)), "name": name, "parent_id": parent, "pid": 123,
                "iteration": iteration, "warmup": warmup, "success": True,
                "iteration_id": None if iteration is None else f"{'warmup' if warmup else 'measured'}-{iteration:03d}",
                "start_ns": MONO + begin * MS, "end_ns": MONO + end * MS}
        item["nvtx_label"] = "cuda_zlib:phase:" + item["id"]
        item["nvtx_start_bracket_ns"] = [item["start_ns"] - 10, item["start_ns"] + 10]
        item["nvtx_end_bracket_ns"] = [item["end_ns"] - 10, item["end_ns"] + 10]
        phases.append(item)
        return item["id"]

    phase("prepare", 0, 1)
    phase("init", 1, 2)
    warm = phase("warmup", 2, 10)
    measured = phase("measured_loop", 10, 40)
    for warmup, iteration, parent, bounds in [(True, 0, warm, (2, 3, 8, 9)),
                                             (False, 0, measured, (10, 11, 20, 21)),
                                             (False, 1, measured, (21, 22, 35, 39))]:
        for name, begin, end in zip(("upload", "decode", "download_check"), bounds, bounds[1:]):
            phase(name, begin, end, parent, iteration, warmup)
    telemetry = {"schema_version": 1, "complete": True, "source_revision": "a" * 40,
                 "harness_sha256": timeline.digest(root / "benchmarks/profile_timeline.py"),
                 "benchmark_sha256": timeline.digest(root / "benchmarks/benchmark.py"),
                 "source_sha256": {p.relative_to(package).as_posix(): timeline.digest(p)
                                   for p in package.rglob("*") if p.is_file()},
                 "native_build": native, "environment": {"python": "fixture"}, "fixture": fixture,
                 "arguments": args, "collector_pid": 99, "errors": [],
                 "selected_gpu": {"ordinal": 0, "uuid": "GPU-" + UUID, "name": "Test GPU"},
                 "clock_anchor": {"label": "cuda_zlib:origin:fixture", "monotonic_before_ns": MONO - 100,
                                  "monotonic_after_ns": MONO + 101},
                 "worker": {"pid": 123, "start_ns": MONO - 1000, "end_ns": MONO + 50 * MS,
                            "process_ids": [123, 124], "exit_code": 0},
                 "phase_intervals": phases, "samples": [],
                 "methodology": {"sample_interval_requested_ns": 10 * MS}}
    telemetry["worker"]["result"] = {key: copy.deepcopy(telemetry[key]) for key in
        ("source_revision", "harness_sha256", "benchmark_sha256", "source_sha256", "native_build", "environment", "fixture")}
    telemetry["worker"]["result"].update(complete=True, pid=123, iterations=[
        {"iteration_id": f"{'warmup' if warmup else 'measured'}-{i:03d}", "iteration": i, "warmup": warmup,
         "status": [0, 0], "byte_exact": True, "cpu_codec_forbidden": True}
        for warmup, count in ((True, 1), (False, 2)) for i in range(count)])
    for index in range(3):
        stamp = MONO + index * 10 * MS
        telemetry["samples"].append({"timestamp_ns": stamp, "query_start_ns": stamp - 10,
            "query_end_ns": stamp + 10, "processes": [
                {"pid": pid, "create_time": 1.0, "cpu_user_seconds": index * 0.005,
                 "cpu_system_seconds": 0.0, "rss_bytes": rss, "missing": False,
                 "query_start_ns": stamp - 5, "query_end_ns": stamp + 5} for pid, rss in ((123, 100), (124, 200))],
            "gpu": {"uuid": "GPU-" + UUID, "total_memory_bytes": 1000, "used_memory_bytes": 300,
                    "free_memory_bytes": 700, "gpu_utilization_percent": 25, "memory_utilization_percent": 10,
                    "owned_compute_memory_bytes": 100, "compute_processes": [
                        {"pid": 123, "owned": True, "used_memory_bytes": 100},
                        {"pid": 999, "owned": False, "used_memory_bytes": 200}],
                    "queries": {"memory": {"start_ns": stamp - 5, "end_ns": stamp + 5, "error": None}}}})
    sqlite_path = directory / "capture.sqlite"
    with sqlite3.connect(sqlite_path) as db:
        db.executescript("""
            CREATE TABLE StringIds(id INTEGER, value TEXT);
            CREATE TABLE PROCESSES(globalPid INTEGER, pid INTEGER, name TEXT);
            CREATE TABLE TARGET_INFO_CUDA_DEVICE(pid INTEGER, cudaId INTEGER, gpuId INTEGER, uuid TEXT);
            CREATE TABLE TARGET_INFO_GPU(id INTEGER, uuid TEXT, name TEXT, computeMajor INTEGER, computeMinor INTEGER, smCount INTEGER);
            CREATE TABLE NVTX_EVENTS(start INTEGER, end INTEGER, eventType INTEGER, rangeId INTEGER, text TEXT, globalTid INTEGER, textId INTEGER);
            CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL(start INTEGER, end INTEGER, globalPid INTEGER, deviceId INTEGER,
                streamId INTEGER, correlationId INTEGER, demangledName INTEGER, gridX INTEGER, gridY INTEGER, gridZ INTEGER,
                blockX INTEGER, blockY INTEGER, blockZ INTEGER, graphNodeId INTEGER, graphId INTEGER);
            CREATE TABLE CUPTI_ACTIVITY_KIND_MEMCPY(start INTEGER, end INTEGER, globalPid INTEGER, deviceId INTEGER,
                streamId INTEGER, correlationId INTEGER, bytes INTEGER, copyKind INTEGER);
            CREATE TABLE CUPTI_ACTIVITY_KIND_MEMSET(start INTEGER, end INTEGER, globalPid INTEGER, deviceId INTEGER,
                streamId INTEGER, correlationId INTEGER, bytes INTEGER, value INTEGER);
            CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME(start INTEGER, end INTEGER, globalTid INTEGER,
                correlationId INTEGER, nameId INTEGER, returnValue INTEGER);
            CREATE TABLE ENUM_CUDA_MEMCPY_OPER(id INTEGER, name TEXT, label TEXT);
            CREATE TABLE META_DATA_EXPORT(name TEXT, value TEXT);
            CREATE TABLE ENUM_DIAGNOSTIC_SEVERITY_LEVEL(id INTEGER, name TEXT);
            CREATE TABLE DIAGNOSTIC_EVENT(severity INTEGER, text TEXT);
        """)
        db.executemany("INSERT INTO StringIds VALUES(?,?)", [(1, "emit_blocks"), (2, "VerifyDecompression"),
            (3, "cudaLaunchKernel"), (4, "cudaGraphLaunch"), (5, "cudaMemcpyAsync"), (6, "cudaMemsetAsync"),
            (7, "foreign_unmapped_kernel"), (8, telemetry["clock_anchor"]["label"])])
        db.executemany("INSERT INTO PROCESSES VALUES(?,?,?)", [(global_pid(pid), pid, "python") for pid in (99, 123, 124, 999)])
        db.executemany("INSERT INTO TARGET_INFO_CUDA_DEVICE VALUES(?,?,?,?)", [(123, 0, 7, UUID), (124, 0, 7, UUID), (123, 1, 0, OTHER_UUID)])
        db.executemany("INSERT INTO TARGET_INFO_GPU VALUES(?,?,?,?,?,?)", [(0, OTHER_UUID, "Other GPU", 8, 6, 82), (7, UUID, "Test GPU", 8, 9, 128)])
        db.execute("INSERT INTO NVTX_EVENTS VALUES(?,?,?,?,?,?,?)", (NSIGHT, None, 34, None, None, global_pid(99) + 99, 8))
        for row in phases:
            db.execute("INSERT INTO NVTX_EVENTS VALUES(?,?,?,?,?,?,?)", (NSIGHT + row["start_ns"] - MONO,
                NSIGHT + row["end_ns"] - MONO, 59, None, row["nvtx_label"], global_pid(123) + 123, None))
        for pid, begin, end, correlation, name, node in [(123, 5, 15, 1, 1, None), (124, 10, 20, 2, 2, 3), (124, 12, 16, 2, 2, 4),
                                                        (999, 5, 10, 1, 7, None)]:
            db.execute("INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (NSIGHT + begin * MS, NSIGHT + end * MS, global_pid(pid), 0, 3, correlation, name, 1, 1, 1, 32, 1, 1, node, 9 if node else None))
        db.execute("INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                   (NSIGHT + 5 * MS, NSIGHT + 10 * MS, global_pid(123), 1, 3, 99, 7, 1, 1, 1, 1, 1, 1, None, None))
        for pid, begin, end, correlation, size, kind in [(123, 1, 10, 3, 11, 1), (123, 40, 50, 4, 22, 2), (124, 19, 21, 5, 33, 8)]:
            db.execute("INSERT INTO CUPTI_ACTIVITY_KIND_MEMCPY VALUES(?,?,?,?,?,?,?,?)",
                       (NSIGHT + begin * MS, NSIGHT + end * MS, global_pid(pid), 0, 3, correlation, size, kind))
        db.execute("INSERT INTO CUPTI_ACTIVITY_KIND_MEMSET VALUES(?,?,?,?,?,?,?,?)",
                   (NSIGHT + 4 * MS, NSIGHT + 5 * MS, global_pid(123), 0, 3, 6, 44, 0))
        for pid, correlation, name in [(123, 1, 3), (124, 2, 4), (123, 3, 5), (123, 4, 5), (124, 5, 5), (123, 6, 6)]:
            db.execute("INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES(?,?,?,?,?,?)",
                       (NSIGHT + 1, NSIGHT + 2, global_pid(pid) + pid, correlation, name, 0))
        db.executemany("INSERT INTO ENUM_CUDA_MEMCPY_OPER VALUES(?,?,?)", [(1, "CUDA_MEMCPY_KIND_HTOD", "Host-to-Device"),
            (2, "CUDA_MEMCPY_KIND_DTOH", "Device-to-Host"), (8, "CUDA_MEMCPY_KIND_DTOD", "Device-to-Device")])
        db.execute("INSERT INTO META_DATA_EXPORT VALUES(?,?)", ("EXPORT_PARAM_INPUT_FILE", "capture.nsys-rep"))
        db.execute("INSERT INTO ENUM_DIAGNOSTIC_SEVERITY_LEVEL VALUES(?,?)", (2, "Warning"))
        db.execute("INSERT INTO DIAGNOSTIC_EVENT VALUES(?,?)", (2, "Diagnostic warning retained"))
    files = {"sqlite": sqlite_path, "telemetry": directory / "telemetry.json", "nsys_report": directory / "capture.nsys-rep",
             "command": directory / "command.json", "log": directory / "capture.log", "export_log": directory / "export.log",
             "worker_log": directory / "worker.log"}
    files["nsys_report"].write_bytes(b"trace fixture")
    files["command"].write_text(json.dumps(["nsys", "profile", "--trace=cuda,nvtx,osrt", "--cuda-graph-trace=node",
                                           "--nvtx-domain-exclude=TSL", "python", "benchmarks/profile_timeline.py"]))
    for key in ("log", "export_log", "worker_log"):
        files[key].write_text("Warning: " + key + " retained\n")
    receipt_path = directory / "capture.json"

    def refresh():
        telemetry["worker"]["log_sha256"] = timeline.digest(files["worker_log"])
        files["telemetry"].write_text(json.dumps(telemetry))
        artifacts = {key: {"path": path.name, "sha256": timeline.digest(path),
                          **({"private": True} if key == "sqlite" else {"published_path": "benchmarks/results/timeline/captures/" + path.name})}
                     for key, path in files.items()}
        receipt_path.write_text(json.dumps({"source_revision": "a" * 40, "capture_started_utc": "2026-10-09T10:00:00+00:00",
                                           "toolchain": {"nsys": "fixture"}, "artifacts": artifacts}))

    refresh()
    return SimpleNamespace(root=root, telemetry=telemetry, paths=files, receipt=receipt_path, refresh=refresh,
        run=lambda: timeline.extract(sqlite_path, files["telemetry"], receipt_path, root))


def mutate(capture, statement, parameters=()):
    with sqlite3.connect(capture.paths["sqlite"]) as db:
        db.execute(statement, parameters)
    capture.refresh()


def test_complete_source_bound_report_preserves_intervals_and_clock(capture):
    before = {key: timeline.digest(path) for key, path in capture.paths.items()}
    report = capture.run()
    assert report["time_origin"]["monotonic_ns"] == MONO
    assert report["time_origin"]["uncertainty_ns"] == 101
    assert report["time_origin"]["half_bracket_ns"] == 100.5
    assert report["device"]["name"] == "Test GPU"  # ordinal0 maps to physical7
    assert len(report["gpu_activities"]) == 7
    assert report["filtered_activity_counts"] == {"kernel_other_process": 1, "kernel_other_device": 1}
    assert report["views"]["measured_loop"] == {"start_ns": 10 * MS, "end_ns": 40 * MS}
    assert report["samples"][0]["cpu_percent"] is None
    assert report["samples"][1]["cpu_percent"] == pytest.approx(100)
    assert report["samples"][1]["rss_bytes"] == 300
    assert report["samples"][1]["gpu"]["used_memory_bytes"] == 300
    assert report["samples"][1]["gpu"]["queries"]["memory"]["start_ns"] == 10 * MS - 5
    assert report["phase_semantics"]["upload"]["kind"] == "enqueue"
    assert report["fixture_oracle_verified"] is True
    assert report["artifacts"]["sqlite"]["private"] is True
    assert not any(name.endswith(".sqlite") for name in report["artifact_sha256"])
    assert report["artifacts"]["capture_receipt"]["path"] in report["artifact_sha256"]
    assert {warning["source"] for warning in report["warnings"]} == {"sqlite", "log", "export_log", "worker_log"}
    assert {key: timeline.digest(path) for key, path in capture.paths.items()} == before


def test_graph_nodes_share_host_launch_without_losing_activity(capture):
    report = capture.run()
    nodes = [row for row in report["gpu_activities"] if row.get("graph_node_id") is not None]
    assert {row["graph_node_id"] for row in nodes} == {3, 4}
    assert nodes[0]["host_launch"] == nodes[1]["host_launch"]
    assert all("phase_id" not in row for row in nodes)


@pytest.mark.parametrize("statement,parameters,message", [
    ("UPDATE StringIds SET value='unknown_kernel' WHERE id=1", (), "unmapped CUDA kernel"),
    ("DELETE FROM NVTX_EVENTS WHERE eventType=34", (), "NVTX origin"),
    ("INSERT INTO NVTX_EVENTS SELECT * FROM NVTX_EVENTS WHERE eventType=34", (), "NVTX origin"),
    ("UPDATE NVTX_EVENTS SET globalTid=? WHERE eventType=34", (global_pid(123) + 123,), "another supervisor"),
    ("UPDATE TARGET_INFO_CUDA_DEVICE SET gpuId=0,uuid=? WHERE pid=123 AND cudaId=0", (OTHER_UUID,), "another GPU"),
    ("UPDATE NVTX_EVENTS SET start=start+1000 WHERE text='cuda_zlib:phase:0'", (), "call bracket"),
    ("DELETE FROM CUPTI_ACTIVITY_KIND_RUNTIME WHERE correlationId=1", (), "host correlation"),
    ("UPDATE CUPTI_ACTIVITY_KIND_KERNEL SET end=start WHERE correlationId=1 AND globalPid=?", (global_pid(123),), "nonpositive"),
])
def test_trace_rejections_are_explicit(capture, statement, parameters, message):
    mutate(capture, statement, parameters)
    with pytest.raises(ValueError, match=message):
        capture.run()


@pytest.mark.parametrize("key,value", [("status", [6, 0]), ("status", [False, 0]),
                                       ("byte_exact", False), ("cpu_codec_forbidden", False)])
def test_failed_or_forged_workflow_not_complete(capture, key, value):
    capture.telemetry["worker"]["result"]["iterations"][0][key] = value
    capture.refresh()
    with pytest.raises(ValueError, match="workflow status or byte oracle"):
        capture.run()


def test_missing_iteration_and_changed_fixture_are_rejected(capture):
    capture.telemetry["worker"]["result"]["iterations"].pop()
    capture.refresh()
    with pytest.raises(ValueError, match="missing/extra"):
        capture.run()


def test_source_and_artifact_hashes_fail_closed(capture):
    capture.paths["log"].write_text("changed log")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        capture.run()
    capture.refresh()
    (capture.root / "src/cuda_zlib/_decode_kernels.py").write_text("CUDA_SOURCE='changed'\n")
    with pytest.raises(ValueError, match="stale runtime"):
        capture.run()


def test_seeded_payload_regeneration_is_authoritative(capture):
    for source in (capture.telemetry, capture.telemetry["worker"]["result"]):
        source["fixture"]["input_sha256"] = "f" * 64
    capture.refresh()
    with pytest.raises(ValueError, match="seeded payload"):
        capture.run()


def test_missing_samples_and_pid_lifetimes_preserve_gaps(capture):
    raw = copy.deepcopy(capture.telemetry)
    raw["samples"][1]["processes"][0]["missing"] = True
    raw["samples"][1]["gpu"] = None
    samples = timeline.normalize_samples(raw, MONO)
    assert samples[1]["cpu_percent"] is None and samples[1]["rss_bytes"] is None and samples[1]["gpu"] is None
    assert samples[2]["cpu_percent"] is None
    raw = copy.deepcopy(capture.telemetry)
    raw["samples"][1]["processes"][0]["create_time"] = 2
    assert timeline.normalize_samples(raw, MONO)[1]["cpu_percent"] is None


def test_unknown_process_ownership_does_not_become_zero(capture):
    raw = copy.deepcopy(capture.telemetry)
    raw["samples"][0]["process_tree_query"] = {"error": {"type": "AccessDenied"}}
    raw["samples"][0]["gpu"]["owned_compute_memory_bytes"] = 0
    assert timeline.normalize_samples(raw, MONO)[0]["gpu"]["owned_compute_memory_bytes"] is None


def test_metric_clocks_and_cpu_elapsed_use_actual_query_brackets(capture):
    raw = copy.deepcopy(capture.telemetry)
    for index, sample in enumerate(raw["samples"]):
        stamp = sample["timestamp_ns"]
        sample["query_start_ns"], sample["query_end_ns"] = stamp - 10, stamp + 1000
        for process in sample["processes"]:
            process["query_start_ns"], process["query_end_ns"] = stamp + index * 100 + 90, stamp + index * 100 + 110
        sample["gpu"]["queries"] = {"memory": {"start_ns": stamp + 500, "end_ns": stamp + 600},
                                   "compute_processes": {"start_ns": stamp + 700, "end_ns": stamp + 900}}
    samples = timeline.normalize_samples(raw, MONO)
    row = samples[1]
    assert row["timestamp_ns"] == 10 * MS
    assert row["metric_timestamps_ns"]["rss_bytes"] == 10 * MS + 200
    assert row["metric_timestamps_ns"]["cpu_percent"] == 5 * MS + 150
    assert row["cpu_interval_ns"] == {"start_ns": 100, "end_ns": 10 * MS + 200}
    assert row["cpu_percent"] == pytest.approx(0.01 * 1e11 / (10 * MS + 100))
    assert row["metric_timestamps_ns"]["gpu_memory"] == 10 * MS + 550
    assert row["metric_timestamps_ns"]["gpu_owned_memory"] == 10 * MS + 800
    assert row["metric_timestamps_ns"]["gpu_utilization"] is None


def test_union_bins_use_actual_width_and_copy_end_boundaries():
    activities = [{"kind": "kernel", "start_ns": 0, "end_ns": 15},
                  {"kind": "kernel", "start_ns": 5, "end_ns": 20},
                  {"kind": "memcpy", "start_ns": 1, "end_ns": 10, "direction": "H2D", "bytes": 11},
                  {"kind": "memcpy", "start_ns": 20, "end_ns": 25, "direction": "D2H", "bytes": 22}]
    bins = timeline.activity_bins(activities, {"start_ns": 0, "end_ns": 25}, 10)
    assert [row["kernel_union_ns"] for row in bins] == [10, 10, 0]
    assert [row["kernel_union_fraction"] for row in bins] == [1, 1, 0]
    assert bins[0]["copy_bytes_completed"]["H2D"] == 0
    assert bins[1]["copy_bytes_completed"]["H2D"] == 11
    assert bins[2]["end_ns"] - bins[2]["start_ns"] == 5
    assert bins[2]["copy_bytes_completed"]["D2H"] == 22


def test_nested_phase_cycles_rejected_even_with_equal_bounds(capture):
    raw = copy.deepcopy(capture.telemetry)
    first, second = raw["phase_intervals"][:2]
    second["start_ns"], second["end_ns"] = first["start_ns"], first["end_ns"]
    first["parent_id"], second["parent_id"] = second["id"], first["id"]
    with pytest.raises(ValueError, match="cyclic"):
        timeline.normalize_phases(raw, MONO)
