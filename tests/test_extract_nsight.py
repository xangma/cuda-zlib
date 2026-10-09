# Copyright (c) 2026 xangma
# SPDX-License-Identifier: MIT
"""CPU-only checks of imported activity accounting and fail-closed extraction."""

import hashlib
import importlib.util
import json
from pathlib import Path
import sqlite3

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("extract_nsight", ROOT / "benchmarks/extract_nsight.py")
extractor = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(extractor)


@pytest.fixture
def trace(tmp_path):
    path = tmp_path / "case.sqlite"
    pid = 123
    global_pid = (1 << 48) + (pid << 24)
    with sqlite3.connect(path) as db:
        db.executescript("""
            CREATE TABLE StringIds(id INTEGER, value TEXT);
            CREATE TABLE PROCESSES(globalPid INTEGER, pid INTEGER, name TEXT);
            CREATE TABLE TARGET_INFO_CUDA_DEVICE(pid INTEGER, cudaId INTEGER, gpuId INTEGER, uuid TEXT);
            CREATE TABLE TARGET_INFO_GPU(id INTEGER, uuid TEXT, name TEXT,
                computeMajor INTEGER, computeMinor INTEGER, smCount INTEGER);
            CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL(start INTEGER, end INTEGER, globalPid INTEGER,
                deviceId INTEGER, streamId INTEGER, correlationId INTEGER, demangledName INTEGER,
                gridX INTEGER, gridY INTEGER, gridZ INTEGER, blockX INTEGER, blockY INTEGER,
                blockZ INTEGER, graphNodeId INTEGER, graphId INTEGER);
            CREATE TABLE CUPTI_ACTIVITY_KIND_MEMSET(start INTEGER, end INTEGER, globalPid INTEGER,
                deviceId INTEGER, streamId INTEGER, correlationId INTEGER, bytes INTEGER, value INTEGER);
            CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME(start INTEGER, end INTEGER, globalTid INTEGER,
                correlationId INTEGER, nameId INTEGER, returnValue INTEGER);
            CREATE TABLE ENUM_DIAGNOSTIC_SEVERITY_LEVEL(id INTEGER, name TEXT);
            CREATE TABLE DIAGNOSTIC_EVENT(severity INTEGER, text TEXT);
            CREATE TABLE META_DATA_EXPORT(name TEXT, value TEXT);
        """)
        db.executemany("INSERT INTO StringIds VALUES(?,?)", [
            (1, "emit_blocks"), (2, "<unnamed>::VerifyDecompression(const unsigned int *)"),
            (3, "cudaLaunchKernel_v7000"), (4, "cudaMemsetAsync_v3020"), (5, "cudaGraphLaunch_v10000")])
        db.execute("INSERT INTO PROCESSES VALUES(?,?,?)", (global_pid, pid, "python"))
        db.execute("INSERT INTO TARGET_INFO_CUDA_DEVICE VALUES(?,?,?,?)", (pid, 0, 7, "right-gpu"))
        # A CUDA ordinal is not the TARGET_INFO_GPU primary key.
        db.executemany("INSERT INTO TARGET_INFO_GPU VALUES(?,?,?,?,?,?)", [
            (0, "wrong-gpu", "Other GPU", 8, 6, 82), (7, "right-gpu", "Test GPU", 8, 9, 128)])
        db.executemany("INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", [
            (150, 200, global_pid, 0, 14, 1, 1, 2, 1, 1, 32, 1, 1, None, None),
            (250, 300, global_pid, 0, 14, 2, 2, 1, 1, 1, 1, 1, 1, None, None)])
        db.execute("INSERT INTO CUPTI_ACTIVITY_KIND_MEMSET VALUES(?,?,?,?,?,?,?,?)",
                   (100, 125, global_pid, 0, 14, 3, 4096, 0))
        db.executemany("INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES(?,?,?,?,?,?)", [
            (90, 95, global_pid + pid, 1, 3, 0),
            (96, 98, global_pid + pid, 2, 3, 0),
            (80, 85, global_pid + pid, 3, 4, 0)])
        db.executemany("INSERT INTO ENUM_DIAGNOSTIC_SEVERITY_LEVEL VALUES(?,?)",
                       [(1, "Info"), (2, "Warning"), (3, "Error")])
        db.execute("INSERT INTO DIAGNOSTIC_EVENT VALUES(?,?)",
                   (2, "Not all CUDA events might have been collected."))
        db.execute("INSERT INTO META_DATA_EXPORT VALUES(?,?)", ("EXPORT_PARAM_INPUT_FILE", "case.nsys-rep"))
    return path


def mutate(path, statement, parameters=()):
    with sqlite3.connect(path) as db:
        db.execute(statement, parameters)


def test_stage_accounting_memset_gaps_and_device_join_are_independent(trace):
    before = extractor.digest(trace)
    result = extractor.extract_sqlite(trace, "case.nsys-rep")
    assert result["device"]["uuid"] == "right-gpu"
    assert result["kernel_sum_ns"] == 100
    assert result["stage_ns"]["emission"] == result["stage_ns"]["verification"] == 50
    assert result["memory_operation_ns"] == 25
    assert result["gpu_span_ns"] == 200
    assert result["gpu_activity_union_ns"] == 125
    assert result["gap_ns"] == 75
    assert result["gpu_activities"][0]["start_ns"] == 0
    assert result["gpu_activities"][1]["host_launch"]["start_ns"] == -10
    assert result["stream_ids"] == [14]
    assert result["warnings"][0]["text"] == "Not all CUDA events might have been collected."
    assert extractor.digest(trace) == before


def test_overlapping_kernel_sums_do_not_replace_interval_union(trace):
    mutate(trace, "UPDATE CUPTI_ACTIVITY_KIND_KERNEL SET start=175,end=225 WHERE correlationId=2")
    result = extractor.extract_sqlite(trace, "case.nsys-rep")
    assert result["kernel_sum_ns"] == 100
    assert result["gpu_activity_union_ns"] == 100  # 75 kernel union plus 25 memset
    assert result["gpu_span_ns"] == 125 and result["gap_ns"] == 25


def test_graph_launch_can_correlate_multiple_observed_nodes(trace):
    mutate(trace, "UPDATE CUPTI_ACTIVITY_KIND_RUNTIME SET nameId=5 WHERE correlationId=1")
    mutate(trace, "DELETE FROM CUPTI_ACTIVITY_KIND_RUNTIME WHERE correlationId=2")
    mutate(trace, "UPDATE CUPTI_ACTIVITY_KIND_KERNEL SET correlationId=1,graphId=17,graphNodeId=gridX")
    result = extractor.extract_sqlite(trace, "case.nsys-rep")
    kernels = [row for row in result["gpu_activities"] if row["kind"] == "kernel"]
    assert len(kernels) == 2 and {row["graph_node_id"] for row in kernels} == {1, 2}
    assert all(row["host_launch"]["api"] == "cudaGraphLaunch_v10000" for row in kernels)


@pytest.mark.parametrize("statement,match", [
    ("DELETE FROM CUPTI_ACTIVITY_KIND_KERNEL", "empty imported"),
    ("UPDATE StringIds SET value='unreviewed_new_kernel' WHERE id=1", "unmapped CUDA kernel"),
    ("UPDATE CUPTI_ACTIVITY_KIND_KERNEL SET end=start WHERE correlationId=1", "nonpositive"),
    ("DELETE FROM CUPTI_ACTIVITY_KIND_RUNTIME WHERE correlationId=1", "host correlation"),
    ("UPDATE CUPTI_ACTIVITY_KIND_RUNTIME SET returnValue=1 WHERE correlationId=1", "host correlation"),
    ("UPDATE CUPTI_ACTIVITY_KIND_RUNTIME SET correlationId=99 WHERE correlationId=1", "host correlation"),
    ("UPDATE CUPTI_ACTIVITY_KIND_KERNEL SET deviceId=1 WHERE correlationId=1", "one CUDA device"),
    ("UPDATE DIAGNOSTIC_EVENT SET severity=3", "collection/import failure"),
    ("UPDATE DIAGNOSTIC_EVENT SET text='TargetProfilingFailed: importer stopped'", "collection/import failure"),
    ("UPDATE META_DATA_EXPORT SET value='different.nsys-rep'", "different Nsight report"),
])
def test_untrusted_or_incomplete_activity_is_rejected(trace, statement, match):
    mutate(trace, statement)
    with pytest.raises(ValueError, match=match):
        extractor.extract_sqlite(trace, "case.nsys-rep")


def test_extra_host_launch_without_activity_is_rejected(trace):
    mutate(trace, "INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME SELECT start,end,globalTid,99,nameId,0 "
           "FROM CUPTI_ACTIVITY_KIND_RUNTIME WHERE correlationId=1")
    with pytest.raises(ValueError, match="without imported"):
        extractor.extract_sqlite(trace, "case.nsys-rep")


@pytest.mark.parametrize("name,stage", [
    ("<unnamed>::SmallSharedDecodeSpecialized(unsigned char *)", "fused_decode"),
    ("(anonymous namespace)::SmallDecode(unsigned char *)", "fused_decode"),
    ("void cub::DeviceRadixSortOnesweepKernel<int, (bool)0>(T1 *, int)", "sorting"),
    ("describe_candidates_counted", "description"),
    ("emit_blocks_pipeline", "emission"),
    ("fixed_summaries", "description"),
])
def test_stage_mapping_uses_exact_outer_symbol(name, stage):
    assert extractor.kernel_stage(name) == stage
    with pytest.raises(ValueError, match="unmapped"):
        extractor.kernel_stage("unknown::" + name)


def test_startup_warning_does_not_hide_capture_failures():
    startup = ("Error processing line 1 of /env/pyannote_audio-nspkg.pth:\n\n"
               "  Traceback (most recent call last):\n  AttributeError: loader\n\nRemainder of file ignored")
    assert extractor.log_warnings(startup, "log")[0]["text"] == startup
    with pytest.raises(ValueError, match="failure"):
        extractor.log_warnings(startup + "\nTargetProfilingFailed", "log")
    with pytest.raises(ValueError, match="failure"):
        extractor.log_warnings("Traceback (most recent call last):\nRuntimeError: codec broke", "log")


def test_native_binding_rederives_headers_and_rejects_stale_sources(tmp_path):
    package = tmp_path / "src/cuda_zlib"
    (package / "native").mkdir(parents=True)
    (tmp_path / "benchmarks").mkdir()
    generated = {}
    for name, module in (("encoder.cuh", "_encode_kernels.py"), ("decoder.cuh", "_decode_kernels.py"),
                         ("postprocess.cuh", "_postprocess.py")):
        value = "// " + name + "\n"
        (package / module).write_text("CUDA_SOURCE = " + repr(value) + "\n")
        generated[name] = hashlib.sha256(value.encode()).hexdigest()
    for name in ("codec_ffi.cu", "batch_encode.cuh", "batch_decode.cuh"):
        (package / "native" / name).write_text("// " + name)
        generated[name] = extractor.digest(package / "native" / name)
    for name in ("benchmark.py", "profile_resident.py"):
        (tmp_path / "benchmarks" / name).write_text("# harness\n")
    inventory = {p.relative_to(package).as_posix(): extractor.digest(p)
                 for p in package.rglob("*") if p.is_file()}
    stage = {"source_sha256": inventory, "harness_sha256": extractor.digest(tmp_path / "benchmarks/profile_resident.py"),
             "benchmark_sha256": extractor.digest(tmp_path / "benchmarks/benchmark.py")}
    identity = {"sources": generated}
    native = {"source_sha256": inventory, "identity": identity,
              "cache_key": hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest(),
              "library_sha256": "a" * 64, "build_sha256": "b" * 64}
    extractor.source_binding(stage, native, tmp_path)
    generated["decoder.cuh"] = "c" * 64
    with pytest.raises(ValueError, match="different generated CUDA"):
        extractor.source_binding(stage, native, tmp_path)
    (package / "_decode_kernels.py").write_text("CUDA_SOURCE = 'changed'\n")
    with pytest.raises(ValueError, match="stale runtime"):
        extractor.source_binding(stage, native, tmp_path)


def test_hash_validation_rejects_changed_artifact(tmp_path):
    path = tmp_path / "trace.nsys-rep"
    path.write_bytes(b"original trace")
    expected = extractor.digest(path)
    path.write_bytes(b"different trace")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        extractor.check_hash(path, expected)
