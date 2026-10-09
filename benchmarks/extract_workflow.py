#!/usr/bin/env python3
# Copyright (c) 2026 xangma
# SPDX-License-Identifier: MIT
"""Extract independently source-bound compression/decompression workflow timelines.

Published timeline tools remain immutable. This reader reuses their MIT clock,
resource and interval helpers without mutating globals. Host gaps are the
complement of recorded annotations, not inferred CPU work. CUDA overlap is an
observation, not causal ownership. Default fixture regeneration needs NumPy;
normalization and telemetry-only controls otherwise use the standard library.
"""

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path, PurePosixPath
import re
import sqlite3
import statistics
import sys
from datetime import datetime
from collections import defaultdict

ROOT = Path(__file__).resolve().parents[1]
HELPER_PATH = Path(__file__).with_name("extract_timeline.py")
SPEC = importlib.util.spec_from_file_location("workflow_timeline_helpers", HELPER_PATH)
timeline = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(timeline)
helpers = timeline.helpers
require, digest, check_hash = timeline.require, timeline.digest, timeline.check_hash
integer, interval, gpu_uuid = timeline.integer, timeline.interval, timeline.gpu_uuid
origin_from_nvtx, nvtx_text = timeline.origin_from_nvtx, timeline.nvtx_text
_copy_direction = timeline._copy_direction
DEPENDENCY_PATHS = {"benchmarks/profile_timeline.py", "benchmarks/profile_resident.py", "benchmarks/benchmark.py"}
EXTRACTOR_DEPENDENCIES = {"benchmarks/extract_timeline.py": digest(HELPER_PATH),
                          "benchmarks/extract_nsight.py": digest(timeline.HELPER_PATH)}
COMPRESSION_STAGES = {"encode_chunks": "encoding", "CompressionPrefix": "compression_prefix",
                      "pack_chunks": "packing", "write_wrapper": "wrapper", "SmallFinish": "fused_encode"}
STAGES = list(helpers.STAGES) + ["encoding", "compression_prefix", "packing", "wrapper", "fused_encode"]
PHASE_SEMANTICS = {
    "prepare": {"kind": "host_only", "description": "Seeded CPU fixture preparation and stdlib oracle"},
    "init": {"kind": "host_annotation", "description": "Initialization may enqueue backend/native GPU work"},
    "warmup": {"kind": "container", "description": "Completed warmup workflows"},
    "measured_loop": {"kind": "container", "description": "Completed measured workflows"},
    "upload": {"kind": "enqueue", "description": "Device upload enqueue; following codec completion includes input dependency"},
    "codec": {"kind": "completed", "description": "Wait for output and metadata from the selected CUDA operation"},
    "download_check": {"kind": "container", "description": "Metadata/status before output transfer, host conversion and validation"},
    "metadata_download": {"kind": "completed", "description": "Download native checked metadata"},
    "status_check": {"kind": "host_only", "description": "Check the operation-specific native status before reading output"},
    "output_download": {"kind": "completed", "description": "Download the full returned device array"},
    "host_bytes": {"kind": "host_only", "description": "Convert the host array and select meaningful output bytes"},
    "validate": {"kind": "host_only", "description": "Output oracle outside the CPU-codec guard"},
}


def kernel_stage(name):
    symbol = name.replace("<unnamed>::", "").replace("(anonymous namespace)::", "")
    symbol = symbol.removeprefix("void ").split("(", 1)[0].split("<", 1)[0].strip()
    return COMPRESSION_STAGES[symbol] if symbol in COMPRESSION_STAGES else helpers.kernel_stage(name)


# The row reader follows the immutable MIT timeline reader, with a local kernel
# classifier so encoder symbols require no mutation of published helper globals.
def read_sqlite(path, telemetry, expected_report):
    worker = telemetry["worker"]
    pids = {integer(pid, "worker process ID", 1) for pid in worker["process_ids"]}
    pids.add(integer(worker["pid"], "worker PID", 1))
    selected = telemetry["selected_gpu"]
    selected_uuid = gpu_uuid(selected["uuid"])
    ordinal = integer(selected["ordinal"], "CUDA ordinal", 0)
    with sqlite3.connect(path.resolve().as_uri() + "?mode=ro&immutable=1", uri=True) as db:
        db.row_factory = sqlite3.Row
        require(db.execute("PRAGMA quick_check").fetchone()[0] == "ok", "invalid SQLite export")
        tables = {row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        require({"StringIds", "PROCESSES", "TARGET_INFO_CUDA_DEVICE", "TARGET_INFO_GPU",
                 "NVTX_EVENTS", "CUPTI_ACTIVITY_KIND_KERNEL"} <= tables, "missing Nsight tables")
        rows = lambda name: helpers.rows(db, name, tables)
        string_rows = rows("StringIds")
        strings = {row["id"]: row["value"] for row in string_rows}
        require(len(strings) == len(string_rows), "ambiguous StringIds")
        nvtx = rows("NVTX_EVENTS")
        processes = {row["globalPid"]: row for row in rows("PROCESSES")}
        origin = origin_from_nvtx(nvtx, strings, telemetry["clock_anchor"], processes,
                                  telemetry.get("collector_pid"))
        worker_processes = {key: row for key, row in processes.items() if row["pid"] in pids}
        require(worker_processes and any(row["pid"] == worker["pid"]
                                        for row in worker_processes.values()), "worker PID absent from trace")
        require(len({row["pid"] for row in worker_processes.values()}) == len(worker_processes),
                "ambiguous worker PID lifetime in trace")
        cuda_devices, gpus = rows("TARGET_INFO_CUDA_DEVICE"), rows("TARGET_INFO_GPU")
        selected_gpus = [row for row in gpus if gpu_uuid(row["uuid"]) == selected_uuid]
        require(len(selected_gpus) == 1, "selected UUID does not identify a unique physical GPU")
        gpu = selected_gpus[0]
        device = {key: gpu.get(key) for key in
                  ("name", "uuid", "computeMajor", "computeMinor", "smCount")}
        device["cuda_ordinal"] = ordinal
        mappings = {}
        for row in cuda_devices:
            if row["pid"] not in pids:
                continue
            physical = [item for item in gpus if item["id"] == row["gpuId"]
                        and gpu_uuid(item["uuid"]) == gpu_uuid(row["uuid"])]
            require(len(physical) == 1, "ambiguous CUDA ordinal to physical GPU mapping")
            key = row["pid"], row["cudaId"]
            require(key not in mappings, "duplicate CUDA device mapping")
            mappings[key] = physical[0]
        require((worker["pid"], ordinal) in mappings
                and gpu_uuid(mappings[worker["pid"], ordinal]["uuid"]) == selected_uuid,
                "requested CUDA ordinal maps to another GPU")
        api_by_id, api_intervals = defaultdict(list), []
        for table in ("CUPTI_ACTIVITY_KIND_RUNTIME", "CUPTI_ACTIVITY_KIND_DRIVER"):
            for row in rows(table):
                global_pid = row["globalTid"] & ~0xFFFFFF
                if global_pid not in worker_processes:
                    continue
                api = {"api": strings[row["nameId"]], "pid": processes[global_pid]["pid"],
                       "kind": "runtime" if table.endswith("RUNTIME") else "driver",
                       "start_ns": row["start"] - origin["nsight_ns"],
                       "end_ns": row["end"] - origin["nsight_ns"],
                       "correlation_id": row["correlationId"], "return_value": row["returnValue"]}
                interval(api, "CUDA API")
                api_by_id[global_pid, row["correlationId"]].append(api)
                api_intervals.append(api)
        copy_names = {row["id"]: row.get("label") or row.get("name")
                      for row in rows("ENUM_CUDA_MEMCPY_OPER")}
        activities, filtered = [], defaultdict(int)
        for kind, table in (("kernel", "CUPTI_ACTIVITY_KIND_KERNEL"),
                            ("memcpy", "CUPTI_ACTIVITY_KIND_MEMCPY"),
                            ("memset", "CUPTI_ACTIVITY_KIND_MEMSET")):
            for row in rows(table):
                global_pid = row["globalPid"]
                if global_pid not in worker_processes:
                    filtered[kind + "_other_process"] += 1
                    continue
                pid = processes[global_pid]["pid"]
                require((pid, row["deviceId"]) in mappings, "unmapped worker CUDA device")
                physical = mappings[pid, row["deviceId"]]
                if gpu_uuid(physical["uuid"]) != selected_uuid:
                    filtered[kind + "_other_device"] += 1
                    continue
                name = strings[row["demangledName"]] if kind == "kernel" else kind
                item = {"kind": kind, "name": name,
                        "stage": kernel_stage(name) if kind == "kernel" else None,
                        "start_ns": row["start"] - origin["nsight_ns"],
                        "end_ns": row["end"] - origin["nsight_ns"],
                        "duration_ns": row["end"] - row["start"],
                        "pid": pid, "device_uuid": physical["uuid"],
                        "stream_id": row["streamId"], "correlation_id": row["correlationId"],
                        "graph_node_id": row.get("graphNodeId"), "graph_id": row.get("graphId")}
                interval(item, "CUDA activity")
                calls = api_by_id[global_pid, row["correlationId"]]
                runtime = [call for call in calls if call["kind"] == "runtime"]
                calls = runtime or calls
                require(len(calls) == 1 and calls[0]["return_value"] == 0,
                        f"missing/ambiguous/failed host correlation for {name}")
                item["host_launch"] = calls[0]
                if kind == "kernel":
                    item.update(grid=[row["grid" + axis] for axis in "XYZ"],
                                block=[row["block" + axis] for axis in "XYZ"],
                                registers_per_thread=row.get("registersPerThread"),
                                static_shared_bytes=row.get("staticSharedMemory"),
                                dynamic_shared_bytes=row.get("dynamicSharedMemory"))
                else:
                    item["bytes"] = integer(row["bytes"], "CUDA copy/set bytes", 0)
                    item["value"] = row.get("value")
                    item["copy_kind"] = row.get("copyKind")
                    if kind == "memcpy":
                        item["direction"], item["copy_kind_name"] = _copy_direction(row["copyKind"], copy_names)
                activities.append(item)
        require(any(row["kind"] == "kernel" for row in activities), "no worker kernels on selected GPU")
        annotations = []
        for row in nvtx:
            if row.get("globalTid") is None:
                continue
            global_pid = row["globalTid"] & ~0xFFFFFF
            if global_pid in worker_processes:
                annotations.append({"name": nvtx_text(row, strings), "pid": processes[global_pid]["pid"],
                                    "start_ns": row["start"] - origin["nsight_ns"],
                                    "end_ns": None if row.get("end") is None else row["end"] - origin["nsight_ns"],
                                    "event_type": row.get("eventType"), "range_id": row.get("rangeId")})
        warnings = []
        severity = {row["id"]: row["name"] for row in rows("ENUM_DIAGNOSTIC_SEVERITY_LEVEL")}
        for row in rows("DIAGNOSTIC_EVENT"):
            level = severity.get(row["severity"], "Unknown")
            require(level in ("Info", "Warning") and not any(token in row["text"].lower()
                    for token in helpers.BAD_IMPORT), "Nsight collection/import failure: " + row["text"])
            if level == "Warning":
                global_pid = row.get("globalPid")
                pid = processes.get(global_pid, {}).get("pid")
                scope = "worker" if global_pid in worker_processes else \
                    "collector" if pid is not None and pid == telemetry.get("collector_pid") else \
                    "session" if global_pid is None or global_pid < 0 else "other"
                warnings.append({"source": "sqlite", "severity": level, "text": row["text"],
                                 "global_pid": global_pid, "pid": pid, "scope": scope,
                                 "recorded_timestamp": row.get("timestamp"),
                                 "timestamp_type": row.get("timestampType"), "diagnostic_source": row.get("source")})
        export = {row["name"]: row["value"] for row in rows("META_DATA_EXPORT")}
        require(PurePosixPath(export["EXPORT_PARAM_INPUT_FILE"]).name == expected_report,
                "SQLite export names a different Nsight report")
    activities.sort(key=lambda row: (row["start_ns"], row["end_ns"], row["kind"]))
    return {"time_origin": origin, "device": device, "gpu_activities": activities,
            "cuda_api_intervals": sorted(api_intervals, key=lambda row: row["start_ns"]),
            "nvtx_annotations": annotations, "filtered_activity_counts": dict(filtered),
            "captured_processes": list(worker_processes.values()), "warnings": warnings,
            "export_metadata": export}


def source_binding(raw, root):
    dependencies = raw["dependencies_sha256"]
    require(set(dependencies) == DEPENDENCY_PATHS, "different workflow harness dependencies")
    check_hash(root / "benchmarks/profile_workflow.py", raw["harness_sha256"])
    for name, value in dependencies.items():
        check_hash(root / name, value)
    # Reuse the immutable generated/native source proof, with its own harness
    # dependency explicitly supplied; the new workflow harness was checked above.
    proof = {**raw, "harness_sha256": dependencies["benchmarks/profile_timeline.py"],
             "benchmark_sha256": dependencies["benchmarks/benchmark.py"],
             "profile_resident_sha256": dependencies["benchmarks/profile_resident.py"]}
    timeline.source_binding(proof, root)


def validate_worker(raw, root, verify_fixture=True):
    args, worker, result = raw["arguments"], raw["worker"], raw["worker"]["result"]
    operation = args["operation"]
    require(operation in ("compress", "decompress"), "unknown workflow operation")
    require(raw["operation"] == operation and raw["metadata_layout"] ==
            (["encoded_length", "status"] if operation == "compress" else ["status", "reserved_zero"]),
            "different operation or metadata layout")
    require(raw["schema_version"] == 1 and raw["complete"] is True and not raw.get("errors"),
            "incomplete workflow collector report")
    require(type(worker["exit_code"]) is int and worker["exit_code"] == 0
            and result["complete"] is True and result["pid"] == worker["pid"], "failed/different workflow worker")
    for key in ("source_revision", "harness_sha256", "dependencies_sha256", "source_sha256",
                "native_build", "environment", "fixture"):
        require(result[key] == raw[key], f"worker/collector {key} differs")
    size = integer(args["size"], "payload size", 1)
    chunk = integer(args["chunk_bytes"], "chunk bytes", 256)
    require(chunk <= 65535, "invalid compression chunk bytes")
    counts = {False: integer(args["iterations"], "measured iterations", 1),
              True: integer(args["warmups"], "warmup iterations", 1)}
    expected = {(warmup, i) for warmup, count in counts.items() for i in range(count)}
    records = result["iterations"]
    require(len(records) == len(expected), "missing/extra validated workflows")
    fixture = raw["fixture"]
    require(fixture["workload"] == args["workload"] and fixture["seed"] == args["seed"]
            and fixture["output_bytes"] == size and fixture["chunk_bytes"] == chunk,
            "different workflow fixture")
    capacity = size + ((size + chunk - 1) // chunk) * 5 + 6
    require(fixture["output_buffer_bytes"] == (capacity if operation == "compress" else size),
            "different output buffer extent")
    require(fixture["input_bytes"] == (size if operation == "compress" else fixture["stdlib_stream_bytes"]),
            "different upload extent")
    require(fixture["input_sha256"] == fixture["operation_input_sha256"], "different upload hash aliases")
    require(fixture["encoded_bytes"] == fixture["stdlib_stream_bytes"]
            and fixture["stream_sha256"] == fixture["stdlib_stream_sha256"], "different stdlib fixture aliases")
    if operation == "compress":
        require(fixture["raw_sha256"] == fixture["input_sha256"] and fixture["stdlib_stream_bytes"] is None
                and fixture["stdlib_stream_sha256"] is None and fixture["stdlib_level"] is None,
                "compression fixture must describe raw upload")
    else:
        require(fixture["input_sha256"] == fixture["stdlib_stream_sha256"] and fixture["stdlib_level"] == 6,
                "decompression fixture must describe the level-6 stream upload")
    for key in ("raw_sha256", "input_sha256"):
        require(re.fullmatch(r"[0-9a-f]{64}", fixture[key]) is not None, "invalid fixture hash")
    required_guard = {"upload", "codec", "metadata_download", "status_check", "output_download", "host_bytes"}
    seen = set()
    for row in records:
        require(type(row["warmup"]) is bool and type(row["iteration"]) is int, "invalid iteration identity")
        key = row["warmup"], row["iteration"]
        require(key in expected and key not in seen, "duplicate/unknown iteration")
        seen.add(key)
        label = f"{'warmup' if row['warmup'] else 'measured'}-{row['iteration']:03d}"
        require(row["iteration_id"] == label and row["operation"] == operation, "different iteration operation/label")
        status = row["status"]
        require(isinstance(status, list) and len(status) == 2 and all(type(v) is int for v in status),
                "invalid native metadata")
        require(type(row["status_code"]) is int and row["status_code"] == 0
                and row["byte_exact"] is True and row["cpu_codec_forbidden"] is True
                and row["validation_outside_cpu_codec_guard"] is True,
                "failed status/output oracle or CPU codec guard")
        require(set(row["cpu_codec_forbidden_phases"]) == required_guard, "different CPU codec guard scope")
        encoded = integer(row["encoded_length"], "encoded length", 8)
        for name in ("output_bytes", "downloaded_bytes", "host_bytes_bytes"):
            integer(row[name], name, 1)
        if operation == "compress":
            require(status == [encoded, 0] and encoded <= capacity and row["downloaded_bytes"] == capacity
                    and row["host_bytes_bytes"] == encoded and row["validation_oracle"] == "stdlib.zlib.decompress",
                    "different compression metadata/output extent/oracle")
        else:
            require(status == [0, 0] and encoded == fixture["stdlib_stream_bytes"]
                    and row["downloaded_bytes"] == size and row["host_bytes_bytes"] == size
                    and row["validation_oracle"] == "byte_compare", "different decompression metadata/output extent/oracle")
        require(row["output_bytes"] == size, "different raw output size")
        phases = [phase for phase in raw["phase_intervals"] if phase.get("iteration_id") == label]
        names = {name: [phase for phase in phases if phase["name"] == name]
                 for name in ("upload", "codec", "download_check", "metadata_download", "status_check",
                              "output_download", "host_bytes", "validate")}
        require(len(phases) == len(names) and all(len(items) == 1 for items in names.values()) and all(phase.get("success") is True
                and phase.get("operation") == operation and phase.get("warmup") is row["warmup"]
                and phase.get("iteration") == row["iteration"] for phase in phases),
                "missing/failed/different detailed iteration phases")
        download = names["download_check"][0]
        for name in ("metadata_download", "status_check", "output_download", "host_bytes", "validate"):
            require(names[name][0]["parent_id"] == download["id"], "download leaf has another parent")
        ordered = [names[name][0] for name in ("metadata_download", "status_check", "output_download", "host_bytes", "validate")]
        require(all(a["end_ns"] <= b["start_ns"] for a, b in zip(ordered, ordered[1:])),
                "metadata/status/output validation ordering differs")
        roots = [names[name][0] for name in ("upload", "codec", "download_check")]
        require(all(a["end_ns"] <= b["start_ns"] for a, b in zip(roots, roots[1:])), "workflow phase ordering differs")
    require(seen == expected, "missing validated workflows")
    validation = {"payload_regenerated": False, "stdlib_stream_regenerated": False,
                  "compression_oracle": "source-bound worker stdlib decompression outside codec guard"}
    if verify_fixture:
        spec = importlib.util.spec_from_file_location("workflow_payloads", root / "benchmarks/benchmark.py")
        payloads = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(payloads)
        raw_bytes = payloads.make_payload(args["workload"], size, args["seed"])
        require(len(raw_bytes) == size and hashlib.sha256(raw_bytes).hexdigest() == fixture["raw_sha256"],
                "seeded raw payload differs")
        validation["payload_regenerated"] = True
        if operation == "decompress":
            import zlib
            stream = zlib.compress(raw_bytes, 6)
            require(len(stream) == fixture["stdlib_stream_bytes"] and hashlib.sha256(stream).hexdigest()
                    == fixture["stdlib_stream_sha256"] == fixture["input_sha256"]
                    and zlib.decompress(stream) == raw_bytes, "seeded stdlib stream differs")
            validation.update(stdlib_stream_regenerated=True, local_zlib_runtime=zlib.ZLIB_RUNTIME_VERSION)
    return validation


def normalize_phases(raw, origin, profiled):
    if profiled:
        return timeline.normalize_phases(raw, origin)
    # NVTX-disabled controls retain explicit null brackets without asking the
    # immutable NVTX normalizer to convert them to numeric clock coordinates.
    brackets = ("nvtx_start_bracket_ns", "nvtx_end_bracket_ns")
    clean = {**raw, "phase_intervals": [{key: value for key, value in phase.items()
             if key not in brackets or all(stamp is not None for stamp in value)} for phase in raw["phase_intervals"]]}
    phases, views = timeline.normalize_phases(clean, origin)
    originals = {phase["id"]: phase for phase in raw["phase_intervals"]}
    for phase in phases:
        for key in brackets:
            if key not in phase and key in originals[phase["id"]]:
                phase[key] = originals[phase["id"]][key]
    return phases, views


def complement(start, end, intervals):
    gaps, cursor = [], start
    for left, right in sorted(intervals):
        left, right = max(left, start), min(right, end)
        if left >= right:
            continue
        if left > cursor:
            gaps.append({"start_ns": cursor, "end_ns": left, "duration_ns": left - cursor})
        cursor = max(cursor, right)
    if cursor < end:
        gaps.append({"start_ns": cursor, "end_ns": end, "duration_ns": end - cursor})
    return gaps


def observed_overlap(rows, start, end):
    if rows is None:
        return None
    spans = [(max(row["start_ns"], start), min(row["end_ns"], end)) for row in rows
             if row["start_ns"] < end and row["end_ns"] > start]
    return helpers.union_ns(spans)


def host_statistics(phases, bounds, activities=None, api_intervals=None):
    """Durations and annotation complements; CUDA intersections do not assign ownership."""
    indexed = {row["id"]: row for row in phases}
    require(len(indexed) == len(phases), "duplicate phase IDs")
    children, groups = defaultdict(list), defaultdict(list)
    for row in phases:
        if row.get("parent_id") is not None:
            children[row["parent_id"]].append(row)
        if row.get("iteration_id") is not None:
            groups[row["iteration_id"]].append(row)
    phase_stats = []
    for row in phases:
        start, end = interval(row, "host phase")
        child_union = helpers.union_ns((child["start_ns"], child["end_ns"]) for child in children[row["id"]])
        phase_stats.append({key: row.get(key) for key in ("id", "name", "parent_id", "operation", "iteration_id", "iteration", "warmup")}
            | {"start_ns": start, "end_ns": end, "duration_ns": end - start,
               "children_union_ns": child_union, "exclusive_annotation_ns": end - start - child_union,
               "observed_gpu_overlap_ns": observed_overlap(activities, start, end),
               "observed_cuda_api_overlap_ns": observed_overlap(api_intervals, start, end)})
    iterations = []
    for label, rows in groups.items():
        operation = {row["operation"] for row in rows}
        require(len(operation) == 1, "mixed operations inside iteration")
        ids = {row["id"] for row in rows}
        roots = [row for row in rows if row.get("parent_id") not in ids]
        start, end = min(row["start_ns"] for row in roots), max(row["end_ns"] for row in roots)
        spans = [(row["start_ns"], row["end_ns"]) for row in roots]
        gaps = complement(start, end, spans)
        iterations.append({"iteration_id": label, "operation": next(iter(operation)),
            "iteration": rows[0]["iteration"], "warmup": rows[0]["warmup"],
            "start_ns": start, "end_ns": end, "duration_ns": end - start,
            "annotated_union_ns": helpers.union_ns(spans), "unannotated_intervals": gaps,
            "unannotated_ns": sum(row["duration_ns"] for row in gaps),
            "phase_ids": [row["id"] for row in rows],
            "observed_gpu_overlap_ns": observed_overlap(activities, start, end),
            "observed_cuda_api_overlap_ns": observed_overlap(api_intervals, start, end)})
    summaries = {}
    for warmup in (False, True):
        selected = [row for row in phase_stats if row["warmup"] is warmup]
        values = defaultdict(list)
        for row in selected:
            values[row["name"]].append(row["duration_ns"])
        summaries["warmup" if warmup else "measured"] = {
            name: {"count": len(times), "median_ns": statistics.median(times), "minimum_ns": min(times), "maximum_ns": max(times)}
            for name, times in sorted(values.items())}
    top = [row for row in phases if row.get("parent_id") is None]
    gaps = complement(bounds["start_ns"], bounds["end_ns"], [(row["start_ns"], row["end_ns"]) for row in top])
    return {"phase_stats": phase_stats, "iteration_stats": sorted(iterations, key=lambda row: row["start_ns"]),
            "phase_summary": summaries, "whole_process_unannotated_intervals": gaps,
            "attribution_method": "Measured host annotation durations and their interval complements. "
                "Exclusive annotation time excludes direct child interval unions, not CPU/GPU execution. "
                "Unannotated intervals are not attributed to CPU work. CUDA intersections are observed overlaps only; "
                "absent CUDA measurements are null, not zero."}


def validate_command(path, raw):
    command = json.loads(path.read_text())
    require(isinstance(command, list) and all(isinstance(value, str) for value in command)
            and len(command) > 2 and command[1] == "profile", "invalid Nsight workflow command")
    require(set(timeline.command_option(command, "--trace").split(",")) in ({"cuda", "nvtx"}, {"cuda", "nvtx", "osrt"})
            and timeline.command_option(command, "--cuda-graph-trace") == "node", "different workflow tracing")
    require(not any(value.startswith("--capture-range") for value in command), "expected full-process capture")
    positions = [i for i, value in enumerate(command) if PurePosixPath(value).name == "profile_workflow.py"]
    require(len(positions) == 1, "missing/ambiguous workflow harness")
    arguments = command[positions[0] + 1:]
    for name in ("operation", "size", "seed", "iterations", "warmups", "device", "chunk_bytes", "telemetry"):
        require(timeline.command_option(arguments, "--" + name.replace("_", "-")) == str(raw["arguments"][name]),
                "command differs from recorded " + name)
    require(timeline.command_option(arguments, "--source-revision") == raw["source_revision"], "different command revision")
    return command


def extract(telemetry_path, sqlite_path=None, receipt_path=None, root=ROOT, verify_fixture=True):
    telemetry_path, root = Path(telemetry_path), Path(root)
    raw = json.loads(telemetry_path.read_text())
    require(re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", raw["source_revision"]) is not None,
            "full source revision required")
    source_binding(raw, root)
    validation = validate_worker(raw, root, verify_fixture)
    profiled = sqlite_path is not None
    if profiled:
        require(receipt_path is not None, "profiled extraction requires capture receipt")
        sqlite_path, receipt_path = Path(sqlite_path), Path(receipt_path)
        receipt = json.loads(receipt_path.read_text())
        require(receipt["source_revision"] == raw["source_revision"], "different capture revision")
        require(datetime.fromisoformat(receipt["capture_started_utc"]).utcoffset() is not None, "capture timestamp needs timezone")
        paths, artifacts, hashes = timeline.artifact_inventory(receipt_path, receipt, sqlite_path, telemetry_path)
        for key, metadata in (("worker_result", "result_sha256"), ("worker_log", "log_sha256")):
            if key in paths:
                require(digest(paths[key]) == raw["worker"][metadata], "different worker artifact")
        if "worker_result" in paths:
            require(json.loads(paths["worker_result"].read_text()) == raw["worker"]["result"], "different worker result")
        command = validate_command(paths["command"], raw)
        parsed = read_sqlite(sqlite_path, raw, paths["nsys_report"].name)
        warnings = parsed.pop("warnings")
        for key in ("log", "export_log", "worker_log"):
            if key in paths:
                warnings.extend(helpers.log_warnings(paths[key].read_text(), key))
        for key, path in paths.items():
            check_hash(path, receipt["artifacts"][key]["sha256"])
        check_hash(receipt_path, artifacts["capture_receipt"]["sha256"])
        capture = {"capture_started_utc": receipt["capture_started_utc"], "toolchain": receipt["toolchain"], "command": command}
        if "runner_sha256" in receipt:
            require(re.fullmatch(r"[0-9a-f]{64}", receipt["runner_sha256"]) is not None, "invalid runner hash")
            capture["runner_sha256"] = receipt["runner_sha256"]
    else:
        require(receipt_path is None, "control extraction does not accept an Nsight receipt")
        parsed = {"time_origin": {"method": "worker monotonic start; no cross-trace alignment", "monotonic_ns": raw["worker"]["start_ns"],
                                   "nsight_ns": None, "uncertainty_ns": None},
                  "gpu_activities": [], "cuda_api_intervals": [], "nvtx_annotations": [],
                  "device": raw.get("selected_gpu"), "filtered_activity_counts": {}}
        warnings, artifacts, hashes, capture = [], {}, {}, {"telemetry_sha256": digest(telemetry_path)}
    origin = parsed["time_origin"]["monotonic_ns"]
    phases, views = normalize_phases(raw, origin, profiled)
    if profiled:
        timeline.validate_phase_nvtx(phases, parsed["nvtx_annotations"], parsed["time_origin"]["uncertainty_ns"])
        bounds, tolerance = views["whole_process"], parsed["time_origin"]["uncertainty_ns"]
        require(all(bounds["start_ns"] - tolerance <= row["start_ns"] < row["end_ns"] <= bounds["end_ns"] + tolerance
                    for row in parsed["gpu_activities"]), "CUDA activity outside aligned worker lifetime")
    samples = timeline.normalize_samples(raw, origin) if raw["samples"] else []
    stats = host_statistics(phases, views["whole_process"], parsed["gpu_activities"] if profiled else None,
                            parsed["cuda_api_intervals"] if profiled else None)
    spacings = [b["timestamp_ns"] - a["timestamp_ns"] for a, b in zip(samples, samples[1:])]
    sampling = {"sample_count": len(samples), "requested_interval_ns": raw.get("methodology", {}).get("sample_interval_requested_ns"),
                "median_observed_interval_ns": statistics.median(spacings) if spacings else None,
                "minimum_observed_interval_ns": min(spacings) if spacings else None,
                "maximum_observed_interval_ns": max(spacings) if spacings else None,
                "median_query_duration_ns": statistics.median(row["query_end_ns"] - row["query_start_ns"] for row in samples) if samples else None}
    args = raw["arguments"]
    return {"schema": 1, "schema_version": 1, "kind": "workflow_timeline" if profiled else "workflow_control",
            "complete": True, "profiled": profiled, "telemetry_present": bool(samples),
            "operation": args["operation"], "source_revision": raw["source_revision"],
            "source_sha256": raw["source_sha256"], "harness_sha256": raw["harness_sha256"],
            "dependencies_sha256": raw["dependencies_sha256"],
            "extractor_sha256": digest(Path(__file__)), "extractor_dependencies_sha256": EXTRACTOR_DEPENDENCIES,
            "benchmark_sha256": raw["dependencies_sha256"]["benchmarks/benchmark.py"],
            "native_build": raw["native_build"], "environment": raw["environment"],
            "gpu_uuid": raw.get("selected_gpu", {}).get("uuid"), "timestamp_unit": "ns", "timestamp_reference": "time_origin",
            "worker": timeline.relative_times(raw["worker"], origin), "metadata_layout": raw["metadata_layout"],
            "workload": {"kind": args["workload"], "input_bytes": raw["fixture"]["input_bytes"], "output_bytes": args["size"],
                         "iterations": args["iterations"], "warmups": args["warmups"], "seed": args["seed"], "chunk_bytes": args["chunk_bytes"]},
            "fixture": raw["fixture"], "fixture_oracle_verified": verify_fixture, "fixture_validation": validation,
            "phase_intervals": phases, "phase_semantics": PHASE_SEMANTICS, "views": views, "samples": samples, "sampling": sampling,
            "bin_width_ns": timeline.BIN_NS, "bins": timeline.activity_bins(parsed["gpu_activities"], views["whole_process"]) if profiled else [],
            "artifacts": artifacts, "artifact_sha256": hashes, "warnings": warnings + raw.get("warnings", []), "stages": STAGES,
            "methodology": "Detailed completed compression/decompression workflows with native metadata checked before output access. "
                "Compression downloads the full padded device array; meaningful encoded extent is separate. "
                "Its stdlib output oracle runs outside the CPU codec guard. Decompression checks raw byte equality. "
                "Host annotations and interval complements do not assign GPU ownership or CPU work. "
                "CUDA activity/kernel unions and completed-copy bytes are observed only for profiled captures. "
                "Resource clocks retain per-query midpoints; CPU is interval-averaged and NVML utilization is whole-device. "
                "Instrumented controls/captures are diagnostic, not adoption benchmarks.", **parsed, **stats, **capture}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sqlite", type=Path)
    parser.add_argument("--receipt", type=Path)
    parser.add_argument("--telemetry", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--control", action="store_true", help="normalize telemetry-only control without claiming CUDA observations")
    parser.add_argument("--skip-fixture-oracle", action="store_true")
    args = parser.parse_args()
    try:
        require(args.control and args.sqlite is None and args.receipt is None or
                not args.control and args.sqlite is not None and args.receipt is not None,
                "provide --sqlite/--receipt or explicit --control")
        report = extract(args.telemetry, args.sqlite, args.receipt, args.root, not args.skip_fixture_oracle)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
    except (OSError, KeyError, TypeError, ValueError, sqlite3.Error) as error:
        print(f"Workflow extraction failed: {error}", file=sys.stderr)
        return 1
    print(f"Validated {report['operation']} workflow: {len(report['iteration_stats'])} iterations, "
          f"{len(report['gpu_activities'])} CUDA activities, {len(report['samples'])} resource samples.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
