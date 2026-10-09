#!/usr/bin/env python3
# Copyright (c) 2026 xangma
# SPDX-License-Identifier: MIT
"""Normalize a source-bound full-process Nsight capture and resource samples.

All normalized timestamps are integer nanoseconds relative to the bracketed
NVTX origin. Kernel unions describe observed activity, not occupancy. Copy bins
contain bytes completed at each copy's end, not a PCIe bandwidth estimate.
GPU utilization and total memory use are whole-device observations; process
CPU, RSS and owned compute memory are separately scoped to the worker tree.
Normalization needs only the Python standard library and the MIT extract_nsight
helpers; the default seeded fixture oracle also imports benchmark's NumPy.
It never opens SQLite for writing or loads a CUDA library.
"""

import argparse
import ast
from collections import defaultdict
from datetime import datetime
import hashlib
import importlib.util
import json
import math
from pathlib import Path, PurePosixPath
import re
import sqlite3
import statistics
import sys


ROOT = Path(__file__).resolve().parents[1]
HELPER_PATH = Path(__file__).with_name("extract_nsight.py")
SPEC = importlib.util.spec_from_file_location("timeline_nsight_helpers", HELPER_PATH)
helpers = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(helpers)
require, digest, check_hash = helpers.require, helpers.digest, helpers.check_hash
BIN_NS = 10_000_000
REQUIRED_ARTIFACTS = {"nsys_report", "telemetry", "command", "log", "export_log", "sqlite"}
PHASE_SEMANTICS = {
    "prepare": {"kind": "host_only", "description": "CPU fixture preparation and stdlib input oracle"},
    "init": {"kind": "host_annotation", "description": "Initialization annotation; backend/native APIs may enqueue GPU work"},
    "warmup": {"kind": "container", "description": "Complete warmup workflows"},
    "measured_loop": {"kind": "container", "description": "Complete measured workflows"},
    "upload": {"kind": "enqueue", "description": "jax.device_put; no added GPU synchronization"},
    "decode": {"kind": "completed", "description": "block_until_ready on output and checked metadata"},
    "download_check": {"kind": "completed", "description": "Metadata transfer/status check before output transfer and byte oracle"},
}


def integer(value, label, minimum=None):
    require(type(value) is int and (minimum is None or value >= minimum),
            f"invalid integer {label}")
    return value


def interval(row, label):
    start = integer(row["start_ns"], label + " start")
    end = integer(row["end_ns"], label + " end")
    require(end > start, f"nonpositive {label} interval")
    return start, end


def gpu_uuid(value):
    require(isinstance(value, str), "missing GPU UUID")
    normalized = value.lower().removeprefix("gpu-").replace("-", "")
    require(re.fullmatch(r"[0-9a-f]{32}", normalized), "unsupported GPU UUID format")
    return normalized


def origin_from_nvtx(nvtx, strings, anchor, processes=None, collector_pid=None):
    before = integer(anchor["monotonic_before_ns"], "origin before", 0)
    after = integer(anchor["monotonic_after_ns"], "origin after", before)
    label = anchor["label"]
    require(isinstance(label, str) and label, "missing origin label")
    matches = [row for row in nvtx if nvtx_text(row, strings) == label]
    require(len(matches) == 1, "missing or ambiguous NVTX origin")
    require(matches[0].get("end") is None and matches[0].get("eventType") == 34,
            "NVTX origin must be a recorded markA instant")
    if collector_pid is not None:
        global_pid = matches[0]["globalTid"] & ~0xFFFFFF
        require(global_pid in processes and processes[global_pid]["pid"] == collector_pid,
                "NVTX origin belongs to another supervisor")
    stamp = integer(matches[0]["start"], "Nsight origin", 0)
    midpoint = (before + after) // 2
    return {"method": "NVTX instant to bracketed monotonic midpoint",
            "label": label, "monotonic_ns": midpoint, "nsight_ns": stamp,
            "monotonic_before_ns": before, "monotonic_after_ns": after,
            "half_bracket_ns": (after - before) / 2,
            "uncertainty_ns": max(midpoint - before, after - midpoint),
            "rounding": "integer midpoint rounded down; uncertainty rounded up"}


def nvtx_text(row, strings):
    text = row.get("text")
    if text is not None:
        require(isinstance(text, str), "invalid NVTX text")
        return text
    key = row.get("textId")
    return strings[key] if key is not None else ""


def source_binding(telemetry, root):
    package = root / "src/cuda_zlib"
    actual = {path.relative_to(package).as_posix(): digest(path)
              for path in package.rglob("*")
              if path.is_file() and path.suffix in (".py", ".cu", ".cuh")}
    native = telemetry["native_build"]
    require(actual == telemetry["source_sha256"], "stale runtime source inventory")
    if "source_sha256" in native:
        require(actual == native["source_sha256"], "stale native source inventory")
    check_hash(root / "benchmarks/profile_timeline.py", telemetry["harness_sha256"])
    check_hash(root / "benchmarks/benchmark.py", telemetry["benchmark_sha256"])
    if "profile_resident_sha256" in telemetry:
        check_hash(root / "benchmarks/profile_resident.py", telemetry["profile_resident_sha256"])
    generated = {}
    for filename, module in (("encoder.cuh", "_encode_kernels.py"),
                             ("decoder.cuh", "_decode_kernels.py"),
                             ("postprocess.cuh", "_postprocess.py")):
        assignments = [node.value for node in ast.parse((package / module).read_text()).body
                       if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name)
                       and target.id == "CUDA_SOURCE" for target in node.targets)]
        require(len(assignments) == 1, f"ambiguous CUDA_SOURCE in {module}")
        generated[filename] = hashlib.sha256(ast.literal_eval(assignments[0]).encode()).hexdigest()
    generated.update({name: digest(package / "native" / name)
                      for name in ("codec_ffi.cu", "batch_encode.cuh", "batch_decode.cuh")})
    require(generated == native["identity"]["sources"], "different native generated sources")
    require(hashlib.sha256(json.dumps(native["identity"], sort_keys=True).encode()).hexdigest()
            == native["cache_key"], "native cache key does not match build identity")
    for key in ("library_sha256", "build_sha256"):
        require(re.fullmatch(r"[0-9a-f]{64}", native[key]) is not None, f"invalid {key}")
    architecture = native["identity"]["architecture"]
    require(native["ffi_targets"] == [f"cuda_zlib_{action}_{architecture}" for action in
            ("compress", "decompress", "compress_batch", "decompress_batch")], "different FFI targets")


def validate_worker(telemetry, root, verify_fixture):
    worker, args = telemetry["worker"], telemetry["arguments"]
    result = worker["result"]
    require(result["complete"] is True and result["pid"] == worker["pid"], "incomplete/different worker result")
    require(not telemetry.get("errors"), "collector reported errors")
    for key in ("source_revision", "harness_sha256", "benchmark_sha256", "source_sha256",
                "native_build", "environment", "fixture"):
        require(result[key] == telemetry[key], f"worker/collector {key} differs")
    if "profile_resident_sha256" in telemetry:
        require(result["profile_resident_sha256"] == telemetry["profile_resident_sha256"],
                "worker/collector resident harness differs")
    count = integer(args["iterations"], "measured iteration count", 1)
    warmups = integer(args["warmups"], "warmup count", 1)
    expected = {(warmup, i) for warmup, n in ((True, warmups), (False, count)) for i in range(n)}
    records = result["iterations"]
    require(len(records) == len(expected), "missing/extra validated workflows")
    seen = set()
    for row in records:
        require(type(row["warmup"]) is bool and type(row["iteration"]) is int, "invalid workflow identity")
        key = row["warmup"], row["iteration"]
        require(key in expected and key not in seen, "missing/duplicate workflow identity")
        seen.add(key)
        require(row["iteration_id"] == f"{'warmup' if row['warmup'] else 'measured'}-{row['iteration']:03d}",
                "different workflow label")
        require(row["byte_exact"] is True and row["cpu_codec_forbidden"] is True
                and row["status"] == [0, 0] and all(type(value) is int for value in row["status"]),
                "workflow status or byte oracle failed")
    require(seen == expected, "missing validated workflows")
    phases = telemetry["phase_intervals"]
    for warmup, i in expected:
        for name in ("upload", "decode", "download_check"):
            matches = [row for row in phases if row["name"] == name
                       and row.get("warmup") is warmup and row.get("iteration") == i]
            require(len(matches) == 1 and matches[0].get("success") is True,
                    f"missing/failed {name} workflow phase")
    fixture = telemetry["fixture"]
    require(fixture["workload"] == args["workload"] and fixture["output_bytes"] == args["size"]
            and fixture["seed"] == args["seed"] and fixture["stdlib_level"] == 6,
            "different workload fixture")
    for key in ("input_sha256", "stream_sha256"):
        require(re.fullmatch(r"[0-9a-f]{64}", fixture[key]) is not None, f"invalid fixture {key}")
    if verify_fixture:
        # Only this optional fixture check imports benchmark's NumPy dependency.
        spec = importlib.util.spec_from_file_location("timeline_payloads", root / "benchmarks/benchmark.py")
        payloads = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(payloads)
        import zlib
        raw = payloads.make_payload(args["workload"], args["size"], args["seed"])
        stream = zlib.compress(raw, 6)
        require(len(raw) == fixture["output_bytes"] and hashlib.sha256(raw).hexdigest() == fixture["input_sha256"],
                "seeded payload differs from captured fixture")
        require(len(stream) == fixture["encoded_bytes"] and hashlib.sha256(stream).hexdigest() == fixture["stream_sha256"]
                and zlib.decompress(stream) == raw, "stdlib stream oracle differs from captured fixture")
        return {"payload_regenerated": True, "stream_regenerated": True,
                "local_zlib_runtime": zlib.ZLIB_RUNTIME_VERSION,
                "captured_zlib_runtime": fixture["zlib_runtime"]}
    return {"payload_regenerated": False, "stream_regenerated": False,
            "captured_zlib_runtime": fixture["zlib_runtime"]}


def relative_times(value, origin):
    if isinstance(value, list):
        return [relative_times(item, origin) for item in value]
    if not isinstance(value, dict):
        return value
    result = {}
    for key, item in value.items():
        if key in {"start_ns", "end_ns", "timestamp_ns", "query_start_ns", "query_end_ns", "received_ns",
                   "launch_start_ns", "first_seen_ns", "last_seen_ns", "nvtx_before_ns", "nvtx_after_ns"}:
            result[key] = None if item is None else integer(item, key) - origin
        elif key in {"nvtx_start_bracket_ns", "nvtx_end_bracket_ns"}:
            result[key] = [integer(stamp, key) - origin for stamp in item]
        else:
            result[key] = relative_times(item, origin)
    return result


def artifact_inventory(receipt_path, receipt, sqlite_path, telemetry_path):
    artifacts = receipt["artifacts"]
    require(REQUIRED_ARTIFACTS <= artifacts.keys(), "incomplete capture artifact inventory")
    paths, normalized, hashes = {}, {}, {}
    for key, item in artifacts.items():
        relative = PurePosixPath(item["path"])
        require(not relative.is_absolute() and ".." not in relative.parts,
                "artifact path must stay inside capture directory")
        path = receipt_path.parent / relative
        check_hash(path, item["sha256"])
        paths[key] = path
        if key == "sqlite":
            require(item.get("private") is True and "published_path" not in item,
                    "SQLite must remain private")
            normalized[key] = {"filename": str(relative), "sha256": item["sha256"], "private": True}
        else:
            published = PurePosixPath(item["published_path"])
            require(not published.is_absolute() and ".." not in published.parts
                    and published.name == relative.name, "invalid published artifact path")
            require(str(published) not in hashes, "duplicate published artifact")
            hashes[str(published)] = item["sha256"]
            normalized[key] = {"path": str(published), "sha256": item["sha256"]}
    require(paths["sqlite"].resolve() == sqlite_path.resolve(), "different receipt SQLite")
    require(paths["telemetry"].resolve() == telemetry_path.resolve(), "different receipt telemetry")
    published_receipt = str(PurePosixPath(normalized["telemetry"]["path"]).parent / receipt_path.name)
    require(published_receipt not in hashes, "receipt collides with artifact")
    hashes[published_receipt] = digest(receipt_path)
    normalized["capture_receipt"] = {"path": published_receipt, "sha256": hashes[published_receipt]}
    return paths, normalized, dict(sorted(hashes.items()))


def command_option(command, flag):
    values = [item.split("=", 1)[1] for item in command if item.startswith(flag + "=")]
    values.extend(command[i + 1] for i, item in enumerate(command[:-1]) if item == flag)
    require(len(values) == 1, f"missing or ambiguous {flag}")
    return values[0]


def validate_command(path):
    command = json.loads(path.read_text())
    require(isinstance(command, list) and all(isinstance(x, str) for x in command)
            and len(command) > 2 and command[1] == "profile", "invalid Nsight command")
    require(set(command_option(command, "--trace").split(",")) == {"cuda", "nvtx", "osrt"},
            "expected CUDA/NVTX/OSRT capture")
    require(command_option(command, "--cuda-graph-trace") == "node", "graph node tracing required")
    require(not any(item.startswith("--capture-range") for item in command),
            "expected full-process capture without capture-range gating")
    require(sum(PurePosixPath(item).name == "profile_timeline.py" for item in command) == 1,
            "missing or ambiguous timeline harness")
    return command


def _copy_direction(kind, names):
    label = names.get(kind)
    # CUPTI activity copy kinds are retained even when they are not one of the
    # three plotted directions (for example array, host-host or peer copies).
    compact = re.sub(r"[^a-z0-9]", "", label.lower()) if label else ""
    known = {"htod": "H2D", "h2d": "H2D", "hosttodevice": "H2D",
             "dtoh": "D2H", "d2h": "D2H", "devicetohost": "D2H",
             "dtod": "D2D", "d2d": "D2D", "devicetodevice": "D2D"}
    return known.get(compact, {1: "H2D", 2: "D2H", 8: "D2D"}.get(kind, "other")), label


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
                        "stage": helpers.kernel_stage(name) if kind == "kernel" else None,
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


def normalize_phases(telemetry, origin):
    worker = telemetry["worker"]
    start = integer(worker["start_ns"], "worker start", 0)
    end = integer(worker["end_ns"], "worker end", start + 1)
    pids = set(worker["process_ids"]) | {worker["pid"]}
    phases, identifiers = [], {}
    for raw in telemetry["phase_intervals"]:
        row = relative_times(raw, origin)
        a, b = interval(raw, "host phase")
        require(start <= a < b <= end and row["pid"] in pids, "phase outside worker lifetime/tree")
        require(isinstance(row["id"], str) and row["id"] and row["id"] not in identifiers,
                "missing/duplicate host phase ID")
        require(isinstance(row["name"], str) and row["name"], "missing phase name")
        phases.append(row)
        identifiers[row["id"]] = row
    for row in phases:
        if row.get("parent_id") is not None:
            require(row["parent_id"] in identifiers, "unknown host phase parent")
            parent = identifiers[row["parent_id"]]
            require(parent["start_ns"] <= row["start_ns"] and row["end_ns"] <= parent["end_ns"]
                    and parent["id"] != row["id"], "invalid nested host phase")
        visited, ancestor = set(), row
        while ancestor.get("parent_id") is not None:
            require(ancestor["id"] not in visited, "cyclic host phase hierarchy")
            visited.add(ancestor["id"])
            ancestor = identifiers[ancestor["parent_id"]]
    measured = [row for row in phases if row["name"] == "measured_loop"]
    require(len(measured) == 1, "missing or ambiguous measured_loop phase")
    return phases, {"whole_process": {"start_ns": start - origin, "end_ns": end - origin},
                    "measured_loop": {key: measured[0][key] for key in ("start_ns", "end_ns")}}


def validate_phase_nvtx(phases, annotations, uncertainty):
    for phase in phases:
        matches = [row for row in annotations if row["name"] == phase["nvtx_label"]
                   and row["pid"] == phase["pid"]]
        require(len(matches) == 1, "missing/ambiguous worker NVTX phase")
        observed = matches[0]
        interval(observed, "NVTX phase")
        for key, bracket in (("start_ns", "nvtx_start_bracket_ns"), ("end_ns", "nvtx_end_bracket_ns")):
            before, after = phase[bracket]
            require(before <= after and before - uncertainty <= observed[key] <= after + uncertainty,
                    "NVTX phase outside aligned host call bracket")


def normalize_samples(telemetry, origin):
    result, previous = [], None
    pids = set(telemetry["worker"]["process_ids"]) | {telemetry["worker"]["pid"]}
    selected_uuid = gpu_uuid(telemetry["selected_gpu"]["uuid"])
    for raw in telemetry["samples"]:
        row = relative_times(raw, origin)
        stamp = integer(raw["timestamp_ns"], "sample timestamp", 0)
        before = integer(raw["query_start_ns"], "sample query start", 0)
        after = integer(raw["query_end_ns"], "sample query end", before)
        require(before <= stamp <= after, "sample timestamp outside query bracket")
        require(previous is None or stamp > previous[0], "nonincreasing resource sample timestamps")
        processes, counters, query_spans, rss, complete = [], {}, [], 0, True
        for process in row["processes"]:
            require(process["pid"] in pids, "resource process outside worker tree")
            item = dict(process)
            processes.append(item)
            values = [item.get(key) for key in ("cpu_user_seconds", "cpu_system_seconds", "rss_bytes")]
            if item.get("missing") or item.get("create_time") is None or any(value is None for value in values):
                complete = False
                continue
            if item.get("query_start_ns") is None or item.get("query_end_ns") is None:
                complete = False
                continue
            qstart, qend = item["query_start_ns"], item["query_end_ns"]
            require(row["query_start_ns"] <= qstart <= qend <= row["query_end_ns"],
                    "process query outside sample bracket")
            require(all(isinstance(value, (int, float)) and not isinstance(value, bool)
                        and math.isfinite(value) and value >= 0 for value in values), "invalid resource counter")
            key = item["pid"], item["create_time"]
            require(key not in counters, "duplicate resource process lifetime")
            counters[key] = values[0] + values[1], (qstart + qend) // 2
            query_spans.append((qstart, qend))
            rss += integer(values[2], "RSS bytes", 0)
        complete = complete and bool(processes)
        process_midpoint = (min(a for a, b in query_spans) + max(b for a, b in query_spans)) // 2 if complete else None
        cpu = None
        cpu_timestamp, cpu_interval = None, None
        if complete and previous and previous[2] and counters.keys() == previous[1].keys():
            deltas = [(value - previous[1][key][0], midpoint - previous[1][key][1])
                      for key, (value, midpoint) in counters.items()]
            require(all(delta >= 0 and elapsed > 0 for delta, elapsed in deltas),
                    "decreasing process CPU counter/query time")
            cpu = sum(delta * 1e11 / elapsed for delta, elapsed in deltas)
            cpu_timestamp = (previous[3] + process_midpoint) // 2
            cpu_interval = {"start_ns": previous[3], "end_ns": process_midpoint}
        gpu = dict(row["gpu"]) if row.get("gpu") is not None else None
        metric_times = {"cpu_percent": cpu_timestamp, "rss_bytes": process_midpoint,
                        "gpu_memory": None, "gpu_owned_memory": None,
                        "gpu_utilization": None, "gpu_memory_utilization": None}
        if gpu is not None:
            require(gpu_uuid(gpu["uuid"]) == selected_uuid, "sample belongs to another GPU")
            for original, normalized in (("total_memory_bytes", "total_bytes"), ("used_memory_bytes", "used_bytes"),
                                         ("free_memory_bytes", "free_bytes"), ("gpu_utilization_percent", "util_gpu_percent"),
                                         ("memory_utilization_percent", "util_memory_percent")):
                gpu[normalized] = gpu.get(original, gpu.get(normalized))
            for key in ("total_bytes", "used_bytes", "free_bytes", "owned_compute_memory_bytes"):
                if gpu.get(key) is not None:
                    integer(gpu[key], "GPU " + key, 0)
            for key in ("util_gpu_percent", "util_memory_percent"):
                value = gpu.get(key)
                require(value is None or (isinstance(value, (int, float)) and not isinstance(value, bool)
                        and math.isfinite(value) and 0 <= value <= 100), "invalid GPU utilization")
            if row.get("process_tree_query", {}).get("error") is not None:
                gpu["owned_compute_memory_bytes"] = None
            for metric, query in (("gpu_memory", "memory"), ("gpu_owned_memory", "compute_processes"),
                                  ("gpu_utilization", "utilization"), ("gpu_memory_utilization", "utilization")):
                bracket = gpu.get("queries", {}).get(query)
                if bracket is not None:
                    a, b = bracket["start_ns"], bracket["end_ns"]
                    require(row["query_start_ns"] <= a <= b <= row["query_end_ns"],
                            "GPU query outside sample bracket")
                    metric_times[metric] = (a + b) // 2
        row.update(timestamp_ns=stamp - origin, query_start_ns=before - origin,
                   query_end_ns=after - origin, processes=processes, gpu=gpu,
                   cpu_percent=cpu, rss_bytes=rss if complete else None,
                   metric_timestamps_ns=metric_times, cpu_interval_ns=cpu_interval)
        result.append(row)
        previous = stamp, counters, complete, process_midpoint
    require(result, "no resource samples")
    return result


def activity_bins(activities, bounds, width=BIN_NS):
    start, end = interval(bounds, "timeline view")
    integer(width, "bin width", 1)
    bins = [{"start_ns": left, "end_ns": min(left + width, end), "kernel_union_ns": 0,
             "kernel_union_fraction": 0.0,
             "copy_bytes_completed": {key: 0 for key in ("H2D", "D2H", "D2D", "other")}}
            for left in range(start, end, width)]
    intervals = defaultdict(list)
    for row in activities:
        a, b = interval(row, "CUDA activity")
        if row["kind"] == "kernel":
            a, b = max(a, start), min(b, end)
            if a < b:
                for index in range((a - start) // width, (b - 1 - start) // width + 1):
                    item = bins[index]
                    intervals[index].append((max(a, item["start_ns"]), min(b, item["end_ns"])))
        elif row["kind"] == "memcpy" and start <= b <= end:
            # Internal boundaries belong to the next bin; the view's final
            # endpoint belongs to its final bin, so no completed copy is lost.
            index = min((b - start) // width, len(bins) - 1)
            bins[index]["copy_bytes_completed"][row["direction"]] += row["bytes"]
    for index, item in enumerate(bins):
        item["kernel_union_ns"] = helpers.union_ns(intervals[index])
        item["kernel_union_fraction"] = item["kernel_union_ns"] / (item["end_ns"] - item["start_ns"])
    return bins


def extract(sqlite_path, telemetry_path, receipt_path, root=ROOT, verify_fixture=True):
    sqlite_path, telemetry_path, receipt_path, root = map(Path, (sqlite_path, telemetry_path, receipt_path, root))
    receipt, telemetry = (json.loads(path.read_text()) for path in (receipt_path, telemetry_path))
    require(telemetry["schema_version"] == 1 and telemetry["complete"] is True,
            "incomplete timeline collector report")
    revision = telemetry["source_revision"]
    require(re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", revision) is not None
            and receipt["source_revision"] == revision, "different or invalid source revision")
    require(datetime.fromisoformat(receipt["capture_started_utc"]).utcoffset() is not None,
            "capture timestamp must include timezone")
    require(telemetry["worker"]["exit_code"] == 0, "worker failed")
    source_binding(telemetry, root)
    fixture_validation = validate_worker(telemetry, root, verify_fixture)
    paths, inventory, hashes = artifact_inventory(receipt_path, receipt, sqlite_path, telemetry_path)
    if "worker_result" in paths:
        require(json.loads(paths["worker_result"].read_text()) == telemetry["worker"]["result"]
                and digest(paths["worker_result"]) == telemetry["worker"]["result_sha256"],
                "different worker result artifact")
    if "worker_log" in paths:
        require(digest(paths["worker_log"]) == telemetry["worker"]["log_sha256"],
                "different worker log artifact")
    command = validate_command(paths["command"])
    parsed = read_sqlite(sqlite_path, telemetry, paths["nsys_report"].name)
    phases, views = normalize_phases(telemetry, parsed["time_origin"]["monotonic_ns"])
    validate_phase_nvtx(phases, parsed["nvtx_annotations"], parsed["time_origin"]["uncertainty_ns"])
    samples = normalize_samples(telemetry, parsed["time_origin"]["monotonic_ns"])
    tolerance = parsed["time_origin"]["uncertainty_ns"]
    bounds = views["whole_process"]
    require(all(bounds["start_ns"] - tolerance <= row["start_ns"] < row["end_ns"]
                <= bounds["end_ns"] + tolerance for row in parsed["gpu_activities"]),
            "GPU activity outside aligned worker lifetime")
    warnings = parsed.pop("warnings")
    for key in ("log", "export_log", "worker_log"):
        if key not in paths:
            continue
        warnings.extend(helpers.log_warnings(paths[key].read_text(), key))
    warnings.extend(telemetry.get("warnings", []))
    for key, path in paths.items():
        check_hash(path, receipt["artifacts"][key]["sha256"])
    check_hash(receipt_path, inventory["capture_receipt"]["sha256"])
    args = telemetry["arguments"]
    spacings = [right["timestamp_ns"] - left["timestamp_ns"] for left, right in zip(samples, samples[1:])]
    sampling = {"sample_count": len(samples),
                "requested_interval_ns": telemetry.get("methodology", {}).get("sample_interval_requested_ns"),
                "median_observed_interval_ns": statistics.median(spacings) if spacings else None,
                "minimum_observed_interval_ns": min(spacings) if spacings else None,
                "maximum_observed_interval_ns": max(spacings) if spacings else None,
                "median_query_duration_ns": statistics.median(row["query_end_ns"] - row["query_start_ns"]
                                                             for row in samples)}
    report = {"schema": 1, "schema_version": 1, "kind": "application_gpu_resource_timeline",
              "complete": True, "source_revision": revision,
              "source_sha256": telemetry["source_sha256"], "harness_sha256": telemetry["harness_sha256"],
              "benchmark_sha256": telemetry["benchmark_sha256"],
              "extractor_sha256": digest(Path(__file__)), "helper_sha256": digest(HELPER_PATH),
              "native_build": telemetry["native_build"], "environment": telemetry["environment"],
              "capture_started_utc": receipt["capture_started_utc"], "toolchain": receipt["toolchain"],
              "gpu_uuid": telemetry["selected_gpu"]["uuid"],
              "worker": relative_times(telemetry["worker"], parsed["time_origin"]["monotonic_ns"]),
              "workload": {"kind": args["workload"], "output_bytes": args["size"],
                           "iterations": args["iterations"], "warmups": args["warmups"], "seed": args["seed"]},
              "timestamp_unit": "ns", "timestamp_reference": "time_origin",
              "fixture": telemetry["fixture"], "fixture_oracle_verified": verify_fixture,
              "fixture_validation": fixture_validation, "sampling": sampling,
              "phase_intervals": phases, "phase_semantics": PHASE_SEMANTICS,
              "views": views, "samples": samples, "bin_width_ns": BIN_NS,
              "bins": activity_bins(parsed["gpu_activities"], bounds),
              "artifacts": inventory, "artifact_sha256": hashes, "command": command,
              "warnings": warnings, "stages": list(helpers.STAGES),
              "methodology": "Full worker-process-tree CUDA/NVTX/OSRT capture, including startup and warmup. "
                  "GPU timestamps are shifted by one bracketed NVTX origin; alignment uncertainty is explicit. "
                  "Host phases do not assign GPU ownership. Every selected-GPU worker kernel/copy/set is retained. "
                  "Kernel union/bin is activity coverage, not occupancy or process efficiency. Copy bins contain "
                  "bytes completed at copy end, not PCIe bandwidth. GPU utilization/memory totals are whole-device. "
                  "Resource samples retain actual timestamps; requested polling cadence is not achieved cadence "
                  "or NVML utilization resolution. RSS sums can double-count shared pages. "
                  "CPU100 is one core; counter/lifetime gaps remain null. Native identity is the collector receipt; "
                  "CPU uses per-process query elapsed time and is plotted at the midpoint of its averaging interval. "
                  "RSS and each NVML metric use their own query midpoint, retaining all query brackets. "
                  "the remote library is not rehashed by this reader.", **parsed}
    if "profile_resident_sha256" in telemetry:
        report["profile_resident_sha256"] = telemetry["profile_resident_sha256"]
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sqlite", type=Path, required=True)
    parser.add_argument("--telemetry", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--skip-fixture-oracle", action="store_true",
                        help="omit seeded NumPy fixture regeneration; records fixture_oracle_verified=false")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        report = extract(args.sqlite, args.telemetry, args.receipt, args.root, not args.skip_fixture_oracle)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
    except (OSError, KeyError, TypeError, ValueError, sqlite3.Error) as error:
        print(f"Timeline extraction failed: {error}", file=sys.stderr)
        return 1
    print(f"Validated {len(report['gpu_activities'])} CUDA activities and {len(report['samples'])} resource samples.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
