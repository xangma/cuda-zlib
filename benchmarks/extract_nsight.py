#!/usr/bin/env python3
# Copyright (c) 2026 xangma
# SPDX-License-Identifier: MIT
"""Validate source-bound, one-case Nsight Systems SQLite exports and extract stages.

No CUDA or Nsight installation is needed to read existing exports. Generate an
export with ``nsys export --type sqlite --output NAME.sqlite NAME.nsys-rep``.
The capture directory must also contain NAME-profile.json, NAME-command.json,
NAME-capture.json, NAME.log and NAME-export.log. The capture receipt pins the
original trace, profile, log and SQLite hashes. Re-exported SQLite files require
the explicit --reexported-sqlite option; their new digest is recorded separately.

Times describe imported activity from one instrumented, warmed resident call.
Kernel sums can overlap. Gaps are uncovered portions of the imported GPU span;
they are not measurements of CPU work. A fused decoder is indivisible here.
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
import zlib


ROOT = Path(__file__).resolve().parents[1]
STAGES = ("fused_decode", "framing", "discovery", "sorting", "description",
          "chain_selection", "emission", "refinement", "checksum", "verification")
KERNEL_STAGES = {
    "SmallSharedDecodeSpecialized": "fused_decode", "SmallDecode": "fused_decode",
    "DecompressionFraming": "framing", "ClearFusedFailure": "framing",
    "scan_prefixes": "discovery", "validate_prefixes": "discovery",
    "ResetDenseDiscovery": "discovery", "discover": "discovery",
    "FinalizeCandidates": "discovery",
    "cub::DeviceRadixSortHistogramKernel": "sorting",
    "cub::DeviceRadixSortExclusiveSumKernel": "sorting",
    "cub::DeviceRadixSortOnesweepKernel": "sorting",
    "cub::DeviceRadixSortSingleTileKernel": "sorting",
    "cub::DeviceRadixSortUpsweepKernel": "sorting",
    "cub::DeviceRadixSortDownsweepKernel": "sorting",
    "describe_candidates": "description", "describe_candidates_counted": "description",
    "fixed_summaries": "description",
    "select_chain": "chain_selection", "select_chain_counted": "chain_selection",
    "ProbeStoredBlocks": "chain_selection",
    "emit_blocks": "emission", "emit_blocks_pipeline": "emission",
    "emit_blocks_warp": "emission", "emit_stored": "emission", "CopyStoredBlocks": "emission",
    "refine_roots": "refinement",
    "adler_parts": "checksum", "write_adler_parts": "checksum", "adler_finish": "checksum",
    "SetMetadata": "verification", "ChainStatus": "verification",
    "EmissionStatus": "verification", "RefinementStatus": "verification",
    "VerifyDecompression": "verification", "VerifyFusedChecksum": "verification",
}
BAD_IMPORT = ("targetprofilingfailed", "cannot find string for exterior index",
              "errors occurred while processing", "failed to import", "importerror")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    with Path(path).open("rb") as handle:
        result = hashlib.sha256()
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def check_hash(path, expected):
    require(isinstance(expected, str) and re.fullmatch(r"[0-9a-f]{64}", expected),
            f"invalid SHA-256: {path}")
    actual = digest(path)
    require(actual == expected, f"SHA-256 mismatch: {path}")
    return actual


def kernel_stage(name):
    # CUB includes its version and target architecture in an inline namespace.
    # Preserve the outer kernel allowlist after removing that implementation tag.
    symbol = name.replace("<unnamed>::", "").replace("(anonymous namespace)::", "")
    symbol = symbol.removeprefix("void ").split("(", 1)[0].split("<", 1)[0].strip()
    symbol = re.sub(r"^cub::CUB_[0-9]+_[0-9]+_NS::", "cub::", symbol)
    require(symbol in KERNEL_STAGES, f"unmapped CUDA kernel: {name}")
    return KERNEL_STAGES[symbol]


def union_ns(intervals):
    total, end = 0, None
    for start, stop in sorted(intervals):
        require(type(start) is int and type(stop) is int and stop > start,
                "GPU activity requires positive integer duration")
        total += stop - max(start, end) if end is not None and stop > end else \
            stop - start if end is None else 0
        end = max(end, stop) if end is not None else stop
    return total


def log_warnings(text, source):
    warnings = []
    # This site-startup error precedes the harness and is unrelated to CUDA
    # collection. Retain it rather than silently treating every traceback as safe.
    pattern = r"Error processing line \d+ of [^\n]*pyannote[^\n]*\.pth:\n.*?Remainder of file ignored"
    for match in re.finditer(pattern, text, flags=re.DOTALL):
        warnings.append({"source": source, "severity": "Warning", "text": match.group()})
    remaining = re.sub(pattern, "", text, flags=re.DOTALL)
    lowered = remaining.lower()
    require(not any(token in lowered for token in BAD_IMPORT +
                    ("traceback (most recent call last)", "error:", "[error]")),
            f"capture/import failure in {source}")
    warnings.extend({"source": source, "severity": "Warning", "text": line.strip()}
                    for line in remaining.splitlines() if "warning" in line.lower())
    return warnings


def rows(db, table, tables):
    if table not in tables:
        return []
    return [dict(row) for row in db.execute('SELECT * FROM "' + table + '"')]


def extract_sqlite(path, expected_report):
    """Read recorded device work; retain synchronization/events separately."""
    with sqlite3.connect(Path(path).resolve().as_uri() + "?mode=ro&immutable=1", uri=True) as db:
        db.row_factory = sqlite3.Row
        require(db.execute("PRAGMA quick_check").fetchone()[0] == "ok", "invalid SQLite export")
        tables = {row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        require("CUPTI_ACTIVITY_KIND_KERNEL" in tables, "no imported CUPTI kernel activity")
        strings = dict(db.execute("SELECT id,value FROM StringIds"))
        kernels = rows(db, "CUPTI_ACTIVITY_KIND_KERNEL", tables)
        require(kernels, "empty imported CUPTI kernel activity")
        processes = {row["globalPid"]: row for row in rows(db, "PROCESSES", tables)}
        pids = {row["globalPid"] for row in kernels}
        require(len(pids) == 1, "expected one captured CUDA process")
        global_pid = next(iter(pids))
        process = processes[global_pid]
        device_ids = {row["deviceId"] for row in kernels}
        require(len(device_ids) == 1, "expected one CUDA device")
        device_id = next(iter(device_ids))
        cuda_devices = [row for row in rows(db, "TARGET_INFO_CUDA_DEVICE", tables)
                        if row["pid"] == process["pid"] and row["cudaId"] == device_id]
        require(len(cuda_devices) == 1, "ambiguous CUDA ordinal to GPU mapping")
        cuda_device = cuda_devices[0]
        devices = [row for row in rows(db, "TARGET_INFO_GPU", tables)
                   if row["id"] == cuda_device["gpuId"] and row["uuid"] == cuda_device["uuid"]]
        require(len(devices) == 1, "CUDA ordinal does not identify a unique GPU")
        gpu = devices[0]
        device = {key: gpu[key] for key in ("name", "uuid", "computeMajor", "computeMinor", "smCount")}
        device["cuda_ordinal"] = device_id
        runtime = rows(db, "CUPTI_ACTIVITY_KIND_RUNTIME", tables)
        api_by_id = defaultdict(list)
        for row in runtime:
            # Nsight globalTid has its thread ID in the low 24 bits.
            if row["globalTid"] & ~0xFFFFFF == global_pid:
                row["api"] = strings[row["nameId"]]
                api_by_id[row["correlationId"]].append(row)
        activities, matched = [], set()
        for kind, table in (("kernel", "CUPTI_ACTIVITY_KIND_KERNEL"),
                            ("memset", "CUPTI_ACTIVITY_KIND_MEMSET"),
                            ("memcpy", "CUPTI_ACTIVITY_KIND_MEMCPY")):
            for row in rows(db, table, tables):
                require(row["globalPid"] == global_pid and row["deviceId"] == device_id,
                        "device work from another process/device in one-case capture")
                require(type(row["start"]) is int and type(row["end"]) is int and row["end"] > row["start"],
                        "nonpositive CUDA activity duration")
                name = strings[row["demangledName"]] if kind == "kernel" else kind
                calls = api_by_id[row["correlationId"]]
                if kind == "kernel":
                    calls = [call for call in calls if re.match(
                        r"cuda(?:Launch(?:Cooperative)?Kernel|GraphLaunch)(?:_|$)", call["api"])]
                require(len(calls) == 1 and calls[0]["returnValue"] == 0,
                        f"missing/ambiguous/failed host correlation for {name}")
                launch = calls[0]
                matched.add(row["correlationId"])
                item = {"kind": kind, "name": name,
                        "stage": kernel_stage(name) if kind == "kernel" else None,
                        "start_ns": row["start"], "end_ns": row["end"],
                        "duration_ns": row["end"] - row["start"],
                        "stream_id": row["streamId"], "correlation_id": row["correlationId"],
                        "graph_node_id": row.get("graphNodeId"), "graph_id": row.get("graphId"),
                        "grid": [row["grid" + axis] for axis in "XYZ"] if kind == "kernel" else None,
                        "block": [row["block" + axis] for axis in "XYZ"] if kind == "kernel" else None,
                        "host_launch": {"api": launch["api"], "start_ns": launch["start"],
                                        "end_ns": launch["end"], "correlation_id": launch["correlationId"]}}
                if kind == "kernel":
                    item.update(registers_per_thread=row.get("registersPerThread"),
                                static_shared_bytes=row.get("staticSharedMemory"),
                                dynamic_shared_bytes=row.get("dynamicSharedMemory"))
                else:
                    item.update(bytes=row["bytes"], value=row.get("value"), copy_kind=row.get("copyKind"))
                activities.append(item)
        require(all(correlation in matched for correlation, calls in api_by_id.items()
                    for call in calls if re.match(r"cudaLaunch(?:Cooperative)?Kernel(?:_|$)", call["api"])),
                "host kernel launches without imported device activities")
        diagnostics = rows(db, "DIAGNOSTIC_EVENT", tables)
        severity = {row["id"]: row["name"] for row in rows(db, "ENUM_DIAGNOSTIC_SEVERITY_LEVEL", tables)}
        warnings = []
        for row in diagnostics:
            level = severity.get(row["severity"], "Unknown")
            require(level in ("Info", "Warning") and not any(token in row["text"].lower() for token in BAD_IMPORT),
                    "Nsight diagnostic reports collection/import failure: " + row["text"])
            if level == "Warning":
                warnings.append({"source": "sqlite", "severity": level, "text": row["text"]})
        export = dict(db.execute("SELECT name,value FROM META_DATA_EXPORT"))
        require(PurePosixPath(export["EXPORT_PARAM_INPUT_FILE"]).name == expected_report,
                "SQLite export names a different Nsight report")
        activities.sort(key=lambda row: (row["start_ns"], row["end_ns"], row["kind"]))
        origin = activities[0]["start_ns"]
        for row in activities:
            for key in ("start_ns", "end_ns"):
                row[key] -= origin
                row["host_launch"][key] -= origin
        span = max(row["end_ns"] for row in activities)
        union = union_ns((row["start_ns"], row["end_ns"]) for row in activities)
        stage_ns = {stage: sum(row["duration_ns"] for row in activities if row["stage"] == stage)
                    for stage in STAGES}
        kernel_sum = sum(stage_ns.values())
        auxiliary = {table: len(rows(db, table, tables)) for table in
                     ("CUPTI_ACTIVITY_KIND_CUDA_EVENT", "CUPTI_ACTIVITY_KIND_SYNCHRONIZATION",
                      "CUPTI_ACTIVITY_KIND_GRAPH_HOST_NODE_AND_HOST_LAUNCH", "PROFILER_OVERHEAD")}
        return {"stage_ns": stage_ns, "kernel_sum_ns": kernel_sum, "gpu_span_ns": span,
                "gpu_activity_union_ns": union, "gap_ns": span - union,
                "memory_operation_ns": sum(row["duration_ns"] for row in activities if row["kind"] != "kernel"),
                "gpu_activities": activities, "kernel_activities": len(kernels),
                "matched_kernel_activities": len(kernels),
                "stream_ids": sorted({row["stream_id"] for row in activities}),
                "auxiliary_activity_counts": auxiliary, "warnings": warnings,
                "device": device, "export_metadata": export,
                "activity_origin_ns": origin, "captured_pid": process["pid"]}


def source_binding(stage, native, root):
    package = root / "src/cuda_zlib"
    actual = {path.relative_to(package).as_posix(): digest(path) for path in package.rglob("*")
              if path.is_file() and path.suffix in (".py", ".cu", ".cuh")}
    require(actual == stage["source_sha256"] == native["source_sha256"], "stale runtime source inventory")
    check_hash(root / "benchmarks/profile_resident.py", stage["harness_sha256"])
    check_hash(root / "benchmarks/benchmark.py", stage["benchmark_sha256"])
    generated = {}
    for filename, module in (("encoder.cuh", "_encode_kernels.py"),
                             ("decoder.cuh", "_decode_kernels.py"),
                             ("postprocess.cuh", "_postprocess.py")):
        assignments = [node.value for node in ast.parse((package / module).read_text()).body
                       if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and
                       target.id == "CUDA_SOURCE" for target in node.targets)]
        require(len(assignments) == 1, f"ambiguous CUDA_SOURCE in {module}")
        generated[filename] = hashlib.sha256(ast.literal_eval(assignments[0]).encode()).hexdigest()
    generated.update({name: digest(package / "native" / name)
                      for name in ("codec_ffi.cu", "batch_encode.cuh", "batch_decode.cuh")})
    require(generated == native["identity"]["sources"], "native build has different generated CUDA sources")
    require(hashlib.sha256(json.dumps(native["identity"], sort_keys=True).encode()).hexdigest() == native["cache_key"],
            "native cache key does not match its build identity")
    for key in ("library_sha256", "build_sha256"):
        require(re.fullmatch(r"[0-9a-f]{64}", native[key]) is not None, f"invalid {key}")


def extract(capture_dir, stage_path, native_path, revision, artifact_prefix,
            root=ROOT, reexported_sqlite=False):
    capture_dir, stage_path, native_path = map(Path, (capture_dir, stage_path, native_path))
    prefix = PurePosixPath(artifact_prefix)
    require(not prefix.is_absolute() and ".." not in prefix.parts, "artifact-prefix must be repo-relative")
    require(re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", revision), "full source revision required")
    stage, native = read_json(stage_path), read_json(native_path)
    require(stage["source_revision"] == revision, "stage revision differs from requested revision")
    source_binding(stage, native, root)
    # benchmark imports NumPy, but imports CUDA/JAX only inside GPU entrypoints.
    spec = importlib.util.spec_from_file_location("nsight_payloads", root / "benchmarks/benchmark.py")
    payloads = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(payloads)
    artifacts = {str(prefix / "STAGE.json"): digest(stage_path), str(prefix / "native.json"): digest(native_path)}
    cases, environment, native_build = [], None, None
    for receipt_path in sorted(capture_dir.glob("*-capture.json")):
        name = receipt_path.name.removesuffix("-capture.json")
        paths = {key: capture_dir / (name + suffix) for key, suffix in
                 (("raw_profile", "-profile.json"), ("command", "-command.json"),
                  ("capture_receipt", "-capture.json"), ("nsys_report", ".nsys-rep"),
                  ("log", ".log"), ("export_log", "-export.log"), ("sqlite", ".sqlite"))}
        receipt, profile, command = (read_json(paths[key]) for key in ("capture_receipt", "raw_profile", "command"))
        require(receipt["source_revision"] == revision, f"{name}: stale capture revision")
        require(datetime.fromisoformat(receipt["capture_started_utc"]).utcoffset() is not None,
                "capture timestamp must include timezone")
        for key in ("nsys_report", "log"):
            check_hash(paths[key], receipt["sha256"][paths[key].name])
        check_hash(paths["raw_profile"], receipt["profile_sha256"])
        sqlite_hash = digest(paths["sqlite"])
        original_sqlite_hash = receipt["sha256"][paths["sqlite"].name]
        require(reexported_sqlite or sqlite_hash == original_sqlite_hash, f"{name}: stale SQLite export")
        require(command == receipt["command"], f"{name}: command receipt mismatch")
        require(command[1] == "profile" and "--trace=cuda" in command and
                "--capture-range=cudaProfilerApi" in command and "--capture-range-end=stop" in command and
                "--cuda-graph-trace=node" in command and "--sample=none" in command and "--cpuctxsw=none" in command,
                f"{name}: expected CUDA-only profiler-range capture with graph nodes")
        require(profile["schema_version"] == 1 and profile["complete"] is True and
                profile["all_workflows_warmed_before_capture"] is True, f"{name}: incomplete resident profile")
        require(profile["source_sha256"] == stage["source_sha256"] and
                profile["harness_sha256"] == stage["harness_sha256"], f"{name}: stale profile sources")
        for key in ("cache_key", "library_sha256", "build_sha256", "identity"):
            require(profile["native_build"][key] == native[key], f"{name}: different native {key}")
        require(len(profile["cases"]) == 1, f"{name}: one case per trace required")
        case, args = profile["cases"][0], profile["arguments"]
        workload, size = case["workload"], case["input_bytes"]
        require(name == f"{workload}-{size}" and args["sizes"] == [size] and args["workloads"] == [workload]
                and args["samples"] == 1 and args["cuda_profiler_range"] is True,
                f"{name}: command/profile case mismatch")
        harness_indices = [i for i, value in enumerate(command) if value == "benchmarks/profile_resident.py"]
        require(len(harness_indices) == 1, f"{name}: missing/ambiguous resident harness command")
        expected_args = ["--sizes", str(size), "--workloads", workload, "--samples", "1",
                         "--seed", str(args["seed"]), "--cuda-profiler-range", "--output", args["output"]]
        require(command[harness_indices[0] + 1:] == expected_args and
                PurePosixPath(args["output"]).name == paths["raw_profile"].name and args["device"] == 0,
                f"{name}: launch arguments differ from one-case default-device profile")
        export_command = receipt["export_command"]
        require(export_command[1:4] == ["export", "--type", "sqlite"] and
                PurePosixPath(export_command[-1]).name == paths["nsys_report"].name and
                PurePosixPath(export_command[export_command.index("--output") + 1]).name == paths["sqlite"].name,
                f"{name}: export command names different files")
        architecture = native["identity"]["architecture"]
        require(profile["native_build"]["ffi_targets"] == [f"cuda_zlib_{action}_{architecture}" for action in
                ("compress", "decompress", "compress_batch", "decompress_batch")], f"{name}: different FFI targets")
        require(case["byte_exact"] is True and case["cpu_codec_forbidden"] is True,
                f"{name}: failed validation or CPU codec guard")
        require(len(case["seconds"]) == 1 and all(math.isfinite(v) and v > 0 for v in case["seconds"])
                and statistics.median(case["seconds"]) == case["median_seconds"], f"{name}: invalid timing record")
        raw = payloads.make_payload(workload, size, args["seed"])
        stream = zlib.compress(raw, 6)
        require(len(raw) == size and hashlib.sha256(raw).hexdigest() == case["input_sha256"],
                f"{name}: seeded payload differs from measured fixture")
        require(len(stream) == case["encoded_bytes"] and hashlib.sha256(stream).hexdigest() == case["stream_sha256"]
                and zlib.decompress(stream) == raw, f"{name}: stdlib input stream oracle differs")
        del raw, stream
        parsed = extract_sqlite(paths["sqlite"], paths["nsys_report"].name)
        require(digest(paths["sqlite"]) == sqlite_hash, f"{name}: SQLite changed during extraction")
        require(parsed["kernel_activities"] == receipt["kernel_activities"], f"{name}: kernel count differs from receipt")
        require(parsed["device"]["cuda_ordinal"] == args["device"], f"{name}: unexpected CUDA ordinal")
        warnings = parsed["warnings"]
        for key in ("log", "export_log"):
            warnings.extend(log_warnings(paths[key].read_text(), key))
        log = paths["log"].read_text()
        require("Capture range started in the application." in log and
                "Capture range ended in the application." in log and "Generated:" in log,
                f"{name}: missing completed capture range")
        provenance = {key: {"path": str(prefix / path.name), "sha256": digest(path)}
                      for key, path in paths.items() if key != "sqlite"}
        artifacts.update({item["path"]: item["sha256"] for item in provenance.values()})
        provenance["sqlite"] = {"filename": paths["sqlite"].name, "sha256": sqlite_hash,
                                "receipt_sha256": original_sqlite_hash, "private": True,
                                "reexported": sqlite_hash != original_sqlite_hash}
        provenance.update(source_revision=revision, harness_sha256=stage["harness_sha256"],
                          source_sha256=stage["source_sha256"], native_receipt_sha256=digest(native_path),
                          cache_key=native["cache_key"], library_sha256=native["library_sha256"],
                          build_sha256=native["build_sha256"])
        parsed.update(name=name, workload=workload, output_bytes=size, encoded_bytes=case["encoded_bytes"],
                      capture_started_utc=receipt["capture_started_utc"], seed=args["seed"],
                      input_sha256=case["input_sha256"], stream_sha256=case["stream_sha256"],
                      byte_exact=True, cpu_codec_forbidden=True, fixture_oracle_verified=True,
                      diagnostic_completed_call_ns=case["seconds"][0] * 1e9, provenance=provenance)
        if environment is None:
            environment, native_build = profile["environment"], profile["native_build"]
        else:
            require(all(profile["environment"][key] == environment[key] for key in
                        ("python", "jax", "jaxlib", "numpy", "backend", "cuda_platform", "zlib_runtime")),
                    "mixed runtime environments")
            require(profile["native_build"] == native_build, "mixed native builds")
            require(parsed["device"] == cases[0]["device"], "mixed captured GPUs")
        cases.append(parsed)
    require(cases, "no capture receipts found")
    require(len({row["name"] for row in cases}) == len(cases), "duplicate capture case")
    return {"schema": 1, "schema_version": 1, "complete": True, "source_revision": revision,
            "source_sha256": stage["source_sha256"], "harness_sha256": stage["harness_sha256"],
            "benchmark_sha256": stage["benchmark_sha256"], "extractor_sha256": digest(Path(__file__)),
            "native_build": native_build, "environment": environment, "device": cases[0]["device"],
            "created_utc": min(row["capture_started_utc"] for row in cases),
            "capture_manifest_sha256": digest(stage_path), "stages": list(STAGES),
            "methodology": "Imported CUDA-only Nsight Systems activity from one warmed resident JIT checked call per trace. "
                           "Kernel sums and stage shares include kernels only; memory operation durations are separate. "
                           "GPU span covers first to last imported kernel/memory activity. Gaps are span minus interval union, "
                           "without attributing them to CPU work. Synchronization, CUDA event and graph host activities are "
                           "counted separately. Guarded launches retain recorded duration even when they skip useful work. "
                           "Fused decoder kernels cannot be split into parsing/copy/checksum with Systems. "
                           "Single-call instrumented durations are diagnostic, not adoption timings. "
                           "Native binary identity is reported by the capture harness; the remote library is not rehashed here.",
            "warnings": [{"case": case["name"], **warning} for case in cases for warning in case["warnings"]],
            "artifact_sha256": dict(sorted(artifacts.items())), "cases": cases}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-dir", type=Path, required=True)
    parser.add_argument("--stage-receipt", type=Path, required=True)
    parser.add_argument("--native-receipt", type=Path, required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--artifact-prefix", default="benchmarks/results/nsight/captures")
    parser.add_argument("--reexported-sqlite", action="store_true",
                        help="allow a new SQLite digest, preserving both original and re-exported hashes")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = extract(args.capture_dir, args.stage_receipt, args.native_receipt,
                         args.source_revision, args.artifact_prefix, reexported_sqlite=args.reexported_sqlite)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    except (OSError, KeyError, TypeError, ValueError, sqlite3.Error) as error:
        print(f"Nsight extraction failed: {error}", file=sys.stderr)
        return 1
    print(f"Validated {len(result['cases'])} source-bound captures; "
          f"{sum(row['kernel_activities'] for row in result['cases'])} imported kernels.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
