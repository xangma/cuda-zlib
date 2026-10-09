#!/usr/bin/env python3
# Copyright (c) 2026 xangma
# SPDX-License-Identifier: MIT
"""Diagnose fresh NumPy/pinned downloads and a queue bounded to two results.

Inputs stay resident; every case invokes the JIT codec for a fresh result.
Metadata is checked before any output host transfer. Serial modes complete the
codec before downloading. depth2pinned enqueues the next independent codec call
before consuming the previous result, with at most two live results. This is a
scheduling experiment, not proof of GPU overlap. Compare queued batch wall time
with serial batch wall time; per-case queued latency includes waiting in queue.

Each phase reports wall/thread CPU time and process page-fault deltas. Concurrent
JAX activity can contribute to process fault counts. Optional thread fault
deltas are also retained. A separate cache probe reads the same fresh result
twice and is excluded from workflow counts and summaries. No CPU comparisons,
affinity or numerical-thread environment changes are made.
"""

import argparse
from collections import deque
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import resource
import statistics
import sys
import time
import traceback


ROOT = Path(__file__).resolve().parents[1]
WORKFLOW_PATH = Path(__file__).with_name("profile_workflow.py")
SPEC = importlib.util.spec_from_file_location("_download_workflow_helpers", WORKFLOW_PATH)
workflow = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(workflow)
require, sha, write_json = workflow.require, workflow.sha, workflow.write_json
MODES = ("numpy", "npinned", "depth2pinned")


def source_identity(revision):
    result = workflow.source_identity(revision)
    result["harness_sha256"] = sha(__file__)
    result["dependencies_sha256"]["benchmarks/profile_workflow.py"] = sha(WORKFLOW_PATH)
    return result


def snapshot():
    process = resource.getrusage(resource.RUSAGE_SELF)
    thread = resource.getrusage(resource.RUSAGE_THREAD) if hasattr(resource, "RUSAGE_THREAD") else None
    return {"thread_cpu_ns": time.thread_time_ns(), "minor_faults": process.ru_minflt,
            "major_faults": process.ru_majflt,
            "thread_minor_faults": None if thread is None else thread.ru_minflt,
            "thread_major_faults": None if thread is None else thread.ru_majflt}


@contextmanager
def phase(row, name):
    start = time.monotonic_ns()
    before = snapshot()
    success = False
    try:
        yield
        success = True
    finally:
        after = snapshot()
        end = time.monotonic_ns()
        row["phases"].append({"name": name, "start_ns": start, "end_ns": end,
                              "wall_ns": end - start, "success": success,
                              **{key: None if before[key] is None or after[key] is None else after[key] - before[key]
                                 for key in before}})


def run_queue(count, launch, consume, depth, cleanup=None):
    """Hold at most depth independent results, including the current consumer."""
    require(depth in (1, 2) and count >= 0, "queue depth must be one or two")
    pending, rows, next_index, max_pending, current = deque(), [], 0, 0, None
    try:
        while next_index < count or pending:
            while next_index < count and len(pending) < depth:
                pending.append(launch(next_index))
                next_index += 1
                max_pending = max(max_pending, len(pending))
            current = pending.popleft()
            rows.append(consume(current))
            current = None
        return rows, max_pending
    except BaseException:
        if cleanup:
            for item in ([current] if current is not None else []) + list(pending):
                cleanup(item)
        raise


def summary(rows, elapsed_ns, depth):
    by_phase = {}
    for name in sorted({p["name"] for row in rows for p in row["phases"]}):
        selected = [p for row in rows for p in row["phases"] if p["name"] == name]
        by_phase[name] = {key: {"median": statistics.median([p[key] for p in selected if p[key] is not None]),
                               "sum": sum(p[key] for p in selected if p[key] is not None)}
                          for key in ("wall_ns", "thread_cpu_ns", "minor_faults", "major_faults")}
    return {"cases": len(rows), "queue_depth": depth, "batch_wall_ns": elapsed_ns,
            "batch_wall_per_case_ns": elapsed_ns / len(rows),
            "case_latency_median_ns": statistics.median(row["end_ns"] - row["start_ns"] for row in rows),
            "phase_metrics": by_phase}


def operation_report(args, operation, device, jax, np, zlib, cuda_zlib, _codec):
    from benchmark import make_payload
    raw = make_payload(args.workload, args.size, args.seed)
    stream = zlib.compress(raw, 6) if operation == "decompress" else None
    if stream is not None:
        require(zlib.decompress(stream) == raw, "input stream oracle failed")
    encoded_input = stream if stream is not None else raw
    resident = jax.device_put(np.frombuffer(encoded_input, dtype=np.uint8), device)
    resident.block_until_ready()
    capacity = args.size + max(1, (args.size + args.chunk_bytes - 1) // args.chunk_bytes) * 5 + 6 \
        if operation == "compress" else args.size
    codec = jax.jit(lambda value: cuda_zlib.compress_zlib_padded(value, device, chunk_bytes=args.chunk_bytes)) \
        if operation == "compress" else jax.jit(lambda value: cuda_zlib.decompress_zlib_checked(value, args.size, device))
    pinned = jax.sharding.SingleDeviceSharding(device, memory_kind="pinned_host")
    result = {"operation": operation, "fixture": {"workload": args.workload, "seed": args.seed,
              "output_bytes": args.size, "input_bytes": len(encoded_input), "output_buffer_bytes": capacity,
              "raw_sha256": hashlib.sha256(raw).hexdigest(), "input_sha256": hashlib.sha256(encoded_input).hexdigest(),
              "stdlib_stream_sha256": None if stream is None else hashlib.sha256(stream).hexdigest(),
              "chunk_bytes": args.chunk_bytes}, "modes": [], "cache_probe": None}

    def launch(index, mode, warmup):
        row = {"iteration": index, "warmup": warmup, "mode": mode, "operation": operation,
               "start_ns": time.monotonic_ns(), "phases": []}
        with workflow.forbid_cpu_codec(zlib):
            with phase(row, "codec_enqueue" if mode == "depth2pinned" else "codec_completed"):
                outputs = codec(resident)
                if mode != "depth2pinned":
                    outputs = jax.block_until_ready(outputs)
        return {"row": row, "outputs": outputs}

    def consume(item):
        row = item["row"]
        output, metadata = item["outputs"]
        with workflow.forbid_cpu_codec(zlib):
            with phase(row, "metadata_download"):
                host_metadata = np.asarray(metadata)
            with phase(row, "status_check"):
                extent, values = workflow.checked_extent(operation, host_metadata, capacity, _codec._check_value)
                require(output.shape == (capacity,), "unexpected output extent")
            with phase(row, "output_download"):
                if row["mode"] == "numpy":
                    host = np.asarray(output)
                else:
                    placed = jax.device_put(output, pinned)
                    host = np.asarray(placed.block_until_ready())
            with phase(row, "host_bytes"):
                data = host[:extent].tobytes()
        with phase(row, "validation"):
            decoded = zlib.decompress(data) if operation == "compress" else data
            require(decoded == raw, "byte-exact validation failed")
        row.update(status=values, status_code=0, byte_exact=True, codec_transfer_cpu_codec_forbidden=True,
                   validation_outside_cpu_codec_guard=True, downloaded_bytes=capacity, host_bytes_bytes=extent,
                   validation_oracle="stdlib.zlib.decompress" if operation == "compress" else "byte_compare",
                   host_array_writeable=bool(host.flags.writeable), host_array_owndata=bool(host.flags.owndata),
                   host_array_base_type=type(host.base).__name__, end_ns=time.monotonic_ns())
        return row

    def cleanup(item):
        # Wait only; failed metadata must never trigger an output host transfer.
        jax.block_until_ready(item["outputs"])

    for mode in args.modes:
        depth = 2 if mode == "depth2pinned" else 1
        warmups, warm_max = run_queue(args.warmups, lambda i: launch(i, mode, True), consume, depth, cleanup)
        start = time.monotonic_ns()
        rows, max_pending = run_queue(args.iterations, lambda i: launch(i, mode, False), consume, depth, cleanup)
        elapsed = time.monotonic_ns() - start
        result["modes"].append({"name": mode, "queue_depth": depth, "max_live_results": max(warm_max, max_pending),
                                "warmups": warmups, "cases": rows, "summary": summary(rows, elapsed, depth)})

    probe = {"phases": [], "excluded_from_workflows": True}
    with workflow.forbid_cpu_codec(zlib):
        with phase(probe, "codec_completed"):
            output, metadata = jax.block_until_ready(codec(resident))
        with phase(probe, "metadata_download"):
            host_metadata = np.asarray(metadata)
        with phase(probe, "status_check"):
            extent, values = workflow.checked_extent(operation, host_metadata, capacity, _codec._check_value)
        with phase(probe, "fresh_np_asarray"):
            first = np.asarray(output)
        with phase(probe, "repeated_np_asarray_same_result"):
            repeated = np.asarray(output)
        with phase(probe, "host_bytes"):
            data = repeated[:extent].tobytes()
    with phase(probe, "validation"):
        decoded = zlib.decompress(data) if operation == "compress" else data
        require(decoded == raw and np.array_equal(first, repeated), "cache probe byte oracle failed")
    probe.update(status=values, byte_exact=True, same_numpy_object=first is repeated,
                 arrays_share_memory=bool(np.shares_memory(first, repeated)), downloaded_bytes=capacity,
                 host_bytes_bytes=extent)
    result["cache_probe"] = probe
    return result


def run(args):
    report = {"schema_version": 1, "complete": False, **source_identity(args.source_revision),
              "created_utc": datetime.now(timezone.utc).isoformat(), "pid": os.getpid(),
              "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
              "start_ns": time.monotonic_ns(), "operations": []}
    require(not args.output.exists(), "use a fresh output path")
    try:
        import numpy as np
        import zlib
        import jax
        import jaxlib
        import cuda_zlib
        from cuda_zlib import _codec, _ffi
        device = _codec._select_device(args.device)
        cuda_zlib.compile_kernels(device)
        library, targets = _ffi._backend(device)
        library_path = Path(library._name).resolve()
        build_path = library_path.with_name("build.json")
        identity = json.loads(build_path.read_text())
        require(hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest() == library_path.parent.name,
                "native cache identity differs")
        report["native_build"] = {"cache_key": library_path.parent.name, "library_sha256": sha(library_path),
                                  "build_sha256": sha(build_path), "identity": identity, "ffi_targets": list(targets)}
        report["environment"] = {"python": sys.version, "jax": jax.__version__, "jaxlib": jaxlib.__version__,
                                 "numpy": np.__version__, "zlib_runtime": zlib.ZLIB_RUNTIME_VERSION,
                                 "device_kind": device.device_kind, "local_hardware_id": int(device.local_hardware_id),
                                 "cuda_platform": device.client.platform_version,
                                 "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")}
        for operation in (("compress", "decompress") if args.operation == "both" else (args.operation,)):
            report["operations"].append(operation_report(args, operation, device, jax, np, zlib, cuda_zlib, _codec))
        current = source_identity(args.source_revision)
        require(current == {key: report[key] for key in current}, "source changed during diagnostics")
        require(sha(library_path) == report["native_build"]["library_sha256"] and
                sha(build_path) == report["native_build"]["build_sha256"], "native build changed during diagnostics")
        report["methodology"] = {"scope": "diagnostic resident codec + host-array + bytes + independent validation; no CPU comparison",
            "fresh_results": "new JIT result per case; one resident immutable input per operation",
            "numpy": "serial completed codec, then np.asarray of full fresh result",
            "npinned": "serial completed codec, then device_put to SingleDeviceSharding(memory_kind=pinned_host), block and np.asarray",
            "depth2pinned": "at most two independent GPU results; enqueue next codec before consuming prior; metadata before each pinned output copy",
            "comparison": "queue batch wall time and per-case batch throughput differ from serial/queued case latency; GPU overlap unproven without trace",
            "faults": "minor/major_faults are process-wide deltas and can include concurrent JAX work; thread fault deltas separately recorded when supported",
            "cpu": "thread_time_ns for calling thread; wall times include waiting",
            "compression": "full padded buffer transferred; meaningful prefix copied to bytes after metadata check; stdlib validation outside guard",
            "cache_probe": "separate fresh result read twice; excluded from warmup/measured summaries",
            "order": "fixed operation/mode order, separately warmed; allocation/cache/device-state effects remain possible",
            "environment": "no affinity or numerical-thread environment forced"}
        report["complete"] = True
        code = 0
    except BaseException as error:
        report["error"] = {"type": type(error).__name__, "message": str(error)}
        traceback.print_exc()
        code = 1
    finally:
        report["end_ns"] = time.monotonic_ns()
        write_json(args.output, report)
    return code


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--operation", choices=("both", "compress", "decompress"), default="both")
    parser.add_argument("--modes", choices=MODES, nargs="+", default=list(MODES))
    parser.add_argument("--size", type=int, default=64 * 1024**2)
    parser.add_argument("--iterations", type=int, default=12)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--workload", choices=("zeros", "text", "uint32", "float32", "random"), default="float32")
    parser.add_argument("--seed", type=int, default=20261008)
    parser.add_argument("--chunk-bytes", type=int, default=32768)
    args = parser.parse_args()
    require(re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", args.source_revision), "full source revision required")
    require(0 < args.size < 256 * 1024**2 and args.iterations > 0 and args.warmups > 0 and args.device >= 0,
            "positive bounded size/count/device required")
    require(256 <= args.chunk_bytes <= 65535 and len(set(args.modes)) == len(args.modes), "invalid chunks or duplicate modes")
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
