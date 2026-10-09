#!/usr/bin/env python3
# Copyright (c) 2026 xangma
# SPDX-License-Identifier: MIT
"""Profile completed compression/decompression roundtrips and download subphases.

The worker uses fresh results for two warmups and twenty measured workflows.
Upload is enqueue time; the codec call completes output and metadata. Download
separately records metadata transfer, status/extent checks, output transfer,
host bytes allocation, and validation. Compression downloads the full padded
buffer, copies only the meaningful prefix to bytes, then independently validates
with stdlib zlib outside the CPU-codec-forbidden codec/transfer phases.

Omit --nvtx-library for controls without NVTX. --telemetry none avoids psutil and
NVML imports/queries. Process-tree telemetry reuses the immutable timeline
sampler, retaining actual per-query times, nulls and errors. Requested polling
does not establish the actual sampling cadence or NVML utilization resolution.
"""

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import operator
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import threading
import time
import traceback
import uuid


ROOT = Path(__file__).resolve().parents[1]
HELPER_PATH = Path(__file__).with_name("profile_timeline.py")
SPEC = importlib.util.spec_from_file_location("_workflow_timeline_helpers", HELPER_PATH)
helpers = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(helpers)
require, sha, write_json = helpers.require, helpers.sha, helpers.write_json
EVENT_PREFIX = helpers.EVENT_PREFIX
PHASE_SEMANTICS = {
    "prepare": {"kind": "host_only"}, "init": {"kind": "host_annotation"},
    "warmup": {"kind": "container"}, "measured_loop": {"kind": "container"},
    "upload": {"kind": "enqueue", "description": "device_put; no additional synchronization"},
    "codec": {"kind": "completed", "description": "selected codec operation; output and metadata ready"},
    "download_check": {"kind": "container"},
    "metadata_download": {"kind": "completed", "description": "np.asarray(metadata)"},
    "status_check": {"kind": "host_only", "description": "status and meaningful extent checked before output read"},
    "output_download": {"kind": "completed", "description": "np.asarray(full device output buffer)"},
    "host_bytes": {"kind": "host_only", "description": "meaningful host-array prefix copied to bytes"},
    "validate": {"kind": "host_only", "description": "independent oracle outside CPU codec guard"},
}


def source_identity(revision):
    base = helpers.source_identity(revision)
    base["harness_sha256"] = sha(__file__)
    base["dependencies_sha256"] = {str(path.relative_to(ROOT)): sha(path) for path in
                                    (HELPER_PATH, ROOT / "benchmarks/benchmark.py", ROOT / "benchmarks/profile_resident.py")}
    return base


class StageRecorder:
    def __init__(self, nvtx, run_id, operation):
        self.nvtx, self.run_id, self.operation, self.counter = nvtx, run_id, operation, 0

    @contextmanager
    def phase(self, name, parent_id=None, iteration=None, warmup=None):
        require(name in PHASE_SEMANTICS, "unknown workflow phase")
        self.counter += 1
        event_id = f"{self.run_id}:{self.counter}"
        label = f"cuda_zlib:{self.operation}:{name}:{event_id}" if self.nvtx else None
        common = {"id": event_id, "name": name, "parent_id": parent_id, "operation": self.operation,
                  "iteration": iteration, "warmup": warmup, "pid": os.getpid(), "nvtx_label": label,
                  "iteration_id": None if iteration is None else f"{'warmup' if warmup else 'measured'}-{iteration:03d}"}
        before = time.monotonic_ns()
        if self.nvtx:
            self.nvtx.library.nvtxRangePushA(label.encode())
        after = time.monotonic_ns()
        print(EVENT_PREFIX + json.dumps({**common, "event": "start", "timestamp_ns": before,
              "nvtx_before_ns": before if self.nvtx else None, "nvtx_after_ns": after if self.nvtx else None}),
              file=sys.stderr, flush=True)
        success = False
        try:
            yield event_id
            success = True
        finally:
            before = time.monotonic_ns()
            if self.nvtx:
                self.nvtx.library.nvtxRangePop()
            after = time.monotonic_ns()
            print(EVENT_PREFIX + json.dumps({**common, "event": "end", "timestamp_ns": after,
                  "nvtx_before_ns": before if self.nvtx else None, "nvtx_after_ns": after if self.nvtx else None,
                  "success": success}), file=sys.stderr, flush=True)


def parse_stage_line(line, pid, operation):
    if not line.startswith(EVENT_PREFIX):
        return None
    event = json.loads(line[len(EVENT_PREFIX):])
    require(event["pid"] == pid and event["operation"] == operation, "stage PID/operation differs")
    require(event["event"] in ("start", "end") and event["name"] in PHASE_SEMANTICS and
            type(event["timestamp_ns"]) is int and event["timestamp_ns"] > 0, "invalid workflow stage")
    return event


@contextmanager
def forbid_cpu_codec(zlib):
    originals = {name: getattr(zlib, name) for name in ("compress", "decompress", "compressobj", "decompressobj")}

    def forbidden(*a, **kw):
        raise AssertionError("CPU codec used in CUDA codec/transfer phase")

    for name in originals:
        setattr(zlib, name, forbidden)
    try:
        yield
    finally:
        for name, original in originals.items():
            setattr(zlib, name, original)


def checked_extent(operation, metadata, capacity, check_status):
    require(getattr(metadata, "ndim", 1) == 1 and len(metadata) == 2, "unexpected metadata shape")
    values = [operator.index(value) for value in metadata]
    require(all(0 <= value <= 0xFFFFFFFF for value in values), "invalid uint32 metadata")
    status = values[1] if operation == "compress" else values[0]
    check_status(status, "CUDA workflow " + operation)
    if operation == "compress":
        require(8 <= values[0] <= capacity, "encoded extent outside padded output")
        extent = values[0]
    else:
        require(values[1] == 0, "unexpected decompression metadata")
        extent = capacity
    return extent, values


def worker(args):
    result = {"schema_version": 1, "complete": False, "operation": args.operation, "workflow": "roundtrip",
              "pid": os.getpid(), "start_ns": time.monotonic_ns(), **source_identity(args.source_revision), "iterations": []}
    try:
        nvtx = helpers.NVTX(args.nvtx_library) if args.nvtx_library else None
        stages = StageRecorder(nvtx, args.run_id, args.operation)
        with stages.phase("prepare"):
            import numpy as np
            import zlib
            from benchmark import make_payload
            raw = make_payload(args.workload, args.size, args.seed)
            stream = zlib.compress(raw, 6) if args.operation == "decompress" else None
            if stream is not None:
                require(zlib.decompress(stream) == raw, "input stream oracle failed")
            operation_input = stream if stream is not None else raw
            host_input = np.frombuffer(operation_input, dtype=np.uint8)
            capacity = args.size + max(1, (args.size + args.chunk_bytes - 1) // args.chunk_bytes) * 5 + 6 \
                if args.operation == "compress" else args.size
            result["fixture"] = {"workload": args.workload, "output_bytes": args.size, "input_bytes": len(operation_input),
                                 "encoded_bytes": None if stream is None else len(stream), "seed": args.seed,
                                 "raw_sha256": hashlib.sha256(raw).hexdigest(),
                                 "input_sha256": hashlib.sha256(operation_input).hexdigest(),
                                 "operation_input_sha256": hashlib.sha256(operation_input).hexdigest(),
                                 "stream_sha256": None if stream is None else hashlib.sha256(stream).hexdigest(),
                                 "stdlib_stream_sha256": None if stream is None else hashlib.sha256(stream).hexdigest(),
                                 "stdlib_stream_bytes": None if stream is None else len(stream),
                                 "stdlib_level": None if stream is None else 6, "zlib_runtime": zlib.ZLIB_RUNTIME_VERSION,
                                 "chunk_bytes": args.chunk_bytes, "output_buffer_bytes": capacity}
        with stages.phase("init"):
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
                    "native build cache identity differs")
            result["native_build"] = {"cache_key": library_path.parent.name, "library_sha256": sha(library_path),
                                      "build_sha256": sha(build_path), "identity": identity, "ffi_targets": list(targets)}
            result["environment"] = {"python": sys.version, "jax": jax.__version__, "jaxlib": jaxlib.__version__,
                                     "numpy": np.__version__, "cuda_platform": device.client.platform_version,
                                     "device_kind": device.device_kind, "local_hardware_id": int(device.local_hardware_id),
                                     "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"), "backend": "JAX typed CUDA FFI"}
            if args.operation == "compress":
                codec = jax.jit(lambda value: cuda_zlib.compress_zlib_padded(value, device, chunk_bytes=args.chunk_bytes))
            else:
                codec = jax.jit(lambda value: cuda_zlib.decompress_zlib_checked(value, args.size, device))

        def workflow(iteration, warmup, parent_id):
            with forbid_cpu_codec(zlib):
                with stages.phase("upload", parent_id, iteration, warmup):
                    resident = jax.device_put(host_input, device)
                with stages.phase("codec", parent_id, iteration, warmup):
                    output, metadata = jax.block_until_ready(codec(resident))
            with stages.phase("download_check", parent_id, iteration, warmup) as download_id:
                with forbid_cpu_codec(zlib):
                    with stages.phase("metadata_download", download_id, iteration, warmup):
                        host_metadata = np.asarray(metadata)
                    with stages.phase("status_check", download_id, iteration, warmup):
                        extent, values = checked_extent(args.operation, host_metadata, capacity, _codec._check_value)
                        require(output.shape == (capacity,), "unexpected output buffer extent")
                    with stages.phase("output_download", download_id, iteration, warmup):
                        host_output = np.asarray(output)
                    with stages.phase("host_bytes", download_id, iteration, warmup):
                        meaningful = host_output[:extent].tobytes()
                with stages.phase("validate", download_id, iteration, warmup):
                    decoded = zlib.decompress(meaningful) if args.operation == "compress" else meaningful
                    require(decoded == raw, "byte-exact validation failed")
            result["iterations"].append({"operation": args.operation, "iteration": iteration, "warmup": warmup,
                "iteration_id": f"{'warmup' if warmup else 'measured'}-{iteration:03d}", "status": values, "status_code": 0,
                "encoded_length": extent if args.operation == "compress" else len(stream), "output_bytes": args.size,
                "downloaded_bytes": capacity, "host_bytes_bytes": extent, "byte_exact": True, "cpu_codec_forbidden": True,
                "cpu_codec_forbidden_phases": ["upload", "codec", "metadata_download", "status_check", "output_download", "host_bytes"],
                "validation_outside_cpu_codec_guard": True,
                "validation_oracle": "stdlib.zlib.decompress" if args.operation == "compress" else "byte_compare"})

        with stages.phase("warmup") as parent_id:
            for iteration in range(args.warmups):
                workflow(iteration, True, parent_id)
        with stages.phase("measured_loop") as parent_id:
            for iteration in range(args.iterations):
                workflow(iteration, False, parent_id)
        require(len(result["iterations"]) == args.warmups + args.iterations, "missing validated workflows")
        current = source_identity(args.source_revision)
        require(current == {key: result[key] for key in current}, "source changed during worker execution")
        require(sha(library_path) == result["native_build"]["library_sha256"] and
                sha(build_path) == result["native_build"]["build_sha256"], "native build changed during execution")
        result["complete"] = True
        code = 0
    except BaseException as error:
        result["error"] = {"type": type(error).__name__, "message": str(error)}
        traceback.print_exc()
        code = 1
    finally:
        result["end_ns"] = time.monotonic_ns()
        write_json(args.worker_result, result)
    return code


def validate_iterations(result, args):
    rows = result["iterations"]
    expected = {(warmup, i) for warmup, count in ((True, args.warmups), (False, args.iterations)) for i in range(count)}
    require(len(rows) == len(expected) and {(row["warmup"], row["iteration"]) for row in rows} == expected,
            "missing validated worker iterations")
    require(len({row["iteration_id"] for row in rows}) == len(rows), "duplicate iteration IDs")
    for row in rows:
        require(row["operation"] == args.operation and row["status_code"] == 0 and row["byte_exact"] is True and
                row["cpu_codec_forbidden"] is True and row["validation_outside_cpu_codec_guard"] is True,
                "worker validation failed")
        if args.operation == "compress":
            require(row["status"] == [row["encoded_length"], 0] and 8 <= row["encoded_length"] <= row["downloaded_bytes"] and
                    row["host_bytes_bytes"] == row["encoded_length"] and row["validation_oracle"] == "stdlib.zlib.decompress",
                    "invalid compression result")
        else:
            require(row["status"] == [0, 0] and row["downloaded_bytes"] == row["host_bytes_bytes"] == args.size and
                    row["validation_oracle"] == "byte_compare", "invalid decompression result")


def supervisor(args):
    result = {"schema_version": 1, "complete": False, **source_identity(args.source_revision),
              "operation": args.operation, "workflow": "roundtrip", "phase_semantics": PHASE_SEMANTICS,
              "metadata_layout": ["encoded_length", "status"] if args.operation == "compress" else ["status", "reserved_zero"],
              "created_utc": datetime.now(timezone.utc).isoformat(), "collector_pid": os.getpid(),
              "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
              "start_ns": time.monotonic_ns(), "samples": [], "stage_events": [], "errors": [],
              "clock_anchor": None, "telemetry_versions": {}, "nvtx_enabled": args.nvtx_library is not None,
              "selected_gpu": {"ordinal": args.device, "uuid": args.gpu_uuid,
                               "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")}}
    output = args.output.resolve()
    worker_result = output.with_name(output.stem + "-worker.json")
    worker_log = output.with_name(output.stem + "-worker.log")
    require(not output.exists() and not worker_result.exists() and not worker_log.exists(), "use fresh output paths")
    output.parent.mkdir(parents=True, exist_ok=True)
    run_id = uuid.uuid4().hex
    process, nvml, nvml_ready, identities, event_errors, threads = None, None, False, {}, [], []
    log_lock = threading.Lock()

    def cancel(signum, frame):
        raise InterruptedError(f"collector received signal {signum}")

    previous_term = signal.signal(signal.SIGTERM, cancel)
    try:
        if args.telemetry == "process-tree":
            import psutil
            import pynvml as nvml
            nvml.nvmlInit()
            nvml_ready = True
            handle = nvml.nvmlDeviceGetHandleByUUID(args.gpu_uuid)
            result["telemetry_versions"] = {"psutil": psutil.__version__, "nvidia-ml-py": importlib.metadata.version("nvidia-ml-py")}
            for key, value in (("uuid", nvml.nvmlDeviceGetUUID(handle)), ("name", nvml.nvmlDeviceGetName(handle)),
                               ("driver_version", nvml.nvmlSystemGetDriverVersion())):
                result["selected_gpu"][key] = value.decode() if isinstance(value, bytes) else value
        if args.nvtx_library:
            result["clock_anchor"] = helpers.NVTX(args.nvtx_library).mark("cuda_zlib:workflow-origin:" + run_id)
        command = [sys.executable, str(Path(__file__).resolve()), "--worker", "--worker-result", str(worker_result),
                   "--run-id", run_id, "--operation", args.operation, "--source-revision", args.source_revision,
                   "--gpu-uuid", args.gpu_uuid, "--device", str(args.device), "--size", str(args.size), "--workload", args.workload,
                   "--seed", str(args.seed), "--chunk-bytes", str(args.chunk_bytes), "--warmups", str(args.warmups),
                   "--iterations", str(args.iterations), "--telemetry", "none"]
        if args.nvtx_library:
            command += ["--nvtx-library", str(args.nvtx_library)]
        launch_start = time.monotonic_ns()
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                                   bufsize=1, start_new_session=True)
        result["worker"] = {"pid": process.pid, "command": command, "launch_start_ns": launch_start,
                            "start_ns": time.monotonic_ns(), "result_path": worker_result.name, "log_path": worker_log.name}

        def consume(pipe, stream, log):
            for line in pipe:
                received = time.monotonic_ns()
                with log_lock:
                    log.write(f"{received} {stream} {line}")
                    log.flush()
                if stream == "stderr":
                    try:
                        event = parse_stage_line(line, process.pid, args.operation)
                        if event is not None:
                            result["stage_events"].append({**event, "received_ns": received})
                    except (ValueError, KeyError, TypeError) as error:
                        event_errors.append(str(error))
            pipe.close()

        with worker_log.open("w") as log:
            for stream in ("stdout", "stderr"):
                thread = threading.Thread(target=consume, args=(getattr(process, stream), stream, log), daemon=True)
                thread.start()
                threads.append(thread)
            deadline = launch_start + int(args.timeout * 1e9)
            interval = int(args.sample_ms * 1e6)
            next_sample = time.monotonic_ns()
            try:
                while process.poll() is None:
                    start = time.monotonic_ns()
                    if args.telemetry == "process-tree":
                        processes, tree_query = helpers.sample_processes(psutil, process.pid, identities)
                        gpu = helpers.sample_gpu(nvml, handle, {row["pid"] for row in processes})
                        gpu["uuid"] = result["selected_gpu"]["uuid"]
                        end = time.monotonic_ns()
                        result["samples"].append({"timestamp_ns": (start + end) // 2, "query_start_ns": start,
                                                  "query_end_ns": end, "processes": processes,
                                                  "process_tree_query": tree_query, "gpu": gpu})
                    else:
                        end = time.monotonic_ns()
                    require(end < deadline, "worker timeout")
                    next_sample = max(next_sample + interval, end)
                    time.sleep(max(0, (next_sample - time.monotonic_ns()) / 1e9))
            finally:
                helpers.stop_worker(process)
                process.wait()
                for thread in threads:
                    thread.join(timeout=5)
                require(not any(thread.is_alive() for thread in threads), "worker log reader did not finish")
        result["worker"].update(exit_code=process.returncode, end_ns=time.monotonic_ns(),
                               process_ids=sorted({process.pid} | {p["pid"] for p in identities.values()}),
                               process_identities=list(identities.values()), log_sha256=sha(worker_log))
        require(not event_errors, "; ".join(event_errors))
        result["phase_intervals"] = helpers.phase_intervals(result["stage_events"])
        require(all(row["success"] for row in result["phase_intervals"]), "worker phase failed")
        completed = json.loads(worker_result.read_text())
        result["worker"].update(result_sha256=sha(worker_result), result=completed)
        require(process.returncode == 0 and completed["complete"] is True and completed["pid"] == process.pid,
                "worker did not finish successfully")
        identity = source_identity(args.source_revision)
        require(all(completed[key] == result[key] == identity[key] for key in identity), "worker/collector/source identities differ")
        validate_iterations(completed, args)
        result.update(native_build=completed["native_build"], environment=completed["environment"], fixture=completed["fixture"])
        result["methodology"] = {"telemetry": args.telemetry, "sample_interval_requested_ns": interval,
            "cpu": "cumulative process-tree times;100%onecore; use per-process query bounds",
            "gpu": "UUID-selected whole-device memory/utilization; compute memory separately PID-scoped; use per-metric query bounds",
            "nvml_utilization": "native reporting window; requested polling does not establish observed cadence",
            "compression_download": "full padded output array; only meaningful prefix allocated as bytes after metadata validation",
            "cpu_codec_guard": "upload,codec,metadata_download,status_check,output_download,host_bytes; validation outside guard",
            "validation": "compression stdlib decompression/byte compare; decompression byte compare; reported separately",
            "scope": "instrumented diagnostic roundtrip; fresh results; no affinity/numerical-thread environment forced"}
        result["complete"] = True
    except BaseException as error:
        result["errors"].append({"type": type(error).__name__, "message": str(error)})
        if process is not None:
            helpers.stop_worker(process)
            if "worker" in result:
                result["worker"].update(exit_code=process.returncode, end_ns=time.monotonic_ns())
        traceback.print_exc()
    finally:
        signal.signal(signal.SIGTERM, previous_term)
        if nvml_ready:
            nvml.nvmlShutdown()
        result["end_ns"] = time.monotonic_ns()
        write_json(output, result)
    return 0 if result["complete"] else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--operation", choices=("compress", "decompress"), required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--gpu-uuid", required=True)
    parser.add_argument("--nvtx-library", type=Path)
    parser.add_argument("--telemetry", choices=("none", "process-tree"), default="process-tree")
    parser.add_argument("--size", type=int, default=64 * 1024**2)
    parser.add_argument("--workload", choices=("zeros", "text", "uint32", "float32", "random"), default="float32")
    parser.add_argument("--seed", type=int, default=20261008)
    parser.add_argument("--chunk-bytes", type=int, default=32768)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--sample-ms", type=float, default=10)
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--worker-result", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--run-id", help=argparse.SUPPRESS)
    args = parser.parse_args()
    require(re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", args.source_revision), "full source revision required")
    require(0 < args.size < 256 * 1024**2 and args.warmups >= 1 and args.iterations >= 1 and args.device >= 0,
            "positive bounded size/count/device required")
    require(256 <= args.chunk_bytes <= 65535, "chunk-bytes must be in[256,65535]")
    require(math.isfinite(args.sample_ms) and args.sample_ms > 0 and math.isfinite(args.timeout) and args.timeout > 0,
            "positive finite sampling interval/timeout required")
    require((args.worker and args.worker_result is not None and args.run_id) or
            (not args.worker and args.output is not None), "output path required")
    return worker(args) if args.worker else supervisor(args)


if __name__ == "__main__":
    sys.exit(main())
