#!/usr/bin/env python3
# Copyright (c) 2026 xangma
# SPDX-License-Identifier: MIT
"""Collect process-tree CPU/RSS and UUID-selected NVML telemetry around a worker.

Run the supervisor under Nsight Systems with CUDA/NVTX process-tree tracing.
The worker prepares one stdlib level-6 stream, initializes the backend, performs
two full warmups and twenty completed upload/decode/download/check workflows.
Upload ranges describe enqueue time; decode ranges complete output and status.
Status is checked before output bytes. CPU codec calls are forbidden throughout
each workflow. The supervisor requires psutil and nvidia-ml-py; the worker needs
the package's CUDA dependencies. All samples use actual monotonic timestamps;
10 ms polling does not change NVML's native utilization reporting window.
"""

import argparse
from contextlib import contextmanager
import ctypes
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
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
EVENT_PREFIX = "CUDA_ZLIB_STAGE_EVENT "
PHASES = {"prepare", "init", "warmup", "measured_loop", "upload", "decode", "download_check"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def source_identity(revision):
    package = ROOT / "src/cuda_zlib"
    return {"source_revision": revision, "harness_sha256": sha(__file__),
            "profile_resident_sha256": sha(ROOT / "benchmarks/profile_resident.py"),
            "benchmark_sha256": sha(ROOT / "benchmarks/benchmark.py"),
            "source_sha256": {p.relative_to(package).as_posix(): sha(p)
                              for p in sorted(package.rglob("*"))
                              if p.is_file() and p.suffix in (".py", ".cu", ".cuh")}}


class NVTX:
    def __init__(self, path):
        self.library = ctypes.CDLL(str(path))
        self.library.nvtxMarkA.argtypes = [ctypes.c_char_p]
        self.library.nvtxMarkA.restype = None
        self.library.nvtxRangePushA.argtypes = [ctypes.c_char_p]
        self.library.nvtxRangePushA.restype = ctypes.c_int
        self.library.nvtxRangePop.argtypes = []
        self.library.nvtxRangePop.restype = ctypes.c_int

    def mark(self, label):
        before = time.monotonic_ns()
        self.library.nvtxMarkA(label.encode())
        after = time.monotonic_ns()
        return {"label": label, "monotonic_before_ns": before, "monotonic_after_ns": after,
                "midpoint_ns": (before + after) // 2, "half_bracket_ns": (after - before) / 2,
                "wall_time_ns": time.time_ns(), "kind": "NVTX instant"}


class StageRecorder:
    def __init__(self, nvtx, run_id):
        self.nvtx, self.run_id, self.counter = nvtx, run_id, 0

    @contextmanager
    def phase(self, name, parent_id=None, iteration=None, warmup=None):
        require(name in PHASES, "unknown worker phase")
        self.counter += 1
        event_id = f"{self.run_id}:{self.counter}"
        iteration_id = None if iteration is None else f"{'warmup' if warmup else 'measured'}-{iteration:03d}"
        label = f"cuda_zlib:{name}:{event_id}"
        common = {"id": event_id, "name": name, "parent_id": parent_id, "iteration": iteration,
                  "iteration_id": iteration_id, "warmup": warmup, "pid": os.getpid(),
                  "nvtx_label": label}
        before = time.monotonic_ns()
        self.nvtx.library.nvtxRangePushA(label.encode())
        after = time.monotonic_ns()
        print(EVENT_PREFIX + json.dumps({**common, "event": "start", "timestamp_ns": before,
              "nvtx_before_ns": before, "nvtx_after_ns": after}), file=sys.stderr, flush=True)
        success = False
        try:
            yield event_id
            success = True
        finally:
            before = time.monotonic_ns()
            self.nvtx.library.nvtxRangePop()
            after = time.monotonic_ns()
            print(EVENT_PREFIX + json.dumps({**common, "event": "end", "timestamp_ns": after,
                  "nvtx_before_ns": before, "nvtx_after_ns": after, "success": success}),
                  file=sys.stderr, flush=True)


def parse_stage_line(line, expected_pid):
    if not line.startswith(EVENT_PREFIX):
        return None
    event = json.loads(line[len(EVENT_PREFIX):])
    require(event["pid"] == expected_pid, "stage event belongs to another PID")
    require(event["event"] in ("start", "end") and event["name"] in PHASES, "invalid stage event")
    require(isinstance(event["id"], str) and event["id"] and
            type(event["timestamp_ns"]) is int and event["timestamp_ns"] > 0, "invalid stage identity/time")
    return event


def phase_intervals(events):
    active, intervals, seen = {}, [], set()
    for event in events:
        event_id = event["id"]
        if event["event"] == "start":
            require(event_id not in seen, "duplicate stage start")
            seen.add(event_id)
            active[event_id] = event
        else:
            require(event_id in active, "stage end without start")
            start = active.pop(event_id)
            require(all(start.get(key) == event.get(key) for key in
                        ("name", "parent_id", "iteration", "iteration_id", "warmup", "pid", "nvtx_label")),
                    "stage start/end identity differs")
            require(event["timestamp_ns"] >= start["timestamp_ns"], "stage clock reversed")
            intervals.append({key: value for key, value in start.items() if key not in
                              ("event", "timestamp_ns", "nvtx_before_ns", "nvtx_after_ns", "received_ns")}
                             | {"start_ns": start["timestamp_ns"], "end_ns": event["timestamp_ns"],
                                "success": event.get("success") is True,
                                "nvtx_start_bracket_ns": [start["nvtx_before_ns"], start["nvtx_after_ns"]],
                                "nvtx_end_bracket_ns": [event["nvtx_before_ns"], event["nvtx_after_ns"]]})
    require(not active, "unfinished worker stages")
    indexed = {row["id"]: row for row in intervals}
    for row in intervals:
        if row["parent_id"] is not None:
            parent = indexed[row["parent_id"]]
            require(parent["start_ns"] <= row["start_ns"] <= row["end_ns"] <= parent["end_ns"],
                    "child stage outside parent")
    return sorted(intervals, key=lambda row: (row["start_ns"], row["end_ns"]))


def worker(args):
    result = {"schema_version": 1, "complete": False, "pid": os.getpid(),
              "start_ns": time.monotonic_ns(), **source_identity(args.source_revision), "iterations": []}
    try:
        stages = StageRecorder(NVTX(args.nvtx_library), args.run_id)
        with stages.phase("prepare"):
            import numpy as np
            import zlib
            from benchmark import make_payload
            raw = make_payload(args.workload, args.size, args.seed)
            stream = zlib.compress(raw, 6)
            require(zlib.decompress(stream) == raw, "input stream oracle failed")
            host_input = np.frombuffer(stream, dtype=np.uint8)
            result["fixture"] = {"workload": args.workload, "output_bytes": len(raw), "encoded_bytes": len(stream),
                                 "seed": args.seed, "input_sha256": hashlib.sha256(raw).hexdigest(),
                                 "stream_sha256": hashlib.sha256(stream).hexdigest(), "stdlib_level": 6,
                                 "zlib_runtime": zlib.ZLIB_RUNTIME_VERSION}
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
                                     "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                                     "backend": "JAX typed CUDA FFI"}
            decode = jax.jit(lambda value: cuda_zlib.decompress_zlib_checked(value, args.size, device))

        originals = {name: getattr(zlib, name) for name in ("compress", "decompress", "compressobj", "decompressobj")}

        def forbidden(*a, **kw):
            raise AssertionError("CPU codec used in CUDA workflow")

        def workflow(iteration, warmup, parent_id):
            for name in originals:
                setattr(zlib, name, forbidden)
            try:
                with stages.phase("upload", parent_id, iteration, warmup):
                    resident = jax.device_put(host_input, device)
                with stages.phase("decode", parent_id, iteration, warmup):
                    output, metadata = jax.block_until_ready(decode(resident))
                with stages.phase("download_check", parent_id, iteration, warmup):
                    status = np.asarray(metadata)
                    _codec._check_value(int(status[0]), "CUDA timeline decompression")
                    require(status.shape == (2,) and int(status[1]) == 0, "unexpected checked metadata")
                    require(np.asarray(output).tobytes() == raw, "byte-exact validation failed")
                result["iterations"].append({"iteration_id": f"{'warmup' if warmup else 'measured'}-{iteration:03d}",
                                             "iteration": iteration, "warmup": warmup, "status": [0, 0],
                                             "byte_exact": True, "cpu_codec_forbidden": True})
            finally:
                for name, original in originals.items():
                    setattr(zlib, name, original)

        with stages.phase("warmup") as parent_id:
            for iteration in range(args.warmups):
                workflow(iteration, True, parent_id)
        with stages.phase("measured_loop") as parent_id:
            for iteration in range(args.iterations):
                workflow(iteration, False, parent_id)
        require(len(result["iterations"]) == args.warmups + args.iterations, "missing validated workflows")
        require(source_identity(args.source_revision) == {key: result[key] for key in source_identity(args.source_revision)},
                "source changed during worker execution")
        require(sha(library_path) == result["native_build"]["library_sha256"] and
                sha(build_path) == result["native_build"]["build_sha256"], "native build changed during execution")
        result["complete"] = True
        return_code = 0
    except BaseException as error:
        result["error"] = {"type": type(error).__name__, "message": str(error)}
        traceback.print_exc()
        return_code = 1
    finally:
        result["end_ns"] = time.monotonic_ns()
        write_json(args.worker_result, result)
    return return_code


def query(function):
    before = time.monotonic_ns()
    try:
        value, error = function(), None
    except Exception as exception:
        value, error = None, {"type": type(exception).__name__, "message": str(exception)}
    after = time.monotonic_ns()
    return value, {"start_ns": before, "end_ns": after, "error": error}


def sample_processes(psutil, worker_pid, identities):
    discovered, discovery = query(lambda: [psutil.Process(worker_pid)] + psutil.Process(worker_pid).children(recursive=True))
    processes = []
    for process in discovered or []:
        before = time.monotonic_ns()
        row = {"pid": process.pid, "create_time": None, "cpu_user_seconds": None,
               "cpu_system_seconds": None, "rss_bytes": None, "missing": False, "error": None}
        try:
            row["create_time"] = process.create_time()
            identity = (process.pid, row["create_time"])
            identities.setdefault(identity, {"pid": process.pid, "create_time": row["create_time"], "first_seen_ns": before})
            identities[identity]["last_seen_ns"] = before
            cpu = process.cpu_times()
            row.update(cpu_user_seconds=cpu.user, cpu_system_seconds=cpu.system,
                       rss_bytes=process.memory_info().rss)
        except (psutil.NoSuchProcess, psutil.AccessDenied) as error:
            row.update(missing=True, error={"type": type(error).__name__, "message": str(error)})
        row.update(query_start_ns=before, query_end_ns=time.monotonic_ns())
        processes.append(row)
    return processes, discovery


def sample_gpu(nvml, handle, owned_pids):
    memory, memory_query = query(lambda: nvml.nvmlDeviceGetMemoryInfo(handle))
    utilization, utilization_query = query(lambda: nvml.nvmlDeviceGetUtilizationRates(handle))
    running, processes_query = query(lambda: nvml.nvmlDeviceGetComputeRunningProcesses(handle))
    processes = None
    if running is not None:
        processes = []
        for process in running:
            used = process.usedGpuMemory
            if used == getattr(nvml, "NVML_VALUE_NOT_AVAILABLE", None):
                used = None
            processes.append({"pid": process.pid, "used_memory_bytes": used, "owned": process.pid in owned_pids})
    owned = [row for row in processes or [] if row["owned"]]
    owned_memory = None if processes is None or any(row["used_memory_bytes"] is None for row in owned) else \
        sum(row["used_memory_bytes"] for row in owned)
    return {"total_memory_bytes": None if memory is None else memory.total,
            "used_memory_bytes": None if memory is None else memory.used,
            "free_memory_bytes": None if memory is None else memory.free,
            "gpu_utilization_percent": None if utilization is None else utilization.gpu,
            "memory_utilization_percent": None if utilization is None else utilization.memory,
            "compute_processes": processes, "owned_compute_memory_bytes": owned_memory,
            "queries": {"memory": memory_query, "utilization": utilization_query, "compute_processes": processes_query}}


def stop_worker(process):
    if process.poll() is None:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=5)


def supervisor(args):
    import psutil
    import pynvml
    result = {"schema_version": 1, "complete": False, **source_identity(args.source_revision),
              "created_utc": datetime.now(timezone.utc).isoformat(),
              "arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
              "collector_pid": os.getpid(), "start_ns": time.monotonic_ns(), "samples": [], "stage_events": [],
              "errors": [], "telemetry_versions": {"psutil": psutil.__version__,
                                                       "nvidia-ml-py": importlib.metadata.version("nvidia-ml-py")}}
    output = args.output.resolve()
    worker_result = output.with_name(output.stem + "-worker.json")
    worker_log = output.with_name(output.stem + "-worker.log")
    require(not output.exists() and not worker_result.exists() and not worker_log.exists(), "use fresh output paths")
    output.parent.mkdir(parents=True, exist_ok=True)
    run_id = uuid.uuid4().hex
    nvml_ready, process, threads = False, None, []
    identities, event_errors = {}, []
    log_lock = threading.Lock()

    def cancel(signum, frame):
        raise InterruptedError(f"collector received signal {signum}")

    previous_term = signal.signal(signal.SIGTERM, cancel)
    try:
        pynvml.nvmlInit()
        nvml_ready = True
        handle = pynvml.nvmlDeviceGetHandleByUUID(args.gpu_uuid)
        gpu_uuid = pynvml.nvmlDeviceGetUUID(handle)
        gpu_name = pynvml.nvmlDeviceGetName(handle)
        result["selected_gpu"] = {"ordinal": args.device, "uuid": gpu_uuid.decode() if isinstance(gpu_uuid, bytes) else gpu_uuid,
                                  "name": gpu_name.decode() if isinstance(gpu_name, bytes) else gpu_name,
                                  "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                                  "driver_version": str(pynvml.nvmlSystemGetDriverVersion())}
        nvtx = NVTX(args.nvtx_library)
        result["clock_anchor"] = nvtx.mark("cuda_zlib:origin:" + run_id)
        command = [sys.executable, str(Path(__file__).resolve()), "--worker", "--worker-result", str(worker_result),
                   "--run-id", run_id, "--source-revision", args.source_revision, "--gpu-uuid", args.gpu_uuid,
                   "--nvtx-library", str(args.nvtx_library), "--device", str(args.device), "--size", str(args.size),
                   "--workload", args.workload, "--seed", str(args.seed), "--warmups", str(args.warmups),
                   "--iterations", str(args.iterations)]
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
                        event = parse_stage_line(line, process.pid)
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
                    processes, tree_query = sample_processes(psutil, process.pid, identities)
                    current_pids = {row["pid"] for row in processes}
                    gpu = sample_gpu(pynvml, handle, current_pids)
                    gpu["uuid"] = result["selected_gpu"]["uuid"]
                    end = time.monotonic_ns()
                    result["samples"].append({"timestamp_ns": (start + end) // 2, "query_start_ns": start,
                                              "query_end_ns": end, "processes": processes, "process_tree_query": tree_query,
                                              "gpu": gpu})
                    require(end < deadline, "worker timeout")
                    next_sample += interval
                    if next_sample < end:
                        next_sample = end
                    time.sleep(max(0, (next_sample - time.monotonic_ns()) / 1e9))
            finally:
                stop_worker(process)
                process.wait()
                for thread in threads:
                    thread.join(timeout=5)
                require(not any(thread.is_alive() for thread in threads), "worker log reader did not finish")
        result["worker"].update(exit_code=process.returncode, end_ns=time.monotonic_ns(),
                               process_ids=sorted({row["pid"] for row in identities.values()}),
                               process_identities=list(identities.values()), log_sha256=sha(worker_log))
        require(not event_errors, "; ".join(event_errors))
        result["phase_intervals"] = phase_intervals(result["stage_events"])
        require(all(row["success"] for row in result["phase_intervals"]), "worker phase failed")
        completed = json.loads(worker_result.read_text())
        result["worker"].update(result_sha256=sha(worker_result), result=completed)
        require(process.returncode == 0 and completed["complete"] is True and completed["pid"] == process.pid,
                "worker did not finish successfully")
        require(all(completed[key] == result[key] for key in
                    ("source_revision", "harness_sha256", "profile_resident_sha256", "benchmark_sha256", "source_sha256")),
                "worker and collector source identities differ")
        iterations = completed["iterations"]
        expected = {(warmup, iteration) for warmup, count in ((True, args.warmups), (False, args.iterations))
                    for iteration in range(count)}
        require(len(iterations) == len(expected) and {(row["warmup"], row["iteration"]) for row in iterations} == expected and
                all(row["byte_exact"] is True and row["cpu_codec_forbidden"] is True and row["status"] == [0, 0]
                    for row in iterations), "missing validated worker iterations")
        result.update(native_build=completed["native_build"], environment=completed["environment"], fixture=completed["fixture"])
        result["methodology"] = {"sample_interval_requested_ns": interval,
                                 "cpu": "process-tree cumulative user/system seconds; 100% is one CPU core",
                                 "rss": "per-process RSS; summing may double-count shared pages",
                                 "gpu": "NVML UUID-selected whole-device memory/utilization; compute process memory separately PID-scoped",
                                 "nvml_utilization": "native reporting window; 10 ms polling does not imply 10 ms utilization resolution",
                                 "upload": "device_put enqueue only; decode completion waits for its input/output/status",
                                 "download_check": "metadata download/status check before output download and byte oracle",
                                 "clock": "absolute monotonic ns; unique supervisor NVTX instant bracket supplies cross-trace origin",
                                 "scope": "instrumented diagnostic workflow; no affinity or numerical-thread environment forced"}
        result["complete"] = True
    except BaseException as error:
        result["errors"].append({"type": type(error).__name__, "message": str(error)})
        if process is not None:
            stop_worker(process)
            if "worker" in result:
                result["worker"].update(exit_code=process.returncode, end_ns=time.monotonic_ns())
        traceback.print_exc()
    finally:
        signal.signal(signal.SIGTERM, previous_term)
        if nvml_ready:
            pynvml.nvmlShutdown()
        result["end_ns"] = time.monotonic_ns()
        write_json(output, result)
    return 0 if result["complete"] else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--gpu-uuid", required=True)
    parser.add_argument("--nvtx-library", type=Path, required=True)
    parser.add_argument("--size", type=int, default=64 * 1024**2)
    parser.add_argument("--workload", choices=("zeros", "text", "uint32", "float32", "random"), default="float32")
    parser.add_argument("--seed", type=int, default=20261008)
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
    require(math.isfinite(args.sample_ms) and args.sample_ms > 0 and math.isfinite(args.timeout) and args.timeout > 0,
            "positive finite sampling interval/timeout required")
    require((args.worker and args.worker_result is not None and args.run_id) or
            (not args.worker and args.output is not None), "output path required")
    return worker(args) if args.worker else supervisor(args)


if __name__ == "__main__":
    sys.exit(main())
