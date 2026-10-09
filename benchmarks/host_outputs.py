#!/usr/bin/env python3
# Copyright (c) 2026 xangma
# SPDX-License-Identifier: MIT
"""Measure synchronous host-input decompression for three consumer contracts.

Each timed CUDA call uses the public decompress_zlib_host API with compressed
HOST bytes, returning fresh completed pinned storage with native status checked.
ndarray and memoryview share that storage; bytes adds tobytes allocation/copy.
Separate CPU series decode fresh stdlib zlib bytes every call. np.frombuffer and
memoryview share those fresh CPU bytes without an additional output copy.

The timer includes API/status/upload/download/output conversion and excludes
fixture creation, native initialization, output release and oracle comparisons.
Every warmup and sample is checked byte-exact after timing, without tobytes for
array/view consumers. All six series are shuffled in each fixed-seed round.
This is a synchronous consumer benchmark, not a GPU-only timing or overlap test.
"""

import argparse
import csv
import ctypes
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import random
import re
import resource
import statistics
import subprocess
import sys
import time
import traceback
import uuid


ROOT = Path(__file__).resolve().parents[1]
WORKFLOW_PATH = Path(__file__).with_name("profile_workflow.py")
SPEC = importlib.util.spec_from_file_location("_host_outputs_workflow", WORKFLOW_PATH)
workflow = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(workflow)
require, sha, write_json = workflow.require, workflow.sha, workflow.write_json
CONTRACTS = ("ndarray", "memoryview", "bytes")
SERIES = tuple((backend, contract) for backend in ("cpu", "cuda") for contract in CONTRACTS)
BACKING = {"cpu": "fresh stdlib zlib bytes", "cuda": "fresh JAX-owned pinned host storage"}


def source_identity(revision):
    result = workflow.source_identity(revision)
    result["harness_sha256"] = sha(__file__)
    result["dependencies_sha256"]["benchmarks/profile_workflow.py"] = sha(WORKFLOW_PATH)
    return result


def runtime_package(module):
    path = Path(module.__file__).resolve().parent
    require(path == (ROOT / "src/cuda_zlib").resolve(),
            "loaded cuda_zlib differs from the recorded source tree; set PYTHONPATH=src")
    return path


def schedule(warmups, samples, seed):
    rng = random.Random(seed)
    result = []
    for warmup, count in ((True, warmups), (False, samples)):
        for iteration in range(count):
            names = [f"{backend}_{contract}" for backend, contract in SERIES]
            rng.shuffle(names)
            result.extend({"warmup": warmup, "round": iteration, "series": name} for name in names)
    return result


def measure(call, row):
    before = resource.getrusage(resource.RUSAGE_SELF)
    cpu_before = time.thread_time_ns()
    start = time.monotonic_ns()
    try:
        return call()
    finally:
        end = time.monotonic_ns()
        cpu_after = time.thread_time_ns()
        after = resource.getrusage(resource.RUSAGE_SELF)
        row.update(start_ns=start, end_ns=end, wall_ns=end - start,
                   thread_cpu_ns=cpu_after - cpu_before,
                   minor_faults=after.ru_minflt - before.ru_minflt,
                   major_faults=after.ru_majflt - before.ru_majflt)


def consumer_call(backend, contract, stream, size, device, host_api, np, zlib):
    """No fixture decode or output is cached between calls."""
    require((backend, contract) in SERIES, "unknown consumer series")

    def call():
        if backend == "cuda":
            result = host_api(stream, size, device)
            if contract == "ndarray":
                return result
            return memoryview(result) if contract == "memoryview" else result.tobytes()
        result = zlib.decompress(stream)
        if contract == "bytes":
            return result
        return np.frombuffer(result, dtype=np.uint8) if contract == "ndarray" else memoryview(result)

    return call


def validate_output(result, contract, raw, expected, np):
    """Validate the actual consumer object without allocating bytes for views."""
    proof = {"output_type": type(result).__name__, "output_bytes": len(raw)}
    if contract == "ndarray":
        require(isinstance(result, np.ndarray) and result.dtype == np.uint8 and
                result.shape == (len(raw),) and result.flags.c_contiguous and not result.flags.writeable,
                "invalid ndarray consumer contract")
        equal = np.array_equal(result, expected)
        proof.update(readonly=not result.flags.writeable, dtype=str(result.dtype), shape=list(result.shape),
                     owns_data=bool(result.flags.owndata), backing_type=type(result.base).__name__)
    elif contract == "memoryview":
        require(isinstance(result, memoryview) and result.ndim == 1 and result.format == "B" and
                result.nbytes == len(raw) and result.c_contiguous and result.readonly,
                "invalid memoryview consumer contract")
        equal = np.array_equal(np.frombuffer(result, dtype=np.uint8), expected)
        proof.update(readonly=result.readonly, format=result.format, shape=list(result.shape),
                     backing_type=type(result.obj).__name__)
    else:
        require(contract == "bytes" and type(result) is bytes and len(result) == len(raw),
                "invalid bytes consumer contract")
        equal = result == raw
        proof.update(readonly=True, backing_type=None)
    require(equal, "consumer output differs from fixture")
    return {**proof, "byte_exact": True}


def summarize(rows):
    require(rows and all(row["complete"] and row["byte_exact"] for row in rows), "incomplete consumer samples")
    return {"median_wall_ns": statistics.median(row["wall_ns"] for row in rows),
            "min_wall_ns": min(row["wall_ns"] for row in rows), "max_wall_ns": max(row["wall_ns"] for row in rows),
            **{f"median_{key}": statistics.median(row[key] for row in rows)
               for key in ("thread_cpu_ns", "minor_faults", "major_faults")}}


def collect_case(args, size, device, host_api, np, zlib, make_payload, cases):
    raw = make_payload(args.workload, size, args.seed)
    stream = zlib.compress(raw, 6)
    require(8 <= len(stream) <= 2**28 and zlib.decompress(stream) == raw, "invalid external stdlib fixture")
    expected = np.frombuffer(raw, dtype=np.uint8)
    seed = args.order_seed + size
    case = {"workload": args.workload, "output_bytes": size, "input_bytes": len(stream),
            "fixture": {"raw_sha256": hashlib.sha256(raw).hexdigest(),
                        "stdlib_stream_sha256": hashlib.sha256(stream).hexdigest(),
                        "stdlib_stream_bytes": len(stream), "stdlib_level": 6, "seed": args.seed},
            "order_seed": seed, "order": schedule(args.warmups, args.samples, seed), "series": []}
    cases.append(case)  # Preserve completed observations if a later call fails.
    series = {}
    calls = {}
    for backend, contract in SERIES:
        name = f"{backend}_{contract}"
        item = {"name": name, "backend": backend, "contract": contract,
                "backing_storage": "fresh Python bytes copied from JAX-owned pinned host storage"
                if backend == "cuda" and contract == "bytes" else BACKING[backend],
                "samples": [], "warmups": []}
        case["series"].append(item)
        series[name] = item
        calls[name] = consumer_call(backend, contract, stream, size, device, host_api, np, zlib)
    for index, event in enumerate(case["order"]):
        item = series[event["series"]]
        row = {"order_index": index, "round": event["round"], "warmup": event["warmup"], "complete": False}
        item["warmups" if event["warmup"] else "samples"].append(row)
        if item["backend"] == "cuda":
            with workflow.forbid_cpu_codec(zlib):
                output = measure(calls[event["series"]], row)
        else:
            output = measure(calls[event["series"]], row)
        row.update(validate_output(output, item["contract"], raw, expected, np), complete=True)
        del output  # Release outside the next timed call; do not retain full results.
    for item in case["series"]:
        item["summary"] = summarize(item["samples"])
    return case


def selected_gpu_uuid(device):
    """Use the same CUDA driver ordinal as the native backend's device query."""
    class CuUuid(ctypes.Structure):
        _fields_ = [("bytes", ctypes.c_ubyte * 16)]

    driver = ctypes.CDLL("libcuda.so.1")
    driver.cuInit.argtypes, driver.cuInit.restype = [ctypes.c_uint], ctypes.c_int
    driver.cuDeviceGet.argtypes, driver.cuDeviceGet.restype = [ctypes.POINTER(ctypes.c_int), ctypes.c_int], ctypes.c_int
    driver.cuDeviceGetUuid.argtypes, driver.cuDeviceGetUuid.restype = [ctypes.POINTER(CuUuid), ctypes.c_int], ctypes.c_int
    handle, identifier = ctypes.c_int(), CuUuid()
    require(driver.cuInit(0) == 0 and driver.cuDeviceGet(ctypes.byref(handle), int(device.local_hardware_id)) == 0 and
            driver.cuDeviceGetUuid(ctypes.byref(identifier), handle.value) == 0, "selected CUDA GPU UUID query failed")
    return "GPU-" + str(uuid.UUID(bytes=bytes(identifier.bytes)))


def gpu_snapshot():
    result = {"start_ns": time.monotonic_ns(), "queries": [], "gpus": [], "compute_processes": [], "errors": []}
    queries = (("gpus", "gpu", ("uuid", "name", "driver_version", "memory.used", "memory.total", "utilization.gpu")),
               ("compute_processes", "compute-apps", ("pid", "gpu_uuid", "used_memory")))
    for key, kind, fields in queries:
        command = ["nvidia-smi", f"--query-{kind}=" + ",".join(fields), "--format=csv,noheader,nounits"]
        query = {"command": command, "start_ns": time.monotonic_ns()}
        try:
            completed = subprocess.run(command, capture_output=True, text=True, timeout=15)
            query.update(exit_code=completed.returncode, stdout=completed.stdout, stderr=completed.stderr)
            require(completed.returncode == 0, "nvidia-smi query failed")
            for values in csv.reader(completed.stdout.splitlines()):
                require(len(values) == len(fields), "unexpected nvidia-smi columns")
                row = dict(zip(fields, (value.strip() for value in values)))
                if key == "compute_processes":
                    row["owned_pid"] = row["pid"] == str(os.getpid())
                result[key].append(row)
        except (OSError, subprocess.SubprocessError, ValueError) as error:
            query["error"] = str(error)
            result["errors"].append(str(error))
        query["end_ns"] = time.monotonic_ns()
        result["queries"].append(query)
    result["end_ns"] = time.monotonic_ns()
    return result


def run(args):
    require(not args.output.exists(), "use a fresh output path")
    report = {"schema_version": 1, "kind": "host_outputs", "complete": False,
              "created_utc": datetime.now(timezone.utc).isoformat(), "pid": os.getpid(),
              "arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
              "start_ns": time.monotonic_ns(), "cases": []}
    code = 1
    try:
        report.update(source_identity(args.source_revision))
        import numpy as np
        import zlib
        import jax
        import jaxlib
        import cuda_zlib
        from cuda_zlib import _codec, _encode_kernels, _decode_kernels, _postprocess
        from benchmark import make_payload, native_build
        package = runtime_package(cuda_zlib)
        device = _codec._select_device(args.device)
        cuda_zlib.compile_kernels(device)
        native = native_build(device)
        require(hashlib.sha256(json.dumps(native["identity"], sort_keys=True).encode()).hexdigest() == native["cache_key"],
                "native cache identity differs")
        native_sources = {name: sha(package / "native" / name)
                          for name in ("codec_ffi.cu", "batch_encode.cuh", "batch_decode.cuh")}
        native_sources.update({name: hashlib.sha256(module.CUDA_SOURCE.encode()).hexdigest() for name, module in
                               (("encoder.cuh", _encode_kernels), ("decoder.cuh", _decode_kernels), ("postprocess.cuh", _postprocess))})
        require(native["identity"]["sources"] == native_sources, "loaded native sources differ from runtime")
        report["native_build"] = native
        report["environment"] = {"python": sys.version, "platform": platform.platform(), "numpy": np.__version__,
            "jax": jax.__version__, "jaxlib": jaxlib.__version__, "cuda_zlib": cuda_zlib.__version__,
            "zlib_runtime": zlib.ZLIB_RUNTIME_VERSION, "device_kind": device.device_kind,
            "local_hardware_id": int(device.local_hardware_id), "gpu_uuid": selected_gpu_uuid(device),
            "cuda_platform": device.client.platform_version, "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "backend": "JAX typed CUDA FFI", "runtime_package": str(package)}
        report["gpu_snapshot_before"] = gpu_snapshot()
        for size in args.sizes:
            collect_case(args, size, device, cuda_zlib.decompress_zlib_host, np, zlib, make_payload, report["cases"])
        report["gpu_snapshot_after"] = gpu_snapshot()
        current = source_identity(args.source_revision)
        require(current == {key: report[key] for key in current}, "source changed during benchmark")
        require(sha(native["library"]) == native["library_sha256"] and sha(native["build"]) == native["build_sha256"],
                "native artifacts changed during benchmark")
        report["methodology"] = {
            "scope": "full synchronous public host-input decompression plus consumer conversion",
            "timing_includes": "host compressed input, public API validation, upload, native decode/status check, pinned download, consumer conversion",
            "timing_excludes": "fixture generation, initialization/build, output release, post-timing oracle comparisons",
            "cuda_api": "cuda_zlib.decompress_zlib_host; successful return is public status-check evidence, no raw metadata returned",
            "cpu_api": "fresh stdlib zlib.decompress each call; ndarray np.frombuffer and memoryview reuse that fresh bytes storage",
            "cuda_outputs": "ndarray/memoryview share completed JAX-owned pinned storage; bytes adds tobytes allocation/copy",
            "validation": "every warmup/sample byte-exact after timer; ndarray/memoryview checks do not call tobytes",
            "ordering": "all six series shuffled once per round, warmups first, random.Random(order_seed + output_bytes)",
            "faults": "minor/major_faults are process-wide deltas and may include concurrent JAX activity",
            "cpu_time": "calling thread_time_ns; counter snapshots bracket the timed call",
            "limits": "single-file synchronous API; no batch, pipelining or GPU-only claim; no affinity/thread environment forced"}
        report["complete"], code = True, 0
    except BaseException as error:
        report["error"] = {"type": type(error).__name__, "message": str(error)}
        traceback.print_exc()
    finally:
        report["end_ns"] = time.monotonic_ns()
        write_json(args.output, report)
    return code


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--sizes", type=int, nargs="+", default=[65536, 1048576, 67108864])
    parser.add_argument("--workload", choices=("float32", "uint32", "text", "zeros", "random"), default="float32")
    parser.add_argument("--seed", type=int, default=20261008)
    parser.add_argument("--order-seed", type=int, default=20261009)
    parser.add_argument("--samples", type=int, default=12)
    parser.add_argument("--warmups", type=int, default=2)
    args = parser.parse_args()
    require(re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", args.source_revision), "full source revision required")
    require(args.device >= 0 and args.samples > 0 and args.warmups > 0 and args.sizes and
            len(set(args.sizes)) == len(args.sizes) and all(0 < size < 2**28 for size in args.sizes),
            "positive bounded unique sizes, counts and device required")
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
