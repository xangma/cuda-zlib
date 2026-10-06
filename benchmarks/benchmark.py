#!/usr/bin/env python3
# Copyright (c) 2026 xangma
# SPDX-License-Identifier: MIT
"""Reproducible synthetic byte-stream benchmarks; see BENCHMARKS.md."""

import argparse
import gc
import hashlib
import json
import platform
import statistics
import subprocess
import time
import zlib
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

WORKLOADS = ("zeros", "text", "uint32", "float32", "random")
MAX_BYTES = 256 * 1024 ** 2
CHUNK_BYTES = 32768


def make_payload(name, size, seed):
    """Return exactly size bytes, independent of workload execution order."""
    rng = np.random.default_rng(seed)
    if name == "zeros":
        return bytes(size)
    if name == "text":
        records = (
            f"2026-01-01T12:{i // 60:02d}:{i % 60:02d}Z "
            f"service=worker-{i % 17} event=request status={200 + i % 5} "
            f"route=/api/items/{i % 137} elapsed_us={100 + i % 997}\n"
            for i in range(8192)
        )
        block = "".join(records).encode("ascii")
        return (block * ((size + len(block) - 1) // len(block)))[:size]
    if name == "uint32":
        return np.arange((size + 3) // 4, dtype="<u4").tobytes()[:size]
    if name == "float32":
        return rng.standard_normal((size + 3) // 4).astype("<f4").tobytes()[:size]
    if name == "random":
        return rng.integers(0, 256, size=size, dtype=np.uint8).tobytes()
    raise ValueError(name)


def measure(fn, samples, size, synchronize=lambda: None):
    """One untimed warmup, then synchronized wall-clock measurements."""
    result = fn()
    synchronize()
    times = []
    for _ in range(samples):
        del result
        synchronize()
        start = time.perf_counter()
        result = fn()
        synchronize()
        times.append(time.perf_counter() - start)
    median = statistics.median(times)
    return result, {
        "seconds": times,
        "median_seconds": median,
        "min_seconds": min(times),
        "max_seconds": max(times),
        "mib_per_second": size / (1024 ** 2) / median,
    }


def gpu_snapshot():
    fields = "name,driver_version,memory.used,memory.total,utilization.gpu,temperature.gpu"
    try:
        return subprocess.check_output(
            ["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader,nounits"],
            text=True, timeout=10,
        ).strip().splitlines()
    except (OSError, subprocess.SubprocessError):
        return []


def cpu_model():
    if Path("/proc/cpuinfo").exists():
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    if platform.system() == "Darwin":
        return subprocess.check_output(
            ["sysctl", "-n", "machdep.cpu.brand_string"], text=True, timeout=10).strip()
    return platform.processor()


def cpu_only(args):
    report = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "arguments": {k: v for k, v in vars(args).items() if k != "output"},
        "environment": {"os": platform.system(), "machine": platform.machine(),
                        "cpu": cpu_model(), "python": platform.python_version(),
                        "numpy": np.__version__, "zlib_runtime": zlib.ZLIB_RUNTIME_VERSION},
        "methodology": {"clock": "perf_counter", "warmup_calls_per_measurement": 1,
                        "throughput_denominator": "uncompressed input bytes / 2**20",
                        "cpu": "single-threaded stdlib zlib; levels 1 and 6",
                        "timing_includes": "output allocation",
                        "timing_excludes": "payload generation, validation"},
        "cases": [],
    }
    for size in args.sizes:
        for workload in args.workloads:
            payload = make_payload(workload, size, args.seed)
            times, lengths = {}, {}
            for level in (1, 6):
                stream, times[f"cpu_compress_level{level}"] = measure(
                    lambda level=level: zlib.compress(payload, level), args.cpu_samples, size)
                decoded, times[f"cpu_decompress_level{level}"] = measure(
                    lambda: zlib.decompress(stream), args.cpu_samples, size)
                assert decoded == payload
                lengths[f"zlib{level}"] = len(stream)
                del stream, decoded
            report["cases"].append({
                "workload": workload, "input_bytes": size,
                "input_sha256": hashlib.sha256(payload).hexdigest(),
                "encoded_bytes": lengths,
                "encoded_percent": {k: 100 * v / size for k, v in lengths.items()},
                "timings": times, "validation": "both levels decoded byte-exact",
            })
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, indent=2) + "\n")
            print(f"pass {workload} {size}: encoded={lengths}", flush=True)
    print(f"complete {len(report['cases'])} CPU cases", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", nargs="+", type=int, default=[65536, 1048576, 67108864])
    parser.add_argument("--workloads", nargs="+", choices=WORKLOADS, default=list(WORKLOADS))
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--cpu-samples", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20261006)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--cpu-only", action="store_true", help="run CPU baselines without CuPy or CUDA")
    parser.add_argument("--output", type=Path, default=Path("results.json"))
    args = parser.parse_args()
    if min(args.sizes) < 1 or max(args.sizes) > MAX_BYTES:
        parser.error("sizes must be between 1 byte and 256 MiB")
    if min(args.samples, args.cpu_samples) < 1 or args.device < 0:
        parser.error("sample counts must be positive and device must be nonnegative")
    if args.cpu_only:
        cpu_only(args)
        return
    for size in args.sizes:
        chunks = (size + CHUNK_BYTES - 1) // CHUNK_BYTES
        if size + chunks * 5 + 6 > MAX_BYTES:
            parser.error("CUDA sizes must leave room for worst-case block/framing overhead within 256 MiB")

    start = time.perf_counter()
    import cupy as cp
    import cuda_zlib
    cp.cuda.Device(args.device).use()
    cp.cuda.runtime.deviceSynchronize()
    init_seconds = time.perf_counter() - start
    start = time.perf_counter()
    cuda_zlib.compile_kernels(args.device)
    cp.cuda.runtime.deviceSynchronize()
    compile_seconds = time.perf_counter() - start
    synchronize = cp.cuda.runtime.deviceSynchronize
    properties = cp.cuda.runtime.getDeviceProperties(args.device)
    name = properties["name"]
    module_root = Path(cuda_zlib.__file__).parent
    report = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "arguments": {k: v for k, v in vars(args).items() if k != "output"},
        "environment": {
            "os": platform.system(), "machine": platform.machine(),
            "cpu": cpu_model(), "python": platform.python_version(),
            "numpy": np.__version__, "cupy": cp.__version__,
            "cuda_zlib": cuda_zlib.__version__,
            "zlib_runtime": zlib.ZLIB_RUNTIME_VERSION,
            "cuda_runtime": cp.cuda.runtime.runtimeGetVersion(),
            "cuda_driver": cp.cuda.runtime.driverGetVersion(),
            "gpu": name.decode() if isinstance(name, bytes) else name,
            "gpu_snapshot_before": gpu_snapshot(),
            "codec_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                             for p in sorted(module_root.glob("*.py"))},
        },
        "startup": {"imports_and_cuda_init_seconds": init_seconds,
                    "compile_kernels_seconds": compile_seconds},
        "methodology": {
            "clock": "perf_counter; deviceSynchronize before and after CUDA calls",
            "warmup_calls_per_measurement": 1,
            "throughput_denominator": "uncompressed input bytes / 2**20",
            "compression_chunk_bytes": CHUNK_BYTES,
            "timing_includes": "API allocations; transfers only for named host workflows",
            "timing_excludes": "payload generation, initial resident uploads, validation",
            "cpu": "single-threaded stdlib zlib; levels 1 and 6; no level equivalence implied",
        },
        "cases": [],
    }
    for size in args.sizes:
        for workload in args.workloads:
            print(f"begin {workload} {size}", flush=True)
            payload = make_payload(workload, size, args.seed)
            resident = cp.asarray(np.frombuffer(payload, dtype=np.uint8))
            synchronize()
            times = {}
            encoded, times["cuda_compress_device_device"] = measure(
                lambda: cuda_zlib.compress_zlib(resident, args.device),
                args.samples, size, synchronize)
            encoded_host = encoded.get().tobytes()
            assert zlib.decompress(encoded_host) == payload
            host_encoded, times["cuda_compress_host_device"] = measure(
                lambda: cuda_zlib.compress_zlib(payload, args.device),
                args.samples, size, synchronize)
            assert host_encoded.get().tobytes() == encoded_host
            host_bytes, times["cuda_compress_host_host"] = measure(
                lambda: cuda_zlib.compress_zlib(payload, args.device).get().tobytes(),
                args.samples, size, synchronize)
            assert host_bytes == encoded_host
            del host_encoded, host_bytes
            cpu_streams = {}
            for level in (1, 6):
                stream, times[f"cpu_compress_level{level}"] = measure(
                    lambda level=level: zlib.compress(payload, level), args.cpu_samples, size)
                assert zlib.decompress(stream) == payload
                cpu_streams[level] = stream
            if not 8 <= len(cpu_streams[6]) <= MAX_BYTES:
                raise ValueError("external level-6 stream exceeds the CUDA decoder extent bounds")
            external = cp.asarray(np.frombuffer(cpu_streams[6], dtype=np.uint8))
            synchronize()
            for label, compressed in (("codec", encoded), ("level6", external)):
                decoded, times[f"cuda_decompress_{label}_device_device"] = measure(
                    lambda compressed=compressed: cuda_zlib.decompress_zlib(compressed, size, args.device),
                    args.samples, size, synchronize)
                assert decoded.get().tobytes() == payload
                del decoded
                compressed_host = encoded_host if label == "codec" else cpu_streams[6]
                decoded_host, times[f"cpu_decompress_{label}"] = measure(
                    lambda compressed_host=compressed_host: zlib.decompress(compressed_host),
                    args.cpu_samples, size)
                assert decoded_host == payload
                del decoded_host
            decoded_host, times["cuda_decompress_level6_host_host"] = measure(
                lambda: cuda_zlib.decompress_zlib(cpu_streams[6], size, args.device).get().tobytes(),
                args.samples, size, synchronize)
            assert decoded_host == payload
            case = {
                "workload": workload, "input_bytes": size,
                "input_sha256": hashlib.sha256(payload).hexdigest(),
                "encoded_bytes": {"cuda": len(encoded_host),
                                  "zlib1": len(cpu_streams[1]), "zlib6": len(cpu_streams[6])},
                "encoded_percent": {"cuda": 100 * len(encoded_host) / size,
                                    "zlib1": 100 * len(cpu_streams[1]) / size,
                                    "zlib6": 100 * len(cpu_streams[6]) / size},
                "timings": times,
                "validation": "all outputs byte-exact; stdlib decoded CUDA output; CUDA decoded stdlib level 6",
            }
            report["cases"].append(case)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, indent=2) + "\n")
            print(f"pass {workload} {size}: encoded={case['encoded_bytes']}", flush=True)
            del payload, resident, encoded, encoded_host, cpu_streams, external, decoded_host, compressed
            gc.collect()
            cp.get_default_memory_pool().free_all_blocks()
    report["environment"]["gpu_snapshot_after"] = gpu_snapshot()
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"complete {len(report['cases'])} cases", flush=True)


if __name__ == "__main__":
    main()
