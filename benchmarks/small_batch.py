#!/usr/bin/env python3
# Copyright (c) 2026 xangma
# SPDX-License-Identifier: MIT
"""Measure independent small zlib streams: CPU, single-file calls, and batches."""

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

try:
    from .benchmark import CHUNK_BYTES, MAX_BYTES, cpu_model, gpu_snapshot, make_payload, native_build, source_revision
except ImportError:
    from benchmark import CHUNK_BYTES, MAX_BYTES, cpu_model, gpu_snapshot, make_payload, native_build, source_revision

REPORT_TYPE = "cuda-zlib-small-batch"
WORKLOADS = ("zeros", "text", "random")


def measure(fn, samples, total_bytes, files, ready=lambda value: value):
    """Warm once, then time completed outputs without oracle/status transfers."""
    start = time.perf_counter()
    result = ready(fn())
    warmup_seconds = time.perf_counter() - start
    elapsed = []
    for _ in range(samples):
        del result
        start = time.perf_counter()
        result = ready(fn())
        elapsed.append(time.perf_counter() - start)
    median = statistics.median(elapsed)
    return result, {
        "seconds": elapsed,
        "median_seconds": median,
        "min_seconds": min(elapsed),
        "max_seconds": max(elapsed),
        "median_seconds_per_file": median / files,
        "mib_per_second": total_bytes / (1024 ** 2) / median,
        "warmup_seconds": warmup_seconds,
    }


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def source_snapshot(module_root=None):
    root = Path(__file__).resolve().parents[1]
    paths = [Path(__file__).resolve(), Path(__file__).with_name("benchmark.py").resolve(),
             Path(__file__).with_name("plot_small_batch.py").resolve()]
    if module_root is not None:
        paths += sorted(module_root.glob("*.py"))
        paths += sorted(module_root.glob("native/*.cu"))
        paths += sorted(module_root.glob("native/*.cuh"))
    hashes = {}
    for path in paths:
        if path.exists():
            try:
                label = str(path.relative_to(root))
            except ValueError:
                label = f"cuda_zlib/{path.relative_to(module_root)}"
            hashes[label] = sha256(path.read_bytes())
    git = {}
    for key, command in (("commit", ["rev-parse", "HEAD"]),
                         ("status", ["status", "--porcelain"])):
        try:
            git[key] = subprocess.check_output(
                ["git", "-C", str(root), *command], text=True, timeout=10,
                stderr=subprocess.DEVNULL).strip()
        except (OSError, subprocess.SubprocessError):
            git[key] = None
    return {"sha256": hashes, "git": git}


def save(report, output):
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n")
    temporary.replace(output)


def check_compressed(streams, inputs):
    if len(streams) != len(inputs):
        raise AssertionError("wrong number of compressed streams")
    for stream, payload in zip(streams, inputs):
        if zlib.decompress(stream) != payload:
            raise AssertionError("compressed stream differs from the input")


def compression_slots(result, capacities, inputs):
    output, metadata = result
    values = np.asarray(metadata)
    if values.shape != (len(inputs), 2) or np.any(values[:, 1]):
        raise AssertionError(f"batch compression status: {values.tolist()}")
    packed = np.asarray(output).tobytes()
    offset, streams = 0, []
    for index, capacity in enumerate(capacities):
        length = int(values[index, 0])
        if not 8 <= length <= capacity:
            raise AssertionError("invalid compressed extent")
        streams.append(packed[offset:offset + length])
        if any(packed[offset + length:offset + capacity]):
            raise AssertionError("compression padding is not zero")
        offset += capacity
    check_compressed(streams, inputs)
    return tuple(streams)


def check_single_compressed(results, inputs):
    streams = []
    for (output, metadata), payload in zip(results, inputs):
        length, status = np.asarray(metadata)
        if int(status):
            raise AssertionError(f"single compression status: {status}")
        stream = np.asarray(output)[:int(length)].tobytes()
        if zlib.decompress(stream) != payload:
            raise AssertionError("single compression differs from input")
        streams.append(stream)
    return tuple(streams)


def check_single_decoded(results, inputs):
    for (output, metadata), payload in zip(results, inputs):
        if np.any(np.asarray(metadata)):
            raise AssertionError("single decompression returned a failure status")
        if np.asarray(output).tobytes() != payload:
            raise AssertionError("single decompression differs from input")


def check_batch_decoded(result, inputs):
    output, metadata = result
    if np.asarray(metadata).shape != (len(inputs), 2) or np.any(np.asarray(metadata)):
        raise AssertionError("batch decompression returned a failure status")
    if np.asarray(output).tobytes() != b"".join(inputs):
        raise AssertionError("batch decompression differs from input")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", nargs="+", type=int, default=[256, 4096, 65536])
    parser.add_argument("--counts", nargs="+", type=int, default=[1, 8, 32, 128])
    parser.add_argument("--workloads", nargs="+", choices=WORKLOADS, default=list(WORKLOADS))
    parser.add_argument("--samples", type=int, default=7)
    parser.add_argument("--seed", type=int, default=20261007)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--chunk-bytes", type=int, default=CHUNK_BYTES)
    parser.add_argument("--roundtrip", action="store_true", help="also time one compiled padded batch round trip")
    parser.add_argument("--cpu-only", action="store_true", help="run actual CPU measurements without JAX or CUDA")
    parser.add_argument("--output", type=Path, default=Path("small-batch.json"))
    args = parser.parse_args()
    if args.samples < 1 or args.device < 0:
        parser.error("samples must be positive and device must be nonnegative")
    if any(size < 1 or size > MAX_BYTES for size in args.sizes):
        parser.error("sizes must be between 1 byte and 256 MiB")
    if any(count < 1 or count > 262144 for count in args.counts):
        parser.error("counts must be between 1 and 262144")
    if not 256 <= args.chunk_bytes <= 65535:
        parser.error("chunk-bytes must be between 256 and 65535")
    if len(set(args.sizes)) != len(args.sizes) or len(set(args.counts)) != len(args.counts) or len(set(args.workloads)) != len(args.workloads):
        parser.error("sizes, counts and workloads must each be unique")
    for size in args.sizes:
        chunks = max(1, (size + args.chunk_bytes - 1) // args.chunk_bytes)
        if max(args.counts) * (size + chunks * 5 + 6) > MAX_BYTES:
            parser.error("matrix exceeds the 256 MiB aggregate input/output limit")
        if max(args.counts) * chunks > 262144 or 2 * chunks - 1 > 262144:
            parser.error("matrix exceeds the aggregate chunk limit")
    try:
        revision = source_revision()
    except ValueError as exc:
        parser.error(str(exc))

    environment = {"os": platform.system(), "machine": platform.machine(),
                   "cpu": cpu_model(), "python": platform.python_version(),
                   "numpy": np.__version__, "zlib_runtime": zlib.ZLIB_RUNTIME_VERSION}
    startup = {}
    module_root = None
    if not args.cpu_only:
        start = time.perf_counter()
        import jax
        import jaxlib
        import cuda_zlib
        from cuda_zlib import _codec
        selected = _codec._select_device(args.device)
        jax.device_put(np.zeros(1, dtype=np.uint8), selected).block_until_ready()
        startup["imports_and_cuda_init_seconds"] = time.perf_counter() - start
        start = time.perf_counter()
        cuda_zlib.compile_kernels(selected)
        startup["compile_kernels_seconds"] = time.perf_counter() - start
        module_root = Path(cuda_zlib.__file__).resolve().parent
        loaded_build = native_build(selected)
        environment.update({"jax": jax.__version__, "jaxlib": jaxlib.__version__,
                            "cuda_zlib": cuda_zlib.__version__, "backend": "JAX typed CUDA FFI",
                            "cuda_platform": selected.client.platform_version, "gpu": selected.device_kind,
                            "gpu_snapshot_before": gpu_snapshot(),
                            "native_build": loaded_build,
                            "native_builds": [loaded_build["identity"]],
                            "workspace_pool_before": cuda_zlib.workspace_pool_stats(selected)})

    report = {
        "schema_version": 1, "report_type": REPORT_TYPE,
        "source_revision": revision, "harness_sha256": sha256(Path(__file__).read_bytes()),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "arguments": {key: value for key, value in vars(args).items() if key != "output"},
        "environment": environment, "source": source_snapshot(module_root), "startup": startup,
        "methodology": {
            "clock": "perf_counter; JAX block_until_ready on every returned array",
            "warmup_calls_per_measurement": 1,
            "warmup": "recorded separately; includes per-shape XLA compilation and first execution",
            "throughput_denominator": "total uncompressed bytes / 2**20",
            "per_file_latency": "total completed batch latency / file count; amortized, not individual file latency",
            "cpu": "single-threaded stdlib zlib; level 1 compression; level 6 input streams for decompression",
            "decode_inputs": "identical independent stdlib level 6 streams for CPU, single-file CUDA and batched CUDA",
            "resident": "one warmed jax.jit call; single-file loop unrolled into independent FFI calls, or one packed batch FFI call; inputs already on GPU",
            "resident_status": "per-file device metadata and bytes checked after timing; timed calls wait for metadata completion but do not copy status to the host",
            "host_bytes": "Python bytes inputs and per-file bytes outputs; includes input packing, transfers, synchronous codec status checks, and output materialization",
            "timing_includes": "output allocations, Python/JAX dispatch, GPU execution and completion; host-byte scope additionally includes transfers and bytes copies",
            "timing_excludes": "payload generation, initial resident upload/packing, native build/registration, warmup/XLA compilation, post-timing oracle comparisons",
            "validation": "last output from each timed workflow checked outside timing; stdlib decodes every compressed stream; decoded bytes match every input",
            "roundtrip": "optional compiled padded batch compression then decompression; encoded lengths remain on device; both statuses checked after timing",
        },
        "cases": [], "complete": False,
    }
    save(report, args.output)
    for size in args.sizes:
        for count in args.counts:
            for workload in args.workloads:
                print(f"begin {workload}: {count} files x {size} bytes", flush=True)
                inputs = tuple(make_payload(workload, size, args.seed + index) for index in range(count))
                sizes = (size,) * count
                total_bytes = size * count
                streams = tuple(zlib.compress(payload, 6) for payload in inputs)
                stream_sizes = tuple(map(len, streams))
                if sum(stream_sizes) > MAX_BYTES:
                    raise ValueError("external streams exceed the aggregate compressed-input limit")
                timings = {}
                cpu_encoded, timings["cpu_compress_level1"] = measure(
                    lambda: tuple(zlib.compress(payload, 1) for payload in inputs),
                    args.samples, total_bytes, count)
                check_compressed(cpu_encoded, inputs)
                cpu_decoded, timings["cpu_decompress_level6"] = measure(
                    lambda: tuple(zlib.decompress(stream) for stream in streams),
                    args.samples, total_bytes, count)
                if cpu_decoded != inputs:
                    raise AssertionError("CPU decompression differs from input")
                cuda_encoded = None
                if not args.cpu_only:
                    resident_inputs = tuple(jax.device_put(np.frombuffer(payload, dtype=np.uint8), selected) for payload in inputs)
                    resident_streams = tuple(jax.device_put(np.frombuffer(stream, dtype=np.uint8), selected) for stream in streams)
                    packed_input = jax.device_put(np.frombuffer(b"".join(inputs), dtype=np.uint8), selected)
                    packed_streams = jax.device_put(np.frombuffer(b"".join(streams), dtype=np.uint8), selected)
                    jax.block_until_ready((resident_inputs, resident_streams, packed_input, packed_streams))
                    capacities = _codec._batch_capacities(sizes, args.chunk_bytes)
                    single_compress = jax.jit(lambda values: tuple(
                        cuda_zlib.compress_zlib_padded(value, selected, chunk_bytes=args.chunk_bytes) for value in values))
                    batch_compress = jax.jit(lambda value: cuda_zlib.compress_zlib_batch_padded(
                        value, sizes, selected, chunk_bytes=args.chunk_bytes))
                    single_decode = jax.jit(lambda values: tuple(
                        cuda_zlib.decompress_zlib_checked(value, size, selected) for value in values))
                    batch_decode = jax.jit(lambda value: cuda_zlib.decompress_zlib_batch_checked(
                        value, stream_sizes, sizes, selected))
                    value, timings["cuda_compress_single_resident"] = measure(
                        lambda: single_compress(resident_inputs), args.samples, total_bytes, count, jax.block_until_ready)
                    single_encoded = check_single_compressed(value, inputs)
                    value, timings["cuda_compress_batch_resident"] = measure(
                        lambda: batch_compress(packed_input), args.samples, total_bytes, count, jax.block_until_ready)
                    cuda_encoded = compression_slots(value, capacities, inputs)
                    if cuda_encoded != single_encoded:
                        raise AssertionError("batch and single compression streams differ")
                    value, timings["cuda_decompress_single_resident"] = measure(
                        lambda: single_decode(resident_streams), args.samples, total_bytes, count, jax.block_until_ready)
                    check_single_decoded(value, inputs)
                    value, timings["cuda_decompress_batch_resident"] = measure(
                        lambda: batch_decode(packed_streams), args.samples, total_bytes, count, jax.block_until_ready)
                    check_batch_decoded(value, inputs)
                    value, timings["cuda_compress_single_host_bytes"] = measure(
                        lambda: tuple(np.asarray(cuda_zlib.compress_zlib(payload, selected, chunk_bytes=args.chunk_bytes)).tobytes()
                                      for payload in inputs), args.samples, total_bytes, count)
                    check_compressed(value, inputs)
                    if value != cuda_encoded:
                        raise AssertionError("host and resident compression streams differ")
                    value, timings["cuda_compress_batch_host_bytes"] = measure(
                        lambda: cuda_zlib.compress_zlib_batch_host(inputs, selected, chunk_bytes=args.chunk_bytes),
                        args.samples, total_bytes, count)
                    check_compressed(value, inputs)
                    if value != cuda_encoded:
                        raise AssertionError("host and resident batch compression streams differ")
                    value, timings["cuda_decompress_single_host_bytes"] = measure(
                        lambda: tuple(cuda_zlib.decompress_zlib_host(stream, size, selected).tobytes() for stream in streams),
                        args.samples, total_bytes, count)
                    if value != inputs:
                        raise AssertionError("host single decompression differs from input")
                    value, timings["cuda_decompress_batch_host_bytes"] = measure(
                        lambda: tuple(array.tobytes() for array in cuda_zlib.decompress_zlib_batch_host(streams, sizes, selected)),
                        args.samples, total_bytes, count)
                    if value != inputs:
                        raise AssertionError("host batch decompression differs from input")
                    if args.roundtrip:
                        def roundtrip(value):
                            encoded, encode_metadata = cuda_zlib.compress_zlib_batch_padded(
                                value, sizes, selected, chunk_bytes=args.chunk_bytes)
                            decoded, decode_metadata = cuda_zlib.decompress_zlib_batch_checked(
                                encoded, capacities, sizes, selected, encoded_metadata=encode_metadata)
                            return decoded, decode_metadata, encode_metadata
                        compiled_roundtrip = jax.jit(roundtrip)
                        value, timings["cuda_roundtrip_batch_resident"] = measure(
                            lambda: compiled_roundtrip(packed_input), args.samples, total_bytes, count, jax.block_until_ready)
                        check_batch_decoded(value[:2], inputs)
                        if np.any(np.asarray(value[2])[:, 1]):
                            raise AssertionError("round-trip compression returned a failure status")
                    del resident_inputs, resident_streams, packed_input, packed_streams, value
                encoded = {"zlib1": cpu_encoded, "zlib6": streams}
                if cuda_encoded is not None:
                    encoded["cuda"] = cuda_encoded
                report["cases"].append({
                    "workload": workload, "file_bytes": size, "file_count": count,
                    "total_input_bytes": total_bytes,
                    "input_sha256": [sha256(payload) for payload in inputs],
                    "encoded_bytes": {key: [len(stream) for stream in values] for key, values in encoded.items()},
                    "encoded_sha256": {key: [sha256(stream) for stream in values] for key, values in encoded.items()},
                    "timings": timings,
                    "validation": "last returned output from each timed workflow byte-exact; every returned compressed stream decoded by stdlib",
                })
                save(report, args.output)
                print(f"pass {workload}: {count} files x {size} bytes", flush=True)
                gc.collect()
    if not args.cpu_only:
        report["environment"]["gpu_snapshot_after"] = gpu_snapshot()
        report["environment"]["workspace_pool_after"] = cuda_zlib.workspace_pool_stats(selected)
    report["complete"] = True
    save(report, args.output)
    print(f"complete {len(report['cases'])} cases: {args.output}", flush=True)


if __name__ == "__main__":
    main()
