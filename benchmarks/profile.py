#!/usr/bin/env python3
# Copyright (c) 2026 xangma
# SPDX-License-Identifier: MIT
"""Warm FFI API timings on fixed inputs; optional ranges for an external CUDA profiler."""

import argparse
import hashlib
import json
import platform
import time
import zlib
from pathlib import Path

import ctypes
import jax
import jaxlib
import numpy as np
import cuda_zlib
from cuda_zlib import _codec
from benchmark import WORKLOADS, make_payload, measure, gpu_snapshot


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sizes', nargs='+', type=int, default=[65536, 1048576, 67108864])
    parser.add_argument('--workloads', nargs='+', choices=WORKLOADS, default=list(WORKLOADS))
    parser.add_argument('--samples', type=int, default=5)
    parser.add_argument('--seed', type=int, default=20261006)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--save-streams', type=Path)
    parser.add_argument('--streams-from', type=Path)
    parser.add_argument('--cuda-profiler-range', action='store_true',
                        help='bracket each extra validated call with cudaProfilerStart/Stop for Nsight')
    args = parser.parse_args()
    if args.samples < 1 or any(not 0 < size < 256 * 1024**2 for size in args.sizes):
        parser.error('positive samples and input sizes below 256 MiB required')
    device = _codec._select_device(args.device)
    cuda_zlib.compile_kernels(args.device)
    originals = {name: getattr(zlib, name) for name in
                 ('compress', 'decompress', 'compressobj', 'decompressobj')}
    profiler = None
    if args.cuda_profiler_range:
        from cuda_zlib import _ffi
        toolkit = Path(_ffi._nvcc()).resolve().parent.parent
        profiler = ctypes.CDLL(str(toolkit / 'lib64/libcudart.so'))
        profiler.cudaProfilerStart.restype = ctypes.c_int
        profiler.cudaProfilerStop.restype = ctypes.c_int

    def forbidden(*a, **kw):
        raise AssertionError('CPU codec used in CUDA workflow')

    def guarded(fn):
        for name in originals:
            setattr(zlib, name, forbidden)
        try:
            return fn()
        finally:
            for name, fn in originals.items():
                setattr(zlib, name, fn)

    report = {
        'schema_version': 2,
        'arguments': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        'environment': {'gpu_before': gpu_snapshot(), 'python': platform.python_version(),
                        'jax': jax.__version__, 'jaxlib': jaxlib.__version__, 'numpy': np.__version__,
                        'backend': 'JAX typed CUDA FFI',
                        'cuda_platform': device.client.platform_version,
                        'zlib_runtime': zlib.ZLIB_RUNTIME_VERSION},
        'source_sha256': {str(p.relative_to(Path(cuda_zlib.__file__).parent)): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in sorted([*Path(cuda_zlib.__file__).parent.glob('*.py'),
                                           *Path(cuda_zlib.__file__).parent.glob('native/*.cu')])},
        'methodology': 'Warm completed API wall timings; one extra validated call per workflow. '
                       'Native build and per-workflow XLA compilation excluded. '
                       'Kernel durations require external Nsight capture; optional CUDA profiler API ranges. '
                       'Generation, validation and stream transfers excluded. CPU codec forbidden '
                       'inside timed/profiled calls. Frozen streams permit identical decode comparisons.',
        'cases': [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.save_streams:
        args.save_streams.mkdir(parents=True, exist_ok=True)

    def run(fn, validate, size):
        result, wall = measure(lambda: guarded(fn), args.samples, size)
        validate(result)
        del result
        if profiler and profiler.cudaProfilerStart():
            raise RuntimeError('cudaProfilerStart failed')
        try:
            start = time.perf_counter()
            result = guarded(fn)
            result.block_until_ready()
            elapsed = time.perf_counter() - start
        finally:
            if profiler and profiler.cudaProfilerStop():
                raise RuntimeError('cudaProfilerStop failed')
        validate(result)
        return result, dict(wall=wall, profiled_call_seconds=elapsed)

    for size in args.sizes:
        for workload in args.workloads:
            raw = make_payload(workload, size, args.seed)
            device_raw = jax.device_put(np.frombuffer(raw, dtype=np.uint8), device)
            device_raw.block_until_ready()
            stream, enc = run(lambda: cuda_zlib.compress_zlib(device_raw, args.device),
                              lambda out: check(originals['decompress'](np.asarray(out).tobytes()), raw), size)
            own = np.asarray(stream).tobytes()
            stem = f'{workload}-{size}'
            if args.save_streams:
                (args.save_streams / (stem + '.zlib')).write_bytes(own)
            if args.streams_from:
                own = (args.streams_from / (stem + '.zlib')).read_bytes()
            timings = {'compress': enc}
            stdlib_stream = originals['compress'](raw, 6)
            for name, payload in [('decode_frozen', own), ('decode_zlib6', stdlib_stream)]:
                device_stream = jax.device_put(np.frombuffer(payload, dtype=np.uint8), device)
                device_stream.block_until_ready()
                out, timings[name] = run(
                    lambda: cuda_zlib.decompress_zlib(device_stream, size, args.device),
                    lambda out: check(np.asarray(out).tobytes(), raw), size)
                del out, device_stream
            report['cases'].append(dict(workload=workload, input_bytes=size,
                input_sha256=hashlib.sha256(raw).hexdigest(), encoded_bytes=int(stream.size),
                encoded_sha256=hashlib.sha256(np.asarray(stream).tobytes()).hexdigest(),
                frozen_stream_sha256=hashlib.sha256(own).hexdigest(),
                stdlib_stream_sha256=hashlib.sha256(stdlib_stream).hexdigest(),
                timings=timings, byte_exact=True, cpu_codec_forbidden=True))
            args.output.write_text(json.dumps(report, indent=2) + '\n')
            print(stem, {k: round(v['wall']['median_seconds'] * 1000, 3)
                         for k, v in timings.items()}, flush=True)
            del stream, device_raw
    report['environment']['gpu_after'] = gpu_snapshot()
    args.output.write_text(json.dumps(report, indent=2) + '\n')


def check(actual, expected):
    if actual != expected:
        raise AssertionError('byte-exact validation failed')


if __name__ == '__main__':
    main()
