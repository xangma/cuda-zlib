#!/usr/bin/env python3
# Copyright (c) 2026 xangma
# SPDX-License-Identifier: MIT
"""Separate warm API latency from CUDA-event kernel timings on fixed inputs."""

import argparse
import hashlib
import json
import platform
import time
import zlib
from pathlib import Path

import cupy as cp
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
    args = parser.parse_args()
    if args.samples < 1 or any(not 0 < size < 256 * 1024**2 for size in args.sizes):
        parser.error('positive samples and input sizes below 256 MiB required')
    cp.cuda.Device(args.device).use()
    cuda_zlib.compile_kernels(args.device)
    originals = {name: getattr(zlib, name) for name in
                 ('compress', 'decompress', 'compressobj', 'decompressobj')}
    real_encode, real_decode = _codec._encoder_module, _codec._module
    events = []

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

    def wrap(name, kernel):
        def call(*a, **kw):
            start, end = cp.cuda.Event(), cp.cuda.Event()
            start.record()
            kernel(*a, **kw)
            end.record()
            events.append((name, start, end))
        return call

    encode = {name: wrap(name, k) for name, k in real_encode(args.device).items()}
    decode = {name: wrap(name, k) for name, k in real_decode(args.device).items()}
    report = {
        'schema_version': 1,
        'arguments': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        'environment': {'gpu_before': gpu_snapshot(), 'python': platform.python_version(),
                        'cupy': cp.__version__, 'numpy': np.__version__,
                        'zlib_runtime': zlib.ZLIB_RUNTIME_VERSION},
        'source_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in Path(cuda_zlib.__file__).parent.glob('*.py')},
        'methodology': 'Warm synchronized API wall timings; separate one-call CUDA-event profile. '
                       'Generation, validation and stream transfers excluded. CPU codec forbidden '
                       'inside timed/profiled calls. Frozen streams permit identical decode comparisons.',
        'cases': [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.save_streams:
        args.save_streams.mkdir(parents=True, exist_ok=True)

    def run(fn, validate, size):
        result, wall = measure(lambda: guarded(fn), args.samples, size,
                               cp.cuda.runtime.deviceSynchronize)
        validate(result)
        del result
        events.clear()
        _codec._encoder_module = lambda device: encode
        _codec._module = lambda device: decode
        try:
            start = time.perf_counter()
            result = guarded(fn)
            elapsed = time.perf_counter() - start
            validate(result)
            stages, launches = {}, {}
            for name, before, after in events:
                stages[name] = stages.get(name, 0.0) + cp.cuda.get_elapsed_time(before, after)
                launches[name] = launches.get(name, 0) + 1
            return result, dict(wall=wall, instrumented_seconds=elapsed,
                                kernel_ms=stages, launches=launches)
        finally:
            _codec._encoder_module, _codec._module = real_encode, real_decode

    for size in args.sizes:
        for workload in args.workloads:
            raw = make_payload(workload, size, args.seed)
            device_raw = cp.asarray(np.frombuffer(raw, dtype=np.uint8))
            cp.cuda.runtime.deviceSynchronize()
            stream, enc = run(lambda: cuda_zlib.compress_zlib(device_raw, args.device),
                              lambda out: check(originals['decompress'](out.get().tobytes()), raw), size)
            own = stream.get().tobytes()
            stem = f'{workload}-{size}'
            if args.save_streams:
                (args.save_streams / (stem + '.zlib')).write_bytes(own)
            if args.streams_from:
                own = (args.streams_from / (stem + '.zlib')).read_bytes()
            timings = {'compress': enc}
            stdlib_stream = originals['compress'](raw, 6)
            for name, payload in [('decode_frozen', own), ('decode_zlib6', stdlib_stream)]:
                device_stream = cp.asarray(np.frombuffer(payload, dtype=np.uint8))
                cp.cuda.runtime.deviceSynchronize()
                out, timings[name] = run(
                    lambda: cuda_zlib.decompress_zlib(device_stream, size, args.device),
                    lambda out: check(out.get().tobytes(), raw), size)
                del out, device_stream
            report['cases'].append(dict(workload=workload, input_bytes=size,
                input_sha256=hashlib.sha256(raw).hexdigest(), encoded_bytes=int(stream.size),
                encoded_sha256=hashlib.sha256(stream.get().tobytes()).hexdigest(),
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
