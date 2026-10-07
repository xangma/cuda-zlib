#!/usr/bin/env python3
# Copyright (c) 2026 xangma
# SPDX-License-Identifier: MIT
"""Profile completed resident JIT checked decompression, warming every shape first.

Capture mode retains every case's output and metadata until its single continuous
CUDA profiler range ends. Input streams for every case also remain resident.
Memory therefore grows with the sum of requested input and output sizes. Normal
timing mode retains only the latest output. Stream generation, upload, compilation,
status validation and byte comparisons happen outside timed/profiled calls.
"""

import argparse
import ctypes
import hashlib
import json
from pathlib import Path
import platform
import statistics
import time
import zlib

import jax
import jaxlib
import numpy as np
import cuda_zlib
from cuda_zlib import _codec, _ffi
from benchmark import WORKLOADS, make_payload, gpu_snapshot


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sizes', nargs='+', type=int, default=[65536, 1048576, 67108864])
    parser.add_argument('--workloads', nargs='+', choices=WORKLOADS, default=list(WORKLOADS))
    parser.add_argument('--samples', type=int, default=5,
                        help='normal timing samples; capture uses one call per case')
    parser.add_argument('--seed', type=int, default=20261006)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cuda-profiler-range', action='store_true',
                        help='capture all warmed cases in one continuous cudaProfilerStart/Stop range')
    args = parser.parse_args()
    if args.samples < 1 or any(not 0 < size < 256 * 1024**2 for size in args.sizes):
        parser.error('positive samples and input sizes below 256 MiB required')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    device = _codec._select_device(args.device)
    cuda_zlib.compile_kernels(device)
    library, targets = _ffi._backend(device)
    library_path = Path(library._name).resolve()
    build_path = library_path.parent / 'build.json'
    native_build = {
        'cache_key': library_path.parent.name,
        'library_sha256': hashlib.sha256(library_path.read_bytes()).hexdigest(),
        'build_sha256': hashlib.sha256(build_path.read_bytes()).hexdigest(),
        'identity': json.loads(build_path.read_text()),
        'ffi_targets': list(targets),
    }
    originals = {name: getattr(zlib, name) for name in
                 ('compress', 'decompress', 'compressobj', 'decompressobj')}

    def forbidden(*a, **kw):
        raise AssertionError('CPU codec used in CUDA workflow')

    def completed(fn):
        for name in originals:
            setattr(zlib, name, forbidden)
        try:
            return jax.block_until_ready(fn())
        finally:
            for name, original in originals.items():
                setattr(zlib, name, original)

    def check(result, raw):
        data, metadata = result
        metadata = np.asarray(metadata)
        _codec._check_value(int(metadata[0]), 'CUDA resident decompression')
        if metadata[1]:
            raise AssertionError('unexpected decompression metadata')
        if np.asarray(data).tobytes() != raw:
            raise AssertionError('byte-exact validation failed')

    source = Path(cuda_zlib.__file__).resolve().parent
    report = {
        'schema_version': 1, 'complete': False,
        'harness_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'arguments': {key: str(value) if isinstance(value, Path) else value
                      for key, value in vars(args).items()},
        'environment': {'gpu_before': gpu_snapshot(), 'python': platform.python_version(),
                        'jax': jax.__version__, 'jaxlib': jaxlib.__version__,
                        'numpy': np.__version__, 'backend': 'JAX typed CUDA FFI',
                        'cuda_platform': device.client.platform_version,
                        'zlib_runtime': zlib.ZLIB_RUNTIME_VERSION},
        'source_sha256': {str(path.relative_to(source)): hashlib.sha256(path.read_bytes()).hexdigest()
                          for path in sorted(source.rglob('*'))
                          if path.suffix in ('.py', '.cu', '.cuh')},
        'native_build': native_build,
        'methodology': 'Completed warm JIT checked decompression with resident inputs. '
                       'Native build, XLA compilation, generation, upload, status and byte checks '
                       'excluded. CPU codec forbidden inside calls. All shapes warmed before '
                       'timing or capture. Normal timings sample completed calls; capture uses '
                       'one call per case in a single continuous range with NVTX labels and '
                       'retains output and metadata until stop. Capture wall timings are diagnostic. '
                       'Status is checked before consuming output bytes after capture.',
        'all_workflows_warmed_before_capture': False,
        'cases': [],
    }
    plans = []
    for size in args.sizes:
        for workload in args.workloads:
            raw = make_payload(workload, size, args.seed)
            stream = originals['compress'](raw, 6)
            if originals['decompress'](stream) != raw:
                raise AssertionError('input stream oracle failed')
            resident = jax.device_put(np.frombuffer(stream, np.uint8), device)
            resident.block_until_ready()
            decoder = jax.jit(lambda value, extent=size:
                              cuda_zlib.decompress_zlib_checked(value, extent, device))
            fn = lambda call=decoder, data=resident: call(data)
            result = completed(fn)
            check(result, raw)
            del result
            row = dict(workload=workload, input_bytes=size, encoded_bytes=len(stream),
                       input_sha256=hashlib.sha256(raw).hexdigest(),
                       stream_sha256=hashlib.sha256(stream).hexdigest())
            plans.append((row, fn, raw))
    report['all_workflows_warmed_before_capture'] = True

    def record(row, result, raw, samples):
        check(result, raw)
        row.update(seconds=samples, median_seconds=statistics.median(samples),
                   byte_exact=True, cpu_codec_forbidden=True)
        report['cases'].append(row)

    if args.cuda_profiler_range:
        from cuda_zlib import _ffi
        toolkit = Path(_ffi._nvcc()).resolve().parent.parent
        profiler = ctypes.CDLL(str(toolkit / 'lib64/libcudart.so'))
        profiler.cudaSetDevice.argtypes = [ctypes.c_int]
        profiler.cudaSetDevice.restype = ctypes.c_int
        profiler.cudaProfilerStart.restype = ctypes.c_int
        profiler.cudaProfilerStop.restype = ctypes.c_int
        nvtx = ctypes.CDLL(str(toolkit / 'lib64/libnvToolsExt.so'))
        nvtx.nvtxRangePushA.argtypes = [ctypes.c_char_p]
        nvtx.nvtxRangePushA.restype = ctypes.c_int
        nvtx.nvtxRangePop.restype = ctypes.c_int
        if profiler.cudaSetDevice(int(device.local_hardware_id)):
            raise RuntimeError('cudaSetDevice failed for profiler range')
        captured = []
        if profiler.cudaProfilerStart():
            raise RuntimeError('cudaProfilerStart failed')
        try:
            for row, fn, raw in plans:
                nvtx.nvtxRangePushA(f"decode:{row['workload']}:{row['input_bytes']}".encode())
                try:
                    start = time.perf_counter()
                    result = completed(fn)
                    elapsed = time.perf_counter() - start
                finally:
                    nvtx.nvtxRangePop()
                captured.append((row, result, raw, elapsed))
        finally:
            if profiler.cudaProfilerStop():
                raise RuntimeError('cudaProfilerStop failed')
        for row, result, raw, elapsed in captured:
            record(row, result, raw, [elapsed])
        report['capture'] = 'One warmed continuous range; status and oracle reads after stop.'
    else:
        for row, fn, raw in plans:
            samples = []
            result = None
            for _ in range(args.samples):
                del result
                start = time.perf_counter()
                result = completed(fn)
                samples.append(time.perf_counter() - start)
            record(row, result, raw, samples)
            del result
            args.output.write_text(json.dumps(report, indent=2) + '\n')
            print(row['workload'], row['input_bytes'],
                  round(row['median_seconds'] * 1000, 3), flush=True)
    # Avoid nvidia-smi subprocesses while Nsight is awaiting deferred export.
    report['environment']['gpu_after'] = None if args.cuda_profiler_range else gpu_snapshot()
    report['environment']['workspace_pool_after'] = None if args.cuda_profiler_range else \
        cuda_zlib.workspace_pool_stats(device)
    report['complete'] = True
    args.output.write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
