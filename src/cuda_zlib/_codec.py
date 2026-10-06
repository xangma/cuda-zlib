# Copyright (c) 2026 xangma
# SPDX-License-Identifier: MIT

"""Experimental fresh RFC 1950 decoding on CUDA, without a prepared index.

GPU-discovered blocks supply parallel work. Exact boundary following selects
the real stream; long fixed regions use fresh GPU token-boundary summaries.
Dynamic blocks and fixed segments decode in parallel; stored bytes copy in parallel.
No CPU inflation, recompression, decoded checksum cache, or nvCOMP is used.
CuPy supplies allocation, NVRTC compilation and kernel launch only. Output is
owned by each call after validation of the original Adler32. Container checksums
remain the caller's responsibility; JAX conversion lives in a separate bridge.
"""

import contextlib
import functools
import threading

import numpy as np

from ._errors import BackendUnavailable, CodecError, UnsupportedStream
from ._decode_kernels import KERNEL_NAMES, STATUS_MESSAGES




_MAX_BYTES = 2**28
_MAX_CANDIDATES = 262144
_MAX_BLOCKS = 262144
_FIXED_TILE_BYTES = 2048
_WORKSPACE_POOL_LIMIT = 3 * 2**30
_RESOURCE_CREATION_LOCK = threading.Lock()
_NAMES = KERNEL_NAMES + (
    "refine_roots", "write_adler_parts", "adler_parts", "adler_finish",
)

# Each refinement reads an immutable predecessor map. Local copies have already
# been flattened by the block emitter. Ping-pong rounds bound even pathological
# cross-block chains without concurrent in-place updates or host data.
_POSTPROCESS = r'''
extern "C" __global__ void refine_roots(
    const unsigned int* input, unsigned int* output, unsigned int size,
    unsigned int* pending, unsigned int* status) {
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= size) return;
  unsigned int value = input[i];
  for (int step = 0; step < 32 && !(value & 0x80000000u); ++step) {
    if (value >= i) { atomicExch(status, 6u); return; }
    unsigned int next = input[value];
    if (!(next & 0x80000000u) && next >= value) {
      atomicExch(status, 6u); return;
    }
    value = next;
  }
  output[i] = value;
  if (!(value & 0x80000000u)) atomicExch(pending, 1u);
}
// Resolved roots hold tagged literals. Gather the returned bytes while making
// checksum partials, avoiding a separate gather launch and output read.
extern "C" __global__ void write_adler_parts(
    const unsigned int* roots, unsigned char* output, unsigned int size,
    unsigned long long* partial_a, unsigned long long* partial_b) {
  __shared__ unsigned long long a[256], b[256];
  unsigned int i = blockIdx.x * 4096u + threadIdx.x;
  unsigned long long sa = 0, sb = 0;
  for (unsigned int j = 0; j < 16u; ++j, i += 256u) {
    if (i < size) {
      unsigned int value = (unsigned char)roots[i];
      output[i] = (unsigned char)value;
      sa += value;
      sb += (unsigned long long)(size - i) * value;
    }
  }
  a[threadIdx.x] = sa; b[threadIdx.x] = sb;
  __syncthreads();
  for (int stride = 128; stride; stride >>= 1) {
    if (threadIdx.x < stride) {
      a[threadIdx.x] += a[threadIdx.x + stride];
      b[threadIdx.x] += b[threadIdx.x + stride];
    }
    __syncthreads();
  }
  if (!threadIdx.x) {
    partial_a[blockIdx.x] = a[0]; partial_b[blockIdx.x] = b[0];
  }
}
extern "C" __global__ void adler_parts(
    const unsigned char* data, unsigned int size,
    unsigned long long* partial_a, unsigned long long* partial_b) {
  __shared__ unsigned long long a[256], b[256];
  unsigned int i = blockIdx.x * 4096u + threadIdx.x;
  unsigned long long sa = 0, sb = 0;
  for (unsigned int j = 0; j < 16u; ++j, i += 256u) {
    if (i < size) {
      unsigned int value = data[i];
      sa += value;
      sb += (unsigned long long)(size - i) * value;
    }
  }
  a[threadIdx.x] = sa; b[threadIdx.x] = sb;
  __syncthreads();
  for (int stride = 128; stride; stride >>= 1) {
    if (threadIdx.x < stride) {
      a[threadIdx.x] += a[threadIdx.x + stride];
      b[threadIdx.x] += b[threadIdx.x + stride];
    }
    __syncthreads();
  }
  if (!threadIdx.x) {
    partial_a[blockIdx.x] = a[0]; partial_b[blockIdx.x] = b[0];
  }
}
extern "C" __global__ void adler_finish(
    const unsigned long long* partial_a,
    const unsigned long long* partial_b, unsigned int parts,
    unsigned int size, unsigned int* checksum) {
  __shared__ unsigned long long a[256], b[256];
  unsigned long long sa = 0, sb = 0;
  for (unsigned int i = threadIdx.x; i < parts; i += 256u) {
    sa += partial_a[i]; sb += partial_b[i];
  }
  a[threadIdx.x] = sa; b[threadIdx.x] = sb;
  __syncthreads();
  for (int stride = 128; stride; stride >>= 1) {
    if (threadIdx.x < stride) {
      a[threadIdx.x] += a[threadIdx.x + stride];
      b[threadIdx.x] += b[threadIdx.x + stride];
    }
    __syncthreads();
  }
  if (!threadIdx.x) {
    // size <= 2^28 keeps both unreduced 64-bit sums within range.
    unsigned int low = (unsigned int)((a[0] + 1u) % 65521u);
    unsigned int high = (unsigned int)((b[0] + size) % 65521u);
    *checksum = (high << 16) | low;
  }
}
'''



def _validate_size(size):
    if type(size) is not int or not 0 <= size <= _MAX_BYTES:
        raise ValueError("expected_bytes must be an integer in [0, 2**28]")


def _validate_device(device):
    if type(device) is not int or device < 0:
        raise BackendUnavailable("device must be a nonnegative CUDA ordinal")


def _cupy():
    try:
        import cupy as cp
    except ImportError as exc:
        raise BackendUnavailable("CUDA byte codec requires CuPy") from exc
    return cp


def _device_input(data, cp, device):
    if not isinstance(data, cp.ndarray):
        raise TypeError("input must be bytes or a CuPy array")
    if data.dtype != cp.uint8 or data.ndim != 1 or not data.flags.c_contiguous:
        raise TypeError("CuPy input must be contiguous one-dimensional uint8")
    if data.device.id != device:
        raise ValueError("input array and requested CUDA device differ")
    if int(data.size) > _MAX_BYTES:
        raise ValueError("input exceeds 2**28 bytes")
    return data


def _framing(payload, expected_bytes):
    if not isinstance(payload, bytes):
        raise TypeError("compressed payload must be bytes")
    if (type(expected_bytes) is not int
            or not 0 <= expected_bytes <= _MAX_BYTES):
        raise ValueError("expected_bytes must be an integer in [0, 2**28]")
    if len(payload) < 8:
        raise CodecError("truncated zlib payload")
    if len(payload) > _MAX_BYTES:
        raise ValueError("compressed payload exceeds 2**28 bytes")
    if payload[:2] == b"\x1f\x8b":
        raise UnsupportedStream("native CUDA codec supports RFC 1950 zlib")
    cmf, flg = payload[:2]
    if cmf & 15 != 8 or cmf >> 4 > 7 or (cmf * 256 + flg) % 31:
        raise CodecError("invalid zlib CMF/FLG header")
    if flg & 32:
        raise UnsupportedStream("zlib preset dictionaries are unsupported")
    return memoryview(payload)[2:-4], int.from_bytes(payload[-4:], "big")


@functools.lru_cache(maxsize=8)
def _module(device_id):
    import cupy as cp
    from ._decode_kernels import CUDA_SOURCE

    # CuPy's disk cache includes source/options/compiler/GPU architecture.
    # Only compiled code is cached here, never stream-derived metadata.
    with cp.cuda.Device(device_id):
        module = cp.RawModule(
            code=CUDA_SOURCE + _POSTPROCESS, options=("-std=c++11",),
            name_expressions=_NAMES,
        )
        return {name: module.get_function(name) for name in _NAMES}


def compile_kernels(device=0):
    """Compile/load kernels explicitly to separate startup timing."""
    _validate_device(device)
    _cupy()
    return {"decode": _module(device), "encode": _encoder_module(device)}


@functools.lru_cache(maxsize=None)
def _resources_cached(device_id):
    """Reuse a bounded device workspace pool, serialized on its own stream."""
    import cupy as cp

    with cp.cuda.Device(device_id):
        pool = cp.cuda.MemoryPool()
        pool.set_limit(size=_WORKSPACE_POOL_LIMIT)
        return threading.RLock(), cp.cuda.Stream(non_blocking=True), pool


def _resources(device_id):
    # lru_cache alone permits simultaneous cache misses to create two locks.
    with _RESOURCE_CREATION_LOCK:
        return _resources_cached(device_id)


@contextlib.contextmanager
def _workspace(device_id):
    import cupy as cp

    lock, stream, pool = _resources(device_id)
    with lock, cp.cuda.Device(device_id):
        # Respect the caller's current stream before using our private stream.
        ready = cp.cuda.Event(disable_timing=True)
        ready.record(cp.cuda.get_current_stream())
        stream.wait_event(ready)
        with stream, cp.cuda.using_allocator(pool.malloc):
            try:
                yield stream
            finally:
                stream.synchronize()


def _check(status, operation):
    _check_value(int(status.get()[0]), operation)


def _check_value(value, operation):
    if value:
        message = STATUS_MESSAGES.get(value, "unknown error")
        raise CodecError("%s: %s (Deflate status %d)" %
                             (operation, message, value))


def decompress_zlib(payload, expected_bytes, device=0):
    """Decode zlib bytes to an independently owned CuPy CUDA uint8 array.

    Malformed streams, wrong lengths/checksums, and workspace overflow fail
    explicitly. Unsupported devices/dependencies never trigger CPU inflation.
    Compilation occurs lazily; call :func:`compile_kernels` before timing when
    compilation is to be excluded. Every invocation rediscovers the input.
    """
    _validate_size(expected_bytes)
    if isinstance(payload, bytes):
        body, checksum = _framing(payload, expected_bytes)
    elif not hasattr(payload, "__cuda_array_interface__"):
        raise TypeError("payload must be bytes or a contiguous CuPy uint8 array")
    _validate_device(device)
    cp = _cupy()
    u32 = np.uint32
    with _workspace(device) as stream:
        kernel = _module(device)
        if isinstance(payload, bytes):
            data = cp.asarray(np.frombuffer(body, dtype=np.uint8))
            cmf = payload[0]
        else:
            data = _device_input(payload, cp, device)
            size = int(data.size)
            if not 8 <= size <= _MAX_BYTES:
                raise CodecError("compressed payload extent outside [8, 2**28]")
            header = data[:2].get().tobytes()
            trailer = data[-4:].get().tobytes()
            _, checksum = _framing(header + b"\x03\x00" + trailer, expected_bytes)
            cmf = header[0]
            data = data[2:-4]
        length = int(data.size)
        window = 1 << ((cmf >> 4) + 8)
        starts = cp.empty(_MAX_CANDIDATES, dtype=cp.uint64)
        count = cp.zeros(1, dtype=cp.uint32)
        kernel["discover"]((min(16384, (length + 127) // 128),), (128,),
                           (data, u32(length), starts, count,
                            u32(_MAX_CANDIDATES)))
        candidates = int(count.get()[0])
        if candidates > _MAX_CANDIDATES:
            raise CodecError("Deflate candidate workspace overflow")
        if not candidates:
            raise CodecError("Deflate initial block missing")
        starts = cp.sort(starts[:candidates])
        ends = cp.empty(candidates, dtype=cp.uint64)
        sizes = cp.empty(candidates, dtype=cp.uint32)
        finals = cp.empty(candidates, dtype=cp.uint32)
        status = cp.empty(candidates, dtype=cp.uint32)
        kernel["describe_candidates"](
            (candidates,), (1,),
            (data, u32(length), starts, u32(candidates), u32(expected_bytes),
             ends, sizes, finals, status),
        )
        block_starts = cp.empty(_MAX_BLOCKS, dtype=cp.uint64)
        block_ends = cp.empty(_MAX_BLOCKS, dtype=cp.uint64)
        prefix = cp.empty(_MAX_BLOCKS, dtype=cp.uint32)
        block_sizes = cp.empty(_MAX_BLOCKS, dtype=cp.uint32)
        block_count = cp.zeros(1, dtype=cp.uint32)
        chain_status = cp.zeros(1, dtype=cp.uint32)
        chain_args = (data, u32(length), starts, ends, sizes, finals, status,
                      u32(candidates), u32(expected_bytes), block_starts,
                      block_ends, prefix, block_sizes, block_count,
                      u32(_MAX_BLOCKS), chain_status)
        # These pointers are unused until the exact stream requests summaries.
        # No speculative fixed candidate can force this additional work.
        dummy_summaries = (ends, sizes, ends, sizes, status, status)
        kernel["select_chain"]((1,), (1,),
                               chain_args + (u32(0),) + dummy_summaries)
        if int(chain_status.get()[0]) == 13:
            tiles = (length + _FIXED_TILE_BYTES - 1) // _FIXED_TILE_BYTES
            entries = tiles * 32
            summaries = tuple(cp.empty(entries, dtype=dtype) for dtype in
                              (cp.uint64, cp.uint32, cp.uint64, cp.uint32,
                               cp.uint32, cp.uint32))
            kernel["fixed_summaries"](
                (min(16384, tiles),), (32,),
                (data, u32(length), u32(_FIXED_TILE_BYTES)) + summaries)
            block_count.fill(0)
            chain_status.fill(0)
            kernel["select_chain"](
                (1,), (1,), chain_args + (u32(_FIXED_TILE_BYTES),) + summaries)
        _check(chain_status, "Deflate boundary validation")
        blocks = int(block_count.get()[0])
        roots = cp.empty(expected_bytes, dtype=cp.uint32)
        emit_status = cp.zeros((2, blocks), dtype=cp.uint32)
        emit_args = (data, u32(length), block_starts, block_ends, prefix,
                     block_sizes, u32(blocks), u32(expected_bytes),
                     u32(window), roots, emit_status)
        kernel["emit_blocks"]((blocks,), (1,), emit_args)
        kernel["emit_stored"]((blocks,), (256,), emit_args)
        emission = cp.max(emit_status, axis=1).get()
        _check_value(int(emission[0]), "Deflate output emission")
        # Emitters flatten every local reference. Only accepted references to
        # earlier blocks/segments require another workspace and resolution.
        if int(emission[1]):
            alternate = cp.empty_like(roots)
            pending = cp.zeros(1, dtype=cp.uint32)
            root_status = cp.zeros(1, dtype=cp.uint32)
            grid = ((expected_bytes + 255) // 256,)
            for _ in range(28):
                pending.fill(0)
                kernel["refine_roots"](grid, (256,),
                                       (roots, alternate, u32(expected_bytes),
                                        pending, root_status))
                roots, alternate = alternate, roots
                _check(root_status, "Deflate back-reference resolution")
                if not int(pending.get()[0]):
                    break
            else:
                raise CodecError("Deflate back-reference depth exceeded")
        output = cp.empty(expected_bytes, dtype=cp.uint8)
        parts = max(1, (expected_bytes + 4095) // 4096)
        partial_a = cp.empty(parts, dtype=cp.uint64)
        partial_b = cp.empty_like(partial_a)
        actual_checksum = cp.empty(1, dtype=cp.uint32)
        kernel["write_adler_parts"]((parts,), (256,),
                                    (roots, output, u32(expected_bytes),
                                     partial_a, partial_b))
        kernel["adler_finish"]((1,), (256,),
                               (partial_a, partial_b, u32(parts),
                                u32(expected_bytes), actual_checksum))
        if int(actual_checksum.get()[0]) != checksum:
            raise CodecError("CUDA Adler32 mismatch")
        stream.synchronize()
        return output


@functools.lru_cache(maxsize=8)
def _encoder_module(device):
    cp = _cupy()
    from ._encode_kernels import CUDA_SOURCE, KERNEL_NAMES
    names = KERNEL_NAMES + ("adler_parts", "adler_finish")
    with cp.cuda.Device(device):
        module = cp.RawModule(code=CUDA_SOURCE + _POSTPROCESS,
                              options=("-std=c++11",), name_expressions=names)
        return {name: module.get_function(name) for name in names}


def compress_zlib(data, device=0, *, chunk_bytes=32768):
    """Encode bytes/CuPy uint8 to an owned, completed CuPy RFC 1950 stream.

    Parallel chunks use greedy LZ77 and select stored, fixed or dynamic coding.
    No CPU compression, input readback, dictionary, levels, or shared history.
    The worst-case encoded extent must fit the decoder's 256 MiB input bound.
    """
    if type(chunk_bytes) is not int or not 256 <= chunk_bytes <= 65535:
        raise ValueError("chunk_bytes must be an integer in [256, 65535]")
    if not isinstance(data, bytes) and not hasattr(data, "__cuda_array_interface__"):
        raise TypeError("input must be bytes or a contiguous CuPy uint8 array")
    if isinstance(data, bytes) and len(data) > _MAX_BYTES:
        raise ValueError("input exceeds 2**28 bytes")
    _validate_device(device)
    cp = _cupy()
    u32 = np.uint32
    with _workspace(device):
        raw = (cp.asarray(np.frombuffer(data, dtype=np.uint8))
               if isinstance(data, bytes) else _device_input(data, cp, device))
        size = int(raw.size)
        chunks = max(1, (size + chunk_bytes - 1) // chunk_bytes)
        if 2 * chunks - 1 > _MAX_BLOCKS:
            raise ValueError("chunk count exceeds decoder block limit")
        if size + chunks * 5 + 6 > _MAX_BYTES:
            raise ValueError("worst-case compressed extent exceeds 2**28 bytes")
        slot_bytes = (chunk_bytes * 9 + 7) // 8 + 16
        kernel = _encoder_module(device)
        scratch = cp.empty(chunks * slot_bytes, dtype=cp.uint8)
        tokens = cp.empty(max(1, size), dtype=cp.uint32)
        sizes = cp.empty(chunks, dtype=cp.uint32)
        status = cp.empty(chunks, dtype=cp.uint32)
        kernel["encode_chunks"](
            (chunks,), (256,), (raw, u32(size), u32(chunk_bytes), u32(chunks),
                               u32(slot_bytes), scratch, sizes, status, tokens))
        if int(cp.max(status).get()):
            raise CodecError("CUDA compression workspace overflow")
        ends = cp.cumsum(sizes, dtype=cp.uint64)
        encoded_size = int(ends[-1].get()) + 6
        output = cp.empty(encoded_size, dtype=cp.uint8)
        kernel["pack_chunks"]((chunks,), (256,),
                              (scratch, u32(slot_bytes), sizes, ends, output))
        parts = max(1, (size + 4095) // 4096)
        a = cp.empty(parts, dtype=cp.uint64)
        b = cp.empty_like(a)
        checksum = cp.empty(1, dtype=cp.uint32)
        kernel["adler_parts"]((parts,), (256,), (raw, u32(size), a, b))
        kernel["adler_finish"]((1,), (256,),
                               (a, b, u32(parts), u32(size), checksum))
        kernel["write_wrapper"]((1,), (1,),
                                (output, np.uint64(encoded_size), checksum))
        return output
