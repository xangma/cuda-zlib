# Copyright (c) 2026 xangma
# SPDX-License-Identifier: MIT

"""Bounded RFC 1950 CUDA codec using JAX-owned buffers and XLA FFI."""

import numpy as np

from ._errors import BackendUnavailable, CodecError, UnsupportedStream
from ._decode_kernels import STATUS_MESSAGES

_MAX_BYTES = 2**28
_MAX_CANDIDATES = 262144
_MAX_BLOCKS = 262144
_MESSAGES = {
    **STATUS_MESSAGES,
    14: "truncated zlib payload", 15: "invalid zlib CMF/FLG header",
    16: "native CUDA codec supports RFC 1950 zlib",
    17: "zlib preset dictionaries are unsupported",
    18: "Deflate candidate workspace overflow", 19: "Deflate initial block missing",
    20: "Deflate back-reference depth exceeded", 21: "CUDA Adler32 mismatch",
    22: "CUDA compression workspace overflow", 23: "invalid codec bounds",
}


def _validate_size(size):
    if type(size) is not int or not 0 <= size <= _MAX_BYTES:
        raise ValueError("expected_bytes must be an integer in [0, 2**28]")


def _validate_device(device):
    if type(device) is int:
        if device >= 0:
            return
    elif (getattr(device, "platform", None) == "gpu"
          and "cuda" in str(getattr(getattr(device, "client", None),
                                    "platform_version", "")).lower()):
        return
    raise BackendUnavailable("device must be a nonnegative CUDA ordinal or CUDA JAX device")


def _jax():
    try:
        import jax
        if not hasattr(jax, "ffi"):
            raise BackendUnavailable("CUDA byte codec requires JAX >= 0.11.2")
        return jax
    except ImportError as exc:
        raise BackendUnavailable("CUDA byte codec requires JAX") from exc


def _select_device(device):
    _validate_device(device)
    jax = _jax()
    if type(device) is not int:
        return device
    try:
        devices = jax.local_devices(backend="gpu")
    except RuntimeError as exc:
        raise BackendUnavailable("CUDA JAX backend unavailable") from exc
    for candidate in devices:
        if candidate.local_hardware_id == device:
            _validate_device(candidate)
            return candidate
    raise BackendUnavailable("requested CUDA device is unavailable")


def _input_size(data, label="input"):
    if isinstance(data, bytes):
        size = len(data)
    elif (type(data).__module__.startswith(("jax", "jaxlib"))
          and hasattr(data, "dtype") and hasattr(data, "ndim")):
        if data.dtype != np.uint8 or data.ndim != 1:
            raise TypeError("JAX input must be one-dimensional uint8")
        size = int(data.size)
    else:
        raise TypeError(label + " must be bytes or a one-dimensional JAX uint8 array")
    if size > _MAX_BYTES:
        raise ValueError(label + " exceeds size limit (2**28 bytes)")
    return size


def _device_input(data, jax, device):
    if isinstance(data, bytes):
        data = jax.device_put(np.frombuffer(data, dtype=np.uint8), device)
    if isinstance(data, jax.core.Tracer):
        return jax.lax.with_sharding_constraint(
            data, jax.sharding.SingleDeviceSharding(device))
    if not isinstance(data, jax.Array):
        raise TypeError("input must be bytes or a one-dimensional JAX uint8 array")
    if data.devices() != {device}:
        raise ValueError("input array and requested CUDA device differ")
    return data


def _framing(payload, expected_bytes):
    if not isinstance(payload, bytes):
        raise TypeError("compressed payload must be bytes")
    _validate_size(expected_bytes)
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


def _check_value(value, operation):
    if value:
        message = _MESSAGES.get(value, "unknown error")
        error = UnsupportedStream if value in (16, 17) else CodecError
        raise error("%s: %s (Deflate status %d)" % (operation, message, value))


def compile_kernels(device=0):
    """Build/load the native CUDA library; workflow XLA compilation remains lazy."""
    from ._ffi import load_backend
    load_backend(_select_device(device))


def workspace_pool_stats(device=0):
    """Return retention, reserved/live bytes and pool count for a CUDA device.

    Scratch pools retain up to a 1 GiB release threshold by default. Set
    CUDA_ZLIB_WORKSPACE_RETENTION_BYTES before the first backend load to change
    it; zero disables retention. The threshold is not a hard allocation cap.
    Concurrent operations may change these diagnostic values during the call.
    """
    from ._ffi import workspace_pool_stats as stats
    return stats(_select_device(device))


def trim_workspace_pool(device=0):
    """Try to release unused codec scratch memory on the requested device.

    This does not wait for pending operations or change JAX's allocation pool.
    Complete outstanding codec results first to make their scratch eligible.
    CUDA may retain pages containing live or pending allocations.
    """
    from ._ffi import trim_workspace_pool as trim
    trim(_select_device(device))


def compress_zlib_padded(data, device=0, *, chunk_bytes=32768):
    """Return ``(buffer, [encoded_length, status])`` on CUDA, also under jax.jit.

    The fixed-size buffer contains a zlib stream followed by zero padding.
    Status zero means success; callers must check status before using the stream.
    The device and chunk size must be static when tracing.
    """
    if type(chunk_bytes) is not int or not 256 <= chunk_bytes <= 65535:
        raise ValueError("chunk_bytes must be an integer in [256, 65535]")
    size = _input_size(data)
    chunks = max(1, (size + chunk_bytes - 1) // chunk_bytes)
    if 2 * chunks - 1 > _MAX_BLOCKS:
        raise ValueError("chunk count exceeds decoder block limit")
    capacity = size + chunks * 5 + 6
    if capacity > _MAX_BYTES:
        raise ValueError("worst-case compressed extent exceeds 2**28 bytes")
    selected = _select_device(device)
    jax = _jax()
    raw = _device_input(data, jax, selected)
    from ._ffi import load_backend
    encode, _ = load_backend(selected)
    call = jax.ffi.ffi_call(
        encode, (jax.ShapeDtypeStruct((capacity,), np.uint8),
                 jax.ShapeDtypeStruct((2,), np.uint32)),
        vmap_method="sequential")
    with jax.default_device(selected):
        return call(raw, chunk_bytes=np.int64(chunk_bytes))


def decompress_zlib_checked(payload, expected_bytes, device=0):
    """Return ``(output, [status, 0])`` on CUDA, also under jax.jit.

    Output length and device must be static. Status zero validates the original
    stream, output extent and Adler32; callers must check it before using output.
    """
    _validate_size(expected_bytes)
    size = _input_size(payload, "compressed payload")
    if isinstance(payload, bytes):
        _framing(payload, expected_bytes)
    elif size < 8:
        raise CodecError("truncated zlib payload")
    selected = _select_device(device)
    jax = _jax()
    raw = _device_input(payload, jax, selected)
    from ._ffi import load_backend
    _, decode = load_backend(selected)
    call = jax.ffi.ffi_call(
        decode, (jax.ShapeDtypeStruct((expected_bytes,), np.uint8),
                 jax.ShapeDtypeStruct((2,), np.uint32)),
        vmap_method="sequential")
    with jax.default_device(selected):
        return call(raw, max_candidates=np.int64(_MAX_CANDIDATES),
                    max_blocks=np.int64(_MAX_BLOCKS))


def compress_zlib(data, device=0, *, chunk_bytes=32768):
    """Encode bytes/JAX uint8 to a completed, exact-length JAX CUDA array.

    This synchronous API checks status on the host. For jax.jit use
    compress_zlib_padded and handle its length/status metadata explicitly.
    """
    output, metadata = compress_zlib_padded(data, device, chunk_bytes=chunk_bytes)
    length, status = np.asarray(metadata)
    _check_value(int(status), "CUDA compression")
    result = output[:int(length)]
    result.block_until_ready()
    return result


def decompress_zlib(payload, expected_bytes, device=0):
    """Decode bytes/JAX uint8 to a completed, owned JAX CUDA array.

    Every invocation rediscovers blocks and checks the original Adler32. No CPU
    codec fallback occurs. For jax.jit use decompress_zlib_checked and inspect
    its status metadata explicitly.
    """
    output, metadata = decompress_zlib_checked(payload, expected_bytes, device)
    _check_value(int(np.asarray(metadata)[0]), "CUDA decompression")
    output.block_until_ready()
    return output


def decompress_zlib_host(payload, expected_bytes, device=0) -> np.ndarray:
    """Decode to a completed, read-only, contiguous NumPy uint8 array.

    Uses the same CUDA decoding, limits and validation as decompress_zlib, then
    transfers the result to JAX-owned pinned host memory. The returned array
    retains that storage independently of subsequent calls. A memoryview shares
    it without copying; tobytes() allocates and copies into a new bytes object.
    This synchronous host API cannot be used inside jax.jit.
    """
    output = decompress_zlib(payload, expected_bytes, device)
    jax = _jax()
    sharding = jax.sharding.SingleDeviceSharding(
        next(iter(output.devices())), memory_kind="pinned_host")
    host = jax.device_put(output, sharding)
    result = np.asarray(host.block_until_ready())
    # JAX's empty-array conversion may create an owned, writable NumPy array.
    result.flags.writeable = False
    return result
