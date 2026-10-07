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


def _batch_sizes(sizes, label):
    if not isinstance(sizes, (tuple, list)) or not 1 <= len(sizes) <= _MAX_BLOCKS:
        raise ValueError(label + " must contain between 1 and 262144 static sizes")
    sizes = tuple(sizes)
    if any(type(size) is not int or not 0 <= size <= _MAX_BYTES for size in sizes):
        raise ValueError(label + " must contain integers in [0, 2**28]")
    if sum(sizes) > _MAX_BYTES:
        raise ValueError(label + " total exceeds 2**28 bytes")
    return sizes


def _batch_capacities(sizes, chunk_bytes):
    if type(chunk_bytes) is not int or not 256 <= chunk_bytes <= 65535:
        raise ValueError("chunk_bytes must be an integer in [256, 65535]")
    chunks = tuple(max(1, (size + chunk_bytes - 1) // chunk_bytes) for size in sizes)
    if sum(chunks) > _MAX_BLOCKS or any(2 * n - 1 > _MAX_BLOCKS for n in chunks):
        raise ValueError("batch chunk count exceeds workspace limit")
    capacities = tuple(size + n * 5 + 6 for size, n in zip(sizes, chunks))
    if sum(capacities) > _MAX_BYTES:
        raise ValueError("batch compressed capacity exceeds 2**28 bytes")
    return capacities


def compress_zlib_batch_padded(data, input_sizes, device=0, *, chunk_bytes=32768):
    """Compress packed independent files in one FFI call, including under jit.

    ``input_sizes`` is a static tuple of lengths whose sum equals len(data).
    Return a flat uint8 buffer of consecutive padded slots and uint32 metadata
    of shape (files, 2), with [encoded_length, status] for each file. A slot has
    ``size + 5 * max(1, ceil(size / chunk_bytes)) + 6`` bytes. Check every status
    before using a stream. Batch input and output totals are bounded by 2**28.
    """
    sizes = _batch_sizes(input_sizes, "input_sizes")
    if _input_size(data) != sum(sizes):
        raise ValueError("packed input length differs from input_sizes total")
    capacities = _batch_capacities(sizes, chunk_bytes)
    selected = _select_device(device)
    jax = _jax()
    raw = _device_input(data, jax, selected)
    from ._ffi import load_batch_backend
    encode, _ = load_batch_backend(selected)
    call = jax.ffi.ffi_call(
        encode, (jax.ShapeDtypeStruct((sum(capacities),), np.uint8),
                 jax.ShapeDtypeStruct((len(sizes), 2), np.uint32)))
    with jax.default_device(selected):
        return call(raw, chunk_bytes=np.int64(chunk_bytes),
                    input_sizes=np.asarray(sizes, dtype=np.int64))


def decompress_zlib_batch_checked(payload, input_sizes, expected_sizes, device=0,
                                 *, encoded_metadata=None):
    """Decode packed independent streams in one FFI call, including under jit.

    The two size tuples are static and have matching nonempty lengths. Input
    streams and returned outputs are packed consecutively without padding.
    To decode padded compression slots directly, set input_sizes to their
    static capacities and pass compression's (files, 2) JAX uint32 metadata as
    encoded_metadata. Actual stream lengths remain on device, including under jit.
    Return a flat output and (files, 2) uint32 [status, 0] metadata. Each file's
    framing, history, extent and Adler32 are checked independently on CUDA.
    Check every status before using its output. Totals are bounded by 2**28.
    """
    sizes = _batch_sizes(input_sizes, "input_sizes")
    expected = _batch_sizes(expected_sizes, "expected_sizes")
    if len(sizes) != len(expected):
        raise ValueError("input_sizes and expected_sizes must have matching lengths")
    if _input_size(payload, "compressed payload") != sum(sizes):
        raise ValueError("packed input length differs from input_sizes total")
    selected = _select_device(device)
    jax = _jax()
    raw = _device_input(payload, jax, selected)
    if encoded_metadata is None:
        lengths = jax.device_put(np.empty(0, np.uint32), selected)
    else:
        if (not type(encoded_metadata).__module__.startswith(("jax", "jaxlib")) or
                encoded_metadata.dtype != np.uint32 or
                encoded_metadata.shape != (len(sizes), 2)):
            raise TypeError("encoded_metadata must be a JAX uint32 array with shape (files, 2)")
        lengths = _device_input(encoded_metadata, jax, selected)
    from ._ffi import load_batch_backend
    _, decode = load_batch_backend(selected)
    call = jax.ffi.ffi_call(
        decode, (jax.ShapeDtypeStruct((sum(expected),), np.uint8),
                 jax.ShapeDtypeStruct((len(sizes), 2), np.uint32)))
    with jax.default_device(selected):
        return call(raw, lengths, input_sizes=np.asarray(sizes, dtype=np.int64),
                    output_sizes=np.asarray(expected, dtype=np.int64))


def _pack_batch(inputs, device):
    if not isinstance(inputs, (tuple, list)) or not inputs:
        raise ValueError("inputs must be a nonempty sequence of files")
    sizes = _batch_sizes(tuple(_input_size(data) for data in inputs), "input_sizes")
    if all(isinstance(data, bytes) for data in inputs):
        return b"".join(inputs), sizes
    import jax.numpy as jnp
    jax = _jax()
    return jnp.concatenate(tuple(_device_input(data, jax, device) for data in inputs)), sizes


def _check_batch(metadata, operation, compression=False):
    metadata = np.asarray(metadata)
    for i, row in enumerate(metadata):
        _check_value(int(row[1 if compression else 0]), "%s file %d" % (operation, i))
    return metadata


def _split_batch(buffer, sizes, extents=None):
    offset = 0
    results = []
    for i, size in enumerate(sizes):
        length = size if extents is None else int(extents[i])
        results.append(buffer[offset:offset + length])
        offset += size
    return tuple(results)


def compress_zlib_batch(inputs, device=0, *, chunk_bytes=32768):
    """Return completed exact-length JAX streams for a sequence of files.

    Checks all statuses once on the host. For maximum throughput or jit use
    compress_zlib_batch_padded: exact per-file JAX slicing adds dispatch work.
    """
    selected = _select_device(device)
    packed, sizes = _pack_batch(inputs, selected)
    output, metadata = compress_zlib_batch_padded(packed, sizes, selected, chunk_bytes=chunk_bytes)
    lengths = _check_batch(metadata, "CUDA batch compression", True)[:, 0]
    results = _split_batch(output, _batch_capacities(sizes, chunk_bytes), lengths)
    return _jax().block_until_ready(results)


def decompress_zlib_batch(payloads, expected_sizes, device=0):
    """Return completed JAX outputs for independent streams; check statuses once.

    Use decompress_zlib_batch_checked for packed outputs or inside jax.jit.
    """
    selected = _select_device(device)
    packed, sizes = _pack_batch(payloads, selected)
    expected = _batch_sizes(expected_sizes, "expected_sizes")
    output, metadata = decompress_zlib_batch_checked(packed, sizes, expected, selected)
    _check_batch(metadata, "CUDA batch decompression")
    return _jax().block_until_ready(_split_batch(output, expected))


def compress_zlib_batch_host(inputs, device=0, *, chunk_bytes=32768):
    """Return a tuple of bytes streams, using one packed output transfer.

    This synchronous convenience API is useful for many small host files.
    """
    selected = _select_device(device)
    packed, sizes = _pack_batch(inputs, selected)
    output, metadata = compress_zlib_batch_padded(packed, sizes, selected, chunk_bytes=chunk_bytes)
    lengths = _check_batch(metadata, "CUDA batch compression", True)[:, 0]
    views = _split_batch(np.asarray(output), _batch_capacities(sizes, chunk_bytes), lengths)
    return tuple(view.tobytes() for view in views)


def decompress_zlib_batch_host(payloads, expected_sizes, device=0):
    """Return read-only NumPy views backed by one packed pinned-host transfer.

    Every stream is decoded and checked independently on CUDA. Returned views
    keep their storage alive across subsequent calls. This API is synchronous.
    """
    selected = _select_device(device)
    packed, sizes = _pack_batch(payloads, selected)
    expected = _batch_sizes(expected_sizes, "expected_sizes")
    output, metadata = decompress_zlib_batch_checked(packed, sizes, expected, selected)
    _check_batch(metadata, "CUDA batch decompression")
    jax = _jax()
    sharding = jax.sharding.SingleDeviceSharding(selected, memory_kind="pinned_host")
    host = np.asarray(jax.device_put(output, sharding).block_until_ready())
    host.flags.writeable = False
    return _split_batch(host, expected)


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

    Every invocation parses the original stream and checks its Adler32. No CPU
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
