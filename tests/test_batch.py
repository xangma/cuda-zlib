# Copyright (c) 2026 xangma
# SPDX-License-Identifier: MIT
"""Independent file oracles, packed jit workflow and per-file error isolation."""
import gc
import random
import zlib

import numpy as np
import pytest

import cuda_zlib as codec
from test_decode import cuda_device


@pytest.mark.parametrize("sizes", [(), (-1,), (True,), (1.0,), (2**28 + 1,),
                                   (2**28, 1), 1, "1"])
def test_batch_invalid_static_sizes(sizes):
    with pytest.raises(ValueError):
        codec.compress_zlib_batch_padded(b"", sizes)
    with pytest.raises(ValueError):
        codec.decompress_zlib_batch_checked(b"", sizes, (0,))


def test_batch_invalid_extents_and_chunks():
    with pytest.raises(ValueError, match="packed input"):
        codec.compress_zlib_batch_padded(b"x", (0,))
    with pytest.raises(ValueError, match="matching lengths"):
        codec.decompress_zlib_batch_checked(b"", (0,), (0, 0))
    with pytest.raises(ValueError, match="packed input"):
        codec.decompress_zlib_batch_checked(b"x", (0,), (0,))
    with pytest.raises(ValueError, match="chunk_bytes"):
        codec.compress_zlib_batch_padded(b"", (0,), chunk_bytes=255)


def _inputs():
    return (b"", b"x", bytes(257), b"abc" * 4096,
            random.Random(42).randbytes(32769), b"a" * 65536,
            random.Random(8).randbytes(100000))


@pytest.mark.parametrize("chunk_bytes", [256, 32768, 65535])
def test_batch_encoder_matches_scalar_and_stdlib(cuda_device, monkeypatch, chunk_bytes):
    inputs = _inputs()
    sizes = tuple(map(len, inputs))
    def forbidden(*args, **kwargs):
        pytest.fail("batch attempted CPU encoding")
    with monkeypatch.context() as patch:
        patch.setattr(zlib, "compress", forbidden)
        patch.setattr(zlib, "compressobj", forbidden)
        streams = codec.compress_zlib_batch_host(inputs, cuda_device, chunk_bytes=chunk_bytes)
    for raw, stream in zip(inputs, streams):
        assert zlib.decompress(stream) == raw
        assert stream == np.asarray(codec.compress_zlib(
            raw, cuda_device, chunk_bytes=chunk_bytes)).tobytes()
    decoded = codec.decompress_zlib_batch_host(streams, sizes, cuda_device)
    assert tuple(value.tobytes() for value in decoded) == inputs
    assert all(value.dtype == np.uint8 and value.flags.c_contiguous and
               not value.flags.writeable for value in decoded)
    retained = decoded[4]
    del decoded
    gc.collect()
    codec.decompress_zlib_batch_host((zlib.compress(b"changed"),), (7,), cuda_device)
    assert retained.tobytes() == inputs[4]


def test_batch_jit_dependencies_padding_and_device(cuda_device, monkeypatch):
    import jax
    import jax.numpy as jnp
    device = jax.devices("gpu")[cuda_device]
    inputs = _inputs()[:5]
    sizes = tuple(map(len, inputs))
    source = jax.device_put(np.frombuffer(b"".join(inputs), np.uint8), device)
    encode = jax.jit(lambda value: codec.compress_zlib_batch_padded(
        (value.astype(jnp.uint16) * 17 + 3).astype(jnp.uint8), sizes, device))
    padded, metadata = encode(source)
    buffer, lengths = np.asarray(padded), np.asarray(metadata)
    assert padded.devices() == metadata.devices() == {device}
    assert lengths.shape == (len(sizes), 2) and not np.any(lengths[:, 1])
    transformed = ((np.asarray(source).astype(np.uint16) * 17 + 3) % 256).astype(np.uint8)
    streams = []
    offset = raw_offset = 0
    for size, (length, _) in zip(sizes, lengths):
        capacity = size + 5 * max(1, (size + 32767) // 32768) + 6
        stream = buffer[offset:offset + int(length)].tobytes()
        streams.append(stream)
        assert zlib.decompress(stream) == transformed[raw_offset:raw_offset + size].tobytes()
        assert not np.any(buffer[offset + int(length):offset + capacity])
        offset += capacity
        raw_offset += size
    compressed_sizes = tuple(map(len, streams))
    packed = jax.device_put(np.frombuffer(b"".join(streams), np.uint8), device)
    decode = jax.jit(lambda value: codec.decompress_zlib_batch_checked(
        value, compressed_sizes, sizes, device))
    def forbidden(*args, **kwargs):
        pytest.fail("batch attempted CPU decoding")
    with monkeypatch.context() as patch:
        patch.setattr(zlib, "decompress", forbidden)
        patch.setattr(zlib, "decompressobj", forbidden)
        output, status = decode(packed)
        np.testing.assert_array_equal(np.asarray(output), transformed)
        assert not np.any(np.asarray(status))
    exact_streams = codec.compress_zlib_batch(inputs, device)
    exact_outputs = codec.decompress_zlib_batch(exact_streams, sizes, device)
    assert tuple(np.asarray(value).tobytes() for value in exact_outputs) == inputs


def test_batch_decoder_error_isolation_and_recovery(cuda_device):
    valid = zlib.compress(b"abc" * 100)
    checksum = valid[:-1] + bytes([valid[-1] ^ 1])
    streams = (valid, checksum, b"", b"xx" + valid[2:], valid, valid)
    sizes = tuple(map(len, streams))
    expected = (300, 300, 0, 300, 1, 300)
    output, metadata = codec.decompress_zlib_batch_checked(
        b"".join(streams), sizes, expected, cuda_device)
    assert list(np.asarray(metadata)[:, 0]) == [0, 21, 14, 15, 7, 0]
    assert not np.any(np.asarray(metadata)[:, 1])
    assert np.asarray(output)[:300].tobytes() == b"abc" * 100
    assert np.asarray(output)[-300:].tobytes() == b"abc" * 100
    with pytest.raises(codec.CodecError, match="file 1"):
        codec.decompress_zlib_batch_host(streams[:2], (300, 300), cuda_device)
    assert codec.decompress_zlib_batch_host((valid,), (300,), cuda_device)[0].tobytes() == b"abc" * 100


@pytest.mark.parametrize("strategy", [zlib.Z_DEFAULT_STRATEGY, zlib.Z_FIXED,
                                      zlib.Z_HUFFMAN_ONLY, zlib.Z_RLE])
def test_batch_external_streams_and_cross_block_history(cuda_device, strategy):
    raw = random.Random(25).randbytes(1000) * 96
    compressor = zlib.compressobj(6, zlib.DEFLATED, 15, 8, strategy)
    stream = compressor.compress(raw[:48000]) + compressor.flush(zlib.Z_SYNC_FLUSH)
    stream += compressor.compress(raw[48000:]) + compressor.flush()
    stored = zlib.compress(raw, 0)
    inputs = (stream, zlib.compress(b""), stored)
    output = codec.decompress_zlib_batch_host(inputs, (len(raw), 0, len(raw)), cuda_device)
    assert tuple(value.tobytes() for value in output) == (raw, b"", raw)


def test_batch_padded_roundtrip_inside_jit_and_metadata_bounds(cuda_device):
    import jax
    import jax.numpy as jnp
    device = jax.devices("gpu")[cuda_device]
    inputs = _inputs()[:5]
    sizes = tuple(map(len, inputs))
    capacities = tuple(size + 5 * max(1, (size + 32767) // 32768) + 6 for size in sizes)
    source = jax.device_put(np.frombuffer(b"".join(inputs), np.uint8), device)
    @jax.jit
    def roundtrip(value):
        padded, metadata = codec.compress_zlib_batch_padded(value, sizes, device)
        return codec.decompress_zlib_batch_checked(
            padded, capacities, sizes, device, encoded_metadata=metadata)
    output, status = roundtrip(source)
    assert np.asarray(output).tobytes() == b"".join(inputs)
    assert not np.any(np.asarray(status))
    padded, metadata = codec.compress_zlib_batch_padded(source, sizes, device)
    bad = metadata.at[0, 1].set(jnp.uint32(22)).at[1, 0].set(jnp.uint32(capacities[1] + 1))
    output, status = codec.decompress_zlib_batch_checked(
        padded, capacities, sizes, device, encoded_metadata=bad)
    assert list(np.asarray(status)[:, 0]) == [22, 23, 0, 0, 0]
    assert np.asarray(output)[1:].tobytes() == b"".join(inputs[2:])


def test_batch_concurrent_calls_and_ownership(cuda_device):
    import jax
    from concurrent.futures import ThreadPoolExecutor
    inputs = _inputs()[:4]
    def run(seed):
        files = tuple(bytes([seed]) + raw for raw in inputs)
        streams = codec.compress_zlib_batch_host(files, cuda_device)
        decoded = codec.decompress_zlib_batch_host(streams, tuple(map(len, files)), cuda_device)
        assert tuple(view.tobytes() for view in decoded) == files
        return streams
    with ThreadPoolExecutor(max_workers=3) as pool:
        assert len(list(pool.map(run, range(6)))) == 6
    devices = jax.devices("gpu")
    device = devices[cuda_device]
    source = jax.device_put(np.frombuffer(b"x", np.uint8), device)
    padded, metadata = codec.compress_zlib_batch_padded(source, (1,), device)
    assert padded.devices() == metadata.devices() == {device}
    if len(devices) > 1:
        other = next(candidate for candidate in devices if candidate != device)
        with pytest.raises(ValueError, match="device differ"):
            codec.compress_zlib_batch_padded(source, (1,), other)


def test_batch_native_abi_rejects_invalid_layouts(cuda_device):
    """Bypass Python validation to check native descriptor/output bounds."""
    import jax
    from cuda_zlib._ffi import load_batch_backend
    device = jax.devices("gpu")[cuda_device]
    encode, decode = load_batch_backend(device)
    raw = jax.device_put(np.zeros(1, np.uint8), device)
    empty = jax.device_put(np.empty(0, np.uint32), device)
    for sizes, capacity, shape in (((1,), 12, (2,)),
                                   ((-1,), 12, (1, 2)),
                                   ((2,), 13, (1, 2)),
                                   ((1,), 11, (1, 2))):
        call = jax.ffi.ffi_call(encode, (
            jax.ShapeDtypeStruct((capacity,), np.uint8),
            jax.ShapeDtypeStruct(shape, np.uint32)))
        with pytest.raises(Exception, match="metadata|invalid argument"):
            jax.block_until_ready(call(raw, chunk_bytes=np.int64(32768),
                                      input_sizes=np.asarray(sizes, np.int64)))
    call = jax.ffi.ffi_call(decode, (
        jax.ShapeDtypeStruct((1,), np.uint8),
        jax.ShapeDtypeStruct((1, 2), np.uint32)))
    wrong_metadata = jax.device_put(np.zeros((1, 3), np.uint32), device)
    with pytest.raises(Exception, match="encoded metadata"):
        jax.block_until_ready(call(raw, wrong_metadata,
            input_sizes=np.array([1], np.int64), output_sizes=np.array([1], np.int64)))
    with pytest.raises(Exception, match="matching nonempty"):
        jax.block_until_ready(call(raw, empty,
            input_sizes=np.array([1], np.int64), output_sizes=np.empty(0, np.int64)))
