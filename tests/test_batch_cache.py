# Copyright (c) 2026 xangma
# SPDX-License-Identifier: MIT
"""Bounded host compilation reuse and validation across warm calls."""
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
import sys
import zlib

import numpy as np
import pytest

import cuda_zlib as codec
from cuda_zlib import _codec
from test_decode import cuda_device


@pytest.fixture(autouse=True)
def clear_plans():
    _codec._host_batch_encoder.cache_clear()
    _codec._host_batch_decoder.cache_clear()
    yield
    _codec._host_batch_encoder.cache_clear()
    _codec._host_batch_decoder.cache_clear()


@pytest.fixture
def fake_backend(monkeypatch):
    """Exercise host admission/cache behavior without importing a CUDA runtime."""
    compiled, seen = [], []

    class Array:
        __module__ = "jax"
        dtype, ndim, size = np.uint8, 1, 1

        def devices(self):
            return {1}

    def jit(fn, *, in_shardings):
        compiled.append(in_shardings)

        def run(data):
            seen.append(data.copy())
            return fn(data)
        return run

    def encode(data, sizes, device, *, chunk_bytes):
        capacities = _codec._batch_capacities(sizes, chunk_bytes)
        output = np.zeros(sum(capacities), np.uint8)
        offset = source = 0
        for size, capacity in zip(sizes, capacities):
            output[offset:offset + size] = data[source:source + size]
            offset += capacity
            source += size
        return output, np.array([(size, 0) for size in sizes], np.uint32)

    def decode(data, sizes, expected, device):
        return np.resize(data, sum(expected)), np.zeros((len(sizes), 2), np.uint32)

    fake = SimpleNamespace(
        jit=jit, Array=Array, numpy=np, core=SimpleNamespace(Tracer=type("Tracer", (), {})),
        sharding=SimpleNamespace(SingleDeviceSharding=lambda device, **kwargs: (device, kwargs)),
        device_put=lambda value, sharding: SimpleNamespace(block_until_ready=lambda: value))
    monkeypatch.setitem(sys.modules, "jax", fake)
    monkeypatch.setitem(sys.modules, "jax.numpy", np)
    monkeypatch.setattr(_codec, "_jax", lambda: fake)
    monkeypatch.setattr(_codec, "_select_device", lambda device: device)
    monkeypatch.setattr(_codec, "compress_zlib_batch_padded", encode)
    monkeypatch.setattr(_codec, "decompress_zlib_batch_checked", decode)
    return SimpleNamespace(compiled=compiled, seen=seen, Array=Array)


def test_host_plans_reuse_layout_without_retaining_input(fake_backend):
    assert codec.compress_zlib_batch_host((b"ab", b"c")) == (b"ab", b"c")
    assert codec.compress_zlib_batch_host((b"de", b"f")) == (b"de", b"f")
    assert len(fake_backend.compiled) == 1
    assert [value.tobytes() for value in fake_backend.seen] == [b"abc", b"def"]
    assert fake_backend.compiled[0][0] == 0
    first = codec.decompress_zlib_batch_host((b"ab", b"c"), (2, 1))
    second = codec.decompress_zlib_batch_host((b"de", b"f"), [2, 1])
    assert tuple(value.tobytes() for value in first) == (b"ab", b"c")
    assert tuple(value.tobytes() for value in second) == (b"de", b"f")
    assert all(not value.flags.writeable for value in first + second)
    assert len(fake_backend.compiled) == 2


@pytest.mark.parametrize("chunk", [True, 256.0, 255, 65536, "256"])
def test_warm_encoder_still_validates_chunk(fake_backend, chunk):
    codec.compress_zlib_batch_host((b"x",), chunk_bytes=256)
    with pytest.raises(ValueError, match="chunk_bytes"):
        codec.compress_zlib_batch_host((b"x",), chunk_bytes=chunk)
    assert len(fake_backend.compiled) == 1


@pytest.mark.parametrize("expected", [(True,), (1.0,), (-1,), (2**28 + 1,), (1, 0), ()])
def test_warm_decoder_still_validates_sizes(fake_backend, expected):
    codec.decompress_zlib_batch_host((b"x",), (1,))
    with pytest.raises(ValueError):
        codec.decompress_zlib_batch_host((b"x",), expected)
    assert len(fake_backend.compiled) == 1


@pytest.mark.parametrize("operation", ["encode", "decode"])
def test_warm_plans_still_validate_inputs_and_device(fake_backend, operation):
    def run(inputs):
        if operation == "encode":
            return codec.compress_zlib_batch_host(inputs)
        return codec.decompress_zlib_batch_host(inputs, (1,))
    run((b"x",))
    with pytest.raises(ValueError, match="nonempty"):
        run(())
    with pytest.raises(TypeError, match="must be bytes"):
        run((bytearray(b"x"),))
    with pytest.raises(ValueError, match="device differ"):
        run((fake_backend.Array(),))
    assert len(fake_backend.compiled) == 1


def test_warm_plans_still_validate_totals_and_workspace(fake_backend, monkeypatch):
    codec.compress_zlib_batch_host((b"x",), chunk_bytes=256)
    codec.decompress_zlib_batch_host((b"x",), (1,))
    monkeypatch.setattr(_codec, "_MAX_BYTES", 16)
    with pytest.raises(ValueError, match="input_sizes total"):
        codec.compress_zlib_batch_host((bytes(9), bytes(8)), chunk_bytes=256)
    with pytest.raises(ValueError, match="compressed capacity"):
        codec.compress_zlib_batch_host((bytes(6),), chunk_bytes=256)
    with pytest.raises(ValueError, match="expected_sizes total"):
        codec.decompress_zlib_batch_host((b"x", b"x"), (9, 8))
    monkeypatch.setattr(_codec, "_MAX_BYTES", 2**28)
    monkeypatch.setattr(_codec, "_MAX_BLOCKS", 2)
    with pytest.raises(ValueError, match="chunk count"):
        codec.compress_zlib_batch_host((bytes(257),), chunk_bytes=256)
    assert len(fake_backend.compiled) == 2


@pytest.mark.parametrize("operation", ["encode", "decode"])
def test_plan_keys_and_bounded_eviction(fake_backend, operation):
    if operation == "encode":
        factory = _codec._host_batch_encoder
        original = factory(0, (1,), 256)
        assert factory(0, (1,), 256) is original
        assert factory(1, (1,), 256) is not original
        assert factory(0, (1,), 32768) is not original
        for size in range(2, 35):
            factory(0, (size,), 256)
        assert factory(0, (1,), 256) is not original
    else:
        factory = _codec._host_batch_decoder
        original = factory(0, (1,), (1,))
        assert factory(0, (1,), (1,)) is original
        assert factory(1, (1,), (1,)) is not original
        assert factory(0, (1,), (2,)) is not original
        for size in range(2, 35):
            factory(0, (size,), (1,))
        assert factory(0, (1,), (1,)) is not original
    assert factory.cache_info().currsize == factory.cache_info().maxsize == 32


def test_cached_host_cuda_content_layout_chunks_and_errors(cuda_device, monkeypatch):
    inputs = (b"a" * 256, b"b" * 17)
    changed = (b"c" * 256, b"d" * 17)
    for chunk in (256, 32768):
        for files in (inputs, changed, (b"e" * 257, b"f" * 17)):
            with monkeypatch.context() as patch:
                def forbidden(*args, **kwargs):
                    pytest.fail("cached batch used CPU compression")
                patch.setattr(zlib, "compress", forbidden)
                patch.setattr(zlib, "compressobj", forbidden)
                streams = codec.compress_zlib_batch_host(files, cuda_device, chunk_bytes=chunk)
            assert tuple(zlib.decompress(stream) for stream in streams) == files
            assert streams == tuple(np.asarray(codec.compress_zlib(
                raw, cuda_device, chunk_bytes=chunk)).tobytes() for raw in files)
    assert _codec._host_batch_encoder.cache_info().misses == 4
    assert _codec._host_batch_encoder.cache_info().hits == 2

    # Stored streams retain their layout when content and checksum change.
    streams = tuple(zlib.compress(raw, 0) for raw in inputs)
    changed_streams = tuple(zlib.compress(raw, 0) for raw in changed)
    retained = codec.decompress_zlib_batch_host(streams, (256, 17), cuda_device)
    with monkeypatch.context() as patch:
        def forbidden(*args, **kwargs):
            pytest.fail("cached batch used CPU decompression")
        patch.setattr(zlib, "decompress", forbidden)
        patch.setattr(zlib, "decompressobj", forbidden)
        outputs = codec.decompress_zlib_batch_host(changed_streams, (256, 17), cuda_device)
    assert tuple(view.tobytes() for view in outputs) == changed
    assert tuple(view.tobytes() for view in retained) == inputs
    assert all(not view.flags.writeable for view in retained + outputs)
    bad = changed_streams[1][:-1] + bytes([changed_streams[1][-1] ^ 1])
    with pytest.raises(codec.CodecError, match="CUDA batch decompression file 1: CUDA Adler32 mismatch"):
        codec.decompress_zlib_batch_host((changed_streams[0], bad), (256, 17), cuda_device)
    assert tuple(view.tobytes() for view in codec.decompress_zlib_batch_host(
        changed_streams, (256, 17), cuda_device)) == changed
    assert _codec._host_batch_decoder.cache_info().misses == 1
    assert _codec._host_batch_decoder.cache_info().hits == 3


def test_cached_host_cuda_jax_inputs_and_concurrent_calls(cuda_device):
    import jax
    selected = jax.devices("gpu")[cuda_device]

    def run(seed):
        files = (bytes([seed]) * 256, bytes([seed + 1]) * 17)
        streams = codec.compress_zlib_batch_host(files, selected)
        assert tuple(zlib.decompress(stream) for stream in streams) == files
        external = tuple(zlib.compress(raw, 0) for raw in files)
        outputs = codec.decompress_zlib_batch_host(external, (256, 17), selected)
        assert tuple(view.tobytes() for view in outputs) == files
        return outputs

    retained = run(0)
    with ThreadPoolExecutor(max_workers=3) as pool:
        assert len(list(pool.map(run, range(1, 7)))) == 6
    assert tuple(view.tobytes() for view in retained) == (bytes(256), b"\x01" * 17)
    raw = jax.device_put(np.zeros(256, np.uint8), selected)
    mixed = codec.compress_zlib_batch_host((raw, b"\x01" * 17), selected)
    assert tuple(zlib.decompress(stream) for stream in mixed) == (bytes(256), b"\x01" * 17)
    for other in jax.devices("gpu"):
        if other != selected:
            with pytest.raises(ValueError, match="device differ"):
                codec.compress_zlib_batch_host((raw, b"\x01" * 17), other)
