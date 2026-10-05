# Copyright (c) 2026 xangma
# SPDX-License-Identifier: MIT

"""Independent stdlib oracle, device round trips, ownership and bounds."""
import builtins
import gc
import random
import zlib
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

import cuda_zlib as codec
from test_decode import cuda_device, _alignment_stream


@pytest.mark.parametrize("size", [0, 1, 2, 3, 255, 256, 257, 32767, 32768,
                                  32769, 65535, 65536, 131073])
@pytest.mark.parametrize("kind", ["random", "repeat", "alphabet"])
def test_encoder_stdlib_and_cuda_roundtrip(cuda_device, monkeypatch, size, kind):
    if kind == "random":
        raw = random.Random(size + 19).randbytes(size)
    elif kind == "repeat":
        raw = b"a" * size
    else:
        raw = (bytes(range(256)) * ((size + 255) // 256))[:size]
    def forbidden(*args, **kwargs):
        pytest.fail("CUDA encoder attempted CPU compression")
    with monkeypatch.context() as patch:
        patch.setattr(zlib, "compress", forbidden)
        patch.setattr(zlib, "compressobj", forbidden)
        result = codec.compress_zlib(raw, cuda_device)
    encoded = result.get().tobytes()
    assert zlib.decompress(encoded) == raw
    assert encoded == codec.compress_zlib(raw, cuda_device).get().tobytes()
    assert len(encoded) <= size + 5 * max(1, (size + 32767) // 32768) + 6
    assert codec.decompress_zlib(result, size, cuda_device).get().tobytes() == raw
    if kind == "repeat" and size >= 256:
        assert len(encoded) < size // 4


@pytest.mark.parametrize("chunk", [256, 1024, 16384, 32768, 65535])
def test_chunk_alignment_and_stored_fallback(cuda_device, chunk):
    raw = b"a" * chunk + random.Random(91).randbytes(chunk) + b"bc" * chunk + b"end"
    result = codec.compress_zlib(raw, cuda_device, chunk_bytes=chunk)
    assert zlib.decompress(result.get().tobytes()) == raw
    assert codec.decompress_zlib(result, len(raw), cuda_device).get().tobytes() == raw


def test_device_input_lifetime_and_caller_stream(cuda_device):
    import cupy as cp
    raw = bytes(range(251)) * 1024
    with cp.cuda.Stream(non_blocking=True):
        source = cp.asarray(np.frombuffer(raw, dtype=np.uint8))
        first = codec.compress_zlib(source, cuda_device)
    source.fill(0)
    cp.cuda.get_current_stream().synchronize()
    del source
    gc.collect()
    for _ in range(3):
        codec.compress_zlib(b"z" * len(raw), cuda_device)
    assert zlib.decompress(first.get().tobytes()) == raw


def test_device_input_contract(cuda_device):
    import cupy as cp
    for bad in (cp.zeros(12, dtype=cp.int32), cp.zeros((2, 2), dtype=cp.uint8),
                cp.zeros(20, dtype=cp.uint8)[::2]):
        with pytest.raises(TypeError):
            codec.compress_zlib(bad, cuda_device)
        with pytest.raises(TypeError):
            codec.decompress_zlib(bad, 0, cuda_device)


def test_concurrent_calls_have_independent_results(cuda_device):
    raws = [bytes((i,)) * 8192 + bytes(range(256)) for i in range(8)]
    def roundtrip(raw):
        encoded = codec.compress_zlib(raw, cuda_device)
        decoded = codec.decompress_zlib(encoded, len(raw), cuda_device)
        return encoded, decoded
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(roundtrip, raws))
    for raw, (encoded, decoded) in zip(raws, results):
        assert zlib.decompress(encoded.get().tobytes()) == raw
        assert decoded.get().tobytes() == raw


def test_capacity_errors_and_recovery(cuda_device, monkeypatch):
    from cuda_zlib import _codec
    payload, raw = _alignment_stream(1)
    with monkeypatch.context() as patch:
        patch.setattr(_codec, "_MAX_CANDIDATES", 1)
        with pytest.raises(codec.CodecError, match="candidate workspace overflow"):
            codec.decompress_zlib(payload, len(raw), cuda_device)
    with monkeypatch.context() as patch:
        patch.setattr(_codec, "_MAX_BLOCKS", 1)
        with pytest.raises(codec.CodecError, match="workspace|limit"):
            codec.decompress_zlib(zlib.compress(b"x" * 65536, 0), 65536, cuda_device)
        with pytest.raises(ValueError, match="block limit"):
            codec.compress_zlib(b"x" * 32769, cuda_device)
    result = codec.compress_zlib(b"recovered", cuda_device)
    assert codec.decompress_zlib(result, 9, cuda_device).get().tobytes() == b"recovered"


def test_device_header_and_adler_errors(cuda_device):
    import cupy as cp
    raw = b"guarded" * 512
    encoded = codec.compress_zlib(raw, cuda_device)
    encoded[-1] ^= cp.uint8(1)
    with pytest.raises(codec.CodecError, match="Adler32"):
        codec.decompress_zlib(encoded, len(raw), cuda_device)
    encoded[0] = 0
    with pytest.raises(codec.CodecError, match="header"):
        codec.decompress_zlib(encoded, len(raw), cuda_device)


@pytest.mark.parametrize("chunk", [True, 0, 255, 65536, 1.5])
def test_bad_options_before_backend(monkeypatch, chunk):
    original = builtins.__import__
    def guard(name, *args, **kwargs):
        if name.split(".")[0] in {"cupy", "jax"}:
            pytest.fail("invalid options imported CUDA")
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", guard)
    with pytest.raises(ValueError):
        codec.compress_zlib(b"", chunk_bytes=chunk)


def test_optional_import_and_device_admission(monkeypatch):
    from cuda_zlib import _codec
    with pytest.raises(codec.BackendUnavailable):
        codec.compress_zlib(b"", device=None)
    original = builtins.__import__
    def guard(name, *args, **kwargs):
        if name == "cupy":
            raise ImportError("test absence")
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", guard)
    with pytest.raises(codec.BackendUnavailable, match="CuPy"):
        _codec._cupy()


def test_jax_bridge_owned_output(cuda_device):
    import cupy as cp
    jax = pytest.importorskip("jax")
    from cuda_zlib import jax as bridge
    device = jax.devices("gpu")[0]
    raw = b"jax independent ownership" * 1024
    source = cp.asarray(np.frombuffer(raw, dtype=np.uint8))
    owned = bridge.to_jax_owned(source, device)
    encoded = bridge.compress_zlib(owned, device=device)
    decoded = bridge.decompress_zlib(zlib.compress(raw), len(raw), device=device)
    source.fill(0)
    cp.cuda.get_current_stream().synchronize()
    assert np.asarray(owned).tobytes() == raw
    assert np.asarray(decoded).tobytes() == raw
    assert zlib.decompress(np.asarray(encoded).tobytes()) == raw
