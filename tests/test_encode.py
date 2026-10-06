# Copyright (c) 2026 xangma
# SPDX-License-Identifier: MIT

"""Independent stdlib oracle, device round trips, ownership and bounds."""
import builtins
import gc
import hashlib
import random
import zlib
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

import cuda_zlib as codec
from test_decode import cuda_device, _alignment_stream, _assert_bytes


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
    encoded = np.asarray(result).tobytes()
    assert zlib.decompress(encoded) == raw
    assert encoded == np.asarray(codec.compress_zlib(raw, cuda_device)).tobytes()
    assert len(encoded) <= size + 5 * max(1, (size + 32767) // 32768) + 6
    assert np.asarray(codec.decompress_zlib(result, size, cuda_device)).tobytes() == raw
    if kind == "repeat" and size >= 256:
        assert len(encoded) < size // 4


@pytest.mark.parametrize("chunk", [256, 1024, 16384, 32768, 65535])
def test_chunk_alignment_and_stored_fallback(cuda_device, chunk):
    raw = b"a" * chunk + random.Random(91).randbytes(chunk) + b"bc" * chunk + b"end"
    result = codec.compress_zlib(raw, cuda_device, chunk_bytes=chunk)
    assert zlib.decompress(np.asarray(result).tobytes()) == raw
    assert np.asarray(codec.decompress_zlib(result, len(raw), cuda_device)).tobytes() == raw


@pytest.mark.parametrize("kind,encoded_size,encoded_sha256", [
    ("empty", 8, "09d469dfeeaf4c436fd3f80ba7e168bcc34e5050993b5da3c8416fbae680f18f"),
    ("one", 9, "8697003ea5c8f20f906cf3e1c5a02071cd8030421091c335812e8c3555cd673c"),
    ("repeat", 53, "0e90a17da0061571bacea268276bda67e1722967a274d4296164095ae2118402"),
    ("ramp", 454, "489850606f508ad255fe309bebdb2b87a7890de28bc345d169c8922129d8f9be"),
    ("mixed", 32826, "8c948d5c7640be206ec1dd70a6fcecff616616d1fa7e5638d3ede9fc22d688e0"),
])
def test_encoder_matches_original_streams(cuda_device, kind, encoded_size, encoded_sha256):
    # Frozen device-produced streams from the original deterministic matcher.
    if kind == "mixed":
        raw = np.random.default_rng(1729).integers(
            0, 256, 32768, dtype=np.uint8).tobytes() + b"a" * 32768
    else:
        raw = {"empty": b"", "one": b"a", "repeat": b"a" * 32768,
               "ramp": bytes(range(256)) * 128}[kind]
    encoded = np.asarray(codec.compress_zlib(raw, cuda_device)).tobytes()
    assert len(encoded) == encoded_size
    assert hashlib.sha256(encoded).hexdigest() == encoded_sha256
    assert zlib.decompress(encoded) == raw


@pytest.mark.parametrize("size", [0, 1, 256, 32768, 65535])
def test_maximum_literal_token_capacity_roundtrip(cuda_device, size):
    # A de Bruijn sequence of order two has no repeated byte pairs, hence no
    # three-byte matches. Native scratch must hold one token per input byte.
    sequence = b"".join(bytes((i,)) + b"".join(bytes((i, j))
                       for j in range(i + 1, 256)) for i in range(256))
    raw = sequence[:size]
    result = codec.compress_zlib(raw, cuda_device, chunk_bytes=max(256, size))
    assert zlib.decompress(np.asarray(result).tobytes()) == raw
    _assert_bytes(codec.decompress_zlib(result, size, cuda_device), raw, cuda_device)


def test_device_input_dependencies_and_output_lifetime(cuda_device):
    import jax
    import jax.numpy as jnp
    device = jax.devices("gpu")[cuda_device]
    host = np.arange(251 * 1024, dtype=np.uint32).astype(np.uint8)
    source = jax.device_put(host, device)
    # Dispatch dependent GPU work without an explicit host wait. The FFI call
    # must consume the produced bytes on JAX's supplied execution stream.
    transform = jax.jit(lambda value: (value.astype(jnp.uint16) * 17 + 3).astype(jnp.uint8))
    source = transform(source)
    raw = ((host.astype(np.uint16) * 17 + 3) % 256).astype(np.uint8).tobytes()
    first = codec.compress_zlib(source, cuda_device)
    assert isinstance(first, jax.Array)
    del source
    gc.collect()
    # Reuse similarly sized temporary allocations while retaining the result.
    for _ in range(3):
        codec.compress_zlib(b"z" * len(raw), cuda_device)
    assert zlib.decompress(np.asarray(first).tobytes()) == raw


def test_device_input_contract(cuda_device):
    import jax
    device = jax.devices("gpu")[cuda_device]
    for host in (np.zeros(12, dtype=np.int32), np.zeros((2, 2), dtype=np.uint8)):
        bad = jax.device_put(host, device)
        with pytest.raises(TypeError):
            codec.compress_zlib(bad, cuda_device)
        with pytest.raises(TypeError):
            codec.decompress_zlib(bad, 0, cuda_device)
    # JAX slicing materializes a valid logical array rather than a strided view.
    good = jax.device_put(np.arange(20, dtype=np.uint8), device)[::2]
    stream = codec.compress_zlib(good, device)
    assert zlib.decompress(np.asarray(stream).tobytes()) == bytes(range(0, 20, 2))


def test_wrong_device_is_rejected(cuda_device):
    import jax
    devices = jax.devices("gpu")
    if len(devices) < 2:
        pytest.skip("two CUDA devices required for device mismatch regression")
    other = devices[1 if cuda_device == 0 else 0]
    raw = jax.device_put(np.arange(32, dtype=np.uint8), other)
    with pytest.raises((TypeError, ValueError), match="device"):
        codec.compress_zlib(raw, cuda_device)
    stream = jax.device_put(np.frombuffer(zlib.compress(b"test"), dtype=np.uint8), other)
    with pytest.raises((TypeError, ValueError), match="device"):
        codec.decompress_zlib(stream, 4, cuda_device)


def test_concurrent_calls_have_independent_results(cuda_device):
    raws = [bytes((i,)) * 8192 + bytes(range(256)) for i in range(8)]
    def roundtrip(raw):
        encoded = codec.compress_zlib(raw, cuda_device)
        decoded = codec.decompress_zlib(encoded, len(raw), cuda_device)
        return encoded, decoded
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(roundtrip, raws))
    for raw, (encoded, decoded) in zip(raws, results):
        assert zlib.decompress(np.asarray(encoded).tobytes()) == raw
        assert np.asarray(decoded).tobytes() == raw


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
    assert np.asarray(codec.decompress_zlib(result, 9, cuda_device)).tobytes() == b"recovered"


def test_device_header_and_adler_errors(cuda_device):
    import jax.numpy as jnp
    raw = b"guarded" * 512
    encoded = codec.compress_zlib(raw, cuda_device)
    bad_adler = encoded.at[-1].set(encoded[-1] ^ jnp.uint8(1))
    with pytest.raises(codec.CodecError, match="Adler32"):
        codec.decompress_zlib(bad_adler, len(raw), cuda_device)
    bad_header = encoded.at[0].set(jnp.uint8(0))
    with pytest.raises((codec.CodecError, ValueError), match="header"):
        codec.decompress_zlib(bad_header, len(raw), cuda_device)
    # Functional updates must not corrupt the owned original compressed array.
    _assert_bytes(codec.decompress_zlib(encoded, len(raw), cuda_device), raw, cuda_device)


@pytest.mark.parametrize("chunk", [True, 0, 255, 65536, 1.5])
def test_bad_options_before_backend(monkeypatch, chunk):
    original = builtins.__import__
    def guard(name, *args, **kwargs):
        if name.split(".")[0] in {"jax"}:
            pytest.fail("invalid options imported CUDA")
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", guard)
    with pytest.raises(ValueError):
        codec.compress_zlib(b"", chunk_bytes=chunk)


def test_optional_import_and_device_admission(monkeypatch):
    with pytest.raises(codec.BackendUnavailable):
        codec.compress_zlib(b"", device=None)
    original = builtins.__import__
    def guard(name, *args, **kwargs):
        if name.split(".")[0] == "jax":
            raise ImportError("test absence")
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", guard)
    with pytest.raises(codec.BackendUnavailable, match="JAX"):
        codec.compress_zlib(b"test")


def test_jax_bridge_owned_output(cuda_device):
    import jax
    from cuda_zlib import jax as bridge
    device = jax.devices("gpu")[cuda_device]
    raw = b"jax independent ownership" * 1024
    source = jax.device_put(np.frombuffer(raw, dtype=np.uint8), device)
    encoded = bridge.compress_zlib(source, device=device)
    decoded = bridge.decompress_zlib(zlib.compress(raw), len(raw), device=device)
    assert isinstance(encoded, jax.Array) and isinstance(decoded, jax.Array)
    del source
    gc.collect()
    for _ in range(3):
        bridge.compress_zlib(b"z" * len(raw), device=device)
        bridge.decompress_zlib(zlib.compress(b"y" * len(raw)), len(raw), device=device)
    _assert_bytes(decoded, raw, device)
    assert zlib.decompress(np.asarray(encoded).tobytes()) == raw


def test_extent_bounds_before_backend_and_recovery(cuda_device, monkeypatch):
    from cuda_zlib import _codec
    raw = b"extent admission" * 16
    payload = zlib.compress(raw)
    with monkeypatch.context() as patch:
        patch.setattr(_codec, "_MAX_BYTES", 64)
        with pytest.raises(ValueError, match="limit|extent|size|256|expected_bytes"):
            codec.compress_zlib(raw, cuda_device)
        with pytest.raises(ValueError, match="limit|extent|size|256|expected_bytes"):
            codec.decompress_zlib(payload, len(raw), cuda_device)
    encoded = codec.compress_zlib(raw, cuda_device)
    _assert_bytes(codec.decompress_zlib(encoded, len(raw), cuda_device), raw, cuda_device)
