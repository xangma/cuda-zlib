# Copyright (c) 2026 xangma
# SPDX-License-Identifier: MIT

"""Validated host arrays retain their storage without a bytes conversion."""

import builtins
import gc
import zlib

import numpy as np
import pytest

from cuda_zlib import CodecError, decompress_zlib_host


@pytest.fixture(scope="module")
def cuda_device():
    jax = pytest.importorskip("jax")
    try:
        devices = jax.local_devices(backend="gpu")
    except RuntimeError as exc:
        pytest.skip(str(exc))
    if not devices or "cuda" not in devices[0].client.platform_version.lower():
        pytest.skip("JAX CUDA backend unavailable")
    return devices[0]


@pytest.fixture
def forbid_cpu_inflation(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("CUDA host decoding attempted CPU zlib inflation")

    monkeypatch.setattr(zlib, "decompress", forbidden)
    monkeypatch.setattr(zlib, "decompressobj", forbidden)


def _raw(size):
    block = bytes(range(251))
    return (block * ((size + len(block) - 1) // len(block)))[:size]


def _input(payload, resident, device):
    if resident:
        import jax
        return jax.device_put(np.frombuffer(payload, dtype=np.uint8), device)
    return payload


@pytest.mark.parametrize("payload,expected,error", [
    (b"", 0, CodecError),
    (b"\x78\x00" + b"\x00" * 8, 0, CodecError),
    (b"", -1, ValueError),
    (b"", True, ValueError),
])
def test_host_rejects_invalid_input_before_loading_jax(
    monkeypatch, payload, expected, error,
):
    actual_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name.split(".")[0] == "jax":
            pytest.fail("invalid host input loaded the optional GPU backend")
        return actual_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    with pytest.raises(error):
        decompress_zlib_host(payload, expected)


@pytest.mark.parametrize("resident", [False, True], ids=["bytes", "resident"])
@pytest.mark.parametrize("size", [0, 65537, 1048576])
def test_host_array_is_completed_readonly_and_byte_exact(
    cuda_device, forbid_cpu_inflation, resident, size,
):
    raw = _raw(size)
    payload = _input(zlib.compress(raw), resident, cuda_device)
    result = decompress_zlib_host(payload, size, cuda_device)
    assert isinstance(result, np.ndarray)
    assert result.dtype == np.uint8
    assert result.shape == (size,)
    assert result.flags.c_contiguous
    assert not result.flags.writeable
    assert memoryview(result).readonly
    assert memoryview(result) == memoryview(raw)


@pytest.mark.parametrize("resident", [False, True], ids=["bytes", "resident"])
@pytest.mark.parametrize("change", ["block-type", "truncated", "checksum",
                                   "short-output", "long-output"])
def test_host_decode_rejects_corruption_and_wrong_extent_then_recovers(
    cuda_device, forbid_cpu_inflation, resident, change,
):
    raw = _raw(65537)
    valid = zlib.compress(raw)
    payload, expected = valid, len(raw)
    if change == "block-type":
        payload = valid[:2] + bytes(((valid[2] & ~6) | 6,)) + valid[3:]
    elif change == "truncated":
        payload = valid[:-6] + valid[-4:]
    elif change == "checksum":
        payload = valid[:-1] + bytes((valid[-1] ^ 1,))
    elif change == "short-output":
        expected -= 1
    else:
        expected += 1
    match = "Adler32 mismatch" if change == "checksum" else None
    with pytest.raises(CodecError, match=match):
        decompress_zlib_host(_input(payload, resident, cuda_device), expected,
                             cuda_device)
    recovered = decompress_zlib_host(_input(valid, resident, cuda_device),
                                     len(raw), cuda_device)
    assert memoryview(recovered) == memoryview(raw)


def test_host_memoryview_survives_gc_and_subsequent_decode(
    cuda_device, forbid_cpu_inflation,
):
    raw = _raw(1048576)
    result = decompress_zlib_host(zlib.compress(raw), len(raw), cuda_device)
    view = memoryview(result)
    del result
    gc.collect()
    second_raw = b"different output and extent" * 4096
    second = decompress_zlib_host(zlib.compress(second_raw), len(second_raw),
                                 cuda_device)
    assert memoryview(second) == memoryview(second_raw)
    del second
    gc.collect()
    assert view.readonly
    assert view == memoryview(raw)
