# Copyright (c) 2026 xangma
# SPDX-License-Identifier: MIT
"""GPU regressions for bounded literal lookahead, errors and queued ownership."""
import contextlib
import gc
import zlib

import numpy as np
import pytest

from literal_lookahead_fixtures import (
    dynamic_fallback, dynamic_full_window, fixed, fixed_repeat, literal_payload,
)


@pytest.fixture(scope="module")
def cuda_device():
    jax = pytest.importorskip("jax")
    try:
        devices = jax.devices("gpu")
    except RuntimeError:
        pytest.skip("CUDA JAX backend is unavailable")
    if not devices:
        pytest.skip("CUDA JAX backend is unavailable")
    return devices[0]


@contextlib.contextmanager
def no_cpu_codec(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("CPU codec called during checked decoding")
    with monkeypatch.context() as guard:
        for name in ("compress", "decompress", "compressobj", "decompressobj"):
            guard.setattr(zlib, name, forbidden)
        yield


def assert_result(result, expected, device, status=0):
    output, metadata = result
    assert output.devices() == metadata.devices() == {device}
    assert output.dtype == np.uint8
    assert output.shape == (len(expected),)
    assert np.asarray(output).tobytes() == expected
    np.testing.assert_array_equal(np.asarray(metadata), [status, 0])


def checked(payload, size, device, monkeypatch):
    import jax
    import cuda_zlib as codec
    source = jax.device_put(np.frombuffer(payload, np.uint8), device)
    decode = jax.jit(lambda value: codec.decompress_zlib_checked(value, size, device))
    with no_cpu_codec(monkeypatch):
        result = decode(source)
        jax.block_until_ready(result)
    return result


@pytest.mark.parametrize("size", [0, 1, 4, 5, 7, 8, 31, 32, 33, 65, 65535, 65536])
@pytest.mark.parametrize("high", [False, True], ids=["literal8", "literal9"])
def test_wide_fixed_literal_cached_tail(cuda_device, monkeypatch, size, high):
    # Genuine one-block literal-only streams cover high-half starts and EOF.
    payload, raw = fixed(literal_payload(size, high))
    assert zlib.decompress(payload) == raw
    assert (payload[2] >> 1) & 3 == 1
    result = checked(payload, len(raw), cuda_device, monkeypatch)
    assert_result(result, raw, cuda_device)


@pytest.mark.parametrize("high", [False, True], ids=["literal8", "literal9"])
def test_wide_fixed_unaligned_eob_transition(cuda_device, monkeypatch, high):
    # xyz/EOB puts the second header at bit34, preserving its true alignment.
    payload, raw = fixed(literal_payload(65, high), prefix=b"xyz")
    assert zlib.decompress(payload) == raw
    assert_result(checked(payload, len(raw), cuda_device, monkeypatch), raw, cuda_device)


@pytest.mark.parametrize("size", [33, 65536])
def test_wide_primary_zero_canonical_fallback(cuda_device, monkeypatch, size):
    # A length10 literal follows five length8 literals; its9-bit primary is zero.
    pattern = literal_payload(5) + b"\xff"
    raw = (pattern * ((size + 5) // 6))[:size]
    payload, expected = dynamic_fallback(raw)
    assert zlib.decompress(payload) == expected
    assert (payload[2] >> 1) & 3 == 2
    assert_result(checked(payload, size, cuda_device, monkeypatch), raw, cuda_device)


@pytest.mark.parametrize("size", [256, 65536])
def test_wide_full_cached_window_commit(cuda_device, monkeypatch, size):
    # The 256-byte specimen reaches pos2600/cached64/consumed64 after a
    # canonical ten-bit literal; committing the cache must avoid a shift by64.
    payload, raw = dynamic_full_window(size)
    assert zlib.decompress(payload) == raw
    assert_result(checked(payload, size, cuda_device, monkeypatch), raw, cuda_device)


@pytest.mark.parametrize("size", [80, 65533])
def test_wide_literal_stop_before_overlapping_match(cuda_device, monkeypatch, size):
    payload, raw = fixed_repeat(literal_payload(size, high=True))
    assert zlib.decompress(payload) == raw
    assert_result(checked(payload, len(raw), cuda_device, monkeypatch), raw, cuda_device)


@pytest.mark.parametrize("ending", ["reserved-ll", "reserved-distance"])
def test_wide_reserved_token_preserves_partial_output(cuda_device, monkeypatch, ending):
    payload, prefix = fixed(literal_payload(80), ending=ending)
    with pytest.raises(zlib.error):
        zlib.decompress(payload)
    result = checked(payload, len(prefix) + 3, cuda_device, monkeypatch)
    assert_result(result, prefix + bytes(3), cuda_device, status=4)


@pytest.mark.parametrize("kind", ["truncated", "checksum", "header", "bounds"])
def test_wide_failure_output_and_metadata(cuda_device, monkeypatch, kind):
    raw = literal_payload(80)
    payload, expected = fixed(raw)
    if kind == "truncated":
        payload, _ = fixed(raw, ending="truncated")
        decoder = zlib.decompressobj()
        assert decoder.decompress(payload[:-4]) == raw and not decoder.eof
        expected, status = raw + bytes(3), 1
    elif kind == "checksum":
        payload = payload[:-1] + bytes([payload[-1] ^ 1])
        expected, status = raw, 21
        with pytest.raises(zlib.error):
            zlib.decompress(payload)
    elif kind == "header":
        payload = bytes(1) + payload[1:]
        expected, status = bytes(len(raw)), 15
        with pytest.raises(zlib.error):
            zlib.decompress(payload)
    else:
        assert zlib.decompress(payload) == raw
        # Available capacity stops inside a long literal run without overrunning.
        expected, status = raw[:37], 7
    assert_result(checked(payload, len(expected), cuda_device, monkeypatch),
                  expected, cuda_device, status)


def test_wide_retained_queued_output_status_and_recovery(cuda_device, monkeypatch):
    import jax
    import cuda_zlib as codec
    raw = literal_payload(83, high=True)
    raw_reverse = raw[::-1]
    good = [fixed(value)[0] for value in (raw, raw_reverse)]
    bad, prefix = fixed(literal_payload(80), ending="reserved-ll")
    checksum = good[1][:-1] + bytes([good[1][-1] ^ 1])
    header = bytes(1) + good[0][1:]
    neighbor = literal_payload(65537, high=True)
    neighbor_payload = fixed(neighbor)[0]
    for payload, expected in zip(good, (raw, raw_reverse)):
        assert zlib.decompress(payload) == expected
    assert zlib.decompress(neighbor_payload) == neighbor
    inputs = [jax.device_put(np.frombuffer(value, np.uint8), cuda_device)
              for value in [*good, bad, checksum, header, neighbor_payload]]
    decode = jax.jit(lambda value: codec.decompress_zlib_checked(value, 83, cuda_device))
    other = jax.jit(lambda value: codec.decompress_zlib_checked(value, 65537, cuda_device))
    wanted = [(raw, 0), (prefix + bytes(3), 4), (raw_reverse, 0),
              (raw_reverse, 21), (bytes(83), 15), (neighbor, 0), (raw, 0)]
    with no_cpu_codec(monkeypatch):
        retained = [decode(inputs[0]), decode(inputs[2]), decode(inputs[1]),
                    decode(inputs[3]), decode(inputs[4]), other(inputs[5]), decode(inputs[0])]
        jax.block_until_ready(retained)
        del inputs
        gc.collect()
        codec.trim_workspace_pool(cuda_device)
        replacement = jax.device_put(np.frombuffer(good[1], np.uint8), cuda_device)
        later = decode(replacement)
        jax.block_until_ready(later)
        assert_result(later, raw_reverse, cuda_device)
        for result, (expected, status) in zip(retained, wanted):
            assert_result(result, expected, cuda_device, status)
