# Copyright (c) 2026 xangma
# SPDX-License-Identifier: MIT

"""Stored-stream routing, checked fallback, and output/metadata ownership."""

import gc
import zlib

import numpy as np
import pytest

from test_decode import (
    _assert_bytes, _bits, _codes, _compressed, _dynamic_literal_fields,
    _header, _limited_checked_decoder, _stored, _wrap, cuda_device,
)
from test_regressions import _no_cpu_codec


_STORED_LENGTHS = (
    (0, 1, 65535, 0, 1, 0),
    (65535,) * 16 + (17, 0),
)


def _stored_stream(lengths, raw=None, padding=False):
    size = sum(lengths)
    if raw is None:
        raw = (bytes(range(251)) * ((size + 250) // 251))[:size]
    assert len(raw) == size
    blocks, offset = [], 0
    for index, length in enumerate(lengths):
        block = _stored(raw[offset:offset + length], index == len(lengths) - 1)
        if padding:
            # Stored-block alignment discards the upper five header bits.
            block = bytes((block[0] | 0xF8,)) + block[1:]
        blocks.append(block)
        offset += length
    return _wrap(b"".join(blocks), raw), raw


def _mixed_stream(kind):
    prefix = bytes(range(251)) * 261 + bytes(range(24))
    assert len(prefix) == 65535
    if kind == "fixed":
        # 23 nine-bit literals + header/EOB occupy 28 bytes: five overhead
        # bytes, just like a stored block, despite a different block type.
        tail = bytes(range(144, 167))
        codes = _codes([8] * 144 + [9] * 112 + [7] * 24 + [8] * 8)
        fields = [(1, 1), (1, 2)] + [codes[value] for value in tail]
        fields.append(codes[256])
    else:
        assert kind == "dynamic"
        tail = b"A" * 30
        fields = _dynamic_literal_fields(tail)
    raw = prefix + tail
    payload = _wrap(_stored(prefix, final=False) + _bits(fields), raw)
    # Deliberately ambiguous total extent: device parsing must distinguish
    # these valid mixed streams from a chain containing only stored blocks.
    assert len(payload) >= len(raw) + 11
    assert (len(payload) - len(raw) - 6) % 5 == 0
    return payload, raw


def _error_streams():
    payload, raw = _stored_stream((32768, 32769))
    bad_length = bytearray(payload)
    bad_length[5] ^= 1  # First stored LEN/NLEN pair no longer complements.
    bad_checksum = payload[:-1] + bytes((payload[-1] ^ 1,))
    combined = bytes(bad_length[:-1]) + bytes((bad_length[-1] ^ 1,))
    missing_final = bytearray(payload)
    missing_final[2 + 5 + 32768] &= ~1
    cases = (
        ("length", bytes(bad_length), len(raw), 4),
        ("length-and-adler", combined, len(raw), 4),
        ("truncated-body", payload[:-5] + payload[-4:], len(raw), 1),
        ("extra-body", payload[:-4] + b"\0" * 5 + payload[-4:], len(raw), 12),
        ("header-and-body", b"\0" + combined[1:], len(raw), 15),
        ("dictionary-and-body", _header(flags=0x20) + combined[2:], len(raw), 17),
        ("adler", bad_checksum, len(raw), 21),
        ("short-output", payload, len(raw) - 5, 7),
        ("long-output", payload, len(raw) + 5, 11),
        ("missing-final", bytes(missing_final), len(raw), 1),
    )
    return payload, raw, cases


def _limited_streams(kind):
    if kind == "candidates":
        marker = _bits(_dynamic_literal_fields(b"", single_eob=True))
        raw = (marker * ((65537 + len(marker) - 1) // len(marker)))[:65537]
        payload, _ = _stored_stream((65535, 2), raw)
        recovery, recovery_raw = _stored_stream((65535, 2), b"\0" * 65537)
        return payload, recovery, recovery_raw, 1, 256, 18
    assert kind == "blocks"
    payload, raw = _stored_stream((65535, 2))
    # A single compressed block fits the same reduced block budget.
    recovery_raw = b"\0" * len(raw)
    return payload, _compressed(recovery_raw), recovery_raw, 4096, 1, 9


def _strict_oracle(payload):
    decoder = zlib.decompressobj()
    raw = decoder.decompress(payload) + decoder.flush()
    assert decoder.eof and not decoder.unused_data and not decoder.unconsumed_tail
    return raw


def test_stored_route_fixtures_match_independent_stdlib():
    for lengths in _STORED_LENGTHS:
        payload, raw = _stored_stream(lengths, padding=True)
        assert _strict_oracle(payload) == raw
    for kind in ("fixed", "dynamic"):
        payload, raw = _mixed_stream(kind)
        assert _strict_oracle(payload) == raw
    empty, raw = _stored_stream((0,) * 26215)
    assert len(empty) > 131072
    assert _strict_oracle(empty) == raw == b""
    payload, raw, cases = _error_streams()
    assert _strict_oracle(payload) == raw
    for name, malformed, _, _ in cases:
        if name in ("short-output", "long-output"):
            assert _strict_oracle(malformed) == raw
        else:
            with pytest.raises((zlib.error, AssertionError)):
                _strict_oracle(malformed)
    for kind in ("candidates", "blocks"):
        payload, recovery, raw, _, _, _ = _limited_streams(kind)
        assert len(_strict_oracle(payload)) == len(raw)
        assert _strict_oracle(recovery) == raw


@pytest.mark.parametrize("lengths", _STORED_LENGTHS, ids=("empty-and-varied", "over-1MiB"))
def test_stored_checked_jit_varied_blocks(cuda_device, monkeypatch, lengths):
    import jax
    from cuda_zlib import decompress_zlib_checked

    payload, raw = _stored_stream(lengths, padding=True)
    assert _strict_oracle(payload) == raw
    device = jax.devices("gpu")[cuda_device]
    source = jax.device_put(np.frombuffer(payload, np.uint8), device)
    decode = jax.jit(lambda value: decompress_zlib_checked(value, len(raw), device))
    with _no_cpu_codec(monkeypatch):
        output, metadata = decode(source)
        jax.block_until_ready((output, metadata))
    assert metadata.devices() == {device}
    np.testing.assert_array_equal(np.asarray(metadata), [0, 0])
    _assert_bytes(output, raw, device)


@pytest.mark.parametrize("kind", ("fixed", "dynamic"))
def test_stored_mixed_stream_checked_fallback(cuda_device, monkeypatch, kind):
    import jax
    from cuda_zlib import decompress_zlib_checked

    payload, raw = _mixed_stream(kind)
    assert _strict_oracle(payload) == raw
    device = jax.devices("gpu")[cuda_device]
    source = jax.device_put(np.frombuffer(payload, np.uint8), device)
    decode = jax.jit(lambda value: decompress_zlib_checked(value, len(raw), device))
    with _no_cpu_codec(monkeypatch):
        output, metadata = decode(source)
        jax.block_until_ready((output, metadata))
    np.testing.assert_array_equal(np.asarray(metadata), [0, 0])
    _assert_bytes(output, raw, device)


def test_stored_checked_errors_recovery_and_retained_metadata(cuda_device, monkeypatch):
    import jax
    from cuda_zlib import decompress_zlib_checked, trim_workspace_pool

    payload, raw, cases = _error_streams()
    assert _strict_oracle(payload) == raw
    compressed = _compressed(raw)
    assert _strict_oracle(compressed) == raw
    compressed_bad = compressed[:-1] + bytes((compressed[-1] ^ 1,))
    replacement_payload, replacement_raw = _stored_stream((32768, 32769), raw[::-1])
    device = jax.devices("gpu")[cuda_device]
    sources = [jax.device_put(np.frombuffer(value, np.uint8), device)
               for value in (payload,) + tuple(case[1] for case in cases)
               + (compressed, compressed_bad)]
    decode = jax.jit(lambda value, expected: decompress_zlib_checked(value, expected, device),
                     static_argnums=1)
    with _no_cpu_codec(monkeypatch):
        first, first_metadata = decode(sources[0], len(raw))
        retained = []
        for source, (name, _, expected, status) in zip(sources[1:1 + len(cases)], cases):
            failed, metadata = decode(source, expected)
            recovered, recovered_metadata = decode(sources[0], len(raw))
            retained.append((name, failed, metadata, recovered, recovered_metadata, status))
        control, control_metadata = decode(sources[-2], len(raw))
        failed, metadata = decode(sources[-1], len(raw))
        recovered, recovered_metadata = decode(sources[-2], len(raw))
        retained.append(("compressed-adler", failed, metadata, recovered, recovered_metadata, 21))
        jax.block_until_ready((first, first_metadata, control, control_metadata, retained))
        del sources
        gc.collect()
        trim_workspace_pool(device)
        replacement = jax.device_put(np.frombuffer(replacement_payload, np.uint8), device)
        later, later_metadata = decode(replacement, len(raw))
        jax.block_until_ready((later, later_metadata))
    # No output is consumed until every retained validation result is checked.
    np.testing.assert_array_equal(np.asarray(first_metadata), [0, 0])
    np.testing.assert_array_equal(np.asarray(control_metadata), [0, 0])
    for name, _, metadata, _, recovered_metadata, status in retained:
        assert metadata.devices() == recovered_metadata.devices() == {device}
        np.testing.assert_array_equal(np.asarray(metadata), [status, 0], err_msg=name)
        np.testing.assert_array_equal(np.asarray(recovered_metadata), [0, 0], err_msg=name)
    np.testing.assert_array_equal(np.asarray(later_metadata), [0, 0])
    _assert_bytes(first, raw, device)
    _assert_bytes(control, raw, device)
    for _, _, _, recovered, _, _ in retained:
        _assert_bytes(recovered, raw, device)
    _assert_bytes(later, replacement_raw, device)


def test_empty_output_long_stored_chain_checked_recovery(cuda_device, monkeypatch):
    import jax
    from cuda_zlib import decompress_zlib_checked

    payload, raw = _stored_stream((0,) * 26215)
    assert len(payload) > 131072
    assert _strict_oracle(payload) == raw == b""
    malformed = payload[:-1] + bytes((payload[-1] ^ 1,))
    device = jax.devices("gpu")[cuda_device]
    source, invalid = [jax.device_put(np.frombuffer(value, np.uint8), device)
                       for value in (payload, malformed)]
    decode = jax.jit(lambda value: decompress_zlib_checked(value, 0, device))
    with _no_cpu_codec(monkeypatch):
        first, first_metadata = decode(source)
        failed, metadata = decode(invalid)
        recovered, recovered_metadata = decode(source)
        jax.block_until_ready((first, first_metadata, failed, metadata,
                              recovered, recovered_metadata))
    for result, status in ((first_metadata, 0), (metadata, 21), (recovered_metadata, 0)):
        assert result.devices() == {device}
        np.testing.assert_array_equal(np.asarray(result), [status, 0])
    _assert_bytes(first, raw, device)
    _assert_bytes(recovered, raw, device)


@pytest.mark.parametrize("kind", ("candidates", "blocks"))
def test_stored_checked_respects_reduced_limits(cuda_device, monkeypatch, kind):
    import jax

    payload, recovery, raw, candidates, blocks, status = _limited_streams(kind)
    assert len(_strict_oracle(payload)) == len(raw)
    assert _strict_oracle(recovery) == raw
    device = jax.devices("gpu")[cuda_device]
    source, valid = [jax.device_put(np.frombuffer(value, np.uint8), device)
                     for value in (payload, recovery)]
    decode = _limited_checked_decoder(device, len(raw), candidates, blocks)
    with _no_cpu_codec(monkeypatch):
        failed, metadata = decode(source)
        recovered, recovered_metadata = decode(valid)
        jax.block_until_ready((failed, metadata, recovered, recovered_metadata))
    np.testing.assert_array_equal(np.asarray(metadata), [status, 0])
    np.testing.assert_array_equal(np.asarray(recovered_metadata), [0, 0])
    _assert_bytes(recovered, raw, device)
