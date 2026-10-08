# Copyright (c) 2026 xangma
# SPDX-License-Identifier: MIT
"""Queued decoding across parser/writer boundaries and serial fallbacks.

Nondefault workspace caps exercise queued decoding for small dense streams.
Only completed output bytes and status are inspected.
"""
import hashlib
import zlib

import numpy as np
import pytest

from test_regressions import (
    _Writer, _match, _match_block_header, _literals, _inflate_exact, _no_cpu_codec,
)
from test_decode import (
    _wrap, _stored, _assert_bytes, _limited_checked_decoder, cuda_device,
    _codes as _cost_canonical, _dynamic_header as _cost_dynamic_header,
)


def _repeat_block(writer, output, kind, count, tail=b"", final=True):
    literal_codes, distance_codes = _match_block_header(writer, kind, final=final)
    # The first match uses the preceding stored/dense block. Later matches
    # reuse overlapping history written by earlier tokens and batches.
    for _ in range(count):
        _match(writer, output, 258, 1, literal_codes, distance_codes)
    _literals(writer, tail, literal_codes)
    output.extend(tail)
    writer.put(*literal_codes[256])


def _repeat_stream(kind, count, tail=b""):
    seed = b"Q"
    writer = _Writer()
    writer.aligned(_stored(seed, final=False))
    output = bytearray(seed)
    _repeat_block(writer, output, kind, count, tail)
    raw = bytes(output)
    return _wrap(writer.finish(), raw), raw


def _size_boundary_stream(kind, block_bytes):
    # 127*258 == 32766: one, two or three independent literal tails reach
    # either side of the minimum accepted-block size without changing history.
    tail = b"\x00\x7f\xff"[:block_bytes - 127 * 258]
    assert 1 <= len(tail) <= 3
    return _repeat_stream(kind, 127, tail)


def _mixed_stream():
    seed = b"Q"
    writer = _Writer()
    writer.aligned(_stored(seed, final=False))
    output = bytearray(seed)
    # Several serial-sized dense blocks keep the complete stream dense despite
    # the following large literal-only block. No dense block exceeds 64 KiB.
    for _ in range(4):
        _repeat_block(writer, output, "dynamic", 160, final=False)
    literal_codes, _ = _match_block_header(writer, "fixed", final=False)
    small = bytes((19 * i + 7) & 255 for i in range(99))
    _literals(writer, small, literal_codes)
    output.extend(small)
    writer.put(*literal_codes[256])
    literal_codes, _ = _match_block_header(writer, "dynamic", final=True)
    sparse = bytes((41 * i + 13) & 255 for i in range(32769))
    _literals(writer, sparse, literal_codes)
    output.extend(sparse)
    writer.put(*literal_codes[256])
    raw = bytes(output)
    return _wrap(writer.finish(), raw), raw


def _prepare_call(device, payload, expected):
    import jax
    selected = jax.devices("gpu")[device]
    call = _limited_checked_decoder(selected, expected, candidates=4096, blocks=128)
    value = jax.device_put(np.frombuffer(payload, dtype=np.uint8), selected)
    return call, value


def _completed(call, value):
    import jax
    result, metadata = jax.block_until_ready(call(value))
    return result, np.asarray(metadata)


@pytest.mark.parametrize("kind", ["fixed", "dynamic"])
@pytest.mark.parametrize("count", [159, 160, 161])
def test_pipeline_ordered_history_and_terminal_boundaries(kind, count, cuda_device, monkeypatch):
    # Match counts end after a partial batch, an exact fill, and its next token.
    # There is no local seed literal inside the compressed block.
    payload, raw = _repeat_stream(kind, count)
    assert _inflate_exact(payload) == raw
    call, value = _prepare_call(cuda_device, payload, len(raw))
    with _no_cpu_codec(monkeypatch):
        result, metadata = _completed(call, value)
        assert metadata.tolist() == [0, 0]
        _assert_bytes(result, raw, cuda_device)


@pytest.mark.parametrize("kind", ["fixed", "dynamic"])
@pytest.mark.parametrize("block_bytes", [32767, 32768, 32769])
def test_pipeline_minimum_size_and_literal_tails(kind, block_bytes, cuda_device, monkeypatch):
    payload, raw = _size_boundary_stream(kind, block_bytes)
    assert _inflate_exact(payload) == raw
    call, value = _prepare_call(cuda_device, payload, len(raw))
    with _no_cpu_codec(monkeypatch):
        result, metadata = _completed(call, value)
        assert metadata.tolist() == [0, 0]
        _assert_bytes(result, raw, cuda_device)


@pytest.mark.parametrize("kind", ["fixed", "dynamic"])
def test_pipeline_checksum_failure_then_same_shape_recovery(kind, cuda_device, monkeypatch):
    import jax
    payload, raw = _repeat_stream(kind, 160)
    assert _inflate_exact(payload) == raw
    corrupted = payload[:-1] + bytes((payload[-1] ^ 1,))
    with pytest.raises(zlib.error):
        _inflate_exact(corrupted)
    call, valid = _prepare_call(cuda_device, payload, len(raw))
    invalid = jax.device_put(np.frombuffer(corrupted, dtype=np.uint8), jax.devices("gpu")[cuda_device])
    with _no_cpu_codec(monkeypatch):
        _, metadata = _completed(call, invalid)
        assert metadata.tolist() == [21, 0]
        # Reuse the compiled call and workspace shape after the checksum error.
        result, metadata = _completed(call, valid)
        assert metadata.tolist() == [0, 0]
        _assert_bytes(result, raw, cuda_device)


def test_pipeline_dense_stream_with_small_and_sparse_blocks(cuda_device, monkeypatch):
    payload, raw = _mixed_stream()
    assert _inflate_exact(payload) == raw
    call, value = _prepare_call(cuda_device, payload, len(raw))
    with _no_cpu_codec(monkeypatch):
        result, metadata = _completed(call, value)
        assert metadata.tolist() == [0, 0]
        _assert_bytes(result, raw, cuda_device)


# LL widths, literal prefix, match length/count, literal tail, encoded bytes,
# and frozen independent stream identity. All matches have distance one.
_COST_STREAM_CASES = {
    "literal_zero_dynamic": (
        {0: 1, 256: 1}, 1048576, 0, 0, 0, 131120,
        "6c97c47776f5313294c90c88968c8afad419eff916543effe1a8398151d50d26"),
    "short_repeat_dynamic": (
        {0: 2, 256: 2, 257: 1}, 1, 3, 349525, 0, 87430,
        "c0053fdea9897bfb78ed449bf4220ad8f657ccb81c76ae54072e34383b31ce0f"),
    "match_prefix_literal_tail": (
        {0: 1, 256: 2, 285: 2}, 1, 258, 256, 982527, 122964,
        "9475a27ecf37deb15fdda97a64f93eff10648577cf8e29cf8cf7defa8fd70d7c"),
    "literal_prefix_match_tail": (
        {0: 1, 256: 2, 285: 2}, 262192, 258, 3048, 0, 33969,
        "87c067b6b65048e20debf013ebc4eafdc4855df7f8f62e09a7617c2f4b103244"),
}


def _cost_control_stream(kind):
    widths, prefix, length, matches, tail, _, _ = _COST_STREAM_CASES[kind]
    raw = bytes(1048576)
    assert prefix + matches * length + tail == len(raw)
    literal_lengths = [widths.get(symbol, 0) for symbol in range(max(widths) + 1)]
    code_lengths = [0] * 19
    code_lengths[0], code_lengths[1], code_lengths[2] = 1, 2, 2
    writer = _Writer()
    writer.fields(_cost_dynamic_header(len(literal_lengths), 1, code_lengths))
    cl = _cost_canonical(code_lengths)
    writer.fields(cl[width] for width in literal_lengths + [1])
    ll = _cost_canonical(literal_lengths)

    def literals(count):
        value, width = ll[0]
        if value == 0:
            writer.put(0, count * width)
        else:
            for _ in range(count):
                writer.put(value, width)

    literals(prefix)
    if matches:
        match_code = ll[257 if length == 3 else 285]
        if match_code[0] == 0:
            writer.put(0, matches * (match_code[1] + 1))
        else:
            for _ in range(matches):
                writer.put(*match_code)
                writer.put(0, 1)  # DD0: distance one, no extra bits.
    literals(tail)
    writer.put(*ll[256])
    return _wrap(writer.finish(), raw), raw


@pytest.mark.parametrize("kind", list(_COST_STREAM_CASES))
def test_pipeline_token_cost_control_outcomes(kind, cuda_device, monkeypatch):
    payload, raw = _cost_control_stream(kind)
    assert len(payload) == _COST_STREAM_CASES[kind][-2]
    assert hashlib.sha256(payload).hexdigest() == _COST_STREAM_CASES[kind][-1]
    assert _inflate_exact(payload) == raw
    # The existing helper selects the actual JAX device and uses nondefault
    # 4096-candidate/128-block caps to exercise queued decode.
    call, value = _prepare_call(cuda_device, payload, len(raw))
    with _no_cpu_codec(monkeypatch):
        result, metadata = _completed(call, value)
        assert metadata.tolist() == [0, 0]
        _assert_bytes(result, raw, cuda_device)
