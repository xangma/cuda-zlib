# Copyright (c) 2026 xangma
# SPDX-License-Identifier: MIT

"""Extracted PyCBC byte-level regression fixtures; no PyCBC/JAX import required."""

import builtins
import contextlib
import functools
import gc
import random
import struct
import sys
import types
import zlib

import numpy as np
import pytest

from cuda_zlib import CodecError as GWFFormatError


@pytest.fixture(scope="module")
def decoder():
    from cuda_zlib import decompress_zlib
    return decompress_zlib


@pytest.fixture(scope="module")
def cuda_device():
    cupy = pytest.importorskip("cupy")
    try:
        count = cupy.cuda.runtime.getDeviceCount()
    except cupy.cuda.runtime.CUDARuntimeError as exc:
        pytest.skip(str(exc))
    if not count:
        pytest.skip("CUDA unavailable")
    return 0


@pytest.fixture
def forbid_cpu_inflation(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("fresh CUDA decoding attempted CPU zlib inflation")

    # Install before the first candidate call, not after cache preparation.
    monkeypatch.setattr(zlib, "decompress", forbidden)
    monkeypatch.setattr(zlib, "decompressobj", forbidden)


def _header(cmf=0x78, flags=0):
    return bytes((cmf, flags | (-(cmf * 256 + flags) % 31)))


def _wrap(body, raw):
    return _header() + body + struct.pack(">I", zlib.adler32(raw))


def _stored(raw, final=True):
    assert len(raw) <= 65535
    return (bytes((int(final),))
            + struct.pack("<HH", len(raw), len(raw) ^ 0xFFFF) + raw)


def _compressed(raw, level=6, strategy=zlib.Z_DEFAULT_STRATEGY,
                wbits=zlib.MAX_WBITS, flush=zlib.Z_FINISH):
    compressor = zlib.compressobj(level, zlib.DEFLATED, wbits, 8, strategy)
    return compressor.compress(raw) + compressor.flush(flush)


def _dynamic_raw(size=32768):
    rng = random.Random(87123)
    return bytes(rng.choices(range(16), weights=range(1, 17), k=size))


def _bits(fields):
    """Pack RFC 1951 numeric fields least-significant bit first."""
    result, offset = bytearray(), 0
    for value, width in fields:
        for bit in range(width):
            if offset % 8 == 0:
                result.append(0)
            result[-1] |= ((value >> bit) & 1) << (offset % 8)
            offset += 1
    return bytes(result)


def _code(value, width):
    # Huffman codes appear most-significant bit first within the bit stream.
    return int(format(value, "0%db" % width)[::-1], 2), width


def _codes(lengths):
    counts = [lengths.count(width) for width in range(16)]
    next_code, code = {}, 0
    for width in range(1, 16):
        code = (code + (counts[width - 1] if width > 1 else 0)) << 1
        next_code[width] = code
    result = {}
    for symbol, width in enumerate(lengths):
        if width:
            result[symbol] = _code(next_code[width], width)
            next_code[width] += 1
    return result


def _dynamic_header(literal_count, distance_count, code_lengths, final=True):
    order = [16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3,
             13, 2, 14, 1, 15]
    count = max(4, max(order.index(symbol) + 1
                       for symbol, width in enumerate(code_lengths) if width))
    return ([(int(final), 1), (2, 2), (literal_count - 257, 5),
             (distance_count - 1, 5), (count - 4, 4)]
            + [(code_lengths[symbol], 3) for symbol in order[:count]])


def _dynamic_tree_stream(kind):
    literal = [0] * (258 if kind == "single-distance" else 257)
    distance = [0]
    if kind == "single-eob":
        literal[256] = 1
        raw, tokens = b"", [("literal", 256)]
    elif kind == "single-distance":
        literal[65], literal[256], literal[257] = 1, 2, 2
        distance[0] = 1
        raw = b"AAAA"
        tokens = [("literal", 65), ("literal", 257),
                  ("distance", 0), ("literal", 256)]
    else:
        literal[65], literal[256] = 1, 1
        raw = b"A" * 4096
        tokens = [("literal", 65)] * len(raw) + [("literal", 256)]
    # Complete code-length tree: zero=0, one=10, two=11.
    code_lengths = [0] * 19
    code_lengths[0], code_lengths[1], code_lengths[2] = 1, 2, 2
    fields = _dynamic_header(len(literal), len(distance), code_lengths)
    length_codes = _codes(code_lengths)
    fields += [length_codes[width] for width in literal + distance]
    tables = {"literal": _codes(literal), "distance": _codes(distance)}
    fields += [tables[alphabet][symbol] for alphabet, symbol in tokens]
    return _wrap(_bits(fields), raw), raw


def _invalid_repeat_stream(kind):
    symbol = 16 if kind == "repeat-before-length" else 18
    code_lengths = [0] * 19
    code_lengths[0] = code_lengths[symbol] = 1
    fields = _dynamic_header(257, 1, code_lengths)
    symbol_code = _codes(code_lengths)[symbol]
    if symbol == 16:
        # Repeat 3 previous lengths, before any length has been emitted.
        fields += [symbol_code, (0, 2)]
    else:
        fields += [symbol_code, (127, 7)] * 2  # 276 lengths exceed 258 slots.
    return _wrap(_bits(fields), b"")


def _padded_final_stream():
    # A fixed literal A and EOB use 18 bits. The remaining 6 bits are padding.
    body = _bits([(1, 1), (1, 2), _code(113, 8), _code(0, 7)])
    body = body[:-1] + bytes((body[-1] | 0xFC,))
    return _wrap(body, b"A")


def _distance_window_stream():
    prefix = bytes(range(256)) * 2
    # Fixed length 3, distance symbol 16 (base 257), then EOB.
    tail = _bits([(1, 1), (1, 2), _code(1, 7), _code(16, 5),
                  (0, 7), _code(0, 7)])
    body = _stored(prefix, final=False) + tail
    raw = prefix + prefix[-257:-254]
    payload = _wrap(body, raw)
    return payload, _header(cmf=0x08) + payload[2:], raw


def _dynamic_literal_fields(raw, final=True, long_codes=False,
                            single_eob=False):
    """Independent literal-only trees, with no LZ77 compressor decisions."""
    literal = [0] * 257
    code_lengths = [0] * 19
    if long_codes:
        # A complete tree with depths 1..14, then two leaves at depth 15.
        for symbol in range(14):
            literal[symbol] = symbol + 1
        literal[14] = literal[256] = 15
        code_lengths[:16] = [4] * 16
    else:
        literal[256] = 1
        if not single_eob:
            literal[65] = 1
        code_lengths[0] = code_lengths[1] = 1
    fields = _dynamic_header(257, 1, code_lengths, final)
    length_codes, literal_codes = _codes(code_lengths), _codes(literal)
    fields += [length_codes[width] for width in literal + [0]]
    fields += [literal_codes[symbol] for symbol in raw]
    fields.append(literal_codes[256])
    return fields


def _alignment_stream(prefix_count):
    # Fixed literals 144..255 have 9-bit codes. The next block starts at
    # (10 + 9 * prefix_count) mod 8, covering every alignment for counts 0..7.
    prefix = ([(0, 1), (1, 2)] + [_code(400, 9)] * prefix_count
              + [_code(0, 7)])
    suffix = b"A" * 37
    body = _bits(prefix + _dynamic_literal_fields(suffix))
    raw = b"\x90" * prefix_count + suffix
    return _wrap(body, raw), raw


def _embedded_candidate_stream():
    empty = _bits(_dynamic_literal_fields(b"", single_eob=True))
    valid = _bits(_dynamic_literal_fields(b"A"))
    invalid_fields = _dynamic_literal_fields(b"", single_eob=True)
    invalid = _bits(invalid_fields[:-1] + [(1, 1)])
    # These are ordinary stored literals, including complete BFINAL blocks
    # and a valid dynamic header whose body uses its unused Huffman code.
    stored = (b"candidate-boundary\x00" + empty + valid + invalid) * 16
    suffix = b"A" * 2048
    body = _stored(stored, final=False) + _bits(
        _dynamic_literal_fields(suffix))
    return _wrap(body, stored + suffix), stored + suffix


def _many_dynamic_stream(empty):
    fields, raw_parts = [], []
    for index in range(128):
        raw = b"" if empty else b"A" * (index % 3)
        raw_parts.append(raw)
        fields += _dynamic_literal_fields(raw, final=index == 127)
    raw = b"".join(raw_parts)
    return _wrap(_bits(fields), raw), raw


def _long_literal_stream():
    # Force 65,536 literal tokens at the maximum Deflate code length; zlib
    # would normally replace this plaintext with short overlapping matches.
    raw = bytes((14,)) * 65536
    body = _bits(_dynamic_literal_fields(raw, long_codes=True))
    return _wrap(body, raw), raw


def _truncated_prefix_tail_stream(bit_offset):
    code_lengths = [0] * 19
    code_lengths[0] = code_lengths[1] = 1
    # The complete code-length header is 71 bits. Discovery sees a valid
    # prefix within the final 9–10 bytes, but the literal tree is truncated.
    fields = [(0, bit_offset)] + _dynamic_header(257, 1, code_lengths)
    raw = b"bounded discovery tail\x00" + _bits(fields)
    return _wrap(_stored(raw), raw), raw


def _grid_stride_stream():
    # Exceed the current 16,384-CTA * 128-byte first scan pass. The actual
    # dynamic boundary lies in a subsequent grid-stride iteration.
    prefix = b"\x00" * (2**21 + 37)
    blocks = [_stored(prefix[index:index + 65535], final=False)
              for index in range(0, len(prefix), 65535)]
    suffix = b"A" * 37
    body = b"".join(blocks) + _bits(_dynamic_literal_fields(suffix))
    return _wrap(body, prefix + suffix), prefix + suffix


def _assert_bytes(result, raw, device):
    import cupy as cp
    assert result.dtype == cp.uint8
    assert result.device.id == device
    assert result.get().tobytes() == raw


def test_handbuilt_deflate_fixtures_match_independent_stdlib():
    # Oracle validation is outside the candidate's CPU-inflation guard.
    for kind in ["literal-only-empty-distance", "single-eob",
                 "single-distance"]:
        payload, raw = _dynamic_tree_stream(kind)
        assert zlib.decompress(payload) == raw
    for kind in ["repeat-before-length", "repeat-overflow"]:
        with pytest.raises(zlib.error):
            zlib.decompress(_invalid_repeat_stream(kind))
    assert zlib.decompress(_padded_final_stream()) == b"A"
    payload, small_window, raw = _distance_window_stream()
    assert zlib.decompress(payload) == raw
    # stdlib accepts this smaller CINFO despite its over-window distance.
    assert zlib.decompress(small_window) == raw


def test_discovery_stress_fixtures_match_independent_stdlib():
    streams = [_alignment_stream(index) for index in range(8)]
    streams += [_embedded_candidate_stream(), _many_dynamic_stream(True),
                _many_dynamic_stream(False), _long_literal_stream()]
    streams += [_truncated_prefix_tail_stream(index) for index in range(8)]
    streams.append(_grid_stride_stream())
    for payload, raw in streams:
        assert zlib.decompress(payload) == raw
    # Invalid speculative bodies are harmless only when they are stored data.
    fields = _dynamic_literal_fields(b"", single_eob=True)
    bad_body = _bits(fields[:-1] + [(1, 1)])
    with pytest.raises(zlib.error):
        zlib.decompress(_wrap(bad_body, b""))


@pytest.mark.parametrize("expected", [-1, 1.5, "4", True])
def test_invalid_size_is_rejected_before_optional_backend(
    decoder, monkeypatch, expected,
):
    actual_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name.split(".")[0] in {"jax", "cupy"}:
            pytest.fail("invalid input loaded the optional GPU backend")
        return actual_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    with pytest.raises((TypeError, ValueError)):
        decoder(zlib.compress(b"test"), expected, device=0)


@pytest.mark.parametrize("payload", [
    b"", b"x", b"\x78\x9c", b"\x78\x00" + b"\x00" * 8,
    _header(cmf=0x79) + b"\x00" * 8,  # Unsupported compression method.
    _header(cmf=0x88) + b"\x00" * 8,  # Invalid window size.
    _header(flags=0x20) + b"\x00" * 12,  # Preset dictionary.
    b"\x1f\x8b" + b"\x00" * 12,  # Gzip is not RFC 1950 zlib.
], ids=["empty", "one-byte", "header-only", "header-check", "method",
        "window", "dictionary", "gzip"])
def test_bad_wrapper_is_rejected_without_cuda(decoder, monkeypatch, payload):
    from cuda_zlib import UnsupportedStream as GWFDeflateUnavailable

    actual_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name.split(".")[0] in {"jax", "cupy"}:
            pytest.fail("invalid framing loaded the optional GPU backend")
        return actual_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    expected = (GWFDeflateUnavailable if payload.startswith(b"\x1f\x8b")
                else (ValueError, GWFDeflateUnavailable))
    match = "RFC 1950 zlib" if expected is GWFDeflateUnavailable else None
    with pytest.raises(expected, match=match):
        decoder(payload, 0, device=0)


@pytest.mark.parametrize("encoding", ["stored", "fixed", "dynamic"])
def test_fresh_zlib_block_types(
    decoder, cuda_device, forbid_cpu_inflation, encoding,
):
    if encoding == "stored":
        raw = bytes(range(256)) * 512
        payload = _compressed(raw, level=0)
        block_type = 0
    elif encoding == "fixed":
        raw = b"fixed Huffman, overlapping history\x00" * 256
        payload = _compressed(raw, strategy=zlib.Z_FIXED)
        block_type = 1
    else:
        raw = _dynamic_raw()
        payload = _compressed(raw)
        block_type = 2
    assert (payload[2] >> 1) & 3 == block_type
    _assert_bytes(decoder(payload, len(raw), cuda_device), raw, cuda_device)


@pytest.mark.parametrize("raw", [
    b"", b"A" * 65536, bytes(range(251)) * 256,
], ids=["empty", "overlap-distance-one", "distinct-history"])
def test_empty_and_overlapping_or_distinct_history(
    decoder, cuda_device, forbid_cpu_inflation, raw,
):
    payload = _compressed(raw, strategy=zlib.Z_FIXED)
    _assert_bytes(decoder(payload, len(raw), cuda_device), raw, cuda_device)


@pytest.mark.parametrize("distance", [8192, 32768])
def test_history_survives_sync_flush_block_boundary(
    decoder, cuda_device, forbid_cpu_inflation, distance,
):
    rng = random.Random(11219)
    first = bytes(rng.randrange(256) for _ in range(distance))
    compressor = zlib.compressobj()
    prefix = compressor.compress(first) + compressor.flush(zlib.Z_SYNC_FLUSH)
    # At the 32 KiB boundary zlib's lookahead reduces the usable distance.
    repeat = first if distance < 32768 else first[1024:]
    suffix = compressor.compress(repeat) + compressor.flush()
    assert len(suffix) < len(repeat) // 4, (
        "fixture must use prior-block history")
    raw = first + repeat
    _assert_bytes(decoder(prefix + suffix, len(raw), cuda_device),
                  raw, cuda_device)


def test_mixed_stored_fixed_dynamic_blocks(
    decoder, cuda_device, forbid_cpu_inflation,
):
    stored = bytes(range(256)) * 16
    fixed = b"three-byte overlapping repetition" * 256
    dynamic = _dynamic_raw()
    fixed_body = _compressed(fixed, strategy=zlib.Z_FIXED, wbits=-15,
                             flush=zlib.Z_SYNC_FLUSH)
    dynamic_body = _compressed(dynamic, wbits=-15, flush=zlib.Z_SYNC_FLUSH)
    assert (fixed_body[0] >> 1) & 3 == 1
    assert (dynamic_body[0] >> 1) & 3 == 2
    raw = stored + fixed + dynamic
    body = (_stored(stored, final=False) + fixed_body + dynamic_body
            + _stored(b""))
    _assert_bytes(decoder(_wrap(body, raw), len(raw), cuda_device),
                  raw, cuda_device)


def test_sync_and_full_flush_preserve_output(
    decoder, cuda_device, forbid_cpu_inflation,
):
    chunks = [_dynamic_raw(8192), b"AB" * 4096, bytes(range(256)) * 32]
    compressor = zlib.compressobj()
    parts = []
    for chunk, mode in zip(chunks, [zlib.Z_SYNC_FLUSH, zlib.Z_FULL_FLUSH,
                                    zlib.Z_FINISH]):
        parts.append(compressor.compress(chunk) + compressor.flush(mode))
    raw = b"".join(chunks)
    _assert_bytes(decoder(b"".join(parts), len(raw), cuda_device),
                  raw, cuda_device)


@pytest.mark.parametrize("kind", [
    "literal-only-empty-distance", "single-eob", "single-distance",
])
def test_dynamic_literal_only_and_single_symbol_trees(
    decoder, cuda_device, forbid_cpu_inflation, kind,
):
    payload, raw = _dynamic_tree_stream(kind)
    _assert_bytes(decoder(payload, len(raw), cuda_device), raw, cuda_device)


@pytest.mark.parametrize("prefix_count", range(8))
def test_dynamic_discovery_at_every_bit_alignment(
    decoder, cuda_device, forbid_cpu_inflation, prefix_count,
):
    payload, raw = _alignment_stream(prefix_count)
    _assert_bytes(decoder(payload, len(raw), cuda_device), raw, cuda_device)


def test_valid_and_invalid_speculative_blocks_inside_stored_literals(
    decoder, cuda_device, forbid_cpu_inflation,
):
    payload, raw = _embedded_candidate_stream()
    _assert_bytes(decoder(payload, len(raw), cuda_device), raw, cuda_device)


@pytest.mark.parametrize("empty", [True, False], ids=["empty", "tiny"])
def test_many_dynamic_blocks_with_fewer_output_bytes_than_blocks(
    decoder, cuda_device, forbid_cpu_inflation, empty,
):
    payload, raw = _many_dynamic_stream(empty)
    assert len(raw) < 128
    _assert_bytes(decoder(payload, len(raw), cuda_device), raw, cuda_device)


def test_large_dynamic_literal_stream_with_maximum_length_codes(
    decoder, cuda_device, forbid_cpu_inflation,
):
    payload, raw = _long_literal_stream()
    _assert_bytes(decoder(payload, len(raw), cuda_device), raw, cuda_device)


@pytest.mark.parametrize("bit_offset", range(8))
def test_discovery_ignores_truncated_speculative_header_at_payload_end(
    decoder, cuda_device, forbid_cpu_inflation, bit_offset,
):
    payload, raw = _truncated_prefix_tail_stream(bit_offset)
    _assert_bytes(decoder(payload, len(raw), cuda_device), raw, cuda_device)


def test_discovery_reaches_dynamic_block_after_large_stored_prefix(
    decoder, cuda_device, forbid_cpu_inflation,
):
    payload, raw = _grid_stride_stream()
    _assert_bytes(decoder(payload, len(raw), cuda_device), raw, cuda_device)


@pytest.mark.parametrize("kind", ["repeat-before-length", "repeat-overflow"])
def test_malformed_dynamic_code_length_repeat(
    decoder, cuda_device, forbid_cpu_inflation, kind,
):
    with pytest.raises(GWFFormatError, match="Deflate|Adler32"):
        decoder(_invalid_repeat_stream(kind), 0, cuda_device)


def test_nonzero_final_byte_padding_is_valid(
    decoder, cuda_device, forbid_cpu_inflation,
):
    _assert_bytes(decoder(_padded_final_stream(), 1, cuda_device),
                  b"A", cuda_device)


def test_backward_distance_obeys_declared_zlib_window(
    decoder, cuda_device, forbid_cpu_inflation,
):
    payload, small_window, raw = _distance_window_stream()
    _assert_bytes(decoder(payload, len(raw), cuda_device), raw, cuda_device)
    # CINFO=0 advertises a 256-byte window, smaller than distance 257.
    with pytest.raises(GWFFormatError, match="Deflate|Adler32"):
        decoder(small_window, len(raw), cuda_device)


@pytest.mark.parametrize("change", [
    "too-small", "too-large", "adler", "truncated-trailer", "truncated-body",
    "trailing-byte", "concatenated", "reserved-block", "stored-length",
    "distance-before-history",
])
def test_bad_stream_fails_without_cpu_inflation(
    decoder, cuda_device, forbid_cpu_inflation, change,
):
    raw = b"a checked fresh compressed stream\x00" * 128
    payload, expected = _compressed(raw), len(raw)
    if change == "too-small":
        expected -= 1
    elif change == "too-large":
        expected += 1
    elif change == "adler":
        payload = payload[:-1] + bytes((payload[-1] ^ 1,))
    elif change == "truncated-trailer":
        payload = payload[:-1]
    elif change == "truncated-body":
        payload = payload[:-6] + payload[-4:]
    elif change == "trailing-byte":
        payload += b"\x00"
    elif change == "concatenated":
        payload += payload
    elif change == "reserved-block":
        payload = _wrap(b"\x07\x00", raw)
    elif change == "stored-length":
        body = bytearray(_stored(raw))
        body[3] ^= 1  # LEN and NLEN must be complements.
        payload = _wrap(bytes(body), raw)
    elif change == "distance-before-history":
        # Fixed code 257 (length 3), distance 1, before any literal output.
        payload, expected = _wrap(b"\x03\x02\x00", b"AAA"), 3
    with pytest.raises(GWFFormatError, match="Deflate|Adler32"):
        decoder(payload, expected, cuda_device)


def test_returned_array_survives_subsequent_decode(
    decoder, cuda_device, forbid_cpu_inflation,
):
    first_raw = bytes(range(256)) * 16
    first = decoder(_compressed(first_raw), len(first_raw), cuda_device)
    second_raw = b"z" * len(first_raw)
    second = decoder(_compressed(second_raw), len(second_raw), cuda_device)
    del second
    gc.collect()
    _assert_bytes(first, first_raw, cuda_device)

