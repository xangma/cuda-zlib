# Copyright (c) 2026 xangma
# SPDX-License-Identifier: MIT

"""Independent byte-level regression fixtures; GPU imports remain optional."""

import builtins
import gc
import random
import struct
import zlib

import numpy as np
import pytest

from cuda_zlib import CodecError


@pytest.fixture(scope="module")
def decoder():
    from cuda_zlib import decompress_zlib
    return decompress_zlib


@pytest.fixture(scope="module")
def cuda_device():
    jax = pytest.importorskip("jax")
    try:
        devices = jax.devices("gpu")
    except RuntimeError as exc:
        pytest.skip(str(exc))
    if not devices or devices[0].platform != "gpu":
        pytest.skip("JAX CUDA backend unavailable")
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


def _prefixed_distance_window_stream(prefix_bytes):
    payload, _, tail = _distance_window_stream()
    prefix = random.Random(prefix_bytes).randbytes(prefix_bytes)
    blocks = [_stored(prefix[index:index + 65535], final=False)
              for index in range(0, prefix_bytes, 65535)]
    raw = prefix + tail
    payload = _wrap(b"".join(blocks) + payload[2:-4], raw)
    return payload, _header(cmf=0x08) + payload[2:], raw


def _medium_period_stream():
    period = random.Random(91577).randbytes(257)
    raw = (period * ((262144 + 256) // 257))[:262144]
    return _compressed(raw), raw


def _medium_compound_stream(history_error=False, window_error=False, reserved=False):
    fixed = _codes([8] * 144 + [9] * 112 + [7] * 24 + [8] * 8)
    fields = [(1, 1), (1, 2)]  # Final fixed-Huffman block.
    if history_error:
        # Length 258, distance 1 before any literal. The raw bytes below give
        # an intended extent/checksum, not a valid history oracle.
        fields += [fixed[285], _code(0, 5)]
        produced = 258
    else:
        fields.append(fixed[65])
        produced = 1
    matches, literals = divmod(70000 - produced, 258)
    fields += [fixed[285], _code(0, 5)] * matches
    fields += [fixed[65]] * literals
    raw = b"A" * 70000
    if window_error:
        # Existing history is sufficient, but CINFO=0 declares only 256 bytes.
        fields += [fixed[285], _code(16, 5), (0, 7)]  # Distance 257.
        raw += b"A" * 258
    fields.append(fixed[286 if reserved else 256])
    payload = _wrap(_bits(fields), raw)
    if window_error:
        payload = _header(cmf=0x08) + payload[2:]
    return payload, raw


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


def _late_code_length_stream(prefix_count):
    # With 19 code lengths, the header extends beyond the 64-bit scan word
    # at every alignment. The tree becomes complete only at the last entry.
    code_lengths = [0] * 19
    code_lengths[0] = code_lengths[1] = 2
    code_lengths[15] = 1
    literal = [0] * 257
    literal[65] = literal[256] = 1
    codes = _codes(code_lengths)
    fields = _dynamic_header(257, 1, code_lengths)
    fields += [codes[width] for width in literal + [0]]
    literal_codes = _codes(literal)
    fields += [literal_codes[65]] * 37 + [literal_codes[256]]
    prefix = ([(0, 1), (1, 2)] + [_code(400, 9)] * prefix_count
              + [_code(0, 7)])
    raw = b"\x90" * prefix_count + b"A" * 37
    return _wrap(_bits(prefix + fields), raw), raw


def test_late_code_length_fixture_matches_stdlib():
    for prefix_count in range(8):
        payload, raw = _late_code_length_stream(prefix_count)
        assert zlib.decompress(payload) == raw


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


def _deep_reference_stream(match_blocks):
    seed = bytes(range(256)) + b"\x00\x01"
    fixed = _codes([8] * 144 + [9] * 112 + [7] * 24 + [8] * 8)
    fields = []
    for index in range(match_blocks):
        # Length 258, distance 258: every output byte references the prior block.
        fields += [(int(index + 1 == match_blocks), 1), (1, 2), fixed[285],
                   _code(16, 5), (1, 7), fixed[256]]
    raw = seed * (match_blocks + 1)
    return _wrap(_stored(seed, final=False) + _bits(fields), raw), raw


def _fixed_summary_stream(invalid=False):
    prefix, literals = b"A" * 17, bytes(range(256)) * 4
    fixed = _codes([8] * 144 + [9] * 112 + [7] * 24 + [8] * 8)
    fields = _dynamic_literal_fields(prefix, final=False) + [(1, 1), (1, 2)]
    fields += [fixed[value] for value in literals]
    if invalid:
        fields.append(fixed[286])  # Reserved literal/length symbol after valid data.
    fields.append(fixed[256])
    raw = prefix + literals
    return _wrap(_bits(fields), raw), raw


def _limited_checked_decoder(device, expected, candidates=4096, blocks=256):
    import jax
    from cuda_zlib._ffi import load_backend

    _, target = load_backend(device)
    call = jax.ffi.ffi_call(target, (
        jax.ShapeDtypeStruct((expected,), np.uint8),
        jax.ShapeDtypeStruct((2,), np.uint32)), vmap_method="sequential")
    return jax.jit(lambda value: call(value, max_candidates=np.int64(candidates),
                                    max_blocks=np.int64(blocks)))


def _assert_bytes(result, raw, device):
    import jax
    assert isinstance(result, jax.Array)
    assert result.dtype == np.uint8
    expected_device = jax.devices("gpu")[device] if isinstance(device, int) else device
    assert result.devices() == {expected_device}
    assert np.asarray(result.block_until_ready()).tobytes() == raw


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
    for match_blocks in (257, 1050):
        payload, raw = _deep_reference_stream(match_blocks)
        assert zlib.decompress(payload) == raw
    for prefix_bytes in (65537, 1048577):
        payload, _, raw = _prefixed_distance_window_stream(prefix_bytes)
        assert zlib.decompress(payload) == raw


def _dense_prefix_stored_stream():
    # A complete code-length prefix followed by an invalid repeat. Repetition
    # creates many speculative matches but no usable dynamic block at each
    # marker; all markers are ordinary literals in a valid stored stream.
    marker = _invalid_repeat_stream("repeat-before-length")[2:-4]
    raw = (marker * ((2 * 1024 ** 2 + len(marker) - 1) // len(marker)))[:2 * 1024 ** 2]
    chunks = [raw[i:i + 65535] for i in range(0, len(raw), 65535)]
    body = b"".join(_stored(chunk, final=i == len(chunks) - 1)
                    for i, chunk in enumerate(chunks))
    return _wrap(body, raw), raw


def test_discovery_stress_fixtures_match_independent_stdlib():
    streams = [_alignment_stream(index) for index in range(8)]
    streams += [_embedded_candidate_stream(), _many_dynamic_stream(True),
                _many_dynamic_stream(False), _long_literal_stream()]
    streams += [_truncated_prefix_tail_stream(index) for index in range(8)]
    streams += [_grid_stride_stream(), _dense_prefix_stored_stream()]
    for payload, raw in streams:
        assert zlib.decompress(payload) == raw
    # Invalid speculative bodies are harmless only when they are stored data.
    fields = _dynamic_literal_fields(b"", single_eob=True)
    bad_body = _bits(fields[:-1] + [(1, 1)])
    with pytest.raises(zlib.error):
        zlib.decompress(_wrap(bad_body, b""))


def test_medium_repetitive_fixtures_match_independent_stdlib():
    for size in (65536, 65537, 262144, 1048576, 1048577):
        raw = bytes(size)
        assert zlib.decompress(_compressed(raw)) == raw
    payload, raw = _medium_period_stream()
    assert len(raw) / len(payload) >= 64
    # Unique three-byte phases require distance >=257 for every encoded match.
    period = raw[:257] + raw[:2]
    assert len({period[index:index + 3] for index in range(257)}) == 257
    assert zlib.decompress(payload) == raw


def test_medium_compound_fixtures_match_independent_stdlib():
    for window_error in (False, True):
        payload, raw = _medium_compound_stream(window_error=window_error)
        # stdlib accepts the CINFO violation; the codec separately enforces it.
        assert zlib.decompress(payload) == raw
    for options in ({"history_error": True, "reserved": True},
                    {"window_error": True, "reserved": True}):
        payload, _ = _medium_compound_stream(**options)
        with pytest.raises(zlib.error):
            zlib.decompress(payload)
    payload, _ = _medium_compound_stream()
    with pytest.raises(zlib.error):
        zlib.decompress(payload[:-1] + bytes((payload[-1] ^ 1,)))


def test_fixed_summary_and_missing_final_fixtures_match_stdlib():
    payload, raw = _fixed_summary_stream()
    assert zlib.decompress(payload) == raw
    with pytest.raises(zlib.error):
        zlib.decompress(_fixed_summary_stream(invalid=True)[0])
    with pytest.raises(zlib.error):
        zlib.decompress(_wrap(_stored(b"A", final=False), b"A"))


@pytest.mark.parametrize("expected", [-1, 1.5, "4", True])
def test_invalid_size_is_rejected_before_optional_backend(
    decoder, monkeypatch, expected,
):
    actual_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name.split(".")[0] in {"jax"}:
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
    from cuda_zlib import UnsupportedStream

    actual_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name.split(".")[0] in {"jax"}:
            pytest.fail("invalid framing loaded the optional GPU backend")
        return actual_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    expected = (UnsupportedStream if payload.startswith(b"\x1f\x8b")
                else (ValueError, UnsupportedStream))
    match = "RFC 1950 zlib" if expected is UnsupportedStream else None
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


@pytest.mark.parametrize("prefix_count", range(8))
def test_dynamic_discovery_with_late_code_length(
    decoder, cuda_device, forbid_cpu_inflation, prefix_count,
):
    payload, raw = _late_code_length_stream(prefix_count)
    _assert_bytes(decoder(payload, len(raw), cuda_device), raw, cuda_device)


@pytest.mark.parametrize("dense", [False, True], ids=["random", "prefix-queue-overflow"])
def test_large_stored_stream_discovery(
    decoder, cuda_device, forbid_cpu_inflation, dense,
):
    if dense:
        payload, raw = _dense_prefix_stored_stream()
    else:
        raw = random.Random(92771).randbytes(2 * 1024 ** 2)
        payload = _compressed(raw, level=0)
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
    with pytest.raises(CodecError, match="Deflate|Adler32"):
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
    with pytest.raises(CodecError, match="Deflate|Adler32"):
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
    with pytest.raises(CodecError, match="Deflate|Adler32"):
        decoder(payload, expected, cuda_device)


@pytest.mark.parametrize("length", [0, 4095, 4096, 4097])
@pytest.mark.parametrize("encoding", ["stored", "local-fixed", "literal-dynamic"])
def test_local_and_literal_emission_validate_fused_checksum(
    decoder, cuda_device, forbid_cpu_inflation, length, encoding,
):
    # Exercise checksum seams and each resolved emission path through native FFI.
    raw = (bytes(index % 251 for index in range(length))
           if encoding == "stored" else b"A" * length)
    if encoding == "stored":
        payload = _wrap(_stored(raw), raw)
    elif encoding == "local-fixed":
        payload = _compressed(raw, strategy=zlib.Z_FIXED)
    else:
        payload = _wrap(_bits(_dynamic_literal_fields(raw)), raw)
    _assert_bytes(decoder(payload, length, cuda_device), raw, cuda_device)
    bad_checksum = payload[:-1] + bytes((payload[-1] ^ 1,))
    with pytest.raises(CodecError, match="Adler32"):
        decoder(bad_checksum, length, cuda_device)
    # A failed native status must not poison the next invocation.
    _assert_bytes(decoder(payload, length, cuda_device), raw, cuda_device)


def test_external_emission_resolves_cross_block_history_and_checksum(
    decoder, cuda_device, forbid_cpu_inflation,
):
    payload, _, raw = _distance_window_stream()
    _assert_bytes(decoder(payload, len(raw), cuda_device), raw, cuda_device)
    bad_checksum = payload[:-1] + bytes((payload[-1] ^ 1,))
    with pytest.raises(CodecError, match="Adler32"):
        decoder(bad_checksum, len(raw), cuda_device)
    _assert_bytes(decoder(payload, len(raw), cuda_device), raw, cuda_device)


def test_external_reference_flag_does_not_hide_another_blocks_error(
    decoder, cuda_device, forbid_cpu_inflation,
):
    # Both dynamic blocks describe three bytes from distance one. The first
    # has no preceding history and must fail; the second requires refinement.
    literal = [0] * 258
    literal[256] = literal[257] = 1
    lengths = [0] * 19
    lengths[0] = lengths[1] = 1
    length_codes, literal_codes = _codes(lengths), _codes(literal)
    fields = []
    for final in [False, True]:
        fields += _dynamic_header(258, 1, lengths, final)
        fields += [length_codes[width] for width in literal + [1]]
        fields += [literal_codes[257], (0, 1), literal_codes[256]]
    payload = _wrap(_bits(fields), b"A" * 6)
    with pytest.raises(CodecError, match="backward distance|Deflate status 6"):
        decoder(payload, 6, cuda_device)


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


@pytest.mark.parametrize("size", [65537, 1048577])
def test_large_jit_checked_errors_and_recovery(
    cuda_device, forbid_cpu_inflation, size,
):
    import jax
    import jax.numpy as jnp
    from cuda_zlib import decompress_zlib_checked

    device = jax.devices("gpu")[cuda_device]
    raw = random.Random(size).randbytes(size)
    # Stored data keeps the larger input above the prefix-discovery threshold.
    payload = _compressed(raw, level=0)
    source = jax.device_put(np.frombuffer(payload, np.uint8), device)
    decode = jax.jit(
        lambda value, expected: decompress_zlib_checked(value, expected, device),
        static_argnums=1,
    )
    cases = (
        (source.at[0].set(jnp.uint8(0)), size, 15),
        (source.at[-1].set(source[-1] ^ jnp.uint8(1)), size, 21),
        (source.at[2].set((source[2] & jnp.uint8(0xF8)) | jnp.uint8(7)),
         size, None),
        (source, size - 1, None),
        (source, size + 1, None),
    )
    for malformed, expected, wanted_status in cases:
        failed, metadata = decode(malformed, expected)
        assert failed.shape == (expected,) and failed.dtype == np.uint8
        assert metadata.shape == (2,) and metadata.dtype == np.uint32
        assert failed.devices() == metadata.devices() == {device}
        status, reserved = map(int, np.asarray(metadata))
        assert reserved == 0
        if wanted_status is None:
            assert status != 0
        else:
            assert status == wanted_status
        recovered, recovered_metadata = decode(source, size)
        np.testing.assert_array_equal(np.asarray(recovered_metadata), [0, 0])
        _assert_bytes(recovered, raw, device)


@pytest.mark.parametrize("prefix_bytes", [65537, 1048577],
                         ids=["direct-discovery", "prefix-discovery"])
def test_jit_resident_framing_precedence_window_and_recovery(
    cuda_device, forbid_cpu_inflation, prefix_bytes,
):
    import jax
    from cuda_zlib import decompress_zlib_checked

    device = jax.devices("gpu")[cuda_device]
    payload, small_window, raw = _prefixed_distance_window_stream(prefix_bytes)
    assert len(raw) > 65536
    assert (len(payload) - 6 > 2**20) == (prefix_bytes > 2**20)
    decode = jax.jit(lambda value: decompress_zlib_checked(value, len(raw), device))
    source = jax.device_put(np.frombuffer(payload, np.uint8), device)
    # Framing must win even when the Deflate body and Adler32 are also invalid.
    damaged = b"\x07" + payload[3:-1] + bytes((payload[-1] ^ 1,))
    cases = (
        (_header(cmf=0x88) + damaged, 15),
        (payload[:1] + bytes((payload[1] ^ 1,)) + damaged, 15),
        (b"\x1f\x8b" + damaged, 16),
        (_header(flags=0x20) + damaged, 17),
        (small_window, 6),
    )
    for malformed, status in cases:
        resident = jax.device_put(np.frombuffer(malformed, np.uint8), device)
        _, metadata = decode(resident)
        assert metadata.devices() == {device}
        np.testing.assert_array_equal(np.asarray(metadata), [status, 0])
        recovered, recovered_metadata = decode(source)
        np.testing.assert_array_equal(np.asarray(recovered_metadata), [0, 0])
        _assert_bytes(recovered, raw, device)


def test_jit_dense_prefix_framing_precedence_and_retained_metadata(
    cuda_device, forbid_cpu_inflation,
):
    import jax
    import jax.numpy as jnp
    from cuda_zlib import decompress_zlib_checked

    device = jax.devices("gpu")[cuda_device]
    payload, raw = _dense_prefix_stored_stream()
    source = jax.device_put(np.frombuffer(payload, np.uint8), device)
    decode = jax.jit(lambda value: decompress_zlib_checked(value, len(raw), device))
    first, first_metadata = decode(source)
    failed, bad_metadata = decode(source.at[1].set(source[1] ^ jnp.uint8(1)))
    recovered, recovered_metadata = decode(source)
    # Retain the first status/output through later calls before any host reads.
    jax.block_until_ready((first, first_metadata, failed, bad_metadata,
                           recovered, recovered_metadata))
    del failed
    gc.collect()
    assert first.devices() == first_metadata.devices() == bad_metadata.devices() == {device}
    np.testing.assert_array_equal(np.asarray(first_metadata), [0, 0])
    np.testing.assert_array_equal(np.asarray(bad_metadata), [15, 0])
    np.testing.assert_array_equal(np.asarray(recovered_metadata), [0, 0])
    _assert_bytes(first, raw, device)
    _assert_bytes(recovered, raw, device)


@pytest.mark.parametrize("size", [65536, 65537, 262144, 1048576, 1048577])
def test_jit_checked_medium_zeros_checksum_recovery_and_retained_metadata(
    cuda_device, forbid_cpu_inflation, size,
):
    import jax
    import jax.numpy as jnp
    from cuda_zlib import decompress_zlib_checked

    device = jax.devices("gpu")[cuda_device]
    raw = bytes(size)
    source = jax.device_put(np.frombuffer(_compressed(raw), np.uint8), device)
    decode = jax.jit(lambda value: decompress_zlib_checked(value, size, device))
    first, first_metadata = decode(source)
    failed, bad_metadata = decode(source.at[-1].set(source[-1] ^ jnp.uint8(1)))
    recovered, recovered_metadata = decode(source)
    # Preserve success/error metadata through all calls before reading it.
    jax.block_until_ready((first, first_metadata, failed, bad_metadata,
                           recovered, recovered_metadata))
    del failed
    gc.collect()
    for metadata, status in ((first_metadata, 0), (bad_metadata, 21),
                              (recovered_metadata, 0)):
        assert metadata.devices() == {device}
        np.testing.assert_array_equal(np.asarray(metadata), [status, 0])
    _assert_bytes(first, raw, device)
    _assert_bytes(recovered, raw, device)


def test_jit_checked_medium_compound_precedence_and_recovery(
    cuda_device, forbid_cpu_inflation,
):
    import jax
    from cuda_zlib import decompress_zlib_checked

    device = jax.devices("gpu")[cuda_device]
    payload, raw = _medium_compound_stream()
    history_reserved, _ = _medium_compound_stream(history_error=True, reserved=True)
    window_reserved, window_raw = _medium_compound_stream(window_error=True, reserved=True)
    window, _ = _medium_compound_stream(window_error=True)
    cases = (
        # Reserved symbols take precedence over history/window emission checks.
        (history_reserved, len(raw), 4),
        (window_reserved, len(window_raw), 4),
        # The last match exceeds both the requested extent and declared window.
        (window, len(window_raw) - 1, 7),
        (payload[:-1] + bytes((payload[-1] ^ 1,)), len(raw), 21),
    )
    decode = jax.jit(
        lambda value, expected: decompress_zlib_checked(value, expected, device),
        static_argnums=1,
    )
    source = jax.device_put(np.frombuffer(payload, np.uint8), device)
    queued = [decode(source, len(raw))]
    statuses = [0]
    for malformed, expected, status in cases:
        resident = jax.device_put(np.frombuffer(malformed, np.uint8), device)
        queued.extend((decode(resident, expected), decode(source, len(raw))))
        statuses.extend((status, 0))
    # Retain every output/status until all failures and recoveries have run.
    jax.block_until_ready(queued)
    del source, resident
    gc.collect()
    for (output, metadata), status in zip(queued, statuses):
        assert output.devices() == metadata.devices() == {device}
        np.testing.assert_array_equal(np.asarray(metadata), [status, 0])
        if status == 0:
            _assert_bytes(output, raw, device)


def test_jit_checked_medium_repetition_respects_declared_window(
    cuda_device, forbid_cpu_inflation,
):
    import jax
    from cuda_zlib import decompress_zlib_checked

    device = jax.devices("gpu")[cuda_device]
    payload, raw = _medium_period_stream()
    small_window = _header(cmf=0x08) + payload[2:]
    source, limited = [jax.device_put(np.frombuffer(value, np.uint8), device)
                       for value in (payload, small_window)]
    decode = jax.jit(lambda value: decompress_zlib_checked(value, len(raw), device))
    first, first_metadata = decode(source)
    failed, window_metadata = decode(limited)
    recovered, recovered_metadata = decode(source)
    jax.block_until_ready((first, first_metadata, failed, window_metadata,
                           recovered, recovered_metadata))
    for metadata, status in ((first_metadata, 0), (window_metadata, 6),
                              (recovered_metadata, 0)):
        assert metadata.devices() == {device}
        np.testing.assert_array_equal(np.asarray(metadata), [status, 0])
    _assert_bytes(first, raw, device)
    _assert_bytes(recovered, raw, device)


@pytest.mark.parametrize("match_blocks", [257, 1050], ids=["two-rounds", "three-rounds"])
def test_jit_checked_deep_cross_block_references(
    cuda_device, forbid_cpu_inflation, match_blocks,
):
    import jax
    from cuda_zlib import decompress_zlib_checked

    device = jax.devices("gpu")[cuda_device]
    payload, raw = _deep_reference_stream(match_blocks)
    # Both exceed SmallDecode's output limit; their chains need 2/3 32-hop rounds.
    assert len(raw) > 65536
    source = jax.device_put(np.frombuffer(payload, np.uint8), device)
    decode = jax.jit(lambda value: decompress_zlib_checked(value, len(raw), device))
    result, metadata = decode(source)
    assert result.devices() == metadata.devices() == {device}
    np.testing.assert_array_equal(np.asarray(metadata), [0, 0])
    _assert_bytes(result, raw, device)


def test_jit_checked_status_gates_numeric_consumer(
    cuda_device, forbid_cpu_inflation,
):
    import jax
    import jax.numpy as jnp
    from cuda_zlib import decompress_zlib_checked

    device = jax.devices("gpu")[cuda_device]
    raw = (bytes(range(251)) * 262)[:65537]
    source = jax.device_put(np.frombuffer(_compressed(raw), np.uint8), device)

    @jax.jit
    def consume(value):
        output, metadata = decompress_zlib_checked(value, len(raw), device)
        result = jax.lax.cond(
            metadata[0] == 0,
            lambda data: jnp.sum(data.astype(jnp.int32) * 3 + 7, dtype=jnp.int32),
            lambda data: jnp.int32(-1),
            output,
        )
        return result, metadata

    wanted = int(np.sum(np.frombuffer(raw, np.uint8).astype(np.int64) * 3 + 7))
    bad = source.at[-1].set(source[-1] ^ jnp.uint8(1))
    for value, expected, status in ((source, wanted, 0), (bad, -1, 21),
                                     (source, wanted, 0)):
        result, metadata = consume(value)
        assert result.devices() == metadata.devices() == {device}
        np.testing.assert_array_equal(np.asarray(metadata), [status, 0])
        assert int(np.asarray(result)) == expected


def test_queued_checked_output_and_metadata_ownership(
    cuda_device, forbid_cpu_inflation,
):
    import jax
    import jax.numpy as jnp
    from cuda_zlib import decompress_zlib_checked, trim_workspace_pool

    device = jax.devices("gpu")[cuda_device]
    raw = random.Random(441).randbytes(65537)
    sources = [jax.device_put(np.frombuffer(_compressed(value, level=0), np.uint8),
                              device) for value in (raw, raw[::-1])]
    decode = jax.jit(lambda value: decompress_zlib_checked(value, len(raw), device))
    queued = [decode(sources[0]),
              decode(sources[1].at[-1].set(sources[1][-1] ^ jnp.uint8(1))),
              decode(sources[1]), decode(sources[0].at[0].set(jnp.uint8(0)))]
    # Submit every call before waiting or reading any output/status on the host.
    jax.block_until_ready(queued)
    first, first_metadata = queued[0]
    errors = (queued[1][1], queued[3][1])
    del queued, sources
    gc.collect()
    trim_workspace_pool(device)
    replacement = jax.device_put(
        np.frombuffer(_compressed(raw[::-1], level=0), np.uint8), device)
    later, later_metadata = decode(replacement)
    jax.block_until_ready((later, later_metadata))
    assert first.devices() == first_metadata.devices() == {device}
    assert all(metadata.devices() == {device} for metadata in errors)
    np.testing.assert_array_equal(np.asarray(first_metadata), [0, 0])
    for metadata, status in zip(errors, (21, 15)):
        np.testing.assert_array_equal(np.asarray(metadata), [status, 0])
    np.testing.assert_array_equal(np.asarray(later_metadata), [0, 0])
    _assert_bytes(first, raw, device)
    _assert_bytes(later, raw[::-1], device)


def test_native_checked_limits_and_error_recovery(cuda_device, forbid_cpu_inflation):
    import jax

    device = jax.devices("gpu")[cuda_device]
    empty, raw = _dynamic_tree_stream("single-eob")
    embedded, embedded_raw = _embedded_candidate_stream()
    many, _ = _many_dynamic_stream(empty=True)
    cases = (
        (empty, 0, 0, 256, 23),
        (empty, 0, 4096, 0, 23),
        (empty, 0, 262145, 256, 23),
        (empty, 0, 4096, 262145, 23),
        (b"", 0, 4096, 256, 14),
        (b"\x00" + empty[1:], 0, 4096, 256, 15),
        (_wrap(_stored(b"A", final=False), b"A"), 1, 4096, 256, 1),
        (embedded, len(embedded_raw), 1, 256, 18),
        (many, 0, 4096, 127, 9),
    )
    source = jax.device_put(np.frombuffer(empty, np.uint8), device)
    recover = _limited_checked_decoder(device, 0)
    retained = []
    for payload, expected, candidates, blocks, status in cases:
        decode = _limited_checked_decoder(device, expected, candidates, blocks)
        resident = jax.device_put(np.frombuffer(payload, np.uint8), device)
        failed, metadata = decode(resident)
        recovered, recovered_metadata = recover(source)
        retained.append((failed, metadata, recovered, recovered_metadata, status))
    jax.block_until_ready(retained)
    for _, metadata, recovered, recovered_metadata, status in retained:
        assert metadata.devices() == recovered_metadata.devices() == {device}
        np.testing.assert_array_equal(np.asarray(metadata), [status, 0])
        np.testing.assert_array_equal(np.asarray(recovered_metadata), [0, 0])
        _assert_bytes(recovered, raw, device)


def test_native_checked_fixed_summary_fallback_and_recovery(
    cuda_device, forbid_cpu_inflation,
):
    import jax

    device = jax.devices("gpu")[cuda_device]
    payload, raw = _fixed_summary_stream()
    invalid, _ = _fixed_summary_stream(invalid=True)
    decode = _limited_checked_decoder(device, len(raw))
    source, malformed = [jax.device_put(np.frombuffer(value, np.uint8), device)
                         for value in (payload, invalid)]
    # The fixed block follows an unaligned dynamic boundary without a stored seed.
    first, metadata = decode(source)
    failed, error = decode(malformed)
    recovered, recovered_metadata = decode(source)
    jax.block_until_ready((first, metadata, failed, error, recovered, recovered_metadata))
    for value, status in ((metadata, 0), (error, 4), (recovered_metadata, 0)):
        assert value.devices() == {device}
        np.testing.assert_array_equal(np.asarray(value), [status, 0])
    _assert_bytes(first, raw, device)
    _assert_bytes(recovered, raw, device)


def test_native_checked_dense_fallback_discards_earlier_seeds(
    cuda_device, forbid_cpu_inflation,
):
    import jax
    import jax.numpy as jnp

    device = jax.devices("gpu")[cuda_device]
    earlier, earlier_raw = _many_dynamic_stream(empty=False)
    dense, dense_raw = _dense_prefix_stored_stream()
    prime = _limited_checked_decoder(device, len(earlier_raw))
    decode = _limited_checked_decoder(device, len(dense_raw))
    previous, previous_metadata = prime(jax.device_put(np.frombuffer(earlier, np.uint8), device))
    source = jax.device_put(np.frombuffer(dense, np.uint8), device)
    first, metadata = decode(source)
    failed, error = decode(source.at[-1].set(source[-1] ^ jnp.uint8(1)))
    recovered, recovered_metadata = decode(source)
    jax.block_until_ready((previous, previous_metadata, first, metadata, failed,
                           error, recovered, recovered_metadata))
    for value, status in ((previous_metadata, 0), (metadata, 0), (error, 21),
                          (recovered_metadata, 0)):
        assert value.devices() == {device}
        np.testing.assert_array_equal(np.asarray(value), [status, 0])
    _assert_bytes(previous, earlier_raw, device)
    _assert_bytes(first, dense_raw, device)
    _assert_bytes(recovered, dense_raw, device)
