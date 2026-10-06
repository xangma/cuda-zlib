# Copyright (c) 2026 xangma
# SPDX-License-Identifier: MIT

"""Long-block and entropy regressions, with independent RFC 1951 fixtures."""

import contextlib
import functools
import random
import threading
import time
import zlib
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

import cuda_zlib as codec
from test_decode import (
    _assert_bytes, _codes, _compressed, _dynamic_header, _dynamic_literal_fields,
    _header, _stored, _wrap, cuda_device,
)


_MIB = 1 << 20
_FIXED = _codes([8] * 144 + [9] * 112 + [7] * 24 + [8] * 8)
_DISTANCE = _codes([5] * 32)
_LENGTH_BASE = (3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 17, 19, 23,
                27, 31, 35, 43, 51, 59, 67, 83, 99, 115, 131, 163,
                195, 227, 258)
_LENGTH_EXTRA = (0,) * 8 + (1,) * 4 + (2,) * 4 + (3,) * 4 + (4,) * 4 + (5,) * 4 + (0,)
_DISTANCE_BASE = (1, 2, 3, 4, 5, 7, 9, 13, 17, 25, 33, 49, 65, 97,
                  129, 193, 257, 385, 513, 769, 1025, 1537, 2049,
                  3073, 4097, 6145, 8193, 12289, 16385, 24577)
_DISTANCE_EXTRA = (0,) * 4 + tuple(width for width in range(1, 14) for _ in range(2))


class _Writer:
    """Pack bits continuously; block boundaries never receive implicit padding."""

    def __init__(self):
        self.data = bytearray()
        self.pending = self.width = self.bits = 0

    def put(self, value, width):
        assert 0 <= value < (1 << width) if width else value == 0
        self.pending |= value << self.width
        self.width += width
        self.bits += width
        while self.width >= 8:
            self.data.append(self.pending & 255)
            self.pending >>= 8
            self.width -= 8

    def fields(self, fields):
        for value, width in fields:
            self.put(value, width)

    def aligned(self, data):
        assert not self.width
        self.data.extend(data)
        self.bits += len(data) * 8

    def finish(self):
        return bytes(self.data) + (bytes((self.pending,)) if self.width else b"")


def _literals(writer, raw, codes=_FIXED):
    for value in raw:
        writer.put(*codes[value])


def _match(writer, output, length, distance,
           literal_codes=_FIXED, distance_codes=_DISTANCE):
    assert 3 <= length <= 258 and 1 <= distance <= min(32768, len(output))
    length_symbol = max(index for index, base in enumerate(_LENGTH_BASE)
                        if base <= length)
    distance_symbol = max(index for index, base in enumerate(_DISTANCE_BASE)
                          if base <= distance)
    writer.put(*literal_codes[257 + length_symbol])
    writer.put(length - _LENGTH_BASE[length_symbol], _LENGTH_EXTRA[length_symbol])
    writer.put(*distance_codes[distance_symbol])
    writer.put(distance - _DISTANCE_BASE[distance_symbol],
               _DISTANCE_EXTRA[distance_symbol])
    # The independent oracle repeats the preceding history for overlapping
    # copies, including distance 1; it does not use the candidate's root model.
    history = bytes(output[-distance:])
    output.extend((history * ((length + distance - 1) // distance))[:length])


@functools.lru_cache(maxsize=None)
def _long_fixed(tail="valid"):
    raw = bytes(range(256)) * (_MIB // 256)
    writer = _Writer()
    writer.put(3, 3)  # BFINAL=1, BTYPE=01.
    _literals(writer, raw)
    if tail == "reserved-symbol":
        writer.put(*_FIXED[286])
    elif tail == "reserved-distance":
        writer.put(*_FIXED[257])  # Length 3, then reserved distance code 30.
        writer.put(*_DISTANCE[30])
    else:
        assert tail == "valid"
        writer.put(*_FIXED[256])
    return _wrap(writer.finish(), raw), raw


@functools.lru_cache(maxsize=None)
def _chained_fixed(kind, alignment=0):
    writer = _Writer()
    if kind == "stored":
        prefix = random.Random(142).randbytes(32768)
        writer.aligned(_stored(prefix, final=False))
    else:
        assert kind == "dynamic"
        # This literal-only dynamic tree consumes 330 + len(prefix) bits.
        # Adjust the prefix to place the fixed header at every bit alignment.
        prefix = b"A" * (32768 + (alignment - 330) % 8)
        writer.fields(_dynamic_literal_fields(prefix, final=False))
    assert writer.bits % 8 == alignment
    output = bytearray(prefix)
    writer.put(3, 3)
    _match(writer, output, 258, 32768)  # Cross the preceding block boundary.
    literals = _long_fixed()[1]
    _literals(writer, literals)
    output.extend(literals)
    _match(writer, output, 258, 1)
    _match(writer, output, 258, 32768)
    writer.put(*_FIXED[256])
    raw = bytes(output)
    return _wrap(writer.finish(), raw), raw


@functools.lru_cache(maxsize=1)
def _window_matches():
    # A match starts one byte before 32 KiB, then successive 258-byte copies
    # cross many output partitions, using overlap and maximum-window history.
    seed = random.Random(897).randbytes(32767)
    writer, output = _Writer(), bytearray(seed)
    writer.put(3, 3)
    _literals(writer, seed)
    distances = (1, 2, 3, 257, 258, 32767, 32768)
    for index in range(1024):
        _match(writer, output, 258, distances[index % len(distances)])
    writer.put(*_FIXED[256])
    raw = bytes(output)
    return _wrap(writer.finish(), raw), raw


_SEED_BOUNDARIES = [
    (distance, length)
    for distance in (2, 3, 7, 31, 257)
    for length in sorted({distance - 1, distance, distance + 1, 258})
    if 3 <= length <= 258
]


def _seed_match_stream(kind, distance=3, length=258, value=ord("R")):
    writer = _Writer()
    if kind == "referenced":
        prefix = b"pq"
        assert distance == 3
    elif kind == "mixed":
        # The seed ends with a local literal; wrapping returns to a root in
        # the preceding block. The 257-byte seed includes every byte value.
        prefix = (b"pq" if distance == 3 else
                  bytes((19 + 41 * index) & 255 for index in range(distance - 1)))
    elif kind == "external":
        assert distance == 1
        prefix = bytes((value,))
    else:
        assert kind == "local" and distance == 1
        prefix = b""
    if prefix:
        writer.aligned(_stored(prefix, final=False))
    output = bytearray(prefix)
    writer.put(3, 3)
    if kind == "referenced":
        # These three local roots are references into the preceding block,
        # rather than tagged literals. The next match must preserve them.
        _match(writer, output, 3, 2)
    elif kind != "external":
        _literals(writer, bytes((value,)))
        output.append(value)
    _match(writer, output, length, distance)
    writer.put(*_FIXED[256])
    raw = bytes(output)
    return _wrap(writer.finish(), raw), raw


def _match_without_history():
    writer = _Writer()
    writer.put(3, 3)
    writer.put(*_FIXED[257])  # Length 3 with distance 1 and no preceding byte.
    writer.put(*_DISTANCE[0])
    writer.put(*_FIXED[256])
    return _wrap(writer.finish(), b"\x00" * 3)


def _match_block_header(writer, kind, final=True):
    if kind == "fixed":
        writer.put(2 | int(final), 3)
        return _FIXED, _DISTANCE
    assert kind == "dynamic"
    # Complete trees covering every literal and usable length/distance.
    literal_lengths = [8] * 226 + [9] * 60
    distance_lengths = [4] * 2 + [5] * 28
    code_lengths = [0] * 19
    for symbol in (4, 5, 8, 9):
        code_lengths[symbol] = 2
    writer.fields(_dynamic_header(286, 30, code_lengths, final=final))
    length_codes = _codes(code_lengths)
    writer.fields(length_codes[width]
                  for width in literal_lengths + distance_lengths)
    return _codes(literal_lengths), _codes(distance_lengths)


def _warp_match_stream(kind, tail="valid", block_size=0, following=None):
    writer = _Writer()
    prefix = bytes(range(256)) * 128
    writer.aligned(_stored(prefix, final=False))
    output = bytearray(prefix)
    literal_codes, distance_codes = _match_block_header(
        writer, kind, final=following is None)

    def match(length, distance):
        _match(writer, output, length, distance, literal_codes, distance_codes)

    # First copy external roots, then immediately reuse roots written by that
    # cooperative copy. Later literal runs exercise leader-to-warp ordering.
    match(258, 32768)
    match(3, 2)
    match(258, 3)
    match(258, 1)
    lengths = (3, 7, 31, 32, 33, 63, 65, 127, 255, 257, 258)
    distances = (1, 2, 3, 7, 31, 32, 33, 257, 258, 32768)
    for index, length in enumerate(lengths * 2):
        literals = bytes((19 * index + 13 * j) & 255
                         for j in range(1 + (7 * index) % 35)) + b"\x00\x7f\xff"
        _literals(writer, literals, literal_codes)
        output.extend(literals)
        match(length, distances[index % len(distances)])
        # Non-warp tails and repeated consumption of newly written roots.
        match(32 + index % 3, 1 + index % 3)
    while len(output) - len(prefix) < block_size:
        length = min(258, block_size - (len(output) - len(prefix)))
        if length >= 3:
            match(length, 1)
        else:
            literals = bytes((output[-1],)) * length
            _literals(writer, literals, literal_codes)
            output.extend(literals)
    if tail == "valid":
        writer.put(*literal_codes[256])
    elif tail == "reserved-distance":
        assert kind == "fixed"
        writer.put(*literal_codes[257])
        writer.put(*distance_codes[30])
        writer.put(*literal_codes[256])
    else:
        assert tail == "missing-eob"
        if kind == "fixed" and writer.bits % 8 == 1:
            # Seven zero padding bits would accidentally encode a fixed EOB.
            _literals(writer, b"\xff", literal_codes)
            output.append(255)
    if following is not None:
        assert tail == "valid"
        literal_codes, distance_codes = _match_block_header(writer, following)
        _match(writer, output, 3, 1, literal_codes, distance_codes)
        writer.put(*literal_codes[256])
    raw = bytes(output)
    return _wrap(writer.finish(), raw), raw


def _warp_history_stream(kind, invalid=False, block_size=262):
    writer = _Writer()
    literal_codes, distance_codes = _match_block_header(writer, kind)
    output = bytearray(b"\xff")
    _literals(writer, output, literal_codes)
    _match(writer, output, 258, 1, literal_codes, distance_codes)
    if invalid:
        # Metadata can validate this token's size/window, but only emission
        # knows that 259 bytes of history cannot supply distance 32768.
        writer.put(*literal_codes[257])
        writer.put(*distance_codes[29])
        writer.put(8191, 13)
        output.extend(b"\xff" * 3)  # Exact declared size, not a valid copy.
    else:
        _match(writer, output, 3, 1, literal_codes, distance_codes)
    while len(output) < block_size:
        length = min(258, block_size - len(output))
        if length >= 3:
            _match(writer, output, length, 1, literal_codes, distance_codes)
        else:
            literals = b"\xff" * length
            _literals(writer, literals, literal_codes)
            output.extend(literals)
    writer.put(*literal_codes[256])
    raw = bytes(output)
    return _wrap(writer.finish(), raw), raw


@contextlib.contextmanager
def _no_cpu_codec(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("candidate call used a CPU zlib codec")

    with monkeypatch.context() as patch:
        for name in ("compress", "compressobj", "decompress", "decompressobj"):
            patch.setattr(zlib, name, forbidden)
        yield


def _inflate_exact(payload):
    decoder = zlib.decompressobj()
    raw = decoder.decompress(payload) + decoder.flush()
    assert decoder.eof and not decoder.unused_data and not decoder.unconsumed_tail
    return raw


def test_long_fixed_fixture_oracles():
    payload, raw = _long_fixed()
    assert (payload[2] >> 1) & 3 == 1
    assert len(raw) == _MIB and _inflate_exact(payload) == raw
    # Invalid tokens occur after the entire long literal run, not at admission.
    for tail in ("reserved-symbol", "reserved-distance"):
        with pytest.raises(zlib.error):
            _inflate_exact(_long_fixed(tail)[0])
    payload, raw = _window_matches()
    assert _inflate_exact(payload) == raw


def test_bit_aligned_chain_fixture_oracles():
    for kind, alignment in [("stored", 0)] + [("dynamic", n) for n in range(8)]:
        payload, raw = _chained_fixed(kind, alignment)
        assert _inflate_exact(payload) == raw


def test_overlap_seed_fixture_oracles():
    assert _seed_match_stream("mixed")[1] == b"pqR" * 87
    assert _seed_match_stream("referenced")[1] == b"pqpqp" + b"pqp" * 86
    cases = [_seed_match_stream("mixed", distance, length)
             for distance, length in _SEED_BOUNDARIES]
    cases.append(_seed_match_stream("referenced"))
    cases.extend(_seed_match_stream(kind, 1, length, value)
                 for kind in ("local", "external")
                 for value in (0x00, 0x7f, 0xff)
                 for length in (3, 258))
    for payload, raw in cases:
        assert _inflate_exact(payload) == raw
    with pytest.raises(zlib.error):
        _inflate_exact(_match_without_history())


@pytest.mark.parametrize("distance,length", _SEED_BOUNDARIES)
def test_overlap_seed_crosses_block_prefix_and_wrap(
    cuda_device, monkeypatch, distance, length,
):
    payload, raw = _seed_match_stream("mixed", distance, length)
    assert _inflate_exact(payload) == raw
    with _no_cpu_codec(monkeypatch):
        result = codec.decompress_zlib(payload, len(raw), cuda_device)
    _assert_bytes(result, raw, cuda_device)


def test_overlap_local_seed_contains_prior_block_references(cuda_device, monkeypatch):
    payload, raw = _seed_match_stream("referenced")
    assert _inflate_exact(payload) == raw
    with _no_cpu_codec(monkeypatch):
        result = codec.decompress_zlib(payload, len(raw), cuda_device)
    _assert_bytes(result, raw, cuda_device)


@pytest.mark.parametrize("kind", ["local", "external"])
@pytest.mark.parametrize("value", [0x00, 0x7f, 0xff])
@pytest.mark.parametrize("length", [3, 258])
def test_distance_one_seed_preserves_literal_tags(
    cuda_device, monkeypatch, kind, value, length,
):
    payload, raw = _seed_match_stream(kind, 1, length, value)
    assert _inflate_exact(payload) == raw
    with _no_cpu_codec(monkeypatch):
        result = codec.decompress_zlib(payload, len(raw), cuda_device)
    _assert_bytes(result, raw, cuda_device)


def test_distance_one_without_history_fails_then_recovers(cuda_device, monkeypatch):
    invalid = _match_without_history()
    payload, raw = _seed_match_stream("local", 1, 258, 0xff)
    with pytest.raises(zlib.error):
        _inflate_exact(invalid)
    assert _inflate_exact(payload) == raw
    with _no_cpu_codec(monkeypatch):
        with pytest.raises(codec.CodecError):
            codec.decompress_zlib(invalid, 3, cuda_device)
        recovered = codec.decompress_zlib(payload, len(raw), cuda_device)
    _assert_bytes(recovered, raw, cuda_device)


def test_warp_match_fixture_oracles():
    for kind in ("fixed", "dynamic"):
        payload, raw = _warp_match_stream(kind)
        assert _inflate_exact(payload) == raw
        with pytest.raises(zlib.error):
            zlib.decompress(_warp_match_stream(kind, "missing-eob")[0])
        payload, raw = _warp_history_stream(kind)
        assert raw == b"\xff" * 262 and _inflate_exact(payload) == raw
        with pytest.raises(zlib.error):
            zlib.decompress(_warp_history_stream(kind, invalid=True)[0])
    with pytest.raises(zlib.error):
        zlib.decompress(_warp_match_stream("fixed", "reserved-distance")[0])
    for size in (_MIB, _MIB + 1):
        payload, raw = _warp_match_stream("dynamic", block_size=size)
        assert len(raw) == 32768 + size and _inflate_exact(payload) == raw
    payload, raw = _warp_history_stream("dynamic", block_size=_MIB + 1)
    assert len(raw) == _MIB + 1 and _inflate_exact(payload) == raw
    with pytest.raises(zlib.error):
        zlib.decompress(_warp_history_stream(
            "dynamic", invalid=True, block_size=_MIB + 1)[0])
    for kind in ("fixed", "dynamic"):
        payload, raw = _warp_match_stream(
            "dynamic", block_size=_MIB + 1, following=kind)
        assert len(raw) == 32768 + _MIB + 4 and _inflate_exact(payload) == raw


@pytest.mark.parametrize("kind", ["fixed", "dynamic"])
def test_mixed_literal_match_order_and_warp_tails(cuda_device, monkeypatch, kind):
    payload, raw = _warp_match_stream(kind)
    assert _inflate_exact(payload) == raw
    with _no_cpu_codec(monkeypatch):
        result = codec.decompress_zlib(payload, len(raw), cuda_device)
    _assert_bytes(result, raw, cuda_device)


@pytest.mark.parametrize("kind,tail", [
    ("fixed", "missing-eob"), ("dynamic", "missing-eob"),
    ("fixed", "reserved-distance"),
])
def test_error_after_cooperative_match_then_fresh_recovery(
    cuda_device, monkeypatch, kind, tail,
):
    invalid, invalid_raw = _warp_match_stream(kind, tail)
    payload, raw = _warp_match_stream(kind)
    with pytest.raises(zlib.error):
        zlib.decompress(invalid)
    assert _inflate_exact(payload) == raw
    with _no_cpu_codec(monkeypatch):
        with pytest.raises(codec.CodecError):
            codec.decompress_zlib(invalid, len(invalid_raw), cuda_device)
        recovered = codec.decompress_zlib(payload, len(raw), cuda_device)
    _assert_bytes(recovered, raw, cuda_device)


@pytest.mark.parametrize("kind", ["fixed", "dynamic"])
def test_bad_history_after_cooperative_match_status_then_recovery(
    cuda_device, monkeypatch, kind,
):
    invalid, invalid_raw = _warp_history_stream(kind, invalid=True)
    payload, raw = _warp_history_stream(kind)
    with pytest.raises(zlib.error):
        zlib.decompress(invalid)
    assert _inflate_exact(payload) == raw
    with _no_cpu_codec(monkeypatch):
        _, bad_metadata = codec.decompress_zlib_checked(
            invalid, len(invalid_raw), cuda_device)
        np.testing.assert_array_equal(np.asarray(bad_metadata), [6, 0])
        recovered, metadata = codec.decompress_zlib_checked(
            payload, len(raw), cuda_device)
        np.testing.assert_array_equal(np.asarray(metadata), [0, 0])
    _assert_bytes(recovered, raw, cuda_device)


@pytest.mark.parametrize("block_size", [_MIB, _MIB + 1])
def test_large_dynamic_matches_on_both_emitter_paths(
    cuda_device, monkeypatch, block_size,
):
    # One accepted dynamic block spans the dispatch boundary. Its leading
    # copies combine prior stored-block roots, local seeds, and warp tails.
    payload, raw = _warp_match_stream("dynamic", block_size=block_size)
    assert len(raw) == 32768 + block_size and _inflate_exact(payload) == raw
    with _no_cpu_codec(monkeypatch):
        result = codec.decompress_zlib(payload, len(raw), cuda_device)
    _assert_bytes(result, raw, cuda_device)


def test_large_warp_history_error_status_then_recovery(cuda_device, monkeypatch):
    # The later tokens make metadata select the large-block emitter, but it
    # must reject distance 32768 after only 259 bytes of valid local history.
    invalid, invalid_raw = _warp_history_stream(
        "dynamic", invalid=True, block_size=_MIB + 1)
    payload, raw = _warp_history_stream("dynamic", block_size=_MIB + 1)
    with pytest.raises(zlib.error):
        zlib.decompress(invalid)
    assert _inflate_exact(payload) == raw
    with _no_cpu_codec(monkeypatch):
        _, bad_metadata = codec.decompress_zlib_checked(
            invalid, len(invalid_raw), cuda_device)
        np.testing.assert_array_equal(np.asarray(bad_metadata), [6, 0])
        recovered, metadata = codec.decompress_zlib_checked(
            payload, len(raw), cuda_device)
        np.testing.assert_array_equal(np.asarray(metadata), [0, 0])
    _assert_bytes(recovered, raw, cuda_device)


@pytest.mark.parametrize("following", ["fixed", "dynamic"])
def test_small_block_copies_large_block_before_its_emission(
    cuda_device, monkeypatch, following,
):
    # The serial emitter runs first, so it must retain an external index to
    # the large predecessor rather than read roots the warp has not written.
    payload, raw = _warp_match_stream(
        "dynamic", block_size=_MIB + 1, following=following)
    assert len(raw) == 32768 + _MIB + 4 and _inflate_exact(payload) == raw
    with _no_cpu_codec(monkeypatch):
        result = codec.decompress_zlib(payload, len(raw), cuda_device)
    _assert_bytes(result, raw, cuda_device)


@pytest.mark.parametrize("size", [_MIB, 4 * _MIB], ids=["1MiB", "4MiB"])
@pytest.mark.parametrize("kind", ["stored", "fixed", "dynamic"])
def test_external_large_blocks(cuda_device, monkeypatch, size, kind):
    if kind == "stored":
        raw = random.Random(109).randbytes(size)
        payload, block_type = _compressed(raw, level=0), 0
    elif kind == "fixed":
        raw = (bytes(range(256)) * ((size + 255) // 256))[:size]
        payload = _compressed(raw, strategy=zlib.Z_FIXED)
        block_type = 1
    else:
        raw = bytes(value | 128 for value in random.Random(19).randbytes(size))
        payload = _compressed(raw, strategy=zlib.Z_HUFFMAN_ONLY)
        block_type = 2
    assert (payload[2] >> 1) & 3 == block_type
    assert _inflate_exact(payload) == raw
    with _no_cpu_codec(monkeypatch):
        result = codec.decompress_zlib(payload, size, cuda_device)
    _assert_bytes(result, raw, cuda_device)


def test_true_single_long_fixed_literal_block(cuda_device, monkeypatch):
    payload, raw = _long_fixed()
    with _no_cpu_codec(monkeypatch):
        result = codec.decompress_zlib(payload, len(raw), cuda_device)
    _assert_bytes(result, raw, cuda_device)


@pytest.mark.parametrize("kind,alignment", [("stored", 0)] +
                         [("dynamic", n) for n in range(8)])
def test_long_fixed_after_stored_or_dynamic(cuda_device, monkeypatch, kind, alignment):
    payload, raw = _chained_fixed(kind, alignment)
    with _no_cpu_codec(monkeypatch):
        result = codec.decompress_zlib(payload, len(raw), cuda_device)
    _assert_bytes(result, raw, cuda_device)


def test_fixed_overlap_and_full_window_across_output_partitions(cuda_device, monkeypatch):
    payload, raw = _window_matches()
    with _no_cpu_codec(monkeypatch):
        result = codec.decompress_zlib(payload, len(raw), cuda_device)
    _assert_bytes(result, raw, cuda_device)


@pytest.mark.parametrize("change", [
    "too-small", "too-large", "adler", "truncated-body", "missing-eob",
    "reserved-symbol", "reserved-distance", "extra-deflate-byte",
])
def test_long_fixed_failure_then_fresh_recovery(cuda_device, monkeypatch, change):
    valid, raw = _long_fixed()
    payload, expected = valid, len(raw)
    if change == "too-small":
        expected -= 1
    elif change == "too-large":
        expected += 1
    elif change == "adler":
        payload = payload[:-1] + bytes((payload[-1] ^ 1,))
    elif change == "truncated-body":
        payload = payload[:-11] + payload[-4:]
    elif change == "missing-eob":
        # The last body byte contains the final two EOB bits.
        payload = payload[:-5] + payload[-4:]
    elif change in ("reserved-symbol", "reserved-distance"):
        payload = _long_fixed(change)[0]
    else:
        payload = payload[:-4] + b"\x00" + payload[-4:]
    with _no_cpu_codec(monkeypatch):
        with pytest.raises(codec.CodecError):
            codec.decompress_zlib(payload, expected, cuda_device)
        recovered = codec.decompress_zlib(valid, len(raw), cuda_device)
    _assert_bytes(recovered, raw, cuda_device)


def test_long_fixed_declared_window_failure_and_recovery(cuda_device, monkeypatch):
    valid, raw = _window_matches()
    # CINFO=6 advertises 16 KiB, smaller than this stream's 32 KiB distances.
    small_window = _header(cmf=0x68) + valid[2:]
    with _no_cpu_codec(monkeypatch):
        with pytest.raises(codec.CodecError):
            codec.decompress_zlib(small_window, len(raw), cuda_device)
        recovered = codec.decompress_zlib(valid, len(raw), cuda_device)
    _assert_bytes(recovered, raw, cuda_device)


@functools.lru_cache(maxsize=None)
def _encoder_input(kind):
    rng = random.Random(589)
    if kind == "high-literals":
        # Uniform over high 128 symbols: about 7 bits of literal entropy,
        # few repeated triples, and mostly 9-bit fixed Huffman codes.
        return bytes(value | 128 for value in rng.randbytes(_MIB + 257))
    if kind == "skewed":
        symbols = list(range(144, 256))
        weights = [8192, 4096, 2048, 1024, 512, 256, 128, 64] + [1] * 104
        return bytes(rng.choices(symbols, weights=weights, k=512 * 1024 + 137))
    if kind == "rare-symbols":
        # Include singleton symbols beside long overlap runs and chunk seams.
        return b"\xff" * 65534 + bytes(range(256)) + b"\xfe" * 196609
    assert kind == "collision-prefixes"
    # Many equal three-byte prefixes with unrelated suffixes stress replacing
    # match candidates without assuming a particular hash implementation.
    return b"".join(b"ABC" + rng.randbytes(29) for _ in range(16384)) + b"ABCend"


def test_entropy_fixture_has_dynamic_compression_headroom():
    raw = _encoder_input("high-literals")
    payload = _compressed(raw, strategy=zlib.Z_HUFFMAN_ONLY)
    assert (payload[2] >> 1) & 3 == 2
    assert len(payload) < len(raw) * 0.95
    assert _inflate_exact(payload) == raw


@pytest.mark.parametrize("chunk", [32768, 65535])
@pytest.mark.parametrize("kind", ["high-literals", "skewed", "rare-symbols",
                                  "collision-prefixes"])
def test_encoder_entropy_skew_and_colliding_prefixes(cuda_device, monkeypatch, kind, chunk):
    raw = _encoder_input(kind)
    with _no_cpu_codec(monkeypatch):
        encoded = codec.compress_zlib(raw, cuda_device, chunk_bytes=chunk)
        decoded = codec.decompress_zlib(encoded, len(raw), cuda_device)
        if kind == "high-literals":
            repeated = codec.compress_zlib(raw, cuda_device, chunk_bytes=chunk)
    payload = np.asarray(encoded).tobytes()
    assert _inflate_exact(payload) == raw
    _assert_bytes(decoded, raw, cuda_device)
    if kind == "high-literals":
        assert np.asarray(repeated).tobytes() == payload
        assert (payload[2] >> 1) & 3 == 2, "biased literals need dynamic Huffman coding"
        assert len(payload) < len(raw) * 0.95


def test_native_library_cache_serializes_concurrent_misses(monkeypatch):
    """CPU fake exercises the real loader lock without JAX or CUDA imports."""
    from cuda_zlib import _ffi
    constructions = []

    @functools.lru_cache(maxsize=None)
    def fake_cached(architecture):
        # A bare lru_cache permits duplicate constructions during concurrent misses.
        time.sleep(0.01)
        constructions.append(architecture)
        return object(), (architecture + "_encode", architecture + "_decode")

    barrier = threading.Barrier(8)
    def concurrent_miss(_):
        barrier.wait(timeout=5)
        return _ffi._load_library("sm_86")

    monkeypatch.setattr(_ffi, "_load_library_cached", fake_cached)
    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(concurrent_miss, range(8)))
    first = results[0]
    assert all(result is first for result in results)
    assert constructions == ["sm_86"]
    other = _ffi._load_library("sm_89")
    assert other is not first
    assert _ffi._load_library("sm_86") is first
    assert _ffi._load_library("sm_89") is other
    assert constructions == ["sm_86", "sm_89"]


def test_jit_padded_compression_preserves_stream_and_dependencies(cuda_device, monkeypatch):
    import jax
    import jax.numpy as jnp
    device = jax.devices("gpu")[cuda_device]
    host = np.arange(32769, dtype=np.uint32).astype(np.uint8)
    source = jax.device_put(host, device)
    encode = jax.jit(lambda value: codec.compress_zlib_padded(
        (value.astype(jnp.uint16) * 17 + 3).astype(jnp.uint8), device))
    with _no_cpu_codec(monkeypatch):
        output, metadata = encode(source)
        output.block_until_ready()
        metadata.block_until_ready()
    length, status = map(int, np.asarray(metadata))
    assert status == 0 and 8 <= length <= output.size
    assert output.devices() == metadata.devices() == {device}
    buffer = np.asarray(output)
    raw = ((host.astype(np.uint16) * 17 + 3) % 256).astype(np.uint8).tobytes()
    assert _inflate_exact(buffer[:length].tobytes()) == raw
    assert not np.any(buffer[length:]), "padded output must not expose stale scratch bytes"


def test_jit_checked_decode_reports_device_errors_and_recovers(cuda_device, monkeypatch):
    import jax
    import jax.numpy as jnp
    device = jax.devices("gpu")[cuda_device]
    raw = b"native JAX checked status" * 512
    payload = zlib.compress(raw)
    source = jax.device_put(np.frombuffer(payload, dtype=np.uint8), device)
    decode = jax.jit(lambda value: codec.decompress_zlib_checked(value, len(raw), device))
    with _no_cpu_codec(monkeypatch):
        result, metadata = decode(source)
        assert int(np.asarray(metadata)[0]) == 0
        _assert_bytes(result, raw, device)
        malformed = source.at[-1].set(source[-1] ^ jnp.uint8(1))
        _, bad_metadata = decode(malformed)
        assert int(np.asarray(bad_metadata)[0]) == 21
        bad_header = source.at[0].set(jnp.uint8(0))
        _, header_metadata = decode(bad_header)
        assert int(np.asarray(header_metadata)[0]) == 15
        recovered, recovered_metadata = decode(source)
        assert int(np.asarray(recovered_metadata)[0]) == 0
        _assert_bytes(recovered, raw, device)
    assert metadata.devices() == bad_metadata.devices() == {device}


@pytest.mark.parametrize("operation", ["compress", "decompress"])
def test_jit_explicit_device_rejects_other_device_input(cuda_device, operation):
    import jax
    devices = jax.devices("gpu")
    if len(devices) < 2:
        pytest.skip("two CUDA devices required for traced device admission")
    device = devices[cuda_device]
    other = devices[1 if cuda_device == 0 else 0]
    raw = b"explicit FFI CUDA device" * 32
    if operation == "compress":
        payload = raw
        run = jax.jit(lambda value: codec.compress_zlib_padded(value, device))
    else:
        payload = zlib.compress(raw)
        run = jax.jit(lambda value: codec.decompress_zlib_checked(value, len(raw), device))
    source = jax.device_put(np.frombuffer(payload, dtype=np.uint8), other)
    with pytest.raises(ValueError, match="device|incompatible"):
        result, metadata = run(source)
        result.block_until_ready()
        metadata.block_until_ready()
