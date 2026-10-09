# Copyright (c) 2026 xangma
# SPDX-License-Identifier: MIT
"""Independent Deflate fixture writer. CPU codecs are used only as oracles."""
import struct
import random
import zlib


def codes(lengths):
    counts = [lengths.count(width) for width in range(16)]
    next_code = [0] * 16
    value = 0
    for width in range(1, 16):
        value = (value + (counts[width - 1] if width > 1 else 0)) << 1
        next_code[width] = value
    result = {}
    for symbol, width in enumerate(lengths):
        if width:
            bit_code = next_code[width]
            next_code[width] += 1
            reversed_code = int(f"{bit_code:0{width}b}"[::-1], 2)
            result[symbol] = reversed_code, width
    return result


FIXED_LENGTHS = [8] * 144 + [9] * 112 + [7] * 24 + [8] * 8
FIXED_CODES = codes(FIXED_LENGTHS)


def bits(fields):
    result = bytearray()
    value = held = 0
    for number, width in fields:
        assert 0 <= number < 1 << width
        value |= number << held
        held += width
        while held >= 8:
            result.append(value & 255)
            value >>= 8
            held -= 8
    if held:
        result.append(value)
    return bytes(result)


def wrap(body, raw):
    return b"\x78\x9c" + body + struct.pack(">I", zlib.adler32(raw))


def fixed(raw, prefix=b"", ending="eob"):
    """Optional nonfinal fixed block preserves an unaligned next header."""
    fields = []
    if prefix:
        fields += [(0, 1), (1, 2)] + [FIXED_CODES[value] for value in prefix]
        fields += [FIXED_CODES[256]]
    fields += [(1, 1), (1, 2)] + [FIXED_CODES[value] for value in raw]
    if ending == "eob":
        fields += [FIXED_CODES[256]]
    elif ending == "reserved-ll":
        fields += [FIXED_CODES[286]]
    elif ending == "reserved-distance":
        fields += [FIXED_CODES[257], (15, 5)]  # reversed fixed distance30
    else:
        assert ending == "truncated"  # Five padding bits cannot finish EOB7.
    return wrap(bits(fields), prefix + raw), prefix + raw


def dynamic_fallback(raw):
    """Complete LL tree: literal255 has10 bits and its primary prefix is zero."""
    lengths = [8] * 255 + [10, 9, 10]
    cl_lengths = [0] * 19
    for symbol in (1, 8, 9, 10):
        cl_lengths[symbol] = 2
    order = [16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15]
    fields = [(1, 1), (2, 2), (1, 5), (0, 5), (14, 4)]
    fields += [(cl_lengths[symbol], 3) for symbol in order[:18]]
    cl = codes(cl_lengths)
    fields += [cl[width] for width in lengths + [1]]
    table = codes(lengths)
    fields += [table[value] for value in raw] + [table[256]]
    return wrap(bits(fields), raw), raw


def dynamic_full_window(size):
    """A canonical long literal can refill a full 64-bit literal reservoir."""
    lengths = [7] + [8] * 252 + [9] * 2 + [10, 9, 10]
    cl_lengths = [0] * 19
    for symbol in (1, 7, 8):
        cl_lengths[symbol] = 2
    for symbol in (9, 10):
        cl_lengths[symbol] = 3
    order = [16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15]
    fields = [(1, 1), (2, 2), (1, 5), (0, 5), (14, 4)]
    fields += [(cl_lengths[symbol], 3) for symbol in order[:18]]
    cl = codes(cl_lengths)
    fields += [cl[width] for width in lengths + [1]]
    randomizer = random.Random(20261009)
    for _ in range(6):
        pattern = bytes(randomizer.choice((0, 1, 253, 255)) for _ in range(256))
    raw = (pattern * ((size + 255) // 256))[:size]
    table = codes(lengths)
    fields += [table[value] for value in raw] + [table[256]]
    return wrap(bits(fields), raw), raw


def fixed_repeat(raw):
    """Literal prefix, then length3/distance1 and EOB; independent token writer."""
    assert raw
    expected = raw + raw[-1:] * 3
    fields = [(1, 1), (1, 2)] + [FIXED_CODES[value] for value in raw]
    fields += [FIXED_CODES[257], (0, 5), FIXED_CODES[256]]
    return wrap(bits(fields), expected), expected


def literal_payload(size, high=False):
    alphabet = range(144, 256) if high else range(1, 144)
    alphabet = bytes(alphabet)
    return bytes(alphabet[index % len(alphabet)] for index in range(size))


def fixture_checks():
    """No GPU imports; validate the same real fixtures used by prepared tests."""
    checks = []
    for high in (False, True):
        for size in (0, 1, 4, 5, 7, 8, 31, 32, 33, 63, 64, 65, 65535, 65536):
            for prefix in (b"", b"xyz"):
                # Keep the prefixed specimen within the production small gate.
                if size + len(prefix) > 65536:
                    continue
                payload, raw = fixed(literal_payload(size, high), prefix)
                assert zlib.decompress(payload) == raw
                checks.append(("fixed", size, high, bool(prefix)))
    for size in (1, 8, 33, 65, 65536):
        raw = literal_payload(size)
        raw = raw[:-1] + b"\xff"
        payload, expected = dynamic_fallback(raw)
        assert zlib.decompress(payload) == expected
        checks.append(("dynamic-fallback", size))
    for ending in ("reserved-ll", "reserved-distance", "truncated"):
        payload, prefix = fixed(literal_payload(80), ending=ending)
        decoder = zlib.decompressobj()
        # Exclude checksum bytes: the malformed Deflate body is the actual input.
        if ending == "truncated":
            assert decoder.decompress(payload[:-4]) == prefix and not decoder.eof
        else:
            assert decoder.decompress(payload[:2 + (3 + len(prefix) * 8) // 8]) == prefix[:-1]
            try:
                zlib.decompress(payload)
            except zlib.error:
                pass
            else:
                raise AssertionError("invalid fixture unexpectedly decoded")
        checks.append((ending, len(prefix)))
    for size in (80, 65533):
        payload, expected = fixed_repeat(literal_payload(size, high=True))
        assert zlib.decompress(payload) == expected
        checks.append(("fixed-repeat", size))
    return checks
