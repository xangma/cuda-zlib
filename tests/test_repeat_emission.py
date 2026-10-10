# Copyright (c) 2026 xangma
# SPDX-License-Identifier: MIT
"""Actual repeat helpers: bounded root writes and parser checkpoint recovery."""

import ast
import ctypes
import os
from pathlib import Path
import shutil
import subprocess
import zlib

import numpy as np
import pytest

import cuda_zlib as codec
from test_decode import _assert_bytes, _limited_checked_decoder, _stored, _wrap, cuda_device
from test_regressions import (
    _DISTANCE, _FIXED, _Writer, _literals, _match, _match_block_header, _no_cpu_codec,
)


_DRIVER = r'''
#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <vector>
#include <sys/mman.h>
#include <unistd.h>

TOKEN_SOURCE

struct { u32 x; } threadIdx;
struct alignas(16) uint4 { u32 x, y, z, w; };
u32 shuffled;
u32 __shfl_sync(u32 mask, u32 value, int source) {
    if (mask != 0xffffffffu || source != 0) std::exit(3);
    if (!threadIdx.x) shuffled = value;
    return shuffled;
}

EMISSION_SOURCE

void emit(bool serial, u32* roots, u32 prefix, u32 begin, u32 distance, u32 length) {
    if (serial) {
        emit_match_roots(roots, prefix, begin, distance, length);
    } else {
        // Lane zero publishes the real helper's seed; other lanes run in a
        // permutation. All oracle values are captured before any writes.
        for (u32 turn = 0; turn < 32; ++turn) {
            threadIdx.x = (turn * 13) % 32;
            emit_warp_match(roots, prefix, begin, distance, length, 0xffffffffu);
        }
    }
}

void check_writes(u32* roots, u32 words, u32 prefix, u32 begin,
                  u32 length, u32 seed, bool serial) {
    const u32 marker = 0x5a5a5a5a;
    std::fill(roots, roots + words, marker);
    roots[begin - 1] = seed;
    const u32 expected = begin - 1 < prefix ? begin - 1 : seed;
    emit(serial, roots, prefix, begin, 1, length);
    for (u32 i = 0; i < words; ++i) {
        const u32 wanted = i >= begin && i - begin < length ? expected :
                           i == begin - 1 ? seed : marker;
        if (roots[i] != wanted) {
            std::cerr << "root write: begin=" << begin << " length=" << length
                      << " prefix=" << prefix << " index=" << i << '\n';
            std::exit(1);
        }
    }
}

void check_seed(u32* roots, u32 words, u32 prefix, u32 begin,
                u32 distance, u32 length, bool serial) {
    std::fill(roots, roots + words, 0x5a5a5a5a);
    const u32 values[] = {0x800000abu, 17, 0x80000000u, 0, 0x800000ffu};
    const u32 first = begin - distance;
    std::vector<u32> history(distance);
    for (u32 i = 0; i < distance; ++i) {
        roots[first + i] = values[i % 5];
        history[i] = first + i < prefix ? first + i : roots[first + i];
    }
    std::vector<u32> expected(roots, roots + words);
    // Independent periodic-seed oracle, including overlapping copies.
    for (u32 i = 0; i < length; ++i) expected[begin + i] = history[i % distance];
    emit(serial, roots, prefix, begin, distance, length);
    for (u32 i = 0; i < words; ++i) {
        if (roots[i] != expected[i]) {
            std::cerr << "seed write: begin=" << begin << " length=" << length
                      << " distance=" << distance << " prefix=" << prefix
                      << " index=" << i << '\n';
            std::exit(1);
        }
    }
}

void protected_seed(u32* external, u32* roots, u32 words, u32 prefix,
                    u32 distance, u32 length, u32 split, bool serial) {
    std::fill(roots, roots + words, 0x5a5a5a5a);
    const u32 values[] = {0x800000abu, 17, 0x800000ffu};
    const u32 first = prefix - split, begin = first + distance;
    std::vector<u32> history(distance);
    for (u32 i = 0; i < distance; ++i) {
        if (i < split) history[i] = first + i;
        else history[i] = roots[i - split] = values[i % 3];
    }
    std::vector<u32> expected(roots, roots + words);
    for (u32 i = 0; i < length; ++i)
        expected[begin - prefix + i] = history[i % distance];
    emit(serial, external, prefix, begin, distance, length);
    for (u32 i = 0; i < words; ++i)
        if (roots[i] != expected[i]) std::exit(1);
}

void protected_read_tail(u32* roots, u32 words, u32 page_words,
                         u32 length, u32 alignment, bool serial) {
    std::fill(roots, roots + words, 0x5a5a5a5a);
    const u32 first = page_words * 2 - length;
    const u32 begin = page_words * 3 + alignment;
    const u32 values[] = {0x800000abu, 17, 0x80000000u, 0, 0x800000ffu};
    for (u32 i = 0; i < length; ++i) roots[first + i] = values[i % 5];
    std::vector<u32> expected(roots, roots + words);
    // The independent seed ends immediately before an inaccessible page;
    // even unused loads past the requested scalar tail must fail.
    std::copy(expected.begin() + first, expected.begin() + first + length,
              expected.begin() + begin);
    u32* guard = roots + page_words * 2;
    if (mprotect(guard, page_words * sizeof(u32), PROT_NONE)) std::exit(2);
    emit(serial, roots, 0, begin, begin - first, length);
    if (mprotect(guard, page_words * sizeof(u32), PROT_READ | PROT_WRITE)) std::exit(2);
    for (u32 i = 0; i < words; ++i)
        if (roots[i] != expected[i]) std::exit(1);
}

int main(int argc, char** argv) {
    const bool serial = argc == 2 && argv[1][0] == 's';
    const long page = sysconf(_SC_PAGESIZE);
    const u32 active = 16;
    void* mapping = mmap(nullptr, page * (active + 2), PROT_NONE,
                         MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (mapping == MAP_FAILED) return 2;
    u32* roots = reinterpret_cast<u32*>(static_cast<char*>(mapping) + page);
    if (mprotect(roots, page * active, PROT_READ | PROT_WRITE)) return 2;
    const u32 words = u32(page * active / sizeof(u32));
    const u32 lengths[] = {0,1,2,3,4,31,32,33,127,128,129,257,258,8191,8192,8193};
    const u32 seeds[] = {0,17,0x80000000u,0x800000ffu};
    for (u32 alignment = 0; alignment < 4; ++alignment) {
        for (u32 length : lengths) {
            // At most three scalar canaries separate the requested end from
            // a protected page; the roots base remains 16-byte aligned.
            const u32 tail = (words - length - alignment) & 3u;
            const u32 begin = words - length - tail;
            for (u32 seed : seeds) {
                check_writes(roots, words, begin, begin, length, seed, serial);
                check_writes(roots, words, begin - 1, begin, length, seed, serial);
                check_writes(roots, words, begin - 7, begin, length, seed, serial);
            }
        }
    }
    // A local seed may be the very first readable root.
    for (u32 length : lengths) check_writes(roots, words, 0, 1, length, 0x800000ab, serial);
    // The three-root fast path must preserve each independent seed, including
    // zero, one, two, or three roots belonging to the preceding block.
    const u32 distances[] = {3,4,5,31,32,33,255,256,257,258,4096};
    for (u32 alignment = 0; alignment < 4; ++alignment) {
        const u32 begin = words - 7 + alignment;
        for (u32 distance : distances)
            for (u32 split = 0; split <= 3; ++split)
                check_seed(roots, words, begin - distance + split, begin,
                           distance, 3, serial);
        // Distance two bypasses that path and repeats both differently tagged
        // seeds even when the copy crosses multiple warp/vector widths.
        for (u32 length : lengths) {
            const u32 tail = (words - length - alignment) & 3u;
            const u32 at = words - length - tail;
            for (u32 split = 0; split <= 2; ++split)
                check_seed(roots, words, at - 2 + split, at, 2, length, serial);
        }
    }
    // External distance-one history must use its numeric index without a
    // load: this seed lies in the preceding inaccessible page.
    std::fill(roots, roots + words, 0x5a5a5a5a);
    u32* external = reinterpret_cast<u32*>(mapping);
    const u32 begin = u32(page / sizeof(u32));
    for (u32 length : lengths) {
        std::fill(roots, roots + words, 0x5a5a5a5a);
        emit(serial, external, begin, begin, 1, length);
        for (u32 i = 0; i < words; ++i)
            if (roots[i] != (i < length ? begin - 1 : 0x5a5a5a5a)) return 1;
    }
    // Prefix straddles also place every external source in the protected page.
    // Only local seed roots are readable; none may be fetched speculatively.
    for (u32 distance : distances)
        for (u32 split = 0; split <= 3; ++split)
            protected_seed(external, roots, words, begin, distance, 3, split, serial);
    for (u32 split = 0; split <= 2; ++split)
        protected_seed(external, roots, words, begin, 2, 258, split, serial);
    // Every DEFLATE length in the grouped-read range, both sides of the
    // overlap boundary, all output alignments and all four tail sizes.
    for (u32 length = 4; length <= 258; ++length) {
        for (u32 distance : {length - 1, length, length + 1}) {
            const u32 middle = (distance / 2) & ~3u;
            const u32 cuts[] = {0,1,2,3,4,middle,middle+1,middle+2,middle+3,
                                distance-3,distance-2,distance-1,distance};
            for (u32 split : cuts) {
                if (split > distance) continue;
                for (u32 alignment = 0; alignment < 4; ++alignment) {
                    const u32 tail = (words - length - alignment) & 3u;
                    const u32 at = words - length - tail;
                    check_seed(roots, words, at - distance + split, at,
                               distance, length, serial);
                }
                // Prefix-straddled groups put each external seed behind a
                // protected page, catching loads from the external history.
                protected_seed(external, roots, words, begin, distance,
                               length, split, serial);
            }
        }
        for (u32 alignment = 0; alignment < 4; ++alignment)
            protected_read_tail(roots, words, begin, length, alignment, serial);
    }
    // Eight-seed groups need every prefix split, including all offsets within
    // a middle group. Keep these additional canary checks in one page beside
    // the final protected page rather than revisiting the full scratch buffer.
    u32* short_roots = roots + words - begin;
    for (u32 length = 4; length <= 258; ++length) {
        for (u32 distance : {length - 1, length, length + 1}) {
            const u32 middle = (distance / 2) & ~7u;
            std::vector<u32> cuts;
            for (u32 offset = 0; offset <= 8; ++offset) {
                cuts.push_back(offset);
                cuts.push_back(middle + offset);
            }
            std::sort(cuts.begin(), cuts.end());
            cuts.erase(std::unique(cuts.begin(), cuts.end()), cuts.end());
            for (u32 split : cuts) {
                if (split > distance) continue;
                for (u32 alignment = 0; alignment < 8; ++alignment) {
                    const u32 tail = (begin - length - alignment) & 7u;
                    const u32 at = begin - length - tail;
                    check_seed(short_roots, begin, at - distance + split, at,
                               distance, length, serial);
                }
                protected_seed(external, roots, begin, begin, distance,
                               length, split, serial);
            }
        }
        for (u32 alignment = 4; alignment < 8; ++alignment)
            protected_read_tail(roots, begin * 4, begin, length, alignment, serial);
    }
    return munmap(mapping, page * (active + 2)) != 0;
}

extern "C" void run_extend(const u8* data, u64 bits, u64 start,
                           u32 initial, u32 available, u64 end,
                           u32 accepted, u64* result) {
    DecodeTables tables{};
    if (fixed_tables(tables.ll, tables.dd)) std::exit(4);
    BitReader reader{data, bits, start, 0};
    reader.peek(u32(std::min<u64>(16, bits - start)));
    BitReader checkpoint = reader;
    // The fixture supplies known accepted token boundaries. Replay those
    // tokens to compare the complete parser/cache checkpoint, not just pos.
    for (u32 i = 0; i < accepted; ++i) {
        u32 size, distance;
        fixed_token(checkpoint, tables, size, distance);
    }
    result[0] = extend_repeat_run(reader, tables, initial, available, 32768, end);
    result[1] = reader.data == checkpoint.data && reader.bits == checkpoint.bits &&
                reader.pos == checkpoint.pos && reader.error == checkpoint.error &&
                reader.cache == checkpoint.cache && reader.cached == checkpoint.cached;
    result[2] = reader.pos;
    result[3] = reader.error;
    u32 size, distance;
    result[4] = u64(fixed_token(reader, tables, size, distance));
    result[5] = size;
    result[6] = distance;
    result[7] = reader.error;
    result[8] = reader.pos;
}
'''


@pytest.fixture(scope="module")
def repeat_helpers(tmp_path_factory):
    if os.name != "posix":
        pytest.skip("guard-page repeat checks require POSIX mmap")
    compiler = shutil.which("c++") or shutil.which("g++")
    if compiler is None:
        pytest.skip("a C++ compiler is required for actual repeat helper checks")
    path = Path(__file__).resolve().parents[1] / "src/cuda_zlib/_decode_kernels.py"
    source = next(ast.literal_eval(node.value) for node in ast.parse(path.read_text()).body
                  if isinstance(node, ast.Assign)
                  and any(isinstance(target, ast.Name) and target.id == "CUDA_SOURCE"
                          for target in node.targets))
    tokens = source[:source.index("struct BlockInfo {")]
    tokens += source[source.index("__device__ __forceinline__ int fixed_token("):
                     source.index("__device__ FixedSummary scan_fixed_region(")]
    emission = source[source.index("__device__ __forceinline__ void emit_match_roots("):
                      source.index("__device__ BlockInfo emit_fixed_segment(")]
    emission += source[source.index("__device__ __forceinline__ void emit_warp_match("):
                      source.index("__device__ __noinline__ BlockInfo emit_warp_block(")]
    cpp_source = _DRIVER.replace("TOKEN_SOURCE", tokens).replace("EMISSION_SOURCE", emission)
    for qualifier in ("__device__", "__noinline__", "__constant__"):
        cpp_source = cpp_source.replace(qualifier, "")
    cpp_source = cpp_source.replace("__forceinline__", "inline")
    directory = tmp_path_factory.mktemp("repeat-helpers")
    cpp, executable, library = directory / "repeat.cpp", directory / "repeat", directory / "repeat.so"
    cpp.write_text(cpp_source)
    for flags, output in (([], executable), (["-shared", "-fPIC"], library)):
        built = subprocess.run([compiler, "-std=c++11", "-O2", *flags, str(cpp), "-o", str(output)],
                               capture_output=True, text=True)
        assert built.returncode == 0, built.stdout + built.stderr
    native = ctypes.CDLL(str(library))
    native.run_extend.restype = None
    native.run_extend.argtypes = [ctypes.POINTER(ctypes.c_uint8), ctypes.c_uint64,
                                 ctypes.c_uint64, ctypes.c_uint32, ctypes.c_uint32,
                                 ctypes.c_uint64, ctypes.c_uint32,
                                 ctypes.POINTER(ctypes.c_uint64)]
    return executable, native.run_extend


@pytest.mark.parametrize("mode", ["warp", "serial"])
def test_actual_match_root_writes_are_bounded(repeat_helpers, mode):
    result = subprocess.run([str(repeat_helpers[0]), mode], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr


_STOPS = ("literal", "distance", "eob", "reserved", "reserved_distance",
          "truncated", "available", "segment", "cap_exact", "cap_cross", "cap_below", "end")


def _token_fixture(stop, alignment):
    writer, output = _Writer(), bytearray(b"AB")
    writer.put(0, alignment)
    accepted = [258, 31]
    if stop == "cap_exact":
        accepted = [258] * 30 + [194]
    elif stop == "cap_cross":
        accepted = [258] * 30
    elif stop == "cap_below":
        accepted = [258] * 30 + [193]
    elif stop == "end":
        accepted = []
    for length in accepted:
        _match(writer, output, length, 1)
    checkpoint = writer.bits
    status, size, distance, symbol = 0, 0, 0, 256
    if stop == "literal":
        writer.put(*_FIXED[90])
        size, symbol = 1, 90
    elif stop in ("distance", "available", "segment", "cap_exact", "cap_cross", "cap_below", "end"):
        size = {"cap_cross": 258, "cap_below": 3, "end": 258}.get(stop, 33)
        distance = 2 if stop == "distance" else 1
        _match(writer, output, size, distance)
        symbol = None
    elif stop == "reserved":
        writer.put(*_FIXED[286])
        status, symbol = 4, 286
    elif stop == "reserved_distance":
        writer.put(*_FIXED[257])
        writer.put(*_DISTANCE[30])
        status, size, symbol = 4, 3, 257
    elif stop == "truncated":
        _match(writer, output, 258, 1)
        status, symbol = 1, None
    else:
        writer.put(*_FIXED[256])
    next_end = writer.bits
    bits = writer.bits - (stop == "truncated")
    end = next_end - 1 if stop == "segment" else alignment if stop == "end" else bits
    total = 258 + sum(accepted)
    available = total + 32 if stop == "available" else 1 << 20
    return (writer.finish(), bits, end, len(accepted), total, available,
            checkpoint, (status, size, distance, symbol, next_end))


@pytest.mark.parametrize("stop", _STOPS)
def test_actual_repeat_extension_preserves_next_token_and_checkpoint(repeat_helpers, stop):
    for alignment in range(8):
        data, bits, end, accepted, total, available, checkpoint, next_token = _token_fixture(stop, alignment)
        array = (ctypes.c_uint8 * len(data)).from_buffer_copy(data)
        result = (ctypes.c_uint64 * 9)()
        repeat_helpers[1](array, bits, alignment, 258, available, end, accepted, result)
        assert list(result[:4]) == [total, 1, checkpoint, 0], (stop, alignment, list(result))
        status, size, distance, symbol, next_end = next_token
        assert result[7] == status
        if status != 1:
            assert (result[5], result[6], result[8]) == (size, distance, next_end)
        if symbol is not None:
            assert result[4] == symbol
        assert bytes(array) == data


def _grouped_match_stream(kind):
    writer = _Writer()
    seed = bytes((index * 37 + 11) % 256 for index in range(512))
    writer.aligned(_stored(seed, final=False))
    output = bytearray(seed)
    lengths = (4,5,6,7,8,15,16,17,31,32,33,127,128,129,257,258)
    for alignment in range(4):
        prefix = len(output)
        literals, distances = _match_block_header(writer, kind, final=alignment == 3)
        local = bytes((index * 73 + alignment * 29) % 256 for index in range(260))
        _literals(writer, local, literals)
        output.extend(local)
        for length in lengths:
            # The selected source begins on either side of the block prefix.
            # Every external/local split uses the ordinary byte history oracle.
            cuts = dict.fromkeys((0,1,2,3,length-3,length-2,length-1,length))
            for split in cuts:
                padding = bytes((0x5b,)) * ((alignment - len(output)) % 4)
                _literals(writer, padding, literals)
                output.extend(padding)
                distance = len(output) - prefix + split
                _match(writer, output, length, distance, literals, distances)
            # Exercise exact non-overlap and its neighboring overlap boundary
            # with roots produced by earlier matches, as well as literal roots.
            for distance in (length - 1, length, length + 1):
                padding = bytes((0xa6,)) * ((alignment - len(output)) % 4)
                _literals(writer, padding, literals)
                output.extend(padding)
                _match(writer, output, length, distance, literals, distances)
        writer.put(*literals[256])
    raw = bytes(output)
    return _wrap(writer.finish(), raw), raw


@pytest.mark.parametrize("kind", ["fixed", "dynamic"])
def test_grouped_match_fixtures_match_stdlib(kind):
    payload, raw = _grouped_match_stream(kind)
    assert len(raw) < 65536
    assert zlib.decompress(payload) == raw


@pytest.mark.parametrize("kind", ["fixed", "dynamic"])
def test_cuda_grouped_match_seeds_through_queued_root_pipeline(cuda_device, monkeypatch, kind):
    import jax

    payload, raw = _grouped_match_stream(kind)
    assert zlib.decompress(payload) == raw
    device = jax.devices("gpu")[cuda_device]
    source = jax.device_put(np.frombuffer(payload, np.uint8), device)
    with _no_cpu_codec(monkeypatch):
        decode = _limited_checked_decoder(device, len(raw), blocks=64)
        output, metadata = decode(source)
        np.testing.assert_array_equal(np.asarray(metadata), [0, 0])
        _assert_bytes(output, raw, device)


def _repeat_stream(kind, bad_history=False, cycles=3):
    writer = _Writer()
    # The first run reads an external block's seed; later literal stops supply
    # local tagged roots. Both representations must survive repeated stores.
    writer.aligned(_stored(b"A", final=False))
    literals, distances = _match_block_header(writer, kind)
    output = bytearray(b"A")
    for index, run in enumerate((8191, 8192, 8193) * cycles):
        for length in [258] * (run // 258) + [run % 258]:
            _match(writer, output, length, 1, literals, distances)
        if bad_history and index == 0:
            # A valid code/window but impossible preceding history. The run
            # lookahead must leave this failure to the ordinary parser.
            writer.put(*literals[257])
            writer.put(*distances[29])
            writer.put(8191, 13)
            output.extend(b"A" * 3)
        elif index % 2:
            _match(writer, output, 33, 2, literals, distances)
        else:
            literal = bytes((66 + index,))
            _literals(writer, literal, literals)
            output.extend(literal)
    writer.put(*literals[256])
    raw = bytes(output)
    return _wrap(writer.finish(), raw), raw


@pytest.mark.parametrize("kind", ["fixed", "dynamic"])
def test_consecutive_repeat_fixtures_match_stdlib(kind):
    payload, raw = _repeat_stream(kind)
    assert len(raw) > 65536
    assert zlib.decompress(payload) == raw
    invalid, _ = _repeat_stream(kind, bad_history=True)
    with pytest.raises(zlib.error):
        zlib.decompress(invalid)


@pytest.mark.parametrize("kind", ["fixed", "dynamic"])
def test_cuda_repeat_runs_failures_neighbors_and_fresh_recovery(cuda_device, monkeypatch, kind):
    import jax

    payload, raw = _repeat_stream(kind)
    invalid, invalid_raw = _repeat_stream(kind, bad_history=True)
    assert zlib.decompress(payload) == raw
    checksum = payload[:-1] + bytes((payload[-1] ^ 1,))
    truncated = payload[:-5] + payload[-4:]
    with pytest.raises(zlib.error):
        zlib.decompress(truncated)
    streams = (payload, invalid, checksum, truncated, payload, payload)
    expected = (len(raw), len(invalid_raw), len(raw), len(raw), len(raw) - 1, len(raw))
    large = _repeat_stream(kind, cycles=43) if kind == "dynamic" else None
    if large is not None:
        assert len(large[1]) > 1 << 20
        assert zlib.decompress(large[0]) == large[1]
    with _no_cpu_codec(monkeypatch):
        output, metadata = codec.decompress_zlib_batch_checked(
            b"".join(streams), tuple(map(len, streams)), expected, cuda_device)
        statuses = np.asarray(metadata)
        assert list(statuses[:, 0])[:3] == [0, 6, 21]
        assert statuses[3, 0] != 0
        assert list(statuses[:, 0])[-2:] == [7, 0]
        assert not np.any(statuses[:, 1])
        recovered, recovered_metadata = codec.decompress_zlib_checked(payload, len(raw), cuda_device)
        np.testing.assert_array_equal(np.asarray(recovered_metadata), [0, 0])
        _assert_bytes(recovered, raw, cuda_device)
        # A reduced block budget selects the queued general parser for these
        # small compressed streams, exercising its serial root emission too.
        device = jax.devices("gpu")[cuda_device]
        decode = _limited_checked_decoder(device, len(raw), blocks=64)
        source, malformed = [jax.device_put(np.frombuffer(value, np.uint8), device)
                             for value in (payload, checksum)]
        queued, queued_metadata = decode(source)
        _, queued_error = decode(malformed)
        queued_recovery, queued_recovery_metadata = decode(source)
        for metadata, status in ((queued_metadata, 0), (queued_error, 21),
                                 (queued_recovery_metadata, 0)):
            np.testing.assert_array_equal(np.asarray(metadata), [status, 0])
        _assert_bytes(queued, raw, cuda_device)
        _assert_bytes(queued_recovery, raw, cuda_device)
        if large is not None:
            warped, warped_metadata = codec.decompress_zlib_checked(
                large[0], len(large[1]), cuda_device)
            np.testing.assert_array_equal(np.asarray(warped_metadata), [0, 0])
            _assert_bytes(warped, large[1], cuda_device)
        packed = np.asarray(output)
        assert packed[:len(raw)].tobytes() == packed[-len(raw):].tobytes() == raw
