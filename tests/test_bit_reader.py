# Copyright (c) 2026 xangma
# SPDX-License-Identifier: MIT
"""Bit-reader bounds/cache checks and real CUDA truncated-stream recovery."""

import ast
import os
from pathlib import Path
import shutil
import subprocess
import zlib

import numpy as np
import pytest

import cuda_zlib as codec
from test_decode import (
    _assert_bytes, _bits, _compressed, _dynamic_literal_fields, _wrap, cuda_device,
)
from test_regressions import _no_cpu_codec


_DRIVER = r'''
#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <sys/mman.h>
#include <unistd.h>

typedef unsigned char u8;
typedef unsigned int u32;
typedef unsigned long long u64;

READER_SOURCE

void require(bool ok, const char* message, u64 bits, u64 pos, u32 width) {
    if (!ok) {
        std::cerr << message << ": bits=" << bits << " pos=" << pos
                  << " width=" << width << '\n';
        std::exit(1);
    }
}

// Read each requested bit independently; no cache or word-load algorithm.
u32 reference(const u8* data, u64 pos, u32 width) {
    u32 value = 0;
    for (u32 i = 0; i < width; ++i)
        value |= u32((data[(pos + i) / 8] >> ((pos + i) % 8)) & 1) << i;
    return value;
}

void requests(u8* end) {
    for (u64 bits = 0; bits <= 137; ++bits) {
        u8* data = end - (bits + 7) / 8;
        for (u64 i = 0; i < (bits + 7) / 8; ++i)
            data[i] = u8(0xa7 + 73 * i + bits);
        for (u64 start = 0; start <= bits + 1; ++start) {
            for (u32 width = 0; width <= 16; ++width) {
                const bool bad = start > bits || width > bits - start;
                const u32 expected = bad ? 0 : reference(data, start, width);
                BitReader reader{data, bits, start, 0, 0, 0};
                require(reader.peek(width) == expected, "peek", bits, start, width);
                require(reader.pos == start && reader.error == u32(bad),
                        "peek state", bits, start, width);
                // A second peek must preserve the value and cursor.
                require(reader.peek(width) == expected, "repeated peek", bits, start, width);
                require(reader.take(width) == expected, "take", bits, start, width);
                require(reader.pos == start + (bad ? 0 : width) &&
                        reader.error == u32(bad), "take state", bits, start, width);
                if (bad) {
                    reader.seek(0);
                    require(reader.take(0) == 0 && reader.error == 1 && reader.pos == 0,
                            "sticky error after seek", bits, start, width);
                }
            }
        }
    }
}

void reuse(u8* end) {
    for (u64 bits = 1; bits <= 137; ++bits) {
        u8* data = end - (bits + 7) / 8;
        for (u64 i = 0; i < (bits + 7) / 8; ++i)
            data[i] = u8(0x5b ^ (117 * i + bits));
        for (u64 start = 0; start < 8 && start <= bits; ++start) {
            BitReader reader{data, bits, start, 0, 0, 0};
            u64 pos = start;
            for (u32 step = 0; step < 128; ++step) {
                if (step % 5 == 0) {
                    pos = (step * 37 + start) % (bits + 1);
                    reader.seek(pos);
                }
                const u32 width = std::min<u64>((step * 11 + start) % 17, bits - pos);
                const u32 expected = reference(data, pos, width);
                require(reader.peek(0) == 0, "zero peek", bits, pos, 0);
                require(reader.peek(width) == expected, "reused peek", bits, pos, width);
                // Narrow and broad peeks must coexist without consuming bits.
                const u32 narrow = width / 2;
                require(reader.peek(narrow) == reference(data, pos, narrow),
                        "narrow peek", bits, pos, narrow);
                if (step % 2) {
                    require(reader.take(width) == expected, "reused take", bits, pos, width);
                } else {
                    reader.drop(width);
                }
                pos += width;
                require(reader.pos == pos && reader.error == 0,
                        "reused state", bits, pos, width);
            }
            // Seek to a short tail after a full refill, then cross its bound.
            reader.seek(0);
            reader.peek(std::min<u64>(16, bits));
            pos = bits - std::min<u64>(15, bits);
            reader.seek(pos);
            const u32 tail = u32(bits - pos);
            require(reader.peek(tail) == reference(data, pos, tail),
                    "seek tail", bits, pos, tail);
            require(reader.take(tail + 1) == 0 && reader.error == 1 && reader.pos == pos,
                    "cached tail bounds", bits, pos, tail + 1);
        }
    }
}

int main(int argc, char** argv) {
    const long page = sysconf(_SC_PAGESIZE);
    void* mapping = mmap(nullptr, page * 2, PROT_READ | PROT_WRITE,
                         MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (mapping == MAP_FAILED || mprotect(static_cast<u8*>(mapping) + page,
                                         page, PROT_NONE)) return 2;
    // Every payload ends immediately before an inaccessible page. Both byte
    // alignment and the number of valid bits vary, including an empty payload.
    u8* end = static_cast<u8*>(mapping) + page;
    if (argc == 2 && argv[1][0] == 'r') reuse(end);
    else requests(end);
    return munmap(mapping, page * 2) != 0;
}
'''


@pytest.fixture(scope="module")
def bit_reader_helper(tmp_path_factory):
    if os.name != "posix":
        pytest.skip("guard-page reader checks require POSIX mmap")
    compiler = shutil.which("c++") or shutil.which("g++")
    if compiler is None:
        pytest.skip("a C++ compiler is required for bit-reader checks")
    path = Path(__file__).resolve().parents[1] / "src/cuda_zlib/_decode_kernels.py"
    source = next(ast.literal_eval(node.value) for node in ast.parse(path.read_text()).body
                  if isinstance(node, ast.Assign)
                  and any(isinstance(target, ast.Name) and target.id == "CUDA_SOURCE"
                          for target in node.targets))
    reader = source[source.index("struct BitReader {"):
                    source.index("__device__ u32 reverse_code(")]
    reader = reader.replace("__device__", "").replace("__forceinline__", "inline")
    directory = tmp_path_factory.mktemp("bit-reader-helper")
    cpp, executable = directory / "reader.cpp", directory / "reader"
    cpp.write_text(_DRIVER.replace("READER_SOURCE", reader))
    built = subprocess.run([compiler, "-std=c++11", "-O2", str(cpp), "-o", str(executable)],
                           capture_output=True, text=True)
    assert built.returncode == 0, built.stdout + built.stderr
    return executable


@pytest.mark.parametrize("mode", ["bounds", "reuse"])
def test_actual_bit_reader_at_guarded_payload_tails(bit_reader_helper, mode):
    result = subprocess.run([str(bit_reader_helper), mode], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr


def _tail_fixture(kind):
    if kind == "fixed":
        raw = bytes(range(32)) * 17 + b"\x8f\x90\xff"
        payload = _compressed(raw, strategy=zlib.Z_FIXED)
        assert (payload[2] >> 1) & 3 == 1
    else:
        raw = bytes(range(15)) * 2
        payload = _wrap(_bits(_dynamic_literal_fields(raw, long_codes=True)), raw)
        assert (payload[2] >> 1) & 3 == 2
    # Retain the real zlib header/checksum while cutting every body-byte tail.
    truncated = tuple(payload[:2] + payload[2:2 + cut] + payload[-4:]
                      for cut in range(len(payload) - 6))
    return payload, raw, truncated


@pytest.mark.parametrize("kind", ["fixed", "dynamic"])
def test_reader_tail_fixtures_match_stdlib(kind):
    payload, raw, truncated = _tail_fixture(kind)
    assert zlib.decompress(payload) == raw
    for invalid in truncated:
        with pytest.raises(zlib.error):
            zlib.decompress(invalid)


@pytest.mark.parametrize("kind", ["fixed", "dynamic"])
def test_cuda_all_body_truncations_checked_recovery(cuda_device, monkeypatch, kind):
    import jax

    payload, raw, truncated = _tail_fixture(kind)
    assert zlib.decompress(payload) == raw
    checksum = payload[:-1] + bytes((payload[-1] ^ 1,))
    streams = (payload, *truncated, checksum, payload)
    sizes = tuple(map(len, streams))
    expected = (len(raw),) * len(streams)
    device = jax.devices("gpu")[cuda_device]
    source = jax.device_put(np.frombuffer(b"".join(streams), np.uint8), device)
    decode = jax.jit(lambda value: codec.decompress_zlib_batch_checked(
        value, sizes, expected, device))
    with _no_cpu_codec(monkeypatch):
        result, metadata = decode(source)
        statuses = np.asarray(metadata)
        assert metadata.dtype == np.uint32 and metadata.devices() == {device}
        assert statuses.shape == (len(streams), 2)
        assert not np.any(statuses[:, 1])
        assert statuses[0, 0] == statuses[-1, 0] == 0
        assert np.all(statuses[1:-2, 0] != 0)
        assert statuses[-2, 0] == 21
        recovered, recovered_metadata = codec.decompress_zlib_checked(
            payload, len(raw), device)
        np.testing.assert_array_equal(np.asarray(recovered_metadata), [0, 0])
        _assert_bytes(recovered, raw, device)
        # Retained neighbors must remain valid after malformed calls/recovery.
        output = np.asarray(result)
        assert output[:len(raw)].tobytes() == output[-len(raw):].tobytes() == raw
