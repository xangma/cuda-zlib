# Copyright (c) 2026 xangma
# SPDX-License-Identifier: MIT
"""CPU checks of the serial CUDA Huffman helper, without a JAX/CUDA runtime."""

import ast
import ctypes
import hashlib
from pathlib import Path
import shutil
import struct
import subprocess

import pytest


def _helper_source(path):
    assignments = ast.parse(path.read_text()).body
    source = next(ast.literal_eval(node.value) for node in assignments
                  if isinstance(node, ast.Assign)
                  and any(isinstance(target, ast.Name) and target.id == "CUDA_SOURCE"
                          for target in node.targets))
    declarations = source.split("struct BitWriter", 1)[0]
    reverse = source[source.index("__device__ enc_u32 enc_reverse("):
                     source.index("__device__ void fixed_symbol(")]
    tree = source[source.index("__device__ bool enc_less("):
                  source.index("__device__ enc_u8 enc_length_at(")]
    wrapper = r'''
extern "C" bool run_tree(const enc_u32* freq, enc_u32 alphabet, enc_u32 limit,
                        enc_u8* lengths, enc_u16* codes, enc_u16* parent,
                        enc_u16* heap, enc_u32* weight) {
  return enc_tree(freq, alphabet, limit, lengths, codes, parent, heap, weight);
}
'''
    return (declarations + reverse + tree + wrapper).replace("__device__", "")


def _compile_helper(path, directory):
    compiler = shutil.which("c++") or shutil.which("g++")
    if compiler is None:
        pytest.skip("a C++ compiler is required for serial Huffman helper checks")
    source = directory / "tree.cpp"
    library = directory / "tree.so"
    source.write_text(_helper_source(path))
    result = subprocess.run([compiler, "-std=c++11", "-shared", "-fPIC", "-O2",
                             str(source), "-o", str(library)], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    native = ctypes.CDLL(str(library))
    native.run_tree.restype = ctypes.c_bool
    native.run_tree.argtypes = [ctypes.POINTER(ctypes.c_uint32),
                               ctypes.c_uint32, ctypes.c_uint32,
                               ctypes.POINTER(ctypes.c_uint8),
                               ctypes.POINTER(ctypes.c_uint16),
                               ctypes.POINTER(ctypes.c_uint16),
                               ctypes.POINTER(ctypes.c_uint16),
                               ctypes.POINTER(ctypes.c_uint32)]
    return native.run_tree


@pytest.fixture(scope="module")
def tree_helper(tmp_path_factory):
    source = Path(__file__).resolve().parents[1] / "src/cuda_zlib/_encode_kernels.py"
    return _compile_helper(source, tmp_path_factory.mktemp("huffman-helper"))


class _Guarded:
    def __init__(self, kind, size, marker):
        self.kind, self.size, self.marker = kind, size, marker
        self.storage = (kind * (size + 2))(*([marker] * (size + 2)))
        self.pointer = ctypes.cast(ctypes.byref(self.storage, ctypes.sizeof(kind)),
                                   ctypes.POINTER(kind))

    def values(self):
        return list(self.storage)[1:-1]

    def check(self):
        assert self.storage[0] == self.storage[-1] == self.marker


class _Scratch:
    def __init__(self):
        # Capacities are the shared arrays used by the production caller.
        self.parent = _Guarded(ctypes.c_uint16, 571, 0xA55A)
        self.heap = _Guarded(ctypes.c_uint16, 286, 0x5AA5)
        self.weight = _Guarded(ctypes.c_uint32, 571, 0xD00DFEED)

    def check(self):
        for array in (self.parent, self.heap, self.weight):
            array.check()


def _run(helper, frequencies, limit, scratch):
    alphabet = len(frequencies)
    freq = _Guarded(ctypes.c_uint32, alphabet, 0xABCD1234)
    for i, value in enumerate(frequencies):
        freq.pointer[i] = value
    lengths = _Guarded(ctypes.c_uint8, alphabet, 0xA5)
    codes = _Guarded(ctypes.c_uint16, alphabet, 0xBEEF)
    assert helper(freq.pointer, alphabet, limit, lengths.pointer, codes.pointer,
                  scratch.parent.pointer, scratch.heap.pointer, scratch.weight.pointer)
    for array in (freq, lengths, codes):
        array.check()
    scratch.check()
    assert freq.values() == frequencies
    return lengths.values(), codes.values()


def _frequencies(alphabet, case):
    frequencies = [0] * alphabet
    if case == "empty":
        return frequencies
    if case in ("one-first", "one-last"):
        frequencies[0 if case == "one-first" else alphabet - 1] = 32768
    elif case == "ties":
        frequencies = [7] * alphabet
    elif case == "sparse":
        for index, frequency in zip((0, 2, alphabet // 2, alphabet - 2, alphabet - 1),
                                    (3, 11, 3, 11, 27)):
            frequencies[index] = frequency
    elif case == "limit-edge":
        count = 8 if alphabet == 19 else 16
        frequencies[:count] = [1] + [1 << i for i in range(count - 1)]
    elif case == "skew":
        # Fibonacci weights produce an overlong initial tree at both RFC limits.
        a, b = 1, 1
        for index in range(min(alphabet, 20)):
            frequencies[index] = a
            a, b = b, a + b
    else:
        raise AssertionError(case)
    return frequencies


def _digest(lengths, codes):
    # Explicit little endian packing makes the baseline fixture portable.
    return hashlib.sha256(bytes(lengths) + struct.pack(f"<{len(codes)}H", *codes)).hexdigest()


def _check_tree(frequencies, limit, lengths, codes):
    active = {i for i, value in enumerate(frequencies) if value}
    if not active:
        active = {0, 1}
    elif len(active) == 1:
        active.add(0 if 0 not in active else 1)
    assert {i for i, length in enumerate(lengths) if length} == active
    assert all(1 <= lengths[i] <= limit for i in active)
    assert sum(1 << (limit - lengths[i]) for i in active) == 1 << limit
    assert all(code < 1 << length for code, length in zip(codes, lengths))
    # DEFLATE emits these reversed codes least-significant bit first.
    for i in active:
        for j in active:
            if i != j and lengths[i] <= lengths[j]:
                assert codes[j] & ((1 << lengths[i]) - 1) != codes[i]
    if len(active) == 2:
        assert sorted((lengths[i], codes[i]) for i in active) == [(1, 0), (1, 1)]
    if len(set(frequencies)) == 1 and frequencies[0]:
        shorter = len(frequencies).bit_length() - 1
        assert set(lengths) <= {shorter, shorter + 1}
        assert lengths.count(shorter) == (1 << (shorter + 1)) - len(frequencies)


# Frozen from merged 22d597ba3de691a9352f03e72b9d1c5f0d0454cd, not an
# independent implementation of the same algorithm. Prefix and support checks
# above independently verify the mathematical properties of each result.
_BASELINE = {
    (286, "empty"): "0873acbc2ca1433d8b701935dcbf9e7a2437827334f32d1e9e2bb0f6efebbba3",
    (286, "one-first"): "0873acbc2ca1433d8b701935dcbf9e7a2437827334f32d1e9e2bb0f6efebbba3",
    (286, "one-last"): "991feda24ec7e17bfe026784638434539387964928a06977f216f91f8e8ea195",
    (286, "ties"): "06ae33d1b35f9d65ac1646fd5e4f6f59f6e26158e4cd43c1b5ca17c148789612",
    (286, "sparse"): "4c8af4b3b5deb19eaeed585eebe8912799dc60bbbbab720c2dfe19c726d325dd",
    (286, "limit-edge"): "49c571bd68c5a47c0b3255607e179a0dcc3b85c56802c8eed7b868c2595c87b4",
    (286, "skew"): "901de3b9caf6be031a90775e422a4e3c650e1025a1f95cda273b44ff139b1cf8",
    (30, "empty"): "07f14f364baf543eb1445a2ef8f1d70ee5d098ad1af8343ab1ae300cbf2764e9",
    (30, "one-first"): "07f14f364baf543eb1445a2ef8f1d70ee5d098ad1af8343ab1ae300cbf2764e9",
    (30, "one-last"): "7ed951348595c984008d013f38f4c531ec6867b2415714c91919e0d781d080fb",
    (30, "ties"): "9255af13461b8e26e5cc32a9b62a8ada7cb270e10b2bdc967a17e45dc0e0104b",
    (30, "sparse"): "4147e2965859b58d1a2bbc4bf84028333104025a1015363b286bec06c102dc1a",
    (30, "limit-edge"): "23dd5be0f1c69d456f24db91816b5596ef48e5f63c8b1a8d1e3e2ef42b78e784",
    (30, "skew"): "36c18d6cff61318a9a955727a720d96a3a8161a3e6710656594043e9b6ea5147",
    (19, "empty"): "1fc0bffa8bb4919a9bb8808ad89db23afb939b0de9b613f936d7e527a9ec8181",
    (19, "one-first"): "1fc0bffa8bb4919a9bb8808ad89db23afb939b0de9b613f936d7e527a9ec8181",
    (19, "one-last"): "dcf3984683c083fec86a9d2c346da390073e5dd91b4324fa4c027f38f5d6a47c",
    (19, "ties"): "02c947f23e07d7a2d66cf5b69f346f1b0f0a711576bdf578d94426605913fb43",
    (19, "sparse"): "cc883c9e496d4282f73874fdd378536699a53736749886f04579ba4ec4fd175d",
    (19, "limit-edge"): "8570d01908a79f629819ddc0ce97ae2a671d783a58ba7465dd44f95c0c514539",
    (19, "skew"): "df3a6bc62dbda87b7eed3aeaf063a11d2f1666c61ab7ff7368b643116070ef56",
}


@pytest.mark.parametrize("alphabet,limit", [(286, 15), (30, 15), (19, 7)])
@pytest.mark.parametrize("case", ["empty", "one-first", "one-last", "ties", "sparse", "limit-edge", "skew"])
def test_huffman_lengths_and_codes(tree_helper, alphabet, limit, case):
    frequencies = _frequencies(alphabet, case)
    scratch = _Scratch()
    lengths, codes = _run(tree_helper, frequencies, limit, scratch)
    _check_tree(frequencies, limit, lengths, codes)
    assert _digest(lengths, codes) == _BASELINE[alphabet, case]
    if case == "limit-edge":
        assert max(lengths) == limit
    if case == "skew":
        # The original positive weights are unchanged on a no-retry path.
        # Reduced leaf weights prove that depth limiting actually retried.
        assert scratch.weight.values()[min(alphabet, 20) - 1] < max(frequencies)


def test_huffman_shared_scratch_reuse(tree_helper):
    scratch = _Scratch()
    # Match the three alphabet calls sharing one workspace in encode_chunk;
    # repeat with very different support to expose stale internal parents/depths.
    for case in ("ties", "skew", "empty", "sparse", "one-last", "limit-edge", "one-first", "ties"):
        for alphabet, limit in ((286, 15), (30, 15), (19, 7)):
            frequencies = _frequencies(alphabet, case)
            lengths, codes = _run(tree_helper, frequencies, limit, scratch)
            _check_tree(frequencies, limit, lengths, codes)
            assert _digest(lengths, codes) == _BASELINE[alphabet, case]
