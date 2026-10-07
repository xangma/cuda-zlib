# Copyright (c) 2026 xangma
# SPDX-License-Identifier: MIT
"""Byte compatibility when dynamic-tree lower bounds prune tiny chunks."""

import hashlib
import random
import zlib

import numpy as np
import pytest

import cuda_zlib as codec
from test_decode import cuda_device


# Frozen from merged 22d597ba3de691a9352f03e72b9d1c5f0d0454cd. Include
# zeros256, where the minimum dynamic extent ties the winning fixed extent;
# byte rounding, final/nonfinal chunk transitions, and retained block modes.
_CASES = [
    (0, "zeros", 256, 8, "09d469dfeeaf4c436fd3f80ba7e168bcc34e5050993b5da3c8416fbae680f18f"),
    (1, "zeros", 256, 9, "94b176b91437ca9041d4081d518e88e36c71ffc4bc598800b2bccd6d9a080f03"),
    (31, "random", 32768, 40, "4de3c8d0b428600430a56db31bc4b93e78adc16229b5e98f0fb950ef2f98ba6b"),
    (32, "random", 32768, 42, "de9402b5ef4480c17b6a237218379c019564193ae36e0d9475a44c52d1b0bf6f"),
    (33, "random", 32768, 42, "8627d436bf1917ff6a2b7315c509c21c3691aab48f971d9e73e6c6b965d24d19"),
    (255, "zeros", 256, 11, "2e11d3e28b5ca8118490548f7a113d481f8ec8cd706deae2899e0ecfa02d08a3"),
    (256, "zeros", 256, 11, "8a629670f880bd07dc82c3144b18bb1bb907ff7c6886f37c1ac630f302e0f6f4"),
    (257, "zeros", 256, 18, "db98f082e98955b13b0009bd20d26e7db69437f04fb26713e79bdff666927588"),
    (257, "zeros", 32768, 11, "edbc3d65efb25811f92d6ca6414495feab3e11d60a493a0a0d73bc7ab1a2ab75"),
    (511, "text", 256, 30, "b06e5bc021358426747a7709d912695024810542418ad1fb3e779d7d4c17b9e9"),
    (512, "text", 256, 30, "3b0c798cf10b5b24b6620c415bbfdaa55385e30b58f2433a9d4afcca773fc743"),
    (513, "text", 256, 37, "70adf549bfc314f2f01168462aea697d6a75894726bef16ce1445a324f49a57d"),
    (511, "random", 256, 527, "5a99992ae38f06ef08d4054a9b640b357f7b8728c424c127aaec2eded5fb6821"),
    (512, "random", 256, 528, "446ad1ff887ee73c67b68efa281af4543e69dbbbaa629c9f2d26e16f7d4d612a"),
    (513, "random", 256, 531, "df739d13a86c0e796d26c9122ddfba45f5e396f0133949a7e3c5bf48c9c9ef4b"),
    (65536, "zeros", 32768, 102, "566c852d00fe42c5ca9f2d6be9766aec46b24d6d2c4d8dae2fc856f6b62c1546"),
    (65536, "text", 32768, 148, "b24e3886d88ee3610dcdfb723325e6e61220c98467b76fc095ac961e0aa31abd"),
    (65536, "random", 32768, 65552, "8aa4f264e8b63691a5c4a56a03e5731005197d2f0eaa08d5c252a5d9171e3574"),
]


@pytest.mark.parametrize("size,kind,chunk_bytes,encoded_size,expected", _CASES)
def test_encoder_dynamic_bound_compatibility(
        cuda_device, size, kind, chunk_bytes, encoded_size, expected):
    if kind == "zeros":
        raw = bytes(size)
    elif kind == "text":
        raw = (b"ABCxyz" * ((size + 5) // 6))[:size]
    else:
        raw = random.Random(123 + size).randbytes(size)
    stream = np.asarray(codec.compress_zlib(
        raw, cuda_device, chunk_bytes=chunk_bytes)).tobytes()
    assert len(stream) == encoded_size
    assert hashlib.sha256(stream).hexdigest() == expected
    assert zlib.decompress(stream) == raw
    if size == 65536:
        assert (stream[2] >> 1) & 3 == (0 if kind == "random" else 2)
    if kind == "zeros" and size == 256:
        assert (stream[2] >> 1) & 3 == 1
    # Ragged neighbors exercise per-file finality and normal status handling.
    inputs = (b"", raw, raw[:1])
    streams = codec.compress_zlib_batch_host(
        inputs, cuda_device, chunk_bytes=chunk_bytes)
    assert streams[1] == stream
    assert tuple(zlib.decompress(value) for value in streams) == inputs
