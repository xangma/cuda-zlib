# Copyright (c) 2026 xangma
# SPDX-License-Identifier: MIT
"""Encoded-byte compatibility at cooperative-match boundaries and collisions."""
import hashlib
import random
import zlib

import numpy as np
import pytest

import cuda_zlib as codec
from test_decode import cuda_device


def check_scalar_and_batch(raw, expected, device):
    stream = np.asarray(codec.compress_zlib(raw, device)).tobytes()
    assert hashlib.sha256(stream).hexdigest() == expected
    assert zlib.decompress(stream) == raw
    # Include empty and short ragged siblings in the shared encoder call.
    inputs = (raw, b"", raw[:260])
    streams = codec.compress_zlib_batch_host(inputs, device)
    assert streams[0] == stream
    assert tuple(zlib.decompress(value) for value in streams) == inputs


@pytest.mark.parametrize("matched,expected", [
    (3, "b77ec108683faa391a8a9c4ddf50f3478e739ba88fd04ad5e1f47ea2ed124a19"),
    (31, "e1a9a576f16f711f6aea64033ad8add6728afca60a387e0b7302d9e6b4315ccb"),
    (32, "6bdbf9d2f7025895ca0c13fc005c25efa238cce0038752186cd5a0ab93bc2917"),
    (33, "492f272f34bc5b68251c5e86fdd264b7b88e7e14eee968037389de25bd7b9a70"),
    (34, "1cfa9186b059302c6845b623c3e11d4d34b9d36a805b1f5f20ae93b73ba1c195"),
    (35, "ab0cd9154e547425874b0c62a0b850b2068d314824a09cb9e95195bec7bfb7b0"),
    (36, "da585920dc1dc844be010f2b095397d0530b168bdd144c6a24b320cd71c83907"),
    (63, "e3eda6aaa34853dead5be24ed2ed9eccf7f4f9c0c8efa0dc7d28807bd83c5e61"),
    (64, "d5f9bdd7015a855830af962f058bb268df07b0e0c3121b226e9a83c1e2c71706"),
    (65, "bbc7fb6350a4748a75ce4ba9c9184e04efff9c73b36db37bd2a54e2a94a5c361"),
    (257, "df92f5406db058bcfd3f7b0192c20179e8b71b328b87965be52c18e0c9c52a0d"),
    (258, "d6b68b7c07a4da781b004f281488769881a46a2698e541352550eb9aa1dc5687"),
])
def test_encoder_match_boundaries(cuda_device, matched, expected):
    # Frozen pre-vectorization streams; the zero tail forces compressed mode
    # so errors in the earlier match cannot be hidden by stored-block fallback.
    unit = random.Random(811).randbytes(258)
    raw = (unit + b"\xff\x00" + unit[:matched] + b"\x01" +
           random.Random(813).randbytes(37) + bytes(4096))
    check_scalar_and_batch(raw, expected, cuda_device)


@pytest.mark.parametrize("alphabet,expected", [
    (2, "035f6f9dbce0dd5029396b0054150852140837d46e5d8aa41360c0e284fd5dab"),
    (4, "172e8a68fd558f87f6b96534e4a6dab945a1ff99e7538a87cabb0f2266fcb34c"),
    (16, "3371492b6ec5c4e13d1445d802bcc94f47bb433620c1b9aae2c0626530e44728"),
])
def test_encoder_dictionary_collisions(cuda_device, alphabet, expected):
    raw = np.random.default_rng(811).integers(
        0, alphabet, 4096, dtype=np.uint8).tobytes()
    check_scalar_and_batch(raw, expected, cuda_device)
