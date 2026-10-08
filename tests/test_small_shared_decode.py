# Copyright (c) 2026 xangma
# SPDX-License-Identifier: MIT
"""Completed small/shared-route boundary outputs and retained metadata."""

import gc
import random
import zlib

import numpy as np
import pytest

import cuda_zlib as codec
from test_decode import _assert_bytes, _compressed, cuda_device
from test_regressions import _no_cpu_codec


@pytest.mark.parametrize("size", [65535, 65536, 65537])
@pytest.mark.parametrize("stored", [False, True], ids=["dynamic", "stored"])
def test_small_decode_tail_and_queued_status_ownership(
    cuda_device, monkeypatch, size, stored,
):
    import jax
    import jax.numpy as jnp

    rng = random.Random(66823 + size)
    raw = bytes(rng.choices(range(32), weights=range(1, 33), k=size - 33))
    raw += bytes(range(223, 256))  # Distinct nonzero bytes exercise the last warp.
    alternate = raw[::-1]
    level = 0 if stored else 6
    payloads = [_compressed(value, level=level) for value in (raw, alternate)]
    assert all(zlib.decompress(payload) == value
               for payload, value in zip(payloads, (raw, alternate)))
    neighbor = random.Random(9153).randbytes(65537)
    neighbor_payload = _compressed(neighbor, level=0)
    device = jax.devices("gpu")[cuda_device]
    sources = [jax.device_put(np.frombuffer(payload, np.uint8), device)
               for payload in payloads]
    neighbor_source = jax.device_put(np.frombuffer(neighbor_payload, np.uint8), device)
    decode = jax.jit(lambda value: codec.decompress_zlib_checked(value, size, device))
    other = jax.jit(lambda value: codec.decompress_zlib_checked(value, len(neighbor), device))
    with _no_cpu_codec(monkeypatch):
        queued = [decode(sources[0]),
                  decode(sources[1].at[-1].set(sources[1][-1] ^ jnp.uint8(1))),
                  other(neighbor_source), decode(sources[1]),
                  decode(sources[0].at[0].set(jnp.uint8(0)))]
        jax.block_until_ready(queued)
        first, first_metadata = queued[0]
        other_data, other_metadata = queued[2]
        reverse, reverse_metadata = queued[3]
        errors = (queued[1][1], queued[4][1])
        del queued, sources, neighbor_source
        gc.collect()
        codec.trim_workspace_pool(device)
        replacement = jax.device_put(np.frombuffer(payloads[0], np.uint8), device)
        later, later_metadata = decode(replacement)
        jax.block_until_ready((later, later_metadata))
        for metadata in (first_metadata, other_metadata, reverse_metadata, later_metadata):
            assert metadata.devices() == {device}
            np.testing.assert_array_equal(np.asarray(metadata), [0, 0])
        for metadata, status in zip(errors, (21, 15)):
            assert metadata.devices() == {device}
            np.testing.assert_array_equal(np.asarray(metadata), [status, 0])
        for data, wanted in ((first, raw), (reverse, alternate),
                             (other_data, neighbor), (later, raw)):
            _assert_bytes(data, wanted, device)
