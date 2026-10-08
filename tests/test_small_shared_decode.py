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
from test_decode import _wrap
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


def _block_transition_payload(kind):
    if kind == "fixed":
        # Nonfinal fixed literal A/EOB, then a fixed header at bit 18 and B/EOB.
        return bytes.fromhex("789c7204cc090000c60084"), b"AB"
    prefix = _compressed(b"A", strategy=zlib.Z_FIXED, wbits=-15,
                         flush=zlib.Z_SYNC_FLUSH)
    assert (prefix[0] >> 1) & 3 == 1
    if kind == "stored":
        suffix = bytes(range(129, 162))
        encoded = _compressed(suffix, level=0, wbits=-15)
        assert (encoded[0] >> 1) & 3 == 0
    else:
        assert kind == "dynamic"
        suffix = bytes(random.Random(971).choices(
            range(16), weights=range(1, 17), k=511)) + b"\xee"
        encoded = _compressed(suffix, wbits=-15)
        assert (encoded[0] >> 1) & 3 == 2
    # SYNC_FLUSH ends a nonfinal fixed block and an empty stored block.
    raw = b"A" + suffix
    return _wrap(prefix + encoded, raw), raw


@pytest.mark.parametrize("kind", ["fixed", "stored", "dynamic"])
def test_small_literal_eob_block_transition(cuda_device, monkeypatch, kind):
    import jax

    payload, raw = _block_transition_payload(kind)
    assert zlib.decompress(payload) == raw
    device = jax.devices("gpu")[cuda_device]
    source = jax.device_put(np.frombuffer(payload, np.uint8), device)
    decode = jax.jit(lambda value: codec.decompress_zlib_checked(value, len(raw), device))
    with _no_cpu_codec(monkeypatch):
        data, metadata = decode(source)
        jax.block_until_ready((data, metadata))
        _assert_bytes(data, raw, device)
        assert metadata.devices() == {device}
        np.testing.assert_array_equal(np.asarray(metadata), [0, 0])


@pytest.mark.parametrize("bad_hex, message", [
    ("789c731c030000000001", "invalid literal/length code"),
    ("789c73043e00000001", "invalid distance code"),
], ids=["reserved-ll", "reserved-distance"])
def test_small_literal_reserved_code_queued_recovery(
    cuda_device, monkeypatch, bad_hex, message,
):
    import jax

    # Fixed literal A, then reserved LL286 or length 3 and reserved distance30.
    bad = bytes.fromhex(bad_hex)
    assert zlib.decompressobj().decompress(bad[:4]) == b"A"
    with pytest.raises(zlib.error, match=message):
        zlib.decompress(bad)
    raws = (b"recovery" * 4, (b"recovery" * 4)[::-1])
    payloads = [_compressed(raw) for raw in raws]
    assert all(zlib.decompress(payload) == raw
               for payload, raw in zip(payloads, raws))
    device = jax.devices("gpu")[cuda_device]
    sources = [jax.device_put(np.frombuffer(payload, np.uint8), device)
               for payload in [*payloads, bad]]
    decode = jax.jit(lambda value: codec.decompress_zlib_checked(value, len(raws[0]), device))
    order = (0, 2, 1, 2, 0)
    with _no_cpu_codec(monkeypatch):
        queued = [decode(sources[index]) for index in order]
        jax.block_until_ready(queued)
        for (data, metadata), index in zip(queued, order):
            assert metadata.devices() == {device}
            np.testing.assert_array_equal(np.asarray(metadata), [4 if index == 2 else 0, 0])
            wanted = b"A" + bytes(len(raws[0]) - 1) if index == 2 else raws[index]
            _assert_bytes(data, wanted, device)
