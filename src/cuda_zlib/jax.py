# Copyright (c) 2026 xangma
# SPDX-License-Identifier: MIT

"""Explicit JAX bridge. Copies are completed before returning independent arrays."""

from . import _codec
from ._errors import BackendUnavailable


def _admit(device):
    if (device is None or getattr(device, "platform", None) != "gpu"
            or "cuda" not in device.client.platform_version.lower()):
        raise BackendUnavailable("CUDA JAX device required")
    return int(device.local_hardware_id)


def to_jax_owned(array, device):
    """Copy a completed CuPy array into independent JAX ownership."""
    ordinal = _admit(device)
    if array.device.id != ordinal:
        raise ValueError("CuPy and JAX devices differ")
    try:
        import jax
        import jax.numpy as jnp
    except ImportError as exc:
        raise BackendUnavailable("JAX bridge requires JAX") from exc
    with jax.default_device(device):
        alias = jax.dlpack.from_dlpack(array, device=device, copy=False)
        result = jnp.array(alias, copy=True)
        result.block_until_ready()
    return result


def decompress_zlib(payload, expected_bytes, *, device):
    if isinstance(payload, bytes):
        _codec._framing(payload, expected_bytes)
    array = _codec.decompress_zlib(payload, expected_bytes, _admit(device))
    return to_jax_owned(array, device)


def compress_zlib(data, *, device, chunk_bytes=32768):
    ordinal = _admit(device)
    cp = _codec._cupy()
    if not isinstance(data, (bytes, cp.ndarray)):
        # JAX exports a stream-aware DLPack capsule; no input readback is needed.
        data = cp.from_dlpack(data)
    array = _codec.compress_zlib(data, ordinal, chunk_bytes=chunk_bytes)
    return to_jax_owned(array, device)
