# Copyright (c) 2026 xangma
# SPDX-License-Identifier: MIT

"""JAX device convenience API for the native CUDA FFI backend."""

from . import _codec
from ._errors import BackendUnavailable


def _admit(device):
    _codec._validate_device(device)
    if type(device) is int:
        raise BackendUnavailable("CUDA JAX device required")
    return int(device.local_hardware_id)


def decompress_zlib(payload, expected_bytes, *, device):
    _admit(device)
    return _codec.decompress_zlib(payload, expected_bytes, device)


def compress_zlib(data, *, device, chunk_bytes=32768):
    _admit(device)
    return _codec.compress_zlib(data, device, chunk_bytes=chunk_bytes)


def compress_zlib_padded(data, *, device, chunk_bytes=32768):
    _admit(device)
    return _codec.compress_zlib_padded(data, device, chunk_bytes=chunk_bytes)


def decompress_zlib_checked(payload, expected_bytes, *, device):
    _admit(device)
    return _codec.decompress_zlib_checked(payload, expected_bytes, device)
