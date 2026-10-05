# Copyright (c) 2026 xangma
# SPDX-License-Identifier: MIT

"""Proposed PyCBC adapter; copy here only during an approved migration."""

import numpy as np
from .gwf_jax import CompressionKind, GWFFormatError, _raw_kernel


class GWFDeflateUnavailable(NotImplementedError):
    """Optional CUDA zlib backend unavailable or unsupported."""


def _backend():
    try:
        import cuda_zlib
        from cuda_zlib import jax as bridge
    except ImportError as exc:
        raise GWFDeflateUnavailable("CUDA decoding requires cuda-zlib[jax]") from exc
    return cuda_zlib, bridge


def decompress_zlib(payload, expected_bytes, device=None):
    codec, bridge = _backend()
    if device is None:
        from pycbc import scheme
        device = getattr(scheme.mgr.state, "jax_device", None)
    try:
        # Preserve CPU admission and the old explicit optional-dependency error.
        # Ordinal mapping happens only after the backend is available.
        if device is not None and getattr(device, "platform", None) == "gpu":
            try:
                import cupy  # noqa: F401
            except ImportError as exc:
                raise GWFDeflateUnavailable(
                    "direct CUDA GWF decoding requires JAX and CuPy") from exc
        return bridge.decompress_zlib(payload, expected_bytes, device=device)
    except codec.CodecError as exc:
        raise GWFFormatError(str(exc)) from exc
    except codec.UnsupportedStream as exc:
        # Preserve the existing distinction for zlib preset dictionaries.
        if isinstance(payload, bytes) and len(payload) > 1 and payload[1] & 32:
            raise GWFFormatError(str(exc)) from exc
        raise GWFDeflateUnavailable(str(exc)) from exc
    except codec.BackendUnavailable as exc:
        raise GWFDeflateUnavailable(str(exc)) from exc


def compile_kernels(device):
    codec, bridge = _backend()
    return codec.compile_kernels(bridge._admit(device))


def decode_fr_vect_deflate(vector, device=None):
    if vector.compression.kind != CompressionKind.GZIP or vector.type_code == 8:
        raise GWFDeflateUnavailable("native CUDA codec requires numeric zlib")
    raw = decompress_zlib(vector.payload,
                          vector.n_data * np.dtype(vector.dtype).itemsize,
                          device=device)
    result = _raw_kernel(vector.type_code, vector.n_data,
                         vector.compression.endian)(raw)
    result.block_until_ready()
    return result
