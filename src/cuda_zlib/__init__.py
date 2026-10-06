# Copyright (c) 2026 xangma
# SPDX-License-Identifier: MIT

"""Synchronous CUDA byte codec; importing this package does not initialize CUDA."""

from ._codec import (compress_zlib, decompress_zlib, compile_kernels,
                     compress_zlib_padded, decompress_zlib_checked,
                     decompress_zlib_host)
from ._errors import BackendUnavailable, CodecError, UnsupportedStream

__all__ = ["compress_zlib", "decompress_zlib", "compile_kernels",
           "compress_zlib_padded", "decompress_zlib_checked", "decompress_zlib_host",
           "BackendUnavailable", "CodecError", "UnsupportedStream"]
__version__ = "0.1.0a1"
