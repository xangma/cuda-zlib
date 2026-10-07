# Copyright (c) 2026 xangma
# SPDX-License-Identifier: MIT

"""Synchronous CUDA byte codec; importing this package does not initialize CUDA."""

from ._codec import (compress_zlib, decompress_zlib, compile_kernels,
                     compress_zlib_padded, decompress_zlib_checked,
                     decompress_zlib_host, workspace_pool_stats, trim_workspace_pool)
from ._codec import (compress_zlib_batch_padded, decompress_zlib_batch_checked,
                     compress_zlib_batch, decompress_zlib_batch,
                     compress_zlib_batch_host, decompress_zlib_batch_host)
from ._errors import BackendUnavailable, CodecError, UnsupportedStream

__all__ = ["compress_zlib", "decompress_zlib", "compile_kernels",
           "compress_zlib_padded", "decompress_zlib_checked", "decompress_zlib_host",
           "workspace_pool_stats", "trim_workspace_pool",
           "compress_zlib_batch_padded", "decompress_zlib_batch_checked",
           "compress_zlib_batch", "decompress_zlib_batch",
           "compress_zlib_batch_host", "decompress_zlib_batch_host",
           "BackendUnavailable", "CodecError", "UnsupportedStream"]
__version__ = "0.1.0a1"
