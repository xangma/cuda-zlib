# Copyright (c) 2026 xangma
# SPDX-License-Identifier: MIT

"""Errors distinguish malformed input from unavailable/unsupported backends."""

class CodecError(ValueError):
    """Malformed stream, checksum/extent mismatch, or codec workspace limit."""


class BackendUnavailable(NotImplementedError):
    """The requested CUDA device or optional runtime is unavailable."""


class UnsupportedStream(NotImplementedError):
    """Valid wrapper features outside this bounded implementation's subset."""
