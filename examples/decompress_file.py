#!/usr/bin/env python3
# Copyright (c) 2026 xangma
# SPDX-License-Identifier: MIT
"""Decode one bounded zlib stream and write its host buffer without tobytes()."""

import argparse
from pathlib import Path

import cuda_zlib


def decompress_file(source, destination, expected_bytes, device=0):
    """Validate on CUDA before creating a new output file; refuse overwrite.

    The memoryview retains the completed pinned host array throughout writing.
    File I/O still transfers data to the OS; no full-size Python bytes output
    is allocated. The codec's input and output size limits apply.
    """
    host = cuda_zlib.decompress_zlib_host(
        Path(source).read_bytes(), expected_bytes, device=device)
    view = memoryview(host)
    with Path(destination).open("xb", buffering=0) as output:
        while view:
            written = output.write(view)
            if written is None or written <= 0:
                raise OSError("output write made no progress")
            view = view[written:]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--expected-bytes", required=True, type=int)
    parser.add_argument("--device", default=0, type=int)
    args = parser.parse_args()
    decompress_file(args.source, args.destination, args.expected_bytes, args.device)


if __name__ == "__main__":
    main()
