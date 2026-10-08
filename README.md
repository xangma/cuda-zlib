# cuda-zlib

Experimental compression and decompression of general zlib byte streams on CUDA,
using JAX FFI. JAX owns input/output buffers and supplies the CUDA execution
stream. Version `0.1.0a1` is an alpha release; performance and compatibility
depend on the workload and CUDA environment.

```python
import numpy as np
import cuda_zlib

encoded = cuda_zlib.compress_zlib(b"hello" * 1000, device=0)
decoded = cuda_zlib.decompress_zlib(encoded, 5000, device=0)
assert np.asarray(decoded).tobytes() == b"hello" * 1000
```

Both functions accept `bytes` or one-dimensional JAX `uint8` arrays on the
requested CUDA device. `device=` accepts a CUDA ordinal or JAX CUDA device.
They return completed JAX arrays with independent output buffers and check
codec status before returning. No CPU codec fallback occurs. Package import
does not import JAX or initialize CUDA. `cuda_zlib.jax` also provides convenience
wrappers requiring an explicit JAX device.

For output on the host, use `decompress_zlib_host`:

```python
host = cuda_zlib.decompress_zlib_host(encoded, 5000, device=0)
view = memoryview(host)
assert view.readonly
assert view == b"hello" * 1000
```

It accepts the same inputs and validates the same stream features and limits as
`decompress_zlib`. It returns a completed, read-only, contiguous NumPy `uint8`
array. Nonempty outputs use JAX-owned pinned host memory. The array retains its
storage across subsequent calls. `memoryview(host)` shares that storage without
another copy; `host.tobytes()` allocates and copies into a Python `bytes` object.

For compiled workflows, use the fixed-shape interfaces:

```python
import jax

device = jax.devices("gpu")[0]
encode = jax.jit(lambda x: cuda_zlib.compress_zlib_padded(x, device))
buffer, metadata = encode(jax.device_put(np.zeros(4096, dtype=np.uint8), device))
length, status = np.asarray(metadata)
assert status == 0
stream = buffer[:int(length)]

decode = jax.jit(lambda x: cuda_zlib.decompress_zlib_checked(x, 4096, device))
output, metadata = decode(stream)
assert np.asarray(metadata)[0] == 0
```

Compression metadata is `[encoded_length, status]`; the unused buffer tail is
zero padded. Decompression metadata is `[status, 0]`. **Check status before using
output:** zero means success. The synchronous exact-length APIs perform this
check on the host and cannot be used inside `jax.jit`. Devices, chunk sizes and
expected output lengths must be static while tracing. These byte operations
have no automatic differentiation rule. Handler execution includes bounded
host control synchronization; a compiled call does not imply an entirely
asynchronous pipeline.

To process decoded bytes inside a compiled workflow, gate the consumer on the
device status and return that status to the caller:

```python
import jax.numpy as jnp

@jax.jit
def decode_and_sum(stream):
    output, metadata = cuda_zlib.decompress_zlib_checked(stream, 4096, device)
    value = jax.lax.cond(
        metadata[0] == 0,
        lambda x: x.astype(jnp.float32).sum(),
        lambda x: jnp.float32(-1),
        output,
    )
    return value, metadata
```

Both returned buffers are owned by JAX. Failed decode output is invalid;
the failure branch must return an explicit sentinel or error result without
processing those bytes. Check the returned metadata at the application boundary.

## Multiple files

Batch independent files to share dispatch, workspace and transfer costs:

```python
files = (b"first file" * 100, b"second file" * 500, b"")
sizes = tuple(map(len, files))
streams = cuda_zlib.compress_zlib_batch_host(files, device=0)
outputs = cuda_zlib.decompress_zlib_batch_host(streams, sizes, device=0)
assert tuple(view.tobytes() for view in outputs) == files
```

Files can have different lengths, including zero. Each gets its own zlib
wrapper, history and checksum. The host compression API returns a tuple of
`bytes`; host decompression returns read-only NumPy views of one packed pinned
host allocation. `compress_zlib_batch` and `decompress_zlib_batch` return tuples
of completed JAX arrays. Exact per-file JAX slicing adds dispatch work; prefer
packed outputs in compiled workflows.

Host batch APIs reuse compiled calls in bounded process caches keyed by device
and file sizes, plus chunk size for compression. The first call for each layout
includes XLA compilation; warm that layout before measuring steady throughput.

For `jax.jit`, supply one flat buffer and static file sizes:

```python
sizes = (4096, 256, 0)
chunk_bytes = 32768
capacities = tuple(n + 5 * max(1, (n + chunk_bytes - 1) // chunk_bytes) + 6
                   for n in sizes)

@jax.jit
def roundtrip(packed):
    streams, encoded = cuda_zlib.compress_zlib_batch_padded(
        packed, sizes, device, chunk_bytes=chunk_bytes)
    output, decoded = cuda_zlib.decompress_zlib_batch_checked(
        streams, capacities, sizes, device, encoded_metadata=encoded)
    return output, encoded, decoded

packed = jax.device_put(np.zeros(sum(sizes), dtype=np.uint8), device)
output, encoded, decoded = roundtrip(packed)
assert not np.any(np.asarray(encoded)[:, 1])
assert not np.any(np.asarray(decoded)[:, 0])
```

Compression places each stream in a fixed-capacity slot, with zero padding.
Its `(files, 2)` metadata rows contain `[encoded_length, status]`. Passing that
metadata to decompression keeps actual lengths on device. For externally
encoded streams, concatenate their exact bytes, supply their exact static
lengths as `input_sizes`, and omit `encoded_metadata`. Decoded files are packed
without padding; metadata rows are `[status, 0]`. Check every file's status
before using its output. Convenience APIs raise an error naming the first
failed file.

Batching supports at most 262144 files, with combined input and output bounds
of 256 MiB. Compression also bounds the combined chunk count and padded
capacity. Batch decoding assigns a CUDA warp to each file and is intended for
many small streams; use the single-stream API for large files. Small single
streams use fused finishing and decoding paths, with cooperative byte copies
and checksum reductions. DEFLATE bit parsing still follows stream order.

## Performance

On an RTX 4090 with an AMD Threadripper PRO 3995WX, warm **64 MiB** host-byte
compression measured **7.64–16.34×** the throughput of single-threaded CPU
zlib level 1. Decoding identical stdlib level-6 streams measured **1.15–5.24×**
CPU throughput. Both comparisons include uploads, downloads, allocations, codec
validation and conversion to Python `bytes`.

At **64 KiB**, CPU compression and decompression were faster for every workload.
At **1 MiB**, host-byte compression measured **1.02–5.18×** CPU level-1
throughput. Generated-text compression was near parity. Decompression was faster
on CUDA for zeros (**1.16×**) and Gaussian float32 (**1.31×**); CPU won the
other three workloads.
Compression sizes differ across codecs; CUDA has no zlib-equivalent level.

**CPU was faster for every measured single-file small-input case** (256 B, 4 KiB
and 64 KiB), including resident CUDA workflows.
With **128 independent 64 KiB files**, host-byte compression measured
**2.43–18.80×** CPU level-1 throughput. Host-byte decompression measured
**11.15×** CPU for zeros, **2.11×** for text and **0.94×** for random bytes.
See [small-file results and batch-count plots](BENCHMARKS.md#independent-small-files-on-rtx-4090).

For compiled consumers, [resident checked decode measurements](BENCHMARKS.md#resident-checked-decoding-on-rtx-4090)
report completed JIT latency without uploads or host status transfers. That
dataset has a separate timing scope and does not report CPU speedup.

![64 MiB host-byte GPU speedup over same-host CPU](benchmarks/figures/cpu-speedup.png)

[BENCHMARKS.md](BENCHMARKS.md) records the measured source, per-workload ratios,
compressed sizes, startup costs, memory accounting and reproduction commands.

## Installation

Install from a source checkout with CUDA-enabled JAX:

```sh
git clone https://github.com/xangma/cuda-zlib.git
cd cuda-zlib
python -m pip install ".[cuda12]"
```

Codec calls require Python 3.12 or later, Linux, an NVIDIA GPU, JAX/JAXlib 0.11.2 or later, and a
compatible CUDA toolkit (12.0 or later) containing `nvcc`. Set `CUDACXX=/path/to/nvcc` or
`CUDA_HOME=/path/to/cuda` when the compiler is not on `PATH`. The `cuda12` extra
installs JAX's CUDA 12 runtime; the toolkit compiler is installed separately.
The `jax` extra installs JAX without selecting its GPU runtime. For an existing
compatible GPU JAX installation, install the base package with
`python -m pip install .`. See [JAX GPU installation instructions](https://docs.jax.dev/en/latest/installation.html).
CPU-only framing and optional-backend tests can run without JAX or CUDA.

The native library builds lazily and persists in
`$XDG_CACHE_HOME/cuda-zlib` (default `~/.cache/cuda-zlib`). Set
`CUDA_ZLIB_CACHE_DIR` to choose another directory. Its cache key includes source,
FFI headers, JAXlib version, CUDA compiler, architecture and compiler options;
process and thread locks serialize cache construction. `compile_kernels(0)`
builds/loads the library explicitly. XLA compilation for each input/output shape
and workflow remains lazy; warm each workflow before measuring steady throughput.
No prebuilt native wheel or PyPI release is provided for this backend.

The decoder supports fresh RFC 1950 zlib streams with stored, fixed, and dynamic
DEFLATE blocks, including cross-block history. It checks CMF/FLG, advertised
window, exact output size, exact stream extent, and original Adler32. It rejects
gzip, raw DEFLATE, dictionaries, concatenation, and trailing bytes. This is a
bounded subset, not a complete implementation of all RFC 1950 features.

RFC 1950 framing, declared window and checksum parameters stay on the GPU in
checked decoding. Discovery and accepted-chain counts, conditional retries and
reference-resolution state also stay on device. The native checked path queues
bounded sorting and launches without intermediate counter copies or explicit
stream waits. Output and final status remain JAX device arrays for compiled
consumers; completing the returned arrays waits for all decoding and validation.

Input/output bounds are 256 MiB; candidate and block limits are 262144.
Each invocation uses its own temporary workspace, allocated and freed on XLA's
stream. Private CUDA pools reuse freed workspace allocations, with one pool per
CUDA context and device. The default release threshold is 1 GiB per pool;
it is a retention policy, not a memory limit. Set
`CUDA_ZLIB_WORKSPACE_RETENTION_BYTES=0` before loading the backend to disable
retention, or set another nonnegative byte count. Pool handles live for the
backend's process lifetime. `workspace_pool_stats(device)` reports retained and
live bytes; `trim_workspace_pool(device)` asks CUDA to release unused pages.
Complete outstanding calls first to make their scratch eligible for trimming.

Compression conservatively requires its worst-case encoded extent to
fit the decoder input limit. Each 256–65535 byte chunk has fresh LZ77 history and
one hash candidate. Exact bit costs select stored, fixed or dynamic Huffman
coding; stored blocks bound expansion. There are no zlib compression levels or
dictionaries. Long fixed regions use fresh GPU token-boundary summaries when
the actual stream requires them.

Compression caches match tokens in a workspace of four bytes per input byte
(256 MiB for a 64 MiB input). Decoding above 64 KiB reserves eight bytes per
declared output byte for reference buffers under the default limits, plus up to 128 MiB of fixed
tile summaries and bounded candidate/block metadata. Scratch is allocated before
the GPU determines the stream's status, including for eligible medium fast parses
and malformed input. The 256 MiB output limit can require 2 GiB of reference
scratch alone. Reference resolution and final error checks run on the device;
output bytes are gathered while calculating checksum partials. Large compressed
inputs use a speculative-prefix queue of about one eighth the compressed input
size. The queue is bounded; dense prefixes use a GPU fallback path. Allocation
failures remain possible even within the codec bounds.

`CodecError` means malformed input, integrity failure, or a codec workspace limit.
`UnsupportedStream` means an unsupported wrapper feature. `BackendUnavailable`
means an absent backend, native build/load failure or invalid/unavailable device.
Other argument errors use `TypeError`/`ValueError`; unexpected CUDA/XLA runtime
errors propagate.

## Validation

```sh
python -m pip install ".[test]"
python -m pytest -q tests
```

GitHub CI builds and tests the installed wheel on CPU runners. CUDA tests skip
when the GPU backend is unavailable; CPU CI does not establish GPU correctness.
Tests use stdlib zlib only as an independent oracle/baseline and forbid CPU codec
calls inside CUDA operations.

## License

[MIT](LICENSE), copyright xangma. See [PROVENANCE.md](PROVENANCE.md) for
ownership and dependency information. Dependencies retain their own licenses.
