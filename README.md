# cuda-zlib

Experimental compression and decompression of general zlib byte streams on CUDA.
The core uses NumPy and CuPy. The optional JAX bridge also works with general
byte arrays. Version `0.1.0a1` is an alpha release;
performance and compatibility depend on the workload and CUDA environment.

```python
import cuda_zlib
encoded = cuda_zlib.compress_zlib(b"hello" * 1000, device=0)
decoded = cuda_zlib.decompress_zlib(encoded, 5000, device=0)
assert decoded.get().tobytes() == b"hello" * 1000
```

Both functions accept `bytes` or contiguous, one-dimensional CuPy `uint8`
arrays on the requested CUDA ordinal. They return independent CuPy allocations
and synchronize their private stream before returning. Input arrays must be
ready on the caller's current stream; writes from another stream require caller
synchronization. A per-device lock serializes private workspaces. Callers must
not mutate an input concurrently. No CPU codec fallback occurs.

`cuda_zlib.jax` provides explicit `device=` bridges returning independently
owned, completed JAX arrays. Compression also accepts a JAX `uint8` array through
stream-aware DLPack. The core does not import JAX; base package import
does not initialize CUDA. `compile_kernels(0)` loads both kernel modules for
separately measured startup.

## Installation

For the tested CUDA 12 / CuPy 13 configuration, install the release wheel:

```sh
python -m pip install "cuda-zlib[cuda12] @ https://github.com/xangma/cuda-zlib/releases/download/v0.1.0a1/cuda_zlib-0.1.0a1-py3-none-any.whl"
```

Or install a tagged source checkout:

```sh
git clone --branch v0.1.0a1 https://github.com/xangma/cuda-zlib.git
cd cuda-zlib
python -m pip install ".[cuda12]"
```

Python 3.10 or later and a CUDA-capable NVIDIA GPU are required for codec calls.
If you already have a matching CuPy installation, install the base package with
`python -m pip install .` and keep exactly one CuPy distribution. This release is
available through GitHub; no PyPI release has been published.
The optional `cuda12` extra targets the tested CuPy 13.x CUDA 12 configuration;
the `jax` extra supplies JAX without choosing or installing its GPU runtime.
Use [JAX's GPU installation instructions](https://docs.jax.dev/en/latest/installation.html)
for your machine. No CuPy wheel is
provided for macOS; framing and optional-backend tests still run there.

The decoder supports fresh RFC 1950 zlib streams with stored, fixed, and dynamic
DEFLATE blocks, including cross-block history. It checks CMF/FLG, advertised
window, exact output size, exact stream extent, and original Adler32. It rejects
gzip, raw DEFLATE, dictionaries, concatenation, and trailing bytes. This is a
bounded subset, not a complete implementation of all RFC 1950 features.

Input/output bounds are 256 MiB, candidate and block limits are 262144, and the
private workspace pool is capped at 3 GiB per device. Compression conservatively
requires its worst-case encoded extent to fit that decoder limit. Each 256–65535
byte chunk has fresh LZ77 history and one hash candidate. Exact bit costs select
stored, fixed or dynamic Huffman coding; stored blocks bound expansion. There
are no zlib compression levels or dictionaries. Long fixed regions use fresh
GPU token-boundary summaries only when the actual stream requires them. Stored
copying uses a separate parallel kernel; dynamic blocks retain parallel decoding.
Cold NVRTC compilation is additional to warmed measurements.

See [BENCHMARKS.md](BENCHMARKS.md) for measured compression ratios, startup costs,
warm timings and CPU baselines. Compression remains experimental; measurements
depend on the workload and CUDA environment and do not establish a general
compatibility or performance guarantee.

`CodecError` means malformed input, integrity failure, or a codec workspace limit.
`UnsupportedStream` means an unsupported wrapper feature. `BackendUnavailable`
means an absent optional backend or invalid device argument. A nonexistent
nonnegative CUDA ordinal raises the underlying CuPy/CUDA error. Other argument
errors use `TypeError`/`ValueError`; unexpected CuPy/CUDA runtime errors propagate.

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

[MIT](LICENSE). The code was originally developed for xangma in an unmerged
development branch and is released independently with the owner's authorization.
See [PROVENANCE.md](PROVENANCE.md). Dependencies retain their own licenses.
