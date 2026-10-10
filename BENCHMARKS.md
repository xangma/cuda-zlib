# General byte-stream benchmarks

Four timing scopes are reported separately on an RTX 3090 and an AMD
Threadripper PRO 3995WX. CPU comparisons use single-threaded stdlib zlib on the
same host. Compiled resident measurements exclude transfers and host status
checks; host-byte measurements include packing, transfers and byte copies.

| Dataset | Inputs | CUDA timing scope | CPU comparison |
| --- | --- | --- | --- |
| Single streams | 64 KiB, 1 MiB, 64 MiB; five workloads | Eager exact-length APIs; resident and host outputs reported separately | Same-host CPU, with matching host-byte outputs for speedup |
| Consumer output formats | 64 KiB, 1 MiB, 64 MiB; float32 | Synchronous host-input decode returning an array, memoryview or bytes | Same-host CPU with matching output formats |
| Independent small files | 256 B, 4 KiB, 64 KiB; 1, 8, 32, 128 files | Compiled resident calls and synchronous host-byte calls | Same-host CPU processing the same files |
| Resident checked decode | 64 KiB–64 MiB; five workloads | One completed checked `jax.jit` call | None; no CPU speedup inferred |

[Nsight profiles](PROFILING.md) provide compression and decompression timelines
with kernel stages, detailed host phases, transfers, CPU and memory samples.
These diagnostic captures are separate from the benchmark samples.

Compression levels are not equivalent across codecs. Read throughput alongside
encoded size. These synthetic workloads were measured on a shared workstation.
Device snapshots do not establish isolation. The measurements do not identify
an exact crossover or measure parallel CPU compression.

The compressor divides input into independent **32 KiB chunks** by default,
which can be encoded concurrently. External DEFLATE streams can contain larger
compressed blocks, variable-length Huffman codes and back-references into
earlier output, including across block boundaries up to 32 KiB away.
[RFC 1951](https://www.rfc-editor.org/rfc/rfc1951.html#section-2) describes these
block and history rules. Those parsing and history dependencies can limit the
parallel work available during decoding, so results are separated by stream
layout and timing scope.

Eligible dense serial blocks use concurrent token parsing and reference-root
emission in one CUDA kernel on XLA's stream. A shared FIFO has two slots of
32 tokens each. A conservative whole-stream cost gate keeps literal-heavy,
short and mixed serial workloads on the standard emitter. The FIFO uses fixed
shared memory; high-expansion streams also use the token-count workspace
reported below.

## Single-stream results on RTX 3090

For **64 MiB** inputs, warm host-byte compression measured **4.65–12.85×** CPU
zlib level-1 throughput. Decompression of identical stdlib level-6 streams
measured **1.34–4.88×** CPU throughput. Both comparisons include GPU uploads,
downloads, allocations, codec status checks and conversion to Python `bytes`.
At **64 KiB**, CPU compression and decompression were faster for every workload.
At **1 MiB**, host-byte compression measured **0.90–4.48×** CPU level-1
throughput. CUDA was faster for four workloads; CPU was faster for generated
text (**0.90×**). Decompression was faster on CUDA for zeros
(**1.32×**) and Gaussian float32 (**1.14×**). CPU won the other three
workloads.

### GPU versus CPU, including transfers

**Speedup = CPU median elapsed time / CUDA median elapsed time.** Above 1× favors
CUDA; below 1× favors CPU. Compression compares `cuda_compress_host_host` with
CPU compression. Decompression compares `cuda_decompress_level6_host_host` with
CPU decoding the identical level-6 stream. The host-array workflow returns a
read-only NumPy view of pinned storage; the host-byte workflow additionally
calls `.tobytes()`. CPU speedup uses matching Python `bytes` outputs.

![64 MiB host-byte speedup over same-host CPU on RTX 3090](benchmarks/figures/cpu-speedup.png)

[SVG](benchmarks/figures/cpu-speedup.svg) · [PDF](benchmarks/figures/cpu-speedup.pdf)

### 64 MiB speedup

| Workload | Compression vs CPU level 1 | Compression vs CPU level 6 | Decompression vs CPU, level-6 stream |
| --- | ---: | ---: | ---: |
| Zero bytes | 10.29× | 23.82× | 4.30× |
| Generated text | 4.65× | 14.87× | 2.35× |
| Integer counters | 5.87× | 42.10× | 3.95× |
| Gaussian float32 | 12.85× | 14.32× | 4.88× |
| Uniform random bytes | 10.62× | 10.61× | 1.34× |

### Smaller inputs

These ratios have the same host-byte scope; compression uses CPU level 1 and
decompression uses identical level-6 streams.

| Workload | 64 KiB compression | 1 MiB compression | 64 KiB decompression | 1 MiB decompression |
| --- | ---: | ---: | ---: | ---: |
| Zero bytes | 0.07× | 1.48× | 0.22× | 1.32× |
| Generated text | 0.05× | 0.90× | 0.03× | 0.13× |
| Integer counters | 0.09× | 1.46× | 0.02× | 0.49× |
| Gaussian float32 | 0.26× | 4.37× | 0.04× | 1.14× |
| Uniform random bytes | 0.28× | 4.48× | 0.04× | 0.46× |

Only the three recorded sizes were measured. Lines between points do not
identify intermediate performance or a crossover size.

### 64 MiB encoded size

Compressed bytes divided by input bytes; lower is better. Values over 100%
indicate expansion. CUDA uses 32768-byte chunks and has no compression level.

| Workload | CUDA | CPU level 1 | CPU level 6 |
| --- | ---: | ---: | ---: |
| Zero bytes | 0.153% | 0.436% | 0.097% |
| Generated text | 13.179% | 12.121% | 10.293% |
| Integer counters | 34.621% | 34.589% | 34.576% |
| Gaussian float32 | 92.629% | 92.887% | 92.617% |
| Uniform random bytes | 100.015% | 100.030% | 100.031% |

![Encoded size for 64 MiB inputs on RTX 3090](benchmarks/figures/encoded-size.png)

[SVG](benchmarks/figures/encoded-size.svg) · [PDF](benchmarks/figures/encoded-size.pdf)

### 64 MiB compression throughput

Values are **MiB/s of uncompressed input**. Resident inputs and outputs stay on
the GPU; eager API status checks and exact-length slicing remain included.
Host-byte timings include both transfers and output conversion. Allocations
and synchronization are included; startup and warmup compilation are excluded.

| Workload | CUDA resident | CUDA host bytes | CPU level 1 | CPU level 6 |
| --- | ---: | ---: | ---: | ---: |
| Zero bytes | 6078.3 | 3755.4 | 365.1 | 157.6 |
| Generated text | 925.6 | 798.7 | 171.7 | 53.7 |
| Integer counters | 381.8 | 288.0 | 49.1 | 6.8 |
| Gaussian float32 | 396.8 | 223.1 | 17.4 | 15.6 |
| Uniform random bytes | 631.1 | 270.9 | 25.5 | 25.5 |

![Compression throughput across measured sizes on RTX 3090](benchmarks/figures/compression-throughput.png)

[SVG](benchmarks/figures/compression-throughput.svg) · [PDF](benchmarks/figures/compression-throughput.pdf)

### 64 MiB decompression throughput

Values are **MiB/s of decoded output**. Each CPU/CUDA pair decodes the same
compressed bytes. Codec-produced and stdlib level-6 streams have different
layouts and are shown separately. Host-array output and host-byte output also
have separate timing scopes.

| Workload | CUDA resident, codec stream | CPU, codec stream | CUDA resident, level-6 stream | CUDA host array, level-6 stream | CUDA host bytes, level-6 stream | CPU, level-6 stream |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Zero bytes | 9204.7 | 178.6 | 3434.2 | 3195.7 | 775.9 | 180.5 |
| Generated text | 3276.7 | 242.0 | 2473.2 | 2138.3 | 694.8 | 295.7 |
| Integer counters | 2062.1 | 168.7 | 2376.6 | 1964.7 | 667.6 | 169.0 |
| Gaussian float32 | 1475.7 | 122.8 | 1594.4 | 1301.8 | 583.9 | 119.7 |
| Uniform random bytes | 19887.8 | 652.1 | 20300.8 | 4943.1 | 866.7 | 648.7 |

![Decompression of identical stdlib level-6 streams on RTX 3090](benchmarks/figures/decompression-throughput.png)

[SVG](benchmarks/figures/decompression-throughput.svg) · [PDF](benchmarks/figures/decompression-throughput.pdf)

![Eager resident decoding by compressed stream layout at 64 MiB on RTX 3090](benchmarks/figures/decode-stream-layout.png)

[SVG](benchmarks/figures/decode-stream-layout.svg) · [PDF](benchmarks/figures/decode-stream-layout.pdf)

The stream-layout figure compares current workflows on identical inputs within
each CPU/CUDA pair. It is separate from the checked JIT dataset below.

[Raw 15-case results](benchmarks/results/rtx3090-20261010.json) contain every sample, encoded size,
payload/stream hash, startup measurement and environment record. There are
15 CUDA samples and five CPU samples per workflow after one untimed warmup.
All timed calls complete; the **last returned output from each workflow** is
checked outside timing against the original bytes. Stdlib decodes CUDA output
and CUDA decodes independent level-6 streams. A failed check stops the run.

## Consumer output formats on RTX 3090

These measurements start with **host compressed bytes** and finish with the
requested completed host object. CUDA uses `decompress_zlib_host`, including
input validation, upload, native decoding/status checks and a pinned-host
download. Returning its array or `memoryview(array)` shares that storage;
requesting `array.tobytes()` additionally allocates and copies the decoded data.
CPU uses fresh `zlib.decompress` output for every call. Its arrays and memoryviews
share the resulting immutable Python bytes without another output copy.

Values are median **milliseconds**, lower is better. All six series decode
identical stdlib level-6 streams containing Gaussian float32 bytes.

| Raw size | CPU array | CUDA array | CPU memoryview | CUDA memoryview | CPU bytes | CUDA bytes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 64 KiB | 0.460 | 11.241 | 0.458 | 11.282 | 0.464 | 11.343 |
| 1 MiB | 7.249 | 6.467 | 7.233 | 6.468 | 7.234 | 6.546 |
| 64 MiB | 564.304 | 51.919 | 564.239 | 51.968 | 564.283 | 112.541 |

At 64 MiB, CUDA array/memoryview output takes **46% of the time** required
for CUDA bytes output. The CUDA implementation is identical across these output
formats. This benefit applies to consumers that accept the shared buffer;
callers requiring Python bytes still pay its allocation/copy cost. CPU is
faster at 64 KiB for every output format. CUDA is faster at 1 MiB and 64 MiB for
this float32 fixture; these results do not establish a general crossover.

![Completed host-output decompression latency by consumer format](benchmarks/figures/host-outputs/host-output-latency.png)

[SVG](benchmarks/figures/host-outputs/host-output-latency.svg) ·
[PDF](benchmarks/figures/host-outputs/host-output-latency.pdf) ·
[Raw samples](benchmarks/results/host-outputs/rtx3090-float32.json)

Each series has two untimed warmups and 12 samples, interleaved in seeded,
randomized rounds. Every one of the **252 outputs**, including warmups, is
checked outside timing; array/view checks do not construct Python bytes.
Fixture generation, native initialization, output release, oracle comparisons
and file I/O are excluded. The figure displays all 216 measured observations,
with independent latency scales for each size. Measurements used the shared
RTX 3090 workstation, JAX/JAXlib 0.11.2, NumPy 2.2.5 and zlib 1.3.1, at source
[`d323fee`](https://github.com/xangma/cuda-zlib/commit/d323fee215658d2c03731c2e282b395860a27ac8).
The report records runtime/helper/native hashes, the selected GPU UUID and
before/after resource snapshots; snapshots do not prove isolation throughout.

Reproduce with a fresh output path:

```sh
PYTHONPATH=src CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false \
python benchmarks/host_outputs.py --source-revision "$(git rev-parse HEAD)" \
  --sizes 65536 1048576 67108864 --workload float32 --samples 12 --warmups 2 \
  --output /tmp/cuda-zlib-host-outputs.json
python benchmarks/plot_host_outputs.py --input /tmp/cuda-zlib-host-outputs.json \
  --output-dir /tmp/cuda-zlib-host-output-figures
```

The [file consumer example](examples/decompress_file.py) passes a memoryview
directly to unbuffered file writes. For numerical streams,
`np.frombuffer(host, dtype=...)` shares the decoded buffer when the recorded
dtype and byte order are supplied. Retaining either view keeps the host storage
alive. File I/O remains outside the timing scope above.

## Independent small files on RTX 3090

**CPU host-byte workflows were faster for every measured single-file small-input
case** (256 B, 4 KiB and 64 KiB) than the matching CUDA host-byte workflows.
With **128 independent 64 KiB files**, host-byte compression measured
**2.33–12.66×** CPU level-1 throughput. Host-byte decompression measured
**12.65×** CPU for zeros, **2.50×** for text and **1.04×** for
random bytes. The random-byte result is close to parity.

Packing independent streams shares dispatch costs. Each file has its own
wrapper, history and checksum; batch decoding assigns one CUDA warp per file.
The resident single-file comparison is one warmed `jax.jit` containing
independent FFI calls; the packed workflow
uses one batch FFI call. Host single-file timings use a Python loop of
synchronous APIs. Host batch APIs reuse compiled calls for static layouts.

### Amortized latency with 128 files

Values are **microseconds per file = total completed call time / 128**, rather
than individual-file latency. Lower is better. CPU and CUDA host workflows
start and finish with Python `bytes`; CUDA includes packing, transfers, status
checks and output copies. Resident buffers and metadata stay on the GPU.
Decompression uses identical independent stdlib level-6 streams.

| File size | Workload | CPU compression, level 1 | CUDA batch compression, resident | CUDA batch compression, host bytes | CPU decompression | CUDA batch decompression, resident | CUDA batch decompression, host bytes |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 256 B | Zero bytes | 4.20 | 1.48 | 8.18 | 1.55 | 1.74 | 7.62 |
| 256 B | Generated text | 10.16 | 2.79 | 8.96 | 3.93 | 2.09 | 7.78 |
| 256 B | Uniform random bytes | 37.63 | 4.56 | 10.73 | 0.71 | 1.17 | 7.49 |
| 4 KiB | Zero bytes | 9.63 | 2.50 | 11.04 | 17.95 | 1.74 | 8.51 |
| 4 KiB | Generated text | 24.38 | 7.46 | 16.23 | 10.81 | 3.58 | 10.42 |
| 4 KiB | Uniform random bytes | 122.40 | 10.25 | 19.25 | 2.92 | 1.26 | 9.48 |
| 64 KiB | Zero bytes | 129.33 | 13.35 | 55.55 | 249.27 | 4.01 | 19.71 |
| 64 KiB | Generated text | 309.71 | 85.24 | 131.25 | 113.29 | 27.05 | 45.31 |
| 64 KiB | Uniform random bytes | 2160.99 | 122.82 | 170.63 | 42.04 | 7.26 | 40.58 |

### Small-file plots

All six figure families show sizes 256 B, 4 KiB and 64 KiB at counts 1, 8, 32 and
128. Points are medians of seven completed calls; whiskers show sample minimum
and maximum. Lines connect measured counts. Resident calls complete output and
metadata before timing ends; host status checks and oracle checks follow.
Host APIs validate codec status within their timed calls. The first native build
and per-layout XLA compilation are excluded from steady-state timings.

![Zero bytes: amortized compression/decompression latency, host bytes](benchmarks/figures/small-batch/rtx3090-zeros-host-bytes.png)

[SVG](benchmarks/figures/small-batch/rtx3090-zeros-host-bytes.svg) · [PDF](benchmarks/figures/small-batch/rtx3090-zeros-host-bytes.pdf)

![Zero bytes: amortized compression/decompression latency, resident](benchmarks/figures/small-batch/rtx3090-zeros-resident.png)

[SVG](benchmarks/figures/small-batch/rtx3090-zeros-resident.svg) · [PDF](benchmarks/figures/small-batch/rtx3090-zeros-resident.pdf)

![Generated text: amortized compression/decompression latency, host bytes](benchmarks/figures/small-batch/rtx3090-text-host-bytes.png)

[SVG](benchmarks/figures/small-batch/rtx3090-text-host-bytes.svg) · [PDF](benchmarks/figures/small-batch/rtx3090-text-host-bytes.pdf)

![Generated text: amortized compression/decompression latency, resident](benchmarks/figures/small-batch/rtx3090-text-resident.png)

[SVG](benchmarks/figures/small-batch/rtx3090-text-resident.svg) · [PDF](benchmarks/figures/small-batch/rtx3090-text-resident.pdf)

![Uniform random bytes: amortized compression/decompression latency, host bytes](benchmarks/figures/small-batch/rtx3090-random-host-bytes.png)

[SVG](benchmarks/figures/small-batch/rtx3090-random-host-bytes.svg) · [PDF](benchmarks/figures/small-batch/rtx3090-random-host-bytes.pdf)

![Uniform random bytes: amortized compression/decompression latency, resident](benchmarks/figures/small-batch/rtx3090-random-resident.png)

[SVG](benchmarks/figures/small-batch/rtx3090-random-resident.svg) · [PDF](benchmarks/figures/small-batch/rtx3090-random-resident.pdf)

[Raw 36-case results](benchmarks/results/small-batch-rtx3090-20261010.json) include seven samples per workflow,
encoded lengths/hashes and an optional compiled padded batch round trip.
Every final returned workflow output passed byte checks; compressed streams
were decoded by stdlib and resident metadata/padding were checked after timing.
Round-trip encoded lengths stay on device; its timings are recorded, not plotted.
The report's `complete` marker is true only after the whole matrix finishes.

## Resident checked decoding on RTX 3090

These measurements use one warmed `jax.jit` checked decode of an independent
stdlib level-6 stream. The input, output and final metadata stay on the device;
completion waits for both returned arrays. Timings exclude compilation, initial
uploads and post-call host status/byte checks. GPU codec validation and temporary
allocations are included. This timing scope differs from the host-byte and eager
API measurements elsewhere; no CPU speedup is inferred from this dataset.

| Stream size | Workload | Compressed bytes | Completed latency (ms) | Throughput (MiB/s) |
| --- | --- | ---: | ---: | ---: |
| 64 KiB | Zero bytes | 84 | 0.451 | 138.5 |
| 64 KiB | Generated text | 7064 | 3.296 | 19.0 |
| 64 KiB | Integer counters | 22701 | 12.463 | 5.0 |
| 64 KiB | Gaussian float32 | 60709 | 10.184 | 6.1 |
| 64 KiB | Uniform random bytes | 65562 | 0.330 | 189.1 |
| 128 KiB | Zero bytes | 149 | 0.548 | 228.2 |
| 128 KiB | Generated text | 13844 | 6.209 | 20.1 |
| 128 KiB | Integer counters | 45357 | 8.452 | 14.8 |
| 128 KiB | Gaussian float32 | 121355 | 4.985 | 25.1 |
| 128 KiB | Uniform random bytes | 131118 | 0.425 | 294.4 |
| 256 KiB | Zero bytes | 277 | 0.821 | 304.5 |
| 256 KiB | Generated text | 27438 | 12.121 | 20.6 |
| 256 KiB | Integer counters | 90669 | 8.450 | 29.6 |
| 256 KiB | Gaussian float32 | 242724 | 5.035 | 49.7 |
| 256 KiB | Uniform random bytes | 262230 | 0.425 | 588.8 |
| 1 MiB | Zero bytes | 1039 | 2.294 | 435.9 |
| 1 MiB | Generated text | 108353 | 13.852 | 72.2 |
| 1 MiB | Integer counters | 362577 | 8.903 | 112.3 |
| 1 MiB | Gaussian float32 | 970987 | 5.394 | 185.4 |
| 1 MiB | Uniform random bytes | 1048902 | 0.490 | 2040.2 |
| 8 MiB | Zero bytes | 8163 | 15.590 | 513.2 |
| 8 MiB | Generated text | 863857 | 14.580 | 548.7 |
| 8 MiB | Integer counters | 2900368 | 9.225 | 867.2 |
| 8 MiB | Gaussian float32 | 7768654 | 7.575 | 1056.1 |
| 8 MiB | Uniform random bytes | 8391174 | 0.735 | 10891.1 |
| 64 MiB | Zero bytes | 65238 | 16.625 | 3849.7 |
| 64 MiB | Generated text | 6907553 | 26.072 | 2454.7 |
| 64 MiB | Integer counters | 23203543 | 26.909 | 2378.4 |
| 64 MiB | Gaussian float32 | 62154385 | 40.809 | 1568.3 |
| 64 MiB | Uniform random bytes | 67129345 | 2.559 | 25006.8 |

![Resident checked decode latency and throughput on RTX 3090](benchmarks/figures/resident-checked.png)

[SVG](benchmarks/figures/resident-checked.svg) · [PDF](benchmarks/figures/resident-checked.pdf)

Medians use 31 completed calls after all shapes are warmed; whiskers show sample
minimum/maximum. Axes are logarithmic. The final returned sample in each case
passed status and independent byte checks; CPU codec functions are forbidden
during every timed CUDA call.
[Raw samples and hashes](benchmarks/results/resident-checked-rtx3090-20261010.json).

Measured 2026-10-10 (Europe/London) on RTX 3090, driver 610.57.04, JAX/JAXlib 0.11.2,
Python 3.13.3 and nvcc 12.6.85. Package and harness hashes match
[d323fee](https://github.com/xangma/cuda-zlib/tree/d323fee215658d2c03731c2e282b395860a27ac8).
The raw report identifies the loaded native library with its cache key, library
and build hashes, source/header hashes, compiler, flags and FFI targets.
The resident report's package and harness hashes and loaded-native identity
bind the measurements to that checkout.
The private workspace pool ended with
576 MiB retained and zero live scratch, using a 1 GiB release threshold.
Pool reservation excludes JAX inputs and outputs.
See the [workspace bounds](README.md#installation).
The workstation was shared; device snapshots do not establish isolation.

```sh
git checkout d323fee215658d2c03731c2e282b395860a27ac8
CUDACXX=/path/to/nvcc CUDA_ZLIB_CACHE_DIR=/path/to/codec-cache \
  XLA_PYTHON_CLIENT_PREALLOCATE=false \
  python benchmarks/profile_resident.py --sizes 65536 131072 262144 1048576 8388608 67108864 \
  --workloads zeros text uint32 float32 random --seed 20261008 --samples 31 --output resident.json
python benchmarks/plot_resident.py resident.json --output-dir resident-figures
```

## Workloads

The general and small-file datasets use seed `20261008`. Small-file payloads
use `seed + file_index`; all other payloads use the same seed for each workload.
The text and zero generators ignore the seed, so files within each text/zero
batch contain identical bytes. Random files within a batch have distinct seeded
contents. Each measured batch uses one workload and one file size.

- `zeros`: all zero bytes.
- `text`: generated ASCII request records with varying fields; an 8192-record
  block repeats to fill larger inputs.
- `uint32`: ascending little-endian unsigned 32-bit counters.
- `float32`: seeded Gaussian samples encoded as little-endian float32.
- `random`: seeded uniform bytes.

Generation, initial resident uploads and post-timing oracle checks are outside
timed regions. Runtime codec validation and allocations are included. Payload
and encoded-stream hashes establish the exact inputs in the raw reports.

## Recorded CUDA results

The general and small-file runs were measured on 2026-10-10 (Europe/London)
using an NVIDIA GeForce RTX 3090 and an AMD Ryzen Threadripper PRO 3995WX 64-Cores on Linux x86_64,
Python 3.13.3, NumPy 2.2.5,
JAX/JAXlib 0.11.2 and stdlib zlib 1.3.1. Driver version was 610.57.04.
JAX reported CUDA platform `cuda 12060`; the native codec compiler was
`nvcc` 12.6.85 targeting `sm_86`. The JAX platform string and native compiler
version describe different components.
File dates use Europe/London; `created_utc` timestamps in the raw reports use UTC.

Measured checkout: [d323fee](https://github.com/xangma/cuda-zlib/tree/d323fee215658d2c03731c2e282b395860a27ac8).
Reports record `source_revision`, `harness_sha256` and codec source hashes.
`environment.native_build` identifies the **actually loaded library** with its
cache key, library/build SHA-256 hashes, build identity, FFI targets, compiler,
flags and source/header hashes. The compatible `native_builds` field contains
that single build identity. These records identify the loaded backend separately
from the benchmark harness. Figure manifests pin their source report and exports.

All four datasets identify native library SHA-256
`83420785b15e2fb659e87008bb994fa4572fec4c510f4ba406ebd32db582c405`,
cache key `f23e64aefe987fab7f182221c3db7ca4f512676ec9874e2ee1de3e72e8691d4e`
and build SHA-256 `a1f169c838f67397cdb7a6052f1b98f0bea1558dd32b6df7c661689ef237d68d`.

Both runs reused the matching native library in a task-specific cache. The recorded
`compile_kernels` time measures cached loading and FFI registration, rather than
a cold build. Startup and final private-pool accounting:

| Run | Imports and CUDA init (s) | Native load and registration (s) | Reserved workspace after run (MiB) | Live scratch after run (bytes) |
| --- | ---: | ---: | ---: | ---: |
| Single streams | 1.144 | 0.010 | 576 | 0 |
| Small-file batches | 1.160 | 0.010 | 64 | 0 |

The release threshold was **1 GiB**. Reserved pages permit reuse and remain
allocated after scratch is freed; this is a retention policy, not a memory cap.
Pool accounting excludes JAX input/output and pinned-host buffers. For the
high-expansion route, additional token-count scratch is
`4 * (candidate_capacity + block_capacity + 1)` bytes; it is separate from the
fixed shared FIFO and included in private-pool accounting. The general and
small-file runs used `XLA_PYTHON_CLIENT_PREALLOCATE=false`. The workstation
was shared; device snapshots do not establish isolation.

Separate [Apple M4 Max CPU reference measurements](benchmarks/CPU_BASELINES.md)
were recorded on 2026-10-06 on macOS. They retain their own environment and are
not GPU speedup baselines.

## Reproduce

Check out the recorded harness and codec, and use a compatible GPU JAX runtime.
The recorded Python and NumPy versions were 3.13.3 and 2.2.5; JAX/JAXlib were
0.11.2. Stdlib zlib's version depends on the Python build.

```sh
git clone https://github.com/xangma/cuda-zlib.git
cd cuda-zlib
git checkout d323fee215658d2c03731c2e282b395860a27ac8
python -m pip install . matplotlib
export CUDA_ZLIB_CACHE_DIR=/path/to/codec-cache
CUDACXX=/path/to/nvcc XLA_PYTHON_CLIENT_PREALLOCATE=false \
  python -c "import cuda_zlib; cuda_zlib.compile_kernels(0)"
CUDACXX=/path/to/nvcc XLA_PYTHON_CLIENT_PREALLOCATE=false \
  CUDA_ZLIB_WORKSPACE_RETENTION_BYTES=1073741824 \
  python benchmarks/benchmark.py --sizes 65536 1048576 67108864 \
  --workloads zeros text uint32 float32 random --samples 15 --cpu-samples 5 \
  --seed 20261008 --device 0 --output results-cuda.json
python benchmarks/plot_results.py --input results-cuda.json
CUDACXX=/path/to/nvcc XLA_PYTHON_CLIENT_PREALLOCATE=false \
  CUDA_ZLIB_WORKSPACE_RETENTION_BYTES=1073741824 \
  python benchmarks/small_batch.py --sizes 256 4096 65536 --counts 1 8 32 128 \
  --workloads zeros text random --samples 7 --seed 20261008 --device 0 \
  --roundtrip --output small-batch.json
python benchmarks/plot_small_batch.py small-batch.json --output-dir small-batch-figures
```

Install `".[cuda12]"` when selecting JAX's CUDA 12 runtime for a new environment;
the toolkit compiler is installed separately. Set `CUDACXX` to its `nvcc` path.
The native cache key covers codec sources, FFI headers, compiler, architecture,
flags and JAXlib version. The prebuild call above prepares the matching library
before timing its cached load; an empty cache instead includes native compilation
in startup. Native load/build and per-shape XLA compilation are excluded from
steady-state measurements. Each workflow has one untimed warmup.
For a source export without `.git`, set `CUDA_ZLIB_SOURCE_REVISION` to the full
checkout revision; it records supplied metadata, while source hashes identify
the actual files.

Measurement commands above pin their recorded harnesses. To regenerate the
published figures, use the **current checkout's renderers** and recorded JSON
reports, without rerunning CUDA:

```sh
python benchmarks/plot_results.py --input benchmarks/results/rtx3090-20261010.json
python benchmarks/plot_small_batch.py benchmarks/results/small-batch-rtx3090-20261010.json \
  --output-dir benchmarks/figures/small-batch --prefix rtx3090
python benchmarks/plot_resident.py benchmarks/results/resident-checked-rtx3090-20261010.json \
  --output-dir benchmarks/figures
python benchmarks/verify_figures.py
```

The renderers produce PNG, SVG and PDF exports; their versions and source hashes
are recorded in figure manifests. The figure manifests record the rendering environment separately from the
NumPy 2.2.5 benchmark environment. General
figures are in [benchmarks/figures](benchmarks/figures), small-file figures in
[benchmarks/figures/small-batch](benchmarks/figures/small-batch).
Use `--validate-only` to check a report without writing figures.

For CPU-only measurements without JAX or CUDA:

```sh
python -m pip install numpy
python benchmarks/benchmark.py --cpu-only --cpu-samples 5 \
  --seed 20261008 --output results-cpu.json
python benchmarks/small_batch.py --cpu-only --samples 7 \
  --seed 20261008 --output small-batch-cpu.json
```

For a focused general smoke check, use `--sizes 65536 --samples 1 --cpu-samples 1`.
Choose workload and output type before comparing timings; these measurements
cover the documented bounded codec and do not establish universal performance.

## Profiling

Capture warmed resident checked decompression with Nsight Systems:

```sh
CUDACXX=/path/to/nvcc python benchmarks/trace.py \
  --nsys /path/to/nsys --output traces/decode --cuda-profiler-single-range -- \
  python benchmarks/profile_resident.py --sizes 1048576 8388608 \
  --workloads zeros text random --cuda-profiler-range --output traces/decode.json
```

The workload warms every requested shape, then completes one decode per case
inside a single continuous profiler range. NVTX labels identify workload and
uncompressed size. Inputs stay resident; both output and status are completed on
the GPU. Status is checked before output bytes after capture, and CPU codec calls
are forbidden during decoding. Capture retains all outputs until the range ends,
so device memory grows with the sum of requested input and output sizes. Use
fewer cases per process when memory is limited. Startup, compilation, uploads,
oracle checks and file writes are outside the capture.

For ordinary completed resident timings, run the workload directly without
`--cuda-profiler-range` and choose `--samples`. These timings exclude initial
uploads and post-call host status/byte checks. They measure a checked JIT API
workflow and have a different scope from host-byte benchmarks above. Timings
under Nsight are diagnostic; use separate normal runs for performance comparisons.
`benchmarks/profile.py` also measures eager compression and both codec-produced
and stdlib streams; `trace.py --cuda-profiler-range` supports its separate ranges.

`trace.py` writes the Nsight report, SQLite export, log and a `.trace.json`
manifest containing the CLI version, capture mode and record counts. It requires
successful execution, imported GPU kernels and CUDA API records, and rejects
known import errors or a disconnected agent even when Nsight returns zero. Choose
a fresh output prefix for each run. The single-range mode stops collection after
one range while allowing the workload to finish its validation.

The default `--nvtx-domain-exclude=TSL` retains CUDA and other NVTX records while
omitting JAX TSL annotations for import compatibility on JAX 0.11.2 / Nsight
Systems 2026.1.3. On a compatible stack, `--nvtx-domain-exclude=""` includes all
domains. Capture workloads leave `environment.gpu_after` unset; collect device
snapshots after `trace.py` exits to avoid subprocesses during deferred export.
