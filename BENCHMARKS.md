# General byte-stream benchmarks

Three timing scopes are reported separately on an RTX 4090 and an AMD
Threadripper PRO 3995WX. CPU comparisons use single-threaded stdlib zlib on the
same host. Compiled resident measurements exclude transfers and host status
checks; host-byte measurements include packing, transfers and byte copies.

| Dataset | Inputs | CUDA timing scope | CPU comparison |
| --- | --- | --- | --- |
| Single streams | 64 KiB, 1 MiB, 64 MiB; five workloads | Eager exact-length APIs; resident and host outputs reported separately | Same-host CPU, with matching host-byte outputs for speedup |
| Independent small files | 256 B, 4 KiB, 64 KiB; 1, 8, 32, 128 files | Compiled resident calls and synchronous host-byte calls | Same-host CPU processing the same files |
| Resident checked decode | 64 KiB–64 MiB; five workloads | One completed checked `jax.jit` call | None; no CPU speedup inferred |

Compression levels are not equivalent across codecs. Read throughput alongside
encoded size. These synthetic workloads and one shared workstation do not
establish an exact crossover or measure parallel CPU compression.

## Single-stream results on RTX 4090

For **64 MiB** inputs, warm host-byte compression measured **7.55–16.53×** CPU
zlib level-1 throughput. Decompression of identical stdlib level-6 streams
measured **1.12–5.12×** CPU throughput. Both comparisons include GPU uploads,
downloads, allocations, codec status checks and conversion to Python `bytes`.
At **64 KiB**, CPU compression and decompression were faster for every workload.
At **1 MiB**, host-byte compression measured **1.04–5.20×** CPU level-1
throughput. Decompression was faster on CUDA only for Gaussian float32
(**1.12×**); CPU won the other four workloads.

### GPU versus CPU, including transfers

**Speedup = CPU median elapsed time / CUDA median elapsed time.** Above 1× favors
CUDA; below 1× favors CPU. Compression compares `cuda_compress_host_host` with
CPU compression. Decompression compares `cuda_decompress_level6_host_host` with
CPU decoding the identical level-6 stream. The host-array workflow returns a
read-only NumPy view of pinned storage; the host-byte workflow additionally
calls `.tobytes()`. CPU speedup uses matching Python `bytes` outputs.

![64 MiB host-byte speedup over same-host CPU on RTX 4090](benchmarks/figures/cpu-speedup.png)

[SVG](benchmarks/figures/cpu-speedup.svg) · [PDF](benchmarks/figures/cpu-speedup.pdf)

### 64 MiB speedup

| Workload | Compression vs CPU level 1 | Compression vs CPU level 6 | Decompression vs CPU, level-6 stream |
| --- | ---: | ---: | ---: |
| Zero bytes | 13.31× | 30.99× | 3.61× |
| Generated text | 7.55× | 24.02× | 2.13× |
| Integer counters | 8.97× | 64.15× | 4.16× |
| Gaussian float32 | 16.53× | 18.43× | 5.12× |
| Uniform random bytes | 12.28× | 12.23× | 1.12× |

### Smaller inputs

These ratios have the same host-byte scope; compression uses CPU level 1 and
decompression uses identical level-6 streams.

| Workload | 64 KiB compression | 1 MiB compression | 64 KiB decompression | 1 MiB decompression |
| --- | ---: | ---: | ---: | ---: |
| Zero bytes | 0.08× | 1.50× | 0.18× | 0.63× |
| Generated text | 0.07× | 1.04× | 0.03× | 0.06× |
| Integer counters | 0.13× | 1.73× | 0.02× | 0.46× |
| Gaussian float32 | 0.35× | 4.96× | 0.02× | 1.12× |
| Uniform random bytes | 0.36× | 5.20× | 0.04× | 0.33× |

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
| Gaussian float32 | 92.629% | 92.886% | 92.616% |
| Uniform random bytes | 100.015% | 100.030% | 100.031% |

![Encoded size for 64 MiB inputs on RTX 4090](benchmarks/figures/encoded-size.png)

[SVG](benchmarks/figures/encoded-size.svg) · [PDF](benchmarks/figures/encoded-size.pdf)

### 64 MiB compression throughput

Values are **MiB/s of uncompressed input**. Resident inputs and outputs stay on
the GPU; eager API status checks and exact-length slicing remain included.
Host-byte timings include both transfers and output conversion. Allocations
and synchronization are included; startup and warmup compilation are excluded.

| Workload | CUDA resident | CUDA host bytes | CPU level 1 | CPU level 6 |
| --- | ---: | ---: | ---: | ---: |
| Zero bytes | 11987.7 | 5812.4 | 436.8 | 187.6 |
| Generated text | 2128.7 | 1566.2 | 207.5 | 65.2 |
| Integer counters | 935.1 | 538.5 | 60.1 | 8.4 |
| Gaussian float32 | 891.4 | 345.2 | 20.9 | 18.7 |
| Uniform random bytes | 1360.0 | 376.2 | 30.6 | 30.8 |

![Compression throughput across measured sizes on RTX 4090](benchmarks/figures/compression-throughput.png)

[SVG](benchmarks/figures/compression-throughput.svg) · [PDF](benchmarks/figures/compression-throughput.pdf)

### 64 MiB decompression throughput

Values are **MiB/s of decoded output**. Each CPU/CUDA pair decodes the same
compressed bytes. Codec-produced and stdlib level-6 streams have different
layouts and are shown separately. Host-array output and host-byte output also
have separate timing scopes.

| Workload | CUDA resident, codec stream | CPU, codec stream | CUDA resident, level-6 stream | CUDA host array, level-6 stream | CUDA host bytes, level-6 stream | CPU, level-6 stream |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Zero bytes | 11942.2 | 209.2 | 2601.5 | 2319.9 | 772.2 | 213.7 |
| Generated text | 6590.1 | 283.0 | 2343.7 | 1999.3 | 753.9 | 353.4 |
| Integer counters | 3464.1 | 205.1 | 3761.9 | 2812.0 | 851.8 | 205.0 |
| Gaussian float32 | 2649.6 | 147.3 | 2871.8 | 1952.0 | 739.0 | 144.3 |
| Uniform random bytes | 6056.1 | 752.9 | 5478.1 | 2884.6 | 851.0 | 761.2 |

![Decompression of identical stdlib level-6 streams on RTX 4090](benchmarks/figures/decompression-throughput.png)

[SVG](benchmarks/figures/decompression-throughput.svg) · [PDF](benchmarks/figures/decompression-throughput.pdf)

![Eager resident decoding by compressed stream layout at 64 MiB on RTX 4090](benchmarks/figures/decode-stream-layout.png)

[SVG](benchmarks/figures/decode-stream-layout.svg) · [PDF](benchmarks/figures/decode-stream-layout.pdf)

The stream-layout figure compares current workflows on identical inputs within
each CPU/CUDA pair. It is separate from the checked JIT dataset below.

[Raw 15-case results](benchmarks/results/rtx4090-20261007.json) contain every sample, encoded size,
payload/stream hash, startup measurement and environment record. There are
15 CUDA samples and five CPU samples per workflow after one untimed warmup.
All timed calls complete; the **last returned output from each workflow** is
checked outside timing against the original bytes. Stdlib decodes CUDA output
and CUDA decodes independent level-6 streams. A failed check stops the run.

## Independent small files on RTX 4090

**CPU was faster for every measured single-file small-input case** (256 B, 4 KiB
and 64 KiB), including resident CUDA workflows.
With **128 independent 64 KiB files**, host-byte compression measured
**2.76–18.70×** CPU level-1 throughput. Host-byte decompression measured
**9.26×** CPU for zeros, **1.97×** for text and **0.96×** for random bytes.

Packing independent streams shares dispatch costs. Each file has its own
wrapper, history and checksum; batch decoding assigns one CUDA warp per file.
The resident single-file comparison
is one warmed `jax.jit` containing independent FFI calls; the packed workflow
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
| 256 B | Zero bytes | 3.47 | 1.17 | 7.48 | 1.26 | 1.68 | 7.13 |
| 256 B | Generated text | 8.39 | 2.26 | 7.65 | 3.24 | 1.87 | 7.24 |
| 256 B | Uniform random bytes | 31.01 | 3.70 | 10.39 | 0.58 | 0.91 | 6.78 |
| 4 KiB | Zero bytes | 7.89 | 2.02 | 8.32 | 14.68 | 1.58 | 7.95 |
| 4 KiB | Generated text | 20.08 | 5.51 | 13.67 | 8.78 | 3.37 | 9.84 |
| 4 KiB | Uniform random bytes | 99.90 | 7.90 | 17.19 | 2.39 | 1.14 | 8.71 |
| 64 KiB | Zero bytes | 105.39 | 4.79 | 38.18 | 203.19 | 4.07 | 21.95 |
| 64 KiB | Generated text | 253.16 | 29.00 | 69.04 | 92.35 | 27.43 | 46.97 |
| 64 KiB | Uniform random bytes | 1729.88 | 47.40 | 92.49 | 35.14 | 3.76 | 36.75 |

### Small-file plots

All six figure families show sizes 256 B, 4 KiB and 64 KiB at counts 1, 8, 32 and
128. Points are medians of seven completed calls; whiskers show sample minimum
and maximum. Lines connect measured counts. Resident calls complete output and
metadata before timing ends; host status checks and oracle checks follow.
Host APIs validate codec status within their timed calls. The first native build
and per-layout XLA compilation are excluded from steady-state timings.

![Zero bytes: amortized compression/decompression latency, host bytes](benchmarks/figures/small-batch/rtx4090-zeros-host-bytes.png)

[SVG](benchmarks/figures/small-batch/rtx4090-zeros-host-bytes.svg) · [PDF](benchmarks/figures/small-batch/rtx4090-zeros-host-bytes.pdf)

![Zero bytes: amortized compression/decompression latency, resident](benchmarks/figures/small-batch/rtx4090-zeros-resident.png)

[SVG](benchmarks/figures/small-batch/rtx4090-zeros-resident.svg) · [PDF](benchmarks/figures/small-batch/rtx4090-zeros-resident.pdf)

![Generated text: amortized compression/decompression latency, host bytes](benchmarks/figures/small-batch/rtx4090-text-host-bytes.png)

[SVG](benchmarks/figures/small-batch/rtx4090-text-host-bytes.svg) · [PDF](benchmarks/figures/small-batch/rtx4090-text-host-bytes.pdf)

![Generated text: amortized compression/decompression latency, resident](benchmarks/figures/small-batch/rtx4090-text-resident.png)

[SVG](benchmarks/figures/small-batch/rtx4090-text-resident.svg) · [PDF](benchmarks/figures/small-batch/rtx4090-text-resident.pdf)

![Uniform random bytes: amortized compression/decompression latency, host bytes](benchmarks/figures/small-batch/rtx4090-random-host-bytes.png)

[SVG](benchmarks/figures/small-batch/rtx4090-random-host-bytes.svg) · [PDF](benchmarks/figures/small-batch/rtx4090-random-host-bytes.pdf)

![Uniform random bytes: amortized compression/decompression latency, resident](benchmarks/figures/small-batch/rtx4090-random-resident.png)

[SVG](benchmarks/figures/small-batch/rtx4090-random-resident.svg) · [PDF](benchmarks/figures/small-batch/rtx4090-random-resident.pdf)

[Raw 36-case results](benchmarks/results/small-batch-rtx4090-20261007.json) include seven samples per workflow,
encoded lengths/hashes and an optional compiled padded batch round trip.
Every final returned workflow output passed byte checks; compressed streams
were decoded by stdlib and resident metadata/padding were checked after timing.
Round-trip encoded lengths stay on device; its timings are recorded, not plotted.
The report's `complete` marker is true only after the whole matrix finishes.

## Resident checked decoding on RTX 4090

These measurements use one warmed `jax.jit` checked decode of an independent
stdlib level-6 stream. The input, output and final metadata stay on the device;
completion waits for both returned arrays. Timings exclude compilation, initial
uploads and post-call host status/byte checks. GPU codec validation and temporary
allocations are included. This timing scope differs from the host-byte and eager
API measurements elsewhere; no CPU speedup is inferred from this dataset.

| Stream size | Workload | Compressed bytes | Completed latency (ms) | Throughput (MiB/s) |
| --- | --- | ---: | ---: | ---: |
| 64 KiB | Zero bytes | 84 | 0.525 | 119.1 |
| 64 KiB | Generated text | 7064 | 3.662 | 17.1 |
| 64 KiB | Integer counters | 22701 | 15.420 | 4.1 |
| 64 KiB | Gaussian floats | 60690 | 17.159 | 3.6 |
| 64 KiB | Random bytes | 65562 | 0.482 | 129.7 |
| 128 KiB | Zero bytes | 149 | 0.793 | 157.6 |
| 128 KiB | Generated text | 13844 | 6.657 | 18.8 |
| 128 KiB | Integer counters | 45357 | 9.341 | 13.4 |
| 128 KiB | Gaussian floats | 121341 | 5.309 | 23.5 |
| 128 KiB | Random bytes | 131118 | 0.410 | 304.7 |
| 256 KiB | Zero bytes | 277 | 1.266 | 197.5 |
| 256 KiB | Generated text | 27438 | 12.866 | 19.4 |
| 256 KiB | Integer counters | 90669 | 9.329 | 26.8 |
| 256 KiB | Gaussian floats | 242751 | 5.309 | 47.1 |
| 256 KiB | Random bytes | 262230 | 0.445 | 561.8 |
| 1 MiB | Zero bytes | 1039 | 4.262 | 234.6 |
| 1 MiB | Generated text | 108353 | 22.218 | 45.0 |
| 1 MiB | Integer counters | 362577 | 9.629 | 103.8 |
| 1 MiB | Gaussian floats | 971070 | 5.495 | 182.0 |
| 1 MiB | Random bytes | 1048902 | 0.691 | 1447.5 |
| 8 MiB | Zero bytes | 8163 | 21.198 | 377.4 |
| 8 MiB | Generated text | 863857 | 22.591 | 354.1 |
| 8 MiB | Integer counters | 2900368 | 9.756 | 820.0 |
| 8 MiB | Gaussian floats | 7769095 | 6.834 | 1170.6 |
| 8 MiB | Random bytes | 8391174 | 1.869 | 4280.9 |
| 64 MiB | Zero bytes | 65238 | 23.136 | 2766.3 |
| 64 MiB | Generated text | 6907553 | 26.782 | 2389.7 |
| 64 MiB | Integer counters | 23203543 | 15.970 | 4007.6 |
| 64 MiB | Gaussian floats | 62153557 | 21.658 | 2955.0 |
| 64 MiB | Random bytes | 67129345 | 11.336 | 5645.9 |

![Resident checked decode latency and throughput on RTX 4090](benchmarks/figures/resident-checked.png)

Medians use 31 completed calls after all shapes are warmed; whiskers show sample
minimum/maximum. Axes are logarithmic. The final returned sample in each case
passed status and independent byte checks; CPU codec functions are forbidden
during every timed CUDA call.
[SVG](benchmarks/figures/resident-checked.svg) ·
[PDF](benchmarks/figures/resident-checked.pdf) ·
[Raw samples and hashes](benchmarks/results/resident-checked-rtx4090-20261007.json).

Measured 2026-10-07 on RTX 4090, driver 610.57.04, JAX/JAXlib 0.11.2,
Python 3.12.8 and nvcc 12.1.105. Package and harness hashes match
[cb5a629](https://github.com/xangma/cuda-zlib/tree/cb5a6299a26a9221bbac7e6f1201ce052930bcc9).
The raw report records the loaded native library's cache identity, source/header
hashes, compiler and flags. The private workspace pool ended with
576 MiB retained and zero live scratch, using a 1 GiB release threshold.
Pool reservation excludes JAX inputs and outputs. Decoding above 64 KiB eagerly
allocates reference, candidate and summary buffers; see the [workspace bounds](README.md#installation).
The workstation was shared; device snapshots do not establish isolation.

```sh
git checkout cb5a6299a26a9221bbac7e6f1201ce052930bcc9
CUDACXX=/path/to/nvcc XLA_PYTHON_CLIENT_PREALLOCATE=false \
  python benchmarks/profile_resident.py --sizes 65536 131072 262144 1048576 8388608 67108864 \
  --workloads zeros text uint32 float32 random --seed 20261007 --samples 31 --output resident.json
python benchmarks/plot_resident.py resident.json --output-dir resident-figures
```

## Workloads

The general and small-file datasets use seed `20261007`. Small-file payloads
use `seed + file_index`; all other payloads use the same seed for each workload.

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

The general and small-file runs were measured on 2026-10-07 using an RTX 4090,
AMD Threadripper PRO 3995WX, Linux x86_64, Python 3.12.8, NumPy 2.2.6,
JAX/JAXlib 0.11.2 and stdlib zlib 1.3.1. Driver version was 610.57.04.
JAX reported CUDA platform `cuda 13040`; the native codec compiler was
`nvcc` 12.1.105 targeting `sm_89`. The JAX platform string and native compiler
version describe different components.

Measured checkout: [22d63fa](https://github.com/xangma/cuda-zlib/tree/22d63fa93c1d590a7ed91ad8afd8fff579ceee67).
Reports record `source_revision`, `harness_sha256` and codec source hashes.
`environment.native_build` identifies the **actually loaded library** with its
cache key, library/build SHA-256 hashes, build identity, FFI targets, compiler,
flags and source/header hashes. The compatible `native_builds` field contains
that single build identity. These records identify the loaded backend separately
from the benchmark harness. Figure manifests pin their source report and exports.

Both runs used fresh native caches. Startup and final private-pool accounting:

| Run | Imports and CUDA init (s) | Native build and registration (s) | Reserved workspace after run (MiB) | Live scratch after run (bytes) |
| --- | ---: | ---: | ---: | ---: |
| Single streams | 1.199 | 35.418 | 576 | 0 |
| Small-file batches | 1.175 | 35.128 | 64 | 0 |

The release threshold was **1 GiB**. Reserved pages permit reuse and remain
allocated after scratch is freed; this is a retention policy, not a memory cap.
Pool accounting excludes JAX input/output and pinned-host buffers. The general
and small-file runs used `XLA_PYTHON_CLIENT_PREALLOCATE=false`. The workstation
was shared; before/after snapshots do not establish isolation.

Separate [Apple M4 Max CPU reference measurements](benchmarks/CPU_BASELINES.md)
retain their recorded date and environment; they are not GPU speedup baselines.

## Reproduce

Check out the recorded harness and codec, and use a compatible GPU JAX runtime.
The recorded Python and NumPy versions were 3.12.8 and 2.2.6; JAX/JAXlib were
0.11.2. Stdlib zlib's version depends on the Python build.

```sh
git clone https://github.com/xangma/cuda-zlib.git
cd cuda-zlib
git checkout 22d63fa93c1d590a7ed91ad8afd8fff579ceee67
python -m pip install . matplotlib
CUDACXX=/path/to/nvcc XLA_PYTHON_CLIENT_PREALLOCATE=false \
  CUDA_ZLIB_WORKSPACE_RETENTION_BYTES=1073741824 CUDA_ZLIB_CACHE_DIR="$(mktemp -d)" \
  python benchmarks/benchmark.py --sizes 65536 1048576 67108864 \
  --workloads zeros text uint32 float32 random --samples 15 --cpu-samples 5 \
  --seed 20261007 --device 0 --output results-cuda.json
python benchmarks/plot_results.py --input results-cuda.json
CUDACXX=/path/to/nvcc XLA_PYTHON_CLIENT_PREALLOCATE=false \
  CUDA_ZLIB_WORKSPACE_RETENTION_BYTES=1073741824 CUDA_ZLIB_CACHE_DIR="$(mktemp -d)" \
  python benchmarks/small_batch.py --sizes 256 4096 65536 --counts 1 8 32 128 \
  --workloads zeros text random --samples 7 --seed 20261007 --device 0 \
  --roundtrip --output small-batch.json
python benchmarks/plot_small_batch.py small-batch.json --output-dir small-batch-figures
```

Install `".[cuda12]"` when selecting JAX's CUDA 12 runtime for a new environment;
the toolkit compiler is installed separately. Set `CUDACXX` to its `nvcc` path.
The native cache key covers codec sources, FFI headers, compiler, architecture,
flags and JAXlib version. Native build/registration and per-shape XLA compilation
are excluded by warmup and recorded separately from steady-state timings.
For a source export without `.git`, set `CUDA_ZLIB_SOURCE_REVISION` to the full
checkout revision; it records supplied metadata, while source hashes identify
the actual files.

Measurement commands above pin their recorded harnesses. To regenerate the
published figures, use the **current checkout's renderers** and recorded JSON
reports, without rerunning CUDA:

```sh
python benchmarks/plot_results.py --input benchmarks/results/rtx4090-20261007.json
python benchmarks/plot_small_batch.py benchmarks/results/small-batch-rtx4090-20261007.json \
  --output-dir benchmarks/figures/small-batch --prefix rtx4090
python benchmarks/plot_resident.py benchmarks/results/resident-checked-rtx4090-20261007.json \
  --output-dir benchmarks/figures
python benchmarks/verify_figures.py
```

The renderers produce PNG, SVG and PDF exports; their versions and source hashes
are recorded in figure manifests. Published exports used Matplotlib 3.11.1 and
NumPy 2.5.2, separately from the NumPy 2.2.6 benchmark environment. General
figures are in [benchmarks/figures](benchmarks/figures), small-file figures in
[benchmarks/figures/small-batch](benchmarks/figures/small-batch).
Use `--validate-only` to check a report without writing figures.

For CPU-only measurements without JAX or CUDA:

```sh
python -m pip install numpy
python benchmarks/benchmark.py --cpu-only --cpu-samples 5 \
  --seed 20261007 --output results-cpu.json
python benchmarks/small_batch.py --cpu-only --samples 7 \
  --seed 20261007 --output small-batch-cpu.json
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
