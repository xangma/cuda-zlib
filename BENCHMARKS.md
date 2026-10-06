# General byte-stream benchmarks

These results compare cuda-zlib on an **RTX 3090** with **single-threaded stdlib
zlib on the same AMD Threadripper PRO 3995WX CPU**, using five synthetic workloads
at 64 KiB, 1 MiB and 64 MiB.

For warm **64 MiB** inputs, CUDA compression achieved **3.87–12.53×** CPU zlib
level-1 throughput. CUDA decompression achieved **1.02–4.29×** CPU throughput
on identical stdlib level-6 streams; CUDA was faster for zeros, generated text,
integer counters and Gaussian float32; random bytes were close to CPU parity.
**Both comparisons include uploads, downloads, codec validation and conversion
to host bytes.** At **64 KiB**, CPU compression and
decompression were faster for every workload. At **1 MiB**, compression was
mixed and CPU decompression was faster for every workload.

Compression sizes differ between codecs; the CUDA compressor has no equivalent
to zlib's compression levels. Read throughput alongside encoded size. These
single-threaded CPU baselines do not measure parallel CPU compression.

## GPU versus CPU, including transfers

**Speedup = CPU median elapsed time / CUDA median elapsed time.** Above 1× favors
CUDA; below 1× favors CPU. The CUDA workflows used in these ratios start and
finish with Python `bytes`.
Compression includes upload, encoding, download and host conversion; decompression
includes upload, decoding, download and host conversion. Allocations, codec
validation and synchronization are included; startup is excluded. CPU and CUDA
decompression use identical stdlib level-6 compressed bytes.

The bytes-returning decompression workflow uses
`decompress_zlib_host(payload, expected_bytes, device).tobytes()`. The separate
host-array workflow returns a completed, read-only, contiguous NumPy `uint8`
array backed by JAX-owned pinned host memory. Its timing includes upload,
decoding and the transfer to host memory, and excludes the final copy into
Python `bytes`. CPU speedup ratios use the bytes-returning workflow so both
outputs have the same type. A `memoryview` of the host array shares its storage
without another copy.

![64 MiB GPU speedup over same-host CPU, including transfers](benchmarks/figures/cpu-speedup.png)

[Speedup SVG](benchmarks/figures/cpu-speedup.svg) ·
[Speedup PDF](benchmarks/figures/cpu-speedup.pdf)

### 64 MiB speedup

| Workload | Compression vs CPU level 1 | Compression vs CPU level 6 | Decompression vs CPU, identical level-6 stream |
| --- | ---: | ---: | ---: |
| Zero bytes | 4.10× | 9.45× | 3.69× |
| Generated text | 3.87× | 12.36× | 1.75× |
| Integer counters | 6.21× | 45.68× | 3.67× |
| Gaussian float32 | 12.53× | 13.91× | 4.29× |
| Uniform random bytes | 10.27× | 10.37× | 1.02× |

### Smaller inputs

These ratios use CPU level 1 for compression and identical level-6 streams for
decompression, with the same host-to-host CUDA timing scope as above.

| Workload | 64 KiB compression | 1 MiB compression | 64 KiB decompression | 1 MiB decompression |
| --- | ---: | ---: | ---: | ---: |
| Zero bytes | 0.03× | 0.71× | 0.14× | 0.39× |
| Generated text | 0.04× | 0.75× | 0.02× | 0.07× |
| Integer counters | 0.10× | 1.63× | 0.02× | 0.34× |
| Gaussian float32 | 0.25× | 4.15× | 0.06× | 0.83× |
| Uniform random bytes | 0.25× | 4.09× | 0.03× | 0.30× |

Only these three sizes were measured; they do not identify an exact crossover
size. Launch and transfer costs make small inputs less favorable to CUDA.

## Workloads

Each workload runs at 64 KiB, 1 MiB and 64 MiB with seed `20261006`:

- `zeros`: all zero bytes, representing extremely compressible input.
- `text`: generated ASCII request records with varying fields. An 8192-record
  block repeats to fill larger inputs; this is synthetic repetitive text.
- `uint32`: ascending counters encoded as little-endian unsigned 32-bit integers.
- `float32`: seeded Gaussian samples encoded as little-endian 32-bit floats.
- `random`: seeded uniformly distributed bytes, representing incompressible input.

Payload generation and post-timing oracle comparisons occur outside timed
regions. Codec framing, bounds, status and checksum validation remain included.
Payload SHA-256 hashes, all timing samples and software versions are recorded in
the result JSON.

## Recorded CUDA results

Measured 2026-10-06 on GPU 0 of a two-GPU RTX 3090 workstation (24 GiB per
GPU), with an AMD Ryzen Threadripper PRO 3995WX CPU, Linux x86_64, Python
3.13.3, NumPy 2.2.5, JAX and JAXlib 0.11.2, the JAX typed CUDA FFI backend,
CUDA 12.6, NVIDIA driver 610.57.04 and stdlib zlib 1.3.1. The native library
was built with `nvcc` 12.6.85 for `sm_86`. The installed wheel's source snapshot is
[6ded403](https://github.com/xangma/cuda-zlib/tree/6ded403465802785c7ed6cc62662e61183669eff);
its nine codec source SHA-256 hashes, native build flags and FFI header hashes
are recorded in the raw results.
All 15 workload/size cases passed byte-exact validation, including stdlib
decoding CUDA output and CUDA decoding independent stdlib level-6 output.

Each operation has one untimed warmup, then 15 CUDA samples or five CPU
samples. Throughput tables report median MiB/s, using uncompressed byte counts. CPU figures
below come from the same run and host. These are single-run measurements on a
shared workstation; recorded utilization snapshots do not establish isolation.

With a fresh native library cache, imports and CUDA initialization took
1.208 s, and `compile_kernels` took 33.779 s for the native build and FFI
registration. These startup costs and each workflow's initial XLA compilation
are excluded from the warmed tables.

The codec used a **1 GiB release threshold** for its private workspace pool.
After the run, one pool retained **544 MiB** of reserved pages with **0 bytes**
of live scratch. Retaining freed pages allows allocation reuse between calls
at the cost of keeping GPU memory reserved. These warm throughput measurements
use that retention policy; changing the threshold or trimming unused pages can
change subsequent allocation costs. This pool accounting excludes JAX input,
output and pinned-host buffers.

### Plots

The figures use the recorded RTX 3090 run and its same-host CPU baselines.
Throughput points are medians; error bars show the observed sample range, not
confidence intervals. The throughput plots use logarithmic axes; the speedup
plot above uses linear axes and labels each ratio directly. Lines connect
measured sizes; intermediate sizes were not measured.

![Compression throughput across input sizes](benchmarks/figures/compression-throughput.png)

[Compression SVG](benchmarks/figures/compression-throughput.svg) ·
[Compression PDF](benchmarks/figures/compression-throughput.pdf)

![Compressed size at 64 MiB](benchmarks/figures/encoded-size.png)

The size comparison uses 64 MiB inputs. Encoded percentage is compressed bytes
divided by input bytes; values above 100% indicate expansion.
[Size SVG](benchmarks/figures/encoded-size.svg) ·
[Size PDF](benchmarks/figures/encoded-size.pdf)

![Decompression throughput for identical stdlib level-6 streams](benchmarks/figures/decompression-throughput.png)

These CPU and CUDA workflows decode identical stdlib level-6 streams.
CUDA host-array output and host-byte output are separate timing series;
CPU output is Python `bytes`.
[Decompression SVG](benchmarks/figures/decompression-throughput.svg) ·
[Decompression PDF](benchmarks/figures/decompression-throughput.pdf)

![Resident decompression by compressed stream layout at 64 MiB](benchmarks/figures/decode-stream-layout.png)

The layout comparison uses 64 MiB inputs and resident GPU timings. Each CPU/CUDA
pair decodes the same stream; codec-produced and stdlib level-6 streams are shown
separately.
[Layout SVG](benchmarks/figures/decode-stream-layout.svg) ·
[Layout PDF](benchmarks/figures/decode-stream-layout.pdf)

To regenerate the figures from a current checkout:

```sh
python -m pip install matplotlib
python benchmarks/plot_results.py --input benchmarks/results/rtx3090-ffi-20261006-3.json
```

The script reads the existing JSON without running CUDA benchmarks. PNG, SVG
and PDF exports and a source-hash manifest are in
[benchmarks/figures](benchmarks/figures).

### 64 MiB encoded size

Encoded percentage is compressed bytes divided by input bytes; lower is better.
Values over 100% indicate expansion. The CUDA compressor has no level equivalent
to zlib's levels 1 or 6, so throughput should be considered alongside these sizes.

| Workload | CUDA | CPU level 1 | CPU level 6 |
| --- | ---: | ---: | ---: |
| Zero bytes | 0.153% | 0.436% | 0.097% |
| Generated text | 13.179% | 12.121% | 10.293% |
| Integer counters | 34.621% | 34.589% | 34.576% |
| Gaussian float32 | 92.629% | 92.886% | 92.617% |
| Uniform random bytes | 100.015% | 100.030% | 100.031% |

### 64 MiB compression throughput

Resident timings exclude initial input upload. Host-to-host timings include input
upload, compression, output download and conversion to host bytes. API output and
workspace allocations remain included in both workflows.

| Workload | CUDA resident | CUDA host-to-host | CPU level 1 | CPU level 6 |
| --- | ---: | ---: | ---: | ---: |
| Zero bytes | 1772.6 | 1493.4 | 364.1 | 158.0 |
| Generated text | 746.3 | 662.6 | 171.4 | 53.6 |
| Integer counters | 425.3 | 311.9 | 50.2 | 6.8 |
| Gaussian float32 | 379.2 | 217.3 | 17.3 | 15.6 |
| Uniform random bytes | 593.8 | 262.8 | 25.6 | 25.3 |

### 64 MiB decompression throughput

Each CPU/CUDA pair decodes the **same compressed stream**. Codec-produced and
stdlib level-6 streams have different block layouts and must be compared separately.
Host-to-host level-6 decoding includes both transfers and host byte conversion.
Host-array decoding includes both transfers and returns the NumPy view of
completed pinned host storage, without the final copy to Python `bytes`.

| Workload | CUDA resident, codec stream | CPU, codec stream | CUDA resident, level-6 stream | CUDA host array, level-6 stream | CUDA host bytes, level-6 stream | CPU, level-6 stream |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Zero bytes | 9231.2 | 178.5 | 2228.4 | 1980.9 | 666.0 | 180.4 |
| Generated text | 3095.2 | 272.3 | 1504.3 | 1368.8 | 597.8 | 342.2 |
| Integer counters | 1543.0 | 167.9 | 1795.2 | 1555.4 | 619.1 | 168.6 |
| Gaussian float32 | 1088.3 | 123.0 | 1147.4 | 993.8 | 514.7 | 120.1 |
| Uniform random bytes | 2521.0 | 655.3 | 2347.8 | 1729.3 | 664.0 | 653.5 |

For example, resident zero-byte decoding reaches 9231.2 MiB/s for this codec's
stream and 2228.4 MiB/s for the level-6 stream. This difference is a property of
the stream layout and decoder paths, not a general speedup over the CPU.

[Raw RTX 3090 and same-host CPU results](benchmarks/results/rtx3090-ffi-20261006-3.json)
include all three input sizes, every sample, min/max timings, host-to-device
compression, encoded sizes, payload hashes, startup measurements and environment
metadata. Separate [Apple M4 Max CPU measurements](benchmarks/CPU_BASELINES.md)
are available for reference; they are not used to calculate GPU speedup.

## Reproduce

From this repository checkout, install the measured codec source snapshot and
run the harness:

```sh
python -m pip install "cuda-zlib[cuda12] @ git+https://github.com/xangma/cuda-zlib.git@6ded403465802785c7ed6cc62662e61183669eff" matplotlib
CUDACXX=/usr/local/cuda-12.6/bin/nvcc CUDA_ZLIB_WORKSPACE_RETENTION_BYTES=1073741824 \
  CUDA_ZLIB_CACHE_DIR="$(mktemp -d)" python benchmarks/benchmark.py \
  --sizes 65536 1048576 67108864 --samples 15 --cpu-samples 5 \
  --seed 20261006 --device 0 --output results-cuda.json
python benchmarks/plot_results.py --input results-cuda.json
```

Select the CUDA toolkit compiler with `CUDACXX`. If it is unset, the runtime
looks for `nvcc` under `CUDA_HOME` or `CUDA_PATH`, then on `PATH`, then at
`/usr/local/cuda/bin/nvcc`. The installed codec source hashes and payload hashes
should match the raw results.

For the recorded software environment, use Python 3.13.3 and pin
`numpy==2.2.5`, `jax[cuda12]==0.11.2` and `jaxlib==0.11.2`. Stdlib zlib was
1.3.1; its version depends on the Python build.

The JAX FFI backend builds a native CUDA library with `nvcc` and caches it in
`CUDA_ZLIB_CACHE_DIR`, defaulting to `$XDG_CACHE_HOME/cuda-zlib` or
`~/.cache/cuda-zlib`. The persistent cache key covers sources, FFI headers,
compiler path and version, GPU architecture, build flags and JAXlib version.
The fresh temporary cache above measures the initial build rather than cache reuse.

The harness records import/CUDA initialization and `compile_kernels` durations
separately. `compile_kernels` builds or loads the native library and registers
FFI targets; shape-specific XLA compilation remains lazy. One untimed warmup
before each operation excludes that workflow's initial XLA compilation.
CUDA API calls are synchronous, including status checks and temporary workspace;
host workflows include transfers and completed host output. Compression converts
the JAX output to NumPy and Python `bytes`. Decompression uses
`decompress_zlib_host`; the host-array measurement returns its NumPy array and
the bytes-returning measurement additionally calls `.tobytes()`.
Avoid other active GPU work during measurements; GPU utilization, memory usage
and temperature are recorded before and after the run, but these snapshots do
not prove isolation.

CUDA measurements include:

- Compression with resident device input and output, host input to device
  output, and complete host input to host output.
- Resident decompression of codec-produced streams and independent zlib
  level-6 streams, plus level-6 decoding from host input to either a host array
  or Python `bytes`.
- Single-threaded stdlib zlib compression at levels 1 and 6 and CPU decoding
  of the same compressed streams used by the CUDA decoder.

Initial resident uploads, workload generation and post-timing oracle comparisons
are excluded from resident timings. API allocations and codec validation remain
included; host workflows include their transfers and host copies. All CUDA and
CPU results are checked against the original bytes. Stdlib zlib independently
decodes the CUDA compressor's output,
and CUDA decodes the stdlib level-6 output. A failed check stops the run.
The CUDA compressor uses its default 32768-byte chunks; it has no compression
level equivalent to zlib's levels 1 or 6.

For a CPU-only reproduction without JAX or CUDA:

```sh
python -m pip install numpy
python benchmarks/benchmark.py --cpu-only --cpu-samples 5 \
  --seed 20261006 --output results-cpu.json
```

For a focused smoke check, add `--sizes 65536 --samples 1 --cpu-samples 1`.
Small inputs can be dominated by launch and transfer costs. Highly repetitive
streams can exercise different decoding paths from random bytes. Compare both
encoded size and end-to-end throughput for the intended workload; synthetic
results and the alpha codec's documented bounds are not a universal performance
or compatibility guarantee.

## Profiling

Capture warmed CUDA calls with Nsight Systems:

```sh
CUDACXX=/usr/local/cuda-12.6/bin/nvcc python benchmarks/trace.py \
  --nsys /path/to/nsys --output traces/codec --cuda-profiler-range -- \
  python benchmarks/profile.py --sizes 1048576 --workloads zeros random \
  --samples 3 --cuda-profiler-range --output traces/wall.json
```

The command was validated with Nsight Systems 2026.5.1. Each profiler API range
captures an extra completed codec call after warmup; startup and oracle checks
are outside those ranges. The workload checks byte-exact outputs and forbids
CPU codec calls during CUDA operations. Profiler overhead can affect timings;
use the benchmark harness for throughput comparisons.

`trace.py` writes the Nsight report, SQLite export, log and a `.trace.json`
manifest containing the CLI version and record counts. It requires successful
execution, imported GPU kernels and CUDA API records, and rejects known import
errors even when Nsight returns zero. Choose a fresh output prefix for each run.
