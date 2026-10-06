# General byte-stream benchmarks

These results compare cuda-zlib on an **RTX 3090** with **single-threaded stdlib
zlib on the same AMD Threadripper PRO 3995WX CPU**, using five synthetic workloads
at 64 KiB, 1 MiB and 64 MiB.

For warm **64 MiB** inputs, CUDA compression achieved **3.41–12.12×** CPU zlib
level-1 throughput. CUDA decompression achieved **0.96–2.49×** CPU throughput
on identical stdlib level-6 streams; CPU was faster for generated text and
slightly faster for random bytes. **Both comparisons include uploads,
downloads and conversion to host bytes.** At **64 KiB**, CPU compression and
decompression were faster for every workload. At **1 MiB**, compression was
mixed and CPU decompression was faster for every workload.

Compression sizes differ between codecs; the CUDA compressor has no equivalent
to zlib's compression levels. Read throughput alongside encoded size. These
single-threaded CPU baselines do not measure parallel CPU compression.

## GPU versus CPU, including transfers

**Speedup = CPU median elapsed time / CUDA median elapsed time.** Above 1× favors
CUDA; below 1× favors CPU. Each CUDA workflow starts and finishes with host bytes.
Compression includes upload, encoding, download and host conversion; decompression
includes upload, decoding, download and host conversion. Allocations and
synchronization are included; startup is excluded. CPU and CUDA decompression
use identical stdlib level-6 compressed bytes.

![64 MiB GPU speedup over same-host CPU, including transfers](benchmarks/figures/cpu-speedup.png)

[Speedup SVG](benchmarks/figures/cpu-speedup.svg) ·
[Speedup PDF](benchmarks/figures/cpu-speedup.pdf)

### 64 MiB speedup

| Workload | Compression vs CPU level 1 | Compression vs CPU level 6 | Decompression vs CPU, identical level-6 stream |
| --- | ---: | ---: | ---: |
| Zero bytes | 3.41× | 7.89× | 1.88× |
| Generated text | 3.55× | 11.36× | 0.96× |
| Integer counters | 6.05× | 43.39× | 2.00× |
| Gaussian float32 | 12.12× | 13.55× | 2.49× |
| Uniform random bytes | 9.90× | 10.01× | 0.99× |

### Smaller inputs

These ratios use CPU level 1 for compression and identical level-6 streams for
decompression, with the same host-to-host CUDA timing scope as above.

| Workload | 64 KiB compression | 1 MiB compression | 64 KiB decompression | 1 MiB decompression |
| --- | ---: | ---: | ---: | ---: |
| Zero bytes | 0.03× | 0.67× | 0.12× | 0.38× |
| Generated text | 0.04× | 0.74× | 0.02× | 0.06× |
| Integer counters | 0.09× | 1.54× | 0.02× | 0.32× |
| Gaussian float32 | 0.24× | 4.07× | 0.05× | 0.76× |
| Uniform random bytes | 0.24× | 4.07× | 0.02× | 0.26× |

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

Generation and validation occur outside timed regions. Payload SHA-256 hashes,
all timing samples and software versions are recorded in the result JSON.

## Recorded CUDA results

Measured 2026-10-06 on GPU 0 of a two-GPU RTX 3090 workstation (24 GiB per
GPU), with an AMD Ryzen Threadripper PRO 3995WX CPU, Linux x86_64, Python
3.13.3, NumPy 2.2.5, JAX and JAXlib 0.11.2, the JAX typed CUDA FFI backend,
CUDA 12.6, NVIDIA driver 610.57.04 and stdlib zlib 1.3.1. The native library
was built with `nvcc` 12.6.85 for `sm_86`. The installed wheel's source snapshot is
[b8f736e](https://github.com/xangma/cuda-zlib/tree/b8f736e3512f450c3a871c85bf3e6d6c46e1e583);
its nine codec source SHA-256 hashes, native build flags and FFI header hashes
are recorded in the raw results.
All 15 workload/size cases passed byte-exact validation, including stdlib
decoding CUDA output and CUDA decoding independent stdlib level-6 output.

Each operation has one untimed warmup, then 15 CUDA samples or five CPU
samples. Throughput tables report median MiB/s, using uncompressed byte counts. CPU figures
below come from the same run and host. These are single-run measurements on a
shared workstation; recorded utilization snapshots do not establish isolation.

With a fresh native library cache, imports and CUDA initialization took
1.310 s, and `compile_kernels` took 29.892 s for the native build and FFI
registration. These startup costs and each workflow's initial XLA compilation
are excluded from the warmed tables.

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
python benchmarks/plot_results.py
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
| Zero bytes | 1414.4 | 1243.8 | 364.5 | 157.7 |
| Generated text | 678.9 | 608.1 | 171.2 | 53.5 |
| Integer counters | 402.8 | 296.9 | 49.1 | 6.8 |
| Gaussian float32 | 360.8 | 210.5 | 17.4 | 15.5 |
| Uniform random bytes | 547.4 | 253.5 | 25.6 | 25.3 |

### 64 MiB decompression throughput

Each CPU/CUDA pair decodes the **same compressed stream**. Codec-produced and
stdlib level-6 streams have different block layouts and must be compared separately.
Host-to-host level-6 decoding includes both transfers and host byte conversion.

| Workload | CUDA resident, codec stream | CPU, codec stream | CUDA resident, level-6 stream | CUDA host-to-host, level-6 stream | CPU, level-6 stream |
| --- | ---: | ---: | ---: | ---: | ---: |
| Zero bytes | 3571.9 | 178.1 | 1219.4 | 338.6 | 180.0 |
| Generated text | 2169.5 | 271.9 | 1113.5 | 329.8 | 342.2 |
| Integer counters | 1258.8 | 168.3 | 1227.2 | 338.0 | 168.7 |
| Gaussian float32 | 901.1 | 122.9 | 850.7 | 297.8 | 119.8 |
| Uniform random bytes | 1771.2 | 557.3 | 1669.5 | 552.6 | 556.7 |

For example, resident zero-byte decoding reaches 3571.9 MiB/s for this codec's
stream and 1219.4 MiB/s for the level-6 stream. This difference is a property of
the stream layout and decoder paths, not a general speedup over the CPU.

[Raw RTX 3090 and same-host CPU results](benchmarks/results/rtx3090-ffi-20261006.json)
include all three input sizes, every sample, min/max timings, host-to-device
compression, encoded sizes, payload hashes, startup measurements and environment
metadata. Separate [Apple M4 Max CPU measurements](benchmarks/CPU_BASELINES.md)
are available for reference; they are not used to calculate GPU speedup.

## Reproduce

From this repository checkout, install the measured codec source snapshot and
run the harness:

```sh
python -m pip install "cuda-zlib[cuda12] @ git+https://github.com/xangma/cuda-zlib.git@b8f736e3512f450c3a871c85bf3e6d6c46e1e583" matplotlib
CUDACXX=/usr/local/cuda-12.6/bin/nvcc CUDA_ZLIB_CACHE_DIR="$(mktemp -d)" python benchmarks/benchmark.py \
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
host-to-host workflows also copy the JAX output to NumPy and host bytes.
Avoid other active GPU work during measurements; GPU utilization, memory usage
and temperature are recorded before and after the run, but these snapshots do
not prove isolation.

CUDA measurements include:

- Compression with resident device input and output, host input to device
  output, and complete host input to host output.
- Resident decompression of codec-produced streams and independent zlib
  level-6 streams, plus complete host-to-host decoding of level-6 streams.
- Single-threaded stdlib zlib compression at levels 1 and 6 and CPU decoding
  of the same compressed streams used by the CUDA decoder.

Resident uploads, workload generation and byte comparisons are excluded from
resident timings. API allocations remain included; host workflows include their
transfers and host copies. All CUDA and CPU results are checked against the
original bytes. Stdlib zlib independently decodes the CUDA compressor's output,
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
