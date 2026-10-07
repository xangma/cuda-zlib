# General byte-stream benchmarks

Two measured datasets are reported separately: independent small files on an
RTX 4090, and single larger streams on an RTX 3090. Each uses single-threaded
stdlib zlib on its own host CPU. Results from different GPUs are not combined.

## Independent small files on RTX 4090

**CPU was faster for every single-file case**, even with resident CUDA buffers.
Packing multiple independent streams into one CUDA call amortizes dispatch and
allows files to execute concurrently. At 128 files, packed resident compression
was **35–121×** faster than a compiled loop of single-file CUDA calls, and
resident decompression was **11–129×** faster. These compare available API
workflows on the same source snapshot.

That is not the speedup over CPU. Including transfers and Python `bytes`
outputs, 128-file batches beat CPU compression for random data at all three
sizes and for 64 KiB zero/text files. The 4 KiB text compression result was
near CPU parity. CPU remained faster for 256-byte zero/text compression,
4 KiB zero compression, all 256-byte decompression, and 4 KiB text/random
decompression. At 64 KiB, batched host decompression was **9.49×** CPU for
zeros and **1.80×** for text; CPU was slightly faster for random bytes.

### Amortized latency with 128 files

Values are **microseconds per file = total completed call time / 128**; they
are not the latency of an individual file. Lower is better. CPU uses one thread
and host `bytes`; CUDA resident inputs/outputs stay on device. CUDA host timings
start and finish with `bytes`, including packing, transfers, status checks and
output copies. Decompression uses identical independent stdlib level-6 streams.

| File size | Workload | CPU compression, level 1 | CUDA batch compression, resident | CUDA batch compression, host bytes | CPU decompression | CUDA batch decompression, resident | CUDA batch decompression, host bytes |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 256 B | Zero bytes | 3.44 | 1.61 | 9.40 | 1.27 | 1.74 | 8.90 |
| 256 B | Generated text | 8.20 | 2.38 | 9.62 | 3.22 | 2.04 | 9.89 |
| 256 B | Random bytes | 30.73 | 3.68 | 13.05 | 0.58 | 1.09 | 9.54 |
| 4 KiB | Zero bytes | 7.85 | 3.51 | 12.61 | 14.65 | 1.59 | 10.65 |
| 4 KiB | Generated text | 19.81 | 6.48 | 18.66 | 8.78 | 3.52 | 14.39 |
| 4 KiB | Random bytes | 99.94 | 8.36 | 20.77 | 2.39 | 1.37 | 12.97 |
| 64 KiB | Zero bytes | 105.15 | 16.59 | 57.62 | 224.15 | 4.23 | 23.62 |
| 64 KiB | Generated text | 251.77 | 38.48 | 79.43 | 92.35 | 27.04 | 51.45 |
| 64 KiB | Random bytes | 1725.84 | 49.73 | 100.35 | 35.06 | 3.64 | 36.93 |

### Small-file plots

The plots show 256 B, 4 KiB and 64 KiB files at counts 1, 8, 32 and 128.
Points are medians of seven completed calls; whiskers show sample minimum and
maximum. Lines connect measured counts, not an inferred crossover. The
resident single-file comparison is one warmed `jax.jit` containing independent
FFI calls; the packed workflow uses one batch FFI call. Host single-file timings
use synchronous convenience APIs in a Python loop. Resident timings wait for
all returned arrays, including metadata; metadata transfer and status checks
occur afterward. Host APIs check status before returning.

![Small-file host-byte compression and decompression, generated text](benchmarks/figures/small-batch/rtx4090-text-host-bytes.png)

[Host text SVG](benchmarks/figures/small-batch/rtx4090-text-host-bytes.svg) ·
[Host text PDF](benchmarks/figures/small-batch/rtx4090-text-host-bytes.pdf) ·
[Resident text plot](benchmarks/figures/small-batch/rtx4090-text-resident.png) ·
[Zero-byte host plot](benchmarks/figures/small-batch/rtx4090-zeros-host-bytes.png) ·
[Random-byte host plot](benchmarks/figures/small-batch/rtx4090-random-host-bytes.png).
All six plots have PNG, SVG and PDF exports in
[the small-file figure directory](benchmarks/figures/small-batch).

### Small-file environment and reproduction

Measured 2026-10-07 on an RTX 4090 with an AMD Threadripper PRO 3995WX,
Linux x86_64, Python 3.12.8, NumPy 2.2.6, JAX/JAXlib 0.11.2 and zlib 1.3.1.
JAX reported CUDA platform `cuda 13040`, driver 610.57.04; the native library
used `nvcc` 12.1.105 for `sm_89`. All codec and harness hashes in `source.sha256` in the
[raw 36-case report](benchmarks/results/small-batch-rtx4090-20261007.json)
match [8363446](https://github.com/xangma/cuda-zlib/tree/8363446113eddc6a782d2c31a803a6d37ca87cec).
The staging directory was a source export without Git metadata; exact source
SHA-256 hashes establish that snapshot. Payload and compressed-stream hashes,
all samples, software versions and native build flags are also recorded.
`environment.native_builds` inventories the cache, including an older build;
it is not a list of libraries loaded for this run.

Each workflow has one untimed warmup, including per-shape XLA compilation,
and seven timed completed calls. Native build/registration, initial resident
uploads, payload generation and oracle checks are excluded. Benchmark oracle
checks of status, bytes and zero padding occur after timing; host APIs also
validate status within their timed calls. Stdlib decodes CUDA output, and CUDA
decodes independent stdlib streams. All 36 cases passed, including a compiled
padded batch round trip with encoded lengths kept on device; round-trip samples
are in the JSON but are not plotted. Compression sizes differ between codecs;
encoded sizes are recorded and CUDA has no equivalent to zlib's levels.

The native cache was already built: imports/CUDA initialization took 1.178 s,
and cache load/registration took 0.008 s. This does not measure a cold native
build. The private workspace pool used a 1 GiB retention threshold and ended
with 64 MiB reserved and zero live scratch. This was a shared workstation;
utilization snapshots do not prove isolation. No parallel CPU codec baseline
was measured.

```sh
git checkout 8363446113eddc6a782d2c31a803a6d37ca87cec
python -m pip install . matplotlib
CUDACXX=/path/to/nvcc XLA_PYTHON_CLIENT_PREALLOCATE=false \
  python benchmarks/small_batch.py --sizes 256 4096 65536 \
  --counts 1 8 32 128 --workloads zeros text random --samples 7 \
  --seed 20261007 --device 0 --roundtrip --output small-batch.json
python benchmarks/plot_small_batch.py small-batch.json \
  --output-dir small-batch-figures
```

Use an existing compatible GPU JAX installation, or install `".[cuda12]"`
when setting up a CUDA 12 runtime. `--cpu-only`
runs real CPU measurements without JAX or CUDA. The plotter rejects incomplete
matrices and inconsistent summaries; its manifest hashes every export and the
source report.

## Single-stream results on RTX 3090

These results compare cuda-zlib on an **RTX 3090** with **single-threaded stdlib
zlib on the same AMD Threadripper PRO 3995WX CPU**, using five synthetic workloads
at 64 KiB, 1 MiB and 64 MiB.

For warm **64 MiB** inputs, CUDA compression achieved **3.88–12.52×** CPU zlib
level-1 throughput. CUDA decompression achieved **1.06–4.42×** CPU throughput
on identical stdlib level-6 streams; CUDA was faster across all five workloads,
with random bytes close to CPU parity.
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
| Zero bytes | 4.14× | 9.58× | 3.68× |
| Generated text | 3.88× | 12.41× | 2.06× |
| Integer counters | 6.32× | 45.42× | 3.67× |
| Gaussian float32 | 12.52× | 13.96× | 4.42× |
| Uniform random bytes | 10.34× | 10.35× | 1.06× |

### Smaller inputs

These ratios use CPU level 1 for compression and identical level-6 streams for
decompression, with the same host-to-host CUDA timing scope as above.

| Workload | 64 KiB compression | 1 MiB compression | 64 KiB decompression | 1 MiB decompression |
| --- | ---: | ---: | ---: | ---: |
| Zero bytes | 0.03× | 0.71× | 0.13× | 0.39× |
| Generated text | 0.04× | 0.76× | 0.02× | 0.07× |
| Integer counters | 0.10× | 1.58× | 0.02× | 0.34× |
| Gaussian float32 | 0.25× | 4.17× | 0.06× | 0.83× |
| Uniform random bytes | 0.25× | 4.17× | 0.03× | 0.31× |

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

Measured 2026-10-07 on GPU 0 of a two-GPU RTX 3090 workstation (24 GiB per
GPU), with an AMD Ryzen Threadripper PRO 3995WX CPU, Linux x86_64, Python
3.13.3, NumPy 2.2.5, JAX and JAXlib 0.11.2, the JAX typed CUDA FFI backend,
CUDA 12.6, NVIDIA driver 610.57.04 and stdlib zlib 1.3.1. The native library
was built with `nvcc` 12.6.85 for `sm_86`. The public source checkout is
[0a12b10](https://github.com/xangma/cuda-zlib/tree/0a12b1082ce3dd550a9c9d09bef982342c0dd2be);
its nine codec source SHA-256 hashes, native build flags and FFI header hashes
are recorded in the raw results.
All 15 workload/size cases passed byte-exact validation, including stdlib
decoding CUDA output and CUDA decoding independent stdlib level-6 output.

Each operation has one untimed warmup, then 15 CUDA samples or five CPU
samples. Throughput tables report median MiB/s, using uncompressed byte counts. CPU figures
below come from the same run and host. These are single-run measurements on a
shared workstation; recorded utilization snapshots do not establish isolation.

With a fresh native library cache, imports and CUDA initialization took
1.196 s, and `compile_kernels` took 34.646 s for the native build and FFI
registration. These startup costs and each workflow's initial XLA compilation
are excluded from the warmed tables.

The codec used a **1 GiB release threshold** for its private workspace pool.
After the run, one pool retained **544 MiB** of reserved pages with **0 bytes**
of live scratch. Retaining freed pages allows allocation reuse between calls
at the cost of keeping GPU memory reserved. These warm throughput measurements
use that retention policy; changing the threshold or trimming unused pages can
change subsequent allocation costs. This pool accounting excludes JAX input,
output and pinned-host buffers.

Large compressed inputs use a speculative-prefix queue of about one eighth
the compressed input size. The queue is bounded; dense prefixes use a GPU
fallback path.

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
python benchmarks/plot_results.py --input benchmarks/results/rtx3090-ffi-20261007.json
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
| Zero bytes | 1780.1 | 1510.9 | 364.8 | 157.7 |
| Generated text | 750.4 | 665.5 | 171.5 | 53.6 |
| Integer counters | 425.1 | 310.6 | 49.1 | 6.8 |
| Gaussian float32 | 379.6 | 217.8 | 17.4 | 15.6 |
| Uniform random bytes | 593.4 | 264.0 | 25.5 | 25.5 |

### 64 MiB decompression throughput

Each CPU/CUDA pair decodes the **same compressed stream**. Codec-produced and
stdlib level-6 streams have different block layouts and must be compared separately.
Host-to-host level-6 decoding includes both transfers and host byte conversion.
Host-array decoding includes both transfers and returns the NumPy view of
completed pinned host storage, without the final copy to Python `bytes`.

| Workload | CUDA resident, codec stream | CPU, codec stream | CUDA resident, level-6 stream | CUDA host array, level-6 stream | CUDA host bytes, level-6 stream | CPU, level-6 stream |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Zero bytes | 9257.0 | 177.9 | 2235.5 | 1987.9 | 665.9 | 180.7 |
| Generated text | 3181.4 | 241.6 | 1540.1 | 1394.3 | 606.5 | 294.6 |
| Integer counters | 1597.8 | 168.6 | 1820.7 | 1561.8 | 620.3 | 168.8 |
| Gaussian float32 | 1181.2 | 122.9 | 1246.3 | 1065.2 | 529.9 | 120.0 |
| Uniform random bytes | 3017.9 | 636.6 | 2797.2 | 1939.0 | 684.5 | 647.4 |

For example, resident zero-byte decoding reaches 9257.0 MiB/s for this codec's
stream and 2235.5 MiB/s for the level-6 stream. This difference is a property of
the stream layout and decoder paths, not a general speedup over the CPU.

[Raw RTX 3090 and same-host CPU results](benchmarks/results/rtx3090-ffi-20261007.json)
include all three input sizes, every sample, min/max timings, host-to-device
compression, encoded sizes, payload hashes, startup measurements and environment
metadata. Separate [Apple M4 Max CPU measurements](benchmarks/CPU_BASELINES.md)
are available for reference; they are not used to calculate GPU speedup.

## Reproduce

Check out the recorded source snapshot, install its dependencies and run the
harness:

```sh
git clone https://github.com/xangma/cuda-zlib.git
cd cuda-zlib
git checkout 0a12b1082ce3dd550a9c9d09bef982342c0dd2be
python -m pip install ".[cuda12]" matplotlib
CUDACXX=/usr/local/cuda-12.6/bin/nvcc CUDA_ZLIB_WORKSPACE_RETENTION_BYTES=1073741824 \
  CUDA_ZLIB_CACHE_DIR="$(mktemp -d)" python benchmarks/benchmark.py \
  --sizes 65536 1048576 67108864 --samples 15 --cpu-samples 5 \
  --seed 20261006 --device 0 --output results-cuda.json
python benchmarks/plot_results.py --input results-cuda.json
```

Select the CUDA toolkit compiler with `CUDACXX`. If it is unset, the runtime
looks for `nvcc` under `CUDA_HOME` or `CUDA_PATH`, then on `PATH`, then at
`/usr/local/cuda/bin/nvcc`. The codec source hashes and payload hashes
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
