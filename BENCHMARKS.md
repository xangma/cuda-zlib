# General byte-stream benchmarks

The reproducible harness uses synthetic byte streams without external datasets.
It measures encoded size, compression throughput and decompression throughput.
Recorded results cover the CUDA codec on an RTX 3090 with same-host CPU
baselines, plus independent Apple CPU baselines. Small-input and large-input
results are shown separately; compressed size and stream layout also matter.

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
3.12.3, NumPy 2.2.6, CuPy 13.3.0, CUDA runtime 12.6, NVIDIA driver 610.57.04
and stdlib zlib 1.3. The original `0.1.0a1` release wheel was installed;
its six Python module hashes are recorded in the results. The harness is from
[commit 612fd07](https://github.com/xangma/cuda-zlib/tree/612fd07efbf65402b1641388431b885d56f3914d).
All 15 workload/size cases passed byte-exact validation, including stdlib
decoding CUDA output and CUDA decoding independent stdlib level-6 output.
Payload hashes match the recorded Apple CPU run despite different NumPy versions.

Each operation has one untimed warmup, then five CUDA samples or three CPU
samples. Tables report median MiB/s, using uncompressed byte counts. CPU figures
below come from the same run and host. These are single-run measurements on a
shared workstation; recorded utilization snapshots do not establish isolation.

With a fresh CuPy cache, imports and CUDA initialization took
0.356 s, and `compile_kernels` took
12.323 s. These startup costs are excluded from the warmed tables.

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
| Zero bytes | 879.7 | 811.2 | 402.3 | 164.8 |
| Generated text | 406.7 | 385.0 | 175.6 | 55.1 |
| Integer counters | 285.1 | 273.3 | 49.6 | 6.7 |
| Gaussian float32 | 278.2 | 216.9 | 19.9 | 17.2 |
| Uniform random bytes | 587.9 | 327.8 | 26.2 | 25.2 |

### 64 MiB decompression throughput

Each CPU/CUDA pair decodes the **same compressed stream**. Codec-produced and
stdlib level-6 streams have different block layouts and must be compared separately.
Host-to-host level-6 decoding includes both transfers and host byte conversion.

| Workload | CUDA resident, codec stream | CPU, codec stream | CUDA resident, level-6 stream | CUDA host-to-host, level-6 stream | CPU, level-6 stream |
| --- | ---: | ---: | ---: | ---: | ---: |
| Zero bytes | 5489.0 | 178.2 | 512.1 | 313.3 | 180.2 |
| Generated text | 2698.0 | 236.6 | 898.0 | 428.5 | 290.0 |
| Integer counters | 1368.1 | 166.2 | 1738.3 | 546.5 | 165.9 |
| Gaussian float32 | 955.1 | 113.8 | 1024.4 | 430.2 | 111.5 |
| Uniform random bytes | 2195.7 | 421.1 | 2097.2 | 566.7 | 419.0 |

For example, resident zero-byte decoding reaches 5489.0 MiB/s for this codec's
stream and 512.1 MiB/s for the level-6 stream. This difference is a property of
the stream layout and decoder paths, not a general speedup over the CPU.

### 64 KiB transfer and launch costs

At 64 MiB, host-to-host CUDA compression exceeded both CPU compression baselines
for all five workloads in this run. At 64 KiB, every measured host-to-host CUDA
compression and level-6 decode was slower than its CPU counterpart:

| Workload | CUDA host compression | CPU level-1 compression | CUDA host decode, level-6 stream | CPU decode, level-6 stream |
| --- | ---: | ---: | ---: | ---: |
| Zero bytes | 9.5 | 552.9 | 22.0 | 251.9 |
| Generated text | 5.1 | 195.5 | 8.8 | 517.3 |
| Integer counters | 3.6 | 51.8 | 4.7 | 202.2 |
| Gaussian float32 | 3.9 | 22.1 | 8.2 | 139.5 |
| Uniform random bytes | 8.0 | 31.1 | 55.3 | 1607.4 |

[Raw RTX 3090 results](benchmarks/results/rtx3090.json) include all three input
sizes, every sample, min/max timings, host-to-device compression, encoded sizes,
startup measurements and environment metadata. No cross-machine speedup is inferred
from the Apple measurements below.

## Recorded CPU baselines

Measured 2026-10-06 on Apple M4 Max, macOS arm64, Python 3.14.6,
NumPy 2.5.3 and stdlib zlib 1.2.12. Each operation has one untimed warmup
and five measured samples. Timings are single-threaded wall-clock medians;
output allocation is included. This is one run on one workstation.

Encoded percentage is compressed bytes divided by input bytes; lower is better.
Throughput is uncompressed MiB divided by elapsed seconds. Values over 100%
indicate expansion. Decode uses the stream produced by zlib level 6.

| Input | Workload | Encoded, level 1 | Encoded, level 6 | Compress level 1, MiB/s | Compress level 6, MiB/s | Decode level 6, MiB/s |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 64 KiB | Zero bytes | 0.468% | 0.128% | 1529.1 | 622.7 | 1554.4 |
| 64 KiB | Generated text | 12.273% | 10.779% | 797.9 | 233.8 | 3401.4 |
| 64 KiB | Integer counters | 34.636% | 34.639% | 190.0 | 18.2 | 698.3 |
| 64 KiB | Gaussian float32 | 92.839% | 92.599% | 58.2 | 45.3 | 529.1 |
| 64 KiB | Uniform random bytes | 100.040% | 100.040% | 88.6 | 90.6 | 10714.9 |
| 1 MiB | Zero bytes | 0.438% | 0.099% | 1217.7 | 571.8 | 3473.7 |
| 1 MiB | Generated text | 12.127% | 10.333% | 559.7 | 184.7 | 4042.4 |
| 1 MiB | Integer counters | 34.590% | 34.578% | 173.1 | 16.8 | 661.9 |
| 1 MiB | Gaussian float32 | 92.888% | 92.625% | 46.2 | 37.8 | 522.9 |
| 1 MiB | Uniform random bytes | 100.031% | 100.031% | 69.5 | 69.6 | 11650.4 |
| 64 MiB | Zero bytes | 0.436% | 0.097% | 1193.7 | 561.2 | 4256.4 |
| 64 MiB | Generated text | 12.121% | 10.293% | 586.7 | 193.8 | 3938.9 |
| 64 MiB | Integer counters | 34.589% | 34.576% | 173.1 | 16.7 | 652.3 |
| 64 MiB | Gaussian float32 | 92.886% | 92.617% | 45.7 | 37.1 | 514.0 |
| 64 MiB | Uniform random bytes | 100.030% | 100.031% | 70.2 | 68.0 | 12113.2 |

[Raw CPU results](benchmarks/results/apple-cpu.json) include every sample and
both levels' decoding measurements. These baselines cannot establish GPU speedup
across machines. CUDA comparisons should use the CPU measurements from the same
CUDA run and the same payload hashes.

## Reproduce

Use a current `main` checkout, which includes the benchmark harness added after
`v0.1.0a1`. Install the package and run:

```sh
python -m pip install ".[cuda12]"
CUPY_CACHE_DIR="$(mktemp -d)" python benchmarks/benchmark.py \
  --sizes 65536 1048576 67108864 --samples 5 --cpu-samples 3 \
  --seed 20261006 --device 0 --output results-cuda.json
```

For the recorded RTX 3090 environment, pin `numpy==2.2.6` and
`cupy-cuda12x==13.3.0`, and select the CUDA 12.6 toolkit with
`CUDA_PATH=/usr/local/cuda-12.6` if a different toolkit is the system default.
The installed codec module hashes and payload hashes should match the raw results.

A fresh CuPy cache separates initial kernel compilation from warm throughput.
The harness records import/CUDA initialization and `compile_kernels` durations
separately. It performs one untimed warmup before each timed operation and
synchronizes the device before and after CUDA calls. Avoid other active GPU
work during measurements; GPU utilization, memory usage and temperature are
recorded before and after the run, but these snapshots do not prove isolation.

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

For a CPU-only reproduction without CuPy or CUDA:

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
