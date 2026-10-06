# General byte-stream benchmarks

The reproducible harness uses synthetic byte streams without external datasets.
It measures encoded size, compression throughput and decompression throughput.
The current recorded results are CPU baselines; fresh CUDA results for these
workloads are pending. These CPU figures do not measure the CUDA codec.

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
