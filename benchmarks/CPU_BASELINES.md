# Independent Apple CPU baselines

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

[Raw CPU results](results/apple-cpu.json) include every sample and
both levels' decoding measurements. These baselines cannot establish GPU speedup
across machines. CUDA comparisons should use the CPU measurements from the same
CUDA run and the same payload hashes.

See [the same-host CUDA comparison](../BENCHMARKS.md) for GPU speedup ratios.
