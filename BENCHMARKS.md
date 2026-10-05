# Measured behavior

Measured on 2026-10-05 with an RTX 4090, driver 610.57.04, Python 3.12.8,
NumPy 2.2.6, CuPy 13.3.0, JAX/JAXlib 0.11.2 and CUDA runtime 12.6.
H1/L1 labels identify example strain byte vectors, not required input formats.
These measurements used the validated pre-release candidate whose executable
codec logic and CUDA source strings are retained by this MIT release.

Codec workflows used one warmup and five synchronized samples. CPU compression
used three samples. The encoding table shows **first call / warm median in ms**.
First calls share one process and benefit from earlier work; they are not
independent cold starts.

| Payload | Frozen CUDA bytes | New CUDA bytes | zlib 1 bytes | zlib 6 bytes | New size above zlib 6 |
| --- | --- | --- | --- | --- | --- |
| H1 1 MiB | 1,048,742 | 1,015,199 | 1,015,922 | 1,013,805 | 0.14% |
| L1 1 MiB | 1,048,742 | 982,505 | 981,709 | 977,367 | 0.53% |
| H1 64 MiB | 63,758,206 | 58,014,271 | 57,810,594 | 57,893,444 | 0.21% |
| L1 64 MiB | 66,998,538 | 62,216,405 | 62,056,788 | 61,844,345 | 0.60% |

| Payload | CUDA resident → resident | CUDA host → device | CUDA host → host | zlib 1 | zlib 6 |
| --- | --- | --- | --- | --- | --- |
| H1 1 MiB | 123.75 / 12.67 | 12.90 / 12.81 | 13.08 / 12.89 | 40.57 / 41.71 | 44.82 / 44.84 |
| L1 1 MiB | 14.70 / 14.42 | 14.52 / 14.51 | 14.85 / 14.85 | 37.44 / 37.12 | 47.82 / 47.38 |
| H1 64 MiB | 129.54 / 123.90 | 132.02 / 131.56 | 193.14 / 198.96 | 2364.29 / 2360.48 | 3959.41 / 3951.12 |
| L1 64 MiB | 120.51 / 115.16 | 122.76 / 122.66 | 184.58 / 191.85 | 2634.23 / 2645.08 | 3027.28 / 3033.82 |

New CUDA streams decoded in 10.65/10.95 ms for 1 MiB H1/L1 inputs and
28.33/28.78 ms for the corresponding 64 MiB inputs. The earlier encoder's frozen
64 MiB streams took 9.67/2.36 seconds before repair and now take about 37/33 ms.
Old values were single samples; repaired values are warm medians. Independent
before/after GPU profiles confirmed the serial-selection bottleneck and repair.
Original dynamic-stream medians were about 0.4–5% slower in the final run.
Independent stdlib fixed/stored 64 MiB H1 streams took 156.89/19.62 ms warm.

The new 64 MiB outputs were 9.01/7.14% smaller than the earlier fixed-code
encoder's output, but resident encoding rose from 72.46/66.42 ms to
123.90/115.16 ms. On these inputs the new large outputs remained slightly larger
than stdlib level 1. Small dynamic outputs decode more slowly than the earlier,
larger stored outputs. There are independent chunk histories and a bounded
greedy matcher rather than every stdlib zlib compression decision.

An isolated PyCBC development integration's large complete reads took
300.70/319.50 ms with CUDA versus 798.11/831.11 ms on the host path. Both small
complete CUDA reads remained slower than host. All four reads matched the public
LAL reader byte-for-byte, including dtype, sample interval and start time.
Readers cleared parsed-frame caches and constructed a fresh source for every
sample, with the OS file cache warm. These integration timings include parsing,
CRCs, JAX ownership, numeric conversion and slicing; they are not codec timings.

Fresh import/device startup took 4.27 seconds and compiling both modules with a
new NVRTC cache took 10.28 seconds. The private pool retained 0.95 GiB after the
four cases. Startup and instrumented kernel measurements were separate from
warm wall-clock medians. Workload and hardware affect these results.

The pre-release source and installed wheel each passed 157 CUDA tests. The
frozen PyCBC adapter suite passed 187 tests, with two fixture-dependent skips
and one obsolete private-cache test deselected. The macOS installed wheel
without CuPy/JAX passed 24 host-only tests and skipped 133 GPU tests. Coverage
includes independent fixed/stored/dynamic streams, mixed blocks, alignments,
overlapping history, corruption, exact bounds, recovery, output lifetime,
concurrent calls and JAX ownership. Multiple physical GPUs, actual 256 MiB
limits, allocator exhaustion and asynchronous driver failures remain untested.
