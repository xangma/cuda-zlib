# Decompression experiment tracker

Reviewed 2026-10-10 against `98b756e32cdb74246a81f4d77f0d792023396430`.
This file tracks hypotheses, controls and decisions; published performance results
remain in [BENCHMARKS.md](../BENCHMARKS.md) and
[PROFILING.md](../PROFILING.md). The review added no CUDA implementation or GPU
measurements. Source-supported opportunities are not established speedups.

## Review outcome

The analysis identifies useful structural targets: expose more emission work,
avoid fallback work after authenticated fast success, and remove avoidable host
waits. Arithmetic token decoding and captured Huffman metadata are contained
starting experiments. Dynamic-block checkpoints have greater scope and need a
separate correctness and memory design.

Adjustments to the proposed programme:

- Inspect generated instructions before replacing tables or bit reversal. Source
  syntax does not establish memory loads or instruction cost.
- The current pipeline has a scalar consumer at thread 32. Its scalar helper
  has no second-warp lane bug; a **new cooperative consumer** must use warp-local
  lane indices and preserve queue ownership and cross-token ordering.
- Four-worker parser packing has already regressed integer decoding. Complete
  the prepared shared-reservation control before another packing proposal.
- Descriptor Nsight Compute counters already exist. Extend those measurements
  to emission, discovery and stored validation rather than repeat broad traces.
- Conditional graphs need a compatibility preflight: recorded native builds use
  CUDA 12.1, while conditional nodes first appeared in CUDA 12.3.
- Refinement already follows up to 32 links per output byte. Pointer jumping
  alone is not a new experiment.
- Array/memoryview output and Python bytes have different materialisation costs;
  track their completed timings separately.

## Baseline and evidence

Runtime sources at the reviewed commit match measurement source
`bb00aea62bae71542cbb896b339da4842c976499`. Recheck that relationship for each new
run; do not silently carry this baseline forward.

The [published Nsight Systems data](../benchmarks/results/nsight/rtx4090-decode.json)
contains 16 complete cases on RTX 4090. The following percentages are shares of
summed recorded **kernel duration** from one warmed, checked, resident JIT call
per trace. They are not end-to-end speedups or instruction-bottleneck diagnoses.

| Output/workload | Dominant recorded stages | Experiment focus |
| --- | --- | --- |
| 64 MiB zeros | Emission 60.20%; description 33.05% | E01, E04, L01 |
| 64 MiB text | Emission 45.57%; description 24.71%; refinement 21.65% | E01, E04, R01–R02 |
| 64 MiB uint32 counters | Emission 42.60%; description 37.93% | D01–D02, E01, E04 |
| 64 MiB float32 | Discovery 35.61%; description 25.46%; emission 22.00% | S01–S03, D01–D02 |
| 64 MiB random | Chain selection 76.88%, including stored-chain probing | F01, F04 |
| 128 KiB uint32 counters | Description plus emission 98.61% | D01–D02; route-specific emission |

The [host-output benchmark](../benchmarks/results/host-outputs/rtx4090-float32.json)
records 64 MiB float32 decoding at 34.895 ms for CUDA host-array output and
87.384 ms for CUDA Python bytes. The corresponding CPU zlib results are
466.839 ms and 466.343 ms. Compare equal output contracts; kernel-only timings
cannot account for a required bytes allocation/copy.

Source references below use these aliases and the reviewed commit's line numbers:

| Alias | Source | Relevant entry points |
| --- | --- | --- |
| K | [_decode_kernels.py](../src/cuda_zlib/_decode_kernels.py) | `Huffman::decode`, `parse_block`, `pipeline_consume`, `scan_prefixes`, `select_chain_impl`, `emit_blocks_pipeline` |
| B | [batch_decode.cuh](../src/cuda_zlib/native/batch_decode.cuh) | `SmallHuffmanDecode`, `SmallTokenRemainder` |
| N | [codec_ffi.cu](../src/cuda_zlib/native/codec_ffi.cu) | `Decompress`, `DecompressBatchImpl`, `ProbeStoredBlocks`, `RefinementRounds`, workspace ownership |
| P | [_postprocess.py](../src/cuda_zlib/_postprocess.py) | `refine_roots`, output gathering and checksum kernels |

Review used graph Verify evidence from generation `2026-10-09T14:23:45Z`, with
best-effort coverage checks on six runtime files and direct source checks for
embedded CUDA. Profile JSON was read directly because its indexed metadata had
changed. Review receipts and prior diagnostic artifacts listed below are
workspace-local paths, outside this repository; publish their manifests and raw
data before using them as public benchmark evidence.

## Existing work and controls

| Record | State | Observation and implication | Workspace evidence |
| --- | --- | --- | --- |
| OBS01 | Diagnostic complete | Four-worker descriptors reserve 8848 B static shared versus 2208 B; allocated shared rises 3328→9984 B. Theoretical occupancy falls 50→20.83%; achieved occupancy also falls. Integer local-load sectors decrease, so a new-spill explanation is unsupported. Relative causes remain unisolated. | `outputs/parser-pressure-20261009/RESULTS.md` |
| OBS02 | Blanket variant rejected | Unprofiled descriptor timing: float32 4.916→3.317 ms, uint32 4.548→6.430 ms. This is not a universal packing win. | `outputs/parser-workers-20261009/RESULTS.md` |
| PRE01 | Prepared; not run | Keep one worker, grid and assignment unchanged; add 6640 B dynamic shared only to the large ordinary descriptor launch. Verify actual allocation/occupancy before attributing any change to reservation. No native build or GPU result exists yet. | `outputs/shared-reservation-20261010-v1/PLAN.md` |
| OBS03 | Host-copy diagnostic complete | Retaining allocator arenas reduced median 64 MiB pinned-buffer `.tobytes()` copy time from about 52.5 to 3.7 ms; each measured group still had a slow first copy. Peak RSS rose roughly 88 MiB. This CPU-copy proxy is not a codec speedup or a library allocator-setting recommendation. | `outputs/host-copy-20261010-v1/RESULTS.md` |
| OBS04 | Existing API | Host array output already uses readonly pinned storage; memoryview shares it. Converting either to bytes still allocates/copies. | N; host-output benchmark above |
| OBS05 | Algebra verified only | Tile-local Adler identity matched Python zlib for 48 cases, including empty, all-255, random and tile boundaries; 100 composition checks passed. No CUDA performance validation. | `outputs/analysis-review-20261010/algebra-checks.json`, `check_algebra.py` |

## Experiment catalogue

### Campaign checks, 2026-10-10

Independent variants are being tested on roni1's two RTX 3090 GPUs with
CUDA 12.6, JAX/JAXlib 0.11.2 and Python 3.13.3. Each comparison keeps its
baseline and candidate on the same physical GPU. These checks establish
correctness or diagnostic observations; no candidate has been accepted or
published as faster yet.

| Trial | Observed checks | Pending decision |
| --- | --- | --- |
| D01 arithmetic | CUDA build and 36 focused checks pass; generated code removes four local loads, 50 local stores and 192 stack bytes in each affected kernel | Completed paired latency and profile |
| D02 metadata | CUDA build and 36 focused checks pass; repeated metadata loads move before parser loops | Completed paired latency and profile |
| E01 cooperative consumer | 36 existing checks and 11 targeted fixtures pass, including recovery and queued outputs; memcheck and synccheck pass | Racecheck reports 340 hazards in both original and candidate; isolate tool/protocol behavior before acceptance |
| C01 local Adler reductions | CUDA build and full suite: 680 passed, four skipped | Completed compression/decompression latency and profiles |
| S02 grouped Kraft lookup | CUDA build and 36 focused checks pass | Completed discovery latency and profile |
| B01 decoder descriptor operand | CUDA build passes; CPU lowering and bounds proofs complete | GPU batch acceptance and completed latency |
| F04 regular stored layout | Exact published zlib fixture has 26 distinct nonfinal lengths and rejects the hypothesis; native-produced fixture accepts | Measure native-input gain and extra fallback cost separately |
| R02 refinement walk | Independent walks 1/4/8 compile; proven safe maximum schedules are 18/8/6 rounds | GPU checks and latency; extra submitted launches may outweigh shorter walks |

The first zeros `emit_blocks_pipeline` Nsight Compute capture executes only its
early-exit guard. Its counters cannot characterize productive emission. Targeted
queue fixtures are used for the replacement capture. A failed paired-harness
identity check stopped before timing; its unavailable runtime UUID call is being
replaced with the documented driver API, retaining all source/device checks.

P1 = first controlled trials or prerequisite measurement; P2 = next targeted
trials; P3 = later research or optional backend/format work. All rows below are
**planned** unless another state is explicit. Each candidate needs its own
baseline comparison and result entry. Dependencies refer to stable IDs.

| ID | Priority/state | Experiment and decisive check |
| --- | --- | --- |
| D01 | P1 | Share arithmetic length/distance decoding with the general and pipeline parsers. Compare SASS, registers and completed latency; preserve invalid-symbol and extent checks. |
| D02 | P1 | Capture immutable Huffman maximum/lookup metadata after construction. Trial separately from D01; confirm any load reduction survives compilation without harmful register growth. |
| D03 | P3 | Compare `__brev` with manual reversal during table construction. Stop if baseline already emits equivalent instructions or construction cost is immaterial. |
| D04 | P1, measurement | Count true primary-table misses, short-tail fallbacks, code lengths and fixed-table construction cost. Keep instrumented timing out of performance comparisons. |
| D05 | P2; needs D04 | Bounded secondary Huffman tables where misses justify them. Include build time, scratch and occupancy; preserve canonical fallback for legal short tails and errors. |
| D06 | P2; needs D04 | Immutable precomputed fixed tables. Test fixed-heavy and mixed streams; reject a build-time saving that loses to table traffic or shared reservation. |
| E01 | P1 | Cooperative pipeline consumer warp for long matches. Measure useful lanes, rendezvous cost and completed emission latency; retain a cheap short-match/literal path. |
| E02 | P1, design first | Per-block serial/pipeline/warp/stored ownership instead of whole-stream pipeline gating. Trial mixed blocks; require disjoint output/status ownership and a gain after dispatch overhead. |
| E03 | P2; needs PRE01 | Revisit independent parser packing using measured resource limits and underfill. OBS02 rejects blanket four-worker deployment; isolate worker count from reservation and code layout. |
| E04 | P2, design first | Record token-boundary checkpoints during description, then emit accepted dynamic blocks as segments. Sweep token/output spacing, including 16–64 KiB output targets; measure added external references, refinement and memory. |
| E05 | P3, research | Composable dynamic-payload tile transitions after table discovery. Prove state/boundary handling and bound speculative work before benchmarking; fixed-code summaries are not a drop-in solution. |
| S01 | P2 | Aligned bulk input loads plus shuffles/funnel shifts, with bounded head/tail paths. Require identical candidates and fewer instructions or lower discovery time; cache hits alone are not a bandwidth diagnosis. |
| S02 | P2 | A 512-entry table sums three 3-bit Kraft contributions per lookup. Preserve exact tail masking and rejection; compare divergent lookup cost against current arithmetic and early exits. |
| S03 | P1, measurement | Inspect compiled `dynamic_valid` frame, local traffic, register count and memory issue pressure. Change storage only if those measurements identify material cost. |
| F01 | P1, design first | Avoid capacity-sized CUB sorting after authenticated stored/medium success. Distinguish sort work, gated launch overhead and eager workspace cost; preserve fallback without adding a blocking success readback. |
| F02 | P2 | Plan invocation-owned, JAX-supplied scratch operands. Verify sizes, aliasing, simultaneous calls and stream lifetime before attempting command-buffer compatibility. |
| F03 | P2; needs F02 and compatibility preflight | Separate plain graph replay from conditional fallback/refinement trials. Verify installed CUDA/JAX/XLA support, capture restrictions and concurrency; measure submission and kernel work separately. |
| F04 | P2 | Guarded regular stored-block hypothesis with parallel header validation. Validate all inferred locations and exact chain extent; send valid irregular layouts to the existing fallback. |
| B01 | P1 | Supply batch descriptors as device operands, or retain immutable staging/cache storage until completion. Remove the lifetime-protecting host wait only after proving readiness and queued-use ownership. |
| L01 | P2; frequency first | Coalesce consecutive equal-distance matches, starting with periods 2/4/8/16/32. Reuse seed roots while preserving phase, token validation and boundaries; measure frequency and rendezvous savings. |
| C01 | P2; algebra verified, GPU untested | Tile-local 32-bit Adler reductions with 64-bit global combination. Confirm tails and failure gates, then measure reduction/barrier cost and completed decoding. |
| C02 | P3; needs C01 | Packed byte sums/local weights using unsigned `__dp4a`. Prove coefficient packing and overflow bounds; reject if packing/gather cost exceeds the arithmetic saving. |
| R01 | P3 | Compact unresolved positions after most roots become literals. Include compaction traffic, target-root retention and extra launches in the whole-call comparison. |
| R02 | P2, proof first | Sweep refinement walk limits 1/4/8/32 with a matching round/depth guarantee. Do not retain the current round count blindly after reducing hops. |
| R03 | P3, research | Retain literals/match-range descriptors and resolve coarser dependencies before expansion. Budget metadata and lookup work; include literal-heavy cases and avoid per-byte binary searches. |
| H01 | P2, API proposal | Caller-owned host output storage for repeated decoding. Specify ownership, mutability and async lifetime; compare allocation, transfer and materialisation separately. Python bytes remains a separate contract. |
| A01 | P3; needs a useful tiled schedule | `cp.async` compressed-input/shared staging. Demonstrate useful producer/consumer overlap with alignment and synchronization; this is not asynchronous global-to-global LZ copying. |
| A02 | P3, optional reference | Benchmark nvCOMP ordinary Deflate on trusted valid inputs, especially batches. Report wrapper/checksum/transfer contracts; do not substitute it for the strict malformed-input decoder. |
| A03 | P3, separate format | Evaluate GDeflate only when the producer/format can change. It does not decode arbitrary existing zlib streams. |
| A04 | Deferred: other hardware | Evaluate dedicated decompression on supported Blackwell hardware, not the profiled RTX 4090. Query chunk limits and verify input, output and scratch allocation compatibility independently. |

## Correctness and design constraints

### Token decoding and Huffman tables: D01–D06

B:17/44 already contains captured metadata and arithmetic token mapping.
K:660/678/834 retains base/extra arrays in the general parsers. Preserve reserved
symbols and the existing output-extent-before-history/window error precedence.
K:195 restarts canonical decoding from length one after a primary miss; insufficient
bits near EOF can also take that path. Secondary lookup instrumentation must
distinguish those cases and retain empty-distance/single-symbol legality.
Manual reversal at K:126 is used in table construction, not every decoded token.

### Emission ordering and routing: E01–E05, L01

N:907 launches 64 threads, but K:1436–1444 uses only thread 0 for production and
thread 32 for scalar consumption. A consumer warp needs `threadIdx.x & 31` or an
explicit group. Release a queue slot only after every participating lane finishes
reading it; publish a match's writes before the next dependent token reads seeds.
Drain the queue before terminal/status publication or table reuse. Parallel bytes
within a match do not establish independence between successive tokens.

Pipeline routing has both an outer compressibility gate (N:755) and per-block
eligibility checks (K:1105). Exact selection uses `has_pipeline && !has_fallback`
(K:1275). E02 must redesign dispatch ownership; simply loosening that condition
can duplicate or omit work and revive costly mixed-path behaviour.

Checkpoints need compressed bit position, output offset, durable accepted-table
identity and original block/final/error limits. Bound speculative storage and
retain a fallback. More segments can increase external references and required
refinement depth. The maximum match token length, 15+5+15+13=48 bits, bounds
boundary overlap; it does not bound dynamic headers or justify enumerating all
48-bit patterns. Terminate on EOB before a table change.
[DEFLATE specification](https://www.rfc-editor.org/rfc/rfc1951.html).

For equal-distance coalescing, preserve periodic phase, bounded aggregation,
reader rollback and validation of every original token. Do not cross EOB or
segment boundaries. Existing distance-one coalescing is already implemented.

### Discovery: S01–S03

K:959 assembles up to eleven bounded bytes. Nineteen 3-bit lengths plus header
fields and starting-bit offset can require 81 bits; a single 64-bit replacement
is insufficient. Do not introduce arbitrary unaligned word dereferences or
out-of-extent reads. Preserve candidate multiset/count, overflow and fallback.
For S02 use `f(0)=0`, `f(l)=2^(7-l)`, and
`T[a+8*b+64*c]=f(a)+f(b)+f(c)`; maximum 192 fits a byte. Mask unused lengths
and preserve rejection above 128. Divergent constant-memory lookups can serialize.

### Orchestration and scratch: F01–F04, B01

The small route returns early (N:682). The stored/medium routes can succeed yet
still reach capacity-sized CUB sorting (N:810), summary allocation (N:864) and
default root allocation (N:892). Sorting already uses bits `[0,32)` of U64 keys;
reducing a supposed 64-bit radix range would repeat an existing optimisation.
Device predicates alone do not remove host allocations or opaque CUB work.

OpenXLA's command-buffer trait requires, among other conditions, passed device
buffer allocations and no runtime allocation or stream-status query. Check the
**installed** headers before implementing F02/F03; do not merely add the trait.
[OpenXLA FFI API](https://raw.githubusercontent.com/openxla/xla/main/xla/ffi/api/api.h).
Conditional graph bodies allow a restricted node set that excludes allocation
nodes. Preplanned scratch and concurrent-invocation ownership remain necessary.
[CUDA graph requirements](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html#conditional-node-body-graph-requirements).
Conditional nodes were introduced in CUDA 12.3; the recorded CUDA 12.1 builds
therefore do not establish support.
[CUDA 12.3 guide](https://docs.nvidia.com/cuda/archive/12.3.0/cuda-c-programming-guide/index.html).

B01 replaces a real ownership guarantee: N:1099 synchronizes after uploading a
local descriptor vector. A cache must distinguish device/context and full layout,
establish cross-stream readiness, and retain entries through queued uses. An
operand avoids those cache-lifetime questions. Deleting the wait alone is unsafe.

F04 must validate BTYPE, BFINAL, LEN/NLEN, wrapper, bounds and exact final input
and output extents, including empty blocks. A rejected regular-layout hypothesis
must preserve decoding of valid irregular stored chains and checksum checking.

### Adler and refinement: C01–C02, R01–R03

For a tile of actual length `n <= 4096`, beginning at output offset `o`, define:

```text
S = sum(x[j])
W = sum((n-j)*x[j])
tile_global_weight = W + (N-o-n)*S
A = (1 + sum(S)) mod 65521
B = (N + sum(tile_global_weight)) mod 65521
```

Bounds: `S <= 1,044,480`, `W <= 2,139,617,280`. Tile sums fit U32;
**widen before** multiplying by the global offset. Use actual tail length;
empty input gives A=1, B=0. Adjacent summaries compose as:

```text
(SX,WX,nX) compose (SY,WY,nY)
  = (SX+SY, WX+WY+nY*SX, nX+nY)
```

Keep parser/refinement errors ahead of checksum work; never read unwritten
partials or roots, and preserve existing failed-output/status behaviour.
Native 32-bit warp integer reduction support is architecture-dependent; measure
the resulting implementation rather than assuming instruction savings imply
decoding gains.
[Ampere tuning guide](https://docs.nvidia.com/cuda/archive/11.4.1/ampere-tuning-guide/index.html).

The default general route reserves two U32 root planes: 512 MiB for 64 MiB of
output, excluding other scratch. Small decoding avoids them; `max_blocks=1`
aliases the alternate plane. P:17 already follows up to 32 links per byte from
an immutable snapshot. N:442 budgets rounds using capacity, with at most four
default passes. Smaller hop limits and additional checkpoint segments need a
new depth/schedule guarantee. Compaction must retain root targets, monotonic
reference checks, pending reduction and completed-snapshot publication.

### Optional CUDA/backend work: A01–A04

`cp.async` is global-to-shared with alignment/completion requirements; use it
only with a justified staging schedule.
[PTX instructions](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html).
nvCOMP documents undefined behaviour for corrupt Deflate inputs and limited
validation. A checksum after decoding does not protect against unsafe malformed
input handling.
[nvCOMP C API](https://docs.nvidia.com/cuda/nvcomp/c_api.html).
GDeflate changes the format to expose parallel decoding.
[nvCOMP formats](https://docs.nvidia.com/cuda/nvcomp/).

NVIDIA lists Deflate-capable decompression engines on B200/B300/GB200/GB300.
Query `CU_DEVICE_ATTRIBUTE_MEM_DECOMPRESS_MAXIMUM_LENGTH`; verify allocation
requirements, including `cudaMemPoolCreateUsageHwDecompress` where applicable.
JAX input/output buffers need independent checks. Hardware chunk limits do not
permit cutting an arbitrary DEFLATE stream at byte offsets.
[Decompression engine FAQ](https://docs.nvidia.com/cuda/nvcomp/decompression_engine_faq.html).

## Measurement and acceptance protocol

1. Pin baseline/candidate commits, native-library hashes, fixture hashes, device,
   toolkit/driver/JAX versions and exact harness command. Keep one causal change
   per initial trial; declare workload and memory/regression budgets before running.
2. Check full output bytes, exact extents/status and metadata. Include malformed
   and truncated streams, bad checksums, mixed block types, cross-block history,
   overlapping matches, empty/tail cases and concurrent queued invocations.
   Preserve fallback/error precedence and recovery after failure.
3. Inspect PTX/SASS, registers, static/dynamic shared allocation and local traffic.
   Use targeted Nsight Compute LaunchStats, Occupancy, SchedulerStats,
   SourceCounters, InstructionStats and MemoryWorkloadAnalysis on dominant kernels.
   Distinguish memory bandwidth from memory-pipeline issue limits. Profiled replay
   duration is diagnostic, not the benchmark outcome.
   [Nsight Compute profiling guide](https://archive.docs.nvidia.com/nsight-compute/2026.2/ProfilingGuide/index.html).
4. Time warmed, paired baseline/candidate runs without instrumentation, retain
   distributions and controls, and cover small/medium/64 MiB zeros, text, uint32,
   float32 and random fixtures plus relevant mixed batches. Record cold behaviour
   and peak memory separately. Include an unchanged control to expose run noise.
5. Report resident checked JIT, host array/memoryview, Python bytes and file/batch
   contracts separately. Include transfer, allocation, validation and materialisation
   in whichever completed-call contract is claimed. CPU comparisons use the same
   data, checking, output type and declared CPU parallelism.
6. Accept only a reproducible completed-latency benefit above control variability,
   within declared correctness/memory/regression budgets. A faster isolated kernel,
   fewer instructions or higher occupancy alone is insufficient. Record neutral
   and rejected trials so the same recommendation is not rediscovered.

## Initial execution order

1. Finish PRE01; inspect D01/D02/S03 codegen and reuse OBS01 counters. These are
   separate controls, not a combined patch.
2. Trial D01 and D02 independently. Collect D04 frequencies while designing E01.
3. Trial E01, then E02 with explicit ownership. Independently resolve F01/F02
   feasibility and B01 descriptor lifetime; compare the relevant fast/batch paths.
4. Design E04 for emission-heavy large blocks. For float32, prioritise S01/S02
   when discovery counters support them. For random/stored data, prioritise F04.
5. Schedule C01 and L01 when their measured contribution/frequency warrants it;
   leave E05/R03 and backend/format changes as separate research tracks.

## Run/result record template

Append one record per immutable candidate; update catalogue state after review.

```text
ID / date / owner / status:
Hypothesis and predeclared acceptance or rejection criteria:
Baseline commit + native hash / candidate commit + native hash:
Host / device / toolkit / driver / JAX / fixture hashes:
Command / cwd / job ID or PID / log / next check / stop command (if running):
Controls and correctness cases / passed counts / failure evidence:
Raw timing distribution + timing contract / counters / peak memory:
Artifacts + manifest hashes / code review:
Decision (keep, reject, inconclusive) and reason:
Next decisive check / dependencies:
```
