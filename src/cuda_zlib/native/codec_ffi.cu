// Copyright (c) 2026 xangma
// SPDX-License-Identifier: MIT

// Fresh CUDA codec execution on the stream supplied by XLA. Generated headers
// contain the unchanged CUDA_SOURCE strings, with no host codec implementation.
#include <cuda_runtime.h>
#include <cuda.h>
#include <dlfcn.h>
#if CUDA_VERSION < 12000
#error "CUDA byte codec workspace pools require CUDA toolkit 12.0 or newer"
#endif
#include <cub/device/device_radix_sort.cuh>
#include <xla/ffi/api/ffi.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

namespace encoder {
#include "encoder.cuh"
}
namespace decoder {
#include "decoder.cuh"
}
namespace checksum {
#include "postprocess.cuh"
}

namespace ffi = xla::ffi;

namespace {
using U8 = unsigned char;
using U32 = unsigned int;
using U64 = unsigned long long;
constexpr U32 kMaxBytes = 1u << 28;
constexpr U32 kMaxCandidates = 262144;
constexpr U32 kMaxBlocks = 262144;
constexpr U32 kFixedTileBytes = 2048;

// Deflate statuses 1..13 come directly from the existing decoder kernels.
// Compress metadata is [encoded extent, status]; decode is [status, 0].
enum Status : U32 {
  kTruncatedZlib = 14,
  kInvalidHeader = 15,
  kUnsupportedGzip = 16,
  kUnsupportedDictionary = 17,
  kCandidateOverflow = 18,
  kInitialBlockMissing = 19,
  kReferenceDepthExceeded = 20,
  kAdlerMismatch = 21,
  kCompressionOverflow = 22,
  kInvalidBounds = 23,
};

#include "batch_encode.cuh"
#include "batch_decode.cuh"

#define CUDA_TRY(expression)                     \
  do {                                           \
    cudaError_t error = (expression);             \
    if (error != cudaSuccess) return error;       \
  } while (false)

// Driver symbols are resolved from the real driver, never the toolkit stub.
// These process-lifetime objects deliberately make no CUDA calls at teardown.
struct Driver {
  decltype(&cuStreamGetCtx) stream_context = nullptr;
  decltype(&cuCtxGetCurrent) current_context = nullptr;
  decltype(&cuCtxGetDevice) context_device = nullptr;
  decltype(&cuCtxGetId) context_id = nullptr;
  decltype(&cuCtxPushCurrent) push_context = nullptr;
  decltype(&cuCtxPopCurrent) pop_context = nullptr;
  bool ready = false;

  Driver() {
    void* handle = dlopen("libcuda.so.1", RTLD_NOW | RTLD_LOCAL);
    if (!handle) return;
#define LOAD_DRIVER(member, symbol) \
    member = reinterpret_cast<decltype(member)>(dlsym(handle, symbol))
    LOAD_DRIVER(stream_context, "cuStreamGetCtx");
    LOAD_DRIVER(current_context, "cuCtxGetCurrent");
    LOAD_DRIVER(context_device, "cuCtxGetDevice");
    LOAD_DRIVER(context_id, "cuCtxGetId");
    LOAD_DRIVER(push_context, "cuCtxPushCurrent_v2");
    LOAD_DRIVER(pop_context, "cuCtxPopCurrent_v2");
#undef LOAD_DRIVER
    ready = stream_context && current_context && context_device && context_id &&
            push_context && pop_context;
  }
  static Driver& Get() {
    static Driver* driver = new Driver;
    return *driver;
  }
};

cudaError_t DriverError(CUresult result) {
  return result == CUDA_SUCCESS ? cudaSuccess : cudaErrorInvalidResourceHandle;
}

class ScopedContext {
 public:
  cudaError_t Enter(CUcontext context) {
    Driver& driver = Driver::Get();
    if (!driver.ready) return cudaErrorInsufficientDriver;
    CUcontext current = nullptr;
    CUDA_TRY(DriverError(driver.current_context(&current)));
    if (current != context) {
      CUDA_TRY(DriverError(driver.push_context(context)));
      pushed_ = true;
    }
    return cudaSuccess;
  }
  ~ScopedContext() noexcept {
    if (pushed_) {
      CUcontext previous;
      Driver::Get().pop_context(&previous);
    }
  }
 private:
  bool pushed_ = false;
};

struct Pool {
  CUcontext context;
  U64 context_id;
  int device;
  cudaMemPool_t handle;
};
struct PoolRegistry {
  std::mutex mutex;
  std::vector<Pool> pools;
  U64 retention = 1ull << 30;
};
PoolRegistry& Pools() {
  static PoolRegistry* registry = new PoolRegistry;
  return *registry;
}

cudaError_t GetPool(CUcontext context, cudaMemPool_t* result) {
  Driver& driver = Driver::Get();
  U64 id;
  int device;
  CUDA_TRY(DriverError(driver.context_id(context, &id)));
  CUDA_TRY(DriverError(driver.context_device(&device)));
  PoolRegistry& registry = Pools();
  std::lock_guard<std::mutex> lock(registry.mutex);
  for (const Pool& pool : registry.pools) {
    if (pool.context_id == id && pool.device == device) {
      *result = pool.handle;
      return cudaSuccess;
    }
  }
  cudaMemPoolProps properties{};
  properties.allocType = cudaMemAllocationTypePinned;
  properties.handleTypes = cudaMemHandleTypeNone;
  properties.location.type = cudaMemLocationTypeDevice;
  properties.location.id = device;
  cudaMemPool_t handle;
  CUDA_TRY(cudaMemPoolCreate(&handle, &properties));
  cudaError_t error = cudaMemPoolSetAttribute(
      handle, cudaMemPoolAttrReleaseThreshold, &registry.retention);
  if (error != cudaSuccess) {
    cudaMemPoolDestroy(handle);
    return error;
  }
  try {
    registry.pools.push_back({context, id, device, handle});
  } catch (...) {
    cudaMemPoolDestroy(handle);
    throw;
  }
  *result = handle;
  return cudaSuccess;
}

cudaError_t InspectPools(int device, U64* stats, bool trim) {
  std::vector<Pool> pools;
  {
    PoolRegistry& registry = Pools();
    std::lock_guard<std::mutex> lock(registry.mutex);
    stats[0] = registry.retention;
    stats[1] = stats[2] = stats[3] = 0;
    for (const Pool& pool : registry.pools)
      if (pool.device == device) pools.push_back(pool);
  }
  Driver& driver = Driver::Get();
  if (!pools.empty() && !driver.ready) return cudaErrorInsufficientDriver;
  for (const Pool& pool : pools) {
    U64 id;
    CUDA_TRY(DriverError(driver.context_id(pool.context, &id)));
    if (id != pool.context_id) return cudaErrorInvalidResourceHandle;
    ScopedContext context;
    CUDA_TRY(context.Enter(pool.context));
    if (trim) CUDA_TRY(cudaMemPoolTrimTo(pool.handle, 0));
    U64 reserved, used;
    CUDA_TRY(cudaMemPoolGetAttribute(
        pool.handle, cudaMemPoolAttrReservedMemCurrent, &reserved));
    CUDA_TRY(cudaMemPoolGetAttribute(
        pool.handle, cudaMemPoolAttrUsedMemCurrent, &used));
    stats[1] += reserved;
    stats[2] += used;
    ++stats[3];
  }
  return cudaSuccess;
}

// Every invocation owns its pointers; allocation reuse and dependencies are
// managed by CUDA. All frees follow the final use on the supplied XLA stream.
class Workspace {
 public:
  explicit Workspace(cudaStream_t stream) : stream_(stream) {
    Driver& driver = Driver::Get();
    if (!driver.ready) {
      error_ = cudaErrorInsufficientDriver;
      return;
    }
    CUcontext context = nullptr;
    error_ = DriverError(driver.stream_context(
        reinterpret_cast<CUstream>(stream), &context));
    if (error_ == cudaSuccess) error_ = context_.Enter(context);
    if (error_ == cudaSuccess) error_ = GetPool(context, &pool_);
  }
  Workspace(const Workspace&) = delete;
  Workspace& operator=(const Workspace&) = delete;
  ~Workspace() noexcept { Release(); }
  cudaError_t error() const { return error_; }

  template <typename T>
  cudaError_t Allocate(T** result, std::size_t elements) {
    if (count_ == allocations_.size()) return cudaErrorMemoryAllocation;
    void* pointer = nullptr;
    CUDA_TRY(error_);
    CUDA_TRY(cudaMallocFromPoolAsync(
        &pointer, std::max<std::size_t>(1, elements) * sizeof(T), pool_, stream_));
    allocations_[count_++] = pointer;
    *result = static_cast<T*>(pointer);
    return cudaSuccess;
  }

  cudaError_t Release() noexcept {
    cudaError_t first = cudaSuccess;
    while (count_) {
      cudaError_t error = cudaFreeAsync(allocations_[--count_], stream_);
      if (first == cudaSuccess) first = error;
    }
    return first;
  }

 private:
  cudaStream_t stream_;
  ScopedContext context_;
  cudaMemPool_t pool_ = nullptr;
  cudaError_t error_ = cudaSuccess;
  std::array<void*, 32> allocations_{};
  std::size_t count_ = 0;
};

template <ffi::DataType Type>
bool IsVector(const ffi::Buffer<Type>& buffer) {
  auto dimensions = buffer.dimensions();
  return dimensions.size() == 1 && dimensions[0] >= 0;
}

__global__ void SetMetadata(U32* metadata, U32 first, U32 second) {
  if (!blockIdx.x && !threadIdx.x) {
    metadata[0] = first;
    metadata[1] = second;
  }
}

cudaError_t SetStatus(cudaStream_t stream, U32* metadata, U32 status,
                      bool compression = false) {
  SetMetadata<<<1, 1, 0, stream>>>(metadata, compression ? 0u : status,
                                 compression ? status : 0u);
  return cudaGetLastError();
}

cudaError_t ReadWords(cudaStream_t stream, const U32* device, U32* host,
                      std::size_t count) {
  CUDA_TRY(cudaMemcpyAsync(host, device, count * sizeof(U32),
                           cudaMemcpyDeviceToHost, stream));
  return cudaStreamSynchronize(stream);
}

// A bounded serial device scan avoids reading per-chunk sizes/status on host.
// No invalid or unwritten chunk size can reach packing after an encoder error.
__global__ void CompressionPrefix(const U32* sizes, const U32* status,
                                  U32 chunks, U32 slot_bytes, U64 capacity,
                                  U64* ends, U32* metadata) {
  if (blockIdx.x || threadIdx.x) return;
  metadata[0] = 0;
  metadata[1] = 0;
  for (U32 i = 0; i < chunks; ++i) {
    if (status[i]) { metadata[1] = kCompressionOverflow; return; }
  }
  U64 total = 6;
  for (U32 i = 0; i < chunks; ++i) {
    if (sizes[i] > slot_bytes || total + sizes[i] > capacity) {
      metadata[1] = kCompressionOverflow;
      return;
    }
    total += sizes[i];
    ends[i] = total - 6;
  }
  metadata[0] = U32(total);
}

__global__ void EmissionStatus(const U32* input, U32 blocks,
                               checksum::DecodeState* state) {
  __shared__ U32 errors[256], pending[256];
  U32 error = 0, external = 0;
  for (U32 i = threadIdx.x; i < blocks; i += blockDim.x) {
    error = max(error, input[i]);
    external = max(external, input[blocks + i]);
  }
  errors[threadIdx.x] = error;
  pending[threadIdx.x] = external;
  __syncthreads();
  for (U32 stride = 128; stride; stride >>= 1) {
    if (threadIdx.x < stride) {
      errors[threadIdx.x] = max(errors[threadIdx.x], errors[threadIdx.x + stride]);
      pending[threadIdx.x] = max(pending[threadIdx.x], pending[threadIdx.x + stride]);
    }
    __syncthreads();
  }
  if (!threadIdx.x) {
    state->status = errors[0];
    state->active = !errors[0] && pending[0];
    state->selector = 0;
    state->pending = 0;
    state->refine_status = 0;
  }
}

// Each valid nonliteral emission root points before its accepted block's
// prefix: local copies inherit a literal or that same earlier-block index.
// Thus the initial reference depth is at most blocks-1. Every refinement pass
// follows 32 links in the preceding snapshot; ceil(log32(blocks)) passes are
// sufficient, with at most four passes under kMaxBlocks.
U32 RefinementRounds(U32 blocks) {
  U32 rounds = 0, covered = 1;
  while (covered < blocks) { covered *= 32; ++rounds; }
  return rounds;
}

__global__ void RefinementStatus(checksum::DecodeState* state, U32 last_round) {
  if (blockIdx.x || threadIdx.x || !state->active) return;
  if (state->refine_status) {
    state->status = state->refine_status;
    state->active = 0;
    return;
  }
  // Advance only after a completed pass. Skipped launches must not select an
  // unwritten alternate buffer or move away from the first resolved snapshot.
  state->selector ^= 1u;
  if (!state->pending) state->active = 0;
  else if (last_round) {
    state->status = kReferenceDepthExceeded;
    state->active = 0;
  }
  state->pending = 0;
  state->refine_status = 0;
}

__global__ void VerifyDecompression(const checksum::DecodeState* state,
                                    const U32* actual, U32 wanted,
                                    U32* metadata) {
  if (blockIdx.x || threadIdx.x) return;
  U32 error = state->status;
  if (!error && state->active) error = kReferenceDepthExceeded;
  // Gather/checksum are skipped on prior failure, so actual may be unwritten.
  if (!error && *actual != wanted) error = kAdlerMismatch;
  metadata[0] = error;
  metadata[1] = 0;
}

cudaError_t Compress(cudaStream_t stream, std::int64_t chunk_bytes,
                      ffi::Buffer<ffi::U8> input,
                      ffi::ResultBuffer<ffi::U8> output,
                      ffi::ResultBuffer<ffi::U32> metadata,
                      Workspace& workspace) {
  U32* result = metadata->typed_data();
  if (!IsVector(*output) || output->element_count() > kMaxBytes)
    return SetStatus(stream, result, kInvalidBounds, true);
  const U64 capacity = U64(output->element_count());
  if (capacity) CUDA_TRY(cudaMemsetAsync(output->typed_data(), 0, capacity, stream));
  if (!IsVector(input) || input.element_count() > kMaxBytes ||
      chunk_bytes < 256 || chunk_bytes > 65535)
    return SetStatus(stream, result, kInvalidBounds, true);
  const U64 size = U64(input.element_count());
  const U32 chunk = U32(chunk_bytes);
  const U32 chunks = U32(std::max<U64>(1, (size + chunk - 1) / chunk));
  if (2u * U64(chunks) - 1 > kMaxBlocks ||
      size + U64(chunks) * 5 + 6 > kMaxBytes ||
      capacity != size + U64(chunks) * 5 + 6)
    return SetStatus(stream, result, kInvalidBounds, true);
  const U32 slot_bytes = (chunk * 9 + 7) / 8 + 16;
  U8* scratch = nullptr;
  U32 *tokens = nullptr, *sizes = nullptr, *status = nullptr, *adler = nullptr;
  U64 *ends = nullptr, *partial_a = nullptr, *partial_b = nullptr;
  CUDA_TRY(workspace.Allocate(&scratch, std::size_t(chunks) * slot_bytes));
  CUDA_TRY(workspace.Allocate(&tokens, size));
  CUDA_TRY(workspace.Allocate(&sizes, chunks));
  CUDA_TRY(workspace.Allocate(&status, chunks));
  encoder::encode_chunks<<<chunks, 256, 0, stream>>>(
      input.typed_data(), U32(size), chunk, chunks, slot_bytes, scratch,
      sizes, status, tokens);
  CUDA_TRY(cudaGetLastError());
  if (size <= 65536) {
    SmallFinish<<<1, 256, 0, stream>>>(
        input.typed_data(), U32(size), chunks, slot_bytes, scratch, sizes,
        status, output->typed_data(), U32(capacity), result);
    return cudaGetLastError();
  }
  CUDA_TRY(workspace.Allocate(&ends, chunks));
  CompressionPrefix<<<1, 1, 0, stream>>>(sizes, status, chunks, slot_bytes,
                                         capacity, ends, result);
  CUDA_TRY(cudaGetLastError());
  std::array<U32, 2> host{};
  CUDA_TRY(ReadWords(stream, result, host.data(), host.size()));
  if (host[1]) return cudaSuccess;
  encoder::pack_chunks<<<chunks, 256, 0, stream>>>(
      scratch, slot_bytes, sizes, ends, output->typed_data());
  CUDA_TRY(cudaGetLastError());
  const U32 parts = U32(std::max<U64>(1, (size + 4095) / 4096));
  CUDA_TRY(workspace.Allocate(&partial_a, parts));
  CUDA_TRY(workspace.Allocate(&partial_b, parts));
  CUDA_TRY(workspace.Allocate(&adler, 1));
  checksum::adler_parts<<<parts, 256, 0, stream>>>(
      input.typed_data(), U32(size), partial_a, partial_b);
  CUDA_TRY(cudaGetLastError());
  checksum::adler_finish<<<1, 256, 0, stream>>>(
      partial_a, partial_b, parts, U32(size), adler);
  CUDA_TRY(cudaGetLastError());
  encoder::write_wrapper<<<1, 1, 0, stream>>>(
      output->typed_data(), U64(host[0]), adler);
  return cudaGetLastError();
}

cudaError_t Decompress(cudaStream_t stream, std::int64_t max_candidates,
                        std::int64_t max_blocks, ffi::Buffer<ffi::U8> input,
                        ffi::ResultBuffer<ffi::U8> output,
                        ffi::ResultBuffer<ffi::U32> metadata,
                        Workspace& workspace) {
  U32* result = metadata->typed_data();
  if (!IsVector(*output) || output->element_count() > kMaxBytes)
    return SetStatus(stream, result, kInvalidBounds);
  const U32 expected = U32(output->element_count());
  auto fail = [&](U32 status) -> cudaError_t {
    if (expected) CUDA_TRY(cudaMemsetAsync(output->typed_data(), 0, expected, stream));
    return SetStatus(stream, result, status);
  };
  if (!IsVector(input)) return fail(kInvalidBounds);
  const U64 full_size = U64(input.element_count());
  if (full_size > kMaxBytes || max_candidates < 1 ||
      max_candidates > kMaxCandidates || max_blocks < 1 || max_blocks > kMaxBlocks)
    return fail(kInvalidBounds);
  if (full_size < 8) return fail(kTruncatedZlib);
  if (expected <= 65536 && full_size <= 131072 &&
      max_candidates == kMaxCandidates && max_blocks == kMaxBlocks) {
    SmallDecode<<<1, 32, 0, stream>>>(
        input.typed_data(), U32(full_size), output->typed_data(), expected,
        U32(max_blocks), result);
    return cudaGetLastError();
  }
  if (expected) CUDA_TRY(cudaMemsetAsync(output->typed_data(), 0, expected, stream));

  // Only RFC 1950 framing bytes cross to host. Token parsing, index discovery,
  // matching, output reconstruction and checksum computation remain on CUDA.
  std::array<U8, 6> framing{};
  CUDA_TRY(cudaMemcpyAsync(framing.data(), input.typed_data(), 2,
                           cudaMemcpyDeviceToHost, stream));
  CUDA_TRY(cudaMemcpyAsync(framing.data() + 2, input.typed_data() + full_size - 4,
                           4, cudaMemcpyDeviceToHost, stream));
  CUDA_TRY(cudaStreamSynchronize(stream));
  const U32 cmf = framing[0], flg = framing[1];
  if (cmf == 0x1f && flg == 0x8b)
    return SetStatus(stream, result, kUnsupportedGzip);
  if ((cmf & 15) != 8 || (cmf >> 4) > 7 || (cmf * 256 + flg) % 31)
    return SetStatus(stream, result, kInvalidHeader);
  if (flg & 32) return SetStatus(stream, result, kUnsupportedDictionary);
  const U32 wanted_checksum = (U32(framing[2]) << 24) | (U32(framing[3]) << 16) |
                             (U32(framing[4]) << 8) | U32(framing[5]);
  const U8* data = input.typed_data() + 2;
  const U32 length = U32(full_size - 6);
  const U32 window = 1u << ((cmf >> 4) + 8);
  const U32 candidate_capacity = U32(max_candidates);
  const U32 block_capacity = U32(max_blocks);
  U64 *unsorted = nullptr, *starts = nullptr, *ends = nullptr;
  U32 *control = nullptr, *sizes = nullptr, *finals = nullptr, *status = nullptr;
  CUDA_TRY(workspace.Allocate(&unsorted, candidate_capacity));
  CUDA_TRY(workspace.Allocate(&starts, candidate_capacity));
  CUDA_TRY(workspace.Allocate(&control, 4));
  CUDA_TRY(cudaMemsetAsync(control, 0, 4 * sizeof(U32), stream));
  const U32 discovery_grid = std::min<U32>(16384, (length + 127) / 128);
  std::array<U32, 2> host{};
  if (length > (1u << 20)) {
    // Prefix matches are speculative and have a separate capacity from valid
    // candidates. Scratch is bounded to one eighth of the compressed input.
    const U32 prefix_capacity = (length + 63) / 64;
    U64* prefixes = nullptr;
    CUDA_TRY(workspace.Allocate(&prefixes, prefix_capacity));
    decoder::scan_prefixes<<<discovery_grid, 128, 0, stream>>>(
        data, length, unsorted, control, candidate_capacity,
        prefixes, control + 1, prefix_capacity);
    CUDA_TRY(cudaGetLastError());
    decoder::validate_prefixes<<<std::min<U32>(16384, (prefix_capacity + 127) / 128),
                                  128, 0, stream>>>(
        data, length, unsorted, control, candidate_capacity,
        prefixes, control + 1, prefix_capacity);
    CUDA_TRY(cudaGetLastError());
    CUDA_TRY(ReadWords(stream, control, host.data(), host.size()));
    if (host[1] > prefix_capacity) {
      // A dense prefix stream must retain the original discovery semantics.
      // No validators ran; discard scan seeds before repeating discovery.
      CUDA_TRY(cudaMemsetAsync(control, 0, 2 * sizeof(U32), stream));
      decoder::discover<<<discovery_grid, 128, 0, stream>>>(
          data, length, unsorted, control, candidate_capacity);
      CUDA_TRY(cudaGetLastError());
      CUDA_TRY(ReadWords(stream, control, host.data(), 1));
    }
  } else {
    decoder::discover<<<discovery_grid, 128, 0, stream>>>(
        data, length, unsorted, control, candidate_capacity);
    CUDA_TRY(cudaGetLastError());
    CUDA_TRY(ReadWords(stream, control, host.data(), 1));
  }
  const U32 candidates = host[0];
  if (candidates > candidate_capacity)
    return SetStatus(stream, result, kCandidateOverflow);
  if (!candidates) return SetStatus(stream, result, kInitialBlockMissing);

  std::size_t sort_bytes = 0;
  CUDA_TRY(cub::DeviceRadixSort::SortKeys(nullptr, sort_bytes, unsorted, starts,
                                          int(candidates), 0, 64, stream));
  U8* sort_workspace = nullptr;
  CUDA_TRY(workspace.Allocate(&sort_workspace, sort_bytes));
  CUDA_TRY(cub::DeviceRadixSort::SortKeys(sort_workspace, sort_bytes, unsorted,
                                          starts, int(candidates), 0, 64, stream));
  CUDA_TRY(workspace.Allocate(&ends, candidates));
  CUDA_TRY(workspace.Allocate(&sizes, candidates));
  CUDA_TRY(workspace.Allocate(&finals, candidates));
  CUDA_TRY(workspace.Allocate(&status, candidates));
  decoder::describe_candidates<<<candidates, 1, 0, stream>>>(
      data, length, starts, candidates, expected, ends, sizes, finals, status);
  CUDA_TRY(cudaGetLastError());
  U64 *block_starts = nullptr, *block_ends = nullptr;
  U32 *prefix = nullptr, *block_sizes = nullptr;
  CUDA_TRY(workspace.Allocate(&block_starts, block_capacity));
  CUDA_TRY(workspace.Allocate(&block_ends, block_capacity));
  CUDA_TRY(workspace.Allocate(&prefix, block_capacity));
  CUDA_TRY(workspace.Allocate(&block_sizes, block_capacity));

  // Dummy summary columns are untouched unless the exact chain requests tiles.
  decoder::select_chain<<<1, 1, 0, stream>>>(
      data, length, starts, ends, sizes, finals, status, candidates, expected,
      block_starts, block_ends, prefix, block_sizes, control, block_capacity,
      control + 1, 0, ends, sizes, ends, sizes, status, status);
  CUDA_TRY(cudaGetLastError());
  CUDA_TRY(ReadWords(stream, control, host.data(), host.size()));
  if (host[1] == 13) {
    const U32 tiles = (length + kFixedTileBytes - 1) / kFixedTileBytes;
    const std::size_t entries = std::size_t(tiles) * 32;
    U64 *summary_ends = nullptr, *first_ends = nullptr;
    U32 *summary_sizes = nullptr, *first_sizes = nullptr;
    U32 *summary_flags = nullptr, *summary_status = nullptr;
    CUDA_TRY(workspace.Allocate(&summary_ends, entries));
    CUDA_TRY(workspace.Allocate(&summary_sizes, entries));
    CUDA_TRY(workspace.Allocate(&first_ends, entries));
    CUDA_TRY(workspace.Allocate(&first_sizes, entries));
    CUDA_TRY(workspace.Allocate(&summary_flags, entries));
    CUDA_TRY(workspace.Allocate(&summary_status, entries));
    decoder::fixed_summaries<<<std::min<U32>(16384, tiles), 32, 0, stream>>>(
        data, length, kFixedTileBytes, summary_ends, summary_sizes, first_ends,
        first_sizes, summary_flags, summary_status);
    CUDA_TRY(cudaGetLastError());
    decoder::select_chain<<<1, 1, 0, stream>>>(
        data, length, starts, ends, sizes, finals, status, candidates, expected,
        block_starts, block_ends, prefix, block_sizes, control, block_capacity,
        control + 1, kFixedTileBytes, summary_ends, summary_sizes, first_ends,
        first_sizes, summary_flags, summary_status);
    CUDA_TRY(cudaGetLastError());
    CUDA_TRY(ReadWords(stream, control, host.data(), host.size()));
  }
  if (host[1]) return SetStatus(stream, result, host[1]);
  const U32 blocks = host[0];
  if (!blocks) return SetStatus(stream, result, kInitialBlockMissing);
  if (blocks > block_capacity) return SetStatus(stream, result, 9);
  U32 *roots = nullptr, *emission = nullptr;
  checksum::DecodeState* decode_state = nullptr;
  CUDA_TRY(workspace.Allocate(&roots, expected));
  CUDA_TRY(workspace.Allocate(&emission, std::size_t(blocks) * 2));
  CUDA_TRY(workspace.Allocate(&decode_state, 1));
  U32* alternate = roots;
  const U32 rounds = RefinementRounds(blocks);
  if (rounds) CUDA_TRY(workspace.Allocate(&alternate, expected));
  decoder::emit_blocks<<<blocks, 1, 0, stream>>>(
      data, length, block_starts, block_ends, prefix, block_sizes, blocks,
      expected, window, roots, emission);
  CUDA_TRY(cudaGetLastError());
  decoder::emit_blocks_warp<<<blocks, 32, 0, stream>>>(
      data, length, block_starts, block_ends, prefix, block_sizes, blocks,
      expected, window, roots, emission);
  CUDA_TRY(cudaGetLastError());
  decoder::emit_stored<<<blocks, 256, 0, stream>>>(
      data, length, block_starts, block_ends, prefix, block_sizes, blocks,
      expected, window, roots, emission);
  CUDA_TRY(cudaGetLastError());
  EmissionStatus<<<1, 256, 0, stream>>>(emission, blocks, decode_state);
  CUDA_TRY(cudaGetLastError());
  const U32 refine_grid = std::max<U32>(
      1, std::min<U32>(16384, (expected + 255) / 256));
  for (U32 round = 0; round < rounds; ++round) {
    checksum::refine_roots<<<refine_grid, 256, 0, stream>>>(
        roots, alternate, expected, decode_state);
    CUDA_TRY(cudaGetLastError());
    RefinementStatus<<<1, 1, 0, stream>>>(decode_state, round + 1 == rounds);
    CUDA_TRY(cudaGetLastError());
  }
  const U32 parts = std::max<U32>(1, (expected + 4095) / 4096);
  U64 *partial_a = nullptr, *partial_b = nullptr;
  CUDA_TRY(workspace.Allocate(&partial_a, parts));
  CUDA_TRY(workspace.Allocate(&partial_b, parts));
  checksum::write_adler_parts<<<parts, 256, 0, stream>>>(
      roots, alternate, output->typed_data(), expected, partial_a, partial_b,
      decode_state);
  CUDA_TRY(cudaGetLastError());
  checksum::adler_finish<<<1, 256, 0, stream>>>(
      partial_a, partial_b, parts, expected, control + 2, decode_state);
  CUDA_TRY(cudaGetLastError());
  VerifyDecompression<<<1, 1, 0, stream>>>(
      decode_state, control + 2, wanted_checksum, result);
  return cudaGetLastError();
}

ffi::Error RuntimeError(cudaError_t error) {
  if (error == cudaSuccess) return ffi::Error::Success();
  return ffi::Error::Internal(std::string("native CUDA codec: ") +
                              cudaGetErrorString(error));
}

template <typename Function>
ffi::Error Execute(cudaStream_t stream, ffi::ResultBuffer<ffi::U32> metadata,
                    Function&& function, std::size_t files = 0) {
  try {
    // A malformed ABI result cannot safely receive the normal metadata status.
    const auto dims = metadata->dimensions();
    if (files ? (dims.size() != 2 || dims[0] != files || dims[1] != 2) :
                (!IsVector(*metadata) || metadata->element_count() != 2))
      return ffi::Error::InvalidArgument(files ?
          "batch codec metadata must have shape (files, 2)" :
          "codec metadata must have shape (2,)");
    Workspace workspace(stream);
    cudaError_t error = workspace.error();
    if (error == cudaSuccess) error = function(workspace);
    cudaError_t release = workspace.Release();
    return RuntimeError(error == cudaSuccess ? release : error);
  } catch (const std::exception&) {
    return ffi::Error::Internal("native CUDA codec handler exception");
  } catch (...) {
    return ffi::Error::Internal("native CUDA codec handler exception");
  }
}

ffi::Error CompressImpl(cudaStream_t stream, std::int64_t chunk_bytes,
                         ffi::Buffer<ffi::U8> input,
                         ffi::ResultBuffer<ffi::U8> output,
                         ffi::ResultBuffer<ffi::U32> metadata) {
  return Execute(stream, metadata, [&](Workspace& workspace) {
    return Compress(stream, chunk_bytes, input, output, metadata, workspace);
  });
}

ffi::Error DecompressImpl(cudaStream_t stream, std::int64_t max_candidates,
                           std::int64_t max_blocks, ffi::Buffer<ffi::U8> input,
                           ffi::ResultBuffer<ffi::U8> output,
                           ffi::ResultBuffer<ffi::U32> metadata) {
  return Execute(stream, metadata, [&](Workspace& workspace) {
    return Decompress(stream, max_candidates, max_blocks, input, output,
                      metadata, workspace);
  });
}

using Sizes = ffi::Span<const std::int64_t>;

ffi::Error CompressBatchImpl(cudaStream_t stream, std::int64_t chunk_bytes,
                             Sizes input_sizes, ffi::Buffer<ffi::U8> input,
                             ffi::ResultBuffer<ffi::U8> output,
                             ffi::ResultBuffer<ffi::U32> metadata) {
  const std::size_t count = input_sizes.size();
  if (!count || count > kMaxBlocks)
    return ffi::Error::InvalidArgument("batch file count must be in [1, 262144]");
  return Execute(stream, metadata, [&](Workspace& workspace) -> cudaError_t {
    if (!IsVector(input) || !IsVector(*output) ||
        input.element_count() > kMaxBytes || output->element_count() > kMaxBytes ||
        chunk_bytes < 256 || chunk_bytes > 65535) return cudaErrorInvalidValue;
    const U32 chunk = U32(chunk_bytes);
    U64 input_extent = 0, output_extent = 0, total_chunks = 0;
    std::vector<BatchEncodeFile> files;
    std::vector<BatchEncodeChunk> chunks;
    files.reserve(count);
    for (std::size_t i = 0; i < count; ++i) {
      const auto size = input_sizes[i];
      if (size < 0 || size > kMaxBytes) return cudaErrorInvalidValue;
      const U64 n = std::max<U64>(1, (U64(size) + chunk - 1) / chunk);
      const U64 capacity = U64(size) + n * 5 + 6;
      if (2 * n - 1 > kMaxBlocks || input_extent + size > kMaxBytes ||
          output_extent + capacity > kMaxBytes || total_chunks + n > kMaxBlocks)
        return cudaErrorInvalidValue;
      files.push_back({U32(input_extent), U32(size), U32(output_extent),
                       U32(capacity), U32(total_chunks), U32(n)});
      for (U32 j = 0; j < n; ++j) chunks.push_back({U32(i), j});
      input_extent += size;
      output_extent += capacity;
      total_chunks += n;
    }
    if (input_extent != input.element_count() ||
        output_extent != output->element_count()) return cudaErrorInvalidValue;
    BatchEncodeFile* device_files = nullptr;
    BatchEncodeChunk* device_chunks = nullptr;
    U8* scratch = nullptr;
    U32 *tokens = nullptr, *sizes = nullptr, *status = nullptr;
    const U32 slot_bytes = (chunk * 9 + 7) / 8 + 16;
    CUDA_TRY(workspace.Allocate(&device_files, count));
    CUDA_TRY(workspace.Allocate(&device_chunks, total_chunks));
    CUDA_TRY(workspace.Allocate(&scratch, total_chunks * slot_bytes));
    CUDA_TRY(workspace.Allocate(&tokens, input_extent));
    CUDA_TRY(workspace.Allocate(&sizes, total_chunks));
    CUDA_TRY(workspace.Allocate(&status, total_chunks));
    CUDA_TRY(cudaMemcpyAsync(device_files, files.data(), count * sizeof(files[0]),
                             cudaMemcpyHostToDevice, stream));
    CUDA_TRY(cudaMemcpyAsync(device_chunks, chunks.data(), chunks.size() * sizeof(chunks[0]),
                             cudaMemcpyHostToDevice, stream));
    // Descriptor storage is owned by this invocation. Complete the two uploads
    // before releasing the host vectors; there are no per-file synchronizations.
    CUDA_TRY(cudaStreamSynchronize(stream));
    CUDA_TRY(cudaMemsetAsync(output->typed_data(), 0, output_extent, stream));
    BatchEncode<<<U32(total_chunks), 256, 0, stream>>>(
        input.typed_data(), chunk, slot_bytes, device_files, device_chunks,
        U32(count), U32(total_chunks), scratch, sizes, status, tokens);
    CUDA_TRY(cudaGetLastError());
    BatchFinish<<<U32(count), 256, 0, stream>>>(
        input.typed_data(), device_files, U32(count), U32(total_chunks), slot_bytes,
        scratch, sizes, status, output->typed_data(), metadata->typed_data());
    return cudaGetLastError();
  }, count);
}

ffi::Error DecompressBatchImpl(cudaStream_t stream, Sizes input_sizes,
                               Sizes output_sizes, ffi::Buffer<ffi::U8> input,
                               ffi::Buffer<ffi::U32> encoded_metadata,
                               ffi::ResultBuffer<ffi::U8> output,
                               ffi::ResultBuffer<ffi::U32> metadata) {
  const std::size_t count = input_sizes.size();
  if (!count || count > kMaxBlocks || output_sizes.size() != count)
    return ffi::Error::InvalidArgument("batch sizes must have matching nonempty lengths");
  const auto dims = encoded_metadata.dimensions();
  const bool has_lengths = encoded_metadata.element_count() != 0;
  if (has_lengths ? (dims.size() != 2 || dims[0] != count || dims[1] != 2) :
                    (dims.size() != 1))
    return ffi::Error::InvalidArgument("encoded metadata must have shape (files, 2) or (0,)");
  return Execute(stream, metadata, [&](Workspace& workspace) -> cudaError_t {
    if (!IsVector(input) || !IsVector(*output) ||
        input.element_count() > kMaxBytes || output->element_count() > kMaxBytes)
      return cudaErrorInvalidValue;
    std::vector<BatchDecodeFile> files;
    files.reserve(count);
    U64 input_extent = 0, output_extent = 0;
    for (std::size_t i = 0; i < count; ++i) {
      const auto size = input_sizes[i], expected = output_sizes[i];
      if (size < 0 || expected < 0 || size > kMaxBytes || expected > kMaxBytes ||
          input_extent + size > kMaxBytes || output_extent + expected > kMaxBytes)
        return cudaErrorInvalidValue;
      files.push_back({U32(input_extent), U32(size), U32(output_extent), U32(expected)});
      input_extent += size;
      output_extent += expected;
    }
    if (input_extent != input.element_count() || output_extent != output->element_count())
      return cudaErrorInvalidValue;
    BatchDecodeFile* device_files = nullptr;
    CUDA_TRY(workspace.Allocate(&device_files, count));
    CUDA_TRY(cudaMemcpyAsync(device_files, files.data(), count * sizeof(files[0]),
                             cudaMemcpyHostToDevice, stream));
    CUDA_TRY(cudaStreamSynchronize(stream));
    BatchDecode<<<U32(count), 32, 0, stream>>>(
        input.typed_data(), output->typed_data(), device_files, U32(count),
        kMaxBlocks, metadata->typed_data(),
        has_lengths ? encoded_metadata.typed_data() : nullptr);
    return cudaGetLastError();
  }, count);
}

#undef CUDA_TRY
}  // namespace

// Configuration is installed once, before this library registers its handlers.
extern "C" int CudaZlibSetWorkspaceRetention(U64 bytes) noexcept {
  try {
    PoolRegistry& registry = Pools();
    std::lock_guard<std::mutex> lock(registry.mutex);
    if (!registry.pools.empty()) return cudaErrorInvalidValue;
    registry.retention = bytes;
    return cudaSuccess;
  } catch (...) {
    return cudaErrorUnknown;
  }
}

extern "C" int CudaZlibWorkspacePoolStats(int device, U64* stats) noexcept {
  if (device < 0 || !stats) return cudaErrorInvalidValue;
  try {
    return InspectPools(device, stats, false);
  } catch (...) {
    return cudaErrorUnknown;
  }
}

extern "C" int CudaZlibTrimWorkspacePool(int device) noexcept {
  if (device < 0) return cudaErrorInvalidValue;
  try {
    U64 stats[4];
    return InspectPools(device, stats, true);
  } catch (...) {
    return cudaErrorUnknown;
  }
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    CudaZlibCompress, CompressImpl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Attr<std::int64_t>("chunk_bytes")
        .Arg<ffi::Buffer<ffi::U8>>()
        .Ret<ffi::Buffer<ffi::U8>>()
        .Ret<ffi::Buffer<ffi::U32>>());

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    CudaZlibDecompress, DecompressImpl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Attr<std::int64_t>("max_candidates")
        .Attr<std::int64_t>("max_blocks")
        .Arg<ffi::Buffer<ffi::U8>>()
        .Ret<ffi::Buffer<ffi::U8>>()
        .Ret<ffi::Buffer<ffi::U32>>());

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    CudaZlibCompressBatch, CompressBatchImpl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Attr<std::int64_t>("chunk_bytes")
        .Attr<Sizes>("input_sizes")
        .Arg<ffi::Buffer<ffi::U8>>()
        .Ret<ffi::Buffer<ffi::U8>>()
        .Ret<ffi::Buffer<ffi::U32>>());

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    CudaZlibDecompressBatch, DecompressBatchImpl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Attr<Sizes>("input_sizes")
        .Attr<Sizes>("output_sizes")
        .Arg<ffi::Buffer<ffi::U8>>()
        .Arg<ffi::Buffer<ffi::U32>>()
        .Ret<ffi::Buffer<ffi::U8>>()
        .Ret<ffi::Buffer<ffi::U32>>());
