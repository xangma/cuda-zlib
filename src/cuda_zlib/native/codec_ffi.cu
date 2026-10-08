// Copyright (c) 2026 xangma
// SPDX-License-Identifier: MIT

// Fresh CUDA codec execution on the stream supplied by XLA. Generated headers
// contain the unchanged CUDA_SOURCE strings, with no host codec implementation.
#include <cuda_runtime.h>
#include <cuda/atomic>
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

cudaError_t GetPool(CUcontext context, cudaMemPool_t* result,
                    U64* context_id, int* device_id) {
  Driver& driver = Driver::Get();
  U64 id;
  int device;
  CUDA_TRY(DriverError(driver.context_id(context, &id)));
  CUDA_TRY(DriverError(driver.context_device(&device)));
  *context_id = id;
  *device_id = device;
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
    if (error_ == cudaSuccess)
      error_ = GetPool(context, &pool_, &context_id_, &device_);
  }
  Workspace(const Workspace&) = delete;
  Workspace& operator=(const Workspace&) = delete;
  ~Workspace() noexcept { Release(); }
  cudaError_t error() const { return error_; }
  U64 context_id() const { return context_id_; }
  int device() const { return device_; }

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
  U64 context_id_ = 0;
  int device_ = -1;
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

// Raw discovery counters never alias the finalized count or accepted chain.
// Framing and discovery results are immutable after their finalizers.
constexpr U32 kFramingOffset = 4;
constexpr U32 kCandidateCountOffset = 8;
constexpr U32 kBlockCountOffset = 9;
constexpr U32 kChainStatusOffset = 10;
constexpr U32 kDiscoveryStatusOffset = 11;
constexpr U32 kDiscoveryWords = 12;

__global__ void DecompressionFraming(const U8* input, U32 size, U32* framing,
                                    const U32* fast_metadata = nullptr) {
  if (blockIdx.x || threadIdx.x) return;
  U32 error = 0, window = 0, wanted = 0;
  if (size < 8) error = kTruncatedZlib;
  else {
    const U32 cmf = input[0], flg = input[1];
    if (cmf == 0x1f && flg == 0x8b) error = kUnsupportedGzip;
    else if ((cmf & 15) != 8 || (cmf >> 4) > 7 ||
             (cmf * 256 + flg) % 31) error = kInvalidHeader;
    else if (flg & 32) error = kUnsupportedDictionary;
    else {
      window = 1u << ((cmf >> 4) + 8);
      wanted = (U32(input[size - 4]) << 24) | (U32(input[size - 3]) << 16) |
               (U32(input[size - 2]) << 8) | U32(input[size - 1]);
    }
  }
  framing[0] = error;
  framing[1] = window;
  framing[2] = wanted;
  // Fast status is finalized before this launch and immutable thereafter.
  framing[3] = !error && !(fast_metadata && !fast_metadata[0]);
}

__global__ void ResetDenseDiscovery(U32* control, U32 prefix_capacity) {
  if (blockIdx.x || threadIdx.x) return;
  U32* framing = control + kFramingOffset;
  const U32 repeat = !framing[0] && control[1] > prefix_capacity;
  framing[3] = repeat;
  if (repeat) {
    // The completed validator skipped an overflowing prefix list. Discard all
    // speculative stored seeds before the gated full discovery replaces them.
    control[0] = 0;
    control[1] = 0;
  }
}

__global__ void FinalizeCandidates(U64* starts, U32 capacity, U32* control) {
  // Run after the final dense retry: initial filling would leave stale stored
  // seeds when that retry resets the raw count. All CUB input keys are written
  // even on error, and no descriptor ever receives the sentinel suffix.
  const U32 count = control[0], frame_error = control[kFramingOffset];
  const U32 error = frame_error ? frame_error :
      count > capacity ? U32(kCandidateOverflow) :
      !count ? U32(kInitialBlockMissing) : 0;
  if (!blockIdx.x && !threadIdx.x) {
    control[kCandidateCountOffset] = error ? 0 : count;
    control[kDiscoveryStatusOffset] = error;
  }
  const U64 stride = U64(blockDim.x) * gridDim.x;
  for (U64 i = U64(blockIdx.x) * blockDim.x + threadIdx.x;
       i < capacity; i += stride)
    if (error || i >= count) starts[i] = 0xffffffffULL;
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

__global__ void ChainStatus(const U32* block_count, const U32* chain_status,
                            U32 capacity, checksum::DecodeState* state) {
  if (blockIdx.x || threadIdx.x) return;
  U32 error = *chain_status;
  if (!error && !*block_count) error = kInitialBlockMissing;
  if (!error && *block_count > capacity) error = 9;
  state->status = error;
  state->active = 0;
  state->selector = 0;
  state->pending = 0;
  state->refine_status = 0;
}

__global__ void EmissionStatus(const U32* input, const U32* block_count,
                               checksum::DecodeState* state) {
  // Chain failure leaves accepted arrays, roots and emission planes unwritten.
  // status is immutable until the final lane below, so the barrier gate is
  // uniform and cannot erase a framing/discovery/chain error.
  if (state->status) return;
  __shared__ U32 errors[256], pending[256];
  const U32 blocks = *block_count;
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
                                    const U32* actual, const U32* wanted,
                                    U32* metadata,
                                    const U32* fast_metadata = nullptr) {
  if (blockIdx.x || threadIdx.x) return;
  if (fast_metadata && !fast_metadata[0]) {
    metadata[0] = 0;
    metadata[1] = 0;
    return;
  }
  U32 error = state->status;
  if (!error && state->active) error = kReferenceDepthExceeded;
  // Gather/checksum are skipped on prior failure, so actual may be unwritten.
  if (!error && *actual != *wanted) error = kAdlerMismatch;
  metadata[0] = error;
  metadata[1] = 0;
}

// Discard a failed fast parse's decoded prefix before the ordinary pipeline.
// Its finalized status is immutable for every thread in this launch.
__global__ void ClearFusedFailure(const U32* fast_metadata, U8* output, U32 size) {
  if (!fast_metadata || !fast_metadata[0]) return;
  const U64 stride = U64(blockDim.x) * gridDim.x;
  for (U64 i = U64(blockIdx.x) * blockDim.x + threadIdx.x;
       i < size; i += stride)
    output[i] = 0;
}

// The parser finalized status before the parallel checksum launches. Failed
// output remains fully initialized, but its checksum must not replace that error.
__global__ void VerifyFusedChecksum(const U8* input, U32 input_size,
                                   const U32* actual, U32* metadata) {
  if (blockIdx.x || threadIdx.x) return;
  if (!metadata[0]) {
    const U32 wanted = (U32(input[input_size - 4]) << 24) |
                       (U32(input[input_size - 3]) << 16) |
                       (U32(input[input_size - 2]) << 8) |
                       U32(input[input_size - 1]);
    if (*actual != wanted) metadata[0] = kAdlerMismatch;
  }
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

// BEGIN SMALL SHARED CONFIGURATION
// This library owns SmallSharedDecodeSpecialized's attribute configuration. Workspace has
// entered the XLA stream context and reuses GetPool's unique context ID/device.
// Cache entries are per loaded codec module and per context; a recreated context
// receives a new ID, even when CUDA reuses its address. Only success is cached.
struct SmallSharedConfiguration {
  U64 context_id;
  int device;
  U32 bytes;
};
struct SmallSharedRegistry {
  std::mutex mutex;
  std::vector<SmallSharedConfiguration> entries;
};
SmallSharedRegistry& SmallSharedConfigurations() {
  // Process-lifetime registry, matching the native module/pool lifetime. Avoid
  // CUDA calls at teardown; records own no device/context resources.
  static SmallSharedRegistry* registry = new SmallSharedRegistry;
  return *registry;
}
cudaError_t SmallSharedLimit(const Workspace& workspace, U32* result) {
  SmallSharedRegistry& registry = SmallSharedConfigurations();
  std::lock_guard<std::mutex> lock(registry.mutex);
  for (const SmallSharedConfiguration& entry : registry.entries) {
    if (entry.context_id == workspace.context_id() &&
        entry.device == workspace.device()) {
      *result = entry.bytes;
      return cudaSuccess;
    }
  }
  cudaFuncAttributes attributes{};
  int optin = 0;
  CUDA_TRY(cudaFuncGetAttributes(&attributes, SmallSharedDecodeSpecialized));
  CUDA_TRY(cudaDeviceGetAttribute(
      &optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, workspace.device()));
  U32 bytes = 0;
  if (optin > 0 && attributes.sharedSizeBytes < std::size_t(optin))
    bytes = U32(std::min<std::size_t>(
        65536, std::size_t(optin) - attributes.sharedSizeBytes));
  if (int(bytes) > attributes.maxDynamicSharedSizeBytes)
    CUDA_TRY(cudaFuncSetAttribute(SmallSharedDecodeSpecialized,
        cudaFuncAttributeMaxDynamicSharedMemorySize, int(bytes)));
  registry.entries.push_back({workspace.context_id(), workspace.device(), bytes});
  *result = bytes;
  return cudaSuccess;
}
// END SMALL SHARED CONFIGURATION

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
    U32 shared_limit = 0;
    CUDA_TRY(SmallSharedLimit(workspace, &shared_limit));
    if (expected <= shared_limit) {
      SmallSharedDecodeSpecialized<<<1, 32, expected, stream>>>(
          input.typed_data(), U32(full_size), output->typed_data(), expected,
          U32(max_blocks), result);
    } else {
      SmallDecode<<<1, 32, 0, stream>>>(
          input.typed_data(), U32(full_size), output->typed_data(), expected,
          U32(max_blocks), result);
    }
    return cudaGetLastError();
  }
  if (expected) CUDA_TRY(cudaMemsetAsync(output->typed_data(), 0, expected, stream));

  // A bounded fast parser handles valid medium streams. Any
  // failure falls through to the ordinary queued pipeline and its precedence.
  U32* fast_metadata = nullptr;
  if (expected > 65536 && expected <= 1048576 && full_size <= 29133 &&
      max_candidates == kMaxCandidates && max_blocks == kMaxBlocks) {
    const U32 fast_parts = (expected + 4095) / 4096;
    U64 *fast_partial_a = nullptr, *fast_partial_b = nullptr;
    U32* fast_actual = nullptr;
    CUDA_TRY(workspace.Allocate(&fast_metadata, 2));
    CUDA_TRY(workspace.Allocate(&fast_partial_a, fast_parts));
    CUDA_TRY(workspace.Allocate(&fast_partial_b, fast_parts));
    CUDA_TRY(workspace.Allocate(&fast_actual, 1));
    SmallDecode<<<1, 32, 0, stream>>>(
        input.typed_data(), U32(full_size), output->typed_data(), expected,
        U32(max_blocks), fast_metadata, true);
    CUDA_TRY(cudaGetLastError());
    checksum::adler_parts<<<fast_parts, 256, 0, stream>>>(
        output->typed_data(), expected, fast_partial_a, fast_partial_b);
    CUDA_TRY(cudaGetLastError());
    checksum::adler_finish<<<1, 256, 0, stream>>>(
        fast_partial_a, fast_partial_b, fast_parts, expected, fast_actual);
    CUDA_TRY(cudaGetLastError());
    VerifyFusedChecksum<<<1, 1, 0, stream>>>(
        input.typed_data(), U32(full_size), fast_actual, fast_metadata);
    CUDA_TRY(cudaGetLastError());
  }
  // Pointer presence depends only on the static route, never a host status read.
  if (fast_metadata) {
    ClearFusedFailure<<<std::min<U32>(16384, (expected + 255) / 256),
                        256, 0, stream>>>(fast_metadata, output->typed_data(), expected);
    CUDA_TRY(cudaGetLastError());
  }

  const U8* data = input.typed_data() + 2;
  const U32 length = U32(full_size - 6);
  const bool pipeline_stream = length && U64(expected) >= U64(4) * U64(length);
  // Discovery has at most eight bit starts and one stored-end seed per body
  // byte, including duplicates. The split prefix path has the same bound;
  // its start-zero seed replaces bit zero. The extra slot is conservative.
  const U32 candidate_capacity = U32(std::min<U64>(U64(max_candidates),
                                                   U64(length) * 9 + 1));
  const U32 block_capacity = U32(max_blocks);
  U64 *unsorted = nullptr, *starts = nullptr, *ends = nullptr;
  U32 *control = nullptr, *sizes = nullptr, *finals = nullptr, *status = nullptr;
  CUDA_TRY(workspace.Allocate(&unsorted, candidate_capacity));
  CUDA_TRY(workspace.Allocate(&starts, candidate_capacity));
  CUDA_TRY(workspace.Allocate(&control, kDiscoveryWords));
  CUDA_TRY(cudaMemsetAsync(control, 0, kDiscoveryWords * sizeof(U32), stream));
  U32* framing = control + kFramingOffset;
  U32* candidates = control + kCandidateCountOffset;
  U32* blocks = control + kBlockCountOffset;
  U32* chain_status = control + kChainStatusOffset;
  U32* discovery_status = control + kDiscoveryStatusOffset;
  DecompressionFraming<<<1, 1, 0, stream>>>(
      input.typed_data(), U32(full_size), framing, fast_metadata);
  CUDA_TRY(cudaGetLastError());
  const U32 discovery_grid = std::min<U32>(16384, (length + 127) / 128);
  if (length > (1u << 20)) {
    // Prefix matches are speculative and have a separate capacity from valid
    // candidates. Scratch is bounded to one eighth of the compressed input.
    const U32 prefix_capacity = (length + 63) / 64;
    U64* prefixes = nullptr;
    CUDA_TRY(workspace.Allocate(&prefixes, prefix_capacity));
    decoder::scan_prefixes<<<discovery_grid, 128, 0, stream>>>(
        data, length, unsorted, control, candidate_capacity,
        prefixes, control + 1, prefix_capacity, framing);
    CUDA_TRY(cudaGetLastError());
    decoder::validate_prefixes<<<std::min<U32>(16384, (prefix_capacity + 127) / 128),
                                  128, 0, stream>>>(
        data, length, unsorted, control, candidate_capacity,
        prefixes, control + 1, prefix_capacity, framing);
    CUDA_TRY(cudaGetLastError());
    ResetDenseDiscovery<<<1, 1, 0, stream>>>(control, prefix_capacity);
    CUDA_TRY(cudaGetLastError());
    decoder::discover<<<discovery_grid, 128, 0, stream>>>(
        data, length, unsorted, control, candidate_capacity, framing + 3);
    CUDA_TRY(cudaGetLastError());
  } else {
    decoder::discover<<<discovery_grid, 128, 0, stream>>>(
        data, length, unsorted, control, candidate_capacity, framing + 3);
    CUDA_TRY(cudaGetLastError());
  }
  FinalizeCandidates<<<std::min<U32>(512, (candidate_capacity + 255) / 256),
                         256, 0, stream>>>(unsorted, candidate_capacity, control);
  CUDA_TRY(cudaGetLastError());

  // full_size <= 2^28 and the body excludes six framing bytes, so every real
  // bit position is below 2^31. Upper key words are zero, and the low-32-bit
  // sentinel sorts strictly after all real keys, preserving duplicates.
  std::size_t sort_bytes = 0;
  CUDA_TRY(cub::DeviceRadixSort::SortKeys(nullptr, sort_bytes, unsorted, starts,
                                          int(candidate_capacity), 0, 32, stream));
  U8* sort_workspace = nullptr;
  CUDA_TRY(workspace.Allocate(&sort_workspace, sort_bytes));
  CUDA_TRY(cub::DeviceRadixSort::SortKeys(sort_workspace, sort_bytes, unsorted,
                                          starts, int(candidate_capacity), 0, 32, stream));
  CUDA_TRY(workspace.Allocate(&ends, candidate_capacity));
  CUDA_TRY(workspace.Allocate(&sizes, candidate_capacity));
  CUDA_TRY(workspace.Allocate(&finals, candidate_capacity));
  CUDA_TRY(workspace.Allocate(&status, candidate_capacity));
  U32 *candidate_tokens = nullptr, *block_tokens = nullptr;
  if (pipeline_stream) CUDA_TRY(workspace.Allocate(&candidate_tokens, candidate_capacity));
  // Large bodies need enough independent CTAs to balance block parsing.
  const U32 parser_grid = length > (1u << 20) ? 8192u : 512u;
  if (pipeline_stream) {
    decoder::describe_candidates_counted<<<std::min<U32>(parser_grid, candidate_capacity), 1, 0, stream>>>(
        data, length, starts, expected, ends, sizes, finals, status,
        candidates, discovery_status, candidate_tokens);
  } else {
    decoder::describe_candidates<<<std::min<U32>(parser_grid, candidate_capacity), 1, 0, stream>>>(
        data, length, starts, expected, ends, sizes, finals, status,
        candidates, discovery_status);
  }
  CUDA_TRY(cudaGetLastError());
  U64 *block_starts = nullptr, *block_ends = nullptr;
  U32 *prefix = nullptr, *block_sizes = nullptr;
  CUDA_TRY(workspace.Allocate(&block_starts, block_capacity));
  CUDA_TRY(workspace.Allocate(&block_ends, block_capacity));
  CUDA_TRY(workspace.Allocate(&prefix, block_capacity));
  CUDA_TRY(workspace.Allocate(&block_sizes, block_capacity));
  if (pipeline_stream) CUDA_TRY(workspace.Allocate(&block_tokens, std::size_t(block_capacity) + 1));

  // Dummy summary columns are untouched unless the exact chain requests tiles.
  if (pipeline_stream) {
    decoder::select_chain_counted<<<1, 1, 0, stream>>>(
        data, length, starts, ends, sizes, finals, status, 0, expected,
        block_starts, block_ends, prefix, block_sizes, blocks, block_capacity,
        chain_status, 0, ends, sizes, ends, sizes, status, status,
        candidate_tokens, block_tokens, candidates, discovery_status);
  } else {
    decoder::select_chain<<<1, 1, 0, stream>>>(
        data, length, starts, ends, sizes, finals, status, 0, expected,
        block_starts, block_ends, prefix, block_sizes, blocks, block_capacity,
        chain_status, 0, ends, sizes, ends, sizes, status, status,
        candidates, discovery_status);
  }
  CUDA_TRY(cudaGetLastError());
  // Eager storage is bounded to 1024 bytes per 2048-byte body tile (<=128MiB).
  // Both launches skip unless the exact first chain requests fixed summaries.
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
      first_sizes, summary_flags, summary_status, chain_status);
  CUDA_TRY(cudaGetLastError());
  if (pipeline_stream) {
    decoder::select_chain_counted<<<1, 1, 0, stream>>>(
        data, length, starts, ends, sizes, finals, status, 0, expected,
        block_starts, block_ends, prefix, block_sizes, blocks, block_capacity,
        chain_status, kFixedTileBytes, summary_ends, summary_sizes, first_ends,
        first_sizes, summary_flags, summary_status, candidate_tokens, block_tokens, candidates, discovery_status,
        chain_status);
  } else {
    decoder::select_chain<<<1, 1, 0, stream>>>(
        data, length, starts, ends, sizes, finals, status, 0, expected,
        block_starts, block_ends, prefix, block_sizes, blocks, block_capacity,
        chain_status, kFixedTileBytes, summary_ends, summary_sizes, first_ends,
        first_sizes, summary_flags, summary_status, candidates, discovery_status,
        chain_status);
  }
  CUDA_TRY(cudaGetLastError());
  U32 *roots = nullptr, *emission = nullptr;
  checksum::DecodeState* decode_state = nullptr;
  CUDA_TRY(workspace.Allocate(&roots, expected));
  CUDA_TRY(workspace.Allocate(&emission, std::size_t(block_capacity) * 2));
  CUDA_TRY(workspace.Allocate(&decode_state, 1));
  ChainStatus<<<1, 1, 0, stream>>>(blocks, chain_status, block_capacity, decode_state);
  CUDA_TRY(cudaGetLastError());
  U32* alternate = roots;
  const U32 rounds = RefinementRounds(block_capacity);
  if (rounds) CUDA_TRY(workspace.Allocate(&alternate, expected));
  const U32 emission_grid = std::min<U32>(parser_grid, block_capacity);
  // Accepted-chain density selects original serial or queued emission on device.
  if (pipeline_stream) {
    decoder::emit_blocks<<<emission_grid, 1, 0, stream>>>(
        data, length, block_starts, block_ends, prefix, block_sizes,
        expected, roots, emission, framing + 1, blocks, block_tokens + block_capacity);
    CUDA_TRY(cudaGetLastError());
    decoder::emit_blocks_pipeline<<<emission_grid, 64, 0, stream>>>(
        data, length, block_starts, block_ends, prefix, block_sizes,
        expected, roots, emission, framing + 1, blocks, &decode_state->status,
        block_tokens, block_tokens + block_capacity);
  } else {
    decoder::emit_blocks<<<emission_grid, 1, 0, stream>>>(
        data, length, block_starts, block_ends, prefix, block_sizes,
        expected, roots, emission, framing + 1, blocks, &decode_state->status);
  }
  CUDA_TRY(cudaGetLastError());
  decoder::emit_blocks_warp<<<emission_grid, 32, 0, stream>>>(
      data, length, block_starts, block_ends, prefix, block_sizes,
      expected, roots, emission, framing + 1, blocks, &decode_state->status);
  CUDA_TRY(cudaGetLastError());
  decoder::emit_stored<<<emission_grid, 256, 0, stream>>>(
      data, length, block_starts, block_ends, prefix, block_sizes,
      expected, roots, emission, framing + 1, blocks, &decode_state->status);
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
      decode_state, control + 2, framing + 2, result, fast_metadata);
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
