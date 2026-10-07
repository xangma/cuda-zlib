// Copyright (c) 2026 xangma
// SPDX-License-Identifier: MIT

// Include inside the native anonymous namespace, after the byte aliases and
// Status enum. Descriptors are constructed and bounds-checked on the host.
struct BatchEncodeFile {
  U32 input_offset, input_size, output_offset, capacity, first_chunk, chunks;
};
struct BatchEncodeChunk {
  U32 file, local_chunk;
};

__global__ void BatchEncode(
    const U8* input, U32 chunk_bytes, U32 slot_bytes,
    const BatchEncodeFile* files, const BatchEncodeChunk* descriptors,
    U32 file_count, U32 total_chunks, U8* scratch, U32* sizes, U32* status,
    U32* tokens) {
  U32 chunk = blockIdx.x;
  if (chunk >= total_chunks) return;
  BatchEncodeChunk descriptor = descriptors[chunk];
  if (descriptor.file >= file_count) {
    if (!threadIdx.x) status[chunk] = kInvalidBounds;
    return;
  }
  BatchEncodeFile file = files[descriptor.file];
  if (descriptor.local_chunk >= file.chunks ||
      U64(file.first_chunk) + descriptor.local_chunk != chunk) {
    if (!threadIdx.x) status[chunk] = kInvalidBounds;
    return;
  }
  encoder::encode_chunk(
      input + file.input_offset, file.input_size, chunk_bytes, file.chunks,
      descriptor.local_chunk, slot_bytes,
      scratch + U64(file.first_chunk) * slot_bytes,
      sizes + file.first_chunk, status + file.first_chunk,
      tokens + file.input_offset);
}

__device__ __forceinline__ void FinishEncodedFile(
    const U8* input, BatchEncodeFile file, U32 total_chunks, U32 slot_bytes,
    const U8* scratch, const U32* sizes, const U32* status, U8* output,
    U32* metadata) {
  __shared__ U64 a[256], b[256];
  __shared__ U32 extent, error;
  if (!threadIdx.x) {
    extent = 0;
    error = 0;
    metadata[0] = 0;
    metadata[1] = 0;
    if (!file.chunks || U64(file.first_chunk) + file.chunks > total_chunks) {
      error = kInvalidBounds;
    } else {
      U64 total = 6;
      for (U32 i = 0; i < file.chunks; ++i) {
        U32 chunk = file.first_chunk + i;
        if (status[chunk] || sizes[chunk] > slot_bytes ||
            total + sizes[chunk] > file.capacity) {
          error = status[chunk] == kInvalidBounds ? kInvalidBounds
                                                 : kCompressionOverflow;
          break;
        }
        total += sizes[chunk];
      }
      if (!error) extent = U32(total);
    }
    if (error) metadata[1] = error;
  }
  __syncthreads();
  if (error) return;

  U8* destination = output + file.output_offset;
  U64 offset = 2;
  for (U32 i = 0; i < file.chunks; ++i) {
    U32 chunk = file.first_chunk + i;
    U32 size = sizes[chunk];
    const U8* source = scratch + U64(chunk) * slot_bytes;
    for (U32 j = threadIdx.x; j < size; j += blockDim.x)
      destination[offset + j] = source[j];
    offset += size;
  }

  const U8* source = input + file.input_offset;
  U64 sa = 0, sb = 0;
  for (U32 i = threadIdx.x; i < file.input_size; i += blockDim.x) {
    U32 value = source[i];
    sa += value;
    sb += U64(file.input_size - i) * value;
  }
  a[threadIdx.x] = sa;
  b[threadIdx.x] = sb;
  __syncthreads();
  for (U32 stride = 128; stride; stride >>= 1) {
    if (threadIdx.x < stride) {
      a[threadIdx.x] += a[threadIdx.x + stride];
      b[threadIdx.x] += b[threadIdx.x + stride];
    }
    __syncthreads();
  }
  if (!threadIdx.x) {
    // The native file-size bound is the same 2^28-byte bound as adler_finish.
    U32 low = U32((a[0] + 1u) % 65521u);
    U32 high = U32((b[0] + file.input_size) % 65521u);
    U32 checksum = (high << 16) | low;
    destination[0] = 0x78;
    destination[1] = 0x01;
    destination[extent - 4u] = U8(checksum >> 24);
    destination[extent - 3u] = U8(checksum >> 16);
    destination[extent - 2u] = U8(checksum >> 8);
    destination[extent - 1u] = U8(checksum);
    metadata[0] = extent;
  }
}

__global__ void BatchFinish(
    const U8* input, const BatchEncodeFile* files, U32 file_count,
    U32 total_chunks, U32 slot_bytes, const U8* scratch, const U32* sizes,
    const U32* status, U8* output, U32* metadata) {
  U32 index = blockIdx.x;
  if (index >= file_count) return;
  FinishEncodedFile(input, files[index], total_chunks, slot_bytes, scratch,
                    sizes, status, output, metadata + 2u * index);
}

__global__ void SmallFinish(
    const U8* input, U32 input_size, U32 chunks, U32 slot_bytes,
    const U8* scratch, const U32* sizes, const U32* status, U8* output,
    U32 capacity, U32* metadata) {
  if (blockIdx.x) return;
  BatchEncodeFile file = {0, input_size, 0, capacity, 0, chunks};
  FinishEncodedFile(input, file, chunks, slot_bytes, scratch, sizes, status,
                    output, metadata);
}
