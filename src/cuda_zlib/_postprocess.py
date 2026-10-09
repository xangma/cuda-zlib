# Copyright (c) 2026 xangma
# SPDX-License-Identifier: MIT

"""CUDA checksum and reference-resolution kernels."""

CUDA_SOURCE = r'''
// Fields used to gate later kernels are immutable during each launch. Pending
// and refine_status are scratch reductions, consumed by a separate finalizer.
struct DecodeState {
  unsigned int status;
  unsigned int active;
  unsigned int selector;
  unsigned int pending;
  unsigned int refine_status;
};

extern "C" __global__ void refine_roots(
    unsigned int* first, unsigned int* second, unsigned int size,
    DecodeState* state) {
  // active is changed only between kernels, so all threads reach the vote
  // together. Never gate on the error reduction written during this launch.
  if (!state->active) return;
  const unsigned int* input = state->selector ? second : first;
  unsigned int* output = state->selector ? first : second;
  const unsigned long long stride = (unsigned long long)blockDim.x * gridDim.x;
  int unresolved = 0;
  for (unsigned long long index = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
       index < size; index += stride) {
    const unsigned int i = (unsigned int)index;
    unsigned int value = input[i];
    bool valid = true;
    for (int step = 0; step < 32 && !(value & 0x80000000u); ++step) {
      if (value >= i) { atomicExch(&state->refine_status, 6u); valid = false; break; }
      unsigned int next = input[value];
      if (!(next & 0x80000000u) && next >= value) {
        atomicExch(&state->refine_status, 6u); valid = false; break;
      }
      value = next;
    }
    if (valid) {
      output[i] = value;
      unresolved |= !(value & 0x80000000u);
    }
  }
  int block_pending = __syncthreads_or(unresolved);
  if (threadIdx.x == 0 && block_pending) atomicExch(&state->pending, 1u);
}
// Resolved roots hold tagged literals. Gather the returned bytes while making
// checksum partials, avoiding a separate gather launch and output read.
extern "C" __global__ void write_adler_parts(
    const unsigned int* first, const unsigned int* second,
    unsigned char* output, unsigned int size,
    unsigned long long* partial_a, unsigned long long* partial_b,
    const DecodeState* state) {
  // An emission/refinement error can leave roots unwritten. Keep the native
  // output's initial zeros and do not read either root buffer in that case.
  if (state->status || state->active) return;
  const unsigned int* roots = state->selector ? second : first;
  __shared__ unsigned long long a[256], b[256];
  unsigned int i = blockIdx.x * 4096u + threadIdx.x;
  unsigned long long sa = 0, sb = 0;
  for (unsigned int j = 0; j < 16u; ++j, i += 256u) {
    if (i < size) {
      unsigned int value = (unsigned char)roots[i];
      output[i] = (unsigned char)value;
      sa += value;
      sb += (unsigned long long)(size - i) * value;
    }
  }
  a[threadIdx.x] = sa; b[threadIdx.x] = sb;
  __syncthreads();
  for (int stride = 128; stride; stride >>= 1) {
    if (threadIdx.x < stride) {
      a[threadIdx.x] += a[threadIdx.x + stride];
      b[threadIdx.x] += b[threadIdx.x + stride];
    }
    __syncthreads();
  }
  if (!threadIdx.x) {
    partial_a[blockIdx.x] = a[0]; partial_b[blockIdx.x] = b[0];
  }
}
extern "C" __global__ void adler_parts(
    const unsigned char* data, unsigned int size,
    unsigned long long* partial_a, unsigned long long* partial_b,
    const unsigned int* parser_status = nullptr) {
  if (parser_status && *parser_status) return;
  __shared__ unsigned long long a[256], b[256];
  unsigned int i = blockIdx.x * 4096u + threadIdx.x;
  unsigned long long sa = 0, sb = 0;
  for (unsigned int j = 0; j < 16u; ++j, i += 256u) {
    if (i < size) {
      unsigned int value = data[i];
      sa += value;
      sb += (unsigned long long)(size - i) * value;
    }
  }
  a[threadIdx.x] = sa; b[threadIdx.x] = sb;
  __syncthreads();
  for (int stride = 128; stride; stride >>= 1) {
    if (threadIdx.x < stride) {
      a[threadIdx.x] += a[threadIdx.x + stride];
      b[threadIdx.x] += b[threadIdx.x + stride];
    }
    __syncthreads();
  }
  if (!threadIdx.x) {
    partial_a[blockIdx.x] = a[0]; partial_b[blockIdx.x] = b[0];
  }
}
extern "C" __global__ void adler_finish(
    const unsigned long long* partial_a,
    const unsigned long long* partial_b, unsigned int parts,
    unsigned int size, unsigned int* checksum, const DecodeState* state = nullptr,
    const unsigned int* parser_status = nullptr) {
  // Compression has no decode state. Failed decoding leaves partials unwritten.
  if (state && (state->status || state->active)) return;
  if (parser_status && *parser_status) return;
  __shared__ unsigned long long a[256], b[256];
  unsigned long long sa = 0, sb = 0;
  for (unsigned int i = threadIdx.x; i < parts; i += 256u) {
    sa += partial_a[i]; sb += partial_b[i];
  }
  a[threadIdx.x] = sa; b[threadIdx.x] = sb;
  __syncthreads();
  for (int stride = 128; stride; stride >>= 1) {
    if (threadIdx.x < stride) {
      a[threadIdx.x] += a[threadIdx.x + stride];
      b[threadIdx.x] += b[threadIdx.x + stride];
    }
    __syncthreads();
  }
  if (!threadIdx.x) {
    // size <= 2^28 keeps both unreduced 64-bit sums within range.
    unsigned int low = (unsigned int)((a[0] + 1u) % 65521u);
    unsigned int high = (unsigned int)((b[0] + size) % 65521u);
    *checksum = (high << 16) | low;
  }
}

'''
