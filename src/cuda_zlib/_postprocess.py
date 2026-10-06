# Copyright (c) 2026 xangma
# SPDX-License-Identifier: MIT

"""CUDA checksum and reference-resolution kernels."""

CUDA_SOURCE = r'''
extern "C" __global__ void refine_roots(
    const unsigned int* input, unsigned int* output, unsigned int size,
    unsigned int* pending, unsigned int* status) {
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  int unresolved = 0;
  if (i < size) {
    unsigned int value = input[i];
    bool valid = true;
    for (int step = 0; step < 32 && !(value & 0x80000000u); ++step) {
      if (value >= i) { atomicExch(status, 6u); valid = false; break; }
      unsigned int next = input[value];
      if (!(next & 0x80000000u) && next >= value) {
        atomicExch(status, 6u); valid = false; break;
      }
      value = next;
    }
    if (valid) {
      output[i] = value;
      unresolved = !(value & 0x80000000u);
    }
  }
  // Include inactive and invalid threads in the barrier, but only valid
  // unresolved roots in the vote. Signal pending once per block.
  int block_pending = __syncthreads_or(unresolved);
  if (threadIdx.x == 0 && block_pending) atomicExch(pending, 1u);
}
// Resolved roots hold tagged literals. Gather the returned bytes while making
// checksum partials, avoiding a separate gather launch and output read.
extern "C" __global__ void write_adler_parts(
    const unsigned int* roots, unsigned char* output, unsigned int size,
    unsigned long long* partial_a, unsigned long long* partial_b) {
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
    unsigned long long* partial_a, unsigned long long* partial_b) {
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
    unsigned int size, unsigned int* checksum) {
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
