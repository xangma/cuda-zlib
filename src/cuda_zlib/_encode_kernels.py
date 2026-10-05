# Copyright (c) 2026 xangma
# SPDX-License-Identifier: MIT

"""Independent RFC 1951 encoder with bounded per-chunk Huffman trees.

Chunks have no shared history. A histogram pass and deterministic replay use
the same greedy matcher. Complete stored, fixed and dynamic extents determine
the representation; empty stored blocks align nonfinal compressed chunks.
"""

KERNEL_NAMES = ("encode_chunks", "pack_chunks", "write_wrapper")
CUDA_SOURCE = r'''
typedef unsigned char enc_u8;
typedef unsigned short enc_u16;
typedef unsigned int enc_u32;
typedef unsigned long long enc_u64;

struct BitWriter {
  enc_u8* data;
  enc_u32 size, capacity, cache, bits, error;
  __device__ void byte(enc_u32 value) {
    if (size >= capacity) { error = 1; return; }
    data[size++] = (enc_u8)value;
  }
  __device__ void put(enc_u32 value, enc_u32 n) {
    cache |= value << bits;
    bits += n;
    while (bits >= 8) { byte(cache & 255u); cache >>= 8; bits -= 8; }
  }
  __device__ void align() {
    if (bits) { byte(cache & 255u); cache = 0; bits = 0; }
  }
};
__device__ enc_u32 enc_reverse(enc_u32 x, enc_u32 n) {
  enc_u32 result = 0;
  for (enc_u32 i = 0; i < n; ++i) { result = (result << 1) | (x & 1u); x >>= 1; }
  return result;
}
__device__ void fixed_symbol(BitWriter& w, enc_u32 symbol) {
  enc_u32 code, bits;
  if (symbol <= 143) { code = 0x30u + symbol; bits = 8; }
  else if (symbol <= 255) { code = 0x190u + symbol - 144u; bits = 9; }
  else if (symbol <= 279) { code = symbol - 256u; bits = 7; }
  else { code = 0xc0u + symbol - 280u; bits = 8; }
  w.put(enc_reverse(code, bits), bits);
}
__device__ enc_u32 enc_hash(const enc_u8* data, enc_u32 pos) {
  enc_u32 value = (enc_u32)data[pos] | ((enc_u32)data[pos+1] << 8) |
                 ((enc_u32)data[pos+2] << 16);
  return (value * 2654435761u) >> 19; // 8192 slots, 32 KiB shared memory
}
__device__ __constant__ enc_u32 enc_lb[29] = {
  3,4,5,6,7,8,9,10,11,13,15,17,19,23,27,31,35,43,51,59,67,83,99,
  115,131,163,195,227,258};
__device__ __constant__ enc_u32 enc_le[29] = {
  0,0,0,0,0,0,0,0,1,1,1,1,2,2,2,2,3,3,3,3,4,4,4,4,5,5,5,5,0};
__device__ __constant__ enc_u32 enc_db[30] = {
  1,2,3,4,5,7,9,13,17,25,33,49,65,97,129,193,257,385,513,769,
  1025,1537,2049,3073,4097,6145,8193,12289,16385,24577};
__device__ __constant__ enc_u32 enc_de[30] = {
  0,0,0,0,1,1,2,2,3,3,4,4,5,5,6,6,7,7,8,8,9,9,10,10,11,11,12,12,13,13};
__device__ __constant__ enc_u8 enc_cl_order[19] = {
  16,17,18,0,8,7,9,6,10,5,11,4,12,3,13,2,14,1,15};

__device__ bool enc_less(enc_u32 a, enc_u32 b, const enc_u32* weight) {
  return weight[a] < weight[b] || (weight[a] == weight[b] && a < b);
}
__device__ void enc_heap_push(enc_u32 node, enc_u16* heap, enc_u32& count,
                              const enc_u32* weight) {
  enc_u32 i = count++;
  while (i) {
    enc_u32 p = (i - 1u) >> 1;
    if (!enc_less(node, heap[p], weight)) break;
    heap[i] = heap[p]; i = p;
  }
  heap[i] = (enc_u16)node;
}
__device__ enc_u32 enc_heap_pop(enc_u16* heap, enc_u32& count,
                               const enc_u32* weight) {
  enc_u32 result = heap[0], node = heap[--count];
  if (!count) return result;
  enc_u32 i = 0;
  while (2u * i + 1u < count) {
    enc_u32 child = 2u * i + 1u;
    if (child + 1u < count && enc_less(heap[child+1u], heap[child], weight)) ++child;
    if (!enc_less(heap[child], node, weight)) break;
    heap[i] = heap[child]; i = child;
  }
  heap[i] = (enc_u16)node;
  return result;
}

// Build complete prefix trees, then canonical bit-reversed codes. Positive
// frequency flattening bounds depth without an overflow repair that could
// oversubscribe a tree. By shift=16 all active leaves have equal weight; at
// most 286 leaves then need 9 bits. Actual frequencies remain unchanged for cost.
__device__ bool enc_tree(const enc_u32* freq, enc_u32 alphabet, enc_u32 limit,
                        enc_u8* lengths, enc_u16* codes, enc_u16* parent,
                        enc_u16* heap, enc_u32* weight) {
  for (enc_u32 shift = 0; shift <= 16u; ++shift) {
    enc_u32 heap_size = 0;
    for (enc_u32 i = 0; i < 2u * alphabet - 1u; ++i) parent[i] = 65535u;
    for (enc_u32 i = 0; i < alphabet; ++i) {
      lengths[i] = 0; codes[i] = 0;
      weight[i] = freq[i] ? ((freq[i] + (1u << shift) - 1u) >> shift) : 0;
      if (weight[i]) enc_heap_push(i, heap, heap_size, weight);
    }
    // Empty distance alphabets and one-symbol alphabets get an unused dummy
    // leaf. This retains a complete tree, including for empty input.
    if (!heap_size) {
      weight[0] = 1; enc_heap_push(0, heap, heap_size, weight);
    }
    if (heap_size == 1u) {
      enc_u32 dummy = heap[0] ? 0u : 1u;
      weight[dummy] = 1; enc_heap_push(dummy, heap, heap_size, weight);
    }
    enc_u32 next = alphabet;
    while (heap_size > 1u) {
      enc_u32 a = enc_heap_pop(heap, heap_size, weight);
      enc_u32 b = enc_heap_pop(heap, heap_size, weight);
      weight[next] = weight[a] + weight[b];
      parent[a] = (enc_u16)next; parent[b] = (enc_u16)next;
      enc_heap_push(next++, heap, heap_size, weight);
    }
    bool overlong = false;
    // Internal weights are dead once the heap is consumed. Reuse that shared
    // scratch instead of spilling dynamically indexed automatic arrays.
    // Leaf weights [0, alphabet) still identify every participating symbol.
    enc_u32* count = weight + alphabet;
    enc_u32* first = count + 16;
    for (enc_u32 i = 0; i < 16u; ++i) count[i] = first[i] = 0;
    for (enc_u32 i = 0; i < alphabet; ++i) {
      if (!weight[i]) continue;
      enc_u32 node = i, depth = 0;
      while (parent[node] != 65535u) {
        node = parent[node];
        if (++depth > limit) { overlong = true; break; }
      }
      if (depth <= limit) { lengths[i] = (enc_u8)depth; ++count[depth]; }
    }
    if (overlong) continue;
    enc_u32 code = 0;
    for (enc_u32 bits = 1; bits <= limit; ++bits) {
      code = (code + count[bits-1u]) << 1; first[bits] = code;
    }
    for (enc_u32 i = 0; i < alphabet; ++i)
      if (lengths[i]) codes[i] = (enc_u16)enc_reverse(first[lengths[i]]++, lengths[i]);
    return true;
  }
  return false;
}

__device__ enc_u8 enc_length_at(enc_u32 i, enc_u32 nll,
                              const enc_u8* ll, const enc_u8* dist) {
  return i < nll ? ll[i] : dist[i - nll];
}
__device__ enc_u32 enc_rle(enc_u32 nll, enc_u32 ndist, const enc_u8* ll,
                         const enc_u8* dist, enc_u8* symbols, enc_u8* extras,
                         enc_u32* freq) {
  enc_u32 i = 0, size = 0;
  while (i < nll + ndist) {
    enc_u32 value = enc_length_at(i, nll, ll, dist), run = 1;
    while (i + run < nll + ndist && enc_length_at(i+run, nll, ll, dist) == value) ++run;
    i += run;
    if (value) {
      symbols[size] = (enc_u8)value; extras[size++] = 0; ++freq[value]; --run;
      while (run >= 3u) {
        enc_u32 take = run < 6u ? run : 6u;
        symbols[size] = 16; extras[size++] = (enc_u8)(take-3u); ++freq[16]; run -= take;
      }
    } else {
      while (run >= 11u) {
        enc_u32 take = run < 138u ? run : 138u;
        symbols[size] = 18; extras[size++] = (enc_u8)(take-11u); ++freq[18]; run -= take;
      }
      if (run >= 3u) {
        symbols[size] = 17; extras[size++] = (enc_u8)(run-3u); ++freq[17]; run = 0;
      }
    }
    while (run) {
      symbols[size] = (enc_u8)value; extras[size++] = 0; ++freq[value]; --run;
    }
  }
  return size;
}

// Both passes intentionally preserve the original one-candidate matcher.
// No token indexes are saved: the emission pass reads fresh input again.
template<bool Collect> __device__ void enc_parse(
    const enc_u8* data, enc_u32 n, enc_u32* last,
    enc_u32* ll_freq, enc_u32* d_freq, enc_u32& extra_bits, BitWriter& w,
    bool dynamic, const enc_u8* ll_lengths, const enc_u16* ll_codes,
    const enc_u8* d_lengths, const enc_u16* d_codes) {
  enc_u32 pos = 0;
  while (pos < n) {
    enc_u32 match = 0, distance = 0;
    if (n - pos >= 3) {
      enc_u32 hash = enc_hash(data, pos);
      enc_u32 previous = last[hash];
      last[hash] = pos + 1u;
      if (previous) {
        enc_u32 earlier = previous - 1u;
        distance = pos - earlier;
        if (distance && distance <= 32768u) {
          enc_u32 limit = n - pos;
          if (limit > 258u) limit = 258u;
          while (match < limit && data[earlier + match] == data[pos + match]) ++match;
        }
      }
    }
    if (match >= 3) {
      enc_u32 li = 0, di = 0;
      while (li < 28u && match >= enc_lb[li+1u]) ++li;
      while (di < 29u && distance >= enc_db[di+1u]) ++di;
      if (Collect) {
        ++ll_freq[257u+li]; ++d_freq[di]; extra_bits += enc_le[li] + enc_de[di];
      } else {
        if (dynamic) w.put(ll_codes[257u+li], ll_lengths[257u+li]);
        else fixed_symbol(w, 257u+li);
        w.put(match - enc_lb[li], enc_le[li]);
        w.put(dynamic ? d_codes[di] : enc_reverse(di, 5), dynamic ? d_lengths[di] : 5u);
        w.put(distance - enc_db[di], enc_de[di]);
      }
      for (enc_u32 j = 1; j < match && n - (pos + j) >= 3; ++j)
        last[enc_hash(data, pos + j)] = pos + j + 1u;
      pos += match;
    } else {
      if (Collect) ++ll_freq[data[pos]];
      else if (dynamic) w.put(ll_codes[data[pos]], ll_lengths[data[pos]]);
      else fixed_symbol(w, data[pos]);
      ++pos;
    }
  }
  if (Collect) ++ll_freq[256];
  else if (dynamic) w.put(ll_codes[256], ll_lengths[256]);
  else fixed_symbol(w, 256);
}

__device__ enc_u32 enc_extent(enc_u32 bits, bool final) {
  return final ? (bits+7u)/8u : (bits+3u+7u)/8u+4u;
}
extern "C" __global__ void encode_chunks(
    const enc_u8* input, enc_u32 total, enc_u32 chunk_bytes, enc_u32 chunks,
    enc_u32 slot_bytes, enc_u8* scratch, enc_u32* sizes, enc_u32* status) {
  __shared__ enc_u32 last[8192];
  __shared__ enc_u32 freq[335], weight[571];
  __shared__ enc_u16 codes[335], parent[571], heap[286];
  __shared__ enc_u8 lengths[335], rle_symbols[316], rle_extras[316];
  __shared__ enc_u32 extra_bits, nll, ndist, ncl, rle_size, mode, extent;
  enc_u32 chunk = blockIdx.x;
  if (chunk >= chunks) return;
  enc_u32 start = chunk * chunk_bytes;
  enc_u32 n = total - start;
  if (n > chunk_bytes) n = chunk_bytes;
  const enc_u8* data = input + start;
  enc_u8* slot = scratch + (enc_u64)chunk * slot_bytes;
  bool final = chunk + 1u == chunks;
  for (enc_u32 i = threadIdx.x; i < 8192u; i += blockDim.x) last[i] = 0;
  for (enc_u32 i = threadIdx.x; i < 335u; i += blockDim.x) {
    freq[i] = 0; lengths[i] = 0; codes[i] = 0;
  }
  __syncthreads();
  BitWriter w = {slot, 0, slot_bytes, 0, 0, 0};
  if (!threadIdx.x) {
    extra_bits = 0; mode = 0; extent = n + 5u; status[chunk] = 0;
    enc_parse<true>(data, n, last, freq, freq+286, extra_bits, w, false,
                    lengths, codes, lengths+286, codes+286);
    bool trees = enc_tree(freq, 286, 15, lengths, codes, parent, heap, weight) &&
                 enc_tree(freq+286, 30, 15, lengths+286, codes+286, parent, heap, weight);
    nll = 286; ndist = 30;
    while (nll > 257u && !lengths[nll-1u]) --nll;
    while (ndist > 1u && !lengths[286u+ndist-1u]) --ndist;
    rle_size = enc_rle(nll, ndist, lengths, lengths+286, rle_symbols, rle_extras, freq+316);
    trees = trees && enc_tree(freq+316, 19, 7, lengths+316, codes+316, parent, heap, weight);
    ncl = 19;
    while (ncl > 4u && !lengths[316u+enc_cl_order[ncl-1u]]) --ncl;
    if (!trees) { mode = 3; status[chunk] = 1; }
    else {
      enc_u32 fixed_bits = 3u + extra_bits, dynamic_bits = 17u + 3u*ncl + extra_bits;
      for (enc_u32 i = 0; i < 286u; ++i) {
        fixed_bits += freq[i] * (i <= 143u ? 8u : i <= 255u ? 9u : i <= 279u ? 7u : 8u);
        dynamic_bits += freq[i] * lengths[i];
      }
      for (enc_u32 i = 0; i < 30u; ++i) {
        fixed_bits += freq[286u+i] * 5u;
        dynamic_bits += freq[286u+i] * lengths[286u+i];
      }
      for (enc_u32 i = 0; i < 19u; ++i)
        dynamic_bits += freq[316u+i] * (lengths[316u+i] + (i == 16u ? 2u : i == 17u ? 3u : i == 18u ? 7u : 0u));
      enc_u32 fixed_extent = enc_extent(fixed_bits, final);
      enc_u32 dynamic_extent = enc_extent(dynamic_bits, final);
      if (fixed_extent < extent) { extent = fixed_extent; mode = 1; }
      if (dynamic_extent < extent) { extent = dynamic_extent; mode = 2; }
    }
  }
  __syncthreads();
  if (mode == 3u) return;
  if (!mode) {
    if (!threadIdx.x) {
      slot[0] = final ? 1u : 0u;
      slot[1] = (enc_u8)n; slot[2] = (enc_u8)(n >> 8);
      enc_u32 complement = n ^ 65535u;
      slot[3] = (enc_u8)complement; slot[4] = (enc_u8)(complement >> 8);
      sizes[chunk] = extent;
    }
    for (enc_u32 i = threadIdx.x; i < n; i += blockDim.x) slot[5u+i] = data[i];
    return;
  }
  for (enc_u32 i = threadIdx.x; i < 8192u; i += blockDim.x) last[i] = 0;
  __syncthreads();
  if (threadIdx.x) return;
  bool dynamic = mode == 2u;
  w.put((final ? 1u : 0u) | (dynamic ? 4u : 2u), 3);
  if (dynamic) {
    w.put(nll-257u, 5); w.put(ndist-1u, 5); w.put(ncl-4u, 4);
    for (enc_u32 i = 0; i < ncl; ++i) w.put(lengths[316u+enc_cl_order[i]], 3);
    for (enc_u32 i = 0; i < rle_size; ++i) {
      enc_u32 symbol = rle_symbols[i];
      w.put(codes[316u+symbol], lengths[316u+symbol]);
      if (symbol >= 16u) w.put(rle_extras[i], symbol == 16u ? 2u : symbol == 17u ? 3u : 7u);
    }
  }
  enc_parse<false>(data, n, last, freq, freq+286, extra_bits, w, dynamic,
                   lengths, codes, lengths+286, codes+286);
  if (!final) {
    w.put(0, 3); // Nonfinal empty stored block realigns the next chunk.
    w.align();
    w.byte(0); w.byte(0); w.byte(255); w.byte(255);
  } else w.align();
  if (w.error) { status[chunk] = 1; return; }
  if (w.size != extent) { status[chunk] = 1; return; }
  sizes[chunk] = w.size;
  status[chunk] = w.error;
}
extern "C" __global__ void pack_chunks(
    const enc_u8* scratch, enc_u32 slot_bytes, const enc_u32* sizes,
    const enc_u64* ends, enc_u8* output) {
  enc_u32 chunk = blockIdx.x;
  enc_u64 offset = chunk ? ends[chunk-1] : 0;
  for (enc_u32 i = threadIdx.x; i < sizes[chunk]; i += blockDim.x)
    output[2u + offset + i] = scratch[(enc_u64)chunk * slot_bytes + i];
}
extern "C" __global__ void write_wrapper(
    enc_u8* output, enc_u64 size, const enc_u32* checksum) {
  if (threadIdx.x || blockIdx.x) return;
  output[0] = 0x78; output[1] = 0x01; // 32 KiB window, FCHECK, no dictionary
  enc_u32 value = checksum[0];
  output[size-4] = (enc_u8)(value >> 24);
  output[size-3] = (enc_u8)(value >> 16);
  output[size-2] = (enc_u8)(value >> 8);
  output[size-1] = (enc_u8)value;
}
'''
