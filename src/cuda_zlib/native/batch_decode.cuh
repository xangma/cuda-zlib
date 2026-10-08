// Copyright (c) 2026 xangma
// SPDX-License-Identifier: MIT

// Included inside the codec's unnamed namespace after its types and statuses.
// Host validation guarantees that each descriptor fits its packed buffers and
// that output ranges are disjoint. Both kernels launch exactly 32 threads.
struct BatchDecodeFile {
  U32 input_offset;
  U32 input_size;
  U32 output_offset;
  U32 output_size;
};

// Metadata is captured after a successful table build and held as values.
// Output stores cannot alias these scalars; the canonical decoder is unchanged.
template<int N, int PrimaryBits>
__device__ __forceinline__ int SmallHuffmanDecode(
    decoder::BitReader& r, const decoder::Huffman<N, PrimaryBits>& table,
    U32 maximum, U32 lookup) {
  if (lookup && !r.error && r.pos <= r.bits &&
      PrimaryBits <= r.bits - r.pos) {
      decoder::u16 entry = table.primary[r.peek(PrimaryBits)];
      if (entry) {
          r.drop(entry >> 9);
          return entry & 511;
      }
  }
  decoder::u32 code = 0, first = 0, index = 0;
  for (decoder::u32 len = 1; len <= maximum; ++len) {
      code |= r.take(1);
      if (r.error) return -1;
      if (code < first + table.count[len])
          return int(table.symbols[index + code - first]);
      index += table.count[len];
      first = (first + table.count[len]) << 1;
      code <<= 1;
  }
  r.error = 4;
  return -1;
}

// Small/batch token decoding keeps base/extra calculation in registers.
// The ordinary decoder retains its original helpers.
__device__ __forceinline__ int SmallTokenRemainder(
    decoder::BitReader& r, const decoder::DecodeTables& tables, int symbol,
    U32 dd_maximum, U32 dd_lookup, U32& size, U32& distance,
    U32 window_bytes = 32768) {
  size = 0;
  distance = 0;
  if (r.error || symbol == 256) return symbol;
  if (symbol < 256) { size = 1; return symbol; }
  if (symbol < 257 || symbol > 285) { r.error = 4; return symbol; }
  const U32 index = U32(symbol - 257);
  const U32 length_extra = index < 8 || index == 28 ? 0 : (index >> 2) - 1;
  const U32 length_base = index < 8 ? index + 3 : index == 28 ? 258 :
      3 + ((4 + (index & 3)) << length_extra);
  size = length_base + r.take(length_extra);
  if (r.error) return symbol;
  const int ds = SmallHuffmanDecode(r, tables.dd, dd_maximum, dd_lookup);
  if (r.error) return symbol;
  if (ds < 0 || ds > 29) { r.error = 4; return symbol; }
  const U32 distance_index = U32(ds);
  const U32 distance_extra = distance_index < 4 ? 0 : (distance_index >> 1) - 1;
  const U32 distance_base = distance_index < 4 ? distance_index + 1 :
      1 + ((2 + (distance_index & 1)) << distance_extra);
  distance = distance_base + r.take(distance_extra);
  if (!r.error && (!distance || distance > window_bytes)) r.error = 6;
  return symbol;
}

__device__ __forceinline__ int SmallFixedToken(
    decoder::BitReader& r, const decoder::DecodeTables& tables,
    U32 ll_maximum, U32 ll_lookup, U32 dd_maximum, U32 dd_lookup,
    U32& size, U32& distance, U32 window_bytes = 32768) {
  const int symbol = SmallHuffmanDecode(r, tables.ll, ll_maximum, ll_lookup);
  return SmallTokenRemainder(r, tables, symbol, dd_maximum, dd_lookup,
      size, distance, window_bytes);
}

__device__ __forceinline__ U32 SmallExtendRepeatRun(
    decoder::BitReader& r, const decoder::DecodeTables& tables,
    U32 ll_maximum, U32 ll_lookup, U32 dd_maximum, U32 dd_lookup,
    U32 length, U32 available, U32 window_bytes, U64 end) {
  // Preserve the ordinary parser's checkpoint, limit and error precedence.
  while (length < 8192 && r.pos < end) {
    const decoder::BitReader saved = r;
    U32 next_length, next_distance;
    const int symbol = SmallFixedToken(
        r, tables, ll_maximum, ll_lookup, dd_maximum, dd_lookup,
        next_length, next_distance, window_bytes);
    if (r.error || symbol == 256 || next_distance != 1 || r.pos > end ||
        next_length > available - length || next_length > 8192 - length) {
      r = saved;
      break;
    }
    length += next_length;
  }
  return length;
}

__device__ __forceinline__ void DecodeMatch(U8* output, U32 begin,
                                          U32 distance, U32 length) {
  const U32 lane = threadIdx.x;
  const U32 first = begin - distance;
  if (distance == 1) {
    U32 value = 0;
    if (!lane) value = output[first];
    value = __shfl_sync(0xffffffffu, value, 0);
    for (U32 j = lane; j < length; j += 32) output[begin + j] = U8(value);
  } else {
    // Only the already decoded distance-byte seed is read, including when
    // the match overlaps its destination. No lane reads another lane's store.
    U32 source = first + lane % distance;
    const U32 stride = 32 % distance;
    for (U32 j = lane; j < length; j += 32) {
      output[begin + j] = output[source];
      source += stride;
      if (source >= begin) source -= distance;
    }
  }
}

__device__ __noinline__ void DecodeOneFile(
    const U8* input, U32 input_size, U8* output, U32 output_size,
    U32 max_blocks, U32* metadata, decoder::DecodeTables& tables,
    bool parallel_finish = false) {
  constexpr U32 mask = 0xffffffffu;
  const U32 lane = threadIdx.x;
  U32 error = 0, produced = 0, final = 0, blocks = 0;
  U32 window = 0, wanted_checksum = 0;
  U32 ll_maximum = 0, ll_lookup = 0, dd_maximum = 0, dd_lookup = 0;
  bool need_header = true, complete = false;
  decoder::BitReader reader = {nullptr, 0, 0, 0, 0, 0};

  if (input_size > kMaxBytes || output_size > kMaxBytes || !max_blocks ||
      max_blocks > kMaxBlocks) {
    if (!lane) { metadata[0] = kInvalidBounds; metadata[1] = 0; }
    return;
  }
  // Parallel finishing requires a stream-ordered output clear before
  // parsing and an Adler verifier afterward. Small/batch callers retain both.
  if (!parallel_finish)
    for (U32 i = lane; i < output_size; i += 32) output[i] = 0;
  __syncwarp(mask);
  if (!lane) {
    if (input_size < 8) {
      error = kTruncatedZlib;
    } else {
      const U32 cmf = input[0], flg = input[1];
      if (cmf == 0x1f && flg == 0x8b) error = kUnsupportedGzip;
      else if ((cmf & 15) != 8 || (cmf >> 4) > 7 ||
               (cmf * 256 + flg) % 31) error = kInvalidHeader;
      else if (flg & 32) error = kUnsupportedDictionary;
      else {
        reader.data = input + 2;
        reader.bits = U64(input_size - 6) * 8;
        window = 1u << ((cmf >> 4) + 8);
        wanted_checksum = (U32(input[input_size - 4]) << 24) |
                          (U32(input[input_size - 3]) << 16) |
                          (U32(input[input_size - 2]) << 8) |
                          U32(input[input_size - 1]);
      }
    }
  }

  while (true) {
    // Lane zero advances through literals and short matches. The remaining
    // lanes rendezvous only for a stored block, a long match or termination.
    U32 action = 0, length = 0, begin = 0, distance = 0, input_begin = 0;
    if (!lane) {
      while (!error && !complete) {
        if (need_header) {
          if (blocks >= max_blocks) { error = 9; break; }
          ++blocks;
          final = reader.take(1);
          const U32 type = reader.take(2);
          if (reader.error) { error = reader.error; break; }
          if (type == 3) { error = 2; break; }
          if (type == 0) {
            reader.seek((reader.pos + 7) & ~U64(7));
            const U32 n = reader.take(16), complement = reader.take(16);
            if (reader.error) { error = reader.error; break; }
            if ((n ^ complement) != 65535u) { error = 4; break; }
            if (n > output_size - produced) { error = 7; break; }
            if (reader.pos > reader.bits ||
                U64(n) * 8 > reader.bits - reader.pos) { error = 1; break; }
            begin = produced;
            length = n;
            input_begin = U32(reader.pos >> 3) + 2;
            produced += n;
            reader.seek(reader.pos + U64(n) * 8);
            complete = final != 0;
            action = 2;
            break;
          }
          error = type == 1 ? decoder::fixed_tables(tables.ll, tables.dd) :
              decoder::dynamic_tables(reader, tables.ll, tables.dd, tables.cl);
          need_header = false;
          if (error) break;
          ll_maximum = tables.ll.maximum;
          ll_lookup = tables.ll.lookup;
          dd_maximum = tables.dd.maximum;
          dd_lookup = tables.dd.lookup;
        }
        U32 size, token_distance;
        const int symbol = SmallFixedToken(
            reader, tables, ll_maximum, ll_lookup, dd_maximum, dd_lookup,
            size, token_distance, window);
        if (reader.error) { error = reader.error; break; }
        if (symbol == 256) {
          complete = final != 0;
          need_header = true;
          continue;
        }
        if (size > output_size - produced) { error = 7; break; }
        if (!token_distance) {
          output[produced++] = U8(symbol);
          continue;
        }
        if (token_distance > produced) { error = 6; break; }
        if (size >= 32) {
          begin = produced;
          length = token_distance == 1 ? SmallExtendRepeatRun(
              reader, tables, ll_maximum, ll_lookup, dd_maximum, dd_lookup,
              size, output_size - produced, window,
              reader.bits) : size;
          distance = token_distance;
          produced += length;
          action = 1;
          break;
        }
        const U32 first = produced - token_distance;
        if (token_distance == 1) {
          const U8 value = output[first];
          for (U32 j = 0; j < size; ++j) output[produced + j] = value;
        } else if (size == 3 && token_distance >= 3) {
          const U8 a = output[first], b = output[first + 1], c = output[first + 2];
          output[produced] = a;
          output[produced + 1] = b;
          output[produced + 2] = c;
        } else {
          U32 source = first;
          for (U32 j = 0; j < size; ++j) {
            output[produced + j] = output[source];
            if (++source == produced) source = first;
          }
        }
        produced += size;
      }
      if (!action && !error) {
        if (produced != output_size) error = 11;
        // Unused bits in the final byte are allowed; extra bytes are not.
        else if ((reader.pos + 7) / 8 != input_size - 6) error = 12;
      }
    }
    action = __shfl_sync(mask, action, 0);
    if (!action) break;
    length = __shfl_sync(mask, length, 0);
    begin = __shfl_sync(mask, begin, 0);
    __syncwarp(mask);
    if (action == 1) {
      distance = __shfl_sync(mask, distance, 0);
      DecodeMatch(output, begin, distance, length);
    } else {
      input_begin = __shfl_sync(mask, input_begin, 0);
      for (U32 j = lane; j < length; j += 32)
        output[begin + j] = input[input_begin + j];
    }
    // Publish cooperative output before the parser consumes its history.
    __syncwarp(mask);
  }

  error = __shfl_sync(mask, error, 0);
  if (!error && !parallel_finish) {
    __syncwarp(mask);
    U64 sum = 0, weighted = 0;
    for (U32 i = lane; i < output_size; i += 32) {
      const U32 value = output[i];
      sum += value;
      weighted += U64(output_size - i) * value;
    }
    for (U32 step = 16; step; step >>= 1) {
      sum += __shfl_down_sync(mask, sum, step);
      weighted += __shfl_down_sync(mask, weighted, step);
    }
    if (!lane) {
      const U32 checksum = U32((weighted + output_size) % 65521) << 16 |
                           U32((sum + 1) % 65521);
      if (checksum != wanted_checksum) error = kAdlerMismatch;
    }
  }
  if (!lane) {
    metadata[0] = error;
    metadata[1] = 0;
  }
}

__global__ void BatchDecode(const U8* input, U8* output,
                            const BatchDecodeFile* descriptors, U32 files,
                            U32 max_blocks, U32* metadata,
                            const U32* encoded_metadata) {
  __shared__ decoder::DecodeTables tables;
  if (blockIdx.x >= files) return;
  const BatchDecodeFile file = descriptors[blockIdx.x];
  U32 input_size = file.input_size;
  if (encoded_metadata) {
    // Compression slots keep static offsets while their encoded lengths vary.
    // Failed or oversized slots must never be passed to the Deflate parser.
    input_size = encoded_metadata[2 * blockIdx.x];
    const U32 encode_status = encoded_metadata[2 * blockIdx.x + 1];
    U32 error = encode_status ? kCompressionOverflow :
                input_size > file.input_size ? kInvalidBounds : 0;
    if (file.output_size > kMaxBytes) error = kInvalidBounds;
    if (error) {
      if (file.output_size <= kMaxBytes)
        for (U32 i = threadIdx.x; i < file.output_size; i += 32)
          output[file.output_offset + i] = 0;
      if (!threadIdx.x) {
        metadata[2 * blockIdx.x] = error;
        metadata[2 * blockIdx.x + 1] = 0;
      }
      return;
    }
  }
  DecodeOneFile(input + file.input_offset, input_size,
                output + file.output_offset, file.output_size, max_blocks,
                metadata + 2 * blockIdx.x, tables);
}

__global__ void SmallDecode(const U8* input, U32 input_size, U8* output,
                            U32 output_size, U32 max_blocks, U32* metadata,
                            bool parallel_finish = false) {
  __shared__ decoder::DecodeTables tables;
  DecodeOneFile(input, input_size, output, output_size, max_blocks, metadata,
                tables, parallel_finish);
}

// Separate shared-address helpers keep the original global decoder callers
// unchanged. The scratch array contains the complete decoded output extent.
// Explicit byte operations preserve all bounds/history/checksum semantics.
struct SharedDecodeByte {
  U32 address;
  __device__ __forceinline__ operator U8() const {
    U32 value;
    asm volatile("ld.shared.u8 %0, [%1];"
                 : "=r"(value) : "r"(address) : "memory");
    return U8(value);
  }
  __device__ __forceinline__ void operator=(U8 value) const {
    asm volatile("st.shared.u8 [%0], %1;"
                 : : "r"(address), "r"(U32(value)) : "memory");
  }
  __device__ __forceinline__ void operator=(const SharedDecodeByte& source) const {
    *this = U8(source);
  }
};
struct SharedDecodeOutput {
  U32 address;
  __device__ __forceinline__ SharedDecodeByte operator[](U32 index) const {
    return {address + index};
  }
};

__device__ __forceinline__ void DecodeMatchShared(SharedDecodeOutput output, U32 begin,
                                          U32 distance, U32 length) {
  const U32 lane = threadIdx.x;
  const U32 first = begin - distance;
  if (distance == 1) {
    U32 value = 0;
    if (!lane) value = output[first];
    value = __shfl_sync(0xffffffffu, value, 0);
    for (U32 j = lane; j < length; j += 32) output[begin + j] = U8(value);
  } else {
    // Only the already decoded distance-byte seed is read, including when
    // the match overlaps its destination. No lane reads another lane's store.
    U32 source = first + lane % distance;
    const U32 stride = 32 % distance;
    for (U32 j = lane; j < length; j += 32) {
      output[begin + j] = output[source];
      source += stride;
      if (source >= begin) source -= distance;
    }
  }
}

// Each lane probes one bounded bit offset, then all lanes follow only the
// actual literal successor chain. The caller publishes stores with __syncwarp.
__device__ __forceinline__ U32 SmallSharedLiteralRun(
    const decoder::u16* primary, U64 cache, U32 cached,
    SharedDecodeOutput output, U32 begin, U32 available, U32& consumed) {
  constexpr U32 mask = 0xffffffffu;
  const U32 lane = threadIdx.x;
  const U32 entry = lane + 9 <= cached ?
      U32(primary[U32(cache >> lane) & 511u]) : 0;
  const U32 capacity = available < 32 ? available : 32;
  U32 offset = 0, count = 0, byte = 0;
  while (offset < 32 && count < capacity) {
    const U32 current = __shfl_sync(mask, entry, offset);
    if (!current || (current & 511u) >= 256) break;
    if (lane == count) byte = current & 511u;
    offset += current >> 9;
    ++count;
  }
  if (lane < count) output[begin + lane] = U8(byte);
  consumed = offset;
  return count;
}

__device__ __noinline__ void DecodeOneFileShared(
    const U8* input, U32 input_size, SharedDecodeOutput output, U32 output_size,
    U32 max_blocks, U32* metadata, decoder::DecodeTables& tables,
    bool parallel_finish = false) {
  constexpr U32 mask = 0xffffffffu;
  const U32 lane = threadIdx.x;
  U32 error = 0, produced = 0, final = 0, blocks = 0;
  U32 window = 0, wanted_checksum = 0;
  U32 ll_maximum = 0, ll_lookup = 0, dd_maximum = 0, dd_lookup = 0;
  bool need_header = true, complete = false;
  decoder::BitReader reader = {nullptr, 0, 0, 0, 0, 0};

  if (input_size > kMaxBytes || output_size > kMaxBytes || !max_blocks ||
      max_blocks > kMaxBlocks) {
    if (!lane) { metadata[0] = kInvalidBounds; metadata[1] = 0; }
    return;
  }
  // Parallel finishing requires a stream-ordered output clear before
  // parsing and an Adler verifier afterward. Small/batch callers retain both.
  if (!parallel_finish)
    for (U32 i = lane; i < output_size; i += 32) output[i] = 0;
  __syncwarp(mask);
  if (!lane) {
    if (input_size < 8) {
      error = kTruncatedZlib;
    } else {
      const U32 cmf = input[0], flg = input[1];
      if (cmf == 0x1f && flg == 0x8b) error = kUnsupportedGzip;
      else if ((cmf & 15) != 8 || (cmf >> 4) > 7 ||
               (cmf * 256 + flg) % 31) error = kInvalidHeader;
      else if (flg & 32) error = kUnsupportedDictionary;
      else {
        reader.data = input + 2;
        reader.bits = U64(input_size - 6) * 8;
        window = 1u << ((cmf >> 4) + 8);
        wanted_checksum = (U32(input[input_size - 4]) << 24) |
                          (U32(input[input_size - 3]) << 16) |
                          (U32(input[input_size - 2]) << 8) |
                          U32(input[input_size - 1]);
      }
    }
  }

  while (true) {
    // Lane zero keeps the canonical token path; a completed literal can
    // rendezvous with the warp for a bounded additional literal prefix.
    U32 action = 0, length = 0, begin = 0, distance = 0, input_begin = 0;
    U64 literal_cache = 0;
    U32 literal_cached = 0;
    if (!lane) {
      while (!error && !complete) {
        if (need_header) {
          if (blocks >= max_blocks) { error = 9; break; }
          ++blocks;
          final = reader.take(1);
          const U32 type = reader.take(2);
          if (reader.error) { error = reader.error; break; }
          if (type == 3) { error = 2; break; }
          if (type == 0) {
            reader.seek((reader.pos + 7) & ~U64(7));
            const U32 n = reader.take(16), complement = reader.take(16);
            if (reader.error) { error = reader.error; break; }
            if ((n ^ complement) != 65535u) { error = 4; break; }
            if (n > output_size - produced) { error = 7; break; }
            if (reader.pos > reader.bits ||
                U64(n) * 8 > reader.bits - reader.pos) { error = 1; break; }
            begin = produced;
            length = n;
            input_begin = U32(reader.pos >> 3) + 2;
            produced += n;
            reader.seek(reader.pos + U64(n) * 8);
            complete = final != 0;
            action = 2;
            break;
          }
          error = type == 1 ? decoder::fixed_tables(tables.ll, tables.dd) :
              decoder::dynamic_tables(reader, tables.ll, tables.dd, tables.cl);
          need_header = false;
          if (error) break;
          ll_maximum = tables.ll.maximum;
          ll_lookup = tables.ll.lookup;
          dd_maximum = tables.dd.maximum;
          dd_lookup = tables.dd.lookup;
        }
        U32 size, token_distance;
        int symbol = SmallHuffmanDecode(reader, tables.ll, ll_maximum, ll_lookup);
      finish_token:
        symbol = SmallTokenRemainder(
            reader, tables, symbol, dd_maximum, dd_lookup,
            size, token_distance, window);
        if (reader.error) { error = reader.error; break; }
        if (symbol == 256) {
          complete = final != 0;
          need_header = true;
          continue;
        }
        if (size > output_size - produced) { error = 7; break; }
        if (!token_distance) {
          output[produced++] = U8(symbol);
          if (ll_lookup && !reader.error && reader.pos <= reader.bits &&
              9 <= reader.bits - reader.pos && produced < output_size) {
            const U32 entry = U32(tables.ll.primary[reader.peek(9)]);
            if (entry && (entry & 511u) < 256) {
              literal_cache = reader.cache;
              literal_cached = reader.cached;
              begin = produced;
              length = output_size - produced;
              action = 3;
              break;
            }
            if (entry) {
              // Decode this token at the current position without a loop handoff.
              reader.drop(entry >> 9);
              symbol = int(entry & 511u);
              goto finish_token;
            }
          }
          continue;
        }
        if (token_distance > produced) { error = 6; break; }
        if (size >= 32) {
          begin = produced;
          length = token_distance == 1 ? SmallExtendRepeatRun(
              reader, tables, ll_maximum, ll_lookup, dd_maximum, dd_lookup,
              size, output_size - produced, window,
              reader.bits) : size;
          distance = token_distance;
          produced += length;
          action = 1;
          break;
        }
        const U32 first = produced - token_distance;
        if (token_distance == 1) {
          const U8 value = output[first];
          for (U32 j = 0; j < size; ++j) output[produced + j] = value;
        } else if (size == 3 && token_distance >= 3) {
          const U8 a = output[first], b = output[first + 1], c = output[first + 2];
          output[produced] = a;
          output[produced + 1] = b;
          output[produced + 2] = c;
        } else {
          U32 source = first;
          for (U32 j = 0; j < size; ++j) {
            output[produced + j] = output[source];
            if (++source == produced) source = first;
          }
        }
        produced += size;
      }
      if (!action && !error) {
        if (produced != output_size) error = 11;
        // Unused bits in the final byte are allowed; extra bytes are not.
        else if ((reader.pos + 7) / 8 != input_size - 6) error = 12;
      }
    }
    action = __shfl_sync(mask, action, 0);
    if (!action) break;
    length = __shfl_sync(mask, length, 0);
    begin = __shfl_sync(mask, begin, 0);
    __syncwarp(mask);
    if (action == 1) {
      distance = __shfl_sync(mask, distance, 0);
      DecodeMatchShared(output, begin, distance, length);
    } else if (action == 3) {
      literal_cache = __shfl_sync(mask, literal_cache, 0);
      literal_cached = __shfl_sync(mask, literal_cached, 0);
      U32 consumed = 0;
      const U32 literals = SmallSharedLiteralRun(
          tables.ll.primary, literal_cache, literal_cached,
          output, begin, length, consumed);
      if (!lane && literals) {
        // The leader reader still holds the dispatched literal reservoir.
        reader.drop(consumed);
        produced += literals;
      }
    } else {
      input_begin = __shfl_sync(mask, input_begin, 0);
      for (U32 j = lane; j < length; j += 32)
        output[begin + j] = input[input_begin + j];
    }
    // Publish cooperative output before the parser consumes its history.
    __syncwarp(mask);
  }

  error = __shfl_sync(mask, error, 0);
  if (!error && !parallel_finish) {
    __syncwarp(mask);
    U64 sum = 0, weighted = 0;
    for (U32 i = lane; i < output_size; i += 32) {
      const U32 value = output[i];
      sum += value;
      weighted += U64(output_size - i) * value;
    }
    for (U32 step = 16; step; step >>= 1) {
      sum += __shfl_down_sync(mask, sum, step);
      weighted += __shfl_down_sync(mask, weighted, step);
    }
    if (!lane) {
      const U32 checksum = U32((weighted + output_size) % 65521) << 16 |
                           U32((sum + 1) % 65521);
      if (checksum != wanted_checksum) error = kAdlerMismatch;
    }
  }
  if (!lane) {
    metadata[0] = error;
    metadata[1] = 0;
  }
}

__global__ void SmallSharedDecodeSpecialized(
    const U8* input, U32 input_size, U8* output, U32 output_size,
    U32 max_blocks, U32* metadata) {
  __shared__ decoder::DecodeTables tables;
  extern __shared__ U8 shared_output[];
  const SharedDecodeOutput scratch{
      U32(__cvta_generic_to_shared(shared_output))};
  DecodeOneFileShared(input, input_size, scratch, output_size, max_blocks,
                      metadata, tables);
  __syncwarp(0xffffffffu);
  for (U32 i = threadIdx.x; i < output_size; i += 32)
    output[i] = U8(scratch[i]);
}
