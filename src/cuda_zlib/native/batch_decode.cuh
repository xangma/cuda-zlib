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
        }
        U32 size, token_distance;
        const int symbol = decoder::fixed_token(
            reader, tables, size, token_distance, window);
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
          length = token_distance == 1 ? decoder::extend_repeat_run(
              reader, tables, size, output_size - produced, window,
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
