# Copyright (c) 2026 xangma
# SPDX-License-Identifier: MIT

"""Bounded GPU discovery and reconstruction of original raw Deflate streams.

The decoder parallelizes independent Deflate blocks and fixed-code tiles.
Fresh fixed summaries are requested only when the exact boundary chain needs
them. No host-generated block index, CPU inflation, or transcoding is used. The caller
compiles ``CUDA_SOURCE`` into a native JAX FFI library, sorts candidate metadata on
device, resolves backward roots on device, and verifies the original checksum.

Bit offsets use uint64; counts, output offsets, sizes and roots use uint32.
Output must be smaller than 2**31 bytes. Emission accepts the zlib header's
declared window size (raw Deflate uses 32768). A root is an earlier output index or
``0x80000000 | literal``. Emission never reads another block's root workspace.

Initialize discover's count to zero; allocate starts[max_candidates]. A returned
count greater than capacity is an error, and must not be passed to other kernels.
Optional discovery gates must remain immutable during each kernel launch; null
gates retain raw Deflate discovery without framing validation.
All sorted candidate columns must use the same permutation. Empty blocks are
supported; block capacity is independent of output length. Speculative errors
are local; only errors on the exact boundary chain invalidate the stream.

``discover`` maps threads to bytes. ``describe_candidates`` and ``emit_blocks``
use one-thread CTAs with shared tables. ``emit_blocks`` owns accepted blocks or
fixed segments of at most 1 MiB decoded output; ``emit_blocks_warp`` owns larger
ones, with lane zero parsing and all lanes expanding long matches. Launch both
emitters: their output and status writes are disjoint. ``fixed_summaries`` uses
32 threads per tile. ``emit_stored`` uses cooperative CTAs and follows both
emitters without clearing block statuses.
Counts and status gates stay on device. Grid stride supports bounded static
grids. ``select_chain`` uses one CTA/thread.
"""

STATUS_MESSAGES = {
    0: "success",
    1: "truncated Deflate payload",
    2: "reserved Deflate block type",
    3: "invalid Huffman tree",
    4: "invalid or reserved Huffman symbol",
    5: "invalid code-length repeat",
    6: "invalid backward distance or history",
    7: "decoded output exceeds the declared bound",
    9: "Deflate block workspace capacity exceeded",
    10: "dynamic block boundary was not discovered",
    11: "final decoded size differs from the declared size",
    12: "block endpoint or payload extent mismatch",
    13: "fixed-block tile summaries required",
}

FIXED_TILE_BYTES = 2048
FIXED_ENTRY_COUNT = 32
WARP_MIN_OUTPUT_BYTES = 1 << 20

KERNEL_NAMES = (
    "discover", "scan_prefixes", "validate_prefixes", "describe_candidates", "fixed_summaries", "select_chain",
    "emit_blocks", "emit_blocks_warp", "emit_stored",
)

CUDA_SOURCE = r'''
typedef unsigned char u8;
typedef unsigned short u16;
typedef unsigned int u32;
typedef unsigned long long u64;

const u32 WARP_MIN_OUTPUT_BYTES = 1u << 20;

struct BitReader {
    const u8* data;
    u64 bits;
    u64 pos;
    u32 error;
    u64 cache;
    u32 cached;

    __device__ __forceinline__ u32 peek(u32 n) {
        // Requests are at most 16 bits. cached counts only bits proven within
        // [pos, bits), so this fast path preserves the slow path's bounds.
        if (!error && cached && n <= cached)
            return u32(cache) & ((1u << n) - 1);
        if (error || pos > bits || n > bits - pos) {
            error = 1;
            return 0;
        }
        if (!n) return 0;

        const u64 remaining = bits - pos;
        const u32 skip = u32(pos & 7);
        const u64 byte = pos >> 3;
        const u32 full_window_bits = 64 - skip;
        // Full windows retain 57..64 bits. Near EOF, read only real bytes.
        const u32 bytes = remaining >= full_window_bits ? 8u :
            u32((remaining + skip + 7) / 8);
        u64 word = 0;
        // Byte assembly is safe for arbitrary pointer alignment. No rounded-
        // down address or speculative read outside the payload is used.
        #pragma unroll
        for (u32 j = 0; j < 8; ++j)
            if (j < bytes) word |= u64(data[byte + j]) << (8 * j);
        cache = word >> skip;
        cached = remaining < full_window_bits ? u32(remaining) :
            full_window_bits;
        return u32(cache) & ((1u << n) - 1);
    }

    __device__ __forceinline__ void drop(u32 n) {
        cache >>= n;
        cached -= n;
        pos += n;
    }

    __device__ __forceinline__ u32 take(u32 n) {
        u32 value = peek(n);
        if (!error) drop(n);
        return value;
    }

    __device__ __forceinline__ void seek(u64 next) {
        pos = next;
        cache = 0;
        cached = 0;
    }
};

__device__ u32 reverse_code(u32 code, u32 length) {
    code = ((code & 0x55555555u) << 1) | ((code >> 1) & 0x55555555u);
    code = ((code & 0x33333333u) << 2) | ((code >> 2) & 0x33333333u);
    code = ((code & 0x0f0f0f0fu) << 4) | ((code >> 4) & 0x0f0f0f0fu);
    code = ((code & 0x00ff00ffu) << 8) | ((code >> 8) & 0x00ff00ffu);
    code = (code << 16) | (code >> 16);
    return code >> (32 - length);
}

template<int N, int PrimaryBits> struct Huffman {
    u16 count[16];
    u16 symbols[N];
    // Packed length/symbol, zero for a longer code or an invalid prefix.
    u16 primary[1 << PrimaryBits];
    u32 maximum;
    u32 lookup;

    __device__ u32 build(const u8* lens, u32 n, u32 empty_ok,
                         u32 single_ok, u32 limit, bool make_lookup = true) {
        for (u32 i = 0; i < 16; ++i) count[i] = 0;
        maximum = 0;
        lookup = 0;
        if (n > N) return 3;
        for (u32 i = 0; i < n; ++i) {
            if (lens[i] > limit) return 3;
            ++count[lens[i]];
            if (lens[i] > maximum) maximum = lens[i];
        }
        if (!maximum) return empty_ok ? 0 : 3;
        int left = 1;
        for (u32 i = 1; i <= limit; ++i) {
            left = (left << 1) - count[i];
            if (left < 0) return 3;
        }
        if (left && !(single_ok && maximum == 1 && count[1] == 1))
            return 3;
        u16 offset[16];
        offset[1] = 0;
        for (u32 i = 1; i < 15; ++i)
            offset[i + 1] = offset[i] + count[i];
        for (u32 i = 0; i < n; ++i)
            if (lens[i]) symbols[offset[lens[i]]++] = u16(i);
        if (make_lookup) {
            for (u32 i = 0; i < (1u << PrimaryBits); ++i) primary[i] = 0;
            u32 code = 0, index = 0;
            for (u32 len = 1; len <= PrimaryBits; ++len) {
                for (u32 j = 0; j < count[len]; ++j) {
                    u32 reversed = reverse_code(code + j, len);
                    u16 entry = u16((len << 9) | symbols[index + j]);
                    for (u32 k = reversed; k < (1u << PrimaryBits);
                         k += 1u << len) primary[k] = entry;
                }
                index += count[len];
                code = (code + count[len]) << 1;
            }
            lookup = 1;
        }
        return 0;
    }

    __device__ __forceinline__ int decode(BitReader& r) const {
        if (lookup && !r.error && r.pos <= r.bits &&
            PrimaryBits <= r.bits - r.pos) {
            u16 entry = primary[r.peek(PrimaryBits)];
            if (entry) {
                r.drop(entry >> 9);
                return entry & 511;
            }
        }
        u32 code = 0, first = 0, index = 0;
        for (u32 len = 1; len <= maximum; ++len) {
            code |= r.take(1);
            if (r.error) return -1;
            if (code < first + count[len])
                return int(symbols[index + code - first]);
            index += count[len];
            first = (first + count[len]) << 1;
            code <<= 1;
        }
        r.error = 4;
        return -1;
    }
};

struct DecodeTables {
    Huffman<288, 9> ll;
    Huffman<32, 6> dd;
    Huffman<19, 7> cl;
};

__device__ u32 dynamic_tables(BitReader& r, Huffman<288, 9>& ll,
                              Huffman<32, 6>& dd,
                              Huffman<19, 7>& cl,
                              bool make_lookup = true) {
    u32 nl = r.take(5) + 257;
    u32 nd = r.take(5) + 1;
    u32 nc = r.take(4) + 4;
    if (r.error) return r.error;
    if (nl > 286) return 3;
    const u8 order[19] = {16,17,18,0,8,7,9,6,10,5,11,4,12,3,13,2,14,1,15};
    u8 clens[19] = {0};
    for (u32 i = 0; i < nc; ++i) clens[order[i]] = u8(r.take(3));
    if (r.error) return r.error;
    u32 err = cl.build(clens, 19, 0, 0, 7, make_lookup);
    if (err) return err;
    u8 lens[320];
    u32 i = 0;
    while (i < nl + nd) {
        int symbol = cl.decode(r);
        if (r.error) return r.error;
        if (symbol < 16) {
            lens[i++] = u8(symbol);
            continue;
        }
        u32 repeat, value;
        if (symbol == 16) {
            if (!i) return 5;
            repeat = r.take(2) + 3;
            value = lens[i - 1];
        } else if (symbol == 17) {
            repeat = r.take(3) + 3;
            value = 0;
        } else if (symbol == 18) {
            repeat = r.take(7) + 11;
            value = 0;
        } else {
            return 4;
        }
        if (r.error) return r.error;
        if (repeat > nl + nd - i) return 5;
        for (u32 j = 0; j < repeat; ++j) lens[i++] = u8(value);
    }
    if (!lens[256]) return 3;
    err = ll.build(lens, nl, 0, 1, 15, make_lookup);
    if (err) return err;
    // A literal-only dynamic block may declare no usable distance codes.
    return dd.build(lens + nl, nd, 1, 1, 15, make_lookup);
}

__device__ u32 fixed_tables(Huffman<288, 9>& ll, Huffman<32, 6>& dd) {
    u8 lens[288];
    for (u32 i = 0; i < 288; ++i)
        lens[i] = i < 144 ? 8 : i < 256 ? 9 : i < 280 ? 7 : 8;
    u32 err = ll.build(lens, 288, 0, 1, 15);
    for (u32 i = 0; i < 32; ++i) lens[i] = 5;
    return err ? err : dd.build(lens, 32, 0, 1, 15);
}

// Each scan thread loads one bounded bit window for eight possible starts.
// This changes only the cheap header prefilter; full header validation below
// still uses the existing canonical-tree validity checks.
__device__ __forceinline__ bool dynamic_prefix_window(
    u64 low, u32 high, u32 bit, u64 available_bits) {
    if (available_bits < bit + 17) return false;
    if (((low >> (bit + 1)) & 3u) != 2u) return false;
    if (((low >> (bit + 3)) & 31u) > 29u) return false;
    u32 nc = u32((low >> (bit + 13)) & 15u) + 4;
    if (bit + 17 + 3 * nc > available_bits) return false;
    // At most 19 three-bit lengths fit in one word after the header.
    u64 lengths = (low >> (bit + 17)) | (u64(high) << (47 - bit));
    u32 kraft = 0;
    for (u32 i = 0; i < nc; ++i) {
        u32 len = u32(lengths) & 7u;
        lengths >>= 3;
        if (len) kraft += 128u >> len;
        if (kraft > 128) return false;
    }
    return kraft == 128;
}

__device__ __noinline__ bool dynamic_valid(const u8* data, u32 bytes,
                                         u64 start) {
    BitReader r = {data, u64(bytes) * 8, start + 3, 0};
    Huffman<288, 9> ll;
    Huffman<32, 6> dd;
    Huffman<19, 7> cl;
    return dynamic_tables(r, ll, dd, cl, false) == 0;
}

struct BlockInfo {
    u64 end;
    u32 size;
    u32 final;
    u32 status;
    u32 external;
};

// A fixed token spans at most 32 bits. After crossing a tile boundary, its
// next token therefore starts within the next tile's first 32 positions.
// Summaries follow adjacent fixed blocks too, avoiding a serial partial-tile
// parse at every original block boundary. The first EOB is retained separately
// because a speculative tile entry does not know its current block's BFINAL.
struct FixedSummary {
    u64 end;
    u64 first_end;
    u32 size;
    u32 first_size;
    u32 flags;
    u32 status;
};
const u64 FIXED_SEGMENT = u64(1) << 63;
const u64 FIXED_FINAL = u64(1) << 62;
const u64 FIXED_POSITION = FIXED_FINAL - 1;

__device__ __forceinline__ int fixed_token(
    BitReader& r, const DecodeTables& tables, u32& size, u32& distance,
    u32 window_bytes = 32768) {
    int symbol = tables.ll.decode(r);
    size = 0;
    distance = 0;
    if (r.error || symbol == 256) return symbol;
    if (symbol < 256) { size = 1; return symbol; }
    if (symbol < 257 || symbol > 285) { r.error = 4; return symbol; }
    const u16 length_base[29] = {
        3,4,5,6,7,8,9,10,11,13,15,17,19,23,27,31,35,43,51,
        59,67,83,99,115,131,163,195,227,258};
    const u8 length_extra[29] = {
        0,0,0,0,0,0,0,0,1,1,1,1,2,2,2,2,3,3,3,3,4,4,4,4,5,5,5,5,0};
    const u16 distance_base[30] = {
        1,2,3,4,5,7,9,13,17,25,33,49,65,97,129,193,257,385,
        513,769,1025,1537,2049,3073,4097,6145,8193,12289,16385,24577};
    const u8 distance_extra[30] = {
        0,0,0,0,1,1,2,2,3,3,4,4,5,5,6,6,7,7,8,8,9,9,10,10,11,11,12,12,13,13};
    u32 index = u32(symbol - 257);
    size = length_base[index] + r.take(length_extra[index]);
    if (r.error) return symbol;
    int ds = tables.dd.decode(r);
    if (r.error) return symbol;
    if (ds < 0 || ds > 29) { r.error = 4; return symbol; }
    distance = distance_base[ds] + r.take(distance_extra[ds]);
    if (!r.error && (!distance || distance > window_bytes)) r.error = 6;
    return symbol;
}

__device__ __forceinline__ u32 extend_repeat_run(
    BitReader& r, const DecodeTables& tables, u32 length, u32 available,
    u32 window_bytes, u64 end) {
    // Consecutive distance-one matches share the same preceding seed. Keep
    // runs bounded, and leave every other token (including errors/EOB) for
    // the ordinary parser so its checks and failure precedence are preserved.
    while (length < 8192 && r.pos < end) {
        const BitReader saved = r;
        u32 next_length, next_distance;
        const int symbol = fixed_token(r, tables, next_length, next_distance,
                                       window_bytes);
        if (r.error || symbol == 256 || next_distance != 1 || r.pos > end ||
            next_length > available - length || next_length > 8192 - length) {
            r = saved;
            break;
        }
        length += next_length;
    }
    return length;
}

__device__ FixedSummary scan_fixed_region(
    BitReader& r, u64 tile_end, const DecodeTables& tables) {
    FixedSummary info = {r.pos, ~u64(0), 0, 0, 0, 0};
    u32 ended = 0, final = 2, kind = 0;
    while (r.pos < tile_end && !r.error) {
        u32 size, distance;
        int symbol = fixed_token(r, tables, size, distance);
        if (r.error) break;
        if (symbol != 256) { info.size += size; continue; }
        if (info.first_end == ~u64(0)) {
            info.first_end = r.pos;
            info.first_size = info.size;
        }
        ++ended;
        if (final == 1) { kind = 2; break; }
        // Leave a header to the exact chain when its type is not fixed or
        // when its start lies at/beyond the tile boundary.
        if (r.pos >= tile_end) { kind = 1; break; }
        u64 header = r.pos;
        u32 next_final = r.take(1), type = r.take(2);
        if (r.error) break;
        if (type != 1) { r.seek(header); kind = 1; break; }
        final = next_final;
    }
    info.end = r.pos;
    info.flags = kind | (final << 2) | (ended << 4);
    info.status = r.error;
    return info;
}

// Overlapping DEFLATE matches repeat the distance-byte seed preceding the
// match. Read only that seed so emission never depends on its own stores.
__device__ __forceinline__ void emit_match_roots(
    u32* roots, u32 prefix, u32 begin, u32 distance, u32 length) {
    u32 first = begin - distance;
    if (distance == 1) {
        u32 value = first < prefix ? first : roots[first];
        for (u32 j = 0; j < length; ++j) roots[begin + j] = value;
    } else if (length == 3 && distance >= 3) {
        // Independent seed reads can overlap before publishing this short match.
        const u32 a = first < prefix ? first : roots[first];
        const u32 b = first + 1 < prefix ? first + 1 : roots[first + 1];
        const u32 c = first + 2 < prefix ? first + 2 : roots[first + 2];
        roots[begin] = a;
        roots[begin + 1] = b;
        roots[begin + 2] = c;
    } else if (length >= 4 && distance >= length) {
        // All seeds precede this match. Read independent roots before stores
        // so their memory latency can overlap, including prefix straddles.
        u32 j = 0;
        for (; j + 4 <= length; j += 4) {
            const u32 source = first + j;
            const u32 a = source < prefix ? source : roots[source];
            const u32 b = source + 1 < prefix ? source + 1 : roots[source + 1];
            const u32 c = source + 2 < prefix ? source + 2 : roots[source + 2];
            const u32 d = source + 3 < prefix ? source + 3 : roots[source + 3];
            roots[begin + j] = a;
            roots[begin + j + 1] = b;
            roots[begin + j + 2] = c;
            roots[begin + j + 3] = d;
        }
        for (; j < length; ++j) {
            const u32 source = first + j;
            roots[begin + j] = source < prefix ? source : roots[source];
        }
    } else {
        u32 source = first;
        for (u32 j = 0; j < length; ++j) {
            roots[begin + j] = source < prefix ? source : roots[source];
            if (++source == begin) source = first;
        }
    }
}

__device__ BlockInfo emit_fixed_segment(
    const u8* data, u32 bytes, u64 start, u64 end, u32 limit, u32 prefix,
    u32* roots, DecodeTables& tables, u32 window_bytes, u32 final) {
    BitReader r = {data, u64(bytes) * 8, start, 0};
    BlockInfo result = {start, 0, 0, fixed_tables(tables.ll, tables.dd), 0};
    if (result.status) return result;
    while (r.pos < end && !r.error) {
        u32 size, distance;
        int symbol = fixed_token(r, tables, size, distance, window_bytes);
        if (r.error) break;
        if (symbol == 256) {
            if (r.pos == end) break;
            if (final) { r.error = 12; break; }
            final = r.take(1);
            u32 type = r.take(2);
            if (!r.error && type != 1) r.error = 12;
            continue;
        }
        if (size > limit - result.size) { r.error = 7; break; }
        if (!distance) roots[prefix + result.size] = 0x80000000u | u32(symbol);
        else {
            if (distance > prefix + result.size) { r.error = 6; break; }
            if (distance > result.size) result.external = 1;
            emit_match_roots(roots, prefix, prefix + result.size,
                             distance, size);
        }
        result.size += size;
    }
    result.end = r.pos;
    result.status = r.error;
    if (!result.status && r.pos != end) result.status = 12;
    return result;
}

__device__ __forceinline__ void emit_warp_match(
    u32* roots, u32 prefix, u32 begin, u32 distance, u32 length, u32 mask) {
    u32 lane = threadIdx.x, first = begin - distance;
    if (distance == 1) {
        u32 value = 0;
        if (!lane) value = first < prefix ? first : roots[first];
        value = __shfl_sync(mask, value, 0);
        // Align the middle to 16 bytes, retaining bounded scalar ends.
        u32 head = (4 - (begin & 3u)) & 3u;
        if (head > length) head = length;
        if (lane < head) roots[begin + lane] = value;
        const u32 vectors = (length - head) / 4;
        uint4* aligned = reinterpret_cast<uint4*>(roots + begin + head);
        const uint4 repeated = {value, value, value, value};
        for (u32 j = lane; j < vectors; j += 32) aligned[j] = repeated;
        const u32 tail = head + vectors * 4;
        if (lane < length - tail) roots[begin + tail + lane] = value;
    } else {
        // Compute modulo only once per lane, then advance within the seed.
        // Every source precedes begin, even when length exceeds distance.
        u32 source = first + lane % distance, stride = 32 % distance;
        for (u32 j = lane; j < length; j += 32) {
            roots[begin + j] = source < prefix ? source : roots[source];
            source += stride;
            if (source >= begin) source -= distance;
        }
    }
}

__device__ __noinline__ BlockInfo emit_warp_block(
    const u8* data, u32 bytes, u64 start, u64 end, u32 limit, u32 prefix,
    u32* roots, DecodeTables& tables, u32 window_bytes, bool segment,
    u32 final) {
    const u32 mask = blockDim.x >= 32 ? 0xffffffffu : (1u << blockDim.x) - 1u;
    const u32 lane = threadIdx.x;
    BitReader r = {data, u64(bytes) * 8, start, 0};
    BlockInfo result = {start, 0, 0, 0, 0};
    if (!lane) {
        u32 type = 1;
        if (!segment) {
            result.final = r.take(1);
            type = r.take(2);
        }
        if (!r.error) {
            if (type == 3) r.error = 2;
            else if (type == 0) r.error = 12;  // Handled by emit_stored.
            else r.error = type == 1 ? fixed_tables(tables.ll, tables.dd) :
                dynamic_tables(r, tables.ll, tables.dd, tables.cl);
        }
    }
    while (true) {
        u32 length = 0, distance = 0, begin = 0;
        if (!lane) {
            // Literals and short matches remain serial: only a long match
            // or termination makes the other lanes rendezvous with the parser.
            while (!r.error && (!segment || r.pos < end)) {
                u32 size, token_distance;
                int symbol = fixed_token(r, tables, size, token_distance,
                                         segment ? window_bytes : 32768);
                if (r.error) break;
                if (symbol == 256) {
                    if (!segment || r.pos == end) break;
                    if (final) { r.error = 12; break; }
                    final = r.take(1);
                    u32 type = r.take(2);
                    if (!r.error && type != 1) r.error = 12;
                    continue;
                }
                if (size > limit - result.size) { r.error = 7; break; }
                if (!token_distance) {
                    roots[prefix + result.size] = 0x80000000u | u32(symbol);
                } else {
                    if (token_distance > window_bytes ||
                        token_distance > prefix + result.size) {
                        r.error = 6; break;
                    }
                    if (token_distance > result.size) result.external = 1;
                    if (size >= 32 && blockDim.x >= 32) {
                        length = token_distance == 1 ? extend_repeat_run(
                            r, tables, size, limit - result.size, window_bytes,
                            segment ? end : r.bits) : size;
                        distance = token_distance;
                        begin = prefix + result.size;
                        result.size += length;
                        break;
                    }
                    emit_match_roots(roots, prefix, prefix + result.size,
                                     token_distance, size);
                }
                result.size += size;
            }
        }
        length = __shfl_sync(mask, length, 0);
        if (!length) break;  // Includes every parser error and EOB.
        distance = __shfl_sync(mask, distance, 0);
        begin = __shfl_sync(mask, begin, 0);
        // Shuffles alone do not order memory. Publish the parser's literal
        // and short-match roots before seed reads, and finish this match
        // before the next token can consume any of its roots.
        __syncwarp(mask);
        emit_warp_match(roots, prefix, begin, distance, length, mask);
        __syncwarp(mask);
    }
    if (!lane) {
        result.end = r.pos;
        result.status = r.error;
        if (!result.status && r.pos != end) result.status = 12;
    }
    return result;
}

__device__ __noinline__ BlockInfo parse_block(const u8* data, u32 bytes,
                                            u64 start, u32 limit,
                                            u32 prefix, u32* roots,
                                            DecodeTables& tables,
                                            u32 window_bytes = 32768,
                                            u32 fixed_budget = 0) {
    BitReader r = {data, u64(bytes) * 8, start, 0};
    BlockInfo result = {start, 0, 0, 0, 0};
    result.final = r.take(1);
    u32 type = r.take(2);
    if (r.error) { result.status = r.error; return result; }
    if (type == 3) { result.status = 2; return result; }
    u32 produced = 0;
    if (type == 0) {
        r.seek((r.pos + 7) & ~u64(7));
        u32 n = r.take(16), complement = r.take(16);
        if (r.error) { result.status = r.error; return result; }
        if ((n ^ complement) != 65535) {
            result.status = 4;
            return result;
        }
        if (n > limit) { result.status = 7; return result; }
        if (r.pos > r.bits || u64(n) * 8 > r.bits - r.pos) {
            result.status = 1;
            return result;
        }
        if (roots)
            for (u32 j = 0; j < n; ++j)
                roots[prefix + j] = 0x80000000u | data[(r.pos >> 3) + j];
        produced = n;
        r.seek(r.pos + u64(n) * 8);
    } else {
        Huffman<288, 9>& ll = tables.ll;
        Huffman<32, 6>& dd = tables.dd;
        u32 err = type == 1 ? fixed_tables(ll, dd) :
                             dynamic_tables(r, ll, dd, tables.cl);
        if (err) { result.status = err; return result; }
        const u16 length_base[29] = {
            3,4,5,6,7,8,9,10,11,13,15,17,19,23,27,31,35,43,51,
            59,67,83,99,115,131,163,195,227,258};
        const u8 length_extra[29] = {
            0,0,0,0,0,0,0,0,1,1,1,1,2,2,2,2,3,3,3,3,4,4,4,4,5,5,5,5,0};
        const u16 distance_base[30] = {
            1,2,3,4,5,7,9,13,17,25,33,49,65,97,129,193,257,385,
            513,769,1025,1537,2049,3073,4097,6145,8193,12289,16385,24577};
        const u8 distance_extra[30] = {
            0,0,0,0,1,1,2,2,3,3,4,4,5,5,6,6,7,7,8,8,9,9,10,10,11,11,12,12,13,13};
        while (!r.error) {
            if (type == 1 && fixed_budget && !roots &&
                (produced >= fixed_budget || r.pos - start >= u64(fixed_budget) * 8)) {
                r.error = 13;
                break;
            }
            int symbol = ll.decode(r);
            if (r.error) break;
            if (symbol == 256) break;
            if (symbol < 256) {
                if (produced >= limit) { r.error = 7; break; }
                if (roots) roots[prefix + produced] = 0x80000000u | u32(symbol);
                ++produced;
                continue;
            }
            if (symbol < 257 || symbol > 285) { r.error = 4; break; }
            u32 index = u32(symbol - 257);
            u32 length = length_base[index] + r.take(length_extra[index]);
            if (r.error) break;
            int distance_symbol = dd.decode(r);
            if (r.error) break;
            if (distance_symbol < 0 || distance_symbol > 29) {
                r.error = 4;
                break;
            }
            u32 distance = distance_base[distance_symbol] +
                           r.take(distance_extra[distance_symbol]);
            if (r.error) break;
            if (length > limit - produced) { r.error = 7; break; }
            if (!distance || distance > window_bytes) { r.error = 6; break; }
            // Speculative metadata cannot know preceding-block history.
            // Emission has the accepted chain's absolute output prefix.
            if (roots) {
                if (distance > prefix + produced) { r.error = 6; break; }
                if (distance > produced) result.external = 1;
                emit_match_roots(roots, prefix, prefix + produced,
                                 distance, length);
            }
            produced += length;
        }
    }
    result.end = r.pos;
    result.size = produced;
    result.status = r.error;
    return result;
}

extern "C" __global__ void discover(const u8* data, u32 input_bytes,
                                    u64* starts, u32* count,
                                    u32 max_candidates,
                                    const u32* enabled = nullptr) {
    if (enabled && !*enabled) return;
    u64 stride = u64(blockDim.x) * gridDim.x;
    for (u64 byte = u64(blockIdx.x) * blockDim.x + threadIdx.x;
         byte < input_bytes; byte += stride) {
        u64 low = 0;
        u32 high = 0;
        // An unaligned wide pointer cast would be undefined; assemble from
        // individual bounded byte reads, shared by all eight prefix tests.
        u64 available_bytes = u64(input_bytes) - byte;
        #pragma unroll
        for (u32 j = 0; j < 8; ++j)
            if (j < available_bytes) low |= u64(data[byte + j]) << (8 * j);
        #pragma unroll
        for (u32 j = 0; j < 3; ++j)
            if (j + 8 < available_bytes)
                high |= u32(data[byte + j + 8]) << (8 * j);
        // A stored block has aligned LEN/NLEN fields and a cheaply validated
        // endpoint. Seed a following fixed header there instead of forcing
        // the exact-chain selector to decode its entire token body serially.
        // These remain speculative starts: embedded LEN/NLEN words cannot
        // change the accepted chain, and full token/history checks still run.
        if (byte && available_bytes >= 5) {
            u32 n = u32(low) & 65535u;
            u32 complement = u32(low >> 16) & 65535u;
            if ((n ^ complement) == 65535u &&
                u64(n) + 4 < available_bytes) {
                u32 before = u32(data[byte - 1]) << 8;
                if (byte >= 2) before |= u32(data[byte - 2]);
                bool possible_stored = false;
                // Alignment after the three-bit header permits eight starts.
                for (u32 gap = 3; gap <= 10; ++gap)
                    if (byte * 8 >= gap &&
                        ((before >> (17 - gap)) & 3u) == 0)
                        possible_stored = true;
                u64 endpoint = byte + 4 + n;
                if (possible_stored && ((data[endpoint] >> 1) & 3u) == 1) {
                    u32 slot = atomicAdd(count, 1u);
                    if (slot < max_candidates) starts[slot] = endpoint * 8;
                }
            }
        }
        for (u32 bit = 0; bit < 8; ++bit) {
            u64 start = byte * 8 + bit;
            bool valid = start == 0 ||
                (dynamic_prefix_window(low, high, bit, available_bytes * 8) &&
                 dynamic_valid(data, input_bytes, start));
            if (valid) {
                u32 slot = atomicAdd(count, 1u);
                if (slot < max_candidates) starts[slot] = start;
            }
        }
    }
}

// Large inputs scan with a lightweight kernel. Only prefix matches allocate
// the Huffman validator's per-thread stack in the following kernel.
extern "C" __global__ void scan_prefixes(const u8* data, u32 input_bytes,
    u64* starts, u32* count, u32 max_candidates,
    u64* prefixes, u32* prefix_count, u32 prefix_capacity,
    const u32* frame_status = nullptr) {
    if (frame_status && *frame_status) return;
    u64 stride = u64(blockDim.x) * gridDim.x;
    for (u64 byte = u64(blockIdx.x) * blockDim.x + threadIdx.x;
         byte < input_bytes; byte += stride) {
        u64 low = 0;
        u32 high = 0;
        // An unaligned wide pointer cast would be undefined; assemble from
        // individual bounded byte reads, shared by all eight prefix tests.
        u64 available_bytes = u64(input_bytes) - byte;
        #pragma unroll
        for (u32 j = 0; j < 8; ++j)
            if (j < available_bytes) low |= u64(data[byte + j]) << (8 * j);
        #pragma unroll
        for (u32 j = 0; j < 3; ++j)
            if (j + 8 < available_bytes)
                high |= u32(data[byte + j + 8]) << (8 * j);
        // A stored block has aligned LEN/NLEN fields and a cheaply validated
        // endpoint. Seed a following fixed header there instead of forcing
        // the exact-chain selector to decode its entire token body serially.
        // These remain speculative starts: embedded LEN/NLEN words cannot
        // change the accepted chain, and full token/history checks still run.
        if (byte && available_bytes >= 5) {
            u32 n = u32(low) & 65535u;
            u32 complement = u32(low >> 16) & 65535u;
            if ((n ^ complement) == 65535u &&
                u64(n) + 4 < available_bytes) {
                u32 before = u32(data[byte - 1]) << 8;
                if (byte >= 2) before |= u32(data[byte - 2]);
                bool possible_stored = false;
                // Alignment after the three-bit header permits eight starts.
                for (u32 gap = 3; gap <= 10; ++gap)
                    if (byte * 8 >= gap &&
                        ((before >> (17 - gap)) & 3u) == 0)
                        possible_stored = true;
                u64 endpoint = byte + 4 + n;
                if (possible_stored && ((data[endpoint] >> 1) & 3u) == 1) {
                    u32 slot = atomicAdd(count, 1u);
                    if (slot < max_candidates) starts[slot] = endpoint * 8;
                }
            }
        }
        u32 mask = 0;
        for (u32 bit = 0; bit < 8; ++bit)
            if ((byte || bit) && dynamic_prefix_window(low, high, bit, available_bytes * 8))
                mask |= 1u << bit;
        if (!byte) {
            u32 slot = atomicAdd(count, 1u);
            if (slot < max_candidates) starts[slot] = 0;
        }
        if (mask) {
            u32 slot = atomicAdd(prefix_count, 1u);
            if (slot < prefix_capacity) prefixes[slot] = (byte << 8) | mask;
        }
    }
}

extern "C" __global__ void validate_prefixes(const u8* data, u32 input_bytes,
    u64* starts, u32* count, u32 max_candidates,
    const u64* prefixes, const u32* prefix_count, u32 prefix_capacity,
    const u32* frame_status = nullptr) {
    if (frame_status && *frame_status) return;
    if (*prefix_count > prefix_capacity) return;
    u64 stride = u64(blockDim.x) * gridDim.x;
    for (u64 i = u64(blockIdx.x) * blockDim.x + threadIdx.x;
         i < *prefix_count; i += stride) {
        u64 entry = prefixes[i];
        u64 byte = entry >> 8;
        u32 mask = u32(entry) & 255u;
        while (mask) {
            u32 bit = __ffs(mask) - 1;
            mask &= mask - 1;
            u64 start = byte * 8 + bit;
            if (dynamic_valid(data, input_bytes, start)) {
                u32 slot = atomicAdd(count, 1u);
                if (slot < max_candidates) starts[slot] = start;
            }
        }
    }
}
extern "C" __global__ void describe_candidates(
    const u8* data, u32 input_bytes, const u64* starts,
    u32 expected_bytes, u64* ends, u32* sizes, u32* finals, u32* status,
    const u32* count_device, const u32* discovery_status) {
    __shared__ DecodeTables tables;
    if (*discovery_status) return;
    if (threadIdx.x) return;
    const u32 count = *count_device;
    for (u64 i = blockIdx.x; i < count; i += gridDim.x) {
        BlockInfo info = parse_block(data, input_bytes, starts[i],
                                     expected_bytes, 0, (u32*)0, tables,
                                     32768, 65536);
        ends[i] = info.end;
        sizes[i] = info.size;
        finals[i] = info.final;
        status[i] = info.status;
    }
}

extern "C" __global__ void fixed_summaries(
    const u8* data, u32 input_bytes, u32 tile_bytes, u64* ends, u32* sizes,
    u64* first_ends, u32* first_sizes, u32* flags, u32* status,
    const u32* chain_status = nullptr) {
    __shared__ DecodeTables tables;
    // The request is immutable during this launch; every lane skips together.
    if (chain_status && *chain_status != 13) return;
    if (!tile_bytes || blockDim.x != 32) return;
    if (!threadIdx.x) fixed_tables(tables.ll, tables.dd);
    __syncthreads();
    u64 bits = u64(input_bytes) * 8, tile_bits = u64(tile_bytes) * 8;
    u64 tiles = (u64(input_bytes) + tile_bytes - 1) / tile_bytes;
    for (u64 tile = blockIdx.x; tile < tiles; tile += gridDim.x) {
        u64 entry = tile * tile_bits + threadIdx.x;
        u64 tile_end = (tile + 1) * tile_bits;
        if (tile_end > bits) tile_end = bits;
        BitReader r = {data, bits, entry, entry >= bits ? 1u : 0u};
        FixedSummary info = scan_fixed_region(r, tile_end, tables);
        u64 i = tile * 32 + threadIdx.x;
        ends[i] = info.end;
        sizes[i] = info.size;
        first_ends[i] = info.first_end;
        first_sizes[i] = info.first_size;
        flags[i] = info.flags;
        status[i] = info.status;
    }
}

extern "C" __global__ void select_chain(
    const u8* data, u32 input_bytes, const u64* starts, const u64* ends,
    const u32* sizes, const u32* finals, const u32* status, u32 count,
    u32 expected_bytes, u64* block_starts, u64* block_ends,
    u32* output_prefix, u32* block_sizes, u32* block_count,
    u32 max_blocks, u32* chain_status, u32 tile_bytes,
    const u64* summary_ends, const u32* summary_sizes,
    const u64* summary_first_ends, const u32* summary_first_sizes,
    const u32* summary_flags, const u32* summary_status,
    const u32* count_device = nullptr, const u32* discovery_status = nullptr,
    const u32* retry_status = nullptr) {
    __shared__ DecodeTables tables;
    if (blockIdx.x || threadIdx.x) return;
    // A skipped retry must preserve the first chain's count and error. The
    // retry request may alias chain_status, so consume it before clearing.
    if (retry_status && *retry_status != 13) return;
    *block_count = 0;
    *chain_status = discovery_status ? *discovery_status : 0;
    if (*chain_status) return;
    if (count_device) count = *count_device;
    if (expected_bytes >= 0x80000000u) { *chain_status = 7; return; }
    u64 next = 0;
    u32 output = 0, n = 0, real_blocks = 0;
    u32 candidate = 0;
    u64 candidate_start = count ? starts[0] : ~u64(0);
    bool fixed_ready = false;
    for (;;) {
        if (n >= max_blocks) { *chain_status = 9; return; }
        BitReader header = {data, u64(input_bytes) * 8, next, 0};
        u32 final = header.take(1), type = header.take(2);
        if (header.error) { *chain_status = header.error; return; }
        if (type == 3) { *chain_status = 2; return; }
        if (type == 1 && tile_bytes) {
            if (!fixed_ready) {
                u32 err = fixed_tables(tables.ll, tables.dd);
                if (err) { *chain_status = err; return; }
                fixed_ready = true;
            }
            u64 cursor = header.pos, tile_bits = u64(tile_bytes) * 8;
            for (;;) {
                if (cursor >= u64(input_bytes) * 8) {
                    *chain_status = 1; return;
                }
                u64 tile = cursor / tile_bits, offset = cursor % tile_bits;
                FixedSummary info;
                if (offset < 32) {
                    u64 index = tile * 32 + offset;
                    info.end = summary_ends[index];
                    info.size = summary_sizes[index];
                    info.first_end = summary_first_ends[index];
                    info.first_size = summary_first_sizes[index];
                    info.flags = summary_flags[index];
                    info.status = summary_status[index];
                } else {
                    // Only a fixed region after a stored/dynamic boundary
                    // needs a partial head. Adjacent fixed headers were
                    // already followed by the parallel tile summary.
                    u64 tile_end = (tile + 1) * tile_bits;
                    if (tile_end > u64(input_bytes) * 8)
                        tile_end = u64(input_bytes) * 8;
                    BitReader r = {data, u64(input_bytes) * 8, cursor, 0};
                    info = scan_fixed_region(r, tile_end, tables);
                }
                u32 kind = info.flags & 3u, ended = info.flags >> 4;
                if (final && info.first_end != ~u64(0)) {
                    // Any speculative parsing after the first EOB is
                    // irrelevant when the actual incoming block is final.
                    info.end = info.first_end;
                    info.size = info.first_size;
                    info.status = 0;
                    kind = 2;
                    ended = 1;
                }
                if (info.status) { *chain_status = info.status; return; }
                if (info.end <= cursor || info.end > u64(input_bytes) * 8) {
                    *chain_status = 12; return;
                }
                if (info.size > expected_bytes - output) {
                    *chain_status = 7; return;
                }
                if (n >= max_blocks || ended > max_blocks - real_blocks) {
                    *chain_status = 9; return;
                }
                block_starts[n] = cursor | FIXED_SEGMENT |
                                  (final ? FIXED_FINAL : u64(0));
                block_ends[n] = info.end;
                output_prefix[n] = output;
                block_sizes[n] = info.size;
                output += info.size;
                real_blocks += ended;
                ++n;
                cursor = info.end;
                if (kind == 2) {
                    if (output != expected_bytes) { *chain_status = 11; return; }
                    if ((cursor + 7) / 8 != input_bytes) {
                        *chain_status = 12; return;
                    }
                    *block_count = n;
                    return;
                }
                if (kind == 1) { next = cursor; break; }
                u32 carried_final = (info.flags >> 2) & 3u;
                if (carried_final < 2) final = carried_final;
            }
            continue;
        }
        // Exact block ends advance monotonically through sorted candidates.
        // Adjacent entries need one load; skipped speculative starts retain
        // logarithmic lookup and select the first duplicate at the boundary.
        if (candidate_start < next) {
            ++candidate;
            candidate_start = candidate < count ? starts[candidate] : ~u64(0);
            if (candidate_start < next) {
                u32 lo = candidate + 1, hi = count;
                while (lo < hi) {
                    u32 mid = lo + (hi - lo) / 2;
                    if (starts[mid] < next) lo = mid + 1;
                    else hi = mid;
                }
                candidate = lo;
                candidate_start = candidate < count ? starts[candidate] : ~u64(0);
            }
        }
        BlockInfo info;
        if (candidate < count && candidate_start == next) {
            info.end = ends[candidate];
            info.size = sizes[candidate];
            info.final = finals[candidate];
            info.status = status[candidate];
        } else {
            // Request fresh summaries only on an undiscovered exact fixed
            // boundary, never because a speculative candidate was expensive.
            if (type == 1) { *chain_status = 13; return; }
            if (type == 2) { *chain_status = 10; return; }
            info = parse_block(data, input_bytes, next,
                               expected_bytes - output, 0, (u32*)0, tables);
        }
        if (info.status) { *chain_status = info.status; return; }
        if (info.end <= next || info.end > u64(input_bytes) * 8) {
            *chain_status = 12;
            return;
        }
        if (info.size > expected_bytes - output) { *chain_status = 7; return; }
        if (real_blocks >= max_blocks) { *chain_status = 9; return; }
        block_starts[n] = next;
        block_ends[n] = info.end;
        output_prefix[n] = output;
        block_sizes[n] = info.size;
        output += info.size;
        ++real_blocks;
        ++n;
        next = info.end;
        if (info.final) {
            if (output != expected_bytes) { *chain_status = 11; return; }
            // Final byte padding is unconstrained; extra payload bytes are not.
            if ((next + 7) / 8 != input_bytes) { *chain_status = 12; return; }
            *block_count = n;
            return;
        }
    }
}

// Routing uses the accepted block's compressed extent, including any header.
// Medium highly repetitive blocks benefit from cooperative history expansion.
__device__ __forceinline__ bool use_warp_emission(u32 size, u64 start, u64 end) {
    if (size > WARP_MIN_OUTPUT_BYTES) return true;
    if (size <= 65536) return false;
    const u64 begin = (start & FIXED_SEGMENT) ? (start & FIXED_POSITION) : start;
    return end > begin && (end - begin + 7) / 8 <= size / 64;
}

extern "C" __global__ void emit_blocks(
    const u8* data, u32 input_bytes, const u64* block_starts,
    const u64* block_ends, const u32* output_prefix, const u32* block_sizes,
    u32 expected_bytes, u32* roots, u32* block_status,
    const u32* window_device, const u32* count_device,
    const u32* chain_status) {
    __shared__ DecodeTables tables;
    if (*chain_status) return;
    if (threadIdx.x) return;
    const u32 count = *count_device;
    const u32 window_bytes = *window_device;
    for (u64 i = blockIdx.x; i < count; i += gridDim.x) {
        u32 prefix = output_prefix[i], size = block_sizes[i];
        if (use_warp_emission(size, block_starts[i], block_ends[i])) continue;
        u32 err = 0, external = 0;
        if (!window_bytes || window_bytes > 32768) err = 6;
        else if (expected_bytes >= 0x80000000u ||
                 prefix > expected_bytes || size > expected_bytes - prefix)
            err = 7;
        else {
            u64 start = block_starts[i];
            if (start & FIXED_SEGMENT) {
                BlockInfo info = emit_fixed_segment(
                    data, input_bytes, start & FIXED_POSITION, block_ends[i],
                    size, prefix, roots, tables, window_bytes,
                    u32((start & FIXED_FINAL) != 0));
                err = info.status;
                external = info.external;
                if (!err && (info.end != block_ends[i] || info.size != size))
                    err = 12;
            } else {
                BitReader r = {data, u64(input_bytes) * 8, start, 0};
                r.take(1);
                u32 type = r.take(2);
                err = r.error;
                // Stored roots are filled by the following cooperative
                // kernel, keeping compressed CTAs at one resident warp.
                if (!err && type != 0) {
                    BlockInfo info = parse_block(
                        data, input_bytes, start, size, prefix, roots,
                        tables, window_bytes);
                    err = info.status;
                    external = info.external;
                    if (!err && (info.end != block_ends[i] || info.size != size))
                        err = 12;
                }
            }
        }
        block_status[i] = err;
        // Separate planes preserve every error when another block requires
        // refinement. Local history has already been flattened to literals.
        block_status[count + i] = external;
    }
}

extern "C" __global__ void emit_blocks_warp(
    const u8* data, u32 input_bytes, const u64* block_starts,
    const u64* block_ends, const u32* output_prefix, const u32* block_sizes,
    u32 expected_bytes, u32* roots, u32* block_status,
    const u32* window_device, const u32* count_device,
    const u32* chain_status) {
    __shared__ DecodeTables tables;
    if (*chain_status) return;
    if (threadIdx.x >= 32) return;
    const u32 count = *count_device;
    const u32 window_bytes = *window_device;
    for (u64 i = blockIdx.x; i < count; i += gridDim.x) {
        u32 prefix = output_prefix[i], size = block_sizes[i];
        if (!use_warp_emission(size, block_starts[i], block_ends[i])) continue;
        u32 err = 0, external = 0;
        if (!window_bytes || window_bytes > 32768) err = 6;
        else if (expected_bytes >= 0x80000000u ||
                 prefix > expected_bytes || size > expected_bytes - prefix)
            err = 7;
        else {
            u64 start = block_starts[i];
            if (start & FIXED_SEGMENT) {
                BlockInfo info = emit_warp_block(
                    data, input_bytes, start & FIXED_POSITION, block_ends[i],
                    size, prefix, roots, tables, window_bytes,
                    true, u32((start & FIXED_FINAL) != 0));
                if (!threadIdx.x) {
                    err = info.status;
                    external = info.external;
                    if (!err && (info.end != block_ends[i] || info.size != size))
                        err = 12;
                }
            } else {
                BitReader r = {data, u64(input_bytes) * 8, start, 0};
                r.take(1);
                u32 type = r.take(2);
                err = r.error;
                // Stored roots are filled by the following cooperative kernel.
                if (!err && type != 0) {
                    BlockInfo info = emit_warp_block(
                        data, input_bytes, start, block_ends[i], size, prefix,
                        roots, tables, window_bytes, false, 0);
                    if (!threadIdx.x) {
                        err = info.status;
                        external = info.external;
                        if (!err && (info.end != block_ends[i] || info.size != size))
                            err = 12;
                    }
                }
            }
        }
        if (!threadIdx.x) {
            block_status[i] = err;
            // Separate planes preserve every error when another block requires
            // refinement. Local history has already been flattened to literals.
            block_status[count + i] = external;
        }
    }
}

extern "C" __global__ void emit_stored(
    const u8* data, u32 input_bytes, const u64* block_starts,
    const u64* block_ends, const u32* output_prefix, const u32* block_sizes,
    u32 expected_bytes, u32* roots, u32* block_status,
    const u32* window_device, const u32* count_device,
    const u32* chain_status) {
    __shared__ u64 stored_payload;
    __shared__ u32 stored, err;
    if (*chain_status) return;
    const u32 count = *count_device;
    const u32 window_bytes = threadIdx.x ? 0 : *window_device;
    for (u64 i = blockIdx.x; i < count; i += gridDim.x) {
        u32 prefix = output_prefix[i], size = block_sizes[i];
        if (!threadIdx.x) {
            stored = 0;
            err = 0;
            u64 start = block_starts[i];
            if (!(start & FIXED_SEGMENT)) {
                BitReader r = {data, u64(input_bytes) * 8, start, 0};
                r.take(1);
                u32 type = r.take(2);
                if (!r.error && type == 0) {
                    stored = 1;
                    if (!window_bytes || window_bytes > 32768) err = 6;
                    else if (expected_bytes >= 0x80000000u ||
                             prefix > expected_bytes ||
                             size > expected_bytes - prefix) err = 7;
                    else {
                        r.seek((r.pos + 7) & ~u64(7));
                        u32 n = r.take(16), complement = r.take(16);
                        err = r.error;
                        if (!err && (n ^ complement) != 65535) err = 4;
                        if (!err && (r.pos > r.bits ||
                            u64(n) * 8 > r.bits - r.pos)) err = 1;
                        if (!err && (n != size ||
                            r.pos + u64(n) * 8 != block_ends[i])) err = 12;
                        if (!err) stored_payload = r.pos >> 3;
                    }
                    block_status[i] = err;
                }
            }
        }
        __syncthreads();
        if (stored && !err)
            for (u32 j = threadIdx.x; j < size; j += blockDim.x)
                roots[prefix + j] = 0x80000000u | data[stored_payload + j];
        // A grid-stride CTA must finish using the previous shared metadata.
        __syncthreads();
    }
}
'''
