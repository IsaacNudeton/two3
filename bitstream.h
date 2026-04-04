/*
 * bitstream.h — BitStream primitives for onetwo_encode
 *
 * Extracted from xyzt.c for standalone use in two3.
 * Fixed 4096-bit (512 byte) fingerprint support.
 */

#ifndef BITSTREAM_H
#define BITSTREAM_H

#include <stdint.h>
#include <string.h>

#define BS_WORDS   128
#define BS_MAXBITS (BS_WORDS * 64)

typedef struct { uint64_t w[BS_WORDS]; int len; } BitStream;

static inline void bs_init(BitStream *b) { memset(b, 0, sizeof(*b)); }

static inline void bs_push(BitStream *b, int bit) {
    if (b->len >= BS_MAXBITS) return;
    int idx = b->len/64, off = b->len%64;
    if (bit) b->w[idx] |= (1ULL << off);
    else     b->w[idx] &= ~(1ULL << off);
    b->len++;
}

static inline int bs_get(const BitStream *b, int i) {
    if (i < 0 || i >= b->len) return 0;
    return (b->w[i/64] >> (i%64)) & 1;
}

static inline void bs_set(BitStream *b, int pos, int bit) {
    if (pos < 0 || pos >= BS_MAXBITS) return;
    int idx = pos / 64, off = pos % 64;
    if (bit) b->w[idx] |= (1ULL << off);
    else     b->w[idx] &= ~(1ULL << off);
    if (pos >= b->len) b->len = pos + 1;
}

static int bs_popcount(const BitStream *b) {
    int ones = 0, full = b->len/64;
    for (int i = 0; i < full; i++) ones += __builtin_popcountll(b->w[i]);
    int rem = b->len%64;
    if (rem) ones += __builtin_popcountll(b->w[full] & ((1ULL << rem) - 1));
    return ones;
}

/* FNV-1a hash */
static uint32_t hash32(const uint8_t *data, int len) {
    uint32_t h = 2166136261u;
    for (int i = 0; i < len; i++) { h ^= data[i]; h *= 16777619u; }
    return h;
}

/* Bloom filter: k deterministic positions per element */
static void bloom_set(BitStream *bs, int base_off, int bloom_len,
                      const uint8_t *data, int data_len, int n_hashes) {
    uint32_t h1 = hash32(data, data_len);
    uint32_t h2 = (h1 >> 16) | (h1 << 16);
    h2 |= 1;
    for (int i = 0; i < n_hashes; i++) {
        int pos = (h1 + i * h2) % bloom_len;
        bs_set(bs, base_off + pos, 1);
    }
}

/* Convert BitStream to float array for model input.
 * Each bit → float (0.0 or 1.0). Output: [4096] floats. */
static void bs_to_float(const BitStream *b, float *out, int nbits) {
    for (int i = 0; i < nbits; i++)
        out[i] = (float)bs_get(b, i);
}

#endif /* BITSTREAM_H */
