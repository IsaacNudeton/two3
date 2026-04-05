/*
 * ibc.h — Intelligent Byte Code
 *
 * Native format for two3. Maps bytes to int8 vectors the ternary
 * matmul consumes directly. No float embedding. No quantization loss.
 *
 * Like FBC maps test patterns to pin vectors for burn-in controllers:
 * - Lossless: byte ↔ vector is invertible
 * - Structured: similar bytes → similar vectors
 * - Native: int8 in, ternary multiply, int8 out
 *
 * The codebook IS the compiler. The inverse lookup IS the decoder.
 * The model is the DUT. It predicts the next vector, not the next byte.
 *
 * Isaac & CC — April 2026
 */

#ifndef IBC_H
#define IBC_H

#include <stdint.h>
#include <string.h>

/* Codebook width — must match model input dimension.
 * Each byte maps to IBC_WIDTH int8 values. */
#ifndef IBC_WIDTH
#define IBC_WIDTH 128
#endif

/* ═══════════════════════════════════════════════════════
 * Codebook: byte → int8[IBC_WIDTH]
 *
 * Structure: each of the byte's 8 bits controls IBC_WIDTH/8
 * positions in the output vector. Bit set → +127, bit clear → -127.
 * Two bytes differing by 1 bit → vectors differ in IBC_WIDTH/8
 * positions. Adjacent bytes share most of their structure.
 *
 * This is the pin map. The byte is the device address.
 * The vector is the pin state pattern for that address.
 * ═══════════════════════════════════════════════════════ */

typedef struct {
    int8_t vectors[256][IBC_WIDTH];  /* the codebook */
    int    width;                     /* == IBC_WIDTH */
} IBCCodebook;

/* Deterministic hash for codebook generation */
static uint32_t ibc_hash(uint32_t x) {
    x = ((x >> 16) ^ x) * 0x45d9f3b;
    x = ((x >> 16) ^ x) * 0x45d9f3b;
    x = (x >> 16) ^ x;
    return x;
}

/* Build the codebook. Deterministic — same codebook every time.
 * Each byte's 8 bits seed the vector structure:
 * - Bit k controls positions [k*stride .. (k+1)*stride)
 * - Within each stripe, a hash determines the pattern
 * - Bit set → values lean positive, bit clear → lean negative
 * This guarantees: similar bytes → similar vectors. */
static void ibc_codebook_init(IBCCodebook *cb) {
    cb->width = IBC_WIDTH;
    int stripe = IBC_WIDTH / 8;  /* positions per bit */

    for (int byte_val = 0; byte_val < 256; byte_val++) {
        int8_t *v = cb->vectors[byte_val];

        for (int bit = 0; bit < 8; bit++) {
            int bit_set = (byte_val >> bit) & 1;
            int base = bit * stripe;

            for (int j = 0; j < stripe; j++) {
                /* Hash determines the magnitude pattern within the stripe */
                uint32_t h = ibc_hash((uint32_t)(byte_val * 1000 + bit * 100 + j));
                int8_t mag = (int8_t)(32 + (h % 96));  /* range [32, 127] */

                /* Bit set → positive, bit clear → negative */
                v[base + j] = bit_set ? mag : -mag;
            }
        }
    }
}

/* Encode: byte → int8 vector. Just a lookup. */
static const int8_t* ibc_encode(const IBCCodebook *cb, uint8_t byte_val) {
    return cb->vectors[byte_val];
}

/* Decode: int8 vector → byte. Find nearest codebook entry.
 * Uses dot product (native to ternary matmul) as similarity.
 * O(256 × IBC_WIDTH) — fast for 256 entries. */
static uint8_t ibc_decode(const IBCCodebook *cb, const int8_t *vector) {
    int best = 0;
    int32_t best_sim = -2147483647;

    for (int b = 0; b < 256; b++) {
        int32_t sim = 0;
        for (int k = 0; k < cb->width; k++)
            sim += (int32_t)vector[k] * (int32_t)cb->vectors[b][k];
        if (sim > best_sim) {
            best_sim = sim;
            best = b;
        }
    }
    return (uint8_t)best;
}

/* Encode a sequence of bytes into int8 vectors.
 * Output: [seq_len × IBC_WIDTH] int8 array. */
static void ibc_encode_sequence(
    const IBCCodebook *cb,
    const uint8_t *bytes, int seq_len,
    int8_t *out  /* [seq_len × IBC_WIDTH] */
) {
    for (int t = 0; t < seq_len; t++)
        memcpy(out + t * cb->width, cb->vectors[bytes[t]], cb->width * sizeof(int8_t));
}

/* Convert int8 vector to float for model input.
 * Scale to [-1, 1] range by dividing by 127. */
static void ibc_to_float(const int8_t *vector, float *out, int width) {
    for (int i = 0; i < width; i++)
        out[i] = (float)vector[i] / 127.0f;
}

/* Convert float model output back to int8 for decoding.
 * Clamp to [-127, 127] and round. */
static void ibc_from_float(const float *vector, int8_t *out, int width) {
    for (int i = 0; i < width; i++) {
        float v = vector[i] * 127.0f;
        if (v > 127.0f) v = 127.0f;
        if (v < -127.0f) v = -127.0f;
        out[i] = (int8_t)(v > 0 ? v + 0.5f : v - 0.5f);
    }
}

/* Print codebook statistics */
static void ibc_codebook_info(const IBCCodebook *cb) {
    /* Check invertibility: encode then decode every byte */
    int correct = 0;
    for (int b = 0; b < 256; b++) {
        uint8_t decoded = ibc_decode(cb, cb->vectors[b]);
        if (decoded == (uint8_t)b) correct++;
    }

    /* Check structure: average similarity between adjacent bytes */
    int64_t adj_sim = 0, rand_sim = 0;
    for (int b = 0; b < 255; b++) {
        int32_t sim = 0;
        for (int k = 0; k < cb->width; k++)
            sim += (int32_t)cb->vectors[b][k] * (int32_t)cb->vectors[b+1][k];
        adj_sim += sim;
    }
    for (int i = 0; i < 255; i++) {
        int b1 = i, b2 = (i * 97 + 31) % 256;  /* pseudo-random pairs */
        int32_t sim = 0;
        for (int k = 0; k < cb->width; k++)
            sim += (int32_t)cb->vectors[b1][k] * (int32_t)cb->vectors[b2][k];
        rand_sim += sim;
    }

    printf("[ibc] codebook: %d entries × %d width\n", 256, cb->width);
    printf("[ibc] invertibility: %d/256 correct (%.1f%%)\n", correct, 100.0f * correct / 256);
    printf("[ibc] avg adjacent similarity: %.0f\n", (double)adj_sim / 255);
    printf("[ibc] avg random similarity:   %.0f\n", (double)rand_sim / 255);
}

#endif /* IBC_H */
