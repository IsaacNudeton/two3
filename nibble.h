/*
 * nibble.h — Nibble-Level Input for {2,3} Architecture
 *
 * The question: what IS the input unit?
 *
 * Bytes are an arbitrary boundary. 8 bits grouped by hardware convention.
 * The {2,3} computational unit is 2 bits. The natural input unit should
 * match the computational unit or be a clean multiple of it.
 *
 * Nibble = 4 bits = exactly 2 ternary weights.
 * vocab = 16. Embedding table = 16 × dim.
 * Sequence 2× longer, but:
 *   - Structural similarity preserved: 0xAB and 0xAC share nibble 0xA
 *   - Embedding table fits in L1 cache (64KB at dim=1024)
 *   - No tokenizer. No BPE. No vocabulary file. Just split bytes.
 *
 * From XYZT "position is meaning": the nibble position within the byte
 * (high/low) carries structural information. We encode this as a
 * 1-bit position signal added to the embedding, not as separate vocabs.
 *
 * Isaac & CC — March 2026
 */

#ifndef NIBBLE_H
#define NIBBLE_H

#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define NIBBLE_VOCAB 16

/* ═══════════════════════════════════════════════════════
 * Nibble conversion — bytes to nibble sequences
 *
 * Each byte becomes 2 nibbles: high nibble first, low nibble second.
 * "Hello" (5 bytes) → 10 nibbles.
 *
 * This preserves:
 *   - Byte boundaries (every 2 nibbles = 1 byte)
 *   - Structural similarity (0xAB, 0xAC share high nibble 0xA)
 *   - Hex-natural (nibble = hex digit)
 * ═══════════════════════════════════════════════════════ */

/* Convert byte sequence to nibble sequence.
 * nibbles must be allocated to 2 * n_bytes. */
static void bytes_to_nibbles(
    const uint8_t *bytes, int n_bytes,
    uint8_t *nibbles  /* [2 * n_bytes] output, each 0-15 */
) {
    for (int i = 0; i < n_bytes; i++) {
        nibbles[2 * i]     = (bytes[i] >> 4) & 0x0F;  /* high nibble */
        nibbles[2 * i + 1] = bytes[i] & 0x0F;          /* low nibble */
    }
}

/* Convert nibble sequence back to bytes.
 * n_nibbles must be even. */
static void nibbles_to_bytes(
    const uint8_t *nibbles, int n_nibbles,
    uint8_t *bytes  /* [n_nibbles / 2] output */
) {
    for (int i = 0; i < n_nibbles / 2; i++) {
        bytes[i] = (nibbles[2 * i] << 4) | nibbles[2 * i + 1];
    }
}

/* ═══════════════════════════════════════════════════════
 * Nibble embedding — 16 × dim + positional nibble signal
 *
 * Two design choices:
 *
 * A) Flat: just 16 embeddings, RoPE handles position.
 *    Pro: simplest. Con: high/low nibble have same embedding
 *    even though they carry different magnitude (high = ×16).
 *
 * B) Position-aware: embed[nibble] + pos_bias[high_or_low].
 *    Pro: captures that high nibble contributes more to byte value.
 *    Con: 2 extra dim-vectors to learn.
 *
 * We go with B. The position bias is 2 × dim (just 2 vectors),
 * learned alongside the embedding. This is "position is meaning"
 * applied at the sub-byte level.
 * ═══════════════════════════════════════════════════════ */

typedef struct {
    float *embed;       /* [16 × dim] nibble embeddings */
    float *pos_bias;    /* [2 × dim]  high/low nibble bias */
    int dim;
} NibbleEmbed;

static void nibble_embed_init(NibbleEmbed *ne, int dim) {
    ne->dim = dim;
    ne->embed = (float*)malloc(NIBBLE_VOCAB * dim * sizeof(float));
    ne->pos_bias = (float*)malloc(2 * dim * sizeof(float));

    float scale = 1.0f / sqrtf((float)dim);
    for (int i = 0; i < NIBBLE_VOCAB * dim; i++)
        ne->embed[i] = scale * (2.0f * (float)rand() / RAND_MAX - 1.0f);

    /* Position bias starts small — let it learn how much
     * high vs low nibble matters */
    for (int i = 0; i < 2 * dim; i++)
        ne->pos_bias[i] = 0.01f * (2.0f * (float)rand() / RAND_MAX - 1.0f);
}

static void nibble_embed_free(NibbleEmbed *ne) {
    free(ne->embed);
    free(ne->pos_bias);
}

/* Look up nibble embedding + position bias.
 * nibble_val: 0-15
 * nibble_pos: 0 = high nibble, 1 = low nibble (within byte) */
static void nibble_embed_lookup(
    float *out,                 /* [dim] output */
    const NibbleEmbed *ne,
    int nibble_val,             /* 0-15 */
    int nibble_pos              /* 0=high, 1=low */
) {
    int dim = ne->dim;
    const float *emb = ne->embed + nibble_val * dim;
    const float *bias = ne->pos_bias + nibble_pos * dim;

    for (int d = 0; d < dim; d++)
        out[d] = emb[d] + bias[d];
}

/* Compute logits against nibble embedding table (weight-tied).
 * Returns 16-way logits for next nibble prediction. */
static void nibble_logits(
    float *logits,              /* [16] output */
    const float *hidden,        /* [dim] */
    const NibbleEmbed *ne
) {
    int dim = ne->dim;
    for (int n = 0; n < NIBBLE_VOCAB; n++) {
        float sum = 0;
        for (int d = 0; d < dim; d++)
            sum += hidden[d] * ne->embed[n * dim + d];
        logits[n] = sum;
    }
}

/* ═══════════════════════════════════════════════════════
 * Nibble-level cross-entropy loss
 *
 * Same as byte-level but 16-way instead of 256-way.
 * softmax over 16 is cheaper. exp(16 values) vs exp(256).
 * ═══════════════════════════════════════════════════════ */

#define NIBBLE_LOGIT_CLIP 30.0f

static float nibble_cross_entropy(
    const float *logits,    /* [16] */
    int target,             /* 0-15 */
    float *d_logits         /* [16] output gradient */
) {
    float clipped[16];
    for (int i = 0; i < 16; i++) {
        clipped[i] = logits[i];
        if (clipped[i] > NIBBLE_LOGIT_CLIP) clipped[i] = NIBBLE_LOGIT_CLIP;
        if (clipped[i] < -NIBBLE_LOGIT_CLIP) clipped[i] = -NIBBLE_LOGIT_CLIP;
    }

    float max_l = clipped[0];
    for (int i = 1; i < 16; i++)
        if (clipped[i] > max_l) max_l = clipped[i];

    float sum_exp = 0;
    float probs[16];
    for (int i = 0; i < 16; i++) {
        probs[i] = expf(clipped[i] - max_l);
        sum_exp += probs[i];
    }
    for (int i = 0; i < 16; i++)
        probs[i] /= sum_exp;

    float loss = -logf(probs[target] + 1e-10f);

    for (int i = 0; i < 16; i++)
        d_logits[i] = probs[i] - (i == target ? 1.0f : 0.0f);

    return loss;
}

/* ═══════════════════════════════════════════════════════
 * Nibble dataset — wraps byte dataset with nibble conversion
 * ═══════════════════════════════════════════════════════ */

typedef struct {
    uint8_t *nibble_data;   /* all data as nibbles */
    size_t total_nibbles;
    int *chunk_offsets;     /* offsets into nibble_data */
    int n_chunks;
    int chunk_cap;
    int seq_len;            /* nibble sequence length (2× byte seq_len) */
} NibbleDataset;

static void nibble_dataset_init(NibbleDataset *nds, int byte_seq_len) {
    nds->seq_len = byte_seq_len * 2;  /* 2 nibbles per byte */
    nds->total_nibbles = 0;
    nds->nibble_data = NULL;
    nds->n_chunks = 0;
    nds->chunk_cap = 1024;
    nds->chunk_offsets = (int*)malloc(nds->chunk_cap * sizeof(int));
}

static void nibble_dataset_free(NibbleDataset *nds) {
    free(nds->nibble_data);
    free(nds->chunk_offsets);
}

/* Load a text file, convert to nibbles, chunk. */
static int nibble_dataset_load_file(NibbleDataset *nds, const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) return -1;

    fseek(f, 0, SEEK_END);
    long fsize = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (fsize <= 0) { fclose(f); return 0; }

    /* Read bytes */
    uint8_t *bytes = (uint8_t*)malloc(fsize);
    size_t read_bytes = fread(bytes, 1, fsize, f);
    fclose(f);

    /* Convert to nibbles */
    size_t n_nibbles = read_bytes * 2;
    size_t old_total = nds->total_nibbles;
    nds->total_nibbles += n_nibbles;
    nds->nibble_data = (uint8_t*)realloc(nds->nibble_data, nds->total_nibbles);

    bytes_to_nibbles(bytes, (int)read_bytes,
                     nds->nibble_data + old_total);
    free(bytes);

    /* Create chunks with 50% overlap (in nibble space) */
    int stride = nds->seq_len / 2;
    if (stride < 2) stride = 2;  /* must be even to preserve byte alignment */

    for (size_t off = old_total;
         off + nds->seq_len <= nds->total_nibbles;
         off += stride) {
        if (nds->n_chunks >= nds->chunk_cap) {
            nds->chunk_cap *= 2;
            nds->chunk_offsets = (int*)realloc(nds->chunk_offsets,
                                                nds->chunk_cap * sizeof(int));
        }
        nds->chunk_offsets[nds->n_chunks++] = (int)off;
    }

    printf("[nibble] loaded %s: %ld bytes → %zu nibbles, %d chunks (seq=%d)\n",
           path, (long)read_bytes, n_nibbles, nds->n_chunks, nds->seq_len);
    return (int)read_bytes;
}

static void nibble_dataset_shuffle(NibbleDataset *nds, unsigned int seed) {
    srand(seed);
    for (int i = nds->n_chunks - 1; i > 0; i--) {
        int j = rand() % (i + 1);
        int tmp = nds->chunk_offsets[i];
        nds->chunk_offsets[i] = nds->chunk_offsets[j];
        nds->chunk_offsets[j] = tmp;
    }
}

static uint8_t* nibble_dataset_get(const NibbleDataset *nds, int idx) {
    return nds->nibble_data + nds->chunk_offsets[idx];
}

/* ═══════════════════════════════════════════════════════
 * Edge encoding — XOR between consecutive bytes
 *
 * From XYZT "position is meaning" / positional edge maps:
 * don't process the signal, process where the signal CHANGES.
 *
 * edge[i] = bytes[i] XOR bytes[i-1]  (edge[0] = bytes[0])
 *
 * Steady-state (repeating bytes) → 0x00 → all substrate nibbles.
 * The ternary kernel naturally skips zeros. Sparse input = faster.
 *
 * This is optional preprocessing. Can be composed with nibble input:
 * bytes → edge → nibbles → model.
 * ═══════════════════════════════════════════════════════ */

static void edge_encode(
    const uint8_t *bytes, int n_bytes,
    uint8_t *edges  /* [n_bytes] output */
) {
    edges[0] = bytes[0];  /* first byte has no predecessor */
    for (int i = 1; i < n_bytes; i++)
        edges[i] = bytes[i] ^ bytes[i - 1];
}

/* Reconstruct bytes from edges (inverse) */
static void edge_decode(
    const uint8_t *edges, int n_bytes,
    uint8_t *bytes  /* [n_bytes] output */
) {
    bytes[0] = edges[0];
    for (int i = 1; i < n_bytes; i++)
        bytes[i] = edges[i] ^ bytes[i - 1];
}

/* ═══════════════════════════════════════════════════════
 * Combined pipeline: bytes → edge → nibbles
 *
 * This is the full "position is meaning" + "match the
 * computational unit" pipeline.
 *
 * Properties:
 * - Repeating bytes produce 0x0 0x0 nibbles → substrate
 * - Character class changes produce large XOR → strong signal
 * - vocab = 16 (edges are still bytes, nibbles split them)
 * - Invertible: nibbles → bytes → edge_decode → original
 * ═══════════════════════════════════════════════════════ */

static void edge_nibble_encode(
    const uint8_t *bytes, int n_bytes,
    uint8_t *nibbles  /* [2 * n_bytes] output */
) {
    uint8_t *edges = (uint8_t*)malloc(n_bytes);
    edge_encode(bytes, n_bytes, edges);
    bytes_to_nibbles(edges, n_bytes, nibbles);
    free(edges);
}

static void edge_nibble_decode(
    const uint8_t *nibbles, int n_nibbles,
    uint8_t *bytes  /* [n_nibbles / 2] output */
) {
    int n_bytes = n_nibbles / 2;
    uint8_t *edges = (uint8_t*)malloc(n_bytes);
    nibbles_to_bytes(nibbles, n_nibbles, edges);
    edge_decode(edges, n_bytes, bytes);
    free(edges);
}

#endif /* NIBBLE_H */
