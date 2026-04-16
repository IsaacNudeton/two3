/*
 * model.h — Complete {2,3} Model
 *
 * Layer 2: wraps all components into a full model with REAL attention.
 *
 * Architecture:
 *   byte_in (0-255) → embedding (256 × dim, float) → gain_norm
 *   → [attention block + MoE FFN block] × N layers
 *   → final gain_norm → unembed → logits (256-way)
 *
 * No tokenizer. No BPE. No vocabulary beyond 256 bytes.
 * Sequences are longer but ternary matmul at 408 GOPS handles it.
 * Embedding table is 512KB, not 64MB.
 *
 * THIS VERSION: real causal attention, real ternary projections,
 * full-sequence forward pass. No stubs. No simulations.
 *
 * Isaac & CC — March 2026
 */

#ifndef MODEL_H
#define MODEL_H

#include "two3.h"
#include "gain.h"
#include "rope.h"
#include "activation.h"
#include "ffn.h"
#ifdef TWO3_BINARY
#include "binary.h"
#endif
#if defined(TWO3_FP_EMBED) && defined(TWO3_LEX_EMBED)
#error "TWO3_FP_EMBED and TWO3_LEX_EMBED are mutually exclusive"
#endif
#ifdef TWO3_IBC
#include "ibc.h"
#endif

#include <string.h>
#include <float.h>

/* ═══════════════════════════════════════════════════════
 * Model config
 * ═══════════════════════════════════════════════════════ */

#define BYTE_VOCAB 256  /* raw bytes, not tokens */

/* Inference early exit (model_forward_sequence_cpu):
 * Reservoir depletion gate — if mean(C - R) across both gain states
 * (attn + ffn) is below threshold, the layer's reservoirs barely fired.
 * Signal is free (already computed in gain_forward), adaptive (tracks
 * training), and has stability guarantees from Jury conditions.
 *
 * Old heuristic (logit margin) removed — it was a bolted-on heuristic
 * with no connection to the gain kernel dynamics. */
#ifndef TWO3_EXIT_DEPLETION_THRESH
#define TWO3_EXIT_DEPLETION_THRESH (0.08f)
#endif

/* model_forward_sequence_cpu(..., forward_flags) — last parameter */
#define MODEL_FWD_FLAGS_DEFAULT     0
#define MODEL_FWD_FORCE_FULL_DEPTH  1  /* ignore TWO3_EARLY_EXIT (calibration / parity) */

typedef struct {
    int dim;            /* hidden dimension (1024) */
    int n_heads;        /* query heads (8) */
    int n_kv_heads;     /* KV heads for GQA (4) */
    int head_dim;       /* dim / n_heads (128) */
    int intermediate;   /* MoE expert intermediate (2048) */
    int n_layers;       /* number of transformer blocks (12) */
    int max_seq;        /* max sequence length in bytes (4096) */
    float rope_theta;   /* RoPE base (1000000.0) */
} ModelConfig;

/* Default config for 500M-active model */
static ModelConfig model_config_default(void) {
    ModelConfig c;
    c.dim = 1024;
    c.n_heads = 8;
    c.n_kv_heads = 4;
    c.head_dim = 128;     /* 1024 / 8 */
    c.intermediate = 4096;  /* 4× dim, dense FFN */
    c.n_layers = 12;
    c.max_seq = 4096;
    c.rope_theta = 1000000.0f;
    return c;
}

/* ═══════════════════════════════════════════════════════
 * Model layers
 * ═══════════════════════════════════════════════════════ */

typedef struct {
    /* Attention */
#ifdef TWO3_BINARY
    BinaryWeights W_q;  /* [dim, dim] binary */
    BinaryWeights W_k;  /* [dim, kv_dim] binary */
    BinaryWeights W_v;  /* [dim, kv_dim] binary */
    BinaryWeights W_o;  /* [dim, dim] binary */
#else
    Two3Weights W_q;    /* [dim, dim] ternary */
    Two3Weights W_k;    /* [dim, kv_dim] ternary */
    Two3Weights W_v;    /* [dim, kv_dim] ternary */
    Two3Weights W_o;    /* [dim, dim] ternary */
#endif
    GainState   gain_attn;

    /* Dense FFN */
    DenseFFN    ffn;
    GainState   gain_ffn;
} ModelLayer;

/* ═══════════════════════════════════════════════════════
 * Full model
 * ═══════════════════════════════════════════════════════ */

typedef struct {
    ModelConfig  cfg;

    /* Byte embedding: 256 × dim, float */
    float       *embed;         /* [256 × dim] on host */

#ifdef TWO3_FP_EMBED
    /* Structural fingerprint embedding — four dimensional projections.
     * Each XYZT dimension (1024 bits) projects to dim/4, concat → dim.
     * Replaces byte_embed_cpu with pre-compiled structural encoding. */
    Two3Weights  fp_Wx;         /* [dim/4, 1024] ternary — repetition */
    Two3Weights  fp_Wy;         /* [dim/4, 1024] ternary — opposition */
    Two3Weights  fp_Wz;         /* [dim/4, 1024] ternary — nesting */
    Two3Weights  fp_Wt;         /* [dim/4, 1024] ternary — meta */

    /* Pre-computed fingerprints (mmap'd from .fp file) */
    uint8_t     *fp_data;       /* packed [corpus_size × 512 bytes] — 1 bit per fp element */
    int          fp_corpus_size;
#endif

#ifdef TWO3_LEX_EMBED
    /* Structural context embeddings — additive on top of byte identity.
     * v1: class + brace_depth + paren_depth + mode (normal/string/comment).
     * Initialized at 0.1/sqrt(dim) scale — 10× smaller than identity embed
     * to avoid swamping the byte signal early in training. */
    float       *lex_class_embed;       /* [16 × dim] character class */
    float       *lex_brace_embed;       /* [16 × dim] brace nesting depth */
    float       *lex_paren_embed;       /* [16 × dim] paren nesting depth */
    float       *lex_mode_embed;        /* [3 × dim] 0=normal, 1=string/char, 2=comment */

    /* Pre-computed lex annotations (loaded from .lex file) */
    uint8_t     *lex_classes;           /* [corpus_size] byte class per position */
    uint8_t     *lex_brace_depths;      /* [corpus_size] brace depth per position */
    uint8_t     *lex_paren_depths;      /* [corpus_size] paren depth per position */
    uint8_t     *lex_modes;             /* [corpus_size] mode per position */
    int          lex_corpus_size;
#endif

    /* Layers */
    ModelLayer  *layers;        /* [n_layers] */

    /* Final normalization */
    GainState    gain_final;

    /* Output head: weight-tied with embedding.
     * logits[byte] = hidden · embed[byte] */

    /* RoPE table — precomputed once */
    RoPETable    rope;
} Model;

/* ═══════════════════════════════════════════════════════
 * Embedding — just index into the table
 * ═══════════════════════════════════════════════════════ */

static void byte_embed_cpu(float *out, const float *embed, int byte_val, int dim) {
    memcpy(out, embed + byte_val * dim, dim * sizeof(float));
}

static void byte_logits_cpu(float *logits, const float *hidden, const float *embed, int dim) {
    for (int b = 0; b < 256; b++) {
        float sum = 0;
        for (int d = 0; d < dim; d++)
            sum += hidden[d] * embed[b * dim + d];
        logits[b] = sum;
    }
}

/* Argmax, runner-up, margin (top − second) on 256-way logits — for exit-probe diagnostics. */
static void byte_probe_top2(
    const float *logits256,
    int *out_best,
    float *out_top,
    float *out_second,
    float *out_margin
) {
    int best = 0;
    for (int b = 1; b < 256; b++)
        if (logits256[b] > logits256[best]) best = b;
    int second = (best == 0) ? 1 : 0;
    float s = logits256[second];
    for (int b = 0; b < 256; b++) {
        if (b == best) continue;
        if (logits256[b] > s) {
            s = logits256[b];
            second = b;
        }
    }
    *out_best = best;
    *out_top = logits256[best];
    *out_second = s;
    *out_margin = logits256[best] - s;
}

#ifdef __CUDACC__

__global__ void kernel_byte_embed(
    float *out, const float *embed, int byte_val, int dim
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < dim) out[i] = embed[byte_val * dim + i];
}

__global__ void kernel_byte_logits(
    float *logits, const float *hidden, const float *embed, int dim
) {
    int byte_val = threadIdx.x;
    if (byte_val >= 256) return;
    float sum = 0;
    for (int d = 0; d < dim; d++)
        sum += hidden[d] * embed[byte_val * dim + d];
    logits[byte_val] = sum;
}

#endif /* __CUDACC__ */

/* Softmax + sample */
static int byte_sample(float *logits, float temperature) {
    for (int i = 0; i < 256; i++)
        logits[i] /= temperature;

    float max_l = logits[0];
    for (int i = 1; i < 256; i++)
        if (logits[i] > max_l) max_l = logits[i];

    float sum = 0;
    for (int i = 0; i < 256; i++) {
        logits[i] = expf(logits[i] - max_l);
        sum += logits[i];
    }
    for (int i = 0; i < 256; i++)
        logits[i] /= sum;

    float r = (float)rand() / (float)RAND_MAX;
    float cumsum = 0;
    for (int i = 0; i < 256; i++) {
        cumsum += logits[i];
        if (r < cumsum) return i;
    }
    return 255;
}

/* ═══════════════════════════════════════════════════════
 * Ternary projection helper — CPU path through GPU kernel
 *
 * Takes float input, quantizes to int8, runs ternary matmul
 * on GPU, dequantizes back to float. This is the REAL path.
 * ═══════════════════════════════════════════════════════ */

/* Batched ternary projection: S vectors in one GPU round-trip.
 * input:  [S, dim_in] contiguous row-major
 * output: [S, dim_out] contiguous row-major
 *
 */
static void ternary_project_batch_cpu(
    const Two3Weights *W,
    const float *input,     /* [S, dim_in] */
    float *output,          /* [S, dim_out] */
    int S, int dim_in
) {
    Two3Activations X = two3_quantize_acts(input, S, dim_in);
    Two3Output Y = two3_forward(W, &X);
    two3_dequantize_output(&Y, W, &X, output);
    two3_free_output(&Y);
    two3_free_acts(&X);
    /* Output is O(1) because weight_scale includes 1/sqrt(K). */
}

/* Multi-projection: quantize input ONCE, project against N weight matrices.
 * FBC principle: keep data compressed, only inflate at point of use.
 * Eliminates N-1 redundant quantize+memcpy when projecting same input
 * through multiple weights (Q/K/V share pre_attn_normed). */
static void ternary_project_multi_cpu(
    const Two3Weights **W_list,  /* [N] weight matrices */
    float **output_list,         /* [N] output arrays, each [S, W_list[i]->rows] */
    const float *input,          /* [S, dim_in] shared input */
    int N, int S, int dim_in
) {
    /* Quantize input ONCE */
    Two3Activations X = two3_quantize_acts(input, S, dim_in);

    /* Project against each weight matrix — activations stay on device.
     * Output is O(1) because weight_scale includes 1/sqrt(K). */
    for (int i = 0; i < N; i++) {
        Two3Output Y = two3_forward(W_list[i], &X);
        two3_dequantize_output(&Y, W_list[i], &X, output_list[i]);
        two3_free_output(&Y);
    }

    /* Free quantized activations once */
    two3_free_acts(&X);
}

/* Single-vector convenience wrapper */
static void ternary_project_cpu(
    const Two3Weights *W,
    const float *input,     /* [dim_in] */
    float *output,          /* [dim_out] */
    int dim_in
) {
    ternary_project_batch_cpu(W, input, output, 1, dim_in);
}

#ifdef TWO3_FP_EMBED
/* ═══════════════════════════════════════════════════════
 * Fingerprint embedding — four ternary projections
 *
 * Pre-compiled structure → dim/4 per XYZT dimension → concat → dim.
 * Replaces byte_embed_cpu. Model sees structure, not raw bytes.
 * ═══════════════════════════════════════════════════════ */

static void fp_embed_cpu(float *out, Model *m, int corpus_pos, int dim) {
    int qdim = dim / 4;
    /* Unpack 512 bytes → 4096 floats on the stack (16KB — fine) */
    float fp[4096];
    const uint8_t *raw = m->fp_data + (size_t)corpus_pos * 512;
    for (int i = 0; i < 4096; i++)
        fp[i] = (raw[i >> 3] >> (i & 7)) & 1 ? 1.0f : 0.0f;

    ternary_project_cpu(&m->fp_Wx, fp,        out,            1024);
    ternary_project_cpu(&m->fp_Wy, fp + 1024, out + qdim,     1024);
    ternary_project_cpu(&m->fp_Wz, fp + 2048, out + 2*qdim,   1024);
    ternary_project_cpu(&m->fp_Wt, fp + 3072, out + 3*qdim,   1024);
}

static int fp_load(Model *m, const char *fp_path) {
    FILE *f = fopen(fp_path, "rb");
    if (!f) return -1;

    uint32_t magic, corpus_size, win_size, fp_bits;
    fread(&magic, 4, 1, f);
    fread(&corpus_size, 4, 1, f);
    fread(&win_size, 4, 1, f);
    fread(&fp_bits, 4, 1, f);

    if (magic != 0x46503031 || fp_bits != 4096) { fclose(f); return -2; }

    m->fp_corpus_size = (int)corpus_size;
    /* Store packed: 512 bytes per position (4096 bits), not expanded floats.
     * fp_embed_cpu unpacks on the fly — 572MB RAM instead of 18GB. */
    m->fp_data = (uint8_t*)malloc((size_t)corpus_size * 512);
    fread(m->fp_data, 512, (size_t)corpus_size, f);

    fclose(f);
    printf("[fp] loaded %s: %d positions, %d bits/pos\n", fp_path, corpus_size, fp_bits);
    return 0;
}
#endif /* TWO3_FP_EMBED */

#ifdef TWO3_LEX_EMBED
#include "lexer.h"

/* Load pre-computed .lex annotations and unpack into per-signal arrays.
 * The model owns the memory — freed in model_free. */
static int lex_load(Model *m, const char *lex_path) {
    FILE *f = fopen(lex_path, "rb");
    if (!f) return -1;

    LexFileHeader hdr;
    fread(&hdr, sizeof(hdr), 1, f);
    if (hdr.magic != LEX_MAGIC) { fclose(f); return -2; }

    int n = (int)hdr.corpus_size;
    LexToken *tokens = (LexToken*)malloc(n * sizeof(LexToken));
    fread(tokens, sizeof(LexToken), n, f);
    fclose(f);

    /* Unpack into separate arrays for fast indexed access during training */
    m->lex_classes      = (uint8_t*)malloc(n);
    m->lex_brace_depths = (uint8_t*)malloc(n);
    m->lex_paren_depths = (uint8_t*)malloc(n);
    m->lex_modes        = (uint8_t*)malloc(n);

    for (int i = 0; i < n; i++) {
        m->lex_classes[i]      = tokens[i].byte_class;
        m->lex_brace_depths[i] = tokens[i].depth_id / 16;   /* brace depth */
        m->lex_paren_depths[i] = tokens[i].depth_id % 16;   /* paren depth */

        /* Mode: 0=normal, 1=string/char, 2=comment */
        uint8_t fl = tokens[i].flags;
        if (fl & (LEX_FLAG_IN_LINE_COMMENT | LEX_FLAG_IN_BLOCK_COMMENT))
            m->lex_modes[i] = 2;
        else if (fl & (LEX_FLAG_IN_STRING | LEX_FLAG_IN_CHAR))
            m->lex_modes[i] = 1;
        else
            m->lex_modes[i] = 0;
    }
    m->lex_corpus_size = n;

    free(tokens);
    printf("[lex] loaded %s: %d positions, %d classes\n", lex_path, n, (int)hdr.n_classes);

    /* Print mode distribution */
    int mode_counts[3] = {0, 0, 0};
    for (int i = 0; i < n; i++) mode_counts[m->lex_modes[i]]++;
    printf("[lex] modes: normal=%d (%.1f%%), string/char=%d (%.1f%%), comment=%d (%.1f%%)\n",
           mode_counts[0], 100.0*mode_counts[0]/n,
           mode_counts[1], 100.0*mode_counts[1]/n,
           mode_counts[2], 100.0*mode_counts[2]/n);

    return 0;
}
#endif /* TWO3_LEX_EMBED */

/* ═══════════════════════════════════════════════════════
 * Causal attention — the real thing
 *
 * For position `pos`, attending to positions 0..pos:
 *   scores[t] = Q[pos] · K[t] / sqrt(head_dim)  for t = 0..pos
 *   weights = softmax(scores[0..pos])
 *   out[h] = sum_t weights[t] * V[t][kv_h]
 *
 * GQA: multiple query heads share the same KV head.
 * heads_per_kv = n_heads / n_kv_heads
 * ═══════════════════════════════════════════════════════ */

static void causal_attention_cpu(
    const float *q,          /* [n_heads * head_dim] for current position */
    const float *k_store,    /* [pos+1, n_kv_heads * head_dim] all K so far */
    const float *v_store,    /* [pos+1, n_kv_heads * head_dim] all V so far */
    float *out,              /* [dim] = [n_heads * head_dim] output */
    int pos,                 /* current position (0-indexed) */
    int n_heads, int n_kv_heads, int head_dim
) {
    int kv_dim = n_kv_heads * head_dim;
    int heads_per_kv = n_heads / n_kv_heads;
    float scale = 1.0f / sqrtf((float)head_dim);
    int seq_len = pos + 1;  /* attend to positions 0..pos inclusive */

    float *scores = (float*)malloc(seq_len * sizeof(float));

    for (int h = 0; h < n_heads; h++) {
        int kv_h = h / heads_per_kv;

        /* Compute attention scores: Q[h] · K[t][kv_h] for t = 0..pos */
        for (int t = 0; t < seq_len; t++) {
            float dot = 0;
            for (int d = 0; d < head_dim; d++)
                dot += q[h * head_dim + d] * k_store[t * kv_dim + kv_h * head_dim + d];
            scores[t] = dot * scale;
        }
        /* Causal mask: all positions 0..pos are visible. No masking needed
         * because we only compute scores for t <= pos. */

        /* Softmax over scores[0..pos] */
        float max_s = scores[0];
        for (int t = 1; t < seq_len; t++)
            if (scores[t] > max_s) max_s = scores[t];

        float sum_exp = 0;
        for (int t = 0; t < seq_len; t++) {
            scores[t] = expf(scores[t] - max_s);
            sum_exp += scores[t];
        }
        for (int t = 0; t < seq_len; t++)
            scores[t] /= sum_exp;

        /* Weighted sum of V: out[h] = sum_t scores[t] * V[t][kv_h] */
        for (int d = 0; d < head_dim; d++) {
            float val = 0;
            for (int t = 0; t < seq_len; t++)
                val += scores[t] * v_store[t * kv_dim + kv_h * head_dim + d];
            out[h * head_dim + d] = val;
        }
    }

    free(scores);
}

/* ═══════════════════════════════════════════════════════
 * Random ternary weight generation (for init)
 * ═══════════════════════════════════════════════════════ */

static float* make_random_ternary(int rows, int cols) {
    float *w = (float*)malloc(rows * cols * sizeof(float));
    for (int i = 0; i < rows * cols; i++) {
        float r = (float)rand() / (float)RAND_MAX;
        if (r < 0.25f) w[i] = 0.0f;
        else if (r < 0.625f) w[i] = 1.0f;
        else w[i] = -1.0f;
    }
    return w;
}

/* ═══════════════════════════════════════════════════════
 * Model init — random ternary weights, including MoE experts
 * ═══════════════════════════════════════════════════════ */

static void model_init(Model *m, ModelConfig cfg) {
    m->cfg = cfg;
    int D = cfg.dim;
    int KV = cfg.n_kv_heads * cfg.head_dim;
    int INTER = cfg.intermediate;

#ifdef TWO3_IBC
    /* IBC codebook embedding: deterministic, fixed, no trainable params.
     * Each byte maps to a structured int8 vector, scaled to [-1/127, 1/127]. */
    m->embed = (float*)malloc(256 * D * sizeof(float));
    {
        IBCCodebook cb;
        ibc_codebook_init(&cb);
        for (int b = 0; b < 256; b++)
            ibc_to_float(cb.vectors[b], m->embed + b * D, D);
    }
#elif defined(TWO3_TERNARY_CODEBOOK)
    /* Ternary codebook: impedance-matched to the substrate.
     * 256 × dim entries of {-1, 0, +1}, trimodal (1/3 each).
     * Fixed forever — not learned. Random ternary vectors in dim=128
     * are nearly orthogonal (Johnson-Lindenstrauss): expected dot = 0,
     * std = sqrt(dim/3) ≈ 6.5, max = dim. 5% cross-talk.
     * The codebook has the same impedance as the weight matrices,
     * so signal enters L0 without format conversion. */
    m->embed = (float*)malloc(256 * D * sizeof(float));
    {
        /* Deterministic seed for reproducibility */
        uint32_t rng = 0x23CB00C;  /* "23 codebook" */
        for (int i = 0; i < 256 * D; i++) {
            rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5;
            float r = (float)(rng & 0xFFFF) / 65536.0f;
            if (r < 1.0f / 3.0f)      m->embed[i] = -1.0f;
            else if (r < 2.0f / 3.0f) m->embed[i] =  0.0f;
            else                      m->embed[i] =  1.0f;
        }
    }
    printf("[init] ternary codebook: 256 × %d, trimodal {-1,0,+1}, frozen\n", D);
#else
    /* Byte embedding: tiny — 256 × dim */
    m->embed = (float*)malloc(256 * D * sizeof(float));
    float scale = 1.0f / sqrtf((float)D);
    for (int i = 0; i < 256 * D; i++)
        m->embed[i] = scale * (2.0f * (float)rand() / RAND_MAX - 1.0f);
#endif

#ifdef TWO3_LEX_EMBED
    /* Lex embedding tables — small random init, 10× smaller than identity.
     * Additive: hidden[t] = embed[byte] + class_embed[class] + ... */
    {
        float lex_scale = 0.1f / sqrtf((float)D);
        m->lex_class_embed = (float*)malloc(16 * D * sizeof(float));
        m->lex_brace_embed = (float*)malloc(16 * D * sizeof(float));
        m->lex_paren_embed = (float*)malloc(16 * D * sizeof(float));
        m->lex_mode_embed  = (float*)malloc(3 * D * sizeof(float));
        for (int i = 0; i < 16 * D; i++)
            m->lex_class_embed[i] = lex_scale * (2.0f * (float)rand() / RAND_MAX - 1.0f);
        for (int i = 0; i < 16 * D; i++)
            m->lex_brace_embed[i] = lex_scale * (2.0f * (float)rand() / RAND_MAX - 1.0f);
        for (int i = 0; i < 16 * D; i++)
            m->lex_paren_embed[i] = lex_scale * (2.0f * (float)rand() / RAND_MAX - 1.0f);
        for (int i = 0; i < 3 * D; i++)
            m->lex_mode_embed[i] = lex_scale * (2.0f * (float)rand() / RAND_MAX - 1.0f);
        m->lex_classes = NULL;
        m->lex_brace_depths = NULL;
        m->lex_paren_depths = NULL;
        m->lex_modes = NULL;
        m->lex_corpus_size = 0;
    }
#endif

    /* RoPE */
    rope_init(&m->rope, cfg.head_dim, cfg.max_seq, cfg.rope_theta);

    /* Layers */
    m->layers = (ModelLayer*)calloc(cfg.n_layers, sizeof(ModelLayer));
    for (int l = 0; l < cfg.n_layers; l++) {
        ModelLayer *ly = &m->layers[l];

#ifdef TWO3_BINARY
        /* Attention weights — binary */
        {
            float *wq = (float*)malloc(D * D * sizeof(float));
            float *wk = (float*)malloc(KV * D * sizeof(float));
            float *wv = (float*)malloc(KV * D * sizeof(float));
            float *wo = (float*)malloc(D * D * sizeof(float));
            for (int i = 0; i < D*D; i++) wq[i] = (float)rand() / (float)RAND_MAX;
            for (int i = 0; i < KV*D; i++) wk[i] = (float)rand() / (float)RAND_MAX;
            for (int i = 0; i < KV*D; i++) wv[i] = (float)rand() / (float)RAND_MAX;
            for (int i = 0; i < D*D; i++) wo[i] = (float)rand() / (float)RAND_MAX;
            ly->W_q = binary_pack_weights(wq, D, D); free(wq);
            ly->W_k = binary_pack_weights(wk, KV, D); free(wk);
            ly->W_v = binary_pack_weights(wv, KV, D); free(wv);
            ly->W_o = binary_pack_weights(wo, D, D); free(wo);
        }
#else
        /* Attention weights — ternary */
        {
            float *wq = make_random_ternary(D, D);
            ly->W_q = two3_pack_weights(wq, D, D); free(wq);
            float *wk = make_random_ternary(KV, D);
            ly->W_k = two3_pack_weights(wk, KV, D); free(wk);
            float *wv = make_random_ternary(KV, D);
            ly->W_v = two3_pack_weights(wv, KV, D); free(wv);
            float *wo = make_random_ternary(D, D);
            ly->W_o = two3_pack_weights(wo, D, D); free(wo);
        }
#endif

        gain_init(&ly->gain_attn, D);
        gain_init(&ly->gain_ffn, D);

        /* Dense FFN — gate/up/down ternary */
        ly->ffn.dim = D;
        ly->ffn.intermediate = INTER;
#ifdef TWO3_BINARY
        {
            /* Binary init: random float [0,1], pack to binary at threshold 0.5 */
            float *wg = (float*)malloc(INTER * D * sizeof(float));
            float *wu = (float*)malloc(INTER * D * sizeof(float));
            float *wd = (float*)malloc(D * INTER * sizeof(float));
            for (int i = 0; i < INTER * D; i++) wg[i] = (float)rand() / (float)RAND_MAX;
            for (int i = 0; i < INTER * D; i++) wu[i] = (float)rand() / (float)RAND_MAX;
            for (int i = 0; i < D * INTER; i++) wd[i] = (float)rand() / (float)RAND_MAX;
            ly->ffn.gate = binary_pack_weights(wg, INTER, D); free(wg);
            ly->ffn.up = binary_pack_weights(wu, INTER, D); free(wu);
            ly->ffn.down = binary_pack_weights(wd, D, INTER); free(wd);
        }
#else
        {
            float *wg = make_random_ternary(INTER, D);
            ly->ffn.gate = two3_pack_weights(wg, INTER, D); free(wg);
            float *wu = make_random_ternary(INTER, D);
            ly->ffn.up = two3_pack_weights(wu, INTER, D); free(wu);
            float *wd = make_random_ternary(D, INTER);
            ly->ffn.down = two3_pack_weights(wd, D, INTER); free(wd);
        }
#endif
        dense_ffn_init_buffers(&ly->ffn, cfg.max_seq);
    }

    /* Final gain norm */
    gain_init(&m->gain_final, D);

#ifdef TWO3_FP_EMBED
    /* Fingerprint projection weights — four ternary [dim/4 × 1024] */
    {
        int qdim = D / 4;
        float *wx = make_random_ternary(qdim, 1024);
        m->fp_Wx = two3_pack_weights(wx, qdim, 1024); free(wx);
        float *wy = make_random_ternary(qdim, 1024);
        m->fp_Wy = two3_pack_weights(wy, qdim, 1024); free(wy);
        float *wz = make_random_ternary(qdim, 1024);
        m->fp_Wz = two3_pack_weights(wz, qdim, 1024); free(wz);
        float *wt = make_random_ternary(qdim, 1024);
        m->fp_Wt = two3_pack_weights(wt, qdim, 1024); free(wt);
        m->fp_data = NULL;
        m->fp_corpus_size = 0;
    }
#endif
}

static void model_free(Model *m) {
    free(m->embed);
#ifdef TWO3_LEX_EMBED
    free(m->lex_class_embed);
    free(m->lex_brace_embed);
    free(m->lex_paren_embed);
    free(m->lex_mode_embed);
    if (m->lex_classes) free(m->lex_classes);
    if (m->lex_brace_depths) free(m->lex_brace_depths);
    if (m->lex_paren_depths) free(m->lex_paren_depths);
    if (m->lex_modes) free(m->lex_modes);
#endif
#ifdef TWO3_FP_EMBED
    two3_free_weights(&m->fp_Wx);
    two3_free_weights(&m->fp_Wy);
    two3_free_weights(&m->fp_Wz);
    two3_free_weights(&m->fp_Wt);
    if (m->fp_data) free(m->fp_data);
#endif
    rope_free(&m->rope);
    for (int l = 0; l < m->cfg.n_layers; l++) {
        ModelLayer *ly = &m->layers[l];
#ifdef TWO3_BINARY
        binary_free_weights(&ly->W_q);
        binary_free_weights(&ly->W_k);
        binary_free_weights(&ly->W_v);
        binary_free_weights(&ly->W_o);
#else
        two3_free_weights(&ly->W_q);
        two3_free_weights(&ly->W_k);
        two3_free_weights(&ly->W_v);
        two3_free_weights(&ly->W_o);
#endif
        gain_free(&ly->gain_attn);
        gain_free(&ly->gain_ffn);
#ifdef TWO3_BINARY
        binary_free_weights(&ly->ffn.gate);
        binary_free_weights(&ly->ffn.up);
        binary_free_weights(&ly->ffn.down);
#else
        two3_free_weights(&ly->ffn.gate);
        two3_free_weights(&ly->ffn.up);
        two3_free_weights(&ly->ffn.down);
#endif
        dense_ffn_free_buffers(&ly->ffn);
    }
    free(m->layers);
    gain_free(&m->gain_final);
}

/* ═══════════════════════════════════════════════════════
 * Full-sequence forward pass
 *
 * bytes_in[0..seq_len-1] → all_logits[0..seq_len-1][256]
 *
 * This is the TRAINING path. Every position computed.
 * Causal attention: position t attends to 0..t.
 *
 * For each layer, we store K and V for all positions
 * so attention can look backwards.
 *
 * forward_flags: MODEL_FWD_FLAGS_DEFAULT, or MODEL_FWD_FORCE_FULL_DEPTH
 * (disables TWO3_EARLY_EXIT for parity / calibration).
 * ═══════════════════════════════════════════════════════ */

static void model_forward_sequence_cpu(
    Model *m,
    const uint8_t *bytes_in,   /* [seq_len] input bytes */
    int seq_len,
    float *all_logits,         /* [seq_len × 256] output logits */
    int forward_flags          /* MODEL_FWD_* */
) {
#ifndef TWO3_EARLY_EXIT
    (void)forward_flags;
#endif
    int D = m->cfg.dim;
    int KV = m->cfg.n_kv_heads * m->cfg.head_dim;
    int HD = m->cfg.head_dim;
    int NH = m->cfg.n_heads;
    int NKV = m->cfg.n_kv_heads;

    /* Hidden states for all positions: [seq_len × dim] */
    float *hidden = (float*)calloc(seq_len * D, sizeof(float));

    /* Embed all bytes */
    for (int t = 0; t < seq_len; t++)
        byte_embed_cpu(hidden + t * D, m->embed, bytes_in[t], D);

    /* Batched work buffers — [seq_len × dim] contiguous for GPU batch calls */
    float *normed_all = (float*)malloc(seq_len * D * sizeof(float));
    float *q_all      = (float*)malloc(seq_len * D * sizeof(float));
    float *k_all      = (float*)malloc(seq_len * KV * sizeof(float));
    float *v_all      = (float*)malloc(seq_len * KV * sizeof(float));
    float *attn_out_all = (float*)malloc(seq_len * D * sizeof(float));
    float *o_proj_all = (float*)malloc(seq_len * D * sizeof(float));
    float *moe_out    = (float*)malloc(D * sizeof(float));
    float *normed_one = (float*)malloc(D * sizeof(float));

    float res_scale = 1.0f;  /* Gap 1 fix: dequant is O(1), gain normalizes */

#ifdef TWO3_EARLY_EXIT
    int layers_early_stop = 0;
#endif

    for (int l = 0; l < m->cfg.n_layers; l++) {
        ModelLayer *ly = &m->layers[l];

        /* ── Attention block (batched projections) ── */

        /* Phase 1: gain norm all positions (sequential — R depends on previous) */
        for (int t = 0; t < seq_len; t++)
            gain_forward_cpu(normed_all + t * D, hidden + t * D,
                             ly->gain_attn.R, ly->gain_attn.C, D);

        /* Phase 2: multi-projection Q/K/V — quantize once, project 3x.
         * 1 quantize + 3 GPU matmuls instead of 3 quantize + 3 matmuls. */
#ifdef TWO3_BINARY
        {
            const BinaryWeights *W_qkv[3] = { &ly->W_q, &ly->W_k, &ly->W_v };
            float *out_qkv[3] = { q_all, k_all, v_all };
            binary_project_multi_cpu(W_qkv, out_qkv, normed_all, 3, seq_len, D);
        }
#else
        {
            const Two3Weights *W_qkv[3] = { &ly->W_q, &ly->W_k, &ly->W_v };
            float *out_qkv[3] = { q_all, k_all, v_all };
            ternary_project_multi_cpu(W_qkv, out_qkv, normed_all, 3, seq_len, D);
        }
#endif

        /* Phase 3: RoPE all positions (CPU, cheap) */
        for (int t = 0; t < seq_len; t++)
            rope_apply_cpu(q_all + t * D, k_all + t * KV, &m->rope, t, NH, NKV);

        /* Phase 4: causal attention per position (sequential dependency) */
        for (int t = 0; t < seq_len; t++)
            causal_attention_cpu(q_all + t * D, k_all, v_all,
                                 attn_out_all + t * D, t, NH, NKV, HD);

        /* Phase 5: batch O projection — 1 GPU call instead of seq_len */
#ifdef TWO3_BINARY
        binary_project_batch_cpu(&ly->W_o, attn_out_all, o_proj_all, seq_len, D);
#else
        ternary_project_batch_cpu(&ly->W_o, attn_out_all, o_proj_all, seq_len, D);
#endif

        /* Phase 6: scale + residual add (CPU) */
        {
            float s = res_scale;
            for (int t = 0; t < seq_len; t++)
                for (int i = 0; i < D; i++)
                    hidden[t * D + i] += s * o_proj_all[t * D + i];
        }

        /* ── FFN block (expert-grouped batched projections) ── */

        /* Phase 7: gain norm all positions */
        for (int t = 0; t < seq_len; t++)
            gain_forward_cpu(normed_all + t * D, hidden + t * D,
                             ly->gain_ffn.R, ly->gain_ffn.C, D);

        /* Phase 8: dense FFN — batch all positions */
        {
            float *ffn_out = (float*)malloc(seq_len * D * sizeof(float));
            dense_ffn_forward_batch(&ly->ffn, normed_all, ffn_out,
                                     seq_len, D, m->cfg.intermediate);

            /* Phase 9: residual add */
            for (int t = 0; t < seq_len; t++)
                for (int i = 0; i < D; i++)
                    hidden[t * D + i] += res_scale * ffn_out[t * D + i];
            free(ffn_out);
        }

        /* ═══════════════════════════════════════════════════════
         * Reservoir depletion early exit.
         * mean(C - R) across attn + ffn gain states for this layer.
         * If < threshold, reservoirs barely depleted — layer had little
         * effect, skip remaining layers. Only for seq_len==1 inference.
         * ═══════════════════════════════════════════════════════ */
#if defined(TWO3_EARLY_EXIT) || defined(TWO3_DEBUG_EXIT_METRICS)
        {
            float depletion_sum = 0.f;
            for (int i = 0; i < D; i++) {
                depletion_sum += (ly->gain_attn.C[i] - ly->gain_attn.R[i]);
                depletion_sum += (ly->gain_ffn.C[i] - ly->gain_ffn.R[i]);
            }
            float mean_depletion = depletion_sum / (2.0f * D);
#ifdef TWO3_DEBUG_EXIT_METRICS
            printf("[exit_reservoir] layer=%d mean_depletion=%.6f thresh=%.4f\n",
                   l, mean_depletion, (double)TWO3_EXIT_DEPLETION_THRESH);
            fflush(stdout);
#endif
#ifdef TWO3_EARLY_EXIT
            if (seq_len == 1 && l >= 1 && mean_depletion < TWO3_EXIT_DEPLETION_THRESH
                && !(forward_flags & MODEL_FWD_FORCE_FULL_DEPTH))
                layers_early_stop = 1;
#endif
        }
#endif
#ifdef TWO3_EARLY_EXIT
        if (layers_early_stop)
            break;
#endif
    }

    /* Final gain norm + logits for each position */
    for (int t = 0; t < seq_len; t++) {
        gain_forward_cpu(normed_one, hidden + t * D,
                         m->gain_final.R, m->gain_final.C, D);
        byte_logits_cpu(all_logits + t * 256, normed_one, m->embed, D);
    }

    free(hidden);
    free(normed_all); free(q_all); free(k_all); free(v_all);
    free(attn_out_all); free(o_proj_all);
    free(moe_out); free(normed_one);
}

/* ═══════════════════════════════════════════════════════
 * Single-byte forward — inference/generation path
 *
 * Uses stored KV from previous positions. Caller manages
 * the KV store across calls.
 *
 * This wraps model_forward_sequence_cpu for single-byte use:
 * inefficient (recomputes all previous positions) but correct.
 * KV cache optimization is Layer 2.5.
 * ═══════════════════════════════════════════════════════ */

typedef struct {
    uint8_t *bytes;     /* accumulated byte sequence */
    int      len;       /* current length */
    int      capacity;  /* allocated capacity */
} GenerationContext;

/* ═══════════════════════════════════════════════════════
 * KV Cache — device-resident, enables O(T·D) generation
 *
 * K,V at position t are deterministic and immutable after
 * computation. Caching them reduces generation from O(T²·D)
 * to O(T·D). Memory: ~192 MB for 12 layers, 4096 seq, 512 kv_dim.
 * ═══════════════════════════════════════════════════════ */

typedef struct {
    float *K;       /* [n_layers × max_seq × kv_dim] device memory */
    float *V;       /* [n_layers × max_seq × kv_dim] device memory */
    int    len;     /* current cached length */
    int    max_seq;
    int    kv_dim;
    int    n_layers;
} KVCache;

static void kv_cache_init(KVCache *kv, int n_layers, int max_seq, int kv_dim) {
    kv->n_layers = n_layers;
    kv->max_seq = max_seq;
    kv->kv_dim = kv_dim;
    kv->len = 0;
    size_t total = (size_t)n_layers * max_seq * kv_dim * sizeof(float);
    cudaMalloc(&kv->K, total);
    cudaMalloc(&kv->V, total);
    cudaMemset(kv->K, 0, total);
    cudaMemset(kv->V, 0, total);
}

static void kv_cache_free(KVCache *kv) {
    cudaFree(kv->K);
    cudaFree(kv->V);
    kv->K = kv->V = NULL;
}

static float* kv_K_at(KVCache *kv, int l, int t) {
    return kv->K + ((size_t)l * kv->max_seq + t) * kv->kv_dim;
}

static float* kv_V_at(KVCache *kv, int l, int t) {
    return kv->V + ((size_t)l * kv->max_seq + t) * kv->kv_dim;
}

/* ═══════════════════════════════════════════════════════
 * Cached forward — single-step generation with KV cache
 *
 * Uses cached K,V from previous positions. O(T·D) per step
 * instead of O(T²·D). Call kv_cache_init before generation,
 * kv_cache_free when done.
 * ═══════════════════════════════════════════════════════ */

static void model_forward_cached(
    Model *m,
    KVCache *kv,
    uint8_t byte_in,
    float *logits          /* [256] output logits */
) {
    int D = m->cfg.dim;
    int KV = m->cfg.n_kv_heads * m->cfg.head_dim;
    int HD = m->cfg.head_dim;
    int NH = m->cfg.n_heads;
    int NKV = m->cfg.n_kv_heads;
    int pos = kv->len;

    float *hidden = (float*)malloc(D * sizeof(float));
    float *normed = (float*)malloc(D * sizeof(float));
    float *q = (float*)malloc(D * sizeof(float));
    float *k_new = (float*)malloc(KV * sizeof(float));
    float *v_new = (float*)malloc(KV * sizeof(float));
    float *attn_out = (float*)malloc(D * sizeof(float));
    float *o_proj = (float*)malloc(D * sizeof(float));
    float *moe_out = (float*)calloc(D, sizeof(float));

    /* Embed single byte */
    byte_embed_cpu(hidden, m->embed, byte_in, D);

    float res_scale = 1.0f;  /* Gap 1 fix: dequant is O(1), gain normalizes */

    for (int l = 0; l < m->cfg.n_layers; l++) {
        ModelLayer *ly = &m->layers[l];

        /* Gain norm */
        gain_forward_cpu(normed, hidden, ly->gain_attn.R, ly->gain_attn.C, D);

        /* Q/K/V projections — single position */
#ifdef TWO3_BINARY
        binary_project_cpu(&ly->W_q, normed, q, D);
        binary_project_cpu(&ly->W_k, normed, k_new, D);
        binary_project_cpu(&ly->W_v, normed, v_new, D);
#else
        ternary_project_cpu(&ly->W_q, normed, q, D);
        ternary_project_cpu(&ly->W_k, normed, k_new, D);
        ternary_project_cpu(&ly->W_v, normed, v_new, D);
#endif

        /* Store K,V into cache (device-resident) */
        cudaMemcpy(kv_K_at(kv, l, pos), k_new, KV * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(kv_V_at(kv, l, pos), v_new, KV * sizeof(float), cudaMemcpyHostToDevice);

        /* RoPE for current position */
        rope_apply_cpu(q, k_new, &m->rope, pos, NH, NKV);

        /* Attention: Q[pos] against K[0..pos], V[0..pos] from cache */
        /* Need to copy cached K,V back to host for attention */
        float *k_cached = (float*)malloc((pos + 1) * KV * sizeof(float));
        float *v_cached = (float*)malloc((pos + 1) * KV * sizeof(float));
        for (int t = 0; t <= pos; t++) {
            cudaMemcpy(k_cached + t * KV, kv_K_at(kv, l, t), KV * sizeof(float), cudaMemcpyDeviceToHost);
            cudaMemcpy(v_cached + t * KV, kv_V_at(kv, l, t), KV * sizeof(float), cudaMemcpyDeviceToHost);
        }
        causal_attention_cpu(q, k_cached, v_cached, attn_out, pos + 1, NH, NKV, HD);
        free(k_cached);
        free(v_cached);

        /* O projection */
#ifdef TWO3_BINARY
        binary_project_cpu(&ly->W_o, attn_out, o_proj, D);
#else
        ternary_project_cpu(&ly->W_o, attn_out, o_proj, D);
#endif

        /* Residual */
        for (int i = 0; i < D; i++)
            hidden[i] += res_scale * o_proj[i];

        /* FFN block (dense) */
        gain_forward_cpu(normed, hidden, ly->gain_ffn.R, ly->gain_ffn.C, D);

        float *ffn_out = (float*)malloc(D * sizeof(float));
        dense_ffn_forward(&ly->ffn, normed, ffn_out, D, m->cfg.intermediate);

        /* Residual add */
        {
            for (int i = 0; i < D; i++)
                hidden[i] += res_scale * ffn_out[i];
        }
        free(ffn_out);
    }

    /* Final norm + logits */
    gain_forward_cpu(normed, hidden, m->gain_final.R, m->gain_final.C, D);
    byte_logits_cpu(logits, normed, m->embed, D);

    kv->len++;

    free(hidden); free(normed); free(q); free(k_new); free(v_new);
    free(attn_out); free(o_proj); free(moe_out);
}

static void gen_ctx_init(GenerationContext *ctx, int max_seq) {
    ctx->bytes = (uint8_t*)malloc(max_seq);
    ctx->len = 0;
    ctx->capacity = max_seq;
}

static void gen_ctx_free(GenerationContext *ctx) {
    free(ctx->bytes);
    ctx->bytes = NULL;
}

static void gen_ctx_append(GenerationContext *ctx, uint8_t byte) {
    if (ctx->len < ctx->capacity)
        ctx->bytes[ctx->len++] = byte;
}

/* Generate: feed accumulated context, get logits for last position.
 * NOTE: This recomputes everything from scratch each call.
 * Proper KV cache comes at Layer 2.5. For now, correctness > speed. */
static void model_generate_cpu(
    Model *m,
    GenerationContext *ctx,
    float *logits       /* [256] output logits for next byte */
) {
    float *all_logits = (float*)malloc(ctx->len * 256 * sizeof(float));
    model_forward_sequence_cpu(m, ctx->bytes, ctx->len, all_logits, MODEL_FWD_FLAGS_DEFAULT);

    /* Copy logits for the last position */
    memcpy(logits, all_logits + (ctx->len - 1) * 256, 256 * sizeof(float));
    free(all_logits);
}

/* ═══════════════════════════════════════════════════════
 * Legacy single-byte interface (for backward compat)
 *
 * WARNING: Without attention context, this produces
 * position-independent output. Use model_generate_cpu
 * or model_forward_sequence_cpu for real inference/training.
 * ═══════════════════════════════════════════════════════ */

static void model_forward_cpu(
    Model *m,
    int byte_in,
    int pos,
    float *hidden,      /* [dim] — unused in new path, kept for compat */
    float *logits       /* [256] output logits */
) {
    /* Route through sequence forward with length=1.
     * This means no attention context from previous bytes.
     * For real generation, use GenerationContext. */
    uint8_t b = (uint8_t)byte_in;
    float all_logits[256];
    model_forward_sequence_cpu(m, &b, 1, all_logits, MODEL_FWD_FLAGS_DEFAULT);
    memcpy(logits, all_logits, 256 * sizeof(float));

    /* Also write hidden for backward compat (grab from embed) */
    byte_embed_cpu(hidden, m->embed, byte_in, m->cfg.dim);
}

#endif /* MODEL_H */
