/*
 * model.h — Complete {2,3} Model
 *
 * Layer 2: wraps all components into a full model.
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
 * Isaac & CC — March 2026
 */

#ifndef MODEL_H
#define MODEL_H

#include "two3.h"
#include "gain.h"
#include "rope.h"
#include "activation.h"
#include "moe.h"

/* ═══════════════════════════════════════════════════════
 * Model config
 * ═══════════════════════════════════════════════════════ */

#define BYTE_VOCAB 256  /* raw bytes, not tokens */

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
    c.intermediate = 2048;
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
    Two3Weights W_q;    /* [dim, dim] ternary */
    Two3Weights W_k;    /* [dim, kv_dim] ternary */
    Two3Weights W_v;    /* [dim, kv_dim] ternary */
    Two3Weights W_o;    /* [dim, dim] ternary */
    GainState   gain_attn;

    /* MoE FFN */
    MoELayer    moe;
    GainState   gain_ffn;
} ModelLayer;

/* ═══════════════════════════════════════════════════════
 * Full model
 * ═══════════════════════════════════════════════════════ */

typedef struct {
    ModelConfig  cfg;

    /* Byte embedding: 256 × dim, float16 stored as float */
    float       *embed;         /* [256 × dim] on host */
    float       *d_embed;       /* [256 × dim] on GPU */

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
 * Embedding lookup — just index into the table
 * ═══════════════════════════════════════════════════════ */

#ifdef __CUDACC__

__global__ void kernel_byte_embed(
    float *out,             /* [dim] */
    const float *embed,     /* [256 × dim] */
    int byte_val, int dim
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < dim) out[i] = embed[byte_val * dim + i];
}

/* Output logits: dot product of hidden with each embedding row */
__global__ void kernel_byte_logits(
    float *logits,          /* [256] */
    const float *hidden,    /* [dim] */
    const float *embed,     /* [256 × dim] */
    int dim
) {
    int byte_val = threadIdx.x;  /* one thread per byte value */
    if (byte_val >= 256) return;

    float sum = 0;
    for (int d = 0; d < dim; d++)
        sum += hidden[d] * embed[byte_val * dim + d];
    logits[byte_val] = sum;
}

#endif /* __CUDACC__ */

/* CPU reference */
static void byte_embed_cpu(float *out, const float *embed, int byte_val, int dim) {
    for (int i = 0; i < dim; i++)
        out[i] = embed[byte_val * dim + i];
}

static void byte_logits_cpu(float *logits, const float *hidden, const float *embed, int dim) {
    for (int b = 0; b < 256; b++) {
        float sum = 0;
        for (int d = 0; d < dim; d++)
            sum += hidden[d] * embed[b * dim + d];
        logits[b] = sum;
    }
}

/* Softmax + sample */
static int byte_sample(float *logits, float temperature) {
    /* Divide by temperature */
    for (int i = 0; i < 256; i++)
        logits[i] /= temperature;

    /* Softmax */
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

    /* Sample from distribution */
    float r = (float)rand() / (float)RAND_MAX;
    float cumsum = 0;
    for (int i = 0; i < 256; i++) {
        cumsum += logits[i];
        if (r < cumsum) return i;
    }
    return 255;
}

/* ═══════════════════════════════════════════════════════
 * Model init — random ternary weights
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

static void model_init(Model *m, ModelConfig cfg) {
    m->cfg = cfg;
    int D = cfg.dim;
    int KV = cfg.n_kv_heads * cfg.head_dim;
    int INTER = cfg.intermediate;

    /* Byte embedding: tiny — 256 × dim */
    m->embed = (float*)malloc(256 * D * sizeof(float));
    float scale = 1.0f / sqrtf((float)D);
    for (int i = 0; i < 256 * D; i++)
        m->embed[i] = scale * (2.0f * (float)rand() / RAND_MAX - 1.0f);

    /* RoPE */
    rope_init(&m->rope, cfg.head_dim, cfg.max_seq, cfg.rope_theta);

    /* Layers */
    m->layers = (ModelLayer*)calloc(cfg.n_layers, sizeof(ModelLayer));
    for (int l = 0; l < cfg.n_layers; l++) {
        ModelLayer *ly = &m->layers[l];

        /* Attention weights */
        float *wq = make_random_ternary(D, D);
        ly->W_q = two3_pack_weights(wq, D, D); free(wq);

        float *wk = make_random_ternary(KV, D);
        ly->W_k = two3_pack_weights(wk, KV, D); free(wk);

        float *wv = make_random_ternary(KV, D);
        ly->W_v = two3_pack_weights(wv, KV, D); free(wv);

        float *wo = make_random_ternary(D, D);
        ly->W_o = two3_pack_weights(wo, D, D); free(wo);

        gain_init(&ly->gain_attn, D);
        gain_init(&ly->gain_ffn, D);

        /* MoE */
        moe_router_init(&ly->moe.router, D, MOE_NUM_EXPERTS);
        ly->moe.dim = D;
        ly->moe.intermediate = INTER;
        /* Expert weights would go here — skipped for now,
         * MoE forward uses simulated output in CPU reference */
    }

    /* Final gain norm */
    gain_init(&m->gain_final, D);
}

static void model_free(Model *m) {
    free(m->embed);
    rope_free(&m->rope);
    for (int l = 0; l < m->cfg.n_layers; l++) {
        ModelLayer *ly = &m->layers[l];
        two3_free_weights(&ly->W_q);
        two3_free_weights(&ly->W_k);
        two3_free_weights(&ly->W_v);
        two3_free_weights(&ly->W_o);
        gain_free(&ly->gain_attn);
        gain_free(&ly->gain_ffn);
        moe_router_free(&ly->moe.router);
    }
    free(m->layers);
    gain_free(&m->gain_final);
}

/* ═══════════════════════════════════════════════════════
 * Model forward — one byte in, 256 logits out
 *
 * byte_in → embed → [gain → attn → residual → gain → moe → residual] × N
 *         → final_gain → logits (dot with embed, weight-tied)
 * ═══════════════════════════════════════════════════════ */

static void model_forward_cpu(
    Model *m,
    int byte_in,        /* input byte (0-255) */
    int pos,            /* position in sequence */
    float *hidden,      /* [dim] work buffer — also output */
    float *logits       /* [256] output logits */
) {
    int D = m->cfg.dim;
    float scale = 1.0f / sqrtf((float)D);

    /* Embed */
    byte_embed_cpu(hidden, m->embed, byte_in, D);

    /* N layers */
    float *normed = (float*)malloc(D * sizeof(float));

    for (int l = 0; l < m->cfg.n_layers; l++) {
        ModelLayer *ly = &m->layers[l];

        /* Gain norm → attention (simulated: just gain + residual) */
        gain_forward_cpu(normed, hidden, ly->gain_attn.R, ly->gain_attn.C, D);

        /* Ternary Q projection on GPU, dequant */
        Two3Activations X = two3_quantize_acts(normed, 1, D);
        Two3Output Y = two3_forward(&ly->W_q, &X);
        float *q = (float*)malloc(D * sizeof(float));
        two3_dequantize_output(&Y, &ly->W_q, &X, q);
        two3_free_output(&Y);
        two3_free_acts(&X);

        /* Scale + squared ReLU */
        for (int i = 0; i < D; i++) q[i] *= scale;
        squared_relu_cpu(q, q, D);

        /* Residual */
        for (int i = 0; i < D; i++) hidden[i] += q[i];
        free(q);

        /* Gain norm → MoE FFN (simulated) */
        gain_forward_cpu(normed, hidden, ly->gain_ffn.R, ly->gain_ffn.C, D);

        MoESelection sel;
        moe_route(&ly->moe.router, normed, &sel);

        float *moe_out = (float*)calloc(D, sizeof(float));
        moe_forward_cpu_ref(&ly->moe, normed, moe_out, &sel);

        for (int i = 0; i < D; i++) moe_out[i] *= scale;
        squared_relu_cpu(moe_out, moe_out, D);

        /* Residual */
        for (int i = 0; i < D; i++) hidden[i] += moe_out[i];
        free(moe_out);
    }

    /* Final gain norm */
    gain_forward_cpu(normed, hidden, m->gain_final.R, m->gain_final.C, D);

    /* Logits: dot product with embedding (weight-tied) */
    byte_logits_cpu(logits, normed, m->embed, D);

    free(normed);
}

#endif /* MODEL_H */
