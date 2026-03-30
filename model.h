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
#include "moe.h"

#include <string.h>
#include <float.h>

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

    /* Byte embedding: 256 × dim, float */
    float       *embed;         /* [256 × dim] on host */

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
 * output: [S, dim_out] contiguous row-major */
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

    /* Project against each weight matrix — activations stay on device */
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

/* ═══════════════════════════════════════════════════════
 * MoE expert forward — REAL ternary gate/up/down
 *
 * gate = ternary_matmul(x, expert.gate) → squared_relu
 * up   = ternary_matmul(x, expert.up)
 * h    = gate * up
 * out  = ternary_matmul(h, expert.down)
 * ═══════════════════════════════════════════════════════ */

static void moe_expert_forward_real(
    const MoEExpert *expert,
    const float *x,          /* [dim] */
    float *output,           /* [dim] */
    int dim, int intermediate
) {
    float *gate = (float*)malloc(intermediate * sizeof(float));
    float *up   = (float*)malloc(intermediate * sizeof(float));

    /* Gate + Up share input — quantize once, project 2x */
    {
        const Two3Weights *W_gu[2] = { &expert->gate, &expert->up };
        float *out_gu[2] = { gate, up };
        ternary_project_multi_cpu(W_gu, out_gu, x, 2, 1, dim);
    }

    /* Scale BEFORE squared ReLU — critical: squaring amplifies magnitude.
     * Same pattern as Layer 1 integration test. */
    float scale = 1.0f / sqrtf((float)dim);
    for (int i = 0; i < intermediate; i++) {
        gate[i] *= scale;
        up[i] *= scale;
    }

    /* Squared ReLU on gate */
    squared_relu_cpu(gate, gate, intermediate);

    /* Element-wise: gate * up */
    for (int i = 0; i < intermediate; i++)
        gate[i] *= up[i];

    /* Down projection (ternary) */
    ternary_project_cpu(&expert->down, gate, output, intermediate);

    free(gate);
    free(up);
}

/* MoE forward with real expert computation */
static void moe_forward_real(
    const MoELayer *moe,
    const float *x,          /* [dim] */
    float *output,           /* [dim] */
    const MoESelection *sel
) {
    int D = moe->dim;
    memset(output, 0, D * sizeof(float));

    float *expert_out = (float*)malloc(D * sizeof(float));

    for (int k = 0; k < MOE_TOP_K; k++) {
        int eid = sel->expert_ids[k];
        float w = sel->expert_weights[k];

        moe_expert_forward_real(&moe->experts[eid], x, expert_out,
                                 D, moe->intermediate);

        for (int i = 0; i < D; i++)
            output[i] += w * expert_out[i];
    }

    free(expert_out);
}

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

        /* Attention weights — ternary */
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

        /* MoE — router (float) + expert weights (ternary) */
        moe_router_init(&ly->moe.router, D, MOE_NUM_EXPERTS);
        ly->moe.dim = D;
        ly->moe.intermediate = INTER;

        for (int e = 0; e < MOE_NUM_EXPERTS; e++) {
            float *wg = make_random_ternary(INTER, D);
            ly->moe.experts[e].gate = two3_pack_weights(wg, INTER, D); free(wg);

            float *wu = make_random_ternary(INTER, D);
            ly->moe.experts[e].up = two3_pack_weights(wu, INTER, D); free(wu);

            float *wd = make_random_ternary(D, INTER);
            ly->moe.experts[e].down = two3_pack_weights(wd, D, INTER); free(wd);
        }
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
        for (int e = 0; e < MOE_NUM_EXPERTS; e++) {
            two3_free_weights(&ly->moe.experts[e].gate);
            two3_free_weights(&ly->moe.experts[e].up);
            two3_free_weights(&ly->moe.experts[e].down);
        }
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
 * ═══════════════════════════════════════════════════════ */

static void model_forward_sequence_cpu(
    Model *m,
    const uint8_t *bytes_in,   /* [seq_len] input bytes */
    int seq_len,
    float *all_logits          /* [seq_len × 256] output logits */
) {
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

    float res_scale = 1.0f / sqrtf(2.0f * (float)m->cfg.n_layers);

    for (int l = 0; l < m->cfg.n_layers; l++) {
        ModelLayer *ly = &m->layers[l];

        /* ── Attention block (batched projections) ── */

        /* Phase 1: gain norm all positions (sequential — R depends on previous) */
        for (int t = 0; t < seq_len; t++)
            gain_forward_cpu(normed_all + t * D, hidden + t * D,
                             ly->gain_attn.R, ly->gain_attn.C, D);

        /* Phase 2: multi-projection Q/K/V — quantize once, project 3x.
         * 1 quantize + 3 GPU matmuls instead of 3 quantize + 3 matmuls. */
        {
            const Two3Weights *W_qkv[3] = { &ly->W_q, &ly->W_k, &ly->W_v };
            float *out_qkv[3] = { q_all, k_all, v_all };
            ternary_project_multi_cpu(W_qkv, out_qkv, normed_all, 3, seq_len, D);
        }

        /* Phase 3: RoPE all positions (CPU, cheap) */
        for (int t = 0; t < seq_len; t++)
            rope_apply_cpu(q_all + t * D, k_all + t * KV, &m->rope, t, NH, NKV);

        /* Phase 4: causal attention per position (sequential dependency) */
        for (int t = 0; t < seq_len; t++)
            causal_attention_cpu(q_all + t * D, k_all, v_all,
                                 attn_out_all + t * D, t, NH, NKV, HD);

        /* Phase 5: batch O projection — 1 GPU call instead of seq_len */
        ternary_project_batch_cpu(&ly->W_o, attn_out_all, o_proj_all, seq_len, D);

        /* Phase 6: scale + residual add (CPU) */
        {
            float s = res_scale / sqrtf((float)D);
            for (int t = 0; t < seq_len; t++)
                for (int i = 0; i < D; i++)
                    hidden[t * D + i] += s * o_proj_all[t * D + i];
        }

        /* ── FFN block (MoE still per-position for now) ── */

        /* Phase 7: gain norm + MoE per position */
        for (int t = 0; t < seq_len; t++) {
            float *h = hidden + t * D;
            gain_forward_cpu(normed_one, h, ly->gain_ffn.R, ly->gain_ffn.C, D);

            MoESelection sel;
            moe_route(&ly->moe.router, normed_one, &sel);
            moe_forward_real(&ly->moe, normed_one, moe_out, &sel);

            for (int i = 0; i < D; i++)
                h[i] += res_scale * moe_out[i];
        }
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
    model_forward_sequence_cpu(m, ctx->bytes, ctx->len, all_logits);

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
    model_forward_sequence_cpu(m, &b, 1, all_logits);
    memcpy(logits, all_logits, 256 * sizeof(float));

    /* Also write hidden for backward compat (grab from embed) */
    byte_embed_cpu(hidden, m->embed, byte_in, m->cfg.dim);
}

#endif /* MODEL_H */
