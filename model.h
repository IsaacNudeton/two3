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

    /* Normalize output by 1/sqrt(K) — outside dequant so backward
     * gradients flow at natural scale through STE. */
    float inv_sqrt_K = 1.0f / sqrtf((float)dim_in);
    for (int i = 0; i < S * W->rows; i++)
        output[i] *= inv_sqrt_K;
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
    float inv_sqrt_K = 1.0f / sqrtf((float)dim_in);
    for (int i = 0; i < N; i++) {
        Two3Output Y = two3_forward(W_list[i], &X);
        two3_dequantize_output(&Y, W_list[i], &X, output_list[i]);
        two3_free_output(&Y);

        /* Normalize output — outside dequant for clean backward */
        int out_size = S * W_list[i]->rows;
        for (int j = 0; j < out_size; j++)
            output_list[i][j] *= inv_sqrt_K;
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

    float res_scale = 1.0f / sqrtf(2.0f * (float)m->cfg.n_layers);

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
            float s = res_scale;
            for (int t = 0; t < seq_len; t++)
                for (int i = 0; i < D; i++)
                    hidden[t * D + i] += s * o_proj_all[t * D + i];
        }

        /* ── FFN block (expert-grouped batched projections) ── */

        /* Phase 7: gain norm all positions (sequential) */
        for (int t = 0; t < seq_len; t++)
            gain_forward_cpu(normed_all + t * D, hidden + t * D,
                             ly->gain_ffn.R, ly->gain_ffn.C, D);

        /* Phase 8: route all positions */
        MoESelection *all_sel = (MoESelection*)malloc(seq_len * sizeof(MoESelection));
        for (int t = 0; t < seq_len; t++)
            moe_route(&ly->moe.router, normed_all + t * D, &all_sel[t]);

        /* Phase 9: expert-grouped forward */
        {
            int *expert_pos_flat = (int*)malloc(MOE_NUM_EXPERTS * seq_len * sizeof(int));
            #define EP(e, i) expert_pos_flat[(e) * seq_len + (i)]
            int expert_cnt[MOE_NUM_EXPERTS];
            float *gather_in  = (float*)malloc(seq_len * D * sizeof(float));
            int INTER = m->cfg.intermediate;
            float *gate_b     = (float*)malloc(seq_len * INTER * sizeof(float));
            float *up_b       = (float*)malloc(seq_len * INTER * sizeof(float));
            float *h_exp      = (float*)malloc(seq_len * INTER * sizeof(float));
            float *down_b     = (float*)malloc(seq_len * D * sizeof(float));
            float *moe_result = (float*)calloc(seq_len * D, sizeof(float));

            for (int k_sel = 0; k_sel < MOE_TOP_K; k_sel++) {
                memset(expert_cnt, 0, sizeof(expert_cnt));
                for (int t = 0; t < seq_len; t++) {
                    int eid = all_sel[t].expert_ids[k_sel];
                    EP(eid, expert_cnt[eid]++) = t;
                }

                for (int e = 0; e < MOE_NUM_EXPERTS; e++) {
                    int cnt = expert_cnt[e];
                    if (cnt == 0) continue;

                    for (int i = 0; i < cnt; i++)
                        memcpy(gather_in + i * D, normed_all + EP(e, i) * D,
                               D * sizeof(float));

                    const Two3Weights *W_gu[2] = { &ly->moe.experts[e].gate, &ly->moe.experts[e].up };
                    float *out_gu[2] = { gate_b, up_b };
                    ternary_project_multi_cpu(W_gu, out_gu, gather_in, 2, cnt, D);

                    float scale = 1.0f / sqrtf((float)D);
                    for (int i = 0; i < cnt * INTER; i++) {
                        gate_b[i] *= scale;
                        up_b[i] *= scale;
                    }
                    for (int i = 0; i < cnt * INTER; i++) {
                        float g = gate_b[i];
                        gate_b[i] = (g > 0.0f) ? g * g : 0.0f;
                    }
                    for (int i = 0; i < cnt * INTER; i++)
                        h_exp[i] = gate_b[i] * up_b[i];

                    ternary_project_batch_cpu(&ly->moe.experts[e].down,
                                             h_exp, down_b, cnt, INTER);

                    for (int i = 0; i < cnt; i++) {
                        int t = EP(e, i);
                        float w = all_sel[t].expert_weights[k_sel];
                        for (int d = 0; d < D; d++)
                            moe_result[t * D + d] += w * down_b[i * D + d];
                    }
                }
            }

            for (int t = 0; t < seq_len; t++)
                for (int i = 0; i < D; i++)
                    hidden[t * D + i] += res_scale * moe_result[t * D + i];

            free(expert_pos_flat);
            #undef EP
            free(gather_in); free(gate_b); free(up_b);
            free(h_exp); free(down_b); free(moe_result);
        }
        free(all_sel);

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

    float res_scale = 1.0f / sqrtf(2.0f * (float)m->cfg.n_layers);

    for (int l = 0; l < m->cfg.n_layers; l++) {
        ModelLayer *ly = &m->layers[l];

        /* Gain norm */
        gain_forward_cpu(normed, hidden, ly->gain_attn.R, ly->gain_attn.C, D);

        /* Q/K/V projections — single position */
        ternary_project_cpu(&ly->W_q, normed, q, D);
        ternary_project_cpu(&ly->W_k, normed, k_new, D);
        ternary_project_cpu(&ly->W_v, normed, v_new, D);

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
        ternary_project_cpu(&ly->W_o, attn_out, o_proj, D);

        /* Residual */
        for (int i = 0; i < D; i++)
            hidden[i] += res_scale * o_proj[i];

        /* FFN block (MoE) */
        gain_forward_cpu(normed, hidden, ly->gain_ffn.R, ly->gain_ffn.C, D);
        MoESelection sel;
        moe_route(&ly->moe.router, normed, &sel);

        /* Process top-K experts */
        float *expert_out = (float*)calloc(D, sizeof(float));
        for (int k = 0; k < MOE_TOP_K; k++) {
            int e = sel.expert_ids[k];
            float *gate_h = (float*)malloc(m->cfg.intermediate * sizeof(float));
            float *up_h = (float*)malloc(m->cfg.intermediate * sizeof(float));
            float *h = (float*)malloc(m->cfg.intermediate * sizeof(float));

            ternary_project_cpu(&ly->moe.experts[e].gate, normed, gate_h, D);
            ternary_project_cpu(&ly->moe.experts[e].up, normed, up_h, D);

            float scale = 1.0f / sqrtf((float)D);
            for (int i = 0; i < m->cfg.intermediate; i++) {
                gate_h[i] *= scale;
                up_h[i] *= scale;
            }

            /* Squared ReLU */
            for (int i = 0; i < m->cfg.intermediate; i++) {
                float g = gate_h[i];
                gate_h[i] = (g > 0.0f) ? g * g : 0.0f;
            }

            /* Hadamard product */
            for (int i = 0; i < m->cfg.intermediate; i++)
                h[i] = gate_h[i] * up_h[i];

            /* Down projection */
            float *down_out = (float*)malloc(D * sizeof(float));
            ternary_project_cpu(&ly->moe.experts[e].down, h, down_out, m->cfg.intermediate);

            /* Weighted combine (down projection scaled by 1/sqrt(INTER)) */
            float w = sel.expert_weights[k];
            float ds = 1.0f / sqrtf((float)m->cfg.intermediate);
            for (int i = 0; i < D; i++)
                expert_out[i] += w * ds * down_out[i];

            free(gate_h); free(up_h); free(h); free(down_out);
        }

        /* Residual add (scaled by 1/sqrt(D) like attention residual) */
        {
            for (int i = 0; i < D; i++)
                hidden[i] += res_scale * expert_out[i];
        }
        free(expert_out);
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
