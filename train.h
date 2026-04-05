/*
 * train.h — Training Infrastructure for {2,3} Architecture
 *
 * Layer 4: where the model LEARNS.
 *
 * Key ideas:
 *   - Latent float weights alongside ternary projections
 *   - STE (straight-through estimator): forward uses ternary,
 *     backward pretends weights are continuous floats
 *   - Adam optimizer on latent weights
 *   - Cross-entropy loss on byte prediction (256-way)
 *   - Logit clipping to prevent exp(114K) = inf
 *
 * What's trainable:
 *   - Embedding (float) — normal gradients
 *   - Gain capacity C (float) — normal gradients
 *   - MoE router W (float) — normal gradients
 *   - All ternary projections — STE through latent floats
 *
 * What's NOT trainable:
 *   - Gain reservoir R — running state, not a parameter
 *   - RoPE tables — fixed positional encoding
 *
 * Isaac & CC — March 2026
 */

#ifndef TRAIN_H
#define TRAIN_H

#include "model.h"
#include "two3_tiled.h"
#include <math.h>
#include <string.h>

/* ═══════════════════════════════════════════════════════
 * Muon optimizer GPU — wrapper for train.h integration
 *
 * When TWO3_MUON_GPU is defined, use GPU Newton-Schulz
 * via cuBLAS. Requires cublas linked and two3_muon_gpu_init
 * called on backward_ctx.
 * ═══════════════════════════════════════════════════════ */

#if defined(TWO3_MUON_GPU) || defined(TWO3_GPU_RESIDENT)
#include <cuda_runtime.h>
#include <stdio.h>

#ifndef CUDA_CHECK
#define CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(1); \
    } \
} while(0)
#endif

static void check_cuda(cudaError_t err, const char *msg) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error at %s: %s\n", msg, cudaGetErrorString(err));
        exit(1);
    }
}

#ifdef TWO3_MUON_GPU
/* Muon update on GPU for a single weight tensor.
 * H2D / D2H staging uses d_W_latent and d_dW (safe after backward, before next forward).
 * Newton-Schulz scratch (d_muon_XXtX, d_muon_XXtXXtX) must not alias params/grads. */
static void muon_update_gpu_tensor(
    Two3BackwardCtx *ctx,
    float *params_host,      /* [rows × cols] host, read/write */
    const float *grads_host, /* [rows × cols] host, read */
    float *host_momentum,    /* [rows × cols] host MuonState.momentum */
    int rows, int cols,
    float lr, float momentum_beta, float weight_decay
) {
    int size = rows * cols;
    size_t bytes = size * sizeof(float);

    float *d_params = ctx->d_W_latent;
    float *d_grads = ctx->d_dW;

    check_cuda(cudaMemcpy(ctx->d_muon_momentum, host_momentum, bytes,
                          cudaMemcpyHostToDevice),
               "cudaMemcpy momentum H2D");
    check_cuda(cudaMemcpy(d_params, params_host, bytes, cudaMemcpyHostToDevice),
               "cudaMemcpy params H2D");
    check_cuda(cudaMemcpy(d_grads, grads_host, bytes, cudaMemcpyHostToDevice),
               "cudaMemcpy grads H2D");

    two3_muon_update_gpu(ctx, d_params, d_grads, rows, cols,
                         lr, momentum_beta, weight_decay);

    check_cuda(cudaMemcpy(params_host, d_params, bytes, cudaMemcpyDeviceToHost),
               "cudaMemcpy params D2H");
    check_cuda(cudaMemcpy(host_momentum, ctx->d_muon_momentum, bytes,
                          cudaMemcpyDeviceToHost),
               "cudaMemcpy momentum D2H");
}

#endif /* TWO3_MUON_GPU (muon_update_gpu_tensor) */

#endif /* TWO3_MUON_GPU || TWO3_GPU_RESIDENT */

#define LOGIT_CLIP 30.0f

static void clip_logits(float *logits, int n) {
    for (int i = 0; i < n; i++) {
        if (logits[i] > LOGIT_CLIP) logits[i] = LOGIT_CLIP;
        if (logits[i] < -LOGIT_CLIP) logits[i] = -LOGIT_CLIP;
    }
}

/* ═══════════════════════════════════════════════════════
 * Cross-entropy loss for byte prediction
 *
 * L = -log(softmax(logits)[target])
 * dL/d(logits[i]) = softmax[i] - (i == target ? 1 : 0)
 * ═══════════════════════════════════════════════════════ */

static float cross_entropy_loss(
    const float *logits,    /* [256] */
    int target,             /* target byte 0-255 */
    float *d_logits         /* [256] output gradient */
) {
    float clipped[256];
    memcpy(clipped, logits, 256 * sizeof(float));
    clip_logits(clipped, 256);

    /* Stable softmax */
    float max_l = clipped[0];
    for (int i = 1; i < 256; i++)
        if (clipped[i] > max_l) max_l = clipped[i];

    float sum_exp = 0;
    float probs[256];
    for (int i = 0; i < 256; i++) {
        probs[i] = expf(clipped[i] - max_l);
        sum_exp += probs[i];
    }
    for (int i = 0; i < 256; i++)
        probs[i] /= sum_exp;

    /* Loss = -log(prob[target]) */
    float loss = -logf(probs[target] + 1e-10f);

    /* Gradient: softmax - one_hot */
    for (int i = 0; i < 256; i++)
        d_logits[i] = probs[i] - (i == target ? 1.0f : 0.0f);

    return loss;
}

/* ═══════════════════════════════════════════════════════
 * Adam optimizer state for a single parameter tensor
 * ═══════════════════════════════════════════════════════ */

typedef struct {
    float *m;       /* first moment */
    float *v;       /* second moment */
    int    size;
} AdamState;

#include "six_q.h"

static void adam_init(AdamState *s, int size) {
    s->size = size;
    s->m = (float*)calloc(size, sizeof(float));
    s->v = (float*)calloc(size, sizeof(float));
}

static void adam_free(AdamState *s) {
    free(s->m); free(s->v);
    s->m = s->v = NULL;
}

/* ═══════════════════════════════════════════════════════
 * Muon optimizer — Newton-Schulz orthogonal projection
 *
 * Preserves singular value spectrum of weight matrices.
 * For ternary weights: coherent flip patterns, not chaos.
 * Memory: 1 buffer (momentum) vs Adam's 2 (m, v).
 * Savings: ~2.6 GB for 654M params on 8GB card.
 * ═══════════════════════════════════════════════════════ */

typedef struct {
    float *momentum;  /* [rows × cols] */
    float *scratch;   /* [cols × cols] pre-allocated NS scratch */
    int    rows;      /* M dimension */
    int    cols;      /* K dimension */
} MuonState;

static void muon_init(MuonState *s, int rows, int cols) {
    s->rows = rows;
    s->cols = cols;
    s->momentum = (float*)calloc(rows * cols, sizeof(float));
    s->scratch  = (float*)calloc(cols * cols, sizeof(float));
}

static void muon_free(MuonState *s) {
    free(s->momentum); s->momentum = NULL;
    free(s->scratch);  s->scratch  = NULL;
}

/* Newton-Schulz iteration: 5 steps to orthogonalize.
 * Input: G [rows × cols] gradient matrix, modified in-place.
 * Scratch: [cols × cols] temporary for X^T @ X.
 * Coefficients from Muon paper: a=3.4445, b=-4.7750, c=2.0315 */
static void newton_schulz_orthogonalize(
    float *G, int rows, int cols, float *scratch
) {
    const float a = 3.4445f, b = -4.7750f, c = 2.0315f;

    /* Normalize G by Frobenius norm */
    float norm_sq = 0;
    for (int i = 0; i < rows * cols; i++)
        norm_sq += G[i] * G[i];
    float norm = sqrtf(norm_sq + 1e-30f);
    for (int i = 0; i < rows * cols; i++)
        G[i] /= norm;

    float *XtX = scratch;  /* [cols × cols] */

    /* Pre-allocate G_new once outside loop */
    float *G_new = (float*)malloc(rows * cols * sizeof(float));

    /* 5 iterations of polynomial orthogonalization */
    for (int iter = 0; iter < 5; iter++) {
        /* Compute X^T @ X */
        memset(XtX, 0, cols * cols * sizeof(float));
        for (int i = 0; i < cols; i++)
            for (int j = 0; j < cols; j++)
                for (int k = 0; k < rows; k++)
                    XtX[i * cols + j] += G[k * cols + i] * G[k * cols + j];

        /* Compute X @ XtX and X @ XtX @ XtX inline */
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                /* (X @ XtX)[i][j] */
                float x_xtx = 0;
                for (int k = 0; k < cols; k++)
                    x_xtx += G[i * cols + k] * XtX[k * cols + j];

                /* (X @ XtX @ XtX)[i][j] */
                float x_xtx_xtx = 0;
                for (int k = 0; k < cols; k++) {
                    float xtx_xtx_kj = 0;
                    for (int m = 0; m < cols; m++)
                        xtx_xtx_kj += XtX[k * cols + m] * XtX[m * cols + j];
                    x_xtx_xtx += G[i * cols + k] * xtx_xtx_kj;
                }

                G_new[i * cols + j] = a * G[i * cols + j] + b * x_xtx + c * x_xtx_xtx;
            }
        }
        memcpy(G, G_new, rows * cols * sizeof(float));
    }
    free(G_new);
}

/* Muon update step — replaces Adam for ternary weights */
static void muon_update(
    float *params,         /* [rows × cols] latent weights */
    const float *grads,    /* [rows × cols] gradients */
    MuonState *s,
    float lr,
    float momentum_beta,   /* 0.95 typical */
    float weight_decay     /* 0.01 typical */
) {
    int size = s->rows * s->cols;

    /* Momentum: m = β·m + g */
    for (int i = 0; i < size; i++)
        s->momentum[i] = momentum_beta * s->momentum[i] + grads[i];

    /* Newton-Schulz orthogonalization of momentum (pre-allocated scratch) */
    newton_schulz_orthogonalize(s->momentum, s->rows, s->cols, s->scratch);

    /* Update with weight decay */
    for (int i = 0; i < size; i++)
        params[i] -= lr * (s->momentum[i] + weight_decay * params[i]);
}

/* Gradient clipping: per-element clamp THEN L2 norm clip.
 * Per-element clamp prevents one explosive flip from dominating
 * the L2 norm and starving all other gradients via norm scaling. */
#define GRAD_ELEM_CLAMP 5.0f

static float clip_grad_norm(float *grads, int size, float max_norm) {
    /* Per-element clamp first — prevents flip cascade */
    for (int i = 0; i < size; i++) {
        if (grads[i] > GRAD_ELEM_CLAMP) grads[i] = GRAD_ELEM_CLAMP;
        if (grads[i] < -GRAD_ELEM_CLAMP) grads[i] = -GRAD_ELEM_CLAMP;
    }
    /* Then L2 norm clip */
    float norm_sq = 0;
    for (int i = 0; i < size; i++)
        norm_sq += grads[i] * grads[i];
    float norm = sqrtf(norm_sq);
    if (norm > max_norm) {
        float scale = max_norm / norm;
        for (int i = 0; i < size; i++)
            grads[i] *= scale;
    }
    return norm;
}

/* Muon (NS) updates use orthogonalized momentum — same LR as SGD can diverge; clip a bit tighter too. */
#if defined(TWO3_MUON_GPU) || defined(TWO3_USE_MUON_TERNARY)
#define GRAD_CLIP_NORM 0.5f
#else
#define GRAD_CLIP_NORM 1.0f
#endif

/* Adam update: modifies params in-place */
static void adam_update(
    float *params,
    const float *grads,
    AdamState *s,
    int step,           /* 1-indexed */
    float lr,           /* learning rate */
    float beta1,        /* 0.9 */
    float beta2,        /* 0.999 */
    float eps           /* 1e-8 */
) {
    float b1_corr = 1.0f / (1.0f - powf(beta1, (float)step));
    float b2_corr = 1.0f / (1.0f - powf(beta2, (float)step));

    for (int i = 0; i < s->size; i++) {
        float g = grads[i];
        s->m[i] = beta1 * s->m[i] + (1.0f - beta1) * g;
        s->v[i] = beta2 * s->v[i] + (1.0f - beta2) * g * g;
        float m_hat = s->m[i] * b1_corr;
        float v_hat = s->v[i] * b2_corr;
        float update = lr * m_hat / (sqrtf(v_hat) + eps);

        /* CFL: update cannot exceed ternary grid spacing (0.33).
         * Without this, Adam's adaptive denominator can teleport
         * a weight past multiple ternary boundaries in one step.
         * One grid cell per tick. Same as FDTD Yee grid. */
        if (update >  0.1f) update =  0.1f;
        if (update < -0.1f) update = -0.1f;
        params[i] -= update;
    }
}

/* ═══════════════════════════════════════════════════════
 * STE: ternary quantization + straight-through backward
 *
 * Forward: w_ternary = quantize(w_latent)
 *   w > +threshold → +1
 *   w < -threshold → -1
 *   else           →  0
 *
 * Backward: dL/dw_latent = dL/dw_ternary (pass through)
 *   BUT: clip gradient to zero if |w_latent| > 1.5
 *   (prevents latent weights from drifting too far)
 * ═══════════════════════════════════════════════════════ */

#define STE_THRESHOLD   0.33f    /* |w| > 0.33 → ternary ±1 */
#define STE_CLIP        1.5f     /* clip STE gradient beyond this */

/* Quantize latent float to ternary value */
static float ternary_quantize(float w) {
    if (w > STE_THRESHOLD) return 1.0f;
    if (w < -STE_THRESHOLD) return -1.0f;
    return 0.0f;
}

/* STE backward: pass gradient through, clip if latent too far */
static float ste_backward(float grad, float w_latent) {
    if (fabsf(w_latent) > STE_CLIP) return 0.0f;
    return grad;
}

/* ═══════════════════════════════════════════════════════
 * Trainable model — latent float weights for STE
 *
 * Structure mirrors Model but adds float latent weights
 * and Adam states for every trainable parameter.
 * Ternary weights (W_q/k/v/o, expert gate/up/down) use Muon.
 * Float weights (embedding, router, gain C) use Adam.
 * ═══════════════════════════════════════════════════════ */

typedef struct {
    float *W_q;     /* [dim × dim] latent */
    float *W_k;     /* [dim × kv_dim] latent */
    float *W_v;     /* [dim × kv_dim] latent */
    float *W_o;     /* [dim × dim] latent */

    /* MoE expert latent weights */
    float *expert_gate[MOE_NUM_EXPERTS];  /* [intermediate × dim] */
    float *expert_up[MOE_NUM_EXPERTS];    /* [intermediate × dim] */
    float *expert_down[MOE_NUM_EXPERTS];  /* [dim × intermediate] */

    /* Gradient accumulators (persist across batch for accumulation) */
    float *grad_Wq, *grad_Wk, *grad_Wv, *grad_Wo;
    float *grad_gate[MOE_NUM_EXPERTS];
    float *grad_up[MOE_NUM_EXPERTS];
    float *grad_down[MOE_NUM_EXPERTS];

#if defined(TWO3_MUON_GPU) || defined(TWO3_USE_MUON_TERNARY)
    /* Muon states for ternary weights (momentum only, half memory) */
    MuonState muon_Wq, muon_Wk, muon_Wv, muon_Wo;
    MuonState muon_gate[MOE_NUM_EXPERTS];
    MuonState muon_up[MOE_NUM_EXPERTS];
    MuonState muon_down[MOE_NUM_EXPERTS];
#else
    /* Adam states for ternary weights (default path) */
    AdamState adam_Wq, adam_Wk, adam_Wv, adam_Wo;
    AdamState adam_gate[MOE_NUM_EXPERTS];
    AdamState adam_up[MOE_NUM_EXPERTS];
    AdamState adam_down[MOE_NUM_EXPERTS];
#endif
} TrainableLayerWeights;

typedef struct {
    Model         model;     /* the actual model (ternary weights) */
    ModelConfig   cfg;

    /* Latent float weights (what Adam actually updates) */
    float *latent_embed;     /* [256 × dim] */
    TrainableLayerWeights *layer_weights;  /* [n_layers] */

    /* Adam states for non-ternary params */
    AdamState adam_embed;
    AdamState *adam_router;   /* [n_layers] for router W */
    AdamState *adam_gain_C_attn;  /* [n_layers] */
    AdamState *adam_gain_C_ffn;   /* [n_layers] */
    AdamState adam_gain_C_final;

    /* Pre-allocated GPU backward context (eliminates per-call malloc) */
    Two3BackwardCtx backward_ctx;

#ifdef TWO3_GPU_RESIDENT
    /* GPU-resident latent weights — all ternary latents live on device.
     * Single allocation, layers index into it. Eliminates H2D per requantize. */
    float *d_latent_pool;         /* [total_ternary_params] device */
    float **d_latent_layer_ptrs;  /* [n_layers] array of offsets into pool */
    int   *layer_param_offsets;   /* [n_layers+1] cumulative param counts */
    int    total_ternary_params;
    float *d_grad_buf;            /* [max_layer_params] device — per-layer scratch */
    float *d_adam_m;              /* [max_layer_params] device — Adam first moment */
    float *d_adam_v;              /* [max_layer_params] device — Adam second moment */
    float *h_norm_scratch;        /* host scratch for norm reduction */
    float *d_norm_scratch;        /* device scratch for norm reduction */
    float *h_staging;             /* [max_layer_params] host staging for bulk H2D */
    float *h_staging2;            /* [max_layer_params] second staging buffer */
    int    max_layer_params;
#endif

    /* Training hyperparams */
    float lr;
    float beta1, beta2, eps;
    int   step;              /* global step counter */

    /* Gradient accumulators for non-STE params */
    float *grad_embed;       /* [256 × dim] */
    float **grad_router;     /* [n_layers][dim × MOE_NUM_EXPERTS] */
    float **grad_gain_C_attn;
    float **grad_gain_C_ffn;
    float *grad_gain_C_final;

#ifdef TWO3_LAYER_SKIP
    LayerSkipState layer_skip;
#endif

#ifdef TWO3_FP_EMBED
    int fp_corpus_offset;  /* set by caller before forward_backward */
    /* Latent weights for fp projections (trainable) */
    float *fp_latent_Wx, *fp_latent_Wy, *fp_latent_Wz, *fp_latent_Wt;
    /* Gradient accumulators for fp projections */
    float *fp_grad_Wx, *fp_grad_Wy, *fp_grad_Wz, *fp_grad_Wt;
    /* Adam states for fp projections */
    AdamState fp_adam_Wx, fp_adam_Wy, fp_adam_Wz, fp_adam_Wt;
#endif
} TrainableModel;

/* ═══════════════════════════════════════════════════════
 * Init / Free
 * ═══════════════════════════════════════════════════════ */

static void trainable_model_init(TrainableModel *tm, ModelConfig cfg) {
    tm->cfg = cfg;
    tm->lr = 3e-3f;   /* 10x standard — STE needs larger steps to cross ternary boundaries */
    tm->beta1 = 0.9f;
    tm->beta2 = 0.999f;
    tm->eps = 1e-8f;
    tm->step = 0;

    int D = cfg.dim;
    int KV = cfg.n_kv_heads * cfg.head_dim;
    int INTER = cfg.intermediate;
    int L = cfg.n_layers;

    /* Initialize the base model with random weights */
    srand(42);
    model_init(&tm->model, cfg);

    /* Copy embedding as latent (it's already float) */
    tm->latent_embed = (float*)malloc(256 * D * sizeof(float));
    memcpy(tm->latent_embed, tm->model.embed, 256 * D * sizeof(float));
    adam_init(&tm->adam_embed, 256 * D);
    tm->grad_embed = (float*)calloc(256 * D, sizeof(float));

    /* Per-layer latent weights and Adam states */
    tm->layer_weights = (TrainableLayerWeights*)calloc(L, sizeof(TrainableLayerWeights));
    tm->adam_router = (AdamState*)calloc(L, sizeof(AdamState));
    tm->adam_gain_C_attn = (AdamState*)calloc(L, sizeof(AdamState));
    tm->adam_gain_C_ffn = (AdamState*)calloc(L, sizeof(AdamState));
    tm->grad_router = (float**)calloc(L, sizeof(float*));
    tm->grad_gain_C_attn = (float**)calloc(L, sizeof(float*));
    tm->grad_gain_C_ffn = (float**)calloc(L, sizeof(float*));

    for (int l = 0; l < L; l++) {
        TrainableLayerWeights *tw = &tm->layer_weights[l];

        /* Initialize latent weights near ternary values {-1, 0, +1}.
         * Same distribution as make_random_ternary() but with small noise
         * so STE threshold (0.33) produces the right quantization. */
        float noise = 0.05f;  /* small perturbation around ternary centers */

        tw->W_q = (float*)malloc(D * D * sizeof(float));
        tw->W_k = (float*)malloc(KV * D * sizeof(float));
        tw->W_v = (float*)malloc(KV * D * sizeof(float));
        tw->W_o = (float*)malloc(D * D * sizeof(float));

        #define INIT_TERNARY_LATENT(arr, sz) do { \
            for (int _i = 0; _i < (sz); _i++) { \
                float r = (float)rand() / (float)RAND_MAX; \
                float center; \
                if (r < 0.25f) center = 0.0f; \
                else if (r < 0.625f) center = 1.0f; \
                else center = -1.0f; \
                (arr)[_i] = center + noise * (2.0f * (float)rand() / RAND_MAX - 1.0f); \
            } \
        } while(0)

        INIT_TERNARY_LATENT(tw->W_q, D * D);
        INIT_TERNARY_LATENT(tw->W_k, KV * D);
        INIT_TERNARY_LATENT(tw->W_v, KV * D);
        INIT_TERNARY_LATENT(tw->W_o, D * D);

#if defined(TWO3_MUON_GPU) || defined(TWO3_USE_MUON_TERNARY)
        muon_init(&tw->muon_Wq, D, D);
        muon_init(&tw->muon_Wk, KV, D);
        muon_init(&tw->muon_Wv, KV, D);
        muon_init(&tw->muon_Wo, D, D);
#else
        adam_init(&tw->adam_Wq, D * D);
        adam_init(&tw->adam_Wk, KV * D);
        adam_init(&tw->adam_Wv, KV * D);
        adam_init(&tw->adam_Wo, D * D);
#endif

        /* Persistent gradient accumulators for batch accumulation */
        tw->grad_Wq = (float*)calloc(D * D, sizeof(float));
        tw->grad_Wk = (float*)calloc(KV * D, sizeof(float));
        tw->grad_Wv = (float*)calloc(KV * D, sizeof(float));
        tw->grad_Wo = (float*)calloc(D * D, sizeof(float));

        for (int e = 0; e < MOE_NUM_EXPERTS; e++) {
            tw->expert_gate[e] = (float*)malloc(INTER * D * sizeof(float));
            tw->expert_up[e]   = (float*)malloc(INTER * D * sizeof(float));
            tw->expert_down[e] = (float*)malloc(D * INTER * sizeof(float));

            INIT_TERNARY_LATENT(tw->expert_gate[e], INTER * D);
            INIT_TERNARY_LATENT(tw->expert_up[e], INTER * D);
            INIT_TERNARY_LATENT(tw->expert_down[e], D * INTER);

#if defined(TWO3_MUON_GPU) || defined(TWO3_USE_MUON_TERNARY)
            muon_init(&tw->muon_gate[e], INTER, D);
            muon_init(&tw->muon_up[e], INTER, D);
            muon_init(&tw->muon_down[e], D, INTER);
#else
            adam_init(&tw->adam_gate[e], INTER * D);
            adam_init(&tw->adam_up[e], INTER * D);
            adam_init(&tw->adam_down[e], D * INTER);
#endif

            tw->grad_gate[e] = (float*)calloc(INTER * D, sizeof(float));
            tw->grad_up[e]   = (float*)calloc(INTER * D, sizeof(float));
            tw->grad_down[e] = (float*)calloc(D * INTER, sizeof(float));
        }

        /* Router and gain Adam states */
        adam_init(&tm->adam_router[l], D * MOE_NUM_EXPERTS);
        adam_init(&tm->adam_gain_C_attn[l], D);
        adam_init(&tm->adam_gain_C_ffn[l], D);

        tm->grad_router[l] = (float*)calloc(D * MOE_NUM_EXPERTS, sizeof(float));
        tm->grad_gain_C_attn[l] = (float*)calloc(D, sizeof(float));
        tm->grad_gain_C_ffn[l] = (float*)calloc(D, sizeof(float));
    }

    adam_init(&tm->adam_gain_C_final, D);
    tm->grad_gain_C_final = (float*)calloc(D, sizeof(float));

    /* Pre-allocate GPU backward buffers.
     * max_M = max(D, KV, INTER), max_K = max(D, INTER).
     * Covers all projection dimensions: D×D, KV×D, INTER×D, D×INTER. */
    {
        int max_M = D > INTER ? D : INTER;
        if (KV > max_M) max_M = KV;
        int max_K = D > INTER ? D : INTER;
        tm->backward_ctx = two3_backward_ctx_init(max_M, max_K,
            cfg.max_seq, D, KV, cfg.n_heads);
    }

#ifdef TWO3_MUON_GPU
    /* Initialize Muon GPU buffers (cuBLAS handle, device buffers) */
    {
        int max_M = D > INTER ? D : INTER;
        if (KV > max_M) max_M = KV;
        int max_K = D > INTER ? D : INTER;
        printf("[init] muon_gpu_init max_M=%d max_K=%d\n", max_M, max_K); fflush(stdout);
        two3_muon_gpu_init(&tm->backward_ctx, max_M, max_K);
    }
#endif

#ifdef TWO3_GPU_RESIDENT
    /* GPU-resident latent weights: allocate device pool, upload latents */
    {
        /* Count total ternary params per layer and overall */
        int per_layer = D*D + KV*D + KV*D + D*D  /* Q,K,V,O */
                      + MOE_NUM_EXPERTS * (INTER*D + INTER*D + D*INTER);  /* experts */
        tm->total_ternary_params = per_layer * L;
        size_t pool_bytes = (size_t)tm->total_ternary_params * sizeof(float);

        printf("[gpu-resident] allocating %.1f MB device latent pool\n",
               pool_bytes / 1e6); fflush(stdout);
        CUDA_CHECK(cudaMalloc(&tm->d_latent_pool, pool_bytes));

        /* Build layer offset table and upload initial latent weights */
        tm->layer_param_offsets = (int*)malloc((L + 1) * sizeof(int));
        tm->d_latent_layer_ptrs = (float**)malloc(L * sizeof(float*));
        int offset = 0;
        for (int l = 0; l < L; l++) {
            tm->layer_param_offsets[l] = offset;
            tm->d_latent_layer_ptrs[l] = tm->d_latent_pool + offset;
            TrainableLayerWeights *tw = &tm->layer_weights[l];

            /* Upload each weight tensor into the contiguous pool */
            float *d = tm->d_latent_pool + offset;
            cudaMemcpy(d, tw->W_q, D*D*sizeof(float), cudaMemcpyHostToDevice); d += D*D;
            cudaMemcpy(d, tw->W_k, KV*D*sizeof(float), cudaMemcpyHostToDevice); d += KV*D;
            cudaMemcpy(d, tw->W_v, KV*D*sizeof(float), cudaMemcpyHostToDevice); d += KV*D;
            cudaMemcpy(d, tw->W_o, D*D*sizeof(float), cudaMemcpyHostToDevice); d += D*D;
            for (int e = 0; e < MOE_NUM_EXPERTS; e++) {
                cudaMemcpy(d, tw->expert_gate[e], INTER*D*sizeof(float), cudaMemcpyHostToDevice); d += INTER*D;
                cudaMemcpy(d, tw->expert_up[e], INTER*D*sizeof(float), cudaMemcpyHostToDevice); d += INTER*D;
                cudaMemcpy(d, tw->expert_down[e], D*INTER*sizeof(float), cudaMemcpyHostToDevice); d += D*INTER;
            }
            offset += per_layer;
        }
        tm->layer_param_offsets[L] = offset;

        /* Per-layer device buffers for GPU Adam */
        CUDA_CHECK(cudaMalloc(&tm->d_grad_buf, per_layer * sizeof(float)));
        CUDA_CHECK(cudaMemset(tm->d_grad_buf, 0, per_layer * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&tm->d_adam_m, per_layer * sizeof(float)));
        CUDA_CHECK(cudaMemset(tm->d_adam_m, 0, per_layer * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&tm->d_adam_v, per_layer * sizeof(float)));
        CUDA_CHECK(cudaMemset(tm->d_adam_v, 0, per_layer * sizeof(float)));

        /* Norm reduction scratch */
        tm->h_norm_scratch = (float*)malloc(1024 * sizeof(float));
        CUDA_CHECK(cudaMalloc(&tm->d_norm_scratch, 1024 * sizeof(float)));

        /* Host staging buffers for bulk H2D per layer */
        tm->max_layer_params = per_layer;
        tm->h_staging = (float*)malloc(per_layer * sizeof(float));
        tm->h_staging2 = (float*)malloc(per_layer * sizeof(float));

        printf("[gpu-resident] %d params, %d layers, %.1f MB on device\n",
               tm->total_ternary_params, L, pool_bytes / 1e6); fflush(stdout);
    }
#endif

#ifdef TWO3_LAYER_SKIP
    layer_skip_init(&tm->layer_skip, L);
#endif

#ifdef TWO3_FP_EMBED
    /* Fingerprint projection trainable state */
    {
        int qdim = D / 4;
        int fp_size = qdim * 1024;

        #define INIT_FP_LATENT(name) \
            tm->fp_latent_##name = (float*)malloc(fp_size * sizeof(float)); \
            for (int i = 0; i < fp_size; i++) \
                tm->fp_latent_##name[i] = 0.6f * (2.0f * (float)rand() / RAND_MAX - 1.0f); \
            tm->fp_grad_##name = (float*)calloc(fp_size, sizeof(float)); \
            adam_init(&tm->fp_adam_##name, fp_size);

        INIT_FP_LATENT(Wx);
        INIT_FP_LATENT(Wy);
        INIT_FP_LATENT(Wz);
        INIT_FP_LATENT(Wt);
        #undef INIT_FP_LATENT

        tm->fp_corpus_offset = 0;
        printf("[init] fp embed: 4 × [%d × 1024] ternary projections\n", qdim);
    }
#endif

    printf("[init] trainable_model_init complete\n"); fflush(stdout);
}

/* Re-quantize all latent weights → ternary for the model.
 * Called after init and after each Adam step. */
static void trainable_requantize(TrainableModel *tm) {
    int D = tm->cfg.dim;
    int KV = tm->cfg.n_kv_heads * tm->cfg.head_dim;
    int INTER = tm->cfg.intermediate;

    /* Re-quantize embedding (it stays float, just sync) */
    memcpy(tm->model.embed, tm->latent_embed, 256 * D * sizeof(float));

    /* Fused GPU requantize: latent → ternary → packed in one pass.
     * No malloc/free per matrix. No printf spam. No host round-trip.
     * Uses backward_ctx device buffers as scratch. */
    for (int l = 0; l < tm->cfg.n_layers; l++) {
        TrainableLayerWeights *tw = &tm->layer_weights[l];
        ModelLayer *ly = &tm->model.layers[l];

#ifdef TWO3_GPU_RESIDENT
        /* GPU-resident: read from device latent pool directly */
        float *d = tm->d_latent_layer_ptrs[l];
        requantize_gpu(&tm->backward_ctx, NULL, &ly->W_q, D, D, STE_THRESHOLD, d); d += D*D;
        requantize_gpu(&tm->backward_ctx, NULL, &ly->W_k, KV, D, STE_THRESHOLD, d); d += KV*D;
        requantize_gpu(&tm->backward_ctx, NULL, &ly->W_v, KV, D, STE_THRESHOLD, d); d += KV*D;
        requantize_gpu(&tm->backward_ctx, NULL, &ly->W_o, D, D, STE_THRESHOLD, d); d += D*D;
        for (int e = 0; e < MOE_NUM_EXPERTS; e++) {
            requantize_gpu(&tm->backward_ctx, NULL, &ly->moe.experts[e].gate, INTER, D, STE_THRESHOLD, d); d += INTER*D;
            requantize_gpu(&tm->backward_ctx, NULL, &ly->moe.experts[e].up, INTER, D, STE_THRESHOLD, d); d += INTER*D;
            requantize_gpu(&tm->backward_ctx, NULL, &ly->moe.experts[e].down, D, INTER, STE_THRESHOLD, d); d += D*INTER;
        }
#else
        /* Legacy: upload from host */
        requantize_gpu(&tm->backward_ctx, tw->W_q, &ly->W_q, D, D, STE_THRESHOLD, NULL);
        requantize_gpu(&tm->backward_ctx, tw->W_k, &ly->W_k, KV, D, STE_THRESHOLD, NULL);
        requantize_gpu(&tm->backward_ctx, tw->W_v, &ly->W_v, KV, D, STE_THRESHOLD, NULL);
        requantize_gpu(&tm->backward_ctx, tw->W_o, &ly->W_o, D, D, STE_THRESHOLD, NULL);
        for (int e = 0; e < MOE_NUM_EXPERTS; e++) {
            requantize_gpu(&tm->backward_ctx, tw->expert_gate[e],
                          &ly->moe.experts[e].gate, INTER, D, STE_THRESHOLD, NULL);
            requantize_gpu(&tm->backward_ctx, tw->expert_up[e],
                          &ly->moe.experts[e].up, INTER, D, STE_THRESHOLD, NULL);
            requantize_gpu(&tm->backward_ctx, tw->expert_down[e],
                          &ly->moe.experts[e].down, D, INTER, STE_THRESHOLD, NULL);
        }
#endif
    }
}

static void trainable_model_free(TrainableModel *tm) {
    int L = tm->cfg.n_layers;

    free(tm->latent_embed);
    adam_free(&tm->adam_embed);
    free(tm->grad_embed);

    for (int l = 0; l < L; l++) {
        TrainableLayerWeights *tw = &tm->layer_weights[l];
        free(tw->W_q); free(tw->W_k); free(tw->W_v); free(tw->W_o);
        free(tw->grad_Wq); free(tw->grad_Wk); free(tw->grad_Wv); free(tw->grad_Wo);
#if defined(TWO3_MUON_GPU) || defined(TWO3_USE_MUON_TERNARY)
        muon_free(&tw->muon_Wq); muon_free(&tw->muon_Wk);
        muon_free(&tw->muon_Wv); muon_free(&tw->muon_Wo);
#else
        adam_free(&tw->adam_Wq); adam_free(&tw->adam_Wk);
        adam_free(&tw->adam_Wv); adam_free(&tw->adam_Wo);
#endif

        for (int e = 0; e < MOE_NUM_EXPERTS; e++) {
            free(tw->expert_gate[e]); free(tw->expert_up[e]); free(tw->expert_down[e]);
            free(tw->grad_gate[e]); free(tw->grad_up[e]); free(tw->grad_down[e]);
#if defined(TWO3_MUON_GPU) || defined(TWO3_USE_MUON_TERNARY)
            muon_free(&tw->muon_gate[e]); muon_free(&tw->muon_up[e]);
            muon_free(&tw->muon_down[e]);
#else
            adam_free(&tw->adam_gate[e]); adam_free(&tw->adam_up[e]);
            adam_free(&tw->adam_down[e]);
#endif
        }

        adam_free(&tm->adam_router[l]);
        adam_free(&tm->adam_gain_C_attn[l]);
        adam_free(&tm->adam_gain_C_ffn[l]);
        free(tm->grad_router[l]);
        free(tm->grad_gain_C_attn[l]);
        free(tm->grad_gain_C_ffn[l]);
    }

    free(tm->layer_weights);
    free(tm->adam_router);
    free(tm->adam_gain_C_attn);
    free(tm->adam_gain_C_ffn);
    free(tm->grad_router);
    free(tm->grad_gain_C_attn);
    free(tm->grad_gain_C_ffn);

    adam_free(&tm->adam_gain_C_final);
    free(tm->grad_gain_C_final);

#ifdef TWO3_LAYER_SKIP
    layer_skip_free(&tm->layer_skip);
#endif

#ifdef TWO3_GPU_RESIDENT
    cudaFree(tm->d_latent_pool);
    cudaFree(tm->d_grad_buf);
    cudaFree(tm->d_adam_m);
    cudaFree(tm->d_adam_v);
    cudaFree(tm->d_norm_scratch);
    free(tm->h_norm_scratch);
    free(tm->h_staging);
    free(tm->h_staging2);
    free(tm->d_latent_layer_ptrs);
    free(tm->layer_param_offsets);
#endif

#ifdef TWO3_MUON_GPU
    /* Free Muon GPU buffers before backward_ctx free */
    two3_muon_gpu_free(&tm->backward_ctx);
#endif

    two3_backward_ctx_free(&tm->backward_ctx);

    model_free(&tm->model);
}

/* ═══════════════════════════════════════════════════════
 * Backward pass — CPU reference
 *
 * This is the training forward+backward for a single sequence.
 * Saves all activations needed for backprop, then propagates
 * gradients back through every layer.
 *
 * The STE magic: ternary projections use the ternary weights
 * in forward, but the gradient goes to the latent floats.
 * For Y = X @ W_ternary^T, dL/dX = dL/dY @ W_ternary (use ternary)
 * but dL/dW_latent = dL/dY^T @ X (gradient on latent float).
 * ═══════════════════════════════════════════════════════ */

/* CPU ternary matmul using latent weights (for gradient computation).
 * This does Y = X @ W^T using the QUANTIZED ternary values but stores
 * the result — same as ternary_project_cpu but we also need the
 * ternary weight values for backward. */

/* Backward through ternary projection (STE):
 * Forward was: Y[m] = sum_k W_ternary[m][k] * X[k]
 * dL/dX[k] = sum_m dL/dY[m] * W_ternary[m][k]  (use ternary for input grad)
 * dL/dW_latent[m][k] = dL/dY[m] * X[k]          (STE: grad to latent)
 */
static void ternary_project_backward_cpu(
    const float *dY,           /* [rows] gradient from above */
    const float *X,            /* [cols] saved input */
    const float *W_latent,     /* [rows × cols] latent float weights */
    float *dX,                 /* [cols] gradient to pass back (ACCUMULATE) */
    float *dW_latent,          /* [rows × cols] gradient for latent (ACCUMULATE) */
    int rows, int cols
) {
    /* dL/dX: use ternary-quantized weights */
    for (int k = 0; k < cols; k++) {
        float sum = 0;
        for (int m = 0; m < rows; m++) {
            float wt = ternary_quantize(W_latent[m * cols + k]);
            sum += dY[m] * wt;
        }
        dX[k] += sum;
    }

    /* dL/dW_latent: STE — accumulate gradient on latent float */
    for (int m = 0; m < rows; m++) {
        for (int k = 0; k < cols; k++) {
            float g = dY[m] * X[k];
            /* STE clip: zero gradient if latent weight too far from ternary range */
            dW_latent[m * cols + k] += ste_backward(g, W_latent[m * cols + k]);
        }
    }
}

/* GPU-accelerated ternary backward — uses pre-allocated context.
 * Needs the packed Two3Weights for the transposed ternary matmul. */
/* Batched GPU backward — S vectors in one call.
 * dY: [S, rows], X: [S, cols], dX: [S, cols] accumulate, dW: [rows, cols] accumulate */
static void ternary_project_backward_gpu_batch(
    Two3BackwardCtx *ctx,
    const Two3Weights *W_packed,
    const float *dY, const float *X, const float *W_latent,
    float *dX, float *dW_latent,
    int S, int rows, int cols
) {
    /* No gradient scaling — let 1/sqrt(K) from forward chain rule flow naturally.
     * Adam is scale-invariant once v-estimate converges. Scale the LR instead. */
    two3_backward_fast(ctx, W_packed, dY, X, W_latent, dX, dW_latent,
                       S, rows, cols, STE_CLIP);
}

/* Single-vector backward (S=1 convenience wrapper) */
static void ternary_project_backward_gpu(
    Two3BackwardCtx *ctx,
    const Two3Weights *W_packed,
    const float *dY, const float *X, const float *W_latent,
    float *dX, float *dW_latent,
    int rows, int cols
) {
    two3_backward_fast(ctx, W_packed, dY, X, W_latent, dX, dW_latent,
                       1, rows, cols, STE_CLIP);
}

/* Backward through gain normalization (CPU version of kernel_gain_backward) */
static void gain_backward_cpu(
    float *dx,          /* [dim] gradient w.r.t. input (OUTPUT) */
    const float *dy,    /* [dim] gradient from above */
    const float *x,     /* [dim] saved input */
    const float *R,     /* [dim] reservoir at time of forward */
    float *dC,          /* [dim] gradient w.r.t. capacity (ACCUMULATE) */
    int dim
) {
    for (int i = 0; i < dim; i++) {
        float gain = 1.0f + GAIN_ALPHA * R[i] - GAIN_BETA;
        dx[i] = dy[i] * gain;
        dC[i] += dy[i] * x[i] * GAIN_ALPHA * GAIN_GAMMA;
    }
}

/* Backward through squared ReLU */
static void squared_relu_backward_cpu(
    float *dx,          /* [n] gradient w.r.t. input */
    const float *dy,    /* [n] gradient from above */
    const float *x,     /* [n] saved pre-activation input */
    int n
) {
    for (int i = 0; i < n; i++) {
        float v = x[i];
        dx[i] = (v > 0.0f) ? dy[i] * 2.0f * v : 0.0f;
    }
}

/* Backward through causal attention for a single position.
 *
 * Forward: scores = Q @ K^T / sqrt(d), attn = softmax(scores), out = attn @ V
 * This computes dQ, dK (accumulated for all t<=pos), dV (accumulated)
 */
static void causal_attention_backward_cpu(
    const float *d_out,        /* [dim] = [n_heads * head_dim] gradient */
    const float *q,            /* [n_heads * head_dim] query at this pos */
    const float *k_store,      /* [pos+1, kv_dim] all keys */
    const float *v_store,      /* [pos+1, kv_dim] all values */
    float *dq,                 /* [n_heads * head_dim] OUTPUT (zeroed first) */
    float *dk_store,           /* [pos+1, kv_dim] ACCUMULATE */
    float *dv_store,           /* [pos+1, kv_dim] ACCUMULATE */
    int pos, int n_heads, int n_kv_heads, int head_dim
) {
    int kv_dim = n_kv_heads * head_dim;
    int heads_per_kv = n_heads / n_kv_heads;
    float scale = 1.0f / sqrtf((float)head_dim);
    int seq_len = pos + 1;

    float *scores = (float*)malloc(seq_len * sizeof(float));
    float *attn_weights = (float*)malloc(seq_len * sizeof(float));

    memset(dq, 0, n_heads * head_dim * sizeof(float));

    for (int h = 0; h < n_heads; h++) {
        int kv_h = h / heads_per_kv;

        /* Recompute forward: scores and softmax weights */
        for (int t = 0; t < seq_len; t++) {
            float dot = 0;
            for (int d = 0; d < head_dim; d++)
                dot += q[h * head_dim + d] * k_store[t * kv_dim + kv_h * head_dim + d];
            scores[t] = dot * scale;
        }

        float max_s = scores[0];
        for (int t = 1; t < seq_len; t++)
            if (scores[t] > max_s) max_s = scores[t];

        float sum_exp = 0;
        for (int t = 0; t < seq_len; t++) {
            attn_weights[t] = expf(scores[t] - max_s);
            sum_exp += attn_weights[t];
        }
        for (int t = 0; t < seq_len; t++)
            attn_weights[t] /= sum_exp;

        /* d_out for this head */
        const float *d_out_h = d_out + h * head_dim;

        /* dV: d_attn_weights[t] contributes to dV[t]
         * out[d] = sum_t attn[t] * V[t][d]
         * dV[t][d] += attn[t] * d_out[d] */
        for (int t = 0; t < seq_len; t++) {
            for (int d = 0; d < head_dim; d++)
                dv_store[t * kv_dim + kv_h * head_dim + d] +=
                    attn_weights[t] * d_out_h[d];
        }

        /* d_attn_weights[t] = sum_d d_out[d] * V[t][d] */
        float *d_attn = (float*)calloc(seq_len, sizeof(float));
        for (int t = 0; t < seq_len; t++) {
            for (int d = 0; d < head_dim; d++)
                d_attn[t] += d_out_h[d] * v_store[t * kv_dim + kv_h * head_dim + d];
        }

        /* Backward through softmax:
         * d_scores[t] = attn[t] * (d_attn[t] - sum_j attn[j] * d_attn[j]) */
        float dot_da = 0;
        for (int t = 0; t < seq_len; t++)
            dot_da += attn_weights[t] * d_attn[t];

        float *d_scores = (float*)malloc(seq_len * sizeof(float));
        for (int t = 0; t < seq_len; t++)
            d_scores[t] = attn_weights[t] * (d_attn[t] - dot_da);

        /* dQ and dK from scores = Q @ K^T * scale */
        for (int t = 0; t < seq_len; t++) {
            float ds = d_scores[t] * scale;
            for (int d = 0; d < head_dim; d++) {
                dq[h * head_dim + d] += ds * k_store[t * kv_dim + kv_h * head_dim + d];
                dk_store[t * kv_dim + kv_h * head_dim + d] += ds * q[h * head_dim + d];
            }
        }

        free(d_attn);
        free(d_scores);
    }

    free(scores);
    free(attn_weights);
}

/* Backward through MoE router:
 * logits = x @ W, then softmax, then top-2.
 * We need dW and dx. For simplicity, we backprop through
 * the selected experts only (the non-selected get zero gradient). */
static void moe_router_backward_cpu(
    const float *d_expert_weights, /* [MOE_TOP_K] gradient on expert weights */
    const MoESelection *sel,
    const float *x,                /* [dim] router input */
    const MoERouter *router,
    float *dx,                     /* [dim] ACCUMULATE */
    float *dW                      /* [dim × n_experts] ACCUMULATE */
) {
    int D = router->dim;
    int N = router->n_experts;

    /* Recompute softmax for selected experts */
    float logits[MOE_NUM_EXPERTS];
    for (int e = 0; e < N; e++) {
        float sum = 0;
        for (int d = 0; d < D; d++)
            sum += x[d] * router->W[d * N + e];
        logits[e] = sum;
    }

    float max_l = logits[0];
    for (int e = 1; e < N; e++)
        if (logits[e] > max_l) max_l = logits[e];
    float probs[MOE_NUM_EXPERTS];
    float sum_exp = 0;
    for (int e = 0; e < N; e++) {
        probs[e] = expf(logits[e] - max_l);
        sum_exp += probs[e];
    }
    for (int e = 0; e < N; e++)
        probs[e] /= sum_exp;

    /* Renormalized weights: w_k = probs[sel_k] / sum(probs[sel_*])
     * Backward through renormalization + softmax to get d_logits */
    float d_logits[MOE_NUM_EXPERTS];
    memset(d_logits, 0, sizeof(d_logits));

    /* Chain: d_expert_weights → d_probs (through renorm) → d_logits (through softmax) */
    float sel_sum = 0;
    for (int k = 0; k < MOE_TOP_K; k++)
        sel_sum += probs[sel->expert_ids[k]];

    /* d_probs from renormalization */
    float d_probs[MOE_NUM_EXPERTS];
    memset(d_probs, 0, sizeof(d_probs));
    for (int k = 0; k < MOE_TOP_K; k++) {
        int eid = sel->expert_ids[k];
        /* w_k = probs[eid] / sel_sum */
        /* dw_k/d(probs[eid]) = (sel_sum - probs[eid]) / sel_sum^2 */
        /* dw_k/d(probs[other_sel]) = -probs[eid] / sel_sum^2 */
        for (int j = 0; j < MOE_TOP_K; j++) {
            int jid = sel->expert_ids[j];
            if (jid == eid)
                d_probs[jid] += d_expert_weights[k] * (sel_sum - probs[eid]) / (sel_sum * sel_sum);
            else
                d_probs[jid] += d_expert_weights[k] * (-probs[eid]) / (sel_sum * sel_sum);
        }
    }

    /* Backward through softmax: d_logits[e] = probs[e] * (d_probs[e] - sum_j probs[j] * d_probs[j]) */
    float dot_dp = 0;
    for (int e = 0; e < N; e++)
        dot_dp += probs[e] * d_probs[e];
    for (int e = 0; e < N; e++)
        d_logits[e] = probs[e] * (d_probs[e] - dot_dp);

    /* d_logits → dW and dx */
    for (int e = 0; e < N; e++) {
        for (int d = 0; d < D; d++) {
            dW[d * N + e] += d_logits[e] * x[d];
            dx[d] += d_logits[e] * router->W[d * N + e];
        }
    }
}

/* ═══════════════════════════════════════════════════════
 * Batch training API
 *
 * Zero → accumulate N sequences → optimize once.
 *   trainable_zero_grads(&tm);
 *   for (i = 0; i < batch_size; i++)
 *       trainable_forward_backward(&tm, seq[i], len, &result);
 *   trainable_optimizer_step(&tm);
 * ═══════════════════════════════════════════════════════ */

static void trainable_zero_grads(TrainableModel *tm) {
    int D = tm->cfg.dim;
    int KV = tm->cfg.n_kv_heads * tm->cfg.head_dim;
    int INTER = tm->cfg.intermediate;
    int L = tm->cfg.n_layers;

    memset(tm->grad_embed, 0, 256 * D * sizeof(float));
    memset(tm->grad_gain_C_final, 0, D * sizeof(float));

    for (int l = 0; l < L; l++) {
        TrainableLayerWeights *tw = &tm->layer_weights[l];
        memset(tw->grad_Wq, 0, D * D * sizeof(float));
        memset(tw->grad_Wk, 0, KV * D * sizeof(float));
        memset(tw->grad_Wv, 0, KV * D * sizeof(float));
        memset(tw->grad_Wo, 0, D * D * sizeof(float));
        for (int e = 0; e < MOE_NUM_EXPERTS; e++) {
            memset(tw->grad_gate[e], 0, INTER * D * sizeof(float));
            memset(tw->grad_up[e], 0, INTER * D * sizeof(float));
            memset(tw->grad_down[e], 0, D * INTER * sizeof(float));
        }
        memset(tm->grad_router[l], 0, D * MOE_NUM_EXPERTS * sizeof(float));
        memset(tm->grad_gain_C_attn[l], 0, D * sizeof(float));
        memset(tm->grad_gain_C_ffn[l], 0, D * sizeof(float));
    }

#ifdef TWO3_FP_EMBED
    {
        int fp_size = (D / 4) * 1024;
        memset(tm->fp_grad_Wx, 0, fp_size * sizeof(float));
        memset(tm->fp_grad_Wy, 0, fp_size * sizeof(float));
        memset(tm->fp_grad_Wz, 0, fp_size * sizeof(float));
        memset(tm->fp_grad_Wt, 0, fp_size * sizeof(float));
    }
#endif
}

static void trainable_optimizer_step(TrainableModel *tm) {
    int D = tm->cfg.dim;
    int KV = tm->cfg.n_kv_heads * tm->cfg.head_dim;
    int INTER = tm->cfg.intermediate;
    int L = tm->cfg.n_layers;

    tm->step++;

    for (int l = 0; l < L; l++) {
        TrainableLayerWeights *tw = &tm->layer_weights[l];
        ModelLayer *ly = &tm->model.layers[l];

        /* Clip gradients */
        clip_grad_norm(tw->grad_Wq, D * D, GRAD_CLIP_NORM);
        clip_grad_norm(tw->grad_Wk, KV * D, GRAD_CLIP_NORM);
        clip_grad_norm(tw->grad_Wv, KV * D, GRAD_CLIP_NORM);
        clip_grad_norm(tw->grad_Wo, D * D, GRAD_CLIP_NORM);

        /* ═══════════════════════════════════════════════════════
         * Optimizer selection:
         *   TWO3_MUON_GPU     → GPU Newton-Schulz via cuBLAS (fastest)
         *   TWO3_USE_MUON_TERNARY → CPU Newton-Schulz (slow, debug only)
         *   default           → SGD (fast, production)
         * ═══════════════════════════════════════════════════════ */

#ifdef TWO3_MUON_GPU
        /* GPU Muon — Newton-Schulz on GPU via cuBLAS */
        muon_update_gpu_tensor(&tm->backward_ctx,
            tw->W_q, tw->grad_Wq, tw->muon_Wq.momentum, D, D,
            tm->lr, 0.95f, 0.01f);
        muon_update_gpu_tensor(&tm->backward_ctx,
            tw->W_k, tw->grad_Wk, tw->muon_Wk.momentum, KV, D,
            tm->lr, 0.95f, 0.01f);
        muon_update_gpu_tensor(&tm->backward_ctx,
            tw->W_v, tw->grad_Wv, tw->muon_Wv.momentum, KV, D,
            tm->lr, 0.95f, 0.01f);
        muon_update_gpu_tensor(&tm->backward_ctx,
            tw->W_o, tw->grad_Wo, tw->muon_Wo.momentum, D, D,
            tm->lr, 0.95f, 0.01f);

        for (int e = 0; e < MOE_NUM_EXPERTS; e++) {
            clip_grad_norm(tw->grad_gate[e], INTER * D, GRAD_CLIP_NORM);
            clip_grad_norm(tw->grad_up[e], INTER * D, GRAD_CLIP_NORM);
            clip_grad_norm(tw->grad_down[e], D * INTER, GRAD_CLIP_NORM);
            muon_update_gpu_tensor(&tm->backward_ctx,
                tw->expert_gate[e], tw->grad_gate[e], tw->muon_gate[e].momentum,
                INTER, D, tm->lr, 0.95f, 0.01f);
            muon_update_gpu_tensor(&tm->backward_ctx,
                tw->expert_up[e], tw->grad_up[e], tw->muon_up[e].momentum,
                INTER, D, tm->lr, 0.95f, 0.01f);
            muon_update_gpu_tensor(&tm->backward_ctx,
                tw->expert_down[e], tw->grad_down[e], tw->muon_down[e].momentum,
                D, INTER, tm->lr, 0.95f, 0.01f);
        }
#elif defined(TWO3_USE_MUON_TERNARY)
        /* CPU Muon — Newton-Schulz on CPU (slow, ~100x SGD) */
        muon_update(tw->W_q, tw->grad_Wq, &tw->muon_Wq, tm->lr, 0.95f, 0.01f);
        muon_update(tw->W_k, tw->grad_Wk, &tw->muon_Wk, tm->lr, 0.95f, 0.01f);
        muon_update(tw->W_v, tw->grad_Wv, &tw->muon_Wv, tm->lr, 0.95f, 0.01f);
        muon_update(tw->W_o, tw->grad_Wo, &tw->muon_Wo, tm->lr, 0.95f, 0.01f);

        for (int e = 0; e < MOE_NUM_EXPERTS; e++) {
            clip_grad_norm(tw->grad_gate[e], INTER * D, GRAD_CLIP_NORM);
            clip_grad_norm(tw->grad_up[e], INTER * D, GRAD_CLIP_NORM);
            clip_grad_norm(tw->grad_down[e], D * INTER, GRAD_CLIP_NORM);
            muon_update(tw->expert_gate[e], tw->grad_gate[e], &tw->muon_gate[e], tm->lr, 0.95f, 0.01f);
            muon_update(tw->expert_up[e], tw->grad_up[e], &tw->muon_up[e], tm->lr, 0.95f, 0.01f);
            muon_update(tw->expert_down[e], tw->grad_down[e], &tw->muon_down[e], tm->lr, 0.95f, 0.01f);
        }
#elif defined(TWO3_GPU_RESIDENT)
        /* GPU-resident Adam: pack layer grads+m+v, bulk H2D, GPU kernels, bulk D2H.
         * Latent weights already on device (d_latent_pool). */
        {
            int per_layer = tm->max_layer_params;
            float *d_latent = tm->d_latent_layer_ptrs[l];

            /* Pack grads into staging and H2D */
            float *s = tm->h_staging;
            memcpy(s, tw->grad_Wq, D*D*sizeof(float)); s += D*D;
            memcpy(s, tw->grad_Wk, KV*D*sizeof(float)); s += KV*D;
            memcpy(s, tw->grad_Wv, KV*D*sizeof(float)); s += KV*D;
            memcpy(s, tw->grad_Wo, D*D*sizeof(float)); s += D*D;
            for (int e = 0; e < MOE_NUM_EXPERTS; e++) {
                /* Clip expert grads on CPU before upload */
                clip_grad_norm(tw->grad_gate[e], INTER * D, GRAD_CLIP_NORM);
                clip_grad_norm(tw->grad_up[e], INTER * D, GRAD_CLIP_NORM);
                clip_grad_norm(tw->grad_down[e], D * INTER, GRAD_CLIP_NORM);
                memcpy(s, tw->grad_gate[e], INTER*D*sizeof(float)); s += INTER*D;
                memcpy(s, tw->grad_up[e], INTER*D*sizeof(float)); s += INTER*D;
                memcpy(s, tw->grad_down[e], D*INTER*sizeof(float)); s += D*INTER;
            }
            CUDA_CHECK(cudaMemcpy(tm->d_grad_buf, tm->h_staging,
                                  per_layer * sizeof(float), cudaMemcpyHostToDevice));

            /* Pack Adam m into staging and H2D */
            s = tm->h_staging;
            memcpy(s, tw->adam_Wq.m, D*D*sizeof(float)); s += D*D;
            memcpy(s, tw->adam_Wk.m, KV*D*sizeof(float)); s += KV*D;
            memcpy(s, tw->adam_Wv.m, KV*D*sizeof(float)); s += KV*D;
            memcpy(s, tw->adam_Wo.m, D*D*sizeof(float)); s += D*D;
            for (int e = 0; e < MOE_NUM_EXPERTS; e++) {
                memcpy(s, tw->adam_gate[e].m, INTER*D*sizeof(float)); s += INTER*D;
                memcpy(s, tw->adam_up[e].m, INTER*D*sizeof(float)); s += INTER*D;
                memcpy(s, tw->adam_down[e].m, D*INTER*sizeof(float)); s += D*INTER;
            }
            CUDA_CHECK(cudaMemcpy(tm->d_adam_m, tm->h_staging,
                                  per_layer * sizeof(float), cudaMemcpyHostToDevice));

            /* Pack Adam v into staging and H2D */
            s = tm->h_staging;
            memcpy(s, tw->adam_Wq.v, D*D*sizeof(float)); s += D*D;
            memcpy(s, tw->adam_Wk.v, KV*D*sizeof(float)); s += KV*D;
            memcpy(s, tw->adam_Wv.v, KV*D*sizeof(float)); s += KV*D;
            memcpy(s, tw->adam_Wo.v, D*D*sizeof(float)); s += D*D;
            for (int e = 0; e < MOE_NUM_EXPERTS; e++) {
                memcpy(s, tw->adam_gate[e].v, INTER*D*sizeof(float)); s += INTER*D;
                memcpy(s, tw->adam_up[e].v, INTER*D*sizeof(float)); s += INTER*D;
                memcpy(s, tw->adam_down[e].v, D*INTER*sizeof(float)); s += D*INTER;
            }
            CUDA_CHECK(cudaMemcpy(tm->d_adam_v, tm->h_staging,
                                  per_layer * sizeof(float), cudaMemcpyHostToDevice));

            /* GPU: grad clip (already clipped attn on CPU, experts in pack loop) */
            gpu_clip_grad_norm(tm->d_grad_buf, D*D, GRAD_CLIP_NORM, 5.0f,
                               tm->d_norm_scratch, tm->h_norm_scratch);
            gpu_clip_grad_norm(tm->d_grad_buf + D*D, KV*D, GRAD_CLIP_NORM, 5.0f,
                               tm->d_norm_scratch, tm->h_norm_scratch);
            gpu_clip_grad_norm(tm->d_grad_buf + D*D + KV*D, KV*D, GRAD_CLIP_NORM, 5.0f,
                               tm->d_norm_scratch, tm->h_norm_scratch);
            gpu_clip_grad_norm(tm->d_grad_buf + D*D + 2*KV*D, D*D, GRAD_CLIP_NORM, 5.0f,
                               tm->d_norm_scratch, tm->h_norm_scratch);

            /* GPU: Adam update — all params in one kernel launch */
            float bc1 = 1.0f / (1.0f - powf(tm->beta1, (float)tm->step));
            float bc2 = 1.0f / (1.0f - powf(tm->beta2, (float)tm->step));
            int threads = 256;
            int blocks = (per_layer + threads - 1) / threads;
            kernel_adam_update<<<blocks, threads>>>(
                d_latent, tm->d_adam_m, tm->d_adam_v, tm->d_grad_buf,
                per_layer, tm->lr, tm->beta1, tm->beta2, tm->eps, bc1, bc2);
            CUDA_CHECK(cudaDeviceSynchronize());

            /* D2H: Adam m back to host */
            CUDA_CHECK(cudaMemcpy(tm->h_staging, tm->d_adam_m,
                                  per_layer * sizeof(float), cudaMemcpyDeviceToHost));
            s = tm->h_staging;
            memcpy(tw->adam_Wq.m, s, D*D*sizeof(float)); s += D*D;
            memcpy(tw->adam_Wk.m, s, KV*D*sizeof(float)); s += KV*D;
            memcpy(tw->adam_Wv.m, s, KV*D*sizeof(float)); s += KV*D;
            memcpy(tw->adam_Wo.m, s, D*D*sizeof(float)); s += D*D;
            for (int e = 0; e < MOE_NUM_EXPERTS; e++) {
                memcpy(tw->adam_gate[e].m, s, INTER*D*sizeof(float)); s += INTER*D;
                memcpy(tw->adam_up[e].m, s, INTER*D*sizeof(float)); s += INTER*D;
                memcpy(tw->adam_down[e].m, s, D*INTER*sizeof(float)); s += D*INTER;
            }

            /* D2H: Adam v back to host */
            CUDA_CHECK(cudaMemcpy(tm->h_staging, tm->d_adam_v,
                                  per_layer * sizeof(float), cudaMemcpyDeviceToHost));
            s = tm->h_staging;
            memcpy(tw->adam_Wq.v, s, D*D*sizeof(float)); s += D*D;
            memcpy(tw->adam_Wk.v, s, KV*D*sizeof(float)); s += KV*D;
            memcpy(tw->adam_Wv.v, s, KV*D*sizeof(float)); s += KV*D;
            memcpy(tw->adam_Wo.v, s, D*D*sizeof(float)); s += D*D;
            for (int e = 0; e < MOE_NUM_EXPERTS; e++) {
                memcpy(tw->adam_gate[e].v, s, INTER*D*sizeof(float)); s += INTER*D;
                memcpy(tw->adam_up[e].v, s, INTER*D*sizeof(float)); s += INTER*D;
                memcpy(tw->adam_down[e].v, s, D*INTER*sizeof(float)); s += D*INTER;
            }

            /* Also sync latent back to host (for checkpoints/diagnostics) */
            CUDA_CHECK(cudaMemcpy(tm->h_staging, d_latent,
                                  per_layer * sizeof(float), cudaMemcpyDeviceToHost));
            s = tm->h_staging;
            memcpy(tw->W_q, s, D*D*sizeof(float)); s += D*D;
            memcpy(tw->W_k, s, KV*D*sizeof(float)); s += KV*D;
            memcpy(tw->W_v, s, KV*D*sizeof(float)); s += KV*D;
            memcpy(tw->W_o, s, D*D*sizeof(float)); s += D*D;
            for (int e = 0; e < MOE_NUM_EXPERTS; e++) {
                memcpy(tw->expert_gate[e], s, INTER*D*sizeof(float)); s += INTER*D;
                memcpy(tw->expert_up[e], s, INTER*D*sizeof(float)); s += INTER*D;
                memcpy(tw->expert_down[e], s, D*INTER*sizeof(float)); s += D*INTER;
            }
        }

        /* Expert grads already clipped in pack loop above */
#else
        /* Adam on STE ternary weights (default/CPU path) */
        adam_update(tw->W_q, tw->grad_Wq, &tw->adam_Wq, tm->step, tm->lr, tm->beta1, tm->beta2, tm->eps);
        adam_update(tw->W_k, tw->grad_Wk, &tw->adam_Wk, tm->step, tm->lr, tm->beta1, tm->beta2, tm->eps);
        adam_update(tw->W_v, tw->grad_Wv, &tw->adam_Wv, tm->step, tm->lr, tm->beta1, tm->beta2, tm->eps);
        adam_update(tw->W_o, tw->grad_Wo, &tw->adam_Wo, tm->step, tm->lr, tm->beta1, tm->beta2, tm->eps);

        for (int e = 0; e < MOE_NUM_EXPERTS; e++) {
            clip_grad_norm(tw->grad_gate[e], INTER * D, GRAD_CLIP_NORM);
            clip_grad_norm(tw->grad_up[e], INTER * D, GRAD_CLIP_NORM);
            clip_grad_norm(tw->grad_down[e], D * INTER, GRAD_CLIP_NORM);
            adam_update(tw->expert_gate[e], tw->grad_gate[e], &tw->adam_gate[e], tm->step, tm->lr, tm->beta1, tm->beta2, tm->eps);
            adam_update(tw->expert_up[e], tw->grad_up[e], &tw->adam_up[e], tm->step, tm->lr, tm->beta1, tm->beta2, tm->eps);
            adam_update(tw->expert_down[e], tw->grad_down[e], &tw->adam_down[e], tm->step, tm->lr, tm->beta1, tm->beta2, tm->eps);
        }
#endif

        /* Router and gain C — Adam (float weights, not ternary) */
        clip_grad_norm(tm->grad_router[l], D * MOE_NUM_EXPERTS, GRAD_CLIP_NORM);
        clip_grad_norm(tm->grad_gain_C_attn[l], D, GRAD_CLIP_NORM);
        clip_grad_norm(tm->grad_gain_C_ffn[l], D, GRAD_CLIP_NORM);
        adam_update(ly->moe.router.W, tm->grad_router[l], &tm->adam_router[l], tm->step, tm->lr, tm->beta1, tm->beta2, tm->eps);
        /* C is FROZEN — not updated by Adam.
         * Lean stability proofs (Thm 68a-d) assume fixed C.
         * Making C learnable invalidates the 65× CFL safety margin.
         * dC = dy*x*α*γ systematically pushes C negative in deep layers,
         * collapsing the reservoir. C stays at init value (1.0). */
    }

#ifdef TWO3_FP_EMBED
    /* Fingerprint projection optimizer — Adam on four ternary projections */
    {
        int qdim = D / 4;
        int fp_size = qdim * 1024;

        clip_grad_norm(tm->fp_grad_Wx, fp_size, GRAD_CLIP_NORM);
        clip_grad_norm(tm->fp_grad_Wy, fp_size, GRAD_CLIP_NORM);
        clip_grad_norm(tm->fp_grad_Wz, fp_size, GRAD_CLIP_NORM);
        clip_grad_norm(tm->fp_grad_Wt, fp_size, GRAD_CLIP_NORM);

        adam_update(tm->fp_latent_Wx, tm->fp_grad_Wx, &tm->fp_adam_Wx, tm->step, tm->lr, tm->beta1, tm->beta2, tm->eps);
        adam_update(tm->fp_latent_Wy, tm->fp_grad_Wy, &tm->fp_adam_Wy, tm->step, tm->lr, tm->beta1, tm->beta2, tm->eps);
        adam_update(tm->fp_latent_Wz, tm->fp_grad_Wz, &tm->fp_adam_Wz, tm->step, tm->lr, tm->beta1, tm->beta2, tm->eps);
        adam_update(tm->fp_latent_Wt, tm->fp_grad_Wt, &tm->fp_adam_Wt, tm->step, tm->lr, tm->beta1, tm->beta2, tm->eps);

        /* Requantize fp projections */
        requantize_gpu(&tm->backward_ctx, tm->fp_latent_Wx, &tm->model.fp_Wx, qdim, 1024, STE_THRESHOLD, NULL);
        requantize_gpu(&tm->backward_ctx, tm->fp_latent_Wy, &tm->model.fp_Wy, qdim, 1024, STE_THRESHOLD, NULL);
        requantize_gpu(&tm->backward_ctx, tm->fp_latent_Wz, &tm->model.fp_Wz, qdim, 1024, STE_THRESHOLD, NULL);
        requantize_gpu(&tm->backward_ctx, tm->fp_latent_Wt, &tm->model.fp_Wt, qdim, 1024, STE_THRESHOLD, NULL);
    }
#endif

    /* Embedding and final gain — Adam */
    clip_grad_norm(tm->grad_embed, 256 * D, GRAD_CLIP_NORM);
    clip_grad_norm(tm->grad_gain_C_final, D, GRAD_CLIP_NORM);
    adam_update(tm->latent_embed, tm->grad_embed, &tm->adam_embed, tm->step, tm->lr, tm->beta1, tm->beta2, tm->eps);
    /* gain_final.C also frozen — same reasoning as per-layer C. */

    /* Sync embedding */
    memcpy(tm->model.embed, tm->latent_embed, 256 * D * sizeof(float));

    /* Requantize ternary weights.
     * GPU-resident: latent already on device from GPU Adam kernel.
     * Legacy: uploads from host via H2D. */
    trainable_requantize(tm);
}

/* ═══════════════════════════════════════════════════════
 * Full training step: forward + backward + Adam update
 *
 * Input: sequence of bytes
 * Target: next byte prediction (shift by 1)
 * Loss: cross-entropy averaged over sequence
 * ═══════════════════════════════════════════════════════ */

/* ═══════════════════════════════════════════════════════
 * Plateau diagnostics — call every N steps during training
 *
 * Dumps: per-layer reservoir depletion, per-expert routing
 * fractions, ternary weight entropy (how diverse are the
 * weights — entropy 0 = all same, log2(3) = uniform ternary).
 * ═══════════════════════════════════════════════════════ */

static void trainable_dump_diagnostics(TrainableModel *tm, int step) {
    int L = tm->cfg.n_layers;
    int D = tm->cfg.dim;

    printf("\n  [diag step=%d]\n", step);

    /* Per-layer reservoir depletion: mean(C - R) for attn and ffn */
    for (int l = 0; l < L; l++) {
        ModelLayer *ly = &tm->model.layers[l];
        float dep_attn = 0.f, dep_ffn = 0.f;
        float min_R_attn = 1e30f, min_R_ffn = 1e30f;
        for (int i = 0; i < D; i++) {
            dep_attn += ly->gain_attn.C[i] - ly->gain_attn.R[i];
            dep_ffn  += ly->gain_ffn.C[i]  - ly->gain_ffn.R[i];
            if (ly->gain_attn.R[i] < min_R_attn) min_R_attn = ly->gain_attn.R[i];
            if (ly->gain_ffn.R[i]  < min_R_ffn)  min_R_ffn  = ly->gain_ffn.R[i];
        }
        printf("    L%d reservoir: attn=%.4f (min_R=%.4f)  ffn=%.4f (min_R=%.4f)\n",
               l, dep_attn / D, min_R_attn, dep_ffn / D, min_R_ffn);
    }

    /* Per-expert routing fractions from reservoir state */
    for (int l = 0; l < L; l++) {
        MoERouter *r = &tm->model.layers[l].moe.router;
        printf("    L%d experts R: [", l);
        for (int e = 0; e < r->n_experts; e++)
            printf("%.3f%s", r->R_expert[e], e < r->n_experts - 1 ? " " : "");
        printf("]\n");
    }

    /* Ternary weight entropy per layer (sample W_q):
     * count {-1, 0, +1} fractions, compute H = -sum(p*log2(p)) */
    for (int l = 0; l < L; l++) {
        TrainableLayerWeights *tw = &tm->layer_weights[l];
        int size = D * D;
        int cnt[3] = {0, 0, 0};  /* -1, 0, +1 */
        for (int i = 0; i < size; i++) {
            float w = tw->W_q[i];
            if (w > 0.5f) cnt[2]++;
            else if (w < -0.5f) cnt[0]++;
            else cnt[1]++;
        }
        float H = 0.f;
        for (int k = 0; k < 3; k++) {
            float p = (float)cnt[k] / (float)size;
            if (p > 1e-10f) H -= p * log2f(p);
        }
        printf("    L%d W_q entropy: %.4f (max=%.4f)  [-1]=%.1f%% [0]=%.1f%% [+1]=%.1f%%\n",
               l, H, log2f(3.0f),
               100.f * cnt[0] / size, 100.f * cnt[1] / size, 100.f * cnt[2] / size);
    }

    printf("\n");
    fflush(stdout);
}

typedef struct {
    float loss;         /* average cross-entropy loss */
    float max_grad;     /* max gradient magnitude (for monitoring) */
    int   correct;      /* number of correct top-1 predictions */
} TrainResult;

/* Forward + backward only — accumulates gradients, does NOT update.
 * Call trainable_zero_grads before first call in batch,
 * trainable_optimizer_step after last call in batch. */
static TrainResult trainable_forward_backward(
    TrainableModel *tm,
    const uint8_t *bytes,    /* [seq_len] input sequence */
    int seq_len              /* must be >= 2 (need input + target) */
) {
    TrainResult result = {0};
    int D = tm->cfg.dim;
    int KV = tm->cfg.n_kv_heads * tm->cfg.head_dim;
    int HD = tm->cfg.head_dim;
    int NH = tm->cfg.n_heads;
    int NKV = tm->cfg.n_kv_heads;
    int INTER = tm->cfg.intermediate;
    int L = tm->cfg.n_layers;
    int T = seq_len - 1;  /* number of prediction positions */

    if (seq_len < 2) {
        fprintf(stderr,
                "trainable_forward_backward: seq_len must be >= 2 (next-byte target). got %d\n",
                seq_len);
        return result;
    }

    /* ═══════════════════════════════════════════════════════
     * FORWARD PASS — save everything for backprop
     * ═══════════════════════════════════════════════════════ */

    static int _timing_done = 0;
    clock_t _t0, _t1;
    double _ms_gain_attn = 0, _ms_qkv_proj = 0, _ms_rope_attn = 0;
    double _ms_o_proj = 0, _ms_moe_fwd = 0, _ms_loss = 0;
    double _ms_bwd_logits = 0, _ms_bwd_moe = 0, _ms_bwd_attn = 0;
    double _ms_bwd_qkv = 0, _ms_bwd_gain = 0;
    #define T_START() _t0 = clock()
    #define T_ACC(var) do { _t1 = clock(); var += (double)(_t1 - _t0) * 1000.0 / CLOCKS_PER_SEC; } while(0)

    /* Per-position hidden states: [seq_len × dim] */
    /* We need to save hidden states BEFORE and AFTER each sublayer */
    float *hidden = (float*)calloc(seq_len * D, sizeof(float));

#ifdef TWO3_FP_EMBED
    /* Fingerprint embedding: four ternary projections X/Y/Z/T → dim/4 each.
     * corpus_offset is stored in the dataset — for now, use byte position
     * from the data pointer offset into the loaded corpus. */
    {
        int qdim = D / 4;
        for (int t = 0; t < seq_len; t++) {
            /* Look up pre-computed fingerprint for this corpus position.
             * bytes pointer offset gives us the corpus position. */
            int corpus_pos = tm->fp_corpus_offset + t;
            if (corpus_pos >= 0 && corpus_pos < tm->model.fp_corpus_size) {
                fp_embed_cpu(hidden + t * D, &tm->model, corpus_pos, D);
            } else {
                /* Fallback: zero embedding for out-of-range positions */
                memset(hidden + t * D, 0, D * sizeof(float));
            }
        }
    }
#else
    /* Byte embedding: index lookup */
    for (int t = 0; t < seq_len; t++)
        byte_embed_cpu(hidden + t * D, tm->model.embed, bytes[t], D);
#endif

    /* Saved activations per layer (for backprop) */
    typedef struct {
        float *pre_attn_normed;     /* [seq_len × D] after gain norm, before Q/K/V */
        float *R_attn_saved;        /* [D] reservoir state before gain norm */
        float *q_all;               /* [seq_len × D] Q projections */
        float *k_store;             /* [seq_len × KV] K store */
        float *v_store;             /* [seq_len × KV] V store */
        float *attn_out;            /* [seq_len × D] attention output */
        float *o_proj;              /* [seq_len × D] after O projection */
        float *pre_ffn_normed;      /* [seq_len × D] after gain norm, before MoE */
        float *R_ffn_saved;         /* [D] reservoir state before FFN gain */
        MoESelection *moe_sel;      /* [seq_len] routing decisions */
        /* Per-expert saved activations */
        float *expert_gate_pre;     /* [seq_len × INTER] gate before squared ReLU */
        float *expert_up_out;       /* [seq_len × INTER] up projection output */
        float *moe_out;             /* [seq_len × D] MoE output before residual */
        float *expert1_out;         /* [seq_len × D] expert 1's raw output (for router grad) */
        int   *selected_experts;    /* [seq_len × MOE_TOP_K] which experts were used */
        float *hidden_pre_attn;     /* [seq_len × D] hidden before attention block */
        float *hidden_pre_ffn;      /* [seq_len × D] hidden before FFN block */
    } LayerSaved;

    LayerSaved *saved = (LayerSaved*)calloc(L, sizeof(LayerSaved));

    /* Residual scaling for deep models: scale each residual add by 1/sqrt(2*L).
     * Prevents hidden state magnitude from growing with depth.
     * For 2 layers: scale = 0.5.  For 4 layers: scale = 0.354. */
    float res_scale = 1.0f / sqrtf(2.0f * (float)L);

    /* Same token, consecutive layer outputs — mirrors model_forward_sequence_cpu.
     * Training never calls that path; log here when TWO3_DEBUG_EXIT_METRICS is set. */
#ifdef TWO3_DEBUG_EXIT_METRICS
    const int t_exit = (seq_len > 0) ? (seq_len - 1) : 0;
    float *h_layer_prev = NULL;
    int prev_probe_pred = -1;
    if (seq_len > 0) {
        h_layer_prev = (float*)malloc((size_t)D * sizeof(float));
        memcpy(h_layer_prev, hidden + t_exit * D, (size_t)D * sizeof(float));
    }
#endif

    for (int l = 0; l < L; l++) {
        ModelLayer *ly = &tm->model.layers[l];
        LayerSaved *sv = &saved[l];

#ifdef TWO3_DEBUG_MOE
        static int debug_layer_hit = 0;
        if (!debug_layer_hit) {
            printf("DEBUG_FWD_HIT l=%d seq_len=%d D=%d\n", l, seq_len, D);
            debug_layer_hit = 1;
        }
#endif

        sv->pre_attn_normed = (float*)calloc(seq_len * D, sizeof(float));
        sv->R_attn_saved = (float*)malloc(D * sizeof(float));
        sv->q_all = (float*)calloc(seq_len * D, sizeof(float));
        sv->k_store = (float*)calloc(seq_len * KV, sizeof(float));
        sv->v_store = (float*)calloc(seq_len * KV, sizeof(float));
        sv->attn_out = (float*)calloc(seq_len * D, sizeof(float));
        sv->o_proj = (float*)calloc(seq_len * D, sizeof(float));
        sv->pre_ffn_normed = (float*)calloc(seq_len * D, sizeof(float));
        sv->R_ffn_saved = (float*)malloc(D * sizeof(float));
        sv->moe_sel = (MoESelection*)calloc(seq_len, sizeof(MoESelection));
        sv->expert_gate_pre = (float*)calloc(seq_len * INTER, sizeof(float));
        sv->expert_up_out = (float*)calloc(seq_len * INTER, sizeof(float));
        sv->moe_out = (float*)calloc(seq_len * D, sizeof(float));
        sv->expert1_out = (float*)calloc(seq_len * D, sizeof(float));
        sv->hidden_pre_attn = (float*)malloc(seq_len * D * sizeof(float));
        sv->hidden_pre_ffn = (float*)malloc(seq_len * D * sizeof(float));

        /* Save hidden state before attention block */
        memcpy(sv->hidden_pre_attn, hidden, seq_len * D * sizeof(float));

        /* Save reservoir state before this layer's gain */
        memcpy(sv->R_attn_saved, ly->gain_attn.R, D * sizeof(float));

        /* ── Attention block (batched projections) ── */

        /* Phase 1: gain norm all positions (sequential — R depends on previous) */
        T_START();
        for (int t = 0; t < seq_len; t++)
            gain_forward_cpu(sv->pre_attn_normed + t * D, hidden + t * D,
                             ly->gain_attn.R, ly->gain_attn.C, D);

        T_ACC(_ms_gain_attn);

        /* Phase 2: multi-projection Q/K/V — quantize once, project 3x */
        T_START();
        {
            const Two3Weights *W_qkv[3] = { &ly->W_q, &ly->W_k, &ly->W_v };
            float *out_qkv[3] = { sv->q_all, sv->k_store, sv->v_store };
            ternary_project_multi_cpu(W_qkv, out_qkv, sv->pre_attn_normed, 3, seq_len, D);
        }

        T_ACC(_ms_qkv_proj);

        /* Phase 3: RoPE all positions (CPU, cheap) */
        T_START();
        for (int t = 0; t < seq_len; t++)
            rope_apply_cpu(sv->q_all + t * D, sv->k_store + t * KV,
                           &tm->model.rope, t, NH, NKV);

        /* Phase 4: causal attention per position (sequential) */
        for (int t = 0; t < seq_len; t++)
            causal_attention_cpu(sv->q_all + t * D, sv->k_store, sv->v_store,
                                 sv->attn_out + t * D, t, NH, NKV, HD);

        T_ACC(_ms_rope_attn);

        /* Phase 5: batch O projection — 1 GPU call instead of seq_len */
        T_START();
        ternary_project_batch_cpu(&ly->W_o, sv->attn_out, sv->o_proj, seq_len, D);

        /* Phase 6: scale + residual add */
        {
            float s = res_scale;
            for (int t = 0; t < seq_len; t++)
                for (int i = 0; i < D; i++)
                    hidden[t * D + i] += s * sv->o_proj[t * D + i];
        }

        T_ACC(_ms_o_proj);

        /* Save hidden before FFN block */
        memcpy(sv->hidden_pre_ffn, hidden, seq_len * D * sizeof(float));

        /* Save reservoir state before FFN gain */
        T_START();
        memcpy(sv->R_ffn_saved, ly->gain_ffn.R, D * sizeof(float));

        /* ── MoE block (expert-grouped batched projections) ── */

        /* Step 7: Gain norm all positions (sequential) */
        for (int t = 0; t < seq_len; t++)
            gain_forward_cpu(sv->pre_ffn_normed + t * D, hidden + t * D,
                             ly->gain_ffn.R, ly->gain_ffn.C, D);

        /* Step 8: Route all positions */
        for (int t = 0; t < seq_len; t++)
            moe_route(&ly->moe.router, sv->pre_ffn_normed + t * D, &sv->moe_sel[t]);

        /* Step 9: Expert-grouped forward — gather, batch, scatter.
         * For each expert: gather positions routed to it, batch-project
         * gate+up+down, scatter outputs back. */
        memset(sv->moe_out, 0, seq_len * D * sizeof(float));
        {
            /* Group positions by expert (top-K routing) */
            int *expert_pos_flat = (int*)malloc(MOE_NUM_EXPERTS * seq_len * sizeof(int));
            #define EP(e, i) expert_pos_flat[(e) * seq_len + (i)]
            int expert_cnt[MOE_NUM_EXPERTS];

            /* Scratch buffers for batched projections */
            float *gather_in  = (float*)malloc(seq_len * D * sizeof(float));
            float *gate_batch = (float*)malloc(seq_len * INTER * sizeof(float));
            float *up_batch   = (float*)malloc(seq_len * INTER * sizeof(float));
            float *h_expert   = (float*)malloc(seq_len * INTER * sizeof(float));
            float *down_batch = (float*)malloc(seq_len * D * sizeof(float));

            for (int k_sel = 0; k_sel < MOE_TOP_K; k_sel++) {
                /* Collect which positions route to each expert for this top-K slot */
                memset(expert_cnt, 0, sizeof(expert_cnt));
                for (int t = 0; t < seq_len; t++) {
                    int eid = sv->moe_sel[t].expert_ids[k_sel];
                    EP(eid, expert_cnt[eid]++) = t;
                }

                /* Process each active expert */
                for (int e = 0; e < MOE_NUM_EXPERTS; e++) {
                    int cnt = expert_cnt[e];
                    if (cnt == 0) continue;

                    /* Gather: collect input vectors for this expert */
                    for (int i = 0; i < cnt; i++)
                        memcpy(gather_in + i * D, sv->pre_ffn_normed + EP(e, i) * D,
                               D * sizeof(float));

                    /* Batch gate + up — quantize once, project 2x */
                    {
                        const Two3Weights *W_gu[2] = { &ly->moe.experts[e].gate, &ly->moe.experts[e].up };
                        float *out_gu[2] = { gate_batch, up_batch };
                        ternary_project_multi_cpu(W_gu, out_gu, gather_in, 2, cnt, D);
                    }

                    /* Scale */
                    float scale = 1.0f / sqrtf((float)D);
                    for (int i = 0; i < cnt * INTER; i++) {
                        gate_batch[i] *= scale;
                        up_batch[i] *= scale;
                    }

                    /* Save pre-activation gate + up BEFORE squared ReLU.
                     * Backward needs original x for dx = (x>0) ? dy*2x : 0. */
                    if (k_sel == 0) {
                        for (int i = 0; i < cnt; i++) {
                            int t = EP(e, i);
                            memcpy(sv->expert_gate_pre + t * INTER,
                                   gate_batch + i * INTER, INTER * sizeof(float));
                            memcpy(sv->expert_up_out + t * INTER,
                                   up_batch + i * INTER, INTER * sizeof(float));
                        }
                    }

                    /* Squared ReLU in-place, then multiply */
                    for (int i = 0; i < cnt * INTER; i++) {
                        float g = gate_batch[i];
                        gate_batch[i] = (g > 0.0f) ? g * g : 0.0f;
                    }
                    for (int i = 0; i < cnt * INTER; i++)
                        h_expert[i] = gate_batch[i] * up_batch[i];

                    /* Batch down projection */
                    ternary_project_batch_cpu(&ly->moe.experts[e].down,
                                             h_expert, down_batch, cnt, INTER);

                    /* Scatter outputs + save expert1_out for backward */
                    for (int i = 0; i < cnt; i++) {
                        int t = EP(e, i);
                        float w = sv->moe_sel[t].expert_weights[k_sel];
                        for (int d = 0; d < D; d++)
                            sv->moe_out[t * D + d] += w * down_batch[i * D + d];

                        if (k_sel == 0)
                            memcpy(sv->expert1_out + t * D,
                                   down_batch + i * D, D * sizeof(float));
                    }
                }

                /* Update reservoirs for load balancing (after expert dispatch, before scatter) */
                moe_update_reservoir(&ly->moe.router, expert_cnt, seq_len);

#ifdef TWO3_DEBUG_MOE
                if (l == 0) { /* layer 0 only, cheap */
                    printf("step=%d layer=%d ", tm->step, l);
                    printf("counts:");
                    for (int e = 0; e < MOE_NUM_EXPERTS; e++) printf(" %d", expert_cnt[e]);
                    printf("  R:");
                    for (int e = 0; e < MOE_NUM_EXPERTS; e++) printf(" %.3f", ly->moe.router.R_expert[e]);
                    printf("\n");
                }
#endif
            }

            free(expert_pos_flat);
            #undef EP
            free(gather_in); free(gate_batch); free(up_batch);
            free(h_expert); free(down_batch);
        }

        /* Step 10: Residual add (scaled by 1/sqrt(D) like attention) */
        {
            float moe_s = res_scale;
            for (int t = 0; t < seq_len; t++)
                for (int i = 0; i < D; i++)
                    hidden[t * D + i] += moe_s * sv->moe_out[t * D + i];
        }

        T_ACC(_ms_moe_fwd);

#ifdef TWO3_DEBUG_GAIN
        /* Log gain reservoir depletion as difficulty signal */
        if (l == 0) { /* layer 0 only, cheap */
            float total_dep = 0.0f;
            for (int i = 0; i < D; i++)
                total_dep += ly->gain_ffn.C[i] - ly->gain_ffn.R[i];
            float mean_dep = total_dep / (float)D;
            printf("step=%d layer=%d mean_depletion=%.4f\n", tm->step, l, mean_dep);
        }
#endif

        /* Diagnostic: check for activation explosion per layer */
        {
            float max_h = 0, min_R = 1e30f, max_R = 0;
            int nan_cnt = 0;
            for (int i = 0; i < seq_len * D; i++) {
                if (hidden[i] != hidden[i]) nan_cnt++;
                if (fabsf(hidden[i]) > max_h) max_h = fabsf(hidden[i]);
            }
            for (int i = 0; i < D; i++) {
                if (ly->gain_attn.R[i] < min_R) min_R = ly->gain_attn.R[i];
                if (ly->gain_attn.R[i] > max_R) max_R = ly->gain_attn.R[i];
            }
            if (nan_cnt > 0 || max_h > 1e4f) {
                printf("  !! LAYER %d: max_h=%.1f  NaN=%d  R=[%.4f,%.4f]\n",
                       l, max_h, nan_cnt, min_R, max_R);
            }
        }

#ifdef TWO3_DEBUG_EXIT_METRICS
        if (h_layer_prev && seq_len > 0) {
            const float *h = hidden + t_exit * D;
            float dot = 0.f, na = 0.f, nb = 0.f;
            for (int i = 0; i < D; i++) {
                float a = h_layer_prev[i];
                float b = h[i];
                dot += a * b;
                na += a * a;
                nb += b * b;
            }
            float cos_sim = dot / (sqrtf(na + 1e-10f) * sqrtf(nb + 1e-10f));
            float probe_logits[256];
            byte_logits_cpu(probe_logits, h, tm->model.embed, D);
            int best_byte;
            float top_l, second_l, margin_l;
            byte_probe_top2(probe_logits, &best_byte, &top_l, &second_l, &margin_l);
            int stable = (l > 0 && prev_probe_pred >= 0 && best_byte == prev_probe_pred) ? 1 : 0;
            printf("[exit_probe] train step=%d layer=%d t=%d cos=%.6f pred=%d top=%.4f 2nd=%.4f margin=%.4f stable_vs_prev=%d\n",
                   tm->step, l, t_exit, cos_sim, best_byte, top_l, second_l, margin_l, stable);
            fflush(stdout);
            prev_probe_pred = best_byte;
            memcpy(h_layer_prev, h, (size_t)D * sizeof(float));
        }
#endif
    }

#ifdef TWO3_DEBUG_EXIT_METRICS
    free(h_layer_prev);
#endif

    /* Final gain norm + logits */
    T_START();
    float *final_normed = (float*)calloc(seq_len * D, sizeof(float));
    float *R_final_saved = (float*)malloc(D * sizeof(float));
    memcpy(R_final_saved, tm->model.gain_final.R, D * sizeof(float));

    float *all_logits = (float*)malloc(seq_len * 256 * sizeof(float));

    for (int t = 0; t < seq_len; t++) {
        gain_forward_cpu(final_normed + t * D, hidden + t * D,
                         tm->model.gain_final.R, tm->model.gain_final.C, D);
        byte_logits_cpu(all_logits + t * 256, final_normed + t * D, tm->model.embed, D);
        clip_logits(all_logits + t * 256, 256);
    }

    /* ═══════════════════════════════════════════════════════
     * LOSS: cross-entropy on next-byte prediction
     * Position t predicts bytes[t+1] (teacher forcing)
     * ═══════════════════════════════════════════════════════ */

    float total_loss = 0;
    float *d_logits_all = (float*)calloc(seq_len * 256, sizeof(float));

#ifdef TWO3_WEIGHTED_LOSS
    float *per_pos_loss = (float*)malloc(T * sizeof(float));
#endif

    for (int t = 0; t < T; t++) {
        int target = bytes[t + 1];
        float loss = cross_entropy_loss(
            all_logits + t * 256, target, d_logits_all + t * 256);
        total_loss += loss;
#ifdef TWO3_WEIGHTED_LOSS
        per_pos_loss[t] = loss;
#endif

        /* Track accuracy */
        int pred = 0;
        float pred_val = all_logits[t * 256];
        for (int i = 1; i < 256; i++) {
            if (all_logits[t * 256 + i] > pred_val) {
                pred_val = all_logits[t * 256 + i];
                pred = i;
            }
        }
        if (pred == target) result.correct++;
    }
    /* Last position has no target — zero gradient */

#ifdef TWO3_WEIGHTED_LOSS
    /* Weight gradients by difficulty: easy tokens → 10%, hard tokens → 100% */
    {
        float *loss_weights = (float*)malloc(T * sizeof(float));
        compute_loss_weights(loss_weights, per_pos_loss, T);
        apply_loss_weights(d_logits_all, loss_weights, T);
        free(loss_weights);
    }
    free(per_pos_loss);
#endif

    result.loss = total_loss / (float)T;

    /* (timing handled by T_ACC macros) */

    /* ═══════════════════════════════════════════════════════
     * BACKWARD PASS — accumulates into persistent grad buffers
     * ═══════════════════════════════════════════════════════ */
    T_ACC(_ms_loss);

    /* ═══════════════════════════════════════════════════════
     * (caller must zero first via trainable_zero_grads)
     * ═══════════════════════════════════════════════════════ */

    /* d_hidden: gradient on hidden states [seq_len × D] */
    float *d_hidden = (float*)calloc(seq_len * D, sizeof(float));

    /* Backward through logits → final_normed → hidden */
    for (int t = 0; t < T; t++) {
        /* d_logits → d_final_normed (backward through byte_logits = hidden @ embed^T) */
        float *d_fn = (float*)calloc(D, sizeof(float));
        for (int d = 0; d < D; d++) {
            float sum = 0;
            for (int b = 0; b < 256; b++)
                sum += d_logits_all[t * 256 + b] * tm->model.embed[b * D + d];
            d_fn[d] = sum;
        }

        /* d_logits → d_embed (through logits = normed @ embed^T) */
        for (int b = 0; b < 256; b++) {
            for (int d = 0; d < D; d++)
                tm->grad_embed[b * D + d] += d_logits_all[t * 256 + b] * final_normed[t * D + d];
        }

        /* Backward through final gain norm */
        gain_backward_cpu(d_hidden + t * D, d_fn, hidden + t * D,
                          R_final_saved, tm->grad_gain_C_final, D);

        free(d_fn);
    }

    T_START();

#ifdef TWO3_LAYER_SKIP
    /* Update convergence state from gain reservoirs */
    {
        GainState *ga = (GainState*)malloc(L * sizeof(GainState));
        GainState *gf = (GainState*)malloc(L * sizeof(GainState));
        for (int l = 0; l < L; l++) {
            ga[l] = tm->model.layers[l].gain_attn;
            gf[l] = tm->model.layers[l].gain_ffn;
        }
        layer_skip_update(&tm->layer_skip, ga, gf, D);
        free(ga); free(gf);
    }
#endif

    /* Backward through layers (reverse order).
     * Gradients accumulate into persistent tw->grad_* buffers. */
    for (int l = L - 1; l >= 0; l--) {
        ModelLayer *ly = &tm->model.layers[l];
        LayerSaved *sv = &saved[l];
        TrainableLayerWeights *tw = &tm->layer_weights[l];

#ifdef TWO3_LAYER_SKIP
        if (layer_skip_should_skip(&tm->layer_skip, l)) {
            /* Skip weight gradients — gradient flows through residual unchanged.
             * Still need to free saved activations. */
            free(sv->pre_attn_normed); free(sv->R_attn_saved);
            free(sv->q_all); free(sv->k_store); free(sv->v_store);
            free(sv->attn_out); free(sv->o_proj);
            free(sv->pre_ffn_normed); free(sv->R_ffn_saved);
            free(sv->moe_sel); free(sv->expert_gate_pre); free(sv->expert_up_out);
            free(sv->moe_out); free(sv->expert1_out);
            free(sv->hidden_pre_attn); free(sv->hidden_pre_ffn);
            tm->layer_skip.total_skipped++;
            continue;
        }
#endif

        /* Layer-wise gradient clipping: prevent gradient explosion in deep models.
         * Clip d_hidden norm per-position before it enters this layer's backward.
         * Without this, 4+ layers reach 10^23 gradient magnitude. */
        for (int t = 0; t < seq_len; t++) {
            clip_grad_norm(d_hidden + t * D, D, GRAD_CLIP_NORM);
        }

        /* Use persistent gradient buffers (zeroed by trainable_zero_grads) */
        float *dW_q = tw->grad_Wq;
        float *dW_k = tw->grad_Wk;
        float *dW_v = tw->grad_Wv;
        float *dW_o = tw->grad_Wo;
        float *dW_gate[MOE_NUM_EXPERTS], *dW_up[MOE_NUM_EXPERTS], *dW_down[MOE_NUM_EXPERTS];
        for (int e = 0; e < MOE_NUM_EXPERTS; e++) {
            dW_gate[e] = tw->grad_gate[e];
            dW_up[e]   = tw->grad_up[e];
            dW_down[e] = tw->grad_down[e];
        }

        /* ── Backward through FFN block (expert-grouped) ── */
        T_START();
        {
            float scale = 1.0f / sqrtf((float)D);

            /* Step 10 backward: d_moe_out_all = (res_scale/sqrt(D)) * d_hidden */
            float moe_s = res_scale;
            float *d_moe_out_all = (float*)malloc(seq_len * D * sizeof(float));
            for (int i = 0; i < seq_len * D; i++)
                d_moe_out_all[i] = moe_s * d_hidden[i];

            /* d_normed_ffn_all accumulates gradients from expert backward + router */
            float *d_normed_ffn_all = (float*)calloc(seq_len * D, sizeof(float));

            /* Group positions by top-1 expert for backward */
            int *expert_pos_flat = (int*)malloc(MOE_NUM_EXPERTS * seq_len * sizeof(int));
            #define EPB(e, i) expert_pos_flat[(e) * seq_len + (i)]
            int expert_cnt[MOE_NUM_EXPERTS];
            memset(expert_cnt, 0, sizeof(expert_cnt));

            for (int t = 0; t < seq_len; t++) {
                int eid = sv->moe_sel[t].expert_ids[0];
                EPB(eid, expert_cnt[eid]++) = t;
            }

            /* Scratch buffers for batched backward */
            float *g_d_expert_out = (float*)malloc(seq_len * D * sizeof(float));
            float *g_h_expert     = (float*)malloc(seq_len * INTER * sizeof(float));
            float *g_d_h_expert   = (float*)calloc(seq_len * INTER, sizeof(float));
            float *g_d_gate       = (float*)malloc(seq_len * INTER * sizeof(float));
            float *g_d_up         = (float*)malloc(seq_len * INTER * sizeof(float));
            float *g_normed_in    = (float*)malloc(seq_len * D * sizeof(float));
            float *g_d_normed_out = (float*)calloc(seq_len * D, sizeof(float));

            for (int e = 0; e < MOE_NUM_EXPERTS; e++) {
                int cnt = expert_cnt[e];
                if (cnt == 0) continue;

                /* Gather: d_expert_out = d_moe_out * w0 */
                for (int i = 0; i < cnt; i++) {
                    int t = EPB(e, i);
                    float w0 = sv->moe_sel[t].expert_weights[0];
                    for (int d = 0; d < D; d++)
                        g_d_expert_out[i * D + d] = d_moe_out_all[t * D + d] * w0;

                    /* Recompute h_expert from saved gate_pre and up_out */
                    float *gate = sv->expert_gate_pre + t * INTER;
                    float *up   = sv->expert_up_out + t * INTER;
                    for (int j = 0; j < INTER; j++) {
                        float gs = gate[j] * scale;
                        float us = up[j] * scale;
                        float ga = (gs > 0.0f) ? gs * gs : 0.0f;
                        g_h_expert[i * INTER + j] = ga * us;
                    }

                    memcpy(g_normed_in + i * D, sv->pre_ffn_normed + t * D,
                           D * sizeof(float));
                }

                /* Batch down backward: [cnt, D] → [cnt, INTER] */
                memset(g_d_h_expert, 0, cnt * INTER * sizeof(float));
                ternary_project_backward_gpu_batch(&tm->backward_ctx,
                    &ly->moe.experts[e].down,
                    g_d_expert_out, g_h_expert,
                    tw->expert_down[e],
                    g_d_h_expert, dW_down[e],
                    cnt, D, INTER);

                /* CPU: d_gate and d_up from d_h_expert (per element) */
                for (int i = 0; i < cnt; i++) {
                    int t = EPB(e, i);
                    float *gate = sv->expert_gate_pre + t * INTER;
                    float *up   = sv->expert_up_out + t * INTER;
                    for (int j = 0; j < INTER; j++) {
                        float gs = gate[j] * scale;
                        float us = up[j] * scale;
                        float ga = (gs > 0.0f) ? gs * gs : 0.0f;
                        float dh = g_d_h_expert[i * INTER + j];
                        /* d_gate_act = dh * up_scaled, d_up_scaled = dh * gate_activated */
                        float d_gate_act = dh * us;
                        float d_up_scaled = dh * ga;
                        /* squared_relu backward: d_gate_scaled = (gs > 0) ? d_gate_act * 2*gs : 0 */
                        float d_gate_scaled = (gs > 0.0f) ? d_gate_act * 2.0f * gs : 0.0f;
                        /* backward through scaling */
                        g_d_gate[i * INTER + j] = d_gate_scaled * scale;
                        g_d_up[i * INTER + j] = d_up_scaled * scale;
                    }
                }

                /* Batch gate backward: [cnt, INTER] dY, [cnt, D] X → accumulate dW_gate, d_normed */
                memset(g_d_normed_out, 0, cnt * D * sizeof(float));
                ternary_project_backward_gpu_batch(&tm->backward_ctx,
                    &ly->moe.experts[e].gate,
                    g_d_gate, g_normed_in,
                    tw->expert_gate[e],
                    g_d_normed_out, dW_gate[e],
                    cnt, INTER, D);

                /* Batch up backward: same pattern, accumulate into same d_normed */
                ternary_project_backward_gpu_batch(&tm->backward_ctx,
                    &ly->moe.experts[e].up,
                    g_d_up, g_normed_in,
                    tw->expert_up[e],
                    g_d_normed_out, dW_up[e],
                    cnt, INTER, D);

                /* Scatter d_normed back to position order */
                for (int i = 0; i < cnt; i++) {
                    int t = EPB(e, i);
                    for (int d = 0; d < D; d++)
                        d_normed_ffn_all[t * D + d] += g_d_normed_out[i * D + d];
                }
            }

            /* Router backward (CPU, per position — cheap) */
            for (int t = 0; t < seq_len; t++) {
                float d_expert_w[MOE_TOP_K] = {0};
                for (int i = 0; i < D; i++)
                    d_expert_w[0] += d_moe_out_all[t * D + i] * sv->expert1_out[t * D + i];
                moe_router_backward_cpu(d_expert_w, &sv->moe_sel[t],
                    sv->pre_ffn_normed + t * D, &ly->moe.router,
                    d_normed_ffn_all + t * D, tm->grad_router[l]);
            }

            /* Gain backward per position */
            for (int t = 0; t < seq_len; t++) {
                float *d_h_pre_ffn = (float*)calloc(D, sizeof(float));
                gain_backward_cpu(d_h_pre_ffn, d_normed_ffn_all + t * D,
                                  sv->hidden_pre_ffn + t * D,
                                  sv->R_ffn_saved, tm->grad_gain_C_ffn[l], D);
                for (int i = 0; i < D; i++)
                    d_hidden[t * D + i] += d_h_pre_ffn[i];
                free(d_h_pre_ffn);
            }

            free(d_moe_out_all); free(d_normed_ffn_all);
            free(expert_pos_flat);
            #undef EPB
            free(g_d_expert_out); free(g_h_expert); free(g_d_h_expert);
            free(g_d_gate); free(g_d_up); free(g_normed_in); free(g_d_normed_out);
        }

        T_ACC(_ms_bwd_moe);

        /* ── Backward through attention block (batched projections) ── */
        T_START();

        /* Phase 1: batch O backward — 1 GPU call instead of seq_len.
         * Forward: h += res_scale/sqrt(D) * W_o @ attn_out
         * Backward: d_o_proj_all = scale * d_hidden, then backward through W_o */
        float *d_o_proj_all = (float*)malloc(seq_len * D * sizeof(float));
        float *d_attn_out_all = (float*)calloc(seq_len * D, sizeof(float));
        {
            float s = res_scale;
            for (int i = 0; i < seq_len * D; i++)
                d_o_proj_all[i] = s * d_hidden[i];
        }
        ternary_project_backward_gpu_batch(&tm->backward_ctx,
            &ly->W_o, d_o_proj_all, sv->attn_out,
            tw->W_o, d_attn_out_all, dW_o, seq_len, D, D);
        free(d_o_proj_all);

        /* Phase 2: batched causal attention backward on GPU */
        float *dq_all  = (float*)calloc(seq_len * D, sizeof(float));
        float *dk_store = (float*)calloc(seq_len * KV, sizeof(float));
        float *dv_store = (float*)calloc(seq_len * KV, sizeof(float));

        two3_attention_backward_fast(
            &tm->backward_ctx,
            sv->q_all, sv->k_store, sv->v_store, d_attn_out_all,
            dq_all, dk_store, dv_store,
            seq_len, NH, NKV, HD);

        free(d_attn_out_all);

        /* Phase 2.5: Inverse RoPE per position */
        for (int t = 0; t < seq_len; t++)
            rope_unapply_cpu(dq_all + t * D, dk_store + t * KV,
                             &tm->model.rope, t, NH, NKV);

        T_ACC(_ms_bwd_attn);

        /* Phase 3: batch Q/K/V backward — 3 GPU calls instead of 3×seq_len.
         * d_normed_attn_all accumulates from Q, K, V backward. */
        T_START();
        float *d_normed_attn_all = (float*)calloc(seq_len * D, sizeof(float));

        ternary_project_backward_gpu_batch(&tm->backward_ctx,
            &ly->W_q, dq_all, sv->pre_attn_normed,
            tw->W_q, d_normed_attn_all, dW_q, seq_len, D, D);

        ternary_project_backward_gpu_batch(&tm->backward_ctx,
            &ly->W_k, dk_store, sv->pre_attn_normed,
            tw->W_k, d_normed_attn_all, dW_k, seq_len, KV, D);

        ternary_project_backward_gpu_batch(&tm->backward_ctx,
            &ly->W_v, dv_store, sv->pre_attn_normed,
            tw->W_v, d_normed_attn_all, dW_v, seq_len, KV, D);

        free(dq_all); free(dk_store); free(dv_store);

        T_ACC(_ms_bwd_qkv);

        /* Phase 4: gain backward per position (CPU, sequential) */
        T_START();
        for (int t = 0; t < seq_len; t++) {
            float *d_h_pre_attn = (float*)calloc(D, sizeof(float));
            gain_backward_cpu(d_h_pre_attn, d_normed_attn_all + t * D,
                              sv->hidden_pre_attn + t * D,
                              sv->R_attn_saved, tm->grad_gain_C_attn[l], D);
            for (int i = 0; i < D; i++)
                d_hidden[t * D + i] += d_h_pre_attn[i];
            free(d_h_pre_attn);
        }

        free(d_normed_attn_all);
        T_ACC(_ms_bwd_gain);

        /* Track max gradient for monitoring */
        for (int i = 0; i < D * D; i++) {
            if (fabsf(dW_q[i]) > result.max_grad) result.max_grad = fabsf(dW_q[i]);
        }

        /* Gradients accumulated in tw->grad_* — no Adam here.
         * Caller invokes trainable_optimizer_step after batch. */
    }

#ifdef TWO3_FP_EMBED
    /* Fingerprint embedding backward: gradients for four ternary projections.
     * d_hidden[t] → backward through fp_Wx/Wy/Wz/Wt → grad accumulation. */
    {
        int qdim = D / 4;
        for (int t = 0; t < seq_len; t++) {
            int corpus_pos = tm->fp_corpus_offset + t;
            if (corpus_pos < 0 || corpus_pos >= tm->model.fp_corpus_size) continue;

            float *fp = tm->model.fp_data + (size_t)corpus_pos * 4096;
            float *dh = d_hidden + t * D;

            /* Each dimension: dW += dh[qdim_slice] @ fp[1024_slice]^T
             * (STE: use ternary-quantized latent for dX, raw gradient for dW) */
            for (int m = 0; m < qdim; m++) {
                for (int k = 0; k < 1024; k++) {
                    float g;
                    /* X projection backward */
                    g = dh[m] * fp[k];
                    tm->fp_grad_Wx[m * 1024 + k] += ste_backward(g, tm->fp_latent_Wx[m * 1024 + k]);
                    /* Y projection backward */
                    g = dh[qdim + m] * fp[1024 + k];
                    tm->fp_grad_Wy[m * 1024 + k] += ste_backward(g, tm->fp_latent_Wy[m * 1024 + k]);
                    /* Z projection backward */
                    g = dh[2*qdim + m] * fp[2048 + k];
                    tm->fp_grad_Wz[m * 1024 + k] += ste_backward(g, tm->fp_latent_Wz[m * 1024 + k]);
                    /* T projection backward */
                    g = dh[3*qdim + m] * fp[3072 + k];
                    tm->fp_grad_Wt[m * 1024 + k] += ste_backward(g, tm->fp_latent_Wt[m * 1024 + k]);
                }
            }
        }
    }
#else
    /* Embedding gradient: backward through embedding lookup
     * d_embed[byte] += d_hidden[t] for each t where bytes[t] == byte */
    for (int t = 0; t < seq_len; t++) {
        int b = bytes[t];
        for (int d = 0; d < D; d++)
            tm->grad_embed[b * D + d] += d_hidden[t * D + d];
    }
#endif

    /* Adam + requantize deferred to trainable_optimizer_step */

    /* ═══════════════════════════════════════════════════════
     * Cleanup
     * ═══════════════════════════════════════════════════════ */

    for (int l = 0; l < L; l++) {
        LayerSaved *sv = &saved[l];
        free(sv->pre_attn_normed); free(sv->R_attn_saved);
        free(sv->q_all); free(sv->k_store); free(sv->v_store);
        free(sv->attn_out); free(sv->o_proj);
        free(sv->pre_ffn_normed); free(sv->R_ffn_saved);
        free(sv->moe_sel);
        free(sv->expert_gate_pre); free(sv->expert_up_out);
        free(sv->moe_out); free(sv->expert1_out);
        free(sv->hidden_pre_attn); free(sv->hidden_pre_ffn);
    }
    free(saved);
    free(hidden); free(d_hidden);
    free(final_normed); free(R_final_saved);
    free(all_logits); free(d_logits_all);

    if (!_timing_done) {
        double fwd = _ms_gain_attn + _ms_qkv_proj + _ms_rope_attn + _ms_o_proj + _ms_moe_fwd;
        double bwd = _ms_bwd_logits + _ms_bwd_moe + _ms_bwd_attn + _ms_bwd_qkv + _ms_bwd_gain;
        double total = fwd + _ms_loss + bwd;
        printf("\n  [PROFILE] total=%.0fms  fwd=%.0fms (%.0f%%)  bwd=%.0fms (%.0f%%)\n",
               total, fwd, 100*fwd/total, bwd, 100*bwd/total);
        printf("    FWD: gain=%.0f  QKV=%.0f  rope+attn=%.0f  O=%.0f  MoE=%.0f\n",
               _ms_gain_attn, _ms_qkv_proj, _ms_rope_attn, _ms_o_proj, _ms_moe_fwd);
        printf("    BWD: logits=%.0f  MoE=%.0f  attn=%.0f  QKV=%.0f  gain=%.0f\n",
               _ms_bwd_logits, _ms_bwd_moe, _ms_bwd_attn, _ms_bwd_qkv, _ms_bwd_gain);
        printf("    LOSS: %.0f\n\n", _ms_loss);
        _timing_done = 1;
    }

    #undef T_START
    #undef T_ACC

    return result;
}

/* Convenience: zero + forward_backward + optimizer_step in one call.
 * Same behavior as the old trainable_train_step. */
static TrainResult trainable_train_step(
    TrainableModel *tm,
    const uint8_t *bytes,
    int seq_len
) {
    trainable_zero_grads(tm);
    TrainResult r = trainable_forward_backward(tm, bytes, seq_len);
    trainable_optimizer_step(tm);
    return r;
}

/* ═══════════════════════════════════════════════════════
 * Checkpoint save/load
 * ═══════════════════════════════════════════════════════ */

static int trainable_save(const TrainableModel *tm, const char *path) {
    FILE *f = fopen(path, "wb");
    if (!f) return -1;

    int D = tm->cfg.dim;
    int KV = tm->cfg.n_kv_heads * tm->cfg.head_dim;
    int INTER = tm->cfg.intermediate;
    int L = tm->cfg.n_layers;

    /* Header */
    uint32_t magic = 0x54324C34;  /* "T2L4" = two3 layer 4 */
    fwrite(&magic, 4, 1, f);
    fwrite(&tm->cfg, sizeof(ModelConfig), 1, f);
    fwrite(&tm->step, sizeof(int), 1, f);

    /* Embedding */
    fwrite(tm->latent_embed, sizeof(float), 256 * D, f);

    /* Per-layer latent weights */
    for (int l = 0; l < L; l++) {
        const TrainableLayerWeights *tw = &tm->layer_weights[l];
        fwrite(tw->W_q, sizeof(float), D * D, f);
        fwrite(tw->W_k, sizeof(float), KV * D, f);
        fwrite(tw->W_v, sizeof(float), KV * D, f);
        fwrite(tw->W_o, sizeof(float), D * D, f);

        for (int e = 0; e < MOE_NUM_EXPERTS; e++) {
            fwrite(tw->expert_gate[e], sizeof(float), INTER * D, f);
            fwrite(tw->expert_up[e], sizeof(float), INTER * D, f);
            fwrite(tw->expert_down[e], sizeof(float), D * INTER, f);
        }

        /* Router weights */
        fwrite(tm->model.layers[l].moe.router.W, sizeof(float), D * MOE_NUM_EXPERTS, f);

        /* Gain C */
        fwrite(tm->model.layers[l].gain_attn.C, sizeof(float), D, f);
        fwrite(tm->model.layers[l].gain_ffn.C, sizeof(float), D, f);
    }

    /* Final gain C */
    fwrite(tm->model.gain_final.C, sizeof(float), D, f);

    fclose(f);
    return 0;
}

static int trainable_load(TrainableModel *tm, const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) return -1;

    uint32_t magic;
    fread(&magic, 4, 1, f);
    if (magic != 0x54324C34) { fclose(f); return -2; }

    ModelConfig cfg;
    fread(&cfg, sizeof(ModelConfig), 1, f);
    fread(&tm->step, sizeof(int), 1, f);

    int D = cfg.dim;
    int KV = cfg.n_kv_heads * cfg.head_dim;
    int INTER = cfg.intermediate;
    int L = cfg.n_layers;

    /* Embedding */
    fread(tm->latent_embed, sizeof(float), 256 * D, f);

    /* Per-layer */
    for (int l = 0; l < L; l++) {
        TrainableLayerWeights *tw = &tm->layer_weights[l];
        fread(tw->W_q, sizeof(float), D * D, f);
        fread(tw->W_k, sizeof(float), KV * D, f);
        fread(tw->W_v, sizeof(float), KV * D, f);
        fread(tw->W_o, sizeof(float), D * D, f);

        for (int e = 0; e < MOE_NUM_EXPERTS; e++) {
            fread(tw->expert_gate[e], sizeof(float), INTER * D, f);
            fread(tw->expert_up[e], sizeof(float), INTER * D, f);
            fread(tw->expert_down[e], sizeof(float), D * INTER, f);
        }

        fread(tm->model.layers[l].moe.router.W, sizeof(float), D * MOE_NUM_EXPERTS, f);
        fread(tm->model.layers[l].gain_attn.C, sizeof(float), D, f);
        fread(tm->model.layers[l].gain_ffn.C, sizeof(float), D, f);
    }

    fread(tm->model.gain_final.C, sizeof(float), D, f);

    fclose(f);

    /* Sync */
    memcpy(tm->model.embed, tm->latent_embed, 256 * D * sizeof(float));
    trainable_requantize(tm);

    return 0;
}

#endif /* TRAIN_H */
