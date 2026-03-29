/*
 * moe.h — Mixture of Experts Router for {2,3} Architecture
 *
 * 8 expert FFNs, top-2 routing per token.
 * Router weights stay float (16K params — precision matters here).
 * Expert weights are ternary (the heavy computation).
 *
 * Why MoE: 500M active / 2B total. Only 2 of 8 experts fire.
 * Per-token compute = 2x one expert = 2x dense FFN.
 * Total params = 8x, but memory for weights is cheap in ternary
 * (2 bits per weight, 8x compression vs FP16).
 *
 * Isaac & CC — March 2026
 */

#ifndef MOE_H
#define MOE_H

#include "two3.h"
#include <math.h>
#include <stdlib.h>

#define MOE_NUM_EXPERTS  8
#define MOE_TOP_K        2

/* ═══════════════════════════════════════════════════════
 * Router — float weights, softmax selection
 *
 * router_logits = x @ W_router  (float matmul, tiny: dim × 8)
 * top_k = argmax(softmax(logits), k=2)
 * ═══════════════════════════════════════════════════════ */

typedef struct {
    float *W;           /* [dim × n_experts] — float, not ternary */
    int    dim;
    int    n_experts;
} MoERouter;

typedef struct {
    int   expert_ids[MOE_TOP_K];     /* which experts fire */
    float expert_weights[MOE_TOP_K]; /* softmax weights for combining */
} MoESelection;

/* ═══════════════════════════════════════════════════════
 * Expert FFN — ternary gate/up/down per expert
 * ═══════════════════════════════════════════════════════ */

typedef struct {
    Two3Weights gate;   /* [dim, intermediate] */
    Two3Weights up;     /* [dim, intermediate] */
    Two3Weights down;   /* [intermediate, dim] */
} MoEExpert;

typedef struct {
    MoERouter   router;
    MoEExpert   experts[MOE_NUM_EXPERTS];
    int         dim;
    int         intermediate;
} MoELayer;

/* ═══════════════════════════════════════════════════════
 * Router forward — CPU is fine for 8-way selection
 * ═══════════════════════════════════════════════════════ */

static void moe_route(
    const MoERouter *router,
    const float *x,           /* [dim] hidden state */
    MoESelection *sel
) {
    int D = router->dim;
    int N = router->n_experts;

    /* Compute logits: x @ W_router */
    float logits[MOE_NUM_EXPERTS];
    for (int e = 0; e < N; e++) {
        float sum = 0;
        for (int d = 0; d < D; d++)
            sum += x[d] * router->W[d * N + e];
        logits[e] = sum;
    }

    /* Softmax */
    float max_l = logits[0];
    for (int e = 1; e < N; e++)
        if (logits[e] > max_l) max_l = logits[e];
    float sum_exp = 0;
    for (int e = 0; e < N; e++) {
        logits[e] = expf(logits[e] - max_l);
        sum_exp += logits[e];
    }
    for (int e = 0; e < N; e++)
        logits[e] /= sum_exp;

    /* Top-K selection */
    for (int k = 0; k < MOE_TOP_K; k++) {
        int best = 0;
        float best_val = -1e30f;
        for (int e = 0; e < N; e++) {
            /* Skip already selected */
            int skip = 0;
            for (int j = 0; j < k; j++)
                if (sel->expert_ids[j] == e) skip = 1;
            if (skip) continue;

            if (logits[e] > best_val) {
                best_val = logits[e];
                best = e;
            }
        }
        sel->expert_ids[k] = best;
        sel->expert_weights[k] = best_val;
    }

    /* Renormalize selected weights */
    float w_sum = 0;
    for (int k = 0; k < MOE_TOP_K; k++)
        w_sum += sel->expert_weights[k];
    if (w_sum > 0)
        for (int k = 0; k < MOE_TOP_K; k++)
            sel->expert_weights[k] /= w_sum;
}

/* ═══════════════════════════════════════════════════════
 * MoE forward — route to top-2 experts, combine outputs
 *
 * For each selected expert:
 *   gate = squared_relu(ternary_matmul(x, expert.gate))
 *   up   = ternary_matmul(x, expert.up)
 *   h    = gate * up
 *   out  = ternary_matmul(h, expert.down)
 *
 * Final = sum_k(weight_k * expert_k_output)
 * ═══════════════════════════════════════════════════════ */

/* CPU reference only — GPU would run two experts in parallel */
/* ═══════════════════════════════════════════════════════
 * Init / Free
 * ═══════════════════════════════════════════════════════ */

static void moe_router_init(MoERouter *r, int dim, int n_experts) {
    r->dim = dim;
    r->n_experts = n_experts;
    r->W = (float*)malloc(dim * n_experts * sizeof(float));
    /* Xavier init */
    float scale = 1.0f / sqrtf((float)dim);
    for (int i = 0; i < dim * n_experts; i++)
        r->W[i] = scale * (2.0f * (float)rand() / RAND_MAX - 1.0f);
}

static void moe_router_free(MoERouter *r) {
    free(r->W); r->W = NULL;
}

#ifdef __CUDACC__

/* GPU kernel: router logits (small matmul — dim × n_experts) */
__global__ void kernel_moe_router(
    float *logits,          /* [n_experts] output */
    const float *x,         /* [dim] input */
    const float *W,         /* [dim × n_experts] */
    int dim, int n_experts
) {
    int e = threadIdx.x;
    if (e >= n_experts) return;

    float sum = 0;
    for (int d = 0; d < dim; d++)
        sum += x[d] * W[d * n_experts + e];
    logits[e] = sum;
}

#endif /* __CUDACC__ */

#endif /* MOE_H */
