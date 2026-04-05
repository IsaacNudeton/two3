/*
 * ffn.h — Dense Feed-Forward Network for {2,3} Architecture
 *
 * Replaces moe.h. Same gate/up/down structure as a single expert,
 * but at full intermediate width. No router. No dispatch. No float
 * parameters. All ternary.
 *
 * Why dense over MoE at this scale:
 *   - Single GPU: MoE dispatch overhead > compute savings
 *   - Ternary training: MoE multiplies flip boundaries by 8x
 *   - Gain kernel already does continuous dimension-level routing
 *   - Self-Routing paper (April 2026): learned routers barely help
 *   - Hash Layer paper: random routing works nearly as well
 *   - VRAM: MoE latent weights + Adam states ~7GB, dense ~1.7GB
 *
 * Architecture per layer:
 *   x_norm = gain_forward(x)           ← continuous routing
 *   gate   = ternary_matmul(x_norm, W_gate)   [dim → intermediate]
 *   up     = ternary_matmul(x_norm, W_up)     [dim → intermediate]
 *   h      = squared_relu(gate) * up
 *   out    = ternary_matmul(h, W_down)         [intermediate → dim]
 *   x      = x + scale * out            ← residual
 *
 * The gain kernel before the FFN handles what the MoE router tried
 * to do: modulate which dimensions carry signal based on context.
 *
 * Isaac & CC — April 2026
 */

#ifndef FFN_H
#define FFN_H

#include "two3.h"
#include "activation.h"
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

/* ═══════════════════════════════════════════════════════
 * Dense FFN weights — three ternary matrices
 * ═══════════════════════════════════════════════════════ */

typedef struct {
    Two3Weights gate;    /* [intermediate, dim] ternary */
    Two3Weights up;      /* [intermediate, dim] ternary */
    Two3Weights down;    /* [dim, intermediate] ternary */
    int dim;
    int intermediate;

    /* Pre-allocated work buffers — allocated once in init,
     * reused every forward/backward. No malloc in the hot loop.
     * Sized for max_batch × intermediate. */
    float *buf_gate;     /* [max_batch × intermediate] */
    float *buf_up;       /* [max_batch × intermediate] */
    int    buf_capacity; /* max_batch this buffer supports */
} DenseFFN;

static void dense_ffn_init_buffers(DenseFFN *ffn, int max_batch) {
    ffn->buf_capacity = max_batch;
    ffn->buf_gate = (float*)malloc(max_batch * ffn->intermediate * sizeof(float));
    ffn->buf_up   = (float*)malloc(max_batch * ffn->intermediate * sizeof(float));
}

static void dense_ffn_free_buffers(DenseFFN *ffn) {
    free(ffn->buf_gate); ffn->buf_gate = NULL;
    free(ffn->buf_up);   ffn->buf_up = NULL;
    ffn->buf_capacity = 0;
}

/* ═══════════════════════════════════════════════════════
 * Forward — CPU reference
 *
 * gate = ternary_project(x, W_gate)       → [intermediate]
 * up   = ternary_project(x, W_up)         → [intermediate]
 * gate = squared_relu(gate)
 * h    = gate * up                        → [intermediate]
 * out  = ternary_project(h, W_down)       → [dim]
 *
 * Scale: 1/sqrt(dim) before squaring to prevent magnitude
 * explosion. Same pattern as the MoE expert forward.
 * ═══════════════════════════════════════════════════════ */

static void dense_ffn_forward(
    DenseFFN *ffn,
    const float *x,          /* [dim] normalized input */
    float *output,           /* [dim] output */
    int dim, int intermediate
) {
    float *gate_h = ffn->buf_gate;  /* pre-allocated */
    float *up_h   = ffn->buf_up;

    /* Gate + Up share input — quantize once, project 2x */
    {
        const Two3Weights *W_gu[2] = { &ffn->gate, &ffn->up };
        float *out_gu[2] = { gate_h, up_h };
        /* Quantize input ONCE, project against both weight matrices */
        Two3Activations X = two3_quantize_acts(x, 1, dim);
        for (int i = 0; i < 2; i++) {
            Two3Output Y = two3_forward(W_gu[i], &X);
            two3_dequantize_output(&Y, W_gu[i], &X, out_gu[i]);
            two3_free_output(&Y);
        }
        two3_free_acts(&X);
    }

    /* Scale before squaring — prevents magnitude explosion */
    float scale = 1.0f / sqrtf((float)dim);
    for (int i = 0; i < intermediate; i++) {
        gate_h[i] *= scale;
        up_h[i] *= scale;
    }

    /* Squared ReLU on gate */
    squared_relu_cpu(gate_h, gate_h, intermediate);

    /* Element-wise: gate * up */
    for (int i = 0; i < intermediate; i++)
        gate_h[i] *= up_h[i];

    /* Down projection + 1/sqrt(INTER) scale */
    {
        Two3Activations X = two3_quantize_acts(gate_h, 1, intermediate);
        Two3Output Y = two3_forward(&ffn->down, &X);
        two3_dequantize_output(&Y, &ffn->down, &X, output);
        two3_free_output(&Y);
        two3_free_acts(&X);
        float ds = 1.0f / sqrtf((float)intermediate);
        for (int i = 0; i < dim; i++) output[i] *= ds;
    }
}

/* ═══════════════════════════════════════════════════════
 * Batched forward — S tokens at once
 *
 * Same as above but processes [S × dim] input.
 * Single quantize + batch matmul for each projection.
 * ═══════════════════════════════════════════════════════ */

static void dense_ffn_forward_batch(
    DenseFFN *ffn,
    const float *x,          /* [S × dim] normalized input */
    float *output,           /* [S × dim] output */
    int S, int dim, int intermediate
) {
    /* Ensure buffers are large enough */
    if (S > ffn->buf_capacity) {
        fprintf(stderr, "FATAL: dense_ffn batch %d > buf_capacity %d\n",
                S, ffn->buf_capacity);
        exit(1);
    }

    float *gate_b = ffn->buf_gate;  /* pre-allocated */
    float *up_b   = ffn->buf_up;

    /* Quantize input ONCE, project gate and up */
    {
        Two3Activations X = two3_quantize_acts(x, S, dim);
        const Two3Weights *W_gu[2] = { &ffn->gate, &ffn->up };
        float *out_gu[2] = { gate_b, up_b };
        for (int i = 0; i < 2; i++) {
            Two3Output Y = two3_forward(W_gu[i], &X);
            two3_dequantize_output(&Y, W_gu[i], &X, out_gu[i]);
            two3_free_output(&Y);
        }
        two3_free_acts(&X);
    }

    /* Scale + squared ReLU + hadamard */
    float scale = 1.0f / sqrtf((float)dim);
    for (int i = 0; i < S * intermediate; i++) {
        gate_b[i] *= scale;
        up_b[i] *= scale;
    }
    for (int i = 0; i < S * intermediate; i++) {
        float g = gate_b[i];
        gate_b[i] = (g > 0.0f) ? g * g : 0.0f;
    }
    for (int i = 0; i < S * intermediate; i++)
        gate_b[i] *= up_b[i];

    /* Down projection — batch + 1/sqrt(INTER) scale */
    {
        Two3Activations X = two3_quantize_acts(gate_b, S, intermediate);
        Two3Output Y = two3_forward(&ffn->down, &X);
        two3_dequantize_output(&Y, &ffn->down, &X, output);
        two3_free_output(&Y);
        two3_free_acts(&X);
        float ds = 1.0f / sqrtf((float)intermediate);
        for (int i = 0; i < S * dim; i++) output[i] *= ds;
    }
}

#endif /* FFN_H */
