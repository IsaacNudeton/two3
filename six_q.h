/*
 * six_q.h — Six-Question Training Optimizations for {2,3}
 *
 * WHO:  Sparse optimizer — only update params near STE threshold
 * WHAT: Reservoir-weighted loss — hard tokens get full gradient
 * WHEN: (future) Adaptive compute from global depletion
 * WHERE: Skip backward on converged layers
 * WHY:  (future) Curriculum from reservoir difficulty
 * HOW:  (future) Flip-aware sparse backward
 *
 * Each optimization uses the same signal: the gain kernel reservoir.
 * One equation, six answers.
 *
 * Usage: #define TWO3_SIX_Q before including train.h
 *   Sub-toggles:
 *     TWO3_WEIGHTED_LOSS   — reservoir-weighted loss gradients
 *     TWO3_SPARSE_OPTIM    — skip optimizer on far-from-threshold params
 *     TWO3_LAYER_SKIP      — skip backward on converged layers
 *
 * Isaac & Claude — April 2026
 */

#ifndef SIX_Q_H
#define SIX_Q_H

#include <math.h>
#include <string.h>

#ifdef TWO3_SIX_Q
#ifndef TWO3_WEIGHTED_LOSS
#define TWO3_WEIGHTED_LOSS
#endif
#ifndef TWO3_SPARSE_OPTIM
#define TWO3_SPARSE_OPTIM
#endif
#ifndef TWO3_LAYER_SKIP
#define TWO3_LAYER_SKIP
#endif
#endif

/* ═══════════════════════════════════════════════════════
 * WHAT: Reservoir-weighted loss
 *
 * Easy tokens (low depletion) contribute 10% of gradient.
 * Hard tokens (high depletion) contribute 100%.
 *
 * The difficulty of position t is measured by how much
 * the last layer's gain kernel worked on it.
 * Proxy: use per-position loss relative to batch median.
 * High loss = model doesn't know = hard = full gradient.
 * Low loss = model already knows = easy = reduce gradient.
 *
 * Why loss-as-difficulty works: a token the model already
 * predicts correctly has low loss AND low reservoir depletion.
 * Both measure the same thing — how much work was needed.
 * Loss is already computed. Zero extra compute.
 * ═══════════════════════════════════════════════════════ */

#ifdef TWO3_WEIGHTED_LOSS

#define WLOSS_FLOOR 0.1f    /* easy tokens keep 10% gradient */
#define WLOSS_CEIL  1.0f    /* hard tokens keep 100% */

/* Compute per-position difficulty weights from losses.
 * weights[t] ∈ [FLOOR, CEIL] based on loss[t] relative to median.
 * Tokens below median get floor. Tokens above get scaled to ceil. */
static void compute_loss_weights(
    float *weights,         /* [T] output weights */
    const float *losses,    /* [T] per-position losses */
    int T
) {
    if (T <= 0) return;

    /* Find median loss via partial sort (selection) */
    float *sorted = (float*)malloc(T * sizeof(float));
    memcpy(sorted, losses, T * sizeof(float));

    /* Simple selection for median — O(T log T) but T is small (seq_len) */
    for (int i = 0; i < T; i++)
        for (int j = i + 1; j < T; j++)
            if (sorted[j] < sorted[i]) {
                float tmp = sorted[i];
                sorted[i] = sorted[j];
                sorted[j] = tmp;
            }

    float median = sorted[T / 2];
    free(sorted);

    /* Scale: below median → FLOOR, above median → linear to CEIL */
    if (median < 1e-6f) median = 1e-6f;  /* avoid div by zero */

    for (int t = 0; t < T; t++) {
        float ratio = losses[t] / median;
        if (ratio <= 1.0f) {
            weights[t] = WLOSS_FLOOR;
        } else {
            /* Linear interpolation from FLOOR to CEIL as ratio goes 1→2+ */
            float alpha = (ratio - 1.0f);  /* 0 at median, 1 at 2x median */
            if (alpha > 1.0f) alpha = 1.0f;
            weights[t] = WLOSS_FLOOR + alpha * (WLOSS_CEIL - WLOSS_FLOOR);
        }
    }
}

/* Apply weights to loss gradients in-place.
 * d_logits_all[t * 256 + b] *= weights[t] for all bytes b. */
static void apply_loss_weights(
    float *d_logits_all,    /* [T × 256] gradients, modified in-place */
    const float *weights,   /* [T] per-position weights */
    int T
) {
    for (int t = 0; t < T; t++) {
        float w = weights[t];
        for (int b = 0; b < 256; b++)
            d_logits_all[t * 256 + b] *= w;
    }
}

#endif /* TWO3_WEIGHTED_LOSS */


/* ═══════════════════════════════════════════════════════
 * WHO: Sparse optimizer
 *
 * A ternary weight only matters when it FLIPS — crosses
 * the STE threshold from one quantization region to another.
 * Params far from any threshold won't flip this step.
 * Their Adam update is wasted compute.
 *
 * Skip optimizer on params where distance to nearest
 * threshold exceeds the maximum possible Adam step.
 * The maximum Adam step is lr * 1.0 (when m_hat/sqrt(v_hat) ≈ 1).
 *
 * This doesn't skip the gradient computation (needed for
 * downstream layers). It skips the OPTIMIZER UPDATE.
 *
 * Three thresholds to check distance from:
 *   +STE_THRESHOLD (0.33)  — +1/0 boundary
 *   -STE_THRESHOLD (-0.33) — -1/0 boundary
 *    0.0                    — not a threshold but the center of 0-region
 * ═══════════════════════════════════════════════════════ */

#ifdef TWO3_SPARSE_OPTIM

#define SPARSE_MARGIN_MULT  3.0f  /* skip if distance > lr * this */

/* Compute a skip mask for optimizer: 1 = update, 0 = skip.
 * Returns count of active (non-skipped) params. */
static int compute_sparse_mask(
    uint8_t *mask,              /* [size] output: 1=update, 0=skip */
    const float *latent,        /* [size] current latent weights */
    int size,
    float lr,                   /* current learning rate */
    float ste_threshold         /* STE_THRESHOLD */
) {
    float max_step = lr * SPARSE_MARGIN_MULT;
    int active = 0;

    for (int i = 0; i < size; i++) {
        float w = latent[i];

        /* Distance to nearest quantization boundary */
        float d_pos = fabsf(w - ste_threshold);   /* +1/0 boundary */
        float d_neg = fabsf(w + ste_threshold);    /* -1/0 boundary */
        float d_min = (d_pos < d_neg) ? d_pos : d_neg;

        /* Also check STE_CLIP boundary (beyond which gradient is zero) */
        float d_clip = fabsf(fabsf(w) - 1.5f);    /* STE_CLIP = 1.5 */

        if (d_min < max_step || d_clip < max_step) {
            mask[i] = 1;
            active++;
        } else {
            mask[i] = 0;
        }
    }

    return active;
}

/* Sparse Adam: only update masked elements.
 * m and v still get updated for ALL elements (momentum continuity).
 * Only the param update (w -= lr * ...) is skipped for masked-out elements. */
static void adam_update_sparse(
    float *params,
    const float *grads,
    AdamState *s,
    const uint8_t *mask,    /* 1=update param, 0=skip param update */
    int step,
    float lr,
    float beta1, float beta2, float eps
) {
    float b1_corr = 1.0f / (1.0f - powf(beta1, (float)step));
    float b2_corr = 1.0f / (1.0f - powf(beta2, (float)step));

    for (int i = 0; i < s->size; i++) {
        float g = grads[i];

        /* Always update moments (needed for future steps) */
        s->m[i] = beta1 * s->m[i] + (1.0f - beta1) * g;
        s->v[i] = beta2 * s->v[i] + (1.0f - beta2) * g * g;

        /* Only update param if near threshold */
        if (mask[i]) {
            float m_hat = s->m[i] * b1_corr;
            float v_hat = s->v[i] * b2_corr;
            params[i] -= lr * m_hat / (sqrtf(v_hat) + eps);
        }
    }
}

#endif /* TWO3_SPARSE_OPTIM */


/* ═══════════════════════════════════════════════════════
 * WHERE: Skip backward on converged layers
 *
 * A layer is converged when its gain reservoir stays
 * near capacity — low depletion means the layer isn't
 * doing much work. Backpropagating through it is waste.
 *
 * The gradient still passes THROUGH the skip (via the
 * residual connection), but weight gradients for the
 * layer's projections are not computed.
 *
 * Track per-layer convergence across steps. A layer is
 * marked converged after N consecutive steps with low
 * depletion. It can unconverge if depletion rises.
 * ═══════════════════════════════════════════════════════ */

#ifdef TWO3_LAYER_SKIP

#define LSKIP_DEPLETION_THRESH  0.05f   /* mean(C-R)/C below this = converged */
#define LSKIP_CONSECUTIVE       50      /* steps of low depletion before skip */
#define LSKIP_MIN_LAYER         2       /* never skip first 2 layers */

typedef struct {
    int   *consecutive_low;     /* [n_layers] consecutive steps below threshold */
    int   *converged;           /* [n_layers] 1 = skip backward for this layer */
    int    n_layers;
    int    total_skipped;       /* running count of skipped backward passes */
} LayerSkipState;

static void layer_skip_init(LayerSkipState *ls, int n_layers) {
    ls->n_layers = n_layers;
    ls->consecutive_low = (int*)calloc(n_layers, sizeof(int));
    ls->converged = (int*)calloc(n_layers, sizeof(int));
    ls->total_skipped = 0;
}

static void layer_skip_free(LayerSkipState *ls) {
    free(ls->consecutive_low);
    free(ls->converged);
}

/* Update convergence state after a training step.
 * Call after forward pass, before backward pass.
 * Returns number of layers to skip this step. */
static int layer_skip_update(
    LayerSkipState *ls,
    const GainState *gain_attn,  /* [n_layers] attention gain states */
    const GainState *gain_ffn,   /* [n_layers] FFN gain states */
    int dim
) {
    int skip_count = 0;

    for (int l = 0; l < ls->n_layers; l++) {
        /* Compute mean relative depletion for this layer */
        float dep_attn = 0, dep_ffn = 0;
        for (int i = 0; i < dim; i++) {
            float c_a = gain_attn[l].C[i];
            float c_f = gain_ffn[l].C[i];
            if (c_a > 1e-6f) dep_attn += (c_a - gain_attn[l].R[i]) / c_a;
            if (c_f > 1e-6f) dep_ffn  += (c_f - gain_ffn[l].R[i]) / c_f;
        }
        float mean_dep = (dep_attn + dep_ffn) / (2.0f * dim);

        if (mean_dep < LSKIP_DEPLETION_THRESH) {
            ls->consecutive_low[l]++;
        } else {
            ls->consecutive_low[l] = 0;
            ls->converged[l] = 0;  /* unconverge if depletion rose */
        }

        /* Mark converged after N consecutive low-depletion steps */
        if (ls->consecutive_low[l] >= LSKIP_CONSECUTIVE && l >= LSKIP_MIN_LAYER) {
            ls->converged[l] = 1;
        }

        if (ls->converged[l]) skip_count++;
    }

    return skip_count;
}

/* Check if a layer's backward should be skipped.
 * Even when skipping, gradient flows through residual connection. */
static int layer_skip_should_skip(const LayerSkipState *ls, int layer) {
    return ls->converged[layer];
}

#endif /* TWO3_LAYER_SKIP */


/* ═══════════════════════════════════════════════════════
 * Integration guide — where these plug into train.h
 *
 * 1. WEIGHTED LOSS (WHAT):
 *    In trainable_forward_backward, after the loss loop:
 *
 *    #ifdef TWO3_WEIGHTED_LOSS
 *    float *per_pos_loss = (float*)malloc(T * sizeof(float));
 *    // (save loss per position during loss loop)
 *    float *loss_weights = (float*)malloc(T * sizeof(float));
 *    compute_loss_weights(loss_weights, per_pos_loss, T);
 *    apply_loss_weights(d_logits_all, loss_weights, T);
 *    free(per_pos_loss); free(loss_weights);
 *    #endif
 *
 * 2. SPARSE OPTIMIZER (WHO):
 *    In trainable_optimizer_step, for each weight tensor:
 *
 *    #ifdef TWO3_SPARSE_OPTIM
 *    uint8_t *mask = (uint8_t*)malloc(size);
 *    int active = compute_sparse_mask(mask, latent, size, tm->lr, STE_THRESHOLD);
 *    adam_update_sparse(latent, grads, &adam, mask, tm->step, ...);
 *    free(mask);
 *    // Log: printf("  sparse: %d/%d active (%.1f%%)\n", active, size, 100.f*active/size);
 *    #else
 *    adam_update(latent, grads, &adam, tm->step, ...);
 *    #endif
 *
 * 3. LAYER SKIP (WHERE):
 *    Add LayerSkipState to TrainableModel.
 *    In trainable_forward_backward, before backward layer loop:
 *
 *    #ifdef TWO3_LAYER_SKIP
 *    GainState gain_attn_arr[MAX_LAYERS], gain_ffn_arr[MAX_LAYERS];
 *    for (int l = 0; l < L; l++) {
 *        gain_attn_arr[l] = tm->model.layers[l].gain_attn;
 *        gain_ffn_arr[l]  = tm->model.layers[l].gain_ffn;
 *    }
 *    int skipped = layer_skip_update(&tm->layer_skip, gain_attn_arr, gain_ffn_arr, D);
 *    #endif
 *
 *    Inside backward layer loop, at top:
 *
 *    #ifdef TWO3_LAYER_SKIP
 *    if (layer_skip_should_skip(&tm->layer_skip, l)) {
 *        // Skip all weight gradient computation
 *        // But still propagate d_hidden through residual:
 *        // d_hidden is unchanged (residual = identity in backward)
 *        tm->layer_skip.total_skipped++;
 *        continue;
 *    }
 *    #endif
 *
 * ═══════════════════════════════════════════════════════ */

#endif /* SIX_Q_H */
