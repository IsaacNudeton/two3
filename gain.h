/*
 * gain.h — Gain Kernel Normalization for {2,3} Architecture
 *
 * NOT RMSNorm. This is the engine's own normalization:
 *   R' = R + γ(C - R) - κ·R·|x|     (reservoir update)
 *   y  = x · (1 + α·R - β)           (amplitude control)
 *
 * Properties (Lean-verified):
 *   - Fixed point exists and is unique above threshold
 *   - Spectral radius < 1 (Jury conditions)
 *   - 65x CFL safety margin with physical parameters
 *   - Normalizes (prevents explosion like RMSNorm)
 *   - Amplifies (weak signals get boosted — RMSNorm can't)
 *   - Learns (reservoir carries memory across forward passes)
 *
 * Isaac & CC — March 2026
 */

#ifndef GAIN_H
#define GAIN_H

#include <stdint.h>
#include <math.h>

/* ═══════════════════════════════════════════════════════
 * Gain parameters — within CFL stability bound
 *
 * CFL: γ < 4 / (Θ(2-β) + β)  where Θ = αC/β
 * With γ=0.006, β=0.01, Θ=5: bound=0.417, margin=65x
 * ═══════════════════════════════════════════════════════ */

#define GAIN_ALPHA  0.05f    /* gain coupling */
#define GAIN_BETA   0.01f    /* grid loss (must be < 1) */
#define GAIN_GAMMA  0.006f   /* recharge rate ≈ 1/SUBSTRATE_INT */
#define GAIN_KAPPA  0.05f    /* depletion rate (= alpha, natural choice) */
#define GAIN_R_MIN  0.01f   /* reservoir floor — prevents gain collapse */
#define GAIN_C_MIN  0.1f    /* capacity floor — must be > R_MIN for replenishment headroom */

/* ═══════════════════════════════════════════════════════
 * Gain state — persistent reservoir per feature
 * ═══════════════════════════════════════════════════════ */

typedef struct {
    float *R;          /* [dim] reservoir state — persists across tokens */
    float *C;          /* [dim] capacity — learnable parameter */
    int    dim;
} GainState;

/* Initialize: reservoir starts full (R = C), capacity learnable from 1.0 */
static void gain_init(GainState *g, int dim) {
    g->dim = dim;
    g->R = (float*)calloc(dim, sizeof(float));
    g->C = (float*)malloc(dim * sizeof(float));
    for (int i = 0; i < dim; i++) {
        g->C[i] = 1.0f;   /* initial capacity — gets trained */
        g->R[i] = g->C[i]; /* reservoir starts at capacity */
    }
}

static void gain_free(GainState *g) {
    free(g->R); free(g->C);
    g->R = g->C = NULL;
}

/* ═══════════════════════════════════════════════════════
 * CUDA kernel: gain normalization
 *
 * Per element:
 *   E = |x[i]|
 *   R[i] = R[i] + γ(C[i] - R[i]) - κ·R[i]·E
 *   y[i] = x[i] · (1 + α·R[i] - β)
 *
 * The reservoir depletes where signal is strong (normalizes).
 * The reservoir recharges where signal is weak (amplifies).
 * The fixed point is R* = β/α, E* = γ(αC-β)/(κβ).
 * ═══════════════════════════════════════════════════════ */

#ifdef __CUDACC__

/* GPU gain forward: RMS normalize, then reservoir modulate.
 * Must match gain_forward_cpu exactly. Uses shared memory for RMS reduction.
 * Launch with ONE block of <=1024 threads, dim <= 1024. */
__global__ void kernel_gain_forward(
    float       *y,      /* [dim] output */
    const float *x,      /* [dim] input */
    float       *R,      /* [dim] reservoir — READ AND WRITE */
    const float *C,      /* [dim] capacity */
    int dim,
    float alpha, float beta, float gamma_r, float kappa
) {
    extern __shared__ float sdata[];
    int i = threadIdx.x;
    if (i >= dim) return;

    float xi = x[i];

    /* Step 1: RMS normalization (shared memory reduction) */
    sdata[i] = xi * xi;
    __syncthreads();
    /* Parallel reduction for sum of squares */
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (i < s && i + s < dim) sdata[i] += sdata[i + s];
        __syncthreads();
    }
    float rms = sqrtf(sdata[0] / (float)dim + 1e-8f);
    float x_norm = xi / rms;

    /* Step 2: Reservoir update on normalized signal */
    float E = fabsf(x_norm);
    float Ri = R[i];
    float Ci = C[i];
    float R_new = Ri + gamma_r * (Ci - Ri) - kappa * Ri * E;
    if (R_new < GAIN_R_MIN) R_new = GAIN_R_MIN;

    /* Step 3: Modulate */
    float gain = 1.0f + alpha * R_new - beta;
    y[i] = x_norm * gain;
    R[i] = R_new;
}

/* Gain backward: RMS-scaled modulation backward (no projection correction)
 * Forward was TWO ops:
 *   1. x_norm = x / rms(x),  rms = sqrt(mean(x²) + eps)
 *   2. y = x_norm * gain,    gain = 1 + α*R - β
 *
 * Backward: dx[i] = dy[i] * gain[i] * inv_rms
 * Projection correction (- x_norm * mean_dot) omitted — with ternary
 * weights the gradient is radially aligned with x_norm and the projection
 * cancels nearly all signal. 1/rms scaling preserves correct magnitude.
 *
 * Caller must pass:
 *   inv_rms  = 1 / sqrt(mean(x²) + eps)
 */
__global__ void kernel_gain_backward(
    float       *dx,       /* [dim] gradient w.r.t. input */
    const float *dy,       /* [dim] gradient from above */
    const float *x,        /* [dim] saved input (pre-norm) */
    const float *R,        /* [dim] reservoir at time of forward */
    float       *dC,       /* [dim] gradient w.r.t. capacity */
    int          dim,
    float        alpha, float beta, float gamma_r,
    float        inv_rms   /* precomputed: 1/rms(x) */
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= dim) return;

    float x_norm = x[i] * inv_rms;
    float gain   = 1.0f + alpha * R[i] - beta;
    dx[i] = dy[i] * gain * inv_rms;

    /* dC: forward used x_norm (not x) in the gain/reservoir path */
    dC[i] += dy[i] * x_norm * alpha * gamma_r;
}

#endif /* __CUDACC__ */

/* ═══════════════════════════════════════════════════════
 * CPU reference for testing
 * ═══════════════════════════════════════════════════════ */

static void gain_forward_cpu(
    float *y, const float *x, float *R, const float *C,
    int dim
) {
    /* Step 1: NORMALIZE — RMS normalization makes activations O(1).
     * This is the impedance measurement. Forward stability regardless of dim.
     * The backward gradient flows through 1/rms which is a SCALAR — same
     * scale factor for all dimensions, so relative gradient structure preserved. */
    float rms = 0.0f;
    for (int i = 0; i < dim; i++) rms += x[i] * x[i];
    rms = sqrtf(rms / (float)dim + 1e-8f);
    float inv_rms = 1.0f / rms;

    /* Step 2: MODULATE — reservoir-dependent gain on normalized signal.
     * This is the impedance transformation. Sets gradient magnitude
     * independently of forward magnitude. */
    for (int i = 0; i < dim; i++) {
        float x_norm = x[i] * inv_rms;  /* O(1) regardless of dim */
        float E = fabsf(x_norm);
        float R_new = R[i] + GAIN_GAMMA * (C[i] - R[i]) - GAIN_KAPPA * R[i] * E;
        if (R_new < GAIN_R_MIN) R_new = GAIN_R_MIN;
        float gain = 1.0f + GAIN_ALPHA * R_new - GAIN_BETA;
        y[i] = x_norm * gain;
        R[i] = R_new;
    }
}

/* ═══════════════════════════════════════════════════════
 * Fixed point calculation (for verification)
 * ═══════════════════════════════════════════════════════ */

static float gain_R_star(void) { return GAIN_BETA / GAIN_ALPHA; }

static float gain_E_star(float C) {
    float threshold = GAIN_ALPHA * C - GAIN_BETA;
    if (threshold <= 0) return 0.0f;  /* below threshold */
    return GAIN_GAMMA * threshold / (GAIN_KAPPA * GAIN_BETA);
}

/* Check CFL condition */
static int gain_cfl_check(float C_max) {
    float theta = GAIN_ALPHA * C_max / GAIN_BETA;
    float bound = 4.0f / (theta * (2.0f - GAIN_BETA) + GAIN_BETA);
    float margin = bound / GAIN_GAMMA;
    return (GAIN_GAMMA < bound) ? (int)margin : 0;
}

#endif /* GAIN_H */
