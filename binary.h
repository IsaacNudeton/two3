/*
 * binary.h — {2,3} weight kernel
 *
 * Latent weights float in [0,1]. The "binary" name is historical —
 * packing is now ternary-at-readout, with two thresholds at exactly
 * 1/3 and 2/3. Three output states: -1, 0, +1.
 *
 * Training dynamics stay binary-style (Adam in [0,1] with headroom),
 * but the quantization boundary honors the physics:
 *   cos²θ = 1/3 confined fraction (reflect, -1)
 *   sin²θ = 2/3 free fraction (transmit, +1)
 *   the gap between is substrate (not connected, 0)
 *
 * Latent mapping:
 *   latent < 1/3            → -1  (reflect / confined)
 *   1/3 ≤ latent < 2/3      →  0  (substrate / not connected)
 *   latent ≥ 2/3            → +1  (transmit / free)
 *
 * Packing: 2 bits per weight (two parallel masks), 32 weights per uint32.
 * Matmul: masked sum plus minus masked sum neg = sum_plus(X) - sum_neg(X).
 *
 * The w=0.5 inversion vs old binary:
 *   old binary: w=0.5 was the flip boundary (max mobility)
 *   new two3:   w=0.5 is mid-substrate (min mobility — an attractor)
 *
 * Isaac & CC — April 2026
 */

#ifndef BINARY_H
#define BINARY_H

#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>

/* ═══════════════════════════════════════════════════════
 * Ternary thresholds — from the cube diagonal projection
 * cos²θ = 1/3, sin²θ = 2/3 (Why_Three.lean, complete_map.md)
 * ═══════════════════════════════════════════════════════ */

#define TWO3_T_LOW  (1.0f / 3.0f)
#define TWO3_T_HIGH (2.0f / 3.0f)

/* Latent valid range extension for STE.
 * The latent lives in [0,1] but gradient can push it temporarily outside;
 * clip dead gradient far outside the valid region. */
#ifndef BINARY_STE_CLIP
#define BINARY_STE_CLIP 1.5f
#endif

/* ═══════════════════════════════════════════════════════
 * Ternary weight matrix — two parallel bit masks
 * packed_plus bit set  ↔ weight is +1
 * packed_neg  bit set  ↔ weight is -1
 * neither set          ↔ weight is  0 (substrate)
 * Both set is impossible (invariant maintained by pack function)
 * ═══════════════════════════════════════════════════════ */

typedef struct {
    uint32_t *packed_plus;     /* [rows × packed_cols] +1 mask */
    uint32_t *packed_neg;      /* [rows × packed_cols] -1 mask */
    float     density_plus;    /* fraction of +1s */
    float     density_neg;     /* fraction of -1s */
    int       rows;            /* output dimension (M) */
    int       cols;            /* input dimension (K) */
} BinaryWeights;

/* Active density = fraction of non-zero weights (both +1 and -1).
 * At uniform latent init, each third gets 1/3 → active = 2/3.
 * After training, the model finds its own distribution. */
static inline float binary_active_density(const BinaryWeights *w) {
    return w->density_plus + w->density_neg;
}

#ifdef __CUDACC__
#define BCUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err)); exit(1); } \
} while(0)
#endif

/* ═══════════════════════════════════════════════════════
 * Pack float weights → two-mask ternary
 *
 * latent < 1/3            → -1  (set packed_neg bit)
 * 1/3 ≤ latent < 2/3      →  0  (neither bit set)
 * latent ≥ 2/3            → +1  (set packed_plus bit)
 * ═══════════════════════════════════════════════════════ */

static BinaryWeights binary_pack_weights(const float *w_float, int rows, int cols) {
    BinaryWeights result;
    result.rows = rows;
    result.cols = cols;

    int packed_cols = (cols + 31) / 32;

    uint32_t *plus_host = (uint32_t*)calloc(rows * packed_cols, sizeof(uint32_t));
    uint32_t *neg_host  = (uint32_t*)calloc(rows * packed_cols, sizeof(uint32_t));

    int count_plus = 0;
    int count_neg  = 0;
    int total = rows * cols;

    for (int m = 0; m < rows; m++) {
        for (int k = 0; k < cols; k++) {
            float w = w_float[m * cols + k];
            int word = k / 32;
            int bit  = k % 32;
            if (w >= TWO3_T_HIGH) {
                plus_host[m * packed_cols + word] |= (1u << bit);
                count_plus++;
            } else if (w < TWO3_T_LOW) {
                neg_host[m * packed_cols + word] |= (1u << bit);
                count_neg++;
            }
            /* else: substrate, both masks stay 0 */
        }
    }

    result.density_plus = (float)count_plus / (float)total;
    result.density_neg  = (float)count_neg  / (float)total;
    result.packed_plus  = plus_host;
    result.packed_neg   = neg_host;
    return result;
}

static void binary_free_weights(BinaryWeights *w) {
    if (w->packed_plus) free(w->packed_plus);
    if (w->packed_neg)  free(w->packed_neg);
    w->packed_plus = NULL;
    w->packed_neg  = NULL;
}

/* ═══════════════════════════════════════════════════════
 * CPU matmul: signed masked sum
 *
 * acc[m] = sum_{k: W[m,k]=+1} x_q[k]  -  sum_{k: W[m,k]=-1} x_q[k]
 *
 * Input: int8 quantized activations
 * Output: int32 accumulator
 * ═══════════════════════════════════════════════════════ */

static void binary_matmul_cpu(
    const uint32_t *packed_plus,
    const uint32_t *packed_neg,
    const int8_t *x_q,
    int32_t *acc,
    int rows, int cols
) {
    int packed_cols = (cols + 31) / 32;

    for (int m = 0; m < rows; m++) {
        int32_t sum = 0;
        for (int w = 0; w < packed_cols; w++) {
            uint32_t bp = packed_plus[m * packed_cols + w];
            uint32_t bn = packed_neg [m * packed_cols + w];
            int base_k = w * 32;
            for (int bit = 0; bit < 32 && base_k + bit < cols; bit++) {
                uint32_t mask = 1u << bit;
                if (bp & mask) sum += (int32_t)x_q[base_k + bit];
                if (bn & mask) sum -= (int32_t)x_q[base_k + bit];
            }
        }
        acc[m] = sum;
    }
}

/* ═══════════════════════════════════════════════════════
 * Dequantize: acc * (act_scale/127) / sqrt(active_density × K)
 *
 * Active density = P(+1) + P(-1), i.e. fraction of non-zero weights.
 * At uniform init: 2/3 = sin²θ (free fraction, the two3 number).
 * The √ normalization comes from CLT — signed sum over active_density×K
 * terms has std ~ sqrt(active_density × K).
 * ═══════════════════════════════════════════════════════ */

static void binary_dequantize(
    const int32_t *acc, int S, int M,
    const float *act_scales,
    float active_density,
    int K,
    float *y_float
) {
    float norm = 1.0f / sqrtf(fmaxf(active_density, 1e-6f) * (float)K);
    for (int s = 0; s < S; s++) {
        float a_scale = act_scales[s] / 127.0f;
        float combined = a_scale * norm;
        for (int m = 0; m < M; m++) {
            y_float[s * M + m] = (float)acc[s * M + m] * combined;
        }
    }
}

/* ═══════════════════════════════════════════════════════
 * High-level forward: float in, float out
 * ═══════════════════════════════════════════════════════ */

static void binary_project_cpu(
    const BinaryWeights *W,
    const float *input,
    float *output,
    int K
) {
    int M = W->rows;

    float absmax = 0.0f;
    for (int k = 0; k < K; k++) {
        float a = fabsf(input[k]);
        if (a > absmax) absmax = a;
    }
    if (absmax < 1e-10f) absmax = 1e-10f;

    int8_t *x_q = (int8_t*)malloc(K * sizeof(int8_t));
    for (int k = 0; k < K; k++)
        x_q[k] = (int8_t)(input[k] * 127.0f / absmax + (input[k] > 0 ? 0.5f : -0.5f));

    int32_t *acc = (int32_t*)calloc(M, sizeof(int32_t));
    binary_matmul_cpu(W->packed_plus, W->packed_neg, x_q, acc, M, K);

    float active = binary_active_density(W);
    float scale = (absmax / 127.0f) / sqrtf(fmaxf(active, 1e-6f) * (float)K);
    for (int m = 0; m < M; m++)
        output[m] = (float)acc[m] * scale;

    free(x_q);
    free(acc);
}

/* Batched forward: S vectors */
static void binary_project_batch_cpu(
    const BinaryWeights *W,
    const float *input,
    float *output,
    int S, int K
) {
    int M = W->rows;
    float active = binary_active_density(W);
    float active_norm = 1.0f / sqrtf(fmaxf(active, 1e-6f) * (float)K);

    for (int s = 0; s < S; s++) {
        const float *xv = input + s * K;
        float *yv = output + s * M;

        float absmax = 0.0f;
        for (int k = 0; k < K; k++) {
            float a = fabsf(xv[k]);
            if (a > absmax) absmax = a;
        }
        if (absmax < 1e-10f) absmax = 1e-10f;

        int8_t *x_q = (int8_t*)malloc(K * sizeof(int8_t));
        for (int k = 0; k < K; k++)
            x_q[k] = (int8_t)(xv[k] * 127.0f / absmax + (xv[k] > 0 ? 0.5f : -0.5f));

        int32_t *acc = (int32_t*)calloc(M, sizeof(int32_t));
        binary_matmul_cpu(W->packed_plus, W->packed_neg, x_q, acc, M, K);

        float scale = (absmax / 127.0f) * active_norm;
        for (int m = 0; m < M; m++)
            yv[m] = (float)acc[m] * scale;

        free(x_q);
        free(acc);
    }
}

/* Multi-project: quantize input ONCE, project against N weight matrices */
static void binary_project_multi_cpu(
    const BinaryWeights **W_list,
    float **output_list,
    const float *input,
    int N, int S, int K
) {
    for (int s = 0; s < S; s++) {
        const float *xv = input + s * K;

        float absmax = 0.0f;
        for (int k = 0; k < K; k++) {
            float a = fabsf(xv[k]);
            if (a > absmax) absmax = a;
        }
        if (absmax < 1e-10f) absmax = 1e-10f;

        int8_t *x_q = (int8_t*)malloc(K * sizeof(int8_t));
        for (int k = 0; k < K; k++)
            x_q[k] = (int8_t)(xv[k] * 127.0f / absmax + (xv[k] > 0 ? 0.5f : -0.5f));

        for (int n = 0; n < N; n++) {
            int M = W_list[n]->rows;
            int32_t *acc = (int32_t*)calloc(M, sizeof(int32_t));
            binary_matmul_cpu(W_list[n]->packed_plus, W_list[n]->packed_neg, x_q, acc, M, K);

            float active = binary_active_density(W_list[n]);
            float scale = (absmax / 127.0f) / sqrtf(fmaxf(active, 1e-6f) * (float)K);
            float *yv = output_list[n] + s * M;
            for (int m = 0; m < M; m++)
                yv[m] = (float)acc[m] * scale;

            free(acc);
        }
        free(x_q);
    }
}

/* ═══════════════════════════════════════════════════════
 * Backward: STE through {2,3} weights
 *
 * dX[k] = sum_{m: W[m,k]=+1} dY[m]  -  sum_{m: W[m,k]=-1} dY[m]
 *         all scaled by 1/sqrt(active_density × M)
 *
 * dW_latent[m,k] += dY[m] * X[k], clipped outside [-clip, 1+clip].
 *
 * The STE clip keeps the same latent range [0,1]+padding as old binary —
 * the latent space didn't change, only where the quantization cuts live.
 * ═══════════════════════════════════════════════════════ */

static void binary_backward_cpu(
    const float *dY,
    const float *X,
    const float *W_latent,
    const BinaryWeights *W_packed,
    float *dX,
    float *dW_latent,
    int M, int K
) {
    int packed_cols = (K + 31) / 32;

    /* dX with ternary sign cancellation — CLT √(active×M) */
    float active = binary_active_density(W_packed);
    float bwd_scale = 1.0f / sqrtf(fmaxf(active, 1e-6f) * (float)M);

    for (int k = 0; k < K; k++) {
        float sum = 0.0f;
        int word = k / 32;
        uint32_t mask = 1u << (k % 32);
        for (int m = 0; m < M; m++) {
            uint32_t bp = W_packed->packed_plus[m * packed_cols + word];
            uint32_t bn = W_packed->packed_neg [m * packed_cols + word];
            if (bp & mask) sum += dY[m];
            if (bn & mask) sum -= dY[m];
        }
        dX[k] += sum * bwd_scale;
    }

    /* dW_latent: outer product with STE clip.
     * Clip stays on the latent extremes — it kills dead gradient when
     * the latent has drifted far outside [0,1], not on the ternary boundaries. */
    for (int m = 0; m < M; m++) {
        for (int k = 0; k < K; k++) {
            float g = dY[m] * X[k];
            float w = W_latent[m * K + k];
            if (w < -BINARY_STE_CLIP || w > 1.0f + BINARY_STE_CLIP)
                g = 0.0f;
            dW_latent[m * K + k] += g;
        }
    }
}

static void binary_backward_batch_cpu(
    const float *dY,
    const float *X,
    const float *W_latent,
    const BinaryWeights *W_packed,
    float *dX,
    float *dW_latent,
    int S, int M, int K
) {
    for (int s = 0; s < S; s++) {
        binary_backward_cpu(
            dY + s * M, X + s * K, W_latent, W_packed,
            dX + s * K, dW_latent, M, K);
    }
}

/* ═══════════════════════════════════════════════════════
 * Stats: report distribution across the three states
 * ═══════════════════════════════════════════════════════ */

static void binary_print_stats(const BinaryWeights *w) {
    float zero_frac = 1.0f - w->density_plus - w->density_neg;
    printf("[two3] %d×%d  -1=%.1f%%  0=%.1f%%  +1=%.1f%%  active=%.1f%%  %d bytes\n",
           w->rows, w->cols,
           100.0f * w->density_neg,
           100.0f * zero_frac,
           100.0f * w->density_plus,
           100.0f * (w->density_plus + w->density_neg),
           2 * w->rows * ((w->cols + 31) / 32) * 4);
}

#endif /* BINARY_H */
