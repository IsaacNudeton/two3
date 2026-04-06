/*
 * binary.h — Binary weight kernel for {2,3}
 *
 * Weights are {0, 1}. Not {-1, 0, +1}.
 * 0 = not connected. 1 = connected.
 * The model learns topology. The gain kernel handles the rest.
 *
 * Sign lives in the signal. Topology lives in the weights.
 * One threshold (0.5). One boundary. Half the flip surface.
 * Opening and closing valves, not reversing polarity.
 *
 * Packing: 1 bit per weight, 32 per uint32.
 * Matmul: masked sum — add input where connected, skip where not.
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
 * Binary weight matrix
 * ═══════════════════════════════════════════════════════ */

typedef struct {
    uint32_t *packed;    /* [rows × packed_cols] device memory */
    float     density;   /* fraction of 1s (connection density) */
    int       rows;      /* output dimension (M) */
    int       cols;      /* input dimension (K) */
} BinaryWeights;

/* ═══════════════════════════════════════════════════════
 * Pack float weights to binary
 *
 * w > threshold → 1 (connected)
 * w <= threshold → 0 (not connected)
 *
 * 32 weights per uint32. Bit j of word i = weight at position i*32+j.
 * ═══════════════════════════════════════════════════════ */

#define BINARY_THRESHOLD 0.5f

#ifdef __CUDACC__
#define BCUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err)); exit(1); } \
} while(0)
#endif

static BinaryWeights binary_pack_weights(const float *w_float, int rows, int cols) {
    BinaryWeights result;
    result.rows = rows;
    result.cols = cols;

    int packed_cols = (cols + 31) / 32;
    int packed_total = rows * packed_cols * sizeof(uint32_t);

    uint32_t *packed_host = (uint32_t*)calloc(rows * packed_cols, sizeof(uint32_t));

    int count_ones = 0;
    int total = rows * cols;

    for (int m = 0; m < rows; m++) {
        for (int k = 0; k < cols; k++) {
            int connected = (w_float[m * cols + k] > BINARY_THRESHOLD) ? 1 : 0;
            if (connected) {
                int word = k / 32;
                int bit = k % 32;
                packed_host[m * packed_cols + word] |= (1u << bit);
                count_ones++;
            }
        }
    }

    result.density = (float)count_ones / (float)total;

    /* Always keep host copy for CPU matmul */
    result.packed = packed_host;
    return result;
}

static void binary_free_weights(BinaryWeights *w) {
    if (w->packed) free(w->packed);
    w->packed = NULL;
}

/* ═══════════════════════════════════════════════════════
 * Binary matmul: masked sum
 *
 * For each output row m:
 *   acc[m] = sum_{k where w[m,k]=1} x_q[k]
 *
 * Input: int8 quantized activations (same as ternary path)
 * Output: int32 accumulator (same as ternary path)
 *
 * The key difference: no sign decode. Just test bit and add.
 * On FPGA: AND gate. On GPU: __popc or bitwise.
 * ═══════════════════════════════════════════════════════ */

#ifdef __CUDACC__
/* GPU kernel: one thread per output element */
__global__ void kernel_binary_matmul(
    const uint32_t *packed_W,  /* [rows × packed_cols] */
    const int8_t *x_q,        /* [K] quantized input */
    int32_t *acc,              /* [rows] output accumulator */
    int rows, int cols
) {
    int m = blockIdx.x * blockDim.x + threadIdx.x;
    if (m >= rows) return;

    int packed_cols = (cols + 31) / 32;
    int32_t sum = 0;

    for (int w = 0; w < packed_cols; w++) {
        uint32_t bits = packed_W[m * packed_cols + w];
        int base_k = w * 32;

        /* Unroll: test each bit, add if connected */
        while (bits) {
            int bit = __ffs(bits) - 1;  /* find first set bit (0-indexed) */
            int k = base_k + bit;
            if (k < cols) sum += (int32_t)x_q[k];
            bits &= bits - 1;  /* clear lowest set bit */
        }
    }

    acc[m] = sum;
}

/* Batched: S input vectors */
__global__ void kernel_binary_matmul_batch(
    const uint32_t *packed_W,  /* [rows × packed_cols] */
    const int8_t *x_q,        /* [S × K] quantized inputs */
    int32_t *acc,              /* [S × rows] output accumulators */
    int S, int rows, int cols
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int s = idx / rows;
    int m = idx % rows;
    if (s >= S || m >= rows) return;

    int packed_cols = (cols + 31) / 32;
    const int8_t *xv = x_q + s * cols;
    int32_t sum = 0;

    for (int w = 0; w < packed_cols; w++) {
        uint32_t bits = packed_W[m * packed_cols + w];
        int base_k = w * 32;

        while (bits) {
            int bit = __ffs(bits) - 1;
            int k = base_k + bit;
            if (k < cols) sum += (int32_t)xv[k];
            bits &= bits - 1;
        }
    }

    acc[s * rows + m] = sum;
}
#endif /* __CUDACC__ */

/* CPU reference: masked sum */
static void binary_matmul_cpu(
    const uint32_t *packed_W,
    const int8_t *x_q,
    int32_t *acc,
    int rows, int cols
) {
    int packed_cols = (cols + 31) / 32;

    for (int m = 0; m < rows; m++) {
        int32_t sum = 0;
        for (int w = 0; w < packed_cols; w++) {
            uint32_t bits = packed_W[m * packed_cols + w];
            int base_k = w * 32;

            for (int bit = 0; bit < 32 && base_k + bit < cols; bit++) {
                if (bits & (1u << bit))
                    sum += (int32_t)x_q[base_k + bit];
            }
        }
        acc[m] = sum;
    }
}

/* ═══════════════════════════════════════════════════════
 * Dequantize: acc * (act_scale/127) * density_scale
 *
 * density_scale replaces the ternary absmean.
 * It accounts for the average connection density:
 * if 50% of weights are 1, the output magnitude is ~K/2
 * instead of K. density normalizes this.
 * ═══════════════════════════════════════════════════════ */

static void binary_dequantize(
    const int32_t *acc, int S, int M,
    const float *act_scales,  /* [S] per-vector absmax */
    float density,
    int K,                    /* input dimension */
    float *y_float            /* [S × M] output */
) {
    for (int s = 0; s < S; s++) {
        float a_scale = act_scales[s] / 127.0f;
        float combined = a_scale * density;
        for (int m = 0; m < M; m++) {
            y_float[s * M + m] = (float)acc[s * M + m] * combined;
        }
    }
}

/* ═══════════════════════════════════════════════════════
 * High-level forward: float in, float out
 * Handles quantization, matmul, and dequant internally.
 * ═══════════════════════════════════════════════════════ */

static void binary_project_cpu(
    const BinaryWeights *W,
    const float *input,     /* [K] */
    float *output,          /* [M] */
    int K
) {
    int M = W->rows;

    /* Quantize input to int8 */
    float absmax = 0.0f;
    for (int k = 0; k < K; k++) {
        float a = fabsf(input[k]);
        if (a > absmax) absmax = a;
    }
    if (absmax < 1e-10f) absmax = 1e-10f;

    int8_t *x_q = (int8_t*)malloc(K * sizeof(int8_t));
    for (int k = 0; k < K; k++)
        x_q[k] = (int8_t)(input[k] * 127.0f / absmax + (input[k] > 0 ? 0.5f : -0.5f));

    /* Binary matmul */
    int32_t *acc = (int32_t*)calloc(M, sizeof(int32_t));
    binary_matmul_cpu(W->packed, x_q, acc, M, K);

    /* Dequant */
    float scale = (absmax / 127.0f) * W->density;
    for (int m = 0; m < M; m++)
        output[m] = (float)acc[m] * scale;

    free(x_q);
    free(acc);
}

/* Batched forward: S vectors */
static void binary_project_batch_cpu(
    const BinaryWeights *W,
    const float *input,     /* [S × K] */
    float *output,          /* [S × M] */
    int S, int K
) {
    int M = W->rows;

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
        binary_matmul_cpu(W->packed, x_q, acc, M, K);

        float scale = (absmax / 127.0f) * W->density;
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
    const float *input,     /* [S × K] */
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
            binary_matmul_cpu(W_list[n]->packed, x_q, acc, M, K);

            float scale = (absmax / 127.0f) * W_list[n]->density;
            float *yv = output_list[n] + s * M;
            for (int m = 0; m < M; m++)
                yv[m] = (float)acc[m] * scale;

            free(acc);
        }
        free(x_q);
    }
}

/* ═══════════════════════════════════════════════════════
 * Backward: STE through binary weights
 *
 * dX[k] = sum_{m where w[m,k]=1} dY[m]  (transpose masked sum)
 * dW_latent[m,k] += dY[m] * X[k]         (outer product, STE-clipped)
 *
 * STE clip: zero gradient if latent weight too far from threshold.
 * ═══════════════════════════════════════════════════════ */

#define BINARY_STE_CLIP 1.5f

static void binary_backward_cpu(
    const float *dY,           /* [M] gradient from above */
    const float *X,            /* [K] saved input (float, pre-quantize) */
    const float *W_latent,     /* [M × K] latent float weights */
    const BinaryWeights *W_packed,
    float *dX,                 /* [K] gradient to pass back (ACCUMULATE) */
    float *dW_latent,          /* [M × K] gradient for latent (ACCUMULATE) */
    int M, int K
) {
    int packed_cols = (K + 31) / 32;

    /* dX = W^T @ dY — transpose masked sum, scaled by 1/sqrt(density*M).
     * Without scaling, binary backward sums O(density*M) terms without
     * sign cancellation (all positive connections). Ternary had natural
     * cancellation from ±1 weights → O(sqrt(M)). Binary needs explicit
     * sqrt normalization to match. */
    float bwd_scale = 1.0f / sqrtf(W_packed->density * (float)M + 1e-6f);
    for (int k = 0; k < K; k++) {
        float sum = 0.0f;
        int word = k / 32;
        uint32_t mask = 1u << (k % 32);
        for (int m = 0; m < M; m++) {
            if (W_packed->packed[m * packed_cols + word] & mask)
                sum += dY[m];
        }
        dX[k] += sum * bwd_scale;
    }

    /* dW_latent = dY @ X^T — outer product with STE clip */
    for (int m = 0; m < M; m++) {
        for (int k = 0; k < K; k++) {
            float g = dY[m] * X[k];
            /* STE clip: zero gradient if latent too far from any meaningful region */
            float w = W_latent[m * K + k];
            if (w < -BINARY_STE_CLIP || w > 1.0f + BINARY_STE_CLIP)
                g = 0.0f;
            dW_latent[m * K + k] += g;
        }
    }
}

/* Batched backward: S vectors */
static void binary_backward_batch_cpu(
    const float *dY,           /* [S × M] */
    const float *X,            /* [S × K] */
    const float *W_latent,     /* [M × K] */
    const BinaryWeights *W_packed,
    float *dX,                 /* [S × K] ACCUMULATE */
    float *dW_latent,          /* [M × K] ACCUMULATE */
    int S, int M, int K
) {
    for (int s = 0; s < S; s++) {
        binary_backward_cpu(
            dY + s * M, X + s * K, W_latent, W_packed,
            dX + s * K, dW_latent, M, K);
    }
}

/* Print stats */
static void binary_print_stats(const BinaryWeights *w) {
    printf("[binary] %d×%d, density=%.1f%%, %d bytes packed\n",
           w->rows, w->cols, 100.0f * w->density,
           w->rows * ((w->cols + 31) / 32) * 4);
}

#endif /* BINARY_H */
