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

#ifdef __CUDACC__
    BCUDA_CHECK(cudaMalloc(&result.packed, packed_total));
    BCUDA_CHECK(cudaMemcpy(result.packed, packed_host, packed_total,
                            cudaMemcpyHostToDevice));
#else
    result.packed = packed_host;
    packed_host = NULL;
#endif

    if (packed_host) free(packed_host);
    return result;
}

static void binary_free_weights(BinaryWeights *w) {
#ifdef __CUDACC__
    if (w->packed) cudaFree(w->packed);
#else
    if (w->packed) free(w->packed);
#endif
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

/* Print stats */
static void binary_print_stats(const BinaryWeights *w) {
    printf("[binary] %d×%d, density=%.1f%%, %d bytes packed\n",
           w->rows, w->cols, 100.0f * w->density,
           w->rows * ((w->cols + 31) / 32) * 4);
}

#endif /* BINARY_H */
