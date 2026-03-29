/*
 * two3.cu — {2,3} Computing Kernel Implementation
 * Isaac Oravec & Claude | XYZT Computing Project | 2026
 *
 * Every operation in the forward pass is integer.
 * The only float ops are quantization (before) and dequantization (after).
 * The matmul itself: addition, subtraction, structure. That's it.
 */

#include "two3.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

/* ================================================================
 * Error checking
 * ================================================================ */

#define CUDA_CHECK(call) do {                                          \
    cudaError_t err = (call);                                          \
    if (err != cudaSuccess) {                                          \
        fprintf(stderr, "CUDA error at %s:%d — %s\n",                 \
                __FILE__, __LINE__, cudaGetErrorString(err));          \
        exit(1);                                                       \
    }                                                                  \
} while(0)

/* ================================================================
 * Device kernels
 * ================================================================ */

/*
 * Kernel: two3_matmul_kernel
 *
 * The heart. Y[s][m] = sum_k sign(W[m][k]) * X[s][k]
 *
 * Each thread computes one output element.
 * Weights are packed: 4 ternary values per byte.
 * Decode is branchless:
 *   bits & 1       → 1 if entity A (+1), 0 otherwise
 *   bits >> 1      → 1 if entity B (-1), 0 otherwise
 *   sign = (bits & 1) - (bits >> 1)  →  +1, -1, or 0
 *
 * On FPGA this sign computation becomes a 2-input mux.
 * On CUDA it's one AND, one SHIFT, one SUB, one IMUL, one IADD.
 * All integer pipeline. Zero float.
 */
__global__ void two3_matmul_kernel(
    const uint8_t* __restrict__ W,   /* [M, K/4] packed ternary      */
    const int8_t*  __restrict__ X,   /* [S, K]   quantized int8      */
    int32_t*       __restrict__ Y,   /* [S, M]   output accumulators */
    int S, int M, int K)
{
    int m = blockIdx.x * blockDim.x + threadIdx.x;  /* output feature   */
    int s = blockIdx.y * blockDim.y + threadIdx.y;  /* sequence position */

    if (m >= M || s >= S) return;

    int K4 = K / 4;
    int32_t acc = 0;

    for (int pk = 0; pk < K4; pk++) {
        uint8_t packed = W[m * K4 + pk];
        int base_k = pk * 4;

        #pragma unroll
        for (int j = 0; j < 4; j++) {
            uint8_t bits = (packed >> (j * 2)) & 0x3;

            /* {2,3} decode — branchless
             * 00 → sign=0  (substrate: skip)
             * 01 → sign=+1 (entity A: add)
             * 10 → sign=-1 (entity B: subtract) */
            int sign = (int)(bits & 1) - (int)(bits >> 1);

            /* On custom silicon this is a mux, not a multiply.
             * On CUDA, IMUL by {-1,0,+1} is 1 cycle. */
            acc += sign * (int32_t)X[s * K + base_k + j];
        }
    }

    Y[s * M + m] = acc;
}

/*
 * Kernel: quantize_activations_kernel
 *
 * Per-token absmax quantization: float → int8
 * Each thread handles one element. Scale is pre-computed per token.
 */
__global__ void quantize_acts_kernel(
    const float* __restrict__ x_float,   /* [S, K] input           */
    int8_t*      __restrict__ x_quant,   /* [S, K] output          */
    const float* __restrict__ scales,    /* [S]    per-token scales */
    int S, int K)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= S * K) return;

    int s = idx / K;
    float scale = scales[s];
    float inv_scale = (scale > 0.0f) ? (127.0f / scale) : 0.0f;

    float val = x_float[idx] * inv_scale;

    /* Clamp to [-127, 127] */
    val = fminf(fmaxf(val, -127.0f), 127.0f);
    x_quant[idx] = (int8_t)rintf(val);
}

/*
 * Kernel: find_absmax_kernel
 *
 * Find the maximum absolute value per token (per row).
 * Simple reduction — one block per token.
 */
__global__ void find_absmax_kernel(
    const float* __restrict__ x_float,  /* [S, K] */
    float*       __restrict__ scales,   /* [S]    */
    int K)
{
    extern __shared__ float sdata[];

    int s = blockIdx.x;                /* one block per token */
    int tid = threadIdx.x;
    int stride = blockDim.x;

    /* Each thread finds local max over its chunk */
    float local_max = 0.0f;
    for (int k = tid; k < K; k += stride) {
        float v = fabsf(x_float[s * K + k]);
        if (v > local_max) local_max = v;
    }

    sdata[tid] = local_max;
    __syncthreads();

    /* Tree reduction in shared memory */
    for (int s_red = blockDim.x / 2; s_red > 0; s_red >>= 1) {
        if (tid < s_red && sdata[tid + s_red] > sdata[tid]) {
            sdata[tid] = sdata[tid + s_red];
        }
        __syncthreads();
    }

    if (tid == 0) {
        scales[blockIdx.x] = sdata[0];
    }
}

/* ================================================================
 * Host functions
 * ================================================================ */

/* ---- Weight quantization and packing ---- */

/*
 * Ternary quantization: absmean scheme
 *   scale = mean(|W|)
 *   W_ternary = round(W / scale), clamped to {-1, 0, +1}
 *
 * Then pack 4 ternary values into each byte.
 */
Two3Weights two3_pack_weights(const float* w_float, int rows, int cols) {
    Two3Weights result;
    result.rows = rows;
    result.cols = cols;

    if (cols % 4 != 0) {
        fprintf(stderr, "two3: cols (%d) must be divisible by 4\n", cols);
        exit(1);
    }

    int total = rows * cols;
    int packed_cols = cols / 4;
    int packed_total = rows * packed_cols;

    /* Compute absmean scale */
    double sum_abs = 0.0;
    for (int i = 0; i < total; i++) {
        sum_abs += fabs((double)w_float[i]);
    }
    result.scale = (float)(sum_abs / total);

    if (result.scale < 1e-10f) {
        fprintf(stderr, "two3: weight scale is near zero\n");
        result.scale = 1e-10f;
    }

    float inv_scale = 1.0f / result.scale;

    /* Quantize to ternary and pack */
    uint8_t* packed_host = (uint8_t*)calloc(packed_total, 1);

    int count_zero = 0, count_plus = 0, count_minus = 0;

    for (int r = 0; r < rows; r++) {
        for (int pc = 0; pc < packed_cols; pc++) {
            uint8_t byte = 0;
            for (int j = 0; j < 4; j++) {
                int c = pc * 4 + j;
                float val = w_float[r * cols + c] * inv_scale;
                int q = (int)roundf(val);

                /* Clamp to {-1, 0, +1} */
                if (q > 1) q = 1;
                if (q < -1) q = -1;

                /* Encode: 0→0b00, +1→0b01, -1→0b10 */
                uint8_t bits;
                if (q == 0)       { bits = TWO3_ENCODE_ZERO;  count_zero++;  }
                else if (q == 1)  { bits = TWO3_ENCODE_PLUS;  count_plus++;  }
                else              { bits = TWO3_ENCODE_MINUS;  count_minus++; }

                byte |= (bits << (j * 2));
            }
            packed_host[r * packed_cols + pc] = byte;
        }
    }

    printf("[two3] weights %dx%d → packed %dx%d (%d bytes)\n",
           rows, cols, rows, packed_cols, packed_total);
    printf("[two3] substrate: %.1f%%  entity A(+1): %.1f%%  entity B(-1): %.1f%%\n",
           100.0f * count_zero  / total,
           100.0f * count_plus  / total,
           100.0f * count_minus / total);
    printf("[two3] weight scale (absmean): %.6f\n", result.scale);

    /* Copy to device */
    CUDA_CHECK(cudaMalloc(&result.packed, packed_total));
    CUDA_CHECK(cudaMemcpy(result.packed, packed_host, packed_total,
                           cudaMemcpyHostToDevice));

    free(packed_host);
    return result;
}

void two3_free_weights(Two3Weights* w) {
    if (w->packed) { cudaFree(w->packed); w->packed = NULL; }
}

/* ---- Activation quantization ---- */

Two3Activations two3_quantize_acts(const float* x_float, int tokens, int dim) {
    Two3Activations result;
    result.tokens = tokens;
    result.dim = dim;

    int total = tokens * dim;

    /* Allocate device memory */
    float* d_x_float;
    CUDA_CHECK(cudaMalloc(&d_x_float, total * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_x_float, x_float, total * sizeof(float),
                           cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMalloc(&result.scales, tokens * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&result.data, total * sizeof(int8_t)));

    /* Step 1: find absmax per token */
    int reduce_threads = 256;
    find_absmax_kernel<<<tokens, reduce_threads,
                         reduce_threads * sizeof(float)>>>(
        d_x_float, result.scales, dim);
    CUDA_CHECK(cudaGetLastError());

    /* Step 2: quantize to int8 */
    int quant_threads = 256;
    int quant_blocks = (total + quant_threads - 1) / quant_threads;
    quantize_acts_kernel<<<quant_blocks, quant_threads>>>(
        d_x_float, result.data, result.scales, tokens, dim);
    CUDA_CHECK(cudaGetLastError());

    cudaFree(d_x_float);
    return result;
}

void two3_free_acts(Two3Activations* a) {
    if (a->data)   { cudaFree(a->data);   a->data = NULL;   }
    if (a->scales) { cudaFree(a->scales); a->scales = NULL; }
}

/* ---- Forward pass ---- */

Two3Output two3_forward(const Two3Weights* W, const Two3Activations* X) {
    Two3Output result;
    result.tokens  = X->tokens;
    result.out_dim = W->rows;

    int S = X->tokens;
    int M = W->rows;

    CUDA_CHECK(cudaMalloc(&result.acc, S * M * sizeof(int32_t)));

    /* Launch the {2,3} kernel */
    dim3 block(TWO3_BLOCK_X, TWO3_BLOCK_Y);
    dim3 grid((M + block.x - 1) / block.x,
              (S + block.y - 1) / block.y);

    two3_matmul_kernel<<<grid, block>>>(
        W->packed, X->data, result.acc,
        S, M, W->cols);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    return result;
}

/* ---- Dequantize output ---- */

void two3_dequantize_output(const Two3Output* Y,
                            const Two3Weights* W,
                            const Two3Activations* X,
                            float* y_float) {
    int S = Y->tokens;
    int M = Y->out_dim;

    /* Copy accumulators to host */
    int32_t* acc_host = (int32_t*)malloc(S * M * sizeof(int32_t));
    CUDA_CHECK(cudaMemcpy(acc_host, Y->acc, S * M * sizeof(int32_t),
                           cudaMemcpyDeviceToHost));

    /* Copy activation scales to host */
    float* scales_host = (float*)malloc(S * sizeof(float));
    CUDA_CHECK(cudaMemcpy(scales_host, X->scales, S * sizeof(float),
                           cudaMemcpyDeviceToHost));

    /* Dequantize:
     * y_float[s][m] = acc[s][m] * (act_scale[s] / 127.0) * weight_scale
     *
     * The int8 activation was: x_q = round(x_float * 127 / absmax)
     * So x_float ≈ x_q * absmax / 127
     * The ternary weight was: w_t = round(w_float / absmean)
     * So w_float ≈ w_t * absmean
     * Therefore: y_float ≈ acc * (absmax/127) * absmean */
    float w_scale = W->scale;

    for (int s = 0; s < S; s++) {
        float a_scale = scales_host[s] / 127.0f;
        float combined = a_scale * w_scale;
        for (int m = 0; m < M; m++) {
            y_float[s * M + m] = (float)acc_host[s * M + m] * combined;
        }
    }

    free(acc_host);
    free(scales_host);
}

void two3_free_output(Two3Output* y) {
    if (y->acc) { cudaFree(y->acc); y->acc = NULL; }
}

/* ---- Reference matmul (CPU, float, for verification) ---- */

void two3_ref_matmul(const float* X, const float* W, float* C,
                     int S, int K, int M) {
    /* C[s][m] = sum_k X[s][k] * W[m][k]  (Y = X @ W^T) */
    for (int s = 0; s < S; s++) {
        for (int m = 0; m < M; m++) {
            double acc = 0.0;
            for (int k = 0; k < K; k++) {
                acc += (double)X[s * K + k] * (double)W[m * K + k];
            }
            C[s * M + m] = (float)acc;
        }
    }
}

/* ---- Stats ---- */

void two3_print_stats(const Two3Weights* w) {
    int packed_cols = w->cols / 4;
    int packed_total = w->rows * packed_cols;

    uint8_t* host_packed = (uint8_t*)malloc(packed_total);
    CUDA_CHECK(cudaMemcpy(host_packed, w->packed, packed_total,
                           cudaMemcpyDeviceToHost));

    int total = w->rows * w->cols;
    int counts[3] = {0, 0, 0}; /* substrate, +1, -1 */

    for (int i = 0; i < packed_total; i++) {
        uint8_t byte = host_packed[i];
        for (int j = 0; j < 4; j++) {
            uint8_t bits = (byte >> (j * 2)) & 0x3;
            if (bits < 3) counts[bits]++;
        }
    }

    printf("[two3] Weight distribution:\n");
    printf("  substrate (0):  %7d / %d  (%.1f%%)\n",
           counts[0], total, 100.0f * counts[0] / total);
    printf("  entity A (+1): %7d / %d  (%.1f%%)\n",
           counts[1], total, 100.0f * counts[1] / total);
    printf("  entity B (-1): %7d / %d  (%.1f%%)\n",
           counts[2], total, 100.0f * counts[2] / total);

    free(host_packed);
}
