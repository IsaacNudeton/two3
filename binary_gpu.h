/*
 * binary_gpu.h — Tiled {2,3} matmul kernel
 *
 * GPU acceleration for ternary-at-readout weights. Two parallel bit masks:
 *   packed_plus  — bit set iff weight = +1
 *   packed_neg   — bit set iff weight = -1
 *   neither      — weight = 0 (substrate)
 *
 * Latent float in [0,1]; thresholds at 1/3 and 2/3 (binary.h).
 * Signed matmul: Y[s,m] = sum_{k: W[m,k]=+1} X[s,k] - sum_{k: W[m,k]=-1} X[s,k]
 *
 * The tiled kernel loads BOTH masks into shared memory and does one fused
 * sum per output element — avoids the two-pass approach (acc_plus - acc_neg).
 *
 * Isaac & Claude — April 2026
 */

#ifndef BINARY_GPU_H
#define BINARY_GPU_H

#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#ifdef TWO3_TENSOR_CORE
#include <mma.h>
#endif

/* ═══════════════════════════════════════════════════════
 * Tile dimensions — tuned for SM 7.5 (RTX 2080 Super)
 *
 * BTILE_M × BTILE_N = 16 × 16 = 256 threads per block
 * BTILE_K = 128 features per tile
 *
 * Shared memory per tile (two-mask version):
 *   activations:  BTILE_N × BTILE_K × sizeof(int8_t) = 16 × 128 = 2048 bytes
 *   plus weights: BTILE_M × (BTILE_K/32) × sizeof(uint32_t) = 256 bytes
 *   neg weights:  same = 256 bytes
 *   Total: 2560 bytes (same as old ternary path)
 * ═══════════════════════════════════════════════════════ */

#define BTILE_M 16
#define BTILE_N 16
#define BTILE_K 128

#ifndef BGPU_CHECK
#define BGPU_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err)); exit(1); } \
} while(0)
#endif

/* Two-threshold quantization constants (mirror binary.h) */
#ifndef TWO3_T_LOW
#define TWO3_T_LOW  (1.0f / 3.0f)
#endif
#ifndef TWO3_T_HIGH
#define TWO3_T_HIGH (2.0f / 3.0f)
#endif

/* ═══════════════════════════════════════════════════════
 * {2,3} GPU weight state: host + device copies of both masks,
 * plus the latent weights for backward STE.
 * ═══════════════════════════════════════════════════════ */

typedef struct {
    uint32_t *d_packed_plus;  /* device: +1 mask (bitmask, for backward dX transpose) */
    uint32_t *d_packed_neg;   /* device: -1 mask */
    uint32_t *h_packed_plus;  /* host: +1 mask */
    uint32_t *h_packed_neg;   /* host: -1 mask */
    int8_t   *d_W_tc;         /* device: INT8 dense {-1,0,+1} for WMMA forward (row-major [rows × cols]) */
    int8_t   *h_W_tc;         /* host mirror */
    float    *d_W_latent;     /* device: float latent weights (for backward STE) */
    float     density_plus;
    float     density_neg;
    int       rows;
    int       cols;
} BinaryWeightsGPU;

static inline float binary_gpu_active_density(const BinaryWeightsGPU *w) {
    return w->density_plus + w->density_neg;
}

/* ═══════════════════════════════════════════════════════
 * Tiled signed matmul: loads both masks, fused plus-minus
 *
 * Y[s][m] = sum_{k: W_plus[m,k]} X[s][k] - sum_{k: W_neg[m,k]} X[s][k]
 * ═══════════════════════════════════════════════════════ */

static __global__ void binary_matmul_tiled(
    const uint32_t* __restrict__ W_plus,
    const uint32_t* __restrict__ W_neg,
    const int8_t*   __restrict__ X,
    int32_t*        __restrict__ Y,
    int S, int M, int K
) {
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int m = blockIdx.x * BTILE_M + tx;
    int s = blockIdx.y * BTILE_N + ty;

    int packed_cols = (K + 31) / 32;
    int TK32 = BTILE_K / 32;

    __shared__ int8_t   sX[BTILE_N][BTILE_K];
    __shared__ uint32_t sWp[BTILE_M][BTILE_K / 32];
    __shared__ uint32_t sWn[BTILE_M][BTILE_K / 32];

    int32_t acc = 0;
    int n_tiles = (K + BTILE_K - 1) / BTILE_K;

    for (int tile = 0; tile < n_tiles; tile++) {
        int k_base = tile * BTILE_K;

        /* Load activations tile (cooperative). */
        {
            int tid = ty * BTILE_M + tx;
            #pragma unroll
            for (int i = 0; i < 8; i++) {
                int idx = tid * 8 + i;
                int load_n = idx / BTILE_K;
                int load_k = idx % BTILE_K;
                int global_s = blockIdx.y * BTILE_N + load_n;
                int global_k = k_base + load_k;
                if (global_s < S && global_k < K)
                    sX[load_n][load_k] = X[global_s * K + global_k];
                else
                    sX[load_n][load_k] = 0;
            }
        }

        /* Load both mask tiles (cooperative). */
        {
            int tid = ty * BTILE_M + tx;
            if (tid < BTILE_M * TK32) {
                int load_m = tid / TK32;
                int load_w = tid % TK32;
                int global_m = blockIdx.x * BTILE_M + load_m;
                int global_w = (k_base / 32) + load_w;
                if (global_m < M && global_w < packed_cols) {
                    sWp[load_m][load_w] = W_plus[global_m * packed_cols + global_w];
                    sWn[load_m][load_w] = W_neg [global_m * packed_cols + global_w];
                } else {
                    sWp[load_m][load_w] = 0;
                    sWn[load_m][load_w] = 0;
                }
            }
        }

        __syncthreads();

        /* Compute: signed masked sum from this tile. */
        if (m < M && s < S) {
            #pragma unroll
            for (int w = 0; w < TK32; w++) {
                uint32_t bp = sWp[tx][w];
                uint32_t bn = sWn[tx][w];
                int base_k = w * 32;
                #pragma unroll
                for (int b = 0; b < 32; b++) {
                    uint32_t mask = 1u << b;
                    int k = base_k + b;
                    if (k < BTILE_K) {
                        int8_t xv = sX[ty][k];
                        if (bp & mask) acc += (int32_t)xv;
                        if (bn & mask) acc -= (int32_t)xv;
                    }
                }
            }
        }

        __syncthreads();
    }

    if (m < M && s < S) {
        Y[s * M + m] = acc;
    }
}

/* ═══════════════════════════════════════════════════════
 * Tiled backward dX: signed transpose masked sum
 *
 * dX[s][k] = sum_{m: W_plus[m,k]} dY[s][m] - sum_{m: W_neg[m,k]} dY[s][m]
 *
 * Scaled by 1/sqrt(active_density × M) for CLT gradient balance.
 * active_density = density_plus + density_neg.
 * ═══════════════════════════════════════════════════════ */

#define BTILE_M_BW 64

static __global__ void binary_backward_dx_tiled(
    const float*    __restrict__ dY,
    const uint32_t* __restrict__ W_plus,
    const uint32_t* __restrict__ W_neg,
    float*          __restrict__ dX,
    int S, int M, int K,
    float bwd_scale
) {
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int k = blockIdx.x * BTILE_M + tx;
    int s = blockIdx.y * BTILE_N + ty;

    int packed_cols = (K + 31) / 32;

    __shared__ float sdY[BTILE_N][BTILE_M_BW];

    float sum = 0.0f;
    int n_tiles_m = (M + BTILE_M_BW - 1) / BTILE_M_BW;

    for (int tile = 0; tile < n_tiles_m; tile++) {
        int m_base = tile * BTILE_M_BW;

        {
            int tid = ty * BTILE_M + tx;
            #pragma unroll
            for (int i = 0; i < (BTILE_N * BTILE_M_BW) / (BTILE_M * BTILE_N); i++) {
                int idx = tid * ((BTILE_N * BTILE_M_BW) / (BTILE_M * BTILE_N)) + i;
                int load_n = idx / BTILE_M_BW;
                int load_m = idx % BTILE_M_BW;
                int global_s = blockIdx.y * BTILE_N + load_n;
                int global_m = m_base + load_m;
                if (global_s < S && global_m < M)
                    sdY[load_n][load_m] = dY[global_s * M + global_m];
                else
                    sdY[load_n][load_m] = 0.0f;
            }
        }

        __syncthreads();

        if (k < K && s < S) {
            int word = k / 32;
            uint32_t mask = 1u << (k % 32);

            #pragma unroll
            for (int mi = 0; mi < BTILE_M_BW; mi++) {
                int global_m = m_base + mi;
                if (global_m < M) {
                    uint32_t wp = W_plus[global_m * packed_cols + word];
                    uint32_t wn = W_neg [global_m * packed_cols + word];
                    float dyv = sdY[ty][mi];
                    if (wp & mask) sum += dyv;
                    if (wn & mask) sum -= dyv;
                }
            }
        }

        __syncthreads();
    }

    if (k < K && s < S) {
        dX[s * K + k] += sum * bwd_scale;
    }
}

/* ═══════════════════════════════════════════════════════
 * dW kernel — outer product dY^T @ X with STE clip.
 *
 * Latent lives in [0,1], clipped outside [-BINARY_STE_CLIP, 1+BINARY_STE_CLIP].
 * The ternary quantization thresholds at {1/3, 2/3} don't affect this kernel —
 * gradient flows to the latent, not to the packed form.
 * ═══════════════════════════════════════════════════════ */

#ifndef BINARY_STE_CLIP
#define BINARY_STE_CLIP 1.5f
#endif

#define BDWK_BLOCK 16

static __global__ void binary_backward_dw_tiled(
    const float* __restrict__ dY,
    const float* __restrict__ X,
    const float* __restrict__ W_latent,
    float*       __restrict__ dW,
    int S, int M, int K,
    float ste_clip
) {
    int m = blockIdx.y * BDWK_BLOCK + threadIdx.y;
    int k = blockIdx.x * BDWK_BLOCK + threadIdx.x;

    if (m >= M || k >= K) return;

    float w = W_latent[m * K + k];
    if (w < -ste_clip || w > 1.0f + ste_clip) return;

    float sum = 0.0f;
    for (int s = 0; s < S; s++)
        sum += dY[s * M + m] * X[s * K + k];

    dW[m * K + k] += sum;
}

/* ═══════════════════════════════════════════════════════
 * Single-shot backward (host alloc, reference path — not the fast path).
 * ═══════════════════════════════════════════════════════ */

static void binary_backward_batch_gpu(
    const float *dY,
    const float *X,
    const float *W_latent,
    const BinaryWeightsGPU *W,
    float *dX,
    float *dW_latent,
    int S, int M, int K
) {
    float *d_dY, *d_X, *d_dX, *d_W_latent, *d_dW;

    BGPU_CHECK(cudaMalloc(&d_dY, (size_t)S * M * sizeof(float)));
    BGPU_CHECK(cudaMalloc(&d_X,  (size_t)S * K * sizeof(float)));
    BGPU_CHECK(cudaMalloc(&d_dX, (size_t)S * K * sizeof(float)));
    BGPU_CHECK(cudaMalloc(&d_W_latent, (size_t)M * K * sizeof(float)));
    BGPU_CHECK(cudaMalloc(&d_dW, (size_t)M * K * sizeof(float)));

    BGPU_CHECK(cudaMemcpy(d_dY, dY, (size_t)S * M * sizeof(float), cudaMemcpyHostToDevice));
    BGPU_CHECK(cudaMemcpy(d_X, X, (size_t)S * K * sizeof(float), cudaMemcpyHostToDevice));
    BGPU_CHECK(cudaMemset(d_dX, 0, (size_t)S * K * sizeof(float)));
    BGPU_CHECK(cudaMemcpy(d_W_latent, W_latent, (size_t)M * K * sizeof(float), cudaMemcpyHostToDevice));
    BGPU_CHECK(cudaMemcpy(d_dW, dW_latent, (size_t)M * K * sizeof(float), cudaMemcpyHostToDevice));

    {
        float active = binary_gpu_active_density(W);
        float bwd_scale = 1.0f / sqrtf(fmaxf(active, 1e-6f) * (float)M);
        dim3 block(BTILE_M, BTILE_N);
        dim3 grid((K + BTILE_M - 1) / BTILE_M,
                  (S + BTILE_N - 1) / BTILE_N);
        binary_backward_dx_tiled<<<grid, block>>>(
            d_dY, W->d_packed_plus, W->d_packed_neg, d_dX, S, M, K, bwd_scale);
        BGPU_CHECK(cudaGetLastError());
    }

    {
        dim3 block(BDWK_BLOCK, BDWK_BLOCK);
        dim3 grid((K + BDWK_BLOCK - 1) / BDWK_BLOCK,
                  (M + BDWK_BLOCK - 1) / BDWK_BLOCK);
        binary_backward_dw_tiled<<<grid, block>>>(
            d_dY, d_X, d_W_latent, d_dW, S, M, K, BINARY_STE_CLIP);
        BGPU_CHECK(cudaGetLastError());
    }

    BGPU_CHECK(cudaDeviceSynchronize());

    {
        float *h_dX = (float*)malloc((size_t)S * K * sizeof(float));
        float *h_dW = (float*)malloc((size_t)M * K * sizeof(float));
        BGPU_CHECK(cudaMemcpy(h_dX, d_dX, (size_t)S * K * sizeof(float), cudaMemcpyDeviceToHost));
        BGPU_CHECK(cudaMemcpy(h_dW, d_dW, (size_t)M * K * sizeof(float), cudaMemcpyDeviceToHost));
        for (int i = 0; i < S * K; i++) dX[i] += h_dX[i];
        for (int i = 0; i < M * K; i++) dW_latent[i] = h_dW[i];
        free(h_dX);
        free(h_dW);
    }

    cudaFree(d_dY); cudaFree(d_X); cudaFree(d_dX);
    cudaFree(d_W_latent); cudaFree(d_dW);
}

/* ═══════════════════════════════════════════════════════
 * Pack float weights → two device masks.
 * ═══════════════════════════════════════════════════════ */

static BinaryWeightsGPU binary_pack_weights_gpu(const float *w_float, int rows, int cols) {
    BinaryWeightsGPU result;
    memset(&result, 0, sizeof(result));
    result.rows = rows;
    result.cols = cols;

    int packed_cols = (cols + 31) / 32;
    size_t packed_bytes = (size_t)rows * packed_cols * sizeof(uint32_t);
    size_t tc_bytes     = (size_t)rows * cols * sizeof(int8_t);

    result.h_packed_plus = (uint32_t*)calloc(rows * packed_cols, sizeof(uint32_t));
    result.h_packed_neg  = (uint32_t*)calloc(rows * packed_cols, sizeof(uint32_t));
    result.h_W_tc        = (int8_t*)  calloc(rows * cols,        sizeof(int8_t));

    int count_plus = 0;
    int count_neg  = 0;
    for (int m = 0; m < rows; m++) {
        for (int k = 0; k < cols; k++) {
            float w = w_float[m * cols + k];
            int word = k / 32;
            int bit  = k % 32;
            if (w >= TWO3_T_HIGH) {
                result.h_packed_plus[m * packed_cols + word] |= (1u << bit);
                result.h_W_tc[m * cols + k] = (int8_t)+1;
                count_plus++;
            } else if (w < TWO3_T_LOW) {
                result.h_packed_neg[m * packed_cols + word] |= (1u << bit);
                result.h_W_tc[m * cols + k] = (int8_t)-1;
                count_neg++;
            }
            /* else: substrate, h_W_tc stays 0 from calloc */
        }
    }
    result.density_plus = (float)count_plus / (float)(rows * cols);
    result.density_neg  = (float)count_neg  / (float)(rows * cols);

    BGPU_CHECK(cudaMalloc(&result.d_packed_plus, packed_bytes));
    BGPU_CHECK(cudaMalloc(&result.d_packed_neg,  packed_bytes));
    BGPU_CHECK(cudaMalloc(&result.d_W_tc,        tc_bytes));
    BGPU_CHECK(cudaMemcpy(result.d_packed_plus, result.h_packed_plus, packed_bytes, cudaMemcpyHostToDevice));
    BGPU_CHECK(cudaMemcpy(result.d_packed_neg,  result.h_packed_neg,  packed_bytes, cudaMemcpyHostToDevice));
    BGPU_CHECK(cudaMemcpy(result.d_W_tc,        result.h_W_tc,        tc_bytes,     cudaMemcpyHostToDevice));

    return result;
}

static void binary_free_weights_gpu(BinaryWeightsGPU *w) {
    if (w->d_packed_plus) cudaFree(w->d_packed_plus);
    if (w->d_packed_neg)  cudaFree(w->d_packed_neg);
    if (w->d_W_tc)        cudaFree(w->d_W_tc);
    if (w->h_packed_plus) free(w->h_packed_plus);
    if (w->h_packed_neg)  free(w->h_packed_neg);
    if (w->h_W_tc)        free(w->h_W_tc);
    if (w->d_W_latent)    cudaFree(w->d_W_latent);
    w->d_packed_plus = NULL;
    w->d_packed_neg  = NULL;
    w->d_W_tc        = NULL;
    w->h_packed_plus = NULL;
    w->h_packed_neg  = NULL;
    w->h_W_tc        = NULL;
    w->d_W_latent    = NULL;
}

/* Re-upload all weight formats after a requantize on host. */
static void binary_sync_to_device(BinaryWeightsGPU *w) {
    int packed_cols = (w->cols + 31) / 32;
    size_t packed_bytes = (size_t)w->rows * packed_cols * sizeof(uint32_t);
    size_t tc_bytes     = (size_t)w->rows * w->cols * sizeof(int8_t);
    BGPU_CHECK(cudaMemcpy(w->d_packed_plus, w->h_packed_plus, packed_bytes, cudaMemcpyHostToDevice));
    BGPU_CHECK(cudaMemcpy(w->d_packed_neg,  w->h_packed_neg,  packed_bytes, cudaMemcpyHostToDevice));
    BGPU_CHECK(cudaMemcpy(w->d_W_tc,        w->h_W_tc,        tc_bytes,     cudaMemcpyHostToDevice));
}

/* ═══════════════════════════════════════════════════════
 * Launch helpers
 * ═══════════════════════════════════════════════════════ */

static void binary_matmul_gpu(
    const uint32_t *d_W_plus,
    const uint32_t *d_W_neg,
    const int8_t *d_x_q,
    int32_t *d_acc,
    int S, int M, int K
) {
    dim3 block(BTILE_M, BTILE_N);
    dim3 grid((M + BTILE_M - 1) / BTILE_M,
              (S + BTILE_N - 1) / BTILE_N);
    binary_matmul_tiled<<<grid, block>>>(d_W_plus, d_W_neg, d_x_q, d_acc, S, M, K);
}

/* ═══════════════════════════════════════════════════════════════
 * Tensor Core forward path (TWO3_TENSOR_CORE)
 *
 * Turing INT8 WMMA: 16×16×16 fragments, one wmma.mma.sync per fragment,
 * 4096 MACs per instruction. Fragment layout:
 *   a_frag:  matrix_a, row_major — X[s_tile, k_tile] row-major [S×K], ld=K
 *   b_frag:  matrix_b, col_major — W[m_tile, k_tile] row-major [M×K] is
 *                                  equivalent to col-major B of shape [K×M]
 *                                  with ld=K
 *   c_frag:  accumulator, int32
 * Launch: one warp per 16×16 output tile.
 * Alignment requirement: M, S, K all multiples of 16.
 * ═══════════════════════════════════════════════════════════════ */

#ifdef TWO3_TENSOR_CORE

#define TC_WMMA_M 16
#define TC_WMMA_N 16
#define TC_WMMA_K 16

static __global__ void binary_matmul_wmma(
    const int8_t*  __restrict__ X,   /* [S, K] row-major */
    const int8_t*  __restrict__ W,   /* [M, K] row-major (= col-major [K, M]) */
    int32_t*       __restrict__ Y,   /* [S, M] row-major */
    int S, int M, int K
) {
    using namespace nvcuda::wmma;

    int tile_m = blockIdx.x;
    int tile_s = blockIdx.y;

    int s_base = tile_s * TC_WMMA_M;
    int m_base = tile_m * TC_WMMA_N;

    fragment<matrix_a,    TC_WMMA_M, TC_WMMA_N, TC_WMMA_K, signed char, row_major> a_frag;
    fragment<matrix_b,    TC_WMMA_M, TC_WMMA_N, TC_WMMA_K, signed char, col_major> b_frag;
    fragment<accumulator, TC_WMMA_M, TC_WMMA_N, TC_WMMA_K, int32_t>                c_frag;

    fill_fragment(c_frag, 0);

    for (int k_tile = 0; k_tile < K; k_tile += TC_WMMA_K) {
        const signed char *a_ptr = (const signed char*)(X + s_base * K + k_tile);
        const signed char *b_ptr = (const signed char*)(W + m_base * K + k_tile);
        load_matrix_sync(a_frag, a_ptr, K);
        load_matrix_sync(b_frag, b_ptr, K);
        mma_sync(c_frag, a_frag, b_frag, c_frag);
    }

    int32_t *c_ptr = Y + s_base * M + m_base;
    store_matrix_sync(c_ptr, c_frag, M, mem_row_major);
}

static inline bool binary_wmma_eligible(int S, int M, int K) {
    return (S % TC_WMMA_M == 0) && (M % TC_WMMA_N == 0) && (K % TC_WMMA_K == 0);
}

static bool binary_matmul_gpu_wmma(
    const int8_t *d_W_tc,
    const int8_t *d_x_q,
    int32_t *d_acc,
    int S, int M, int K
) {
    if (!binary_wmma_eligible(S, M, K)) return false;
    dim3 block(32);
    dim3 grid(M / TC_WMMA_N, S / TC_WMMA_M);
    binary_matmul_wmma<<<grid, block>>>(d_x_q, d_W_tc, d_acc, S, M, K);
    return true;
}

#endif /* TWO3_TENSOR_CORE */

/* Auto launcher: Tensor Core path if eligible + flag, otherwise bitmask kernel. */
static void binary_matmul_gpu_auto(
    const BinaryWeightsGPU *W,
    const int8_t *d_x_q,
    int32_t *d_acc,
    int S, int K
) {
    int M = W->rows;
#ifdef TWO3_TENSOR_CORE
    if (binary_matmul_gpu_wmma(W->d_W_tc, d_x_q, d_acc, S, M, K)) return;
#endif
    binary_matmul_gpu(W->d_packed_plus, W->d_packed_neg, d_x_q, d_acc, S, M, K);
}

/* Full forward: float input → quantize → tiled signed matmul → dequant → float output. */
static void binary_project_batch_gpu(
    const BinaryWeightsGPU *W,
    const float *input,
    float *output,
    int S, int K
) {
    int M = W->rows;

    int8_t *d_x_q;
    int32_t *d_acc;

    BGPU_CHECK(cudaMalloc(&d_x_q, (size_t)S * K * sizeof(int8_t)));
    BGPU_CHECK(cudaMalloc(&d_acc, (size_t)S * M * sizeof(int32_t)));

    int8_t *h_x_q = (int8_t*)malloc((size_t)S * K * sizeof(int8_t));
    float *h_scales = (float*)malloc(S * sizeof(float));

    for (int s = 0; s < S; s++) {
        float absmax = 0.0f;
        for (int k = 0; k < K; k++) {
            float v = fabsf(input[s * K + k]);
            if (v > absmax) absmax = v;
        }
        h_scales[s] = absmax;
        float inv = (absmax > 0.0f) ? 127.0f / absmax : 0.0f;
        for (int k = 0; k < K; k++) {
            float v = input[s * K + k] * inv;
            if (v > 127.0f) v = 127.0f;
            if (v < -127.0f) v = -127.0f;
            h_x_q[s * K + k] = (int8_t)v;
        }
    }

    BGPU_CHECK(cudaMemcpy(d_x_q, h_x_q, (size_t)S * K * sizeof(int8_t),
                          cudaMemcpyHostToDevice));

    BGPU_CHECK(cudaMemset(d_acc, 0, (size_t)S * M * sizeof(int32_t)));
    binary_matmul_gpu_auto(W, d_x_q, d_acc, S, K);

    int32_t *h_acc = (int32_t*)malloc((size_t)S * M * sizeof(int32_t));
    BGPU_CHECK(cudaMemcpy(h_acc, d_acc, (size_t)S * M * sizeof(int32_t),
                          cudaMemcpyDeviceToHost));

    float active = binary_gpu_active_density(W);
    float active_norm = 1.0f / sqrtf(fmaxf(active, 1e-6f) * (float)K);
    for (int s = 0; s < S; s++) {
        float a_scale = h_scales[s] / 127.0f;
        float combined = a_scale * active_norm;
        for (int m = 0; m < M; m++) {
            output[s * M + m] = (float)h_acc[s * M + m] * combined;
        }
    }

    cudaFree(d_x_q);
    cudaFree(d_acc);
    free(h_x_q);
    free(h_scales);
    free(h_acc);
}

/* Multi-output forward: quantize input ONCE, project through N weight matrices. */
static void binary_project_multi_gpu(
    const BinaryWeightsGPU *W_list[],
    float *output_list[],
    const float *input,
    int N, int S, int K
) {
    int8_t *h_x_q = (int8_t*)malloc((size_t)S * K * sizeof(int8_t));
    float *h_scales = (float*)malloc(S * sizeof(float));

    for (int s = 0; s < S; s++) {
        float absmax = 0.0f;
        for (int k = 0; k < K; k++) {
            float v = fabsf(input[s * K + k]);
            if (v > absmax) absmax = v;
        }
        h_scales[s] = absmax;
        float inv = (absmax > 0.0f) ? 127.0f / absmax : 0.0f;
        for (int k = 0; k < K; k++) {
            float v = input[s * K + k] * inv;
            if (v > 127.0f) v = 127.0f;
            if (v < -127.0f) v = -127.0f;
            h_x_q[s * K + k] = (int8_t)v;
        }
    }

    int8_t *d_x_q;
    BGPU_CHECK(cudaMalloc(&d_x_q, (size_t)S * K * sizeof(int8_t)));
    BGPU_CHECK(cudaMemcpy(d_x_q, h_x_q, (size_t)S * K * sizeof(int8_t),
                          cudaMemcpyHostToDevice));

    for (int n = 0; n < N; n++) {
        int M = W_list[n]->rows;
        int32_t *d_acc;
        BGPU_CHECK(cudaMalloc(&d_acc, (size_t)S * M * sizeof(int32_t)));
        BGPU_CHECK(cudaMemset(d_acc, 0, (size_t)S * M * sizeof(int32_t)));

        binary_matmul_gpu_auto(W_list[n], d_x_q, d_acc, S, K);

        int32_t *h_acc = (int32_t*)malloc((size_t)S * M * sizeof(int32_t));
        BGPU_CHECK(cudaMemcpy(h_acc, d_acc, (size_t)S * M * sizeof(int32_t),
                              cudaMemcpyDeviceToHost));

        float active = binary_gpu_active_density(W_list[n]);
        float active_norm = 1.0f / sqrtf(fmaxf(active, 1e-6f) * (float)K);
        for (int s = 0; s < S; s++) {
            float a_scale = h_scales[s] / 127.0f;
            float combined = a_scale * active_norm;
            float *yv = output_list[n] + s * M;
            for (int m = 0; m < M; m++)
                yv[m] = (float)h_acc[s * M + m] * combined;
        }

        cudaFree(d_acc);
        free(h_acc);
    }

    cudaFree(d_x_q);
    free(h_x_q);
    free(h_scales);
}

#endif /* BINARY_GPU_H */
