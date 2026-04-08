/*
 * binary_gpu.h — Tiled Binary Matmul Kernel for {2,3}
 *
 * GPU acceleration for binary {0,1} weight matrices.
 * Adapted from two3_tiled.h (ternary tiled kernel) and OB1 reduce
 * kernel patterns (326 GB/s on RTX 2080 Super).
 *
 * Key differences from ternary tiled kernel:
 *   - 1 bit per weight (32 per uint32) vs 2 bits (4 per byte)
 *   - No sign decode — just bit test and add
 *   - 8× denser packing: 32 weights per memory load vs 4
 *   - Shared memory for weights is 4× smaller
 *
 * Patterns from OB1 reduce kernel (326 GB/s):
 *   - Shared memory reduction
 *   - 256-thread blocks (tuned for SM 7.5 / Turing)
 *   - Coalesced global reads
 *   - No warp divergence (fully unrolled, no __ffs loop)
 *
 * Isaac & Claude — April 2026
 */

#ifndef BINARY_GPU_H
#define BINARY_GPU_H

#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>

/* ═══════════════════════════════════════════════════════
 * Tile dimensions — tuned for SM 7.5 (RTX 2080 Super)
 *
 * BTILE_M × BTILE_N = 16 × 16 = 256 threads per block
 * BTILE_K = 128 features per tile
 *
 * Shared memory per tile:
 *   activations: BTILE_N × BTILE_K × sizeof(int8_t) = 16 × 128 = 2048 bytes
 *   weights:     BTILE_M × (BTILE_K/32) × sizeof(uint32_t) = 16 × 4 × 4 = 256 bytes
 *   Total: 2304 bytes (vs 2560 for ternary — denser packing)
 *
 * 48 KB / 2304 = 21 tiles can coexist per SM.
 * Occupancy limited by warps (64/SM), not shared memory.
 * ═══════════════════════════════════════════════════════ */

#define BTILE_M 16    /* output features per block */
#define BTILE_N 16    /* tokens per block */
#define BTILE_K 128   /* input features per tile */

/* Error check macro */
#ifndef BGPU_CHECK
#define BGPU_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err)); exit(1); } \
} while(0)
#endif

/* ═══════════════════════════════════════════════════════
 * Tiled binary matmul kernel
 *
 * Y[s][m] = sum_{k where W[m,k]=1} X[s][k]
 *
 * Grid: (ceil(M/BTILE_M), ceil(S/BTILE_N))
 * Block: (BTILE_M, BTILE_N)
 *
 * No __ffs loop — fully unrolled bit tests.
 * No warp divergence. Predictable control flow.
 * ═══════════════════════════════════════════════════════ */

static __global__ void binary_matmul_tiled(
    const uint32_t* __restrict__ W,   /* [M, packed_cols] binary packed */
    const int8_t*   __restrict__ X,   /* [S, K] quantized int8 */
    int32_t*        __restrict__ Y,   /* [S, M] output accumulators */
    int S, int M, int K
) {
    int tx = threadIdx.x;  /* output feature within tile */
    int ty = threadIdx.y;  /* token within tile */

    int m = blockIdx.x * BTILE_M + tx;
    int s = blockIdx.y * BTILE_N + ty;

    int packed_cols = (K + 31) / 32;
    int TK32 = BTILE_K / 32;  /* uint32s per weight row per tile = 4 */

    /* Shared memory tiles */
    __shared__ int8_t   sX[BTILE_N][BTILE_K];          /* [16][128] = 2048 bytes */
    __shared__ uint32_t sW[BTILE_M][BTILE_K / 32];     /* [16][4]   = 256 bytes  */

    int32_t acc = 0;
    int n_tiles = (K + BTILE_K - 1) / BTILE_K;

    for (int tile = 0; tile < n_tiles; tile++) {
        int k_base = tile * BTILE_K;

        /* ── Cooperative load: activations ──
         * Total elements = BTILE_N × BTILE_K = 16 × 128 = 2048
         * Threads = 256. Each thread loads 2048/256 = 8 elements.
         * Coalesced: consecutive threads read consecutive bytes. */
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

        /* ── Cooperative load: weights ──
         * Total elements = BTILE_M × TK32 = 16 × 4 = 64 uint32s
         * Threads = 256. Only 64 loads needed — first 64 threads load.
         * Each uint32 is 4 bytes, coalesced within warp. */
        {
            int tid = ty * BTILE_M + tx;

            if (tid < BTILE_M * TK32) {
                int load_m = tid / TK32;
                int load_w = tid % TK32;

                int global_m = blockIdx.x * BTILE_M + load_m;
                int global_w = (k_base / 32) + load_w;

                if (global_m < M && global_w < packed_cols)
                    sW[load_m][load_w] = W[global_m * packed_cols + global_w];
                else
                    sW[load_m][load_w] = 0;
            }
        }

        __syncthreads();

        /* ── Compute: masked sum from this tile ──
         * For each uint32 of packed weights (4 words per tile):
         *   For each of 32 bits: if set, add corresponding activation.
         *
         * Fully unrolled inner loop. No __ffs. No while.
         * Every thread does the same number of operations. */
        if (m < M && s < S) {
            #pragma unroll
            for (int w = 0; w < TK32; w++) {
                uint32_t bits = sW[tx][w];
                int base_k = w * 32;

                /* Unroll 32 bit tests — no divergence.
                 * Compiler can use predicated adds (no branch). */
                #pragma unroll
                for (int b = 0; b < 32; b++) {
                    if (bits & (1u << b)) {
                        int k = base_k + b;
                        if (k < BTILE_K)  /* only needed at boundary tile */
                            acc += (int32_t)sX[ty][k];
                    }
                }
            }
        }

        __syncthreads();
    }

    /* Write result */
    if (m < M && s < S) {
        Y[s * M + m] = acc;
    }
}

/* ═══════════════════════════════════════════════════════
 * Tiled backward dX kernel
 *
 * dX[s][k] = sum_{m where W[m,k]=1} dY[s][m]
 *
 * Transpose of the forward: iterate over M in tiles.
 * Includes 1/sqrt(density*M) scaling for gradient balance.
 * ═══════════════════════════════════════════════════════ */

#define BTILE_M_BW 64  /* M dimension tile for backward */

static __global__ void binary_backward_dx_tiled(
    const float*    __restrict__ dY,      /* [S, M] */
    const uint32_t* __restrict__ W,       /* [M, packed_cols] */
    float*          __restrict__ dX,      /* [S, K] output */
    int S, int M, int K,
    float bwd_scale  /* 1/sqrt(density*M) */
) {
    int tx = threadIdx.x;  /* k within tile */
    int ty = threadIdx.y;  /* s within tile */

    int k = blockIdx.x * BTILE_M + tx;   /* using BTILE_M=16 for K dim */
    int s = blockIdx.y * BTILE_N + ty;

    int packed_cols = (K + 31) / 32;

    __shared__ float    sdY[BTILE_N][BTILE_M_BW];   /* [16][64] = 4096 bytes */
    __shared__ uint32_t sW_col[BTILE_M_BW];          /* [64] = 256 bytes */

    float sum = 0.0f;
    int n_tiles_m = (M + BTILE_M_BW - 1) / BTILE_M_BW;

    for (int tile = 0; tile < n_tiles_m; tile++) {
        int m_base = tile * BTILE_M_BW;

        /* Load dY tile */
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

        /* For this k, check which m's are connected and sum their dY */
        if (k < K && s < S) {
            int word = k / 32;
            uint32_t mask = 1u << (k % 32);

            #pragma unroll
            for (int mi = 0; mi < BTILE_M_BW; mi++) {
                int global_m = m_base + mi;
                if (global_m < M) {
                    uint32_t w_word = W[global_m * packed_cols + word];
                    if (w_word & mask)
                        sum += sdY[ty][mi];
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
 * GPU forward path: quantize → tiled matmul → dequantize
 *
 * Replaces binary_project_batch_cpu with GPU kernel.
 * Uses existing Two3Activations quantize path for input,
 * then calls binary_matmul_tiled instead of two3_forward.
 * ═══════════════════════════════════════════════════════ */

/* Dual-storage binary weights: host copy + device copy.
 * Host copy for CPU fallback/debug. Device copy for GPU kernels.
 * The BinaryWeights struct in binary.h stores one pointer.
 * This helper manages the dual copies. */

typedef struct {
    uint32_t *d_packed;   /* device copy */
    uint32_t *h_packed;   /* host copy */
    float     density;
    int       rows;
    int       cols;
} BinaryWeightsGPU;

static BinaryWeightsGPU binary_pack_weights_gpu(const float *w_float, int rows, int cols) {
    BinaryWeightsGPU result;
    result.rows = rows;
    result.cols = cols;

    int packed_cols = (cols + 31) / 32;
    size_t packed_bytes = (size_t)rows * packed_cols * sizeof(uint32_t);

    /* Pack on host */
    result.h_packed = (uint32_t*)calloc(rows * packed_cols, sizeof(uint32_t));
    int count_ones = 0;

    for (int m = 0; m < rows; m++) {
        for (int k = 0; k < cols; k++) {
            if (w_float[m * cols + k] > 0.5f) {
                int word = k / 32;
                int bit = k % 32;
                result.h_packed[m * packed_cols + word] |= (1u << bit);
                count_ones++;
            }
        }
    }
    result.density = (float)count_ones / (float)(rows * cols);

    /* Copy to device */
    BGPU_CHECK(cudaMalloc(&result.d_packed, packed_bytes));
    BGPU_CHECK(cudaMemcpy(result.d_packed, result.h_packed, packed_bytes,
                           cudaMemcpyHostToDevice));

    return result;
}

static void binary_free_weights_gpu(BinaryWeightsGPU *w) {
    if (w->d_packed) cudaFree(w->d_packed);
    if (w->h_packed) free(w->h_packed);
    w->d_packed = NULL;
    w->h_packed = NULL;
}

/* Update device copy after requantize (host weights changed) */
static void binary_sync_to_device(BinaryWeightsGPU *w) {
    int packed_cols = (w->cols + 31) / 32;
    size_t packed_bytes = (size_t)w->rows * packed_cols * sizeof(uint32_t);
    BGPU_CHECK(cudaMemcpy(w->d_packed, w->h_packed, packed_bytes,
                           cudaMemcpyHostToDevice));
}

/* ═══════════════════════════════════════════════════════
 * Launch helpers
 * ═══════════════════════════════════════════════════════ */

static void binary_matmul_gpu(
    const uint32_t *d_packed_W,   /* [M, packed_cols] device */
    const int8_t *d_x_q,         /* [S, K] device */
    int32_t *d_acc,               /* [S, M] device */
    int S, int M, int K
) {
    dim3 block(BTILE_M, BTILE_N);  /* 16 × 16 = 256 threads */
    dim3 grid((M + BTILE_M - 1) / BTILE_M,
              (S + BTILE_N - 1) / BTILE_N);

    binary_matmul_tiled<<<grid, block>>>(d_packed_W, d_x_q, d_acc, S, M, K);
}

/* Full forward: float input → quantize → tiled matmul → dequantize → float output.
 * Same interface as binary_project_batch_cpu but uses GPU. */
static void binary_project_batch_gpu(
    const BinaryWeightsGPU *W,
    const float *input,      /* [S, K] host */
    float *output,           /* [S, M] host */
    int S, int K
) {
    int M = W->rows;
    int packed_cols = (K + 31) / 32;

    /* Quantize input to int8 on device (reuse two3 quantize path) */
    /* Allocate device buffers */
    int8_t *d_x_q;
    float *d_scales;
    int32_t *d_acc;
    float *d_output;

    BGPU_CHECK(cudaMalloc(&d_x_q, (size_t)S * K * sizeof(int8_t)));
    BGPU_CHECK(cudaMalloc(&d_scales, S * sizeof(float)));
    BGPU_CHECK(cudaMalloc(&d_acc, (size_t)S * M * sizeof(int32_t)));

    /* Quantize on host, copy to device (simple path — GPU quantize is an optimization) */
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
    BGPU_CHECK(cudaMemcpy(d_scales, h_scales, S * sizeof(float),
                           cudaMemcpyHostToDevice));

    /* Tiled matmul on GPU */
    BGPU_CHECK(cudaMemset(d_acc, 0, (size_t)S * M * sizeof(int32_t)));
    binary_matmul_gpu(W->d_packed, d_x_q, d_acc, S, M, K);

    /* Dequantize on host (read back acc + scales) */
    int32_t *h_acc = (int32_t*)malloc((size_t)S * M * sizeof(int32_t));
    BGPU_CHECK(cudaMemcpy(h_acc, d_acc, (size_t)S * M * sizeof(int32_t),
                           cudaMemcpyDeviceToHost));

    for (int s = 0; s < S; s++) {
        float a_scale = h_scales[s] / 127.0f;
        /* Theorem 69a: CLT variance of masked sum → scale by 1/sqrt(density*K) */
        float combined = a_scale / sqrtf(W->density * (float)K);
        for (int m = 0; m < M; m++) {
            output[s * M + m] = (float)h_acc[s * M + m] * combined;
        }
    }

    /* Cleanup */
    cudaFree(d_x_q);
    cudaFree(d_scales);
    cudaFree(d_acc);
    free(h_x_q);
    free(h_scales);
    free(h_acc);
}

#endif /* BINARY_GPU_H */
