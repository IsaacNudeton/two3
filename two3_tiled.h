/*
 * two3_tiled.h — Tiled {2,3} Ternary Matmul Kernel
 *
 * Drop-in replacement for two3_matmul_kernel.
 * Uses shared memory tiling to reduce global memory traffic ~8×.
 *
 * The insight: ternary matmul is memory-bound, not compute-bound.
 * Each weight is 2 bits. Each activation is 8 bits. The "multiply"
 * is a sign select (1 cycle). The bottleneck is getting data to
 * the ALU. Tiling fixes that.
 *
 * Tuned for SM 7.5 (RTX 2080 Super):
 *   - 48 KB shared memory per SM
 *   - 64 warps per SM
 *   - 1024 threads per block max
 *
 * Isaac & CC — March 2026
 */

#ifndef TWO3_TILED_H
#define TWO3_TILED_H

#include <cuda_runtime.h>
#include <stdint.h>

/*
 * Tile dimensions.
 *
 * TILE_M × TILE_N = threads per block = 16×16 = 256
 * TILE_K = how many input features we process per tile = 128
 *   → 128 int8 activations = 128 bytes per row in shared mem
 *   → 128/4 = 32 packed weight bytes per row in shared mem
 *
 * Shared memory per tile:
 *   activations: TILE_N × TILE_K × sizeof(int8_t) = 16 × 128 = 2048 bytes
 *   weights:     TILE_M × (TILE_K/4) × sizeof(uint8_t) = 16 × 32 = 512 bytes
 *   Total: 2560 bytes — well within 48KB limit
 *
 * Global memory reads per output element:
 *   Before: K/4 weight bytes + K activation bytes (all from global)
 *   After:  K/TILE_K tiles × (TILE_K/4 + TILE_K)/TILE_M loads
 *   For K=1024: 8 tiles vs 256+1024 individual loads. ~8× reduction.
 */

#define TILE_M 16   /* output features per block */
#define TILE_N 16   /* tokens per block */
#define TILE_K 128  /* input features per tile */

/*
 * Tiled ternary matmul kernel.
 *
 * Y[s][m] = sum_k sign(W[m][k]) * X[s][k]
 *
 * Grid: (ceil(M/TILE_M), ceil(S/TILE_N))
 * Block: (TILE_M, TILE_N)
 *
 * Each thread computes one output element by accumulating
 * partial sums across K/TILE_K tiles.
 */
static __global__ void two3_matmul_tiled(
    const uint8_t* __restrict__ W,   /* [M, K/4] packed ternary */
    const int8_t*  __restrict__ X,   /* [S, K]   quantized int8 */
    int32_t*       __restrict__ Y,   /* [S, M]   output accumulators */
    int S, int M, int K)
{
    /* Thread position within the block */
    int tx = threadIdx.x;  /* which output feature within tile (0..TILE_M-1) */
    int ty = threadIdx.y;  /* which token within tile (0..TILE_N-1) */

    /* Global output position */
    int m = blockIdx.x * TILE_M + tx;
    int s = blockIdx.y * TILE_N + ty;

    int K4 = K / 4;
    int TK4 = TILE_K / 4;  /* packed bytes per tile = 32 */

    /* Shared memory tiles */
    __shared__ int8_t  sX[TILE_N][TILE_K];       /* [16][128] = 2048 bytes */
    __shared__ uint8_t sW[TILE_M][TILE_K / 4];   /* [16][32]  = 512 bytes  */

    int32_t acc = 0;

    /* Iterate over tiles along K dimension */
    int n_tiles = (K + TILE_K - 1) / TILE_K;

    for (int tile = 0; tile < n_tiles; tile++) {
        int k_base = tile * TILE_K;

        /* ── Cooperative load: activations ── */
        /* Total elements = TILE_N * TILE_K. Threads = TILE_M * TILE_N. */
        #define ACT_LOADS_PER_THREAD ((TILE_N * TILE_K) / (TILE_M * TILE_N))
        {
            int tid = ty * TILE_M + tx;  /* linear thread ID */

            #pragma unroll
            for (int i = 0; i < ACT_LOADS_PER_THREAD; i++) {
                int idx = tid * ACT_LOADS_PER_THREAD + i;
                int load_n = idx / TILE_K;  /* which row (token) */
                int load_k = idx % TILE_K;  /* which column (feature) */

                int global_s = blockIdx.y * TILE_N + load_n;
                int global_k = k_base + load_k;

                if (global_s < S && global_k < K)
                    sX[load_n][load_k] = X[global_s * K + global_k];
                else
                    sX[load_n][load_k] = 0;
            }
        }

        /* ── Cooperative load: weights ── */
        /* Total elements = TILE_M * TK4 = 16 * 32 = 512 */
        /* Threads per block = 256 */
        /* Each thread loads 512/256 = 2 elements */
        {
            int tid = ty * TILE_M + tx;

            #pragma unroll
            for (int i = 0; i < 2; i++) {
                int idx = tid * 2 + i;
                int load_m = idx / TK4;
                int load_pk = idx % TK4;

                int global_m = blockIdx.x * TILE_M + load_m;
                int global_pk = (k_base / 4) + load_pk;

                if (global_m < M && global_pk < K4)
                    sW[load_m][load_pk] = W[global_m * K4 + global_pk];
                else
                    sW[load_m][load_pk] = 0;
            }
        }

        __syncthreads();

        /* ── Compute partial sum from this tile ── */
        if (m < M && s < S) {
            #pragma unroll
            for (int pk = 0; pk < TK4; pk++) {
                uint8_t packed = sW[tx][pk];

                /* Unroll 4 weights per byte */
                #pragma unroll
                for (int j = 0; j < 4; j++) {
                    uint8_t bits = (packed >> (j * 2)) & 0x3;
                    /* {2,3} decode: 00→0, 01→+1, 10→-1 */
                    int sign = (int)(bits & 1) - (int)(bits >> 1);
                    acc += sign * (int32_t)sX[ty][pk * 4 + j];
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

/*
 * Tiled backward dX kernel.
 *
 * dX[s][k] = sum_m dY[s][m] * sign(W[m][k])
 *
 * Transposed ternary matmul: iterate over M dimension.
 * Tiles BOTH dY and W into shared memory.
 *
 * Tile dimensions (reuse forward tile sizes where possible):
 *   Block: TILE_M × TILE_N = 16 × 16 threads
 *   Thread (tx, ty) → output element dX[s, k]
 *     tx = k within tile, ty = s within tile
 *   Tile over M in chunks of TILE_M_BW = 64
 *
 * Shared memory per tile:
 *   dY:  TILE_N × TILE_M_BW × sizeof(float) = 16 × 64 × 4 = 4096 bytes
 *   W:   TILE_M_BW × (TILE_M/4) × sizeof(uint8_t) = 64 × 4 = 256 bytes
 *   Total: 4352 bytes — well within 48KB
 *
 * Global memory reads per output element:
 *   Before: M reads of dY + M reads of W (all from global)
 *   After:  M/TILE_M_BW tiles, shared within block
 */
#define TILE_M_BW 64  /* M-dimension tile for backward */

static __global__ void two3_backward_dx_tiled(
    const uint8_t* __restrict__ W,    /* [M, K/4] packed ternary */
    const float*   __restrict__ dY,   /* [S, M]   gradient */
    float*         __restrict__ dX,   /* [S, K]   output gradient (accumulate) */
    int S, int M, int K)
{
    int tx = threadIdx.x;  /* k within tile (0..TILE_M-1, reused as K tile) */
    int ty = threadIdx.y;  /* s within tile (0..TILE_N-1) */

    int k = blockIdx.x * TILE_M + tx;
    int s = blockIdx.y * TILE_N + ty;

    int K4 = K / 4;
    int j_shift = (k < K) ? (k % 4) * 2 : 0;

    float acc = 0.0f;

    /* Iterate over M in tiles */
    for (int m_base = 0; m_base < M; m_base += TILE_M_BW) {
        /* Shared memory for this tile */
        __shared__ float   sdY[TILE_N][TILE_M_BW];   /* [16][64] = 4KB */
        __shared__ uint8_t sW[TILE_M_BW][4];          /* [64][4]  = 256B — 4 packed bytes per M row */

        /* ── Cooperative load: dY tile [TILE_N × TILE_M_BW] ── */
        /* Total: 16 × 64 = 1024 floats. Threads: 256. Each loads 4. */
        {
            int tid = ty * TILE_M + tx;
            #pragma unroll
            for (int i = 0; i < 4; i++) {
                int idx = tid * 4 + i;
                int ln = idx / TILE_M_BW;   /* which s row */
                int lm = idx % TILE_M_BW;   /* which m col */
                int gs = blockIdx.y * TILE_N + ln;
                int gm = m_base + lm;
                sdY[ln][lm] = (gs < S && gm < M) ? dY[gs * M + gm] : 0.0f;
            }
        }

        /* ── Cooperative load: W tile [TILE_M_BW × 4 packed bytes] ── */
        /* For this block's k range (blockIdx.x * TILE_M .. +15),
         * the packed byte columns needed are pk0..pk0+3 (since 16/4=4).
         * Total: 64 × 4 = 256 bytes. Threads: 256. Each loads 1. */
        {
            int tid = ty * TILE_M + tx;
            int pk_base = (blockIdx.x * TILE_M) / 4;

            if (tid < TILE_M_BW * 4) {
                int lm  = tid / 4;
                int lpk = tid % 4;
                int gm  = m_base + lm;
                int gpk = pk_base + lpk;
                sW[lm][lpk] = (gm < M && gpk < K4) ? W[gm * K4 + gpk] : 0;
            }
        }

        __syncthreads();

        /* ── Accumulate from this tile ── */
        if (k < K && s < S) {
            int lpk = (k / 4) - (blockIdx.x * TILE_M) / 4;  /* local packed byte index (0..3) */
            int end_m = TILE_M_BW;
            if (m_base + end_m > M) end_m = M - m_base;

            for (int lm = 0; lm < end_m; lm++) {
                uint8_t packed = sW[lm][lpk];
                uint8_t bits = (packed >> j_shift) & 0x3;
                int sign = (int)(bits & 1) - (int)(bits >> 1);
                acc += (float)sign * sdY[ty][lm];
            }
        }

        __syncthreads();
    }

    if (k < K && s < S) {
        dX[s * K + k] += acc;
    }
}

/*
 * Fused requantize kernel.
 *
 * Goes directly from latent float weights to packed ternary on GPU.
 * No host round-trip. No intermediate malloc.
 *
 * Each thread handles one packed byte (4 ternary weights).
 *
 * Input:  latent[rows * cols] float on device
 * Output: packed[rows * (cols/4)] uint8_t on device
 *         scale_out[1] float on device (absmean)
 */
static __global__ void kernel_fused_requantize(
    const float* __restrict__ latent,  /* [rows * cols] */
    uint8_t*     __restrict__ packed,  /* [rows * (cols/4)] */
    int rows, int cols,
    float threshold  /* STE_THRESHOLD = 0.33 */
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int packed_cols = cols / 4;
    int total_packed = rows * packed_cols;

    if (idx >= total_packed) return;

    int r = idx / packed_cols;
    int pc = idx % packed_cols;

    uint8_t byte = 0;

    #pragma unroll
    for (int j = 0; j < 4; j++) {
        int c = pc * 4 + j;
        float w = latent[r * cols + c];

        /* Ternary quantize: same as ternary_quantize() */
        uint8_t bits;
        if (w > threshold)       bits = 0x01;  /* +1 entity A */
        else if (w < -threshold) bits = 0x02;  /* -1 entity B */
        else                     bits = 0x00;  /* 0  substrate */

        byte |= (bits << (j * 2));
    }

    packed[idx] = byte;
}

/*
 * Fused absmean scale computation — only over non-zero quantized weights.
 *
 * The scale must match what the old two3_pack_weights computed:
 * scale = mean(|quantized|) = (count of non-zero weights) / total.
 * NOT mean(|latent|), which includes sub-threshold values and >1 values.
 *
 * scale_out[0] = sum of |quantized values| (always 0 or 1)
 * scale_out[1] = count of non-zero weights
 * Caller divides: scale = scale_out[0] / total = nonzero_count / total.
 */
static __global__ void kernel_compute_absmean(
    const float* __restrict__ latent,
    float* __restrict__ scale_out,  /* [2]: sum, count */
    int total,
    float threshold
) {
    extern __shared__ float sdata[];  /* [blockDim.x * 2] */
    float *s_sum = sdata;
    float *s_cnt = sdata + blockDim.x;
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x * 2 + threadIdx.x;

    float sum = 0.0f;
    float cnt = 0.0f;
    if (i < total) {
        float v = fabsf(latent[i]);
        if (v > threshold) { sum += 1.0f; cnt += 1.0f; }
    }
    if (i + blockDim.x < total) {
        float v = fabsf(latent[i + blockDim.x]);
        if (v > threshold) { sum += 1.0f; cnt += 1.0f; }
    }
    s_sum[tid] = sum;
    s_cnt[tid] = cnt;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_sum[tid] += s_sum[tid + s];
            s_cnt[tid] += s_cnt[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(&scale_out[0], s_sum[0]);
        atomicAdd(&scale_out[1], s_cnt[0]);
    }
}

/* NOTE: Persistent backward buffers already implemented as Two3BackwardCtx
 * in two3.h/two3.cu this session. Not duplicated here. */

/*
 * Compute L1 sum (sum of absolute values) for initial scale.
 * Each thread processes multiple elements, accumulates locally.
 */
static __global__ void kernel_compute_l1_sum(
    const float* __restrict__ data,  /* [rows, cols] */
    float* __restrict__ scratch,     /* [1] device: sum */
    int n
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    float local_sum = 0.0f;
    for (int i = tid; i < n; i += stride) {
        local_sum += fabsf(data[i]);
    }

    /* Block-level reduce */
    __shared__ float s_sum[256];
    s_sum[threadIdx.x] = local_sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            s_sum[threadIdx.x] += s_sum[threadIdx.x + s];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        atomicAdd(scratch, s_sum[0]);
    }
}

/*
 * Compute active-weight sum and count for scale refinement.
 * Active = |w| > threshold * scale.
 * Each thread processes multiple elements, accumulates locally.
 */
static __global__ void kernel_compute_active_scale(
    const float* __restrict__ data,  /* [rows, cols] */
    float* __restrict__ scratch,     /* [2] device: [0]=sum, [1]=count */
    int n,
    float ref_scale,
    float threshold
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    float local_sum = 0.0f;
    int local_cnt = 0;

    for (int i = tid; i < n; i += stride) {
        float abs_w = fabsf(data[i]);
        if (abs_w > threshold * ref_scale) {
            local_sum += abs_w;
            local_cnt++;
        }
    }

    /* Block-level reduce */
    __shared__ float s_sum[256];
    __shared__ int s_cnt[256];
    s_sum[threadIdx.x] = local_sum;
    s_cnt[threadIdx.x] = local_cnt;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            s_sum[threadIdx.x] += s_sum[threadIdx.x + s];
            s_cnt[threadIdx.x] += s_cnt[threadIdx.x + s];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        atomicAdd(scratch, s_sum[0]);
        atomicAdd(scratch + 1, (float)s_cnt[0]);
    }
}

/*
 * Hadamard transform kernel — applies FWT to each row.
 * Each thread handles one row. Uses iterative Cooley-Tukey pattern.
 * Normalized: output scaled by 1/sqrt(cols).
 */
static __global__ void kernel_hadamard_rows(
    float* __restrict__ data,  /* [rows, cols] row-major, transformed in-place */
    int rows, int cols
) {
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= rows) return;

    float* row = data + r * cols;

    /* Iterative fast Walsh-Hadamard transform */
    for (int len = 1; len < cols; len <<= 1) {
        for (int i = 0; i < cols; i += len << 1) {
            for (int j = 0; j < len; j++) {
                float u = row[i + j];
                float v = row[i + j + len];
                row[i + j]       = u + v;
                row[i + j + len] = u - v;
            }
        }
    }

    /* Normalize by 1/sqrt(cols) */
    float scale = 1.0f / sqrtf((float)cols);
    for (int i = 0; i < cols; i++)
        row[i] *= scale;
}

/*
 * Fused requantize on GPU — no host round-trip, no printf.
 *
 * Uses Two3BackwardCtx.d_W_latent as scratch for latent upload.
 * Writes directly to existing Two3Weights.packed buffer on device.
 * Uses Two3BackwardCtx.d_dY[0..1] as scratch for scale computation.
 *
 * Hadamard pre-quantization: applies fast Walsh-Hadamard transform
 * to each row before quantization. Reduces quantization error by
 * spreading outlier energy uniformly across dimensions.
 *
 * Lloyd-Max scale refinement: 3-pass iterative refinement where
 * scale = mean of only active weights (|w| > threshold * scale).
 * Prevents outliers from stretching the quantization grid.
 *
 * Replaces old trainable_requantize() which called two3_pack_weights()
 * (112 malloc/free + 112 printf per optimizer step).
 */
/* requantize_gpu: latent → Hadamard → quantize → pack.
 * If d_latent_device is non-NULL, uses it directly (GPU-resident path).
 * Otherwise uploads latent_host via H2D (legacy path). */
static void requantize_gpu(
    Two3BackwardCtx *ctx,
    const float *latent_host,    /* [rows, cols] latent float (NULL if GPU-resident) */
    Two3Weights *w,              /* existing weights — packed buffer reused */
    int rows, int cols,
    float threshold,             /* STE_THRESHOLD */
    float *d_latent_device       /* device pointer (NULL = use latent_host via H2D) */
) {
    int total = rows * cols;
    int packed_total = rows * (cols / 4);

    float *d_src;
    if (d_latent_device) {
        /* GPU-resident: copy from device latent to scratch (Hadamard modifies in-place) */
        cudaMemcpy(ctx->d_W_latent, d_latent_device, total * sizeof(float),
                   cudaMemcpyDeviceToDevice);
        d_src = ctx->d_W_latent;
    } else {
        /* Legacy: upload from host */
        cudaMemcpy(ctx->d_W_latent, latent_host, total * sizeof(float),
                   cudaMemcpyHostToDevice);
        d_src = ctx->d_W_latent;
    }

    /* Hadamard pre-quantization: transform each row on GPU.
     * Each thread handles one row. */
    {
        int hadamard_threads = 256;
        int hadamard_blocks = (rows + hadamard_threads - 1) / hadamard_threads;
        kernel_hadamard_rows<<<hadamard_blocks, hadamard_threads>>>(
            ctx->d_W_latent, rows, cols);
    }

    /* Compute absmean scale — only over values that survive quantization.
     * scale = nonzero_count / total (fraction of active weights).
     * Uses d_dY[0..1] as scratch: [0]=sum of |quantized|, [1]=count. */
    cudaMemset(ctx->d_dY, 0, 2 * sizeof(float));
    int threads = 256;
    int blocks = (total + threads * 2 - 1) / (threads * 2);
    kernel_compute_absmean<<<blocks, threads, threads * 2 * sizeof(float)>>>(
        ctx->d_W_latent, ctx->d_dY, total, threshold);

    float scale_buf[2];
    cudaMemcpy(scale_buf, ctx->d_dY, 2 * sizeof(float), cudaMemcpyDeviceToHost);
    float scale = scale_buf[0] / (float)total;  /* = nonzero_count / total */
    scale /= sqrtf((float)cols);  /* bake 1/sqrt(K) into weight scale */
    if (scale < 1e-10f) scale = 1e-10f;

    /* Fused quantize + pack on GPU — writes directly to w->packed */
    int pack_threads = 256;
    int pack_blocks = (packed_total + pack_threads - 1) / pack_threads;
    kernel_fused_requantize<<<pack_blocks, pack_threads>>>(
        ctx->d_W_latent, w->packed, rows, cols, threshold);

    cudaDeviceSynchronize();

    w->scale = scale;
    w->rows = rows;
    w->cols = cols;
}

#endif /* TWO3_TILED_H */
