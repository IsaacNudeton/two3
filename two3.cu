/*
 * two3.cu — {2,3} Computing Kernel Implementation
 * Isaac Oravec & Claude | XYZT Computing Project | 2026
 *
 * Every operation in the forward pass is integer.
 * The only float ops are quantization (before) and dequantization (after).
 * The matmul itself: addition, subtraction, structure. That's it.
 */

#include "two3.h"
#include "two3_tiled.h"
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
 * Kernel: two3_backward_dx_kernel
 *
 * Backward through ternary matmul for input gradient.
 * Forward was: Y[s][m] = sum_k W[m][k] * X[s][k]
 * Backward:    dX[s][k] = sum_m dY[s][m] * W[m][k]
 *
 * This is the transposed ternary matmul. Each thread computes
 * one dX element by accumulating over output dimension M.
 * Uses same branchless decode as forward.
 */
__global__ void two3_backward_dx_kernel(
    const uint8_t* __restrict__ W,    /* [M, K/4] packed ternary */
    const float*   __restrict__ dY,   /* [S, M]   gradient from above */
    float*         __restrict__ dX,   /* [S, K]   output gradient */
    int S, int M, int K)
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;  /* input feature */
    int s = blockIdx.y * blockDim.y + threadIdx.y;  /* sequence position */

    if (k >= K || s >= S) return;

    int K4 = K / 4;
    int pk = k / 4;     /* which packed byte */
    int j  = k % 4;     /* which 2-bit slot within byte */

    float acc = 0.0f;

    for (int m = 0; m < M; m++) {
        uint8_t packed = W[m * K4 + pk];
        uint8_t bits = (packed >> (j * 2)) & 0x3;
        int sign = (int)(bits & 1) - (int)(bits >> 1);
        acc += (float)sign * dY[s * M + m];
    }

    dX[s * K + k] += acc;  /* accumulate */
}

/*
 * Kernel: two3_backward_dw_kernel
 *
 * Gradient for latent weights with STE.
 * dW_latent[m][k] += dY[m] * X[k]   (outer product)
 * STE: zero gradient if |w_latent| > clip threshold
 *
 * Each thread handles one (m, k) element.
 */
__global__ void two3_backward_dw_kernel(
    const float*   __restrict__ dY,       /* [S, M] gradient */
    const float*   __restrict__ X,        /* [S, K] input activations */
    const float*   __restrict__ W_latent, /* [M, K] latent float weights */
    float*         __restrict__ dW,       /* [M, K] output gradient (accumulate) */
    int S, int M, int K, float ste_clip)
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    int m = blockIdx.y * blockDim.y + threadIdx.y;

    if (k >= K || m >= M) return;

    float w = W_latent[m * K + k];
    if (fabsf(w) > ste_clip) return;  /* STE: zero grad outside clip range */

    float grad = 0.0f;
    for (int s = 0; s < S; s++) {
        grad += dY[s * M + m] * X[s * K + k];
    }

    dW[m * K + k] += grad;  /* accumulate across batch */
}

/*
 * Kernel: attention_backward_kernel
 *
 * Batched causal attention backward for ALL positions.
 * One thread per (head, position) pair.
 *
 * Recomputes attention scores + softmax, then backprops:
 *   dV[t] += attn[t] * d_out[h]         (atomicAdd — multiple positions write same t)
 *   d_attn[t] = d_out[h] · V[t]
 *   d_scores via softmax backward
 *   dQ[pos][h] += d_scores * K[t]        (no atomic — only this thread writes pos,h)
 *   dK[t] += d_scores * Q[pos][h]        (atomicAdd — multiple positions write same t)
 *
 * Uses global scratch buffer for scores/attn weights per (head, position).
 */
__global__ void attention_backward_kernel(
    const float* __restrict__ Q,        /* [seq_len, D] where D = n_heads * head_dim */
    const float* __restrict__ K,        /* [seq_len, KV] where KV = n_kv_heads * head_dim */
    const float* __restrict__ V,        /* [seq_len, KV] */
    const float* __restrict__ d_out,    /* [seq_len, D] */
    float*       __restrict__ dQ,       /* [seq_len, D] output (pre-zeroed) */
    float*       __restrict__ dK,       /* [seq_len, KV] output (pre-zeroed) */
    float*       __restrict__ dV,       /* [seq_len, KV] output (pre-zeroed) */
    float*       __restrict__ scratch,  /* [n_heads, seq_len, seq_len] */
    int seq_len, int n_heads, int n_kv_heads, int head_dim
) {
    int h   = blockIdx.x;
    int pos = blockIdx.y * blockDim.x + threadIdx.x;
    if (pos >= seq_len || h >= n_heads) return;

    int D  = n_heads * head_dim;
    int KV = n_kv_heads * head_dim;
    int heads_per_kv = n_heads / n_kv_heads;
    int kv_h = h / heads_per_kv;
    float scale = rsqrtf((float)head_dim);
    int causal_len = pos + 1;

    /* Scratch pointer for this (head, position) */
    float* my_buf = scratch + ((size_t)h * seq_len + pos) * seq_len;

    /* ── Step 1: Recompute attention scores + softmax ── */
    float max_s = -1e30f;
    for (int t = 0; t < causal_len; t++) {
        float dot = 0;
        for (int d = 0; d < head_dim; d++)
            dot += Q[pos * D + h * head_dim + d] * K[t * KV + kv_h * head_dim + d];
        my_buf[t] = dot * scale;
        if (my_buf[t] > max_s) max_s = my_buf[t];
    }

    float sum_exp = 0;
    for (int t = 0; t < causal_len; t++) {
        my_buf[t] = expf(my_buf[t] - max_s);
        sum_exp += my_buf[t];
    }
    float inv_sum = 1.0f / sum_exp;
    for (int t = 0; t < causal_len; t++)
        my_buf[t] *= inv_sum;

    /* attn_weights are now in my_buf[0..causal_len-1] */
    const float* d_out_h = d_out + pos * D + h * head_dim;

    /* ── Step 2: dV[t] += attn[t] * d_out_h ── */
    for (int t = 0; t < causal_len; t++) {
        float aw = my_buf[t];
        for (int d = 0; d < head_dim; d++)
            atomicAdd(&dV[t * KV + kv_h * head_dim + d], aw * d_out_h[d]);
    }

    /* ── Steps 3-4: softmax backward → dQ, dK ── */
    /* attn weights are still in my_buf. Compute d_attn and dot_da in one pass,
     * then d_scores and dQ/dK in a second pass. Two passes, no recompute. */
    float dot_da = 0;
    for (int t = 0; t < causal_len; t++) {
        float da = 0;
        for (int d = 0; d < head_dim; d++)
            da += d_out_h[d] * V[t * KV + kv_h * head_dim + d];
        dot_da += my_buf[t] * da;
    }

    /* d_scores[t] = attn[t] * (d_attn[t] - dot_da), then dQ/dK */
    for (int t = 0; t < causal_len; t++) {
        float da = 0;
        for (int d = 0; d < head_dim; d++)
            da += d_out_h[d] * V[t * KV + kv_h * head_dim + d];
        float d_score = my_buf[t] * (da - dot_da) * scale;

        for (int d = 0; d < head_dim; d++) {
            dQ[pos * D + h * head_dim + d] += d_score * K[t * KV + kv_h * head_dim + d];
            atomicAdd(&dK[t * KV + kv_h * head_dim + d],
                      d_score * Q[pos * D + h * head_dim + d]);
        }
    }
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

    /* Launch the {2,3} tiled kernel — shared memory reuse */
    dim3 block(TILE_M, TILE_N);  /* 16×16 = 256 threads */
    dim3 grid((M + TILE_M - 1) / TILE_M,
              (S + TILE_N - 1) / TILE_N);

    two3_matmul_tiled<<<grid, block>>>(
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

/* ---- Backward pass (GPU) ---- */

/*
 * Backward through ternary projection — GPU accelerated.
 *
 * Computes both:
 *   dX[k] += sum_m dY[m] * W_ternary[m][k]     (transposed ternary matmul)
 *   dW_latent[m][k] += dY[m] * X[k] * STE_mask  (float outer product)
 *
 * All on GPU. Host only provides/receives float arrays.
 */
void two3_backward(
    const Two3Weights* W,     /* packed ternary weights (on device) */
    const float* dY_host,     /* [1, M] gradient from above */
    const float* X_host,      /* [1, K] saved input */
    const float* W_latent,    /* [M, K] latent float weights (host) */
    float* dX_host,           /* [K] gradient to pass back (ACCUMULATE, host) */
    float* dW_host,           /* [M, K] gradient for latent (ACCUMULATE, host) */
    int M, int K,
    float ste_clip
) {
    int S = 1;  /* single token for now */

    /* Allocate device buffers */
    float *d_dY, *d_X, *d_dX, *d_W_latent, *d_dW;
    CUDA_CHECK(cudaMalloc(&d_dY, S * M * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_X, S * K * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dX, S * K * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_W_latent, M * K * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dW, M * K * sizeof(float)));

    /* Copy inputs to device */
    CUDA_CHECK(cudaMemcpy(d_dY, dY_host, S * M * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_X, X_host, S * K * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_dX, 0, S * K * sizeof(float)));

    /* Copy latent weights and existing gradients to device */
    CUDA_CHECK(cudaMemcpy(d_W_latent, W_latent, M * K * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_dW, dW_host, M * K * sizeof(float), cudaMemcpyHostToDevice));

    /* Kernel 1: dX = dY @ W_ternary (transposed ternary matmul) */
    {
        dim3 block(TWO3_BLOCK_X, TWO3_BLOCK_Y);
        dim3 grid((K + block.x - 1) / block.x,
                  (S + block.y - 1) / block.y);
        two3_backward_dx_kernel<<<grid, block>>>(
            W->packed, d_dY, d_dX, S, M, K);
        CUDA_CHECK(cudaGetLastError());
    }

    /* Kernel 2: dW = dY^T @ X with STE (float outer product) */
    {
        dim3 block(TWO3_BLOCK_X, TWO3_BLOCK_Y);
        dim3 grid((K + block.x - 1) / block.x,
                  (M + block.y - 1) / block.y);
        two3_backward_dw_kernel<<<grid, block>>>(
            d_dY, d_X, d_W_latent, d_dW,
            S, M, K, ste_clip);
        CUDA_CHECK(cudaGetLastError());
    }

    CUDA_CHECK(cudaDeviceSynchronize());

    /* Copy results back — ACCUMULATE into host buffers */
    float *dX_tmp = (float*)malloc(K * sizeof(float));
    CUDA_CHECK(cudaMemcpy(dX_tmp, d_dX, K * sizeof(float), cudaMemcpyDeviceToHost));
    for (int i = 0; i < K; i++) dX_host[i] += dX_tmp[i];
    free(dX_tmp);

    CUDA_CHECK(cudaMemcpy(dW_host, d_dW, M * K * sizeof(float), cudaMemcpyDeviceToHost));

    /* Free device buffers */
    cudaFree(d_dY); cudaFree(d_X); cudaFree(d_dX);
    cudaFree(d_W_latent); cudaFree(d_dW);
}

/* ================================================================
 * Pre-allocated backward context — eliminates per-call malloc
 * ================================================================ */

Two3BackwardCtx two3_backward_ctx_init(int max_M, int max_K,
                                        int max_seq, int D, int KV, int n_heads) {
    Two3BackwardCtx ctx;
    memset(&ctx, 0, sizeof(ctx));

    /* Ternary projection buffers */
    ctx.max_M = max_M;
    ctx.max_K = max_K;
    CUDA_CHECK(cudaMalloc(&ctx.d_dY, max_M * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&ctx.d_X, max_K * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&ctx.d_dX, max_K * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&ctx.d_W_latent, (size_t)max_M * max_K * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&ctx.d_dW, (size_t)max_M * max_K * sizeof(float)));
    ctx.h_dX_tmp = (float*)malloc(max_K * sizeof(float));

    /* Attention backward buffers */
    if (max_seq > 0 && D > 0) {
        /* Safety: scratch = n_heads * max_seq^2. Warn if > 256MB. */
        size_t scratch_bytes = (size_t)n_heads * max_seq * max_seq * sizeof(float);
        if (scratch_bytes > 256 * 1024 * 1024) {
            fprintf(stderr, "[two3] WARNING: attention scratch = %zuMB "
                    "(n_heads=%d, max_seq=%d). Consider reducing max_seq.\n",
                    scratch_bytes / (1024*1024), n_heads, max_seq);
        }
        ctx.attn_max_seq = max_seq;
        ctx.attn_D = D;
        ctx.attn_KV = KV;
        ctx.attn_n_heads = n_heads;
        CUDA_CHECK(cudaMalloc(&ctx.d_attn_Q,       (size_t)max_seq * D  * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&ctx.d_attn_K,       (size_t)max_seq * KV * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&ctx.d_attn_V,       (size_t)max_seq * KV * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&ctx.d_attn_dout,    (size_t)max_seq * D  * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&ctx.d_attn_dQ,      (size_t)max_seq * D  * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&ctx.d_attn_dK,      (size_t)max_seq * KV * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&ctx.d_attn_dV,      (size_t)max_seq * KV * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&ctx.d_attn_scratch,  (size_t)n_heads * max_seq * max_seq * sizeof(float)));
        ctx.h_attn_dK_tmp = (float*)malloc(max_seq * KV * sizeof(float));
        ctx.h_attn_dV_tmp = (float*)malloc(max_seq * KV * sizeof(float));
    }

    return ctx;
}

void two3_backward_ctx_free(Two3BackwardCtx* ctx) {
    if (ctx->d_dY)      cudaFree(ctx->d_dY);
    if (ctx->d_X)       cudaFree(ctx->d_X);
    if (ctx->d_dX)      cudaFree(ctx->d_dX);
    if (ctx->d_W_latent) cudaFree(ctx->d_W_latent);
    if (ctx->d_dW)      cudaFree(ctx->d_dW);
    if (ctx->h_dX_tmp)  free(ctx->h_dX_tmp);
    /* Attention buffers */
    if (ctx->d_attn_Q)       cudaFree(ctx->d_attn_Q);
    if (ctx->d_attn_K)       cudaFree(ctx->d_attn_K);
    if (ctx->d_attn_V)       cudaFree(ctx->d_attn_V);
    if (ctx->d_attn_dout)    cudaFree(ctx->d_attn_dout);
    if (ctx->d_attn_dQ)      cudaFree(ctx->d_attn_dQ);
    if (ctx->d_attn_dK)      cudaFree(ctx->d_attn_dK);
    if (ctx->d_attn_dV)      cudaFree(ctx->d_attn_dV);
    if (ctx->d_attn_scratch) cudaFree(ctx->d_attn_scratch);
    if (ctx->h_attn_dK_tmp)  free(ctx->h_attn_dK_tmp);
    if (ctx->h_attn_dV_tmp)  free(ctx->h_attn_dV_tmp);
    memset(ctx, 0, sizeof(*ctx));
}

void two3_backward_fast(
    Two3BackwardCtx* ctx,
    const Two3Weights* W,
    const float* dY_host,
    const float* X_host,
    const float* W_latent,
    float* dX_host,
    float* dW_host,
    int M, int K,
    float ste_clip
) {
    int S = 1;

    /* Copy inputs to pre-allocated device buffers */
    CUDA_CHECK(cudaMemcpy(ctx->d_dY, dY_host, S * M * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(ctx->d_X, X_host, S * K * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(ctx->d_dX, 0, S * K * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(ctx->d_W_latent, W_latent, M * K * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(ctx->d_dW, dW_host, M * K * sizeof(float), cudaMemcpyHostToDevice));

    /* Kernel 1: dX = dY @ W_ternary (tiled transposed ternary matmul) */
    {
        dim3 block(TILE_M, TILE_N);
        dim3 grid((K + TILE_M - 1) / TILE_M,
                  (S + TILE_N - 1) / TILE_N);
        two3_backward_dx_tiled<<<grid, block>>>(
            W->packed, ctx->d_dY, ctx->d_dX, S, M, K);
        CUDA_CHECK(cudaGetLastError());
    }

    /* Kernel 2: dW = dY^T @ X with STE (float outer product) */
    {
        dim3 block(TWO3_BLOCK_X, TWO3_BLOCK_Y);
        dim3 grid((K + block.x - 1) / block.x,
                  (M + block.y - 1) / block.y);
        two3_backward_dw_kernel<<<grid, block>>>(
            ctx->d_dY, ctx->d_X, ctx->d_W_latent, ctx->d_dW,
            S, M, K, ste_clip);
        CUDA_CHECK(cudaGetLastError());
    }

    CUDA_CHECK(cudaDeviceSynchronize());

    /* Copy results back — ACCUMULATE dX into host buffer */
    CUDA_CHECK(cudaMemcpy(ctx->h_dX_tmp, ctx->d_dX, K * sizeof(float), cudaMemcpyDeviceToHost));
    for (int i = 0; i < K; i++) dX_host[i] += ctx->h_dX_tmp[i];

    CUDA_CHECK(cudaMemcpy(dW_host, ctx->d_dW, M * K * sizeof(float), cudaMemcpyDeviceToHost));
}

/* ================================================================
 * Batched causal attention backward on GPU
 * ================================================================ */

void two3_attention_backward_fast(
    Two3BackwardCtx* ctx,
    const float* Q_host, const float* K_host, const float* V_host,
    const float* d_out_host,
    float* dQ_host, float* dK_host, float* dV_host,
    int seq_len, int n_heads, int n_kv_heads, int head_dim
) {
    int D  = n_heads * head_dim;
    int KV = n_kv_heads * head_dim;

    /* Copy inputs to pre-allocated device buffers */
    CUDA_CHECK(cudaMemcpy(ctx->d_attn_Q,    Q_host,     seq_len * D  * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(ctx->d_attn_K,    K_host,     seq_len * KV * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(ctx->d_attn_V,    V_host,     seq_len * KV * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(ctx->d_attn_dout, d_out_host, seq_len * D  * sizeof(float), cudaMemcpyHostToDevice));

    /* Zero output buffers */
    CUDA_CHECK(cudaMemset(ctx->d_attn_dQ, 0, seq_len * D  * sizeof(float)));
    CUDA_CHECK(cudaMemset(ctx->d_attn_dK, 0, seq_len * KV * sizeof(float)));
    CUDA_CHECK(cudaMemset(ctx->d_attn_dV, 0, seq_len * KV * sizeof(float)));

    /* Launch kernel */
    int block_size = 128;
    dim3 grid(n_heads, (seq_len + block_size - 1) / block_size);
    attention_backward_kernel<<<grid, block_size>>>(
        ctx->d_attn_Q, ctx->d_attn_K, ctx->d_attn_V, ctx->d_attn_dout,
        ctx->d_attn_dQ, ctx->d_attn_dK, ctx->d_attn_dV, ctx->d_attn_scratch,
        seq_len, n_heads, n_kv_heads, head_dim);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    /* Copy dQ back (output, not accumulated) */
    CUDA_CHECK(cudaMemcpy(dQ_host, ctx->d_attn_dQ, seq_len * D * sizeof(float), cudaMemcpyDeviceToHost));

    /* Copy dK, dV back and ACCUMULATE into host buffers */
    CUDA_CHECK(cudaMemcpy(ctx->h_attn_dK_tmp, ctx->d_attn_dK, seq_len * KV * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(ctx->h_attn_dV_tmp, ctx->d_attn_dV, seq_len * KV * sizeof(float), cudaMemcpyDeviceToHost));
    for (int i = 0; i < seq_len * KV; i++) {
        dK_host[i] += ctx->h_attn_dK_tmp[i];
        dV_host[i] += ctx->h_attn_dV_tmp[i];
    }
}

/* Original attention backward (allocates per call — kept for compat). */
void two3_attention_backward(
    const float* Q_host,     /* [seq_len, D] where D = n_heads * head_dim */
    const float* K_host,     /* [seq_len, KV] where KV = n_kv_heads * head_dim */
    const float* V_host,     /* [seq_len, KV] */
    const float* d_out_host, /* [seq_len, D] */
    float* dQ_host,          /* [seq_len, D] OUTPUT (zeroed by this function) */
    float* dK_host,          /* [seq_len, KV] ACCUMULATE */
    float* dV_host,          /* [seq_len, KV] ACCUMULATE */
    int seq_len, int n_heads, int n_kv_heads, int head_dim
) {
    int D  = n_heads * head_dim;
    int KV = n_kv_heads * head_dim;

    /* Allocate device buffers */
    float *d_Q, *d_K, *d_V, *d_out, *d_dQ, *d_dK, *d_dV, *d_scratch;
    CUDA_CHECK(cudaMalloc(&d_Q,   (size_t)seq_len * D  * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_K,   (size_t)seq_len * KV * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_V,   (size_t)seq_len * KV * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out, (size_t)seq_len * D  * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dQ,  (size_t)seq_len * D  * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dK,  (size_t)seq_len * KV * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dV,  (size_t)seq_len * KV * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_scratch, (size_t)n_heads * seq_len * seq_len * sizeof(float)));

    /* Copy inputs to device */
    CUDA_CHECK(cudaMemcpy(d_Q,   Q_host,     seq_len * D  * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_K,   K_host,     seq_len * KV * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_V,   V_host,     seq_len * KV * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_out, d_out_host, seq_len * D  * sizeof(float), cudaMemcpyHostToDevice));

    /* Zero output buffers on device */
    CUDA_CHECK(cudaMemset(d_dQ, 0, seq_len * D  * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_dK, 0, seq_len * KV * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_dV, 0, seq_len * KV * sizeof(float)));

    /* Launch: one block per head, threads handle positions */
    int block_size = 128;
    dim3 grid(n_heads, (seq_len + block_size - 1) / block_size);
    attention_backward_kernel<<<grid, block_size>>>(
        d_Q, d_K, d_V, d_out,
        d_dQ, d_dK, d_dV, d_scratch,
        seq_len, n_heads, n_kv_heads, head_dim);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    /* Copy dQ back (replace — output is zeroed on host by caller's original code) */
    CUDA_CHECK(cudaMemcpy(dQ_host, d_dQ, seq_len * D * sizeof(float), cudaMemcpyDeviceToHost));

    /* Copy dK, dV back and ACCUMULATE into host buffers */
    float *dK_tmp = (float*)malloc(seq_len * KV * sizeof(float));
    float *dV_tmp = (float*)malloc(seq_len * KV * sizeof(float));
    CUDA_CHECK(cudaMemcpy(dK_tmp, d_dK, seq_len * KV * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(dV_tmp, d_dV, seq_len * KV * sizeof(float), cudaMemcpyDeviceToHost));
    for (int i = 0; i < seq_len * KV; i++) {
        dK_host[i] += dK_tmp[i];
        dV_host[i] += dV_tmp[i];
    }
    free(dK_tmp);
    free(dV_tmp);

    /* Free device buffers */
    cudaFree(d_Q); cudaFree(d_K); cudaFree(d_V); cudaFree(d_out);
    cudaFree(d_dQ); cudaFree(d_dK); cudaFree(d_dV); cudaFree(d_scratch);
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
