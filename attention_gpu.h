/*
 * attention_gpu.h — GPU causal attention forward + backward
 *
 * Replaces causal_attention_cpu (model.h:433) and
 * causal_attention_backward_cpu (train.h:1630) with CUDA kernels.
 *
 * Forward:  All positions × all heads in one kernel launch.
 *           Each thread block handles one (position, head) pair.
 * Backward: Fused dQ/dK/dV computation, one block per (position, head).
 *
 * Profile target: 28ms forward + 37ms backward → ~5ms total on 2080 Super.
 *
 * Isaac & CC — April 2026
 */

#ifndef ATTENTION_GPU_H
#define ATTENTION_GPU_H

#include <cuda_runtime.h>
#include <math.h>

/* ═══════════════════════════════════════════════════════
 * Forward: causal attention on GPU
 *
 * One block per (position, head). Threads collaborate on
 * the dot products and reductions within a head.
 *
 * Grid:  (seq_len, n_heads)
 * Block: min(seq_len_rounded, 256) threads
 *
 * Each block computes:
 *   1. scores[t] = Q[pos,h] · K[t,kv_h] / sqrt(d) for t=0..pos
 *   2. softmax over scores
 *   3. out[h,d] = sum_t attn[t] * V[t,kv_h,d]
 * ═══════════════════════════════════════════════════════ */

__global__ void kernel_causal_attention_forward(
    const float *q_all,       /* [S × dim] all queries (after RoPE) */
    const float *k_store,     /* [S × kv_dim] all keys */
    const float *v_store,     /* [S × kv_dim] all values */
    float *out,               /* [S × dim] output */
    int seq_len, int n_heads, int n_kv_heads, int head_dim
) {
    int pos = blockIdx.x;
    int h   = blockIdx.y;
    int tid = threadIdx.x;
    int causal_len = pos + 1;

    int kv_dim = n_kv_heads * head_dim;
    int heads_per_kv = n_heads / n_kv_heads;
    int kv_h = h / heads_per_kv;
    float scale = 1.0f / sqrtf((float)head_dim);

    /* Shared memory: scores[causal_len] — dynamically sized */
    extern __shared__ float smem[];
    float *scores = smem;

    /* Step 1: compute scores — each thread handles a subset of t values */
    for (int t = tid; t < causal_len; t += blockDim.x) {
        float dot = 0.0f;
        const float *qi = q_all + pos * (n_heads * head_dim) + h * head_dim;
        const float *ki = k_store + t * kv_dim + kv_h * head_dim;
        for (int d = 0; d < head_dim; d++)
            dot += qi[d] * ki[d];
        scores[t] = dot * scale;
    }
    __syncthreads();

    /* Step 2: softmax — find max (parallel reduction) */
    /* Use a second region of shared memory for reduction */
    float *red = smem + causal_len;  /* [blockDim.x] */

    float local_max = -1e30f;
    for (int t = tid; t < causal_len; t += blockDim.x)
        if (scores[t] > local_max) local_max = scores[t];
    red[tid] = local_max;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s && red[tid + s] > red[tid])
            red[tid] = red[tid + s];
        __syncthreads();
    }
    float max_s = red[0];

    /* Exp + sum */
    float local_sum = 0.0f;
    for (int t = tid; t < causal_len; t += blockDim.x) {
        scores[t] = expf(scores[t] - max_s);
        local_sum += scores[t];
    }
    red[tid] = local_sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) red[tid] += red[tid + s];
        __syncthreads();
    }
    float sum_exp = red[0];

    /* Normalize */
    for (int t = tid; t < causal_len; t += blockDim.x)
        scores[t] /= sum_exp;
    __syncthreads();

    /* Step 3: weighted sum of V — each thread handles a subset of d values */
    float *out_h = out + pos * (n_heads * head_dim) + h * head_dim;
    for (int d = tid; d < head_dim; d += blockDim.x) {
        float val = 0.0f;
        for (int t = 0; t < causal_len; t++)
            val += scores[t] * v_store[t * kv_dim + kv_h * head_dim + d];
        out_h[d] = val;
    }
}

/* Launch wrapper */
static void causal_attention_forward_gpu(
    const float *d_q_all,     /* [S × dim] device */
    const float *d_k_store,   /* [S × kv_dim] device */
    const float *d_v_store,   /* [S × kv_dim] device */
    float *d_out,             /* [S × dim] device */
    int seq_len, int n_heads, int n_kv_heads, int head_dim
) {
    dim3 grid(seq_len, n_heads);
    /* Block size: enough threads for the longest sequence position */
    int block_size = 128;
    if (seq_len > 128) block_size = 256;

    /* Shared memory: scores[seq_len] + reduction[block_size] */
    size_t smem_bytes = (seq_len + block_size) * sizeof(float);

    kernel_causal_attention_forward<<<grid, block_size, smem_bytes>>>(
        d_q_all, d_k_store, d_v_store, d_out,
        seq_len, n_heads, n_kv_heads, head_dim);
}

/* ═══════════════════════════════════════════════════════
 * Backward: causal attention on GPU
 *
 * One block per (position, head). Computes dQ, dK, dV.
 *
 * Recomputes forward softmax (standard memory-efficient approach).
 * Accumulates dK and dV with atomicAdd (across positions).
 *
 * Grid:  (seq_len, n_heads)
 * Block: min(seq_len_rounded, 256)
 * ═══════════════════════════════════════════════════════ */

__global__ void kernel_causal_attention_backward(
    const float *d_out,       /* [S × dim] gradient from above */
    const float *q_all,       /* [S × dim] saved queries */
    const float *k_store,     /* [S × kv_dim] saved keys */
    const float *v_store,     /* [S × kv_dim] saved values */
    float *dq_all,            /* [S × dim] OUTPUT (zeroed before launch) */
    float *dk_store,          /* [S × kv_dim] ACCUMULATE (atomicAdd) */
    float *dv_store,          /* [S × kv_dim] ACCUMULATE (atomicAdd) */
    int seq_len, int n_heads, int n_kv_heads, int head_dim
) {
    int pos = blockIdx.x;
    int h   = blockIdx.y;
    int tid = threadIdx.x;
    int causal_len = pos + 1;

    int kv_dim = n_kv_heads * head_dim;
    int dim = n_heads * head_dim;
    int heads_per_kv = n_heads / n_kv_heads;
    int kv_h = h / heads_per_kv;
    float scale = 1.0f / sqrtf((float)head_dim);

    extern __shared__ float smem[];
    float *scores = smem;                    /* [causal_len] */
    float *red    = smem + causal_len;       /* [blockDim.x] */

    const float *qi = q_all + pos * dim + h * head_dim;

    /* ── Recompute forward: scores + softmax ── */
    for (int t = tid; t < causal_len; t += blockDim.x) {
        float dot = 0.0f;
        const float *ki = k_store + t * kv_dim + kv_h * head_dim;
        for (int d = 0; d < head_dim; d++)
            dot += qi[d] * ki[d];
        scores[t] = dot * scale;
    }
    __syncthreads();

    /* Max reduction */
    float local_max = -1e30f;
    for (int t = tid; t < causal_len; t += blockDim.x)
        if (scores[t] > local_max) local_max = scores[t];
    red[tid] = local_max;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s && red[tid + s] > red[tid]) red[tid] = red[tid + s];
        __syncthreads();
    }
    float max_s = red[0];

    /* Exp + sum */
    float local_sum = 0.0f;
    for (int t = tid; t < causal_len; t += blockDim.x) {
        scores[t] = expf(scores[t] - max_s);
        local_sum += scores[t];
    }
    red[tid] = local_sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) red[tid] += red[tid + s];
        __syncthreads();
    }
    float sum_exp = red[0];
    for (int t = tid; t < causal_len; t += blockDim.x)
        scores[t] /= sum_exp;
    __syncthreads();
    /* scores[t] = attn_weights[t] now */

    const float *d_out_h = d_out + pos * dim + h * head_dim;

    /* ── dV: dV[t] += attn[t] * d_out[h] ── */
    for (int t = tid; t < causal_len; t += blockDim.x) {
        float a = scores[t];
        for (int d = 0; d < head_dim; d++)
            atomicAdd(&dv_store[t * kv_dim + kv_h * head_dim + d],
                      a * d_out_h[d]);
    }

    /* ── d_attn[t] = d_out[h] · V[t] ── */
    /* Reuse 'red' area for d_attn — but we need causal_len floats.
     * Use scores area since we still need scores for softmax backward.
     * Store d_attn in a separate region. */
    float *d_attn = red;  /* Reuse — we're done with reduction. Size >= causal_len for seq<=256 */
    /* But blockDim.x might be < causal_len. Use scores for temp storage after softmax grad. */
    /* Simple approach: each thread computes its d_attn values serially */
    for (int t = tid; t < causal_len; t += blockDim.x) {
        float da = 0.0f;
        for (int d = 0; d < head_dim; d++)
            da += d_out_h[d] * v_store[t * kv_dim + kv_h * head_dim + d];
        d_attn[t] = da;  /* Store temporarily — safe if blockDim.x >= causal_len */
    }
    __syncthreads();

    /* ── Softmax backward: d_scores[t] = attn[t] * (d_attn[t] - dot_da) ── */
    /* dot_da = sum_t attn[t] * d_attn[t] */
    float local_dot = 0.0f;
    for (int t = tid; t < causal_len; t += blockDim.x)
        local_dot += scores[t] * d_attn[t];

    /* Reduce dot_da — reuse part of smem after d_attn */
    /* We need another reduction buffer. Use scores[causal_len..] if available,
     * or do it in registers with warp shuffle. Simple: use the same red buffer
     * but offset past d_attn. For small seq_len this is fine. */
    __shared__ float dot_da_shared;
    if (tid == 0) dot_da_shared = 0.0f;
    __syncthreads();
    atomicAdd(&dot_da_shared, local_dot);
    __syncthreads();
    float dot_da = dot_da_shared;

    /* d_scores[t] = attn[t] * (d_attn[t] - dot_da) */
    /* ── dQ and dK from d_scores ── */
    float *dq_h = dq_all + pos * dim + h * head_dim;
    for (int t = tid; t < causal_len; t += blockDim.x) {
        float ds = scores[t] * (d_attn[t] - dot_da) * scale;
        const float *ki = k_store + t * kv_dim + kv_h * head_dim;
        for (int d = 0; d < head_dim; d++) {
            atomicAdd(&dq_h[d], ds * ki[d]);
            atomicAdd(&dk_store[t * kv_dim + kv_h * head_dim + d],
                      ds * qi[d]);
        }
    }
}

/* Launch wrapper */
static void causal_attention_backward_gpu(
    const float *d_d_out,     /* [S × dim] device */
    const float *d_q_all,     /* [S × dim] device */
    const float *d_k_store,   /* [S × kv_dim] device */
    const float *d_v_store,   /* [S × kv_dim] device */
    float *d_dq_all,          /* [S × dim] device — zeroed before call */
    float *d_dk_store,        /* [S × kv_dim] device — zeroed before call */
    float *d_dv_store,        /* [S × kv_dim] device — zeroed before call */
    int seq_len, int n_heads, int n_kv_heads, int head_dim
) {
    dim3 grid(seq_len, n_heads);
    int block_size = 128;
    if (seq_len > 128) block_size = 256;

    /* Shared memory: scores[seq_len] + d_attn_or_red[max(block_size, seq_len)] */
    int smem_floats = seq_len + (block_size > seq_len ? block_size : seq_len);
    size_t smem_bytes = smem_floats * sizeof(float);

    /* Zero outputs */
    int dim = n_heads * head_dim;
    int kv_dim = n_kv_heads * head_dim;
    cudaMemset(d_dq_all, 0, seq_len * dim * sizeof(float));
    cudaMemset(d_dk_store, 0, seq_len * kv_dim * sizeof(float));
    cudaMemset(d_dv_store, 0, seq_len * kv_dim * sizeof(float));

    kernel_causal_attention_backward<<<grid, block_size, smem_bytes>>>(
        d_d_out, d_q_all, d_k_store, d_v_store,
        d_dq_all, d_dk_store, d_dv_store,
        seq_len, n_heads, n_kv_heads, head_dim);
}

#endif /* ATTENTION_GPU_H */
