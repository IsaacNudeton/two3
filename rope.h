/*
 * rope.h — Rotary Position Embedding for {2,3} Architecture
 *
 * RoPE rotates query/key vectors by position-dependent angles.
 * This is the ONE place float arithmetic is unavoidable —
 * cos/sin are irrational. But it's O(dim), not O(dim²).
 * The heavy matmul stays integer. RoPE is a thin float layer.
 *
 * Precompute rotation tables once. Apply per token.
 *
 * Isaac & CC — March 2026
 */

#ifndef ROPE_H
#define ROPE_H

#include <math.h>
#include <stdlib.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* ═══════════════════════════════════════════════════════
 * RoPE frequency table — precomputed per (position, dim)
 *
 * freq[d] = 1.0 / (theta ^ (2d / dim))
 * For position p: angle = p * freq[d]
 * Apply: q'[2d]   = q[2d]*cos(a) - q[2d+1]*sin(a)
 *        q'[2d+1] = q[2d]*sin(a) + q[2d+1]*cos(a)
 * ═══════════════════════════════════════════════════════ */

typedef struct {
    float *cos_table;   /* [max_seq × head_dim/2] precomputed */
    float *sin_table;   /* [max_seq × head_dim/2] precomputed */
    int    head_dim;
    int    max_seq;
    float  theta;
} RoPETable;

static void rope_init(RoPETable *r, int head_dim, int max_seq, float theta) {
    r->head_dim = head_dim;
    r->max_seq = max_seq;
    r->theta = theta;

    int half = head_dim / 2;
    r->cos_table = (float*)malloc(max_seq * half * sizeof(float));
    r->sin_table = (float*)malloc(max_seq * half * sizeof(float));

    for (int pos = 0; pos < max_seq; pos++) {
        for (int d = 0; d < half; d++) {
            float freq = 1.0f / powf(theta, (float)(2 * d) / head_dim);
            float angle = (float)pos * freq;
            r->cos_table[pos * half + d] = cosf(angle);
            r->sin_table[pos * half + d] = sinf(angle);
        }
    }
}

static void rope_free(RoPETable *r) {
    free(r->cos_table); free(r->sin_table);
    r->cos_table = r->sin_table = NULL;
}

/* ═══════════════════════════════════════════════════════
 * CUDA kernel: apply RoPE to Q and K vectors
 *
 * One block per head. Threads process dimension pairs.
 * Works on float vectors (after dequant from int32 accumulators).
 * ═══════════════════════════════════════════════════════ */

#ifdef __CUDACC__

__global__ void kernel_rope_apply(
    float       *q,          /* [n_heads × head_dim] — modified in place */
    float       *k,          /* [n_kv_heads × head_dim] — modified in place */
    const float *cos_tab,    /* [head_dim/2] for this position */
    const float *sin_tab,    /* [head_dim/2] for this position */
    int n_heads, int n_kv_heads, int head_dim
) {
    int h = blockIdx.x;   /* head index */
    int d = threadIdx.x;  /* dimension pair index */
    int half = head_dim / 2;
    if (d >= half) return;

    float c = cos_tab[d];
    float s = sin_tab[d];

    /* Rotate Q heads */
    if (h < n_heads) {
        int i = h * head_dim + 2 * d;
        float q0 = q[i], q1 = q[i + 1];
        q[i]     = q0 * c - q1 * s;
        q[i + 1] = q0 * s + q1 * c;
    }

    /* Rotate K heads (fewer for GQA) */
    if (h < n_kv_heads) {
        int i = h * head_dim + 2 * d;
        float k0 = k[i], k1 = k[i + 1];
        k[i]     = k0 * c - k1 * s;
        k[i + 1] = k0 * s + k1 * c;
    }
}

#endif /* __CUDACC__ */

/* ═══════════════════════════════════════════════════════
 * CPU reference
 * ═══════════════════════════════════════════════════════ */

static void rope_apply_cpu(
    float *q, float *k,
    const RoPETable *r, int pos,
    int n_heads, int n_kv_heads
) {
    int half = r->head_dim / 2;
    const float *ct = r->cos_table + pos * half;
    const float *st = r->sin_table + pos * half;

    for (int h = 0; h < n_heads; h++) {
        for (int d = 0; d < half; d++) {
            int i = h * r->head_dim + 2 * d;
            float q0 = q[i], q1 = q[i + 1];
            q[i]     = q0 * ct[d] - q1 * st[d];
            q[i + 1] = q0 * st[d] + q1 * ct[d];
        }
    }
    for (int h = 0; h < n_kv_heads; h++) {
        for (int d = 0; d < half; d++) {
            int i = h * r->head_dim + 2 * d;
            float k0 = k[i], k1 = k[i + 1];
            k[i]     = k0 * ct[d] - k1 * st[d];
            k[i + 1] = k0 * st[d] + k1 * ct[d];
        }
    }
}

/* Inverse RoPE: undo rotation on gradients.
 * Forward: [x0, x1] → [x0*c - x1*s, x0*s + x1*c]
 * Inverse: [g0, g1] → [g0*c + g1*s, -g0*s + g1*c]  (negate sin) */
static void rope_unapply_cpu(
    float *dq, float *dk,
    const RoPETable *r, int pos,
    int n_heads, int n_kv_heads
) {
    int half = r->head_dim / 2;
    const float *ct = r->cos_table + pos * half;
    const float *st = r->sin_table + pos * half;

    for (int h = 0; h < n_heads; h++) {
        for (int d = 0; d < half; d++) {
            int i = h * r->head_dim + 2 * d;
            float g0 = dq[i], g1 = dq[i + 1];
            dq[i]     =  g0 * ct[d] + g1 * st[d];
            dq[i + 1] = -g0 * st[d] + g1 * ct[d];
        }
    }
    for (int h = 0; h < n_kv_heads; h++) {
        for (int d = 0; d < half; d++) {
            int i = h * r->head_dim + 2 * d;
            float g0 = dk[i], g1 = dk[i + 1];
            dk[i]     =  g0 * ct[d] + g1 * st[d];
            dk[i + 1] = -g0 * st[d] + g1 * ct[d];
        }
    }
}

/* ═══════════════════════════════════════════════════════
 * GPU batched RoPE: all positions in one launch
 *
 * Grid:  (seq_len, max(n_heads, n_kv_heads))
 * Block: head_dim/2 threads
 *
 * Requires device-side cos/sin tables (uploaded once at init).
 * ═══════════════════════════════════════════════════════ */

#ifdef __CUDACC__

/* Device-side RoPE table pointers (set once by rope_upload_tables) */
static float *d_rope_cos = NULL;
static float *d_rope_sin = NULL;

static void rope_upload_tables(const RoPETable *r) {
    int n = r->max_seq * (r->head_dim / 2);
    if (d_rope_cos) cudaFree(d_rope_cos);
    if (d_rope_sin) cudaFree(d_rope_sin);
    cudaMalloc(&d_rope_cos, n * sizeof(float));
    cudaMalloc(&d_rope_sin, n * sizeof(float));
    cudaMemcpy(d_rope_cos, r->cos_table, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_rope_sin, r->sin_table, n * sizeof(float), cudaMemcpyHostToDevice);
}

/* Batched forward RoPE: apply to all positions at once.
 * q_all: [seq_len × n_heads × head_dim], k_store: [seq_len × n_kv_heads × head_dim]
 * Both modified in-place on device. */
__global__ void kernel_rope_apply_batched(
    float *q_all, float *k_store,
    const float *cos_tab, const float *sin_tab,
    int seq_len, int n_heads, int n_kv_heads, int head_dim
) {
    int pos = blockIdx.x;
    int h   = blockIdx.y;
    int d   = threadIdx.x;
    int half = head_dim / 2;
    if (d >= half) return;

    float c = cos_tab[pos * half + d];
    float s = sin_tab[pos * half + d];

    int dim = n_heads * head_dim;
    int kv_dim = n_kv_heads * head_dim;

    if (h < n_heads) {
        int i = pos * dim + h * head_dim + 2 * d;
        float q0 = q_all[i], q1 = q_all[i + 1];
        q_all[i]     = q0 * c - q1 * s;
        q_all[i + 1] = q0 * s + q1 * c;
    }
    if (h < n_kv_heads) {
        int i = pos * kv_dim + h * head_dim + 2 * d;
        float k0 = k_store[i], k1 = k_store[i + 1];
        k_store[i]     = k0 * c - k1 * s;
        k_store[i + 1] = k0 * s + k1 * c;
    }
}

/* Batched inverse RoPE: unapply on gradients. Negate sin. */
__global__ void kernel_rope_unapply_batched(
    float *dq_all, float *dk_store,
    const float *cos_tab, const float *sin_tab,
    int seq_len, int n_heads, int n_kv_heads, int head_dim
) {
    int pos = blockIdx.x;
    int h   = blockIdx.y;
    int d   = threadIdx.x;
    int half = head_dim / 2;
    if (d >= half) return;

    float c = cos_tab[pos * half + d];
    float s = sin_tab[pos * half + d];

    int dim = n_heads * head_dim;
    int kv_dim = n_kv_heads * head_dim;

    if (h < n_heads) {
        int i = pos * dim + h * head_dim + 2 * d;
        float g0 = dq_all[i], g1 = dq_all[i + 1];
        dq_all[i]     =  g0 * c + g1 * s;
        dq_all[i + 1] = -g0 * s + g1 * c;
    }
    if (h < n_kv_heads) {
        int i = pos * kv_dim + h * head_dim + 2 * d;
        float g0 = dk_store[i], g1 = dk_store[i + 1];
        dk_store[i]     =  g0 * c + g1 * s;
        dk_store[i + 1] = -g0 * s + g1 * c;
    }
}

static void rope_apply_gpu(float *d_q, float *d_k, int seq_len,
                            int n_heads, int n_kv_heads, int head_dim) {
    int max_h = n_heads > n_kv_heads ? n_heads : n_kv_heads;
    dim3 grid(seq_len, max_h);
    int block = head_dim / 2;
    kernel_rope_apply_batched<<<grid, block>>>(
        d_q, d_k, d_rope_cos, d_rope_sin,
        seq_len, n_heads, n_kv_heads, head_dim);
}

static void rope_unapply_gpu(float *d_dq, float *d_dk, int seq_len,
                              int n_heads, int n_kv_heads, int head_dim) {
    int max_h = n_heads > n_kv_heads ? n_heads : n_kv_heads;
    dim3 grid(seq_len, max_h);
    int block = head_dim / 2;
    kernel_rope_unapply_batched<<<grid, block>>>(
        d_dq, d_dk, d_rope_cos, d_rope_sin,
        seq_len, n_heads, n_kv_heads, head_dim);
}

#endif /* __CUDACC__ */

#endif /* ROPE_H */
