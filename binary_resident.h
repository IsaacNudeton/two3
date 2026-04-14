/*
 * binary_resident.h — Device-resident backward + optimizer for binary weights
 *
 * GPU-resident gate+up path: backward kernels write dW directly to device,
 * optimizer (grad clip + Adam headroom) runs on device, latent weights
 * stay on device. Host only sees dX output and periodic checkpoint syncs.
 *
 * Proven: 15.6× speedup over workspace+CPU path (bench_resident_optimizer.cu)
 *
 * Isaac & CC — April 2026
 */

#ifndef BINARY_RESIDENT_H
#define BINARY_RESIDENT_H

#include <cuda_runtime.h>
#include <math.h>
#include "binary_gpu.h"

/* ═══════════════════════════════════════════════════════
 * Per-projection device state (reusable across groups)
 * ═══════════════════════════════════════════════════════ */

typedef struct {
    float *d_grad;      /* [M × K] gradient accumulator */
    float *d_adam_m;    /* [M × K] Adam first moment */
    float *d_adam_v;    /* [M × K] Adam second moment */
    float *d_dY;        /* [max_S × M] backward input staging */
    int M, K;           /* dimensions */
} ResidentWeightBufs;

static void resident_weight_init(ResidentWeightBufs *w, int max_S, int M, int K) {
    w->M = M; w->K = K;
    size_t mk = (size_t)M * K * sizeof(float);
    BGPU_CHECK(cudaMalloc(&w->d_grad, mk));
    BGPU_CHECK(cudaMalloc(&w->d_adam_m, mk));
    BGPU_CHECK(cudaMalloc(&w->d_adam_v, mk));
    BGPU_CHECK(cudaMalloc(&w->d_dY, (size_t)max_S * M * sizeof(float)));
    BGPU_CHECK(cudaMemset(w->d_adam_m, 0, mk));
    BGPU_CHECK(cudaMemset(w->d_adam_v, 0, mk));
}

static void resident_weight_free(ResidentWeightBufs *w) {
    if (w->d_grad) cudaFree(w->d_grad);
    if (w->d_adam_m) cudaFree(w->d_adam_m);
    if (w->d_adam_v) cudaFree(w->d_adam_v);
    if (w->d_dY) cudaFree(w->d_dY);
    memset(w, 0, sizeof(*w));
}

/* ═══════════════════════════════════════════════════════
 * Per-layer device state for QKV resident path
 * ═══════════════════════════════════════════════════════ */

typedef struct {
    ResidentWeightBufs q, k, v, o;
    float *d_X;         /* [max_S × D] shared input staging */
    float *d_dX;        /* [max_S × D] backward output */
    float *d_norm;      /* [1] scratch for norm reduction */
    int max_S, D;
} ResidentAttnState;

static void resident_attn_init(ResidentAttnState *s, int max_S, int D, int KV) {
    s->max_S = max_S; s->D = D;
    resident_weight_init(&s->q, max_S, D, D);
    resident_weight_init(&s->k, max_S, KV, D);
    resident_weight_init(&s->v, max_S, KV, D);
    resident_weight_init(&s->o, max_S, D, D);
    BGPU_CHECK(cudaMalloc(&s->d_X, (size_t)max_S * D * sizeof(float)));
    BGPU_CHECK(cudaMalloc(&s->d_dX, (size_t)max_S * D * sizeof(float)));
    BGPU_CHECK(cudaMalloc(&s->d_norm, sizeof(float)));
}

static void resident_attn_free(ResidentAttnState *s) {
    resident_weight_free(&s->q);
    resident_weight_free(&s->k);
    resident_weight_free(&s->v);
    resident_weight_free(&s->o);
    if (s->d_X) cudaFree(s->d_X);
    if (s->d_dX) cudaFree(s->d_dX);
    if (s->d_norm) cudaFree(s->d_norm);
    memset(s, 0, sizeof(*s));
}

/* Upload Adam state from host (init / checkpoint resume) */
static void resident_attn_sync_adam_h2d(ResidentAttnState *s,
    const float *mq, const float *vq, const float *mk, const float *vk,
    const float *mv, const float *vv, const float *mo, const float *vo
) {
    size_t sq = (size_t)s->q.M * s->q.K * sizeof(float);
    size_t sk = (size_t)s->k.M * s->k.K * sizeof(float);
    size_t so = (size_t)s->o.M * s->o.K * sizeof(float);
    BGPU_CHECK(cudaMemcpy(s->q.d_adam_m, mq, sq, cudaMemcpyHostToDevice));
    BGPU_CHECK(cudaMemcpy(s->q.d_adam_v, vq, sq, cudaMemcpyHostToDevice));
    BGPU_CHECK(cudaMemcpy(s->k.d_adam_m, mk, sk, cudaMemcpyHostToDevice));
    BGPU_CHECK(cudaMemcpy(s->k.d_adam_v, vk, sk, cudaMemcpyHostToDevice));
    BGPU_CHECK(cudaMemcpy(s->v.d_adam_m, mv, sk, cudaMemcpyHostToDevice));
    BGPU_CHECK(cudaMemcpy(s->v.d_adam_v, vv, sk, cudaMemcpyHostToDevice));
    BGPU_CHECK(cudaMemcpy(s->o.d_adam_m, mo, so, cudaMemcpyHostToDevice));
    BGPU_CHECK(cudaMemcpy(s->o.d_adam_v, vo, so, cudaMemcpyHostToDevice));
}

/* Download Adam state to host (checkpointing) */
static void resident_attn_sync_adam_d2h(ResidentAttnState *s,
    float *mq, float *vq, float *mk, float *vk,
    float *mv, float *vv, float *mo, float *vo
) {
    size_t sq = (size_t)s->q.M * s->q.K * sizeof(float);
    size_t sk = (size_t)s->k.M * s->k.K * sizeof(float);
    size_t so = (size_t)s->o.M * s->o.K * sizeof(float);
    BGPU_CHECK(cudaMemcpy(mq, s->q.d_adam_m, sq, cudaMemcpyDeviceToHost));
    BGPU_CHECK(cudaMemcpy(vq, s->q.d_adam_v, sq, cudaMemcpyDeviceToHost));
    BGPU_CHECK(cudaMemcpy(mk, s->k.d_adam_m, sk, cudaMemcpyDeviceToHost));
    BGPU_CHECK(cudaMemcpy(vk, s->k.d_adam_v, sk, cudaMemcpyDeviceToHost));
    BGPU_CHECK(cudaMemcpy(mv, s->v.d_adam_m, sk, cudaMemcpyDeviceToHost));
    BGPU_CHECK(cudaMemcpy(vv, s->v.d_adam_v, sk, cudaMemcpyDeviceToHost));
    BGPU_CHECK(cudaMemcpy(mo, s->o.d_adam_m, so, cudaMemcpyDeviceToHost));
    BGPU_CHECK(cudaMemcpy(vo, s->o.d_adam_v, so, cudaMemcpyDeviceToHost));
}

/* ═══════════════════════════════════════════════════════
 * Per-layer device state for gate+up resident path
 * ═══════════════════════════════════════════════════════ */

typedef struct {
    /* Gradient accumulators — written by backward, consumed by optimizer */
    /* gate+up: M=INTER, K=D */
    float *d_grad_gate;     /* [INTER × D] */
    float *d_grad_up;       /* [INTER × D] */
    float *d_adam_m_gate, *d_adam_v_gate;  /* [INTER × D] each */
    float *d_adam_m_up, *d_adam_v_up;      /* [INTER × D] each */
    float *d_dY_gate;       /* [max_S × INTER] */
    float *d_dY_up;         /* [max_S × INTER] */
    float *d_X;             /* [max_S × D] shared input (pre_ffn_normed) */
    float *d_dX;            /* [max_S × D] backward output */

    /* down: M=D, K=INTER */
    float *d_grad_down;     /* [D × INTER] */
    float *d_adam_m_down, *d_adam_v_down;  /* [D × INTER] each */
    float *d_dY_down;       /* [max_S × D] */
    float *d_X_down;        /* [max_S × INTER] input (ffn_h) */
    float *d_dX_down;       /* [max_S × INTER] backward output */

    /* Scratch */
    float *d_norm;          /* [1] for grad norm reduction */

    /* Capacity */
    int max_S, K, M;       /* K=D, M=INTER for gate+up */
} ResidentFFNState;

/* ═══════════════════════════════════════════════════════
 * Init / Free
 * ═══════════════════════════════════════════════════════ */

static void resident_ffn_init(ResidentFFNState *s, int max_S, int K, int M) {
    s->max_S = max_S;
    s->K = K;
    s->M = M;

    size_t mk = (size_t)M * K * sizeof(float);
    size_t sm = (size_t)max_S * M * sizeof(float);
    size_t sk = (size_t)max_S * K * sizeof(float);

    BGPU_CHECK(cudaMalloc(&s->d_grad_gate, mk));
    BGPU_CHECK(cudaMalloc(&s->d_grad_up, mk));
    BGPU_CHECK(cudaMalloc(&s->d_adam_m_gate, mk));
    BGPU_CHECK(cudaMalloc(&s->d_adam_v_gate, mk));
    BGPU_CHECK(cudaMalloc(&s->d_adam_m_up, mk));
    BGPU_CHECK(cudaMalloc(&s->d_adam_v_up, mk));
    BGPU_CHECK(cudaMalloc(&s->d_dY_gate, sm));
    BGPU_CHECK(cudaMalloc(&s->d_dY_up, sm));
    BGPU_CHECK(cudaMalloc(&s->d_X, sk));
    BGPU_CHECK(cudaMalloc(&s->d_dX, sk));
    BGPU_CHECK(cudaMalloc(&s->d_norm, sizeof(float)));

    /* Down projection: M=K(D), K=M(INTER) — reversed dimensions */
    size_t dk = (size_t)K * M * sizeof(float);  /* D × INTER = K × M */
    size_t s_down_dy = (size_t)max_S * K * sizeof(float);  /* S × D */
    size_t s_down_x = (size_t)max_S * M * sizeof(float);   /* S × INTER */
    BGPU_CHECK(cudaMalloc(&s->d_grad_down, dk));
    BGPU_CHECK(cudaMalloc(&s->d_adam_m_down, dk));
    BGPU_CHECK(cudaMalloc(&s->d_adam_v_down, dk));
    BGPU_CHECK(cudaMalloc(&s->d_dY_down, s_down_dy));
    BGPU_CHECK(cudaMalloc(&s->d_X_down, s_down_x));
    BGPU_CHECK(cudaMalloc(&s->d_dX_down, s_down_x));

    /* Zero Adam moments */
    BGPU_CHECK(cudaMemset(s->d_adam_m_gate, 0, mk));
    BGPU_CHECK(cudaMemset(s->d_adam_v_gate, 0, mk));
    BGPU_CHECK(cudaMemset(s->d_adam_m_up, 0, mk));
    BGPU_CHECK(cudaMemset(s->d_adam_v_up, 0, mk));
    BGPU_CHECK(cudaMemset(s->d_adam_m_down, 0, dk));
    BGPU_CHECK(cudaMemset(s->d_adam_v_down, 0, dk));
}

static void resident_ffn_free(ResidentFFNState *s) {
    if (s->d_grad_gate) cudaFree(s->d_grad_gate);
    if (s->d_grad_up) cudaFree(s->d_grad_up);
    if (s->d_adam_m_gate) cudaFree(s->d_adam_m_gate);
    if (s->d_adam_v_gate) cudaFree(s->d_adam_v_gate);
    if (s->d_adam_m_up) cudaFree(s->d_adam_m_up);
    if (s->d_adam_v_up) cudaFree(s->d_adam_v_up);
    if (s->d_dY_gate) cudaFree(s->d_dY_gate);
    if (s->d_dY_up) cudaFree(s->d_dY_up);
    if (s->d_X) cudaFree(s->d_X);
    if (s->d_dX) cudaFree(s->d_dX);
    if (s->d_grad_down) cudaFree(s->d_grad_down);
    if (s->d_adam_m_down) cudaFree(s->d_adam_m_down);
    if (s->d_adam_v_down) cudaFree(s->d_adam_v_down);
    if (s->d_dY_down) cudaFree(s->d_dY_down);
    if (s->d_X_down) cudaFree(s->d_X_down);
    if (s->d_dX_down) cudaFree(s->d_dX_down);
    if (s->d_norm) cudaFree(s->d_norm);
    memset(s, 0, sizeof(*s));
}

/* Upload Adam state from host (init / checkpoint resume) */
static void resident_ffn_sync_adam_h2d(ResidentFFNState *s,
    const float *h_m_gate, const float *h_v_gate,
    const float *h_m_up, const float *h_v_up,
    const float *h_m_down, const float *h_v_down
) {
    size_t mk = (size_t)s->M * s->K * sizeof(float);
    size_t dk = (size_t)s->K * s->M * sizeof(float);  /* D × INTER for down */
    BGPU_CHECK(cudaMemcpy(s->d_adam_m_gate, h_m_gate, mk, cudaMemcpyHostToDevice));
    BGPU_CHECK(cudaMemcpy(s->d_adam_v_gate, h_v_gate, mk, cudaMemcpyHostToDevice));
    BGPU_CHECK(cudaMemcpy(s->d_adam_m_up, h_m_up, mk, cudaMemcpyHostToDevice));
    BGPU_CHECK(cudaMemcpy(s->d_adam_v_up, h_v_up, mk, cudaMemcpyHostToDevice));
    BGPU_CHECK(cudaMemcpy(s->d_adam_m_down, h_m_down, dk, cudaMemcpyHostToDevice));
    BGPU_CHECK(cudaMemcpy(s->d_adam_v_down, h_v_down, dk, cudaMemcpyHostToDevice));
}

/* Download Adam state to host (checkpointing) */
static void resident_ffn_sync_adam_d2h(ResidentFFNState *s,
    float *h_m_gate, float *h_v_gate,
    float *h_m_up, float *h_v_up,
    float *h_m_down, float *h_v_down
) {
    size_t mk = (size_t)s->M * s->K * sizeof(float);
    size_t dk = (size_t)s->K * s->M * sizeof(float);
    BGPU_CHECK(cudaMemcpy(h_m_gate, s->d_adam_m_gate, mk, cudaMemcpyDeviceToHost));
    BGPU_CHECK(cudaMemcpy(h_v_gate, s->d_adam_v_gate, mk, cudaMemcpyDeviceToHost));
    BGPU_CHECK(cudaMemcpy(h_m_up, s->d_adam_m_up, mk, cudaMemcpyDeviceToHost));
    BGPU_CHECK(cudaMemcpy(h_v_up, s->d_adam_v_up, mk, cudaMemcpyDeviceToHost));
    BGPU_CHECK(cudaMemcpy(h_m_down, s->d_adam_m_down, dk, cudaMemcpyDeviceToHost));
    BGPU_CHECK(cudaMemcpy(h_v_down, s->d_adam_v_down, dk, cudaMemcpyDeviceToHost));
}

/* ═══════════════════════════════════════════════════════
 * GPU kernels — optimizer
 * ═══════════════════════════════════════════════════════ */

/* L2 norm: sum of squares (block reduction → atomicAdd) */
__global__ void resident_kernel_sum_sq(const float *data, float *out, int n) {
    __shared__ float sdata[256];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    sdata[tid] = (i < n) ? data[i] * data[i] : 0.0f;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if (tid == 0) atomicAdd(out, sdata[0]);
}

/* Conditional scale: if norm > max_norm, scale by max_norm/norm */
__global__ void resident_kernel_scale_if(float *data, const float *norm_sq, float max_norm, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float norm = sqrtf(*norm_sq);
    if (norm > max_norm)
        data[i] *= max_norm / norm;
}

/* Adam with {2,3} headroom modulation.
 *
 * Ternary quantization boundaries at 1/3 and 2/3 define the flip surfaces.
 * Attractors at {0, 1/2, 1} — midpoints of the three ternary regions.
 *
 * Symmetric headroom = distance-to-nearest-boundary.
 *   d = min(|w - 1/3|, |w - 2/3|)
 *   h = max(0.1, 2 * (1 - 6*d))        // peaks at d=0, drops to 0 at d=1/6
 *
 * Verification:
 *   w=0    (d=1/3) → h=0.1   (committed to -1)
 *   w=1/6  (d=1/6) → h=0.1   (midpoint of -1 region)
 *   w=1/3  (d=0)   → h=2.0   (flip boundary -1↔0, MAX mobility)
 *   w=1/2  (d=1/6) → h=0.1   (midpoint of substrate — was max in old binary)
 *   w=2/3  (d=0)   → h=2.0   (flip boundary 0↔+1, MAX mobility)
 *   w=5/6  (d=1/6) → h=0.1   (midpoint of +1 region)
 *   w=1    (d=1/3) → h=0.1   (committed to +1)
 */
__global__ void resident_kernel_adam_headroom(
    float *params, const float *grads, float *m, float *v,
    int n, float lr, float beta1, float beta2, float eps,
    float b1_corr, float b2_corr, float headroom_peak
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    float g = grads[i];
    m[i] = beta1 * m[i] + (1.0f - beta1) * g;
    v[i] = beta2 * v[i] + (1.0f - beta2) * g * g;
    float m_hat = m[i] * b1_corr;
    float v_hat = v[i] * b2_corr;
    float update = lr * m_hat / (sqrtf(v_hat) + eps);

    /* CFL clamp */
    if (update >  0.1f) update =  0.1f;
    if (update < -0.1f) update = -0.1f;

    /* {2,3} headroom: symmetric distance to nearest ternary boundary.
     * headroom_peak controls boundary mobility:
     *   peak=1.0 → plain Adam (no headroom modulation)
     *   peak=10.0 → boundary weights move 10× faster than committed */
    float w = params[i];
    float wc = fmaxf(0.0f, fminf(1.0f, w));
    float d1 = fabsf(wc - (1.0f / 3.0f));
    float d2 = fabsf(wc - (2.0f / 3.0f));
    float min_d = fminf(d1, d2);
    float h = fmaxf(0.1f, headroom_peak * (1.0f - 6.0f * min_d));
    params[i] -= update * h;

    /* Hard clamp [0, 1] */
    if (params[i] < 0.0f) params[i] = 0.0f;
    if (params[i] > 1.0f) params[i] = 1.0f;
}

/* Orchestrator: grad clip + Adam headroom on one tensor.
 * headroom_peak: 10.0 for boundary mobility, 1.0 for plain Adam. */
static void resident_grad_clip_adam(
    float *d_W_latent, float *d_grad, float *d_m, float *d_v,
    int size, float max_norm, float lr, float beta1, float beta2, float eps,
    int step, float *d_norm_buf, float headroom_peak
) {
    float b1_corr = 1.0f / (1.0f - powf(beta1, (float)step));
    float b2_corr = 1.0f / (1.0f - powf(beta2, (float)step));

    int threads = 256;
    int blocks = (size + threads - 1) / threads;

    /* Grad clip */
    BGPU_CHECK(cudaMemset(d_norm_buf, 0, sizeof(float)));
    resident_kernel_sum_sq<<<blocks, threads>>>(d_grad, d_norm_buf, size);
    resident_kernel_scale_if<<<blocks, threads>>>(d_grad, d_norm_buf, max_norm, size);

    /* Adam headroom */
    resident_kernel_adam_headroom<<<blocks, threads>>>(
        d_W_latent, d_grad, d_m, d_v, size,
        lr, beta1, beta2, eps, b1_corr, b2_corr, headroom_peak);
}

/* ═══════════════════════════════════════════════════════
 * Resident backward for gate+up
 *
 * Uploads: d_gate_pre, d_up, pre_ffn_normed (from host)
 * Computes: dX on device, dW to persistent device buffers
 * Downloads: dX (d_normed_ffn_all) for gain backward
 * dW stays on device for optimizer
 * ═══════════════════════════════════════════════════════ */

static void resident_backward_gate_up(
    const float *h_dY_gate,         /* [S × M] host — from SwiGLU backward */
    const float *h_dY_up,           /* [S × M] host */
    const float *h_X,               /* [S × K] host — pre_ffn_normed */
    const BinaryWeightsGPU *W_gate, /* packed + d_W_latent */
    const BinaryWeightsGPU *W_up,
    ResidentFFNState *rs,
    float *h_dX,                    /* [S × K] host — ACCUMULATE d_normed_ffn_all */
    int S, int K, int M             /* K=D, M=INTER */
) {
    /* Upload inputs */
    BGPU_CHECK(cudaMemcpy(rs->d_dY_gate, h_dY_gate, (size_t)S * M * sizeof(float), cudaMemcpyHostToDevice));
    BGPU_CHECK(cudaMemcpy(rs->d_dY_up, h_dY_up, (size_t)S * M * sizeof(float), cudaMemcpyHostToDevice));
    BGPU_CHECK(cudaMemcpy(rs->d_X, h_X, (size_t)S * K * sizeof(float), cudaMemcpyHostToDevice));

    /* Zero outputs */
    BGPU_CHECK(cudaMemset(rs->d_dX, 0, (size_t)S * K * sizeof(float)));
    BGPU_CHECK(cudaMemset(rs->d_grad_gate, 0, (size_t)M * K * sizeof(float)));
    BGPU_CHECK(cudaMemset(rs->d_grad_up, 0, (size_t)M * K * sizeof(float)));

    /* Backward for gate */
    {
        float active = binary_gpu_active_density(W_gate);
        float bwd_scale = 1.0f / sqrtf(fmaxf(active, 1e-6f) * (float)M);
        dim3 b1(BTILE_M, BTILE_N);
        dim3 g1((K + BTILE_M - 1) / BTILE_M, (S + BTILE_N - 1) / BTILE_N);
        binary_backward_dx_tiled<<<g1, b1>>>(
            rs->d_dY_gate, W_gate->d_packed_plus, W_gate->d_packed_neg,
            rs->d_dX, S, M, K, bwd_scale);

        dim3 b2(BDWK_BLOCK, BDWK_BLOCK);
        dim3 g2((K + BDWK_BLOCK - 1) / BDWK_BLOCK, (M + BDWK_BLOCK - 1) / BDWK_BLOCK);
        binary_backward_dw_tiled<<<g2, b2>>>(
            rs->d_dY_gate, rs->d_X, W_gate->d_W_latent, rs->d_grad_gate, S, M, K, BINARY_STE_CLIP);
    }

    /* Backward for up (dX accumulates into same buffer) */
    {
        float active = binary_gpu_active_density(W_up);
        float bwd_scale = 1.0f / sqrtf(fmaxf(active, 1e-6f) * (float)M);
        dim3 b1(BTILE_M, BTILE_N);
        dim3 g1((K + BTILE_M - 1) / BTILE_M, (S + BTILE_N - 1) / BTILE_N);
        binary_backward_dx_tiled<<<g1, b1>>>(
            rs->d_dY_up, W_up->d_packed_plus, W_up->d_packed_neg,
            rs->d_dX, S, M, K, bwd_scale);

        dim3 b2(BDWK_BLOCK, BDWK_BLOCK);
        dim3 g2((K + BDWK_BLOCK - 1) / BDWK_BLOCK, (M + BDWK_BLOCK - 1) / BDWK_BLOCK);
        binary_backward_dw_tiled<<<g2, b2>>>(
            rs->d_dY_up, rs->d_X, W_up->d_W_latent, rs->d_grad_up, S, M, K, BINARY_STE_CLIP);
    }

    BGPU_CHECK(cudaDeviceSynchronize());

    /* Download dX only — dW stays on device for optimizer */
    {
        float *h_buf = (float*)malloc((size_t)S * K * sizeof(float));
        BGPU_CHECK(cudaMemcpy(h_buf, rs->d_dX, (size_t)S * K * sizeof(float), cudaMemcpyDeviceToHost));
        for (int i = 0; i < S * K; i++) h_dX[i] += h_buf[i];
        free(h_buf);
    }
}

/* ═══════════════════════════════════════════════════════
 * Resident backward for QKV
 *
 * 3 projections sharing pre_attn_normed as input.
 * Q: [D×D], K: [KV×D], V: [KV×D]
 * dW stays on device. dX downloaded for gain backward.
 * ═══════════════════════════════════════════════════════ */

static void resident_backward_helper(
    const float *h_dY, int S, int M, int K,
    const BinaryWeightsGPU *W, ResidentWeightBufs *wb,
    float *d_X, float *d_dX  /* shared device buffers, dX accumulates */
) {
    /* Upload dY */
    BGPU_CHECK(cudaMemcpy(wb->d_dY, h_dY, (size_t)S * M * sizeof(float), cudaMemcpyHostToDevice));
    BGPU_CHECK(cudaMemset(wb->d_grad, 0, (size_t)M * K * sizeof(float)));

    /* Kernel 1: dX += W^T @ dY (signed) */
    float active = binary_gpu_active_density(W);
    float bwd_scale = 1.0f / sqrtf(fmaxf(active, 1e-6f) * (float)M);
    dim3 b1(BTILE_M, BTILE_N);
    dim3 g1((K + BTILE_M - 1) / BTILE_M, (S + BTILE_N - 1) / BTILE_N);
    binary_backward_dx_tiled<<<g1, b1>>>(
        wb->d_dY, W->d_packed_plus, W->d_packed_neg, d_dX, S, M, K, bwd_scale);

    /* Kernel 2: dW = dY^T @ X with STE clip */
    dim3 b2(BDWK_BLOCK, BDWK_BLOCK);
    dim3 g2((K + BDWK_BLOCK - 1) / BDWK_BLOCK, (M + BDWK_BLOCK - 1) / BDWK_BLOCK);
    binary_backward_dw_tiled<<<g2, b2>>>(wb->d_dY, d_X, W->d_W_latent, wb->d_grad, S, M, K, BINARY_STE_CLIP);
}

static void resident_backward_qkv(
    const float *h_dY_q,            /* [S × D] host */
    const float *h_dY_k,            /* [S × KV] host */
    const float *h_dY_v,            /* [S × KV] host */
    const float *h_X,               /* [S × D] host — pre_attn_normed */
    const BinaryWeightsGPU *W_q,
    const BinaryWeightsGPU *W_k,
    const BinaryWeightsGPU *W_v,
    ResidentAttnState *rs,
    float *h_dX,                    /* [S × D] host — ACCUMULATE */
    int S, int D, int KV
) {
    /* Upload shared input */
    BGPU_CHECK(cudaMemcpy(rs->d_X, h_X, (size_t)S * D * sizeof(float), cudaMemcpyHostToDevice));
    BGPU_CHECK(cudaMemset(rs->d_dX, 0, (size_t)S * D * sizeof(float)));

    /* Backward for Q, K, V — all accumulate into same dX */
    resident_backward_helper(h_dY_q, S, D, D, W_q, &rs->q, rs->d_X, rs->d_dX);
    resident_backward_helper(h_dY_k, S, KV, D, W_k, &rs->k, rs->d_X, rs->d_dX);
    resident_backward_helper(h_dY_v, S, KV, D, W_v, &rs->v, rs->d_X, rs->d_dX);

    BGPU_CHECK(cudaDeviceSynchronize());

    /* Download dX only */
    {
        float *h_buf = (float*)malloc((size_t)S * D * sizeof(float));
        BGPU_CHECK(cudaMemcpy(h_buf, rs->d_dX, (size_t)S * D * sizeof(float), cudaMemcpyDeviceToHost));
        for (int i = 0; i < S * D; i++) h_dX[i] += h_buf[i];
        free(h_buf);
    }
}

/* ═══════════════════════════════════════════════════════
 * Resident backward for down projection (single)
 *
 * dY=d_ffn_out [S×D], X=ffn_h [S×INTER]
 * dW stays on device. dX (d_h) downloaded for SwiGLU backward.
 * ═══════════════════════════════════════════════════════ */

/* ═══════════════════════════════════════════════════════
 * Resident backward for O projection (single)
 *
 * dY=d_o_proj_all [S×D], X=attn_out [S×D]
 * dW stays on device. dX downloaded for attn backward.
 * ═══════════════════════════════════════════════════════ */

static void resident_backward_o(
    const float *h_dY,              /* [S × D] host — d_o_proj_all */
    const float *h_X,               /* [S × D] host — attn_out */
    const BinaryWeightsGPU *W_o,
    ResidentAttnState *rs,
    float *h_dX,                    /* [S × D] host — ACCUMULATE */
    int S, int D
) {
    /* Upload inputs — reuse attn staging buffers */
    BGPU_CHECK(cudaMemcpy(rs->d_X, h_X, (size_t)S * D * sizeof(float), cudaMemcpyHostToDevice));
    BGPU_CHECK(cudaMemset(rs->d_dX, 0, (size_t)S * D * sizeof(float)));

    resident_backward_helper(h_dY, S, D, D, W_o, &rs->o, rs->d_X, rs->d_dX);

    BGPU_CHECK(cudaDeviceSynchronize());

    /* Download dX */
    {
        float *h_buf = (float*)malloc((size_t)S * D * sizeof(float));
        BGPU_CHECK(cudaMemcpy(h_buf, rs->d_dX, (size_t)S * D * sizeof(float), cudaMemcpyDeviceToHost));
        for (int i = 0; i < S * D; i++) h_dX[i] += h_buf[i];
        free(h_buf);
    }
}

static void resident_backward_down(
    const float *h_dY,              /* [S × D] host — d_ffn_out */
    const float *h_X,               /* [S × INTER] host — ffn_h */
    const BinaryWeightsGPU *W_down,
    ResidentFFNState *rs,
    float *h_dX,                    /* [S × INTER] host — ACCUMULATE d_h */
    int S, int D, int INTER
) {
    int M = D, K = INTER;  /* down: D rows, INTER cols */

    /* Upload inputs */
    BGPU_CHECK(cudaMemcpy(rs->d_dY_down, h_dY, (size_t)S * M * sizeof(float), cudaMemcpyHostToDevice));
    BGPU_CHECK(cudaMemcpy(rs->d_X_down, h_X, (size_t)S * K * sizeof(float), cudaMemcpyHostToDevice));

    /* Zero outputs */
    BGPU_CHECK(cudaMemset(rs->d_dX_down, 0, (size_t)S * K * sizeof(float)));
    BGPU_CHECK(cudaMemset(rs->d_grad_down, 0, (size_t)M * K * sizeof(float)));

    /* Kernel 1: dX = W^T @ dY (signed) */
    float active_dn = binary_gpu_active_density(W_down);
    float bwd_scale = 1.0f / sqrtf(fmaxf(active_dn, 1e-6f) * (float)M);
    dim3 b1(BTILE_M, BTILE_N);
    dim3 g1((K + BTILE_M - 1) / BTILE_M, (S + BTILE_N - 1) / BTILE_N);
    binary_backward_dx_tiled<<<g1, b1>>>(
        rs->d_dY_down, W_down->d_packed_plus, W_down->d_packed_neg,
        rs->d_dX_down, S, M, K, bwd_scale);

    /* Kernel 2: dW = dY^T @ X with STE clip */
    dim3 b2(BDWK_BLOCK, BDWK_BLOCK);
    dim3 g2((K + BDWK_BLOCK - 1) / BDWK_BLOCK, (M + BDWK_BLOCK - 1) / BDWK_BLOCK);
    binary_backward_dw_tiled<<<g2, b2>>>(
        rs->d_dY_down, rs->d_X_down, W_down->d_W_latent, rs->d_grad_down, S, M, K, BINARY_STE_CLIP);

    BGPU_CHECK(cudaDeviceSynchronize());

    /* Download dX — needed by SwiGLU backward on host */
    {
        float *h_buf = (float*)malloc((size_t)S * K * sizeof(float));
        BGPU_CHECK(cudaMemcpy(h_buf, rs->d_dX_down, (size_t)S * K * sizeof(float), cudaMemcpyDeviceToHost));
        for (int i = 0; i < S * K; i++) h_dX[i] += h_buf[i];
        free(h_buf);
    }
}

#endif /* BINARY_RESIDENT_H */
