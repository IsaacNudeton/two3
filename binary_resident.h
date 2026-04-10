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
 * Per-layer device state for gate+up resident path
 * ═══════════════════════════════════════════════════════ */

typedef struct {
    /* Gradient accumulators — written by backward, consumed by optimizer */
    float *d_grad_gate;     /* [M × K] = [INTER × D] */
    float *d_grad_up;       /* [M × K] */

    /* Adam moments — persistent on device */
    float *d_adam_m_gate, *d_adam_v_gate;  /* [M × K] each */
    float *d_adam_m_up, *d_adam_v_up;      /* [M × K] each */

    /* Backward staging — input data uploaded per call */
    float *d_dY_gate;       /* [max_S × M] */
    float *d_dY_up;         /* [max_S × M] */
    float *d_X;             /* [max_S × K] shared input (pre_ffn_normed) */
    float *d_dX;            /* [max_S × K] backward output */

    /* Scratch */
    float *d_norm;          /* [1] for grad norm reduction */

    /* Capacity */
    int max_S, K, M;
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

    /* Zero Adam moments */
    BGPU_CHECK(cudaMemset(s->d_adam_m_gate, 0, mk));
    BGPU_CHECK(cudaMemset(s->d_adam_v_gate, 0, mk));
    BGPU_CHECK(cudaMemset(s->d_adam_m_up, 0, mk));
    BGPU_CHECK(cudaMemset(s->d_adam_v_up, 0, mk));
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
    if (s->d_norm) cudaFree(s->d_norm);
    memset(s, 0, sizeof(*s));
}

/* Upload Adam state from host (init / checkpoint resume) */
static void resident_ffn_sync_adam_h2d(ResidentFFNState *s,
    const float *h_m_gate, const float *h_v_gate,
    const float *h_m_up, const float *h_v_up
) {
    size_t mk = (size_t)s->M * s->K * sizeof(float);
    BGPU_CHECK(cudaMemcpy(s->d_adam_m_gate, h_m_gate, mk, cudaMemcpyHostToDevice));
    BGPU_CHECK(cudaMemcpy(s->d_adam_v_gate, h_v_gate, mk, cudaMemcpyHostToDevice));
    BGPU_CHECK(cudaMemcpy(s->d_adam_m_up, h_m_up, mk, cudaMemcpyHostToDevice));
    BGPU_CHECK(cudaMemcpy(s->d_adam_v_up, h_v_up, mk, cudaMemcpyHostToDevice));
}

/* Download Adam state to host (checkpointing) */
static void resident_ffn_sync_adam_d2h(ResidentFFNState *s,
    float *h_m_gate, float *h_v_gate,
    float *h_m_up, float *h_v_up
) {
    size_t mk = (size_t)s->M * s->K * sizeof(float);
    BGPU_CHECK(cudaMemcpy(h_m_gate, s->d_adam_m_gate, mk, cudaMemcpyDeviceToHost));
    BGPU_CHECK(cudaMemcpy(h_v_gate, s->d_adam_v_gate, mk, cudaMemcpyDeviceToHost));
    BGPU_CHECK(cudaMemcpy(h_m_up, s->d_adam_m_up, mk, cudaMemcpyDeviceToHost));
    BGPU_CHECK(cudaMemcpy(h_v_up, s->d_adam_v_up, mk, cudaMemcpyDeviceToHost));
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

/* Adam with headroom modulation — direct port of CPU adam_update_headroom */
__global__ void resident_kernel_adam_headroom(
    float *params, const float *grads, float *m, float *v,
    int n, float lr, float beta1, float beta2, float eps,
    float b1_corr, float b2_corr
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

    /* Headroom modulation */
    float w = params[i];
    float wc = fmaxf(0.0f, fminf(1.0f, w));
    float h;
    if (update > 0.0f)
        h = fmaxf(0.1f, 2.0f * (1.0f - wc));
    else
        h = fmaxf(0.1f, 2.0f * wc);
    params[i] -= update * h;

    /* Hard clamp [0, 1] */
    if (params[i] < 0.0f) params[i] = 0.0f;
    if (params[i] > 1.0f) params[i] = 1.0f;
}

/* Orchestrator: grad clip + Adam headroom on one tensor */
static void resident_grad_clip_adam(
    float *d_W_latent, float *d_grad, float *d_m, float *d_v,
    int size, float max_norm, float lr, float beta1, float beta2, float eps,
    int step, float *d_norm_buf
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
        lr, beta1, beta2, eps, b1_corr, b2_corr);
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
        float bwd_scale = 1.0f / sqrtf(W_gate->density * (float)M + 1e-6f);
        dim3 b1(BTILE_M, BTILE_N);
        dim3 g1((K + BTILE_M - 1) / BTILE_M, (S + BTILE_N - 1) / BTILE_N);
        binary_backward_dx_tiled<<<g1, b1>>>(
            rs->d_dY_gate, W_gate->d_packed, rs->d_dX, S, M, K, bwd_scale);

        dim3 b2(BDWK_BLOCK, BDWK_BLOCK);
        dim3 g2((K + BDWK_BLOCK - 1) / BDWK_BLOCK, (M + BDWK_BLOCK - 1) / BDWK_BLOCK);
        binary_backward_dw_tiled<<<g2, b2>>>(
            rs->d_dY_gate, rs->d_X, W_gate->d_W_latent, rs->d_grad_gate, S, M, K, BINARY_STE_CLIP);
    }

    /* Backward for up (dX accumulates into same buffer) */
    {
        float bwd_scale = 1.0f / sqrtf(W_up->density * (float)M + 1e-6f);
        dim3 b1(BTILE_M, BTILE_N);
        dim3 g1((K + BTILE_M - 1) / BTILE_M, (S + BTILE_N - 1) / BTILE_N);
        binary_backward_dx_tiled<<<g1, b1>>>(
            rs->d_dY_up, W_up->d_packed, rs->d_dX, S, M, K, bwd_scale);

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

#endif /* BINARY_RESIDENT_H */
