/*
 * binary_gpu_workspace.h — GPU-Resident Workspace for Binary Projection
 *
 * Refactors binary_gpu.h to eliminate per-call cudaMalloc/cudaFree.
 * Keeps the same math kernels, just persistent buffers.
 *
 * Isaac & CC — April 2026
 */

#ifndef BINARY_GPU_WORKSPACE_H
#define BINARY_GPU_WORKSPACE_H

#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include "binary_gpu.h"

/* ═══════════════════════════════════════════════════════
 * GPU-resident workspace — reusable buffers
 * ═══════════════════════════════════════════════════════ */

typedef struct {
    /* Persistent device buffers */
    int8_t  *d_x_q;      /* [max_S × max_K] quantized activations */
    float   *d_scales;   /* [max_S] absmax scales */
    int32_t *d_acc;      /* [max_S × max_M] accumulators */
    float   *d_output;   /* [max_S × max_M] float output (optional) */

    /* Backward device buffers */
    float *d_dY_bw;      /* [max_S × max_M] gradient from above */
    float *d_X_bw;       /* [max_S × max_K] saved input */
    float *d_dX_bw;      /* [max_S × max_K] gradient w.r.t. input (accumulate) */
    float *d_W_latent_bw;/* [max_M × max_K] latent weights */
    float *d_dW_bw;      /* [max_M × max_K] gradient w.r.t. weights (accumulate) */

    /* Capacity */
    int max_S;  /* max sequence length / batch */
    int max_K;  /* max input dimension */
    int max_M;  /* max output dimension */

    /* Host staging buffers (for H2D/D2H) */
    int8_t  *h_x_q;
    float   *h_scales;
    int32_t *h_acc;
    float   *h_dX_bw;    /* backward dX staging */
    float   *h_dW_bw;    /* backward dW staging */
} BinaryGPUWorkspace;

/* ═══════════════════════════════════════════════════════
 * Initialize workspace with given capacity
 * ═══════════════════════════════════════════════════════ */

static void binary_workspace_init(BinaryGPUWorkspace *ws, int max_S, int max_K, int max_M) {
    ws->max_S = max_S;
    ws->max_K = max_K;
    ws->max_M = max_M;

    /* Allocate forward device buffers */
    BGPU_CHECK(cudaMalloc(&ws->d_x_q, (size_t)max_S * max_K * sizeof(int8_t)));
    BGPU_CHECK(cudaMalloc(&ws->d_scales, max_S * sizeof(float)));
    BGPU_CHECK(cudaMalloc(&ws->d_acc, (size_t)max_S * max_M * sizeof(int32_t)));
    BGPU_CHECK(cudaMalloc(&ws->d_output, (size_t)max_S * max_M * sizeof(float)));

    /* Allocate backward device buffers */
    BGPU_CHECK(cudaMalloc(&ws->d_dY_bw, (size_t)max_S * max_M * sizeof(float)));
    BGPU_CHECK(cudaMalloc(&ws->d_X_bw, (size_t)max_S * max_K * sizeof(float)));
    BGPU_CHECK(cudaMalloc(&ws->d_dX_bw, (size_t)max_S * max_K * sizeof(float)));
    BGPU_CHECK(cudaMalloc(&ws->d_W_latent_bw, (size_t)max_M * max_K * sizeof(float)));
    BGPU_CHECK(cudaMalloc(&ws->d_dW_bw, (size_t)max_M * max_K * sizeof(float)));

    /* Allocate host staging buffers */
    ws->h_x_q = (int8_t*)malloc((size_t)max_S * max_K * sizeof(int8_t));
    ws->h_scales = (float*)malloc(max_S * sizeof(float));
    ws->h_acc = (int32_t*)malloc((size_t)max_S * max_M * sizeof(int32_t));
    ws->h_dX_bw = (float*)malloc((size_t)max_S * max_K * sizeof(float));
    ws->h_dW_bw = (float*)malloc((size_t)max_M * max_K * sizeof(float));
}

/* ═══════════════════════════════════════════════════════
 * Ensure workspace has capacity for given dimensions
 * (Reallocate if needed — should be rare)
 * ═══════════════════════════════════════════════════════ */

static void binary_workspace_ensure(BinaryGPUWorkspace *ws, int S, int K, int M) {
    if (S <= ws->max_S && K <= ws->max_K && M <= ws->max_M)
        return;  /* Already have capacity */

    /* Free old buffers */
    cudaFree(ws->d_x_q);
    cudaFree(ws->d_scales);
    cudaFree(ws->d_acc);
    cudaFree(ws->d_output);
    cudaFree(ws->d_dY_bw);
    cudaFree(ws->d_X_bw);
    cudaFree(ws->d_dX_bw);
    cudaFree(ws->d_W_latent_bw);
    cudaFree(ws->d_dW_bw);
    free(ws->h_x_q);
    free(ws->h_scales);
    free(ws->h_acc);
    free(ws->h_dX_bw);
    free(ws->h_dW_bw);

    /* Reallocate with new capacity */
    ws->max_S = S;
    ws->max_K = K;
    ws->max_M = M;

    BGPU_CHECK(cudaMalloc(&ws->d_x_q, (size_t)S * K * sizeof(int8_t)));
    BGPU_CHECK(cudaMalloc(&ws->d_scales, S * sizeof(float)));
    BGPU_CHECK(cudaMalloc(&ws->d_acc, (size_t)S * M * sizeof(int32_t)));
    BGPU_CHECK(cudaMalloc(&ws->d_output, (size_t)S * M * sizeof(float)));
    BGPU_CHECK(cudaMalloc(&ws->d_dY_bw, (size_t)S * M * sizeof(float)));
    BGPU_CHECK(cudaMalloc(&ws->d_X_bw, (size_t)S * K * sizeof(float)));
    BGPU_CHECK(cudaMalloc(&ws->d_dX_bw, (size_t)S * K * sizeof(float)));
    BGPU_CHECK(cudaMalloc(&ws->d_W_latent_bw, (size_t)M * K * sizeof(float)));
    BGPU_CHECK(cudaMalloc(&ws->d_dW_bw, (size_t)M * K * sizeof(float)));

    ws->h_x_q = (int8_t*)malloc((size_t)S * K * sizeof(int8_t));
    ws->h_scales = (float*)malloc(S * sizeof(float));
    ws->h_acc = (int32_t*)malloc((size_t)S * M * sizeof(int32_t));
    ws->h_dX_bw = (float*)malloc((size_t)S * K * sizeof(float));
    ws->h_dW_bw = (float*)malloc((size_t)M * K * sizeof(float));
}

/* ═══════════════════════════════════════════════════════
 * Free workspace
 * ═══════════════════════════════════════════════════════ */

static void binary_workspace_free(BinaryGPUWorkspace *ws) {
    /* Forward buffers */
    if (ws->d_x_q) cudaFree(ws->d_x_q);
    if (ws->d_scales) cudaFree(ws->d_scales);
    if (ws->d_acc) cudaFree(ws->d_acc);
    if (ws->d_output) cudaFree(ws->d_output);

    /* Backward buffers */
    if (ws->d_dY_bw) cudaFree(ws->d_dY_bw);
    if (ws->d_X_bw) cudaFree(ws->d_X_bw);
    if (ws->d_dX_bw) cudaFree(ws->d_dX_bw);
    if (ws->d_W_latent_bw) cudaFree(ws->d_W_latent_bw);
    if (ws->d_dW_bw) cudaFree(ws->d_dW_bw);

    /* Host staging buffers */
    if (ws->h_x_q) free(ws->h_x_q);
    if (ws->h_scales) free(ws->h_scales);
    if (ws->h_acc) free(ws->h_acc);
    if (ws->h_dX_bw) free(ws->h_dX_bw);
    if (ws->h_dW_bw) free(ws->h_dW_bw);

    ws->d_x_q = NULL;
    ws->d_scales = NULL;
    ws->d_acc = NULL;
    ws->d_output = NULL;
    ws->d_dY_bw = NULL;
    ws->d_X_bw = NULL;
    ws->d_dX_bw = NULL;
    ws->d_W_latent_bw = NULL;
    ws->d_dW_bw = NULL;
    ws->h_x_q = NULL;
    ws->h_scales = NULL;
    ws->h_acc = NULL;
    ws->h_dX_bw = NULL;
    ws->h_dW_bw = NULL;
}

/* ═══════════════════════════════════════════════════════
 * Quantize float input to int8 (host → device)
 * Uses workspace buffers, no allocation
 * ═══════════════════════════════════════════════════════ */

static void binary_workspace_quantize(BinaryGPUWorkspace *ws,
                                       const float *h_input,  /* [S × K] host */
                                       int S, int K) {
    /* Quantize on host into staging buffer */
    for (int s = 0; s < S; s++) {
        float absmax = 0.0f;
        for (int k = 0; k < K; k++) {
            float v = fabsf(h_input[s * K + k]);
            if (v > absmax) absmax = v;
        }
        ws->h_scales[s] = (absmax > 0.0f) ? absmax : 1e-10f;
        float inv = 127.0f / ws->h_scales[s];
        for (int k = 0; k < K; k++) {
            float v = h_input[s * K + k] * inv;
            if (v > 127.0f) v = 127.0f;
            if (v < -127.0f) v = -127.0f;
            ws->h_x_q[s * K + k] = (int8_t)v;
        }
    }
    
    /* H2D transfer */
    BGPU_CHECK(cudaMemcpy(ws->d_x_q, ws->h_x_q, (size_t)S * K * sizeof(int8_t),
                          cudaMemcpyHostToDevice));
    BGPU_CHECK(cudaMemcpy(ws->d_scales, ws->h_scales, S * sizeof(float),
                          cudaMemcpyHostToDevice));
}

/* ═══════════════════════════════════════════════════════
 * Dequantize accumulators to float (device → host)
 * Uses workspace buffers, no allocation
 * ═══════════════════════════════════════════════════════ */

static void binary_workspace_dequant(BinaryGPUWorkspace *ws,
                                      float *h_output,  /* [S × M] host */
                                      int S, int M,
                                      float density, int K) {
    /* D2H transfer */
    BGPU_CHECK(cudaMemcpy(ws->h_acc, ws->d_acc, (size_t)S * M * sizeof(int32_t),
                          cudaMemcpyDeviceToHost));
    
    /* Dequantize on host */
    for (int s = 0; s < S; s++) {
        float a_scale = ws->h_scales[s] / 127.0f;
        float combined = a_scale / sqrtf(density * (float)K);
        for (int m = 0; m < M; m++) {
            h_output[s * M + m] = (float)ws->h_acc[s * M + m] * combined;
        }
    }
}

/* ═══════════════════════════════════════════════════════
 * Full forward: float input (host) → float output (host)
 * Uses workspace buffers — no per-call allocation
 * ═══════════════════════════════════════════════════════ */

static void binary_project_batch_gpu_ws(
    const BinaryWeightsGPU *W,
    BinaryGPUWorkspace *ws,
    const float *h_input,   /* [S × K] host */
    float *h_output,        /* [S × M] host */
    int S, int K
) {
    int M = W->rows;
    
    /* Ensure capacity */
    binary_workspace_ensure(ws, S, K, M);
    
    /* Quantize: host → device */
    binary_workspace_quantize(ws, h_input, S, K);
    
    /* Matmul on GPU */
    BGPU_CHECK(cudaMemset(ws->d_acc, 0, (size_t)S * M * sizeof(int32_t)));
    binary_matmul_gpu(W->d_packed, ws->d_x_q, ws->d_acc, S, M, K);
    
    /* Dequantize: device → host */
    binary_workspace_dequant(ws, h_output, S, M, W->density, K);
}

/* ═══════════════════════════════════════════════════════
 * Multi-output workspace forward: quantize ONCE, project N times
 * Used for QKV (N=3) and gate+up (N=2). Saves N-1 quantizations.
 * Same interface as binary_project_multi_gpu but uses workspace.
 * ═══════════════════════════════════════════════════════ */

static void binary_project_multi_gpu_ws(
    const BinaryWeightsGPU *W_list[],  /* [N] weight matrices */
    float *h_output_list[],            /* [N] output buffers (host) */
    BinaryGPUWorkspace *ws,
    const float *h_input,              /* [S × K] host */
    int N, int S, int K
) {
    /* Find max output dimension across all weights */
    int max_M = 0;
    for (int i = 0; i < N; i++) {
        if (W_list[i]->rows > max_M) max_M = W_list[i]->rows;
    }

    /* Ensure capacity BEFORE any work — prevents reallocation mid-call */
    binary_workspace_ensure(ws, S, K, max_M);

    /* Quantize input ONCE on host */
    for (int s = 0; s < S; s++) {
        float absmax = 0.0f;
        for (int k = 0; k < K; k++) {
            float v = fabsf(h_input[s * K + k]);
            if (v > absmax) absmax = v;
        }
        ws->h_scales[s] = (absmax > 0.0f) ? absmax : 1e-10f;
        float inv = 127.0f / ws->h_scales[s];
        for (int k = 0; k < K; k++) {
            float v = h_input[s * K + k] * inv;
            if (v > 127.0f) v = 127.0f;
            if (v < -127.0f) v = -127.0f;
            ws->h_x_q[s * K + k] = (int8_t)v;
        }
    }

    /* H2D transfer */
    BGPU_CHECK(cudaMemcpy(ws->d_x_q, ws->h_x_q, (size_t)S * K * sizeof(int8_t),
                          cudaMemcpyHostToDevice));
    BGPU_CHECK(cudaMemcpy(ws->d_scales, ws->h_scales, S * sizeof(float),
                          cudaMemcpyHostToDevice));

    /* Project through each weight matrix */
    for (int i = 0; i < N; i++) {
        int M = W_list[i]->rows;

        BGPU_CHECK(cudaMemset(ws->d_acc, 0, (size_t)S * M * sizeof(int32_t)));
        binary_matmul_gpu(W_list[i]->d_packed, ws->d_x_q, ws->d_acc, S, M, K);

        /* D2H and dequantize */
        BGPU_CHECK(cudaMemcpy(ws->h_acc, ws->d_acc, (size_t)S * M * sizeof(int32_t),
                              cudaMemcpyDeviceToHost));

        for (int s = 0; s < S; s++) {
            float a_scale = ws->h_scales[s] / 127.0f;
            float combined = a_scale / sqrtf(W_list[i]->density * (float)K);
            for (int m = 0; m < M; m++) {
                h_output_list[i][s * M + m] = (float)ws->h_acc[s * M + m] * combined;
            }
        }
    }
}

/* ═══════════════════════════════════════════════════════
 * Device-resident forward: float input (device) → float output (device)
 * No H2D/D2H at all — for full GPU training pipeline
 * NOTE: Currently experimental — not integrated into training.
 * ═══════════════════════════════════════════════════════ */

__global__ void kernel_quantize_acts_device(
    int8_t *d_x_q, float *d_scales, const float *d_input,
    int S, int K
) {
    int s = blockIdx.x;
    if (s >= S) return;

    /* Find absmax for this sample (single thread does full reduction) */
    float absmax = 0.0f;
    for (int k = 0; k < K; k++) {
        float v = fabsf(d_input[s * K + k]);
        if (v > absmax) absmax = v;
    }
    d_scales[s] = (absmax > 0.0f) ? absmax : 1e-10f;

    /* Quantize: each thread in block handles elements */
    float inv = 127.0f / d_scales[s];
    for (int k = threadIdx.x; k < K; k += blockDim.x) {
        float v = d_input[s * K + k] * inv;
        if (v > 127.0f) v = 127.0f;
        if (v < -127.0f) v = -127.0f;
        d_x_q[s * K + k] = (int8_t)v;
    }
}

__global__ void kernel_dequant_acts_device(
    float *d_output, const int32_t *d_acc, const float *d_scales,
    int S, int M, float density, int K
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = S * M;
    if (idx >= total) return;

    int s = idx / M;
    int m = idx % M;

    float a_scale = d_scales[s] / 127.0f;
    float combined = a_scale / sqrtf(density * (float)K);
    d_output[s * M + m] = (float)d_acc[s * M + m] * combined;
}

static void binary_project_batch_gpu_resident(
    const BinaryWeightsGPU *W,
    BinaryGPUWorkspace *ws,
    const float *d_input,   /* [S × K] device */
    float *d_output,        /* [S × M] device */
    int S, int K
) {
    int M = W->rows;

    /* Ensure capacity BEFORE any work */
    binary_workspace_ensure(ws, S, K, M);

    /* Quantize on device */
    int threads = 256;
    kernel_quantize_acts_device<<<S, threads, 0, NULL>>>(
        ws->d_x_q, ws->d_scales, d_input, S, K);

    /* Matmul on GPU */
    BGPU_CHECK(cudaMemset(ws->d_acc, 0, (size_t)S * M * sizeof(int32_t)));
    binary_matmul_gpu(W->d_packed, ws->d_x_q, ws->d_acc, S, M, K);

    /* Dequantize on device */
    int blocks = (S * M + threads - 1) / threads;
    kernel_dequant_acts_device<<<blocks, threads>>>(
        d_output, ws->d_acc, ws->d_scales, S, M, W->density, K);
}

/* ═══════════════════════════════════════════════════════
 * Multi-output device-resident: quantize ONCE, project N times
 * Used for QKV (N=3) and gate+up (N=2)
 * NOTE: Currently experimental — not integrated into training.
 * ═══════════════════════════════════════════════════════ */

static void binary_project_multi_gpu_resident(
    const BinaryWeightsGPU *W_list[],  /* [N] weight matrices */
    float *d_output_list[],            /* [N] output buffers */
    BinaryGPUWorkspace *ws,
    const float *d_input,              /* [S × K] device */
    int N, int S, int K
) {
    /* Find max output dimension across all weights */
    int max_M = 0;
    for (int i = 0; i < N; i++) {
        if (W_list[i]->rows > max_M) max_M = W_list[i]->rows;
    }

    /* Ensure capacity BEFORE any work */
    binary_workspace_ensure(ws, S, K, max_M);

    /* Quantize input ONCE */
    int threads = 256;
    kernel_quantize_acts_device<<<S, threads, 0, NULL>>>(
        ws->d_x_q, ws->d_scales, d_input, S, K);

    /* Project through each weight matrix */
    for (int i = 0; i < N; i++) {
        int M = W_list[i]->rows;

        BGPU_CHECK(cudaMemset(ws->d_acc, 0, (size_t)S * M * sizeof(int32_t)));
        binary_matmul_gpu(W_list[i]->d_packed, ws->d_x_q, ws->d_acc, S, M, K);

        int blocks = (S * M + threads - 1) / threads;
        kernel_dequant_acts_device<<<blocks, threads>>>(
            d_output_list[i], ws->d_acc, ws->d_scales, S, M,
            W_list[i]->density, K);
    }
}

/* ═══════════════════════════════════════════════════════
 * Workspace backward: dX + dW with persistent buffers
 * Same interface as binary_backward_batch_gpu but uses workspace.
 * ═══════════════════════════════════════════════════════ */

static void binary_backward_batch_gpu_ws(
    const float *dY,             /* [S × M] host */
    const float *X,              /* [S × K] host */
    const float *W_latent,       /* [M × K] host, float latent weights */
    const BinaryWeightsGPU *W,   /* packed binary (device) */
    BinaryGPUWorkspace *ws,
    float *dX,                   /* [S × K] host, ACCUMULATE */
    float *dW_latent,            /* [M × K] host, ACCUMULATE */
    int S, int M, int K
) {
    /* Ensure capacity BEFORE any work */
    binary_workspace_ensure(ws, S, K, M);

    /* H2D transfer */
    BGPU_CHECK(cudaMemcpy(ws->d_dY_bw, dY, (size_t)S * M * sizeof(float), cudaMemcpyHostToDevice));
    BGPU_CHECK(cudaMemcpy(ws->d_X_bw, X, (size_t)S * K * sizeof(float), cudaMemcpyHostToDevice));
    BGPU_CHECK(cudaMemset(ws->d_dX_bw, 0, (size_t)S * K * sizeof(float)));
    BGPU_CHECK(cudaMemcpy(ws->d_W_latent_bw, W_latent, (size_t)M * K * sizeof(float), cudaMemcpyHostToDevice));
    BGPU_CHECK(cudaMemcpy(ws->d_dW_bw, dW_latent, (size_t)M * K * sizeof(float), cudaMemcpyHostToDevice));

    /* Kernel 1: dX = W^T @ dY (tiled, with backward scale) */
    {
        float bwd_scale = 1.0f / sqrtf(W->density * (float)M + 1e-6f);
        dim3 block(BTILE_M, BTILE_N);
        dim3 grid((K + BTILE_M - 1) / BTILE_M,
                  (S + BTILE_N - 1) / BTILE_N);
        binary_backward_dx_tiled<<<grid, block>>>(
            ws->d_dY_bw, W->d_packed, ws->d_dX_bw, S, M, K, bwd_scale);
        BGPU_CHECK(cudaGetLastError());
    }

    /* Kernel 2: dW = dY^T @ X with STE clip */
    {
        dim3 block(BDWK_BLOCK, BDWK_BLOCK);
        dim3 grid((K + BDWK_BLOCK - 1) / BDWK_BLOCK,
                  (M + BDWK_BLOCK - 1) / BDWK_BLOCK);
        binary_backward_dw_tiled<<<grid, block>>>(
            ws->d_dY_bw, ws->d_X_bw, ws->d_W_latent_bw, ws->d_dW_bw, S, M, K, BINARY_STE_CLIP);
        BGPU_CHECK(cudaGetLastError());
    }

    BGPU_CHECK(cudaDeviceSynchronize());

    /* D2H — accumulate into host buffers */
    BGPU_CHECK(cudaMemcpy(ws->h_dX_bw, ws->d_dX_bw, (size_t)S * K * sizeof(float), cudaMemcpyDeviceToHost));
    BGPU_CHECK(cudaMemcpy(ws->h_dW_bw, ws->d_dW_bw, (size_t)M * K * sizeof(float), cudaMemcpyDeviceToHost));
    for (int i = 0; i < S * K; i++) dX[i] += ws->h_dX_bw[i];
    for (int i = 0; i < M * K; i++) dW_latent[i] = ws->h_dW_bw[i];
}

#endif /* BINARY_GPU_WORKSPACE_H */
