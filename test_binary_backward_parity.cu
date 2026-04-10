/*
 * test_binary_backward_parity.cu — GPU vs GPU-Workspace Backward Parity
 * Multi-shape validation
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <cuda_runtime.h>
#include "binary_gpu.h"
#include "binary_gpu_workspace.h"

#define CHECK_CUDA(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err)); exit(1); \
    } \
} while(0)

static void init_weights(float *w, int rows, int cols) {
    for (int i = 0; i < rows * cols; i++)
        w[i] = (float)rand() / (float)RAND_MAX;
}

static void init_input(float *x, int S, int K) {
    for (int i = 0; i < S * K; i++)
        x[i] = ((float)rand() / (float)RAND_MAX) * 2.0f - 1.0f;
}

static void init_grad(float *g, int S, int M) {
    for (int i = 0; i < S * M; i++)
        g[i] = ((float)rand() / (float)RAND_MAX) * 2.0f - 1.0f;
}

static float max_error(const float *a, const float *b, int n) {
    float max_err = 0.0f;
    for (int i = 0; i < n; i++) {
        float err = fabsf(a[i] - b[i]);
        if (err > max_err) max_err = err;
    }
    return max_err;
}

typedef struct { int S, K, M; } Shape;

int main(void) {
    printf("============================================\n");
    printf("  Binary Backward: Multi-Shape Parity\n");
    printf("============================================\n\n");
    
    Shape shapes[] = {
        { 4,  256,  256 },  /* small */
        { 8,  512,  512 },  /* medium (dim=128 training) */
        { 16, 1024, 1024 }, /* large (dim=256 training) */
        { 32, 2048, 2048 }, /* xlarge (dim=512 training) */
        { 64,  512, 2048 }, /* asymmetric (FFN) */
    };
    int n_shapes = sizeof(shapes) / sizeof(shapes[0]);
    
    int passed = 0;
    
    for (int i = 0; i < n_shapes; i++) {
        Shape sh = shapes[i];
        printf("Testing S=%d, K=%d, M=%d... ", sh.S, sh.K, sh.M);
        fflush(stdout);
        
        /* Allocate host memory */
        float *h_weights = (float*)malloc(sh.M * sh.K * sizeof(float));
        float *h_input = (float*)malloc(sh.S * sh.K * sizeof(float));
        float *h_dY = (float*)malloc(sh.S * sh.M * sizeof(float));
        float *h_dX_gpu = (float*)calloc(sh.S * sh.K, sizeof(float));
        float *h_dX_ws = (float*)calloc(sh.S * sh.K, sizeof(float));
        float *h_dW_gpu = (float*)calloc(sh.M * sh.K, sizeof(float));
        float *h_dW_ws = (float*)calloc(sh.M * sh.K, sizeof(float));
        
        /* Init */
        srand(42 + i);
        init_weights(h_weights, sh.M, sh.K);
        init_input(h_input, sh.S, sh.K);
        init_grad(h_dY, sh.S, sh.M);
        
        /* Pack weights for GPU */
        BinaryWeightsGPU gpu_weights = binary_pack_weights_gpu(h_weights, sh.M, sh.K);
        
        /* Original GPU backward */
        binary_backward_batch_gpu(h_dY, h_input, h_weights, &gpu_weights,
                                   h_dX_gpu, h_dW_gpu, sh.S, sh.M, sh.K);
        
        /* Workspace setup */
        BinaryGPUWorkspace ws;
        binary_workspace_init(&ws, sh.S, sh.K, sh.M);
        
        /* Workspace backward */
        binary_backward_batch_gpu_ws(h_dY, h_input, h_weights, &gpu_weights,
                                      &ws, h_dX_ws, h_dW_ws, sh.S, sh.M, sh.K);
        
        /* Compare */
        float err_dX = max_error(h_dX_gpu, h_dX_ws, sh.S * sh.K);
        float err_dW = max_error(h_dW_gpu, h_dW_ws, sh.M * sh.K);
        float max_err = (err_dX > err_dW) ? err_dX : err_dW;
        
        if (max_err < 1e-5) {
            printf("PASS (dX=%.2e, dW=%.2e)\n", err_dX, err_dW);
            passed++;
        } else {
            printf("FAIL (dX=%.2e, dW=%.2e)\n", err_dX, err_dW);
        }
        
        /* Cleanup */
        binary_free_weights_gpu(&gpu_weights);
        binary_workspace_free(&ws);
        free(h_weights);
        free(h_input);
        free(h_dY);
        free(h_dX_gpu);
        free(h_dX_ws);
        free(h_dW_gpu);
        free(h_dW_ws);
    }
    
    /* ═══════════════════════════════════════════════════════
     * Multi-backward parity: binary_backward_multi_gpu_ws
     * vs N × binary_backward_batch_gpu_ws
     * ═══════════════════════════════════════════════════════ */

    printf("\n============================================\n");
    printf("  Multi-Backward Parity (shared input)\n");
    printf("============================================\n\n");

    int multi_passed = 0;
    int multi_total = 0;

    /* Case 1: QKV-style — N=3, shared K=256, M=[256, 128, 128] */
    {
        multi_total++;
        int S = 64, K = 256;
        int Ms[3] = { 256, 128, 128 };
        int N = 3;
        printf("QKV-style N=3, S=%d, K=%d, M=[%d,%d,%d]... ", S, K, Ms[0], Ms[1], Ms[2]);
        fflush(stdout);

        srand(1337);
        float *X = (float*)malloc(S * K * sizeof(float));
        init_input(X, S, K);

        float *dY[3], *W_lat[3], *dW_single[3], *dW_multi[3];
        BinaryWeightsGPU Wgpu[3];
        for (int i = 0; i < N; i++) {
            dY[i] = (float*)malloc(S * Ms[i] * sizeof(float));
            W_lat[i] = (float*)malloc(Ms[i] * K * sizeof(float));
            dW_single[i] = (float*)calloc(Ms[i] * K, sizeof(float));
            dW_multi[i] = (float*)calloc(Ms[i] * K, sizeof(float));
            init_grad(dY[i], S, Ms[i]);
            init_weights(W_lat[i], Ms[i], K);
            Wgpu[i] = binary_pack_weights_gpu(W_lat[i], Ms[i], K);
        }

        float *dX_single = (float*)calloc(S * K, sizeof(float));
        float *dX_multi = (float*)calloc(S * K, sizeof(float));

        BinaryGPUWorkspace ws;
        int max_M = 256;
        binary_workspace_init(&ws, S, K, max_M);

        /* Reference: 3 × single backward */
        for (int i = 0; i < N; i++) {
            binary_backward_batch_gpu_ws(dY[i], X, W_lat[i], &Wgpu[i],
                &ws, dX_single, dW_single[i], S, Ms[i], K);
        }

        /* Test: multi backward */
        const float *dY_c[3] = { dY[0], dY[1], dY[2] };
        const float *Wl_c[3] = { W_lat[0], W_lat[1], W_lat[2] };
        const BinaryWeightsGPU *Wg_c[3] = { &Wgpu[0], &Wgpu[1], &Wgpu[2] };
        binary_backward_multi_gpu_ws(dY_c, X, Wl_c, Wg_c,
            &ws, dX_multi, dW_multi, N, S, K);

        float err_dX = max_error(dX_single, dX_multi, S * K);
        float max_dW_err = 0.0f;
        for (int i = 0; i < N; i++) {
            float e = max_error(dW_single[i], dW_multi[i], Ms[i] * K);
            if (e > max_dW_err) max_dW_err = e;
        }
        float max_err = (err_dX > max_dW_err) ? err_dX : max_dW_err;

        if (max_err < 1e-4) {
            printf("PASS (dX=%.2e, dW=%.2e)\n", err_dX, max_dW_err);
            multi_passed++;
        } else {
            printf("FAIL (dX=%.2e, dW=%.2e)\n", err_dX, max_dW_err);
        }

        for (int i = 0; i < N; i++) {
            free(dY[i]); free(W_lat[i]); free(dW_single[i]); free(dW_multi[i]);
            binary_free_weights_gpu(&Wgpu[i]);
        }
        free(X); free(dX_single); free(dX_multi);
        binary_workspace_free(&ws);
    }

    /* Case 2: gate+up style — N=2, shared K=256, M=[1024, 1024] */
    {
        multi_total++;
        int S = 64, K = 256;
        int Ms[2] = { 1024, 1024 };
        int N = 2;
        printf("gate+up style N=2, S=%d, K=%d, M=[%d,%d]... ", S, K, Ms[0], Ms[1]);
        fflush(stdout);

        srand(7777);
        float *X = (float*)malloc(S * K * sizeof(float));
        init_input(X, S, K);

        float *dY[2], *W_lat[2], *dW_single[2], *dW_multi[2];
        BinaryWeightsGPU Wgpu[2];
        for (int i = 0; i < N; i++) {
            dY[i] = (float*)malloc(S * Ms[i] * sizeof(float));
            W_lat[i] = (float*)malloc(Ms[i] * K * sizeof(float));
            dW_single[i] = (float*)calloc(Ms[i] * K, sizeof(float));
            dW_multi[i] = (float*)calloc(Ms[i] * K, sizeof(float));
            init_grad(dY[i], S, Ms[i]);
            init_weights(W_lat[i], Ms[i], K);
            Wgpu[i] = binary_pack_weights_gpu(W_lat[i], Ms[i], K);
        }

        float *dX_single = (float*)calloc(S * K, sizeof(float));
        float *dX_multi = (float*)calloc(S * K, sizeof(float));

        BinaryGPUWorkspace ws;
        binary_workspace_init(&ws, S, K, 1024);

        /* Reference: 2 × single backward */
        for (int i = 0; i < N; i++) {
            binary_backward_batch_gpu_ws(dY[i], X, W_lat[i], &Wgpu[i],
                &ws, dX_single, dW_single[i], S, Ms[i], K);
        }

        /* Test: multi backward */
        const float *dY_c[2] = { dY[0], dY[1] };
        const float *Wl_c[2] = { W_lat[0], W_lat[1] };
        const BinaryWeightsGPU *Wg_c[2] = { &Wgpu[0], &Wgpu[1] };
        binary_backward_multi_gpu_ws(dY_c, X, Wl_c, Wg_c,
            &ws, dX_multi, dW_multi, N, S, K);

        float err_dX = max_error(dX_single, dX_multi, S * K);
        float max_dW_err = 0.0f;
        for (int i = 0; i < N; i++) {
            float e = max_error(dW_single[i], dW_multi[i], Ms[i] * K);
            if (e > max_dW_err) max_dW_err = e;
        }
        float max_err = (err_dX > max_dW_err) ? err_dX : max_dW_err;

        if (max_err < 1e-4) {
            printf("PASS (dX=%.2e, dW=%.2e)\n", err_dX, max_dW_err);
            multi_passed++;
        } else {
            printf("FAIL (dX=%.2e, dW=%.2e)\n", err_dX, max_dW_err);
        }

        for (int i = 0; i < N; i++) {
            free(dY[i]); free(W_lat[i]); free(dW_single[i]); free(dW_multi[i]);
            binary_free_weights_gpu(&Wgpu[i]);
        }
        free(X); free(dX_single); free(dX_multi);
        binary_workspace_free(&ws);
    }

    int total_passed = passed + multi_passed;
    int total_tests = n_shapes + multi_total;

    printf("\n============================================\n");
    printf("  Results: %d/%d tests passed\n", total_passed, total_tests);
    printf("    Single backward: %d/%d\n", passed, n_shapes);
    printf("    Multi backward:  %d/%d\n", multi_passed, multi_total);
    printf("============================================\n");

    return (total_passed == total_tests) ? 0 : 1;
}
