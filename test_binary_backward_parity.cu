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
    
    printf("\n============================================\n");
    printf("  Results: %d/%d shapes passed\n", passed, n_shapes);
    printf("============================================\n");
    
    return (passed == n_shapes) ? 0 : 1;
}
