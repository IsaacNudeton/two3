/*
 * test_two3_tensor_core.cu — WMMA vs bitmask matmul parity
 *
 * Builds with -DTWO3_TENSOR_CORE to enable the WMMA kernel,
 * directly calls both paths on the same inputs, compares int32 outputs.
 *
 * Shapes chosen to be 16-aligned so WMMA eligibility passes.
 * Expected: bit-exact (0 error) since both kernels compute the same
 * integer sum-of-products with int32 accumulators on the same silicon.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <cuda_runtime.h>
#include "binary_gpu.h"

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

static void init_acts(int8_t *x, int S, int K) {
    for (int i = 0; i < S * K; i++)
        x[i] = (int8_t)((rand() % 255) - 127);
}

typedef struct { int S, K, M; } Shape;

int main(void) {
#ifndef TWO3_TENSOR_CORE
    printf("[skip] built without TWO3_TENSOR_CORE\n");
    return 0;
#else
    printf("============================================\n");
    printf("  WMMA vs Bitmask matmul parity\n");
    printf("============================================\n\n");

    Shape shapes[] = {
        { 16,  256,  256 },   /* smallest 16-aligned */
        { 32,  256,  256 },
        { 16,  512,  128 },
        { 32, 1024,  512 },
        { 64,  256, 1024 },
        { 64,  512, 2048 },   /* FFN-shape-like */
        { 64, 1024, 2048 },
    };
    int n_shapes = sizeof(shapes) / sizeof(shapes[0]);

    int passed = 0;
    for (int i = 0; i < n_shapes; i++) {
        Shape sh = shapes[i];
        printf("Testing S=%d K=%d M=%d ... ", sh.S, sh.K, sh.M);
        fflush(stdout);

        srand(42 + i);
        float *h_weights = (float*)malloc(sh.M * sh.K * sizeof(float));
        int8_t *h_x_q   = (int8_t*)malloc(sh.S * sh.K * sizeof(int8_t));
        init_weights(h_weights, sh.M, sh.K);
        init_acts(h_x_q, sh.S, sh.K);

        BinaryWeightsGPU W = binary_pack_weights_gpu(h_weights, sh.M, sh.K);

        int8_t  *d_x_q;
        int32_t *d_acc_bitmask;
        int32_t *d_acc_wmma;
        CHECK_CUDA(cudaMalloc(&d_x_q,         (size_t)sh.S * sh.K * sizeof(int8_t)));
        CHECK_CUDA(cudaMalloc(&d_acc_bitmask, (size_t)sh.S * sh.M * sizeof(int32_t)));
        CHECK_CUDA(cudaMalloc(&d_acc_wmma,    (size_t)sh.S * sh.M * sizeof(int32_t)));
        CHECK_CUDA(cudaMemcpy(d_x_q, h_x_q, (size_t)sh.S * sh.K * sizeof(int8_t),
                              cudaMemcpyHostToDevice));

        /* Bitmask path */
        CHECK_CUDA(cudaMemset(d_acc_bitmask, 0, (size_t)sh.S * sh.M * sizeof(int32_t)));
        binary_matmul_gpu(W.d_packed_plus, W.d_packed_neg, d_x_q, d_acc_bitmask,
                          sh.S, sh.M, sh.K);
        CHECK_CUDA(cudaDeviceSynchronize());

        /* WMMA path — force it */
        CHECK_CUDA(cudaMemset(d_acc_wmma, 0, (size_t)sh.S * sh.M * sizeof(int32_t)));
        bool used_wmma = binary_matmul_gpu_wmma(W.d_W_tc, d_x_q, d_acc_wmma,
                                                 sh.S, sh.M, sh.K);
        CHECK_CUDA(cudaDeviceSynchronize());

        if (!used_wmma) {
            printf("SKIP (WMMA ineligible for this shape)\n");
            goto cleanup;
        }

        /* Compare int32 outputs */
        int32_t *h_bitmask = (int32_t*)malloc((size_t)sh.S * sh.M * sizeof(int32_t));
        int32_t *h_wmma    = (int32_t*)malloc((size_t)sh.S * sh.M * sizeof(int32_t));
        CHECK_CUDA(cudaMemcpy(h_bitmask, d_acc_bitmask,
                              (size_t)sh.S * sh.M * sizeof(int32_t), cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(h_wmma, d_acc_wmma,
                              (size_t)sh.S * sh.M * sizeof(int32_t), cudaMemcpyDeviceToHost));

        int n_diff = 0;
        int max_abs_diff = 0;
        int first_idx = -1;
        int32_t first_b = 0, first_w = 0;
        for (int j = 0; j < sh.S * sh.M; j++) {
            int32_t diff = h_bitmask[j] - h_wmma[j];
            if (diff) {
                if (first_idx < 0) { first_idx = j; first_b = h_bitmask[j]; first_w = h_wmma[j]; }
                n_diff++;
                int abs_diff = diff < 0 ? -diff : diff;
                if (abs_diff > max_abs_diff) max_abs_diff = abs_diff;
            }
        }

        if (n_diff == 0) {
            printf("PASS (bit-exact, WMMA used)\n");
            passed++;
        } else {
            printf("FAIL (%d/%d differ, max |diff|=%d, first@%d: bitmask=%d wmma=%d)\n",
                   n_diff, sh.S * sh.M, max_abs_diff, first_idx, first_b, first_w);
        }

        free(h_bitmask); free(h_wmma);

    cleanup:
        cudaFree(d_x_q); cudaFree(d_acc_bitmask); cudaFree(d_acc_wmma);
        binary_free_weights_gpu(&W);
        free(h_weights); free(h_x_q);
    }

    printf("\n============================================\n");
    printf("  Results: %d/%d shapes bit-exact\n", passed, n_shapes);
    printf("============================================\n");
    return (passed == n_shapes) ? 0 : 1;
#endif
}
