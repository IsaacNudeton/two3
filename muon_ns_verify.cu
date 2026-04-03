/*
 * muon_ns_verify.cu — CPU vs GPU Newton-Schulz parity check
 *
 * Keeps the CPU reference loop in sync with train.h:newton_schulz_orthogonalize.
 * Build: build_driver.bat verify-moon   (passes -DTWO3_MUON_GPU for this file + two3.cu)
 * Run:   muon_ns_verify.exe
 */

#include "two3.h"

#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CUDA_OK(call) do { \
    cudaError_t e = (call); \
    if (e != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
        exit(1); \
    } \
} while (0)

/* Mirror of train.h newton_schulz_orthogonalize (must stay aligned). */
static void newton_schulz_cpu(float *G, int rows, int cols, float *scratch) {
    const float a = 3.4445f, b = -4.7750f, c = 2.0315f;

    float norm_sq = 0;
    for (int i = 0; i < rows * cols; i++)
        norm_sq += G[i] * G[i];
    float norm = sqrtf(norm_sq + 1e-30f);
    for (int i = 0; i < rows * cols; i++)
        G[i] /= norm;

    float *XtX = scratch;

    for (int iter = 0; iter < 5; iter++) {
        memset(XtX, 0, (size_t)cols * cols * sizeof(float));
        for (int i = 0; i < cols; i++)
            for (int j = 0; j < cols; j++)
                for (int k = 0; k < rows; k++)
                    XtX[i * cols + j] += G[k * cols + i] * G[k * cols + j];

        float *G_new = (float *)malloc((size_t)rows * cols * sizeof(float));
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                float x_xtx = 0;
                for (int k = 0; k < cols; k++)
                    x_xtx += G[i * cols + k] * XtX[k * cols + j];

                float x_xtx_xtx = 0;
                for (int k = 0; k < cols; k++) {
                    float xtx_xtx_kj = 0;
                    for (int m = 0; m < cols; m++)
                        xtx_xtx_kj += XtX[k * cols + m] * XtX[m * cols + j];
                    x_xtx_xtx += G[i * cols + k] * xtx_xtx_kj;
                }

                G_new[i * cols + j] = a * G[i * cols + j] + b * x_xtx + c * x_xtx_xtx;
            }
        }
        memcpy(G, G_new, (size_t)rows * cols * sizeof(float));
        free(G_new);
    }
}

static float max_abs_diff(const float *a, const float *b, int n) {
    float m = 0.f;
    for (int i = 0; i < n; i++) {
        float d = fabsf(a[i] - b[i]);
        if (d > m) m = d;
    }
    return m;
}

static int run_case(int rows, int cols, unsigned seed) {
    int n = rows * cols;
    float *orig = (float *)malloc((size_t)n * sizeof(float));
    float *h_cpu = (float *)malloc((size_t)n * sizeof(float));
    float *scratch = (float *)calloc((size_t)cols * cols, sizeof(float));
    srand(seed);
    for (int i = 0; i < n; i++)
        orig[i] = (float)rand() / (float)RAND_MAX * 2.f - 1.f;

    memcpy(h_cpu, orig, (size_t)n * sizeof(float));
    newton_schulz_cpu(h_cpu, rows, cols, scratch);

    int capM = rows > cols ? rows : cols;
    Two3BackwardCtx ctx = two3_backward_ctx_init(capM, capM, 0, 0, 0, 0);
    two3_muon_gpu_init(&ctx, capM, capM);

    float *d_G = NULL;
    CUDA_OK(cudaMalloc(&d_G, (size_t)n * sizeof(float)));
    CUDA_OK(cudaMemcpy(d_G, orig, (size_t)n * sizeof(float), cudaMemcpyHostToDevice));
    two3_newton_schulz_gpu(&ctx, d_G, rows, cols);
    float *h_gpu = (float *)malloc((size_t)n * sizeof(float));
    CUDA_OK(cudaMemcpy(h_gpu, d_G, (size_t)n * sizeof(float), cudaMemcpyDeviceToHost));
    cudaFree(d_G);

    two3_muon_gpu_free(&ctx);
    two3_backward_ctx_free(&ctx);

    float md = max_abs_diff(h_cpu, h_gpu, n);
    printf("  [%d x %d]  max|CPU-GPU| = %.4e", rows, cols, md);
    /* cuBLAS SGEMM vs nested CPU loops: allow a few ulp in the thousands of ops */
    int ok = md < 2e-3f;
    if (ok)
        printf("  OK\n");
    else
        printf("  FAIL (threshold 2e-3)\n");

    free(orig);
    free(h_cpu);
    free(h_gpu);
    free(scratch);
    return ok ? 0 : 1;
}

int main(void) {
    printf("Muon Newton-Schulz CPU vs GPU parity\n");
    int err = 0;
    err |= run_case(32, 32, 1);
    err |= run_case(64, 128, 2);   /* cols < rows — was cuBLAS lda bug */
    err |= run_case(128, 64, 3);   /* rows < cols */
    err |= run_case(17, 55, 4);    /* odd sizes */
    if (err)
        printf("\nPARITY FAIL\n");
    else
        printf("\nPARITY OK\n");
    return err;
}
