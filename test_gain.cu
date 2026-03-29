/*
 * test_gain.cu — Verify the gain kernel normalization
 *
 * Tests:
 * 1. Fixed point convergence: constant input → R converges to R*
 * 2. Normalization: large input → output bounded (like RMSNorm)
 * 3. Amplification: small input → output boosted (unlike RMSNorm)
 * 4. CFL margin verification
 * 5. GPU vs CPU reference match
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include "gain.h"

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        printf("CUDA error: %s\n", cudaGetErrorString(err)); exit(1); \
    } \
} while(0)

static int test_fixed_point(void) {
    printf("--- Test: fixed point convergence ---\n");
    int dim = 64;

    /* Constant input → reservoir should converge to R* = β/α */
    float *R = (float*)malloc(dim * sizeof(float));
    float *C = (float*)malloc(dim * sizeof(float));
    float *x = (float*)malloc(dim * sizeof(float));
    float *y = (float*)malloc(dim * sizeof(float));

    for (int i = 0; i < dim; i++) {
        C[i] = 1.0f;
        R[i] = C[i];  /* start at capacity */
        x[i] = 0.5f;  /* constant input */
    }

    /* Run 1000 ticks with constant input */
    for (int t = 0; t < 1000; t++)
        gain_forward_cpu(y, x, R, C, dim);

    float R_star = gain_R_star();
    float err = fabsf(R[0] - R_star);
    printf("  R* = %.6f (expected β/α = %.6f)\n", R[0], R_star);
    printf("  error: %.2e\n", err);

    int pass = (err < 0.01f);
    printf("  result: %s\n\n", pass ? "PASS" : "FAIL");

    free(R); free(C); free(x); free(y);
    return pass;
}

static int test_normalization(void) {
    printf("--- Test: normalization (prevents explosion) ---\n");
    int dim = 256;

    float *R = (float*)malloc(dim * sizeof(float));
    float *C = (float*)malloc(dim * sizeof(float));
    float *x = (float*)malloc(dim * sizeof(float));
    float *y = (float*)malloc(dim * sizeof(float));

    for (int i = 0; i < dim; i++) {
        C[i] = 1.0f;
        R[i] = C[i];
    }

    /* Feed increasingly large inputs — output should stay bounded */
    float max_out = 0;
    for (int t = 0; t < 100; t++) {
        float scale = 1.0f + (float)t * 0.5f;  /* growing input */
        for (int i = 0; i < dim; i++)
            x[i] = scale * ((float)(i % 7) / 3.0f - 1.0f);

        gain_forward_cpu(y, x, R, C, dim);

        for (int i = 0; i < dim; i++)
            if (fabsf(y[i]) > max_out) max_out = fabsf(y[i]);
    }

    /* After depletion, output should be suppressed relative to input */
    float input_max = 50.5f * 2.0f;  /* max input at t=100 */
    float ratio = max_out / input_max;
    printf("  max output: %.2f (input max: %.2f, ratio: %.2f)\n",
           max_out, input_max, ratio);
    printf("  reservoir R[0]: %.6f (depleted from %.6f)\n", R[0], C[0]);

    int pass = (max_out < input_max * 2.0f);  /* output shouldn't double the input */
    printf("  result: %s\n\n", pass ? "PASS" : "FAIL");

    free(R); free(C); free(x); free(y);
    return pass;
}

static int test_amplification(void) {
    printf("--- Test: amplification (boosts weak signals) ---\n");
    int dim = 64;

    float *R = (float*)malloc(dim * sizeof(float));
    float *C = (float*)malloc(dim * sizeof(float));
    float *x = (float*)malloc(dim * sizeof(float));
    float *y = (float*)malloc(dim * sizeof(float));

    for (int i = 0; i < dim; i++) {
        C[i] = 1.0f;
        R[i] = C[i];  /* full reservoir */
    }

    /* Tiny input with full reservoir → should be amplified */
    for (int i = 0; i < dim; i++)
        x[i] = 0.001f;

    gain_forward_cpu(y, x, R, C, dim);

    /* With full reservoir and α*R ≈ α*C = 0.05, gain ≈ 1.04
     * Small but positive amplification */
    float gain = y[0] / x[0];
    printf("  input: %.6f, output: %.6f, gain: %.4f\n", x[0], y[0], gain);
    printf("  expected gain ≈ 1 + α*C - β = %.4f\n",
           1.0f + GAIN_ALPHA * C[0] - GAIN_BETA);

    int pass = (gain > 1.0f);  /* amplification: gain > 1 */
    printf("  result: %s\n\n", pass ? "PASS" : "FAIL");

    free(R); free(C); free(x); free(y);
    return pass;
}

static int test_cfl(void) {
    printf("--- Test: CFL margin ---\n");
    int margin = gain_cfl_check(5.0f);  /* C_max = 5.0 */
    printf("  CFL margin: %dx (need > 1x)\n", margin);

    int pass = (margin > 1);
    printf("  result: %s\n\n", pass ? "PASS" : "FAIL");
    return pass;
}

static int test_gpu_cpu_match(void) {
    printf("--- Test: GPU vs CPU match ---\n");
    int dim = 512;

    /* CPU path */
    float *R_cpu = (float*)malloc(dim * sizeof(float));
    float *C_cpu = (float*)malloc(dim * sizeof(float));
    float *x_cpu = (float*)malloc(dim * sizeof(float));
    float *y_cpu = (float*)malloc(dim * sizeof(float));

    srand(42);
    for (int i = 0; i < dim; i++) {
        C_cpu[i] = 0.5f + (float)rand() / RAND_MAX;
        R_cpu[i] = C_cpu[i];
        x_cpu[i] = 2.0f * (float)rand() / RAND_MAX - 1.0f;
    }

    gain_forward_cpu(y_cpu, x_cpu, R_cpu, C_cpu, dim);

    /* GPU path */
    float *d_R, *d_C, *d_x, *d_y;
    CUDA_CHECK(cudaMalloc(&d_R, dim * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_C, dim * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_x, dim * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_y, dim * sizeof(float)));

    /* Reset R to match CPU initial state */
    srand(42);
    float *R_init = (float*)malloc(dim * sizeof(float));
    float *C_init = (float*)malloc(dim * sizeof(float));
    float *x_init = (float*)malloc(dim * sizeof(float));
    for (int i = 0; i < dim; i++) {
        C_init[i] = 0.5f + (float)rand() / RAND_MAX;
        R_init[i] = C_init[i];
        x_init[i] = 2.0f * (float)rand() / RAND_MAX - 1.0f;
    }

    CUDA_CHECK(cudaMemcpy(d_R, R_init, dim * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_C, C_init, dim * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_x, x_init, dim * sizeof(float), cudaMemcpyHostToDevice));

    kernel_gain_forward<<<(dim+255)/256, 256>>>(
        d_y, d_x, d_R, d_C, dim,
        GAIN_ALPHA, GAIN_BETA, GAIN_GAMMA, GAIN_KAPPA);
    CUDA_CHECK(cudaDeviceSynchronize());

    float *y_gpu = (float*)malloc(dim * sizeof(float));
    CUDA_CHECK(cudaMemcpy(y_gpu, d_y, dim * sizeof(float), cudaMemcpyDeviceToHost));

    /* Compare */
    double max_err = 0;
    for (int i = 0; i < dim; i++) {
        double err = fabs((double)y_gpu[i] - (double)y_cpu[i]);
        if (err > max_err) max_err = err;
    }

    printf("  max error (GPU vs CPU): %.2e\n", max_err);
    printf("  y_cpu[0..3]: %.6f %.6f %.6f %.6f\n",
           y_cpu[0], y_cpu[1], y_cpu[2], y_cpu[3]);
    printf("  y_gpu[0..3]: %.6f %.6f %.6f %.6f\n",
           y_gpu[0], y_gpu[1], y_gpu[2], y_gpu[3]);

    int pass = (max_err < 1e-5);
    printf("  result: %s\n\n", pass ? "PASS" : "FAIL");

    cudaFree(d_R); cudaFree(d_C); cudaFree(d_x); cudaFree(d_y);
    free(R_cpu); free(C_cpu); free(x_cpu); free(y_cpu);
    free(R_init); free(C_init); free(x_init); free(y_gpu);
    return pass;
}

int main(void) {
    printf("============================================\n");
    printf("  Gain Kernel Normalization — Test Suite\n");
    printf("  Reservoir + amplitude. Not RMSNorm.\n");
    printf("  Lean-verified stability.\n");
    printf("============================================\n\n");

    int passed = 0, total = 0;

    total++; passed += test_fixed_point();
    total++; passed += test_normalization();
    total++; passed += test_amplification();
    total++; passed += test_cfl();
    total++; passed += test_gpu_cpu_match();

    printf("============================================\n");
    printf("  Results: %d / %d tests passed\n", passed, total);
    printf("============================================\n");

    if (passed == total)
        printf("\n  Gain kernel verified.\n"
               "  Normalizes. Amplifies. Learns. Stable.\n"
               "  The reservoir IS the normalization.\n\n");

    return (passed == total) ? 0 : 1;
}
