/*
 * test_rope.cu — Verify RoPE for {2,3} architecture
 *
 * Tests:
 * 1. Rotation preserves vector magnitude
 * 2. Different positions give different outputs
 * 3. Position 0 = identity (cos=1, sin=0)
 * 4. GPU vs CPU match
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include "rope.h"

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        printf("CUDA error: %s\n", cudaGetErrorString(err)); exit(1); \
    } \
} while(0)

static int test_magnitude_preserved(void) {
    printf("--- Test: rotation preserves magnitude ---\n");
    int head_dim = 128, n_heads = 16, n_kv = 8;

    RoPETable r;
    rope_init(&r, head_dim, 2048, 1000000.0f);

    float *q = (float*)malloc(n_heads * head_dim * sizeof(float));
    float *k = (float*)malloc(n_kv * head_dim * sizeof(float));
    srand(42);
    for (int i = 0; i < n_heads * head_dim; i++)
        q[i] = 2.0f * (float)rand() / RAND_MAX - 1.0f;
    for (int i = 0; i < n_kv * head_dim; i++)
        k[i] = 2.0f * (float)rand() / RAND_MAX - 1.0f;

    /* Compute magnitude before */
    double mag_before = 0;
    for (int i = 0; i < n_heads * head_dim; i++)
        mag_before += (double)q[i] * q[i];
    mag_before = sqrt(mag_before);

    rope_apply_cpu(q, k, &r, 42, n_heads, n_kv);

    double mag_after = 0;
    for (int i = 0; i < n_heads * head_dim; i++)
        mag_after += (double)q[i] * q[i];
    mag_after = sqrt(mag_after);

    double err = fabs(mag_before - mag_after) / mag_before;
    printf("  |q| before: %.6f, after: %.6f, rel_err: %.2e\n",
           mag_before, mag_after, err);

    int pass = (err < 1e-5);
    printf("  result: %s\n\n", pass ? "PASS" : "FAIL");

    free(q); free(k);
    rope_free(&r);
    return pass;
}

static int test_position_changes_output(void) {
    printf("--- Test: different positions → different outputs ---\n");
    int head_dim = 64, n_heads = 1, n_kv = 1;

    RoPETable r;
    rope_init(&r, head_dim, 2048, 1000000.0f);

    float *q0 = (float*)calloc(head_dim, sizeof(float));
    float *q1 = (float*)calloc(head_dim, sizeof(float));
    float *k_dummy = (float*)calloc(head_dim, sizeof(float));

    /* Same input vector */
    for (int i = 0; i < head_dim; i++)
        q0[i] = q1[i] = 1.0f;

    rope_apply_cpu(q0, k_dummy, &r, 0, n_heads, n_kv);
    rope_apply_cpu(q1, k_dummy, &r, 100, n_heads, n_kv);

    double diff = 0;
    for (int i = 0; i < head_dim; i++)
        diff += fabs((double)q0[i] - (double)q1[i]);

    printf("  sum|q(pos=0) - q(pos=100)|: %.4f\n", diff);

    int pass = (diff > 0.01);  /* should be different */
    printf("  result: %s\n\n", pass ? "PASS" : "FAIL");

    free(q0); free(q1); free(k_dummy);
    rope_free(&r);
    return pass;
}

static int test_position_zero_identity(void) {
    printf("--- Test: position 0 = near identity ---\n");
    int head_dim = 64, n_heads = 1, n_kv = 1;

    RoPETable r;
    rope_init(&r, head_dim, 2048, 1000000.0f);

    float *q = (float*)malloc(head_dim * sizeof(float));
    float *q_orig = (float*)malloc(head_dim * sizeof(float));
    float *k_dummy = (float*)calloc(head_dim, sizeof(float));

    srand(42);
    for (int i = 0; i < head_dim; i++)
        q[i] = q_orig[i] = 2.0f * (float)rand() / RAND_MAX - 1.0f;

    rope_apply_cpu(q, k_dummy, &r, 0, n_heads, n_kv);

    double max_err = 0;
    for (int i = 0; i < head_dim; i++) {
        double err = fabs((double)q[i] - (double)q_orig[i]);
        if (err > max_err) max_err = err;
    }

    printf("  max|q(pos=0) - q_orig|: %.2e (cos(0)=1, sin(0)=0 → identity)\n", max_err);

    int pass = (max_err < 1e-6);
    printf("  result: %s\n\n", pass ? "PASS" : "FAIL");

    free(q); free(q_orig); free(k_dummy);
    rope_free(&r);
    return pass;
}

static int test_gpu_cpu_match(void) {
    printf("--- Test: GPU vs CPU match ---\n");
    int head_dim = 128, n_heads = 16, n_kv = 8, pos = 42;

    RoPETable r;
    rope_init(&r, head_dim, 2048, 1000000.0f);

    int q_size = n_heads * head_dim;
    int k_size = n_kv * head_dim;
    int half = head_dim / 2;

    float *q_cpu = (float*)malloc(q_size * sizeof(float));
    float *k_cpu = (float*)malloc(k_size * sizeof(float));
    srand(99);
    for (int i = 0; i < q_size; i++)
        q_cpu[i] = 2.0f * (float)rand() / RAND_MAX - 1.0f;
    for (int i = 0; i < k_size; i++)
        k_cpu[i] = 2.0f * (float)rand() / RAND_MAX - 1.0f;

    /* Copy for GPU path */
    float *q_gpu_h = (float*)malloc(q_size * sizeof(float));
    float *k_gpu_h = (float*)malloc(k_size * sizeof(float));
    memcpy(q_gpu_h, q_cpu, q_size * sizeof(float));
    memcpy(k_gpu_h, k_cpu, k_size * sizeof(float));

    /* CPU */
    rope_apply_cpu(q_cpu, k_cpu, &r, pos, n_heads, n_kv);

    /* GPU */
    float *d_q, *d_k, *d_cos, *d_sin;
    CUDA_CHECK(cudaMalloc(&d_q, q_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_k, k_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_cos, half * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_sin, half * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_q, q_gpu_h, q_size * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_k, k_gpu_h, k_size * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_cos, r.cos_table + pos * half, half * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_sin, r.sin_table + pos * half, half * sizeof(float), cudaMemcpyHostToDevice));

    int max_heads = n_heads > n_kv ? n_heads : n_kv;
    kernel_rope_apply<<<max_heads, half>>>(
        d_q, d_k, d_cos, d_sin, n_heads, n_kv, head_dim);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(q_gpu_h, d_q, q_size * sizeof(float), cudaMemcpyDeviceToHost));

    double max_err = 0;
    for (int i = 0; i < q_size; i++) {
        double err = fabs((double)q_gpu_h[i] - (double)q_cpu[i]);
        if (err > max_err) max_err = err;
    }

    printf("  max error (GPU vs CPU): %.2e\n", max_err);
    printf("  q_cpu[0..3]: %.6f %.6f %.6f %.6f\n",
           q_cpu[0], q_cpu[1], q_cpu[2], q_cpu[3]);
    printf("  q_gpu[0..3]: %.6f %.6f %.6f %.6f\n",
           q_gpu_h[0], q_gpu_h[1], q_gpu_h[2], q_gpu_h[3]);

    int pass = (max_err < 1e-5);
    printf("  result: %s\n\n", pass ? "PASS" : "FAIL");

    cudaFree(d_q); cudaFree(d_k); cudaFree(d_cos); cudaFree(d_sin);
    free(q_cpu); free(k_cpu); free(q_gpu_h); free(k_gpu_h);
    rope_free(&r);
    return pass;
}

int main(void) {
    printf("============================================\n");
    printf("  RoPE — Rotary Position Embedding\n");
    printf("  O(dim) float on top of O(dim²) integer.\n");
    printf("============================================\n\n");

    int passed = 0, total = 0;
    total++; passed += test_magnitude_preserved();
    total++; passed += test_position_changes_output();
    total++; passed += test_position_zero_identity();
    total++; passed += test_gpu_cpu_match();

    printf("============================================\n");
    printf("  Results: %d / %d tests passed\n", passed, total);
    printf("============================================\n");

    if (passed == total)
        printf("\n  RoPE verified. Position encoded.\n\n");

    return (passed == total) ? 0 : 1;
}
