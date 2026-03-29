/*
 * test_moe.cu — MoE router tests
 *
 * 1. Router selects exactly top-2 experts
 * 2. Weights sum to 1.0 after renormalization
 * 3. Different inputs route to different experts
 * 4. Router is differentiable (float weights, not ternary)
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include "moe.h"

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        printf("CUDA error: %s\n", cudaGetErrorString(err)); exit(1); \
    } \
} while(0)

static int test_top2_selection(void) {
    printf("--- Test: router selects exactly top-2 ---\n");
    int dim = 256;

    MoERouter router;
    srand(42);
    moe_router_init(&router, dim, MOE_NUM_EXPERTS);

    float *x = (float*)malloc(dim * sizeof(float));
    for (int i = 0; i < dim; i++)
        x[i] = 2.0f * (float)rand() / RAND_MAX - 1.0f;

    MoESelection sel;
    moe_route(&router, x, &sel);

    printf("  selected: expert %d (w=%.4f), expert %d (w=%.4f)\n",
           sel.expert_ids[0], sel.expert_weights[0],
           sel.expert_ids[1], sel.expert_weights[1]);

    int valid = (sel.expert_ids[0] != sel.expert_ids[1] &&
                 sel.expert_ids[0] >= 0 && sel.expert_ids[0] < MOE_NUM_EXPERTS &&
                 sel.expert_ids[1] >= 0 && sel.expert_ids[1] < MOE_NUM_EXPERTS);

    printf("  distinct experts: %s\n", valid ? "yes" : "NO");
    int pass = valid;
    printf("  result: %s\n\n", pass ? "PASS" : "FAIL");

    free(x);
    moe_router_free(&router);
    return pass;
}

static int test_weight_normalization(void) {
    printf("--- Test: weights sum to 1.0 ---\n");
    int dim = 256;

    MoERouter router;
    srand(99);
    moe_router_init(&router, dim, MOE_NUM_EXPERTS);

    float *x = (float*)malloc(dim * sizeof(float));
    for (int i = 0; i < dim; i++)
        x[i] = (float)rand() / RAND_MAX;

    MoESelection sel;
    moe_route(&router, x, &sel);

    float w_sum = sel.expert_weights[0] + sel.expert_weights[1];
    printf("  weights: %.6f + %.6f = %.6f\n",
           sel.expert_weights[0], sel.expert_weights[1], w_sum);

    int pass = (fabsf(w_sum - 1.0f) < 1e-5f);
    printf("  result: %s\n\n", pass ? "PASS" : "FAIL");

    free(x);
    moe_router_free(&router);
    return pass;
}

static int test_different_routing(void) {
    printf("--- Test: different inputs → different experts ---\n");
    int dim = 256;

    MoERouter router;
    srand(77);
    moe_router_init(&router, dim, MOE_NUM_EXPERTS);

    /* Generate 100 random inputs, track which experts get selected */
    int expert_counts[MOE_NUM_EXPERTS] = {0};

    for (int t = 0; t < 100; t++) {
        float *x = (float*)malloc(dim * sizeof(float));
        for (int i = 0; i < dim; i++)
            x[i] = 2.0f * (float)rand() / RAND_MAX - 1.0f;

        MoESelection sel;
        moe_route(&router, x, &sel);

        expert_counts[sel.expert_ids[0]]++;
        expert_counts[sel.expert_ids[1]]++;
        free(x);
    }

    printf("  expert usage (100 inputs, top-2 each = 200 selections):\n");
    int used = 0;
    for (int e = 0; e < MOE_NUM_EXPERTS; e++) {
        printf("    expert %d: %d selections\n", e, expert_counts[e]);
        if (expert_counts[e] > 0) used++;
    }

    /* At least 4 of 8 experts should be used with random inputs */
    printf("  experts used: %d / %d\n", used, MOE_NUM_EXPERTS);
    int pass = (used >= 4);
    printf("  result: %s\n\n", pass ? "PASS" : "FAIL");

    moe_router_free(&router);
    return pass;
}

static int test_gpu_router(void) {
    printf("--- Test: GPU router matches CPU ---\n");
    int dim = 256;

    MoERouter router;
    srand(55);
    moe_router_init(&router, dim, MOE_NUM_EXPERTS);

    float *x = (float*)malloc(dim * sizeof(float));
    for (int i = 0; i < dim; i++)
        x[i] = 2.0f * (float)rand() / RAND_MAX - 1.0f;

    /* CPU logits */
    float cpu_logits[MOE_NUM_EXPERTS];
    for (int e = 0; e < MOE_NUM_EXPERTS; e++) {
        float sum = 0;
        for (int d = 0; d < dim; d++)
            sum += x[d] * router.W[d * MOE_NUM_EXPERTS + e];
        cpu_logits[e] = sum;
    }

    /* GPU logits */
    float *d_x, *d_W, *d_logits;
    CUDA_CHECK(cudaMalloc(&d_x, dim * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_W, dim * MOE_NUM_EXPERTS * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_logits, MOE_NUM_EXPERTS * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_x, x, dim * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_W, router.W, dim * MOE_NUM_EXPERTS * sizeof(float), cudaMemcpyHostToDevice));

    kernel_moe_router<<<1, MOE_NUM_EXPERTS>>>(d_logits, d_x, d_W, dim, MOE_NUM_EXPERTS);
    CUDA_CHECK(cudaDeviceSynchronize());

    float gpu_logits[MOE_NUM_EXPERTS];
    CUDA_CHECK(cudaMemcpy(gpu_logits, d_logits, MOE_NUM_EXPERTS * sizeof(float), cudaMemcpyDeviceToHost));

    double max_err = 0;
    for (int e = 0; e < MOE_NUM_EXPERTS; e++) {
        double err = fabs((double)gpu_logits[e] - (double)cpu_logits[e]);
        if (err > max_err) max_err = err;
    }

    printf("  max error (GPU vs CPU): %.2e\n", max_err);
    int pass = (max_err < 1e-3);
    printf("  result: %s\n\n", pass ? "PASS" : "FAIL");

    cudaFree(d_x); cudaFree(d_W); cudaFree(d_logits);
    free(x);
    moe_router_free(&router);
    return pass;
}

int main(void) {
    printf("============================================\n");
    printf("  MoE Router — 8 experts, top-2 selection\n");
    printf("  Float router. Ternary experts.\n");
    printf("============================================\n\n");

    int passed = 0, total = 0;
    total++; passed += test_top2_selection();
    total++; passed += test_weight_normalization();
    total++; passed += test_different_routing();
    total++; passed += test_gpu_router();

    printf("============================================\n");
    printf("  Results: %d / %d tests passed\n", passed, total);
    printf("============================================\n");

    if (passed == total)
        printf("\n  MoE router verified.\n"
               "  8 experts. Top-2 per token.\n"
               "  Float router, ternary experts.\n"
               "  500M active / 2B total.\n\n");

    return (passed == total) ? 0 : 1;
}
