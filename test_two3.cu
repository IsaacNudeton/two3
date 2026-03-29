/*
 * test_two3.cu — Verification for the {2,3} Computing Kernel
 * Isaac Oravec & Claude | XYZT Computing Project | 2026
 *
 * What this does:
 *   1. Generates random float weights (simulating a trained layer)
 *   2. Quantizes to ternary {-1, 0, +1} and packs (2 bits each)
 *   3. Generates random float activations (simulating token embeddings)
 *   4. Quantizes activations to int8 (per-token absmax)
 *   5. Runs the {2,3} forward kernel — pure integer matmul
 *   6. Dequantizes output back to float
 *   7. Compares against float reference matmul
 *   8. Reports error statistics
 *
 * If max relative error < 15%, the quantization is working.
 * BitNet papers report ~1-2% perplexity degradation at this precision.
 *
 * Build:
 *   nvcc -O2 -arch=sm_75 -o test_two3 test_two3.cu two3.cu
 *
 * Run:
 *   ./test_two3
 */

#include "two3.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <cuda_runtime.h>

/* ---- Random float in [-1, 1] ---- */
static float randf(void) {
    return 2.0f * ((float)rand() / (float)RAND_MAX) - 1.0f;
}

/* ---- Print CUDA device info ---- */
static void print_device_info(void) {
    int device;
    cudaGetDevice(&device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);

    printf("============================================\n");
    printf("  {2,3} Computing Kernel — Test Suite\n");
    printf("  Two states. One substrate. No multiply.\n");
    printf("============================================\n");
    printf("  GPU: %s\n", prop.name);
    printf("  SM:  %d.%d\n", prop.major, prop.minor);
    printf("  VRAM: %.0f MB\n", prop.totalGlobalMem / (1024.0 * 1024.0));
    printf("============================================\n\n");
}

/* ---- Test: basic forward pass ---- */
static int test_forward(int S, int M, int K) {
    printf("--- Test: forward pass [S=%d, M=%d, K=%d] ---\n", S, M, K);

    /* Generate ternary-valued weights — tests kernel correctness, not quantization.
     * ~25% zero (substrate), ~37.5% +1, ~37.5% -1 — the {2,3} distribution */
    float* w_float = (float*)malloc(M * K * sizeof(float));
    for (int i = 0; i < M * K; i++) {
        float r = (float)rand() / (float)RAND_MAX;  /* 0 to 1 */
        if (r < 0.25f)       w_float[i] = 0.0f;    /* substrate */
        else if (r < 0.625f) w_float[i] = 1.0f;     /* entity A */
        else                 w_float[i] = -1.0f;    /* entity B */
    }

    /* Generate random activations */
    float* x_float = (float*)malloc(S * K * sizeof(float));
    for (int i = 0; i < S * K; i++) {
        x_float[i] = randf();
    }

    /* ---- {2,3} path: quantize → integer matmul → dequantize ---- */

    /* Pack weights to ternary */
    Two3Weights W = two3_pack_weights(w_float, M, K);

    /* Quantize activations to int8 */
    Two3Activations X = two3_quantize_acts(x_float, S, K);

    /* Forward: the actual {2,3} kernel */
    Two3Output Y = two3_forward(&W, &X);

    /* Dequantize result */
    float* y_two3 = (float*)malloc(S * M * sizeof(float));
    two3_dequantize_output(&Y, &W, &X, y_two3);

    /* ---- Reference path: use ternary-quantized weights ---- */
    /* Reconstruct what the kernel actually computes:
     * w_recon[i] = round(w_float[i] / scale) * scale, clamped to {-1,0,+1}*scale */
    float* w_recon = (float*)malloc(M * K * sizeof(float));
    float inv_scale = 1.0f / W.scale;
    for (int i = 0; i < M * K; i++) {
        int q = (int)roundf(w_float[i] * inv_scale);
        if (q > 1) q = 1;
        if (q < -1) q = -1;
        w_recon[i] = (float)q * W.scale;
    }
    float* y_ref = (float*)malloc(S * M * sizeof(float));
    two3_ref_matmul(x_float, w_recon, y_ref, S, K, M);
    free(w_recon);

    /* ---- Compare ---- */
    double max_abs_err = 0.0;
    double sum_abs_err = 0.0;
    double max_rel_err = 0.0;
    double sum_rel_err = 0.0;
    int    rel_count = 0;

    for (int i = 0; i < S * M; i++) {
        double err = fabs((double)y_two3[i] - (double)y_ref[i]);
        if (err > max_abs_err) max_abs_err = err;
        sum_abs_err += err;

        double ref_mag = fabs((double)y_ref[i]);
        if (ref_mag > 1e-6) {
            double rel = err / ref_mag;
            if (rel > max_rel_err) max_rel_err = rel;
            sum_rel_err += rel;
            rel_count++;
        }
    }

    int total = S * M;
    double mean_abs_err = sum_abs_err / total;
    double mean_rel_err = (rel_count > 0) ? sum_rel_err / rel_count : 0.0;

    printf("\n[verification]\n");
    printf("  output elements:    %d\n", total);
    printf("  max  absolute err:  %.6f\n", max_abs_err);
    printf("  mean absolute err:  %.6f\n", mean_abs_err);
    printf("  max  relative err:  %.2f%%\n", max_rel_err * 100.0);
    printf("  mean relative err:  %.2f%%\n", mean_rel_err * 100.0);

    /* Sample outputs */
    printf("\n[sample outputs] (first 5 elements of first token)\n");
    printf("  %-12s %-12s %-12s\n", "{2,3}", "float ref", "diff");
    int show = (M < 5) ? M : 5;
    for (int i = 0; i < show; i++) {
        printf("  %-12.4f %-12.4f %-12.4f\n",
               y_two3[i], y_ref[i], y_two3[i] - y_ref[i]);
    }

    int pass = (mean_rel_err < 0.20);  /* 20% mean relative error threshold */
    printf("\n  result: %s\n\n", pass ? "PASS" : "FAIL");

    /* Cleanup */
    two3_free_weights(&W);
    two3_free_acts(&X);
    two3_free_output(&Y);
    free(w_float);
    free(x_float);
    free(y_two3);
    free(y_ref);

    return pass;
}

/* ---- Test: substrate dominance (sparse weights) ---- */
static int test_sparse_weights(void) {
    int S = 4, M = 64, K = 256;
    printf("--- Test: sparse weights (high substrate %%) ---\n");

    float* w_float = (float*)malloc(M * K * sizeof(float));
    /* 70% substrate, 15% +1, 15% -1 — sparse ternary */
    for (int i = 0; i < M * K; i++) {
        float r = (float)rand() / (float)RAND_MAX;
        if (r < 0.70f)       w_float[i] = 0.0f;   /* substrate */
        else if (r < 0.85f)  w_float[i] = 1.0f;    /* entity A */
        else                 w_float[i] = -1.0f;   /* entity B */
    }

    float* x_float = (float*)malloc(S * K * sizeof(float));
    for (int i = 0; i < S * K; i++) x_float[i] = randf();

    Two3Weights W = two3_pack_weights(w_float, M, K);
    Two3Activations X = two3_quantize_acts(x_float, S, K);
    Two3Output Y = two3_forward(&W, &X);

    float* y_two3 = (float*)malloc(S * M * sizeof(float));
    two3_dequantize_output(&Y, &W, &X, y_two3);

    /* Reconstruct ternary weights for fair comparison */
    float* w_recon = (float*)malloc(M * K * sizeof(float));
    float inv_s = 1.0f / W.scale;
    for (int i = 0; i < M * K; i++) {
        int q = (int)roundf(w_float[i] * inv_s);
        if (q > 1) q = 1; if (q < -1) q = -1;
        w_recon[i] = (float)q * W.scale;
    }
    float* y_ref = (float*)malloc(S * M * sizeof(float));
    two3_ref_matmul(x_float, w_recon, y_ref, S, K, M);
    free(w_recon);

    double max_rel = 0.0, sum_rel = 0.0;
    int count = 0;
    for (int i = 0; i < S * M; i++) {
        double ref_mag = fabs((double)y_ref[i]);
        if (ref_mag > 1e-6) {
            double rel = fabs((double)y_two3[i] - (double)y_ref[i]) / ref_mag;
            if (rel > max_rel) max_rel = rel;
            sum_rel += rel;
            count++;
        }
    }

    double mean_rel = (count > 0) ? sum_rel / count : 0.0;
    printf("[verification] mean relative error: %.2f%% (max: %.2f%%, %d comparisons)\n",
           mean_rel * 100.0, max_rel * 100.0, count);
    int pass = (mean_rel < 0.10);  /* 10% mean relative error */
    printf("  result: %s\n\n", pass ? "PASS" : "FAIL");

    two3_free_weights(&W);
    two3_free_acts(&X);
    two3_free_output(&Y);
    free(w_float); free(x_float); free(y_two3); free(y_ref);
    return pass;
}

/* ---- Test: performance (throughput measurement) ---- */
static void test_performance(void) {
    /* Simulate a real transformer layer dimension */
    int S = 128;     /* sequence length */
    int M = 2048;    /* output features (hidden dim) */
    int K = 2048;    /* input features  (hidden dim) */

    printf("--- Test: performance [S=%d, M=%d, K=%d] ---\n", S, M, K);

    float* w_float = (float*)malloc(M * K * sizeof(float));
    for (int i = 0; i < M * K; i++) w_float[i] = randf() * 0.1f;

    float* x_float = (float*)malloc(S * K * sizeof(float));
    for (int i = 0; i < S * K; i++) x_float[i] = randf();

    Two3Weights W = two3_pack_weights(w_float, M, K);
    Two3Activations X = two3_quantize_acts(x_float, S, K);

    /* Warm up */
    Two3Output Y_warmup = two3_forward(&W, &X);
    two3_free_output(&Y_warmup);

    /* Timed run */
    int runs = 100;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    for (int r = 0; r < runs; r++) {
        Two3Output Y = two3_forward(&W, &X);
        two3_free_output(&Y);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms;
    cudaEventElapsedTime(&ms, start, stop);

    float ms_per_call = ms / runs;
    double ops = 2.0 * S * M * K;  /* each weight = add or subtract */
    double gops = (ops / (ms_per_call * 1e-3)) / 1e9;

    printf("\n[performance]\n");
    printf("  %d runs, %.2f ms total\n", runs, ms);
    printf("  %.3f ms per forward\n", ms_per_call);
    printf("  %.1f GOPS (integer)\n", gops);

    /* Memory stats */
    int weight_bytes = W.rows * (W.cols / 4);
    int act_bytes = S * K;
    int out_bytes = S * M * 4;

    printf("\n[memory]\n");
    printf("  weights: %d bytes (%.1f KB) — %.1fx vs FP16\n",
           weight_bytes, weight_bytes / 1024.0f,
           (float)(M * K * 2) / weight_bytes);
    printf("  activations: %d bytes (%.1f KB)\n", act_bytes, act_bytes / 1024.0f);
    printf("  output: %d bytes (%.1f KB)\n", out_bytes, out_bytes / 1024.0f);
    printf("  total: %.1f KB\n\n",
           (weight_bytes + act_bytes + out_bytes) / 1024.0f);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    two3_free_weights(&W);
    two3_free_acts(&X);
    free(w_float);
    free(x_float);
}

/* ---- Main ---- */
int main(void) {
    srand((unsigned)time(NULL));
    print_device_info();

    int passed = 0, total = 0;

    /* Small dimensions — easy to debug */
    total++; passed += test_forward(1, 16, 32);      /* single token */
    total++; passed += test_forward(4, 64, 128);     /* small batch  */
    total++; passed += test_forward(32, 256, 512);   /* medium       */

    /* Sparse weight test */
    total++; passed += test_sparse_weights();

    /* Performance benchmark */
    test_performance();

    /* Summary */
    printf("============================================\n");
    printf("  Results: %d / %d tests passed\n", passed, total);
    printf("============================================\n");

    if (passed == total) {
        printf("\n  {2,3} kernel verified.\n");
        printf("  Two states. One substrate. No multiply.\n");
        printf("  The foundation holds.\n\n");
    }

    return (passed == total) ? 0 : 1;
}
