/*
 * hadamard_ablation.cu — Hadamard pre-quant vs plain quant (host reference)
 *
 * What this measures
 *   - 1D: same vector length as power-of-2 (sanity).
 *   - 2D row-wise: matches training layout (kernel_hadamard_rows: Hadamard on each row of [rows×cols]).
 *
 * What this is NOT
 *   - Not bit-identical to requantize_gpu (different scale rule: we use per-row absmean; GPU uses
 *     fused absmean kernel). Still first-principles signal on whether Hadamard helps ternary MSE.
 *
 * Build: build_driver.bat hadamard
 *    or: nvcc -O2 -o hadamard_ablation.exe hadamard_ablation.cu
 * Run:  hadamard_ablation.exe
 */

#include "two3.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Match train.h STE — compare |w/scale| to threshold */
#ifndef STE_THRESHOLD
#define STE_THRESHOLD 0.33f
#endif

/* Ternary in “float reconstruction” form: {-scale, 0, +scale} */
static void quantize_vec(const float *w, float *out, int n) {
    float scale = 0.f;
    for (int i = 0; i < n; i++)
        scale += fabsf(w[i]);
    scale /= (float)n;
    if (scale < 1e-10f)
        scale = 1.f;
    for (int i = 0; i < n; i++) {
        float t = w[i] / scale;
        if (t > STE_THRESHOLD)
            out[i] = scale;
        else if (t < -STE_THRESHOLD)
            out[i] = -scale;
        else
            out[i] = 0.f;
    }
}

static void quantize_vec_hadamard(const float *w, float *out, int n) {
    if ((n & (n - 1)) != 0) {
        fprintf(stderr, "n=%d must be power of 2\n", n);
        exit(1);
    }
    float *tmp = (float *)malloc((size_t)n * sizeof(float));
    memcpy(tmp, w, (size_t)n * sizeof(float));
    hadamard_transform(tmp, n);

    /* Iterative scale refinement (Lloyd-style): re-estimate scale from weights that would be ±1
     * under current scale. Use same boundary as quantizer: |w| > STE_THRESHOLD * scale → active. */
    float scale = 0.f;
    for (int i = 0; i < n; i++)
        scale += fabsf(tmp[i]);
    scale /= (float)n;
    if (scale < 1e-10f)
        scale = 1.f;

    for (int iter = 0; iter < 3; iter++) {
        float sum_active = 0.f;
        int count_active = 0;
        const float boundary = STE_THRESHOLD * scale;
        for (int i = 0; i < n; i++) {
            float abs_w = fabsf(tmp[i]);
            if (abs_w > boundary) {
                sum_active += abs_w;
                count_active++;
            }
        }
        if (count_active == 0)
            break;
        scale = sum_active / (float)count_active;
    }

    for (int i = 0; i < n; i++) {
        float t = tmp[i] / scale;
        if (t > STE_THRESHOLD)
            tmp[i] = scale;
        else if (t < -STE_THRESHOLD)
            tmp[i] = -scale;
        else
            tmp[i] = 0.f;
    }
    /* Orthonormal FWT: inverse equals forward */
    hadamard_transform(tmp, n);
    memcpy(out, tmp, (size_t)n * sizeof(float));
    free(tmp);
}

static void generate_weights(float *w, int n, float outlier_frac) {
    for (int i = 0; i < n; i++) {
        float u = (float)rand() / (float)RAND_MAX;
        if (u < outlier_frac)
            w[i] = ((float)rand() / (float)RAND_MAX - 0.5f) * 6.0f;
        else
            w[i] = ((float)rand() / (float)RAND_MAX - 0.5f) * 2.0f;
    }
}

static float excess_kurtosis(const float *x, int n) {
    float mean = 0.f;
    for (int i = 0; i < n; i++)
        mean += x[i];
    mean /= (float)n;
    float var = 0.f;
    for (int i = 0; i < n; i++) {
        float d = x[i] - mean;
        var += d * d;
    }
    var /= (float)n;
    float std = sqrtf(var + 1e-30f);
    float m4 = 0.f;
    for (int i = 0; i < n; i++) {
        float z = (x[i] - mean) / std;
        m4 += z * z * z * z;
    }
    m4 /= (float)n;
    return m4 - 3.f;
}

static float mse(const float *a, const float *b, int n) {
    float s = 0.f;
    for (int i = 0; i < n; i++) {
        float d = a[i] - b[i];
        s += d * d;
    }
    return s / (float)n;
}

static float mse_matrix(const float *a, const float *b, int rows, int cols) {
    return mse(a, b, rows * cols);
}

static void quantize_matrix_plain(const float *w, float *out, int rows, int cols) {
    for (int r = 0; r < rows; r++)
        quantize_vec(w + r * cols, out + r * cols, cols);
}

static void quantize_matrix_hadamard(const float *w, float *out, int rows, int cols) {
    if ((cols & (cols - 1)) != 0) {
        fprintf(stderr, "cols=%d must be power of 2\n", cols);
        exit(1);
    }
    for (int r = 0; r < rows; r++)
        quantize_vec_hadamard(w + r * cols, out + r * cols, cols);
}

/* Simple Hadamard quantization (L1 mean scale, no Lloyd-Max refinement) */
static void quantize_vec_hadamard_simple(const float *w, float *out, int n) {
    if ((n & (n - 1)) != 0) {
        fprintf(stderr, "n=%d must be power of 2\n", n);
        exit(1);
    }
    float *tmp = (float *)malloc((size_t)n * sizeof(float));
    memcpy(tmp, w, (size_t)n * sizeof(float));
    hadamard_transform(tmp, n);

    /* Simple L1 mean scale */
    float scale = 0.f;
    for (int i = 0; i < n; i++)
        scale += fabsf(tmp[i]);
    scale /= (float)n;
    if (scale < 1e-10f)
        scale = 1.f;

    for (int i = 0; i < n; i++) {
        float t = tmp[i] / scale;
        if (t > STE_THRESHOLD)
            tmp[i] = scale;
        else if (t < -STE_THRESHOLD)
            tmp[i] = -scale;
        else
            tmp[i] = 0.f;
    }
    hadamard_transform(tmp, n);
    memcpy(out, tmp, (size_t)n * sizeof(float));
    free(tmp);
}

static void quantize_matrix_hadamard_simple(const float *w, float *out, int rows, int cols) {
    if ((cols & (cols - 1)) != 0) {
        fprintf(stderr, "cols=%d must be power of 2\n", cols);
        exit(1);
    }
    for (int r = 0; r < rows; r++)
        quantize_vec_hadamard_simple(w + r * cols, out + r * cols, cols);
}

int main(void) {
    srand(42);
    printf("Hadamard pre-quant ablation (host, STE_THRESHOLD=%.2f)\n", STE_THRESHOLD);
    printf("--- 1D (power-of-2 length) ---\n");

    int sizes[] = {256, 512, 1024, 2048};
    float outlier_fracs[] = {0.f, 0.05f, 0.1f, 0.2f};
    int n_sz = (int)(sizeof(sizes) / sizeof(sizes[0]));
    int n_of = (int)(sizeof(outlier_fracs) / sizeof(outlier_fracs[0]));
    double sum_improve_1d = 0.;
    int n_tests_1d = 0;

    for (int si = 0; si < n_sz; si++) {
        for (int oi = 0; oi < n_of; oi++) {
            int n = sizes[si];
            float *w = (float *)malloc((size_t)n * sizeof(float));
            float *q0 = (float *)malloc((size_t)n * sizeof(float));
            float *q1 = (float *)malloc((size_t)n * sizeof(float));
            generate_weights(w, n, outlier_fracs[oi]);
            float kurt = excess_kurtosis(w, n);
            quantize_vec(w, q0, n);
            quantize_vec_hadamard(w, q1, n);
            float m0 = mse(w, q0, n);
            float m1 = mse(w, q1, n);
            float ratio = (m0 > 1e-20f) ? (m1 / m0) : 0.f;
            float theory = 1.f / (1.f + kurt / 3.f);
            if (theory < 0.f)
                theory = 0.f;
            if (theory > 1.f)
                theory = 1.f;
            printf("n=%4d out=%.0f%% kurt=%6.3f  MSE=%.6f  MSE_H=%.6f  ratio=%.4f  theory~%.4f\n",
                   n, outlier_fracs[oi] * 100.f, kurt, m0, m1, ratio, theory);
            sum_improve_1d += (double)(1.f - ratio);
            n_tests_1d++;
            free(w);
            free(q0);
            free(q1);
        }
    }
    printf("1D avg (1 - MSE_H/MSE) * 100 = %.2f%%\n\n", (sum_improve_1d / n_tests_1d) * 100.0);

    printf("--- 2D row-wise (same as Hadamard per row in requantize_gpu) ---\n");
    struct {
        int rows, cols;
    } mats[] = {{32, 256}, {64, 256}, {128, 512}, {16, 1024}};
    int n_mats = (int)(sizeof(mats) / sizeof(mats[0]));
    double sum_improve_2d = 0.;
    int n_tests_2d = 0;

    for (int mi = 0; mi < n_mats; mi++) {
        int rows = mats[mi].rows, cols = mats[mi].cols;
        int n = rows * cols;
        float *w = (float *)malloc((size_t)n * sizeof(float));
        float *q0 = (float *)malloc((size_t)n * sizeof(float));
        float *q1 = (float *)malloc((size_t)n * sizeof(float));
        generate_weights(w, n, 0.1f);
        quantize_matrix_plain(w, q0, rows, cols);
        quantize_matrix_hadamard(w, q1, rows, cols);
        float m0 = mse_matrix(w, q0, rows, cols);
        float m1 = mse_matrix(w, q1, rows, cols);
        float ratio = (m0 > 1e-20f) ? (m1 / m0) : 0.f;
        printf("[%3d x %4d]  MSE=%.6f  MSE_H=%.6f  ratio=%.4f\n", rows, cols, m0, m1, ratio);
        sum_improve_2d += (double)(1.f - ratio);
        n_tests_2d++;
        free(w);
        free(q0);
        free(q1);
    }
    printf("2D avg improvement %% = %.2f\n\n", (sum_improve_2d / n_tests_2d) * 100.0);

    printf("--- Hadamard: L1-mean scale vs iterative active-scale refinement ---\n");
    for (int mi = 0; mi < n_mats; mi++) {
        int rows = mats[mi].rows, cols = mats[mi].cols;
        int n = rows * cols;
        float *w = (float *)malloc((size_t)n * sizeof(float));
        float *q_simple = (float *)malloc((size_t)n * sizeof(float));
        float *q_lloyd = (float *)malloc((size_t)n * sizeof(float));
        generate_weights(w, n, 0.1f);
        
        /* Simple Hadamard (L1 mean scale) */
        quantize_matrix_hadamard_simple(w, q_simple, rows, cols);
        /* Lloyd-Max refined scale */
        quantize_matrix_hadamard(w, q_lloyd, rows, cols);
        
        float m_simple = mse_matrix(w, q_simple, rows, cols);
        float m_lloyd = mse_matrix(w, q_lloyd, rows, cols);
        float improvement = (m_simple - m_lloyd) / m_simple * 100;
        printf("[%3d x %4d]  MSE_simple=%.6f  MSE_LM=%.6f  improvement=%.2f%%\n",
               rows, cols, m_simple, m_lloyd, improvement);
        free(w);
        free(q_simple);
        free(q_lloyd);
    }

    return 0;
}
