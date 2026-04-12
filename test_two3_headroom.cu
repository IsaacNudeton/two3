/*
 * test_two3_headroom.cu — {2,3} headroom shape verification
 *
 * Tests:
 *   1. Headroom profile at w ∈ {0, 1/6, 1/3, 1/2, 2/3, 5/6, 1}
 *      Expected: peaks at 1/3 and 2/3, troughs at {0, 1/2, 1}
 *   2. Behavioral inversion: w=0.5 must resist flipping (low movement)
 *      vs w=1/3 or w=2/3 which should move easily (high movement)
 *   3. The old binary "w=0.5 is the boundary" assumption is now broken —
 *      w=0.5 is the center of the substrate band.
 *
 * Setup: one Adam step with fixed m, v, gradient. Measure latent movement
 * at each probe point. Assert movement[w=1/3] > movement[w=1/2].
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <cuda_runtime.h>
#include "binary_resident.h"

#define CHECK_CUDA(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err)); exit(1); \
    } \
} while(0)

int main(void) {
    printf("============================================\n");
    printf("  {2,3} Headroom Shape Test\n");
    printf("============================================\n\n");

    /* Probe points covering attractors and flip surfaces */
    const int N = 7;
    float probes[7] = {
        0.0f,
        1.0f / 6.0f,
        1.0f / 3.0f,
        0.5f,
        2.0f / 3.0f,
        5.0f / 6.0f,
        1.0f
    };
    const char *labels[7] = {
        "w=0     (attractor -1)",
        "w=1/6   (mid of -1 region)",
        "w=1/3   (flip boundary -1<->0)",
        "w=1/2   (mid of substrate 0)",
        "w=2/3   (flip boundary 0<->+1)",
        "w=5/6   (mid of +1 region)",
        "w=1     (attractor +1)"
    };

    /* Device buffers — one element per probe */
    float *d_params, *d_grads, *d_m, *d_v;
    CHECK_CUDA(cudaMalloc(&d_params, N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_grads,  N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_m,      N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_v,      N * sizeof(float)));

    /* Probe 1: positive-push test.
     * Set up Adam state so the update is exactly +0.1 (CFL-clamped from above).
     * m_hat/sqrt(v_hat) >> 0.1/lr, then CFL clamp kicks in.
     * With lr=1.0, m=10, v=1, b1_corr=1, b2_corr=1 → update=10 → clamp to 0.1. */
    float initial[7];
    memcpy(initial, probes, sizeof(initial));
    float grads[7]; for (int i = 0; i < N; i++) grads[i] = 10.0f;
    float ms[7];    for (int i = 0; i < N; i++) ms[i]    = 10.0f;
    float vs[7];    for (int i = 0; i < N; i++) vs[i]    = 1.0f;

    CHECK_CUDA(cudaMemcpy(d_params, initial, N * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_grads,  grads,   N * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_m,      ms,      N * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_v,      vs,      N * sizeof(float), cudaMemcpyHostToDevice));

    /* Run one headroom step — lr=1 so the raw update is large, CFL clamps to ±0.1 */
    resident_kernel_adam_headroom<<<1, N>>>(
        d_params, d_grads, d_m, d_v, N,
        1.0f, 0.9f, 0.999f, 1e-8f, 1.0f, 1.0f);
    CHECK_CUDA(cudaDeviceSynchronize());

    float result[7];
    CHECK_CUDA(cudaMemcpy(result, d_params, N * sizeof(float), cudaMemcpyDeviceToHost));

    printf("Positive push (grad>0, Adam wants to DECREASE w):\n");
    printf("  %-32s  %6s  %10s  %8s\n", "probe", "w_in", "w_out", "|delta|");
    float movement[7];
    for (int i = 0; i < N; i++) {
        movement[i] = fabsf(result[i] - initial[i]);
        printf("  %-32s  %6.4f  %10.6f  %8.5f\n",
               labels[i], initial[i], result[i], movement[i]);
    }

    /* Assertions on the shape */
    printf("\n--- Behavioral inversion check ---\n");
    float mov_half    = movement[3];  /* w=0.5  (substrate attractor) */
    float mov_third   = movement[2];  /* w=1/3  (flip boundary) */
    float mov_twothrd = movement[4];  /* w=2/3  (flip boundary) */

    int pass = 1;
    if (mov_third > mov_half) {
        printf("  [PASS] w=1/3 moves MORE than w=1/2  (%.5f > %.5f)\n",
               mov_third, mov_half);
    } else {
        printf("  [FAIL] w=1/3 should move more than w=1/2  (%.5f <= %.5f)\n",
               mov_third, mov_half);
        pass = 0;
    }

    if (mov_twothrd > mov_half) {
        printf("  [PASS] w=2/3 moves MORE than w=1/2  (%.5f > %.5f)\n",
               mov_twothrd, mov_half);
    } else {
        printf("  [FAIL] w=2/3 should move more than w=1/2  (%.5f <= %.5f)\n",
               mov_twothrd, mov_half);
        pass = 0;
    }

    /* Peak/trough ratio: the boundaries should move at least 10x more than
     * the substrate center (h=2.0 vs h=0.1 ratio). */
    float ratio = mov_third / fmaxf(mov_half, 1e-10f);
    printf("  Ratio mov(1/3)/mov(1/2) = %.2f  (expected ~20 for 2.0/0.1 headroom ratio)\n", ratio);
    if (ratio < 10.0f) {
        printf("  [FAIL] ratio too small — headroom shape may be wrong\n");
        pass = 0;
    } else {
        printf("  [PASS] ratio confirms peak-at-boundary, trough-at-substrate\n");
    }

    /* Attractor test: {0, 1/6, 1/2, 5/6, 1} should all have similar low movement. */
    printf("\n--- Attractor stability check ---\n");
    float attractor_mov[5] = {
        movement[0],  /* w=0 */
        movement[1],  /* w=1/6 */
        movement[3],  /* w=1/2 */
        movement[5],  /* w=5/6 */
        movement[6]   /* w=1 */
    };
    float max_attractor = 0.0f;
    for (int i = 0; i < 5; i++) if (attractor_mov[i] > max_attractor) max_attractor = attractor_mov[i];
    if (max_attractor < mov_third * 0.2f) {
        printf("  [PASS] all attractor points have low mobility (max %.5f vs boundary %.5f)\n",
               max_attractor, mov_third);
    } else {
        printf("  [WARN] some attractor point moved substantially (max %.5f vs boundary %.5f)\n",
               max_attractor, mov_third);
    }

    /* Cleanup */
    cudaFree(d_params); cudaFree(d_grads); cudaFree(d_m); cudaFree(d_v);

    printf("\n============================================\n");
    printf("  %s\n", pass ? "HEADROOM SHAPE: PASS" : "HEADROOM SHAPE: FAIL");
    printf("============================================\n");
    return pass ? 0 : 1;
}
