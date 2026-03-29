/*
 * test_train.cu — Layer 4 verification: training with STE
 *
 * Tests:
 * 1. Loss computes without NaN/inf
 * 2. Gradient is non-zero (information flows backward)
 * 3. Loss decreases over multiple steps (learning happens)
 * 4. STE quantization: latent weights change, ternary follows
 * 5. Checkpoint save/load round-trip
 *
 * Isaac & CC — March 2026
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <cuda_runtime.h>
#include "train.h"

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        printf("CUDA error: %s\n", cudaGetErrorString(err)); exit(1); \
    } \
} while(0)

/* Small config for fast tests */
static ModelConfig test_train_config(void) {
    ModelConfig c;
    c.dim = 64;
    c.n_heads = 2;
    c.n_kv_heads = 1;
    c.head_dim = 32;
    c.intermediate = 128;
    c.n_layers = 1;
    c.max_seq = 64;
    c.rope_theta = 1000000.0f;
    return c;
}

/* ═══════════════════════════════════════════════════════
 * Test 1: Loss computes without NaN/inf
 * ═══════════════════════════════════════════════════════ */

static int test_loss_valid(void) {
    printf("--- Test 1: loss computes without NaN/inf ---\n");
    ModelConfig cfg = test_train_config();

    TrainableModel tm;
    trainable_model_init(&tm, cfg);
    trainable_requantize(&tm);

    uint8_t input[] = "Hello";
    int len = 5;

    TrainResult r = trainable_train_step(&tm, input, len);

    printf("  input: \"Hello\" (%d bytes)\n", len);
    printf("  loss: %.4f\n", r.loss);
    printf("  max_grad: %.6f\n", r.max_grad);
    printf("  correct: %d / %d\n", r.correct, len - 1);

    int nan_loss = (r.loss != r.loss);
    int inf_loss = (r.loss > 1e30f || r.loss < -1e30f);

    int pass = (!nan_loss && !inf_loss && r.loss > 0.0f);
    printf("  result: %s\n\n", pass ? "PASS" : "FAIL");

    trainable_model_free(&tm);
    return pass;
}

/* ═══════════════════════════════════════════════════════
 * Test 2: Gradient is non-zero
 * ═══════════════════════════════════════════════════════ */

static int test_gradient_nonzero(void) {
    printf("--- Test 2: gradient is non-zero ---\n");
    ModelConfig cfg = test_train_config();

    TrainableModel tm;
    trainable_model_init(&tm, cfg);
    trainable_requantize(&tm);

    uint8_t input[] = "Test input for gradient check";
    int len = (int)strlen((char*)input);

    TrainResult r = trainable_train_step(&tm, input, len);

    printf("  max_grad: %.6f\n", r.max_grad);
    printf("  loss: %.4f\n", r.loss);

    int pass = (r.max_grad > 1e-10f);
    printf("  gradient flows: %s\n", pass ? "YES" : "NO (dead gradient)");
    printf("  result: %s\n\n", pass ? "PASS" : "FAIL");

    trainable_model_free(&tm);
    return pass;
}

/* ═══════════════════════════════════════════════════════
 * Test 3: Loss decreases over steps
 * ═══════════════════════════════════════════════════════ */

static int test_loss_decreases(void) {
    printf("--- Test 3: loss decreases over 20 steps ---\n");
    ModelConfig cfg = test_train_config();

    TrainableModel tm;
    trainable_model_init(&tm, cfg);
    trainable_requantize(&tm);

    /* Simple repeating pattern — should be learnable */
    uint8_t pattern[] = "abcabcabcabcabcabc";
    int len = (int)strlen((char*)pattern);

    float first_loss = 0, last_loss = 0;

    for (int step = 0; step < 20; step++) {
        TrainResult r = trainable_train_step(&tm, pattern, len);
        if (step == 0) first_loss = r.loss;
        if (step == 19) last_loss = r.loss;

        if (step % 5 == 0 || step == 19)
            printf("  step %2d: loss=%.4f  correct=%d/%d  max_grad=%.6f\n",
                   step, r.loss, r.correct, len - 1, r.max_grad);
    }

    printf("  first_loss=%.4f  last_loss=%.4f  ratio=%.3f\n",
           first_loss, last_loss, last_loss / first_loss);

    int pass = (last_loss < first_loss);
    printf("  loss decreased: %s\n", pass ? "YES" : "NO");
    printf("  result: %s\n\n", pass ? "PASS" : "FAIL");

    trainable_model_free(&tm);
    return pass;
}

/* ═══════════════════════════════════════════════════════
 * Test 4: STE changes ternary weights
 * ═══════════════════════════════════════════════════════ */

static int test_ste_quantization(void) {
    printf("--- Test 4: STE updates latent weights ---\n");
    ModelConfig cfg = test_train_config();

    TrainableModel tm;
    trainable_model_init(&tm, cfg);
    trainable_requantize(&tm);

    int D = cfg.dim;

    /* Save initial latent weights */
    float *init_wq = (float*)malloc(D * D * sizeof(float));
    memcpy(init_wq, tm.layer_weights[0].W_q, D * D * sizeof(float));

    /* Train for 50 steps on a pattern */
    uint8_t pattern[] = "The quick brown fox jumps";
    int len = (int)strlen((char*)pattern);

    for (int step = 0; step < 50; step++)
        trainable_train_step(&tm, pattern, len);

    /* Check how much latent weights moved */
    double max_diff = 0, sum_diff = 0;
    int flipped = 0;
    for (int i = 0; i < D * D; i++) {
        double d = fabs((double)tm.layer_weights[0].W_q[i] - (double)init_wq[i]);
        if (d > max_diff) max_diff = d;
        sum_diff += d;
        /* Check if ternary quantization changed */
        float tq_before = ternary_quantize(init_wq[i]);
        float tq_after = ternary_quantize(tm.layer_weights[0].W_q[i]);
        if (tq_before != tq_after) flipped++;
    }

    printf("  W_q latent (dim=%d, total=%d):\n", D, D * D);
    printf("    max weight change: %.6f\n", max_diff);
    printf("    mean weight change: %.6f\n", sum_diff / (D * D));
    printf("    ternary flips: %d / %d\n", flipped, D * D);

    /* Latent weights MUST change (Adam is updating them) */
    int pass = (max_diff > 1e-6);
    printf("  latent weights moved: %s\n", pass ? "YES" : "NO");
    printf("  result: %s\n\n", pass ? "PASS" : "FAIL");

    free(init_wq);
    trainable_model_free(&tm);
    return pass;
}

/* ═══════════════════════════════════════════════════════
 * Test 5: Checkpoint round-trip
 * ═══════════════════════════════════════════════════════ */

static int test_checkpoint(void) {
    printf("--- Test 5: checkpoint save/load ---\n");
    ModelConfig cfg = test_train_config();

    TrainableModel tm;
    trainable_model_init(&tm, cfg);
    trainable_requantize(&tm);

    /* Train a few steps so weights are non-trivial */
    uint8_t pattern[] = "checkpoint test data";
    int len = (int)strlen((char*)pattern);
    for (int i = 0; i < 5; i++)
        trainable_train_step(&tm, pattern, len);

    /* Save */
    int save_ok = trainable_save(&tm, "test_checkpoint.t2l4");
    printf("  save: %s\n", save_ok == 0 ? "OK" : "FAILED");

    /* Load into new model */
    TrainableModel tm2;
    trainable_model_init(&tm2, cfg);
    int load_ok = trainable_load(&tm2, "test_checkpoint.t2l4");
    printf("  load: %s\n", load_ok == 0 ? "OK" : "FAILED");

    /* Compare embedding */
    int D = cfg.dim;
    double max_diff = 0;
    for (int i = 0; i < 256 * D; i++) {
        double d = fabs((double)tm.latent_embed[i] - (double)tm2.latent_embed[i]);
        if (d > max_diff) max_diff = d;
    }
    printf("  max embed diff: %.2e\n", max_diff);

    /* Compare W_q */
    double max_wq_diff = 0;
    for (int i = 0; i < D * D; i++) {
        double d = fabs((double)tm.layer_weights[0].W_q[i] - (double)tm2.layer_weights[0].W_q[i]);
        if (d > max_wq_diff) max_wq_diff = d;
    }
    printf("  max W_q diff: %.2e\n", max_wq_diff);

    int pass = (save_ok == 0 && load_ok == 0 && max_diff < 1e-6 && max_wq_diff < 1e-6);
    printf("  result: %s\n\n", pass ? "PASS" : "FAIL");

    trainable_model_free(&tm);
    trainable_model_free(&tm2);

    /* Cleanup checkpoint file */
    remove("test_checkpoint.t2l4");

    return pass;
}

/* ═══════════════════════════════════════════════════════ */

int main(void) {
    printf("============================================\n");
    printf("  {2,3} Layer 4 — Training with STE\n");
    printf("  Adam + STE + Cross-Entropy\n");
    printf("  This is where it LEARNS.\n");
    printf("============================================\n\n");

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("  GPU: %s\n\n", prop.name);

    int passed = 0, total = 0;

    total++; passed += test_loss_valid();
    total++; passed += test_gradient_nonzero();
    total++; passed += test_loss_decreases();
    total++; passed += test_ste_quantization();
    total++; passed += test_checkpoint();

    printf("============================================\n");
    printf("  Results: %d / %d tests passed\n", passed, total);
    printf("============================================\n");

    if (passed == total)
        printf("\n  Layer 4 verified.\n"
               "  Loss computes. Gradients flow.\n"
               "  Loss decreases. Weights change.\n"
               "  Checkpoints persist.\n"
               "  The model LEARNS.\n"
               "  Next: real data, longer training.\n\n");

    return (passed == total) ? 0 : 1;
}
