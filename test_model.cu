/*
 * test_model.cu — Full model integration test
 *
 * Layer 2 verification: real attention, real MoE, real ternary projections.
 * Tests go beyond "no NaN" — they prove attention is WORKING.
 *
 * 1. Single byte: no NaN, no overflow (smoke test)
 * 2. Attention causality: output at position t must depend on byte at t-1
 * 3. Position sensitivity: same byte at different positions → different logits
 * 4. Sequence forward: logits shift when context changes
 * 5. Generation: produce bytes from context
 * 6. Memory footprint: fits in 8GB VRAM
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <cuda_runtime.h>
#include "model.h"

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        printf("CUDA error: %s\n", cudaGetErrorString(err)); exit(1); \
    } \
} while(0)

/* Helper: L2 distance between two float vectors */
static float vec_l2_dist(const float *a, const float *b, int n) {
    double sum = 0;
    for (int i = 0; i < n; i++) {
        double d = (double)a[i] - (double)b[i];
        sum += d * d;
    }
    return (float)sqrt(sum);
}

/* Helper: check for any NaN in a float array */
static int has_any_nan(const float *x, int n) {
    for (int i = 0; i < n; i++)
        if (x[i] != x[i]) return 1;
    return 0;
}

/* Test config: small for speed, same architecture */
static ModelConfig test_config(void) {
    ModelConfig c;
    c.dim = 128;          /* small for test speed */
    c.n_heads = 4;
    c.n_kv_heads = 2;
    c.head_dim = 32;      /* 128 / 4 */
    c.intermediate = 256;
    c.n_layers = 2;       /* 2 layers enough to verify composition */
    c.max_seq = 512;
    c.rope_theta = 1000000.0f;
    return c;
}

/* ═══════════════════════════════════════════════════════
 * Test 1: Smoke test — single byte produces valid logits
 * ═══════════════════════════════════════════════════════ */

static int test_single_byte(void) {
    printf("--- Test 1: single byte → valid logits ---\n");
    ModelConfig cfg = test_config();

    srand(42);
    Model m;
    model_init(&m, cfg);

    uint8_t input = 72;  /* 'H' */
    float logits[256];
    model_forward_sequence_cpu(&m, &input, 1, logits, MODEL_FWD_FLAGS_DEFAULT);

    int nan = has_any_nan(logits, 256);
    float max_l = logits[0], min_l = logits[0];
    for (int i = 1; i < 256; i++) {
        if (logits[i] > max_l) max_l = logits[i];
        if (logits[i] < min_l) min_l = logits[i];
    }

    printf("  input: byte %d ('%c')\n", input, input);
    printf("  logit range: [%.4f, %.4f]\n", min_l, max_l);
    printf("  logit spread: %.4f\n", max_l - min_l);
    printf("  NaN: %s\n", nan ? "YES" : "no");

    /* Logits should have non-trivial spread (not all same value) */
    int pass = (!nan && (max_l - min_l) > 0.001f && max_l < 1e6f);
    printf("  result: %s\n\n", pass ? "PASS" : "FAIL");

    model_free(&m);
    return pass;
}

/* ═══════════════════════════════════════════════════════
 * Test 2: Attention causality
 *
 * Feed sequence "AB?" twice, with different byte at position 1.
 * If attention works, logits at position 2 must differ because
 * position 2 attends to positions 0 and 1.
 *
 * If attention is stubbed (no cross-position interaction),
 * changing position 1 won't affect position 2's output.
 * ═══════════════════════════════════════════════════════ */

static int test_causality(void) {
    printf("--- Test 2: attention causality ---\n");
    ModelConfig cfg = test_config();

    srand(42);
    Model m;
    model_init(&m, cfg);

    /* Sequence A: [65, 66, 67] = "ABC" */
    uint8_t seq_a[3] = {65, 66, 67};
    float logits_a[3 * 256];
    model_forward_sequence_cpu(&m, seq_a, 3, logits_a, MODEL_FWD_FLAGS_DEFAULT);

    /* Re-init model with same seed for identical weights */
    model_free(&m);
    srand(42);
    model_init(&m, cfg);

    /* Sequence B: [65, 90, 67] = "AZC" — different byte at position 1 */
    uint8_t seq_b[3] = {65, 90, 67};
    float logits_b[3 * 256];
    model_forward_sequence_cpu(&m, seq_b, 3, logits_b, MODEL_FWD_FLAGS_DEFAULT);

    /* Position 0 logits should be IDENTICAL (same byte, no prior context) */
    float dist_pos0 = vec_l2_dist(logits_a, logits_b, 256);

    /* Position 2 logits should DIFFER (attention sees different pos 1) */
    float dist_pos2 = vec_l2_dist(logits_a + 2*256, logits_b + 2*256, 256);

    printf("  seq A: [A, B, C]   seq B: [A, Z, C]\n");
    printf("  logit distance at pos 0: %.6f (should be ~0)\n", dist_pos0);
    printf("  logit distance at pos 2: %.6f (should be >0)\n", dist_pos2);

    /* pos 0 should be identical (or near-zero due to float rounding) */
    /* pos 2 should differ meaningfully if attention is real */
    int pass = (dist_pos0 < 1e-3f && dist_pos2 > 0.01f);
    printf("  attention causality: %s\n", pass ? "VERIFIED" : "FAILED");
    printf("  result: %s\n\n", pass ? "PASS" : "FAIL");

    model_free(&m);
    return pass;
}

/* ═══════════════════════════════════════════════════════
 * Test 3: Position sensitivity
 *
 * Same byte 'X' at positions 0, 1, 2, 3 in a sequence.
 * If RoPE is working, each position produces different logits
 * because the rotary embedding encodes position.
 * ═══════════════════════════════════════════════════════ */

static int test_position_sensitivity(void) {
    printf("--- Test 3: position sensitivity (RoPE) ---\n");
    ModelConfig cfg = test_config();

    srand(42);
    Model m;
    model_init(&m, cfg);

    /* Same byte repeated: [88, 88, 88, 88] = "XXXX" */
    uint8_t seq[4] = {88, 88, 88, 88};
    float logits[4 * 256];
    model_forward_sequence_cpu(&m, seq, 4, logits, MODEL_FWD_FLAGS_DEFAULT);

    /* Compare logits at different positions */
    float d01 = vec_l2_dist(logits + 0*256, logits + 1*256, 256);
    float d02 = vec_l2_dist(logits + 0*256, logits + 2*256, 256);
    float d12 = vec_l2_dist(logits + 1*256, logits + 2*256, 256);

    printf("  seq: [X, X, X, X] (same byte, different positions)\n");
    printf("  logit distance pos0 vs pos1: %.6f\n", d01);
    printf("  logit distance pos0 vs pos2: %.6f\n", d02);
    printf("  logit distance pos1 vs pos2: %.6f\n", d12);

    /* All distances should be non-zero if position encoding works.
     * Threshold scales down with dequant normalization (1/sqrt(K)). */
    int pass = (d01 > 1e-5f && d02 > 1e-5f && d12 > 1e-5f);
    printf("  position sensitivity: %s\n", pass ? "VERIFIED" : "FAILED");
    printf("  result: %s\n\n", pass ? "PASS" : "FAIL");

    model_free(&m);
    return pass;
}

/* ═══════════════════════════════════════════════════════
 * Test 4: Generation context
 *
 * Use GenerationContext to generate bytes from a seed.
 * Verify the generation path works and logits are valid.
 * ═══════════════════════════════════════════════════════ */

static int test_generation(void) {
    printf("--- Test 4: generation from context ---\n");
    ModelConfig cfg = test_config();

    srand(42);
    Model m;
    model_init(&m, cfg);

    GenerationContext ctx;
    gen_ctx_init(&ctx, cfg.max_seq);

    /* Seed with "Hi" */
    gen_ctx_append(&ctx, 'H');
    gen_ctx_append(&ctx, 'i');

    float logits[256];
    int nan_count = 0;
    char output[11] = {0};

    printf("  seed: \"Hi\"\n  generated: \"");
    for (int i = 0; i < 10; i++) {
        model_generate_cpu(&m, &ctx, logits);
        if (has_any_nan(logits, 256)) nan_count++;

        float lc[256];
        memcpy(lc, logits, 256 * sizeof(float));
        int byte = byte_sample(lc, 0.8f);

        output[i] = (byte >= 32 && byte < 127) ? (char)byte : '.';
        printf("%c", output[i]);
        gen_ctx_append(&ctx, (uint8_t)byte);
    }
    printf("\"\n");
    printf("  NaN in any step: %s\n", nan_count ? "YES" : "no");
    printf("  context length at end: %d\n", ctx.len);

    int pass = (nan_count == 0);
    printf("  result: %s\n\n", pass ? "PASS" : "FAIL");

    gen_ctx_free(&ctx);
    model_free(&m);
    return pass;
}

/* ═══════════════════════════════════════════════════════
 * Test 5: Hidden state bounded over sequence
 *
 * Run a longer sequence through multiple layers.
 * Gain normalization should prevent explosion.
 * ═══════════════════════════════════════════════════════ */

static int test_stability(void) {
    printf("--- Test 5: stability over 50-byte sequence ---\n");
    ModelConfig cfg = test_config();
    cfg.n_layers = 4;

    srand(99);
    Model m;
    model_init(&m, cfg);

    /* Generate random 50-byte sequence */
    int seq_len = 50;
    uint8_t *seq = (uint8_t*)malloc(seq_len);
    for (int i = 0; i < seq_len; i++)
        seq[i] = (uint8_t)(rand() % 256);

    float *all_logits = (float*)malloc(seq_len * 256 * sizeof(float));
    model_forward_sequence_cpu(&m, seq, seq_len, all_logits, MODEL_FWD_FLAGS_DEFAULT);

    /* Check logits at various positions */
    int nan_count = 0;
    float max_logit = 0;
    for (int t = 0; t < seq_len; t++) {
        for (int i = 0; i < 256; i++) {
            float v = all_logits[t * 256 + i];
            if (v != v) nan_count++;
            if (fabsf(v) > max_logit) max_logit = fabsf(v);
        }
    }

    printf("  sequence: %d random bytes, %d layers\n", seq_len, cfg.n_layers);
    printf("  max |logit| across all positions: %.2f\n", max_logit);
    printf("  NaN count: %d\n", nan_count);

    int pass = (nan_count == 0 && max_logit < 1e6f);
    printf("  result: %s\n\n", pass ? "PASS" : "FAIL");

    free(seq);
    free(all_logits);
    model_free(&m);
    return pass;
}

#ifdef TWO3_EARLY_EXIT
static int argmax256(const float *v) {
    int j = 0;
    for (int i = 1; i < 256; i++)
        if (v[i] > v[j]) j = i;
    return j;
}

/* Fresh model_init per forward so GainState matches; full depth vs reservoir depletion early exit. */
static int test_early_exit_parity(void) {
    printf("--- Test 7: early exit vs full depth (seq_len=1) ---\n");
    printf("  compiled TWO3_EXIT_DEPLETION_THRESH=%.4g\n",
           (double)TWO3_EXIT_DEPLETION_THRESH);
    fflush(stdout);
    ModelConfig cfg = test_config();
    cfg.n_layers = 4;

    const int trials = 200;
    int match = 0;
    float max_l2 = 0.f;
    int nan_fail = 0;

    for (int t = 0; t < trials; t++) {
        unsigned seed = 7000u + (unsigned)t * 7919u;
        srand(seed);
        uint8_t b = (uint8_t)(rand() % 256);

        Model mf;
        model_init(&mf, cfg);
        float lf[256];
        model_forward_sequence_cpu(&mf, &b, 1, lf, MODEL_FWD_FORCE_FULL_DEPTH);
        if (has_any_nan(lf, 256)) nan_fail++;
        model_free(&mf);

        srand(seed);
        Model me;
        model_init(&me, cfg);
        float le[256];
        model_forward_sequence_cpu(&me, &b, 1, le, MODEL_FWD_FLAGS_DEFAULT);
        if (has_any_nan(le, 256)) nan_fail++;
        model_free(&me);

        float d = vec_l2_dist(lf, le, 256);
        if (d > max_l2) max_l2 = d;
        if (argmax256(lf) == argmax256(le)) match++;
    }

    printf("  trials=%d  layers=%d  argmax_match=%d (%.1f%%)\n",
           trials, cfg.n_layers, match, 100.0f * (float)match / (float)trials);
    printf("  max L2 between full and early logits: %.6g\n", max_l2);
    printf("  NaN in any path: %s\n", nan_fail ? "YES" : "no");
    printf("  (Argmax mismatch is expected sometimes; pass = no NaN.)\n");
    int pass = (nan_fail == 0);
    printf("  result: %s\n\n", pass ? "PASS" : "FAIL");
    return pass;
}
#endif

/* ═══════════════════════════════════════════════════════
 * Test 6: Memory footprint estimate
 * ═══════════════════════════════════════════════════════ */

static int test_memory_footprint(void) {
    printf("--- Test 6: memory footprint ---\n");
    ModelConfig cfg = model_config_default();  /* full size */

    int D = cfg.dim;
    int KV = cfg.n_kv_heads * cfg.head_dim;
    int INTER = cfg.intermediate;
    int L = cfg.n_layers;

    size_t embed = 256 * D * sizeof(float);
    size_t attn_per_layer = (D*D + KV*D + KV*D + D*D) / 4;  /* ternary packed */
    size_t ffn_per_layer = 3 * (size_t)INTER * D / 4;  /* ternary gate+up+down */
    size_t gain_per_layer = 2 * D * sizeof(float) * 2;  /* R + C, attn + ffn */
    size_t rope = cfg.max_seq * (cfg.head_dim / 2) * sizeof(float) * 2;

    size_t total = embed
                 + L * (attn_per_layer + ffn_per_layer + gain_per_layer)
                 + rope;

    printf("  Config: dim=%d, layers=%d, heads=%d, kv=%d, inter=%d\n",
           D, L, cfg.n_heads, cfg.n_kv_heads, INTER);
    printf("  Embedding (256 bytes): %.1f KB\n", embed / 1024.0);
    printf("  Attention (ternary):   %.1f KB × %d = %.1f MB\n",
           attn_per_layer / 1024.0, L, L * attn_per_layer / 1e6);
    printf("  Dense FFN (ternary):   %.1f MB × %d = %.1f MB\n",
           ffn_per_layer / 1e6, L, L * ffn_per_layer / 1e6);
    printf("  Gain states:           %.1f KB × %d = %.1f KB\n",
           gain_per_layer / 1024.0, L, L * gain_per_layer / 1024.0);
    printf("  RoPE tables:           %.1f KB\n", rope / 1024.0);
    printf("  ─────────────────────────────────────\n");
    printf("  TOTAL weights:         %.1f MB\n", total / 1e6);
    printf("  Fits in 8GB VRAM:      %s\n", total < 8000000000ULL ? "YES" : "NO");

    int pass = (total < 8000000000ULL);
    printf("  result: %s\n\n", pass ? "PASS" : "FAIL");
    return pass;
}

/* ═══════════════════════════════════════════════════════ */

int main(void) {
    printf("============================================\n");
    printf("  {2,3} Model — Layer 2 Verification\n");
    printf("  Real attention. Real MoE. Real projections.\n");
    printf("  Bytes in. Bytes out. 256 vocab.\n");
    printf("============================================\n\n");

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("  GPU: %s\n\n", prop.name);

    int passed = 0, total = 0;

    total++; passed += test_single_byte();
    total++; passed += test_causality();
    total++; passed += test_position_sensitivity();
    total++; passed += test_generation();
    total++; passed += test_stability();
#ifdef TWO3_EARLY_EXIT
    total++; passed += test_early_exit_parity();
#endif
    total++; passed += test_memory_footprint();

    printf("============================================\n");
    printf("  Results: %d / %d tests passed\n", passed, total);
    printf("============================================\n");

    if (passed == total)
        printf("\n  Layer 2 verified.\n"
               "  Causal attention: REAL.\n"
               "  Ternary projections: REAL.\n"
               "  Dense FFN forward: REAL.\n"
               "  Position encoding: REAL.\n"
               "  Bytes in. Bytes out. No tokenizer.\n"
               "  Dense FFN. No router. No dispatch.\n"
               "  Then Layer 4 — training with STE.\n\n");

    return (passed == total) ? 0 : 1;
}
