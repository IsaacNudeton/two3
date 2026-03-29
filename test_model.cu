/*
 * test_model.cu — Full model integration test
 *
 * byte → embed → 12 layers → logits → sample byte
 * No tokenizer. No BPE. Raw bytes in, raw bytes out.
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include "model.h"

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        printf("CUDA error: %s\n", cudaGetErrorString(err)); exit(1); \
    } \
} while(0)

static int test_byte_roundtrip(void) {
    printf("--- Test: byte in → logits out (no NaN, no overflow) ---\n");
    ModelConfig cfg = model_config_default();
    cfg.dim = 256;       /* smaller for test speed */
    cfg.n_heads = 4;
    cfg.n_kv_heads = 2;
    cfg.head_dim = 64;
    cfg.intermediate = 512;
    cfg.n_layers = 4;    /* fewer for test speed */

    srand(42);
    Model m;
    model_init(&m, cfg);

    float *hidden = (float*)malloc(cfg.dim * sizeof(float));
    float logits[256];

    /* Feed byte 'H' (72) */
    model_forward_cpu(&m, 72, 0, hidden, logits);

    /* Check: no NaN in logits */
    int has_nan = 0;
    float max_logit = -1e30f, min_logit = 1e30f;
    for (int i = 0; i < 256; i++) {
        if (logits[i] != logits[i]) has_nan = 1;
        if (logits[i] > max_logit) max_logit = logits[i];
        if (logits[i] < min_logit) min_logit = logits[i];
    }

    printf("  input: byte %d ('%c')\n", 72, 72);
    printf("  logit range: [%.4f, %.4f]\n", min_logit, max_logit);
    printf("  NaN: %s\n", has_nan ? "YES" : "no");

    /* Sample a byte */
    float logits_copy[256];
    for (int i = 0; i < 256; i++) logits_copy[i] = logits[i];
    int sampled = byte_sample(logits_copy, 1.0f);
    printf("  sampled output: byte %d ('%c')\n", sampled,
           (sampled >= 32 && sampled < 127) ? sampled : '?');

    int pass = (!has_nan && max_logit < 1e6f);
    printf("  result: %s\n\n", pass ? "PASS" : "FAIL");

    free(hidden);
    model_free(&m);
    return pass;
}

static int test_sequence_generation(void) {
    printf("--- Test: generate 20 bytes from 'Hello' ---\n");
    ModelConfig cfg = model_config_default();
    cfg.dim = 256;
    cfg.n_heads = 4;
    cfg.n_kv_heads = 2;
    cfg.head_dim = 64;
    cfg.intermediate = 512;
    cfg.n_layers = 4;

    srand(42);
    Model m;
    model_init(&m, cfg);

    float *hidden = (float*)malloc(cfg.dim * sizeof(float));
    float logits[256];

    /* Seed with "Hello" */
    const char *seed = "Hello";
    printf("  seed: \"%s\"\n  generated: \"", seed);

    for (int i = 0; seed[i]; i++)
        model_forward_cpu(&m, (unsigned char)seed[i], i, hidden, logits);

    /* Generate 20 more bytes */
    int has_nan = 0;
    char output[21] = {0};
    int last_byte = (unsigned char)seed[4];
    for (int i = 0; i < 20; i++) {
        model_forward_cpu(&m, last_byte, 5 + i, hidden, logits);
        for (int j = 0; j < 256; j++)
            if (logits[j] != logits[j]) has_nan = 1;

        float lc[256];
        for (int j = 0; j < 256; j++) lc[j] = logits[j];
        last_byte = byte_sample(lc, 0.8f);
        output[i] = (last_byte >= 32 && last_byte < 127) ? last_byte : '.';
        printf("%c", output[i]);
    }
    printf("\"\n");
    printf("  NaN in any step: %s\n", has_nan ? "YES" : "no");

    int pass = !has_nan;
    printf("  result: %s\n\n", pass ? "PASS" : "FAIL");

    free(hidden);
    model_free(&m);
    return pass;
}

static int test_hidden_bounded(void) {
    printf("--- Test: hidden state bounded after 100 bytes ---\n");
    ModelConfig cfg = model_config_default();
    cfg.dim = 256;
    cfg.n_heads = 4;
    cfg.n_kv_heads = 2;
    cfg.head_dim = 64;
    cfg.intermediate = 512;
    cfg.n_layers = 12;  /* full depth */

    srand(99);
    Model m;
    model_init(&m, cfg);

    float *hidden = (float*)malloc(cfg.dim * sizeof(float));
    float logits[256];

    /* Feed 100 random bytes through 12 layers */
    float max_hidden = 0;
    for (int t = 0; t < 100; t++) {
        int byte_in = rand() % 256;
        model_forward_cpu(&m, byte_in, t, hidden, logits);

        float mag = 0;
        for (int i = 0; i < cfg.dim; i++)
            mag += hidden[i] * hidden[i];
        mag = sqrtf(mag);
        if (mag > max_hidden) max_hidden = mag;
    }

    printf("  max |hidden| across 100 bytes × 12 layers: %.2f\n", max_hidden);

    int pass = (max_hidden < 1e6f && max_hidden == max_hidden);
    printf("  result: %s\n\n", pass ? "PASS" : "FAIL");

    free(hidden);
    model_free(&m);
    return pass;
}

static int test_memory_footprint(void) {
    printf("--- Test: memory footprint ---\n");
    ModelConfig cfg = model_config_default();  /* full size */

    /* Compute sizes */
    int D = cfg.dim;
    int KV = cfg.n_kv_heads * cfg.head_dim;
    int INTER = cfg.intermediate;
    int L = cfg.n_layers;

    size_t embed = 256 * D * sizeof(float);
    size_t attn_per_layer = (D*D/4 + KV*D/4 + KV*D/4 + D*D/4);  /* ternary packed */
    size_t moe_router = D * MOE_NUM_EXPERTS * sizeof(float);
    /* Expert weights not allocated in test, estimate */
    size_t moe_experts = MOE_NUM_EXPERTS * 3 * INTER * D / 4;  /* ternary */
    size_t gain_per_layer = 2 * D * sizeof(float) * 2;  /* R + C, attn + ffn */
    size_t rope = cfg.max_seq * (cfg.head_dim / 2) * sizeof(float) * 2;

    size_t total = embed + L * (attn_per_layer + moe_router + moe_experts + gain_per_layer) + rope;

    printf("  Config: dim=%d, layers=%d, heads=%d, kv=%d, inter=%d\n",
           D, L, cfg.n_heads, cfg.n_kv_heads, INTER);
    printf("  Embedding (256 bytes): %.1f KB\n", embed / 1024.0);
    printf("  Attention (ternary):   %.1f KB × %d = %.1f MB\n",
           attn_per_layer / 1024.0, L, L * attn_per_layer / 1e6);
    printf("  MoE router (float):    %.1f KB × %d = %.1f KB\n",
           moe_router / 1024.0, L, L * moe_router / 1024.0);
    printf("  MoE experts (ternary): %.1f MB × %d = %.1f MB\n",
           moe_experts / 1e6, L, L * moe_experts / 1e6);
    printf("  Gain states:           %.1f KB × %d = %.1f KB\n",
           gain_per_layer / 1024.0, L, L * gain_per_layer / 1024.0);
    printf("  RoPE tables:           %.1f KB\n", rope / 1024.0);
    printf("  ─────────────────────────────────────\n");
    printf("  TOTAL:                 %.1f MB\n", total / 1e6);
    printf("  Fits in 8GB VRAM:      %s\n", total < 8000000000 ? "YES" : "NO");

    int pass = (total < 8000000000);
    printf("  result: %s\n\n", pass ? "PASS" : "FAIL");
    return pass;
}

int main(void) {
    printf("============================================\n");
    printf("  {2,3} Model — Full Integration\n");
    printf("  Bytes in. Bytes out. No tokenizer.\n");
    printf("  256 vocab. Ternary weights. Gain norm.\n");
    printf("============================================\n\n");

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("  GPU: %s\n\n", prop.name);

    int passed = 0, total = 0;
    total++; passed += test_byte_roundtrip();
    total++; passed += test_sequence_generation();
    total++; passed += test_hidden_bounded();
    total++; passed += test_memory_footprint();

    printf("============================================\n");
    printf("  Results: %d / %d tests passed\n", passed, total);
    printf("============================================\n");

    if (passed == total)
        printf("\n  Layer 2 complete. The model exists.\n"
               "  Bytes in. Bytes out. 256 vocab.\n"
               "  Ternary weights. Gain normalization.\n"
               "  No tokenizer. No BPE. No float multiply.\n"
               "  Next: Layer 4 — training with STE.\n\n");

    return (passed == total) ? 0 : 1;
}
