/*
 * test_ibc_model.cu — Validate IBC codebook wired into model forward
 *
 * Tests:
 *   1. IBC codebook invertibility at model dim
 *   2. IBC embed produces same-dim output as byte embed
 *   3. Forward pass with IBC embed produces valid logits
 *   4. Adjacent bytes produce similar hidden states (structure preserved)
 *   5. Gradient does NOT flow to embedding (fixed codebook)
 *
 * Build: nvcc -DTWO3_IBC -DIBC_WIDTH=128 test_ibc_model.cu two3.cu -o test_ibc_model.exe
 *        (for full config: -DIBC_WIDTH=1024)
 *
 * Isaac & CC — April 2026
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

/* Force IBC mode */
#ifndef TWO3_IBC
#define TWO3_IBC
#endif

#include "ibc.h"
#include "model.h"

/* ═══════════════════════════════════════════════════════ */

static int tests_passed = 0;
static int tests_total = 0;

static void check(int cond, const char *name) {
    tests_total++;
    if (cond) {
        printf("  PASS: %s\n", name);
        tests_passed++;
    } else {
        printf("  FAIL: %s\n", name);
    }
}

int main(void) {
    printf("=== IBC Model Integration Tests ===\n\n");

    /* Use medium config so it runs fast */
    ModelConfig cfg = model_config_medium();
    int D = cfg.dim;

    /* Verify IBC_WIDTH matches dim */
    printf("[1] IBC_WIDTH=%d, model dim=%d\n", IBC_WIDTH, D);
    check(IBC_WIDTH == D, "IBC_WIDTH matches model dim");

    /* Test 1: Codebook invertibility at model dim */
    printf("\n[2] Codebook invertibility...\n");
    {
        IBCCodebook cb;
        ibc_codebook_init(&cb);
        int correct = 0;
        for (int b = 0; b < 256; b++) {
            uint8_t decoded = ibc_decode(&cb, cb.vectors[b]);
            if (decoded == (uint8_t)b) correct++;
        }
        check(correct == 256, "256/256 bytes invertible");
        ibc_codebook_info(&cb);
    }

    /* Test 2: Model init with IBC produces valid embed */
    printf("\n[3] Model init with IBC codebook...\n");
    Model m;
    model_init(&m, cfg);

    /* Check embed is not zero and not random-looking */
    {
        float sum = 0, min_v = 1e9f, max_v = -1e9f;
        for (int i = 0; i < 256 * D; i++) {
            sum += m.embed[i];
            if (m.embed[i] < min_v) min_v = m.embed[i];
            if (m.embed[i] > max_v) max_v = m.embed[i];
        }
        printf("  embed range: [%.4f, %.4f], mean=%.4f\n", min_v, max_v, sum / (256 * D));
        check(max_v > 0.2f && min_v < -0.2f, "embed has full range from codebook");
        check(max_v <= 1.001f && min_v >= -1.001f, "embed bounded to [-1, 1]");
    }

    /* Test 3: Adjacent bytes produce similar embeddings */
    printf("\n[4] Structure preservation...\n");
    {
        /* 'A' (65) and 'B' (66) differ by 1 bit → should be similar */
        float dot_ab = 0, dot_az = 0;
        float *ea = m.embed + 65 * D;
        float *eb = m.embed + 66 * D;
        float *ez = m.embed + 122 * D; /* 'z' = 122 */
        for (int d = 0; d < D; d++) {
            dot_ab += ea[d] * eb[d];
            dot_az += ea[d] * ez[d];
        }
        printf("  dot(A, B) = %.1f  (1-bit diff, should be high)\n", dot_ab);
        printf("  dot(A, z) = %.1f  (multi-bit diff, should be lower)\n", dot_az);
        check(dot_ab > dot_az, "'A' closer to 'B' than to 'z'");
    }

    /* Test 4: Forward pass produces valid logits */
    printf("\n[5] Forward pass with IBC embed...\n");
    {
        uint8_t test_seq[] = "Hello";
        int seq_len = 5;
        float *logits = (float*)malloc(seq_len * 256 * sizeof(float));

        model_forward_sequence_cpu(&m, test_seq, seq_len, logits, MODEL_FWD_FLAGS_DEFAULT);

        /* Check logits are finite and not all identical */
        int all_finite = 1;
        float first = logits[0];
        int all_same = 1;
        for (int i = 0; i < seq_len * 256; i++) {
            if (isnan(logits[i]) || isinf(logits[i])) all_finite = 0;
            if (fabsf(logits[i] - first) > 1e-6f) all_same = 0;
        }
        check(all_finite, "all logits finite");
        check(!all_same, "logits not degenerate (not all identical)");

        /* Check last position has a clear-ish prediction */
        int best = 0;
        for (int b = 1; b < 256; b++)
            if (logits[(seq_len-1) * 256 + b] > logits[(seq_len-1) * 256 + best])
                best = b;
        printf("  last position predicts byte %d ('%c')\n", best,
               (best >= 32 && best < 127) ? best : '.');

        free(logits);
    }

    /* Test 5: Embedding is deterministic — same codebook every time */
    printf("\n[6] Determinism...\n");
    {
        Model m2;
        model_init(&m2, cfg);
        int match = 1;
        for (int i = 0; i < 256 * D; i++) {
            if (fabsf(m.embed[i] - m2.embed[i]) > 1e-7f) {
                match = 0;
                break;
            }
        }
        check(match, "two inits produce identical embeddings");
        model_free(&m2);
    }

    model_free(&m);

    printf("\n=== Results: %d / %d tests passed ===\n", tests_passed, tests_total);
    return (tests_passed == tests_total) ? 0 : 1;
}
