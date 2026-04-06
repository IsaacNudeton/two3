/* Quick test for binary matmul */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "binary.h"

int main(void) {
    printf("=== Binary Weight Tests ===\n\n");

    int rows = 4, cols = 8;
    float w_float[] = {
        /* row 0: connect dims 0,1,2,3 (first half) */
        0.9f, 0.8f, 0.7f, 0.6f, 0.1f, 0.2f, 0.3f, 0.4f,
        /* row 1: connect dims 4,5,6,7 (second half) */
        0.1f, 0.2f, 0.3f, 0.4f, 0.9f, 0.8f, 0.7f, 0.6f,
        /* row 2: connect all */
        0.9f, 0.9f, 0.9f, 0.9f, 0.9f, 0.9f, 0.9f, 0.9f,
        /* row 3: connect none */
        0.1f, 0.1f, 0.1f, 0.1f, 0.1f, 0.1f, 0.1f, 0.1f,
    };

    BinaryWeights bw = binary_pack_weights(w_float, rows, cols);
    binary_print_stats(&bw);

    /* Input: first half positive, second half negative */
    int8_t x_q[] = {10, 20, 30, 40, -10, -20, -30, -40};
    int32_t acc[4];
    binary_matmul_cpu(bw.packed, x_q, acc, rows, cols);

    printf("\nInput: [10 20 30 40 -10 -20 -30 -40]\n");
    printf("Row 0 (connect first half):  acc=%d  (expect 10+20+30+40=100)\n", acc[0]);
    printf("Row 1 (connect second half): acc=%d  (expect -10-20-30-40=-100)\n", acc[1]);
    printf("Row 2 (connect all):         acc=%d  (expect 0)\n", acc[2]);
    printf("Row 3 (connect none):        acc=%d  (expect 0)\n", acc[3]);

    int pass = (acc[0] == 100 && acc[1] == -100 && acc[2] == 0 && acc[3] == 0);
    printf("\nResult: %s\n", pass ? "PASS" : "FAIL");

    /* Test: density */
    printf("\nDensity: %.1f%% (expect 50%%)\n", 100.0f * bw.density);

    /* Test: sign through selection */
    printf("\nSign test: binary weight selecting negative dims produces negative output.\n");
    printf("Row 1 acc = %d (negative) — sign lives in the signal, not the weight.\n", acc[1]);

    binary_free_weights(&bw);
    return pass ? 0 : 1;
}
