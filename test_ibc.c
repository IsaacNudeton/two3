/* Quick test for IBC codebook */
#include <stdio.h>
#include "ibc.h"

int main(void) {
    IBCCodebook cb;
    ibc_codebook_init(&cb);
    ibc_codebook_info(&cb);

    /* Test round-trip on some known bytes */
    uint8_t test_bytes[] = "Hello, World! The quick brown fox.";
    int len = 34;
    printf("\nRound-trip test: \"%.34s\"\n", test_bytes);

    int8_t encoded[34 * IBC_WIDTH];
    ibc_encode_sequence(&cb, test_bytes, len, encoded);

    printf("Decoded:         \"");
    int match = 0;
    for (int t = 0; t < len; t++) {
        uint8_t decoded = ibc_decode(&cb, encoded + t * IBC_WIDTH);
        printf("%c", decoded >= 32 && decoded < 127 ? (char)decoded : '.');
        if (decoded == test_bytes[t]) match++;
    }
    printf("\"\n");
    printf("Match: %d/%d (%.1f%%)\n", match, len, 100.0f * match / len);

    return 0;
}
