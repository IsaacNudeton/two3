/*
 * ibc_precompute.c — Pre-compute onetwo_parse fingerprints for training
 *
 * For each position t in the corpus, computes the structural fingerprint
 * of the context window [max(0, t-WINDOW)..t]. Outputs a binary file
 * of [N × 512] bytes (N positions × 4096 bits each).
 *
 * The training loop mmap's this file and uses fingerprints as embeddings
 * instead of learned byte lookups.
 *
 * Usage: ibc_precompute <corpus.txt> <output.fp> [window_size]
 *
 * Isaac & CC — April 2026
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>

/* Include BitStream primitives and onetwo_encode */
#include "bitstream.h"

/* __builtin_popcount compatibility for MSVC */
#ifdef _MSC_VER
#include <intrin.h>
#define __builtin_popcount __popcnt
static inline int __builtin_popcountll(unsigned long long x) {
    return (int)__popcnt64(x);
}
#endif

#include "onetwo_encode.c"

#define DEFAULT_WINDOW  128   /* context window size in bytes */
#define FP_BYTES        512   /* 4096 bits = 512 bytes */

int main(int argc, char **argv) {
    if (argc < 3) {
        printf("Usage: ibc_precompute <corpus.txt> <output.fp> [window_size]\n");
        printf("  Computes onetwo_parse fingerprint at every byte position.\n");
        printf("  Output: [N × 512] bytes, one 4096-bit fingerprint per position.\n");
        return 1;
    }

    const char *corpus_path = argv[1];
    const char *output_path = argv[2];
    int window = argc > 3 ? atoi(argv[3]) : DEFAULT_WINDOW;

    /* Load corpus */
    FILE *f = fopen(corpus_path, "rb");
    if (!f) { fprintf(stderr, "Cannot open %s\n", corpus_path); return 1; }
    fseek(f, 0, SEEK_END);
    long fsize = ftell(f);
    fseek(f, 0, SEEK_SET);
    uint8_t *text = (uint8_t*)malloc(fsize);
    fread(text, 1, fsize, f);
    fclose(f);
    printf("[fp] loaded %s: %ld bytes\n", corpus_path, fsize);
    printf("[fp] window size: %d bytes\n", window);
    printf("[fp] output: %ld positions × %d bytes = %.1f MB\n",
           fsize, FP_BYTES, (double)fsize * FP_BYTES / 1e6);

    /* Open output file */
    FILE *out = fopen(output_path, "wb");
    if (!out) { fprintf(stderr, "Cannot open %s for writing\n", output_path); return 1; }

    /* Write header: magic + corpus_size + window_size + fp_bits */
    uint32_t magic = 0x46503031;  /* "FP01" */
    uint32_t corpus_size = (uint32_t)fsize;
    uint32_t win_size = (uint32_t)window;
    uint32_t fp_bits = 4096;
    fwrite(&magic, 4, 1, out);
    fwrite(&corpus_size, 4, 1, out);
    fwrite(&win_size, 4, 1, out);
    fwrite(&fp_bits, 4, 1, out);

    /* Pre-compute fingerprints */
    clock_t t_start = clock();
    int report_interval = (int)(fsize / 20);
    if (report_interval < 1000) report_interval = 1000;

    BitStream fp;
    for (long t = 0; t < fsize; t++) {
        /* Context window: [max(0, t-window+1) .. t] inclusive */
        long start = t - window + 1;
        if (start < 0) start = 0;
        int ctx_len = (int)(t - start + 1);

        onetwo_parse(text + start, ctx_len, &fp);

        /* Write 512 bytes (4096 bits) */
        fwrite(fp.w, 1, FP_BYTES, out);

        if (t % report_interval == 0 && t > 0) {
            double elapsed = (double)(clock() - t_start) / CLOCKS_PER_SEC;
            double rate = t / elapsed;
            double remaining = (fsize - t) / rate;
            printf("[fp] position %ld / %ld (%.0f%%) — %.0f pos/sec, ~%.0fs remaining\n",
                   t, fsize, 100.0 * t / fsize, rate, remaining);
            fflush(stdout);
        }
    }

    fclose(out);
    double total_time = (double)(clock() - t_start) / CLOCKS_PER_SEC;

    printf("[fp] done: %ld fingerprints in %.1fs (%.0f pos/sec)\n",
           fsize, total_time, fsize / total_time);
    printf("[fp] output: %s (%.1f MB)\n",
           output_path, (double)fsize * FP_BYTES / 1e6 + 16);

    free(text);
    return 0;
}
