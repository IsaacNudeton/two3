/*
 * data.h — Text Data Loading for {2,3} Training
 *
 * No tokenizer. No BPE. No vocabulary file.
 * Text files are raw byte sequences. vocab = 256.
 *
 * Load one or more text files, chunk into fixed-length sequences,
 * shuffle, iterate. That's it.
 *
 * Usage:
 *   Dataset ds;
 *   dataset_init(&ds, 128);                  // seq_len = 128 bytes
 *   dataset_load_file(&ds, "data.txt");      // load text
 *   dataset_shuffle(&ds, 42);                // shuffle chunks
 *   for (int i = 0; i < ds.n_chunks; i++) {
 *       uint8_t *chunk = dataset_get(&ds, i); // [seq_len] bytes
 *       trainable_train_step(&tm, chunk, ds.seq_len);
 *   }
 *
 * Isaac & CC — March 2026
 */

#ifndef DATA_H
#define DATA_H

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ═══════════════════════════════════════════════════════
 * Dataset — chunked byte sequences from text files
 * ═══════════════════════════════════════════════════════ */

typedef struct {
    uint8_t  *data;       /* raw bytes, all files concatenated */
    size_t    total_bytes; /* total byte count */
    size_t    capacity;    /* allocated capacity */

    int      *chunk_offsets; /* start offset of each chunk */
    int       n_chunks;      /* number of seq_len chunks */
    int       chunk_cap;     /* allocated chunk capacity */

    int       seq_len;       /* bytes per training sequence */
} Dataset;

static void dataset_init(Dataset *ds, int seq_len) {
    ds->seq_len = seq_len;
    ds->total_bytes = 0;
    ds->capacity = 1024 * 1024;  /* 1MB initial */
    ds->data = (uint8_t*)malloc(ds->capacity);

    ds->n_chunks = 0;
    ds->chunk_cap = 1024;
    ds->chunk_offsets = (int*)malloc(ds->chunk_cap * sizeof(int));
}

static void dataset_free(Dataset *ds) {
    free(ds->data);
    free(ds->chunk_offsets);
    ds->data = NULL;
    ds->chunk_offsets = NULL;
}

/* Load a text file and append its bytes to the dataset.
 * Returns number of bytes loaded, or -1 on error. */
static int dataset_load_file(Dataset *ds, const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) {
        printf("[data] cannot open: %s\n", path);
        return -1;
    }

    /* Get file size */
    fseek(f, 0, SEEK_END);
    long fsize = ftell(f);
    fseek(f, 0, SEEK_SET);

    if (fsize <= 0) {
        fclose(f);
        printf("[data] empty file: %s\n", path);
        return 0;
    }

    /* Grow buffer if needed */
    while (ds->total_bytes + fsize > ds->capacity) {
        ds->capacity *= 2;
        ds->data = (uint8_t*)realloc(ds->data, ds->capacity);
    }

    /* Read */
    size_t read = fread(ds->data + ds->total_bytes, 1, fsize, f);
    fclose(f);

    size_t start = ds->total_bytes;
    ds->total_bytes += read;

    /* Create chunks from this file's data.
     * Chunks overlap by 1 byte (last byte of chunk N = first target of chunk N+1)
     * to maximize data usage. Stride = seq_len/2 for 50% overlap. */
    int stride = ds->seq_len / 2;
    if (stride < 1) stride = 1;

    for (size_t off = start; off + ds->seq_len <= ds->total_bytes; off += stride) {
        if (ds->n_chunks >= ds->chunk_cap) {
            ds->chunk_cap *= 2;
            ds->chunk_offsets = (int*)realloc(ds->chunk_offsets,
                                              ds->chunk_cap * sizeof(int));
        }
        ds->chunk_offsets[ds->n_chunks++] = (int)off;
    }

    printf("[data] loaded %s: %ld bytes, %d chunks (seq_len=%d, stride=%d)\n",
           path, (long)read, ds->n_chunks, ds->seq_len, stride);
    return (int)read;
}

/* Load multiple files by calling dataset_load_file repeatedly.
 * Directory globbing removed — use explicit file paths. */

/* Fisher-Yates shuffle of chunk order */
static void dataset_shuffle(Dataset *ds, unsigned int seed) {
    srand(seed);
    for (int i = ds->n_chunks - 1; i > 0; i--) {
        int j = rand() % (i + 1);
        int tmp = ds->chunk_offsets[i];
        ds->chunk_offsets[i] = ds->chunk_offsets[j];
        ds->chunk_offsets[j] = tmp;
    }
}

/* Get chunk i as a pointer to seq_len bytes */
static uint8_t* dataset_get(const Dataset *ds, int chunk_idx) {
    return ds->data + ds->chunk_offsets[chunk_idx];
}

/* Print dataset stats */
static void dataset_info(const Dataset *ds) {
    printf("[data] total: %zu bytes, %d chunks, seq_len=%d\n",
           ds->total_bytes, ds->n_chunks, ds->seq_len);
    if (ds->total_bytes > 0) {
        /* Show first 80 bytes as preview */
        printf("[data] preview: \"");
        int show = ds->total_bytes < 80 ? (int)ds->total_bytes : 80;
        for (int i = 0; i < show; i++) {
            uint8_t b = ds->data[i];
            if (b >= 32 && b < 127) printf("%c", b);
            else printf(".");
        }
        printf("...\"\n");
    }
}

#endif /* DATA_H */
