/*
 * ibc_compile.cu — Phase 1: .IBC Corpus Compiler
 *
 * Takes raw text, counts distributional statistics, builds
 * the .xyzt pinout table and compresses training data into
 * .ibc format for VRAM-resident training.
 *
 * Pipeline:
 *   1. Tokenize into words (whitespace + punctuation split)
 *   2. Count unigrams, bigrams → PMI for X-pins
 *   3. Brown clustering on PMI vectors → Y-pins (synonymy groups)
 *   4. Transitional probability drops → Z-pins (phrase boundaries)
 *   5. Build pinout table: word → {X, Y, Z, T} pin vector
 *   6. Compress sequences into .ibc opcodes
 *
 * Output:
 *   .xyzt  — pinout table (word → pin vector mapping)
 *   .ibc   — compressed training data (opcodes + pin vectors)
 *
 * Isaac & CC — April 2026
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <ctype.h>

/* ═══════════════════════════════════════════════════════
 * Configuration
 * ═══════════════════════════════════════════════════════ */

#define MAX_VOCAB       8192    /* max unique words */
#define MAX_WORD_LEN    64      /* max chars per word */
#define PMI_WINDOW      5       /* co-occurrence window for X-pins */
#define N_CLUSTERS      64      /* Brown clusters for Y-pins */
#define PIN_DIM         128     /* total pin vector dimension */
#define X_PINS          48      /* PMI-derived co-occurrence features */
#define Y_PINS          32      /* cluster/synonymy features */
#define Z_PINS          32      /* boundary/constituency features */
#define T_PINS          16      /* positional features */

/* ═══════════════════════════════════════════════════════
 * Vocabulary
 * ═══════════════════════════════════════════════════════ */

typedef struct {
    char word[MAX_WORD_LEN];
    int  count;
    int  id;
} VocabEntry;

typedef struct {
    VocabEntry entries[MAX_VOCAB];
    int size;
    int total_tokens;   /* total word occurrences in corpus */
} Vocabulary;

/* Simple hash for word lookup */
static unsigned word_hash(const char *w) {
    unsigned h = 5381;
    while (*w) h = h * 33 + (unsigned char)*w++;
    return h % MAX_VOCAB;
}

static int vocab_find(Vocabulary *v, const char *word) {
    unsigned h = word_hash(word);
    for (int i = 0; i < MAX_VOCAB; i++) {
        int idx = (h + i) % MAX_VOCAB;
        if (v->entries[idx].word[0] == '\0') return -1;
        if (strcmp(v->entries[idx].word, word) == 0) return idx;
    }
    return -1;
}

static int vocab_add(Vocabulary *v, const char *word) {
    unsigned h = word_hash(word);
    for (int i = 0; i < MAX_VOCAB; i++) {
        int idx = (h + i) % MAX_VOCAB;
        if (v->entries[idx].word[0] == '\0') {
            strncpy(v->entries[idx].word, word, MAX_WORD_LEN - 1);
            v->entries[idx].count = 1;
            v->entries[idx].id = v->size;
            v->size++;
            return idx;
        }
        if (strcmp(v->entries[idx].word, word) == 0) {
            v->entries[idx].count++;
            return idx;
        }
    }
    return -1;  /* full */
}

/* ═══════════════════════════════════════════════════════
 * Tokenizer — simple whitespace + punctuation split
 * ═══════════════════════════════════════════════════════ */

typedef struct {
    int *token_ids;     /* sequence of vocab indices */
    int  length;
} TokenSequence;

static int is_word_char(char c) {
    return isalnum((unsigned char)c) || c == '\'' || c == '-';
}

static TokenSequence tokenize(const char *text, int text_len, Vocabulary *vocab) {
    TokenSequence seq;
    seq.token_ids = (int*)malloc(text_len * sizeof(int));  /* worst case: 1 token per char */
    seq.length = 0;

    int i = 0;
    char word[MAX_WORD_LEN];

    while (i < text_len) {
        /* Skip whitespace */
        while (i < text_len && isspace((unsigned char)text[i])) i++;
        if (i >= text_len) break;

        /* Extract word or punctuation */
        int wlen = 0;
        if (is_word_char(text[i])) {
            while (i < text_len && is_word_char(text[i]) && wlen < MAX_WORD_LEN - 1) {
                word[wlen++] = tolower((unsigned char)text[i++]);
            }
        } else {
            /* Single punctuation character as its own token */
            word[wlen++] = text[i++];
        }
        word[wlen] = '\0';

        int idx = vocab_add(vocab, word);
        if (idx >= 0) {
            seq.token_ids[seq.length++] = idx;
            vocab->total_tokens++;
        }
    }

    return seq;
}

/* ═══════════════════════════════════════════════════════
 * X-Pins: PMI co-occurrence features
 *
 * For each word, compute PMI with its top-K co-occurring
 * words within a window. The PMI vector IS the X-pin.
 * ═══════════════════════════════════════════════════════ */

typedef struct {
    float *pmi;         /* [vocab_size × X_PINS] — top-K PMI features per word */
    int   *top_k_ids;   /* [vocab_size × X_PINS] — which words are the top-K */
} PMITable;

static void compute_pmi(Vocabulary *vocab, TokenSequence *seq, PMITable *pmi_out) {
    int V = vocab->size;
    int N = seq->length;

    /* Count co-occurrences within window */
    int *cooccur = (int*)calloc((size_t)V * V, sizeof(int));

    for (int i = 0; i < N; i++) {
        int w1 = seq->token_ids[i];
        for (int j = i + 1; j < N && j <= i + PMI_WINDOW; j++) {
            int w2 = seq->token_ids[j];
            cooccur[w1 * V + w2]++;
            cooccur[w2 * V + w1]++;
        }
    }

    /* Compute PMI: log(P(w1,w2) / (P(w1) * P(w2))) */
    pmi_out->pmi = (float*)calloc((size_t)V * X_PINS, sizeof(float));
    pmi_out->top_k_ids = (int*)calloc((size_t)V * X_PINS, sizeof(int));

    float total_pairs = 0;
    for (int i = 0; i < V; i++)
        for (int j = 0; j < V; j++)
            total_pairs += cooccur[i * V + j];
    if (total_pairs < 1) total_pairs = 1;

    for (int w = 0; w < V; w++) {
        /* Find top-K co-occurring words by PMI */
        float *scores = (float*)malloc(V * sizeof(float));
        float p_w = (float)vocab->entries[w].count / (float)N;
        if (p_w < 1e-10f) p_w = 1e-10f;

        for (int j = 0; j < V; j++) {
            float p_j = (float)vocab->entries[j].count / (float)N;
            if (p_j < 1e-10f) p_j = 1e-10f;
            float p_wj = (float)cooccur[w * V + j] / total_pairs;
            if (p_wj < 1e-10f) {
                scores[j] = 0.0f;
            } else {
                scores[j] = logf(p_wj / (p_w * p_j));
            }
        }

        /* Extract top-K */
        for (int k = 0; k < X_PINS && k < V; k++) {
            int best = 0;
            for (int j = 1; j < V; j++) {
                if (scores[j] > scores[best]) best = j;
            }
            pmi_out->pmi[w * X_PINS + k] = scores[best];
            pmi_out->top_k_ids[w * X_PINS + k] = best;
            scores[best] = -1e30f;  /* exclude from next round */
        }
        free(scores);
    }

    free(cooccur);
}

/* ═══════════════════════════════════════════════════════
 * Y-Pins: Brown clustering (simplified)
 *
 * Cluster words by PMI similarity. Words in the same
 * cluster are "interchangeable" — Y-bus.
 * ═══════════════════════════════════════════════════════ */

static void brown_cluster(Vocabulary *vocab, PMITable *pmi, int *cluster_ids) {
    int V = vocab->size;

    /* Simple k-means on PMI vectors */
    float *centroids = (float*)calloc((size_t)N_CLUSTERS * X_PINS, sizeof(float));

    /* Init centroids from first N_CLUSTERS words */
    for (int c = 0; c < N_CLUSTERS && c < V; c++)
        memcpy(centroids + c * X_PINS, pmi->pmi + c * X_PINS, X_PINS * sizeof(float));

    /* K-means iterations */
    for (int iter = 0; iter < 20; iter++) {
        /* Assign */
        for (int w = 0; w < V; w++) {
            float best_dist = 1e30f;
            int best_c = 0;
            for (int c = 0; c < N_CLUSTERS; c++) {
                float dist = 0;
                for (int k = 0; k < X_PINS; k++) {
                    float d = pmi->pmi[w * X_PINS + k] - centroids[c * X_PINS + k];
                    dist += d * d;
                }
                if (dist < best_dist) {
                    best_dist = dist;
                    best_c = c;
                }
            }
            cluster_ids[w] = best_c;
        }

        /* Update centroids */
        int *counts = (int*)calloc(N_CLUSTERS, sizeof(int));
        memset(centroids, 0, (size_t)N_CLUSTERS * X_PINS * sizeof(float));
        for (int w = 0; w < V; w++) {
            int c = cluster_ids[w];
            counts[c]++;
            for (int k = 0; k < X_PINS; k++)
                centroids[c * X_PINS + k] += pmi->pmi[w * X_PINS + k];
        }
        for (int c = 0; c < N_CLUSTERS; c++) {
            if (counts[c] > 0)
                for (int k = 0; k < X_PINS; k++)
                    centroids[c * X_PINS + k] /= counts[c];
        }
        free(counts);
    }

    free(centroids);
}

/* ═══════════════════════════════════════════════════════
 * Z-Pins: Transitional probability boundaries
 *
 * Z fires when P(w_t | w_{t-1}) drops sharply — phrase
 * boundary. Simple: compute bigram probability, mark
 * positions where it drops below threshold.
 * ═══════════════════════════════════════════════════════ */

static void compute_z_pins(
    Vocabulary *vocab, TokenSequence *seq,
    float *z_strength   /* [seq->length] boundary strength at each position */
) {
    int N = seq->length;
    int V = vocab->size;

    /* Count bigrams */
    int *bigram = (int*)calloc((size_t)V * V, sizeof(int));
    for (int i = 0; i < N - 1; i++)
        bigram[seq->token_ids[i] * V + seq->token_ids[i + 1]]++;

    /* Compute transition probability P(w_t | w_{t-1}) */
    z_strength[0] = 1.0f;  /* first position is always a boundary */
    for (int i = 1; i < N; i++) {
        int prev = seq->token_ids[i - 1];
        int curr = seq->token_ids[i];
        int prev_count = vocab->entries[prev].count;
        if (prev_count < 1) prev_count = 1;
        float p_trans = (float)bigram[prev * V + curr] / (float)prev_count;

        /* Boundary strength = 1 - P(transition) */
        z_strength[i] = 1.0f - p_trans;
    }

    free(bigram);
}

/* ═══════════════════════════════════════════════════════
 * Pinout Table: word → {X, Y, Z, T} pin vector
 * ═══════════════════════════════════════════════════════ */

typedef struct {
    float *pins;        /* [vocab_size × PIN_DIM] */
    int    vocab_size;
    int    pin_dim;
    char  *words;       /* [vocab_size × MAX_WORD_LEN] for reverse lookup */
} PinoutTable;

static void build_pinout_table(
    Vocabulary *vocab, PMITable *pmi, int *cluster_ids,
    PinoutTable *table
) {
    int V = vocab->size;
    table->vocab_size = V;
    table->pin_dim = PIN_DIM;
    table->pins = (float*)calloc((size_t)V * PIN_DIM, sizeof(float));
    table->words = (char*)calloc((size_t)V * MAX_WORD_LEN, sizeof(char));

    for (int w = 0; w < V; w++) {
        float *pin = table->pins + w * PIN_DIM;

        /* X-pins [0..X_PINS): normalized PMI features */
        float x_norm = 0;
        for (int k = 0; k < X_PINS; k++) x_norm += pmi->pmi[w * X_PINS + k] * pmi->pmi[w * X_PINS + k];
        x_norm = sqrtf(x_norm + 1e-10f);
        for (int k = 0; k < X_PINS; k++)
            pin[k] = pmi->pmi[w * X_PINS + k] / x_norm;

        /* Y-pins [X_PINS..X_PINS+Y_PINS): one-hot cluster ID + noise */
        int c = cluster_ids[w];
        pin[X_PINS + (c % Y_PINS)] = 1.0f;

        /* Z-pins: filled per-position during sequence compilation */
        /* T-pins: filled per-position during sequence compilation */

        /* Store word for reverse lookup */
        strncpy(table->words + w * MAX_WORD_LEN, vocab->entries[w].word, MAX_WORD_LEN - 1);
    }
}

/* ═══════════════════════════════════════════════════════
 * .IBC Opcodes — compressed training format
 * ═══════════════════════════════════════════════════════ */

#define IBC_RAW         0x00    /* raw pin vector follows */
#define IBC_REPEAT      0x01    /* repeat previous vector N times */
#define IBC_NGRAM       0x02    /* common sequence by ID */
#define IBC_BOUNDARY    0x03    /* phrase boundary marker (Z fires) */
#define IBC_END         0xFF    /* end of sequence */

typedef struct {
    uint8_t *data;
    int      size;
    int      capacity;
} IBCBuffer;

static void ibc_init(IBCBuffer *buf) {
    buf->capacity = 1024 * 1024;
    buf->data = (uint8_t*)malloc(buf->capacity);
    buf->size = 0;
}

static void ibc_write(IBCBuffer *buf, const void *data, int len) {
    while (buf->size + len > buf->capacity) {
        buf->capacity *= 2;
        buf->data = (uint8_t*)realloc(buf->data, buf->capacity);
    }
    memcpy(buf->data + buf->size, data, len);
    buf->size += len;
}

static void ibc_write_byte(IBCBuffer *buf, uint8_t b) {
    ibc_write(buf, &b, 1);
}

static void ibc_write_u16(IBCBuffer *buf, uint16_t v) {
    ibc_write(buf, &v, 2);
}

/* ═══════════════════════════════════════════════════════
 * .IBC File Format
 *
 * Header:
 *   4 bytes: magic "IBC1"
 *   4 bytes: vocab_size
 *   4 bytes: pin_dim
 *   vocab_size × pin_dim × 4 bytes: pinout table (float32)
 *   vocab_size × MAX_WORD_LEN bytes: word strings
 *
 * Body:
 *   stream of opcodes + payloads
 * ═══════════════════════════════════════════════════════ */

static void write_ibc_file(
    const char *path,
    PinoutTable *table,
    IBCBuffer *body
) {
    FILE *f = fopen(path, "wb");
    if (!f) { fprintf(stderr, "Cannot open %s\n", path); return; }

    /* Header */
    fwrite("IBC1", 1, 4, f);
    fwrite(&table->vocab_size, 4, 1, f);
    fwrite(&table->pin_dim, 4, 1, f);
    fwrite(table->pins, sizeof(float), table->vocab_size * table->pin_dim, f);
    fwrite(table->words, 1, table->vocab_size * MAX_WORD_LEN, f);

    /* Body */
    fwrite(body->data, 1, body->size, f);

    fclose(f);
    printf("[ibc] wrote %s: %d vocab, %d pin_dim, %d body bytes\n",
           path, table->vocab_size, table->pin_dim, body->size);
}

/* ═══════════════════════════════════════════════════════
 * Main — compile corpus to .ibc
 * ═══════════════════════════════════════════════════════ */

int main(int argc, char **argv) {
    if (argc < 2) {
        printf("Usage: ibc_compile <corpus.txt> [output.ibc]\n");
        return 1;
    }

    const char *corpus_path = argv[1];
    const char *output_path = argc > 2 ? argv[2] : "training.ibc";

    /* Load corpus */
    FILE *f = fopen(corpus_path, "rb");
    if (!f) { fprintf(stderr, "Cannot open %s\n", corpus_path); return 1; }
    fseek(f, 0, SEEK_END);
    long fsize = ftell(f);
    fseek(f, 0, SEEK_SET);
    char *text = (char*)malloc(fsize + 1);
    fread(text, 1, fsize, f);
    text[fsize] = '\0';
    fclose(f);
    printf("[ibc] loaded %s: %ld bytes\n", corpus_path, fsize);

    /* Phase 1: Tokenize + build vocabulary */
    Vocabulary vocab;
    memset(&vocab, 0, sizeof(vocab));
    TokenSequence seq = tokenize(text, (int)fsize, &vocab);
    printf("[ibc] vocab: %d unique words, %d total tokens\n", vocab.size, vocab.total_tokens);

    /* Phase 2: PMI for X-pins */
    printf("[ibc] computing PMI...\n");
    PMITable pmi;
    compute_pmi(&vocab, &seq, &pmi);

    /* Phase 3: Brown clustering for Y-pins */
    printf("[ibc] clustering...\n");
    int *cluster_ids = (int*)calloc(vocab.size, sizeof(int));
    brown_cluster(&vocab, &pmi, cluster_ids);

    /* Phase 4: Z-pins (boundary detection) */
    printf("[ibc] boundary detection...\n");
    float *z_strength = (float*)malloc(seq.length * sizeof(float));
    compute_z_pins(&vocab, &seq, z_strength);

    /* Phase 5: Build pinout table */
    printf("[ibc] building pinout table...\n");
    PinoutTable table;
    build_pinout_table(&vocab, &pmi, cluster_ids, &table);

    /* Phase 6: Compress to .ibc */
    printf("[ibc] compressing...\n");
    IBCBuffer body;
    ibc_init(&body);

    int prev_id = -1;
    int repeat_count = 0;

    for (int i = 0; i < seq.length; i++) {
        int wid = seq.token_ids[i];

        /* Check for repeats */
        if (wid == prev_id) {
            repeat_count++;
            continue;
        }

        /* Flush pending repeats */
        if (repeat_count > 0) {
            ibc_write_byte(&body, IBC_REPEAT);
            ibc_write_u16(&body, (uint16_t)repeat_count);
            repeat_count = 0;
        }

        /* Z-boundary marker */
        if (z_strength[i] > 0.8f) {
            ibc_write_byte(&body, IBC_BOUNDARY);
        }

        /* Write token */
        ibc_write_byte(&body, IBC_RAW);
        ibc_write_u16(&body, (uint16_t)wid);

        prev_id = wid;
    }

    /* Flush final repeats */
    if (repeat_count > 0) {
        ibc_write_byte(&body, IBC_REPEAT);
        ibc_write_u16(&body, (uint16_t)repeat_count);
    }
    ibc_write_byte(&body, IBC_END);

    /* Write .ibc file */
    write_ibc_file(output_path, &table, &body);

    /* Stats */
    float compression = (float)fsize / (float)(body.size + table.vocab_size * table.pin_dim * 4);
    printf("[ibc] compression: %.1fx (%ld bytes → %d header + %d body)\n",
           compression, fsize, table.vocab_size * table.pin_dim * 4, body.size);

    /* Count boundaries */
    int n_boundaries = 0;
    for (int i = 0; i < seq.length; i++)
        if (z_strength[i] > 0.8f) n_boundaries++;
    printf("[ibc] boundaries: %d (%.1f%% of tokens)\n",
           n_boundaries, 100.0f * n_boundaries / seq.length);

    /* Cluster stats */
    int *cluster_counts = (int*)calloc(N_CLUSTERS, sizeof(int));
    for (int w = 0; w < vocab.size; w++) cluster_counts[cluster_ids[w]]++;
    int nonempty = 0;
    for (int c = 0; c < N_CLUSTERS; c++) if (cluster_counts[c] > 0) nonempty++;
    printf("[ibc] clusters: %d non-empty / %d total\n", nonempty, N_CLUSTERS);
    free(cluster_counts);

    /* Cleanup */
    free(text);
    free(seq.token_ids);
    free(pmi.pmi);
    free(pmi.top_k_ids);
    free(cluster_ids);
    free(z_strength);
    free(table.pins);
    free(table.words);
    free(body.data);

    return 0;
}
