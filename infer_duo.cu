/*
 * infer_duo.cu — Prompt-based inference for the duo model
 *
 * Loads a checkpoint, takes an ISAAC: prompt, generates a CC: continuation.
 * Supports both single-prompt and batch-eval modes.
 *
 * Usage:
 *   infer_duo.exe <ckpt> --prompt "how should we handle X?"
 *   infer_duo.exe <ckpt> --eval corpus/corpus_val.txt
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <cuda_runtime.h>
#include "train.h"

static void sample_nucleus(const float *logits, int n, float temperature, float top_p,
                           uint8_t *out_byte) {
    /* Apply temperature */
    float max_l = logits[0];
    for (int i = 1; i < n; i++) if (logits[i] > max_l) max_l = logits[i];
    float sum_exp = 0.0f;
    float probs[256];
    for (int i = 0; i < n; i++) {
        probs[i] = expf((logits[i] - max_l) / temperature);
        sum_exp += probs[i];
    }
    for (int i = 0; i < n; i++) probs[i] /= sum_exp;

    /* Sort indices by prob (descending) — insertion sort on small array */
    int idx[256];
    for (int i = 0; i < n; i++) idx[i] = i;
    for (int i = 0; i < n - 1; i++)
        for (int j = i + 1; j < n; j++)
            if (probs[idx[j]] > probs[idx[i]]) {
                int tmp = idx[i]; idx[i] = idx[j]; idx[j] = tmp;
            }

    /* Accumulate until top_p mass reached */
    float cumsum = 0.0f;
    int nucleus = 0;
    for (int i = 0; i < n; i++) {
        cumsum += probs[idx[i]];
        nucleus++;
        if (cumsum >= top_p) break;
    }

    /* Sample from nucleus */
    float r = (float)rand() / (float)RAND_MAX * cumsum;
    float running = 0.0f;
    int pick = idx[0];
    for (int i = 0; i < nucleus; i++) {
        running += probs[idx[i]];
        if (running >= r) { pick = idx[i]; break; }
    }
    *out_byte = (uint8_t)pick;
}

static void generate(TrainableModel *tm, const char *prompt, int max_new, int max_ctx,
                     float temperature, float top_p) {
    int prompt_len = (int)strlen(prompt);
    if (prompt_len == 0) return;

    int total_max = prompt_len + max_new;
    if (total_max > max_ctx) total_max = max_ctx;

    uint8_t *buf = (uint8_t*)malloc(total_max);
    int len = prompt_len > max_ctx ? max_ctx : prompt_len;
    memcpy(buf, prompt + (prompt_len - len), len);

    /* Print prompt as-is */
    fwrite(prompt, 1, prompt_len, stdout);
    fflush(stdout);

    int generated = 0;
    while (generated < max_new && len < max_ctx) {
        /* Allocate logits for the whole sequence */
        float *logits = (float*)malloc(len * 256 * sizeof(float));
        model_forward_sequence_cpu(&tm->model, buf, len, logits, MODEL_FWD_FLAGS_DEFAULT);

        /* Sample next byte from last-position logits */
        uint8_t next;
        sample_nucleus(logits + (len - 1) * 256, 256, temperature, top_p, &next);
        free(logits);

        /* Stop on newline followed by "ISAAC:" marker start — simple heuristic */
        if (next == '\n' && generated > 0) {
            /* Check if previous bytes form end of assistant turn */
            if (generated >= 1 && buf[len - 1] == '\n') {
                /* Double newline — end of turn */
                putchar('\n');
                break;
            }
        }

        putchar((next >= 32 && next < 127) ? next : (next == '\n' ? '\n' : '.'));
        fflush(stdout);
        buf[len++] = next;
        generated++;
    }
    printf("\n");
    free(buf);
}

static int read_prompt_turn(const char *val_path, int turn_idx,
                             char *prompt_out, int prompt_max) {
    FILE *f = fopen(val_path, "rb");
    if (!f) return -1;
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    char *buf = (char*)malloc(sz + 1);
    fread(buf, 1, sz, f);
    buf[sz] = 0;
    fclose(f);

    /* Walk turns — each is "ISAAC: ... \nCC: ... \n\n" */
    const char *p = buf;
    int idx = 0;
    int found = 0;
    while (*p) {
        if (strncmp(p, "ISAAC:", 6) == 0) {
            if (idx == turn_idx) {
                /* Find end of this ISAAC: line / up to \nCC: */
                const char *cc = strstr(p, "\nCC:");
                if (cc) {
                    int len = (int)(cc - p) + 4;  /* include "\nCC:" */
                    if (len >= prompt_max) len = prompt_max - 1;
                    memcpy(prompt_out, p, len);
                    prompt_out[len] = 0;
                    found = 1;
                }
                break;
            }
            idx++;
        }
        p++;
    }
    free(buf);
    return found ? 0 : -2;
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: infer_duo.exe <ckpt> [--prompt TEXT | --eval FILE] [--max-new N] [--temp T] [--top-p P]\n");
        return 1;
    }

    const char *ckpt = argv[1];
    const char *prompt = NULL;
    const char *eval_file = NULL;
    int max_new = 200;
    float temperature = 0.8f;
    float top_p = 0.9f;
    int n_eval = 5;

    for (int i = 2; i < argc; i++) {
        if (strcmp(argv[i], "--prompt") == 0 && i + 1 < argc) prompt = argv[++i];
        else if (strcmp(argv[i], "--eval") == 0 && i + 1 < argc) eval_file = argv[++i];
        else if (strcmp(argv[i], "--max-new") == 0 && i + 1 < argc) max_new = atoi(argv[++i]);
        else if (strcmp(argv[i], "--temp") == 0 && i + 1 < argc) temperature = (float)atof(argv[++i]);
        else if (strcmp(argv[i], "--top-p") == 0 && i + 1 < argc) top_p = (float)atof(argv[++i]);
        else if (strcmp(argv[i], "--n-eval") == 0 && i + 1 < argc) n_eval = atoi(argv[++i]);
    }

    /* Load model — need to know config first. Probe the checkpoint header. */
    FILE *f = fopen(ckpt, "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", ckpt); return 1; }
    uint32_t magic;
    fread(&magic, 4, 1, f);
    if (magic != 0x54324C34) { fprintf(stderr, "bad magic in %s\n", ckpt); fclose(f); return 1; }
    ModelConfig cfg;
    fread(&cfg, sizeof(ModelConfig), 1, f);
    fclose(f);

    printf("[load] dim=%d layers=%d heads=%d kv=%d inter=%d seq=%d\n",
           cfg.dim, cfg.n_layers, cfg.n_heads, cfg.n_kv_heads, cfg.intermediate, cfg.max_seq);

    TrainableModel tm;
    trainable_model_init(&tm, cfg);
    int rc = trainable_load(&tm, ckpt);
    if (rc != 0) { fprintf(stderr, "load failed: %d\n", rc); return 1; }
    printf("[load] ok, step=%d\n", tm.step);

    srand(42);

    if (prompt) {
        /* Build full prompt: "ISAAC: <text>\nCC:" */
        char full[4096];
        snprintf(full, sizeof(full), "ISAAC: %s\nCC:", prompt);
        printf("\n===== GENERATION =====\n");
        generate(&tm, full, max_new, cfg.max_seq, temperature, top_p);
    } else if (eval_file) {
        printf("\n===== EVAL =====\n");
        printf("File: %s  (first %d turns)\n\n", eval_file, n_eval);
        for (int i = 0; i < n_eval; i++) {
            char turn[4096];
            if (read_prompt_turn(eval_file, i, turn, sizeof(turn)) != 0) {
                printf("(no more turns at idx %d)\n", i);
                break;
            }
            printf("----- turn %d -----\n", i);
            generate(&tm, turn, max_new, cfg.max_seq, temperature, top_p);
            printf("\n");
        }
    } else {
        /* Default: generic prompt */
        const char *default_prompt = "ISAAC: what should we do next?\nCC:";
        printf("\n===== DEFAULT GENERATION =====\n");
        generate(&tm, default_prompt, max_new, cfg.max_seq, temperature, top_p);
    }

    trainable_model_free(&tm);
    return 0;
}
