/*
 * train_driver.cu — Training Loop for {2,3} Architecture
 *
 * This is what runs overnight. Load text → train → checkpoint → log.
 *
 * Usage:
 *   train_driver.exe data.txt                    # train on one file
 *   train_driver.exe data/                       # train on directory
 *   train_driver.exe data.txt --resume ckpt.t2l4 # resume from checkpoint
 *   train_driver.exe data.txt --lr 0.01          # custom learning rate
 *   train_driver.exe data.txt --medium             # small config for testing
 *
 * Outputs:
 *   train_log.txt      — loss, accuracy, ternary flips per step
 *   checkpoint_N.t2l4  — model checkpoint every 100 steps
 *   final.t2l4         — final model
 *
 * Isaac & CC — March 2026
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <cuda_runtime.h>
#include "train.h"
#include "data.h"

/* ═══════════════════════════════════════════════════════
 * Training config
 * ═══════════════════════════════════════════════════════ */

typedef struct {
    /* Model */
    int use_medium;        /* use small config for testing */
    int use_large;         /* use large config (dim=256, 8 layers) */
    int seq_len;          /* bytes per sequence */

    /* Training */
    float lr;             /* learning rate */
    int   epochs;         /* number of passes through data */
    int   batch_size;     /* sequences per optimizer step */
    int   ckpt_every;     /* save checkpoint every N steps */
    int   log_every;      /* log to console every N steps */

    /* Data */
    char  data_path[256]; /* path to text file or directory */
    char  resume[256];    /* checkpoint to resume from, or empty */
} TrainConfig;

static TrainConfig default_config(void) {
    TrainConfig c;
    c.use_medium = 0;
    c.use_large = 0;
    c.seq_len = 128;
#if defined(TWO3_MUON_GPU) || defined(TWO3_USE_MUON_TERNARY)
    c.lr = 1e-3f;       /* Muon + Newton–Schulz: more conservative than SGD 3e-3 */
#else
    c.lr = 3e-3f;
#endif
    c.epochs = 1;
    c.batch_size = 8;
    c.ckpt_every = 100;
    c.log_every = 10;
    c.data_path[0] = 0;
    c.resume[0] = 0;
    return c;
}

static ModelConfig model_config_medium(void) {
    ModelConfig c;
    c.dim = 128;
    c.n_heads = 4;
    c.n_kv_heads = 2;
    c.head_dim = 32;
    c.intermediate = 256;
    c.n_layers = 4;       /* 4 layers — layer-wise gradient clipping prevents explosion */
    c.max_seq = 512;
    c.rope_theta = 1000000.0f;
    return c;
}

static ModelConfig model_config_large(void) {
    ModelConfig c;
    c.dim = 256;
    c.n_heads = 8;
    c.n_kv_heads = 4;
    c.head_dim = 32;
    c.intermediate = 512;
    c.n_layers = 8;
    c.max_seq = 512;
    c.rope_theta = 1000000.0f;
    return c;
}

static void parse_args(TrainConfig *cfg, int argc, char **argv) {
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--medium") == 0) {
            cfg->use_medium = 1;
        } else if (strcmp(argv[i], "--large") == 0) {
            cfg->use_large = 1;
        } else if (strcmp(argv[i], "--lr") == 0 && i + 1 < argc) {
            cfg->lr = (float)atof(argv[++i]);
        } else if (strcmp(argv[i], "--epochs") == 0 && i + 1 < argc) {
            cfg->epochs = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--resume") == 0 && i + 1 < argc) {
            strncpy(cfg->resume, argv[++i], 255);
        } else if (strcmp(argv[i], "--seq-len") == 0 && i + 1 < argc) {
            cfg->seq_len = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--batch") == 0 && i + 1 < argc) {
            cfg->batch_size = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--ckpt-every") == 0 && i + 1 < argc) {
            cfg->ckpt_every = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--log-every") == 0 && i + 1 < argc) {
            cfg->log_every = atoi(argv[++i]);
        } else if (argv[i][0] != '-') {
            strncpy(cfg->data_path, argv[i], 255);
        }
    }
}

/* ═══════════════════════════════════════════════════════
 * Ternary flip counter — track how many weights change
 * ═══════════════════════════════════════════════════════ */

typedef struct {
    int *prev_ternary;   /* previous ternary values as ints */
    int  size;
    int  total_flips;
} FlipCounter;

static void flip_counter_init(FlipCounter *fc, const float *latent, int size) {
    fc->size = size;
    fc->total_flips = 0;
    fc->prev_ternary = (int*)malloc(size * sizeof(int));
    for (int i = 0; i < size; i++) {
        float tq = ternary_quantize(latent[i]);
        fc->prev_ternary[i] = (tq > 0.5f) ? 1 : (tq < -0.5f) ? -1 : 0;
    }
}

static int flip_counter_update(FlipCounter *fc, const float *latent) {
    int flips = 0;
    for (int i = 0; i < fc->size; i++) {
        float tq = ternary_quantize(latent[i]);
        int cur = (tq > 0.5f) ? 1 : (tq < -0.5f) ? -1 : 0;
        if (cur != fc->prev_ternary[i]) {
            flips++;
            fc->prev_ternary[i] = cur;
        }
    }
    fc->total_flips += flips;
    return flips;
}

static void flip_counter_free(FlipCounter *fc) {
    free(fc->prev_ternary);
}

/* ═══════════════════════════════════════════════════════ */

int main(int argc, char **argv) {
    printf("============================================\n");
    printf("  {2,3} Training Driver\n");
    printf("  Bytes in. Loss out. No tokenizer.\n");
    printf("============================================\n\n");

    /* GPU info */
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("  GPU: %s\n\n", prop.name);

    /* Parse config */
    TrainConfig cfg = default_config();
    parse_args(&cfg, argc, argv);

    if (cfg.data_path[0] == 0) {
        printf("Usage: train_driver.exe <data.txt|data_dir/> [options]\n");
        printf("Options:\n");
        printf("  --medium           Use medium model (dim=128, 4 layers)\n");
        printf("  --large            Use large model (dim=256, 8 layers)\n");
#if defined(TWO3_MUON_GPU) || defined(TWO3_USE_MUON_TERNARY)
        printf("  --lr <float>      Learning rate (default: 1e-3 for Muon builds)\n");
#else
        printf("  --lr <float>      Learning rate (default: 3e-3)\n");
#endif
        printf("  --epochs <int>    Number of epochs (default: 1)\n");
        printf("  --seq-len <int>   Sequence length in bytes (default: 128, min 2)\n");
        printf("  --resume <path>   Resume from checkpoint\n");
        printf("  --ckpt-every <N>  Checkpoint every N steps (default: 100)\n");
        printf("  --log-every <N>   Log every N steps (default: 10)\n");
        return 1;
    }

    if (cfg.seq_len < 2) {
        printf("seq_len must be >= 2 (input + next-byte target). Got %d.\n", cfg.seq_len);
        return 1;
    }

    /* Load data */
    Dataset ds;
    dataset_init(&ds, cfg.seq_len);

    int loaded = dataset_load_file(&ds, cfg.data_path);
    if (loaded <= 0) {
        printf("No data loaded. Exiting.\n");
        return 1;
    }
    dataset_info(&ds);
    printf("\n");

    if (ds.n_chunks == 0) {
        printf("No chunks (file too small for seq_len=%d). Exiting.\n", cfg.seq_len);
        return 1;
    }

    /* Init model */
    ModelConfig mcfg = cfg.use_large ? model_config_large()
                     : cfg.use_medium ? model_config_medium()
                     : model_config_default();
    /* Clamp max_seq to training seq_len — no need to allocate scratch for
     * sequences longer than we'll actually train on. */
    mcfg.max_seq = cfg.seq_len;

    printf("  Model: dim=%d, layers=%d, heads=%d, kv=%d, inter=%d\n",
           mcfg.dim, mcfg.n_layers, mcfg.n_heads, mcfg.n_kv_heads, mcfg.intermediate);
    printf("  MoE: %d experts, top-%d\n", MOE_NUM_EXPERTS, MOE_TOP_K);
    printf("  Seq len: %d bytes\n", cfg.seq_len);
    printf("  LR: %.4f\n", cfg.lr);
#if defined(TWO3_MUON_GPU) || defined(TWO3_USE_MUON_TERNARY)
    printf("  (Muon: tighter grad clip in train.h; use --lr 5e-4 or lower if max_h explodes)\n");
#endif
    printf("  Epochs: %d\n", cfg.epochs);
    printf("  Batch size: %d\n", cfg.batch_size);
    printf("  Chunks per epoch: %d\n", ds.n_chunks);
    int steps_per_epoch = (ds.n_chunks + cfg.batch_size - 1) / cfg.batch_size;
    printf("  Steps per epoch: %d\n", steps_per_epoch);
    printf("  Total steps: %d\n\n", cfg.epochs * steps_per_epoch);

    TrainableModel tm;
    trainable_model_init(&tm, mcfg);
    tm.lr = cfg.lr;

    if (cfg.resume[0]) {
        int r = trainable_load(&tm, cfg.resume);
        if (r == 0) {
            printf("  Resumed from: %s (step %d)\n\n", cfg.resume, tm.step);
        } else {
            printf("  Failed to load checkpoint: %s (error %d)\n", cfg.resume, r);
            return 1;
        }
    }

    trainable_requantize(&tm);

    /* Flip counter on W_q of first layer (representative) */
    int D = mcfg.dim;
    FlipCounter fc;
    flip_counter_init(&fc, tm.layer_weights[0].W_q, D * D);

    /* Open log file */
    FILE *logf = fopen("train_log.txt", "a");
    if (logf) {
        fprintf(logf, "# step  loss  accuracy  max_grad  flips  total_flips  time_ms\n");
        fflush(logf);
    }

    /* ═══════════════════════════════════════════════════════
     * TRAINING LOOP
     * ═══════════════════════════════════════════════════════ */

    printf("  Training...\n\n");
    
#ifdef TWO3_DEBUG_MOE
    printf("  [DEBUG] TWO3_DEBUG_MOE is DEFINED\n");
#else
    printf("  [DEBUG] TWO3_DEBUG_MOE is NOT defined\n");
#endif
#ifdef TWO3_DEBUG_EXIT_METRICS
    printf("  [DEBUG] TWO3_DEBUG_EXIT_METRICS is DEFINED (per-layer [exit_probe] in train forward)\n");
#endif

    clock_t t_start = clock();
    int global_step = tm.step;

    for (int epoch = 0; epoch < cfg.epochs; epoch++) {
        dataset_shuffle(&ds, 42 + epoch);

        float epoch_loss = 0;
        int epoch_correct = 0;
        int epoch_total = 0;

        for (int chunk = 0; chunk < ds.n_chunks; chunk += cfg.batch_size) {
            clock_t step_start = clock();

            /* Accumulate gradients over batch */
            trainable_zero_grads(&tm);

            float batch_loss = 0;
            int batch_correct = 0;
            float batch_max_grad = 0;
            int actual_batch = 0;

            for (int b = 0; b < cfg.batch_size && chunk + b < ds.n_chunks; b++) {
                uint8_t *seq = dataset_get(&ds, chunk + b);
                TrainResult r = trainable_forward_backward(&tm, seq, cfg.seq_len);
                batch_loss += r.loss;
                batch_correct += r.correct;
                if (r.max_grad > batch_max_grad) batch_max_grad = r.max_grad;
                actual_batch++;
            }

            /* One optimizer step for the whole batch */
            trainable_optimizer_step(&tm);

            global_step++;
            
            TrainResult r;
            r.loss = batch_loss / actual_batch;
            r.correct = batch_correct;
            r.max_grad = batch_max_grad;
            epoch_loss += r.loss;
            epoch_correct += batch_correct;
            epoch_total += actual_batch * (cfg.seq_len - 1);

            /* Count ternary flips */
            int flips = flip_counter_update(&fc, tm.layer_weights[0].W_q);

            double step_ms = (double)(clock() - step_start) * 1000.0 / CLOCKS_PER_SEC;

            /* Log */
            if (global_step % cfg.log_every == 0 || chunk == 0) {
                int steps_so_far = (chunk / cfg.batch_size) + 1;
                float avg_loss = epoch_loss / steps_so_far;
                float accuracy = (float)epoch_correct / (float)(epoch_total > 0 ? epoch_total : 1);

                printf("  [epoch %d/%d  step %d]  loss=%.4f  acc=%.3f  "
                       "grad=%.4f  flips=%d/%d  %.0fms/step\n",
                       epoch + 1, cfg.epochs, global_step,
                       r.loss, accuracy, r.max_grad,
                       flips, fc.total_flips, step_ms);

                if (logf) {
                    fprintf(logf, "%d  %.6f  %.4f  %.6f  %d  %d  %.1f\n",
                            global_step, r.loss, accuracy, r.max_grad,
                            flips, fc.total_flips, step_ms);
                    fflush(logf);
                }
            }

            /* Checkpoint */
            if (global_step % cfg.ckpt_every == 0) {
                char ckpt_path[64];
                snprintf(ckpt_path, sizeof(ckpt_path), "checkpoint_%d.t2l4", global_step);
                if (trainable_save(&tm, ckpt_path) == 0)
                    printf("  >> checkpoint: %s\n", ckpt_path);
            }

            /* NaN guard */
            if (r.loss != r.loss) {
                printf("\n  !! NaN loss at step %d. Saving emergency checkpoint.\n", global_step);
                trainable_save(&tm, "emergency.t2l4");
                if (logf) fclose(logf);
                flip_counter_free(&fc);
                trainable_model_free(&tm);
                dataset_free(&ds);
                return 1;
            }
        }

        float epoch_avg_loss = epoch_loss / steps_per_epoch;
        float epoch_acc = (float)epoch_correct / (float)(epoch_total > 0 ? epoch_total : 1);
        printf("\n  Epoch %d complete: avg_loss=%.4f  accuracy=%.3f  "
               "total_flips=%d\n\n",
               epoch + 1, epoch_avg_loss, epoch_acc, fc.total_flips);
    }

    /* ═══════════════════════════════════════════════════════
     * DONE
     * ═══════════════════════════════════════════════════════ */

    double total_sec = (double)(clock() - t_start) / CLOCKS_PER_SEC;

    /* Save final model */
    trainable_save(&tm, "final.t2l4");
    printf("  Saved: final.t2l4\n");

#if defined(TWO3_DEBUG_EXIT_METRICS)
    printf("\n  (TWO3_DEBUG_EXIT_METRICS: [exit_probe] in train forward; inference uses model.h)\n");
#endif
#ifdef TWO3_EARLY_EXIT
    printf("  (TWO3_EARLY_EXIT: margin+stable early exit in model_forward_sequence_cpu, seq_len==1 only)\n");
#endif

    printf("\n============================================\n");
    printf("  Training complete.\n");
    printf("  Steps: %d\n", global_step);
    printf("  Total ternary flips (W_q layer 0): %d / %d\n",
           fc.total_flips, D * D);
    printf("  Time: %.1f seconds (%.1f ms/step)\n",
           total_sec, total_sec * 1000.0 / (global_step > 0 ? global_step : 1));
    printf("============================================\n\n");

    if (logf) fclose(logf);
    flip_counter_free(&fc);
    trainable_model_free(&tm);
    dataset_free(&ds);

    return 0;
}
