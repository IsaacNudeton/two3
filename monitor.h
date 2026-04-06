/*
 * monitor.h — Live Training Monitor for {2,3} Architecture
 *
 * Inline alerts. No external process. No file tailing.
 * Call monitor_step() every step from the training loop.
 * It tracks history, detects patterns, and prints when
 * something actually matters.
 *
 * Detects:
 *   - Gradient cascade (exponential growth → kill run early)
 *   - Accuracy plateau (no improvement over N steps)
 *   - Accuracy milestones (trivial baseline, actual learning)
 *   - Flip rate spikes (requantize instability)
 *   - Loss divergence (loss growing over window)
 *
 * Isaac & Claude — April 2026
 */

#ifndef MONITOR_H
#define MONITOR_H

#include <stdio.h>
#include <math.h>
#include <string.h>

/* ═══════════════════════════════════════════════════════
 * Config — tune these per corpus
 * ═══════════════════════════════════════════════════════ */

#define MON_HISTORY       64    /* rolling window size */
#define MON_PLATEAU_STEPS 2000  /* steps without improvement = plateau */
#define MON_CASCADE_RATIO 5.0f  /* grad growth 5× over window = cascade */
#define MON_LOSS_DIV_STEPS 500  /* loss trending up over this many steps */

/* ═══════════════════════════════════════════════════════
 * Milestones — computed from corpus at init
 * ═══════════════════════════════════════════════════════ */

#define MON_MAX_MILESTONES 8

typedef struct {
    float threshold;
    const char *name;
    int fired;
} MonMilestone;

/* ═══════════════════════════════════════════════════════
 * Monitor state
 * ═══════════════════════════════════════════════════════ */

typedef struct {
    /* Rolling history */
    float grad_history[MON_HISTORY];
    float loss_history[MON_HISTORY];
    float acc_history[MON_HISTORY];
    int   flip_history[MON_HISTORY];
    int   hist_idx;
    int   hist_count;

    /* Peak tracking */
    float best_acc;
    int   best_acc_step;
    float best_loss;
    int   best_loss_step;

    /* Plateau detection */
    int   steps_since_improvement;

    /* Cascade detection */
    float grad_baseline;       /* average grad from first MON_HISTORY steps */
    int   baseline_set;

    /* Milestones */
    MonMilestone milestones[MON_MAX_MILESTONES];
    int n_milestones;

    /* Step counter */
    int step;

    /* Alert suppression — don't spam */
    int last_cascade_alert;
    int last_plateau_alert;
    int last_diverge_alert;
} TrainMonitor;

/* ═══════════════════════════════════════════════════════
 * Init — call once before training loop
 *
 * Pass corpus byte frequencies so milestones are data-driven.
 * counts[256] = frequency of each byte in training data.
 * total = total bytes in corpus.
 * ═══════════════════════════════════════════════════════ */

static void monitor_init(TrainMonitor *mon, const unsigned char *corpus, int corpus_size) {
    memset(mon, 0, sizeof(TrainMonitor));
    mon->best_loss = 999.0f;

    /* Compute byte frequencies */
    int counts[256] = {0};
    for (int i = 0; i < corpus_size; i++)
        counts[(int)corpus[i]]++;

    /* Sort descending to find cumulative thresholds */
    int sorted[256];
    for (int i = 0; i < 256; i++) sorted[i] = counts[i];
    for (int i = 0; i < 255; i++)
        for (int j = i + 1; j < 256; j++)
            if (sorted[j] > sorted[i]) {
                int tmp = sorted[i]; sorted[i] = sorted[j]; sorted[j] = tmp;
            }

    /* Milestone 0: most common byte (trivial baseline) */
    float top1 = (float)sorted[0] / (float)corpus_size;
    mon->milestones[0] = (MonMilestone){top1, "TRIVIAL BASELINE (top-1 byte)", 0};

    /* Milestone 1: top-2 bytes (whitespace likely) */
    float top2 = (float)(sorted[0] + sorted[1]) / (float)corpus_size;
    mon->milestones[1] = (MonMilestone){top2, "TOP-2 BYTES", 0};

    /* Milestone 2: top-5 bytes */
    float top5 = 0;
    for (int i = 0; i < 5 && i < 256; i++) top5 += sorted[i];
    top5 /= (float)corpus_size;
    mon->milestones[2] = (MonMilestone){top5, "TOP-5 BYTES (real learning)", 0};

    /* Milestone 3: 50% accuracy */
    mon->milestones[3] = (MonMilestone){0.50f, "50% ACCURACY", 0};

    mon->n_milestones = 4;

    /* Print corpus stats */
    printf("\n");
    printf("  ┌─────────────────────────────────────────────┐\n");
    printf("  │           TRAINING MONITOR ACTIVE            │\n");
    printf("  ├─────────────────────────────────────────────┤\n");
    printf("  │  Corpus: %d bytes                    \n", corpus_size);
    printf("  │  Baselines:                                 │\n");
    for (int i = 0; i < mon->n_milestones; i++)
        printf("  │    %.1f%% — %s\n",
               mon->milestones[i].threshold * 100.0f, mon->milestones[i].name);
    printf("  │  Cascade threshold: %.0f× grad growth      │\n", (double)MON_CASCADE_RATIO);
    printf("  │  Plateau window: %d steps               │\n", MON_PLATEAU_STEPS);
    printf("  └─────────────────────────────────────────────┘\n");
    printf("\n");
    fflush(stdout);
}

/* ═══════════════════════════════════════════════════════
 * Alert printer — timestamped, unmissable
 * ═══════════════════════════════════════════════════════ */

static void monitor_alert(int step, const char *level, const char *msg) {
    printf("\n  !! [MONITOR step %d] %s: %s\n\n", step, level, msg);
    fflush(stdout);
}

/* ═══════════════════════════════════════════════════════
 * Step — call every step with current metrics
 * ═══════════════════════════════════════════════════════ */

static void monitor_step(
    TrainMonitor *mon,
    int step,
    float loss,
    float accuracy,
    float max_grad,
    int flips,
    int total_flips
) {
    char buf[256];
    mon->step = step;

    /* Push to rolling history */
    int idx = mon->hist_idx;
    mon->grad_history[idx] = max_grad;
    mon->loss_history[idx] = loss;
    mon->acc_history[idx] = accuracy;
    mon->flip_history[idx] = flips;
    mon->hist_idx = (idx + 1) % MON_HISTORY;
    if (mon->hist_count < MON_HISTORY) mon->hist_count++;

    /* ── Milestone detection ── */
    for (int i = 0; i < mon->n_milestones; i++) {
        if (!mon->milestones[i].fired && accuracy >= mon->milestones[i].threshold) {
            mon->milestones[i].fired = 1;
            snprintf(buf, sizeof(buf), "acc %.1f%% crossed %.1f%% — %s",
                     accuracy * 100.0f,
                     mon->milestones[i].threshold * 100.0f,
                     mon->milestones[i].name);
            monitor_alert(step, "MILESTONE", buf);
        }
    }

    /* ── Best accuracy tracking + plateau detection ── */
    if (accuracy > mon->best_acc) {
        mon->best_acc = accuracy;
        mon->best_acc_step = step;
        mon->steps_since_improvement = 0;
    } else {
        mon->steps_since_improvement++;
    }

    if (accuracy < mon->best_loss) {
        mon->best_loss = loss;
        mon->best_loss_step = step;
    }

    /* Plateau alert (every MON_PLATEAU_STEPS, don't spam) */
    if (mon->steps_since_improvement >= MON_PLATEAU_STEPS
        && (step - mon->last_plateau_alert) >= MON_PLATEAU_STEPS) {
        mon->last_plateau_alert = step;
        snprintf(buf, sizeof(buf),
                 "no improvement for %d steps (best=%.1f%% at step %d, now=%.1f%%)",
                 mon->steps_since_improvement,
                 mon->best_acc * 100.0f, mon->best_acc_step,
                 accuracy * 100.0f);
        monitor_alert(step, "PLATEAU", buf);
    }

    /* ── Gradient cascade detection ── */
    if (mon->hist_count >= MON_HISTORY) {
        /* Set baseline from first full window */
        if (!mon->baseline_set) {
            float sum = 0;
            for (int i = 0; i < MON_HISTORY; i++) sum += mon->grad_history[i];
            mon->grad_baseline = sum / MON_HISTORY;
            mon->baseline_set = 1;
        }

        /* Current average over recent window */
        float recent_sum = 0;
        int recent_n = MON_HISTORY / 4;  /* last quarter */
        for (int i = 0; i < recent_n; i++) {
            int j = (mon->hist_idx - 1 - i + MON_HISTORY) % MON_HISTORY;
            recent_sum += mon->grad_history[j];
        }
        float recent_avg = recent_sum / recent_n;

        /* Cascade: recent avg is MON_CASCADE_RATIO× baseline */
        if (mon->grad_baseline > 0 && recent_avg > MON_CASCADE_RATIO * mon->grad_baseline
            && (step - mon->last_cascade_alert) >= 500) {
            mon->last_cascade_alert = step;
            snprintf(buf, sizeof(buf),
                     "grad avg %.0f vs baseline %.0f (%.1f×) — possible cascade",
                     recent_avg, mon->grad_baseline,
                     recent_avg / mon->grad_baseline);
            monitor_alert(step, "CASCADE WARNING", buf);
        }
    }

    /* ── Loss divergence ── */
    if (mon->hist_count >= MON_HISTORY) {
        /* Compare first half vs second half of window */
        float first_half = 0, second_half = 0;
        int half = MON_HISTORY / 2;
        for (int i = 0; i < half; i++) {
            int j_old = (mon->hist_idx + i) % MON_HISTORY;
            int j_new = (mon->hist_idx + half + i) % MON_HISTORY;
            first_half += mon->loss_history[j_old];
            second_half += mon->loss_history[j_new];
        }
        first_half /= half;
        second_half /= half;

        if (second_half > first_half * 1.15f
            && (step - mon->last_diverge_alert) >= MON_LOSS_DIV_STEPS) {
            mon->last_diverge_alert = step;
            snprintf(buf, sizeof(buf),
                     "loss trending up: %.3f → %.3f over %d steps",
                     first_half, second_half, MON_HISTORY);
            monitor_alert(step, "LOSS DIVERGING", buf);
        }
    }

    /* ── Flip spike alert ── */
    if (mon->hist_count > 1) {
        /* Alert if single-step flips exceed 2× the running average */
        float flip_sum = 0;
        for (int i = 0; i < mon->hist_count; i++)
            flip_sum += mon->flip_history[i];
        float flip_avg = flip_sum / mon->hist_count;
        if (flip_avg > 0 && flips > 3.0f * flip_avg && flips > 100) {
            snprintf(buf, sizeof(buf),
                     "%d flips this step (avg %.0f) — %d total",
                     flips, flip_avg, total_flips);
            monitor_alert(step, "FLIP SPIKE", buf);
        }
    }
}

/* ═══════════════════════════════════════════════════════
 * Summary — call at end of training
 * ═══════════════════════════════════════════════════════ */

static void monitor_summary(TrainMonitor *mon) {
    printf("\n");
    printf("  ┌─────────────────────────────────────────────┐\n");
    printf("  │           TRAINING MONITOR SUMMARY           │\n");
    printf("  ├─────────────────────────────────────────────┤\n");
    printf("  │  Best accuracy: %.2f%% at step %d\n",
           mon->best_acc * 100.0f, mon->best_acc_step);
    printf("  │  Best loss:     %.4f at step %d\n",
           mon->best_loss, mon->best_loss_step);
    printf("  │  Final grad baseline: %.1f\n", mon->grad_baseline);
    printf("  │  Milestones reached:\n");
    for (int i = 0; i < mon->n_milestones; i++) {
        printf("  │    [%s] %.1f%% — %s\n",
               mon->milestones[i].fired ? "X" : " ",
               mon->milestones[i].threshold * 100.0f,
               mon->milestones[i].name);
    }
    printf("  └─────────────────────────────────────────────┘\n");
    printf("\n");
    fflush(stdout);
}

#endif /* MONITOR_H */
